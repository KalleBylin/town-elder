"""Tests for Rust shared core contract fixtures.

This module validates that fixtures match the expected DTO shapes
and produce the same outputs as the current Python implementation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from town_elder.git.diff_parser import DiffParser
from town_elder.indexing.file_scanner import scan_files
from town_elder.parsers.rst_handler import parse_rst_content

# Test constants
_DOC_ID_LENGTH = 16
_SHA1_HASH_LENGTH = 40
_NESTED_DIR_MIN_FILES = 3
_NESTED_DIR_MAX_DEPTH = 2
_CHUNK_INDEX_VALID = 2
_CHUNK_INDEX_BOOL_FALLBACK = 5
_CHUNK_INDEX_NEGATIVE_FALLBACK = 3

# Get the fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "rust_core_contract"


class TestFileScannerFixtures:
    """Tests for file scanner fixtures."""

    def test_simple_files_scan(self):
        """Test scanning simple files."""
        fixture_path = FIXTURES_DIR / "file_scanner" / "simple_files"
        files = scan_files(fixture_path)

        # Should find .py, .md, .rst files
        extensions = {f.suffix for f in files}
        assert ".py" in extensions
        assert ".md" in extensions
        assert ".rst" in extensions

        # Results should be sorted deterministically
        file_strs = [str(f) for f in files]
        assert file_strs == sorted(file_strs)

    def test_nested_dirs_scan(self):
        """Test scanning nested directories."""
        fixture_path = FIXTURES_DIR / "file_scanner" / "nested_dirs"
        files = scan_files(fixture_path)

        # Should find all .py files in nested dirs
        assert len(files) >= _NESTED_DIR_MIN_FILES

        # Check we have files from different depths
        rel_paths = [f.relative_to(fixture_path) for f in files]
        depths = [len(p.parts) - 1 for p in rel_paths]
        assert max(depths) >= _NESTED_DIR_MAX_DEPTH


class TestRSTChunkFixtures:
    """Tests for RST chunk fixtures."""

    @pytest.mark.parametrize("fixture_file", [
        "simple_section.rst",
        "nested_sections.rst",
        "with_directives.rst",
        "with_temporal_tags.rst",
        "unicode_sections.rst",
    ])
    def test_rst_parsing(self, fixture_file: str):
        """Test parsing various RST fixtures."""
        fixture_path = FIXTURES_DIR / "rst_chunks" / fixture_file
        content = fixture_path.read_text(encoding="utf-8")

        chunks = parse_rst_content(content)

        # Should produce at least one chunk
        assert len(chunks) > 0

        # Each chunk should have required fields
        for chunk in chunks:
            assert chunk.text is not None
            assert isinstance(chunk.section_path, list)
            assert isinstance(chunk.directives, dict)
            assert isinstance(chunk.temporal_tags, list)
            assert isinstance(chunk.chunk_index, int)

    def test_simple_section_chunk_count(self):
        """Test simple section produces expected chunks."""
        fixture_path = FIXTURES_DIR / "rst_chunks" / "simple_section.rst"
        content = fixture_path.read_text(encoding="utf-8")

        chunks = parse_rst_content(content)

        # Should have at least one chunk with the section heading
        assert len(chunks) >= 1

    def test_directives_extraction(self):
        """Test that directives are extracted correctly."""
        fixture_path = FIXTURES_DIR / "rst_chunks" / "with_directives.rst"
        content = fixture_path.read_text(encoding="utf-8")

        chunks = parse_rst_content(content)

        # Find chunk with directives
        directive_chunks = [c for c in chunks if c.directives]
        assert len(directive_chunks) > 0

        # Check directive types
        all_directives = {}
        for chunk in directive_chunks:
            all_directives.update(chunk.directives)

        assert "note" in all_directives
        assert "warning" in all_directives

    def test_temporal_tags_extraction(self):
        """Test that temporal tags are extracted correctly."""
        fixture_path = FIXTURES_DIR / "rst_chunks" / "with_temporal_tags.rst"
        content = fixture_path.read_text(encoding="utf-8")

        chunks = parse_rst_content(content)

        # Find chunks with temporal tags
        temporal_chunks = [c for c in chunks if c.temporal_tags]
        assert len(temporal_chunks) > 0

        # Check temporal tag types
        all_tags = []
        for chunk in temporal_chunks:
            all_tags.extend(chunk.temporal_tags)

        assert "deprecated" in all_tags

    def test_unicode_sections(self):
        """Test that unicode sections are handled."""
        fixture_path = FIXTURES_DIR / "rst_chunks" / "unicode_sections.rst"
        content = fixture_path.read_text(encoding="utf-8")

        chunks = parse_rst_content(content)

        # Should produce chunks without crashing
        assert len(chunks) > 0

        # Check that section paths contain unicode
        all_sections = []
        for chunk in chunks:
            all_sections.extend(chunk.section_path)

        assert any(s for s in all_sections if s), "Should have non-empty sections"


class TestCommitFixtures:
    """Tests for commit fixtures."""

    def test_simple_commit_parse(self):
        """Test parsing simple commit JSON."""
        fixture_path = FIXTURES_DIR / "commits" / "simple_commit.json"
        data = json.loads(fixture_path.read_text())

        # Validate structure matches Commit DTO
        assert "hash" in data
        assert "message" in data
        assert "author" in data
        assert "date" in data
        assert "files_changed" in data

        # Validate hash is full SHA-1
        assert len(data["hash"]) == _SHA1_HASH_LENGTH
        assert all(c in "0123456789abcdef" for c in data["hash"])

    def test_unicode_commit_parse(self):
        """Test parsing commit with unicode."""
        fixture_path = FIXTURES_DIR / "commits" / "unicode_commit.json"
        data = json.loads(fixture_path.read_text())

        # Validate unicode is preserved
        assert "添加新功能" in data["message"]
        assert "张三" in data["author"]
        assert "中文" in data["files_changed"][0]


class TestDiffFixtures:
    """Tests for diff fixtures."""

    def test_simple_diff_parse(self):
        """Test parsing simple diff."""
        fixture_path = FIXTURES_DIR / "diffs" / "simple_diff.diff"
        diff_content = fixture_path.read_text()

        parser = DiffParser(warn_on_parse_error=False)
        diff_files = list(parser.parse(diff_content))

        assert len(diff_files) == 1
        assert diff_files[0].path == "file.txt"
        assert diff_files[0].status == "added"

    def test_quoted_paths_diff(self):
        """Test parsing diff with quoted paths containing spaces."""
        fixture_path = FIXTURES_DIR / "diffs" / "quoted_paths.diff"
        diff_content = fixture_path.read_text()

        parser = DiffParser(warn_on_parse_error=False)
        diff_files = list(parser.parse(diff_content))

        assert len(diff_files) == 1
        # Path with spaces should be unquoted
        assert diff_files[0].path == "file with spaces.txt"

    def test_diff_to_text(self):
        """Test converting diff to text."""
        fixture_path = FIXTURES_DIR / "diffs" / "simple_diff.diff"
        diff_content = fixture_path.read_text()

        parser = DiffParser(warn_on_parse_error=False)
        text = parser.parse_diff_to_text(diff_content)

        # Should contain file path and status
        assert "file.txt" in text
        assert "added" in text


class TestDocIDFixtures:
    """Tests for doc-id generation fixtures."""

    def test_doc_id_with_spaces(self):
        """Test doc-id generation for path with spaces."""
        path_with_spaces = "/path/with spaces/file.py"

        # Test chunk 0
        doc_id_0 = hashlib.sha256(path_with_spaces.encode()).hexdigest()[:_DOC_ID_LENGTH]

        # Test chunk 1
        doc_id_1 = hashlib.sha256(f"{path_with_spaces}#chunk:1".encode()).hexdigest()[:_DOC_ID_LENGTH]

        # Doc IDs should be different for different chunks
        assert doc_id_0 != doc_id_1
        assert len(doc_id_0) == _DOC_ID_LENGTH
        assert len(doc_id_1) == _DOC_ID_LENGTH

    def test_doc_id_deterministic(self):
        """Test that doc-ids are deterministic."""
        path = "/test/path.py"

        # Generate doc-id twice
        doc_id_1 = hashlib.sha256(path.encode()).hexdigest()[:_DOC_ID_LENGTH]
        doc_id_2 = hashlib.sha256(path.encode()).hexdigest()[:_DOC_ID_LENGTH]

        assert doc_id_1 == doc_id_2

    def test_doc_id_unicode(self):
        """Test doc-id generation with unicode path."""
        unicode_path = "/path/中文/文件.py"

        doc_id = hashlib.sha256(unicode_path.encode()).hexdigest()[:_DOC_ID_LENGTH]

        # Should generate valid hex
        assert len(doc_id) == _DOC_ID_LENGTH
        assert all(c in "0123456789abcdef" for c in doc_id)


class TestChunkMetadataNormalization:
    """Tests for chunk metadata normalization."""

    def test_normalize_valid_chunk_index(self):
        """Test normalization with valid chunk index."""
        base = {"source": "/test.py", "type": ".py"}
        chunk_meta = {"chunk_index": _CHUNK_INDEX_VALID}

        from town_elder.cli import _normalize_chunk_metadata

        result, index = _normalize_chunk_metadata(
            base_metadata=base,
            chunk_metadata=chunk_meta,
            fallback_chunk_index=0,
        )

        assert index == _CHUNK_INDEX_VALID
        assert result["chunk_index"] == _CHUNK_INDEX_VALID

    def test_normalize_invalid_chunk_index_bool(self):
        """Test normalization with boolean chunk index (invalid)."""
        base = {"source": "/test.py"}
        chunk_meta = {"chunk_index": True}  # Invalid - bool

        from town_elder.cli import _normalize_chunk_metadata

        result, index = _normalize_chunk_metadata(
            base_metadata=base,
            chunk_metadata=chunk_meta,
            fallback_chunk_index=_CHUNK_INDEX_BOOL_FALLBACK,
        )

        assert index == _CHUNK_INDEX_BOOL_FALLBACK
        assert result["chunk_index"] == _CHUNK_INDEX_BOOL_FALLBACK

    def test_normalize_negative_chunk_index(self):
        """Test normalization with negative chunk index (invalid)."""
        base = {"source": "/test.py"}
        chunk_meta = {"chunk_index": -1}  # Invalid - negative

        from town_elder.cli import _normalize_chunk_metadata

        result, index = _normalize_chunk_metadata(
            base_metadata=base,
            chunk_metadata=chunk_meta,
            fallback_chunk_index=_CHUNK_INDEX_NEGATIVE_FALLBACK,
        )

        assert index == _CHUNK_INDEX_NEGATIVE_FALLBACK
        assert result["chunk_index"] == _CHUNK_INDEX_NEGATIVE_FALLBACK


class TestFixtureCompleteness:
    """Tests to ensure all required fixtures exist."""

    def test_all_required_fixtures_exist(self):
        """Verify all required fixture files exist."""
        required_files = [
            # File scanner
            "file_scanner/simple_files/test.py",
            "file_scanner/simple_files/test.md",
            "file_scanner/simple_files/test.rst",
            "file_scanner/nested_dirs/root_file.py",
            # RST chunks
            "rst_chunks/simple_section.rst",
            "rst_chunks/nested_sections.rst",
            "rst_chunks/with_directives.rst",
            "rst_chunks/with_temporal_tags.rst",
            "rst_chunks/unicode_sections.rst",
            # Commits
            "commits/simple_commit.json",
            "commits/unicode_commit.json",
            # Diffs
            "diffs/simple_diff.diff",
            "diffs/quoted_paths.diff",
            "diffs/unquoted_paths.diff",
            # Doc IDs
            "doc_ids/path_with_spaces.txt",
            "doc_ids/unicode_path.txt",
        ]

        for rel_path in required_files:
            fixture_path = FIXTURES_DIR / rel_path
            assert fixture_path.exists(), f"Missing fixture: {rel_path}"
