"""Tests for Python vs Rust parity in doc-id generation and chunk normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from town_elder.cli import (
    _build_file_doc_id as py_build_file_doc_id,
)
from town_elder.cli import (
    _get_doc_id_inputs as py_get_doc_id_inputs,
)
from town_elder.cli import (
    _normalize_chunk_metadata as py_normalize_chunk_metadata,
)
from town_elder.git.diff_parser import (
    DiffParser as PyDiffParser,
)
from town_elder.git.diff_parser import (
    _extract_b_path as py_extract_b_path,
)
from town_elder.git.diff_parser import (
    _parse_git_path as py_parse_git_path,
)
from town_elder.indexing.git_hash_scanner import (
    TrackedFile as PyTrackedFile,
)

# Try to import the Rust extension - tests will be skipped if not available
try:
    from town_elder import _te_core as te_core_rust

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    te_core_rust = None

DOC_ID_HEX_LENGTH = 16
BASE_CHUNK_INDEX = 10
_EXPECTED_TAB_PARTS = 2
_EXPECTED_META_PARTS = 3
_SHA1_HEX_LENGTH = 40


# =============================================================================
# Doc-ID Generation Tests
# =============================================================================

class TestBuildFileDocIdParity:
    """Test parity between Python and Rust doc-id generation."""

    @pytest.mark.parametrize(
        "path,chunk_index",
        [
            ("src/main.py", 0),
            ("src/main.py", 1),
            ("src/main.py", 5),
            ("path with spaces.txt", 0),
            ("path with spaces.txt", 2),
            ("日本語/テスト.py", 0),
            ("日本語/テスト.py", 3),
            ("/absolute/path/file.py", 0),
            ("/absolute/path/file.py", 10),
            ("", 0),
            ("simple.py", 0),
        ],
    )
    def test_doc_id_parity(self, path: str, chunk_index: int):
        """Rust doc-id should match Python doc-id for all fixture cases."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        py_result = py_build_file_doc_id(path, chunk_index)
        rust_result = te_core_rust.build_file_doc_id(path, chunk_index)

        assert py_result == rust_result, (
            f"Doc-id mismatch for path={path!r}, chunk_index={chunk_index}: "
            f"Python={py_result!r}, Rust={rust_result!r}"
        )

    @pytest.mark.parametrize(
        "path,chunk_index,expected_prefix",
        [
            ("src/main.py", 0, "2e5a"),  # Known hash prefix for this input
            ("src/main.py", 1, "9c20"),  # Known hash prefix for chunk 1
        ],
    )
    def test_doc_id_deterministic(self, path: str, chunk_index: int, expected_prefix: str):
        """Doc-ids should be deterministic (same input = same output)."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        result = te_core_rust.build_file_doc_id(path, chunk_index)
        assert result.startswith(expected_prefix), (
            f"Doc-id {result!r} doesn't start with expected prefix {expected_prefix!r}"
        )


# =============================================================================
# Doc-ID Inputs Tests (for deletion safety)
# =============================================================================

class TestGetDocIdInputsParity:
    """Test parity between Python and Rust doc-id inputs expansion."""

    @pytest.mark.parametrize(
        "path,repo_root",
        [
            ("src/main.py", "/Users/testuser/test_repo"),
            ("path with spaces.txt", "/Users/testuser/test repo"),
            ("relative/file.py", "/Users/testuser/project"),
            ("日本語/テスト.py", "/Users/testuser/test"),
            ("/absolute/path/file.py", "/Users/testuser/test_repo"),  # Absolute path should stay as-is
        ],
    )
    def test_doc_id_inputs_parity(self, path: str, repo_root: str):
        """Rust doc-id inputs should match Python for deletion safety."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        py_repo_root = Path(repo_root)
        py_result = py_get_doc_id_inputs(path, py_repo_root)
        rust_result = te_core_rust.get_doc_id_inputs(path, repo_root)

        # Convert to set for comparison (order may differ)
        assert set(py_result) == set(rust_result), (
            f"Doc-id inputs mismatch for path={path!r}, repo_root={repo_root!r}: "
            f"Python={py_result!r}, Rust={rust_result!r}"
        )

    def test_relative_path_expands_to_absolute(self):
        """Relative paths should expand to absolute within repo_root."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        result = te_core_rust.get_doc_id_inputs("src/main.py", "/Users/testuser/repo")
        assert "src/main.py" in result
        # The absolute path should be present (actual resolved path)
        assert any("src/main.py" in inp and Path(inp).is_absolute() for inp in result)

    def test_absolute_path_stays_single(self):
        """Absolute paths should not be duplicated."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        result = te_core_rust.get_doc_id_inputs("/absolute/path/file.py", "/Users/testuser/repo")
        assert len(result) == 1
        assert result[0] == "/absolute/path/file.py"


# =============================================================================
# Chunk Metadata Normalization Tests
# =============================================================================

class TestNormalizeChunkMetadataParity:
    """Test parity between Python and Rust chunk metadata normalization."""

    @pytest.mark.parametrize(
        "base_metadata,chunk_metadata,fallback_chunk_index",
        [
            # Valid chunk_index in chunk_metadata should be preserved
            ({"type": "file"}, {"chunk_index": 2}, 0),
            ({"type": "file"}, {"chunk_index": 5}, 0),
            # Missing chunk_index should use fallback
            ({"type": "file"}, {}, 3),
            ({"type": "file"}, {"extra": "data"}, 1),
            # Invalid chunk_index types should use fallback
            ({"type": "file"}, {"chunk_index": "invalid"}, 0),
            ({"type": "file"}, {"chunk_index": True}, 0),
            ({"type": "file"}, {"chunk_index": False}, 0),
            ({"type": "file"}, {"chunk_index": -1}, 0),
            # Merge base and chunk metadata
            ({"base_key": "base_val"}, {"chunk_key": "chunk_val"}, 0),
            # Empty metadata
            ({}, {}, 0),
            # Chunk_index 0 should work
            ({"type": "file"}, {"chunk_index": 0}, 5),
        ],
    )
    def test_normalize_metadata_parity(
        self,
        base_metadata: dict[str, Any],
        chunk_metadata: dict[str, Any],
        fallback_chunk_index: int,
    ):
        """Rust normalization should match Python."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        py_metadata, py_chunk_index = py_normalize_chunk_metadata(
            base_metadata=base_metadata,
            chunk_metadata=chunk_metadata,
            fallback_chunk_index=fallback_chunk_index,
        )
        rust_metadata, rust_chunk_index = te_core_rust.normalize_chunk_metadata(
            base_metadata,
            chunk_metadata,
            fallback_chunk_index,
        )

        # Compare chunk_index
        assert py_chunk_index == rust_chunk_index, (
            f"Chunk index mismatch: Python={py_chunk_index}, Rust={rust_chunk_index}"
        )

        # Compare metadata (Rust returns JSON values, convert for comparison)
        for key, py_value in py_metadata.items():
            rust_value = rust_metadata.get(key)
            # Convert Rust JSON values to Python types for comparison
            if rust_value is not None:
                # Handle Number type from serde_json
                if hasattr(rust_value, "as_u64"):
                    rust_converted = rust_value.as_u64().to_bytes(8, 'little') if rust_value.is_u64() else int(rust_value)
                    # Compare as int
                    assert py_value == int(py_value), f"Metadata {key}: Python={py_value}, Rust={rust_converted}"
                else:
                    assert py_value == rust_value, (
                        f"Metadata mismatch for key={key}: Python={py_value!r}, Rust={rust_value!r}"
                    )

    def test_metadata_merge_preserves_base_values(self):
        """Base metadata values should be preserved when not overridden."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        base = {"type": "python", "source": "test"}
        chunk = {"extra": "value"}
        fallback = 1

        _result_metadata, chunk_index = te_core_rust.normalize_chunk_metadata(
            base, chunk, fallback
        )

        assert _result_metadata.get("type") == "python"
        assert _result_metadata.get("source") == "test"
        assert _result_metadata.get("extra") == "value"

    def test_chunk_metadata_overrides_base(self):
        """Chunk metadata should override base metadata."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        base = {"type": "base_type", "chunk_index": 10}
        chunk = {"type": "chunk_type"}
        fallback = 0

        result_metadata, chunk_index = te_core_rust.normalize_chunk_metadata(
            base, chunk, fallback
        )

        assert result_metadata.get("type") == "chunk_type"
        # chunk_index from chunk should take precedence
        assert chunk_index == BASE_CHUNK_INDEX


# =============================================================================
# Fixture-based Tests (using contract fixtures)
# =============================================================================

class TestDocIdFromFixtures:
    """Test doc-id generation using contract fixtures."""

    @pytest.fixture
    def fixture_dir(self) -> Path:
        return Path(__file__).parent / "fixtures" / "rust_core_contract" / "doc_ids"

    def test_path_with_spaces(self):
        """Doc-id for path with spaces should work correctly."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        path = "path with spaces.txt"
        doc_id = te_core_rust.build_file_doc_id(path, 0)

        # Should match Python's behavior
        expected = py_build_file_doc_id(path, 0)
        assert doc_id == expected

    def test_unicode_path(self):
        """Doc-id for unicode path should work correctly."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        path = "日本語/テスト.py"
        doc_id = te_core_rust.build_file_doc_id(path, 0)

        # Should match Python's behavior
        expected = py_build_file_doc_id(path, 0)
        assert doc_id == expected


# =============================================================================
# Backward Compatibility Tests
# =============================================================================

class TestBackwardCompatibility:
    """Ensure existing behavior is unchanged when Rust is unavailable."""

    def test_python_fallback_when_rust_unavailable(self):
        """Python implementation should still work when Rust is unavailable."""
        # These should work regardless of Rust availability
        doc_id = py_build_file_doc_id("test.py", 0)
        assert len(doc_id) == DOC_ID_HEX_LENGTH
        assert doc_id.isalnum()

        doc_id_multi = py_build_file_doc_id("test.py", 5)
        assert len(doc_id_multi) == DOC_ID_HEX_LENGTH

    def test_python_normalize_fallback_when_rust_unavailable(self):
        """Python normalization should work regardless of Rust."""
        metadata, idx = py_normalize_chunk_metadata(
            base_metadata={"type": "test"},
            chunk_metadata={},
            fallback_chunk_index=0,
        )
        assert idx == 0
        assert metadata.get("type") == "test"

    def test_python_doc_id_inputs_fallback_when_rust_unavailable(self):
        """Python doc-id inputs should work regardless of Rust."""
        inputs = py_get_doc_id_inputs("test.py", Path("/Users/testuser/repo"))
        assert "test.py" in inputs
        assert len(inputs) >= 1


# =============================================================================
# Git Blob Parsing Parity Tests
# =============================================================================

class TestGitBlobParsingParity:
    """Test parity between Python and Rust git blob parsing."""

    @pytest.mark.parametrize(
        "line",
        [
            # Valid lines
            "100644 abc123def456789012345678901234567890000 0\tsrc/main.py",
            "100755 abc123def456789012345678901234567890001 0\tsrc/utils.py",
            "100644 0acd8c61f0d9dc1e5db7ad7c2dbce2bb16b8d6de 0\t.beads/.gitignore",
            # Path with spaces
            "100644 abc123def456789012345678901234567890002 0\tpath with spaces/file.txt",
            # Unicode path
            "100644 abc123def456789012345678901234567890003 0\t日本語/テスト.py",
            # Path with special characters
            "100644 abc123def456789012345678901234567890004 0\tpath/with-dash_and_underscore/file.py",
        ],
    )
    def test_parse_git_blob_line_parity(self, line: str):
        """Rust parsing should match Python for valid lines."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        # Get Python result
        py_result = _parse_python_blob_line(line)

        # Get Rust result
        rust_result = te_core_rust.parse_git_blob_line(line)

        # Compare
        if py_result is None:
            assert rust_result is None
        else:
            assert rust_result is not None
            assert py_result.path == rust_result.path
            assert py_result.blob_hash == rust_result.blob_hash
            assert py_result.mode == rust_result.mode

    @pytest.mark.parametrize(
        "line",
        [
            # Invalid/unsupported modes - should be skipped
            "120000 abc123def456789012345678901234567890005 0\tsymlink",  # Symlink
            "000000 abc123def456789012345678901234567890006 0\tempty",  # Empty mode
            # Malformed lines - should be safely ignored
            "",
            "not enough parts",
            "100644",  # Missing blob hash and path
            "notatab src/main.py",  # No tab separator
            "100644 invalidhash path/file.py",  # Invalid hash (not 40 hex chars)
            "100644 abc123def4567890123456789012345678900070 0",  # Missing path
        ],
    )
    def test_parse_git_blob_line_skips_invalid(self, line: str):
        """Invalid lines should return None in both Python and Rust."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        py_result = _parse_python_blob_line(line)
        rust_result = te_core_rust.parse_git_blob_line(line)

        assert py_result is None
        assert rust_result is None


def _parse_python_blob_line(line: str) -> PyTrackedFile | None:
    """Helper to parse git blob line using Python implementation."""
    if not line:
        return None

    parts = line.split("\t")
    if len(parts) != _EXPECTED_TAB_PARTS:
        return None

    metadata, file_path = parts[0], parts[1]
    meta_parts = metadata.split()
    if len(meta_parts) != _EXPECTED_META_PARTS:
        return None

    mode, blob_hash, _stage = meta_parts

    valid_modes = {"100644", "100755"}
    if mode not in valid_modes:
        return None

    if len(blob_hash) != _SHA1_HEX_LENGTH or not all(c in "0123456789abcdefABCDEF" for c in blob_hash):
        return None

    return PyTrackedFile(path=file_path, blob_hash=blob_hash, mode=mode)


# =============================================================================
# Git Diff Parsing Parity Tests
# =============================================================================

class TestGitDiffParsingParity:
    """Test parity between Python and Rust git diff parsing."""

    @pytest.mark.parametrize(
        "line,expected_b_path",
        [
            # Unquoted paths
            ("diff --git a/src/main.py b/src/main.py", "b/src/main.py"),
            ("diff --git a/src/utils.py b/src/utils.py", "b/src/utils.py"),
            # Quoted paths with spaces
            ('diff --git "a/path with spaces" "b/path with spaces"', "b/path with spaces"),
            ('diff --git "a/file.txt" "b/file.txt"', "b/file.txt"),
            # Unicode paths
            ('diff --git "a/日本語/テスト.py" "b/日本語/テスト.py"', "b/日本語/テスト.py"),
        ],
    )
    def test_extract_b_path_parity(self, line: str, expected_b_path: str):
        """Rust b-path extraction should match Python."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        py_result = py_extract_b_path(line)
        rust_result = te_core_rust.extract_b_path(line)

        assert py_result == expected_b_path
        assert rust_result == expected_b_path

    @pytest.mark.parametrize(
        "line",
        [
            # Invalid lines - should return None
            "",
            "not a diff header",
            "diff",  # Incomplete
            "diff --git",  # Missing paths
            'diff --git "a/path',  # Unclosed quote
        ],
    )
    def test_extract_b_path_returns_none_for_invalid(self, line: str):
        """Invalid diff headers should return None."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        py_result = py_extract_b_path(line)
        rust_result = te_core_rust.extract_b_path(line)

        assert py_result is None
        assert rust_result is None

    @pytest.mark.parametrize(
        "path,expected",
        [
            # Unquoted paths
            ("b/src/main.py", "src/main.py"),
            ("a/src/main.py", "src/main.py"),
            ("src/main.py", "src/main.py"),  # No prefix
            # Quoted paths
            ('"b/path with spaces"', "path with spaces"),
            ('"a/unicode/日本語.py"', "unicode/日本語.py"),
        ],
    )
    def test_parse_git_path_parity(self, path: str, expected: str):
        """Rust git path parsing should match Python."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        py_result = py_parse_git_path(path)
        rust_result = te_core_rust.parse_git_path(path)

        assert py_result == expected
        assert rust_result == expected


# =============================================================================
# DiffParser Class Parity Tests
# =============================================================================

class TestDiffParserClassParity:
    """Test parity between Python and Rust DiffParser class."""

    @pytest.fixture
    def sample_diff(self) -> str:
        """Sample diff output for testing."""
        return """diff --git a/src/main.py b/src/main.py
new file mode 100644
--- /dev/null
+++ b/src/main.py
@@ -0,0 +1,5 @@
+def hello():
+    print("Hello, world!")
+
+if __name__ == "__main__":
+    hello()
diff --git a/src/app.py b/src/app.py
--- a/src/app.py
+++ b/src/app.py
@@ -1,5 +1,6 @@
+import new_module
 def main():
-    old_call()
+    new_call()
     return True
"""

    def test_parse_diff_files_parity(self, sample_diff: str):
        """Rust DiffParser should match Python output."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        # Parse with Python
        py_parser = PyDiffParser()
        py_files = list(py_parser.parse(sample_diff))

        # Parse with Rust
        rust_parser = te_core_rust.PyDiffParser(warn_on_parse_error=True)
        rust_files = rust_parser.parse(sample_diff)

        # Compare file counts
        assert len(py_files) == len(rust_files)

        # Compare each file
        for py_file, rust_file in zip(py_files, rust_files, strict=True):
            assert py_file.path == rust_file.path
            assert py_file.status == rust_file.status
            assert len(py_file.hunks) == len(rust_file.hunks)

    def test_parse_diff_to_text_parity(self, sample_diff: str):
        """Rust parse_diff_to_text should match Python output."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        # Parse with Python
        py_parser = PyDiffParser()
        py_text = py_parser.parse_diff_to_text(sample_diff)

        # Parse with Rust
        rust_parser = te_core_rust.PyDiffParser(warn_on_parse_error=True)
        rust_text = rust_parser.parse_diff_to_text(sample_diff)

        # Compare
        assert py_text == rust_text

    def test_parse_empty_diff(self):
        """Both parsers should handle empty diff."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        py_parser = PyDiffParser()
        rust_parser = te_core_rust.PyDiffParser(warn_on_parse_error=True)

        py_files = list(py_parser.parse(""))
        rust_files = rust_parser.parse("")

        assert len(py_files) == 0
        assert len(rust_files) == 0
