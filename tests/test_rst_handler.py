"""Tests for RST parser module."""

from __future__ import annotations

import tempfile

from town_elder.parsers.rst_handler import (
    RSTChunk,
    _check_temporal_tags,
    _chunk_by_sections,
    _extract_directives_content,
    _find_section_boundaries,
    _get_heading_level,
    get_chunk_metadata,
    parse_rst_content,
    parse_rst_file,
)

# Test constants for heading levels
_LEVEL_1 = 1
_LEVEL_2 = 2
_LEVEL_3 = 3
_LEVEL_4 = 4
_LEVEL_5 = 5


class TestHeadingLevels:
    """Tests for heading level detection."""

    def test_heading_level_equal_sign(self):
        """Test = maps to level 1."""
        assert _get_heading_level("=") == 1
        assert _get_heading_level("===") == 1

    def test_heading_level_hyphen(self):
        """Test - maps to level 2."""
        assert _get_heading_level("-") == _LEVEL_2
        assert _get_heading_level("---") == _LEVEL_2

    def test_heading_level_tilde(self):
        """Test ~ maps to level 3."""
        assert _get_heading_level("~") == _LEVEL_3
        assert _get_heading_level("~~~") == _LEVEL_3

    def test_heading_level_caret(self):
        """Test ^ maps to level 4."""
        assert _get_heading_level("^") == _LEVEL_4
        assert _get_heading_level("^^^") == _LEVEL_4

    def test_heading_level_plus(self):
        """Test + maps to level 5."""
        assert _get_heading_level("+") == _LEVEL_5
        assert _get_heading_level("+++") == _LEVEL_5


class TestSectionBoundaries:
    """Tests for section boundary detection."""

    def test_find_single_section(self):
        """Test finding a single section."""
        content = """Title
=====

Some content here."""
        boundaries = _find_section_boundaries(content)
        assert len(boundaries) == 1
        assert boundaries[0][1] == "Title"

    def test_find_multiple_sections(self):
        """Test finding multiple sections."""
        content = """Title
=====

Section One
-----------

Content one.

Section Two
-----------

Content two."""
        boundaries = _find_section_boundaries(content)
        assert len(boundaries) == _LEVEL_2 + _LEVEL_1  # Title + 2 sections

    def test_no_sections(self):
        """Test with no section headings."""
        content = """Just some plain text.
No headings here."""
        boundaries = _find_section_boundaries(content)
        assert len(boundaries) == 0

    def test_mixed_underline_characters_not_treated_as_heading(self):
        """Underline must be one repeated heading character."""
        content = """Title
=-=

Body text."""
        boundaries = _find_section_boundaries(content)
        assert boundaries == []


class TestChunkBySections:
    """Tests for section-based chunking."""

    def test_chunk_single_section(self):
        """Test chunking a document with single section."""
        content = """Title
=====

This is the content."""
        chunks = _chunk_by_sections(content)
        assert len(chunks) == 1
        start, end, path = chunks[0]
        assert path == ["Title"]

    def test_chunk_multiple_sections(self):
        """Test chunking with multiple sections."""
        content = """Main Title
============

Section A
---------

Content A.

Section B
---------

Content B."""
        chunks = _chunk_by_sections(content)
        # Should have 2 content chunks (after each section heading)
        assert len(chunks) >= 1

    def test_chunk_nested_sections(self):
        """Test chunking with nested sections."""
        content = """Top Level
==========

Section 1
---------

Subsection 1.1
~~~~~~~~~~~~~~

Content 1.1."""
        chunks = _chunk_by_sections(content)
        assert len(chunks) >= 1


class TestDirectiveExtraction:
    """Tests for directive content extraction."""

    def test_extract_note_directive(self):
        """Test extracting note directive."""
        text = """Section
=======

Some text.

.. note::

   This is a note."""
        directives = _extract_directives_content(text)
        assert "note" in directives
        assert any("This is a note" in note for note in directives["note"])

    def test_extract_warning_directive(self):
        """Test extracting warning directive."""
        text = """Section
=======

.. warning::

   This is a warning."""
        directives = _extract_directives_content(text)
        assert "warning" in directives
        assert any("This is a warning" in warn for warn in directives["warning"])

    def test_extract_versionadded_directive(self):
        """Test extracting versionadded directive."""
        text = """Section
=======

.. versionadded:: 2.0

   Added in version 2.0."""
        directives = _extract_directives_content(text)
        assert "versionadded" in directives
        assert any("Added in version 2.0" in va for va in directives["versionadded"])

    def test_extract_multiple_directives(self):
        """Test extracting multiple directives."""
        text = """Section
=======

.. note::

   A note.

.. warning::

   A warning."""
        directives = _extract_directives_content(text)
        assert "note" in directives
        assert "warning" in directives


class TestTemporalTags:
    """Tests for temporal tag detection."""

    def test_deprecated_tag(self):
        """Test detecting deprecated directive."""
        text = """Section
=======

.. deprecated:: 1.5

   Use new_function instead."""
        tags = _check_temporal_tags(text)
        assert "deprecated" in tags

    def test_versionchanged_tag(self):
        """Test detecting versionchanged directive."""
        text = """Section
=======

.. versionchanged:: 2.0

   Changed in version 2.0."""
        tags = _check_temporal_tags(text)
        assert "versionchanged" in tags

    def test_no_temporal_tags(self):
        """Test with no temporal directives."""
        text = """Section
=======

Just some content."""
        tags = _check_temporal_tags(text)
        assert len(tags) == 0


class TestParseRstContent:
    """Tests for parse_rst_content function."""

    def test_parse_simple_rst(self):
        """Test parsing simple RST content."""
        content = """Title
=====

This is content."""
        chunks = parse_rst_content(content)
        assert len(chunks) > 0
        assert chunks[0].text

    def test_parse_rst_with_section_hierarchy(self):
        """Test parsing RST with section hierarchy."""
        content = """Main Title
============

Section One
-----------

Content one."""
        chunks = parse_rst_content(content)
        assert len(chunks) > 0
        # First chunk should have section path
        assert len(chunks[0].section_path) >= 1

    def test_parse_rst_section_chunks_do_not_bleed_next_heading(self):
        """Each section chunk should stop before the next section heading."""
        content = """Main Title
============

Section A
---------

Content A.

Section B
---------

Content B."""
        chunks = parse_rst_content(content)

        section_a_chunks = [
            chunk.text
            for chunk in chunks
            if "Section A" in chunk.text and "Content A." in chunk.text
        ]
        assert len(section_a_chunks) == 1
        assert "Section B" not in section_a_chunks[0]

    def test_parse_rst_extracts_directives(self):
        """Test that parsing extracts directives."""
        content = """Title
=======

.. note::

   Important info."""
        chunks = parse_rst_content(content)
        assert len(chunks) > 0
        if chunks[0].directives:
            assert "note" in chunks[0].directives

    def test_parse_rst_temporal_tags(self):
        """Test that parsing extracts temporal tags."""
        content = """Title
=======

.. deprecated:: 1.0

   Use something else."""
        chunks = parse_rst_content(content)
        assert len(chunks) > 0
        assert "deprecated" in chunks[0].temporal_tags

    def test_parse_empty_content(self):
        """Test parsing empty content."""
        chunks = parse_rst_content("")
        assert len(chunks) == 0

    def test_parse_whitespace_only(self):
        """Test parsing whitespace-only content."""
        chunks = parse_rst_content("   \n\n   ")
        assert len(chunks) == 0


class TestParseRstFile:
    """Tests for parse_rst_file function."""

    def test_parse_existing_file(self):
        """Test parsing an existing RST file."""
        content = """Title
=======

Content here."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".rst", delete=False) as f:
            f.write(content)
            f.flush()
            chunks = parse_rst_file(f.name)
            assert len(chunks) > 0

    def test_parse_nonexistent_file(self):
        """Test parsing nonexistent file returns empty list."""
        chunks = parse_rst_file("/nonexistent/file.rst")
        assert len(chunks) == 0


class TestMalformedInput:
    """Tests for malformed RST input handling."""

    def test_malformed_rst_does_not_crash(self):
        """Test that malformed RST doesn't crash the parser."""
        # Invalid/malformed RST
        malformed = """<<<<<<< HEAD
Invalid syntax
=======
More invalid
>>>>>>> branch"""
        # Should not raise, just return chunks or empty
        chunks = parse_rst_content(malformed)
        # Should not crash - may return empty or partial results
        assert isinstance(chunks, list)

    def test_truncated_underline(self):
        """Test handling of truncated underline."""
        content = """Title
==

Content with short underline."""
        chunks = parse_rst_content(content)
        # Should handle gracefully
        assert isinstance(chunks, list)


class TestChunkMetadata:
    """Tests for chunk metadata extraction."""

    def test_basic_metadata(self):
        """Test basic metadata extraction."""
        chunk = RSTChunk(
            text="Test content",
            section_path=["Section"],
            directives={},
            temporal_tags=[],
            chunk_index=0,
        )
        metadata = get_chunk_metadata(chunk)
        assert metadata["chunk_index"] == 0
        assert metadata["section_path"] == ["Section"]
        assert metadata["has_directives"] is False
        assert metadata["has_temporal_tags"] is False

    def test_metadata_with_directives(self):
        """Test metadata with directives."""
        chunk = RSTChunk(
            text="Test",
            section_path=[],
            directives={"note": ["A note"]},
            temporal_tags=[],
            chunk_index=0,
        )
        metadata = get_chunk_metadata(chunk)
        assert metadata["has_directives"] is True
        assert "directives" in metadata

    def test_metadata_with_temporal_tags(self):
        """Test metadata with temporal tags."""
        chunk = RSTChunk(
            text="Test",
            section_path=[],
            directives={},
            temporal_tags=["deprecated"],
            chunk_index=0,
        )
        metadata = get_chunk_metadata(chunk)
        assert metadata["has_temporal_tags"] is True
        assert "temporal_tags" in metadata

    def test_metadata_section_depth(self):
        """Test section depth in metadata."""
        chunk = RSTChunk(
            text="Test",
            section_path=["Main", "Sub", "SubSub"],
            directives={},
            temporal_tags=[],
            chunk_index=0,
        )
        metadata = get_chunk_metadata(chunk)
        assert metadata["section_depth"] == _LEVEL_3
