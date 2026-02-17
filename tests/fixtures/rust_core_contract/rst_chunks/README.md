# RST Chunk Fixtures

This directory contains RST files for testing RST parsing behavior.

## Files

- `simple_section.rst` - Single section heading
- `nested_sections.rst` - Multiple levels of headings
- `with_directives.rst` - note, warning, versionadded directives
- `with_temporal_tags.rst` - deprecated, versionchanged tags
- `unicode_sections.rst` - Unicode in headings (Chinese, German)

## Purpose

These fixtures validate that:
1. Section boundary detection works correctly
2. Directives (note, warning, versionadded) are extracted
3. Temporal tags (deprecated, versionchanged) are detected
4. Unicode characters in headings are handled properly
