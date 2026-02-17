# Doc-ID Fixtures

This directory contains test inputs for doc-id generation testing.

## Files

- `path_with_spaces.txt` - Path containing spaces
- `unicode_path.txt` - Path with unicode characters

## Purpose

These fixtures validate that:
1. Doc IDs are generated deterministically from path + chunk_index
2. Paths with spaces are handled correctly in hashing
3. Unicode characters in paths are handled correctly
4. SHA256 truncation to 16 hex chars works consistently
