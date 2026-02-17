# Diff Fixtures

This directory contains git diff output samples for testing diff parsing.

## Files

- `simple_diff.diff` - Basic file addition
- `quoted_paths.diff` - Paths with spaces using quoted format
- `unquoted_paths.diff` - Paths with spaces using unquoted format

## Purpose

These fixtures validate that:
1. Diff headers are parsed correctly (diff --git a/path b/path)
2. Quoted paths with spaces are unquoted properly
3. File status (added, modified, deleted) is detected
4. Hunk headers (@@) are parsed correctly
