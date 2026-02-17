# File Scanner Fixtures

This directory contains fixtures for testing file scanning behavior.

## Directories

- `simple_files/` - Basic .py, .md, .rst files in root directory
- `nested_dirs/` - Files in nested subdirectories to test recursive scanning

## Purpose

These fixtures validate that:
1. File scanner finds files with correct extensions
2. Nested directory scanning works correctly
3. Exclusion patterns work properly
