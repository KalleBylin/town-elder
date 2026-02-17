# Commit Fixtures

This directory contains JSON fixtures for testing commit parsing.

## Files

- `simple_commit.json` - Basic commit with ASCII message
- `unicode_commit.json` - Commit with unicode in message and file paths

## Purpose

These fixtures validate that:
1. Commit hash, message, author, date are parsed correctly
2. Unicode in commit messages is preserved
3. Files changed list is populated correctly
