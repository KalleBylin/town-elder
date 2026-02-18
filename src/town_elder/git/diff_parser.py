"""Git diff parser for town_elder."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

# Minimum parts expected in a "diff --git a/path b/path" line
_MIN_DIFF_LINE_PARTS = 4
_MIN_UNQUOTED_PATH_PARTS = 2

# Git diff header patterns
_GIT_DIFF_HEADER_PREFIX = "diff --git"


def _parse_git_path(path: str) -> str:
    """Extract the path from a git diff header component.

    Handles both:
    - Unquoted: b/path or a/path
    - Quoted: "b/path" or "a/path"

    Args:
        path: The path component from the diff header (e.g., 'b/file.txt' or '"b/file.txt"')

    Returns:
        The file path without the a/ or b/ prefix
    """
    # Handle quoted paths - remove surrounding quotes
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1]

    # Remove leading a/ or b/ prefix
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]

    return path


def _extract_b_path(line: str) -> str | None:
    """Extract the 'b/' path from a 'diff --git' line.

    Handles both quoted and unquoted formats:
    - diff --git a/path b/path
    - diff --git "a/path with spaces" "b/path with spaces"

    Args:
        line: The diff --git header line

    Returns:
        The b/ path component (with or without quotes), or None if parsing fails
    """
    # Remove the 'diff --git ' prefix
    if not line.startswith(_GIT_DIFF_HEADER_PREFIX):
        return None

    remainder = line[len(_GIT_DIFF_HEADER_PREFIX):].lstrip()
    if not remainder:
        return None

    # Check if paths are quoted
    if remainder.startswith('"'):
        # Quoted format: "a/path" "b/path"
        # Find the closing quote of the first quoted string
        first_quote_end = remainder.find('"', 1)
        if first_quote_end == -1:
            return None

        # Find the opening quote of the second quoted string
        second_quote_start = remainder.find('"', first_quote_end + 1)
        if second_quote_start == -1:
            return None

        # Find the closing quote of the second quoted string
        second_quote_end = remainder.find('"', second_quote_start + 1)
        if second_quote_end == -1:
            # No closing quote - take rest of line
            second_quote_end = len(remainder)

        # Extract the b/ path (second quoted string)
        return remainder[second_quote_start + 1:second_quote_end]

    # Unquoted format: a/path b/path
    parts = remainder.split()
    if len(parts) >= _MIN_UNQUOTED_PATH_PARTS:
        return parts[-1]  # Last part is the b/ path
    return None


@dataclass
class DiffFile:
    """Represents a file change in a diff."""
    path: str
    status: str  # added, modified, deleted
    hunks: list[str]


class DiffParser:
    """Parser for git diff output."""

    def __init__(self, warn_on_parse_error: bool = True):
        """Initialize the parser.

        Args:
            warn_on_parse_error: If True, log warnings when diff headers fail to parse.
        """
        self.warn_on_parse_error = warn_on_parse_error

    def parse(self, diff_output: str) -> Iterator[DiffFile]:  # noqa: PLR0912
        """Parse git diff output into file changes."""
        import sys

        current_file = None
        current_status = None
        current_hunks: list[str] = []
        current_hunk_lines: list[str] = []
        parse_error_count = 0

        for line in diff_output.split("\n"):
            # New file start
            if line.startswith("diff --git"):
                # Yield previous file if exists
                if current_file:
                    if current_hunk_lines:
                        current_hunks.append("\n".join(current_hunk_lines))
                    yield DiffFile(
                        path=current_file,
                        status=current_status or "modified",
                        hunks=current_hunks,
                    )

                # Parse the file path from "diff --git a/path b/path" or quoted variant
                b_path = _extract_b_path(line)
                if b_path:
                    current_file = _parse_git_path(b_path)
                else:
                    # Failed to parse the diff header - don't associate content with wrong file
                    if self.warn_on_parse_error:
                        print(
                            f"Warning: Failed to parse diff header: {line[:60]}...",
                            file=sys.stderr,
                        )
                    parse_error_count += 1
                    current_file = None  # Explicitly set to None to prevent content association
                current_status = None
                current_hunks = []
                current_hunk_lines = []

            # File status
            elif line.startswith("new file"):
                current_status = "added"
            elif line.startswith("deleted file"):
                current_status = "deleted"
            elif line.startswith("old mode") or line.startswith("new mode"):
                pass  # Ignore mode changes

            # Hunk header
            elif line.startswith("@@"):
                if current_hunk_lines:
                    current_hunks.append("\n".join(current_hunk_lines))
                current_hunk_lines = [line]

            # Regular diff content
            elif current_file:
                current_hunk_lines.append(line)

        # Yield the last file
        if current_file:
            if current_hunk_lines:
                current_hunks.append("\n".join(current_hunk_lines))
            yield DiffFile(
                path=current_file,
                status=current_status or "modified",
                hunks=current_hunks,
            )

    def parse_diff_to_text(self, diff_output: str) -> str:
        """Convert a diff to plain text for embedding."""
        parts = []
        for diff_file in self.parse(diff_output):
            parts.append(f"File: {diff_file.path} ({diff_file.status})")
            for hunk in diff_file.hunks:
                parts.append(hunk)
        return "\n\n".join(parts)
