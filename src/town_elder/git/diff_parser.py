"""Git diff parser for town_elder."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

# Minimum parts expected in a "diff --git a/path b/path" line
_MIN_DIFF_LINE_PARTS = 4


@dataclass
class DiffFile:
    """Represents a file change in a diff."""
    path: str
    status: str  # added, modified, deleted
    hunks: list[str]


class DiffParser:
    """Parser for git diff output."""

    def parse(self, diff_output: str) -> Iterator[DiffFile]:  # noqa: PLR0912
        """Parse git diff output into file changes."""
        current_file = None
        current_status = None
        current_hunks: list[str] = []
        current_hunk_lines: list[str] = []

        for line in diff_output.split("\n"):
            # New file start
            if line.startswith("diff --git"):
                # Yield previous file if exists
                if current_file:
                    yield DiffFile(
                        path=current_file,
                        status=current_status or "modified",
                        hunks=current_hunks,
                    )

                # Parse the file path from "diff --git a/path b/path"
                parts = line.split()
                if len(parts) >= _MIN_DIFF_LINE_PARTS:
                    # Get the "b/" path
                    b_path = parts[-1]
                    current_file = b_path[2:] if b_path.startswith("b/") else b_path
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
