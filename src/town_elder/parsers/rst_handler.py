"""RST (reStructuredText) parser with semantic chunking and directive extraction."""
from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Valid RST heading underline characters (rank from highest to lowest)
RST_HEADING_CHARS = "=-~^+*`"


@dataclass
class RSTChunk:
    """Represents a chunk from an RST document."""

    text: str
    section_path: list[str] = field(default_factory=list)
    directives: dict[str, list[str]] = field(default_factory=dict)
    temporal_tags: list[str] = field(default_factory=list)
    chunk_index: int = 0


def _find_section_boundaries(content: str) -> list[tuple[int, str, str]]:
    """Find all section boundaries in RST content.

    Returns list of (line_number, heading_text, underline) tuples.
    Uses only valid RST heading underline characters.
    """
    lines = content.split("\n")
    boundaries = []

    for i, line in enumerate(lines):
        # Skip empty lines
        if not line.strip():
            continue

        # Check if this line could be an underline (previous line was heading)
        if i > 0:
            prev_line = lines[i - 1]
            # Underline must be at least as long as the heading text
            if len(line) >= len(prev_line.strip()) and all(
                c in RST_HEADING_CHARS for c in line.strip()
            ):
                boundaries.append((i, prev_line.strip(), line.strip()))

    return boundaries


def _get_heading_level(underline: str) -> int:
    """Get heading level from underline character."""
    if not underline:
        return 0
    char = underline[0]
    if char == "=":
        return 1
    if char == "-":
        return 2
    if char == "~":
        return 3
    if char == "^":
        return 4
    if char == "+":
        return 5
    if char == "*":
        return 6
    if char == "`":
        return 7
    return 0


def _build_section_tree(boundaries: list[tuple[int, str, str]]) -> list[dict[str, Any]]:
    """Build a tree-like structure of sections with their levels."""
    sections = []
    for line_num, heading, underline in boundaries:
        level = _get_heading_level(underline)
        sections.append({
            "line": line_num,
            "heading": heading,
            "level": level,
        })
    return sections


def _extract_directives(chunk_text: str) -> tuple[dict[str, list[str]], list[str]]:
    """Extract directive content from chunk text.

    Returns tuple of (directives dict, temporal_tags list).
    """
    directives: dict[str, list[str]] = {}
    temporal_tags: list[str] = []

    # Patterns for extracting directive content
    directive_patterns = {
        "note": re.compile(r"\.\. note::\s*\n((?:[ \t]+.+\n?)+)", re.MULTILINE),
        "warning": re.compile(r"\.\. warning::\s*\n((?:[ \t]+.+\n?)+)", re.MULTILINE),
        "versionadded": re.compile(
            r"\.\. versionadded::\s*(.*?)\n((?:[ \t]+.+\n?)+)", re.MULTILINE
        ),
        "deprecated": re.compile(r"\.\. deprecated::\s*(.*?)\n", re.MULTILINE),
        "versionchanged": re.compile(r"\.\. versionchanged::\s*(.*?)\n", re.MULTILINE),
    }

    for name, pattern in directive_patterns.items():
        matches = pattern.findall(chunk_text)
        if matches:
            if name in ("deprecated", "versionchanged"):
                # These are temporal tags
                for match in matches:
                    temporal_tags.append(f"{name}: {match}" if match else name)
            else:
                # These are directive contents
                directives[name] = [m.strip() if isinstance(m, str) else m[1].strip() for m in matches]

    return directives, temporal_tags


def _extract_directives_content(text: str) -> dict[str, list[str]]:
    """Extract directive content bodies for note, warning, versionadded."""
    directives: dict[str, list[str]] = {}

    # Note directive
    note_pattern = re.compile(
        r"\.\. note::(?:\s+(.*))?\n((?:[ \t]+.+\n?)+)", re.MULTILINE
    )
    notes = note_pattern.findall(text)
    if notes:
        directives["note"] = [content.strip() for _, content in notes]

    # Warning directive
    warning_pattern = re.compile(
        r"\.\. warning::(?:\s+(.*))?\n((?:[ \t]+.+\n?)+)", re.MULTILINE
    )
    warnings_list = warning_pattern.findall(text)
    if warnings_list:
        directives["warning"] = [content.strip() for _, content in warnings_list]

    # Versionadded directive (handles optional blank line after directive)
    versionadded_pattern = re.compile(
        r"\.\. versionadded::(?:\s+(.*))?\n\n?((?:[ \t]+.+\n?)+)", re.MULTILINE
    )
    versions_added = versionadded_pattern.findall(text)
    if versions_added:
        directives["versionadded"] = [content.strip() for _, content in versions_added]
    else:
        # Try alternative pattern with optional arg and content directly after
        versionadded_pattern2 = re.compile(
            r"\.\. versionadded::(?:\s+(.*))?\n((?:[ \t]+.+\n?)+)", re.MULTILINE
        )
        versions_added = versionadded_pattern2.findall(text)
        if versions_added:
            directives["versionadded"] = [content.strip() for _, content in versions_added]

    return directives


def _check_temporal_tags(text: str) -> list[str]:
    """Check for temporal directives and return tags."""
    tags: list[str] = []

    # Deprecated
    deprecated_pattern = re.compile(r"\.\. deprecated::\s*(.*?)(?:\n|$)", re.MULTILINE)
    if deprecated_pattern.search(text):
        tags.append("deprecated")

    # Version changed
    versionchanged_pattern = re.compile(r"\.\. versionchanged::\s*(.*?)(?:\n|$)", re.MULTILINE)
    if versionchanged_pattern.search(text):
        tags.append("versionchanged")

    return tags


def _chunk_by_sections(content: str) -> list[tuple[int, int, list[str]]]:
    """Split content into chunks based on section boundaries.

    Returns list of (start_line, end_line, section_path) tuples.
    """
    lines = content.split("\n")
    boundaries = _find_section_boundaries(content)

    if not boundaries:
        # No sections found, return whole content as single chunk
        return [(0, len(lines), [])]

    chunks = []
    current_path: list[tuple[str, int]] = []  # (heading, level)

    for i, (line_num, heading, underline) in enumerate(boundaries):
        # Update current section path
        level = _get_heading_level(underline)

        # Remove any sections at same or higher level from path
        while current_path and current_path[-1][1] >= level:
            current_path.pop()

        # Add current section
        current_path.append((heading, level))

        # Determine chunk boundaries
        start_line = line_num + 1  # Content starts after underline
        end_line = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(lines)

        # Extract section path as list of headings
        section_path = [h for h, _ in current_path]

        chunks.append((start_line, end_line, section_path))

    return chunks


def _get_chunk_text(lines: list[str], start: int, end: int) -> str:
    """Get text content for a chunk from line range."""
    return "\n".join(lines[start:end]).strip()


def parse_rst_content(content: str) -> list[RSTChunk]:
    """Parse RST content and return list of chunks with metadata.

    This function is multiprocessing-safe (top-level function, no closures).

    Args:
        content: The RST content as a string.

    Returns:
        List of RSTChunk objects, each containing text, section path,
        directives, and temporal tags.
    """
    if not content or not content.strip():
        return []

    chunks: list[RSTChunk] = []

    try:
        # Handle common docutils parsing issues
        # First, try to normalize the content
        normalized = content.replace("\r\n", "\n")

        lines = normalized.split("\n")
        section_chunks = _chunk_by_sections(normalized)

        for idx, (start, end, section_path) in enumerate(section_chunks):
            chunk_text = _get_chunk_text(lines, start, end)

            if not chunk_text.strip():
                continue

            # Extract directives
            directives = _extract_directives_content(chunk_text)

            # Check for temporal tags
            temporal_tags = _check_temporal_tags(chunk_text)

            chunk = RSTChunk(
                text=chunk_text,
                section_path=section_path,
                directives=directives,
                temporal_tags=temporal_tags,
                chunk_index=idx,
            )
            chunks.append(chunk)

    except Exception as e:
        # Log warning but don't crash - malformed RST should not stop indexing
        warnings.warn(f"Failed to parse RST content: {e}", stacklevel=2)
        logger.warning("RST parsing error: %s", e)

    return chunks


def parse_rst_file(file_path: str) -> list[RSTChunk]:
    """Parse an RST file and return list of chunks.

    This function is multiprocessing-safe (top-level function, no closures).

    Args:
        file_path: Path to the RST file.

    Returns:
        List of RSTChunk objects.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
        return parse_rst_content(content)
    except Exception as e:
        # Log warning but don't crash - malformed file should not stop indexing
        warnings.warn(f"Failed to read RST file {file_path}: {e}", stacklevel=2)
        logger.warning("RST file read error for %s: %s", file_path, e)
        return []


def get_chunk_metadata(chunk: RSTChunk) -> dict[str, Any]:
    """Extract metadata from a chunk for indexing.

    Args:
        chunk: An RSTChunk object.

    Returns:
        Dictionary containing chunk metadata.
    """
    metadata: dict[str, Any] = {
        "chunk_index": chunk.chunk_index,
        "section_path": chunk.section_path,
        "has_directives": bool(chunk.directives),
        "has_temporal_tags": bool(chunk.temporal_tags),
    }

    # Add directive content if present
    if chunk.directives:
        metadata["directives"] = chunk.directives

    # Add temporal tags if present
    if chunk.temporal_tags:
        metadata["temporal_tags"] = chunk.temporal_tags

    # Add section depth
    if chunk.section_path:
        metadata["section_depth"] = len(chunk.section_path)

    return metadata
