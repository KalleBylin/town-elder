"""File scanner for indexing project files."""

from __future__ import annotations

from pathlib import Path

# Default file extensions to index
DEFAULT_EXTENSIONS = frozenset({".py", ".md", ".rst"})

# Default exclusion patterns (directories and glob patterns)
DEFAULT_EXCLUDES = frozenset({
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    "venv",
    ".env",
    ".eggs",
    "*.egg-info",
    ".hg",
    ".svn",
    ".bzr",
    "vendor",
    "_build",  # Sphinx build output
})


def _should_exclude(path: Path, exclude_patterns: frozenset[str]) -> bool:
    """Check if path matches any exclusion pattern.

    Args:
        path: The file or directory path to check.
        exclude_patterns: Set of exclusion patterns.

    Returns:
        True if the path should be excluded, False otherwise.
    """
    parts = path.parts
    for pattern in exclude_patterns:
        if pattern.startswith("*"):
            # Handle glob patterns like *.egg-info
            suffix = pattern[1:]  # Remove the leading *
            if any(part.endswith(suffix) for part in parts):
                return True
        else:
            # Handle directory name patterns
            if pattern in parts:
                return True
    return False


def scan_files(
    root_path: Path,
    extensions: frozenset[str] | None = None,
    exclude_patterns: frozenset[str] | None = None,
) -> list[Path]:
    """Scan a directory for files with matching extensions.

    Args:
        root_path: Root directory to scan.
        extensions: File extensions to include (default: DEFAULT_EXTENSIONS).
        exclude_patterns: Patterns to exclude (default: DEFAULT_EXCLUDES).
                        These are added to the default excludes.

    Returns:
        List of Path objects for matching files, sorted for deterministic ordering.
    """
    if extensions is None:
        extensions = DEFAULT_EXTENSIONS

    # Start with default excludes, then add any custom patterns
    effective_excludes = DEFAULT_EXCLUDES.copy()
    if exclude_patterns:
        effective_excludes = effective_excludes | exclude_patterns

    # Find all files with matching extensions
    # Use chain.from_iterable to lazily combine glob results
    all_files: list[Path] = []
    for ext in extensions:
        all_files.extend(root_path.rglob(f"*{ext}"))

    # Filter out excluded files
    files_to_index = [
        f for f in all_files
        if not _should_exclude(f, effective_excludes)
    ]

    # Sort for deterministic ordering (stable across runs)
    # Sort by string representation for consistent ordering
    files_to_index.sort(key=lambda p: str(p))

    return files_to_index


def get_default_excludes() -> frozenset[str]:
    """Return the default exclusion patterns."""
    return DEFAULT_EXCLUDES.copy()


def get_default_extensions() -> frozenset[str]:
    """Return the default file extensions."""
    return DEFAULT_EXTENSIONS.copy()
