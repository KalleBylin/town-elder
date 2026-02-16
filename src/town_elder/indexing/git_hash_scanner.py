"""Git blob hash scanner for incremental indexing.

This module provides functionality to scan git-tracked files and retrieve their
blob SHA-1 hashes, enabling incremental indexing by detecting changed files.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

# Constants for git ls-files --stage parsing
_EXPECTED_TAB_PARTS = 2
_EXPECTED_META_PARTS = 3
_VALID_FILE_MODES = frozenset({"100644", "100755"})


@dataclass
class TrackedFile:
    """Represents a git-tracked file with its blob hash."""
    path: str  # Relative path from repo root
    blob_hash: str  # 40-character SHA-1 hex digest
    mode: str  # File mode (e.g., "100644")

    @property
    def relative_path(self) -> Path:
        """Return path as Path object."""
        return Path(self.path)


def scan_git_blobs(repo_path: Path) -> dict[str, TrackedFile]:
    """Scan git-tracked files and return their blob hashes.

    Uses `git ls-files --stage` to retrieve all tracked files with their
    blob SHA-1 hashes. This is more efficient than computing content hashes
    manually as git already has this information.

    Args:
        repo_path: Path to the git repository root.

    Returns:
        Dictionary mapping relative file paths (strings) to TrackedFile objects.

    Raises:
        subprocess.CalledProcessError: If git command fails.
        FileNotFoundError: If git is not installed.
    """
    cmd = ["git", "-C", str(repo_path), "ls-files", "--stage"]

    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        encoding="utf-8",
        check=True,
    )

    tracked_files: dict[str, TrackedFile] = {}

    for line in result.stdout.strip().split("\n"):
        if not line:
            continue

        # Format: <mode> <blob_hash> <stage>\t<path>
        # Example: 100644 0acd8c61f0d9dc1e5db7ad7c2dbce2bb16b8d6de 0	.beads/.gitignore
        parts = line.split("\t")
        if len(parts) != _EXPECTED_TAB_PARTS:
            continue

        metadata, file_path = parts[0], parts[1]

        # Parse metadata: mode, blob_hash, stage
        meta_parts = metadata.split()
        if len(meta_parts) != _EXPECTED_META_PARTS:
            continue

        mode, blob_hash, _stage = meta_parts

        # Skip symlinks (mode 120000) and other special files
        if mode not in _VALID_FILE_MODES:
            continue

        # file_path could contain spaces, but we split on tab so it's preserved
        tracked_files[file_path] = TrackedFile(
            path=file_path,
            blob_hash=blob_hash,
            mode=mode,
        )

    return tracked_files


def get_git_root(repo_path: Path) -> Path | None:
    """Find the git repository root for a given path.

    Args:
        repo_path: Any path within a git repository.

    Returns:
        Path to the git repository root, or None if not in a git repo.
    """
    try:
        cmd = ["git", "-C", str(repo_path), "rev-parse", "--show-toplevel"]
        result = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            encoding="utf-8",
            check=True,
        )
        root = result.stdout.strip()
        return Path(root) if root else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
