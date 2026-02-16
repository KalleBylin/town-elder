"""Tests for the git hash scanner module."""

import subprocess
from pathlib import Path

import pytest

from town_elder.indexing.git_hash_scanner import (
    TrackedFile,
    get_git_root,
    scan_git_blobs,
)

# SHA-1 hex length constant
_SHA1_HEX_LENGTH = 40


class TestGetGitRoot:
    """Tests for get_git_root function."""

    def test_get_git_root_returns_path_for_git_repo(self):
        """Should return path for a git repository."""
        root = get_git_root(Path("."))
        assert root is not None
        assert (root / ".git").exists()

    def test_get_git_root_finds_root_from_subdirectory(self):
        """Should find root when called from subdirectory."""
        root = get_git_root(Path("src/town_elder"))
        assert root is not None
        assert root.name == "town_elder"

    def test_get_git_root_returns_none_for_non_repo(self, tmp_path):
        """Should return None for non-git directory."""
        result = get_git_root(tmp_path)
        assert result is None


class TestScanGitBlobs:
    """Tests for scan_git_blobs function."""

    def test_scan_git_blobs_returns_tracked_files(self):
        """Should return all tracked files with blob hashes."""
        root = get_git_root(Path("."))
        assert root is not None

        blobs = scan_git_blobs(root)

        # Should find tracked files
        assert len(blobs) > 0

        # Each value should be a TrackedFile with path and blob_hash
        for path, tracked_file in blobs.items():
            assert isinstance(tracked_file, TrackedFile)
            assert tracked_file.path == path
            assert len(tracked_file.blob_hash) == _SHA1_HEX_LENGTH  # noqa: PLR2004
            assert tracked_file.mode in ("100644", "100755")

    def test_scan_git_blobs_finds_source_files(self):
        """Should find Python source files."""
        root = get_git_root(Path("."))
        assert root is not None

        blobs = scan_git_blobs(root)

        # Should find Python files
        py_files = [p for p in blobs if p.endswith(".py")]
        assert len(py_files) > 0

    def test_scan_git_blobs_handles_non_repo(self, tmp_path):
        """Should raise error for non-git directory."""
        with pytest.raises(subprocess.CalledProcessError):
            scan_git_blobs(tmp_path)


class TestTrackedFile:
    """Tests for TrackedFile dataclass."""

    def test_tracked_file_properties(self):
        """TrackedFile should have correct properties."""
        tracked = TrackedFile(
            path="src/main.py",
            blob_hash="a" * _SHA1_HEX_LENGTH,
            mode="100644",
        )

        assert tracked.path == "src/main.py"
        assert tracked.blob_hash == "a" * _SHA1_HEX_LENGTH
        assert tracked.mode == "100644"
        assert tracked.relative_path == Path("src/main.py")
