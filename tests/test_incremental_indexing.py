"""Tests for incremental file indexing behavior.

These tests verify the core logic without requiring full integration setup.
"""

import hashlib
import subprocess
from pathlib import Path

import pytest

from town_elder.cli import (
    _get_file_state_path,
    _get_repo_id,
    _load_file_state,
    _save_file_state,
)
from town_elder.indexing.git_hash_scanner import scan_git_blobs

# SHA-1 hex length constant
_SHA1_HEX_LENGTH = 40
_EXPECTED_FILE_COUNT = 3


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repository with some files."""
    repo = tmp_path / "test_repo"
    repo.mkdir()

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    # Create initial files
    (repo / "main.py").write_text("print('hello')")
    (repo / "README.md").write_text("# Test Project")
    (repo / "docs" / "index.rst").parent.mkdir(parents=True)
    (repo / "docs" / "index.rst").write_text("Test docs")

    # Stage and commit
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    return repo


def get_doc_id(path: Path) -> str:
    """Generate doc_id the same way as CLI."""
    return hashlib.sha256(str(path).encode()).hexdigest()[:16]


class TestIncrementalIndexingLogic:
    """Tests for incremental indexing logic."""

    def test_scan_git_blobs_returns_expected_files(self, git_repo):
        """scan_git_blobs should return all tracked files."""
        blobs = scan_git_blobs(git_repo)

        # Should find 3 files
        assert len(blobs) == _EXPECTED_FILE_COUNT  # noqa: PLR2004
        paths = set(blobs.keys())
        assert "main.py" in paths
        assert "README.md" in paths
        assert "docs/index.rst" in paths

    def test_blob_hashes_are_sha1(self, git_repo):
        """Blob hashes should be valid SHA-1 (40 hex chars)."""
        blobs = scan_git_blobs(git_repo)

        for _path, tracked_file in blobs.items():
            assert len(tracked_file.blob_hash) == _SHA1_HEX_LENGTH  # noqa: PLR2004
            # Should be valid hex
            int(tracked_file.blob_hash, 16)

    def test_state_saves_blob_hashes(self, git_repo, tmp_path):
        """File state should save blob hashes."""
        data_dir = tmp_path / ".town_elder"
        data_dir.mkdir()
        state_file = _get_file_state_path(data_dir)

        blobs = scan_git_blobs(git_repo)
        file_hashes = {path: blob.blob_hash for path, blob in blobs.items()}

        _save_file_state(state_file, git_repo, file_hashes)

        # Verify state was saved correctly
        loaded = _load_file_state(state_file, git_repo)
        repo_id = _get_repo_id(git_repo)

        assert repo_id in loaded
        assert loaded[repo_id]["file_hashes"] == file_hashes

    def test_detect_changed_file(self, git_repo, tmp_path):
        """Should detect when a file has changed."""
        data_dir = tmp_path / ".town_elder"
        data_dir.mkdir()
        state_file = _get_file_state_path(data_dir)

        # Get initial hashes
        initial_blobs = scan_git_blobs(git_repo)
        initial_hashes = {path: blob.blob_hash for path, blob in initial_blobs.items()}
        _save_file_state(state_file, git_repo, initial_hashes)

        # Modify a file
        (git_repo / "main.py").write_text("print('world')")
        subprocess.run(
            ["git", "add", "main.py"],
            cwd=git_repo,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Modify main.py"],
            cwd=git_repo,
            check=True,
            capture_output=True,
        )

        # Get new hashes
        new_blobs = scan_git_blobs(git_repo)
        new_hashes = {path: blob.blob_hash for path, blob in new_blobs.items()}

        # Compare
        assert new_hashes["main.py"] != initial_hashes["main.py"]
        # Other files should be unchanged
        assert new_hashes["README.md"] == initial_hashes["README.md"]
        assert new_hashes["docs/index.rst"] == initial_hashes["docs/index.rst"]

    def test_detect_deleted_file(self, git_repo, tmp_path):
        """Should detect when a file is deleted."""
        data_dir = tmp_path / ".town_elder"
        data_dir.mkdir()
        state_file = _get_file_state_path(data_dir)

        # Get initial hashes
        initial_blobs = scan_git_blobs(git_repo)
        initial_hashes = {path: blob.blob_hash for path, blob in initial_blobs.items()}
        _save_file_state(state_file, git_repo, initial_hashes)

        # Delete a file
        (git_repo / "README.md").unlink()
        subprocess.run(
            ["git", "add", "README.md"],
            cwd=git_repo,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Delete README.md"],
            cwd=git_repo,
            check=True,
            capture_output=True,
        )

        # Get new hashes
        new_blobs = scan_git_blobs(git_repo)

        # README.md should not be in new blobs
        assert "README.md" not in new_blobs
        # Other files should still be present
        assert "main.py" in new_blobs
        assert "docs/index.rst" in new_blobs

    def test_detect_added_file(self, git_repo, tmp_path):
        """Should detect when a file is added."""
        data_dir = tmp_path / ".town_elder"
        data_dir.mkdir()
        state_file = _get_file_state_path(data_dir)

        # Get initial hashes
        initial_blobs = scan_git_blobs(git_repo)
        initial_hashes = {path: blob.blob_hash for path, blob in initial_blobs.items()}
        _save_file_state(state_file, git_repo, initial_hashes)

        # Add a file
        (git_repo / "new_file.py").write_text("# New file")
        subprocess.run(
            ["git", "add", "new_file.py"],
            cwd=git_repo,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Add new file"],
            cwd=git_repo,
            check=True,
            capture_output=True,
        )

        # Get new hashes
        new_blobs = scan_git_blobs(git_repo)

        # new_file.py should be in new blobs
        assert "new_file.py" in new_blobs
        # Original files should still be present
        assert "main.py" in new_blobs


class TestIncrementalIndexingSkipLogic:
    """Tests for determining which files to skip."""

    def test_files_with_same_hash_should_be_skipped(self):
        """Files with same blob hash should be skipped."""
        # Simulate two runs with same hashes
        old_hashes = {
            "main.py": "abc123",
            "README.md": "def456",
        }
        new_hashes = {
            "main.py": "abc123",  # Same - should skip
            "README.md": "def456",  # Same - should skip
        }

        # Files to process = files where hash changed or is new
        files_to_process = []
        for path, new_hash in new_hashes.items():
            old_hash = old_hashes.get(path)
            if old_hash != new_hash:
                files_to_process.append(path)

        assert len(files_to_process) == 0

    def test_files_with_different_hash_should_be_processed(self):
        """Files with different blob hash should be processed."""
        old_hashes = {
            "main.py": "abc123",
            "README.md": "def456",
        }
        new_hashes = {
            "main.py": "abc123updated",  # Changed - should process
            "README.md": "def456",  # Same - should skip
        }

        files_to_process = []
        for path, new_hash in new_hashes.items():
            old_hash = old_hashes.get(path)
            if old_hash != new_hash:
                files_to_process.append(path)

        assert len(files_to_process) == 1
        assert "main.py" in files_to_process

    def test_new_files_should_be_processed(self):
        """New files should always be processed."""
        old_hashes = {
            "main.py": "abc123",
        }
        new_hashes = {
            "main.py": "abc123",
            "new_file.py": "new hash",  # New - should process
        }

        files_to_process = []
        for path, new_hash in new_hashes.items():
            old_hash = old_hashes.get(path)
            if old_hash != new_hash:
                files_to_process.append(path)

        assert len(files_to_process) == 1
        assert "new_file.py" in files_to_process
