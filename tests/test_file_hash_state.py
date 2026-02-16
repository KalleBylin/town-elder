"""Tests for file hash state management."""

import json

import pytest

from town_elder.cli import (
    _get_file_state_path,
    _get_repo_id,
    _load_file_state,
    _save_file_state,
)


@pytest.fixture
def state_file(tmp_path):
    """Create a temporary state file."""
    return tmp_path / "file_index_state.json"


@pytest.fixture
def repo_path(tmp_path):
    """Create a temporary repo path."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    return repo


class TestFileStatePath:
    """Tests for _get_file_state_path function."""

    def test_returns_correct_filename(self, tmp_path):
        """Should return path with correct filename."""
        data_dir = tmp_path / ".town_elder"
        data_dir.mkdir()

        result = _get_file_state_path(data_dir)

        assert result.name == "file_index_state.json"
        assert result.parent == data_dir


class TestLoadFileState:
    """Tests for _load_file_state function."""

    def test_returns_empty_dict_for_missing_file(self, state_file, repo_path):
        """Should return empty dict when file doesn't exist."""
        result = _load_file_state(state_file, repo_path)
        assert result == {}

    def test_loads_existing_state(self, state_file, repo_path):
        """Should load existing file state."""
        repo_id = _get_repo_id(repo_path)
        state_file.write_text(json.dumps({
            repo_id: {
                "repo_path": str(repo_path),
                "file_hashes": {"src/main.py": "abc123"},
            }
        }))

        result = _load_file_state(state_file, repo_path)

        assert repo_id in result
        assert result[repo_id]["file_hashes"]["src/main.py"] == "abc123"

    def test_handles_invalid_json(self, state_file, repo_path):
        """Should return empty dict for invalid JSON."""
        state_file.write_text("not valid json")

        result = _load_file_state(state_file, repo_path)
        assert result == {}


class TestSaveFileState:
    """Tests for _save_file_state function."""

    def test_saves_file_hashes(self, state_file, repo_path):
        """Should save file hashes to state file."""
        file_hashes = {
            "src/main.py": "abc123",
            "src/utils.py": "def456",
        }

        _save_file_state(state_file, repo_path, file_hashes)

        assert state_file.exists()

        # Verify content
        content = json.loads(state_file.read_text())
        repo_id = _get_repo_id(repo_path)

        assert repo_id in content
        assert content[repo_id]["file_hashes"] == file_hashes
        assert "repo_path" in content[repo_id]
        assert "updated_at" in content[repo_id]

    def test_preserves_other_repos_state(self, state_file, repo_path):
        """Should preserve other repo's state when saving."""
        other_repo = repo_path.parent / "other_repo"
        other_repo.mkdir()

        # Save state for first repo
        _save_file_state(state_file, repo_path, {"file1.py": "hash1"})

        # Save state for second repo
        _save_file_state(state_file, other_repo, {"file2.py": "hash2"})

        # Both should be present
        content = json.loads(state_file.read_text())
        assert _get_repo_id(repo_path) in content
        assert _get_repo_id(other_repo) in content

    def test_updates_existing_repo_state(self, state_file, repo_path):
        """Should update existing repo's state."""
        # First save
        _save_file_state(state_file, repo_path, {"file1.py": "hash1"})

        # Second save with new hashes
        _save_file_state(state_file, repo_path, {"file1.py": "hash2", "file2.py": "hash3"})

        content = json.loads(state_file.read_text())
        repo_id = _get_repo_id(repo_path)

        # Should have updated hashes
        assert content[repo_id]["file_hashes"]["file1.py"] == "hash2"
        assert content[repo_id]["file_hashes"]["file2.py"] == "hash3"
