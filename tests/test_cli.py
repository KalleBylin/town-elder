"""CLI integration tests for replay."""
from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def temp_git_repo() -> Iterator[Path]:
    """Create a temporary git repository for testing."""
    with tempfile.TemporaryDirectory() as tmp:
        repo_path = Path(tmp) / "test_repo"
        repo_path.mkdir()
        # Initialize git repo
        subprocess.run(["git", "init"], cwd=repo_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=repo_path,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_path,
            capture_output=True,
        )
        # Create initial commit
        (repo_path / "test.txt").write_text("initial content")
        subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial commit"],
            cwd=repo_path,
            capture_output=True,
        )
        yield repo_path


class TestHookInstall:
    """Tests for hook install functionality."""

    def test_hook_install_preserves_data_dir(self, temp_git_repo: Path):
        """Hook install should preserve --data-dir in generated hook script."""
        custom_data_dir = temp_git_repo / ".custom-replay"

        # Initialize with custom data-dir
        result = subprocess.run(
            [
                "uv", "run", "replay",
                "--data-dir", str(custom_data_dir),
                "init",
                "--path", str(temp_git_repo),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Install hook with custom data-dir
        result = subprocess.run(
            [
                "uv", "run", "replay",
                "--data-dir", str(custom_data_dir),
                "hook", "install",
                "--repo", str(temp_git_repo),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Verify the hook contains the data-dir
        hook_path = temp_git_repo / ".git" / "hooks" / "post-commit"
        hook_content = hook_path.read_text()
        assert str(custom_data_dir) in hook_content
        assert "--data-dir" in hook_content

    def test_hook_install_works_without_data_dir(self, temp_git_repo: Path):
        """Hook install should work without explicit --data-dir."""
        # Initialize with default location
        result = subprocess.run(
            ["uv", "run", "replay", "init", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Install hook without data-dir
        result = subprocess.run(
            ["uv", "run", "replay", "hook", "install", "--repo", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Verify the hook exists and is executable
        hook_path = temp_git_repo / ".git" / "hooks" / "post-commit"
        assert hook_path.exists()
        assert hook_path.stat().st_mode & 0o111  # executable


class TestHookUninstall:
    """Tests for hook uninstall safety."""

    def test_uninstall_refuses_non_replay_hook(self, temp_git_repo: Path):
        """Hook uninstall should refuse to delete non-Replay hooks by default."""
        hook_path = temp_git_repo / ".git" / "hooks" / "post-commit"
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        hook_path.write_text("#!/bin/bash\necho 'custom hook'")
        hook_path.chmod(0o755)

        result = subprocess.run(
            ["uv", "run", "replay", "hook", "uninstall", "--repo", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "not a Replay hook" in result.stderr or "not a Replay hook" in result.stdout

    def test_uninstall_allows_force_on_non_replay_hook(self, temp_git_repo: Path):
        """Hook uninstall should allow deletion with --force."""
        hook_path = temp_git_repo / ".git" / "hooks" / "post-commit"
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        hook_path.write_text("#!/bin/bash\necho 'custom hook'")
        hook_path.chmod(0o755)

        result = subprocess.run(
            ["uv", "run", "replay", "hook", "uninstall", "--repo", str(temp_git_repo), "--force"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert not hook_path.exists()

    def test_uninstall_removes_replay_hook(self, temp_git_repo: Path):
        """Hook uninstall should remove Replay hooks without --force."""
        # Install replay hook first
        subprocess.run(
            ["uv", "run", "replay", "hook", "install", "--repo", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )

        hook_path = temp_git_repo / ".git" / "hooks" / "post-commit"
        assert hook_path.exists()

        result = subprocess.run(
            ["uv", "run", "replay", "hook", "uninstall", "--repo", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert not hook_path.exists()


class TestPathSemantics:
    """Tests for path/data-dir semantics."""

    def test_data_dir_option(self, temp_git_repo: Path):
        """--data-dir option should work across commands."""
        data_dir = temp_git_repo / ".replay"

        # Initialize with explicit path
        result = subprocess.run(
            ["uv", "run", "replay", "--data-dir", str(data_dir), "init", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert data_dir.exists()

        # Stats should use the same data dir
        result = subprocess.run(
            ["uv", "run", "replay", "--data-dir", str(data_dir), "stats"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert str(data_dir) in output

    def test_data_dir_isolation_from_default_store(self, temp_git_repo: Path):
        """--data-dir should isolate reads/writes from the default cwd store."""
        root_marker = "ROOT_MARKER_12345"
        custom_marker = "CUSTOM_MARKER_67890"
        custom_data_dir = temp_git_repo / ".custom-replay"

        # Seed the default cwd store.
        subprocess.run(
            ["uv", "run", "replay", "init", "--path", str(temp_git_repo)],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            ["uv", "run", "replay", "add", "--text", root_marker],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
            check=True,
        )

        # Seed an explicit custom store.
        subprocess.run(
            [
                "uv",
                "run",
                "replay",
                "--data-dir",
                str(custom_data_dir),
                "init",
                "--path",
                str(temp_git_repo),
            ],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            [
                "uv",
                "run",
                "replay",
                "--data-dir",
                str(custom_data_dir),
                "add",
                "--text",
                custom_marker,
            ],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
            check=True,
        )

        # Export from custom store and ensure no leakage from default store.
        result = subprocess.run(
            [
                "uv",
                "run",
                "replay",
                "--data-dir",
                str(custom_data_dir),
                "export",
                "--output",
                "-",
                "--format",
                "json",
            ],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert custom_marker in output
        assert root_marker not in output

    def test_export_to_stdout_without_format_option(self, temp_git_repo: Path):
        """Export to stdout without --format should default to json."""
        # Initialize and add data
        subprocess.run(
            ["uv", "run", "replay", "init", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            ["uv", "run", "replay", "add", "--text", "hello world"],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
            check=True,
        )

        # Export to stdout WITHOUT --format option (the bug case)
        result = subprocess.run(
            ["uv", "run", "replay", "export", "--output", "-"],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Export failed: {result.stderr}"
        output = result.stdout + result.stderr
        # Should contain valid JSON output
        assert "hello world" in output


class TestModuleEntrypoints:
    """Tests for python -m module entrypoints."""

    def test_python_m_replay_cli_runs(self):
        """python -m replay.cli should be executable."""
        result = subprocess.run(
            ["uv", "run", "python", "-m", "replay.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Usage:" in result.stdout
        assert "replay.cli" in result.stdout

    def test_python_m_replay_runs(self):
        """python -m replay should be executable."""
        result = subprocess.run(
            ["uv", "run", "python", "-m", "replay", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Usage:" in result.stdout
        assert "python -m replay" in result.stdout


class TestMetadataValidation:
    """Tests for metadata validation."""

    def test_add_rejects_invalid_json_metadata(self, temp_git_repo: Path):
        """replay add should reject invalid JSON metadata."""
        data_dir = temp_git_repo / ".replay"
        subprocess.run(
            ["uv", "run", "replay", "--data-dir", str(data_dir), "init", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )

        result = subprocess.run(
            ["uv", "run", "replay", "--data-dir", str(data_dir), "add", "--text", "test", "--metadata", "{invalid json"],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "JSON" in result.stderr or "JSON" in result.stdout

    def test_add_accepts_valid_json_metadata(self, temp_git_repo: Path):
        """replay add should accept valid JSON metadata."""
        data_dir = temp_git_repo / ".replay"
        subprocess.run(
            ["uv", "run", "replay", "--data-dir", str(data_dir), "init", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )

        metadata = json.dumps({"source": "test", "type": "example"})
        result = subprocess.run(
            ["uv", "run", "replay", "--data-dir", str(data_dir), "add", "--text", "test document", "--metadata", metadata],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

    def test_add_rejects_non_object_json_metadata(self, temp_git_repo: Path):
        """replay add should reject JSON metadata that is not an object."""
        data_dir = temp_git_repo / ".replay"
        subprocess.run(
            ["uv", "run", "replay", "--data-dir", str(data_dir), "init", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )

        # Test with a JSON array (list)
        metadata = json.dumps(["item1", "item2"])
        result = subprocess.run(
            ["uv", "run", "replay", "--data-dir", str(data_dir), "add", "--text", "test", "--metadata", metadata],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "object" in result.stderr.lower() or "dict" in result.stderr.lower()

    def test_add_rejects_json_string_metadata(self, temp_git_repo: Path):
        """replay add should reject JSON metadata that is a string."""
        data_dir = temp_git_repo / ".replay"
        subprocess.run(
            ["uv", "run", "replay", "--data-dir", str(data_dir), "init", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )

        # Test with a JSON string
        metadata = json.dumps("just a string")
        result = subprocess.run(
            ["uv", "run", "replay", "--data-dir", str(data_dir), "add", "--text", "test", "--metadata", metadata],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "object" in result.stderr.lower() or "dict" in result.stderr.lower()


class TestExitCodes:
    """Tests for exit codes."""

    def test_init_exits_zero_on_success(self, temp_git_repo: Path):
        """replay init should exit with 0 on success."""
        data_dir = temp_git_repo / ".replay_new"
        result = subprocess.run(
            ["uv", "run", "replay", "--data-dir", str(data_dir), "init", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_init_exits_nonzero_on_error(self):
        """replay init should exit with non-zero on error."""
        result = subprocess.run(
            ["uv", "run", "replay", "init", "--path", "/nonexistent/path"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_init_force_clears_existing_data(self, temp_git_repo: Path):
        """replay init --force should clear existing data and reset document count to 0."""
        data_dir = temp_git_repo / ".replay_force_test"

        # First, initialize the database
        result = subprocess.run(
            ["uv", "run", "replay", "--data-dir", str(data_dir), "init", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Initialized" in result.stdout

        # Add a document
        result = subprocess.run(
            ["uv", "run", "replay", "--data-dir", str(data_dir), "add", "--text", "test document"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Verify document count is 1
        result = subprocess.run(
            ["uv", "run", "replay", "--data-dir", str(data_dir), "stats"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Documents: 1" in result.stdout

        # Now reinitialize with --force
        result = subprocess.run(
            ["uv", "run", "replay", "--data-dir", str(data_dir), "init", "--force", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Reinitialized" in result.stdout

        # Verify document count is now 0 (data was cleared)
        result = subprocess.run(
            ["uv", "run", "replay", "--data-dir", str(data_dir), "stats"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Documents: 0" in result.stdout

    def test_stats_exits_nonzero_when_not_initialized(self):
        """replay stats should exit with non-zero when not initialized."""
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["uv", "run", "replay", "--data-dir", f"{tmp}/.replay", "stats"],
                capture_output=True,
                text=True,
            )
            assert result.returncode != 0
