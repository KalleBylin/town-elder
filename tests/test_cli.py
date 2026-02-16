"""CLI integration tests for Town Elder."""
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
        subprocess.run(["git", "init"], cwd=repo_path, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        # Create initial commit
        (repo_path / "test.txt").write_text("initial content")
        subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "initial commit"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        yield repo_path


class TestHookInstall:
    """Tests for hook install functionality."""

    def test_hook_install_preserves_data_dir(self, temp_git_repo: Path):
        """Hook install should preserve --data-dir in generated hook script."""
        custom_data_dir = temp_git_repo / ".custom-town-elder"

        # Initialize with custom data-dir
        result = subprocess.run(
            [
                "uv", "run", "te",
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
                "uv", "run", "te",
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
            ["uv", "run", "te", "init", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Install hook without data-dir
        result = subprocess.run(
            ["uv", "run", "te", "hook", "install", "--repo", str(temp_git_repo)],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Verify the hook exists and is executable
        hook_path = temp_git_repo / ".git" / "hooks" / "post-commit"
        assert hook_path.exists()
        assert hook_path.stat().st_mode & 0o111  # executable


class TestHookLauncherPortability:
    """Tests for hook launcher portability (uv/uvx/te/python fallback)."""

    def test_hook_install_includes_fallback_chain(self, temp_git_repo: Path):
        """Hook install should generate fallback chain: uv -> uvx -> te -> python -m town_elder."""
        # Initialize with default location
        result = subprocess.run(
            ["uv", "run", "te", "init", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Install hook
        result = subprocess.run(
            ["uv", "run", "te", "hook", "install", "--repo", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Verify the hook contains the fallback chain
        hook_path = temp_git_repo / ".git" / "hooks" / "post-commit"
        hook_content = hook_path.read_text()

        # Check for uv run te
        assert "uv run te" in hook_content
        # Check for uvx fallback
        assert "uvx --from town-elder te" in hook_content
        # Check for te fallback
        assert "command -v te >/dev/null 2>&1 && te" in hook_content
        # Check for python -m town_elder fallback
        assert "python -m town_elder" in hook_content

    def test_hook_install_fallback_chain_with_data_dir(self, temp_git_repo: Path):
        """Hook install should include fallback chain even with custom data-dir."""
        custom_data_dir = temp_git_repo / ".custom-town-elder"

        # Initialize with custom data-dir
        result = subprocess.run(
            [
                "uv", "run", "te",
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
                "uv", "run", "te",
                "--data-dir", str(custom_data_dir),
                "hook", "install",
                "--repo", str(temp_git_repo),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Verify the hook contains the fallback chain and data-dir
        hook_path = temp_git_repo / ".git" / "hooks" / "post-commit"
        hook_content = hook_path.read_text()

        # Check for fallback chain
        assert "uv run te" in hook_content
        assert "uvx --from town-elder te" in hook_content
        assert "command -v te >/dev/null 2>&1 && te" in hook_content
        assert "python -m town_elder" in hook_content
        # Check for data-dir
        assert "--data-dir" in hook_content

    def test_hook_fallback_order_correct(self, temp_git_repo: Path):
        """Hook should try uv first, then uvx, then te, then python -m town_elder."""
        # Initialize
        result = subprocess.run(
            ["uv", "run", "te", "init", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Install hook
        result = subprocess.run(
            ["uv", "run", "te", "hook", "install", "--repo", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        hook_path = temp_git_repo / ".git" / "hooks" / "post-commit"
        hook_content = hook_path.read_text()

        # Find positions of each command in the hook (search for command prefixes).
        # Hooks always include --data-dir between "te" and "index commits".
        uv_pos = hook_content.find("command -v uv >/dev/null 2>&1 && uv run te ")
        uvx_pos = hook_content.find("command -v uvx >/dev/null 2>&1 && uvx --from town-elder te ")
        te_pos = hook_content.find("command -v te >/dev/null 2>&1 && te ")
        python_pos = hook_content.find("python -m town_elder ")

        assert "uv run te" in hook_content and "index commits --repo" in hook_content
        assert "uvx --from town-elder te" in hook_content and "index commits --repo" in hook_content
        assert "command -v te >/dev/null 2>&1 && te " in hook_content and "index commits --repo" in hook_content
        assert "python -m town_elder " in hook_content and "index commits --repo" in hook_content

        # Verify all commands are present
        assert uv_pos != -1, "uv run te not found in hook"
        assert uvx_pos != -1, "uvx --from town-elder te not found in hook"
        assert te_pos != -1, "te fallback not found in hook"
        assert python_pos != -1, "python -m town_elder fallback not found in hook"

        # Verify order: uv -> uvx -> te -> python -m
        assert uv_pos < uvx_pos < te_pos < python_pos, "Fallback order should be: uv -> uvx -> te -> python -m"


class TestHookUninstall:
    """Tests for hook uninstall safety."""

    def test_uninstall_refuses_non_replay_hook(self, temp_git_repo: Path):
        """Hook uninstall should refuse to delete non-Town Elder hooks by default."""
        hook_path = temp_git_repo / ".git" / "hooks" / "post-commit"
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        hook_path.write_text("#!/bin/bash\necho 'custom hook'")
        hook_path.chmod(0o755)

        result = subprocess.run(
            ["uv", "run", "te", "hook", "uninstall", "--repo", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "not a Town Elder hook" in result.stderr or "not a Town Elder hook" in result.stdout

    def test_uninstall_allows_force_on_non_replay_hook(self, temp_git_repo: Path):
        """Hook uninstall should allow deletion with --force."""
        hook_path = temp_git_repo / ".git" / "hooks" / "post-commit"
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        hook_path.write_text("#!/bin/bash\necho 'custom hook'")
        hook_path.chmod(0o755)

        result = subprocess.run(
            ["uv", "run", "te", "hook", "uninstall", "--repo", str(temp_git_repo), "--force"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert not hook_path.exists()

    def test_uninstall_removes_replay_hook(self, temp_git_repo: Path):
        """Hook uninstall should remove Town Elder hooks without --force."""
        # Initialize and install Town Elder hook first
        init_result = subprocess.run(
            ["uv", "run", "te", "init", "--path", str(temp_git_repo)],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
        )
        assert init_result.returncode == 0

        install_result = subprocess.run(
            ["uv", "run", "te", "hook", "install", "--repo", str(temp_git_repo)],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
        )
        assert install_result.returncode == 0

        hook_path = temp_git_repo / ".git" / "hooks" / "post-commit"
        assert hook_path.exists()

        result = subprocess.run(
            ["uv", "run", "te", "hook", "uninstall", "--repo", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert not hook_path.exists()


class TestPathSemantics:
    """Tests for path/data-dir semantics."""

    def test_data_dir_option(self, temp_git_repo: Path):
        """--data-dir option should work across commands."""
        data_dir = temp_git_repo / ".town_elder"

        # Initialize with explicit path
        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "init", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert data_dir.exists()

        # Stats should use the same data dir
        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "stats"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        output = result.stdout + result.stderr
        normalized_output = output.replace("\n", "")
        assert str(data_dir) in normalized_output

    def test_data_dir_isolation_from_default_store(self, temp_git_repo: Path):
        """--data-dir should isolate reads/writes from the default cwd store."""
        root_marker = "ROOT_MARKER_12345"
        custom_marker = "CUSTOM_MARKER_67890"
        custom_data_dir = temp_git_repo / ".custom-town-elder"

        # Seed the default cwd store.
        subprocess.run(
            ["uv", "run", "te", "init", "--path", str(temp_git_repo)],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            ["uv", "run", "te", "add", "--text", root_marker],
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
                "te",
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
                "te",
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
                "te",
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
            ["uv", "run", "te", "init", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            ["uv", "run", "te", "add", "--text", "hello world"],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
            check=True,
        )

        # Export to stdout WITHOUT --format option (the bug case)
        result = subprocess.run(
            ["uv", "run", "te", "export", "--output", "-"],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Export failed: {result.stderr}"
        output = result.stdout + result.stderr
        # Should contain valid JSON output
        assert "hello world" in output

    def test_export_to_stdout_preserves_raw_content(self, temp_git_repo: Path):
        """Export to stdout should preserve raw content without Rich markup interpretation."""
        # Initialize and add data with bracket-like text
        subprocess.run(
            ["uv", "run", "te", "init", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
            check=True,
        )
        # Add text containing Rich-like markup tags
        subprocess.run(
            ["uv", "run", "te", "add", "--text", "contains [brackets] and [bold]tags[/bold]"],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
            check=True,
        )

        # Export to stdout
        result = subprocess.run(
            ["uv", "run", "te", "export", "--output", "-", "--format", "json"],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Export failed: {result.stderr}"
        # The raw bracket content must be preserved in output
        assert "[brackets]" in result.stdout
        assert "[bold]" in result.stdout
        assert "[/bold]" in result.stdout
        # Verify it's valid JSON that can be parsed
        import json
        data = json.loads(result.stdout)
        assert len(data) == 1
        assert data[0]["text"] == "contains [brackets] and [bold]tags[/bold]"


class TestModuleEntrypoints:
    """Tests for python -m module entrypoints."""

    def test_python_m_replay_cli_runs(self):
        """python -m town_elder.cli should be executable."""
        result = subprocess.run(
            ["uv", "run", "python", "-m", "town_elder.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Usage:" in result.stdout
        assert "town_elder.cli" in result.stdout

    def test_python_m_replay_runs(self):
        """python -m town_elder should be executable."""
        result = subprocess.run(
            ["uv", "run", "python", "-m", "town_elder", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Usage:" in result.stdout
        assert "python -m town_elder" in result.stdout


class TestMetadataValidation:
    """Tests for metadata validation."""

    def test_add_rejects_invalid_json_metadata(self, temp_git_repo: Path):
        """te add should reject invalid JSON metadata."""
        data_dir = temp_git_repo / ".town_elder"
        subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "init", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
            check=True,
        )

        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "add", "--text", "test", "--metadata", "{invalid json"],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "JSON" in result.stderr or "JSON" in result.stdout

    def test_add_accepts_valid_json_metadata(self, temp_git_repo: Path):
        """te add should accept valid JSON metadata."""
        data_dir = temp_git_repo / ".town_elder"
        subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "init", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
            check=True,
        )

        metadata = json.dumps({"source": "test", "type": "example"})
        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "add", "--text", "test document", "--metadata", metadata],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

    def test_add_rejects_non_object_json_metadata(self, temp_git_repo: Path):
        """te add should reject JSON metadata that is not an object."""
        data_dir = temp_git_repo / ".town_elder"
        subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "init", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
            check=True,
        )

        # Test with a JSON array (list)
        metadata = json.dumps(["item1", "item2"])
        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "add", "--text", "test", "--metadata", metadata],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "object" in result.stderr.lower() or "dict" in result.stderr.lower()

    def test_add_rejects_json_string_metadata(self, temp_git_repo: Path):
        """te add should reject JSON metadata that is a string."""
        data_dir = temp_git_repo / ".town_elder"
        subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "init", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
            check=True,
        )

        # Test with a JSON string
        metadata = json.dumps("just a string")
        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "add", "--text", "test", "--metadata", metadata],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "object" in result.stderr.lower() or "dict" in result.stderr.lower()


class TestExitCodes:
    """Tests for exit codes."""

    def test_init_exits_zero_on_success(self, temp_git_repo: Path):
        """te init should exit with 0 on success."""
        data_dir = temp_git_repo / ".town_elder_new"
        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "init", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_init_exits_nonzero_on_error(self):
        """te init should exit with non-zero on error."""
        result = subprocess.run(
            ["uv", "run", "te", "init", "--path", "/nonexistent/path"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_init_force_clears_existing_data(self, temp_git_repo: Path):
        """te init --force should clear existing data and reset document count to 0."""
        data_dir = temp_git_repo / ".town_elder_custom_force"

        # First, initialize the database
        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "init", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Initialized" in result.stdout

        # Add a document
        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "add", "--text", "test document"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Verify document count is 1
        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "stats"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Documents: 1" in result.stdout

        # Now reinitialize with --force
        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "init", "--force", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Reinitialized" in result.stdout

        # Verify document count is now 0 (data was cleared)
        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "stats"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Documents: 0" in result.stdout

    def test_stats_exits_nonzero_when_not_initialized(self):
        """te stats should exit with non-zero when not initialized."""
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["uv", "run", "te", "--data-dir", f"{tmp}/.town_elder", "stats"],
                capture_output=True,
                text=True,
            )
            assert result.returncode != 0


class TestInitErrorHandling:
    """Tests for init command error handling."""

    def test_init_fails_for_nonexistent_path(self):
        """te init should fail when path doesn't exist."""
        result = subprocess.run(
            ["uv", "run", "te", "init", "--path", "/nonexistent/path/12345"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "does not exist" in result.stderr or "does not exist" in result.stdout

    def test_init_fails_for_file_not_directory(self, temp_git_repo: Path):
        """te init should fail when path is a file, not a directory."""
        file_path = temp_git_repo / "file.txt"
        file_path.write_text("not a directory")

        result = subprocess.run(
            ["uv", "run", "te", "init", "--path", str(file_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "not a directory" in result.stderr or "not a directory" in result.stdout

    def test_init_force_refuses_unsafe_data_dir(self, temp_git_repo: Path):
        """te init --force should refuse unsafe data-dir paths."""
        # First initialize with a safe path
        safe_data_dir = temp_git_repo / ".town_elder"
        subprocess.run(
            ["uv", "run", "te", "--data-dir", str(safe_data_dir), "init", "--path", str(temp_git_repo)],
            capture_output=True,
            check=True,
        )

        # Now try to reinitialize with --force using /tmp as data-dir (should be rejected as unsafe)
        # Note: --data-dir must come BEFORE the subcommand
        unsafe_path = Path("/") / "tmp" / "unsafe_town_elder_test_12345"

        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(unsafe_path), "init", "--force", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )
        # Should fail with safety error (not just "already exists")
        assert result.returncode != 0
        output = result.stderr + result.stdout
        assert "unsafe" in output.lower() or "safety" in output.lower() or "refusing" in output.lower()

    def test_init_force_refuses_arbitrary_hidden_dirs(self, temp_git_repo: Path):
        """te init --force should refuse arbitrary hidden directories like .git."""
        # First initialize with .town_elder
        safe_data_dir = temp_git_repo / ".town_elder"
        subprocess.run(
            ["uv", "run", "te", "--data-dir", str(safe_data_dir), "init", "--path", str(temp_git_repo)],
            capture_output=True,
            check=True,
        )

        # Create a .git directory to simulate the scenario
        git_dir = temp_git_repo / ".git"
        git_dir.mkdir(exist_ok=True)
        (git_dir / "config").write_text("")
        (git_dir / "HEAD").write_text("ref: refs/heads/main")

        # Try to use .git as data-dir with --force (should be rejected as unsafe)
        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(git_dir), "init", "--force", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )
        # Should fail with safety error
        assert result.returncode != 0
        output = result.stderr + result.stdout
        assert "unsafe" in output.lower() or "safety" in output.lower() or "refusing" in output.lower()

        # Verify .git was NOT deleted
        assert git_dir.exists()
        assert (git_dir / "config").exists()

    def test_init_with_force_clears_data(self, temp_git_repo: Path):
        """te init --force should clear existing data."""
        data_dir = temp_git_repo / ".town_elder"

        # First init
        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "init", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Reinit with force
        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "init", "--force", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Reinitialized" in result.stdout

    def test_init_help_flag_does_not_initialize(self, temp_git_repo: Path):
        """te init -h should show help and NOT initialize storage."""
        data_dir = temp_git_repo / ".town_elder_test_help"

        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "init", "-h"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Usage:" in result.stdout or "usage:" in result.stdout.lower()
        # Ensure no .town_elder directory was created
        assert not data_dir.exists()

    def test_init_install_hook_flag_works(self, temp_git_repo: Path):
        """te init --install-hook should install hook when requested."""
        data_dir = temp_git_repo / ".town_elder_hook"

        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "init", "--install-hook", "--path", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert data_dir.exists()


class TestIndexErrorHandling:
    """Tests for index command error handling."""

    def test_index_fails_when_not_initialized(self):
        """te index files should fail when not initialized."""
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["uv", "run", "te", "--data-dir", f"{tmp}/.town_elder", "index", "files", "."],
                capture_output=True,
                text=True,
            )
            assert result.returncode != 0
            output = f"{result.stdout}\n{result.stderr}".lower()
            assert "not initialized" in output or "--data-dir does not exist" in output

    def test_index_fails_for_nonexistent_path(self, temp_git_repo: Path):
        """te index files should fail for nonexistent path."""
        data_dir = temp_git_repo / ".town_elder"

        subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "init", "--path", str(temp_git_repo)],
            capture_output=True,
            check=True,
        )

        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "index", "files", "/nonexistent/path"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "does not exist" in result.stderr or "does not exist" in result.stdout

    def test_index_fails_for_file_path(self, temp_git_repo: Path):
        """te index files should fail when given a file path instead of directory."""
        data_dir = temp_git_repo / ".town_elder"

        subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "init", "--path", str(temp_git_repo)],
            capture_output=True,
            check=True,
        )

        file_path = temp_git_repo / "test.py"
        file_path.write_text("print('hello')")

        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "index", "files", str(file_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "not a directory" in result.stderr or "not a directory" in result.stdout


class TestCommitIndexErrorHandling:
    """Tests for index commits command error handling."""

    def test_commit_index_fails_when_not_initialized(self):
        """te index commits should fail when not initialized."""
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["uv", "run", "te", "--data-dir", f"{tmp}/.town_elder", "index", "commits", "--repo", tmp],
                capture_output=True,
                text=True,
            )
            assert result.returncode != 0
            output = f"{result.stdout}\n{result.stderr}".lower()
            assert "not initialized" in output or "--data-dir does not exist" in output

    def test_commit_index_fails_for_non_git_repo(self, temp_git_repo: Path):
        """te index commits should fail for non-git repository."""
        # Remove .git directory
        import shutil
        shutil.rmtree(temp_git_repo / ".git")

        data_dir = temp_git_repo / ".town_elder"

        # Need to create the directory manually since there's no git
        data_dir.mkdir()

        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "index", "commits", "--repo", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "not a git repository" in result.stderr.lower() or "not a git repository" in result.stdout.lower()


class TestExportErrorHandling:
    """Tests for export command error handling."""

    def test_export_fails_when_not_initialized(self):
        """te export should fail when not initialized."""
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["uv", "run", "te", "--data-dir", f"{tmp}/.town_elder", "export"],
                capture_output=True,
                text=True,
            )
            assert result.returncode != 0
            output = f"{result.stdout}\n{result.stderr}".lower()
            assert "not initialized" in output or "--data-dir does not exist" in output

    def test_export_fails_with_invalid_format(self, temp_git_repo: Path):
        """te export should fail with invalid format."""
        data_dir = temp_git_repo / ".town_elder"

        subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "init", "--path", str(temp_git_repo)],
            capture_output=True,
            check=True,
        )

        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "export", "--format", "invalid_format"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "invalid" in result.stderr.lower() or "invalid" in result.stdout.lower()


class TestSearchErrorHandling:
    """Tests for search command error handling."""

    def test_search_fails_when_not_initialized(self):
        """te search should fail when not initialized."""
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["uv", "run", "te", "--data-dir", f"{tmp}/.town_elder", "search", "test"],
                capture_output=True,
                text=True,
            )
            assert result.returncode != 0
            output = f"{result.stdout}\n{result.stderr}".lower()
            assert "not initialized" in output or "--data-dir does not exist" in output

    def test_search_rejects_zero_top_k(self, temp_git_repo: Path):
        """te search should reject --top-k=0."""
        data_dir = temp_git_repo / ".town_elder"
        data_dir.mkdir()

        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "search", "test", "--top-k", "0"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        output = f"{result.stdout}\n{result.stderr}".lower()
        assert "positive integer" in output

    def test_search_rejects_negative_top_k(self, temp_git_repo: Path):
        """te search should reject negative --top-k values."""
        data_dir = temp_git_repo / ".town_elder"
        data_dir.mkdir()

        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "search", "test", "--top-k", "-1"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        output = f"{result.stdout}\n{result.stderr}".lower()
        assert "positive integer" in output


class TestHookStatus:
    """Tests for hook status command."""

    def test_hook_status_not_installed(self, temp_git_repo: Path):
        """te hook status should show not installed when no hook exists."""
        result = subprocess.run(
            ["uv", "run", "te", "hook", "status", "--repo", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "not installed" in result.stdout.lower() or "not installed" in result.stderr.lower()

    def test_hook_status_installed(self, temp_git_repo: Path):
        """te hook status should show installed when hook exists."""
        data_dir = temp_git_repo / ".town_elder"

        # Initialize and install hook with explicit data-dir
        subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "init", "--path", str(temp_git_repo)],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "hook", "install", "--repo", str(temp_git_repo)],
            capture_output=True,
            check=True,
        )

        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "hook", "status", "--repo", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "installed" in result.stdout.lower()

    def test_hook_status_fails_for_non_git_repo(self):
        """te hook status should fail for non-git repository."""
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["uv", "run", "te", "hook", "status", "--repo", tmp],
                capture_output=True,
                text=True,
            )
            assert result.returncode != 0
            assert "not a git repository" in result.stderr.lower() or "not a git repository" in result.stdout.lower()

    def test_hook_status_handles_non_utf8_hook(self, temp_git_repo: Path):
        """te hook status should handle non-UTF8 hook files without crashing."""
        hook_path = temp_git_repo / ".git" / "hooks" / "post-commit"
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        # Write binary content that is not valid UTF-8
        hook_path.write_bytes(b"\x00\x01\x02\xff\xfe\xfd")

        result = subprocess.run(
            ["uv", "run", "te", "hook", "status", "--repo", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Unknown" in result.stdout


class TestHookNonUtf8:
    """Tests for handling non-UTF8 hook files."""

    def test_hook_install_handles_non_utf8_existing_hook(self, temp_git_repo: Path):
        """te hook install should handle non-UTF8 existing hook without crashing."""
        # Initialize the repo first so .town_elder exists
        subprocess.run(
            ["uv", "run", "te", "init", "--path", str(temp_git_repo)],
            capture_output=True,
            check=True,
        )

        hook_path = temp_git_repo / ".git" / "hooks" / "post-commit"
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        # Write binary content that is not valid UTF-8
        hook_path.write_bytes(b"\x00\x01\x02\xff\xfe\xfd")

        result = subprocess.run(
            ["uv", "run", "te", "hook", "install", "--repo", str(temp_git_repo), "--force"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "not a town elder hook" in result.stderr.lower() or "not a town elder hook" in result.stdout.lower()

    def test_hook_uninstall_handles_non_utf8_hook(self, temp_git_repo: Path):
        """te hook uninstall should handle non-UTF8 hook without crashing."""
        hook_path = temp_git_repo / ".git" / "hooks" / "post-commit"
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        # Write binary content that is not valid UTF-8
        hook_path.write_bytes(b"\x00\x01\x02\xff\xfe\xfd")

        result = subprocess.run(
            ["uv", "run", "te", "hook", "uninstall", "--repo", str(temp_git_repo)],
            capture_output=True,
            text=True,
        )
        # Should refuse to delete without --force
        assert result.returncode == 1
        assert "not a town elder hook" in result.stderr.lower()

    def test_hook_uninstall_force_removes_non_utf8_hook(self, temp_git_repo: Path):
        """te hook uninstall --force should remove non-UTF8 hook."""
        hook_path = temp_git_repo / ".git" / "hooks" / "post-commit"
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        # Write binary content that is not valid UTF-8
        hook_path.write_bytes(b"\x00\x01\x02\xff\xfe\xfd")

        result = subprocess.run(
            ["uv", "run", "te", "hook", "uninstall", "--repo", str(temp_git_repo), "--force"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert not hook_path.exists()


class TestAddErrorHandling:
    """Tests for add command error handling."""

    def test_add_fails_when_not_initialized(self):
        """te add should fail when not initialized."""
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["uv", "run", "te", "--data-dir", f"{tmp}/.town_elder", "add", "--text", "test"],
                capture_output=True,
                text=True,
            )
            assert result.returncode != 0
            output = f"{result.stdout}\n{result.stderr}".lower()
            assert "not initialized" in output or "--data-dir does not exist" in output

    def test_add_requires_text_option(self, temp_git_repo: Path):
        """te add should require --text option."""
        data_dir = temp_git_repo / ".town_elder"

        subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "init", "--path", str(temp_git_repo)],
            capture_output=True,
            check=True,
        )

        result = subprocess.run(
            ["uv", "run", "te", "--data-dir", str(data_dir), "add"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
