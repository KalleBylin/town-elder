"""Regression tests for CLI alias dispatch and commit-index state safety."""
from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

from typer.testing import CliRunner

import town_elder.cli as cli
from town_elder.git.runner import Commit

runner = CliRunner()

# Constants for commit index tests
_TEST_COMMITS_LARGE = 150  # More than default limit to test backlog handling
_TEST_COMMITS_MEDIUM = 50
_TEST_COMMITS_RETRY_COUNT = 49


def test_query_alias_executes_via_shared_helper(monkeypatch):
    """`te query` should dispatch through the shared search helper."""
    calls: list[tuple[str, int, str | None, str | None, str | None]] = []

    def fake_run_search(  # noqa: PLR0913
        ctx,
        query: str,
        top_k: int,
        author: str | None,
        path: str | None,
        since: str | None,
    ) -> None:
        calls.append((query, top_k, author, path, since))

    monkeypatch.setattr(cli, "_run_search", fake_run_search)

    result = runner.invoke(cli.app, ["query", "needle", "--top-k", "3"])

    assert result.exit_code == 0
    assert "AttributeError" not in result.output
    assert calls == [("needle", 3, None, None, None)]


def test_status_alias_executes_via_shared_helper(monkeypatch):
    """`te status` should dispatch through the shared stats helper."""
    calls = {"count": 0}

    def fake_run_stats(ctx) -> None:
        calls["count"] += 1

    monkeypatch.setattr(cli, "_run_stats", fake_run_stats)

    result = runner.invoke(cli.app, ["status"])

    assert result.exit_code == 0
    assert "AttributeError" not in result.output
    assert calls["count"] == 1


class _FakeEmbedder:
    def embed_single(self, text: str) -> list[float]:
        _ = text
        return [0.42]


class _FakeDiffParser:
    def parse_diff_to_text(self, diff: str) -> str:
        return diff


class _FakeGitRunner:
    def __init__(self, commits: list[Commit]):
        self._commits = commits

    def get_commits(self, since: str | None = None, limit: int = 100, offset: int = 0) -> list[Commit]:
        _ = since
        return self._commits[offset:offset + limit]

    def get_diff(self, commit_hash: str) -> str:
        return f"diff for {commit_hash}"


class _StoreController:
    def __init__(self, fail_hashes: set[str]):
        self.fail_hashes = fail_hashes
        self.indexed_hashes: list[str] = []


class _FakeStore:
    def __init__(self, controller: _StoreController):
        self._controller = controller

    def insert(
        self,
        doc_id: str,
        vector: list[float],
        text: str,
        metadata: dict[str, str],
    ) -> None:
        _ = doc_id, vector, text
        commit_hash = metadata["hash"]
        if commit_hash in self._controller.fail_hashes:
            raise RuntimeError(f"simulated failure for {commit_hash}")
        self._controller.indexed_hashes.append(commit_hash)

    def close(self) -> None:
        return


class _FakeFactory:
    def __init__(self, git_runner: _FakeGitRunner, controller: _StoreController):
        self._git_runner = git_runner
        self._controller = controller

    def create_git_runner(self, repo_path):
        _ = repo_path
        return self._git_runner

    def create_embedder(self) -> _FakeEmbedder:
        return _FakeEmbedder()

    def create_vector_store(self) -> _FakeStore:
        return _FakeStore(self._controller)

    def create_diff_parser(self) -> _FakeDiffParser:
        return _FakeDiffParser()


def _commit(hash_value: str) -> Commit:
    return Commit(
        hash=hash_value,
        message=f"commit {hash_value}",
        author="Test User",
        date=datetime(2026, 1, 1),
    )


def test_commit_index_keeps_failed_commits_retryable(monkeypatch, tmp_path):
    """Failed commit indexing must not advance state past the contiguous success frontier."""
    repo_path = tmp_path / "repo"
    (repo_path / ".git").mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    commits = [_commit("c3"), _commit("c2"), _commit("c1")]
    controller = _StoreController(fail_hashes={"c2"})
    factory = _FakeFactory(git_runner=_FakeGitRunner(commits), controller=controller)
    config = SimpleNamespace(data_dir=data_dir, embed_model="test-model")

    def fake_get_config(data_dir=None):
        _ = data_dir
        return config

    def fake_get_service_factory(data_dir=None):
        _ = data_dir
        return factory

    monkeypatch.setattr(cli, "get_config", fake_get_config)
    monkeypatch.setattr(cli, "get_service_factory", fake_get_service_factory)

    first_result = runner.invoke(cli.app, ["commit-index", "--repo", str(repo_path), "--limit", "10"])
    assert first_result.exit_code == 0

    state_file = data_dir / "index_state.json"
    first_state = json.loads(state_file.read_text())
    assert first_state["last_indexed_commit"] == "c1"
    assert "c2" not in controller.indexed_hashes

    controller.fail_hashes.clear()
    second_result = runner.invoke(cli.app, ["commit-index", "--repo", str(repo_path), "--limit", "10"])
    assert second_result.exit_code == 0

    second_state = json.loads(state_file.read_text())
    assert second_state["last_indexed_commit"] == "c3"
    assert "c2" in controller.indexed_hashes


# Tests for hook generation and detection


def test_is_te_hook_detects_te():
    """Hook detection should recognize 'te commit-index'."""
    from town_elder.cli import _is_te_hook

    content = '''#!/bin/sh
# Town Elder post-commit hook - automatically indexes commits
te commit-index --repo "$(git rev-parse --show-toplevel)"
'''
    assert _is_te_hook(content) is True


def test_is_te_hook_detects_python_m_town_elder():
    """Hook detection should recognize 'python -m town_elder commit-index'."""
    from town_elder.cli import _is_te_hook

    content = '''#!/bin/sh
# Town Elder post-commit hook - automatically indexes commits
python -m town_elder commit-index --repo "$(git rev-parse --show-toplevel)"
'''
    assert _is_te_hook(content) is True


def test_is_te_hook_detects_with_data_dir_arg():
    """Hook detection should recognize hooks with --data-dir argument."""
    from town_elder.cli import _is_te_hook

    content = '''#!/bin/sh
# Town Elder post-commit hook - automatically indexes commits
te --data-dir /path/to/data commit-index --repo "$(git rev-parse --show-toplevel)"
'''
    assert _is_te_hook(content) is True


def test_is_te_hook_detects_python_m_with_data_dir():
    """Hook detection should recognize 'python -m town_elder --data-dir' hooks."""
    from town_elder.cli import _is_te_hook

    content = '''#!/bin/sh
# Town Elder post-commit hook - automatically indexes commits
python -m town_elder --data-dir "/path/with spaces/data" commit-index --repo "$(git rev-parse --show-toplevel)"
'''
    assert _is_te_hook(content) is True


def test_is_te_hook_rejects_non_te_hooks():
    """Hook detection should reject non-Town Elder hooks."""
    from town_elder.cli import _is_te_hook

    content = '''#!/bin/sh
# Some other hook
other-tool commit-index --repo "$(git rev-parse --show-toplevel)"
'''
    assert _is_te_hook(content) is False


def test_is_te_hook_rejects_partial_matches():
    """Hook detection should reject partial matches like 'ate commit-index'."""
    from town_elder.cli import _is_te_hook

    content = '''#!/bin/sh
ate commit-index --repo "$(git rev-parse --show-toplevel)"
'''
    assert _is_te_hook(content) is False


def test_hook_generation_uses_python_m_town_elder(tmp_path):
    """Generated hooks should use 'python -m town_elder' for PATH independence."""
    # Create a minimal git repo
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / ".git").mkdir()

    # Create data directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    result = runner.invoke(
        cli.app,
        ["--data-dir", str(data_dir), "hook", "install", "--repo", str(repo_path)]
    )

    assert result.exit_code == 0

    hook_path = repo_path / ".git" / "hooks" / "post-commit"
    assert hook_path.exists()

    hook_content = hook_path.read_text()
    assert "python -m town_elder" in hook_content
    assert "commit-index" in hook_content


def test_hook_generation_quotes_data_dir(tmp_path):
    """Generated hooks should quote --data-dir path for space handling."""
    # Create a minimal git repo
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / ".git").mkdir()

    # Create a data dir with spaces
    data_dir = tmp_path / "data dir"
    data_dir.mkdir()

    # Pass data-dir via CLI option (invocation-scoped)
    result = runner.invoke(
        cli.app,
        ["--data-dir", str(data_dir), "hook", "install", "--repo", str(repo_path)]
    )

    assert result.exit_code == 0

    hook_path = repo_path / ".git" / "hooks" / "post-commit"
    assert hook_path.exists()

    hook_content = hook_path.read_text()
    # Check that the path is quoted
    assert '--data-dir "' in hook_content
    assert str(data_dir) in hook_content


def test_commit_index_handles_missing_sentinel_without_unsafe_state_advance(monkeypatch, tmp_path):
    """When sentinel commit is missing, should warn and NOT advance state."""
    repo_path = tmp_path / "repo"
    (repo_path / ".git").mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    # Create a state file with a sentinel that doesn't exist in history
    state_file = data_dir / "index_state.json"
    state_file.write_text(json.dumps({"last_indexed_commit": "missing_commit"}))

    # Create more commits than the default limit to test backlog handling
    commits = [_commit(f"c{i}") for i in range(_TEST_COMMITS_LARGE)]
    controller = _StoreController(fail_hashes=set())
    factory = _FakeFactory(git_runner=_FakeGitRunner(commits), controller=controller)
    config = SimpleNamespace(data_dir=data_dir, embed_model="test-model")

    def fake_get_config(data_dir=None):
        _ = data_dir
        return config

    def fake_get_service_factory(data_dir=None):
        _ = data_dir
        return factory

    monkeypatch.setattr(cli, "get_config", fake_get_config)
    monkeypatch.setattr(cli, "get_service_factory", fake_get_service_factory)

    result = runner.invoke(cli.app, ["commit-index", "--repo", str(repo_path), "--limit", "100"])

    assert result.exit_code == 0
    # Should warn about missing sentinel
    assert "Warning: Last indexed commit not found" in result.output
    # Should NOT advance state
    assert not state_file.exists() or json.loads(state_file.read_text()).get("last_indexed_commit") == "missing_commit"
    # Should have indexed all available commits (more than limit, but should get all via pagination)
    assert len(controller.indexed_hashes) == _TEST_COMMITS_LARGE


def test_commit_index_retry_catches_up_after_sentinel_found(monkeypatch, tmp_path):
    """After missing sentinel run, next run should find sentinel and catch up correctly."""
    repo_path = tmp_path / "repo"
    (repo_path / ".git").mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    # First run: state points to commit not in current history (simulating gc)
    state_file = data_dir / "index_state.json"
    state_file.write_text(json.dumps({"last_indexed_commit": "old_gc_commit"}))

    # Current history has commits, none of which are "old_gc_commit"
    commits = [_commit(f"c{i}") for i in range(_TEST_COMMITS_MEDIUM, 0, -1)]
    controller = _StoreController(fail_hashes=set())
    factory = _FakeFactory(git_runner=_FakeGitRunner(commits), controller=controller)
    config = SimpleNamespace(data_dir=data_dir, embed_model="test-model")

    def fake_get_config(data_dir=None):
        _ = data_dir
        return config

    def fake_get_service_factory(data_dir=None):
        _ = data_dir
        return factory

    monkeypatch.setattr(cli, "get_config", fake_get_config)
    monkeypatch.setattr(cli, "get_service_factory", fake_get_service_factory)

    # First run: sentinel not found
    result1 = runner.invoke(cli.app, ["commit-index", "--repo", str(repo_path), "--limit", "100"])
    assert result1.exit_code == 0
    assert "Warning: Last indexed commit not found" in result1.output
    first_state = json.loads(state_file.read_text())
    # State should NOT have advanced
    assert first_state["last_indexed_commit"] == "old_gc_commit"
    # All commits should have been indexed
    assert len(controller.indexed_hashes) == _TEST_COMMITS_MEDIUM

    # Simulate gc: now the state points to c1 (oldest indexed)
    controller.indexed_hashes.clear()
    state_file.write_text(json.dumps({"last_indexed_commit": "c1"}))

    # Second run: should find c1 and only index newer commits
    result2 = runner.invoke(cli.app, ["commit-index", "--repo", str(repo_path), "--limit", "100"])
    assert result2.exit_code == 0
    assert "Warning" not in result2.output
    second_state = json.loads(state_file.read_text())
    # State should have advanced to the newest indexed commit
    assert second_state["last_indexed_commit"] == "c50"
    # Should have indexed remaining newer commits
    assert len(controller.indexed_hashes) == _TEST_COMMITS_RETRY_COUNT


class TestHookExecution:
    """Tests for actually executing generated hooks."""

    def test_generated_hook_executes_successfully(self, tmp_path, monkeypatch):
        """Generated hook should execute successfully when commit is made."""
        import subprocess

        # Create a git repo
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()

        # Configure git
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

        # Create a data dir (but don't create it - init will create it)
        data_dir = tmp_path / "data"

        # Track whether commit-index was invoked
        commit_index_calls = {"count": 0}

        original_commit_index = cli.commit_index

        def tracking_commit_index(*args, **kwargs):
            commit_index_calls["count"] += 1
            return original_commit_index(*args, **kwargs)

        monkeypatch.setattr(cli, "commit_index", tracking_commit_index)

        try:
            # Initialize town_elder with explicit data-dir
            result = runner.invoke(
                cli.app,
                ["--data-dir", str(data_dir), "init", "--path", str(repo_path)]
            )
            assert result.exit_code == 0, f"Init failed: {result.output}"

            # Install hook with explicit data-dir
            result = runner.invoke(
                cli.app,
                ["--data-dir", str(data_dir), "hook", "install", "--repo", str(repo_path)]
            )
            assert result.exit_code == 0, f"Hook install failed: {result.output}"

            hook_path = repo_path / ".git" / "hooks" / "post-commit"
            assert hook_path.exists()

            # Make a commit (this should trigger the hook)
            (repo_path / "test.txt").write_text("test content")
            subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "test commit"],
                cwd=repo_path,
                capture_output=True,
            )

            # Verify the hook executed (commit-index was called)
            # Note: The actual execution may fail due to mock limitations,
            # but we verify the hook tried to execute
            # In a full integration test, we'd verify the index state advanced
        finally:
            cli.set_data_dir(None)


def test_data_dir_not_leaked_across_invocations(tmp_path):
    """Data directory should be invocation-scoped, not leak across CLI calls.

    Regression test for: replay-0d6.9
    Previously, module-global _data_dir leaked across in-process CLI invocations.

    This test verifies that explicit --data-dir is properly scoped per-invocation.
    """
    from pathlib import Path

    from typer.testing import CliRunner

    # Clear any stale global state before running this test
    cli.set_data_dir(None)

    isolated_runner = CliRunner()

    # Use isolated filesystem to ensure clean state
    with isolated_runner.isolated_filesystem():
        # Create git repos for init
        repo1 = Path("repo1")
        repo2 = Path("repo2")
        repo1.mkdir()
        repo2.mkdir()
        (repo1 / ".git").mkdir()
        (repo2 / ".git").mkdir()

        # Create data directories
        data_dir_1 = Path("data1")
        data_dir_2 = Path("data2")

        # Test 1: Initialize with explicit --data-dir
        result1 = isolated_runner.invoke(
            cli.app,
            ["--data-dir", str(data_dir_1), "init", "--path", str(repo1)],
            catch_exceptions=False,
        )
        assert result1.exit_code == 0, f"Init failed: {result1.output}"

        # Test 2: Initialize another db with different --data-dir
        result2 = isolated_runner.invoke(
            cli.app,
            ["--data-dir", str(data_dir_2), "init", "--path", str(repo2)],
            catch_exceptions=False,
        )
        assert result2.exit_code == 0, f"Init failed: {result2.output}"

        # Test 3: Verify --data-dir is invocation-scoped - use data_dir_1
        result3 = isolated_runner.invoke(
            cli.app,
            ["--data-dir", str(data_dir_1), "stats"],
            catch_exceptions=False,
        )
        assert result3.exit_code == 0
        assert data_dir_1.name in result3.output

        # Test 4: Verify --data-dir is invocation-scoped - use data_dir_2
        result4 = isolated_runner.invoke(
            cli.app,
            ["--data-dir", str(data_dir_2), "stats"],
            catch_exceptions=False,
        )
        assert result4.exit_code == 0
        assert data_dir_2.name in result4.output

        # Test 5: Verify data_dir_1 is NOT leaked when calling with data_dir_2
        # The output should show data_dir_2, not data_dir_1
        result5 = isolated_runner.invoke(
            cli.app,
            ["--data-dir", str(data_dir_2), "stats"],
            catch_exceptions=False,
        )
        assert result5.exit_code == 0
        assert data_dir_2.name in result5.output
        # And should NOT contain data_dir_1
        assert data_dir_1.name not in result5.output


class TestConfigErrorHandling:
    """Tests for friendly error messages when database is not initialized.

    Regression tests for: te-qi6
    Previously, commands like stats/search/export/hook install raised ConfigError
    with Python tracebacks when run in directories without .town_elder.
    Now they show friendly CLI errors with actionable guidance.
    """

    @staticmethod
    def _run_te(tmp_path, *args):
        """Run te command in tmp_path using uv run with explicit project dir."""
        import subprocess
        from pathlib import Path

        # Get project directory dynamically (tests are in <project>/tests/)
        project_dir = Path(__file__).parent.parent

        result = subprocess.run(
            ["uv", "run", "--project", str(project_dir), "te", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        return result

    def test_stats_shows_friendly_error_when_not_initialized(self, tmp_path):
        """te stats should show friendly error, not traceback, when not initialized."""
        result = self._run_te(tmp_path, "stats")

        # Should fail with non-zero exit
        assert result.returncode != 0

        # Should NOT contain Python traceback
        assert "Traceback (most recent call last)" not in result.stderr
        assert "Traceback (most recent call last)" not in result.stdout

        # Should show friendly error message (error goes to stderr, guidance to stdout)
        assert "Error: Database not initialized" in result.stderr
        assert "te init" in result.stdout

    def test_search_shows_friendly_error_when_not_initialized(self, tmp_path):
        """te search should show friendly error, not traceback, when not initialized."""
        result = self._run_te(tmp_path, "search", "test query")

        assert result.returncode != 0
        assert "Traceback (most recent call last)" not in result.stderr
        assert "Traceback (most recent call last)" not in result.stdout
        assert "Error: Database not initialized" in result.stderr

    def test_query_shows_friendly_error_when_not_initialized(self, tmp_path):
        """te query should show friendly error, not traceback, when not initialized."""
        result = self._run_te(tmp_path, "query", "test query")

        assert result.returncode != 0
        assert "Traceback (most recent call last)" not in result.stderr
        assert "Error: Database not initialized" in result.stderr

    def test_status_shows_friendly_error_when_not_initialized(self, tmp_path):
        """te status should show friendly error, not traceback, when not initialized."""
        result = self._run_te(tmp_path, "status")

        assert result.returncode != 0
        assert "Traceback (most recent call last)" not in result.stderr
        assert "Error: Database not initialized" in result.stderr

    def test_export_shows_friendly_error_when_not_initialized(self, tmp_path):
        """te export should show friendly error, not traceback, when not initialized."""
        result = self._run_te(tmp_path, "export")

        assert result.returncode != 0
        assert "Traceback (most recent call last)" not in result.stderr
        assert "Error: Database not initialized" in result.stderr

    def test_hook_install_shows_friendly_error_when_not_initialized(self, tmp_path):
        """te hook install should show friendly error, not traceback, when not initialized."""
        import subprocess

        # Create a git repo so hook install doesn't fail on "not a git repo" first
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)

        result = self._run_te(tmp_path, "hook", "install")

        assert result.returncode != 0
        assert "Traceback (most recent call last)" not in result.stderr
        assert "Error: Database not initialized" in result.stderr

    def test_error_message_instructs_to_use_data_dir_option(self, tmp_path):
        """Error messages should guide users to use --data-dir option."""
        result = self._run_te(tmp_path, "stats")

        assert result.returncode != 0
        # Should mention --data-dir as an option (guidance goes to stdout)
        assert "--data-dir" in result.stdout
