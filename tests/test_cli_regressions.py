"""Regression tests for CLI alias dispatch and commit-index state safety."""
from __future__ import annotations

import hashlib
import json
import shlex
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import town_elder.cli as cli
import town_elder.cli_services as cli_services
from town_elder.git.runner import Commit

runner = CliRunner()

# Constants for commit index tests
_TEST_COMMITS_LARGE = 150  # More than default limit to test backlog handling
_TEST_COMMITS_MEDIUM = 50
_TEST_COMMITS_RETRY_COUNT = 49
_TEST_LIMIT = 10
_HASH_LENGTH = 40
_HEX_CHARS = set("0123456789abcdef")


def _get_repo_id(repo_path: Path) -> str:
    """Generate repo ID matching the CLI's _get_repo_id function."""
    canonical = str(repo_path.resolve())
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _get_last_indexed_commit(state_file: Path, repo_path: Path) -> str | None:
    """Extract last_indexed_commit from repo-scoped state file."""
    if not state_file.exists():
        return None
    state = json.loads(state_file.read_text())
    repo_id = _get_repo_id(repo_path)
    return state.get("repos", {}).get(repo_id, {}).get("last_indexed_commit")


def test_query_alias_executes_via_shared_helper(monkeypatch):
    """`te query` should dispatch through the shared search helper."""
    calls: list[tuple[str, int]] = []

    def fake_run_search(
        ctx,
        query: str,
        top_k: int,
    ) -> None:
        calls.append((query, top_k))

    monkeypatch.setattr(cli, "_run_search", fake_run_search)

    result = runner.invoke(cli.app, ["query", "needle", "--top-k", "3"])

    assert result.exit_code == 0
    assert "AttributeError" not in result.output
    assert calls == [("needle", 3)]


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


def test_index_commits_alias_executes_via_shared_helper(monkeypatch):
    """`te index-commits` should dispatch through the shared commit-index helper."""
    calls: list[tuple] = []
    repo_path = "repo-path"

    def fake_run_commit_index(*args) -> None:
        _, path, limit, all_history, batch_size, max_diff_size, incremental, force = args
        calls.append((path, limit, all_history, batch_size, max_diff_size, incremental, force))

    monkeypatch.setattr(cli, "_run_commit_index", fake_run_commit_index)

    result = runner.invoke(
        cli.app,
        [
            "index-commits",
            "--repo",
            repo_path,
            "--limit",
            "7",
            "--all",
            "--batch-size",
            "11",
            "--max-diff-size",
            "2048",
            "--mode",
            "full",
            "--force",
        ],
    )

    assert result.exit_code == 0
    assert calls == [(repo_path, 7, True, 11, 2048, False, True)]


def test_commit_index_executes_via_shared_helper(monkeypatch):
    """`te commit-index` should dispatch through the shared commit-index helper."""
    calls: list[tuple] = []
    repo_path = "repo-path"

    def fake_run_commit_index(*args) -> None:
        _, path, limit, all_history, batch_size, max_diff_size, incremental, force = args
        calls.append((path, limit, all_history, batch_size, max_diff_size, incremental, force))

    monkeypatch.setattr(cli, "_run_commit_index", fake_run_commit_index)

    result = runner.invoke(
        cli.app,
        [
            "commit-index",
            "--repo",
            repo_path,
            "--limit",
            "9",
            "--all",
            "--batch-size",
            "25",
            "--max-diff-size",
            "4096",
            "--full",
            "--force",
        ],
    )

    assert result.exit_code == 0
    assert calls == [(repo_path, 9, True, 25, 4096, False, True)]


def test_root_help_shows_only_canonical_top_level_commands():
    """`te --help` should hide legacy aliases from top-level command listing."""
    result = runner.invoke(cli.app, ["--help"])

    assert result.exit_code == 0
    assert "search" in result.output
    assert "stats" in result.output
    assert "add" in result.output
    assert "index" in result.output
    assert "commit-index" in result.output

    # Hidden compatibility aliases should not appear in root help.
    assert "query" not in result.output
    assert "status" not in result.output
    assert "index-commits" not in result.output


class _FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch embed method used by commit-index."""
        return [[0.42] for _ in texts]

    def embed_single(self, text: str) -> list[float]:
        _ = text
        return [0.42]


class _FakeDiffParser:
    def parse_diff_to_text(self, diff: str) -> str:
        return diff


class _FakeGitRunner:
    def __init__(self, commits: list[Commit]):
        self._commits = commits

    def get_commits(self, since: str | None = None, limit: int = 100, offset: int = 0, include_files: bool = False) -> list[Commit]:
        _ = since, include_files
        return self._commits[offset:offset + limit]

    def get_commits_with_files_batch(self, since: str | None = None, limit: int = 100, offset: int = 0) -> list[Commit]:
        _ = since
        return self._commits[offset:offset + limit]

    def get_diffs_batch(self, commit_hashes: list[str], max_size: int = 100 * 1024) -> dict[str, str]:
        _ = max_size
        return {h: f"diff for {h}" for h in commit_hashes}

    def get_diff(self, commit_hash: str, max_size: int = 100 * 1024) -> str:
        _ = max_size
        return f"diff for {commit_hash}"


class _StoreController:
    def __init__(self, fail_hashes: set[str]):
        self.fail_hashes = fail_hashes
        self.indexed_hashes: list[str] = []


class _FakeStore:
    def __init__(self, controller: _StoreController):
        self._controller = controller

    def get(self, doc_id: str) -> dict | None:
        # Extract commit hash from doc_id (format: "commit_{hash}")
        commit_hash = doc_id.replace("commit_", "")
        if commit_hash in self._controller.indexed_hashes:
            return {"hash": commit_hash}  # Document exists
        return None  # Document does not exist

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

    monkeypatch.setattr(cli_services, "get_config", fake_get_config)
    monkeypatch.setattr(cli_services, "get_service_factory", fake_get_service_factory)

    first_result = runner.invoke(cli.app, ["commit-index", "--repo", str(repo_path), "--limit", "10"])
    assert first_result.exit_code == 0

    state_file = data_dir / "index_state.json"
    assert _get_last_indexed_commit(state_file, repo_path) == "c1"
    assert "c2" not in controller.indexed_hashes

    controller.fail_hashes.clear()
    second_result = runner.invoke(cli.app, ["commit-index", "--repo", str(repo_path), "--limit", "10"])
    assert second_result.exit_code == 0

    assert _get_last_indexed_commit(state_file, repo_path) == "c3"
    assert "c2" in controller.indexed_hashes


def test_commit_index_respects_limit_without_all(monkeypatch, tmp_path):
    """Default behavior should honor --limit when --all is not set."""
    repo_path = tmp_path / "repo"
    (repo_path / ".git").mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    commits = [_commit(f"c{i}") for i in range(30, 0, -1)]
    controller = _StoreController(fail_hashes=set())
    factory = _FakeFactory(git_runner=_FakeGitRunner(commits), controller=controller)
    config = SimpleNamespace(data_dir=data_dir, embed_model="test-model")

    def fake_get_config(data_dir=None):
        _ = data_dir
        return config

    def fake_get_service_factory(data_dir=None):
        _ = data_dir
        return factory

    monkeypatch.setattr(cli_services, "get_config", fake_get_config)
    monkeypatch.setattr(cli_services, "get_service_factory", fake_get_service_factory)

    result = runner.invoke(cli.app, ["commit-index", "--repo", str(repo_path), "--limit", str(_TEST_LIMIT)])
    assert result.exit_code == 0
    assert len(controller.indexed_hashes) == _TEST_LIMIT


def test_commit_index_advances_state_when_sentinel_found_after_pagination(monkeypatch, tmp_path):
    """State should advance when sentinel is found in a later pagination batch."""
    repo_path = tmp_path / "repo"
    (repo_path / ".git").mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    state_file = data_dir / "index_state.json"
    state_file.write_text(json.dumps({"last_indexed_commit": "c1"}))

    total_commits = 220
    commits = [_commit(f"c{i}") for i in range(total_commits, 0, -1)]
    controller = _StoreController(fail_hashes=set())
    factory = _FakeFactory(git_runner=_FakeGitRunner(commits), controller=controller)
    config = SimpleNamespace(data_dir=data_dir, embed_model="test-model")

    def fake_get_config(data_dir=None):
        _ = data_dir
        return config

    def fake_get_service_factory(data_dir=None):
        _ = data_dir
        return factory

    monkeypatch.setattr(cli_services, "get_config", fake_get_config)
    monkeypatch.setattr(cli_services, "get_service_factory", fake_get_service_factory)

    result = runner.invoke(
        cli.app,
        ["commit-index", "--repo", str(repo_path), "--limit", "100", "--batch-size", "50"],
    )
    assert result.exit_code == 0
    assert _get_last_indexed_commit(state_file, repo_path) == f"c{total_commits}"
    assert len(controller.indexed_hashes) == total_commits - 1


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


def test_is_te_hook_detects_uvx_from_town_elder():
    """Hook detection should recognize 'uvx --from town-elder te commit-index'."""
    from town_elder.cli import _is_te_hook

    content = '''#!/bin/sh
# Town Elder post-commit hook - automatically indexes commits
uvx --from town-elder te commit-index --repo "$(git rev-parse --show-toplevel)"
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


def test_is_te_hook_rejects_comment_text():
    """Hook detection should reject TE patterns in comment text."""
    from town_elder.cli import _is_te_hook

    # This should NOT be detected as a TE hook even though it contains "te commit-index"
    # in a comment
    content = '''#!/bin/sh
# This is a comment about te commit-index
echo "Running other hook"
other-tool commit-index --repo "$(git rev-parse --show-toplevel)"
'''
    assert _is_te_hook(content) is False


def test_is_te_hook_rejects_shebang_in_comment():
    """Hook detection should still detect TE hooks that have comments above the command."""
    from town_elder.cli import _is_te_hook

    # This SHOULD be detected as a TE hook because the actual command is present
    content = '''#!/bin/sh
# Town Elder post-commit hook - automatically indexes commits
te commit-index --repo "$(git rev-parse --show-toplevel)"
'''
    assert _is_te_hook(content) is True


def test_is_te_hook_rejects_quoted_strings():
    """Hook detection should reject hooks that only contain the TE command in a string.

    This is a regression test for the bug where 'echo "te commit-index"' would
    incorrectly be detected as a Town Elder hook.
    """
    from town_elder.cli import _is_te_hook

    # Non-TE hook with double-quoted string should be rejected
    content = '''#!/bin/sh
echo "te commit-index"
other-tool --run
'''
    assert _is_te_hook(content) is False

    # Non-TE hook with single-quoted string should be rejected
    content = '''#!/bin/sh
echo 'te commit-index'
other-tool --run
'''
    assert _is_te_hook(content) is False


def test_is_te_hook_accepts_quoted_strings_with_actual_command():
    """Hook detection should still detect TE hooks that have quoted strings AND the actual command."""
    from town_elder.cli import _is_te_hook

    # This SHOULD be detected as a TE hook because the actual command is present
    content = '''#!/bin/sh
echo "te commit-index"
te commit-index --repo "$(git rev-parse --show-toplevel)"
echo "done"
'''
    assert _is_te_hook(content) is True


def test_hook_generation_uses_python_m_town_elder(tmp_path):
    """Generated hooks should use 'uv run te' for robustness across pyenv/uv environments."""
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
    # Hook now uses uv run te for robustness across pyenv/uv environments
    assert "uv run te" in hook_content
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
    # Check that the path is shell-escaped (shlex.quote uses single quotes)
    escaped = shlex.quote(str(data_dir))
    assert f"--data-dir {escaped}" in hook_content


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

    monkeypatch.setattr(cli_services, "get_config", fake_get_config)
    monkeypatch.setattr(cli_services, "get_service_factory", fake_get_service_factory)

    result = runner.invoke(cli.app, ["commit-index", "--repo", str(repo_path), "--limit", "100"])

    assert result.exit_code == 0
    # Should warn about missing sentinel
    assert "Warning: Last indexed commit not found" in result.output
    # Should NOT advance state - check repo-scoped format
    state = json.loads(state_file.read_text())
    repo_id = _get_repo_id(repo_path)
    stored_commit = state.get("repos", {}).get(repo_id, {}).get("last_indexed_commit")
    assert stored_commit == "missing_commit", f"State should not advance. Got: {stored_commit}"
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

    monkeypatch.setattr(cli_services, "get_config", fake_get_config)
    monkeypatch.setattr(cli_services, "get_service_factory", fake_get_service_factory)

    # First run: sentinel not found
    result1 = runner.invoke(cli.app, ["commit-index", "--repo", str(repo_path), "--limit", "100"])
    assert result1.exit_code == 0
    assert "Warning: Last indexed commit not found" in result1.output
    # State should NOT have advanced (sentinel not found)
    assert _get_last_indexed_commit(state_file, repo_path) == "old_gc_commit"
    # All commits should have been indexed
    assert len(controller.indexed_hashes) == _TEST_COMMITS_MEDIUM

    # Simulate gc: now the state points to c1 (oldest indexed)
    # Write in legacy format to test migration
    controller.indexed_hashes.clear()
    state_file.write_text(json.dumps({"last_indexed_commit": "c1"}))

    # Second run: should find c1 and only index newer commits
    result2 = runner.invoke(cli.app, ["commit-index", "--repo", str(repo_path), "--limit", "100"])
    assert result2.exit_code == 0
    assert "Warning" not in result2.output
    # State should have advanced to the newest indexed commit
    assert _get_last_indexed_commit(state_file, repo_path) == "c50"
    # Should have indexed remaining newer commits
    assert len(controller.indexed_hashes) == _TEST_COMMITS_RETRY_COUNT


def test_commit_index_multi_repo_isolation(monkeypatch, tmp_path):
    """Two repos sharing same data-dir should maintain independent cursors.

    This is the main regression test for the bug where global cursor state
    caused cross-repo interference when one data-dir is reused across
    multiple repositories.
    """
    # Create two separate repos
    repo_a_path = tmp_path / "repo_a"
    repo_b_path = tmp_path / "repo_b"
    repo_a_path.mkdir(parents=True)
    repo_b_path.mkdir(parents=True)
    (repo_a_path / ".git").mkdir(parents=True)
    (repo_b_path / ".git").mkdir(parents=True)

    # Shared data directory
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    # Repo A has commits c1, c2, c3
    commits_a = [_commit(f"c{i}") for i in range(3, 0, -1)]
    # Repo B has commits d1, d2, d3 (different hashes)
    commits_b = [_commit(f"d{i}") for i in range(3, 0, -1)]

    config = SimpleNamespace(data_dir=data_dir, embed_model="test-model")

    def fake_get_config(data_dir=None):
        _ = data_dir
        return config

    monkeypatch.setattr(cli_services, "get_config", fake_get_config)

    # === Index Repo A ===
    controller_a = _StoreController(fail_hashes=set())
    factory_a = _FakeFactory(git_runner=_FakeGitRunner(commits_a), controller=controller_a)

    def fake_get_service_factory_a(data_dir=None):
        _ = data_dir
        return factory_a

    monkeypatch.setattr(cli_services, "get_service_factory", fake_get_service_factory_a)

    result_a = runner.invoke(cli.app, ["commit-index", "--repo", str(repo_a_path), "--limit", "10"])
    assert result_a.exit_code == 0

    state_file = data_dir / "index_state.json"

    # Repo A should have its own cursor (frontier is newest indexed, i.e., c3)
    assert _get_last_indexed_commit(state_file, repo_a_path) == "c3"
    # Repo B should NOT have any state yet
    assert _get_last_indexed_commit(state_file, repo_b_path) is None

    # === Index Repo B ===
    controller_b = _StoreController(fail_hashes=set())
    factory_b = _FakeFactory(git_runner=_FakeGitRunner(commits_b), controller=controller_b)

    def fake_get_service_factory_b(data_dir=None):
        _ = data_dir
        return factory_b

    monkeypatch.setattr(cli_services, "get_service_factory", fake_get_service_factory_b)

    result_b = runner.invoke(cli.app, ["commit-index", "--repo", str(repo_b_path), "--limit", "10"])
    assert result_b.exit_code == 0
    # Should NOT have "Warning: Last indexed commit not found" - that's the bug!
    assert "Warning: Last indexed commit not found" not in result_b.output

    # Both repos should have independent state now (each has its own frontier)
    assert _get_last_indexed_commit(state_file, repo_a_path) == "c3"
    assert _get_last_indexed_commit(state_file, repo_b_path) == "d3"

    # State file should have both repos (re-read after repo B indexing)
    state_after = json.loads(state_file.read_text())
    assert "repos" in state_after
    repo_a_id = _get_repo_id(repo_a_path)
    repo_b_id = _get_repo_id(repo_b_path)
    assert repo_a_id in state_after["repos"]
    assert repo_b_id in state_after["repos"]


def test_commit_index_legacy_migration(monkeypatch, tmp_path):
    """Legacy state format should be migrated to repo-scoped format."""
    repo_path = tmp_path / "repo"
    (repo_path / ".git").mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    # Write legacy format directly
    state_file = data_dir / "index_state.json"
    state_file.write_text(json.dumps({"last_indexed_commit": "legacy_commit"}))

    commits = [_commit("c3"), _commit("c2"), _commit("c1")]
    controller = _StoreController(fail_hashes=set())
    factory = _FakeFactory(git_runner=_FakeGitRunner(commits), controller=controller)
    config = SimpleNamespace(data_dir=data_dir, embed_model="test-model")

    def fake_get_config(data_dir=None):
        _ = data_dir
        return config

    def fake_get_service_factory(data_dir=None):
        _ = data_dir
        return factory

    monkeypatch.setattr(cli_services, "get_config", fake_get_config)
    monkeypatch.setattr(cli_services, "get_service_factory", fake_get_service_factory)

    result = runner.invoke(cli.app, ["commit-index", "--repo", str(repo_path), "--limit", "10"])
    assert result.exit_code == 0

    # State file should now be in new format (migrated)
    state = json.loads(state_file.read_text())
    assert "repos" in state

    repo_id = _get_repo_id(repo_path)
    assert repo_id in state["repos"]
    # Sentinel was not found (legacy_commit not in c1,c2,c3), so state should NOT advance
    # It remains as the migrated legacy value
    assert state["repos"][repo_id]["last_indexed_commit"] == "legacy_commit"


class TestHookExecution:
    """Tests for actually executing generated hooks."""

    def test_generated_hook_executes_successfully(self, tmp_path):
        """Generated hook should execute successfully when commit is made.

        This is a true integration test that verifies actual hook-triggered
        side effects (index state file creation) rather than mocking.
        """
        import subprocess

        # Create a git repo
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()

        # Configure git
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

        # Create a data dir (but don't create it - init will create it)
        data_dir = tmp_path / "data"

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

            # Verify the generated hook uses fallback chain: uv -> uvx -> te -> python -m town_elder
            hook_content = hook_path.read_text()
            assert "uv run te" in hook_content, (
                f"Hook should use 'uv run te' as primary method. Got: {hook_content}"
            )
            assert "uvx --from town-elder te" in hook_content, (
                "Hook should include uvx fallback for ad-hoc/tool-only usage"
            )
            assert "command -v uv" in hook_content, (
                "Hook should check for uv availability"
            )
            assert "command -v te" in hook_content, (
                "Hook should check for te availability"
            )
            assert "python -m town_elder" in hook_content, (
                "Hook should have python -m town_elder as fallback"
            )

            # Verify no index state before commit
            state_file = data_dir / "index_state.json"
            assert not state_file.exists(), "Index state should not exist before first commit"

            # Make a commit (this should trigger the hook)
            (repo_path / "test.txt").write_text("test content")
            subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True, check=True)
            subprocess.run(
                ["git", "commit", "-m", "test commit"],
                cwd=repo_path,
                capture_output=True,
                check=True,
            )

            # Verify the hook executed by checking real side effects:
            # 1. The index_state.json file was created
            assert state_file.exists(), (
                "Hook did not execute: index_state.json not created. "
                "Hook may have failed or not be triggered."
            )

            # 2. The state file contains a valid commit hash (repo-scoped format)
            state = json.loads(state_file.read_text())
            # New repo-scoped format has state under repos key
            assert "repos" in state, "Index state should use repo-scoped format"
            repos = state["repos"]
            assert len(repos) == 1, "Should have exactly one repo in state"
            repo_id = list(repos.keys())[0]
            repo_state = repos[repo_id]
            assert "last_indexed_commit" in repo_state, "Index state missing last_indexed_commit"
            commit_hash = repo_state["last_indexed_commit"]
            assert commit_hash, "last_indexed_commit is empty"

            # 3. Verify the commit hash is valid (40 char hex for SHA)
            assert len(commit_hash) == _HASH_LENGTH, f"Invalid commit hash length: {len(commit_hash)}"
            assert all(c in _HEX_CHARS for c in commit_hash), f"Invalid commit hash: {commit_hash}"

            # Verify the commit actually exists in the repo
            result = subprocess.run(
                ["git", "rev-parse", commit_hash],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"Commit {commit_hash} not found in repo"
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


class TestDataDirValidation:
    """Tests for --data-dir validation at CLI entry."""

    def test_data_dir_rejects_missing_path(self, tmp_path):
        """CLI should fail early when --data-dir path does not exist."""
        missing_data_dir = tmp_path / "missing-data-dir"

        result = runner.invoke(
            cli.app,
            ["--data-dir", str(missing_data_dir), "stats"],
        )

        assert result.exit_code == 1
        assert "--data-dir does not exist" in result.output

    def test_data_dir_rejects_file_path(self, tmp_path):
        """CLI should fail early when --data-dir points to a file."""
        not_a_dir = tmp_path / "not-a-dir"
        not_a_dir.write_text("file content")

        result = runner.invoke(
            cli.app,
            ["--data-dir", str(not_a_dir), "stats"],
        )

        assert result.exit_code == 1
        assert "--data-dir exists but is not a directory" in result.output


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

        return subprocess.run(
            ["uv", "run", "--project", str(project_dir), "te", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

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
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)

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


def test_get_config_no_cache_leakage_across_cwd_changes(tmp_path):
    """get_config should not return stale cached config when cwd changes.

    Regression test for: te-j50
    When data_dir is not provided, get_config() resolves the default from cwd.
    The cache should not return stale values when cwd changes between calls.
    """
    import os
    from pathlib import Path

    from town_elder.config import get_config

    # Create three separate directories with .town_elder subdirs
    dir_a = tmp_path / "dir_a"
    dir_b = tmp_path / "dir_b"
    dir_c = tmp_path / "dir_c"

    dir_a.mkdir()
    dir_b.mkdir()
    dir_c.mkdir()

    # Create .town_elder in each directory
    (dir_a / ".town_elder").mkdir()
    (dir_b / ".town_elder").mkdir()
    (dir_c / ".town_elder").mkdir()

    original_cwd = Path.cwd()

    try:
        # Test 1: Change to dir_a and get config (no explicit data_dir)
        os.chdir(dir_a)
        config_a = get_config()
        assert config_a.data_dir == dir_a / ".town_elder", (
            f"Expected {dir_a / '.town_elder'}, got {config_a.data_dir}"
        )

        # Test 2: Change to dir_b and get config (should NOT return cached config_a)
        os.chdir(dir_b)
        config_b = get_config()
        assert config_b.data_dir == dir_b / ".town_elder", (
            f"Expected {dir_b / '.town_elder'}, got {config_b.data_dir}. "
            "Cache returned stale config from previous cwd!"
        )

        # Test 3: Change to dir_c and get config (should NOT return cached config_a or config_b)
        os.chdir(dir_c)
        config_c = get_config()
        assert config_c.data_dir == dir_c / ".town_elder", (
            f"Expected {dir_c / '.town_elder'}, got {config_c.data_dir}. "
            "Cache returned stale config from previous cwd!"
        )

        # Test 4: Go back to dir_a to verify it still returns correct value
        os.chdir(dir_a)
        config_a2 = get_config()
        assert config_a2.data_dir == dir_a / ".town_elder", (
            f"Expected {dir_a / '.town_elder'}, got {config_a2.data_dir}. "
            "Cache returned stale config!"
        )
    finally:
        os.chdir(original_cwd)


class TestTopKValidation:
    """Tests for --top-k parameter validation in search/query commands.

    Regression tests for: te-17p
    Previously, negative or zero values for --top-k caused errors.
    Now they are properly validated with friendly error messages.
    """

    def test_search_rejects_zero_top_k(self, tmp_path):
        """te search should reject --top-k=0 with friendly error."""
        # Create a minimal git repo so search can run
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        result = runner.invoke(
            cli.app,
            ["--data-dir", str(data_dir), "search", "test", "--top-k", "0"],
        )

        assert result.exit_code != 0
        assert "positive integer" in result.output.lower()

    def test_search_rejects_negative_top_k(self, tmp_path):
        """te search should reject negative --top-k values with friendly error."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        result = runner.invoke(
            cli.app,
            ["--data-dir", str(data_dir), "search", "test", "--top-k", "-1"],
        )

        assert result.exit_code != 0
        assert "positive integer" in result.output.lower()

    def test_query_rejects_zero_top_k(self, tmp_path):
        """te query should reject --top-k=0 with friendly error."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        result = runner.invoke(
            cli.app,
            ["--data-dir", str(data_dir), "query", "test", "--top-k", "0"],
        )

        assert result.exit_code != 0
        assert "positive integer" in result.output.lower()

    def test_query_rejects_negative_top_k(self, tmp_path):
        """te query should reject negative --top-k values with friendly error."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        result = runner.invoke(
            cli.app,
            ["--data-dir", str(data_dir), "query", "test", "--top-k", "-5"],
        )

        assert result.exit_code != 0
        assert "positive integer" in result.output.lower()

    def test_search_accepts_valid_positive_top_k(self, tmp_path):
        """te search should accept valid positive --top-k values."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()

        # Create a data directory with minimal state
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "vectors.jsonl").write_text("")  # Empty but exists

        result = runner.invoke(
            cli.app,
            ["--data-dir", str(data_dir), "search", "test", "--top-k", "10"],
        )

        # Should not fail with validation error (may fail for other reasons like no results,
        # but should NOT have "positive integer" error)
        assert "positive integer" not in result.output.lower()


class TestNonTextHookFiles:
    """Tests for handling non-text (binary) hook files.

    Regression tests for: te-vay
    Previously, non-UTF8 hook files caused crashes when reading hook content.
    Now they are handled gracefully with None return and "Unknown" type.
    """

    def test_hook_status_handles_non_utf8_hook(self, tmp_path):
        """te hook status should handle non-UTF8 hook files without crashing."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()

        # Create a binary (non-UTF8) hook file
        hook_path = repo_path / ".git" / "hooks" / "post-commit"
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        hook_path.write_bytes(b"\x00\x01\x02\xff\xfe\xfd")

        result = runner.invoke(
            cli.app,
            ["hook", "status", "--repo", str(repo_path)],
        )

        assert result.exit_code == 0
        assert "Unknown" in result.output

    def test_hook_uninstall_handles_non_utf8_hook(self, tmp_path):
        """te hook uninstall should handle non-UTF8 hook files without crashing.

        Non-UTF8 hooks are treated as unknown (not TE hooks). The --force flag
        is needed to delete unknown hooks.
        """
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()

        # Create a binary (non-UTF8) hook file
        hook_path = repo_path / ".git" / "hooks" / "post-commit"
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        hook_path.write_bytes(b"\x00\x01\x02\xff\xfe\xfd")

        result = runner.invoke(
            cli.app,
            ["hook", "uninstall", "--repo", str(repo_path), "--force"],
        )

        # Should not crash - should handle gracefully with --force
        assert result.exit_code == 0

    def test_safe_read_hook_returns_none_for_binary(self, tmp_path):
        """_safe_read_hook should return None for binary (non-UTF8) files."""
        from town_elder.cli import _safe_read_hook

        # Create a binary file
        test_file = tmp_path / "binary_file"
        test_file.write_bytes(b"\x80\x81\x82\xff\xfe\xfd")

        result = _safe_read_hook(test_file)

        assert result is None

    def test_safe_read_hook_returns_content_for_text(self, tmp_path):
        """_safe_read_hook should return content for valid UTF-8 text files."""
        from town_elder.cli import _safe_read_hook

        # Create a text file
        test_file = tmp_path / "text_file"
        test_file.write_text("#!/bin/sh\necho 'test'")

        result = _safe_read_hook(test_file)

        assert result is not None
        assert "echo 'test'" in result

    def test_safe_read_hook_returns_none_for_nonexistent(self, tmp_path):
        """_safe_read_hook should return None for nonexistent files."""
        from town_elder.cli import _safe_read_hook

        nonexistent = tmp_path / "does_not_exist"

        result = _safe_read_hook(nonexistent)

        assert result is None
