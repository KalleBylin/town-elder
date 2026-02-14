"""Regression tests for CLI alias dispatch and commit-index state safety."""
from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

from typer.testing import CliRunner

import replay.cli as cli
from replay.git.runner import Commit

runner = CliRunner()


def test_query_alias_executes_via_shared_helper(monkeypatch):
    """`replay query` should dispatch through the shared search helper."""
    calls: list[tuple[str, int, str | None, str | None, str | None]] = []

    def fake_run_search(
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
    """`replay status` should dispatch through the shared stats helper."""
    calls = {"count": 0}

    def fake_run_stats() -> None:
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

    def insert_with_vector(
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
