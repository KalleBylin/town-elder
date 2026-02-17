"""Regression tests for incremental file indexing in the CLI."""

from __future__ import annotations

import hashlib
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import town_elder.cli as cli
from town_elder.indexing.pipeline import parse_work_item

runner = CliRunner()

_SHA1_HEX_LENGTH = 40


class _FakeEmbedder:
    def __init__(self):
        self.embed_calls = 0
        self.embed_single_calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        return [[0.42] for _ in texts]

    def embed_single(self, text: str) -> list[float]:
        _ = text
        self.embed_single_calls += 1
        return [0.42]


class _RecordingStore:
    def __init__(self, enforce_json_metadata: bool = False):
        self.enforce_json_metadata = enforce_json_metadata
        self.upserts: list[tuple[str, dict[str, str]]] = []
        self.deletes: list[str] = []
        self.bulk_upsert_calls = 0

    def upsert(
        self,
        doc_id: str,
        vector: list[float],
        text: str,
        metadata: dict[str, str],
    ) -> None:
        _ = vector, text
        if self.enforce_json_metadata:
            json.dumps(metadata)
        self.upserts.append((doc_id, metadata))

    def bulk_upsert(
        self,
        docs: list[tuple[str, list[float], str, dict[str, str]]],
    ) -> None:
        self.bulk_upsert_calls += 1
        for doc_id, vector, text, metadata in docs:
            self.upsert(doc_id, vector, text, metadata)

    def delete(self, doc_id: str) -> None:
        self.deletes.append(doc_id)

    def close(self) -> None:
        return


def _make_fake_cli_services(store: _RecordingStore, embedder: _FakeEmbedder):
    @contextmanager
    def _fake_cli_services(ctx):
        _ = ctx
        yield SimpleNamespace(), embedder, store

    return _fake_cli_services


def _patch_cli_services(
    monkeypatch,
    data_dir: Path,
    store: _RecordingStore,
    embedder: _FakeEmbedder | None = None,
) -> _FakeEmbedder:
    def _fake_require_initialized(ctx):
        _ = ctx
        return SimpleNamespace(data_dir=data_dir)

    fake_embedder = embedder or _FakeEmbedder()

    monkeypatch.setattr(
        cli,
        "require_initialized",
        _fake_require_initialized,
    )
    monkeypatch.setattr(
        cli,
        "get_cli_services",
        _make_fake_cli_services(store, fake_embedder),
    )
    monkeypatch.setattr(
        cli,
        "parse_files_pipeline",
        lambda work_items: [parse_work_item(item) for item in work_items],
    )
    return fake_embedder


def _init_git_repo_with_file(tmp_path: Path, file_name: str = "main.py") -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    (repo / file_name).write_text("print('hello')\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    return repo


def test_index_files_incremental_uses_string_blob_hash_metadata(monkeypatch, tmp_path):
    """TrackedFile blob hashes should be converted to strings for metadata/state."""
    repo = _init_git_repo_with_file(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    store = _RecordingStore(enforce_json_metadata=True)
    _patch_cli_services(monkeypatch, data_dir, store)

    result = runner.invoke(
        cli.app,
        ["--data-dir", str(data_dir), "index", "files", str(repo)],
    )

    assert result.exit_code == 0, result.output
    assert len(store.upserts) == 1

    _doc_id, metadata = store.upserts[0]
    blob_hash = metadata["blob_hash"]
    assert isinstance(blob_hash, str)
    assert len(blob_hash) == _SHA1_HEX_LENGTH
    int(blob_hash, 16)  # Should be valid hex

    state_file = cli._get_file_state_path(data_dir)
    saved_state = json.loads(state_file.read_text())
    repo_id = cli._get_repo_id(repo)
    assert saved_state[repo_id]["file_hashes"] == {"main.py": blob_hash}


def test_index_files_incremental_unchanged_run_does_not_delete(monkeypatch, tmp_path):
    """Unchanged incremental runs should skip files without deleting stored docs."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("print('hello')\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    store = _RecordingStore()
    _patch_cli_services(monkeypatch, data_dir, store)

    file_hashes = {"main.py": "a" * _SHA1_HEX_LENGTH}
    monkeypatch.setattr(cli, "get_git_root", lambda _path: project)
    monkeypatch.setattr(cli, "scan_git_blobs", lambda _repo: file_hashes.copy())

    first = runner.invoke(
        cli.app,
        ["--data-dir", str(data_dir), "index", "files", str(project)],
    )
    assert first.exit_code == 0, first.output
    assert len(store.upserts) == 1
    assert store.deletes == []

    second = runner.invoke(
        cli.app,
        ["--data-dir", str(data_dir), "index", "files", str(project)],
    )
    assert second.exit_code == 0, second.output
    assert "skipped 1 unchanged" in second.output
    assert len(store.upserts) == 1
    assert store.deletes == []


def test_index_files_incremental_deletion_uses_absolute_doc_id_and_clears_state(
    monkeypatch, tmp_path
):
    """Deleted files should remove the original doc ID and persist empty file state."""
    project = tmp_path / "project"
    project.mkdir()
    tracked_file = project / "main.py"
    tracked_file.write_text("print('hello')\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    store = _RecordingStore()
    _patch_cli_services(monkeypatch, data_dir, store)

    file_hashes = {"main.py": "b" * _SHA1_HEX_LENGTH}
    monkeypatch.setattr(cli, "get_git_root", lambda _path: project)
    monkeypatch.setattr(cli, "scan_git_blobs", lambda _repo: file_hashes.copy())

    first = runner.invoke(
        cli.app,
        ["--data-dir", str(data_dir), "index", "files", str(project)],
    )
    assert first.exit_code == 0, first.output
    assert len(store.upserts) == 1

    expected_doc_id = hashlib.sha256(str(tracked_file).encode()).hexdigest()[:16]

    tracked_file.unlink()
    file_hashes.clear()

    second = runner.invoke(
        cli.app,
        ["--data-dir", str(data_dir), "index", "files", str(project)],
    )
    assert second.exit_code == 0, second.output
    assert expected_doc_id in store.deletes

    state_file = cli._get_file_state_path(data_dir)
    saved_state = json.loads(state_file.read_text())
    repo_id = cli._get_repo_id(project)
    assert saved_state[repo_id]["file_hashes"] == {}


def test_index_files_incremental_indexes_markdown_files(monkeypatch, tmp_path):
    """Markdown files should flow through incremental indexing with blob metadata."""

    repo = _init_git_repo_with_file(tmp_path, file_name="README.md")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    store = _RecordingStore(enforce_json_metadata=True)
    _patch_cli_services(monkeypatch, data_dir, store)

    result = runner.invoke(
        cli.app,
        ["--data-dir", str(data_dir), "index", "files", str(repo)],
    )

    assert result.exit_code == 0, result.output
    assert len(store.upserts) == 1

    _doc_id, metadata = store.upserts[0]
    assert metadata["type"] == ".md"
    blob_hash = metadata["blob_hash"]
    assert isinstance(blob_hash, str)
    assert len(blob_hash) == _SHA1_HEX_LENGTH

    state_file = cli._get_file_state_path(data_dir)
    saved_state = json.loads(state_file.read_text())
    repo_id = cli._get_repo_id(repo)
    assert saved_state[repo_id]["file_hashes"] == {"README.md": blob_hash}


def test_index_files_uses_batch_embed_and_bulk_upsert(monkeypatch, tmp_path):
    """File indexing should use batch embed + bulk upsert paths."""
    repo = _init_git_repo_with_file(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    store = _RecordingStore()
    embedder = _patch_cli_services(monkeypatch, data_dir, store)

    result = runner.invoke(
        cli.app,
        ["--data-dir", str(data_dir), "index", "files", str(repo)],
    )

    assert result.exit_code == 0, result.output
    assert embedder.embed_calls == 1
    assert embedder.embed_single_calls == 0
    assert store.bulk_upsert_calls == 1


def test_index_files_emits_stage_status_lines_non_interactive(monkeypatch, tmp_path):
    """Non-TTY runs should emit readable stage status and final summary."""
    repo = _init_git_repo_with_file(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    store = _RecordingStore()
    _patch_cli_services(monkeypatch, data_dir, store)

    result = runner.invoke(
        cli.app,
        ["--data-dir", str(data_dir), "index", "files", str(repo)],
    )

    assert result.exit_code == 0, result.output
    assert "Scanning: 0/?" in result.output
    assert "Scanning: 1/1" in result.output
    assert "Parsing: 1/1" in result.output
    assert "Embedding: 1/1" in result.output
    assert "Storing: 1/1" in result.output
    assert "Indexed 1 files" in result.output
