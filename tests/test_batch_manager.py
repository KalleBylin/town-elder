"""Tests for chunk batching helpers used by file indexing."""

from __future__ import annotations

import pytest

from town_elder.cli import _flush_file_chunk_batch
from town_elder.indexing.batch_manager import (
    BatchManager,
    ChunkBatchItem,
    ChunkBatchResult,
)

_BATCH_SIZE = 2
_TOTAL_ITEMS = 5
_EXPECTED_BATCH_SIZES = [_BATCH_SIZE, _BATCH_SIZE, 1]
_EXPECTED_BULK_DOCS = 3


def _chunk_item(index: int) -> ChunkBatchItem:
    return ChunkBatchItem(
        doc_id=f"doc-{index}",
        text=f"text-{index}",
        metadata={"chunk_index": index},
        file_path=f"file-{index}.py",
        relative_path=f"file-{index}.py",
        chunk_index=index,
    )


def test_batch_manager_flushes_fixed_batches_and_final_partial() -> None:
    seen_batch_sizes: list[int] = []

    def flush_fn(batch: list[ChunkBatchItem]) -> list[ChunkBatchResult]:
        seen_batch_sizes.append(len(batch))
        return [ChunkBatchResult(item=item) for item in batch]

    manager = BatchManager(batch_size=_BATCH_SIZE, flush_fn=flush_fn)
    results: list[ChunkBatchResult] = []

    for index in range(_TOTAL_ITEMS):
        results.extend(manager.add(_chunk_item(index)))
    results.extend(manager.flush())

    assert seen_batch_sizes == _EXPECTED_BATCH_SIZES
    assert len(results) == _TOTAL_ITEMS
    assert all(result.success for result in results)


def test_batch_manager_rejects_non_positive_batch_size() -> None:
    def flush_fn(batch: list[ChunkBatchItem]) -> list[ChunkBatchResult]:
        return [ChunkBatchResult(item=item) for item in batch]

    with pytest.raises(ValueError):
        BatchManager(batch_size=0, flush_fn=flush_fn)


class _RecordingEmbedder:
    def __init__(self) -> None:
        self.embed_calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        return [[1.0] for _ in texts]


class _RecordingStore:
    def __init__(self) -> None:
        self.bulk_upsert_calls = 0

    def bulk_upsert(
        self,
        docs: list[tuple[str, list[float], str, dict[str, int]]],
    ) -> None:
        self.bulk_upsert_calls += 1
        assert len(docs) == _EXPECTED_BULK_DOCS


def test_flush_file_chunk_batch_uses_single_embed_and_bulk_store_call() -> None:
    embedder = _RecordingEmbedder()
    store = _RecordingStore()
    batch = [_chunk_item(0), _chunk_item(1), _chunk_item(2)]

    results = _flush_file_chunk_batch(batch, embedder=embedder, store=store)

    assert embedder.embed_calls == 1
    assert store.bulk_upsert_calls == 1
    assert len(results) == len(batch)
    assert all(result.success for result in results)
