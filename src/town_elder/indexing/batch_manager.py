"""Batch helpers for high-throughput file indexing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

DEFAULT_BATCH_SIZE = 256


@dataclass(frozen=True)
class ChunkBatchItem:
    """Single file chunk payload destined for embedding + storage."""

    doc_id: str
    text: str
    metadata: dict[str, Any]
    file_path: str
    relative_path: str
    chunk_index: int


@dataclass(frozen=True)
class ChunkBatchResult:
    """Result for a chunk write attempt."""

    item: ChunkBatchItem
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


class BatchManager:
    """Accumulates chunk payloads and flushes fixed-size batches."""

    def __init__(
        self,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        flush_fn: Callable[[list[ChunkBatchItem]], list[ChunkBatchResult]],
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        self.batch_size = batch_size
        self._flush_fn = flush_fn
        self._pending: list[ChunkBatchItem] = []

    @property
    def pending_count(self) -> int:
        """Current number of queued chunks."""

        return len(self._pending)

    def add(self, item: ChunkBatchItem) -> list[ChunkBatchResult]:
        """Queue one chunk and flush when batch size is reached."""

        self._pending.append(item)
        if len(self._pending) < self.batch_size:
            return []
        return self.flush()

    def flush(self) -> list[ChunkBatchResult]:
        """Flush pending chunks, returning per-chunk results."""

        if not self._pending:
            return []
        batch = self._pending.copy()
        self._pending.clear()
        return self._flush_fn(batch)
