"""File indexing pipeline with producer and parser worker stages."""

from __future__ import annotations

import os
import queue
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from town_elder.parsers.rst_handler import (
    get_chunk_metadata as python_get_chunk_metadata,
)
from town_elder.parsers.rst_handler import (
    parse_rst_content as python_parse_rst_content,
)
from town_elder.rust_adapter import parse_rst_chunks


@dataclass(frozen=True)
class FileWorkItem:
    """Represents one file scheduled for parser workers."""

    sequence: int
    path: str
    relative_path: str
    file_type: str
    blob_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedChunk:
    """Represents one parsed chunk emitted by parser workers."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedFileResult:
    """Parser output for a single source file."""

    work_item: FileWorkItem
    chunks: tuple[ParsedChunk, ...] = ()
    error: str | None = None

    @property
    def has_error(self) -> bool:
        return self.error is not None


def build_file_work_items(
    files_to_process: list[tuple[Path, str, str | None]],
) -> list[FileWorkItem]:
    """Build ordered work items from file scan output."""

    work_items: list[FileWorkItem] = []
    for sequence, (file_path, relative_path, blob_hash) in enumerate(files_to_process):
        work_items.append(
            FileWorkItem(
                sequence=sequence,
                path=str(file_path),
                relative_path=relative_path,
                file_type=file_path.suffix,
                blob_hash=blob_hash,
                metadata={"source": str(file_path), "type": file_path.suffix},
            )
        )
    return work_items


def _parse_plain_text(content: str) -> tuple[ParsedChunk, ...]:
    return (ParsedChunk(text=content, metadata={"chunk_index": 0}),)


def _parse_rst_content(content: str) -> tuple[ParsedChunk, ...]:
    rust_chunks = parse_rst_chunks(content)
    if rust_chunks is not None:
        if not rust_chunks:
            # Preserve old behavior for empty/malformed files by indexing raw text.
            return _parse_plain_text(content)

        return tuple(
            ParsedChunk(text=text, metadata=metadata)
            for text, metadata in rust_chunks
        )

    rst_chunks = python_parse_rst_content(content)
    if not rst_chunks:
        # Preserve old behavior for empty/malformed files by indexing raw text.
        return _parse_plain_text(content)

    parsed_chunks: list[ParsedChunk] = []
    for chunk in rst_chunks:
        parsed_chunks.append(
            ParsedChunk(
                text=chunk.text,
                metadata=python_get_chunk_metadata(chunk),
            )
        )
    return tuple(parsed_chunks)


def parse_work_item(work_item: FileWorkItem) -> ParsedFileResult:
    """Parse one work item in a worker process."""

    file_path = Path(work_item.path)
    try:
        content = file_path.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        return ParsedFileResult(work_item=work_item, error=str(exc))

    try:
        if work_item.file_type == ".rst":
            chunks = _parse_rst_content(content)
        elif work_item.file_type in {".py", ".md"}:
            chunks = _parse_plain_text(content)
        else:
            chunks = _parse_plain_text(content)
    except Exception as exc:
        return ParsedFileResult(work_item=work_item, error=str(exc))

    return ParsedFileResult(work_item=work_item, chunks=chunks)


def _producer_loop(
    work_items: list[FileWorkItem],
    work_queue: queue.Queue[FileWorkItem | object],
    done_token: object,
    cancel_event: threading.Event,
    producer_errors: list[BaseException],
) -> None:
    try:
        for item in work_items:
            while not cancel_event.is_set():
                try:
                    work_queue.put(item, timeout=0.1)
                    break
                except queue.Full:
                    continue
            if cancel_event.is_set():
                break
    except Exception as exc:
        producer_errors.append(exc)
    finally:
        while True:
            try:
                work_queue.put(done_token, timeout=0.1)
                break
            except queue.Full:
                if cancel_event.is_set():
                    break
                continue


def parse_files_pipeline(  # noqa: PLR0912
    work_items: list[FileWorkItem],
    *,
    max_workers: int | None = None,
    queue_size: int = 128,
) -> list[ParsedFileResult]:
    """Parse files through producer + process worker stages."""

    if not work_items:
        return []
    if queue_size <= 0:
        raise ValueError("queue_size must be greater than 0")

    worker_count = max_workers or (os.cpu_count() or 1)
    worker_count = max(worker_count, 1)
    max_in_flight = max(worker_count * 2, 1)

    done_token = object()
    cancel_event = threading.Event()
    producer_errors: list[BaseException] = []
    work_queue: queue.Queue[FileWorkItem | object] = queue.Queue(maxsize=queue_size)

    producer = threading.Thread(
        target=_producer_loop,
        args=(work_items, work_queue, done_token, cancel_event, producer_errors),
        name="file-index-producer",
        daemon=True,
    )
    producer.start()

    in_flight: dict[Future[ParsedFileResult], int] = {}
    buffered: dict[int, ParsedFileResult] = {}
    parsed_results: list[ParsedFileResult] = []
    producer_done = False
    next_sequence = 0

    try:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            while True:
                while not producer_done and len(in_flight) < max_in_flight:
                    item = work_queue.get()
                    if item is done_token:
                        producer_done = True
                        break

                    future = executor.submit(parse_work_item, item)
                    in_flight[future] = item.sequence

                if not in_flight:
                    if producer_done:
                        break
                    continue

                done_futures, _ = wait(
                    tuple(in_flight),
                    return_when=FIRST_COMPLETED,
                )
                for future in done_futures:
                    sequence = in_flight.pop(future)
                    buffered[sequence] = future.result()

                while next_sequence in buffered:
                    parsed_results.append(buffered.pop(next_sequence))
                    next_sequence += 1

        if producer_errors:
            raise RuntimeError("File producer stage failed") from producer_errors[0]

        return parsed_results
    except Exception:
        cancel_event.set()
        for future in in_flight:
            future.cancel()
        raise
    finally:
        cancel_event.set()
        producer.join(timeout=5)
