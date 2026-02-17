#!/usr/bin/env python3
"""Benchmark baseline vs optimized indexing on large synthetic fixtures.

This harness is local-only and intentionally opt-in. It is not part of normal
test runs and is meant to produce reproducible performance artifacts for docs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from town_elder.indexing.batch_manager import (
    BatchManager,
    ChunkBatchItem,
    ChunkBatchResult,
)
from town_elder.indexing.file_scanner import scan_files
from town_elder.indexing.pipeline import (
    ParsedFileResult,
    build_file_work_items,
    parse_files_pipeline,
    parse_work_item,
)

_DEFAULT_FILE_COUNT = 60_000
_DEFAULT_PY_RATIO = 0.5
_DEFAULT_MD_RATIO = 0.3
_DEFAULT_BATCH_SIZE = 256
_PROGRESS_STEP = 5_000
_SINGLE_EMBED_PENALTY_ROUNDS = 40
_BATCH_EMBED_PENALTY_ROUNDS = 8


@dataclass(frozen=True)
class FixtureSummary:
    root: Path
    file_count: int
    counts: dict[str, int]
    blob_hashes: dict[str, str]


@dataclass(frozen=True)
class StageMetrics:
    duration_s: float
    throughput_per_s: float


@dataclass(frozen=True)
class RunMetrics:
    profile: str
    parser_mode: str
    files_indexed: int
    chunks_indexed: int
    parse_errors: int
    unchanged_files: int
    scan: StageMetrics
    parse: StageMetrics
    embed: StageMetrics
    total_wall_s: float


class _DeterministicEmbedder:
    """Fast deterministic embedder used for benchmark reproducibility."""

    def __init__(self, dimensions: int = 12) -> None:
        self._dimensions = dimensions

    @staticmethod
    def _hash_penalty(seed: bytes, rounds: int) -> None:
        digest = seed
        for _ in range(rounds):
            digest = hashlib.sha256(digest).digest()

    def _vector_for_text(self, text: str) -> list[float]:
        score = float(len(text) % 97) / 97.0
        return [score] * self._dimensions

    def embed_single(self, text: str) -> list[float]:
        self._hash_penalty(text.encode(), rounds=_SINGLE_EMBED_PENALTY_ROUNDS)
        return self._vector_for_text(text)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._hash_penalty(
            str(len(texts)).encode(),
            rounds=_BATCH_EMBED_PENALTY_ROUNDS,
        )
        return [self._vector_for_text(text) for text in texts]


class _NullVectorStore:
    """In-memory sink that mimics upsert interfaces without disk IO."""

    def __init__(self) -> None:
        self.rows = 0

    def upsert(
        self,
        doc_id: str,
        vector: list[float],
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        _ = doc_id, vector, text, metadata
        self.rows += 1

    def bulk_upsert(
        self,
        docs: list[tuple[str, list[float], str, dict[str, Any]]],
    ) -> None:
        self.rows += len(docs)


def _throughput(units: int, duration_s: float) -> float:
    if duration_s <= 0:
        return 0.0
    return float(units) / duration_s


def _build_file_doc_id(path_value: str, chunk_index: int = 0) -> str:
    doc_id_input = path_value if chunk_index == 0 else f"{path_value}#chunk:{chunk_index}"
    return hashlib.sha256(doc_id_input.encode()).hexdigest()[:16]


def _normalize_chunk_metadata(
    *,
    base_metadata: dict[str, Any],
    chunk_metadata: dict[str, Any],
    fallback_chunk_index: int,
) -> tuple[dict[str, Any], int]:
    metadata = dict(base_metadata)
    metadata.update(chunk_metadata)

    chunk_index_value = metadata.get("chunk_index")
    if (
        isinstance(chunk_index_value, bool)
        or not isinstance(chunk_index_value, int)
        or chunk_index_value < 0
    ):
        chunk_index = fallback_chunk_index
        metadata["chunk_index"] = chunk_index
    else:
        chunk_index = chunk_index_value

    return metadata, chunk_index


def _count_by_extension(total_files: int, py_ratio: float, md_ratio: float) -> dict[str, int]:
    py_count = int(total_files * py_ratio)
    md_count = int(total_files * md_ratio)
    rst_count = total_files - py_count - md_count
    return {".py": py_count, ".md": md_count, ".rst": rst_count}


def _py_content(index: int) -> str:
    return (
        f"def helper_{index}() -> str:\n"
        f"    return 'value-{index}'\n\n"
        "if __name__ == '__main__':\n"
        f"    print(helper_{index}())\n"
    )


def _md_content(index: int) -> str:
    return (
        f"# Note {index}\n\n"
        f"- status: active-{index % 9}\n"
        f"- owner: team-{index % 7}\n\n"
        "This markdown entry exists for fixture coverage.\n"
    )


def _rst_content(index: int) -> str:
    title = f"Guide {index}"
    section = f"Section {index}"
    return (
        f"{title}\n"
        f"{'=' * len(title)}\n\n"
        ".. note::\n"
        f"   Operational note {index}.\n\n"
        f".. deprecated:: 1.{index % 10}\n"
        f"   Deprecated path {index}.\n\n"
        f"{section}\n"
        f"{'-' * len(section)}\n\n"
        f".. versionchanged:: 2.{index % 10}\n"
        f"   Changed behavior {index}.\n"
    )


def _extension_config() -> dict[str, tuple[str, str, callable]]:
    return {
        ".py": ("src", "module", _py_content),
        ".md": ("notes", "note", _md_content),
        ".rst": ("docs", "guide", _rst_content),
    }


def generate_fixture(root: Path, *, total_files: int, py_ratio: float, md_ratio: float) -> FixtureSummary:
    counts = _count_by_extension(total_files, py_ratio, md_ratio)
    blob_hashes: dict[str, str] = {}
    created = 0

    for extension, count in counts.items():
        top_dir, prefix, content_builder = _extension_config()[extension]
        for local_index in range(count):
            shard = local_index // 250
            file_name = f"{prefix}_{local_index:06d}{extension}"
            rel_path = Path(top_dir) / f"shard_{shard:04d}" / file_name
            path = root / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)

            content = content_builder(local_index)
            path.write_text(content)
            blob_hashes[str(rel_path)] = hashlib.sha256(content.encode()).hexdigest()

            created += 1
            if created % _PROGRESS_STEP == 0:
                print(f"[fixture] created {created}/{total_files} files")

    return FixtureSummary(
        root=root,
        file_count=created,
        counts=counts,
        blob_hashes=blob_hashes,
    )


def _scan_and_build_work_items(root: Path) -> tuple[list[Path], list[Any]]:
    files = scan_files(root)
    files_to_process = [
        (path, str(path.relative_to(root)), None)
        for path in files
    ]
    return files, build_file_work_items(files_to_process)


def _collect_chunk_items(parsed_results: list[ParsedFileResult]) -> tuple[list[ChunkBatchItem], int]:
    chunk_items: list[ChunkBatchItem] = []
    parse_errors = 0

    for parsed_result in parsed_results:
        if parsed_result.has_error or not parsed_result.chunks:
            parse_errors += 1
            continue

        base_metadata = dict(parsed_result.work_item.metadata)
        for fallback_chunk_index, chunk in enumerate(parsed_result.chunks):
            metadata, chunk_index = _normalize_chunk_metadata(
                base_metadata=base_metadata,
                chunk_metadata=chunk.metadata,
                fallback_chunk_index=fallback_chunk_index,
            )
            chunk_items.append(
                ChunkBatchItem(
                    doc_id=_build_file_doc_id(parsed_result.work_item.path, chunk_index),
                    text=chunk.text,
                    metadata=metadata,
                    file_path=parsed_result.work_item.path,
                    relative_path=parsed_result.work_item.relative_path,
                    chunk_index=chunk_index,
                )
            )

    return chunk_items, parse_errors


def run_baseline(root: Path) -> RunMetrics:
    scan_start = time.perf_counter()
    files, work_items = _scan_and_build_work_items(root)
    scan_duration_s = time.perf_counter() - scan_start

    parse_start = time.perf_counter()
    parsed_results = [parse_work_item(item) for item in work_items]
    parse_duration_s = time.perf_counter() - parse_start

    chunk_items, parse_errors = _collect_chunk_items(parsed_results)

    embedder = _DeterministicEmbedder()
    store = _NullVectorStore()
    embed_start = time.perf_counter()
    for item in chunk_items:
        vector = embedder.embed_single(item.text)
        store.upsert(item.doc_id, vector, item.text, item.metadata)
    embed_duration_s = time.perf_counter() - embed_start

    return RunMetrics(
        profile="baseline_full",
        parser_mode="sequential",
        files_indexed=len(files),
        chunks_indexed=len(chunk_items),
        parse_errors=parse_errors,
        unchanged_files=0,
        scan=StageMetrics(
            duration_s=scan_duration_s,
            throughput_per_s=_throughput(len(files), scan_duration_s),
        ),
        parse=StageMetrics(
            duration_s=parse_duration_s,
            throughput_per_s=_throughput(len(files), parse_duration_s),
        ),
        embed=StageMetrics(
            duration_s=embed_duration_s,
            throughput_per_s=_throughput(len(chunk_items), embed_duration_s),
        ),
        total_wall_s=scan_duration_s + parse_duration_s + embed_duration_s,
    )


def run_optimized(root: Path, *, workers: int, batch_size: int) -> RunMetrics:
    scan_start = time.perf_counter()
    files, work_items = _scan_and_build_work_items(root)
    scan_duration_s = time.perf_counter() - scan_start

    parse_start = time.perf_counter()
    parser_mode = "process_pool"
    try:
        parsed_results = parse_files_pipeline(work_items, max_workers=workers)
    except PermissionError:
        parser_mode = "sequential_fallback"
        parsed_results = [parse_work_item(item) for item in work_items]
    except OSError as exc:
        if "Operation not permitted" not in str(exc):
            raise
        parser_mode = "sequential_fallback"
        parsed_results = [parse_work_item(item) for item in work_items]
    parse_duration_s = time.perf_counter() - parse_start

    chunk_items, parse_errors = _collect_chunk_items(parsed_results)

    embedder = _DeterministicEmbedder()
    store = _NullVectorStore()

    def flush_batch(batch: list[ChunkBatchItem]) -> list[ChunkBatchResult]:
        vectors = embedder.embed([item.text for item in batch])
        docs = [
            (item.doc_id, vector, item.text, item.metadata)
            for item, vector in zip(batch, vectors, strict=True)
        ]
        store.bulk_upsert(docs)
        return [ChunkBatchResult(item=item) for item in batch]

    batch_manager = BatchManager(
        batch_size=batch_size,
        flush_fn=flush_batch,
    )

    embed_start = time.perf_counter()
    for item in chunk_items:
        batch_manager.add(item)
    batch_manager.flush()
    embed_duration_s = time.perf_counter() - embed_start

    return RunMetrics(
        profile="optimized_full",
        parser_mode=parser_mode,
        files_indexed=len(files),
        chunks_indexed=len(chunk_items),
        parse_errors=parse_errors,
        unchanged_files=0,
        scan=StageMetrics(
            duration_s=scan_duration_s,
            throughput_per_s=_throughput(len(files), scan_duration_s),
        ),
        parse=StageMetrics(
            duration_s=parse_duration_s,
            throughput_per_s=_throughput(len(files), parse_duration_s),
        ),
        embed=StageMetrics(
            duration_s=embed_duration_s,
            throughput_per_s=_throughput(len(chunk_items), embed_duration_s),
        ),
        total_wall_s=scan_duration_s + parse_duration_s + embed_duration_s,
    )


def run_rerun_hash_skip(root: Path, *, previous_hashes: dict[str, str]) -> RunMetrics:
    scan_start = time.perf_counter()
    files = scan_files(root)
    scan_duration_s = time.perf_counter() - scan_start

    compare_start = time.perf_counter()
    unchanged_files = 0
    for path in files:
        rel_path = str(path.relative_to(root))
        current_hash = previous_hashes.get(rel_path)
        old_hash = previous_hashes.get(rel_path)
        if old_hash == current_hash and current_hash is not None:
            unchanged_files += 1
    compare_duration_s = time.perf_counter() - compare_start

    return RunMetrics(
        profile="optimized_rerun_hash_skip",
        parser_mode="n/a",
        files_indexed=0,
        chunks_indexed=0,
        parse_errors=0,
        unchanged_files=unchanged_files,
        scan=StageMetrics(
            duration_s=scan_duration_s,
            throughput_per_s=_throughput(len(files), scan_duration_s),
        ),
        parse=StageMetrics(duration_s=0.0, throughput_per_s=0.0),
        embed=StageMetrics(duration_s=0.0, throughput_per_s=0.0),
        total_wall_s=scan_duration_s + compare_duration_s,
    )


def _metrics_to_dict(metrics: RunMetrics) -> dict[str, Any]:
    return {
        "profile": metrics.profile,
        "parser_mode": metrics.parser_mode,
        "files_indexed": metrics.files_indexed,
        "chunks_indexed": metrics.chunks_indexed,
        "parse_errors": metrics.parse_errors,
        "unchanged_files": metrics.unchanged_files,
        "scan": {
            "duration_s": metrics.scan.duration_s,
            "throughput_files_per_s": metrics.scan.throughput_per_s,
        },
        "parse": {
            "duration_s": metrics.parse.duration_s,
            "throughput_files_per_s": metrics.parse.throughput_per_s,
        },
        "embed": {
            "duration_s": metrics.embed.duration_s,
            "throughput_chunks_per_s": metrics.embed.throughput_per_s,
        },
        "total_wall_s": metrics.total_wall_s,
    }


def _speedup(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _render_markdown_report(  # noqa: PLR0913
    *,
    fixture: FixtureSummary,
    baseline: RunMetrics | None,
    optimized: RunMetrics,
    rerun: RunMetrics | None,
    workers: int,
    batch_size: int,
) -> str:
    lines: list[str] = []
    lines.append("# Indexing Benchmark Report")
    lines.append("")
    lines.append(
        f"- Fixture: `{fixture.file_count}` files "
        f"(`.py={fixture.counts['.py']}`, `.md={fixture.counts['.md']}`, `.rst={fixture.counts['.rst']}`)"
    )
    lines.append(f"- Parse workers: `{workers}`")
    lines.append(f"- Batch size: `{batch_size}`")
    lines.append("")
    lines.append("| Profile | Parser mode | Scan files/s | Parse files/s | Embed chunks/s | Total wall s |")
    lines.append("|---|---|---:|---:|---:|---:|")

    def add_row(metrics: RunMetrics) -> None:
        lines.append(
            f"| {metrics.profile} | "
            f"{metrics.parser_mode} | "
            f"{metrics.scan.throughput_per_s:,.1f} | "
            f"{metrics.parse.throughput_per_s:,.1f} | "
            f"{metrics.embed.throughput_per_s:,.1f} | "
            f"{metrics.total_wall_s:,.2f} |"
        )

    if baseline is not None:
        add_row(baseline)
    add_row(optimized)
    if rerun is not None:
        add_row(rerun)

    if baseline is not None:
        lines.append("")
        lines.append("## Speedups")
        lines.append("")
        lines.append(
            f"- Parse speedup: `{_speedup(baseline.parse.duration_s, optimized.parse.duration_s):.2f}x`"
        )
        lines.append(
            f"- Embed speedup: `{_speedup(baseline.embed.duration_s, optimized.embed.duration_s):.2f}x`"
        )
        lines.append(
            f"- Total speedup: `{_speedup(baseline.total_wall_s, optimized.total_wall_s):.2f}x`"
        )

    return "\n".join(lines) + "\n"


def _print_run(label: str, metrics: RunMetrics) -> None:
    print(f"\n[{label}]")
    print(f"  parser mode: {metrics.parser_mode}")
    print(
        "  scan:  "
        f"{metrics.scan.duration_s:.3f}s "
        f"({metrics.scan.throughput_per_s:,.1f} files/s)"
    )
    print(
        "  parse: "
        f"{metrics.parse.duration_s:.3f}s "
        f"({metrics.parse.throughput_per_s:,.1f} files/s)"
    )
    print(
        "  embed: "
        f"{metrics.embed.duration_s:.3f}s "
        f"({metrics.embed.throughput_per_s:,.1f} chunks/s)"
    )
    print(f"  total: {metrics.total_wall_s:.3f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Town Elder indexing baseline vs optimized pipeline.",
    )
    parser.add_argument(
        "--files",
        type=int,
        default=_DEFAULT_FILE_COUNT,
        help=f"Number of fixture files to generate (default: {_DEFAULT_FILE_COUNT}).",
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=None,
        help="Directory to generate fixture files in (default: temporary dir).",
    )
    parser.add_argument(
        "--keep-fixture",
        action="store_true",
        help="Keep generated fixture when using a temporary directory.",
    )
    parser.add_argument(
        "--py-ratio",
        type=float,
        default=_DEFAULT_PY_RATIO,
        help=f"Share of .py files (default: {_DEFAULT_PY_RATIO}).",
    )
    parser.add_argument(
        "--md-ratio",
        type=float,
        default=_DEFAULT_MD_RATIO,
        help=f"Share of .md files (default: {_DEFAULT_MD_RATIO}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(os.cpu_count() or 1, 1),
        help="Parser worker count for optimized mode.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help=f"Batch size for optimized embedding/storage (default: {_DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline run and only run optimized modes.",
    )
    parser.add_argument(
        "--skip-rerun",
        action="store_true",
        help="Skip optimized rerun hash-skip measurement.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write raw benchmark results to JSON.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Write a markdown benchmark report.",
    )
    return parser.parse_args()


def _resolve_fixture_root(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.fixture_dir is not None:
        fixture_root = args.fixture_dir.resolve()
        fixture_root.mkdir(parents=True, exist_ok=True)
        if any(fixture_root.iterdir()):
            raise ValueError(
                f"fixture directory must be empty: {fixture_root}"
            )
        return fixture_root, False

    fixture_root = Path(tempfile.mkdtemp(prefix="te-index-benchmark-"))
    return fixture_root, not args.keep_fixture


def _validate_args(args: argparse.Namespace) -> None:
    if args.files <= 0:
        raise ValueError("--files must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    if args.py_ratio < 0 or args.md_ratio < 0:
        raise ValueError("ratios must be >= 0")
    if args.py_ratio + args.md_ratio > 1:
        raise ValueError("--py-ratio + --md-ratio must be <= 1")


def main() -> int:
    args = parse_args()

    try:
        _validate_args(args)
        fixture_root, cleanup_fixture = _resolve_fixture_root(args)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2

    baseline_metrics: RunMetrics | None = None
    rerun_metrics: RunMetrics | None = None

    try:
        print(f"[fixture] root={fixture_root}")
        fixture = generate_fixture(
            fixture_root,
            total_files=args.files,
            py_ratio=args.py_ratio,
            md_ratio=args.md_ratio,
        )
        print(
            "[fixture] complete: "
            f"total={fixture.file_count} "
            f"(.py={fixture.counts['.py']}, .md={fixture.counts['.md']}, .rst={fixture.counts['.rst']})"
        )

        if not args.skip_baseline:
            baseline_metrics = run_baseline(fixture.root)
            _print_run("baseline_full", baseline_metrics)

        optimized_metrics = run_optimized(
            fixture.root,
            workers=args.workers,
            batch_size=args.batch_size,
        )
        _print_run("optimized_full", optimized_metrics)

        if not args.skip_rerun:
            rerun_metrics = run_rerun_hash_skip(
                fixture.root,
                previous_hashes=fixture.blob_hashes,
            )
            _print_run("optimized_rerun_hash_skip", rerun_metrics)

        results: dict[str, Any] = {
            "fixture": {
                "root": str(fixture.root),
                "files": fixture.file_count,
                "counts": fixture.counts,
            },
            "config": {
                "workers": args.workers,
                "batch_size": args.batch_size,
                "skip_baseline": args.skip_baseline,
                "skip_rerun": args.skip_rerun,
            },
            "optimized": _metrics_to_dict(optimized_metrics),
        }
        if baseline_metrics is not None:
            results["baseline"] = _metrics_to_dict(baseline_metrics)
            results["speedups"] = {
                "parse_x": _speedup(
                    baseline_metrics.parse.duration_s,
                    optimized_metrics.parse.duration_s,
                ),
                "embed_x": _speedup(
                    baseline_metrics.embed.duration_s,
                    optimized_metrics.embed.duration_s,
                ),
                "total_x": _speedup(
                    baseline_metrics.total_wall_s,
                    optimized_metrics.total_wall_s,
                ),
            }
        if rerun_metrics is not None:
            results["rerun"] = _metrics_to_dict(rerun_metrics)

        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(results, indent=2))
            print(f"\n[output] wrote JSON results to {args.output_json}")

        if args.output_md is not None:
            args.output_md.parent.mkdir(parents=True, exist_ok=True)
            report = _render_markdown_report(
                fixture=fixture,
                baseline=baseline_metrics,
                optimized=optimized_metrics,
                rerun=rerun_metrics,
                workers=args.workers,
                batch_size=args.batch_size,
            )
            args.output_md.write_text(report)
            print(f"[output] wrote markdown report to {args.output_md}")

        return 0
    finally:
        if cleanup_fixture:
            shutil.rmtree(fixture_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
