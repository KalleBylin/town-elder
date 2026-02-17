# Indexing Guide

This document covers file indexing behavior, `.rst` metadata extraction, and performance benchmarking.

## Full Repository Indexing

Use the canonical full-index command:

```bash
uv run te index --all
```

Equivalent explicit form:

```bash
uv run te index files .
```

By default, file indexing runs in incremental mode and skips unchanged tracked files.

## Indexed File Types and `.rst` Metadata

Town Elder indexes:

- `.py`
- `.md`
- `.rst`

For `.rst`, the parser emits semantic chunk metadata used by the CLI indexing flow:

- `section_path`
- `section_depth`
- `directives` / `has_directives`
- `temporal_tags` / `has_temporal_tags`

Temporal tags include directives like `.. deprecated::` and `.. versionchanged::`.

## Hash-Based Incremental Behavior

State is written to:

- `.town_elder/file_index_state.json`

Expected behavior:

1. First run indexes all tracked files and stores blob-hash state.
2. Rerun compares current tracked blob hashes to saved state.
3. Unchanged files are skipped.
4. Changed/new files are re-indexed.
5. Deleted tracked files are removed from the vector store and state.

Force a full reindex with:

```bash
uv run te index files . --no-incremental
```

## Benchmark Harness

Use `scripts/benchmark_indexing.py` to generate large synthetic fixtures and measure:

- scan throughput
- parse throughput
- embed throughput
- total wall time

### 60k Fixture Benchmark

```bash
PYTHONPATH=src python scripts/benchmark_indexing.py \
  --files 60000 \
  --workers 8 \
  --output-json docs/benchmarks/indexing-60k-2026-02-17.json \
  --output-md docs/benchmarks/indexing-60k-2026-02-17.md
```

Results artifact:

- `docs/benchmarks/indexing-60k-2026-02-17.json`
- `docs/benchmarks/indexing-60k-2026-02-17.md`

## Test Strategy for Benchmark Tooling

- Deterministic smoke coverage runs by default:
  - `tests/test_benchmark_harness.py::test_benchmark_harness_smoke_generates_results`
- Heavy 60k benchmark test is opt-in and skipped by default:
  - `TE_RUN_HEAVY_BENCHMARK=1 uv run pytest tests/test_benchmark_harness.py::test_benchmark_harness_60k_opt_in`
