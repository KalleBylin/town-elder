# Rust Rollout Plan

This document defines rollout gates for enabling Rust-backed paths in Town Elder.

## Current Recommendation

- Default state: **default-off** (`TE_USE_RUST_CORE` opt-in)
- Enablement path: **opt-in** while parity and performance gates are tracked in CI

## Rollout Modes

1. `default-off`
- Rust code paths are available but disabled unless `TE_USE_RUST_CORE=1`.
- Python fallback must remain fully functional.

2. `opt-in`
- Teams can enable Rust path in selected environments.
- Benchmark + parity evidence required for each release candidate.

3. `default-on`
- Requires sustained parity and performance wins across representative repos.

## Decision Gates

### Functional parity

All of the following must pass:

- Rust/Python parity tests (`tests/test_rust_core_parity.py`)
- Adapter + fallback safety tests (`tests/test_rust_adapter.py`)
- Indexing regressions (`tests/test_index_files_incremental_regressions.py`)

### Performance gate

Run reproducible benchmarks:

```bash
PYTHONPATH=src python scripts/benchmark_indexing.py \
  --files 60000 \
  --workers 8 \
  --batch-size 256 \
  --output-json docs/benchmarks/rust-rollout-<date>.json \
  --output-md docs/benchmarks/rust-rollout-<date>.md
```

Required comparisons in output:

- index files (`optimized` vs `rust_enabled_files`)
- index commits prep (`comparisons.index_commits`)
- query baseline (`comparisons.query_baseline`)

### Suggested promotion thresholds

Move from `default-off` to `opt-in` when:

- No parity regressions in CI for 2 consecutive releases
- Rust-enabled file indexing throughput is not worse than Python baseline by more than 5%

Move from `opt-in` to `default-on` when:

- No parity regressions for 4 consecutive releases
- Rust-enabled throughput is >= Python baseline on index files and index commits prep
- Query baseline remains within +/-5% of Python path

## Fallback guarantee

If Rust extension is missing or disabled, all adapter entry points must continue to use Python implementations without user-visible failures.
