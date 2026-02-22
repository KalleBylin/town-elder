# Rust Embedding Distribution Strategy

**Decision:** te-5br
**Date:** 2026-02-22
**Status:** Approved

## Summary

The Rust te-core extension will be distributed as a **companion PyPI wheel** with optional dependency integration, allowing users to choose between Python fastembed (default) or Rust-backed embeddings.

## Chosen Distribution Model

### Primary: Companion Wheel via Optional Extra

The Rust extension (`town-elder-te-core`) is published as a separate PyPI package built with maturin. Users install it via:

```bash
# Default: Python fastembed (current behavior)
pip install town-elder

# Opt-in Rust embeddings
pip install town-elder[rust]
```

The `[rust]` extra will:
1. Declare a dependency on `town-elder-te-core` (the Rust wheel)
2. Enable `TE_USE_RUST_CORE=1` by default via pyproject.toml entry_points or post-install hook

### Fallback Semantics

| Scenario | Behavior |
|----------|----------|
| `town-elder` installed, no extra | Uses Python fastembed (requires `fastembed` in deps) |
| `town-elder[rust]` installed | Uses Rust backend if available, falls back to Python fastembed if not |
| `TE_USE_RUST_CORE=1` set manually | Attempts Rust backend, falls back to Python if unavailable |
| Rust extension unavailable | Graceful fallback to Python fastembed with warning |

## Compatibility Matrix

### Supported Platforms (Initial)

| OS | Architecture | Python 3.10 | Python 3.11 | Python 3.12 |
|----|--------------|-------------|-------------|-------------|
| macOS | x86_64 | ✅ | ✅ | ✅ |
| macOS | ARM64 (Apple Silicon) | ✅ | ✅ | ✅ |
| Linux | x86_64 (manylinux) | ✅ | ✅ | ✅ |
| Linux | ARM64 | ✅ | ✅ | ✅ |
| Windows | x64 | ✅ | ✅ | ✅ |

### Future Extensions (Post-MVP)

- Linux ARM32 (Raspberry Pi)
- musl-based Linux (Alpine)
- PyPy support

## Rollout Plan

### Phase 1: CI Build Infrastructure (te-i1a)

1. Configure maturin to build wheels for all platforms in the matrix
2. Publish to PyPI on release tags (e.g., `town-elder-te-core v0.2.1`)
3. CI workflow publishes wheels alongside Python package

### Phase 2: Package Integration (te-cvo)

1. Add `[rust]` extra to `town-elder` pyproject.toml
2. Add post-install hook or entry_point to set `TE_USE_RUST_CORE=1` when rust extra is used
3. Document the new installation option

### Phase 3: Diagnostics (te-bfh)

1. Improve error messages when Rust extension is missing
2. Add `te doctor` command to detect and report extension status
3. Emit clear diagnostic when auto backend falls back

### Phase 4: Cutover Evaluation (te-ah5v)

Metrics to track:
- Adoption rate of `[rust]` extra (via PyPI download stats)
- Support ticket volume related to embedding backend issues
- Performance benchmark results comparing Python vs Rust paths

## Cutover Criteria for Making fastembed Optional

fastembed will become optional (not a required dependency) when:

1. **Adoption threshold**: >50% of active users have installed `[rust]` extra for 2 consecutive releases
2. **Reliability threshold**: <1% of issues related to Rust extension availability for 4 consecutive releases
3. **Performance threshold**: Rust backend shows >=10% improvement on typical workloads

When cutover is approved:
- Move `fastembed` from `dependencies` to `optional-dependencies[python-embed]`
- Change default backend from "python" to "auto" (prefer Rust, fall back to Python)
- Update documentation to recommend `[rust]` extra

## Implementation Notes

### PyPI Package Name

- **Package:** `town-elder-te-core`
- **Import name:** `town_elder._te_core` (unchanged)
- **Build tool:** maturin

### Version Sync

The Rust extension version should track the Python package version:
- `town-elder v0.2.1` → `town-elder-te-core v0.2.1`
- Release process publishes both packages together

### Build Configuration

The Rust extension uses:
- `maturin` as build backend
- `extension-module` feature for PyO3
- `pyo3` crate with Python bindings
- `fastembed` crate for embedding support

## Alternative Considered

### Single Package (Bundled)

**Rejected** because:
- Adds complexity to Python package build process
- Requires managing platform-specific wheels inside Python package
- Harder to iterate on Rust extension independently
- PyPI doesn't natively support wheel bundling

### Pure Extra (No Auto-Enable)

**Rejected** because:
- Requires users to both install extra AND set env var
- Worse onboarding experience
- Lower adoption rate expected
