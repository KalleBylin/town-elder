# Release Artifacts

Town Elder now publishes Python and Rust artifacts from the same source tree:

- Python package (`town-elder`) via `pyproject.toml`
- Rust extension wheel (`town_elder._te_core`) via `maturin` + `rust/src/Cargo.toml`
- Native CLI binary (`te-rs`) via `cargo build --bin te-rs`

## CI Workflows

- `.github/workflows/publish.yml`: Python sdist/wheel publish pipeline
- `.github/workflows/rust-artifacts.yml`: Rust wheel + native binary build artifacts

## Artifact Naming

Rust artifact workflow uploads:

- `te-core-wheel-<os>` containing `*.whl`
- `te-rs-<os>` containing `te-rs-<OS>-<ARCH>.tar.gz`

## Local Build Commands

```bash
# Build/install extension into current virtualenv
uv run maturin develop --manifest-path rust/src/Cargo.toml

# Build wheel
uv run maturin build --manifest-path rust/src/Cargo.toml --release

# Build native CLI binary
cargo build --manifest-path rust/Cargo.toml --release --bin te-rs
```

## Python-Only Install Path

Python-only usage remains supported when the Rust extension is absent:

- Install and run with `uv sync` and `uv run te ...`
- Rust acceleration stays opt-in behind `TE_USE_RUST_CORE`
- If `_te_core` is unavailable, adapter boundaries fall back to pure-Python implementations
