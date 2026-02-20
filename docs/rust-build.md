# Rust Build and Development Guide

This document covers building the Rust components of Town Elder.

## Prerequisites

- Rust toolchain (via rustup): https://rustup.rs/
- Python 3.10-3.12
- maturin (for PyO3 builds): `pip install maturin`

## Project Structure

```
rust/
├── Cargo.toml              # Workspace root
├── src/
│   ├── lib.rs             # PyO3 module entrypoint
│   └── main.rs            # Clap binary entrypoint
└── crates/
    └── te-core/           # Shared core logic crate
        ├── Cargo.toml
        ├── build.rs       # PyO3 build config
        └── src/lib.rs     # Core implementation
```

## Build Commands

### Cargo Check (Fast Validation)

> **Note**: If using Python 3.14+, set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` to bypass version checks.

```bash
# Check the workspace compiles
cargo check --manifest-path rust/Cargo.toml

# If using Python 3.14+:
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo check --manifest-path rust/Cargo.toml

# Check a specific crate
cargo check -p te-core --manifest-path rust/Cargo.toml

# Check with all features
cargo check --all-features --manifest-path rust/Cargo.toml
```

### Cargo Build

```bash
# Build the workspace (debug)
cargo build --manifest-path rust/Cargo.toml

# Build the workspace (release)
cargo build --release --manifest-path rust/Cargo.toml

# Build a specific binary
cargo build --bin te --manifest-path rust/Cargo.toml
```

### Running the CLI

```bash
# Run the native Rust CLI binary
cargo run --manifest-path rust/Cargo.toml --bin te-rs -- --help

# Run with a subcommand
cargo run --manifest-path rust/Cargo.toml --bin te-rs -- health
cargo run --manifest-path rust/Cargo.toml --bin te-rs -- version
cargo run --manifest-path rust/Cargo.toml --bin te-rs -- index files . --query "deprecated"
```

### Running Tests

```bash
# Run Rust tests
cargo test --manifest-path rust/Cargo.toml
```

## PyO3 Extension Module Build

### Using maturin (Recommended)

```bash
# Install maturin
pip install maturin

# Build the Python extension in-place
cd rust
maturin develop

# Or build a wheel
maturin build --release
```

### Building with Embedding Support

The Rust extension includes text embedding support via fastembed. To build with embedding support:

```bash
# Build wheel with embedding features
maturin build --release --features python --manifest-path rust/src/Cargo.toml

# Or develop in-place with embedding support
maturin develop --features python --manifest-path rust/src/Cargo.toml
```

The `python` feature includes `te-core/fastembed` which brings in the embedding runtime.

### Creating Distributable Wheels

To create wheels for distribution across platforms:

```bash
# Build for current platform
maturin build --release --features python --out dist --manifest-path rust/src/Cargo.toml

# List built wheels
ls -la dist/*.whl
```

Wheels can be installed with pip:
```bash
pip install dist/*.whl
```

### Manual PyO3 Build

```bash
# Set Python path
export PYTHON_SYS_EXECUTABLE=$(which python3)

# Build using maturin
 maturin develop --manifest-path rust/src/Cargo.toml
```

## Python Integration

After building the PyO3 extension, it can be imported in Python:

```python
import town_elder._te_core as te_core

print(te_core.version())      # "0.1.0-scaffold"
print(te_core.health())      # "te-core: OK"
print(te_core.placeholder_fn())  # 42
```

### Text Embeddings

With embedding support enabled:

```python
import town_elder._te_core as te_core

# List available models (returns list of tuples: (model_id, dimension))
models = te_core.PyTextEmbedder.list_supported_models()
print(f"Available models: {models}")

# Select first available model (recommended - avoids hard-coded model names)
model_id = models[0][0] if models else None
if not model_id:
    raise RuntimeError("No embedding models available")

# Create an embedder with the selected model
embedder = te_core.PyTextEmbedder(model_id)
print(f"Model: {embedder.get_model_name()}, Dimension: {embedder.dimension()}")

# Embed text
embedding = embedder.embed_single("Hello, world!")
print(f"Embedding: {embedding[:5]}...")  # [0.1, 0.05, -0.02, ...]
```

> **Note**: Model names returned by `list_supported_models()` are guaranteed to be valid at runtime.
> Avoid hard-coding model IDs—always use the dynamically returned list model to ensure compatibility.

## Development Notes

- The PyO3 extension uses the `extension-module` feature to ensure
  the extension works correctly when loaded by the Python interpreter.
- The `te-core` crate is designed to be compiled as:
  - A native Rust library (`lib`)
  - A Python extension (`cdylib`)
  - A static library for embedding (`staticlib`)
- Keep the module surface minimal in this scaffolding phase;
  implementation tickets will expand functionality.
