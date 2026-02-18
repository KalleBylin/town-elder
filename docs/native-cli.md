# Native Rust CLI (`te-rs`)

`te-rs` is the native Rust command surface backed by shared `te-core` logic.

## Build and Run

```bash
cargo run --manifest-path rust/Cargo.toml --bin te-rs -- --help
```

## Command Tree

- `te-rs health`
- `te-rs version`
- `te-rs index files <PATH> [--exclude ...] [--ext ...] [--query ...] [--top-k N]`

## End-to-End Index + Query Workflow

`index files` uses `te-core` for:

- file scanning (`scan_files`)
- chunk/document construction (`build_documents_from_files`)
- deterministic doc IDs (`build_file_doc_id`)
- in-memory embedding/vector search backend (`InMemoryBackend`)

Example:

```bash
cargo run --manifest-path rust/Cargo.toml --bin te-rs -- \
  index files . \
  --query "deprecated" \
  --top-k 5
```

The command indexes file chunks and immediately runs a query against the in-memory backend in the same process.

## Tests

Native CLI integration coverage lives in:

- `rust/src/tests/native_cli.rs`
