# Rust Backend Feasibility Analysis

## Executive Summary

This document analyzes the feasibility of implementing Rust-native embedding and vector storage backends for the Town Elder CLI, with a focus on zvec bindings and alternative solutions.

**Status**: zvec Rust bindings are **NOT AVAILABLE** for native use. The implementation should proceed with an in-memory backend for local development and plan for Python interop or a pure-Rust alternative.

## zvec Rust Binding Analysis

### Current State (as of February 2026)

zvec (by Alibaba) is a high-performance embedded vector database written in C++ that exposes Python bindings via PyO3. However, there are **no official or community-maintained Rust bindings** available.

### What Works

- **Python bindings**: `pip install zvec` provides full Python API access
- **C++ native**: Can be used as a C++ library with proper build configuration
- **Performance**: Benchmarks show 8000+ QPS on standard datasets

### What Blocks Full Native Parity

| Gap | Severity | Description |
|-----|----------|-------------|
| No Rust Crate | **BLOCKING** | No `crates.io` package for zvec |
| No FFI Bindings | **BLOCKING** | No `bindgen`-generated bindings available |
| C++ Build Complexity | **HIGH** | Requires building Proxima engine from source |
| No Async Support | **MEDIUM** | Synchronous API only |

### Recommended Interim Strategy

1. **Use in-memory backend** (implemented in `te-core`) for local testing
2. **Python interop**: Use `te-core` Python bindings to call zvec via PyO3
3. **Plan for alternatives**: Consider pure-Rust alternatives (see below)

## Pure-Rust Vector Store Alternatives

For a fully native Rust implementation, the following alternatives are available:

### 1. Meilisearch (HTTP Client)

- **Type**: REST API server (not embedded)
- **Pros**: Fast, Rust-native, excellent full-text search
- **Cons**: Requires separate process, not in-memory

### 2. Qdrant (HTTP Client)

- **Type**: REST API server with Rust client
- **Pros**: Production-ready, filtering, quantization
- **Cons**: Requires separate process

### 3. LanceDB (Python with Rust Core)

- **Type**: Embedded (Python-first)
- **Pros**: Rust core, disk-backed, multi-modal
- **Cons**: Python bindings primary, limited Rust API

### 4. Greased (Pure Rust)

- **Type**: In-progress embedded vector search
- **Pros**: Pure Rust, no external dependencies
- **Cons**: Early stage, not production-ready

### 5. USearch

- **Type**: C++ with bindings
- **Pros**: High performance, multiple languages
- **Cons**: No native Rust crate

## Implementation Completed (This Ticket)

### Implemented Components

The following backend abstraction components have been implemented in `rust/crates/te-core/src/lib.rs`:

#### Traits

| Trait | Description |
|-------|-------------|
| `Embedder` | Text to vector embedding conversion |
| `VectorStore` | Vector storage and retrieval with metadata filtering |
| `EmbeddingBackend` | Combined embedder + vector store interface |

#### Types

| Type | Description |
|------|-------------|
| `BackendError` | Error enum for backend operations |
| `Embedding` | Vector with metadata |
| `Document` | Content + ID + metadata |
| `SearchResult` | Retrieved document with score |
| `BackendConfig` | Configuration for backend initialization |

#### Implementations

| Implementation | Description |
|----------------|-------------|
| `InMemoryEmbedder` | Random embedding generator for testing (dimensions configurable) |
| `InMemoryVectorStore` | Hashmap-based storage with cosine similarity search |
| `InMemoryBackend` | Combined embedder + store for convenient local testing |

#### Test Coverage

All new components include comprehensive tests:
- Embedder dimension and embedding generation tests
- Vector store add, search, delete, and filter tests
- Backend integration tests
- 70 tests passing total

## Implementation Decision

### Recommended Path Forward

```
┌─────────────────────────────────────────────────────────────┐
│                    Town Elder CLI                            │
├─────────────────────────────────────────────────────────────┤
│  te-core (Rust)                                            │
│    ├── InMemoryVectorStore  ← Current: Local testing       │
│    ├── InMemoryEmbedder     ← Current: Mock embeddings     │
│    └── Python Interop       ← Future: Call zvec via PyO3   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Option A: Python Runtime (Recommended for Prototype)       │
│    └── zvec via PyO3 bridge                                │
│                                                              │
│  Option B: Pure Rust (Future)                               │
│    └── USearch or custom SIMD implementation               │
└─────────────────────────────────────────────────────────────┘
```

### Gap List Summary

| Priority | Gap | Action |
|----------|-----|--------|
| P0 | No zvec Rust bindings | Use Python interop |
| P1 | Need persistence | Implement file-based serialization for InMemory store |
| P2 | Need real embeddings | Add ONNX Runtime Rust bindings |
| P3 | Production scale | Evaluate Qdrant/Meilisearch |

## Documentation Updates

### Changes to zvec-embedding-db.md

The existing research document should be updated to note:

1. zvec is **not available** for direct Rust usage
2. Python interop is the recommended path for prototype
3. In-memory backend is suitable for CLI testing

## Conclusion

The Rust backend abstraction traits in `te-core` are designed to be backend-agnostic. The in-memory implementation provides a working local path for development and testing. For production use with zvec, the CLI should be designed to optionally delegate to Python via PyO3 when zvec functionality is required.

---

*Document generated: February 2026*
*Related issue: te-9ex*
