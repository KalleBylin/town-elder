# Replay Project - Comprehensive Priority List

Generated from research documents in /Users/bylin/Code/replay/context/

---

## MUST-HAVE (Phase 1: Core Value)

### 1. Basic Infrastructure
| Priority | Item | Source Document | Dependencies |
|----------|------|----------------|-------------|
| P0 | CLI framework setup (typer) | python-implementation.md | None |
| P0 | Project structure with clean architecture | python-implementation.md, clean-architecture.md | None |
| P0 | Configuration management (pydantic) | python-implementation.md, clean-architecture.md | None |
| P0 | Python 3.10+ support | python-implementation.md | None |

### 2. Core Semantic Indexing
| Priority | Item | Source Document | Dependencies |
|----------|------|----------------|-------------|
| P0 | zvec integration for vector storage | architecture.md, rag-implementation.md | None |
| P0 | fastembed integration for embeddings | architecture.md, rag-implementation.md | None |
| P0 | Embedding model: BAAI/bge-small-en-v1.5 (384 dimensions) | rag-implementation.md | zvec |
| P0 | Git diff parsing | architecture.md | Git |
| P0 | Code chunking (basic line-based) | rag-implementation.md | None |
| P0 | `replay init` command | cli-design.md | None |
| P0 | `replay index` command | cli-design.md, architecture.md | zvec, fastembed |
| P0 | `replay query` command | cli-design.md, architecture.md | zvec, fastembed |
| P0 | `replay status` command | cli-design.md | None |

### 3. Git Integration
| Priority | Item | Source Document | Dependencies |
|----------|------|----------------|-------------|
| P0 | Post-commit hook installation | cli-design.md, architecture.md | Git |
| P0 | Commit range indexing | architecture.md | Git |
| P0 | Incremental indexing | architecture.md | zvec |

### 4. Basic Retrieval
| Priority | Item | Source Document | Dependencies |
|----------|------|----------------|-------------|
| P0 | Query embedding generation | rag-implementation.md | fastembed |
| P0 | Vector similarity search | rag-implementation.md | zvec |
| P0 | Return commit messages + diffs | product-direction.md | None |
| P0 | Basic filtering (date, author, file) | cli-design.md | None |

### 5. Data Models
| Priority | Item | Source Document | Dependencies |
|----------|------|----------------|-------------|
| P0 | Document entity | clean-architecture.md, architecture.md | None |
| P0 | Chunk entity (code chunk) | architecture.md | None |
| P0 | Commit entity | architecture.md | None |
| P0 | SearchResult entity | clean-architecture.md | None |

---

## SHOULD-HAVE (Phase 2: Enhanced Features)

### 1. Advanced Indexing
| Priority | Item | Source Document | Dependencies |
|----------|------|----------------|-------------|
| P1 | Tree-sitter based code chunking | rag-implementation.md, devex-analysis.md | tree-sitter |
| P1 | Chunking at function/class boundaries (AST) | rag-implementation.md | tree-sitter |
| P1 | Diff chunking by hunk | rag-implementation.md | None |
| P1 | Hybrid search (BM25 + semantic) | rag-implementation.md | rank-bm25 |
| P1 | Metadata filtering (author, date, file pattern) | rag-implementation.md | zvec filters |

### 2. CLI Enhancements
| Priority | Item | Source Document | Dependencies |
|----------|------|----------------|-------------|
| P1 | Output formats (text, json, compact) | cli-design.md | rich |
| P1 | Shell completions (bash, zsh, fish) | cli-design.md | typer |
| P1 | Verbose/quiet flags | cli-design.md | None |
| P1 | Config file support (replay.yaml, .replay.yaml, ~/.config/replay/) | cli-design.md | None |

### 3. Error Handling
| Priority | Item | Source Document | Dependencies |
|----------|------|----------------|-------------|
| P1 | Custom exception types | clean-architecture.md, python-implementation.md | None |
| P1 | User-friendly error messages | cli-design.md | None |
| P1 | Exit codes (0=success, 1=error, 2=invalid usage, etc.) | cli-design.md | None |

### 4. Testing Infrastructure
| Priority | Item | Source Document | Dependencies |
|----------|------|----------------|-------------|
| P1 | Unit tests (60-70% coverage target) | testing-strategy.md | pytest |
| P1 | Integration tests (20-30%) | testing-strategy.md | pytest |
| P1 | Mock fixtures for zvec, fastembed, git | testing-strategy.md | pytest-mock |
| P1 | Test project structure | testing-strategy.md | None |

### 5. Documentation
| Priority | Item | Source Document | Dependencies |
|----------|------|----------------|-------------|
| P1 | README.md | documentation-plan.md | None |
| P1 | Quick Start guide | documentation-plan.md | None |
| P1 | CLI reference | documentation-plan.md | None |

---

## NICE-TO-HAVE (Phase 3: Advanced Features)

### 1. Agent Integration
| Priority | Item | Source Document | Dependencies |
|----------|------|----------------|-------------|
| P2 | MCP server (Model Context Protocol) | devex-analysis.md, product-direction.md | None |
| P2 | Context formatting for agent prompts | product-direction.md | None |
| P2 | Simple API for agents | product-direction.md | None |

### 2. Advanced Search Features
| Priority | Item | Source Document | Dependencies |
|----------|------|----------------|-------------|
| P2 | Oscillation detection | devex-analysis.md, product-direction.md | None |
| P2 | Semantic blame | product-direction.md | None |
| P2 | Intent-based retrieval (commit message vs code) | rag-implementation.md | None |

### 3. Performance Optimizations
| Priority | Item | Source Document | Dependencies |
|----------|------|----------------|-------------|
| P2 | Background/async indexing | rag-implementation.md | threading |
| P2 | Batch processing optimization | rag-implementation.md | None |
| P2 | Embedding caching | rag-implementation.md | None |
| P2 | Query latency < 50ms target | devex-analysis.md | None |

### 4. Additional Embedding Models
| Priority | Item | Source Document | Dependencies |
|----------|------|----------------|-------------|
| P2 | Model2Vec support (faster) | rag-implementation.md | model2vec |
| P2 | BGE-base-en-v1.5 (higher accuracy) | rag-implementation.md | fastembed |
| P2 | Configurable embedding model | clean-architecture.md | None |

### 5. Extensibility
| Priority | Item | Source Document | Dependencies |
|----------|------|----------------|-------------|
| P2 | Alternative vector stores (Chroma, LanceDB) | clean-architecture.md | chromadb/lancedb |
| P2 | Alternative embedding providers | clean-architecture.md | None |

---

## DEPRECATED / OUT OF SCOPE

| Item | Reason |
|------|--------|
| Multi-repository indexing | Phase 2+ |
| Web interface / GUI | Future |
| Cloud hosting | Never - stay local-first |
| Semantic blame (initial version) | Future |
| Oscillation detection (initial version) | Future |

---

## RISK MITIGATIONS (Should address early)

| Risk | Mitigation | Source |
|------|------------|--------|
| zvec API changes | Design for swapability; version pinning | risks-analysis.md |
| Embedding quality on code | Benchmark on code datasets | risks-analysis.md |
| Model version drift | Version-locked models | risks-analysis.md |
| Post-commit hook performance | Background async indexing | risks-analysis.md |
| Cold start time | Lazy model loading | python-implementation.md |
| Large repo scaling | Incremental indexing, memory limits | risks-analysis.md |

---

## DEPENDENCY GRAPH

```
Core Dependencies:
- typer (CLI framework)
- zvec (vector storage)
- fastembed (embeddings)
- pydantic (configuration)
- rich (output formatting)

Optional:
- tree-sitter-languages (code chunking)
- rank-bm25 (hybrid search)
- pytest (testing)

Build:
- Python 3.10+
- setuptools/wheel
```

---

## SUCCESS METRICS (from devex-analysis.md)

| Metric | Target |
|--------|--------|
| Context Retrieval Relevance | Top-5 recall > 80% |
| Query Latency | < 50ms p95 |
| Index Throughput | > 1000 chunks/sec |
| Setup Time | < 5 minutes |

---

## SUMMARY BY DOCUMENT

### architecture.md (System Architecture)
- **Main decisions**: 4-layer clean architecture (CLI, Application, Domain, Infrastructure)
- **Technical requirements**: zvec, fastembed, typer, clean separation of concerns
- **Dependencies**: Domain layer has no external dependencies; Infrastructure depends on Domain

### cli-design.md (CLI Interface Design)
- **Main decisions**: Commands (init, query, index, status), Typer framework, local-first
- **Technical requirements**: Shell completions, config file precedence, exit codes
- **Dependencies**: Git repository context

### clean-architecture.md (Code Architecture)
- **Main decisions**: Dependency injection, port interfaces, domain entities
- **Technical requirements**: DTOs, domain exceptions, service factory
- **Dependencies**: Domain layer pure Python

### devex-analysis.md (Value Proposition)
- **Main decisions**: Value for AI coding agents, "context crisis" solution
- **Technical requirements**: MCP integration, loop detection
- **Dependencies**: Git hooks

### documentation-plan.md (Docs Plan)
- **Main decisions**: MkDocs with Material theme
- **Technical requirements**: Markdown source files
- **Dependencies**: None

### product-direction.md (Product Scope)
- **Main decisions**: MVP phases, target users (AI agents, developers)
- **Technical requirements**: zvec, fastembed, tree-sitter
- **Dependencies**: Git integration

### python-implementation.md (Implementation Details)
- **Main decisions**: Project structure, dependencies, Python 3.10+
- **Technical requirements**: typer, zvec, fastembed, pydantic, rich
- **Dependencies**: All external libraries

### rag-implementation.md (RAG System)
- **Main decisions**: Embedding strategy (BGE-small), chunking (tree-sitter), hybrid search
- **Technical requirements**: BM25 integration, metadata filtering
- **Dependencies**: zvec, fastembed, tree-sitter

### risks-analysis.md (Risk Mitigation)
- **Main decisions**: Identified risks, mitigation strategies
- **Technical requirements**: Testing, fallback mechanisms
- **Dependencies**: N/A

### testing-strategy.md (Testing Approach)
- **Main decisions**: Test pyramid (Unit 60-70%, Integration 20-30%, E2E 5-10%)
- **Technical requirements**: pytest, mocking strategy
- **Dependencies**: pytest, pytest-mock
