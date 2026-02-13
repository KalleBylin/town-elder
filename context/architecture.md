# Replay Architecture

Replay is a semantic version control system that provides semantic search over git history. It indexes code changes and commit messages, enabling intent-based retrieval of historical context.

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                           CLI (Typer)                            │
│  Commands: index, query, status, init, hook-install             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Core Layer                               │
│  - Orchestration (index/query workflows)                        │
│  - Configuration management                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│    Indexing Pipeline    │     │   Retrieval Pipeline   │
│  - Git diff parsing     │     │  - Query embedding     │
│  - Code chunking        │     │  - Vector search       │
│  - Embedding generation │     │  - Result ranking      │
│  - Vector storage       │     │  - Result formatting   │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   zvec (Storage)       │     │   fastembed (Embedding) │
│   - Vector index       │     │   - ONNX inference      │
│   - Metadata fields   │     │   - Quantized models    │
└─────────────────────────┘     └─────────────────────────┘
```

## Component Breakdown

### 1. CLI Layer (`replay/cli.py`)

Entry point using Typer. Provides command-line interface.

**Commands:**
- `replay init` - Initialize replay storage in `.git/replay`
- `replay index [range]` - Index commits in range (default: HEAD~10..HEAD)
- `replay query <text>` - Semantic search over indexed history
- `replay status` - Show indexing status and statistics
- `replay hook install` - Install git post-commit hook
- `replay hook uninstall` - Remove git hook

### 2. Core Layer (`replay/core.py`)

Orchestrates indexing and retrieval workflows.

**Responsibilities:**
- Coordinate between CLI, indexing, and retrieval modules
- Manage configuration loading/merging
- Handle error propagation and logging

### 3. Indexing Pipeline (`replay/indexing/`)

**Submodules:**

- `parser.py` - Git diff parsing
  - `parse_diff(commit_hash)` - Extract file changes from commit
  - `split_into_chunks(diff)` - Chunk by function/class boundaries

- `embedder.py` - Embedding generation
  - `embed_texts(texts)` - Batch embed via fastembed
  - `embed_commit_message(msg, diff)` - Combine intent + implementation

- `storage.py` - Vector storage operations
  - `init_collection()` - Create zvec collection with schema
  - `upsert_documents(docs)` - Bulk insert/update vectors

### 4. Retrieval Pipeline (`replay/retrieval/`)

- `query_processor.py` - Query handling
  - `embed_query(text)` - Embed user query
  - `search(query_vector, top_k)` - Execute vector search

- `result_formatter.py` - Output formatting
  - `format_results(hits)` - Format as commit list, code snippets, etc.

### 5. Git Integration (`replay/git/`)

- `hook.py` - Post-commit hook management
  - `install_hook()` - Create `.git/hooks/post-commit`
  - `uninstall_hook()` - Remove hook
  - `get_changed_commits()` - Get commits since last index

- `runner.py` - Git command execution
  - `get_commit_range(start, end)` - Get commits in range
  - `get_diff(commit_hash)` - Get diff for commit

### 6. Configuration (`replay/config.py`)

- `Config` dataclass - All configuration options
- `load_config()` - Load from file + environment + defaults
- Config precedence: CLI args > env vars > config file > defaults

## Data Models

### Core Entities

```python
@dataclass
class Chunk:
    """A semantic chunk of code change."""
    id: str                      # SHA256 of content
    commit_hash: str             # Parent commit
    file_path: str               # File being changed
    chunk_type: str              # "addition", "deletion", "modification"
    content: str                 # The actual code chunk
    language: str                # Programming language
    function_name: Optional[str] # Containing function (if detected)
    start_line: int              # Line number in file
    end_line: int

@dataclass
class Commit:
    """A semantic commit record."""
    hash: str                    # Full commit hash
    short_hash: str              # Abbreviated hash (7 chars)
    message: str                 # Commit message (full)
    message_short: str           # First line of message
    author: str                  # Author name
    author_email: str            # Author email
    timestamp: datetime          # Commit timestamp
    changed_files: List[str]     # List of modified files

@dataclass
class IndexedDocument:
    """A document stored in the vector database."""
    id: str                      # Unique document ID
    commit_hash: str             # Associated commit
    doc_type: str                # "commit_message", "code_chunk"
    content: str                 # Text content to embed
    vector: List[float]          # Pre-computed embedding
    metadata: Dict[str, Any]     # Additional metadata
```

### zvec Schema

```python
collection_schema = {
    "name": "replay_index",
    "vectors": {
        "embedding": {
            "dtype": "float32",
            "dimension": 384,  # Matches BGE-small-en-v1.5
            "algorithm": "hnsw",  # Hierarchical Navigable Small World
            "metric": "cosine"    # Cosine similarity
        }
    },
    "fields": {
        "commit_hash": "string",
        "doc_type": "string",      # "commit_message" | "code_chunk"
        "file_path": "string",
        "content": "string",        # Original text for display
        "language": "string",
        "timestamp": "int64",
        "author": "string"
    }
}
```

## Storage Location

```
.git/
└── replay/
    ├── config.yaml          # Local configuration
    ├── index.zvec           # Vector database files
    ├── index.zvec.meta
    └── cache/               # Optional caches
        └── models/           # Downloaded embedding models
```

**Design Rationale:**
- `.git/replay` keeps data alongside version control
- Storage is automatically excluded from git (add `.git/replay` to `.gitignore`)
- Single directory simplifies backup and migration

## zvec Integration Details

### Initialization

```python
import zvec

def init_storage(storage_path: str, dimension: int = 384) -> zvec.Collection:
    """Initialize or open the zvec collection."""
    schema = zvec.CollectionSchema(
        name="replay_index",
        vectors=zvec.VectorSchema(
            name="embedding",
            dtype=zvec.DataType.VECTOR_FP32,
            dimension=dimension,
            algorithm=zvec.Algorithm.HNSW,
            metric=zvec.Metric.COSINE
        )
    )

    # Create or open the collection
    collection = zvec.create_and_open(
        path=storage_path,
        schema=schema
    )

    return collection
```

### Indexing

```python
def index_commits(commits: List[Commit], collection: zvec.Collection):
    """Index commits into zvec."""
    documents = []

    for commit in commits:
        # Index commit message (high importance for intent)
        msg_doc = zvec.Doc(
            id=f"msg:{commit.hash}",
            vectors={"embedding": embed(commit.message)},
            fields={
                "commit_hash": commit.hash,
                "doc_type": "commit_message",
                "file_path": "",
                "content": commit.message,
                "language": "en",
                "timestamp": int(commit.timestamp.timestamp()),
                "author": commit.author
            }
        )
        documents.append(msg_doc)

        # Index code chunks
        for chunk in get_chunks(commit):
            chunk_doc = zvec.Doc(
                id=f"chunk:{chunk.id}",
                vectors={"embedding": embed(chunk.content)},
                fields={
                    "commit_hash": commit.hash,
                    "doc_type": "code_chunk",
                    "file_path": chunk.file_path,
                    "content": chunk.content,
                    "language": chunk.language,
                    "timestamp": int(commit.timestamp.timestamp()),
                    "author": commit.author
                }
            )
            documents.append(chunk_doc)

    # Bulk insert for performance
    collection.insert(documents)
```

### Retrieval

```python
def semantic_search(query: str, collection: zvec.Collection, top_k: int = 10):
    """Execute semantic search."""
    query_vector = embed(query)

    results = collection.query(
        vector=zvec.VectorQuery(
            field="embedding",
            vector=query_vector,
            topk=top_k
        ),
        filter_expr=None,  # Optional: filter by author, file, date
        include_fields=["commit_hash", "doc_type", "content", "file_path"]
    )

    return results
```

## fastembed Integration Details

### Model Selection

| Model | Dimension | Size | Use Case |
|-------|-----------|------|----------|
| BAAI/bge-small-en-v1.5 | 384 | ~20MB | Default - balanced |
| BAAI/bge-base-en-v1.5 | 768 | ~80MB | Higher accuracy |
| intfloat/e5-small-v2 | 384 | ~20MB | Alternative |

### Usage Pattern

```python
from fastembed import TextEmbedding
from typing import List

class Embedder:
    """Lazy-loading embedder with caching."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            # Lazy load - only downloads on first use
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        # Returns generator, materialize to list
        embeddings = list(self.model.embed(texts))
        return [e.tolist() for e in embeddings]

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        return self.embed([query])[0]
```

### Hybrid Search Strategy

For code search, combine semantic similarity with keyword matching:

```python
def search_with_fallback(
    query: str,
    collection: zvec.Collection,
    file_filter: Optional[str] = None
):
    """Search with keyword fallback."""
    # Primary: semantic search
    semantic_results = collection.query(
        vector=embed_query(query),
        topk=20
    )

    # If results are poor, try keyword matching on content field
    if len(semantic_results.hits) == 0 or semantic_results.scores[0] < 0.3:
        keyword_results = collection.query(
            filter_expr=f'content CONTAINS "{query.split()[0]}"',
            topk=10
        )
        return merge_results(semantic_results, keyword_results)

    return semantic_results
```

## Git Hook Integration

### Post-Commit Hook

Location: `.git/hooks/post-commit`

```bash
#!/bin/bash
# Replay post-commit hook
# Triggers incremental indexing of new commits

replay index --incremental
```

### Installation

```python
def install_hook(repo_path: str = "."):
    """Install post-commit hook."""
    hook_path = Path(repo_path) / ".git" / "hooks" / "post-commit"
    hook_content = """#!/bin/bash
# Installed by Replay - Semantic Version Control
replay index --incremental 2>/dev/null || true
"""
    hook_path.write_text(hook_content)
    hook_path.chmod(0o755)

    # Add to .gitignore
    gitignore_path = Path(repo_path) / ".git" / "info" / "exclude"
    # (Or use .gitignore)
```

### Incremental Indexing

```python
def incremental_index(repo_path: str):
    """Index only new commits since last index."""
    last_indexed = get_last_indexed_commit()

    if last_indexed is None:
        # Full index
        commits = get_all_commits()
    else:
        # Incremental: commits after last indexed
        commits = get_commits_after(last_indexed)

    for commit in commits:
        index_commit(commit)

    update_last_indexed(commits[-1])
```

## Configuration Management

### Config Schema

```yaml
# .git/replay/config.yaml (or specified via --config)

storage:
  path: ".git/replay"        # Storage directory
  dimension: 384              # Vector dimension

embedding:
  model: "BAAI/bge-small-en-v1.5"
  batch_size: 32
  cache_dir: ".git/replay/cache/models"

indexing:
  chunk_size: 512            # Max tokens per chunk
  overlap: 50                 # Token overlap between chunks
  include_languages: ["py", "js", "ts", "go", "rs", "java"]
  exclude_patterns: ["*.test.*", "node_modules/*", "vendor/*"]

retrieval:
  default_top_k: 10
  min_similarity: 0.3
  include_types: ["commit_message", "code_chunk"]
  output_format: "json"       # json, table, plain

git:
  default_range: "HEAD~10..HEAD"
  hook_enabled: false
```

### Config Loading

```python
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class Config:
    storage_path: str = ".git/replay"
    dimension: int = 384
    model: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 32
    chunk_size: int = 512
    default_top_k: int = 10
    # ... etc

def load_config(config_path: Optional[Path] = None) -> Config:
    """Load config with precedence: CLI > env > file > defaults"""

    # 1. Start with defaults
    config = Config()

    # 2. Load from file if exists
    if config_path is None:
        config_path = Path(".git/replay/config.yaml")

    if config_path.exists():
        file_config = load_yaml(config_path)
        config = merge(config, file_config)

    # 3. Environment variables override
    if os.getenv("REPLAY_MODEL"):
        config.model = os.getenv("REPLAY_MODEL")
    if os.getenv("REPLAY_STORAGE_PATH"):
        config.storage_path = os.getenv("REPLAY_STORAGE_PATH")

    return config
```

## Project Structure

```
replay/
├── replay/
│   ├── __init__.py
│   ├── cli.py              # Typer CLI entry point
│   ├── core.py             # Core orchestration
│   ├── config.py           # Configuration management
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── parser.py       # Git diff parsing
│   │   ├── chunker.py      # Code chunking
│   │   ├── embedder.py     # fastembed wrapper
│   │   └── storage.py      # zvec operations
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── query.py        # Query processing
│   │   └── formatter.py    # Output formatting
│   ├── git/
│   │   ├── __init__.py
│   │   ├── runner.py       # Git command execution
│   │   └── hook.py         # Hook management
│   └── models.py           # Data classes
├── tests/
│   ├── test_indexing/
│   ├── test_retrieval/
│   └── test_integration/
├── pyproject.toml
├── README.md
└── .git/
    └── replay/              # Created at runtime
```

## Usage Examples

### Initial Setup

```bash
# Initialize in a git repository
cd my-project
replay init

# Install post-commit hook for automatic indexing
replay hook install

# Index initial history
replay index HEAD~100..HEAD
```

### Querying

```bash
# Semantic search
replay query "authentication retry logic"

# Filter by file
replay query "payment validation" --file "src/payment.py"

# Filter by author
replay query "bug fix" --author "alice"

# Show top 20 results
replay query "refactor" --top-k 20
```

### Output Format

```json
{
  "results": [
    {
      "commit": "a1b2c3d",
      "message": "Fix race condition in retry logic",
      "type": "commit_message",
      "score": 0.92,
      "file": "",
      "content": "Fix race condition in retry logic by adding..."
    },
    {
      "commit": "e4f5g6h",
      "message": "Add exponential backoff",
      "type": "code_chunk",
      "score": 0.87,
      "file": "src/auth.py",
      "content": "def retry_with_backoff(...):"
    }
  ]
}
```

## Future Considerations

- **Tree-sitter integration**: For better code chunking at AST boundaries
- **Multiple embeddings**: Support different models for code vs. messages
- **Incremental updates**: Efficient re-indexing on file rename/delete
- **Remote sync**: Optional sync to shared vector store (future)
- **MCP server**: Model Context Protocol integration for IDEs
