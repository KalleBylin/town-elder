# Python Implementation Plan: replay CLI

## Overview

The replay CLI is a local-first semantic memory tool for AI coding agents. It provides vector-based storage and retrieval using zvec and fastembed, with a typer-based command interface.

## Project Structure

```
replay/
├── replay/
│   ├── __init__.py          # Package version
│   ├── cli.py               # Typer CLI entry point
│   ├── commands/             # Command modules
│   │   ├── __init__.py
│   │   ├── init.py          # 'replay init' - Initialize database
│   │   ├── index.py         # 'replay index' - Index content
│   │   ├── search.py        # 'replay search' - Semantic search
│   │   ├── add.py           # 'replay add' - Add documents
│   │   ├── stats.py         # 'replay stats' - Show stats
│   │   └── export.py        # 'replay export' - Export data
│   ├── core/                # Core functionality
│   │   ├── __init__.py
│   │   ├── vector_store.py  # zvec wrapper
│   │   ├── embedder.py      # fastembed wrapper
│   │   └── config.py       # Configuration management
│   ├── models/              # Data models
│   │   ├── __init__.py
│   │   └── document.py      # Document model
│   └── utils/               # Utilities
│       ├── __init__.py
│       └── logger.py        # Logging setup
├── pyproject.toml           # Package configuration
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

## Dependencies

### requirements.txt

```
typer[all]>=0.12.0
zvec>=0.1.0
fastembed>=0.2.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
rich>=13.0.0
python-dotenv>=1.0.0
```

### pyproject.toml

```toml
[project]
name = "replay"
version = "0.1.0"
description = "Local-first semantic memory CLI for AI coding agents"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Developer", email = "dev@example.com"}
]
keywords = ["cli", "vector", "embeddings", "ai", "memory", "semantic"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "typer[all]>=0.12.0",
    "zvec>=0.1.0",
    "fastembed>=0.2.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "rich>=13.0.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "ruff>=0.3.0",
    "mypy>=1.0.0",
]

[project.scripts]
replay = "replay.cli:app"

[build-system]
requires = ["setuptools>=68.0.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
```

## CLI Entry Point

### /Users/bylin/Code/replay/replay/cli.py

```python
"""replay CLI - Main entry point."""
import typer
from rich.console import Console

from replay.commands import init, index, search, add, stats, export

app = typer.Typer(
    name="replay",
    help="Local-first semantic memory CLI for AI coding agents",
    add_completion=False,
)
console = Console()

# Register command groups
app.add_typer(init.app, name="init")
app.add_typer(index.app, name="index")
app.add_typer(search.app, name="search")
app.add_typer(add.app, name="add")
app.add_typer(stats.app, name="stats")
app.add_typer(export.app, name="export")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """replay - Semantic memory for AI agents."""
    if ctx.invoked_subcommand is None:
        console.print("[bold]replay[/bold] - Semantic memory CLI")
        console.print("Use --help for usage information")


if __name__ == "__main__":
    app()
```

## Module Design

### Core Vector Store (/Users/bylin/Code/replay/replay/core/vector_store.py)

```python
"""zvec wrapper for vector storage."""
from pathlib import Path
from typing import Protocol, runtime_checkable

import zvec


class VectorStore(Protocol):
    """Protocol for vector store implementations."""

    def insert(self, text: str, metadata: dict) -> str: ...
    def search(self, query: str, top_k: int) -> list[dict]: ...
    def delete(self, doc_id: str) -> None: ...


class ZvecStore:
    """zvec implementation of VectorStore."""

    DEFAULT_DIMENSION = 384  # bge-small-en-v1.5

    def __init__(self, path: str | Path, dimension: int = DEFAULT_DIMENSION):
        self.path = Path(path)
        self.dimension = dimension
        self._collection = self._init_collection()

    def _init_collection(self):
        """Initialize or open existing zvec collection."""
        schema = zvec.CollectionSchema(
            name="replay_store",
            vectors=zvec.VectorSchema(
                name="embedding",
                data_type=zvec.DataType.VECTOR_FP32,
                dimension=self.dimension,
            ),
        )
        return zvec.create_and_open(path=str(self.path), schema=schema)

    def insert(self, text: str, metadata: dict) -> str:
        """Insert a document with embedding."""
        # Implementation here
        pass

    def search(self, query: str, top_k: int) -> list[dict]:
        """Search for similar documents."""
        # Implementation here
        pass

    def delete(self, doc_id: str) -> None:
        """Delete a document by ID."""
        # Implementation here
        pass
```

### Embedder (/Users/bylin/Code/replay/replay/core/embedder.py)

```python
"""fastembed wrapper for text embeddings."""
from typing import Iterator

import numpy as np
from fastembed import TextEmbedding


class Embedder:
    """Wrapper around fastembed for generating embeddings."""

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model: TextEmbedding | None = None

    @property
    def model(self) -> TextEmbedding:
        """Lazy load the embedding model."""
        if self._model is None:
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> Iterator[np.ndarray]:
        """Generate embeddings for a list of texts."""
        return self.model.embed(texts)

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return list(next(self.embed([text])))

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return 384  # bge-small-en-v1.5
```

### Configuration (/Users/bylin/Code/replay/replay/core/config.py)

```python
"""Configuration management."""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ReplayConfig(BaseSettings):
    """Configuration for replay CLI."""

    model_config = SettingsConfigDict(
        env_prefix="REPLAY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database settings
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".replay")
    db_name: str = "replay.db"

    # Embedding settings
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dimension: int = 384

    # Search settings
    default_top_k: int = 5

    # Logging
    verbose: bool = False


@lru_cache
def get_config() -> ReplayConfig:
    """Get cached configuration instance."""
    return ReplayConfig()
```

### Document Model (/Users/bylin/Code/replay/replay/models/document.py)

```python
"""Document data model."""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Represents a document in the vector store."""

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {
        "frozen": True,
    }
```

## Command Implementations

### Init Command

```python
"""Initialize replay database."""
import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def init(
    path: str = typer.Option(".", help="Directory to initialize"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing"),
) -> None:
    """Initialize a replay database in the specified directory."""
    console.print(f"[green]Initialized replay database at {path}[/green]")
```

### Search Command

```python
"""Search command."""
import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
) -> None:
    """Search for similar documents."""
    console.print(f"[bold]Search results for:[/bold] {query}")
    # Implementation
```

## Error Handling Approach

### Error Types

```python
"""Error types for replay."""
from typing import Any


class ReplayError(Exception):
    """Base exception for replay."""

    pass


class DatabaseError(ReplayError):
    """Database-related errors."""

    pass


class EmbeddingError(ReplayError):
    """Embedding generation errors."""

    pass


class ConfigError(ReplayError):
    """Configuration errors."""

    pass
```

### Error Handling Strategy

1. **Use typer's exception handling** - Let typer handle CLI-level errors gracefully
2. **Wrap database operations** - Catch zvec errors and convert to ReplayError
3. **Validation with Pydantic** - Use pydantic for input validation
4. **Rich error messages** - Use rich for formatted error output
5. **Exit codes** - Return appropriate exit codes (0 success, 1 error)

## Type Hints Strategy

### Approach

1. **Use Python 3.10+ type hints** - Native syntax with `from __future__ import annotations`
2. **Pydantic for data models** - Automatic validation and serialization
3. **Protocol for abstractions** - Define VectorStore as Protocol for testability
4. **mypy for checking** - Strict mode with selective relaxations

### Example

```python
from __future__ import annotations

from typing import Iterator
import numpy as np

def search(
    query: str,
    top_k: int = 5,
    filters: dict[str, str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Search for documents."""
    ...
```

## Setup.py/pyproject.toml Configuration

### Key Configuration Points

1. **Package name**: `replay`
2. **Entry point**: `replay = "replay.cli:app"`
3. **Python version**: >=3.10
4. **Dependencies**: As specified above
5. **Build system**: setuptools with wheel

### Installation

```bash
# Development installation
pip install -e ".[dev]"

# Production installation
pip install .

# Install dependencies only
pip install -r requirements.txt
```

## Implementation Priority

1. **Phase 1** - Core infrastructure
   - Project setup (pyproject.toml, requirements.txt)
   - Configuration management
   - Embedder wrapper
   - Vector store wrapper

2. **Phase 2** - CLI commands
   - Init command
   - Add command
   - Search command

3. **Phase 3** - Advanced features
   - Stats command
   - Export command
   - Index command (bulk indexing)

4. **Phase 4** - Polish
   - Error handling improvements
   - Testing
   - Documentation

## Notes

- All paths should use `pathlib.Path` for cross-platform compatibility
- Use `rich` for formatted CLI output
- Lazy-load embedding model to minimize startup time
- Support both in-memory and file-based zvec storage
- Keep CLI simple; complex operations in separate commands
