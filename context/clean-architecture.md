# Clean Architecture Design for te

## Project Overview

**te** is a Python CLI tool that provides semantic memory capabilities using zvec (embedded vector database) and fastembed (embedding generation). It supports two primary flows: **index** (ingesting documents with embeddings) and **query** (semantic search).

## Architecture Principles

1. **Dependency Inward** - Dependencies point toward the core domain. Infrastructure depends on domain, never the reverse.
2. **Interface Segregation** - Define clear contracts between layers.
3. **Single Responsibility** - Each module has one reason to change.
4. **Testability** - Core domain logic is testable without external dependencies.

---

## Layer Separation

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Layer (Presentation)                 │
│                 typer commands, argument parsing            │
├─────────────────────────────────────────────────────────────┤
│                   Application Layer                         │
│            use cases, orchestration, DTOs                   │
├─────────────────────────────────────────────────────────────┤
│                      Domain Layer                            │
│          entities, value objects, domain services           │
├─────────────────────────────────────────────────────────────┤
│                  Infrastructure Layer                       │
│     zvec repository, embedding service, config management   │
└─────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
town_elder/
├── src/town_elder/
│   ├── __init__.py
│   ├── cli/                      # CLI Layer
│   │   ├── __init__.py
│   │   ├── app.py                # Typer application entry point
│   │   ├── commands/
│   │   │   ├── __init__.py
│   │   │   ├── index.py          # "te index" command
│   │   │   ├── query.py          # "te query" command
│   │   │   └── status.py         # "te status" command
│   │   └── options.py            # Shared CLI options
│   │
│   ├── application/              # Application Layer
│   │   ├── __init__.py
│   │   ├── dtos/                # Data Transfer Objects
│   │   │   ├── __init__.py
│   │   │   ├── document_dto.py
│   │   │   └── search_result_dto.py
│   │   ├── ports/               # Port interfaces (abstract)
│   │   │   ├── __init__.py
│   │   │   ├── vector_store.py  # Interface for vector storage
│   │   │   ├── embedding_service.py  # Interface for embeddings
│   │   │   └── config_service.py     # Interface for config
│   │   └── services/            # Application services
│   │       ├── __init__.py
│   │       ├── index_service.py
│   │       └── query_service.py
│   │
│   ├── domain/                  # Domain Layer
│   │   ├── __init__.py
│   │   ├── entities/
│   │   │   ├── __init__.py
│   │   │   ├── document.py      # Document entity
│   │   │   └── search_result.py # Search result entity
│   │   ├── value_objects/
│   │   │   ├── __init__.py
│   │   │   ├── vector.py        # Vector value object
│   │   │   └── document_id.py    # Document ID value object
│   │   └── exceptions/
│   │       ├── __init__.py
│   │       ├── document_not_found.py
│   │       ├── indexing_error.py
│   │       └── query_error.py
│   │
│   └── infrastructure/          # Infrastructure Layer
│       ├── __init__.py
│       ├── repositories/
│       │   ├── __init__.py
│       │   └── zvec_repository.py   # zvec implementation
│       ├── services/
│       │   ├── __init__.py
│       │   ├── fastembed_service.py  # fastembed implementation
│       │   └── config_service.py     # Config implementation
│       └── factories/
│           ├── __init__.py
│           └── service_factory.py   # DI container/factory
│
├── pyproject.toml
└── README.md
```

---

## Dependency Injection Pattern

### Port Interfaces (in `application/ports/`)

```python
# application/ports/vector_store.py
from abc import ABC, abstractmethod
from typing import List
from town_elder.application.dtos import SearchResultDTO

class VectorStorePort(ABC):
    """Port interface for vector storage operations."""

    @abstractmethod
    def insert(self, documents: List[Document]) -> None:
        """Insert documents into the vector store."""
        pass

    @abstractmethod
    def search(self, query_vector: List[float], top_k: int) -> List[SearchResult]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def delete(self, document_ids: List[str]) -> None:
        """Delete documents by ID."""
        pass


# application/ports/embedding_service.py
from abc import ABC, abstractmethod
from typing import List

class EmbeddingServicePort(ABC):
    """Port interface for embedding generation."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        pass
```

### Infrastructure Implementations

```python
# infrastructure/services/fastembed_service.py
from town_elder.application.ports.embedding_service import EmbeddingServicePort

class FastEmbedService(EmbeddingServicePort):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self._model = None
        self._model_name = model_name

    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [e.tolist() for e in self.model.embed(texts)]

    def get_dimension(self) -> int:
        return 384  # bge-small-en-v1.5 dimension


# infrastructure/repositories/zvec_repository.py
from town_elder.application.ports.vector_store import VectorStorePort

class ZvecRepository(VectorStorePort):
    def __init__(self, collection_path: str, dimension: int):
        import zvec
        schema = zvec.CollectionSchema(
            name="te",
            vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, dimension)
        )
        self._collection = zvec.create_and_open(path=collection_path, schema=schema)

    def insert(self, documents: List[Document]) -> None:
        docs = [
            zvec.Doc(
                id=doc.id.value,
                vectors={"embedding": doc.vector.value},
                fields={"text": doc.text, "metadata": doc.metadata}
            )
            for doc in documents
        ]
        self._collection.insert(docs)

    def search(self, query_vector: List[float], top_k: int) -> List[SearchResult]:
        results = self._collection.query(
            zvec.VectorQuery("embedding", vector=query_vector),
            topk=top_k
        )
        return [SearchResult(id=r.id, score=r.score, text=r.fields["text"]) for r in results]

    def delete(self, document_ids: List[str]) -> None:
        for doc_id in document_ids:
            self._collection.delete(doc_id)
```

### Service Factory (Simple DI Container)

```python
# infrastructure/factories/service_factory.py
from dataclasses import dataclass

@dataclass
class Services:
    vector_store: VectorStorePort
    embedding_service: EmbeddingServicePort
    config: ConfigServicePort

def create_services(config_path: str = None) -> Services:
    config = ConfigService(config_path)
    embedding_service = FastEmbedService(model_name=config.embedding_model)
    vector_store = ZvecRepository(
        collection_path=config.collection_path,
        dimension=embedding_service.get_dimension()
    )
    return Services(
        vector_store=vector_store,
        embedding_service=embedding_service,
        config=config
    )
```

---

## Error Handling Strategy

### Domain Exceptions

```python
# domain/exceptions/__init__.py

class TeError(Exception):
    """Base exception for all te errors."""
    pass

class DocumentNotFoundError(TeError):
    """Raised when a document cannot be found."""
    def __init__(self, document_id: str):
        self.document_id = document_id
        super().__init__(f"Document not found: {document_id}")

class IndexingError(TeError):
    """Raised when indexing fails."""
    pass

class QueryError(TeError):
    """Raised when search fails."""
    pass
```

### Error Handling in CLI

```python
# cli/app.py
import typer
from town_elder.domain.exceptions import TeError

app = typer.Typer()

@app.exception_handler(TeError)
def te_exception_handler(request: typer.Request, exc: TeError):
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(code=1)
```

---

## Extensibility Points

### 1. Different Embedding Models

To swap embedding models, implement a new `EmbeddingServicePort`:

```python
# For local models (Model2Vec)
class Model2VecService(EmbeddingServicePort):
    def __init__(self, model_name: str = "minishlab/potion-base-8M"):
        from model2vec import StaticModel
        self._model = StaticModel.from_pretrained(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._model.encode(text).tolist() for text in texts]

    def get_dimension(self) -> int:
        return 768  # potion-base-8M dimension
```

Configure via settings:
```yaml
# config.yaml
embedding:
  provider: fastembed  # or "model2vec", "llama-cpp"
  model: BAAI/bge-small-en-v1.5
```

### 2. Different Storage Backends

To support alternative vector stores (Chroma, LanceDB), implement a new `VectorStorePort`:

```python
# For Chroma
class ChromaRepository(VectorStorePort):
    def __init__(self, collection_name: str, persist_directory: str):
        import chromadb
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(collection_name)

    def insert(self, documents: List[Document]) -> None:
        # Chroma implementation
        pass
```

Configure via settings:
```yaml
# config.yaml
storage:
  provider: zvec  # or "chroma", "lancedb"
  path: .te/vectors
```

### 3. Adding New Commands

Extend CLI by adding new command modules:

```
cli/commands/
    ├── index.py       # "te index"
    ├── query.py       # "te query"
    ├── delete.py      # "te delete"  (new)
    └── stats.py       # "te stats"  (new)
```

Register in `app.py`:
```python
app.add_typer(commands.index.app, name="index")
app.add_typer(commands.query.app, name="query")
app.add_typer(commands.delete.app, name="delete")  # new
```

---

## Domain Entities

### Document Entity

```python
# domain/entities/document.py
from dataclasses import dataclass
from typing import Dict, Any, Optional

from town_elder.domain.value_objects import DocumentId, Vector

@dataclass
class Document:
    id: DocumentId
    text: str
    vector: Vector
    metadata: Dict[str, Any]

    @classmethod
    def create(cls, text: str, metadata: Optional[Dict[str, Any]] = None) -> "Document":
        doc_id = DocumentId.generate()
        return cls(
            id=doc_id,
            text=text,
            vector=None,  # Set by application layer
            metadata=metadata or {}
        )
```

### Value Objects

```python
# domain/value_objects/vector.py
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class Vector:
    value: List[float]

    def __post_init__(self):
        if not self.value:
            raise ValueError("Vector cannot be empty")

    @property
    def dimension(self) -> int:
        return len(self.value)
```

---

## Application Services

### Index Service

```python
# application/services/index_service.py
class IndexService:
    def __init__(
        self,
        vector_store: VectorStorePort,
        embedding_service: EmbeddingServicePort
    ):
        self._vector_store = vector_store
        self._embedding_service = embedding_service

    def index_documents(self, texts: List[str], metadata: List[Dict]) -> List[str]:
        # 1. Generate embeddings
        vectors = self._embedding_service.embed(texts)

        # 2. Create domain entities
        documents = [
            Document(
                id=DocumentId.generate(),
                text=text,
                vector=Vector(vector),
                metadata=meta
            )
            for text, vector, meta in zip(texts, vectors, metadata)
        ]

        # 3. Persist to vector store
        self._vector_store.insert(documents)

        return [doc.id.value for doc in documents]
```

### Query Service

```python
# application/services/query_service.py
class QueryService:
    def __init__(
        self,
        vector_store: VectorStorePort,
        embedding_service: EmbeddingServicePort
    ):
        self._vector_store = vector_store
        self._embedding_service = embedding_service

    def search(self, query: str, top_k: int = 5) -> List[SearchResultDTO]:
        # 1. Embed query
        query_vector = self._embedding_service.embed([query])[0]

        # 2. Search vector store
        results = self._vector_store.search(query_vector, top_k)

        # 3. Map to DTOs
        return [
            SearchResultDTO(
                id=result.id,
                score=result.score,
                text=result.text,
                metadata=result.metadata
            )
            for result in results
        ]
```

---

## CLI Commands

### Index Command

```python
# cli/commands/index.py
import typer
from typing import Optional, List
from pathlib import Path

app = typer.Typer()

@app.command()
def index(
    path: Path = typer.Option(..., help="Path to text file or directory"),
    batch_size: int = typer.Option(32, help="Batch size for embedding"),
    metadata_file: Optional[Path] = typer.Option(None, help="JSON file with metadata"),
):
    """Index documents for semantic search."""
    from town_elder.infrastructure.factories.service_factory import create_services

    services = create_services()

    # Read documents
    texts = read_documents(path)
    metadata = load_metadata(metadata_file)

    # Index via application service
    index_service = IndexService(
        vector_store=services.vector_store,
        embedding_service=services.embedding_service
    )
    doc_ids = index_service.index_documents(texts, metadata)

    typer.echo(f"Indexed {len(doc_ids)} documents")
```

### Query Command

```python
# cli/commands/query.py
import typer

app = typer.Typer()

@app.command()
def query(
    text: str = typer.Argument(..., help="Query text"),
    top_k: int = typer.Option(5, help="Number of results"),
):
    """Search indexed documents."""
    from town_elder.infrastructure.factories.service_factory import create_services

    services = create_services()

    query_service = QueryService(
        vector_store=services.vector_store,
        embedding_service=services.embedding_service
    )
    results = query_service.search(text, top_k)

    for r in results:
        typer.echo(f"[{r.score:.3f}] {r.text[:100]}...")
```

---

## Configuration

```yaml
# .te/config.yaml
te:
  storage:
    provider: zvec
    path: .te/vectors.db

  embedding:
    provider: fastembed
    model: BAAI/bge-small-en-v1.5
    batch_size: 32

  search:
    default_top_k: 5
```

---

## Summary

| Layer | Responsibility | Dependencies |
|-------|---------------|--------------|
| CLI | User interaction, argument parsing | Application services |
| Application | Use cases, orchestration, DTOs | Domain, Ports |
| Domain | Entities, value objects, exceptions | None (pure Python) |
| Infrastructure | External integrations (zvec, fastembed) | Domain interfaces |

This architecture ensures:
- **Testability**: Domain logic has no external dependencies
- **Extensibility**: Swap implementations via ports
- **Maintainability**: Clear separation of concerns
- **Local-First**: All data stays on the machine
