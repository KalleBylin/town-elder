# RAG Implementation Plan for Replay

This document outlines the implementation strategy for the Retrieval-Augmented Generation (RAG) system in Replay, providing semantic search over git history using zvec and FastEmbed.

## Overview

Replay's RAG system enables semantic search across git history, allowing AI coding agents to:
- Find relevant commits based on intent (not just keywords)
- Understand why code was changed (tribal knowledge)
- Search across diffs, commit messages, and file contents
- Filter by author, date, and file patterns

## 1. Embedding Strategy

### Model Selection

| Use Case | Recommended Model | Dimensions | Rationale |
|----------|-------------------|------------|-----------|
| Primary (default) | `BAAI/bge-small-en-v1.5` | 384 | Best accuracy/speed balance for code |
| High precision | `BAAI/bge-m3` | 1024 | Better multilingual/code understanding |
| Maximum speed | `minishlab/potion-base-8M` | Variable | Model2Vec for bulk indexing |

**Recommendation:** Start with `BAAI/bge-small-en-v1.5` via FastEmbed.

### Configuration

```python
from fastembed import TextEmbedding

# Primary embedding model
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# For batch operations (bulk indexing)
embed_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    max_length=512,  # Truncate longer texts
)
```

### Dimensionality

- **384 dimensions** (BGE-small) - Primary choice
- Schema definition in zvec:
```python
schema = zvec.CollectionSchema(
    name="git_history",
    vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, 384)
)
```

## 2. Chunking Strategy

### Chunks to Index

| Content Type | Chunking Approach | Target Size |
|--------------|-------------------|-------------|
| Commit messages | Full message | Single chunk |
| Diff hunks | By hunk (git diff sections) | 200-500 lines |
| File content | Tree-sitter by function/class | Per AST node |
| PR descriptions | Paragraph split | Natural paragraphs |

### Implementation: Tree-Sitter Based Chunking

```python
import tree_sitter_languages
from tree_sitter import Language, Parser

class CodeChunker:
    """Split code into semantic chunks using tree-sitter."""

    # Mapping of file extensions to language parsers
    LANGUAGE_PARSERS = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.rs': 'rust',
        '.go': 'go',
        '.java': 'java',
    }

    def chunk_file(self, file_path: str, content: str) -> list[CodeChunk]:
        """Split a source file into semantic chunks (functions, classes)."""
        ext = Path(file_path).suffix
        language = self.LANGUAGE_PARSERS.get(ext)

        if not language:
            return self.chunk_by_lines(content)  # Fallback

        parser = Parser(Language(tree_sitter_languages.language(language)))
        tree = parser.parse(bytes(content, 'utf8'))

        chunks = []
        self._extract_nodes(tree.root_node, content, chunks)
        return chunks

    def _extract_nodes(self, node, content: str, chunks: list):
        """Recursively extract semantic units (functions, classes)."""
        semantic_node_types = {'function_definition', 'class_definition', 'method_declaration'}

        if node.type in semantic_node_types:
            chunk = CodeChunk(
                text=node.text.decode('utf8'),
                node_type=node.type,
                start_line=node.start_point.row,
                end_line=node.end_point.row,
            )
            chunks.append(chunk)
        else:
            for child in node.children:
                self._extract_nodes(child, content, chunks)
```

### Fallback: Simple Line-Based Chunking

For files without tree-sitter support or very large files:

```python
def chunk_by_lines(content: str, chunk_size: int = 200) -> list[str]:
    """Simple line-based chunking as fallback."""
    lines = content.split('\n')
    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunk = '\n'.join(lines[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
```

### Diff Chunking

```python
class DiffChunker:
    """Chunk git diffs into semantic units."""

    def chunk_diff(self, diff_text: str) -> list[DiffChunk]:
        """Split a diff into individual file + hunk chunks."""
        chunks = []
        current_file = None

        for line in diff_text.split('\n'):
            # New file in diff
            if line.startswith('diff --git'):
                current_file = self._extract_filename(line)
            # New hunk
            elif line.startswith('@@'):
                hunk_info = self._parse_hunk_header(line)
                chunks.append(DiffChunk(
                    file_path=current_file,
                    hunk_header=line,
                    content='\n'.join([]),  # Accumulate context
                    change_type=hunk_info.get('operation'),
                ))
            # Accumulate content
            if chunks:
                chunks[-1].content += line + '\n'

        return chunks
```

## 3. Retrieval Pipeline

### Data Flow

```
Git Commit → Extract → Chunk → Embed → Store in zvec
                                    ↓
User Query → Embed → Search zvec → Rank → Return Results
```

### Implementation

```python
import zvec
from fastembed import TextEmbedding
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SearchResult:
    commit_hash: str
    file_path: str
    chunk_text: str
    score: float
    metadata: dict

class GitRAG:
    """RAG system for git history search."""

    def __init__(self, db_path: str, embedding_model: str = "BAAI/bge-small-en-v1.5"):
        # Initialize embedding model
        self.embed_model = TextEmbedding(model_name=embedding_model)

        # Initialize zvec
        schema = zvec.CollectionSchema(
            name="git_history",
            vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, 384),
            fields={
                "commit_hash": zvec.Field(zvec.DataType.STRING),
                "file_path": zvec.Field(zvec.DataType.STRING),
                "author": zvec.Field(zvec.DataType.STRING),
                "date": zvec.Field(zvec.DataType.INT64),  # Unix timestamp
                "message": zvec.Field(zvec.DataType.STRING),
                "chunk_type": zvec.Field(zvec.DataType.STRING),  # 'diff', 'commit', 'code'
            }
        )
        self.collection = zvec.create_and_open(path=db_path, schema=schema)

    def index_commit(self, commit: GitCommit):
        """Index a single commit and its changes."""
        chunks = self._create_chunks(commit)

        # Embed all chunks
        texts = [c['text'] for c in chunks]
        embeddings = list(self.embed_model.embed(texts))

        # Insert into zvec
        docs = []
        for chunk, embedding in zip(chunks, embeddings):
            doc = zvec.Doc(
                id=f"{commit.hash}:{chunk['file_path']}:{chunk.get('start_line', 0)}",
                vectors={"embedding": embedding.tolist()},
                fields={
                    "commit_hash": commit.hash,
                    "file_path": chunk['file_path'],
                    "author": commit.author,
                    "date": int(commit.timestamp),
                    "message": commit.message,
                    "chunk_type": chunk['type'],
                }
            )
            docs.append(doc)

        self.collection.insert(docs)

    def search(
        self,
        query: str,
        top_k: int = 10,
        author: Optional[str] = None,
        file_pattern: Optional[str] = None,
        date_from: Optional[int] = None,
        date_to: Optional[int] = None,
    ) -> List[SearchResult]:
        """Semantic search over git history."""

        # Generate query embedding
        query_embedding = list(self.embed_model.embed([query]))[0].tolist()

        # Build filter expression
        filters = []
        if author:
            filters.append(zvec.Filter("author") == author)
        if date_from:
            filters.append(zvec.Filter("date") >= date_from)
        if date_to:
            filters.append(zvec.Filter("date") <= date_to)

        # Execute search
        filter_expr = zvec.And(*filters) if filters else None

        results = self.collection.query(
            zvec.VectorQuery("embedding", vector=query_embedding),
            filter=filter_expr,
            topk=top_k,
        )

        # Process results
        return [
            SearchResult(
                commit_hash=r.fields['commit_hash'],
                file_path=r.fields['file_path'],
                chunk_text=r.fields.get('message', ''),
                score=r.score,
                metadata=r.fields,
            )
            for r in results
        ]
```

## 4. Metadata Filtering

### Supported Filters

| Filter | Field | Example |
|--------|-------|---------|
| Author | `author` | `"author == 'john@example.com'"` |
| Date range | `date` | `"date >= 1704067200 AND date <= 1706659200"` |
| File pattern | `file_path` | `"file_path LIKE '%.py'"` |
| Commit message | `message` | `"message CONTAINS 'fix'"` |

### Filter Implementation

```python
# Simple filter building
def build_filter(
    author: Optional[str] = None,
    date_from: Optional[int] = None,
    date_to: Optional[int] = None,
    file_pattern: Optional[str] = None,
) -> zvec.Filter:
    """Build composite filter for zvec queries."""

    conditions = []

    if author:
        conditions.append(zvec.Filter("author") == author)

    if date_from:
        conditions.append(zvec.Filter("date") >= date_from)

    if date_to:
        conditions.append(zvec.Filter("date") <= date_to)

    if file_pattern:
        # Simple glob-like matching via contains
        # zvec doesn't have native glob, use contains with wildcards
        conditions.append(zvec.Filter("file_path").contains(file_pattern.replace('*', '')))

    if not conditions:
        return None

    return zvec.And(*conditions)
```

### Common Filter Combinations

```python
# Find all changes by Alice in the last month
filter_andrew = build_filter(
    author="alice@example.com",
    date_from=int(time.time() - 30 * 24 * 60 * 60),
)

# Find Python files changed in Q1 2024
filter_q1 = build_filter(
    file_pattern="*.py",
    date_from=1704067200,  # Jan 1, 2024
    date_to=1706659200,    # Mar 31, 2024
)
```

## 5. Hybrid Search

Combine semantic (dense) search with keyword (sparse) search for better recall.

### Strategy: Two-Stage Retrieval

```python
class HybridSearcher:
    """Combine semantic and keyword search."""

    def __init__(self, semantic_searcher: GitRAG, keyword_index: BM25Index):
        self.semantic = semantic_searcher
        self.keyword = keyword_index

    def search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.7,
        **filters,
    ) -> List[SearchResult]:
        """Hybrid search combining semantic and keyword approaches."""

        # Stage 1: Semantic search
        semantic_results = self.semantic.search(query, top_k * 2, **filters)
        semantic_scores = {r.commit_hash: r.score for r in semantic_results}

        # Stage 2: Keyword search
        keyword_results = self.keyword.search(query, top_k * 2)
        keyword_scores = {r.commit_hash: r.score for r in keyword_results}

        # Normalize scores
        all_hashes = set(semantic_scores.keys()) | set(keyword_scores.keys())

        normalized_scores = {}
        max_sem = max(semantic_scores.values()) if semantic_scores else 1.0
        max_kw = max(keyword_scores.values()) if keyword_scores else 1.0

        for h in all_hashes:
            sem_score = semantic_scores.get(h, 0) / max_sem
            kw_score = keyword_scores.get(h, 0) / max_kw
            normalized_scores[h] = (
                semantic_weight * sem_score +
                (1 - semantic_weight) * kw_score
            )

        # Combine and rerank
        combined = []
        for r in semantic_results + keyword_results:
            if r.commit_hash in normalized_scores:
                r.score = normalized_scores[r.commit_hash]
                if r not in combined:
                    combined.append(r)

        combined.sort(key=lambda x: x.score, reverse=True)
        return combined[:top_k]
```

### Keyword Index: BM25 via Rank-BM25

For the keyword component, use `rank-bm25` for pure Python BM25:

```python
from rank_bm25 import BM25Okapi
import numpy as np

class BM25Index:
    """Lightweight BM25 index for keyword search."""

    def __init__(self):
        self.documents = {}  # commit_hash -> text
        self.bm25 = None

    def add_documents(self, docs: List[dict]):
        """Add documents to the index."""
        for doc in docs:
            self.documents[doc['commit_hash']] = doc['text']

        # Rebuild index
        texts = list(self.documents.values())
        tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        """Search for query."""
        if not self.bm25:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top results
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                commit_hash = list(self.documents.keys())[idx]
                results.append({
                    'commit_hash': commit_hash,
                    'score': scores[idx],
                })

        return results
```

## 6. Performance Considerations

### Latency Targets

| Operation | Target Latency | Notes |
|-----------|----------------|-------|
| Single query | <50ms | Including embedding generation |
| Bulk index (100 commits) | <5s | Background processing |
| Index rebuild (10k commits) | <2min | Offline operation |

### Batch Processing

```python
class BatchIndexer:
    """Efficient bulk indexing for git history."""

    def __init__(self, rag: GitRAG, batch_size: int = 32):
        self.rag = rag
        self.batch_size = batch_size
        self.buffer = []

    def add(self, commit: GitCommit):
        """Buffer a commit for indexing."""
        self.buffer.append(commit)

        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        """Flush buffer to zvec."""
        if not self.buffer:
            return

        # Process in batches
        chunks = []
        for commit in self.buffer:
            chunks.extend(self.rag._create_chunks(commit))

        # Batch embed
        texts = [c['text'] for c in chunks]
        embeddings = list(self.rag.embed_model.embed(texts))

        # Batch insert
        docs = []
        for chunk, embedding in zip(chunks, embeddings):
            doc = zvec.Doc(
                id=chunk['id'],
                vectors={"embedding": embedding.tolist()},
                fields=chunk['fields'],
            )
            docs.append(doc)

        self.rag.collection.insert(docs)
        self.buffer.clear()
```

### Memory Management

```python
# Limit memory usage with mmap mode
collection = zvec.open(
    path=db_path,
    options=zvec.OpenOptions(
        memory_limit_mb=512,  # Limit to 512MB
        use_mmap=True,        # Memory-map for large indexes
    )
)
```

### Caching Embeddings

```python
from functools import lru_cache

class CachedEmbedder:
    """Cache embeddings to avoid recomputation."""

    def __init__(self, model: TextEmbedding, cache_size: int = 10000):
        self.model = model
        self.cache = LRUCache(maxsize=cache_size)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts with caching."""
        # Check cache first
        cached = []
        to_compute = []
        indices = []

        for i, text in enumerate(texts):
            key = hash(text)
            if key in self.cache:
                cached.append((i, self.cache[key]))
            else:
                to_compute.append(text)
                indices.append(i)

        # Compute missing
        if to_compute:
            results = list(self.model.embed(to_compute))
            for idx, (i, emb) in enumerate(zip(indices, results)):
                self.cache[hash(to_compute[idx])] = emb

        # Combine
        all_embeddings = [None] * len(texts)
        for i, emb in cached:
            all_embeddings[i] = emb
        for idx, emb in zip(indices, results):
            all_embeddings[idx] = emb

        return np.array(all_embeddings)
```

### Background Indexing

```python
import threading
import queue

class AsyncIndexer:
    """Background indexer for non-blocking updates."""

    def __init__(self, rag: GitRAG):
        self.rag = rag
        self.queue = queue.Queue()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def index_async(self, commit: GitCommit):
        """Queue commit for background indexing."""
        self.queue.put(commit)

    def _worker(self):
        """Background worker process."""
        batch = []
        while True:
            try:
                commit = self.queue.get(timeout=1.0)
                batch.append(commit)

                if len(batch) >= 10:
                    self._index_batch(batch)
                    batch.clear()
            except queue.Empty:
                if batch:
                    self._index_batch(batch)
                    batch.clear()

    def _index_batch(self, commits: List[GitCommit]):
        """Index a batch of commits."""
        # ... implementation
```

## 7. Integration Points

### Git Hook Integration

```bash
# .git/hooks/post-commit
#!/bin/bash
python -m replay.indexer --commit $(git rev-parse HEAD)
```

### CLI Interface

```python
# Proposed CLI commands
 replay search "authentication bug fix" --author john@example.com
 replay search "rate limiting" --file "*.py" --date-after 2024-01-01
 replay index --repo . --full  # Full rebuild
 replay index --repo . --incremental  # New commits only
```

## 8. Open Questions

1. **Index granularity**: Should we index at commit level, file level, or chunk level?
2. **Update strategy**: How to handle rewrites (amended commits, rebased branches)?
3. **Deduplication**: How to handle similar commits across branches?
4. **Model updates**: Strategy for re-indexing when embedding models improve?

## 9. Future Enhancements

- **Intent-based retrieval**: Search by "why" (commit messages) vs "what" (code)
- **Semantic blame**: Find conceptually related changes across the codebase
- **Temporal reasoning**: "Find when the retry logic was added"
- **Multi-repository**: Search across multiple repositories

## Summary

| Component | Technology | Key Decision |
|-----------|------------|--------------|
| Embedding | FastEmbed + BGE-small | Best accuracy/speed balance |
| Storage | zvec | Embedded, in-process |
| Chunking | Tree-sitter (primary) | Semantic AST-based chunks |
| Hybrid | BM25 + Semantic | Two-stage retrieval |
| Filtering | zvec native filters | Author, date, file pattern |

This architecture provides:
- Sub-50ms query latency
- Zero operational overhead (no Docker/services)
- Full metadata filtering
- Hybrid search for improved recall
- Background indexing for non-blocking updates
