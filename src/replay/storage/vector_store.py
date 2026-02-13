"""Vector store implementation using zvec."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class VectorStoreError(Exception):
    """Error for vector store operations."""
    pass


class ZvecStore:
    """zvec implementation of vector store."""

    DEFAULT_DIMENSION = 384  # bge-small-en-v1.5

    def __init__(self, path: str | Path, dimension: int = DEFAULT_DIMENSION):
        self.path = Path(path)
        self.dimension = dimension
        self._documents: dict[str, dict[str, Any]] = {}
        self._vectors: dict[str, np.ndarray] = {}

    def insert(self, text: str, metadata: dict[str, Any]) -> str:
        """Insert a document with embedding."""
        doc_id = metadata.get("id", str(hash(text)))
        self._documents[doc_id] = {
            "text": text,
            "metadata": metadata,
        }
        # Store a placeholder vector - actual embedding happens in service layer
        self._vectors[doc_id] = np.zeros(self.dimension, dtype=np.float32)
        return doc_id

    def insert_with_vector(self, doc_id: str, vector: np.ndarray, text: str, metadata: dict[str, Any]) -> str:
        """Insert a document with pre-computed embedding."""
        self._documents[doc_id] = {
            "text": text,
            "metadata": metadata,
        }
        self._vectors[doc_id] = vector
        return doc_id

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        """Search for similar documents using cosine similarity."""
        results = []
        for doc_id, vector in self._vectors.items():
            # Cosine similarity
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector) + 1e-8
            )
            doc = self._documents[doc_id]
            results.append({
                "id": doc_id,
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": float(similarity),
            })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get(self, doc_id: str) -> dict[str, Any] | None:
        """Get a document by ID."""
        if doc_id in self._documents:
            return self._documents[doc_id]
        return None

    def delete(self, doc_id: str) -> None:
        """Delete a document by ID."""
        self._documents.pop(doc_id, None)
        self._vectors.pop(doc_id, None)

    def count(self) -> int:
        """Return the number of documents."""
        return len(self._documents)

    def close(self) -> None:
        """Close the store (no-op for in-memory)."""
        pass
