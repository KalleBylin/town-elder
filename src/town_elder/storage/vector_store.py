"""Vector store implementation using zvec."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _safe_parse_json(json_str: str) -> dict:
    """Parse JSON string, returning empty dict on failure.

    This handles corrupted metadata in the vector store gracefully.
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}


class VectorStoreError(Exception):
    """Error for vector store operations."""
    pass


class ZvecStore:
    """zvec implementation of vector store."""

    DEFAULT_DIMENSION = 384  # bge-small-en-v1.5

    def __init__(self, path: str | Path, dimension: int = DEFAULT_DIMENSION):
        self.path = Path(path)
        self.dimension = dimension
        self._collection = None

    def _get_collection(self):
        """Get or create the zvec collection."""
        if self._collection is None:
            try:
                import zvec
                from zvec import DataType, FieldSchema, VectorSchema

                path_str = str(self.path)

                # First, check if the collection directory exists
                collection_exists = self.path.exists() and any(self.path.iterdir()) if self.path.exists() else False

                if collection_exists:
                    # Collection directory exists - try to open it
                    # If this fails, it's likely corruption/schema mismatch/permission issue
                    try:
                        self._collection = zvec.open(path=path_str)
                    except Exception as e:
                        error_msg = str(e).lower()
                        # Differentiate between fatal errors and "not found"
                        if "not found" in error_msg or "does not exist" in error_msg:
                            # Collection directory exists but not a valid zvec collection - create new
                            pass  # Fall through to create
                        else:
                            # Fatal error - corruption, schema mismatch, or permission issue
                            raise VectorStoreError(
                                f"Failed to open zvec collection at {self.path}: {e}\n"
                                "This may indicate corruption, schema mismatch, or permission issues.\n"
                                "If the collection is corrupted, you may need to reinitialize the database."
                            )
                else:
                    # Collection doesn't exist - create new one
                    pass

                # Create new collection if needed
                if self._collection is None:
                    text_field = FieldSchema("text", DataType.STRING)
                    metadata_field = FieldSchema("metadata", DataType.STRING)
                    emb_vector = VectorSchema(
                        "embedding",
                        dimension=self.dimension,
                        data_type=DataType.VECTOR_FP32,
                    )

                    schema = zvec.CollectionSchema(
                        name="town_elder",
                        fields=[text_field, metadata_field],
                        vectors=emb_vector,
                    )

                    self._collection = zvec.create_and_open(
                        path=path_str, schema=schema
                    )
            except ImportError:
                raise VectorStoreError("zvec not installed. Run: pip install zvec")
            except VectorStoreError:
                raise  # Re-raise our own VectorStoreError
            except Exception as e:
                raise VectorStoreError(
                    f"Failed to open or create zvec collection: {e}\n"
                    "If this is a permission error, check that you have read/write access to the directory."
                )

        return self._collection

    def insert(self, doc_id: str, vector: np.ndarray, text: str, metadata: dict[str, Any]) -> str:
        """Insert a document with embedding."""
        import json

        import zvec

        collection = self._get_collection()
        collection.insert(
            zvec.Doc(
                id=doc_id,
                vectors={"embedding": vector.tolist()},
                fields={
                    "text": text,
                    "metadata": json.dumps(metadata),
                },
            )
        )
        return doc_id

    def upsert(self, doc_id: str, vector: np.ndarray, text: str, metadata: dict[str, Any]) -> str:
        """Insert a document, updating if it already exists.

        This provides idempotent indexing: running the same upsert multiple times
        produces the same result without duplicating documents.
        """
        # Delete existing document if it exists, then insert new one
        self.delete(doc_id)
        return self.insert(doc_id, vector, text, metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        """Search for similar documents using cosine similarity."""
        import json

        import zvec
        collection = self._get_collection()

        results = collection.query(
            vectors=zvec.VectorQuery(
                field_name="embedding",
                vector=query_vector.tolist(),
            ),
            topk=top_k,
            output_fields=["text", "metadata"],
        )

        output = []
        for result in results:
            output.append({
                "id": result.id,
                "text": result.fields.get("text", ""),
                "metadata": _safe_parse_json(result.fields.get("metadata", "{}")),
                "score": result.score,
            })
        return output

    def get(self, doc_id: str) -> dict[str, Any] | None:
        """Get a document by ID."""
        import json
        collection = self._get_collection()
        # Use fetch to get by ID - returns a dict, not a list
        docs = collection.fetch(ids=[doc_id])
        if doc_id in docs:
            doc = docs[doc_id]
            return {
                "text": doc.fields.get("text", ""),
                "metadata": _safe_parse_json(doc.fields.get("metadata", "{}")),
            }
        return None

    def delete(self, doc_id: str) -> None:
        """Delete a document by ID."""
        collection = self._get_collection()
        collection.delete(doc_id)

    def count(self) -> int:
        """Return the number of documents."""
        collection = self._get_collection()
        stats = collection.stats
        return stats.doc_count

    def close(self) -> None:
        """Close the store."""
        # zvec collections don't need explicit closing
        # but we reset the reference to allow re-opening
        self._collection = None

    def get_all(self, include_vectors: bool = False) -> list[dict[str, Any]]:
        """Get all documents from the store.

        Args:
            include_vectors: If True, include the embedding vectors in the output.

        Returns:
            List of documents with id, text, metadata, and optionally vectors.
        """
        import json


        collection = self._get_collection()

        # Query all documents without a vector (empty filter matches all)
        output_fields = ["text", "metadata"]
        if include_vectors:
            output_fields.append("embedding")

        results = collection.query(
            topk=collection.stats.doc_count,
            output_fields=output_fields,
            include_vector=include_vectors,
        )

        output = []
        for result in results:
            doc = {
                "id": result.id,
                "text": result.fields.get("text", ""),
                "metadata": _safe_parse_json(result.fields.get("metadata", "{}")),
            }
            if include_vectors and hasattr(result, "vectors"):
                doc["vector"] = result.vectors.get("embedding", [])
            output.append(doc)
        return output
