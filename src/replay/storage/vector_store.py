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
        self._collection = None

    def _get_collection(self):
        """Get or create the zvec collection."""
        if self._collection is None:
            try:
                import zvec
                from zvec import FieldSchema, VectorSchema, DataType

                # First, try to open existing collection
                path_str = str(self.path)
                try:
                    self._collection = zvec.open(path=path_str)
                except Exception:
                    # If opening fails, create new collection
                    text_field = FieldSchema("text", DataType.STRING)
                    metadata_field = FieldSchema("metadata", DataType.STRING)
                    emb_vector = VectorSchema(
                        "embedding",
                        dimension=self.dimension,
                        data_type=DataType.VECTOR_FP32,
                    )

                    schema = zvec.CollectionSchema(
                        name="replay",
                        fields=[text_field, metadata_field],
                        vectors=emb_vector,
                    )

                    self._collection = zvec.create_and_open(
                        path=path_str, schema=schema
                    )
            except ImportError:
                raise VectorStoreError("zvec not installed. Run: pip install zvec")
            except Exception as e:
                raise VectorStoreError(f"Failed to open zvec collection: {e}")

        return self._collection

    def insert(self, text: str, metadata: dict[str, Any]) -> str:
        """Insert a document with embedding."""
        import uuid
        import zvec
        import json

        doc_id = metadata.get("id", str(uuid.uuid4()))
        vector = np.zeros(self.dimension, dtype=np.float32)
        self._get_collection().insert(
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

    def insert_with_vector(self, doc_id: str, vector: np.ndarray, text: str, metadata: dict[str, Any]) -> str:
        """Insert a document with pre-computed embedding."""
        import zvec
        import json

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
                "metadata": json.loads(result.fields.get("metadata", "{}")),
                "score": 1.0,  # zvec doesn't provide scores in the same way
            })
        return output

    def get(self, doc_id: str) -> dict[str, Any] | None:
        """Get a document by ID."""
        import json
        collection = self._get_collection()
        try:
            # Use fetch to get by ID - returns a dict, not a list
            docs = collection.fetch(ids=[doc_id])
            if doc_id in docs:
                doc = docs[doc_id]
                return {
                    "text": doc.fields.get("text", ""),
                    "metadata": json.loads(doc.fields.get("metadata", "{}")),
                }
        except Exception:
            pass
        return None

    def delete(self, doc_id: str) -> None:
        """Delete a document by ID."""
        collection = self._get_collection()
        try:
            collection.delete(doc_id)
        except Exception:
            pass

    def count(self) -> int:
        """Return the number of documents."""
        collection = self._get_collection()
        try:
            stats = collection.stats
            return stats.doc_count
        except Exception:
            return 0

    def close(self) -> None:
        """Close the store."""
        # zvec collections don't need explicit closing
        # but we reset the reference to allow re-opening
        self._collection = None
