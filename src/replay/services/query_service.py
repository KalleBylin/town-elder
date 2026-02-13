"""Query service for replay."""
from __future__ import annotations

from replay.embeddings import Embedder
from replay.storage import ZvecStore
from replay.config import get_config


class QueryService:
    """Service for querying documents."""

    def __init__(
        self,
        store: ZvecStore | None = None,
        embedder: Embedder | None = None,
    ):
        config = get_config()
        self.store = store or ZvecStore(config.data_dir / "vectors")
        self.embedder = embedder or Embedder()

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Search for documents similar to the query."""
        config = get_config()
        top_k = top_k or config.default_top_k

        query_vector = self.embedder.embed_single(query)
        results = self.store.search(query_vector, top_k=top_k)

        return results

    def close(self) -> None:
        """Close the service."""
        self.store.close()
