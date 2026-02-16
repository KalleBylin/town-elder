"""Query service for town_elder."""
from __future__ import annotations

from town_elder.config import get_config
from town_elder.embeddings import Embedder
from town_elder.storage import ZvecStore


class QueryService:
    """Service for querying documents."""

    def __init__(
        self,
        store: ZvecStore | None = None,
        embedder: Embedder | None = None,
    ):
        config = get_config()
        self.store = store or ZvecStore(config.data_dir / "vectors", dimension=config.embed_dimension)
        self.embedder = embedder or Embedder(model_name=config.embed_model, embed_dimension=config.embed_dimension)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Search for documents similar to the query.

        Args:
            query: The search query string.
            top_k: Number of results to return. If None, uses default from config.
                   If explicitly set to 0, returns no results (empty list).
        """
        config = get_config()
        if top_k is None:
            top_k = config.default_top_k

        query_vector = self.embedder.embed_single(query)
        return self.store.search(query_vector, top_k=top_k)

    def close(self) -> None:
        """Close the service."""
        self.store.close()
