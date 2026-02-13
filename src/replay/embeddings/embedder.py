"""Embedder implementation using fastembed."""
from __future__ import annotations

from typing import Iterator

import numpy as np


class Embedder:
    """Wrapper around fastembed for generating embeddings."""

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=self.model_name)
            except ImportError:
                # Fallback for when fastembed is not available
                self._model = None

    def embed(self, texts: list[str]) -> Iterator[np.ndarray]:
        """Generate embeddings for a list of texts."""
        self._load_model()
        if self._model is None:
            # Return zero vectors if model not available
            dim = self.dimension
            for _ in texts:
                yield np.zeros(dim, dtype=np.float32)
        else:
            yield from self._model.embed(texts)

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return next(self.embed([text]))

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return 384  # bge-small-en-v1.5
