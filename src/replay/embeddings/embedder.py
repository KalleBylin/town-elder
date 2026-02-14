"""Embedder implementation using fastembed."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class EmbeddingBackendUnavailableError(Exception):
    """Raised when the embedding backend is not available."""

    def __init__(self, message: str = "Embedding backend unavailable"):
        self.message = message
        super().__init__(self.message)


class Embedder:
    """Wrapper around fastembed for generating embeddings."""

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: str = DEFAULT_MODEL, allow_fallback: bool = False):
        self.model_name = model_name
        self._model = None
        self._allow_fallback = allow_fallback

    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=self.model_name)
            except ImportError as e:
                if self._allow_fallback:
                    # Silent fallback only when explicitly enabled
                    self._model = None
                else:
                    raise EmbeddingBackendUnavailableError(
                        f"Failed to load embedding backend: {e}. "
                        "Install fastembed: pip install fastembed "
                        "or use --allow-fallback to use zero vectors."
                    )

    def embed(self, texts: list[str]) -> Iterator[np.ndarray]:
        """Generate embeddings for a list of texts."""
        self._load_model()
        if self._model is None:
            # Return zero vectors if model not available and fallback is allowed
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
