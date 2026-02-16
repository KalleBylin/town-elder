"""Embedder implementation using fastembed."""
from __future__ import annotations

import os
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
    DEFAULT_DIMENSION = 384  # bge-small-en-v1.5
    DEFAULT_BATCH_SIZE = 256

    # Known model dimensions for validation
    MODEL_DIMENSIONS: dict[str, int] = {
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
    }

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        embed_dimension: int | None = None,
        allow_fallback: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
        parallel: int | None = None,
    ):
        """Initialize the Embedder.

        Args:
            model_name: Name of the embedding model to use.
            embed_dimension: Expected embedding dimension from config.
                             If provided, validates against model dimension.
            allow_fallback: If True, allow returning zero vectors when model unavailable.
            batch_size: Default embedding batch size for backend calls.
            parallel: Default embedding parallelism for backend calls.
        """
        self.model_name = model_name
        self._embed_dimension = embed_dimension
        self._model = None
        self._allow_fallback = allow_fallback
        self.batch_size = max(batch_size, 1)
        self.parallel = parallel if parallel is not None else (os.cpu_count() or 1)

        # Validate dimension at initialization if provided
        if embed_dimension is not None:
            expected_dim = self.MODEL_DIMENSIONS.get(model_name)
            if expected_dim is not None and embed_dimension != expected_dim:
                raise ValueError(
                    f"Config embed_dimension ({embed_dimension}) does not match "
                    f"expected dimension for model {model_name} ({expected_dim}). "
                    f"Either update embed_dimension in config to {expected_dim} "
                    f"or use a different model."
                )

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
                        "Install fastembed: pip install fastembed"
                    )

    def embed(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        parallel: int | None = None,
    ) -> Iterator[np.ndarray]:
        """Generate embeddings for a list of texts."""
        self._load_model()
        if self._model is None:
            # Return zero vectors if model not available and fallback is allowed
            dim = self.dimension
            for _ in texts:
                yield np.zeros(dim, dtype=np.float32)
        else:
            effective_batch_size = batch_size if batch_size is not None else self.batch_size
            effective_parallel = parallel if parallel is not None else self.parallel
            try:
                yield from self._model.embed(
                    texts,
                    batch_size=effective_batch_size,
                    parallel=effective_parallel,
                )
            except TypeError as exc:
                # Backwards-compatible fallback for older fastembed versions.
                if "unexpected keyword argument" not in str(exc):
                    raise
                yield from self._model.embed(texts)

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return next(self.embed([text]))

    @property
    def dimension(self) -> int:
        """Return the embedding dimension.

        Returns the configured embed_dimension if provided, otherwise
        falls back to the model's default dimension.
        """
        if self._embed_dimension is not None:
            return self._embed_dimension
        return self.MODEL_DIMENSIONS.get(self.model_name, self.DEFAULT_DIMENSION)
