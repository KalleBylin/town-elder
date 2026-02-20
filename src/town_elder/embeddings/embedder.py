"""Embedder implementation with pluggable backend support (python/rust)."""
from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

import numpy as np

from town_elder.embeddings.backend import (
    EmbedBackendType,
    is_rust_embed_available,
    select_embed_backend,
)
from town_elder.rust_adapter import create_rust_embedder


class EmbedBackend(str, Enum):
    """Embedding backend options."""

    AUTO = "auto"
    PYTHON = "python"
    RUST = "rust"


class EmbeddingBackendUnavailableError(Exception):
    """Raised when the embedding backend is not available."""

    def __init__(self, message: str = "Embedding backend unavailable"):
        self.message = message
        super().__init__(self.message)


@dataclass
class EmbedderConfig:
    """Configuration for Embedder.

    Attributes:
        model_name: Name of the embedding model to use.
        embed_dimension: Expected embedding dimension from config.
                         If provided, validates against model dimension.
        allow_fallback: If True, allow returning zero vectors when model unavailable.
        batch_size: Default embedding batch size for backend calls.
        parallel: Default embedding parallelism for backend calls.
        backend: Backend to use ("auto", "python", or "rust").
    """

    model_name: str = "BAAI/bge-small-en-v1.5"
    embed_dimension: int | None = None
    allow_fallback: bool = False
    batch_size: int = 256
    parallel: int | None = None
    backend: str = "auto"


class Embedder:
    """Wrapper around embedding backends (fastembed or Rust)."""

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
        config: EmbedderConfig | None = None,
        **kwargs: object,
    ):
        """Initialize the Embedder.

        Args:
            config: EmbedderConfig object with all configuration options.
            **kwargs: Alternative to config - individual parameters.
                      If config is provided, kwargs are ignored.
        """
        # Use config if provided, otherwise build from kwargs
        if config is not None:
            self.model_name = config.model_name
            self._embed_dimension = config.embed_dimension
            self._allow_fallback = config.allow_fallback
            self.batch_size = max(config.batch_size, 1)
            self.parallel = config.parallel if config.parallel is not None else (os.cpu_count() or 1)
            self._backend = config.backend.lower() if config.backend else "auto"
        else:
            # Extract individual parameters from kwargs with defaults
            self.model_name = str(kwargs.get("model_name", self.DEFAULT_MODEL))
            self._embed_dimension = kwargs.get("embed_dimension")
            self._allow_fallback = bool(kwargs.get("allow_fallback", False))
            batch_size = int(kwargs.get("batch_size", self.DEFAULT_BATCH_SIZE))
            self.batch_size = max(batch_size, 1)
            parallel = kwargs.get("parallel")
            self.parallel = parallel if parallel is not None else (os.cpu_count() or 1)
            backend = kwargs.get("backend", "auto")
            self._backend = backend.lower() if backend else "auto"

        # Validate backend value
        if self._backend not in {"auto", "python", "rust"}:
            raise ValueError(
                f"Invalid backend: '{self._backend}'. "
                f"Must be one of: auto, python, rust"
            )

        # Resolve actual backend type
        self._backend_type: EmbedBackendType | None = None
        self._backend_impl: object | None = None

        # Validate dimension at initialization if provided
        if self._embed_dimension is not None:
            expected_dim = self.MODEL_DIMENSIONS.get(self.model_name)
            if expected_dim is not None and self._embed_dimension != expected_dim:
                raise ValueError(
                    f"Config embed_dimension ({self._embed_dimension}) does not match "
                    f"expected dimension for model {self.model_name} ({expected_dim}). "
                    f"Either update embed_dimension in config to {expected_dim} "
                    f"or use a different model."
                )

    def _resolve_backend(self) -> EmbedBackendType:
        """Resolve the actual backend to use based on configuration and availability."""
        if self._backend_type is not None:
            return self._backend_type

        # Determine which backend to use
        self._backend_type = select_embed_backend(
            self._backend,
            rust_available=is_rust_embed_available(),
            python_available=True,
        )
        return self._backend_type

    def _load_backend(self) -> None:
        """Lazy load the embedding backend."""
        if self._backend_impl is not None:
            return

        backend_type = self._resolve_backend()

        if backend_type == EmbedBackendType.RUST:
            try:
                self._backend_impl = create_rust_embedder(
                    model=self.model_name,
                )
            except Exception as e:
                if self._allow_fallback:
                    self._backend_impl = None
                else:
                    raise EmbeddingBackendUnavailableError(
                        f"Failed to initialize Rust embedding backend: {e}. "
                        "Set TE_USE_RUST_CORE=1 and ensure the Rust extension is built: "
                        "cd rust && maturin develop"
                    ) from e
        else:
            # Python backend (fastembed)
            try:
                from fastembed import TextEmbedding

                self._backend_impl = TextEmbedding(model_name=self.model_name)
            except ImportError as e:
                if self._allow_fallback:
                    self._backend_impl = None
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
        self._load_backend()
        if self._backend_impl is None:
            # Return zero vectors if model not available and fallback is allowed
            dim = self.dimension
            for _ in texts:
                yield np.zeros(dim, dtype=np.float32)
        else:
            backend_type = self._resolve_backend()
            if backend_type == EmbedBackendType.RUST:
                # Rust backend returns lists directly
                embedding_lists = self._backend_impl.embed(texts)
                for emb in embedding_lists:
                    yield np.array(emb, dtype=np.float32)
            else:
                # Python backend (fastembed)
                effective_batch_size = batch_size if batch_size is not None else self.batch_size
                effective_parallel = parallel if parallel is not None else self.parallel
                try:
                    yield from self._backend_impl.embed(
                        texts,
                        batch_size=effective_batch_size,
                        parallel=effective_parallel,
                    )
                except TypeError as exc:
                    # Backwards-compatible fallback for older fastembed versions.
                    if "unexpected keyword argument" not in str(exc):
                        raise
                    yield from self._backend_impl.embed(texts)

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

    @property
    def backend_type(self) -> str:
        """Return the resolved backend type.

        Returns:
            "python" or "rust" indicating which backend is being used.
        """
        return self._resolve_backend().value
