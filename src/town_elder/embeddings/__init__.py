"""Embeddings module for town_elder."""
from town_elder.embeddings.backend import (
    EmbedBackendType,
    EmbedBackendUnavailableError,
    get_embed_backend_from_config,
    is_python_embed_available,
    is_rust_embed_available,
    select_embed_backend,
)
from town_elder.embeddings.embedder import Embedder

__all__ = [
    "EmbedBackendUnavailableError",
    "EmbedBackendType",
    "Embedder",
    "get_embed_backend_from_config",
    "is_python_embed_available",
    "is_rust_embed_available",
    "select_embed_backend",
]
