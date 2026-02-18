"""Embedding backend selection logic.

This module provides deterministic backend selection for embeddings
without importing heavy model dependencies.

Selection semantics:
- `python`: require Python backend only.
- `rust`: require Rust backend only (error if unavailable).
- `auto`: prefer Rust when enabled+available, otherwise Python.
"""
from __future__ import annotations

from enum import Enum

from town_elder.config import EmbedBackend
from town_elder.exceptions import ConfigError
from town_elder.rust_adapter import get_te_core


class EmbedBackendType(str, Enum):
    """Selected embedding backend type."""

    PYTHON = "python"
    RUST = "rust"


class EmbedBackendUnavailableError(Exception):
    """Raised when the requested embedding backend is not available."""

    def __init__(self, backend: str, reason: str | None = None):
        message = f"Embedding backend '{backend}' is not available"
        if reason:
            message += f": {reason}"
        else:
            message += ". Ensure the required runtime is installed."
        super().__init__(message)
        self.backend = backend
        self.reason = reason


def is_rust_embed_available() -> bool:
    """Check if Rust embedding backend is available.

    Returns:
        True if Rust core is enabled AND the module is available, False otherwise.
    """
    return get_te_core() is not None


def is_python_embed_available() -> bool:
    """Check if Python embedding backend is available.

    The Python backend (fastembed) is always considered available
    as long as the package can be imported when needed.

    Returns:
        Always returns True (actual availability checked at usage time).
    """
    return True


def select_embed_backend(
    backend_config: str,
    rust_available: bool | None = None,
    python_available: bool | None = None,
) -> EmbedBackendType:
    """Select the embedding backend based on configuration and availability.

    Args:
        backend_config: The backend configuration value (auto|python|rust).
        rust_available: Override check for Rust availability (for testing).
        python_available: Override check for Python availability (for testing).

    Returns:
        The selected EmbedBackendType.

    Raises:
        ConfigError: If backend_config contains an invalid value.
        EmbedBackendUnavailableError: If the requested backend is unavailable.
    """
    # Normalize and validate the config value
    config_value = backend_config.lower().strip() if backend_config else "auto"

    # Validate the config value
    from town_elder.config import VALID_EMBED_BACKEND_VALUES

    if config_value not in VALID_EMBED_BACKEND_VALUES:
        raise ConfigError(
            f"Invalid embed_backend value: '{backend_config}'. "
            f"Valid values are: {', '.join(sorted(VALID_EMBED_BACKEND_VALUES))}"
        )

    # Use provided availability or check
    if rust_available is None:
        rust_available = is_rust_embed_available()
    if python_available is None:
        python_available = is_python_embed_available()

    # Selection logic based on config
    if config_value == EmbedBackend.PYTHON.value:
        # Explicitly require Python backend
        if not python_available:
            raise EmbedBackendUnavailableError(
                "python",
                "Python embedding backend (fastembed) is not available. "
                "Install with: pip install fastembed",
            )
        return EmbedBackendType.PYTHON

    if config_value == EmbedBackend.RUST.value:
        # Explicitly require Rust backend
        if not rust_available:
            raise EmbedBackendUnavailableError(
                "rust",
                "Rust embedding backend is not available. "
                "Set TE_USE_RUST_CORE=1 and ensure the Rust extension is built: "
                "cd rust && maturin develop",
            )
        return EmbedBackendType.RUST

    # Auto mode: prefer Rust when available, fallback to Python
    # This preserves backward compatibility with existing behavior
    if config_value == EmbedBackend.AUTO.value:
        if rust_available:
            return EmbedBackendType.RUST
        return EmbedBackendType.PYTHON

    # Should never reach here due to validation above
    raise ConfigError(f"Unhandled embed_backend value: {config_value}")


def get_embed_backend_from_config(
    config_backend: str,
) -> EmbedBackendType:
    """Get the selected embedding backend from configuration.

    This is the main entry point used by service factory and other consumers.

    Args:
        config_backend: The embed_backend value from TownElderConfig.

    Returns:
        The selected EmbedBackendType.

    Raises:
        ConfigError: If backend_config contains an invalid value.
        EmbedBackendUnavailableError: If the requested backend is unavailable.
    """
    return select_embed_backend(config_backend)
