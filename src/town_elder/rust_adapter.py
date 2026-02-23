"""Rust adapter - single import boundary for PyO3 te-core module.

This module provides a feature-gated adapter for the Rust te-core extension.
All future Rust call sites should go through this adapter, never direct module imports.

Feature Flag:
    TE_USE_RUST_CORE: Set to "1" or "true" to enable Rust core usage.
    Defaults to off (backward compatible).

Behavior:
    - Flag OFF (default): Returns None from all accessors, no-op behavior.
    - Flag ON + Module available: Returns the module or functions.
    - Flag ON + Module missing: Raises RustExtensionNotAvailableError with clear diagnostic.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

# =============================================================================
# Feature Flag Configuration
# =============================================================================

# Environment variable to enable Rust core
_ENV_FLAG = "TE_USE_RUST_CORE"

# Values that enable the feature
_ENABLED_VALUES = frozenset({"1", "true", "True", "TRUE", "yes", "Yes", "YES"})


def is_rust_core_enabled() -> bool:
    """Check if Rust core feature flag is enabled.

    Returns:
        True if TE_USE_RUST_CORE is set to an enabled value, False otherwise.
    """
    return os.environ.get(_ENV_FLAG, "").strip() in _ENABLED_VALUES


def _check_rust_available() -> bool:
    """Check if Rust extension module is available for import.

    Returns:
        True if the module can be imported, False otherwise.
    """
    try:
        # pylint: disable=import-error,unused-import
        import town_elder._core  # noqa: F401
        return True
    except ImportError:
        return False


# Cache for the imported module
_te_core_module: Any = None
_module_checked: bool = False


def get_te_core() -> Any | None:
    """Get the te-core Rust module if available and enabled.

    Returns:
        The te-core module if flag is enabled and module is available,
        None otherwise.

    Note:
        Use get_te_core_or_raise() when you need to differentiate between
        "flag off" and "module unavailable" cases.
    """
    global _te_core_module, _module_checked

    if not is_rust_core_enabled():
        return None

    if _module_checked:
        return _te_core_module

    _module_checked = True
    if _check_rust_available():
        try:
            import sys

            cached_module = sys.modules.get("town_elder._core")
            if cached_module is not None:
                _te_core_module = cached_module
                return _te_core_module

            # pylint: disable=import-error
            import town_elder._core as _te_core_module
            return _te_core_module
        except ImportError:
            return None

    return None


class RustExtensionNotAvailableError(Exception):
    """Raised when Rust extension is required but not available.

    This occurs when TE_USE_RUST_CORE is enabled but the extension
    module cannot be imported (not built, wrong platform, etc.).
    """

    def __init__(self, message: str | None = None) -> None:
        if message is None:
            message = (
                "Rust extension (te-core) is not available. "
                "Ensure the extension is built: cd rust && maturin develop. "
                "Or disable Rust core by unsetting TE_USE_RUST_CORE."
            )
        super().__init__(message)


def get_te_core_or_raise() -> Any:
    """Get the te-core Rust module or raise if unavailable.

    Returns:
        The te-core module if available.

    Raises:
        RustExtensionNotAvailableError: If the module cannot be imported.
    """
    if not is_rust_core_enabled():
        raise RustExtensionNotAvailableError(
            "Rust core is disabled. Set TE_USE_RUST_CORE=1 to enable."
        )

    module = get_te_core()
    if module is None:
        raise RustExtensionNotAvailableError()

    return module


# =============================================================================
# Convenience Functions for Common Operations
# =============================================================================

def health_check() -> str | None:
    """Run te-core health check if available.

    Returns:
        Health check string from Rust, or None if not available.
    """
    module = get_te_core()
    if module is None:
        return None
    try:
        return module.health()
    except Exception:  # noqa: BLE001
        # Any error means the extension is not functional
        return None


def version() -> str | None:
    """Get te-core version if available.

    Returns:
        Version string from Rust, or None if not available.
    """
    module = get_te_core()
    if module is None:
        return None
    try:
        return module.version()
    except Exception:  # noqa: BLE001
        return None


def scan_files(
    root_path: Path,
    extensions: frozenset[str] | None = None,
    exclude_patterns: frozenset[str] | None = None,
) -> list[Path]:
    """Scan files using Rust when available, otherwise use Python scanner."""
    module = get_te_core()
    if module is not None:
        try:
            extension_values = sorted(extensions) if extensions else None
            exclude_values = sorted(exclude_patterns) if exclude_patterns else None
            return [
                Path(path_value)
                for path_value in module.scan_files(
                    str(root_path),
                    extension_values,
                    exclude_values,
                )
            ]
        except Exception:
            pass

    from town_elder.indexing.file_scanner import scan_files as python_scan_files

    return python_scan_files(
        root_path,
        extensions=extensions,
        exclude_patterns=exclude_patterns,
    )


def build_file_doc_id(path_value: str, chunk_index: int = 0) -> str:
    """Build deterministic file doc IDs using Rust when available."""
    module = get_te_core()
    if module is not None:
        try:
            return module.build_file_doc_id(path_value, chunk_index)
        except Exception:
            pass

    doc_id_input = path_value if chunk_index == 0 else f"{path_value}#chunk:{chunk_index}"
    return hashlib.sha256(doc_id_input.encode()).hexdigest()[:16]


def get_doc_id_inputs(path_value: str, repo_root: Path) -> set[str]:
    """Return canonical and legacy doc-id inputs."""
    module = get_te_core()
    if module is not None:
        try:
            return set(module.get_doc_id_inputs(path_value, str(repo_root)))
        except Exception:
            pass

    doc_id_inputs = {path_value}
    path_obj = Path(path_value)
    if not path_obj.is_absolute():
        doc_id_inputs.add(str((repo_root / path_obj).resolve()))
    return doc_id_inputs


def normalize_chunk_metadata(
    *,
    base_metadata: dict[str, Any],
    chunk_metadata: dict[str, Any],
    fallback_chunk_index: int,
) -> tuple[dict[str, Any], int]:
    """Normalize chunk metadata and chunk index with parity-safe fallback."""
    module = get_te_core()
    if module is not None:
        try:
            return module.normalize_chunk_metadata(
                base_metadata,
                chunk_metadata,
                fallback_chunk_index,
            )
        except Exception:
            pass

    metadata = dict(base_metadata)
    metadata.update(chunk_metadata)

    chunk_index_value = metadata.get("chunk_index")
    if (
        isinstance(chunk_index_value, bool)
        or not isinstance(chunk_index_value, int)
        or chunk_index_value < 0
    ):
        chunk_index = fallback_chunk_index
        metadata["chunk_index"] = chunk_index
    else:
        chunk_index = chunk_index_value

    return metadata, chunk_index


def parse_rst_chunks(content: str) -> list[tuple[str, dict[str, Any]]] | None:
    """Parse RST chunks via Rust and return `(text, metadata)` rows."""
    module = get_te_core()
    if module is None:
        return None

    try:
        chunks = module.parse_rst_content(content)
        parsed_chunks: list[tuple[str, dict[str, Any]]] = []
        for chunk in chunks:
            metadata = module.get_chunk_metadata(chunk)
            parsed_chunks.append((chunk.text, metadata))
        return parsed_chunks
    except Exception:
        return None


# =============================================================================
# Adapter Status Information
# =============================================================================

def get_adapter_status() -> dict[str, Any]:
    """Get diagnostic information about the adapter state.

    Returns:
        Dictionary with keys:
        - rust_core_enabled: Whether the feature flag is set
        - module_available: Whether the module can be imported
        - flag_environment: The environment variable name used
    """
    module_available = _check_rust_available()

    return {
        "rust_core_enabled": is_rust_core_enabled(),
        "module_available": module_available,
        "flag_environment": _ENV_FLAG,
    }


# =============================================================================
# Reset Function (for testing)
# =============================================================================

def _reset_module_cache() -> None:
    """Reset the cached module state.

    This is primarily for testing purposes to allow re-checking
    the module availability after mock manipulations.
    """
    global _te_core_module, _module_checked
    _te_core_module = None
    _module_checked = False


# =============================================================================
# Diff Parser Adapter
# =============================================================================

class RustDiffParser:
    """Rust-backed diff parser that wraps PyDiffParser.

    This class provides the same interface as Python's DiffParser but uses
    the Rust implementation for parsing diffs to text.
    """

    def __init__(self, warn_on_parse_error: bool = True):
        """Initialize the Rust diff parser.

        Args:
            warn_on_parse_error: If True, log warnings when diff headers fail to parse.
        """
        module = get_te_core()
        if module is None:
            raise RustExtensionNotAvailableError(
                "Rust core is not available for DiffParser"
            )
        self._parser = module.PyDiffParser(warn_on_parse_error)

    def parse(self, diff_output: str):
        """Parse git diff output into file changes.

        Returns:
            Iterator of DiffFile objects.
        """
        # Import here to avoid circular imports
        from town_elder.git.diff_parser import DiffFile
        for df in self._parser.parse(diff_output):
            yield DiffFile(
                path=df.path,
                status=df.status,
                hunks=df.hunks,
            )

    def parse_diff_to_text(self, diff_output: str) -> str:
        """Convert a diff to plain text for embedding.

        Args:
            diff_output: The raw git diff output.

        Returns:
            Plain text representation of the diff.
        """
        return self._parser.parse_diff_to_text(diff_output)


def get_diff_parser_factory() -> type | None:
    """Get the Rust-backed DiffParser class if available and enabled.

    Returns:
        RustDiffParser class if Rust is enabled and available, None otherwise.
    """
    if not is_rust_core_enabled():
        return None

    module = get_te_core()
    if module is None:
        return None

    # Verify PyDiffParser is available
    try:
        _ = module.PyDiffParser
    except AttributeError:
        return None

    return RustDiffParser


# =============================================================================
# Commit Text Assembly Functions
# =============================================================================

def assemble_commit_text(message: str, diff_text: str) -> str:
    """Assemble commit text for embedding using Rust if available.

    Args:
        message: The commit message.
        diff_text: The diff text.

    Returns:
        Assembled commit text for embedding.
    """
    module = get_te_core()
    if module is not None:
        try:
            return module.assemble_commit_text(message, diff_text)
        except Exception:
            pass
    # Fallback to Python implementation
    return f"Commit: {message}\n\n{diff_text}"


def truncate_diff(diff: str, max_size: int) -> tuple[str, bool]:
    """Truncate diff text if it exceeds maximum size using Rust if available.

    Args:
        diff: The diff text.
        max_size: Maximum size in bytes.

    Returns:
        Tuple of (truncated diff, was_truncated).
    """
    module = get_te_core()
    if module is not None:
        try:
            truncated = module.truncate_diff(diff, max_size)
            was_truncated = module.is_diff_truncated(truncated)
            return truncated, was_truncated
        except Exception:
            pass
    # Fallback to Python implementation
    if len(diff.encode()) > max_size:
        return diff[:max_size] + f"\n\n[truncated - exceeded {max_size} byte limit]", True
    return diff, False


# =============================================================================
# Text Embedding Adapter
# =============================================================================


class RustTextEmbedder:
    """Rust-backed text embedder that wraps PyTextEmbedder.

    This class provides the same interface for text embedding but uses
    the Rust fastembed implementation for generating embeddings.
    """

    def __init__(self, model: str, cache_dir: str | None = None):
        """Initialize the Rust text embedder.

        Args:
            model: The model identifier (e.g., "BAAI/bge-small-en-v1.5")
            cache_dir: Optional directory for caching model files

        Raises:
            RustExtensionNotAvailableError: If Rust core is not available.
            ValueError: If the model is not supported or invalid config.
            RuntimeError: If the embedder fails to initialize at runtime.
        """
        module = get_te_core_or_raise()
        try:
            self._embedder = module.PyTextEmbedder(model, cache_dir)
        except ValueError:
            # Re-raise ValueError for invalid model/config errors - don't wrap
            raise
        except Exception as e:
            # Only wrap true runtime init failures, not config errors
            raise RuntimeError(f"Failed to initialize Rust embedder: {e}") from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (each vector is a list of f32 values).
        """
        return self._embedder.embed(texts)

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as a list of f32 values.
        """
        return self._embedder.embed_single(text)

    @property
    def dimension(self) -> int:
        """Get the dimensionality of embeddings produced by this embedder."""
        return self._embedder.dimension()

    @property
    def model_name(self) -> str:
        """Get the model name used by this embedder."""
        return self._embedder.get_model_name()


def create_rust_embedder(
    model: str = "BAAI/bge-small-en-v1.5",
    cache_dir: str | None = None,
) -> RustTextEmbedder:
    """Create a Rust-backed text embedder.

    Args:
        model: The model identifier (e.g., "BAAI/bge-small-en-v1.5").
               Use list_rust_embedder_models() to see available options.
        cache_dir: Optional directory for caching model files.

    Returns:
        RustTextEmbedder instance.

    Raises:
        RustExtensionNotAvailableError: If Rust core is not available or disabled.
        ValueError: If the model is not supported.
        RuntimeError: If the embedder fails to initialize.
    """
    return RustTextEmbedder(model=model, cache_dir=cache_dir)


def is_rust_embed_available() -> bool:
    """Check if Rust embedding backend is available.

    Returns:
        True if Rust core is enabled AND the module is available, False otherwise.
    """
    return get_te_core() is not None


def list_rust_embedder_models() -> list[tuple[str, int]]:
    """List supported embedding models from Rust backend.

    Returns:
        List of (model_name, dimension) tuples, or empty list if unavailable.
    """
    module = get_te_core()
    if module is None:
        return []

    try:
        return module.PyTextEmbedder.list_supported_models()
    except AttributeError:
        return []


def get_embed_backend_status() -> dict[str, bool]:
    """Get diagnostic information about embedding backend availability.

    Returns:
        Dictionary with keys:
        - rust_available: Whether Rust embedding is available
        - rust_enabled: Whether the Rust core flag is enabled
        - module_available: Whether the module can be imported
    """
    module_available = _check_rust_available()

    return {
        "rust_available": is_rust_embed_available(),
        "rust_enabled": is_rust_core_enabled(),
        "module_available": module_available,
    }
