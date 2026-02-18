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

import os
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
        import town_elder._te_core  # noqa: F401
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
            # pylint: disable=import-error
            import town_elder._te_core as _te_core_module
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
            was_truncated = module.is_diff_truncated(diff)
            return truncated, was_truncated
        except Exception:
            pass
    # Fallback to Python implementation
    if len(diff.encode()) > max_size:
        return diff[:max_size] + f"\n\n[truncated - exceeded {max_size} byte limit]", True
    return diff, False
