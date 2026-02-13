"""Custom exceptions for replay."""


class ReplayError(Exception):
    """Base exception for replay."""
    pass


class DatabaseError(ReplayError):
    """Database-related errors."""
    pass


class EmbeddingError(ReplayError):
    """Embedding generation errors."""
    pass


class ConfigError(ReplayError):
    """Configuration errors."""
    pass
