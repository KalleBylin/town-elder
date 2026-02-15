"""Custom exceptions for town_elder."""


class TownElderError(Exception):
    """Base exception for town_elder."""
    pass


class DatabaseError(TownElderError):
    """Database-related errors."""
    pass


class EmbeddingError(TownElderError):
    """Embedding generation errors."""
    pass


class ConfigError(TownElderError):
    """Configuration errors."""
    pass


class VectorError(TownElderError):
    """Vector-related errors."""
    pass


class IndexingError(TownElderError):
    """Indexing-related errors."""
    pass
