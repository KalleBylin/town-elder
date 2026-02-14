"""Configuration management."""
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_default_data_dir() -> Path:
    """Get the default data directory.

    First checks for .replay in current directory, then falls back to home.
    """
    cwd = Path.cwd()
    cwd_replay = cwd / ".replay"
    if cwd_replay.exists() and cwd_replay.is_dir():
        return cwd_replay
    return Path.home() / ".replay"


class ReplayConfig(BaseSettings):
    """Configuration for replay CLI."""

    model_config = SettingsConfigDict(
        env_prefix="REPLAY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database settings
    data_dir: Path = Field(default_factory=_get_default_data_dir)
    db_name: str = "replay.db"

    # Embedding settings
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dimension: int = 384

    # Search settings
    default_top_k: int = 5

    # Logging
    verbose: bool = False


@lru_cache
def get_config(data_dir: str | Path | None = None, clear_cache: bool = False) -> ReplayConfig:
    """Get cached configuration instance.

    Args:
        data_dir: Optional data directory to use. If provided, overrides default.
        clear_cache: If True, clear the cache before returning config.
    """
    if clear_cache:
        get_config.cache_clear()

    if data_dir:
        config = ReplayConfig(data_dir=Path(data_dir))
    else:
        config = ReplayConfig()
    return config
