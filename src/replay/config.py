"""Configuration management."""
import os
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

    def update_data_dir(self, path: Path | str | None = None) -> None:
        """Update the data directory.

        Args:
            path: Optional path to use. If None, uses default logic.
        """
        if path:
            self.data_dir = Path(path)
        else:
            self.data_dir = _get_default_data_dir()


def get_config(clear_cache: bool = False) -> ReplayConfig:
    """Get cached configuration instance.

    Args:
        clear_cache: If True, clear the cache before returning config.
    """
    if clear_cache:
        get_config.cache_clear()
    return ReplayConfig()
