"""Configuration management."""
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ReplayConfig(BaseSettings):
    """Configuration for replay CLI."""

    model_config = SettingsConfigDict(
        env_prefix="REPLAY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database settings
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".replay")
    db_name: str = "replay.db"

    # Embedding settings
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dimension: int = 384

    # Search settings
    default_top_k: int = 5

    # Logging
    verbose: bool = False


@lru_cache
def get_config() -> ReplayConfig:
    """Get cached configuration instance."""
    return ReplayConfig()
