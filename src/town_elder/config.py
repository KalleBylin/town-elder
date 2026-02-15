"""Configuration management."""
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from town_elder.exceptions import ConfigError


def _get_default_data_dir() -> Path:
    """Get the default data directory.

    Only returns a path if .town_elder exists in the current working directory.
    Raises ConfigError if not found.

    Users must explicitly configure data_dir via:
    - TOWN_ELDER_DATA_DIR environment variable
    - --data-dir CLI option
    - Passing data_dir to get_config()
    """
    cwd = Path.cwd()
    cwd_town_elder = cwd / ".town_elder"
    if cwd_town_elder.exists() and cwd_town_elder.is_dir():
        return cwd_town_elder
    raise ConfigError(
        "No .town_elder directory found in current working directory. "
        "Please either:\n"
        "  1. Run 'te init' to initialize a local .town_elder directory, or\n"
        "  2. Set TOWN_ELDER_DATA_DIR environment variable, or\n"
        "  3. Use --data-dir option"
    )


class TownElderConfig(BaseSettings):
    """Configuration for Town Elder CLI."""

    model_config = SettingsConfigDict(
        env_prefix="TOWN_ELDER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database settings
    data_dir: Path = Field(default_factory=_get_default_data_dir)
    db_name: str = "town_elder.db"

    # Embedding settings
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dimension: int = 384

    # Search settings
    default_top_k: int = 5

    # Logging
    verbose: bool = False


@lru_cache
def get_config(data_dir: str | Path | None = None, clear_cache: bool = False) -> TownElderConfig:
    """Get cached configuration instance.

    Args:
        data_dir: Optional data directory to use. If provided, overrides default.
        clear_cache: If True, clear the cache before returning config.
    """
    if clear_cache:
        get_config.cache_clear()

    return TownElderConfig(data_dir=Path(data_dir)) if data_dir else TownElderConfig()
