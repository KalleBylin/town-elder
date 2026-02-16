"""Configuration management."""
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from town_elder.exceptions import ConfigError


def _get_default_data_dir(base_path: Path | None = None) -> Path:
    """Get the default data directory.

    Only returns a path if .town_elder exists in the base_path.
    Raises ConfigError if not found.

    Args:
        base_path: Base path to search for .town_elder. Defaults to cwd.

    Users must explicitly configure data_dir via:
    - TOWN_ELDER_DATA_DIR environment variable
    - --data-dir CLI option
    - Passing data_dir to get_config()
    """
    search_path = base_path if base_path is not None else Path.cwd()
    town_elder_dir = search_path / ".town_elder"
    if town_elder_dir.exists() and town_elder_dir.is_dir():
        return town_elder_dir
    raise ConfigError(
        "No .town_elder directory found. "
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
def _get_config_cached(data_dir: Path) -> TownElderConfig:
    """Cached configuration lookup for explicit data_dir values."""
    return TownElderConfig(data_dir=data_dir)


def get_config(
    data_dir: str | Path | None = None,
    clear_cache: bool = False,
    repo_path: str | Path | None = None,
) -> TownElderConfig:
    """Get configuration instance.

    Args:
        data_dir: Optional data directory to use. If provided, overrides default.
        clear_cache: If True, clear the cache before returning config.
        repo_path: Optional repo path to use for default data_dir resolution.

    Note: When data_dir is None, caching is disabled because the default
    resolution depends on the current working directory, which can change
    between invocations within the same process.
    """
    if clear_cache:
        _get_config_cached.cache_clear()

    # Don't cache when data_dir is None since default depends on cwd or repo_path
    if data_dir is None:
        base_path = Path(repo_path) if repo_path else None
        return TownElderConfig(data_dir=_get_default_data_dir(base_path))

    return _get_config_cached(Path(data_dir))
