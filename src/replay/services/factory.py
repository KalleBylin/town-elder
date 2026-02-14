"""Service factory for dependency injection."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from replay.config import get_config
from replay.embeddings import Embedder
from replay.git import DiffParser, GitRunner
from replay.storage import ZvecStore


class ServiceFactory:
    """Factory for creating service instances with dependency injection."""

    def __init__(self):
        """Initialize the service factory."""
        self._config = get_config()

    def create_embedder(self) -> Embedder:
        """Create an Embedder instance.

        Returns:
            An Embedder configured from the app config.
        """
        return Embedder(model_name=self._config.embed_model)

    def create_vector_store(self) -> ZvecStore:
        """Create a ZvecStore instance.

        Returns:
            A ZvecStore configured from the app config.
        """
        vectors_path = self._config.data_dir / "vectors"
        return ZvecStore(vectors_path, dimension=self._config.embed_dimension)

    def create_git_runner(self, repo_path: Path | str | None = None) -> GitRunner:
        """Create a GitRunner instance.

        Args:
            repo_path: Path to the git repository. Defaults to None (cwd).

        Returns:
            A GitRunner for the specified repository.
        """
        return GitRunner(repo_path)

    def create_diff_parser(self) -> DiffParser:
        """Create a DiffParser instance.

        Returns:
            A DiffParser for parsing git diffs.
        """
        return DiffParser()


@lru_cache
def get_service_factory() -> ServiceFactory:
    """Get a cached ServiceFactory instance.

    Returns:
        A singleton ServiceFactory instance.
    """
    return ServiceFactory()
