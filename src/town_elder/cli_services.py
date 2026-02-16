"""CLI service layer for Town Elder.

Provides consolidated service abstractions for CLI commands, eliminating
duplicated config loading, validation, and service creation patterns.
"""
from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import typer
from rich.console import Console

from town_elder.config import TownElderConfig, get_config
from town_elder.embeddings import Embedder
from town_elder.exceptions import ConfigError
from town_elder.git import DiffParser, GitRunner
from town_elder.services import ServiceFactory, get_service_factory
from town_elder.storage import ZvecStore

# Exit codes
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_INVALID_ARG = 2

console = Console(stderr=False)  # stdout for normal output
error_console = Console(stderr=True)  # stderr for errors


def _escape_rich(text: str) -> str:
    """Escape brackets to prevent Rich markup interpretation."""
    return text.replace("[", "\\[").replace("]", "\\]")


def _is_empty_repo_error(error_msg: str) -> bool:
    """Check if git error message indicates an empty repository (not a fatal error).

    Git uses exit code 128 for both empty repositories and fatal errors.
    This function distinguishes between them by checking the error message.

    Returns True for empty repository, False for fatal errors (not a git repo, corrupt, etc.)
    """
    # Empty repository errors - these are safe to treat as "no commits"
    empty_repo_patterns = [
        "your branch has no commits yet",
        "bad object HEAD",  # Empty repo with no commits
        "does not have any commits",  # Another empty repo variant
    ]
    # Fatal errors that should not be masked - these indicate real problems
    fatal_error_patterns = [
        "not a git repository",
        "fatal: repository ",
        "corrupt repository",
        "could not read",
        "unable to read",
    ]
    error_msg_lower = error_msg.lower()

    # Check for fatal errors first - if any match, it's not an empty repo
    if any(pattern.lower() in error_msg_lower for pattern in fatal_error_patterns):
        return False

    # Check for empty repo patterns
    return any(pattern.lower() in error_msg_lower for pattern in empty_repo_patterns)


class CLIContext:
    """Invocation-scoped context for CLI state.

    Replaces module-global _data_dir to prevent leakage across CLI invocations.
    """

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir


class CLIServiceError(Exception):
    """Base exception for CLI service errors."""

    pass


class DatabaseNotInitializedError(CLIServiceError):
    """Raised when database is not initialized."""

    pass


class ServiceInitError(CLIServiceError):
    """Raised when service initialization fails."""

    pass


class CLIServiceContext:
    """Invocation-scoped context for CLI services.

    Encapsulates:
    - Data directory resolution from Typer context
    - Config loading and validation
    - Service factory creation
    """

    def __init__(self, ctx: typer.Context):
        """Initialize CLI service context from Typer context.

        Args:
            ctx: Typer context with CLIContext in obj
        """
        self._ctx = ctx
        self._config: TownElderConfig | None = None
        self._service_factory: ServiceFactory | None = None

    @property
    def data_dir(self) -> Path | None:
        """Get data directory from CLI context."""
        if self._ctx.obj is not None and isinstance(self._ctx.obj, CLIContext):
            return self._ctx.obj.data_dir
        return None

    def get_config(self) -> TownElderConfig:
        """Load and return config, validating database is initialized.

        Raises:
            DatabaseNotInitializedError: If database not initialized
            ConfigError: If config cannot be loaded
        """
        if self._config is None:
            self._config = get_config(data_dir=self.data_dir)

        # Validate database is initialized
        if not self._config.data_dir.exists():
            raise DatabaseNotInitializedError(
                f"Database not initialized at {self._config.data_dir}"
            )

        return self._config

    def get_service_factory(self) -> ServiceFactory:
        """Get or create service factory.

        Raises:
            DatabaseNotInitializedError: If database not initialized
            ServiceInitError: If services cannot be created
        """
        if self._service_factory is None:
            # Ensure config is loaded first (validates DB exists)
            self.get_config()
            try:
                self._service_factory = get_service_factory(data_dir=self.data_dir)
            except Exception as e:
                raise ServiceInitError(str(e)) from e

        return self._service_factory

    def create_embedder(self) -> Embedder:
        """Create an embedder service."""
        return self.get_service_factory().create_embedder()

    def create_vector_store(self) -> ZvecStore:
        """Create a vector store service."""
        return self.get_service_factory().create_vector_store()

    def create_git_runner(self, repo_path: Path | str | None = None) -> GitRunner:
        """Create a git runner service."""
        return self.get_service_factory().create_git_runner(repo_path)

    def create_diff_parser(self) -> DiffParser:
        """Create a diff parser service."""
        return self.get_service_factory().create_diff_parser()


@contextmanager
def get_cli_services(
    ctx: typer.Context,
    include_embedder: bool = True,
) -> Generator[tuple[CLIServiceContext, Embedder | None, ZvecStore], None, None]:
    """Context manager for CLI services with automatic cleanup.

    Provides validated config, embedder (optional), and vector store with
    automatic cleanup on exit.

    Usage:
        with get_cli_services(ctx) as (svc, embedder, store):
            # use services
            results = store.search(vector, top_k=5)

    Args:
        ctx: Typer context
        include_embedder: Whether to create the embedder (default: True).
            Set to False for commands that don't need embedding (e.g., stats, export).

    Yields:
        Tuple of (CLIServiceContext, Embedder | None, ZvecStore)

    Raises:
        typer.Exit: If services cannot be initialized
    """
    svc = CLIServiceContext(ctx)

    # Validate config/DB first
    try:
        svc.get_config()
    except DatabaseNotInitializedError:
        error_console.print("[red]Error: Database not initialized[/red]")
        console.print("[dim]Run 'te init' first to initialize the database[/dim]")
        raise typer.Exit(code=EXIT_ERROR)
    except ConfigError as e:
        error_console.print("[red]Error: Database not initialized[/red]")
        console.print(f"[dim]{e}[/dim]")
        raise typer.Exit(code=EXIT_ERROR)

    # Create services
    embedder = None
    try:
        if include_embedder:
            embedder = svc.create_embedder()
        store = svc.create_vector_store()
    except ServiceInitError as e:
        console.print(f"[red]Error opening storage:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)
    except Exception as e:
        console.print(f"[red]Error opening storage:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)

    try:
        yield svc, embedder, store
    finally:
        store.close()


def require_initialized(ctx: typer.Context) -> TownElderConfig:
    """Validate that database is initialized, returning config.

    Raises typer.Exit if not initialized.

    Args:
        ctx: Typer context

    Returns:
        Validated TownElderConfig

    Raises:
        typer.Exit: If database not initialized
    """
    svc = CLIServiceContext(ctx)
    try:
        return svc.get_config()
    except DatabaseNotInitializedError:
        error_console.print("[red]Error: Database not initialized[/red]")
        console.print("[dim]Run 'te init' first to initialize the database[/dim]")
        raise typer.Exit(code=EXIT_ERROR)
    except ConfigError as e:
        error_console.print("[red]Error: Database not initialized[/red]")
        console.print(f"[dim]{e}[/dim]")
        raise typer.Exit(code=EXIT_ERROR)
