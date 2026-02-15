"""CLI entry point for Town Elder."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import typer
from rich.console import Console

from town_elder.config import get_config
from town_elder.services import get_service_factory

# Exit codes
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_INVALID_ARG = 2

app = typer.Typer(
    name="te",
    help="Local-first semantic memory CLI for AI coding agents",
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console(stderr=False)  # stdout for normal output
error_console = Console(stderr=True)  # stderr for errors

# Global data directory option - DEPRECATED, use CLIContext instead
_data_dir: Path | None = None


def _is_te_hook(content: str) -> bool:
    """Check if hook content is a Town Elder commit-index hook.

    Uses robust detection that handles:
    - Extra arguments like --data-dir between te and commit-index
    - python -m town_elder invocation
    - uv run te invocation
    - Absolute interpreter paths (e.g., /usr/bin/python or sys.executable)
    - te command invocation
    """
    import re

    # Match patterns for te/town_elder hooks:
    # - te commit-index
    # - uv run te commit-index
    # - python -m town_elder commit-index
    # - /absolute/path/python -m town_elder commit-index
    # - te --data-dir /path commit-index
    # - uv run te --data-dir /path commit-index
    # - python -m town_elder --data-dir /path commit-index
    patterns = [
        # te commit-index (no extra args)
        r'\bte\s+commit-index\b',
        # te [args...] commit-index (with extra args)
        r'\bte\s+\S+.*\s+commit-index\b',
        # uv run te commit-index (no extra args)
        r'\buv\s+run\s+te\s+commit-index\b',
        # uv run te [args...] commit-index (with extra args)
        r'\buv\s+run\s+te\s+\S+.*\s+commit-index\b',
        # python -m town_elder commit-index (no extra args)
        r'\bpython\s+-m\s+town_elder\s+commit-index\b',
        # python -m town_elder [args...] commit-index (with extra args)
        r'\bpython\s+-m\s+town_elder\s+\S+.*\s+commit-index\b',
        # /absolute/path/python -m town_elder commit-index (no extra args)
        r'/[^\\s]+/python\s+-m\s+town_elder\s+commit-index\b',
        # /absolute/path/python -m town_elder [args...] commit-index (with extra args)
        r'/[^\\s]+/python\s+-m\s+town_elder\s+\S+.*\s+commit-index\b',
    ]

    return any(re.search(pattern, content) for pattern in patterns)


def set_data_dir(path: Path | str | None) -> None:
    """Set the global data directory.

    DEPRECATED: This sets module-global state which leaks across invocations.
    Use CLIContext for invocation-scoped data directory instead.
    """
    global _data_dir
    _data_dir = Path(path).expanduser() if path else None


def _get_data_dir_from_context(ctx: typer.Context) -> Path | None:
    """Get data directory from Typer context, falling back to deprecated global.

    This function implements invocation-scoped data directory resolution:
    1. First checks ctx.obj (new context-based approach)
    2. Falls back to deprecated _data_dir global
    3. Returns None if neither is set (will use default resolution)
    """
    # Check new context-based approach first
    if ctx.obj is not None and hasattr(ctx.obj, "data_dir"):
        return ctx.obj.data_dir

    # Fall back to deprecated global
    return _data_dir


def _get_repo_id(repo_path: Path) -> str:
    """Generate a deterministic ID for a repository based on its canonical path.

    Uses SHA256 hash of the resolved absolute path to create a consistent
    identifier that can be used for repo-scoped state storage.
    """
    canonical = str(repo_path.resolve())
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _load_index_state(state_file: Path, repo_path: Path) -> tuple[str | None, dict]:
    """Load incremental index state, scoped by repository.

    Handles migration from legacy global format to repo-scoped format.

    Returns:
        Tuple of (last_indexed_commit, raw_state_dict).
    """
    last_indexed = None

    if not state_file.exists():
        return last_indexed, {"repos": {}}

    try:
        state = json.loads(state_file.read_text())
    except Exception:
        # Ignore invalid state file, treat as empty
        return last_indexed, {"repos": {}}

    # Check for legacy format (global last_indexed_commit key)
    if "last_indexed_commit" in state and "repos" not in state:
        # Migration: convert legacy format to repo-scoped
        repo_id = _get_repo_id(repo_path)
        legacy_commit = state["last_indexed_commit"]
        migrated_state = {
            "repos": {
                repo_id: {
                    "last_indexed_commit": legacy_commit,
                    "repo_path": str(repo_path.resolve()),
                    "updated_at": state.get("updated_at", ""),
                }
            }
        }
        # Save migrated state to file so subsequent reads work correctly
        state_file.write_text(json.dumps(migrated_state))
        return legacy_commit, migrated_state

    # New repo-scoped format
    if "repos" not in state:
        state = {"repos": {}}

    repo_id = _get_repo_id(repo_path)
    repo_state = state["repos"].get(repo_id, {})
    last_indexed = repo_state.get("last_indexed_commit")

    return last_indexed, state


def _save_index_state(state_file: Path, repo_path: Path, frontier_commit_hash: str) -> None:
    """Save incremental index state, scoped by repository.

    Creates or updates the repo-scoped entry in the state file.
    """
    # Load existing state (to preserve other repos' state)
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
        except Exception:
            state = {"repos": {}}
    else:
        state = {"repos": {}}

    if "repos" not in state:
        state = {"repos": {}}

    # Ensure we have a valid dict (could be empty string from migration)
    if not isinstance(state["repos"], dict):
        state["repos"] = {}

    repo_id = _get_repo_id(repo_path)
    from datetime import datetime, timezone

    state["repos"][repo_id] = {
        "last_indexed_commit": frontier_commit_hash,
        "repo_path": str(repo_path.resolve()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    state_file.write_text(json.dumps(state))


class CLIContext:
    """Invocation-scoped context for CLI state.

    Replaces module-global _data_dir to prevent leakage across CLI invocations.
    """

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir


def _is_safe_te_storage_path(data_dir: Path, init_path: Path) -> bool:
    """Validate that data_dir is a safe te-managed storage location.

    Only allows deletion of directories that are clearly te-managed storage:
    1. Default .town_elder directory in init path
    2. Other hidden directories (starting with '.') in init path
    3. Paths explicitly containing '.town_elder' in their name

    Fails for broad paths like home directory, repo root, /tmp, etc.
    """
    resolved_data_dir = data_dir.resolve()
    resolved_init_path = init_path.resolve()

    # Check if it's the default .town_elder in init path
    if resolved_data_dir == resolved_init_path / ".town_elder":
        return True

    # Check if it's a hidden directory in init path
    if resolved_data_dir.parent == resolved_init_path and resolved_data_dir.name.startswith("."):
        return True

    # Check if path explicitly contains .town_elder (user-specified te storage)
    if ".town_elder" in resolved_data_dir.parts:
        return True

    # Check if it's a hidden directory somewhere in init path
    return (resolved_init_path in resolved_data_dir.parents or resolved_data_dir.parent == resolved_init_path) and any(
        part.startswith(".") for part in resolved_data_dir.parts
    )


def _escape_rich(text: str) -> str:
    """Escape brackets to prevent Rich markup interpretation."""
    return text.replace("[", "\\[").replace("]", "\\]")


def _run_search(
    ctx: typer.Context,
    query: str,
    top_k: int,
) -> None:
    """Shared implementation for search-style commands."""
    from town_elder.exceptions import ConfigError

    data_dir = _get_data_dir_from_context(ctx)
    try:
        config = get_config(data_dir=data_dir)
    except ConfigError as e:
        error_console.print("[red]Error: Database not initialized[/red]")
        console.print(f"[dim]{e}[/dim]")
        raise typer.Exit(code=EXIT_ERROR)

    # Validate that database is initialized
    if not config.data_dir.exists():
        error_console.print("[red]Error: Database not initialized[/red]")
        console.print("[dim]Run 'te init' first to initialize the database[/dim]")
        raise typer.Exit(code=EXIT_ERROR)

    try:
        services = get_service_factory(data_dir=data_dir)
        embedder = services.create_embedder()
        store = services.create_vector_store()
    except Exception as e:
        console.print(f"[red]Error opening storage:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)

    try:
        # Embed the query
        query_vector = embedder.embed_single(query)

        # Search
        results = store.search(query_vector, top_k=top_k)
    except Exception as e:
        console.print(f"[red]Error during search:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)
    finally:
        store.close()

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    console.print(f"[bold]Search results for:[/bold] {query}")
    for i, result in enumerate(results, 1):
        console.print(f"\n[bold]{i}.[/bold] Score: {result['score']:.3f}")
        # Escape brackets in text to avoid Rich markup interpretation
        text_snippet = result["text"][:200].replace("[", "\\[").replace("]", "\\]")
        console.print(f"   {text_snippet}...")


def _run_stats(ctx: typer.Context) -> None:
    """Shared implementation for stats-style commands."""
    from town_elder.exceptions import ConfigError

    data_dir = _get_data_dir_from_context(ctx)
    try:
        config = get_config(data_dir=data_dir)
    except ConfigError as e:
        error_console.print("[red]Error: Database not initialized[/red]")
        console.print(f"[dim]{e}[/dim]")
        raise typer.Exit(code=EXIT_ERROR)

    # Validate that database is initialized
    if not config.data_dir.exists():
        error_console.print("[red]Error: Database not initialized[/red]")
        console.print("[dim]Run 'te init' first to initialize the database[/dim]")
        raise typer.Exit(code=EXIT_ERROR)

    try:
        services = get_service_factory(data_dir=data_dir)
        store = services.create_vector_store()
    except Exception as e:
        console.print(f"[red]Error opening storage:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)

    try:
        count = store.count()
    except Exception as e:
        console.print(f"[red]Error getting stats:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)
    finally:
        store.close()

    console.print("[bold]Town Elder Statistics[/bold]")
    console.print(f"  Documents: {count}")
    # Escape brackets in path to avoid Rich markup interpretation
    data_dir_str = str(config.data_dir).replace("[", "\\[").replace("]", "\\]")
    console.print(f"  Data directory: {data_dir_str}")
    console.print(f"  Embedding model: {config.embed_model}")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    data_dir: str | None = typer.Option(
        None,
        "--data-dir",
        "-d",
        help="Data directory (default: .town_elder in current directory)",
    ),
):
    """te - Semantic memory for AI agents.

    A local-first semantic memory CLI for AI coding agents.
    """
    # Use invocation-scoped context instead of global to prevent data-dir leakage
    ctx.obj = CLIContext(data_dir=Path(data_dir).expanduser() if data_dir else None)

    if ctx.invoked_subcommand is None:
        console.print("[bold]Town Elder[/bold] - Semantic memory CLI")
        console.print("Use --help for usage information")
        raise typer.Exit(code=EXIT_SUCCESS)


@app.command(context_settings={"help_option_names": ["-h", "--help"]})
def init(  # noqa: PLR0912
    ctx: typer.Context,
    path: str = typer.Option(
        ".",
        "--path",
        "-p",
        help="Directory to initialize (default: current directory)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing database if it already exists",
    ),
    install_hook: bool = typer.Option(
        False,
        "--install-hook",
        help="Also install post-commit hook for automatic indexing",
    ),
) -> None:
    """Initialize a Town Elder database in the specified directory.

    Creates a hidden .town_elder directory with vector storage.
    Optionally installs a post-commit hook for automatic commit indexing.
    """
    from town_elder.storage import ZvecStore

    init_path = Path(path).resolve()

    # Validate path
    if not init_path.exists():
        error_console.print(f"[red]Error: Path does not exist: {path}[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    if not init_path.is_dir():
        error_console.print(f"[red]Error: Path is not a directory: {path}[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    # Determine data directory
    data_dir = _get_data_dir_from_context(ctx)
    if data_dir is None:
        data_dir = init_path / ".town_elder"

    # Track whether this is a reinitialization
    is_reinit = data_dir.exists()

    if data_dir.exists() and not force:
        error_console.print(f"[yellow]Already initialized at {data_dir}[/yellow]")
        console.print("Use --force to overwrite existing database")
        raise typer.Exit(code=EXIT_ERROR)

    # If --force is used and directory exists, clear it first
    if force and data_dir.exists():
        # Validate that data_dir is a safe te-managed storage location
        if not _is_safe_te_storage_path(data_dir, init_path):
            error_console.print(f"[red]Error: Refusing to delete unsafe data-dir path:[/red] {_escape_rich(str(data_dir))}")
            console.print("[yellow]For safety, --force only allows deletion of te-managed storage:[/yellow]")
            console.print("  - Default .town_elder directory in the init path")
            console.print("  - Other hidden directories (starting with '.') in the init path")
            console.print("  - Paths explicitly containing '.town_elder' in their name")
            console.print("[dim]Use a safe path or initialize without --force[/dim]")
            raise typer.Exit(code=EXIT_INVALID_ARG)
        import shutil
        try:
            shutil.rmtree(data_dir)
        except PermissionError as e:
            error_console.print(f"[red]Error: Cannot remove existing directory:[/red] {_escape_rich(str(e))}")
            raise typer.Exit(code=EXIT_ERROR)

    try:
        data_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        error_console.print(f"[red]Error: Cannot create directory:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)

    try:
        store = ZvecStore(data_dir / "vectors")
        store.close()
    except Exception as e:
        console.print(f"[red]Error initializing storage:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)

    status = "Reinitialized" if is_reinit else "Initialized"
    console.print(f"[green]{status} Town Elder database at {data_dir}[/green]")

    # Optionally install hook
    if install_hook:
        git_dir = init_path / ".git"
        if not git_dir.exists():
            console.print("[yellow]Warning: Not a git repository, skipping hook installation[/yellow]")
        else:
            try:
                import os
                hooks_dir = git_dir / "hooks"
                hook_path = hooks_dir / "post-commit"

                hooks_dir.mkdir(parents=True, exist_ok=True)

                # Check if hook already exists
                if hook_path.exists():
                    console.print("[yellow]Warning: Hook already exists, skipping. Use 'te hook install --force' to overwrite[/yellow]")
                else:
                    # Use uv run for robustness across uv/pyenv environments
                    # where bare 'python' may not be available on PATH
                    # Properly quote path to handle spaces
                    hook_content = f"""#!/bin/sh
# Town Elder post-commit hook - automatically indexes commits
uv run te --data-dir "{data_dir}" commit-index --repo "$(git rev-parse --show-toplevel)"
"""
                    hook_path.write_text(hook_content)
                    os.chmod(hook_path, 0o755)
                    console.print(f"[green]Installed post-commit hook at {hook_path}[/green]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not install hook: {e}[/yellow]")

    console.print("[dim]You can now use 'te add' to add documents[/dim]")
    if not install_hook:
        console.print("[dim]Run 'te init --install-hook' to enable automatic commit indexing[/dim]")


@app.command()
def search(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Search query text"),
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        help="Number of results to return (default: 5)",
    ),
) -> None:
    """Search for similar documents in the vector store.

    Embeds the query text and finds the most similar documents.
    """
    _run_search(ctx, query=query, top_k=top_k)


@app.command("query")
def query(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Search query text"),
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        help="Number of results to return (default: 5)",
    ),
) -> None:
    """Query for similar documents in the vector store.

    Alias for 'search' command. Embeds the query text and finds the most similar documents.
    """
    _run_search(ctx, query=query, top_k=top_k)


@app.command()
def add(
    ctx: typer.Context,
    text: str = typer.Option(
        ...,
        "--text",
        "-t",
        help="Text content to add to the vector store",
    ),
    metadata: str = typer.Option(
        "",
        "--metadata",
        "-m",
        help="JSON metadata string (must be valid JSON)",
    ),
) -> None:
    """Add a document to the vector store.

    Embeds the text and stores it with optional metadata.
    """
    import uuid

    data_dir = _get_data_dir_from_context(ctx)
    config = get_config(data_dir=data_dir)

    # Validate that database is initialized
    if not config.data_dir.exists():
        error_console.print("[red]Error: Database not initialized[/red]")
        console.print("[dim]Run 'te init' first to initialize the database[/dim]")
        raise typer.Exit(code=EXIT_ERROR)

    # Parse metadata with better error message
    meta = {}
    if metadata:
        try:
            meta = json.loads(metadata)
        except json.JSONDecodeError as e:
            error_console.print("[red]Error: Invalid JSON metadata[/red]")
            error_console.print(f"[dim]JSON parse error at position {e.pos}: {e.msg}[/dim]")
            raise typer.Exit(code=EXIT_INVALID_ARG)

        # Validate metadata is an object (dict), not a list or string
        if not isinstance(meta, dict):
            error_console.print(f"[red]Error: Metadata must be a JSON object (dict), not {type(meta).__name__}[/red]")
            raise typer.Exit(code=EXIT_INVALID_ARG)

    try:
        services = get_service_factory(data_dir=data_dir)
        embedder = services.create_embedder()
        store = services.create_vector_store()
    except Exception as e:
        console.print(f"[red]Error opening storage:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)

    try:
        # Embed the text
        doc_id = meta.get("id", str(uuid.uuid4()))
        vector = embedder.embed_single(text)

        # Store
        store.insert(doc_id, vector, text, meta)
    except Exception as e:
        console.print(f"[red]Error storing document:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)
    finally:
        store.close()

    console.print(f"[green]Added document: {doc_id}[/green]")


@app.command()
def stats(ctx: typer.Context) -> None:
    """Show indexing statistics and storage info.

    Displays the number of documents and configuration details.
    """
    _run_stats(ctx)


@app.command("status")
def status(ctx: typer.Context) -> None:
    """Show indexing statistics and storage info.

    Alias for 'stats' command. Displays the number of documents and configuration details.
    """
    _run_stats(ctx)


@app.command()
def index(  # noqa: PLR0912
    ctx: typer.Context,
    path: str = typer.Argument(".", help="Path to directory to index (default: current directory)"),
    exclude: list[str] = typer.Option(
        None,
        "--exclude",
        "-e",
        help="Additional patterns to exclude (can be specified multiple times)",
    ),
) -> None:
    """Index files from a directory.

    Recursively indexes all .py and .md files in the specified directory.
    Excludes .git, .venv, __pycache__, node_modules, and other common ignore patterns.
    """
    data_dir = _get_data_dir_from_context(ctx)
    config = get_config(data_dir=data_dir)

    # Validate that database is initialized
    if not config.data_dir.exists():
        error_console.print("[red]Error: Database not initialized[/red]")
        console.print("[dim]Run 'te init' first to initialize the database[/dim]")
        raise typer.Exit(code=EXIT_ERROR)

    index_path = Path(path).resolve()
    if not index_path.exists():
        error_console.print(f"[red]Error: Path does not exist: {path}[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    if not index_path.is_dir():
        error_console.print(f"[red]Error: Path is not a directory: {path}[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    try:
        services = get_service_factory(data_dir=data_dir)
        embedder = services.create_embedder()
        store = services.create_vector_store()
    except Exception as e:
        console.print(f"[red]Error opening storage:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)

    # Default exclusion patterns
    default_excludes = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".tox", "venv", ".env", ".eggs", "*.egg-info", ".hg", ".svn", ".bzr", "vendor"}
    if exclude:
        default_excludes.update(exclude)

    def should_exclude(path: Path) -> bool:
        """Check if path matches any exclusion pattern."""
        parts = path.parts
        for pattern in default_excludes:
            if pattern.startswith("*"):
                # Handle glob patterns like *.egg-info
                if any(part.endswith(pattern[1:]) for part in parts):
                    return True
            else:
                # Handle directory name patterns
                if pattern in parts:
                    return True
        return False

    # Find all text files with exclusion filtering
    all_files = list(index_path.rglob("*.py")) + list(index_path.rglob("*.md"))
    excluded_files = [f for f in all_files if should_exclude(f)]
    files_to_index = [f for f in all_files if not should_exclude(f)]

    console.print(f"[green]Indexing {len(files_to_index)} files (attempted {len(all_files)}, excluded {len(excluded_files)})...[/green]")

    indexed_count = 0
    skipped_count = 0
    try:
        for file in files_to_index:
            try:
                # Skip binary or unreadable files
                if not file.is_file():
                    skipped_count += 1
                    continue
                text = file.read_text()
                # Use stable ID based on file path hash for idempotent indexing
                # zvec requires alphanumeric doc_ids, so we hash the path
                file_path_str = str(file)
                doc_id = hashlib.sha256(file_path_str.encode()).hexdigest()[:16]
                vector = embedder.embed_single(text)
                store.upsert(
                    doc_id, vector, text,
                    {"source": str(file), "type": file.suffix}
                )
                indexed_count += 1
            except UnicodeDecodeError:
                skipped_count += 1
                error_console.print(f"[yellow]Skipped binary file: {file}[/yellow]")
            except PermissionError:
                skipped_count += 1
                error_console.print(f"[yellow]Skipped unreadable file: {file}[/yellow]")
            except Exception as e:
                skipped_count += 1
                error_console.print(f"[yellow]Skipped {file}: {e}[/yellow]")
    except Exception as e:
        console.print(f"[red]Error during indexing:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)
    finally:
        store.close()

    if skipped_count > 0:
        console.print(f"[green]Indexed {indexed_count} files, skipped {skipped_count} (excluded {len(excluded_files)})[/green]")
    else:
        console.print(f"[green]Indexed {indexed_count} files (excluded {len(excluded_files)})[/green]")


@app.command()
def commit_index(  # noqa: PLR0912
    ctx: typer.Context,
    path: str = typer.Option(
        ".",
        "--repo",
        "-r",
        help="Git repository path (default: current directory)",
    ),
    limit: int = typer.Option(
        100,
        "--limit",
        "-n",
        help="Number of commits to index (default: 100)",
    ),
    incremental: bool = typer.Option(
        True,
        "--incremental/--full",
        help="Only index new commits since last run (default: incremental)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-indexing of all commits, ignoring last indexed state",
    ),
) -> None:
    """Index git commits from a repository.

    Parses commit messages and diffs to create searchable commit history.
    """
    data_dir = _get_data_dir_from_context(ctx)
    config = get_config(data_dir=data_dir)

    # Validate that database is initialized
    if not config.data_dir.exists():
        error_console.print("[red]Error: Database not initialized[/red]")
        console.print("[dim]Run 'te init' first to initialize the database[/dim]")
        raise typer.Exit(code=EXIT_ERROR)

    repo_path = Path(path).resolve()

    if not (repo_path / ".git").exists():
        error_console.print(f"[red]Error: Not a git repository: {path}[/red]")
        console.print("[dim]Ensure the path contains a .git directory[/dim]")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    # Load last indexed commit from state file (repo-scoped)
    state_file = config.data_dir / "index_state.json"
    last_indexed = None
    if incremental and not force:
        last_indexed, _ = _load_index_state(state_file, repo_path)

    try:
        services = get_service_factory(data_dir=data_dir)
        git = services.create_git_runner(repo_path)
        embedder = services.create_embedder()
        store = services.create_vector_store()
        diff_parser = services.create_diff_parser()
    except Exception as e:
        console.print(f"[red]Error initializing services:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)

    try:
        # Get commits - in incremental mode, fetch enough to find last_indexed
        all_commits = git.get_commits(limit=limit)
    except Exception as e:
        console.print(f"[red]Error fetching commits:[/red] {_escape_rich(str(e))}")
        store.close()
        raise typer.Exit(code=EXIT_ERROR)

    # Filter commits if using incremental mode
    commits = all_commits
    sentinel_found = True  # Assume found unless we determine otherwise
    if incremental and last_indexed and not force:
        # Find the position of last indexed commit
        found_last = False
        filtered = []
        for commit in all_commits:
            if commit.hash == last_indexed:
                found_last = True
                break
            filtered.append(commit)
        # If we didn't find the last indexed commit, we may have a backlog
        # Fetch more commits until we find it or exhaust all
        if not found_last:
            # Sentinel not found - this is a stale/missing state situation
            sentinel_found = False
            # Continue fetching in batches until we find last_indexed or run out
            offset = limit
            while not found_last:
                more_commits = git.get_commits(limit=limit, offset=offset)
                if not more_commits:
                    break
                for commit in more_commits:
                    if commit.hash == last_indexed:
                        found_last = True
                        break
                    filtered.append(commit)
                offset += limit
                if found_last:
                    break
                # Check if we've reached the end
                if len(more_commits) < limit:
                    break
        # If we found the last indexed commit, only index newer ones
        if found_last:
            commits = filtered
        else:
            # Sentinel still not found after exhausting all commits.
            # This means the state file points to a garbage-collected or corrupted commit.
            # We must not do partial indexing with unsafe state advance.
            # Index all available commits (from all batches) but don't advance state.
            console.print("[yellow]Warning: Last indexed commit not found in history.[/yellow]")
            console.print("[yellow]This may indicate stale state or garbage-collected commits.[/yellow]")
            console.print("[yellow]Indexing all available commits without advancing state.[/yellow]")
            # Use filtered (contains all commits from all batches), not just first page
            commits = filtered

    # Reverse to index from oldest to newest
    commits = list(reversed(commits))

    if not commits:
        console.print("[green]No new commits to index[/green]")
        store.close()
        return

    console.print(f"[green]Indexing {len(commits)} commits...[/green]")

    indexed_count = 0
    skipped_count = 0
    frontier_commit_hash: str | None = None
    frontier_blocked = False
    try:
        for commit in commits:
            try:
                # Get the diff
                diff = git.get_diff(commit.hash)
                diff_text = diff_parser.parse_diff_to_text(diff)

                # Combine with commit message
                text = f"Commit: {commit.message}\n\n{diff_text}"

                # Index
                doc_id = f"commit_{commit.hash}"
                vector = embedder.embed_single(text)
                store.insert(
                    doc_id, vector, text,
                    {
                        "type": "commit",
                        "hash": commit.hash,
                        "author": commit.author,
                        "message": commit.message,
                    }
                )
                indexed_count += 1
                if not frontier_blocked:
                    frontier_commit_hash = commit.hash
            except Exception as e:
                error_console.print(f"[yellow]Skipped commit {commit.hash[:8]}: {e}[/yellow]")
                skipped_count += 1
                frontier_blocked = True
    except Exception as e:
        console.print(f"[red]Error during indexing:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)
    finally:
        store.close()

    # Save last indexed commit state only if sentinel was found.
    # This prevents unsafe state advance when sentinel is missing/stale.
    if frontier_commit_hash and sentinel_found:
        _save_index_state(state_file, repo_path, frontier_commit_hash)

    if skipped_count > 0:
        console.print(f"[green]Indexed {indexed_count} commits, skipped {skipped_count}[/green]")
    else:
        console.print(f"[green]Indexed {indexed_count} commits[/green]")



hook_app = typer.Typer(name="hook", help="Manage git hooks for automatic indexing")


@hook_app.command()
def install(
    ctx: typer.Context,
    path: str = typer.Option(".", "--repo", "-r", help="Git repository path"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing hook"),
) -> None:
    """Install post-commit hook for automatic commit indexing."""
    import os

    repo_path = Path(path).resolve()
    git_dir = repo_path / ".git"
    hooks_dir = git_dir / "hooks"
    hook_path = hooks_dir / "post-commit"

    # Check if it's a git repository
    if not git_dir.exists():
        console.print(f"[red]Not a git repository: {path}[/red]")
        raise typer.Exit(code=EXIT_ERROR)

    # Create hooks directory if it doesn't exist
    hooks_dir.mkdir(parents=True, exist_ok=True)

    # Check if hook already exists
    if hook_path.exists() and not force:
        error_console.print(f"[yellow]Hook already exists at {hook_path}[/yellow]")
        console.print("Use --force to overwrite, or run 'te hook uninstall' first")
        raise typer.Exit(code=EXIT_ERROR)

    # Check if it's a Town Elder hook (ours or someone else's)
    if hook_path.exists() and force:
        existing_content = hook_path.read_text()
        if not _is_te_hook(existing_content):
            error_console.print("[yellow]Warning: Existing hook is not a Town Elder hook[/yellow]")
            console.print("Use --force to overwrite anyway")

    # Get the data directory from context (invocation-scoped)
    from town_elder.exceptions import ConfigError

    data_dir = _get_data_dir_from_context(ctx)
    try:
        config = get_config(data_dir=data_dir)
    except ConfigError as e:
        error_console.print("[red]Error: Database not initialized[/red]")
        console.print(f"[dim]{e}[/dim]")
        raise typer.Exit(code=EXIT_ERROR)

    # Determine the data_dir to use in the hook - use absolute path for reliability
    # Use uv run for robustness across uv/pyenv environments
    # where bare 'python' may not be available on PATH
    # Properly quote paths to handle spaces
    if data_dir:
        data_dir_arg = f'--data-dir "{config.data_dir}"'
        hook_content = f"""#!/bin/sh
# Town Elder post-commit hook - automatically indexes commits
uv run te {data_dir_arg} commit-index --repo "$(git rev-parse --show-toplevel)"
"""
    else:
        hook_content = """#!/bin/sh
# Town Elder post-commit hook - automatically indexes commits
uv run te commit-index --repo "$(git rev-parse --show-toplevel)"
"""

    hook_path.write_text(hook_content)

    # Make it executable
    os.chmod(hook_path, 0o755)

    console.print(f"[green]Installed post-commit hook at {hook_path}[/green]")
    console.print("Commits will now be automatically indexed")


@hook_app.command()
def uninstall(
    path: str = typer.Option(".", "--repo", "-r", help="Git repository path"),
    force: bool = typer.Option(False, "--force", "-f", help="Allow deletion of non-Town Elder hooks"),
) -> None:
    """Remove post-commit hook."""
    repo_path = Path(path).resolve()
    hook_path = repo_path / ".git" / "hooks" / "post-commit"

    if not hook_path.exists():
        error_console.print(f"[yellow]No post-commit hook found at {hook_path}[/yellow]")
        return

    # Check if it's a Town Elder hook
    content = hook_path.read_text()
    is_te_hook = _is_te_hook(content)

    if not is_te_hook and not force:
        error_console.print("[red]Error: Hook exists but is not a Town Elder hook[/red]")
        console.print("[yellow]Use --force to delete non-Town Elder hooks[/yellow]")
        raise typer.Exit(code=EXIT_ERROR)

    if not is_te_hook and force:
        error_console.print("[yellow]Warning: Deleting non-Town Elder hook (--force specified)[/yellow]")

    hook_path.unlink()
    console.print("[green]Removed post-commit hook[/green]")


@hook_app.command("status")
def hook_status(
    path: str = typer.Option(".", "--repo", "-r", help="Git repository path"),
) -> None:
    """Check if post-commit hook is installed."""
    repo_path = Path(path).resolve()
    git_dir = repo_path / ".git"

    if not git_dir.exists():
        console.print(f"[red]Not a git repository: {path}[/red]")
        raise typer.Exit(code=EXIT_ERROR)

    hook_path = git_dir / "hooks" / "post-commit"

    if not hook_path.exists():
        console.print("[bold]Hook status:[/bold] Not installed")
        console.print("Run 'te hook install' to install")
        return

    console.print("[bold]Hook status:[/bold] Installed")
    console.print(f"Hook path: {hook_path}")

    content = hook_path.read_text()
    if _is_te_hook(content):
        console.print("Hook type: Town Elder (automatic indexing)")
    else:
        console.print("Hook type: Unknown (not a Town Elder hook)")


app.add_typer(hook_app, name="hook")


@app.command()
def export(  # noqa: PLR0912
    ctx: typer.Context,
    output: str = typer.Option(
        "-",
        "--output",
        "-o",
        help="Output file path (default: stdout). Use .jsonl extension for JSONL format.",
    ),
    format: str | None = typer.Option(
        None,
        "--format",
        "-f",
        help="Export format: json or jsonl (default: json, or auto-detect from extension)",
    ),
    include_vectors: bool = typer.Option(
        False,
        "--include-vectors",
        help="Include embedding vectors in export (increases file size significantly)",
    ),
) -> None:
    """Export indexed data to a file.

    Exports all documents from the vector store to JSON or JSONL format.
    """
    from town_elder.exceptions import ConfigError

    data_dir = _get_data_dir_from_context(ctx)
    try:
        config = get_config(data_dir=data_dir)
    except ConfigError as e:
        error_console.print("[red]Error: Database not initialized[/red]")
        console.print(f"[dim]{e}[/dim]")
        raise typer.Exit(code=EXIT_ERROR)

    # Validate that database is initialized
    if not config.data_dir.exists():
        error_console.print("[red]Error: Database not initialized[/red]")
        console.print("[dim]Run 'te init' first to initialize the database[/dim]")
        raise typer.Exit(code=EXIT_ERROR)

    # Determine format from file extension if not explicitly specified
    if output == "-" and not format:
        # Default to json when outputting to stdout
        format = "json"
    elif not format:
        format = "jsonl" if output.endswith(".jsonl") else "json"

    # Validate format
    if format not in ("json", "jsonl"):
        error_console.print(f"[red]Error: Invalid format '{format}'. Use 'json' or 'jsonl'.[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    try:
        services = get_service_factory(data_dir=data_dir)
        store = services.create_vector_store()
    except Exception as e:
        console.print(f"[red]Error opening storage:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)

    try:
        documents = store.get_all(include_vectors=include_vectors)
    except Exception as e:
        console.print(f"[red]Error exporting data:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)
    finally:
        store.close()

    # Serialize to chosen format
    if format == "jsonl":
        output_data = "\n".join(json.dumps(doc) for doc in documents)
    else:
        output_data = json.dumps(documents, indent=2)

    # Write output
    if output == "-":
        # Use built-in print to avoid Rich markup interpretation
        print(output_data)
    else:
        try:
            Path(output).write_text(output_data)
            console.print(f"[green]Exported {len(documents)} documents to {output}[/green]")
        except Exception as e:
            console.print(f"[red]Error writing to file:[/red] {_escape_rich(str(e))}")
            raise typer.Exit(code=EXIT_ERROR)


def run() -> None:
    """Console script entry point."""
    app()


if __name__ == "__main__":
    run()
