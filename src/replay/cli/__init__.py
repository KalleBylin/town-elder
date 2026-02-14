"""CLI entry point for replay."""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from replay.config import get_config
from replay.services import get_service_factory

# Exit codes
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_INVALID_ARG = 2

app = typer.Typer(
    name="replay",
    help="Local-first semantic memory CLI for AI coding agents",
    add_completion=False,
)
console = Console(stderr=False)  # stdout for normal output
error_console = Console(stderr=True)  # stderr for errors

# Global data directory option
_data_dir: Path | None = None


def set_data_dir(path: Path | str | None) -> None:
    """Set the global data directory."""
    global _data_dir
    _data_dir = Path(path).expanduser() if path else None


def _escape_rich(text: str) -> str:
    """Escape brackets to prevent Rich markup interpretation."""
    return text.replace("[", "\\[").replace("]", "\\]")


def _run_search(
    query: str,
    top_k: int,
    author: str | None,
    path: str | None,
    since: str | None,
) -> None:
    """Shared implementation for search-style commands."""
    config = get_config(data_dir=_data_dir)

    # Validate that database is initialized
    if not config.data_dir.exists():
        error_console.print("[red]Error: Database not initialized[/red]")
        console.print("[dim]Run 'replay init' first to initialize the database[/dim]")
        raise typer.Exit(code=EXIT_ERROR)

    try:
        services = get_service_factory(data_dir=_data_dir)
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


def _run_stats() -> None:
    """Shared implementation for stats-style commands."""
    config = get_config(data_dir=_data_dir)

    # Validate that database is initialized
    if not config.data_dir.exists():
        error_console.print("[red]Error: Database not initialized[/red]")
        console.print("[dim]Run 'replay init' first to initialize the database[/dim]")
        raise typer.Exit(code=EXIT_ERROR)

    try:
        services = get_service_factory(data_dir=_data_dir)
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

    console.print("[bold]Replay Statistics[/bold]")
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
        help="Data directory (default: .replay in current directory or home)",
    ),
):
    """replay - Semantic memory for AI agents.

    A local-first semantic memory CLI for AI coding agents.
    """
    if data_dir:
        set_data_dir(data_dir)

    if ctx.invoked_subcommand is None:
        console.print("[bold]replay[/bold] - Semantic memory CLI")
        console.print("Use --help for usage information")
        raise typer.Exit(code=EXIT_SUCCESS)


@app.command()
def init(
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
        "-h",
        help="Also install post-commit hook for automatic indexing",
    ),
) -> None:
    """Initialize a replay database in the specified directory.

    Creates a hidden .replay directory with vector storage.
    Optionally installs a post-commit hook for automatic commit indexing.
    """
    from replay.storage import ZvecStore

    init_path = Path(path).resolve()

    # Validate path
    if not init_path.exists():
        error_console.print(f"[red]Error: Path does not exist: {path}[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    if not init_path.is_dir():
        error_console.print(f"[red]Error: Path is not a directory: {path}[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    data_dir = _data_dir or (init_path / ".replay")

    if data_dir.exists() and not force:
        error_console.print(f"[yellow]Already initialized at {data_dir}[/yellow]")
        console.print("Use --force to overwrite existing database")
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

    console.print(f"[green]Initialized replay database at {data_dir}[/green]")

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
                    console.print("[yellow]Warning: Hook already exists, skipping. Use 'replay hook install --force' to overwrite[/yellow]")
                else:
                    # Use absolute path for data_dir
                    hook_content = f"""#!/bin/sh
# Replay post-commit hook - automatically indexes commits
replay --data-dir {data_dir} commit-index --repo "$(git rev-parse --show-toplevel)"
"""
                    hook_path.write_text(hook_content)
                    os.chmod(hook_path, 0o755)
                    console.print(f"[green]Installed post-commit hook at {hook_path}[/green]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not install hook: {e}[/yellow]")

    console.print("[dim]You can now use 'replay add' to add documents[/dim]")
    if not install_hook:
        console.print("[dim]Run 'replay init --install-hook' to enable automatic commit indexing[/dim]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query text"),
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        help="Number of results to return (default: 5)",
    ),
    author: str | None = typer.Option(
        None,
        "--author",
        help="Filter results by author (not yet implemented)",
    ),
    path: str | None = typer.Option(
        None,
        "--path",
        help="Filter results by file path (not yet implemented)",
    ),
    since: str | None = typer.Option(
        None,
        "--since",
        help="Filter results by date (not yet implemented)",
    ),
) -> None:
    """Search for similar documents in the vector store.

    Embeds the query text and finds the most similar documents.
    """
    _run_search(query=query, top_k=top_k, author=author, path=path, since=since)


@app.command("query")
def query(
    query: str = typer.Argument(..., help="Search query text"),
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        help="Number of results to return (default: 5)",
    ),
    author: str | None = typer.Option(
        None,
        "--author",
        help="Filter results by author (not yet implemented)",
    ),
    path: str | None = typer.Option(
        None,
        "--path",
        help="Filter results by file path (not yet implemented)",
    ),
    since: str | None = typer.Option(
        None,
        "--since",
        help="Filter results by date (not yet implemented)",
    ),
) -> None:
    """Query for similar documents in the vector store.

    Alias for 'search' command. Embeds the query text and finds the most similar documents.
    """
    _run_search(query=query, top_k=top_k, author=author, path=path, since=since)


@app.command()
def add(
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

    config = get_config(data_dir=_data_dir)

    # Validate that database is initialized
    if not config.data_dir.exists():
        error_console.print("[red]Error: Database not initialized[/red]")
        console.print("[dim]Run 'replay init' first to initialize the database[/dim]")
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
        services = get_service_factory(data_dir=_data_dir)
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
        store.insert_with_vector(doc_id, vector, text, meta)
    except Exception as e:
        console.print(f"[red]Error storing document:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)
    finally:
        store.close()

    console.print(f"[green]Added document: {doc_id}[/green]")


@app.command()
def stats() -> None:
    """Show indexing statistics and storage info.

    Displays the number of documents and configuration details.
    """
    _run_stats()


@app.command("status")
def status() -> None:
    """Show indexing statistics and storage info.

    Alias for 'stats' command. Displays the number of documents and configuration details.
    """
    _run_stats()


@app.command()
def index(
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
    import uuid

    config = get_config(data_dir=_data_dir)

    # Validate that database is initialized
    if not config.data_dir.exists():
        error_console.print("[red]Error: Database not initialized[/red]")
        console.print("[dim]Run 'replay init' first to initialize the database[/dim]")
        raise typer.Exit(code=EXIT_ERROR)

    index_path = Path(path).resolve()
    if not index_path.exists():
        error_console.print(f"[red]Error: Path does not exist: {path}[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    if not index_path.is_dir():
        error_console.print(f"[red]Error: Path is not a directory: {path}[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    try:
        services = get_service_factory(data_dir=_data_dir)
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
                doc_id = str(uuid.uuid4())
                vector = embedder.embed_single(text)
                store.insert_with_vector(
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
def commit_index(
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
    config = get_config(data_dir=_data_dir)

    # Validate that database is initialized
    if not config.data_dir.exists():
        error_console.print("[red]Error: Database not initialized[/red]")
        console.print("[dim]Run 'replay init' first to initialize the database[/dim]")
        raise typer.Exit(code=EXIT_ERROR)

    repo_path = Path(path).resolve()

    if not (repo_path / ".git").exists():
        error_console.print(f"[red]Error: Not a git repository: {path}[/red]")
        console.print("[dim]Ensure the path contains a .git directory[/dim]")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    # Load last indexed commit from state file
    state_file = config.data_dir / "index_state.json"
    last_indexed = None
    if state_file.exists() and incremental and not force:
        import json
        try:
            state = json.loads(state_file.read_text())
            last_indexed = state.get("last_indexed_commit")
        except Exception:
            pass  # Ignore invalid state file

    try:
        services = get_service_factory(data_dir=_data_dir)
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
                store.insert_with_vector(
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

    # Save last indexed commit state based only on contiguous successful indexing.
    if frontier_commit_hash:
        import json
        state = {"last_indexed_commit": frontier_commit_hash}
        state_file.write_text(json.dumps(state))

    if skipped_count > 0:
        console.print(f"[green]Indexed {indexed_count} commits, skipped {skipped_count}[/green]")
    else:
        console.print(f"[green]Indexed {indexed_count} commits[/green]")



hook_app = typer.Typer(name="hook", help="Manage git hooks for automatic indexing")


@hook_app.command()
def install(
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
        console.print("Use --force to overwrite, or run 'replay hook uninstall' first")
        raise typer.Exit(code=EXIT_ERROR)

    # Check if it's a replay hook (ours or someone else's)
    if hook_path.exists() and force:
        existing_content = hook_path.read_text()
        if "replay commit-index" not in existing_content:
            error_console.print("[yellow]Warning: Existing hook is not a replay hook[/yellow]")
            console.print("Use --force to overwrite anyway")

    # Get the data directory that was used (or would be used by default)
    config = get_config(data_dir=_data_dir)

    # Determine the data_dir to use in the hook - use absolute path for reliability
    # Only include --data-dir if explicitly configured
    if _data_dir:
        data_dir_arg = f"--data-dir {config.data_dir}"
        hook_content = f"""#!/bin/sh
# Replay post-commit hook - automatically indexes commits
replay {data_dir_arg} commit-index --repo "$(git rev-parse --show-toplevel)"
"""
    else:
        hook_content = """#!/bin/sh
# Replay post-commit hook - automatically indexes commits
replay commit-index --repo "$(git rev-parse --show-toplevel)"
"""

    hook_path.write_text(hook_content)

    # Make it executable
    os.chmod(hook_path, 0o755)

    console.print(f"[green]Installed post-commit hook at {hook_path}[/green]")
    console.print("Commits will now be automatically indexed")


@hook_app.command()
def uninstall(
    path: str = typer.Option(".", "--repo", "-r", help="Git repository path"),
    force: bool = typer.Option(False, "--force", "-f", help="Allow deletion of non-Replay hooks"),
) -> None:
    """Remove post-commit hook."""
    repo_path = Path(path).resolve()
    hook_path = repo_path / ".git" / "hooks" / "post-commit"

    if not hook_path.exists():
        error_console.print(f"[yellow]No post-commit hook found at {hook_path}[/yellow]")
        return

    # Check if it's a replay hook
    content = hook_path.read_text()
    is_replay_hook = "replay commit-index" in content

    if not is_replay_hook and not force:
        error_console.print("[red]Error: Hook exists but is not a Replay hook[/red]")
        console.print("[yellow]Use --force to delete non-Replay hooks[/yellow]")
        raise typer.Exit(code=EXIT_ERROR)

    if not is_replay_hook and force:
        error_console.print("[yellow]Warning: Deleting non-Replay hook (--force specified)[/yellow]")

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
        console.print("Run 'replay hook install' to install")
        return

    console.print("[bold]Hook status:[/bold] Installed")
    console.print(f"Hook path: {hook_path}")

    content = hook_path.read_text()
    if "replay commit-index" in content:
        console.print("Hook type: Replay (automatic indexing)")
    else:
        console.print("Hook type: Unknown (not a replay hook)")


app.add_typer(hook_app, name="hook")


@app.command()
def export(
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
    config = get_config(data_dir=_data_dir)

    # Validate that database is initialized
    if not config.data_dir.exists():
        error_console.print("[red]Error: Database not initialized[/red]")
        console.print("[dim]Run 'replay init' first to initialize the database[/dim]")
        raise typer.Exit(code=EXIT_ERROR)

    # Determine format from file extension if not explicitly specified
    if output != "-" and not format:
        if output.endswith(".jsonl"):
            format = "jsonl"
        else:
            format = "json"

    # Validate format
    if format not in ("json", "jsonl"):
        error_console.print(f"[red]Error: Invalid format '{format}'. Use 'json' or 'jsonl'.[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    try:
        services = get_service_factory(data_dir=_data_dir)
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
        console.print(output_data)
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
