"""CLI entry point for replay."""
from __future__ import annotations

import json
import sys
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
console = Console(stderr=True)


def _escape_rich(text: str) -> str:
    """Escape brackets to prevent Rich markup interpretation."""
    return text.replace("[", "\\[").replace("]", "\\]")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """replay - Semantic memory for AI agents.

    A local-first semantic memory CLI for AI coding agents.
    """
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
) -> None:
    """Initialize a replay database in the specified directory.

    Creates a hidden .replay directory with vector storage.
    """
    from replay.storage import ZvecStore

    init_path = Path(path).resolve()

    # Validate path
    if not init_path.exists():
        console.print(f"[red]Error: Path does not exist: {path}[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    if not init_path.is_dir():
        console.print(f"[red]Error: Path is not a directory: {path}[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    data_dir = init_path / ".replay"

    if data_dir.exists() and not force:
        console.print(f"[yellow]Already initialized at {data_dir}[/yellow]")
        console.print("Use --force to overwrite existing database")
        raise typer.Exit(code=EXIT_ERROR)

    try:
        data_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        console.print(f"[red]Error: Cannot create directory:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)

    try:
        store = ZvecStore(data_dir / "vectors")
        store.close()
    except Exception as e:
        console.print(f"[red]Error initializing storage:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)

    console.print(f"[green]Initialized replay database at {data_dir}[/green]")
    console.print("[dim]You can now use 'replay add' to add documents[/dim]")


@app.command()
def search(
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
    config = get_config()

    # Validate that database is initialized
    if not config.data_dir.exists():
        console.print("[red]Error: Database not initialized[/red]")
        console.print("[dim]Run 'replay init' first to initialize the database[/dim]")
        raise typer.Exit(code=EXIT_ERROR)

    try:
        services = get_service_factory()
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

    config = get_config()

    # Validate that database is initialized
    if not config.data_dir.exists():
        console.print("[red]Error: Database not initialized[/red]")
        console.print("[dim]Run 'replay init' first to initialize the database[/dim]")
        raise typer.Exit(code=EXIT_ERROR)

    # Parse metadata with better error message
    meta = {}
    if metadata:
        try:
            meta = json.loads(metadata)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error: Invalid JSON metadata[/red]")
            console.print(f"[dim]JSON parse error at position {e.pos}: {e.msg}[/dim]")
            raise typer.Exit(code=EXIT_INVALID_ARG)

    try:
        services = get_service_factory()
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
    config = get_config()

    # Validate that database is initialized
    if not config.data_dir.exists():
        console.print("[red]Error: Database not initialized[/red]")
        console.print("[dim]Run 'replay init' first to initialize the database[/dim]")
        raise typer.Exit(code=EXIT_ERROR)

    try:
        services = get_service_factory()
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

    console.print(f"[bold]Replay Statistics[/bold]")
    console.print(f"  Documents: {count}")
    # Escape brackets in path to avoid Rich markup interpretation
    data_dir_str = str(config.data_dir).replace("[", "\\[").replace("]", "\\]")
    console.print(f"  Data directory: {data_dir_str}")
    console.print(f"  Embedding model: {config.embed_model}")


@app.command()
def index(
    path: str = typer.Argument(".", help="Path to directory to index (default: current directory)"),
) -> None:
    """Index files from a directory.

    Recursively indexes all .py and .md files in the specified directory.
    """
    import uuid

    config = get_config()

    # Validate that database is initialized
    if not config.data_dir.exists():
        console.print("[red]Error: Database not initialized[/red]")
        console.print("[dim]Run 'replay init' first to initialize the database[/dim]")
        raise typer.Exit(code=EXIT_ERROR)

    index_path = Path(path).resolve()
    if not index_path.exists():
        console.print(f"[red]Error: Path does not exist: {path}[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    if not index_path.is_dir():
        console.print(f"[red]Error: Path is not a directory: {path}[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    try:
        services = get_service_factory()
        embedder = services.create_embedder()
        store = services.create_vector_store()
    except Exception as e:
        console.print(f"[red]Error opening storage:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)

    # Find all text files
    files = list(index_path.rglob("*.py")) + list(index_path.rglob("*.md"))

    console.print(f"[green]Indexing {len(files)} files...[/green]")

    try:
        for file in files:
            try:
                text = file.read_text()
                doc_id = str(uuid.uuid4())
                vector = embedder.embed_single(text)
                store.insert_with_vector(
                    doc_id, vector, text,
                    {"source": str(file), "type": file.suffix}
                )
            except Exception as e:
                console.print(f"[yellow]Skipped {file}: {e}[/yellow]")
    except Exception as e:
        console.print(f"[red]Error during indexing:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)
    finally:
        store.close()

    console.print(f"[green]Indexed {len(files)} files[/green]")


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
) -> None:
    """Index git commits from a repository.

    Parses commit messages and diffs to create searchable commit history.
    """
    import uuid

    config = get_config()

    # Validate that database is initialized
    if not config.data_dir.exists():
        console.print("[red]Error: Database not initialized[/red]")
        console.print("[dim]Run 'replay init' first to initialize the database[/dim]")
        raise typer.Exit(code=EXIT_ERROR)

    repo_path = Path(path).resolve()

    if not (repo_path / ".git").exists():
        console.print(f"[red]Error: Not a git repository: {path}[/red]")
        console.print("[dim]Ensure the path contains a .git directory[/dim]")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    try:
        services = get_service_factory()
        git = services.create_git_runner(repo_path)
        embedder = services.create_embedder()
        store = services.create_vector_store()
        diff_parser = services.create_diff_parser()
    except Exception as e:
        console.print(f"[red]Error initializing services:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)

    try:
        # Get commits
        commits = git.get_commits(limit=limit)
    except Exception as e:
        console.print(f"[red]Error fetching commits:[/red] {_escape_rich(str(e))}")
        store.close()
        raise typer.Exit(code=EXIT_ERROR)

    console.print(f"[green]Indexing {len(commits)} commits...[/green]")

    try:
        for commit in commits:
            try:
                # Get the diff
                diff = git.get_diff(commit.hash)
                diff_text = diff_parser.parse_diff_to_text(diff)

                # Combine with commit message
                text = f"Commit: {commit.message}\n\n{diff_text}"

                # Index
                doc_id = f"commit:{commit.hash}"
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
            except Exception as e:
                console.print(f"[yellow]Skipped commit {commit.hash[:8]}: {e}[/yellow]")
    except Exception as e:
        console.print(f"[red]Error during indexing:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)
    finally:
        store.close()

    console.print(f"[green]Indexed {len(commits)} commits[/green]")



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
        console.print(f"[yellow]Hook already exists at {hook_path}[/yellow]")
        console.print("Use --force to overwrite, or run 'replay hook uninstall' first")
        raise typer.Exit(code=EXIT_ERROR)

    # Check if it's a replay hook (ours or someone else's)
    if hook_path.exists() and force:
        existing_content = hook_path.read_text()
        if "replay commit-index" not in existing_content:
            console.print(f"[yellow]Warning: Existing hook is not a replay hook[/yellow]")
            console.print("Use --force to overwrite anyway")

    # Create the hook content
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
) -> None:
    """Remove post-commit hook."""
    repo_path = Path(path).resolve()
    hook_path = repo_path / ".git" / "hooks" / "post-commit"

    if not hook_path.exists():
        console.print(f"[yellow]No post-commit hook found at {hook_path}[/yellow]")
        return

    # Check if it's a replay hook
    content = hook_path.read_text()
    if "replay commit-index" not in content:
        console.print(f"[yellow]Warning: Hook exists but may not be a replay hook[/yellow]")
        console.print("Use 'replay hook status' to check")

    hook_path.unlink()
    console.print(f"[green]Removed post-commit hook[/green]")


@hook_app.command()
def status(
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
        console.print(f"[bold]Hook status:[/bold] Not installed")
        console.print(f"Run 'replay hook install' to install")
        return

    console.print(f"[bold]Hook status:[/bold] Installed")
    console.print(f"Hook path: {hook_path}")

    content = hook_path.read_text()
    if "replay commit-index" in content:
        console.print("Hook type: Replay (automatic indexing)")
    else:
        console.print("Hook type: Unknown (not a replay hook)")


app.add_typer(hook_app, name="hook")


if __name__ == "__main__":
    app()
