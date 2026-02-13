"""CLI entry point for replay."""
from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console

from replay.config import get_config

app = typer.Typer(
    name="replay",
    help="Local-first semantic memory CLI for AI coding agents",
    add_completion=False,
)
console = Console()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """replay - Semantic memory for AI agents."""
    if ctx.invoked_subcommand is None:
        console.print("[bold]replay[/bold] - Semantic memory CLI")
        console.print("Use --help for usage information")


@app.command()
def init(
    path: str = typer.Option(".", help="Directory to initialize"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing"),
) -> None:
    """Initialize a replay database in the specified directory."""
    from replay.storage import ZvecStore

    init_path = Path(path).resolve()
    data_dir = init_path / ".replay"

    if data_dir.exists() and not force:
        console.print(f"[yellow]Already initialized at {data_dir}[/yellow]")
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    store = ZvecStore(data_dir / "vectors")
    store.close()

    console.print(f"[green]Initialized replay database at {data_dir}[/green]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
) -> None:
    """Search for similar documents."""
    from replay.embeddings import Embedder
    from replay.storage import ZvecStore
    from replay.config import get_config

    config = get_config()
    embedder = Embedder()
    store = ZvecStore(config.data_dir / "vectors")

    # Embed the query
    query_vector = embedder.embed_single(query)

    # Search
    results = store.search(query_vector, top_k=top_k)

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    console.print(f"[bold]Search results for:[/bold] {query}")
    for i, result in enumerate(results, 1):
        console.print(f"\n[bold]{i}.[/bold] Score: {result['score']:.3f}")
        console.print(f"   {result['text'][:200]}...")

    store.close()


@app.command()
def add(
    text: str = typer.Option(..., "--text", "-t", help="Text to add"),
    metadata: str = typer.Option("", "--metadata", "-m", help="JSON metadata"),
) -> None:
    """Add a document to the vector store."""
    import json
    from replay.embeddings import Embedder
    from replay.storage import ZvecStore
    from replay.config import get_config
    import uuid

    config = get_config()
    embedder = Embedder()
    store = ZvecStore(config.data_dir / "vectors")

    # Parse metadata
    meta = {}
    if metadata:
        try:
            meta = json.loads(metadata)
        except json.JSONDecodeError:
            console.print("[red]Invalid JSON metadata[/red]")
            return

    # Embed the text
    doc_id = meta.get("id", str(uuid.uuid4()))
    vector = embedder.embed_single(text)

    # Store
    store.insert_with_vector(doc_id, vector, text, meta)
    store.close()

    console.print(f"[green]Added document: {doc_id}[/green]")


@app.command()
def stats() -> None:
    """Show indexing statistics and storage info."""
    from replay.storage import ZvecStore
    from replay.config import get_config

    config = get_config()
    store = ZvecStore(config.data_dir / "vectors")

    count = store.count()
    console.print(f"[bold]Replay Statistics[/bold]")
    console.print(f"  Documents: {count}")
    console.print(f"  Data directory: {config.data_dir}")
    console.print(f"  Embedding model: {config.embed_model}")

    store.close()


@app.command()
def index(
    path: str = typer.Argument(".", help="Path to index"),
) -> None:
    """Index files from a directory."""
    from replay.embeddings import Embedder
    from replay.storage import ZvecStore
    from replay.config import get_config
    import uuid

    config = get_config()
    embedder = Embedder()
    store = ZvecStore(config.data_dir / "vectors")

    index_path = Path(path).resolve()
    if not index_path.exists():
        console.print(f"[red]Path does not exist: {path}[/red]")
        return

    # Find all text files
    files = list(index_path.rglob("*.py")) + list(index_path.rglob("*.md"))

    console.print(f"[green]Indexing {len(files)} files...[/green]")

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

    store.close()
    console.print(f"[green]Indexed {len(files)} files[/green]")


@app.command()
def commit_index(
    path: str = typer.Option(".", "--repo", "-r", help="Git repository path"),
    limit: int = typer.Option(100, "--limit", "-n", help="Number of commits to index"),
) -> None:
    """Index git commits from a repository."""
    from pathlib import Path
    from replay.git import GitRunner, DiffParser
    from replay.embeddings import Embedder
    from replay.storage import ZvecStore
    from replay.config import get_config
    import uuid

    config = get_config()
    repo_path = Path(path).resolve()

    if not (repo_path / ".git").exists():
        console.print(f"[red]Not a git repository: {path}[/red]")
        return

    git = GitRunner(repo_path)
    embedder = Embedder()
    store = ZvecStore(config.data_dir / "vectors")
    diff_parser = DiffParser()

    # Get commits
    commits = git.get_commits(limit=limit)

    console.print(f"[green]Indexing {len(commits)} commits...[/green]")

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

    store.close()
    console.print(f"[green]Indexed {len(commits)} commits[/green]")


if __name__ == "__main__":
    app()
