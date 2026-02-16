"""CLI entry point for Town Elder."""
from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from itertools import chain
from pathlib import Path

import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from town_elder import __version__
from town_elder.cli_services import (
    EXIT_ERROR,
    EXIT_INVALID_ARG,
    EXIT_SUCCESS,
    CLIContext,
    CLIServiceContext,
    ServiceInitError,
    _escape_rich,
    _is_empty_repo_error,
    console,
    error_console,
    get_cli_services,
    require_initialized,
)
from town_elder.config import get_config as get_config

app = typer.Typer(
    name="te",
    help="Local-first semantic memory CLI for AI coding agents",
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

# Global data directory option - DEPRECATED, use CLIContext instead
_data_dir: Path | None = None
GIT_ERROR_EXIT_CODE = 128


def _get_git_dir(repo_path: Path) -> Path:
    """Get the .git directory path, handling worktrees where .git is a file.

    In regular git repos, .git is a directory.
    In git worktrees, .git is a file that points to the actual git directory.

    When the gitdir path is relative (common in submodules and some worktree
    setups), it is resolved relative to the .git file's location (repo_path).
    """
    git_path = repo_path / ".git"
    if git_path.is_file():
        # Worktree case: .git is a file containing "gitdir: /path/to/actual/git dir"
        content = git_path.read_text()
        if content.startswith("gitdir:"):
            actual_git_dir = content.split(":", 1)[1].strip()
            # Resolve relative paths against the repo's .git file location
            git_dir_path = Path(actual_git_dir)
            if not git_dir_path.is_absolute():
                git_dir_path = (git_path.parent / git_dir_path).resolve()
            return git_dir_path
    return git_path


def _get_common_git_dir(repo_path: Path) -> Path:
    """Get the common .git directory path for the repository.

    For regular repos, this returns the .git directory.
    For worktrees, this returns the main repository's .git directory (not the
    worktree-specific one). This is the correct directory for hooks since Git
    executes hooks from the common gitdir, not the worktree-private one.

    Uses 'git rev-parse --git-common-dir' to determine the correct path.
    """
    try:
        result = subprocess.run(
            ["/usr/bin/git", "rev-parse", "--git-common-dir"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        git_common_dir = result.stdout.strip()
        git_dir_path = Path(git_common_dir)
        # Resolve relative paths against the repo_path
        if not git_dir_path.is_absolute():
            git_dir_path = (repo_path / git_dir_path).resolve()
        return git_dir_path
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError):
        # Fallback to _get_git_dir if git command fails or path is invalid
        return _get_git_dir(repo_path)


def _get_git_repo_root(repo_path: Path) -> Path | None:
    """Get the git repository root directory using git rev-parse.

    This works for subdirectories within a git repo, unlike checking for .git existence.

    Returns the resolved Path to the repo root, or None if not in a git repository.
    """
    try:
        result = subprocess.run(
            ["/usr/bin/git", "rev-parse", "--show-toplevel"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip()).resolve()
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError):
        return None


def _is_te_hook(content: str) -> bool:
    """Check if hook content is a Town Elder commit-index hook.

    Uses robust detection that handles:
    - Extra arguments like --data-dir between te and commit-index
    - python -m town_elder invocation
    - uv run te invocation
    - uvx --from town-elder te invocation
    - Absolute interpreter paths (e.g., /usr/bin/python or sys.executable)
    - te command invocation

    Note: Only matches actual command invocations, not comment text or quoted strings.
    """
    import re

    # First, filter out quoted strings to avoid false positives like:
    # echo "te commit-index" -> the string "te commit-index" should not match
    # We use a simple approach: remove content within single or double quotes
    content_without_strings = re.sub(r'"[^"]*"', '', content)  # Remove double-quoted strings
    content_without_strings = re.sub(r"'[^']*'", '', content_without_strings)  # Remove single-quoted strings

    # Filter out comment lines to avoid false positives.
    # A line is considered a comment if it starts with # (after stripping leading whitespace)
    # but we exclude shebang lines (#!/...).
    non_comment_lines = []
    for line in content_without_strings.splitlines():
        stripped = line.lstrip()
        # Skip comment lines (but not shebang lines like #!/bin/sh)
        if stripped.startswith("#") and not stripped.startswith("#!"):
            continue
        non_comment_lines.append(line)
    non_comment_content = "\n".join(non_comment_lines)

    # Match patterns for te/town_elder hooks:
    # - te commit-index
    # - uv run te commit-index
    # - uvx --from town-elder te commit-index
    # - python[3[.x]] -m town_elder commit-index
    # - /absolute/path/python[3[.x]] -m town_elder commit-index
    # - te --data-dir /path commit-index
    # - uv run te --data-dir /path commit-index
    # - uvx --from town-elder te --data-dir /path commit-index
    # - python[3[.x]] -m town_elder --data-dir /path commit-index
    patterns = [
        # te commit-index (no extra args)
        r'\bte\s+commit-index\b',
        # te [args...] commit-index (with extra args)
        r'\bte\s+\S+.*\s+commit-index\b',
        # uv run te commit-index (no extra args)
        r'\buv\s+run\s+te\s+commit-index\b',
        # uv run te [args...] commit-index (with extra args)
        r'\buv\s+run\s+te\s+\S+.*\s+commit-index\b',
        # uvx --from town-elder te commit-index (no extra args)
        r'\buvx\s+--from\s+town-elder\s+te\s+commit-index\b',
        # uvx --from town-elder te [args...] commit-index (with extra args)
        r'\buvx\s+--from\s+town-elder\s+te\s+\S+.*\s+commit-index\b',
        # python[3[.x]] -m town_elder commit-index (no extra args)
        # Matches: python -m town_elder, python3 -m town_elder, python3.11 -m town_elder
        r'\bpython3?(\.\d+)?\s+-m\s+town_elder\s+commit-index\b',
        # python[3[.x]] -m town_elder [args...] commit-index (with extra args)
        r'\bpython3?(\.\d+)?\s+-m\s+town_elder\s+\S+.*\s+commit-index\b',
        # /absolute/path/python[3[.x]] -m town_elder commit-index (no extra args)
        r'/[^\\s]+/python3?(\.\d+)?\s+-m\s+town_elder\s+commit-index\b',
        # /absolute/path/python[3[.x]] -m town_elder [args...] commit-index (with extra args)
        r'/[^\\s]+/python3?(\.\d+)?\s+-m\s+town_elder\s+\S+.*\s+commit-index\b',
    ]

    return any(re.search(pattern, non_comment_content) for pattern in patterns)


def _safe_read_hook(hook_path: Path) -> str | None:
    """Safely read hook file content, handling non-UTF8 files gracefully.

    Returns:
        The hook content as a string, or None if the file cannot be read as UTF-8
        or does not exist.
    """
    try:
        return hook_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


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


def _validate_data_dir_option(data_dir: Path, invoked_subcommand: str | None) -> None:
    """Validate explicit --data-dir before command execution."""
    if data_dir.exists():
        if data_dir.is_dir():
            return
        error_console.print(
            "[red]Error: --data-dir exists but is not a directory:[/red] "
            f"{_escape_rich(str(data_dir))}"
        )
        raise typer.Exit(code=EXIT_ERROR)

    # Allow init to create a new data directory.
    if invoked_subcommand == "init":
        return

    error_console.print(
        "[red]Error: --data-dir does not exist:[/red] "
        f"{_escape_rich(str(data_dir))}"
    )
    error_console.print(
        "[dim]Use 'te --data-dir <path> init --path <repo>' to create it.[/dim]"
    )
    raise typer.Exit(code=EXIT_ERROR)


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

    # Ensure state is a dict (handle valid JSON that is not an object, e.g., ["repos"])
    if not isinstance(state, dict):
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

    # Ensure we have a valid dict (could be corrupted or non-dict from manual edit)
    if not isinstance(state["repos"], dict):
        state = {"repos": {}}

    repo_id = _get_repo_id(repo_path)
    repo_state = state["repos"].get(repo_id, {})
    last_indexed = repo_state.get("last_indexed_commit")

    return last_indexed, state


def _save_index_state(state_file: Path, repo_path: Path, frontier_commit_hash: str) -> None:
    """Save incremental index state, scoped by repository.

    Creates or updates the repo-scoped entry in the state file.
    Uses atomic write (temp file + rename) to prevent corruption.
    """
    import os
    import tempfile

    # Load existing state (to preserve other repos' state)
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
        except Exception:
            state = {"repos": {}}
    else:
        state = {"repos": {}}

    # Ensure state is a dict (handle valid JSON that is not an object)
    if not isinstance(state, dict):
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

    # Atomic write: write to temp file, then rename
    # os.replace is atomic on POSIX systems
    state_json = json.dumps(state)
    temp_fd, temp_path = tempfile.mkstemp(
        dir=state_file.parent,
        prefix=".index_state_",
        suffix=".tmp"
    )
    try:
        with os.fdopen(temp_fd, "w") as f:
            f.write(state_json)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, state_file)
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def _is_safe_te_storage_path(data_dir: Path, init_path: Path) -> bool:
    """Validate that data_dir is a safe te-managed storage location.

    Only allows deletion of directories that are clearly te-managed storage:
    1. Default .town_elder directory in init path
    2. Paths with components starting with '.town_elder_' (custom te storage)

    Fails for arbitrary hidden directories (like .git, .config, etc.) and
    broad paths like home directory, repo root, /tmp, etc.
    """
    resolved_data_dir = data_dir.resolve()
    resolved_init_path = init_path.resolve()

    # Check if it's the default .town_elder in init path
    if resolved_data_dir == resolved_init_path / ".town_elder":
        return True

    # Check if any path component is or starts with .town_elder (custom te storage)
    for part in resolved_data_dir.parts:
        if part == ".town_elder" or part.startswith(".town_elder_"):
            return True

    return False


def _run_search(
    ctx: typer.Context,
    query: str,
    top_k: int,
) -> None:
    """Shared implementation for search-style commands."""
    # Validate top_k
    if top_k <= 0:
        console.print(f"[red]Error:[/red] --top-k must be a positive integer, got {top_k}")
        raise typer.Exit(code=EXIT_ERROR)

    with get_cli_services(ctx) as (svc, embedder, store):
        try:
            # Embed the query
            query_vector = embedder.embed_single(query)

            # Search
            results = store.search(query_vector, top_k=top_k)
        except Exception as e:
            console.print(f"[red]Error during search:[/red] {_escape_rich(str(e))}")
            raise typer.Exit(code=EXIT_ERROR)

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
    # Get config for display info
    config = require_initialized(ctx)

    with get_cli_services(ctx, include_embedder=False) as (svc, embedder, store):
        try:
            count = store.count()
        except Exception as e:
            console.print(f"[red]Error getting stats:[/red] {_escape_rich(str(e))}")
            raise typer.Exit(code=EXIT_ERROR)

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
    version: bool | None = typer.Option(
        None,
        "--version",
        help="Show version information",
        is_flag=True,
    ),
):
    """te - Semantic memory for AI agents.

    A local-first semantic memory CLI for AI coding agents.
    """
    if version:
        console.print(f"te version {__version__}")
        raise typer.Exit(code=EXIT_SUCCESS)

    resolved_data_dir: Path | None = None
    if data_dir:
        # Check for null bytes (common path injection)
        if "\0" in data_dir:
            error_console.print(
                "[red]Error: Invalid --data-dir path (contains null byte):[/red] "
                f"{_escape_rich(data_dir)}"
            )
            raise typer.Exit(code=EXIT_ERROR)

        try:
            # Expand user path to catch invalid paths early.
            resolved_data_dir = Path(data_dir).expanduser()
            resolved_data_dir.resolve()
        except (OSError, ValueError) as e:
            error_console.print(
                "[red]Error: Invalid --data-dir path:[/red] "
                f"{_escape_rich(data_dir)}"
            )
            error_console.print(f"[dim]{e}[/dim]")
            raise typer.Exit(code=EXIT_ERROR)

        _validate_data_dir_option(resolved_data_dir, ctx.invoked_subcommand)

    # Use invocation-scoped context instead of global to prevent data-dir leakage
    ctx.obj = CLIContext(data_dir=resolved_data_dir)

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
        git_dir = _get_common_git_dir(init_path)
        if not git_dir.exists():
            console.print("[yellow]Warning: Not a git repository, skipping hook installation[/yellow]")
        else:
            try:
                import os
                hooks_dir = git_dir / "hooks"
                hook_path = hooks_dir / "post-commit"

                hooks_dir.mkdir(parents=True, exist_ok=True)

                # Check for symlinks - refuse to follow for security
                if hook_path.is_symlink():
                    console.print("[yellow]Warning: Hook path is a symlink, skipping installation.[/yellow]")
                    console.print("[dim]Symlink targets may be outside the hooks directory.[/dim]")
                elif hook_path.exists():
                    console.print("[yellow]Warning: Hook already exists, skipping. Use 'te hook install --force' to overwrite[/yellow]")
                else:
                    # Use absolute path for data_dir to ensure hook works from any directory
                    # Shell-escape the data_dir to prevent injection
                    escaped_data_dir = shlex.quote(str(Path(data_dir).resolve()))
                    hook_content = f"""#!/bin/sh
# Town Elder post-commit hook - automatically indexes commits
# Try uv first, then uvx, then te, then python -m town_elder

command -v uv >/dev/null 2>&1 && uv run te --data-dir {escaped_data_dir} commit-index --repo "$(git rev-parse --show-toplevel)" && exit
command -v uvx >/dev/null 2>&1 && uvx --from town-elder te --data-dir {escaped_data_dir} commit-index --repo "$(git rev-parse --show-toplevel)" && exit
command -v te >/dev/null 2>&1 && te --data-dir {escaped_data_dir} commit-index --repo "$(git rev-parse --show-toplevel)" && exit
python -m town_elder --data-dir {escaped_data_dir} commit-index --repo "$(git rev-parse --show-toplevel)"
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


@app.command("query", hidden=True)
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
    text: str = typer.Argument(
        None,
        help="Text content to add to the vector store",
    ),
    text_option: str = typer.Option(
        None,
        "--text",
        "-t",
        help="Text content to add to the vector store (alternative to positional argument)",
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

    # Allow both positional argument and --text option
    if text is None and text_option is None:
        error_console.print("[red]Error: Either provide text as argument or use --text option[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    # Prefer positional argument, fall back to option
    text_content = text if text is not None else text_option

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

    with get_cli_services(ctx) as (svc, embedder, store):
        try:
            # Embed the text
            doc_id = meta.get("id", str(uuid.uuid4()))
            vector = embedder.embed_single(text_content)

            # Store
            store.insert(doc_id, vector, text_content, meta)
        except Exception as e:
            console.print(f"[red]Error storing document:[/red] {_escape_rich(str(e))}")
            raise typer.Exit(code=EXIT_ERROR)

    console.print(f"[green]Added document: {doc_id}[/green]")


@app.command()
def stats(ctx: typer.Context) -> None:
    """Show indexing statistics and storage info.

    Displays the number of documents and configuration details.
    """
    _run_stats(ctx)


@app.command("status", hidden=True)
def status(ctx: typer.Context) -> None:
    """Show indexing statistics and storage info.

    Alias for 'stats' command. Displays the number of documents and configuration details.
    """
    _run_stats(ctx)


@app.command()
def index(  # noqa: PLR0912
    ctx: typer.Context,
    path: str = typer.Option(
        ".",
        "--path",
        "-p",
        help="Path to directory to index (default: current directory)",
    ),
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
    # Validate path first (before services)
    index_path = Path(path).resolve()
    if not index_path.exists():
        error_console.print(f"[red]Error: Path does not exist: {path}[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    if not index_path.is_dir():
        error_console.print(f"[red]Error: Path is not a directory: {path}[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    with get_cli_services(ctx) as (svc, embedder, store):
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
        # Use chain.from_iterable to lazily combine glob results without materializing
        # full lists in memory - important for large repositories
        all_files = chain.from_iterable([index_path.rglob("*.py"), index_path.rglob("*.md")])
        # Single-pass filtering to avoid multiple iterations over the file list
        files_to_index = []
        excluded_files = []
        for f in all_files:
            if should_exclude(f):
                excluded_files.append(f)
            else:
                files_to_index.append(f)

        console.print(f"[green]Indexing {len(files_to_index)} files (attempted {len(files_to_index) + len(excluded_files)}, excluded {len(excluded_files)})...[/green]")

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

    if skipped_count > 0:
        console.print(f"[green]Indexed {indexed_count} files, skipped {skipped_count} (excluded {len(excluded_files)})[/green]")
    else:
        console.print(f"[green]Indexed {indexed_count} files (excluded {len(excluded_files)})[/green]")


# Index subcommand group (files and commits)
index_app = typer.Typer(name="index", help="Index files or commits into the vector store")
app.add_typer(index_app, name="index")


@index_app.command("commits")
def index_commits(  # noqa: PLR0912, PLR0913
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
        help="Number of commits to index (default: 100, use --all for full history)",
    ),
    all_history: bool = typer.Option(
        False,
        "--all",
        help="Index all commits in history (overrides --limit)",
    ),
    batch_size: int = typer.Option(
        500,
        "--batch-size",
        "-b",
        help="Number of commits to process per batch (default: 500)",
    ),
    max_diff_size: int = typer.Option(
        100 * 1024,
        "--max-diff-size",
        help="Maximum diff size in bytes (default: 102400)",
    ),
    mode: str = typer.Option(
        "incremental",
        "--mode",
        "-m",
        help="Indexing mode: 'incremental' (only new commits) or 'full' (all commits)",
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
    # Validate mode
    if mode not in ("incremental", "full"):
        error_console.print("[red]Error: --mode must be 'incremental' or 'full'[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)
    incremental = mode == "incremental"

    # Call the main commit_index logic
    _run_commit_index(ctx, path, limit, all_history, batch_size, max_diff_size, incremental, force)


def _run_commit_index(
    ctx: typer.Context,
    path: str,
    limit: int,
    all_history: bool,
    batch_size: int,
    max_diff_size: int,
    incremental: bool,
    force: bool,
) -> None:
    """Shared implementation for commit indexing."""
    # Validate database is initialized first
    config = require_initialized(ctx)

    # Validate repo path second
    repo_path = Path(path).resolve()

    # Use git to find the repo root - this handles subdirectories correctly
    repo_root = _get_git_repo_root(repo_path)

    # If git rev-parse failed, check for .git as a fallback (handles test mocks)
    if repo_root is None:
        if (repo_path / ".git").exists():
            # Has .git but git command failed - use path as-is
            pass
        else:
            error_console.print(f"[red]Error: Not a git repository: {path}[/red]")
            console.print("[dim]Ensure the path is inside a git repository[/dim]")
            raise typer.Exit(code=EXIT_INVALID_ARG)
    # If path is a subdirectory, use the repo root
    elif repo_root != repo_path:
        repo_path = repo_root

    if limit <= 0:
        error_console.print("[red]Error: --limit must be greater than 0[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)
    if batch_size <= 0:
        error_console.print("[red]Error: --batch-size must be greater than 0[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)
    if max_diff_size <= 0:
        error_console.print("[red]Error: --max-diff-size must be greater than 0[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    state_file = config.data_dir / "index_state.json"

    # Load last indexed commit from state file (repo-scoped)
    last_indexed = None
    if incremental and not force:
        last_indexed, _ = _load_index_state(state_file, repo_path)

    # Get services
    svc = CLIServiceContext(ctx)
    try:
        git = svc.create_git_runner(repo_path)
        embedder = svc.create_embedder()
        store = svc.create_vector_store()
        diff_parser = svc.create_diff_parser()
    except ServiceInitError as e:
        console.print(f"[red]Error initializing services:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)

    try:
        initial_limit = batch_size if all_history else limit
        # Get commits using batch mode (with files in single call)
        initial_commits = git.get_commits_with_files_batch(limit=initial_limit)
    except subprocess.CalledProcessError as e:
        # Check for empty repository (exit code 128 can mean no commits OR fatal error)
        # We must check the error message to distinguish between them
        error_msg = e.stderr.strip() if e.stderr else ""
        if e.returncode == GIT_ERROR_EXIT_CODE and _is_empty_repo_error(error_msg):
            console.print("[yellow]No commits to index.[/yellow]")
            store.close()
            raise typer.Exit(code=0)
        # Fatal error - not an empty repo but a real git error
        console.print(f"[red]Error fetching commits:[/red] {_escape_rich(error_msg or str(e))}")
        store.close()
        raise typer.Exit(code=EXIT_ERROR)
    except Exception as e:
        console.print(f"[red]Error fetching commits:[/red] {_escape_rich(str(e))}")
        store.close()
        raise typer.Exit(code=EXIT_ERROR)

    # Filter commits if using incremental mode
    commits = initial_commits
    sentinel_found = True  # Assume found unless we determine otherwise
    if incremental and last_indexed and not force:
        # Find the position of last indexed commit
        found_last = False
        filtered = []
        for commit in initial_commits:
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
            offset = len(initial_commits)
            while not found_last:
                more_commits = git.get_commits_with_files_batch(limit=batch_size, offset=offset)
                if not more_commits:
                    break
                for commit in more_commits:
                    if commit.hash == last_indexed:
                        found_last = True
                        sentinel_found = True
                        break
                    filtered.append(commit)
                offset += len(more_commits)
                if found_last:
                    break
                # Check if we've reached the end
                if len(more_commits) < batch_size:
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
    elif all_history:
        # For --all mode, get all commits (may need multiple fetches)
        offset = len(commits)
        while True:
            more_commits = git.get_commits_with_files_batch(limit=batch_size, offset=offset)
            if not more_commits:
                break
            commits.extend(more_commits)
            offset += len(more_commits)
            if len(more_commits) < batch_size:
                break

    # Reverse to index from oldest to newest
    commits = list(reversed(commits))

    if not commits:
        console.print("[green]No new commits to index[/green]")
        store.close()
        return

    console.print(f"[green]Indexing {len(commits)} commits (batch size: {batch_size})...[/green]")

    indexed_count = 0
    skipped_count = 0
    frontier_commit_hash: str | None = None
    frontier_blocked = False

    # Process commits in batches
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Indexing commits", total=len(commits))

            for batch_start in range(0, len(commits), batch_size):
                batch_end = min(batch_start + batch_size, len(commits))
                batch_commits = commits[batch_start:batch_end]

                # Batch fetch diffs for all commits in this batch
                commit_hashes = [c.hash for c in batch_commits]
                diffs = git.get_diffs_batch(commit_hashes, max_size=max_diff_size)

                # Prepare all commit data for batch embedding
                commit_data: list[tuple[object, str]] = []
                for commit in batch_commits:
                    try:
                        diff = diffs.get(commit.hash, "")
                        diff_text = diff_parser.parse_diff_to_text(diff)

                        if "[truncated" in diff:
                            diff_text += " [diff was truncated due to size]"

                        text = f"Commit: {commit.message}\n\n{diff_text}"
                        commit_data.append((commit, text))
                    except Exception as e:
                        error_console.print(f"[yellow]Skipped commit {commit.hash[:8]}: {e}[/yellow]")
                        skipped_count += 1
                        progress.advance(task)

                # Batch generate embeddings for all valid commits in this batch
                if commit_data:
                    texts = [text for _, text in commit_data]
                    embeddings = list(embedder.embed(texts))

                    # Insert embeddings into store
                    for (commit, text), vector in zip(commit_data, embeddings, strict=True):
                        doc_id = f"commit_{commit.hash}"
                        # Check if document already exists to handle duplicates gracefully
                        existing = store.get(doc_id)
                        if existing is not None:
                            # Already indexed - treat as success for state advancement
                            indexed_count += 1
                            if not frontier_blocked:
                                frontier_commit_hash = commit.hash
                        else:
                            try:
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
                        progress.advance(task)

                # Save checkpoint after each batch (resumable)
                if frontier_commit_hash and sentinel_found:
                    _save_index_state(state_file, repo_path, frontier_commit_hash)

    except Exception as e:
        console.print(f"[red]Error during indexing:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)
    finally:
        store.close()

    # Save final state
    if frontier_commit_hash and sentinel_found:
        _save_index_state(state_file, repo_path, frontier_commit_hash)

    if skipped_count > 0:
        console.print(f"[green]Indexed {indexed_count} commits, skipped {skipped_count}[/green]")
    else:
        console.print(f"[green]Indexed {indexed_count} commits[/green]")


@app.command()
def commit_index(  # noqa: PLR0912, PLR0913
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
        help="Number of commits to index (default: 100, use --all for full history)",
    ),
    all_history: bool = typer.Option(
        False,
        "--all",
        help="Index all commits in history (overrides --limit)",
    ),
    batch_size: int = typer.Option(
        500,
        "--batch-size",
        "-b",
        help="Number of commits to process per batch (default: 500)",
    ),
    max_diff_size: int = typer.Option(
        100 * 1024,
        "--max-diff-size",
        help="Maximum diff size in bytes (default: 102400)",
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
    # Validate database is initialized first
    config = require_initialized(ctx)

    # Validate repo path second
    repo_path = Path(path).resolve()

    # Use git to find the repo root - this handles subdirectories correctly
    repo_root = _get_git_repo_root(repo_path)

    # If git rev-parse failed, check for .git as a fallback (handles test mocks)
    if repo_root is None:
        if (repo_path / ".git").exists():
            # Has .git but git command failed - use path as-is
            pass
        else:
            error_console.print(f"[red]Error: Not a git repository: {path}[/red]")
            console.print("[dim]Ensure the path is inside a git repository[/dim]")
            raise typer.Exit(code=EXIT_INVALID_ARG)
    # If path is a subdirectory, use the repo root
    elif repo_root != repo_path:
        repo_path = repo_root

    if limit <= 0:
        error_console.print("[red]Error: --limit must be greater than 0[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)
    if batch_size <= 0:
        error_console.print("[red]Error: --batch-size must be greater than 0[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)
    if max_diff_size <= 0:
        error_console.print("[red]Error: --max-diff-size must be greater than 0[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    state_file = config.data_dir / "index_state.json"

    # Load last indexed commit from state file (repo-scoped)
    last_indexed = None
    if incremental and not force:
        last_indexed, _ = _load_index_state(state_file, repo_path)

    # Get services
    svc = CLIServiceContext(ctx)
    try:
        git = svc.create_git_runner(repo_path)
        embedder = svc.create_embedder()
        store = svc.create_vector_store()
        diff_parser = svc.create_diff_parser()
    except ServiceInitError as e:
        console.print(f"[red]Error initializing services:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)

    try:
        initial_limit = batch_size if all_history else limit
        # Get commits using batch mode (with files in single call)
        initial_commits = git.get_commits_with_files_batch(limit=initial_limit)
    except subprocess.CalledProcessError as e:
        # Check for empty repository (exit code 128 can mean no commits OR fatal error)
        # We must check the error message to distinguish between them
        error_msg = e.stderr.strip() if e.stderr else ""
        if e.returncode == GIT_ERROR_EXIT_CODE and _is_empty_repo_error(error_msg):
            console.print("[yellow]No commits to index.[/yellow]")
            store.close()
            raise typer.Exit(code=0)
        # Fatal error - not an empty repo but a real git error
        console.print(f"[red]Error fetching commits:[/red] {_escape_rich(error_msg or str(e))}")
        store.close()
        raise typer.Exit(code=EXIT_ERROR)
    except Exception as e:
        console.print(f"[red]Error fetching commits:[/red] {_escape_rich(str(e))}")
        store.close()
        raise typer.Exit(code=EXIT_ERROR)

    # Filter commits if using incremental mode
    commits = initial_commits
    sentinel_found = True  # Assume found unless we determine otherwise
    if incremental and last_indexed and not force:
        # Find the position of last indexed commit
        found_last = False
        filtered = []
        for commit in initial_commits:
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
            offset = len(initial_commits)
            while not found_last:
                more_commits = git.get_commits_with_files_batch(limit=batch_size, offset=offset)
                if not more_commits:
                    break
                for commit in more_commits:
                    if commit.hash == last_indexed:
                        found_last = True
                        sentinel_found = True
                        break
                    filtered.append(commit)
                offset += len(more_commits)
                if found_last:
                    break
                # Check if we've reached the end
                if len(more_commits) < batch_size:
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
    elif all_history:
        # For --all mode, get all commits (may need multiple fetches)
        offset = len(commits)
        while True:
            more_commits = git.get_commits_with_files_batch(limit=batch_size, offset=offset)
            if not more_commits:
                break
            commits.extend(more_commits)
            offset += len(more_commits)
            if len(more_commits) < batch_size:
                break

    # Reverse to index from oldest to newest
    commits = list(reversed(commits))

    if not commits:
        console.print("[green]No new commits to index[/green]")
        store.close()
        return

    console.print(f"[green]Indexing {len(commits)} commits (batch size: {batch_size})...[/green]")

    indexed_count = 0
    skipped_count = 0
    frontier_commit_hash: str | None = None
    frontier_blocked = False

    # Process commits in batches
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Indexing commits", total=len(commits))

            for batch_start in range(0, len(commits), batch_size):
                batch_end = min(batch_start + batch_size, len(commits))
                batch_commits = commits[batch_start:batch_end]

                # Batch fetch diffs for all commits in this batch
                commit_hashes = [c.hash for c in batch_commits]
                diffs = git.get_diffs_batch(commit_hashes, max_size=max_diff_size)

                # Prepare all commit data for batch embedding
                commit_data: list[tuple[object, str]] = []
                for commit in batch_commits:
                    try:
                        diff = diffs.get(commit.hash, "")
                        diff_text = diff_parser.parse_diff_to_text(diff)

                        if "[truncated" in diff:
                            diff_text += " [diff was truncated due to size]"

                        text = f"Commit: {commit.message}\n\n{diff_text}"
                        commit_data.append((commit, text))
                    except Exception as e:
                        error_console.print(f"[yellow]Skipped commit {commit.hash[:8]}: {e}[/yellow]")
                        skipped_count += 1
                        progress.advance(task)

                # Batch generate embeddings for all valid commits in this batch
                if commit_data:
                    texts = [text for _, text in commit_data]
                    embeddings = list(embedder.embed(texts))

                    # Insert embeddings into store
                    for (commit, text), vector in zip(commit_data, embeddings, strict=True):
                        doc_id = f"commit_{commit.hash}"
                        # Check if document already exists to handle duplicates gracefully
                        existing = store.get(doc_id)
                        if existing is not None:
                            # Already indexed - treat as success for state advancement
                            indexed_count += 1
                            if not frontier_blocked:
                                frontier_commit_hash = commit.hash
                        else:
                            try:
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
                        progress.advance(task)

                # Save checkpoint after each batch (resumable)
                if frontier_commit_hash and sentinel_found:
                    _save_index_state(state_file, repo_path, frontier_commit_hash)

    except Exception as e:
        console.print(f"[red]Error during indexing:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)
    finally:
        store.close()

    # Save final state
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
    git_dir = _get_common_git_dir(repo_path)
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

    # Check if it's a symlink - refuse to follow symlinks for security
    if hook_path.is_symlink():
        error_console.print(f"[red]Error: Refusing to install hook over symlink: {hook_path}[/red]")
        console.print("[yellow]Symlink targets may be outside the hooks directory.[/yellow]")
        console.print("Remove the symlink first, then install the hook.")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    # Check if it's a Town Elder hook (ours or someone else's)
    if hook_path.exists() and force:
        existing_content = _safe_read_hook(hook_path)
        if existing_content is None or not _is_te_hook(existing_content):
            error_console.print("[yellow]Warning: Existing hook is not a Town Elder hook[/yellow]")
            console.print("Use --force to overwrite anyway")

    # Get the data directory from context (invocation-scoped)
    from town_elder.exceptions import ConfigError

    data_dir = _get_data_dir_from_context(ctx)
    try:
        config = get_config(data_dir=data_dir, repo_path=repo_path)
    except ConfigError as e:
        error_console.print("[red]Error: Database not initialized[/red]")
        console.print(f"[dim]{e}[/dim]")
        raise typer.Exit(code=EXIT_ERROR)

    # Determine the data_dir to use in the hook - use resolved config.data_dir
    # Use a fallback chain: uv run te -> uvx --from town-elder te -> te -> python -m town_elder
    # This ensures the hook works even if uv is not installed
    # Shell-escape the data_dir to prevent injection
    # Always include --data-dir to support env var configuration (TOWN_ELDER_DATA_DIR)
    escaped_data_dir = shlex.quote(str(config.data_dir.resolve()))
    data_dir_arg = f'--data-dir {escaped_data_dir}'
    hook_content = f"""#!/bin/sh
# Town Elder post-commit hook - automatically indexes commits
# Try uv first, then uvx, then te, then python -m town_elder

command -v uv >/dev/null 2>&1 && uv run te {data_dir_arg} commit-index --repo "$(git rev-parse --show-toplevel)" && exit
command -v uvx >/dev/null 2>&1 && uvx --from town-elder te {data_dir_arg} commit-index --repo "$(git rev-parse --show-toplevel)" && exit
command -v te >/dev/null 2>&1 && te {data_dir_arg} commit-index --repo "$(git rev-parse --show-toplevel)" && exit
python -m town_elder {data_dir_arg} commit-index --repo "$(git rev-parse --show-toplevel)"
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
    git_dir = _get_common_git_dir(repo_path)
    hook_path = git_dir / "hooks" / "post-commit"

    # Check if it's a git repository
    if not git_dir.exists():
        console.print(f"[red]Not a git repository: {path}[/red]")
        raise typer.Exit(code=EXIT_ERROR)

    if not hook_path.exists():
        error_console.print(f"[yellow]No post-commit hook found at {hook_path}[/yellow]")
        return

    # Check if it's a Town Elder hook
    content = _safe_read_hook(hook_path)
    is_te_hook = content is not None and _is_te_hook(content)

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
    git_dir = _get_common_git_dir(repo_path)

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

    content = _safe_read_hook(hook_path)
    if content is not None and _is_te_hook(content):
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

    with get_cli_services(ctx, include_embedder=False) as (svc, embedder, store):
        try:
            documents = store.get_all(include_vectors=include_vectors)
        except Exception as e:
            console.print(f"[red]Error exporting data:[/red] {_escape_rich(str(e))}")
            raise typer.Exit(code=EXIT_ERROR)

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
