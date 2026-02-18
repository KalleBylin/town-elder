"""CLI entry point for Town Elder."""

from __future__ import annotations

import contextlib
import hashlib
import json
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
from town_elder.indexing.batch_manager import (
    BatchManager,
    ChunkBatchItem,
    ChunkBatchResult,
)
from town_elder.indexing.file_scanner import scan_files
from town_elder.indexing.git_hash_scanner import (
    TrackedFile,
    get_git_root,
    scan_git_blobs,
)
from town_elder.indexing.pipeline import (
    build_file_work_items,
    parse_files_pipeline,
)
from town_elder.rust_adapter import assemble_commit_text

app = typer.Typer(
    name="te",
    help="Local-first semantic memory CLI for AI coding agents",
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

index_app = typer.Typer(
    help="Index project files and commit history",
    no_args_is_help=True,
    invoke_without_command=True,
)


@index_app.callback(context_settings={"help_option_names": ["-h", "--help"]})
def index_callback(
    ctx: typer.Context,
    all: bool = typer.Option(
        False,
        "--all",
        help="Index all project files in current directory (full repository file indexing)",
    ),
) -> None:
    """Index subcommand: use 'te index --all' for full file indexing, or 'te index files' / 'te index commits' for specific indexing."""
    if all and ctx.invoked_subcommand is not None:
        error_console.print(
            f"[red]Error:[/red] --all cannot be combined with 'te index {ctx.invoked_subcommand}'"
        )
        error_console.print(
            "[dim]Use either 'te index --all' or a specific subcommand.[/dim]"
        )
        raise typer.Exit(code=EXIT_INVALID_ARG)

    if all:
        # Route to index_files with default path "."
        # Must pass None for exclude since that's the default in index_files
        ctx.invoke(index_files, ctx=ctx, path=".", exclude=None)


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


def _find_git_repo_root(path: Path) -> Path | None:
    """Find the nearest parent directory containing .git."""
    if not path.exists() or not path.is_dir():
        return None
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _get_common_git_dir(repo_path: Path) -> Path:
    """Get the common .git directory path for the repository.

    For regular repos, this returns the .git directory.
    For worktrees, this returns the main repository's .git directory (not the
    worktree-specific one). This is the correct directory for hooks since Git
    executes hooks from the common gitdir, not the worktree-private one.

    For worktrees, the worktree gitdir contains a "commondir" file pointing to
    the shared .git directory.
    """
    repo_root = _find_git_repo_root(repo_path)
    if repo_root is None:
        # Keep existing fallback behavior for non-repo paths.
        return _get_git_dir(repo_path)

    git_dir = _get_git_dir(repo_root)
    commondir_file = git_dir / "commondir"
    if commondir_file.is_file():
        try:
            common_git_dir = Path(commondir_file.read_text().strip())
            if not common_git_dir.is_absolute():
                common_git_dir = (git_dir / common_git_dir).resolve()
            return common_git_dir
        except OSError:
            # If commondir is unreadable, use best-effort git_dir fallback.
            pass
    return git_dir


def _get_git_repo_root(repo_path: Path) -> Path | None:
    """Get the nearest git repository root for the given directory path."""
    root = _find_git_repo_root(repo_path)
    return root.resolve() if root is not None else None


def _is_te_hook(content: str) -> bool:
    """Check if hook content is a Town Elder index commits hook.

    Uses robust detection that handles:
    - Extra arguments like --data-dir between te and index commits
    - python -m town_elder invocation
    - uv run te invocation
    - uvx --from town-elder te invocation
    - Absolute interpreter paths (e.g., /usr/bin/python or sys.executable)
    - te command invocation

    Note: Only matches actual command invocations, not comment text or quoted strings.
    """
    import re

    # First, filter out quoted strings to avoid false positives like:
    # echo "te index commits" -> the string "te index commits" should not match
    # We use a simple approach: remove content within single or double quotes
    content_without_strings = re.sub(
        r'"[^"]*"', "", content
    )  # Remove double-quoted strings
    content_without_strings = re.sub(
        r"'[^']*'", "", content_without_strings
    )  # Remove single-quoted strings

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
    # - te index commits
    # - uv run te index commits
    # - uvx --from town-elder te index commits
    # - python[3[.x]] -m town_elder index commits
    # - /absolute/path/python[3[.x]] -m town_elder index commits
    # - te --data-dir /path index commits
    # - uv run te --data-dir /path index commits
    # - uvx --from town-elder te --data-dir /path index commits
    # - python[3[.x]] -m town_elder --data-dir /path index commits
    patterns = [
        # te index commits (no extra args)
        r"\bte\s+index\s+commits\b",
        # te [args...] index commits (with extra args)
        r"\bte\s+\S+.*\s+index\s+commits\b",
        # uv run te index commits (no extra args)
        r"\buv\s+run\s+te\s+index\s+commits\b",
        # uv run te [args...] index commits (with extra args)
        r"\buv\s+run\s+te\s+\S+.*\s+index\s+commits\b",
        # uvx --from town-elder te index commits (no extra args)
        r"\buvx\s+--from\s+town-elder\s+te\s+index\s+commits\b",
        # uvx --from town-elder te [args...] index commits (with extra args)
        r"\buvx\s+--from\s+town-elder\s+te\s+\S+.*\s+index\s+commits\b",
        # python[3[.x]] -m town_elder index commits (no extra args)
        # Matches: python -m town_elder, python3 -m town_elder, python3.11 -m town_elder
        r"\bpython3?(\.\d+)?\s+-m\s+town_elder\s+index\s+commits\b",
        # python[3[.x]] -m town_elder [args...] index commits (with extra args)
        r"\bpython3?(\.\d+)?\s+-m\s+town_elder\s+\S+.*\s+index\s+commits\b",
        # /absolute/path/python[3[.x]] -m town_elder index commits (no extra args)
        r"/[^\\s]+/python3?(\.\d+)?\s+-m\s+town_elder\s+index\s+commits\b",
        # /absolute/path/python[3[.x]] -m town_elder [args...] index commits (with extra args)
        r"/[^\\s]+/python3?(\.\d+)?\s+-m\s+town_elder\s+\S+.*\s+index\s+commits\b",
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


def _build_post_commit_hook_content(data_dir_arg: str) -> str:
    """Build a post-commit hook that indexes commit history."""
    return f"""#!/bin/sh
# Town Elder post-commit hook - automatically indexes commits
# Try uv first, then uvx, then te, then python -m town_elder

command -v uv >/dev/null 2>&1 && uv run te {data_dir_arg} index commits --repo "$(git rev-parse --show-toplevel)" && exit
command -v uvx >/dev/null 2>&1 && uvx --from town-elder te {data_dir_arg} index commits --repo "$(git rev-parse --show-toplevel)" && exit
command -v te >/dev/null 2>&1 && te {data_dir_arg} index commits --repo "$(git rev-parse --show-toplevel)" && exit
python -m town_elder {data_dir_arg} index commits --repo "$(git rev-parse --show-toplevel)"
"""


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
        f"[red]Error: --data-dir does not exist:[/red] {_escape_rich(str(data_dir))}"
    )
    error_console.print(
        "[dim]Use 'te --data-dir <path> init --path <repo>' to create it,[/dim]"
    )
    error_console.print(
        "[dim]or run 'te init' to use the default .town_elder in current directory.[/dim]"
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


def _save_index_state(
    state_file: Path, repo_path: Path, frontier_commit_hash: str
) -> None:
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
        dir=state_file.parent, prefix=".index_state_", suffix=".tmp"
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


# File hash state constants
FILE_HASH_STATE_FILENAME = "file_index_state.json"


def _get_file_state_path(data_dir: Path) -> Path:
    """Get the path to the file hash state file."""
    return data_dir / FILE_HASH_STATE_FILENAME


def _load_file_state(
    state_file: Path, repo_path: Path
) -> dict[str, dict[str, Any]]:
    """Load file hash state for incremental file indexing.

    Returns:
        Dictionary keyed by repo_id with file indexing state.
    """
    if not state_file.exists():
        return {}

    try:
        state = json.loads(state_file.read_text())
    except Exception:
        return {}

    if not isinstance(state, dict):
        return {}

    return state


def _save_file_state(
    state_file: Path,
    repo_path: Path,
    file_hashes: dict[str, str],
    file_chunks: dict[str, int] | None = None,
) -> None:
    """Save file hash state for incremental file indexing.

    Uses atomic write (temp file + rename) to prevent corruption.

    Args:
        state_file: Path to the state file.
        repo_path: Path to the repository (used for repo_id).
        file_hashes: Dictionary mapping relative file paths to blob hashes.
        file_chunks: Optional chunk counts keyed by relative file path.
    """
    import os
    import tempfile
    from datetime import datetime, timezone

    repo_id = _get_repo_id(repo_path)

    # Load existing state
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
        except Exception:
            state = {}
    else:
        state = {}

    if not isinstance(state, dict):
        state = {}

    # Update state for this repo
    state[repo_id] = {
        "repo_path": str(repo_path.resolve()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "file_hashes": file_hashes,
        "file_chunks": file_chunks or {},
    }

    # Atomic write
    state_json = json.dumps(state, indent=2)
    temp_fd, temp_path = tempfile.mkstemp(
        dir=state_file.parent, prefix=".file_state_", suffix=".tmp"
    )
    try:
        with os.fdopen(temp_fd, "w") as f:
            f.write(state_json)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, state_file)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def _extract_blob_hash(blob_entry: TrackedFile | str | None) -> str | None:
    """Extract blob hash from scanner output.

    Supports both current `TrackedFile` values and legacy/mock string values.
    """
    if blob_entry is None:
        return None
    if isinstance(blob_entry, str):
        return blob_entry
    return blob_entry.blob_hash


def _extract_file_chunks(repo_state: dict[str, Any]) -> dict[str, int]:
    """Extract normalized chunk-count state for a repository."""

    raw_chunks = repo_state.get("file_chunks")
    if not isinstance(raw_chunks, dict):
        return {}

    normalized: dict[str, int] = {}
    for path, chunk_count in raw_chunks.items():
        if not isinstance(path, str):
            continue
        if isinstance(chunk_count, bool):
            continue
        if isinstance(chunk_count, int) and chunk_count > 0:
            normalized[path] = chunk_count
    return normalized


def _build_file_doc_id(path_value: str, chunk_index: int = 0) -> str:
    """Build deterministic doc ID for file content chunks."""

    doc_id_input = path_value if chunk_index == 0 else f"{path_value}#chunk:{chunk_index}"
    return hashlib.sha256(doc_id_input.encode()).hexdigest()[:16]


def _get_doc_id_inputs(path_value: str, repo_root: Path) -> set[str]:
    """Return canonical and legacy ID input strings for a path."""

    doc_id_inputs = {path_value}
    path_obj = Path(path_value)
    if not path_obj.is_absolute():
        doc_id_inputs.add(str((repo_root / path_obj).resolve()))
    return doc_id_inputs


def _delete_file_docs(
    store: Any,
    path_value: str,
    repo_root: Path,
    *,
    start_chunk: int = 0,
    chunk_count: int = 1,
) -> None:
    """Delete indexed docs for a file across a chunk index range."""

    effective_chunk_count = max(chunk_count, 1)
    for delete_input in _get_doc_id_inputs(path_value, repo_root):
        for chunk_index in range(start_chunk, effective_chunk_count):
            delete_doc_id = _build_file_doc_id(delete_input, chunk_index)
            with contextlib.suppress(Exception):
                # File may not exist in store.
                store.delete(delete_doc_id)


@dataclass
class _FileIndexBatchState:
    """Tracks file-level status while chunk writes are batched."""

    file_path: Path
    relative_path: str
    blob_hash: str | None
    previous_hash: str | None
    previous_chunk_count: int
    expected_chunks: int = 0
    indexed_chunks: int = 0
    errors: list[str] = field(default_factory=list)


_FILE_INDEX_STAGE_SCANNING = "scanning"
_FILE_INDEX_STAGE_PARSING = "parsing"
_FILE_INDEX_STAGE_EMBEDDING = "embedding"
_FILE_INDEX_STAGE_STORING = "storing"


class _FileIndexProgressReporter:
    """Render file-index stage progress for TTY and non-TTY environments."""

    _NON_INTERACTIVE_STEP_TARGET = 10
    _STAGE_LABELS = {
        _FILE_INDEX_STAGE_SCANNING: "Scanning",
        _FILE_INDEX_STAGE_PARSING: "Parsing",
        _FILE_INDEX_STAGE_EMBEDDING: "Embedding",
        _FILE_INDEX_STAGE_STORING: "Storing",
    }

    def __init__(self, *, output_console: Any) -> None:
        self._console = output_console
        self._interactive = (
            output_console.is_terminal
            and not output_console.is_dumb_terminal
            and output_console.is_interactive
        )
        self._progress: Progress | None = None
        self._task_ids: dict[str, int] = {}
        self._totals: dict[str, int | None] = dict.fromkeys(self._STAGE_LABELS, None)
        self._completed: dict[str, int] = dict.fromkeys(self._STAGE_LABELS, 0)
        self._last_emitted: dict[str, str | None] = dict.fromkeys(
            self._STAGE_LABELS,
            None,
        )

    def __enter__(self) -> _FileIndexProgressReporter:
        if self._interactive:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self._console,
            )
            self._progress.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._progress is not None:
            self._progress.stop()

    def begin_stage(self, stage: str, *, total: int | None) -> None:
        self._totals[stage] = total
        self._completed[stage] = 0
        self._sync(stage, force_emit=True)

    def set_total(self, stage: str, *, total: int | None) -> None:
        self._totals[stage] = total
        completed = self._completed[stage]
        if total is not None and completed > total:
            self._completed[stage] = total
        self._sync(stage, force_emit=True)

    def advance(self, stage: str, amount: int = 1) -> None:
        if amount <= 0:
            return

        self._completed[stage] += amount
        total = self._totals[stage]
        if total is not None and self._completed[stage] > total:
            self._completed[stage] = total
        self._sync(stage)

    def complete(self, stage: str) -> None:
        total = self._totals[stage]
        if total is not None:
            self._completed[stage] = total
        self._sync(stage, force_emit=True)

    def _sync(self, stage: str, *, force_emit: bool = False) -> None:
        if self._interactive and self._progress is not None:
            self._sync_interactive(stage)
            return

        self._sync_non_interactive(stage, force_emit=force_emit)

    def _sync_interactive(self, stage: str) -> None:
        if self._progress is None:
            return

        task_id = self._task_ids.get(stage)
        if task_id is None:
            task_id = self._progress.add_task(
                self._status_line(stage),
                total=self._totals[stage],
            )
            self._task_ids[stage] = task_id
        else:
            self._progress.update(
                task_id,
                completed=self._completed[stage],
                total=self._totals[stage],
                description=self._status_line(stage),
            )

    def _sync_non_interactive(self, stage: str, *, force_emit: bool) -> None:
        status = self._status_line(stage)
        if not force_emit and not self._should_emit_non_interactive(stage):
            return
        if status == self._last_emitted[stage]:
            return

        self._console.print(status)
        self._last_emitted[stage] = status

    def _should_emit_non_interactive(self, stage: str) -> bool:
        total = self._totals[stage]
        completed = self._completed[stage]

        if total is None:
            return False

        if completed in {0, total}:
            return True

        if total <= 0:
            return False

        step = max(total // self._NON_INTERACTIVE_STEP_TARGET, 1)
        return completed % step == 0

    def _status_line(self, stage: str) -> str:
        label = self._STAGE_LABELS[stage]
        completed = self._completed[stage]
        total = self._totals[stage]
        total_text = "?" if total is None else str(total)
        return f"{label}: {completed}/{total_text}"


def _normalize_chunk_metadata(
    *,
    base_metadata: dict[str, Any],
    chunk_metadata: dict[str, Any],
    fallback_chunk_index: int,
) -> tuple[dict[str, Any], int]:
    metadata = dict(base_metadata)
    metadata.update(chunk_metadata)

    chunk_index_value = metadata.get("chunk_index")
    if (
        isinstance(chunk_index_value, bool)
        or not isinstance(chunk_index_value, int)
        or chunk_index_value < 0
    ):
        chunk_index = fallback_chunk_index
        metadata["chunk_index"] = chunk_index
    else:
        chunk_index = chunk_index_value

    return metadata, chunk_index


def _upsert_single_chunk(
    store: Any,
    *,
    doc_id: str,
    vector: Any,
    text: str,
    metadata: dict[str, Any],
) -> None:
    if hasattr(store, "upsert"):
        store.upsert(doc_id, vector, text, metadata)
        return
    if hasattr(store, "bulk_upsert"):
        store.bulk_upsert([(doc_id, vector, text, metadata)])
        return
    raise AttributeError("Store does not support upsert or bulk_upsert")


def _bulk_upsert_chunks(
    store: Any,
    docs: list[tuple[str, Any, str, dict[str, Any]]],
) -> None:
    if hasattr(store, "bulk_upsert"):
        store.bulk_upsert(docs)
        return
    for doc_id, vector, text, metadata in docs:
        _upsert_single_chunk(
            store,
            doc_id=doc_id,
            vector=vector,
            text=text,
            metadata=metadata,
        )


def _fallback_index_chunks_individually(
    batch: list[ChunkBatchItem],
    *,
    embedder: Any,
    store: Any,
    batch_error: Exception,
    on_stage_progress: Callable[[str, int], None] | None = None,
) -> list[ChunkBatchResult]:
    """Fallback path that isolates per-chunk failures after batch failure."""

    results: list[ChunkBatchResult] = []
    for item in batch:
        try:
            vector = embedder.embed_single(item.text)
        except Exception as exc:
            if on_stage_progress is not None:
                on_stage_progress(_FILE_INDEX_STAGE_EMBEDDING, 1)
                on_stage_progress(_FILE_INDEX_STAGE_STORING, 1)
            error = (
                f"chunk {item.chunk_index} failed after batch fallback: {exc} "
                f"(batch error: {batch_error})"
            )
            results.append(ChunkBatchResult(item=item, error=error))
            continue

        if on_stage_progress is not None:
            on_stage_progress(_FILE_INDEX_STAGE_EMBEDDING, 1)
        try:
            _upsert_single_chunk(
                store,
                doc_id=item.doc_id,
                vector=vector,
                text=item.text,
                metadata=item.metadata,
            )
            results.append(ChunkBatchResult(item=item))
        except Exception as exc:
            error = (
                f"chunk {item.chunk_index} failed after batch fallback: {exc} "
                f"(batch error: {batch_error})"
            )
            results.append(ChunkBatchResult(item=item, error=error))
        finally:
            if on_stage_progress is not None:
                on_stage_progress(_FILE_INDEX_STAGE_STORING, 1)
    return results


def _fallback_store_chunks_individually(
    batch: list[ChunkBatchItem],
    embeddings: list[Any],
    *,
    store: Any,
    batch_error: Exception,
    on_stage_progress: Callable[[str, int], None] | None = None,
) -> list[ChunkBatchResult]:
    """Fallback path for bulk storage failures."""

    results: list[ChunkBatchResult] = []
    for item, vector in zip(batch, embeddings, strict=True):
        try:
            _upsert_single_chunk(
                store,
                doc_id=item.doc_id,
                vector=vector,
                text=item.text,
                metadata=item.metadata,
            )
            results.append(ChunkBatchResult(item=item))
        except Exception as exc:
            error = (
                f"chunk {item.chunk_index} failed after store fallback: {exc} "
                f"(batch error: {batch_error})"
            )
            results.append(ChunkBatchResult(item=item, error=error))
        finally:
            if on_stage_progress is not None:
                on_stage_progress(_FILE_INDEX_STAGE_STORING, 1)
    return results


def _flush_file_chunk_batch(
    batch: list[ChunkBatchItem],
    *,
    embedder: Any,
    store: Any,
    on_stage_progress: Callable[[str, int], None] | None = None,
) -> list[ChunkBatchResult]:
    texts = [item.text for item in batch]
    try:
        embeddings = list(embedder.embed(texts))
    except Exception as exc:
        error_console.print(
            f"[yellow]Batch embed failed for {len(batch)} chunks; falling back to per-chunk mode: {exc}[/yellow]"
        )
        return _fallback_index_chunks_individually(
            batch,
            embedder=embedder,
            store=store,
            batch_error=exc,
            on_stage_progress=on_stage_progress,
        )

    if len(embeddings) != len(batch):
        length_error = RuntimeError(
            f"embedding count mismatch: expected {len(batch)}, got {len(embeddings)}"
        )
        return _fallback_index_chunks_individually(
            batch,
            embedder=embedder,
            store=store,
            batch_error=length_error,
            on_stage_progress=on_stage_progress,
        )
    if on_stage_progress is not None:
        on_stage_progress(_FILE_INDEX_STAGE_EMBEDDING, len(batch))

    bulk_docs = [
        (item.doc_id, vector, item.text, item.metadata)
        for item, vector in zip(batch, embeddings, strict=True)
    ]
    try:
        _bulk_upsert_chunks(store, bulk_docs)
        if on_stage_progress is not None:
            on_stage_progress(_FILE_INDEX_STAGE_STORING, len(batch))
        return [ChunkBatchResult(item=item) for item in batch]
    except Exception as exc:
        error_console.print(
            f"[yellow]Batch store failed for {len(batch)} chunks; falling back to per-chunk mode: {exc}[/yellow]"
        )
        return _fallback_store_chunks_individually(
            batch,
            embeddings,
            store=store,
            batch_error=exc,
            on_stage_progress=on_stage_progress,
        )


def _apply_chunk_batch_results(
    results: list[ChunkBatchResult],
    file_states: dict[str, _FileIndexBatchState],
) -> None:
    for result in results:
        state = file_states.get(result.item.relative_path)
        if state is None:
            continue
        if result.success:
            state.indexed_chunks += 1
        else:
            state.errors.append(result.error or "unknown indexing error")


def _preserve_previous_file_state(
    *,
    incremental: bool,
    state: _FileIndexBatchState,
    next_file_hashes: dict[str, str],
    next_file_chunks: dict[str, int],
) -> None:
    if incremental and state.blob_hash and state.previous_hash is not None:
        next_file_hashes[state.relative_path] = state.previous_hash
        next_file_chunks[state.relative_path] = state.previous_chunk_count


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
        console.print(
            f"[red]Error:[/red] --top-k must be a positive integer, got {top_k}"
        )
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
        help="Data directory. Default: .town_elder in current directory. "
        "Set TOWN_ELDER_DATA_DIR env var to avoid passing this option. "
        "Use a custom path for isolated sessions or cross-repo operations.",
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
                f"[red]Error: Invalid --data-dir path:[/red] {_escape_rich(data_dir)}"
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
            error_console.print(
                f"[red]Error: Refusing to delete unsafe data-dir path:[/red] {_escape_rich(str(data_dir))}"
            )
            console.print(
                "[yellow]For safety, --force only allows deletion of te-managed storage:[/yellow]"
            )
            console.print("  - Default .town_elder directory in the init path")
            console.print("  - Paths explicitly containing '.town_elder' in their name")
            console.print("[dim]Use a safe path or initialize without --force[/dim]")
            raise typer.Exit(code=EXIT_INVALID_ARG)
        import shutil

        try:
            shutil.rmtree(data_dir)
        except PermissionError as e:
            error_console.print(
                f"[red]Error: Cannot remove existing directory:[/red] {_escape_rich(str(e))}"
            )
            raise typer.Exit(code=EXIT_ERROR)

    try:
        data_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        error_console.print(
            f"[red]Error: Cannot create directory:[/red] {_escape_rich(str(e))}"
        )
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
            console.print(
                "[yellow]Warning: Not a git repository, skipping hook installation[/yellow]"
            )
        else:
            try:
                import os

                hooks_dir = git_dir / "hooks"
                hook_path = hooks_dir / "post-commit"

                hooks_dir.mkdir(parents=True, exist_ok=True)

                # Check for symlinks - refuse to follow for security
                if hook_path.is_symlink():
                    console.print(
                        "[yellow]Warning: Hook path is a symlink, skipping installation.[/yellow]"
                    )
                    console.print(
                        "[dim]Symlink targets may be outside the hooks directory.[/dim]"
                    )
                elif hook_path.exists():
                    console.print(
                        "[yellow]Warning: Hook already exists, skipping. Use 'te hook install --force' to overwrite[/yellow]"
                    )
                else:
                    # Use absolute path for data_dir to ensure hook works from any directory
                    # Shell-escape the data_dir to prevent injection
                    escaped_data_dir = shlex.quote(str(Path(data_dir).resolve()))
                    data_dir_arg = f"--data-dir {escaped_data_dir}"
                    hook_content = _build_post_commit_hook_content(data_dir_arg)
                    hook_path.write_text(hook_content)
                    os.chmod(hook_path, 0o755)
                    console.print(
                        f"[green]Installed post-commit hook at {hook_path}[/green]"
                    )
            except Exception as e:
                console.print(f"[yellow]Warning: Could not install hook: {e}[/yellow]")

    console.print("[dim]You can now use 'te add' to add documents[/dim]")
    if not install_hook:
        console.print(
            "[dim]Run 'te init --install-hook' to enable automatic commit indexing[/dim]"
        )


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
    """Search indexed content with semantic similarity."""
    _run_search(ctx, query=query, top_k=top_k)


def _read_add_stdin_text() -> str:
    """Read text content from stdin for `te add`."""
    import sys

    text_content = sys.stdin.read()
    if not text_content:
        error_console.print("[red]Error: No input provided via stdin[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)
    return text_content


def _resolve_add_text_argument(text: str) -> str:
    """Resolve positional add input as stdin marker, file path, or literal text."""
    if text == "-":
        return _read_add_stdin_text()

    text_path = Path(text)
    if not (text_path.exists() and text_path.is_file()):
        return text

    try:
        return text_path.read_text(encoding="utf-8")
    except Exception as e:
        error_console.print(f"[red]Error reading file:[/red] {_escape_rich(str(e))}")
        raise typer.Exit(code=EXIT_ERROR)


def _get_add_text_content(text: str | None, text_option: str | None) -> str:
    """Get add content from --text, positional argument, or stdin."""
    if text_option is not None:
        return text_option

    if text is not None:
        return _resolve_add_text_argument(text)

    import sys

    if not sys.stdin.isatty():
        return _read_add_stdin_text()

    error_console.print(
        "[red]Error: Provide text as argument, file path, '-' for stdin, "
        "or use --text option[/red]"
    )
    raise typer.Exit(code=EXIT_INVALID_ARG)


def _parse_add_metadata(metadata: str) -> dict:
    """Parse and validate add metadata option."""
    if not metadata:
        return {}

    try:
        parsed_metadata = json.loads(metadata)
    except json.JSONDecodeError as e:
        error_console.print("[red]Error: Invalid JSON metadata[/red]")
        error_console.print(f"[dim]JSON parse error at position {e.pos}: {e.msg}[/dim]")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    if not isinstance(parsed_metadata, dict):
        error_console.print(
            f"[red]Error: Metadata must be a JSON object (dict), not {type(parsed_metadata).__name__}[/red]"
        )
        raise typer.Exit(code=EXIT_INVALID_ARG)

    return parsed_metadata


@app.command()
def add(
    ctx: typer.Context,
    text: str = typer.Argument(
        None,
        help="Text content, file path, or '-' for stdin",
    ),
    text_option: str = typer.Option(
        None,
        "--text",
        "-t",
        help="Text content to add (alternative to positional argument)",
    ),
    metadata: str = typer.Option(
        "",
        "--metadata",
        "-m",
        help="JSON metadata string (must be valid JSON)",
    ),
) -> None:
    """Add one ad-hoc document to the vector store.

    Supports multiple input modes:
    - Direct text: te add "some text"
    - File input: te add path/to/file.txt
    - Stdin input: echo "text" | te add - or te add -
    """
    import uuid

    text_content = _get_add_text_content(text=text, text_option=text_option)
    meta = _parse_add_metadata(metadata)

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


@index_app.command("files")
def index_files(  # noqa: PLR0912
    ctx: typer.Context,
    path: str = typer.Argument(
        ".",
        help="Path to directory to index (default: current directory)",
    ),
    exclude: list[str] = typer.Option(
        None,
        "--exclude",
        "-e",
        help="Additional patterns to exclude (can be specified multiple times)",
    ),
    incremental: bool = typer.Option(
        True,
        "--incremental/--no-incremental",
        help="Enable/disable incremental indexing (skip unchanged files)",
    ),
) -> None:
    """Bulk-index .py, .md, and .rst files from a directory."""
    # Validate path first (before services)
    index_path = Path(path).resolve()
    if not index_path.exists():
        error_console.print(f"[red]Error: Path does not exist: {path}[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    if not index_path.is_dir():
        error_console.print(f"[red]Error: Path is not a directory: {path}[/red]")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    # Find the git root to use for blob hash scanning
    repo_root = get_git_root(index_path)
    if repo_root is None:
        error_console.print(
            "[yellow]Warning: Not a git repository, falling back to non-incremental mode[/yellow]"
        )
        incremental = False
        repo_root = index_path
    else:
        # Use index_path as repo_root if it's within the git repo
        if not index_path.is_relative_to(repo_root):
            repo_root = index_path

    # Get current tracked files with their blob hashes
    current_blobs: dict[str, TrackedFile | str] = {}
    if incremental:
        try:
            current_blobs = scan_git_blobs(repo_root)
        except Exception as e:
            error_console.print(
                f"[yellow]Warning: Could not scan git blobs, falling back to non-incremental: {e}[/yellow]"
            )
            incremental = False

    # Load existing file state for incremental indexing
    file_state: dict[str, str] = {}
    file_chunks: dict[str, int] = {}
    config = require_initialized(ctx)
    data_dir = config.data_dir
    state_file = _get_file_state_path(data_dir)

    if incremental and state_file.exists():
        repo_id = _get_repo_id(index_path)
        all_file_states = _load_file_state(state_file, index_path)
        repo_state = all_file_states.get(repo_id, {})
        raw_file_hashes = repo_state.get("file_hashes")
        if isinstance(raw_file_hashes, dict):
            file_state = {
                rel_path: blob_hash
                for rel_path, blob_hash in raw_file_hashes.items()
                if isinstance(rel_path, str) and isinstance(blob_hash, str)
            }
        file_chunks = _extract_file_chunks(repo_state)

    with (
        get_cli_services(ctx) as (svc, embedder, store),
        _FileIndexProgressReporter(output_console=console) as progress_reporter,
    ):
            progress_reporter.begin_stage(_FILE_INDEX_STAGE_SCANNING, total=None)

            # Build exclusion patterns (additive to defaults)
            user_excludes = frozenset(exclude) if exclude else None

            # Use the scanner module for file discovery
            # This includes .py, .md, .rst by default and excludes _build, .git, etc.
            files_to_index = scan_files(index_path, exclude_patterns=user_excludes)
            progress_reporter.set_total(
                _FILE_INDEX_STAGE_SCANNING,
                total=len(files_to_index),
            )

            # Determine which files need indexing
            files_to_process: list[tuple[Path, str, str | None]] = []
            current_tracked_paths: set[str] = set()
            next_file_hashes: dict[str, str] = {}
            next_file_chunks: dict[str, int] = {}
            for file in files_to_index:
                progress_reporter.advance(_FILE_INDEX_STAGE_SCANNING)

                # Get relative path from repo root
                try:
                    rel_path = file.relative_to(repo_root)
                except ValueError:
                    # File is outside repo root, use absolute path as string
                    rel_path = file

                rel_path_str = str(rel_path)

                if incremental:
                    # Check if file is tracked in git
                    blob_hash = _extract_blob_hash(current_blobs.get(rel_path_str))
                    if blob_hash is None:
                        # File not in git (might be untracked or in exclusion)
                        # Skip it for incremental mode
                        continue

                    current_tracked_paths.add(rel_path_str)

                    # Check if hash changed
                    old_hash = file_state.get(rel_path_str)
                    if old_hash == blob_hash:
                        # File unchanged, skip
                        next_file_hashes[rel_path_str] = blob_hash
                        next_file_chunks[rel_path_str] = file_chunks.get(rel_path_str, 1)
                        continue
                else:
                    blob_hash = None

                files_to_process.append((file, rel_path_str, blob_hash))

            progress_reporter.complete(_FILE_INDEX_STAGE_SCANNING)

            console.print(
                f"[green]Indexing {len(files_to_process)} files"
                + (f" (skipped {len(files_to_index) - len(files_to_process)} unchanged)" if incremental else "")
                + "...[/green]"
            )

            indexed_count = 0
            skipped_count = 0

            try:
                work_items = build_file_work_items(files_to_process)
                progress_reporter.begin_stage(
                    _FILE_INDEX_STAGE_PARSING,
                    total=len(work_items),
                )
                parsed_results = parse_files_pipeline(work_items)
                progress_reporter.advance(
                    _FILE_INDEX_STAGE_PARSING,
                    len(parsed_results),
                )
                progress_reporter.complete(_FILE_INDEX_STAGE_PARSING)

                total_chunks = sum(
                    len(parsed_result.chunks) for parsed_result in parsed_results
                )
                progress_reporter.begin_stage(
                    _FILE_INDEX_STAGE_EMBEDDING,
                    total=total_chunks,
                )
                progress_reporter.begin_stage(
                    _FILE_INDEX_STAGE_STORING,
                    total=total_chunks,
                )

                file_states: dict[str, _FileIndexBatchState] = {}
                ordered_file_states: list[str] = []
                batch_manager = BatchManager(
                    flush_fn=lambda batch: _flush_file_chunk_batch(
                        batch,
                        embedder=embedder,
                        store=store,
                        on_stage_progress=progress_reporter.advance,
                    ),
                )

                for parsed_result in parsed_results:
                    file = Path(parsed_result.work_item.path)
                    rel_path_str = parsed_result.work_item.relative_path
                    blob_hash = parsed_result.work_item.blob_hash
                    previous_hash = file_state.get(rel_path_str)
                    previous_chunk_count = file_chunks.get(rel_path_str, 1)
                    file_state_entry = _FileIndexBatchState(
                        file_path=file,
                        relative_path=rel_path_str,
                        blob_hash=blob_hash,
                        previous_hash=previous_hash,
                        previous_chunk_count=previous_chunk_count,
                    )

                    if parsed_result.has_error:
                        skipped_count += 1
                        error_console.print(
                            f"[yellow]Skipped {file}: {parsed_result.error}[/yellow]"
                        )
                        _preserve_previous_file_state(
                            incremental=incremental,
                            state=file_state_entry,
                            next_file_hashes=next_file_hashes,
                            next_file_chunks=next_file_chunks,
                        )
                        continue

                    if not parsed_result.chunks:
                        skipped_count += 1
                        error_console.print(
                            f"[yellow]Skipped {file}: no parseable content[/yellow]"
                        )
                        _preserve_previous_file_state(
                            incremental=incremental,
                            state=file_state_entry,
                            next_file_hashes=next_file_hashes,
                            next_file_chunks=next_file_chunks,
                        )
                        continue

                    file_states[rel_path_str] = file_state_entry
                    ordered_file_states.append(rel_path_str)

                    base_metadata = dict(parsed_result.work_item.metadata)
                    if blob_hash:
                        base_metadata["blob_hash"] = blob_hash

                    for fallback_chunk_index, chunk in enumerate(parsed_result.chunks):
                        metadata, chunk_index = _normalize_chunk_metadata(
                            base_metadata=base_metadata,
                            chunk_metadata=chunk.metadata,
                            fallback_chunk_index=fallback_chunk_index,
                        )
                        file_state_entry.expected_chunks += 1
                        batch_results = batch_manager.add(
                            ChunkBatchItem(
                                doc_id=_build_file_doc_id(parsed_result.work_item.path, chunk_index),
                                text=chunk.text,
                                metadata=metadata,
                                file_path=parsed_result.work_item.path,
                                relative_path=rel_path_str,
                                chunk_index=chunk_index,
                            )
                        )
                        _apply_chunk_batch_results(batch_results, file_states)

                _apply_chunk_batch_results(batch_manager.flush(), file_states)
                progress_reporter.complete(_FILE_INDEX_STAGE_EMBEDDING)
                progress_reporter.complete(_FILE_INDEX_STAGE_STORING)

                for rel_path_str in ordered_file_states:
                    file_state_entry = file_states[rel_path_str]
                    indexed_chunk_count = file_state_entry.indexed_chunks

                    if indexed_chunk_count <= 0:
                        skipped_count += 1
                        error_console.print(
                            f"[yellow]Skipped {file_state_entry.file_path}: no parseable content[/yellow]"
                        )
                        _preserve_previous_file_state(
                            incremental=incremental,
                            state=file_state_entry,
                            next_file_hashes=next_file_hashes,
                            next_file_chunks=next_file_chunks,
                        )
                        continue

                    if (
                        file_state_entry.errors
                        or indexed_chunk_count != file_state_entry.expected_chunks
                    ):
                        skipped_count += 1
                        failure_reason = (
                            file_state_entry.errors[0]
                            if file_state_entry.errors
                            else "one or more chunks failed to index"
                        )
                        error_console.print(
                            f"[yellow]Skipped {file_state_entry.file_path}: {failure_reason}[/yellow]"
                        )
                        _preserve_previous_file_state(
                            incremental=incremental,
                            state=file_state_entry,
                            next_file_hashes=next_file_hashes,
                            next_file_chunks=next_file_chunks,
                        )
                        continue

                    indexed_count += 1
                    if incremental and file_state_entry.blob_hash:
                        next_file_hashes[rel_path_str] = file_state_entry.blob_hash
                        next_file_chunks[rel_path_str] = indexed_chunk_count

                        if file_state_entry.previous_chunk_count > indexed_chunk_count:
                            _delete_file_docs(
                                store,
                                rel_path_str,
                                repo_root,
                                start_chunk=indexed_chunk_count,
                                chunk_count=file_state_entry.previous_chunk_count,
                            )

                # Handle deletions: files that were in state but are no longer tracked
                if incremental and file_state:
                    deleted_files = set(file_state.keys()) - current_tracked_paths
                    if deleted_files:
                        for deleted_path in deleted_files:
                            _delete_file_docs(
                                store,
                                deleted_path,
                                repo_root,
                                chunk_count=file_chunks.get(deleted_path, 1),
                            )

                # Save updated file state
                if incremental:
                    _save_file_state(
                        state_file,
                        index_path,
                        next_file_hashes,
                        next_file_chunks,
                    )

            except Exception as e:
                console.print(f"[red]Error during indexing:[/red] {_escape_rich(str(e))}")
                raise typer.Exit(code=EXIT_ERROR)

    if skipped_count > 0:
        console.print(
            f"[green]Indexed {indexed_count} files, skipped {skipped_count}[/green]"
        )
    else:
        console.print(f"[green]Indexed {indexed_count} files[/green]")


def _run_commit_index(  # noqa: PLR0912, PLR0913
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
        console.print(
            f"[red]Error fetching commits:[/red] {_escape_rich(error_msg or str(e))}"
        )
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
                more_commits = git.get_commits_with_files_batch(
                    limit=batch_size, offset=offset
                )
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
            console.print(
                "[yellow]Warning: Last indexed commit not found in history.[/yellow]"
            )
            console.print(
                "[yellow]This may indicate stale state or garbage-collected commits.[/yellow]"
            )
            console.print(
                "[yellow]Indexing all available commits without advancing state.[/yellow]"
            )
            # Use filtered (contains all commits from all batches), not just first page
            commits = filtered
    elif all_history:
        # For --all mode, get all commits (may need multiple fetches)
        offset = len(commits)
        while True:
            more_commits = git.get_commits_with_files_batch(
                limit=batch_size, offset=offset
            )
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

    console.print(
        f"[green]Indexing {len(commits)} commits (batch size: {batch_size})...[/green]"
    )

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
            console=console,
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

                        # Use Rust's assemble_commit_text when available (falls back to Python)
                        text = assemble_commit_text(commit.message, diff_text)
                        commit_data.append((commit, text))
                    except Exception as e:
                        error_console.print(
                            f"[yellow]Skipped commit {commit.hash[:8]}: {e}[/yellow]"
                        )
                        skipped_count += 1
                        progress.advance(task)

                # Batch generate embeddings for all valid commits in this batch
                if commit_data:
                    texts = [text for _, text in commit_data]
                    embeddings = list(embedder.embed(texts))

                    # Insert embeddings into store
                    for (commit, text), vector in zip(
                        commit_data, embeddings, strict=True
                    ):
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
                                    doc_id,
                                    vector,
                                    text,
                                    {
                                        "type": "commit",
                                        "hash": commit.hash,
                                        "author": commit.author,
                                        "message": commit.message,
                                    },
                                )
                                indexed_count += 1
                                if not frontier_blocked:
                                    frontier_commit_hash = commit.hash
                            except Exception as e:
                                error_console.print(
                                    f"[yellow]Skipped commit {commit.hash[:8]}: {e}[/yellow]"
                                )
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
        console.print(
            f"[green]Indexed {indexed_count} commits, skipped {skipped_count}[/green]"
        )
    else:
        console.print(f"[green]Indexed {indexed_count} commits[/green]")


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
        help="Index all commits from git history (ignores --limit)",
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
        help="Mode: --incremental (default) checks indexed state, --full re-indexes all",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-index (in incremental mode: ignores last indexed state)",
    ),
) -> None:
    """Index git commits from a repository.

    Parses commit messages and diffs to create searchable commit history.
    """
    _run_commit_index(
        ctx, path, limit, all_history, batch_size, max_diff_size, incremental, force
    )


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
        error_console.print(
            f"[red]Error: Refusing to install hook over symlink: {hook_path}[/red]"
        )
        console.print(
            "[yellow]Symlink targets may be outside the hooks directory.[/yellow]"
        )
        console.print("Remove the symlink first, then install the hook.")
        raise typer.Exit(code=EXIT_INVALID_ARG)

    # Check if it's a Town Elder hook (ours or someone else's)
    if hook_path.exists() and force:
        existing_content = _safe_read_hook(hook_path)
        if existing_content is None or not _is_te_hook(existing_content):
            error_console.print(
                "[yellow]Warning: Existing hook is not a Town Elder hook[/yellow]"
            )
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
    data_dir_arg = f"--data-dir {escaped_data_dir}"
    hook_content = _build_post_commit_hook_content(data_dir_arg)

    hook_path.write_text(hook_content)

    # Make it executable
    os.chmod(hook_path, 0o755)

    console.print(f"[green]Installed post-commit hook at {hook_path}[/green]")
    console.print("Commits will now be automatically indexed")


@hook_app.command()
def uninstall(
    path: str = typer.Option(".", "--repo", "-r", help="Git repository path"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Allow deletion of non-Town Elder hooks"
    ),
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
        error_console.print(
            f"[yellow]No post-commit hook found at {hook_path}[/yellow]"
        )
        return

    # Check if it's a Town Elder hook
    content = _safe_read_hook(hook_path)
    is_te_hook = content is not None and _is_te_hook(content)

    if not is_te_hook and not force:
        error_console.print(
            "[red]Error: Hook exists but is not a Town Elder hook[/red]"
        )
        console.print("[yellow]Use --force to delete non-Town Elder hooks[/yellow]")
        raise typer.Exit(code=EXIT_ERROR)

    if not is_te_hook and force:
        error_console.print(
            "[yellow]Warning: Deleting non-Town Elder hook (--force specified)[/yellow]"
        )

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


app.add_typer(index_app, name="index")
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
        error_console.print(
            f"[red]Error: Invalid format '{format}'. Use 'json' or 'jsonl'.[/red]"
        )
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
            console.print(
                f"[green]Exported {len(documents)} documents to {output}[/green]"
            )
        except Exception as e:
            console.print(f"[red]Error writing to file:[/red] {_escape_rich(str(e))}")
            raise typer.Exit(code=EXIT_ERROR)


def run() -> None:
    """Console script entry point."""
    app()


if __name__ == "__main__":
    run()
