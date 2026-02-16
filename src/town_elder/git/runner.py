"""Git runner implementation using subprocess."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Minimum parts expected in git log format output
_MIN_LOG_PARTS = 4
_LOG_FIELD_SEPARATOR = "\x1f"
_LOG_RECORD_SEPARATOR = "\x1e"
_NUMSTAT_PARTS = 3  # additions, deletions, filename

# Default max diff size in bytes (100KB)
DEFAULT_MAX_DIFF_SIZE = 100 * 1024


@dataclass
class Commit:
    """Represents a git commit."""
    hash: str
    message: str
    author: str
    date: datetime
    files_changed: list[str] = field(default_factory=list)
    diff: str | None = None  # Optional: cached diff for batch operations


class GitRunner:
    """Git runner using subprocess to execute git commands."""

    def __init__(self, repo_path: Path | str | None = None):
        """Initialize the GitRunner.

        Args:
            repo_path: Path to the git repository. Defaults to current directory.
        """
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()

    def _run_git(self, args: list[str]) -> str:
        """Run a git command and return the output.

        Args:
            args: Git command arguments.

        Returns:
            Command output.

        Raises:
            subprocess.CalledProcessError: If the git command fails.
        """
        cmd = ["git", "-C", str(self.repo_path)] + args
        result = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()

    def get_commits(self, since: str | None = None, limit: int = 100, offset: int = 0, include_files: bool = True) -> list[Commit]:
        """Get commits from the repository.

        Args:
            since: Get commits after this date/hash (optional).
            limit: Maximum number of commits to return (default 100).
            offset: Number of commits to skip (default 0).
            include_files: If True, include files changed per commit (batch when possible).

        Returns:
            List of Commit objects.
        """
        args = ["log", "-n", str(limit), "--format=%H%x1f%s%x1f%an%x1f%aI%x1e"]
        if since:
            args.extend([f"--since={since}"])
        if offset > 0:
            args.extend([f"--skip={offset}"])

        output = self._run_git(args)
        commits = []

        for record in output.split(_LOG_RECORD_SEPARATOR):
            if not record.strip():
                continue
            parts = record.lstrip("\n").rstrip("\n").split(_LOG_FIELD_SEPARATOR)
            if len(parts) >= _MIN_LOG_PARTS:
                commit_hash, message, author, date_str = parts[:_MIN_LOG_PARTS]
                commit_hash = commit_hash.strip()

                # Parse the date from git's iso format
                date = datetime.fromisoformat(date_str)

                # Get files changed for this commit
                files: list[str] = []
                if include_files:
                    files = self._get_commit_files(commit_hash)

                commits.append(Commit(
                    hash=commit_hash,
                    message=message,
                    author=author,
                    date=date,
                    files_changed=files
                ))

        return commits

    def get_commits_with_files_batch(self, since: str | None = None, limit: int = 100, offset: int = 0) -> list[Commit]:
        """Get commits with files changed in batch (single git call).

        Uses git log with --numstat to get files changed per commit in one call.

        Args:
            since: Get commits after this date/hash (optional).
            limit: Maximum number of commits to return.
            offset: Number of commits to skip.

        Returns:
            List of Commit objects with files_changed populated.
        """
        args = [
            "log",
            "-n", str(limit),
            "--format=%H%x1f%s%x1f%an%x1f%aI%x1e",
            "--numstat"
        ]
        if since:
            args.append(f"--since={since}")
        if offset > 0:
            args.extend([f"--skip={offset}"])

        try:
            output = self._run_git(args)
        except subprocess.CalledProcessError:
            # Fallback to individual calls
            return self.get_commits(since=since, limit=limit, offset=offset, include_files=True)

        return self._parse_commits_with_numstat(output)

    def _parse_commits_with_numstat(self, output: str) -> list[Commit]:
        """Parse git log output with numstat into Commit objects."""
        commits = []
        current_commit: dict | None = None
        current_files: list[str] = []

        for line in output.split("\n"):
            is_commit_header = _LOG_RECORD_SEPARATOR in line
            if is_commit_header:
                if current_commit is not None and current_commit.get("hash"):
                    commits.append(Commit(
                        hash=current_commit["hash"],
                        message=current_commit["message"],
                        author=current_commit["author"],
                        date=current_commit["date"],
                        files_changed=current_files
                    ))
                current_commit, current_files = self._parse_commit_header(line)
            elif current_commit is not None:
                file_path = self._parse_numstat_line(line)
                if file_path:
                    current_files.append(file_path)

        if current_commit is not None and current_commit.get("hash"):
            commits.append(Commit(
                hash=current_commit["hash"],
                message=current_commit["message"],
                author=current_commit["author"],
                date=current_commit["date"],
                files_changed=current_files
            ))

        return commits

    def _parse_commit_header(self, line: str) -> tuple[dict | None, list[str]]:
        """Parse a git log commit header line."""
        parts = line.rstrip("\n").rstrip(_LOG_RECORD_SEPARATOR).split(_LOG_FIELD_SEPARATOR)
        if len(parts) >= _MIN_LOG_PARTS:
            return (
                {
                    "hash": parts[0],
                    "message": parts[1],
                    "author": parts[2],
                    "date": datetime.fromisoformat(parts[3])
                },
                []
            )
        return None, []

    def _parse_numstat_line(self, line: str) -> str | None:
        """Parse a numstat line to extract file path.

        Returns:
            File path if valid, None otherwise.
        """
        if not line.strip() or "\t" not in line:
            return None
        parts = line.split("\t")
        if len(parts) < _NUMSTAT_PARTS:
            return None
        file_path = parts[2].strip()
        if not file_path or file_path == "-":
            return None
        return file_path

    def _get_commit_files(self, commit_hash: str) -> list[str]:
        """Get the list of files changed in a commit.

        Args:
            commit_hash: The commit hash.

        Returns:
            List of file paths.
        """
        try:
            output = self._run_git(["diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash])
            return [f for f in output.split("\n") if f.strip()]
        except subprocess.CalledProcessError:
            return []

    def get_diff(self, commit_hash: str, max_size: int = DEFAULT_MAX_DIFF_SIZE) -> str:
        """Get the diff for a specific commit.

        Args:
            commit_hash: The commit hash.
            max_size: Maximum size in bytes (truncates if exceeded).

        Returns:
            The diff content (potentially truncated).
        """
        diff = self._run_git(["show", commit_hash, "--format="])
        if len(diff.encode()) > max_size:
            diff = diff[:max_size] + f"\n\n[truncated - exceeded {max_size} byte limit]"
        return diff

    def get_diffs_batch(self, commit_hashes: list[str], max_size: int = DEFAULT_MAX_DIFF_SIZE) -> dict[str, str]:
        """Get diffs for multiple commits in a single git command.

        Uses git log with patch format to get all diffs at once.

        Args:
            commit_hashes: List of commit hashes.
            max_size: Maximum size per diff in bytes.

        Returns:
            Dictionary mapping commit hash to diff content.
        """
        if not commit_hashes:
            return {}

        # Use git log with patch to get all diffs in one call
        # Format: hash followed by patch
        try:
            # Get all diffs in one call using --format and --patch
            # We need to separate commits with a delimiter
            output = self._run_git([
                "log",
                "--format=%H%x1e",
                "--patch",
                "--no-color",
                f"--max-count={len(commit_hashes)}",
                *commit_hashes,
            ])

            # Parse the output - each commit starts with hash + delimiter
            diffs = {}
            current_hash = None
            current_diff_lines: list[str] = []

            for line in output.split("\n"):
                # Check if this line starts a new commit (contains hash + delimiter)
                if _LOG_RECORD_SEPARATOR in line:
                    # Extract hash from the line
                    parts = line.split(_LOG_RECORD_SEPARATOR)
                    if len(parts) >= 1 and parts[0]:
                        # Save previous commit if exists (even if empty - e.g., merge commits)
                        if current_hash:
                            diff_text = "\n".join(current_diff_lines)
                            if len(diff_text.encode()) > max_size:
                                diff_text = diff_text[:max_size] + f"\n\n[truncated - exceeded {max_size} byte limit]"
                            diffs[current_hash] = diff_text

                        # Start new commit
                        current_hash = parts[0]
                        current_diff_lines = []
                elif current_hash:
                    current_diff_lines.append(line)

            # Don't forget the last commit (even if empty - e.g., merge commits)
            if current_hash:
                diff_text = "\n".join(current_diff_lines)
                if len(diff_text.encode()) > max_size:
                    diff_text = diff_text[:max_size] + f"\n\n[truncated - exceeded {max_size} byte limit]"
                diffs[current_hash] = diff_text

            # Verify we got exactly the requested hashes
            # git log may omit commits when hashes are non-contiguous
            if set(diffs.keys()) != set(commit_hashes):
                # Fallback to individual calls to ensure we return exactly requested hashes
                return {h: self.get_diff(h, max_size) for h in commit_hashes}

            return diffs
        except subprocess.CalledProcessError:
            # Fallback to individual calls if batch fails
            return {h: self.get_diff(h, max_size) for h in commit_hashes}

    def get_commit_range(self, start: str, end: str) -> list[str]:
        """Get commits between two references.

        Args:
            start: Start commit hash or ref.
            end: End commit hash or ref.

        Returns:
            List of commit hashes.
        """
        output = self._run_git(["log", "--format=%H", f"{start}..{end}"])
        return [h for h in output.split("\n") if h.strip()]
