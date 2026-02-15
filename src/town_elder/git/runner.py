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


@dataclass
class Commit:
    """Represents a git commit."""
    hash: str
    message: str
    author: str
    date: datetime
    files_changed: list[str] = field(default_factory=list)


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

    def get_commits(self, since: str | None = None, limit: int = 100, offset: int = 0) -> list[Commit]:
        """Get commits from the repository.

        Args:
            since: Get commits after this date/hash (optional).
            limit: Maximum number of commits to return (default 100).
            offset: Number of commits to skip (default 0).

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
            parts = record.rstrip("\n").split(_LOG_FIELD_SEPARATOR)
            if len(parts) >= _MIN_LOG_PARTS:
                commit_hash, message, author, date_str = parts[:_MIN_LOG_PARTS]

                # Parse the date from git's iso format
                date = datetime.fromisoformat(date_str)

                # Get files changed for this commit
                files = self._get_commit_files(commit_hash)

                commits.append(Commit(
                    hash=commit_hash,
                    message=message,
                    author=author,
                    date=date,
                    files_changed=files
                ))

        return commits

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

    def get_diff(self, commit_hash: str) -> str:
        """Get the diff for a specific commit.

        Args:
            commit_hash: The commit hash.

        Returns:
            The diff content.
        """
        return self._run_git(["show", commit_hash, "--format="])

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
