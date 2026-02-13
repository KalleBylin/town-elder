"""Git runner implementation using subprocess."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class Commit:
    """Represents a git commit."""
    hash: str
    message: str
    author: str
    date: datetime
    files_changed: List[str] = field(default_factory=list)


class GitRunner:
    """Git runner using subprocess to execute git commands."""

    def __init__(self, repo_path: Optional[Path | str] = None):
        """Initialize the GitRunner.

        Args:
            repo_path: Path to the git repository. Defaults to current directory.
        """
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()

    def _run_git(self, args: List[str]) -> str:
        """Run a git command and return the output.

        Args:
            args: Git command arguments.

        Returns:
            Command output.

        Raises:
            subprocess.CalledProcessError: If the git command fails.
        """
        cmd = ["git", "-C", str(self.repo_path)] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()

    def get_commits(self, since: Optional[str] = None, limit: int = 100) -> List[Commit]:
        """Get commits from the repository.

        Args:
            since: Get commits after this date/hash (optional).
            limit: Maximum number of commits to return (default 100).

        Returns:
            List of Commit objects.
        """
        args = ["log", f"-n", str(limit), "--format=%H|||%s|||%an|||%ad|||%f", "--date=iso"]
        if since:
            args.extend([f"--since={since}"])

        output = self._run_git(args)
        commits = []

        for line in output.split("\n"):
            if not line.strip():
                continue
            parts = line.split("|||")
            if len(parts) >= 4:
                commit_hash = parts[0]
                message = parts[1]
                author = parts[2]
                date_str = parts[3]

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

    def _get_commit_files(self, commit_hash: str) -> List[str]:
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

    def get_commit_range(self, start: str, end: str) -> List[str]:
        """Get commits between two references.

        Args:
            start: Start commit hash or ref.
            end: End commit hash or ref.

        Returns:
            List of commit hashes.
        """
        output = self._run_git(["log", "--format=%H", f"{start}..{end}"])
        return [h for h in output.split("\n") if h.strip()]
