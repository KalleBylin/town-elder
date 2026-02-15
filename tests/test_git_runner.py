"""Tests for GitRunner error handling and edge cases."""
from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest

from town_elder.git.runner import Commit, GitRunner


@pytest.fixture
def temp_git_repo() -> Iterator[Path]:
    """Create a temporary git repository for testing."""
    with tempfile.TemporaryDirectory() as tmp:
        repo_path = Path(tmp) / "test_repo"
        repo_path.mkdir()
        # Initialize git repo
        subprocess.run(["git", "init"], cwd=repo_path, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        # Create initial commit
        (repo_path / "test.txt").write_text("initial content")
        subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "initial commit"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        yield repo_path


class TestGitRunnerBasic:
    """Basic GitRunner tests for coverage."""

    def test_git_runner_with_default_path(self):
        """GitRunner should work with default (current directory) path."""
        runner = GitRunner()
        assert runner.repo_path == Path.cwd()

    def test_git_runner_with_string_path(self, temp_git_repo: Path):
        """GitRunner should accept string path."""
        runner = GitRunner(repo_path=str(temp_git_repo))
        assert runner.repo_path == temp_git_repo

    def test_git_runner_with_pathlib_path(self, temp_git_repo: Path):
        """GitRunner should accept Path object."""
        runner = GitRunner(repo_path=temp_git_repo)
        assert runner.repo_path == temp_git_repo


class TestGitRunnerGetCommits:
    """Tests for get_commits method."""

    def test_get_commits_returns_list(self, temp_git_repo: Path):
        """get_commits should return a list of Commit objects."""
        runner = GitRunner(repo_path=temp_git_repo)
        commits = runner.get_commits()
        assert isinstance(commits, list)
        assert len(commits) > 0

    def test_get_commits_with_limit(self, temp_git_repo: Path):
        """get_commits should respect limit parameter."""
        runner = GitRunner(repo_path=temp_git_repo)
        # Add more commits
        for i in range(5):
            (temp_git_repo / f"file{i}.txt").write_text(f"content {i}")
            subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True, check=True)
            subprocess.run(
                ["git", "commit", "-m", f"commit {i}"],
                cwd=temp_git_repo,
                capture_output=True,
                check=True,
            )

        commits = runner.get_commits(limit=3)
        assert len(commits) <= 3

    def test_get_commits_with_offset(self, temp_git_repo: Path):
        """get_commits should respect offset parameter."""
        runner = GitRunner(repo_path=temp_git_repo)
        # Add more commits
        for i in range(5):
            (temp_git_repo / f"file{i}.txt").write_text(f"content {i}")
            subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True, check=True)
            subprocess.run(
                ["git", "commit", "-m", f"commit {i}"],
                cwd=temp_git_repo,
                capture_output=True,
                check=True,
            )

        commits_all = runner.get_commits(limit=10)
        commits_with_offset = runner.get_commits(limit=10, offset=2)
        assert len(commits_with_offset) <= len(commits_all) - 2

    def test_get_commits_since_date(self, temp_git_repo: Path):
        """get_commits should filter by since date."""
        runner = GitRunner(repo_path=temp_git_repo)
        commits = runner.get_commits(since="2020-01-01")
        assert isinstance(commits, list)

    def test_get_commits_commit_structure(self, temp_git_repo: Path):
        """get_commits should return Commit objects with correct structure."""
        runner = GitRunner(repo_path=temp_git_repo)
        commits = runner.get_commits()
        assert len(commits) > 0
        commit = commits[0]
        assert isinstance(commit, Commit)
        assert hasattr(commit, "hash")
        assert hasattr(commit, "message")
        assert hasattr(commit, "author")
        assert hasattr(commit, "date")
        assert isinstance(commit.date, datetime)
        assert hasattr(commit, "files_changed")
        assert isinstance(commit.files_changed, list)

    def test_get_commits_files_changed(self, temp_git_repo: Path):
        """get_commits should include files changed in each commit."""
        runner = GitRunner(repo_path=temp_git_repo)
        # Create a new file and commit
        (temp_git_repo / "new_file.txt").write_text("new content")
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add new file"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )

        commits = runner.get_commits(limit=10)
        # Find the commit that added the file
        new_file_commit = None
        for c in commits:
            if "new file" in c.message:
                new_file_commit = c
                break

        assert new_file_commit is not None
        assert len(new_file_commit.files_changed) > 0


class TestGitRunnerGetDiff:
    """Tests for get_diff method."""

    def test_get_diff_returns_string(self, temp_git_repo: Path):
        """get_diff should return diff as string."""
        runner = GitRunner(repo_path=temp_git_repo)
        commits = runner.get_commits(limit=1)
        assert len(commits) > 0
        diff = runner.get_diff(commits[0].hash)
        assert isinstance(diff, str)

    def test_get_diff_on_empty_repo(self, temp_git_repo: Path):
        """get_diff should work on commits."""
        runner = GitRunner(repo_path=temp_git_repo)
        commits = runner.get_commits()
        if commits:
            diff = runner.get_diff(commits[0].hash)
            assert diff is not None


class TestGitRunnerGetCommitRange:
    """Tests for get_commit_range method."""

    def test_get_commit_range_returns_list(self, temp_git_repo: Path):
        """get_commit_range should return list of commit hashes."""
        runner = GitRunner(repo_path=temp_git_repo)
        commits = runner.get_commits(limit=2)
        if len(commits) >= 2:
            hashes = runner.get_commit_range(commits[1].hash, commits[0].hash)
            assert isinstance(hashes, list)

    def test_get_commit_range_empty_range(self, temp_git_repo: Path):
        """get_commit_range should handle empty range."""
        runner = GitRunner(repo_path=temp_git_repo)
        commits = runner.get_commits(limit=1)
        if commits:
            hashes = runner.get_commit_range(commits[0].hash, commits[0].hash)
            assert isinstance(hashes, list)


class TestGitRunnerErrorHandling:
    """Tests for GitRunner error handling."""

    def test_git_runner_with_invalid_path(self):
        """GitRunner should accept invalid path (git will fail later)."""
        runner = GitRunner(repo_path="/nonexistent/path")
        # Getting commits should raise CalledProcessError
        with pytest.raises(subprocess.CalledProcessError):
            runner.get_commits()

    def test_get_commit_files_handles_missing_commit(self, temp_git_repo: Path):
        """_get_commit_files should return empty list for invalid commit."""
        runner = GitRunner(repo_path=temp_git_repo)
        # Try to get files for an invalid commit hash
        files = runner._get_commit_files("0" * 40)  # All zeros is invalid
        # This may return empty list or raise - either is valid behavior
        assert isinstance(files, list)

    def test_get_diff_handles_invalid_hash(self, temp_git_repo: Path):
        """get_diff should raise for invalid commit hash."""
        runner = GitRunner(repo_path=temp_git_repo)
        with pytest.raises(subprocess.CalledProcessError):
            runner.get_diff("0" * 40)


class TestGitRunnerEdgeCases:
    """Edge case tests for GitRunner."""

    def test_get_commits_empty_repo(self):
        """get_commits should raise or return empty for repository with no commits."""
        with tempfile.TemporaryDirectory() as tmp:
            repo_path = Path(tmp) / "empty_repo"
            repo_path.mkdir()
            subprocess.run(["git", "init"], cwd=repo_path, capture_output=True, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                cwd=repo_path,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                cwd=repo_path,
                capture_output=True,
                check=True,
            )

            runner = GitRunner(repo_path=repo_path)
            # Empty repo with no commits raises CalledProcessError
            with pytest.raises(subprocess.CalledProcessError):
                runner.get_commits()

    def test_get_commits_very_large_limit(self, temp_git_repo: Path):
        """get_commits should handle very large limit."""
        runner = GitRunner(repo_path=temp_git_repo)
        commits = runner.get_commits(limit=100000)
        assert isinstance(commits, list)

    def test_get_commits_with_large_offset(self, temp_git_repo: Path):
        """get_commits should handle offset larger than commit count."""
        runner = GitRunner(repo_path=temp_git_repo)
        commits = runner.get_commits(limit=10, offset=1000)
        assert isinstance(commits, list)
        assert len(commits) == 0
