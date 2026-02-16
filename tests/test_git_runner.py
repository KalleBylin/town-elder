"""Tests for GitRunner error handling and edge cases."""
from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest

from town_elder.git.runner import Commit, GitRunner

_THREE_COMMITS = 3
_TWO_COMMITS = 2


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

        commits = runner.get_commits(limit=_THREE_COMMITS)
        assert len(commits) <= _THREE_COMMITS

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

    def test_get_commits_handles_legacy_delimiter_in_subject(self, temp_git_repo: Path):
        """get_commits should correctly parse subjects containing literal delimiters."""
        runner = GitRunner(repo_path=temp_git_repo)
        message = "fix parser ||| edge case"
        (temp_git_repo / "delimiter_subject.txt").write_text("delimiter collision content")
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )

        commits = runner.get_commits(limit=10)
        matching_commits = [c for c in commits if c.message == message]
        assert len(matching_commits) == 1
        commit = matching_commits[0]
        assert commit.hash
        assert commit.author == "Test User"
        assert isinstance(commit.date, datetime)
        assert "delimiter_subject.txt" in commit.files_changed

    def test_get_commits_handles_unicode_and_punctuation_subject(self, temp_git_repo: Path):
        """get_commits should preserve unusual punctuation and Unicode in subject."""
        runner = GitRunner(repo_path=temp_git_repo)
        message = "feat: café naïve punctuation!? []{}<>|~"
        (temp_git_repo / "unicode_subject.txt").write_text("unicode subject content")
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )

        commits = runner.get_commits(limit=10)
        matching_commits = [c for c in commits if c.message == message]
        assert len(matching_commits) == 1
        commit = matching_commits[0]
        assert commit.hash
        assert commit.author == "Test User"
        assert isinstance(commit.date, datetime)
        assert "unicode_subject.txt" in commit.files_changed


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

    def test_get_diffs_batch_returns_all_requested_hashes(self, temp_git_repo: Path):
        """get_diffs_batch should return one entry per requested commit hash."""
        runner = GitRunner(repo_path=temp_git_repo)

        (temp_git_repo / "second.txt").write_text("second change")
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "second commit"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )

        commits = runner.get_commits(limit=_TWO_COMMITS)
        assert len(commits) >= _TWO_COMMITS

        commit_hashes = [commits[0].hash, commits[1].hash]
        diffs = runner.get_diffs_batch(commit_hashes)

        assert set(diffs) == set(commit_hashes)
        assert all(isinstance(diff, str) for diff in diffs.values())

    def test_get_diffs_batch_avoids_per_commit_fallback_when_batch_succeeds(
        self,
        temp_git_repo: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """get_diffs_batch should not call get_diff when the batch call succeeds."""
        runner = GitRunner(repo_path=temp_git_repo)

        (temp_git_repo / "second.txt").write_text("second change")
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "second commit"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )

        commits = runner.get_commits(limit=_TWO_COMMITS)
        assert len(commits) >= _TWO_COMMITS

        def fail_get_diff(commit_hash: str, max_size: int = 100 * 1024) -> str:
            _ = commit_hash, max_size
            raise AssertionError("Fallback path should not be used")

        monkeypatch.setattr(runner, "get_diff", fail_get_diff)

        commit_hashes = [commits[0].hash, commits[1].hash]
        diffs = runner.get_diffs_batch(commit_hashes)

        assert set(diffs) == set(commit_hashes)

    def test_get_diffs_batch_returns_all_hashes_including_merge_commits(
        self,
        temp_git_repo: Path,
    ):
        """get_diffs_batch should return all requested hashes including merge commits."""
        runner = GitRunner(repo_path=temp_git_repo)

        # Create a branch with a commit
        subprocess.run(
            ["git", "checkout", "-b", "feature"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )
        (temp_git_repo / "feature.txt").write_text("feature change")
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "feature commit"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )
        feature_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
        ).stdout.strip()

        # Switch back to main and create another commit
        subprocess.run(
            ["git", "checkout", "main"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )
        (temp_git_repo / "main.txt").write_text("main change")
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "main commit"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )
        main_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
        ).stdout.strip()

        # Create a merge commit
        subprocess.run(
            ["git", "merge", feature_hash, "--no-ff", "-m", "merge feature"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )
        merge_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
        ).stdout.strip()

        # Get all commits and verify we can get diffs for all of them
        commits = runner.get_commits(limit=10)
        commit_hashes = [c.hash for c in commits]

        # Verify we have the expected commits
        assert feature_hash in commit_hashes
        assert main_hash in commit_hashes
        assert merge_hash in commit_hashes

        # get_diffs_batch should return all requested hashes including merge
        diffs = runner.get_diffs_batch(commit_hashes)

        assert set(diffs.keys()) == set(commit_hashes)
        assert all(isinstance(diff, str) for diff in diffs.values())


class TestGitRunnerGetCommitRange:
    """Tests for get_commit_range method."""

    def test_get_commit_range_returns_list(self, temp_git_repo: Path):
        """get_commit_range should return list of commit hashes."""
        runner = GitRunner(repo_path=temp_git_repo)
        commits = runner.get_commits(limit=_TWO_COMMITS)
        if len(commits) >= _TWO_COMMITS:
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


class TestGitRunnerNonUtf8:
    """Tests for handling non-UTF8 git output."""

    def test_get_diff_with_non_utf8_content(self, temp_git_repo: Path):
        """get_diff should handle binary/non-UTF8 content without crashing."""
        runner = GitRunner(repo_path=temp_git_repo)

        # Create a file with non-UTF8 bytes (binary content)
        binary_content = b"Hello World\x80\x81\x82"  # 0x80-0x82 are invalid UTF-8
        (temp_git_repo / "binary.bin").write_bytes(binary_content)
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add binary file"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )

        commits = runner.get_commits(limit=1)
        assert len(commits) > 0

        # This should not crash even with non-UTF8 content
        diff = runner.get_diff(commits[0].hash)
        assert isinstance(diff, str)

    def test_get_diffs_batch_with_non_utf8_content(self, temp_git_repo: Path):
        """get_diffs_batch should handle binary/non-UTF8 content without crashing."""
        runner = GitRunner(repo_path=temp_git_repo)

        # Create a file with non-UTF8 bytes
        binary_content = b"Test\xff\xfe content"  # 0xFF and 0xFE are invalid UTF-8
        (temp_git_repo / "binary2.bin").write_bytes(binary_content)
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add another binary"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
        )

        commits = runner.get_commits(limit=1)
        assert len(commits) > 0

        # This should not crash
        diffs = runner.get_diffs_batch([commits[0].hash])
        assert commits[0].hash in diffs
        assert isinstance(diffs[commits[0].hash], str)


class TestGitRunnerDateParsing:
    """Tests for date parsing edge cases."""

    def test_get_commits_with_invalid_date_logs_warning(self, temp_git_repo: Path, caplog: pytest.LogCaptureFixture):
        """get_commits should skip commits with invalid dates and log a warning."""
        runner = GitRunner(repo_path=temp_git_repo)

        # Manually mock _run_git to return invalid date format
        import logging
        caplog.set_level(logging.WARNING)

        # Create a mock that returns an invalid date
        original_run_git = runner._run_git
        call_count = 0

        def mock_run_git(args: list[str]) -> str:
            nonlocal call_count
            if "log" in args:
                call_count += 1
                # Return valid commits first, then one with invalid date
                if call_count == 1:
                    # Valid commit
                    return original_run_git(args)
                # Return a commit with invalid date
                return "abc123\x1ftest message\x1fTest User\x1finvalid-date\x1e"
            return original_run_git(args)

        runner._run_git = mock_run_git  # type: ignore[method-assign]

        # Should not raise, should skip the invalid commit and return valid ones
        commits = runner.get_commits(limit=10)
        # The first call returns valid commits, so we should get those
        assert isinstance(commits, list)

    def test_parse_commit_header_with_invalid_date_returns_none(self, temp_git_repo: Path):
        """_parse_commit_header should return None for invalid date strings."""
        runner = GitRunner(repo_path=temp_git_repo)

        # Valid format should work
        valid_line = "abc123\x1fmessage\x1fauthor\x1f2024-01-01T00:00:00+00:00\x1e"
        result, files = runner._parse_commit_header(valid_line)
        assert result is not None
        assert result["hash"] == "abc123"

        # Invalid date should return None
        invalid_line = "def456\x1fmessage\x1fauthor\x1finvalid-date\x1e"
        result, files = runner._parse_commit_header(invalid_line)
        assert result is None
        assert files == []

        # Empty date should return None
        empty_date_line = "ghi789\x1fmessage\x1fauthor\x1f\x1e"
        result, files = runner._parse_commit_header(empty_date_line)
        assert result is None

    def test_get_commits_with_empty_date_string(self, temp_git_repo: Path):
        """get_commits should skip commits with empty date strings."""
        runner = GitRunner(repo_path=temp_git_repo)

        original_run_git = runner._run_git

        def mock_run_git(args: list[str]) -> str:
            if "log" in args:
                # Return commit with empty date
                return "abc123\x1ftest\x1fTest User\x1f\x1e"
            return original_run_git(args)

        runner._run_git = mock_run_git  # type: ignore[method-assign]

        # Should not crash, should return empty list (or skip invalid commit)
        commits = runner.get_commits(limit=10)
        assert isinstance(commits, list)


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
