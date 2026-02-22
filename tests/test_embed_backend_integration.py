"""Integration tests for Rust embedding backend in CLI workflows.

These tests validate that the Rust backend correctly handles:
- index files with batch embeddings
- index commits with batch embeddings
- search query embedding

Tests require:
- TE_USE_RUST_CORE=1 environment variable set
- Rust extension built and available
- Embedding model available
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

# Test constants
_MIN_FILES_FOR_BATCH_TEST = 5


# Check if Rust embedding backend is available
def is_rust_embed_available() -> bool:
    """Check if Rust embedding backend is available."""
    # Must have TE_USE_RUST_CORE enabled
    if os.environ.get("TE_USE_RUST_CORE") not in ("1", "true", "True"):
        return False

    try:
        from town_elder.rust_adapter import is_rust_embed_available

        return is_rust_embed_available()
    except ImportError:
        return False


# Conditional skip marker
rust_available = pytest.mark.skipif(
    not is_rust_embed_available(),
    reason="Rust embedding backend not available. Set TE_USE_RUST_CORE=1 and build extension.",
)


@pytest.fixture
def temp_git_repo() -> Iterator[Path]:
    """Create a temporary git repository for testing."""
    with tempfile.TemporaryDirectory() as tmp:
        repo_path = Path(tmp) / "test_repo"
        repo_path.mkdir()

        # Initialize git repo
        subprocess.run(
            ["git", "init"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
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
        subprocess.run(
            ["git", "add", "."],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "initial commit"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        yield repo_path


@pytest.fixture
def data_dir(temp_git_repo: Path) -> Path:
    """Create a data directory outside the git repo for the test."""
    # Use a sibling directory to avoid conflicts with .town_elder in the repo
    data_dir = temp_git_repo.parent / ".town_elder_test"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def run_te_command(
    args: list[str],
    data_dir: Path,
    env_override: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run a te command with the specified arguments."""
    env = os.environ.copy()
    env["TOWN_ELDER_DATA_DIR"] = str(data_dir)
    env["TOWN_ELDER_EMBED_BACKEND"] = "rust"
    if env_override:
        env.update(env_override)

    return subprocess.run(
        ["uv", "run", "te"] + args,
        capture_output=True,
        text=True,
        cwd=data_dir.parent,
        env=env,
    )


class TestRustBackendIndexFiles:
    """Tests for index files command with Rust backend."""

    @rust_available
    def test_index_files_with_rust_backend(self, temp_git_repo: Path, data_dir: Path):
        """Test that index files works with Rust backend."""
        # Create some Python files to index
        (temp_git_repo / "main.py").write_text(
            """def hello():
    '''Say hello.'''
    print("Hello, world!")
"""
        )
        (temp_git_repo / "utils.py").write_text(
            """def add(a, b):
    '''Add two numbers.'''
    return a + b
"""
        )

        # Initialize the database
        init_result = run_te_command(
            ["init", "--path", str(temp_git_repo)],
            data_dir=data_dir,
        )
        assert init_result.returncode == 0, f"Init failed: {init_result.stderr}"

        # Index files with Rust backend
        index_result = run_te_command(
            ["index", "files", str(temp_git_repo), "--no-incremental"],
            data_dir=data_dir,
        )

        # Check command succeeded
        assert index_result.returncode == 0, (
            f"Index files failed: {index_result.stderr}\n"
            f"stdout: {index_result.stdout}"
        )

        # Verify output indicates successful indexing
        # (The exact output format may vary, but should not contain errors)
        assert "error" not in index_result.stderr.lower(), (
            f"Index files had errors: {index_result.stderr}"
        )

        # Check stats to verify documents were indexed
        stats_result = run_te_command(["stats"], data_dir=data_dir)
        assert stats_result.returncode == 0, f"Stats failed: {stats_result.stderr}"

        # Verify at least some documents were indexed
        # The output should contain a document count > 0
        output = stats_result.stdout + stats_result.stderr
        # Look for number patterns that indicate documents were indexed
        import re

        doc_count_match = re.search(r"Documents:\s*(\d+)", output)
        if doc_count_match:
            doc_count = int(doc_count_match.group(1))
            assert doc_count > 0, f"Expected documents to be indexed, got {doc_count}"

    @rust_available
    def test_index_files_batch_embeddings(self, temp_git_repo: Path, data_dir: Path):
        """Test that index files correctly handles batch embeddings with Rust backend."""
        # Create multiple files to test batch processing
        for i in range(5):
            (temp_git_repo / f"file_{i}.py").write_text(
                f'''def function_{i}():
    """Function number {i}."""
    return {i}
'''
            )

        # Initialize
        init_result = run_te_command(
            ["init", "--path", str(temp_git_repo)],
            data_dir=data_dir,
        )
        assert init_result.returncode == 0

        # Index with non-incremental mode
        index_result = run_te_command(
            ["index", "files", str(temp_git_repo), "--no-incremental"],
            data_dir=data_dir,
        )

        assert index_result.returncode == 0, (
            f"Index files failed: {index_result.stderr}"
        )

        # Verify all 5 files were indexed
        stats_result = run_te_command(["stats"], data_dir=data_dir)
        output = stats_result.stdout + stats_result.stderr

        import re

        doc_count_match = re.search(r"Documents:\s*(\d+)", output)
        assert doc_count_match is not None, "Could not find document count in stats output"
        doc_count = int(doc_count_match.group(1))
        # Should have at least 5 documents (some may be chunked)
        assert doc_count >= _MIN_FILES_FOR_BATCH_TEST, f"Expected at least {_MIN_FILES_FOR_BATCH_TEST} documents, got {doc_count}"


class TestRustBackendIndexCommits:
    """Tests for index commits command with Rust backend."""

    @rust_available
    def test_index_commits_with_rust_backend(self, temp_git_repo: Path, data_dir: Path):
        """Test that index commits works with Rust backend."""
        # Make a commit
        (temp_git_repo / "feature.py").write_text(
            """def new_feature():
    '''A new feature.'''
    return "feature"
"""
        )
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Add new feature"],
            cwd=temp_git_repo,
            capture_output=True,
        )

        # Initialize
        init_result = run_te_command(
            ["init", "--path", str(temp_git_repo)],
            data_dir=data_dir,
        )
        assert init_result.returncode == 0

        # Index commits with Rust backend
        index_result = run_te_command(
            ["index", "commits", "--repo", str(temp_git_repo), "--limit", "10"],
            data_dir=data_dir,
        )

        assert index_result.returncode == 0, (
            f"Index commits failed: {index_result.stderr}\n"
            f"stdout: {index_result.stdout}"
        )

        # Verify output indicates successful indexing
        assert "error" not in index_result.stderr.lower(), (
            f"Index commits had errors: {index_result.stderr}"
        )

    @rust_available
    def test_index_commits_batch_embeddings(self, temp_git_repo: Path, data_dir: Path):
        """Test that index commits correctly handles batch embeddings with Rust backend."""
        # Make multiple commits
        for i in range(3):
            (temp_git_repo / f"commit_{i}.py").write_text(
                f'''def commit_feature_{i}():
    """Commit number {i}."""
    return {i}
'''
            )
            subprocess.run(
                ["git", "add", "."],
                cwd=temp_git_repo,
                capture_output=True,
            )
            subprocess.run(
                ["git", "commit", "-m", f"Commit {i}"],
                cwd=temp_git_repo,
                capture_output=True,
            )

        # Initialize
        init_result = run_te_command(
            ["init", "--path", str(temp_git_repo)],
            data_dir=data_dir,
        )
        assert init_result.returncode == 0

        # Index all commits (use --full to force non-incremental)
        index_result = run_te_command(
            [
                "index",
                "commits",
                "--repo",
                str(temp_git_repo),
                "--all",
                "--full",
            ],
            data_dir=data_dir,
        )

        assert index_result.returncode == 0, (
            f"Index commits failed: {index_result.stderr}"
        )

        # Verify commits were indexed (check that indexing ran without errors)
        assert "error" not in index_result.stderr.lower()


class TestRustBackendSearch:
    """Tests for search command with Rust backend."""

    @rust_available
    def test_search_with_rust_backend(self, temp_git_repo: Path, data_dir: Path):
        """Test that search works with Rust backend."""
        # Create a file with searchable content
        (temp_git_repo / "searchable.py").write_text(
            """def calculate_sum(numbers):
    '''Calculate the sum of a list of numbers.

    This function takes a list of numbers and returns their sum.
    It handles empty lists by returning 0.
    '''
    if not numbers:
        return 0
    total = 0
    for num in numbers:
        total += num
    return total
"""
        )

        # Initialize
        init_result = run_te_command(
            ["init", "--path", str(temp_git_repo)],
            data_dir=data_dir,
        )
        assert init_result.returncode == 0

        # Index the file
        index_result = run_te_command(
            ["index", "files", str(temp_git_repo), "--no-incremental"],
            data_dir=data_dir,
        )
        assert index_result.returncode == 0

        # Search using Rust backend
        search_result = run_te_command(
            ["search", "sum of numbers"],
            data_dir=data_dir,
        )

        assert search_result.returncode == 0, (
            f"Search failed: {search_result.stderr}\n"
            f"stdout: {search_result.stdout}"
        )

        # Verify search produced results
        output = search_result.stdout + search_result.stderr
        # Should find results or indicate no results (not crash)
        assert "error" not in output.lower() or "no results" in output.lower()

    @rust_available
    def test_search_returns_structured_results(self, temp_git_repo: Path, data_dir: Path):
        """Test that search returns structurally valid results with Rust backend."""
        # Create files with distinct content
        (temp_git_repo / "math.py").write_text(
            """def add(a, b):
    '''Add two numbers.'''
    return a + b

def multiply(a, b):
    '''Multiply two numbers.'''
    return a * b
"""
        )
        (temp_git_repo / "strings.py").write_text(
            """def reverse_string(s):
    '''Reverse a string.'''
    return s[::-1]

def capitalize(s):
    '''Capitalize a string.'''
    return s.capitalize()
"""
        )

        # Initialize and index
        init_result = run_te_command(
            ["init", "--path", str(temp_git_repo)],
            data_dir=data_dir,
        )
        assert init_result.returncode == 0

        index_result = run_te_command(
            ["index", "files", str(temp_git_repo), "--no-incremental"],
            data_dir=data_dir,
        )
        assert index_result.returncode == 0

        # Search for math-related content
        search_result = run_te_command(
            ["search", "multiply numbers", "--top-k", "3"],
            data_dir=data_dir,
        )

        assert search_result.returncode == 0

        # Verify results are structurally valid
        output = search_result.stdout + search_result.stderr

        # Check for Score: pattern which indicates structured results
        import re

        score_matches = re.findall(r"Score:\s*[\d.]+", output)
        # Should have some scores in output
        assert len(score_matches) > 0, (
            f"Expected score results, got: {output}"
        )


class TestRustBackendErrorHandling:
    """Tests for error handling with Rust backend."""

    @rust_available
    def test_index_files_handles_empty_directory(
        self, temp_git_repo: Path, data_dir: Path
    ):
        """Test that index files handles empty directory gracefully."""
        # Create empty directory
        empty_dir = temp_git_repo / "empty"
        empty_dir.mkdir()

        # Initialize
        init_result = run_te_command(
            ["init", "--path", str(temp_git_repo)],
            data_dir=data_dir,
        )
        assert init_result.returncode == 0

        # Index empty directory - should not crash
        index_result = run_te_command(
            ["index", "files", str(empty_dir)],
            data_dir=data_dir,
        )

        # Should complete without crashing (may warn about no files)
        assert index_result.returncode == 0, (
            f"Index files on empty dir failed: {index_result.stderr}"
        )

    @rust_available
    def test_search_on_empty_index(self, temp_git_repo: Path, data_dir: Path):
        """Test that search on empty index returns appropriate message."""
        # Initialize without indexing
        init_result = run_te_command(
            ["init", "--path", str(temp_git_repo)],
            data_dir=data_dir,
        )
        assert init_result.returncode == 0

        # Search on empty index
        search_result = run_te_command(
            ["search", "test query"],
            data_dir=data_dir,
        )

        # Should complete without crashing
        assert search_result.returncode == 0

        # Should indicate no results
        output = search_result.stdout + search_result.stderr
        assert "no results" in output.lower() or "0" in output
