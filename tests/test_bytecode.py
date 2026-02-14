"""Tests to ensure no bytecode files are tracked in git."""
import subprocess


def test_no_bytecode_files_tracked():
    """Ensure no __pycache__ or .pyc files are tracked in git."""
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
    )
    tracked = [
        line for line in result.stdout.strip().split("\n")
        if ".pyc" in line or "__pycache__" in line
    ]
    assert len(tracked) == 0, f"Tracked bytecode files found: {tracked}"
