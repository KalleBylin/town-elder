"""Tests for the file scanner module."""

import pytest

from town_elder import rust_adapter
from town_elder.indexing.file_scanner import (
    DEFAULT_EXCLUDES,
    DEFAULT_EXTENSIONS,
    scan_files,
)

try:
    import town_elder._te_core  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


class TestDefaultConfiguration:
    """Tests for default configuration values."""

    def test_default_extensions_includes_rst(self):
        """Default extensions should include .rst."""
        assert ".rst" in DEFAULT_EXTENSIONS
        assert ".py" in DEFAULT_EXTENSIONS
        assert ".md" in DEFAULT_EXTENSIONS

    def test_default_excludes_includes_build(self):
        """Default excludes should include _build."""
        assert "_build" in DEFAULT_EXCLUDES

    def test_default_excludes_includes_git(self):
        """Default excludes should include .git and other standard patterns."""
        assert ".git" in DEFAULT_EXCLUDES
        assert ".venv" in DEFAULT_EXCLUDES
        assert "node_modules" in DEFAULT_EXCLUDES


class TestRstInclusion:
    """Tests for .rst file inclusion."""

    def test_scanner_finds_rst_files(self, tmp_path):
        """Scanner should find .rst files by default."""
        # Create test files
        (tmp_path / "test.rst").write_text("Test RST content")
        (tmp_path / "README.rst").write_text("README RST")

        files = scan_files(tmp_path)

        rst_files = [f for f in files if f.suffix == ".rst"]
        assert len(rst_files) == 2  # noqa: PLR2004

    def test_scanner_includes_rst_with_py_and_md(self, tmp_path):
        """Scanner should find .rst, .py, and .md files together."""
        # Create test files
        (tmp_path / "test.py").write_text("print('hello')")
        (tmp_path / "readme.md").write_text("# Readme")
        (tmp_path / "doc.rst").write_text("Doc content")

        files = scan_files(tmp_path)

        assert len(files) == 3  # noqa: PLR2004
        suffixes = {f.suffix for f in files}
        assert suffixes == {".py", ".md", ".rst"}


class TestBuildExclusion:
    """Tests for _build directory exclusion."""

    def test_scanner_excludes_build_directory(self, tmp_path):
        """Scanner should exclude _build directories."""
        # Create test files
        (tmp_path / "test.py").write_text("print('hello')")
        (tmp_path / "_build").mkdir()
        (tmp_path / "_build" / "output.py").write_text("generated")

        files = scan_files(tmp_path)

        # Should only find test.py, not anything in _build
        assert len(files) == 1
        assert files[0].name == "test.py"

    def test_scanner_excludes_nested_build(self, tmp_path):
        """Scanner should exclude _build at any nesting level."""
        # Create nested _build
        (tmp_path / "src" / "_build" / "gen.py").parent.mkdir(parents=True)
        (tmp_path / "src" / "_build" / "gen.py").write_text("generated")
        (tmp_path / "src" / "main.py").write_text("main")

        files = scan_files(tmp_path)

        # Should only find main.py
        assert len(files) == 1
        assert files[0].name == "main.py"

    def test_scanner_excludes_sphinx_build(self, tmp_path):
        """Scanner should exclude typical Sphinx build output."""
        # Create typical Sphinx build structure
        docs_build_html = tmp_path / "docs" / "_build" / "html"
        docs_build_html.mkdir(parents=True)
        (docs_build_html / "index.html").write_text("<html>")
        (tmp_path / "docs" / "index.rst").write_text("Docs")

        files = scan_files(tmp_path)

        # Should only find the source index.rst, not the generated HTML
        assert len(files) == 1
        assert files[0].name == "index.rst"


class TestUserExcludes:
    """Tests for user-provided exclusion patterns."""

    def test_user_excludes_are_additive(self, tmp_path):
        """User-provided excludes should be added to defaults."""
        # Create test files
        (tmp_path / "test.py").write_text("print('hello')")
        (tmp_path / "custom").mkdir()
        (tmp_path / "custom" / "skip.py").write_text("skip me")

        # Exclude 'custom' directory
        files = scan_files(tmp_path, exclude_patterns=frozenset({"custom"}))

        assert len(files) == 1
        assert files[0].name == "test.py"

    def test_user_exclude_overrides_default_behavior(self, tmp_path):
        """User excludes should work alongside defaults."""
        # Even though .git is excluded by default, add another pattern
        (tmp_path / ".git" / "config").parent.mkdir(parents=True)
        (tmp_path / ".git" / "config").write_text("[core]")
        (tmp_path / "main.py").write_text("print('hi')")

        files = scan_files(tmp_path, exclude_patterns=frozenset({"custom"}))

        # Should still find main.py (and .git is excluded by default)
        assert len(files) == 1
        assert files[0].name == "main.py"

    def test_multiple_user_excludes(self, tmp_path):
        """Multiple user excludes should all be applied."""
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")

        skip1_dir = tmp_path / "skip1"
        skip1_dir.mkdir()
        (skip1_dir / "c.py").write_text("c")

        skip2_dir = tmp_path / "skip2"
        skip2_dir.mkdir()
        (skip2_dir / "d.py").write_text("d")

        files = scan_files(
            tmp_path,
            exclude_patterns=frozenset({"skip1", "skip2"})
        )

        assert len(files) == 2  # noqa: PLR2004
        names = {f.name for f in files}
        assert names == {"a.py", "b.py"}


class TestDeterministicOrdering:
    """Tests for deterministic file ordering."""

    def test_scanner_returns_sorted_files(self, tmp_path):
        """Scanner should return files in deterministic order."""
        # Create files in non-alphabetical order
        (tmp_path / "z_file.py").write_text("z")
        (tmp_path / "a_file.py").write_text("a")
        (tmp_path / "m_file.py").write_text("m")

        files = scan_files(tmp_path)

        # Should be sorted alphabetically
        names = [f.name for f in files]
        assert names == ["a_file.py", "m_file.py", "z_file.py"]

    def test_scanner_order_consistent_across_runs(self, tmp_path):
        """Scanner should return the same order on repeated calls."""
        (tmp_path / "file1.py").write_text("1")
        (tmp_path / "file2.py").write_text("2")
        (tmp_path / "file3.py").write_text("3")

        # Call scan_files multiple times
        result1 = scan_files(tmp_path)
        result2 = scan_files(tmp_path)
        result3 = scan_files(tmp_path)

        # All results should be identical
        assert [str(f) for f in result1] == [str(f) for f in result2]
        assert [str(f) for f in result2] == [str(f) for f in result3]

    def test_scanner_handles_mixed_extensions_sorted(self, tmp_path):
        """Scanner should sort files across all extensions."""
        # Create files with different extensions in mixed order
        (tmp_path / "z.rst").write_text("z")
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "m.md").write_text("m")

        files = scan_files(tmp_path)

        # Should be sorted by full path
        names = [f.name for f in files]
        assert names == ["a.py", "m.md", "z.rst"]


class TestPathValidation:
    """Tests for path handling."""

    def test_scanner_handles_empty_directory(self, tmp_path):
        """Scanner should handle empty directories gracefully."""
        files = scan_files(tmp_path)
        assert files == []

    def test_scanner_handles_only_excluded_files(self, tmp_path):
        """Scanner should return empty list when all files are excluded."""
        (tmp_path / "_build" / "gen.py").parent.mkdir(parents=True)
        (tmp_path / "_build" / "gen.py").write_text("generated")

        files = scan_files(tmp_path)
        assert files == []

    def test_scanner_finds_files_in_subdirectories(self, tmp_path):
        """Scanner should find files in nested subdirectories."""
        (tmp_path / "src" / "pkg" / "module.py").parent.mkdir(parents=True)
        (tmp_path / "src" / "pkg" / "module.py").write_text("code")

        files = scan_files(tmp_path)

        assert len(files) == 1
        assert "src/pkg/module.py" in str(files[0])


class TestRustScannerParity:
    """Parity tests for Rust-backed scanner through the adapter boundary."""

    def test_rust_scanner_matches_python_results(self, tmp_path, monkeypatch):
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.md").write_text("b")
        (tmp_path / "_build" / "generated.py").parent.mkdir(parents=True)
        (tmp_path / "_build" / "generated.py").write_text("generated")
        (tmp_path / "custom" / "skip.py").parent.mkdir(parents=True)
        (tmp_path / "custom" / "skip.py").write_text("skip")

        monkeypatch.setenv(rust_adapter._ENV_FLAG, "1")
        rust_adapter._reset_module_cache()

        py_files = scan_files(tmp_path, exclude_patterns=frozenset({"custom"}))
        rust_files = rust_adapter.scan_files(
            tmp_path,
            exclude_patterns=frozenset({"custom"}),
        )

        assert [str(path) for path in py_files] == [str(path) for path in rust_files]

    def test_rust_scanner_deterministic_ordering(self, tmp_path, monkeypatch):
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        (tmp_path / "z.py").write_text("z")
        (tmp_path / "m.py").write_text("m")
        (tmp_path / "a.py").write_text("a")

        monkeypatch.setenv(rust_adapter._ENV_FLAG, "1")
        rust_adapter._reset_module_cache()

        result1 = rust_adapter.scan_files(tmp_path)
        result2 = rust_adapter.scan_files(tmp_path)

        assert [str(path) for path in result1] == [str(path) for path in result2]
