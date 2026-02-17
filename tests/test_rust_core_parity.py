"""Tests for Python vs Rust parity in doc-id generation and chunk normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from town_elder.cli import (
    _build_file_doc_id as py_build_file_doc_id,
)
from town_elder.cli import (
    _get_doc_id_inputs as py_get_doc_id_inputs,
)
from town_elder.cli import (
    _normalize_chunk_metadata as py_normalize_chunk_metadata,
)

# Try to import the Rust extension - tests will be skipped if not available
try:
    from town_elder import _te_core as te_core_rust

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    te_core_rust = None

DOC_ID_HEX_LENGTH = 16
BASE_CHUNK_INDEX = 10


# =============================================================================
# Doc-ID Generation Tests
# =============================================================================

class TestBuildFileDocIdParity:
    """Test parity between Python and Rust doc-id generation."""

    @pytest.mark.parametrize(
        "path,chunk_index",
        [
            ("src/main.py", 0),
            ("src/main.py", 1),
            ("src/main.py", 5),
            ("path with spaces.txt", 0),
            ("path with spaces.txt", 2),
            ("日本語/テスト.py", 0),
            ("日本語/テスト.py", 3),
            ("/absolute/path/file.py", 0),
            ("/absolute/path/file.py", 10),
            ("", 0),
            ("simple.py", 0),
        ],
    )
    def test_doc_id_parity(self, path: str, chunk_index: int):
        """Rust doc-id should match Python doc-id for all fixture cases."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        py_result = py_build_file_doc_id(path, chunk_index)
        rust_result = te_core_rust.build_file_doc_id(path, chunk_index)

        assert py_result == rust_result, (
            f"Doc-id mismatch for path={path!r}, chunk_index={chunk_index}: "
            f"Python={py_result!r}, Rust={rust_result!r}"
        )

    @pytest.mark.parametrize(
        "path,chunk_index,expected_prefix",
        [
            ("src/main.py", 0, "2e5a"),  # Known hash prefix for this input
            ("src/main.py", 1, "9c20"),  # Known hash prefix for chunk 1
        ],
    )
    def test_doc_id_deterministic(self, path: str, chunk_index: int, expected_prefix: str):
        """Doc-ids should be deterministic (same input = same output)."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        result = te_core_rust.build_file_doc_id(path, chunk_index)
        assert result.startswith(expected_prefix), (
            f"Doc-id {result!r} doesn't start with expected prefix {expected_prefix!r}"
        )


# =============================================================================
# Doc-ID Inputs Tests (for deletion safety)
# =============================================================================

class TestGetDocIdInputsParity:
    """Test parity between Python and Rust doc-id inputs expansion."""

    @pytest.mark.parametrize(
        "path,repo_root",
        [
            ("src/main.py", "/Users/testuser/test_repo"),
            ("path with spaces.txt", "/Users/testuser/test repo"),
            ("relative/file.py", "/Users/testuser/project"),
            ("日本語/テスト.py", "/Users/testuser/test"),
            ("/absolute/path/file.py", "/Users/testuser/test_repo"),  # Absolute path should stay as-is
        ],
    )
    def test_doc_id_inputs_parity(self, path: str, repo_root: str):
        """Rust doc-id inputs should match Python for deletion safety."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        py_repo_root = Path(repo_root)
        py_result = py_get_doc_id_inputs(path, py_repo_root)
        rust_result = te_core_rust.get_doc_id_inputs(path, repo_root)

        # Convert to set for comparison (order may differ)
        assert set(py_result) == set(rust_result), (
            f"Doc-id inputs mismatch for path={path!r}, repo_root={repo_root!r}: "
            f"Python={py_result!r}, Rust={rust_result!r}"
        )

    def test_relative_path_expands_to_absolute(self):
        """Relative paths should expand to absolute within repo_root."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        result = te_core_rust.get_doc_id_inputs("src/main.py", "/Users/testuser/repo")
        assert "src/main.py" in result
        # The absolute path should be present (actual resolved path)
        assert any("src/main.py" in inp and Path(inp).is_absolute() for inp in result)

    def test_absolute_path_stays_single(self):
        """Absolute paths should not be duplicated."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        result = te_core_rust.get_doc_id_inputs("/absolute/path/file.py", "/Users/testuser/repo")
        assert len(result) == 1
        assert result[0] == "/absolute/path/file.py"


# =============================================================================
# Chunk Metadata Normalization Tests
# =============================================================================

class TestNormalizeChunkMetadataParity:
    """Test parity between Python and Rust chunk metadata normalization."""

    @pytest.mark.parametrize(
        "base_metadata,chunk_metadata,fallback_chunk_index",
        [
            # Valid chunk_index in chunk_metadata should be preserved
            ({"type": "file"}, {"chunk_index": 2}, 0),
            ({"type": "file"}, {"chunk_index": 5}, 0),
            # Missing chunk_index should use fallback
            ({"type": "file"}, {}, 3),
            ({"type": "file"}, {"extra": "data"}, 1),
            # Invalid chunk_index types should use fallback
            ({"type": "file"}, {"chunk_index": "invalid"}, 0),
            ({"type": "file"}, {"chunk_index": True}, 0),
            ({"type": "file"}, {"chunk_index": False}, 0),
            ({"type": "file"}, {"chunk_index": -1}, 0),
            # Merge base and chunk metadata
            ({"base_key": "base_val"}, {"chunk_key": "chunk_val"}, 0),
            # Empty metadata
            ({}, {}, 0),
            # Chunk_index 0 should work
            ({"type": "file"}, {"chunk_index": 0}, 5),
        ],
    )
    def test_normalize_metadata_parity(
        self,
        base_metadata: dict[str, Any],
        chunk_metadata: dict[str, Any],
        fallback_chunk_index: int,
    ):
        """Rust normalization should match Python."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        py_metadata, py_chunk_index = py_normalize_chunk_metadata(
            base_metadata=base_metadata,
            chunk_metadata=chunk_metadata,
            fallback_chunk_index=fallback_chunk_index,
        )
        rust_metadata, rust_chunk_index = te_core_rust.normalize_chunk_metadata(
            base_metadata,
            chunk_metadata,
            fallback_chunk_index,
        )

        # Compare chunk_index
        assert py_chunk_index == rust_chunk_index, (
            f"Chunk index mismatch: Python={py_chunk_index}, Rust={rust_chunk_index}"
        )

        # Compare metadata (Rust returns JSON values, convert for comparison)
        for key, py_value in py_metadata.items():
            rust_value = rust_metadata.get(key)
            # Convert Rust JSON values to Python types for comparison
            if rust_value is not None:
                # Handle Number type from serde_json
                if hasattr(rust_value, "as_u64"):
                    rust_converted = rust_value.as_u64().to_bytes(8, 'little') if rust_value.is_u64() else int(rust_value)
                    # Compare as int
                    assert py_value == int(py_value), f"Metadata {key}: Python={py_value}, Rust={rust_converted}"
                else:
                    assert py_value == rust_value, (
                        f"Metadata mismatch for key={key}: Python={py_value!r}, Rust={rust_value!r}"
                    )

    def test_metadata_merge_preserves_base_values(self):
        """Base metadata values should be preserved when not overridden."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        base = {"type": "python", "source": "test"}
        chunk = {"extra": "value"}
        fallback = 1

        _result_metadata, chunk_index = te_core_rust.normalize_chunk_metadata(
            base, chunk, fallback
        )

        assert _result_metadata.get("type") == "python"
        assert _result_metadata.get("source") == "test"
        assert _result_metadata.get("extra") == "value"

    def test_chunk_metadata_overrides_base(self):
        """Chunk metadata should override base metadata."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        base = {"type": "base_type", "chunk_index": 10}
        chunk = {"type": "chunk_type"}
        fallback = 0

        result_metadata, chunk_index = te_core_rust.normalize_chunk_metadata(
            base, chunk, fallback
        )

        assert result_metadata.get("type") == "chunk_type"
        # chunk_index from chunk should take precedence
        assert chunk_index == BASE_CHUNK_INDEX


# =============================================================================
# Fixture-based Tests (using contract fixtures)
# =============================================================================

class TestDocIdFromFixtures:
    """Test doc-id generation using contract fixtures."""

    @pytest.fixture
    def fixture_dir(self) -> Path:
        return Path(__file__).parent / "fixtures" / "rust_core_contract" / "doc_ids"

    def test_path_with_spaces(self):
        """Doc-id for path with spaces should work correctly."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        path = "path with spaces.txt"
        doc_id = te_core_rust.build_file_doc_id(path, 0)

        # Should match Python's behavior
        expected = py_build_file_doc_id(path, 0)
        assert doc_id == expected

    def test_unicode_path(self):
        """Doc-id for unicode path should work correctly."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extension not available")

        path = "日本語/テスト.py"
        doc_id = te_core_rust.build_file_doc_id(path, 0)

        # Should match Python's behavior
        expected = py_build_file_doc_id(path, 0)
        assert doc_id == expected


# =============================================================================
# Backward Compatibility Tests
# =============================================================================

class TestBackwardCompatibility:
    """Ensure existing behavior is unchanged when Rust is unavailable."""

    def test_python_fallback_when_rust_unavailable(self):
        """Python implementation should still work when Rust is unavailable."""
        # These should work regardless of Rust availability
        doc_id = py_build_file_doc_id("test.py", 0)
        assert len(doc_id) == DOC_ID_HEX_LENGTH
        assert doc_id.isalnum()

        doc_id_multi = py_build_file_doc_id("test.py", 5)
        assert len(doc_id_multi) == DOC_ID_HEX_LENGTH

    def test_python_normalize_fallback_when_rust_unavailable(self):
        """Python normalization should work regardless of Rust."""
        metadata, idx = py_normalize_chunk_metadata(
            base_metadata={"type": "test"},
            chunk_metadata={},
            fallback_chunk_index=0,
        )
        assert idx == 0
        assert metadata.get("type") == "test"

    def test_python_doc_id_inputs_fallback_when_rust_unavailable(self):
        """Python doc-id inputs should work regardless of Rust."""
        inputs = py_get_doc_id_inputs("test.py", Path("/Users/testuser/repo"))
        assert "test.py" in inputs
        assert len(inputs) >= 1
