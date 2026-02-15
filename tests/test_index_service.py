"""Tests for IndexService."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from town_elder.exceptions import IndexingError
from town_elder.services.index_service import IndexService


class TestIndexService:
    """Tests for the IndexService class."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock store."""
        store = MagicMock()
        store.insert = MagicMock(return_value="doc_id")
        return store

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder."""
        embedder = MagicMock()
        embedder.embed_single = MagicMock(return_value=[0.1] * 384)
        return embedder

    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create a mock config that uses temp_dir."""
        config = MagicMock()
        config.data_dir = temp_dir
        return config

    def test_index_file_success(self, mock_store, mock_embedder, temp_dir, mock_config):
        """Test successful file indexing."""
        # Create a test file
        test_file = temp_dir / "test.py"
        test_file.write_text("print('hello')")

        with patch("town_elder.services.index_service.get_config", return_value=mock_config):
            service = IndexService(store=mock_store, embedder=mock_embedder)
            doc_id = service.index_file(test_file)

        assert doc_id == str(test_file)
        mock_store.insert.assert_called_once()
        mock_embedder.embed_single.assert_called_once_with("print('hello')")

    def test_index_file_read_error(self, mock_store, mock_embedder, temp_dir, mock_config):
        """Test that file read errors raise IndexingError."""
        # Create a path that will fail to read
        test_file = temp_dir / "nonexistent.py"

        with patch("town_elder.services.index_service.get_config", return_value=mock_config):
            service = IndexService(store=mock_store, embedder=mock_embedder)

        with pytest.raises(IndexingError) as exc_info:
            service.index_file(test_file)

        assert "Failed to read file" in str(exc_info.value)
        assert str(test_file) in str(exc_info.value)

    def test_index_file_read_error_contains_cause(self, mock_store, mock_embedder, temp_dir, mock_config):
        """Test that IndexingError contains the original exception as cause."""
        test_file = temp_dir / "nonexistent.py"

        with patch("town_elder.services.index_service.get_config", return_value=mock_config):
            service = IndexService(store=mock_store, embedder=mock_embedder)

        with pytest.raises(IndexingError) as exc_info:
            service.index_file(test_file)

        assert exc_info.value.__cause__ is not None

    def test_index_directory_success(self, mock_store, mock_embedder, temp_dir, mock_config):
        """Test successful directory indexing."""
        # Create test files
        (temp_dir / "test1.py").write_text("print('hello')")
        (temp_dir / "test2.md").write_text("# Hello")

        with patch("town_elder.services.index_service.get_config", return_value=mock_config):
            service = IndexService(store=mock_store, embedder=mock_embedder)
            count = service.index_directory(temp_dir)

        expected_count = 2
        assert count == expected_count

    def test_index_directory_raises_on_error(self, mock_store, mock_embedder, temp_dir, mock_config):
        """Test that index_directory raises IndexingError when indexing fails."""
        # Create test files
        (temp_dir / "test1.py").write_text("print('hello')")
        (temp_dir / "test2.py").write_text("print('world')")

        with patch("town_elder.services.index_service.get_config", return_value=mock_config):
            service = IndexService(store=mock_store, embedder=mock_embedder)

        # Mock index_file to raise on the second call
        original_index_file = service.index_file
        call_count = [0]
        second_call = 2

        def mock_index_file(path):
            call_count[0] += 1
            if call_count[0] == second_call:
                raise IndexingError("test error")
            return original_index_file(path)

        with patch.object(service, "index_file", side_effect=mock_index_file):
            with pytest.raises(IndexingError) as exc_info:
                service.index_directory(temp_dir)

            assert "Failed to index" in str(exc_info.value)

    def test_index_directory_collects_all_errors(self, mock_store, mock_embedder, temp_dir, mock_config):
        """Test that index_directory collects and reports all indexing errors."""
        # Create test files
        (temp_dir / "test1.py").write_text("print('hello')")
        (temp_dir / "test2.py").write_text("print('world')")

        with patch("town_elder.services.index_service.get_config", return_value=mock_config):
            service = IndexService(store=mock_store, embedder=mock_embedder)

        call_count = [0]

        def mock_index_file(path):
            call_count[0] += 1
            if call_count[0] == 1:
                raise IndexingError("error 1")
            raise IndexingError("error 2")

        with patch.object(service, "index_file", side_effect=mock_index_file):
            with pytest.raises(IndexingError) as exc_info:
                service.index_directory(temp_dir)

            assert "Failed to index 2 file(s)" in str(exc_info.value)
            assert "error 1" in str(exc_info.value)
            assert "error 2" in str(exc_info.value)
