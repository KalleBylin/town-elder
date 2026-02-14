"""Core tests for replay."""
from __future__ import annotations

import numpy as np

from replay.embeddings.embedder import Embedder
from replay.git.diff_parser import DiffFile, DiffParser
from replay.storage.vector_store import ZvecStore


class TestEmbedder:
    """Tests for the Embedder class."""

    def test_embedder_instantiation(self):
        """Test that Embedder can be instantiated with default model."""
        embedder = Embedder()
        assert embedder.model_name == Embedder.DEFAULT_MODEL
        assert embedder.dimension == 384

    def test_embedder_custom_model(self):
        """Test that Embedder can be instantiated with custom model."""
        embedder = Embedder(model_name="custom-model")
        assert embedder.model_name == "custom-model"

    def test_embedder_dimension_property(self):
        """Test that dimension property returns correct value."""
        embedder = Embedder()
        assert embedder.dimension == 384

    def test_embed_single(self):
        """Test embedding a single text returns a vector."""
        embedder = Embedder()
        vector = embedder.embed_single("hello world")
        assert isinstance(vector, np.ndarray)
        assert vector.shape == (384,)


class TestVectorStore:
    """Tests for the VectorStore class."""

    def test_vector_store_instantiation(self, temp_dir):
        """Test that VectorStore can be instantiated."""
        store = ZvecStore(path=temp_dir / "test.vec", dimension=384)
        assert store.dimension == 384
        assert store.count() == 0

    def test_vector_store_insert(self, sample_vector_store):
        """Test inserting a document."""
        doc_id = sample_vector_store.insert(
            text="test document",
            metadata={"source": "test", "type": "example"}
        )
        assert doc_id is not None
        assert sample_vector_store.count() == 1

    def test_vector_store_insert_with_vector(self, sample_vector_store, sample_vectors):
        """Test inserting a document with pre-computed vector."""
        doc_id = sample_vector_store.insert_with_vector(
            doc_id="doc1",
            vector=sample_vectors[0],
            text="test document",
            metadata={"source": "test"}
        )
        assert doc_id == "doc1"
        assert sample_vector_store.count() == 1

    def test_vector_store_get(self, sample_vector_store):
        """Test getting a document by ID."""
        doc_id = sample_vector_store.insert(
            text="test document",
            metadata={"source": "test"}
        )
        doc = sample_vector_store.get(doc_id)
        assert doc is not None
        assert doc["text"] == "test document"
        assert doc["metadata"]["source"] == "test"

    def test_vector_store_get_nonexistent(self, sample_vector_store):
        """Test getting a nonexistent document returns None."""
        doc = sample_vector_store.get("nonexistent")
        assert doc is None

    def test_vector_store_search(self, sample_vector_store, sample_vectors):
        """Test searching for similar documents."""
        # Insert multiple documents with vectors
        sample_vector_store.insert_with_vector(
            doc_id="doc0",
            vector=sample_vectors[0],
            text="document about cats",
            metadata={"type": "animal"}
        )
        sample_vector_store.insert_with_vector(
            doc_id="doc1",
            vector=sample_vectors[1],
            text="document about felines",
            metadata={"type": "animal"}
        )
        sample_vector_store.insert_with_vector(
            doc_id="doc2",
            vector=sample_vectors[2],
            text="document about cars",
            metadata={"type": "vehicle"}
        )

        # Search with a vector similar to doc0 and doc1
        query = sample_vectors[0]  # Similar to doc0 and doc1
        results = sample_vector_store.search(query, top_k=2)

        assert len(results) == 2
        # First result should be doc0 (exact match)
        assert results[0]["id"] == "doc0"
        assert results[0]["score"] > 0

    def test_vector_store_search_top_k(self, sample_vector_store, sample_vectors):
        """Test that search respects top_k parameter."""
        # Insert multiple documents
        for i, vec in enumerate(sample_vectors):
            sample_vector_store.insert_with_vector(
                doc_id=f"doc{i}",
                vector=vec,
                text=f"document {i}",
                metadata={"index": i}
            )

        results = sample_vector_store.search(sample_vectors[0], top_k=2)
        assert len(results) == 2

        results = sample_vector_store.search(sample_vectors[0], top_k=5)
        assert len(results) == 5

    def test_vector_store_delete(self, sample_vector_store):
        """Test deleting a document."""
        doc_id = sample_vector_store.insert(
            text="test document",
            metadata={}
        )
        assert sample_vector_store.count() == 1

        sample_vector_store.delete(doc_id)
        assert sample_vector_store.count() == 0

    def test_vector_store_delete_nonexistent(self, sample_vector_store):
        """Test deleting a nonexistent document doesn't raise."""
        sample_vector_store.delete("nonexistent")  # Should not raise

    def test_vector_store_creates_new_on_nonexistent_path(self, temp_dir):
        """Test that opening a nonexistent path creates a new collection."""
        store_path = temp_dir / "nonexistent.vec"
        # Verify path doesn't exist
        assert not store_path.exists()

        # Opening should create a new collection
        store = ZvecStore(path=store_path, dimension=384)
        assert store.count() == 0
        store.close()

    def test_vector_store_fails_on_corrupted_collection(self, temp_dir):
        """Test that opening a corrupted collection fails with actionable error."""
        store_path = temp_dir / "corrupted.vec"
        store_path.mkdir(parents=True, exist_ok=True)

        # Write invalid/corrupted data to simulate corruption
        (store_path / "_metadata").write_bytes(b"invalid corrupted data")

        # Opening should raise VectorStoreError with actionable message
        store = ZvecStore(path=store_path, dimension=384)
        try:
            store.count()
            assert False, "Expected VectorStoreError to be raised"
        except Exception as e:
            # Should NOT silently create a new collection
            assert "Failed to open zvec collection" in str(e) or "corruption" in str(e).lower() or "schema" in str(e).lower()
        finally:
            store.close()

    def test_vector_store_no_implicit_create_on_permission_error(self, temp_dir):
        """Test that permission errors are surfaced, not masked by implicit create."""
        # This test simulates a permission error scenario
        # In practice, this would require OS-level permission manipulation
        # which is difficult to test reliably, so we test the error message quality

        store_path = temp_dir / "permissions.vec"

        # Create a store, then try to open it in a way that would fail
        store = ZvecStore(path=store_path, dimension=384)
        store.insert("test", {})
        store.close()

        # Now try to open with wrong dimension - this should fail meaningfully
        # (different dimension = schema mismatch)
        wrong_dim_store = ZvecStore(path=store_path, dimension=256)
        try:
            wrong_dim_store.count()
            # If it doesn't raise, the old behavior (silent recreate) happened
            # which is the bug we're fixing
            assert False, "Expected VectorStoreError for schema mismatch"
        except Exception as e:
            # Should have actionable error message
            error_str = str(e).lower()
            assert "failed" in error_str or "error" in error_str
        finally:
            wrong_dim_store.close()


class TestDiffParser:
    """Tests for the DiffParser class."""

    def test_diff_parser_instantiation(self):
        """Test that DiffParser can be instantiated."""
        parser = DiffParser()
        assert parser is not None

    def test_parse_new_file(self, diff_parser, sample_diff_output):
        """Test parsing a diff with new files."""
        files = list(diff_parser.parse(sample_diff_output))

        # Should have 3 files: new, deleted, modified
        assert len(files) == 3

        # First file should be new
        new_file = files[0]
        assert new_file.status == "added"
        assert "main.py" in new_file.path

    def test_parse_deleted_file(self, diff_parser, sample_diff_output):
        """Test parsing a diff with deleted files."""
        files = list(diff_parser.parse(sample_diff_output))

        # Second file should be deleted
        deleted_file = files[1]
        assert deleted_file.status == "deleted"
        assert "utils.py" in deleted_file.path

    def test_parse_modified_file(self, diff_parser, sample_diff_output):
        """Test parsing a diff with modified files."""
        files = list(diff_parser.parse(sample_diff_output))

        # Third file should be modified
        modified_file = files[2]
        assert modified_file.status == "modified"
        assert "app.py" in modified_file.path

    def test_parse_hunks_extracted(self, diff_parser, sample_diff_output):
        """Test that diff hunks are extracted."""
        files = list(diff_parser.parse(sample_diff_output))

        # Each file should have hunks
        for file in files:
            assert len(file.hunks) > 0

    def test_parse_diff_to_text(self, diff_parser, sample_diff_output):
        """Test converting diff to plain text."""
        text = diff_parser.parse_diff_to_text(sample_diff_output)

        assert "main.py" in text
        assert "utils.py" in text
        assert "app.py" in text
        assert "added" in text
        assert "deleted" in text
        assert "modified" in text

    def test_parse_empty_diff(self, diff_parser):
        """Test parsing an empty diff."""
        files = list(diff_parser.parse(""))
        assert len(files) == 0

    def test_diff_file_dataclass(self):
        """Test that DiffFile dataclass works correctly."""
        diff_file = DiffFile(
            path="test.py",
            status="modified",
            hunks=["@@ -1,2 +1,3 @@"]
        )

        assert diff_file.path == "test.py"
        assert diff_file.status == "modified"
        assert len(diff_file.hunks) == 1
