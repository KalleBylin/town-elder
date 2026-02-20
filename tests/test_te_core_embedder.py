"""Smoke tests for the PyTextEmbedder extension bindings."""

import pytest

from town_elder import _te_core

# Constants for expected dimensions
DIM_SMALL = 384
DIM_BASE = 768
DIM_LARGE = 1024
NUM_MODELS = 3
NUM_TEST_TEXTS = 3


class TestPyTextEmbedder:
    """Test suite for PyTextEmbedder Python bindings."""

    def test_list_supported_models(self):
        """Test that supported models are listed correctly."""
        models = _te_core.PyTextEmbedder.list_supported_models()
        assert len(models) == NUM_MODELS
        # Check model format - should return BAAI/* naming for Python config compatibility
        for model_name, dimension in models:
            assert isinstance(model_name, str)
            assert isinstance(dimension, int)
            assert dimension > 0
            # Verify BAAI/* naming is returned
            assert model_name.startswith("BAAI/")

    def test_list_supported_models_has_expected_models(self):
        """Test that list_supported_models returns the expected BAAI models."""
        models = _te_core.PyTextEmbedder.list_supported_models()
        model_names = [m[0] for m in models]
        assert "BAAI/bge-small-en-v1.5" in model_names
        assert "BAAI/bge-base-en-v1.5" in model_names
        assert "BAAI/bge-large-en-v1.5" in model_names

    def test_create_embedder_default_model(self):
        """Test creating embedder with default model."""
        embedder = _te_core.PyTextEmbedder("Xenova/bge-small-en-v1.5")
        assert embedder.get_model_name() == "Xenova/bge-small-en-v1.5"
        assert embedder.dimension() == DIM_SMALL

    def test_create_embedder_with_baai_model(self):
        """Test creating embedder with BAAI/* model identifier (Python config default)."""
        embedder = _te_core.PyTextEmbedder("BAAI/bge-small-en-v1.5")
        # The model name should be preserved as provided
        assert embedder.get_model_name() == "BAAI/bge-small-en-v1.5"
        assert embedder.dimension() == DIM_SMALL

    def test_create_embedder_with_xenova_model(self):
        """Test creating embedder with Xenova/* model identifier."""
        embedder = _te_core.PyTextEmbedder("Xenova/bge-small-en-v1.5")
        assert embedder.get_model_name() == "Xenova/bge-small-en-v1.5"
        assert embedder.dimension() == DIM_SMALL

    def test_embedder_with_baai_different_sizes(self):
        """Test creating embedders with different BAAI/* model sizes."""
        # Small model
        embedder_small = _te_core.PyTextEmbedder("BAAI/bge-small-en-v1.5")
        assert embedder_small.dimension() == DIM_SMALL

        # Base model
        embedder_base = _te_core.PyTextEmbedder("BAAI/bge-base-en-v1.5")
        assert embedder_base.dimension() == DIM_BASE

        # Large model
        embedder_large = _te_core.PyTextEmbedder("BAAI/bge-large-en-v1.5")
        assert embedder_large.dimension() == DIM_LARGE

    def test_embed_single_returns_vector(self):
        """Test that embed_single returns a vector of correct dimension."""
        embedder = _te_core.PyTextEmbedder("Xenova/bge-small-en-v1.5")
        vec = embedder.embed_single("Hello, world!")
        assert isinstance(vec, list)
        assert len(vec) == DIM_SMALL
        assert all(isinstance(x, float) for x in vec)

    def test_embed_returns_list_of_vectors(self):
        """Test that embed returns list of vectors with correct dimensions."""
        embedder = _te_core.PyTextEmbedder("Xenova/bge-small-en-v1.5")
        texts = ["Hello", "World", "Test"]
        vectors = embedder.embed(texts)
        assert isinstance(vectors, list)
        assert len(vectors) == NUM_TEST_TEXTS
        assert all(len(v) == DIM_SMALL for v in vectors)

    def test_embed_preserves_order(self):
        """Test that embed preserves input ordering."""
        embedder = _te_core.PyTextEmbedder("Xenova/bge-small-en-v1.5")
        texts = ["first", "second", "third"]
        vectors = embedder.embed(texts)
        # Each text should produce a different vector
        assert vectors[0] != vectors[1]
        assert vectors[1] != vectors[2]
        assert vectors[0] != vectors[2]

    def test_embed_empty_list(self):
        """Test that embed handles empty list."""
        embedder = _te_core.PyTextEmbedder("Xenova/bge-small-en-v1.5")
        vectors = embedder.embed([])
        assert vectors == []

    def test_invalid_model_raises_error(self):
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _te_core.PyTextEmbedder("invalid-model-name")
        assert "Unsupported" in str(exc_info.value)

    def test_embedder_with_different_models(self):
        """Test creating embedders with different model sizes."""
        # Small model
        embedder_small = _te_core.PyTextEmbedder("Xenova/bge-small-en-v1.5")
        assert embedder_small.dimension() == DIM_SMALL

        # Base model
        embedder_base = _te_core.PyTextEmbedder("Xenova/bge-base-en-v1.5")
        assert embedder_base.dimension() == DIM_BASE

        # Large model
        embedder_large = _te_core.PyTextEmbedder("Xenova/bge-large-en-v1.5")
        assert embedder_large.dimension() == DIM_LARGE


class TestPyTextEmbedderIntegration:
    """Integration tests for PyTextEmbedder."""

    def test_vectors_are_consumable_by_zvec(self):
        """Test that returned vectors work with zvec store code."""
        embedder = _te_core.PyTextEmbedder("Xenova/bge-small-en-v1.5")
        vec = embedder.embed_single("test content")
        # Verify it's a simple list of floats that can be serialized
        assert isinstance(vec, list)
        # Verify numeric values are directly usable
        assert all(isinstance(x, (int, float)) for x in vec)
        # Verify we can do basic math operations
        norm = sum(x * x for x in vec) ** 0.5
        assert norm > 0  # Should be non-zero for normalized vectors

    def test_error_message_is_actionable(self):
        """Test that error messages are clear and actionable."""
        with pytest.raises(ValueError) as exc_info:
            _te_core.PyTextEmbedder("invalid-model")
        error_msg = str(exc_info.value)
        # Should mention the invalid model
        assert "invalid-model" in error_msg
        # Should list supported models (now returns BAAI/* naming)
        assert "BAAI/bge-small-en-v1.5" in error_msg
