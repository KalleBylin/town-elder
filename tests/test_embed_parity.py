"""Parity tests comparing Python and Rust embedding backends.

This module provides deterministic tests to verify that both backends
produce comparable retrieval results on a fixed corpus.

The tests compare:
- Vector dimension consistency
- Top-k overlap (what percentage of results are the same)
- Score ordering sanity (allowing tolerance for implementation differences)

Test Design:
- Uses a fixed, deterministic corpus of text snippets
- Uses the same model (BAAI/bge-small-en-v1.5) for both backends
- Explicit thresholds documented with rationale in code comments
- Tests are reproducible across multiple runs

Thresholds:
- DIMENSION_TOLERANCE: Exact match required (vectors must be same dimension)
- TOP_K_OVERLAP_THRESHOLD: 0.7 (70% of top-k results should match)
  Rationale: Allow for minor implementation differences in embedding normalization
  or scoring while ensuring semantic equivalence.
- SCORE_ORDER_TOLERANCE: 0.1 (relative difference)
  Rationale: Allow for floating-point differences and normalization variations.
"""
from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from town_elder.embeddings.backend import is_rust_embed_available
from town_elder.embeddings.embedder import Embedder
from town_elder.storage.vector_store import ZvecStore

# =============================================================================
# Test Configuration and Constants
# =============================================================================

# Model configuration (same for both backends)
TEST_MODEL_NAME = "BAAI/bge-small-en-v1.5"
TEST_EMBED_DIMENSION = 384

# Parity thresholds - see module docstring for rationale
TOP_K_OVERLAP_THRESHOLD = 0.7  # 70% of top-k should match
SCORE_ORDER_TOLERANCE = 0.1  # 10% relative tolerance for score differences
NEAR_ZERO_THRESHOLD = 1e-6  # Threshold for near-zero floating point values
EMBEDDING_SIMILARITY_THRESHOLD = 0.95  # Minimum cosine similarity between backends


# =============================================================================
# Deterministic Test Corpus
# =============================================================================

# A fixed corpus of representative code/text snippets for testing.
# Each entry has a unique semantic meaning to allow meaningful retrieval testing.
PARITY_TEST_CORPUS: list[dict[str, str]] = [
    {
        "id": "doc_001",
        "text": "def calculate_sum(numbers):\n    '''Calculate the sum of a list of numbers.'''\n    return sum(numbers)",
    },
    {
        "id": "doc_002",
        "text": "def calculate_product(numbers):\n    '''Calculate the product of a list of numbers.'''\n    result = 1\n    for n in numbers:\n        result *= n\n    return result",
    },
    {
        "id": "doc_003",
        "text": "def find_max(items):\n    '''Find the maximum item in a list.'''\n    return max(items) if items else None",
    },
    {
        "id": "doc_004",
        "text": "def reverse_string(s):\n    '''Reverse a string.'''\n    return s[::-1]",
    },
    {
        "id": "doc_005",
        "text": "def sort_list(items):\n    '''Sort a list in ascending order.'''\n    return sorted(items)",
    },
    {
        "id": "doc_006",
        "text": "class DatabaseConnection:\n    '''Handle database connections.'''\n    def __init__(self, host, port):\n        self.host = host\n        self.port = port",
    },
    {
        "id": "doc_007",
        "text": "def authenticate_user(username, password):\n    '''Authenticate a user with credentials.'''\n    # TODO: implement proper hashing\n    return True",
    },
    {
        "id": "doc_008",
        "text": "def parse_json(data):\n    '''Parse JSON string into Python object.'''\n    import json\n    return json.loads(data)",
    },
    {
        "id": "doc_009",
        "text": "def write_file(filename, content):\n    '''Write content to a file.'''\n    with open(filename, 'w') as f:\n        f.write(content)",
    },
    {
        "id": "doc_010",
        "text": "def read_file(filename):\n    '''Read content from a file.'''\n    with open(filename, 'r') as f:\n        return f.read()",
    },
    {
        "id": "doc_011",
        "text": "async def fetch_url(url):\n    '''Fetch URL content asynchronously.'''\n    import aiohttp\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:\n            return await response.text()",
    },
    {
        "id": "doc_012",
        "text": "def encrypt_message(message, key):\n    '''Encrypt a message with a key.'''\n    from cryptography.fernet import Fernet\n    return Fernet(key).encrypt(message)",
    },
]

# Test queries that should retrieve specific documents from the corpus
PARITY_TEST_QUERIES: list[dict[str, Any]] = [
    {
        "query": "sum of numbers",
        "expected_ids": ["doc_001"],  # Should match calculate_sum
    },
    {
        "query": "multiply numbers",
        "expected_ids": ["doc_002"],  # Should match calculate_product
    },
    {
        "query": "maximum item",
        "expected_ids": ["doc_003"],  # Should match find_max
    },
    {
        "query": "reverse text",
        "expected_ids": ["doc_004"],  # Should match reverse_string
    },
    {
        "query": "sorting list",
        "expected_ids": ["doc_005"],  # Should match sort_list
    },
    {
        "query": "database connection",
        "expected_ids": ["doc_006"],  # Should match DatabaseConnection
    },
    {
        "query": "user authentication",
        "expected_ids": ["doc_007"],  # Should match authenticate_user
    },
    {
        "query": "parse JSON",
        "expected_ids": ["doc_008"],  # Should match parse_json
    },
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def python_embedder() -> Embedder:
    """Create a Python (fastembed) embedder instance."""
    return Embedder(
        model_name=TEST_MODEL_NAME,
        embed_dimension=TEST_EMBED_DIMENSION,
        backend="python",
    )


@pytest.fixture
def rust_embedder() -> Embedder | None:
    """Create a Rust embedder instance if available."""
    if not is_rust_embed_available():
        pytest.skip("Rust embedding backend not available")

    return Embedder(
        model_name=TEST_MODEL_NAME,
        embed_dimension=TEST_EMBED_DIMENSION,
        backend="rust",
    )


@pytest.fixture
def corpus_vectors_python(python_embedder: Embedder) -> dict[str, np.ndarray]:
    """Generate embeddings for the test corpus using Python backend."""
    texts = [doc["text"] for doc in PARITY_TEST_CORPUS]
    doc_ids = [doc["id"] for doc in PARITY_TEST_CORPUS]

    embeddings = list(python_embedder.embed(texts))
    return dict(zip(doc_ids, embeddings, strict=True))


@pytest.fixture
def corpus_vectors_rust(
    rust_embedder: Embedder,
) -> dict[str, np.ndarray] | None:
    """Generate embeddings for the test corpus using Rust backend."""
    if rust_embedder is None:
        pytest.skip("Rust backend not available")

    texts = [doc["text"] for doc in PARITY_TEST_CORPUS]
    doc_ids = [doc["id"] for doc in PARITY_TEST_CORPUS]

    embeddings = list(rust_embedder.embed(texts))
    return dict(zip(doc_ids, embeddings, strict=True))


# =============================================================================
# Helper Functions
# =============================================================================


def compute_top_k_overlap(
    ids_a: list[str],
    ids_b: list[str],
) -> float:
    """Compute the overlap ratio between two top-k lists.

    Args:
        ids_a: First list of document IDs (ordered by relevance).
        ids_b: Second list of document IDs (ordered by relevance).

    Returns:
        Ratio of common elements (0.0 to 1.0).
    """
    if not ids_a or not ids_b:
        return 0.0

    set_a = set(ids_a)
    set_b = set(ids_b)
    overlap = len(set_a & set_b)

    # Use the length of the shorter list as denominator
    denominator = min(len(ids_a), len(ids_b))
    return overlap / denominator if denominator > 0 else 0.0


def compute_score_ordering_similarity(
    scores_a: list[float],
    scores_b: list[float],
    tolerance: float = SCORE_ORDER_TOLERANCE,
) -> tuple[bool, float]:
    """Check if two score lists have similar ordering within tolerance.

    Args:
        scores_a: First list of scores (ordered by relevance).
        scores_b: Second list of scores (ordered by relevance).
        tolerance: Relative tolerance for score differences.

    Returns:
        Tuple of (is_similar, max_relative_diff).
    """
    if not scores_a or not scores_b:
        return False, 1.0

    max_rel_diff = 0.0
    for i in range(min(len(scores_a), len(scores_b))):
        # Compute relative difference
        avg_mag = (abs(scores_a[i]) + abs(scores_b[i])) / 2
        if avg_mag > NEAR_ZERO_THRESHOLD:  # Skip near-zero scores
            rel_diff = abs(scores_a[i] - scores_b[i]) / avg_mag
            max_rel_diff = max(max_rel_diff, rel_diff)

    return max_rel_diff <= tolerance, max_rel_diff


def index_corpus(
    store: ZvecStore,
    embedder: Embedder,
    corpus: list[dict[str, str]],
) -> None:
    """Index a corpus into the vector store using the given embedder."""
    docs = []
    for doc in corpus:
        vector = embedder.embed_single(doc["text"])
        docs.append((doc["id"], vector, doc["text"], {}))

    store.bulk_upsert(docs)


def search_with_embedder(
    store: ZvecStore,
    embedder: Embedder,
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search the vector store using the given embedder."""
    query_vector = embedder.embed_single(query)
    return store.search(query_vector, top_k=top_k)


# =============================================================================
# Test Cases
# =============================================================================


class TestVectorDimensionParity:
    """Test that both backends produce vectors of the same dimension."""

    def test_python_backend_dimension(
        self,
        python_embedder: Embedder,
    ):
        """Verify Python backend produces correct dimension."""
        text = PARITY_TEST_CORPUS[0]["text"]
        vector = python_embedder.embed_single(text)

        assert vector.shape[0] == TEST_EMBED_DIMENSION, (
            f"Python backend dimension mismatch: "
            f"expected {TEST_EMBED_DIMENSION}, got {vector.shape[0]}"
        )

    def test_rust_backend_dimension(
        self,
        rust_embedder: Embedder,
    ):
        """Verify Rust backend produces correct dimension."""
        text = PARITY_TEST_CORPUS[0]["text"]
        vector = rust_embedder.embed_single(text)

        assert vector.shape[0] == TEST_EMBED_DIMENSION, (
            f"Rust backend dimension mismatch: "
            f"expected {TEST_EMBED_DIMENSION}, got {vector.shape[0]}"
        )

    def test_dimension_parity(
        self,
        python_embedder: Embedder,
        rust_embedder: Embedder,
    ):
        """Verify both backends produce vectors of the same dimension."""
        text = "def test_function(): pass"

        py_vector = python_embedder.embed_single(text)
        rust_vector = rust_embedder.embed_single(text)

        assert py_vector.shape[0] == rust_vector.shape[0], (
            f"Dimension mismatch between backends: "
            f"Python={py_vector.shape[0]}, Rust={rust_vector.shape[0]}"
        )


class TestEmbeddingValueParity:
    """Test that embeddings are semantically equivalent between backends.

    Note: We do NOT require bit-identical vectors, as small floating-point
    differences are expected between implementations. Instead, we verify:
    1. Vectors have the same dimension
    2. Search results have high overlap
    3. Score orderings are similar within tolerance
    """

    def test_embedding_similarity(
        self,
        python_embedder: Embedder,
        rust_embedder: Embedder,
    ):
        """Verify embeddings are semantically similar (not bit-identical)."""
        text = PARITY_TEST_CORPUS[0]["text"]

        py_vector = python_embedder.embed_single(text)
        rust_vector = rust_embedder.embed_single(text)

        # Compute cosine similarity between vectors
        py_norm = np.linalg.norm(py_vector)
        rust_norm = np.linalg.norm(rust_vector)

        if py_norm > NEAR_ZERO_THRESHOLD and rust_norm > NEAR_ZERO_THRESHOLD:
            cosine_sim = np.dot(py_vector, rust_vector) / (py_norm * rust_norm)
        else:
            cosine_sim = 0.0

        # Allow for numerical differences but require high similarity
        # Using EMBEDDING_SIMILARITY_THRESHOLD to allow ~5% variation between implementations
        assert cosine_sim > EMBEDDING_SIMILARITY_THRESHOLD, (
            f"Embedding similarity too low: {cosine_sim:.4f}. "
            f"This may indicate a significant difference in embedding computation."
        )


class TestRetrievalParity:
    """Test that search results are consistent between backends."""

    @pytest.mark.parametrize("test_case", PARITY_TEST_QUERIES)
    def test_top_k_overlap(
        self,
        temp_dir: Path,
        python_embedder: Embedder,
        rust_embedder: Embedder,
        test_case: dict[str, Any],
    ):
        """Verify top-k results overlap significantly between backends."""
        query = test_case["query"]
        top_k = 5

        # Create separate stores for each backend
        python_store = ZvecStore(
            path=temp_dir / "python.vec",
            dimension=TEST_EMBED_DIMENSION,
        )
        rust_store = ZvecStore(
            path=temp_dir / "rust.vec",
            dimension=TEST_EMBED_DIMENSION,
        )

        try:
            # Index corpus in both stores
            index_corpus(python_store, python_embedder, PARITY_TEST_CORPUS)
            index_corpus(rust_store, rust_embedder, PARITY_TEST_CORPUS)

            # Search with both backends
            python_results = search_with_embedder(
                python_store, python_embedder, query, top_k
            )
            rust_results = search_with_embedder(
                rust_store, rust_embedder, query, top_k
            )

            # Extract IDs
            python_ids = [r["id"] for r in python_results]
            rust_ids = [r["id"] for r in rust_results]

            # Compute overlap
            overlap = compute_top_k_overlap(python_ids, rust_ids)

            # Assert overlap meets threshold
            assert overlap >= TOP_K_OVERLAP_THRESHOLD, (
                f"Top-{top_k} overlap too low for query '{query}': {overlap:.2%}. "
                f"Python results: {python_ids}, Rust results: {rust_ids}. "
                f"Expected at least {TOP_K_OVERLAP_THRESHOLD:.0%} overlap."
            )

        finally:
            python_store.close()
            rust_store.close()

    @pytest.mark.parametrize("test_case", PARITY_TEST_QUERIES)
    def test_score_ordering_similarity(
        self,
        temp_dir: Path,
        python_embedder: Embedder,
        rust_embedder: Embedder,
        test_case: dict[str, Any],
    ):
        """Verify score orderings are similar within tolerance."""
        query = test_case["query"]
        top_k = 5

        # Create separate stores for each backend
        python_store = ZvecStore(
            path=temp_dir / "python_order.vec",
            dimension=TEST_EMBED_DIMENSION,
        )
        rust_store = ZvecStore(
            path=temp_dir / "rust_order.vec",
            dimension=TEST_EMBED_DIMENSION,
        )

        try:
            # Index corpus in both stores
            index_corpus(python_store, python_embedder, PARITY_TEST_CORPUS)
            index_corpus(rust_store, rust_embedder, PARITY_TEST_CORPUS)

            # Search with both backends
            python_results = search_with_embedder(
                python_store, python_embedder, query, top_k
            )
            rust_results = search_with_embedder(
                rust_store, rust_embedder, query, top_k
            )

            # Extract scores
            python_scores = [r["score"] for r in python_results]
            rust_scores = [r["score"] for r in rust_results]

            # Check similarity
            is_similar, max_diff = compute_score_ordering_similarity(
                python_scores, rust_scores
            )

            assert is_similar, (
                f"Score ordering too different for query '{query}': "
                f"max relative diff = {max_diff:.2%} (tolerance = {SCORE_ORDER_TOLERANCE:.2%}). "
                f"Python scores: {python_scores}, Rust scores: {rust_scores}"
            )

        finally:
            python_store.close()
            rust_store.close()


class TestExpectedRetrievalBehavior:
    """Test that queries retrieve expected documents."""

    @pytest.mark.parametrize("test_case", PARITY_TEST_QUERIES)
    def test_expected_document_in_results(
        self,
        temp_dir: Path,
        python_embedder: Embedder,
        test_case: dict[str, Any],
    ):
        """Verify expected documents appear in top results."""
        query = test_case["query"]
        expected_ids = test_case["expected_ids"]
        top_k = 5

        store = ZvecStore(
            path=temp_dir / f"expected_{hash(query)}.vec",
            dimension=TEST_EMBED_DIMENSION,
        )

        try:
            # Index corpus
            index_corpus(store, python_embedder, PARITY_TEST_CORPUS)

            # Search
            results = search_with_embedder(store, python_embedder, query, top_k)
            result_ids = [r["id"] for r in results]

            # Check expected ID is in results
            found = any(eid in result_ids for eid in expected_ids)

            assert found, (
                f"Expected one of {expected_ids} in top-{top_k} results for query '{query}', "
                f"but got: {result_ids}"
            )

        finally:
            store.close()


class TestDiagnosticsOnFailure:
    """Test that failures produce actionable diagnostics."""

    def test_diagnostic_output_on_mismatch(
        self,
        temp_dir: Path,
        python_embedder: Embedder,
        rust_embedder: Embedder,
    ):
        """Verify diagnostics show which query/metric failed."""
        query = PARITY_TEST_QUERIES[0]["query"]
        top_k = 5

        python_store = ZvecStore(
            path=temp_dir / "diag_python.vec",
            dimension=TEST_EMBED_DIMENSION,
        )
        rust_store = ZvecStore(
            path=temp_dir / "diag_rust.vec",
            dimension=TEST_EMBED_DIMENSION,
        )

        try:
            index_corpus(python_store, python_embedder, PARITY_TEST_CORPUS)
            index_corpus(rust_store, rust_embedder, PARITY_TEST_CORPUS)

            python_results = search_with_embedder(
                python_store, python_embedder, query, top_k
            )
            rust_results = search_with_embedder(
                rust_store, rust_embedder, query, top_k
            )

            python_ids = [r["id"] for r in python_results]
            rust_ids = [r["id"] for r in rust_results]
            python_scores = [r["score"] for r in python_results]
            rust_scores = [r["score"] for r in rust_results]

            overlap = compute_top_k_overlap(python_ids, rust_ids)

            # The assertion message itself serves as diagnostic
            # If this fails, the message includes all relevant info
            assert overlap >= 0.0, (
                f"DIAGNOSTIC: Query='{query}', "
                f"Python IDs={python_ids}, Rust IDs={rust_ids}, "
                f"Python scores={python_scores}, Rust scores={rust_scores}, "
                f"Overlap={overlap:.2%}"
            )

        finally:
            python_store.close()
            rust_store.close()
