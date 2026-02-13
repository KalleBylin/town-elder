# Testing Strategy for Replay

## Project Overview

Replay is a Python CLI tool using:
- **typer** - CLI framework
- **zvec** - Embedded vector database (Proxima engine)
- **fastembed** - Local embedding generation (ONNX-based)

## Test Pyramid

```
        /\
       /  \      E2E Tests (5-10%)
      /----\     - Full CLI workflows
     /      \    - Real zvec + fastembed
    /--------\   Integration Tests (20-30%)
   /          \  - Component interactions
  /------------\ Unit Tests (60-70%)
 /              \- Pure functions, classes
```

### Unit Tests (60-70%)
- Pure functions and utility classes
- Data transformation/serialization
- CLI argument parsing validation
- Configuration loading
- Mocked external dependencies

### Integration Tests (20-30%)
- zvec collection creation/queries
- fastembed model loading and inference
- typer CLI command execution
- File system operations

### E2E Tests (5-10%)
- Complete CLI workflows
- Real zvec + fastembed integration
- Git hook integration (if applicable)

## Testing Framework

### Primary: pytest

```bash
pip install pytest pytest-cov pytest-mock pytest-asyncio
```

### Extensions
- **pytest-cov** - Coverage reporting
- **pytest-mock** - Mocking utilities
- **pytest-asyncio** - Async test support
- **pytest-xdist** - Parallel test execution

### Project Structure

```
tests/
├── unit/
│   ├── __init__.py
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_embeddings.py
│   └── test_vector_store.py
├── integration/
│   ├── __init__.py
│   ├── test_zvec_integration.py
│   ├── test_fastembed_integration.py
│   └── test_cli_commands.py
├── e2e/
│   ├── __init__.py
│   └── test_full_workflows.py
├── conftest.py          # Shared fixtures
└── pyproject.toml       # Test configuration
```

## Mocking Strategy

### zvec Mocking

Mock the zvec module for unit tests to avoid C++ library issues:

```python
# tests/unit/mocks/mock_zvec.py
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_zvec():
    with patch('replay.vector_store.zvec') as mock:
        # Mock collection schema
        mock.CollectionSchema.return_value = MagicMock()
        mock.VectorSchema.return_value = MagicMock()
        mock.create_and_open.return_value = MagicMock()
        yield mock
```

### fastembed Mocking

Mock fastembed for fast unit tests:

```python
# tests/unit/mocks/mock_fastembed.py
from unittest.mock import patch, MagicMock
import numpy as np

@pytest.fixture
def mock_fastembed():
    with patch('replay.embedding.fastembed') as mock:
        mock_model = MagicMock()
        # Return fixed embedding vectors for deterministic tests
        def mock_embed(texts):
            for _ in texts:
                yield np.array([0.1] * 384)  # Example dimension
        mock_model.embed.side_effect = mock_embed
        mock.TextEmbedding.return_value = mock_model
        yield mock
```

### Git Mocking

Mock git operations:

```python
# tests/unit/mocks/mock_git.py
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_git():
    with patch('replay.vcs.git') as mock:
        mock_repo = MagicMock()
        mock_repo.head.commit.hexsha = "abc123"
        mock_repo.iter_commits.return_value = []
        mock.Repo.return_value = mock_repo
        yield mock
```

### Example Unit Test with Mocks

```python
# tests/unit/test_vector_store.py
import pytest
from unittest.mock import MagicMock, patch

class TestVectorStore:
    def test_insert_document(self, mock_zvec):
        from replay.vector_store import VectorStore

        store = VectorStore(":memory:")
        store.insert("test text", {"source": "test"})

        # Verify zvec was called
        mock_zvec.create_and_open.assert_called_once()

    def test_search_returns_results(self, mock_zvec, mock_fastembed):
        from replay.vector_store import VectorStore

        # Setup mock search results
        mock_collection = mock_zvec.create_and_open.return_value
        mock_collection.query.return_value = [
            MagicMock(id="doc1", fields={"text": "test"})
        ]

        store = VectorStore(":memory:")
        results = store.search("query", top_k=5)

        assert len(results) == 1
```

## Integration Tests

Integration tests use real zvec and fastembed (optional lightweight models).

### zvec Integration

```python
# tests/integration/test_zvec_integration.py
import pytest
import tempfile
import os
import zvec

class TestZvecIntegration:
    @pytest.fixture
    def temp_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield os.path.join(tmpdir, "test.zvec")

    def test_create_and_query_collection(self, temp_db):
        schema = zvec.CollectionSchema(
            name="test",
            vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, 384)
        )
        collection = zvec.create_and_open(path=temp_db, schema=schema)

        # Insert test document
        doc = zvec.Doc(
            id="test1",
            vectors={"embedding": [0.1] * 384},
            fields={"text": "hello world"}
        )
        collection.insert([doc])

        # Query
        results = collection.query(
            zvec.VectorQuery("embedding", vector=[0.1] * 384),
            topk=1
        )

        assert len(results) > 0
```

### fastembed Integration

```python
# tests/integration/test_fastembed_integration.py
import pytest
from fastembed import TextEmbedding

class TestFastEmbedIntegration:
    @pytest.fixture
    def model(self):
        # Use smallest model for fast tests
        return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def test_generate_embedding(self, model):
        embeddings = list(model.embed(["hello world"]))

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384  # bge-small dimension

    def test_batch_embedding(self, model):
        texts = ["text1", "text2", "text3"]
        embeddings = list(model.embed(texts))

        assert len(embeddings) == 3
```

### CLI Integration Tests

```python
# tests/integration/test_cli_commands.py
import pytest
from typer.testing import CliRunner
from replay.cli import app

runner = CliRunner()

class TestCLICommands:
    def test_index_command(self, tmp_path):
        # Create test repository
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        (repo_dir / "test.py").write_text("print('hello')")

        result = runner.invoke(app, ["index", str(repo_dir)])

        assert result.exit_code == 0
        assert "Indexed" in result.stdout

    def test_search_command(self):
        result = runner.invoke(app, ["search", "authentication"])

        assert result.exit_code == 0
```

## E2E Tests

Full workflows with real dependencies.

```python
# tests/e2e/test_full_workflows.py
import pytest
import tempfile
import os
from pathlib import Path
from replay.cli import app
from typer.testing import CliRunner

runner = CliRunner()

class TestFullWorkflows:
    @pytest.fixture
    def test_repo(self, tmp_path):
        """Create a test repository with sample code."""
        repo = tmp_path / "repo"
        repo.mkdir()

        # Initialize git
        os.system(f"git init {repo}")

        # Add sample files
        (repo / "auth.py").write_text("""
def authenticate(username, password):
    '''Authenticate user credentials.'''
    if username == "admin" and password == "secret":
        return True
    return False
""")
        return repo

    def test_index_and_search_workflow(self, test_repo):
        """Test complete index + search workflow."""
        # Index the repository
        index_result = runner.invoke(
            app,
            ["index", str(test_repo), "--model", "fast"]
        )
        assert index_result.exit_code == 0

        # Search for relevant code
        search_result = runner.invoke(
            app,
            ["search", "user authentication"]
        )
        assert search_result.exit_code == 0
        assert "auth" in search_result.stdout.lower()
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[test]"

      - name: Run unit tests (fast, mocked)
        run: |
          pytest tests/unit -v --cov=replay --cov-report=xml

      - name: Run integration tests
        run: |
          pytest tests/integration -v

      - name: Run E2E tests
        run: |
          pytest tests/e2e -v

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage.xml
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: unit-tests
        name: Unit Tests
        entry: pytest tests/unit -v
        language: system
        pass_filenames: false
        always_run: true

      - id: lint
        name: Lint
        entry: ruff check replay/
        language: system
        pass_filenames: true
```

## Test Coverage Targets

| Category | Target | Priority |
|----------|--------|----------|
| Unit Tests | 80%+ | High |
| Integration Tests | 70%+ | High |
| CLI Commands | 90%+ | High |
| Core Logic (embedding, vector store) | 95%+ | Critical |
| Edge Cases | 60%+ | Medium |

### Coverage Configuration

```toml
# pyproject.toml
[tool.coverage.run]
source = ["replay"]
omit = [
    "*/tests/*",
    "*/mocks/*",
    "*/conftest.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

## Property-Based Testing

Use Hypothesis for property-based testing on data transformations.

```bash
pip install hypothesis
```

### Example: Embedding Dimension Validation

```python
# tests/property/test_embeddings.py
from hypothesis import given, settings, strategies as st
import pytest

class TestEmbeddingProperties:
    @given(texts=st.lists(st.text(min_size=1, max_size=1000), min_size=1, max_size=100))
    @settings(max_examples=10)
    def test_embedding_dimension_consistency(self, texts, mock_fastembed):
        """Embeddings should always have consistent dimensions."""
        from replay.embedding import Embedder

        embedder = Embedder()
        embeddings = embedder.embed(texts)

        # All embeddings should have same dimension
        first_dim = len(embeddings[0])
        assert all(len(e) == first_dim for e in embeddings)

    @given(text=st.text(min_size=1, max_size=10000))
    @settings(max_examples=50)
    def test_empty_text_handling(self, text):
        """Non-empty text should produce valid embeddings."""
        from replay.embedding import Embedder

        embedder = Embedder()
        result = embedder.embed([text])

        assert len(result) == 1
        assert len(result[0]) > 0
```

### Property-Based: Vector Store

```python
@given(
    vectors=st.lists(
        st.lists(st.floats(min_value=-1.0, max_value=1.0, min_size=128, max_size=128),
        min_size=1, max_size=100)
    )
)
def test_vector_normalization(vectors):
    """Vectors should be normalized after processing."""
    from replay.vector_store import normalize_vector

    for vector in vectors:
        normalized = normalize_vector(vector)
        magnitude = sum(x**2 for x in normalized) ** 0.5
        assert abs(magnitude - 1.0) < 0.001
```

## Test Fixtures (conftest.py)

```python
# tests/conftest.py
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

# Import fixtures from mocks
from tests.unit.mocks.mock_zvec import mock_zvec
from tests.unit.mocks.mock_fastembed import mock_fastembed
from tests.unit.mocks.mock_git import mock_git

@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_code():
    """Sample code for testing."""
    return '''
def authenticate(username, password):
    """Authenticate user credentials."""
    return username == "admin" and password == "secret"

class User:
    def __init__(self, name):
        self.name = name
'''

@pytest.fixture
def sample_embeddings():
    """Sample embedding vectors."""
    import numpy as np
    return [np.random.rand(384).tolist() for _ in range(5)]

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton state between tests."""
    yield
    # Cleanup after test
```

## Running Tests

```bash
# All tests
pytest

# Unit tests only (fast)
pytest tests/unit -v

# With coverage
pytest --cov=replay --cov-report=html

# Parallel execution
pytest -n auto

# Specific test file
pytest tests/unit/test_cli.py -v

# With verbose output
pytest -vv --tb=long

# Stop on first failure
pytest -x
```

## Best Practices

1. **Test Naming**: Use descriptive names `test_<function>_<expected_behavior>`
2. **Isolation**: Each test should be independent
3. **Fixtures**: Use fixtures for common setup
4. **Mock Heavy External Dependencies**: zvec C++ lib, git, file system
5. **Integration Tests**: Use smallest fastembed model (`baai/bge-small-en-v1.5`)
6. **E2E Only for Critical Paths**: Don't over-do E2E tests
7. **Deterministic**: Avoid random data in tests; use fixed seeds
8. **Fast Feedback**: Unit tests should run in < 30 seconds
