"""Pytest configuration and fixtures for replay tests."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from replay.config import ReplayConfig
from replay.embeddings.embedder import Embedder
from replay.git.diff_parser import DiffParser
from replay.storage.vector_store import ZvecStore


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def mock_config(temp_dir: Path) -> ReplayConfig:
    """Create a mock config for testing."""
    return ReplayConfig(
        data_dir=temp_dir / "data",
        db_name="test.db",
        embed_model="BAAI/bge-small-en-v1.5",
        embed_dimension=384,
        default_top_k=5,
        verbose=True,
    )


@pytest.fixture
def sample_embedder() -> Embedder:
    """Create a sample embedder instance."""
    return Embedder(model_name="BAAI/bge-small-en-v1.5")


@pytest.fixture
def sample_vector_store(temp_dir: Path) -> ZvecStore:
    """Create a sample vector store instance."""
    return ZvecStore(path=temp_dir / "test.vec", dimension=384)


@pytest.fixture
def diff_parser() -> DiffParser:
    """Create a diff parser instance."""
    return DiffParser()


@pytest.fixture
def sample_diff_output() -> str:
    """Sample git diff output for testing."""
    return """diff --git a/src/main.py b/src/main.py
new file mode 100644
--- /dev/null
+++ b/src/main.py
@@ -0,0 +1,5 @@
+def hello():
+    print("Hello, world!")
+
+if __name__ == "__main__":
+    hello()
diff --git a/src/utils.py b/src/utils.py
deleted file mode 100644
--- a/src/utils.py
+++ /dev/null
@@ -1,3 +0,0 @@
-def old_func():
-    pass
-
diff --git a/src/app.py b/src/app.py
--- a/src/app.py
+++ b/src/app.py
@@ -1,5 +1,6 @@
+import new_module
 def main():
-    old_call()
+    new_call()
     return True
"""


@pytest.fixture
def sample_vectors() -> list[np.ndarray]:
    """Create sample vectors for testing search operations."""
    # Create 5 sample vectors with some similarity
    vectors = []
    # Vector 0: [1, 0, 0, ...] - distinct
    v0 = np.zeros(384, dtype=np.float32)
    v0[0] = 1.0
    vectors.append(v0)

    # Vector 1: [1, 0.5, 0, ...] - similar to v0
    v1 = np.zeros(384, dtype=np.float32)
    v1[0] = 1.0
    v1[1] = 0.5
    vectors.append(v1)

    # Vector 2: [0, 1, 0, ...] - distinct
    v2 = np.zeros(384, dtype=np.float32)
    v2[1] = 1.0
    vectors.append(v2)

    # Vector 3: [0, 1, 0.3, ...] - similar to v2
    v3 = np.zeros(384, dtype=np.float32)
    v3[1] = 1.0
    v3[2] = 0.3
    vectors.append(v3)

    # Vector 4: [0, 0, 1, ...] - distinct
    v4 = np.zeros(384, dtype=np.float32)
    v4[2] = 1.0
    vectors.append(v4)

    return vectors
