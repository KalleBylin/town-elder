"""Index service for town_elder."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from town_elder.config import get_config
from town_elder.embeddings import Embedder
from town_elder.exceptions import IndexingError
from town_elder.storage import ZvecStore


class IndexService:
    """Service for indexing documents."""

    def __init__(
        self,
        store: ZvecStore | None = None,
        embedder: Embedder | None = None,
    ):
        config = get_config()
        self.store = store or ZvecStore(config.data_dir / "vectors")
        self.embedder = embedder or Embedder(model_name=config.embed_model)

    def index_text(self, text: str, doc_id: str, metadata: dict[str, Any] | None = None) -> str:
        """Index a single text document."""
        metadata = metadata or {}
        vector = self.embedder.embed_single(text)
        self.store.insert(doc_id, vector, text, metadata)
        return doc_id

    def index_file(self, file_path: Path) -> str:
        """Index a file.

        Raises IndexingError if the file cannot be read or indexed.
        """
        try:
            text = file_path.read_text()
        except OSError as e:
            raise IndexingError(f"Failed to read file {file_path}: {e}") from e
        except UnicodeDecodeError as e:
            raise IndexingError(f"Failed to decode file {file_path}: {e}") from e

        # zvec requires alphanumeric doc_ids, so we hash the path
        file_path_str = str(file_path)
        doc_id = hashlib.sha256(file_path_str.encode()).hexdigest()[:16]
        metadata = {
            "source": str(file_path),
            "type": file_path.suffix,
        }
        return self.index_text(text, doc_id, metadata)

    def index_directory(self, directory: Path, extensions: list[str] | None = None) -> int:
        """Index all files in a directory.

        Raises IndexingError if indexing fails.
        """
        extensions = extensions or [".py", ".md", ".txt", ".json", ".yaml", ".yml"]
        count = 0
        errors: list[str] = []

        for ext in extensions:
            for file_path in directory.rglob(f"*{ext}"):
                try:
                    self.index_file(file_path)
                    count += 1
                except IndexingError as e:
                    errors.append(str(e))

        if errors:
            raise IndexingError(f"Failed to index {len(errors)} file(s): {'; '.join(errors)}")

        return count

    def close(self) -> None:
        """Close the service."""
        self.store.close()
