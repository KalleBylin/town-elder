"""Index service for replay."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from replay.config import get_config
from replay.embeddings import Embedder
from replay.storage import ZvecStore


class IndexService:
    """Service for indexing documents."""

    def __init__(
        self,
        store: ZvecStore | None = None,
        embedder: Embedder | None = None,
    ):
        config = get_config()
        self.store = store or ZvecStore(config.data_dir / "vectors")
        self.embedder = embedder or Embedder()

    def index_text(self, text: str, doc_id: str, metadata: dict[str, Any] | None = None) -> str:
        """Index a single text document."""
        metadata = metadata or {}
        vector = self.embedder.embed_single(text)
        self.store.insert(doc_id, vector, text, metadata)
        return doc_id

    def index_file(self, file_path: Path) -> str | None:
        """Index a file."""
        try:
            text = file_path.read_text()
            doc_id = str(file_path)
            metadata = {
                "source": str(file_path),
                "type": file_path.suffix,
            }
            return self.index_text(text, doc_id, metadata)
        except Exception:
            return None

    def index_directory(self, directory: Path, extensions: list[str] | None = None) -> int:
        """Index all files in a directory."""
        extensions = extensions or [".py", ".md", ".txt", ".json", ".yaml", ".yml"]
        count = 0

        for ext in extensions:
            for file_path in directory.rglob(f"*{ext}"):
                if self.index_file(file_path):
                    count += 1

        return count

    def close(self) -> None:
        """Close the service."""
        self.store.close()
