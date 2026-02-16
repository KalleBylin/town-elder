"""Tests for the file indexing pipeline stages."""

from __future__ import annotations

from pathlib import Path

from town_elder.indexing.pipeline import (
    FileWorkItem,
    build_file_work_items,
    parse_files_pipeline,
    parse_work_item,
)


def _make_item(path: Path, relative_path: str) -> FileWorkItem:
    return FileWorkItem(
        sequence=0,
        path=str(path),
        relative_path=relative_path,
        file_type=path.suffix,
        blob_hash=None,
        metadata={"source": str(path), "type": path.suffix},
    )


def test_parse_work_item_supports_markdown(tmp_path: Path) -> None:
    md_file = tmp_path / "README.md"
    md_text = "# Town Elder\n\nMarkdown content.\n"
    md_file.write_text(md_text)

    result = parse_work_item(_make_item(md_file, "README.md"))

    assert not result.has_error
    assert len(result.chunks) == 1
    assert result.chunks[0].text == md_text
    assert result.chunks[0].metadata["chunk_index"] == 0


def test_parse_work_item_rst_emits_semantic_chunk_metadata(tmp_path: Path) -> None:
    rst_file = tmp_path / "guide.rst"
    rst_file.write_text(
        "Guide\n=====\n\nIntro.\n\nDetails\n-------\n\nBody text.\n"
    )

    result = parse_work_item(_make_item(rst_file, "guide.rst"))

    assert not result.has_error
    assert len(result.chunks) >= 1
    assert all("chunk_index" in chunk.metadata for chunk in result.chunks)
    assert any(chunk.metadata.get("section_depth", 0) >= 1 for chunk in result.chunks)


def test_parse_files_pipeline_preserves_work_item_order(tmp_path: Path) -> None:
    py_file = tmp_path / "a.py"
    py_file.write_text("print('a')\n")
    md_file = tmp_path / "b.md"
    md_file.write_text("# B\n")
    rst_file = tmp_path / "c.rst"
    rst_file.write_text("Title\n=====\n\nRST body.\n")

    files_to_process = [
        (py_file, "a.py", None),
        (md_file, "b.md", None),
        (rst_file, "c.rst", None),
    ]
    work_items = build_file_work_items(files_to_process)

    parsed_results = parse_files_pipeline(work_items, max_workers=2, queue_size=1)

    assert [result.work_item.relative_path for result in parsed_results] == [
        "a.py",
        "b.md",
        "c.rst",
    ]
    assert all(not result.has_error for result in parsed_results)
