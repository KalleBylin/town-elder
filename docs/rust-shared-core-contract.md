# Rust Shared Core Interoperability Contract

This document defines the canonical input/output DTOs shared between Python and Rust for the Town Elder indexing system. It serves as the source of truth for all parity checks between implementations.

## 1. File Scanning Results

### Input DTO: ScanRequest

```python
@dataclass
class ScanRequest:
    root_path: Path          # Root directory to scan
    extensions: frozenset[str] | None = None   # File extensions (default: .py, .md, .rst)
    exclude_patterns: frozenset[str] | None = None  # Additional exclude patterns
```

### Output DTO: ScanResult

```python
@dataclass
class ScanResult:
    files: list[Path]  # Sorted list of matching file paths
```

### Python Implementation Reference

- Location: `src/town_elder/indexing/file_scanner.py:55`
- Default extensions: `frozenset({".py", ".md", ".rst"})`
- Default excludes: `.git`, `.venv`, `node_modules`, `__pycache__`, etc.
- **Normalization Rules**:
  - Results are sorted by string representation (`str(p)`) for deterministic ordering
  - Paths are absolute/resolved from the root_path

### Fixtures Required

- `tests/fixtures/rust_core_contract/file_scanner/`:
  - `simple_files/` - basic .py, .md, .rst files
  - `nested_dirs/` - files in nested directories
  - `excluded_dirs/` - test exclusion patterns work correctly

---

## 2. Git Blob Hash Scanner

### Input DTO: BlobScanRequest

```python
@dataclass
class BlobScanRequest:
    repo_path: Path  # Root of the git repository
```

### Output DTO: TrackedFile

```python
@dataclass
class TrackedFile:
    path: str       # Relative path from repo root
    blob_hash: str  # 40-character SHA-1 hex digest
    mode: str       # File mode (e.g., "100644", "100755")
```

### Python Implementation Reference

- Location: `src/town_elder/indexing/git_hash_scanner.py:32`
- Uses: `git ls-files --stage`
- **Normalization Rules**:
  - Returns dictionary keyed by relative file path
  - Skips symlinks (mode 120000) and other special files
  - File paths with spaces are preserved (split on tab delimiter)

---

## 3. Parsed File Chunks + Metadata

### Input DTO: FileWorkItem

```python
@dataclass(frozen=True)
class FileWorkItem:
    sequence: int              # Order index
    path: str                 # Absolute file path as string
    relative_path: str        # Relative path from repo root
    file_type: str           # File extension (e.g., ".py", ".rst")
    blob_hash: str | None = None   # Git blob hash (for incremental indexing)
    metadata: dict[str, Any] = field(default_factory=dict)
```

### Output DTO: ParsedFileResult

```python
@dataclass(frozen=True)
class ParsedChunk:
    text: str                      # Chunk text content
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ParsedFileResult:
    work_item: FileWorkItem
    chunks: tuple[ParsedChunk, ...] = ()
    error: str | None = None
```

### RST-Specific Output: RSTChunk

```python
@dataclass
class RSTChunk:
    text: str                              # Chunk text content
    section_path: list[str] = field(default_factory=list)  # Hierarchical section path
    directives: dict[str, list[str]] = field(default_factory=dict)  # note, warning, versionadded
    temporal_tags: list[str] = field(default_factory=list)  # deprecated, versionchanged
    chunk_index: int = 0
```

### Metadata Schema (RST)

```python
def get_chunk_metadata(chunk: RSTChunk) -> dict[str, Any]:
    return {
        "chunk_index": int,           # Zero-based index
        "section_path": list[str],    # ["Section", "Subsection"]
        "has_directives": bool,
        "has_temporal_tags": bool,
        "directives": dict | None,    # e.g., {"note": ["text"], "warning": ["text"]}
        "temporal_tags": list | None, # e.g., ["deprecated", "versionchanged: 2.0"]
        "section_depth": int | None,  # Length of section_path
    }
```

### Python Implementation Reference

- Location: `src/town_elder/indexing/pipeline.py:16-46`
- RST Parser: `src/town_elder/parsers/rst_handler.py:244`
- **Normalization Rules**:
  - Chunk index is zero-based
  - Empty RST returns fallback to plain text with `chunk_index: 0`
  - Invalid chunk_index values are normalized using fallback index

### Fixtures Required

- `tests/fixtures/rust_core_contract/rst_chunks/`:
  - `simple_section.rst` - single section heading
  - `nested_sections.rst` - multiple levels of headings
  - `with_directives.rst` - note, warning, versionadded directives
  - `with_temporal_tags.rst` - deprecated, versionchanged
  - `unicode_sections.rst` - unicode in headings (e.g., "简介", "Überblick")

---

## 4. Commit Log Parsing Records

### Input DTO: CommitQuery

```python
@dataclass
class CommitQuery:
    repo_path: Path
    since: str | None = None   # Commit hash or date
    limit: int = 100
    offset: int = 0
```

### Output DTO: Commit

```python
@dataclass
class Commit:
    hash: str                    # Full 40-char SHA-1 hash
    message: str                 # Commit subject/message
    author: str                  # Author name
    date: datetime               # ISO format datetime
    files_changed: list[str] = field(default_factory=list)
    diff: str | None = None      # Optional cached diff
```

### Python Implementation Reference

- Location: `src/town_elder/git/runner.py:23-30`
- Uses git log format: `%H%x1f%s%x1f%an%x1f%aI%x1e`
- Uses `--numstat` for batch file retrieval
- **Normalization Rules**:
  - Date parsed from ISO format via `datetime.fromisoformat()`
  - File paths extracted from numstat (tab-separated: `additions\tdeletions\tpath`)
  - Unicode in commit messages is preserved (UTF-8)

### Fixtures Required

- `tests/fixtures/rust_core_contract/commits/`:
  - `simple_commit.json` - basic commit
  - `unicode_commit.json` - commit with unicode in message (e.g., "添加新功能", "Fix encoding")
  - `merge_commit.json` - commit with multiple parents

---

## 5. Diff-to-Text Output

### Input DTO: DiffInput

```python
@dataclass
class DiffInput:
    commit_hashes: list[str]    # List of commit hashes
    max_size: int = 102400      # Max diff size in bytes (default 100KB)
```

### Output DTO: DiffFile

```python
@dataclass
class DiffFile:
    path: str           # File path (without a/ or b/ prefix)
    status: str         # "added", "modified", or "deleted"
    hunks: list[str]    # List of hunk contents
```

### Python Implementation Reference

- Location: `src/town_elder/git/diff_parser.py:89-95,174-181`
- **Normalization Rules**:
  - Path extracted from `diff --git a/path b/path` header
  - Handles quoted paths: `diff --git "a/file with spaces" "b/file with spaces"`
  - Handles unquoted paths: `diff --git a/file b/file`
  - Quoted paths are unquoted (strip surrounding quotes)
  - a/ and b/ prefixes are stripped

### Fixtures Required

- `tests/fixtures/rust_core_contract/diffs/`:
  - `simple_diff.txt` - basic add/modify/delete
  - `quoted_paths.diff` - paths with spaces (quoted format)
  - `unquoted_paths.diff` - paths with spaces (unquoted format)
  - `binary_file.diff` - binary file changes
  - `renamed_file.diff` - file rename detection

---

## 6. Deterministic Doc-ID Helpers

### Core Functions

```python
def _build_file_doc_id(path_value: str, chunk_index: int = 0) -> str:
    """Build deterministic doc ID for file content chunks."""
    doc_id_input = path_value if chunk_index == 0 else f"{path_value}#chunk:{chunk_index}"
    return hashlib.sha256(doc_id_input.encode()).hexdigest()[:16]

def _get_doc_id_inputs(path_value: str, repo_root: Path) -> set[str]:
    """Return canonical and legacy ID input strings for a path."""
    doc_id_inputs = {path_value}
    path_obj = Path(path_value)
    if not path_obj.is_absolute():
        doc_id_inputs.add(str((repo_root / path_obj).resolve()))
    return doc_id_inputs
```

### Chunk Metadata Normalization

```python
def _normalize_chunk_metadata(
    *,
    base_metadata: dict[str, Any],
    chunk_metadata: dict[str, Any],
    fallback_chunk_index: int,
) -> tuple[dict[str, Any], int]:
    """Normalize chunk metadata with fallback for invalid chunk_index."""
    metadata = dict(base_metadata)
    metadata.update(chunk_metadata)

    chunk_index_value = metadata.get("chunk_index")
    if (
        isinstance(chunk_index_value, bool)
        or not isinstance(chunk_index_value, int)
        or chunk_index_value < 0
    ):
        chunk_index = fallback_chunk_index
        metadata["chunk_index"] = chunk_index
    else:
        chunk_index = chunk_index_value

    return metadata, chunk_index
```

### Python Implementation Reference

- Location: `src/town_elder/cli/__init__.py:568-582,756-776`
- **Normalization Rules**:
  - Doc IDs use SHA256 truncated to 16 hex characters
  - Chunk 0: just the path
  - Chunk N>0: `{path}#chunk:{chunk_index}`
  - Chunk index must be non-negative integer; invalid values use fallback

### Fixtures Required

- `tests/fixtures/rust_core_contract/doc_ids/`:
  - `path_with_spaces.txt` - test path containing spaces
  - `unicode_path.txt` - test path with unicode characters

---

## 7. Test Fixture Schema

All fixtures should follow this JSON schema when serialized:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "description": "Fixture for Rust core contract validation"
}
```

### Fixture Naming Convention

- `{category}_{description}.{ext}` for data files
- `expected_{output_type}.json` for expected outputs
- `README.md` in each directory documenting fixture purpose

---

## 8. Acceptance Criteria

1. All DTOs have explicit field definitions with types
2. All normalization rules are documented with examples
3. Fixtures cover edge cases: unicode, spaces, special characters
4. Tests validate fixture shape against current Python behavior
5. No changes to user-facing CLI behavior

---

## 9. Related Files

- `src/town_elder/cli/__init__.py` - CLI entry point, doc-id helpers
- `src/town_elder/indexing/file_scanner.py` - File scanning
- `src/town_elder/indexing/pipeline.py` - File parsing pipeline
- `src/town_elder/indexing/git_hash_scanner.py` - Git blob scanning
- `src/town_elder/parsers/rst_handler.py` - RST parsing
- `src/town_elder/git/runner.py` - Git commit operations
- `src/town_elder/git/diff_parser.py` - Diff parsing
