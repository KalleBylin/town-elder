# te CLI Design

## Overview

`te` is a semantic version control system that provides context-aware search over git history. It indexes commits and code changes as semantic vectors, enabling developers to query their project's history by meaning rather than just keywords.

## Core Philosophy

- **Local-First**: No external services, all data stored in `.git/te/`
- **Zero-Ops**: `pip install town-elder` and it just works
- **Dual Output**: Human-readable for exploration, JSON for scripting
- **Typer-Native**: Rich CLI with auto-completion, validation, and good help text

---

## Command Structure

```
te [OPTIONS] COMMAND [ARGS]...
```

### Main Commands

| Command | Description |
|---------|-------------|
| `query` | Search the semantic index |
| `index` | Manually trigger indexing |
| `status` | Show index health and stats |
| `init` | Initialize te in a repository |

---

## Commands Detail

### 1. `te init`

Initialize te in the current git repository. Creates the vector store and installs git hooks.

```bash
te init [OPTIONS]
```

**Options:**
- `--force` - Reinitialize even if already initialized

**Output:**
```
Initialized te in /path/to/repo/.git/te/
✓ Created vector store
✓ Installed post-commit hook
✓ Downloaded embedding model (first run only)

Run 'te status' to verify setup.
```

---

### 2. `te query <query_text>`

Search the semantic index for commits and code changes related to your query.

```bash
te query [OPTIONS] QUERY
```

**Arguments:**
- `QUERY` - The search query (required)

**Options:**
- `-n, --top-k <N>` - Number of results to return (default: 5, max: 20)
- `--json` - Output as JSON for machine parsing
- `--format <FORMAT>` - Output format: `text` (default), `json`, `compact`
- `--since <REF>` - Only search commits after this ref (branch, tag, commit hash)
- `--author <NAME>` - Filter by commit author
- `--path <GLOB>` - Only search commits touching files matching glob (e.g., `src/*.py`)

**Output Formats:**

*Text (default):*
```
Results for "authentication retry logic"

1. commit abc1234 (2 days ago)
   Author: Jane Doe <jane@example.com>
   Message: Fix race condition in retry logic by adding exponential backoff

   Files changed: src/auth.py (+15, -3)

   Score: 0.94
   ─────────────────────────────────────────────────────────

2. commit def5678 (3 weeks ago)
   Author: John Smith <john@example.com>
   Message: Initial implementation of retry mechanism

   Files changed: src/auth.py (+45, -0)

   Score: 0.87
   ─────────────────────────────────────────────────────────
```

*JSON:*
```json
{
  "query": "authentication retry logic",
  "results": [
    {
      "commit": "abc1234",
      "author": "Jane Doe <jane@example.com>",
      "date": "2026-02-11T10:30:00Z",
      "message": "Fix race condition in retry logic by adding exponential backoff",
      "files_changed": [{"path": "src/auth.py", "additions": 15, "deletions": 3}],
      "score": 0.94,
      "chunk_preview": "async def retry_with_backoff..."
    }
  ]
}
```

*Compact (single line per result):*
```
abc1234 0.94 Fix race condition in retry logic by adding exponential backoff
def5678 0.87 Initial implementation of retry mechanism
```

---

### 3. `te index`

Manually trigger indexing of unindexed commits. Normally runs automatically via post-commit hook.

```bash
te index [OPTIONS]
```

**Options:**
- `--commit <REF>` - Index a specific commit (default: HEAD)
- `--range <FROM>..<TO>` - Index a range of commits
- `--all` - Reindex all commits (rebuilds entire index)
- `--dry-run` - Show what would be indexed without actually indexing
- `--verbose` - Show detailed progress

**Output:**
```
Indexing 3 new commits...

[1/3] Commit abc1234
  - Chunking src/auth.py... 2 chunks
  - Chunking src/utils.py... 1 chunk
  - Generating embeddings... done
  - Stored 3 vectors

[2/3] Commit def5678
  - Chunking src/api.py... 4 chunks
  - Generating embeddings... done
  - Stored 4 vectors

[3/3] Commit ghi9012
  - Chunking tests/test_auth.py... 1 chunk
  - Generating embeddings... done
  - Stored 1 vector

✓ Indexed 3 commits (8 chunks, 0.4s)
```

---

### 4. `te status`

Show index health, statistics, and configuration.

```bash
te status [OPTIONS]
```

**Options:**
- `--json` - Output as JSON
- `--verbose` - Show detailed statistics

**Output:**
```
te status

Repository: /path/to/repo
Index: .git/te/

Statistics:
  - Total commits indexed: 1,247
  - Total chunks: 8,934
  - Index size: 12.4 MB
  - Last indexed: 2 hours ago (commit abc1234)

Configuration:
  - Embedding model: BAAI/bge-small-en-v1.5
  - Chunk size: 256 tokens
  - Overlap: 50 tokens

Status: ✓ Healthy
  - 0 pending commits to index
  - Git hook installed (post-commit)
```

---

## Global Options

| Option | Description |
|--------|-------------|
| `--help` | Show help and exit |
| `--version` | Show version |
| `--quiet` | Suppress non-essential output |
| `--verbose` | Enable verbose logging |
| `--config <PATH>` | Custom config file path |

---

## Configuration

### File Location

Config is read from (in order of precedence):
1. `./te.yaml` (project root)
2. `./.te.yaml` (project root)
3. `~/.config/te/config.yaml` (user home)

### Config Schema

```yaml
# te.yaml
index:
  # Embedding model (default: BAAI/bge-small-en-v1.5)
  embedding_model: BAAI/bge-small-en-v1.5

  # Chunking strategy
  chunk_size: 256
  chunk_overlap: 50

  # File patterns to include/exclude
  include:
    - "*.py"
    - "*.js"
    - "*.ts"
  exclude:
    - "*.test.js"
    - "node_modules/**"
    - "__pycache__/**"

query:
  # Default number of results
  default_top_k: 5

  # Minimum score threshold (0-1)
  min_score: 0.5

output:
  # Default format: text, json, compact
  default_format: text

  # Colorize output (auto, always, never)
  color: auto
```

---

## Error Handling

### User-Friendly Errors

All errors include context and suggested fixes:

```
Error: Not a git repository

  te must be run from within a git repository.
  Run 'git init' first, then 'te init'.
```

```
Error: Index not initialized

  Run 'te init' to set up the semantic index.
  This only needs to be done once per repository.
```

```
Error: Embedding model not found

  Failed to download embedding model.
  Check your internet connection, then run:
    te init --force
```

```
Error: No commits to index

  The repository has no commits yet.
  Make at least one commit, then run:
    te index
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid usage (bad arguments) |
| 3 | Not a git repository |
| 4 | Index not initialized |
| 5 | Index corrupted (run `te index --all` to rebuild) |

---

## Git Hook Integration

### Post-Commit Hook

When initialized, te installs a post-commit hook that automatically indexes new commits:

```bash
# .git/hooks/post-commit
#!/bin/sh
te index --commit HEAD --quiet
```

The hook runs silently unless `--verbose` or there's an error.

### Manual Hook Management

```bash
# Install hook manually
te init

# Remove hook
rm .git/hooks/post-commit
```

---

## Shell Completions

Typer provides shell completions for bash, zsh, fish, and PowerShell:

```bash
# Add to .bashrc or .zshrc
eval "$(te --install-completion bash)"
```

This enables:
- Command completion: `te q<TAB>` → `te query`
- Option completion: `te query --f<TAB>` → `--format`
- Argument completion: `te query "auth"<TAB>` (shows recent queries)

---

## Usage Examples

### Typical Workflow

```bash
# Initialize in a new repo
cd my-project
te init

# ... make some commits ...

# Find related changes
te query "payment validation" -n 10

# Check index health
te status

# Rebuild index if corrupted
te index --all
```

### CI/CD Integration

```bash
# In CI: index a commit range after push
te index --range $CI_COMMIT_BEFORE_SHA..$CI_COMMIT_SHA --json > index.json

# Parse results
cat index.json | jq '.indexed'
```

### Scripting

```bash
# Find all commits by author touching specific files
te query "bug fix" --author "jane@example.com" --path "src/*.py" --json | \
  jq '.results[] | {commit: .commit, message: .message}'
```

---

## Future Considerations

### Planned Features (Out of Scope for v1)

- `te blame`: Semantic blame (find conceptually related changes)
- `te diff <query>`: Show semantic diff between commits
- `te context`: Generate AGENTS.md context for AI agents
- `--interactive`: Interactive query mode with refinement

### Integration Points

- **MCP Server**: Expose te as a Model Context Protocol tool
- **GitHub CLI**: `gh extension install town-elder` for GitHub integration
- **Pre-commit**: `te` as a pre-commit hook for arch review
