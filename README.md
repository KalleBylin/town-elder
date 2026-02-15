# Town Elder

Local-first semantic memory for AI coding agents. Index your codebase and git history for intelligent search.

## Value Invariant

Town Elder is only valuable if it reliably provides all three outcomes below:

1. Fast local memory for ad-hoc context retrieval (`add`/`search`) without external services.
2. Semantic commit-history recall (`commit-index`/`search`) to recover the "why" behind changes.
3. Safe multi-repo operation (data-dir isolation, predictable hooks, and trustworthy exports/errors).

Any change that weakens one of these outcomes is a regression and should be treated as a release blocker.

## Quick Start

```bash
# Initialize Town Elder in your project (using uv)
uv run te init

# Index your code files
uv run te index

# Search semantically
uv run te search "authentication logic"
```

## How To Use This Day-To-Day

Town Elder is not a replacement for `grep` or raw `git` commands.
It is a semantic recall layer you use before exact tools.

### When to use Town Elder vs `grep`/`git`

- Use **Town Elder** first when you do not know exact keywords, symbol names, or commit hashes.
  - Examples: "why was retry logic changed?", "what fixed the stale state bug?", "where do we handle not-initialized UX errors?"
- Use **`rg` and `git`** after Town Elder finds likely targets.
  - Examples: open exact lines, inspect specific diffs, edit code, run tests.

### Recommended Agent Workflow

1. Initialize (optionally isolated data dir for experiments):
```bash
uv run te --data-dir /tmp/te-session init --path /path/to/repo
```
2. Build memory from current code + history:
```bash
uv run te --data-dir /tmp/te-session index /path/to/repo
uv run te --data-dir /tmp/te-session commit-index --repo /path/to/repo --limit 200
```
3. Ask intent-level questions:
```bash
uv run te --data-dir /tmp/te-session search "data-dir leakage across CLI invocations"
uv run te --data-dir /tmp/te-session search "last indexed commit not found stale state"
```
4. Pivot to exact tools to implement changes:
```bash
rg -n "data-dir|sentinel|last_indexed_commit" src tests
git log --oneline --grep="data-dir leakage"
git show <commit-hash>
```
5. Save new project knowledge for later runs:
```bash
uv run te --data-dir /tmp/te-session add \
  --text "Safety rule: never delete non-Town Elder hooks unless --force" \
  --metadata '{"source":"engineering-note","topic":"hook-safety"}'
```
6. Keep history memory fresh automatically:
```bash
uv run te --data-dir /tmp/te-session hook install --repo /path/to/repo
```

### Practical Coding Task Examples

1. **Understand a regression before touching code**
```bash
uv run te search "friendly error when database is not initialized"
```
Then:
```bash
rg -n "Database not initialized|ConfigError" src tests
```

2. **Find historical intent behind a bugfix**
```bash
uv run te search "fix incremental backlog loss when last indexed commit missing"
```
Then:
```bash
git log --oneline --grep="last indexed commit"
git show <commit-hash>
```

3. **Capture team rules that are not obvious from code**
```bash
uv run te add --text "Do not remove non-TE git hooks unless --force is explicit"
uv run te search "rule for deleting git hooks"
```

## Installation

### From PyPI

```bash
pip install town-elder
```

### From Source

```bash
git clone https://github.com/yourusername/town-elder.git
cd town-elder
pip install -e .
```

## Running with uv

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. All commands should be run with `uv run`:

```bash
# Run Town Elder commands from the project root
uv run te <command>

# Operate on a different repository (using --data-dir)
uv run te --data-dir /path/to/target-repo/.town_elder <command>
```

For example, to index commits in another repository:

```bash
uv run te --data-dir /path/to/target-repo/.town_elder init --path /path/to/target-repo
uv run te --data-dir /path/to/target-repo/.town_elder commit-index --repo /path/to/target-repo
```

## Usage

### A) Fresh Project (New Repository)

Start from scratch and build semantic memory as you develop.

```bash
# Initialize the database in your project
$ uv run te init
Initialized Town Elder database at /path/to/project/.town_elder

# Index your codebase (Python and Markdown files)
$ uv run te index
Indexing 42 files...
Indexed 42 files

# Add contextual notes for AI agents
$ uv run te add -t "Use FastAPI for REST APIs, Django for full-stack apps" \
  -m '{"source": "architecture-guidelines"}'

# Search your semantic memory
$ uv run te search "API framework"
Search results for: API framework

1. Score: 0.892
   Use FastAPI for REST APIs, Django for full-stack apps...

2. Score: 0.745
   # API Documentation
   This module provides REST endpoints for...
```

### B) Existing Project (With History)

Unlock tribal knowledge from your git history.

```bash
# Navigate to your existing repository
$ cd /path/to/existing-project

# Initialize Town Elder
$ uv run te init

# Index all commits (last 100 by default)
$ uv run te commit-index
Indexing 100 commits...
Indexed 100 commits

# Or limit to a specific number
$ uv run te commit-index --limit 50

# Search git history semantically
$ uv run te search "payment retry bug"
Search results for: payment retry bug

1. Score: 0.923
   Commit: Fixed race condition in retry logic by adding exponential backoff

   diff --git a/payment.py b/payment.py
   -  retry_count = 3
   +  retry_count = 5
   +  sleep_factor = 2  # exponential backoff

# Install automatic indexing on every commit
$ uv run te hook install
Installed post-commit hook at /path/to/.git/hooks/post-commit
Commits will now be automatically indexed
```

## Commands Reference

All commands use `uv run te`:

| Command | Description |
|---------|-------------|
| `uv run te init` | Initialize a Town Elder database in the current directory |
| `uv run te add` | Add a document with optional metadata |
| `uv run te index` | Index all `.py` and `.md` files in a directory |
| `uv run te search` | Search indexed documents semantically |
| `uv run te stats` | Show document count and configuration |
| `uv run te commit-index` | Index git commits from a repository |
| `uv run te hook install` | Install post-commit hook for automatic indexing |
| `uv run te hook uninstall` | Remove post-commit hook |
| `uv run te hook status` | Check if hook is installed |

### Options

- `--data-dir`, `-d`: Data directory (default: .town_elder in current directory)
- `--path`, `-p`: Specify directory path (for init, index)
- `--top-k`, `-k`: Number of search results (default: 5)
- `--limit`, `-n`: Number of commits to index (default: 100)
- `--repo`, `-r`: Git repository path (for commit-index, hook commands)
- `--force`, `-f`: Overwrite existing data
- `--text`, `-t`: Text content to add
- `--metadata`, `-m`: JSON metadata string

## Configuration

Town Elder stores configuration and data in a `.town_elder` directory in your project:

```
your-project/
├── .town_elder/
│   ├── vectors/      # Vector database files
│   └── config.json   # Configuration
└── .git/
```

### Default Settings

- **Embedding model**: Fastembed (BAAI/bge-small-en-v1.5, 384 dimensions)
- **Data location**: `.town_elder/` in your project directory
- **Indexed file types**: `.py` and `.md` files

Configuration is managed via environment variables or `pyproject.toml`. See `town_elder.config` for available options.
