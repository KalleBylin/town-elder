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
