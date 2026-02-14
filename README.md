# Replay

Local-first semantic memory for AI coding agents. Index your codebase and git history for intelligent search.

## Quick Start

```bash
# Initialize replay in your project
replay init

# Index your code files
replay index

# Search semantically
replay search "authentication logic"
```

## Installation

### From PyPI

```bash
pip install replay
```

### From Source

```bash
git clone https://github.com/yourusername/replay.git
cd replay
pip install -e .
```

## Usage

### A) Fresh Project (New Repository)

Start from scratch and build semantic memory as you develop.

```bash
# Initialize the database in your project
$ replay init
Initialized replay database at /path/to/project/.replay

# Index your codebase (Python and Markdown files)
$ replay index
Indexing 42 files...
Indexed 42 files

# Add contextual notes for AI agents
$ replay add -t "Use FastAPI for REST APIs, Django for full-stack apps" \
  -m '{"source": "architecture-guidelines"}'

# Search your semantic memory
$ replay search "API framework"
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

# Initialize replay
$ replay init

# Index all commits (last 100 by default)
$ replay commit-index
Indexing 100 commits...
Indexed 100 commits

# Or limit to a specific number
$ replay commit-index --limit 50

# Search git history semantically
$ replay search "payment retry bug"
Search results for: payment retry bug

1. Score: 0.923
   Commit: Fixed race condition in retry logic by adding exponential backoff

   diff --git a/payment.py b/payment.py
   -  retry_count = 3
   +  retry_count = 5
   +  sleep_factor = 2  # exponential backoff

# Install automatic indexing on every commit
$ replay hook install
Installed post-commit hook at /path/to/.git/hooks/post-commit
Commits will now be automatically indexed
```

## Commands Reference

| Command | Description |
|---------|-------------|
| `replay init` | Initialize a replay database in the current directory |
| `replay add` | Add a document with optional metadata |
| `replay index` | Index all `.py` and `.md` files in a directory |
| `replay search` | Search indexed documents semantically |
| `replay stats` | Show document count and configuration |
| `replay commit-index` | Index git commits from a repository |
| `replay hook install` | Install post-commit hook for automatic indexing |
| `replay hook uninstall` | Remove post-commit hook |
| `replay hook status` | Check if hook is installed |

### Options

- `--path`, `-p`: Specify directory path (for init, index)
- `--top-k`, `-k`: Number of search results (default: 5)
- `--limit`, `-n`: Number of commits to index (default: 100)
- `--repo`, `-r`: Git repository path (for commit-index, hook commands)
- `--force`, `-f`: Overwrite existing data
- `--text`, `-t`: Text content to add
- `--metadata`, `-m`: JSON metadata string

## Configuration

Replay stores configuration and data in a `.replay` directory in your project:

```
your-project/
├── .replay/
│   ├── vectors/      # Vector database files
│   └── config.json   # Configuration
└── .git/
```

### Default Settings

- **Embedding model**: Fastembed (BAAI/bge-small-en-v1.5, 384 dimensions)
- **Data location**: `.replay/` in your project directory
- **Indexed file types**: `.py` and `.md` files

Configuration is managed via environment variables or `pyproject.toml`. See `replay.config` for available options.
