# Town Elder

Town Elder is a local-first semantic memory CLI for AI coding agents.

Like a town elder, it helps your tools remember what your project has already learned and why past decisions were made.

It helps you recover project context when exact keywords are unknown:

- Index source files and notes (`index files`, `add`)
- Index commit messages and diffs (`index commits`)
- Search everything with natural language (`search`)

Town Elder runs locally and stores project memory in `.town_elder`, so you can query context without relying on an external retrieval service. It uses `zvec` as an embedded vector database, which means there is no separate vector DB server to deploy or operate.

## Core Value

Town Elder is useful when it reliably delivers these outcomes:

1. Find likely code and notes quickly when you do not know exact names or paths.
2. Recover intent from git history by searching commit messages and diffs semantically.
3. Get local semantic search powered by embedded `zvec`, with explicit data-dir control for predictable multi-repo use.

## Quick Start

```bash
# Initialize Town Elder in your project (using uv)
uv run te init

# Index your code files (full file indexing)
uv run te index --all

# Search semantically
uv run te search "authentication logic"
```

Output example:

```text
Search results for: authentication logic

1. Score: 0.912
   src/auth/session.py
   validate_token() checks token expiration and issuer before loading user context.

2. Score: 0.857
   Commit: tighten auth retry flow for expired refresh tokens
```

Ad-hoc usage (no dependency setup in the current project, for example a Poetry repo):

```bash
uvx --from town-elder te init
uvx --from town-elder te search "authentication logic"
```

## How To Use This Day-To-Day

Use Town Elder as your semantic first pass, then use exact tools to confirm and implement.

### Decision Rule

- Start with **Town Elder** when you do not know exact keywords, symbol names, or commit hashes.
- Switch to **`rg` and `git`** after Town Elder identifies likely files and commits.

### Typical Workflow

1. From your repo root, initialize once:
```bash
uv run te init
```
2. Build memory from code and commit history:
```bash
uv run te index --all
uv run te index commits --limit 200
```
3. Ask intent-level questions:
```bash
uv run te search "data-dir leakage across CLI invocations"
uv run te search "last indexed commit not found stale state"
```
4. Confirm and implement with exact tools:
```bash
rg -n "data-dir|sentinel|last_indexed_commit" src tests
git log --oneline --grep="data-dir leakage"
git show <commit-hash>
```
5. Save new project knowledge for future sessions:
```bash
uv run te add \
  --text "Safety rule: never delete non-Town Elder hooks unless --force" \
  --metadata '{"source":"engineering-note","topic":"hook-safety"}'
```
6. Keep history memory fresh automatically:
```bash
uv run te hook install
```

Use `--data-dir` when you want storage outside the current repo (for example, a temporary isolated session or operations run from another working directory).

Typical questions for `te search`: "why was retry logic changed?", "what fixed the stale state bug?", "where is initialization failure handled?"

## Installation

### From PyPI

```bash
pip install town-elder
```

### From Source

```bash
git clone https://github.com/KalleBylin/town-elder.git
cd town-elder
uv pip install -e .
```

## Running with uv or uvx

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

- Use `uv run te <command>` when working from a local checkout/environment.
- Use `uvx --from town-elder te <command>` for one-off usage without changing project dependencies.

```bash
# Run Town Elder commands from the project root
uv run te <command>

# Run without modifying the current project's dependency config
uvx --from town-elder te <command>

# Operate on a different repository (using --data-dir)
uv run te --data-dir /path/to/target-repo/.town_elder <command>
uvx --from town-elder te --data-dir /path/to/target-repo/.town_elder <command>
```

For example, to index commits in another repository:

```bash
uv run te --data-dir /path/to/target-repo/.town_elder init --path /path/to/target-repo
uv run te --data-dir /path/to/target-repo/.town_elder index commits --repo /path/to/target-repo
```

## Usage

### A) Fresh Project (New Repository)

Start from scratch and build semantic memory as you develop.

```bash
# Initialize the database in your project
$ uv run te init
Initialized Town Elder database at /path/to/project/.town_elder

# Index your codebase (Python and Markdown files)
$ uv run te index --all
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
$ uv run te index commits
Indexing 100 commits...
Indexed 100 commits

# Or limit to a specific number
$ uv run te index commits --limit 50

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

Commands are shown with `uv run te`. You can use `uvx --from town-elder te` equivalently.

| Command | Description |
|---------|-------------|
| `uv run te init` | Initialize a Town Elder database in the current directory |
| `uv run te add` | Add a document with optional metadata |
| `uv run te index --all` | Index all `.py` and `.md` files in current directory (full repository file indexing) |
| `uv run te index files [PATH]` | Index all `.py` and `.md` files in a specific directory |
| `uv run te index commits` | Index git commits from a repository |
| `uv run te search` | Search indexed documents semantically |
| `uv run te stats` | Show document count and configuration |
| `uv run te hook install` | Install post-commit hook for automatic indexing |
| `uv run te hook uninstall` | Remove post-commit hook |
| `uv run te hook status` | Check if hook is installed |

### Options

- `--data-dir`, `-d`: Data directory (default: .town_elder in current directory)
- `--path`, `-p`: Specify directory path (for init)
- `index files [PATH]`: Positional directory path for file indexing (default: current directory)
- `index --all`: Index all files in current directory (equivalent to `te index files .`)
- `--top-k`, `-k`: Number of search results (default: 5)
- `--limit`, `-n`: Number of commits to index (default: 100)
- `--repo`, `-r`: Git repository path (for index commits, hook commands)
- `--force`, `-f`: Overwrite existing data
- `--text`, `-t`: Text content to add
- `--metadata`, `-m`: JSON metadata string

## Configuration

Town Elder stores data in a `.town_elder` directory in your project:

```
your-project/
├── .town_elder/
│   ├── vectors/         # Vector database files
│   └── index_state.json # Index state for incremental updates
└── .git/
```

### Default Settings

- **Embedding model**: Fastembed (BAAI/bge-small-en-v1.5, 384 dimensions)
- **Data location**: `.town_elder/` in your project directory
- **Indexed file types**: `.py` and `.md` files

Configuration is managed via environment variables or `pyproject.toml`. See `town_elder.config` for available options.

## Troubleshooting

### First-Run Model Download

On first use, Town Elder downloads an embedding model from HuggingFace. This may take some time depending on your internet connection.

**What to expect:**
- Initial commands (`te index files`, `te search`, `te add`, `te index commits`) may take 30-60 seconds on first run
- The model (~100MB) is downloaded once and cached locally
- Subsequent runs will be fast

**Common issues:**

| Issue | Solution |
|-------|----------|
| Slow first run | Normal - model is being downloaded. Wait for completion. |
| "Failed to load embedding backend" | Install fastembed: `pip install fastembed` or `uv pip install fastembed` |
| Network error during download | Ensure internet access to HuggingFace (huggingface.co). Check proxy settings if behind a firewall. |
| Model not found | Ensure fastembed is installed: `pip show fastembed` |

### Hook Prerequisites

The post-commit hook automatically indexes commits after each `git commit`. For hooks to work correctly:

**Requirements:**
1. **Git repository**: Must run in a directory with a `.git` folder
2. **Python, uv, or uvx**: The hook uses a fallback chain (`uv run te` → `uvx --from town-elder te` → `te` → `python -m town_elder`)
3. **Initialized database**: Run `te init` before installing hooks

**Hook fallback chain:**
```sh
# First tries uv
command -v uv >/dev/null 2>&1 && uv run te index commits ... && exit
# Then tries uvx
command -v uvx >/dev/null 2>&1 && uvx --from town-elder te index commits ... && exit
# Then tries te command
command -v te >/dev/null 2>&1 && te index commits ... && exit
# Finally falls back to python module
python -m town_elder index commits ...
```

**Common issues:**

| Issue | Solution |
|-------|----------|
| Commits not being indexed | Check hook status: `te hook status` |
| Hook not found | Install it: `te hook install` |
| "command not found" errors | Ensure `uv`, `uvx`, `te`, or Python is on your PATH |
| Hook runs but nothing happens | Ensure database is initialized: `te init` first |
| uv not found | Install uv, use `uvx --from town-elder te ...`, or ensure `te` is on your PATH |

**Verifying hook installation:**
```bash
# Check if hook is installed
te hook status

# View installed hook content
cat .git/hooks/post-commit
```
