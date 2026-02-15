# Documentation Plan for te

## Project Overview
te is a CLI tool for semantic git retrieval using zvec (embedded vector database) + fastembed (local embeddings). It enables searching git history by meaning rather than just keywords.

---

## Recommended Tool: Simple Markdown with MkDocs

**Choice**: MkDocs with Material theme

**Rationale**:
- Simple markdown files as source (low barrier to contribution)
- Material theme provides modern, clean look out of the box
- Easy to deploy to GitHub Pages
- Supports API documentation via plugin
- Minimal configuration required

**Alternative considered**: Sphinx - too complex for this project size

---

## Documentation Structure

```
docs/
├── index.md                 # Landing page
├── quickstart.md            # 5-minute getting started
├── installation.md          # Detailed installation options
├── usage/
│   ├── index.md             # CLI overview
│   ├── query.md             # Semantic search commands
│   ├── index-flow.md        # Indexing repositories
│   └── git-hooks.md        # Automated indexing
├── examples/
│   ├── index.md             # Example gallery overview
│   └── ...                  # Common use cases
├── architecture/
│   ├── index.md             # High-level overview
│   ├── components.md        # zvec, fastembed, CLI
│   └── data-flow.md        # How indexing/search works
├── reference/
│   ├── cli.md               # Full CLI command reference
│   └── config.md            # Configuration options
└── contributing.md          # Development guide
```

---

## 1. README Structure (repo root)

The README should be a single file with essential info:

```markdown
# te

Semantic git retrieval using zvec + fastembed.

## Quick Start

[3-4 line installation + basic usage]

## Features

- Semantic search through git history
- Local-first (no external services)
- Git hook integration for automatic indexing

## Installation

`pip install town-elder`

[Link to full docs]

## Usage

$ te query "fix authentication bug"
[show example output]

## Documentation

[Link to docs site]
```

**Location**: `/Users/bylin/Code/town_elder/README.md`

---

## 2. Usage Guides

### 2.1 Query Flow (`usage/query.md`)
- Basic semantic search syntax
- Filtering by date, author, file path
- Output formats (JSON, plain text)
- Top-k results configuration

### 2.2 Index Flow (`usage/index-flow.md`)
- Initializing a repository index
- Indexing existing commit history
- Incremental updates
- Index storage location (`.git/te/`)

### 2.3 Git Hooks (`usage/git-hooks.md`)
- Setting up post-commit hooks
- Automatic indexing on push
- Configuration options
- Troubleshooting hook failures

---

## 3. API Documentation

### 3.1 CLI Reference (`reference/cli.md`)
Complete command reference with examples:

```markdown
## te init

Initialize semantic index for repository.

### Options
- `--model TEXT`    Embedding model (default: BAAI/bge-small-en-v1.5)
- `--batch-size INT`  Indexing batch size (default: 32)

### Example
$ te init --model BAAI/bge-small-en-v1.5

## te query

Search commit history semantically.

### Options
- `-k, --top-k INT`    Number of results (default: 5)
- `--author TEXT`      Filter by author
- `--since TEXT`       Filter by date
- `--format FORMAT`    Output format: json, text

### Example
$ te query "race condition in retry logic" -k 10
```

---

## 4. Architecture Docs

### 4.1 High-Level Overview (`architecture/index.md`)
- What te does and why
- Core concepts: semantic search, vector embeddings
- When to use te vs. standard git search

### 4.2 Components (`architecture/components.md`)
- **zvec**: Embedded vector database, Proxima engine
- **fastembed**: Local ONNX-based embedding generation
- **CLI**: Python click/argparse interface

### 4.3 Data Flow (`architecture/data-flow.md`)
- Diagram: Commit → Parse → Embed → Store → Query
- Index file format and location
- Query execution path

---

## 5. Style Guidelines

**Tone**: Technical but accessible
- Clear and straightforward
- Assume developer audience but explain concepts briefly
- Avoid unnecessary jargon

**Formatting**:
- Code blocks with language hints
- Shell prompts for commands (`$`, `>`)
- Bold for UI elements and commands
- Link to external docs for dependencies

**Examples**:
- Show realistic, practical examples
- Include expected output
- Brief explanations, not lengthy tutorials

---

## 6. Implementation Notes

### Initial Documentation Priority
1. README.md (essential)
2. Quick Start guide
3. CLI reference
4. Installation guide

### Later Additions
- Architecture deep-dives
- Advanced usage patterns
- Contributing guide

### Consider Adding Later
- API documentation (if library mode is added)
- Tutorial videos or animated examples
- Migration guide from standard git search

---

## 7. Future Considerations

- **Versioning**: Add version notes when API changes
- **Internationalization**: Not initially needed
- **Multi-language**: Focus on English initially
- **Search**: Add search functionality to docs site if content grows

---

## Summary

| Section | File | Priority |
|---------|------|----------|
| Landing | docs/index.md | High |
| Quick Start | docs/quickstart.md | High |
| README | README.md | High |
| CLI Reference | docs/reference/cli.md | High |
| Installation | docs/installation.md | Medium |
| Query Guide | docs/usage/query.md | Medium |
| Index Guide | docs/usage/index-flow.md | Medium |
| Git Hooks | docs/usage/git-hooks.md | Medium |
| Architecture | docs/architecture/*.md | Low |
