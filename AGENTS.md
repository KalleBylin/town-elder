# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Development

This project uses **uv** for dependency management and **pyenv** for Python version control.

### Setup

```bash
# Install dependencies (uses .python-version and uv.lock)
uv sync

# Run the CLI
uv run te <command>

# Add dependencies
uv add <package>
uv add --dev <package>

# Run tests
uv run pytest
```

### Python Version

Python version is managed by pyenv via `.python-version`. To change:

```bash
pyenv install 3.12    # Install a new version
# Edit .python-version
uv sync               # Re-sync dependencies
```

## Commit Messages (Release Please)

This repository uses **Conventional Commits** so Release Please can determine
semantic version bumps and changelog entries automatically.

**Required commit subject format:**

```text
<type>(<optional-scope>)!: <imperative summary>
```

**Allowed types:**

- `feat` -> minor version bump
- `fix` -> patch version bump
- `perf`, `refactor`, `docs`, `test`, `build`, `ci`, `chore` -> no release bump unless marked breaking

**Breaking changes:**

- Add `!` in the subject (e.g., `feat(cli)!: remove legacy alias`)
- Include a footer:
  `BREAKING CHANGE: <what changed and migration guidance>`

**Examples:**

- `feat(cli): add commit export command`
- `fix(index): handle empty repository state`
- `feat(api)!: replace search payload format`

**PR merge guidance:**

- If using squash merge, the squash commit message **must** follow Conventional
  Commits.
- Keep unrelated changes in separate commits whenever practical.

## Release Automation

- `release-please.yml` monitors `main`, updates changelog/version metadata, and
  creates `v*` tags + GitHub Releases.
- `publish.yml` is triggered by published GitHub Releases and performs build,
  smoke tests, attestations, and PyPI publish via Trusted Publishing.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
