---
name: town-elder-agent-memory
description: Guides agents to use Town Elder for semantic codebase and git-history recall. Use when the user asks to find context, recover why changes were made, search intent without exact keywords, or maintain local project memory. Do NOT use for simple exact-match lookups where rg, grep or direct git commands are sufficient.
license: MIT
metadata:
  author: town-elder
  version: 1.0.0
---

# Town Elder Agent Memory

## Purpose
Use Town Elder as a semantic retrieval layer before exact tools.

Town Elder is most useful when:
- The user does not know exact symbol names, file paths, or commit hashes.
- The task requires recovering change intent ("why"), not only current code state.
- The agent should persist team/project notes for future runs.

Use `rg`, `git log`, and `git show` after Town Elder returns likely targets.

## Command Rules
- Prefer `uv run te ...` in a local checkout/environment.
- Use `uvx --from town-elder te ...` for ad-hoc usage when you do not want to change the current project's dependency config.
- In repo-root workflows, `--data-dir` is optional after `te init` because Town Elder uses `./.town_elder`.
- Use `--data-dir` only when storage should live outside the current repo (isolated sessions, cross-repo operations, CI temp dirs).
- If commands fail with "Database not initialized", run `te init` in the target repo (via either `uv run te` or `uvx --from town-elder te`) or initialize with an explicit `--data-dir`.

## Standard Workflow
1. Initialize once in the repo:

```bash
uv run te init
```

2. Build semantic memory:

```bash
uv run te index files
uv run te index commits --limit 200
```

3. Ask semantic questions:

```bash
uv run te search "<intent query>"
```

4. Pivot to exact tools for implementation:

```bash
rg -n "<keywords from te results>" src tests
git log --oneline --grep="<topic from te results>"
git show <commit-hash>
```

5. Save durable team knowledge when useful:

```bash
uv run te add --text "<rule or context note>" --metadata '{"source":"agent-note"}'
```

6. Keep commit memory current automatically (optional):

```bash
uv run te hook install
```

## Query Patterns
Use `te search` with intent-level prompts, for example:
- "why was retry logic changed"
- "what fixed stale state in indexing"
- "where is database not initialized handled"
- "hook safety behavior for non-TE hooks"

## Task Playbooks
### Understand a bug before editing
1. `uv run te search "<bug description>"`
2. Open likely files/commits with `rg` and `git show`.
3. Implement change only after confirming exact locations.

### Recover historical rationale
1. Ensure commit memory exists: `uv run te index commits` (incremental by default).
2. Search rationale: `uv run te search "<why question>"`.
3. Validate with `git log --oneline` and `git show`.

### Work across multiple repositories
1. Use repo-specific data dirs to avoid mixing memories.
2. Example:

```bash
uv run te --data-dir /tmp/te-repo-a init --path /path/to/repo-a
uv run te --data-dir /tmp/te-repo-a index commits --repo /path/to/repo-a
uv run te --data-dir /tmp/te-repo-a search "authentication rollback reason"
```

## Guardrails
- Do not claim semantic results are definitive; treat them as retrieval candidates.
- Always verify with source files, diffs, and tests before code changes.
- Prefer concise user-facing summaries of retrieved context plus exact follow-up paths/commits.
- Avoid overwriting or uninstalling existing non-Town-Elder hooks unless explicitly requested.
