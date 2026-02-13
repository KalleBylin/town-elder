# DevEx Analysis: Replay for AI Coding Agents

## Executive Summary

Replay is a semantic version control system that addresses the "Context Crisis" in AI coding agents by providing persistent, queryable memory of code history and context. Built on zvec (embedded vector database) and fastembed (local embeddings), it enables AI coding agents to access relevant historical context without manual documentation overhead.

---

## Pain Points Addressed

### 1. Episodic Amnesia in Recursive Loops

**Problem**: AI coding agents operating in "Ralph Wiggum" loops (try fix → run test → fail → retry) have no memory of previous attempts. An agent might try Fix A, see Error B, then try Fix C, see Error A, then revert to Fix A—cycling infinitely because it forgot Fix A caused Error B.

**How Replay Helps**: By indexing each code change and its outcome, Replay allows agents to query "What have I tried before?" and detect when they're oscillating between failed approaches.

### 2. Tribal Knowledge Gap

**Problem**: Senior engineers carry implicit knowledge: "We use this polling mechanism because the legacy API doesn't support webhooks." Agents see the "weird polling code" and try to "fix" it, breaking integrations.

**How Replay Helps**: Replay indexes commit messages and code changes semantically. An agent can query "Why was the retry logic changed?" and retrieve the original context: "Fixed race condition by adding exponential backoff."

### 3. Architectural Drift / "AI Slop"

**Problem**: High-capability models generate working but inconsistent code. An agent might introduce axios when the project strictly enforces a custom FetchWrapper, accumulating technical debt.

**How Replay Helps**: Replay can identify patterns from historical changes, flagging new code that semantically diverges from established patterns.

### 4. Expensive Context Reconstruction

**Problem**: Every session, agents must reconstruct context from static files—expensive in tokens and prone to hallucination.

**How Replay Helps**: Agents query the semantic index to retrieve relevant prior work instantly, rather than parsing entire repositories.

### 5. Static AGENTS.md Rot

**Problem**: Manually maintained context files become stale; agents rely on outdated instructions.

**How Replay Helps**: Enables dynamic, query-time context injection instead of static documentation.

---

## Integration with Agent Workflows

### Claude Code / Claude Desktop

- **Pre-commit hook**: After each agent commit, Replay indexes the diff
- **On-demand query**: Agent runs `replay query "fixing auth bug"` to retrieve related historical context
- **MCP integration**: Exposes Replay as Model Context Protocol tools

### Cursor / Windsurf

- **Tab-aware context**: As agent opens files, Replay surfaces related past changes
- **Agent history**: Query "what did I try in this file previously?" during debugging
- **Semantic search**: Find relevant code across the repository using natural language

### Custom CLI Agents

- **Loop integration**: Insert plan/error pairs into Replay during recursive loops
- **Oscillation detection**: Query similarity before attempting a fix
- **Ephemeral session memory**: Temporary collections for long-running tasks

### Git Hook Integration

```bash
# .git/hooks/post-commit
replay index --commit $COMMIT_HASH

# .git/hooks/prepare-commit-msg
replay suggest --staged  # Suggest related context before commit message
```

---

## Success Metrics

### Primary Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Context Retrieval Relevance** | Top-5 recall > 80% | Human eval of query results |
| **Query Latency** | < 50ms p95 | Local benchmarks |
| **Index Throughput** | > 1000 chunks/sec | Model2Vec benchmark |
| **Agent Loop Efficiency** | 30% fewer iterations | A/B testing in recursive tasks |

### Secondary Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Token Savings** | > 40% reduction | Compare prompt sizes with/without context |
| **First-Attempt Success** | > 25% improvement | Success rate in fixed-duration tasks |
| **Oscillation Detection Rate** | > 90% | Identified repeated failures |
| **Setup Time** | < 5 minutes | Fresh clone to first query |

### Observability

- **Per-query feedback**: After each retrieval, agent rates relevance (implicit via re-query)
- **Session telemetry**: Aggregate loop iteration counts, context utilization rates
- **Index freshness**: Track commit-to-index latency

---

## Competitive Analysis

### vs. Static AGENTS.md

| Dimension | Static AGENTS.md | Replay |
|-----------|-----------------|--------|
| Maintenance | Manual, prone to rot | Automatic via git hooks |
| Query capability | Full-file dump | Semantic retrieval |
| Context precision | All-or-nothing | Relevant subset only |
| Scalability | Degrades with repo size | O(log n) query time |

### vs. Cloud RAG (Pinecone, Weaviate, Qdrant)

| Dimension | Cloud RAG | Replay (zvec + fastembed) |
|-----------|-----------|---------------------------|
| Deployment | Requires Docker/Cloud | pip install, runs in-process |
| Latency | Network overhead | < 1ms local |
| Privacy | Data leaves machine | Zero data exfiltration |
| Cost | API + hosting | Local compute only |
| Offline | Requires connectivity | Fully offline |

### vs. Code Search (Sourcegraph, GitHub Code Search)

| Dimension | Code Search | Replay |
|-----------|-------------|--------|
| Search type | Keyword + regex | Semantic vectors |
| History indexing | Partial | Full commit + diff |
| Agent integration | API-dependent | Local library |
| Query granularity | File/function | Code chunks |

### vs. Session Memory (short-term context windows)

| Dimension | Context Windows | Replay |
|-----------|-----------------|--------|
| Persistence | Lost per session | Retained across sessions |
| Capacity | Fixed token limit | Vector compression |
| Recall | No | Historical queries |

---

## Value Proposition Summary

> **Replay transforms AI coding agents from forgetful task-runners into context-aware engineering partners by providing semantic memory of code history, without the operational overhead of server-based infrastructure.**

### Key Differentiators

1. **Local-First**: No external services, Docker, or cloud dependencies
2. **Zero-Ops**: Single pip install, embedded in the agent's process
3. **Hermetic Stack**: Complete memory system fits alongside local LLMs (Ollama, llama.cpp)
4. **Git-Native**: Indexes directly from git workflow, no migration needed
5. **Semantic Over Symbolic**: Understands "why" not just "what"

---

## Implementation Priorities

### Phase 1: Core Value (Immediate)

- [ ] CLI with `replay index` (post-commit hook)
- [ ] CLI with `replay query` (semantic search)
- [ ] Chunking strategy: Tree-sitter for code parsing
- [ ] Default embedding: fastembed + bge-small

### Phase 2: Agent Integration

- [ ] MCP server exposing query/index tools
- [ ] Cursor/Windsurf extension
- [ ] LoopMem utility for oscillation detection

### Phase 3: Advanced Features

- [ ] ArchGuard: Semantic drift detection
- [ ] TribalSync: Dynamic AGENTS.md generation
- [ ] z-test: Semantic test selection

---

## Risks and Considerations

| Risk | Mitigation |
|------|------------|
| Embedding quality on code | Fine-tune on language-specific corpora; benchmark against code search baselines |
| Index bloat | Implement TTL, archive old commits to cold storage |
| Security (malicious commits) | Sandboxing for embedding generation; input validation |
| Compatibility with agent frameworks | Standard MCP interface; broad Python/Node support |

---

## Conclusion

Replay addresses a fundamental limitation in AI coding agents: the lack of persistent, queryable memory. By leveraging zvec's embedded performance and fastembed's efficiency, it provides a zero-operational-overhead solution that keeps context local and private.

The value is clearest in high-volume agent workflows:
- Debugging loops that previously oscillated now converge faster
- Refactoring tasks that previously broke tribal knowledge now respect history
- Onboarding new agents that previously started from scratch now query accumulated context

For teams running local-first AI coding workflows, Replay completes the "Hermetic Stack"—where intelligence (local LLM), memory (zvec), and structure (Beads) all run on the developer's machine without external dependencies.
