# Product Direction: te

**Version Control for AI Agents**

te is a semantic version control system that helps AI coding agents retrieve context from git history using semantic search. Built on zvec (embedded vector database) and fastembed (local embeddings), te operates entirely locally—no cloud services, no external APIs, no operational overhead.

---

## Target Users

### Primary Users

1. **AI Coding Agents** (Autonomous agents like Claude Code, Cursor, Devin)
   - Agents that operate in long-horizon tasks requiring context from past changes
   - Agents that need to understand "why" something was changed, not just "what" changed

2. **Developers Using AI Agents**
   - Developers who use AI agents for code review, refactoring, or bug fixing
   - Teams wanting to provide agents with tribal knowledge from their repository history

### Secondary Users

- **Individual developers** wanting semantic search over their personal git history
- **Development teams** wanting to index and search across team repositories

---

## Use Cases

### Core Use Cases

1. **Semantic Search of Git History**
   - Query: "Find commits related to authentication retry logic"
   - Instead of keyword matching, te retrieves semantically similar commits
   - Understands conceptual relationships (e.g., "retry" relates to "backoff", "rate limiting")

2. **Finding Related Commits**
   - Given a code change or error, find historically related commits
   - Useful for understanding the context of a bug or feature

3. **Understanding Code Changes**
   - Retrieve the "why" behind code changes via embedded commit messages
   - Link commit intent to implementation details

### Agent-Specific Use Cases

4. **Context Injection for Agent Tasks**
   - Automatically retrieve relevant history when an agent starts a task
   - Populate agent context with tribal knowledge ("Why we use this pattern")

5. **Oscillation Detection** (Future)
   - Detect when an agent is repeating failed attempts
   - Query historical attempts to prevent redundant work

6. **Architectural Pattern Retrieval** (Future)
   - Find examples of similar patterns in the codebase history
   - Maintain consistency with past architectural decisions

---

## MVP Scope

### What to Build First

**Phase 1: Core Semantic Indexing**

- [ ] Git hook integration (post-commit, post-merge)
- [ ] Parse git diffs into semantic chunks (functions, classes)
- [ ] Generate embeddings using fastembed (local, no API)
- [ ] Store in zvec (embedded, in-process)
- [ ] CLI for semantic search queries

**Phase 2: Basic Retrieval**

- [ ] Query interface (CLI)
- [ ] Return commit messages + diffs matching semantic query
- [ ] Basic filtering (date range, author, file path)

**Phase 3: Agent Integration**

- [ ] Simple API for agents to query
- [ ] Context formatting for agent prompts

### What to Defer

- Multi-repository indexing (Phase 2+)
- Semantic blame (future)
- Oscillation detection (future)
- Web interface / GUI (future)
- Cloud hosting (never - stay local-first)

---

## Technical Architecture

### Stack

- **Vector Database**: zvec (embedded, in-process)
- **Embeddings**: fastembed (quantized ONNX, runs locally)
- **Indexing**: Tree-sitter for code chunking
- **Storage**: Local files in `.git/te/`

### Design Principles

1. **Local-First**: All data stays on the developer's machine
2. **Zero-Ops**: No Docker, no external services, no cloud dependencies
3. **Agent-Native**: Designed for programmatic access by AI agents
4. **Minimal Footprint**: Runs alongside local LLMs without resource competition

---

## Roadmap

### Phase 1: The Indexer (Months 1-2)

**Goal**: Enable semantic search of git history

- Post-commit hook that auto-indexes new commits
- CLI tool: `te search "authentication retry logic"`
- Returns matching commits with diffs
- Support for major languages (Python, JS, Go, Rust)

**Milestone**: Developer can find relevant commits using natural language queries

### Phase 2: The Context Engine (Months 3-4)

**Goal**: Provide context to AI agents

- Simple HTTP API or CLI output formatted for agent prompts
- Integration hooks for agent frameworks
- Filter by file path, date range, author

**Milestone**: Agent receives relevant history when starting a task

### Phase 3: The Memory Layer (Months 5-6)

**Goal**: Expand beyond commits to full code context

- Index entire file contents (not just diffs)
- Index PR descriptions, issues, PR comments
- Support for multi-file semantic queries

**Milestone**: Agent can query entire codebase history semantically

### Future Considerations

- Semantic blame: Find who changed code related to a concept
- Oscillation detection: Prevent agent from repeating failed attempts
- Pattern enforcement: Detect architectural drift

---

## Success Criteria

### Adoption Metrics

1. **Installation Count**: Downloads via pip
2. **Active Repositories**: Number of repos with te index initialized
3. **Query Volume**: Number of semantic queries executed

### Engagement Metrics

1. **Query Return Rate**: % of queries that return relevant results
2. **Agent Integration**: Number of agents/tools integrated
3. **Community Contributions**: Stars, forks, PRs

### Impact Metrics

1. **Context Reduction**: Tokens saved in agent prompts by using semantic retrieval
2. **Task Success**: User-reported improvement in agent task completion
3. **Developer Feedback**: Qualitative feedback on usefulness

---

## Pricing Model

### Open Source (MIT License)

te will be free and open source under the MIT license.

**Rationale**:
- Local-first tools benefit from community trust
- No cloud infrastructure costs to recover
- Maximize adoption in developer community
- Enable integration into agent frameworks

### Revenue (Future, Optional)

If commercial support is needed:

1. **Hosted Indexing Service**: For teams wanting cloud-managed indexing
2. **Enterprise Features**: Team dashboards, audit logs, SSO
3. **Custom Integrations**: Paid support for enterprise agent platforms

For now, focus entirely on open source adoption.

---

## Competitive Landscape

### Existing Solutions

| Product | Approach | Limitation |
|---------|----------|-------------|
| GitHub Code Search | Keyword-based | No semantic understanding |
| GitHub Copilot | Code completion | No historical context |
| Pinecone/Weaviate | Cloud vector DB | Requires cloud, not local |
| grep.app | Regex search | No semantic search |

### te's Differentiation

- **Truly Local**: No cloud dependency unlike Pinecone/Weaviate
- **Git-Native**: Built specifically for git history semantics
- **Agent-Ready**: Designed for programmatic agent access
- **Zero-Ops**: No deployment complexity

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| zvec adoption stalls | Design for swapability; zvec is an implementation detail |
| Embedding quality insufficient | Support multiple embedding models; allow configuration |
| Agent frameworks don't integrate | Build simple, well-documented API; reach out to framework maintainers |
| Performance issues at scale | Optimize chunking; add incremental indexing |

---

## Conclusion

te addresses a fundamental problem in AI-assisted development: agent amnesia. By making git history semantically searchable, we enable agents to understand the "why" behind code, not just the "what".

Our local-first, zero-ops approach removes the friction of cloud infrastructure and privacy concerns. te is built for the era of autonomous agents—where memory is infrastructure, and context is king.
