# te Project: Critical Risk Analysis

**Role: Devil's Advocate**
**Date: 2026-02-13**

This document provides a critical analysis of potential issues, risks, and edge cases for the te project (semantic git using zvec + fastembed).

---

## 1. Technical Risks

### 1.1 zvec Stability & Maturity

**Risk:** zvec is a very new library (released early 2026) with limited production track record.

- The Proxima engine is battle-tested at Alibaba scale, but the Python wrapper and zvec-specific implementation are new
- Potential bugs in the FFI layer between Python and C++
- Limited community support andStack Overflow answers
- No established best practices for production deployment

**Mitigation required:** Extensive testing, fallback mechanisms, version pinning

### 1.2 FastEmbed Model Quality

**Risk:** Quantized embedding models have measurable accuracy trade-offs.

- 8-10% accuracy loss vs full-precision models (quantization to INT8)
- Embedding models trained on natural language may poorly understand code semantics
- Variable naming, code patterns, and technical terminology differ from training data
- Model choices (bge-small, all-MiniLM) are compromises between speed and quality

**Mitigation required:** Benchmark on code-specific datasets, not just standard NLP benchmarks

### 1.3 Dependency Churn

**Risk:** The stack involves multiple rapidly-evolving libraries.

- zvec API may change between minor versions
- FastEmbed model updates may break backwards compatibility
- ONNX Runtime version conflicts with other tools (Ollama, etc.)

---

## 2. Architectural Concerns

### 2.1 Chunking Strategy Complexity

**Risk:** How code is split into vectors dramatically affects search quality.

- Tree-sitter parsing is complex and language-specific
- "Logical chunks" (functions, classes) may lose cross-function context
- No clear boundary for semantic units - where does context end?
- Overlapping chunks create redundancy; underlapping lose meaning

**Example failure:** A security vulnerability spans multiple functions across a file. Each function indexed separately loses the attack path.

### 2.2 Embedding Drift

**Risk:** Changing embedding models invalidates existing indexes.

- Upgrading from bge-small-en-v1.5 to bge-small-en-v1.6 produces different vectors
- No migration path for existing semantic history
- Teams may run different versions, creating inconsistent results

**Mitigation required:** Version-locked models, or implement vector re-computation pipeline

### 2.3 Semantic vs Symbolic Gap

**Risk:** Pure semantic search misses exact matches needed in code.

- Query "MaxRetries" should find exact variable, not semantically similar "RetryCount"
- Symbolic (keyword) + semantic (vector) hybrid search is complex to implement correctly
- Filter combinations (file:auth, type:interface) require additional metadata schema

---

## 3. Edge Cases

### 3.1 Large Repository Scaling

**Risk:** Monorepos with 100k+ files create storage and performance challenges.

- Vector storage grows linearly with indexed content
- 10,000 commits × 100 chunks × 384 dimensions × 4 bytes = ~1.5GB per year
- Search latency increases with index size (HNSW helps but not infinite)
- Memory pressure on developer laptops

### 3.2 Binary Files & Non-Text Content

**Risk:** Git repos contain images, compiled binaries, and large files.

- Attempting to embed binary data produces garbage
- Must exclude but determining "text vs binary" is imperfect
- Large text files (generated code, logs) may need special handling

### 3.3 Deleted Code & History Rewrite

**Risk:** Git history rewriting (rebase, amend) creates orphaned vectors.

- Vectors reference commit hashes that no longer exist
- Semantic "blame" returns results for code that was reverted
- Force-push scenarios leave zombie entries in index

### 3.4 Merge Conflicts in Indexed Content

**Risk:** Complex merge scenarios break semantic continuity.

- Three-way merge creates new commit with parts from multiple parents
- Which semantic context does the new commit inherit?
- Conflict resolution changes meaning without clear commit message

---

## 4. Integration Issues

### 4.1 Git Hook Performance

**Risk:** Post-commit hooks that run slowly block developer workflow.

- Embedding generation adds latency to every commit
- If indexing takes 2 seconds per commit, developers will disable hooks
- Background async indexing has race conditions (query before index)

### 4.2 Shell Compatibility

**Risk:** Git hooks written in shell may fail across environments.

- Bash vs Zsh vs Fish differences
- Linux vs macOS edge cases (GNU vs BSD utilities)
- CI/CD environments may lack full shell features

### 4.3 Worktree & Branch Complexity

**Risk:** Git worktrees and branches create index confusion.

- Same file content in different branches - duplicate vectors or shared index?
- Feature branches with rebasing create orphaned entries
- Detached HEAD states

---

## 5. Performance Bottlenecks

### 5.1 Cold Start Time

**Risk:** Initial embedding model load is slow.

- FastEmbed downloads model on first use (~100MB)
- ONNX Runtime initialization adds seconds
- First semantic query after repo clone is painfully slow

### 5.2 Memory Pressure

**Risk:** Multiple tools compete for RAM.

- Developer running: IDE, browser, Ollama/vLLM, zvec, FastEmbed
- Each embedding model consumes 100-500MB
- System swap degrades everything

### 5.3 Write Amplification

**Risk:** Small changes trigger large re-indexing.

- Editing one function in a 1000-line file may require re-chunking entire file
- No incremental update mechanism for changed chunks

---

## 6. Security Concerns

### 6.1 Embedding Data Leakage

**Risk:** Vectors can be reverse-engineered to recover original text.

- Sensitive code (API keys, passwords in comments) embedded in vector store
- Vector database files accessible to anyone with repo access
- Attack: extract embeddings, compare to known sensitive patterns

### 6.2 Model Supply Chain

**Risk:** Embedding model downloads from HuggingFace could be compromised.

- Downstream model poisoning attacks
- Rogue model variants with backdoors
- No supply chain verification

### 6.3 Local Model Privacy

**Risk:** "Local-first" claim may be overstated.

- Check if FastEmbed/ONNX makes outbound network calls
- Telemetry in libraries
- Model files may contain watermarks

---

## 7. Usability Problems

### 7.1 Debugging Non-Obvious Behavior

**Risk:** Semantic search produces unexpected results.

- Why did "authentication" return "authorization" code?
- No clear way to debug or explain vector similarity
- Users distrust "black box" results

### 7.2 False Confidence

**Risk:** High similarity scores create false sense of correctness.

- "Related code" might be tangentially related, not actually relevant
- Agent acts on poor recommendations
- No quality indicators for search results

### 7.3 Onboarding Complexity

**Risk:** New developers face high barriers to entry.

- Must understand: vectors, embeddings, HNSW, chunking
- Troubleshooting requires new skillset
- Team documentation burden

---

## 8. Competitive Alternatives

### 8.1 Existing Solutions

**Risk:** Established tools solve similar problems.

| Alternative | Strengths | Weaknesses |
|-------------|-----------|------------|
| GitHub Code Search | Scale, reliability | Cloud-only, no local |
| Sourcegraph | Semantic code intelligence | Enterprise pricing |
| Claude/GPT-4 | General intelligence | Context window limits |
| Local grep/fzf | Fast, exact matches | No semantic understanding |

### 8.2 Future Threat: Native LLM Features

**Risk:** Anthropic/GPT may add native code understanding.

- Claude Code already has project context
- Future models trained specifically on code
- Window sizes expanding (1M+ tokens)
- Why build semantic git if LLM already understands your repo?

### 8.3 Alternative Vector Approaches

**Risk:** Other embedded vectors may win ecosystem.

- LanceDB (columnar, multi-modal)
- DuckDB (SQL + vectors)
- These have larger communities than zvec

---

## 9. Summary: Critical Questions

Before proceeding, the team should answer:

1. **Validation:** Has semantic search been tested on actual codebases with real queries? Not just MNLI benchmarks.

2. **Scope:** Is the goal searchability, or true "understanding"? Semantic search != reasoning.

3. **Trade-offs:** Are users willing to accept 8% accuracy loss for local-first convenience?

4. **Maintenance:** Who maintains the chunking logic as languages evolve?

5. **Migration:** What happens when zvec 2.0 releases with incompatible API?

---

## 10. Devil's Advocate Conclusion

The core thesis—that semantic memory helps agents—is sound. However, the implementation choices (zvec + FastEmbed + custom chunking) introduce significant operational complexity and accuracy trade-offs that may not be worth the benefits for most use cases.

**Recommendation:** Prototype with strict scope boundaries (e.g., only commit messages, not full file content) before committing to full implementation. Start with Model2Vec for speed, not FastEmbed for quality, to validate the approach quickly.

---

*This analysis represents the Devil's Advocate perspective. The goal is not to kill the project, but to ensure risks are understood and addressed proactively.*
