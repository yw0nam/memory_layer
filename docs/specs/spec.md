# Personal RAG · Agent Memory System — Architecture Overview + Roadmap

> Source foundations
> - Cerebras Knowledge Base design (original blog / PyTorchKR commentary) — pgvector · hybrid search · LLM distillation · bursting · RRF · MCP
> - PIKE-RAG (arXiv 2501.11551, Microsoft Research Asia) — specialized-knowledge & rationale augmentation, multi-layer knowledge base, knowledge atomization, L0–L4 phased roadmap
> - Scope: **personal / PoC scale**, priority **① code repositories · ② agent chat history (custom DB) → ③ n8n · Gmail · GDrive · Slack · wiki**
>
> The technical figures and model names in this document were web-verified as of July 2026 (see "Verified facts" below).

---

## 1. One-line summary

A **personal knowledge & memory system** that compresses the RAG architecture Cerebras built to handle 15,000 internal questions/day down to a single-developer scale, gathering the most valuable personal data — **code repositories + agent (Claude Code, Hermes, etc.) conversation history** — into a single pgvector table for natural-language querying. The advancement axis is layering on PIKE-RAG's multi-layer knowledge base, knowledge atomization, and phased (L0–L4) capability expansion.

Three core design philosophies (inherited directly from Cerebras):
1. **Go to where the data is** — extract directly from each source without moving it.
2. **Never embed raw content** — structure-distill with an LLM first, then embed.
3. **Signal fusion, not a single model** — vector search is not omnipotent; practical accuracy comes only from fusing full-text search, IDF, and time decay via RRF.

---

## 2. Why this combination (reinterpreted for personal scope)

| Cerebras original problem | Meaning at personal scope |
|---|---|
| Slack is the most important and messiest source | **Agent chat history** takes that place. A one-line aside and a long debugging log are mixed in one session → the same hybrid problem |
| Incremental maintenance of 40GB+ code repositories | A few local repositories. Small in scale, but **incremental indexing** is still central (can't recompute everything on every commit) |
| Internal wiki · Docs · Jira | Lower-priority sources (Gmail · GDrive · n8n · Slack · wiki) — added via connectors |
| `who_knows` (expert search) | Unnecessary at personal scope. Instead, **"how did I solve this problem before?"** (episodic memory) takes that place |

**Memory-system perspective**: Agent conversation history corresponds to **episodic + procedural memory** in the standard memory taxonomy (semantic/episodic/procedural/working). Claude Code deletes transcripts after 30 days by default (`cleanupPeriodDays`), so an agent that saves its own distilled decisions and notes into this system through `save_memory` keeps them as **permanent long-term memory beyond the 30-day window** — this is the core motivation behind the name "memory system".

---

## 3. Architecture overview

### 3.1 Overall layers (Cerebras's 6 layers condensed for personal use)

```
Source → Ingest → Distill → [single embedding table] → Hybrid search → Fusion·rerank → Synthesis
                                                                    ↑
                                                        also exposed as thin MCP tools
```

```mermaid
flowchart TD
    subgraph SRC["① Sources (by priority)"]
      A1[Code repositories<br/>git repos]
      A2[Agent consoles<br/>Claude Code · Hermes]
      A3[Lower priority: n8n · Gmail · GDrive · Slack · wiki]
    end
    subgraph ING["② Ingest"]
      B1[CocoIndex<br/>Tree-sitter incremental]
      B2[MCP save_memory<br/>agent-distilled notes/decisions]
      B3[Custom connector]
    end
    subgraph DIST["③ Distillation (LLM structuring)"]
      C1[summary · one-line question · resolution · reference extraction]
      C2[Bursting: preserve individual signals within a session]
      C3[PIKE-RAG: knowledge atomization · auto-tagging]
    end
    DB[(④ Single Postgres + pgvector<br/>embedding · raw summary · metadata<br/>+ FTS(GIN) index)]
    subgraph RET["⑤ Hybrid search"]
      D1[Full-text search FTS]
      D2[Embedding search]
      D3[IDF weighting]
      D4[Time decay]
    end
    subgraph FUSE["⑥ Fusion · rerank"]
      E1[RRF k=60]
      E2[Source dedup · cap]
      E3[Reranker top-10]
      E4[Context restoration]
    end
    F[⑦ Planner → Executor → Synthesis]
    MCP{{MCP thin tools<br/>search · search_code · search_history}}

    A1-->B1; A2-->B2; A3-->B3
    B1-->C1; B2-->C1; B3-->C1
    C1-->C2-->C3-->DB
    DB-->D1 & D2 & D3 & D4
    D1 & D2 & D3 & D4-->E1-->E2-->E3-->E4-->F
    DB-.->MCP
    MCP-.->F
```

### 3.2 Component-by-component design

**① Sources & ② Ingest**

- **Code repositories (priority 1)**: Adopt **CocoIndex**. As of 2026, it chunks via Tree-sitter (syntax-accurate) plus language-specific regex boundaries, and is an incremental engine that **re-embeds only changed chunks**. It tracks sync metadata in Postgres, so it can live in the same DB as the embedding store. Generates **multi-granularity** embeddings at file level, function level, etc.
- **Agent history (priority 1)**: interactive agent consoles (Claude Code, Hermes) contribute directly through the MCP realtime channel — the `save_memory` tool stores a distilled note or decision the agent already wrote in English; there is no batch file parsing. A source-agnostic selection core (triage → burst gate → distillation) and an adapter ABC contract stay in-tree for future batch corpus sources (e.g. Slack) that are not files an interactive agent already has open.
  - Treat **session = Slack thread** → reconstruct the whole session into one conversation state and store it as a single row (isomorphic to Cerebras thread reconstruction) — applies to any future batch adapter.
- **Lower-priority sources**: n8n (workflow logs/notes), Gmail, GDrive, Slack, wiki → each added via a connector. It's done once they all write **identical-schema rows** to the shared DB (plugin architecture).

**③ Distillation (LLM structuring)** — the biggest lever on accuracy

- Never embed raw content. For each thread/session, the LLM extracts: **a likely search question (one line) · a short summary · the resolution · mentioned systems & code references**.
- **Bursting**: To preserve individual signals not captured in the summary of a long session, groups of consecutive same-speaker messages (bursts) are embedded individually with the thread topic prefixed. Low-signal cutoff thresholds — IDF ≥ 4.0, combined length ≥ 200 chars, (plus reaction weighting for Slack). **In agent history, "reactions" are replaced by social weights such as tool success/error and whether the user re-asked.**
- Raw text is keyword-searchable via an FTS (GIN) index the moment it arrives. Distillation is done only for the vector path.

**④ Storage — single pgvector table**

- All sources land in **one embedding table** (unified schema). Example columns: `id, source_type, source_ref, content_raw, distilled, embedding, ts_last_active, idf_meta, metadata(jsonb)`.
- **Dimension choice (important, verified)**: Cerebras uses 3072 dimensions. pgvector's standard `vector` has a **2000-dimension limit under HNSW** → using 3072 requires **`halfvec` (16-bit, up to 4000 dimensions)**. For a personal PoC, to cut storage & complexity, we recommend **voyage-code-3 at 1024/2048 dimensions with int8 quantization (Matryoshka), or the open-weight Qwen3-Embedding** → indexable even with standard `vector`. Keep code and text embeddings separate, but **in the same table via different columns or a source_type distinction**.
- HNSW index (`m`, `ef_construction` at build time / `ef_search` 40–200 at query time). FTS uses GIN.

**⑤ Hybrid search** — no single scorer is trusted alone

- **Full-text search (FTS)**: exact tokens like error strings, flag names, function names. So that no semantic similarity can beat literal matching when a literal match is the best evidence.
- **Embedding search**: connects paraphrases.
- **IDF**: promotes short messages centered on rare tokens; the likes of "Thanks!" converge to 0.
- **Time decay**: favors recent sessions (suppresses an old answer describing legacy infrastructure). **Especially important for code/history memory** — lets the latest beat pre-refactor code or abandoned approaches.

**⑥ Fusion · rerank**

- **RRF**: `score(d) = Σ_L w/(k + rank_L(d))`, `w=1.0, k=60`. The smoothing constant 60 makes **consensus > a single strong vote**.
- Source-level dedup + per-file result cap → a diverse top-20.
- **Reranker** scores 0–10, then top-10. As of 2026, **Voyage rerank-2.5 / rerank-2.5-lite** (instruction-following) recommended; for open-weight, the Qwen3 family.
- **Context restoration**: for a wiki section, reattach adjacent sections; for code, the parent-file context of a function; for history, the surrounding messages — restoring the premises and caveats that chunking severed ("Lost in the Middle" mitigation).

**⑦ Planner → Executor → Synthesis + MCP**

- **Planner** (lightweight LLM): looks at the query and decides which tools/sources to use.
- **Executor**: parallel fan-out of tool calls → normalized into a common evidence schema (score · recency · source hint).
- **Synthesis**: typed evidence + original query → a final answer with citations and caveats.
- **MCP exposure**: exposes the search primitives as thin, LLM-agnostic tools (`search`, `search_code`, `search_history`). Then any agent or orchestrator such as Claude Code can hook straight into personal workflows. **This is the biggest practical payoff at personal scope** — the agent you already use becomes the frontend, with no separate UI.

---

## 4. Can PIKE-RAG be layered on — review result

**Conclusion: yes. It meshes very well.** Cerebras is a "battle-tested operations architecture" (how to handle messy data with a pipeline); PIKE-RAG is a "capability-stage & knowledge-representation theory" (how far to grow the questions it can answer), so **the layers don't overlap and are complementary**. Mapping:

| PIKE-RAG concept | Cerebras counterpart/gap | How to layer onto the personal system | Difficulty |
|---|---|---|---|
| **L0 multi-layer heterogeneous knowledge base** (info-source layer · corpus layer · distilled-knowledge layer graph) | Cerebras is a **flat embedding table** — no explicit hierarchy | Add `parent_ref`/layer tags to the flat table to query **multi-granularity links** (file↔function, session↔burst, doc↔section) like a graph | Medium |
| **Knowledge atomization** (atomic questions as a knowledge index) | **Nearly identical idea** to Cerebras distillation's "one-line question" + bursting | Extend the distillation we already do into "multiple atomic questions per chunk" → strengthens the multi-granularity index | **Low (immediate)** |
| **Auto-tagging** (colloquial query ↔ technical-term mapping) | Absent in Cerebras | Bridge the gap between code/agent terminology and natural-language queries with tags (e.g. "that lock issue back then" ↔ `mutex deadlock`) | Low |
| **Knowledge-aware task decomposition** (multi-hop iterative retrieval-generation) | Cerebras Planner is **one-shot plan · parallel fan-out** — not iterative decomposition | Promote the Planner to an **iterative loop**: retrieve atomic-question candidates → select → accumulate context → re-query. Effective for multi-hop ("what caused the issue referenced by the commit that fixed A?") | Medium–high |
| **L1–L4 capability stages** | Cerebras is a single mature system | **Adopt as the roadmap axis as-is** (section 5 below) | — |
| **Learned decomposer** (domain-specific fine-tuning) | Absent | Over-investment at personal scope. **Deferred** (only after data & an eval set accumulate) | High (later) |

**What not to layer on**: PIKE-RAG's VLM file parsing (multimodal tables/charts) and its multi-agent planning for prediction (L3)/creation (L4) are overkill for personal code/history memory. **Take only the L0–L2 concepts and leave L3–L4 as conceptual room.**

---

## 5. Roadmap (Cerebras pipeline × PIKE-RAG L0–L2 stages)

Each phase is **a system that works on its own**, and the next phase inherits and augments the lower modules (the PIKE-RAG way).

### Phase 0 — Skeleton: single table + code indexing (PIKE-RAG L0)
- Finalize a single-table schema: Postgres + pgvector + FTS (GIN).
- Finalize the embedding-model choice (recommended: voyage-code-3@1024 or Qwen3-Embedding for code; standard `vector` if dimensions ≤ 2000, `halfvec` if insisting on 3072).
- Incrementally index 1–2 code repositories with CocoIndex.
- **Done when**: code search like "where is this function defined?" gives semantically better results than grep.

### Phase 1 — Agent memory ingest + distillation (PIKE-RAG L1 factual)
- Source-agnostic selection core (triage → burst gate → distillation) + session=thread reconstruction, fed by agent consoles through the MCP `save_memory` tool.
- **LLM distillation** (one-line question · summary · resolution · references) + **bursting** (tool-success/re-ask weighting).
- Hybrid search (FTS + embedding + IDF + time decay) running.
- **Done when**: "how did I solve that build error last month?" summons the exact past session.

### Phase 2 — Fusion · rerank · MCP (Cerebras complete form)
- RRF (k=60) fusion + source dedup + reranker (rerank-2.5) top-10 + context restoration.
- Planner → Executor → Synthesis pipeline.
- Expose **MCP thin tools** (`search`, `search_code`, `search_history`) → callable directly from Claude Code.
- **Done when**: a query spanning code + history returns an answer with citations and caveats, and an agent can call it as a tool.

### Phase 3 — Layering + atomization + auto-tagging (PIKE-RAG L1 reinforced)
- Add layer/parent links to the flat table (file↔function, session↔burst, doc↔section).
- **Knowledge atomization** (multiple atomic questions per chunk) + auto-tagging (colloquial↔technical term).
- Introduce project/scope concepts (separate search across code vs memory vs lower-priority sources).
- **Done when**: multi-granularity search improves recall and precision simultaneously (measured with a lightweight eval set).

### Phase 4 — Multi-hop decomposition + lower-priority sources (PIKE-RAG L2 reasoning)
- Promote the Planner to an **iterative retrieval-generation loop** (knowledge-aware task decomposition).
- Add lower-priority connectors: n8n → Gmail → GDrive → Slack → wiki (each a plugin script).
- (Optional) Consider a learned decomposer only after a lightweight eval set accumulates.
- **Done when**: answers 2–3-hop queries like "the root cause of the issue referenced by the session that fixed A".

---

## 6. Tech-stack summary (personal/PoC recommended values)

| Layer | Recommendation | Notes (verified) |
|---|---|---|
| Storage | PostgreSQL + pgvector | HNSW + GIN (FTS) in one DB |
| Vector type | standard `vector` (≤2000d) / `halfvec` if 3072d | halfvec = 16-bit · 4000d · half the storage, ef_search 40–200 |
| Code embedding | voyage-code-3 (1024/2048, int8) or Qwen3-Embedding | Matryoshka · quantization cut storage & cost |
| Text embedding | Gemini Embedding 2 (top-tier) or Qwen3-Embedding-8B (open-weight MTEB leader) | open-weight can run locally for personal use |
| Code indexer | CocoIndex (Tree-sitter, incremental) | Postgres lineage tracking, recomputes only changed chunks |
| Reranker | Voyage rerank-2.5 / -lite | instruction-following |
| Fusion | RRF k=60, w=1.0 | consensus first |
| Exposure | MCP thin tools | Claude Code · MCP agents as orchestrators |

---

## 7. Verified facts (web-checked 2026-07)

- pgvector's standard `vector` has a **2000-dimension limit under HNSW** → 3072 dimensions (the size Cerebras uses) is indexed with **`halfvec`** (up to 4000 dimensions, half the storage, ef_search 40–200 recommended).
- **voyage-code-3**: specialized for code search, supports 2048/1024/512/256 dimensions, and cuts storage & search cost dramatically with Matryoshka + int8/binary quantization (+13.8% on code search vs OpenAI v3-large).
- **Voyage rerank-2.5 / rerank-2.5-lite**: instruction-following rerankers (current generation).
- Open-weight embeddings: **Qwen3-Embedding-8B** is near the top of the open-weight MTEB v2 leaderboard (~75%), and **Gemini Embedding 2** is rated top-tier overall.
- **CocoIndex** (active in 2026): Tree-sitter syntactic chunking + incremental (changed files only) + Postgres lineage. Positioned as an "incremental engine for long-horizon agents".
- **Claude Code history**: session transcripts are **deleted after 30 days by default** (`cleanupPeriodDays`) — a distilled note the agent saves via `save_memory` during the session survives as permanent memory beyond that window.
- Standard agent-memory taxonomy: semantic · episodic · procedural · working → this system maps code = semantic/procedural, agent history = episodic/procedural.

---

## 8. Open decisions (to settle in the next steps)

1. Embeddings **local (Qwen3, zero cost · privacy)** vs **API (Voyage/Gemini, quality · convenience)**? — for personal history, local may be preferable for privacy.
2. Stick with 3072 dimensions (as Cerebras) vs shrink to 1024–2048 (recommended). The latter needs no halfvec.
3. Finalize the agent-history source list (need a sample of Hermes's actual storage format besides Claude Code).
4. Finalize the start order for lower-priority sources (current assumption: n8n → Gmail → GDrive → Slack → wiki).
