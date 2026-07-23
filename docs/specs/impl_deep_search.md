# Implementation spec: knowledge-aware decomposition (deep search, Phase 4)

## Purpose and usage model

Single-pass retrieval answers single-fact questions. Multi-hop questions ("which service
owned the incident that caused the checkout latency fix?") need facts from several chunks
composed step by step. Deep search ports PIKE-RAG's atom-based decomposition loop
(`qa_decompose.py`, Algorithm 1) onto the existing atom index: the atoms written by
document ingestion become the join surface that lets one hop's finding steer the next
hop's retrieval.

Consumers follow the established usage patterns: agents (Claude, Hermes) call the MCP
tool mid-task; the answer CLI uses the same core for cited answers; n8n reaches the REST
endpoint. Deep search is **opt-in per call** — the default `search` path is untouched.

## Scope

- A decomposition loop over `memory_chunks` (`code_chunks` is out of scope —
  decomposition composes knowledge atoms, and code has none).
- **Source rename**: the retrieval source for `memory_chunks` is named `memory`
  everywhere a client sees it (see Source naming). The name `history` is a legacy of
  session-transcript ingestion, which no longer exists.
- Read-only: no schema changes, no new tables, no writes of any kind.
- Out of scope: automatic question classification (the caller decides when to go deep),
  decomposer fine-tuning, cross-encoder atom selection, and any ranking change to the
  default search path.

## Source naming (`history` → `memory`)

The `memory_chunks` table stores documents, notes, decisions, and atoms — not
conversation history. The client-facing source name says so:

- `/search` accepts `source ∈ code|memory|all` — a **hard rename, no alias**: the
  value `history` is rejected exactly like any other unknown source (400, message
  naming `code|memory|all`). Every response `source` field says `memory`.
- The MCP tool `search_history` is renamed `search_memory` (same signature and
  behavior; no alias tool — MCP clients re-list tools on connect). With `deep_search`
  added, the server exposes exactly **six** tools: `search`, `search_code`,
  `search_memory`, `save_memory`, `ingest_document`, `deep_search`.
- `serve/answer.py`'s planner prompt and `--source` choices use `code|memory|all`;
  the eval runner calls `source="memory"`.
- Retrieval-log rows written from now on carry `memory`; existing rows are not
  rewritten.
- The public `search()` contract renames with it: `source` accepts `code|memory|all`
  only and returned `Hit.source` is always `memory`; the module's standalone CLI
  advertises `code|memory|all` in its choices and help text.
- `docs/specs/impl_docs_ingest.md` is updated in the same change: its filter rule reads
  `source="memory"` and its tool reference reads `search_memory` (current-state docs
  must match the shipped contract).
- Internal identifiers (`_search_history` and similar) follow the rename where touched;
  no behavior change.

## Module boundaries

- `retrieval/decompose.py` — the loop: proposal, hop retrieval, selection, evidence
  accumulation, stop handling. Depends on `retrieval/search.py` internals (atom query,
  hybrid search) and the LLM client from `common.py`. Source-agnostic: it sees rows and
  atoms, never document formats.
- `serve/api.py` — `POST /search/deep` route; validation mirrors `/search`.
- `serve/mcp_server.py` — `deep_search` tool, a thin proxy to the REST endpoint.
- `serve/answer.py` — `--deep` flag: runs the loop in-process, then the existing
  synthesis prompt over the returned evidence.

## The loop (per query)

State: `evidence` — an ordered list of chosen parent rows (each with the atom question
that selected it and the hop number); `chosen_parent_ids` — their id set.

Each hop (while `len(evidence) < max_hops`):

1. **Proposal** — one LLM call (JSON mode): input is the original question plus the
   accumulated evidence texts; output `{"continue": bool, "sub_questions": [...]}` with
   at most 3 English sub-questions. `continue=false` stops the loop
   (`stopped_reason="done"`).
2. **Hop retrieval** — candidate atoms come from the first non-empty stage of this
   backup chain, always excluding atoms whose parent is already in `chosen_parent_ids`
   (exclusion is a SQL predicate, applied before each LIMIT):
   a. atom vector search over all proposed sub-questions (union), collapsed by parent
      id keeping the highest-cosine atom, capped at 8 candidates;
   b. atom vector search with the original question, same collapsing and cap;
   c. a dedicated memory-backup helper that runs the **complete** memory ranking
      pipeline (vector + FTS candidates, RRF fusion, time decay, dedup/cap) for the
      original question and returns its top 8 ranked rows as synthetic candidates
      (`atom_question=None`). It accepts `excluded_parent_ids` and applies the
      exclusion inside both SQL queries, before their LIMITs — the existing per-signal
      candidate helper returns unfused candidates and is not used directly.
   An empty chain result stops the loop (`stopped_reason="no_candidates"`).
3. **Selection** — one LLM call (JSON mode): input is the original question, the
   accumulated evidence, and the candidate list (atom question + a 500-char parent
   snippet each); output `{"selected": <index>|null}`. `null` stops the loop
   (`stopped_reason="no_selection"`). A valid index appends that candidate's full parent
   row to `evidence`.

Loop exit at the hop cap sets `stopped_reason="max_hops"`. Hops are **1-based**; every
attempted proposal appends one `trace` entry (with empty `sub_questions` when parsing
never succeeded); `hops_used = len(evidence)` (selected evidence count — `trace` may be
longer than `hops_used` when a hop stops after its proposal).

**LLM failure is fail-open with an honest flag**: any proposal/selection call that still
fails after one retry (transport, invalid JSON, wrong shape) stops the loop with
`stopped_reason="llm_error"` and the evidence collected so far is returned. Nothing is
stored, so the selective-storage principle is not in play; a partial evidence set is
more useful to the caller than an error. The two prompts are ported from PIKE-RAG's
decomposition protocols (`pikerag/prompts/decomposition/atom_based.py`), adapted to JSON
mode.

## Bounds

- `DEEP_MAX_HOPS` (env, default 3) — also the evidence cap; a request may lower it per
  call (`max_hops` parameter, validated `1..DEEP_MAX_HOPS`).
- At most 3 sub-questions per proposal; candidate cap 8 per hop; parent snippet 500
  chars in the selection prompt.
- Work per query: at most `2 × max_hops` **logical LLM operations** (proposal +
  selection per hop); with one retry each, at most `4 × max_hops` actual LLM requests.
  A total wall-clock budget `DEEP_TIMEOUT_SECONDS` (env, default 180) is an **absolute
  monotonic deadline** set at request start: every LLM and embedding attempt runs under
  `min(SERVICE_TIMEOUT_SECONDS, remaining_budget)`, and deadline expiry — whether
  between hops or mid-hop — stops the loop with `stopped_reason="timeout"` and returns
  the evidence collected so far. Calls are sequential — no new concurrency machinery.
- The MCP proxy's HTTP client for `deep_search` uses a request timeout of
  `DEEP_TIMEOUT_SECONDS + 30` (the default `httpx` timeout is seconds-scale and would
  cut legitimate multi-minute deep queries).

## API surface

`POST /search/deep` body: `{query, max_hops?, kind?, tags?, include_archived?}`.
`kind`/`tags` reuse `/search` validation and semantics and apply as SQL predicates
inside every hop-retrieval stage (atom parents and backup hybrid search alike).
Errors use the existing `{"error": message}` 400 shape. Response:

```
{
  "evidence": [ {ref, text, kind, tags, date, hop, atom_question|null, id} ... ],
  "trace":    [ {hop, sub_questions, selected_ref|null} ... ],
  "hops_used": int,
  "stopped_reason": "done"|"max_hops"|"no_candidates"|"no_selection"|"llm_error"|"timeout"
}
```

`evidence` is ordered by hop; `ref` resolves exactly as in `/search`
(`metadata.search_ref` fallback `source_ref`); `text` is the canonical retrieval text
(`distilled` fallback `content_raw`), truncated to 2,000 chars in the REST response
exactly like `/search`. Access logging credits every evidence row. The MCP tool
`deep_search(query, max_hops=None, kind=None, tags=None)` proxies the endpoint and
returns the same payload; REST 400s surface as tool errors.

`uv run python -m memory_base.serve.answer "question" --deep` **bypasses the planner**
(deep search operates on `memory_chunks` only) and runs the loop in-process; an
explicit `--source` other than `memory` combined with `--deep` is a CLI validation
error. The
loop's evidence entries convert to `Hit` objects preserving id, ref, timestamp, kind,
tags, hop, and atom question, then feed the existing synthesis prompt (citations keep
working: evidence entries are numbered in loop order).

## Validation (acceptance)

- **Eval extension**: `tests/fixtures/retrieval_eval.jsonl` gains a `multi_hop` class
  (≥ 5 queries) whose `relevant_ids` span two different fixture documents; a bridging
  fixture document is added so 2-hop paths exist. The **existing atom A/B gate is
  computed only over the three pre-existing single-hop classes** — `multi_hop` labels
  are excluded from it, so its numbers are unchanged by construction. The runner gains
  a deep section for `multi_hop` queries reporting: (a) Recall@5/MRR@10 of baseline
  `search` versus the deep evidence list (ranked), and (b) **path completion** — the
  fraction of queries whose deep evidence contains at least one labeled row from
  *each* labeled document. The baseline invocation is **pinned** — `search(query,
  source="memory", include_atoms=True, rerank=True, include_archived=False)` — so the
  gate cannot drift with the `ATOMS_RETRIEVE` environment; a unit test asserts the
  pinned arguments. Gate: deep Recall@5 ≥ baseline Recall@5 on `multi_hop`
  **and** path completion ≥ 0.4 (a nonzero bar — equality-at-zero does not pass). The
  report is PR runtime evidence.
- **Unit tests** (no DB/LLM; mocked search and LLM): proposal/selection JSON parsing
  (shape errors, out-of-range index, retry-then-fail), every `stopped_reason` path
  including `timeout`, chosen-parent exclusion across hops (SQL predicate before
  LIMIT), backup-chain order (a→b→c, first non-empty wins), per-call `max_hops`
  validation, filter pass-through into every stage, trace/hops_used accounting on
  partial hops, evidence-to-`Hit` conversion for the CLI, rejection of `--deep` with
  `--source` other than `memory`,
  REST validation mirror and the 2,000-char text cap, MCP proxy forwarding and its
  extended timeout. Rename semantics are pinned by tests: canonical `memory` input
  accepted, `history` input rejected with a 400 naming `code|memory|all`, validation
  errors and responses naming only `memory`, retrieval-log rows carrying `memory`,
  eval runner calls using `memory`, and exact MCP registration (`search_memory`
  present, `search_history` absent, six tools total); existing assertions in the
  proxy, answer, supersede, access-log, and integration tests update accordingly.
- **Integration test**: one end-to-end deep query over ingested fixtures that requires
  two hops and returns evidence from two documents.

## Implementation split (single PR, staged commits)

1. `feat:` decomposition core (`retrieval/decompose.py`) + prompts + unit tests.
2. `feat:` REST `/search/deep`, MCP `deep_search`, `answer.py --deep` + tests.
3. `feat:` eval multi-hop extension (bridging fixture, labels, deep report section).
