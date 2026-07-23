# Implementation spec: decision-row ranking in hybrid retrieval

## Problem

A decision-targeted query (e.g. "which default VRM model download approach was chosen")
must surface the matching `chunk_kind='decision'` row in the top 3. Today the decision row
loses ranking to broad session summaries: the reranker scores topically-adjacent summaries
0.80–0.93, and `PER_FILE_CAP=3` lets a session's summary and burst rows crowd out its own
one-line decision before rerank.

## Chosen mechanism: per-kind slot (deterministic, retrieval-side only)

Two small changes in `src/memory_base/retrieval/search.py`, nothing outside retrieval:

1. **Kind is visible to retrieval.** `_search_history` selects `chunk_kind` and carries it
   as `meta["kind"]` on the `Hit`. Code hits have no kind.

2. **Cap exemption.** `_dedup_cap` does not count decision rows against the per-session
   `PER_FILE_CAP`, and decision rows are never evicted by it. Decision rows are one-liners;
   exempting them cannot meaningfully bloat the fused list, which stays bounded by
   `FUSED_TOP`.

3. **Post-rerank decision slot.** After `_rerank`, if the reranked list contains at least
   one decision row but the top 3 does not, the highest-scoring decision row is promoted to
   position 3 (index 2); everything below shifts down. The promotion applies only when the
   decision row's `rerank_score >= DECISION_SLOT_MIN` (module constant, initial value 0.5)
   so an irrelevant decision is never force-fed into unrelated queries.

Rejected alternatives:

- *Kind-aware rerank instruction*: depends on the reranker following a natural-language
  instruction; non-deterministic and unverifiable per query.
- *Lower `PER_FILE_CAP` for session-kind rows*: does not help when the crowding comes from
  adjacent sessions of the same project, and costs burst recall.

## Tests

- Unit (pure, no DB): `_dedup_cap` keeps a decision row that would otherwise be evicted by
  its own session's summary + bursts; a non-decision fourth row from the same session is
  still capped.
- Unit (pure, no DB): the slot function promotes the best decision row to index 2 when the
  top 3 has none and its score clears `DECISION_SLOT_MIN`; leaves the list untouched when a
  decision is already in the top 3, when no decision row exists, or when the best decision
  scores below the floor.
- Integration (DB + rerank endpoint): the VRM-decision query plus 2 other decision-targeted
  queries return their decision row in the top 3; the 3 existing recall validation queries
  keep returning their expected rows.

## Out of scope

Ingest, schema, serving, and the `note` kind are unchanged. `note` rows get no slot: they
are agent-curated but not answer-shaped; revisit only with observed evidence.
