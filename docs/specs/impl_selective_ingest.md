# Implementation spec: strengthen the selective gate + 2-pass ingest (Issue #7)

## Principle (Cerebras original)

**"Low-signal data never reaches the DB."** Only what passes the gate is distilled (LLM cost), embedded, and stored.

## Target files

- Modify: `src/history_index.py` (this file only)
- Test: `tests/test_selective_gates.py` (new)
- The existing 46 tests must keep passing. Pure-function contracts such as `mean_idf` in
  `tests/test_history_unit.py` are preserved, so there is no conflict.

## Change 1 — 2-pass ingest: recompute corpus DF in full on every run

Current: incremental DF accounting via the df_stats/history_session_tokens tables (complex; the N<20 bootstrap is ineffective).

Change:
- **Pass 1 (statistics pass)**: at the start of every run, parse and tokenize all selected files (regardless of change)
  and recompute DF/N in memory. No LLM/embedding/DB writes. A few seconds at personal scale (145 files).
  - New pure function: `build_corpus_df(transcripts: Iterable[str]) -> tuple[dict[str, int], int]`
    — count df from each transcript's unique token set; N = number of transcripts.
  - Only transcripts of valid sessions (passing `_valid_session`) are included in the corpus.
- **Pass 2 (ingest pass)**: process only changed files as before, but IDF uses Pass 1's in-memory DF.
- **Remove**: delete the `df_stats` and `history_session_tokens` tables and their accounting logic (the DF part of
  `_corpus_without_file`, the DF update in `_write_file`) entirely. In `_ensure_schema`, run
  `DROP TABLE IF EXISTS ...df_stats`, `...history_session_tokens` (safe, as they are derived caches).
  **Keep** `history_file_sessions` for idempotent re-ingest (session deletion).
- Keep the `mean_idf` function signature and formula (protects existing tests).
- Keep the N<20 bootstrap bypass; because it operates corpus-wide it is effectively inactive.

## Change 2 — session triage gate (low-cost selection before distillation)

New pure function `triage_heuristic(session: Session) -> str` — "keep" | "skip" | "borderline":

Decision order (first match from the top):
1. 0 assistant text messages → "skip"
2. user text messages ≤ 2 AND total user text < 200 chars → "skip"  (simple-command session)
3. total text ≥ 5000 chars OR message count ≥ 20 → "keep"
4. otherwise → "borderline"

- Only "borderline" gets a low-cost LLM decision: `_triage_llm(session) -> bool(keep)`.
  Prompt: give the first 3000 chars of the transcript and ask "is this a session worth searching for later
  via 'how did I solve this?'" as `{"keep": true|false, "reason": "..."}` JSON. temperature=0.
  On LLM failure, **fail-open to keep** (prevent data loss) + warning log.
- Under `--dry-run`, apply only the heuristic without LLM calls; count borderline separately.
- Keep the existing `_valid_session` (messages ≥ 5, 500 chars) as the minimum gate, then apply triage after it.
- Even skipped sessions record file processing itself as complete (ingest_state) for idempotency — but no rows are stored.

## Change 3 — incorporate the social signal into the burst gate

New pure function `passes_burst_gate(burst: Burst, mean_idf_value: float, document_count: int) -> bool`:

```
length gate: keep group_bursts's min_chars=200 (existing)
pass = (document_count < 20 or mean_idf_value >= 4.0) and burst.social_weight > 1.0
```

- social_weight > 1.0 = the burst has a tool error or is followed by a user re-ask within 3 minutes (keep existing computation).
- The agent-history counterpart of Cerebras's "reactions-included" AND gate (spec §3.2). If the pass rate is too low,
  keep the gate function in one place so it can be tuned later.
  Note the comment `# ponytail: strict AND gate, relax to scoring if recall suffers`.

## Change 4 — gate-statistics visibility

Add to the summary output (Counter keys):
- `triage_keep`, `triage_skip_heuristic`, `triage_borderline`, `triage_llm_keep`, `triage_llm_skip`
- `burst_no_social` (passed IDF but dropped on social), `low_idf_burst` (existing)
- Print the same statistics under dry-run too (the LLM-decision slot uses the borderline count).

## TDD procedure (important)

This work splits into two steps:

**Step A (test author)**: write failing tests against the interfaces above in `tests/test_selective_gates.py` first.
- `build_corpus_df`: verify DF/N by hand on a small corpus.
- `triage_heuristic`: verify each of the 4 rules + priority (0 assistant messages takes precedence over rule 3).
- `passes_burst_gate`: verify the 4 idf/social combinations + N<20 bypass.
- Import via `from history_index import build_corpus_df, triage_heuristic, passes_burst_gate`.
  Since the functions do not exist yet, **failing with ImportError (red) is expected**.
- No DB/LLM/network dependency.

**Step B (implementer)**: modify history_index.py to turn the tests green. No behavior changes beyond the spec.

## Acceptance criteria

1. `uv run pytest tests/test_selective_gates.py -q` — red (ImportError) right after Step A, green after Step B.
2. `uv run pytest -m "not integration" -q` — all pass, including existing.
3. `uv run python src/history_index.py --dry-run` (all files) — prints gate statistics, no LLM/DB access.
4. (orchestrator-run) `--limit 10` real ingest confirms selection behavior & skip statistics; re-run maintains incrementality.

## Forbidden

- Do not modify src files other than history_index.py. No new dependencies.
- Do not change the memory_chunks schema (keep the search.py contract).
