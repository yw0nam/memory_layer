# Implementation spec: weighted-sum burst gate + decision extraction (Issue #9)

## Rationale

Cerebras original: "each burst is scored against a **weighted combination of signals** and must
clear a threshold before it is embedded … reactions, providing a **social boost**."
→ Move the strict AND (current) to a weighted sum faithful to the original. User-confirmed.

## Target files

- Modify: `src/history_index.py`, `tests/test_selective_gates.py` (only the gate-contract changes)
- New tests: decision-extraction related (add to an existing file or a new file, author's discretion)
- Do not modify other src files. Do not change the memory_chunks schema (no column additions —
  distinguish by chunk_kind value only, keep the search.py contract).

## Change 1 — `passes_burst_gate` to a weighted sum

```python
def burst_score(mean_idf_value: float, has_social: bool) -> float:
    return mean_idf_value + (1.0 if has_social else 0.0)

def passes_burst_gate(burst: Burst, mean_idf_value: float, document_count: int) -> bool:
    # original: weighted combination clears threshold; social is a bonus (+1.0)
    if document_count < 20:  # bootstrap: region where IDF is untrustworthy
        return True
    return burst_score(mean_idf_value, burst.social_weight > 1.0) >= 4.0
```

- Keep the 200-char requirement via the existing `group_bursts(min_chars=200)`.
- Social-signal definition unchanged: tool error or user re-ask within 3 minutes (keep the `social_weight=1.5` computation as-is).
- On bootstrap (N<20), pass regardless of social (length gate only) — the existing test
  `bootstrap_bypass_still_requires_social_signal` is **updated to the new contract** (Step A below).
- Statistics key change: `burst_no_social` → `burst_below_threshold` (below the sum, whether IDF or social).
  Remove the `low_idf_burst` key and fold it into `burst_below_threshold`.

## Change 2 — add decisions to session distillation

- Add `"decisions"` to the JSON schema of the `_distill` prompt:
  `"decisions": ["main decisions made in the session. One sentence each on what was decided and why. Empty array if none"]`
- Add a `decisions: list[str]` field to the `Distillation` dataclass (default `[]`, tolerant parsing:
  missing/non-list → empty array; items are str-converted, whitespace-trimmed, empties removed).
- `Distillation.text` (the session-row embedding target) **keeps the existing 4 elements** — decisions are separate rows,
  so they are not duplicated into the session row.
- Storage: add a row per decision per session —
  - `id = f"{session_id}:decision:{i}"`, `chunk_kind = "decision"`
  - `content_raw = decision raw text`, `distilled = f"[{one_line_question}] decision: {decision}"`
  - embedding over distilled, `idf_score = NULL`, `ts_last_active` = same as the session
  - `metadata` = same as the session row + `{"index": i}`
- Cap decisions at 10 per session (guard against LLM runaway; drop the overflow with a warning log).

## Step A — tests first (red)

Update the gate portion of `tests/test_selective_gates.py` to the new contract + add decision-parsing tests:

- `burst_score`: verify the ±1.0 bonus for signal presence/absence.
- `passes_burst_gate` new contract: IDF 4.0 alone passes / IDF 3.0 + signal (=4.0) passes /
  IDF 3.9 alone fails / IDF 2.9 + signal (=3.9) fails / N<20 passes even without signal (inverts the existing test) /
  N=20 boundary is not bootstrap.
- decision parsing (the pure part of the `Distillation` construction path): decisions missing → [], non-string items cleaned,
  over-10 truncated. Requires splitting the parsing logic into a pure function: `parse_distillation(parsed: dict) -> Distillation`
  (extracted from the parsing part inside the existing `_distill` — the implementation step makes `_distill` call it).
- red confirmation: the new/updated tests must fail on the current implementation (absent `burst_score`/`parse_distillation`
  → ImportError; gate-behavior difference → assert failure).

## Step B — implementation (green)

- Implement Changes 1 & 2 above, including the `parse_distillation` pure-function split.
- Verify: `uv run pytest tests/test_selective_gates.py -q` green,
  `uv run pytest -m "not integration" -q` all green.

## Acceptance criteria (including orchestrator-run)

1. red right after Step A, green after Step B. All existing unrelated tests preserved.
2. `--dry-run` full: confirm burst survival rate increases vs strict AND (6) but not all pass.
3. `--limit 10 --full` real ingest: confirm decision rows are created, and decision hits via `search.py --source history`.
4. Incremental re-run preserved.
