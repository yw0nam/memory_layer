# Implementation spec: disable batch burst rows for claude_code (Issue #17)

## Principle

Interactive agent sources do not emit burst rows from the batch pipeline. Measured
on the real corpus, every stored claude_code burst passed the gate only via the
degenerate social boost, and the valuable ones duplicate what session distillation
already captures; real-time high-value capture is the `save_memory` channel. The
bursting mechanism (grouping + weighted gate) **stays in the source-agnostic
core** for future sources with genuine social signals.

## Design: per-adapter toggle

```python
class SourceAdapter(ABC):
    source_type: ClassVar[str]
    emit_bursts: ClassVar[bool] = True   # new
```

- `ClaudeCodeAdapter.emit_bursts = False`. All other (future) adapters default on.
- A toggle, not an always-false `has_social`: "never passes the gate" and "the
  channel is off" are different statements, and the stats must say the latter.

## Behavior

- `build_rows`: when `adapter.emit_bursts` is False, skip burst grouping/scoring
  entirely — rows are session + decision only.
- `_process_file` / `_count_dry_run_session`: when off, do not run `group_bursts`;
  `burst_rows`, `burst_below_threshold`, and `dry_run_burst_candidates` count 0
  for that adapter's files (honest statistics: no candidates were considered).
- Retrieval, schema, gate values: untouched. Existing burst rows already in the
  DB are removed by the next `--full` re-ingest (per-file delete-then-insert),
  not by a migration in this PR.

## TDD

**Step A (red)**: extend `tests/test_adapter_contract.py` (new test functions
only; existing ones unmodified):
- `ClaudeCodeAdapter.emit_bursts` is False; `SourceAdapter` subclasses default
  to True (FakeAdapter).
- `build_rows` with an `emit_bursts=False` adapter returns session + decision
  rows only, for a session whose bursts would pass the gate with
  `emit_bursts=True` (prove the toggle, not the gate, removed them).
- `build_rows` with the default FakeAdapter still emits burst rows (mechanism
  alive in the core).

**Step B (green)**: implement the toggle. Existing tests that pin claude_code
burst emission (if any) are updated to the new intended behavior — the only
permitted edits to existing tests, each flagged in the report.

## Acceptance criteria

1. Red → green; full suite + ruff clean.
2. Full-corpus dry run: `burst_rows=0 burst_below_threshold=0
   dry_run_burst_candidates=0`; session/triage statistics unchanged vs main
   (orchestrator compares back-to-back).
3. Real `--full` ingest stores zero claude_code burst rows; DB kind counts
   reported in the PR.
4. Burst mechanism tests (group_bursts, passes_burst_gate, burst_score) remain
   present and passing.

## Forbidden

Deleting bursting code from the core. Changes to retrieval/search.py, serve/*,
common.py, gate constants, schema.
