# Agents contribute via MCP only; claude_code file-ingestion channel removed (#34)

User-confirmed decisions (interview): interactive agent consoles (Claude Code, Hermes) contribute
to memory exclusively through the MCP realtime channel — `save_memory` for writes, the search
tools for reads. Batch ingestion of console session history is removed. The source-agnostic
selection core and the adapter contract stay in-tree because corpus sources (e.g. Slack) are
planned; existing `memory_chunks` rows are kept as retrieval assets.

## 1. Resulting architecture

```
Agent consoles ──(MCP: save_memory / search·search_code·search_history)──> REST API ──> memory_chunks
Code repos    ──(CocoIndex incremental indexing)─────────────────────────────────────> code chunks
Future corpus sources (Slack, …) ──(adapter ABC + selection core, no active adapter yet)
```

There is no file-based history ingestion. `ADAPTERS` is an empty registry awaiting future corpus
adapters.

## 2. Removed

- `adapters/claude_code.py` — `~/.claude/projects` discovery, JSONL parsing, the claude_code
  adapter.
- File-channel machinery in `ingest/history.py`: `run`/`main` CLI, file selection and scanning,
  `ingest_state` bookkeeping, per-file write path (`_write_file`, `_old_session_ids`,
  `_process_file`), dry-run reporting, `ACTIVE_WINDOW_SECONDS`.
- `ingest_state` and `history_file_sessions` tables — `ensure_schema` drops them idempotently
  (same pattern as earlier retired tables).
- Tests covering claude_code parsing, file scanning, and the CLI.

## 3. Kept

- `adapters/base.py` — `SourceAdapter` ABC, `SourceFile`, `Message`, `Burst`, `Session`.
- `ingest/history.py` — the source-agnostic selection core only: `_valid_session`,
  `triage_heuristic`, LLM triage, `_distill`/`parse_distillation`, `build_transcript`,
  `group_sessions`, `group_bursts`, `tokenize`/`mean_idf`/`build_corpus_df`, `burst_score`/
  `passes_burst_gate`, `build_rows`. No I/O entrypoint; future corpus adapters feed it.
- Existing `memory_chunks` rows (sessions/decisions/notes) — retrieval assets, unchanged.
- Retrieval, REST API, MCP tools (all four), save_memory, access logging, CocoIndex code
  indexing.

## 4. Tests

- Unit coverage of the kept core is preserved (gates, distillation parsing, burst grouping,
  transcript building, IDF stats) with fixture-based sessions instead of parsed claude_code
  files.
- Adapter contract tests keep exercising the ABC through a fake adapter; the registry test pins
  `ADAPTERS == {}`.
- Integration evidence: `search_history` still returns pre-existing rows; schema migration
  drops the two tables on a live DB without touching `memory_chunks`.

## 5. Docs

`AGENTS.md` project structure and commands reflect the state above (no history-ingest CLI).
