# Implementation spec: expand test coverage (Issue #5)

## Goal

Write **characterization & integration tests** against the existing implementation (only manually verified so far)
to establish a regression-prevention baseline. This work writes tests only — **do not modify src/ code**
(if you find a bug, do not fix it; attach `@pytest.mark.xfail(reason=...)` to the test and record it in the report).

## Context

- Repository: memory_base (personal RAG · agent memory system). Work in the designated worktree directory.
- Existing tests: `tests/test_answer_mcp.py` (12), `tests/test_history_parse.py` (3) — must be kept.
- Target modules (all under `src/`, flat-import convention):
  - `search.py`: `_rrf_fuse(lists)`, `_apply_time_decay(hits)`, `_dedup_cap(hits)`, the `Hit` dataclass,
    `search(query, source, rerank)` (needs DB + embedder + reranker)
  - `history_index.py`: `mean_idf`, `tokenize`, `build_transcript`, `group_sessions` (add pure-function coverage)
  - `mcp_server.py`: the FastMCP server `mcp`, three tools
  - `answer.py`: `plan`/`execute`/`synthesize` (needs LLM)
- Runtime environment: `.env` has LLM_URL/EMB_URL/RERANK_URL/DB_URL. DB is docker `memory_base_db` (port 5439),
  already loaded with 34 code_chunks rows and 38 memory_chunks rows.
- pytest config: register the `integration` marker via `[tool.pytest.ini_options]` in `pyproject.toml`
  (editing this file is allowed — limited to marker registration & testpaths).

## Deliverables

### 1. `tests/test_search_unit.py` — no DB/network dependency

- `_rrf_fuse`: verify the k=60 formula — a doc ranked #1 in both lists scores 2/(60+1); a doc agreed by both lists
  scores higher than one present in only one list (consensus > a single strong vote).
- `_apply_time_decay`: for two Hits with equal rrf, the one 90 days older is exactly half (90-day half-life); age 0 has no decay.
- `_dedup_cap`: 5 chunks of the same file → capped to 3 (PER_FILE_CAP), an overall FUSED_TOP=20 cap,
  descending-rrf order preserved.
- `Hit` defaults (meta dict independence, etc.) briefly verified.

### 2. `tests/test_history_unit.py` — no DB/LLM dependency

- `mean_idf`: verify the N/df formula (compare against hand-computed values on a known small corpus); empty text → 0.0.
- `tokenize`: extract identifiers (`snake_case`, CamelCase lowercased) and Hangul 2+ chars, excluding single chars & symbols.
- `build_transcript`: verify head 60% + marker + tail 40% structure when over 100k.
- `group_sessions`: per-session grouping, ts_last_active = max message timestamp, fallback behavior.

### 3. `tests/test_integration.py` — `@pytest.mark.integration`, uses real services

At the top of the module, on DB-connection failure, skip the whole module via `pytest.skip(allow_module_level=True)` (CI-safe).

- `search(q, source="code", rerank=False)` → at least 1 Hit, source=="code", ref contains ":L",
  rrf > 0. (rerank=False for speed without reranker dependency)
- `search(q, source="history", rerank=False)` → verify a source=="history" Hit is returned.
- `search(q, source="all", rerank=True)` → the full pipeline including the reranker path, verify rerank_score exists.
- FTS exact-token check: searching a literal that definitely exists in code_chunks (e.g. "halfvec") hits that file.
- MCP in-process: actually call the `search_code` tool via `mcp.shared.memory.create_connected_server_and_client_session` etc.
  and verify the JSON result schema (source/ref/date/score/text).
  (if hard for the SDK, substitute a direct call to `mcp_server._run_search` and note the reason in a comment)
- LLM integration (answer.plan) is **just one**: a code question → source is "code" or "all" and queries include the original
  query. (loose assertions only, given LLM non-determinism)

### 4. Register the marker in `pyproject.toml`

```toml
[tool.pytest.ini_options]
markers = ["integration: requires local DB/vLLM services"]
testpaths = ["tests"]
```

## Acceptance criteria

1. `uv run pytest -m "not integration" -q` → the existing 15 + new units all pass, no network/DB access.
2. `uv run pytest -m integration -q` → passes (run against real services in this environment).
3. `uv run pytest -q` → all pass.
4. No diff in src/ files (confirm with `git status`; only the pyproject.toml marker addition is allowed).

## Forbidden

- Do not modify src/ code (on finding a bug, xfail + report).
- Do not add new dependencies (no pytest-asyncio — use the asyncio.run() pattern for async, see existing tests).
- Do not write rows to the DB in integration tests (read-only; do not run history_index).
