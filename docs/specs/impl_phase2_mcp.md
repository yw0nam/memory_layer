# Implementation spec: Phase 2 — Planner→Executor→Synthesis + MCP thin tools

## Goal

Implement two files.

1. `src/answer.py` — a CLI: query → Planner (source selection) → Executor (hybrid search) → Synthesis (answer with citations).
2. `src/mcp_server.py` — expose the search primitives as thin tools of an MCP stdio server: `search`, `search_code`, `search_history`.

## Context (existing code — do not modify, read and conform)

- Project root: `/home/spow12/codes/2026_upper/agents/memory/memory_layer/repos/memory_base`
- Python 3.12, uv project. Deps: openai (AsyncOpenAI), asyncpg, httpx, numpy, python-dotenv installed.
- `src/common.py`: `llm_client() -> AsyncOpenAI` (vLLM, base_url is LLM_URL from .env), `LLM_MODEL`.
- `src/search.py`: `async def search(query: str, source: str = "all", rerank: bool = True) -> list[Hit]`.
  `Hit` fields: `source` ("code"|"history"), `ref` (file:line or session ref), `text`, `ts` (epoch sec),
  `rrf`, `rerank_score`, `meta` (dict; for code, "context" may hold adjacent chunks).
  It already performs FTS + vector + time-decay + RRF + reranker + context restoration in full. **Do not reimplement the search logic.**
- Always run from the repository root as `uv run python src/answer.py ...`. Modules under src/ use
  flat imports like `from common import ...` between themselves (keep the existing convention).

## 1) `src/answer.py`

```
uv run python src/answer.py "question" [--source auto|code|history|all]
```

- **Planner**: calls the LLM once only when `--source auto` (default). Given the query, it returns JSON
  `{"source": "code"|"history"|"all", "queries": ["1–3 search terms"]}`
  (response_format json_object). code if the query is a code-structure question; history for retrospective
  questions like "how did I solve X before/back then"; all if ambiguous. queries are the original query
  reshaped to be search-friendly (the original query itself must always be included).
- **Executor**: for each query, run `search.search(q, source=decided source)` **in parallel via
  asyncio.gather** → dedup Hits by (source, ref), then top-10 by descending rerank_score (or rrf if absent).
- **Synthesis**: one LLM call. State the principles in the system prompt:
  "Use only the provided evidence; cite after each claim in the form [1][2]; say so if evidence is insufficient;
  when stale info (old ts) conflicts with recent info, prefer the recent one and note the caveat; answer in Korean."
  The user message contains numbered evidence blocks (each with source/ref/date (ts as YYYY-MM-DD)/text, text truncated to 2000 chars,
  plus meta["context"] for code Hits if present) + the original question.
- Output: the answer body + a "References:" section at the end (number → ref mapping).
- If there are 0 pieces of evidence, print "No relevant evidence found" without calling the LLM.

## 2) `src/mcp_server.py`

- Dependency: `uv add mcp` (official Python SDK). Use FastMCP (`from mcp.server.fastmcp import FastMCP`).
- stdio server, server name "memory-base". **No LLM calls** — thin search tools only (spec §3.2 ⑦).
- Three tools, all delegating to `search.search()`:
  - `search(query: str, top_k: int = 10)` → source="all"
  - `search_code(query: str, top_k: int = 10)` → source="code"
  - `search_history(query: str, top_k: int = 10)` → source="history"
- Returns: a JSON-serializable list[dict] — `{"source", "ref", "date" (YYYY-MM-DD), "score" (rerank_score
  or rrf), "text" (2000-char truncated), "context" (if present)}`. Truncated by top_k.
- Write the docstrings faithfully: the tool description is the manual the agent sees. Specify when to use which tool.
- Run: `uv run python src/mcp_server.py` brings up the stdio server.
- Instead of updating the README, put a one-line Claude Code registration example in the file's top docstring:
  `claude mcp add memory-base -- uv --directory <abs-path> run python src/mcp_server.py`

## Tests (required deliverable)

`tests/test_answer_mcp.py` — must run **without LLM/DB** (pytest; add via `uv add --dev pytest` if not already):
- Split answer.py's evidence dedup/sort/evidence-block-format functions into pure functions and unit-verify them.
- Split mcp_server's Hit→dict conversion into a pure function and unit-verify it (create Hit as search.Hit directly).
- MCP server in-process check: use `mcp.shared.memory.create_connected_server_and_client_session` or
  FastMCP's built-in test utility to confirm the three tools appear in tools/list. If this is hard for the SDK version,
  substitute a check that the tool registration objects exist (but leave the reason in a test comment).
- Stub `search.search` via monkeypatch in the checks to block network/DB access.

## Acceptance criteria

1. `uv run pytest tests/test_answer_mcp.py` passes.
2. `uv run python src/answer.py "where is RRF fusion implemented?" --source code` prints a Korean answer with citations
   (real LLM/DB required — verify in the implementer's environment).
3. `uv run python src/mcp_server.py` starts without error (waiting on stdio).

## Forbidden

- Do not modify `src/common.py`, `src/search.py`, `src/code_index.py`, `src/history_index.py`.
- Do not reimplement the search pipeline (RRF/rerank, etc.) — only call search.search().
- Do not add new dependencies beyond mcp.
