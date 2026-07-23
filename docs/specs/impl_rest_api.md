# REST API server as the single backend (#31)

All architecture decisions below are user-confirmed via interview. This spec elaborates them into
an implementation plan; it does not re-decide anything. Open points that surface during
implementation go back to the user as questions.

## 1. Goal

A REST API server becomes the single backend for memory_base. It owns DB access, search, and
storage. The MCP server becomes a thin proxy over REST and keeps exposing exactly the four
existing tools (`search`, `search_code`, `search_history`, `save_memory`) with unchanged result
shapes. Access logging (`last_hit_at` / `hit_count` columns + `retrieval_log` table) ships with
the REST server as the prerequisite for the cold tier (#30).

Out of scope here (follow-ups): the ingestion push endpoint and SessionEnd hook (#32),
management endpoints (#30).

## 2. Components

One module per responsibility; HTTP handlers contain no business logic.

```
src/memory_base/
  schema.py        # NEW: memory_chunks / retrieval_log DDL (_ensure_schema moves here
                   #      from ingest/history.py — serve must not import from ingest)
  serve/
    api.py         # NEW: Starlette app + routes only (parse/validate -> delegate -> JSON)
    notes.py       # NEW: build_note_row + save_note (validation + DB write),
                   #      moved out of mcp_server.py
    access_log.py  # NEW: log_retrieval (retrieval_log insert + hit-column update)
    mcp_server.py  # REWRITTEN: FastMCP tool definitions + httpx proxy calls, nothing else
```

`ingest/history.py` re-exports `_ensure_schema` from `schema.py` so existing imports keep
working; the DDL itself no longer lives in the ingest layer. (The broader split of
`history.py` — parse/triage/distill/pipeline — is not forced here; #32 rewires ingestion
through the server anyway and is the natural point to extract the pipeline. If wanted earlier,
it becomes its own refactor issue.)

Framework: **Starlette + uvicorn**, both already installed transitively via the `mcp`
dependency — no new dependencies. Request validation is manual (5 small endpoints); FastAPI is
not needed.

## 3. REST surface

All endpoints return JSON. Validation failures return `400 {"error": "<message>"}`.

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/health` | – | `{"status": "ok"}` (also checks DB connectivity; 503 on failure) |
| POST | `/search` | `{"query": str, "source": "all"\|"code"\|"history" = "all", "top_k": int = 10}` | `[hit, ...]` |
| POST | `/save_memory` | `{"content": str, "kind": "note"\|"decision" = "note", "tags": [str] \| null}` | `{"id", "kind", "stored"}` |

`hit` is exactly the current `hit_to_dict` shape: `{source, ref, date, score, text[, context]}`.
One `/search` endpoint with a `source` parameter covers all three MCP search tools; each MCP tool
proxies with its fixed `source` value.

Search internals (`retrieval/search.py`) are unchanged. `build_note_row` and the save-memory
insert relocate from `mcp_server.py` into `serve/notes.py`; access logging lives in
`serve/access_log.py`; `api.py` routes only wire them together. The MCP server no longer
touches the DB.

No auth; the server binds inside the local Docker network and is published on localhost only
(user decision: local-only, no auth).

## 4. Access logging

Added to `_ensure_schema` (idempotent DDL):

```sql
ALTER TABLE memory_chunks ADD COLUMN IF NOT EXISTS last_hit_at double precision;
ALTER TABLE memory_chunks ADD COLUMN IF NOT EXISTS hit_count bigint NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS retrieval_log (
  id bigserial PRIMARY KEY,
  query text NOT NULL,
  source text NOT NULL,
  hit_ids text[] NOT NULL,
  ts double precision NOT NULL
);
```

On every `/search` response:

- `retrieval_log` gets one row: the query, the requested source, the refs/ids of all returned
  hits (code hits log their `ref`, history/note hits their chunk id), and the timestamp.
- `memory_chunks` rows among the hits get `hit_count = hit_count + 1, last_hit_at = <now>` in one
  `UPDATE ... WHERE id = ANY($1)`. Code hits live in the CocoIndex-managed table and carry no
  `memory_chunks` id, so they appear in `retrieval_log` only — the cold tier (#30) targets
  `memory_chunks`, so this is sufficient.

Logging failures must not break search: log-write errors are caught and reported in server logs,
the search response still returns.

## 5. MCP thin proxy

`mcp_server.py` keeps FastMCP, the four tool signatures, and the tool docstrings (they are the
agent-facing contract). Each tool body becomes an httpx call:

- `search` / `search_code` / `search_history` → `POST {REST_URL}/search` with the fixed source.
- `save_memory` → `POST {REST_URL}/save_memory`; a 400 response is re-raised as `ValueError` with
  the server's error message, preserving current tool behavior.

`REST_URL` comes from the environment (default `http://localhost:8010`). The MCP server imports
nothing from `ingest/` or `retrieval/` anymore and opens no DB connections.

## 6. Docker

Both services build from the same image; compose overrides the command:

```yaml
  api:
    build: .
    container_name: memory_base_api
    depends_on: [db]
    ports: ["8010:8010"]
    environment:
      DB_URL: postgres://memory:memory@db:5432/memory_base
      EMB_URL/RERANK_URL/LLM_URL: from .env
    command: uv run --no-sync uvicorn memory_base.serve.api:app --host 0.0.0.0 --port 8010

  mcp:
    depends_on: [api]
    environment:
      MCP_TRANSPORT: sse
      REST_URL: http://api:8010
    # DB_URL no longer needed
```

MCP registration for Claude Code is unchanged (`--transport sse ... :8765/sse`).

## 7. Tests (TDD)

Unit (CI, no DB):

- Route validation: empty query → 400; bad `source` → 400; save_memory validation errors
  (empty / >4000 chars / bad kind) → 400 with the same messages as today.
- `build_note_row` pins keep passing from their new location.
- MCP proxy: tools call the expected REST paths/payloads and map responses (httpx transport
  mocked via `httpx.MockTransport`); tool list stays exactly
  `{search, search_code, search_history, save_memory}`.

Integration (real DB + vLLM):

- `/search` returns the same result shape and content as calling `search()` directly.
- After a `/search` with N history hits: `retrieval_log` has one new row with N ids, and each hit
  row's `hit_count` incremented and `last_hit_at` set.
- `/save_memory` stores, dedupes (`stored: false` on repeat), and the note is findable via
  `/search`.
- MCP SSE round-trip: tools proxied through a running REST container return correct results.

## 8. Runtime evidence (PR)

- `docker compose up -d --build api mcp`; `curl /health`, `curl /search`, `curl /save_memory`
  transcripts.
- Real MCP client call through SSE showing identical tool output to pre-change behavior.
- `SELECT` from `retrieval_log` and hit columns after the calls.
