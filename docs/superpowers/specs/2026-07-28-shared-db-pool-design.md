# Shared asyncpg connection pool (issue #74)

## Goal

Replace per-call `asyncpg.connect(db_url())` in the request paths (`serve/`,
`retrieval/`) with one process-wide `asyncpg.create_pool()`. Query bodies stay
unchanged; only connection acquisition changes.

## Design

### New module: `src/memory_base/core/db.py`

```python
_pool: asyncpg.Pool | None = None
_pool_loop: asyncio.AbstractEventLoop | None = None
_pool_lock = asyncio.Lock()

async def get_pool() -> asyncpg.Pool:
    """Lazy-create the process-wide pool, bound to the running event loop."""

async def close_pool() -> None:
    """Close and drop the pool; safe to call when no pool exists."""

@asynccontextmanager
async def acquire() -> AsyncIterator[asyncpg.Connection]:
    """async with acquire() as conn: — acquire a connection from the shared pool."""
```

- `get_pool()` double-checks under `_pool_lock` and calls
  `asyncpg.create_pool(db_url(), min_size=DB_POOL_MIN, max_size=DB_POOL_MAX)`.
  Sizing is env-configurable (`DB_POOL_MIN` default 1, `DB_POOL_MAX` default 10)
  so the first use does not eagerly open asyncpg's default 10 connections and
  operators can tune per deployment.
- Initialization is failure-atomic: the pool is created into a local variable
  and published to the module global only after `create_pool` succeeds; on
  failure nothing is cached, so the next call retries cleanly.
- The pool is bound to the event loop that created it (`_pool_loop`). If
  `get_pool()` runs on a different loop (test suites drive the app through
  separate `asyncio.run()` calls, where lifespan never runs), the stale pool is
  discarded and a fresh one is created. Discarding without closing leaks the old
  loop's connections until process exit — acceptable because it only happens in
  test processes; long-lived servers stay on one loop.
- `acquire()` is the single entry point call sites use. It exists (rather than
  exposing the pool directly) so issue #73 can later wrap the acquisition in a
  transaction with `SET LOCAL app.user_id` without touching call sites again.
- Pooled connections are reused across requests: call sites must not leave
  session state behind (no session-level `SET`, temp tables, or LISTEN — none
  exist today; asyncpg's release-time reset covers transactions and session
  settings but not temp tables or prepared statements). Future per-user context
  (#73) must be `SET LOCAL` inside a transaction.
- `close_pool()` closes and resets the module globals so tests can cycle the
  pool. A background job that acquires after shutdown recreates a pool that
  dies unclosed with the process; accepted — Postgres reaps the backends, and
  draining job tasks at shutdown is the upgrade path if it ever matters.

### Application shutdown

`serve/api.py` gains a `lifespan` context manager passed to `Starlette(...)`
that calls `close_pool()` on shutdown. No startup work: the pool stays lazy so
importing the app never requires a database. Clients that never run lifespan
(`TestClient` without a context manager, `httpx.ASGITransport`) are covered by
the loop-binding in `get_pool()`.

### Call-site sweep

Replace the pattern

```python
conn = await asyncpg.connect(db_url())
try:
    await ensure_schema_once(conn)   # where present
    ...
finally:
    await conn.close()
```

with

```python
async with acquire() as conn:
    ...
```

at these request-path sites (16 total):

| File | Sites |
|---|---|
| `serve/api.py` | 1 (`db_healthy`) |
| `serve/notes.py` | 1 |
| `serve/admin.py` | 8 |
| `serve/access_log.py` | 1 |
| `serve/ingest_api.py` | 2 |
| `serve/repos.py` | 1 |
| `retrieval/search.py` | 1 (`search()`) |
| `retrieval/decompose.py` | 1 |

Sites that call `ensure_schema_once(conn)` today (`notes.py`, `ingest_api.py`)
keep that call inside their `acquire()` block — it is an in-process set check
after the first run, and keeping it preserves today's semantics exactly
(`db_healthy` stays a plain `SELECT 1` and never triggers DDL). Unused
`asyncpg` / `db_url` imports are cleaned up per file.

### No pooled connection held across external calls

A per-request connection could idle through slow model calls harmlessly; a
pooled one starves other requests. Invariant: **a pooled connection is never
held across an await of an embedding, LLM, or rerank HTTP call.**

- `retrieval/search.py` `search()`: the FTS/vector/atom queries run inside one
  `acquire()` block which is exited before `_rerank` (HTTP call);
  `_restore_context` then runs in a second short `acquire()` block.
- `retrieval/decompose.py`: acquire per DB step inside the multi-hop loop
  instead of one connection for the whole run; LLM/embedding awaits happen
  outside any `acquire()` block.

### Out of scope

- `eval/retrieval.py` and any `__main__` blocks keep direct connects (not
  request paths; the issue allows either).
- `ingest/code.py` already uses its own `create_pool` inside the CocoIndex
  batch path — unchanged.
- `serve/mcp_server.py` talks to the REST API only — unchanged.
- Issue #73 (per-user RLS): not implemented here; `acquire()` is the extension
  point it will build on.

## Testing (TDD order)

1. `test:` unit — `tests/core/test_db.py`:
   - source scan: no `asyncpg.connect(` remains anywhere under
     `src/memory_base/serve/` or `src/memory_base/retrieval/` (the CLI
     `__main__` in `retrieval/search.py` calls the search function, so no
     exemption is needed).
   - `get_pool()` returns the same object across calls on one loop;
     `close_pool()` resets it; a second `asyncio.run()` gets a fresh pool
     (integration mark — needs the DB).
2. `test:` integration — with the pool warmed first (`min_size` connections
   already open), repeated search requests keep the `pg_stat_activity`
   backend count flat.
3. `feat:` — `core/db.py`, lifespan, call-site sweep. Existing tests that
   monkeypatch `asyncpg.connect` or `ensure_schema_once` are updated to patch
   `core.db.acquire` / `get_pool` instead.

Existing unit and integration suites must stay green:
`uv run pytest -m "not integration"` and, with the DB up, `uv run pytest`.
Note for tests that rebuild schemas: `close_pool()` does not reset the
process-level `_prepared_schemas` guard in `core/schema.py` (same as today's
behavior); such tests call `ensure_schema` directly.

## Acceptance criteria (from #74)

- No `asyncpg.connect(db_url())` remains in `serve/` or `retrieval/` request paths.
- Pool closed cleanly on application shutdown.
- Backend count in `pg_stat_activity` stays flat under repeated search requests.
- Existing tests stay green.
