# Shared asyncpg connection pool (issue #74)

## Goal

Replace per-call `asyncpg.connect(db_url())` in the request paths (`serve/`,
`retrieval/`) with one process-wide `asyncpg.create_pool()`. Query bodies stay
unchanged; only connection acquisition changes.

## Design

### New module: `src/memory_base/core/db.py`

```python
_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()

async def get_pool() -> asyncpg.Pool:
    """Lazy-create the process-wide pool; run ensure_schema_once on first creation."""

async def close_pool() -> None:
    """Close and drop the pool; safe to call when no pool exists."""

@asynccontextmanager
async def acquire() -> AsyncIterator[asyncpg.Connection]:
    """async with acquire() as conn: — acquire a connection from the shared pool."""
```

- `get_pool()` double-checks under `_pool_lock`, calls
  `asyncpg.create_pool(db_url())` with library-default sizing, and runs
  `core.schema.ensure_schema_once` on one acquired connection before returning.
- `acquire()` is the single entry point call sites use. It exists (rather than
  exposing the pool directly) so issue #73 can later wrap the acquisition in a
  transaction with `SET LOCAL app.user_id` without touching call sites again.
- `close_pool()` resets the module global so tests can cycle the pool.

### Application shutdown

`serve/api.py` gains a `lifespan` context manager passed to `Starlette(...)`
that calls `close_pool()` on shutdown. No startup work: the pool stays lazy so
importing the app never requires a database.

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
| `retrieval/search.py` | 1 (`run_search`) |
| `retrieval/decompose.py` | 1 |

Per-call `ensure_schema_once(conn)` invocations at these sites are removed —
the pool runs it once at creation. Unused `asyncpg` / `db_url` /
`ensure_schema_once` imports are cleaned up per file.

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
   - `get_pool()` returns the same object across calls; `close_pool()` resets it
     (integration mark — needs the DB).
2. `test:` integration — repeated search requests keep the
   `pg_stat_activity` backend count flat (no new backend per request).
3. `feat:` — `core/db.py`, lifespan, call-site sweep.

Existing unit and integration suites must stay green:
`uv run pytest -m "not integration"` and, with the DB up, `uv run pytest`.

## Acceptance criteria (from #74)

- No `asyncpg.connect(db_url())` remains in `serve/` or `retrieval/` request paths.
- Pool closed cleanly on application shutdown.
- Backend count in `pg_stat_activity` stays flat under repeated search requests.
- Existing tests stay green.
