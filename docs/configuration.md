# Configuration

`.env` (gitignored) holds endpoints and credentials — never hardcode them.

| variable | purpose |
|---|---|
| `EMB_URL`, `RERANK_URL` | vLLM OpenAI-compatible endpoints for embedding and rerank |
| `EMB_MODEL`, `RERANK_MODEL` | embedding and rerank model names |
| `DB_URL` | Postgres connection string |
| `DB_POOL_MIN`, `DB_POOL_MAX` | asyncpg pool size bounds (default `1` / `10`) |
| `DB_POOL_ACQUIRE_TIMEOUT` | seconds to wait for a pooled connection before failing (default `30`) |
| `POSTGRES_PASSWORD` | required; consumed by docker-compose for the db service and the api `DB_URL` |
| `DATA_ROOT` | required; host directory docker-compose mounts the Postgres data directory (`pgdata`) under |
| `REST_URL` | backend the MCP server proxies to |
| `MEMORY_API_KEY` | the MCP server's `X-API-Key` over stdio transport; streamable HTTP forwards the caller's own header instead |
| `MEMORY_AUTHOR` | the MCP server's `X-Author` over stdio transport; streamable HTTP forwards the caller's own header instead |
| `MCP_TRANSPORT` | `stdio` (default), `sse`, or `streamable-http` |
| `MCP_HOST`, `MCP_PORT` | bind address of the MCP HTTP transports (default `0.0.0.0` / `8765`) |
| `MCP_ALLOWED_HOSTS` | comma-separated `Host` header values accepted by MCP HTTP transports; empty keeps the loopback-only SDK default |
| `LOG_DIR` | file-sink directory (default `logs/`) |

Tuning knobs, all optional: `NOTE_SIMILAR_THRESHOLD`, `MESSAGE_TTL_DAYS` (1..30, default 7;
the expiry of a general message sent without `expires_at` — a completed handoff expires
after 3 days and another handoff has none; the server refuses to boot outside that
range).
