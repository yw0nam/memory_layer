# Configuration

`.env` (gitignored) holds endpoints and credentials — never hardcode them.

| variable | purpose |
|---|---|
| `LLM_URL`, `EMB_URL`, `RERANK_URL` | vLLM OpenAI-compatible endpoints |
| `LLM_MODEL`, `EMB_MODEL`, `RERANK_MODEL` | model names |
| `DB_URL` | Postgres connection string |
| `DB_POOL_MIN`, `DB_POOL_MAX` | asyncpg pool size bounds (default `1` / `10`) |
| `DB_POOL_ACQUIRE_TIMEOUT` | seconds to wait for a pooled connection before failing (default `30`) |
| `POSTGRES_PASSWORD` | required; consumed by docker-compose for the db service and the api `DB_URL` |
| `TABLES_QUERY_PASSWORD` | required; login password of `memory_tables_query`, the restricted role the SQL table lane connects as — set and kept current by `ensure_schema` at startup |
| `DATA_ROOT` | required; host directory docker-compose mounts persisted state under (`pgdata`, `repos_cache`, `cocoindex_state`, `ingest-spool`) |
| `REPO_CACHE` | git checkout root |
| `INGEST_SPOOL` | durable uploaded-document spool root |
| `REST_URL` | backend the MCP server proxies to |
| `MEMORY_API_KEY` | the MCP server's `X-API-Key` over stdio transport; streamable HTTP forwards the caller's own header instead |
| `MCP_ALLOWED_HOSTS` | comma-separated `Host` header values accepted by MCP HTTP transports; empty keeps the loopback-only SDK default |
| `LOG_DIR` | file-sink directory (default `logs/`) |

Tuning knobs, all optional: `NOTE_SIMILAR_THRESHOLD`, `INGEST_MAX_BYTES`,
`INGEST_BACKLOG_PER_KEY`, `INGEST_BACKLOG_MAX`, `INGEST_MAX_CONCURRENT_JOBS`,
`REPO_MAX_QUEUED`, `REPO_MAX_BYTES`, `REPO_DISK_HEADROOM_BYTES`, `JOB_RETENTION_SECONDS`,
`COLD_AGE_DAYS`, `COLD_UNHIT_DAYS`, `HIT_FLUSH_INTERVAL_SECONDS`,
`RETRIEVAL_LOG_RETENTION_DAYS`.

## Private repositories

`git-credentials` (gitignored) at the repo root holds one credential line per host and is
mounted read-only at `/run/git-credentials`, where the container's system-wide
`credential.helper` reads it. Create it before starting the api container — Docker mounts
a directory in its place otherwise:

```bash
touch git-credentials                                   # empty — public repos only
```

Each line is a URL carrying the credential for one host, with any `@` or `:` inside the
username or token percent-encoded:

```
https://user:token@github.com
https://x-bitbucket-api-token-auth:api-token@bitbucket.org
```

Bitbucket API tokens authenticate git over https with the static username
`x-bitbucket-api-token-auth` (the account email is for the REST API only). Credentials never
enter repo URLs or git remotes, so `GET /repos` and job logs cannot leak them.
`GIT_TERMINAL_PROMPT=0` makes a clone with no matching credential fail immediately instead
of waiting on a prompt.

`COCOINDEX_DB` is set by `docker-compose.yml`, not by `.env` — together with `REPO_CACHE`
and `INGEST_SPOOL` it points at a container-local path bound under `DATA_ROOT` on the host,
so one ledger tracks one repo cache and uploaded documents remain available across API
restarts. Losing the repo cache or the ledger orphans `code_chunks` rows until the repos are
re-added.
