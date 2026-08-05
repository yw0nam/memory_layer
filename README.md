# memory-base

A selective memory layer for coding agents: distilled notes, chunked documents, and
indexed code in one pgvector store, served through a single REST API and an MCP server.

Only high-signal content is stored. Agent notes arrive already distilled, documents pass
through deterministic chunking and a junk gate before embedding, and code is chunked by
tree-sitter. Raw transcripts and raw files are never embedded.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  CONSUMERS      coding agents · n8n · scripts                                 │
└───────┬──────────────────────────────────────────────────────────────────────┘
        │ MCP  (stdio | SSE :8765)
        ▼
   ┌─────────────┐  8 tools: search / search_code / search_memory / save_memory
   │ mcp_server  │           ingest_document / ingest_repo / remove_repo
   └──────┬──────┘           list_repos                    (thin proxy, no logic)
          │ HTTP
          ▼
   ╔═══════════════════════════════════════════════════════════╗
   ║              REST API  :8010   (the only backend)         ║
   ╚═╤═══════════════╤═══════════════╤═══════════════╤═════════╝
     │ WRITE         │ WRITE         │ WRITE         │ READ + LIFECYCLE
     ▼               ▼               ▼               ▼
  notes.py     ingest_api.py     repos.py      search.py · admin.py
     │               │               │               │
     └───────────────┴───────┬───────┴───────────────┘
                             ▼
              Postgres 17 + pgvector + pg_textsearch  :5439
              memory_chunks · code_chunks · jobs · retrieval_log

  side services:  vLLM (LLM / embedding / rerank)
```

Every consumer reaches stored chunks through the REST API, never through the database
directly.

## Data flow

### Write paths

```
① NOTE                  ② DOCUMENT                    ③ CODE
POST /save_memory       POST /ingest/document         POST /repos {url}
   │                       │ 202 {job_id}                │ 202 {job_id}
   ▼                       ▼                             ▼
 validate ≤4000         MarkItDown worker            url validated
 kind ∈ note|decision   (killable, 120 s)            free disk? → 507
   │                       │                             │
   ▼                       ▼                             ▼
 id = sha256(content)   chunk 1500 / 2000 / 200      git clone --filter=blob:none
 └ same content         junk gate ✂                  + size watchdog (2 GiB)
   → idempotent no-op      │                             │
   │                       ▼                             ▼
   ▼                    doc rows ────┐               cocoindex update
 embed (vLLM)           caller tags  │               tree-sitter 1000 / 300
   │                       ▼         │                  │  mtime = commit time
   ▼                    heading-path embedding text     ▼
 INSERT                    │  embed  │               incremental per file
 ON CONFLICT               ▼         │               (ledger: COCOINDEX_DB)
 DO NOTHING             one transaction per document     │
   │                       │                             │
   ▼                       ▼                             ▼
┌──────────────────────────────────────┐      ┌────────────────────────┐
│ memory.memory_chunks                 │      │ memory.code_chunks     │
│ note │ decision │ doc │ atom         │      │ repo · file · L1-L40   │
│ halfvec(2048) + HNSW + BM25          │      │ halfvec + HNSW + BM25  │
└──────────────────────────────────────┘      └────────────────────────┘
        ▲                                                ▲
        └──── the only contract between write and read ──┘
```

Notes are stored exactly as written — the server never summarizes. The response carries
`similar[]`: active notes above `NOTE_SIMILAR_THRESHOLD` cosine, so the caller can
consolidate instead of accumulating near-duplicates. A prior-note id in the payload
archives that row.

Document uploads enter a durable Postgres backlog capped by `INGEST_BACKLOG_PER_KEY` and
`INGEST_BACKLOG_MAX`. Two document workers dispatch fairly across API keys while serializing
jobs for the same document. Jobs and their spooled uploads survive API restarts; startup
requeues interrupted work and fails a job clearly when its spool file is missing. Jobs are
observable at `GET /ingest/jobs/{job_id}` and listable at `GET /ingest/jobs`, with optional
`origin` and `status` filters. Their stages are `queued → converting → chunking →
embedding → writing → done`, with `chunks_total`, `chunks_done`, `chunks_dropped`,
`rows_written`, and `enrichment_retries`. Re-uploading identical bytes in `upsert` mode
short-circuits to `no_op`. Markdown ingest calls no LLM: chunks are stored as written, and
the optional repeated `tags` upload field lands on every chunk of the document. CSV takes a
sampling branch instead: header plus the first 20 rows become one LLM-summarized knowledge
card through an extra `enriching` stage.

Repo jobs use the same durable jobs table and dispatch one at a time, so interrupted clone,
pull, remove, and index work is retried after an API restart.

The code indexer mounts every subdirectory of `REPO_CACHE` as an independent codebase, so
adding or removing a checkout adds or tears down its rows on the next run. Clones keep
full history (`--filter=blob:none`, lazy blobs) precisely so each file carries its real
last-commit time into time-decay scoring.

`POST /repos` accepts http(s) URLs only, and rejects credentials embedded in the URL.
Private repositories authenticate through the git credential store — see
[Private repositories](#private-repositories).

A repo's owner is the key label that first ingested it, recorded in
`REPO_CACHE/.owners/<name>` after the first successful clone or pull and never
transferred by later re-ingests. `DELETE /repos/{name}` is restricted to the owner or an
admin key; a repo with no owner record (ingested before ownership tracking, or never
successfully ingested) is admin-only to remove. `GET /repos` reports each repo's owner,
`null` when unrecorded.

### Read path — hybrid search (`POST /search`)

```
      "query"
         │
         ├─ embed (Qwen3 instruction prefix) ──┐
         │                                     │
 ┌───────┴────────────┐              ┌─────────┴──────────┐
 │    code_chunks     │              │   memory_chunks    │  atom + archived excluded
 │ vec50 · fts50 · rec│              │vec50·fts50·rec·idf │  optional kind / tags
 │  optional repo     │              │                    │
 └───────┬────────────┘              └─────────┬──────────┘
         └──────────────┬───────────────────────┘
                        ▼
                RRF   Σ w/(60 + rank)   w: vec 1.0 · fts 0.2 · rec/idf 0.25
                        ▼
                ⏳ time decay (90-day half-life)
                        ▼
                per-file / per-session cap 3  → top 20
                        ▼
                + atom lane merged in
                  (question embeddings pull their parent rows up)
                        ▼
                🎯 rerank (vLLM) → top 10
                        ▼
                code hits get ±40-line neighbour chunks as `context`
                        ▼
                    hits[]  ─────► in-process buffer
                                   (flushed on an interval)
```

Response text is truncated to 2000 chars. `score` is the rerank score, falling back to the
fused RRF score. `include_archived` surfaces archived rows and turns recency decay off for
memory so they are not buried; code hits have no archived state and keep decaying. `/search`
hits mark archived rows `"archived": true` since an archived note may have been superseded by
a newer one.

The read path writes nothing to the database. Returned hit ids and per-chunk counters land
in an in-process buffer that a background task flushes every `HIT_FLUSH_INTERVAL_SECONDS`
(default 30) and on shutdown: one batched `retrieval_log` insert plus one deduplicated
`hit_count` update, so repeated hits on a popular row collapse into a single `+ n`. The same
cycle prunes `retrieval_log` rows older than `RETRIEVAL_LOG_RETENTION_DAYS` at startup and
then at most hourly. An unclean stop loses at most one interval of counters, which only feed
lifecycle decisions.

### Lifecycle loop

```
 row returned by a search ──► buffered, then flushed
                              hit_count += n , last_hit_at
                                   │
                                   ▼
        ts_last_active older than COLD_AGE_DAYS
        AND unhit for COLD_UNHIT_DAYS
                                   │
                                   ▼
                    POST /admin/archive          (preview)
                                   │  {"confirm": true}
                                   ▼
                            archived_at set
                                   │
          excluded from search ◄───┴───► include_archived=true brings it back
                                          (memory decay off, code decay on)
```

`GET /admin/duplicates` lists near-duplicate pairs by cosine, `GET /admin/notes` lists old
agent notes, and `POST /admin/restore` clears `archived_at`. Every mutating admin route
previews by default and acts only with `{"confirm": true}`.

## Retrieval quality

Doc-level scores on two corpora — ZX Bank (RAG-Multi-Corpus, 71 docs / 100 queries,
document ingest pipeline) and BEIR SciFact (5,183 docs / 300 queries):

| corpus | mode | hit@5 | hit@10 | MRR@10 |
|---|---|---|---|---|
| ZX Bank | vector only | 0.93 | 0.95 | 0.84 |
| ZX Bank | hybrid | 0.93 | **0.98** | 0.81 |
| ZX Bank | hybrid + rerank (default) | 0.93 | 0.95 | 0.81 |
| SciFact | vector only | 0.82 | 0.87 | 0.69 |
| SciFact | hybrid | 0.82 | 0.89 | **0.71** |
| SciFact | hybrid + rerank (default) | **0.86** | **0.90** | **0.75** |

The BM25 leg adds recall the embedder misses (ZX hit@10 0.98 vs 0.95 vector-only) and
lifts hybrid above vector-only outright on SciFact. Method, single-leg ablations, and
known trade-offs: [docs/benchmarks/retrieval.md](docs/benchmarks/retrieval.md).

## Storage

`memory.memory_chunks` — one table for every non-code source.

| column | meaning |
|---|---|
| `id` | `note:<hash>` · `doc:<document_id>:<ordinal>` · `…:atom:<n>` |
| `source_type` | `agent_note` · `document` |
| `source_ref` | `save_memory` or the document id |
| `chunk_kind` | `note` · `decision` · `doc` · `atom` |
| `content_raw` / `distilled` | stored text; BM25 index on `content_raw`, hits display `distilled` first |
| `embedding` | `halfvec(2048)`, HNSW cosine index |
| `ts_last_active`, `idf_score` | ranking signals |
| `metadata` | jsonb: `tags`, `parent_id`, `heading_path`, `content_hash`, `search_ref`, … |
| `hit_count`, `last_hit_at`, `archived_at` | lifecycle counters |

`memory.code_chunks` — written and torn down entirely by CocoIndex: `repo`, `filename`,
`code`, `embedding`, `start_line`, `end_line`, `mtime` (last commit time).

These two tables are the only contract between the write side and the read side. Adding a
source means adding an adapter, not touching retrieval or serving.

## REST API

| method | path | purpose |
|---|---|---|
| `GET` | `/health` | liveness — `200 {status}` whenever the process serves HTTP; reaches nothing outside it, and backs the container healthcheck |
| `GET` | `/health/services` | dependency health — `{status, checks:{db, embedding, rerank, llm}}`; `503` when db, embedding, or rerank is down |
| `POST` | `/search` | hybrid search — `query`, `source` (`all`\|`code`\|`memory`), `top_k`, `kind`, `tags`, `repo`, `include_atoms`, `include_archived` |
| `POST` | `/save_memory` | store a distilled note — `content`, `kind`, `tags`, and the optional id of a prior note to archive |
| `POST` | `/ingest/document` | multipart upload — `file`, `document_id`, `mode` (`upsert`\|`force`), `origin`, repeated `tags` |
| `GET` | `/ingest/jobs` | newest document jobs, optionally filtered by exact `origin` and `status` |
| `GET` | `/ingest/jobs/{job_id}` | document job state |
| `POST` | `/repos` | add or re-sync a git repo — `url`, `branch`, `name` |
| `GET` | `/repos` | cached repos with url, branch, head, chunk count, owner |
| `DELETE` | `/repos/{name}` | remove a repo and re-index — the repo's owner or an admin key only |
| `GET` | `/repos/jobs/{job_id}` | repo job state |
| `POST` | `/namespaces` | register a namespace — `name` (`^[a-z0-9_-]{1,64}$`), `visibility` (`public`\|`private`, default `public`); a private namespace records the caller's key label as owner |
| `GET` | `/namespaces` | list namespaces the caller can access (every namespace for an admin key) |
| `DELETE` | `/namespaces/{name}` | unregister an empty namespace — the namespace's owner or an admin key only; the reserved `default` namespace cannot be deleted |
| `GET` | `/admin/notes` | active agent notes older than `older_than_days` |
| `POST` | `/admin/notes/delete` | preview, or delete with `confirm` |
| `GET` | `/admin/duplicates` | near-duplicate pairs above `threshold` |
| `POST` | `/admin/archive` | preview cold rows, or archive with `confirm` |
| `POST` | `/admin/restore` | preview, or restore with `confirm` |

Filters are bound to the source they belong to: `kind` and `tags` require
`source="memory"`, `repo` requires `source="code"`, and `source="all"` takes neither.
`repo` is a list of cache directory names as reported by `GET /repos`; an unknown name
matches nothing. Code hits carry their `repo` whether or not the filter is set.

## MCP tools

`mcp_server.py` is a thin proxy over the REST API — no logic of its own. Transport is
stdio by default; `MCP_TRANSPORT=sse|streamable-http` with `MCP_HOST`/`MCP_PORT` serves
over HTTP (Docker serves streamable HTTP on `:8765/mcp`).

`search` · `search_code` · `search_memory` · `save_memory` ·
`ingest_document` (text formats only) · `ingest_repo` · `remove_repo` · `list_repos`

Each tool takes the REST options its source supports: `include_archived` on `search` and
`search_memory`; `kind` and `tags` only where `source="memory"` holds,
so `search` and `search_code` do not offer them; `repo` on `search_code` alone. Lifecycle
routes (`/admin/*`) have no MCP tool — archiving and restoring stay operator actions, while
reading archived rows does not.

## Running

```bash
uv sync
docker compose up -d --build db          # pgvector + pg_textsearch on :5439
docker compose up -d --build api         # REST backend on :8010
docker compose up -d --build mcp         # MCP server, streamable HTTP on :8765
claude mcp add --transport http memory-base http://localhost:8765/mcp \
  --header "X-API-Key: <key>"            # mint one — see Authentication
```

Index cached repos manually (the repo routes do this for you). The indexer runs inside the
API container, which owns the only copy of the ledger and the repo cache:

Do not run `cocoindex update` manually while an API repo job is active; manual runs are not
covered by the server's repo-job serialization.

```bash
docker compose exec api uv run cocoindex update src/memory_base/ingest/code.py     # incremental
docker compose exec api uv run cocoindex update -L src/memory_base/ingest/code.py  # live watch
uv run python -m memory_base.retrieval.search "your query" --source code
```

The memory schema is created on first write (`ensure_schema`); the code table is created
by the indexer.

## Authentication

Every route except `/health` and `/health/services` requires an `X-API-Key` header
(`ApiKeyAuthMiddleware` in `serve/auth.py`); a missing, unknown, or revoked key gets a
fail-closed `401`.

Keys are provisioned with an operator CLI, not through the API:

```bash
uv run python -m memory_base.serve.keys new <label> [--home <namespace>] [--admin]
uv run python -m memory_base.serve.keys list
uv run python -m memory_base.serve.keys revoke <key-hash-prefix>
```

`new` prints the plaintext key once — only its sha256 hash is stored. `--home` sets the
namespace `save_memory` and document ingest default into (`default` when omitted);
minting fails if that namespace does not exist or is not accessible to the label.
`revoke` takes an 8+ character prefix of the stored hash, as shown by `list`, and
revokes every active key matching it.

An admin key (`--admin`) can read and act in every namespace. A member key's allowed
set is every public namespace plus any private namespace it owns — ownership is set to
the minting key's label when the namespace is created with `visibility: private`.
Requests naming a namespace outside that set get `403`.

The MCP server needs the same header: over streamable HTTP it forwards the caller's own
`X-API-Key`, and over stdio (no inbound HTTP request to read one from) it reads the
`MEMORY_API_KEY` environment variable instead.

## Configuration

`.env` (gitignored) holds endpoints and credentials — never hardcode them.

| variable | purpose |
|---|---|
| `LLM_URL`, `EMB_URL`, `RERANK_URL` | vLLM OpenAI-compatible endpoints |
| `LLM_MODEL`, `EMB_MODEL`, `RERANK_MODEL` | model names |
| `DB_URL` | Postgres connection string |
| `DB_POOL_MIN`, `DB_POOL_MAX` | asyncpg pool size bounds (default `1` / `10`) |
| `DB_POOL_ACQUIRE_TIMEOUT` | seconds to wait for a pooled connection before failing (default `30`) |
| `POSTGRES_PASSWORD` | required; consumed by docker-compose for the db service and the api `DB_URL` |
| `DATA_ROOT` | required; host directory docker-compose mounts persisted state under (`pgdata`, `repos_cache`, `cocoindex_state`, `ingest-spool`) |
| `REPO_CACHE` | git checkout root |
| `INGEST_SPOOL` | durable uploaded-document spool root |
| `REST_URL` | backend the MCP server proxies to |
| `MEMORY_API_KEY` | the MCP server's `X-API-Key` over stdio transport; streamable HTTP forwards the caller's own header instead |
| `LOG_DIR` | file-sink directory (default `logs/`) |

Tuning knobs, all optional: `ATOM_RETRIEVE_K`, `ATOMS_RETRIEVE`,
`NOTE_SIMILAR_THRESHOLD`, `INGEST_MAX_BYTES`,
`INGEST_BACKLOG_PER_KEY`, `INGEST_BACKLOG_MAX`, `INGEST_MAX_CONCURRENT_JOBS`,
`REPO_MAX_QUEUED`, `REPO_MAX_BYTES`, `REPO_DISK_HEADROOM_BYTES`, `JOB_RETENTION_SECONDS`,
`COLD_AGE_DAYS`, `COLD_UNHIT_DAYS`, `HIT_FLUSH_INTERVAL_SECONDS`,
`RETRIEVAL_LOG_RETENTION_DAYS`.

### Private repositories

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

## Development

```bash
uv run pytest                                        # integration tests skip without DB/vLLM
uv run pytest -m "not integration"                   # what CI runs
uv run ruff format --check . && uv run ruff check .
uv run python -m memory_base.eval.retrieval          # retrieval eval with atom-lane A/B
```

Work happens in a git worktree and lands via PR; `main` requires a PR and green CI (lint,
unit tests, test-guard, PR title). See `AGENTS.md` for the full contributor contract.
