# memory-base

A selective memory layer for coding agents: distilled notes, enriched documents, and
indexed code in one pgvector store, served through a single REST API and an MCP server.

Only high-signal content is stored. Agent notes arrive already distilled, documents pass
through chunking and LLM enrichment before embedding, and code is chunked by tree-sitter.
Raw transcripts and raw files are never embedded.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  CONSUMERS      coding agents · n8n · scripts                                 │
└───────┬──────────────────────────────────────────────────────────────────────┘
        │ MCP  (stdio | SSE :8765)
        ▼
   ┌─────────────┐  9 tools: search / search_code / search_memory / deep_search
   │ mcp_server  │           save_memory / ingest_document / ingest_repo
   └──────┬──────┘           remove_repo / list_repos      (thin proxy, no logic)
          │ HTTP
          ▼
   ╔═══════════════════════════════════════════════════════════╗
   ║              REST API  :8010   (the only backend)         ║
   ╚═╤═══════════════╤═══════════════╤═══════════════╤═════════╝
     │ WRITE         │ WRITE         │ WRITE         │ READ + LIFECYCLE
     ▼               ▼               ▼               ▼
  notes.py     ingest_api.py     repos.py      search.py · decompose.py · admin.py
     │               │               │               │
     └───────────────┴───────┬───────┴───────────────┘
                             ▼
              Postgres 17 + pgvector  :5439
              memory_chunks · code_chunks · retrieval_log

  side services:  vLLM (LLM / embedding / rerank)    Redis :6379 (job state)
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
   ▼                    LLM enrich ──┐               cocoindex update
 embed (vLLM)           atom Q + tags│               tree-sitter 1000 / 300
   │                       ▼         │                  │  mtime = commit time
   ▼                    doc row + atom rows              ▼
 INSERT                    │  embed  │               incremental per file
 ON CONFLICT               ▼         │               (ledger: COCOINDEX_DB)
 DO NOTHING             one transaction per document     │
   │                       │                             │
   ▼                       ▼                             ▼
┌──────────────────────────────────────┐      ┌────────────────────────┐
│ memory.memory_chunks                 │      │ memory.code_chunks     │
│ note │ decision │ doc │ atom         │      │ repo · file · L1-L40   │
│ halfvec(2048) + HNSW + GIN FTS       │      │ halfvec + HNSW + FTS   │
└──────────────────────────────────────┘      └────────────────────────┘
        ▲                                                ▲
        └──── the only contract between write and read ──┘
```

Notes are stored exactly as written — the server never summarizes. The response carries
`similar[]`: active notes above `NOTE_SIMILAR_THRESHOLD` cosine, so the caller can
consolidate instead of accumulating near-duplicates. A prior-note id in the payload
archives that row.

Document jobs are bounded (`INGEST_MAX_QUEUED` queued, `INGEST_MAX_CONCURRENT_JOBS`
running) and observable at `GET /ingest/jobs/{job_id}`: `queued → converting → chunking →
enriching → embedding → writing → done`, with `chunks_total`, `chunks_done`,
`chunks_dropped`, `rows_written`, `enrichment_retries`. Re-uploading identical bytes in
`upsert` mode short-circuits to `no_op`. CSV takes a sampling branch instead: header plus
the first 20 rows become one knowledge card.

The code indexer mounts every subdirectory of `REPO_CACHE` as an independent codebase, so
adding or removing a checkout adds or tears down its rows on the next run. Clones keep
full history (`--filter=blob:none`, lazy blobs) precisely so each file carries its real
last-commit time into time-decay scoring.

### Read path 1 — hybrid search (`POST /search`)

```
      "query"
         │
         ├─ embed (Qwen3 instruction prefix) ──┐
         │                                     │
 ┌───────┴────────────┐              ┌─────────┴──────────┐
 │    code_chunks     │              │   memory_chunks    │  atom + archived excluded
 │ vec50 · fts50 · rec│              │vec50·fts50·rec·idf │  optional kind / tags
 └───────┬────────────┘              └─────────┬──────────┘
         └──────────────┬───────────────────────┘
                        ▼
                RRF   Σ 1/(60 + rank)
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
                    hits[]  ─────► retrieval_log INSERT
                                   hit_count++ / last_hit_at
```

Response text is truncated to 2000 chars. `score` is the rerank score, falling back to the
fused RRF score. `include_archived` surfaces archived rows and turns recency decay off so
they are not buried.

### Read path 2 — deep search (`POST /search/deep`)

Multi-hop decomposition over memory only, bounded by `DEEP_TIMEOUT_SECONDS` and
`DEEP_MAX_HOPS`.

```
 question
    │
    ▼
 ┌───────────────────────── hop loop ───────────────────────────┐
 │  🤖 propose    {continue, sub_questions ≤3}                   │
 │       │ continue = false ──────────────► stop "done"          │
 │       ▼                                                       │
 │  🔍 retrieve   sub-question embeddings → nearest atoms         │
 │       │        → best atom per parent → 8 candidates           │
 │       │        (parents already chosen are excluded)           │
 │       │        fallback: atoms for the original query,         │
 │       │                  then plain hybrid memory search       │
 │       ▼                                                       │
 │  🤖 select     one candidate, or null ──► stop "no_selection"  │
 │       ▼                                                       │
 │   evidence += chosen parent ──┐                               │
 └───────────────────────────────┴────────────► next hop ────────┘
                 │
                 ▼
   {evidence[hop, atom_question], trace[], hops_used, stopped_reason}
   stopped_reason ∈ done · max_hops · no_candidates · no_selection
                    · timeout · llm_error
```

### Lifecycle loop

```
 row returned by a search ──► hit_count++ , last_hit_at
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
                                          (and disables recency decay)
```

`GET /admin/duplicates` lists near-duplicate pairs by cosine, `GET /admin/notes` lists old
agent notes, and `POST /admin/restore` clears `archived_at`. Every mutating admin route
previews by default and acts only with `{"confirm": true}`.

## Storage

`memory.memory_chunks` — one table for every non-code source.

| column | meaning |
|---|---|
| `id` | `note:<hash>` · `doc:<document_id>:<ordinal>` · `…:atom:<n>` |
| `source_type` | `agent_note` · `document` |
| `source_ref` | `save_memory` or the document id |
| `chunk_kind` | `note` · `decision` · `doc` · `atom` |
| `content_raw` / `distilled` | stored text; searches read `distilled` first |
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
| `POST` | `/search` | hybrid search — `query`, `source` (`all`\|`code`\|`memory`), `top_k`, `kind`, `tags`, `include_atoms`, `include_archived` |
| `POST` | `/search/deep` | multi-hop memory search — `query`, `max_hops`, `kind`, `tags` |
| `POST` | `/save_memory` | store a distilled note — `content`, `kind`, `tags`, and the optional id of a prior note to archive |
| `POST` | `/ingest/document` | multipart upload — `file`, `document_id`, `mode` (`upsert`\|`force`), `origin` |
| `GET` | `/ingest/jobs/{job_id}` | document job state |
| `POST` | `/repos` | add or re-sync a git repo — `url`, `branch`, `name` |
| `GET` | `/repos` | cached repos with url, branch, head, chunk count |
| `DELETE` | `/repos/{name}` | remove a repo and re-index |
| `GET` | `/repos/jobs/{job_id}` | repo job state |
| `GET` | `/admin/notes` | active agent notes older than `older_than_days` |
| `POST` | `/admin/notes/delete` | preview, or delete with `confirm` |
| `GET` | `/admin/duplicates` | near-duplicate pairs above `threshold` |
| `POST` | `/admin/archive` | preview cold rows, or archive with `confirm` |
| `POST` | `/admin/restore` | preview, or restore with `confirm` |

`kind` and `tags` filters require `source="memory"`.

## MCP tools

`mcp_server.py` is a thin proxy over the REST API — no logic of its own. Transport is
stdio by default; `MCP_TRANSPORT=sse|streamable-http` with `MCP_HOST`/`MCP_PORT` serves
over HTTP (Docker serves SSE on `:8765`).

`search` · `search_code` · `search_memory` · `deep_search` · `save_memory` ·
`ingest_document` (text formats only) · `ingest_repo` · `remove_repo` · `list_repos`

## Running

```bash
uv sync
docker compose up -d db redis            # pgvector on :5439, job state on :6379
docker compose up -d --build api         # REST backend on :8010
docker compose up -d --build mcp         # MCP server, SSE on :8765
claude mcp add --transport sse memory-base http://localhost:8765/sse
```

Index cached repos manually (the repo routes do this for you). The indexer runs inside the
API container, which owns the only copy of the ledger and the repo cache:

```bash
docker compose exec api uv run cocoindex update src/memory_base/ingest/code.py     # incremental
docker compose exec api uv run cocoindex update -L src/memory_base/ingest/code.py  # live watch
uv run python -m memory_base.retrieval.search "your query" --source code
```

The memory schema is created on first write (`ensure_schema`); the code table is created
by the indexer.

## Configuration

`.env` (gitignored) holds endpoints and credentials — never hardcode them.

| variable | purpose |
|---|---|
| `LLM_URL`, `EMB_URL`, `RERANK_URL` | vLLM OpenAI-compatible endpoints |
| `LLM_MODEL`, `EMB_MODEL`, `RERANK_MODEL` | model names |
| `DB_URL` | Postgres connection string |
| `REPO_CACHE` | git checkout root |
| `REDIS_URL` | job-state mirror |
| `REST_URL` | backend the MCP server proxies to |
| `LOG_DIR` | file-sink directory (default `logs/`) |

Tuning knobs, all optional: `ATOM_RETRIEVE_K`, `ATOMS_RETRIEVE`, `ATOMS_GENERATE`,
`NOTE_SIMILAR_THRESHOLD`, `DEEP_MAX_HOPS`, `DEEP_TIMEOUT_SECONDS`, `INGEST_MAX_BYTES`,
`INGEST_MAX_QUEUED`, `INGEST_MAX_CONCURRENT_JOBS`, `REPO_MAX_QUEUED`, `REPO_MAX_BYTES`,
`REPO_DISK_HEADROOM_BYTES`, `COLD_AGE_DAYS`, `COLD_UNHIT_DAYS`.

`COCOINDEX_DB` is set by `docker-compose.yml`, not by `.env` — together with `REPO_CACHE`
and `REDIS_URL` it points at a container-local path bound under `DATA_ROOT` on the host, so
one ledger tracks one repo cache. Losing the repo cache or the ledger orphans `code_chunks`
rows until the repos are re-added; losing the Redis directory drops job history only.

## Development

```bash
uv run pytest                                        # integration tests skip without DB/vLLM
uv run pytest -m "not integration"                   # what CI runs
uv run ruff format --check . && uv run ruff check .
uv run python -m memory_base.eval.retrieval          # retrieval eval with atom-lane A/B
```

Work happens in a git worktree and lands via PR; `main` requires a PR and green CI (lint,
unit tests, test-guard, PR title). See `AGENTS.md` for the full contributor contract.
