# memory-base

A selective memory layer for coding agents: distilled notes, chunked documents, indexed
code, and SQL-queryable tables in one pgvector store, served through a single REST API
and an MCP server.

Only high-signal content is embedded. Agent notes arrive already distilled, documents
pass through deterministic chunking and a junk gate before embedding, and code is
chunked by tree-sitter. Raw transcripts and raw files are never embedded. Tabular
documents additionally keep their data rows as structured, never-embedded rows behind a
read-only SQL interface, so questions about the numbers are computed rather than
retrieved.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  CONSUMERS      coding agents · n8n · scripts                                 │
└───────┬──────────────────────────────────────────────────────────────────────┘
        │ MCP  (stdio | SSE :8765)
        ▼
   ┌─────────────┐  10 tools: search / search_code / search_memory / save_memory
   │ mcp_server  │            ingest_document / remove_document / query_table
   └──────┬──────┘            ingest_repo / remove_repo / list_repos
          │ HTTP                                       (thin proxy, no logic)
          ▼
   ╔═══════════════════════════════════════════════════════════╗
   ║              REST API  :8010   (the only backend)         ║
   ╚═╤═══════════════╤═══════════════╤═══════════════╤═════════╝
     │ WRITE         │ WRITE         │ WRITE         │ READ + LIFECYCLE
     ▼               ▼               ▼               ▼
  notes.py     ingest_api.py     repos.py      search.py · tables.py · admin.py
     │               │               │               │
     └───────────────┴───────┬───────┴───────────────┘
                             ▼
              Postgres 17 + pgvector + pg_textsearch  :5439
              memory_chunks · code_chunks · doc_rows · jobs · retrieval_log

  side services:  vLLM (LLM / embedding / rerank)
```

Every consumer reaches stored chunks through the REST API, never through the database
directly. Three tables are the entire contract between the write side and the read
side: `memory_chunks` and `code_chunks` feed search, and `doc_rows` holds a tabular
document's data rows for the SQL read path alone — never embedded, never returned by
search ([ADR-0001](docs/adr/0001-table-rows-third-read-contract.md)).

How notes, documents, CSV rows, and code move through the system — write pipelines,
the hybrid-search and table-SQL read paths, the archive lifecycle, and the storage
schema — is documented in [docs/data-flow.md](docs/data-flow.md).

## Retrieval quality

Hybrid search beats vector-only on both evaluated corpora (ZX Bank hit@10 0.98 vs 0.95;
SciFact hit@5 0.86 with rerank vs 0.82). Scores, method, single-leg ablations, and known
trade-offs: [docs/benchmarks/retrieval.md](docs/benchmarks/retrieval.md).

## REST API

| method | path | purpose |
|---|---|---|
| `GET` | `/health` | liveness — `200 {status}` whenever the process serves HTTP; reaches nothing outside it, and backs the container healthcheck |
| `GET` | `/health/services` | dependency health — `{status, checks:{db, embedding, rerank, llm}}`; `503` when db, embedding, or rerank is down |
| `POST` | `/search` | hybrid search — `query`, `source` (`all`\|`code`\|`memory`), `top_k`, `kind`, `tags`, `repo`, `include_archived` |
| `POST` | `/save_memory` | store a distilled note — `content`, `kind`, `tags`, and the optional id of a prior note to archive |
| `POST` | `/ingest/document` | multipart upload — `file`, `document_id`, `mode` (`upsert`\|`force`), `origin`, repeated `tags`; overwriting an existing `document_id` is creator-or-admin only |
| `DELETE` | `/ingest/documents/{document_id}` | remove a document's chunks and table rows in one namespace (`namespace` query param, default the key's home) — the document's creator or an admin key only |
| `POST` | `/tables/query` | read-only SQL over `memory.doc_rows` — `sql` (`SELECT`/`WITH`), `namespace`; 1,000-row / 5 MB / 10 s caps |
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
`ingest_document` (text formats and CSV) · `remove_document` · `query_table` ·
`ingest_repo` · `remove_repo` · `list_repos`

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

`.env` (gitignored) holds every endpoint and credential. Required to boot:
`LLM_URL`/`EMB_URL`/`RERANK_URL` and their model names, `DB_URL`, `POSTGRES_PASSWORD`,
`TABLES_QUERY_PASSWORD`, and `DATA_ROOT`. The full variable reference, optional tuning
knobs, and private-repository credentials are documented in
[docs/configuration.md](docs/configuration.md).

## Development

```bash
uv run pytest                                        # integration tests skip without DB/vLLM
uv run pytest -m "not integration"                   # what CI runs
uv run ruff format --check . && uv run ruff check .
uv run python -m memory_base.eval.retrieval          # retrieval eval report
```

Work happens in a git worktree and lands via PR; `main` requires a PR and green CI (lint,
unit tests, test-guard, PR title). See `AGENTS.md` for the full contributor contract.
