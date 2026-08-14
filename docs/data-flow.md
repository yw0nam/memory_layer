# Data flow

How content enters the store, how it comes back out, and where it lands.

## Write paths

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
│ note │ decision │ doc                │      │ repo · file · L1-L40   │
│ halfvec(2048) + HNSW + BM25          │      │ halfvec + HNSW + BM25  │
└──────────────────────────────────────┘      └────────────────────────┘
        ▲                                                ▲
        └────── the write→read contract, with one more ──┘
               memory.doc_rows — tabular rows, SQL-only
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
the optional repeated `tags` upload field lands on every chunk of the document.

CSV takes a tabular branch instead. Every data row is validated (unique non-empty
headers, consistent row width, no NUL bytes, ≤5 MB and ≤100,000 rows) and stored
verbatim in `memory.doc_rows` — cells as strings, empty cells as JSON null — in the same
transaction that publishes the document's one LLM-summarized knowledge Card. The Card is
the only embedded artifact: its metadata carries the `columns` list and marks the rows
as loaded, and search hits expose those columns next to the `document_id` handle, so a
consumer can go from a search hit straight to a SQL query. A same-bytes upsert of a CSV
whose rows are not yet loaded re-runs the pipeline instead of no-opping.

Every document chunk records its creator: ingestion stamps the authenticated key's label
into metadata as `created_by`, fixed at first ingest and never rewritten by later
uploads. Overwriting an existing `document_id` (any mode) and deleting a document are
allowed only for that creator or an admin key; a document with no `created_by` record is
admin-only, fail-closed. `DELETE /ingest/documents/{document_id}?namespace=…` removes
the document's chunks and its `doc_rows` together.

Repo jobs use the same durable jobs table and dispatch one at a time, so interrupted clone,
pull, remove, and index work is retried after an API restart.

The code indexer mounts every subdirectory of `REPO_CACHE` as an independent codebase, so
adding or removing a checkout adds or tears down its rows on the next run. Clones keep
full history (`--filter=blob:none`, lazy blobs) precisely so each file carries its real
last-commit time into time-decay scoring.

`POST /repos` accepts http(s) URLs only, and rejects credentials embedded in the URL.
Private repositories authenticate through the git credential store — see
[Private repositories](configuration.md#private-repositories).

A repo's owner is the key label that first ingested it, recorded in
`REPO_CACHE/.owners/<name>` after the first successful clone or pull and never
transferred by later re-ingests. `DELETE /repos/{name}` is restricted to the owner or an
admin key; a repo with no owner record (ingested before ownership tracking, or never
successfully ingested) is admin-only to remove. `GET /repos` reports each repo's owner,
`null` when unrecorded.

## Read path — hybrid search (`POST /search`)

```
      "query"
         │
         ├─ embed (Qwen3 instruction prefix) ──┐
         │                                     │
 ┌───────┴────────────┐              ┌─────────┴──────────┐
 │    code_chunks     │              │   memory_chunks    │  archived excluded
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
                🎯 rerank (vLLM) → top 10
                        ▼
                code hits get ±40-line neighbour chunks as `context`
                        ▼
                    hits[]  ─────► in-process buffer
                                   (flushed on an interval)
```

Response text is truncated to 2000 chars. `score` is the rerank score, falling back to the
fused RRF score. Hits below the min_score floor (default 0.25, request-adjustable, 0 disables)
are dropped after reranking. `include_archived` surfaces archived rows and turns recency decay off for
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

## Read path — table SQL (`POST /tables/query`)

Tabular questions are answered by computing over stored rows, not by retrieval. The
working loop:

```
search_memory("developer productivity sleep")
  └► hit: ref = "developer-productivity-metrics#card-0"
         columns = ["developer_id", "ai_usage", "sleep_hours", "commits"]
              │     source_ref is the document_id, columns are the JSON keys
              ▼
POST /tables/query  {"sql": "...", "namespace": "default"}
  SELECT data->>'ai_usage'                        AS grp,
         AVG((data->>'sleep_hours')::numeric)     AS mean_sleep
  FROM memory.doc_rows
  WHERE document_id = 'developer-productivity-metrics'
  GROUP BY 1
              │
              ▼
  {"columns": ["grp", "mean_sleep"], "rows": [["high", 6.5], …],
   "row_count": 3, "truncated": false}
```

Rows are jsonb: cast for numbers (`(data->>'col')::numeric`), filter by `document_id` to
scope one table, or aggregate across every table in the namespace by leaving the filter
off. Results cap at 1,000 rows (`truncated: true` beyond that — page with `row_index`
ranges), responses at 5 MB, statements at 10 s (`408`). Decimal, date, and UUID values
arrive JSON-normalized.

The SQL author is an LLM, so the lane is fenced at the database, not by string
inspection. Queries must start with `SELECT`/`WITH` and run on a dedicated pool
authenticated as `memory_tables_query` — a role with `SELECT` on `memory.doc_rows` and
nothing else — inside a read-only transaction, single-statement by protocol, with forced
row-level security pinning each request to its validated namespace. Side-effect
functions (`set_config`, `pg_notify`, advisory locks) are revoked, and role-level
resource limits bound memory and lock waits. Other Postgres errors return `400` with the
engine's message so the caller can correct its SQL.

## Lifecycle loop

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

## Storage

`memory.memory_chunks` — one table for every non-code source.

| column | meaning |
|---|---|
| `id` | `note:<hash>` · `doc:<document_id>:<ordinal>` |
| `source_type` | `agent_note` · `document` |
| `source_ref` | `save_memory` or the document id |
| `chunk_kind` | `note` · `decision` · `doc` |
| `content_raw` / `distilled` | stored text; BM25 index on `content_raw`, hits display `distilled` first |
| `embedding` | `halfvec(2048)`, HNSW cosine index |
| `ts_last_active`, `idf_score` | ranking signals |
| `metadata` | jsonb: `tags`, `heading_path`, `content_hash`, `search_ref`, `created_by`, `columns`, … |
| `hit_count`, `last_hit_at`, `archived_at` | lifecycle counters |

`memory.code_chunks` — written and torn down entirely by CocoIndex: `repo`, `filename`,
`code`, `embedding`, `start_line`, `end_line`, `mtime` (last commit time).

`memory.doc_rows` — a tabular document's data rows: `namespace`, `document_id`,
`row_index`, `data` (jsonb, one object per row keyed by the CSV header). No embedding,
no search indexes; read exclusively by `POST /tables/query` under the restricted role,
written and deleted in the same transactions as the document's Card.

These three tables are the only contract between the write side and the read side
([ADR-0001](adr/0001-table-rows-third-read-contract.md)). Adding a source means
adding an adapter, not touching retrieval or serving.
