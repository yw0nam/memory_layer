# Data flow

How content enters the store, how it comes back out, and where it lands.

## Write paths

```
① NOTE                  ② DOCUMENT                    ③ CODE
POST /save_memory       POST /ingest/document         POST /repos {url}
   │                       │ 202 {job_id}                │ 202 {job_id}
   ▼                       ▼                             ▼
 validate ≤4000         MarkItDown worker            url validated
 kind ∈ note|decision|episode   (killable, 120 s)            free disk? → 507
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
│ note │ decision │ episode │ doc      │      │ repo · file · L1-L40   │
│ halfvec(2048) + HNSW + BM25          │      │ halfvec + HNSW + BM25  │
└──────────────────────────────────────┘      └────────────────────────┘
        ▲                                                ▲
        └────── the write→read contract, with one more ──┘
               memory.doc_rows — tabular rows, SQL-only
               memory.messages — addressed signals, claim-only

④ MESSAGE / HANDOFF
POST /messages {subject, status, result, …}
   │
   ▼
 validate purpose (scope ⇒ handoff), status, next,
 verification {command, status, result}, refs ≤ 10 https,
 expires_at (future, ≤ 30 days; default MESSAGE_TTL_DAYS)
   │
   ▼
 render canonical Markdown (blockquote every user line,
 headings cannot escape; reject past 4 KiB — no truncation)
   │
   ▼
 INSERT memory.messages (no embedding, no content gate)
 handoff: same transaction terminalizes older pending
 snapshots of the same namespace+scope+subject_key
```

Notes are stored exactly as written — the server never summarizes. Before embedding,
every non-episode note passes the content gate: the chat model judges the text on one
criterion — does it record something a future session could not recover from the systems
of record and would need, or is it already answered there or bound to the moment it was
written — and a note that fails is refused with HTTP 409 and the reason;
`allow_restatement` overrides the refusal and stamps `metadata.content_gate =
"overridden"`, while a judge failure saves the note stamped `content_gate =
"unavailable"`. A note landing next to
active notes above `NOTE_SIMILAR_THRESHOLD` cosine is refused with HTTP 409 listing them,
unless `supersedes` names one of them or `allow_similar` is set; an accepted override
records the neighbours' ids in `metadata.similar_ack`. The response carries `similar[]`
either way. A prior-note id in the payload archives that row.

Every note records the agent that wrote it in `metadata.author`, drawn from the calling
key's allowlist in `api_keys.authors`; a key with an empty allowlist cannot save. A note
archived by a save or by a targeted archive additionally carries `metadata.archived_by`,
which a restore removes.

Messages are stored exactly as the renderer produced them — the server never
summarizes, embeds, or gates them. A send without a scope is a general message
(`status: "info"`) for one namespace; a send with a portable scope is a handoff
snapshot (`status: "in_progress"`, `"blocked"`, or `"completed"`), and any sender who
can access the namespace may publish the next snapshot of a subject. A `repo:` scope
names a remote host, so local paths, localhost, and private/loopback/link-local IP
literals are refused as not portable. Verification is
exactly `{command, status, result}` with `passed|failed|not_run`; refs are at most 10
absolute https URLs, with credentials, localhost, private/loopback/link-local IP
literals, and local paths rejected without fetching. An optional `idempotency_key`
(unique per sender key) replays an identical effective request with 200 and refuses a
different one with 409; deleting the row releases the key.

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
                min_score floor (default 0.25, rerank scale only)
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
`hit_count` update, so repeated hits on a popular row collapse into a single `+ n`. Each
`retrieval_log` row carries the request's narrowing options — `kind`, `tags`, `repo`, `since`,
`until`, `min_score`, `author`, `namespaces`, `include_archived`, `top_k` — in a `filters`
jsonb column, so an empty result is attributable to the filters that produced it. The same
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

## Read path — messages (`GET /messages`, claim, cancel)

Messages are read by address, never by similarity. `GET /messages` lists pending,
unexpired messages newest-first (created_at DESC, id DESC) with `namespace`, `purpose`,
`scope`, and `subject` filters — the subject filter normalizes its argument the same
way the send path does, so any spelling variant of a subject finds its snapshot chain —
and a `limit` (default 50, max 100). No query, no embedding call, no access to
`memory_chunks`.

Delivery is at-most-once. `POST /messages/{id}/claim` is a single conditional UPDATE
on the lifecycle timestamps (`claimed_at IS NULL AND cancelled_at IS NULL AND
superseded_at IS NULL AND expires_at > clock_timestamp()`), so two concurrent claims are decided by
database commit order: exactly one returns the row, the other gets 409. There is no
lease, ack, or re-read. `DELETE /messages/{id}` cancels a pending message — the
sender's own, or any accessible one for an admin key. Terminal rows are invisible to
the list and unclaimable; a stale superseded snapshot id gets 409, an unknown or
out-of-scope id a 404.

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

`GET /admin/duplicates` lists near-duplicate pairs by cosine with each side's author,
`GET /admin/notes` lists old agent notes, and `POST /admin/restore` clears `archived_at`
and `metadata.archived_by`. `POST /admin/archive` archives the rows named by `ids`, or
the cold ones when `ids` is omitted; the no-ids preview distinguishes
`notes_to_archive` from `messages_to_delete`, and the confirm pass archives the notes
and deletes claimed, cancelled, superseded, and expired messages, which also releases
their idempotency keys. Archiving a note is reversible and deleting a message is not,
so a member key purges the message half only in the namespaces it owns — enough to
drain one before unregistering it, not enough to touch a shared namespace. An admin
key purges everywhere. Every mutating admin route previews by default and
acts only with `{"confirm": true}`, and each is reachable over MCP as
`list_memory_duplicates`, `archive_notes`, `restore_notes`, and `delete_notes`.
`archive_notes` always names ids, so the message purge is a REST-only call.

Namespace deletion counts messages as content: a namespace with messages — pending or
terminal — cannot be unregistered until a purge removes them.

Retirement is manual: no scheduler runs in-process, so terminal rows survive until a
caller runs the `/admin/archive` preview and confirm pass. A deployment that wants it
periodic drives that pair from outside, e.g. a cron job or an n8n schedule.

## Storage

`memory.memory_chunks` — one table for every non-code source.

| column | meaning |
|---|---|
| `id` | `note:<hash>` · `doc:<document_id>:<ordinal>` |
| `source_type` | `agent_note` · `document` |
| `source_ref` | `save_memory` or the document id |
| `chunk_kind` | `note` · `decision` · `episode` · `doc` |
| `content_raw` / `distilled` | stored text; BM25 index on `content_raw`, hits display `distilled` first |
| `embedding` | `halfvec(2048)`, HNSW cosine index |
| `ts_last_active`, `idf_score` | ranking signals |
| `metadata` | jsonb: `tags`, `author`, `archived_by`, `similar_ack`, `heading_path`, `content_hash`, `search_ref`, `created_by`, `columns`, … |
| `hit_count`, `last_hit_at`, `archived_at` | lifecycle counters |

`memory.code_chunks` — written and torn down entirely by CocoIndex: `repo`, `filename`,
`code`, `embedding`, `start_line`, `end_line`, `mtime` (last commit time).

`memory.doc_rows` — a tabular document's data rows: `namespace`, `document_id`,
`row_index`, `data` (jsonb, one object per row keyed by the CSV header). No embedding,
no search indexes; read exclusively by `POST /tables/query` under the restricted role,
written and deleted in the same transactions as the document's Card.

`memory.messages` — addressed, once-claimed signals: `namespace`, `purpose`
(`message` | `handoff`), `scope`, `subject` with its normalized `subject_key`,
`status` (the report state: `info` | `in_progress` | `blocked` | `completed`),
`content` (canonical Markdown), `author`, `sender_key`, `idempotency_key`, and the
timestamptz lifecycle fields `created_at`, `claimed_at`, `cancelled_at`,
`superseded_at`, `expires_at`. No embedding column, no search index; a partial index
serves the pending listing, and a unique partial index on
(`sender_key`, `idempotency_key`) backs idempotent sends. The lifecycle timestamps stay
internal — responses carry the report status only.

`doc_rows` and `messages` are both outside the retrieval contract: they are read by
compute and by address respectively, and are never granted to the SQL query role or
returned by search ([ADR-0001](adr/0001-table-rows-third-read-contract.md),
[ADR-0002](adr/0002-messages-addressed-once-claimed-lane.md)). Adding a source means
adding an adapter, not touching retrieval or serving.
