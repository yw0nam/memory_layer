# Data flow

How content enters the store, how it comes back out, and where it lands.

## Write paths

```
① NOTE                  ② DOCUMENT                    ③ CODE
POST /save_memory       POST /ingest/document         POST /repos {url}
   │                       │ credential scan of          │
   │                       │ upload fields → 400         │
   │                       │ 202 {job_id}                │ 202 {job_id}
   ▼                       ▼                             ▼
 validate ≤4000         MarkItDown worker            url validated
 kind ∈ personal|work           (killable, 120 s)            free disk? → 507
   │                       │                             │
   ▼                       ▼                             │
 credential scan        credential scan of the           │
 content + raw tags     text or every CSV cell           │
 → 409                  → job failed, nothing            │
   │                      stored; then same              │
   │                      bytes → no_op                  │
   │                       │                             │
   ▼                       ▼                             ▼
 id = sha256(content)   chunk 1500 / 2000 / 200      git clone --filter=blob:none
 └ same → no-op         junk gate ✂                  + size watchdog (2 GiB)
   │                       │                             │
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
│ personal │ work │ doc                │      │ repo · file · L1-L40   │
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
 expires_at (future, ≤ 30 days; default MESSAGE_TTL_DAYS
 for a message, none for a handoff)
   │
   ▼
 render canonical Markdown (blockquote every user line,
 headings cannot escape; reject past 16 KiB — no truncation)
   │
   ▼
 INSERT memory.messages (no embedding)
 handoff: same transaction terminalizes older pending
 snapshots of the same namespace+scope+subject_key
```

Every note and document write starts with a deterministic credential scan
(`core/secrets.py`): the fixed-format detectors of `detect-secrets` — the npm, JWT, and
legacy OpenAI patterns anchored at token boundaries so a scan stays linear in the text
length — plus local Anthropic, OpenAI project/service key, and Google API key patterns,
with no entropy or keyword detection and no model call. It recognizes provider API keys
(AWS access key ids, GitHub, GitLab, Slack, Stripe,
SendGrid, npm, PyPI, OpenAI, Anthropic, Google), private-key headers, JSON Web Tokens, and
credentials embedded in URLs (`scheme://user:pass@host`). A hit refuses the write whole —
nothing is redacted — and the reason names only the detector type, never the matched
value. A note is scanned together with its raw tags before embedding and refused
with HTTP 409. A document upload's filename, `document_id`, `origin`, and tags are scanned
before the job is admitted and refused with HTTP 400. The worker scans the extracted text
— for a CSV, every header name and cell — before the same-bytes `no_op` check, the Card
summary, embedding, or any write, so the job fails, nothing from the document is stored,
and an identical re-upload is refused the same way. Prose that quotes a literal
`BEGIN … PRIVATE KEY` header is refused as a private key.

A note is stored exactly as written — the server never rewrites one and writes no notes
itself, and no chat model judges its content. What is worth keeping is the writing
agent's call, guided by the `save_memory` description and the server instructions; the
server applies only deterministic checks: validation, the credential scan, the
near-duplicate refusal, and supersede. A note landing next to active notes above
`NOTE_SIMILAR_THRESHOLD` cosine is refused with HTTP 409 listing them, unless `supersedes`
names one of them or `allow_similar` is set; an accepted override records the neighbours'
ids in `metadata.similar_ack`. The response carries `similar[]` either way. A prior-note
id in the payload archives that row; the replacement records the archived note's id as
`metadata.supersedes`, which `GET /notes` rows and memory hits carry as `supersedes`, and
the archived note records the replacement's id as `metadata.replaced_by`.
When the save's content is identical to an active note, the insert no-ops and the target
is archived without a pointer written on the existing row; the target's `replaced_by`
names that row. The save is refused with HTTP
400 when it would leave no active note — the content is identical to the note it names,
or to an archived note, which `restore_notes` brings back instead.

A note's id is `note:<namespace>:<sha256(content)[:16]>`, the hash taken over its
stripped content, so a re-save is an idempotent no-op. `session_id` is always the note's
own id. `occurred_at` (ISO 8601, not in the future) records when the remembered event
happened in its own column; `ts_last_active` is always the save time.

Every note records the agent that wrote it in `metadata.author`, drawn from the calling
key's allowlist in `api_keys.authors`; a key with an empty allowlist cannot save. A note
archived by a save or by a targeted archive additionally carries `metadata.archived_by`,
which a restore removes together with `replaced_by` and `consolidated_into`.

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
last-commit time into the recency signal.

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
 │ vec50 · fts50 · rec│              │ vec50 · fts50 · rec│  optional kind / tags
 │  optional repo     │              │                    │
 └───────┬────────────┘              └─────────┬──────────┘
         └──────────────┬───────────────────────┘
                        ▼
                RRF   Σ w/(60 + rank)   w: vec 1.0 · fts 0.2 · rec 0.25
                        ▼
                per-file / per-session cap 3  → top 40
                        ▼
                🎯 rerank (vLLM)
                        ▼
                top 10 · min_score floor (default 0.25, rerank scale only)
                  │ skipped when budget_tokens is set
                        ▼
                code hits get ±40-line neighbour chunks as `context`
                        ▼
                budget_tokens set: hits in rerank order until the estimate
                (chars / 4 of text + context) would exceed the budget
                        ▼
                    hits[]  ─────► in-process buffer
                                   (flushed on an interval)
```

Memory hits (notes, document chunks, CSV cards) carry the row's `id`, `kind`, and
`tags`, plus `supersedes` for a note that replaced one; `date` is
the note's `occurred_at` when recorded, else `ts_last_active`. `since`/`until` bound the
same event time, `COALESCE(occurred_at, ts_last_active)`, so an episode is found by the
day it happened; recency voting and decay read `ts_last_active` alone. Memory hits carry their stored text whole; each is
bounded at write time instead — notes ≤4000 chars, document chunks ≤2000 (hard split), CSV
cards ≤2000 (the ingest job fails if the summary runs longer). Code hits have no such
bound — CocoIndex's chunk_size is a target, not a limit — so the response still cuts them
to 2000 chars. `score` is the rerank score, falling back to the
fused RRF score. Hits below the min_score floor (default 0.25, request-adjustable, 0 disables)
are dropped after reranking. Age enters ranking only as the recency voter in the fusion, so an
old row with a strong vector or BM25 rank reaches the reranker. `budget_tokens` (1–32000)
replaces the count and the floor: every fused candidate is reranked, and hits come back in
rerank order until the running estimate of the returned text (`max(1, chars // 4)` per hit,
restored code context included) would exceed the budget, so a multi-fact question gets every
relevant note that fits rather than ten. `include_archived` surfaces
archived rows; `/search` hits mark archived rows `"archived": true` since an archived note may have been superseded by
a newer one.

The read path writes nothing to the database. Returned hit ids and per-chunk counters land
in an in-process buffer that a background task flushes every `HIT_FLUSH_INTERVAL_SECONDS`
(default 30) and on shutdown: one batched `retrieval_log` insert plus one deduplicated
`hit_count` update, so repeated hits on a popular row collapse into a single `+ n`. Each
`retrieval_log` row carries the request's narrowing options — `kind`, `tags`, `repo`, `since`,
`until`, `min_score`, `budget_tokens`, `author`, `namespaces`, `include_archived`, `top_k` — in a `filters`
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
              │     the part of ref before "#" is the document_id, columns
              │     are the JSON keys
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

A general message sent without `expires_at` expires after `MESSAGE_TTL_DAYS`; a handoff
sent without one has a null `expires_at` and never expires — it leaves the pending set
only by claim, supersede, or cancel. A caller-supplied `expires_at` must be in the future
and at most 30 days out for either purpose.

Messages are read by address, never by similarity. `GET /messages` lists pending,
unexpired messages newest-first (created_at DESC, id DESC) with `namespace`, `purpose`,
`scope`, and `subject` filters — the subject filter normalizes its argument the same
way the send path does, so any spelling variant of a subject finds its snapshot chain —
and a `limit` (default 50, max 100). No query, no embedding call, no access to
`memory_chunks`.

Delivery is at-most-once. `POST /messages/{id}/claim` is a single conditional UPDATE
on the lifecycle timestamps (`claimed_at IS NULL AND cancelled_at IS NULL AND
superseded_at IS NULL AND (expires_at IS NULL OR expires_at > clock_timestamp())`), so
two concurrent claims are decided by database commit order: exactly one returns the row,
the other gets 409. There is no
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
```

`GET /admin/duplicates` lists near-duplicate agent-note pairs by cosine with each side's author,
`GET /admin/notes` lists old agent notes, and `POST /admin/restore` clears `archived_at`
and `metadata.archived_by`, `replaced_by`, and `consolidated_into`. `POST /admin/archive` archives the rows named by `ids`, or
the cold agent notes when `ids` is omitted; the no-ids preview distinguishes
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
terminal — cannot be unregistered until they are removed. `POST /admin/notes/move`
rewrites a note's id to `note:<target>:<hash>`.

Retirement is manual: no scheduler runs in-process, so terminal rows survive until a
caller runs the `/admin/archive` preview and confirm pass. A deployment that wants it
periodic drives that pair from outside, e.g. a cron job or an n8n schedule.

## Consolidation groups

`GET /admin/consolidate/groups` finds groups of active agent notes that may state the
same fact or rule, for an agent to judge. It changes no note and calls no model; only an
admin key whose authors include `consolidator` may call it (403 otherwise). Every query
parameter is optional and a bad, repeated, or out-of-range value is a 400:

| parameter | range | default |
|---|---|---|
| `namespace` | repeatable; each a registered namespace | every registered namespace |
| `threshold` | 0 < x ≤ 1 | 0.72 |
| `neighbors` | 1–50 | 5 |
| `max_group` | 2–20 | 6 |
| `max_group_chars` | ≥ 500 | 12000 |
| `limit` | 1–1000, groups returned per namespace | 200 |

```
 per namespace, one read-only REPEATABLE READ snapshot
   active agent notes ──► exact nearest neighbours by stored embedding
                          (index scans off, `neighbors` per note, no embedding call)
                                   │ pairs with cosine ≥ threshold
                                   ▼
         pair listed in either note's `similar_ack` ──► ignored, counted `acknowledged`
                                   │ unacknowledged pairs
                                   ▼
                     greedy clique packing, highest score first
     a note joins a group only above the threshold with every member;
     capped by max_group and max_group_chars; ties broken by id
                                   │
                   ┌───────────────┴───────────────┐
                   ▼                               ▼
              group (first `limit`)          note with an edge
              → `groups`                     left out of every group
                                             → `deferred` with a reason
```

An acknowledged pair is the writer's assertion that the two notes state distinct facts,
so it is never an edge; each note's other pairs still group. A group already judged is
not returned and is counted in `cached`: its key has a recorded verdict, or its members
are exactly those of an undone action — the second match ignores the procedure version,
so an undone group stays out after `PROCEDURE_VERSION` changes. Judged groups are removed
before `limit` applies. A pair whose combined text
exceeds `max_group_chars` is not grouped. Each note left out appears once in `deferred`
with the first reason recorded for it while packing — `over max_group_chars` or
`over max_group` — else `no clique` (its partners joined other groups). Groups are
ordered by their highest edge score, then smallest member id; members by save time, then
id; deferred notes by id.

```
{
  "params": {"namespace": [...], "threshold", "neighbors", "max_group", "max_group_chars", "limit"},
  "procedure_version": "1",
  "namespaces": {
    "<ns>": {
      "active_notes": int, "pairs": int, "acknowledged": int, "cached": int,
      "groups": [{"key", "min_score", "max_score",
                  "members": [{"id", "kind", "author", "saved": "YYYY-MM-DD",
                               "occurred_at": ISO 8601 or null, "tags",
                               "supersedes": id or null, "text"}]}],
      "deferred": [{"id", "reason"}],
      "truncated": bool
    }
  }
}
```

`pairs` counts every pair at or above the threshold, `acknowledged` the ones among them
that were ignored. `truncated` is true when more groups existed than `limit`. A group's `key` is
the sha256 of canonical JSON (sorted keys, no whitespace) of the procedure version, the
namespace, and, per member sorted by id, its id, the sha256 of its text, kind, author,
save time, `occurred_at`, sorted tags, and `supersedes`. It changes when membership or
any of those fields changes. `PROCEDURE_VERSION` covers the grouping procedure and the
verdict validator policy, and changes whenever either does.

## Consolidation verdicts

`POST /admin/consolidate/verdicts` applies an agent's judgement of issued groups. It
takes the same key as the groups route (an admin key with `consolidator` in its authors),
and the body's `author` must be one of the key's authors (403 otherwise). The server calls
no model.

| field | type / range | default |
|---|---|---|
| `run_id` | string, 1–100 chars | required |
| `author` | string, one of the key's authors | required |
| `model` | string or null; recorded, not checked | null |
| `dry_run` | boolean | false |
| `threshold`, `neighbors`, `max_group`, `max_group_chars` | as for the groups route; pass the values the groups were fetched with | as for the groups route |
| `max_actions` | 1–500; retire and merge actions per `run_id` and namespace | 20 |
| `verdicts` | 1–200 verdicts | required |

A verdict is `namespace` (registered), `group_key`, `idempotency_key` (1–200 chars),
`member_ids` (2–20), `action` (`keep` · `retire` · `merge`), `retire_ids` (retire only),
`merged_text` (merge only), and `reason` (1–1000 chars). An unknown field, a wrong type, an
out-of-range value, `retire_ids` or `merged_text` on another action, or an unregistered
namespace refuses the whole request with 400 and processes no verdict.

```
 each verdict in order, on its own
   preflight: read-only REPEATABLE READ snapshot, exact search, no locks
       │ plan ──► duplicate · cached · stale · rejected   (returned as is)
       │ valid plan
       ├── dry_run ──► planned          (no write, no embedding call)
       ▼
   embed the merged text when the plan creates a replacement
       ▼
   one transaction
     pg_advisory_xact_lock(namespace) ──► members, replacement id FOR UPDATE, by id
     plan again on the current rows
       terminal ──► returned as is
       differs from the preflight plan ──► stale
       same ──► replacement · archive members · action row ──► applied
```

The plan runs these checks in order; the first that fails decides the status.

| # | check | status when it fails |
|---|---|---|
| 1 | no action has the `idempotency_key` | `duplicate` with the recorded result when the payload hash matches, else `rejected` |
| 2 | no action has the `group_key`, and no undone action has exactly these members | `cached` |
| 3 | `group_key` and the sorted `member_ids` equal a group that the groups procedure builds now from the request's group parameters (no `limit`, no cache filter) | `stale`, with `current_groups`: the current groups that share a member, in the groups-route shape |
| 4 | retire or merge: fewer than `max_actions` retire and merge actions recorded for this `run_id` and namespace; keep is never capped and never counts | `rejected` (`action cap reached`) |
| 5 | retire: `retire_ids` non-empty, without repeats, all members, and at least one member left | `rejected` |
| 6 | merge: every member has one kind; `merged_text` passes the note validation (non-blank, at most 4000 chars) and the credential scan; the token check passes | `rejected` |

A merge then resolves its replacement:

| merged text | effect |
|---|---|
| equal to a member's text after stripping and collapsing whitespace runs | a retire of the other members into that member; `reason` says so |
| hashes to an active agent note with the members' kind and exactly the stripped text, not a member | that note is reused and not changed |
| hashes to an archived note | `rejected`: restore that note instead |
| hashes to any other existing row | `rejected` |
| hashes to no row | a new replacement note |

The payload hash is the sha256 of the verdict as submitted plus `run_id`, `author`,
`model`, and the four group parameters; `dry_run` and `max_actions` are outside it, so a
dry run and the apply of one verdict share an idempotency key.

An applied verdict writes, in one transaction: a new replacement through the note insert
(`ON CONFLICT (id) DO NOTHING`; a conflict rolls the verdict back as `stale`) with the
members' kind, the sorted union of their tags, the latest member `occurred_at`, the
request's `author`, the apply time as its save time, and `merged_from`, `merged_dates`
(`{id: {"saved", "occurred_at"}}`), and `consolidation_action` in its metadata; each
archived member gets `archived_at` = the apply time, `archived_by`, and
`consolidated_into` (the survivors, or the replacement); and the action row, with the
archived members' metadata from before the change. A reused replacement and the survivors
of a retire are not changed. A keep writes the action row only.

Each verdict fails alone. An unexpected error while it is processed — the embedding call,
or the database during its transaction — rolls that verdict back and returns it as
`failed`, with the error's class and message (at most 300 characters) as the reason; the
next verdict runs. `failed` is transient: the agent may resubmit the verdict with the same
idempotency key.

```
{"results": [{"group_key", "status": "applied" | "planned" | "cached" | "duplicate" | "stale" | "rejected" | "failed",
              "reason", "action_id" or null, "archived_ids", "survivor_ids",
              "replacement_id" or null, "current_groups" or null}]}
```

The token check reads three kinds of token from each text, as exact strings:

- numbers and dates: `2026-09-30`, `04:00`, `1,500`, `-5`, `0.72`, `1.5e3`; digits inside an
  identifier (`abc123`, `note:x9`, `v2.0`) are not tokens, nor is a numbered-list marker
  at the start of a line (`1.` or `2)` followed by whitespace);
- the content of each backticked span;
- names: a word other than `I` that starts with a capital and does not start a sentence
  (text start, a newline, or whitespace after `.`, `!`, `?`, `:`, or a list marker; in
  `foo.Bar`, `Bar` is a name), and any word with two or more capitals (`GLM`, `PR`, `iOS`,
  `McDonald`).

A token counts as dropped when a member has it under the rules above and the merged text
has no such word even at a sentence start, and as added when the merged text has it and no
member has it even at a sentence start. `I` is never a name in either reading. A name that
moves between a sentence start and mid-sentence therefore passes, and a sentence-initial
word (such as `May` or `Then`) can match a same-spelled name on the other side. It is a
conservative filter, not proof of meaning: a changed name at the start of a sentence in
both texts, a negation, names in scripts without case (Korean), and a changed version
inside an identifier (`v2.0` → `v3.1`) pass it.

## Consolidation undo and actions

`POST /admin/consolidate/undo` takes `{"action_id": int, "author": str}` with the same
authorization, and runs under the namespace lock with the touched rows locked.

| case | response |
|---|---|
| unknown action | 404 |
| action already undone | 200 with its recorded undo result |
| keep action | 409, nothing to undo |
| an archived note is gone, has another `archived_at`, or another `consolidated_into` | 409 with the reason; nothing changes |
| a created replacement is gone or archived, an active note is reachable from it through `metadata.supersedes` links (through archived notes in between), or a retire or merge action not undone lists it as a member | 409 with the reason; nothing changes |
| otherwise | 200: each archived note is active again with its metadata from before the action; a created replacement is archived with `archived_by` and `undone_action`; the action records `undone_at`, `undone_by`, and the result |

A reused replacement and a retire's survivors are never checked or changed. An archive and
restore of the replacement with no successor does not block the undo. The action row
stays, so the group is not issued again.

`GET /admin/consolidate/actions` lists actions newest first, with the same authorization.
Each query parameter is optional and given at most once: `namespace`, `run_id`, `note_id`
(matches member, archived, survivor, and replacement ids), and `limit` (1–500, default
50). The response is `{"actions": [...], "notes": {id: ...}}`: every action column except
the payload hash, with ISO 8601 times, and, for every note id the actions name, its full
text, kind, author, save time, `occurred_at`, `archived`, and its lineage fields
(`supersedes`, `replaced_by`, `consolidated_into`, `merged_from`, `merged_dates`,
`consolidation_action`, `archived_by`, `undone_action`). A deleted or moved note is absent
from `notes`, and an undo of an action that archived it fails with 409.

## Profiles

A profile is standing text an agent receives at every session start, outside search:
facts and rules that apply to every task match the topic of almost no message, so
per-message prefetch rarely ranks them. Profiles belong to agents, not to namespaces
([ADR-0009](adr/0009-profiles-owned-by-agents-user-part-approved-by-the-user.md)).

An **owner** is an agent's author slug (`claude-code`, `natsume`; never `user` or
`consolidator`). Each owner has two **parts**, each a versioned text:

| part | holds | written by |
|---|---|---|
| `self` | the agent's persona, working rules, and conventions | the owner, whenever it chooses |
| `user` | the user as this agent needs to know them | the user, by approving the owner's proposal |

Each owner's `user` part is its own document. The server calls no model and judges no
content; profiles are never embedded and never read by search or consolidation.

**Authority** comes only from the key's authors. A key acts for an owner when the owner
is one of its authors; a key carrying the `user` author reads every owner's profile and
proposals and is the only key that approves or rejects. Admin status, the key label, the
home namespace, and namespace permissions grant no profile access. Only the user's key
carries `user`. The agents' key stays an admin key, and an admin key can rewrite any
label's authors through `PUT /keys/{label}/authors`, so the separation prevents mistakes,
not a determined agent on the same host. Isolation between agents on reads is the
client's: the hook and the Hermes provider request only their configured owner.

```
 agent ── PUT /profiles/self ──────────────────────► self v n+1   (or unchanged)
 agent ── POST /profiles/user/proposals {base_version}
            base == current user version? ── no ──► 409 stale {version}
            supersede the pending proposal, insert ──► pending proposal
 user  ── mb_profile.py show <id> ──► proposal + diff against the current user part
 user  ── mb_profile.py approve <id>
            pending? ── no ──► 409 not_pending {status}
            base == current user version? ── no ──► 409 stale {version}, stays pending
            insert user v n+1 (author "user", proposal_id) + mark approved
 GET /profiles?owner= ──► SessionStart hook (Claude Code) · system_prompt_block (Hermes)
```

**Requests.** Every mutation body is a JSON object holding only its listed fields; a
missing or unknown field, malformed JSON, or a wrong type is a 400. `owner` matches
`^[a-z0-9][a-z0-9-]{0,39}$` and is neither `user` nor `consolidator`. `content`,
`reason`, and `note` are stripped; each is scanned for credentials, and a hit is a 400
naming only the credential type. Query parameters are each known and given at most once;
anything else is a 400. A proposal id is a positive 64-bit integer (400 otherwise; 404
when unknown). Lists are JSON arrays; times are ISO 8601 UTC strings.

| route | key | behaviour |
|---|---|---|
| `GET /profiles?owner=` | the owner's, or `user` | one REPEATABLE READ snapshot: `{owner, self_version, self, user_version, user, pending_proposal}` |
| `PUT /profiles/self` | the owner's | `{owner, content, max_chars?}`; `max_chars` 200–20000, default 4000 |
| `POST /profiles/user/proposals` | the owner's | `{owner, content, reason, base_version, max_chars?}`; `max_chars` 200–20000, default 3000; 201 `{id, status: "pending", superseded}` |
| `GET /profiles/user/proposals?owner=&status=&limit=` | with `owner`: the owner's, or `user`; without: `user` only | newest first; `status` one of `pending`, `approved`, `rejected`, `superseded`; `limit` 1–200, default 20 |
| `GET /profiles/user/proposals/{id}` | the proposal owner's, or `user` | the proposal plus `current_user_version` and `current_user_content`, from one snapshot |
| `POST /profiles/user/proposals/{id}/approve` | `user` only | `{note?}`; 200 `{status: "approved", version}` |
| `POST /profiles/user/proposals/{id}/reject` | `user` only | `{note?}`; 200 `{status: "rejected"}` |
| `GET /profiles/versions?owner=&part=&limit=` | the owner's, or `user` | newest first `{version, content, author, proposal_id, created_at}`; `limit` 1–200, default 20 |

**Read.** `self_version` and `user_version` are the latest stored version of each part,
0 when none. A part object `{content, created_at}` is null when no version exists or the
latest content is empty; an older non-empty version is never served, and an emptied part
keeps its version. `pending_proposal` is `{id, created_at, reason, base_version}` or null.

**Self.** The content must fit `max_chars` (otherwise 400 with `chars` and `max_chars`).
Content equal to the latest version returns 200 `{status: "unchanged", version}`; any
other content is stored as the next version, 200 `{status: "written", version}`. Empty
content stores an empty version, which clears the part.

**Proposal.** `reason` holds 1–1000 characters; `base_version` is an integer
0–2147483647 and must equal the owner's current user version (0 when none), otherwise
409 `{"error": "stale", "version": <current>}`. The content has the same length and
credential checks as `self`. A new proposal starts `pending` with `decided_at` and
`decision_note` null; the owner's earlier pending proposal becomes `superseded`, with
`decided_at` set to the new proposal's `created_at` and `decision_note` null. An owner has
at most one pending proposal.

**Decision.** `note`, when given, is a string of 0–1000 characters after stripping and is
stored as the decision note; an omitted note is stored as null. A proposal that is not
pending returns 409 `{"error": "not_pending", "status": <current>}`. Approval then
requires the proposal's `base_version` to equal the current user version, otherwise 409
`{"error": "stale", "version": <current>}` and the proposal stays pending. An approval
stores the proposal's content as the next user version with author `user` and the
proposal's id, and marks the proposal `approved`; a rejection marks it `rejected` without
checking the base. Both set `decided_at` to the decision time. A refused or failed
operation changes no field of any proposal.

**Concurrency.** Every mutation runs in one READ COMMITTED transaction. A decision first
reads the proposal's owner without a row lock (404 when absent); every mutation then
takes `pg_advisory_xact_lock(hashtextextended('profile:' || owner, 0))` and only then
reads the proposal status, its base, and the latest versions in fresh statements. No
proposal row lock is taken before the owner lock. An approval's version insert and its
decision commit or roll back together, as do a supersede and the new proposal.

**Delivery.** The Claude Code SessionStart hook runs at every session start, resume,
clear, and compaction. It calls `GET /profiles?owner=<MEMORY_BASE_AUTHOR>` (default
`claude-code`), lists the repository's pending handoffs, and prints one
`<memory-context>` fence:

```
Memory: standing profile for <owner>. Apply it to every task.
## user (v<user_version>)
<content, or (empty)>
## self (v<self_version>)
<content, or (empty)>
A proposed change to the user profile (proposal <id>) awaits the user's approval. Ask the user to run `! python3 ~/.config/memory-base/mb_profile.py show <id>` to inspect its diff, then approve or reject it with the memory-profile-approval skill.
```

then the handoffs. Both version lines print on every successful fetch, so the session
always has the `base_version` a proposal needs; the notice prints only while a proposal is
pending and never carries its content. A failed or malformed fetch prints no profile
block and no version. The two fetches fail independently. The Hermes provider builds the
same block body, without the fence, for its configured `owner` once in `initialize` and
returns it from `system_prompt_block` for the session; without an `owner` it fetches no
profile. Memory-context tags inside a part are defused to `[memory-context]`. Per-message
prefetch does not include profiles.

**Approval.** The user inspects and decides with `integrations/profile_approval/mb_profile.py`
(`pending`, `show`, `approve`, `reject`), a stdlib CLI that reads only the user's key from
`~/.config/memory-base/user.env` or the environment. The `memory-profile-approval` skill
(`integrations/skills/memory-profile-approval/SKILL.md`) tells an agent to hand that step
to the user and never to run it.

## Storage

`memory.memory_chunks` — one table for every non-code source.

| column | meaning |
|---|---|
| `id` | `note:<namespace>:<hash>` · `doc:<document_id>:<ordinal>` |
| `source_type` | `agent_note` · `document` |
| `source_ref` | `save_memory` for an agent note, or the document id for a document chunk |
| `session_id` | the note's own id, or the document id for a document chunk — the unit the search cap (`PER_FILE_CAP` per `(namespace, session_id)`) is keyed on |
| `chunk_kind` | `personal` · `work` · `doc` |
| `content_raw` / `distilled` | stored text; BM25 index on `content_raw`, hits display `distilled` first |
| `embedding` | `halfvec(2048)`, HNSW cosine index |
| `ts_last_active` | save time; recency ranking and decay, and `since`/`until` when `occurred_at` is null |
| `occurred_at` | when the remembered event happened; a hit's `date` and the `since`/`until` bound |
| `metadata` | jsonb: `tags`, `author`, `archived_by`, `supersedes`, `replaced_by`, `similar_ack`, `consolidated_into`, `merged_from`, `merged_dates`, `consolidation_action`, `undone_action`, `heading_path`, `content_hash`, `search_ref`, `created_by`, `columns`, … |
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
`superseded_at`, `expires_at` (null for a handoff sent without one). No embedding
column, no search index; a partial index serves the pending listing, and a unique partial index on
(`sender_key`, `idempotency_key`) backs idempotent sends. The lifecycle timestamps stay
internal — responses carry the report status only.

`memory.consolidation_actions` — one row per accepted consolidation verdict: `id`,
`idempotency_key` (unique), `payload_hash`, `run_id`, `namespace`, `action`
(`keep` | `retire` | `merge`), `group_key` (unique; the verdict cache), `member_ids`,
`archived_ids`, `survivor_ids`, `replacement_id`, `replacement_created`, `prior` (each
archived note's metadata before the action), `applied_at` (the `archived_at` the action
wrote), `author`, `model`, `reason`, `result` (the response returned on apply), and
`undone_at`, `undone_by`, `undo_result`; indexed on (`namespace`, `run_id`). It is read
only by the consolidation routes.

`memory.agent_profiles` — every version of every owner's parts: `owner`, `part` (`self` |
`user`), `version` (1, 2, … per owner and part; unique together), `content` (empty when
the version clears the part), `author` (the owner for `self`; `user` for an approved user
version), `proposal_id` (set on user versions), and `created_at` (epoch seconds). Every
version is kept.

`memory.profile_proposals` — every proposal to replace an owner's `user` part: `owner`,
`content`, `reason`, `base_version` (the user version it was written against; 0 when
none), `status` (`pending` | `approved` | `rejected` | `superseded`), `created_at`, and the
decision's `decided_at` and `decision_note`. A partial unique index keeps at most one
pending proposal per owner. Every proposal is kept with its decision.

Neither profile table has an embedding column or a namespace; both are read only by the
profile routes, and a namespace is unregistered without regard to them.

`doc_rows` and `messages` are outside the retrieval contract: they are read by compute
and by address, respectively, and are never granted to the SQL query role or returned by
search ([ADR-0001](adr/0001-table-rows-third-read-contract.md),
[ADR-0002](adr/0002-messages-addressed-once-claimed-lane.md)). Adding a source means
adding an adapter, not touching retrieval or serving.
