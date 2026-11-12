# Data flow

How notes and messages enter the store, how it comes back out, and where it lands.

## Write paths

```
① NOTE
POST /save_memory
   │
   ▼
 validate ≤1750
 kind ∈ personal|work
   │
   ▼
 credential scan
 content + raw tags
 → 409
   │
   ▼
 id = sha256(author + content)
 └ same → no-op
   │
   ▼
 embed (vLLM)
   │
   ▼
 INSERT
 ON CONFLICT
 DO NOTHING
   │
   ▼
┌──────────────────────────────────────┐
│ memory.memory_chunks                 │
│ personal │ work                      │
│ halfvec(2048) + HNSW + BM25          │
└──────────────────────────────────────┘
   memory.messages — addressed signals, claim-only

② MESSAGE / HANDOFF
POST /messages {subject, status, result, …}
   │
   ▼
 validate purpose (scope ⇒ handoff), status, next,
 verification {command, status, result}, refs ≤ 10 https,
 expires_at (future, ≤ 30 days; default MESSAGE_TTL_DAYS
 for a message, 3 days for a completed
 handoff, none for another handoff)
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

Every note write starts with a deterministic credential scan
(`core/secrets.py`): the fixed-format detectors of `detect-secrets` — the npm, JWT, and
legacy OpenAI patterns anchored at token boundaries so a scan stays linear in the text
length — plus local Anthropic, OpenAI project/service key, and Google API key patterns,
and a short-code rule, with no entropy detection and no model call. The short-code rule
splits the text into sentences (on `.`, `;`, `!`, `?`, and newlines) and flags a
sentence that carries both a code keyword (`pin`, `password`, `combination`, `otp`, or a
`verification`, `pickup`, `locker`, `two-factor`, or similar `... code`) and a digit code
of 3 to 8 digits; sentences about a `status`, `exit`, `error`, or similar code, dates,
years, quantities with units, and versions are not flagged. It recognizes provider API
keys (AWS access key ids, GitHub, GitLab, Slack, Stripe, SendGrid, npm, PyPI, OpenAI,
Anthropic, Google), private-key headers, JSON Web Tokens, and
credentials embedded in URLs (`scheme://user:pass@host`). A hit refuses the write whole —
nothing is redacted — and the reason names only the detector type, never the matched
value. A note is scanned together with its raw tags before embedding and refused
with HTTP 409. Prose that quotes a literal
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
or to an archived note.

A note's id is `note:<namespace>:<sha256(author + "\n" + content)[:16]>`, the hash taken
over the author and the stripped content, so a re-save is an idempotent no-op and two
authors saving the same text store two notes. `session_id` is always the note's
own id. `occurred_at` (ISO 8601, not in the future) records when the remembered event
happened in its own column; `ts_last_active` is always the save time.

Every note records the agent that wrote it in `metadata.author`, the slug the request's
`X-Author` header declared; the server checks the slug pattern and grants nothing by it.
A note saved with `shared: true` carries `metadata.shared` and is in every author's
default read scope; the flag is set at creation only. A write without a
namespace lands in `default`. A note
archived by a save or by an archive additionally carries `metadata.archived_by`.

Messages are stored exactly as the renderer produced them — the server never
summarizes, embeds, or gates them. A send without a scope is a general message
(`status: "info"`) for one namespace; a send with a portable scope is a handoff
snapshot (`status: "in_progress"`, `"blocked"`, or `"completed"`), and any sender may
publish the next snapshot of a subject. A `repo:` scope
names a remote host, so local paths, localhost, and private/loopback/link-local IP
literals are refused as not portable. Verification is
exactly `{command, status, result}` with `passed|failed|not_run`; refs are at most 10
absolute https URLs, with credentials, localhost, private/loopback/link-local IP
literals, and local paths rejected without fetching. An optional `idempotency_key`
(unique per sender key) replays an identical effective request with 200 and refuses a
different one with 409; deleting the row releases the key.

## Read path — hybrid search (`POST /search`)

```
      "query"
         │
         ├─ embed (Qwen3 instruction prefix) ──┐
         │                                     │
         │                          ┌──────────┴─────────┐
         │                          │   memory_chunks    │  archived excluded
         │                          │ vec50 · fts50 · rec│  optional kind / tags
         │                          └──────────┬─────────┘
         └─────────────────────────────────────┤
                                               ▼
                RRF   Σ w/(60 + rank)   w: vec 1.0 · fts 0.2 · rec 0.25
                        ▼
                top 20 by fused score (top 40 when budget_tokens is set)
                        ▼
                🎯 rerank (vLLM)
                        ▼
                top 10 · min_score floor (default 0.25, rerank scale only)
                  │ skipped when budget_tokens is set
                        ▼
                budget_tokens set: hits in rerank order until the estimate
                (chars / 4 of text) would exceed the budget
                        ▼
                    hits[]
```

A note hit carries the row's `id`, `kind`, and
`tags`, plus `supersedes` for a note that replaced one and `shared: true` on a shared
note. Omitted, `author` returns the caller's default read scope — the calling author's own
notes plus the shared notes; `all` searches every author; a list of slugs returns
exactly those authors' notes, so a caller reads another agent's notes only by naming it.
The MCP tools take the same list, e.g. `author=["natsume", "claude-code"]`. `GET /notes` takes the `kind`, `tags`, and `namespace` filters. `date` is
the note's `occurred_at` when recorded, else `ts_last_active`. `since`/`until` bound the
same event time, `COALESCE(occurred_at, ts_last_active)`, so an episode is found by the
day it happened; recency voting and decay read `ts_last_active` alone. Hits carry their stored text whole; each note is
bounded at write time instead, at ≤1750 chars. `score` is the rerank score, falling back to the
fused RRF score. Hits below the min_score floor (default 0.25, request-adjustable, 0 disables)
are dropped after reranking. Age enters ranking only as the recency voter in the fusion, so an
old row with a strong vector or BM25 rank reaches the reranker. `budget_tokens` (1–32000)
replaces the count and the floor: every fused candidate is reranked, and hits come back in
rerank order until the running estimate of the returned text (`max(1, chars // 4)` per hit) would exceed the budget, so a multi-fact question gets every
relevant note that fits rather than ten. `include_archived` surfaces
archived rows; `/search` hits mark archived rows `"archived": true` since an archived note may have been superseded by
a newer one.

The read path writes nothing to the database.

## Read path — messages (`GET /messages`, claim, cancel)

A general message sent without `expires_at` expires after `MESSAGE_TTL_DAYS` and a
completed handoff 3 days after it is sent; an in_progress or blocked handoff sent without
one has a null `expires_at` and never expires — it leaves the pending set only by claim,
supersede, or cancel. A caller-supplied `expires_at` must be in the future
and at most 30 days out for either purpose.

Messages are read by address, never by similarity. `GET /messages` lists pending,
unexpired messages newest-first (created_at DESC, id DESC) with `namespace`, `purpose`,
`scope`, and `subject` filters — the subject filter matches a case-insensitive
substring of the normalized subject, so a full subject in any spelling variant still finds
its snapshot chain — and a `limit` (default 50, max 100). Rows are headers (id,
namespace, purpose, scope, subject, status, author, created_at, expires_at);
`include_content=true` adds `content`, and a claim returns the content of the message it
takes. No query, no embedding call, no access to
`memory_chunks`.

Delivery is at-most-once. `POST /messages/{id}/claim` is a single conditional UPDATE
on the lifecycle timestamps (`claimed_at IS NULL AND cancelled_at IS NULL AND
superseded_at IS NULL AND (expires_at IS NULL OR expires_at > clock_timestamp())`), so
two concurrent claims are decided by database commit order: exactly one returns the row,
the other gets 409. There is no
lease, ack, or re-read. `DELETE /messages/{id}` cancels a pending message. Terminal rows are invisible to
the list and unclaimable; a stale superseded snapshot id gets 409, an unknown id a 404.
The sending key partitions the retries of an `idempotency_key`; it grants nothing.

## Lifecycle loop

```
 POST /admin/archive {"ids": [...]}   (preview)
                 │  {"confirm": true}
                 ▼
          archived_at set
                 │
 excluded from search ◄───┴───► include_archived=true brings it back
```

`POST /admin/archive` archives the agent notes named by `ids` and stamps the header author on
every row archived; its preview lists them as `notes_to_archive`, and a body without `ids` is
refused with 400. `POST /admin/messages/purge` previews claimed, cancelled, superseded, and
expired messages as `messages_to_delete`, and its confirm pass deletes them, which also
releases their idempotency keys. Archiving a note is reversible by a new save and deleting a
message is not, so a purge covers every namespace. `POST /admin/notes/delete`,
`POST /admin/archive`, and `POST /admin/messages/purge` preview by default and act only with
`{"confirm": true}`. The note routes are reachable over MCP as `archive_notes` and
`delete_notes`; the message purge is a REST-only call.

Namespace deletion counts messages as content: a namespace with messages — pending or
terminal — cannot be unregistered until they are removed.

Retirement is manual: no scheduler runs in-process, so notes stay active until a caller runs
the `/admin/archive` preview and confirm pass, and terminal messages survive until a caller
runs the `/admin/messages/purge` pair. A deployment that wants either periodic drives it from
outside, e.g. a cron job or an n8n schedule.

## Session start

The Claude Code plugin's SessionStart hook runs at every session start, resume, clear, and
compaction. It lists the repository's open handoffs (at most ten) and one line counting the
completed ones, and prints one `<memory-context>` fence:

```
Memory: pending handoffs for this repo. Nothing is claimed; ask to continue one to claim it.
- <date>  <subject>  (status: <status>, id: <id>)
Memory: completed handoffs delivered for this repo: <n>. Claim one to read it; list_messages names their ids.
```

A repository without a portable `origin` remote, a repository with no handoffs, and a failed
fetch print nothing. Memory-context tags inside a subject are defused to `[memory-context]`.
The hook claims nothing and records its connection state in `status.json`.

## Storage

`memory.memory_chunks` — one table for notes.

| column | meaning |
|---|---|
| `id` | `note:<namespace>:<hash>` |
| `source_type` | `agent_note` |
| `source_ref` | `save_memory` |
| `session_id` | the note's own id |
| `chunk_kind` | `personal` · `work` |
| `content_raw` / `distilled` | stored text; BM25 index on `content_raw`, hits display `distilled` first |
| `embedding` | `halfvec(2048)`, HNSW cosine index |
| `ts_last_active` | save time; recency ranking, and `since`/`until` when `occurred_at` is null |
| `occurred_at` | when the remembered event happened; a hit's `date` and the `since`/`until` bound |
| `metadata` | jsonb: `tags`, `author`, `shared`, `archived_by`, `supersedes`, `replaced_by`, `similar_ack`, … |
| `archived_at` | when the row was archived; archived rows leave search |

`memory.messages` — addressed, once-claimed signals: `namespace`, `purpose`
(`message` | `handoff`), `scope`, `subject` with its normalized `subject_key`,
`status` (the report state: `info` | `in_progress` | `blocked` | `completed`),
`content` (canonical Markdown), `author`, `sender_key`, `idempotency_key`, and the
timestamptz lifecycle fields `created_at`, `claimed_at`, `cancelled_at`,
`superseded_at`, `expires_at` (null for an open handoff sent without one). No embedding
column, no search index; a partial index serves the pending listing, and a unique partial index on
(`sender_key`, `idempotency_key`) backs idempotent sends. The lifecycle timestamps stay
internal — responses carry the report status only.

`messages` is outside the retrieval contract: it is read by address, never returned by
search ([ADR-0002](adr/0002-messages-addressed-once-claimed-lane.md)).
