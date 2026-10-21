# memory-base

A selective memory layer for coding agents: distilled notes, chunked documents, indexed
code, and SQL-queryable tables in one pgvector store, served through a single REST API
and an MCP server.

Only distilled content is embedded. Agent notes arrive already distilled by the writing
agent and pass deterministic checks (format, credentials such as API keys, tokens, PINs,
and one-time codes, near duplicates), documents pass through deterministic chunking and a
junk gate before embedding, and code is chunked by tree-sitter. Raw transcripts and raw
files are never embedded. Tabular documents keep their data rows as structured,
never-embedded rows behind a read-only SQL interface, so questions about the numbers are
computed rather than retrieved. Agent-authored messages — one-time signals and handoff
snapshots — are stored as canonical Markdown, never embedded, listed while pending and
consumed by claiming instead of search.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  CONSUMERS      coding agents · n8n · scripts                                 │
└───────┬──────────────────────────────────────────────────────────────────────┘
        │ MCP  (stdio | streamable HTTP :8765)
        ▼
   ┌─────────────┐  26 tools: search / search_code / search_memory /
   │ mcp_server  │            save_memory / list_notes /
   └──────┬──────┘            ingest_document / remove_document
          │                   query_table / ingest_repo / remove_repo / list_repos
          │ HTTP              list_memory_duplicates / archive_notes /
          │                   restore_notes / delete_notes / send_message /
          │                   list_messages / claim_message / cancel_message
          │                   (thin proxy, no logic)
          ▼
   ╔═══════════════════════════════════════════════════════════╗
   ║              REST API  :8010   (the only backend)         ║
   ╚═╤═══════════════╤═══════════════╤═══════════════╤═════════╝
     │ WRITE         │ WRITE         │ WRITE         │ READ + LIFECYCLE
     ▼               ▼               ▼               ▼
  notes/store  documents/        repos/        search/ · tables/ · notes/curation
     │               │               │               │
     └───────────────┴───────┬───────┴───────────────┘
                             ▼
              Postgres 17 + pgvector + pg_textsearch  :5439
              memory_chunks · code_chunks · doc_rows · messages · namespaces ·
              jobs · retrieval_log · consolidation_actions

  side services:  vLLM (LLM / embedding / rerank)
```

Every consumer reaches stored chunks through the REST API, never through the database
directly. `memory_chunks` and `code_chunks` feed search, `doc_rows` holds a tabular
document's data rows for the SQL read path alone ([ADR-0001](docs/adr/0001-table-rows-third-read-contract.md)),
`messages` holds addressed, once-claimed signals that are never embedded and never
searched ([ADR-0002](docs/adr/0002-messages-addressed-once-claimed-lane.md)). `turns`
holds the user's and the assistant's messages verbatim, one row per message, read by
time, session, and literal substring and never ranked ([ADR-0012](docs/adr/0012-a-verbatim-turn-lane-read-on-demand-never-ranked.md)).
`consolidation_actions` records each verdict an agent applied to a group of notes the
server issued; the server judges no content ([ADR-0008](docs/adr/0008-consolidation-judged-by-an-agent-applied-by-the-server.md)).
`entities` holds the registry of people, projects, and things that facts are about, each
change made with the user's consent; it is never embedded or searched.

The same components as an explorable diagram, with guided views and image export:
[docs/diagrams/memory-base-architecture.html](docs/diagrams/memory-base-architecture.html)
— a self-contained page, opened in a browser.

How notes, documents, CSV rows, and code move through the system — write pipelines,
the hybrid-search and table-SQL read paths, the archive lifecycle, and the storage
schema — is documented in [docs/data-flow.md](docs/data-flow.md) and drawn in
[docs/diagrams/memory-base-dataflow.html](docs/diagrams/memory-base-dataflow.html).
Both pages are rendered from the JSON specification beside them.

## Retrieval quality

ZX Bank hit@10: hybrid without rerank 0.98 vs vector-only 0.95; with the default rerank,
hybrid ties vector-only at 0.95. SciFact hit@5: hybrid with rerank 0.86 vs vector-only
0.82. Scores, method, single-leg ablations, and known trade-offs: [docs/benchmarks/retrieval.md](docs/benchmarks/retrieval.md).
End-to-end memory QA on a LongMemEval_S subset through the agent-distilled write path:
[docs/benchmarks/longmemeval.md](docs/benchmarks/longmemeval.md).

## REST API

| method | path | purpose |
|---|---|---|
| `GET` | `/health` | liveness — `200 {status}` whenever the process serves HTTP; reaches nothing outside it, and backs the container healthcheck |
| `GET` | `/health/services` | dependency health — `{status, checks:{db, embedding, rerank, llm}}`; `503` when db, embedding, or rerank is down |
| `POST` | `/search` | hybrid search — `query`, `source` (`all`\|`code`\|`memory`), `top_k`, `min_score`, `budget_tokens`, `kind`, `tags`, `author`, `repo`, `namespaces`, `since`/`until`, `include_archived`; without `budget_tokens` the reranked results are capped at 10 before `top_k` applies; `author` omitted returns the calling author's own and shared notes, `all` every author's, a list of slugs exactly those authors' — document chunks and CSV cards have no author, so they pass an omitted or `all` read and drop out of a slug read |
| `POST` | `/save_memory` | store a distilled note, attributed to the `X-Author` header — `content`, the required `tags`, the optional `entities` (at most ten registered entity slugs; 400 on a malformed slug, an unknown one, or more than ten), the required `kind` (`personal` or `work`, a label for search and listing), the optional `shared` (a note every author's default read scope returns; set at creation only), the optional id of a prior note to archive (400 when the save would leave no active note), and an optional `occurred_at` (ISO 8601, the event's date, stored beside the save time); no chat model judges the note; refused with 409 when a near-identical active note in the caller's default scope exists unless `supersedes` names it or `allow_similar` is set, and before embedding when the content or a tag carries a credential (the error names the credential type only) |
| `GET` | `/notes` | list agent notes newest-first without a query or embedding call — repeated `tags` and `namespace` params, `entity` (a slug), `kind` (`personal` or `work`), repeated `author` (omitted: the caller's own and shared notes; `all`: every author; slugs: exactly those authors), `since`/`until`, `include_archived`, `limit` (default 50, max 200) |
| `POST` | `/notes/entities` | add or remove entity slugs on 1–200 existing active notes — `ids`, `add` (registered entity slugs), `remove` (slugs, any state), `confirm`; each note's set becomes (current ∪ `add`) − `remove`, at most ten; 400 naming the ids when any is not an active note or would exceed ten; without `confirm` returns `{notes: [{id, before, after, changed}]}` and writes nothing; a changed note records `entities_set_by` (the header author) and `entities_set_at` |
| `GET` | `/entities` | the entity registry ordered by slug; see [data flow](docs/data-flow.md#entity-registry) |
| `POST` | `/entities` | create or edit 1–50 entities in one batch — `entities` (`slug`, and `kind`, `name`, `aliases`, `description` on a create), `dry_run` (run every check, write nothing), `consent_method` (`elicitation`\|`conversation`\|`import`, recorded as the client states it); the change is attributed to the `X-Author` header; 409 `{error, holder, value}` when a slug, name, or alias is held by another entity |
| `POST` | `/entities/{slug}/delete` | delete an entity: unlink it from every note and delete its facts and the facts pointing at it — `dry_run` (returns the plan `{note_ids, notes, facts, incoming}` and writes nothing), `expect` (`{notes, facts, incoming}` counts of the preview; a changed plan is 409); its slug, name, and aliases become free |
| `GET` | `/entities/{slug}` | an entity's card — `{entity, facts, incoming}`; `include_past` (`true`\|`false`, default `false`) adds closed facts, `namespace` (repeatable) narrows to those namespaces; 404 for an unknown slug; see [data flow](docs/data-flow.md#facts-and-cards) |
| `POST` | `/admin/facts` | apply 1–100 fact ops (`add`, `replace`, `remove`) all or nothing — `ops`, `dry_run`; the batch is attributed to the `X-Author` header; 400 names the refused op |
| `POST` | `/turns` | append one verbatim turn — `id` (client-chosen UUID), `origin` (`claude_code`\|`hermes`), `session_id`, `role` (`user`\|`assistant`), `content` (at most 64,000 characters), optional `reasoning`, `repo` (`repo:<host>/<path>`), `namespace`; returns `{id, seq, stored}`; a replay with the same payload returns the stored `seq` with `stored: false`, the same `id` with a different payload is 409, a credential is refused, and a session's namespace is fixed by its first turn |
| `GET` | `/sessions` | sessions newest first — `since`, `until`, `origin`, `repo`, repeated `namespace`, `limit` (default 50, max 200); each `{origin, session_id, namespace, repo, started_at, last_at, turns, title}` |
| `GET` | `/sessions/{origin}/{session_id}` | a session's turns in `seq` order — `offset` (a seq), `limit`, and `contains` (case-insensitive substring of at least 2 characters: the matching turns and one neighbour each side, matches marked `match: true`); cut by `limit` or 40,000 characters with `next_offset`; see [data flow](docs/data-flow.md#turn-lane) |
| `GET` | `/turns/search` | turns whose content holds the literal substring `q` (at least 2 characters), newest first, each with a 300-character window — `since`, `until`, `origin`, `repo`, repeated `namespace`, `limit` |
| `POST` | `/messages` | send an addressed message (status `info`, no scope) or, with a `scope` (`repo:<origin>` or `project:<organization>/<project>`), a handoff snapshot (status `in_progress`\|`blocked`\|`completed`) — subject, result, optional `next`/`verification`/`refs`, optional `idempotency_key` and `expires_at`; attributed to the `X-Author` header; rendered to canonical Markdown, rejected past 16 KiB; an identical replay returns 200 |
| `GET` | `/messages` | pending, unexpired messages newest-first without a query or embedding call — repeated `namespace`, `purpose`, `scope`, `subject` (normalized match), `limit` (default 50, max 100) |
| `POST` | `/messages/{id}/claim` | claim a pending message at most once — the loser of a race gets 409; a stale superseded or expired id gets 409, an unknown id a 404 |
| `DELETE` | `/messages/{id}` | cancel a pending message |
| `POST` | `/ingest/document` | multipart upload — `file`, `document_id`, `mode` (`upsert`\|`force`), `origin`, `namespace` (default `default`), repeated `tags`; a credential in the filename, `document_id`, `origin`, or a tag is refused with 400, and one in the content fails the job with nothing stored |
| `DELETE` | `/ingest/documents/{document_id}` | remove a document's chunks and table rows in one namespace (`namespace` query param, default `default`) |
| `POST` | `/tables/query` | read-only SQL over `memory.doc_rows` — `sql` (`SELECT`/`WITH`), optional `namespace` (omitted: every namespace); 1,000-row / 5 MB / 10 s caps |
| `GET` | `/ingest/jobs` | newest document jobs, optionally filtered by exact `origin` and `status` |
| `GET` | `/ingest/jobs/{job_id}` | document job state |
| `POST` | `/repos` | add or re-sync a git repo — `url`, `branch`, `name` |
| `GET` | `/repos` | cached repos with url, branch, head, chunk count |
| `DELETE` | `/repos/{name}` | remove a repo and re-index |
| `GET` | `/repos/jobs/{job_id}` | repo job state |
| `POST` | `/namespaces` | register a namespace — `name` (`^[a-z0-9_-]{1,64}$`) |
| `GET` | `/namespaces` | list the registered namespaces |
| `DELETE` | `/namespaces/{name}` | unregister a namespace with no notes, chunks, table rows, messages, or turns; the reserved `default` namespace cannot be deleted |
| `GET` | `/admin/notes` | active agent notes older than `older_than_days`, in the caller's default read scope unless repeated `author` params name slugs or `all` |
| `POST` | `/admin/notes/delete` | preview, or delete with `confirm`; an id that is not an agent note makes the whole call a 400 that lists them as `missing` |
| `POST` | `/admin/notes/move` | move agent notes into another registered namespace — `ids`, `namespace`; returns `{moved, skipped}`, rewriting each moved note's id to `note:<target>:<hash>`; an id already present in the target, or naming no agent note, is skipped |
| `GET` | `/admin/duplicates` | near-duplicate agent-note pairs above `threshold` (default 0.75) — repeated `author` (omitted: the caller's own and shared notes; `all`: every author; slugs: exactly those authors); each side carries its author |
| `GET` | `/admin/consolidate/groups` | groups of active agent notes that may state the same thing, for an agent to judge; a pair either note lists in `similar_ack` is ignored and counted; a group with a recorded verdict, or with exactly the members of an undone action, is left out and counted `cached`; changes no note — `namespace` (repeatable, default every registered namespace), `author` (omitted: exactly the calling author's notes; a slug: that author; `all`: every author present, one author per group — groups never mix authors), `threshold` (0 < x ≤ 1, default 0.72), `neighbors` (1–50, default 5), `max_group` (2–20, default 6), `max_group_chars` (≥ 500, default 12000), `limit` (groups per namespace, 1–1000, default 200); see [data flow](docs/data-flow.md#consolidation-groups) |
| `POST` | `/admin/consolidate/verdicts` | apply an agent's `keep` / `retire` / `merge` verdicts on issued groups, each alone in one transaction under a per-namespace lock, attributed to the `X-Author` header — `run_id`, `model`, `dry_run` (plan only, no write), the groups call's `threshold` / `neighbors` / `max_group` / `max_group_chars`, `max_actions` (retire and merge per run and namespace, 1–500, default 20), 1–200 `verdicts`; a schema violation refuses the whole request with 400; a merged note carries the members' author, and an existing note is reused as replacement only under the members' author and `shared` rule; a verdict whose group members do not share one author is `rejected` with the reason `group mixes authors: a, b`; each result is `applied`, `planned`, `cached`, `duplicate`, `stale` (with the current groups), `rejected`, or `failed` (an embedder or database error rolled that verdict back; retry it with the same idempotency key); a merge text must pass a two-direction token check (numbers, dates, backticked spans, names) that a changed sentence-initial name, a negation, a caseless-script name, or a version inside an identifier (`v2.0`) passes; see [data flow](docs/data-flow.md#consolidation-verdicts) |
| `POST` | `/admin/consolidate/undo` | reverse one action — `action_id`; the undo records the header author as `undone_by`; restores the notes it archived with their prior metadata and archives a replacement it created; 409 with nothing changed when a later change touched them or for a keep, 404 for an unknown id, the recorded result when already undone; see [data flow](docs/data-flow.md#consolidation-undo-and-actions) |
| `GET` | `/admin/consolidate/actions` | consolidation actions newest first with the full text and lineage of every note they name — `namespace`, `run_id`, `note_id`, `limit` (1–500, default 50) |
| `POST` | `/admin/archive` | preview cold agent notes (`notes_to_archive`), then archive them with `confirm`; `ids` selects agent notes, and the header author is stamped on every row archived; without `ids` the preview selects by the `author` query param (omitted: the caller's default scope; slugs or `all` replace it); an id that is not an active note makes the whole call a 400 that lists them as `missing` |
| `DELETE` | `/admin/sessions/{origin}/{session_id}` | hard-delete every turn of a session; returns `{deleted}` |
| `POST` | `/admin/messages/purge` | preview claimed, cancelled, superseded, and expired messages (`messages_to_delete`), then delete them with `confirm` (`deleted`); deletion is permanent and releases their idempotency keys |
| `POST` | `/admin/restore` | preview, or restore with `confirm`; restoring clears the archiving author and the `replaced_by` and `consolidated_into` lineage; an id that names no stored row makes the whole call a 400 that lists them as `missing` |

The `/admin/*` routes are maintenance routes; any valid key may call them.

An agent runs the consolidation routes on a schedule by following [docs/consolidation-procedure.md](docs/consolidation-procedure.md).

An agent's directions are the description of its own entity, changed with
`register_entity` and the user's consent. Clients deliver the agent's card and the
user's card at session start: the Claude Code plugin's SessionStart hook (options
`author` and `user_entity`) and the Hermes provider's system prompt block (config
`author` and `user_entity`, [integrations/hermes/README.md](integrations/hermes/README.md)).
`Remember` and `Don't remember` lines in the description narrow or widen what the agent
keeps with `save_memory`
([ADR-0010](docs/adr/0010-save-policy-a-minimal-default-that-profiles-refine.md)).

Filters are bound to the source they belong to: `kind`, `tags`, `author`, and
`since`/`until` require `source="memory"`, `repo` requires `source="code"`, and
`source="all"` takes none of them. `repo` is a list of cache directory names as reported by `GET /repos`; an
unknown name matches nothing. Code hits carry their `repo` whether or not the filter is
set. `since` and `until` are ISO 8601 dates or datetimes bounding a row's `occurred_at`,
or `ts_last_active` when none is recorded: a bare date covers that whole day, and naive
values are read as UTC.

A memory hit carries the stored row's `id` (the note id `supersedes` takes), `kind`, and
`tags`, and `shared: true` when the note is shared; a note saved with `supersedes` carries the archived note's id as `supersedes`,
in hits and in `GET /notes` rows; a supersede save whose content matches an active note
archives the target without recording a pointer on the active note. The archived note
records the id of the note that replaced it as `replaced_by`. `GET /notes` rows also carry the lineage
fields they record: `archived_by`, `replaced_by`, `consolidated_into`, `merged_from`,
`merged_dates`, `consolidation_action`, and `undone_action`. `date` is the note's
`occurred_at` when recorded, else its save time. Code hits carry `repo` and optional
`context` instead. `GET /notes` rows carry the same date.

## MCP tools

Each MCP tool is a thin proxy over the REST API, defined in its feature package's
`tools.py` and registered by `mcp_server.py`, which holds no logic of its own. Transport is
stdio by default; `MCP_TRANSPORT=sse|streamable-http` with `MCP_HOST`/`MCP_PORT` serves
over HTTP (Docker serves streamable HTTP on `:8765/mcp`).

`search` · `search_code` · `search_memory` · `save_memory` · `list_notes` ·
`ingest_document` (text formats and CSV) · `remove_document` · `query_table` ·
`ingest_repo` · `remove_repo` · `list_repos` · `list_memory_duplicates` ·
`archive_notes` · `link_entities` · `restore_notes` · `delete_notes` · `send_message` ·
`list_messages` · `claim_message` · `cancel_message` · `list_entities` ·
`register_entity` · `read_entity` · `list_sessions` · `read_session` · `search_transcript`

Each tool takes the REST options its source supports: `include_archived` on `search` and
`search_memory`; `kind`, `tags`, and `since`/`until` only where `source="memory"` holds,
so `search` and `search_code` do not offer them; `repo` on `search_code` alone;
`budget_tokens` on `search` and `search_memory`, which returns hits in rerank order up to
a token budget instead of `top_k` hits above `min_score`.
`list_notes` reads notes by filters alone — no query, no embedding call — for
deterministic reads like every note carrying one subject tag or a time window. The
`author` argument of `search_memory`, `list_notes`, `list_memory_duplicates`, and
`read_entity` is three-way:
omitted returns the caller's own notes and shared notes, `all` every agent's notes, a slug
that agent's notes only. `search` takes no author: its note hits are the caller's own and
shared notes, and `search_memory` widens.

Curation runs over the same tools: `list_memory_duplicates` reads near-duplicate agent-note pairs
with both sides' authors, as its `author` argument selects, and `archive_notes` (`ids`
required; archived rows are stamped with the header author),
`restore_notes`, and `delete_notes` preview by default and act only with `confirm`.
Archiving is restorable and records its author; deleting is permanent and records
nothing. `link_entities` (`ids`, `add`, `remove`) adds or removes entity slugs on
existing notes without changing their content, id, or dates; it previews by default and
writes only with `confirm`.

The message lane is addressed, not searched: `list_messages` reads pending messages at
the start of a session and `claim_message` takes exclusive delivery of each one acted
on — never automatically. `send_message` publishes a general message or, with a scope,
a handoff snapshot; `cancel_message` withdraws a sender's own pending message. A
general message is operational and expires after `MESSAGE_TTL_DAYS`; a handoff stays
pending until it is claimed, superseded, or cancelled unless its sender gives an
`expires_at`; a note is durable knowledge.

## Running

```bash
uv sync
docker compose up -d --build db          # pgvector + pg_textsearch on :5439
docker compose up -d --build api         # REST backend on :8010
docker compose up -d --build mcp         # MCP server, streamable HTTP on :8765
```

### Clients

| client | integration |
|---|---|
| Claude Code | the plugin in [integrations/claude_code](integrations/claude_code/README.md): `/plugin marketplace add yw0nam/memory_layer`, then `/plugin install memory-base@memory_layer`; it brings the hooks, the MCP server, the turn capture, and the `init` and `connect-agent` skills |
| Hermes | the memory provider in [integrations/hermes](integrations/hermes/README.md) |

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

### Backup

`scripts/backup.sh <backup-dir> [keep]` writes a gzipped `pg_dump` of the database and
keeps the newest `keep` dumps (default 14). A daily dump is one crontab line:

```cron
0 4 * * * /path/to/memory_base/scripts/backup.sh /path/to/backups >> /path/to/backup.log 2>&1
```

The database is the only state that needs backing up — the repo cache and CocoIndex
ledger are rebuilt by re-adding repos.

[memory-base-example](https://github.com/yw0nam/memory-base-example) is a runnable
deployment of this stack with a chat interface in front of it: Hermes Agent as an
OpenAI-compatible backend, Open WebUI as the client, and a seed that loads public
documents, a metrics CSV, and source repositories into the three lanes.

## Authentication

Every route except `/health` and `/health/services` requires an `X-API-Key` header
(`ApiKeyAuthMiddleware` in `serve/access/auth.py`); a missing, unknown, or revoked key gets a
fail-closed `401`. The same routes require an `X-Author` header: the author slug the
client declares, matching `^[a-z0-9][a-z0-9-]{0,39}$`; a missing, malformed, or reserved
(`all`) value gets a 400.

A key is a credential for the network boundary and grants nothing else: any valid key may
call every route, and every key is equal. Each client holds its own key, so one host or
agent is cut off by revoking its key alone. The `label` of a key names the client that
holds it.

Keys are provisioned with an operator CLI, not through the API:

```bash
uv run python -m memory_base.serve.access.keys new <label>
uv run python -m memory_base.serve.access.keys list
uv run python -m memory_base.serve.access.keys revoke <key-hash-prefix>
```

`new` prints the plaintext key once, on its last line — only its sha256 hash is stored.
`revoke` takes an 8+ character prefix of the stored hash, as shown by `list`, and
revokes every active key matching it.

Every write — saved note, archive, link, message, fact, consolidation action — is
attributed to the `X-Author` header's author. A note read returns the caller's own notes
and the notes marked `shared` unless it names an `author`
([ADR-0015](docs/adr/0015-author-declared-by-the-client-header-and-default-read-scope.md)).
A namespace is a registered label for filtering:
a write without `namespace` lands in `default`, a read without a namespace filter covers
every namespace, and nothing in a namespace is private.

Each agent gets its own key and author slug, sent as the `X-Author` header. The `connect-agent` skill
([integrations/claude_code/skills/connect-agent/SKILL.md](integrations/claude_code/skills/connect-agent/SKILL.md))
walks through connecting and disconnecting an agent.

The MCP server needs the same headers: over streamable HTTP it forwards the caller's own
`X-API-Key` and `X-Author`, and over stdio (no inbound HTTP request to read them from) it
reads the `MEMORY_API_KEY` and `MEMORY_AUTHOR` environment variables instead.

## Configuration

`.env` (gitignored) holds every endpoint and credential. Required to boot: `DB_URL`,
`POSTGRES_PASSWORD`, `TABLES_QUERY_PASSWORD`, `DATA_ROOT`, and the embedding/rerank vLLM
endpoints (`EMB_URL`/`EMB_MODEL`, `RERANK_URL`/`RERANK_MODEL`). The chat model, used to summarize a CSV into its card, is chosen by
the first non-empty API key (`OPENCODE_API_KEY`, `ZAI_API_KEY`, `OPENAI_API_KEY`, `CLAUDE_API_KEY`), falling back
to the configured vLLM endpoint (`VLLM_URL`/`VLLM_MODEL`) when none is set. The full variable
reference, optional tuning knobs, and private-repository credentials are documented in
[docs/configuration.md](docs/configuration.md).

## Development

```bash
uv run pytest                                        # unit + integration (throwaway Postgres container, needs docker)
uv run pytest -m "not integration"                   # what CI runs
uv run ruff format --check . && uv run ruff check .
uv run python -m memory_base.eval.retrieval          # fixture retrieval eval report
uv run python -m memory_base.eval.retrieval --notes  # labeled real-query replay against live memory
uv run python scripts/longmemeval/extract.py --dataset PATH  # LongMemEval harness, see docs/benchmarks/longmemeval.md
uv run python -m memory_base.eval.mcp_writer --dataset PATH --questions ID[,ID...]  # LongMemEval writer through the real MCP tools (headless Claude Code, throwaway Postgres)
```

Work happens in a git worktree and lands via PR; `main` requires a PR and green CI (lint,
unit tests, test-guard, PR title). See `AGENTS.md` for the full contributor contract.
Runtime evidence for a PR comes from `uv run python -m memory_base.eval.sandbox`, which serves the
real REST API on a throwaway Postgres and prints its URL and a key.
