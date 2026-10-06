# memory-base

A selective memory layer for coding agents: distilled notes, chunked documents, indexed
code, and SQL-queryable tables in one pgvector store, served through a single REST API
and an MCP server.

Only distilled content is embedded. Agent notes arrive already distilled by the writing
agent and pass deterministic checks (format, credentials, near duplicates), documents
pass through deterministic chunking and a junk gate before embedding, and code is
chunked by tree-sitter. Raw transcripts and raw files are never embedded. Tabular
documents keep their data rows as structured, never-embedded rows behind a read-only
SQL interface, so questions about the numbers are computed rather than retrieved.
Agent-authored messages — one-time signals and handoff snapshots — are stored as
canonical Markdown, never embedded, listed while pending and consumed by claiming
instead of search.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  CONSUMERS      coding agents · n8n · scripts                                 │
└───────┬──────────────────────────────────────────────────────────────────────┘
        │ MCP  (stdio | streamable HTTP :8765)
        ▼
   ┌─────────────┐  21 tools: search / search_code / search_memory /
   │ mcp_server  │            save_memory / list_notes /
   └──────┬──────┘            ingest_document / remove_document
          │                   query_table / ingest_repo / remove_repo / list_repos
          │ HTTP              list_memory_duplicates / archive_notes /
          │                   restore_notes / delete_notes / send_message /
          │                   list_messages / claim_message / cancel_message /
          │                   get_my_profile / update_my_profile /
          │                   propose_user_profile
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
searched ([ADR-0002](docs/adr/0002-messages-addressed-once-claimed-lane.md)). No raw
conversation turns are stored ([ADR-0005](docs/adr/0005-no-raw-conversation-storage.md)).
`consolidation_actions` records each verdict an agent applied to a group of notes the
server issued; the server judges no content ([ADR-0008](docs/adr/0008-consolidation-judged-by-an-agent-applied-by-the-server.md)).
`agent_profiles` and `profile_proposals` hold each agent's profile versions and the
proposals the user decides; they are never embedded or searched.

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
| `POST` | `/search` | hybrid search — `query`, `source` (`all`\|`code`\|`memory`), `top_k`, `min_score`, `budget_tokens`, `kind`, `tags`, `author`, `repo`, `namespaces`, `since`/`until`, `include_archived`; without `budget_tokens` the reranked results are capped at 10 before `top_k` applies |
| `POST` | `/save_memory` | store a distilled note — `content`, the required `author` and `tags`, the required `kind` (`personal` or `work`, a label for search and listing), the optional id of a prior note to archive (400 when the save would leave no active note), and an optional `occurred_at` (ISO 8601, the event's date, stored beside the save time); no chat model judges the note; refused with 409 when a near-identical active note exists unless `supersedes` names it or `allow_similar` is set, and before embedding when the content or a tag carries a credential (the error names the credential type only) |
| `GET` | `/notes` | list agent notes newest-first without a query or embedding call — repeated `tags` and `namespace` params, `kind` (`personal` or `work`), `author`, `since`/`until`, `include_archived`, `limit` (default 50, max 200) |
| `GET` | `/profiles` | one owner's profile — `owner` (an agent's author slug) required; `{owner, self_version, self, user_version, user, pending_proposal}` from one snapshot, a part `{content, created_at}` or null when absent or emptied, its version kept; a key whose authors hold the owner or `user` only; see [data flow](docs/data-flow.md#profiles) |
| `PUT` | `/profiles/self` | replace the owner's `self` part (persona, working rules, conventions) — `owner` one of the key's authors, `content`, `max_chars` (200–20000, default 4000); `written` or `unchanged` with the version; empty content clears the part; 400 for an over-budget or credential-bearing text |
| `POST` | `/profiles/user/proposals` | propose a full replacement of the owner's `user` part — `owner` one of the key's authors, `content`, `reason` (1–1000 chars), `base_version` (the current user version, 0 when none), `max_chars` (200–20000, default 3000); 201 `{id, status, superseded}`, superseding the owner's pending proposal; 409 `{error: "stale", version}` when the base is not current |
| `GET` | `/profiles/user/proposals` | proposals newest first — `owner`, `status`, `limit` (1–200, default 20); without `owner`, every owner's, for keys carrying the `user` author only |
| `GET` | `/profiles/user/proposals/{id}` | one proposal with `current_user_version` and `current_user_content` — the proposal owner's key or a `user` key |
| `POST` | `/profiles/user/proposals/{id}/approve` | store a pending proposal as the owner's next user version (author `user`) — keys carrying the `user` author only; optional `note`; 409 `not_pending` with the status, or `stale` with the current version (the proposal stays pending) |
| `POST` | `/profiles/user/proposals/{id}/reject` | reject a pending proposal — keys carrying the `user` author only; optional `note`; 409 `not_pending` with the status |
| `GET` | `/profiles/versions` | one part's versions newest first with `content`, `author`, `proposal_id`, `created_at` — `owner`, `part` (`self`\|`user`), `limit` (1–200, default 20) |
| `POST` | `/messages` | send an addressed message (status `info`, no scope) or, with a `scope` (`repo:<origin>` or `project:<organization>/<project>`), a handoff snapshot (status `in_progress`\|`blocked`\|`completed`) — subject, result, optional `next`/`verification`/`refs`, `author`, optional `idempotency_key` and `expires_at`; rendered to canonical Markdown, rejected past 16 KiB; an identical replay returns 200 |
| `GET` | `/messages` | pending, unexpired messages newest-first without a query or embedding call — repeated `namespace`, `purpose`, `scope`, `subject` (normalized match), `limit` (default 50, max 100) |
| `POST` | `/messages/{id}/claim` | claim a pending message at most once — the loser of a race gets 409; a stale superseded or expired id gets 409, an unknown or out-of-scope id a 404 |
| `DELETE` | `/messages/{id}` | cancel a pending message — the sender, or an admin key for any accessible one |
| `POST` | `/ingest/document` | multipart upload — `file`, `document_id`, `mode` (`upsert`\|`force`), `origin`, `namespace` (default the key's home), repeated `tags`; overwriting an existing `document_id` is creator-or-admin only; a credential in the filename, `document_id`, `origin`, or a tag is refused with 400, and one in the content fails the job with nothing stored |
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
| `DELETE` | `/namespaces/{name}` | unregister a namespace with no notes, chunks, table rows, or messages — the namespace's owner or an admin key only; the reserved `default` namespace cannot be deleted |
| `GET` | `/keys/{label}/authors` | a label's author allowlist — an admin key reads any label, a member key only its own |
| `PUT` | `/keys/{label}/authors` | replace a label's allowlist — `authors` (slugs matching `^[a-z0-9][a-z0-9-]{0,39}$`); admin keys only |
| `GET` | `/admin/notes` | active agent notes older than `older_than_days` |
| `POST` | `/admin/notes/delete` | preview, or delete with `confirm` |
| `POST` | `/admin/notes/move` | move agent notes into another registered namespace — admin key only; `ids`, `namespace`; returns `{moved, skipped}`, rewriting each moved note's id to `note:<target>:<hash>`; an id already present in the target, or naming no agent note, is skipped |
| `GET` | `/admin/duplicates` | near-duplicate agent-note pairs above `threshold` |
| `GET` | `/admin/consolidate/groups` | groups of active agent notes that may state the same thing, for an agent to judge; a pair either note lists in `similar_ack` is ignored and counted; a group with a recorded verdict, or with exactly the members of an undone action, is left out and counted `cached`; changes no note — an admin key with `consolidator` in its authors only; `namespace` (repeatable, default every registered namespace), `threshold` (0 < x ≤ 1, default 0.72), `neighbors` (1–50, default 5), `max_group` (2–20, default 6), `max_group_chars` (≥ 500, default 12000), `limit` (groups per namespace, 1–1000, default 200); see [data flow](docs/data-flow.md#consolidation-groups) |
| `POST` | `/admin/consolidate/verdicts` | apply an agent's `keep` / `retire` / `merge` verdicts on issued groups, each alone in one transaction under a per-namespace lock — same key as the groups route, `author` one of its authors; `run_id`, `model`, `dry_run` (plan only, no write), the groups call's `threshold` / `neighbors` / `max_group` / `max_group_chars`, `max_actions` (retire and merge per run and namespace, 1–500, default 20), 1–200 `verdicts`; a schema violation refuses the whole request with 400; each result is `applied`, `planned`, `cached`, `duplicate`, `stale` (with the current groups), `rejected`, or `failed` (an embedder or database error rolled that verdict back; retry it with the same idempotency key); a merge text must pass a two-direction token check (numbers, dates, backticked spans, names) that a changed sentence-initial name, a negation, a caseless-script name, or a version inside an identifier (`v2.0`) passes; see [data flow](docs/data-flow.md#consolidation-verdicts) |
| `POST` | `/admin/consolidate/undo` | reverse one action — `action_id`, `author`; restores the notes it archived with their prior metadata and archives a replacement it created; 409 with nothing changed when a later change touched them or for a keep, 404 for an unknown id, the recorded result when already undone; see [data flow](docs/data-flow.md#consolidation-undo-and-actions) |
| `GET` | `/admin/consolidate/actions` | consolidation actions newest first with the full text and lineage of every note they name — `namespace`, `run_id`, `note_id`, `limit` (1–500, default 50) |
| `POST` | `/admin/archive` | preview cold agent notes (`notes_to_archive`), then archive them with `confirm`; `ids` selects agent notes in the caller's scope and requires an `author`, stamped on every row archived |
| `POST` | `/admin/messages/purge` | preview claimed, cancelled, superseded, and expired messages (`messages_to_delete`), then delete them with `confirm` (`deleted`); deletion is permanent and releases their idempotency keys, so a member key purges only the namespaces it owns |
| `POST` | `/admin/restore` | preview, or restore with `confirm`; restoring clears the archiving author and the `replaced_by` and `consolidated_into` lineage |

An agent runs the consolidation routes on a schedule by following [docs/consolidation-procedure.md](docs/consolidation-procedure.md).

Profiles belong to agents
([ADR-0009](docs/adr/0009-profiles-owned-by-agents-user-part-approved-by-the-user.md)).
Each owner (an agent's author slug) reads both parts with the MCP tool `get_my_profile`,
has a `self` part it writes with `update_my_profile`, and a `user` part that changes only
when the user approves the agent's `propose_user_profile` proposal with a key carrying the
`user` author. Clients deliver the configured owner's profile at session start: the Claude
Code SessionStart hook (`integrations/claude_code/session_start_hook.py`, owner
`MEMORY_BASE_AUTHOR`, default `claude-code`, installed with a 10-second timeout as
documented in `integrations/claude_code/prefetch_hook.py`) and the Hermes provider's
system prompt block (config `owner`,
[integrations/hermes/README.md](integrations/hermes/README.md)). Both print the two parts
under their version lines and a notice while a proposal is pending. `Remember` and `Don't
remember` sections in either part narrow or widen what the agent keeps with `save_memory`;
the user part's sections outrank the agent's own
([ADR-0010](docs/adr/0010-save-policy-a-minimal-default-that-profiles-refine.md)). The
user decides with the approval CLI and the agents follow the `memory-profile-approval`
skill; install both as described in
[integrations/profile_approval/README.md](integrations/profile_approval/README.md).

Filters are bound to the source they belong to: `kind`, `tags`, `author`, and
`since`/`until` require `source="memory"`, `repo` requires `source="code"`, and
`source="all"` takes none of them. `repo` is a list of cache directory names as reported by `GET /repos`; an
unknown name matches nothing. Code hits carry their `repo` whether or not the filter is
set. `since` and `until` are ISO 8601 dates or datetimes bounding a row's `occurred_at`,
or `ts_last_active` when none is recorded: a bare date covers that whole day, and naive
values are read as UTC.

A memory hit carries the stored row's `id` (the note id `supersedes` takes), `kind`, and
`tags`; a note saved with `supersedes` carries the archived note's id as `supersedes`,
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
`archive_notes` · `restore_notes` · `delete_notes` · `send_message` ·
`list_messages` · `claim_message` · `cancel_message` · `get_my_profile` ·
`update_my_profile` · `propose_user_profile`

Each tool takes the REST options its source supports: `include_archived` on `search` and
`search_memory`; `kind`, `tags`, and `since`/`until` only where `source="memory"` holds,
so `search` and `search_code` do not offer them; `repo` on `search_code` alone;
`budget_tokens` on `search` and `search_memory`, which returns hits in rerank order up to
a token budget instead of `top_k` hits above `min_score`.
`list_notes` reads notes by filters alone — no query, no embedding call — for
deterministic reads like every note carrying one subject tag or a time window. `author` filters
`search_memory` and `list_notes` to one agent's notes.

Curation runs over the same tools: `list_memory_duplicates` reads near-duplicate agent-note pairs
with both sides' authors, and `archive_notes` (`ids` and `author` required),
`restore_notes`, and `delete_notes` preview by default and act only with `confirm`.
Archiving is restorable and records its author; deleting is permanent and records
nothing.

The message lane is addressed, not searched: `list_messages` reads pending messages at
the start of a session and `claim_message` takes exclusive delivery of each one acted
on — never automatically. `send_message` publishes a general message or, with a scope,
a handoff snapshot; `cancel_message` withdraws a sender's own pending message. A
general message is operational and expires after `MESSAGE_TTL_DAYS`; a handoff stays
pending until it is claimed, superseded, or cancelled unless its sender gives an
`expires_at`; a note is durable knowledge.

The profile tools read and write an agent's own profile: `get_my_profile` returns both
parts with their versions and the pending proposal, `update_my_profile` replaces its whole
`self` part, and `propose_user_profile` proposes a full replacement of its `user` part
against the user version delivered at session start or read with `get_my_profile`. A stale
proposal fails with the current version and asks the agent to refresh its profile context
and reconsider the replacement. No MCP tool approves or rejects a profile; the user
decides with the approval CLI.

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
fail-closed `401`.

Keys are provisioned with an operator CLI, not through the API:

```bash
uv run python -m memory_base.serve.access.keys new <label> [--home <namespace>] [--admin] [--author <slug>]...
uv run python -m memory_base.serve.access.keys authors <label> [<slug>...]
uv run python -m memory_base.serve.access.keys list
uv run python -m memory_base.serve.access.keys revoke <key-hash-prefix>
```

`new` prints the plaintext key once, on its last line — only its sha256 hash is stored. `--home` sets the
namespace `save_memory` and document ingest default into (`default` when omitted);
minting fails if that namespace does not exist or is not accessible to the label.
`revoke` takes an 8+ character prefix of the stored hash, as shown by `list`, and
revokes every active key matching it.

An admin key (`--admin`) can read and act in every namespace. A member key's allowed
set is every public namespace plus any private namespace it owns — ownership is set to
the minting key's label when the namespace is created with `visibility: private`.
Requests naming a namespace outside that set get `403`.

Every saved note names its author, and the value must be in the key's author allowlist
(`api_keys.authors`; a key minted without `--author` cannot save until an allowlist is set).
Admin keys get no bypass. `keys authors <label>` prints a label's allowlist and, given
slugs, replaces it on every active key of the label; over REST the same list is
`GET /keys/{label}/authors` and `PUT /keys/{label}/authors` (admin only).

Each agent gets its own key whose allowlist is its own author slug, so it writes only its
own notes and profile. The `memory-base-connect-agent` skill
([integrations/skills/memory-base-connect-agent/SKILL.md](integrations/skills/memory-base-connect-agent/SKILL.md))
walks through connecting, changing, and disconnecting an agent.

The MCP server needs the same header: over streamable HTTP it forwards the caller's own
`X-API-Key`, and over stdio (no inbound HTTP request to read one from) it reads the
`MEMORY_API_KEY` environment variable instead.

## Configuration

`.env` (gitignored) holds every endpoint and credential. Required to boot: `DB_URL`,
`POSTGRES_PASSWORD`, `TABLES_QUERY_PASSWORD`, `DATA_ROOT`, and the embedding/rerank vLLM
endpoints (`EMB_URL`/`EMB_MODEL`, `RERANK_URL`/`RERANK_MODEL`). The chat model, used to summarize a CSV into its card, is chosen by
the first non-empty API key (`ZAI_API_KEY`, `OPENAI_API_KEY`, `CLAUDE_API_KEY`), falling back
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
