# memory-base

A selective memory layer for coding agents: distilled notes in a pgvector store and
addressed messages between sessions, served through a single REST API and an MCP server.

Only distilled content is embedded. Agent notes arrive already distilled by the writing
agent and pass deterministic checks (format, credentials such as API keys, tokens, PINs,
and one-time codes, near duplicates); raw transcripts are never stored. Agent-authored
messages — one-time signals and handoff snapshots — are stored as canonical Markdown,
never embedded, listed while pending and consumed by claiming instead of search.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  CONSUMERS      coding agents · scripts                                       │
└───────┬──────────────────────────────────────────────────────────────────────┘
        │ MCP  (stdio | streamable HTTP :8765)
        ▼
   ┌─────────────┐  9 tools: search_memory / save_memory / list_notes /
   │ mcp_server  │           archive_notes / delete_notes / send_message /
   └──────┬──────┘           list_messages / claim_message / cancel_message
          │ HTTP             (thin proxy, no logic)
          ▼
   ╔═══════════════════════════════════════════════════════════╗
   ║              REST API  :8010   (the only backend)         ║
   ╚═╤═══════════════════╤═══════════════════╤═════════════════╝
     │ WRITE             │ READ              │ LIFECYCLE
     ▼                   ▼                   ▼
  notes/store         search/           notes/curation · messages/
     │                   │                   │
     └───────────────────┴─────────┬─────────┘
                                   ▼
              Postgres 17 + pgvector + pg_textsearch  :5439
              memory_chunks · messages · namespaces · api_keys

  side services:  vLLM (embedding / rerank)
```

Every consumer reaches stored notes through the REST API, never through the database
directly. `memory_chunks` feeds search, `messages` holds addressed, once-claimed signals
that are never embedded and never searched
([ADR-0002](docs/adr/0002-messages-addressed-once-claimed-lane.md)).

How notes and messages move through the system — the write pipeline, the hybrid-search
read path, the archive lifecycle, and the storage schema — is documented in
[docs/data-flow.md](docs/data-flow.md).

## REST API

| method | path | purpose |
|---|---|---|
| `GET` | `/health` | liveness — `200 {status}` whenever the process serves HTTP; reaches nothing outside it, and backs the container healthcheck |
| `GET` | `/health/services` | dependency health — `{status, checks:{db, embedding, rerank}}`; `503` when any of them is down |
| `POST` | `/search` | hybrid note search — `query`, `top_k`, `min_score`, `budget_tokens`, `kind`, `tags`, `author`, `namespaces`, `since`/`until`, `include_archived`; without `budget_tokens` the reranked results are capped at 10 before `top_k` applies; `author` omitted returns the calling author's own and shared notes, `all` every author's, a list of slugs exactly those authors' |
| `POST` | `/save_memory` | store a distilled note, attributed to the `X-Author` header — `content`, the required `tags`, the required `kind` (`personal` or `work`, a label for search and listing), the optional `shared` (a note every author's default read scope returns; set at creation only), the optional id of a prior note to archive (400 when the save would leave no active note), and an optional `occurred_at` (ISO 8601, the event's date, stored beside the save time); refused with 409 when a near-identical active note in the caller's default scope exists unless `supersedes` names it or `allow_similar` is set, and before embedding when the content or a tag carries a credential (the error names the credential type only) |
| `GET` | `/notes` | list agent notes newest-first without a query or embedding call — repeated `tags` and `namespace` params, `kind` (`personal` or `work`), repeated `author` (omitted: the caller's own and shared notes; `all`: every author; slugs: exactly those authors), `since`/`until`, `include_archived`, `limit` (default 50, max 200) |
| `POST` | `/messages` | send an addressed message (status `info`, no scope) or, with a `scope` (`repo:<origin>` or `project:<organization>/<project>`), a handoff snapshot (status `in_progress`\|`blocked`\|`completed`) — subject, result, optional `next`/`verification`/`refs`, optional `idempotency_key` and `expires_at`; attributed to the `X-Author` header; rendered to canonical Markdown, rejected past 16 KiB; an identical replay returns 200 |
| `GET` | `/messages` | pending, unexpired messages newest-first without a query or embedding call — repeated `namespace`, `purpose`, `scope`, `subject` (case-insensitive substring of the normalized subject), `include_content` (default false: headers only), `limit` (default 50, max 100) |
| `POST` | `/messages/{id}/claim` | claim a pending message at most once — the loser of a race gets 409; a stale superseded or expired id gets 409, an unknown id a 404 |
| `DELETE` | `/messages/{id}` | cancel a pending message |
| `POST` | `/namespaces` | register a namespace — `name` (`^[a-z0-9_-]{1,64}$`) |
| `GET` | `/namespaces` | list the registered namespaces |
| `DELETE` | `/namespaces/{name}` | unregister a namespace with no notes or messages; the reserved `default` namespace cannot be deleted |
| `POST` | `/admin/notes/delete` | preview, or delete with `confirm`; an id that is not an agent note makes the whole call a 400 that lists them as `missing` |
| `POST` | `/admin/archive` | preview the agent notes named by `ids` (`notes_to_archive`), then archive them with `confirm`, stamping the header author on every row archived; a body without `ids`, or an id that is not an active note, is a 400 (the id case lists them as `missing`) |
| `POST` | `/admin/messages/purge` | preview claimed, cancelled, superseded, and expired messages (`messages_to_delete`), then delete them with `confirm` (`deleted`); deletion is permanent and releases their idempotency keys |

The `/admin/*` routes are maintenance routes; any valid key may call them.

`since` and `until` are ISO 8601 dates or datetimes bounding a row's `occurred_at`, or
`ts_last_active` when none is recorded: a bare date covers that whole day, and naive values
are read as UTC.

A search hit carries the stored row's `id` (the note id `supersedes` takes), `kind`, and
`tags`, and `shared: true` when the note is shared; a note saved with `supersedes` carries the archived note's id as `supersedes`,
in hits and in `GET /notes` rows; a supersede save whose content matches an active note
archives the target without recording a pointer on the active note. The archived note
records the id of the note that replaced it as `replaced_by`. `GET /notes` rows also carry the lineage
fields they record: `supersedes`, `archived_by`, and `replaced_by`. `date` is the note's
`occurred_at` when recorded, else its save time. `GET /notes` rows carry the same date.

## MCP tools

Each MCP tool is a thin proxy over the REST API, defined in its feature package's
`tools.py` and registered by `mcp_server.py`, which holds no logic of its own. Transport is
stdio by default; `MCP_TRANSPORT=sse|streamable-http` with `MCP_HOST`/`MCP_PORT` serves
over HTTP (Docker serves streamable HTTP on `:8765/mcp`).

`search_memory` · `save_memory` · `list_notes` · `archive_notes` · `delete_notes` ·
`send_message` · `list_messages` · `claim_message` · `cancel_message`

`search_memory` takes `include_archived`, `kind`, `tags`, `since`/`until`, `author`, and
`budget_tokens`, which returns hits in rerank order up to a token budget instead of
`top_k` hits above `min_score`.
`list_notes` reads notes by filters alone — no query, no embedding call — for
deterministic reads like every note carrying one subject tag or a time window. The
`author` argument of `search_memory` and `list_notes` is three-way:
omitted returns the caller's own notes and shared notes, `all` every agent's notes, a slug
that agent's notes only.

Curation runs over the same tools: `archive_notes` (`ids` required; archived rows are
stamped with the header author) and `delete_notes` preview by default and act only with
`confirm`. Archiving records its author; deleting is permanent and records nothing.

The message lane is addressed, not searched: `list_messages` reads pending messages at
the start of a session and `claim_message` takes exclusive delivery of each one acted
on — never automatically. `send_message` publishes a general message or, with a scope,
a handoff snapshot; `cancel_message` withdraws a sender's own pending message. A
general message is operational and expires after `MESSAGE_TTL_DAYS`; a completed
handoff expires after 3 days and an in_progress or blocked one stays pending until it
is claimed, superseded, or cancelled, unless its sender gives an `expires_at`; a note
is durable knowledge.

## Running

```bash
uv sync
docker compose up -d --build db          # pgvector + pg_textsearch on :5439
docker compose up -d --build api         # REST backend on :8010
docker compose up -d --build mcp         # MCP server, streamable HTTP on :8765
uv run python -m memory_base.serve.access.keys new <label>   # prints a client's API key once
```

### Clients

| client | integration |
|---|---|
| Claude Code | the plugin in [integrations/claude_code](integrations/claude_code/README.md): `/plugin marketplace add yw0nam/memory_layer`, then `/plugin install memory-base@memory_layer`; it brings the SessionStart hook that prints the repo's handoff headers, the MCP server, and the `init` and `connect-agent` skills |
| Hermes | an `mcp_servers` entry for the MCP server with the `X-API-Key` and `X-Author` headers |

The memory schema is created on first write (`ensure_schema`).

### Deploy

`scripts/deploy.sh` fast-forwards the main checkout, rebuilds the `api` and `mcp`
services, and waits until they report healthy. It refuses a checkout that is not on
`main`. An MCP client reconnects after the `mcp` container restarts.

### Backup

`scripts/backup.sh <backup-dir> [keep]` writes a gzipped `pg_dump` of the database and
keeps the newest `keep` dumps (default 14). A daily dump is one crontab line:

```cron
0 4 * * * /path/to/memory_base/scripts/backup.sh /path/to/backups >> /path/to/backup.log 2>&1
```

The database is the only state that needs backing up.

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

Every write — saved note, archive, message — is
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
`POSTGRES_PASSWORD`, `DATA_ROOT`, and the embedding/rerank vLLM endpoints
(`EMB_URL`/`EMB_MODEL`, `RERANK_URL`/`RERANK_MODEL`). The full variable reference and
optional tuning knobs are documented in [docs/configuration.md](docs/configuration.md).

## Development

```bash
uv run pytest                                        # unit + integration (throwaway Postgres container, needs docker)
uv run pytest -m "not integration"                   # what CI runs
uv run ruff format --check . && uv run ruff check .
uv run python -m memory_base.eval.sandbox  # real REST API on a throwaway Postgres; prints URL and key (runtime evidence)
```

Work happens in a git worktree and lands via PR; `main` requires a PR and green CI (lint,
unit tests, test-guard, docs-guard, PR title). The docs-guard job and the `PostToolUse` edit
hook share `scripts/ci/docs-guard.sh`, which flags change-narrative vocabulary in markdown.
See `AGENTS.md` for the full contributor contract.
