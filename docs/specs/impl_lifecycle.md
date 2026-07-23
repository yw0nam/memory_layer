# Memory lifecycle management: note lifecycle, dedup, cold tier (#30)

All architecture decisions below are user-confirmed via interview. This spec elaborates them; it
does not re-decide. Management operations live on the REST API only — the MCP server never
exposes them. Every destructive endpoint defaults to dry-run and executes only with an explicit
`confirm: true`.

## 1. Archived state

`memory_chunks` gains one additive column (idempotent DDL in `schema.py`):

```sql
ALTER TABLE memory_chunks ADD COLUMN IF NOT EXISTS archived_at double precision;
```

`archived_at IS NULL` = active. Archiving is the only "destructive" v1 operation and is a soft,
reversible state change — no row deletion except explicit note pruning (below).

Retrieval excludes archived rows by default: the history legs in `retrieval/search.py` add
`WHERE archived_at IS NULL`; `search()` gains `include_archived: bool = False` which lifts the
filter and also disables time-decay weighting for that call — the flag requests archival
recall, and recency decay would bury exactly the rows old enough to be archived. REST `POST /search` accepts an optional `"include_archived": false` field and passes it
through. The MCP search tools keep their current signatures (always default search — archived
recall is a management concern, and management never crosses MCP).

## 2. Note lifecycle

### save_memory `supersedes` ("both" mode, confirmed)

- `save_memory(content, kind, tags, supersedes=None)` — MCP tool, REST body, and `save_note`
  all gain the optional `supersedes: <note id>` parameter.
- When given: validate the id refers to an existing `agent_note` row (else error 400 /
  ValueError); in one transaction, insert the new note and set `archived_at = now` on the old
  one. The superseded note is archived, never deleted.
- The save response gains `"similar"`: up to 3 existing active notes whose embedding cosine
  similarity to the new content exceeds `NOTE_SIMILAR_THRESHOLD` (default 0.85), each as
  `{id, score, text}` (text truncated). Computed with the embedding already produced for the
  insert — one extra pgvector query. The agent uses these hints to decide follow-up supersedes.
  Response shape: `{"id", "kind", "stored", "superseded": <id or null>, "similar": [...]}`.

### Review list and explicit pruning

- `GET /admin/notes?older_than_days=N` — active `agent_note` rows with `ts_last_active` older
  than N days (default 90), with `hit_count`/`last_hit_at`, ordered oldest-first. Read-only
  listing for manual review; nothing is deleted silently.
- `POST /admin/notes/delete` — body `{"ids": [...], "confirm": false}`. Dry-run returns the
  matching rows; `confirm: true` deletes them. Only `agent_note` rows are deletable through this
  endpoint (other kinds → 400). This is the explicit manual-pruning path.

## 3. Near-duplicate detection (candidates only, confirmed)

- `GET /admin/duplicates?threshold=0.9&kind=<optional>&limit=50` — active-row pairs whose
  embedding cosine similarity ≥ threshold, as
  `{"pairs": [{"a": {id, kind, text}, "b": {...}, "score"}]}`, highest similarity first.
- Self-join on `memory_chunks` with pgvector cosine distance (fine at current scale; `limit`
  caps output). Read-only: the API never merges or deletes — the managing agent decides per pair
  and acts through existing endpoints (`/admin/notes/delete`, save_memory with `supersedes`, or
  `/admin/archive` below).

## 4. Cold tier

- Env-tunable thresholds (confirmed conservative defaults): `COLD_AGE_DAYS=180`,
  `COLD_UNHIT_DAYS=90`.
- Candidate rule: `archived_at IS NULL AND ts_last_active < now - COLD_AGE_DAYS AND
  COALESCE(last_hit_at, ts_last_active) < now - COLD_UNHIT_DAYS` — a never-hit row falls back
  to its own activity timestamp, so freshly tracked rows are not archived merely because hit
  logging is younger than they are; the dry-run listing is the safety net either way.
- `POST /admin/archive` — body `{"confirm": false}`. Dry-run returns candidates (id, kind,
  age days, `hit_count`, last hit); `confirm: true` stamps `archived_at = now` and reports the
  count. Admin listings (`/admin/notes`, `/admin/archive` dry-run) always carry
  `hit_count`/`last_hit_at` so the managing agent can weigh retrieval frequency before acting.
- `POST /admin/restore` — body `{"ids": [...], "confirm": false}`. Clears `archived_at` on the
  given rows; same dry-run/confirm semantics. Makes archiving fully reversible.

## 5. Components

```
src/memory_base/
  schema.py          # + archived_at column
  retrieval/search.py  # archived filter + include_archived flag
  serve/
    admin.py         # NEW: /admin/* handlers (notes list/delete, duplicates, archive, restore)
    notes.py         # save_note: supersedes + similar hints
    api.py           # routes only: /admin/* wiring, /search include_archived pass-through
    mcp_server.py    # save_memory tool signature + docstring gain supersedes (proxy unchanged otherwise)
```

`admin.py` follows the established boundary: handlers in `api.py` parse/validate and delegate;
queries and state changes live in `admin.py`. No management tool is added to the MCP server.

## 6. Tests (TDD)

Unit (CI, no DB):

- Every `/admin/*` endpoint: dry-run is the default; mutation function is NOT called without
  `confirm: true` and IS called with it (monkeypatched admin functions); validation errors
  (bad ids, non-note kinds on delete, bad threshold) → 400.
- save_memory validation: unknown `supersedes` id → error; response shape pins including
  `superseded`/`similar` keys.
- Cold-tier candidate rule pinned as a pure predicate (timestamp arithmetic, COALESCE fallback).
- MCP proxy: save_memory forwards `supersedes`; tool list still exactly the four tools.

Integration (real DB + vLLM):

- Seeded old synthetic rows: `/admin/archive` dry-run lists exactly them; confirm archives;
  archived rows disappear from default `/search` and reappear with `include_archived`;
  `/admin/restore` brings them back. Cleanup in finally.
- save_memory with `supersedes`: old note archived, new stored, `similar` hints returned for a
  seeded near-identical note.
- `/admin/duplicates` finds a seeded near-dup pair; `/admin/notes` lists an aged note;
  `/admin/notes/delete` dry-run then confirm removes it.

## 7. Runtime evidence (PR)

- Maintenance run on the real corpus: `/admin/archive` dry-run listing with before/after stats
  (confirmed run only if the listing looks sane), `/admin/duplicates` on the real 622 rows
  (expected: the known near-dup session pair), full curl transcripts, and DB state checks.
