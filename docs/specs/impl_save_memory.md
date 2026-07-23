# Implementation spec: save_memory MCP tool (Issue #16)

## Principle

Real-time, high-precision write channel. The conversing agent is the selectivity
judge: it stores already-distilled, high-signal content, so the server does **no
LLM distillation** — embed and insert only. Batch ingestion remains the coverage
channel. Retrieval and serving are unchanged: note rows surface through the same
hybrid search as session/decision rows.

## Tool contract (serve/mcp_server.py)

```python
@mcp.tool()
async def save_memory(content: str, kind: str = "note", tags: list[str] | None = None) -> dict
```

- `content`: the memory text, **must be English** (the tool description mandates
  it regardless of conversation language) and already distilled — a decision, a
  user preference, or a hard-won troubleshooting conclusion. Empty/whitespace or
  > 4000 chars raises ValueError (FastMCP surfaces it as a tool error).
- `kind`: `"note"` (default) or `"decision"` — anything else raises ValueError.
- `tags`: optional labels stored in `metadata.tags`.
- Returns `{"id": <id>, "kind": <kind>, "stored": <bool>}`; `stored` is False
  when the id already exists (dedup no-op).

The tool description states the storage criteria explicitly (worth remembering
in a future session; not running commentary; English only) to prevent dumping.

## Row mapping (memory_chunks — schema unchanged)

| column | value |
|---|---|
| id | `note:<sha256(content)[:16]>` — content-addressed, idempotent |
| source_type | `"agent_note"` |
| source_ref | `"save_memory"` |
| chunk_kind | `kind` param |
| session_id | = id (column is NOT NULL; notes have no session) |
| content_raw | content |
| distilled | content (the agent already distilled it) |
| embedding | server-side `VllmEmbedder.embed(content)` (document mode) |
| ts_last_active | insert time |
| idf_score | NULL |
| metadata | `{"tags": [...]}`; `{}` when no tags |

Insert uses `ON CONFLICT (id) DO NOTHING` — re-saving identical content is a
no-op reported via `stored: False`. Schema creation reuses
`memory_base.ingest.history._ensure_schema` (import; do not duplicate DDL).

## TDD

**Step A (red)**: `tests/test_save_memory.py`
- Pure part (no DB/embedder): `build_note_row(content, kind, tags, now) -> dict`
  in `serve/mcp_server.py` — id scheme (same content → same id, different →
  different), validation errors (empty content, oversized content, unknown
  kind), row shape (keys above minus embedding), tags land in metadata.
- Integration (`integration` marker): call the tool function against the real
  DB + embedder — row lands in `memory_chunks`; a duplicate save returns
  `stored: False`; `search(..., source="history")` finds the stored note.
- Red via ImportError: `from memory_base.serve.mcp_server import build_note_row,
  save_memory`.

**Step B (green)**: implement `build_note_row` + the tool + the insert path.

## Acceptance criteria

1. Red → green; existing tests unmodified; full suite + ruff clean.
2. Tool visible over SSE from the rebuilt Docker container and callable with a
   real MCP client (orchestrator verifies).
3. A stored note is retrievable via `search_history`.
4. No changes to retrieval logic, ingest gates, or the DB schema.

## Forbidden

New dependencies. LLM calls in the write path. Schema migrations. Changes to
`retrieval/search.py`, `ingest/*`, `common.py`.
