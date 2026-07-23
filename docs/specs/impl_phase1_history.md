# Implementation spec: Phase 1 — Agent history ingest & distillation pipeline

## Goal

Implement an **incremental & idempotent** pipeline `src/history_index.py` that parses
Claude Code session JSONL (`~/.claude/projects/<encoded-path>/<session-id>.jsonl`),
distills it with an LLM, and loads it into the `memory.memory_chunks` table together
with embeddings.

## Context (existing code — do not modify, read and conform)

- Project root: `/home/spow12/codes/2026_upper/agents/memory/memory_layer/repos/memory_base`
- Python 3.12, uv project. Installed deps: asyncpg, openai (AsyncOpenAI), numpy, python-dotenv, httpx, cocoindex (unrelated)
- `src/common.py`: `VllmEmbedder().embed(text, query=False) -> np.float16[2048]` (async),
  `llm_client() -> AsyncOpenAI`, `LLM_MODEL`, `DB_URL`, `PG_SCHEMA = "memory"`. Must be reused.
- DB: Postgres (port 5439, docker `memory_base_db`), pgvector 0.8.5, schema `memory` exists.
- The `memory.code_chunks` table is managed by CocoIndex — **never touch it**.
- `src/search.py`'s `_search_history()` already SELECTs the following columns (contract):
  `id, source_ref, distilled, content_raw, ts_last_active, idf_score, embedding(halfvec)`.
  Do not break this contract. Run: `cd src && uv run python search.py "query" --source history`

## Table (owned & created by this pipeline, IF NOT EXISTS)

```sql
CREATE TABLE IF NOT EXISTS memory.memory_chunks (
  id             text PRIMARY KEY,          -- "{session_id}:session" or "{session_id}:burst:{n}"
  source_type    text NOT NULL,             -- 'claude_code' (Hermes etc. via a later adapter)
  source_ref     text NOT NULL,             -- "{project_dir_name}/{session_id}" (human-traceable)
  chunk_kind     text NOT NULL,             -- 'session' | 'burst'
  session_id     text NOT NULL,
  content_raw    text NOT NULL,             -- raw text for FTS (session: reconstructed transcript; burst: burst raw text)
  distilled      text,                      -- embedded text (session: distillation result; burst: topic prefix + burst)
  embedding      halfvec(2048) NOT NULL,
  ts_last_active double precision NOT NULL, -- epoch sec of the session's last message
  idf_score      double precision,          -- burst mean-IDF (NULL allowed for session rows)
  metadata       jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS memory_chunks__fts ON memory.memory_chunks
  USING GIN (to_tsvector('simple', content_raw));
CREATE INDEX IF NOT EXISTS memory_chunks__vec ON memory.memory_chunks
  USING hnsw (embedding halfvec_cosine_ops);
CREATE INDEX IF NOT EXISTS memory_chunks__session ON memory.memory_chunks (session_id);
-- incremental state tracking
CREATE TABLE IF NOT EXISTS memory.ingest_state (
  file_path text PRIMARY KEY,
  mtime     double precision NOT NULL,
  size      bigint NOT NULL,
  ingested_at double precision NOT NULL
);
-- corpus token DF (for IDF computation)
CREATE TABLE IF NOT EXISTS memory.df_stats (
  token text PRIMARY KEY,
  df    bigint NOT NULL
);
-- Store the total document count in df_stats under the special token '__N__', or a separate single-row table. Implementer's discretion.
```

## JSONL parsing rules (real format confirmed)

One line = one JSON object. Based on the `type` field:

- **Handled**: only `type in ("user", "assistant")`. Everything else (`attachment, ai-title, mode,
  system, file-history-snapshot, queue-operation, bridge-session, ...`) is skipped.
- Lines with `isSidechain == true` are skipped (sub-agent traffic).
- Common fields: `timestamp` (ISO8601, "2026-06-25T05:08:12.347Z"), `sessionId`, `cwd`, `gitBranch`, `uuid`, `parentUuid`.
- **user**: if `message.content` is a str, use it as text directly. If a list, it's a block array —
  take only the text of `{"type":"text"}` blocks; do not include `{"type":"tool_result"}` blocks as
  text, but aggregate their `is_error` field as a tool-failure signal.
- **assistant**: `message.content` is a block list — concatenate only the text of `{"type":"text"}`.
  Skip `{"type":"thinking"}`. Do not include `{"type":"tool_use"}` as text, but aggregate the tool name (name).
- Empty-text messages are discarded.
- A file may contain corrupt lines (unparseable JSON) — skip only that line.

## Session reconstruction → row generation

**Session = thread.** Per session:

1. **One session row**: `content_raw` = a reconstructed transcript in the form "USER: ...\nASSISTANT: ..."
   (if over 100k chars, truncate to head 60% + "\n...[truncated]...\n" + tail 40%).
   `distilled` = the LLM distillation result (below); the embedding is computed over distilled.
2. **0–N burst rows**: only groups of consecutive same-speaker text messages (bursts) with
   **combined length ≥ 200 chars AND mean-IDF ≥ 4.0**.
   `distilled` = `"[{one-line session topic}] {burst raw text}"` (the topic reuses one_line_question from the distillation result);
   the embedding is computed over this distilled. `content_raw` = burst raw text.
   - **Social weight (spec §3.2)**: if the burst span has a tool error (is_error) or an immediately following user re-ask
     (same user speaks again within 3 minutes), `metadata.social_weight = 1.5`, otherwise 1.0.
     Do not multiply idf_score by it; record it only in metadata (for later use on the search side).

Sessions with fewer than 5 messages or under 500 total text chars are skipped (noise).

## LLM distillation

- Use `common.llm_client()` + `LLM_MODEL`, JSON mode (response_format={"type":"json_object"}).
- Put the (truncated) transcript in the prompt and extract the following JSON:
  `{"one_line_question": "...", "summary": "...", "resolution": "...", "references": ["mentioned files/systems/commands"]}`
  - one_line_question: "a one-line search question you'd likely ask to find this session later"
  - summary: a 3–5 sentence summary; resolution: the final solution/conclusion (if none, "unresolved")
  - Output language: Korean (code, error strings, and identifiers appearing in the original stay verbatim).
- The session row's `distilled` = one_line_question + "\n" + summary + "\n" + resolution + "\n" + ", ".join(references)
- On LLM call failure (timeout/refusal): do not leave that session with distilled=None; instead **skip it and log a warning**
  (do not record it in ingest_state, so it retries on the next run).

## IDF

- Tokenization: a simple tokenizer at the level of `re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}|[\uAC00-\uD7A3]{2,}", text.lower())` (the Hangul range written as unicode escapes).
- DF update: count as session-unit documents (unique token set of a session → df += 1). N = total number of sessions.
- `idf(t) = ln((N+1)/(df(t)+1)) + 1`. Burst mean-IDF = mean idf of the burst's unique tokens.
- Ignore the bootstrap problem (small initial N) — if N < 20, let the IDF filter pass (apply only the length condition).

## Incremental & idempotent

- Target files: `~/.claude/projects/*/*.jsonl` (~145 files). After glob, compare mtime+size against `ingest_state`,
  process only changed files. `--full` reprocesses everything.
- **Active-session protection**: skip files whose mtime is within the last 10 minutes (still being written).
- On session reprocessing: within a transaction, `DELETE FROM memory_chunks WHERE session_id=$1` then re-insert.
- Sequential embedding calls per session are sufficient (PoC). LLM distillation runs with a semaphore of up to 4 concurrent.

## CLI

```
uv run python src/history_index.py [--limit N] [--project SUBSTR] [--full] [--dry-run]
```
- `--limit N`: only N files by most recent mtime (for testing)
- `--project SUBSTR`: substring-match filter on the project directory name
- `--dry-run`: no DB writes, print only parse/filter statistics
- On exit, print a summary: files processed, session/burst rows created, per-reason skip counts.

## Tests (required deliverable)

`tests/test_history_parse.py` — must run **without** DB/LLM/embeddings:
- Unit-verify the parser with inline fixtures (a few lines of JSONL strings in the format above):
  sidechain skip, thinking skip, tool_result text exclusion + is_error aggregation, transcript reconstruction.
- Verify burst grouping and the 200-char filter.
- Run: `uv run pytest tests/test_history_parse.py` (pytest is added via `uv add --dev pytest`).
- To enable this, split the parsing/bursting logic into pure functions with no I/O.

## Acceptance criteria

1. `uv run pytest tests/test_history_parse.py` passes.
2. `uv run python src/history_index.py --limit 5` succeeds, creating rows in memory_chunks.
3. `cd src && uv run python search.py "any relevant query" --source history` returns results (without error).
4. Re-running the same command does not reprocess unchanged files (incremental behavior, verifiable in logs).

## Forbidden

- Do not modify `src/common.py`, `src/search.py`, `src/code_index.py`.
- Do not add new heavy dependencies (only the pytest dev dependency is allowed). Implement with the standard library + already-installed packages.
- Do not access the memory.code_chunks table.
