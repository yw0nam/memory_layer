# Implementation spec: agent-history selection core

## Scope

`src/memory_base/ingest/history.py` is the source-agnostic selection core that turns a
`Session` (see `src/memory_base/adapters/base.py`) into `memory_chunks` row dicts. It has no
I/O entrypoint of its own and issues no DB writes — a future batch corpus adapter would call
into it, and today's row-level content instead reaches the DB through the MCP `save_memory`
tool (`docs/specs/impl_mcp_only_agents.md`).

## Pipeline

1. **Grouping**: `group_sessions` buckets messages by `session_id` into `Session` objects;
   `build_transcript` reconstructs the "ROLE: text" transcript, truncating to the head 60% +
   `TRUNCATION_MARKER` + tail 40% past 100k chars.
2. **Triage**: `triage_heuristic(session)` classifies a session as `"skip"` (no assistant
   messages, or too few/short user messages), `"keep"` (large enough by char/message count),
   or `"borderline"`.
3. **Distillation**: `_distill(session, semaphore)` calls the LLM (JSON mode) to produce a
   `Distillation` — `one_line_question`, `summary`, `resolution`, `references`, `decisions`
   (capped at 10) — in English regardless of the source transcript's language.
   `parse_distillation` validates and parses the raw JSON.
4. **Bursting**: `group_bursts(messages, min_chars=200)` groups consecutive same-role
   messages into `Burst`s, applying the length filter and computing `social_weight` (1.5 when
   the burst has a tool error or is followed by a same-user re-ask within 180s, else 1.0).
5. **Burst gate**: `mean_idf(text, n, dfs)` scores a burst against corpus document
   frequencies (`build_corpus_df`); `passes_burst_gate` accepts unconditionally when the
   corpus has fewer than 20 documents, otherwise requires
   `burst_score(mean_idf, has_social) >= 4.0`.
6. **Row assembly**: `build_rows(session, distillation, adapter, dfs, n)` is a pure function
   (no network/DB) that emits one `session` row, one `decision` row per distilled decision,
   and — when `adapter.emit_bursts` is true — one `burst` row per burst that passed the gate.
   `source_ref` is the session id; `metadata` carries `cwd`, `git_branch`, `tool_names`, and
   `tool_error_count` from the session's last message.

## Contract with retrieval

Rows produced by `build_rows` (plus a computed `embedding`, added by an effectful caller) are
inserted into `memory_chunks`, the single contract with `src/memory_base/retrieval/search.py`.
Adding a future corpus source is a matter of implementing `SourceAdapter` and feeding its
`Session`s through this pipeline — no changes to retrieval or serving code.
