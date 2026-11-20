# ADR-0017: The session record lane keeps Claude Code transcripts verbatim, read on demand

## Status

Accepted; supersedes [ADR-0005](0005-no-raw-conversation-storage.md).

## Context

The owner used a session-history tool that summarised every session with a model, which cost
quota on each summary. The owner wanted earlier sessions readable from any machine, not only the
one that ran them. Measured on one machine over 30 days: 307 main sessions, 19 of which used the
old tool's history search, and its tool-call history was read 5 times. The verbatim lane that
ADR-0005 removed went away when the server was narrowed, and it is not restored as it was.

## Decision

- Claude Code main-session transcripts are stored verbatim in `session_events`, per author: user
  text, assistant text, tool calls, and tool results. Tool input and tool results are cut at 2,000
  characters; other text at 64,000. A cut value ends with a ` [cut, N chars]` marker giving its original length.
- The plugin's session record uploader posts them. A detached process, started by the Stop,
  SessionEnd, and SessionStart hooks, reads the transcript from a per-session checkpoint and
  uploads only complete lines. Each event id is derived from the session, the transcript entry, and
  the block index, so a replay is recognised and stored once.
- No credential check runs on the record. Nothing is embedded and no model summarises it.
- The record is read only on demand, through `list_sessions`, `read_session`, and
  `search_transcript` (REST and MCP). It is never part of `/search` or `/recall` and is never
  prefetched.
- Subagent transcripts are not stored.

## Consequences

- Any valid key with the same author reads the record ([ADR-0014](0014-one-kind-of-key-the-server-grants-nothing-beyond-authentication.md)):
  the author is the only boundary between sessions.
- The record holds whatever was in the session, secrets included. A secret in a session is stored
  as text and is not refused.
- A session is deleted whole through `DELETE /admin/sessions/{session_id}`, which removes the
  caller's rows of that session. The hooks do not undo a delete: lines appended later are stored
  from the checkpoint, so they form a new partial session.
- A replay after a delete stores the session again. A replay starts from a reset checkpoint, such
  as `scripts/backfill_sessions.py` run over a fresh state directory.
- Events are not memory: they never appear in note search, decision recall, or the handoff lane.
