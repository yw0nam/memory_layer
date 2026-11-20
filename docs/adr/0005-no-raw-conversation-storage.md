# ADR-0005: memory-base stores no raw conversation turns

## Status

Superseded by [ADR-0017](0017-the-session-record-lane.md) only in its decision to store no raw
conversation turns; its note id rule and the session-local tail file below stay in force. The
content gate's judge prompt was superseded by ADR-0007.

The Claude Code plugin keeps the last 1500 characters of the previous assistant message in a
session-local file of the plugin data directory, overwritten each turn and removed on clear and
compact. Nothing of it is sent to the server.

## Context

A stored session can only help through a link from a note to it, so it cannot recover a
note that was never written. Evidence of past work is already carried by handoff
messages (ADR-0002), git history, and the tracker. No read path consults raw turns: the
agent-facing tools search notes and read messages by address. Raw turns of a personal
chat are a privacy surface that has no reader.

## Decision

memory-base stores no raw conversation turns. Distilled notes and addressed handoff
messages are the only things that cross sessions.

## Consequences

- A fact the writer missed is a writer problem fixed at write time: the `save_memory`
  instructions and the content gate's judge prompt (ADR-0006), not a stored session to
  mine later.
- There is no capture hook and no conversation endpoint; the Claude Code plugin prints
  handoff headers at session start.
- A note id is `note:{namespace}:{sha256(content)[:16]}`, taken over the stripped
  content alone.
