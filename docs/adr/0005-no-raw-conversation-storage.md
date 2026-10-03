# ADR-0005: memory-base stores no raw conversation turns

## Status

Accepted

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
- There is no capture hook and no conversation endpoint; the Claude Code and Hermes
  integrations only prefetch notes into a session.
- A note id is `note:{namespace}:{sha256(content)[:16]}`, taken over the stripped
  content alone.
