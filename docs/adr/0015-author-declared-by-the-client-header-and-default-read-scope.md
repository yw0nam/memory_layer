# ADR-0015: The author is declared by the client header and is the default read scope of notes

## Status

Accepted; supersedes the sentence "Nothing in the store is hidden from another agent" of
[ADR-0014](0014-one-kind-of-key-the-server-grants-nothing-beyond-authentication.md)

## Context

Several agents share one store, and each agent carries its own persona: two Hermes
profiles, a Claude Code session, a scheduled consolidation agent. The boundary between
personas is the author, the slug a note is attributed to. An `author` argument the model
fills on every write lets a model attribute a note to another agent, and a read that
returns every author's notes delivers one persona's notes into another persona's context
as plain memory.

## Decision

- The author comes from one request header, `X-Author`, set once in each client's
  configuration. The model never fills it. The header is required on every route that
  requires `X-API-Key` — everything except `/health*` — and a missing, malformed, or
  reserved value (`all`) is refused with 400. The check is shape only; the author grants
  nothing beyond attribution (ADR-0014 stands).
- The `author` body field and tool argument are removed from every write; a write body
  that still carries `author` is refused with 400. Every write — saved note, archive,
  entity change, link, message, fact, consolidation verdict, undo — is attributed to the
  header author.
- The default read scope of notes is the notes whose author equals the header author plus
  the notes marked `shared`. It applies to every discovery read of notes: note hits of
  `/search`, `GET /notes`, `GET /admin/notes`, `GET /admin/duplicates`, and filter-based
  archive discovery. It does not apply to entity cards and facts, messages, handoffs,
  turns, document chunks, CSV cards, code search, or any operation addressed by id. A
  read's `author` filter is three-way: omitted is the default scope, `all` is every
  author's notes, a slug is exactly that author's notes.
- `save_memory` takes a boolean `shared`. A shared note stays attributed to its writer and
  is in every author's default scope. The flag is set at creation only.
- A note's id hashes the author beside the content, so two authors saving the same text
  store two notes. The near-duplicate check runs over the writer's default scope in the
  namespace, so another author's shared note is still refused as a near-duplicate and
  another author's private note is invisible and accepted.
- Consolidation runs per persona: each agent judges groups drawn from its own author.
  Groups never mix authors, a merged note carries the members' author, and the action and
  every record it writes carry the header author.

## Consequences

- A note outside a client's default scope is reached with `author: "all"`. That a read
  without a namespace filter covers every namespace is unchanged.
- Connecting an agent is giving it a key and an author slug, and configuring the client to
  send both on every request. No persona receives another persona's notes unless they are
  shared or the read names their author.
- The same text from two authors is two notes, and the near-duplicate refusal crosses the
  persona boundary only through shared notes.
- Rendering marks foreign authors: a prefetched note line shows its author when another
  agent wrote it, and stays plain for the reader's own notes.
