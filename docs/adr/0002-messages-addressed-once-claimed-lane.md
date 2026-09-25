# ADR-0002: Messages are a non-retrieval lane, addressed and claimed once

## Status

Accepted

## Context

The note content gate refuses progress reports, tracker-artefact restatements, and
file descriptions — correctly, because embedding them diluted retrieval. But the gate
only says *not here*. Two kinds of genuinely useful traffic had nowhere to go: a
one-time signal for another session ("the staging DB was re-seeded, re-run anything
that cached row counts"), and the state a session wants to leave for whoever continues
its work in the same scope. Similarity search is the wrong read path for both — "what
was I in the middle of" is answered by address, not by ranking a new prompt against
stored text.

## Decision

Agent-authored messages live in a separate `messages` table on the same pattern as
`doc_rows` (ADR-0001): written without an embedding call, never content-gated, and
invisible to every search path. The write side renders a validated field set into
canonical Markdown (`# Subject`, `## Status`, `## Result`, optional `## Next`,
`## Verification`, `## References`), blockquoting every line of user-controlled text so
nothing escapes the skeleton, and rejects instead of truncating past 4 KiB.

- The namespace is the access boundary; a message names its purpose (`message` for a
  namespace-wide signal, `handoff` when it carries a portable scope:
  `repo:<normalized-origin>` or `project:<organization>/<project>`).
- A subject is stored with its normalized key (NFKC, trim, whitespace collapse,
  casefold) so any spelling variant addresses the same snapshot chain. A new handoff
  snapshot atomically terminalizes older undelivered snapshots of the same
  namespace+scope+subject_key.
- Delivery is at-most-once: a claim is a single conditional UPDATE on the lifecycle
  timestamps, so the database decides the winner by commit order. There is no lease,
  ack, or re-read. Cancel and expiry are the other terminal transitions, and the
  admin purge deletes delivered, cancelled, superseded, and expired rows — no
  scheduler and no new sweep.
- Responses expose the report status (`info`, `in_progress`, `blocked`, `completed`)
  and never the lifecycle timestamps or key identities; a 200 claim or cancel response
  itself proves the transition.

Selective storage is unchanged: a message is a short, deliberately authored, distilled
signal — not transcript capture — and the note gate's refusal is what defines the lane.

## Consequences

- The REST surface grows by `/messages` routes and the MCP server by four thin proxy
  tools; retrieval code is untouched.
- The `messages` table must never be granted to the SQL query role; the grant list
  stays part of the trust boundary.
- Namespace deletion counts messages as content, so a namespace cannot be dropped out
  from under undelivered signals.
- A message cannot carry durable knowledge: it expires, and the MCP instructions draw
  the line explicitly.
