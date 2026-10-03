# ADR-0008: Consolidation is judged by an agent and applied by the server

## Status

Accepted

## Context

Active notes pile up restatements of one fact or rule, and values that a newer note
replaced. The write-time near-duplicate refusal (cosine 0.85) misses most of them:
differently worded statements of one rule pair at 0.72 to 0.79. To decide whether notes
state the same thing, which one to keep, and how to word a merge is a judgement about
meaning. ADR-0007 gives that judgement to agents and keeps the server to deterministic
checks. A merge also replaces wording that a reader can need later, so each change must
be recorded and reversible.

## Decision

The server finds groups and applies changes; an agent judges each group.

- `GET /admin/consolidate/groups` issues groups: exact nearest neighbours over the stored
  embeddings, deterministic clique packing, and a key over the procedure version and every
  prompt-visible field of every member.
- `POST /admin/consolidate/verdicts` accepts `keep`, `retire`, or `merge` only for a group
  the server issues now with the same group parameters: the key and the member ids must
  equal those of a current group. Any other verdict is `stale` and returns the current
  groups that share a member.
- Validation is deterministic. A retire leaves at least one member. A merge text passes the
  note validation, the credential scan, and a two-direction token check over numbers and
  dates, backticked spans, and capitalized names; the check is a conservative filter, not
  proof of meaning. A merge text equal to a member after whitespace normalization is
  applied as a retire into that member. A merge text equal to an active note reuses that
  note; one equal to an archived note is refused.
- Each verdict is applied alone, in one transaction under a per-namespace advisory lock,
  with the touched rows locked, after the server plans it again on the current rows. A plan
  that changed since the preflight is `stale`. An embedder or database error rolls back
  only its verdict, which is reported `failed` and may be retried with the same idempotency
  key. Retire and merge actions per run and namespace are capped by the request.
- Every accepted verdict, keep included, is a row in `consolidation_actions`. Its
  idempotency key makes a retry a duplicate. Its group key is the verdict cache: a judged
  group is not issued again until a member changes or the procedure version changes. The
  agent and model that judged are recorded, not keyed.
- Lineage is written on both sides. Archived members carry `consolidated_into`; a
  replacement carries `merged_from` and `merged_dates`; an agent's supersede writes
  `replaced_by` on the replaced note.
- `POST /admin/consolidate/undo` reverses one action. It refuses with 409 and changes
  nothing when a later change touched what it would reverse: a changed archived note, or a
  replacement that is archived, superseded by an active note, or a member of a later retire
  or merge. A later keep does not block it. An undone group stays out of
  the issued groups.
- The server calls no chat model for consolidation.

## Consequences

- A run cannot change a note that the server did not issue in the judged state, and cannot
  go past its action cap.
- A merge is only as good as the agent's wording. The token check refuses a dropped or
  added number, date, backticked identifier, or mid-sentence name, but passes a changed
  sentence-initial name, a negation, names in scripts without case, and a changed version
  inside an identifier such as `v2.0`.
- Consolidation never deletes a note, so `GET /admin/consolidate/actions` shows the full
  history with full text.
- An applied verdict runs the namespace's exact neighbour search twice, so its cost grows
  with the number of active notes in the namespace.
- A change to the grouping procedure or the validator policy bumps `PROCEDURE_VERSION`,
  which issues every judged group again; a group whose action was undone stays out.
