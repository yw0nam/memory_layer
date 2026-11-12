# ADR-0010: The save policy is a minimal default

## Status

Accepted

## Context

The `save_memory` description listed fourteen kinds of note worth keeping, one by one,
repeated the `kind` label from its own field and the archive step from the server
instructions, and stood close to the 2048 characters a client shows. A list of kinds reads
as a checklist a writer matches against. One fixed list also cannot fit every agent: a
companion agent keeps the moments it shares with the user, and a coding agent keeps none of
the user's private life.

## Decision

- The description holds only the points a save decision rests on: keep a note when the
  user would otherwise be asked again, when it is something the agent gave them that they
  may want again, or when code, version control, the tracker, and documents cannot answer
  it; a work decision keeps its reason; session progress, copies of what a PR, issue,
  commit, or file says, generic advice, and filler are not saved; the note's form; search
  before saving and supersede a changed value.
- The server enforces one of these rules: a note carrying a credential is refused. The
  rest is a default the writer applies.

## Consequences

- The description is about 900 characters; the `kind` label is explained in its field and
  the archive step in the server instructions.
- The description is guidance, not enforcement: a writer can ignore it, and only the
  credential refusal holds regardless of the writer.
