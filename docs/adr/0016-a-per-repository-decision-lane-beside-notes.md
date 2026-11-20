# ADR-0016: A per-repository decision lane beside notes

## Status

Accepted; supersedes the clause of [ADR-0010](0010-save-policy-a-minimal-default.md) that
a work decision keeps its reason as a note.

## Context

Decisions the owner makes while working with an agent were saved as notes. Decision rows
extracted into the note pool were outranked by broad summaries, and the fix that filtered
note search by kind and tags was later removed. A team repository cannot hold one person's
decisions as its issues, and repository documents such as `AGENTS.md` and `CLAUDE.md` stay
neutral of any memory layer, so the memory rule cannot be written into them.

## Decision

- Decisions the owner makes with the agent are stored in their own table, `decisions`, and
  read by their own routes and tools. Note search and note ranking never read the table.
- A decision's key is its repository scope (`repo:<origin>`, normalized as a handoff scope
  is) and its author, the client's `X-Author` slug. The number `D-N` is per repository and
  author: the next number, assigned under a transaction-scoped advisory lock.
- A decision has fixed fields: `title` (at most 200 chars), `decision`, an optional `why`
  where rejected alternatives go, `links` (at most 10 strings of at most 500 chars: issue,
  PR, ADR, or session URLs), and `decided_on`. `decision` and `why` together are at most
  1500 chars.
- A decision is recorded, superseded, or retired only when the owner asks. The server does
  not judge when to write; the tool descriptions and the server instructions state the
  rule. A supersede records a new decision that names an active one of the same scope and
  author; the replaced one leaves lists and recall and stays as history. A retire takes a
  reason. Nothing is deleted.
- `/recall` scores the notes and the repository's active decisions separately. Each lane
  has its own floor (`min_score`, default 0.5) and the two lanes run concurrently. A lane
  that fails is named in `errors` and the other lane still answers.

## Consequences

- The author is the client's persona slug. Changing a plugin's author orphans its
  decisions: the rows stay under the old slug, and the new slug reads none of them.
- Other agents cannot read them. Reads are scoped to the caller's author, and the `all`
  author option of note reads does not reach decisions.
- Key rotation has no effect on them: a key grants nothing beyond authentication
  ([ADR-0014](0014-one-kind-of-key-the-server-grants-nothing-beyond-authentication.md)),
  so a new key with the same author reads the same decisions.
- A decision is not a note: it does not appear in `/search` hits or in `GET /notes`.
