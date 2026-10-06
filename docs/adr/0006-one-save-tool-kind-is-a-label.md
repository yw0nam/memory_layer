# ADR-0006: One save tool; the kind is a label, the gate refuses low signal only

## Status

Accepted; the judge prompt and the refusal recovery are superseded by ADR-0007, and what
the tool description states by ADR-0010

## Context

Two kind-fixed save tools each excluded the other's domain, in their descriptions and in
their gates. That left a hole between them: a result the assistant built for the user's
work from facts stated only in the conversation (a team roster with names and shifts) is
neither memory of the user's life nor a decision with its reason, and a writer agent with
both tools saved it with neither. Routing by gate was not reliable in the other direction
either: the gate fails open, both prompts accepted a communication preference, and a
companion agent's personal facts sent to the work tool passed the work gate. An overlap
between the kinds costs little, because prefetch and search do not filter by kind unless
asked; a hole loses the memory. The two prompts' refusal lists were the same apart from the
cross-domain and mixed-note refusals.

## Decision

One MCP tool, `save_memory`, saves notes. Its `kind` argument is required and is
`personal` or `work`; it labels the note for search and listing and never decides whether
the note is stored. The REST route stays `POST /save_memory` with a required `kind`.

- One judge prompt judges every note. It accepts memory of the user and knowledge of their
  work, and refuses low signal: session narration, progress and next steps, a copy of what
  a record held elsewhere says, a description of a file or function, generic advice, and
  greetings or filler. It never refuses a note for its domain or for mixing the two.
  Something built in the conversation from facts no record holds is not a copy.
- The tool description states both bars, including that a decision without its reason is
  not worth keeping; the gate does not enforce that bar.
- A refusal carries the reason and one recovery: rewrite once, or carry progress on
  `send_message`; it never points to another save tool.
- Parameter notes live in the input schema, so the description fits the 2048 characters a
  client shows of it, and the tool's first sentence fits a 60-character catalog summary.
- The note id does not include the kind; identical content saved with the other kind is a
  no-op that keeps the first kind. The similar-note check, `allow_similar`, and supersede
  stay namespace-scoped across kinds.

## Consequences

- A writer never has to decide which tool owns a note before saving it; it only labels it.
- A mislabelled note is still found by a search that does not filter by kind.
- The work bar (a decision with its reason, a lesson with its failure) is guidance in the
  description, so a careless writer can store a weaker work note than before.
- Both gate replay fixtures run through the one prompt.
