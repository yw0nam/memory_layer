# ADR-0007: Notes are stored without an LLM content gate

## Status

Accepted

## Context

A chat model judged every note before it was embedded and refused low signal. Measured on
LongMemEval sessions replayed through the real MCP tools, the gate did not earn its cost:

- With the gate recording verdicts without refusing, a weaker writer stored 1.7 times the
  notes of a stronger one, and the gate flagged about 3% of the notes of either writer.
  Blind labels put its recall of low-signal notes at 12% to 22%.
- The notes it missed most often were recaps of a general topic with an inferred interest,
  the same shape as the recommendations and lists its prompt accepts by name, so a prompt
  tuned on them would fit the sample.
- Most of its refusals of the stronger writer's notes were notes the labellers kept.
- At the read settings the clients use, the rerank score floor kept unrelated notes from
  the agent with or without the gate: prefetch delivered no junk either way, and with no
  floor the notes the gate would refuse made up 2% of the junk hits.

The gate also cost a chat call and its latency on every save, failed open whenever the
provider did not answer, and sent a refused writer into a rewrite loop.

## Decision

`save_note` stores a note that passes the deterministic checks — validation, the
credential scan, the near-duplicate refusal with `allow_similar`, and supersede — with no
chat-model call. The client agent decides what is worth keeping, guided by the
`save_memory` description and the server instructions: write rarely, search before
saving, supersede a changed value, and the list of what not to save.

The chat model is called in one place: the CSV branch, to summarize a sampled table into
one card.

This supersedes the judge prompt and the refusal recovery of ADR-0006. Its other decisions
stand: one `save_memory` tool, the kind as a label, parameter notes in the input schema.

## Consequences

- A save costs one embedding call and no chat call, and works with no chat provider.
- A careless writer can store more low-signal notes than before. The rerank floor keeps
  unrelated notes out of what is delivered; duplicates and stale values are curated with
  `list_memory_duplicates`, supersede, and `archive_notes`.
- The server no longer claims that low-signal content never reaches the embedding path.
- A refusal now always names a deterministic cause, so a writer never rewrites a note to
  pass a judge.
