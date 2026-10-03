# ADR-0004: Two save tools, one judge prompt per kind

## Status

Superseded by ADR-0006

## Context

One content-gate prompt served two jobs: memory of the user and their life, which wants a
generous bar, and work knowledge that code, version control, and the tracker cannot
answer, which wants a strict one. A single prompt compromised both. Server-side
distillation, which carried one prompt per job, is gone, so nothing else separates the
jobs. Some clients truncate server instructions while a tool description travels with the
tool, and leaving the kind to the agent produced inconsistent labels.

## Decision

Two MCP tools save notes: `save_personal_memory` and `save_work_memory`. Each has its own
judge prompt and states its own bar in its description. The tool fixes the stored kind
(`personal` or `work`), which is the record of which tool saved the note; the tools have
no kind parameter. The REST route stays one route, `POST /save_memory`, with a required
`kind`.

- The note id does not include the kind. Identical content saved through the other tool is
  a no-op that keeps the first kind, once the chosen tool's validation and gate pass.
- The similar-note check, `allow_similar`, and supersede stay namespace-scoped across
  kinds; a supersede may name a note of either kind.
- Existing rows are converted once at deploy by SQL: namespace `personal` or kind
  `episode` became `personal`, everything else `work`.
- Handoffs and progress stay on `send_message`.

## Consequences

- Per-job criteria live in the tool descriptions, not in the shared server instructions.
- A misrouted save is usually caught by the other gate, not guaranteed: the gate fails
  open, and both prompts accept a communication preference the user states — the
  personal prompt as a preference of the user, the work prompt as a working convention.
- A refusal carries the kind's recovery; a refused note may be rewritten once in total
  across both tools, and a note refused as a mix is split between them.
- The benchmark judges and loads every extracted unit as `personal`.
- The two replay fixtures pin one prompt each: coding-agent notes replay through `work`,
  conversation-memory notes through `personal`.
