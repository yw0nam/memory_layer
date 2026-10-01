# ADR-0003: The content gate judges conversation memory, not only coding notes

## Status

Superseded by ADR-0004

## Context

`save_memory` refuses low-signal notes before they cost an embedding call, and its judge
prompt tested provenance as if every note came from a coding agent: accept a conclusion
nothing else states, a stated constraint, an observed environment fact, a lesson from a
failure. Memory of personal conversations fails that test — a sister's name, a camera
recommendation, a half-marathon finish read as generic or bound to their moment — yet the
product holds memory of every conversation, coding and personal alike, and a future
conversation that loses these facts must ask the user again.

## Decision

One judge prompt with two explicit lists governs the gate. A note is accepted when a
future conversation would otherwise have to ask again and it records a durable fact about
the user or the people, places, and things around them; what the user has, uses, does
regularly, likes, dislikes, or plans, with dates when stated; a dated event the user took
part in and its outcome (kind `episode`); a decision with its reason or the alternatives
it ruled out (kind `decision`); a specific answer the assistant gave that the user may
ask for again; or a constraint, preference, environment fact, or lesson from a failure
stated by a person — including how the user's systems behave in use, stated by no record.
A note is refused when it reports what a record held elsewhere says — version control,
the tracker, the filesystem, the running system — its contents, scope, changes, or
status; progress, status, or a narration of what was done in a coding session; a
description of what a file or function does; generic advice true of anyone; or greetings,
filler, a restated question. An episode is judged only on provenance: was the event lived
by a person rather than recorded by a tracker or version control.

The save path around the gate is unchanged: the credential scan, the similar-note check,
fail-open on judge failure with `content_gate: "unavailable"`, and the output contract
(`accepted`, one-sentence `reason`). Two labelled note fixtures replayed against the live
gate — coding-agent notes and conversation-memory notes — pin the verdicts.

## Consequences

- Personal and coding memory share one acceptance rule; the benchmark harness measures
  this same production prompt on the notes it extracts, and its `--gate off` run records
  notes unjudged.
- Tracker copies, progress reports, and file descriptions stay refused; the messages lane
  (ADR-0002) remains the carrier for operational state the gate refuses.
- Verdicts vary between runs on a live model; the fixtures define the contract, and a
  note that flaps marks a prompt boundary to resolve, not a label to flip.
