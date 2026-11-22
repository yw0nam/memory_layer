---
name: record-decision
description: "Use when the user says to record a decision, or to supersede or retire one. Drafts each decision, checks it against the repository's active decisions, records it and shows its D-number."
---

# Record a decision

The decision lane holds the choices the owner made while working with you that bind later work
in one repository. They are recorded, superseded and retired only on the owner's word. Asking
for a record is that word, so you do not ask again before saving.

## 1. Find the decisions to record

- **The owner named them:** record those.
- **The owner said only "record":** take each choice the owner settled since the last record
  that binds later work: a design, a rule, a method adopted or rejected. A choice that holds
  only for the moment ("use the GPU host this time") is left out.

You are done when every binding choice of that span is a draft or has been left out for a
stated reason.

## 2. Check the active decisions

Find the scope: the scope line the SessionStart hook printed, or `repo:` followed by the
output of `git remote get-url origin`. Then run `list_decisions` for it, and `get_decision` on
each active decision a draft touches.
- **The draft changes an active decision:** if the owner named that decision as the one to
  replace, the draft supersedes it. If not, show both and ask whether to supersede.
- **An active decision already says the same:** drop the draft.

## 3. Draft each entry

- **title:** one line that names the choice.
- **decision:** the rule as it now stands, readable without this session.
- **why:** the reason, and each alternative that was rejected with the reason it lost.
- **links:** the issue, PR or ADR the decision came from or produced. If the repository's own
  instructions record decisions elsewhere too (a decision issue, an ADR), follow them as well,
  and link that record here.
- **decided_on:** the owner's local date of the decision.

## 4. Record and show

Run `record_decision` for each draft, in the order the decisions were made. Then show the
owner each D-number with its title and text, and the D-number each new decision superseded.

## Retire a decision

When the owner asks to retire D-N, read it with `get_decision`, run `retire_decision` with the
owner's reason, and show the decision that was retired.
