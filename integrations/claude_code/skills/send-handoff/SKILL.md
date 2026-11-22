---
name: send-handoff
description: "Use when the user asks for a handoff: pausing work, passing it to another agent, or before /compact with work unfinished. Sends the repository's work state and next steps, and withdraws your stale handoffs."
---

# Send a handoff

A handoff is the latest snapshot of unfinished work in one repository. The next session or
agent working there claims it once and resumes from it. The handoff is complete when that
reader could resume without this session's transcript.

## 1. Find the scope

Use the scope line the SessionStart hook printed. Without one, use `repo:` followed by the
output of `git remote get-url origin`.

## 2. Write down the state

- **Done:** what was finished, with the paths, ids, branches, worktrees and PRs a reader
  needs. Name anything left uncommitted.
- **Traps:** each trap found that the next step will hit.
- **Open questions:** each question the owner still has to answer.
- **Next:** the next steps, numbered, in order.

A lesson that matters beyond this work belongs in a note (`save-notes`). A decision the owner
made goes into the report as a draft, as step 5 describes.

## 3. Send it

`send_message` with:
- the scope;
- `status`: `in_progress` or `blocked`, or `completed` when nothing is left to do;
- a `subject` that names the work and its state;
- the state from step 2 in `result`, with the next steps in `next_step`;
- the PR and issue URLs in `refs`.

## 4. Withdraw your stale handoffs

A new handoff supersedes only a pending one with the same subject. Run `list_messages` for the
scope, and `cancel_message` each pending handoff of yours that the new one makes stale.
- Leave handoffs from other authors alone.
- A handoff another session has already claimed cannot be cancelled
  (`no cancellable message`). That is expected; skip it.

## 5. Report

- The subject and id of the new handoff.
- Each handoff you cancelled, and each one from another author that is still pending.
- **Decision drafts:** each decision the owner made in the session that the decision lane
  does not hold, with a title, the decision, the reason and the rejected alternatives. They
  are recorded only when the owner says to record them (`record-decision`).
