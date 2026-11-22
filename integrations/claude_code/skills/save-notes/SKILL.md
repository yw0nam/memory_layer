---
name: save-notes
description: "Use when the user asks to save what this session learned to memory notes, or is about to /compact. Picks the lessons a later session needs, checks the notes already held, and saves or supersedes them."
---

# Save a session's lessons as notes

A note holds a **lesson**: knowledge a later session needs that code, version control, the
tracker and documents cannot answer. Each note is read again whenever a later session's work
matches it, in any repository.

## 1. Gather the lessons

Read back over the whole session, not only its last turns. List each method, tool or gotcha
that worked or failed, with its measurement, and each preference or correction the user
stated. You are done when every subject the session worked on is on the list or has been
ruled out.

Some candidates go elsewhere:
- **Decision:** a choice the owner made that binds later work in a repository (a design, a
  rule, a method adopted or rejected). Keep it out of the note and list it as a decision
  draft in the report. A lesson that carries such a choice is split: the choice goes to the
  draft, and the how-to stays in the note.
- **Progress:** where unfinished work stands. It belongs to a handoff (`send-handoff`).
- **Already recorded:** a PR, issue, commit, file or active note already states it. Drop it.
- **One-off:** a choice that holds only for the moment ("run it on the GPU host this
  time"). Drop it.

## 2. Check the notes already held

Run `search_memory` once per lesson subject, in parallel.
- A hit that states an older value: the new note supersedes it.
- A hit that already says the same thing: drop the lesson.

## 3. Write the notes

- **One subject per note.**
- **First line:** the subject, the date and the source (repository, issue, set). A
  superseding note says in its first line what changed from the note it replaces.
- **Body:** bullets that keep the measurements and file paths. A failed method keeps the
  reason it failed.
- **Refusals:** on a length refusal, split the note by subject and send each part again. On
  any other refusal, do what its message says. If a permission check denies a save (a prompt
  or the auto-mode classifier), show the note's full text in the report, so nothing is lost.

## 4. Report

- **New notes** and **superseded notes**, one line each. For each superseded note, say what
  changed.
- **Decision drafts:** each with a title, the decision, the reason and the rejected
  alternatives. They are recorded only when the owner says to record them (`record-decision`).
- **Dropped:** each dropped candidate, and where it is already held.
