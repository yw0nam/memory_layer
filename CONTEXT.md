# Memory Base

A retrieval layer whose memory is distilled content, kept beside a verbatim record of
the conversations it came from. Agents write memories through a validating tool;
documents and code repositories are ingested through source adapters; the client's hook
posts each message of a conversation as it happens. Every consumer reads back through
one REST API.

## Language

### Stored content

**Chunk**:
The unit of storage and retrieval — one embedded, independently searchable piece of content.
_Avoid_: record, entry, item, document (for the unit)

**Note**:
An agent-authored memory, distilled by the agent before it arrives rather than extracted from a file. A note names the registered entities it is about in `entities`.
_Avoid_: memo, observation

**Kind**:
The label a note carries: `personal` (the user: their life, their day, and moments with the agent) or `work` (knowledge of the work), chosen by the agent when it calls `save_memory`; it narrows search and listing and never decides whether the note is stored. A document chunk's kind is `doc`.
_Avoid_: track, lane, category, type

**Document**:
An uploaded file that is converted and split into chunks. The whole file, never one of its pieces.
_Avoid_: file, doc, upload

**Card**:
A single distilled description of a tabular document — its fields, shape, and the
patterns visible in a sample — standing in for data that would be noise if chunked
directly. Distinct from the chunks of a prose document. The card is also the handle to
the document's table rows: it carries the column list and the document id a SQL query
needs.
_Avoid_: summary, table summary, doc chunk

**Table rows**:
A tabular document's data rows, stored verbatim in `doc_rows` and read exclusively
through the SQL query lane. Never embedded, never returned by search — the card is
found, the rows are computed over.
_Avoid_: table chunks, row chunks, records

**Message**:
An agent-authored, addressed one-time signal, stored as canonical Markdown in `messages`,
listed while pending and consumed by claiming — never embedded, never returned by search. Operational state with
a report status (`info`), distinct from a note's durable knowledge.
_Avoid_: notification, inbox item, chat message

**Handoff**:
A message snapshotting the state of a stretch of work, addressed to a portable scope
(`repo:<origin>` or `project:<organization>/<project>`) instead of a namespace. Carries
a report status (`in_progress`, `blocked`, or `completed`); a new snapshot of the same
subject supersedes the pending one. Has no expiry unless its sender gives one: it stays
pending until claimed, superseded, or cancelled.
_Avoid_: session summary, progress note, status report

**Turn**:
One message of a conversation, the user's or the assistant's, stored verbatim in `turns` as the client delivered it, with the assistant's reasoning when the client exposes it. Read by time, session, and literal substring; never embedded, never returned by search, never prefetched.
_Avoid_: transcript line, utterance, log entry

**Session**:
The conversation a client ran, named by the client's own session id under its origin (`claude_code` or `hermes`); the unit `list_sessions` lists and `read_session` reads in order. A note may name the session it came from as `source_session`.
_Avoid_: conversation (reserved for the lane as a whole), thread, chat

**Claim**:
The at-most-once delivery of a pending message: exactly one claimer wins, and the
loser is refused. Terminal — a claimed message is never delivered again.
_Avoid_: pop, read receipt, acknowledge

**Code chunk**:
A span of source lines from an indexed repository. Retrieved from its own lane, separate from memories.
_Avoid_: snippet, fragment

**Repo**:
A git repository cached and indexed for code retrieval. Managed and re-indexable, unlike a document.
_Avoid_: codebase, project, source

### Organization

**Namespace**:
A registered label that partitions memories for filtering; nothing in it is private.
_Avoid_: tenant, workspace, bucket, collection

**Tag**:
An uploader- or author-supplied label on a chunk, used to narrow retrieval. Not a boundary.
_Avoid_: label, category, topic

**Supersede**:
To replace a note with a newer one, archiving the old rather than deleting it. Superseded notes stay retrievable on request.
_Avoid_: overwrite, replace, delete, revoke

**Group**:
A set of active notes in one namespace, every pair at or above the similarity threshold and none acknowledged as distinct through `similar_ack`, offered to an agent to judge whether they state the same thing. Built by `GET /admin/consolidate/groups`; a note is in at most one group.
_Avoid_: cluster, duplicate set

**Group key**:
A hash over a group's members and each member's prompt-visible fields (text, kind, author, save time, event date, tags, entities, supersedes) and the procedure version; it changes whenever any of them does.
_Avoid_: group id, fingerprint

**Verdict**:
An agent's judgement of one issued group — `keep`, `retire` (with the ids to archive), or `merge` (with the merged text) — with a reason and an idempotency key. The server accepts it only for a group it issues now under the same key and members, and checks it deterministically; it judges no content.
_Avoid_: decision, ruling, review

**Consolidation action**:
The record of one accepted verdict in `consolidation_actions`: what it archived, what survived or replaced the members, the archived notes' metadata before the change, and whether it was undone. Its group key is the verdict cache; undo reverses exactly one action.
_Avoid_: job, operation, change set

**Lineage fields**:
Note metadata that links a note to the notes that replaced or absorbed it: `supersedes` and `replaced_by` from a supersede save, `consolidated_into` on a note a consolidation archived, and `merged_from` and `merged_dates` on a merge's replacement.
_Avoid_: history, provenance, links

**Entity**:
A registered person, agent, project, organisation, place, or thing, keyed by a slug the user consented to. Aliases resolve other spellings of its name; the server infers no equivalence between names and merges nothing.
_Avoid_: node, subject, topic

**Fact**:
One property or relation of an entity, keyed by entity and attribute and distilled from a note by the consolidation run. A `value_entity` instead of a `value_text` makes it a relation. A fact is current while its `archived_at` is null; a replace or remove closes it and a deleted note archives it.
_Avoid_: triple, attribute row

**Card**:
An entity's registry row, the current facts it holds and the facts that point at it, and the notes naming it.
_Avoid_: profile, summary

**Consent**:
The user's answer, inside the conversation, to a change of the entity registry; recorded as the method the client reported: `elicitation`, `conversation`, or `import`.
_Avoid_: approval, permission, authorization

**Distilled**:
Reduced to its worth-keeping form by the writer before storage. The server stores a note without a content gate; distillation is the writer's job, guided by the tool description and the server instructions.
_Avoid_: summarized, cleaned, processed
