# Memory Base

A retrieval layer whose memory is distilled content. Agents write memories through a
validating tool; documents and code repositories are ingested through source adapters.
Every consumer reads back through one REST API.

## Language

### Stored content

**Chunk**:
The unit of storage and retrieval — one embedded, independently searchable piece of content.
_Avoid_: record, entry, item, document (for the unit)

**Note**:
An agent-authored memory, distilled by the agent before it arrives rather than extracted from a file.
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
subject supersedes the pending one. A `completed` handoff expires 3 days after it is
sent; an `in_progress` or `blocked` one has no expiry. Either way its sender may give an
`expires_at`, and it stays pending until claimed, superseded, cancelled, or expired.
_Avoid_: session summary, progress note, status report

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

**Author**:
The slug a client declares in the `X-Author` header: the attribution of everything the
client writes and the default scope of its note reads — a note read returns that author's
own notes and the shared notes unless the read names an author. Shape is checked; it
grants nothing beyond that.
_Avoid_: agent id, profile name, owner

**Shared**:
A note flag, set at creation, that puts the note in every author's default read scope; the
note stays attributed to the writer.
_Avoid_: public, global

**Tag**:
An uploader- or author-supplied label on a chunk, used to narrow retrieval. Not a boundary.
_Avoid_: label, category, topic

**Supersede**:
To replace a note with a newer one, archiving the old rather than deleting it. Superseded notes stay retrievable on request.
_Avoid_: overwrite, replace, delete, revoke

**Lineage fields**:
Note metadata that links a note to the note that replaced it: `supersedes` and `replaced_by` from a supersede save, and `archived_by` on an archived note.
_Avoid_: history, provenance, links

**Distilled**:
Reduced to its worth-keeping form by the writer before storage. The server stores a note without a content gate; distillation is the writer's job, guided by the tool description and the server instructions.
_Avoid_: summarized, cleaned, processed
