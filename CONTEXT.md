# Memory Base

A retrieval layer that stores only distilled, high-signal content. Agents write
memories through a validating tool; documents and code repositories are ingested
through source adapters. Every consumer reads back through one REST API.

## Language

### Stored content

**Chunk**:
The unit of storage and retrieval — one embedded, independently searchable piece of content.
_Avoid_: record, entry, item, document (for the unit)

**Note**:
An agent-authored memory, distilled by the agent before it arrives rather than extracted from a file.
_Avoid_: memo, observation, fact

**Decision**:
A note recording a choice that was made and is expected to hold, as opposed to an observation.
_Avoid_: ruling, conclusion

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

**Creator**:
The key label that first ingested a document or repo. Fixed at first ingest; the only
non-admin identity allowed to overwrite or delete it.
_Avoid_: owner (reserved for namespaces), author, uploader

**Code chunk**:
A span of source lines from an indexed repository. Retrieved from its own lane, separate from memories.
_Avoid_: snippet, fragment

**Repo**:
A git repository cached and indexed for code retrieval. Managed and re-indexable, unlike a document.
_Avoid_: codebase, project, source

### Organization

**Namespace**:
A registered isolation boundary that partitions memories. Either public, or private to the key that created it.
_Avoid_: tenant, workspace, bucket, collection

**Tag**:
An uploader- or author-supplied label on a chunk, used to narrow retrieval. Not a boundary.
_Avoid_: label, category, topic

**Supersede**:
To replace a note with a newer one, archiving the old rather than deleting it. Superseded notes stay retrievable on request.
_Avoid_: overwrite, replace, delete, revoke

**Distilled**:
Reduced to its high-signal form before storage. The property that qualifies content for the store at all.
_Avoid_: summarized, cleaned, processed
