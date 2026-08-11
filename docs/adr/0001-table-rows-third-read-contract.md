# ADR-0001: Table rows are a third read contract, reachable by SQL only

## Status

Accepted

## Context

`memory_chunks` and `code_chunks` are the contract between the write side and the read
side: retrieval and serving know only these two tables, and adding a source must not
change them. A tabular document, however, is stored as one Card, and any question whose
answer lives in the rows rather than in the prose is unanswerable. Prose cannot carry
2,000 rows of numbers; no amount of better summarization fixes an aggregate query.

## Decision

Tabular documents additionally load their rows into `doc_rows (namespace, document_id,
row_index, data jsonb)`, a third read contract with its own read path:

- Rows are read through SQL only (`POST /tables/query`), executed under a dedicated
  read-only role that can see `doc_rows` and nothing else, inside a read-only
  transaction, row-level-secured to the request's namespace.
- Rows are never embedded, never scored, and never returned by `search`, `search_code`,
  or `search_memory`. The Card remains the only embedded artifact of a tabular document
  and carries the handle (`document_id`, column list) a consumer needs to query.

Selective storage is unchanged: "low-signal data never reaches the DB" and "raw files
are never embedded" govern the embedding path, which table rows never enter. Rows are
structured data behind a compute interface, not content competing for retrieval.

## Consequences

- Consumers that can issue SQL gain aggregate answers over tabular data in one call;
  the search surface is unaffected.
- Every future table added to the schema must NOT be granted to the query role;
  the grant list is part of the trust boundary.
- Deleting a document and deleting a namespace must account for `doc_rows` in addition
  to `memory_chunks`.
- The read side is no longer describable by two tables alone; this document is the
  record of that widening.
