# Implementation spec: document ingestion + PIKE-RAG enrichment (Phase 3)

## Usage model

The system serves four usage patterns, which anchor every design decision below:

1. **Agent work-history lookup** — agents (Claude Code, Hermes) search past logs, decisions,
   and work records mid-conversation or mid-automation.
2. **Companion long-term memory** — a real-time conversational agent (Hermes companion)
   retrieves memories from conversation fragments, not only explicit questions.
3. **Personal knowledge base** — the user saves noteworthy items (`save_memory`) and ingests
   documents ("store this, I'll want it in my KB later").
4. **Automation output** — n8n workflows persist results/logs. n8n is a **REST consumer**,
   not a source adapter: its HTTP node calls `save_memory` / `POST /ingest/document` directly.

## Scope

Ingestible extensions, exactly: `.md .markdown .txt .rst .html .htm .pdf .docx .pptx .csv`.
Legacy Office formats (`.doc .xls .ppt`) are rejected with `415`.

Out of scope: URL fetching, OCR for scanned PDFs (extraction yielding no text fails the job
with an explicit error), row-level CSV chunking, tag-vocabulary normalization (add only if
tag divergence is observed), and any revival of session-transcript ingestion (agent
conversations contribute via `save_memory` only).

**Selective-storage interpretation for documents.** The transcript rule (distill-then-embed,
never embed raw) exists because transcripts are noisy byproducts. Documents are curated
content the user deliberately uploads: an LLM paraphrase of a document chunk would lose
fidelity and add cost without retrieval gain. For documents the selection chain is:
deliberate upload (triage) → conversion → chunking with the junk gate and minimum-size
rule below (selection; sub-minimum residues are **dropped, never stored**) → enrichment
(English atoms/tags — the generated, distilled layer). Every stored chunk passed the gate
and is enriched.

## Module boundaries (source-specific logic stays in the adapter)

- `adapters/document.py` — all document-format knowledge: the extension matrix, markitdown
  conversion, CSV sampling and card-row mapping, and the mapping from converted content to
  `memory_chunks` row dicts. Output contract: a list of row dicts (the same shape every
  write path emits).
- `ingest/enrich.py` — source-agnostic enrichment with two generic operations:
  `atomize_and_tag(text, context)` → `{atom_questions, tags}` and
  `summarize_and_tag(text, context)` → `{summary, tags}`. Neither knows about CSV or any
  other format; the caller picks the operation and supplies the prompt context.
- `serve/ingest_api.py` — orchestration only: upload handling, job registry, the
  atomic-replacement transaction. No format knowledge.
- Retrieval gains no document-specific branching: the atom lane treats atoms as generic
  children of any parent row (see Retrieval).

## Intake

### REST

`POST /ingest/document` — multipart form (Starlette + `python-multipart`):

- `file` (required): document bytes; filename from the part. Bodies spool to a temp file,
  deleted when the job leaves the queue terminally.
- `document_id` (optional): stable logical identity. Default: normalized filename
  (lowercased, path-stripped, extension kept). After normalization it must match
  `^[a-z0-9][a-z0-9._-]{0,120}$` — else `400`. (No colons: row-id patterns stay
  collision-free.) Two different files sharing a filename need distinct `document_id`s; a
  renamed file keeps its identity only if the client passes the same `document_id`.
- `mode` (optional): `upsert` (default) — unchanged content hash is a no-op; `force` —
  re-ingest regardless (the re-enrichment path after prompt/chunker/model changes; no
  separate enrichment API).
- `origin` (optional): informational client-side locator, stored in metadata verbatim.
  The server does not retain the original file.

Responses: `202 {job_id, status, status_url}` · `400` malformed form / bad `document_id` ·
`413` over `INGEST_MAX_BYTES` (default 25 MB) · `415` unsupported extension · `429` queue
full.

`GET /ingest/jobs/{job_id}` → `{job_id, document_id, status, stage, chunks_total,
chunks_done, chunks_dropped, rows_written, enrichment_retries, content_hash, error,
created_at, updated_at}`; `status ∈ queued|running|succeeded|failed|no_op`;
`stage ∈ queued|converting|chunking|enriching|embedding|writing|done`; `404` means unknown
**or expired**. Error bodies everywhere use the existing `{"error": message}` shape.

### Bounds (personal scale, but never unbounded)

- Queue: at most `INGEST_MAX_QUEUED` (default 10) jobs waiting → `429`.
- Concurrency: `INGEST_MAX_CONCURRENT_JOBS` (default 2); enrichment LLM calls per job
  under a semaphore of 4.
- Derived-size limits, each failing the job with a recorded error (`422` semantics in the
  job): extracted text over 2,000,000 chars; over 2,000 accepted chunks; over 5,000 total
  rows (chunks + atoms + cards).
- Completed jobs expire after 24 h or beyond the newest 100. Single-worker deployment is
  an invariant (ponytail: move jobs to a table if that ever changes). Background tasks
  hold strong references.

### MCP

`ingest_document(content, filename, document_id=None, origin=None, mode="upsert")` — text
formats only (`.md .markdown .txt .rst .html .htm`); proxies to REST and returns
`{job_id, status_url}`. Binary formats upload via REST directly; no base64 tunneling.

### Idempotency and atomic replacement

The whole pipeline (convert → chunk → gate → enrich → embed) completes in memory first;
then a **single transaction** runs `DELETE FROM memory_chunks WHERE source_type='document'
AND source_ref = $document_id` and inserts the new rows. A failure at any earlier stage
leaves the previous version intact. A per-`document_id` asyncio lock serializes concurrent
uploads of the same document. `content_hash` (sha256 of the uploaded bytes) is stored on
every row; the upsert no-op check compares it against any existing row for that
`document_id`. A pipeline ending with zero accepted rows **fails** the job (nothing is
stored, no hash is recorded).

## Conversion and chunking

- All formats convert to markdown via `markitdown[pdf,docx,pptx]` (pinned minor version),
  converter name+version recorded in row metadata. Empty extraction fails the job.
  `content_raw` is **converted source text** — not byte-verbatim for pdf/docx/pptx/html —
  in the source language. CSV bypasses conversion (see CSV cards).
- **Conversion is bounded by construction**: it runs in a separate OS process (killable —
  not a thread): a worker entrypoint in this package, spawned via `python -m`, calls the
  markitdown Python API and writes converted text to a temp file through a **bounded
  writer that aborts the child with a distinct exit code once 16 MB is written** (a
  compressed docx/pptx/pdf can expand far past the 25 MB upload cap). The parent
  additionally kills the child after 120 s, and — regardless of exit path — **`stat`s the
  output file and fails the job if it exceeds 16 MB before reading a byte**. After a clean
  exit within bounds the text is read back and the 2,000,000-char limit applies. Temp
  files (spooled upload + conversion output) are deleted on every exit path, including
  kills. Tests include a fast-write-and-exit oversized output.
- **Chunker** (pure, deterministic — one algorithm, in order; all lengths Python `len`):
  1. *Blocks.* Scan lines once. An ATX heading (`#{1,6} `) updates the current heading
     path and emits no block. A fenced code block (``` to closing ```) is one atomic
     block. Remaining text splits into paragraph blocks at blank lines. Every block
     carries the heading path in effect where it starts.
  2. *Packing* (greedy, left-to-right, never across a heading-path change): append the
     next block to the current chunk (joined with `\n\n`) while the result stays ≤ 1,500
     (soft target); otherwise close the chunk and start a new one with that block.
  3. *Oversized blocks* (> 2,000 alone): paragraphs split at the last whitespace before
     char 2,000 (hard cut at 2,000 if none); fenced blocks split at the last newline
     before char 2,000. Fenced blocks ≤ 2,000 never split.
  4. *Residue pass* (left-to-right cursor over packed chunks): a chunk under 200 chars
     merges into its **predecessor** when both share the same heading path and the result
     stays ≤ 2,000; else into its **successor** under the same two conditions
     (cross-heading merges are prohibited); else it is **dropped** and counted in
     `chunks_dropped`. After a predecessor merge the cursor stays on the merged chunk
     (it may absorb consecutive residues while the bound holds); after a successor merge
     the merged result is **reconsidered at the same position** until it reaches 200,
     merges again, or is dropped. The pass ends only when no remaining chunk is under
     200 — a sub-200 chunk is never stored.
  5. *Junk gate* (below) runs over the surviving chunks.
  6. *Ordinals* `0..n-1` are assigned in document order to the final accepted chunks,
     after merging, dropping, and the junk gate.
- **Junk gate** (deterministic; chunker step 5, before ordinal assignment): a chunk is
  dropped and counted when
  letters/total-non-whitespace-chars < 0.3 (Unicode letter class), or when more than half
  of its non-blank lines are link lines (a line whose markdown-link/bare-URL spans cover
  more than half its characters).

## Storage (`memory_chunks` unchanged)

Full column mapping (columns not listed: `idf_score = NULL`, `archived_at = NULL`):

| Column | doc chunk | atom | CSV card |
|---|---|---|---|
| id | `doc:{document_id}:{ordinal}` | `doc:{document_id}:{ordinal}:atom:{i}` | `doc:{document_id}:card:{i}` |
| source_type | `document` | `document` | `document` |
| source_ref / session_id | `{document_id}` | `{document_id}` | `{document_id}` |
| chunk_kind | `doc` | `atom` | `doc` |
| content_raw | chunk text (source language) | English question | English summary card |
| distilled | `NULL` (retrieval falls back to content_raw) | the question | the card |
| embedding | heading path + chunk text | the question | the card |
| ts_last_active | ingest time | ingest time | ingest time |
| metadata | filename, document_id, heading_path, ordinal, content_hash, format, converter, origin, tags, search_ref | parent_id (the doc chunk id), parent_kind, document_id, content_hash | filename, document_id, card_index, format=csv, row/column counts, content_hash, origin, tags, search_ref |

Search `ref` is produced source-agnostically: the **adapter** writes
`metadata.search_ref` on parent rows (`{document_id}#chunk-{ordinal}` for doc chunks,
`{document_id}#card-{i}` for CSV cards) and retrieval returns `metadata.search_ref`
when present, else `source_ref` — no document branching in retrieval. Atom-resolved
parents surface the parent's `search_ref`. **Dedup grouping is unchanged**: `Hit.meta`
keeps `source_ref`, and `_dedup_cap` groups history hits by it (not by the displayed
ref), so one document still occupies at most `PER_FILE_CAP` baseline slots even though
each chunk displays a distinct `search_ref`. Everything the pipeline generates — atoms, tags, cards — is English;
`content_raw` keeps the source language (the embedder is multilingual; FTS uses the
`simple` config either way).

## Enrichment (PIKE-RAG L2 atomizing + tagging; `ingest/enrich.py`)

- Atom generation is gated on `ATOMS_GENERATE` (env, default true); tagging always runs.
- One `atomize_and_tag` call per chunk (JSON mode) returns
  `{"atom_questions": [...], "tags": [...]}`: atom questions entity-explicit, no pronouns
  (prompt ported from PIKE-RAG `atom_question_tagging.py`), capped at 10 per chunk, each
  ≤ 200 chars, deduplicated case-insensitively; tags 3–7 phrases matching
  `^[a-z0-9][a-z0-9 -]{1,40}$` after lowercasing/trimming; invalid entries dropped.
- **Success criteria** (both operations, checked after dropping invalid entries):
  `atomize_and_tag` succeeds when the response is a JSON object with list-typed fields
  and ≥ 1 valid tag remains (an empty atom list is valid — some chunks yield no good
  question); `summarize_and_tag` succeeds when `summary` is a non-empty string and ≥ 1
  valid tag remains. Anything else — transport error, invalid JSON, wrong shape,
  zero valid tags, empty summary — counts as a failure and is **retried once**.
- **Fail-closed**: an enrichment call that fails again after its retry **fails the whole
  job** (the atomic transaction never runs; the previous document version stays intact;
  `error` names the chunk ordinal or card). This keeps the invariant that every stored
  chunk is enriched. Retries that succeeded are counted in `enrichment_retries`. Turning
  enrichment down is an explicit choice (`ATOMS_GENERATE=false` skips atoms), never a
  silent degradation.
- **Tag normalization is write-side and universal**: `save_memory` (`notes.py`) gains the
  same validation — tags must arrive as a list of strings, are lowercased/trimmed/
  deduplicated, empties dropped, non-conforming requests `400`.
- **Legacy tag migration** (one-off idempotent `UPDATE` in `ensure_schema`, by JSON type
  of `metadata->'tags'`): a JSON array keeps only its string elements, lowercased/trimmed/
  deduplicated, empties dropped — if the result is empty the `tags` key is removed; any
  non-array value (string, object, number) removes the key; only rows whose normalized
  value differs are updated. This keeps JSONB containment queries correct for legacy rows.
- **CSV cards** (all CSV knowledge in `adapters/document.py`): the `csv` module reads
  header + first 20 rows + row/column counts (5 MB cap, malformed CSV fails the job); the
  adapter builds the prompt context and calls the generic `summarize_and_tag`, mapping
  `{summary, tags}` to one English card row. `enrich.py` stays format-unaware. The
  original file is **not retained**; `origin` metadata is the only pointer back. No
  cell-level lookup is promised.

## Retrieval (two lanes; atoms never displace baseline candidates)

- **Baseline lane**: `_search_history` excludes atom rows (`chunk_kind <> 'atom'`), now
  selects `chunk_kind` + `metadata`, and runs to completion exactly as today: RRF fusion,
  time decay, `_dedup_cap` to `FUSED_TOP` (20). `Hit.meta` carries kind and tags.
- **Atom lane** (runs when `include_atoms` resolves true and the active `kind`/`tags`/
  archive filters could match a parent): one vector query over `chunk_kind = 'atom'` rows
  **joined to their parent row**, with parent kind/tags/`archived_at` predicates applied
  inside the query, before ordering and `LIMIT 3 × ATOM_RETRIEVE_K` (default
  `ATOM_RETRIEVE_K` 8). Fetched atom rows are then **collapsed by parent id**, keeping
  the highest-cosine atom per parent, and truncated to `ATOM_RETRIEVE_K` unique parents
  (fewer is fine — no re-fetch). `include_archived=True` admits archived parents,
  matching baseline behavior. Atom rows never carry their own `archived_at`: lifecycle
  candidate selection and archive/restore `UPDATE`s add `chunk_kind <> 'atom'`, so a
  request naming an atom id changes nothing and parent state alone controls atom
  visibility. Atoms get no recency/IDF votes. Each resolved parent becomes a
  candidate carrying the parent's text/ts/kind/tags, `meta.atom_id`, and the matched
  question; parents already among the 20 baseline candidates are not duplicated (atom
  evidence only annotates them). The union — at most 20 + 8 candidates, with **no further
  shared truncation** — feeds the reranker. With rerank disabled, atom-derived parents
  keep `rrf = 0.0` and order after baseline hits, sorted by atom cosine similarity among
  themselves. Access logging credits the parent row id.
- **Filters**: `kind` (exact, one of `doc|note|decision`) and `tags` (list of normalized
  lowercase phrases, **ANY** semantics) apply to `memory_chunks` only and require
  `source="history"`; any other source combination is `400`. An empty `tags` list is
  `400`. Filters are SQL predicates inside both the vector and FTS queries, before their
  LIMITs. `ensure_schema` adds a GIN index on `(metadata->'tags')`.
- `include_atoms` (search option, default env `ATOMS_RETRIEVE`, default true) lets
  retrieval exclude atoms without deleting rows — this is the A/B switch.
- REST `/search` and the MCP `search`/`search_history` tools expose `kind`, `tags`, and
  `include_atoms`.

## Validation (acceptance)

- **Reproducible atom A/B gate**: the eval corpus is checked in
  (`tests/fixtures/eval_docs/` — small markdown/csv fixtures) together with
  `tests/fixtures/retrieval_eval.jsonl` (20–30 labeled queries `{query, query_class,
  relevant_ids}`, ≥ 5 queries per class, covering usage patterns 1–3 including
  statement-form companion fragments). `uv run python -m memory_base.eval.retrieval`
  ingests the fixtures into a scratch schema, runs every query with `include_atoms` on
  and off, and prints Recall@5 and MRR@10 per class plus the embedding/rerank model ids.
  A query whose `relevant_ids` are absent from the corpus counts as a recall miss. Gate:
  per-class Recall@5 with atoms on ≥ atoms off − 0.05, and overall Recall@5 does not
  drop. The report is PR runtime evidence. If the gate fails, ship
  `ATOMS_RETRIEVE=false` and revisit at Phase 4 (decomposition).
- **Kind filter**: `kind="decision"` returns only decision rows; a decision-targeted eval
  query places its target in the top 3 under the filter.
- **Idempotency/atomicity**: unchanged re-upload → `no_op`, zero row changes; modified
  upload replaces exactly that document's rows (and no other source's); a fault-injected
  mid-pipeline failure leaves the previous rows intact; a zero-accepted-rows pipeline
  fails without writing.
- **Unit tests** (no DB/LLM): chunker (heading/size/fence behavior, merges near the
  2,000-char bound in both directions, unmergeable-residue drop, a 5,000-char paragraph,
  a mixed-heading fixture where both neighbor merges fit size-wise and the heading rule
  decides, consecutive residues, a successor merge still under 200 that must merge again
  or drop, residues at a heading-path end), junk gate (ratio denominator, link-line
  classification, ordinals contiguous after junk drops), enrichment parser (caps, tag
  pattern, retry on invalid JSON / wrong shape / zero valid tags / empty summary,
  persistent failure → job failure incl. the CSV card path), CSV card
  builder with a mocked LLM, tag normalization in `save_memory` (malformed input), legacy
  tag migration by JSON type (string, object, mixed array, empty array, missing key),
  filter validation (kind/tags/source combinations), `document_id` validation and row-id
  collision cases, `search_ref` resolution (doc-chunk hit, card hit, atom-resolved
  parent hit, a row without `search_ref` falling back to `source_ref`, and multiple
  chunks of one document staying subject to `PER_FILE_CAP` despite distinct refs),
  atom-lane
  parent resolution (dangling parents; several atoms of one
  parent collapse to the highest-cosine one), lifecycle atom exclusion (archive candidates
  omit atoms; archiving an atom id is a no-op; archiving a parent hides its atoms from the
  atom lane).
- **Integration tests**: end-to-end markdown ingest (upload → job → rows → searchable),
  CSV card ingest, force-mode re-ingest, oversized conversion output (mock converter
  exceeding the byte cap → process killed, job fails, temp files removed, prior rows
  intact).

## Implementation split

1. **Retrieval filters + atom lane plumbing** — `kind`/`tags`/`include_atoms`, atom-row
   exclusion in baseline, `chunk_kind`/metadata selection, GIN index, `save_memory` tag
   normalization + legacy migration, lifecycle atom exclusion (`chunk_kind <> 'atom'` in
   candidate selection and archive/restore updates), REST/MCP exposure. Immediate value
   for `save_memory` content; the atom lane is dormant until atom rows exist.
2. **Document pipeline** — `adapters/document.py` (conversion, chunker, junk gate, CSV
   cards, row mapping), `ingest/enrich.py`, `serve/ingest_api.py` (job registry, REST
   endpoint, atomic replacement), MCP tool.
3. **Eval harness** — fixtures, eval runner, A/B report (lands with or right after 2).
