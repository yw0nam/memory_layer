# Retrieval benchmarks

Doc-level retrieval quality of the hybrid search stack, measured on two corpora.
Numbers are produced by the configuration in `retrieval/search.py`: BM25 FTS legs
(`pg_textsearch`, `<@>` / `to_bm25query`), vector search over halfvec embeddings,
and weighted RRF fusion (`vector 1.0, fts 0.2, recency 0.25, idf 0.25`) with
optional cross-encoder reranking.

## Method

- **Scoring**: ranked chunks are collapsed to ordered unique `source_ref`
  (document-level ranking); hit@k = any ground-truth doc in the top k,
  MRR@10 = reciprocal rank of the first ground-truth doc within the top 10.
- **ZX Bank**: the ZX Bank organization of the public RAG-Multi-Corpus dataset —
  71 markdown documents ingested through the full document pipeline (chunking +
  enrichment) into an isolated namespace; 100 QA queries sampled
  deterministically (every `len/100`-th row) from the dataset's query CSV,
  ground truth from its `Supporting Facts` filenames.
- **SciFact**: the BEIR SciFact test split — 5,183 title+abstract records seeded
  directly into `memory_chunks` with embeddings (retrieval-layer benchmark;
  product ingest is not under test), 300 claim queries scored against BEIR qrels.
- FTS-only / vector-only rows are single-leg ablations using the same SQL shape
  as the production legs; hybrid rows run `_search_memory` (all four RRF voters).

## ZX Bank (71 docs, 100 queries)

| mode | hit@5 | hit@10 | MRR@10 |
|---|---|---|---|
| FTS leg only | 0.87 | 0.94 | 0.76 |
| vector only | 0.93 | 0.95 | 0.84 |
| hybrid | 0.93 | 0.98 | 0.81 |
| FTS only + rerank | 0.89 | 0.93 | 0.76 |
| vector only + rerank | 0.92 | 0.94 | 0.81 |
| hybrid + rerank (default) | 0.93 | 0.95 | 0.81 |

Hybrid hit@10 (0.98) exceeds vector-only (0.95): the BM25 leg surfaces documents
the embedder misses. Run-to-run variance on this set is about ±0.01.

## SciFact (5,183 docs, 300 queries)

| mode | hit@5 | hit@10 | MRR@10 |
|---|---|---|---|
| FTS ablation: `websearch_to_tsquery` AND over tsvector | 0.02 | 0.02 | 0.02 |
| FTS leg only (BM25) | 0.78 | 0.84 | 0.66 |
| vector only | 0.82 | 0.87 | 0.69 |
| hybrid, equal weights + AND-tsquery leg (ablation) | 0.82 | 0.87 | 0.69 |
| hybrid | 0.82 | 0.89 | 0.71 |
| hybrid + rerank (default) | 0.86 | 0.90 | 0.75 |

The AND-tsquery ablation shows why the FTS leg is BM25: ANDing every term of a
natural-language query across paragraph-sized chunks matches almost nothing
(MRR 0.02), while the BM25 leg lands in the published BEIR-SciFact BM25 range
and lifts hybrid above vector-only.

## Known behavior

- Quoted phrases have no adjacency enforcement — the BM25 index stores term
  frequencies, not positions. Rare-token (identifier) lookups rank via IDF.
- FTS runs over `content_raw` (original language); embeddings run over the
  distilled English records. On non-English raw text the FTS leg contributes
  exact-token matching (identifiers, error strings, names) and the multilingual
  embedder carries semantic matching.
- Recency/idf RRF voters are tie-breakers (weight 0.25); ranked recency
  preference is enforced post-fusion by the time-decay multiplier.
- The bm25 index scan returns up to `bm25_catalog.bm25_limit` (default 100)
  candidates before outer predicate filters; namespace-heavy deployments can
  raise the GUC.
