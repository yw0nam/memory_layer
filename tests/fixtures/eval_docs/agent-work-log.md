# Checkout Latency Incident

On Tuesday the checkout API p95 latency rose from 180 milliseconds to 2.4 seconds after deployment 8f31c2. The investigation traced the regression to a new inventory client that opened a fresh TLS connection for every request. The team rolled back 8f31c2, restored p95 latency below 220 milliseconds, and recorded connection reuse as the required fix before redeployment. The incident owner was Mira, and the verification dashboard was the checkout-latency board.

## Inventory Client Follow-up

The replacement inventory client uses a pool of 64 persistent connections with a five-second acquisition timeout. Load testing at 900 requests per second held p95 latency under 240 milliseconds and produced no connection acquisition failures. The release checklist requires a ten-minute canary, comparison of error rate and latency against the previous version, and an immediate rollback if either metric exceeds the stated budget.

# Search Index Migration

The search service moved product embeddings from a flat index to an HNSW index. The accepted configuration uses `m=24`, `ef_construction=160`, and cosine distance over half-precision vectors. During rollout, `ef_search` starts at 80 and may be raised at runtime when recall falls below the evaluation target. The team chose a concurrent index build so reads remained available, then swapped indexes only after row counts and nearest-neighbor samples matched.

## Migration Recovery

The first migration attempt exhausted temporary disk space because the build ran beside an obsolete index. Recovery consisted of cancelling the build, dropping only the obsolete shadow index, expanding the temporary volume to 80 GB, and restarting the concurrent build. The runbook prohibits deleting the active index until the replacement passes row-count, null-vector, and ten-query relevance checks. Noah owns the final production migration.
