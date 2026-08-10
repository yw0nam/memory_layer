# Changelog

## [0.1.1] - 2026-08-10

### Fixed
- strip multi-line quoted spans before matching git verbs (#142)
- refuse to cut a release from an agent session (#145)
- create an annotated release tag (#140)

### Refactored
- clear the review follow-ups from the 2026-08-07 batch (#146)

## [0.1.0] - 2026-08-07

### Added
- generated changelog and a manual release script (#136)
- log silently swallowed failures in retrieval and ingest paths (#124)
- remove deep_search and the query decomposition pipeline (#121)
- restrict repo removal to the ingesting owner or an admin (#109)
- parameterize docker-compose DB credentials (#108)
- drop per-chunk LLM enrichment from document ingest; accept caller-supplied tags (#104)
- buffer search hit counts and retrieval logs off the request path (#99)
- durable Postgres job backlog with per-key fair dispatch, replacing in-memory queues and Redis (#96)
- BM25 FTS leg via pg_textsearch with weighted RRF fusion (#88)
- private git repository ingestion via mounted git credential store (#85)
- API-key authentication with namespace visibility model (#83)
- namespace-scoped memory separation with an explicit registry (#82)
- repo-scoped code search and MCP/REST search-option parity (#69)
- separate liveness from service health (#68)
- prepare the schema once per process and health-check every dependency (#64)
- hold all container state under one host data root (#61)
- bound a repo checkout and refuse ingestion on a nearly full disk (#60)
- make job state durable via Redis (#59)
- index by commit time and persist repo job state (#58)
- URL-driven multi-repo code ingestion via REST and MCP (#56)
- adopt core/logger.py as the unified logging path (#55)
- multi-hop eval extension with path-completion gate
- deep-search surface and memory source rename
- atom-based decomposition loop for multi-hop retrieval
- reproducible retrieval eval harness with atom-lane A/B gate
- document ingestion pipeline (conversion, chunking, enrichment, REST/MCP)
- retrieval kind/tags filters, atom lane plumbing, and tag normalization
- memory lifecycle management (admin endpoints, supersede, cold tier) (#36)
- REST API server as the single backend; MCP becomes a thin proxy (#33)
- English-only policy for stored memory and repo content (#26)
- disable batch burst rows for claude_code (per-adapter toggle) (#25)
- save_memory MCP tool (agent-driven note channel) (#24)
- weighted-sum burst gate (faithful to source) + decision extraction (#12)
- serve MCP over SSE from Docker (#11)
- selective ingest gates + two-pass corpus DF (#8)
- Phase 1 agent history ingestion + distillation pipeline (#4)
- Phase 2 MCP thin tools + answer pipeline tests (#3)

### Fixed
- scope the git guard to this repository (#138)
- route access-log test DB calls through the client's portal loop (#134)
- schema-qualify the bm25 index name in FTS queries (#133)
- keep backlog integration test rows unclaimable by production workers (#102)
- apply the Qwen3 reranker chat template so relevance scores separate (#79)
- require DB_URL instead of falling back to hardcoded credentials (#71)
- persist the api container's CocoIndex ledger on its own volume (#57)
- guard MCP proxy 400-error parsing against non-JSON bodies (#48)
- deterministic decomposition decisions and honest multi-hop labels
- planner prompt forbids copying non-English words into queries
- one_line_question anchors on the session core topic, not the opening message (#39)
- fall back to a trivial plan on malformed planner JSON; unify hit scoring (#40)

### Refactored
- remove the dead atom retrieval lane (#130)
- share job dataclass plumbing between IngestJob and RepoJob (#128)
- single-source search option validation (#127)
- consolidate serve-layer request and response helpers (#126)
- replace the eval global schema patch with explicit scoping (#125)
- collapse the repeated HTTP call pattern in MCP tools (#122)
- delete dead code and single-source redundant definitions (#120)
- require the model names from the environment (#80)
- bound duplicate detection and prune retrieval log, stale migration, and document locks (#78)
- replace per-call asyncpg connects with one shared connection pool (#77)
- collapse repeated terminations and extract the document row builders (#72)
- update MCP transport from SSE to streamable HTTP in documentation and configuration
- run both job kinds through one registry (#67)
- name shared retrieval helpers publicly and drop dead indirection (#66)
- hold shared config and schema in the core package (#65)
- agents contribute via MCP only; remove the claude_code file-ingestion channel (#35)
- source adapter architecture (ABC contract) (#23)

### Documentation
- align README, AGENTS.md, and image defaults with the current system (#123)
- correct BM25 candidate-scan note with measured streaming behavior (#98)
- correct Bitbucket git-over-https username to x-bitbucket-api-token-auth (#86)
- restructure agent guide; drop supersede from docs guard vocabulary
- add agent skills config (issue tracker, triage labels, domain docs)
- update agent instruction use docker exec not uv run
- add README with architecture and data-flow structure (#62)
- implementation spec for knowledge-aware decomposition (deep search)
- implementation spec for document ingestion and PIKE-RAG enrichment

### Tests
- pass required tags to admit_document in backlog integration tests (#105)
- scope archive-flow assertions to the test's own rows (#100)
- mirror the src package layout in tests/ (#70)
- make integration tests self-seeding and order-independent (#63)
- mock LLM accepts the temperature argument
- keep the archive marker token out of the near-dup rows
- characterization + integration coverage for core paths (#6)

### Maintenance
- remove the dormant batch selection core (#52)
- unify timeouts, name constants, declare direct deps (#51)
- align the written record with the code and drop the unused answer CLI (#49)
- remove dead code, unused pgvector dep, and stale spec content (#37)
- src-layout package, ruff lint, CI + repo templates (#15)

### Other
- first commit
