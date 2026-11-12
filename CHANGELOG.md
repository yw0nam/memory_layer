# Changelog

## [0.6.0] - 2026-11-12

### Added
- the server drops documents, tables, repos, ingest, the chat LLM, and the eval harness (#420)
- the server drops entities, facts, consolidation, the turn lane, and search logging (#419)
- a completed handoff expires 3 days after it is sent (#411)
- list_messages returns headers by default and matches a subject fragment (#409)
- top_k searches rerank 20 fused candidates and budget-mode searches rerank 40 (#399)
- the Claude Code prompt hook only captures the turn (#397)
- memory_base.eval.sandbox serves the real REST API on a throwaway Postgres (#390)
- scripts/worktree.sh creates a ready worktree with one command (#389)
- the session-start hook lists completed handoffs on their own line (#388)
- author read filter takes a list of agents (#381)
- the client declares its author in the X-Author header and notes read by default in the author's own scope (#379)
- OpenCode Go as the first chat provider (#377)
- delete an entity, unlinking its notes and removing its facts, in place of archive (#374)
- one kind of key; the server grants nothing beyond authentication (#371)
- link and unlink entities on existing notes with link_entities (#368)
- one card per entity replaces the profile, with notes naming entities and facts distilled by consolidation (#366)
- a verbatim turn lane with its readers, the Claude Code plugin, and the Hermes provider (#356)
- the user part of a profile changes by consent inside the agent loop (#348)
- entity registry with the user's consent inside the agent loop (#347)
- expose the profile read as the get_my_profile MCP tool (#334)
- selective benchmark data from Memora weekly timelines and MemOps update units (#332)
- tell agents how a profile narrows or widens what save_memory keeps (#329)
- reduce the save_memory description to the points a save decision rests on (#327)
- one key per agent with keys CLI authors and a connect-agent skill (#313)
- agent-owned profiles with a user part changed only by the user's approval (#307)
- profile slots user and work-rules delivered at session start (#302)
- apply consolidation verdicts with lineage, a verdict cache, and undo (#300)
- list consolidation candidate groups at GET /admin/consolidate/groups (#299)
- remove the LLM content gate from the note save path (#298)
- save notes through one save_memory tool with kind as a label and a low-signal gate (#294)
- replay LongMemEval sessions through a headless agent with the real MCP save tools (#292)
- fit each save tool's bar into the summary a tool catalog shows (#281)
- remove conversation sources and session capture (#275)
- split note saving into save_personal_memory and save_work_memory (#276)
- measure write-time supersede on LongMemEval with an agent writer, hit judge, and frontier (#272)
- stop server-side distillation and let agents list captured sessions and search within one (#271)
- record the superseded note id on its replacement and tell agents to search before saving (#270)
- lower the prefetch floors and skip oversize hits when packing (#265)
- sweep read settings per read path from one candidate search (#264)
- raise the message content limit to 16 KiB (#263)
- capture every Claude Code and Hermes session at session end and distill it server-side (#262)
- rank memory without pre-rerank age decay and pack search hits into a token budget (#260)
- accept conversation memory at the save_memory content gate (#261)
- prefetch every allowed namespace, announce handoffs without claiming, and keep a handoff until claimed (#259)
- store conversation sources and link every note to its conversation (#258)
- measure the agent-distilled write path on a LongMemEval_S subset (#251)
- refuse notes and documents that contain credentials before any model call (#246)
- remove the allow_restatement content-gate override (#234)
- add addressed messages and handoff snapshots (#227)

### Fixed
- the Bash guard denies a pytest run piped into tail or head without pipefail (#412)
- the reranker call retries one connect failure on a 2 s connect timeout and a search 503 logs its cause (#406)
- notes are capped at 1750 characters so a note fits the Hermes prefetch (#405)
- read_sweep delivers only what each read path delivers in production (#404)
- the Bash guard denies only a read whose file operand is the secrets file (#400)
- delete and restore name the ids they skip, like archive (#387)
- refuse a keyword-stated PIN, lock combination, or one-time code as a credential (#341)
- archive by ids names the missing ids and list_memory_duplicates defaults to 0.75 (#339)
- duplicate listing and cold-tier archive read only agent notes (#309)
- a name that moves to or from a sentence start no longer fails the merge token check (#305)
- key the memory candidate cap on session so every note can reach the reranker (#249)
- return a note whole from search instead of cutting it at 2000 chars (#244)
- remove the phantom idf RRF voter and the idf_score column (#245)
- refuse a supersede that would leave zero active notes (#233)
- hold the namespace row for the whole save_note transaction (#232)
- carry the failure reason on every MCP tool error (#229)

### Refactored
- split search and health out of the serve composing modules (#321)
- split access and tables into serve feature packages (#320)
- split documents and repos into serve feature packages over common/job_store (#319)
- split consolidation into a serve/consolidation feature package (#318)
- split notes into a serve/notes feature package (#317)
- split messages into a serve/messages feature package with its own purge route (#316)
- split profiles into a serve/profiles feature package (#315)
- fold schema migrations into CREATE TABLE and remove dead code (#314)

### Documentation
- AGENTS.md states each rule positively and points at worktree.sh and the sandbox (#392)
- the domain glossary is GLOSSARY.md, the name the domain-modeling skills read and write (#386)
- install guide explains author, user entity, and namespace (#383)
- card step holds a claim the description states and a rule (#375)
- name the safeguard each admin route takes; drop key scoping from the documents routes docstring (#373)
- a fact value is one short value without a comma (#369)
- ADR-0013 one card per entity replaces the profile (#357)
- ADR-0011 entities and facts beside notes, with the user's consent inside the agent loop (#342)
- judge every consolidation group twice, hold disagreements for the owner, require compressed merges (#324)
- owner decisions for unsettled consolidation groups and Markdown run reports (#323)
- align README, data-flow, CONTEXT, MCP descriptions, and ADR status lines with the code (#311)
- adopt established-products, least-code, composing-module, layering, and test-scope principles (#312)
- default consolidation verdicts by overlap type (#310)
- describe the scheduled consolidation procedure for an agent (#301)
- show 20 MCP tools and the personal and work note kinds in the diagrams (#277)

### Tests
- the notes replay labels point at notes in the active corpus (#407)
- prune redundant tests without losing coverage (#322)
- re-read all 44 notes replay labels against the live corpus (#247)
- run integration tests against a throwaway Postgres container (#231)

### Maintenance
- scripts/deploy.sh deploys main, and the domain guide lists where a contract is stated (#413)
- the docs current-state guard runs on every PR from one shared script (#391)
- add tool-less LongMemEval answerer and judge agents (#250)

### Other
- selective benchmark data until the benchmark is ready for the README (#332) (#333)

## [0.5.0] - 2026-09-21

### Added
- skip the prefetch search for a desire-tick turn (#217)
- judge an episode on provenance alone instead of exempting it from the gate (#216)
- build and score a labeled replay set of real memory queries (#212)
- refuse a note whose content restates a tracker artefact or reports progress (#211)
- select the LLM chat provider from API keys in .env (#210)

### Fixed
- keep an answer archived since labelling out of the notes replay's score (#220)
- judge a note by provenance instead of the write policy's examples (#214)

## [0.4.0] - 2026-09-14

### Added
- refuse a near-duplicate note unless it supersedes or acknowledges its neighbours (#209)
- require tags on every saved note (#208)
- drop the whole client-injected context block from prefetch queries (#206)
- record the filters that narrowed each logged search (#205)
- remove the recency episode digest from the Hermes plugin (#202)
- memory lifecycle tools over MCP with note authorship (#201)
- quiet health-check and MCP ping noise in serving logs (#196)
- fail the query path fast with an explicit unavailable message when a vLLM endpoint is down (#191)
- default the prefetch hook env file to ~/.config/memory-base/env (#189)
- claude code prefetch hook that injects memory into sessions (#188)
- strip client_context boilerplate from the hermes prefetch query (#186)
- let config api_key override the env var for memory_base provider (#184)
- hermes memory provider integration under integrations/hermes (#183)
- episode notes, occurred_at backdating, and admin notes move (#182)
- restart services after reboot and add a database backup script (#178)
- since/until search bounds and a query-less notes listing (#179)

### Fixed
- keep buffered retrieval access when a flush fails, document MCP transport vars, pin CI actions (#197)
- clear the access-log buffer around every test (#195)

## [0.3.0] - 2026-08-18

### Added
- add a min_score relevance floor to search (#177)
- advertise memory-base usage instructions in the MCP handshake (#176)

### Fixed
- keep vendored bundles out of code ingest and bound its embedding fan-out (#169)
- refresh uv.lock before staging a release commit (#168)
- return no code hits when code_chunks has never been created (#167)

### Refactored
- unify the missing-chunk-table guard in retrieval on one mechanism (#174)

### Documentation
- add architecture and data-flow diagrams (#166)
- link the runnable example deployment (#165)

### Maintenance
- read mcp allowlist in docker compose
- fail a PR whose uv.lock has drifted from pyproject.toml (#173)

## [0.2.0] - 2026-08-11

### Added
- load tabular documents into a SQL-queryable doc_rows lane (#161)
- document ownership and deletion lifecycle (#160)
- license the project under MIT (#148)

### Fixed
- keep code-chunk ids out of the memory-chunk hit counters (#155)
- let a deployment declare the MCP transport's allowed hosts (#153)
- create the schema and vector extension before the memory table (#152)

### Documentation
- split data-flow and configuration references out of the README (#164)
- describe the SQL table lane and the document ownership lifecycle (#163)

### Maintenance
- add env.example

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
