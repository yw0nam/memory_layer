# Agent Guide

## Core principles

- **Low-signal data never reaches the DB.** Agent-authored notes arrive already distilled through the MCP `save_memory` tool (validation, dedup, supersede) and are stored without embedding raw text. Documents go through chunking and LLM enrichment (atomize/summarize) before embedding. Raw transcripts and raw files are never embedded directly; only distilled, high-signal content is stored.
- **Source-specific knowledge lives only in source adapters (`adapters/`).** `ingest/`, the storage schema, retrieval (`retrieval/`), and consumers (`serve/`) are source-agnostic. The `memory_chunks` and `code_chunks` tables are the only contract between the write side and the read side — adding a new source must not change retrieval or serving code.
- **Assemble verified engines; don't reinvent.** Chunking/incremental indexing/vector storage/serving use proven components (CocoIndex, pgvector, FastMCP). Design decisions are grounded in web research and confirmed with the user — not invented ad hoc.

## Project structure

The REST API is the single backend: every consumer (MCP server, n8n, scripts) reaches
stored chunks through it, never through the DB directly.

```
src/memory_base/
  core/
    config.py         # shared constants + LLM/embedding clients (vLLM, OpenAI-compatible)
    schema.py         # memory_chunks + retrieval-log DDL
    logger.py         # unified loguru setup: colored stderr + daily-rotated file sink
  ingest/
    enrich.py         # generic JSON-mode enrichment for stored content
    code.py           # CocoIndex app: multi-repo code chunking + embedding over the repo cache
  adapters/
    document.py       # document conversion, chunking, CSV sampling, storage-row mapping
    document_worker.py# killable MarkItDown conversion worker
  retrieval/
    search.py         # hybrid search: FTS + vector + IDF + time decay → RRF → rerank
    decompose.py      # knowledge-aware decomposition: multi-hop retrieval over memory atoms
  serve/
    api.py            # Starlette REST API (search, deep search, save, repos, admin) — the backend
    ingest_api.py     # bounded async document-ingestion orchestration
    repos.py          # URL-driven git repo cache management + code re-indexing
    job_store.py      # Redis mirror of job state: survives a restart, degrades to memory-only
    notes.py          # validation + storage for agent-authored notes
    admin.py          # memory lifecycle operations (duplicates, archive, restore)
    access_log.py     # best-effort persistence of retrieval activity
    mcp_server.py     # MCP server over REST (stdio | SSE | streamable-http), Docker serves streamable HTTP
  eval/
    retrieval.py      # reproducible retrieval evaluation with an atom-lane A/B report
tests/                # pytest mirrors the covered src package; `integration` marks DB/vLLM dependence
  core/
  adapters/
  ingest/
  retrieval/
  eval/
  serve/
  conftest.py         # shared pytest configuration and fixtures
  fixtures/           # shared fixture data
  test_compose_wiring.py # repository-level Compose wiring
scripts/ci/           # CI helper scripts (test-guard)
```

## Commands

```bash
uv sync                                              # install deps + editable package
uv run pytest                                        # all tests (integration skips if DB is down)
uv run pytest -m "not integration"                   # unit tests only (what CI runs)
uv run ruff format --check . && uv run ruff check .  # lint (ruff is the only Python linter)
docker compose up -d db redis                        # pgvector on :5439, job state on :6379
docker compose up -d --build api                     # REST backend on :8010
docker compose up -d --build mcp                     # MCP server, streamable HTTP on :8765
docker compose exec api uv run cocoindex update src/memory_base/ingest/code.py   # (re)index every cached repo
claude mcp add --transport http memory-base http://localhost:8765/mcp
```

Code repositories are added and removed by git URL at runtime — `POST /repos {url}`,
`DELETE /repos/{name}`, `GET /repos`, and the matching `ingest_repo` / `remove_repo` /
`list_repos` MCP tools. Each mutation clones or removes a checkout under `REPO_CACHE` and
re-runs the indexer, which mounts every cache subdirectory as an independent codebase.
A checkout is bounded by `REPO_MAX_BYTES` (default 2 GiB): the git process is killed once the
checkout grows past the cap, and a rejected clone leaves nothing behind. `POST /repos` answers
`507` without creating a job unless free space covers `REPO_DISK_HEADROOM_BYTES` (default 1 GiB)
plus one full-size checkout.
Both repo and document ingestion answer `202 {job_id, status_url}`; poll that URL for the
outcome.

Endpoints and credentials live in `.env` (gitignored): `LLM_URL`, `EMB_URL`, `RERANK_URL`, `DB_URL`, `LLM_MODEL`, `EMB_MODEL`, `RERANK_MODEL`, `DATA_ROOT`. Never hardcode them.

`DATA_ROOT` is the one host directory every container writes state into — `pgdata` (Postgres), `redis` (job state), `repos_cache` (git checkouts) and `cocoindex_state` (incremental ledger) are bound under it by `docker-compose.yml`. Compose refuses to start when it is unset rather than binding the host root. `REPO_CACHE` and `COCOINDEX_DB` are set by `docker-compose.yml` to container-local paths so the container never inherits a host location, and one ledger tracks one repo cache. Losing the repo cache or the ledger orphans `code_chunks` rows until the repos are re-added; losing the Redis directory drops job history only.

Logging is configured once per process via `memory_base.core.logger.setup_logging()`; modules log through `loguru` or stdlib `logging` (intercepted into the same sinks). `LOG_DIR` (optional, default `logs/`) sets the file-sink directory.

## Development work

- **Worktree → PR.** All work happens in a git worktree and lands via PR; `main` requires a PR and green CI (`PreToolUse(Bash)` guard denies `git commit`/`git push` on `main` — the agent cannot commit/push to `main` and must request the user to run it directly).
- **GitHub tracker in English.** Issues, issue comments, and PR titles/bodies are written in English (chat with the user is any language); enforced by the `pr-title` CI job.
- **Tests accompany behavior.** New or changed behavior ships its test in the same PR; the `test-guard` CI job enforces this (`skip-tests` label bypasses). Write the failing test first (`test:`), then implementation (`feat:`), then refactor if needed (`refactor:`).
- **Verify what you can verify before asking the user.** Anything observable (CLI output / DB state / MCP responses / logs) — verify yourself and attach proof to the PR's Runtime-evidence section; ask the user only for things that genuinely require them.
- **Comments: minimal, present-tense only.** Comment only what the code cannot say itself, in one line; no decision-history, spec-citation, or issue-number breadcrumbs. And Should use only English other language is not allowed.
- **Docs: current-state only.** Write what the system *is*, declaratively, matching the code — no change-narrative, no PR/issue numbers as prose, no dated changelogs, no future/unbuilt work (`PostToolUse` docs guard enforces this).

## Tracker & commit conventions

- **Issues and PRs use the `.github/` templates.** Open every issue from the matching template in `.github/ISSUE_TEMPLATE/` (bug · feature_task · spike) and fill `.github/PULL_REQUEST_TEMPLATE.md` for PRs.
- **No AI attribution.** Never append an "AI worked on this" trailer — `Co-Authored-By: Claude…`, `Generated with …`, `🤖`, "gpt-5.5 작성", or any equivalent — to commit messages or PR bodies. Write the message as the change itself. This overrides any default trailer the harness suggests.

## Agent skills

### Issue tracker

Issues live in this repo's GitHub Issues, operated via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage labels are used as-is: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
