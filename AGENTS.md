# Agent Guide

## Core principles

- **Low-signal data never reaches the DB.** Agent-authored notes arrive already distilled through the MCP `save_memory` tool (validation, dedup, supersede) and are stored without embedding raw text. Documents go through chunking and LLM enrichment (atomize/summarize) before embedding. Raw transcripts and raw files are never embedded directly; only distilled, high-signal content is stored.
- **Source-specific knowledge lives only in source adapters (`adapters/`).** `ingest/`, the storage schema, retrieval (`retrieval/`), and consumers (`serve/`) are source-agnostic. The `memory_chunks` table is the single contract between the write side and the read side — adding a new source must not change retrieval or serving code.
- **Assemble verified engines; don't reinvent.** Chunking/incremental indexing/vector storage/serving use proven components (CocoIndex, pgvector, FastMCP). Design decisions are grounded in web research and confirmed with the user — not invented ad hoc.

## Project structure

The REST API is the single backend: every consumer (MCP server, n8n, scripts) reaches
`memory_chunks` through it, never through the DB directly.

```
src/memory_base/
  common.py           # shared constants + LLM/embedding clients (vLLM, OpenAI-compatible)
  schema.py           # memory_chunks + retrieval-log DDL
  ingest/
    enrich.py         # generic JSON-mode enrichment for stored content
    code.py           # CocoIndex app: repo code chunking + embedding
  adapters/
    document.py       # document conversion, chunking, CSV sampling, storage-row mapping
    document_worker.py# killable MarkItDown conversion worker
  retrieval/
    search.py         # hybrid search: FTS + vector + IDF + time decay → RRF → rerank
    decompose.py      # knowledge-aware decomposition: multi-hop retrieval over memory atoms
  serve/
    api.py            # Starlette REST API (search, deep search, save, admin) — the backend
    ingest_api.py     # bounded async document-ingestion orchestration
    notes.py          # validation + storage for agent-authored notes
    admin.py          # memory lifecycle operations (duplicates, archive, restore)
    access_log.py     # best-effort persistence of retrieval activity
    mcp_server.py     # MCP server over REST (stdio | SSE | streamable-http), Docker serves SSE
  eval/
    retrieval.py      # reproducible retrieval evaluation with an atom-lane A/B report
tests/                # pytest; DB/vLLM-dependent tests carry the `integration` marker
scripts/ci/           # CI helper scripts (test-guard)
```

## Commands

```bash
uv sync                                              # install deps + editable package
uv run pytest                                        # all tests (integration skips if DB is down)
uv run pytest -m "not integration"                   # unit tests only (what CI runs)
uv run ruff format --check . && uv run ruff check .  # lint (ruff is the only Python linter)
docker compose up -d db                              # pgvector on localhost:5439
docker compose up -d --build api                     # REST backend on :8010
docker compose up -d --build mcp                     # MCP server, SSE on :8765
uv run cocoindex update src/memory_base/ingest/code.py   # (re)index repo code
claude mcp add --transport sse memory-base http://localhost:8765/sse
```

Endpoints and credentials live in `.env` (gitignored): `LLM_URL`, `EMB_URL`, `RERANK_URL`, `DB_URL`, `COCOINDEX_DB`, `LLM_MODEL`, `EMB_MODEL`, `RERANK_MODEL`. Never hardcode them.

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
