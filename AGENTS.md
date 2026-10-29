# Agent Guide

## Core principles

- **Notes arrive distilled; the server checks them deterministically.** Agent-authored notes arrive distilled by the client agent through the MCP `save_memory` tool, labelled `personal` or `work`; the server validates them, refuses a credential or a near-duplicate, applies supersede, and stores them without embedding raw text — what is worth keeping is the writer's call, guided by the tool description and the server instructions (ADR-0007). Documents go through deterministic chunking and a junk gate before embedding, with tags supplied by the uploader; the chat model is called in one place — the CSV branch, to summarize a sampled table into one card; the chat provider is chosen from API keys in `.env` (OpenCode Go, z.ai, OpenAI, Anthropic), falling back to the configured vLLM endpoint when none is set. Raw transcripts and raw files are never embedded directly. A tabular document's data rows are stored as structured rows in `doc_rows`, read by SQL only — never embedded, never in search results (ADR-0001). Agent-authored messages (the `messages` table: general signals and handoff snapshots) are rendered to canonical Markdown and stored unembedded; they are read by address and claimed once, never ranked (ADR-0002). The user's and the assistant's messages are kept verbatim in the `turns` table, one row per message, posted by the client's hook; they are read by time, session, and literal substring, never embedded, never ranked, never prefetched (ADR-0012).
- **Source-specific knowledge lives only in source adapters (`adapters/`).** `ingest/`, the storage schema, retrieval (`retrieval/`), and consumers (`serve/`) are source-agnostic. The `memory_chunks`, `code_chunks`, `doc_rows`, `messages`, and `turns` tables are the only contract between the write side and the read side — adding a new source must not change retrieval or serving code.
- **The REST API is the single backend.** Every consumer (MCP server, n8n, scripts) reaches stored chunks through it, never through the DB directly.
- **Assemble verified engines; don't reinvent.** Chunking/incremental indexing/vector storage/serving use proven components (CocoIndex, pgvector, FastMCP). Design decisions are confirmed with the user.
- **Endpoints and credentials live in `.env` (gitignored).** Never hardcode them.

## Working principles

- **Look at established products first.** Before designing a solution, look at how established products solve the same problem; adopt proven patterns and conventions instead of inventing approaches from scratch.
- **No backward compatibility.** Delete unused paths instead of adding compat layers, fallbacks, or migrations.
- **Simplest implementation that fully meets current requirements.** Write the least code that does the job while keeping functionality, readability, and project structure intact; every abstraction, config value, and indirection layer serves a current requirement.
- **Grow the system in layers.** Start from the smallest end-to-end working version and add features on top of a working result; each step leaves the system working.
- **Components are modules with a clear separation of concerns.** Core logic lives in its own module behind an explicit interface (inputs in, results out). `serve/api.py` and `serve/mcp_server.py` are the composing modules: they hold only the app, logging setup, middleware, lifespan, route table, FastMCP setup, server instructions, transport helpers, and tool registrations, and every handler lives in its feature module. Signs of logic in a composing module: handler bodies that query or mutate storage, module-level state read by nested functions, a startup routine that knows every component's internals.
- **Packages are layered; group files by feature inside a layer.** `src/memory_base/` is ordered `core` → `adapters` · `ingest` · `retrieval` → `serve` → `eval`; a package imports only from packages to its left (the middle three do not import each other). Inside `serve/`, a feature imports only `common/` and `access/`, except consolidation → notes and search → notes. A flat package that mixes several features and tells them apart by a name prefix or suffix (`*_api.py`) is an anti-pattern; in `serve/` each feature is one module or one package (`health.py` is a module), and a new feature gets its own.
- **Prefer proven, maintained libraries** when they lower overall complexity or raise stability. Check installed dependencies before hand-rolling or adding a package, and read a library's docs and types before concluding it lacks a feature.
- **Permissive licenses only.** Dependencies and Postgres extensions carry MIT-class licenses (MIT / BSD / Apache-2.0 / PostgreSQL); no copyleft (GPL/AGPL) — commercial deployment must stay unencumbered.
- **Architecture decisions are long-term.** Each change is the design the system keeps.

## Development work

- **Worktree → PR.** All work happens in a git worktree from `scripts/worktree.sh <branch>` (based on `origin/main`, with the primary `.env` linked) and lands via PR; `main` requires a PR and green CI. The `PreToolUse(Bash)` guard refuses `git commit`/`git push` on `main`; the user runs those.
- **GitHub tracker in English.** Issues, issue comments, and PR titles/bodies are written in English (chat with the user is any language); enforced by the `pr-title` CI job.
- **Tests accompany behavior.** New or changed behavior ships its test in the same PR; the `test-guard` CI job enforces this (`skip-tests` label bypasses). Write the failing test first (`test:`), then implementation (`feat:`), then refactor if needed (`refactor:`). Write only the test cases that confirm the behavior.
- **Tests never touch the deployment.** Unit tests fake the database and see an unreachable `DB_URL`. Integration tests (`@pytest.mark.integration`, local only, need docker) run against a throwaway Postgres container built from `db.Dockerfile` on tmpfs that the test session starts, prepares, and removes; every on-disk path (ingest spool, repo cache, CocoIndex state, logs) lives in a session tempdir. A test seeds the rows it reads instead of relying on existing data. The embedder and reranker are the configured live endpoints.
- **Verify what you can verify before asking the user.** Verify anything observable (CLI output / DB state / MCP responses / logs) yourself and attach the proof to the PR's Runtime-evidence section; `uv run python -m memory_base.eval.sandbox` serves the real REST API on a throwaway Postgres for that proof. Ask the user only for what needs them.
- **Comments: one line, present tense, English.** Comment only what the code cannot say itself, and state what the code does now; its history, spec, and issue number belong in the commit and PR.
- **Docs: current-state only.** Write what the system *is*, declaratively, matching the code — no change-narrative, no PR/issue numbers as prose, no future/unbuilt work (the `PostToolUse` hook and the CI `docs-guard` job enforce this through `scripts/ci/docs-guard.sh`). `CHANGELOG.md` is the one sanctioned exception: it is generated from git history, one section per released version, and is never hand-edited. Versioning is SemVer on `0.x`: `minor` for features and breaking changes, `patch` for fixes.

## Tracker & commit conventions

- **Issues and PRs use the `.github/` templates.** Open every issue from the matching template in `.github/ISSUE_TEMPLATE/` (bug · feature_task · spike) and fill `.github/PULL_REQUEST_TEMPLATE.md` for PRs, with one `Closes #N` line per issue the PR resolves.
- **No AI attribution.** Write each commit message and PR body as the change itself, ending at its last line of content. This overrides any default trailer the harness suggests (`Co-Authored-By: Claude…`, `Generated with …`, `🤖`, or an equivalent).

## Agent skills

### Issue tracker

Issues live in this repo's GitHub Issues, operated via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage labels are used as-is: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

`GLOSSARY.md` and `docs/adr/` at the repo root hold domain vocabulary and decisions,
created lazily as terms or decisions get resolved rather than scaffolded upfront.
See `docs/agents/domain.md`, which also lists every place a tool or REST contract is stated.
