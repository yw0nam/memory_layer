# Agent Guide

## Core principles

- **Low-signal data never reaches the DB.** Agent-authored notes arrive distilled through the MCP `save_memory` tool (validation, dedup, supersede) and are stored without embedding raw text. Documents go through chunking and LLM enrichment (atomize/summarize) before embedding. Raw transcripts and raw files are never embedded directly; only distilled, high-signal content is stored.
- **Source-specific knowledge lives only in source adapters (`adapters/`).** `ingest/`, the storage schema, retrieval (`retrieval/`), and consumers (`serve/`) are source-agnostic. The `memory_chunks` and `code_chunks` tables are the only contract between the write side and the read side — adding a new source must not change retrieval or serving code.
- **The REST API is the single backend.** Every consumer (MCP server, n8n, scripts) reaches stored chunks through it, never through the DB directly.
- **Assemble verified engines; don't reinvent.** Chunking/incremental indexing/vector storage/serving use proven components (CocoIndex, pgvector, FastMCP). Design decisions are grounded in web research and confirmed with the user — not invented ad hoc.
- **Endpoints and credentials live in `.env` (gitignored).** Never hardcode them.

## Working principles

- **No backward compatibility.** Delete unused paths instead of adding compat layers, fallbacks, or migrations.
- **Simplest implementation that fully meets current requirements.** No speculative abstractions, config values, or indirection layers.
- **Grow the system in layers.** Start from the smallest end-to-end working version and add features on top of a working result; never trade working code for unfinished complexity.
- **Components are modules with a clear separation of concerns.**
- **Check installed dependencies first** before hand-rolling or adding a package; never claim a library lacks a feature without reading its docs and types.
- **Permissive licenses only.** Dependencies and Postgres extensions carry MIT-class licenses (MIT / BSD / Apache-2.0 / PostgreSQL); no copyleft (GPL/AGPL) — commercial deployment must stay unencumbered.
- **Architecture decisions are long-term.** No stopgaps that only get past today and need replacing later.

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
