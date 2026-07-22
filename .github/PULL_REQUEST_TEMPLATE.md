<!-- Title in English, conventional-commit format (becomes the squash-merge subject). e.g. "feat: selective ingest gates" -->

## Summary
<!-- What and why -->

## Related issues
<!-- Closes #__ -->

## Related specs / docs
<!-- Touched docs: docs/specs/... -->

## Runtime evidence (required for ingest / retrieval / serving behavior)
<!-- Unit tests are not runtime verification. For any behavioral change, paste
     the proving output: ingest stats line, search results for a real query,
     an MCP client call, or relevant logs. State "N/A — no runtime behavior
     change" only when the diff is non-runtime (docs, config, tooling). -->

## Verification
- [ ] `uv run pytest` passes (integration tests run locally when DB/vLLM are reachable)
- [ ] `uv run ruff format --check .` and `uv run ruff check .` pass
- [ ] `docker compose build mcp` passes (Dockerfile / dependency changes)
- [ ] Runtime evidence attached above, or N/A justified

## Checklist (core principles)
- [ ] Selective storage — new write paths go through triage/gates/distillation; raw transcripts are never embedded
- [ ] Source-specific logic stays out of the source-agnostic core (ingest pipeline, retrieval, serving)
- [ ] `memory_chunks` schema changes update `docs/specs/` and the retrieval contract alongside the code
- [ ] No hardcoded endpoints/models — configuration lives in `.env`
- [ ] New/changed behavior ships a test in this PR (or `skip-tests` label justified)
