---
name: Feature / Task
about: Build work tied to a roadmap phase (Phase 0–4)
title: "feat: "
labels: ["feature"]
---

## Related phase / spec
<!-- Example: Phase 2 (MCP serving), docs/specs/impl_phase2_mcp.md -->
- Phase (0–4):
- Spec (`docs/specs/`):

## Work description
<!-- What is being built. One issue = one unit of work. -->

## Acceptance criteria
<!-- Concrete, verifiable. Copy from the spec when one exists. -->
- [ ]

## Dependencies
<!-- Prerequisite issues, schema/contract deliverables -->

## Checklist
- [ ] TDD order: failing test first (`test:`), then implementation (`feat:`), then refactor
- [ ] `memory_chunks` / retrieval contract changes are reflected in `docs/specs/` before the code
- [ ] No unverified assumptions (web/context7 cross-check, user confirmation for design decisions)
- [ ] Selective storage upheld — no write path bypasses triage/gates/distillation
