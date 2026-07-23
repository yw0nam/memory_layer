# Implementation spec: English-only policy (Issue #20)

## Principle

Everything the system **stores or emits** is English, and everything the repo
**contains** is English — even when user/agent input arrives in Korean or
Japanese. Input handling stays multilingual: transcripts are parsed, tokenized
(Hangul token branch in `TOKEN_RE` stays, rewritten with `가-힣`
escapes), embedded, and reranked in any language.

## Code changes (TDD)

### ingest/history.py
- Distillation prompt: distill **in English** (summary, one_line_question,
  resolution, decisions), keeping code, error strings, and identifiers
  verbatim; explicitly instructs English output even for Korean/Japanese
  transcripts.
- Triage LLM prompt: English, `reason` in English.
- `parse_distillation` default resolution `"unresolved"` (was `"미해결"`).
- Decision row prefix `Decision:` (was `결정:`).
- `TOKEN_RE`: Hangul range via unicode escapes (behavior identical).

### serve/answer.py
- `PLANNER_SYSTEM_PROMPT`: English; queries MUST be English regardless of the
  question language (fixes the FTS leg: 'simple' FTS cannot match Korean
  tokens against English rows; vector/rerank legs are unaffected).
- `SYNTHESIS_SYSTEM_PROMPT`: English; answers in English.
- User-facing literals (evidence block labels, no-evidence message, CLI
  docstring): English.

### serve/mcp_server.py
- Remaining Korean docstrings/comments: English.

## Repo sweep

- Translate the Korean spec docs (spec.md, impl_phase1_history.md,
  impl_phase2_mcp.md, impl_selective_ingest.md, impl_tests_coverage.md,
  impl_weighted_gate_decisions.md, impl_mcp_sse_docker.md) to English,
  preserving structure and technical content; current-state style.
- tests/: English test names, comments, and fixture strings — except fixtures
  that deliberately exercise Korean **input** handling (Hangul tokenization,
  Korean transcript parsing), which keep minimal Hangul data with English
  test names and comments.

## Guard test (permanent)

`tests/test_english_only.py`: scans `src/**/*.py` and `docs/specs/*.md` for
Hangul (U+AC00–U+D7AF, U+1100–U+11FF, U+3130–U+318F) and Kana
(U+3040–U+30FF) codepoints; fails listing file:line on any hit. No allowlist:
source keeps Hangul out via unicode escapes; docs are fully English.

## Acceptance criteria

1. Red → green. Guard test passes on the whole repo.
2. Integration (LLM): a Korean fixture session distilled via the real LLM
   produces English output (Hangul ratio ≈ 0 outside preserved identifiers).
3. Integration (cross-lingual recall): a Korean question through
   `answer.plan()` yields ASCII search queries, and retrieval over
   English-distilled rows returns relevant hits.
4. Full suite + ruff clean.

## Forbidden

Behavior changes beyond language: gate values, row shapes, retrieval logic,
schema. Deleting the Hangul token branch (input stays multilingual).
