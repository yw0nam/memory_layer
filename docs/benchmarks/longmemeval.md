# LongMemEval

End-to-end memory QA on the agent-distilled write path, measured on a subset of
LongMemEval_S (Wu et al., 2024; MIT). An emulated agent distills each benchmark
session into notes, the production content gate judges every note, the kept notes
are stored through `save_note`, production search retrieves them, a model answers
from the retrieved notes alone, and a judge grades the answer with the official
LongMemEval grading prompts.

## Subset

`longmemeval_s_cleaned.json` holds 500 questions (30 abstention questions, id suffix
`_abs`). The harness scores a seeded sample of 100 (seed 0), allocated proportionally
per `question_type` by largest remainder with ties going to the larger type:

| question_type | questions |
|---|---|
| multi-session | 27 |
| temporal-reasoning | 27 |
| knowledge-update | 15 |
| single-session-user | 14 |
| single-session-assistant | 11 |
| single-session-preference | 6 |

Five of the 100 are abstention questions. The subset spans 4,742 extraction units: a
unit is a `(session_id, date)` pair, because the same session appears under more than
one date in some haystacks and each appearance is distilled on its own date.

## Pipeline

```
dataset json (path given on the command line)
   |
   | scripts/longmemeval/extract.py            emulated agent, outside the server
   v
notes.jsonl      one line per note: session_id, date, content, kind, gate, gate_reason
sessions.jsonl   one line per completed unit (zero-note units included): tokens, seconds
   |
   | memory_base.eval.longmemeval retrieve     throwaway Postgres, one namespace per question
   v
packets.jsonl    one line per question: load counts, hits with their benchmark sessions
   |
   | write-prompts --stage answer  ->  prompts/answer/<question_id>.txt
   | coordinator runs lme-answerer ->  replies/answer.jsonl
   | ingest --stage answer         ->  answers.jsonl
   | write-prompts --stage judge   ->  prompts/judge/<question_id>.txt
   | coordinator runs lme-judge    ->  replies/judge.jsonl
   | ingest --stage judge          ->  judgments.jsonl
   v
score            report.md / report.json and the manifest's score section
```

Every stage resumes: extraction skips units already in `sessions.jsonl` and discards a
partial last line or the notes of a unit that never completed; `retrieve` skips
questions already in `packets.jsonl`; `write-prompts` writes prompt files only for
questions without an accepted reply to their current prompt.

### Extraction

The extractor is `glm-5.3-flash` on the z.ai endpoint resolved from `.env`
(`resolve_llm_provider`; any other provider is refused), temperature 0, thinking
disabled, JSON output, with the committed prompt
`scripts/longmemeval/extract_prompt.txt`. Note kinds follow the `save_memory` contract
(`note`, `decision`, `episode`).

Each returned note then goes through the checks `save_note` applies before storing:
kind and length validation, the credential scan, and the content gate
(`judge_note_content`, the production judge prompt on the chat provider from `.env`).
The verdict is recorded as `stored` or `refused` with its reason. A gate call that fails
is retried three times with backoff; a unit whose gate stays unavailable is not written
and is retried by the next run, so no note carries an `unavailable` verdict. Refusals by
the gate's date-bound rule are part of the measured product.

### Retrieval

`retrieve` builds `db.Dockerfile`, starts it on tmpfs bound to `127.0.0.1`, points
`DB_URL` at it before the first connection, and removes it at the end; the deployment
database is never touched. Per question it registers namespace `lme-<question_id>` and
saves every `stored` note of the question's units in date order through `save_note`
with `occurred_at` set to the session date, a constant tag, and `allow_similar=True`
(a near-duplicate refusal would drop knowledge-update facts; each acknowledged
neighbour is counted as a similar ack). The gate is pinned open at load because its
verdict was recorded at extraction. The question then runs through
`search(question, source="memory", namespaces=[namespace])` with production rerank and
the production score floor.

A note id is a content hash, so identical notes from two sessions share one row; the
harness keeps a note-to-sessions map, so a hit counts toward every session it came from.

The search's time decay multiplies fused scores by `0.5^(age/90 days)` against wall
clock; for notes dated years back the ratio between two notes depends only on their
date difference, so ranking matches ranking at the question date. Decay affects only
which candidates reach the reranker; the floor applies to rerank scores.

### Answer and judge

Answering and judging run as Claude Code subagents, one fresh subagent per question,
from the committed tool-less agent definitions `.claude/agents/lme-answerer.md` and
`.claude/agents/lme-judge.md` (model alias `sonnet`, Claude Sonnet 5). The subagent
receives only the prompt file's text, never the dataset path. The coordinator appends
each reply as `{question_id, text, tool_uses, model}`, with `tool_uses` and `model`
taken from the task notification. `ingest` and `score` discard any row whose
`tool_uses` is missing or above `--baseline` (default 0); discarded questions get a new
prompt file on the next `write-prompts`.

The answer prompt is upstream's facts template from `src/generation/run_generation.py`,
with hits sorted by date and rendered in upstream's session-block format. The judge
prompts are copied byte-exact from `get_anscheck_prompt()` in
`src/evaluation/evaluate_qa.py`, chosen by `question_type` and the `_abs` suffix, and a
reply counts as correct when it contains `yes` after strip and lowercase, as upstream
parses it. Both are in `src/memory_base/eval/longmemeval_prompts.py`, pinned to upstream
commit `9e0b455f4ef0e2ab8f2e582289761153549043fc`. A judgment counts only when it
graded the current answer: each accepted row carries the sha256 of the prompt it
answered.

## Metrics

- **QA accuracy**: correct / judged, overall, per `question_type`, and over abstention
  questions (graded with the abstention prompt).
- **Retrieval**: upstream's session-level `recall_all@k` (every answer session among the
  first k distinct sessions) and `ndcg_any@k` for k = 5 and 10, over the ordered
  distinct benchmark sessions of the hits (at most `RERANK_TOP` hits survive rerank and
  the floor). The DCG is upstream's: rank 1 undiscounted and rank r >= 2 divided by
  log2(r). Abstention questions are excluded. Zero-hit packets are counted.
- **Write path**: notes per unit, refused-save rate overall and per `question_type`,
  refusals by cause (gate, validation, credential), similar acks, and extraction and
  gate token totals. The overall refused-save rate counts each unit once; a per-type
  rate counts a unit once per question of that type that contains it.

## Dated-embedding variant

`retrieve --variant dated` reruns the temporal-reasoning questions with each note
embedded as `"{date}: {content}"` (ISO date of its session) while the stored text stays
unchanged. The eval process rebinds `notes.embed_text` to a wrapper that reads the date
the loader binds around each `save_note` call; nothing in the server changes. The
variant also moves near-duplicate neighbour scores, so similar acks are reported for
both runs. Its packets, prompts, replies, and judgments carry a `-dated` suffix, and
`score` reports it beside the baseline.

## Disclosures

- The answerer and judge are Claude Code subagents: the Claude Code system prompt is
  present, and tool isolation rests on the tool-less agent definitions plus the
  `tool_uses` audit.
- Claude Sonnet 5 may have seen LongMemEval during training.
- The official LongMemEval judge is gpt-4o; these scores use a Claude judge and are not
  directly comparable with published LongMemEval numbers.

## Artefacts

The data directory (default `data/longmemeval/`) is gitignored: the filler sessions come
from ShareGPT, whose provenance is unclear. The committed manifest
`docs/benchmarks/longmemeval-manifest.json` records what a run was: dataset sha256, the
ordered question ids with per-type and abstention counts, code revision, upstream
commit, the sha256 of every prompt (extract, gate judge, answer, judge), model ids and
parameters (extractor, gate, embedder, reranker, answerer, judge as reported by the
harness), `NOTE_SIMILAR_THRESHOLD`, `MIN_SCORE`, `RERANK_TOP`, `FUSED_TOP`, the decay
half-life, the database image id and extensions, token totals, counts, the tool-use
audit, and the sha256 of every jsonl artefact.

## Cost

Measured per unit on two subset questions (86 units): about 2.7k extractor input
tokens, 2.8 notes, and 450 gate input tokens per gate call. Over the subset's 4,742
units that is about 13M extractor input tokens and 6M gate input tokens.

## Commands

```bash
uv run python scripts/longmemeval/extract.py --dataset PATH
uv run python -m memory_base.eval.longmemeval retrieve --dataset PATH
uv run python -m memory_base.eval.longmemeval retrieve --dataset PATH --variant dated
uv run python -m memory_base.eval.longmemeval write-prompts --dataset PATH --stage answer
uv run python -m memory_base.eval.longmemeval ingest --dataset PATH --stage answer
uv run python -m memory_base.eval.longmemeval write-prompts --dataset PATH --stage judge
uv run python -m memory_base.eval.longmemeval ingest --dataset PATH --stage judge
uv run python -m memory_base.eval.longmemeval score --dataset PATH
```

Every command takes `--data-dir` and `--manifest`; the prompt, ingest, and score
commands take `--variant dated` for the variant's files; `extract` and `retrieve` take
`--questions ID,ID` to run part of the subset.
