# LongMemEval

End-to-end memory QA on the agent-distilled write path, measured on a subset of
LongMemEval_S (Wu et al., 2024; MIT). An emulated agent distills each benchmark
session into notes, the production content gate judges every note, the kept notes
are stored through `save_note`, production search retrieves them, a model answers
from the retrieved notes alone, and a judge grades the answer with the official
LongMemEval grading prompts. Every model stage uses one open-provider model,
`glm-5.3-flash`.

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
   | scripts/longmemeval/answer.py answer      upstream answer prompt -> answers.jsonl
   | scripts/longmemeval/answer.py judge       upstream judge prompt  -> judgments.jsonl
   | memory_base.eval.longmemeval audit-sample 20 seeded judgments    -> judge-audit.jsonl
   v
score            report.md / report.json and the manifest's score section
```

Every stage resumes: extraction skips units already in `sessions.jsonl` and discards a
partial last line or the notes of a unit that never completed; `retrieve` skips
questions already in `packets.jsonl`; `answer` and `judge` skip questions that already
have a row for their current prompt.

## Runs

| run | command flag | questions | notes loaded |
|---|---|---|---|
| gate-on (baseline) | none | all 100 | gate-stored |
| gate-off | `--gate off` | all 100 | gate-stored and gate-refused |
| dated | `--variant dated` | the 27 temporal-reasoning | gate-stored, embedded as `"{date}: {content}"` |

Gate-on is the product number: it measures what an agent's memory holds after the
content gate. The gate is tuned for coding-agent memory, whose notes must record
something no tracker, repository, or file already holds; it refuses personal plans
(for example, "the user is considering a bachata festival next month") and records
of what the assistant recommended in a past session, and LongMemEval asks about
both. Gate-off loads those refused notes as well, so it isolates extraction and
retrieval from the gate's policy. Notes refused by validation or the credential scan
stay out of gate-off too, because `save_note` never stores them. Each run writes its
own files (`packets-gate-off.jsonl`, `answers-gate-off.jsonl`, and so on; the dated run
uses `-dated`), and `score` reports every run present side by side.

## Extraction

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
and is retried by the next run, so no note carries an `unavailable` verdict. The
production gate call sets no temperature, so its verdicts vary between runs; recording
them once fixes them for every later stage.

## Retrieval

`retrieve` builds `db.Dockerfile`, starts it on tmpfs bound to `127.0.0.1`, points
`DB_URL` at it before the first connection, and removes it at the end; the deployment
database is never touched. Per question it registers namespace `lme-<question_id>` and
saves the run's notes of the question's units in date order through `save_note` with
`occurred_at` set to the session date, a constant tag, and `allow_similar=True` (a
near-duplicate refusal would drop knowledge-update facts; each acknowledged neighbour is
counted as a similar ack). The gate is pinned open at load because its verdict was
recorded at extraction. The question then runs through
`search(question, source="memory", namespaces=[namespace])` with production rerank and
the production score floor.

A note id is a content hash, so identical notes from two sessions share one row; the
harness keeps a note-to-sessions map, so a hit counts toward every session it came from.

The search's time decay multiplies fused scores by `0.5^(age/90 days)` against wall
clock; for notes dated years back the ratio between two notes depends only on their
date difference, so ranking matches ranking at the question date. Decay affects only
which candidates reach the reranker; the floor applies to rerank scores.

The dated run embeds each note as `"{date}: {content}"` (ISO date of its session) while
the stored text stays unchanged: the eval process rebinds `notes.embed_text` to a
wrapper that reads the date the loader binds around each `save_note` call, and nothing
in the server changes. It also moves near-duplicate neighbour scores, so similar acks
are reported for both runs.

## Answer and judge

`scripts/longmemeval/answer.py` is a client like the extractor: it builds its model
client from `resolve_llm_provider`, requires the z.ai provider, and calls
`glm-5.3-flash` with temperature 0, thinking disabled, and upstream's token limits
(500 for an answer, 10 for a judgment), five questions at a time. Each question gets
one fresh single-message request; each reply is appended with the sha256 of its prompt,
its token usage, and its latency.

The answer prompt is upstream's facts template from `src/generation/run_generation.py`,
with hits sorted by date and rendered in upstream's session-block format. The judge
prompts are copied byte-exact from `get_anscheck_prompt()` in
`src/evaluation/evaluate_qa.py`, chosen by `question_type` and the `_abs` suffix, and a
reply counts as correct when it contains `yes` after strip and lowercase, as upstream
parses it. Both are in `src/memory_base/eval/longmemeval_prompts.py`, pinned to upstream
commit `9e0b455f4ef0e2ab8f2e582289761153549043fc`. A judgment counts only when it
graded the current answer; the latest row per prompt wins.

### Judge audit

The official LongMemEval judge is gpt-4o; this harness judges with `glm-5.3-flash`.
`audit-sample` draws 20 judgments at random (seed 0) into `judge-audit.jsonl` with the
question, reference answer, model response, judge reply, and judge label. A person sets
each row's `human_label` to `true` or `false`, and `score` reports the agreement rate
between those labels and the judge over the labeled rows, per run, in the report and
the manifest.

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
- **Model usage**: answer and judge input and output tokens and seconds per question.
- **Judge agreement**: the hand-audit agreement rate.

## Comparability

The official LongMemEval judge is gpt-4o, and published numbers are graded by it. This
harness answers and judges with `glm-5.3-flash`, so its accuracies are not directly
comparable with published ones; the judge audit reports how often this judge agrees
with a person on the run.

## Artefacts

The data directory (default `data/longmemeval/`) is gitignored: the filler sessions come
from ShareGPT, whose provenance is unclear. The committed manifest
`docs/benchmarks/longmemeval-manifest.json` records what a run was: dataset sha256, the
ordered question ids with per-type and abstention counts, code revision, upstream
commit, the sha256 of every prompt (extract, gate judge, answer, judge), model ids and
parameters (extractor, gate, embedder, reranker, answerer, judge), `NOTE_SIMILAR_THRESHOLD`,
`MIN_SCORE`, `RERANK_TOP`, `FUSED_TOP`, the decay half-life, the database image id and
extensions, token totals for every model stage, counts, the judge agreement, and the
sha256 of every jsonl artefact.

## Cost

Measured per unit on two subset questions (86 units): about 2.7k extractor input
tokens, 2.8 notes, and 450 gate input tokens per gate call. Over the subset's 4,742
units that is about 13M extractor input tokens and 6M gate input tokens.

## Commands

```bash
uv run python scripts/longmemeval/extract.py --dataset PATH
uv run python -m memory_base.eval.longmemeval retrieve --dataset PATH
uv run python -m memory_base.eval.longmemeval retrieve --dataset PATH --gate off
uv run python -m memory_base.eval.longmemeval retrieve --dataset PATH --variant dated
uv run python scripts/longmemeval/answer.py answer --dataset PATH
uv run python scripts/longmemeval/answer.py judge --dataset PATH
uv run python -m memory_base.eval.longmemeval audit-sample --dataset PATH
uv run python -m memory_base.eval.longmemeval score --dataset PATH
```

`answer.py` and `audit-sample` take `--gate off` or `--variant dated` for those runs;
every command takes `--data-dir` and `--manifest`; `extract` and `retrieve` take
`--questions ID,ID` to run part of the subset.
