# LongMemEval

End-to-end memory QA on the agent-distilled write path, measured on a subset of
LongMemEval_S (Wu et al., 2024; MIT). An emulated agent distills each benchmark
session into notes, the production content gate judges every note, the kept notes
are stored through `save_note`, production search retrieves them, a model answers
from the retrieved notes alone, and a judge grades the answer with the official
LongMemEval grading prompts. Extraction, answer, and judge run on `glm-5.3-flash`
(`--backend zai`, the default) or in a headless Claude Code session
(`--backend claude-code`, `claude-sonnet-5-5`); the content gate runs on the chat
provider from `.env`.

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
   | scripts/longmemeval/answer.py judge-hits  per-hit labels         -> hit-judgments.jsonl
   | memory_base.eval.longmemeval audit-sample 20 seeded judgments    -> judge-audit.jsonl
   v
score            report.md / report.json and the manifest's score section
frontier         coverage and junk per read cell, printed from a candidates run
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
content gate. Every note is judged by the personal-memory prompt, the one
`save_personal_memory` uses: it accepts memory of the user — facts about them and the
people around them, plans, dated episodes, preferences, and answers worth keeping — and
refuses work knowledge, filler, and generic advice. Gate-off loads refused notes as well, so it isolates extraction and
retrieval from the gate's policy. Notes refused by validation or the credential scan
stay out of gate-off too, because `save_note` never stores them. Each run writes its
own files (`packets-gate-off.jsonl`, `answers-gate-off.jsonl`, and so on; the dated run
uses `-dated`), and `score` reports every run present side by side.

`--read` selects how each run reads the search:

| read | search options | hits kept | run name |
|---|---|---|---|
| `search` (default) | production floor | first 10 | unprefixed |
| `prefetch` | `min_score=0.25` | first 5 | `prefetch-…` |
| `budget` | `budget_tokens=4000` | every packed hit | `budget-…` |
| `candidates` | a budget no packet reaches | every fused candidate, in rerank order | `candidates-…` |

A budget packet records its `budget_tokens`, and the manifest records the read setting
of every run.

The `candidates` read feeds `memory_base.eval.read_sweep`, which cuts every read-settings
grid point from the same candidates (see `read-settings.md`).

## Extraction

The extractor is `glm-5.3-flash` on the z.ai endpoint resolved from `.env`
(`resolve_llm_provider`; any other provider is refused), temperature 0, thinking
disabled, JSON output, with the committed prompt
`scripts/longmemeval/extract_prompt.txt`. Each note keeps the label the extractor gave it
(`note`, `decision`, `episode`); every note is judged with the personal-memory prompt and
loaded as kind `personal`. `--prompt digest` and `--prompt personal` run the
committed extraction prompts instead (`src/memory_base/eval/prompts/`), with each turn
rendered as `[index] role: text` and the replies parsed by
`memory_base.eval.extraction.parse_extraction`.

Each returned note then goes through the checks `save_note` applies before storing:
label and length validation, the credential scan, and the content gate
(`judge_note_content` with the production personal-memory prompt on the chat provider from `.env`).
The verdict is recorded as `stored` or `refused` with its reason. A gate call that fails
is retried three times with backoff; a unit whose gate stays unavailable is not written
and is retried by the next run, so no note carries an `unavailable` verdict. A session
the provider's content filter refuses (z.ai error code 1301) is recorded as a completed
unit with `provider_refused: "content_filter"` and contributes no notes. The
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
the run's read setting.

A note id hashes its content alone, so identical notes from two sessions share one row;
the harness keeps a note-to-sessions map, so a hit counts toward every session it came
from. A packet hit's date is the note's `occurred_at`.

Age enters the search only as the recency voter in the fusion, which ranks the
candidates by date; it is relative, so ranking against wall clock matches ranking at the
question date. The floor applies to rerank scores.

The dated run embeds each note as `"{date}: {content}"` (ISO date of its session) while
the stored text stays unchanged: the eval process rebinds `notes.embed_text` to a
wrapper that reads the date the loader binds around each `save_note` call, and nothing
in the server changes. It also moves near-duplicate neighbour scores, so similar acks
are reported for both runs.

### Agent writer

`--writer agent` puts an emulated client agent in front of each save, the way a client
agent decides supersedes; the decision stays in eval code and the server makes none.
Per note, in date order within the question:

```
note (content, session date)
   |
   | search(content, source="memory", namespaces=[namespace], min_score=MIN_SCORE)
   v
top 5 candidates, in rerank order: [i] (YYYY-MM-DD) text
   |  none -> plain save, no model call
   v
writer model (claude-sonnet-5-5, effort medium, headless Claude Code)
   |  {"action": "new"}  -> plain save
   |  {"action": "supersede", "index": i, "archive": [j, ...], "content": ...}
   |       -> save the rewrite with supersedes=<candidate i id>,
   |          then archive_rows([candidate j ids], namespaces=[namespace],
   v          archived_by="lme-writer")
LoadStats: agent_calls, superseded, archived, agent_errors
```

The rewrite states the current value and the previous value with their dates (for
example "20 dozen eggs as of 2023-05 (30 dozen as of 2023-01)") and keeps the rest of
the new note. `archive` is optional (default empty) and lists the other candidates that
state the same stale value as candidate i, such as one fact extracted from two sessions;
they are archived in the question namespace through `archive_rows`, the call behind the
`archive_notes` tool, once the rewrite is saved. An `archive` entry that is not a
candidate index, repeats, or equals `index` makes the reply malformed. A malformed reply
or a failed call is retried up to three times; a search that fails, three failed
attempts, or a rewrite that `save_note` refuses each count one agent error and save the
original note plainly (archiving nothing), so every note still reaches the store. An
archive call that fails counts one agent error and keeps the saved rewrite. A listed
candidate whose id is the rewrite's own (the rewritten text hashes to that row) is never
archived. The rewrite's id maps to its own session and to every session of the notes it
superseded or archived, so a hit on it counts toward the sessions of the values it
replaced.
`--writer-model` and `--writer-effort` choose another model or effort.

| counter | counts |
|---|---|
| `agent_calls` | writer model calls, retries included |
| `superseded` | notes saved as a rewrite with `supersedes` |
| `archived` | candidate rows archived as stale duplicates of a superseded note |
| `agent_errors` | notes saved plainly after a search, model, or rewrite failure, plus failed archive calls |

Every packet row records its writer (`{"kind": "plain"}`, or the agent's kind, model,
effort, and the sha256 of its prompts) and so does the manifest's retrieve section.
`retrieve` refuses to run over a packets file whose rows carry another writer config,
whether or not any question is pending. An agent run keeps its own data directory and
manifest: before running it, the operator creates the new data directory and symlinks
`notes.jsonl` and `sessions.jsonl` into it from the baseline data directory, and passes a
manifest inside the new directory (for example `data/longmemeval-agent/manifest.json`),
so its packets, answers, judgments, and retrieve section stay apart from the baseline's. The writer searches raw
note text, so it runs on the baseline and gate-off variants; `--variant dated` is refused.

## Answer and judge

`scripts/longmemeval/answer.py` is a client like the extractor: it builds its model
client from `resolve_llm_provider`, requires the z.ai provider, and calls
`glm-5.3-flash` with temperature 0 and thinking disabled, five questions at a time.
An answer keeps upstream's 500-token limit. A judgment gets 200 tokens instead of
upstream's 10: `glm-5.3-flash` sometimes emits reasoning even with thinking disabled,
and at 10 tokens that reasoning can use the whole budget and leave the reply empty.
An empty reply is retried and never recorded. Each question gets one fresh
single-message request; each reply is appended with the sha256 of its prompt, its
token usage, and its latency.

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

## Hit judge

`answer.py judge-hits` labels the first ten hits of every packet of a run, normally the
candidates run (`--gate off --read candidates`), in one call per question: the question
and its date, the reference answer, and the notes in rerank order with their dates. The
default judge is `claude-sonnet-5-5` at effort medium (`--backend claude-code`); the zai
backend sends the same instructions as a system turn. Each note gets one label:

| label | definition |
|---|---|
| useful | contains information a careful answerer would use to produce the reference answer (a fact the answer states, a date needed to compute it, or a fact that rules out a wrong answer). |
| related | not needed for this answer, but about the user and the same subject the question asks about (the same entity, activity, or category), so it is sensible context that would not push the answerer toward a wrong answer. |
| misleading | would push a careful answerer toward a wrong answer: it contradicts the reference answer, states a value the answer has superseded, or looks like it answers the question but does not (the wrong event, person, item, or time window). |
| unrelated | about a different subject than the question. |

The judge also names `min_prefix`, the smallest n such that notes 0..n-1 together are
enough for a careful answerer to produce the reference answer, or null when all the notes
together are not. A reply whose labels miss or add an index, use another label, or give
a `min_prefix` outside 1..n is malformed and retried, up to three calls. Each question
appends one row to `hit-judgments[-run].jsonl` with the judged hit `texts`, `labels`,
`min_prefix`, the model, the prompt sha256, tokens, and seconds; a question that fails
three times appends an error row. Abstention questions and zero-hit packets are skipped
(the frontier counts a zero-hit packet as judged with no labels and no `min_prefix`), and a
rerun skips every question whose packet already has a non-error row for the same hit
texts.

**Evidence coverage** is the primary recall metric: the share of questions whose
delivered notes contain enough to produce the reference answer (the delivered count
reaches `min_prefix`). Session-level `recall_all` is a reference number. A longer prefix
can also add misleading notes, so coverage is not answer accuracy.

## Frontier

`frontier` reads a run's packets and hit judgments and prints one Markdown table; it
writes no file. The question set is every non-abstention packet that has no hits or has a
non-error judgment row for its current first-ten hit texts; a zero-hit packet counts as
judged with no labels and no `min_prefix`, so it is not covered, adds nothing to the junk
shares, and has recall 0. The header counts the questions excluded for no judgment, a
stale judgment (rows only for other hit texts), and abstention.

Each cell of `top_k` in (1, 2, 3, 5, 10) by `floor` in (0, 0.05, 0.1, 0.25, 0.4, 0.5)
delivers the hits among the first `top_k` whose score reaches the floor, a prefix of the
judged hits because candidates come in descending score order:

| field | value |
|---|---|
| `coverage` | share of questions whose delivered count reaches `min_prefix`; a question with no delivered hit, zero-hit packets included, is not covered |
| `coverage_agg` / `coverage_lookup` | coverage over multi-session, temporal-reasoning, and knowledge-update questions / over the other types |
| `junk` | misleading plus unrelated labels over all delivered hits, summed across questions; empty when nothing is delivered |
| `misleading` / `related` | the same share for one label |
| `recall_all` | mean session-level recall of the delivered hits over their distinct sessions (reference) |
| `hits_per_question` | mean delivered hits |

Cells sort by `junk` ascending (empty last), then `coverage` descending, then
`hits_per_question` ascending. The best cell is the one with the highest `coverage` among
the cells whose `junk` is at most 0.10, the fewer hits per question breaking a tie.

## Metrics

- **QA accuracy**: correct / judged, overall, per `question_type`, and over abstention
  questions (graded with the abstention prompt).
- **Retrieval**: upstream's session-level `recall_all@k` (every answer session among the
  first k distinct sessions) and `ndcg_any@k` for k = 5 and 10, over the ordered
  distinct benchmark sessions of the hits (at most `RERANK_TOP` hits survive rerank and
  the floor under the `search` read; a `budget` read keeps as many as fit the budget). The DCG is upstream's: rank 1 undiscounted and rank r >= 2 divided by
  log2(r). Abstention questions are excluded. Zero-hit packets are counted.
- **Evidence coverage**: the hit judge's share of questions whose delivered notes suffice
  for the reference answer, per read cell of the frontier (see Hit judge).
- **Write path**: notes per unit, refused-save rate overall and per `question_type`,
  refusals by cause (gate, validation, credential), similar acks, and extraction and
  gate token totals. The overall refused-save rate counts each unit once; a per-type
  rate counts a unit once per question of that type that contains it.
- **Model usage**: answer and judge input and output tokens and seconds per question.
- **Judge agreement**: the hand-audit agreement rate.

## Results

The 100-question subset with the gate off, one pass of every stage, answered and judged
by `claude-sonnet-5-5` (`answer.py --backend claude-code --effort high`). Notes were
extracted by Sonnet 5.5 under the personal extraction policy with the gate off, and
budget-mode recall counts about 12.6 distinct sessions per packet against 1.6 under
search, so the budget row and the search rows are not one-to-one comparable.

| run | read | FUSED_TOP | recall_all@10 | ndcg_any@10 | QA accuracy | multi-session QA | mean hits per packet | mean packet chars |
|---|---|---|---|---|---|---|---|---|
| search, 90-day age decay before the reranker | top 10, floor 0.25 | 20 | 0.663 | 0.809 | 0.74 | 0.667 | 1.84 | 653 |
| search | top 10, floor 0.25 | 20 | 0.779 | 0.863 | 0.81 | 0.667 | 1.98 | 699 |
| budget (shipped) | `budget_tokens=4000` | 40 | 0.926 | 0.927 | 0.91 | 0.926 | 36.21 | 15,573 |

### QA accuracy

| question_type | questions | search, decay | search | budget |
|---|---|---|---|---|
| overall | 100 | 0.740 | 0.810 | 0.910 |
| knowledge-update | 15 | 0.667 | 1.000 | 0.933 |
| multi-session | 27 | 0.667 | 0.667 | 0.926 |
| single-session-assistant | 11 | 0.455 | 0.545 | 0.545 |
| single-session-preference | 6 | 1.000 | 1.000 | 1.000 |
| single-session-user | 14 | 1.000 | 1.000 | 1.000 |
| temporal-reasoning | 27 | 0.778 | 0.815 | 0.963 |
| abstention (included above) | 5 | 1.000 | 1.000 | 0.800 |

### Retrieval

Session-level `recall_all@10` / `ndcg_any@10` over the 95 non-abstention questions.

| question_type | questions | search, decay | search | budget |
|---|---|---|---|---|
| overall | 95 | 0.663 / 0.809 | 0.779 / 0.863 | 0.926 / 0.927 |
| knowledge-update | 14 | 0.143 / 0.571 | 0.857 / 0.929 | 1.000 / 1.000 |
| multi-session | 24 | 0.667 / 0.878 | 0.667 / 0.878 | 0.958 / 0.975 |
| single-session-assistant | 11 | 0.455 / 0.455 | 0.455 / 0.455 | 0.455 / 0.455 |
| single-session-preference | 6 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| single-session-user | 14 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| temporal-reasoning | 26 | 0.769 / 0.875 | 0.808 / 0.880 | 1.000 / 0.986 |

| | search, decay | search | budget |
|---|---|---|---|
| zero-hit packets (of 100) | 10 | 10 | 0 |
| answer input tokens | 83,441 | 85,645 | 726,934 |

The budget packs at most `FUSED_TOP` reranked candidates: 86 of the 100 budget packets
reach 3,800 estimated tokens or more, and hit counts range from 30 to 40. At
`FUSED_TOP=20` the same budget returns all 20 candidates for every question, with
identical `recall_all@10` and `ndcg_any@10` and QA 0.89 (multi-session 0.852).

## Comparability

The official LongMemEval judge is gpt-4o, and published numbers are graded by it. This
harness answers and judges with `glm-5.3-flash`, so its accuracies are not directly
comparable with published ones; the judge audit reports how often this judge agrees
with a person on the run. `glm-5.3-flash` may have seen LongMemEval during training.

## Artefacts

The data directory (default `data/longmemeval/`) is gitignored: the filler sessions come
from ShareGPT, whose provenance is unclear. The committed manifest
`docs/benchmarks/longmemeval-manifest.json` records what a run was: dataset sha256, the
ordered question ids with per-type and abstention counts, code revision, upstream
commit, the sha256 of every prompt (extract, gate judge, answer, judge), model ids and
parameters (extractor, gate, embedder, reranker, answerer, judge), `NOTE_SIMILAR_THRESHOLD`,
`MIN_SCORE`, `RERANK_TOP`, `FUSED_TOP`, the database image id and
extensions, token totals for every model stage, counts, the judge agreement, and the
sha256 of every jsonl artefact.

## Cost

Measured token totals for the subset:

| stage | calls | input tokens | output tokens |
|---|---|---|---|
| extraction | 4,742 units (11 retries) | 12,196,913 | 638,142 |
| content gate | 12,932 (132 retries) | 5,768,264 | 929,510 |
| answer, gate-on / gate-off / dated | 100 / 100 / 27 | 18,668 / 20,403 / 5,840 | 9,951 / 9,498 / 2,843 |
| judge, gate-on / gate-off / dated | 100 / 100 / 27 | 23,516 / 23,091 / 7,459 | 588 / 553 / 105 |

A unit takes 14.4 s on average at concurrency 5 (4.2 s of it the extraction call, the
rest its gate calls). An answer takes 3.3 to 4.1 s and a judgment about 1.6 s.

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
uv run python -m memory_base.eval.longmemeval retrieve --dataset PATH --gate off --read candidates \
    --writer agent --data-dir data/longmemeval-agent --manifest data/longmemeval-agent/manifest.json
uv run python scripts/longmemeval/answer.py judge-hits --dataset PATH --gate off --read candidates --backend claude-code
uv run python -m memory_base.eval.longmemeval frontier --dataset PATH --gate off --read candidates
```

`answer.py` and `audit-sample` take `--gate off` or `--variant dated` for those runs;
every command takes `--data-dir` and `--manifest`; `extract` and `retrieve` take
`--questions ID,ID` to run part of the subset.
