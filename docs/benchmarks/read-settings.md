# Read settings

How `top_k`, `min_score`, and `budget_tokens` trade answer accuracy against injected tokens
and junk on the two read paths, measured on the LongMemEval_S personal corpus and on a
snapshot of the deployed corpus with `memory_base.eval.read_sweep`.

## Read paths

| path | caller | shipped read | what decides it |
|---|---|---|---|
| A. search | an agent calling `search_memory` while answering | `top_k=10`, floor `MIN_SCORE` 0.25 (`min_score` omitted); `budget_tokens=4000` on request | answer accuracy, then tokens |
| B. prefetch, Hermes | `integrations/hermes/memory_base`, every turn | `top_k=5`, `min_score=0.25`, body cut at `PREFETCH_CHAR_BUDGET=2000` characters | tokens per turn and junk on personal memory; most turns carry no memory intent |

## Method

```
query ──search(budget_tokens=10**9)──> every fused candidate (BUDGET_FUSED_TOP=40), rerank order
                                              │
          grid point, production helpers      │  _apply_min_score / _pack_budget
                                              v
    top_k mode : first RERANK_TOP=10 -> floor -> first top_k
    budget mode: floor -> pack until the chars/4 estimate passes budget_tokens
                                              │
          read path delivery                  v
    search      : every cut hit, tokens = sum of estimate_tokens
    hermes      : MemoryBaseClient.build_prefetch,    tokens = block chars / 4
```

One search per query feeds every grid point, so all points see the same candidates and
rerank scores. A budget point with a floor drops candidates below the floor before packing;
production `search()` ignores `min_score` in budget mode, so the budget rows with a floor
describe that packing rule rather than a shipped read. The delivery step runs the Hermes
client's rendering code: a hit that reached the prompt counts, a hit the client cut does not.

Grid: `top_k` in {3, 5, 10} x `min_score` in {0, 0.25, 0.4, 0.6}; `budget_tokens` in
{800, 1500, 2500, 4000} x `min_score` in {0, 0.25, 0.4}. A setting is named `k<top_k>-f<floor>`
or `b<budget>-f<floor>`.

With 40 fused candidates reranked in both modes, a real `longmemeval retrieve` on ten
questions returns the derived hits exactly for `k5-f0.6` and `k10-f0.25` (10 of 10 packets identical). For `b4000-f0`, 4 of 10 packets are
identical and the rest share 29 to 38 of their 30 to 38 hits: the reranker's scores below
0.01 vary between calls, which reorders the packed tail.

## Corpora

| corpus | content | queries | measures |
|---|---|---|---|
| LME personal | LongMemEval_S subset, 100 questions, notes extracted by Sonnet 5.5, every extracted note the save path accepts loaded, one namespace per question, throwaway Postgres | the 100 benchmark questions | recall_all, ndcg_any@10, tokens, junk share, zero-hit rate, QA |
| LME probe | the namespace of question `c5e8278d` (first of the subset) | 40 off-topic coding prompts, 20 coding memory-intent prompts (`tests/fixtures/read_sweep_probes.jsonl`) | off-topic fire rate |
| deployed | `memory.memory_chunks` copied over a `default_transaction_read_only` connection into a throwaway Postgres on the current schema: 334 rows, 178 active (141 `default`, 37 `personal`); search spans every namespace | the 34 labelled notes-replay queries (`tests/fixtures/retrieval_eval_notes.jsonl`: 25 scored, 9 expect-empty) and the 60 probe prompts | recall@5, MRR@10, expect-empty passes, off-topic fire rate and tokens |

Metric definitions:

| metric | definition |
|---|---|
| recall_all | a question counts when every answer session is among the sessions of the delivered hits; abstention questions excluded |
| recall_all@10 | the same over the first 10 distinct sessions (upstream definition) |
| mean tokens | mean estimated tokens (characters / 4) a packet puts into the prompt |
| junk share | delivered hits from no answer session, over all delivered hits of scored questions |
| zero-hit rate | packets with no delivered hit, over the 100 questions |
| off-topic fire | share of the 40 off-topic prompts that deliver at least one hit; every such hit is junk |
| deployed R@5 / MRR@10 | notes-replay metrics over the 25 scored labels |
| QA | Sonnet 5.5 answer and judge (`answer.py --backend claude-code --model claude-sonnet-5-5 --effort high`), 100 questions, one pass |

## Rerank scores

The reranker's scores are bimodal on both corpora: 92.8 % of LME candidates and 76.3 % of
deployed candidates score below 0.01, and 4.0 % and 4.3 % score 0.6 or more. Answer notes
for aggregate questions (multi-session, temporal) often score below 0.25 because each note
holds one part of the answer, so any floor removes them. The coding probe prompts never
score above 0.06 on the LME personal corpus, so every floor from 0.25 up rejects all of them
there.

## A. search

| setting | recall_all | recall_all@10 | mean tokens | junk share | zero-hit rate | deployed R@5 | deployed MRR@10 | deployed expect-empty | deployed off-topic fire | deployed off-topic tokens | LME off-topic fire | QA | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k3-f0 | 0.811 | 0.811 | 268 | 0.232 | 0.00 | 0.840 | 0.793 | 0/9 | 1.000 | 487 | 1.00 | - |  |
| k3-f0.25 | 0.747 | 0.747 | 156 | 0.028 | 0.10 | 0.800 | 0.753 | 5/9 | 0.125 | 35 | 0.00 | - |  |
| k3-f0.4 | 0.705 | 0.705 | 146 | 0.024 | 0.11 | 0.760 | 0.713 | 6/9 | 0.050 | 11 | 0.00 | - |  |
| k3-f0.6 | 0.642 | 0.642 | 130 | 0.000 | 0.14 | 0.740 | 0.693 | 6/9 | 0.025 | 8 | 0.00 | - |  |
| k5-f0 | 0.874 | 0.874 | 443 | 0.387 | 0.00 | 0.913 | 0.813 | 0/9 | 1.000 | 752 | 1.00 | - |  |
| k5-f0.25 | 0.779 | 0.779 | 171 | 0.026 | 0.10 | 0.863 | 0.773 | 5/9 | 0.125 | 35 | 0.00 | - |  |
| k5-f0.4 | 0.737 | 0.737 | 159 | 0.022 | 0.11 | 0.800 | 0.723 | 6/9 | 0.050 | 11 | 0.00 | - |  |
| k5-f0.6 | 0.653 | 0.653 | 137 | 0.000 | 0.14 | 0.780 | 0.703 | 6/9 | 0.025 | 8 | 0.00 | - |  |
| k10-f0 | 0.926 | 0.926 | 850 | 0.609 | 0.00 | 0.913 | 0.819 | 0/9 | 1.000 | 1443 | 1.00 | 0.90 | selected: QA 0.90 at 850 tokens |
| k10-f0.25 | 0.779 | 0.779 | 174 | 0.026 | 0.10 | 0.863 | 0.779 | 5/9 | 0.125 | 35 | 0.00 | - | search_memory default read |
| k10-f0.4 | 0.737 | 0.737 | 160 | 0.022 | 0.11 | 0.800 | 0.729 | 6/9 | 0.050 | 11 | 0.00 | - |  |
| k10-f0.6 | 0.653 | 0.653 | 137 | 0.000 | 0.14 | 0.780 | 0.709 | 6/9 | 0.025 | 8 | 0.00 | - |  |
| b800-f0 | 0.926 | 0.926 | 746 | 0.591 | 0.00 | 0.883 | 0.819 | 0/9 | 1.000 | 689 | 1.00 | - | k10-f0 recall at 746 tokens; QA not run |
| b800-f0.25 | 0.779 | 0.779 | 174 | 0.026 | 0.10 | 0.843 | 0.779 | 5/9 | 0.125 | 35 | 0.00 | - |  |
| b800-f0.4 | 0.737 | 0.737 | 160 | 0.022 | 0.11 | 0.790 | 0.729 | 6/9 | 0.050 | 11 | 0.00 | - |  |
| b1500-f0 | 0.926 | 0.926 | 1431 | 0.745 | 0.00 | 0.913 | 0.819 | 0/9 | 1.000 | 1419 | 1.00 | 0.90 | QA 0.90 at 1,431 tokens; dominated by k10-f0 |
| b1500-f0.25 | 0.779 | 0.779 | 174 | 0.026 | 0.10 | 0.863 | 0.779 | 5/9 | 0.125 | 35 | 0.00 | - |  |
| b1500-f0.4 | 0.737 | 0.737 | 160 | 0.022 | 0.11 | 0.800 | 0.729 | 6/9 | 0.050 | 11 | 0.00 | - |  |
| b2500-f0 | 0.926 | 0.926 | 2419 | 0.815 | 0.00 | 0.913 | 0.819 | 0/9 | 1.000 | 2402 | 1.00 | - | dominated by k10-f0 |
| b2500-f0.25 | 0.779 | 0.779 | 174 | 0.026 | 0.10 | 0.863 | 0.779 | 5/9 | 0.125 | 35 | 0.00 | - |  |
| b2500-f0.4 | 0.737 | 0.737 | 160 | 0.022 | 0.11 | 0.800 | 0.729 | 6/9 | 0.050 | 11 | 0.00 | - |  |
| b4000-f0 | 0.937 | 0.926 | 3879 | 0.863 | 0.00 | 0.913 | 0.819 | 0/9 | 1.000 | 3890 | 1.00 | 0.91 | most accurate: QA 0.91 at 3,879 tokens |
| b4000-f0.25 | 0.779 | 0.779 | 174 | 0.026 | 0.10 | 0.863 | 0.779 | 5/9 | 0.125 | 35 | 0.00 | - |  |
| b4000-f0.4 | 0.737 | 0.737 | 160 | 0.022 | 0.11 | 0.800 | 0.729 | 6/9 | 0.050 | 11 | 0.00 | - |  |

`k10-f0` reaches the recall_all@10 of `b4000-f0` (0.926) at 22 % of its tokens (850 against
3,879) and answers 90 of 100 against 91; answer input tokens over the 100 questions are
210,277 against 726,478. `b1500-f0` answers 90 at 1,431 tokens. The floor 0.25 that
`search_memory` applies when `min_score` is omitted cuts recall_all from 0.926 to 0.779
on LME and deployed R@5 from 0.913 to 0.863. The deployed corpus holds longer notes, so
`k10-f0` costs 1,545 tokens per call there.

## B. prefetch, Hermes

| setting | recall_all | recall_all@10 | mean tokens | junk share | zero-hit rate | deployed R@5 | deployed MRR@10 | deployed expect-empty | deployed off-topic fire | deployed off-topic tokens | LME off-topic fire | QA | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k3-f0 | 0.811 | 0.811 | 315 | 0.225 | 0.00 | 0.660 | 0.653 | 1/9 | 0.975 | 374 | 1.00 | - |  |
| k3-f0.25 | 0.747 | 0.747 | 201 | 0.023 | 0.10 | 0.660 | 0.653 | 5/9 | 0.125 | 34 | 0.00 | - |  |
| k3-f0.4 | 0.705 | 0.705 | 190 | 0.018 | 0.11 | 0.620 | 0.613 | 6/9 | 0.050 | 13 | 0.00 | - |  |
| k3-f0.6 | 0.642 | 0.642 | 174 | 0.000 | 0.14 | 0.620 | 0.613 | 6/9 | 0.025 | 10 | 0.00 | - |  |
| k5-f0 | 0.842 | 0.842 | 420 | 0.342 | 0.00 | 0.700 | 0.663 | 1/9 | 0.975 | 385 | 1.00 | - | 39 of 40 off-topic prompts fire |
| k5-f0.25 | 0.768 | 0.768 | 209 | 0.022 | 0.10 | 0.700 | 0.663 | 5/9 | 0.125 | 34 | 0.00 | 0.82 | selected: QA 0.82 |
| k5-f0.4 | 0.726 | 0.726 | 198 | 0.017 | 0.11 | 0.660 | 0.623 | 6/9 | 0.050 | 13 | 0.00 | - | 2 of 40 off-topic prompts fire |
| k5-f0.6 | 0.653 | 0.653 | 180 | 0.000 | 0.14 | 0.660 | 0.623 | 6/9 | 0.025 | 10 | 0.00 | - | provider read |
| k10-f0 | 0.842 | 0.842 | 442 | 0.380 | 0.00 | 0.700 | 0.663 | 1/9 | 0.975 | 385 | 1.00 | - |  |
| k10-f0.25 | 0.768 | 0.768 | 210 | 0.022 | 0.10 | 0.700 | 0.663 | 5/9 | 0.125 | 34 | 0.00 | - |  |
| k10-f0.4 | 0.726 | 0.726 | 198 | 0.017 | 0.11 | 0.660 | 0.623 | 6/9 | 0.050 | 13 | 0.00 | - |  |
| k10-f0.6 | 0.653 | 0.653 | 180 | 0.000 | 0.14 | 0.660 | 0.623 | 6/9 | 0.025 | 10 | 0.00 | - |  |

`k5-f0.25` answers 82 of 100 at 209 tokens per turn. The provider skips a hit whose line
does not fit its character budget and keeps filling with later hits; 5 of the 176 distinct
deployed notes seen are longer than 1,812 characters, and deployed R@5 is 0.700 against 0.863
before the provider's cut. On the deployed corpus 5 of 40 off-topic prompts inject at floor
0.25 and 2 at 0.4; on the LME personal corpus none do at any floor from 0.25 up.

## Settings the measurement selects

| path | setting | deciding numbers |
|---|---|---|
| A. search | `top_k=10`, `min_score=0` | QA 0.90 against 0.91 for `budget_tokens=4000` at 22 % of the tokens; recall_all 0.926 against 0.779 at the 0.25 floor |
| B. Hermes | `top_k=5`, `min_score=0.25` | QA 0.82; recall_all 0.768 against 0.653 at the 0.6 floor; no off-topic prompt fires on the LME personal corpus, 5 of 40 on the deployed corpus |

QA at `k5-f0.6` on the Hermes path is not measured.

## Fused candidates

Production `search()` reranks `FUSED_TOP=20` fused candidates in `top_k` mode and
`BUDGET_FUSED_TOP=40` in budget mode, so the `top_k` rows above derive from a larger candidate
set than the shipped read. Reranking time is linear in the candidate count, about 24 ms per
candidate. `/search` latency on 40 replayed prompts (`top_k=3`, `min_score=0.4`, a data copy of
the deployed notes, live embedder and reranker, sequential):

| fused candidates | p50 ms | p90 ms |
|---|---|---|
| 10 | 483 | 584 |
| 20 | 721 | 807 |
| 30 | 913 | 1,253 |
| 40 | 936 | 1,091 |

On the deployed snapshot, 10, 20, and 40 candidates give the same R@5, MRR@10, and
expect-empty for `k3-f0.4`, `k5-f0.25` (Hermes), and `k10-f0.25` (search). Only 5 labels score
against the active corpus. Against 40 candidates, the delivered hits of `k3-f0.4` differ on
3 of 94 labelled and probe queries at 20 and on 6 of 94 at 10, all unlabelled. On LongMemEval,
20 and 40 candidates give the same retrieval metrics (`longmemeval.md`); 10 is not measured
there.

## Transfer between corpora

| conclusion | LME personal | deployed coding | transfers |
|---|---|---|---|
| a floor cuts aggregate-question recall | recall_all 0.926 -> 0.779 at 0.25 | R@5 0.913 -> 0.863 at 0.25 | yes; smaller on the deployed labels, which are mostly single-note lookups |
| top 10 without a floor matches the 4000-token budget | recall_all@10 equal, QA 0.90 / 0.91 | R@5 equal (0.913) | recall yes; QA measured on LME only |
| off-topic prompts pass a 0.25 floor | never | 5 of 40 | no: coding prompts match coding notes, not personal ones |
| QA numbers | measured | not measured (no reference answers) | no |

## Commands

```bash
uv run python -m memory_base.eval.longmemeval retrieve --dataset PATH --data-dir DIR \
    --manifest DIR/manifest.json --read candidates
uv run python -m memory_base.eval.read_sweep lme-probe --dataset PATH --data-dir DIR
uv run python -m memory_base.eval.read_sweep deployed --out FILE
uv run python -m memory_base.eval.read_sweep report --dataset PATH --data-dir DIR --deployed FILE
uv run python -m memory_base.eval.read_sweep export --data-dir DIR --setting k10-f0 \
    --path search --out QA_DIR
uv run python scripts/longmemeval/answer.py answer --dataset PATH --data-dir QA_DIR \
    --manifest QA_DIR/manifest.json --backend claude-code \
    --model claude-sonnet-5-5 --effort high --concurrency 8
uv run python scripts/longmemeval/answer.py judge ...   # same flags
uv run python -m memory_base.eval.longmemeval score --dataset PATH --data-dir QA_DIR \
    --manifest QA_DIR/manifest.json
```
