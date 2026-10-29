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
query ──search(budget_tokens=10**9)──> hits:       every fused candidate (BUDGET_FUSED_TOP=40), rerank order
      └─search(min_score=0.0)────────> top_k_hits: the production top_k read (FUSED_TOP=20 reranked,
                                                   at most RERANK_TOP=10 returned, no floor)
                                              │
          grid point, production helpers      │  _apply_min_score / _pack_budget
                                              v
    top_k mode : top_k_hits -> floor -> first top_k
    budget mode: hits       -> floor -> pack until the chars/4 estimate passes budget_tokens
                                              │
          read path delivery                  v
    search      : every cut hit, tokens = sum of estimate_tokens
    hermes      : MemoryBaseClient.build_prefetch,    tokens = block chars / 4
```

Each query is searched twice: the budget search feeds the budget points, and the `top_k` search
(the read `search_memory` and the Hermes prefetch issue) feeds the `top_k` points, so a `top_k`
row sees the candidates a shipped read reranks. A budget point with a floor drops candidates
below the floor before packing; production `search()` ignores `min_score` in budget mode, so
the budget rows with a floor describe that packing rule rather than a shipped read. The
delivery step runs the Hermes client's rendering code: a hit that reached the prompt counts, a
hit the client skipped does not.

Grid: `top_k` in {3, 5, 10} x `min_score` in {0, 0.25, 0.4, 0.6}; `budget_tokens` in
{800, 1500, 2500, 4000} x `min_score` in {0, 0.25, 0.4}. A setting is named `k<top_k>-f<floor>`
or `b<budget>-f<floor>`.

## Corpora

| corpus | content | queries | measures |
|---|---|---|---|
| LME personal | LongMemEval_S subset, 100 questions, notes extracted by Sonnet 5.5, every extracted note the save path accepts loaded, one namespace per question, throwaway Postgres | the 100 benchmark questions | recall_all, ndcg_any@10, tokens, junk share, zero-hit rate, QA |
| LME probe | the namespace of question `c5e8278d` (first of the subset) | 40 off-topic coding prompts, 20 coding memory-intent prompts (`tests/fixtures/read_sweep_probes.jsonl`) | off-topic fire rate |
| deployed | `memory.memory_chunks` copied over a `default_transaction_read_only` connection into a throwaway Postgres on the current schema: 337 rows, 86 active; search spans every namespace | the 34 labelled notes-replay queries (`tests/fixtures/retrieval_eval_notes.jsonl`: 12 with an expected note, all 12 score because their relevant notes are active in the snapshot; 22 expect-empty) and the 60 probe prompts | recall@5, MRR@10, expect-empty passes, off-topic fire rate and tokens |

Metric definitions:

| metric | definition |
|---|---|
| recall_all | a question counts when every answer session is among the sessions of the delivered hits; abstention questions excluded |
| recall_all@10 | the same over the first 10 distinct sessions (upstream definition) |
| mean tokens | mean estimated tokens (characters / 4) a packet puts into the prompt |
| junk share | delivered hits from no answer session, over all delivered hits of scored questions |
| zero-hit rate | packets with no delivered hit, over the 100 questions |
| off-topic fire | share of the 40 off-topic prompts that deliver at least one hit; every such hit is junk |
| deployed R@5 / MRR@10 | notes-replay metrics over the 12 scored labels (every label with an expected note scores) |
| QA | Sonnet 5.5 answer and judge (`answer.py --backend claude-code --model claude-sonnet-5-5 --effort high`), 100 questions, one pass |

## Rerank scores

The reranker's scores are bimodal on both corpora. Over the 40-candidate `hits` lists of the
100 LME questions and of the 34 deployed labels, 92.8 % of LME candidates and 64.8 % of
deployed candidates score below 0.01, and 4.0 % and 5.4 % score 0.6 or more. Answer notes
for aggregate questions (multi-session, temporal) often score below 0.25 because each note
holds one part of the answer, so any floor removes them. The coding probe prompts never
score above 0.06 on the LME personal corpus, so every floor from 0.25 up rejects all of them
there.

## A. search

| setting | recall_all | recall_all@10 | mean tokens | junk share | zero-hit rate | deployed R@5 | deployed MRR@10 | deployed expect-empty | deployed off-topic fire | deployed off-topic tokens | LME off-topic fire | QA | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k3-f0 | 0.811 | 0.811 | 274 | 0.228 | 0.792 | 0.736 | 0/22 | 1.000 | 554 | 628 | 1.00 | - |  |
| k3-f0.25 | 0.747 | 0.747 | 158 | 0.028 | 0.792 | 0.736 | 12/22 | 0.100 | 26 | 26 | 0.00 | - |  |
| k3-f0.4 | 0.705 | 0.705 | 146 | 0.024 | 0.708 | 0.653 | 15/22 | 0.025 | 8 | 8 | 0.00 | - |  |
| k3-f0.6 | 0.653 | 0.653 | 131 | 0.000 | 0.708 | 0.653 | 17/22 | 0.025 | 8 | 8 | 0.00 | - |  |
| k5-f0 | 0.863 | 0.863 | 459 | 0.383 | 0.958 | 0.753 | 0/22 | 1.000 | 922 | 1036 | 1.00 | - |  |
| k5-f0.25 | 0.768 | 0.768 | 173 | 0.026 | 0.958 | 0.753 | 12/22 | 0.100 | 26 | 26 | 0.00 | - |  |
| k5-f0.4 | 0.726 | 0.726 | 158 | 0.023 | 0.875 | 0.669 | 15/22 | 0.025 | 8 | 8 | 0.00 | - |  |
| k5-f0.6 | 0.663 | 0.663 | 137 | 0.000 | 0.750 | 0.653 | 17/22 | 0.025 | 8 | 8 | 0.00 | - |  |
| k10-f0 | 0.926 | 0.926 | 939 | 0.601 | 0.958 | 0.753 | 0/22 | 1.000 | 1912 | 2165 | 1.00 | 0.88 | selected: recall_all@10 of b4000-f0 at 24 % of its tokens; QA 0.88 against 0.91 |
| k10-f0.25 | 0.768 | 0.768 | 175 | 0.026 | 0.958 | 0.753 | 12/22 | 0.100 | 26 | 26 | 0.00 | - | search_memory default read |
| k10-f0.4 | 0.726 | 0.726 | 158 | 0.022 | 0.875 | 0.669 | 15/22 | 0.025 | 8 | 8 | 0.00 | - |  |
| k10-f0.6 | 0.663 | 0.663 | 137 | 0.000 | 0.750 | 0.653 | 17/22 | 0.025 | 8 | 8 | 0.00 | - |  |
| b800-f0 | 0.926 | 0.926 | 741 | 0.590 | 0.875 | 0.736 | 0/22 | 1.000 | 682 | 656 | 1.00 | - | k10-f0 recall at 741 tokens; QA not run |
| b800-f0.25 | 0.779 | 0.779 | 174 | 0.026 | 0.875 | 0.736 | 12/22 | 0.100 | 26 | 26 | 0.00 | - |  |
| b800-f0.4 | 0.737 | 0.737 | 159 | 0.022 | 0.750 | 0.653 | 15/22 | 0.025 | 8 | 8 | 0.00 | - |  |
| b1500-f0 | 0.926 | 0.926 | 1437 | 0.745 | 0.875 | 0.747 | 0/22 | 1.000 | 1389 | 1368 | 1.00 | 0.90 | QA 0.90 at 1,437 tokens |
| b1500-f0.25 | 0.779 | 0.779 | 174 | 0.026 | 0.875 | 0.747 | 12/22 | 0.100 | 26 | 26 | 0.00 | - |  |
| b1500-f0.4 | 0.737 | 0.737 | 159 | 0.022 | 0.750 | 0.663 | 15/22 | 0.025 | 8 | 8 | 0.00 | - |  |
| b2500-f0 | 0.926 | 0.926 | 2416 | 0.814 | 0.875 | 0.747 | 0/22 | 1.000 | 2386 | 2356 | 1.00 | - | k10-f0 recall at 2,416 tokens; QA not run |
| b2500-f0.25 | 0.779 | 0.779 | 174 | 0.026 | 0.875 | 0.747 | 12/22 | 0.100 | 26 | 26 | 0.00 | - |  |
| b2500-f0.4 | 0.737 | 0.737 | 159 | 0.022 | 0.750 | 0.663 | 15/22 | 0.025 | 8 | 8 | 0.00 | - |  |
| b4000-f0 | 0.937 | 0.926 | 3884 | 0.863 | 0.875 | 0.747 | 0/22 | 1.000 | 3875 | 3785 | 1.00 | 0.91 | most accurate: QA 0.91 at 3,884 tokens |
| b4000-f0.25 | 0.779 | 0.779 | 174 | 0.026 | 0.875 | 0.747 | 12/22 | 0.100 | 26 | 26 | 0.00 | - |  |
| b4000-f0.4 | 0.737 | 0.737 | 159 | 0.022 | 0.750 | 0.663 | 15/22 | 0.025 | 8 | 8 | 0.00 | - |  |

`k10-f0` reaches the recall_all@10 of `b4000-f0` (0.926) at 24 % of its tokens (939 against
3,884) and answers 88 of 100 against 91; answer input tokens over the 100 questions are
235,097 against 739,087. `b1500-f0` answers 90 at 1,437 tokens (328,954 answer input tokens),
two questions above `k10-f0` for 53 % more tokens. The floor 0.25 that
`search_memory` applies when `min_score` is omitted cuts recall_all from 0.926 to 0.768
on LME and deployed R@5 from 0.867 to 0.667 (5 scored labels). The deployed corpus holds
longer notes, so `k10-f0` costs 1,890 tokens per call there.

## B. prefetch, Hermes

| setting | recall_all | recall_all@10 | mean tokens | junk share | zero-hit rate | deployed R@5 | deployed MRR@10 | deployed expect-empty | deployed off-topic fire | deployed off-topic tokens | LME off-topic fire | QA | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k3-f0 | 0.811 | 0.811 | 319 | 0.221 | 0.708 | 0.694 | 0/22 | 1.000 | 401 | 399 | 1.00 | - |  |
| k3-f0.25 | 0.747 | 0.747 | 203 | 0.023 | 0.708 | 0.694 | 12/22 | 0.100 | 28 | 28 | 0.00 | - |  |
| k3-f0.4 | 0.705 | 0.705 | 190 | 0.018 | 0.625 | 0.611 | 15/22 | 0.025 | 10 | 10 | 0.00 | - |  |
| k3-f0.6 | 0.653 | 0.653 | 175 | 0.000 | 0.625 | 0.611 | 17/22 | 0.025 | 10 | 10 | 0.00 | - |  |
| k5-f0 | 0.832 | 0.832 | 429 | 0.346 | 0.708 | 0.694 | 0/22 | 1.000 | 442 | 440 | 1.00 | - | 40 of 40 off-topic prompts fire |
| k5-f0.25 | 0.758 | 0.758 | 212 | 0.022 | 0.708 | 0.694 | 12/22 | 0.100 | 28 | 28 | 0.00 | 0.79 | selected: QA 0.79; provider read |
| k5-f0.4 | 0.716 | 0.716 | 197 | 0.017 | 0.625 | 0.611 | 15/22 | 0.025 | 10 | 10 | 0.00 | - | 1 of 40 off-topic prompts fire |
| k5-f0.6 | 0.663 | 0.663 | 181 | 0.000 | 0.625 | 0.611 | 17/22 | 0.025 | 10 | 10 | 0.00 | - |  |
| k10-f0 | 0.863 | 0.863 | 477 | 0.412 | 0.708 | 0.694 | 0/22 | 1.000 | 457 | 458 | 1.00 | - |  |
| k10-f0.25 | 0.758 | 0.758 | 213 | 0.021 | 0.708 | 0.694 | 12/22 | 0.100 | 28 | 28 | 0.00 | - |  |
| k10-f0.4 | 0.716 | 0.716 | 197 | 0.017 | 0.625 | 0.611 | 15/22 | 0.025 | 10 | 10 | 0.00 | - |  |
| k10-f0.6 | 0.663 | 0.663 | 181 | 0.000 | 0.625 | 0.611 | 17/22 | 0.025 | 10 | 10 | 0.00 | - |  |

`k5-f0.25` answers 79 of 100 at 212 tokens per turn. The provider skips a hit whose line
does not fit the 2000-character budget and keeps filling with later hits, so a note longer
than about 1,810 characters never reaches the prompt. Three of the 5 scored deployed labels
share one 1,962-character relevant note, which puts Hermes deployed R@5 at 0.708 against 0.958
on the search path at floor 0.
On the deployed corpus 4 of 40 off-topic prompts inject at floor 0.25 and 1 at 0.4; on the
LME personal corpus none do at any floor from 0.25 up.

## Settings the measurement selects

| path | setting | deciding numbers |
|---|---|---|
| A. search | `top_k=10`, `min_score=0` | recall_all@10 0.926, equal to `budget_tokens=4000`, at 24 % of the tokens; QA 0.88 against 0.91; recall_all 0.926 against 0.768 at the 0.25 floor |
| B. Hermes | `top_k=5`, `min_score=0.25` | QA 0.79; recall_all 0.758 against 0.663 at the 0.6 floor; no off-topic prompt fires on the LME personal corpus, 4 of 40 on the deployed corpus |

QA is measured for the two shipped reads and for `b1500-f0` and `b4000-f0`; every other row's QA is `-`.

## Fused candidates

Production `search()` reranks `FUSED_TOP=20` fused candidates in `top_k` mode and
`BUDGET_FUSED_TOP=40` in budget mode. From 10 to 30 candidates, each one adds about 21 ms of reranking;
30 and 40 differ by 23 ms. `/search` latency on 40 replayed prompts (`top_k=3`, `min_score=0.4`, a data copy of
the deployed notes, live embedder and reranker, sequential):

| fused candidates | p50 ms | p90 ms |
|---|---|---|
| 10 | 483 | 584 |
| 20 | 721 | 807 |
| 30 | 913 | 1,253 |
| 40 | 936 | 1,091 |

On LongMemEval, 20 and 40 candidates give the same retrieval metrics (`longmemeval.md`); 10 is
not measured there.

## Transfer between corpora

| conclusion | LME personal | deployed coding | transfers |
|---|---|---|---|
| a floor cuts aggregate-question recall | recall_all 0.926 -> 0.768 at 0.25 | R@5 0.867 -> 0.667 at 0.25 | yes |
| top 10 without a floor matches the 4000-token budget | recall_all@10 equal, QA 0.88 / 0.91 | R@5 equal (0.867) | recall yes; QA measured on LME only |
| off-topic prompts pass a 0.25 floor | never | 4 of 40 | no: coding prompts match coding notes, not personal ones |
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
