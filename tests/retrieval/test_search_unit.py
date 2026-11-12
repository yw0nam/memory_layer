"""Unit tests for the pure functions in src/search.py: no DB/network access.

rrf_fuse and the score floor operate purely on Hit objects /
plain dicts, so they're exercised directly with hand-built inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from memory_base.core.config import vector_literal
from memory_base.retrieval.search import (
    MIN_SCORE,
    QWEN3_RERANK_PREFIX,
    QWEN3_RERANK_SUFFIX,
    RERANK_TEXT_LIMIT,
    RRF_K,
    Hit,
    _apply_min_score,
    rerank_payload,
    rrf_fuse,
)


# ---- rrf_fuse ---------------------------------------------------------


def test_rrf_fuse_formula_k60():
    # doc "a" ranked 1st in both lists; "b"/"c" ranked 2nd in only one.
    scores = rrf_fuse([["a", "b"], ["a", "c"]])
    assert scores["a"] == pytest.approx(2 / (RRF_K + 1))
    assert scores["b"] == pytest.approx(1 / (RRF_K + 2))
    assert scores["c"] == pytest.approx(1 / (RRF_K + 2))


def test_rrf_fuse_consensus_beats_single_strong_vote():
    # "consensus" appears 2nd in both lists; "x"/"y" appear 1st in only one
    # each. Two moderate votes should outscore one strong vote.
    scores = rrf_fuse([["x", "consensus"], ["y", "consensus"]])
    single_strong_vote = scores["x"]
    consensus = scores["consensus"]
    assert consensus > single_strong_vote


def test_rrf_fuse_empty_lists_yields_empty_scores():
    assert rrf_fuse([]) == {}
    assert rrf_fuse([[], []]) == {}


def test_rrf_fuse_applies_explicit_per_list_weights():
    scores = rrf_fuse([["a", "b"], ["a", "c"]], [2.0, 0.5])

    assert scores["a"] == pytest.approx(2.5 / (RRF_K + 1))
    assert scores["b"] == pytest.approx(2.0 / (RRF_K + 2))
    assert scores["c"] == pytest.approx(0.5 / (RRF_K + 2))


def test_rrf_fuse_none_weights_preserves_default_scores():
    lists = [["a", "b"], ["a", "c"]]

    assert rrf_fuse(lists, None) == rrf_fuse(lists)


# ---- _apply_min_score -----------------------------------------------------


def _reranked_hit(ref, score):
    return Hit(ref=ref, text="", ts=0.0, rrf=0.02, rerank_score=score)


def _rrf_only_hit(ref, score):
    return Hit(ref=ref, text="", ts=0.0, rrf=score)


def test_apply_min_score_drops_below_and_keeps_at_or_above_floor():
    below = _reranked_hit("a", MIN_SCORE - 0.01)
    at_floor = _reranked_hit("b", MIN_SCORE)
    above = _reranked_hit("c", MIN_SCORE + 0.1)
    out = _apply_min_score([below, at_floor, above], min_score=None, rerank=True)
    assert out == [at_floor, above]


def test_apply_min_score_default_filters_nothing_without_rerank():
    hits = [_rrf_only_hit("a", 0.02), _rrf_only_hit("b", 0.016)]
    out = _apply_min_score(hits, min_score=None, rerank=False)
    assert out == hits


def test_apply_min_score_explicit_value_applies_to_rrf_score_without_rerank():
    keep = _rrf_only_hit("a", 0.05)
    drop = _rrf_only_hit("b", 0.02)
    out = _apply_min_score([keep, drop], min_score=0.03, rerank=False)
    assert out == [keep]


def test_apply_min_score_explicit_zero_disables_default_floor_with_rerank():
    hits = [_reranked_hit("a", 0.1), _reranked_hit("b", 0.0)]
    out = _apply_min_score(hits, min_score=0.0, rerank=True)
    assert out == hits


def test_apply_min_score_uses_min_score_constant_as_rerank_default():
    dropped = _reranked_hit("a", MIN_SCORE - 0.01)
    kept = _reranked_hit("b", MIN_SCORE)
    out = _apply_min_score([dropped, kept], min_score=None, rerank=True)
    assert out == [kept]


# ---- Hit dataclass --------------------------------------------------------


def test_hit_score_returns_rerank_score_when_set():
    h = Hit(ref="a", text="", ts=0.0, rrf=0.3, rerank_score=0.9)
    assert h.score == 0.9


def test_hit_score_falls_back_to_rrf_when_rerank_is_none():
    h = Hit(ref="a", text="", ts=0.0, rrf=0.4, rerank_score=None)
    assert h.score == 0.4


def test_hit_defaults_and_meta_dict_independence():
    h1 = Hit(ref="a", text="", ts=0.0)
    h2 = Hit(ref="b", text="", ts=0.0)
    assert h1.rrf == 0.0
    assert h1.rerank_score is None
    h1.meta["x"] = 1
    assert h2.meta == {}  # default_factory gives each Hit its own dict


# ---- rerank_payload --------------------------------------------------------


def test_rerank_payload_templates_qwen3_model():
    payload = rerank_payload("Qwen/Qwen3-Reranker-4B", "what is rrf", ["doc one", "doc two"])
    assert payload["query"].startswith(QWEN3_RERANK_PREFIX)
    assert "<Instruct>:" in payload["query"]
    assert "<Query>: what is rrf" in payload["query"]
    for doc, text in zip(payload["documents"], ["doc one", "doc two"], strict=True):
        assert doc == f"<Document>: {text}{QWEN3_RERANK_SUFFIX}"


def test_rerank_payload_leaves_non_qwen3_model_untemplated():
    payload = rerank_payload("BAAI/bge-reranker-v2-m3", "what is rrf", ["doc one", "doc two"])
    assert payload == {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": "what is rrf",
        "documents": ["doc one", "doc two"],
    }


def test_rerank_payload_gating_is_case_insensitive():
    upper = rerank_payload("Qwen/Qwen3-Reranker-4B", "q", ["d"])
    lower = rerank_payload("qwen/qwen3-reranker-4b", "q", ["d"])
    assert upper["query"] == lower["query"]
    assert upper["documents"] == lower["documents"]


def test_rerank_payload_truncates_before_templating():
    long_doc = "x" * (RERANK_TEXT_LIMIT + 500)
    payload = rerank_payload("Qwen/Qwen3-Reranker-4B", "q", [long_doc])
    templated = payload["documents"][0]
    assert (
        templated
        == f"<Document>: {'x' * RERANK_TEXT_LIMIT}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert templated.endswith("<think>\n\n</think>\n\n")


def test_rerank_payload_keeps_model_unchanged_in_both_branches():
    qwen = rerank_payload("Qwen/Qwen3-Reranker-4B", "q", ["d"])
    other = rerank_payload("BAAI/bge-reranker-v2-m3", "q", ["d"])
    assert qwen["model"] == "Qwen/Qwen3-Reranker-4B"
    assert other["model"] == "BAAI/bge-reranker-v2-m3"


# ---- vector_literal -------------------------------------------------------


def test_vector_literal_format():
    vec = np.array([1.0, 0.5, 0.25], dtype=np.float16)
    result = vector_literal(vec)
    assert result == "[1.000000,0.500000,0.250000]"


def test_vector_literal_empty():
    vec = np.array([], dtype=np.float16)
    assert vector_literal(vec) == "[]"
