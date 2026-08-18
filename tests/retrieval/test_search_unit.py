"""Unit tests for the pure functions in src/search.py: no DB/network access.

rrf_fuse, _apply_time_decay, and _dedup_cap operate purely on Hit objects /
plain dicts, so they're exercised directly with hand-built inputs.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from memory_base.core.config import vector_literal
from memory_base.retrieval.search import (
    FUSED_TOP,
    MIN_SCORE,
    PER_FILE_CAP,
    QWEN3_RERANK_PREFIX,
    QWEN3_RERANK_SUFFIX,
    RERANK_TEXT_LIMIT,
    RRF_K,
    Hit,
    _apply_min_score,
    _apply_time_decay,
    _decay_targets,
    _dedup_cap,
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


# ---- _apply_time_decay --------------------------------------------------


def test_time_decay_half_life_90_days():
    now = time.time()
    fresh = Hit(source="code", ref="a", text="", ts=now, rrf=1.0)
    ninety_days_old = Hit(source="code", ref="b", text="", ts=now - 90 * 86400, rrf=1.0)
    _apply_time_decay([fresh, ninety_days_old])
    assert fresh.rrf == pytest.approx(1.0, rel=1e-3)
    assert ninety_days_old.rrf == pytest.approx(0.5, rel=1e-3)


def test_time_decay_age_zero_means_no_decay():
    now = time.time()
    hit = Hit(source="code", ref="a", text="", ts=now, rrf=0.7)
    _apply_time_decay([hit])
    assert hit.rrf == pytest.approx(0.7, rel=1e-3)


def test_time_decay_180_days_quarters_score():
    now = time.time()
    hit = Hit(source="code", ref="a", text="", ts=now - 180 * 86400, rrf=1.0)
    _apply_time_decay([hit])
    assert hit.rrf == pytest.approx(0.25, rel=1e-3)


# ---- _decay_targets ------------------------------------------------------


def test_decay_targets_are_every_hit_by_default():
    code = Hit(source="code", ref="a", text="", ts=0.0, rrf=1.0)
    memory = Hit(source="memory", ref="b", text="", ts=0.0, rrf=1.0)
    assert _decay_targets([code, memory], include_archived=False) == [code, memory]


def test_decay_targets_keep_code_when_archived_memory_is_included():
    code = Hit(source="code", ref="a", text="", ts=0.0, rrf=1.0)
    memory = Hit(source="memory", ref="b", text="", ts=0.0, rrf=1.0)
    assert _decay_targets([code, memory], include_archived=True) == [code]


# ---- _dedup_cap ----------------------------------------------------------


def _hit(ref, rrf, filename=None):
    meta = {"filename": filename} if filename else {}
    return Hit(source="code", ref=ref, text="", ts=0.0, rrf=rrf, meta=meta)


def test_dedup_cap_caps_same_file_to_per_file_cap():
    hits = [_hit(f"a.py:L{i}", rrf=float(10 - i), filename="a.py") for i in range(5)]
    out = _dedup_cap(hits)
    assert len(out) == PER_FILE_CAP == 3
    # keeps the highest-rrf chunks for that file
    assert [h.rrf for h in out] == [10.0, 9.0, 8.0]


def test_dedup_cap_overall_fused_top_cap():
    hits = [_hit(f"f{i}.py:L1", rrf=float(30 - i), filename=f"f{i}.py") for i in range(30)]
    out = _dedup_cap(hits)
    assert len(out) == FUSED_TOP == 20


def test_dedup_cap_keeps_rrf_descending_order():
    hits = [
        _hit("b.py:L1", rrf=0.3, filename="b.py"),
        _hit("a.py:L1", rrf=0.9, filename="a.py"),
        _hit("c.py:L1", rrf=0.6, filename="c.py"),
    ]
    out = _dedup_cap(hits)
    assert [h.rrf for h in out] == [0.9, 0.6, 0.3]


def test_dedup_cap_falls_back_to_ref_when_no_filename_meta():
    # history hits have no meta["filename"] -> dedup key falls back to ref
    hits = [Hit(source="memory", ref="sess-1", text="", ts=0.0, rrf=1.0, meta={})]
    out = _dedup_cap(hits)
    assert len(out) == 1
    assert out[0].ref == "sess-1"


# ---- _apply_min_score -----------------------------------------------------


def _reranked_hit(ref, score):
    return Hit(source="code", ref=ref, text="", ts=0.0, rrf=0.02, rerank_score=score)


def _rrf_only_hit(ref, score):
    return Hit(source="code", ref=ref, text="", ts=0.0, rrf=score)


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
    h = Hit(source="code", ref="a", text="", ts=0.0, rrf=0.3, rerank_score=0.9)
    assert h.score == 0.9


def test_hit_score_falls_back_to_rrf_when_rerank_is_none():
    h = Hit(source="code", ref="a", text="", ts=0.0, rrf=0.4, rerank_score=None)
    assert h.score == 0.4


def test_hit_defaults_and_meta_dict_independence():
    h1 = Hit(source="code", ref="a", text="", ts=0.0)
    h2 = Hit(source="code", ref="b", text="", ts=0.0)
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
