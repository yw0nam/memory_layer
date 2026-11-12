"""Tests for serve/search/routes.py's hit serialization.

Runs without DB access: pure-function unit tests use hand-built search.Hit objects.
"""

from __future__ import annotations

from memory_base.retrieval.search import Hit

from memory_base.serve.search import routes as search_routes


MEMORY_META = {
    "id": "note:default:0000000000000000",
    "kind": "work",
    "tags": [],
    "occurred_at": None,
}


def _hit(text="hello", rrf=0.5, rerank_score=None, meta=None):
    return Hit(
        ref="save_memory",
        text=text,
        ts=1_700_000_000.0,
        rrf=rrf,
        rerank_score=rerank_score,
        meta={**MEMORY_META, **(meta or {})},
    )


# ---- hit serialization -------------------------------------------------------


def test_hit_to_dict_basic_fields():
    d = search_routes.hit_to_dict(_hit(text="body", rrf=0.4, rerank_score=0.8))
    assert d["id"] == MEMORY_META["id"]
    assert d["ref"] == "save_memory"
    assert d["score"] == 0.8  # prefers rerank_score over rrf
    assert d["text"] == "body"
    assert "date" in d and len(d["date"]) == 10  # YYYY-MM-DD


def test_hit_to_dict_falls_back_to_rrf_when_no_rerank_score():
    assert search_routes.hit_to_dict(_hit(rrf=0.4))["score"] == 0.4


def test_hit_to_dict_returns_the_text_whole():
    assert len(search_routes.hit_to_dict(_hit(text="n" * 4000))["text"]) == 4000


def test_hit_to_dict_marks_archived_hits():
    d = search_routes.hit_to_dict(_hit(meta={"archived": True}))
    assert d["archived"] is True


def test_hit_to_dict_omits_archived_key_for_live_hits():
    assert "archived" not in search_routes.hit_to_dict(_hit(meta={"archived": False}))
    assert "archived" not in search_routes.hit_to_dict(_hit())


def test_hit_to_dict_exposes_the_note_author_when_present():
    hit = _hit(meta={**MEMORY_META, "author": "natsume"})
    assert search_routes.hit_to_dict(hit)["author"] == "natsume"


def test_hit_to_dict_omits_the_author_when_unrecorded():
    assert "author" not in search_routes.hit_to_dict(_hit(meta={**MEMORY_META, "author": None}))
    assert "author" not in search_routes.hit_to_dict(_hit())


def test_hit_to_dict_exposes_the_superseded_id_when_present():
    hit = _hit(meta={**MEMORY_META, "supersedes": "note:old0000000000"})
    assert search_routes.hit_to_dict(hit)["supersedes"] == "note:old0000000000"


def test_hit_to_dict_omits_the_superseded_id_when_unrecorded():
    assert "supersedes" not in search_routes.hit_to_dict(
        _hit(meta={**MEMORY_META, "supersedes": None})
    )
    assert "supersedes" not in search_routes.hit_to_dict(_hit())


def test_memory_hit_dict_carries_no_conversation_link_keys():
    d = search_routes.hit_to_dict(
        _hit(meta={"conversation_id": "conv:0", "turn_start": 0, "turn_end": 1})
    )
    assert not {"conversation_id", "turn_start", "turn_end"} & set(d)
