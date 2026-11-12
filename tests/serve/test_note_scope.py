"""The note read scope: the resolver and the scope each discovery read passes on.

No DB, no network: each route is called with its store function monkeypatched, so the
assertions pin the scope value the route resolves from ``X-Author`` and the request.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from memory_base.serve import api
from memory_base.serve.common.http import RequestError, note_scope
from memory_base.serve.notes import store
from memory_base.serve.search import routes as search_routes

client = TestClient(api.app, headers={"X-API-Key": "test-key", "X-Author": "claude-code"})


# ---- the resolver -----------------------------------------------------------


def test_omitted_scope_is_own_and_shared_notes():
    assert note_scope("claude-code", None) == (None, "claude-code")


def test_all_lifts_the_scope():
    assert note_scope("claude-code", "all") == (None, None)


def test_a_list_names_exactly_those_authors():
    assert note_scope("claude-code", ["natsume", "claude-code"]) == (
        ["natsume", "claude-code"],
        None,
    )


@pytest.mark.parametrize("requested", ["Not A Slug", "", " all ", [], ["natsume", "Bad"]])
def test_an_invalid_value_is_refused(requested):
    with pytest.raises(RequestError, match='author must be "all" or a list of author slugs'):
        note_scope("claude-code", requested)


# ---- one test per route in the default scope --------------------------------


@pytest.fixture()
def searched(monkeypatch):
    captured = {}

    async def fake_search(query, **options):
        captured.update(
            {"authors": options.get("authors"), "visible_to": options.get("visible_to")}
        )
        return []

    monkeypatch.setattr(search_routes, "search", fake_search)
    return captured


@pytest.fixture()
def listed(monkeypatch):
    captured = {}

    async def fake_list_notes(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(store, "list_notes", fake_list_notes)
    return captured


def test_search_defaults_to_own_and_shared_notes(searched):
    assert client.post("/search", json={"query": "q"}).status_code == 200
    assert searched["authors"] is None
    assert searched["visible_to"] == "claude-code"


def test_search_narrows_to_the_listed_authors(searched):
    body = {"query": "q", "author": ["natsume", "claude-code"]}
    assert client.post("/search", json=body).status_code == 200
    assert searched == {"authors": ["natsume", "claude-code"], "visible_to": None}


def test_notes_defaults_to_own_and_shared_notes(listed):
    assert client.get("/notes").status_code == 200
    assert listed["authors"] is None
    assert listed["visible_to"] == "claude-code"


def test_notes_narrows_to_the_repeated_author_params(listed):
    assert client.get("/notes?author=natsume&author=claude-code").status_code == 200
    assert listed["authors"] == ["natsume", "claude-code"]
    assert listed["visible_to"] is None


# ---- the SQL clause ---------------------------------------------------------


def test_a_non_string_author_is_a_400():
    response = client.post("/search", json={"query": "q", "author": 5})
    assert response.status_code == 400
    assert response.json()["error"] == 'author must be "all" or a list of author slugs'


def test_history_predicates_carries_the_visible_to_clause():
    from memory_base.retrieval.search import history_predicates

    clause, args = history_predicates(
        include_archived=False, kind=None, tags=None, visible_to="claude-code"
    )
    assert clause.endswith(
        " AND (source_type <> 'agent_note' OR metadata->>'author' = $2 "
        "OR metadata @> '{\"shared\": true}'::jsonb)"
    )
    assert args == ["claude-code"]


def test_history_predicates_applies_the_alias_prefix():
    from memory_base.retrieval.search import history_predicates

    clause, _ = history_predicates(
        include_archived=False, kind=None, tags=None, alias="candidate", visible_to="claude-code"
    )
    assert clause.endswith(
        " AND (candidate.source_type <> 'agent_note' "
        "OR candidate.metadata->>'author' = $2 "
        "OR candidate.metadata @> '{\"shared\": true}'::jsonb)"
    )


def test_history_predicates_binds_visible_to_after_the_other_filters():
    from memory_base.retrieval.search import history_predicates

    _, args = history_predicates(
        include_archived=False,
        kind="work",
        tags=["infra"],
        namespaces=["team-a"],
        visible_to="claude-code",
    )
    assert args == ["work", ["infra"], ["team-a"], "claude-code"]


def test_a_note_hit_carries_shared():
    from memory_base.retrieval.search import Hit
    from memory_base.serve.search.routes import hit_to_dict

    hit = Hit(
        ref="save_memory",
        text="t",
        ts=0.0,
        meta={
            "id": "note:a",
            "kind": "work",
            "tags": [],
            "author": "natsume",
            "shared": True,
            "occurred_at": None,
        },
    )
    assert hit_to_dict(hit)["shared"] is True


def test_a_note_hit_without_shared_omits_it():
    from memory_base.retrieval.search import Hit
    from memory_base.serve.search.routes import hit_to_dict

    hit = Hit(
        ref="save_memory",
        text="t",
        ts=0.0,
        meta={
            "id": "note:a",
            "kind": "work",
            "tags": [],
            "author": "natsume",
            "occurred_at": None,
        },
    )
    assert "shared" not in hit_to_dict(hit)
