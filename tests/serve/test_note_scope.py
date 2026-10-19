"""The note read scope: the resolver and the scope each discovery read passes on.

No DB, no network: each route is called with its store function monkeypatched, so the
assertions pin the scope value the route resolves from ``X-Author`` and the request.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from starlette.testclient import TestClient

from memory_base.serve import api
from memory_base.serve.access import namespaces
from memory_base.serve.common.http import RequestError, note_scope
from memory_base.serve.consolidation import groups
from memory_base.serve.facts import tools as fact_tools
from memory_base.serve.notes import curation, store
from memory_base.serve.search import routes as search_routes

client = TestClient(api.app, headers={"X-API-Key": "test-key", "X-Author": "claude-code"})


# ---- the resolver -----------------------------------------------------------


def test_omitted_scope_is_own_and_shared_notes():
    assert note_scope("claude-code", None) == (None, "claude-code")


def test_all_lifts_the_scope():
    assert note_scope("claude-code", "all") == (None, None)


def test_a_slug_narrows_to_that_author():
    assert note_scope("claude-code", "natsume") == ("natsume", None)


@pytest.mark.parametrize("requested", ["Not A Slug", "", " all "])
def test_an_invalid_value_is_refused(requested):
    with pytest.raises(RequestError, match='author must be an author slug or "all"'):
        note_scope("claude-code", requested)


# ---- one test per route in the default scope --------------------------------


@pytest.fixture()
def searched(monkeypatch):
    captured = {}

    async def fake_search(query, **options):
        captured.update({"author": options.get("author"), "visible_to": options.get("visible_to")})
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


@pytest.fixture()
def old_listed(monkeypatch):
    captured = {}

    async def fake_list_old_notes(older_than_days, author=None, visible_to=None):
        captured.update(author=author, visible_to=visible_to)
        return []

    monkeypatch.setattr(curation, "list_old_notes", fake_list_old_notes)
    return captured


@pytest.fixture()
def consolidated(monkeypatch):
    captured = {}

    async def fake_list_namespaces():
        return [{"name": "default"}]

    async def fake_read_snapshots(names, author, threshold, neighbors):
        captured["author"] = author
        return {}

    monkeypatch.setattr(namespaces, "list_namespaces", fake_list_namespaces)
    monkeypatch.setattr(groups, "read_snapshots", fake_read_snapshots)
    return captured


def test_search_defaults_to_own_and_shared_notes(searched):
    assert client.post("/search", json={"query": "q", "source": "memory"}).status_code == 200
    assert searched["author"] is None
    assert searched["visible_to"] == "claude-code"


def test_notes_defaults_to_own_and_shared_notes(listed):
    assert client.get("/notes").status_code == 200
    assert listed["author"] is None
    assert listed["visible_to"] == "claude-code"


def test_admin_notes_defaults_to_own_and_shared_notes(old_listed):
    assert client.get("/admin/notes").status_code == 200
    assert old_listed == {"author": None, "visible_to": "claude-code"}


def test_consolidation_groups_default_to_exactly_the_header_author(consolidated):
    assert client.get("/admin/consolidate/groups").status_code == 200
    assert consolidated["author"] == "claude-code"


def test_read_entity_notes_default_to_own_and_shared_notes(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/entities/natsume":
            return httpx.Response(
                200, json={"entity": {"slug": "natsume"}, "facts": [], "incoming": []}
            )
        seen["params"] = request.url.params.multi_items()
        return httpx.Response(200, json=[])

    from memory_base.serve.common import rest_client

    monkeypatch.setattr(
        rest_client,
        "client",
        lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://testserver"
        ),
    )
    asyncio.run(fact_tools.read_entity("natsume"))
    assert "author" not in dict(seen["params"])


# ---- "all" lifts the scope on every read ------------------------------------


@pytest.mark.parametrize(
    "fixture,request_fn",
    [
        ("searched", lambda: client.post("/search", json={"query": "q", "author": "all"})),
        ("listed", lambda: client.get("/notes", params={"author": "all"})),
        ("old_listed", lambda: client.get("/admin/notes", params={"author": "all"})),
    ],
)
def test_author_all_covers_every_author(request, fixture, request_fn):
    captured = request.getfixturevalue(fixture)
    assert request_fn().status_code == 200
    assert captured["author"] is None
    assert captured["visible_to"] is None


def test_groups_author_all_asks_for_every_author(consolidated):
    assert client.get("/admin/consolidate/groups", params={"author": "all"}).status_code == 200
    assert consolidated["author"] == "all"


def test_read_entity_forwards_its_author_argument(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/entities/natsume":
            return httpx.Response(
                200, json={"entity": {"slug": "natsume"}, "facts": [], "incoming": []}
            )
        seen["params"] = request.url.params.multi_items()
        return httpx.Response(200, json=[])

    from memory_base.serve.common import rest_client

    monkeypatch.setattr(
        rest_client,
        "client",
        lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://testserver"
        ),
    )
    asyncio.run(fact_tools.read_entity("natsume", author="all"))
    assert ("author", "all") in seen["params"]


# ---- a slug narrows to that author ------------------------------------------


@pytest.mark.parametrize(
    "fixture,request_fn",
    [
        (
            "searched",
            lambda: client.post("/search", json={"query": "q", "author": "natsume"}),
        ),
        ("listed", lambda: client.get("/notes", params={"author": "natsume"})),
        ("old_listed", lambda: client.get("/admin/notes", params={"author": "natsume"})),
    ],
)
def test_author_slug_narrows_to_it(request, fixture, request_fn):
    captured = request.getfixturevalue(fixture)
    assert request_fn().status_code == 200
    assert captured["author"] == "natsume"
    assert captured["visible_to"] is None


def test_groups_author_slug_narrows_to_it(consolidated):
    assert client.get("/admin/consolidate/groups", params={"author": "natsume"}).status_code == 200
    assert consolidated["author"] == "natsume"


def test_groups_echo_the_resolved_author(consolidated):
    body = client.get("/admin/consolidate/groups", params={"author": "all"}).json()
    assert body["params"]["author"] == "all"


# ---- the SQL clause ---------------------------------------------------------


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
        source="memory",
        ref="save_memory",
        text="t",
        ts=0.0,
        meta={
            "id": "note:a",
            "kind": "work",
            "tags": [],
            "entities": [],
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
        source="memory",
        ref="save_memory",
        text="t",
        ts=0.0,
        meta={
            "id": "note:a",
            "kind": "work",
            "tags": [],
            "entities": [],
            "author": "natsume",
            "occurred_at": None,
        },
    )
    assert "shared" not in hit_to_dict(hit)


def test_archive_candidates_binds_the_scope_after_the_cold_args(monkeypatch):
    from contextlib import asynccontextmanager

    captured = {}

    class Conn:
        async def fetch(self, query, *args):
            captured["query"] = query
            captured["args"] = args
            return []

    @asynccontextmanager
    async def acquire(timeout=None):
        yield Conn()

    monkeypatch.setattr(curation.db, "acquire", acquire)
    asyncio.run(curation.archive_candidates(1.0, visible_to="claude-code"))
    assert "metadata @> '{\"shared\": true}'::jsonb" in captured["query"]
    assert "$4" in captured["query"]
    assert captured["args"] == (
        1.0,
        curation.COLD_AGE_DAYS,
        curation.COLD_UNHIT_DAYS,
        "claude-code",
    )


def _curation_conn(monkeypatch, row):
    from contextlib import asynccontextmanager

    class Conn:
        async def fetch(self, query, *args):
            return [row]

    @asynccontextmanager
    async def acquire(timeout=None):
        yield Conn()

    monkeypatch.setattr(curation.db, "acquire", acquire)


def test_the_old_notes_rows_carry_author_and_shared(monkeypatch):
    _curation_conn(
        monkeypatch,
        {
            "id": "note:a",
            "text": "t",
            "ts_last_active": 1.0,
            "hit_count": 0,
            "last_hit_at": None,
            "author": "natsume",
            "shared": True,
        },
    )
    [note] = asyncio.run(curation.list_old_notes(90))
    assert note["author"] == "natsume"
    assert note["shared"] is True


def test_the_archive_rows_carry_author_and_omit_an_unset_shared(monkeypatch):
    base = {
        "id": "note:a",
        "kind": "work",
        "age_days": 200.0,
        "hit_count": 0,
        "last_hit_at": None,
        "author": "natsume",
        "shared": False,
    }
    _curation_conn(monkeypatch, base)
    [note] = asyncio.run(curation.archive_candidates(0.0))
    assert note["author"] == "natsume"
    assert "shared" not in note
