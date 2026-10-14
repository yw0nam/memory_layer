"""link_entities: the REST route, the locked delta on existing notes, and the MCP tool.

No DB, no network: the curation runs on a fake connection that answers the locked read.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest
from starlette.testclient import TestClient

from memory_base.serve import api
from memory_base.serve.access import auth
from memory_base.serve.common import rest_client
from memory_base.serve.consolidation import verdicts
from memory_base.serve.notes import curation, store
from memory_base.serve.notes import tools as note_tools

client = TestClient(api.app, headers={"X-API-Key": "test-key"})
A, B, C = "note:work:a", "note:work:b", "note:work:c"


class _Tx:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return None


class LinkConn:
    """Holds note metadata by id and the active slugs; records every statement."""

    def __init__(self, notes, active=("kuro", "shiro", "nova")):
        self.notes = notes
        self.active = active
        self.updates: list[tuple] = []

    def transaction(self):
        return _Tx()

    async def fetch(self, query, *args):
        if ".memory_chunks" in query:
            return [
                {"id": i, "metadata": json.dumps(self.notes[i])}
                for i in sorted(args[0])
                if i in self.notes
            ]
        return [{"slug": slug} for slug in args[0] if slug in self.active]

    async def execute(self, query, *args):
        if query.lstrip().startswith("UPDATE"):
            self.updates.append(args)
        return "UPDATE 1"


@pytest.fixture
def use(monkeypatch):
    def _use(conn):
        @asynccontextmanager
        async def acquire(timeout=None):
            yield conn

        async def noop(conn):
            return None

        monkeypatch.setattr(curation.db, "acquire", acquire)
        monkeypatch.setattr(curation, "ensure_schema_once", noop)
        return conn

    return _use


def _link(add=(), remove=(), ids=(A, B), confirm=False, scope=None):
    return asyncio.run(
        curation.link_entities(list(ids), list(add), list(remove), "natsume", confirm, scope)
    )


def _by_id(result):
    return {note["id"]: note for note in result["notes"]}


# ---- delta ------------------------------------------------------------------


def test_the_delta_keeps_current_order_then_appends_adds_and_drops_removes(use):
    use(LinkConn({A: {"entities": ["kuro", "shiro"]}, B: {}}))
    notes = _by_id(_link(add=["nova", "kuro"], remove=["shiro"]))
    assert notes[A] == {
        "id": A,
        "before": ["kuro", "shiro"],
        "after": ["kuro", "nova"],
        "changed": True,
    }
    assert notes[B] == {"id": B, "before": [], "after": ["nova", "kuro"], "changed": True}


def test_a_preview_writes_nothing_and_a_confirm_stamps_only_the_changed_notes(use):
    conn = use(LinkConn({A: {"entities": ["kuro"], "author": "hermes"}, B: {}}))
    assert _link(add=["kuro"]) and conn.updates == []
    notes = _by_id(_link(add=["kuro"], confirm=True))
    assert notes[A]["changed"] is False and notes[B]["changed"] is True
    [(note_id, metadata)] = conn.updates
    written = json.loads(metadata)
    assert note_id == B
    assert written["entities"] == ["kuro"] and written["entities_set_by"] == "natsume"
    assert isinstance(written["entities_set_at"], float)


def test_removing_the_last_slug_drops_the_key_and_keeps_other_metadata(use):
    conn = use(LinkConn({A: {"entities": ["kuro"], "author": "hermes", "tags": ["x"]}}))
    _link(remove=["kuro"], ids=[A], confirm=True)
    written = json.loads(conn.updates[0][1])
    assert "entities" not in written and written["author"] == "hermes" and written["tags"] == ["x"]


def test_a_removed_slug_needs_no_active_entity_but_an_added_one_does(use):
    use(LinkConn({A: {"entities": ["gone"]}}))
    assert _link(remove=["gone"], ids=[A])["notes"][0]["after"] == []
    with pytest.raises(ValueError, match="unknown or archived entity 'gone'"):
        _link(add=["gone"], ids=[A])


def test_an_unknown_id_refuses_the_whole_call_naming_it_and_writes_nothing(use):
    conn = use(LinkConn({A: {}}))
    with pytest.raises(curation.LinkRefused) as refused:
        _link(add=["kuro"], ids=[A, B], confirm=True)
    assert refused.value.ids == [B] and conn.updates == []


def test_a_result_over_ten_slugs_refuses_naming_the_ids(use):
    full = [f"e{i}" for i in range(10)]
    conn = use(LinkConn({A: {"entities": full}, B: {}}, active=("kuro",)))
    with pytest.raises(curation.LinkRefused) as refused:
        _link(add=["kuro"], confirm=True)
    assert refused.value.ids == [A] and conn.updates == []


# ---- route ------------------------------------------------------------------


@pytest.fixture
def linked(monkeypatch):
    calls = []

    async def fake(ids, add, remove, author, confirm, namespaces):
        calls.append((ids, add, remove, author, confirm, namespaces))
        return {"notes": []}

    monkeypatch.setattr(curation, "link_entities", fake)
    return calls


@pytest.mark.parametrize(
    "body",
    [
        {"ids": [A], "author": "natsume"},
        {"ids": [A], "author": "natsume", "add": ["kuro"], "remove": ["kuro"]},
        {"ids": [], "author": "natsume", "add": ["kuro"]},
        {"ids": [A, A], "author": "natsume", "add": ["kuro"]},
        {"ids": [f"n{i}" for i in range(201)], "author": "natsume", "add": ["kuro"]},
        {"ids": [A], "author": "natsume", "add": ["Bad Slug"]},
        {"ids": [A], "author": "natsume", "add": None},
        {"ids": [A], "author": "natsume", "add": ["kuro"], "confirm": "yes"},
        {"ids": [A], "author": "natsume", "add": ["kuro"], "entities": ["kuro"]},
        {"ids": [A], "add": ["kuro"]},
    ],
)
def test_a_malformed_body_is_refused_before_any_write(linked, body):
    assert client.post("/notes/entities", json=body).status_code == 400
    assert linked == []


def test_an_author_outside_the_keys_authors_is_refused(linked):
    body = {"ids": [A], "author": "stranger", "add": ["kuro"]}
    assert client.post("/notes/entities", json=body).status_code == 403
    assert linked == []


def test_the_call_runs_in_the_callers_namespaces(monkeypatch, linked):
    identity = auth.KeyIdentity(
        key_id="k",
        label="work",
        home="work",
        is_admin=False,
        allowed=frozenset({"work", "public"}),
        authors=frozenset({"natsume"}),
    )

    async def authenticate(plaintext_key):
        return identity

    monkeypatch.setattr(auth, "authenticate_request", authenticate)
    body = {"ids": [A], "author": "natsume", "add": [" kuro "], "confirm": True}
    assert client.post("/notes/entities", json=body).status_code == 200
    assert linked == [([A], ["kuro"], [], "natsume", True, ["public", "work"])]


def test_a_refusal_names_the_ids(monkeypatch):
    async def refuse(*args):
        raise curation.LinkRefused("not active notes in scope", [B])

    monkeypatch.setattr(curation, "link_entities", refuse)
    response = client.post(
        "/notes/entities", json={"ids": [B], "author": "natsume", "add": ["kuro"]}
    )
    assert response.status_code == 400 and response.json()["ids"] == [B]


# ---- tool, list, undo -------------------------------------------------------


def test_the_tool_posts_the_delta_and_sends_confirm_only_when_true(monkeypatch):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"notes": []})

    monkeypatch.setattr(
        rest_client,
        "client",
        lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://testserver"
        ),
    )
    monkeypatch.setenv("MEMORY_API_KEY", "env-key")
    asyncio.run(note_tools.link_entities([A], "natsume", add=["kuro"]))
    asyncio.run(note_tools.link_entities([A], "natsume", remove=["kuro"], confirm=True))
    assert seen[0] == ("/notes/entities", {"ids": [A], "author": "natsume", "add": ["kuro"]})
    assert seen[1][1] == {"ids": [A], "author": "natsume", "remove": ["kuro"], "confirm": True}


def test_list_notes_shows_who_set_the_entities_and_when(monkeypatch):
    row = {
        "id": A,
        "kind": "work",
        "text": "t",
        "ts_last_active": 1.0,
        "namespace": "work",
        "archived_at": None,
        "occurred_at": None,
        "metadata": {"entities": ["kuro"], "entities_set_by": "natsume", "entities_set_at": 5.5},
    }

    class Conn:
        async def fetch(self, query, *args):
            return [row]

    @asynccontextmanager
    async def acquire(timeout=None):
        yield Conn()

    monkeypatch.setattr(store.db, "acquire", acquire)
    [note] = asyncio.run(store.list_notes())
    assert note["entities_set_by"] == "natsume" and note["entities_set_at"] == 5.5


def test_an_undo_is_refused_when_the_created_replacement_was_relinked_after_the_action():
    action = {
        "archived_ids": [],
        "survivor_ids": ["note:work:r"],
        "replacement_created": True,
        "replacement_id": "note:work:r",
        "applied_at": 100.0,
    }

    def replacement(**metadata):
        return {"note:work:r": {"archived_at": None, "metadata": json.dumps(metadata)}}

    assert verdicts.undo_refusal(action, replacement(), None, False) is None
    reason = verdicts.undo_refusal(action, replacement(entities_set_at=150.0), None, False)
    assert "entities" in reason
