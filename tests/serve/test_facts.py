"""Unit tests for facts and the entity card: the batch rules, the read scoping, the routes.

No DB: the store runs against a fake connection that answers each statement by its shape
and emulates the transaction rollback and the advisory lock.
"""

from __future__ import annotations

import copy
import uuid
from contextlib import asynccontextmanager

import pytest
from starlette.testclient import TestClient

from memory_base.serve import api
from memory_base.serve.facts import store

client = TestClient(api.app, headers={"X-API-Key": "test-key", "X-Author": "claude-code"})

NOW = 1_700_000_000.0
GITHUB_TOKEN = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"


class FakeTransaction:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        self.saved = copy.deepcopy(self.conn.facts)
        self.conn.locked = False
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.conn.facts = self.saved
        return False


class FakeConnection:
    def __init__(self):
        self.entities = {slug: _entity(slug) for slug in ("natsume", "yui")}
        self.notes = {
            "note:a": {"namespace": "team-a", "archived": False},
            "note:b": {"namespace": "team-b", "archived": False},
            "note:gone": {"namespace": "team-a", "archived": True},
        }
        self.facts: list[dict] = []
        self.locked = False

    def transaction(self, **options):
        assert options["isolation"] == "read_committed"
        return FakeTransaction(self)

    async def execute(self, query, *args):
        assert "pg_advisory_xact_lock" in query
        self.locked = True
        return "SELECT 1"

    async def fetchrow(self, query, *args):
        q = " ".join(query.split())
        if q.startswith("SELECT namespace"):
            note = self.notes.get(args[0])
            ok = note is not None and not note["archived"]
            return {"namespace": note["namespace"]} if ok else None
        if "FROM" in q and ".entities WHERE slug" in q:
            return copy.deepcopy(self.entities.get(args[0]))
        assert self.locked, "every fact write runs under the advisory lock"
        if q.startswith("INSERT INTO"):
            keys = ("id", "namespace", "entity", "attribute", "value_text", "value_entity",
                    "note_id", "author", "created_at")  # fmt: skip
            row = dict(zip(keys, args)) | {"archived_at": None, "replaced_by": None}
            self.facts.append(row)
            return copy.deepcopy(row)
        row = next((f for f in self.facts if f["id"] == args[0]), None)
        if q.startswith("UPDATE"):
            row["archived_at"], row["replaced_by"] = args[1], args[2]
        return copy.deepcopy(row)

    async def fetch(self, query, *args):
        q = " ".join(query.split())
        if ".entities WHERE slug = ANY" in q:
            return [{"slug": s} for s in args[0] if s in self.entities]
        if "LIMIT" in q:
            column = "value_entity" if "value_entity = $1" in q else "entity"
            include_past, scope = args[1], args[2]
            rows = [
                f for f in self.facts
                if f[column] == args[0]
                and (include_past or f["archived_at"] is None)
                and (scope is None or f["namespace"] in scope)
            ]  # fmt: skip
            return copy.deepcopy(sorted(rows, key=lambda f: -f["created_at"]))
        entity, attribute = args
        return copy.deepcopy([
            f for f in self.facts
            if f["entity"] == entity and f["attribute"] == attribute and f["archived_at"] is None
        ])  # fmt: skip


def _entity(slug):
    return dict(
        slug=slug, kind="thing", name=slug.title(), aliases=[], description="",
        created_by="t", updated_by="t", consent_method="import", consented_at=NOW,
        created_at=NOW, updated_at=NOW,
    )  # fmt: skip


@pytest.fixture()
def fake_db(monkeypatch):
    conn = FakeConnection()
    tick = iter(range(1000))

    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    async def noop(conn):
        return None

    monkeypatch.setattr(store.db, "acquire", acquire)
    monkeypatch.setattr(store, "ensure_schema_once", noop)
    monkeypatch.setattr(store, "_now", lambda: NOW + next(tick))
    return conn


def _add(**fields):
    return {
        "op": "add", "entity": "natsume", "attribute": "city", "value_text": "Busan",
        "note_id": "note:a",
    } | fields  # fmt: skip


def _post(*ops, headers=None, **extra):
    return client.post("/admin/facts", json={"ops": list(ops)} | extra, headers=headers or {})


def _card(slug="natsume", query=""):
    return client.get(f"/entities/{slug}{query}")


def _write_one(op):
    response = _post(op)
    assert response.status_code == 200, response.text
    return response.json()["results"][0]


# ---- write: add, replace, remove --------------------------------------------


def test_add_replace_remove_leave_a_history_the_card_shows_with_include_past(fake_db):
    added = _write_one(_add())
    assert added["status"] == "written"
    assert (added["entity"], added["attribute"], added["namespace"]) == (
        "natsume",
        "city",
        "team-a",
    )

    replaced = _write_one(
        {"op": "replace", "id": added["id"], "value_text": "Seoul", "note_id": "note:b"}
    )
    assert replaced["status"] == "written"
    assert (replaced["attribute"], replaced["namespace"]) == ("city", "team-b")
    _write_one({"op": "remove", "id": replaced["id"], "note_id": "note:b"})

    assert _card().json()["facts"] == []
    past = {f["id"]: f for f in _card(query="?include_past=true").json()["facts"]}
    assert past[added["id"]]["replaced_by"] == replaced["id"]
    assert past[added["id"]]["archived_at"] is not None
    assert past[replaced["id"]]["replaced_by"] is None
    assert past[replaced["id"]]["archived_at"] is not None


def test_an_add_equal_to_a_current_row_is_unchanged_and_writes_nothing(fake_db):
    first = _write_one(_add())
    again = _write_one(_add(note_id="note:b"))
    assert (again["status"], again["id"]) == ("unchanged", first["id"])
    assert len(fake_db.facts) == 1


def test_a_relation_fact_is_read_from_both_sides(fake_db):
    fact = _write_one(_add(attribute="friend", value_text=None, value_entity="yui"))
    assert [f["id"] for f in _card("natsume").json()["facts"]] == [fact["id"]]
    card = _card("yui").json()
    assert card["facts"] == []
    assert [f["id"] for f in card["incoming"]] == [fact["id"]]
    assert card["entity"]["slug"] == "yui"


def test_allow_multiple_adds_a_second_row_under_one_key(fake_db):
    _write_one(_add())
    _write_one(_add(value_text="Seoul", allow_multiple=True))
    assert len(_card().json()["facts"]) == 2


# ---- refusals ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("op", "message"),
    [
        (_add(entity="ghost"), "ghost"),
        (_add(attribute="Bad Attr"), "attribute"),
        (_add(note_id=None), "note_id"),
        (_add(note_id="note:gone"), "note:gone"),
        (_add(note_id="note:missing"), "note:missing"),
        (_add(value_text="x" * 301), "value_text"),
        (_add(value_text=f"token {GITHUB_TOKEN}"), "credential"),
        (_add(value_text="Alice, Bob and Carol"), "one row per element"),
        (_add(value_text="Alice and Bob"), "one row per element"),
        (_add(value_entity="yui"), "exactly one"),
        (_add(value_text=None), "exactly one"),
    ],
    ids=[
        "unknown-entity",
        "bad-attribute",
        "missing-note",
        "archived-note",
        "unknown-note",
        "301-chars",
        "credential",
        "list-value",
        "and-value",
        "both-values",
        "no-value",
    ],
)
def test_a_refused_add_names_its_reason_and_writes_nothing(fake_db, op, message):
    response = _post(op)
    assert response.status_code == 400
    assert message in response.json()["error"]
    assert fake_db.facts == []


def test_a_name_in_a_longer_phrase_is_not_a_list_of_names(fake_db):
    assert _write_one(_add(value_text="Research and Development team"))["status"] == "written"


def test_an_add_on_a_held_key_names_the_current_row(fake_db):
    first = _write_one(_add())
    response = _post(_add(value_text="Seoul"))
    assert response.status_code == 400
    assert first["id"] in response.json()["error"]


def test_replace_and_remove_need_a_current_fact(fake_db):
    first = _write_one(_add())
    _write_one({"op": "remove", "id": first["id"], "note_id": "note:a"})
    for op in (
        {"op": "replace", "id": first["id"], "value_text": "Seoul", "note_id": "note:a"},
        {"op": "remove", "id": first["id"], "note_id": "note:a"},
        {"op": "remove", "id": str(uuid.uuid4()), "note_id": "note:a"},
    ):
        response = _post(op)
        assert response.status_code == 400, op
        assert "current" in response.json()["error"]


def test_a_refused_op_aborts_the_batch_and_names_its_index(fake_db):
    response = _post(_add(), _add(entity="ghost"))
    assert response.status_code == 400
    assert response.json()["error"].startswith("op 1:")
    assert fake_db.facts == []


def test_a_dry_run_returns_the_planned_rows_and_writes_nothing(fake_db):
    response = _post(_add(), dry_run=True)
    assert response.status_code == 200
    [result] = response.json()["results"]
    assert (result["status"], result["value_text"]) == ("planned", "Busan")
    assert fake_db.facts == []


# ---- author -----------------------------------------------------------------


def test_the_header_author_is_recorded(fake_db):
    assert _post(_add(), headers={"X-Author": "natsume"}).status_code == 200
    assert [fact["author"] for fact in fake_db.facts] == ["natsume"]


def test_a_body_author_is_refused(fake_db):
    response = _post(_add(), author="natsume")
    assert response.status_code == 400
    assert response.json()["error"] == "unknown field: author"
    assert fake_db.facts == []


# ---- the card ---------------------------------------------------------------


def test_the_card_covers_every_namespace_and_is_narrowed_by_namespace(fake_db):
    _write_one(_add())
    _write_one(_add(attribute="job", value_text="Chef", note_id="note:b"))
    ids = lambda r: sorted(f["attribute"] for f in r.json()["facts"])  # noqa: E731

    assert ids(_card()) == ["city", "job"]
    assert ids(_card(query="?namespace=team-b")) == ["job"]


def test_an_unknown_slug_is_404(fake_db):
    assert _card("ghost").status_code == 404


# ---- description edits against a stale read ---------------------------------


def _edit(monkeypatch, fake_db, item, **extra):
    from memory_base.serve.entities import store as entity_store

    class Conn(FakeConnection):
        async def execute(self, query, *args):
            return "SELECT 1"

        async def fetch(self, query, *args):
            return copy.deepcopy(sorted(self.entities.values(), key=lambda r: r["slug"]))

        async def fetchrow(self, query, *args):
            slug, kind, name, aliases, description, author, method, now = args
            self.entities[slug].update(description=description, updated_at=now)
            return copy.deepcopy(self.entities[slug])

    conn = Conn()

    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    async def noop(conn):
        return None

    monkeypatch.setattr(entity_store.db, "acquire", acquire)
    monkeypatch.setattr(entity_store, "ensure_schema_once", noop)
    return client.post(
        "/entities",
        json={"entities": [item], "consent_method": "conversation"} | extra,
        headers={"X-Author": "claude-code"},
    )


STAMP = "2023-11-14T22:13:20+00:00"


def test_a_description_edit_needs_the_current_base_updated_at(monkeypatch, fake_db):
    edit = {"slug": "natsume", "description": "A friend"}
    missing = _edit(monkeypatch, fake_db, edit)
    assert missing.status_code == 400 and "base_updated_at" in missing.json()["error"]

    stale = _edit(monkeypatch, fake_db, edit | {"base_updated_at": "2023-11-14T00:00:00+00:00"})
    assert stale.status_code == 409
    assert stale.json() == {"error": "stale", "slug": "natsume", "updated_at": STAMP}

    current = _edit(monkeypatch, fake_db, edit | {"base_updated_at": STAMP})
    assert current.status_code == 200
    assert current.json()["entities"][0]["description"] == "A friend"
