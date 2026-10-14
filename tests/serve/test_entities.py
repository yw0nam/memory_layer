"""Unit tests for the entity registry: store rules and the REST routes.

No DB: the store runs against a fake connection that answers each statement by its verb
and emulates the transaction rollback and the registry lock.
"""

from __future__ import annotations

import copy
from contextlib import asynccontextmanager

import pytest
from starlette.testclient import TestClient

from memory_base.serve import api
from memory_base.serve.entities import store

client = TestClient(api.app, headers={"X-API-Key": "test-key"})

NOW = 1_700_000_000.0
GITHUB_TOKEN = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"


class FakeTransaction:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        self.saved = copy.deepcopy(self.conn.rows)
        self.conn.locked = False
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.conn.rows = self.saved
        return False


class FakeConnection:
    def __init__(self):
        self.rows: dict[str, dict] = {}
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
        assert self.locked, "every read and write runs under the registry lock"
        assert q.endswith("RETURNING *")
        if q.startswith("INSERT INTO"):
            slug, kind, name, aliases, description, author, method, now = args
            self.rows[slug] = dict(
                slug=slug, kind=kind, name=name, aliases=aliases, description=description,
                created_by=author, updated_by=author, consent_method=method, consented_at=now,
                created_at=now, updated_at=now,
            )  # fmt: skip
        else:
            slug, kind, name, aliases, description, author, method, now = args
            self.rows[slug].update(
                kind=kind, name=name, aliases=aliases, description=description,
                updated_by=author, consent_method=method, consented_at=now, updated_at=now,
            )  # fmt: skip
        return copy.deepcopy(self.rows[slug])

    async def fetch(self, query, *args):
        assert self.locked or "pg_advisory" not in query
        return copy.deepcopy(sorted(self.rows.values(), key=lambda r: r["slug"]))


@pytest.fixture()
def fake_db(monkeypatch):
    conn = FakeConnection()

    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    async def noop(conn):
        return None

    monkeypatch.setattr(store.db, "acquire", acquire)
    monkeypatch.setattr(store, "ensure_schema_once", noop)
    monkeypatch.setattr(store, "_now", lambda: NOW)
    return conn


def _item(slug="natsume", kind="person", name="Natsume", **fields):
    return {"slug": slug, "kind": kind, "name": name, **fields}


def _post(*items, author="claude-code", consent_method="conversation", **extra):
    return client.post(
        "/entities",
        json={"entities": list(items), "author": author, "consent_method": consent_method} | extra,
    )


def _delete(slug="natsume", author="claude-code", **extra):
    return client.post(f"/entities/{slug}/delete", json={"author": author} | extra)


# ---- create and edit --------------------------------------------------------


def test_create_stores_the_row_with_consent(fake_db):
    response = _post(_item(aliases=["Nat"], description="A friend"), consent_method="elicitation")
    assert response.status_code == 200
    [row] = response.json()["entities"]
    assert row["slug"] == "natsume"
    assert row["aliases"] == ["Nat"]
    assert row["created_by"] == row["updated_by"] == "claude-code"
    assert row["consent_method"] == "elicitation"
    assert row["consented_at"] == row["created_at"] == "2023-11-14T22:13:20+00:00"


def test_edit_keeps_omitted_fields_and_replaces_aliases(fake_db):
    _post(_item(aliases=["Nat", "Natchan"], description="A friend"))
    response = _post({"slug": "natsume", "aliases": ["Natsu"], "name": "Natsume K"})
    [row] = response.json()["entities"]
    assert (row["kind"], row["name"], row["description"]) == ("person", "Natsume K", "A friend")
    assert row["aliases"] == ["Natsu"]


def test_create_without_kind_or_name_is_refused(fake_db):
    response = _post({"slug": "natsume", "name": "Natsume"})
    assert response.status_code == 400
    assert "kind" in response.json()["error"]


def test_invalid_kind_or_consent_method_is_a_bad_request(fake_db):
    assert _post(_item(kind="robot")).status_code == 400
    assert _post(_item(), consent_method="telepathy").status_code == 400
    assert fake_db.rows == {}


# ---- uniqueness -------------------------------------------------------------


def _collision(response, holder, value):
    assert response.status_code == 409
    assert response.json() == {
        "error": response.json()["error"],
        "holder": holder,
        "value": value,
    }
    assert holder in response.json()["error"]


def test_a_name_collides_by_nfkc_and_case(fake_db):
    _post(_item(slug="yui", name="ＹＵＩ"))
    _collision(_post(_item(slug="yui2", name="yui")), "yui", "yui")


def test_an_alias_collides_with_another_entitys_alias(fake_db):
    _post(_item(aliases=["Nat"]))
    _collision(_post(_item(slug="other", name="Other", aliases=["nat"])), "natsume", "nat")


@pytest.mark.parametrize(
    ("new", "value"),
    [
        (_item(slug="other", name="Other", aliases=["natsume"]), "natsume"),
        (_item(slug="nat", name="Nat Two"), "nat"),
    ],
    ids=["alias-equals-a-slug", "slug-equals-an-alias"],
)
def test_a_new_value_equal_to_an_existing_one_collides_either_way(fake_db, new, value):
    _post(_item(aliases=["Nat"]))
    _collision(_post(new), "natsume", value)


def test_a_repeated_slug_in_one_batch_is_refused(fake_db):
    response = _post(_item(), _item(name="Another"))
    assert response.status_code == 400
    assert "natsume" in response.json()["error"]
    assert fake_db.rows == {}


def test_a_colliding_second_item_writes_nothing(fake_db):
    response = _post(_item(slug="a", name="A"), _item(slug="b", name="a"))
    assert response.status_code == 409
    assert fake_db.rows == {}


def test_a_credential_in_the_description_is_refused(fake_db):
    response = _post(_item(description=f"token {GITHUB_TOKEN}"))
    assert response.status_code == 400
    assert "credential" in response.json()["error"]
    assert fake_db.rows == {}


# ---- delete -----------------------------------------------------------------


def test_delete_rejects_a_malformed_request_before_touching_the_store(fake_db):
    assert _delete(slug="Bad Slug").status_code == 400
    assert _delete(expect={"notes": 1}).status_code == 400
    assert _delete(expect={"notes": "1", "facts": 0, "incoming": 0}).status_code == 400
    assert _delete(consent_method="conversation").status_code == 400


# ---- author and consent ------------------------------------------------------


def test_a_non_slug_author_is_refused_and_any_slug_is_accepted(fake_db):
    assert _post(_item(), author="Not A Slug").status_code == 400
    assert _delete(author="Not A Slug").status_code == 400
    assert _post(_item(), author="someone-else").status_code == 200


def test_import_is_recorded_as_the_client_states_it(fake_db):
    assert _post(_item(), consent_method="import").status_code == 200


# ---- dry run ----------------------------------------------------------------


def test_a_dry_run_validates_and_returns_the_rows_without_writing(fake_db):
    response = _post(_item(), dry_run=True)
    assert response.status_code == 200
    assert response.json()["entities"][0]["slug"] == "natsume"
    assert fake_db.rows == {}
    _post(_item())
    assert _post(_item(slug="other", name="natsume"), dry_run=True).status_code == 409
