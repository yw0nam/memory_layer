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
from memory_base.serve.access import auth
from memory_base.serve.entities import store

client = TestClient(api.app, headers={"X-API-Key": "test-key"})

NOW = 1_700_000_000.0
GITHUB_TOKEN = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
COLUMNS = (
    "slug kind name aliases description created_by updated_by consent_method "
    "consented_at created_at updated_at archived_at"
).split()


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
        q = " ".join(query.split())
        if "pg_advisory_xact_lock" in q:
            self.locked = True
            return "SELECT 1"
        assert self.locked, "every read and write runs under the registry lock"
        if q.startswith("INSERT INTO"):
            row = dict(zip(COLUMNS, args)) | {"archived_at": None}
            self.rows[row["slug"]] = row
        elif q.startswith("UPDATE") and "archived_at = $" in q:
            slug, now, author, method = args
            self.rows[slug].update(
                archived_at=now, updated_at=now, updated_by=author,
                consent_method=method, consented_at=now,
            )  # fmt: skip
        elif q.startswith("UPDATE"):
            slug, kind, name, aliases, description, author, method, now = args
            self.rows[slug].update(
                kind=kind, name=name, aliases=aliases, description=description,
                updated_by=author, consent_method=method, consented_at=now, updated_at=now,
            )  # fmt: skip
        else:
            raise AssertionError(f"unexpected execute: {q}")
        return "OK"

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


def _use_identity(monkeypatch, authors, is_admin=False):
    identity = auth.KeyIdentity(
        key_id="k", label="agents", home="default", is_admin=is_admin,
        allowed=frozenset(), authors=frozenset(authors),
    )  # fmt: skip

    async def fake_authenticate_request(plaintext_key):
        return identity if plaintext_key == "test-key" else None

    monkeypatch.setattr(auth, "authenticate_request", fake_authenticate_request)


@pytest.fixture()
def agent(monkeypatch):
    _use_identity(monkeypatch, ("claude-code",))


def _item(slug="natsume", kind="person", name="Natsume", **fields):
    return {"slug": slug, "kind": kind, "name": name, **fields}


def _post(*items, author="claude-code", consent_method="conversation"):
    return client.post(
        "/entities",
        json={"entities": list(items), "author": author, "consent_method": consent_method},
    )


def _archive(slug="natsume", author="claude-code", consent_method="conversation"):
    return client.post(
        f"/entities/{slug}/archive", json={"author": author, "consent_method": consent_method}
    )


# ---- create and edit --------------------------------------------------------


def test_create_stores_the_row_with_consent(fake_db, agent):
    response = _post(_item(aliases=["Nat"], description="A friend"), consent_method="elicitation")
    assert response.status_code == 200
    [row] = response.json()["entities"]
    assert row["slug"] == "natsume"
    assert row["aliases"] == ["Nat"]
    assert row["created_by"] == row["updated_by"] == "claude-code"
    assert row["consent_method"] == "elicitation"
    assert row["consented_at"] == row["created_at"] == "2023-11-14T22:13:20+00:00"
    assert row["archived_at"] is None


def test_edit_keeps_omitted_fields_and_replaces_aliases(fake_db, agent):
    _post(_item(aliases=["Nat", "Natchan"], description="A friend"))
    response = _post({"slug": "natsume", "aliases": ["Natsu"], "name": "Natsume K"})
    [row] = response.json()["entities"]
    assert (row["kind"], row["name"], row["description"]) == ("person", "Natsume K", "A friend")
    assert row["aliases"] == ["Natsu"]


def test_edit_of_an_archived_entity_is_refused(fake_db, agent):
    _post(_item())
    _archive()
    response = _post({"slug": "natsume", "name": "Other"})
    assert response.status_code == 400
    assert "archived; restore is not supported" in response.json()["error"]


def test_create_without_kind_or_name_is_refused(fake_db, agent):
    response = _post({"slug": "natsume", "name": "Natsume"})
    assert response.status_code == 400
    assert "kind" in response.json()["error"]


def test_invalid_kind_is_refused(fake_db, agent):
    assert _post(_item(kind="robot")).status_code == 400
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


def test_a_name_collides_by_nfkc_and_case(fake_db, agent):
    _post(_item(slug="yui", name="ＹＵＩ"))
    _collision(_post(_item(slug="yui2", name="yui")), "yui", "yui")


def test_a_new_slug_collides_with_an_existing_name(fake_db, agent):
    _post(_item(slug="natsume", name="Natsume"))
    _collision(_post(_item(slug="other", name="Other", aliases=["natsume"])), "natsume", "natsume")


def test_an_alias_collides_with_another_entitys_alias(fake_db, agent):
    _post(_item(aliases=["Nat"]))
    _collision(_post(_item(slug="other", name="Other", aliases=["nat"])), "natsume", "nat")


def test_a_new_slug_equal_to_an_existing_alias_collides(fake_db, agent):
    _post(_item(aliases=["nat"]))
    _collision(_post(_item(slug="nat", name="Nat Two")), "natsume", "nat")


def test_a_repeated_slug_in_one_batch_is_refused(fake_db, agent):
    response = _post(_item(), _item(name="Another"))
    assert response.status_code == 400
    assert "natsume" in response.json()["error"]
    assert fake_db.rows == {}


def test_an_archived_entitys_name_stays_reserved(fake_db, agent):
    _post(_item())
    _archive()
    _collision(_post(_item(slug="new", name="natsume")), "natsume", "natsume")


def test_a_colliding_second_item_writes_nothing(fake_db, agent):
    response = _post(_item(slug="a", name="A"), _item(slug="b", name="a"))
    assert response.status_code == 409
    assert fake_db.rows == {}


def test_a_credential_in_the_description_is_refused(fake_db, agent):
    response = _post(_item(description=f"token {GITHUB_TOKEN}"))
    assert response.status_code == 400
    assert "credential" in response.json()["error"]
    assert fake_db.rows == {}


# ---- archive and list -------------------------------------------------------


def test_archive_sets_the_fields_and_refuses_twice(fake_db, agent):
    _post(_item())
    response = _archive(consent_method="elicitation")
    assert response.status_code == 200
    row = response.json()["entity"]
    assert row["archived_at"] == "2023-11-14T22:13:20+00:00"
    assert row["consent_method"] == "elicitation"
    assert _archive().status_code == 400
    assert _archive(slug="ghost").status_code == 400


def test_list_excludes_archived_unless_asked(fake_db, agent):
    _post(_item(), _item(slug="yui", name="Yui"))
    _archive()
    slugs = lambda r: [e["slug"] for e in r.json()["entities"]]  # noqa: E731
    assert slugs(client.get("/entities")) == ["yui"]
    assert slugs(client.get("/entities?include_archived=true")) == ["natsume", "yui"]


# ---- authority --------------------------------------------------------------


def test_an_author_outside_the_allowlist_is_forbidden(fake_db, agent):
    assert _post(_item(), author="someone-else").status_code == 403
    assert _archive(author="someone-else").status_code == 403


def test_import_needs_an_admin_key_carrying_the_user_author(fake_db, agent, monkeypatch):
    assert _post(_item(), consent_method="import").status_code == 403
    _use_identity(monkeypatch, ("user",), is_admin=False)
    assert _post(_item(), author="user", consent_method="import").status_code == 403
    _use_identity(monkeypatch, ("user",), is_admin=True)
    assert _post(_item(), author="user", consent_method="import").status_code == 200


def test_an_unknown_consent_method_is_a_bad_request(fake_db, agent):
    assert _post(_item(), consent_method="telepathy").status_code == 400
