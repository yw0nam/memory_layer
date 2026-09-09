"""Unit tests for the author-allowlist routes and the keys functions behind them.

No DB: ``keys.get_authors``/``keys.set_authors`` are monkeypatched for the route
tests, and exercised against a fake connection for the SQL shape, matching the
convention in tests/serve/test_keys_unit.py.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest
from starlette.testclient import TestClient

from memory_base.serve import api, auth, keys

client = TestClient(api.app, headers={"X-API-Key": "test-key"})


def _member_client(monkeypatch, label="alice"):
    identity = auth.KeyIdentity(
        key_id="alice-hash",
        label=label,
        home="default",
        is_admin=False,
        allowed=frozenset({"default"}),
        authors=frozenset({"claude-code"}),
    )

    async def fake_authenticate_request(plaintext_key):
        return identity if plaintext_key == "member-key" else None

    monkeypatch.setattr(auth, "authenticate_request", fake_authenticate_request)
    return TestClient(api.app, headers={"X-API-Key": "member-key"})


# ---- keys.validate_authors ---------------------------------------------------


def test_validate_authors_sorts_and_deduplicates():
    assert keys.validate_authors(["natsume", "claude-code", "natsume"]) == [
        "claude-code",
        "natsume",
    ]


def test_validate_authors_accepts_an_empty_list():
    assert keys.validate_authors([]) == []


@pytest.mark.parametrize("authors", ["natsume", {"a": 1}, None, [1], ["ok", None]])
def test_validate_authors_rejects_non_string_lists(authors):
    with pytest.raises(keys.AuthorError, match="list of strings"):
        keys.validate_authors(authors)


@pytest.mark.parametrize(
    "author", ["Natsume", "-natsume", "natsume_bot", "natsume bot", "", "a" * 41]
)
def test_validate_authors_rejects_bad_slugs(author):
    with pytest.raises(keys.AuthorError, match="a-z0-9"):
        keys.validate_authors([author])


# ---- keys.get_authors / keys.set_authors ------------------------------------


class FakeConnection:
    def __init__(self, fetch_results=None, status="UPDATE 1"):
        self._fetch = list(fetch_results or [])
        self.status = status
        self.queries: list[tuple] = []

    async def fetch(self, query, *args):
        self.queries.append((query, args))
        return self._fetch.pop(0)

    async def execute(self, query, *args):
        self.queries.append((query, args))
        return self.status


def _patch_acquire(monkeypatch, conn):
    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    monkeypatch.setattr(keys.db, "acquire", acquire)

    async def _noop_ensure_schema_once(conn):
        return None

    monkeypatch.setattr(keys, "ensure_schema_once", _noop_ensure_schema_once)


def test_get_authors_unions_rows_sorted(monkeypatch):
    conn = FakeConnection(fetch_results=[[{"authors": ["natsume"]}, {"authors": ["claude-code"]}]])
    _patch_acquire(monkeypatch, conn)
    assert asyncio.run(keys.get_authors("alice")) == ["claude-code", "natsume"]
    query, args = conn.queries[0]
    assert "revoked_at IS NULL" in query
    assert args == ("alice",)


def test_get_authors_unknown_label_is_none(monkeypatch):
    conn = FakeConnection(fetch_results=[[]])
    _patch_acquire(monkeypatch, conn)
    assert asyncio.run(keys.get_authors("ghost")) is None


def test_set_authors_replaces_every_non_revoked_row(monkeypatch):
    conn = FakeConnection(status="UPDATE 2")
    _patch_acquire(monkeypatch, conn)
    assert asyncio.run(keys.set_authors("alice", ["claude-code"])) == ["claude-code"]
    query, args = conn.queries[0]
    assert "SET authors = $2::text[]" in query
    assert "revoked_at IS NULL" in query
    assert args == ("alice", ["claude-code"])


def test_set_authors_unknown_label_is_none(monkeypatch):
    conn = FakeConnection(status="UPDATE 0")
    _patch_acquire(monkeypatch, conn)
    assert asyncio.run(keys.set_authors("ghost", ["claude-code"])) is None


# ---- GET /keys/{label}/authors ----------------------------------------------


def test_admin_reads_any_label(monkeypatch):
    async def fake_get_authors(label):
        assert label == "bob"
        return ["natsume"]

    monkeypatch.setattr(keys, "get_authors", fake_get_authors)
    response = client.get("/keys/bob/authors")
    assert response.status_code == 200
    assert response.json() == {"label": "bob", "authors": ["natsume"]}


def test_member_reads_its_own_label(monkeypatch):
    member = _member_client(monkeypatch)

    async def fake_get_authors(label):
        return ["claude-code"]

    monkeypatch.setattr(keys, "get_authors", fake_get_authors)
    response = member.get("/keys/alice/authors")
    assert response.status_code == 200
    assert response.json() == {"label": "alice", "authors": ["claude-code"]}


def test_member_reading_another_label_403(monkeypatch):
    member = _member_client(monkeypatch)

    async def unreachable(label):
        raise AssertionError("must not read another label's allowlist")

    monkeypatch.setattr(keys, "get_authors", unreachable)
    response = member.get("/keys/bob/authors")
    assert response.status_code == 403
    assert "error" in response.json()


def test_get_unknown_label_404(monkeypatch):
    async def fake_get_authors(label):
        return None

    monkeypatch.setattr(keys, "get_authors", fake_get_authors)
    response = client.get("/keys/ghost/authors")
    assert response.status_code == 404
    assert "error" in response.json()


# ---- PUT /keys/{label}/authors ----------------------------------------------


def test_put_replaces_the_whole_list(monkeypatch):
    captured = {}

    async def fake_set_authors(label, authors):
        captured["label"] = label
        captured["authors"] = authors
        return authors

    monkeypatch.setattr(keys, "set_authors", fake_set_authors)
    response = client.put(
        "/keys/alice/authors", json={"authors": ["natsume", "claude-code", "natsume"]}
    )
    assert response.status_code == 200
    assert response.json() == {"label": "alice", "authors": ["claude-code", "natsume"]}
    assert captured == {"label": "alice", "authors": ["claude-code", "natsume"]}


def test_put_requires_an_admin_key(monkeypatch):
    member = _member_client(monkeypatch)

    async def unreachable(label, authors):
        raise AssertionError("must not write an allowlist without an admin key")

    monkeypatch.setattr(keys, "set_authors", unreachable)
    response = member.put("/keys/alice/authors", json={"authors": ["claude-code"]})
    assert response.status_code == 403
    assert "error" in response.json()


@pytest.mark.parametrize("authors", ["natsume", None, [1], ["Natsume"], ["natsume_bot"]])
def test_put_rejects_malformed_authors_400(monkeypatch, authors):
    async def unreachable(label, authors):
        raise AssertionError("must not write a malformed allowlist")

    monkeypatch.setattr(keys, "set_authors", unreachable)
    response = client.put("/keys/alice/authors", json={"authors": authors})
    assert response.status_code == 400
    assert "error" in response.json()


def test_put_malformed_json_400():
    response = client.put(
        "/keys/alice/authors",
        content=b"{not valid json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    assert "error" in response.json()


def test_put_unknown_label_404(monkeypatch):
    async def fake_set_authors(label, authors):
        return None

    monkeypatch.setattr(keys, "set_authors", fake_set_authors)
    response = client.put("/keys/ghost/authors", json={"authors": ["natsume"]})
    assert response.status_code == 404
    assert "error" in response.json()
