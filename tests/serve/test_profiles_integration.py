"""Integration tests for agent-owned profiles against the throwaway Postgres.

Keys are real `api_keys` rows with authors, resolved by the real authentication path.
Concurrency tests hold the owner lock on a separate connection, queue the contending
operations behind it, and release it.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager

import asyncpg
import pytest
from starlette.testclient import TestClient

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA, db_url
from memory_base.serve import api, keys, namespaces
from memory_base.serve.profiles import store

pytestmark = [pytest.mark.integration, pytest.mark.real_auth]

LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtextextended('profile:' || $1, 0))"


def _owner():
    return f"it-{uuid.uuid4().hex[:10]}"


async def _mint(label, is_admin, authors):
    plaintext = await keys.new_key(label, "default", is_admin)
    await keys.set_authors(label, authors)
    await db.close_pool()
    return plaintext


def _client(owners=(), authors=None, is_admin=True):
    label = f"it-key-{uuid.uuid4().hex[:8]}"
    plaintext = asyncio.run(
        _mint(label, is_admin, list(authors if authors is not None else owners))
    )
    return TestClient(api.app, headers={"X-API-Key": plaintext})


def _user_client():
    return _client(authors=["user"], is_admin=False)


async def _fetch(sql, *args):
    conn = await asyncpg.connect(db_url())
    try:
        return await conn.fetch(sql, *args)
    finally:
        await conn.close()


def _proposals(owner):
    return asyncio.run(
        _fetch(
            f'SELECT id, status, decided_at, decision_note FROM "{PG_SCHEMA}".profile_proposals '
            "WHERE owner = $1 ORDER BY id",
            owner,
        )
    )


def _versions(owner, part):
    return asyncio.run(
        _fetch(
            f'SELECT version, content, author, proposal_id FROM "{PG_SCHEMA}".agent_profiles '
            "WHERE owner = $1 AND part = $2 ORDER BY version",
            owner,
            part,
        )
    )


def _propose(api_client, owner, content, base_version, reason="learned it"):
    return api_client.post(
        "/profiles/user/proposals",
        json={"owner": owner, "content": content, "reason": reason, "base_version": base_version},
    )


def test_an_owner_writes_self_proposes_and_the_user_approves():
    owner, other = _owner(), _owner()
    agents = _client([owner, other])
    user = _user_client()

    written = agents.put("/profiles/self", json={"owner": owner, "content": "I review diffs."})
    assert written.json() == {"status": "written", "version": 1}

    first = _propose(agents, owner, "Lives in Busan.", 0)
    assert first.status_code == 201, first.json()
    second = _propose(agents, owner, "Lives in Seoul.", 0, reason="they moved")
    assert second.status_code == 201
    assert second.json()["superseded"] == first.json()["id"]
    second_id = second.json()["id"]

    denied = agents.post(f"/profiles/user/proposals/{second_id}/approve", json={})
    assert denied.status_code == 403
    pending = user.get("/profiles/user/proposals", params={"status": "pending"}).json()
    assert [row["id"] for row in pending if row["owner"] == owner] == [second_id]

    shown = user.get(f"/profiles/user/proposals/{second_id}").json()
    assert shown["current_user_version"] == 0
    assert shown["current_user_content"] is None
    assert shown["content"] == "Lives in Seoul."

    approved = user.post(f"/profiles/user/proposals/{second_id}/approve", json={"note": "yes"})
    assert approved.json() == {"status": "approved", "version": 1}
    stale = _propose(agents, owner, "Lives in Daegu.", 0)
    assert stale.status_code == 409
    assert stale.json() == {"error": "stale", "version": 1}

    served = agents.get("/profiles", params={"owner": owner}).json()
    assert served["self_version"] == 1
    assert served["self"]["content"] == "I review diffs."
    assert served["user_version"] == 1
    assert served["user"]["content"] == "Lives in Seoul."
    assert served["pending_proposal"] is None
    assert served["user"]["created_at"].endswith("+00:00")

    history = user.get("/profiles/versions", params={"owner": owner, "part": "user"}).json()
    assert [(row["version"], row["author"], row["proposal_id"]) for row in history] == [
        (1, "user", second_id)
    ]
    rows = _proposals(owner)
    assert [row["status"] for row in rows] == ["superseded", "approved"]
    assert rows[1]["decision_note"] == "yes"
    assert rows[0]["decision_note"] is None
    assert rows[0]["decided_at"] is not None

    untouched = agents.get("/profiles", params={"owner": other}).json()
    assert untouched["user_version"] == 0 and untouched["self_version"] == 0
    assert _propose(agents, other, "Natsume's view.", 0).status_code == 201

    cleared = _propose(agents, owner, "", 1, reason="nothing holds")
    assert cleared.status_code == 201
    done = user.post(f"/profiles/user/proposals/{cleared.json()['id']}/approve", json={})
    assert done.json() == {"status": "approved", "version": 2}
    served = agents.get("/profiles", params={"owner": owner}).json()
    assert served["user_version"] == 2
    assert served["user"] is None
    assert _propose(agents, owner, "Lives in Seoul again.", 2).status_code == 201


def test_a_key_without_the_owner_or_user_author_is_refused():
    owner = _owner()
    agents = _client([owner])
    stranger = _client(authors=["natsume-other"])
    assert stranger.get("/profiles", params={"owner": owner}).status_code == 403
    assert stranger.put("/profiles/self", json={"owner": owner, "content": "x"}).status_code == 403
    proposal = _propose(agents, owner, "x", 0).json()["id"]
    assert stranger.get(f"/profiles/user/proposals/{proposal}").status_code == 403
    assert stranger.post(f"/profiles/user/proposals/{proposal}/reject", json={}).status_code == 403
    assert TestClient(api.app).get("/profiles", params={"owner": owner}).status_code == 401


def test_a_revoked_user_key_cannot_decide():
    owner = _owner()
    agents = _client([owner])
    label = f"it-key-{uuid.uuid4().hex[:8]}"
    plaintext = asyncio.run(_mint(label, False, ["user"]))

    async def revoke():
        await keys.revoke_key(keys.hash_key(plaintext))
        await db.close_pool()

    asyncio.run(revoke())
    proposal = _propose(agents, owner, "x", 0).json()["id"]
    revoked = TestClient(api.app, headers={"X-API-Key": plaintext})
    assert revoked.post(f"/profiles/user/proposals/{proposal}/approve", json={}).status_code == 401


def test_namespace_deletion_ignores_profiles():
    owner = _owner()
    agents = _client([owner])
    name = f"it-profiles-{uuid.uuid4().hex[:8]}"
    asyncio.run(namespaces.create_namespace(name))
    assert agents.put("/profiles/self", json={"owner": owner, "content": "x"}).status_code == 200
    assert _propose(agents, owner, "y", 0).status_code == 201
    asyncio.run(namespaces.delete_namespace(name))
    assert not asyncio.run(namespaces.namespace_exists(name))


# ---- concurrency ------------------------------------------------------------


async def _queue_behind_the_lock(owner, operations):
    """Start each operation while another connection holds the owner lock, then release it."""
    holder = await asyncpg.connect(db_url())
    tasks = []
    try:
        transaction = holder.transaction()
        await transaction.start()
        await holder.execute(LOCK_SQL, owner)
        for operation in operations:
            tasks.append(asyncio.ensure_future(operation()))
            await _wait_for_waiters(holder, len(tasks))
        await transaction.commit()
        return await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        await holder.close()
        await db.close_pool()


async def _wait_for_waiters(conn, count):
    for _ in range(500):
        waiting = await conn.fetchval(
            "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND NOT granted"
        )
        if waiting >= count:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"{count} operations never queued behind the owner lock")


async def _seed_proposal(owner, content="x", base_version=0):
    result = await store.propose(owner, content, "seed", base_version)
    await db.close_pool()
    return result["id"]


def test_concurrent_self_writes_take_unique_sequential_versions():
    owner = _owner()
    results = asyncio.run(
        _queue_behind_the_lock(
            owner, [lambda i=i: store.write_self(owner, f"text {i}") for i in range(4)]
        )
    )
    assert sorted(result["version"] for result in results) == [1, 2, 3, 4]
    assert [row["version"] for row in _versions(owner, "self")] == [1, 2, 3, 4]


def test_concurrent_proposals_leave_one_pending():
    owner = _owner()
    results = asyncio.run(
        _queue_behind_the_lock(
            owner, [lambda i=i: store.propose(owner, f"p{i}", "r", 0) for i in range(4)]
        )
    )
    assert all(result["status"] == "pending" for result in results)
    rows = _proposals(owner)
    assert [row["status"] for row in rows].count("pending") == 1
    assert [row["status"] for row in rows].count("superseded") == 3
    assert sum(result["superseded"] is None for result in results) == 1


def test_duplicate_approval_writes_one_version():
    owner = _owner()
    proposal = asyncio.run(_seed_proposal(owner))
    results = asyncio.run(
        _queue_behind_the_lock(
            owner,
            [lambda: store.approve(proposal, None), lambda: store.approve(proposal, None)],
        )
    )
    assert results[0] == {"status": "approved", "version": 1}
    assert isinstance(results[1], store.NotPending)
    assert results[1].status == "approved"
    assert len(_versions(owner, "user")) == 1


def test_approve_and_reject_race_to_one_terminal_decision():
    owner = _owner()
    proposal = asyncio.run(_seed_proposal(owner))
    results = asyncio.run(
        _queue_behind_the_lock(
            owner,
            [lambda: store.reject(proposal, "no"), lambda: store.approve(proposal, "yes")],
        )
    )
    assert results[0] == {"status": "rejected"}
    assert isinstance(results[1], store.NotPending)
    assert results[1].status == "rejected"
    assert _versions(owner, "user") == []
    (row,) = _proposals(owner)
    assert (row["status"], row["decision_note"]) == ("rejected", "no")


def test_a_proposal_and_an_approval_never_approve_a_stale_base():
    owner = _owner()
    proposal = asyncio.run(_seed_proposal(owner))
    results = asyncio.run(
        _queue_behind_the_lock(
            owner,
            [
                lambda: store.approve(proposal, None),
                lambda: store.propose(owner, "newer", "r", 0),
            ],
        )
    )
    approved, proposed = results
    if isinstance(approved, dict):
        assert approved == {"status": "approved", "version": 1}
        assert isinstance(proposed, store.Stale)
        assert proposed.version == 1
    else:
        assert isinstance(approved, store.NotPending)
        assert approved.status == "superseded"
        assert proposed["superseded"] == proposal
    statuses = [row["status"] for row in _proposals(owner)]
    assert statuses.count("pending") <= 1
    versions = _versions(owner, "user")
    assert len(versions) <= 1
    assert all(row["proposal_id"] == proposal for row in versions)


# ---- rollback ---------------------------------------------------------------


class FailingConnection:
    """Delegates to a real connection and fails the first statement that `fails` matches."""

    def __init__(self, conn, fails):
        self._conn = conn
        self._fails = fails

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def _check(self, query):
        if self._fails(" ".join(query.split())):
            raise RuntimeError("injected failure")

    async def execute(self, query, *args):
        self._check(query)
        return await self._conn.execute(query, *args)

    async def fetchval(self, query, *args):
        self._check(query)
        return await self._conn.fetchval(query, *args)

    async def fetchrow(self, query, *args):
        self._check(query)
        return await self._conn.fetchrow(query, *args)

    async def fetch(self, query, *args):
        self._check(query)
        return await self._conn.fetch(query, *args)


def _inject(monkeypatch, fails):
    original = db.acquire

    @asynccontextmanager
    async def acquire(timeout=None):
        async with original() as conn:
            yield FailingConnection(conn, fails)

    monkeypatch.setattr(db, "acquire", acquire)


async def _attempt(operation):
    try:
        with pytest.raises(RuntimeError, match="injected failure"):
            await operation()
    finally:
        await db.close_pool()


def test_a_failure_between_supersede_and_insert_rolls_both_back(monkeypatch):
    owner = _owner()
    first = asyncio.run(_seed_proposal(owner))
    _inject(monkeypatch, lambda q: q.startswith("INSERT INTO") and "profile_proposals" in q)
    asyncio.run(_attempt(lambda: store.propose(owner, "second", "r", 0)))
    rows = _proposals(owner)
    assert [(row["id"], row["status"], row["decided_at"]) for row in rows] == [
        (first, "pending", None)
    ]


def test_a_failure_between_version_insert_and_decision_rolls_both_back(monkeypatch):
    owner = _owner()
    proposal = asyncio.run(_seed_proposal(owner))
    _inject(monkeypatch, lambda q: q.startswith("UPDATE") and "profile_proposals" in q)
    asyncio.run(_attempt(lambda: store.approve(proposal, "yes")))
    assert _versions(owner, "user") == []
    (row,) = _proposals(owner)
    assert (row["status"], row["decided_at"], row["decision_note"]) == ("pending", None, None)
