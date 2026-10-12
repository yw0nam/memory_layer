"""Integration tests for agent-owned profiles against the throwaway Postgres.

Keys are real `api_keys` rows with authors, resolved by the real authentication path.
Concurrency tests hold the owner lock on a separate connection, queue the contending
operations behind it, and release it. The migration test runs `ensure_schema` over the
earlier table shapes in a scratch schema.
"""

from __future__ import annotations

import asyncio
import uuid

import asyncpg
import pytest
from starlette.testclient import TestClient

from memory_base.core import db, schema
from memory_base.core.config import PG_SCHEMA, db_url
from memory_base.serve import api
from memory_base.serve.access import keys, namespaces
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


def _versions(owner, part):
    return asyncio.run(
        _fetch(
            "SELECT version, content, author, reason, consent_method, consented_at "
            f'FROM "{PG_SCHEMA}".agent_profiles WHERE owner = $1 AND part = $2 ORDER BY version',
            owner,
            part,
        )
    )


def _put_user(api_client, owner, content, base_version, **fields):
    body = {
        "owner": owner,
        "content": content,
        "reason": "learned it",
        "base_version": base_version,
        "consent_method": "conversation",
    }
    return api_client.put("/profiles/user", json={**body, **fields})


def test_an_owner_writes_self_and_the_user_part_with_consent():
    owner, other = _owner(), _owner()
    agents = _client([owner, other])
    user = _user_client()

    written = agents.put("/profiles/self", json={"owner": owner, "content": "I review diffs."})
    assert written.json() == {"status": "written", "version": 1}

    dry = _put_user(agents, owner, "Lives in Seoul.", 0, dry_run=True)
    assert dry.json() == {"status": "valid", "version": 0, "current": ""}
    assert _versions(owner, "user") == []

    first = _put_user(agents, owner, "Lives in Seoul.", 0, reason="they moved")
    assert first.json() == {"status": "written", "version": 1}
    assert _put_user(agents, owner, "Lives in Seoul.", 1).json() == {
        "status": "unchanged",
        "version": 1,
    }
    stale = _put_user(agents, owner, "Lives in Daegu.", 0)
    assert stale.status_code == 409
    assert stale.json() == {"error": "stale", "version": 1}
    dry = _put_user(agents, owner, "Lives in Daegu.", 1, dry_run=True)
    assert dry.json() == {"status": "valid", "version": 1, "current": "Lives in Seoul."}

    (row,) = _versions(owner, "user")
    assert (row["author"], row["reason"], row["consent_method"]) == (
        owner,
        "they moved",
        "conversation",
    )
    assert row["consented_at"] is not None

    served = user.get("/profiles", params={"owner": owner}).json()
    assert served["user_version"] == 1
    assert served["user"]["content"] == "Lives in Seoul."
    assert "pending_proposal" not in served
    history = user.get("/profiles/versions", params={"owner": owner, "part": "user"}).json()
    assert [(r["version"], r["consent_method"]) for r in history] == [(1, "conversation")]
    assert history[0]["consented_at"].endswith("+00:00")

    cleared = _put_user(agents, owner, "", 1, consent_method="elicitation")
    assert cleared.json() == {"status": "written", "version": 2}
    assert agents.get("/profiles", params={"owner": owner}).json()["user"] is None
    assert agents.get("/profiles", params={"owner": other}).json()["user_version"] == 0


def test_a_key_without_the_owner_or_user_author_is_refused():
    owner = _owner()
    stranger = _client(authors=["natsume-other"])
    assert stranger.get("/profiles", params={"owner": owner}).status_code == 403
    assert stranger.put("/profiles/self", json={"owner": owner, "content": "x"}).status_code == 403
    assert _put_user(stranger, owner, "x", 0).status_code == 403
    assert TestClient(api.app).get("/profiles", params={"owner": owner}).status_code == 401


def test_the_import_method_needs_an_admin_key_with_the_user_author():
    owner = _owner()
    plain = _client([owner], is_admin=False)
    assert _put_user(plain, owner, "x", 0, consent_method="import").status_code == 403
    importer = _client([owner, "user"], is_admin=True)
    assert _put_user(importer, owner, "x", 0, consent_method="import").status_code == 200
    assert _versions(owner, "user")[0]["consent_method"] == "import"


def test_namespace_deletion_ignores_profiles():
    owner = _owner()
    agents = _client([owner])
    name = f"it-profiles-{uuid.uuid4().hex[:8]}"
    asyncio.run(namespaces.create_namespace(name))
    assert agents.put("/profiles/self", json={"owner": owner, "content": "x"}).status_code == 200
    assert _put_user(agents, owner, "y", 0).status_code == 200
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


def test_concurrent_self_writes_take_unique_sequential_versions():
    owner = _owner()
    results = asyncio.run(
        _queue_behind_the_lock(
            owner, [lambda i=i: store.write_self(owner, f"text {i}") for i in range(4)]
        )
    )
    assert sorted(result["version"] for result in results) == [1, 2, 3, 4]
    assert [row["version"] for row in _versions(owner, "self")] == [1, 2, 3, 4]


def test_concurrent_user_writes_against_one_base_store_exactly_one_version():
    owner = _owner()
    results = asyncio.run(
        _queue_behind_the_lock(
            owner,
            [
                lambda i=i: store.write_user(owner, f"text {i}", "r", 0, "conversation", False)
                for i in range(4)
            ],
        )
    )
    assert [r for r in results if isinstance(r, dict)] == [{"status": "written", "version": 1}]
    assert sum(isinstance(r, store.Stale) for r in results) == 3
    assert [row["version"] for row in _versions(owner, "user")] == [1]


# ---- migration --------------------------------------------------------------


async def _run_migration(monkeypatch):
    name = f"it_migrate_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_url())
    try:
        await conn.execute(
            f"""
            CREATE SCHEMA "{name}";
            CREATE TABLE "{name}".profile_proposals (
              id bigserial PRIMARY KEY, owner text NOT NULL, content text NOT NULL,
              reason text NOT NULL, base_version int NOT NULL, status text NOT NULL,
              created_at double precision NOT NULL, decided_at double precision,
              decision_note text
            );
            CREATE TABLE "{name}".agent_profiles (
              id bigserial PRIMARY KEY, owner text NOT NULL, part text NOT NULL,
              version int NOT NULL, content text NOT NULL, author text NOT NULL,
              proposal_id bigint, created_at double precision NOT NULL,
              UNIQUE (owner, part, version)
            );
            INSERT INTO "{name}".profile_proposals
              (owner, content, reason, base_version, status, created_at)
              VALUES ('o', 'c', 'they moved', 0, 'approved', 1);
            INSERT INTO "{name}".agent_profiles
              (owner, part, version, content, author, proposal_id, created_at)
              VALUES ('o', 'user', 1, 'c', 'user', 1, 1), ('o', 'self', 1, 's', 'o', NULL, 1);
            """
        )
        monkeypatch.setattr(schema, "PG_SCHEMA", name)
        await schema.ensure_schema(conn)
        await schema.ensure_schema(conn)
        rows = await conn.fetch(
            f'SELECT part, reason, consent_method FROM "{name}".agent_profiles ORDER BY part'
        )
        columns = await conn.fetchval(
            "SELECT array_agg(column_name::text) FROM information_schema.columns "
            "WHERE table_schema = $1 AND table_name = 'agent_profiles'",
            name,
        )
        remaining = await conn.fetchval(
            "SELECT to_regclass($1)::text", f'"{name}".profile_proposals'
        )
        return rows, columns, remaining
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{name}" CASCADE')
        await conn.close()


def test_ensure_schema_carries_a_proposal_reason_onto_its_version_and_drops_the_table(
    monkeypatch,
):
    rows, columns, remaining = asyncio.run(_run_migration(monkeypatch))
    assert [(r["part"], r["reason"], r["consent_method"]) for r in rows] == [
        ("self", None, None),
        ("user", "they moved", None),
    ]
    assert "proposal_id" not in columns
    assert {"reason", "consent_method", "consented_at"} <= set(columns)
    assert remaining is None
