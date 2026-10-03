"""Integration tests for profile slots against the throwaway Postgres.

Notes are seeded through ``save_note`` with the live embedder; profiles are written and
read through the REST routes.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import asyncpg
import pytest
from starlette.testclient import TestClient

from memory_base.core.config import PG_SCHEMA, db_url
from memory_base.serve import api, auth, namespaces
from memory_base.serve.notes import save_note

pytestmark = pytest.mark.integration

PERSONAL = (
    "The user lives in Seoul and works from home on Fridays.",
    "The user is vegetarian and avoids fish sauce.",
)
RULES = (
    "Delegate implementation work to a subagent in its own git worktree.",
    "Ask the user before merging any pull request.\nNever push to main directly.",
)

client = TestClient(api.app, headers={"X-API-Key": "test-key"})


@pytest.fixture(autouse=True)
def consolidator(monkeypatch):
    identity = auth.KeyIdentity(
        key_id="consolidator-key-hash",
        label="consolidator",
        home="default",
        is_admin=True,
        allowed=frozenset(),
        authors=frozenset({"consolidator"}),
    )

    async def fake_authenticate_request(plaintext_key):
        return identity if plaintext_key == "test-key" else None

    monkeypatch.setattr(auth, "authenticate_request", fake_authenticate_request)


async def _execute(sql, *args):
    conn = await asyncpg.connect(db_url())
    try:
        return await conn.execute(sql, *args)
    finally:
        await conn.close()


async def _seed(namespace):
    await namespaces.create_namespace(namespace)
    ids = {}
    for kind, texts in (("personal", PERSONAL), ("work", RULES)):
        ids[kind] = [
            (
                await save_note(
                    text,
                    kind=kind,
                    tags=["profile-test"],
                    namespace=namespace,
                    author="claude-code",
                    allow_similar=True,
                )
            )["id"]
            for text in texts
        ]
    return ids


async def _drop(namespace):
    for table in ("profiles", "memory_chunks", "namespaces"):
        column = "name" if table == "namespaces" else "namespace"
        await _execute(f'DELETE FROM "{PG_SCHEMA}".{table} WHERE {column} = $1', namespace)


def _sources(namespace, slot):
    response = client.get("/admin/profiles/sources", params={"namespace": namespace, "slot": slot})
    assert response.status_code == 200, response.json()
    return response.json()


def _write(namespace, slot, source_hash, **fields):
    body = {
        "namespace": namespace,
        "slot": slot,
        "source_hash": source_hash,
        "author": "consolidator",
        "model": "test-model",
        **fields,
    }
    return client.put("/admin/profiles", json=body)


def test_profiles_are_written_from_sources_served_and_kept_when_a_write_goes_stale():
    namespace = f"it-profiles-{uuid.uuid4().hex[:8]}"
    ids = asyncio.run(_seed(namespace))
    try:
        user_sources = _sources(namespace, "user")
        assert user_sources["stale"] is True
        assert user_sources["current"] is None
        assert sorted(n["id"] for n in user_sources["notes"]) == sorted(ids["personal"])
        assert {n["kind"] for n in user_sources["notes"]} == {"personal"}

        user = _write(
            namespace,
            "user",
            user_sources["source_hash"],
            content="Lives in Seoul, works from home on Fridays; vegetarian, no fish sauce.",
        )
        assert user.status_code == 200, user.json()
        assert user.json()["status"] == "written"
        assert user.json()["version"] == 1

        rules_sources = _sources(namespace, "work-rules")
        selection = [ids["work"][1], ids["work"][0]]
        planned = _write(
            namespace,
            "work-rules",
            rules_sources["source_hash"],
            note_ids=selection,
            dry_run=True,
        )
        expected = (
            "- Ask the user before merging any pull request.\n"
            "  Never push to main directly.\n"
            "- Delegate implementation work to a subagent in its own git worktree."
        )
        assert planned.json() == {"status": "planned", "content": expected, "chars": len(expected)}
        rules = _write(namespace, "work-rules", rules_sources["source_hash"], note_ids=selection)
        assert rules.json() == {"status": "written", "version": 1, "chars": len(expected)}

        served = client.get("/profiles", params={"namespace": namespace}).json()
        assert [(p["slot"], p["version"]) for p in served] == [("user", 1), ("work-rules", 1)]
        assert served[1]["content"] == expected
        assert _sources(namespace, "work-rules")["stale"] is False

        again = _write(namespace, "work-rules", rules_sources["source_hash"], note_ids=selection)
        assert again.json() == {"status": "unchanged", "version": 1}

        asyncio.run(
            _execute(
                f'UPDATE "{PG_SCHEMA}".memory_chunks SET archived_at = $2 WHERE id = $1',
                ids["work"][0],
                time.time(),
            )
        )
        stale = _sources(namespace, "work-rules")
        assert stale["stale"] is True
        assert stale["current"]["version"] == 1
        assert [n["id"] for n in stale["notes"]] == [ids["work"][1]]
        refused = _write(namespace, "work-rules", rules_sources["source_hash"], note_ids=selection)
        assert refused.status_code == 409
        assert refused.json() == {"error": "stale", "source_hash": stale["source_hash"]}
        served = client.get("/profiles", params={"namespace": namespace}).json()
        assert served[1]["version"] == 1
        assert served[1]["content"] == expected

        rewritten = _write(namespace, "work-rules", stale["source_hash"], note_ids=[ids["work"][1]])
        assert rewritten.json()["version"] == 2

        history = client.get(
            "/admin/profiles/versions", params={"namespace": namespace, "slot": "work-rules"}
        ).json()
        assert [v["version"] for v in history["versions"]] == [2, 1]
        assert history["versions"][1]["source_ids"] == selection
        assert history["versions"][1]["source_hash"] == rules_sources["source_hash"]
        assert history["versions"][0]["author"] == "consolidator"
        assert history["versions"][0]["model"] == "test-model"
    finally:
        asyncio.run(_drop(namespace))


def test_profile_history_alone_keeps_a_namespace_registered():
    namespace = f"it-profiles-{uuid.uuid4().hex[:8]}"
    asyncio.run(_seed(namespace))
    try:
        sources = _sources(namespace, "user")
        written = _write(namespace, "user", sources["source_hash"], content="Lives in Seoul.")
        assert written.json()["status"] == "written"
        asyncio.run(
            _execute(f'DELETE FROM "{PG_SCHEMA}".memory_chunks WHERE namespace = $1', namespace)
        )
        with pytest.raises(namespaces.NamespaceNotEmptyError):
            asyncio.run(namespaces.delete_namespace(namespace))
        asyncio.run(_execute(f'DELETE FROM "{PG_SCHEMA}".profiles WHERE namespace = $1', namespace))
        asyncio.run(namespaces.delete_namespace(namespace))
        assert not asyncio.run(namespaces.namespace_exists(namespace))
    finally:
        asyncio.run(_drop(namespace))
