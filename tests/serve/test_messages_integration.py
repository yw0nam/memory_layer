"""Integration tests for the message lane against the live DB (postgres:5439).

Covers what a mocked connection cannot: the real `messages` schema, atomic
at-most-once claims under real concurrency (two separate backends), the
handoff snapshot supersede chain, expiry, idempotency-key release, namespace
deletion, and the admin purge — all through the real REST app plus dedicated
asyncpg connections. Every seeded row is deleted in ``finally``. Skipped when
the DB is unreachable, matching the other integration modules.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import asyncpg
import pytest
from starlette.testclient import TestClient

from memory_base.core.config import PG_SCHEMA, db_url
from memory_base.core.schema import ensure_schema
from memory_base.serve import api, auth, messages, notes

pytestmark = pytest.mark.integration

client = TestClient(api.app, headers={"X-API-Key": "test-key"})

IDENTITY = auth.KeyIdentity(
    key_id="test-key-hash",
    label="test",
    home="default",
    is_admin=True,
    allowed=frozenset(),
    authors=frozenset({"claude-code", "natsume"}),
)


def _db_reachable() -> bool:
    async def _check() -> None:
        conn = await asyncpg.connect(db_url(), timeout=5)
        await conn.close()

    try:
        asyncio.run(_check())
        return True
    except Exception:
        return False


if not _db_reachable():
    pytest.skip("DB is not configured or not reachable", allow_module_level=True)


@asynccontextmanager
async def connections(count=2):
    opened = [await asyncpg.connect(db_url()) for _ in range(count)]
    try:
        await ensure_schema(opened[0])
        yield opened
    finally:
        for connection in opened:
            await connection.close()


async def _cleanup(marker: str, namespace: str | None = None) -> None:
    conn = await asyncpg.connect(db_url())
    try:
        await conn.execute(
            f'DELETE FROM "{PG_SCHEMA}".messages WHERE subject LIKE $1', f"{marker}%"
        )
        if namespace is not None:
            await conn.execute(f'DELETE FROM "{PG_SCHEMA}".namespaces WHERE name = $1', namespace)
    finally:
        await conn.close()


def _member_client(monkeypatch, label, allowed):
    identity = auth.KeyIdentity(
        key_id=f"{label}-hash",
        label=label,
        home="default",
        is_admin=False,
        allowed=frozenset(allowed),
        authors=frozenset({"claude-code"}),
    )

    async def fake_authenticate_request(plaintext_key):
        return identity if plaintext_key == "member-key" else None

    monkeypatch.setattr(auth, "authenticate_request", fake_authenticate_request)
    return TestClient(api.app, headers={"X-API-Key": "member-key"})


def _send(subject, status="info", result="r", **overrides):
    body = {"subject": subject, "status": status, "result": result, "author": "claude-code"}
    body.update(overrides)
    return client.post("/messages", json=body)


def _send_handoff(subject, status="in_progress", **overrides):
    body = {
        "subject": subject,
        "status": status,
        "result": "state of the work",
        "next": "carry on here",
        "scope": "repo:github.com/o/r",
        "author": "claude-code",
    }
    body.update(overrides)
    return client.post("/messages", json=body)


# ---- schema -----------------------------------------------------------------


def test_schema_creates_messages_table_with_contracted_columns():
    async def _columns():
        conn = await asyncpg.connect(db_url())
        try:
            await ensure_schema(conn)
            rows = await conn.fetch(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = $1 AND table_name = 'messages'
                """,
                PG_SCHEMA,
            )
            return {row["column_name"]: row["data_type"] for row in rows}
        finally:
            await conn.close()

    columns = asyncio.run(_columns())
    for name in (
        "id",
        "namespace",
        "purpose",
        "scope",
        "subject",
        "subject_key",
        "status",
        "content",
        "author",
        "sender_key",
        "idempotency_key",
        "created_at",
        "expires_at",
    ):
        assert name in columns, name
    assert columns["id"] == "uuid"
    assert columns["created_at"].startswith("timestamp")
    assert columns["expires_at"].startswith("timestamp")

    async def _grants():
        conn = await asyncpg.connect(db_url())
        try:
            return await conn.fetchval(
                """
                SELECT count(*)
                FROM information_schema.role_table_grants
                WHERE table_schema = $1 AND table_name = 'messages'
                  AND grantee = 'memory_tables_query'
                """,
                PG_SCHEMA,
            )
        finally:
            await conn.close()

    # The restricted SQL role must not see the new table (ADR-0001 grant list).
    assert asyncio.run(_grants()) == 0


# ---- write path: no embedding, no content gate, no search visibility ----------


def test_send_stores_rendered_content_without_touching_the_note_path(monkeypatch):
    marker = f"zzmsg_{uuid.uuid4().hex[:8]}"

    async def _no_gate(content, kind):
        raise AssertionError("the note content gate must not run for messages")

    def _no_embed(*args, **kwargs):
        raise AssertionError("messages must never reach the embedding path")

    monkeypatch.setattr(notes, "judge_note_content", _no_gate)
    monkeypatch.setattr(notes, "embed_text", _no_embed)
    try:
        response = _send(
            f"{marker} re-seed staging",
            result="Re-run anything that cached row counts.",
        )
        assert response.status_code == 201
        row = response.json()
        assert row["purpose"] == "message"
        assert row["status"] == "info"
        assert row["delivery"] == "pending"
        assert row["scope"] is None
        assert set(row) == {
            "id",
            "namespace",
            "purpose",
            "scope",
            "subject",
            "status",
            "delivery",
            "author",
            "created_at",
            "expires_at",
            "content",
        }
        assert row["content"].startswith(f"# {marker} re-seed staging")
        assert "## Status" in row["content"]
    finally:
        asyncio.run(_cleanup(marker))


def test_message_content_never_lands_in_any_searched_table():
    marker = f"zzmsg_{uuid.uuid4().hex[:8]}"
    needle = f"{marker} signum certum quod quaeritur"
    counts = {}

    async def _run():
        conn = await asyncpg.connect(db_url())
        try:
            counts["memory_chunks"] = await conn.fetchval(
                f'SELECT count(*) FROM "{PG_SCHEMA}".memory_chunks '
                "WHERE content_raw LIKE $1 OR distilled LIKE $1",
                f"%{needle}%",
            )
            counts["doc_rows"] = await conn.fetchval(
                f'SELECT count(*) FROM "{PG_SCHEMA}".doc_rows WHERE data::text LIKE $1',
                f"%{needle}%",
            )
            counts["messages"] = await conn.fetchval(
                f'SELECT count(*) FROM "{PG_SCHEMA}".messages WHERE subject = $1', needle
            )
        finally:
            await conn.close()

    try:
        assert _send(needle, result="only addressed").status_code == 201
        asyncio.run(_run())
        assert counts["memory_chunks"] == 0
        assert counts["doc_rows"] == 0
        assert counts["messages"] == 1
    finally:
        asyncio.run(_cleanup(marker))


# ---- claim atomicity under real concurrency ------------------------------------


def test_concurrent_double_claim_exactly_one_wins():
    marker = f"zzmsg_{uuid.uuid4().hex[:8]}"
    try:
        response = _send(f"{marker} exclusive claim", result="r")
        message_id = uuid.UUID(response.json()["id"])

        async def _run():
            async with connections(2) as (first, second):
                first_claim = messages.claim_message(message_id, IDENTITY, connection=first)
                second_claim = messages.claim_message(message_id, IDENTITY, connection=second)
                return await asyncio.gather(first_claim, second_claim, return_exceptions=True)

        results = asyncio.run(_run())
        claimed = [r for r in results if not isinstance(r, Exception)]
        refused = [r for r in results if isinstance(r, Exception)]
        assert len(claimed) == 1
        assert len(refused) == 1
        assert isinstance(refused[0], messages.MessageConflict)

        async def _claimed_at():
            conn = await asyncpg.connect(db_url())
            try:
                return await conn.fetchval(
                    f'SELECT claimed_at FROM "{PG_SCHEMA}".messages WHERE id = $1', message_id
                )
            finally:
                await conn.close()

        assert asyncio.run(_claimed_at()) is not None
    finally:
        asyncio.run(_cleanup(marker))


def test_claim_and_cancel_race_admit_exactly_one_winner():
    marker = f"zzmsg_{uuid.uuid4().hex[:8]}"
    try:
        response = _send(f"{marker} race", result="r")
        message_id = uuid.UUID(response.json()["id"])

        async def _run():
            async with connections(2) as (first, second):
                claim = messages.claim_message(message_id, IDENTITY, connection=first)
                cancel = messages.cancel_message(message_id, IDENTITY, connection=second)
                return await asyncio.gather(claim, cancel, return_exceptions=True)

        claim, cancel = asyncio.run(_run())
        # Exception instances are non-None, so winners are picked by type: the
        # successful result is the dict, the loser is a MessageConflict.
        claim_won = not isinstance(claim, Exception)
        cancel_won = not isinstance(cancel, Exception)
        assert claim_won != cancel_won
        refused = cancel if claim_won else claim
        assert isinstance(refused, messages.MessageConflict)
        winner = claim if claim_won else cancel
        assert isinstance(winner, dict)
        assert winner["delivery"] == ("claimed" if claim_won else "cancelled")
        assert winner["status"] == "info"
    finally:
        asyncio.run(_cleanup(marker))


# ---- handoff snapshots -----------------------------------------------------------


def test_only_latest_pending_snapshot_is_claimable_and_stale_id_gets_409():
    marker = f"zzmsg_{uuid.uuid4().hex[:8]}"
    subject = f"{marker} shared state"
    try:
        first = _send_handoff(subject)
        assert first.status_code == 201
        first_id = first.json()["id"]
        second = _send_handoff(subject)
        second_id = second.json()["id"]

        listed = client.get(
            "/messages",
            params={"purpose": "handoff", "scope": "repo:github.com/o/r", "subject": subject},
        ).json()
        assert [row["id"] for row in listed] == [second_id]

        stale = client.post(f"/messages/{first_id}/claim")
        assert stale.status_code == 409
        fresh = client.post(f"/messages/{second_id}/claim")
        assert fresh.status_code == 200
        assert fresh.json()["delivery"] == "claimed"
        assert fresh.json()["status"] == "in_progress"
    finally:
        asyncio.run(_cleanup(marker))


def test_completed_handoff_is_delivered_once_and_can_be_reopened():
    marker = f"zzmsg_{uuid.uuid4().hex[:8]}"
    subject = f"{marker} finished"
    try:
        done = _send_handoff(subject, status="completed", next=None)
        assert done.status_code == 201
        done_id = done.json()["id"]

        delivered = client.post(f"/messages/{done_id}/claim")
        assert delivered.status_code == 200
        assert client.post(f"/messages/{done_id}/claim").status_code == 409

        reopened = _send_handoff(subject, status="in_progress")
        assert reopened.status_code == 201
        reopen_claim = client.post(f"/messages/{reopened.json()['id']}/claim")
        assert reopen_claim.status_code == 200
    finally:
        asyncio.run(_cleanup(marker))


def test_supersede_touches_only_pending_and_different_subjects_coexist():
    marker = f"zzmsg_{uuid.uuid4().hex[:8]}"
    subject = f"{marker} same state"
    try:
        claimed_old = _send_handoff(subject)
        old_id = claimed_old.json()["id"]
        assert client.post(f"/messages/{old_id}/claim").status_code == 200

        # A new snapshot must not disturb the already-terminal one.
        new = _send_handoff(subject)
        assert new.status_code == 201

        async def _timestamps_of(message_id):
            conn = await asyncpg.connect(db_url())
            try:
                return await conn.fetchrow(
                    f"""
                    SELECT claimed_at, cancelled_at, superseded_at
                    FROM "{PG_SCHEMA}".messages WHERE id = $1
                    """,
                    uuid.UUID(message_id),
                )
            finally:
                await conn.close()

        old_timestamps = asyncio.run(_timestamps_of(old_id))
        assert old_timestamps["claimed_at"] is not None
        assert old_timestamps["superseded_at"] is None

        # A different subject under the same scope coexists with the pending one.
        other = _send_handoff(f"{marker} other subject")
        assert other.status_code == 201
        other_listed = client.get(
            "/messages",
            params={"purpose": "handoff", "subject": f"{marker} other subject"},
        ).json()
        assert [row["id"] for row in other_listed] == [other.json()["id"]]
    finally:
        asyncio.run(_cleanup(marker))


def test_subject_normalization_groups_snapshots_by_subject_key():
    marker = f"zzmsg_{uuid.uuid4().hex[:8]}"
    try:
        first = _send_handoff(f"{marker} Fix  LOGIN Flow")
        second = _send_handoff(f"{marker} fix login  flow\u00a0")
        assert first.status_code == 201 and second.status_code == 201
        # Same subject_key, so the first snapshot is stale.
        stale = client.post(f"/messages/{first.json()['id']}/claim")
        assert stale.status_code == 409
        # The filter normalizes its argument the same way: any spelling variant
        # of the subject finds the one pending snapshot.
        by_variant = client.get(
            "/messages", params={"subject": f"  {marker.upper()} FIX LOGIN  FLOW "}
        ).json()
        assert [row["id"] for row in by_variant] == [second.json()["id"]]
    finally:
        asyncio.run(_cleanup(marker))


def test_any_namespace_authorized_sender_can_publish_next_snapshot(monkeypatch):
    namespace = f"msg-it-{uuid.uuid4().hex[:8]}"
    subject = f"zzmsg_{uuid.uuid4().hex[:8]} handover"
    try:
        assert client.post("/namespaces", json={"name": namespace}).status_code == 201
        admin_snapshot = _send_handoff(subject, namespace=namespace)
        assert admin_snapshot.status_code == 201

        member = _member_client(monkeypatch, "msgmember", {namespace, "default"})
        member_snapshot = member.post(
            "/messages",
            json={
                "subject": subject,
                "status": "blocked",
                "result": "waiting on CI",
                "next": "retry after green",
                "scope": "repo:github.com/o/r",
                "author": "claude-code",
                "namespace": namespace,
            },
        )
        assert member_snapshot.status_code == 201

        listed = client.get(
            "/messages",
            params={"namespace": namespace, "purpose": "handoff", "subject": subject},
        ).json()
        assert [row["id"] for row in listed] == [member_snapshot.json()["id"]]
    finally:
        asyncio.run(_cleanup(subject, namespace=namespace))


def test_general_message_never_supersedes_a_pending_handoff():
    marker = f"zzmsg_{uuid.uuid4().hex[:8]}"
    subject = f"{marker} state"
    try:
        handoff = _send_handoff(subject)
        handoff_id = handoff.json()["id"]
        general = _send(subject, result="unrelated signal")
        assert general.status_code == 201
        assert general.json()["scope"] is None

        listed = client.get(
            "/messages",
            params={"purpose": "handoff", "subject": subject},
        ).json()
        assert [row["id"] for row in listed] == [handoff_id]
    finally:
        asyncio.run(_cleanup(marker))


# ---- expiry ------------------------------------------------------------------------


def test_expired_messages_leave_list_and_claim_and_purge_deletes_them():
    marker = f"zzmsg_{uuid.uuid4().hex[:8]}"
    subject = f"{marker} dying signal"
    try:
        response = _send(subject, result="r")
        message_id = response.json()["id"]

        async def _expire():
            conn = await asyncpg.connect(db_url())
            try:
                await conn.execute(
                    f'UPDATE "{PG_SCHEMA}".messages '
                    "SET expires_at = now() - interval '1 second' WHERE id = $1",
                    uuid.UUID(message_id),
                )
            finally:
                await conn.close()

        asyncio.run(_expire())
        assert client.get("/messages", params={"subject": subject}).json() == []
        expired_claim = client.post(f"/messages/{message_id}/claim")
        assert expired_claim.status_code == 409

        preview = client.post("/admin/archive", json={}).json()
        assert any(row["id"] == message_id for row in preview["messages_to_delete"]), (
            "expired message must appear in the purge preview"
        )
        confirm = client.post("/admin/archive", json={"confirm": True}).json()
        assert confirm["deleted"] >= 1

        async def _gone():
            conn = await asyncpg.connect(db_url())
            try:
                return await conn.fetchval(
                    f'SELECT count(*) FROM "{PG_SCHEMA}".messages WHERE id = $1',
                    uuid.UUID(message_id),
                )
            finally:
                await conn.close()

        assert asyncio.run(_gone()) == 0
    finally:
        asyncio.run(_cleanup(marker))


def test_expires_at_must_be_future_and_within_thirty_days():
    marker = f"zzmsg_{uuid.uuid4().hex[:8]}"
    try:
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        too_far = (datetime.now(timezone.utc) + timedelta(days=31)).isoformat()
        ok = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
        assert _send(f"{marker} a", expires_at=past).status_code == 400
        assert _send(f"{marker} b", expires_at=too_far).status_code == 400
        accepted = _send(f"{marker} c", expires_at=ok)
        assert accepted.status_code == 201
        assert accepted.json()["expires_at"] == ok
    finally:
        asyncio.run(_cleanup(marker))


# ---- idempotency ----------------------------------------------------------------------


def test_idempotency_replays_conflicts_and_releases_on_delete():
    marker = f"zzmsg_{uuid.uuid4().hex[:8]}"
    body = {
        "subject": f"{marker} idempotent",
        "status": "info",
        "result": "same intent",
        "author": "claude-code",
        "idempotency_key": f"run-{marker}",
    }
    try:
        first = client.post("/messages", json=body)
        assert first.status_code == 201
        replay = client.post("/messages", json=body)
        assert replay.status_code == 200
        assert replay.json()["id"] == first.json()["id"]

        different = client.post("/messages", json=body | {"result": "different intent"})
        assert different.status_code == 409

        message_id = first.json()["id"]
        assert client.post(f"/messages/{message_id}/claim").status_code == 200
        confirm = client.post("/admin/archive", json={"confirm": True}).json()
        assert confirm["deleted"] >= 1

        released = client.post(
            "/messages", json=body | {"result": "brand new intent, key released"}
        )
        assert released.status_code == 201
        assert released.json()["id"] != message_id
    finally:
        asyncio.run(_cleanup(marker))


# ---- namespace deletion ------------------------------------------------------------------


def test_namespace_deletion_waits_for_messages(monkeypatch):
    namespace = f"msg-it-{uuid.uuid4().hex[:8]}"
    subject = f"zzmsg_{uuid.uuid4().hex[:8]} blocking"
    try:
        assert client.post("/namespaces", json={"name": namespace}).status_code == 201
        member = _member_client(monkeypatch, "msgmember", {namespace, "default"})
        sent = member.post(
            "/messages",
            json={
                "subject": subject,
                "status": "info",
                "result": "r",
                "author": "claude-code",
                "namespace": namespace,
            },
        )
        assert sent.status_code == 201
        message_id = sent.json()["id"]

        blocked = client.delete(f"/namespaces/{namespace}")
        assert blocked.status_code == 409

        assert client.delete(f"/messages/{message_id}").status_code == 200
        client.post("/admin/archive", json={"confirm": True})
        assert client.delete(f"/namespaces/{namespace}").status_code == 200
    finally:
        asyncio.run(_cleanup(subject, namespace=namespace))


# ---- admin purge distinguishes the two halves ----------------------------------------------


def test_admin_archive_purge_deletes_only_terminal_messages():
    marker = f"zzmsg_{uuid.uuid4().hex[:8]}"
    try:
        pending = _send(f"{marker} keep", result="r")
        cancelled = _send(f"{marker} drop", result="r")
        cancelled_id = cancelled.json()["id"]
        assert client.delete(f"/messages/{cancelled_id}").status_code == 200

        preview = client.post("/admin/archive", json={}).json()
        preview_ids = {row["id"] for row in preview["messages_to_delete"]}
        assert cancelled_id in preview_ids
        assert pending.json()["id"] not in preview_ids

        client.post("/admin/archive", json={"confirm": True})
        kept = client.get("/messages", params={"subject": f"{marker} keep"}).json()
        assert [row["id"] for row in kept] == [pending.json()["id"]]
        dropped = client.get("/messages", params={"subject": f"{marker} drop"}).json()
        assert dropped == []
    finally:
        asyncio.run(_cleanup(marker))


# ---- rendering: injection containment, multiline commands, ref rejection ---------------------


def test_markdown_injection_cannot_escape_the_canonical_render():
    marker = f"zzmsg_{uuid.uuid4().hex[:8]}"
    try:
        response = _send(
            f"{marker} real\n## Injected heading",
            result="line one\n## Fake next\n> fake quote",
        )
        assert response.status_code == 201
        content = response.json()["content"]
        lines = content.splitlines()
        assert lines[0] == f"# {marker} real ## Injected heading"
        assert "> ## Fake next" in lines
        assert "> > fake quote" in lines
        assert [line for line in lines if line.startswith("#")] == [
            f"# {marker} real ## Injected heading",
            "## Status",
            "## Result",
        ]
    finally:
        asyncio.run(_cleanup(marker))


def test_multiline_verification_command_is_rendered_line_by_line():
    marker = f"zzmsg_{uuid.uuid4().hex[:8]}"
    try:
        response = _send(
            f"{marker} verified",
            verification={
                "command": "uv run pytest tests/auth\n-v",
                "status": "passed",
                "result": "12 green",
            },
        )
        assert response.status_code == 201
        lines = response.json()["content"].splitlines()
        assert "> uv run pytest tests/auth" in lines
        assert "> -v" in lines
        assert "> passed" in lines
        assert "> 12 green" in lines
    finally:
        asyncio.run(_cleanup(marker))


@pytest.mark.parametrize(
    "ref",
    [
        "https://user:token@example.com/x",
        "https://localhost/x",
        "https://127.0.0.1/x",
        "https://10.0.0.9/x",
        "https://169.254.3.3/x",
        "https://[::1]/x",
        "file:///etc/passwd",
        "http://example.com/x",
        "/etc/passwd",
    ],
)
def test_refs_rejected_at_the_rest_boundary(ref):
    marker = f"zzmsg_{uuid.uuid4().hex[:8]}"
    try:
        response = _send(f"{marker} refs", refs=[ref])
        assert response.status_code == 400
        assert client.get("/messages", params={"subject": f"{marker} refs"}).json() == []
    finally:
        asyncio.run(_cleanup(marker))


def test_more_than_ten_refs_rejected():
    marker = f"zzmsg_{uuid.uuid4().hex[:8]}"
    try:
        response = _send(
            f"{marker} many refs",
            refs=[f"https://example.com/{i}" for i in range(11)],
        )
        assert response.status_code == 400
    finally:
        asyncio.run(_cleanup(marker))


# ---- listing: pending-only, ordering, filters, limits ------------------------------------------


def test_listing_is_pending_only_newest_first_with_filters_and_limit():
    marker = f"zzmsg_{uuid.uuid4().hex[:8]}"
    subject = f"{marker} lane"
    try:
        oldest = _send(subject, result="r")
        claimed = _send(subject, result="r")
        newest = _send(subject, result="r")
        handoff = _send_handoff(subject)
        assert client.post(f"/messages/{claimed.json()['id']}/claim").status_code == 200

        listed = client.get("/messages", params={"subject": subject}).json()
        assert [row["id"] for row in listed] == [
            handoff.json()["id"],
            newest.json()["id"],
            oldest.json()["id"],
        ]

        only_messages = client.get(
            "/messages", params={"subject": subject, "purpose": "message"}
        ).json()
        assert [row["id"] for row in only_messages] == [
            newest.json()["id"],
            oldest.json()["id"],
        ]

        limited = client.get("/messages", params={"subject": subject, "limit": 2}).json()
        assert [row["id"] for row in limited] == [
            handoff.json()["id"],
            newest.json()["id"],
        ]
    finally:
        asyncio.run(_cleanup(marker))


def test_response_never_exposes_key_identity():
    marker = f"zzmsg_{uuid.uuid4().hex[:8]}"
    try:
        response = _send(f"{marker} privacy", idempotency_key=f"key-{marker}")
        row = response.json()
        assert "sender_key" not in row
        assert "idempotency_key" not in row
        listed = client.get("/messages", params={"subject": f"{marker} privacy"}).json()[0]
        assert "sender_key" not in listed
        claimed = client.post(f"/messages/{row['id']}/claim").json()
        assert "sender_key" not in claimed
    finally:
        asyncio.run(_cleanup(marker))
