"""Unit tests for the /messages REST endpoints (memory_base.serve.api).

No DB: memory_base.serve.messages functions are monkeypatched directly,
matching the convention of tests/serve/test_namespaces_api.py. The fixed
``test-key`` header (tests/serve/conftest.py) stubs to an admin identity with
label "test" and authors {claude-code, natsume}.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from memory_base.serve import api, auth, messages

client = TestClient(api.app, headers={"X-API-Key": "test-key"})

ROW = {
    "id": "5f0d9d44-9a9d-4f0e-b7f6-6fa1e2b3c4d5",
    "namespace": "default",
    "purpose": "message",
    "scope": None,
    "subject": "Re-seed the staging DB",
    "status": "pending",
    "author": "claude-code",
    "created_at": "2026-02-03T10:00:00+00:00",
    "expires_at": "2026-02-10T10:00:00+00:00",
    "content": "# Re-seed the staging DB\n\n## Status\n\n> info",
}


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


# ---- POST /messages -------------------------------------------------------------


def test_post_message_delegates_and_returns_201_with_public_row(monkeypatch):
    captured = {}

    async def fake_send(key, **kwargs):
        captured.update(kwargs)
        captured["key"] = key
        return dict(ROW), False

    monkeypatch.setattr(messages, "send_message", fake_send)
    response = client.post(
        "/messages",
        json={
            "subject": "Re-seed the staging DB",
            "status": "info",
            "result": "Re-run anything that cached row counts.",
            "author": "claude-code",
        },
    )
    assert response.status_code == 201
    assert response.json() == ROW
    assert captured["namespace"] == "default"
    assert captured["author"] == "claude-code"
    assert captured["subject"] == "Re-seed the staging DB"
    assert captured["status"] == "info"
    assert "sender_key" not in response.json()
    assert "idempotency_key" not in response.json()


def test_post_message_unknown_fields_rejected(monkeypatch):
    response = client.post(
        "/messages",
        json={
            "subject": "s",
            "status": "info",
            "result": "r",
            "author": "claude-code",
            "priority": "high",
        },
    )
    assert response.status_code == 400
    assert "priority" in response.json()["error"]


def test_post_message_namespace_outside_caller_403(monkeypatch):
    member = _member_client(monkeypatch, "eve", {"default"})
    response = member.post(
        "/messages",
        json={
            "subject": "s",
            "status": "info",
            "result": "r",
            "author": "claude-code",
            "namespace": "team-b",
        },
    )
    assert response.status_code == 403


def test_post_message_author_outside_allowlist_403(monkeypatch):
    response = client.post(
        "/messages",
        json={"subject": "s", "status": "info", "result": "r", "author": "ghost"},
    )
    assert response.status_code == 403


def test_post_message_validation_error_400(monkeypatch):
    async def fake_send(key, **kwargs):
        raise ValueError("handoff status must be one of")

    monkeypatch.setattr(messages, "send_message", fake_send)
    response = client.post(
        "/messages",
        json={"subject": "s", "status": "info", "result": "r", "author": "claude-code"},
    )
    assert response.status_code == 400


def test_post_message_handoff_fields_reach_messages(monkeypatch):
    captured = {}

    async def fake_send(key, **kwargs):
        captured.update(kwargs)
        return dict(ROW, purpose="handoff", scope="repo:github.com/o/r"), False

    monkeypatch.setattr(messages, "send_message", fake_send)
    response = client.post(
        "/messages",
        json={
            "subject": "s",
            "status": "in_progress",
            "result": "r",
            "next": "n",
            "scope": "repo:github.com/o/r",
            "author": "claude-code",
            "idempotency_key": "run-1",
            "expires_at": "2026-03-01T00:00:00+00:00",
        },
    )
    assert response.status_code == 201
    assert captured["scope"] == "repo:github.com/o/r"
    assert captured["next_text"] == "n"
    assert captured["idempotency_key"] == "run-1"
    assert captured["expires_at"] == "2026-03-01T00:00:00+00:00"


def test_post_message_idempotency_conflict_409(monkeypatch):
    async def fake_send(key, **kwargs):
        raise messages.MessageConflict("idempotency_key already used for a different message")

    monkeypatch.setattr(messages, "send_message", fake_send)
    response = client.post(
        "/messages",
        json={
            "subject": "s",
            "status": "info",
            "result": "r",
            "author": "claude-code",
            "idempotency_key": "run-1",
        },
    )
    assert response.status_code == 409


# ---- GET /messages ----------------------------------------------------------------


def test_get_messages_delegates_with_scope_and_filters(monkeypatch):
    captured = {}

    async def fake_list(**kwargs):
        captured.update(kwargs)
        return [ROW]

    monkeypatch.setattr(messages, "list_messages", fake_list)
    response = client.get(
        "/messages",
        params={
            "namespace": "team-a",
            "purpose": "handoff",
            "scope": "repo:github.com/o/r",
            "subject": "Fix  LOGIN",
            "limit": 10,
        },
    )
    assert response.status_code == 200
    assert response.json() == [ROW]
    assert captured["namespaces"] == ["team-a"]
    assert captured["purpose"] == "handoff"
    assert captured["scope"] == "repo:github.com/o/r"
    assert captured["subject"] == "Fix  LOGIN"
    assert captured["limit"] == 10


def test_get_messages_default_limit_50_and_max_100(monkeypatch):
    captured = {}

    async def fake_list(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(messages, "list_messages", fake_list)
    assert client.get("/messages").json() == []
    assert captured["limit"] == 50
    assert client.get("/messages", params={"limit": "100"}).status_code == 200
    assert captured["limit"] == 100
    assert client.get("/messages", params={"limit": "101"}).status_code == 400
    assert client.get("/messages", params={"limit": "zero"}).status_code == 400


def test_get_messages_unknown_purpose_400(monkeypatch):
    response = client.get("/messages", params={"purpose": "memo"})
    assert response.status_code == 400


def test_get_messages_bad_subject_400(monkeypatch):
    response = client.get("/messages", params={"subject": "   "})
    assert response.status_code == 400


def test_get_messages_bad_scope_400(monkeypatch):
    response = client.get("/messages", params={"scope": "/home/user/checkout"})
    assert response.status_code == 400


def test_get_messages_member_scoped_to_allowed_namespaces(monkeypatch):
    captured = {}

    async def fake_list(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(messages, "list_messages", fake_list)
    member = _member_client(monkeypatch, "eve", {"default", "team-a"})
    response = member.get("/messages")
    assert response.status_code == 200
    assert captured["namespaces"] == ["default", "team-a"]


def test_get_messages_member_requesting_outside_set_403(monkeypatch):
    member = _member_client(monkeypatch, "eve", {"default"})
    response = member.get("/messages", params={"namespace": "team-b"})
    assert response.status_code == 403


# ---- POST /messages/{id}/claim ------------------------------------------------------


def test_claim_returns_public_row(monkeypatch):
    captured = {}

    async def fake_claim(message_id, key, connection=None):
        captured["id"] = message_id
        return dict(ROW, status="claimed")

    monkeypatch.setattr(messages, "claim_message", fake_claim)
    response = client.post(f"/messages/{ROW['id']}/claim")
    assert response.status_code == 200
    assert response.json()["status"] == "claimed"
    assert str(captured["id"]) == ROW["id"]


def test_claim_malformed_id_400():
    response = client.post("/messages/not-a-uuid/claim")
    assert response.status_code == 400


def test_claim_unknown_404(monkeypatch):
    async def fake_claim(message_id, key, connection=None):
        raise messages.MessageNotFound("no claimable message")

    monkeypatch.setattr(messages, "claim_message", fake_claim)
    assert client.post(f"/messages/{ROW['id']}/claim").status_code == 404


def test_claim_stale_or_terminal_409(monkeypatch):
    async def fake_claim(message_id, key, connection=None):
        raise messages.MessageConflict("message is not pending")

    monkeypatch.setattr(messages, "claim_message", fake_claim)
    response = client.post(f"/messages/{ROW['id']}/claim")
    assert response.status_code == 409


# ---- DELETE /messages/{id} ------------------------------------------------------------


def test_cancel_returns_public_row(monkeypatch):
    captured = {}

    async def fake_cancel(message_id, key, connection=None):
        captured["id"] = message_id
        return dict(ROW, status="cancelled")

    monkeypatch.setattr(messages, "cancel_message", fake_cancel)
    response = client.delete(f"/messages/{ROW['id']}")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert str(captured["id"]) == ROW["id"]


def test_cancel_malformed_id_400():
    assert client.delete("/messages/nope").status_code == 400


def test_cancel_unauthorized_or_unknown_404(monkeypatch):
    async def fake_cancel(message_id, key, connection=None):
        raise messages.MessageNotFound("no cancellable message")

    monkeypatch.setattr(messages, "cancel_message", fake_cancel)
    assert client.delete(f"/messages/{ROW['id']}").status_code == 404


def test_cancel_non_pending_409(monkeypatch):
    async def fake_cancel(message_id, key, connection=None):
        raise messages.MessageConflict("message is not pending")

    monkeypatch.setattr(messages, "cancel_message", fake_cancel)
    assert client.delete(f"/messages/{ROW['id']}").status_code == 409


def test_message_routes_require_auth():
    plain = TestClient(api.app)
    assert plain.get("/messages").status_code == 401
    assert plain.post("/messages", json={}).status_code == 401
    assert plain.post(f"/messages/{ROW['id']}/claim").status_code == 401
    assert plain.delete(f"/messages/{ROW['id']}").status_code == 401
