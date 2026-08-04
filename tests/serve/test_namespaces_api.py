"""Unit tests for the /namespaces REST endpoints (memory_base.serve.api).

No DB: memory_base.serve.namespaces functions are monkeypatched directly,
matching the convention used for admin.* in tests/serve/test_admin_api.py.
The fixed ``test-key`` header (tests/serve/conftest.py) stubs to an admin
identity with label "test"; ownership/scoping tests override the stub for a
second key to exercise a non-admin caller.
"""

from __future__ import annotations

from starlette.testclient import TestClient

from memory_base.serve import api, auth, namespaces

client = TestClient(api.app, headers={"X-API-Key": "test-key"})


def _non_admin_client(monkeypatch, label, allowed):
    identity = auth.KeyIdentity(
        key_id=f"{label}-hash",
        label=label,
        home="default",
        is_admin=False,
        allowed=frozenset(allowed),
    )

    async def fake_authenticate_request(plaintext_key):
        return identity if plaintext_key == "member-key" else None

    monkeypatch.setattr(auth, "authenticate_request", fake_authenticate_request)
    return TestClient(api.app, headers={"X-API-Key": "member-key"})


# ---- POST /namespaces --------------------------------------------------------


def test_create_namespace_delegates_and_returns_201(monkeypatch):
    captured = {}

    async def fake_create(name, visibility, owner):
        captured["name"] = name
        captured["visibility"] = visibility
        captured["owner"] = owner
        return {"name": name, "created_at": 123.0, "visibility": visibility, "owner": owner}

    monkeypatch.setattr(namespaces, "create_namespace", fake_create)
    response = client.post("/namespaces", json={"name": "team-a"})
    assert response.status_code == 201
    assert response.json() == {
        "name": "team-a",
        "created_at": 123.0,
        "visibility": "public",
        "owner": None,
    }
    assert captured == {"name": "team-a", "visibility": "public", "owner": None}


def test_create_namespace_private_records_caller_as_owner(monkeypatch):
    captured = {}

    async def fake_create(name, visibility, owner):
        captured["visibility"] = visibility
        captured["owner"] = owner
        return {"name": name, "created_at": 123.0, "visibility": visibility, "owner": owner}

    monkeypatch.setattr(namespaces, "create_namespace", fake_create)
    response = client.post("/namespaces", json={"name": "team-a", "visibility": "private"})
    assert response.status_code == 201
    assert captured == {"visibility": "private", "owner": "test"}
    assert response.json()["owner"] == "test"


def test_create_namespace_bad_slug_400(monkeypatch):
    async def fake_create(name, visibility, owner):
        raise namespaces.NamespaceError("namespace must match ^[a-z0-9_-]{1,64}$")

    monkeypatch.setattr(namespaces, "create_namespace", fake_create)
    response = client.post("/namespaces", json={"name": "Bad Name"})
    assert response.status_code == 400
    assert "error" in response.json()


def test_create_namespace_duplicate_409(monkeypatch):
    async def fake_create(name, visibility, owner):
        raise namespaces.NamespaceExistsError(f"namespace already exists: {name}")

    monkeypatch.setattr(namespaces, "create_namespace", fake_create)
    response = client.post("/namespaces", json={"name": "team-a"})
    assert response.status_code == 409
    assert "error" in response.json()


def test_create_namespace_malformed_json_400():
    response = client.post(
        "/namespaces", content=b"{not valid json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 400
    assert "error" in response.json()


# ---- GET /namespaces ----------------------------------------------------------


def test_list_namespaces_returns_registry_verbatim_for_admin(monkeypatch):
    rows = [
        {"name": "default", "created_at": 0.0, "visibility": "public", "owner": None},
        {"name": "team-a", "created_at": 5.0, "visibility": "private", "owner": "bob"},
    ]

    async def fake_list():
        return rows

    monkeypatch.setattr(namespaces, "list_namespaces", fake_list)
    response = client.get("/namespaces")
    assert response.status_code == 200
    assert response.json() == rows


def test_list_namespaces_filters_to_non_admin_allowed_set(monkeypatch):
    rows = [
        {"name": "default", "created_at": 0.0, "visibility": "public", "owner": None},
        {"name": "team-a", "created_at": 5.0, "visibility": "private", "owner": "alice"},
        {"name": "team-b", "created_at": 6.0, "visibility": "private", "owner": "bob"},
    ]

    async def fake_list():
        return rows

    monkeypatch.setattr(namespaces, "list_namespaces", fake_list)
    member_client = _non_admin_client(monkeypatch, "alice", {"default", "team-a"})
    response = member_client.get("/namespaces")
    assert response.status_code == 200
    assert {row["name"] for row in response.json()} == {"default", "team-a"}


# ---- DELETE /namespaces/{name} -------------------------------------------------


def test_delete_namespace_success(monkeypatch):
    calls = {}

    async def fake_get_namespace(name):
        return {"name": name, "created_at": 5.0, "visibility": "public", "owner": None}

    async def fake_delete(name):
        calls["name"] = name

    monkeypatch.setattr(namespaces, "get_namespace", fake_get_namespace)
    monkeypatch.setattr(namespaces, "delete_namespace", fake_delete)
    response = client.delete("/namespaces/team-a")
    assert response.status_code == 200
    assert response.json() == {"deleted": "team-a"}
    assert calls["name"] == "team-a"


def test_delete_reserved_default_400():
    response = client.delete("/namespaces/default")
    assert response.status_code == 400
    assert "error" in response.json()


def test_delete_unknown_namespace_404(monkeypatch):
    async def fake_get_namespace(name):
        return None

    monkeypatch.setattr(namespaces, "get_namespace", fake_get_namespace)
    response = client.delete("/namespaces/ghost")
    assert response.status_code == 404
    assert "error" in response.json()


def test_delete_non_empty_namespace_409(monkeypatch):
    async def fake_get_namespace(name):
        return {"name": name, "created_at": 5.0, "visibility": "public", "owner": None}

    async def fake_delete(name):
        raise namespaces.NamespaceNotEmptyError(f"namespace still has rows: {name}")

    monkeypatch.setattr(namespaces, "get_namespace", fake_get_namespace)
    monkeypatch.setattr(namespaces, "delete_namespace", fake_delete)
    response = client.delete("/namespaces/team-a")
    assert response.status_code == 409
    assert "error" in response.json()


def test_delete_by_owner_succeeds(monkeypatch):
    async def fake_get_namespace(name):
        return {"name": name, "created_at": 5.0, "visibility": "private", "owner": "alice"}

    async def fake_delete(name):
        return None

    monkeypatch.setattr(namespaces, "get_namespace", fake_get_namespace)
    monkeypatch.setattr(namespaces, "delete_namespace", fake_delete)
    member_client = _non_admin_client(monkeypatch, "alice", {"default", "team-a"})
    response = member_client.delete("/namespaces/team-a")
    assert response.status_code == 200


def test_delete_by_non_owner_non_admin_403(monkeypatch):
    async def fake_get_namespace(name):
        return {"name": name, "created_at": 5.0, "visibility": "private", "owner": "alice"}

    monkeypatch.setattr(namespaces, "get_namespace", fake_get_namespace)
    member_client = _non_admin_client(monkeypatch, "eve", {"default"})
    response = member_client.delete("/namespaces/team-a")
    assert response.status_code == 403
    assert "error" in response.json()
