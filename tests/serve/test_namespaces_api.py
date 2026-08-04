"""Unit tests for the /namespaces REST endpoints (memory_base.serve.api).

No DB: memory_base.serve.namespaces functions are monkeypatched directly,
matching the convention used for admin.* in tests/serve/test_admin_api.py.
"""

from __future__ import annotations

from starlette.testclient import TestClient

from memory_base.serve import api, namespaces

client = TestClient(api.app)


# ---- POST /namespaces --------------------------------------------------------


def test_create_namespace_delegates_and_returns_201(monkeypatch):
    captured = {}

    async def fake_create(name):
        captured["name"] = name
        return {"name": name, "created_at": 123.0}

    monkeypatch.setattr(namespaces, "create_namespace", fake_create)
    response = client.post("/namespaces", json={"name": "team-a"})
    assert response.status_code == 201
    assert response.json() == {"name": "team-a", "created_at": 123.0}
    assert captured["name"] == "team-a"


def test_create_namespace_bad_slug_400(monkeypatch):
    async def fake_create(name):
        raise namespaces.NamespaceError("namespace must match ^[a-z0-9_-]{1,64}$")

    monkeypatch.setattr(namespaces, "create_namespace", fake_create)
    response = client.post("/namespaces", json={"name": "Bad Name"})
    assert response.status_code == 400
    assert "error" in response.json()


def test_create_namespace_duplicate_409(monkeypatch):
    async def fake_create(name):
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


def test_list_namespaces_returns_registry_verbatim(monkeypatch):
    rows = [{"name": "default", "created_at": 0.0}, {"name": "team-a", "created_at": 5.0}]

    async def fake_list():
        return rows

    monkeypatch.setattr(namespaces, "list_namespaces", fake_list)
    response = client.get("/namespaces")
    assert response.status_code == 200
    assert response.json() == rows


# ---- DELETE /namespaces/{name} -------------------------------------------------


def test_delete_namespace_success(monkeypatch):
    calls = {}

    async def fake_delete(name):
        calls["name"] = name

    monkeypatch.setattr(namespaces, "delete_namespace", fake_delete)
    response = client.delete("/namespaces/team-a")
    assert response.status_code == 200
    assert response.json() == {"deleted": "team-a"}
    assert calls["name"] == "team-a"


def test_delete_reserved_default_400(monkeypatch):
    async def fake_delete(name):
        raise namespaces.NamespaceReservedError("the 'default' namespace is reserved")

    monkeypatch.setattr(namespaces, "delete_namespace", fake_delete)
    response = client.delete("/namespaces/default")
    assert response.status_code == 400
    assert "error" in response.json()


def test_delete_unknown_namespace_404(monkeypatch):
    async def fake_delete(name):
        raise namespaces.NamespaceNotFoundError(f"unknown namespace: {name}")

    monkeypatch.setattr(namespaces, "delete_namespace", fake_delete)
    response = client.delete("/namespaces/ghost")
    assert response.status_code == 404
    assert "error" in response.json()


def test_delete_non_empty_namespace_409(monkeypatch):
    async def fake_delete(name):
        raise namespaces.NamespaceNotEmptyError(f"namespace still has rows: {name}")

    monkeypatch.setattr(namespaces, "delete_namespace", fake_delete)
    response = client.delete("/namespaces/team-a")
    assert response.status_code == 409
    assert "error" in response.json()
