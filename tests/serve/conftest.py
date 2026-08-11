"""Shared fixtures for tests/serve: stub API-key auth for REST-layer tests.

Every REST test in this package sends ``X-API-Key: test-key``; this fixture
stubs ``memory_base.serve.auth.authenticate_request`` to resolve that key to
an admin identity (sees every namespace) so pre-existing tests that exercise
arbitrary namespace names keep working without a real namespace registry or
DB. Tests exercising the real auth/namespace-visibility behavior mark
themselves ``@pytest.mark.real_auth`` to opt out and drive the actual code
path (see tests/serve/test_auth_unit.py, test_namespaces_scope_unit.py).
"""

from __future__ import annotations

import pytest

from memory_base.serve import auth, ingest_api, job_store

TEST_API_KEY = "test-key"


@pytest.fixture(autouse=True)
def _stub_api_key_auth(request, monkeypatch):
    if "real_auth" in request.keywords:
        return
    identity = auth.KeyIdentity(
        key_id="test-key-hash",
        label="test",
        home="default",
        is_admin=True,
        allowed=frozenset(),
    )

    async def fake_authenticate_request(plaintext_key):
        return identity if plaintext_key == TEST_API_KEY else None

    monkeypatch.setattr(auth, "authenticate_request", fake_authenticate_request)


@pytest.fixture(autouse=True)
def _isolated_ingest_spool(monkeypatch, tmp_path):
    spool = tmp_path / "ingest-spool"
    monkeypatch.setattr(ingest_api, "INGEST_SPOOL", spool)
    monkeypatch.setattr(job_store, "INGEST_SPOOL", spool)


@pytest.fixture(autouse=True)
def _stub_existing_document_owner(request, monkeypatch):
    """Unit tests never reach a real DB for the ownership gate; integration tests do.

    Tests that care about ownership override this per-test with their own fake.
    """
    if request.node.get_closest_marker("integration") is not None:
        return

    async def fake_existing_document_owner(document_id, namespace="default", schema=None):
        return False, None

    monkeypatch.setattr(ingest_api, "_existing_document_owner", fake_existing_document_owner)
