"""Shared fixtures for tests/serve: stub API-key auth for REST-layer tests.

Every REST test in this package sends ``X-API-Key: test-key`` and
``X-Author: claude-code``; this fixture stubs
``memory_base.serve.access.auth.authenticate_request`` to resolve that key to an
identity so the tests run without a DB. Tests exercising the real auth code
path mark themselves ``@pytest.mark.real_auth`` to opt out (see
tests/serve/test_auth_unit.py).
"""

from __future__ import annotations

import pytest

from memory_base.serve.access import auth
from memory_base.serve.documents import pipeline as document_pipeline

TEST_API_KEY = "test-key"
TEST_AUTHOR = "claude-code"


@pytest.fixture(autouse=True)
def _stub_api_key_auth(request, monkeypatch):
    if "real_auth" in request.keywords:
        return
    identity = auth.KeyIdentity(key_id="test-key-hash", label="test")

    async def fake_authenticate_request(plaintext_key):
        return identity if plaintext_key == TEST_API_KEY else None

    monkeypatch.setattr(auth, "authenticate_request", fake_authenticate_request)


@pytest.fixture(autouse=True)
def _isolated_ingest_spool(monkeypatch, tmp_path):
    spool = tmp_path / "ingest-spool"
    monkeypatch.setattr(document_pipeline, "INGEST_SPOOL", spool)
