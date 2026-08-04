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

from memory_base.serve import auth

TEST_API_KEY = "test-key"


@pytest.fixture(autouse=True)
def _stub_api_key_auth(request, monkeypatch):
    if "real_auth" in request.keywords:
        return
    identity = auth.KeyIdentity(label="test", home="default", is_admin=True, allowed=frozenset())

    async def fake_authenticate_request(plaintext_key):
        return identity if plaintext_key == TEST_API_KEY else None

    monkeypatch.setattr(auth, "authenticate_request", fake_authenticate_request)
