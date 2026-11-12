"""Every note write path refuses a credential before any embedding call or content write."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import httpx
import pytest
from loguru import logger

from memory_base.serve import api
from memory_base.serve.common import rest_client
from memory_base.serve.notes import store, tools

# Uppercase on purpose: tag normalization lowercases, which would hide it from the detector.
AWS_KEY = "AKIA" + "Q" * 16


class Calls:
    """Records every call a refused write must never make."""

    def __init__(self) -> None:
        self.made: list[str] = []

    def forbid(self, monkeypatch, target, name: str) -> None:
        def record(*args, **kwargs):
            self.made.append(name)
            raise AssertionError(f"{name} must not be reached")

        async def arecord(*args, **kwargs):
            record()

        original = getattr(target, name)
        is_async = asyncio.iscoroutinefunction(original)
        monkeypatch.setattr(target, name, arecord if is_async else record)


@pytest.fixture
def calls(monkeypatch):
    recorder = Calls()
    for target, name in [
        (store, "embed_text"),
    ]:
        recorder.forbid(monkeypatch, target, name)

    @asynccontextmanager
    async def acquire(*args, **kwargs):
        recorder.made.append("db.acquire")
        raise AssertionError("db must not be reached")
        yield

    monkeypatch.setattr(store.db, "acquire", acquire)
    return recorder


@pytest.fixture
def log_lines():
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(str(message)), level="TRACE")
    yield lines
    logger.remove(sink)


def _rest():
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app),
        base_url="http://testserver",
        headers={"X-API-Key": "test-key", "X-Author": "claude-code"},
    )


def _mcp_through_rest(monkeypatch):
    monkeypatch.setenv("MEMORY_API_KEY", "test-key")
    monkeypatch.setenv("MEMORY_AUTHOR", "claude-code")
    monkeypatch.setattr(
        rest_client,
        "client",
        lambda: httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url=rest_client.REST_URL
        ),
    )


# ---- notes ------------------------------------------------------------------


def test_save_note_refuses_a_credential_before_the_embedding_and_db(calls, log_lines):
    with pytest.raises(store.CredentialNoteError) as refused:
        asyncio.run(
            store.save_note(
                f"deploy with {AWS_KEY} from the vault", tags=["deploy"], kind="work", author="a"
            )
        )
    assert refused.value.secret_type == "AWS Access Key"
    assert isinstance(refused.value, ValueError)
    assert str(refused.value) == (
        "note contains a credential (AWS Access Key); store the fact without the secret"
    )
    assert calls.made == []
    assert not any(AWS_KEY in line for line in log_lines)


def test_save_note_refuses_a_credential_in_a_raw_tag(calls):
    with pytest.raises(store.CredentialNoteError) as refused:
        asyncio.run(
            store.save_note(
                "prefer ruff for linting", tags=["deploy", AWS_KEY], kind="work", author="a"
            )
        )
    assert refused.value.secret_type == "AWS Access Key"
    assert calls.made == []


def _save_memory_body(content, tags):
    return {"content": content, "kind": "work", "tags": tags}


@pytest.mark.parametrize(
    ("content", "tags"),
    [(f"the deploy key is {AWS_KEY}", ["deploy"]), ("prefer ruff for linting", [AWS_KEY])],
    ids=["content", "tag"],
)
def test_rest_save_memory_maps_a_credential_to_409_without_echoing_it(calls, content, tags):
    async def run():
        async with _rest() as client:
            return await client.post("/save_memory", json=_save_memory_body(content, tags))

    response = asyncio.run(run())
    assert response.status_code == 409
    assert "note contains a credential (AWS Access Key)" in response.json()["error"]
    assert AWS_KEY not in response.text
    assert calls.made == []


def test_mcp_save_memory_surfaces_the_credential_refusal(monkeypatch, calls):
    _mcp_through_rest(monkeypatch)
    with pytest.raises(ValueError) as refused:
        asyncio.run(
            tools.save_memory(
                content=f"the deploy key is {AWS_KEY}",
                tags=["deploy"],
                kind="work",
            )
        )
    assert "note contains a credential (AWS Access Key)" in str(refused.value)
    assert AWS_KEY not in str(refused.value)
    assert calls.made == []
