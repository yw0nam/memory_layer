"""The user-profile tool's consent step on a client with elicitation and on one without.

No DB/network: rest_client.client is mocked via httpx.MockTransport; the client is the
in-process MCP session, with or without an elicitation callback.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import ElicitResult

from memory_base.serve import mcp_server
from memory_base.serve.common import rest_client

ARGUMENTS = {
    "owner": "natsume",
    "content": "Lives in Seoul.",
    "reason": "they moved",
    "base_version": 1,
}


@pytest.fixture()
def backend(monkeypatch):
    """The fake REST backend: `writes` are the real writes, `dry_runs` the validations."""
    state = SimpleNamespace(writes=[], dry_runs=[])

    def handler(request):
        body = json.loads(request.content)
        if body.get("dry_run"):
            state.dry_runs.append(body)
            return httpx.Response(
                200, json={"status": "valid", "version": 1, "current": "Lives in Busan."}
            )
        state.writes.append(body)
        return httpx.Response(200, json={"status": "written", "version": 2})

    monkeypatch.setattr(
        rest_client,
        "client",
        lambda: httpx.AsyncClient(
            base_url=rest_client.REST_URL, transport=httpx.MockTransport(handler)
        ),
    )
    return state


def _call(arguments, answer=None):
    """Run update_user_profile; `answer` is the elicitation result, None for no elicitation."""
    messages = []

    async def elicit(context, params):
        messages.append(params.message)
        return answer

    async def run():
        callback = elicit if answer is not None else None
        async with create_connected_server_and_client_session(
            mcp_server.mcp._mcp_server, elicitation_callback=callback
        ) as client:
            return await client.call_tool("update_user_profile", arguments)

    return asyncio.run(run()), messages


def _accept(approve=True):
    return ElicitResult(action="accept", content={"approve": approve})


def test_an_accepted_dialog_writes_with_the_elicitation_method(backend):
    result, messages = _call(ARGUMENTS, _accept())
    assert not result.isError
    assert backend.writes == [{**ARGUMENTS, "consent_method": "elicitation"}]
    assert "Reason: they moved" in messages[0]


def test_a_declined_dialog_is_an_error_and_writes_nothing(backend):
    result, _ = _call(ARGUMENTS, _accept(approve=False))
    assert result.isError
    assert backend.writes == []


def test_confirmed_is_refused_on_a_client_that_asks_the_user_itself(backend):
    result, messages = _call({**ARGUMENTS, "confirmed": True}, _accept())
    assert result.isError
    assert messages == []
    assert backend.writes == []


def test_a_client_without_elicitation_gets_the_prompt_then_writes_when_confirmed(backend):
    first, _ = _call(ARGUMENTS)
    assert json.loads(first.content[0].text)["status"] == "needs_consent"
    assert backend.writes == []
    second, _ = _call({**ARGUMENTS, "confirmed": True})
    assert not second.isError
    assert backend.writes == [{**ARGUMENTS, "consent_method": "conversation"}]
