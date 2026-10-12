"""Which clients the consent step treats as able to show a dialog."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from mcp.types import ElicitationCapability, FormElicitationCapability, UrlElicitationCapability

import asyncio

from mcp.server.elicitation import AcceptedElicitation

from memory_base.serve.common import consent
from memory_base.serve.common.consent import elicitation_available


def _ctx(elicitation):
    capabilities = SimpleNamespace(elicitation=elicitation)
    return SimpleNamespace(
        session=SimpleNamespace(client_params=SimpleNamespace(capabilities=capabilities))
    )


@pytest.mark.parametrize(
    ("ctx", "expected"),
    [
        (None, False),
        (_ctx(None), False),
        (_ctx(ElicitationCapability()), True),
        (_ctx(ElicitationCapability(form=FormElicitationCapability())), True),
        (_ctx(ElicitationCapability(url=UrlElicitationCapability())), False),
    ],
    ids=["no-ctx", "absent", "empty-means-form", "form", "url-only"],
)
def test_elicitation_available(ctx, expected):
    assert elicitation_available(ctx) is expected


def _eliciting(ctx):
    async def elicit(message, schema):
        return AcceptedElicitation(data=consent.Approve(approve=True))

    ctx.elicit = elicit
    return ctx


def test_obtain_treats_a_url_only_client_as_conversational():
    ctx = _eliciting(_ctx(ElicitationCapability(url=UrlElicitationCapability())))
    assert asyncio.run(consent.obtain(ctx, "change", False)) == consent.NeedsConsent("change")
    assert asyncio.run(consent.obtain(ctx, "change", True)) == consent.Consent("conversation")


def test_obtain_treats_an_empty_capability_as_elicitation():
    ctx = _eliciting(_ctx(ElicitationCapability()))
    assert asyncio.run(consent.obtain(ctx, "change", False)) == consent.Consent("elicitation")
