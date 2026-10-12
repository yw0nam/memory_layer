"""Which clients the consent step treats as able to show a dialog."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from mcp.types import ElicitationCapability, FormElicitationCapability, UrlElicitationCapability

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
