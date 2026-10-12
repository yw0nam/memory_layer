"""The consent step: the user's answer, inside the conversation, to a change put to them.

A client that advertises form elicitation shows a dialog inside the tool call. Any other
client gets the summary back; the agent asks the user and calls again with `confirmed`.
The recorded method is what the client reported, not proof that a person answered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from mcp.server.fastmcp import Context
from pydantic import BaseModel


@dataclass(frozen=True)
class Consent:
    method: Literal["elicitation", "conversation"]


@dataclass(frozen=True)
class NeedsConsent:
    summary: str


class ConsentRefused(ValueError):
    """The user declined, or the call is not allowed on this client."""


class Approve(BaseModel):
    approve: bool


def elicitation_available(ctx: Context | None) -> bool:
    """True when the client advertises form elicitation; an empty capability means form."""
    try:
        capability = ctx.session.client_params.capabilities.elicitation
    except (AttributeError, ValueError):
        return False
    return capability is not None and (capability.form is not None or capability.url is None)


async def obtain(ctx: Context | None, prompt: str, confirmed: bool) -> Consent | NeedsConsent:
    """The user's consent to `prompt`, or a request for the agent to ask the user.

    Raises ConsentRefused when the user declines or `confirmed` is sent to a client that
    asks the user itself.
    """
    if not elicitation_available(ctx):
        return Consent("conversation") if confirmed else NeedsConsent(prompt)
    if confirmed:
        raise ConsentRefused("this client asks the user itself; call again without confirmed")
    result: Any = await ctx.elicit(prompt, Approve)
    if result.action == "accept" and result.data.approve:
        return Consent("elicitation")
    raise ConsentRefused("the user did not consent")
