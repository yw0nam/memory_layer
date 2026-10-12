"""MCP tools for agent-owned profiles: an agent reads its profile, writes its `self` part,
and changes its `user` part with the user's consent.
"""

from __future__ import annotations

import difflib
from typing import Any

from mcp.server.fastmcp import Context

from memory_base.serve.common import consent, rest_client


async def get_my_profile(owner: str, ctx: Context | None = None) -> dict[str, Any]:
    """Return this agent's current profile: its `self` part and its `user` part (how it knows
    the user), each with its version.

    `owner` is this agent's author slug, one of the key's authors. Read it before
    update_my_profile or update_user_profile when the session-start delivery is missing or
    may be stale; `user_version` is the base_version an update must be written against.
    Returns {owner, self_version, self, user_version, user}.
    """
    return await rest_client.call(
        "GET",
        "/profiles",
        params={"owner": owner},
        headers=rest_client.auth_headers(ctx),
    )


async def update_my_profile(owner: str, content: str, ctx: Context | None = None) -> dict[str, Any]:
    """Replace this agent's own standing document: its persona, working rules, and the
    conventions it follows. Delivered back at every session start.

    `owner` is this agent's author slug, one of the key's authors. Each call
    replaces the whole text, so start from the current version, delivered at session start
    or read with get_my_profile, and send the complete document; empty content clears it.
    Never put facts about the user here: how this agent knows the user changes only through
    update_user_profile. `Remember` and `Don't remember` sections here are rules for what
    this agent keeps with save_memory, not facts; the user part's sections win over them.
    Returns {status: "written" | "unchanged", version}.
    """
    return await rest_client.call(
        "PUT",
        "/profiles/self",
        json={"owner": owner, "content": content},
        headers=rest_client.auth_headers(ctx),
    )


def change_body(current: str, content: str, base_version: int) -> str:
    """What the user is shown of the change: a unified diff, or the whole text when shorter."""
    if not content:
        return f"Clear the user profile. Current text:\n{current}"
    diff = "\n".join(
        difflib.unified_diff(
            current.splitlines(),
            content.splitlines(),
            fromfile=f"v{base_version}",
            tofile=f"v{base_version + 1}",
            lineterm="",
            n=2,
        )
    )
    if not current or len(diff) >= len(content):
        return f"Full replacement:\n{content}"
    return diff


def _stale(exc: rest_client.BackendError) -> ValueError | None:
    payload = exc.payload or {}
    version = payload.get("version")
    if exc.status == 409 and payload.get("error") == "stale" and type(version) is int:
        return ValueError(
            f"stale: the user profile is now at version {version}. Refresh your profile "
            "context with get_my_profile, then reconsider the whole replacement against "
            "that version before updating again; changing only base_version is not enough."
        )
    return None


async def _put_user(body: dict[str, Any], ctx: Context | None) -> dict[str, Any]:
    try:
        return await rest_client.call(
            "PUT", "/profiles/user", json=body, headers=rest_client.auth_headers(ctx)
        )
    except rest_client.BackendError as exc:
        stale = _stale(exc)
        if stale is None:
            raise
        raise stale from None


async def update_user_profile(
    owner: str,
    content: str,
    reason: str,
    base_version: int,
    confirmed: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Replace how this agent knows the user, with the user's consent.

    `owner` is this agent's author slug. `content` is the complete new user profile, not a
    diff. `reason` (1-1000 characters) says why. `base_version` is the user version the
    content was written against, as delivered at session start or read with get_my_profile
    (0 when none); a stale base is refused before any consent, so re-read and reconsider.
    The user consents in a dialog in this call; on a client without one the tool returns
    {status: "needs_consent", summary} and writes nothing, so ask the user in the
    conversation, then call again with confirmed=true and the same input. In an unattended
    session never pass confirmed; send a message instead. What the user wants kept or left
    out of save_memory goes in its `Remember` and `Don't remember` sections, which outrank
    this agent's own. Returns {status: "written" | "unchanged", version}.
    """
    body = {"owner": owner, "content": content, "reason": reason, "base_version": base_version}
    dry = await _put_user({**body, "consent_method": "conversation", "dry_run": True}, ctx)
    if dry["status"] == "unchanged":
        return dry
    prompt = (
        f"memory-base asks for your consent to change how {owner} knows you "
        f"(user profile v{base_version} -> v{base_version + 1}):\n"
        f"Reason: {reason}\n{change_body(dry['current'], content, base_version)}"
    )
    result = await consent.obtain(ctx, prompt, confirmed)
    if isinstance(result, consent.NeedsConsent):
        return {"status": "needs_consent", "summary": prompt}
    return await _put_user({**body, "consent_method": result.method}, ctx)
