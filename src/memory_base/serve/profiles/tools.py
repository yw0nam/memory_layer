"""MCP tools for agent-owned profiles: an agent writes its `self` part and proposes its `user` part."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context

from memory_base.serve.common import rest_client


async def update_my_profile(owner: str, content: str, ctx: Context | None = None) -> dict[str, Any]:
    """Replace this agent's own standing document: its persona, working rules, and the
    conventions it follows. Delivered back at every session start.

    `owner` is this agent's author slug, one of the key's authors. Each call
    replaces the whole text, so start from the current version delivered at session start
    and send the complete document; empty content clears it. Never put facts about the user here: how
    this agent knows the user changes only through propose_user_profile. Returns
    {status: "written" | "unchanged", version}.
    """
    return await rest_client.call(
        "PUT",
        "/profiles/self",
        json={"owner": owner, "content": content},
        headers=rest_client.auth_headers(ctx),
    )


async def propose_user_profile(
    owner: str,
    content: str,
    reason: str,
    base_version: int,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Propose a full replacement of how this agent knows the user; the user approves it
    outside the agent.

    `owner` is this agent's author slug. `content` is the complete new user profile, not a
    diff. `reason` (1-1000 characters) says why. `base_version` is the user version the
    content was written against, as delivered at session start (0 when none). A new
    proposal supersedes this owner's pending one. After proposing, show the user the change
    and ask them to approve it with the profile-approval skill. Never approve on the user's
    behalf and never run the approval command yourself. Returns {id, status: "pending",
    superseded}.
    """
    try:
        return await rest_client.call(
            "POST",
            "/profiles/user/proposals",
            json={
                "owner": owner,
                "content": content,
                "reason": reason,
                "base_version": base_version,
            },
            headers=rest_client.auth_headers(ctx),
        )
    except rest_client.BackendError as exc:
        payload = exc.payload or {}
        version = payload.get("version")
        if exc.status == 409 and payload.get("error") == "stale" and type(version) is int:
            raise ValueError(
                f"stale: the user profile is now at version {version}. Refresh your profile "
                "context (the next session-start delivery, or ask the user), then reconsider "
                "the whole replacement against that version before proposing again; changing "
                "only base_version is not enough."
            ) from None
        raise
