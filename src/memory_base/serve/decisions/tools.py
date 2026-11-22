"""MCP tools for the decision lane: record, retire, list and read a repository's decisions."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context

from memory_base.serve.common import rest_client


async def record_decision(
    scope: str,
    title: str,
    decision: str,
    why: str | None = None,
    links: list[str] | None = None,
    decided_on: str | None = None,
    supersedes: int | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Record a decision the owner made, under the next number D-N of the repository.

    Record, and supersede, only when the owner asks. Draft the entry, save it, then show
    the owner the D-number and the text.

    `title` is 1-200 chars; `decision` is the decision itself; `why` is the reason, and
    the alternatives that were rejected; `decision` and `why` together hold at most 1500
    chars. `links` holds at most 10 strings (issue, PR, ADR or session URLs) of at most
    500 chars each. `decided_on` is an ISO date (YYYY-MM-DD): pass the owner's local date;
    omitted, the server uses today in UTC.

    `scope` is `repo:` followed by the output of `git remote get-url origin` (https and
    scp-style origins are accepted), or the scope line the SessionStart hook prints.

    `supersedes` is the number of an active decision of yours in the same scope that
    this one replaces: the old one leaves lists and recall and stays as history. A
    number that is not an active decision of yours is refused and uses no number.
    Decisions are yours alone; no other author reads them.
    """
    body: dict[str, Any] = {"scope": scope, "title": title, "decision": decision}
    for name, value in (
        ("why", why),
        ("links", links),
        ("decided_on", decided_on),
        ("supersedes", supersedes),
    ):
        if value is not None:
            body[name] = value
    return await rest_client.call(
        "POST", "/decisions", json=body, headers=rest_client.auth_headers(ctx)
    )


async def retire_decision(
    scope: str, number: int, reason: str, ctx: Context | None = None
) -> dict[str, Any]:
    """Retire an active decision D-N with a reason; it leaves lists and recall.

    Retire only when the owner asks. A decision that is not active is refused. `scope`
    is the repository scope, as in `record_decision`.
    """
    return await rest_client.call(
        "POST",
        "/decisions/retire",
        json={"scope": scope, "number": number, "reason": reason},
        headers=rest_client.auth_headers(ctx),
    )


async def list_decisions(
    scope: str, include_inactive: bool = False, ctx: Context | None = None
) -> list[dict[str, Any]]:
    """List your decisions of a repository, newest number first.

    Rows hold number, title, decided_on and status. Only active decisions come back
    unless `include_inactive` is true, which adds the superseded and retired ones.
    `get_decision` returns one in full. `scope` is the repository scope, as in
    `record_decision`.
    """
    return await rest_client.call(
        "GET",
        "/decisions",
        params={"scope": scope, "status": "all" if include_inactive else "active"},
        headers=rest_client.auth_headers(ctx),
    )


async def get_decision(scope: str, number: int, ctx: Context | None = None) -> dict[str, Any]:
    """Read decision D-N of a repository in full.

    Returns number, title, decision, why, links, decided_on, status, the number it
    `supersedes`, the number that replaced it (`superseded_by`) and `retired_reason`.
    An unknown number gets a 404. `scope` is the repository scope, as in `record_decision`.
    """
    return await rest_client.call(
        "GET",
        f"/decisions/{number}",
        params={"scope": scope},
        headers=rest_client.auth_headers(ctx),
    )
