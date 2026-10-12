"""MCP tools for the entity registry: read it, and register entities with the user's consent."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context

from memory_base.serve.common import consent, rest_client

FIELDS = ("kind", "name", "aliases", "description")


def _show(value: Any) -> str:
    return ", ".join(value) if isinstance(value, list) else str(value)


def _line(item: dict[str, Any], current: dict[str, Any] | None) -> str:
    slug = item["slug"]
    if current is None:
        aliases = _show(item.get("aliases", []))
        description = item.get("description", "")
        return (
            f"create {slug} ({item.get('kind')}): {item.get('name')}; "
            f"aliases: {aliases}; description: {description}"
        )
    changes = [
        f"{field}: {_show(current[field])} -> {_show(item[field])}"
        for field in FIELDS
        if field in item and item[field] != current[field]
    ]
    return f"edit {slug}: " + ("; ".join(changes) or "no change")


def _summary(
    items: list[dict[str, Any]], registry: dict[str, dict[str, Any]], archive: bool
) -> str:
    if archive:
        slug = items[0]["slug"]
        current = registry.get(slug)
        return f"archive {slug} ({current['name'] if current else 'unknown'})"
    return "\n".join(_line(item, registry.get(item["slug"])) for item in items)


async def list_entities(
    include_archived: bool = False, ctx: Context | None = None
) -> dict[str, Any]:
    """List the entity registry: every person, agent, project, organisation, place, or thing
    the user has consented to, as {slug, kind, name, aliases, description, ...}.

    Read it before register_entity: the registry is what makes two writers agree on one slug
    for one thing. Match a name by its slug, name, or any alias. `include_archived` also
    returns archived entities, whose names stay reserved.
    """
    return await rest_client.call(
        "GET",
        "/entities",
        params={"include_archived": str(include_archived).lower()},
        headers=rest_client.auth_headers(ctx),
    )


async def register_entity(
    entities: list[dict[str, Any]],
    author: str,
    confirmed: bool = False,
    archive: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Create, edit, or archive entities in the registry, with the user's consent.

    Reuse before create: call list_entities (or read the registry the server shows) first; a
    slug, name, or alias that already exists is refused and the refusal names its holder.
    `entities` are {slug, kind, name, aliases, description}; an edit sends the slug and the
    fields that change (aliases replace the list); archive=true takes one {slug}. Every
    create, edit, and archive is put to the user: on a client with elicitation a dialog
    appears in this call; on others the tool returns {status: "needs_consent", summary} and
    writes nothing, so ask the user in the conversation, then call again with confirmed=true
    and the same input. In an unattended session (cron, no user present) never pass confirmed;
    send a message naming the entity on the messages lane instead, and the next interactive
    session asks. The user is an entity of kind person; a project entity carries its
    repository origin among its aliases. `author` is one of the key's authors.
    """
    if archive and (len(entities) != 1 or set(entities[0]) != {"slug"}):
        raise ValueError("archive takes exactly one item with only a slug")
    headers = rest_client.auth_headers(ctx)
    current = await rest_client.call(
        "GET", "/entities", params={"include_archived": "true"}, headers=headers
    )
    registry = {row["slug"]: row for row in current["entities"]}
    summary = _summary(entities, registry, archive)
    consent_result = await consent.obtain(ctx, summary, confirmed)
    if isinstance(consent_result, consent.NeedsConsent):
        return {"status": "needs_consent", "summary": summary}
    body = {"author": author, "consent_method": consent_result.method}
    path, body = (
        (f"/entities/{entities[0]['slug']}/archive", body)
        if archive
        else ("/entities", {"entities": entities, **body})
    )
    return await rest_client.call("POST", path, json=body, headers=headers)
