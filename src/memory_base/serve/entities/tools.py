"""MCP tools for the entity registry: read it, and register entities with the user's consent."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context

from memory_base.serve.common import consent, rest_client

FIELDS = ("kind", "name", "aliases", "description")
PROMPT_HEADER = "memory-base asks for your consent to change the entity registry:"


def _show(value: Any) -> str:
    return ", ".join(value) if isinstance(value, list) else str(value)


def _line(item: dict[str, Any], current: dict[str, Any] | None) -> str | None:
    """The summary line of one item; None for an edit that changes nothing."""
    slug = item["slug"]
    if current is None:
        return (
            f"create {slug} ({item.get('kind')}): {item.get('name')}; "
            f"aliases: {_show(item.get('aliases', []))}; "
            f"description: {item.get('description', '')}"
        )
    changes = [
        f"{field}: {_show(current[field])} -> {_show(item[field])}"
        for field in FIELDS
        if field in item and item[field] != current[field]
    ]
    return f"edit {slug}: " + "; ".join(changes) if changes else None


def _based(item: dict[str, Any], current: dict[str, Any] | None) -> dict[str, Any]:
    """An edit of a description carries the `updated_at` of the row it was written against."""
    if current is None or "description" not in item:
        return item
    return {**item, "base_updated_at": current["updated_at"]}


def _delete_summary(slug: str, name: str, plan: dict[str, Any]) -> str:
    notes = plan["note_ids"]
    return (
        f"delete {slug} ({name}): unlink it from {plan['notes']} notes"
        f"{': ' + ', '.join(notes) if notes else ''}; delete {plan['facts']} facts of it and "
        f"{plan['incoming']} facts of other entities that point at it"
    )


async def list_entities(ctx: Context | None = None) -> dict[str, Any]:
    """List the entity registry: every person, agent, project, organisation, place, or thing
    the user has consented to, as {slug, kind, name, aliases, description, ...}.

    Read it before register_entity: the registry is what makes two writers agree on one slug
    for one thing. Match a name by its slug, name, or any alias.
    """
    return await rest_client.call("GET", "/entities", headers=rest_client.auth_headers(ctx))


async def register_entity(
    entities: list[dict[str, Any]],
    confirmed: bool = False,
    delete: bool = False,
    expect: dict[str, int] | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Create, edit, or delete entities in the registry, with the user's consent.

    Reuse before create: call list_entities (or read the registry the server shows) first; a
    slug, name, or alias that already exists is refused and the refusal names its holder.
    `entities` are {slug, kind, name, aliases, description}; an edit sends the slug and the
    fields that change (aliases replace the list). delete=true takes one {slug} and, in one
    step, unlinks the slug from every note (archived ones included), deletes the entity's
    facts and the facts of other entities that point at it, and frees its slug, name, and
    aliases; nothing reattaches when the name is registered again. Every create, edit, and
    delete is put to the user: on a client with elicitation a dialog appears in this call;
    on others the tool returns {status: "needs_consent", summary} and writes nothing, so ask
    the user in the conversation, then call again with confirmed=true and the same input; a
    delete's reply also carries expect, the {notes, facts, incoming} counts, to pass back
    unchanged; the delete is refused if the counts changed meanwhile. In an unattended session (cron, no user
    present) never pass confirmed; send a message naming the entity on the messages lane
    instead, and the next interactive session asks. A refused request never reaches the user.
    The user is an entity of kind person; a project entity carries its repository origin
    among its aliases. For a person the description says who they are; for an agent it holds
    the user's standing directions to that agent, including its `Remember` and `Don't
    remember` lines, and only the agent's own description governs it.
    """
    headers = rest_client.auth_headers(ctx)
    current = await rest_client.call("GET", "/entities", headers=headers)
    registry = {row["slug"]: row for row in current["entities"]}
    if delete:
        return await _delete(entities, confirmed, expect, registry, headers, ctx)
    entities = [_based(item, registry.get(item["slug"])) for item in entities]
    pairs = [(item, _line(item, registry.get(item["slug"]))) for item in entities]
    entities = [item for item, line in pairs if line is not None]
    if not entities:
        return {"status": "unchanged"}
    summary = "\n".join(line for _, line in pairs if line is not None)
    body = {"entities": entities}
    await rest_client.call(
        "POST",
        "/entities",
        json={**body, "consent_method": "conversation", "dry_run": True},
        headers=headers,
    )
    result = await consent.obtain(ctx, f"{PROMPT_HEADER}\n{summary}", confirmed)
    if isinstance(result, consent.NeedsConsent):
        return {"status": "needs_consent", "summary": summary}
    return await rest_client.call(
        "POST", "/entities", json={**body, "consent_method": result.method}, headers=headers
    )


async def _delete(
    entities: list[dict[str, Any]],
    confirmed: bool,
    expect: dict[str, int] | None,
    registry: dict[str, dict[str, Any]],
    headers: dict[str, str],
    ctx: Context | None,
) -> dict[str, Any]:
    """Preview a delete with a dry run, put it to the user, and delete with the previewed counts."""
    if len(entities) != 1 or set(entities[0]) != {"slug"}:
        raise ValueError("delete takes exactly one item with only a slug")
    if confirmed and expect is None:
        raise ValueError("a confirmed delete needs expect, the counts of the summary")
    slug = entities[0]["slug"]
    path = f"/entities/{slug}/delete"
    plan = await rest_client.call("POST", path, json={"dry_run": True}, headers=headers)
    counts = {key: plan[key] for key in ("notes", "facts", "incoming")}
    summary = _delete_summary(slug, registry[slug]["name"], plan)
    result = await consent.obtain(ctx, f"{PROMPT_HEADER}\n{summary}", confirmed)
    if isinstance(result, consent.NeedsConsent):
        return {"status": "needs_consent", "summary": summary, "expect": counts}
    body = {"expect": counts if result.method == "elicitation" else expect}
    return await rest_client.call("POST", path, json=body, headers=headers)
