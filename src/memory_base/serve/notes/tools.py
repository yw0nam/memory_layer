"""MCP tools for agent notes: save, list, and curate them through the REST API."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from mcp.server.fastmcp import Context
from pydantic import Field

from memory_base.serve.common import rest_client
from memory_base.serve.common.http import AUTHOR_SCOPE_DESCRIPTION, AuthorScope, author_params


async def list_notes(
    tags: list[str] | None = None,
    kind: str | None = None,
    since: str | None = None,
    until: str | None = None,
    include_archived: bool = False,
    namespace: str | None = None,
    author: Annotated[AuthorScope, Field(description=AUTHOR_SCOPE_DESCRIPTION)] = None,
    limit: int | None = None,
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    """List stored notes deterministically — no search query, no relevance ranking.

    Use this for exact reads a similarity search cannot promise to be complete:
    every note carrying a tag (e.g. loading preference notes at
    session start), or every note saved in a time window ("what was saved last
    week"). Works without the embedding backend. Returns up to `limit` notes
    (default 50, max 200) newest-first, each with id, kind, text (truncated to
    2000 chars), tags, author, namespace, and date (YYYY-MM-DD), plus the
    lineage it records: `supersedes`, `archived_by`, and `replaced_by` (the note that
    superseded it).

    `tags` matches notes carrying any of the given tags. `kind` is "personal" or
    "work". `since`/`until` are ISO 8601 dates or datetimes (a bare date
    covers that whole day; naive values are read as UTC). `include_archived`
    adds superseded notes, marked "archived": true. `author` narrows the listing.
    `namespace` narrows to one namespace; omitted, it covers every namespace.
    """
    params: list[tuple[str, str]] = [("tags", tag) for tag in tags or []]
    if kind is not None:
        params.append(("kind", kind))
    if since is not None:
        params.append(("since", since))
    if until is not None:
        params.append(("until", until))
    if include_archived:
        params.append(("include_archived", "true"))
    if namespace is not None:
        params.append(("namespace", namespace))
    params += author_params(author)
    if limit is not None:
        params.append(("limit", str(limit)))
    return await rest_client.call(
        "GET",
        "/notes",
        params=params,
        headers=rest_client.auth_headers(ctx),
    )


async def save_memory(
    content: str,
    tags: Annotated[
        list[str],
        Field(
            description="Required; the first tag names the subject so a later search can "
            "narrow to it."
        ),
    ],
    kind: Annotated[
        Literal["personal", "work"],
        Field(
            description='"personal" (the user, their life, their day, moments with you) or '
            '"work" (their work and projects); a label for search that never decides whether the '
            "note is stored."
        ),
    ],
    supersedes: Annotated[
        str | None, Field(description="Id of the note this one replaces; that note is archived.")
    ] = None,
    allow_similar: Annotated[
        bool,
        Field(
            description="True only when a near-identical active note records a genuinely "
            "different fact."
        ),
    ] = False,
    shared: Annotated[
        bool,
        Field(
            description="True makes the note every agent sees in its default read scope; "
            "you stay its author."
        ),
    ] = False,
    namespace: Annotated[str | None, Field(description="Defaults to `default`.")] = None,
    occurred_at: Annotated[
        str | None,
        Field(
            description="ISO 8601, not in the future: when the remembered event happened; "
            "pass it when the note records an event."
        ),
    ] = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Remember the user and their work for later sessions.

    Save a note when a later session needs it and cannot get it elsewhere: the user would
    otherwise be asked again, it is something you gave them that they may want again, or
    code, version control, the tracker, and documents cannot answer it. A work decision
    keeps its reason. Do NOT save session progress (send_message carries it), what a PR,
    issue, commit, or file already says, generic advice, or filler. The server refuses API
    keys, tokens, and a digit code (PIN, password, lock combination, one-time, pickup, or
    verification code) stated in the same sentence as its name; keep every other code,
    password, or combination out yourself.

    Write `content` in English, standalone, subject first, with absolute dates ("on
    2026-09-30"), at most 1,750 characters; a longer note becomes one note per fact.
    Before saving, search_memory the same subject; if the new note replaces one, pass
    `supersedes` and state the previous value, e.g. "20 dozen eggs as of 2023-05 (30 dozen
    as of 2023-01)".

    `shared: true` puts the note in every agent's default read scope; you stay its author.
    """
    body: dict[str, Any] = {
        "content": content,
        "kind": kind,
        "tags": tags,
        "supersedes": supersedes,
        "allow_similar": allow_similar,
    }
    if shared:
        body["shared"] = True
    if namespace is not None:
        body["namespace"] = namespace
    if occurred_at is not None:
        body["occurred_at"] = occurred_at
    return await rest_client.call(
        "POST",
        "/save_memory",
        json=body,
        headers=rest_client.auth_headers(ctx),
    )


async def archive_notes(
    ids: list[str],
    confirm: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Archive the named notes, recording you as the agent that archived them.

    Without `confirm` this previews the rows instead of changing them. Archived
    notes leave search results.
    """
    body: dict[str, Any] = {"ids": ids}
    if confirm:
        body["confirm"] = True
    return await rest_client.call(
        "POST",
        "/admin/archive",
        json=body,
        headers=rest_client.auth_headers(ctx),
    )


async def delete_notes(
    ids: list[str],
    confirm: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Delete the named notes permanently, recording nothing.

    Without `confirm` this previews the rows instead of deleting them. Prefer
    `archive_notes` unless the row must never resurface: a deleted note leaves no trace of who removed it.
    """
    body: dict[str, Any] = {"ids": ids}
    if confirm:
        body["confirm"] = True
    return await rest_client.call(
        "POST",
        "/admin/notes/delete",
        json=body,
        headers=rest_client.auth_headers(ctx),
    )
