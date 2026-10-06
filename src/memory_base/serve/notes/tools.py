"""MCP tools for agent notes: save, list, and curate them through the REST API."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from mcp.server.fastmcp import Context
from pydantic import Field

from memory_base.serve.common import rest_client


async def list_notes(
    tags: list[str] | None = None,
    kind: str | None = None,
    since: str | None = None,
    until: str | None = None,
    include_archived: bool = False,
    namespace: str | None = None,
    author: str | None = None,
    limit: int | None = None,
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    """List stored notes deterministically — no search query, no relevance ranking.

    Use this for exact reads a similarity search cannot promise to be complete:
    every note carrying a tag (e.g. loading profile or preference notes at
    session start), or every note saved in a time window ("what was saved last
    week"). Works without the embedding backend. Returns up to `limit` notes
    (default 50, max 200) newest-first, each with id, kind, text (truncated to
    2000 chars), tags, author, namespace, and date (YYYY-MM-DD), plus the
    lineage it records: `supersedes`, `archived_by`, `replaced_by` (the note that
    superseded it), `consolidated_into` (the notes a consolidation folded it
    into), `merged_from` and `merged_dates` (the notes a consolidation merged
    into it), `consolidation_action`, and `undone_action`.

    `tags` matches notes carrying any of the given tags. `kind` is "personal" or
    "work". `since`/`until` are ISO 8601 dates or datetimes (a bare date
    covers that whole day; naive values are read as UTC). `include_archived`
    adds superseded notes, marked "archived": true. `author` narrows to notes
    saved by one agent. `namespace` narrows to one namespace the caller's API
    key can access; omitted, it covers every namespace the key can access.
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
    if author is not None:
        params.append(("author", author))
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
    author: Annotated[
        str, Field(description="The saving agent; must be in the key's author allowlist.")
    ],
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
    namespace: Annotated[
        str | None, Field(description="Defaults to the key's home namespace.")
    ] = None,
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
    issue, commit, or file already says, generic advice, or filler. A credential is always
    refused. The session-start profile may narrow or widen these rules; the more specific
    rule wins.

    Write `content` in English, standalone, subject first, with absolute dates ("on
    2026-09-30"). Before saving, search_memory the same subject; if the new note replaces
    one, pass `supersedes` and state the previous value, e.g. "20 dozen eggs as of 2023-05
    (30 dozen as of 2023-01)".
    """
    body: dict[str, Any] = {
        "content": content,
        "author": author,
        "kind": kind,
        "tags": tags,
        "supersedes": supersedes,
        "allow_similar": allow_similar,
    }
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


async def list_memory_duplicates(
    threshold: float | None = None,
    kind: str | None = None,
    limit: int | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """List active note pairs whose meaning nearly coincides.

    Read-only. Each pair carries both notes' id, kind, author, and text plus
    their cosine score, over the namespaces the caller's API key can access.
    Read both sides before acting: merge them into one note with `save_memory` with
    `supersedes`, or drop one with `archive_notes`. `threshold` (default 0.9), `kind`, and `limit` (default 50) narrow the
    scan; `kind` is "personal" or "work".
    """
    params: list[tuple[str, str]] = []
    if threshold is not None:
        params.append(("threshold", str(threshold)))
    if kind is not None:
        params.append(("kind", kind))
    if limit is not None:
        params.append(("limit", str(limit)))
    return await rest_client.call(
        "GET",
        "/admin/duplicates",
        params=params,
        headers=rest_client.auth_headers(ctx),
    )


async def archive_notes(
    ids: list[str],
    author: str,
    confirm: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Archive the named notes, recording `author` as the agent that archived them.

    Without `confirm` this previews the rows instead of changing them. Archived
    notes leave search results and prefetch but stay restorable with
    `restore_notes`; `author` must be in the calling key's author allowlist.
    """
    body: dict[str, Any] = {"ids": ids, "author": author}
    if confirm:
        body["confirm"] = True
    return await rest_client.call(
        "POST",
        "/admin/archive",
        json=body,
        headers=rest_client.auth_headers(ctx),
    )


async def restore_notes(
    ids: list[str],
    confirm: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Bring archived notes back into search results.

    Without `confirm` this previews the rows instead of changing them.
    Restoring clears the archiving agent's name from the note.
    """
    body: dict[str, Any] = {"ids": ids}
    if confirm:
        body["confirm"] = True
    return await rest_client.call(
        "POST",
        "/admin/restore",
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
    `archive_notes` unless the row must never resurface: a deleted note cannot
    be restored and leaves no trace of who removed it.
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
