"""MCP tools for documents: queue a text document for ingestion and remove a stored one."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context

from memory_base.adapters.document import MCP_TEXT_EXTENSIONS, extension_for
from memory_base.serve.common import rest_client


async def ingest_document(
    content: str,
    filename: str,
    document_id: str | None = None,
    origin: str | None = None,
    mode: str = "upsert",
    namespace: str | None = None,
    tags: list[str] | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Queue a text document for conversion, chunking, and atomic storage.

    The filename must use a supported text extension: .md, .markdown, .txt,
    .rst, .html, .htm, or .csv. Binary documents upload through REST directly.

    `namespace` picks one namespace the caller's API key can access; omitted,
    the document lands in the key's home namespace. A namespace the key
    cannot access is rejected by the server.

    `tags` are lowercased topical labels stamped on every chunk of the
    document and usable as the `tags` search filter.

    A document carrying a credential (an API key, token, private key, JWT, or
    password in a URL) is refused whole: in its filename, `document_id`,
    `origin`, or tags the call fails; in its content the job fails and
    nothing from it is stored.
    """
    try:
        extension = extension_for(filename)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if extension not in MCP_TEXT_EXTENSIONS:
        raise ValueError("MCP document ingestion supports text formats only")
    data: dict[str, Any] = {"filename": filename, "mode": mode}
    if namespace is not None:
        data["namespace"] = namespace
    if tags:
        data["tags"] = tags
    if document_id is not None:
        data["document_id"] = document_id
    if origin is not None:
        data["origin"] = origin
    payload = await rest_client.call(
        "POST",
        "/ingest/document",
        data=data,
        files={"file": (filename, content.encode("utf-8"))},
        headers=rest_client.auth_headers(ctx),
    )
    return {"job_id": payload["job_id"], "status_url": payload["status_url"]}


async def remove_document(
    document_id: str, namespace: str | None = None, ctx: Context | None = None
) -> dict[str, Any]:
    """Delete a document's stored chunks and table rows from one namespace by its document_id.

    Restricted to an admin key or the document's creator (the key that first
    ingested it); a non-creator, non-admin caller gets a 403, and an unknown
    document in that namespace gets a 404. `namespace` defaults to the
    caller's home namespace. Returns {document_id, namespace, deleted} where
    `deleted` is the number of chunk rows removed; a tabular document's
    `doc_rows` rows are removed with them.
    """
    params = {"namespace": namespace} if namespace is not None else None
    return await rest_client.call(
        "DELETE",
        f"/ingest/documents/{document_id}",
        params=params,
        headers=rest_client.auth_headers(ctx),
    )
