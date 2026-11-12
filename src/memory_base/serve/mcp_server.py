"""MCP server exposing memory_base's REST API as thin tools.

Transport is stdio by default (local dev); set MCP_TRANSPORT=sse|streamable-http
to serve over HTTP instead (e.g. in Docker). MCP_HOST/MCP_PORT control the
bind address (defaults 0.0.0.0:8765).

Every REST call carries the X-API-Key and X-Author headers: over streamable HTTP
they are read from the incoming MCP request's own headers and forwarded verbatim;
over stdio (no HTTP request to read from) they come from the MEMORY_API_KEY and
MEMORY_AUTHOR environment variables.

Claude Code reaches the server through the memory-base plugin's `.mcp.json` (streamable
HTTP, X-API-Key from the plugin's configuration); stdio stays for local runs with
MEMORY_API_KEY.

Run directly:
    uv run python -m memory_base.serve.mcp_server
"""

from __future__ import annotations

import logging
import os
from typing import Mapping

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from memory_base.core.logger import setup_logging
from memory_base.serve.messages import tools as message_tools
from memory_base.serve.notes import tools as note_tools
from memory_base.serve.search import tools as search_tools

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765

# Served in the initialize response, so it is stated once per client session:
# the store's invariants only. Per-consumer usage belongs to the consumer.
_SERVER_INSTRUCTIONS_OPENING = """\
memory-base holds distilled notes (memory of the user and of the work) plus an addressed
message lane for one-time signals between sessions (never embedded, never searched).

Read first. Before starting a task or answering from recall, search_memory for earlier
decisions on the subject — arriving without them is this server's most common misuse.

"""

_WRITE_POLICY = """\
Write rarely. Save with save_memory and label the note with kind: "personal" for something
about the user (their life, their day, or a moment they shared with you, even during
work), "work" for work knowledge that code, version control, and the tracker cannot
answer; progress or state for the next session goes to send_message. The kind only labels
a note and never decides whether it is stored. The server stores a note once it passes
the validation, credential, and near-duplicate checks, so what is worth keeping is your
call. Before saving, search_memory the same subject; when the new note replaces one,
supersede it rather than adding a note that contradicts it, and if other active notes
state the same stale value, archive them with archive_notes. `shared: true` saves a note
every agent sees."""

_MESSAGE_LANE = """\
Messages are an addressed, one-time signal lane beside the notes: never embedded, never
searchable, listed while pending and consumed by claiming. At the start of a session,
list_messages for your namespaces (it returns headers: subject and status) and
claim_message each one you act on, which returns its content — a claim is exclusive,
and it fires only when called, never automatically from a prefetch hook.
send_message takes two shapes: a general message (status "info", no scope) addressed to
a namespace, or — with a scope repo:<origin> or project:<organization>/<project> — a
handoff, the latest snapshot of a work state statused in_progress, blocked, or
completed. Whoever next works in that scope claims it; a new snapshot supersedes the
pending one, and a completed handoff remains the delivered record of that state. A
completed handoff expires 3 days after it is sent and an in_progress or blocked one stays
pending until it is claimed, superseded, or cancelled, unless its sender gives an
expires_at; a general message expires after the server's default TTL. Keep the
lanes straight: a note is durable knowledge, memory of the user or of the work, read
again whenever it matches; a message is operational state and is consumed once. That
makes a message the right carrier for progress and next steps, and the wrong place for
anything meant to be read more than once."""

_SERVER_INSTRUCTIONS_CLOSING = """\
A namespace is a label for filtering, and a write without one lands in `default`; a read
without a namespace covers every namespace, and no namespace is private. The author is the
X-Author header your client sends: writes are attributed to it, and reads return your own
notes and shared notes. Another agent's notes come back only when `author` names it, as a
list of slugs such as author=["natsume", "claude-code"] (include your own slug to keep
yours), or "all" for every agent. A note's first tag names its subject, usually the repository or domain it belongs
to, so that a later search can narrow to it.

Curate rarely. archive_notes retires a stale note and delete_notes is for rows that
must never resurface; archiving is otherwise always preferred."""

SERVER_INSTRUCTIONS = "\n\n".join(
    (
        _SERVER_INSTRUCTIONS_OPENING.rstrip("\n"),
        _WRITE_POLICY,
        _MESSAGE_LANE,
        _SERVER_INSTRUCTIONS_CLOSING,
    )
)


def resolve_transport_security(
    env: Mapping[str, str],
) -> TransportSecuritySettings | None:
    """Return configured transport security or defer to FastMCP defaults."""
    allowed_hosts = [
        host.strip() for host in env.get("MCP_ALLOWED_HOSTS", "").split(",") if host.strip()
    ]
    if not allowed_hosts:
        return None
    return TransportSecuritySettings(allowed_hosts=allowed_hosts)


mcp = FastMCP(
    "memory-base",
    instructions=SERVER_INSTRUCTIONS,
    transport_security=resolve_transport_security(os.environ),
)


mcp.tool()(search_tools.search_memory)
mcp.tool()(note_tools.list_notes)
mcp.tool()(note_tools.save_memory)
mcp.tool()(message_tools.send_message)
mcp.tool()(message_tools.list_messages)
mcp.tool()(message_tools.claim_message)
mcp.tool()(message_tools.cancel_message)
mcp.tool()(note_tools.archive_notes)
mcp.tool()(note_tools.delete_notes)


def quiet_request_noise() -> None:
    """Routine Ping/ListTools request lines drown real events at INFO."""
    logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)


def resolve_transport(env: Mapping[str, str]) -> tuple[str, str, int]:
    """Return (transport, host, port) from the MCP environment settings.

    MCP_TRANSPORT defaults to stdio; MCP_HOST defaults to "0.0.0.0"; MCP_PORT
    defaults to 8765. An invalid transport value raises ValueError.
    """
    transport = env.get("MCP_TRANSPORT", "stdio").lower()
    if transport not in ("stdio", "sse", "streamable-http"):
        raise ValueError(f"invalid MCP_TRANSPORT: {transport!r}")
    host = env.get("MCP_HOST", DEFAULT_HOST)
    port_raw = env.get("MCP_PORT", str(DEFAULT_PORT))
    try:
        port = int(port_raw)
    except ValueError as e:
        raise ValueError(f"invalid MCP_PORT: {port_raw!r}") from e
    return transport, host, port


if __name__ == "__main__":
    setup_logging()
    quiet_request_noise()
    _transport, _host, _port = resolve_transport(os.environ)
    if _transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.settings.host = _host
        mcp.settings.port = _port
        mcp.run(transport=_transport)
