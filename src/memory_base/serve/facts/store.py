"""Facts: SQL and the write rules of one batch, under one advisory lock.

A fact is one property or relation of an entity, cited to the agent note it came from; its
namespace is that note's. A current row has no `archived_at`; a replace or remove closes it.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.core.schema import ensure_schema_once
from memory_base.serve.common import entities
from memory_base.serve.common.http import iso

_FACTS = f'"{PG_SCHEMA}".facts'
MAX_CARD_FACTS = 200

LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtextextended('facts', 0))"
NOTE_SQL = f"""
SELECT namespace FROM "{PG_SCHEMA}".memory_chunks
WHERE id = $1 AND source_type = 'agent_note' AND archived_at IS NULL
"""
FACT_SQL = f"SELECT * FROM {_FACTS} WHERE id = $1"
CURRENT_SQL = f"""
SELECT * FROM {_FACTS} WHERE entity = $1 AND attribute = $2 AND archived_at IS NULL
ORDER BY created_at
"""
INSERT_SQL = f"""
INSERT INTO {_FACTS}
  (id, namespace, entity, attribute, value_text, value_entity, note_id, author, created_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
RETURNING *
"""
CLOSE_SQL = f"UPDATE {_FACTS} SET archived_at = $2, replaced_by = $3 WHERE id = $1 RETURNING *"
CARD_SQL = f"""
SELECT * FROM {_FACTS}
WHERE {{column}} = $1 AND ($2 OR archived_at IS NULL)
  AND ($3::text[] IS NULL OR namespace = ANY($3::text[]))
ORDER BY created_at DESC
LIMIT {MAX_CARD_FACTS}
"""


class _DryRun(Exception):
    """Raised inside the transaction to roll it back after a validation-only run."""

    def __init__(self, result: Any) -> None:
        self.result = result


def _now() -> float:
    return time.time()


def public(row: dict[str, Any]) -> dict[str, Any]:
    """A fact row as JSON: ids as strings, timestamps as ISO 8601."""
    return {
        **row,
        "id": str(row["id"]),
        "replaced_by": None if row["replaced_by"] is None else str(row["replaced_by"]),
        "created_at": iso(row["created_at"]),
        "archived_at": iso(row["archived_at"]),
    }


async def _note_namespace(conn: Any, note_id: str) -> str:
    row = await conn.fetchrow(NOTE_SQL, note_id)
    if row is None:
        raise ValueError(f"note_id {note_id!r} is not an active agent note")
    return row["namespace"]


async def _current_fact(conn: Any, fact_id: uuid.UUID) -> dict[str, Any]:
    row = await conn.fetchrow(FACT_SQL, fact_id)
    if row is None or row["archived_at"] is not None:
        raise ValueError(f"no current fact {fact_id}")
    return dict(row)


async def _insert(
    conn: Any, namespace: str, entity: str, attribute: str, op: dict[str, Any], author: str
) -> dict[str, Any]:
    return dict(
        await conn.fetchrow(
            INSERT_SQL,
            uuid.uuid4(),
            namespace,
            entity,
            attribute,
            op.get("value_text"),
            op.get("value_entity"),
            op["note_id"],
            author,
            _now(),
        )  # fmt: skip
    )


async def _apply_one(conn: Any, op: dict[str, Any], author: str) -> tuple[str, dict[str, Any]]:
    namespace = await _note_namespace(conn, op["note_id"])
    if op["op"] == "add":
        entity, attribute = op["entity"], op["attribute"]
    else:
        old = await _current_fact(conn, op["id"])
        entity, attribute = old["entity"], op.get("attribute", old["attribute"])
    if op["op"] != "remove":
        value_entity = op.get("value_entity")
        await entities.require_active(conn, [entity, *([value_entity] if value_entity else [])])
    if op["op"] == "remove":
        return "written", dict(await conn.fetchrow(CLOSE_SQL, op["id"], _now(), None))
    if op["op"] == "add":
        current = [dict(row) for row in await conn.fetch(CURRENT_SQL, entity, attribute)]
        for row in current:
            if (row["value_text"], row["value_entity"]) == (
                op.get("value_text"), op.get("value_entity"),
            ):  # fmt: skip
                return "unchanged", row
        if current and not op.get("allow_multiple"):
            raise ValueError(
                f"{entity!r} already has a current {attribute!r} fact, {current[0]['id']}; "
                "replace it or pass allow_multiple"
            )
    new = await _insert(conn, namespace, entity, attribute, op, author)
    if op["op"] == "replace":
        await conn.fetchrow(CLOSE_SQL, op["id"], new["created_at"], new["id"])
    return "written", new


async def apply(
    ops: list[dict[str, Any]], author: str, dry_run: bool = False
) -> list[dict[str, Any]]:
    """Apply a batch of add, replace, and remove ops all or nothing.

    A dry run does everything under the lock, then rolls back and returns the planned rows.
    Raises ValueError naming the index of the refused op.
    """
    try:
        async with db.acquire() as conn:
            await ensure_schema_once(conn)
            async with conn.transaction(isolation="read_committed"):
                await conn.execute(LOCK_SQL)
                results = []
                for index, op in enumerate(ops):
                    try:
                        status, row = await _apply_one(conn, op, author)
                    except ValueError as exc:
                        raise ValueError(f"op {index}: {exc}") from None
                    if status == "written" and dry_run:
                        status = "planned"
                    results.append({"op": op["op"], "status": status, **public(row)})
                if dry_run:
                    raise _DryRun(results)
    except _DryRun as stop:
        return stop.result
    return results


async def card(
    slug: str, include_past: bool, namespaces: list[str] | None
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]] | None:
    """The registry row of `slug`, the facts it holds, and the facts that point at it.

    Facts are current ones unless `include_past`, newest first, within `namespaces` (None: all).
    Returns None for an unknown slug.
    """
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        entity = await entities.get(conn, slug)
        if entity is None:
            return None
        sides = [
            [
                public(dict(row))
                for row in await conn.fetch(
                    CARD_SQL.format(column=column), slug, include_past, namespaces
                )
            ]
            for column in ("entity", "value_entity")
        ]
    return entity, sides[0], sides[1]
