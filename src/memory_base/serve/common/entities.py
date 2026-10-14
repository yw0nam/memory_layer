"""The entity slug rule, the registry row shape, the registry lock, and the slug checks shared by features.

The registry lock is one advisory lock. Registry mutations (register, edit, delete) hold it
exclusive; every write that adds or removes entity slugs on notes or facts holds it shared and
takes it before any row lock, so a slug it validated cannot be deleted before it commits.
"""

from __future__ import annotations

import json
import re
from typing import Any

from memory_base.core.config import PG_SCHEMA
from memory_base.serve.common.http import iso

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
TIMESTAMPS = ("consented_at", "created_at", "updated_at")
LOCK_KEY = "hashtextextended('entities', 0)"


def entity_row(row: dict[str, Any]) -> dict[str, Any]:
    """A registry row with its epoch timestamps as ISO 8601 strings."""
    return {**row, **{name: iso(row[name]) for name in TIMESTAMPS}}


async def lock_registry(conn: Any) -> None:
    """Hold the registry lock exclusive until the transaction ends."""
    await conn.execute(f"SELECT pg_advisory_xact_lock({LOCK_KEY})")


async def lock_slugs(conn: Any) -> None:
    """Hold the registry lock shared until the transaction ends."""
    await conn.execute(f"SELECT pg_advisory_xact_lock_shared({LOCK_KEY})")


async def get(conn: Any, slug: str) -> dict[str, Any] | None:
    """The registry row of a slug; None when unknown."""
    row = await conn.fetchrow(f'SELECT * FROM "{PG_SCHEMA}".entities WHERE slug = $1', slug)
    return None if row is None else dict(row)


async def require_registered(conn: Any, slugs: list[str]) -> None:
    """Refuse the first slug that is not a registered entity."""
    if not slugs:
        return
    registered = {
        row["slug"]
        for row in await conn.fetch(
            f'SELECT slug FROM "{PG_SCHEMA}".entities WHERE slug = ANY($1)', slugs
        )
    }
    for slug in slugs:
        if slug not in registered:
            raise ValueError(
                f"unknown entity {slug!r}; register it with register_entity "
                "or reuse a slug from list_entities"
            )


async def set_note_entities(
    conn: Any, note_id: str, metadata: dict[str, Any], slugs: list[str], author: str, now: float
) -> None:
    """Replace a note's entity slugs; an empty list drops the key. Stamps who and when."""
    metadata = {k: v for k, v in metadata.items() if k != "entities"}
    if slugs:
        metadata["entities"] = slugs
    metadata["entities_set_by"] = author
    metadata["entities_set_at"] = now
    await conn.execute(
        f'UPDATE "{PG_SCHEMA}".memory_chunks SET metadata = $2::jsonb WHERE id = $1',
        note_id,
        json.dumps(metadata, ensure_ascii=False),
    )
