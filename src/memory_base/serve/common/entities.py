"""The entity slug rule, the registry row shape, and the active-entity check shared by features."""

from __future__ import annotations

import re
from typing import Any

from memory_base.core.config import PG_SCHEMA
from memory_base.serve.common.http import iso

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
TIMESTAMPS = ("consented_at", "created_at", "updated_at", "archived_at")


def entity_row(row: dict[str, Any]) -> dict[str, Any]:
    """A registry row with its epoch timestamps as ISO 8601 strings."""
    return {**row, **{name: iso(row[name]) for name in TIMESTAMPS}}


async def get(conn: Any, slug: str) -> dict[str, Any] | None:
    """The registry row of a slug, archived or not; None when unknown."""
    row = await conn.fetchrow(f'SELECT * FROM "{PG_SCHEMA}".entities WHERE slug = $1', slug)
    return None if row is None else dict(row)


async def require_active(conn: Any, slugs: list[str]) -> None:
    """Refuse the first slug that is not an active entity."""
    if not slugs:
        return
    active = {
        row["slug"]
        for row in await conn.fetch(
            f'SELECT slug FROM "{PG_SCHEMA}".entities WHERE slug = ANY($1) AND archived_at IS NULL',
            slugs,
        )
    }
    for slug in slugs:
        if slug not in active:
            raise ValueError(
                f"unknown or archived entity {slug!r}; register it with register_entity "
                "or reuse a slug from list_entities"
            )
