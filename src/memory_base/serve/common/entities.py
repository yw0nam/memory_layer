"""The entity slug rule and the active-entity check shared by the registry and the notes."""

from __future__ import annotations

import re
from typing import Any

from memory_base.core.config import PG_SCHEMA

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")


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
