"""The entity registry: SQL and the uniqueness rules, under one lock.

An entity's slug, normalized name, and normalized aliases are each held by at most one
entity, archived ones included. The server infers no equivalence between names.
"""

from __future__ import annotations

import time
import unicodedata
from typing import Any

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.core.schema import ensure_schema_once
from memory_base.core.secrets import find_secret

_ENTITIES = f'"{PG_SCHEMA}".entities'

LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtextextended('entities', 0))"
LIST_SQL = f"SELECT * FROM {_ENTITIES} ORDER BY slug"
INSERT_SQL = f"""
INSERT INTO {_ENTITIES}
  (slug, kind, name, aliases, description, created_by, updated_by, consent_method,
   consented_at, created_at, updated_at)
VALUES ($1, $2, $3, $4, $5, $6, $6, $7, $8, $8, $8)
"""
UPDATE_SQL = f"""
UPDATE {_ENTITIES}
SET kind = $2, name = $3, aliases = $4, description = $5, updated_by = $6,
    consent_method = $7, consented_at = $8, updated_at = $8
WHERE slug = $1
"""
ARCHIVE_SQL = f"""
UPDATE {_ENTITIES}
SET archived_at = $2, updated_at = $2, updated_by = $3, consent_method = $4, consented_at = $2
WHERE slug = $1
"""


class Collision(ValueError):
    """A slug, name, or alias is already held by another entity."""

    def __init__(self, holder: str, value: str) -> None:
        super().__init__(f"{value!r} is already held by entity {holder!r}; reuse it")
        self.holder = holder
        self.value = value


def _now() -> float:
    return time.time()


def normalize(text: str) -> str:
    """The comparison form of a name: NFKC, case-folded, whitespace collapsed."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _claims(row: dict[str, Any]) -> set[str]:
    return {row["slug"], normalize(row["name"]), *map(normalize, row["aliases"])}


def _merge(current: dict[str, Any] | None, item: dict[str, Any]) -> dict[str, Any]:
    slug = item["slug"]
    if current is None:
        missing = [field for field in ("kind", "name") if field not in item]
        if missing:
            raise ValueError(f"creating {slug!r} needs {' and '.join(missing)}")
        return {"aliases": [], "description": "", **item}
    if current["archived_at"] is not None:
        raise ValueError(f"{slug!r} is archived; restore is not supported")
    return {**current, **item}


def _validate(registry: dict[str, dict[str, Any]], merged: list[dict[str, Any]]) -> None:
    for row in merged:
        for text in (row["name"], *row["aliases"], row["description"]):
            secret = find_secret(text)
            if secret is not None:
                raise ValueError(f"entity {row['slug']!r} contains a credential ({secret})")
        for other in registry.values():
            if other["slug"] != row["slug"]:
                shared = _claims(row) & _claims(other)
                if shared:
                    raise Collision(other["slug"], sorted(shared)[0])


async def list_entities(include_archived: bool) -> list[dict[str, Any]]:
    """Every entity ordered by slug; archived ones only when asked."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        rows = [dict(row) for row in await conn.fetch(LIST_SQL)]
    return [row for row in rows if include_archived or row["archived_at"] is None]


async def upsert(
    items: list[dict[str, Any]], author: str, consent_method: str
) -> list[dict[str, Any]]:
    """Create or edit entities as one batch: all are validated, then all are written.

    Raises ValueError for a refused item and Collision when a name is already held.
    """
    slugs = [item["slug"] for item in items]
    repeated = sorted({slug for slug in slugs if slugs.count(slug) > 1})
    if repeated:
        raise ValueError(f"slug given more than once in one request: {', '.join(repeated)}")
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction(isolation="read_committed"):
            await conn.execute(LOCK_SQL)
            registry = {row["slug"]: dict(row) for row in await conn.fetch(LIST_SQL)}
            merged = [_merge(registry.get(item["slug"]), item) for item in items]
            _validate(registry | {row["slug"]: row for row in merged}, merged)
            now = _now()
            for row in merged:
                if row["slug"] in registry:
                    await conn.execute(
                        UPDATE_SQL, row["slug"], row["kind"], row["name"], row["aliases"],
                        row["description"], author, consent_method, now,
                    )  # fmt: skip
                else:
                    await conn.execute(
                        INSERT_SQL, row["slug"], row["kind"], row["name"], row["aliases"],
                        row["description"], author, consent_method, now,
                    )  # fmt: skip
    written = []
    for row in merged:
        old = registry.get(row["slug"])
        created = {} if old else {"created_by": author, "created_at": now, "archived_at": None}
        written.append(
            {**row, **created, "updated_by": author, "consent_method": consent_method,
             "consented_at": now, "updated_at": now}
        )  # fmt: skip
    return written


async def archive(slug: str, author: str, consent_method: str) -> dict[str, Any]:
    """Archive an active entity. Raises ValueError for an unknown or archived slug."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction(isolation="read_committed"):
            await conn.execute(LOCK_SQL)
            registry = {row["slug"]: dict(row) for row in await conn.fetch(LIST_SQL)}
            row = registry.get(slug)
            if row is None or row["archived_at"] is not None:
                raise ValueError(f"no active entity {slug!r}")
            now = _now()
            await conn.execute(ARCHIVE_SQL, slug, now, author, consent_method)
    return {
        **row, "archived_at": now, "updated_at": now, "updated_by": author,
        "consent_method": consent_method, "consented_at": now,
    }  # fmt: skip
