"""The entity registry: SQL and the uniqueness rules, under one lock.

An entity's slug, normalized name, and normalized aliases are each held by at most one
entity, archived ones included. The server infers no equivalence between names.
"""

from __future__ import annotations

import time
import unicodedata
from datetime import datetime, timezone
from typing import Any

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.core.schema import ensure_schema_once
from memory_base.core.secrets import find_secret
from memory_base.serve.common.http import iso

_ENTITIES = f'"{PG_SCHEMA}".entities'

LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtextextended('entities', 0))"
LIST_SQL = f"SELECT * FROM {_ENTITIES} ORDER BY slug"
INSERT_SQL = f"""
INSERT INTO {_ENTITIES}
  (slug, kind, name, aliases, description, created_by, updated_by, consent_method,
   consented_at, created_at, updated_at)
VALUES ($1, $2, $3, $4, $5, $6, $6, $7, $8, $8, $8)
RETURNING *
"""
UPDATE_SQL = f"""
UPDATE {_ENTITIES}
SET kind = $2, name = $3, aliases = $4, description = $5, updated_by = $6,
    consent_method = $7, consented_at = $8, updated_at = $8
WHERE slug = $1
RETURNING *
"""
ARCHIVE_SQL = f"""
UPDATE {_ENTITIES}
SET archived_at = $2, updated_at = $2, updated_by = $3, consent_method = $4, consented_at = $2
WHERE slug = $1
RETURNING *
"""


class Collision(ValueError):
    """A slug, name, or alias is already held by another entity."""

    def __init__(self, holder: str, value: str) -> None:
        super().__init__(f"{value!r} is already held by entity {holder!r}; reuse it")
        self.holder = holder
        self.value = value


class Stale(ValueError):
    """An edit was written against an `updated_at` the row no longer has."""

    def __init__(self, slug: str, updated_at: float) -> None:
        super().__init__(f"{slug!r} changed since it was read; read it again")
        self.slug = slug
        self.updated_at = iso(updated_at)


def _now() -> float:
    return time.time()


def normalize(text: str) -> str:
    """The comparison form of a name: NFKC, case-folded, whitespace collapsed."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _claims(row: dict[str, Any]) -> set[str]:
    return {row["slug"], normalize(row["name"]), *map(normalize, row["aliases"])}


def _merge(current: dict[str, Any] | None, item: dict[str, Any]) -> dict[str, Any]:
    slug = item["slug"]
    base = item.get("base_updated_at")
    item = {name: value for name, value in item.items() if name != "base_updated_at"}
    if current is None:
        if base is not None:
            raise ValueError(f"creating {slug!r} takes no base_updated_at")
        missing = [field for field in ("kind", "name") if field not in item]
        if missing:
            raise ValueError(f"creating {slug!r} needs {' and '.join(missing)}")
        return {"aliases": [], "description": "", **item}
    if current["archived_at"] is not None:
        raise ValueError(f"{slug!r} is archived; restore is not supported")
    changes_description = "description" in item and item["description"] != current["description"]
    if base is None and changes_description:
        raise ValueError(f"editing the description of {slug!r} needs base_updated_at")
    if base is not None and base != datetime.fromtimestamp(current["updated_at"], tz=timezone.utc):
        raise Stale(slug, current["updated_at"])
    return {**current, **item}


def _refuse_secrets(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        for text in (row["name"], *row["aliases"], row["description"]):
            secret = find_secret(text)
            if secret is not None:
                raise ValueError(f"entity {row['slug']!r} contains a credential ({secret})")


def _refuse_collisions(registry: dict[str, dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    for row in rows:
        for other in registry.values():
            if other["slug"] != row["slug"]:
                shared = _claims(row) & _claims(other)
                if shared:
                    raise Collision(other["slug"], sorted(shared)[0])


class _DryRun(Exception):
    """Raised inside the transaction to roll it back after a validation-only run."""

    def __init__(self, result: Any) -> None:
        self.result = result


async def _locked_registry(conn: Any) -> dict[str, dict[str, Any]]:
    """Take the registry lock, then read every row, archived included."""
    await conn.execute(LOCK_SQL)
    return {row["slug"]: dict(row) for row in await conn.fetch(LIST_SQL)}


async def list_entities(include_archived: bool) -> list[dict[str, Any]]:
    """Every entity ordered by slug; archived ones only when asked."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        rows = [dict(row) for row in await conn.fetch(LIST_SQL)]
    return [row for row in rows if include_archived or row["archived_at"] is None]


async def upsert(
    items: list[dict[str, Any]], author: str, consent_method: str, dry_run: bool = False
) -> list[dict[str, Any]]:
    """Create or edit entities as one batch: all are validated, then all are written.

    A dry run does everything under the lock, then rolls back and returns the rows it would
    have written. Raises ValueError for a refused item and Collision for a held name.
    """
    slugs = [item["slug"] for item in items]
    repeated = sorted({slug for slug in slugs if slugs.count(slug) > 1})
    if repeated:
        raise ValueError(f"slug given more than once in one request: {', '.join(repeated)}")
    try:
        async with db.acquire() as conn:
            await ensure_schema_once(conn)
            async with conn.transaction(isolation="read_committed"):
                registry = await _locked_registry(conn)
                merged = [_merge(registry.get(item["slug"]), item) for item in items]
                _refuse_secrets(merged)
                _refuse_collisions(registry | {row["slug"]: row for row in merged}, merged)
                now = _now()
                written = []
                for row in merged:
                    statement = UPDATE_SQL if row["slug"] in registry else INSERT_SQL
                    arguments = (
                        row["slug"], row["kind"], row["name"], row["aliases"],
                        row["description"], author, consent_method, now,
                    )  # fmt: skip
                    written.append(dict(await conn.fetchrow(statement, *arguments)))
                if dry_run:
                    raise _DryRun(written)
    except _DryRun as stop:
        return stop.result
    return written


async def archive(
    slug: str, author: str, consent_method: str, dry_run: bool = False
) -> dict[str, Any]:
    """Archive an active entity; a dry run validates and rolls back.

    Raises ValueError for an unknown or archived slug.
    """
    try:
        async with db.acquire() as conn:
            await ensure_schema_once(conn)
            async with conn.transaction(isolation="read_committed"):
                registry = await _locked_registry(conn)
                row = registry.get(slug)
                if row is None or row["archived_at"] is not None:
                    raise ValueError(f"no active entity {slug!r}")
                archived = dict(
                    await conn.fetchrow(ARCHIVE_SQL, slug, _now(), author, consent_method)
                )
                if dry_run:
                    raise _DryRun(archived)
    except _DryRun as stop:
        return stop.result
    return archived
