"""The entity registry: SQL and the uniqueness rules, under the registry lock.

An entity's slug, normalized name, and normalized aliases are each held by at most one
entity. The server infers no equivalence between names. Deleting an entity unlinks its notes
and deletes its facts, so its names are free again and nothing reattaches to a new entity.
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
from memory_base.retrieval.search import metadata_dict
from memory_base.serve.common import entities
from memory_base.serve.common.dry_run import DryRun
from memory_base.serve.common.http import iso

_ENTITIES = f'"{PG_SCHEMA}".entities'
_FACTS = f'"{PG_SCHEMA}".facts'

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
LINKED_NOTES_SQL = f"""
SELECT id, metadata FROM "{PG_SCHEMA}".memory_chunks
WHERE source_type = 'agent_note' AND metadata->'entities' ? $1
ORDER BY id
FOR UPDATE
"""
FACTS_SQL = f"""
SELECT id, entity = $1 AS own FROM {_FACTS} WHERE entity = $1 OR value_entity = $1
"""
CLEAR_REPLACED_SQL = f"UPDATE {_FACTS} SET replaced_by = NULL WHERE replaced_by = ANY($1)"
DELETE_FACTS_SQL = f"DELETE FROM {_FACTS} WHERE id = ANY($1)"
DELETE_SQL = f"DELETE FROM {_ENTITIES} WHERE slug = $1"


class Collision(ValueError):
    """A slug, name, or alias is already held by another entity."""

    def __init__(self, holder: str, value: str) -> None:
        super().__init__(f"{value!r} is already held by entity {holder!r}; reuse it")
        self.holder = holder
        self.value = value


class PlanChanged(ValueError):
    """The delete plan under the lock differs from the one the caller expected."""

    def __init__(self, plan: dict[str, Any]) -> None:
        super().__init__("the delete plan changed since the preview; preview it again")
        self.plan = plan


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
    changes_description = "description" in item and item["description"] != current["description"]
    if changes_description and base is None:
        raise ValueError(f"editing the description of {slug!r} needs base_updated_at")
    if changes_description and base != datetime.fromtimestamp(
        current["updated_at"], tz=timezone.utc
    ):
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


async def _locked_registry(conn: Any) -> dict[str, dict[str, Any]]:
    """Take the registry lock, then read every row."""
    await entities.lock_registry(conn)
    return {row["slug"]: dict(row) for row in await conn.fetch(LIST_SQL)}


async def list_entities() -> list[dict[str, Any]]:
    """Every entity ordered by slug."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        return [dict(row) for row in await conn.fetch(LIST_SQL)]


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
                    raise DryRun(written)
    except DryRun as stop:
        return stop.result
    return written


async def delete(
    slug: str, author: str, dry_run: bool = False, expect: dict[str, int] | None = None
) -> dict[str, Any]:
    """Delete an entity: unlink it from every agent note, delete its facts, drop its row.

    The plan is `{note_ids, facts, incoming}`: the notes holding the slug, the entity's own
    facts, and other entities' facts pointing at it. A dry run returns it and writes nothing.
    `expect` is the `{notes, facts, incoming}` counts of the preview; a plan that differs
    under the lock raises PlanChanged. Raises ValueError for an unknown slug.
    """
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction(isolation="read_committed"):
            await _locked_registry(conn)
            if await entities.get(conn, slug) is None:
                raise ValueError(f"no entity {slug!r}")
            notes = await conn.fetch(LINKED_NOTES_SQL, slug)
            facts = await conn.fetch(FACTS_SQL, slug)
            plan = {
                "note_ids": [row["id"] for row in notes],
                "facts": sum(row["own"] for row in facts),
                "incoming": sum(not row["own"] for row in facts),
            }
            counts = {"notes": len(notes), "facts": plan["facts"], "incoming": plan["incoming"]}
            if dry_run:
                return plan
            if expect is not None and expect != counts:
                raise PlanChanged(plan)
            now = _now()
            for row in notes:
                before = metadata_dict(row["metadata"])
                after = [s for s in before["entities"] if s != slug]
                await entities.set_note_entities(conn, row["id"], before, after, author, now)
            fact_ids = [row["id"] for row in facts]
            await conn.execute(CLEAR_REPLACED_SQL, fact_ids)
            await conn.execute(DELETE_FACTS_SQL, fact_ids)
            await conn.execute(DELETE_SQL, slug)
    return plan
