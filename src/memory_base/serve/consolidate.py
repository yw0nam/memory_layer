"""Read-only discovery of note groups that may state the same thing, for an agent to judge."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.retrieval.search import metadata_dict

PROCEDURE_VERSION = "1"
DEFAULT_THRESHOLD = 0.72
DEFAULT_NEIGHBORS = 5
DEFAULT_MAX_GROUP = 6
DEFAULT_MAX_GROUP_CHARS = 12000
DEFAULT_LIMIT = 200
# The threshold lies in (MIN_THRESHOLD, MAX_THRESHOLD]; every other bound is inclusive.
MIN_THRESHOLD = 0.0
MAX_THRESHOLD = 1.0
MIN_NEIGHBORS = 1
MAX_NEIGHBORS = 50
MIN_MAX_GROUP = 2
MAX_MAX_GROUP = 20
MIN_MAX_GROUP_CHARS = 500
MIN_LIMIT = 1
MAX_LIMIT = 1000

# Disabling index scans keeps the planner off the approximate HNSW index.
EXACT_SEARCH_SETTINGS = (
    "SET LOCAL enable_indexscan = off",
    "SET LOCAL enable_bitmapscan = off",
)

PAIRS_SQL = f"""
WITH active AS MATERIALIZED (
  SELECT id, embedding FROM "{PG_SCHEMA}".memory_chunks
  WHERE source_type = 'agent_note' AND archived_at IS NULL AND namespace = $1
)
SELECT a.id AS a_id, near.id AS b_id, 1 - near.distance AS score
FROM active AS a
CROSS JOIN LATERAL (
  SELECT b.id, b.embedding <=> a.embedding AS distance
  FROM active AS b
  WHERE b.id <> a.id
  ORDER BY distance, b.id
  LIMIT $2
) AS near
WHERE 1 - near.distance >= $3
"""

NOTES_SQL = f"""
SELECT id, chunk_kind AS kind, ts_last_active, occurred_at, content_raw AS text, metadata
FROM "{PG_SCHEMA}".memory_chunks
WHERE source_type = 'agent_note' AND archived_at IS NULL AND namespace = $1
"""


@dataclass(frozen=True)
class Note:
    id: str
    kind: str
    author: str | None
    saved: float
    occurred_at: float | None
    tags: tuple[str, ...]
    similar_ack: tuple[str, ...]
    supersedes: str | None
    text: str


@dataclass(frozen=True)
class Pair:
    a: str
    b: str
    score: float


@dataclass(frozen=True)
class Group:
    members: tuple[str, ...]
    min_score: float
    max_score: float


@dataclass(frozen=True)
class Deferred:
    id: str
    reason: str


def _note(row: Any) -> Note:
    metadata = metadata_dict(row["metadata"])
    return Note(
        id=row["id"],
        kind=row["kind"],
        author=metadata.get("author"),
        saved=row["ts_last_active"],
        occurred_at=row["occurred_at"],
        tags=tuple(metadata.get("tags") or ()),
        similar_ack=tuple(metadata.get("similar_ack") or ()),
        supersedes=metadata.get("supersedes"),
        text=row["text"],
    )


async def candidate_pairs(
    conn: Any, namespace: str, threshold: float, neighbors: int
) -> tuple[list[Pair], dict[str, Note]]:
    """Exact nearest-neighbour pairs among one namespace's active agent notes.

    Returns the pairs at or above `threshold` and every active note in the namespace,
    read from one repeatable-read snapshot.
    """
    async with conn.transaction(isolation="repeatable_read", readonly=True):
        for statement in EXACT_SEARCH_SETTINGS:
            await conn.execute(statement)
        rows = await conn.fetch(PAIRS_SQL, namespace, neighbors, threshold)
        notes = {row["id"]: _note(row) for row in await conn.fetch(NOTES_SQL, namespace)}
    best: dict[tuple[str, str], float] = {}
    for row in rows:
        ends = (min(row["a_id"], row["b_id"]), max(row["a_id"], row["b_id"]))
        best[ends] = max(best.get(ends, row["score"]), row["score"])
    pairs = [Pair(a, b, score) for (a, b), score in sorted(best.items())]
    return pairs, notes


async def read_snapshots(
    namespaces: list[str], threshold: float, neighbors: int
) -> dict[str, tuple[list[Pair], dict[str, Note]]]:
    """Each namespace's pairs and notes, on one connection released before the caller builds."""
    async with db.acquire() as conn:
        return {
            namespace: await candidate_pairs(conn, namespace, threshold, neighbors)
            for namespace in namespaces
        }


def _acknowledged(pair: Pair, notes: dict[str, Note]) -> bool:
    return pair.b in notes[pair.a].similar_ack or pair.a in notes[pair.b].similar_ack


def acknowledged_pairs(pairs: Iterable[Pair], notes: dict[str, Note]) -> int:
    """How many pairs a writer acknowledged as distinct facts through `similar_ack`."""
    return sum(_acknowledged(pair, notes) for pair in pairs)


def build_groups(
    pairs: Iterable[Pair],
    notes: dict[str, Note],
    threshold: float,
    max_group: int,
    max_group_chars: int,
) -> tuple[list[Group], list[Deferred]]:
    """Pack unacknowledged edges into cliques greedily and deterministically.

    Returns the groups and every note with an edge that ended in no group.
    """
    edges: dict[frozenset[str], float] = {}
    for pair in pairs:
        if pair.score >= threshold and pair.a != pair.b and not _acknowledged(pair, notes):
            key = frozenset((pair.a, pair.b))
            edges[key] = max(edges.get(key, pair.score), pair.score)
    neighbours: dict[str, set[str]] = {}
    for key in edges:
        a, b = sorted(key)
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)
    ordered = sorted((-score, *sorted(key)) for key, score in edges.items())

    assigned: set[str] = set()
    groups: list[Group] = []
    reasons: dict[str, str] = {}
    for _, a, b in ordered:
        if a in assigned or b in assigned:
            continue
        if len(notes[a].text) + len(notes[b].text) > max_group_chars:
            reasons.setdefault(a, "over max_group_chars")
            reasons.setdefault(b, "over max_group_chars")
            continue
        members = [a, b]
        chars = len(notes[a].text) + len(notes[b].text)
        skipped: set[str] = set()
        while True:
            candidates = sorted(
                (
                    (min(edges[frozenset((c, m))] for m in members), c)
                    for c in set.intersection(*(neighbours[m] for m in members))
                    if c not in assigned and c not in skipped
                ),
                key=lambda item: (-item[0], item[1]),
            )
            added = False
            for _, candidate in candidates:
                if len(members) >= max_group:
                    reasons.setdefault(candidate, "over max_group")
                    skipped.add(candidate)
                elif chars + len(notes[candidate].text) > max_group_chars:
                    reasons.setdefault(candidate, "over max_group_chars")
                    skipped.add(candidate)
                else:
                    members.append(candidate)
                    chars += len(notes[candidate].text)
                    added = True
                    break
            if not added:
                break
        assigned.update(members)
        groups.append(_group(members, edges))

    groups.sort(key=lambda g: (-g.max_score, g.members[0]))
    deferred = [
        Deferred(note_id, reasons.get(note_id, "no clique"))
        for note_id in sorted(neighbours)
        if note_id not in assigned
    ]
    return groups, deferred


def _group(members: list[str], edges: dict[frozenset[str], float]) -> Group:
    ordered = tuple(sorted(members))
    scores = [edges[frozenset((a, b))] for i, a in enumerate(ordered) for b in ordered[i + 1 :]]
    return Group(ordered, min(scores), max(scores))


def group_key(namespace: str, members: Iterable[Note]) -> str:
    """sha256 over membership and every prompt-visible field of each member."""
    payload = {
        "v": PROCEDURE_VERSION,
        "namespace": namespace,
        "members": [
            [
                note.id,
                hashlib.sha256(note.text.encode()).hexdigest(),
                note.kind,
                note.author,
                note.saved,
                note.occurred_at,
                sorted(note.tags),
                note.supersedes,
            ]
            for note in sorted(members, key=lambda note: note.id)
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _member(note: Note) -> dict[str, Any]:
    return {
        "id": note.id,
        "kind": note.kind,
        "author": note.author,
        "saved": _day(note.saved),
        "occurred_at": None
        if note.occurred_at is None
        else datetime.fromtimestamp(note.occurred_at, tz=timezone.utc).isoformat(),
        "tags": list(note.tags),
        "supersedes": note.supersedes,
        "text": note.text,
    }


def namespace_report(
    namespace: str,
    pairs: list[Pair],
    notes: dict[str, Note],
    *,
    threshold: float,
    max_group: int,
    max_group_chars: int,
    limit: int,
) -> dict[str, Any]:
    """One namespace's section of the response: groups up to `limit`, and deferrals."""
    groups, deferred = build_groups(pairs, notes, threshold, max_group, max_group_chars)
    return {
        "active_notes": len(notes),
        "pairs": len(pairs),
        "acknowledged": acknowledged_pairs(pairs, notes),
        "groups": [
            {
                "key": group_key(namespace, (notes[i] for i in g.members)),
                "min_score": g.min_score,
                "max_score": g.max_score,
                "members": [
                    _member(notes[i]) for i in sorted(g.members, key=lambda i: (notes[i].saved, i))
                ],
            }
            for g in groups[:limit]
        ],
        "deferred": [{"id": d.id, "reason": d.reason} for d in deferred],
        "truncated": len(groups) > limit,
    }
