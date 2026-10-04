"""Consolidation verdicts: validate an agent's judgement of an issued group, apply it, undo it.

The server judges no content. It accepts a verdict only for a group it would issue now,
checks it deterministically, and applies it in one transaction per verdict under a
per-namespace advisory lock. Every applied verdict is recorded in `consolidation_actions`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import asyncpg

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA, VllmEmbedder, embed_text
from memory_base.core.schema import ensure_schema_once
from memory_base.core.secrets import find_secret
from memory_base.serve import consolidate
from memory_base.serve.consolidate import Note, Pair, build_groups, group_entry, group_key, iso
from memory_base.serve.notes import (
    INSERT_NOTE_SQL,
    LINEAGE_FIELDS,
    build_note_row,
    insert_note_args,
    note_id,
)

DEFAULT_MAX_ACTIONS = 20
MIN_MAX_ACTIONS = 1
MAX_MAX_ACTIONS = 500
MAX_VERDICTS = 200
MAX_RUN_ID_CHARS = 100
MAX_IDEMPOTENCY_KEY_CHARS = 200
MAX_REASON_CHARS = 1000
DEFAULT_ACTIONS_LIMIT = 50
MAX_ACTIONS_LIMIT = 500
MAX_FAILURE_CHARS = 300

ACTIONS = ("keep", "retire", "merge")
BATCH_FIELDS = frozenset(
    {
        "run_id",
        "author",
        "model",
        "dry_run",
        "threshold",
        "neighbors",
        "max_group",
        "max_group_chars",
        "max_actions",
        "verdicts",
    }
)
VERDICT_FIELDS = frozenset(
    {
        "namespace",
        "group_key",
        "idempotency_key",
        "member_ids",
        "action",
        "retire_ids",
        "merged_text",
        "reason",
    }
)
UNDO_FIELDS = frozenset({"action_id", "author"})
IDEMPOTENCY_CONSTRAINT = "consolidation_actions_idempotency_key_key"
REUSED_KEY = "idempotency key reused with a different payload"
JUDGED = "group already judged"

_TABLE = f'"{PG_SCHEMA}".consolidation_actions'
_NOTES = f'"{PG_SCHEMA}".memory_chunks'

NAMESPACE_LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtextextended('consolidate:' || $1, 0))"
LOCK_ROWS_SQL = f"""
SELECT id, archived_at, metadata FROM {_NOTES}
WHERE id = ANY($1::text[])
ORDER BY id
FOR UPDATE
"""
RECORDED_SQL = f"SELECT payload_hash, result FROM {_TABLE} WHERE idempotency_key = $1"
KEY_JUDGED_SQL = f"SELECT EXISTS(SELECT 1 FROM {_TABLE} WHERE group_key = $1)"
UNDONE_MEMBERS_SQL = f"""
SELECT EXISTS(
  SELECT 1 FROM {_TABLE}
  WHERE namespace = $1 AND undone_at IS NOT NULL AND member_ids = $2::text[]
)
"""
USED_ACTIONS_SQL = f"""
SELECT count(*) FROM {_TABLE}
WHERE run_id = $1 AND namespace = $2 AND action IN ('retire', 'merge')
"""
EXISTING_SQL = f"""
SELECT id, source_type, chunk_kind, content_raw, archived_at FROM {_NOTES} WHERE id = $1
"""
NEXT_ACTION_ID_SQL = f"SELECT nextval(pg_get_serial_sequence('{_TABLE}', 'id'))"
ARCHIVE_MEMBERS_SQL = f"""
UPDATE {_NOTES}
SET archived_at = $2,
    metadata = metadata || jsonb_build_object(
      'archived_by', $3::text, 'consolidated_into', $4::jsonb)
WHERE id = ANY($1::text[])
"""
INSERT_ACTION_SQL = f"""
INSERT INTO {_TABLE}
  (id, idempotency_key, payload_hash, run_id, namespace, action, group_key, member_ids,
   archived_ids, survivor_ids, replacement_id, replacement_created, prior, applied_at,
   author, model, reason, result)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8::text[], $9::text[], $10::text[], $11, $12,
        $13::jsonb, $14, $15, $16, $17, $18::jsonb)
"""
ACTION_NAMESPACE_SQL = f"SELECT namespace FROM {_TABLE} WHERE id = $1"
ACTION_FOR_UPDATE_SQL = f"SELECT * FROM {_TABLE} WHERE id = $1 FOR UPDATE"
# Walks supersedes links from the replacement through archived notes to any active one.
ACTIVE_SUCCESSOR_SQL = f"""
WITH RECURSIVE successors(id, archived_at) AS (
  SELECT id, archived_at FROM {_NOTES}
  WHERE source_type = 'agent_note' AND metadata->>'supersedes' = $1
  UNION
  SELECT note.id, note.archived_at
  FROM {_NOTES} AS note
  JOIN successors ON note.metadata->>'supersedes' = successors.id
  WHERE note.source_type = 'agent_note'
)
SELECT id FROM successors WHERE archived_at IS NULL ORDER BY id LIMIT 1
"""
LATER_ACTION_SQL = f"""
SELECT EXISTS(
  SELECT 1 FROM {_TABLE}
  WHERE undone_at IS NULL AND action IN ('retire', 'merge') AND $1 = ANY(member_ids)
)
"""
RESTORE_PRIOR_SQL = f"UPDATE {_NOTES} SET archived_at = NULL, metadata = $2::jsonb WHERE id = $1"
ARCHIVE_REPLACEMENT_SQL = f"""
UPDATE {_NOTES}
SET archived_at = $2,
    metadata = metadata || jsonb_build_object('archived_by', $3::text, 'undone_action', $4::bigint)
WHERE id = $1
"""
MARK_UNDONE_SQL = f"""
UPDATE {_TABLE} SET undone_at = $2, undone_by = $3, undo_result = $4::jsonb WHERE id = $1
"""
LIST_ACTIONS_SQL = f"""
SELECT * FROM {_TABLE}
WHERE ($1::text IS NULL OR namespace = $1)
  AND ($2::text IS NULL OR run_id = $2)
  AND ($3::text IS NULL OR $3 = ANY(member_ids) OR $3 = ANY(archived_ids)
       OR $3 = ANY(survivor_ids) OR replacement_id = $3)
ORDER BY id DESC
LIMIT $4
"""
LINEAGE_NOTES_SQL = f"""
SELECT id, chunk_kind AS kind, content_raw AS text, metadata, ts_last_active, occurred_at,
       archived_at
FROM {_NOTES}
WHERE source_type = 'agent_note' AND id = ANY($1::text[])
"""


class RequestError(ValueError):
    """A verdict or undo request body breaks the schema; REST maps it to 400."""


class Rollback(Exception):
    """Ends one verdict's transaction without writing; carries the verdict's result."""

    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__(result.get("reason"))
        self.result = result


class UndoNotFound(LookupError):
    """No consolidation action has the requested id."""


class UndoRefused(ValueError):
    """An action cannot be undone as recorded; nothing was changed."""


@dataclass(frozen=True)
class Params:
    threshold: float
    neighbors: int
    max_group: int
    max_group_chars: int


@dataclass(frozen=True)
class Verdict:
    namespace: str
    group_key: str
    idempotency_key: str
    member_ids: tuple[str, ...]
    action: str
    retire_ids: tuple[str, ...] | None
    merged_text: str | None
    reason: str
    item: dict[str, Any] = field(compare=False, hash=False)


@dataclass(frozen=True)
class Batch:
    run_id: str
    author: str
    model: str | None
    dry_run: bool
    params: Params
    max_actions: int
    verdicts: tuple[Verdict, ...]


@dataclass(frozen=True)
class Recorded:
    payload_hash: str
    result: dict[str, Any]


@dataclass(frozen=True)
class Existing:
    """The row already stored under a merge's replacement id."""

    id: str
    source_type: str
    kind: str
    text: str
    archived: bool


@dataclass(frozen=True)
class State:
    """Everything a plan reads: the recorded verdict, the cache, the current groups, the cap."""

    recorded: Recorded | None
    cached: str | None
    pairs: list[Pair]
    notes: dict[str, Note]
    used_actions: int
    replacement: Existing | None


@dataclass(frozen=True)
class Plan:
    action: str
    archived_ids: tuple[str, ...]
    survivor_ids: tuple[str, ...]
    replacement_id: str | None
    replacement_created: bool
    reason: str | None


def _jsonb(value: Any) -> Any:
    """A jsonb column value as Python: asyncpg returns it as text, fakes may pass objects."""
    return json.loads(value) if isinstance(value, str) else value


def _metadata(value: Any) -> dict[str, Any]:
    return _jsonb(value) or {}


# ---- request schema ---------------------------------------------------------


def _text(value: Any, name: str, max_chars: int | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RequestError(f"{name} must be a non-blank string")
    if max_chars is not None and len(value) > max_chars:
        raise RequestError(f"{name} must be at most {max_chars} characters")
    return value


def _int(body: dict[str, Any], name: str, default: int, low: int, high: float) -> int:
    value = body.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        bound = f"between {low} and {high}" if math.isfinite(high) else f"of at least {low}"
        raise RequestError(f"{name} must be an integer {bound}")
    return value


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        raise RequestError(f"{name} must be a list of strings")
    return tuple(value)


def _unknown(mapping: dict[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise RequestError(f"unknown field in {where}: {', '.join(unknown)}")


def _params(body: dict[str, Any]) -> Params:
    threshold = body.get("threshold", consolidate.DEFAULT_THRESHOLD)
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not consolidate.MIN_THRESHOLD < threshold <= consolidate.MAX_THRESHOLD
    ):
        raise RequestError(
            f"threshold must be a number in ({consolidate.MIN_THRESHOLD:g}, "
            f"{consolidate.MAX_THRESHOLD:g}]"
        )
    return Params(
        float(threshold),
        _int(
            body,
            "neighbors",
            consolidate.DEFAULT_NEIGHBORS,
            consolidate.MIN_NEIGHBORS,
            consolidate.MAX_NEIGHBORS,
        ),
        _int(
            body,
            "max_group",
            consolidate.DEFAULT_MAX_GROUP,
            consolidate.MIN_MAX_GROUP,
            consolidate.MAX_MAX_GROUP,
        ),
        _int(
            body,
            "max_group_chars",
            consolidate.DEFAULT_MAX_GROUP_CHARS,
            consolidate.MIN_MAX_GROUP_CHARS,
            math.inf,
        ),
    )


def _verdict(item: Any, index: int, registered: set[str]) -> Verdict:
    where = f"verdicts[{index}]"
    if not isinstance(item, dict):
        raise RequestError(f"{where} must be an object")
    _unknown(item, VERDICT_FIELDS, where)
    namespace = _text(item.get("namespace"), f"{where}.namespace")
    if namespace not in registered:
        raise RequestError(f"unregistered namespace: {namespace}")
    member_ids = _strings(item.get("member_ids"), f"{where}.member_ids")
    if not consolidate.MIN_MAX_GROUP <= len(member_ids) <= consolidate.MAX_MAX_GROUP:
        raise RequestError(
            f"{where}.member_ids must hold {consolidate.MIN_MAX_GROUP} to "
            f"{consolidate.MAX_MAX_GROUP} ids"
        )
    action = item.get("action")
    if action not in ACTIONS:
        raise RequestError(f"{where}.action must be one of {ACTIONS}")
    retire_ids = None
    if "retire_ids" in item:
        if action != "retire":
            raise RequestError(f"{where}.retire_ids is allowed only on a retire")
        retire_ids = _strings(item["retire_ids"], f"{where}.retire_ids")
    merged_text = None
    if "merged_text" in item:
        if action != "merge":
            raise RequestError(f"{where}.merged_text is allowed only on a merge")
        if not isinstance(item["merged_text"], str):
            raise RequestError(f"{where}.merged_text must be a string")
        merged_text = item["merged_text"]
    return Verdict(
        namespace=namespace,
        group_key=_text(item.get("group_key"), f"{where}.group_key"),
        idempotency_key=_text(
            item.get("idempotency_key"), f"{where}.idempotency_key", MAX_IDEMPOTENCY_KEY_CHARS
        ),
        member_ids=member_ids,
        action=action,
        retire_ids=retire_ids,
        merged_text=merged_text,
        reason=_text(item.get("reason"), f"{where}.reason", MAX_REASON_CHARS),
        item=dict(item),
    )


def parse_batch(body: Any, registered: set[str]) -> Batch:
    """Validate a verdict request strictly; any violation refuses the whole request."""
    if not isinstance(body, dict):
        raise RequestError("JSON body must be an object")
    _unknown(body, BATCH_FIELDS, "the request")
    model = body.get("model")
    if model is not None and not isinstance(model, str):
        raise RequestError("model must be a string or null")
    dry_run = body.get("dry_run", False)
    if not isinstance(dry_run, bool):
        raise RequestError("dry_run must be a boolean")
    items = body.get("verdicts")
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_VERDICTS:
        raise RequestError(f"verdicts must be a list of 1 to {MAX_VERDICTS} objects")
    return Batch(
        run_id=_text(body.get("run_id"), "run_id", MAX_RUN_ID_CHARS),
        author=_text(body.get("author"), "author"),
        model=model,
        dry_run=dry_run,
        params=_params(body),
        max_actions=_int(
            body, "max_actions", DEFAULT_MAX_ACTIONS, MIN_MAX_ACTIONS, MAX_MAX_ACTIONS
        ),
        verdicts=tuple(_verdict(item, i, registered) for i, item in enumerate(items)),
    )


def parse_undo(body: Any) -> tuple[int, str]:
    """Validate an undo request: `{"action_id": int, "author": str}`."""
    if not isinstance(body, dict):
        raise RequestError("JSON body must be an object")
    _unknown(body, UNDO_FIELDS, "the request")
    action_id = body.get("action_id")
    if isinstance(action_id, bool) or not isinstance(action_id, int) or action_id < 1:
        raise RequestError("action_id must be a positive integer")
    return action_id, _text(body.get("author"), "author")


def payload_hash(verdict: Verdict, batch: Batch) -> str:
    """sha256 of the verdict as submitted plus the run, author, model, and group params."""
    payload = {
        "verdict": verdict.item,
        "run_id": batch.run_id,
        "author": batch.author,
        "model": batch.model,
        "params": asdict(batch.params),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---- token check ------------------------------------------------------------

NUMBER_RE = re.compile(r"(?<![\w.])[+-]?\d+(?:[.,:/\-]\d+)*(?:[eE][+-]?\d+)?(?!\w)")
BACKTICK_RE = re.compile(r"`([^`]+)`")
WORD_RE = re.compile(r"(?<![\w'\-])[^\W\d_][\w'\-]*")
NAME_RE = re.compile(r"[A-Z][\w'\-]*")
LIST_MARKER_RE = re.compile(r"[ \t]*(?:[-*+•]|\d+[.)])")
NUMBERED_MARKER_RE = re.compile(r"^[ \t]*(\d+)[.)](?=\s)", re.MULTILINE)


def _starts_sentence(text: str, start: int) -> bool:
    before = text[:start]
    stripped = before.rstrip()
    gap = before[len(stripped) :]
    if not stripped or "\n" in gap:
        return True
    if not gap:
        return False
    return stripped[-1] in ".!?:" or bool(
        LIST_MARKER_RE.fullmatch(stripped[stripped.rfind("\n") + 1 :])
    )


def tokens(text: str, *, include_sentence_starts: bool = False) -> set[str]:
    """Numbers and dates, backticked spans, and names: the facts a merge must carry over.

    A name is a capitalized word other than `I` that does not start a sentence, or any
    word with two or more capitals; `include_sentence_starts` also counts a capitalized word that
    starts a sentence. A numbered-list marker is not a number.
    """
    markers = {m.start(1) for m in NUMBERED_MARKER_RE.finditer(text)}
    found = {m.group() for m in NUMBER_RE.finditer(text) if m.start() not in markers}
    found.update(BACKTICK_RE.findall(text))
    for match in WORD_RE.finditer(text):
        word = match.group()
        if word == "I":
            continue
        if sum(c.isupper() for c in word) >= 2 or (
            NAME_RE.fullmatch(word)
            and (include_sentence_starts or not _starts_sentence(text, match.start()))
        ):
            found.add(word)
    return found


def token_check(member_texts: list[str], merged: str) -> str | None:
    """Why the merged text fails the two-direction token check, or None when it passes.

    A token counts as dropped or added only when no sentence-initial reading of the other
    side carries it, so a name that moves to or from a sentence start passes.
    """
    strict_members = set().union(*(tokens(text) for text in member_texts))
    lenient_members = set().union(
        *(tokens(text, include_sentence_starts=True) for text in member_texts)
    )
    strict_merged = tokens(merged)
    lenient_merged = tokens(merged, include_sentence_starts=True)
    problems = []
    if missing := sorted(strict_members - lenient_merged):
        problems.append(f"drops {', '.join(missing)}")
    if added := sorted(strict_merged - lenient_members):
        problems.append(f"adds {', '.join(added)}")
    return f"merged_text {' and '.join(problems)}" if problems else None


def normalize_text(text: str) -> str:
    """Stripped, with every whitespace run collapsed to one space."""
    return " ".join(text.split())


# ---- plan -------------------------------------------------------------------


def _result(
    verdict: Verdict,
    status: str,
    reason: str | None = None,
    *,
    plan: Plan | None = None,
    action_id: int | None = None,
    current_groups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "group_key": verdict.group_key,
        "status": status,
        "reason": reason,
        "action_id": action_id,
        "archived_ids": [] if plan is None else list(plan.archived_ids),
        "survivor_ids": [] if plan is None else list(plan.survivor_ids),
        "replacement_id": None if plan is None else plan.replacement_id,
        "current_groups": current_groups,
    }


def _current_groups(verdict: Verdict, batch: Batch, state: State) -> list[dict[str, Any]]:
    p = batch.params
    groups, _ = build_groups(state.pairs, state.notes, p.threshold, p.max_group, p.max_group_chars)
    submitted = set(verdict.member_ids)
    return [
        group_entry(verdict.namespace, g, state.notes) for g in groups if submitted & set(g.members)
    ]


def replacement_row(
    verdict: Verdict, batch: Batch, notes: dict[str, Note], now: float
) -> dict[str, Any]:
    """The merged note: the members' kind, their tags' union, and their latest event date."""
    members = [notes[i] for i in sorted(verdict.member_ids)]
    occurred = [n.occurred_at for n in members if n.occurred_at is not None]
    return build_note_row(
        verdict.merged_text or "",
        members[0].kind,
        sorted({tag for n in members for tag in n.tags}),
        now,
        verdict.namespace,
        batch.author,
        occurred_at=max(occurred) if occurred else None,
    )


def _plan_retire(verdict: Verdict, members: tuple[str, ...]) -> Plan | dict[str, Any]:
    retire = verdict.retire_ids or ()
    problem = None
    if not retire:
        problem = "retire needs a non-empty retire_ids"
    elif len(set(retire)) != len(retire):
        problem = "retire_ids repeats an id"
    elif outside := sorted(set(retire) - set(members)):
        problem = f"retire_ids outside the group: {', '.join(outside)}"
    elif set(retire) == set(members):
        problem = "retire must leave at least one member active"
    if problem is not None:
        return _result(verdict, "rejected", problem)
    survivors = tuple(i for i in members if i not in retire)
    return Plan("retire", tuple(sorted(retire)), survivors, None, False, None)


def _plan_merge(
    verdict: Verdict, batch: Batch, state: State, members: tuple[str, ...]
) -> Plan | dict[str, Any]:
    notes = state.notes
    if len({notes[i].kind for i in members}) != 1:
        return _result(verdict, "rejected", "members differ in kind; merge needs one kind")
    text = verdict.merged_text or ""
    try:
        row = replacement_row(verdict, batch, notes, 0.0)
    except ValueError as exc:
        return _result(verdict, "rejected", f"merged_text: {exc}")
    secret = find_secret(text)
    if secret is not None:
        return _result(verdict, "rejected", f"merged_text contains a credential ({secret})")
    mismatch = token_check([notes[i].text for i in members], text)
    if mismatch is not None:
        return _result(verdict, "rejected", mismatch)
    normalized = normalize_text(text)
    same = next((i for i in members if normalize_text(notes[i].text) == normalized), None)
    if same is not None:
        return Plan(
            "retire",
            tuple(i for i in members if i != same),
            (same,),
            None,
            False,
            f"merged_text restates {same}; the other members are retired into it",
        )
    replacement_id = row["id"]
    existing = state.replacement
    if existing is None:
        return Plan("merge", members, (replacement_id,), replacement_id, True, None)
    if existing.archived:
        return _result(
            verdict,
            "rejected",
            f"merged_text is identical to archived note {replacement_id}; restore it instead",
        )
    if (
        existing.source_type != "agent_note"
        or existing.kind != notes[members[0]].kind
        or existing.text != row["raw"]
        or replacement_id in members
    ):
        return _result(
            verdict, "rejected", f"note {replacement_id} exists with a different kind or text"
        )
    return Plan("merge", members, (replacement_id,), replacement_id, False, None)


def plan_verdict(verdict: Verdict, batch: Batch, state: State) -> Plan | dict[str, Any]:
    """The change a verdict would make over the current rows, or its terminal result."""
    if state.recorded is not None:
        if state.recorded.payload_hash == payload_hash(verdict, batch):
            return {**state.recorded.result, "status": "duplicate"}
        return _result(verdict, "rejected", REUSED_KEY)
    if state.cached is not None:
        return _result(verdict, "cached", state.cached)
    p = batch.params
    groups, _ = build_groups(state.pairs, state.notes, p.threshold, p.max_group, p.max_group_chars)
    issued = next(
        (
            g
            for g in groups
            if group_key(verdict.namespace, (state.notes[i] for i in g.members))
            == verdict.group_key
        ),
        None,
    )
    members = tuple(sorted(verdict.member_ids))
    if issued is None or issued.members != members:
        reason = (
            "group_key matches no current group"
            if issued is None
            else "member_ids differ from the issued group"
        )
        return _result(
            verdict, "stale", reason, current_groups=_current_groups(verdict, batch, state)
        )
    if verdict.action != "keep" and state.used_actions >= batch.max_actions:
        return _result(verdict, "rejected", "action cap reached")
    if verdict.action == "keep":
        return Plan("keep", (), members, None, False, None)
    if verdict.action == "retire":
        return _plan_retire(verdict, members)
    return _plan_merge(verdict, batch, state, members)


# ---- read, lock, write ------------------------------------------------------


async def _exact_search(conn: Any) -> None:
    for statement in consolidate.EXACT_SEARCH_SETTINGS:
        await conn.execute(statement)


def _merged_id(verdict: Verdict) -> str | None:
    text = verdict.merged_text or ""
    return note_id(verdict.namespace, text) if verdict.action == "merge" and text.strip() else None


async def load_state(conn: Any, verdict: Verdict, batch: Batch) -> State:
    """Read what `plan_verdict` needs, on the caller's transaction."""
    row = await conn.fetchrow(RECORDED_SQL, verdict.idempotency_key)
    recorded = None if row is None else Recorded(row["payload_hash"], _jsonb(row["result"]))
    cached = None
    if await conn.fetchval(KEY_JUDGED_SQL, verdict.group_key):
        cached = JUDGED
    elif await conn.fetchval(UNDONE_MEMBERS_SQL, verdict.namespace, sorted(verdict.member_ids)):
        cached = "group matches the members of an undone action"
    if recorded is not None or cached is not None:
        return State(recorded, cached, [], {}, 0, None)
    pairs, notes = await consolidate.read_pairs(
        conn, verdict.namespace, batch.params.threshold, batch.params.neighbors
    )
    used = await conn.fetchval(USED_ACTIONS_SQL, batch.run_id, verdict.namespace)
    replacement = None
    merged_id = _merged_id(verdict)
    if merged_id is not None:
        found = await conn.fetchrow(EXISTING_SQL, merged_id)
        if found is not None:
            replacement = Existing(
                found["id"],
                found["source_type"],
                found["chunk_kind"],
                found["content_raw"],
                found["archived_at"] is not None,
            )
    return State(None, None, pairs, notes, used, replacement)


async def lock_rows(conn: Any, ids: list[str]) -> dict[str, Any]:
    """Lock the existing rows among `ids` in id order; returns them by id."""
    return {row["id"]: row for row in await conn.fetch(LOCK_ROWS_SQL, sorted(ids))}


async def apply_plan(
    conn: Any,
    verdict: Verdict,
    batch: Batch,
    plan: Plan,
    state: State,
    rows: dict[str, Any],
    embedding: str | None,
) -> dict[str, Any]:
    """Write a plan on the caller's transaction: replacement, archive, and action row.

    `embedding` is the replacement's embedding, required when the plan creates one.
    Raises Rollback when the replacement id was taken concurrently.
    """
    now = time.time()
    action_id = await conn.fetchval(NEXT_ACTION_ID_SQL)
    applied = _result(verdict, "applied", plan.reason, plan=plan, action_id=action_id)
    members = sorted(verdict.member_ids)
    if plan.replacement_created:
        if embedding is None:
            raise ValueError("a created replacement needs its embedding")
        row = replacement_row(verdict, batch, state.notes, now)
        row["metadata"]["merged_from"] = members
        row["metadata"]["merged_dates"] = {
            i: {"saved": iso(state.notes[i].saved), "occurred_at": iso(state.notes[i].occurred_at)}
            for i in members
        }
        row["metadata"]["consolidation_action"] = action_id
        inserted = await conn.fetchval(
            INSERT_NOTE_SQL + " RETURNING id",
            *insert_note_args(row, embedding, verdict.namespace),
        )
        if inserted is None:
            raise Rollback(_result(verdict, "stale", "replacement appeared concurrently"))
    if plan.archived_ids:
        await conn.execute(
            ARCHIVE_MEMBERS_SQL,
            list(plan.archived_ids),
            now,
            batch.author,
            json.dumps(list(plan.survivor_ids)),
        )
    prior = {i: _metadata(rows[i]["metadata"]) for i in plan.archived_ids}
    await conn.execute(
        INSERT_ACTION_SQL,
        action_id,
        verdict.idempotency_key,
        payload_hash(verdict, batch),
        batch.run_id,
        verdict.namespace,
        plan.action,
        verdict.group_key,
        members,
        list(plan.archived_ids),
        list(plan.survivor_ids),
        plan.replacement_id,
        plan.replacement_created,
        json.dumps(prior, ensure_ascii=False),
        now,
        batch.author,
        batch.model,
        verdict.reason,
        json.dumps(applied, ensure_ascii=False),
    )
    return applied


async def _idempotency_conflict(conn: Any, verdict: Verdict, batch: Batch) -> dict[str, Any]:
    row = await conn.fetchrow(RECORDED_SQL, verdict.idempotency_key)
    if row is not None and row["payload_hash"] == payload_hash(verdict, batch):
        return {**_jsonb(row["result"]), "status": "duplicate"}
    return _result(verdict, "rejected", REUSED_KEY)


async def process_verdict(verdict: Verdict, batch: Batch) -> dict[str, Any]:
    """Plan on a snapshot, then apply under the namespace lock after planning again."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction(isolation="repeatable_read", readonly=True):
            await _exact_search(conn)
            state = await load_state(conn, verdict, batch)
    preflight = plan_verdict(verdict, batch, state)
    if not isinstance(preflight, Plan):
        return preflight
    if batch.dry_run:
        return _result(verdict, "planned", preflight.reason, plan=preflight)
    embedding = None
    if preflight.replacement_created:
        embedding = await embed_text(VllmEmbedder(), (verdict.merged_text or "").strip())
    lock_ids = set(verdict.member_ids)
    if (merged_id := _merged_id(verdict)) is not None:
        lock_ids.add(merged_id)
    async with db.acquire() as conn:
        try:
            async with conn.transaction():
                await conn.execute(NAMESPACE_LOCK_SQL, verdict.namespace)
                rows = await lock_rows(conn, sorted(lock_ids))
                await _exact_search(conn)
                state = await load_state(conn, verdict, batch)
                plan = plan_verdict(verdict, batch, state)
                if not isinstance(plan, Plan):
                    return plan
                if plan != preflight:
                    return _result(
                        verdict,
                        "stale",
                        "the notes changed while the verdict was applied",
                        current_groups=_current_groups(verdict, batch, state),
                    )
                return await apply_plan(conn, verdict, batch, plan, state, rows, embedding)
        except Rollback as exc:
            return exc.result
        except asyncpg.UniqueViolationError as exc:
            if getattr(exc, "constraint_name", None) != IDEMPOTENCY_CONSTRAINT:
                raise
            return await _idempotency_conflict(conn, verdict, batch)


async def process_batch(batch: Batch) -> list[dict[str, Any]]:
    """Each verdict in order, alone: one verdict's failure never blocks the next.

    An unexpected error (embedder, database) rolls its verdict back and becomes a
    `failed` result naming the error; the verdict may be retried with the same key.
    """
    results = []
    for verdict in batch.verdicts:
        try:
            results.append(await process_verdict(verdict, batch))
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"[:MAX_FAILURE_CHARS]
            results.append(_result(verdict, "failed", reason))
    return results


# ---- undo -------------------------------------------------------------------


def undo_refusal(
    action: Any, rows: dict[str, Any], active_successor: str | None, later_action: bool
) -> str | None:
    """Why an action cannot be undone as recorded, or None.

    Every archived note must still carry this action's archive; a created replacement must
    be active with no active successor and no later action built on it.
    """
    survivors = list(action["survivor_ids"])
    for archived_id in action["archived_ids"]:
        row = rows.get(archived_id)
        if row is None:
            return f"note {archived_id} no longer exists"
        if row["archived_at"] != action["applied_at"]:
            return f"note {archived_id} was restored or archived again after the action"
        if _metadata(row["metadata"]).get("consolidated_into") != survivors:
            return f"note {archived_id} no longer records consolidated_into {survivors}"
    if action["replacement_created"]:
        replacement = action["replacement_id"]
        row = rows.get(replacement)
        if row is None:
            return f"replacement {replacement} no longer exists"
        if row["archived_at"] is not None:
            return f"replacement {replacement} is archived"
        if active_successor is not None:
            return f"replacement {replacement} is superseded by active note {active_successor}"
        if later_action:
            return f"a later action lists replacement {replacement} as a member"
    return None


async def undo(action_id: int, author: str) -> dict[str, Any]:
    """Reverse one action: restore what it archived, archive a replacement it created."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        namespace = await conn.fetchval(ACTION_NAMESPACE_SQL, action_id)
        if namespace is None:
            raise UndoNotFound(f"no consolidation action {action_id}")
        async with conn.transaction():
            await conn.execute(NAMESPACE_LOCK_SQL, namespace)
            action = await conn.fetchrow(ACTION_FOR_UPDATE_SQL, action_id)
            if action["undone_at"] is not None:
                return _jsonb(action["undo_result"])
            if action["action"] == "keep":
                raise UndoRefused("nothing to undo")
            replacement = action["replacement_id"] if action["replacement_created"] else None
            ids = set(action["archived_ids"]) | ({replacement} if replacement else set())
            rows = await lock_rows(conn, sorted(ids))
            successor, later = None, False
            if replacement is not None:
                successor = await conn.fetchval(ACTIVE_SUCCESSOR_SQL, replacement)
                later = await conn.fetchval(LATER_ACTION_SQL, replacement)
            refusal = undo_refusal(action, rows, successor, later)
            if refusal is not None:
                raise UndoRefused(refusal)
            now = time.time()
            prior = _metadata(action["prior"])
            for archived_id in action["archived_ids"]:
                await conn.execute(
                    RESTORE_PRIOR_SQL,
                    archived_id,
                    json.dumps(prior[archived_id], ensure_ascii=False),
                )
            if replacement is not None:
                await conn.execute(ARCHIVE_REPLACEMENT_SQL, replacement, now, author, action_id)
            result = {
                "action_id": action_id,
                "restored_ids": list(action["archived_ids"]),
                "archived_ids": [] if replacement is None else [replacement],
                "undone_at": iso(now),
                "undone_by": author,
            }
            await conn.execute(MARK_UNDONE_SQL, action_id, now, author, json.dumps(result))
            return result


# ---- listing ----------------------------------------------------------------


def _action_entry(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "run_id": row["run_id"],
        "namespace": row["namespace"],
        "action": row["action"],
        "group_key": row["group_key"],
        "idempotency_key": row["idempotency_key"],
        "member_ids": list(row["member_ids"]),
        "archived_ids": list(row["archived_ids"]),
        "survivor_ids": list(row["survivor_ids"]),
        "replacement_id": row["replacement_id"],
        "replacement_created": row["replacement_created"],
        "prior": _jsonb(row["prior"]),
        "applied_at": iso(row["applied_at"]),
        "author": row["author"],
        "model": row["model"],
        "reason": row["reason"],
        "result": _jsonb(row["result"]),
        "undone_at": iso(row["undone_at"]),
        "undone_by": row["undone_by"],
        "undo_result": _jsonb(row["undo_result"]),
    }


def _note_entry(row: Any) -> dict[str, Any]:
    metadata = _metadata(row["metadata"])
    entry = {
        "kind": row["kind"],
        "author": metadata.get("author"),
        "text": row["text"],
        "saved": iso(row["ts_last_active"]),
        "occurred_at": iso(row["occurred_at"]),
        "archived": row["archived_at"] is not None,
    }
    for name in LINEAGE_FIELDS:
        if metadata.get(name) is not None:
            entry[name] = metadata[name]
    return entry


async def list_actions(
    *, namespace: str | None, run_id: str | None, note_id: str | None, limit: int
) -> dict[str, Any]:
    """Actions newest first, and the full current state of every note they reference."""
    async with db.acquire() as conn:
        rows = await conn.fetch(LIST_ACTIONS_SQL, namespace, run_id, note_id, limit)
        ids = sorted(
            {
                i
                for row in rows
                for i in (
                    *row["member_ids"],
                    *row["archived_ids"],
                    *row["survivor_ids"],
                    *([row["replacement_id"]] if row["replacement_id"] else []),
                )
            }
        )
        notes = await conn.fetch(LINEAGE_NOTES_SQL, ids) if ids else []
    return {
        "actions": [_action_entry(row) for row in rows],
        "notes": {row["id"]: _note_entry(row) for row in notes},
    }
