"""Notes name the entities they are about: save, list, search hit, and consolidation.

No DB, no network: the save runs on a recording connection, the list on a fake one.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import pytest

from memory_base.retrieval.search import Hit
from memory_base.serve.consolidation import verdicts
from memory_base.serve.consolidation.groups import Note, Pair, group_key
from memory_base.serve.notes import store
from memory_base.serve.search.routes import hit_to_dict

AUG_12 = 1_786_000_000.0
NS = "work"
A, B, C = "note:work:a", "note:work:b", "note:work:c"
MERGED = "Weekday deploys run at 04:00."
TEXTS = {
    A: "Deploys run at 04:00 on weekdays.",
    B: "Weekday deploys start at 04:00.",
    C: "On weekdays the deploy job starts at 04:00.",
}


class _Tx:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return None


class SaveConn:
    """Answers the entity lookup with the active slugs and records every statement."""

    def __init__(self, active=("kuro", "shiro"), insert_status="INSERT 0 1"):
        self.active = active
        self.insert_status = insert_status
        self.statements: list[tuple[str, tuple]] = []

    def transaction(self):
        return _Tx()

    async def fetchval(self, query, *args):
        self.statements.append((" ".join(query.split()), args))
        return 1 if "FOR SHARE" in query else True

    async def fetch(self, query, *args):
        self.statements.append((" ".join(query.split()), args))
        if ".entities" in query:
            return [{"slug": slug} for slug in args[0] if slug in self.active]
        return []

    async def execute(self, query, *args):
        self.statements.append((" ".join(query.split()), args))
        return self.insert_status if "INSERT INTO" in query else "UPDATE 1"


@pytest.fixture
def use(monkeypatch):
    def _use(conn):
        @asynccontextmanager
        async def acquire(timeout=None):
            yield conn

        async def noop(conn):
            return None

        async def embed(embedder, text):
            return [0.0] * 4

        monkeypatch.setattr(store.db, "acquire", acquire)
        monkeypatch.setattr(store, "ensure_schema_once", noop)
        monkeypatch.setattr(store, "embed_text", embed)
        return conn

    return _use


def _save(**overrides):
    fields = {"tags": ["test"], "kind": "work", "namespace": "team-a"}
    fields.update(overrides)
    return asyncio.run(store.save_note("prefer ruff for linting", **fields))


def _inserted_metadata(conn):
    [insert] = [s for s in conn.statements if "INSERT INTO" in s[0]]
    return json.loads(insert[1][10])


# ---- save -------------------------------------------------------------------


def test_a_save_stores_the_slugs_beside_the_tags(use):
    conn = use(SaveConn())
    _save(entities=[" kuro ", "shiro", "kuro"])
    assert _inserted_metadata(conn)["entities"] == ["kuro", "shiro"]


def test_a_save_without_entities_stores_none(use):
    conn = use(SaveConn())
    _save()
    assert "entities" not in _inserted_metadata(conn)


@pytest.mark.parametrize("active,missing", [(("shiro",), "kuro"), ((), "shiro")])
def test_an_unknown_or_archived_slug_is_refused_and_nothing_is_written(use, active, missing):
    conn = use(SaveConn(active=active))
    with pytest.raises(ValueError) as refused:
        _save(entities=["shiro", "kuro"])
    assert str(refused.value) == (
        f"unknown entity '{missing}'; register it with register_entity "
        "or reuse a slug from list_entities"
    )
    assert not [s for s in conn.statements if "INSERT INTO" in s[0]]


def test_eleven_slugs_are_refused(use):
    use(SaveConn())
    with pytest.raises(ValueError, match="entities must hold at most 10 slugs"):
        _save(entities=[f"e{i}" for i in range(11)])


@pytest.mark.parametrize("slug", ["Kuro", "-kuro", "ku ro", "k" * 41])
def test_a_malformed_slug_is_refused(use, slug):
    use(SaveConn())
    with pytest.raises(ValueError, match="slug"):
        _save(entities=[slug])


def test_an_identical_content_save_changes_nothing(use):
    conn = use(SaveConn(insert_status="INSERT 0 0"))
    result = _save(entities=["kuro"])
    assert result["stored"] is False
    writes = [s for s in conn.statements if s[0].startswith(("INSERT", "UPDATE"))]
    assert len(writes) == 1 and "ON CONFLICT (id) DO NOTHING" in writes[0][0]


def test_a_supersede_stores_the_slugs_passed_with_the_new_note(use):
    conn = use(SaveConn())
    _save(entities=["shiro"], supersedes="note:team-a:old")
    assert _inserted_metadata(conn)["entities"] == ["shiro"]


# ---- list and search --------------------------------------------------------


class ListConn:
    def __init__(self, rows):
        self.rows = rows
        self.query = ""
        self.args: tuple = ()

    async def fetch(self, query, *args):
        self.query, self.args = query, args
        return self.rows


def _list(monkeypatch, rows=(), **kwargs):
    conn = ListConn(list(rows))

    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    monkeypatch.setattr(store.db, "acquire", acquire)
    return conn, asyncio.run(store.list_notes(**kwargs))


def test_list_notes_filters_by_entity_and_composes_with_the_other_filters(monkeypatch):
    conn, _ = _list(monkeypatch, entity="kuro", kind="work", tags=["Infra"])
    assert "metadata->'entities' ? $" in conn.query
    assert "chunk_kind = $" in conn.query and "metadata->'tags' ?| $" in conn.query
    assert "kuro" in conn.args and "work" in conn.args and ["infra"] in conn.args


def test_list_rows_carry_entities_and_default_to_empty(monkeypatch):
    base = {
        "id": "note:a",
        "kind": "work",
        "text": "t",
        "ts_last_active": AUG_12,
        "namespace": "default",
        "archived_at": None,
        "occurred_at": None,
    }
    _, rows = _list(
        monkeypatch,
        [
            {**base, "metadata": {"tags": [], "entities": ["kuro"]}},
            {**base, "metadata": {"tags": []}},
        ],
    )
    assert [r["entities"] for r in rows] == [["kuro"], []]


def test_a_search_hit_carries_entities():
    meta = {"id": "note:a", "kind": "work", "tags": [], "entities": ["kuro"], "occurred_at": None}
    hit = Hit(source="memory", ref="r", text="t", ts=AUG_12, meta=meta)
    assert hit_to_dict(hit)["entities"] == ["kuro"]


# ---- consolidation ----------------------------------------------------------


def _notes(**entities):
    return {
        i: Note(
            id=i,
            kind="work",
            author="claude-code",
            saved=1_700_000_000.0,
            occurred_at=None,
            tags=("deploy",),
            entities=tuple(entities.get(i[-1], ())),
            similar_ack=(),
            supersedes=None,
            text=TEXTS[i],
        )
        for i in (A, B, C)
    }


def _merge(notes):
    verdict = {
        "namespace": NS,
        "group_key": group_key(NS, notes.values()),
        "idempotency_key": "k-1",
        "member_ids": [A, B, C],
        "action": "merge",
        "reason": "one rule",
        "merged_text": MERGED,
    }
    batch = verdicts.parse_batch(
        {"run_id": "r", "author": "consolidator", "model": "m", "verdicts": [verdict]}, {NS}
    )
    pairs = [Pair(A, B, 0.80), Pair(A, C, 0.78), Pair(B, C, 0.75)]
    current = verdicts.State(
        recorded=None,
        cached=None,
        pairs=pairs,
        notes=notes,
        used_actions=0,
        replacement=None,
    )
    return batch, batch.verdicts[0], current


def test_group_key_changes_when_a_members_entities_change():
    assert group_key(NS, _notes(a=("kuro",)).values()) != group_key(NS, _notes().values())


def test_a_merge_gives_the_replacement_the_union_of_the_members_entities():
    batch, verdict, _ = _merge(_notes(a=("kuro",), b=("shiro",), c=("kuro",)))
    row = verdicts.replacement_row(
        verdict, batch, _notes(a=("kuro",), b=("shiro",), c=("kuro",)), 0.0
    )
    assert row["metadata"]["entities"] == ["kuro", "shiro"]


def test_a_merge_whose_union_exceeds_ten_is_rejected():
    notes = _notes(a=tuple(f"e{i:02d}" for i in range(6)), b=tuple(f"f{i}" for i in range(5)))
    batch, verdict, current = _merge(notes)
    result = verdicts.plan_verdict(verdict, batch, current)
    assert result["status"] == "rejected"
    assert result["reason"] == "merged note would name 11 entities; at most 10"
