"""Unit tests for atom exclusion from lifecycle operations."""

from __future__ import annotations

import asyncio

from memory_base.serve import admin


class FakeAdminConnection:
    def __init__(self):
        self.rows = {
            "parent": {"chunk_kind": "doc", "archived_at": None},
            "atom": {"chunk_kind": "atom", "archived_at": None},
        }
        self.closed = False

    async def fetch(self, query, *args):
        assert "chunk_kind <> 'atom'" in query
        return [
            {"id": row_id, "kind": row["chunk_kind"]}
            for row_id, row in self.rows.items()
            if row["chunk_kind"] != "atom" and row["archived_at"] is None
        ]

    async def execute(self, query, ids, timestamp=None):
        assert "chunk_kind <> 'atom'" in query
        changed = 0
        for row_id in ids:
            row = self.rows.get(row_id)
            if row is None or row["chunk_kind"] == "atom":
                continue
            if "SET archived_at = NULL" in query:
                if row["archived_at"] is not None:
                    row["archived_at"] = None
                    changed += 1
            elif row["archived_at"] is None:
                row["archived_at"] = timestamp
                changed += 1
        return f"UPDATE {changed}"

    async def close(self):
        self.closed = True


def test_archive_candidates_omit_atoms(monkeypatch):
    monkeypatch.setenv("DB_URL", "postgres://fake/db")
    conn = FakeAdminConnection()

    async def connect(url):
        return conn

    monkeypatch.setattr(admin.asyncpg, "connect", connect)
    rows = asyncio.run(admin.archive_candidates(2_000_000_000.0))
    assert [row["id"] for row in rows] == ["parent"]
    assert conn.closed


def test_archiving_and_restoring_atom_id_are_noops(monkeypatch):
    monkeypatch.setenv("DB_URL", "postgres://fake/db")
    conn = FakeAdminConnection()

    async def connect(url):
        return conn

    monkeypatch.setattr(admin.asyncpg, "connect", connect)
    assert asyncio.run(admin.archive_rows(["atom"], 2_000_000_000.0)) == 0
    assert asyncio.run(admin.restore_rows(["atom"])) == 0
    assert conn.rows["atom"]["archived_at"] is None
