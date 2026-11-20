"""Unit tests for the per-session store of recalled note ids."""

from __future__ import annotations

import pytest

from hook_common import data_dir
from recall import dedup


@pytest.fixture(autouse=True)
def state_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path / "data"))


def test_a_session_keeps_the_ids_added_to_it_until_reset():
    dedup.add("sess-1", ["n1"])
    dedup.add("sess-1", ["n2"])

    assert dedup.load("sess-1") == {"n1", "n2"}
    assert dedup.load("sess-2") == set()

    dedup.reset("sess-1")

    assert dedup.load("sess-1") == set()


def test_a_failed_write_is_dropped():
    (data_dir() / "sessions").write_text("a file where the session directory belongs")

    dedup.add("sess-1", ["n1"])

    assert dedup.load("sess-1") == set()
