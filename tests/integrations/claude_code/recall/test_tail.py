"""Unit tests for the previous assistant text: the Stop hook's tail file and the transcript scan."""

from __future__ import annotations

import json

import pytest

from hook_common import data_dir
from recall import tail


@pytest.fixture(autouse=True)
def state_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path / "data"))


def _entry(kind, *blocks, **flags):
    content = [{"type": b[0], **({"text": b[1]} if b[0] == "text" else {})} for b in blocks]
    return json.dumps({"type": kind, "message": {"role": kind, "content": content}, **flags})


def _text(text, **flags):
    return _entry("assistant", ("text", text), **flags)


def _transcript(tmp_path, *lines, raw=None):
    path = tmp_path / "t.jsonl"
    path.write_bytes(raw if raw is not None else ("\n".join(lines) + "\n").encode())
    return str(path)


def test_save_keeps_the_last_1500_chars_by_atomic_replace():
    tail.save("sess-1", "old")
    tail.save("sess-1", "x" * 100 + "y" * 1500)

    assert tail.previous("sess-1", "") == "y" * 1500
    assert [p.name for p in (data_dir() / "sessions").iterdir()] == ["sess-1.tail"]


def test_a_failed_save_is_silent():
    (data_dir() / "sessions").write_text("a file where the session directory belongs")

    tail.save("sess-1", "text")

    assert tail.previous("sess-1", "") == ""


def test_the_tail_file_wins_over_the_transcript(tmp_path):
    tail.save("sess-1", "from the stop hook")
    path = _transcript(tmp_path, _text("from the transcript"))

    assert tail.previous("sess-1", path) == "from the stop hook"


def test_the_transcript_gives_the_newest_assistant_text_cut_to_1500_chars(tmp_path):
    path = _transcript(tmp_path, _text("older"), _text("a" * 1000 + "b" * 1500))

    assert tail.previous("sess-1", path) == "b" * 1500


def test_an_assistant_entry_with_only_tool_use_gives_the_previous_text(tmp_path):
    path = _transcript(
        tmp_path,
        _text("the answer"),
        _entry("user", ("text", "go on")),
        _entry("assistant", ("tool_use",)),
    )

    assert tail.previous("sess-1", path) == "the answer"


def test_text_blocks_of_one_entry_concatenate(tmp_path):
    path = _transcript(
        tmp_path, _entry("assistant", ("text", "one"), ("tool_use",), ("text", "two"))
    )

    assert tail.previous("sess-1", path) == "one\ntwo"


def test_sidechain_and_meta_entries_are_skipped(tmp_path):
    path = _transcript(
        tmp_path,
        _text("main"),
        _text("sub-agent", isSidechain=True),
        _text("meta", isMeta=True),
    )

    assert tail.previous("sess-1", path) == "main"


def test_a_partial_first_and_last_line_are_ignored(tmp_path):
    lines = [_text("filler " * 40000), _text("kept")]
    raw = ("\n".join(lines) + '\n{"type": "assistant", "message": {"con').encode()

    assert tail.previous("sess-1", _transcript(tmp_path, raw=raw)) == "kept"


def test_a_single_line_over_256_kb_gives_nothing(tmp_path):
    path = _transcript(tmp_path, _text("x" * (300 * 1024)))

    assert tail.previous("sess-1", path) == ""


@pytest.mark.parametrize("path", ["", "/missing/transcript.jsonl"], ids=["none", "missing"])
def test_no_readable_transcript_gives_nothing(path):
    assert tail.previous("sess-1", path) == ""


def test_a_line_nested_too_deep_to_parse_is_skipped(tmp_path):
    path = _transcript(tmp_path, _text("kept"), "[" * 200000)

    assert tail.previous("sess-1", path) == "kept"


def test_remove_deletes_the_tail_file():
    tail.save("sess-1", "text")

    tail.remove("sess-1")
    tail.remove("sess-1")

    assert tail.previous("sess-1", "") == ""
