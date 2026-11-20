"""Unit tests for the transcript line -> session event mapping."""

from __future__ import annotations

import json
import uuid

import pytest

from session_record import mapper

WHERE = {"repo": "repo:github.com/o/r", "host": "box", "cwd": "/repo"}
AT = "2026-11-20T10:00:00.000Z"


def _line(kind, content, uid="u1", **flags):
    entry = {
        "type": kind,
        "uuid": uid,
        "timestamp": AT,
        "message": {"role": kind, "content": content},
        **flags,
    }
    return json.dumps(entry).encode() + b"\n"


def _id(session, uid, index):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{session}:{uid}:{index}"))


def _event(session, uid, index, role, content, tool_name=None, tool_use_id=None):
    return {
        "id": _id(session, uid, index),
        "session_id": session,
        "role": role,
        "content": content,
        "tool_name": tool_name,
        "tool_use_id": tool_use_id,
        "event_at": AT,
        **WHERE,
    }


def test_prompts_text_tool_calls_and_tool_results_become_events_in_block_order():
    lines = [
        _line("user", "fix the hair", uid="u1"),
        _line(
            "assistant",
            [
                {"type": "thinking", "thinking": "hidden"},
                {"type": "text", "text": "Reading it."},
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/a"}},
                {"type": "tool_use", "id": "t2", "name": "Noop", "input": {}},
            ],
            uid="u2",
        ),
        _line(
            "user",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "text", "text": "line 1"}, {"type": "image"}],
                },
                {"type": "text", "text": "and this"},
            ],
            uid="u3",
        ),
    ]

    events = [event for line in lines for event in mapper.events(line, "s1", WHERE)]

    assert events == [
        _event("s1", "u1", 0, "user", "fix the hair"),
        _event("s1", "u2", 1, "assistant", "Reading it."),
        _event("s1", "u2", 2, "tool_call", '{"file_path": "/a"}', "Read", "t1"),
        _event("s1", "u2", 3, "tool_call", "", "Noop", "t2"),
        _event("s1", "u3", 0, "tool_result", "line 1\n[image]", None, "t1"),
        _event("s1", "u3", 1, "user", "and this"),
    ]


@pytest.mark.parametrize(
    "line",
    [
        _line("user", "meta", isMeta=True),
        _line("assistant", [{"type": "text", "text": "side"}], isSidechain=True),
        _line("user", "summary of the compacted part", isCompactSummary=True),
        _line("system", "a system entry"),
        _line("attachment", "an attachment"),
        _line(
            "assistant",
            [
                {"type": "thinking", "thinking": "t"},
                {"type": "redacted_thinking", "data": "r"},
                {"type": "image", "source": {}},
            ],
        ),
        b'{"type": "user", "uuid": "u1", "mess\n',
    ],
    ids=["meta", "sidechain", "compact-summary", "system", "attachment", "blocks", "bad-json"],
)
def test_skipped_entries_and_blocks_give_no_events(line):
    assert mapper.events(line, "s1", WHERE) == []


def test_tool_input_and_result_cut_at_2000_and_other_text_at_64000_with_the_marker():
    long_input = {"command": "x" * 3000}  # 3015 chars as JSON
    lines = [
        _line("assistant", [{"type": "tool_use", "id": "t", "name": "Bash", "input": long_input}]),
        _line("user", [{"type": "tool_result", "tool_use_id": "t", "content": "y" * 2500}]),
        _line("assistant", [{"type": "text", "text": "z" * 70000}]),
        _line("user", "w" * 64000),
    ]

    call, result, text, prompt = (mapper.events(line, "s1", WHERE)[0]["content"] for line in lines)

    assert (len(call), call[-18:]) == (2000, " [cut, 3015 chars]")
    assert (len(result), result[-18:]) == (2000, " [cut, 2500 chars]")
    assert (len(text), text[-19:]) == (64000, " [cut, 70000 chars]")
    assert prompt == "w" * 64000


def test_ids_are_stable_across_reruns_and_distinct_across_sessions():
    line = _line("assistant", [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])

    first = [e["id"] for e in mapper.events(line, "s1", WHERE)]
    again = [e["id"] for e in mapper.events(line, "s1", WHERE)]
    forked = [e["id"] for e in mapper.events(line, "s2", WHERE)]

    assert first == again == [_id("s1", "u1", 0), _id("s1", "u1", 1)]
    assert not set(first) & set(forked)


def test_odd_shapes_never_raise_content_is_a_string_and_tool_names_are_cut_to_200():
    lines = [
        _line("user", [{"type": "tool_result", "tool_use_id": "t", "content": ["x"]}]),
        _line("user", [{"type": "tool_result", "tool_use_id": "t", "content": {"a": 1}}]),
        _line("assistant", [{"type": "text", "text": {"a": 1}}]),
        _line("assistant", [{"type": "tool_use", "id": "i" * 300, "name": "n" * 300}]),
        _line("assistant", ["a bare string", 7, None]),
    ]

    events = [mapper.events(line, "s1", WHERE) for line in lines]

    assert [[e["content"] for e in found] for found in events] == [
        ["x"],
        ['{"a": 1}'],
        ['{"a": 1}'],
        [""],
        [],
    ]
    call = events[3][0]
    assert (len(call["tool_name"]), len(call["tool_use_id"])) == (200, 200)


def test_an_entry_with_its_own_cwd_keeps_it_and_one_without_takes_the_given_cwd():
    own = _line("user", "moved", uid="u1", cwd="/repo/sub")
    plain = _line("user", "here", uid="u2")

    assert [mapper.events(line, "s1", WHERE)[0]["cwd"] for line in (own, plain)] == [
        "/repo/sub",
        "/repo",
    ]
