import json
from datetime import datetime, timezone

from memory_base.ingest.history import Message, build_transcript, group_bursts, parse_jsonl


def line(**values):
    return json.dumps(values, ensure_ascii=False)


def test_parser_contract():
    fixture = "\n".join(
        [
            line(type="system", message={"content": "discard"}),
            line(
                type="user",
                sessionId="s1",
                timestamp="2026-06-25T05:08:12.347Z",
                message={"content": "실제 질문"},
            ),
            line(
                type="assistant",
                sessionId="s1",
                timestamp="2026-06-25T05:08:13.347Z",
                message={
                    "content": [
                        {"type": "thinking", "thinking": "숨은 사고"},
                        {"type": "text", "text": "확인"},
                        {"type": "tool_use", "name": "Read"},
                    ]
                },
            ),
            line(
                type="user",
                sessionId="s1",
                message={
                    "content": [{"type": "tool_result", "content": "도구 출력", "is_error": True}]
                },
            ),
            line(
                type="assistant",
                sessionId="s1",
                isSidechain=True,
                message={"content": [{"type": "text", "text": "sidechain"}]},
            ),
            "{broken",
        ]
    )
    messages = parse_jsonl(fixture)
    assert [m.text for m in messages] == ["실제 질문", "확인"]
    assert messages[1].tool_names == ("Read",)
    assert messages[1].tool_error is True
    assert build_transcript(messages) == "USER: 실제 질문\nASSISTANT: 확인"


def msg(role, text, seconds, error=False):
    return Message(
        role,
        text,
        datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() + seconds,
        "s1",
        tool_error=error,
    )


def test_burst_group_and_length_filter():
    bursts = group_bursts(
        [
            msg("user", "가" * 110, 0),
            msg("user", "나" * 100, 5),
            msg("assistant", "짧음", 10),
            msg("user", "다" * 199, 20),
        ]
    )
    assert len(bursts) == 1
    assert bursts[0].text == "가" * 110 + "\n" + "나" * 100


def test_burst_social_signals():
    assert group_bursts([msg("assistant", "x" * 200, 0, True)])[0].social_weight == 1.5
    assert (
        group_bursts([msg("assistant", "x" * 200, 0), msg("user", "y" * 200, 120)])[0].social_weight
        == 1.5
    )
