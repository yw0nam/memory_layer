"""The extraction prompts, reply parser, and turn renderer the LongMemEval harness uses."""

from __future__ import annotations

import pytest

from memory_base.eval import extraction


def test_the_digest_prompt_carries_the_turn_range_and_tags_contract():
    prompt = extraction.load_prompt("digest")
    assert "turn_start" in prompt and "turn_end" in prompt and "tags" in prompt
    assert "{date}" in prompt and "{session}" in prompt
    assert "turn_start" in extraction.EXTRACTION_SYSTEM_PROMPT


def test_the_personal_prompt_asks_for_content_only():
    prompt = extraction.load_prompt("personal")
    assert "{date}" in prompt and "{session}" in prompt
    for field in ("turn_start", "tags", "kind", "episode"):
        assert field not in prompt
    assert extraction.PERSONAL_SYSTEM_PROMPT == 'Return only JSON: {"notes": [{"content": string}]}'
    assert extraction.parse_extraction('{"notes": [{"content": "a"}]}') == [
        {"content": "a", "kind": "note"}
    ]


def test_parse_units_keeps_well_typed_optional_fields_only():
    units = extraction.parse_units(
        {
            "notes": [
                {
                    "content": "a",
                    "kind": "episode",
                    "turn_start": 1,
                    "turn_end": 2,
                    "tags": ["x"],
                    "date": "2026-01-02",
                },
                {"content": "b", "turn_start": "1", "tags": "x", "date": 3},
            ]
        }
    )
    assert units == [
        {
            "content": "a",
            "kind": "episode",
            "turn_start": 1,
            "turn_end": 2,
            "tags": ["x"],
            "date": "2026-01-02",
        },
        {"content": "b", "kind": "note"},
    ]


@pytest.mark.parametrize(
    "payload",
    [[], {"facts": []}, {"notes": {}}, {"notes": ["a"]}, {"notes": [{"content": 1}]}],
)
def test_parse_units_rejects_a_malformed_reply(payload):
    with pytest.raises(ValueError):
        extraction.parse_units(payload)


def test_parse_extraction_reads_the_json_text():
    assert extraction.parse_extraction('{"notes": [{"content": "a", "kind": "note"}]}') == [
        {"content": "a", "kind": "note"}
    ]
    with pytest.raises(ValueError):
        extraction.parse_extraction("not json")


def test_an_oversized_turn_is_cut_to_the_batch_cap_and_numbered_from_first():
    huge = "w" * (extraction.BATCH_CHARS + 1000)
    rendered = extraction.render_turns(
        [{"role": "user", "text": huge}, {"role": "assistant", "text": "short"}], first=4
    )
    assert rendered == f"[4] user: {'w' * extraction.BATCH_CHARS}\n[5] assistant: short"
