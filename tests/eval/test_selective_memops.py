"""Unit coverage for the MemOps update-chain and credential unit set."""

from __future__ import annotations

import shutil
from pathlib import Path

from memory_base.eval.selective import memops

FIXTURE = Path(__file__).parent / "fixtures" / "memops_A01_update.json"


def test_an_update_chain_and_a_credential_case_are_extracted(tmp_path):
    shutil.copy(FIXTURE, tmp_path / "A01_update.json")
    chain, credential = memops.convert(tmp_path)

    assert chain["kind"] == "update_chain"
    assert chain["case_id"] == "A01_update/title_chain"
    assert [(s["old_value"], s["new_value"], s["validity"]) for s in chain["steps"]] == [
        (None, "Junior Data Analyst", "confirmed"),
        ("Junior Data Analyst", "Senior Data Analyst", "confirmed"),
        ("Senior Data Analyst", "Lead Data Analyst", "tentative"),
    ]
    assert chain["current"] == "Senior Data Analyst"
    assert [s["segment_index"] for s in chain["segments"]] == [1, 2, 3]
    assert chain["segments"][1]["turns"][0]["role"] == "user"

    assert credential["kind"] == "credential"
    assert (credential["case_id"], credential["value"]) == ("A01_update/op4", "7-24-33")
    assert [s["segment_index"] for s in credential["segments"]] == [1]
