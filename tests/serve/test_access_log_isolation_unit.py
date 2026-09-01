"""The access-log buffer is cleared between tests, so unit-test /search
traffic can never be flushed into the real retrieval_log by a later
integration test in the same pytest run.

The two tests below are order-dependent by design (pytest runs a module's
tests in definition order): the first dirties the module-global buffer the
way any unit test hitting /search does, the second proves the isolation
fixture wiped it.
"""

from __future__ import annotations

from memory_base.retrieval.search import Hit
from memory_base.serve import access_log

NOW = 1_700_000_000.0


def _hit(chunk_id: str) -> Hit:
    return Hit(source="memory", ref=f"ref/{chunk_id}", text="body", ts=NOW, meta={"id": chunk_id})


def test_a_dirties_the_module_global_buffer():
    access_log.record_retrieval("leak-probe", "memory", [_hit("chunk-a")], now=NOW)

    assert access_log._pending_logs
    assert access_log._pending_hits


def test_b_buffer_starts_empty_after_a_dirty_test():
    assert access_log._pending_logs == []
    assert access_log._pending_hits == {}
