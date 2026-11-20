"""Unit tests for the session record uploader: checkpointed, chunked, in-order upload."""

from __future__ import annotations

import json
import os
import urllib.error
import uuid

import pytest

from hook_common import data_dir
from session_record import uploader


@pytest.fixture(autouse=True)
def state_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path / "data"))


class FakeServer:
    """Stores events by id as the contract says; `down` refuses, `lose_ack` stores then fails,
    `statuses` answers the next posts with those HTTP errors."""

    def __init__(self):
        self.rows, self.bodies = {}, []
        self.down = self.lose_ack = False
        self.on_post = None
        self.statuses = []

    def __call__(self, body):
        if self.on_post:
            self.on_post()
        if self.down:
            raise OSError("connection refused")
        self.bodies.append(body)
        if self.statuses:
            raise urllib.error.HTTPError("/session_events", self.statuses.pop(0), "no", {}, None)
        stored = known = 0
        conflicts = []
        for event in body["events"]:
            old = self.rows.get(event["id"])
            if old is None:
                self.rows[event["id"]] = event
                stored += 1
            elif old == event:
                known += 1
            else:
                conflicts.append(event["id"])
        if self.lose_ack:
            self.lose_ack = False
            raise OSError("reply lost")
        return {"stored": stored, "known": known, "conflicts": conflicts}

    def sent(self):
        return [event["content"] for body in self.bodies for event in body["events"]]


def _line(uid, *texts, kind="assistant"):
    blocks = [{"type": "text", "text": text} for text in texts]
    entry = {
        "type": kind,
        "uuid": uid,
        "timestamp": "2026-11-20T10:00:00Z",
        "message": {"role": kind, "content": blocks},
    }
    return (json.dumps(entry) + "\n").encode()


def _write(path, *lines, mode="wb"):
    with open(path, mode) as handle:
        handle.write(b"".join(lines))


def _upload(path, server, session="s1"):
    uploader.upload(str(path), session, "/repo", server)


def test_a_block_appended_after_a_run_goes_with_the_next_run_and_nothing_is_resent(tmp_path):
    path, server = tmp_path / "t.jsonl", FakeServer()
    _write(path, _line("u1", "first"), _line("u2", "second"))

    _upload(path, server)
    _write(path, _line("u3", "final"), mode="ab")
    _upload(path, server)

    assert server.sent() == ["first", "second", "final"]
    assert [len(body["events"]) for body in server.bodies] == [2, 1]
    assert {e["session_id"] for e in server.rows.values()} == {"s1"}


def test_a_partial_last_line_waits_until_it_is_complete(tmp_path):
    path, server = tmp_path / "t.jsonl", FakeServer()
    whole = _line("u2", "second")
    _write(path, _line("u1", "first"), whole[:20])

    _upload(path, server)
    assert server.sent() == ["first"]

    _write(path, whole[20:], mode="ab")
    _upload(path, server)
    assert server.sent() == ["first", "second"]


def test_a_server_down_through_the_run_is_caught_up_by_a_sweep(tmp_path):
    path, server = tmp_path / "t.jsonl", FakeServer()
    _write(path, _line("u1", "first"), _line("u2", "second"))
    server.down = True

    _upload(path, server)
    server.down = False
    uploader.sweep(server)

    assert server.sent() == ["first", "second"]


def test_a_backlog_over_500_events_drains_in_order_without_splitting_a_line(tmp_path):
    path, server = tmp_path / "t.jsonl", FakeServer()
    texts = [f"e{n}" for n in range(1200)]
    _write(path, *(_line(f"u{n}", *texts[n : n + 3]) for n in range(0, 1200, 3)))

    _upload(path, server)

    sizes = [len(body["events"]) for body in server.bodies]
    assert len(sizes) > 2 and all(size <= 500 and size % 3 == 0 for size in sizes)
    assert server.sent() == texts


def test_a_backlog_over_the_byte_cap_posts_bodies_under_1_9_mb(tmp_path):
    path, server = tmp_path / "t.jsonl", FakeServer()
    _write(path, *(_line(f"u{n}", "x" * 60000) for n in range(40)))

    _upload(path, server)

    sizes = [len(json.dumps(body).encode()) for body in server.bodies]
    assert len(sizes) > 1 and max(sizes) <= 1_900_000
    assert len(server.sent()) == 40


def test_a_lost_ack_replays_the_same_ids_and_the_server_knows_them(tmp_path):
    path, server = tmp_path / "t.jsonl", FakeServer()
    _write(path, _line("u1", "first"), _line("u2", "second"))
    server.lose_ack = True

    _upload(path, server)
    _upload(path, server)

    first, replay = ([e["id"] for e in body["events"]] for body in server.bodies)
    assert first == replay and len(server.rows) == 2


def _truncate(path):
    _write(path, _line("v1", "new"))


def _replace(path):
    other = path.with_name("other.jsonl")
    _write(other, path.read_bytes())
    os.replace(other, path)


def _rewrite(path):
    _write(path, _line("v1", "new"), _line("v2", "new two"), _line("v3", "new three"))


@pytest.mark.parametrize(
    "change", [_truncate, _replace, _rewrite], ids=["truncated", "replaced", "rewritten"]
)
def test_a_truncated_replaced_or_rewritten_file_replays_from_the_start(tmp_path, change):
    path, server = tmp_path / "t.jsonl", FakeServer()
    _write(path, _line("u1", "first"), _line("u2", "second"))
    _upload(path, server)
    sent = len(server.sent())

    change(path)
    _upload(path, server)

    replayed = [
        json.loads(line)["message"]["content"][0]["text"] for line in path.read_bytes().splitlines()
    ]
    assert server.sent()[sent:] == replayed


def test_a_second_uploader_on_the_same_session_exits_while_the_first_runs(tmp_path):
    path, server = tmp_path / "t.jsonl", FakeServer()
    _write(path, _line("u1", "first"))
    nested = FakeServer()

    def second_uploader():
        server.on_post = None
        _upload(path, nested)

    server.on_post = second_uploader
    _upload(path, server)

    assert (server.sent(), nested.bodies) == (["first"], [])


def test_conflicts_still_advance_and_the_log_names_their_ids_without_content(tmp_path):
    path, server = tmp_path / "t.jsonl", FakeServer()
    _write(path, _line("u1", "secret words"))
    stored = str(uuid.uuid5(uuid.NAMESPACE_URL, "s1:u1:0"))
    server.rows[stored] = {"id": stored, "content": "different"}

    _upload(path, server)
    _upload(path, server)

    assert len(server.bodies) == 1
    log = (data_dir() / "records.log").read_text()
    assert stored in log and "secret" not in log


def _backlog(path, count=600):
    _write(path, *(_line(f"u{n}", f"secret {n}") for n in range(count)))


def test_a_chunk_rejected_with_400_is_logged_skipped_and_the_next_chunk_is_sent(tmp_path):
    path, server = tmp_path / "t.jsonl", FakeServer()
    _backlog(path)
    server.statuses = [400]

    _upload(path, server)
    _upload(path, server)

    rejected, sent = server.bodies
    assert [len(rejected["events"]), len(sent["events"])] == [500, 100]
    assert len(server.rows) == 100
    log = (data_dir() / "records.log").read_text()
    assert f"rejected=500 first_id={rejected['events'][0]['id']} status=400" in log
    assert "secret" not in log


def test_a_503_stops_the_pass_without_advancing(tmp_path):
    path, server = tmp_path / "t.jsonl", FakeServer()
    _backlog(path)
    server.statuses = [503]

    _upload(path, server)
    _upload(path, server)

    first, *again = ([e["id"] for e in body["events"]] for body in server.bodies)
    assert len(server.bodies) == 3 and again[0] == first
    assert len(server.rows) == 600


def test_a_sweep_removes_the_files_of_a_session_whose_transcript_is_gone(tmp_path):
    path, server = tmp_path / "t.jsonl", FakeServer()
    _write(path, _line("u1", "first"))
    _upload(path, server)
    path.unlink()

    uploader.sweep(server)

    assert list((data_dir() / "records").iterdir()) == []
