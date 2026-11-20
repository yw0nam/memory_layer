"""The session record hook: a pending checkpoint, then a detached uploader — in process and
as the registered process against a local stub server."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from session_record import checkpoint

PLUGIN = Path(__file__).resolve().parents[3] / "integrations" / "claude_code"
UPLOADER = [sys.executable, "-m", "session_record.uploader"]

_spec = importlib.util.spec_from_file_location(
    "session_record_hook", PLUGIN / "hooks" / "session_record.py"
)
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)


def _env(url, tmp_path, configured=True):
    return {
        "CLAUDE_PLUGIN_OPTION_URL": url,
        "CLAUDE_PLUGIN_OPTION_API_KEY": "key" if configured else "",
        "CLAUDE_PLUGIN_OPTION_AUTHOR": "claude-code",
        "CLAUDE_PLUGIN_DATA": str(tmp_path / "data"),
    }


@pytest.fixture
def configured(monkeypatch, tmp_path):
    for name, value in _env("http://127.0.0.1:9", tmp_path).items():
        monkeypatch.setenv(name, value)


def _payload(event, tmp_path, session="s1"):
    return {
        "hook_event_name": event,
        "session_id": session,
        "transcript_path": str(tmp_path / "t.jsonl"),
        "cwd": "/repo",
    }


@pytest.mark.parametrize("event", ["Stop", "SessionEnd"])
def test_stop_and_session_end_write_the_pending_checkpoint_before_spawning(
    configured, tmp_path, event
):
    seen = []

    def spawn(argv):
        seen.append((argv, checkpoint.load("s1")))

    hook.run(_payload(event, tmp_path), spawn)

    [(argv, point)] = seen
    assert argv == [*UPLOADER, str(tmp_path / "t.jsonl"), "s1", "/repo"]
    assert (point["path"], point["offset"]) == (str(tmp_path / "t.jsonl"), 0)


def test_session_start_spawns_the_sweep(configured, tmp_path):
    seen = []

    hook.run(_payload("SessionStart", tmp_path), seen.append)

    assert seen == [[*UPLOADER, "--sweep"]]


def test_an_unconfigured_plugin_spawns_nothing(monkeypatch, tmp_path):
    for name, value in _env("http://127.0.0.1:9", tmp_path, configured=False).items():
        monkeypatch.setenv(name, value)
    seen = []

    hook.run(_payload("Stop", tmp_path), seen.append)

    assert seen == []


@pytest.fixture
def slow_server():
    bodies = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            time.sleep(2)
            bodies.append((self.path, body))
            reply = json.dumps(
                {"stored": len(body["events"]), "known": 0, "conflicts": []}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", bodies
    server.shutdown()
    server.server_close()


def test_the_registered_stop_hook_exits_at_once_and_the_upload_lands_later(slow_server, tmp_path):
    url, bodies = slow_server
    hooks = json.loads((PLUGIN / "hooks" / "hooks.json").read_text())["hooks"]
    commands = {
        event: [
            h
            for entry in hooks[event]
            for h in entry["hooks"]
            if "session_record.py" in h["command"]
        ]
        for event in ("Stop", "SessionEnd", "SessionStart")
    }
    assert all(len(found) == 1 and found[0]["timeout"] <= 3 for found in commands.values())
    entry = {
        "type": "user",
        "uuid": "u1",
        "timestamp": "2026-11-20T10:00:00Z",
        "message": {"role": "user", "content": "hello"},
    }
    (tmp_path / "t.jsonl").write_text(json.dumps(entry) + "\n")
    env = {**os.environ, "CLAUDE_PLUGIN_ROOT": str(PLUGIN), **_env(url, tmp_path)}

    started = time.monotonic()
    result = subprocess.run(
        commands["Stop"][0]["command"],
        shell=True,
        input=json.dumps(_payload("Stop", tmp_path)),
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    elapsed = time.monotonic() - started

    assert (result.returncode, result.stdout, bodies) == (0, "", [])
    assert elapsed < 1.5
    deadline = time.monotonic() + 10
    while not bodies and time.monotonic() < deadline:
        time.sleep(0.1)
    [(path, body)] = bodies
    assert path == "/session_events"
    assert [(e["role"], e["content"], e["session_id"]) for e in body["events"]] == [
        ("user", "hello", "s1")
    ]
