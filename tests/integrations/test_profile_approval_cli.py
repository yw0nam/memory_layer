"""The profile approval CLI against a fake HTTP server on a free local port.

mb_profile.py is stdlib only and importable via the ``pythonpath`` test config entry.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import mb_profile

USER_KEY = "user-key-0123456789abcdef"

PENDING = [
    {
        "id": 5,
        "owner": "natsume",
        "status": "pending",
        "base_version": 0,
        "reason": "she mentioned it twice",
        "content": "Likes tea.",
        "created_at": "2026-10-03T08:00:00+00:00",
        "decided_at": None,
        "decision_note": None,
    },
    {
        "id": 4,
        "owner": "claude-code",
        "status": "pending",
        "base_version": 2,
        "reason": "the user moved",
        "content": "Lives in Seoul.",
        "created_at": "2026-10-02T08:00:00+00:00",
        "decided_at": None,
        "decision_note": None,
    },
]


def _proposal(**fields):
    body = dict(PENDING[1], current_user_version=2, current_user_content="Lives in Busan.")
    body.update(fields)
    return body


class FakeServer:
    """Answers (method, path) with a configured (status, body); records every request."""

    def __init__(self):
        self.routes: dict[tuple[str, str], tuple[int, object]] = {}
        self.requests: list[dict] = []
        self.delay = 0.0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self):
                parts = urllib.parse.urlsplit(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                server.requests.append(
                    {
                        "method": self.command,
                        "path": parts.path,
                        "query": urllib.parse.parse_qs(parts.query),
                        "key": self.headers.get("X-API-Key"),
                        "content_type": self.headers.get("Content-Type"),
                        "body": json.loads(raw) if raw else None,
                    }
                )
                if server.delay:
                    time.sleep(server.delay)
                status, body = server.routes.get((self.command, parts.path), (404, {"error": "no"}))
                payload = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except OSError:
                    pass

            do_GET = _answer
            do_POST = _answer

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    fake = FakeServer()
    yield fake
    fake.close()


@pytest.fixture
def cli_env(monkeypatch, tmp_path, server):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("MEMORY_BASE_USER_ENV", raising=False)
    monkeypatch.setenv("MEMORY_BASE_URL", server.url)
    monkeypatch.setenv("MEMORY_BASE_USER_KEY", USER_KEY)
    monkeypatch.setenv("MEMORY_BASE_API_KEY", "agents-key-must-not-be-used")
    return tmp_path


def _run(capsys, *argv):
    code = mb_profile.main(list(argv))
    out, err = capsys.readouterr()
    assert USER_KEY not in out + err
    assert "Traceback" not in out + err
    return code, out, err


def test_pending_lists_every_owners_pending_proposal(server, cli_env, capsys):
    server.routes[("GET", "/profiles/user/proposals")] = (200, PENDING)
    code, out, _ = _run(capsys, "pending")
    assert code == 0
    assert server.requests == [
        {
            "method": "GET",
            "path": "/profiles/user/proposals",
            "query": {"status": ["pending"]},
            "key": USER_KEY,
            "content_type": None,
            "body": None,
        }
    ]
    assert "5" in out and "natsume" in out and "she mentioned it twice" in out
    assert "claude-code" in out and "the user moved" in out
    assert out.index("natsume") < out.index("claude-code")


def test_pending_narrows_to_one_owner(server, cli_env, capsys):
    server.routes[("GET", "/profiles/user/proposals")] = (200, [])
    code, out, _ = _run(capsys, "pending", "--owner", "claude-code")
    assert code == 0
    assert server.requests[0]["query"] == {"status": ["pending"], "owner": ["claude-code"]}
    assert "no pending proposals" in out


def test_show_prints_the_proposal_and_a_diff_against_the_current_content(server, cli_env, capsys):
    server.routes[("GET", "/profiles/user/proposals/4")] = (200, _proposal())
    code, out, err = _run(capsys, "show", "4")
    assert code == 0
    assert [r["path"] for r in server.requests] == ["/profiles/user/proposals/4"]
    assert "claude-code" in out
    assert "pending" in out
    assert "the user moved" in out
    assert "-Lives in Busan." in out.splitlines()
    assert "+Lives in Seoul." in out.splitlines()
    assert "stale" not in out + err


def test_show_warns_when_a_pending_proposal_is_stale(server, cli_env, capsys):
    server.routes[("GET", "/profiles/user/proposals/4")] = (
        200,
        _proposal(base_version=1, current_user_version=2),
    )
    code, out, err = _run(capsys, "show", "4")
    assert code == 0
    assert "stale" in (out + err).lower()
    assert "v1" in out + err and "v2" in out + err


def test_show_of_a_terminal_older_proposal_diffs_against_the_live_content(server, cli_env, capsys):
    server.routes[("GET", "/profiles/user/proposals/2")] = (
        200,
        _proposal(
            id=2,
            status="rejected",
            base_version=0,
            content="Lives in Daegu.",
            decided_at="2026-10-02T09:00:00+00:00",
            decision_note="not true",
            current_user_version=2,
            current_user_content="Lives in Busan.",
        ),
    )
    code, out, err = _run(capsys, "show", "2")
    assert code == 0
    assert "rejected" in out
    assert "not true" in out
    assert "-Lives in Busan." in out.splitlines()
    assert "+Lives in Daegu." in out.splitlines()
    assert "stale" not in (out + err).lower()


def test_show_against_an_absent_user_part_diffs_from_nothing(server, cli_env, capsys):
    server.routes[("GET", "/profiles/user/proposals/4")] = (
        200,
        _proposal(base_version=0, current_user_version=0, current_user_content=None),
    )
    code, out, _ = _run(capsys, "show", "4")
    assert code == 0
    assert "+Lives in Seoul." in out.splitlines()


@pytest.mark.parametrize(
    ("argv", "body", "path"),
    [
        (["approve", "4"], {}, "/profiles/user/proposals/4/approve"),
        (["approve", "4", "--note", "confirmed"], {"note": "confirmed"}, "/profiles/user/proposals/4/approve"),
        (["reject", "4"], {}, "/profiles/user/proposals/4/reject"),
        (["reject", "4", "--note", "wrong city"], {"note": "wrong city"}, "/profiles/user/proposals/4/reject"),
    ],
)  # fmt: skip
def test_decisions_post_a_json_body(server, cli_env, capsys, argv, body, path):
    server.routes[("POST", "/profiles/user/proposals/4/approve")] = (
        200,
        {"status": "approved", "version": 3},
    )
    server.routes[("POST", "/profiles/user/proposals/4/reject")] = (200, {"status": "rejected"})
    code, out, _ = _run(capsys, *argv)
    assert code == 0
    (request,) = server.requests
    assert (request["method"], request["path"], request["body"]) == ("POST", path, body)
    assert request["content_type"] == "application/json"
    assert request["key"] == USER_KEY
    assert argv[0].rstrip("e") in out


def test_a_server_refusal_prints_the_error_and_exits_non_zero(server, cli_env, capsys):
    server.routes[("POST", "/profiles/user/proposals/4/approve")] = (
        409,
        {"error": "stale", "version": 3},
    )
    code, out, err = _run(capsys, "approve", "4")
    assert code != 0
    assert "stale" in err
    assert "409" in err


@pytest.mark.parametrize(
    ("command", "route", "body"),
    [
        (["pending"], ("GET", "/profiles/user/proposals"), b"not json"),
        (["pending"], ("GET", "/profiles/user/proposals"), {"error": "x"}),
        (["pending"], ("GET", "/profiles/user/proposals"), [{"id": 1}]),
        (["show", "4"], ("GET", "/profiles/user/proposals/4"), [1, 2]),
        (["show", "4"], ("GET", "/profiles/user/proposals/4"), {"id": 4}),
        (["approve", "4"], ("POST", "/profiles/user/proposals/4/approve"), b"<html>"),
        (["approve", "4"], ("POST", "/profiles/user/proposals/4/approve"), {"status": "approved"}),
    ],
)
def test_a_malformed_response_exits_non_zero(server, cli_env, capsys, command, route, body):
    server.routes[route] = (200, body)
    code, _, err = _run(capsys, *command)
    assert code != 0
    assert err.strip()


def test_a_timeout_exits_non_zero(server, cli_env, capsys, monkeypatch):
    monkeypatch.setattr(mb_profile, "HTTP_TIMEOUT_SECONDS", 0.2)
    server.delay = 1.0
    server.routes[("GET", "/profiles/user/proposals")] = (200, [])
    code, _, err = _run(capsys, "pending")
    assert code != 0
    assert "timed out" in err.lower()


def test_an_unreachable_server_exits_non_zero(cli_env, capsys, monkeypatch):
    monkeypatch.setenv("MEMORY_BASE_URL", "http://127.0.0.1:9")
    code, _, err = _run(capsys, "pending")
    assert code != 0
    assert err.strip()


def test_the_default_timeout_is_ten_seconds():
    assert mb_profile.HTTP_TIMEOUT_SECONDS == 10


# ---- configuration ----------------------------------------------------------


def _env_file(path, *lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def test_the_default_user_env_file_supplies_url_and_key(server, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("MEMORY_BASE_USER_ENV", "MEMORY_BASE_URL", "MEMORY_BASE_USER_KEY"):
        monkeypatch.delenv(name, raising=False)
    _env_file(
        tmp_path / ".config" / "memory-base" / "user.env",
        "# the user's key",
        "",
        f"  MEMORY_BASE_URL = {server.url}  ",
        f"MEMORY_BASE_USER_KEY={USER_KEY}",
    )
    server.routes[("GET", "/profiles/user/proposals")] = (200, [])
    code, _, _ = _run(capsys, "pending")
    assert code == 0
    assert server.requests[0]["key"] == USER_KEY


def test_a_non_empty_environment_value_overrides_the_file(server, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    env_file = _env_file(
        tmp_path / "custom.env",
        "MEMORY_BASE_URL=http://127.0.0.1:9",
        "MEMORY_BASE_USER_KEY=file-key",
    )
    monkeypatch.setenv("MEMORY_BASE_USER_ENV", str(env_file))
    monkeypatch.setenv("MEMORY_BASE_URL", server.url)
    monkeypatch.setenv("MEMORY_BASE_USER_KEY", "")
    server.routes[("GET", "/profiles/user/proposals")] = (200, [])
    code, _, _ = _run(capsys, "pending")
    assert code == 0
    assert server.requests[0]["key"] == "file-key"
    monkeypatch.setenv("MEMORY_BASE_USER_KEY", USER_KEY)
    code, _, _ = _run(capsys, "pending")
    assert server.requests[1]["key"] == USER_KEY


def test_the_user_env_path_expands_the_home_directory(server, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("MEMORY_BASE_USER_KEY", raising=False)
    monkeypatch.setenv("MEMORY_BASE_URL", server.url)
    _env_file(tmp_path / "keys" / "u.env", f"MEMORY_BASE_USER_KEY={USER_KEY}")
    monkeypatch.setenv("MEMORY_BASE_USER_ENV", "~/keys/u.env")
    server.routes[("GET", "/profiles/user/proposals")] = (200, [])
    assert _run(capsys, "pending")[0] == 0
    assert server.requests[0]["key"] == USER_KEY


def test_the_file_is_never_executed_or_expanded(server, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    marker = tmp_path / "executed"
    env_file = _env_file(
        tmp_path / "u.env",
        f"touch {marker}",
        f"export MEMORY_BASE_USER_KEY=$(touch {marker})",
        "MEMORY_BASE_USER_KEY=$HOME-key",
        "MEMORY_BASE_API_KEY=agents-key",
        "OTHER=1",
    )
    monkeypatch.setenv("MEMORY_BASE_USER_ENV", str(env_file))
    monkeypatch.setenv("MEMORY_BASE_URL", server.url)
    monkeypatch.delenv("MEMORY_BASE_USER_KEY", raising=False)
    server.routes[("GET", "/profiles/user/proposals")] = (200, [])
    assert _run(capsys, "pending")[0] == 0
    assert server.requests[0]["key"] == "$HOME-key"
    assert not marker.exists()


def test_a_missing_user_key_exits_before_any_request(server, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MEMORY_BASE_URL", server.url)
    monkeypatch.delenv("MEMORY_BASE_USER_KEY", raising=False)
    monkeypatch.delenv("MEMORY_BASE_USER_ENV", raising=False)
    monkeypatch.setenv("MEMORY_BASE_API_KEY", "agents-key")
    _env_file(tmp_path / ".config" / "memory-base" / "env", "MEMORY_BASE_API_KEY=agents-key")
    code, out, err = _run(capsys, "approve", "4")
    assert code != 0
    assert "MEMORY_BASE_USER_KEY" in err
    assert server.requests == []
    assert "agents-key" not in out + err


def test_the_url_defaults_to_the_local_api():
    assert mb_profile.DEFAULT_URL == "http://127.0.0.1:8010"
