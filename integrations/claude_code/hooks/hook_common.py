"""Shared by the memory-base hooks: plugin options, the REST client, the repo scope, turn capture.

Stdlib only — the hooks run under whatever python3 Claude Code invokes, outside any venv.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

READ_TIMEOUT = 3.0
TURN_TIMEOUT = 2.0
GIT_TIMEOUT_SECONDS = 1.0
RETRY_SECONDS = 60
ORIGIN = "claude_code"
MESSAGE_LIMIT = 120

_FENCE_TAG = re.compile(r"<\s*/?\s*memory-context", re.IGNORECASE)
_SCP_ORIGIN = re.compile(r"(?:[^:@/]+@)?(?P<host>[^:/]+):(?P<path>.+)")
_ORIGIN_HOST = re.compile(r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")


@dataclass(frozen=True)
class Config:
    url: str
    api_key: str
    author: str
    namespace: str | None
    user_entity: str | None

    @classmethod
    def from_env(cls) -> Config:
        env = os.environ
        return cls(
            url=env.get("CLAUDE_PLUGIN_OPTION_URL", ""),
            api_key=env.get("CLAUDE_PLUGIN_OPTION_API_KEY", ""),
            author=env.get("CLAUDE_PLUGIN_OPTION_AUTHOR", ""),
            namespace=env.get("CLAUDE_PLUGIN_OPTION_NAMESPACE") or None,
            user_entity=env.get("CLAUDE_PLUGIN_OPTION_USER_ENTITY") or None,
        )

    @property
    def configured(self) -> bool:
        return bool(self.url and self.api_key and self.author)


class Unreachable(Exception):
    """The server did not answer: connection error or timeout."""


class Refused(Exception):
    """The server answered with an HTTP error status."""

    def __init__(self, status: int, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message


def defuse(text: str) -> str:
    """Defuse memory-context tags inside retrieved text so it cannot close the fence."""
    return _FENCE_TAG.sub("[memory-context]", text)


def _now() -> float:
    return time.time()


def data_dir() -> Path:
    path = Path(
        os.environ.get("CLAUDE_PLUGIN_DATA") or Path(tempfile.gettempdir()) / "memory_base_plugin"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _request(cfg: Config, req: urllib.request.Request, timeout: float):
    req.add_header("X-API-Key", cfg.api_key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        try:
            message = str(json.loads(exc.read()).get("error", ""))
        except Exception:
            message = ""
        raise Refused(exc.code, message or str(exc.reason)) from None
    except (urllib.error.URLError, OSError) as exc:
        raise Unreachable(str(exc)) from None


def get(cfg: Config, path: str, params: dict | None = None, timeout: float = READ_TIMEOUT):
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    return _request(cfg, urllib.request.Request(f"{cfg.url.rstrip('/')}{path}{query}"), timeout)


def post(cfg: Config, path: str, body: dict, timeout: float = READ_TIMEOUT):
    req = urllib.request.Request(
        f"{cfg.url.rstrip('/')}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return _request(cfg, req, timeout)


def _is_local_host(host: str) -> bool:
    host = host.lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address.is_link_local


def repo_scope(origin: str | None) -> str | None:
    """The server's repo scope for a git origin, or None when it is not portable.

    Matches `normalize_scope` on the server: the hostname is lowercased, path
    case is kept, and a trailing `.git` and slashes are dropped. Credentials,
    user names, and ports never reach the scope.
    """
    origin = (origin or "").strip()
    if not origin or any(char.isspace() for char in origin):
        return None
    if "://" in origin:
        parts = urllib.parse.urlsplit(origin)
        host, path = parts.hostname or "", parts.path
    elif (scp := _SCP_ORIGIN.fullmatch(origin)) is not None:
        host, path = scp["host"], scp["path"]
    else:
        return None
    path = path.split("?", 1)[0].split("#", 1)[0].strip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    if not path or not _ORIGIN_HOST.fullmatch(host) or _is_local_host(host):
        return None
    return f"repo:{host.lower()}/{path}"


def git_origin(cwd: str) -> str | None:
    """The `origin` remote URL of the repository holding cwd, or None."""
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except Exception:
        return None
    url = result.stdout.strip()
    return url if result.returncode == 0 and url else None


def _marker_path(session_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", session_id)
    return data_dir() / "sessions" / f"{safe}.json"


def _load_marker(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_marker(path: Path, marker: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(marker))


def post_turn(cfg: Config, session_id: str, role: str, content: str, cwd: str) -> str | None:
    """Append one turn to the server; the one-line `systemMessage` to show the user, or None."""
    path = _marker_path(session_id)
    marker = _load_marker(path)
    if _now() - marker.get("failed_at", float("-inf")) < RETRY_SECONDS:
        return None
    body = {
        "id": str(uuid.uuid4()),
        "origin": ORIGIN,
        "session_id": session_id,
        "role": role,
        "content": content,
    }
    if repo := repo_scope(git_origin(cwd)) if cwd else None:
        body["repo"] = repo
    if cfg.namespace:
        body["namespace"] = cfg.namespace
    try:
        post(cfg, "/turns", body, TURN_TIMEOUT)
    except Unreachable:
        notified = marker.get("notified", False)
        _save_marker(path, {**marker, "failed_at": _now(), "notified": True})
        return None if notified else "memory-base unreachable; turns are not being saved"
    except Refused as exc:
        refusals = marker.get("refusals", [])
        if exc.status in refusals:
            return None
        _save_marker(path, {**marker, "refusals": [*refusals, exc.status]})
        hint = "; check the API key in /config" if exc.status == 401 else ""
        return f"memory-base refused this turn ({exc.status}: {exc.message[:MESSAGE_LIMIT]}){hint}"
    if "failed_at" in marker:
        path.unlink(missing_ok=True)
        return "memory-base is reachable again; turns are being saved"
    return None


def emit(event: str, context: str | None, system_message: str | None) -> None:
    """Print the hook's output: plain context, or one JSON object when a user line is present."""
    if system_message is None:
        if context:
            print(context)
        return
    output: dict = {"systemMessage": system_message}
    if context:
        output["hookSpecificOutput"] = {"hookEventName": event, "additionalContext": context}
    print(json.dumps(output, ensure_ascii=False))
