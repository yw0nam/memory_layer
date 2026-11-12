"""Shared by the memory-base hooks: plugin options, the REST client, the repo scope.

Stdlib only — the hooks run under whatever python3 Claude Code invokes, outside any venv.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import subprocess
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

READ_TIMEOUT = 3.0
GIT_TIMEOUT_SECONDS = 1.0

_FENCE_TAG = re.compile(r"<\s*/?\s*memory-context", re.IGNORECASE)
_SCP_ORIGIN = re.compile(r"(?:[^:@/]+@)?(?P<host>[^:/]+):(?P<path>.+)")
_ORIGIN_HOST = re.compile(r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")


@dataclass(frozen=True)
class Config:
    url: str
    api_key: str
    author: str

    @classmethod
    def from_env(cls) -> Config:
        env = os.environ
        return cls(
            url=env.get("CLAUDE_PLUGIN_OPTION_URL", ""),
            api_key=env.get("CLAUDE_PLUGIN_OPTION_API_KEY", ""),
            author=env.get("CLAUDE_PLUGIN_OPTION_AUTHOR", ""),
        )

    @property
    def configured(self) -> bool:
        return bool(self.url and self.api_key and self.author)


def defuse(text: str) -> str:
    """Defuse memory-context tags inside retrieved text so it cannot close the fence."""
    return _FENCE_TAG.sub("[memory-context]", text)


def data_dir() -> Path:
    path = Path(
        os.environ.get("CLAUDE_PLUGIN_DATA") or Path(tempfile.gettempdir()) / "memory_base_plugin"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _request(cfg: Config, req: urllib.request.Request, timeout: float):
    req.add_header("X-API-Key", cfg.api_key)
    req.add_header("X-Author", cfg.author)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def get(cfg: Config, path: str, params: dict | None = None, timeout: float = READ_TIMEOUT):
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    return _request(cfg, urllib.request.Request(f"{cfg.url.rstrip('/')}{path}{query}"), timeout)


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


def emit(context: str | None, system_message: str | None) -> None:
    """Print the hook's output: plain context, or a JSON `systemMessage` line for the user."""
    if system_message is not None:
        print(json.dumps({"systemMessage": system_message}, ensure_ascii=False))
    elif context:
        print(context)
