"""Portable scope validation shared by the message and decision lanes."""

from __future__ import annotations

import ipaddress
import re
from typing import Any
from urllib.parse import urlsplit

SCOPE_MAX_CHARS = 512

_PROJECT_SEGMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SCP_ORIGIN_RE = re.compile(r"(?:(?P<user>[^:@]+)@)?(?P<host>[^:/]+):(?P<path>.+)")
_ORIGIN_HOST_RE = re.compile(r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")


def normalize_scope(scope: Any) -> str:
    """Validate a portable scope: repo:<normalized-origin> or project:<org>/<proj>.

    Canonical repo scopes (`repo:github.com/org/repo`) round-trip unchanged;
    https and scp-style SSH origins additionally normalize onto that form. The
    hostname is lowercased; repository path case is preserved. A local checkout
    is not portable, so the host must be a dotted public hostname.
    """
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError("scope must be repo:<origin> or project:<organization>/<project>")
    scope = scope.strip()
    if scope.startswith("repo:"):
        origin = scope[len("repo:") :]
        if any(char.isspace() for char in origin):
            raise ValueError("repo scope origin must not contain whitespace")
        if "://" in origin:
            parts = urlsplit(origin)
            if parts.scheme != "https" or not parts.hostname:
                raise ValueError("repo scope origin must be an https URL, scp origin, or host/path")
            if parts.username is not None or parts.password is not None:
                raise ValueError("repo scope origin must not embed credentials")
            if parts.port is not None:
                raise ValueError("repo scope origin must not include a port")
            host, path = parts.hostname, parts.path
        elif (scp := _SCP_ORIGIN_RE.fullmatch(origin)) is not None:
            if scp["user"] not in (None, "git"):
                raise ValueError("repo scope origin must not embed credentials")
            host, path = scp["host"], scp["path"]
        else:
            host, _, path = origin.partition("/")
        path = path.split("?", 1)[0].split("#", 1)[0]
        if not _ORIGIN_HOST_RE.fullmatch(host) or is_local_host(host):
            raise ValueError(f"repo scope origin must name a remote host, not {host!r}")
        path = "/" + path.strip("/")
        if path.endswith(".git"):
            path = path[: -len(".git")]
        if not path.strip("/"):
            raise ValueError("repo scope origin must include a repository path")
        return _capped(f"repo:{host.lower()}{path}")
    if scope.startswith("project:"):
        rest = scope[len("project:") :]
        segments = rest.split("/")
        if len(segments) != 2 or not all(_PROJECT_SEGMENT_RE.fullmatch(s) for s in segments):
            raise ValueError("project scope must be project:<organization>/<project>")
        return _capped(f"project:{segments[0].lower()}/{segments[1].lower()}")
    raise ValueError("scope must be repo:<origin> or project:<organization>/<project>")


def _capped(scope: str) -> str:
    if len(scope) > SCOPE_MAX_CHARS:
        raise ValueError(f"scope must be at most {SCOPE_MAX_CHARS} chars")
    return scope


def is_local_host(host: str) -> bool:
    host = host.lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address.is_link_local
