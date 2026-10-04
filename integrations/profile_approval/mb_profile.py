"""Approve or reject an agent's proposed change to its user profile, with the user's key.

    python3 mb_profile.py pending [--owner OWNER]
    python3 mb_profile.py show ID
    python3 mb_profile.py approve ID [--note TEXT]
    python3 mb_profile.py reject ID [--note TEXT]

`show` prints the stored proposal and a unified diff against the owner's current user
content, and warns when a pending proposal was written against an older user version
(approving it would be refused as stale).

The user's key carries the `user` author; an agent's key does not. Configuration:
MEMORY_BASE_USER_ENV names a file (default ~/.config/memory-base/user.env) of
`NAME=value` lines, read as data and never executed; only MEMORY_BASE_URL (default
http://127.0.0.1:8010) and MEMORY_BASE_USER_KEY are taken from it, and a non-empty
environment variable of the same name wins. No other key is ever used. Stdlib only.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8010"
DEFAULT_USER_ENV = "~/.config/memory-base/user.env"
HTTP_TIMEOUT_SECONDS = 10
SETTINGS = ("MEMORY_BASE_URL", "MEMORY_BASE_USER_KEY")


class CliError(Exception):
    """A failure reported as one line on stderr with a non-zero exit."""


def _file_settings(path: Path) -> dict[str, str]:
    try:
        text = path.read_text()
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise CliError(f"cannot read {path}: {exc.strerror}") from None
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = (part.strip() for part in line.split("=", 1))
        if name in SETTINGS:
            values[name] = value
    return values


def load_settings(environ=os.environ) -> tuple[str, str]:
    """The API URL and the user's key: a non-empty environment value beats the file."""
    path = Path(environ.get("MEMORY_BASE_USER_ENV") or DEFAULT_USER_ENV).expanduser()
    values = _file_settings(path)
    for name in SETTINGS:
        if environ.get(name):
            values[name] = environ[name]
    key = values.get("MEMORY_BASE_USER_KEY", "")
    if not key:
        raise CliError(f"MEMORY_BASE_USER_KEY is not set in the environment or in {path}")
    return values.get("MEMORY_BASE_URL") or DEFAULT_URL, key


class Api:
    def __init__(self, url: str, key: str) -> None:
        self.url = url.rstrip("/")
        self.key = key

    def request(self, method: str, path: str, params=None, body=None):
        query = f"?{urllib.parse.urlencode(params)}" if params else ""
        data = None if body is None else json.dumps(body).encode()
        headers = {"X-API-Key": self.key}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.url}{path}{query}", data=data, method=method, headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise CliError(f"HTTP {exc.code}: {_server_error(exc)}") from None
        except (TimeoutError, socket.timeout):
            raise CliError(f"{method} {path} timed out after {HTTP_TIMEOUT_SECONDS}s") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise CliError(f"{method} {path} timed out after {HTTP_TIMEOUT_SECONDS}s") from None
            raise CliError(f"cannot reach {self.url}: {exc.reason}") from None
        except OSError as exc:
            raise CliError(f"{method} {path} failed: {exc}") from None
        try:
            return json.loads(raw)
        except ValueError:
            raise CliError(f"{method} {path} returned a response that is not JSON") from None


def _server_error(exc: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(exc.read())
    except Exception:
        return exc.reason or "no detail"
    return json.dumps(body, ensure_ascii=False) if isinstance(body, dict) else str(body)


PROPOSAL_FIELDS = {
    "id": int,
    "owner": str,
    "status": str,
    "base_version": int,
    "reason": str,
    "content": str,
    "created_at": str,
}


def _proposal(row, extra=()) -> dict:
    fields = dict(PROPOSAL_FIELDS, **dict(extra))
    if not isinstance(row, dict) or any(
        type(row.get(name)) is not kind for name, kind in fields.items()
    ):
        raise CliError("the server returned a malformed proposal")
    return row


def pending(api: Api, owner: str | None) -> None:
    params = {"status": "pending"}
    if owner:
        params["owner"] = owner
    rows = api.request("GET", "/profiles/user/proposals", params=params)
    if not isinstance(rows, list):
        raise CliError("the server returned a malformed proposal list")
    rows = [_proposal(row) for row in rows]
    if not rows:
        print("no pending proposals")
        return
    for row in rows:
        print(
            f"proposal {row['id']}  owner={row['owner']}  base=v{row['base_version']}  "
            f"created={row['created_at']}"
        )
        print(f"  reason: {row['reason']}")


def show(api: Api, proposal_id: int) -> None:
    row = _proposal(
        api.request("GET", f"/profiles/user/proposals/{proposal_id}"),
        {"current_user_version": int},
    )
    current = row.get("current_user_content")
    if current is not None and not isinstance(current, str):
        raise CliError("the server returned a malformed proposal")
    version = row["current_user_version"]
    print(f"proposal {row['id']}  owner={row['owner']}  status={row['status']}")
    print(f"written against user v{row['base_version']}; current user version v{version}")
    print(f"created: {row['created_at']}")
    if row.get("decided_at"):
        print(f"decided: {row['decided_at']}")
    if row.get("decision_note"):
        print(f"note: {row['decision_note']}")
    print(f"reason: {row['reason']}")
    if row["status"] == "pending" and row["base_version"] != version:
        print(
            f"warning: stale — written against v{row['base_version']} but the current user "
            f"version is v{version}; approving it would be refused",
            file=sys.stderr,
        )
    diff = list(
        difflib.unified_diff(
            (current or "").splitlines(),
            row["content"].splitlines(),
            fromfile=f"user v{version} (current)",
            tofile=f"proposal {row['id']}",
            lineterm="",
        )
    )
    print("\n".join(diff) if diff else "(no difference from the current user profile)")


def decide(api: Api, action: str, proposal_id: int, note: str | None) -> None:
    body = {} if note is None else {"note": note}
    result = api.request("POST", f"/profiles/user/proposals/{proposal_id}/{action}", body=body)
    if action == "approve":
        if not isinstance(result, dict) or result.get("status") != "approved":
            raise CliError("the server returned a malformed decision")
        if type(result.get("version")) is not int:
            raise CliError("the server returned a malformed decision")
        print(f"approved proposal {proposal_id}: the user profile is now v{result['version']}")
    else:
        if not isinstance(result, dict) or result.get("status") != "rejected":
            raise CliError("the server returned a malformed decision")
        print(f"rejected proposal {proposal_id}")


def _proposal_id(text: str) -> int:
    if not text.isascii() or not text.isdigit() or int(text) < 1:
        raise argparse.ArgumentTypeError("a proposal id is a positive integer")
    return int(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mb_profile.py", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("pending", help="list pending proposals")
    listing.add_argument("--owner", help="one owner only")
    shown = commands.add_parser("show", help="print a proposal and its diff")
    shown.add_argument("id", type=_proposal_id)
    for action in ("approve", "reject"):
        command = commands.add_parser(action, help=f"{action} a pending proposal")
        command.add_argument("id", type=_proposal_id)
        command.add_argument("--note", help="a note stored with the decision")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api = None
    try:
        api = Api(*load_settings())
        if args.command == "pending":
            pending(api, args.owner)
        elif args.command == "show":
            show(api, args.id)
        else:
            decide(api, args.command, args.id, args.note)
    except CliError as exc:
        message = str(exc) if api is None else str(exc).replace(api.key, "[key]")
        print(f"error: {message}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: {type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
