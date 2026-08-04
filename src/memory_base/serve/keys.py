"""CLI for provisioning memory_base API keys.

    uv run python -m memory_base.serve.keys new <label> [--home <ns>] [--admin]
    uv run python -m memory_base.serve.keys list
    uv run python -m memory_base.serve.keys revoke <prefix-or-hash>

`new` prints the plaintext key exactly once; only its sha256 hash is stored.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.core.schema import ensure_schema_once
from memory_base.serve.auth import generate_key, hash_key


class HomeNamespaceError(ValueError):
    """A key-provisioning request is invalid."""


async def _home_is_accessible(conn: Any, home: str, label: str, is_admin: bool) -> bool:
    row = await conn.fetchrow(
        f'SELECT visibility, owner FROM "{PG_SCHEMA}".namespaces WHERE name = $1', home
    )
    if row is None:
        return False
    if row["visibility"] == "public":
        return True
    return is_admin or row["owner"] == label


async def new_key(label: str, home: str = "default", is_admin: bool = False) -> str:
    """Mint a key for `label`; returns the plaintext (never stored)."""
    plaintext = generate_key()
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        if not await _home_is_accessible(conn, home, label, is_admin):
            raise HomeNamespaceError(
                f"home namespace {home!r} does not exist or is not accessible to {label!r}"
            )
        await conn.execute(
            f"""
            INSERT INTO "{PG_SCHEMA}".api_keys (key_hash, label, home, is_admin)
            VALUES ($1, $2, $3, $4)
            """,
            hash_key(plaintext),
            label,
            home,
            is_admin,
        )
    return plaintext


async def list_keys() -> list[dict[str, Any]]:
    """List every key's metadata (never the hash's plaintext source, which isn't stored)."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        rows = await conn.fetch(
            f"""
            SELECT key_hash, label, home, is_admin, created_at, revoked_at
            FROM "{PG_SCHEMA}".api_keys
            ORDER BY created_at
            """
        )
    return [dict(row) for row in rows]


async def revoke_key(prefix_or_hash: str) -> int:
    """Revoke every active key whose hash starts with `prefix_or_hash`; returns the count."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        status = await conn.execute(
            f"""
            UPDATE "{PG_SCHEMA}".api_keys
            SET revoked_at = now()
            WHERE key_hash LIKE $1 || '%' AND revoked_at IS NULL
            """,
            prefix_or_hash,
        )
    return int(status.rsplit(" ", 1)[-1])


def _print_keys(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("no keys")
        return
    for row in rows:
        status = "revoked" if row["revoked_at"] else "active"
        role = "admin" if row["is_admin"] else "member"
        print(
            f"{row['label']:<20} home={row['home']:<20} {role:<6} {status:<8} "
            f"created={row['created_at']} hash={row['key_hash'][:12]}..."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memory_base.serve.keys")
    subparsers = parser.add_subparsers(dest="command", required=True)

    new_parser = subparsers.add_parser("new", help="mint a key")
    new_parser.add_argument("label")
    new_parser.add_argument("--home", default="default")
    new_parser.add_argument("--admin", action="store_true")

    subparsers.add_parser("list", help="list keys")

    revoke_parser = subparsers.add_parser("revoke", help="revoke a key")
    revoke_parser.add_argument("prefix_or_hash")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "new":
        try:
            plaintext = asyncio.run(new_key(args.label, args.home, args.admin))
        except HomeNamespaceError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"API key for {args.label!r} (store this now, it will not be shown again):")
        print(plaintext)
    elif args.command == "list":
        _print_keys(asyncio.run(list_keys()))
    elif args.command == "revoke":
        count = asyncio.run(revoke_key(args.prefix_or_hash))
        print(f"revoked {count} key(s)")


if __name__ == "__main__":
    main()
