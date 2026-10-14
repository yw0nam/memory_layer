"""API-key provisioning CLI.

    uv run python -m memory_base.serve.access.keys new <label>
    uv run python -m memory_base.serve.access.keys list
    uv run python -m memory_base.serve.access.keys revoke <prefix-or-hash>

`new` prints the plaintext key exactly once, on its last line; only its sha256 hash is stored.
The label names the client that holds the key; every key is equal and revoked on its own.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.core.schema import ensure_schema_once
from memory_base.serve.access.auth import generate_key, hash_key


class KeyPrefixError(ValueError):
    """A key prefix is too short to revoke safely."""


async def new_key(label: str) -> str:
    """Mint a key for the client `label`; returns the plaintext (never stored)."""
    plaintext = generate_key()
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        await conn.execute(
            f'INSERT INTO "{PG_SCHEMA}".api_keys (key_hash, label) VALUES ($1, $2)',
            hash_key(plaintext),
            label,
        )
    return plaintext


async def list_keys() -> list[dict[str, Any]]:
    """List every key's metadata (never the hash's plaintext source, which isn't stored)."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        rows = await conn.fetch(
            f"""
            SELECT key_hash, label, created_at, revoked_at
            FROM "{PG_SCHEMA}".api_keys
            ORDER BY created_at
            """
        )
    return [dict(row) for row in rows]


async def revoke_key(prefix_or_hash: str) -> int:
    """Revoke every active key whose hash starts with `prefix_or_hash`; returns the count."""
    if len(prefix_or_hash) < 8:
        raise KeyPrefixError("key hash prefix must be at least 8 characters")
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
        print(
            f"{row['label']:<20} {status:<8} "
            f"created={row['created_at']} hash={row['key_hash'][:12]}..."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memory_base.serve.access.keys")
    subparsers = parser.add_subparsers(dest="command", required=True)

    new_parser = subparsers.add_parser("new", help="mint a key")
    new_parser.add_argument("label")

    subparsers.add_parser("list", help="list keys")

    revoke_parser = subparsers.add_parser("revoke", help="revoke a key")
    revoke_parser.add_argument("prefix_or_hash")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "new":
        plaintext = asyncio.run(new_key(args.label))
        print(f"API key for {args.label!r} (store this now, it will not be shown again):")
        print(plaintext)
    elif args.command == "list":
        _print_keys(asyncio.run(list_keys()))
    elif args.command == "revoke":
        try:
            count = asyncio.run(revoke_key(args.prefix_or_hash))
        except KeyPrefixError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"revoked {count} key(s)")


if __name__ == "__main__":
    main()
