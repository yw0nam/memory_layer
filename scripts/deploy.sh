#!/usr/bin/env bash
# Deploy main: fast-forward, rebuild the api and mcp services, and wait until they are healthy.
# Usage: scripts/deploy.sh   (run from any directory; the checkout must be on main)
set -euo pipefail
cd "$(dirname "$0")/.."

[ "$(git branch --show-current)" = main ] || {
  echo "deploy.sh runs on the main checkout" >&2
  exit 1
}
git pull --ff-only
docker compose up -d --build --wait --wait-timeout 120 api mcp
echo "deployed $(git log --oneline -1); reconnect MCP clients (Claude Code: /mcp)"
