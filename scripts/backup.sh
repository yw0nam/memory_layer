#!/usr/bin/env bash
# Dump the memory_base Postgres database to a gzipped file and prune old dumps.
# Usage: scripts/backup.sh <backup-dir> [keep]   keep: dumps retained, default 14.
set -euo pipefail
cd "$(dirname "$0")/.."

dir=${1:?usage: backup.sh <backup-dir> [keep]}
keep=${2:-14}
mkdir -p "$dir"
out="$dir/memory_base_$(date +%Y%m%d_%H%M%S).sql.gz"

docker compose exec -T db pg_dump -U memory memory_base | gzip > "$out" || {
  rm -f "$out"
  exit 1
}

ls -1t "$dir"/memory_base_*.sql.gz | tail -n +"$((keep + 1))" | xargs -r rm --
echo "$out"
