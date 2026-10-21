#!/usr/bin/env bash
# Usage: scripts/worktree.sh <branch>; creates a worktree from origin/main with the primary .env linked.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: worktree.sh <branch>" >&2
  exit 2
fi

branch=$1
primary=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
worktree="$primary.wt/${branch//\//-}"

git -C "$primary" fetch -q origin
git -C "$primary" worktree add -q --no-track -b "$branch" "$worktree" origin/main

if [[ -e "$primary/.env" && ! -e "$worktree/.env" ]]; then
  ln -s "$primary/.env" "$worktree/.env"
fi

echo "$worktree"
