#!/usr/bin/env bash
# Create a worktree from origin/main with the primary checkout's .env linked.
# Usage: scripts/worktree.sh <branch>
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: worktree.sh <branch>" >&2
  exit 2
fi

branch=$1
primary=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
path="$primary.wt/${branch//\//-}"

git -C "$primary" fetch -q origin
git -C "$primary" worktree add -q -b "$branch" "$path" origin/main

if [[ -e "$primary/.env" ]]; then
  ln -s "$primary/.env" "$path/.env"
fi

echo "$path"
