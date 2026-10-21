#!/usr/bin/env bash
# Flags change-narrative vocabulary in markdown (AGENTS.md: docs are current-state
# only). Shared by the PostToolUse edit hook and the CI docs-guard job.
# Usage: docs-guard.sh            check stdin text
#        docs-guard.sh <base-ref> check lines added on HEAD since <base-ref>
# Exit: 0 clean, 1 flagged, 2 usage or git error (message on stderr).
set -u

# Lines quoting the vocabulary list itself (e.g. the AGENTS.md rule) are skipped.
SKIP='제거/대체/축소'
VOCAB='더 이상|이전엔|이전에는|기존에는|제거(했|됐|되었)|대체(했|됐|되었)|축소(했|됐|되었)|추가했다|no longer'

usage() {
  echo "usage: docs-guard.sh [base-ref] (no argument checks stdin text)" >&2
  exit 2
}

[ $# -le 1 ] || usage

if [ $# -eq 0 ]; then
  hits=$(grep -vE "$SKIP" | grep -nE "$VOCAB" | head -5)
  if [ -n "$hits" ]; then
    printf '%s\n' "$hits"
    exit 1
  fi
  exit 0
fi

base=$1
if ! base_commit=$(git merge-base "$base" HEAD 2>/dev/null); then
  echo "docs-guard: unknown base ref: $base" >&2
  exit 2
fi

diff=$(git -c core.quotePath=false diff -U0 --no-color "$base_commit" HEAD -- '*.md' \
  ':(exclude,glob)**/node_modules/**' \
  ':(exclude,glob)**/.github/**' \
  ':(exclude,glob)**/.claude/**' \
  ':(exclude,glob)**/CHANGELOG.md') || {
  echo "docs-guard: git diff failed for base ref: $base" >&2
  exit 2
}

file=""
in_hunk=false
hits=0
while IFS= read -r line; do
  if $in_hunk; then
    case "$line" in
      "+"*)
        [ -n "$file" ] || continue
        added=${line#'+'}
        case "$added" in
          *"$SKIP"*) ;;
          *)
            if printf '%s\n' "$added" | grep -qE "$VOCAB"; then
              printf '%s: %s\n' "$file" "$added"
              hits=$((hits + 1))
              [ "$hits" -ge 5 ] && break
            fi
            ;;
        esac
        ;;
    esac
  else
    # A file header runs from diff --git to its first @@.
    case "$line" in
      "diff --git "*) file="" ;;
      "+++ /dev/null") file="" ;;
      "+++ "*) file=${line#'+++ '} ;;
      "@@"*) in_hunk=true ;;
    esac
  fi
done <<< "$diff"

if [ "$hits" -gt 0 ]; then
  echo 'Docs are current-state only (AGENTS.md, "Docs: current-state only"); describe what the system is.'
  exit 1
fi

echo "docs-guard OK."
exit 0
