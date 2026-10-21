#!/usr/bin/env bash
# Flags change-narrative vocabulary in markdown; shared by the edit hook and the CI docs-guard job.
set -u

# Lines quoting the vocabulary list itself (e.g. the AGENTS.md rule) are skipped.
SKIP='제거/대체/축소'
VOCAB='더 이상|이전엔|이전에는|기존에는|제거(했|됐|되었)|대체(했|됐|되었)|축소(했|됐|되었)|추가했다|no longer'

usage() {
  echo "usage: docs-guard.sh [base-ref] (no argument checks stdin text)" >&2
  echo "exit: 0 clean, 1 flagged, 2 usage or git error" >&2
  exit 2
}

[ $# -le 1 ] || usage

if [ $# -eq 0 ]; then
  hits=$(awk -v skip="$SKIP" -v vocab="$VOCAB" \
    'index($0, skip) == 0 && $0 ~ vocab { print NR ":" $0 }' | head -5)
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

# -M keeps rename detection on; a renamed file is checked only for the lines it adds.
diff=$(git -c core.quotePath=false diff -U0 --no-color -M "$base_commit" HEAD -- '*.md' \
  ':(exclude,glob)**/node_modules/**' \
  ':(exclude,glob)**/.github/**' \
  ':(exclude,glob)**/.claude/**' \
  ':(exclude,glob)**/CHANGELOG.md') || {
  echo "docs-guard: git diff failed for base ref: $base" >&2
  exit 2
}

# Emit "<path>: <added line>" for added lines that hit the vocabulary.
hits=$(printf '%s\n' "$diff" | awk -v skip="$SKIP" -v vocab="$VOCAB" '
  /^diff --git / { path = ""; inhunk = 0; next }
  !inhunk {
    if ($0 ~ /^\+\+\+ \/dev\/null$/) path = ""
    else if ($0 ~ /^\+\+\+ b\//) {
      path = substr($0, 7)
      sub(/\t$/, "", path)
    }
    else if ($0 ~ /^@@/) inhunk = 1
    next
  }
  path != "" && /^\+/ {
    line = substr($0, 2)
    if (index(line, skip) == 0 && line ~ vocab) print path ": " line
  }
' | head -5)

if [ -n "$hits" ]; then
  printf '%s\n' "$hits"
  echo 'Docs are current-state only (AGENTS.md, "Docs: current-state only"); describe what the system is.'
  exit 1
fi

echo "docs-guard OK."
exit 0
