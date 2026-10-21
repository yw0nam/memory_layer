#!/usr/bin/env bash
# PostToolUse(Write|Edit|NotebookEdit) guard. Blocks change-narrative
# vocabulary landing in markdown docs (docs are current-state only).
# Fails OPEN on any error.
set -u

input=$(cat 2>/dev/null) || exit 0
fp=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null) || exit 0
[ -z "$fp" ] && exit 0
text=$(printf '%s' "$input" | jq -r '.tool_input.content // .tool_input.new_string // empty' 2>/dev/null)

case "$fp" in */node_modules/*|*/.github/*|*/.claude/*|*/CHANGELOG.md) exit 0 ;; esac

# Lines quoting the vocabulary list itself (e.g. the AGENTS.md rule) are skipped.
project_dir=${CLAUDE_PROJECT_DIR:-}
case "$project_dir:$fp" in
  :*.md|"${project_dir%/}":"${project_dir%/}"/*.md)
    ;;
  *:*.md)
    fp=""
    ;;
esac
case "$fp" in
  *.md)
    hits=$(printf '%s' "$text" | bash "$(dirname "$0")/../../scripts/ci/docs-guard.sh")
    status=$?
    if [ "$status" -eq 1 ]; then
      jq -cn --arg b "$hits" \
        '{decision:"block",reason:("Intentional guard: LLMs habitually narrate the diff — \"was X, now Y\", \"previously\", \"no longer\", \"제거했다\" — and this hook deliberately blocks that. Docs here are current-state only: describe what the system IS now, declaratively, as if it had always been this way. No before/after, no change history (rule: AGENTS.md). Flagged:\n" + $b)}'
      exit 0
    fi
    # Status 0 (clean) or 2 (guard error) and a missing script all fail open.
    ;;
esac

exit 0