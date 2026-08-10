#!/usr/bin/env bash
# PreToolUse(Bash) guard. Denies shell reads of .env (secret exposure)
# and git commit/push while on main (worktree → PR rule). The agent cannot
# commit/push to main; it must request the user. Fails OPEN on any error.
set -u

deny() {
  jq -cn --arg r "$1" \
    '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:"deny",permissionDecisionReason:$r}}'
  exit 0
}

input=$(cat 2>/dev/null) || exit 0
cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // empty' 2>/dev/null) || exit 0
[ -z "$cmd" ] && exit 0
cwd=$(printf '%s' "$input" | jq -r '.cwd // empty' 2>/dev/null)

if printf '%s' "$cmd" | grep -qE '(^|[;&|[:space:]])(cat|less|more|head|tail|bat|grep|rg|sed|awk|strings|base64|xxd|source)[^;&|]*\.env([^.A-Za-z0-9_-]|$)'; then
  deny ".env holds internal endpoints and DB credentials — reading it into the transcript is blocked. Check existence with ls; copy it into a worktree with cp without exposing contents."
fi

git_cmd=$(printf '%s' "$cmd" | sed "s/'[^']*'//g; s/\"[^\"]*\"//g") || git_cmd=""
if printf '%s' "$git_cmd" | grep -qE '(^|[;&|[:space:]])git(([[:space:]]+-[^[:space:];&|]+)*[[:space:]]+|[[:space:]]+-C[[:space:]]+([^[:space:];&|]+[[:space:]]+)?)(commit|push)([[:space:]]|$)'; then
  git_dir="${cwd:-.}"
  target=$(printf '%s' "$cmd" \
    | grep -oE '(^|[;&|[:space:]])git[[:space:]]+-C[[:space:]]+("[^"]*"|'"'"'[^'"'"']*'"'"'|[^[:space:];&|]+)' \
    | head -1 \
    | sed -E 's/^.*git[[:space:]]+-C[[:space:]]+//') || target=""
  if [ -z "$target" ]; then
    target=$(printf '%s' "$cmd" \
      | grep -oE '(^|[;&|])[[:space:]]*cd[[:space:]]+("[^"]*"|'"'"'[^'"'"']*'"'"'|[^[:space:];&|]+)' \
      | tail -1 \
      | sed -E 's/^.*cd[[:space:]]+//') || target=""
  fi
  case "$target" in
    \"*\") target=${target#\"}; target=${target%\"} ;;
    \'*\') target=${target#\'}; target=${target%\'} ;;
  esac
  if [ -n "$target" ] && ! printf '%s' "$target" | grep -q '[[:space:]]'; then
    case "$target" in
      /*) git_dir="$target" ;;
      *) git_dir="${cwd:-.}/$target" ;;
    esac
  fi
  # Repository scoping covers every worktree and defaults to strict when unidentified.
  common_of() { [ -n "$1" ] && git -C "$1" rev-parse --path-format=absolute --git-common-dir 2>/dev/null; }
  project_common=$(common_of "${CLAUDE_PROJECT_DIR:-}") || project_common=""
  target_common=$(common_of "$git_dir") || target_common=""
  if [ -n "$project_common" ] && [ -n "$target_common" ] &&
     [ "$project_common" != "$target_common" ]; then
    exit 0
  fi

  branch=$(git -C "$git_dir" branch --show-current 2>/dev/null) || branch=""
  if [ "$branch" = "main" ]; then
    deny "Current branch is main — work happens in a worktree and lands via PR (AGENTS.md). The agent cannot commit/push to main; request the user to run it directly."
  fi
fi

exit 0
