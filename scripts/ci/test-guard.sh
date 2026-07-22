#!/usr/bin/env bash
# Fails a PR that changes src/ Python code without touching tests/
# (AGENTS.md: tests accompany behavior). Bypass: `skip-tests` label.
set -euo pipefail

base=${1:?usage: test-guard.sh <base-ref>}

if [ "${TEST_GUARD_SKIP:-}" = "1" ]; then
  echo "skip-tests label present — guard bypassed."
  exit 0
fi

changed=$(git diff --name-only "$(git merge-base "$base" HEAD)" HEAD)
src_changed=$(printf '%s\n' "$changed" | grep -E '^src/.*\.py$' || true)
tests_changed=$(printf '%s\n' "$changed" | grep -E '^tests/.*\.py$' || true)

if [ -n "$src_changed" ] && [ -z "$tests_changed" ]; then
  echo "src/ changed without any tests/ change:"
  printf '%s\n' "$src_changed"
  echo
  echo "Ship the test in the same PR, or add the 'skip-tests' label with justification."
  exit 1
fi

echo "test-guard OK."
