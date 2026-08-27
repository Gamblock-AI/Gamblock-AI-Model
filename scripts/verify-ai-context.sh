#!/usr/bin/env bash
set -euo pipefail

EXPECTED_CONTEXT_VERSION="2026-08-28.1"
ALLOW_UNTRACKED=false
ERRORS=0

usage() {
  cat <<'EOF'
Usage: scripts/verify-ai-context.sh [--allow-untracked]

Validate the repository-local AI context contract. Strict mode requires all
context files to be tracked. --allow-untracked is intended only while authoring
new context files locally.
EOF
}

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  ERRORS=$((ERRORS + 1))
}

for arg in "$@"; do
  case "$arg" in
    --allow-untracked)
      ALLOW_UNTRACKED=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
  printf 'ERROR: run this script inside a Git repository\n' >&2
  exit 1
fi
cd "$REPO_ROOT"

required_files=(
  ".gitattributes"
  ".agents/skills/verify-gamblock-change/SKILL.md"
  ".agents/skills/verify-gamblock-change/agents/openai.yaml"
  "AGENTS.md"
  "README.md"
  "docs/ai/README.md"
  "docs/ai/manifest.yaml"
  "CLAUDE.md"
  "GEMINI.md"
  ".github/copilot-instructions.md"
  ".cursor/rules/gamblock-ai.mdc"
  "scripts/verify-ai-context.sh"
)

for file in "${required_files[@]}"; do
  if [[ ! -f "$file" ]]; then
    fail "required context file is missing: $file"
    continue
  fi

  if [[ "$ALLOW_UNTRACKED" == false ]] &&
    ! git ls-files --error-unmatch -- "$file" >/dev/null 2>&1; then
    fail "required context file is not tracked: $file"
  fi

  if ! grep -Fq "  - $file" docs/ai/manifest.yaml; then
    fail "manifest does not list required file: $file"
  fi
done

if [[ ! -x "scripts/verify-ai-context.sh" ]]; then
  fail "scripts/verify-ai-context.sh must be executable"
fi

if ! grep -Fxq "context_version: \"$EXPECTED_CONTEXT_VERSION\"" \
  docs/ai/manifest.yaml && \
  ! grep -Fxq "context_version: '$EXPECTED_CONTEXT_VERSION'" \
  docs/ai/manifest.yaml && \
  ! grep -Fxq "context_version: $EXPECTED_CONTEXT_VERSION" \
  docs/ai/manifest.yaml; then
  fail "manifest context_version must be $EXPECTED_CONTEXT_VERSION"
fi

versioned_docs=("AGENTS.md" "docs/ai/README.md")
for file in "${versioned_docs[@]}"; do
  if ! grep -Fq "$EXPECTED_CONTEXT_VERSION" "$file"; then
    fail "$file does not declare context version $EXPECTED_CONTEXT_VERSION"
  fi
done

for file in CLAUDE.md GEMINI.md; do
  if ! grep -Fxq '@./AGENTS.md' "$file"; then
    fail "$file must import @./AGENTS.md"
  fi
done

if ! grep -Fq 'alwaysApply: true' .cursor/rules/gamblock-ai.mdc; then
  fail ".cursor rule must set alwaysApply: true"
fi
if ! grep -Fxq '@AGENTS.md' .cursor/rules/gamblock-ai.mdc && \
   ! grep -Fxq '@./AGENTS.md' .cursor/rules/gamblock-ai.mdc; then
  fail ".cursor rule must reference @AGENTS.md"
fi
if ! grep -Fq 'AGENTS.md' .github/copilot-instructions.md; then
  fail ".github/copilot-instructions.md must reference AGENTS.md"
fi

if ((ERRORS > 0)); then
  printf 'AI context verification failed with %d error(s).\n' "$ERRORS" >&2
  exit 1
fi

if [[ "$ALLOW_UNTRACKED" == true ]]; then
  mode="authoring (untracked files allowed)"
else
  mode="strict (tracked files required)"
fi
printf 'AI context verification passed: version=%s, mode=%s\n' \
  "$EXPECTED_CONTEXT_VERSION" "$mode"
