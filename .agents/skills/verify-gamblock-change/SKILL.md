---
name: verify-gamblock-change
description: Verify a Gamblock-AI repository change before handoff. Use when finishing or reviewing implementation, documentation, privacy-sensitive payloads, error codes, or cross-repository contracts and when reporting whether a change is ready.
---

# Verify a Gamblock-AI change

1. Read the repository `AGENTS.md` and `docs/ai/README.md`.
2. Inspect `git status` and the task-owned diff. Separate unrelated user changes
   from the files under verification.
3. Confirm the documented capability state matches evidence: distinguish
   `implemented`, `stub`, `not wired`, and `planned`.
4. Run `./scripts/verify-ai-context.sh`. During initial authoring of new,
   unstaged context files, run it with `--allow-untracked` and explicitly report
   that relaxed tracking check.
5. Run only the relevant repository linter/analyzer and AI-context validator by
   default. Do not run tests, builds, packaging, coverage, E2E, or composite
   full verification unless the user explicitly requests them in the current
   conversation. Do not claim an unrun check passed.
6. For privacy, error-catalog, API, or WebSocket changes, identify every related
   repository. Verify available siblings and name exact follow-ups for absent
   ones.
7. Check that README, `AGENTS.md`, `docs/ai/`, and capability status still match
   actual paths, commands, and behavior.
8. Report changed scope, commands and outcomes, limitations, and unresolved
   cross-repository work. Do not deploy, push, release, or mutate secrets.
