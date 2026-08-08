# Gamblock-AI Model AI Context

Jika ada pertentangan dengan `pkm_proposal.md`, proposal PKM adalah sumber mutlak.

This repository is intentionally self-contained. A clone does not need a parent workspace to discover its product constraints, model architecture, or privacy rules.

Context version: `2026-08-09.2`

## Source hierarchy

1. `AGENTS.md` is the canonical source of repository instructions.
2. `docs/ai/manifest.yaml` declares the context version, required files, and contracts.
3. `CLAUDE.md`, `GEMINI.md`, `.github/copilot-instructions.md`, and `.cursor/rules/gamblock-ai.mdc` adapt supported tools to `AGENTS.md`.
4. `models/gamblock_hybrid_metadata.json` and `reports/` provide verified model parameters and metrics.

## Model and Evaluation Architecture

- **Method**: Hybrid Analysis (Rule-Based System + Logistic Regression).
- **ML Weight**: 0.75 (evaluated via Bag-of-Words on title and DOM/content, plus 14 numeric URL structural features).
- **Rule Weight**: 0.25 (evaluated via `models/gambling_keywords.json`).
- **Hybrid Threshold**: 0.4.
- **Deployment Artifacts**: `models/gamblock_logistic_regression.onnx` for lightweight on-device inference on Android and Windows clients.

## Performance Metrics

- Accuracy: 0.9738 (97.38%)
- Precision: 0.9638 (96.38%)
- Recall: 0.9546 (95.46%)
- F1-Score: 0.9592 (95.92%)

## Privacy Boundary

All inference and classification operations run strictly on-device (*Edge AI*). The model pipeline never transmits raw browsing history, raw DOM content, full URLs, or keystrokes to external servers or cloud APIs.

## Verification

```sh
scripts/verify-ai-context.sh --allow-untracked   # Authoring mode
scripts/verify-ai-context.sh                     # Strict mode (all files tracked)
```
