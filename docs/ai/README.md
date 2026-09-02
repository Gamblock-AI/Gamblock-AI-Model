# Gamblock-AI Model AI Context

Jika ada pertentangan dengan `pkm_proposal.md`, proposal PKM adalah sumber mutlak.

This repository is intentionally self-contained. A clone does not need a parent workspace to discover its product constraints, model architecture, or privacy rules.

Context version: `2026-09-02.5`

## Source hierarchy

1. `AGENTS.md` is the canonical source of repository instructions.
2. `docs/ai/manifest.yaml` declares the context version, required files, and contracts.
3. `CLAUDE.md`, `GEMINI.md`, `.github/copilot-instructions.md`, and `.cursor/rules/gamblock-ai.mdc` adapt supported tools to `AGENTS.md`.
4. `models/gamblock_hybrid_metadata.json` and `reports/evaluation/` provide
   model parameters and frozen prediction snapshots; their evidence maturity is
   determined by `scripts/evaluate_model_evidence.py`.

## Model and Evaluation Architecture

- **Method**: Hybrid Analysis (Rule-Based System + Logistic Regression).
- **ML Weight**: 0.80 (evaluated via Bag-of-Words on the bounded title, heading, and anchor-text surface, plus 14 numeric URL structural features).
- **Rule Weight**: 0.20 (evaluated via `models/gambling_keywords.json`).
- **Hybrid Threshold**: 0.45.
- **Deployment Artifacts**: `models/gamblock_logistic_regression.onnx` for lightweight on-device inference on Android and Windows clients. Stable client-facing artifact paths remain under `models/`; evaluation and tuning outputs are separated under `reports/evaluation/` and `reports/tuning/`.

## Frozen Snapshot Metrics

- Accuracy: 0.9738 (97.38%)
- Precision: 0.9638 (96.38%)
- Recall: 0.9546 (95.46%)
- F1-Score: 0.9592 (95.92%)

These values are reproducible for the checked-in full-content prediction
snapshot, but are provisional rather than a deployment claim: the dataset card
is incomplete and the frozen split is not domain-grouped. Use the evaluator to
emit counts, hashes, leakage checks, metrics, and audit maturity without
emitting raw browsing data:

```sh
python3 scripts/evaluate_model_evidence.py
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Deployment-aligned candidate workflow

`scripts/train_deployment_projection.py` reconstructs only the passive Windows
sensor surface (title, up to 10 `h1`-`h3` values, and up to 50 anchor texts)
from the local HTML snapshots. It selects the fusion policy solely from a
stratified validation subset of `train.csv`; it searches five
client-compatible Logistic Regression configurations and selects a policy that
passes every validation target, maximizes recall, and retains a 1.5% validation
FPR buffer. `test.csv` is held for its final report. The resulting ONNX artifact is explicitly marked
`candidate_not_promoted` and is never copied into `models/` or client assets by
the script. The latest candidate was manually promoted after all numeric
offline deployment targets and client-artifact parity passed. Its evidence
remains provisional until the domain-grouped/provenance and device-runtime
gates are complete.

## Privacy Boundary

All inference and classification operations run strictly on-device (*Edge AI*). The model pipeline never transmits raw browsing history, raw DOM content, full URLs, or keystrokes to external servers or cloud APIs.

## Cross-repository testing

Model replay and cross-repository evaluation results are published only in the
canonical [Gamblock-AI-Testing model report](https://github.com/Gamblock-AI/Gamblock-AI-Testing/blob/main/model/report.md).
This model snapshot documents artifacts and implementation status without
duplicating that report. When an explicit model evaluation is requested for
project evidence, the agent must synchronize `model/report.md` through the
testing runner and provide a test receipt listing public and private/local data
changes. Raw predictions and temporary replay files remain private/local.

## Verification

```sh
scripts/verify-ai-context.sh --allow-untracked   # Authoring mode
scripts/verify-ai-context.sh                     # Strict mode (all files tracked)
```
