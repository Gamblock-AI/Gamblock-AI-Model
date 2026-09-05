# Gamblock-AI Model AI Context

Jika ada pertentangan dengan `pkm_proposal.md`, proposal PKM adalah sumber mutlak.

This repository is intentionally self-contained. A clone does not need a parent workspace to discover its product constraints, model architecture, or privacy rules.

Context version: `2026-09-05.2`

## Source hierarchy

1. `AGENTS.md` is the canonical source of repository instructions.
2. `docs/ai/manifest.yaml` declares the context version, required files, and contracts.
3. `CLAUDE.md`, `GEMINI.md`, `.github/copilot-instructions.md`, and `.cursor/rules/gamblock-ai.mdc` adapt supported tools to `AGENTS.md`.
4. `models/gamblock_hybrid_metadata.json` provides model parameters. Permanent
   model evaluation evidence is owned by `gamblock-ai-testing/model/evidence/`
   and its maturity is determined by the testing-repository runner.

## Model and Evaluation Architecture

- **Method**: Hybrid Analysis (Rule-Based System + Logistic Regression).
- **ML Weight**: 0.80 (evaluated via Bag-of-Words on the bounded title, heading, and anchor-text surface, plus 14 numeric URL structural features).
- **Rule Weight**: 0.20 (evaluated via `models/gambling_keywords.json`).
- **Hybrid Threshold**: 0.45.
- **Deployment provenance and runtime artifacts**: `models/gamblock_logistic_regression.onnx` is the reproducible source artifact. Current Android/Windows authorities load the serialized Hybrid model/rules JSON under the client assets; its declared ONNX hash is checked as provenance. Stable model-source paths remain under `models/`; tuning outputs remain under `reports/tuning/`, while permanent evaluation outputs belong to `gamblock-ai-testing/model/evidence/`.

## Progress evaluation

The progress report covers the deployment-aligned projection and the grouped
candidate evaluation. Both use the bounded title, heading, and anchor-text
surface available to the passive Windows sensor and preserve aggregate-only
evidence in the testing repository:

```sh
python3 ../gamblock-ai-testing/docs/tools/run_evaluation.py \
  --workspace-root .. --run-model-replay
python3 -m unittest discover -s tests -p 'test_*.py'
```

The runner uses the single active target configuration and current progress
report. It does not select among parallel target files, and replay reuses the
existing artifact without retraining.

`developmental_checkpoint` is accuracy, precision, recall, and F1-score >=90%
with FPR <=5%, used for candidate screening and engineering regression. The
current progress gate uses the same 90%/5% boundary. These numeric gates do not
replace the proposal or automatically promote a candidate.

## Deployment-aligned candidate workflow

`scripts/train_deployment_projection.py` reconstructs only the passive Windows
sensor surface (title, up to 10 `h1`-`h3` values, and up to 50 anchor texts)
from the local HTML snapshots. The canonical candidate `train.csv` and
`test.csv` are connected-group splits built from duplicate model text,
processed text, and registrable domain/site family. Conflicting-label groups
are excluded and recorded in the split manifest. The trainer selects the
fusion policy solely from a grouped validation subset of `train.csv`; it
searches five client-compatible Logistic Regression configurations, augments
only the training frame with in-memory camouflage variants, weights short
positive DOM samples, and selects a policy using a recall/F1 robustness floor
within the 5% progress-evaluation FPR gate. `test.csv` is held for its final report.
The resulting ONNX artifact is explicitly marked `candidate_not_promoted` and
is never copied into `models/` or client assets by the script. The grouped
candidate is evaluated separately and is never promoted automatically.
Device-runtime is excluded from this progress report.

The grouped evaluator writes aggregate-only evidence to
`gamblock-ai-testing/model/evidence/aggregate/domain_grouped_evidence.json`.
The runner writes deployment aggregate evidence plus approved aggregate charts
under `gamblock-ai-testing/model/evidence/`. It checks deterministic
text-and-registrable-domain isolation, frozen grouped final-test metrics,
five-fold three-repetition grouped validation, ablations, short-DOM and
in-memory camouflage robustness, threshold sensitivity, calibration, error
slices, duplicate/leakage counts, split-manifest integrity, offline prediction
speed, confidence intervals, and ONNX parity. Repeated grouped validation is
a fixed-candidate stability check, not a nested estimate of model-selection
generalization. Four aggregate-only PNGs are written under
`gamblock-ai-testing/model/evidence/visuals/`.
It does not
emit raw URLs, DOM text, row identifiers, or predictions. Device-runtime is an
explicit scope exclusion for the current progress report.

## Privacy Boundary

All inference and classification operations run strictly on-device (*Edge AI*). The model pipeline never transmits raw browsing history, raw DOM content, full URLs, or keystrokes to external servers or cloud APIs.

## Cross-repository testing

Model evaluation and cross-repository results are published only in the
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
