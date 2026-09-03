# Gamblock-AI Model Agent Rules

This repository contains the dataset, training pipelines, model artifacts, metadata, and evaluation reports for Gamblock-AI machine learning components. It must remain safe and understandable as a standalone clone; no parent workspace files are required. Read `docs/ai/README.md` and `context/pkm_proposal.md` for background, capability status, and research contracts.

Context version: `2026-09-03.2`

## Start and finish

1. Inspect `git status` and preserve unrelated user changes.
2. Read the implementation, adjacent data/report files, and relevant README/context before editing.
3. Keep one model artifact or research behavior per change unit.
4. Default verification runs only the local AI context check. Do not run heavy notebook executions, training pipelines, or test runs unless the user explicitly requests an explicit test or full re-evaluation in the current conversation.
5. Update `README.md`, this file, and `docs/ai/` when model architectures, metadata, feature extraction schemas, thresholds, or privacy boundaries change.

## Architecture and Model Specifications

Gamblock-AI uses a **Hybrid Analysis** detection method combining:
- **Logistic Regression Model (ML Weight = 0.80)**: Evaluates Bag-of-Words from the bounded title, heading, and anchor-text surface supplied by the passive extension, combined with numeric URL features (length, digit count, symbols, domain properties).
- **Rule-Based System (Rule Weight = 0.20)**: Evaluates explicit keyword patterns using `gambling_keywords.json`.
- **Hybrid Decision Threshold**: `hybrid_score = (0.80 * ml_probability) + (0.20 * rule_score)`. If `hybrid_score >= 0.45`, the site is classified as gambling (`judi`), triggering local blocking and Pattern Interrupt interventions.

Client authorities add an evidence gate after calculating the artifact score:
explicit URL/content rules remain decisive, while model-only blocking requires
committed page content whose text-only score is independently suspicious. URL
shape features alone are supporting evidence and must not block opaque links.

Serialized artifacts:
- `models/gamblock_logistic_regression.onnx`: Exported ONNX model for on-device inference on Android and Windows clients.
- `models/gambling_keywords.json`: Keyword ruleset for the rule-based component.
- `models/gamblock_hybrid_metadata.json`: Canonical parameters, feature columns, weights, thresholds, and benchmark metrics.
- `models/gamblock_training_metadata.json`: Immutable metadata for the promoted training run.

The current progress-report gate is accuracy, precision, recall, and F1-score
at least 90%, with FPR at most 5%. This is a provisional progress checkpoint,
not a numeric target stated in the PKM proposal or an automatic promotion rule.

## Repository layout

- `notebooks/hybrid_model_training.ipynb`: Reproducible authoring notebook.
- `data/raw/`: Source CSV files and captured HTML/images grouped by label.
- `data/processed/`: Clean dataset and `splits/` train/test outputs.
- `models/`: Stable deployment artifacts and model metadata.
- `reports/evaluation/`: Legacy source path only; do not write new permanent
  evaluation results here. Canonical model evidence belongs in
  `gamblock-ai-testing/model/evidence/`, and raw local snapshots belong in its
  ignored `model/private/` directory.
- `reports/tuning/`: Hyperparameter and hybrid threshold search outputs.
- `scripts/train_deployment_projection.py`: Explicit candidate-only training
  workflow for the bounded title/heading/anchor surface actually supplied by
  the passive Windows sensor. It selects policy on a grouped train-derived
  validation split, uses train-only robustness augmentation, and never promotes
  an artifact automatically.
- `scripts/prepare_grouped_splits.py`: Rebuilds the canonical candidate split
  by connecting duplicate model text, processed text, and registrable domain;
  conflicting-label groups are excluded and counted in the manifest.
- `scripts/grouped_split.py`: Shared deterministic grouping and stratified
  whole-group assignment logic used by preparation and evaluation.
- `scripts/camouflage.py`: Shared deterministic in-memory camouflage
  transforms used identically by training and evaluation.
- `docs/integration/`: Client integration contract.

All retained model evaluations must be run through the testing-repository
runner. It writes aggregate JSON and approved aggregate visualizations under
`gamblock-ai-testing/model/evidence/`; it never writes permanent evaluation
outputs under this repository's `reports/evaluation/` path.

## Privacy Boundary

All classification and inference run strictly on-device (*Edge AI* / *On-Device AI*).
- The model training and inference specifications must never require transmitting raw browsing history, raw DOM content, full URLs, or keystrokes to external servers or cloud APIs.
- Exported ONNX models and metadata are deployed locally to native clients (Android/Windows) for offline, zero-latency, privacy-preserving inference.

## Validation Policy

```sh
./scripts/verify-ai-context.sh --allow-untracked   # Authoring mode
./scripts/verify-ai-context.sh                     # Strict verification
```

Explicit opt-in test requests:
- Explicit model re-evaluation or pipeline testing is only performed upon explicit user instruction.
  The deployment-aligned evaluation can be run without retraining:

  ```sh
  python3 ../gamblock-ai-testing/docs/tools/run_evaluation.py \
    --workspace-root .. --run-model-replay --run-model-tests
  python3 -m unittest discover -s tests -p 'test_*.py'
  python3 scripts/train_deployment_projection.py --output-dir /tmp/candidate
  ```

## Cross-repository testing handoff

When the user explicitly requests model testing to be retained as project
evidence, run the model evaluator through the testing repository runner with
`--run-model-replay`. A direct model command alone is not a completed evidence
handoff. Inspect both repositories and provide the test receipt required by
the umbrella/testing context, including `model/report.md` changes,
private/local artifacts, validation, and commit/push status. If the testing
checkout is unavailable, report the exact blocker and do not claim that the
model report was updated. Raw predictions, source rows, and temporary replay
outputs must remain private/local.
