# Gamblock-AI Model Agent Rules

This repository contains the dataset, training pipelines, model artifacts, metadata, and evaluation reports for Gamblock-AI machine learning components. It must remain safe and understandable as a standalone clone; no parent workspace files are required. Read `docs/ai/README.md` and `context/pkm_proposal.md` for background, capability status, and research contracts.

Context version: `2026-08-09.3`

## Start and finish

1. Inspect `git status` and preserve unrelated user changes.
2. Read the implementation, adjacent data/report files, and relevant README/context before editing.
3. Keep one model artifact or research behavior per change unit.
4. Default verification runs only the local AI context check. Do not run heavy notebook executions, training pipelines, or test runs unless the user explicitly requests an explicit test or full re-evaluation in the current conversation.
5. Update `README.md`, this file, and `docs/ai/` when model architectures, metadata, feature extraction schemas, thresholds, or privacy boundaries change.

## Architecture and Model Specifications

Gamblock-AI uses a **Hybrid Analysis** detection method combining:
- **Logistic Regression Model (ML Weight = 0.75)**: Evaluates page-content features via Bag-of-Words (BoW) from page title, headings, and DOM/content, combined with numeric URL features (length, digit count, symbols, domain properties).
- **Rule-Based System (Rule Weight = 0.25)**: Evaluates explicit keyword patterns using `gambling_keywords.json`.
- **Hybrid Decision Threshold**: `hybrid_score = (0.75 * ml_probability) + (0.25 * rule_score)`. If `hybrid_score >= 0.4`, the site is classified as gambling (`judi`), triggering local blocking and Pattern Interrupt interventions.

Serialized artifacts:
- `models/gamblock_logistic_regression.onnx`: Exported ONNX model for on-device inference on Android and Windows clients.
- `models/gamblock_logistic_regression.pkl`: Python Scikit-Learn pipeline for training and offline validation.
- `models/gambling_keywords.json`: Keyword ruleset for the rule-based component.
- `models/gamblock_hybrid_metadata.json`: Canonical parameters, feature columns, weights, thresholds, and benchmark metrics.

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
