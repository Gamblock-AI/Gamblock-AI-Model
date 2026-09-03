# Gamblock-AI Model

Repository dataset, training notebook, deployment artifacts, and evaluation
reports for Gamblock-AI's on-device hybrid detector.

## Detection contract

The detector combines two local signals:

1. Logistic Regression with Bag-of-Words text features from page title,
   headings, and DOM/content, plus 14 numeric URL features.
2. A rule-based matcher using `models/gambling_keywords.json`.

The exported artifact score is:

```text
hybrid_score = (0.80 * ml_probability) + (0.20 * rule_score)
```

The promoted deployment-aligned artifact threshold is `0.45`. Client protection authorities apply
an additional evidence policy: explicit URL/content rules remain decisive,
while model-only blocking requires committed page content whose text-only score
is independently suspicious. URL-shape evidence alone cannot block opaque
short links.

All inference and classification remain on-device. Raw URLs, DOM, screenshots,
and browsing history are not sent to the backend or any cloud service.

## Repository layout

```text
.
├── data/
│   ├── raw/
│   │   ├── judi.csv
│   │   ├── non_judi.csv
│   │   ├── html/{0,1}/       # captured page HTML by label
│   │   └── images/{0,1}/     # page images by label
│   └── processed/
│       ├── dataset_clean.csv
│       ├── split-manifest.json             # legacy baseline split metadata
│       └── splits/
│           ├── baseline_row_stratified_{train,test}.csv
│           ├── split-manifest.json         # canonical leakage-safe candidate split
│           └── {train,test}.csv
├── docs/
│   ├── ai/                   # repository AI context and manifest
│   └── integration/
│       └── model-integration.md
├── models/                   # deployment contract; filenames are stable
│   ├── gamblock_logistic_regression.onnx
│   ├── gamblock_hybrid_metadata.json
│   ├── gamblock_training_metadata.json
│   └── gambling_keywords.json
├── notebooks/
│   └── hybrid_model_training.ipynb
├── reports/
│   ├── evaluation/           # legacy source path; new results go to testing
│   └── tuning/
│       ├── hyperparameter_search.csv
│       └── hybrid_threshold_search.csv
├── scripts/
│   └── verify-ai-context.sh
└── requirements.txt
```

The `0` and `1` dataset folders represent `non_judi` and `judi` respectively.
CSV path columns use repository-relative POSIX paths, so they work consistently
on Linux, macOS, and Windows.

## Grouped evaluation evidence

Candidate training and evaluation use
`data/processed/splits/train.csv` and `test.csv`, which are created by:

```sh
python3 scripts/prepare_grouped_splits.py
```

This split connects duplicate model text, processed text, and registrable
domain/site family before assigning whole groups to train or test. Groups with
conflicting labels are excluded and recorded in
`data/processed/splits/split-manifest.json`.

## Deployment-projection candidate training

The Windows passive sensor is intentionally limited to title, `h1`-`h3`, and
anchor text. To avoid training a replacement model on full-page content and
deploying it on that narrower surface, use the explicit candidate workflow:

```sh
python3 scripts/train_deployment_projection.py --output-dir /tmp/gamblock-candidate
```

It derives validation only from grouped `train.csv`, freezes grouped `test.csv`
for one final evaluation, and writes no raw URL or DOM data. It searches five
client-compatible Logistic Regression configurations, adds label-preserving
camouflage variants only to the training frame in memory, gives extra weight to
short positive DOM samples, and selects a policy that passes every validation
target while maximizing robust recall within the 5% progress-evaluation FPR gate
buffer. Positive samples receive extra training weight, and the
character-substitution negative controls are retained during training to limit
false positives. The current active artifact is reported separately from the
grouped candidate and is not replaced automatically.

## Text-and-domain grouped evaluation

Run the deployment-aligned candidate evaluation through the testing runner:

```sh
python3 gamblock-ai-testing/docs/tools/run_evaluation.py \
  --workspace-root . --run-model-replay
```

The evaluator uses deterministic connected model-text and registrable-domain
grouping, excludes conflicting-label groups, selects policy only on grouped
validation data, freezes grouped final test data, and reports ablations,
short-DOM and camouflage robustness, threshold sensitivity, calibration, error
slices, repeated grouped validation, duplicate/leakage audits, split-manifest
integrity, offline speed, and ONNX parity. Repeated grouped validation is a
fixed-candidate stability check rather than a nested model-selection estimate.
It writes aggregate metrics, confidence intervals, split hashes, artifact
hashes, and four PNG visualizations only. Camouflage variants
are created in memory and used for training augmentation and evaluation; they
are not persisted as a dataset. Device-runtime remains an explicit scope
exclusion for this model progress report.

## Training workflow

Open `notebooks/hybrid_model_training.ipynb` from the repository root or from
the `notebooks/` directory. The notebook discovers the repository root, reads
raw CSV inputs, writes processed splits, trains and evaluates the model, and
exports candidate artifacts to the requested temporary output directory. The
canonical evaluation report and aggregate evidence are owned by
`gamblock-ai-testing`.

The notebook is an authoring workflow. Training and re-evaluation are
explicit opt-in operations; the checked-in ONNX artifact is the current
deployment output and must not be replaced casually.

## Validation

```sh
./scripts/verify-ai-context.sh --allow-untracked
```

Use strict mode after staging all required context files:

```sh
./scripts/verify-ai-context.sh
```

See [the integration guide](docs/integration/model-integration.md) for the
client-facing artifact contract.
