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
│       ├── split-manifest.json
│       └── splits/{train,test}.csv
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
│   ├── evaluation/
│   │   ├── classification_report.txt  # historical full-content snapshot
│   │   ├── confusion_matrix.png
│   │   ├── deployment_projection_evidence.json
│   │   └── predictions.csv
│   └── tuning/
│       ├── hyperparameter_search.csv
│       └── hybrid_threshold_search.csv
├── scripts/
│   ├── evaluate_model_evidence.py
│   └── verify-ai-context.sh
└── requirements.txt
```

The `0` and `1` dataset folders represent `non_judi` and `judi` respectively.
CSV path columns use repository-relative POSIX paths, so they work consistently
on Linux, macOS, and Windows.

## Frozen snapshot evidence

`data/dataset-card.json` and `data/processed/split-manifest.json` record the
facts that can be recovered from the checked-in snapshot. The raw snapshot has
12,964 rows (4,184 judi; 8,780 non-judi); its clean snapshot has 12,960 rows.
The frozen stratified split is 10,368 train rows (3,347 judi; 7,021 non-judi)
and 2,592 test rows (837 judi; 1,755 non-judi). Tuning notes record a 2,074-row
validation holdout before the final model was refit on the train split.

Generate a reproducible aggregate/hash-only report without retraining:

```sh
python3 scripts/evaluate_model_evidence.py
python3 -m unittest discover -s tests -p 'test_*.py'
```

The report intentionally remains `provisional`: dataset source/governance are
not recorded, four raw rows lack a clean-snapshot exclusion reason, and the
frozen split has two exact hostnames shared by train and test. Its high snapshot
metrics therefore do not constitute an evaluated deployment-runtime claim.

## Deployment-projection candidate training

The Windows passive sensor is intentionally limited to title, `h1`-`h3`, and
anchor text. To avoid training a replacement model on full-page content and
deploying it on that narrower surface, use the explicit candidate workflow:

```sh
python3 scripts/train_deployment_projection.py --output-dir /tmp/gamblock-candidate
```

It derives validation only from `train.csv`, freezes `test.csv` for one final
evaluation, and writes no raw URL or DOM data. It searches five
client-compatible Logistic Regression configurations, then selects a policy
that passes every validation target and maximizes recall while retaining a 1.5%
validation FPR buffer. The current artifact was manually promoted after the
frozen 2,592-row offline deployment projection passed every numeric target:
accuracy 97.22%, precision 96.25%, recall 95.10%, F1 95.67%, and FPR 1.77%.
This remains provisional offline evidence because domain-grouped split,
provenance, and device-runtime gates are incomplete.

## Training workflow

Open `notebooks/hybrid_model_training.ipynb` from the repository root or from
the `notebooks/` directory. The notebook discovers the repository root, reads
raw CSV inputs, writes processed splits, trains and evaluates the model, and
exports artifacts and reports to the directories shown above.

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
