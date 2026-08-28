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
hybrid_score = (0.75 * ml_probability) + (0.25 * rule_score)
```

The canonical artifact threshold is `0.4`. Client protection authorities apply
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
│       └── splits/{train,test}.csv
├── docs/
│   ├── ai/                   # repository AI context and manifest
│   └── integration/
│       └── model-integration.md
├── models/                   # deployment contract; filenames are stable
│   ├── gamblock_logistic_regression.onnx
│   ├── gamblock_logistic_regression.pkl
│   ├── gamblock_hybrid_metadata.json
│   ├── gamblock_training_metadata.json
│   └── gambling_keywords.json
├── notebooks/
│   └── hybrid_model_training.ipynb
├── reports/
│   ├── evaluation/
│   │   ├── classification_report.txt
│   │   ├── confusion_matrix.png
│   │   └── predictions.csv
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

## Training workflow

Open `notebooks/hybrid_model_training.ipynb` from the repository root or from
the `notebooks/` directory. The notebook discovers the repository root, reads
raw CSV inputs, writes processed splits, trains and evaluates the model, and
exports artifacts and reports to the directories shown above.

The notebook is an authoring workflow. Training and re-evaluation are
explicit opt-in operations; the checked-in ONNX/PKL artifacts are the current
deployment outputs and must not be replaced casually.

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
