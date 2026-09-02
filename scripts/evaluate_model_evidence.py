#!/usr/bin/env python3
"""Build privacy-safe, reproducible evidence for the frozen model snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "accuracy_min": 0.90,
    "precision_min": 0.90,
    "recall_min": 0.90,
    "f1_min": 0.90,
    "false_positive_rate_max": 0.05,
}


def _raise_csv_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_raise_csv_limit()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def host(value: str) -> str:
    try:
        hostname = (urlsplit(value).hostname or "").lower()
    except ValueError:
        return ""
    return hostname.removeprefix("www.")


def label_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(row.get("label", "") for row in rows).items()))


def snapshot(path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    identifiers = [row.get("id", "") for row in rows]
    addresses = [row.get("url", "") for row in rows]
    return {
        "sha256": sha256(path),
        "rows": len(rows),
        "label_counts": label_counts(rows),
        "unique_id_count": len(set(identifiers)),
        "duplicate_id_rows": len(identifiers) - len(set(identifiers)),
        "unique_url_count": len(set(addresses)),
        "duplicate_url_rows": len(addresses) - len(set(addresses)),
    }


def wilson(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    pairs = [(int(row["label"]), int(row["prediction"])) for row in rows]
    tp = sum(actual == guess == 1 for actual, guess in pairs)
    tn = sum(actual == guess == 0 for actual, guess in pairs)
    fp = sum(actual == 0 and guess == 1 for actual, guess in pairs)
    fn = sum(actual == 1 and guess == 0 for actual, guess in pairs)
    accuracy = (tp + tn) / len(pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    values = {
        "samples": len(pairs),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": false_positive_rate,
        "precision_ci95_wilson": wilson(tp, tp + fp),
        "recall_ci95_wilson": wilson(tp, tp + fn),
        "false_positive_rate_ci95_wilson": wilson(fp, fp + tn),
    }
    values["target_checks"] = {
        "accuracy": accuracy >= TARGETS["accuracy_min"],
        "precision": precision >= TARGETS["precision_min"],
        "recall": recall >= TARGETS["recall_min"],
        "f1_score": f1 >= TARGETS["f1_min"],
        "false_positive_rate": false_positive_rate <= TARGETS["false_positive_rate_max"],
    }
    values["numeric_gate_passed"] = all(values["target_checks"].values())
    return values


def build_report(
    root: Path = REPOSITORY_ROOT,
    prediction_path: Path | None = None,
) -> dict[str, Any]:
    raw_judi_path = root / "data/raw/judi.csv"
    raw_non_judi_path = root / "data/raw/non_judi.csv"
    clean_path = root / "data/processed/dataset_clean.csv"
    train_path = root / "data/processed/splits/baseline_row_stratified_train.csv"
    test_path = root / "data/processed/splits/baseline_row_stratified_test.csv"
    if prediction_path is None:
        raise ValueError("prediction_path is required; use the testing repository private staging input")
    model_path = root / "models/gamblock_logistic_regression.onnx"
    rules_path = root / "models/gambling_keywords.json"
    dataset_card_path = root / "data/dataset-card.json"
    split_manifest_path = root / "data/processed/split-manifest.json"

    raw_judi = read_csv(raw_judi_path)
    raw_non_judi = read_csv(raw_non_judi_path)
    clean = read_csv(clean_path)
    train = read_csv(train_path)
    test = read_csv(test_path)
    predictions = read_csv(prediction_path)

    test_by_id = {row["id"]: row for row in test}
    prediction_by_id = {row["id"]: row for row in predictions}
    prediction_alignment = (
        set(test_by_id) == set(prediction_by_id)
        and all(test_by_id[key]["label"] == prediction_by_id[key]["label"] for key in test_by_id)
    )

    raw_ids = {row.get("id", "") for row in raw_judi + raw_non_judi}
    clean_ids = {row.get("id", "") for row in clean}
    train_ids = {row.get("id", "") for row in train}
    test_ids = {row.get("id", "") for row in test}
    train_urls = {row.get("url", "") for row in train}
    test_urls = {row.get("url", "") for row in test}
    train_hosts = {host(row.get("url", "")) for row in train} - {""}
    test_hosts = {host(row.get("url", "")) for row in test} - {""}
    overlapping_hosts = train_hosts & test_hosts
    host_isolated_predictions = [
        row for row in predictions if host(row.get("url", "")) not in overlapping_hosts
    ]

    dataset_card = json.loads(dataset_card_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    provenance_complete = dataset_card.get("provenance_status") == "complete"
    lineage_gap = len(raw_ids - clean_ids)
    audit_checks = {
        "prediction_alignment": prediction_alignment,
        "clean_split_partition": clean_ids == train_ids | test_ids and not train_ids & test_ids,
        "train_test_url_isolation": not train_urls & test_urls,
        "train_test_exact_host_isolation": not overlapping_hosts,
        "raw_to_clean_lineage_complete": lineage_gap == 0,
        "dataset_provenance_complete": provenance_complete,
    }

    return {
        "schema_version": 1,
        "report_kind": "gamblock_model_evidence",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_maturity": "verified" if all(audit_checks.values()) else "provisional",
        "targets": TARGETS,
        "dataset": {
            "raw": {
                "total_rows": len(raw_judi) + len(raw_non_judi),
                "judi_rows": len(raw_judi),
                "non_judi_rows": len(raw_non_judi),
                "judi_sha256": sha256(raw_judi_path),
                "non_judi_sha256": sha256(raw_non_judi_path),
            },
            "clean": snapshot(clean_path, clean),
            "train": snapshot(train_path, train),
            "test": snapshot(test_path, test),
            "historical_baseline": {
                "method": "stratified random row split",
                "source": "baseline_row_stratified_train.csv and baseline_row_stratified_test.csv",
            },
            "validation_during_tuning": {
                "rows": 2074,
                "label_counts": {"0": 1404, "1": 670},
                "source": "notebook recorded output; final model was refit on the full train split",
            },
            "lineage_gap_rows": lineage_gap,
            "train_test_id_overlap_count": len(train_ids & test_ids),
            "train_test_url_overlap_count": len(train_urls & test_urls),
            "train_test_exact_host_overlap_count": len(overlapping_hosts),
            "dataset_card_sha256": sha256(dataset_card_path),
            "split_manifest_sha256": sha256(split_manifest_path),
        },
        "evaluation": {
            "prediction_snapshot_sha256": sha256(prediction_path),
            "all_test_rows": metrics(predictions),
            "exact_host_isolated_subset": metrics(host_isolated_predictions),
            "excluded_exact_host_overlap_rows": len(predictions) - len(host_isolated_predictions),
        },
        "artifacts": {
            "onnx": {
                "sha256": sha256(model_path),
                "bytes": model_path.stat().st_size,
                "mib": model_path.stat().st_size / 1024 / 1024,
                "under_5_mib": model_path.stat().st_size < 5 * 1024 * 1024,
            },
            "rules_sha256": sha256(rules_path),
        },
        "audit": {
            "checks": audit_checks,
            "passed": all(audit_checks.values()),
            "limitations": [
                "Dataset source, collection dates, license, and labeling governance are not recorded.",
                "The frozen row-level split contains exact hostname overlap and is not a domain-grouped final split.",
                "The checked-in predictions evaluate full cleaned content, not the bounded deployed DOM projection.",
            ],
        },
        "privacy": {
            "raw_url_or_dom_emitted": False,
            "participant_data_emitted": False,
        },
        "review": {"approved": False, "reviewer": None, "reviewed_at": None},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Aggregate evidence output path; use the testing repository evidence directory.",
    )
    parser.add_argument(
        "--prediction-input",
        type=Path,
        required=True,
        help="Local frozen prediction snapshot; raw content is never emitted in the report.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail when the audit is incomplete")
    args = parser.parse_args()
    report = build_report(REPOSITORY_ROOT, args.prediction_input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "maturity": report["evidence_maturity"],
        "numeric_gate_passed": report["evaluation"]["all_test_rows"]["numeric_gate_passed"],
        "audit_passed": report["audit"]["passed"],
    }, sort_keys=True))
    return 2 if args.strict and not report["audit"]["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
