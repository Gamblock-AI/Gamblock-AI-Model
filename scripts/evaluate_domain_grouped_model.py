#!/usr/bin/env python3
"""Evaluate a deployment-aligned candidate with text-and-domain-isolated splits.

The command produces aggregate-only evidence.  It never writes raw rows,
URLs, DOM text, or predictions to the evidence output.  Candidate model
artifacts are written only when an explicit temporary path is supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from camouflage import CAMOUFLAGE_VARIANTS, camouflage_text
    from grouped_split import (
        assignment_hash,
        domain_group as canonical_domain_group,
        normalize_signature,
    )
    from grouped_split import stratified_group_split as canonical_group_split
except ModuleNotFoundError:  # Imported through a test/evaluator module path.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from camouflage import CAMOUFLAGE_VARIANTS, camouflage_text
    from grouped_split import (
        assignment_hash,
        domain_group as canonical_domain_group,
        normalize_signature,
    )
    from grouped_split import stratified_group_split as canonical_group_split


ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "accuracy_min": 0.90,
    "precision_min": 0.90,
    "recall_min": 0.90,
    "f1_score_min": 0.90,
    "false_positive_rate_max": 0.05,
}
KAGGLE_SOURCE = "https://www.kaggle.com/datasets/sahalmaghfud/illegal-web"
URL_FEATURES = [
    "url_length",
    "url_digit_count",
    "url_dot_count",
    "url_slash_count",
    "url_hyphen_count",
    "url_question_count",
    "url_equal_count",
    "url_keyword_count",
    "url_has_number",
    "url_has_https",
    "url_is_valid",
    "domain_length",
    "subdomain_length",
    "suffix_length",
]
CV_FOLDS = 5
CV_SEEDS = (42, 43, 44)
THRESHOLD_GRID = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
VISUAL_FILES = {
    "confusion_matrix": "domain_grouped_confusion_matrix.png",
    "ablation_metrics": "domain_grouped_ablation_metrics.png",
    "threshold_sensitivity": "domain_grouped_threshold_sensitivity.png",
    "calibration": "domain_grouped_calibration.png",
}
def load_trainer() -> Any:
    script = ROOT / "scripts/train_deployment_projection.py"
    spec = importlib.util.spec_from_file_location("deployment_projection_trainer", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load deployment trainer: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_integrity_audit(
    clean_path: Path,
    train_path: Path,
    test_path: Path,
    clean_rows: list[dict[str, str]],
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    split_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Verify that the evaluated split files still match their manifest."""
    clean_ids = [row.get("id", "") for row in clean_rows]
    train_ids = [row.get("id", "") for row in train_rows]
    test_ids = [row.get("id", "") for row in test_rows]
    clean_id_set = set(clean_ids)
    train_id_set = set(train_ids)
    test_id_set = set(test_ids)
    source_manifest = split_manifest.get("source", {})
    train_manifest = split_manifest.get("train", {})
    test_manifest = split_manifest.get("test", {})
    expected_excluded_rows = int(
        split_manifest.get("conflicting_groups_excluded", {}).get("rows", 0)
    )
    expected_eligible_rows = int(split_manifest.get("eligible", {}).get("rows", 0))
    outer_assignments = {
        **{row_id: "train" for row_id in train_ids},
        **{row_id: "test" for row_id in test_ids},
    }
    actual_excluded_rows = len(clean_id_set - train_id_set - test_id_set)
    checks = {
        "source_sha256_matches_manifest": sha256(clean_path) == source_manifest.get("dataset_clean_sha256"),
        "train_sha256_matches_manifest": sha256(train_path) == train_manifest.get("sha256"),
        "test_sha256_matches_manifest": sha256(test_path) == test_manifest.get("sha256"),
        "clean_ids_present": all(clean_ids),
        "train_ids_present": all(train_ids),
        "test_ids_present": all(test_ids),
        "clean_ids_unique": len(clean_ids) == len(clean_id_set),
        "train_ids_unique": len(train_ids) == len(train_id_set),
        "test_ids_unique": len(test_ids) == len(test_id_set),
        "train_test_ids_disjoint": not train_id_set & test_id_set,
        "split_ids_subset_of_clean": (train_id_set | test_id_set) <= clean_id_set,
        "train_row_count_matches_manifest": len(train_rows) == int(train_manifest.get("rows", -1)),
        "test_row_count_matches_manifest": len(test_rows) == int(test_manifest.get("rows", -1)),
        "eligible_row_count_matches_manifest": len(train_rows) + len(test_rows) == expected_eligible_rows,
        "excluded_row_count_matches_manifest": actual_excluded_rows == expected_excluded_rows,
        "clean_partition_matches_manifest": len(clean_rows) == expected_eligible_rows + expected_excluded_rows,
        "manifest_assignment_hash_matches": assignment_hash(outer_assignments)
        == split_manifest.get("assignment_sha256"),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not failed_checks else "failed",
        "checks": checks,
        "failed_checks": failed_checks,
        "counts": {
            "clean_rows": len(clean_rows),
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "actual_excluded_rows": actual_excluded_rows,
            "expected_excluded_rows": expected_excluded_rows,
            "expected_eligible_rows": expected_eligible_rows,
        },
    }


def stratified_group_folds(
    rows: list[dict[str, str]],
    fold_count: int,
    seed: int,
    group_field: str | None = None,
) -> list[tuple[list[dict[str, str]], list[dict[str, str]], dict[str, str]]]:
    """Assign homogeneous groups to balanced, deterministic folds."""
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group = row.get(group_field, "") if group_field else ""
        if not group:
            group = canonical_domain_group(row.get("url", ""), row.get("id", ""))
        groups[group].append(row)

    by_label: dict[int, list[tuple[str, list[dict[str, str]]]]] = defaultdict(list)
    for group, members in groups.items():
        by_label[int(members[0]["label"])].append((group, members))

    rng = random.Random(seed)
    fold_groups: list[list[str]] = [[] for _ in range(fold_count)]
    fold_label_rows: list[Counter[int]] = [Counter() for _ in range(fold_count)]
    fold_rows = [0 for _ in range(fold_count)]
    for label in sorted(by_label):
        candidates = list(by_label[label])
        rng.shuffle(candidates)
        candidates.sort(key=lambda item: len(item[1]), reverse=True)
        for group, members in candidates:
            target = min(
                range(fold_count),
                key=lambda index: (fold_label_rows[index][label], fold_rows[index], index),
            )
            fold_groups[target].append(group)
            fold_label_rows[target][label] += len(members)
            fold_rows[target] += len(members)

    fold_lookup = {
        group: fold_index
        for fold_index, group_names in enumerate(fold_groups)
        for group in group_names
    }
    folds: list[tuple[list[dict[str, str]], list[dict[str, str]], dict[str, str]]] = []
    for validation_fold in range(fold_count):
        train: list[dict[str, str]] = []
        validation: list[dict[str, str]] = []
        assignments: dict[str, str] = {}
        for group, members in sorted(groups.items()):
            split = "validation" if fold_lookup[group] == validation_fold else "train"
            for row in members:
                assignments[row["id"]] = split
                (validation if split == "validation" else train).append(row)
        folds.append((train, validation, assignments))
    return folds


def aggregate_fold_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not fold_metrics:
        return {"status": "pending", "reason": "no grouped folds were evaluated"}
    metric_names = ("accuracy", "precision", "recall", "f1_score", "false_positive_rate")
    summary: dict[str, Any] = {
        "status": "passed" if all(metric.get("numeric_gate_passed") for metric in fold_metrics) else "failed",
        "fold_count": len(fold_metrics),
        "numeric_gate_pass_rate": sum(metric.get("numeric_gate_passed", False) for metric in fold_metrics) / len(fold_metrics),
        "mean": {},
        "std": {},
        "minimum": {},
        "maximum": {},
    }
    for name in metric_names:
        values = [float(metric[name]) for metric in fold_metrics]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        summary["mean"][name] = mean
        summary["std"][name] = math.sqrt(variance)
        summary["minimum"][name] = min(values)
        summary["maximum"][name] = max(values)
    return summary


def threshold_sensitivity(
    labels: list[int],
    model_scores: list[float],
    content_scores: list[float],
    rules: list[float],
    has_dom: list[bool],
    policy: dict[str, float],
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for threshold in THRESHOLD_GRID:
        predictions = [
            int(
                (policy["ml_weight"] * model_score + policy["rule_weight"] * rule_score) >= threshold
                and (rule_score > 0.0 or (dom and content_score >= threshold))
            )
            for model_score, content_score, rule_score, dom in zip(
                model_scores, content_scores, rules, has_dom
            )
        ]
        metrics = metric_summary(labels, predictions)
        metrics["threshold"] = threshold
        metrics["selected"] = math.isclose(threshold, policy["threshold"])
        results.append(metrics)
    return {"selection_source": "grouped validation only", "selected_threshold": policy["threshold"], "results": results}


def calibration_summary(labels: list[int], scores: list[float], bin_count: int = 10) -> dict[str, Any]:
    if not labels or len(labels) != len(scores):
        return {"status": "pending", "reason": "empty calibration input"}
    bins: list[dict[str, Any]] = []
    brier = sum((score - label) ** 2 for score, label in zip(scores, labels)) / len(labels)
    expected_calibration_error = 0.0
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        selected = [
            (score, label)
            for score, label in zip(scores, labels)
            if (lower <= score < upper) or (index == bin_count - 1 and score == upper)
        ]
        if selected:
            mean_score = sum(score for score, _ in selected) / len(selected)
            observed_rate = sum(label for _, label in selected) / len(selected)
            expected_calibration_error += len(selected) / len(labels) * abs(mean_score - observed_rate)
        else:
            mean_score = None
            observed_rate = None
        bins.append({
            "lower": lower,
            "upper": upper,
            "samples": len(selected),
            "mean_confidence": mean_score,
            "observed_positive_rate": observed_rate,
        })
    return {
        "status": "reported",
        "samples": len(labels),
        "score_source": "ML probability on frozen grouped final test",
        "brier_score": brier,
        "expected_calibration_error": expected_calibration_error,
        "bins": bins,
    }


def duplicate_leakage_audit(
    rows: list[dict[str, str]],
    assignments: dict[str, str],
) -> dict[str, Any]:
    fields = {
        "normalized_url": "url_clean",
        "clean_text": "text_clean",
        "combined_text": "text_combined",
    }
    result: dict[str, Any] = {}
    for name, field in fields.items():
        values: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            value = normalize_signature(row.get(field, ""))
            if value:
                values[value].append(row["id"])
        duplicate_groups = [members for members in values.values() if len(members) > 1]
        cross_split = [
            members
            for members in duplicate_groups
            if len(
                {
                    assignments.get(row_id, "unknown")
                    for row_id in members
                    if assignments.get(row_id, "unknown") in {"train", "validation", "test"}
                }
            ) > 1
        ]
        result[name] = {
            "groups_with_duplicates": len(duplicate_groups),
            "duplicate_rows": sum(len(members) for members in duplicate_groups),
            "cross_split_duplicate_groups": len(cross_split),
        }
    result["audit_passed"] = not any(
        value["cross_split_duplicate_groups"]
        for value in result.values()
        if isinstance(value, dict)
    )
    result["normalization"] = "NFKC, casefold, and contiguous-whitespace normalization"
    return result


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


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


def metric_summary(actual: list[int], predicted: list[int]) -> dict[str, Any]:
    if len(actual) != len(predicted) or not actual:
        return {"status": "pending", "reason": "empty evaluation slice"}
    tp = sum(label == 1 and guess == 1 for label, guess in zip(actual, predicted))
    tn = sum(label == 0 and guess == 0 for label, guess in zip(actual, predicted))
    fp = sum(label == 0 and guess == 1 for label, guess in zip(actual, predicted))
    fn = sum(label == 1 and guess == 0 for label, guess in zip(actual, predicted))
    accuracy = (tp + tn) / len(actual)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    checks = {
        "accuracy": accuracy >= TARGETS["accuracy_min"],
        "precision": precision >= TARGETS["precision_min"],
        "recall": recall >= TARGETS["recall_min"],
        "f1_score": f1_score >= TARGETS["f1_score_min"],
        "false_positive_rate": false_positive_rate <= TARGETS["false_positive_rate_max"],
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "samples": len(actual),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "false_positive_rate": false_positive_rate,
        "precision_ci95_wilson": wilson(tp, tp + fp),
        "recall_ci95_wilson": wilson(tp, tp + fn),
        "false_positive_rate_ci95_wilson": wilson(fp, fp + tn),
        "target_checks": checks,
        "numeric_gate_passed": all(checks.values()),
    }


def _deployment_records_chunk(
    rows_and_keywords: tuple[list[dict[str, str]], list[str]],
) -> list[dict[str, Any]]:
    """Build one feature chunk in a worker so the snapshot is processed faster."""
    trainer = load_trainer()
    rows, keywords = rows_and_keywords
    return trainer.deployment_records(rows, keywords)


def deployment_records_parallel(
    rows: list[dict[str, str]],
    keywords: list[str],
) -> list[dict[str, Any]]:
    """Extract only the bounded deployment surface, using concurrent workers."""
    if len(rows) < 256:
        return _deployment_records_chunk((rows, keywords))
    worker_count = min(4, max(1, os.cpu_count() or 1))
    chunk_size = math.ceil(len(rows) / worker_count)
    chunks = [rows[start : start + chunk_size] for start in range(0, len(rows), chunk_size)]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = executor.map(
            _deployment_records_chunk,
            ((chunk, keywords) for chunk in chunks),
        )
        records: list[dict[str, Any]] = []
        for chunk_records in results:
            records.extend(chunk_records)
    return records


def predict_with_policy(
    trainer: Any,
    pipeline: Any,
    frame: Any,
    keywords: list[str],
    policy: dict[str, float],
    bundle: tuple[Any, ...],
) -> dict[str, Any]:
    model_scores, content_scores = trainer.client_scores(pipeline, frame, bundle)
    records = frame.to_dict("records")
    rules = trainer.rule_scores(records, keywords)
    has_dom = frame["has_dom_content"].tolist()
    hybrid = [
        int(
            (policy["ml_weight"] * model_score + policy["rule_weight"] * rule_score) >= policy["threshold"]
            and (rule_score > 0.0 or (dom and content_score >= policy["threshold"]))
        )
        for model_score, content_score, rule_score, dom in zip(model_scores, content_scores, rules, has_dom)
    ]
    return {
        "model_scores": model_scores,
        "content_scores": content_scores,
        "rules": rules,
        "hybrid": hybrid,
        "model_only": [int(score >= policy["threshold"]) for score in model_scores],
        "rule_only": [int(score > 0.0) for score in rules],
    }


def candidate_metrics(
    trainer: Any,
    pipeline: Any,
    frame: Any,
    keywords: list[str],
    policy: dict[str, float],
    bundle: tuple[Any, ...],
) -> dict[str, Any]:
    predictions = predict_with_policy(trainer, pipeline, frame, keywords, policy, bundle)
    actual = frame["label"].astype(int).tolist()
    return {
        "deployed_hybrid": metric_summary(actual, predictions["hybrid"]),
        "model_only": metric_summary(actual, predictions["model_only"]),
        "rule_only": metric_summary(actual, predictions["rule_only"]),
        "prediction_details": predictions,
    }


def build_surface_pipeline(bundle: tuple[Any, ...], config: dict[str, Any], surface: str) -> Any:
    _, _, _, ColumnTransformer, CountVectorizer, LogisticRegression, _, Pipeline, StandardScaler = bundle
    if surface == "url_only":
        transformers = [("url_features", StandardScaler(), URL_FEATURES)]
    elif surface == "dom_only":
        transformers = [
            (
                "text_bow",
                CountVectorizer(
                    max_features=config["max_features"],
                    ngram_range=(1, 2),
                    min_df=config["min_df"],
                    token_pattern=r"[a-zA-Z0-9_]+",
                ),
                "deployment_text",
            )
        ]
    else:
        raise ValueError(f"unknown ablation surface: {surface}")
    return Pipeline(
        steps=[
            ("preprocessor", ColumnTransformer(transformers=transformers)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    C=config["c"],
                    random_state=42,
                ),
            ),
        ]
    )


def select_binary_threshold(labels: list[int], scores: list[float]) -> tuple[float, dict[str, Any]]:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for threshold in THRESHOLD_GRID:
        metrics = metric_summary(labels, [int(score >= threshold) for score in scores])
        candidates.append((threshold, metrics))
    threshold, metrics = max(
        candidates,
        key=lambda item: (
            int(item[1]["false_positive_rate"] <= TARGETS["false_positive_rate_max"]),
            item[1]["recall"],
            item[1]["f1_score"],
            item[1]["precision"],
            -item[1]["false_positive_rate"],
        ),
    )
    return threshold, metrics


def surface_ablation_metrics(
    bundle: tuple[Any, ...],
    config: dict[str, Any],
    surface: str,
    train_frame: Any,
    validation_frame: Any,
    final_fit_frame: Any,
    test_frame: Any,
) -> dict[str, Any]:
    selection_pipeline = build_surface_pipeline(bundle, config, surface)
    selection_pipeline.fit(train_frame, train_frame["label"])
    validation_scores = selection_pipeline.predict_proba(validation_frame)[:, 1].tolist()
    threshold, validation_metrics = select_binary_threshold(
        validation_frame["label"].astype(int).tolist(), validation_scores
    )
    final_pipeline = build_surface_pipeline(bundle, config, surface)
    final_pipeline.fit(final_fit_frame, final_fit_frame["label"])
    test_scores = final_pipeline.predict_proba(test_frame)[:, 1].tolist()
    test_metrics = metric_summary(
        test_frame["label"].astype(int).tolist(),
        [int(score >= threshold) for score in test_scores],
    )
    test_metrics["selection_threshold"] = threshold
    test_metrics["selection_source"] = "grouped validation only"
    test_metrics["validation"] = {
        "threshold": threshold,
        "metrics": validation_metrics,
    }
    return test_metrics


def camouflage_frame(frame: Any, variant: str, keywords: list[str] | None = None, trainer: Any | None = None) -> Any:
    transformed = frame.copy()
    normalize = trainer.normalize_model_text if trainer is not None else lambda value, _: value
    transformed["deployment_text"] = transformed["deployment_text"].astype(str).map(
        lambda value: normalize(camouflage_text(value, variant), keywords or [])
    )
    transformed["has_dom_content"] = transformed["deployment_text"].astype(bool)
    return transformed


def camouflage_metrics(
    trainer: Any,
    pipeline: Any,
    frame: Any,
    keywords: list[str],
    policy: dict[str, float],
    bundle: tuple[Any, ...],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    labels = frame["label"].astype(int).tolist()
    for variant in CAMOUFLAGE_VARIANTS:
        transformed = camouflage_frame(frame, variant, keywords, trainer)
        metrics = candidate_metrics(trainer, pipeline, transformed, keywords, policy, bundle)
        results[variant] = metrics["deployed_hybrid"]
    return {
        "source": "frozen grouped final test rows with in-memory label-preserving transformations",
        "variants": results,
        "variant_count": len(CAMOUFLAGE_VARIANTS),
        "samples_per_variant": len(labels),
    }


def error_slice_metrics(
    frame: Any,
    predictions: list[int],
    reference_frame: Any,
) -> dict[str, Any]:
    slice_inputs = {
        "url_digit_count": (
            frame["url_digit_count"].astype(float).tolist(),
            reference_frame["url_digit_count"].astype(float).tolist(),
        ),
    }
    actual = frame["label"].astype(int).tolist()
    slices: dict[str, Any] = {}
    for name, (values, reference_values) in slice_inputs.items():
        lower = quantile(reference_values, 0.25)
        upper = quantile(reference_values, 0.75)
        for suffix, mask in (
            ("low", [value <= lower for value in values]),
            ("high", [value >= upper for value in values]),
        ):
            selected = [index for index, included in enumerate(mask) if included]
            slices[f"{name}_{suffix}"] = {
                "boundaries": {"q25": lower, "q75": upper},
                "samples": len(selected),
                "metrics": metric_summary(
                    [actual[index] for index in selected],
                    [predictions[index] for index in selected],
                ),
            }
    return slices


def offline_speed(
    trainer: Any,
    pipeline: Any,
    frame: Any,
    keywords: list[str],
    policy: dict[str, float],
    bundle: tuple[Any, ...],
) -> dict[str, Any]:
    predict_with_policy(trainer, pipeline, frame, keywords, policy, bundle)
    durations: list[float] = []
    for _ in range(5):
        started = time.perf_counter()
        predict_with_policy(trainer, pipeline, frame, keywords, policy, bundle)
        durations.append((time.perf_counter() - started) * 1000)
    durations.sort()
    samples = len(frame)
    percentile = lambda fraction: durations[min(len(durations) - 1, round((len(durations) - 1) * fraction))]
    return {
        "status": "reported",
        "scope": "offline model prediction on the evaluation host; excludes browser/UI/device latency",
        "samples_per_run": samples,
        "runs": len(durations),
        "mean_ms": sum(durations) / len(durations),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": max(durations),
        "mean_ms_per_sample": sum(durations) / len(durations) / samples if samples else None,
    }


def repeated_grouped_cv(
    trainer: Any,
    bundle: tuple[Any, ...],
    rows: list[dict[str, str]],
    frame: Any,
    keywords: list[str],
    config: dict[str, Any],
    policy: dict[str, float],
) -> dict[str, Any]:
    frame_by_id = frame.set_index("id")
    fold_metrics: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    for seed in CV_SEEDS:
        folds = stratified_group_folds(rows, CV_FOLDS, seed, "split_group_id")
        for fold_index, (train_rows, validation_rows, _) in enumerate(folds, start=1):
            train_ids = [row["id"] for row in train_rows]
            validation_ids = [row["id"] for row in validation_rows]
            train_frame = frame_by_id.loc[train_ids].reset_index()
            validation_frame = frame_by_id.loc[validation_ids].reset_index()
            pipeline = trainer.build_pipeline(bundle, config)
            short_cutoff = trainer.short_dom_cutoff(train_frame)
            augmented = trainer.augment_training_frame(train_frame, keywords, bundle)
            trainer.fit_pipeline(
                pipeline,
                augmented,
                trainer.training_sample_weights(augmented, short_cutoff),
            )
            metrics = candidate_metrics(
                trainer,
                pipeline,
                validation_frame,
                keywords,
                policy,
                bundle,
            )["deployed_hybrid"]
            fold_metrics.append(metrics)
            fold_records.append({"seed": seed, "fold": fold_index, "metrics": metrics})
    return {
        "method": "fixed deployment candidate evaluated on deterministic text-and-domain grouped folds",
        "folds": CV_FOLDS,
        "repetitions": len(CV_SEEDS),
        "total_evaluations": len(fold_metrics),
        "fixed_configuration": config,
        "fixed_policy": policy,
        "summary": aggregate_fold_metrics(fold_metrics),
        "fold_results": fold_records,
    }


def plot_visuals(
    plot_dir: Path,
    final_metrics: dict[str, Any],
    ablations: dict[str, dict[str, Any]],
    threshold_results: list[dict[str, Any]],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        return {"status": "blocked", "reason": f"matplotlib unavailable: {error}", "files": {}}

    plot_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Any] = {}

    confusion = final_metrics["confusion_matrix"]
    matrix = [[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]]
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title("Text-and-domain grouped final test confusion matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks([0, 1], ["non_judi", "judi"])
    axis.set_yticks([0, 1], ["non_judi", "judi"])
    for row_index in range(2):
        for column_index in range(2):
            axis.text(column_index, row_index, matrix[row_index][column_index], ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figures["confusion_matrix"] = figure

    names = list(ablations)
    metric_names = ("accuracy", "precision", "recall", "f1_score")
    figure, axis = plt.subplots(figsize=(10, 5))
    positions = list(range(len(names)))
    width = 0.18
    for offset, metric_name in enumerate(metric_names):
        values = [ablations[name].get(metric_name, 0.0) for name in names]
        axis.bar([position + offset * width for position in positions], values, width, label=metric_name)
    axis.set_title("Text-and-domain grouped ablation metrics")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.set_xticks([position + width * 1.5 for position in positions], names, rotation=20, ha="right")
    axis.legend()
    figure.tight_layout()
    figures["ablation_metrics"] = figure

    figure, axis = plt.subplots(figsize=(8, 5))
    thresholds = [result["threshold"] for result in threshold_results]
    for metric_name in ("precision", "recall", "f1_score", "false_positive_rate"):
        axis.plot(thresholds, [result[metric_name] for result in threshold_results], marker="o", label=metric_name)
    axis.set_title("Threshold sensitivity on grouped validation")
    axis.set_xlabel("Threshold")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figures["threshold_sensitivity"] = figure

    figure, axis = plt.subplots(figsize=(6, 5))
    bins = [item for item in calibration.get("bins", []) if item.get("mean_confidence") is not None]
    confidence = [item["mean_confidence"] for item in bins]
    observed = [item["observed_positive_rate"] for item in bins]
    axis.plot([0, 1], [0, 1], linestyle="--", label="ideal")
    axis.plot(confidence, observed, marker="o", label="model")
    axis.set_title("Model calibration on grouped final test")
    axis.set_xlabel("Mean confidence")
    axis.set_ylabel("Observed positive rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figures["calibration"] = figure

    artifacts: dict[str, Any] = {}
    for name, figure in figures.items():
        filename = VISUAL_FILES[name]
        path = plot_dir / filename
        figure.savefig(path, dpi=180, format="png")
        plt.close(figure)
        artifacts[name] = {"filename": filename, "sha256": sha256(path), "bytes": path.stat().st_size}
    return {"status": "created", "files": artifacts}


def onnx_parity(
    pipeline: Any,
    frame: Any,
    onnx_path: Path,
    threshold: float = 0.5,
) -> dict[str, Any]:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as error:
        return {"status": "blocked", "reason": f"ONNX runtime dependency unavailable: {error}"}

    try:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        feed: dict[str, Any] = {"deployment_text": frame["deployment_text"].astype(str).to_numpy().reshape(-1, 1)}
        for feature in pipeline.named_steps["preprocessor"].transformers_[1][2]:
            feed[feature] = frame[feature].astype("float32").to_numpy().reshape(-1, 1)
        outputs = session.run(None, feed)
        expected = pipeline.predict_proba(frame)[:, 1]
        observed: Any = None
        for value in outputs:
            if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] >= 2:
                observed = value[:, 1]
                break
            if isinstance(value, list) and value and isinstance(value[0], dict):
                observed = np.asarray([item.get(1, item.get("1", 0.0)) for item in value])
                break
        if observed is None:
            return {"status": "failed", "reason": "ONNX probability output was not found"}
        max_error = float(np.max(np.abs(expected - observed)))
        predicted_match = bool(np.array_equal(expected >= threshold, observed >= threshold))
        tolerance = 1e-5
        return {
            "status": "passed" if max_error <= tolerance and predicted_match else "failed",
            "samples": len(expected),
            "max_probability_absolute_error": max_error,
            "prediction_match": predicted_match,
            "decision_threshold": threshold,
            "tolerance": tolerance,
        }
    except Exception as error:  # pragma: no cover - backend-specific failure detail
        return {"status": "failed", "reason": f"ONNX parity execution failed: {error}"}


def build_evidence(
    output: Path,
    candidate_onnx: Path | None = None,
    plot_dir: Path | None = None,
    public_safe: bool = False,
) -> dict[str, Any]:
    trainer = load_trainer()
    bundle = trainer.dependencies()
    _, pd, _, _, _, _, _, _, _ = bundle
    rules_path = ROOT / "models/gambling_keywords.json"
    keywords = [trainer.normalize_for_rules(value) for value in json.loads(rules_path.read_text(encoding="utf-8"))]
    clean_path = ROOT / "data/processed/dataset_clean.csv"
    dataset_card_path = ROOT / "data/dataset-card.json"
    clean_rows = read_rows(clean_path)
    split_manifest_path = ROOT / "data/processed/splits/split-manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    candidate_train_path = ROOT / "data/processed/splits/train.csv"
    final_test_path = ROOT / "data/processed/splits/test.csv"
    candidate_train_rows = read_rows(candidate_train_path)
    final_test_rows = read_rows(final_test_path)
    split_integrity = split_integrity_audit(
        clean_path,
        candidate_train_path,
        final_test_path,
        clean_rows,
        candidate_train_rows,
        final_test_rows,
        split_manifest,
    )
    candidate_group_ids = {
        row["id"]: row["split_group_id"]
        for row in candidate_train_rows + final_test_rows
    }
    eligible_rows = candidate_train_rows + final_test_rows
    conflict_groups = [
        None
    ] * int(split_manifest.get("conflicting_groups_excluded", {}).get("groups", 0))
    train_rows, validation_rows, second_assignments = canonical_group_split(
        candidate_train_rows,
        candidate_group_ids,
        0.2,
        43,
    )
    model_train_rows = candidate_train_rows

    train_ids = {row["id"] for row in train_rows}
    validation_ids = {row["id"] for row in validation_rows}
    test_ids = {row["id"] for row in final_test_rows}
    assignments = {
        **{row["id"]: "test" for row in final_test_rows},
        **{row["id"]: "validation" if row["id"] in validation_ids else "train" for row in candidate_train_rows},
    }
    candidate_records = deployment_records_parallel(candidate_train_rows, keywords)
    candidate_records_by_id = {record["id"]: record for record in candidate_records}
    train_records = [candidate_records_by_id[row["id"]] for row in train_rows]
    validation_records = [candidate_records_by_id[row["id"]] for row in validation_rows]
    test_records = deployment_records_parallel(final_test_rows, keywords)
    train_frame = pd.DataFrame(train_records)
    validation_frame = pd.DataFrame(validation_records)
    test_frame = pd.DataFrame(test_records)
    validation_rules = trainer.rule_scores(validation_records, keywords)

    candidates: list[dict[str, Any]] = []
    short_cutoff = trainer.short_dom_cutoff(train_frame)
    for config in trainer.MODEL_CONFIGS:
        pipeline = trainer.build_pipeline(bundle, config)
        augmented_train = trainer.augment_training_frame(train_frame, keywords, bundle)
        trainer.fit_pipeline(
            pipeline,
            augmented_train,
            trainer.training_sample_weights(augmented_train, short_cutoff),
        )
        validation_model, validation_content = trainer.client_scores(pipeline, validation_frame, bundle)
        policy, selected_metrics = trainer.select_policy(
            validation_model,
            validation_content,
            validation_rules,
            validation_frame["has_dom_content"].tolist(),
            validation_frame["label"].astype(int).tolist(),
            validation_frame,
            keywords,
            short_cutoff,
            pipeline,
            bundle,
        )
        candidates.append({"configuration": config, "policy": policy, "validation": selected_metrics})

    feasible = [candidate for candidate in candidates if candidate["validation"].get("validation_target_feasible")]
    selected = max(feasible or candidates, key=lambda candidate: trainer.policy_rank(candidate["validation"]))
    policy = selected["policy"]
    selection_pipeline = trainer.build_pipeline(bundle, selected["configuration"])
    augmented_train = trainer.augment_training_frame(train_frame, keywords, bundle)
    trainer.fit_pipeline(
        selection_pipeline,
        augmented_train,
        trainer.training_sample_weights(augmented_train, short_cutoff),
    )
    selection_predictions = predict_with_policy(
        trainer,
        selection_pipeline,
        validation_frame,
        keywords,
        policy,
        bundle,
    )
    selection_thresholds = threshold_sensitivity(
        validation_frame["label"].astype(int).tolist(),
        selection_predictions["model_scores"],
        selection_predictions["content_scores"],
        selection_predictions["rules"],
        validation_frame["has_dom_content"].tolist(),
        policy,
    )
    final_pipeline = trainer.build_pipeline(bundle, selected["configuration"])
    final_fit_frame = pd.concat([train_frame, validation_frame], ignore_index=True)
    augmented_final = trainer.augment_training_frame(final_fit_frame, keywords, bundle)
    trainer.fit_pipeline(
        final_pipeline,
        augmented_final,
        trainer.training_sample_weights(augmented_final, short_cutoff),
    )
    validation_result = candidate_metrics(trainer, final_pipeline, validation_frame, keywords, policy, bundle)
    final_result = candidate_metrics(trainer, final_pipeline, test_frame, keywords, policy, bundle)

    test_predictions = final_result.pop("prediction_details")
    actual = test_frame["label"].astype(int).tolist()
    slice_results = error_slice_metrics(test_frame, test_predictions["hybrid"], final_fit_frame)

    outer_train_groups = {
        row.get("split_group_id", "") for row in model_train_rows if row.get("split_group_id")
    }
    outer_test_groups = {
        row.get("split_group_id", "") for row in final_test_rows if row.get("split_group_id")
    }
    inner_train_groups = {
        row.get("split_group_id", "") for row in train_rows if row.get("split_group_id")
    }
    inner_validation_groups = {
        row.get("split_group_id", "") for row in validation_rows if row.get("split_group_id")
    }
    outer_train_domains = {
        canonical_domain_group(row.get("url", ""), row.get("id", "")) for row in model_train_rows
    }
    outer_test_domains = {
        canonical_domain_group(row.get("url", ""), row.get("id", "")) for row in final_test_rows
    }
    inner_train_domains = {
        canonical_domain_group(row.get("url", ""), row.get("id", "")) for row in train_rows
    }
    inner_validation_domains = {
        canonical_domain_group(row.get("url", ""), row.get("id", "")) for row in validation_rows
    }
    overlap_checks = {
        "outer_train_test_group_overlap": len(outer_train_groups & outer_test_groups),
        "inner_train_validation_group_overlap": len(inner_train_groups & inner_validation_groups),
        "outer_train_test_domain_overlap": len(outer_train_domains & outer_test_domains),
        "inner_train_validation_domain_overlap": len(inner_train_domains & inner_validation_domains),
    }
    eligible_ids = {row["id"] for row in eligible_rows}
    audit_assignments = {
        **assignments,
        **{
            row["id"]: "conflicting_group"
            for row in clean_rows
            if row["id"] not in eligible_ids
        },
    }
    leakage_audit = duplicate_leakage_audit(clean_rows, audit_assignments)
    manifest_audit = split_manifest.get("leakage_audit", {})
    for name, public_name in (("model_text", "model_text"), ("domain", "site_group")):
        if isinstance(manifest_audit.get(name), dict):
            details = manifest_audit[name]
            leakage_audit[public_name] = {
                "groups_with_duplicates": details.get("groups_with_duplicates", 0),
                "duplicate_rows": details.get("duplicate_rows", 0),
                "cross_split_duplicate_groups": details.get("cross_split_groups", 0),
            }
    leakage_audit["audit_passed"] = bool(
        manifest_audit.get("audit_passed") and leakage_audit.get("audit_passed")
    )

    parity: dict[str, Any]
    artifact: dict[str, Any]
    if candidate_onnx is None:
        parity = {"status": "pending", "reason": "candidate ONNX output was not requested"}
        artifact = {"status": "not_requested"}
    else:
        candidate_onnx.parent.mkdir(parents=True, exist_ok=True)
        trainer.export_onnx(final_pipeline, candidate_onnx)
        parity = onnx_parity(final_pipeline, test_frame, candidate_onnx, policy["threshold"])
        artifact = {"status": "created", "sha256": sha256(candidate_onnx), "bytes": candidate_onnx.stat().st_size}

    ablations: dict[str, dict[str, Any]] = {
        "deployed_hybrid": final_result["deployed_hybrid"],
        "model_only": final_result["model_only"],
        "rule_only": final_result["rule_only"],
        "url_only": surface_ablation_metrics(
            bundle,
            selected["configuration"],
            "url_only",
            train_frame,
            validation_frame,
            final_fit_frame,
            test_frame,
        ),
        "dom_only": surface_ablation_metrics(
            bundle,
            selected["configuration"],
            "dom_only",
            train_frame,
            validation_frame,
            final_fit_frame,
            test_frame,
        ),
    }
    camouflage = camouflage_metrics(
        trainer,
        final_pipeline,
        test_frame,
        keywords,
        policy,
        bundle,
    )
    calibration = calibration_summary(actual, test_predictions["model_scores"])
    speed = offline_speed(trainer, final_pipeline, test_frame, keywords, policy, bundle)
    stability = repeated_grouped_cv(
        trainer,
        bundle,
        model_train_rows,
        pd.concat([train_frame, validation_frame], ignore_index=True),
        keywords,
        selected["configuration"],
        policy,
    )
    if plot_dir is None:
        raise ValueError("plot_dir is required; use the testing repository visual evidence directory")
    visuals = plot_visuals(
        plot_dir,
        final_result["deployed_hybrid"],
        ablations,
        selection_thresholds["results"],
        calibration,
    )

    dataset_card = json.loads(dataset_card_path.read_text(encoding="utf-8"))
    source = {
        "kaggle_url": KAGGLE_SOURCE,
        "kaggle_owner": "sahalmaghfud",
        "local_snapshot": "data/processed/dataset_clean.csv",
        "local_snapshot_sha256": sha256(clean_path),
        "license_status": dataset_card.get("license", "unverified"),
        "license_note": "Kaggle license metadata and redistribution rights require direct verification; no license is inferred from a secondary mirror.",
    }
    if public_safe:
        source = {
            "dataset_reference": "local dataset card; source URL withheld from public evidence",
            "local_snapshot_sha256": source["local_snapshot_sha256"],
            "license_status": source["license_status"],
            "license_note": source["license_note"],
        }
    all_numeric_passed = bool(final_result["deployed_hybrid"].get("numeric_gate_passed"))
    audit_passed = (
        split_integrity["status"] == "passed"
        and not overlap_checks["outer_train_test_group_overlap"]
        and not overlap_checks["inner_train_validation_group_overlap"]
        and not overlap_checks["outer_train_test_domain_overlap"]
        and not overlap_checks["inner_train_validation_domain_overlap"]
        and leakage_audit["audit_passed"]
    )
    evidence = {
        "schema_version": 3,
        "report_kind": "text_and_domain_grouped_deployment_aligned_model_evidence",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_maturity": "verified" if audit_passed and all_numeric_passed and parity.get("status") == "passed" else "provisional",
        "source": source,
        "dataset": {
            "clean_rows": len(clean_rows),
            "eligible_rows": len(eligible_rows),
            "conflicting_group_count": len(conflict_groups),
            "conflicting_row_count": int(split_manifest.get("conflicting_groups_excluded", {}).get("rows", 0)),
            "group_count": int(split_manifest.get("eligible", {}).get("groups", 0)),
            "label_counts": {
                "judi": sum(int(row["label"]) == 1 for row in eligible_rows),
                "non_judi": sum(int(row["label"]) == 0 for row in eligible_rows),
            },
        },
        "split": {
            "method": "deterministic connected model-text and registrable-domain grouped stratification",
            "seed_outer": 42,
            "seed_inner": 43,
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "final_test_rows": len(final_test_rows),
            "final_test_frozen_before_threshold_selection": True,
            "assignment_sha256": assignment_hash(assignments),
            "overlap_checks": overlap_checks,
            "integrity_audit": split_integrity,
            "audit_passed": audit_passed,
        },
        "policy": {
            "ml_weight": policy["ml_weight"],
            "rule_weight": policy["rule_weight"],
            "threshold": policy["threshold"],
            "selection_source": "grouped validation only",
            "configuration": selected["configuration"],
        },
        "validation": {
            "metrics": {key: value for key, value in validation_result["deployed_hybrid"].items() if key != "status"},
            "candidate_count": len(candidates),
            "target_feasible": bool(selected["validation"].get("validation_target_feasible")),
        },
        "evaluation": {
            "final_test": final_result["deployed_hybrid"],
            "ablations": ablations,
            "slices": slice_results,
            "camouflage": camouflage,
            "threshold_sensitivity": selection_thresholds,
            "calibration": calibration,
            "offline_speed": speed,
            "repeated_grouped_cv": stability,
        },
        "artifacts": {"candidate_onnx": artifact, "visuals": visuals},
        "parity": parity,
        "leakage_audit": leakage_audit,
        "limitations": {
            "repeated_grouped_cv": "fixed-candidate stability evaluation; not a nested estimate of hyperparameter-selection generalization",
        },
        "scope_exclusions": {
            "runtime_device_evaluation": "out_of_scope for this model progress report",
        },
        "privacy": {
            "raw_url_or_dom_emitted": False,
            "raw_predictions_emitted": False,
            "participant_data_emitted": False,
            "camouflage_rows_persisted": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate-onnx", type=Path)
    parser.add_argument("--plot-dir", required=True, type=Path)
    parser.add_argument(
        "--public-safe",
        action="store_true",
        help="Omit source URLs and local paths from the aggregate evidence output.",
    )
    args = parser.parse_args()
    try:
        evidence = build_evidence(args.output, args.candidate_onnx, args.plot_dir, args.public_safe)
    except Exception as error:
        print(json.dumps({"status": "blocked", "reason": str(error)}, sort_keys=True))
        return 2
    print(json.dumps({
        "output": str(args.output),
        "evidence_maturity": evidence["evidence_maturity"],
        "numeric_gate_passed": evidence["evaluation"]["final_test"]["numeric_gate_passed"],
        "split_audit_passed": evidence["split"]["audit_passed"],
        "onnx_parity": evidence["parity"]["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
