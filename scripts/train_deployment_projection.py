#!/usr/bin/env python3
"""Train a candidate Hybrid-v2 model on the bounded extension DOM surface.

The existing artifact was trained on full cleaned page content, whereas the
Windows sensor intentionally supplies only title, heading, and anchor text.
This script builds that same bounded input locally from the checked-in HTML,
selects the hybrid policy on a validation split taken only from ``train.csv``,
then evaluates the frozen ``test.csv`` once. It writes aggregate metrics and
artifacts to an explicit output directory; it never uploads or emits raw URLs
or DOM snapshots.

The candidate is intentionally not promoted to ``models/`` or the Android /
Windows app assets by this command. Promotion requires review of the generated
validation and final-test evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
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
TARGETS = {
    "accuracy_min": 0.95,
    "precision_min": 0.95,
    "recall_min": 0.95,
    "f1_score_min": 0.95,
    "false_positive_rate_max": 0.02,
}
VALIDATION_FPR_BUFFER_MAX = 0.015


MODEL_CONFIGS = (
    {"name": "baseline", "max_features": 10_000, "min_df": 3, "c": 0.05},
    {"name": "expanded_vocabulary", "max_features": 15_000, "min_df": 2, "c": 0.05},
    {"name": "expanded_vocabulary_c_0_1", "max_features": 15_000, "min_df": 2, "c": 0.10},
    {"name": "stable_vocabulary_c_0_1", "max_features": 10_000, "min_df": 3, "c": 0.10},
    {"name": "stable_vocabulary_c_0_2", "max_features": 10_000, "min_df": 3, "c": 0.20},
)


def dependencies() -> tuple[Any, ...]:
    try:
        import numpy as np
        import pandas as pd
        from scipy.special import expit
        from sklearn.compose import ColumnTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise RuntimeError(
            "Training dependencies are missing. Install requirements into an isolated "
            "environment before running this explicit training command."
        ) from error
    return (
        np,
        pd,
        expit,
        ColumnTransformer,
        CountVectorizer,
        LogisticRegression,
        train_test_split,
        Pipeline,
        StandardScaler,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def truncate_utf8(value: str, maximum_bytes: int) -> str:
    if len(value.encode("utf-8")) <= maximum_bytes:
        return value
    result: list[str] = []
    used = 0
    for character in value:
        size = len(character.encode("utf-8"))
        if used + size > maximum_bytes:
            break
        result.append(character)
        used += size
    return "".join(result)


class DOMExtractor(HTMLParser):
    """Matches the supported passive sensor: title, h1-h3, and anchor text."""

    tracked_tags = {"title", "h1", "h2", "h3", "a"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, list[str]]] = []
        self.active: list[tuple[str, list[str]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag in self.tracked_tags:
            block = (tag, [])
            self.blocks.append(block)
            self.active.append(block)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        for index in range(len(self.active) - 1, -1, -1):
            if self.active[index][0] == tag:
                del self.active[index]
                break

    def handle_data(self, data: str) -> None:
        for _, values in self.active:
            values.append(data)

    def text(self) -> str:
        title = ""
        headings: list[str] = []
        anchors: list[str] = []
        for tag, values in self.blocks:
            value = "".join(values).strip()
            if not value:
                continue
            if tag == "title" and not title:
                title = value
            elif tag in {"h1", "h2", "h3"} and len(headings) < 10:
                headings.append(value)
            elif tag == "a" and len(value) < 200 and len(anchors) < 50:
                anchors.append(value)
        bounded_title = truncate_utf8(title, 512).strip()[:512]
        bounded_headings = [truncate_utf8(value, 192).strip()[:256] for value in headings]
        bounded_anchors = [truncate_utf8(value, 160).strip()[:256] for value in anchors]
        return " ".join([bounded_title, *bounded_headings, *bounded_anchors]).strip()


def normalize_for_rules(value: str) -> str:
    normalized = []
    for character in value.lower():
        normalized.append(character if character.isascii() and (character.isalnum() or character == "_") else " ")
    return "".join(normalized).strip()


def contains_phrase(haystack: str, needle: str) -> bool:
    return bool(needle) and f" {needle} " in f" {haystack} "


def url_feature_values(url: str, keywords: list[str]) -> dict[str, float]:
    try:
        parsed = urlsplit(url)
        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        scheme, hostname = "", ""
    labels = [label for label in hostname.split(".") if label]
    suffix = labels[-1] if labels else ""
    subdomain = ".".join(labels[:-2]) if len(labels) > 2 else ""
    normalized_url = normalize_for_rules(url)
    keyword_count = sum(contains_phrase(normalized_url, keyword) for keyword in keywords)
    return {
        "url_length": float(len(url)),
        "url_digit_count": float(sum(character.isdigit() for character in url)),
        "url_dot_count": float(url.count(".")),
        "url_slash_count": float(url.count("/")),
        "url_hyphen_count": float(url.count("-")),
        "url_question_count": float(url.count("?")),
        "url_equal_count": float(url.count("=")),
        "url_keyword_count": float(keyword_count),
        "url_has_number": 1.0 if any(character.isdigit() for character in url) else 0.0,
        "url_has_https": 1.0 if scheme == "https" else 0.0,
        "url_is_valid": 1.0 if scheme in {"http", "https"} and bool(hostname) else 0.0,
        "domain_length": float(len(hostname)),
        "subdomain_length": float(len(subdomain)),
        "suffix_length": float(len(suffix)),
    }


def deployment_records(rows: Iterable[dict[str, str]], keywords: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        html_path = ROOT / row["html_path"]
        try:
            html = html_path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            raise RuntimeError(f"missing deployment HTML snapshot: {row['html_path']}") from error
        extractor = DOMExtractor()
        extractor.feed(html)
        extractor.close()
        text = extractor.text()
        record: dict[str, Any] = {
            "id": row["id"],
            "label": int(row["label"]),
            "url": truncate_utf8(row["url"], 2048)[:2048],
            "deployment_text": text,
            "has_dom_content": bool(text),
        }
        record.update(url_feature_values(record["url"], keywords))
        records.append(record)
    return records


def rule_scores(records: list[dict[str, Any]], keywords: list[str]) -> list[float]:
    scores: list[float] = []
    for record in records:
        evidence = normalize_for_rules(f"{record['url']} {record['deployment_text']}")
        scores.append(1.0 if any(contains_phrase(evidence, keyword) for keyword in keywords) else 0.0)
    return scores


def metric_summary(actual: list[int], predicted: list[int]) -> dict[str, Any]:
    tp = sum(label == 1 and decision == 1 for label, decision in zip(actual, predicted))
    tn = sum(label == 0 and decision == 0 for label, decision in zip(actual, predicted))
    fp = sum(label == 0 and decision == 1 for label, decision in zip(actual, predicted))
    fn = sum(label == 1 and decision == 0 for label, decision in zip(actual, predicted))
    accuracy = (tp + tn) / len(actual)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "samples": len(actual),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "false_positive_rate": fpr,
        "target_checks": {
            "accuracy": accuracy >= TARGETS["accuracy_min"],
            "precision": precision >= TARGETS["precision_min"],
            "recall": recall >= TARGETS["recall_min"],
            "f1_score": f1_score >= TARGETS["f1_score_min"],
            "false_positive_rate": fpr <= TARGETS["false_positive_rate_max"],
        },
    }


def policy_rank(metrics: dict[str, Any]) -> tuple[int, float, float, float, float, float]:
    """Prefer recall while retaining a validation FPR stability margin."""
    return (
        int(metrics["false_positive_rate"] <= VALIDATION_FPR_BUFFER_MAX),
        metrics["recall"],
        metrics["f1_score"],
        metrics["precision"],
        metrics["accuracy"],
        -metrics["false_positive_rate"],
    )


def select_policy(
    model_scores: list[float],
    content_scores: list[float],
    rules: list[float],
    has_dom: list[bool],
    labels: list[int],
) -> tuple[dict[str, float], dict[str, Any]]:
    candidates: list[tuple[dict[str, float], dict[str, Any]]] = []
    for ml_weight in (0.65, 0.70, 0.75, 0.80, 0.85):
        rule_weight = 1.0 - ml_weight
        for threshold in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
            predicted = [
                int(
                    (ml_weight * model_score + rule_weight * rule_score) >= threshold
                    and (rule_score > 0.0 or (dom and content_score >= threshold))
                )
                for model_score, content_score, rule_score, dom in zip(
                    model_scores, content_scores, rules, has_dom
                )
            ]
            metrics = metric_summary(labels, predicted)
            metrics["all_targets_passed"] = all(metrics["target_checks"].values())
            policy = {"ml_weight": ml_weight, "rule_weight": rule_weight, "threshold": threshold}
            candidates.append((policy, metrics))

    passing = [candidate for candidate in candidates if candidate[1]["all_targets_passed"]]
    buffered = [
        candidate
        for candidate in passing
        if candidate[1]["false_positive_rate"] <= VALIDATION_FPR_BUFFER_MAX
    ]
    pool = buffered or passing or candidates
    # All selection happens on the train-derived validation split. The FPR
    # buffer protects the 2% report gate while recall is maximized among safe
    # policies so that a precision-oriented historical setting cannot suppress
    # recoverable gambling detections.
    policy, metrics = max(pool, key=lambda candidate: policy_rank(candidate[1]))
    metrics = dict(metrics)
    metrics["validation_target_feasible"] = bool(passing)
    metrics["selection_pool"] = (
        "all_targets_with_fpr_buffer"
        if buffered
        else "all_targets"
        if passing
        else "best_effort"
    )
    return policy, metrics


def build_pipeline(dependencies_bundle: tuple[Any, ...], config: dict[str, Any]) -> Any:
    (
        _, _, _, ColumnTransformer, CountVectorizer, LogisticRegression,
        _, Pipeline, StandardScaler,
    ) = dependencies_bundle
    return Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        (
                            "text_bow",
                            CountVectorizer(
                                max_features=config["max_features"],
                                ngram_range=(1, 2),
                                min_df=config["min_df"],
                                token_pattern=r"(?u)[a-zA-Z0-9_]+",
                            ),
                            "deployment_text",
                        ),
                        ("url_features", StandardScaler(), URL_FEATURES),
                    ]
                ),
            ),
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


def client_scores(pipeline: Any, frame: Any, dependencies_bundle: tuple[Any, ...]) -> tuple[list[float], list[float]]:
    np, _, expit, _, _, _, _, _, _ = dependencies_bundle
    preprocessor = pipeline.named_steps["preprocessor"]
    vectorizer = preprocessor.named_transformers_["text_bow"]
    classifier = pipeline.named_steps["classifier"]
    text_matrix = vectorizer.transform(frame["deployment_text"])
    text_width = text_matrix.shape[1]
    content_linear = text_matrix @ classifier.coef_[0, :text_width] + classifier.intercept_[0]
    content_scores = expit(np.asarray(content_linear).reshape(-1))
    model_scores = pipeline.predict_proba(frame)[:, 1]
    return model_scores.tolist(), content_scores.tolist()


def export_onnx(pipeline: Any, path: Path) -> None:
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType, StringTensorType
    except ImportError as error:
        raise RuntimeError("skl2onnx is required to export a candidate ONNX artifact") from error
    initial_types = [("deployment_text", StringTensorType([None, 1]))]
    initial_types.extend((name, FloatTensorType([None, 1])) for name in URL_FEATURES)
    onnx_model = convert_sklearn(pipeline, initial_types=initial_types, target_opset=15)
    path.write_bytes(onnx_model.SerializeToString())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--allow-below-target",
        action="store_true",
        help="Write a candidate even if validation cannot satisfy every configured target.",
    )
    args = parser.parse_args()
    bundle = dependencies()
    _, pd, _, _, _, _, train_test_split, _, _ = bundle
    rules_path = ROOT / "models/gambling_keywords.json"
    keywords = [normalize_for_rules(value) for value in json.loads(rules_path.read_text(encoding="utf-8"))]
    train_rows = read_rows(ROOT / "data/processed/splits/train.csv")
    test_rows = read_rows(ROOT / "data/processed/splits/test.csv")
    train_records = deployment_records(train_rows, keywords)
    test_records = deployment_records(test_rows, keywords)
    train_frame = pd.DataFrame(train_records)
    test_frame = pd.DataFrame(test_records)
    train_base, validation = train_test_split(
        train_frame,
        test_size=0.2,
        random_state=42,
        stratify=train_frame["label"],
    )
    validation_rules = rule_scores(validation.to_dict("records"), keywords)
    tuning_candidates: list[dict[str, Any]] = []
    for config in MODEL_CONFIGS:
        base_pipeline = build_pipeline(bundle, config)
        base_pipeline.fit(train_base, train_base["label"])
        validation_model, validation_content = client_scores(base_pipeline, validation, bundle)
        policy, validation_metrics = select_policy(
            validation_model,
            validation_content,
            validation_rules,
            validation["has_dom_content"].tolist(),
            validation["label"].tolist(),
        )
        tuning_candidates.append(
            {
                "configuration": config,
                "policy": policy,
                "validation": validation_metrics,
            }
        )
    feasible_candidates = [
        candidate
        for candidate in tuning_candidates
        if candidate["validation"]["validation_target_feasible"]
    ]
    selected_candidate = max(
        feasible_candidates or tuning_candidates,
        key=lambda candidate: policy_rank(candidate["validation"]),
    )
    config = selected_candidate["configuration"]
    policy = selected_candidate["policy"]
    validation_metrics = selected_candidate["validation"]
    if not validation_metrics["validation_target_feasible"] and not args.allow_below_target:
        print(json.dumps({"validation": validation_metrics, "written": False}, sort_keys=True))
        return 2

    final_pipeline = build_pipeline(bundle, config)
    final_pipeline.fit(train_frame, train_frame["label"])
    test_model, test_content = client_scores(final_pipeline, test_frame, bundle)
    test_rules = rule_scores(test_records, keywords)
    threshold = policy["threshold"]
    test_prediction = [
        int(
            (policy["ml_weight"] * model_score + policy["rule_weight"] * rule_score) >= threshold
            and (rule_score > 0.0 or (dom and content_score >= threshold))
        )
        for model_score, content_score, rule_score, dom in zip(
            test_model, test_content, test_rules, test_frame["has_dom_content"].tolist()
        )
    ]
    test_metrics = metric_summary(test_frame["label"].tolist(), test_prediction)
    test_metrics["all_targets_passed"] = all(test_metrics["target_checks"].values())

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    onnx_path = output / "deployment_projection_candidate.onnx"
    metadata_path = output / "deployment_projection_metadata.json"
    report_path = output / "deployment_projection_evidence.json"
    export_onnx(final_pipeline, onnx_path)
    metadata = {
        "model_name": "Gamblock-AI deployment projection candidate",
        "model_source": "train.csv deployment DOM projection",
        "text_representation": "bounded title + h1-h3 + anchor text from passive extension contract",
        "random_state": 42,
        "training_configuration": config,
        "validation_selection": {
            "strategy": "all targets, then recall-first with 1.5% FPR buffer",
            "fpr_buffer_max": VALIDATION_FPR_BUFFER_MAX,
        },
        "ml_weight": policy["ml_weight"],
        "rule_weight": policy["rule_weight"],
        "threshold": threshold,
        "url_feature_columns": URL_FEATURES,
        "validation_metrics": validation_metrics,
        "evaluation_metrics": test_metrics,
        "status": "candidate_not_promoted",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "schema_version": 1,
        "report_kind": "deployment_projection_training_candidate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "privacy": {"raw_url_or_dom_emitted": False, "participant_data_emitted": False},
        "input_contract": "title + h1-h3 (max 10) + anchor text (max 50), UTF-8 bounded as extension payload",
        "split": {
            "train_rows": len(train_records),
            "validation_rows": len(validation),
            "final_test_rows": len(test_records),
            "final_test_used_for_selection": False,
        },
        "policy": policy,
        "training_configuration": config,
        "validation_selection": {
            "strategy": "all targets, then recall-first with 1.5% FPR buffer",
            "fpr_buffer_max": VALIDATION_FPR_BUFFER_MAX,
            "candidates": tuning_candidates,
        },
        "validation": validation_metrics,
        "final_test": test_metrics,
        "artifacts": {
            "onnx": {"path": onnx_path.name, "sha256": sha256(onnx_path), "bytes": onnx_path.stat().st_size},
            "metadata": {"path": metadata_path.name, "sha256": sha256(metadata_path)},
            "rules_sha256": sha256(rules_path),
        },
        "status": "candidate_not_promoted",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "validation_target_feasible": validation_metrics["validation_target_feasible"], "final_test_target_passed": test_metrics["all_targets_passed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
