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
import html as html_module
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

try:
    from camouflage import CAMOUFLAGE_VARIANTS, camouflage_text
    from grouped_split import stratified_group_split
except ModuleNotFoundError:  # Imported through a test/evaluator module path.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from camouflage import CAMOUFLAGE_VARIANTS, camouflage_text
    from grouped_split import stratified_group_split
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
    "accuracy_min": 0.90,
    "precision_min": 0.90,
    "recall_min": 0.90,
    "f1_score_min": 0.90,
    "false_positive_rate_max": 0.05,
}
VALIDATION_FPR_BUFFER_MAX = TARGETS["false_positive_rate_max"]
AUGMENTED_NEGATIVE_VARIANTS = {"character_substitution"}
POSITIVE_SAMPLE_WEIGHT = 1.25
SHORT_POSITIVE_SAMPLE_WEIGHT = 2.5
CAMOUFLAGE_NEGATIVE_SAMPLE_WEIGHT = 0.5
LEET_TRANSLATIONS = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
CONFUSABLE_TRANSLATIONS = str.maketrans({
    "а": "a", "е": "e", "і": "i", "о": "o", "с": "c", "ѕ": "s", "т": "t",
    "Α": "a", "Ε": "e", "Ι": "i", "Ο": "o", "Τ": "t",
})
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")
TRACKED_BLOCK_PATTERN = re.compile(
    r"<(title|h[1-3]|a)\b[^>]*>(.*?)</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
HTML_TAG_PATTERN = re.compile(r"<[^>]*>")


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


def deployment_text_from_html(html: str) -> str:
    """Extract the bounded passive surface without parsing unrelated page bytes.

    The checked-in snapshots can contain multi-megabyte scripts and styles. A
    targeted scan is equivalent for the valid, closed tracked elements used by
    the passive sensor and avoids walking every unrelated HTML token. The
    standards parser remains a fallback for malformed snapshots with no
    targeted matches.
    """
    blocks: list[tuple[str, str]] = []
    for match in TRACKED_BLOCK_PATTERN.finditer(html):
        tag = match.group(1).lower()
        value = html_module.unescape(HTML_TAG_PATTERN.sub(" ", match.group(2))).strip()
        if value:
            blocks.append((tag, value))
    if not blocks and html:
        extractor = DOMExtractor()
        extractor.feed(html)
        extractor.close()
        return extractor.text()

    title = ""
    headings: list[str] = []
    anchors: list[str] = []
    for tag, value in blocks:
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


def normalize_for_rules(value: str, keywords: Iterable[str] = ()) -> str:
    return normalize_model_text(value, keywords)


def _normalize_token(token: str) -> str:
    if not any(character.isalpha() for character in token):
        return token
    if not any(character in "013457@$" for character in token):
        return token
    return token.translate(LEET_TRANSLATIONS)


def normalize_model_text(value: str, keywords: Iterable[str] = ()) -> str:
    """Normalize Unicode/confusable text while keeping the client token contract."""
    value = unicodedata.normalize("NFKC", value or "").casefold().translate(CONFUSABLE_TRANSLATIONS)
    tokens = TOKEN_PATTERN.findall(value)
    normalized_tokens: list[str] = []
    for token in tokens:
        normalized = _normalize_token(token)
        normalized_tokens.append(token)
        if normalized != token:
            normalized_tokens.append(normalized)

    compact_keywords = {
        "".join(TOKEN_PATTERN.findall(normalize_model_text(keyword)))
        for keyword in keywords
        if keyword
    }
    index = 0
    while index < len(tokens):
        if len(tokens[index]) != 1:
            index += 1
            continue
        end = index
        while end < len(tokens) and len(tokens[end]) == 1:
            end += 1
        compact = "".join(tokens[index:end])
        if end - index >= 3 and compact in compact_keywords:
            normalized_tokens.append(compact)
        index = end
    return " ".join(normalized_tokens)


def contains_phrase(haystack: str, needle: str) -> bool:
    return bool(needle) and f" {needle} " in f" {haystack} "


def short_dom_cutoff(frame: Any) -> int:
    values = sorted(len(str(value)) for value in frame["deployment_text"])
    if not values:
        return 0
    position = (len(values) - 1) * 0.25
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return int(round(values[lower] + (values[upper] - values[lower]) * fraction))


def augment_training_frame(frame: Any, keywords: list[str], dependencies_bundle: tuple[Any, ...]) -> Any:
    """Add deterministic positive-class camouflage variants in memory only."""
    _, pd, _, _, _, _, _, _, _ = dependencies_bundle
    variants = [frame]
    for variant in CAMOUFLAGE_VARIANTS:
        source = frame if variant in AUGMENTED_NEGATIVE_VARIANTS else frame.loc[frame["label"].astype(int) == 1]
        transformed = source.copy()
        transformed["deployment_text"] = transformed["deployment_text"].astype(str).map(
            lambda value: normalize_model_text(camouflage_text(value, variant), keywords)
        )
        transformed["has_dom_content"] = transformed["deployment_text"].astype(bool)
        transformed["camouflage_variant"] = variant
        if not transformed.empty:
            variants.append(transformed)
    return pd.concat(variants, ignore_index=True)


def training_sample_weights(frame: Any, short_cutoff: int) -> list[float]:
    return [
        (
            SHORT_POSITIVE_SAMPLE_WEIGHT
            if int(label) == 1 and len(str(text)) <= short_cutoff
            else POSITIVE_SAMPLE_WEIGHT
            if int(label) == 1
            else CAMOUFLAGE_NEGATIVE_SAMPLE_WEIGHT
            if isinstance(row_variant, str) and row_variant
            else 1.0
        )
        for label, text, row_variant in zip(
            frame["label"],
            frame["deployment_text"],
            frame.get("camouflage_variant", [""] * len(frame)),
        )
    ]


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
    normalized_url = normalize_for_rules(url, keywords)
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
        text = normalize_model_text(deployment_text_from_html(html), keywords)
        record: dict[str, Any] = {
            "id": row["id"],
            "label": int(row["label"]),
            "url": truncate_utf8(row["url"], 2048)[:2048],
            "deployment_text": text,
            "has_dom_content": bool(text),
        }
        if row.get("split_group_id"):
            record["split_group_id"] = row["split_group_id"]
        record.update(url_feature_values(record["url"], keywords))
        records.append(record)
    return records


def rule_scores(records: list[dict[str, Any]], keywords: list[str]) -> list[float]:
    scores: list[float] = []
    for record in records:
        evidence = normalize_model_text(f"{record['url']} {record['deployment_text']}", keywords)
        scores.append(1.0 if any(contains_phrase(evidence, keyword) for keyword in keywords) else 0.0)
    return scores


def metric_summary(actual: list[int], predicted: list[int]) -> dict[str, Any]:
    if not actual:
        return {
            "status": "pending",
            "samples": 0,
            "reason": "slice contains no eligible rows",
        }
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
        metrics.get("robust_recall", metrics["recall"]),
        metrics.get("robust_f1_score", metrics["f1_score"]),
        metrics["precision"],
        metrics["accuracy"],
        -metrics["false_positive_rate"],
    )


def policy_predictions(
    policy: dict[str, float],
    model_scores: list[float],
    content_scores: list[float],
    rules: list[float],
    has_dom: list[bool],
) -> list[int]:
    return [
        int(
            (policy["ml_weight"] * model_score + policy["rule_weight"] * rule_score) >= policy["threshold"]
            and (rule_score > 0.0 or (dom and content_score >= policy["threshold"]))
        )
        for model_score, content_score, rule_score, dom in zip(
            model_scores, content_scores, rules, has_dom
        )
    ]


def select_policy(
    model_scores: list[float],
    content_scores: list[float],
    rules: list[float],
    has_dom: list[bool],
    labels: list[int],
    validation_frame: Any | None = None,
    keywords: list[str] | None = None,
    short_cutoff: int | None = None,
    pipeline: Any | None = None,
    dependencies_bundle: tuple[Any, ...] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    def predictions(policy: dict[str, float], scores: tuple[list[float], list[float], list[float], list[bool]]) -> list[int]:
        return policy_predictions(policy, *scores)

    robustness_cache: dict[str, Any] | None = None

    def build_robustness_cache() -> dict[str, Any]:
        frame_labels = validation_frame["label"].astype(int).tolist()
        short_indices = [
            index
            for index, value in enumerate(validation_frame["deployment_text"])
            if len(str(value)) <= (short_cutoff if short_cutoff is not None else 0)
        ]
        variants: dict[str, tuple[list[float], list[float], list[float], list[bool]]] = {}
        for variant in CAMOUFLAGE_VARIANTS:
            transformed = validation_frame.copy()
            transformed["deployment_text"] = transformed["deployment_text"].astype(str).map(
                lambda value: normalize_model_text(camouflage_text(value, variant), keywords)
            )
            transformed["has_dom_content"] = transformed["deployment_text"].astype(bool)
            variant_model, variant_content = client_scores(pipeline, transformed, dependencies_bundle)
            variant_rules = rule_scores(transformed.to_dict("records"), keywords)
            variants[variant] = (
                variant_model,
                variant_content,
                variant_rules,
                transformed["has_dom_content"].tolist(),
            )
        return {
            "labels": frame_labels,
            "short_indices": short_indices,
            "variants": variants,
        }

    def robustness_metrics(policy: dict[str, float]) -> dict[str, Any]:
        if validation_frame is None or keywords is None or pipeline is None or dependencies_bundle is None:
            return {}
        nonlocal robustness_cache
        if robustness_cache is None:
            robustness_cache = build_robustness_cache()
        frame_labels = robustness_cache["labels"]
        base_predictions = predictions(policy, (model_scores, content_scores, rules, has_dom))
        variants: dict[str, dict[str, Any]] = {}
        for variant, variant_scores in robustness_cache["variants"].items():
            variants[variant] = {
                "labels": frame_labels,
                "predictions": predictions(
                    policy,
                    variant_scores,
                ),
            }
        slice_metrics: dict[str, dict[str, Any]] = {}
        short_indices = robustness_cache["short_indices"]
        slice_metrics["short_dom"] = metric_summary(
            [frame_labels[index] for index in short_indices],
            [base_predictions[index] for index in short_indices],
        )
        for variant, values in variants.items():
            slice_metrics[variant] = metric_summary(values["labels"], values["predictions"])
        reported_metrics = [
            metric for metric in slice_metrics.values() if metric.get("status") != "pending"
        ]
        robust_recall = min(metric["recall"] for metric in reported_metrics) if reported_metrics else 0.0
        robust_f1 = min(metric["f1_score"] for metric in reported_metrics) if reported_metrics else 0.0
        return {
            "short_dom_cutoff": short_cutoff,
            "slices": slice_metrics,
            "robust_recall": robust_recall,
            "robust_f1_score": robust_f1,
        }

    candidates: list[tuple[dict[str, float], dict[str, Any]]] = []
    for ml_weight in (0.65, 0.70, 0.75, 0.80, 0.85):
        rule_weight = 1.0 - ml_weight
        for threshold in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
            policy = {"ml_weight": ml_weight, "rule_weight": rule_weight, "threshold": threshold}
            predicted = predictions(policy, (model_scores, content_scores, rules, has_dom))
            metrics = metric_summary(labels, predicted)
            metrics["all_targets_passed"] = all(metrics["target_checks"].values())
            if (
                validation_frame is not None
                and keywords is not None
                and pipeline is not None
                and dependencies_bundle is not None
            ):
                metrics.update(robustness_metrics(policy))
            candidates.append((policy, metrics))

    passing = [candidate for candidate in candidates if candidate[1]["all_targets_passed"]]
    buffered = [
        candidate
        for candidate in passing
        if candidate[1]["false_positive_rate"] <= VALIDATION_FPR_BUFFER_MAX
    ]
    pool = buffered or passing or candidates
    # All selection happens on the train-derived validation split. The FPR
    # buffer protects the 5% progress-report gate while recall is maximized among safe
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
                                # Keep the pattern ASCII-only and compatible with
                                # the ONNX Runtime tokenizer used by the client.
                                token_pattern=r"[a-zA-Z0-9_]+",
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


def fit_pipeline(
    pipeline: Any,
    frame: Any,
    sample_weights: list[float] | None = None,
) -> Any:
    fit_parameters = {}
    if sample_weights is not None:
        fit_parameters["classifier__sample_weight"] = sample_weights
    pipeline.fit(frame, frame["label"], **fit_parameters)
    return pipeline


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
    classifier = pipeline.named_steps.get("classifier")
    options = {id(classifier): {"zipmap": False}} if classifier is not None else None
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_types,
        target_opset=15,
        options=options,
    )
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
    _, pd, _, _, _, _, _, _, _ = bundle
    rules_path = ROOT / "models/gambling_keywords.json"
    keywords = [normalize_for_rules(value) for value in json.loads(rules_path.read_text(encoding="utf-8"))]
    train_rows = read_rows(ROOT / "data/processed/splits/train.csv")
    test_rows = read_rows(ROOT / "data/processed/splits/test.csv")
    train_records = deployment_records(train_rows, keywords)
    test_records = deployment_records(test_rows, keywords)
    train_frame = pd.DataFrame(train_records)
    test_frame = pd.DataFrame(test_records)
    train_group_ids = {row["id"]: row["split_group_id"] for row in train_rows}
    train_base_rows, validation_rows, _ = stratified_group_split(train_rows, train_group_ids, 0.2, 43)
    train_records_by_id = {record["id"]: record for record in train_records}
    train_base = pd.DataFrame([train_records_by_id[row["id"]] for row in train_base_rows])
    validation = pd.DataFrame([train_records_by_id[row["id"]] for row in validation_rows])
    short_cutoff = short_dom_cutoff(train_base)
    validation_rules = rule_scores(validation.to_dict("records"), keywords)
    tuning_candidates: list[dict[str, Any]] = []
    for config in MODEL_CONFIGS:
        base_pipeline = build_pipeline(bundle, config)
        train_augmented = augment_training_frame(train_base, keywords, bundle)
        fit_pipeline(
            base_pipeline,
            train_augmented,
            training_sample_weights(train_augmented, short_cutoff),
        )
        validation_model, validation_content = client_scores(base_pipeline, validation, bundle)
        policy, validation_metrics = select_policy(
            validation_model,
            validation_content,
            validation_rules,
            validation["has_dom_content"].tolist(),
            validation["label"].tolist(),
            validation,
            keywords,
            short_cutoff,
            base_pipeline,
            bundle,
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
    train_augmented = augment_training_frame(train_frame, keywords, bundle)
    fit_pipeline(
        final_pipeline,
        train_augmented,
        training_sample_weights(train_augmented, short_cutoff),
    )
    test_model, test_content = client_scores(final_pipeline, test_frame, bundle)
    test_rules = rule_scores(test_records, keywords)
    threshold = policy["threshold"]
    test_prediction = policy_predictions(
        policy,
        test_model,
        test_content,
        test_rules,
        test_frame["has_dom_content"].tolist(),
    )
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
        "preprocessing": {
            "unicode_nfkc": True,
            "confusable_and_leetspeak_normalization": True,
            "separator_keyword_compaction": True,
            "train_only_camouflage_augmentation": list(CAMOUFLAGE_VARIANTS),
            "positive_sample_weight": POSITIVE_SAMPLE_WEIGHT,
            "short_positive_weight": SHORT_POSITIVE_SAMPLE_WEIGHT,
            "camouflage_negative_sample_weight": CAMOUFLAGE_NEGATIVE_SAMPLE_WEIGHT,
            "short_dom_cutoff": short_cutoff,
        },
        "random_state": 42,
        "training_configuration": config,
        "validation_selection": {
        "strategy": "all progress targets, then recall-first within the 5% FPR gate",
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
            "grouped": True,
            "grouping_source": "data/processed/splits/split-manifest.json",
        },
        "policy": policy,
        "training_configuration": config,
        "validation_selection": {
            "strategy": "all progress targets, then recall-first within the 5% FPR gate",
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
