#!/usr/bin/env python3
"""Create deterministic text- and domain-grouped train/test CSV files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from grouped_split import (
        assignment_hash,
        build_group_ids,
        domain_group,
        normalize_signature,
        overlap_count,
        remove_conflicting_groups,
        stratified_group_split,
    )
except ModuleNotFoundError:  # Imported by a test through an arbitrary module path.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from grouped_split import (
        assignment_hash,
        build_group_ids,
        domain_group,
        normalize_signature,
        overlap_count,
        remove_conflicting_groups,
        stratified_group_split,
    )


ROOT = Path(__file__).resolve().parents[1]


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


def deployment_texts(rows: list[dict[str, str]], trainer: Any) -> dict[str, str]:
    """Extract the exact normalized text surface used by the candidate model."""
    rules_path = ROOT / "models/gambling_keywords.json"
    keywords = [trainer.normalize_for_rules(value) for value in json.loads(rules_path.read_text(encoding="utf-8"))]
    records = trainer.deployment_records(rows, keywords)
    return {record["id"]: record["deployment_text"] for record in records}


def class_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(row["label"] for row in rows).items()))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def duplicate_summary(
    rows: list[dict[str, str]],
    model_texts: dict[str, str],
    assignments: dict[str, str],
) -> dict[str, Any]:
    fields = {
        "model_text": lambda row: normalize_signature(model_texts.get(row["id"], "")),
        "text_clean": lambda row: normalize_signature(row.get("text_clean", "")),
        "text_combined": lambda row: normalize_signature(row.get("text_combined", "")),
        "normalized_url": lambda row: normalize_signature(row.get("url_clean", "")),
        "domain": lambda row: domain_group(row.get("url", ""), row["id"]),
    }
    result: dict[str, Any] = {}
    for name, signature_function in fields.items():
        groups: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            signature = signature_function(row)
            if signature:
                groups[signature].append(row["id"])
        duplicate_groups = [members for members in groups.values() if len(members) > 1]
        result[name] = {
            "groups_with_duplicates": len(duplicate_groups),
            "duplicate_rows": sum(len(members) for members in duplicate_groups),
            "cross_split_groups": overlap_count(rows, assignments, signature_function),
        }
    result["audit_passed"] = not any(
        details["cross_split_groups"]
        for details in result.values()
        if isinstance(details, dict)
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/dataset_clean.csv")
    parser.add_argument("--train-output", type=Path, default=ROOT / "data/processed/splits/train.csv")
    parser.add_argument("--test-output", type=Path, default=ROOT / "data/processed/splits/test.csv")
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=ROOT / "data/processed/splits/split-manifest.json",
    )
    args = parser.parse_args()

    trainer = load_trainer()
    rows = read_rows(args.input)
    model_texts = deployment_texts(rows, trainer)
    group_ids, conflicts = build_group_ids(rows, model_texts)
    eligible, conflicts = remove_conflicting_groups(rows, group_ids, conflicts)
    if not eligible:
        raise RuntimeError("no eligible rows remain after conflicting-group exclusion")

    train, test, assignments = stratified_group_split(eligible, group_ids, 0.2, 42)
    if not train or not test:
        raise RuntimeError("grouped split did not produce both train and test rows")

    for row in train + test:
        row["split_group_id"] = group_ids[row["id"]]
    fieldnames = [*rows[0].keys(), "split_group_id"]
    write_rows(args.train_output, train, fieldnames)
    write_rows(args.test_output, test, fieldnames)

    grouped_assignments = {row_id: assignments[row_id] for row_id in sorted(assignments)}
    audit = duplicate_summary(eligible, model_texts, grouped_assignments)
    manifest = {
        "schema_version": 2,
        "random_state": 42,
        "test_fraction": 0.2,
        "primary_split_method": "connected model-text and registrable-domain grouped stratification",
        "grouping_keys": ["model_text", "text_clean", "text_combined", "registrable_domain"],
        "source": {
            "dataset_clean_sha256": sha256(args.input),
            "rows": len(rows),
        },
        "conflicting_groups_excluded": {
            "groups": len(conflicts),
            "rows": sum(item["rows"] for item in conflicts),
        },
        "eligible": {
            "rows": len(eligible),
            "label_counts": class_counts(eligible),
            "groups": len(set(group_ids[row["id"]] for row in eligible)),
        },
        "train": {
            "rows": len(train),
            "label_counts": class_counts(train),
            "groups": len({group_ids[row["id"]] for row in train}),
            "sha256": sha256(args.train_output),
        },
        "test": {
            "rows": len(test),
            "label_counts": class_counts(test),
            "groups": len({group_ids[row["id"]] for row in test}),
            "sha256": sha256(args.test_output),
        },
        "assignment_sha256": assignment_hash(grouped_assignments),
        "final_test_frozen_before_threshold_selection": True,
        "leakage_audit": audit,
        "known_gaps": [
            "Collection dates and upstream labeling governance remain unavailable.",
            "Camouflage variants are generated in memory during model evaluation and are not persisted as a dataset.",
        ],
    }
    baseline_train_path = args.train_output.parent / "baseline_row_stratified_train.csv"
    baseline_test_path = args.test_output.parent / "baseline_row_stratified_test.csv"
    if baseline_train_path.is_file() and baseline_test_path.is_file():
        baseline_train = read_rows(baseline_train_path)
        baseline_test = read_rows(baseline_test_path)
        manifest["historical_baseline"] = {
            "method": "stratified random row split",
            "train": {
                "rows": len(baseline_train),
                "label_counts": class_counts(baseline_train),
                "sha256": sha256(baseline_train_path),
            },
            "test": {
                "rows": len(baseline_test),
                "label_counts": class_counts(baseline_test),
                "sha256": sha256(baseline_test_path),
            },
        }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "train_rows": len(train),
                "test_rows": len(test),
                "excluded_conflict_rows": manifest["conflicting_groups_excluded"]["rows"],
                "audit_passed": audit["audit_passed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
