#!/usr/bin/env python3
"""Deterministic leakage-safe grouping for model train/test splits."""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections import defaultdict
from urllib.parse import urlsplit
from typing import Any, Iterable


_TLD_EXTRACTOR: Any | None = None
_WHITESPACE = re.compile(r"\s+")


def normalize_signature(value: str) -> str:
    """Normalize an exact text signature without changing its meaning."""
    normalized = unicodedata.normalize("NFKC", value or "").casefold()
    return _WHITESPACE.sub(" ", normalized).strip()


def domain_group(url: str, row_id: str = "") -> str:
    """Return a stable registrable-domain group without contacting the network."""
    global _TLD_EXTRACTOR
    try:
        import tldextract

        if _TLD_EXTRACTOR is None:
            _TLD_EXTRACTOR = tldextract.TLDExtract(
                suffix_list_urls=(),
                cache_dir="/tmp/gamblock-tldextract-cache",
            )
        extracted = _TLD_EXTRACTOR(url)
        registered = getattr(extracted, "top_domain_under_public_suffix", "")
        if registered:
            return f"domain:{registered.lower()}"
    except ImportError:
        pass

    try:
        hostname = (urlsplit(url).hostname or "").lower().rstrip(".")
    except ValueError:
        hostname = ""
    if hostname:
        labels = hostname.split(".")
        if len(labels) >= 2 and not all(label.isdigit() for label in labels[-2:]):
            return f"domain:{'.'.join(labels[-2:])}"
        return f"host:{hostname}"
    return f"row:{row_id or 'unknown'}"


class _DisjointSet:
    def __init__(self, identifiers: Iterable[str]) -> None:
        self.parent = {identifier: identifier for identifier in identifiers}

    def find(self, identifier: str) -> str:
        parent = self.parent[identifier]
        if parent != identifier:
            self.parent[identifier] = self.find(parent)
        return self.parent[identifier]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def build_group_ids(
    rows: list[dict[str, str]],
    model_texts: dict[str, str],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Connect duplicate model text and domain rows into homogeneous groups."""
    identifiers = [row["id"] for row in rows]
    disjoint = _DisjointSet(identifiers)
    seen: dict[tuple[str, str], str] = {}
    signatures_by_row: dict[str, dict[str, str]] = {}
    for row in rows:
        row_id = row["id"]
        signatures = {
            "model_text": normalize_signature(model_texts.get(row_id, "")),
            "text_clean": normalize_signature(row.get("text_clean", "")),
            "text_combined": normalize_signature(row.get("text_combined", "")),
            "domain": domain_group(row.get("url", ""), row_id),
        }
        signatures_by_row[row_id] = signatures
        for kind, value in signatures.items():
            if not value or (kind == "model_text" and not value):
                continue
            key = (kind, value)
            previous = seen.get(key)
            if previous is not None:
                disjoint.union(row_id, previous)
            else:
                seen[key] = row_id

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for row_id in identifiers:
        members_by_root[disjoint.find(row_id)].append(row_id)

    group_ids: dict[str, str] = {}
    conflicts: list[dict[str, Any]] = []
    rows_by_id = {row["id"]: row for row in rows}
    for members in sorted(members_by_root.values(), key=lambda values: min(values)):
        ordered = sorted(members)
        group_id = "group:" + hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()[:16]
        labels = sorted({int(rows_by_id[row_id]["label"]) for row_id in ordered})
        for row_id in ordered:
            group_ids[row_id] = group_id
        if len(labels) > 1:
            conflicts.append({"group_id": group_id, "rows": len(ordered), "labels": labels})

    return group_ids, conflicts


def remove_conflicting_groups(
    rows: list[dict[str, str]],
    group_ids: dict[str, str],
    conflicts: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    conflict_ids = {item["group_id"] for item in conflicts}
    eligible = [row for row in rows if group_ids[row["id"]] not in conflict_ids]
    return eligible, conflicts


def stratified_group_split(
    rows: list[dict[str, str]],
    group_ids: dict[str, str],
    test_fraction: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    """Split whole homogeneous groups while preserving both classes."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[group_ids[row["id"]]].append(row)

    by_label: dict[int, list[str]] = defaultdict(list)
    for group, members in grouped.items():
        labels = {int(member["label"]) for member in members}
        if len(labels) != 1:
            raise ValueError("stratified_group_split received a conflicting group")
        by_label[next(iter(labels))].append(group)

    rng = random.Random(seed)
    test_groups: set[str] = set()
    for label in sorted(by_label):
        candidates = list(by_label[label])
        rng.shuffle(candidates)
        if len(candidates) > 1:
            count = max(1, round(len(candidates) * test_fraction))
            count = min(count, len(candidates) - 1)
            test_groups.update(candidates[:count])

    assignments: dict[str, str] = {}
    train: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    for group, members in sorted(grouped.items()):
        split = "test" if group in test_groups else "train"
        for row in members:
            assignments[row["id"]] = split
            (test if split == "test" else train).append(row)
    return train, test, assignments


def assignment_hash(assignments: dict[str, str]) -> str:
    payload = "\n".join(
        f"{row_id}\t{assignments[row_id]}"
        for row_id in sorted(assignments)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def overlap_count(
    rows: list[dict[str, str]],
    assignments: dict[str, str],
    signature_function,
) -> int:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        signature = signature_function(row)
        if signature:
            grouped[signature].add(assignments.get(row["id"], "excluded"))
    return sum(len(splits) > 1 for splits in grouped.values())
