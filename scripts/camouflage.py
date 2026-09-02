"""Deterministic in-memory camouflage transforms used by training and evaluation."""

from __future__ import annotations


CAMOUFLAGE_VARIANTS = (
    "case_variation",
    "separator_insertion",
    "character_substitution",
    "unicode_confusable",
)


def camouflage_text(text: str, variant: str) -> str:
    """Return one deterministic camouflage variant without persisting samples."""
    if variant == "case_variation":
        return text.swapcase()
    if variant == "separator_insertion":
        return "-".join(text.split())
    if variant == "character_substitution":
        substitutions = str.maketrans(
            {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
        )
        return text.translate(substitutions)
    if variant == "unicode_confusable":
        substitutions = str.maketrans(
            {"a": "а", "e": "е", "i": "і", "o": "о", "s": "ѕ", "t": "т"}
        )
        return text.translate(substitutions)
    raise ValueError(f"unknown camouflage variant: {variant}")
