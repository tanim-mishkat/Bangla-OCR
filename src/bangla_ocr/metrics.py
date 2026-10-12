"""Scoring: edit distance -> CER / WER."""

from collections.abc import Sequence


def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    """Levenshtein distance: fewest insertions + deletions + substitutions turning hyp into ref.

    Any sequence works: str -> character edits (CER), list of words -> word edits (WER).
    """
    raise NotImplementedError  # your turn
