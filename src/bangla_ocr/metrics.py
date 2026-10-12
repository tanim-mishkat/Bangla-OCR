"""Scoring: edit distance -> CER / WER."""

from collections.abc import Sequence


def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    """Levenshtein distance: fewest insertions + deletions + substitutions turning hyp into ref.

    Any sequence works: str -> character edits (CER), list of words -> word edits (WER).
    """
    prev = list(range(len(hyp) + 1))

    for i, r in enumerate(ref, 1):
        cur = [i]

        for j, h in enumerate(hyp, 1):
            delete = prev[j] + 1
            insert = cur[j - 1] + 1
            substitute = prev[j - 1] + (r != h)

            cur.append(min(delete, insert, substitute))

        prev = cur

    return prev[-1]
