import random
import tracemalloc
from functools import cache

import pytest

from bangla_ocr.metrics import edit_distance


def reference(a, b) -> int:
    """Textbook recursion: slow, obviously correct. Only for checking."""

    @cache
    def d(i, j):
        if i == 0 or j == 0:
            return i + j
        return min(d(i - 1, j) + 1, d(i, j - 1) + 1, d(i - 1, j - 1) + (a[i - 1] != b[j - 1]))

    return d(len(a), len(b))


@pytest.mark.parametrize(
    ("ref", "hyp", "expected"),
    [
        ("", "", 0),
        ("abc", "", 3),  # all deleted
        ("", "abc", 3),  # all inserted
        ("abc", "abc", 0),
        ("kitten", "sitting", 3),
        ("ab", "ba", 2),  # a swap is 2 edits, not 1
        ("flaw", "lawn", 2),
    ],
)
def test_known_distances(ref, hyp, expected):
    assert edit_distance(ref, hyp) == expected


def test_conjunct_counts_code_points():
    # কর্ম = ক র ্ ম (4 code points); dropping the র্ conjunct part costs 2
    assert edit_distance("কর্ম", "কম") == 2
    # real Tesseract output on test image 1: ন্ -> স্, র্গ -> গগ
    assert edit_distance("অন্তর্গত", "অস্তগগত") == 3


def test_words():
    assert edit_distance(["আমি", "ভাত", "খাই"], ["আমি", "খাই"]) == 1
    assert edit_distance(["আমি", "ভাত"], ["আমি", "ভাট"]) == 1  # one wrong letter = whole word wrong


def test_matches_reference_on_random_pairs():
    rng = random.Random(0)
    alphabet = "কখ্া"
    for _ in range(500):
        a = "".join(rng.choices(alphabet, k=rng.randint(0, 8)))
        b = "".join(rng.choices(alphabet, k=rng.randint(0, 8)))
        assert edit_distance(a, b) == reference(a, b), (a, b)
        assert edit_distance(a, b) == edit_distance(b, a)


def test_two_rows_not_a_matrix():
    # full (n+1)x(m+1) table here peaks ~8 MB; two rows stay in KB
    a, b = "ক" * 500, "খ" * 500
    tracemalloc.start()
    assert edit_distance(a, b) == 500
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 500_000, f"peak {peak:,} bytes"
