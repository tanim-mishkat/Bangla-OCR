import random
import tracemalloc

import polars as pl
import pytest

from bangla_ocr import checks
from bangla_ocr.breakdown import align, bangla_graphemes
from bangla_ocr.metrics import edit_distance

NUKTA = chr(0x09BC)  # NFC stores U+09DC as U+09A1 + NUKTA (composition exclusion)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", []),
        ("ক", ["ক"]),
        ("কর্ম", ["ক", "র্ম"]),  # রেফ: র্ joins the next consonant
        ("অন্তর্গত", ["অ", "ন্ত", "র্গ", "ত"]),
        ("ক্ষমা", ["ক্ষ", "মা"]),
        ("স্বাস্থ্য", ["স্বা", "স্থ্য"]),  # ব-ফলা, য-ফলা
        ("লক্ষ্মী", ["ল", "ক্ষ্মী"]),  # three consonants in one যুক্তাক্ষর
        ("চাঁদ", ["চাঁ", "দ"]),  # কার + চন্দ্রবিন্দু
        ("দুঃখ", ["দুঃ", "খ"]),  # বিসর্গ
        ("কৌ", ["কৌ"]),
        ("উৎস", ["উ", "ৎ", "স"]),  # খণ্ড ত is its own grapheme
        (f"বড{NUKTA}", ["ব", f"ড{NUKTA}"]),  # নুকতা attaches
        ("ক্", ["ক্"]),  # trailing হসন্ত stays
        ("বাক্ যাক", ["বা", "ক্", " ", "যা", "ক"]),  # হসন্ত does not join across a space
        ("১২৩", ["১", "২", "৩"]),
        ("আমি, তুমি।", ["আ", "মি", ",", " ", "তু", "মি", "।"]),
        ("ab", ["a", "b"]),
        ("া", ["া"]),  # orphan কার at the start: its own grapheme, never dropped
    ],
)
def test_known_splits(text, expected):
    assert bangla_graphemes(text) == expected


def test_vowel_after_hasanta_does_not_join():
    # only a consonant completes a যুক্তাক্ষর; an independent vowel starts a new grapheme
    assert bangla_graphemes("ক্অ") == ["ক্", "অ"]


@pytest.mark.skipif(not checks.split_paths()["test"].exists(), reason="processed data missing")
def test_lossless_and_nonempty_on_every_test_label():
    for text in pl.read_parquet(checks.split_paths()["test"], columns=["text"])["text"]:
        graphemes = bangla_graphemes(text)
        assert "".join(graphemes) == text
        assert all(graphemes)
        assert len(graphemes) <= len(text)


# --- align: which characters went wrong, not just how many ---


def cost(ops) -> int:
    return sum(op != "match" for op, _, _ in ops)


def test_align_known_ops():
    assert align("কখগ", "কঘগ") == [("match", "ক", "ক"), ("sub", "খ", "ঘ"), ("match", "গ", "গ")]
    assert align("কখ", "ক") == [("match", "ক", "ক"), ("del", "খ", None)]
    assert align("ক", "কখ") == [("match", "ক", "ক"), ("ins", None, "খ")]


@pytest.mark.parametrize(
    ("ref", "hyp", "expected"),
    [
        ("", "", []),
        ("কখ", "", [("del", "ক", None), ("del", "খ", None)]),  # empty prediction: all deletions
        ("", "ab", [("ins", None, "a"), ("ins", None, "b")]),
    ],
)
def test_align_empty_sides(ref, hyp, expected):
    assert align(ref, hyp) == expected


def test_align_prefers_substitution_on_ties():
    # "ab" -> "ba": two subs or del+ins+... all cost 2; subs make confusion pairs
    assert align("ab", "ba") == [("sub", "a", "b"), ("sub", "b", "a")]


def test_align_on_graphemes():
    # a wrong যুক্তাক্ষর is one substitution of whole units
    ops = align(bangla_graphemes("অন্তর"), bangla_graphemes("অস্তর"))
    assert ops == [("match", "অ", "অ"), ("sub", "ন্ত", "স্ত"), ("match", "র", "র")]


def test_align_is_minimal_and_lossless_on_random_pairs():
    rng = random.Random(0)
    for _ in range(500):
        ref = "".join(rng.choices("কখগা্", k=rng.randint(0, 12)))
        hyp = "".join(rng.choices("কখগা্", k=rng.randint(0, 12)))
        ops = align(ref, hyp)
        assert cost(ops) == edit_distance(ref, hyp)
        assert "".join(r for _, r, _ in ops if r is not None) == ref
        assert "".join(h for _, _, h in ops if h is not None) == hyp
        assert all((op == "match") == (r == h) for op, r, h in ops)


def test_align_table_is_compact():
    # list-of-int-objects table at 500x500 peaks ~7 MB; array("I") ~1 MB
    tracemalloc.start()
    align("ক" * 500, "খ" * 500)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 3_000_000
