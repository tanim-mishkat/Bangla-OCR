import random
import tracemalloc

import polars as pl
import pytest

from bangla_ocr import checks, score
from bangla_ocr.breakdown import (
    CLASSES,
    align,
    bangla_graphemes,
    breakdown_rows,
    by_class,
    char_classes,
    confusions,
    headline,
    row_breakdown,
)
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


# --- classes: which kind of character each error hit ---


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("কর্মী", ["consonant", "conjunct", "conjunct", "conjunct", "kar"]),  # র্ম is one যুক্তাক্ষর
        ("চাঁদ", ["consonant", "kar", "sign", "consonant"]),
        ("উৎস", ["vowel", "consonant", "consonant"]),  # খণ্ড ত is a plain consonant
        ("ক্", ["consonant", "sign"]),  # a lone হসন্ত is not a যুক্তাক্ষর
        ("১, a।", ["digit", "punct", "space", "other", "punct"]),
        ("1", ["other"]),  # a Latin digit is not a Bangla digit
    ],
)
def test_char_classes(text, expected):
    assert char_classes(text) == expected


def test_every_class_is_reportable():
    assert set(char_classes("কর্মী চাঁদ উৎস ১, a।")) == set(CLASSES)


def class_errors(b: dict) -> int:
    return sum(b[f"err_{c}"] for c in CLASSES)


def test_conjunct_error_is_attributed_to_conjunct():
    b = row_breakdown("কর্ম", "কম")  # র্ dropped from the যুক্তাক্ষর
    assert (b["err_conjunct"], b["n_conjunct"], class_errors(b)) == (2, 3, 2)
    assert (b["grapheme_edits"], b["graphemes"]) == (1, 2)
    assert b["confusions"] == ["র্ম → ম"]


def test_bangla_digit_read_as_latin_is_leakage():
    b = row_breakdown("১২", "12")
    assert (b["err_digit"], b["latin_digits"], b["latin_letters"]) == (2, 2, 0)
    # a Latin digit the label really has, read correctly, is not leakage
    assert row_breakdown("COVID-19", "COVID-19")["latin_digits"] == 0


def test_empty_prediction_loses_everything():
    b = row_breakdown("কর্মী ১", "")
    assert all(b[f"err_{c}"] == b[f"n_{c}"] for c in CLASSES)
    assert b["grapheme_edits"] == b["graphemes"] == 4
    assert b["inserted"] == 0


def test_insertions_are_counted_once_outside_classes():
    b = row_breakdown("ক", "কখগ")
    assert (class_errors(b), b["inserted"], b["confusions"]) == (0, 2, ["∅ → খ", "∅ → গ"])
    assert row_breakdown("ক খ", "কখ")["confusions"] == ["␣ → ∅"]  # spaces shown visibly


def test_errors_sum_to_edit_distance_on_random_pairs():
    rng = random.Random(1)
    alphabet = "কখরমা্ ১a,"
    for _ in range(300):
        ref = "".join(rng.choices(alphabet, k=rng.randint(0, 15)))
        hyp = "".join(rng.choices(alphabet, k=rng.randint(0, 15)))
        b = row_breakdown(ref, hyp)
        assert class_errors(b) + b["inserted"] == edit_distance(ref, hyp)
        assert sum(b[f"n_{c}"] for c in CLASSES) == len(ref)
        assert b["grapheme_edits"] == edit_distance(bangla_graphemes(ref), bangla_graphemes(hyp))


# --- reports ---


def report_rows() -> pl.DataFrame:
    """10 texts x 2 fonts; the engine drops র্ from কর্ম in every other text."""
    texts = ["কর্ম ১"] * 20
    hyps = ["কম ১" if i % 2 else t for i, t in enumerate(texts)]
    labels = pl.DataFrame(
        {
            "id": [f"r{i}" for i in range(20)],
            "text_id": [f"t{i // 2}" for i in range(20)],
            "page": [None] * 20,
            "source": ["synth"] * 20,
            "level": ["clean"] * 20,
            "font": ["F"] * 20,
            "region_type": [None] * 20,
            "text": texts,
        },
        schema_overrides={"region_type": pl.String, "page": pl.String},
    )
    preds = pl.DataFrame({"id": labels["id"], "pred": hyps, "seconds": 0.1, "error": None})
    return breakdown_rows(score.score_rows(labels, preds.cast({"error": pl.String})))


def test_by_class_shares_and_totals_reconcile_with_cer():
    rows = report_rows()
    t = by_class(rows, "text_id")
    assert t["errors"].sum() == rows["char_edits"].sum()  # every edit attributed exactly once
    assert t["share"].sum() == pytest.approx(1.0)
    conj = t.filter(pl.col("class").str.starts_with("যুক্তাক্ষর")).row(0, named=True)
    assert conj["rate"] == pytest.approx(20 / 60)  # 10 rows x 2 chars lost of 20 x 3
    assert conj["low"] <= conj["rate"] <= conj["high"]
    assert t.filter(pl.col("total") == 0)["rate"].is_null().all()  # absent class: no rate


def test_headline_cer_matches_score_and_ger_counts_graphemes():
    rows = report_rows()
    h = {r["metric"]: r for r in headline(rows, "text_id").iter_rows(named=True)}
    assert h["CER"]["rate"] == pytest.approx(rows["char_edits"].sum() / rows["chars"].sum())
    assert h["GER"]["rate"] == pytest.approx(10 / (20 * 4))  # ক র্ম space ১ = 4 graphemes


def test_confusions_ranked():
    c = confusions(report_rows())
    assert c.row(0) == ("র্ম → ম", 10)
