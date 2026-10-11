import unicodedata

import pytest

from bangla_ocr.normalise import normalise_text

KA = "ক"  # ক
E_SIGN, AA_SIGN, O_SIGN = "ে", "া", "ো"  # ে া ো
YA, NUKTA, YYA = "য", "়", "য়"  # য ় য়


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("", ""),
        ("আমি ভাত খাই।", "আমি ভাত খাই।"),  # already clean: unchanged
        # NFC: two-part vowel sign composes to one code point
        (KA + E_SIGN + AA_SIGN, KA + O_SIGN),
        (KA + O_SIGN, KA + O_SIGN),
        # Composition exclusion: য় DEcomposes under NFC (base + nukta)
        (YYA, YA + NUKTA),
        (YA + NUKTA, YA + NUKTA),
        # invisibles removed
        ("র‍্য", "র্য"),  # ZWJ (র‍্য)
        ("ক‌খ", "কখ"),  # ZWNJ
        ("﻿আমি", "আমি"),  # BOM
        ("আমি​", "আমি"),  # ZWSP
        # whitespace
        ("আমি ভাত", "আমি ভাত"),  # NBSP
        ("  আমি \t\n ভাত  ", "আমি ভাত"),
        ("   ", ""),
        # digits / punctuation untouched
        ("১২৩ abc 123", "১২৩ abc 123"),
    ],
)
def test_normalise(raw, expected):
    assert normalise_text(raw) == expected


def test_nfc_runs_after_removing_invisibles():
    # ে ZWJ া: only once the ZWJ is gone can the pair compose into ো
    assert normalise_text(KA + E_SIGN + "‍" + AA_SIGN) == KA + O_SIGN


def test_space_left_by_removed_invisible_is_collapsed():
    assert normalise_text("আমি ​ ভাত") == "আমি ভাত"


@pytest.mark.parametrize("raw", ["", YYA + " " + KA + E_SIGN + AA_SIGN, " ক‌খ  ১২৩‍", "  "])
def test_idempotent_and_nfc(raw):
    once = normalise_text(raw)
    assert normalise_text(once) == once
    assert unicodedata.is_normalized("NFC", once)
