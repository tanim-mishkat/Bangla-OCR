import unicodedata

import pytest

from bangla_ocr.normalise import normalise_text

KA = "\u0995"  # ক
E_SIGN, AA_SIGN, O_SIGN = "\u09c7", "\u09be", "\u09cb"  # ে া ো
YA, NUKTA, YYA = "\u09af", "\u09bc", "\u09df"  # য ় য়


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
        ("র\u200d্য", "র্য"),  # ZWJ (ra + ya-phala)
        ("ক\u200cখ", "কখ"),  # ZWNJ
        ("\ufeffআমি", "আমি"),  # BOM
        ("আমি\u200b", "আমি"),  # ZWSP
        # whitespace
        ("আমি\u00a0ভাত", "আমি ভাত"),  # NBSP
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
    assert normalise_text(KA + E_SIGN + "\u200d" + AA_SIGN) == KA + O_SIGN


def test_space_left_by_removed_invisible_is_collapsed():
    assert normalise_text("আমি \u200b ভাত") == "আমি ভাত"


@pytest.mark.parametrize(
    "raw", ["", YYA + " " + KA + E_SIGN + AA_SIGN, "\u00a0ক\u200cখ  ১২৩\u200d", "  "]
)
def test_idempotent_and_nfc(raw):
    once = normalise_text(raw)
    assert normalise_text(once) == once
    assert unicodedata.is_normalized("NFC", once)
