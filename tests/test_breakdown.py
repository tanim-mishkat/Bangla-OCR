import polars as pl
import pytest

from bangla_ocr import checks
from bangla_ocr.breakdown import bangla_graphemes

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
