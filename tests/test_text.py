import pytest

from bangla_ocr.text import bangla_ratio, is_bangla_char


@pytest.mark.parametrize(
    "ch, expected",
    [
        ("\u097f", False),  # just before Bengali block
        ("\u0980", True),  # first character in Bengali block
        ("\u09ff", True),  # last character in Bengali block
        ("\u0a00", False),  # just after Bengali block
        ("ক", True),  # consonant KA
        ("অ", True),  # vowel A
        ("৫", True),  # Bangla digit 5
        ("া", True),  # vowel sign AA
        ("্", True),  # hasant / virama
        ("a", False),
        ("5", False),  # ASCII digit
        (" ", False),
        ("।", False),  # danda is U+0964, outside Bengali block
    ],
)
def test_is_bangla_char(ch, expected):
    assert is_bangla_char(ch) == expected


@pytest.mark.parametrize("bad", ["", "কখ"])
def test_is_bangla_char_rejects_non_single_char(bad):
    with pytest.raises(ValueError):
        is_bangla_char(bad)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", 0.0),
        ("   ", 0.0),
        ("আমি", 1.0),
        ("hello", 0.0),
        ("আমি ok", 0.6),  # 3 Bangla code points out of 5 non-space chars
    ],
)
def test_bangla_ratio(text, expected):
    assert bangla_ratio(text) == pytest.approx(expected)
