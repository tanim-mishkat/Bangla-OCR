import polars as pl
import pytest

from bangla_ocr.corpus import build_corpus, keep_line, to_lines


def test_to_lines_splits_sentences_on_danda_and_question_mark():
    assert to_lines("আমি ভাত খাই। তুমি কি খাও? ভালো!") == ["আমি ভাত খাই।", "তুমি কি খাও?", "ভালো!"]


def test_to_lines_wraps_by_words():
    lines = to_lines("আমার সোনার বাংলা আমি তোমায় ভালোবাসি", max_chars=12)
    assert lines == ["আমার সোনার", "বাংলা আমি", "তোমায়", "ভালোবাসি"]
    assert all(len(line) <= 12 for line in lines)


def test_to_lines_handles_paragraphs_and_blank_text():
    assert to_lines("এক।\n\nদুই।") == ["এক।", "দুই।"]
    assert to_lines("") == []


@pytest.mark.parametrize(
    "line, expected",
    [
        ("আমি ভাত খাই।", True),
        ("আমি", False),  # too short
        ("Hello world, this is English", False),
        ("বাংলা ABCDEFGH", False),  # 5/13 Bangla < 0.8
        ("১৯৭১ সালে স্বাধীন", True),  # Bangla digits count as Bangla
    ],
)
def test_keep_line(line, expected):
    assert keep_line(line) == expected


@pytest.fixture
def wiki_parquet(tmp_path):
    path = tmp_path / "wiki.parquet"
    pl.DataFrame(
        {
            "url": ["u1", "u2", "u3"],
            "text": ["আমি ভাত খাই। তুমি কি খাও?", "আমি ভাত খাই। English only here.", "ভালো আছি।"],
        }
    ).write_parquet(path)
    return path


def test_build_corpus_dedupes_and_keeps_first_url(wiki_parquet):
    df = build_corpus(wiki_parquet, n_lines=100)
    assert sorted(df["text"]) == sorted(["আমি ভাত খাই।", "তুমি কি খাও?", "ভালো আছি।"])
    assert df.filter(pl.col("text") == "আমি ভাত খাই।")["url"].item() == "u1"
    assert df["text_id"].is_unique().all()
    assert set(df["licence"]) == {"CC BY-SA 3.0"}


def test_build_corpus_is_deterministic(wiki_parquet):
    assert build_corpus(wiki_parquet, n_lines=2).equals(build_corpus(wiki_parquet, n_lines=2))
