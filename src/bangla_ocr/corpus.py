"""Build the Bangla line corpus used for synthetic rendering."""

import re
import textwrap
from pathlib import Path

import polars as pl

from bangla_ocr.sources import WIKI_BN
from bangla_ocr.text import bangla_ratio

WIKI_PATH = Path("data/raw") / WIKI_BN.filename
OUT_PATH = Path("data/interim/corpus.parquet")
SENTENCE_END = re.compile(r"(?<=[।?!])\s+")
MAX_CHARS = 40  # code points per rendered line
MIN_CHARS = 5
MIN_BANGLA = 0.8  # tolerate danda, digits, punctuation


def to_lines(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split text into sentences, then wrap each into lines of <= max_chars (by words)."""
    lines = []
    for para in text.splitlines():
        for sentence in SENTENCE_END.split(para.strip()):
            lines.extend(textwrap.wrap(sentence, max_chars, break_long_words=False))
    return lines


def keep_line(line: str) -> bool:
    return len(line) >= MIN_CHARS and bangla_ratio(line) >= MIN_BANGLA


def build_corpus(
    wiki_path: Path = WIKI_PATH, n_articles: int = 5000, n_lines: int = 3000, seed: int = 42
) -> pl.DataFrame:
    """Sample `n_lines` unique Bangla lines from the first `n_articles` articles."""
    articles = pl.scan_parquet(wiki_path).select("url", "text").head(n_articles).collect()

    first_url: dict[str, str] = {}  # dedupe, keep first source for attribution
    for url, text in articles.iter_rows():
        for line in to_lines(text):
            if keep_line(line):
                first_url.setdefault(line, url)

    df = pl.DataFrame({"text": list(first_url), "url": list(first_url.values())})
    return (
        df.sample(n=min(n_lines, df.height), seed=seed)
        .with_row_index("i")
        .select(
            pl.format("wiki_bn/{}", pl.col("i").cast(pl.String).str.zfill(5)).alias("text_id"),
            "text",
            "url",
            pl.lit(WIKI_BN.licence).alias("licence"),
        )
    )


if __name__ == "__main__":
    corpus = build_corpus()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    corpus.write_parquet(OUT_PATH)
    print(f"Wrote {corpus.height} lines to {OUT_PATH}")
