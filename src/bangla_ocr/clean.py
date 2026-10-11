"""Normalise labels, drop empty and duplicate texts, and count every drop."""

import json
from pathlib import Path

import polars as pl

from bangla_ocr import ingest
from bangla_ocr.normalise import normalise_text

OUT_PATH = Path("data/interim/clean.parquet")
REPORT_PATH = Path("data/interim/clean_report.json")


def plan(meta: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    """Decide which rows survive, using only small columns (no image bytes).

    `meta` needs id, source, text, text_id, image_hash. Returns the kept rows (id,
    normalised `text`, original `text_raw`; sorted by id) and {reason: rows dropped}.

    Synth dedupes by text (same text = near-identical render; a line drops with all
    its fonts). REID dedupes by identical image only: same text on different crops
    (e.g. page numbers) are distinct test samples, and REID is test-only so cannot leak.
    """
    df = meta.with_columns(
        text_raw=pl.col("text"),
        text=pl.col("text").fill_null("").map_elements(normalise_text, return_dtype=pl.String),
    ).sort("id")
    is_synth, is_reid = pl.col("source") == "synth", pl.col("source") == "reid2019"

    def first_of(col: str, group: str) -> pl.Expr:
        return pl.col(col) == pl.col(col).min().over("source", group)

    def reid_texts(d: pl.DataFrame) -> pl.Series:
        return d.filter(is_reid)["text"].implode()

    rules = {
        "empty text": lambda d: pl.col("text") != "",
        "duplicate synth text": lambda d: ~is_synth | first_of("text_id", "text"),
        "duplicate reid image": lambda d: ~is_reid | first_of("id", "image_hash"),
        # REID is test-only: the same text in synth train would leak it
        "synth text also in reid": lambda d: ~(is_synth & pl.col("text").is_in(reid_texts(d))),
    }

    report = {"input": df.height}
    for reason, keep in rules.items():
        before = df.height
        df = df.filter(keep(df))
        report[reason] = before - df.height
    report["output"] = df.height
    return df.select("id", "text", "text_raw"), report


def clean(in_path: Path = ingest.OUT_PATH, out_path: Path = OUT_PATH) -> dict[str, int]:
    """Write cleaned rows to `out_path`, streaming the image bytes through a join."""
    lf = pl.scan_parquet(in_path)
    meta = lf.select("id", "source", "text", "text_id", image_hash=pl.col("image").hash())
    kept, report = plan(meta.collect())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (
        lf.drop("text")
        .join(kept.lazy(), on="id", how="inner", maintain_order="left")
        .select([*ingest.SCHEMA, "text_raw"])
        .sink_parquet(out_path)
    )
    return report


if __name__ == "__main__":
    report = clean()
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(pl.scan_parquet(OUT_PATH).group_by("source").len().sort("source").collect())
