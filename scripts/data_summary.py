"""Print dataset summary tables for the write-up (no image bytes loaded).

Usage: uv run python scripts/data_summary.py
"""

import sys

import polars as pl

from bangla_ocr.split import OUT_DIR, SPLITS

sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
pl.Config.set_tbl_hide_dataframe_shape(True)

df = pl.concat(
    pl.scan_parquet(OUT_DIR / f"{s}.parquet")
    .select("source", "text", "text_id", "font", "level", "region_type")
    .with_columns(split=pl.lit(s))
    .collect()
    for s in SPLITS
).with_columns(
    chars=pl.col("text").str.len_chars(),
    has_digit=pl.col("text").str.contains(r"[\u09e6-\u09ef]"),  # Bangla digits ০-৯
)

print("Rows by split and source")
print(
    df.group_by("split", "source")
    .agg(
        rows=pl.len(),
        texts=pl.col("text").n_unique(),
        median_chars=pl.col("chars").median(),
        max_chars=pl.col("chars").max(),
        with_digits=pl.col("has_digit").mean().round(3),
    )
    .sort("split", "source")
)

test = df.filter(pl.col("split") == "test")
print("\nTest synth by level")
print(test.filter(pl.col("source") == "synth").group_by("level").len().sort("level"))
print("\nTest synth by font")
print(test.filter(pl.col("source") == "synth").group_by("font").len().sort("font"))
print("\nTest REID by region_type")
print(
    test.filter(pl.col("source") == "reid2019")
    .group_by("region_type")
    .len()
    .sort("len", descending=True)
)
