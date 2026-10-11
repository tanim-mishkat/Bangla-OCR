"""Leak-free train/val/test split: grouped by text, deterministic, REID test-only."""

import hashlib
from pathlib import Path

import polars as pl

from bangla_ocr import clean

OUT_DIR = Path("data/processed")
SPLITS = ("train", "val", "test")
TRAIN_PCT, VAL_PCT = 80, 10  # rest is test


def assign_split(key: str) -> str:
    """'train', 'val' or 'test' for a group key, from its sha256.

    bucket = first 8 bytes of sha256(utf-8 key) as a big-endian int, % 100
    bucket < TRAIN_PCT -> train; < TRAIN_PCT + VAL_PCT -> val; else test.
    Raises ValueError on an empty key (a missing group id must not land silently in train).
    """
    if not key:
        raise ValueError("Key cannot be empty")

    digest = hashlib.sha256(key.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100

    if bucket < TRAIN_PCT:
        return "train"

    if bucket < TRAIN_PCT + VAL_PCT:
        return "val"

    return "test"


def split(
    in_path: Path = clean.OUT_PATH,
    out_dir: Path = OUT_DIR,
) -> dict[str, dict[str, int]]:
    """Write <out_dir>/<split>.parquet. Returns {split: {source: rows}}.

    Synth groups by text_id (all fonts of a line share a split);
    REID is always test.
    """
    lf = pl.scan_parquet(in_path)

    keys = (
        lf.select("id", "source", "text_id")
        .collect()
        .with_columns(
            split=pl.when(pl.col("source") == "reid2019")
            .then(pl.lit("test"))
            .otherwise(
                pl.col("text_id").map_elements(
                    assign_split,
                    return_dtype=pl.String,
                )
            )
        )
    )

    # map_elements skips nulls, so a missing text_id would vanish from every split
    if n := keys["split"].null_count():
        raise ValueError(f"{n} synth rows have no text_id")
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {}

    for name in SPLITS:
        ids = keys.filter(pl.col("split") == name).select("id")

        lf.join(
            ids.lazy(),
            on="id",
            how="semi",
            maintain_order="left",
        ).sink_parquet(out_dir / f"{name}.parquet")

        counts[name] = dict(
            ids.join(keys, on="id").group_by("source").len().sort("source").iter_rows()
        )

    return counts


if __name__ == "__main__":
    for name, by_source in split().items():
        print(f"{name:5}  {sum(by_source.values()):6}  {by_source}")
