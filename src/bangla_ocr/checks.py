"""Data contract for data/processed/. Each check returns a list of problems (empty = pass).

Used by tests/test_data.py and as blocking Dagster asset checks on `splits`.
"""

import json
from pathlib import Path

import polars as pl

from bangla_ocr import clean, split
from bangla_ocr.ingest import SCHEMA
from bangla_ocr.normalise import normalise_text

EXPECTED_SCHEMA = pl.Schema({**SCHEMA, "text_raw": pl.String})
SYNTH_TARGETS = {"train": 0.80, "val": 0.10, "test": 0.10}

NAMES = (
    "schema",
    "rows_match_report",
    "ids_unique",
    "every_row_has_image",
    "reid_only_in_test",
    "no_text_id_in_two_splits",
    "no_text_in_two_splits",
    "labels_normalised_non_empty",
    "synth_shares",
)


def split_paths(out_dir: Path = split.OUT_DIR) -> dict[str, Path]:
    return {name: out_dir / f"{name}.parquet" for name in split.SPLITS}


def load_meta(out_dir: Path = split.OUT_DIR) -> pl.DataFrame:
    """All splits, small columns only (no image bytes), plus `split` and `has_image`."""
    return pl.concat(
        pl.scan_parquet(p)
        .select(
            "id",
            "source",
            "text",
            "text_id",
            has_image=pl.col("image").is_not_null(),
        )
        .with_columns(split=pl.lit(name))
        .collect()
        for name, p in split_paths(out_dir).items()
    )


def find_leaks(meta: pl.DataFrame, col: str) -> list[str]:
    """Values of `col` found in more than one split, sorted. Nulls ignored (REID has no text_id)."""
    return (
        meta.drop_nulls(col)
        .group_by(col)
        .agg(n=pl.col("split").n_unique())
        .filter(pl.col("n") > 1)[col]
        .sort()
        .to_list()
    )


def share_problems(
    meta: pl.DataFrame,
    targets: dict[str, float] = SYNTH_TARGETS,
    tol: float = 0.02,
) -> list[str]:
    """Synth rows' share per split vs `targets`. One message per split off by more than `tol`.

    Message format (in `targets` order): "val: 0.150 (target 0.10)"
    A split with no synth rows has share 0.0. No synth rows at all -> ["no synth rows"].
    REID rows are ignored (they are test-only by design, not by hashing).
    """
    synth = meta.filter(pl.col("source") == "synth")

    if synth.is_empty():
        return ["no synth rows"]

    counts = dict(synth.group_by("split").len().iter_rows())

    problems = []

    for name, target in targets.items():
        share = counts.get(name, 0) / synth.height

        if abs(share - target) > tol:
            problems.append(f"{name}: {share:.3f} (target {target:.2f})")

    return problems


def run(
    out_dir: Path = split.OUT_DIR,
    report_path: Path = clean.REPORT_PATH,
) -> dict[str, list[str]]:
    """Every check on the real files, keyed by NAMES."""
    meta = load_meta(out_dir)
    expected_rows = json.loads(report_path.read_text(encoding="utf-8"))["output"]

    ids = meta["id"]

    return {
        "schema": [
            f"{name}: {schema}"
            for name, p in split_paths(out_dir).items()
            if (schema := pl.read_parquet_schema(p)) != EXPECTED_SCHEMA
        ],
        "rows_match_report": (
            [] if meta.height == expected_rows else [f"{meta.height} rows, report: {expected_rows}"]
        ),
        "ids_unique": (ids.filter(ids.is_duplicated()).unique().sort().to_list()),
        "every_row_has_image": (meta.filter(~pl.col("has_image"))["id"].to_list()),
        "reid_only_in_test": (
            meta.filter((pl.col("source") == "reid2019") & (pl.col("split") != "test"))[
                "id"
            ].to_list()
        ),
        "no_text_id_in_two_splits": find_leaks(meta, "text_id"),
        "no_text_in_two_splits": find_leaks(meta, "text"),
        "labels_normalised_non_empty": [
            i for i, t in zip(ids, meta["text"], strict=True) if not t or normalise_text(t) != t
        ],
        "synth_shares": share_problems(meta),
    }
