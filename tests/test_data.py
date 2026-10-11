"""Data tests: run against the REAL processed files (skipped if not built).

These are the contract for data/processed/. Phase 4 turns them into Dagster asset checks.
"""

import json

import polars as pl
import pytest

from bangla_ocr import clean
from bangla_ocr.ingest import SCHEMA
from bangla_ocr.normalise import normalise_text
from bangla_ocr.split import OUT_DIR, SPLITS

PATHS = {name: OUT_DIR / f"{name}.parquet" for name in SPLITS}

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in PATHS.values()), reason="processed data not built"
)


@pytest.fixture(scope="module")
def meta() -> pl.DataFrame:
    """All splits, small columns only (no image bytes)."""
    return pl.concat(
        pl.scan_parquet(p)
        .select("id", "source", "text", "text_id", has_image=pl.col("image").is_not_null())
        .with_columns(split=pl.lit(name))
        .collect()
        for name, p in PATHS.items()
    )


def test_schema():
    for p in PATHS.values():
        assert pl.read_parquet_schema(p) == pl.Schema({**SCHEMA, "text_raw": pl.String})


def test_rows_in_equal_rows_out(meta):
    report = json.loads(clean.REPORT_PATH.read_text(encoding="utf-8"))
    assert meta.height == report["output"]


def test_ids_unique_across_splits(meta):
    assert meta["id"].n_unique() == meta.height


def test_every_row_has_an_image(meta):
    assert meta["has_image"].all()


def test_reid_only_in_test(meta):
    assert meta.filter(pl.col("source") == "reid2019")["split"].unique().to_list() == ["test"]


@pytest.mark.parametrize("col", ["text_id", "text"])
def test_no_group_or_text_in_two_splits(meta, col):
    leaks = (
        meta.drop_nulls(col)
        .group_by(col)
        .agg(pl.col("split").n_unique().alias("n"))
        .filter(pl.col("n") > 1)
    )
    assert leaks.height == 0, leaks.head()


def test_labels_normalised_and_non_empty(meta):
    texts = meta["text"].to_list()
    assert all(texts), "empty label"
    bad = [t for t in texts if normalise_text(t) != t]
    assert not bad, bad[:5]


def test_synth_split_shares(meta):
    synth = meta.filter(pl.col("source") == "synth")
    shares = dict(synth.group_by("split").len().iter_rows())
    for name, target in {"train": 0.80, "val": 0.10, "test": 0.10}.items():
        assert abs(shares[name] / synth.height - target) <= 0.02, (name, shares)
