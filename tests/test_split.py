from collections import Counter

import polars as pl
import pytest

from bangla_ocr.ingest import SCHEMA
from bangla_ocr.split import assign_split, split

# --- assign_split (you write it) ---
# Expected values are hard-coded from sha256, not computed with the code under test.


@pytest.mark.parametrize(
    "key, expected",
    [
        ("wiki_bn/00134", "train"),  # bucket 0
        ("wiki_bn/00016", "train"),  # bucket 79: last train
        ("wiki_bn/00063", "val"),  # bucket 80: first val
        ("wiki_bn/00291", "val"),  # bucket 89: last val
        ("wiki_bn/00045", "test"),  # bucket 90: first test
        ("wiki_bn/00108", "test"),  # bucket 99
        ("আমি", "train"),  # bucket 7: non-ASCII key encodes as utf-8
    ],
)
def test_known_buckets(key, expected):
    assert assign_split(key) == expected


def test_empty_key_raises():
    with pytest.raises(ValueError):
        assign_split("")


def test_deterministic():
    assert [assign_split("wiki_bn/00063") for _ in range(3)] == ["val"] * 3


def test_proportions_roughly_80_10_10():
    counts = Counter(assign_split(f"wiki_bn/{i:05d}") for i in range(10_000))
    assert 7_800 <= counts["train"] <= 8_200
    assert 900 <= counts["val"] <= 1_100
    assert 900 <= counts["test"] <= 1_100


# --- split ---


def row(id_: str, source: str, text_id: str | None) -> dict:
    return {
        **dict.fromkeys(SCHEMA),
        "id": id_,
        "source": source,
        "text_id": text_id,
        "image": id_.encode(),
        "text": "ক",
        "text_raw": "ক",
    }


@pytest.fixture
def clean_file(tmp_path):
    rows = [row(f"reid2019/p{i}", "reid2019", None) for i in range(3)]
    for tid in ("wiki_bn/00134", "wiki_bn/00063", "wiki_bn/00045"):  # train, val, test
        rows += [row(f"synth/{tid}/{f}", "synth", tid) for f in ("Galada", "Hind")]
    path = tmp_path / "clean.parquet"
    pl.DataFrame(rows, schema={**SCHEMA, "text_raw": pl.String}).write_parquet(path)
    return path


def test_split_files(clean_file, tmp_path):
    out = tmp_path / "processed"
    counts = split(clean_file, out)

    dfs = {name: pl.read_parquet(out / f"{name}.parquet") for name in ("train", "val", "test")}
    assert counts == {
        "train": {"synth": 2},
        "val": {"synth": 2},
        "test": {"reid2019": 3, "synth": 2},
    }
    assert dfs["train"]["text_id"].unique().to_list() == ["wiki_bn/00134"]
    assert dfs["val"]["text_id"].unique().to_list() == ["wiki_bn/00063"]
    assert dfs["test"].filter(pl.col("source") == "synth")["text_id"].unique().to_list() == [
        "wiki_bn/00045"
    ]
    # every row lands in exactly one split, with its own image
    all_ids = pl.concat(dfs.values())["id"]
    assert all_ids.n_unique() == all_ids.len() == 9
    assert all(df["image"].to_list() == [i.encode() for i in df["id"]] for df in dfs.values())
    assert dfs["test"].columns == pl.read_parquet(clean_file).columns


def test_synth_row_without_text_id_fails_loudly(tmp_path):
    path = tmp_path / "clean.parquet"
    rows = [row("synth/x/Galada", "synth", None)]
    pl.DataFrame(rows, schema={**SCHEMA, "text_raw": pl.String}).write_parquet(path)
    with pytest.raises(ValueError, match="1 synth rows have no text_id"):
        split(path, tmp_path / "processed")
