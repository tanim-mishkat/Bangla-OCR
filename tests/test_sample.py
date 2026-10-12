import json

import polars as pl
import pytest

from bangla_ocr.sample import composition, ids_sha256, load, sample_ids, write

# --- sample_ids (you write it) ---
# Expected values are hard-coded from sha256, not computed with the code under test.
# seed 7 ranks: synth t3 t1 t2 t0 t5 t4 | reid p2 p0 p1 p3
# seed 8 ranks: synth t0 t5 t3 t4 t2 t1 | reid p2 p3 p0 p1


def make_meta() -> pl.DataFrame:
    synth = [
        {"id": f"synth/t{i}/{font}", "source": "synth", "text_id": f"t{i}", "page": None}
        for i in range(6)
        for font in "ab"
    ]
    reid = [
        {"id": f"reid2019/p{p}/r{j}", "source": "reid2019", "text_id": None, "page": f"p{p}"}
        for p, blocks in enumerate([1, 2, 3, 1])
        for j in range(blocks)
    ]
    return pl.DataFrame(synth + reid, schema={c: pl.String for c in synth[0]})


META = make_meta()
SEED7 = [
    "reid2019/p2/r0",
    "reid2019/p2/r1",
    "reid2019/p2/r2",
    "synth/t1/a",
    "synth/t1/b",
    "synth/t3/a",
    "synth/t3/b",
]


def test_known_sample():
    assert sample_ids(META, {"synth": 2, "reid2019": 1}, seed=7) == SEED7


def test_other_seed_other_sample():
    ids = sample_ids(META, {"synth": 2, "reid2019": 1}, seed=8)
    assert [i for i in ids if i.startswith("synth")] == [
        "synth/t0/a",
        "synth/t0/b",
        "synth/t5/a",
        "synth/t5/b",
    ]


def test_row_order_does_not_matter():
    shuffled = META.sample(fraction=1.0, shuffle=True, seed=0)
    assert sample_ids(shuffled, {"synth": 2, "reid2019": 1}, seed=7) == SEED7


def test_whole_clusters_and_exact_quotas():
    ids = sample_ids(META, {"synth": 4, "reid2019": 2}, seed=7)
    picked = META.filter(pl.col("id").is_in(ids))
    synth, reid = (picked.filter(pl.col("source") == s) for s in ["synth", "reid2019"])
    assert synth["text_id"].n_unique() == 4 and synth.height == 8  # both fonts of each text
    assert set(reid["page"]) == {"p2", "p0"} and reid.height == 4  # every block of each page


def test_sorted_and_unique():
    ids = sample_ids(META, {"synth": 6, "reid2019": 4}, seed=7)
    assert ids == sorted(set(ids)) and len(ids) == META.height


def test_bigger_quota_keeps_smaller_sample():
    small = set(sample_ids(META, {"synth": 2, "reid2019": 1}, seed=7))
    assert small <= set(sample_ids(META, {"synth": 4, "reid2019": 3}, seed=7))


def test_unlisted_or_zero_quota_source_not_sampled():
    assert sample_ids(META, {"synth": 2}, seed=7) == SEED7[3:]
    assert sample_ids(META, {"synth": 2, "reid2019": 0}, seed=7) == SEED7[3:]


def test_quota_above_clusters_raises():
    with pytest.raises(ValueError):
        sample_ids(META, {"synth": 7}, seed=7)


def test_null_cluster_key_raises():
    bad = META.with_columns(
        text_id=pl.when(pl.col("id") == "synth/t0/a").then(None).otherwise(pl.col("text_id"))
    )
    with pytest.raises(ValueError):
        sample_ids(bad, {"synth": 2}, seed=7)


def test_returns_python_list_of_str():
    ids = sample_ids(META, {"synth": 1}, seed=7)
    assert isinstance(ids, list) and all(isinstance(i, str) for i in ids)


# --- helpers (written for you) ---


def test_write_load_roundtrip(tmp_path):
    path = write(SEED7, 7, {"synth": 2, "reid2019": 1}, tmp_path / "s.json")
    assert load(path) == SEED7
    assert json.loads(path.read_text(encoding="utf-8"))["n"] == 7


def test_load_rejects_edited_ids(tmp_path):
    path = write(SEED7, 7, {"synth": 2}, tmp_path / "s.json")
    record = json.loads(path.read_text(encoding="utf-8"))
    record["ids"].append("synth/t9/a")
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        load(path)


def test_ids_sha256_depends_on_order_and_content():
    assert ids_sha256(["a", "b"]) != ids_sha256(["b", "a"])
    assert ids_sha256(["a", "b"]) != ids_sha256(["a", "bb"])


def test_composition_shares():
    out = composition(META, SEED7, "source").sort("source")
    assert out["full"].to_list() == pytest.approx([7 / 19, 12 / 19])
    assert out["sample"].to_list() == pytest.approx([3 / 7, 4 / 7])
