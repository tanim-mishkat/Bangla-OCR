import numpy as np
import polars as pl
import pytest

from bangla_ocr.bootstrap import cluster_bootstrap


def rows(edits: list[int], chars: list[int], clusters: list) -> pl.DataFrame:
    return pl.DataFrame(
        {"char_edits": edits, "chars": chars, "c": clusters},
        schema_overrides={"c": pl.String},
    )


def fonts_like(n_texts=60, n_fonts=5, seed=1) -> pl.DataFrame:
    """Like synth: each text rendered in 5 fonts, so a hard text is hard in every font."""
    rng = np.random.default_rng(seed)
    difficulty = rng.uniform(0, 0.4, n_texts)
    chars = rng.integers(20, 60, n_texts)
    return pl.DataFrame(
        {
            "char_edits": [int(d * c) for d, c in zip(difficulty, chars, strict=True)] * n_fonts,
            "chars": chars.tolist() * n_fonts,
            "c": [f"t{i}" for i in range(n_texts)] * n_fonts,
            "id": [f"r{i}" for i in range(n_texts * n_fonts)],
        }
    )


def test_same_seed_same_interval():
    df = fonts_like()
    assert cluster_bootstrap(df, "c", seed=3) == cluster_bootstrap(df, "c", seed=3)
    assert cluster_bootstrap(df, "c", seed=3) != cluster_bootstrap(df, "c", seed=4)


def test_interval_contains_point_estimate():
    df = fonts_like()
    low, high = cluster_bootstrap(df, "c")
    point = df["char_edits"].sum() / df["chars"].sum()
    assert low < point < high
    assert isinstance(low, float) and isinstance(high, float)


def test_one_cluster_has_no_spread():
    # rows differ, but whole-cluster resampling always draws the same single cluster
    low, high = cluster_bootstrap(rows([0, 5, 9], [10, 10, 10], ["a", "a", "a"]), "c")
    assert low == high == pytest.approx(14 / 30)


def test_ratio_of_sums_not_mean_of_ratios():
    # 2 clusters -> resamples AA, AB, BA, BB; the middle 50% of draws are AB/BA = 2/12
    df = rows([0, 2], [10, 2], ["a", "b"])
    low, high = cluster_bootstrap(df, "c", n_boot=2000, alpha=0.8)
    assert low == high == pytest.approx(2 / 12)  # mean of ratios would give 0.5


def test_clusters_widen_the_interval():
    # resampling rows treats 300 correlated rows as independent: too narrow
    df = fonts_like()
    low_c, high_c = cluster_bootstrap(df, "c")
    low_r, high_r = cluster_bootstrap(df, "id")
    assert high_c - low_c > 1.8 * (high_r - low_r)  # ~sqrt(5) with 5 copies per text


def test_null_cluster_key_refuses():
    # REID has text_id = null: clustering by it would make one cluster and a zero-width CI
    with pytest.raises(ValueError, match="null"):
        cluster_bootstrap(rows([1, 2], [5, 5], [None, None]), "c")


def test_empty_refuses():
    with pytest.raises(ValueError, match="no rows"):
        cluster_bootstrap(rows([], [], []), "c")
