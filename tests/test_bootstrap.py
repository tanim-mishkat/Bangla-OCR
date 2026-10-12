import numpy as np
import polars as pl
import pytest

from bangla_ocr.bootstrap import cluster_bootstrap, paired_bootstrap


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


# --- paired: engine A vs engine B on the same rows ---


def better_by_one(df: pl.DataFrame) -> pl.DataFrame:
    """Engine B: one fewer error than A on every row that has any."""
    return df.with_columns((pl.col("char_edits") - 1).clip(lower_bound=0))


def cer(df: pl.DataFrame) -> float:
    return df["char_edits"].sum() / df["chars"].sum()


def test_paired_point_is_the_cer_difference():
    a = fonts_like()
    b = better_by_one(a)
    diff, low, high = paired_bootstrap(a, b, "c")
    assert diff == pytest.approx(cer(a) - cer(b))
    assert low <= diff <= high


def test_identical_engines_differ_by_exactly_zero():
    a = fonts_like()
    assert paired_bootstrap(a, a, "c") == (0.0, 0.0, 0.0)


def test_pairing_detects_what_separate_intervals_cannot():
    # texts vary a lot in difficulty -> wide separate CIs that overlap;
    # pairing cancels text difficulty, so a small consistent gain is clearly > 0
    a = fonts_like()
    b = better_by_one(a)
    a_low, _ = cluster_bootstrap(a, "c")
    _, b_high = cluster_bootstrap(b, "c")
    assert a_low < b_high  # separate intervals overlap: "no clear winner"
    _, low, _ = paired_bootstrap(a, b, "c")
    assert low > 0  # paired: A is worse, with 95% confidence


def test_paired_refuses_different_rows():
    a = fonts_like()
    with pytest.raises(ValueError, match="different rows"):
        paired_bootstrap(a, a.head(10), "c")
    with pytest.raises(ValueError, match="different rows"):
        paired_bootstrap(a, a.with_columns(pl.col("id") + "x"), "c")


def test_paired_refuses_different_labels():
    a = fonts_like()
    with pytest.raises(ValueError, match="different labels"):
        paired_bootstrap(a, a.with_columns(pl.col("chars") + 1), "c")
