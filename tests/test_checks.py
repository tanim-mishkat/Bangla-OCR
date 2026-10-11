import polars as pl

from bangla_ocr.checks import find_leaks, share_problems


def meta(train: int, val: int, test: int, reid_test: int = 0) -> pl.DataFrame:
    """Synth rows per split (+ optional REID rows in test)."""
    splits = ["train"] * train + ["val"] * val + ["test"] * test
    sources = ["synth"] * len(splits) + ["reid2019"] * reid_test
    return pl.DataFrame({"source": sources, "split": splits + ["test"] * reid_test})


# --- share_problems (you write it) ---


def test_exact_targets_pass():
    assert share_problems(meta(80, 10, 10)) == []


def test_within_tolerance_pass():
    assert share_problems(meta(815, 95, 90)) == []


def test_just_outside_tolerance_reported():
    assert share_problems(meta(825, 85, 90)) == ["train: 0.825 (target 0.80)"]


def test_every_bad_split_reported_in_target_order():
    assert share_problems(meta(70, 20, 10)) == [
        "train: 0.700 (target 0.80)",
        "val: 0.200 (target 0.10)",
    ]


def test_missing_split_counts_as_zero():
    assert share_problems(meta(90, 10, 0)) == [
        "train: 0.900 (target 0.80)",
        "test: 0.000 (target 0.10)",
    ]


def test_reid_rows_ignored():
    # 677 REID rows in test would push test to ~0.4 if counted
    assert share_problems(meta(80, 10, 10, reid_test=677)) == []


def test_no_synth_rows():
    assert share_problems(meta(0, 0, 0, reid_test=5)) == ["no synth rows"]


def test_custom_targets_and_tol():
    assert share_problems(meta(50, 50, 0), {"train": 0.5, "val": 0.5}, tol=0.0) == []


# --- find_leaks ---


def test_find_leaks():
    df = pl.DataFrame(
        {
            "text_id": ["b", "b", "a", "a", "c", None, None],
            "split": ["train", "train", "train", "test", "val", "test", "train"],
        }
    )
    assert find_leaks(df, "text_id") == ["a"]  # same split twice is fine, nulls ignored
