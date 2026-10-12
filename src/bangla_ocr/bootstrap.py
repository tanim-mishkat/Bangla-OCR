"""Confidence intervals for CER/WER that respect clustered test data."""

import numpy as np
import polars as pl

N_BOOT = 1000


def cluster_bootstrap(
    rows: pl.DataFrame,
    cluster: str,
    num: str = "char_edits",
    den: str = "chars",
    n_boot: int = N_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI for sum(num) / sum(den), resampling whole clusters with replacement.

    cluster: column whose rows are not independent (synth: "text_id", REID: "page").
    """
    if rows.is_empty():
        raise ValueError("no rows")

    if rows[cluster].null_count() > 0:
        raise ValueError("null cluster keys")

    groups = (
        rows.group_by(cluster)
        .agg(
            pl.col(num).sum(),
            pl.col(den).sum(),
        )
        .sort(cluster)  # group_by order is not guaranteed; sort for reproducibility
    )

    errors = groups[num].to_numpy()
    chars = groups[den].to_numpy()

    k = groups.height
    rng = np.random.default_rng(seed)

    indices = rng.integers(0, k, size=(n_boot, k))

    scores = errors[indices].sum(axis=1) / chars[indices].sum(axis=1)

    low, high = np.quantile(
        scores,
        [alpha / 2, 1 - alpha / 2],
    )

    return float(low), float(high)


def paired_bootstrap(
    a: pl.DataFrame,
    b: pl.DataFrame,
    cluster: str,
    num: str = "char_edits",
    den: str = "chars",
    ref: str = "text",
    **kwargs,
) -> tuple[float, float, float]:
    """(diff, low, high) for ratio(a) - ratio(b), both engines resampled on the same clusters.

    Same labels -> same den per row, so the diff is sum(num_a - num_b) / sum(den): shared
    difficulty (a hard page is hard for both) cancels before resampling.
    ref: label column; must be identical per id (equal lengths alone are not enough).
    """
    if a.height != b.height or a.join(b, on="id", how="anti").height:
        raise ValueError("engines were scored on different rows")
    both = a.select("id", cluster, ref, num, den).join(
        b.select(
            "id",
            pl.col(ref).alias("_ref_b"),
            pl.col(num).alias("_num_b"),
            pl.col(den).alias("_den_b"),
        ),
        on="id",
    )
    if (both[ref] != both["_ref_b"]).any() or (both[den] != both["_den_b"]).any():
        raise ValueError("engines were scored against different labels")
    both = both.with_columns(pl.col(num) - pl.col("_num_b"))
    low, high = cluster_bootstrap(both, cluster, num, den, **kwargs)
    return both[num].sum() / both[den].sum(), low, high
