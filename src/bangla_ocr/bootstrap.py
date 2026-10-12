"""Confidence intervals for CER/WER that respect clustered test data."""

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
    raise NotImplementedError  # your turn
