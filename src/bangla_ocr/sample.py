"""Fixed-seed cluster sample of a split, for engines too slow or rate-limited for all of it.

Samples whole clusters (synth text_id, REID page): the units the bootstrap resamples, so CIs on
the sample stay honest. Selection sees only ids and cluster keys, never any engine's output.
"""

import hashlib
import json
from pathlib import Path

import polars as pl

from bangla_ocr import checks, score

SEED = 20261012  # fixed before sampling, never changed after seeing results
QUOTAS = {"synth": 60, "reid2019": 20}  # clusters per source, sized to the API quota
CLUSTER = {source: cluster for source, (cluster, _) in score.REPORTS.items()}
OUT_PATH = Path("data/samples/test.json")


def sample_ids(meta: pl.DataFrame, quotas: dict[str, int], seed: int = SEED) -> list[str]:
    """Sorted ids of every row in the sampled clusters.

    meta: one row per id with columns id, source and each CLUSTER column.
    Per source in `quotas`: rank its distinct cluster keys by
    sha256(f"{seed}:{key}" as utf-8).hexdigest(), keep the `quotas[source]` smallest.
    Sources missing from `quotas` are not sampled.
    Raises ValueError if a quota exceeds the source's clusters, or a sampled source has
    rows with a null cluster key (they would silently never be picked).
    """

    ids = []

    for source, k in quotas.items():
        rows = meta.filter(pl.col("source") == source)
        col = CLUSTER[source]

        if rows[col].null_count() > 0:
            raise ValueError(f"{source}: null cluster keys in {col}")

        keys = sorted(
            rows[col].unique().to_list(),
            key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(),
        )

        if k > len(keys):
            raise ValueError(f"{source}: requested {k} clusters, only {len(keys)} available")

        selected = keys[:k]
        ids.extend(rows.filter(pl.col(col).is_in(selected))["id"].to_list())

    return sorted(ids)


def ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def write(ids: list[str], seed: int, quotas: dict[str, int], path: Path = OUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"seed": seed, "quotas": quotas, "n": len(ids), "sha256": ids_sha256(ids), "ids": ids}
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def load(path: Path = OUT_PATH) -> list[str]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if ids_sha256(record["ids"]) != record["sha256"]:
        raise ValueError(f"{path}: ids do not match their sha256 (edited by hand?)")
    return record["ids"]


def composition(meta: pl.DataFrame, ids: list[str], by: str) -> pl.DataFrame:
    """Share of rows per `by` value: full split vs sample. A check, never a selection input."""

    def shares(df: pl.DataFrame, name: str) -> pl.DataFrame:
        return df.group_by(by).agg((pl.len() / df.height).alias(name))

    sample = meta.filter(pl.col("id").is_in(ids))
    return shares(meta, "full").join(shares(sample, "sample"), on=by, how="full", coalesce=True)


if __name__ == "__main__":
    test = pl.read_parquet(
        checks.split_paths()["test"],
        columns=["id", "source", "level", "font", "region_type", *CLUSTER.values()],
    )
    ids = sample_ids(test, QUOTAS)
    print(f"{write(ids, SEED, QUOTAS)}: {len(ids)} ids, sha256 {ids_sha256(ids)[:12]}")
    for source, by in [("synth", "level"), ("synth", "font"), ("reid2019", "region_type")]:
        part = test.filter(pl.col("source") == source)
        print(f"\n{source} rows sampled: {part['id'].is_in(ids).sum()} / {part.height}")
        print(composition(part, ids, by).sort(by))
