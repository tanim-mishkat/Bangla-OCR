"""Score cached predictions: per-row edit counts -> CER / WER per slice.

CER = sum(char edits) / sum(label chars) over a slice (micro average: long lines weigh more).
A missing or errored prediction scores as "" (every label char is an edit), never dropped.
CIs resample clusters of non-independent rows: synth by text_id (5 fonts), REID by page.
"""

import argparse
import json
from pathlib import Path

import polars as pl

from bangla_ocr import checks, predict
from bangla_ocr.bootstrap import cluster_bootstrap, paired_bootstrap
from bangla_ocr.download import sha256_file
from bangla_ocr.metrics import edit_distance
from bangla_ocr.normalise import normalise_text

LABEL_COLS = ["id", "text_id", "page", "source", "level", "font", "region_type", "text"]
# REID and synth are different tasks: never blended into one number.
# source -> (bootstrap cluster, groupings to report)
REPORTS = {
    "synth": ("text_id", [[], ["level"], ["font"]]),
    "reid2019": ("page", [[], ["region_type"]]),
}


def score_rows(labels: pl.DataFrame, preds: pl.DataFrame) -> pl.DataFrame:
    """One row per label with char/word edit counts. Every label must have a prediction row."""
    missing = labels.join(preds, on="id", how="anti").height
    if missing:
        raise ValueError(f"{missing} ids have no prediction: finish the run before scoring")
    df = labels.join(preds.select("id", "pred", "error"), on="id", how="left")
    rows = []
    for r in df.iter_rows(named=True):
        ref, hyp = r["text"], normalise_text(r["pred"] or "")
        rows.append(
            {
                "char_edits": edit_distance(ref, hyp),
                "chars": len(ref),
                "word_edits": edit_distance(ref.split(), hyp.split()),
                "words": len(ref.split()),
                "empty": hyp == "",
            }
        )
    return pl.concat([df.rename({"pred": "pred_raw"}), pl.DataFrame(rows)], how="horizontal")


def _cer(rows: pl.DataFrame) -> float:
    return rows["char_edits"].sum() / rows["chars"].sum()


def _slices(rows: pl.DataFrame, by: list[str], keys: list[tuple]) -> list[pl.DataFrame]:
    """Rows of each group key, in the order given. eq_missing: a null key matches null rows."""
    return [
        rows.filter(pl.lit(True), *(pl.col(c).eq_missing(v) for c, v in zip(by, k, strict=True)))
        for k in keys
    ]


def summarise(rows: pl.DataFrame, by: list[str], cluster: str | None = None) -> pl.DataFrame:
    """CER/WER per group; with `cluster`, adds a 95% cluster-bootstrap CI for CER."""
    stats = [
        pl.len().alias("n"),
        (pl.col("char_edits").sum() / pl.col("chars").sum()).alias("cer"),
        (pl.col("word_edits").sum() / pl.col("words").sum()).alias("wer"),
        pl.col("empty").mean().alias("empty_rate"),
        pl.col("error").is_not_null().sum().alias("errors"),
    ]
    out = rows.group_by(by).agg(stats).sort(by) if by else rows.select(stats)
    if cluster is None:
        return out
    ci = []
    for g in _slices(rows, by, out.select(by).rows() if by else [()]):
        low, high = cluster_bootstrap(g, cluster)
        ci.append({"clusters": g[cluster].n_unique(), "cer_low": low, "cer_high": high})
    out = pl.concat([out, pl.DataFrame(ci)], how="horizontal")
    cols = ["n", "clusters", "cer", "cer_low", "cer_high", "wer", "empty_rate", "errors"]
    return out.select(*by, *cols)


def compare(a: pl.DataFrame, b: pl.DataFrame, by: list[str], cluster: str) -> pl.DataFrame:
    """Paired CER(a) - CER(b) per group: negative = a better. `clear` = CI excludes 0."""
    keys = a.select(by).unique().sort(by).rows() if by else [()]
    out = []
    for key, ga, gb in zip(keys, _slices(a, by, keys), _slices(b, by, keys), strict=True):
        diff, low, high = paired_bootstrap(ga, gb, cluster)
        out.append(
            {
                **dict(zip(by, key, strict=True)),
                "n": ga.height,
                "cer_a": _cer(ga),
                "cer_b": _cer(gb),
                "diff": diff,
                "low": low,
                "high": high,
                "clear": low > 0 or high < 0,
            }
        )
    return pl.DataFrame(out)


def score(run_dir: Path, split_path: Path) -> pl.DataFrame:
    """Rows for a finished run; refuses predictions made on different input data."""
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config["data_sha256"] != sha256_file(split_path):
        raise ValueError(f"{run_dir} was predicted on different data than {split_path}")
    labels = pl.read_parquet(split_path, columns=LABEL_COLS)
    return score_rows(labels, predict.load(run_dir))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="e.g. data/predictions/tesseract/<config_id>")
    ap.add_argument("--vs", type=Path, help="second run dir: paired CER(run_dir) - CER(vs)")
    ap.add_argument("--split", default="test", choices=sorted(checks.split_paths()))
    args = ap.parse_args()
    split = checks.split_paths()[args.split]
    rows = score(args.run_dir, split)
    other = score(args.vs, split) if args.vs else None
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3, tbl_hide_dataframe_shape=True):
        for source, (cluster, groupings) in REPORTS.items():
            a = rows.filter(pl.col("source") == source)
            for by in groupings:
                print(f"\n{source} by {', '.join(by) or 'all'} (CI clusters: {cluster})")
                if other is None:
                    print(summarise(a, by, cluster))
                else:
                    print(compare(a, other.filter(pl.col("source") == source), by, cluster))
