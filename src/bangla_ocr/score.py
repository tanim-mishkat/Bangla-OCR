"""Score cached predictions: per-row edit counts -> CER / WER per slice.

CER = sum(char edits) / sum(label chars) over a slice (micro average: long lines weigh more).
A missing or errored prediction scores as "" (every label char is an edit), never dropped.
"""

import argparse
import json
from pathlib import Path

import polars as pl

from bangla_ocr import checks, predict
from bangla_ocr.download import sha256_file
from bangla_ocr.metrics import edit_distance
from bangla_ocr.normalise import normalise_text

LABEL_COLS = ["id", "text_id", "source", "level", "font", "region_type", "text"]
# REID and synth are different tasks: never blended into one number
REPORTS = {"synth": [[], ["level"], ["font"]], "reid2019": [[], ["region_type"]]}


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


def summarise(rows: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    stats = [
        pl.len().alias("n"),
        (pl.col("char_edits").sum() / pl.col("chars").sum()).alias("cer"),
        (pl.col("word_edits").sum() / pl.col("words").sum()).alias("wer"),
        pl.col("empty").mean().alias("empty_rate"),
        pl.col("error").is_not_null().sum().alias("errors"),
    ]
    if not by:
        return rows.select(stats)
    return rows.group_by(by).agg(stats).sort(by)


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
    ap.add_argument("--split", default="test", choices=sorted(checks.split_paths()))
    args = ap.parse_args()
    rows = score(args.run_dir, checks.split_paths()[args.split])
    with pl.Config(tbl_rows=-1, float_precision=3, tbl_hide_dataframe_shape=True):
        for source, groupings in REPORTS.items():
            subset = rows.filter(pl.col("source") == source)
            for by in groupings:
                print(f"\n{source} by {', '.join(by) or 'all'}")
                print(summarise(subset, by))
