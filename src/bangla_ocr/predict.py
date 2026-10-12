"""Run an OCR engine over a split and cache predictions: one row per (engine, id), resumable.

Layout: PRED_DIR/<engine>/<config_id>/{config.json, part-00000.parquet, ...}
config_id hashes the engine config + the input file's sha256, so a new setting or new data
never reuses old rows.
"""

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from bangla_ocr import checks
from bangla_ocr.download import sha256_file

PRED_DIR = Path("data/predictions")
BATCH = 50
SCHEMA = pl.Schema({"id": pl.String, "pred": pl.String, "seconds": pl.Float64, "error": pl.String})


@dataclass(frozen=True)
class Engine:
    name: str
    config: dict  # engine version, model hash, settings: anything that changes the output
    read: Callable[[dict], str]  # row {"id", "source", "image"} -> raw text


def config_id(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def run_config(engine: Engine, split_path: Path) -> dict:
    return {"engine": engine.name, **engine.config, "data_sha256": sha256_file(split_path)}


def run_dir(engine: Engine, split_path: Path, out_dir: Path = PRED_DIR) -> Path:
    return out_dir / engine.name / config_id(run_config(engine, split_path))


def load(path: Path) -> pl.DataFrame:
    """Latest row per id across parts (a retried error is superseded by its retry)."""
    parts = sorted(path.glob("part-*.parquet"))
    if not parts:
        return SCHEMA.to_frame()
    return pl.concat(pl.read_parquet(p) for p in parts).unique(
        "id", keep="last", maintain_order=True
    )


def _predict(engine: Engine, row: dict) -> dict:
    start = time.perf_counter()
    try:
        pred, error = engine.read(row), None
    except Exception as e:  # one bad image must not stop the run; scored as CER 1
        pred, error = None, f"{type(e).__name__}: {e}"
    return {"id": row["id"], "pred": pred, "seconds": time.perf_counter() - start, "error": error}


def _write_part(path: Path, rows: list[dict]) -> None:
    n = len(list(path.glob("part-*.parquet")))
    tmp = path / f"part-{n:05d}.parquet.tmp"
    pl.DataFrame(rows, schema=SCHEMA).write_parquet(tmp)
    tmp.replace(path / f"part-{n:05d}.parquet")  # atomic: a crash never leaves half a part


def run(
    engine: Engine,
    split_path: Path,
    out_dir: Path = PRED_DIR,
    limit: int | None = None,
    batch: int = BATCH,
) -> Path:
    """Predict every row of `split_path` (first `limit` ids, sorted) not yet done without error."""
    config = run_config(engine, split_path)
    path = out_dir / engine.name / config_id(config)
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    ids = pl.scan_parquet(split_path).select("id").sort("id").collect()["id"]
    if limit is not None:
        ids = ids.head(limit)
    done = load(path).filter(pl.col("error").is_null())["id"]
    todo = ids.filter(~ids.is_in(done.implode()))
    print(f"{engine.name} [{path.name}]: {len(ids) - len(todo)} cached, {len(todo)} to run")

    for start in range(0, len(todo), batch):
        chunk = todo.slice(start, batch)
        rows = (
            pl.scan_parquet(split_path)
            .filter(pl.col("id").is_in(chunk.implode()))
            .select("id", "source", "image")
            .collect()
        )
        results = [_predict(engine, r) for r in rows.iter_rows(named=True)]
        _write_part(path, results)
        errors = sum(r["error"] is not None for r in results)
        print(f"  {start + len(chunk)}/{len(todo)}  errors in batch: {errors}")
    return path


def engines() -> dict[str, Callable[[], Engine]]:
    from bangla_ocr import easyocr_engine, tesseract  # lazily: each engine has heavy deps

    return {"tesseract": tesseract.engine, "easyocr": easyocr_engine.engine}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=sorted(engines()))
    ap.add_argument("--split", default="test", choices=sorted(checks.split_paths()))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    out = run(engines()[args.engine](), checks.split_paths()[args.split], limit=args.limit)
    print(out)
