"""Combine all sources into one interim Parquet file with a single schema."""

import io
from pathlib import Path

import polars as pl
from PIL import Image

from bangla_ocr import reid, synth

OUT_PATH = Path("data/interim/ingested.parquet")
SOURCES = {"reid2019", "synth"}

# Union of both source schemas; source-specific columns are null elsewhere.
SCHEMA = {**reid.SCHEMA, **synth.SCHEMA}


def check_row(row: dict) -> list[str]:
    """Return the problems with one ingested row (empty list = row is OK).

    Structural checks only. Empty REID text is allowed here (dropped in Phase 3).
    Messages (exact strings, tests compare them):
      "bad source"           source not in SOURCES
      "bad id"               id empty, or not prefixed "<source>/"
      "missing licence"      licence null or empty
      "image does not decode"
      "size mismatch"        width/height differ from the decoded image
      "synth fields missing" synth row without text, font, or a valid level
    """
    problems = []

    source = row["source"]

    if source not in SOURCES:
        problems.append("bad source")

    if not row["id"] or not row["id"].startswith(f"{source}/"):
        problems.append("bad id")

    if not row["licence"]:
        problems.append("missing licence")

    size = _decoded_size(row["image"])

    if size is None:
        problems.append("image does not decode")
    elif size != (row["width"], row["height"]):
        problems.append("size mismatch")

    if source == "synth":
        if not (row["text"] and row["font"] and row["level"] in synth.LEVELS):
            problems.append("synth fields missing")

    return problems


def combine(
    reid_dir: Path = reid.OUT_DIR, synth_dir: Path = synth.OUT_DIR, out_path: Path = OUT_PATH
) -> None:
    """Stream both sources into `out_path` (never holds all images in RAM)."""
    frames = [pl.scan_parquet(d / "*.parquet") for d in (reid_dir, synth_dir)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (
        pl.concat(frames, how="diagonal_relaxed")
        .select([pl.col(name).cast(dtype) for name, dtype in SCHEMA.items()])
        .sink_parquet(out_path)
    )


def validate(path: Path = OUT_PATH, batch_size: int = 1000) -> dict[str, list[str]]:
    """Run check_row over every row in batches. Returns {id: problems} for bad rows."""
    lf = pl.scan_parquet(path)
    n = lf.select(pl.len()).collect().item()
    bad = {}
    for offset in range(0, n, batch_size):
        for row in lf.slice(offset, batch_size).collect().iter_rows(named=True):
            if problems := check_row(row):
                bad[row["id"]] = problems
    return bad


def _decoded_size(image: bytes | None) -> tuple[int, int] | None:
    """(width, height) of PNG bytes, or None if they don't decode."""
    try:
        with Image.open(io.BytesIO(image)) as img:
            img.load()  # force a full decode: catches truncated files
            return img.size
    except Exception:
        return None


if __name__ == "__main__":
    combine()
    df = pl.scan_parquet(OUT_PATH)
    print(df.group_by("source").len().sort("source").collect())
    bad = validate()
    print(f"{len(bad)} bad rows" + "".join(f"\n  {k}: {v}" for k, v in list(bad.items())[:10]))
