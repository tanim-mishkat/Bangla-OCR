"""Save a few random synthetic images to data/interim/preview/ for eyeballing.

Usage: uv run python scripts/preview_synth.py [n] [seed]
"""

import sys
from pathlib import Path

import polars as pl

OUT = Path("data/interim/preview")
sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1

OUT.mkdir(parents=True, exist_ok=True)
df = pl.read_parquet("data/interim/synth/*.parquet").sample(n, seed=seed)
for k, r in enumerate(df.iter_rows(named=True)):
    (OUT / f"{k}_{r['font']}_{r['level']}.png").write_bytes(r["image"])
print(df.select("text_id", "font", "level", "noise", "blur", "rotation", "width", "height"))
print(f"Saved {n} images to {OUT}")
