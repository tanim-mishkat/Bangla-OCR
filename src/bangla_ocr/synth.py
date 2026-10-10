"""Render corpus lines into synthetic OCR images with controlled degradation."""

import io
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from bangla_ocr.shaping import ensure_raqm
from bangla_ocr.sources import FONTS, WIKI_BN

FONT_DIR = Path("data/raw/fonts")
FONT_PATHS = {f.name: FONT_DIR / Path(f.filename).name for f in FONTS}
CORPUS_PATH = Path("data/interim/corpus.parquet")
OUT_DIR = Path("data/interim/synth")
FONT_SIZE = 32
PAD = 8  # white margin around the text, in px
CHUNK_LINES = 200  # corpus lines per output file (x fonts = rows)

# (low, high) uniform ranges per degradation level
LEVELS = {
    "clean": {"noise": (0, 0), "blur": (0, 0), "rotation": (0, 0)},
    "mild": {"noise": (5, 15), "blur": (0.3, 0.8), "rotation": (-2, 2)},
    "heavy": {"noise": (20, 40), "blur": (0.8, 1.5), "rotation": (-5, 5)},
}

SCHEMA = {
    "id": pl.String,
    "image": pl.Binary,
    "text": pl.String,
    "source": pl.String,
    "licence": pl.String,
    "text_id": pl.String,
    "font": pl.String,
    "level": pl.String,
    "noise": pl.Float32,
    "blur": pl.Float32,
    "rotation": pl.Float32,
    "width": pl.Int32,
    "height": pl.Int32,
}


@cache
def _load_font(path: Path, size: int, engine: ImageFont.Layout) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size, layout_engine=engine)


@cache
def font_codepoints(path: Path) -> frozenset[int]:
    return frozenset(TTFont(path).getBestCmap())


def has_glyphs(text: str, font_path: Path) -> bool:
    """True if the font covers every non-space char (else it would draw tofu boxes)."""
    cmap = font_codepoints(font_path)
    return all(ord(c) in cmap for c in text if not c.isspace())


def render_line(
    text: str,
    font_path: Path,
    size: int = FONT_SIZE,
    engine: ImageFont.Layout = ImageFont.Layout.RAQM,
) -> Image.Image:
    """Draw `text` black on white ("L" mode), cropped to the ink plus PAD."""
    font = _load_font(font_path, size, engine)
    left, top, right, bottom = font.getbbox(text)

    img = Image.new(
        "L",
        (right - left + 2 * PAD, bottom - top + 2 * PAD),
        255,
    )

    ImageDraw.Draw(img).text(
        (PAD - left, PAD - top),
        text,
        font=font,
        fill=0,
    )

    return img


def augment(
    img: Image.Image,
    rng: np.random.Generator,
    noise: float = 0.0,
    blur: float = 0.0,
    rotation: float = 0.0,
) -> Image.Image:
    """Return a degraded copy of `img` ("L" mode); `img` itself is unchanged.

    rotation: degrees (counter-clockwise), canvas expands, new corners are white.
    blur:     Gaussian blur radius in px.
    noise:    std-dev of Gaussian pixel noise (0-255 scale), result clipped to 0-255.
    """
    out = img.copy()

    # 1. Rotate first.
    if rotation:
        out = out.rotate(
            rotation,
            expand=True,
            fillcolor=255,
            resample=Image.Resampling.BICUBIC,
        )

    # 2. Blur second.
    if blur > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=blur))

    # 3. Add Gaussian pixel noise last.
    if noise > 0:
        arr = np.asarray(out, dtype=np.float32)

        random_noise = rng.normal(
            0,
            noise,
            size=arr.shape,
        )

        arr = arr + random_noise
        arr = np.clip(arr, 0, 255)
        arr = np.rint(arr)
        arr = arr.astype(np.uint8)

        out = Image.fromarray(arr)

    return out


def sample_params(rng: np.random.Generator) -> dict:
    """Pick a degradation level, then draw its parameters uniformly."""
    level = str(rng.choice(list(LEVELS)))
    params = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in LEVELS[level].items()}
    return {"level": level, **params}


def line_rows(
    i: int, text_id: str, text: str, fonts: dict[str, Path], seed: int
) -> tuple[list[dict], int]:
    """Render corpus line `i` in every font. Returns (rows, number of fonts skipped)."""
    rows, skipped = [], 0
    for j, (name, path) in enumerate(fonts.items()):
        if not has_glyphs(text, path):
            skipped += 1
            continue
        rng = np.random.default_rng([seed, i, j])  # per-sample seed: order-independent
        params = sample_params(rng)
        img = augment(
            render_line(text, path),
            rng,
            noise=params["noise"],
            blur=params["blur"],
            rotation=params["rotation"],
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        rows.append(
            {
                "id": f"synth/{text_id}/{name}",
                "image": buf.getvalue(),
                "text": text,
                "source": "synth",
                "licence": WIKI_BN.licence,  # ShareAlike: rendered text inherits it
                "text_id": text_id,
                "font": name,
                **params,
                "width": img.width,
                "height": img.height,
            }
        )
    return rows, skipped


def build(
    corpus_path: Path = CORPUS_PATH,
    out_dir: Path = OUT_DIR,
    fonts: dict[str, Path] = FONT_PATHS,
    seed: int = 42,
    chunk_lines: int = CHUNK_LINES,
) -> int:
    """Write corpus x fonts images in chunked Parquet files. Resumable. Returns rows skipped."""
    ensure_raqm()
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = pl.read_parquet(corpus_path, columns=["text_id", "text"])
    n_chunks = -(-corpus.height // chunk_lines)
    total = skipped = 0
    for c in range(n_chunks):
        out = out_dir / f"part-{c:05d}.parquet"
        start = c * chunk_lines
        if not out.exists():
            rows = []
            chunk = corpus.slice(start, chunk_lines)
            for i, (text_id, text) in enumerate(chunk.iter_rows(), start):
                r, s = line_rows(i, text_id, text, fonts, seed)
                rows.extend(r)
                skipped += s
            pl.DataFrame(rows, schema=SCHEMA).write_parquet(out)
        n = pl.scan_parquet(out).select(pl.len()).collect().item()
        total += n
        print(f"[{c + 1}/{n_chunks}] {out.name}: {n} images")
    print(f"Done: {total} images in {out_dir} ({skipped} line-font pairs skipped: missing glyphs)")
    return skipped


if __name__ == "__main__":
    build()
