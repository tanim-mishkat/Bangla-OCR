"""Render corpus lines into synthetic OCR images with controlled degradation."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIR = Path("data/raw/fonts")
FONT_SIZE = 32
PAD = 8  # white margin around the text, in px


def render_line(
    text: str,
    font_path: Path,
    size: int = FONT_SIZE,
    engine: ImageFont.Layout = ImageFont.Layout.RAQM,
) -> Image.Image:
    """Draw `text` black on white ("L" mode), cropped to the ink plus PAD."""
    font = ImageFont.truetype(str(font_path), size, layout_engine=engine)
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
