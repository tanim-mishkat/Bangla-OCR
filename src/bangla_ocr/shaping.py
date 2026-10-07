"""Make sure Pillow can shape Bangla text (needs libraqm + FriBiDi)."""

import os
import sys
from pathlib import Path

FRIBIDI_DIR = Path(__file__).resolve().parents[2] / "tools" / "fribidi"


def ensure_raqm() -> None:
    """Raise RuntimeError unless Pillow's raqm layout engine is available.

    On Windows, adds tools/fribidi to PATH first. Call before any PIL.ImageFont use.
    """
    if sys.platform == "win32" and FRIBIDI_DIR.is_dir():
        os.environ["PATH"] = f"{FRIBIDI_DIR}{os.pathsep}{os.environ.get('PATH', '')}"

    from PIL import features

    if not features.check("raqm"):
        raise RuntimeError(
            "Pillow raqm is unavailable, so Bangla would render with wrong glyph order. "
            "Windows: run scripts/get_fribidi.ps1. Linux: install libfribidi0 (apt)."
        )
