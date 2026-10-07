"""Make sure Pillow can shape Bangla text (needs libraqm + FriBiDi)."""

import os
import sys
from pathlib import Path

FRIBIDI_DIR = Path(__file__).resolve().parents[2] / "tools" / "fribidi"


def add_fribidi_to_path() -> None:
    """On Windows, expose tools/fribidi to the DLL search path.

    Pillow looks for FriBiDi once, when PIL.ImageDraw/ImageFont is first imported,
    so this must run before that. bangla_ocr/__init__.py calls it on import.
    """
    if sys.platform == "win32" and FRIBIDI_DIR.is_dir():
        os.environ["PATH"] = f"{FRIBIDI_DIR}{os.pathsep}{os.environ.get('PATH', '')}"


def ensure_raqm() -> None:
    """Raise RuntimeError unless Pillow's raqm layout engine is available."""
    from PIL import features

    if not features.check("raqm"):
        raise RuntimeError(
            "Pillow raqm is unavailable, so Bangla would render with wrong glyph order. "
            "Windows: run scripts/get_fribidi.ps1 (and import bangla_ocr before PIL). "
            "Linux: install libfribidi0 (apt)."
        )
