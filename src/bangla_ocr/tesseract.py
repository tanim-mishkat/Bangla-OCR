"""Tesseract via its CLI: pinned Bengali LSTM model, PNG bytes in, text out."""

import os
import shutil
import subprocess
from functools import cache
from pathlib import Path

from bangla_ocr.download import fetch
from bangla_ocr.sources import TESS_BEN

WINDOWS_EXE = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
PSM_LINE, PSM_BLOCK = 7, 6  # synthetic = one text line; REID regions = multi-line blocks
OEM_LSTM = 1  # neural engine only (tessdata_best has no legacy model)


def exe() -> str:
    """TESSERACT_CMD env var, else `tesseract` on PATH, else the default Windows install."""
    found = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if not found and WINDOWS_EXE.exists():
        found = str(WINDOWS_EXE)
    if not found:
        raise FileNotFoundError("Tesseract not found: install it or set TESSERACT_CMD")
    return found


@cache
def tessdata_dir() -> Path:
    """Folder holding the pinned ben.traineddata (downloaded + hash-checked once per process)."""
    return fetch(TESS_BEN).parent


def version() -> str:
    out = subprocess.run([exe(), "--version"], capture_output=True, text=True, check=True)
    return out.stdout.splitlines()[0]


def command(psm: int) -> list[str]:
    return [
        exe(),
        "stdin",
        "stdout",
        "--tessdata-dir",
        str(tessdata_dir()),
        "-l",
        "ben",
        "--oem",
        str(OEM_LSTM),
        "--psm",
        str(psm),
    ]


def read(png: bytes, psm: int = PSM_LINE) -> str:
    """OCR one image. Output is raw: normalise before scoring, like the labels."""
    out = subprocess.run(command(psm), input=png, capture_output=True, check=True)
    return out.stdout.decode("utf-8").strip()
