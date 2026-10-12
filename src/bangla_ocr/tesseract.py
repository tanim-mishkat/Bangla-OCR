"""Tesseract via its CLI: pinned Bengali LSTM model, PNG bytes in, text out."""

import os
import shutil
import subprocess
from functools import cache
from pathlib import Path

from bangla_ocr.download import fetch
from bangla_ocr.predict import Engine
from bangla_ocr.sources import TESS_BEN

WINDOWS_EXE = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
PSM_LINE, PSM_BLOCK = 7, 6  # synthetic = one text line; REID regions = multi-line blocks
OEM_LSTM = 1  # neural engine only (tessdata_best has no legacy model)
# Without it, output depends on whether eng.traineddata happens to sit next to ben
BEN_ONLY_CONFIG = Path(__file__).with_name("tesseract_ben_only.cfg")


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
        str(BEN_ONLY_CONFIG),
    ]


def read(png: bytes, psm: int = PSM_LINE) -> str:
    """OCR one image. Output is raw: normalise before scoring, like the labels."""
    out = subprocess.run(command(psm), input=png, capture_output=True, check=True)
    return out.stdout.decode("utf-8").strip()


def engine() -> Engine:
    """Synthetic rows are single lines, REID rows are multi-line regions."""
    return Engine(
        name="tesseract",
        config={
            "version": version(),
            "model_sha256": TESS_BEN.sha256,
            "oem": OEM_LSTM,
            "psm": {"synth": PSM_LINE, "reid2019": PSM_BLOCK},
            "extra_config": BEN_ONLY_CONFIG.read_text(encoding="utf-8"),
        },
        read=lambda row: read(row["image"], PSM_LINE if row["source"] == "synth" else PSM_BLOCK),
    )
