"""EasyOCR on CPU: pinned CRAFT detector + bengali_g1 recognizer, PNG bytes in, text out."""

import os
import shutil
from functools import cache
from importlib.metadata import version
from pathlib import Path

from bangla_ocr.download import RAW_DIR, fetch
from bangla_ocr.predict import Engine
from bangla_ocr.sources import EASYOCR_BEN, EASYOCR_CRAFT

# (source, file inside its zip) -> both must sit in one folder for easyocr.Reader
WEIGHTS = [(EASYOCR_CRAFT, "craft_mlt_25k.pth"), (EASYOCR_BEN, "bengali.pth")]

# Everything here changes the output, so it is part of the cache key. Defaults otherwise:
# REID is test-only, so no setting may be tuned on it.
SETTINGS = {
    "langs": ["bn"],  # Bangla chars + EasyOCR's shared symbols (Latin digits 0-9 can still leak)
    "decoder": "greedy",
    "quantize": True,  # EasyOCR's CPU default (1.7.2 ignores False for the detector anyway)
    "mode": {"synth": "recognize: whole image = one line", "reid2019": "readtext + reading_order"},
}


def model_dir(raw_dir: Path = RAW_DIR) -> Path:
    """One folder with both pinned weight files, hard-linked (no extra disk) from the downloads."""
    out = raw_dir / "easyocr_models"
    out.mkdir(parents=True, exist_ok=True)

    for source, name in WEIGHTS:
        dst = out / name
        if not dst.exists():
            src = fetch(source, raw_dir) / name
            try:
                os.link(src, dst)
            except OSError:  # e.g. another drive: fall back to a copy
                shutil.copy2(src, dst)

    return out


@cache
def reader():
    """Loaded once per process (~2 s, ~1 GB RAM). Downloads disabled: only our pinned files."""
    import easyocr  # heavy (torch): only when the engine actually runs

    return easyocr.Reader(
        SETTINGS["langs"],
        gpu=False,
        model_storage_directory=str(model_dir()),
        download_enabled=False,
        quantize=SETTINGS["quantize"],
        verbose=False,
    )


def reading_order(words: list[tuple[list[list[float]], str]]) -> str:
    """Join detected words into text: lines top-to-bottom, words left-to-right.

    words: (box, text) pairs; box = 4 corner points [[x, y], ...] (any order).
    Rules:
    - a word's centre y = mean of its 4 y values; its height = max y - min y
    - sort words by centre y; walk through them: a word starts a NEW line if its centre y is
      more than half the current line's first-word height below that first word's centre y
    - inside a line, sort by left edge (min x); join words with " ", lines with "\\n"
    - no words -> ""
    """
    positioned = []

    for box, text in words:
        ys = [y for _, y in box]
        cy = sum(ys) / 4
        height = max(ys) - min(ys)
        left = min(x for x, _ in box)
        positioned.append((cy, height, left, text))

    positioned.sort(key=lambda item: item[0])

    # Each line stores (first_cy, first_height, words)
    lines = []

    for cy, height, left, text in positioned:
        if not lines or cy - lines[-1][0] > lines[-1][1] / 2:
            lines.append((cy, height, [(left, text)]))
        else:
            lines[-1][2].append((left, text))

    return "\n".join(" ".join(text for _, text in sorted(items)) for _, _, items in lines)


def read(png: bytes, source: str, ocr=None) -> str:
    """OCR one image. Output is raw: normalise before scoring, like the labels."""
    ocr = ocr or reader()
    decoder = SETTINGS["decoder"]

    if source == "synth":  # like Tesseract --psm 7: the engine is told this is one line
        return " ".join(ocr.recognize(png, decoder=decoder, detail=0)).strip()

    if source == "reid2019":  # like --psm 6: the engine finds the lines itself
        found = ocr.readtext(png, decoder=decoder, detail=1)
        return reading_order([(box, text) for box, text, _conf in found])

    raise ValueError(f"unknown source {source!r}")


def engine() -> Engine:
    return Engine(
        name="easyocr",
        config={
            "version": version("easyocr"),
            "torch": version("torch"),
            "opencv": version("opencv-python-headless"),
            "craft_sha256": EASYOCR_CRAFT.sha256,
            "model_sha256": EASYOCR_BEN.sha256,
            **SETTINGS,
        },
        read=lambda row: read(row["image"], row["source"]),
    )
