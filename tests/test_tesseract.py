import shutil

import polars as pl
import pytest

from bangla_ocr import checks, tesseract
from bangla_ocr.normalise import normalise_text

ENG = tesseract.WINDOWS_EXE.parent / "tessdata" / "eng.traineddata"


def test_exe_prefers_env_var(monkeypatch):
    monkeypatch.setenv("TESSERACT_CMD", "/opt/tess")
    assert tesseract.exe() == "/opt/tess"


def test_exe_missing_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setattr(tesseract.shutil, "which", lambda _: None)
    monkeypatch.setattr(tesseract, "WINDOWS_EXE", tmp_path / "nope.exe")
    with pytest.raises(FileNotFoundError, match="TESSERACT_CMD"):
        tesseract.exe()


def test_command_pins_model_engine_and_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("TESSERACT_CMD", "tess")
    monkeypatch.setattr(tesseract, "tessdata_dir", lambda: tmp_path)
    cmd = tesseract.command(tesseract.PSM_LINE)
    assert cmd[:3] == ["tess", "stdin", "stdout"]
    assert cmd[-1] == str(tesseract.BEN_ONLY_CONFIG) and tesseract.BEN_ONLY_CONFIG.exists()
    flags = dict(zip(cmd[3:-1:2], cmd[4:-1:2], strict=True))
    assert flags == {"--tessdata-dir": str(tmp_path), "-l": "ben", "--oem": "1", "--psm": "7"}


# --- real engine on one real image (skipped if Tesseract or the data is missing) ---


def _installed() -> bool:
    try:
        return bool(tesseract.exe())
    except FileNotFoundError:
        return False


needs_engine_and_data = pytest.mark.skipif(
    not (_installed() and checks.split_paths()["test"].exists()),
    reason="Tesseract or processed data missing",
)


def first_clean_synth() -> dict:
    return (
        pl.scan_parquet(checks.split_paths()["test"])
        .filter((pl.col("source") == "synth") & (pl.col("level") == "clean"))
        .sort("id")
        .select("image", "text")
        .head(1)
        .collect()
        .row(0, named=True)
    )


@needs_engine_and_data
def test_reads_one_clean_synthetic_line():
    row = first_clean_synth()
    pred = normalise_text(tesseract.read(row["image"]))
    shared = set(pred) & set(row["text"])
    assert pred and len(shared) >= len(set(row["text"])) // 2, (pred, row["text"])


@needs_engine_and_data
@pytest.mark.skipif(not ENG.exists(), reason="no eng.traineddata in the Tesseract install")
def test_eng_next_to_ben_does_not_change_output(tmp_path, monkeypatch):
    png = first_clean_synth()["image"]
    alone = tesseract.read(png)
    for model in (tesseract.tessdata_dir() / "ben.traineddata", ENG):
        shutil.copy(model, tmp_path)
    monkeypatch.setattr(tesseract, "tessdata_dir", lambda: tmp_path)
    assert tesseract.read(png) == alone
