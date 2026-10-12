import polars as pl
import pytest

from bangla_ocr import checks, tesseract
from bangla_ocr.normalise import normalise_text


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
    flags = dict(zip(cmd[3::2], cmd[4::2], strict=True))
    assert flags == {"--tessdata-dir": str(tmp_path), "-l": "ben", "--oem": "1", "--psm": "7"}


# --- real engine on one real image (skipped if Tesseract or the data is missing) ---


def _installed() -> bool:
    try:
        return bool(tesseract.exe())
    except FileNotFoundError:
        return False


@pytest.mark.skipif(
    not (_installed() and checks.split_paths()["test"].exists()),
    reason="Tesseract or processed data missing",
)
def test_reads_one_clean_synthetic_line():
    row = (
        pl.scan_parquet(checks.split_paths()["test"])
        .filter((pl.col("source") == "synth") & (pl.col("level") == "clean"))
        .sort("id")
        .select("image", "text")
        .head(1)
        .collect()
        .row(0, named=True)
    )
    pred = normalise_text(tesseract.read(row["image"]))
    shared = set(pred) & set(row["text"])
    assert pred and len(shared) >= len(set(row["text"])) // 2, (pred, row["text"])
