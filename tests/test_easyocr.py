import importlib.util
import io

import polars as pl
import pytest
from PIL import Image

from bangla_ocr import checks, easyocr_engine, predict
from bangla_ocr.download import RAW_DIR
from bangla_ocr.easyocr_engine import reading_order
from bangla_ocr.metrics import edit_distance
from bangla_ocr.normalise import normalise_text
from bangla_ocr.sources import EASYOCR_BEN, EASYOCR_CRAFT


def box(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    """EasyOCR corner order: top-left, top-right, bottom-right, bottom-left."""
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


# --- reading_order (your function) ---


def test_no_words_is_empty_text():
    assert reading_order([]) == ""


def test_one_word():
    assert reading_order([(box(0, 0, 40, 20), "আমি")]) == "আমি"


def test_one_line_sorted_left_to_right():
    words = [(box(100, 0, 140, 20), "খাই"), (box(0, 0, 40, 20), "আমি"), (box(50, 0, 90, 20), "ভাত")]
    assert reading_order(words) == "আমি ভাত খাই"


def test_two_lines_top_to_bottom_whatever_the_input_order():
    words = [
        (box(50, 30, 90, 50), "না"),
        (box(50, 0, 90, 20), "ভাত"),
        (box(0, 30, 40, 50), "খাই"),
        (box(0, 0, 40, 20), "আমি"),
    ]
    assert reading_order(words) == "আমি ভাত\nখাই না"


def test_slightly_skewed_line_stays_one_line():
    # centres 10, 14, 18; first word height 20 -> all within 10 of the first centre
    words = [(box(0, 0, 40, 20), "এক"), (box(50, 4, 90, 24), "দুই"), (box(100, 8, 140, 28), "তিন")]
    assert reading_order(words) == "এক দুই তিন"


def test_exactly_half_height_below_is_still_the_same_line():
    # "more than half" starts a new line; exactly half does not
    words = [(box(0, 0, 40, 20), "ক"), (box(50, 10, 90, 30), "খ")]
    assert reading_order(words) == "ক খ"


def test_just_over_half_height_starts_a_new_line():
    words = [(box(0, 0, 40, 20), "ক"), (box(50, 11, 90, 31), "খ")]
    assert reading_order(words) == "ক\nখ"


def test_new_line_is_measured_from_the_lines_first_word():
    # centres 10, 18, 26: each step is small, but 26 is 16 below the first word (> 10)
    words = [(box(0, 0, 40, 20), "ক"), (box(50, 8, 90, 28), "খ"), (box(0, 16, 40, 36), "গ")]
    assert reading_order(words) == "ক খ\nগ"


def test_corner_order_does_not_matter():
    # same box as box(0, 0, 40, 20), corners listed from the right: the first x is 40,
    # but the left edge is min x = 0, left of the next word (boxes may overlap a little)
    flipped = [[40.0, 20.0], [0.0, 20.0], [0.0, 0.0], [40.0, 0.0]]
    words = [(box(30, 0, 70, 20), "ভাত"), (flipped, "আমি")]
    assert reading_order(words) == "আমি ভাত"


def test_lines_are_grouped_by_centres_not_tops():
    # a tall word and a short word on one line: tops differ by 25, centres by only 10
    words = [(box(0, 0, 40, 40), "কর্ম"), (box(50, 25, 90, 35), "ও")]
    assert reading_order(words) == "কর্ম ও"


# --- glue: which EasyOCR call each source gets (fake reader, no torch) ---


class FakeReader:
    def __init__(self, lines=(), found=()):
        self.lines, self.found, self.calls = list(lines), list(found), []

    def recognize(self, image, **kwargs):
        self.calls.append(("recognize", kwargs))
        return self.lines

    def readtext(self, image, **kwargs):
        self.calls.append(("readtext", kwargs))
        return self.found


def test_synth_is_read_as_one_line_without_the_detector():
    fake = FakeReader(lines=["আমি ভাত"])
    assert easyocr_engine.read(b"png", "synth", ocr=fake) == "আমি ভাত"
    # no horizontal_list -> EasyOCR treats the whole image as one line box
    assert fake.calls == [("recognize", {"decoder": "greedy", "detail": 0})]


def test_reid_uses_detector_then_reading_order(monkeypatch):
    monkeypatch.setattr(easyocr_engine, "reading_order", lambda w: "|".join(t for _, t in w))
    fake = FakeReader(found=[(box(0, 0, 9, 9), "ক", 0.9), (box(20, 0, 29, 9), "খ", 0.4)])
    assert easyocr_engine.read(b"png", "reid2019", ocr=fake) == "ক|খ"
    assert fake.calls == [("readtext", {"decoder": "greedy", "detail": 1})]


def test_unknown_source_refuses():
    with pytest.raises(ValueError, match="unknown source"):
        easyocr_engine.read(b"png", "other", ocr=FakeReader())


def test_model_dir_links_both_pinned_files_into_one_folder(tmp_path, monkeypatch):
    fetched = []

    def fake_fetch(source, raw_dir):
        fetched.append(source.name)
        out = raw_dir / source.name
        out.mkdir(exist_ok=True)
        for _, name in easyocr_engine.WEIGHTS:
            (out / name).write_bytes(source.name.encode())
        return out

    monkeypatch.setattr(easyocr_engine, "fetch", fake_fetch)
    out = easyocr_engine.model_dir(tmp_path)
    assert (out / "craft_mlt_25k.pth").read_bytes() == b"easyocr_craft"
    assert (out / "bengali.pth").read_bytes() == b"easyocr_ben"
    easyocr_engine.model_dir(tmp_path)
    assert fetched == ["easyocr_craft", "easyocr_ben"]  # second call: already linked


# --- config = cache key ---

installed = importlib.util.find_spec("easyocr") is not None
needs_package = pytest.mark.skipif(not installed, reason="easyocr group not installed")


@needs_package
def test_config_pins_weights_versions_and_settings():
    c = easyocr_engine.engine().config
    assert c["craft_sha256"] == EASYOCR_CRAFT.sha256 and c["model_sha256"] == EASYOCR_BEN.sha256
    assert None not in (EASYOCR_CRAFT.sha256, EASYOCR_BEN.sha256)
    assert c["langs"] == ["bn"] and c["decoder"] == "greedy" and c["quantize"] is True
    assert all(c[k] for k in ["version", "torch", "opencv"])


@needs_package
def test_changing_a_setting_changes_the_cache_key(monkeypatch):
    before = predict.config_id(easyocr_engine.engine().config)
    monkeypatch.setitem(easyocr_engine.SETTINGS, "decoder", "beamsearch")
    assert predict.config_id(easyocr_engine.engine().config) != before


# --- real model (val images only: settings are never checked against test) ---

weights = [RAW_DIR / s.name / n for s, n in easyocr_engine.WEIGHTS]
needs_model = pytest.mark.skipif(
    not (installed and all(p.exists() for p in weights) and checks.split_paths()["val"].exists()),
    reason="easyocr, its pinned weights, or the val split missing",
)


def clean_val_lines(n: int) -> list[dict]:
    """First n clean val lines with distinct texts (each text exists in several fonts)."""
    return (
        pl.scan_parquet(checks.split_paths()["val"])
        .filter((pl.col("source") == "synth") & (pl.col("level") == "clean"))
        .sort("id")
        .unique("text", keep="first", maintain_order=True)
        .select("image", "text")
        .head(n)
        .collect()
        .to_dicts()
    )


def cer(ref: str, hyp: str) -> float:
    return edit_distance(ref, normalise_text(hyp)) / len(ref)


@needs_package
def test_missing_weights_raise_instead_of_downloading(tmp_path):
    import easyocr

    with pytest.raises(FileNotFoundError, match="downloads disabled"):
        easyocr.Reader(["bn"], gpu=False, model_storage_directory=str(tmp_path),
                       download_enabled=False, verbose=False)  # fmt: skip


@needs_model
def test_reads_a_clean_val_line_deterministically():
    row = clean_val_lines(1)[0]
    first = easyocr_engine.read(row["image"], "synth")
    assert easyocr_engine.read(row["image"], "synth") == first  # CPU + greedy: no randomness
    assert cer(row["text"], first) < 0.5, (first, row["text"])


@needs_model
def test_block_mode_reads_two_stacked_lines_in_order():
    a, b = clean_val_lines(2)
    top, bottom = (Image.open(io.BytesIO(r["image"])).convert("RGB") for r in (a, b))
    page = Image.new("RGB", (max(top.width, bottom.width), top.height + bottom.height), "white")
    page.paste(top, (0, 0))
    page.paste(bottom, (0, top.height))
    buf = io.BytesIO()
    page.save(buf, format="PNG")
    out = easyocr_engine.read(buf.getvalue(), "reid2019")
    in_order = cer(f"{a['text']} {b['text']}", out)
    assert in_order < cer(f"{b['text']} {a['text']}", out), out
