import numpy as np
import pytest
from PIL import Image, ImageFont

from bangla_ocr.shaping import ensure_raqm
from bangla_ocr.synth import FONT_DIR, PAD, augment, render_line

NOTO = FONT_DIR / "NotoSansBengali.ttf"
needs_fonts = pytest.mark.skipif(not NOTO.exists(), reason="run bangla_ocr.download first")


# --- render_line ---


@needs_fonts
def test_render_line_is_black_text_on_white_with_margin():
    img = render_line("আমার সোনার বাংলা", NOTO)
    px = np.asarray(img)
    assert img.mode == "L"
    assert px.min() == 0  # there is ink
    assert (px[: PAD - 1] == 255).all() and (px[:, : PAD - 1] == 255).all()  # top/left margin
    assert (px[-(PAD - 1) :] == 255).all() and (px[:, -(PAD - 1) :] == 255).all()  # bottom/right


@needs_fonts
@pytest.mark.parametrize("text", ["কি", "ক্ষ", "স্ত্র"])  # pre-base vowel, conjuncts
def test_raqm_shapes_differently_from_basic(text):
    ensure_raqm()
    shaped = render_line(text, NOTO, engine=ImageFont.Layout.RAQM)
    naive = render_line(text, NOTO, engine=ImageFont.Layout.BASIC)
    assert shaped.size != naive.size or shaped.tobytes() != naive.tobytes()


# --- augment (you write it) ---


@pytest.fixture
def block():
    """60x30 white image with a black 20x10 box in the middle."""
    img = Image.new("L", (60, 30), 255)
    img.paste(0, (20, 10, 40, 20))
    return img


def test_augment_with_defaults_is_identity(block):
    out = augment(block, np.random.default_rng(0))
    assert out is not block  # a copy, so callers can edit it safely
    assert out.size == block.size
    assert out.tobytes() == block.tobytes()


def test_augment_does_not_mutate_input(block):
    before = block.tobytes()
    augment(block, np.random.default_rng(0), noise=30, blur=2, rotation=10)
    assert block.tobytes() == before


def test_rotation_expands_canvas_with_white_corners():
    black = Image.new("L", (40, 20), 0)
    out = augment(black, np.random.default_rng(0), rotation=10)
    assert out.mode == "L"
    assert out.width > 40 and out.height > 20
    assert out.getpixel((0, 0)) == 255


def test_rotation_antialiases_edges(block):
    out = augment(block, np.random.default_rng(0), rotation=10)
    assert len(np.unique(np.asarray(out))) > 2  # grey edges, not a jagged staircase


def test_blur_creates_grey_edges(block):
    out = augment(block, np.random.default_rng(0), blur=2)
    assert out.size == block.size
    assert len(np.unique(np.asarray(out))) > 2  # not just pure black and white any more


def test_noise_is_clipped_and_changes_pixels():
    white = Image.new("L", (50, 50), 255)
    out = augment(white, np.random.default_rng(0), noise=40)
    px = np.asarray(out)
    assert out.mode == "L" and px.dtype == np.uint8
    assert px.max() == 255  # clipped, did not wrap around to small values
    assert px.min() < 200  # noise really went in


def test_noise_is_reproducible_from_seed(block):
    a = augment(block, np.random.default_rng(7), noise=25)
    b = augment(block, np.random.default_rng(7), noise=25)
    c = augment(block, np.random.default_rng(8), noise=25)
    assert a.tobytes() == b.tobytes()
    assert a.tobytes() != c.tobytes()
