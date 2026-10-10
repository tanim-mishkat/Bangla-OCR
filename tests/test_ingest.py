import io

import polars as pl
import pytest
from PIL import Image

from bangla_ocr import reid, synth
from bangla_ocr.ingest import SCHEMA, check_row, combine, validate


def png(w: int = 30, h: int = 10) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (w, h), 255).save(buf, format="PNG")
    return buf.getvalue()


def reid_row(**overrides) -> dict:
    row = {
        "id": "reid2019/p1/r0",
        "image": png(),
        "text": "আমার সোনার বাংলা",
        "source": "reid2019",
        "licence": "Public Domain Mark 1.0",
        "page": "p1",
        "region_type": "paragraph",
        "width": 30,
        "height": 10,
    }
    return {**dict.fromkeys(SCHEMA), **row, **overrides}


def synth_row(**overrides) -> dict:
    row = {
        "id": "synth/wiki_bn/00001/Galada",
        "image": png(),
        "text": "আমি ভাত খাই।",
        "source": "synth",
        "licence": "CC BY-SA 3.0",
        "text_id": "wiki_bn/00001",
        "font": "Galada",
        "level": "mild",
        "noise": 10.0,
        "blur": 0.5,
        "rotation": 1.0,
        "width": 30,
        "height": 10,
    }
    return {**dict.fromkeys(SCHEMA), **row, **overrides}


# --- check_row (you write it) ---


def test_good_rows_pass():
    assert check_row(reid_row()) == []
    assert check_row(synth_row()) == []


def test_empty_reid_text_is_allowed_at_ingest():
    assert check_row(reid_row(text="")) == []


@pytest.mark.parametrize(
    "row, problem",
    [
        (reid_row(source="kaggle", id="kaggle/p1/r0"), "bad source"),
        (reid_row(id=""), "bad id"),
        (reid_row(id="synth/p1/r0"), "bad id"),  # prefix must match its own source
        (reid_row(licence=""), "missing licence"),
        (synth_row(licence=None), "missing licence"),
        (reid_row(image=b"not a png"), "image does not decode"),
        (reid_row(image=png()[:40]), "image does not decode"),  # truncated file
        (reid_row(image=None), "image does not decode"),
        (reid_row(width=31), "size mismatch"),
        (synth_row(height=0), "size mismatch"),
        (synth_row(text=""), "synth fields missing"),
        (synth_row(font=None), "synth fields missing"),
        (synth_row(level="extreme"), "synth fields missing"),
    ],
)
def test_each_problem_is_reported(row, problem):
    assert check_row(row) == [problem]


def test_multiple_problems_are_all_reported():
    problems = check_row(synth_row(licence="", width=99, font=None))
    assert sorted(problems) == ["missing licence", "size mismatch", "synth fields missing"]


def test_null_source_also_breaks_the_id_prefix():
    assert sorted(check_row(reid_row(source=None))) == ["bad id", "bad source"]


# --- combine + validate ---


@pytest.fixture
def source_dirs(tmp_path):
    reid_dir, synth_dir = tmp_path / "reid", tmp_path / "synth"
    reid_dir.mkdir()
    synth_dir.mkdir()
    pick = lambda row, schema: {k: row[k] for k in schema}  # noqa: E731
    pl.DataFrame([pick(reid_row(), reid.SCHEMA)], schema=reid.SCHEMA).write_parquet(
        reid_dir / "p1.parquet"
    )
    pl.DataFrame([pick(synth_row(), synth.SCHEMA)], schema=synth.SCHEMA).write_parquet(
        synth_dir / "part-00000.parquet"
    )
    return reid_dir, synth_dir


def test_combine_has_unified_schema_with_nulls(source_dirs, tmp_path):
    out = tmp_path / "ingested.parquet"
    combine(*source_dirs, out)
    df = pl.read_parquet(out)
    assert df.schema == pl.Schema(SCHEMA)
    assert df["source"].to_list() == ["reid2019", "synth"]
    real, fake = df.row(0, named=True), df.row(1, named=True)
    assert real["font"] is None and real["text_id"] is None
    assert fake["page"] is None and fake["region_type"] is None


def test_validate_finds_bad_rows(source_dirs, tmp_path):
    out = tmp_path / "ingested.parquet"
    combine(*source_dirs, out)
    assert validate(out) == {}
    df = pl.read_parquet(out).with_columns(width=pl.lit(5, pl.Int32))
    df.write_parquet(out)
    assert validate(out, batch_size=1) == {
        "reid2019/p1/r0": ["size mismatch"],
        "synth/wiki_bn/00001/Galada": ["size mismatch"],
    }
