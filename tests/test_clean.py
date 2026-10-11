import zlib

import polars as pl

from bangla_ocr.clean import clean, plan
from bangla_ocr.ingest import SCHEMA

ZWJ, NBSP = "\u200d", "\u00a0"
FONTS = ["Galada", "Hind"]
META = {**{k: SCHEMA[k] for k in ("id", "source", "text", "text_id")}, "image_hash": pl.UInt64}


def reid(id_: str, text: str | None, image: str | None = None) -> dict:
    id_ = f"reid2019/{id_}"
    image_hash = zlib.crc32((image or id_).encode())  # distinct image per row unless given
    return {
        "id": id_,
        "source": "reid2019",
        "text": text,
        "text_id": None,
        "image_hash": image_hash,
    }


def synth(text_id: str, text: str) -> list[dict]:
    return [
        {
            "id": f"synth/{text_id}/{f}",
            "source": "synth",
            "text": text,
            "text_id": text_id,
            "image_hash": zlib.crc32(f"{text_id}/{f}".encode()),
        }
        for f in FONTS
    ]


def meta(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=META)


def test_empty_texts_dropped_after_normalising():
    rows = [reid("a", "ক"), reid("b", None), reid("c", ""), reid("d", f" {ZWJ}{NBSP} ")]
    kept, report = plan(meta(rows))
    assert kept["id"].to_list() == ["reid2019/a"]
    assert report["empty text"] == 3


def test_text_is_normalised_and_raw_kept():
    kept, _ = plan(meta([reid("a", f"আমি{NBSP}ভাত{ZWJ}")]))
    row = kept.row(0, named=True)
    assert row["text"] == "আমি ভাত"
    assert row["text_raw"] == f"আমি{NBSP}ভাত{ZWJ}"


def test_reid_same_text_different_images_kept():
    # e.g. page number "৬" on several pages: distinct real test images
    kept, report = plan(meta([reid("a", "৬"), reid("b", "৬")]))
    assert kept.height == 2
    assert report["duplicate reid image"] == 0


def test_reid_identical_images_keep_lowest_id():
    rows = [reid("b", "কখ", image="x"), reid("a", "কখ", image="x"), reid("c", "গ", image="y")]
    kept, report = plan(meta(rows))
    assert kept["id"].to_list() == ["reid2019/a", "reid2019/c"]
    assert report["duplicate reid image"] == 1


def test_synth_duplicates_drop_whole_text_id():
    rows = (
        synth("wiki_bn/00002", "কখ")
        + synth("wiki_bn/00001", f"কখ{NBSP}")  # equal after normalising
        + synth("wiki_bn/00003", "গ")
    )
    kept, report = plan(meta(rows))
    assert {i.rsplit("/", 1)[0] for i in kept["id"]} == {
        "synth/wiki_bn/00001",
        "synth/wiki_bn/00003",
    }
    assert kept.height == 4  # all fonts of each kept line survive
    assert report["duplicate synth text"] == 2


def test_synth_text_matching_reid_is_dropped():
    # REID is test-only: the same text in synth train would leak it
    rows = [reid("a", "কখ")] + synth("wiki_bn/00001", "কখ") + synth("wiki_bn/00002", "গ")
    kept, report = plan(meta(rows))
    assert kept["id"].to_list() == [
        "reid2019/a",
        "synth/wiki_bn/00002/Galada",
        "synth/wiki_bn/00002/Hind",
    ]
    assert report["synth text also in reid"] == 2


def test_report_counts_add_up():
    rows = (
        [reid("a", "ক", image="x"), reid("b", ""), reid("c", "ক", image="x")]
        + synth("wiki_bn/00001", "ক")
        + synth("wiki_bn/00002", "খ")
        + synth("wiki_bn/00003", "খ")
    )
    kept, report = plan(meta(rows))
    assert report == {
        "input": 9,
        "empty text": 1,
        "duplicate synth text": 2,
        "duplicate reid image": 1,
        "synth text also in reid": 2,
        "output": 3,
    }
    assert kept.height == report["output"]


def test_clean_streams_images_through(tmp_path):
    rows = [
        {**dict.fromkeys(SCHEMA), **r, "image": r["id"].encode(), "licence": "x"}
        for r in [reid("a", f"ক{ZWJ}"), reid("b", "")] + synth("wiki_bn/00001", "খ")
    ]
    src, out = tmp_path / "ingested.parquet", tmp_path / "clean.parquet"
    pl.DataFrame(rows, schema={**SCHEMA, "image_hash": pl.UInt64}).drop("image_hash").write_parquet(
        src
    )

    report = clean(src, out)

    df = pl.read_parquet(out)
    assert df.columns == [*SCHEMA, "text_raw"]
    assert df["id"].to_list() == [
        "reid2019/a",
        "synth/wiki_bn/00001/Galada",
        "synth/wiki_bn/00001/Hind",
    ]
    assert df["image"].to_list() == [i.encode() for i in df["id"]]  # image stays with its row
    assert df["text"].to_list() == ["ক", "খ", "খ"]
    assert report["output"] == 3
