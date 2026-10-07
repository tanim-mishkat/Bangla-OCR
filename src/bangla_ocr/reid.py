"""Parse REID2019 PAGE XML and crop text regions into per-page Parquet files."""

import io
import xml.etree.ElementTree as ET  # trusted, checksum-pinned input; py3.12 expat is XXE-safe
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from PIL import Image, ImageDraw

from bangla_ocr.sources import REID2019

SRC_DIR = Path("data/raw/reid2019/Competition_dataset_ImagesPAGEXML")
OUT_DIR = Path("data/interim/reid2019")
SRC_DPI = 400
TARGET_DPI = 300  # standard OCR resolution; ~44% fewer pixels than 400

SCHEMA = {
    "id": pl.String,
    "image": pl.Binary,  # PNG bytes, grayscale
    "text": pl.String,  # raw ground truth, normalised in Phase 3
    "source": pl.String,
    "licence": pl.String,
    "page": pl.String,
    "region_type": pl.String,
    "width": pl.Int32,
    "height": pl.Int32,
}

Point = tuple[int, int]


@dataclass(frozen=True)
class Region:
    id: str
    type: str
    points: list[Point]
    text: str


def _local(tag: str) -> str:
    """'{namespace}TextRegion' -> 'TextRegion'. Files mix PAGE schema versions."""
    return tag.rsplit("}", 1)[-1]


def _child(el: ET.Element, name: str) -> ET.Element | None:
    return next((c for c in el if _local(c.tag) == name), None)


def parse_points(points: str) -> list[Point]:
    """'1,2 3,4' -> [(1, 2), (3, 4)]"""
    return [(int(x), int(y)) for x, y in (p.split(",") for p in points.split())]


def _text(el: ET.Element) -> str:
    """Unicode text of el's own TextEquiv ('' if missing)."""
    equiv = _child(el, "TextEquiv")
    uni = _child(equiv, "Unicode") if equiv is not None else None
    return (uni.text or "") if uni is not None else ""


def iter_regions(xml_path: Path) -> Iterator[Region]:
    """Yield every TextRegion with its polygon and text.

    Text is region-level if present, else its TextLines joined by newlines.
    """
    for el in ET.parse(xml_path).getroot().iter():
        if _local(el.tag) != "TextRegion":
            continue
        coords = _child(el, "Coords")
        text = _text(el)
        if not text.strip():
            text = "\n".join(_text(c) for c in el if _local(c.tag) == "TextLine")
        yield Region(
            id=el.get("id", ""),
            type=el.get("type", ""),
            points=parse_points(coords.get("points", "")) if coords is not None else [],
            text=text,
        )


def crop_region(page: Image.Image, points: list[Point]) -> Image.Image | None:
    """Crop the polygon's bounding box; pixels outside the polygon become white.

    Returns None for polygons too small to crop.
    """
    if len(points) < 3:
        return None
    xs, ys = zip(*points, strict=True)
    left, top = max(min(xs), 0), max(min(ys), 0)
    right, bottom = min(max(xs) + 1, page.width), min(max(ys) + 1, page.height)
    if right - left < 2 or bottom - top < 2:
        return None

    crop = page.crop((left, top, right, bottom))
    mask = Image.new("L", crop.size, 0)
    ImageDraw.Draw(mask).polygon([(x - left, y - top) for x, y in points], fill=255)
    return Image.composite(crop, Image.new("L", crop.size, 255), mask)


def page_rows(xml_path: Path) -> pl.DataFrame:
    """One row per cropped region of a page (image = sibling .tif)."""
    with Image.open(xml_path.with_suffix(".tif")) as im:
        page = im.convert("L")
    scale = TARGET_DPI / SRC_DPI

    rows = []
    for region in iter_regions(xml_path):
        crop = crop_region(page, region.points)
        if crop is None:
            continue
        size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
        crop = crop.resize(size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        rows.append(
            {
                "id": f"{REID2019.name}/{xml_path.stem}/{region.id}",
                "image": buf.getvalue(),
                "text": region.text,
                "source": REID2019.name,
                "licence": REID2019.licence,
                "page": xml_path.stem,
                "region_type": region.type,
                "width": crop.width,
                "height": crop.height,
            }
        )
    return pl.DataFrame(rows, schema=SCHEMA)


def build(src_dir: Path = SRC_DIR, out_dir: Path = OUT_DIR) -> None:
    """Write one Parquet per page. Skips pages already written (resumable)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    xmls = sorted(src_dir.glob("*.xml"))
    total = 0
    for i, xml_path in enumerate(xmls, 1):
        out = out_dir / f"{xml_path.stem}.parquet"
        if not out.exists():
            page_rows(xml_path).write_parquet(out)
        n = pl.scan_parquet(out).select(pl.len()).collect().item()
        total += n
        print(f"[{i}/{len(xmls)}] {xml_path.stem}: {n} regions")
    print(f"Done: {total} regions in {out_dir}")


if __name__ == "__main__":
    build()
