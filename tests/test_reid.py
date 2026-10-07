import io

import pytest
from PIL import Image

from bangla_ocr.reid import crop_region, iter_regions, page_rows, parse_points

NS_2013 = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
NS_2018 = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2018-07-15"


def page_xml(ns: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="{ns}">
  <Page imageFilename="p.tif" imageWidth="100" imageHeight="80">
    <TextRegion id="r0" type="paragraph">
      <Coords points="10,10 50,10 50,30 10,30"/>
      <TextEquiv><Unicode>আমার সোনার বাংলা</Unicode></TextEquiv>
    </TextRegion>
    <TextRegion id="r1" type="page-number">
      <Coords points="60,60 60,65 60,70"/>
      <TextEquiv><Unicode>১২</Unicode></TextEquiv>
    </TextRegion>
  </Page>
</PcGts>"""


@pytest.fixture(params=[NS_2013, NS_2018])
def xml_path(request, tmp_path):
    path = tmp_path / "p.xml"
    path.write_text(page_xml(request.param), encoding="utf-8")
    Image.new("L", (100, 80), 0).save(tmp_path / "p.tif")  # all-black page
    return path


def test_parse_points():
    assert parse_points("1,2 30,40") == [(1, 2), (30, 40)]


def test_iter_regions_any_schema_version(xml_path):
    regions = list(iter_regions(xml_path))
    assert [r.id for r in regions] == ["r0", "r1"]
    assert regions[0].type == "paragraph"
    assert regions[0].text == "আমার সোনার বাংলা"
    assert regions[0].points[0] == (10, 10)


def test_iter_regions_falls_back_to_text_lines(tmp_path):
    path = tmp_path / "lines.xml"
    path.write_text(
        f"""<PcGts xmlns="{NS_2013}"><Page>
          <TextRegion id="r0" type="paragraph">
            <Coords points="0,0 9,0 9,9"/>
            <TextLine id="l0"><TextEquiv><Unicode>প্রথম</Unicode></TextEquiv></TextLine>
            <TextLine id="l1"><TextEquiv><Unicode>দ্বিতীয়</Unicode></TextEquiv></TextLine>
          </TextRegion>
          <TextRegion id="r1"><Coords points="0,0 9,0 9,9"/></TextRegion>
        </Page></PcGts>""",
        encoding="utf-8",
    )
    regions = list(iter_regions(path))
    assert regions[0].text == "প্রথম\nদ্বিতীয়"
    assert regions[1].text == ""  # layout-only region: kept, flagged in Phase 3


def test_crop_region_masks_outside_polygon():
    page = Image.new("L", (100, 100), 0)
    triangle = [(0, 0), (40, 0), (0, 40)]
    crop = crop_region(page, triangle)
    assert crop.size == (41, 41)
    assert crop.getpixel((2, 2)) == 0  # inside: original black
    assert crop.getpixel((39, 39)) == 255  # outside: white


@pytest.mark.parametrize("points", [[], [(1, 1), (5, 5)], [(5, 5), (5, 6), (5, 7)]])
def test_crop_region_rejects_degenerate(points):
    assert crop_region(Image.new("L", (100, 100)), points) is None


def test_page_rows_skips_tiny_regions_and_downscales(xml_path):
    df = page_rows(xml_path)
    assert df["id"].to_list() == ["reid2019/p/r0"]
    row = df.row(0, named=True)
    assert (row["width"], row["height"]) == (31, 16)  # 41x21 at 400dpi -> 300dpi
    assert row["licence"] == "Public Domain Mark 1.0"
    assert Image.open(io.BytesIO(row["image"])).size == (31, 16)
