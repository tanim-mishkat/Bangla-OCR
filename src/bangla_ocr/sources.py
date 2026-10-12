"""Registry of external data sources: where they live and how we may use them."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    filename: str
    sha256: str | None  # None until first download; then pin it here
    licence: str
    citation: str


REID2019 = Source(
    name="reid2019",
    url="https://bl.iro.bl.uk/bitstreams/eaa28f8f-b0a7-4d3c-af44-7b667436e036/download",
    filename="REID2019.zip",
    sha256="c02db96f8393e15af7c4453cc8e366c8fa7f84ab7a63fcad850ef4d207cf1b7f",
    licence="Public Domain Mark 1.0",
    citation=(
        "Ground truth transcriptions for training OCR of historical Bengali printed texts. "
        "British Library; 2020. Version 2. https://doi.org/10.23636/1168"
    ),
)

_WIKI_REV = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
WIKI_BN = Source(
    name="wiki_bn",
    url=(
        f"https://huggingface.co/datasets/wikimedia/wikipedia/resolve/{_WIKI_REV}"
        "/20231101.bn/train-00001-of-00002.parquet"
    ),
    filename="wiki_bn/train-00001-of-00002.parquet",
    sha256="47c0095c58d6595f68d4bccc257ae6646f27a4669a7d259369c11626bd39565a",
    licence="CC BY-SA 3.0",
    citation="Bangla Wikipedia (2023-11-01 dump) via huggingface.co/datasets/wikimedia/wikipedia",
)

_FONTS_REV = "7085eb89a950e85db5b166b7a58d414544b4140c"
_FONTS_URL = f"https://raw.githubusercontent.com/google/fonts/{_FONTS_REV}/ofl/"


def _font(name: str, path: str, sha256: str) -> Source:
    return Source(
        name=name,
        url=_FONTS_URL + path,
        filename=f"fonts/{name}.ttf",
        sha256=sha256,
        licence="OFL-1.1",
        citation=f"{name}, Google Fonts (github.com/google/fonts)",
    )


FONTS = (
    _font(
        "NotoSansBengali",
        "notosansbengali/NotoSansBengali%5Bwdth,wght%5D.ttf",
        "dcd42978094e584a849c84a51450eeac40c8826057d566ea6d4b9627a403a05a",
    ),
    _font(
        "NotoSerifBengali",
        "notoserifbengali/NotoSerifBengali%5Bwdth,wght%5D.ttf",
        "083180a7b47cf3b101ed55a8724aa1b38869daa9baf23b7cf198f9d225365aba",
    ),
    _font(
        "HindSiliguri",
        "hindsiliguri/HindSiliguri-Regular.ttf",
        "45071a7da902a74877f432722f525d00502d61aef71fca4cab3da8a23156c33d",
    ),
    _font(
        "TiroBangla",
        "tirobangla/TiroBangla-Regular.ttf",
        "a763523dc3f90a71962cb8d9400b8f0a86b40b2cb01676f165101a2967127ae4",
    ),
    _font(
        "Galada",
        "galada/Galada-Regular.ttf",
        "acc22b41ee470dc542e15b5f8907c7fbebef1b9c01eb20e4127bdbf052b006ee",
    ),
)

_TESSDATA_REV = "e12c65a915945e4c28e237a9b52bc4a8f39a0cec"
TESS_BEN = Source(
    name="tess_ben",
    url=f"https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/{_TESSDATA_REV}/ben.traineddata",
    filename="tessdata/ben.traineddata",
    sha256="1cd0129288d1f74f6661c35e638e487bd146a66bb732ecd9e011e48dcb0df623",
    licence="Apache-2.0",
    citation="Tesseract tessdata_best Bengali LSTM model (github.com/tesseract-ocr/tessdata_best)",
)

# EasyOCR weights: fetched + sha256-pinned by us, never by EasyOCR's own downloader.
# The releases are tags, but assets can be replaced: the hash is what pins them.
_EASYOCR_URL = "https://github.com/JaidedAI/EasyOCR/releases/download/"
EASYOCR_CRAFT = Source(
    name="easyocr_craft",
    url=_EASYOCR_URL + "pre-v1.1.6/craft_mlt_25k.zip",
    filename="easyocr/craft_mlt_25k.zip",
    sha256="8dc6a1c703a89ed56308ef742d26ebd45c656248cbbbda6e7fe60e569f873e65",
    licence="MIT (CRAFT-pytorch, Clova AI)",
    citation="CRAFT detector (Baek et al., CVPR 2019) via EasyOCR (github.com/JaidedAI/EasyOCR)",
)
EASYOCR_BEN = Source(
    name="easyocr_ben",
    url=_EASYOCR_URL + "v1.1.8/bengali.zip",
    filename="easyocr/bengali.zip",
    sha256="b9f88f7beef9aa5480c62e9f84c50be82214e5b209d0115465c30175eca4f944",
    licence="Apache-2.0",
    citation="EasyOCR bengali_g1 recognition model (github.com/JaidedAI/EasyOCR)",
)
