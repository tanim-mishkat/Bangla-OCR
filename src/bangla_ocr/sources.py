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
