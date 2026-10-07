"""Bangla OCR benchmark: data pipeline and eval harness."""

from bangla_ocr.shaping import add_fribidi_to_path

add_fribidi_to_path()  # must run before anything imports PIL.ImageDraw/ImageFont
