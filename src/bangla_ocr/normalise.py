"""One normalisation policy for every label AND every model prediction."""

import unicodedata

# Format characters an OCR system cannot see, so cannot be expected to output.
INVISIBLE = {"\u200b", "\u200c", "\u200d", "\ufeff"}  # ZWSP, ZWNJ, ZWJ, BOM


def normalise_text(text: str) -> str:
    """Canonical form of a label or prediction.

    1. drop INVISIBLE characters
    2. collapse every whitespace run (incl. NBSP, tabs, newlines) to one space; trim ends
    3. Unicode NFC
    """
    s = "".join(ch for ch in text if ch not in INVISIBLE)
    s = " ".join(s.split())
    return unicodedata.normalize("NFC", s)
