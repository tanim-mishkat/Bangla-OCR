"""Small helpers for working with Bangla text."""

BANGLA_START = 0x0980
BANGLA_END = 0x09FF


def is_bangla_char(ch: str) -> bool:
    """Return True if `ch` is a single character in the Bengali Unicode block.

    Raises ValueError if `ch` is not exactly one character.
    """
    if len(ch) != 1:
        raise ValueError(f"Expected one character, got {len(ch)}: {ch!r}")

    return BANGLA_START <= ord(ch) <= BANGLA_END


def bangla_ratio(text: str) -> float:
    """Return the fraction of non-whitespace characters in `text` that are Bangla.

    Returns 0.0 for empty or whitespace-only text.
    """
    total = 0
    bangla = 0

    for ch in text:
        if ch.isspace():
            continue

        total += 1

        if is_bangla_char(ch):
            bangla += 1

    if total == 0:
        return 0.0

    return bangla / total