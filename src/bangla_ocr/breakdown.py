"""Error breakdown: which kinds of Bangla characters an engine gets wrong."""

from array import array
from collections.abc import Sequence

HASANTA = chr(0x09CD)  # হসন্ত: joins two consonants into a যুক্তাক্ষর
CONSONANTS = {chr(c) for c in [*range(0x0995, 0x09BA), 0x09DC, 0x09DD, 0x09DF]}  # ক..হ, ড় ঢ় য়
KAR = {
    chr(c) for c in [*range(0x09BE, 0x09C5), 0x09C7, 0x09C8, 0x09CB, 0x09CC, 0x09D7, 0x09E2, 0x09E3]
}
SIGNS = {chr(c) for c in [0x0981, 0x0982, 0x0983, 0x09BC]}  # চন্দ্রবিন্দু, অনুস্বার, বিসর্গ, নুকতা
MARKS = KAR | SIGNS  # never start a grapheme: they attach to the one before


def bangla_graphemes(text: str) -> list[str]:
    """Split text into the units a reader sees: one base + any যুক্তাক্ষর + কারচিহ্ন/signs.

    Rules, left to right:
    - a MARK or HASANTA joins the current grapheme
    - a consonant right after a HASANTA joins too (that makes a যুক্তাক্ষর)
    - anything else (consonant, vowel, digit, space, punctuation, Latin) starts a new one
    Lossless: "".join(bangla_graphemes(s)) == s.
    """
    out = []

    for ch in text:
        if out and (ch in MARKS or ch == HASANTA or (ch in CONSONANTS and out[-1][-1] == HASANTA)):
            out[-1] += ch
        else:
            out.append(ch)

    return out


def align(ref: Sequence, hyp: Sequence) -> list[tuple[str, object, object]]:
    """One minimal alignment as (op, r, h): "match", "sub", "del" (h=None), "ins" (r=None).

    Full DP table + backtrace. Ties prefer match/sub > del > ins, so a wrong character
    shows up as a confusion pair, not as a deletion plus an insertion.
    """
    n, m = len(ref), len(hyp)
    d = [array("I", range(m + 1))]  # 4 bytes/cell: a 1374x1374 REID line is ~7.5 MB
    for i in range(1, n + 1):
        prev, row, r = d[-1], array("I", [i]) * (m + 1), ref[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (r != hyp[j - 1]))
        d.append(row)
    ops = []
    i, j = n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            op = "match" if ref[i - 1] == hyp[j - 1] else "sub"
            ops.append((op, ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
    return ops[::-1]
