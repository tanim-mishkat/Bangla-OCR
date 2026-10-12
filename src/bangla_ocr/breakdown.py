"""Error breakdown: which kinds of Bangla characters an engine gets wrong.

Every char edit is attributed once: a sub/del to the class of the reference char it hits,
an insertion to "inserted". So the class errors sum to the row's char_edits (CER numerator).
"""

import argparse
import sys
import unicodedata
from array import array
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from bangla_ocr import checks, score
from bangla_ocr.bootstrap import cluster_bootstrap

HASANTA = chr(0x09CD)  # হসন্ত: joins two consonants into a যুক্তাক্ষর
CONSONANTS = {chr(c) for c in [*range(0x0995, 0x09BA), 0x09DC, 0x09DD, 0x09DF]}  # ক..হ, ড় ঢ় য়
KAR = {
    chr(c) for c in [*range(0x09BE, 0x09C5), 0x09C7, 0x09C8, 0x09CB, 0x09CC, 0x09D7, 0x09E2, 0x09E3]
}
SIGNS = {chr(c) for c in [0x0981, 0x0982, 0x0983, 0x09BC]}  # চন্দ্রবিন্দু, অনুস্বার, বিসর্গ, নুকতা
MARKS = KAR | SIGNS  # never start a grapheme: they attach to the one before
KHANDA_TA = chr(0x09CE)  # ৎ: a consonant that never joins a যুক্তাক্ষর
VOWELS = {chr(c) for c in [*range(0x0985, 0x098D), 0x098F, 0x0990, 0x0993, 0x0994, 0x09E0, 0x09E1]}
DIGITS = {chr(c) for c in range(0x09E6, 0x09F0)}  # ০..৯
# class key -> report name; order = report order
CLASSES = {
    "conjunct": "যুক্তাক্ষর (conjunct)",
    "kar": "কারচিহ্ন (vowel sign)",
    "sign": "চিহ্ন (sign, lone হসন্ত)",
    "consonant": "ব্যঞ্জনবর্ণ (consonant)",
    "vowel": "স্বরবর্ণ (vowel)",
    "digit": "সংখ্যা (Bangla digit)",
    "space": "space",
    "punct": "punctuation",
    "other": "other (Latin, symbols, look-alikes ৰ ৷)",
}
NONE = "∅"  # the missing side of a deletion / insertion in confusion pairs


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


def _char_class(ch: str) -> str:
    if ch in CONSONANTS or ch == KHANDA_TA:
        return "consonant"
    if ch in KAR:
        return "kar"
    if ch in SIGNS or ch == HASANTA:
        return "sign"
    if ch in VOWELS:
        return "vowel"
    if ch in DIGITS:
        return "digit"
    if ch.isspace():
        return "space"
    if unicodedata.category(ch).startswith("P"):
        return "punct"
    return "other"


def char_classes(text: str) -> list[str]:
    """Class per code point. Consonants + হসন্ত inside a যুক্তাক্ষর are all "conjunct"."""
    out = []
    for g in bangla_graphemes(text):
        conjunct = sum(ch in CONSONANTS for ch in g) > 1
        for ch in g:
            joined = conjunct and (ch in CONSONANTS or ch == HASANTA)
            out.append("conjunct" if joined else _char_class(ch))
    return out


def _show(unit: str | None) -> str:
    return NONE if unit is None else unit.replace(" ", "␣")  # a bare space is invisible


def row_breakdown(ref: str, hyp: str) -> dict:
    """Per-row counts: err_/n_ per class, inserted, Latin leakage, grapheme edits, confusions."""
    classes = char_classes(ref)
    out = {f"err_{c}": 0 for c in CLASSES} | {f"n_{c}": 0 for c in CLASSES}
    for c in classes:
        out[f"n_{c}"] += 1
    out |= {"inserted": 0, "latin_digits": 0, "latin_letters": 0}
    i = 0
    for op, r, h in align(ref, hyp):
        if r is None:
            out["inserted"] += 1
        else:
            out[f"err_{classes[i]}"] += op != "match"
            i += 1
        if op != "match" and h is not None and h.isascii():
            out["latin_digits"] += h.isdigit()
            out["latin_letters"] += h.isalpha()
    ref_g, hyp_g = bangla_graphemes(ref), bangla_graphemes(hyp)
    pairs = [f"{_show(r)} → {_show(h)}" for op, r, h in align(ref_g, hyp_g) if op != "match"]
    return out | {"grapheme_edits": len(pairs), "graphemes": len(ref_g), "confusions": pairs}


def breakdown_rows(rows: pl.DataFrame) -> pl.DataFrame:
    """Score rows (needs text, hyp) + row_breakdown columns."""
    extra = [row_breakdown(ref, hyp) for ref, hyp in rows.select("text", "hyp").iter_rows()]
    schema = {"confusions": pl.List(pl.String)}
    return pl.concat([rows, pl.DataFrame(extra, schema_overrides=schema)], how="horizontal")


def _rate(rows: pl.DataFrame, cluster: str, num: str, den: str) -> dict:
    rows = rows.filter(pl.col(den) > 0)  # rows without this class carry no information on it
    if rows.is_empty():
        return {"errors": 0, "total": 0, "rate": None, "low": None, "high": None}
    low, high = cluster_bootstrap(rows, cluster, num, den)
    errors, total = rows[num].sum(), rows[den].sum()
    return {"errors": errors, "total": total, "rate": errors / total, "low": low, "high": high}


def headline(rows: pl.DataFrame, cluster: str) -> pl.DataFrame:
    """CER (code points) and GER (graphemes) with 95% cluster CIs."""
    return pl.DataFrame(
        [
            {"metric": "CER", **_rate(rows, cluster, "char_edits", "chars")},
            {"metric": "GER", **_rate(rows, cluster, "grapheme_edits", "graphemes")},
        ]
    )


def by_class(rows: pl.DataFrame, cluster: str) -> pl.DataFrame:
    """Error rate per reference class; `inserted` is per reference char. `share` sums to 1."""
    out = [
        {"class": name, **_rate(rows, cluster, f"err_{c}", f"n_{c}")} for c, name in CLASSES.items()
    ]
    out.append({"class": "inserted (per ref char)", **_rate(rows, cluster, "inserted", "chars")})
    df = pl.DataFrame(out)
    return df.with_columns(share=pl.col("errors") / pl.col("errors").sum())


def confusions(rows: pl.DataFrame, n: int = 15) -> pl.DataFrame:
    """Most frequent grapheme-level errors: "ref → hyp", ∅ = missing."""
    pairs = rows.select(pl.col("confusions").explode().drop_nulls())
    return (
        pairs.group_by("confusions")
        .len("count")
        .sort(["count", "confusions"], descending=[True, False])
        .head(n)
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Bangla in a Windows console
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="e.g. data/predictions/tesseract/<config_id>")
    ap.add_argument("--split", default="test", choices=sorted(checks.split_paths()))
    args = ap.parse_args()
    rows = breakdown_rows(score.score(args.run_dir, checks.split_paths()[args.split]))
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3, tbl_hide_dataframe_shape=True):
        for source, (cluster, _) in score.REPORTS.items():
            src = rows.filter(pl.col("source") == source)
            nonempty = src.filter(~pl.col("empty"))
            views = [
                ("OVERALL: empty predictions included (all label chars deleted)", src),
                ("DIAGNOSTIC: non-empty predictions only", nonempty),
            ]
            for title, v in views:
                print(f"\n=== {source} | {title} | n={v.height}, CI clusters: {cluster}")
                print(headline(v, cluster))
                print(by_class(v, cluster))
            print(f"\n=== {source} | top grapheme errors (non-empty predictions)")
            print(confusions(nonempty))
            print(
                f"Latin chars in erroneous output: {src['latin_digits'].sum()} digits, "
                f"{src['latin_letters'].sum()} letters"
            )
