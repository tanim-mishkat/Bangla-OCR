"""Hugging Face dataset card (README.md) built from the pipeline's own numbers."""

from bangla_ocr.sources import REID2019, WIKI_BN

HF_SPLIT = {"train": "train", "val": "validation", "test": "test"}  # HF naming convention


def split_table(counts: dict[str, dict[str, int]]) -> str:
    """Markdown table of rows per split (rows, in input order) and source (columns, sorted).

    counts = {"train": {"synth": 8}, "test": {"reid2019": 3, "synth": 1}} ->
    | split | reid2019 | synth | total |
    |---|---:|---:|---:|
    | train | 0 | 8 | 8 |
    | test | 3 | 1 | 4 |
    | **total** | 3 | 9 | 12 |

    A source missing from a split counts as 0. Lines joined with "\\n", no trailing newline.
    """
    # 1. Find every unique source and sort alphabetically.
    sources = sorted({s for c in counts.values() for s in c})

    # 2. Build the table header and separator.
    lines = [
        "| split | " + " | ".join(sources) + " | total |",
        "|---|" + "---:|" * (len(sources) + 1),
    ]

    # 3. Add one row for each split.
    for name, c in counts.items():
        values = [c.get(s, 0) for s in sources]

        lines.append(f"| {name} | " + " | ".join(map(str, [*values, sum(values)])) + " |")

    # 4. Calculate totals for each source.
    totals = [sum(c.get(s, 0) for c in counts.values()) for s in sources]

    # 5. Add the final total row.
    lines.append("| **total** | " + " | ".join(map(str, [*totals, sum(totals)])) + " |")

    # 6. Combine all rows into one Markdown string.
    return "\n".join(lines)


def render(
    counts: dict[str, dict[str, int]],
    report: dict[str, int],
    sha256: dict[str, str],
    code_revision: str,
) -> str:
    """Full card: YAML metadata + description, split method, provenance, limitations."""
    data_files = "\n".join(
        f"  - split: {HF_SPLIT[name]}\n    path: data/{name}.parquet" for name in counts
    )
    drops = "\n".join(f"| {k} | {v} |" for k, v in report.items())
    checksums = "\n".join(f"| `data/{name}.parquet` | `{h}` |" for name, h in sha256.items())
    return f"""---
license: cc-by-sa-3.0
language:
- bn
task_categories:
- image-to-text
tags:
- ocr
- bangla
- bengali
- synthetic
pretty_name: Bangla OCR Benchmark
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
{data_files}
---

# Bangla OCR Benchmark

Text-image pairs for evaluating Bangla (Bengali) OCR: synthetic Wikipedia lines rendered in
5 fonts at 3 degradation levels, plus real historical print from the British Library (REID2019,
test split only). Built by a reproducible pipeline: code revision `{code_revision}`.

## Splits

{split_table(counts)}

- **Synthetic**: split by `sha256(text_id) % 100` (80/10/10). All renders of one line share
  a split, so no text appears in two splits. Stable when data is added.
- **REID2019**: test only. Its ground truth is region-level, so it is a separate
  "real historical print" slice. **Report it separately from synthetic.**
- Test synthetic has ~5 images per unique text: bootstrap CIs by `text_id`, not by image.

## Columns

| column | meaning |
|---|---|
| `id` | unique row id (`reid2019/...` or `synth/<text_id>/<font>`) |
| `image` | PNG bytes (`width` x `height` px) |
| `text` / `text_raw` | normalised label / label as found in the source |
| `source`, `licence` | `reid2019` or `synth`, and that row's licence |
| `page`, `region_type` | REID2019 page and PAGE-XML region type (e.g. paragraph, page-number) |
| `text_id`, `font`, `level` | synthetic line id, font, degradation (clean/mild/heavy) |
| `noise`, `blur`, `rotation` | synthetic degradation parameters actually applied |

**Normalisation** (apply the same to model predictions before scoring): remove ZWSP/ZWNJ/ZWJ/BOM,
collapse whitespace runs to one space, then Unicode NFC.

## Cleaning

| step | rows |
|---|---:|
{drops}

## Quality checks

Every publish is gated by blocking checks: schema, row count vs cleaning report, unique ids,
every row has an image, REID only in test, no `text_id` or `text` in two splits, labels
normalised and non-empty, synthetic split shares within ±2%.

| file | sha256 |
|---|---|
{checksums}

## Limitations

- Synthetic images are line-level crops; page segmentation is not evaluated.
- REID2019 labels are region-level and include page numbers; long regions (up to ~1400 chars).
- Wikipedia text over-represents encyclopedic vocabulary; few digits (~16% of lines).
- `image` is stored as raw PNG bytes (decode with PIL); the Hub viewer shows it as binary.

## Licence and attribution

- Synthetic subset: text from Bangla Wikipedia ({WIKI_BN.licence}, Wikipedia contributors),
  so this dataset is released under **CC BY-SA 3.0**. Fonts (OFL-1.1) are not redistributed.
- REID2019 subset: {REID2019.licence}. Each row carries its own `licence` column.

> {REID2019.citation}
"""
