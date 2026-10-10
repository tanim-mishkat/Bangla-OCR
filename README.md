# Bangla OCR Benchmark

A reproducible benchmark for Bangla (Bengali) OCR: a data pipeline that builds a clean
text-image dataset, and an eval harness that scores OCR systems with CER/WER, bootstrap
confidence intervals, and per-category error breakdowns.

> Status: Phase 2 (ingest) done; Phase 3 (clean & validate) next. See [PROGRESS.md](PROGRESS.md).

## Roadmap

1. **Setup** - repo, uv env, tests
2. **Ingest** - public Bangla OCR data + synthetic renders (fonts, noise, blur, rotation) to Parquet
3. **Clean & validate** - Unicode normalisation, dedupe, schema checks, leak-free split
4. **Orchestrate & publish** - Dagster pipeline, dataset on Hugging Face
5. **Eval harness** - Tesseract, EasyOCR, vision LLMs; CER/WER with bootstrap CIs
6. **Fine-tune** - small OCR model on Kaggle GPU, before/after results
7. **Write-up** - results, error analysis, limitations

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```powershell
uv sync              # create .venv and install exact locked versions
uv run pytest        # run tests
uv run ruff check .  # lint
uv run ruff format . # format
```

Copy `.env.example` to `.env` and fill in API keys (never commit `.env`).

## Build the data

Each step reads the previous step's output and skips work already done.

```powershell
.\scripts\get_fribidi.ps1                 # Windows only: FriBiDi DLL for Bangla text shaping
uv run python -m bangla_ocr.download      # sources -> data/raw/ (sha256-checked)
uv run python -m bangla_ocr.reid          # REID2019 pages -> region crops (1264)
uv run python -m bangla_ocr.corpus        # Wikipedia -> 3000 text lines
uv run python -m bangla_ocr.synth         # lines x 5 fonts -> degraded images (~15k, ~220 MB)
uv run python -m bangla_ocr.ingest        # combine + validate -> data/interim/ingested.parquet
```

## Layout

```
src/bangla_ocr/   package code
tests/            pytest tests
data/             raw / interim / processed (gitignored)
notebooks/        GPU notebooks for Kaggle/Colab
```

## Data & licences

Every download is pinned to a fixed revision and sha256 (see `src/bangla_ocr/sources.py`).
Each row in the dataset carries its own `licence` column.

| Source | Used for | Licence |
|---|---|---|
| [REID2019](https://doi.org/10.23636/1168), British Library | real historical print (test only) | Public Domain Mark 1.0 |
| [Bangla Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (2023-11-01 dump) | text for synthetic images | CC BY-SA 3.0 |
| Noto Sans/Serif Bengali, Hind Siliguri, Tiro Bangla, Galada ([Google Fonts](https://github.com/google/fonts)) | rendering synthetic images | OFL-1.1 |
| [FriBiDi](https://github.com/fribidi/fribidi) | text shaping on Windows (downloaded, not redistributed) | LGPL-2.1 |

**ShareAlike:** synthetic images reproduce Wikipedia text, so the synthetic subset, and
any published dataset that includes it, is released under **CC BY-SA 3.0** with
attribution to Wikipedia contributors. The REID2019 subset stays public domain.
Fonts are only used to render; they are not redistributed.

Citation for REID2019:
> Ground truth transcriptions for training OCR of historical Bengali printed texts.
> British Library; 2020. Version 2. https://doi.org/10.23636/1168
