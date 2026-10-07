# Bangla OCR Benchmark

A reproducible benchmark for Bangla (Bengali) OCR: a data pipeline that builds a clean
text-image dataset, and an eval harness that scores OCR systems with CER/WER, bootstrap
confidence intervals, and per-category error breakdowns.

> Status: Phase 1 (setup). See [PROGRESS.md](PROGRESS.md).

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
```

Copy `.env.example` to `.env` and fill in API keys (never commit `.env`).

## Layout

```
src/bangla_ocr/   package code
tests/            pytest tests
data/             raw / interim / processed (gitignored)
notebooks/        GPU notebooks for Kaggle/Colab
```

## Data & licences

TBD in Phase 2: sources, licences, and attribution for every dataset used.
