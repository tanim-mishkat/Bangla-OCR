# Progress

## Current phase
Phase 2 - Ingest: **in progress** (steps 1-3 of 7 done). Next: step 4 (fonts + source text).

## Environment
- Windows 11, PowerShell, 8GB RAM, no GPU
- Python 3.12 via uv; Git on `main`
- Accounts: GitHub, Kaggle, Colab. Gemini key (to get). Hugging Face: create before Phase 4.
- `.env` not created yet (copy from `.env.example` when a key is needed)

## Done
### Phase 1
- [x] Installed uv
- [x] Git repo + `.gitignore` (data, `.venv`, `.env` excluded)
- [x] Package scaffold (`src/bangla_ocr`), pytest + ruff as dev deps, `uv.lock`
- [x] `is_bangla_char` + `bangla_ratio` written by me, 20 tests passing
- [x] README, PROGRESS, `.env.example`
- [x] Lint + format clean (`ruff check`, `ruff format`)
- [x] Quiz: all 3 correct; interview answer strong (4 layers: code, data, randomness, models)

### Phase 2
- [x] Step 1: polars/pillow/numpy; FriBiDi DLL (`scripts/get_fribidi.ps1`) + `ensure_raqm()` guard
- [x] Step 2: REID2019 download, streaming + pinned sha256 (`bangla_ocr.download`)
- [x] Step 3: PAGE XML -> region crops -> `data/interim/reid2019/*.parquet` (1264 rows)
- [ ] Step 4: fonts + Bangla source text
- [ ] Step 5: synthetic render + augment (I write `augment`)
- [ ] Step 6: combine to `data/interim/ingested.parquet` + schema test (I write row check)
- [ ] Step 7: README licences + quiz

### Data findings (for the write-up)
- REID2019 = 3 PAGE schema versions (2013/2016/2018); parse namespace-agnostic.
- Ground truth is region-level, not line-level -> real data = test-only "historical print" slice.
- 587/1264 regions have no text: 29 of 31 2016-schema pages are layout-only (likely hidden test set).
  Kept in interim; Phase 3 drops + reports them. 677 usable.
- Windows Pillow needs FriBiDi for raqm; it is looked up once when PIL.ImageDraw is imported,
  so `bangla_ocr/__init__.py` sets PATH first.

## Struggled with / learned
- Testing boundaries: first tests missed U+0980/U+09FF edges; `<` vs `<=` bug passed silently.
- Tests must use hard-coded expected values, not the code's own constants.
- Test both sides of an edge (just inside and just outside).
- Reproducibility gaps to remember: system tools (Tesseract) aren't in `uv.lock`;
  Unicode normalisation changes CER; save raw model outputs so scoring is re-runnable.

## Habits
- Before commit: `uv run ruff format . ; uv run ruff check . ; uv run pytest`
- Always `open(path, encoding="utf-8")` on Windows.

## Next
- Phase 2 intro: what ingest means, Parquet, licences, synthetic rendering.
- Find and verify a public Bangla OCR dataset with a reuse-friendly licence.
