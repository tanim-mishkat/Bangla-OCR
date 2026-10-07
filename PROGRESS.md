# Progress

## Current phase
Phase 1 - Setup: **complete**. Next: Phase 2 - Ingest (intro not started).

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
