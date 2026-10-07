# Progress

## Current phase
Phase 1 - Setup (steps 1-5 done; lint check and quiz left)

## Environment
- Windows 11, PowerShell, 8GB RAM, no GPU
- Python 3.12 via uv; Git on `main`
- Accounts: GitHub, Kaggle, Colab. Gemini key (to get). Hugging Face: create before Phase 4.

## Done
- [x] Installed uv
- [x] Git repo + `.gitignore` (data, `.venv`, `.env` excluded)
- [x] Package scaffold (`src/bangla_ocr`), pytest + ruff as dev deps, `uv.lock`
- [x] `is_bangla_char` + `bangla_ratio` written by me, 20 tests passing
- [x] README, PROGRESS, `.env.example`

## Struggled with / learned
- Testing boundaries: first tests missed U+0980/U+09FF edges; `<` vs `<=` bug passed silently.
- Tests must use hard-coded expected values, not the code's own constants.
- Test both sides of an edge (just inside and just outside).

## Next
- Step 6: `uv run ruff check .`
- Step 7: Phase 1 quiz
- Then Phase 2: Ingest (find a licensed public Bangla OCR dataset + synthetic renders)
