# Progress

## Current phase
Phase 4 - Dagster + Hugging Face: Phase 5 - eval harness: **in progress**. Next: step 2 prediction cache.

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
- [x] Step 4: 5 OFL fonts + Bangla Wikipedia shard (pinned) -> `data/interim/corpus.parquet` (3000 lines)
- [x] Step 5: raqm render x 5 fonts + `augment` (written by me) -> `data/interim/synth/` (14984 imgs, 221 MB)
- [x] Step 6: `data/interim/ingested.parquet` (1264 reid + 14984 synth), `check_row` written by me, 0 bad rows
- [x] Step 7: README build commands + licence table (ShareAlike on synth subset)
- [x] Quiz: 3/3 correct (leakage direction, per-sample seeding, tofu = data bug in CER).
  Interview (new-source checks): strong; add eyeballing a sample, distribution comparison,
  and naming the gate (Dagster asset check / CI) + per-reason counts.

### Phase 3
- [x] Step 1: `normalise_text` (written by me): drop ZWSP/ZWNJ/ZWJ/BOM -> collapse whitespace -> NFC
- [x] Step 2: `clean.py` -> `data/interim/clean.parquet` (15661 = 677 reid + 14984 synth) + `clean_report.json`
- [x] Step 3: `assign_split` (written by me, sha256 bucket) -> `data/processed/{train,val,test}.parquet`:
  train 11920 / val 1495 / test 2246 (677 reid + 1569 synth); synth 79.6/10.0/10.5%
- [x] Step 4: `tests/test_data.py` (9 checks on real output, skipped if not built; each proven to fail
  on corrupted data) + `scripts/data_summary.py`
- [x] Quiz: 3/3 correct (ZWJ blocks NFC composition; hash split stable + `hash()` salted;
  dedupe key per source). Interview (3% CER vs prod): strong; add line crops vs full pages
  (segmentation untested) and that blended CER is mostly synth.

### Phase 4
- [x] Step 1: Dagster assets wrap each build step (`defs.py`), metadata per run; graph test;
  full CLI materialise = identical outputs (~1m45s with caches)
- [x] Step 2: `checks.py` = one data contract (9 checks -> list of problems) used by pytest AND
  blocking Dagster asset checks; `share_problems` written by me; blocking proven by test + mutation
- [x] Step 3 (code): `card.py` (`split_table` written by me) + `publish.py` (one atomic HF commit,
  refuses dirty code, private) + `hf_dataset` asset gated by the checks; fake-Hub tests
- [x] Step 3 (live): private `codemishkat/bangla-ocr-bench`. My review -> 2 fixes: card states exact
  split algorithm + bucket ranges; `publish()` re-runs the contract itself (materialising
  `hf_dataset` alone can no longer skip checks). Hub LFS sha256 == local (`publish.verify`).
- [x] Step 4: pinned `publish.PINNED` = revision `94f4021` (code `5956651`); data test fails if
  local splits drift from it. Phase 5 evaluates only this revision.
- [x] Quiz: 3/3 correct (blocking checks are per-run -> gate at point of action; atomic commit =
  no mixed revision; pin commit not `main`, hashes verify bytes). Interview strong; add: version
  tag + changelog of changed ids, re-score old models on both versions, deprecation window.

### Phase 5
- [x] Step 1: Tesseract 5.4.0 + pinned `tessdata_best/ben.traineddata` (11 MB, sha256) in
  `data/raw/tessdata`; `tesseract.py` CLI wrapper (stdin PNG, `--oem 1`, `--psm 7` line / 6 block).
  First 3 test images: decorative font -> conjunct errors; 4 deg rotation + noise -> EMPTY output.

### Data findings (for the write-up)
- REID2019 = 3 PAGE schema versions (2013/2016/2018); parse namespace-agnostic.
- Ground truth is region-level, not line-level -> real data = test-only "historical print" slice.
- 587/1264 regions have no text: 29 of 31 2016-schema pages are layout-only (likely hidden test set).
  Kept in interim; Phase 3 drops + reports them. 677 usable.
- Windows Pillow needs FriBiDi for raqm; it is looked up once when PIL.ImageDraw is imported,
  so `bangla_ocr/__init__.py` sets PATH first.
- Corpus: 3000 lines, median 35 chars, 505 with Bangla digits. Hidden chars: ZWNJ 3, ZWJ 5, NBSP 9
  -> Phase 3 normalisation must decide on these (they change CER while looking identical).
- Wikipedia is CC BY-SA -> synthetic subset (and published dataset) must be CC BY-SA.
- Same line rendered in 5 fonts -> Phase 3 split must group by `text_id` (leakage).
- 16 line-font pairs skipped: font lacks glyph (U+FDFA, ≠, −, ★, ć) -> would render tofu with a correct label.
- Clean: only drops = 587 empty REID. 76 REID rows share text with another (mostly page numbers)
  but are distinct images -> kept (REID is test-only, cannot leak); dedupe REID by image bytes.
  115/677 REID regions are page numbers -> Phase 5 CER by `region_type`.
- Test synth = 1569 images but only 314 unique texts (x5 fonts) -> Phase 5 bootstrap must resample
  by text_id (cluster), not by image, or CIs come out too narrow.
- REID test: median 16 chars, max 1374, 52% contain digits (vs ~16% synth) -> report REID separately.
- Levels ~1/3 each (clean/mild/heavy); heavy still human-readable (eyeballed).
- Noisy PNGs barely compress (noise is random): 221 MB, kept lossless (JPEG adds a hidden 2nd variable).

## Struggled with / learned
- Testing boundaries: first tests missed U+0980/U+09FF edges; `<` vs `<=` bug passed silently.
- Tests must use hard-coded expected values, not the code's own constants.
- `augment` passed all tests but returned the input object + aliased rotation: tests only prove what they check.
- Validators return a list of reasons, not a bool: per-check counts point at the bug.
- Each parametrised bad-row case must contain exactly one fault (test data bug caught by a reference impl).
- Test both sides of an edge (just inside and just outside).
- Shared RNG breaks on skips too: a skipped sample consumes no draws, shifting all later ones.
- Biased bugs (tofu only in some fonts/rare chars) land in exactly the slices the eval breaks down.
- Reproducibility gaps to remember: system tools (Tesseract) aren't in `uv.lock`;
  Unicode normalisation changes CER; save raw model outputs so scoring is re-runnable.

- Dedupe key follows the reason: text for leakage (synth), image bytes for true duplicates (REID).
- Invisible chars in source code: write as `\u200d` escapes, never literals (unreviewable).

- `map_elements` skips nulls: a guard inside the function never fires for them -> check nulls outside.
- Float tolerance edges: abs(0.08-0.10) > 0.02 is True; test just inside/outside, not on the edge.
- Hash split, not shuffle: stable when data is added; never Python `hash()` (per-process salt).

## Habits
- Before commit: `uv run ruff format . ; uv run ruff check . ; uv run pytest`
- Always `open(path, encoding="utf-8")` on Windows.
- Long `python -c` one-liners break in PowerShell (quotes/braces) -> put them in `scripts/`.
- After each commit: `git push`.

## Next
- Phase 4 quiz, then Phase 5: eval harness on `PINNED` (Tesseract, EasyOCR, vision LLMs; CER/WER,
  bootstrap CIs by `text_id`, error breakdowns, REID reported separately).
