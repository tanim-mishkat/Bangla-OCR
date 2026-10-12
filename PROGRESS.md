# Progress

## Current phase
Phase 5 - eval harness: **in progress**. Tesseract + EasyOCR done; Gemini 10A (sampling) done.
Next: AI Studio limits -> final QUOTAS -> 10B Gemini engine, then the Phase 5 quiz.

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
- [x] My review: "Failed loading language 'eng'" warning. Cause: ben.traineddata config has
  `tessedit_load_sublangs eng`; with eng present a Bangla word became "wets". Labels have 0 Latin
  -> `tesseract_ben_only.cfg` disables it (`-c` can't: init-time param). Regression test + mutation.
- [x] Step 2: `predict.py` resumable cache, one row per id (pred, seconds, error) in atomic parquet
  parts; key = hash(engine config + split sha256) -> new settings/data never reuse rows; errors
  stored + retried, `""` kept (CER 1). CLI `python -m bangla_ocr.predict tesseract [--limit N]`.
- [x] Step 3: full Tesseract test run (cache `fd3806085e5a`): 2246 preds, 0 errors, 307 s total;
  rerun = 2246 cached. 129 empty (5.7%): REID 70/677, synth heavy 59/532, clean + mild 0.
- [x] Step 4: `edit_distance` (written by me): two-row Levenshtein; 11 tests incl. random cross-check
  vs recursive reference + tracemalloc test that a full matrix fails.
- [x] Step 5: `score.py`: normalise pred like labels, null/error -> "" (CER 1), refuses incomplete runs
  and predictions made on other data (config `data_sha256`), micro-averaged CER/WER per slice,
  REID never blended with synth. Tesseract: synth CER 0.066 (clean 0.019 / mild 0.022 / heavy 0.155),
  Galada worst 0.098; REID CER 0.236, WER 0.594; page numbers CER 0.98 (57x41 px crops, ~2 chars,
  `--psm 6`, Latin digits leak in) -> REID is test-only, so no tuning possible on it.
- [x] Step 6: `cluster_bootstrap` (written by me): percentile CI, whole clusters, seeded, sorted groups.
  Real design effect (cluster CI width / row CI width): synth by text_id 1.11 (font + degradation
  drive errors, not the text), REID by page 1.84 (scan quality per page). REID text_id is null ->
  clustering by it would give one cluster and a zero-width CI; `cluster` has no default.
- [x] Step 7: `paired_bootstrap` = `cluster_bootstrap` on per-row edit differences (same labels ->
  same chars, so CER diff = sum(diff)/sum(chars)); `score` report has 95% CIs + cluster counts,
  `--vs RUN` prints paired diffs with `clear` = CI excludes 0. Tesseract CER: synth 0.066
  [0.055, 0.077], REID 0.236 [0.183, 0.308]. Slices with few clusters (REID other: 2,
  drop-capital: 1) have meaningless CIs: read the `clusters` column.
- [x] Step 8: `breakdown.py`. `bangla_graphemes` (written by me: MARK/হসন্ত join, consonant after
  হসন্ত joins = যুক্তাক্ষর; own rules because Python 3.12 Unicode 15.0 lacks the 15.1 conjunct break).
  `align` = full DP + backtrace (ties: sub > del > ins), `array("I")` rows (1374-char REID line ~7.5 MB).
  Every char edit attributed once (sub/del -> ref char class, ins -> "inserted"): class errors sum
  to char_edits on all 2246 rows. GER = grapheme edits / ref graphemes (~1.6 code points each).
  Two views: OVERALL (empties = all deleted) and DIAGNOSTIC (non-empty only). Tesseract:
  synth CER 0.066 / GER 0.074, non-empty CER 0.029; REID CER 0.236 / GER 0.308, non-empty 0.208.
  যুক্তাক্ষর = biggest error class (synth 27% of errors, non-empty 38%; error rate 0.097 vs
  consonants 0.048). REID: Bangla digits 0.52, punctuation 0.49, insertions 21% of errors
  (hallucinated ০ . - ␣). Latin digit leakage: REID 125, synth 3 (synth labels have 10 real ones).
- [x] Look-alike decision (Option 1): labels + normalisation unchanged, limitation documented;
  revisit before Phase 7.
- [x] Step 9A/B: EasyOCR 1.7.2 CPU engine (`easyocr_engine.py`), optional `easyocr` dependency group
  (torch 2.14.1+cpu, opencv 5.0 locked). CRAFT + bengali_g1 weights sha256-pinned in `sources.py`
  (md5 = EasyOCR's own config), `download_enabled=False`. Settings in the cache key: langs ["bn"],
  greedy, quantize. Fair layout hint: synth `recognize` (one line, like psm 7), REID `readtext` +
  `reading_order` (written by me: centre-y grouping vs line's first word, > half height = new line,
  min-x sort; 10 tests, 6 mutants caught). Fairness test: precomposed য় (EasyOCR) == য + ় (labels).
- [x] Step 9C: val smoke (100 rows, `7fc4431d6f9f`): 0.455 s/line, peak RAM 790 MB, cache reuse OK.
  Full test run `14e8c29fc5dd`: 2246/2246, 0 errors, no dup ids, same data_sha256 as Tesseract,
  rerun = 2246 cached. 25.9 min compute: synth 0.38 s/line, REID 1.41 s/block (p95 5.7, max 32 s).
- [x] Step 9D: EasyOCR vs Tesseract (paired, cluster CIs; diff = EasyOCR - Tesseract):
  synth CER 0.030 vs 0.066, diff -0.036 [-0.047, -0.025]; heavy 0.037 vs 0.155 (-0.118);
  mild tie; **clean worse** 0.030 vs 0.019 (+0.011 [0.006, 0.016]). REID 0.170 vs 0.236,
  diff -0.066 [-0.126, -0.022]; paragraphs -0.066 clear; page numbers 0.94 vs 0.98 (no clear diff).
  GER: synth 0.049, REID 0.263. WER worse on synth: 0.184 vs 0.150 (REID 0.612 vs 0.594).
  Breakdown: EasyOCR's errors are punctuation, not script. Synth: punctuation 38% of errors
  (। read as | / ! ] [ or dropped; , as ; or )), inserted spaces 23% (220 lines end "word |":
  a space before the mark). Letters-only diagnostic (scratch, punct+space+symbols removed both
  sides): synth 0.016 vs 0.066, clean 0.014 vs 0.019 -> clean deficit = punctuation only.
  যুক্তাক্ষর error rate 0.023 (Tesseract 0.097). REID: 124 empty (98 page numbers, only 463
  chars), Latin digits 207 (। -> 1 56x), Bangla digits 0.65 error rate. Model sometimes emits
  precomposed য় + extra ় (double nukta, 11x synth): real model output, scored as is.
- [x] Step 10 plan (Gemini): `gemini-3.8-flash`, free tier (Google may use inputs: data is public,
  CC BY-SA / PD; note possible future contamination). Pinned model id + response model_version,
  fixed prompt (hash in config), default settings, raw responses + tokens + finish reason cached.
  RECITATION/SAFETY = real output (scored empty); only 429/5xx/network retried. Compare all 3
  engines on the same sampled ids only; report cost + latency too.
- [x] Step 10A: `sample.py`, `sample_ids` written by me: per source, rank distinct cluster keys
  (synth text_id, REID page) by sha256(f"{seed}:{key}"), keep k smallest, all rows of them.
  Order-independent, Python-version-stable, nested (bigger quota keeps smaller sample).
  SEED 20261012. Row counts: synth 40/60/80 texts -> 200/300/400; REID 15/20/25/30 pages ->
  226/282/312/354. Default QUOTAS 60 + 20 = 582 test calls (+~50 val). Final QUOTAS: set from
  AI Studio limits BEFORE any test call.

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
- Labels contain look-alikes: ৷ U+09F7 (currency numerator) used as danda (synth 10, REID 57) and
  Assamese ৰ U+09F0 for র (REID 130). Tesseract confuses । <-> ৷ (62x). Folding both changes CER
  by 0.0004 synth / 0.0036 REID (inside CIs) -> open decision, not silently normalised.
- EasyOCR 1.7.2 bug: `self.quantize=quantize,` (trailing comma = tuple) -> `quantize=False` is
  ignored for the detector. We use the default True, so unaffected; noted for reproducers.
- EasyOCR bengali_g1 charset = shared digits/symbols (0-9, | ! [ ] ; ...) + Latin a-z + Bangla
  incl. । (no ॥, ৷, ৰ). `langs=["bn"]` blocks Latin letters (0 in output) but not Latin digits or
  symbols, so the danda competes with look-alike | ! ] 1 and loses ~half the time.
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
- Sample whole clusters (the bootstrap's unit), or CIs on the sample are too narrow.

## Habits
- Before commit: `uv run ruff format . ; uv run ruff check . ; uv run pytest`
- Always `open(path, encoding="utf-8")` on Windows.
- Long `python -c` one-liners break in PowerShell (quotes/braces) -> put them in `scripts/`.
- After each commit: `git push`.

## Next
- Phase 5: send AI Studio RPM/RPD for gemini-3.8-flash -> fix QUOTAS -> write data/samples/test.json.
  10B: gemini engine + `ids` subset in predict.run (old cache keys must not change) + mocked
  tests. 10C: val smoke (~20 rows) + nondeterminism check (~30 val rows twice). 10D: sample run.
  10E: 3-way paired comparison on the sample + hallucination check. Then the Phase 5 quiz.
- Before Phase 7: revisit look-alikes (৷/।, ৰ/র); consider a letters-only CER column in `score`.
