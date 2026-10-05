# srt-ljubav

Batch SRT subtitle translation pipeline: Croatian religious/theological lecture transcripts → 33 target languages via Gemini API, with YouTube-quality subtitle formatting.

## Project Overview
- Translates source `.srt` files in `input_files/<topic>/` into 33 hardcoded target languages (`TARGET_LANGUAGES` in `src/srtljubav/languages.py`).
- Source content: Croatian seminar/lecture transcripts (religious/theological — Bertha Dudde, Bible references). Translation prompt preserves names, verse refs, and exact SRT numbering/timestamps.
- Pipeline: **segment the Croatian source** (`segment.py`, fragment-anchored: speed fix → split → sentence merge → gap/line breaks; ≤2 lines × 42, ~1–7 s entries) → translate (1:1 mapping) → validate → **rewrap** the translation (lines only). Segmentation is computed on `_Croatian.srt` alone; every `{stem}_{Language}.srt` keeps its entry count, indices and timestamps.
- Stack: Python ≥3.13, `uv`, src-layout package `srtljubav`, `gemini-srt-translator`, `google-genai`, `httpx`, `psycopg[binary]`, `srt`. Parallel via `ProcessPoolExecutor`.
- Output is flat: `{original_stem}_{Language}.srt` next to the source (NOT language subfolders).
- Archive status: fully 1:1 since 2026-08-30 EXCEPT EL6-10 (never repaired; see the 2026-10-03 rules), and Drive still carries the pre-repair copies (`scripts/repair_archive.py --apply` has run) — all 341 shipped translations are aligned to their `_Croatian.srt` sibling; the Postgres mirror (`srt_subtitles`) reflects it.

## Commands
```bash
uv sync                                          # install deps (incl. dev: pytest, ruff)
uv run srt-ljubav translate --file /path/to/x.srt  # USER'S MAIN FLOW: import external SRT + translate ONLY it to 33 langs
uv run srt-ljubav translate --dry-run            # show planned missing translations
uv run srt-ljubav translate                      # translate all missing (3 workers)
uv run srt-ljubav translate --overwrite --languages "Western Punjabi"
uv run srt-ljubav translate --limit-lines 20 --max-jobs 1 --languages English  # cheap test run
uv run srt-ljubav reformat --dry-run             # preview: translations get their LINES re-wrapped only
uv run srt-ljubav reformat                       # apply in place (no API calls, idempotent); reports entries >2 lines + drift from _Croatian.srt
uv run srt-ljubav report                         # totals + per-language time/tokens/cost table
uv run python scripts/compare_segmenters.py      # source / shipped / legacy / anchor readability + timing-truth table
uv run srt-ljubav sync-db                        # mirror archive + per-segment verdicts into Postgres srt_subtitles (run from repo root; DATABASE_URL in .env)
uv run pytest                                    # tests
uv run ruff check                                # lint
```
Key flags: `--model` (default `gemini-3-flash-preview`), `--batch-size` (default 300), `--workers 3`, `--clean-sections` (default on), `--yt-split` (default on), `--segmenter anchor|legacy` (default anchor), `--pricing-mode free`, `--service-tier flex` (half price, higher latency).

## Post-Run Upload (Google Drive)
After a translation run completes, ALWAYS offer the upload flow — and ASK THE USER
where on Google Drive it should go (parent folder link or name) before uploading:
1. Verify cleanliness: `uv run python scripts/scan_untranslated.py` must pass (exit 0).
2. Refresh per-source CSV: `uv run python scripts/per_source_report.py`.
3. Find the EXISTING target folder — NEVER create new folders on Drive. Search
   by lecture name/number via Google Drive MCP (`search_files`) or
   `rclone lsd gdrive: --drive-root-folder-id=<parent>` (convention: parent
   "EVANĐELJE LJUBAVI <N>" → subfolder "<M>. predavanje"). If no matching folder
   is found, ASK the user for the correct folder link — do not create one.
4. Upload the whole topic folder with rclone (NOT the claude.ai connector — too
   slow for ~36 files):
   ```bash
   rclone copy "input_files/<topic>" gdrive: --drive-root-folder-id=<FOLDER_ID> --progress
   rclone lsf gdrive: --drive-root-folder-id=<FOLDER_ID> | wc -l   # expect 36
   ```
   Remote `gdrive` = <mission Google account> (mission account), config in
   `~/.config/rclone/rclone.conf`. Re-auth if needed: `rclone config reconnect gdrive:`.

## Architecture
### Tech Stack
Python 3.13 · uv · hatchling (src layout) · gemini-srt-translator ≥3.8.3,<3.9 · google-genai ≥2.19 · httpx ≥0.28 · psycopg[binary] ≥3.2 (sync-db only) · srt ≥3.5.3 · pytest · ruff

### Key Directories
| Path | Purpose |
|------|---------|
| `src/srtljubav/config.py` | ALL tunables: CleaningConfig, YTConfig, TranslationConfig (picklable), prompt |
| `src/srtljubav/srtio.py` | `srt`-library facade: tolerant parse (pads short millis), compose, reindex |
| `src/srtljubav/cleaning.py` | Pre-translation: timestamp clamping + tiny-fragment merging |
| `src/srtljubav/segment.py` | `segment` — the Croatian source segmentation (default): every word stays in its transcript fragment's time; `SegmentConfig` in `config.py` |
| `src/srtljubav/ytsplit.py` | `reflow` (LEGACY Croatian segmentation, `--segmenter legacy` only) + `rewrap` (translations: lines only, structure frozen); both idempotent |
| `src/srtljubav/scripts.py` | Unicode-script detection → per-script line widths (Latin 42, CJK 18…) |
| `src/srtljubav/validation.py` | Readability layers 2-4 on the raw output: chars/s, repeated n-gram, bidi pairing — thresholds relative to the aligned Croatian entry |
| `src/srtljubav/pipeline.py` | translate_job worker (clean→reflow source→translate→validate→repair→rewrap), reformat, runner |
| `src/srtljubav/cli.py` | Subcommands: `translate` / `reformat` / `report` |
| `src/srtljubav/{languages,discovery,translator,pricing,usage}.py` | Languages, job building, Gemini wrapper, cost tracking |
| `input_files/<topic>/` | Source SRTs + flat translated outputs `{stem}_{Language}.srt` (gitignored) |
| `usage_logs/` | `jobs.csv` (1 row = 1 language: duration/tokens/cost/status), `runs.csv` (append-only run history), `gemini_usage.jsonl` (per API call), `run_summary.txt` (last run) |
| `tests/` | pytest suite (srtio, cleaning, ytsplit, scripts, discovery, pricing, validation, pipeline) |

### Critical Components
- **`discovery.discover_source_srts()`** — treats `input_files/*/*.srt` as sources but MUST filter out files ending in `_{Language}` suffixes; otherwise translated outputs get re-translated as sources.
- **`pipeline.translate_job()` ordering** — 1:1 entry-count validation AND the readability layers (`pipeline.validate_readability` → `validation.validate_entries`) MUST run on the raw translator output BEFORE the output rewrap. Translator runs with `resume=False` in a temp dir.
- **`validation.validate_entries()`** — layers 2 (chars/s), 3 (repeated character n-gram), 4 (bidi pairing). Fails only on the unambiguous, always relative to the aligned Croatian entry; raises `ReadabilityError`, which carries the report so per-entry repair can consume `failed_indices`. Layer 4 only ever warns.
- **`pipeline.repair_failed_entries()`** — ≤ `REPAIR_MAX_ENTRIES` (20) failing entries go back to the same model as a mini-SRT at `batch_size=1` (`_new_translator` builds every translator, so prompt/model/tier/keys are identical), get spliced back at the ORIGINAL indices and the whole file goes through the FULL stack again (content + readability); two attempts, then `ReadabilityError`. Above the ceiling nothing is retried.
- **`pipeline.reformat_existing()` / `srt-ljubav reformat`** — re-wraps LINES only, always. `discover_translated_srts` can never return a `_Croatian.srt`, so a re-segmenting mode would only ever hit translations; `--rewrap-only` survives as a no-op alias.
- **`pipeline.plan_croatian_upload()` / `write_croatian_upload_files()`** — the only path a run may take to `_Croatian.srt`: sources with no job are untouched, and a rewrite that would re-segment a file shipped translations are aligned to is refused and reported (that job is `scripts/repair_archive.py`).
- **`ytsplit.rewrap()`** — the translated-file path: wraps each entry into ≤`max_lines` lines at the script's width and NEVER touches `start`, `end` or the entry count. Text that cannot fit two lines is reported by `rewrap_overflow`, never split into a new entry.
- **`segment.segment()`** — the default Croatian segmentation, from Andrija Dujaković's 2026-09-30 proposal: (1) too-fast fragments fixed worst-first per `speed_tiers` (end into the silence after, start back ≤ `max_lead_in` 0.25 s, then a merge with an adjacent neighbour, then time borrowed from a slow contiguous neighbour); (2) only fragments that do not fit 2×42 (or run far past 7 s) are cut, by a DP over word boundaries, time divided within that fragment; (3) unfinished sentences merge with their continuation only when the pause from the transcript's speech end (`speech_end`, not the stretched end) is ≤ 0.2 s and the result fits 2 lines / 7 s / 20 cps; (4) 0.08 s gaps where an end can spare it, line breaks at punctuation, never after a conjunction/preposition (`fits` is strict about that, so merges/splits never produce one). Text is identical word for word. Deterministic from the source; NOT idempotent on its own output and does not need to be — it always runs on the raw transcript.
- **`pipeline.job_source_text()`** — a translation job translates the `_Croatian.srt` that exists beside the source (after the run's upload plan wrote or refused it), never a freshly derived text; it derives only when no Croatian file exists yet.
- **`ytsplit.reflow()`** — (legacy segmenter) pure + idempotent; splits at sentence > clause > word boundaries, time proportional to chars, contiguous (no overlaps), 10% duration hysteresis so re-runs are no-ops.
- **`CachedGeminiSRTTranslator`** (`translator.py`) — wraps Gemini client via `_get_client()` proxy to record token usage per response; caches model lists/token limits per process.
- **API keys**: `GEMINI_API_KEY` (+ optional `GEMINI_API_KEY2`) from `.env`. Pipeline raises if missing.
- **`LANGUAGE_DIR_OVERRIDES`** — display names like `"Serbian (on Српски!)"` map to clean filename suffixes.

## Project-Specific Rules
- **The Croatian source is segmented by `segment.py` (anchor), not by block-merge + reflow** — the legacy order merged fragments into 180–240 char / 18 s blocks and divided time across each block INCLUDING its pauses, so words appeared seconds before they were spoken (VJK 14 #616: "To znači da postignem savršenstvo." in 1.27 s = 38.6 cps, "Matej 5:48." stretched 7 s over words spoken 4–6 s later). Andrija's order fixes speed on the raw fragments, splits before merging and never merges across a pause. Measured by `scripts/compare_segmenters.py` over his 33-lecture set (35 sources): >20 cps 756 shipped → 91, >7 s 2 461 → 187, <1 s 281 → 6, conjunction ending line 1 3 064 → 0, words shown across a ≥1 s pause before spoken 8 800 → 17; 418 words start after their fragment ended — all on transcript fragments with impossible windows whose speech ran on. His own script's figures on the same set: 148 / 671 / 2 / 49 / 525. Shipped lectures keep their segmentation until re-translated with `translate --overwrite` (all 33 languages, so the plan may rewrite `_Croatian.srt`); `repair_archive.py` pins `segmenter="legacy"` because a regroup is only faithful onto the segmentation the translations were made for (2026-10-03)
- **A job translates the shipped `_Croatian.srt`, never a re-derived text** — `_prepare_source` re-ran `prepare_source_text`, so whenever `plan_croatian_upload` refused to rewrite `_Croatian.srt` (shipped translations aligned to it) the job translated a DIFFERENT segmentation: EL6-10 Hindi/Telugu (2026-09-23) have 565 entries beside a 567-entry Croatian. `job_source_text` reads the file on disk. EL6-10 locally is also wholly pre-repair (July files, 31 languages off 1:1) — it needs a full re-translation (2026-10-03)
- **Google Drive still holds the PRE-2026-08-30 archive for the lectures repaired that day** — e.g. VJK 16 on Drive: `_Croatian` 539 / English 598 / Swahili 602 entries, local 535 everywhere. Andrija's 2026-09-26 list of 24 lectures "not 1:1" is exactly that stale Drive copy, and the "your _Croatian" column of his 2026-09-30 report measures it. Never assume Drive mirrors `input_files/` (2026-10-03)
- **A language must survive transient failures and never be half-published** — on the 2026-09-23 batch 14 of 266 jobs failed: gst counts EVERY exception toward `max_consecutive_errors` (hardcoded 3) before its own 503 pause, so a flex-tier 503 streak or a DNS blip during sleep abandoned the whole language; line-count rejections (Gemini merging two entries, mostly verb-final languages: Korean, Turkish, Tamil, Telugu, Urdu, Punjabi) are intermittent — 8 of 9 isolated probes had zero. Fixes: `UsageTrackingModelsProxy` re-sends 5xx/network errors with backoff (`TRANSIENT_RETRY_DELAYS`; 429 stays with gst so it can switch keys); `TranslationConfig.max_consecutive_errors` = 6; `run_translation` gives retryable failures a second pass at `retry_batch_sizes` (50, 10) on `retry_service_tier` "standard" (flex was the one answering 503); gst writes into the job's temp dir and `_publish_output` moves only a finished file into place — a killed worker used to leave gst's initial COPY of the Croatian source as `{stem}_{Language}.srt`, which discovery then skipped as done. A prompt rule against merging lines was probed and NOT adopted: no measurable effect (2026-09-23)
- **A failed job still reports its cost** — `run_job` wraps `translate_job`, writes the failed calls to `gemini_usage.jsonl` and raises `JobFailedError` (picklable, carries `UsageSummary`) so jobs.csv logs real tokens instead of 0; `scripts/append_ledger.py` sums every jobs.csv row since the run started, because the per-source CSV keeps only the latest row per language (2026-09-23)
- **The archive is 1:1 since 2026-08-30 and must stay that way** — `repair_archive.py --apply` regrouped all 341 translations onto their Croatian segmentation (entries 288 348 → 238 700, >36 cps 4 075 → 1 051, 18 bidi repairs, 11 Croatian upload copies rewritten, >3-line entries 2 881 = 1.2%), and 12 residual file/entry groups were repaired with targeted `repair_failed_entries` calls (never full re-translations). After it, `sync-db` reports 0 files off 1:1 and 0 unchecked segments (244 534 ok / 1 442 warn — all relative warnings). Any future change to a `_Croatian.srt` segmentation goes through `scripts/repair_archive.py`, never as a side effect of `translate`/`reformat` (2026-08-30)
- **A handful of Slovenian entries identical to Croatian are correct translations, not leaks** — short sentences whose valid Slovenian is byte-identical to the Croatian ("To je zakon.", "ona je zaljubljena.", "Timoteju 3:13.", "meni, meni, meni…"; ~14 across 6 lectures). Gemini confirmed the identity on 7 independent attempts, and the anti-echo guards (`repair_echoed_input`, gst's batch content-gap check) reject exactly that answer BY DESIGN — so sending these entries back to Gemini always fails and only burns quota. `scan_untranslated.py` passes on them (isolated entries). Leave them alone (2026-08-30)
- **Targeted probe runs: `translate --file <probe>.srt --input-dir <scratch-dir>`** — a mini-SRT of only the problem segments (keep each entry's original duration, lay them out contiguously) translates to any language list for cents while `input_files/` stays untouched; this validated the WP echo case, the Persian corruption, embedded-English quotes and the 3-line rewrap across 5 languages for $0.0072 before the archive repair was applied (2026-08-30)
- **`reformat` only ever re-wraps lines** — the re-segmenting reflow used to be its default and `--rewrap-only` the opt-in, which meant the documented `uv run srt-ljubav reformat` destroyed the 1:1 structure on every translation it touched (`discover_translated_srts` returns translations only). Reflow now runs on the Croatian source alone, inside `prepare_source_text` (2026-08-30)
- **`_Croatian.srt` is a deliberate output, never a drive-by write** — `run_translation` used to regenerate it for every source before even looking at the job list, so `translate` on a finished archive printed "Nothing to translate" while silently re-segmenting the file all 33 shipped translations are aligned to. `plan_croatian_upload` writes it only for sources this run translates, and refuses (with a report pointing at `scripts/repair_archive.py`) when the new segmentation would desynchronise translations the run does not redo (2026-08-30)
- **A repair must clear the FULL validation stack, and a repair that echoes its input is a failure** — gemini-srt-translator initialises its output as a COPY of the input, so a skipped repair batch comes back as untranslated Croatian with a matching entry count; re-running readability alone accepts it. `repair_failed_entries` rejects the echo by name and re-runs `validate_translated_content` (Croatian echo + expected script) on the spliced file (2026-08-30)
- **`ReadabilityError` must stay picklable** — it crosses the `ProcessPoolExecutor` boundary; without `__reduce__`, `BaseException.__reduce__` replays only the formatted message, the parent's result reader raises `TypeError` and the WHOLE pool is declared broken, killing every other language in the run. Guarded by a real pool round-trip test (2026-08-30)
- **Readability thresholds must stay RELATIVE to the aligned Croatian entry** — absolute ones flag 1674 archive entries of which 4 are real (measured 2026-08-29 over 295 679 entries); legitimate translations of short Croatian entries reach 5-9x the Croatian rate and the speaker's own "tu tu tu…" gives correct translations repetition runs of 10. Fail = chars/s > 500 AND > 10x the Croatian entry, or a repeated run ≥ 50 AND > 5x the Croatian entry's longest run. Everything else (per-script chars/s ceiling, unpaired bidi controls) only warns. Re-measure the numbers on genuine 1:1 output — the design-doc figures come from a noisy reconstruction (2026-08-29)
- **Repair the failing entries, never re-translate the lecture** — a full pass costs ~62k in / 37k out tokens and the real Western Punjabi failure was ONE bad entry out of 460. `translate_job` hands `ValidationReport.failed_indices` to `repair_failed_entries`; a `ReadabilityError` is re-raised out of the batch-size ladder instead of triggering another full translation, and no output file is published. `translate_attempts` is 1 and `fallback_batch_sizes` is `(20,)` (CLI default `--fallback-batch-sizes 20`) — worst case 2 full translations, not 6 (2026-08-29)
- **A repair may only ever change the WORDS of the entries it was asked to redo** — `splice_contents` copies text into the existing entries, so the count, indices and timestamps of the file are structurally untouched; the mini-SRT is composed with its own 1..N numbering, so the mapping back is positional, never by the index gst returns (2026-08-29)
- **Count-only validation cannot catch untranslated batches** — gemini-srt-translator initializes its output as a COPY of the Croatian source and only checks the returned line count; duplicated indexes can hide a skipped run that silently stays Croatian. `validate_translated_content` (identity streak/ratio + expected-script check) MUST run on the raw translator output before the YT reflow, and a failed job MUST delete its output file or discovery treats it as done (2026-07-04)
- **gemini-srt-translator is pinned `>=3.8.3,<3.9` in `pyproject.toml`** — `CachedGeminiSRTTranslator` overrides four private gst methods (`_process_translated_lines`, `_get_client`, `getmodels`, `_get_token_limit`); all four are byte-identical in 3.5.9 and 3.7.1, but re-verify them (guarded by `tests/test_translator.py`) before any version bump. Since 3.6, gst writes output through **pysubs2** (`pysubs2.load(input_file)` → replace texts → `save()`), so translated timestamps are copied verbatim from the source instead of recomposed — verified 1:1 on a real run (2026-08-24)
- **`--service-tier flex` halves the bill** — same tokens, roughly double the latency, which this overnight batch workload does not care about; the tier is recorded per API call in `gemini_usage.jsonl` so `report` prices history correctly. Repricing the whole archive to date: $114.46 standard → $57.23 flex (2026-08-24)
- **(legacy segmenter) `reflow_text_with_anchors` reintroduces under-timed windows, so `repair_windows` runs BOTH before and after the reflow** — anchors pin chunk starts back onto original fragment starts; plain proportional reflow yields 0 entries above 36 cps where the anchored reflow yields 16. Order in `prepare_source_text`: clean → repair → reflow(anchored) → repair → `merge_short_entries`. Guarded by `test_prepare_source_text_repairs_windows_the_anchored_reflow_reintroduces` (2026-08-24)
- **A cross-entry merge must never swallow more than `YTConfig.max_merge_gap` (2.0 s) of silence** — the merged entry is displayed from the EARLIER window's start, so merging across a long pause puts text on screen before it is spoken. Measured over the 14 archive sources: the 31 merges the pipeline performs span 0.00–1.88 s, so the cap costs 0 merges and lecture 4 is byte-identical. Chunk merges inside one entry tile their parent's window (gap 0) and never hit the cap; `ytsplit.entry_merge_decision` reports which cap refused a merge (2026-08-30)
- **Window repair only ever moves ENDS, never STARTS** — a correct start is in sync with the speech; 83% of under-timed entries have no gap behind them anyway. Above `impossible_cps` (40) the transcript mis-assigned the end and the speech ran into the silence that follows, so restore at `repair_target_cps` (17); below `min_entry_duration` the window is right, so extend only to 1.0 s (2026-08-24)
- **`translate --file <path>` is the user's canonical flow** — one external SRT in, 33 translations out; it must stay a single command that copies into `input_files/<stem>/`, restricts the run to that source, and logs to all CSVs (2026-06-06)
- **Only `_Croatian.srt` gets a segmentation** — translations are rewrapped, never reflowed, so they stay 1:1 with the Croatian sibling and the reviewer can diff them side by side (2026-08-29)
- **A translation that outgrows two lines takes a THIRD line, never a new entry** — `YTConfig.rewrap_max_lines` (3) is the rewrap-path budget; the Croatian source keeps `max_lines` (2) because it may create entries, so a third line there would be a regression rather than a rescue. `wrap_lines` is untouched for the two-line case, so entries that already fit are byte-identical and the archive does not churn; `wrap_lines_balanced` binary-searches the narrowest width that fits the budget. Measured over the regrouped archive: entries that cannot be wrapped legally fell from 39 349 (16.5%) to 2 881 (1.2%) — on `10. KVP` German went 169 → 3 and Tamil 412 → 15. Text that will not fit three lines keeps the two-line layout with an over-wide line and is reported by `rewrap_overflow`; it is never grown further (2026-08-30)
- **Validate 1:1 entry counts BEFORE the output rewrap** — validation compares the raw translator output against the prepared Croatian source; the reflow that changes entry counts by design now runs only on that source (2026-06-06)
- **YT reflow must stay idempotent** — proportional time shares drift when whitespace normalization changes chunk lengths; use the 10% duration hysteresis (`_DURATION_TOLERANCE`), don't tighten it (2026-06-06)
- **The `srt` library misreads short millis** — `00:00:00,1` parses as 1 ms, not 100 ms; always go through `srtio.parse()` which pads first (2026-06-06)
- **Flat output structure is canonical** — translations live as `{stem}_{Language}.srt` beside the source; never recreate language subfolders (2026-05-10)
- **Always exclude `_{Language}`-suffixed files from source discovery** — otherwise outputs get re-translated as new sources (2026-05-10)
- **Never alter SRT numbering or timestamps in translation prompts** — timestamp precision is a hard requirement (2026-05-09)
- **Prefer larger `--batch-size` (300) for translation quality** — more context per Gemini call (2026-05-09)
- **Test runs: use `--limit-lines` + `--max-jobs` + `--test-output-dir`** — never burn quota on full runs while testing (2026-05-02)

## Debug / Reference
- Cost/time data: `usage_logs/jobs.csv` (per language), `usage_logs/runs.csv` (per run, append-only), `usage_logs/gemini_usage.jsonl` (per API call), `usage_logs/run_summary.txt` (last run); aggregate via `uv run srt-ljubav report`
- Measured baseline (1h lecture): ~62k in / ~37k out tokens per language; 33 languages ≈ $4.70 standard / $0 free tier
- Pricing table: `GEMINI_PRICING_USD_PER_1M` in `src/srtljubav/pricing.py` (`--pricing-mode free` for free-tier estimates)
- Cleaning tuning: `CleaningConfig` in `src/srtljubav/config.py`; YT limits: `YTConfig` + `SCRIPT_MAX_CHARS` in `src/srtljubav/scripts.py`
- Pipeline flowchart (Mermaid): README.md "Kako radi"
- Audit translations for Croatian leakage before YT upload: `uv run python scripts/scan_untranslated.py` (exit 1 on leaks)
- Per-source CSV report (`{stem}_translations.csv` beside each source): `uv run python scripts/per_source_report.py`
- Google Drive uploads: rclone remote `gdrive` (<mission Google account>); known parent folder "EVANĐELJE LJUBAVI 6" = ID `<FOLDER_ID>`
- Postgres mirror: DB `srt_subtitles` (separate from the Alembic-managed app DB `srt`; `DATABASE_URL` in `.env`), tables `lecture` / `subtitle_file` / `subtitle_entry` with per-segment status+flags; all DDL is `CREATE TABLE IF NOT EXISTS`, refresh via `uv run srt-ljubav sync-db` from the repo root
- Duplicate topic folders `15.`, `16.`, `17.` are verified copies of their `VRIJEME JE KRAJA!` twins BUT each holds a unique zip and a differing `_translations.csv` (`safe to delete: no`); deletion is Petar's call — `repair_archive.py` prints the exact `rm -rf` commands

### Google Drive: where the source SRTs live
All lecture series sit under ONE root on the `gdrive` remote. **Mind the trailing
spaces in the folder names** — `"SRT PRIJEVODI "`, `"EVANĐELJE LJUBAVI 6 "` and
`"Tumačenja snova "` all end in a space; quote every path and copy it verbatim.

```
gdrive:SRT PRIJEVODI /            <- root, note the trailing space
├── EVANĐELJE LJUBAVI 6 /         <- lectures "1." … "15." (also "5. predavanje", "6. predavanje")
├── Jude/                         <- "JUDE U CRKVI" lectures 1..4
├── Jus talionis/
├── KVP/
├── MANNA/
├── Stvari koje nisu naziva kao da jesu/
├── Tumačenja snova /             <- lectures 1..8
├── Vrijeme je kraja/             <- lectures 1..21
└── ZAKON KAO SEBE SAMOGA/
```

A lecture subfolder is named `"<N>. <title>"` and holds the Croatian source
`.srt` alone until translations are uploaded back into it (a shipped folder holds
36 files: source + `_Croatian` + 33 languages). The source filename carries the
series prefix, not the folder name — `"13. Rasprostranjenost grijeha (BD 4853)"`
holds `VRIJEME JE KRAJA! 13. Rasprostranjenost grijeha (BD 4853).srt`, and
`Jude/3. …` holds `JUDE U CRKVI 3. ….srt`.

List and fetch (read-only; never create folders on Drive):
```bash
rclone lsd "gdrive:SRT PRIJEVODI /Vrijeme je kraja"            # lectures in a series
rclone lsf "gdrive:SRT PRIJEVODI /Vrijeme je kraja/13. Rasprostranjenost grijeha (BD 4853)"
rclone copy "gdrive:SRT PRIJEVODI /<series>/<lecture>" "input_files/<lecture>" --include "*.srt"
```

## Runbook: Translate → Audit → Upload

When Petar gives a new `.srt` file, use this exact flow unless he asks for a test run
only. The commands and their flags are documented above; this is the order and
the expected output at each step.

1. If Petar says the file is in Downloads or asks for the newest lecture file,
   identify it before giving/running the command:
   ```bash
   find "$HOME/Downloads" -maxdepth 1 -type f -iname '*.srt' -print0 | xargs -0 ls -t | head -20
   ```
   For Croatian lecture names, prefer the exact file Petar named or the newest
   matching `EVANĐELJE LJUBAVI <series> - <lecture>...srt`. If he says "not the
   last file", do not use the newest blindly; list candidates and choose the
   intended lecture number.
2. In zsh, filenames with `!` can fail as `zsh: event not found`. Before
   running paths that contain exclamation marks, either run `setopt NO_BANG_HIST`
   in that terminal or escape `!` as `\!`. Quoting alone may not be enough in
   interactive zsh.
3. Translate/import the one file into `input_files/<stem>/`:
   ```bash
   uv run srt-ljubav translate --file "/absolute/path/to/file.srt" --workers 6
   ```
   The run writes the `{stem}_Croatian.srt` upload copy for the sources it actually
   translates. If it prints `Croatian upload SRT NOT regenerated`, the new segmentation
   would desynchronise translations already shipped — do not force it; that case is
   `scripts/repair_archive.py`.
4. If some languages fail with status/error `130`, rerun only failed languages
   with detailed Gemini/gst logging and a smaller batch. Start with batch 20:
   ```bash
   uv run srt-ljubav translate --file "/absolute/path/to/file.srt" \
     --workers 1 \
     --languages "Mandarin Chinese,Japanese,Urdu,Tamil,Western Punjabi,Malayalam" \
     --batch-size 20 \
     --overwrite \
     --no-quiet-gst \
     --progress-log
   ```
   If one language still fails with messages like `Expected 20 lines, got 19`
   or `Gemini has returned an empty translation for line ...`, rerun just that
   language with `--batch-size 10`.
5. Confirm there is nothing left to translate:
   ```bash
   uv run srt-ljubav translate --file "/absolute/path/to/file.srt" --workers 1 --dry-run
   ```
   Expected: `Jobs to run: 0`.
6. Audit before any upload:
   ```bash
   uv run srt-ljubav reformat --dry-run
   uv run python scripts/scan_untranslated.py
   uv run python scripts/per_source_report.py
   find "input_files/<topic>" -maxdepth 1 -type f -name '*.srt' | wc -l
   ```
   `reformat --dry-run` re-wraps nothing; it reports entries over two lines and any
   file that is not 1:1 with its `_Croatian.srt` sibling. Expected for a full lecture:
   `All translations clean`, `33/33 translations`, and usually `35` `.srt` files
   (source + Croatian upload copy + 33 target languages). The uploaded folder normally
   has `36` items because `{stem}_translations.csv` is included too.
7. If `scan_untranslated.py` reports `[LEAK]`, inspect the exact flagged entries
   before rerunning blindly. Scripture-only references like `Luka 10:30-37` may
   be legitimate in some languages; real leftover words such as an untranslated
   Bible book name should be fixed or the affected language rerun with a smaller
   batch. Re-run the scan until it prints `All translations clean.`
   Special case: **Slovenian entries identical to the Croatian are usually correct**
   (HR–SL cognate identity — see the Project-Specific Rule dated 2026-08-30). Do NOT send
   those back to Gemini: the anti-echo guards reject an identical answer by design,
   so the rerun always fails and only burns quota.
8. Upload only after Petar gives the exact existing Drive folder link/ID. Never create
   a folder on Drive; the rclone commands and the remote are in "Post-Run Upload" above. In a Drive
   URL such as `https://drive.google.com/drive/u/4/folders/<FOLDER_ID>`
   the folder ID is the final path segment: `<FOLDER_ID>`.

Important: the claude.ai/Google Drive connector is too slow for these folders; use
`rclone` for uploads.

## Runbook: Archive-Wide Maintenance

Proven order (executed in full on 2026-08-30, when the whole archive was brought 1:1):

1. **Probe first, cheaply** — build a mini-SRT of only the suspect segments and run
   `translate --file <probe>.srt --input-dir <scratch-dir> --languages "..." --service-tier flex`;
   `input_files/` stays untouched and the whole test costs cents. Verify the probe
   output with `validation.validate_entries` and `ytsplit.rewrap_overflow` before
   trusting any archive-wide change.
2. **`scripts/repair_archive.py`** — dry-run, read the per-file table, then `--apply`
   only with Petar's explicit go-ahead (back up `input_files/` first; it is gitignored,
   so there is no VCS net under it).
3. **Targeted Gemini repairs** for the entries the apply run lists under
   "ENTRIES THAT STILL NEED A GEMINI CALL" — drive `pipeline.repair_failed_entries`
   against a COPY of the shipped file and overwrite the original only when the full
   validation stack passes. Expect the Slovenian cognate-identity failures (step 7
   note above): those are not repairable and not broken.
4. **`uv run srt-ljubav sync-db`** (from the repo root) — refresh the Postgres mirror
   and confirm the after-picture: 0 files off 1:1, 0 unchecked segments.
5. Close with the standard audit: `scan_untranslated.py` (exit 0),
   `per_source_report.py`, `uv run pytest`.

Recent proven example (2026-07-09, lecture 6-8): initial `--workers 6` left six
failed languages with `130`; batch-size 20 fixed five of them, and Malayalam
needed batch-size 10 after `Gemini has returned an empty translation for line
685`. Final validation was `Jobs to run: 0`, `All translations clean`, `33/33
translations`, `35` local `.srt` files, then `rclone copy` uploaded to Drive and
`rclone lsf ... | wc -l` returned `36`.

## Workflow Orchestration

### 1. Plan Mode Default

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy

- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop

- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done

- Never mark a task complete without proving it works
- Diff behaviour between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)

- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing

- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First:** Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan:** Check in before starting implementation
3. **Track Progress:** Mark items complete as you go
4. **Explain Changes:** High-level summary at each step
5. **Document Results:** Add review section to `tasks/todo.md`
6. **Capture Lessons:** Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First:** Make every change as simple as possible. Minimal code impact.
- **No Laziness:** Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact:** Only touch what's necessary. No side effects, no new bugs.
