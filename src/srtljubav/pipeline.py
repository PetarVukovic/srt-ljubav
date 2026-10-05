"""Orchestration: per-job translation worker, in-place reformat, parallel runner."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import srt
from gemini_srt_translator.logger import set_quiet_mode

from srtljubav import cleaning, scripts, segment, srtio, validation, ytsplit
from srtljubav.config import DEFAULT_JOBS_CSV, DEFAULT_RUNS_CSV, TranslationConfig, YTConfig
from srtljubav.discovery import TranslationJob, build_jobs, discover_source_srts
from srtljubav.languages import (
    SOURCE_LANGUAGE_SUFFIX,
    expected_script,
    language_dir_name,
    translation_suffixes,
)
from srtljubav.pricing import UsageSummary, format_duration
from srtljubav.usage import append_job_csv, append_run_csv, write_run_report, write_usage_log
from srtljubav.validation import ReadabilityError, ValidationReport

if TYPE_CHECKING:  # the translator pulls in the API deps; keep them lazy at runtime
    from srtljubav.translator import CachedGeminiSRTTranslator


@dataclass(frozen=True)
class JobResult:
    language: str
    output_file: str
    usage: UsageSummary
    elapsed_seconds: float = 0.0
    # Readability verdicts (layers 2-4) for the raw translator output. Only
    # warnings can reach here: a failing entry raises inside the worker.
    validation: ValidationReport = field(default_factory=ValidationReport)


class JobFailedError(RuntimeError):
    """A failed job plus the tokens it spent, sent from the worker to the runner.

    Without it the runner logged every failure at 0 tokens and the ledger
    under-counted the real bill. Must stay picklable across the process pool,
    hence ``__reduce__``. ``retryable`` is False for readability failures:
    per-entry repair has already spent its calls on them.
    """

    def __init__(self, message: str, usage: UsageSummary, retryable: bool) -> None:
        super().__init__(message)
        self.usage = usage
        self.retryable = retryable

    def __reduce__(self) -> tuple[type[JobFailedError], tuple[str, UsageSummary, bool]]:
        return (JobFailedError, (str(self), self.usage, self.retryable))


@dataclass
class ReformatResult:
    path: Path
    entries_before: int
    entries_after: int
    changed: bool
    # Entries too long for cfg.max_lines (reported, never split) and how the
    # file's structure differs from its Croatian sibling.
    overflow_indices: list[int] = field(default_factory=list)
    croatian: Path | None = None
    croatian_mismatch: str | None = None


@dataclass
class RunResult:
    job_count: int
    failures: list[tuple[TranslationJob, str]] = field(default_factory=list)
    total_usage: UsageSummary = field(default_factory=UsageSummary)
    elapsed_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.failures


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def validate_output_file(source_file: Path, output_file: Path) -> None:
    """The translator must preserve the 1:1 entry mapping exactly."""
    source_count = len(srtio.parse(srtio.read_srt_text(source_file)))
    output_count = len(srtio.parse(srtio.read_srt_text(output_file)))
    if source_count != output_count:
        raise RuntimeError(
            f"Output validation failed: expected {source_count} SRT entries, got {output_count}"
        )


# Ratio check needs a sample; tiny --limit-lines test runs rely on the streak.
_MIN_ENTRIES_FOR_RATIO = 20


def _normalized(text: str) -> str:
    return " ".join(text.split()).lower()


def _is_nontrivial(norm: str) -> bool:
    """Entries worth comparing: verse refs, numbers and interjections stay
    identical in a correct translation and must not count as leakage."""
    return len(norm) >= 12 and sum(c.isalpha() for c in norm) >= 8


def validate_translated_content(
    source_file: Path, output_file: Path, target_language: str, cfg: TranslationConfig
) -> None:
    """Catch silently-untranslated output BEFORE the YT reflow.

    gemini-srt-translator keeps the Croatian original for any entry the model
    skips and only checks the returned line COUNT, so whole batches can stay
    Croatian while the job reports success. Compares entries positionally
    (counts already validated 1:1) and checks the dominant Unicode script for
    non-Latin targets.
    """
    if target_language == "Croatian":
        return

    output_subs = srtio.parse(srtio.read_srt_text(output_file))
    expected = expected_script(target_language)
    if expected is not None:
        detected = scripts.detect_script(" ".join(sub.content for sub in output_subs))
        if detected != expected:
            raise RuntimeError(
                f"Output validation failed for {target_language}: dominant script is "
                f"{detected.value}, expected {expected.value}"
            )

    source_subs = srtio.parse(srtio.read_srt_text(source_file))
    nontrivial = identical = streak = longest = 0
    for src, out in zip(source_subs, output_subs, strict=True):
        src_norm = _normalized(src.content)
        if not _is_nontrivial(src_norm):
            continue
        nontrivial += 1
        if src_norm == _normalized(out.content):
            identical += 1
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 0

    if nontrivial >= 3 and identical == nontrivial:
        raise RuntimeError(
            f"Output validation failed for {target_language}: all {nontrivial} "
            "nontrivial entries were left untranslated"
        )
    if longest >= cfg.max_identical_streak:
        raise RuntimeError(
            f"Output validation failed for {target_language}: {longest} consecutive "
            "entries left untranslated (identical to the Croatian source)"
        )
    if (
        nontrivial >= _MIN_ENTRIES_FOR_RATIO
        and identical / nontrivial >= cfg.max_identical_ratio
    ):
        raise RuntimeError(
            f"Output validation failed for {target_language}: {identical}/{nontrivial} "
            "entries identical to the Croatian source"
        )


def validate_readability(source_file: Path, output_file: Path) -> ValidationReport:
    """Layers 2-4 on the raw translator output, still 1:1 with the source.

    Must run BEFORE the output rewrap: the thresholds are relative to the
    aligned Croatian entry, which only exists while the mapping is 1:1. The
    returned failing indices are the per-entry repair worklist.
    """
    return validation.validate_entries(
        srtio.parse(srtio.read_srt_text(source_file)),
        srtio.parse(srtio.read_srt_text(output_file)),
    )


# Per-entry repair (design section D). Re-translating a lecture costs ~62k input
# / 37k output tokens, and the real Western Punjabi failure was ONE bad entry out
# of 460, so only the rejected entries are re-translated. Above the ceiling the
# translation is bad as a whole and no repair is worth its call.
REPAIR_MAX_ENTRIES = 20
# The repair pass itself gets one retry; after that the job fails.
REPAIR_ATTEMPTS = 2
REPAIR_BATCH_SIZE = 1


def _new_translator(
    base_kwargs: dict,
    *,
    input_file: Path,
    output_file: Path,
    batch_size: int,
) -> CachedGeminiSRTTranslator:
    """The single place that builds the gst translator.

    Repair must inherit exactly the prompt, model, tier and keys the full pass
    used -- only the file pair and the batch size differ.
    """
    # Imported lazily so the module stays importable without API deps configured.
    from srtljubav.translator import CachedGeminiSRTTranslator

    return CachedGeminiSRTTranslator(
        input_file=str(input_file),
        output_file=str(output_file),
        batch_size=batch_size,
        **base_kwargs,
    )


def entries_by_index(subs: list[srt.Subtitle], indices: tuple[int, ...]) -> list[srt.Subtitle]:
    """The entries carrying *indices*, in file order."""
    wanted = set(indices)
    return [sub for sub in subs if sub.index in wanted]


def splice_contents(
    subs: list[srt.Subtitle], indices: tuple[int, ...], contents: list[str]
) -> list[srt.Subtitle]:
    """Replace the TEXT of *indices* with *contents*, positionally.

    Timestamps, the entry count and every other entry are untouched: a repair
    may only ever change the words of the entries it was asked to redo.
    """
    replacement = dict(zip(indices, contents, strict=True))
    return [
        srt.Subtitle(
            index=sub.index,
            start=sub.start,
            end=sub.end,
            content=replacement.get(sub.index, sub.content),
        )
        for sub in subs
    ]


def repair_echoed_input(
    mini: list[srt.Subtitle], repaired: list[srt.Subtitle]
) -> bool:
    """True when a repair pass handed its Croatian input straight back.

    gemini-srt-translator initialises its output as a COPY of the input and only
    checks the returned line count, so a skipped batch comes back verbatim and
    reads as a successful answer. Trivial entries (numbers, verse references)
    stay identical in a correct translation, so they only count when the
    mini-SRT holds nothing else.
    """
    pairs = [
        (source.content, entry.content)
        for source, entry in zip(mini, repaired, strict=True)
    ]
    comparable = [
        pair for pair in pairs if _is_nontrivial(_normalized(pair[0]))
    ] or pairs
    return all(_normalized(before) == _normalized(after) for before, after in comparable)


def repair_failed_entries(
    *,
    source_file: Path,
    output_file: Path,
    report: ValidationReport,
    target_language: str,
    cfg: TranslationConfig,
    base_kwargs: dict,
    temp_dir: Path,
    usage_records: list[dict],
) -> ValidationReport:
    """Re-translate only the entries readability validation rejected.

    A mini-SRT of just those entries goes back to the same model with the same
    prompt at ``batch_size=1``; the returned text is spliced into the output at
    the ORIGINAL indices and the whole file goes through the FULL validation
    stack again -- content (Croatian echo, expected script) as well as
    readability, because a repair can break anything the first pass had to
    clear. The mini-SRT is composed with its own 1..N numbering, so the mapping
    back is positional.

    Raises:
        ReadabilityError: when more than ``REPAIR_MAX_ENTRIES`` entries fail
            (the translation is bad as a whole, so nothing is retried) or when
            ``REPAIR_ATTEMPTS`` repairs still leave the file failing. Always
            this type, never the content check's ``RuntimeError``: the caller
            must not answer a bad repair with another full re-translation. The
            caller must delete the output file: discovery treats any existing
            file as done and would skip the language forever.
    """
    detail: str | None = None
    for attempt in range(1, REPAIR_ATTEMPTS + 1):
        indices = report.failed_indices
        if len(indices) > REPAIR_MAX_ENTRIES:
            break
        source_subs = srtio.parse(srtio.read_srt_text(source_file))
        mini = entries_by_index(source_subs, indices)
        if len(mini) != len(indices):
            # A reported index the source does not carry: there is nothing to
            # re-translate against, so the file is failed rather than guessed at.
            break
        mini_input = temp_dir / f"repair_{attempt}_in.srt"
        mini_output = temp_dir / f"repair_{attempt}_out.srt"
        mini_input.write_text(srtio.compose(mini), encoding="utf-8")

        translator = _new_translator(
            base_kwargs,
            input_file=mini_input,
            output_file=mini_output,
            batch_size=REPAIR_BATCH_SIZE,
        )
        try:
            translator.translate()
        finally:
            usage_records.extend(translator.usage_records)

        repaired = srtio.parse(srtio.read_srt_text(mini_output))
        if len(repaired) != len(mini):
            # The repair pass lost the 1:1 mapping; never splice a misaligned
            # answer -- spend the second attempt instead.
            detail = (
                f"the repair returned {len(repaired)} entries for {len(mini)} "
                "requested, so nothing could be spliced"
            )
            continue
        if repair_echoed_input(mini, repaired):
            # Not an answer: splicing it would ship the Croatian source text.
            detail = "the repair returned its Croatian input unchanged"
            continue
        output_subs = srtio.parse(srtio.read_srt_text(output_file))
        spliced = splice_contents(
            output_subs, indices, [entry.content for entry in repaired]
        )
        output_file.write_text(srtio.compose(spliced), encoding="utf-8")

        try:
            validate_translated_content(source_file, output_file, target_language, cfg)
        except RuntimeError as exc:
            detail = str(exc)
            continue
        report = validate_readability(source_file, output_file)
        if report.ok:
            return report
        detail = None

    raise ReadabilityError(target_language, report, detail)


def _candidate_batch_sizes(cfg: TranslationConfig) -> tuple[int, ...]:
    sizes: list[int] = []
    for size in (cfg.batch_size, *cfg.fallback_batch_sizes):
        if size < 1:
            continue
        if size > cfg.batch_size:
            continue
        if size not in sizes:
            sizes.append(size)
    return tuple(sizes)


def prepare_source_text(source: Path, cfg: TranslationConfig) -> str:
    """Derive the Croatian upload text -- the segmentation every translation keeps."""
    content = srtio.read_srt_text(source)
    if cfg.yt_split and cfg.segmenter == "anchor":
        subs = srtio.limit_subs(srtio.parse(content), cfg.limit_lines)
        return srtio.compose(segment.segment(subs, cfg.segment))
    if cfg.segmenter not in ("anchor", "legacy"):
        raise ValueError(f"unknown segmenter: {cfg.segmenter!r}")
    anchors_by_index = None
    if cfg.clean_sections:
        subs = srtio.limit_subs(srtio.parse(content), cfg.limit_lines)
        content, anchors_by_index, _report = cleaning.clean_source_with_anchors(
            srtio.compose(subs), cfg.cleaning
        )
        # Repair under-timed windows before the reflow: once a 420-character
        # entry is split into seven chunks, the fact that they share one
        # impossible window is gone and each chunk merely looks short. Only
        # ends move, so the cleaning anchors stay valid.
        content = srtio.compose(
            cleaning.repair_windows(srtio.parse(content), cfg.cleaning)
        )
    elif cfg.limit_lines:
        subs = srtio.limit_subs(srtio.parse(content), cfg.limit_lines)
        content = srtio.compose(subs)

    if cfg.yt_split:
        if anchors_by_index is None:
            content = ytsplit.reflow_text(content, cfg.yt)
        else:
            content = ytsplit.reflow_text_with_anchors(content, cfg.yt, anchors_by_index)
        # Repair again: the anchored reflow pins chunk starts back onto the
        # original fragment starts, which reintroduces exactly the under-timed
        # windows the first pass fixed -- measured 16 unreadable entries across
        # the archive that plain proportional reflow never produces.
        content = srtio.compose(
            cleaning.repair_windows(srtio.parse(content), cfg.cleaning)
        )
        # Whatever no silence could rescue is merged into a neighbour, so the
        # upload carries as little as possible under the comfortable reading floor.
        content = srtio.compose(
            ytsplit.merge_short_entries(srtio.parse(content), cfg.yt)
        )
    return content


def write_croatian_upload_file(source: Path, cfg: TranslationConfig) -> Path:
    """Derive ``{stem}_Croatian.srt`` from *source*, unconditionally.

    The deliberate single-file write, used by the archive repair script and the
    tests. A translation run goes through ``write_croatian_upload_files``, which
    refuses to re-segment a file that shipped translations are aligned to.
    """
    output = source.with_name(f"{source.stem}{SOURCE_LANGUAGE_SUFFIX}.srt")
    output.write_text(prepare_source_text(source, cfg), encoding="utf-8")
    return output


@dataclass(frozen=True)
class CroatianUploadPlan:
    """What a run may do to one source's ``_Croatian.srt``, and why."""

    source: Path
    path: Path
    write: bool
    reason: str | None = None
    # Shipped translations this run does NOT re-translate, which a changed
    # segmentation would silently desynchronise.
    at_risk: tuple[Path, ...] = ()
    # The text a writing plan will store; carried so the source is prepared once.
    prepared_text: str | None = None


def shipped_translations(source: Path, languages_in_run: Sequence[str]) -> tuple[Path, ...]:
    """Existing translations of *source* that this run will NOT regenerate."""
    redone = {f"_{language_dir_name(language)}" for language in languages_in_run}
    keep = translation_suffixes() - {SOURCE_LANGUAGE_SUFFIX} - redone
    return tuple(
        path
        for path in sorted(source.parent.glob(f"{source.stem}_*.srt"))
        if any(path.stem.endswith(suffix) for suffix in keep)
    )


def plan_croatian_upload(
    source: Path, cfg: TranslationConfig, languages_in_run: Sequence[str]
) -> CroatianUploadPlan:
    """Decide whether this run may (re)derive *source*'s ``_Croatian.srt``.

    The Croatian file is the one file whose segmentation is ever computed, so
    every translation beside it is aligned to the version on disk. Re-deriving
    it is therefore only free when nothing is aligned to it yet, when the result
    is structurally identical, or when this run re-translates everything that
    is. Otherwise the rewrite is refused and reported: re-shaping an archive
    that has already shipped is ``scripts/repair_archive.py``'s explicit job.
    """
    path = source.with_name(f"{source.stem}{SOURCE_LANGUAGE_SUFFIX}.srt")
    if not languages_in_run:
        return CroatianUploadPlan(
            source, path, write=False, reason="no translation job for this source"
        )

    prepared = prepare_source_text(source, cfg)
    if not path.exists():
        return CroatianUploadPlan(source, path, write=True, prepared_text=prepared)

    existing = srtio.read_srt_text(path)
    if prepared == existing:
        return CroatianUploadPlan(source, path, write=False, reason="already up to date")

    mismatch = structure_mismatch(srtio.parse(prepared), srtio.parse(existing))
    if mismatch is None:
        return CroatianUploadPlan(source, path, write=True, prepared_text=prepared)

    at_risk = shipped_translations(source, languages_in_run)
    if at_risk:
        return CroatianUploadPlan(
            source,
            path,
            write=False,
            reason=(
                f"regenerating it would re-segment the file ({mismatch}) that "
                f"{len(at_risk)} shipped translation(s) are aligned to; re-shape the "
                "archive with scripts/repair_archive.py instead"
            ),
            at_risk=at_risk,
        )
    return CroatianUploadPlan(source, path, write=True, prepared_text=prepared)


def write_croatian_upload_files(
    sources: Sequence[Path], jobs: Sequence[TranslationJob], cfg: TranslationConfig
) -> list[CroatianUploadPlan]:
    """Write the Croatian upload copy of every source this run translates.

    Never a drive-by side effect: a source with no job in the run is left alone
    entirely, and a rewrite that would desynchronise shipped translations is
    refused rather than performed silently.
    """
    languages_by_source: dict[Path, list[str]] = {}
    for job in jobs:
        languages_by_source.setdefault(job.source.resolve(), []).append(job.target_language)

    plans: list[CroatianUploadPlan] = []
    for source in sources:
        plan = plan_croatian_upload(
            source, cfg, languages_by_source.get(source.resolve(), [])
        )
        if plan.write and plan.prepared_text is not None:
            plan.path.write_text(plan.prepared_text, encoding="utf-8")
        plans.append(plan)
    return plans


def job_source_text(source: Path, cfg: TranslationConfig) -> str:
    """The Croatian text a translation job must be 1:1 with.

    When ``_Croatian.srt`` exists it IS the shipped segmentation -- the run's
    upload plan has already rewritten it or refused to -- so the job translates
    exactly that file. Re-deriving it here translated a different segmentation
    whenever the plan refused a rewrite: on 2026-09-23 EL6-10 Hindi and Telugu
    came out with 565 entries beside a 567-entry ``_Croatian.srt``.
    """
    croatian = source.with_name(f"{source.stem}{SOURCE_LANGUAGE_SUFFIX}.srt")
    if not croatian.exists():
        return prepare_source_text(source, cfg)
    subs = srtio.limit_subs(srtio.parse(srtio.read_srt_text(croatian)), cfg.limit_lines)
    return srtio.compose(subs)


def _prepare_source(job: TranslationJob, cfg: TranslationConfig, temp_dir: Path) -> Path:
    temp_source = temp_dir / job.source.name
    temp_source.write_text(job_source_text(job.source, cfg), encoding="utf-8")
    return temp_source


def _publish_output(work_output: Path, final_output: Path) -> None:
    """Move a finished translation into place so no partial file is ever visible.

    gst starts its output as a COPY of the Croatian source; written straight to
    ``input_files/`` a killed worker left that copy behind and discovery then
    skipped the language as done. The ``.part`` name matches no ``*.srt`` glob.
    """
    staging = final_output.with_name(f"{final_output.name}.part")
    shutil.copyfile(work_output, staging)
    os.replace(staging, final_output)


def translate_job(
    job: TranslationJob,
    cfg: TranslationConfig,
    key_offset: int = 0,
    usage_sink: list[dict] | None = None,
) -> JobResult:
    """Worker: clean -> source YT-reflow -> translate -> validate -> repair -> rewrap.

    Validation runs against the prepared Croatian source, so Gemini must preserve
    the same YouTube-ready segmentation; the translated output is then only
    re-wrapped for target-language line widths, never re-segmented. Every layer
    -- counts, script and Croatian echo, plus readability 2-4 -- therefore runs
    on the RAW output, while it is still 1:1 with the source. Entries readability
    rejects are re-translated one by one (``repair_failed_entries``) instead of
    the whole lecture; when repair cannot save them the job fails. gst writes
    into the temp dir and only a finished translation is published
    (``_publish_output``), since discovery treats any existing file as done.

    ``key_offset`` rotates which GEMINI_API_KEY* is primary for this job so
    parallel workers spread rate-limit pressure across all available keys;
    the other key stays as the in-job quota-switch fallback.

    Every usage record is appended to ``usage_sink`` when given, so a caller can
    still price the calls of a job that raises (``run_job``).
    """
    job_started_at = time.perf_counter()
    keys = [
        key
        for key in (os.environ.get("GEMINI_API_KEY"), os.environ.get("GEMINI_API_KEY2"))
        if key
    ]
    if not keys:
        raise RuntimeError("Missing GEMINI_API_KEY in environment or .env")
    rotation = key_offset % len(keys)
    keys = keys[rotation:] + keys[:rotation]
    api_key = keys[0]
    api_key2 = keys[1] if len(keys) > 1 else None

    job.output_file.parent.mkdir(parents=True, exist_ok=True)
    if cfg.overwrite and job.output_file.exists():
        job.output_file.unlink()

    set_quiet_mode(cfg.quiet_gst)

    with tempfile.TemporaryDirectory(prefix="srt_translate_") as temp_dir:
        temp_source = _prepare_source(job, cfg, Path(temp_dir))
        work_output = Path(temp_dir) / "output" / job.output_file.name
        work_output.parent.mkdir()

        thinking = cfg.thinking
        thinking_level = cfg.thinking_level
        if "gemini-3" in cfg.model:
            thinking = True
            thinking_level = thinking_level or "minimal"

        description = cfg.description
        target_script = expected_script(job.target_language)
        if target_script is not None:
            description += (
                f" Write the translation in the {target_script.value} script "
                "(the target language's native script), never in Latin transliteration."
            )

        base_kwargs = {
            "gemini_api_key": api_key,
            "gemini_api_key2": api_key2,
            "target_language": job.target_language,
            "model_name": cfg.model,
            "streaming": cfg.streaming,
            "thinking": thinking,
            "thinking_budget": cfg.thinking_budget,
            "thinking_level": thinking_level,
            "service_tier": cfg.service_tier,
            "temperature": cfg.temperature,
            "description": description,
            "free_quota": cfg.free_quota,
            "use_colors": False,
            "progress_log": cfg.progress_log,
            "thoughts_log": False,
            "resume": False,
            "max_consecutive_errors": cfg.max_consecutive_errors,
        }

        attempts = max(1, cfg.translate_attempts)
        usage_records: list[dict] = usage_sink if usage_sink is not None else []
        validation_report = ValidationReport()
        batch_sizes = _candidate_batch_sizes(cfg)
        translated = False
        for batch_index, batch_size in enumerate(batch_sizes):
            for attempt in range(1, attempts + 1):
                translator = _new_translator(
                    base_kwargs,
                    input_file=temp_source,
                    output_file=work_output,
                    batch_size=batch_size,
                )
                try:
                    translator.translate()
                    validate_output_file(temp_source, work_output)
                    validate_translated_content(temp_source, work_output, job.target_language, cfg)
                    validation_report = validate_readability(temp_source, work_output)
                    if not validation_report.ok:
                        # Repair the rejected entries instead of re-translating
                        # the lecture; raises ReadabilityError when the whole
                        # translation is bad or the repair does not hold.
                        validation_report = repair_failed_entries(
                            source_file=temp_source,
                            output_file=work_output,
                            report=validation_report,
                            target_language=job.target_language,
                            cfg=cfg,
                            base_kwargs=base_kwargs,
                            temp_dir=Path(temp_dir),
                            usage_records=usage_records,
                        )
                except (Exception, SystemExit) as exc:
                    if isinstance(exc, SystemExit) and exc.code != 130:
                        raise
                    usage_records.extend(translator.usage_records)
                    # The next rung must not start from this attempt's output.
                    work_output.unlink(missing_ok=True)
                    if isinstance(exc, ReadabilityError):
                        # Repair has already spent its calls on the individual
                        # entries; re-translating the lecture would buy the same
                        # defect for ~62k input / 37k output tokens.
                        raise
                    has_smaller_batch = batch_index < len(batch_sizes) - 1
                    if isinstance(exc, SystemExit) and has_smaller_batch:
                        if cfg.progress_log:
                            next_size = batch_sizes[batch_index + 1]
                            print(
                                f"{job.target_language}: gst exhausted batch "
                                f"{batch_size}; retrying with smaller batch size "
                                f"{next_size}",
                                file=sys.stderr,
                            )
                        break
                    if batch_index == len(batch_sizes) - 1 and attempt == attempts:
                        raise
                    if attempt == attempts and cfg.progress_log:
                        next_size = batch_sizes[batch_index + 1]
                        print(
                            f"{job.target_language}: retrying with smaller "
                            f"batch size {next_size}",
                            file=sys.stderr,
                        )
                    continue
                else:
                    usage_records.extend(translator.usage_records)
                    translated = True
                    break
            if translated:
                break

        if cfg.yt_split:
            # Only the LINES are redone, at the target script's width: the
            # prepared Croatian source is the one file whose segmentation is
            # ever computed, so the translation keeps its entry count, indices
            # and timestamps and the two files diff side by side.
            rewrapped = ytsplit.rewrap_text(srtio.read_srt_text(work_output), cfg.yt)
            work_output.write_text(rewrapped, encoding="utf-8")

        _publish_output(work_output, job.output_file)
        usage_summary = write_usage_log(
            cfg.usage_log,
            job=job,
            records=usage_records,
            model_name=cfg.model,
            pricing_mode=cfg.pricing_mode,
            service_tier=cfg.service_tier,
        )

    return JobResult(
        job.target_language,
        str(job.output_file),
        usage_summary,
        elapsed_seconds=time.perf_counter() - job_started_at,
        validation=validation_report,
    )


def run_job(job: TranslationJob, cfg: TranslationConfig, key_offset: int = 0) -> JobResult:
    """Pool entry point: ``translate_job`` whose failures still report their cost.

    The calls a failed job made are written to the usage log here and travel
    to the runner inside ``JobFailedError``, so jobs.csv and the ledger price
    them instead of logging 0 tokens.
    """
    spent: list[dict] = []
    try:
        return translate_job(job, cfg, key_offset, usage_sink=spent)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # noqa: B036 - gst signals give-up with SystemExit(130)
        usage = write_usage_log(
            cfg.usage_log,
            job=job,
            records=spent,
            model_name=cfg.model,
            pricing_mode=cfg.pricing_mode,
            service_tier=cfg.service_tier,
        )
        message = str(exc) or type(exc).__name__
        retryable = not isinstance(exc, ReadabilityError)
        raise JobFailedError(message, usage, retryable) from exc


def retry_config(cfg: TranslationConfig) -> TranslationConfig | None:
    """The config for the second pass over failed jobs, or None when disabled."""
    if not cfg.retry_batch_sizes:
        return None
    first, *rest = cfg.retry_batch_sizes
    return replace(
        cfg,
        batch_size=min(first, cfg.batch_size),
        fallback_batch_sizes=tuple(rest),
        service_tier=cfg.retry_service_tier or cfg.service_tier,
    )


def croatian_sibling(path: Path) -> Path | None:
    """The ``_Croatian.srt`` whose structure a translated file must match."""
    suffixes = sorted(
        translation_suffixes() - {SOURCE_LANGUAGE_SUFFIX}, key=len, reverse=True
    )
    for suffix in suffixes:
        if path.stem.endswith(suffix):
            stem = path.stem[: -len(suffix)]
            sibling = path.with_name(f"{stem}{SOURCE_LANGUAGE_SUFFIX}.srt")
            return sibling if sibling.exists() else None
    return None


def structure_mismatch(
    subs: list[srt.Subtitle], croatian_subs: list[srt.Subtitle]
) -> str | None:
    """How a translation departs from its Croatian sibling, or None when 1:1."""
    if len(subs) != len(croatian_subs):
        return f"{len(subs)} entries vs {len(croatian_subs)} Croatian"
    off = sum(
        1
        for entry, croatian in zip(subs, croatian_subs, strict=True)
        if (entry.start, entry.end) != (croatian.start, croatian.end)
    )
    return f"{off} entries with different timestamps" if off else None


def reformat_existing(
    paths: list[Path], cfg: YTConfig, *, dry_run: bool = False, rewrap_only: bool = True
) -> list[ReformatResult]:
    """Re-wrap the LINES of existing translated SRT files in place. Idempotent.

    Reformatting a translation only ever re-wraps: the entry count, the indices
    and the timestamps stay 1:1 with the ``_Croatian.srt`` sibling, which is the
    single file whose segmentation is ever computed (``prepare_source_text``).
    Entries that overflow ``cfg.max_lines`` and any drift from that sibling are
    reported, never fixed by splitting.

    ``rewrap_only`` is kept as an accepted no-op for the flag that used to opt
    IN to this behaviour; re-segmenting a translation is not an option here at
    all. Re-deriving a Croatian file's segmentation is a deliberate, reported
    action and lives in ``scripts/repair_archive.py``.
    """
    del rewrap_only  # backwards compatibility only: rewrap is what reformat does

    results: list[ReformatResult] = []
    for path in paths:
        original = srtio.read_srt_text(path)
        before = len(srtio.parse(original))

        subs = ytsplit.rewrap(srtio.parse(original), cfg)
        updated = srtio.compose(subs)
        overflow = ytsplit.rewrap_overflow(subs, cfg)
        croatian = croatian_sibling(path)
        mismatch = (
            structure_mismatch(subs, srtio.parse(srtio.read_srt_text(croatian)))
            if croatian is not None
            else None
        )

        after = len(srtio.parse(updated))
        changed = updated != original
        if changed and not dry_run:
            path.write_text(updated, encoding="utf-8")
        results.append(
            ReformatResult(
                path=path,
                entries_before=before,
                entries_after=after,
                changed=changed,
                overflow_indices=overflow,
                croatian=croatian,
                croatian_mismatch=mismatch,
            )
        )
    return results


def _run_jobs(
    jobs: list[TranslationJob], cfg: TranslationConfig, workers: int
) -> tuple[UsageSummary, list[tuple[TranslationJob, BaseException]]]:
    """Run *jobs* in a process pool; log each to jobs.csv and the console.

    Returns everything spent -- failed jobs included -- and the failures.
    """
    failures: list[tuple[TranslationJob, BaseException]] = []
    total_usage = UsageSummary()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_job = {
            executor.submit(run_job, job, cfg, key_offset): job
            for key_offset, job in enumerate(jobs)
        }
        for index, future in enumerate(as_completed(future_to_job), start=1):
            job = future_to_job[future]
            try:
                result = future.result()
                total_usage = total_usage + result.usage
                append_job_csv(
                    DEFAULT_JOBS_CSV,
                    source=str(job.source),
                    language=result.language,
                    output_file=result.output_file,
                    elapsed_seconds=result.elapsed_seconds,
                    usage=result.usage,
                )
                print(
                    f"[{index}/{len(jobs)}] OK {result.language}: {result.output_file} "
                    f"| tokens in/out/total {result.usage.prompt_tokens}/"
                    f"{result.usage.output_tokens}/{result.usage.total_tokens} "
                    f"| est ${result.usage.estimated_cost_usd:.4f} "
                    f"| {format_duration(result.elapsed_seconds)}"
                )
                if result.validation.warnings:
                    print(f"    {result.validation.summary(limit=5)}")
            except BaseException as exc:  # noqa: B036 - worker failures must not kill the run
                failures.append((job, exc))
                spent = exc.usage if isinstance(exc, JobFailedError) else UsageSummary()
                total_usage = total_usage + spent
                # Belt and braces: translate_job only ever publishes a finished
                # file, but a pre-existing partial one must not pass as done.
                job.output_file.unlink(missing_ok=True)
                append_job_csv(
                    DEFAULT_JOBS_CSV,
                    source=str(job.source),
                    language=job.target_language,
                    output_file=str(job.output_file),
                    elapsed_seconds=0.0,
                    usage=spent,
                    status="fail",
                    error=str(exc),
                )
                print(
                    f"[{index}/{len(jobs)}] FAIL {job.target_language}: {job.source} -> {exc}",
                    file=sys.stderr,
                )
    return total_usage, failures


def run_translation(
    cfg: TranslationConfig,
    *,
    input_dir: Path,
    output_root: Path | None,
    target_languages: list[str],
    workers: int,
    max_jobs: int | None,
    run_report: Path | None,
    dry_run: bool = False,
    only_source: Path | None = None,
) -> int:
    started_at = time.perf_counter()
    # Build the full job list first; max_jobs is applied AFTER only_source
    # filtering below so a small cap can't consume another source's jobs and
    # leave the requested --file with zero.
    jobs = build_jobs(
        input_dir,
        overwrite=cfg.overwrite,
        output_root=output_root,
        target_languages=target_languages,
    )
    sources = discover_source_srts(input_dir)
    if only_source is not None:
        target = only_source.resolve()
        jobs = [job for job in jobs if job.source.resolve() == target]
        sources = [source for source in sources if source.resolve() == target]
    if max_jobs is not None:
        jobs = jobs[:max_jobs]

    print(f"Source SRT files: {len(sources)}")
    print(f"Target languages: {len(target_languages)}")
    print(f"Jobs to run: {len(jobs)}")

    def report(total_usage: UsageSummary, failure_count: int) -> None:
        write_run_report(
            run_report,
            model=cfg.model,
            pricing_mode=cfg.pricing_mode,
            service_tier=cfg.service_tier,
            workers=workers,
            usage_log=cfg.usage_log,
            total_usage=total_usage,
            elapsed_seconds=time.perf_counter() - started_at,
            source_count=len(sources),
            target_language_count=len(target_languages),
            job_count=len(jobs),
            failure_count=failure_count,
        )

    if dry_run:
        for job in jobs:
            print(f"[plan] {job.source} -> {job.target_language} -> {job.output_file}")
        return 0

    if cfg.yt_split and cfg.limit_lines is None:
        for plan in write_croatian_upload_files(sources, jobs, cfg):
            if plan.write:
                print(f"Croatian upload SRT: {plan.path}")
            elif plan.at_risk:
                print(f"Croatian upload SRT NOT regenerated: {plan.path}", file=sys.stderr)
                print(f"  {plan.reason}", file=sys.stderr)
            elif plan.path.exists():
                print(f"Croatian upload SRT ({plan.reason}): {plan.path}")

    if not jobs:
        print("Nothing to translate. Use --overwrite to regenerate existing outputs.")
        report(UsageSummary(), failure_count=0)
        return 0

    total_usage, failures = _run_jobs(jobs, cfg, workers)
    second_pass = retry_config(cfg)
    retryable = [job for job, exc in failures if isinstance(exc, JobFailedError) and exc.retryable]
    if second_pass is not None and retryable:
        tier = second_pass.service_tier or "default"
        print(
            f"\nRetrying {len(retryable)} failed job(s): batch sizes "
            f"{(second_pass.batch_size, *second_pass.fallback_batch_sizes)}, "
            f"{tier} tier"
        )
        retry_usage, retry_failures = _run_jobs(retryable, second_pass, workers)
        total_usage = total_usage + retry_usage
        failures = [
            (job, exc) for job, exc in failures if not any(job is r for r in retryable)
        ] + retry_failures

    report(total_usage, failure_count=len(failures))
    append_run_csv(
        DEFAULT_RUNS_CSV,
        model=cfg.model,
        workers=workers,
        jobs=len(jobs),
        failures=len(failures),
        elapsed_seconds=time.perf_counter() - started_at,
        total_usage=total_usage,
    )

    if failures:
        print(f"\nFinished with {len(failures)} failed job(s):", file=sys.stderr)
        for job, exc in failures:
            print(f"- {job.source} -> {job.target_language}: {exc}", file=sys.stderr)
        return 1

    elapsed_seconds = time.perf_counter() - started_at
    print(
        "\nAll translations completed. "
        f"Total tokens in/out/total {total_usage.prompt_tokens}/"
        f"{total_usage.output_tokens}/{total_usage.total_tokens}; "
        f"estimated cost ${total_usage.estimated_cost_usd:.4f}; "
        f"time {format_duration(elapsed_seconds)}."
    )
    if cfg.usage_log:
        print(f"Usage log: {cfg.usage_log}")
    if run_report:
        print(f"Run report: {run_report}")
    return 0
