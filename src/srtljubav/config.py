"""All tunables in one place, as picklable frozen dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

CROATIAN_TRANSLATION_DESCRIPTION = (
    "Source subtitles are Croatian seminar/lecture transcripts, often religious and "
    "theological. Translate faithfully into the requested target language while preserving "
    "all SRT numbering and timestamps exactly. Keep names and references such as Isus, "
    "Bertha Dudde, Agapao, Ivan, Rimljanima, Matej, Psalam, Knjizica/booklet numbers, and "
    "Bible verse references accurate. Use natural subtitle language, but do not summarize, "
    "omit, add commentary, or merge/split subtitle entries."
)

DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_USAGE_LOG = Path("usage_logs/gemini_usage.jsonl")
DEFAULT_RUN_REPORT = Path("usage_logs/run_summary.txt")
DEFAULT_JOBS_CSV = Path("usage_logs/jobs.csv")
DEFAULT_RUNS_CSV = Path("usage_logs/runs.csv")


@dataclass(frozen=True)
class CleaningConfig:
    """Pre-translation cleaning: merge tiny fragments into sentence-ish units.

    Optimized for long Croatian speech-transcript SRTs: keep original sections
    mostly intact, only joining tiny neighboring fragments.
    """

    max_len_steps: tuple[int, ...] = (180, 240)
    min_duration: float = 2.0
    max_dist_forward: float = 0.25
    max_dist_backward: float = 0.05
    max_merged_duration: float = 18.0
    # Window repair. Source transcripts carry two distinct defects and each needs
    # a different truth. Above ``impossible_cps`` no human could have spoken the
    # text in the recorded window, so the transcript mis-assigned the end and the
    # speech really ran on into the silence that follows: restore the window at
    # ``repair_target_cps``, the archive's own median speech rate. A merely brief
    # entry is timed correctly, so it is extended only to ``min_entry_duration``.
    min_entry_duration: float = 1.0
    impossible_cps: float = 40.0
    repair_target_cps: float = 17.0


@dataclass(frozen=True)
class YTConfig:
    """Post-translation reflow into YouTube-friendly entries."""

    max_lines: int = 2
    min_duration: float = 1.0
    max_duration: float = 7.0
    # YouTube rejects entries shorter than this. Reporting threshold only:
    # ``ytsplit.merge_short_entries`` rescues everything under ``min_duration``,
    # so what is left below this line is what no legal merge could reach.
    hard_min_duration: float = 0.5
    # Silence a cross-entry merge may swallow. The merged entry is shown from the
    # START of the earlier window, so merging across a long pause puts text on
    # screen seconds before it is spoken. Measured over the 14 archive sources:
    # the 31 merges the shipped pipeline performs span 0.00-1.88 s of silence,
    # so 2.0 s is the smallest round cap that costs nothing today while refusing
    # everything beyond what the transcripts actually contain. Chunk merges
    # inside one entry are contiguous (gap 0) and never touch this cap.
    max_merge_gap: float = 2.0
    # Line budget for the TRANSLATION path only. A translation is never split,
    # so an entry whose text outgrows two lines has no other rescue: German runs
    # ~20% and Tamil ~43% of entries past the two-line width at 1:1. Adding a
    # line leaves the count, index and timestamp untouched, which splitting
    # would not. The Croatian source keeps ``max_lines`` -- it may create
    # entries, so a third line there would be a regression, not a rescue.
    rewrap_max_lines: int = 3
    default_max_chars: int = 42  # Latin fallback; per-script limits in scripts.py


@dataclass(frozen=True)
class SegmentConfig:
    """Fragment-anchored segmentation of the Croatian source (``segment.py``).

    Every source fragment is an anchor: time is only ever taken from the
    silence beside it, divided within ONE fragment, or shared by fragments
    merged across a pause no longer than a breath. Order matters -- speed is
    fixed on the raw fragments while it is still known where the silences are,
    entries are split before they are merged, and merges never cross a pause.
    Proposed and measured by Andrija Dujaković (2026-09-30) over 33 lectures.
    """

    # Comfortable reading speed: the archive's own median speech rate.
    target_cps: float = 17.0
    # Too-fast fragments are fixed worst first, one tier at a time, so the
    # silence around the worst offenders is spent before milder ones claim it.
    speed_tiers: tuple[float, ...] = (100.0, 50.0, 30.0, 22.0, 17.0)
    # A merge may never produce an entry faster than this.
    merge_ok_cps: float = 20.0
    max_line_chars: int = 42
    max_lines: int = 2
    min_duration: float = 1.0
    max_duration: float = 7.0
    # Sentence merges join only fragments separated by no real pause.
    sentence_merge_gap: float = 0.2
    # A fragment no silence can rescue may merge with a neighbour across this.
    rescue_merge_gap: float = 0.5
    # Two frames at 25 fps: the eye needs a blank between consecutive entries.
    min_gap: float = 0.08
    # How far a start may be pulled back into the preceding silence. Text that
    # appears before it is spoken desynchronises; a quarter second of lead-in is
    # the common subtitling allowance, anything more is reading ahead.
    max_lead_in: float = 0.25


@dataclass(frozen=True)
class TranslationConfig:
    """Picklable per-job configuration handed to ProcessPoolExecutor workers."""

    model: str = DEFAULT_MODEL
    batch_size: int = 300
    streaming: bool = False
    thinking: bool = False
    thinking_budget: int = 0
    thinking_level: str = "minimal"
    temperature: float = 0.2
    description: str = CROATIAN_TRANSLATION_DESCRIPTION
    free_quota: bool = True
    quiet_gst: bool = True
    progress_log: bool = False
    pricing_mode: str = "standard"
    # Gemini service tier. "flex" bills at half the standard rate in exchange for
    # higher latency, which this batch workload does not care about. None leaves
    # the field off the request so Google applies the account default.
    service_tier: str | None = None
    clean_sections: bool = True
    overwrite: bool = False
    limit_lines: int | None = None
    yt_split: bool = True
    # The expensive ladder: every rung is a FULL re-translation (~62k in / 37k
    # out tokens). ``pipeline.repair_failed_entries`` now re-translates just the
    # entries validation rejected, so the ladder is only there for a genuine
    # gemini-srt-translator collapse -- worst case 2 full translations, not 6.
    translate_attempts: int = 1
    fallback_batch_sizes: tuple[int, ...] = (20,)
    # gst abandons the WHOLE language after this many consecutive rejected
    # batches (its own default is 3). One more attempt costs one batch (~1/8
    # of a lecture); giving up costs a full re-translation on the next rung.
    max_consecutive_errors: int = 6
    # Second pass over the jobs that failed: batch sizes (first = primary, rest
    # = fallbacks) and tier. On 2026-09-23 14 of 266 jobs failed, all 14
    # succeeded on a smaller-batch re-run, and the last one only on the
    # standard tier while flex kept answering 503. Empty disables the pass.
    retry_batch_sizes: tuple[int, ...] = (50, 10)
    retry_service_tier: str | None = "standard"
    # Content validation of the raw translator output (before the output
    # rewrap): entries the model silently skips stay Croatian, so
    # identical-to-source entries beyond these limits fail the job instead of
    # reaching YouTube.
    # One or two isolated source echoes are tolerated; three consecutive
    # nontrivial entries indicate a skipped translation block.
    max_identical_streak: int = 3
    max_identical_ratio: float = 0.50
    usage_log: Path | None = DEFAULT_USAGE_LOG
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    yt: YTConfig = field(default_factory=YTConfig)
    # How the Croatian source is segmented: "anchor" (``segment.py``, fragment
    # timing preserved) or "legacy" (clean-merge blocks, then ``ytsplit.reflow``
    # -- kept for the archive repair script, whose shipped translations are
    # aligned to it, and for A/B measurement).
    segmenter: str = "anchor"
    segment: SegmentConfig = field(default_factory=SegmentConfig)
