"""Readability validation layers 2-4, run on the raw translator output.

Layers 1 (entry count), 5 (expected script) and 6 (Croatian echo) live in
``pipeline``; this module adds the per-entry checks that catch what a
file-level statistic dilutes away:

    2  chars per second   degeneration, over-long translation
    3  repeated n-gram    loops, including space-less scripts
    4  bidi pairing       broken RTL

Every threshold is **relative to the aligned Croatian entry**, so the input
must already be 1:1 with the prepared source (layer 1) and the reflow must not
have run yet. Absolute thresholds were measured against the archive and produce
222 detections of which 2 are real: legitimate translations of short Croatian
entries routinely reach 5-9x the Croatian rate, and the speaker's own
``"tu tu tu tu ..."`` makes a correct translation repeat one unit 10 times.

Zero API calls; pure functions over parsed subtitles.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import srt

from srtljubav.scripts import Script, detect_script

# Fail level: the unambiguous only. Measured separation between the worst
# legitimate case and the mildest real defect is 59x (chars/s) and 300x
# (repetition), so these sit ~1.7x above the worst false positive.
# TODO: re-measure on genuine 1:1 output — the archive figures behind them come
# from a noisy reconstruction that merges translated entries into Croatian
# windows, which inflates the false-positive counts.
CPS_FAIL = 500.0
CPS_FAIL_RATIO = 10.0
REPEAT_FAIL_RUN = 50
REPEAT_FAIL_RATIO = 5.0

# Warn level: the per-script comfortable reading ceiling. Never blocks a job;
# medians across the archive run at roughly a third of each ceiling.
CPS_WARN_CEILING: dict[Script, float] = {
    Script.LATIN: 36.0,
    Script.CYRILLIC: 36.0,
    Script.OTHER: 36.0,
    Script.ARABIC: 34.0,
    Script.HEBREW: 34.0,
    Script.DEVANAGARI: 34.0,
    Script.BENGALI: 34.0,
    Script.GURMUKHI: 34.0,
    Script.GUJARATI: 34.0,
    Script.TAMIL: 34.0,
    Script.TELUGU: 34.0,
    Script.MALAYALAM: 34.0,
    Script.THAI: 30.0,
    Script.HANGUL: 22.0,
    Script.CJK: 18.0,
}

# Bidi: embeddings (LRE/RLE) and isolates (LRI/RLI/FSI) open, pops close.
# A correct entry balances them; an unbalanced one renders RTL text broken.
BIDI_OPENERS = ("\u202a", "\u202b", "\u2066", "\u2067", "\u2068")
BIDI_CLOSERS = ("\u202c", "\u2069")

# Longest repeated unit still searched for. Degeneration loops on a short unit
# ("می" 500x); a longer looping phrase is caught by layer 2 instead, and the
# scan costs one pass per n.
_MAX_NGRAM = 32

# An entry can be arbitrarily short after the reflow; keep chars/s finite.
_MIN_MEASURABLE_SECONDS = 0.001


@dataclass(frozen=True)
class EntryIssue:
    """One finding about one entry, addressed by its 1-based SRT index."""

    index: int
    layer: int
    kind: str  # "cps" | "repetition" | "bidi"
    level: str  # "fail" (hand to repair) | "warn" (report only)
    message: str


@dataclass(frozen=True)
class ValidationReport:
    """Verdicts for one translated file, consumed by per-entry repair.

    ``failed_indices`` is the repair worklist: SRT indices, deduplicated and in
    file order, of the entries that must be re-translated. Warnings are for the
    run report only and never block.
    """

    failures: tuple[EntryIssue, ...] = ()
    warnings: tuple[EntryIssue, ...] = ()
    entries_checked: int = 0

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def failed_indices(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(issue.index for issue in self.failures))

    def summary(self, limit: int = 10) -> str:
        """One line per issue, failures first, capped at *limit* of each."""
        lines = [
            f"{len(self.failures)} failing / {len(self.warnings)} warning entries "
            f"out of {self.entries_checked} checked"
        ]
        for label, issues in (("FAIL", self.failures), ("WARN", self.warnings)):
            for issue in issues[:limit]:
                lines.append(f"  {label} entry {issue.index} [{issue.kind}]: {issue.message}")
            if len(issues) > limit:
                lines.append(f"  ... and {len(issues) - limit} more {label} entries")
        return "\n".join(lines)


class ReadabilityError(RuntimeError):
    """A translation that layers 2-4 reject, carrying its own repair worklist.

    The report travels with the exception so a caller (per-entry repair) can
    act on ``failed_indices`` instead of re-running the whole translation.
    ``detail`` carries what a repair pass additionally found (a repair that
    echoed its input, a content check the spliced file no longer passes).

    The error crosses a ``ProcessPoolExecutor`` boundary, so it MUST stay
    picklable: ``BaseException.__reduce__`` would replay only the formatted
    message handed to ``super().__init__``, and the parent's result reader
    would raise ``TypeError`` and declare the whole pool broken -- killing
    every other language in the run. ``__reduce__`` therefore rebuilds the
    error from its real arguments.
    """

    def __init__(
        self, target_language: str, report: ValidationReport, detail: str | None = None
    ) -> None:
        message = f"Readability validation failed for {target_language}: {report.summary()}"
        if detail:
            message = f"{message}\n  {detail}"
        super().__init__(message)
        self.target_language = target_language
        self.report = report
        self.detail = detail

    def __reduce__(
        self,
    ) -> tuple[type[ReadabilityError], tuple[str, ValidationReport, str | None]]:
        return (self.__class__, (self.target_language, self.report, self.detail))


def normalized(text: str) -> str:
    """Line breaks are wrapping, not content: measure the spoken string."""
    return " ".join(text.split())


def entry_cps(text: str, duration_seconds: float) -> float:
    """Characters per second of an entry; the property YouTube cares about."""
    return len(normalized(text)) / max(duration_seconds, _MIN_MEASURABLE_SECONDS)


def longest_repeat_run(text: str, max_ngram: int = _MAX_NGRAM) -> int:
    """Most consecutive repetitions of any unit of 1..*max_ngram* characters.

    Counts CHARACTERS, not tokens: the corrupt Persian entry is 26 402
    characters with no space in it, and CJK and Thai have no word separators
    either, so a token-based check sees a single token and misses the loop.
    """
    compact = normalized(text)
    if not compact:
        return 0

    best = 1
    for size in range(1, min(max_ngram, len(compact) // 2) + 1):
        start = 0
        while start + 2 * size <= len(compact):
            # Cheap reject before slicing: a run needs the same char one unit on.
            if compact[start] != compact[start + size]:
                start += 1
                continue
            unit = compact[start : start + size]
            end = start + size
            repeats = 1
            while compact[end : end + size] == unit:
                repeats += 1
                end += size
            if repeats > 1:
                best = max(best, repeats)
                # Step back one unit so a phase-shifted run is still seen.
                start = end - size + 1
            else:
                start += 1
    return best


def bidi_balance(text: str) -> int:
    """Opening minus closing bidi controls; nonzero means broken RTL."""
    openers = sum(text.count(char) for char in BIDI_OPENERS)
    closers = sum(text.count(char) for char in BIDI_CLOSERS)
    return openers - closers


def _seconds(start: timedelta, end: timedelta) -> float:
    return (end - start).total_seconds()


def _cps_issue(
    index: int, croatian_cps: float, cps: float, ceiling: float
) -> EntryIssue | None:
    """Layer 2. Degeneration fails; merely uncomfortable pacing warns."""
    if cps > CPS_FAIL and cps > CPS_FAIL_RATIO * croatian_cps:
        ratio = cps / croatian_cps if croatian_cps > 0 else float("inf")
        return EntryIssue(
            index,
            2,
            "cps",
            "fail",
            f"{cps:.0f} chars/s, {ratio:.0f}x the Croatian entry's "
            f"{croatian_cps:.0f} chars/s",
        )
    if cps > ceiling:
        return EntryIssue(
            index,
            2,
            "cps",
            "warn",
            f"{cps:.0f} chars/s, above the {ceiling:.0f} chars/s ceiling",
        )
    return None


def _repetition_issue(index: int, croatian_run: int, run: int) -> EntryIssue | None:
    """Layer 3. Absolute run alone flags 31 correct translations."""
    if run >= REPEAT_FAIL_RUN and run > REPEAT_FAIL_RATIO * croatian_run:
        return EntryIssue(
            index,
            3,
            "repetition",
            "fail",
            f"a unit repeats {run}x, against {croatian_run}x in the Croatian entry",
        )
    return None


def _bidi_issue(index: int, text: str) -> EntryIssue | None:
    """Layer 4. Always a warning: unpaired controls are repaired mechanically
    (offline, no API call), so they never justify discarding a translation."""
    balance = bidi_balance(text)
    if balance == 0:
        return None
    side = "opening" if balance > 0 else "closing"
    return EntryIssue(
        index, 4, "bidi", "warn", f"{abs(balance)} unpaired {side} bidi control(s)"
    )


def validate_entries(
    croatian_subs: list[srt.Subtitle], translated_subs: list[srt.Subtitle]
) -> ValidationReport:
    """Run layers 2-4 over a translation still aligned 1:1 with its source.

    The warn ceiling comes from the dominant script of the whole file, not of
    the single entry: a two-word entry rarely carries enough letters to detect
    its own script.

    Raises:
        RuntimeError: when the two lists differ in length (layer 1 must run first).
    """
    if len(croatian_subs) != len(translated_subs):
        raise RuntimeError(
            f"Readability validation needs a 1:1 mapping: {len(translated_subs)} "
            f"translated entries against {len(croatian_subs)} Croatian"
        )

    script = detect_script(" ".join(entry.content for entry in translated_subs))
    ceiling = CPS_WARN_CEILING.get(script, CPS_WARN_CEILING[Script.OTHER])

    failures: list[EntryIssue] = []
    warnings: list[EntryIssue] = []
    for croatian, entry in zip(croatian_subs, translated_subs, strict=True):
        index = entry.index
        candidates = (
            _cps_issue(
                index,
                entry_cps(croatian.content, _seconds(croatian.start, croatian.end)),
                entry_cps(entry.content, _seconds(entry.start, entry.end)),
                ceiling,
            ),
            _repetition_issue(
                index,
                longest_repeat_run(croatian.content),
                longest_repeat_run(entry.content),
            ),
            _bidi_issue(index, entry.content),
        )
        for issue in candidates:
            if issue is None:
                continue
            (failures if issue.level == "fail" else warnings).append(issue)

    return ValidationReport(
        failures=tuple(failures),
        warnings=tuple(warnings),
        entries_checked=len(translated_subs),
    )
