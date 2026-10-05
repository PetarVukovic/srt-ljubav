"""Offline repair of the existing archive (implementation plan, Step 5).

Every archived file predates the structure-preserving pipeline: each language
was reflowed on its own, so no translation shares its segmentation with the
Croatian upload copy, and the Croatian copies themselves predate the window
repair and the short-entry rescue. This script re-derives what can be derived
and re-shapes what can be re-shaped, with **zero API calls**:

1. the Croatian upload copy is regenerated through ``prepare_source_text``,
2. every translation is regrouped into those Croatian windows -- its own
   entries are merged back, in order, into the window each one overlaps, so the
   file ends up 1:1 with Croatian in count, indices and timestamps,
3. unpaired bidi controls are closed per entry,
4. the result is re-wrapped to the target script's line width.

Nothing is written without ``--apply``; the default run only reports, before
and after, per file. Whatever no offline pass can fix -- a degenerated entry,
Croatian text a language was never given -- is LISTED, never touched: those are
the only entries that still need a Gemini call.

    uv run python scripts/repair_archive.py [input_dir] [--topic SUBSTRING]
    uv run python scripts/repair_archive.py --apply

Duplicate topic folders are reported the same way: verified, never deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import srt  # noqa: E402

from srtljubav import scripts, srtio, validation, ytsplit  # noqa: E402
from srtljubav.config import TranslationConfig, YTConfig  # noqa: E402
from srtljubav.discovery import discover_source_srts  # noqa: E402
from srtljubav.languages import (  # noqa: E402
    SOURCE_LANGUAGE_SUFFIX,
    translation_suffixes,
)
from srtljubav.pipeline import (  # noqa: E402
    _is_nontrivial,
    _normalized,
    prepare_source_text,
    structure_mismatch,
)

# Reporting thresholds, the same ones scripts/measure_readability.py uses:
# YouTube rejects sub-second cues, and 36 chars/s is the Latin reading ceiling.
MIN_DURATION = 1.0
CPS_CEILING = 36.0

# An entry left identical to the Croatian source in only a few of a topic's
# languages is real leakage; one identical in most of them is a proper name, a
# verse reference or a number that every correct translation keeps verbatim.
LEFTOVER_MAX_LANGUAGE_SHARE = 0.25

# Bidi controls, split by what closes them (validation.py holds the flat sets).
ISOLATE_OPENERS = ("⁦", "⁧", "⁨")
EMBEDDING_OPENERS = ("‪", "‫")
ISOLATE_CLOSER = "⁩"
EMBEDDING_CLOSER = "‬"


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FileStats:
    """What the reviewer looks at, for one file, before or after the repair."""

    entries: int
    under_min_duration: int
    above_ceiling: int
    worst_cps: float
    one_to_one: bool | None = None


def _cps(sub: srt.Subtitle) -> float:
    return validation.entry_cps(sub.content, srtio.td_to_seconds(sub.end - sub.start))


def measure(
    subs: list[srt.Subtitle], croatian_subs: list[srt.Subtitle] | None = None
) -> FileStats:
    """Readability and structure of one file. ``one_to_one`` is None for Croatian."""
    rates = [_cps(sub) for sub in subs if sub.content.strip()]
    return FileStats(
        entries=len(subs),
        under_min_duration=sum(
            1 for sub in subs if srtio.td_to_seconds(sub.end - sub.start) < MIN_DURATION
        ),
        above_ceiling=sum(1 for rate in rates if rate > CPS_CEILING),
        worst_cps=max(rates, default=0.0),
        one_to_one=(
            None if croatian_subs is None else structure_mismatch(subs, croatian_subs) is None
        ),
    )


# --------------------------------------------------------------------------
# Regrouping a translation into the Croatian windows
# --------------------------------------------------------------------------


def _best_window(entry: srt.Subtitle, croatian: list[srt.Subtitle], lo: int) -> int:
    """Index of the window *entry* overlaps most, never behind window *lo*."""
    best_index: int | None = None
    best_overlap = 0.0
    for index in range(lo, len(croatian)):
        window = croatian[index]
        if window.start >= entry.end:
            break  # windows are ordered: nothing further can overlap
        overlap = srtio.td_to_seconds(
            min(entry.end, window.end) - max(entry.start, window.start)
        )
        if overlap > best_overlap:
            best_overlap, best_index = overlap, index
    if best_index is not None:
        return best_index
    # No overlap at all: the entry sits in silence the repaired windows moved
    # away from. Fall back to the nearest start that is not behind *lo*.
    return min(
        range(lo, len(croatian)),
        key=lambda index: abs(srtio.td_to_seconds(croatian[index].start - entry.start)),
    )


def assign_to_windows(
    translated: list[srt.Subtitle], croatian: list[srt.Subtitle]
) -> list[list[srt.Subtitle]]:
    """Bucket the translated entries into the Croatian windows, in order.

    Assignment is monotonic: a chunk never lands behind the window its
    predecessor took, so the text order of the file is preserved even where the
    old segmentation drifted from the repaired one.
    """
    groups: list[list[srt.Subtitle]] = [[] for _ in croatian]
    lo = 0
    for entry in translated:
        index = _best_window(entry, croatian, lo)
        groups[index].append(entry)
        lo = index
    return groups


def _donor(groups: list[list[srt.Subtitle]], index: int, step: int) -> int | None:
    position = index + step
    while 0 <= position < len(groups):
        if len(groups[position]) > 1:
            return position
        position += step
    return None


def redistribute(groups: list[list[srt.Subtitle]]) -> list[list[srt.Subtitle]]:
    """Fill empty windows by shifting text along from a crowded neighbour.

    The reviewer's own example: a 420-character source entry whose end was cut
    ~25 s early was split into seven 0.1 s chunks, so the whole translated
    passage sits inside two seconds. The repaired windows spread the same text
    over seven windows, and a pure overlap assignment would pile all seven
    chunks into the first of them and leave the rest blank. Shifting one chunk
    per window across the gap restores the intended distribution; it moves
    whole entries only, so the reading order never changes.
    """
    groups = [list(group) for group in groups]
    for index, group in enumerate(groups):
        if group:
            continue
        left = _donor(groups, index, -1)
        if left is not None:
            for position in range(left, index):
                groups[position + 1].insert(0, groups[position].pop())
            continue
        right = _donor(groups, index, 1)
        if right is not None:
            for position in range(right, index, -1):
                groups[position - 1].append(groups[position].pop(0))
    return groups


def _join_group(group: list[srt.Subtitle]) -> str:
    """One window's text: the chunks joined the way the script writes words."""
    texts = [entry.content for entry in group if entry.content.strip()]
    if not texts:
        return ""
    # The script decision is made once per window, over all its chunks: a
    # two-character chunk rarely carries enough letters to detect its own.
    char_mode = scripts.breaks_at_chars(" ".join(texts))
    separator = "" if char_mode else " "
    return separator.join(ytsplit._join_content_lines(text, char_mode) for text in texts)


def regroup_to_croatian(
    translated: list[srt.Subtitle], croatian: list[srt.Subtitle]
) -> list[srt.Subtitle]:
    """The translated text carried into the Croatian structure, 1:1.

    Indices and timestamps come from Croatian and nothing else; the text is the
    translation's own, complete and in order -- entries are only merged, never
    dropped, cut or re-timed.
    """
    groups = redistribute(assign_to_windows(translated, croatian))
    return [
        srt.Subtitle(
            index=window.index,
            start=window.start,
            end=window.end,
            content=_join_group(group),
        )
        for window, group in zip(croatian, groups, strict=True)
    ]


# --------------------------------------------------------------------------
# Bidi repair
# --------------------------------------------------------------------------


def balance_bidi(text: str) -> str:
    """Close every opened bidi control and drop the ones that close nothing.

    The old reflow split entries between an opening control and its pop, which
    leaves both halves rendering broken RTL. Balancing is mechanical -- no
    visible character is added or removed -- so it never needs a model call.
    """
    out: list[str] = []
    stack: list[str] = []
    for char in text:
        if char in ISOLATE_OPENERS or char in EMBEDDING_OPENERS:
            stack.append(ISOLATE_CLOSER if char in ISOLATE_OPENERS else EMBEDDING_CLOSER)
            out.append(char)
        elif char in (ISOLATE_CLOSER, EMBEDDING_CLOSER):
            if stack and stack[-1] == char:
                stack.pop()
                out.append(char)
            # else: a pop with nothing to pop -- its opener stayed in the
            # neighbouring entry, so drop it instead of inventing an opener.
        else:
            out.append(char)
    out.extend(reversed(stack))
    return "".join(out)


# --------------------------------------------------------------------------
# What no offline pass can fix
# --------------------------------------------------------------------------


def croatian_identical_indices(
    croatian: list[srt.Subtitle], subs: list[srt.Subtitle]
) -> tuple[int, ...]:
    """Entries a language left verbatim Croatian, judged per entry.

    Uses the pipeline's own leakage rule (layer 6): only entries long enough to
    be a sentence count, so verse references and numbers never register.
    """
    indices: list[int] = []
    for window, entry in zip(croatian, subs, strict=True):
        source = _normalized(window.content)
        if _is_nontrivial(source) and source == _normalized(entry.content):
            indices.append(entry.index)
    return tuple(indices)


def leftovers_by_language(
    identical: dict[str, tuple[int, ...]],
) -> dict[str, tuple[int, ...]]:
    """Drop the entries every language keeps verbatim; keep the real leakage.

    An entry identical to Croatian in one or two of a topic's 33 languages is
    a batch the model skipped. The same entry identical in all 33 is a name, a
    Bible reference or a number that a correct translation keeps as it is.
    """
    if not identical:
        return {}
    counts: Counter[int] = Counter(index for indices in identical.values() for index in indices)
    ceiling = LEFTOVER_MAX_LANGUAGE_SHARE * len(identical)
    result: dict[str, tuple[int, ...]] = {}
    for language, indices in identical.items():
        leftovers = tuple(index for index in indices if counts[index] <= ceiling)
        if leftovers:
            result[language] = leftovers
    return result


# --------------------------------------------------------------------------
# The per-file repair
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RepairedTranslation:
    """Both stages of one file's repair.

    ``regrouped`` is the intermediate the caller needs anyway -- validation,
    the bidi count and the leakage check all read it -- so it is returned
    rather than recomputed: the window assignment plus redistribution is the
    expensive pass and running it twice per file doubles the archive sweep.
    """

    regrouped: list[srt.Subtitle]
    repaired: list[srt.Subtitle]


def repair_translation(
    translated: list[srt.Subtitle], croatian: list[srt.Subtitle], cfg: YTConfig
) -> RepairedTranslation:
    """Regroup into the Croatian windows, balance bidi, re-wrap. Idempotent."""
    regrouped = regroup_to_croatian(translated, croatian)
    balanced = [
        srt.Subtitle(
            index=entry.index,
            start=entry.start,
            end=entry.end,
            content=balance_bidi(entry.content),
        )
        for entry in regrouped
    ]
    return RepairedTranslation(regrouped=regrouped, repaired=ytsplit.rewrap(balanced, cfg))


# --------------------------------------------------------------------------
# Duplicate topic folders
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DuplicateTopic:
    """A topic folder holding a copy of another topic's source SRT."""

    duplicate: Path
    canonical: Path
    identical: tuple[str, ...]
    differing: tuple[str, ...]
    only_in_duplicate: tuple[str, ...]

    @property
    def safe_to_delete(self) -> bool:
        """True only when every file in it exists, byte for byte, elsewhere."""
        return bool(self.identical) and not self.differing and not self.only_in_duplicate

    @property
    def rm_command(self) -> str:
        return f'rm -rf "{self.duplicate}"'


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compare_topics(duplicate: Path, canonical: Path) -> DuplicateTopic:
    identical: list[str] = []
    differing: list[str] = []
    only_here: list[str] = []
    for path in sorted(duplicate.iterdir()):
        if not path.is_file():
            continue
        twin = canonical / path.name
        if not twin.is_file():
            only_here.append(path.name)
        elif _digest(path) == _digest(twin):
            identical.append(path.name)
        else:
            differing.append(path.name)
    return DuplicateTopic(
        duplicate=duplicate,
        canonical=canonical,
        identical=tuple(identical),
        differing=tuple(differing),
        only_in_duplicate=tuple(only_here),
    )


def duplicate_topic_findings(input_dir: Path) -> list[DuplicateTopic]:
    """Topic folders whose source SRT is named after a DIFFERENT topic folder.

    That is the archive's actual signature: ``15. <title>/`` holds
    ``VRIJEME JE KRAJA! 15. <title>.srt``, and a folder with exactly that name
    sits next to it. Folder names alone cannot be matched -- the prefixed
    copies also rewrote quotes into underscores.
    """
    folders = {path.name: path for path in input_dir.iterdir() if path.is_dir()}
    findings: list[DuplicateTopic] = []
    for folder in sorted(folders.values()):
        for path in sorted(folder.glob("*.srt")):
            canonical = folders.get(path.stem)
            if canonical is None or canonical == folder:
                continue
            findings.append(_compare_topics(folder, canonical))
            break
    return findings


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


@dataclass
class FileReport:
    path: Path
    label: str
    before: FileStats
    after: FileStats
    changed: bool
    bidi_repaired: int = 0
    empty_entries: int = 0
    overflow_entries: int = 0
    failures: tuple[validation.EntryIssue, ...] = ()
    identical_indices: tuple[int, ...] = ()


def _one_to_one(stats: FileStats) -> str:
    if stats.one_to_one is None:
        return "-"
    return "yes" if stats.one_to_one else "no"


def _pair(before: object, after: object) -> str:
    return f"{before} -> {after}"


def format_file_line(report: FileReport) -> str:
    before, after = report.before, report.after
    flags = []
    if report.bidi_repaired:
        flags.append(f"bidi {report.bidi_repaired}")
    if report.empty_entries:
        flags.append(f"EMPTY {report.empty_entries}")
    if report.overflow_entries:
        flags.append(f">{YTConfig().rewrap_max_lines} lines {report.overflow_entries}")
    if report.failures:
        flags.append(f"NEEDS GEMINI {len(report.failures)}")
    return (
        f"  {report.label:<26}"
        f"{_pair(before.entries, after.entries):>15}"
        f"{_pair(before.under_min_duration, after.under_min_duration):>12}"
        f"{_pair(before.above_ceiling, after.above_ceiling):>12}"
        f"{_pair(f'{before.worst_cps:.0f}', f'{after.worst_cps:.0f}'):>14}"
        f"{_pair(_one_to_one(before), _one_to_one(after)):>14}"
        + ("   " + ", ".join(flags) if flags else "")
    )


HEADER = (
    f"  {'file':<26}{'entries':>15}{'< 1.0 s':>12}{'> 36 cps':>12}"
    f"{'worst cps':>14}{'1:1 Croatian':>14}"
)


# --------------------------------------------------------------------------
# The sweep
# --------------------------------------------------------------------------


def _language_of(path: Path, stem: str) -> str:
    return path.stem[len(stem) + 1 :]


def translations_of(source: Path) -> list[Path]:
    """The translated outputs that belong to *source*, Croatian excluded.

    Matched by exact ``{stem}_{Language}.srt`` name, never by "every file in
    the folder that looks translated": a topic folder can hold two lectures,
    and a prefix-blind sweep would rewrite lecture B's translations into
    lecture A's windows. Existence of a ``_Croatian.srt`` is deliberately NOT
    part of the test -- it is written by this same run, so keying on it would
    make the dry run plan less work than ``--apply`` performs.
    """
    suffixes = translation_suffixes() - {SOURCE_LANGUAGE_SUFFIX}
    names = {f"{source.stem}{suffix}.srt" for suffix in suffixes}
    return sorted(path for path in source.parent.glob("*.srt") if path.name in names)


def repair_topic(
    source: Path, cfg: TranslationConfig, *, apply: bool
) -> tuple[list[FileReport], dict[str, tuple[int, ...]]]:
    """Regenerate one topic's Croatian copy and realign all its translations."""
    stem = source.stem
    croatian_path = source.with_name(f"{stem}_Croatian.srt")
    old_croatian = (
        srtio.parse(srtio.read_srt_text(croatian_path)) if croatian_path.exists() else []
    )
    croatian_text = prepare_source_text(source, cfg)
    new_croatian = srtio.parse(croatian_text)

    reports = [
        FileReport(
            path=croatian_path,
            label="Croatian",
            before=measure(old_croatian),
            after=measure(new_croatian),
            changed=(not croatian_path.exists())
            or srtio.read_srt_text(croatian_path) != croatian_text,
        )
    ]
    if apply:
        croatian_path.write_text(croatian_text, encoding="utf-8")

    identical: dict[str, tuple[int, ...]] = {}
    for path in translations_of(source):
        original = srtio.read_srt_text(path)
        subs = srtio.parse(original)
        result = repair_translation(subs, new_croatian, cfg.yt)
        repaired, regrouped = result.repaired, result.regrouped
        report = validation.validate_entries(new_croatian, regrouped)
        language = _language_of(path, stem)
        identical[language] = croatian_identical_indices(new_croatian, regrouped)

        updated = srtio.compose(repaired)
        reports.append(
            FileReport(
                path=path,
                label=language,
                before=measure(subs, old_croatian or new_croatian),
                after=measure(repaired, new_croatian),
                changed=updated != original,
                bidi_repaired=sum(
                    1 for entry in regrouped if validation.bidi_balance(entry.content) != 0
                ),
                empty_entries=sum(1 for entry in repaired if not entry.content.strip()),
                overflow_entries=len(ytsplit.rewrap_overflow(repaired, cfg.yt)),
                failures=report.failures,
                identical_indices=identical[language],
            )
        )
        if apply:
            path.write_text(updated, encoding="utf-8")

    leftovers = leftovers_by_language(identical)
    for report_item in reports[1:]:
        report_item.identical_indices = leftovers.get(report_item.label, ())
    return reports, leftovers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir", nargs="?", type=Path, default=Path("input_files"), help="archive root"
    )
    parser.add_argument(
        "--apply", action="store_true", help="write the repaired files (default: report only)"
    )
    parser.add_argument("--topic", help="only topics whose folder name contains this substring")
    args = parser.parse_args()

    # Regrouping merges each translation's own entries into the Croatian windows
    # they overlap, which is only faithful when the new windows are unions of
    # the old ones. The anchor segmenter cuts elsewhere, so an archive aligned
    # to the legacy segmentation is re-shaped onto the legacy segmentation; a
    # move to the anchor segmentation is a re-translation (``translate
    # --overwrite``), never a regroup.
    cfg = TranslationConfig(segmenter="legacy")
    duplicates = duplicate_topic_findings(args.input_dir)
    duplicate_dirs = {finding.duplicate for finding in duplicates}

    print("DRY RUN -- nothing is written; pass --apply to repair" if not args.apply else "APPLYING")
    print()

    all_reports: list[FileReport] = []
    for source in discover_source_srts(args.input_dir):
        topic = source.parent
        if topic in duplicate_dirs:
            print(f"== {topic.name}\n   skipped: duplicate topic folder (see below)\n")
            continue
        if args.topic and args.topic not in topic.name:
            continue
        reports, _leftovers = repair_topic(source, cfg, apply=args.apply)
        print(f"== {topic.name}")
        print(HEADER)
        for report in reports:
            print(format_file_line(report))
        print()
        all_reports.extend(reports)

    croatian = [report for report in all_reports if report.label == "Croatian"]
    translations = [report for report in all_reports if report.label != "Croatian"]
    print("=" * 100)

    def totals(reports: list[FileReport], attribute: str) -> str:
        before = sum(getattr(report.before, attribute) for report in reports)
        after = sum(getattr(report.after, attribute) for report in reports)
        return _pair(before, after)

    # Reported apart: once a translation is 1:1, its own duration and reading
    # statistics are the Croatian ones, so folding them together would count
    # the same windows 33 times over.
    print(f"TOTALS -- {len(croatian)} Croatian upload copies")
    print(f"  entries                {totals(croatian, 'entries')}")
    print(f"  entries < 1.0 s        {totals(croatian, 'under_min_duration')}")
    print(f"  entries > 36 cps       {totals(croatian, 'above_ceiling')}")

    aligned_before = sum(1 for report in translations if report.before.one_to_one)
    aligned_after = sum(1 for report in translations if report.after.one_to_one)
    entries_after = sum(report.after.entries for report in translations)
    overflow = sum(report.overflow_entries for report in translations)
    print(f"TOTALS -- {len(translations)} translations")
    print(f"  1:1 with Croatian      {aligned_before} -> {aligned_after}")
    print(f"  entries                {totals(translations, 'entries')}")
    print(f"  entries > 36 cps       {totals(translations, 'above_ceiling')}")
    print(f"  bidi entries repaired  {sum(report.bidi_repaired for report in translations)}")
    share = f" ({overflow / entries_after:.1%} of all entries)" if entries_after else ""
    print(f"  entries over {YTConfig().rewrap_max_lines} lines   {overflow}{share}")
    empty = [report for report in translations if report.empty_entries]
    print(f"  windows left empty     {sum(report.empty_entries for report in empty)}"
          f" in {len(empty)} files")
    for report in empty:
        print(f"      {report.path.parent.name} / {report.label}: {report.empty_entries}")

    print()
    print("ENTRIES THAT STILL NEED A GEMINI CALL (not repaired by this script)")
    needy = [report for report in translations if report.failures or report.identical_indices]
    if not needy:
        print("  none")
    for report in needy:
        if report.failures:
            indices = ", ".join(f"#{issue.index} ({issue.kind})" for issue in report.failures)
            print(f"  {report.path.parent.name} / {report.label}: {indices}")
        if report.identical_indices:
            indices = ", ".join(f"#{index}" for index in report.identical_indices)
            print(f"  {report.path.parent.name} / {report.label}: Croatian left in {indices}")

    print()
    print("DUPLICATE TOPIC FOLDERS (verified, never deleted -- your call)")
    for finding in duplicates:
        print(f"  {finding.duplicate.name}")
        print(f"    copy of: {finding.canonical.name}")
        print(f"    byte-identical: {', '.join(finding.identical) or 'none'}")
        if finding.differing:
            print(f"    DIFFERENT content: {', '.join(finding.differing)}")
        if finding.only_in_duplicate:
            print(f"    ONLY here (would be lost): {', '.join(finding.only_in_duplicate)}")
        print(f"    safe to delete: {'yes' if finding.safe_to_delete else 'no'}")
        print(f"    {finding.rm_command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
