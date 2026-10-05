"""Command-line interface: ``srt-ljubav translate | reformat | report``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from srtljubav.config import (
    CROATIAN_TRANSLATION_DESCRIPTION,
    DEFAULT_MODEL,
    DEFAULT_RUN_REPORT,
    DEFAULT_USAGE_LOG,
    TranslationConfig,
    YTConfig,
)
from srtljubav.languages import parse_language_filter

if TYPE_CHECKING:  # pipeline pulls in the API deps; keep it out of the CLI import path
    from srtljubav.pipeline import ReformatResult


def _add_translate_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "translate",
        help="Translate every source SRT in input_files to the target languages.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Path to a source SRT anywhere on disk: it gets copied into "
        "input_files/<stem>/ and ONLY that file is translated.",
    )
    parser.add_argument("--input-dir", type=Path, default=Path("input_files"))
    parser.add_argument("--workers", type=int, default=6, help="Parallel translation processes.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument(
        "--fallback-batch-sizes",
        default="20",
        help="Comma-separated smaller batch sizes to try after normal attempts fail; "
        "use 'none' to disable.",
    )
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--thinking-budget", type=int, default=0)
    parser.add_argument(
        "--thinking-level", choices=("minimal", "low", "medium", "high"), default="minimal"
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--description", default=CROATIAN_TRANSLATION_DESCRIPTION)
    parser.add_argument("--free-quota", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-translate existing output files."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only print planned missing translations."
    )
    parser.add_argument("--progress-log", action="store_true")
    parser.add_argument("--usage-log", type=Path, default=DEFAULT_USAGE_LOG)
    parser.add_argument("--run-report", type=Path, default=DEFAULT_RUN_REPORT)
    parser.add_argument(
        "--pricing-mode",
        choices=("standard", "free"),
        default="standard",
        help="Standard paid Gemini prices for estimates, or free to show charged cost as 0.",
    )
    parser.add_argument(
        "--service-tier",
        choices=("standard", "flex", "priority"),
        default=None,
        help="Gemini service tier. 'flex' bills at half the standard rate for higher "
        "latency, which suits this batch workload; omitted means the account default.",
    )
    parser.add_argument("--quiet-gst", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--limit-lines", type=int, default=None, help="Translate only the first N SRT sections."
    )
    parser.add_argument(
        "--languages", default=None, help="Comma-separated subset of target languages."
    )
    parser.add_argument(
        "--max-jobs", type=int, default=None, help="Run at most N translation jobs."
    )
    parser.add_argument(
        "--test-output-dir",
        type=Path,
        default=None,
        help="Write limited test translations under this folder instead of input_files.",
    )
    parser.add_argument(
        "--clean-sections",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fix timestamps and merge tiny SRT sections before translation.",
    )
    parser.add_argument(
        "--yt-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reflow translated output into YouTube-friendly entries "
        "(<=2 lines, per-script width).",
    )
    parser.add_argument(
        "--segmenter",
        choices=("anchor", "legacy"),
        default="anchor",
        help="How the Croatian source is segmented: 'anchor' keeps every word in "
        "its fragment's time (default); 'legacy' is the old block-merge reflow.",
    )


def _add_reformat_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "reformat",
        help="Re-wrap the lines of existing translated SRT files in place (no API calls). "
        "Entry count, indices and timestamps stay 1:1 with the _Croatian.srt sibling.",
    )
    parser.add_argument("--input-dir", type=Path, default=Path("input_files"))
    parser.add_argument(
        "--languages", default=None, help="Comma-separated subset of target languages."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only report what would change."
    )
    parser.add_argument(
        "--rewrap-only",
        action="store_true",
        help="Deprecated no-op: re-wrapping lines is what reformat does. Entries too "
        "long for two lines are reported, never split.",
    )


def _add_sync_db_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "sync-db",
        help="Sync the archive into Postgres, one row per subtitle segment "
        "(no API calls; never part of a translation run).",
    )
    parser.add_argument("--input-dir", type=Path, default=Path("input_files"))
    parser.add_argument(
        "--languages", default=None, help="Comma-separated subset of target languages."
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Postgres connection string; defaults to DATABASE_URL from the "
        "environment or .env.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be synced; no connection is opened.",
    )
    parser.add_argument(
        "--create-schema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the CREATE TABLE IF NOT EXISTS statements first.",
    )


def _add_report_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "report", help="Summarize token usage and cost from an existing usage JSONL log."
    )
    parser.add_argument("--usage-log", type=Path, default=DEFAULT_USAGE_LOG)
    parser.add_argument("--pricing-mode", choices=("standard", "free"), default="standard")
    parser.add_argument(
        "--service-tier",
        choices=("standard", "flex", "priority"),
        default=None,
        help="Fallback tier for records logged before the tier was recorded per call.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="srt-ljubav",
        description="Batch SRT translation pipeline with YouTube-quality formatting.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_translate_parser(subparsers)
    _add_reformat_parser(subparsers)
    _add_sync_db_parser(subparsers)
    _add_report_parser(subparsers)
    return parser


def _import_source_file(file: Path, input_dir: Path) -> Path:
    """Copy an external SRT into input_files/<stem>/ and return the new path."""
    import shutil

    from srtljubav.languages import translation_suffixes

    if not file.is_file():
        raise ValueError(f"File does not exist: {file}")
    if file.suffix.lower() != ".srt":
        raise ValueError(f"Not an .srt file: {file}")
    if any(file.stem.endswith(suffix) for suffix in translation_suffixes()):
        raise ValueError(
            f"File name ends with a translation suffix ({file.stem}) — "
            "rename it so it is not treated as an already-translated output."
        )

    dest = input_dir / file.stem / file.name
    if dest.resolve() == file.resolve():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(file, dest)
        print(f"Copied source -> {dest}")
    else:
        print(f"Source already present: {dest}")
    return dest


def _cmd_translate(args: argparse.Namespace) -> int:
    from srtljubav.pipeline import load_dotenv, run_translation

    load_dotenv(Path(".env"))
    try:
        target_languages = parse_language_filter(args.languages)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.workers < 1:
        print("--workers must be at least 1", file=sys.stderr)
        return 2

    try:
        fallback_batch_sizes = _parse_fallback_batch_sizes(args.fallback_batch_sizes)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    only_source = None
    if args.file:
        try:
            only_source = _import_source_file(args.file, args.input_dir)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    output_root = args.test_output_dir
    if args.limit_lines and output_root is None:
        output_root = Path("test_outputs")

    cfg = TranslationConfig(
        model=args.model,
        batch_size=args.batch_size,
        fallback_batch_sizes=fallback_batch_sizes,
        streaming=args.streaming,
        thinking=args.thinking,
        thinking_budget=args.thinking_budget,
        thinking_level=args.thinking_level,
        temperature=args.temperature,
        description=args.description,
        free_quota=args.free_quota,
        quiet_gst=args.quiet_gst,
        progress_log=args.progress_log,
        pricing_mode=args.pricing_mode,
        service_tier=args.service_tier,
        clean_sections=args.clean_sections,
        overwrite=args.overwrite,
        limit_lines=args.limit_lines,
        yt_split=args.yt_split,
        segmenter=args.segmenter,
        usage_log=args.usage_log,
    )

    return run_translation(
        cfg,
        input_dir=args.input_dir,
        output_root=output_root,
        target_languages=target_languages,
        workers=args.workers,
        max_jobs=args.max_jobs,
        run_report=args.run_report,
        dry_run=args.dry_run,
        only_source=only_source,
    )


def _parse_fallback_batch_sizes(value: str | None) -> tuple[int, ...]:
    if value is None:
        return ()
    text = value.strip()
    if not text or text.lower() in {"none", "off", "false", "0"}:
        return ()
    sizes: list[int] = []
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            size = int(part)
        except ValueError as exc:
            raise ValueError("--fallback-batch-sizes must contain integers") from exc
        if size < 1:
            raise ValueError("--fallback-batch-sizes values must be at least 1")
        sizes.append(size)
    return tuple(sizes)


def _cmd_reformat(args: argparse.Namespace) -> int:
    from srtljubav.discovery import discover_translated_srts
    from srtljubav.pipeline import reformat_existing

    try:
        target_languages = parse_language_filter(args.languages)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    paths = discover_translated_srts(args.input_dir, target_languages)
    if not paths:
        print("No translated SRT files found.")
        return 0

    if args.rewrap_only:
        print("--rewrap-only is the default and only behaviour; the flag is a no-op.")

    results = reformat_existing(paths, YTConfig(), dry_run=args.dry_run)
    changed = [r for r in results if r.changed]
    mode = "[dry-run] would change" if args.dry_run else "changed"
    for result in changed:
        print(
            f"{mode}: {result.path} ({result.entries_before} -> {result.entries_after} entries)"
        )
    print(f"\n{len(changed)}/{len(results)} file(s) {mode.split(' ')[-1]}.")
    _print_rewrap_report(results)
    return 0


def _print_rewrap_report(results: list[ReformatResult]) -> None:
    """Report what rewrap deliberately does not fix: over-long entries and any
    file whose structure has drifted from its Croatian sibling."""
    max_lines = YTConfig().rewrap_max_lines
    overflowing = [r for r in results if r.overflow_indices]
    for result in overflowing:
        indices = ", ".join(str(i) for i in result.overflow_indices[:10])
        more = "" if len(result.overflow_indices) <= 10 else ", ..."
        print(
            f"overflow: {result.path} "
            f"({len(result.overflow_indices)} entries exceed {max_lines} lines: "
            f"{indices}{more})"
        )
    mismatched = [r for r in results if r.croatian_mismatch]
    for result in mismatched:
        print(f"not 1:1: {result.path} ({result.croatian_mismatch})")
    unmatched = [r for r in results if r.croatian is None]
    print(f"{len(overflowing)}/{len(results)} file(s) with entries over {max_lines} lines.")
    print(
        f"{len(mismatched)}/{len(results)} file(s) not 1:1 with their _Croatian.srt sibling."
    )
    if unmatched:
        print(f"{len(unmatched)}/{len(results)} file(s) have no _Croatian.srt sibling.")


def _cmd_sync_db(args: argparse.Namespace) -> int:
    """Backfill/refresh the Postgres mirror of the archive. No API calls."""
    from srtljubav import db

    try:
        target_languages = parse_language_filter(args.languages)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    lectures = db.collect_lectures(args.input_dir, target_languages)
    files = sum(len(lecture.files) for lecture in lectures)
    entries = sum(len(file.entries) for lecture in lectures for file in lecture.files)
    print(f"{len(lectures)} lecture(s), {files} file(s), {entries} entries on disk.")
    _print_status_totals(lectures)

    if args.dry_run:
        print("[dry-run] nothing written; no connection opened.")
        return 0

    url = db.database_url(args.database_url)
    if not url:
        print(
            "No DATABASE_URL: pass --database-url or set DATABASE_URL in .env",
            file=sys.stderr,
        )
        return 2

    connection = db.connect(url)
    try:
        stats = db.sync_lectures(connection, lectures, create_schema=args.create_schema)
    finally:
        connection.close()

    print(
        f"{stats.files_written} file(s) written, {stats.files_unchanged} unchanged "
        f"(same file_hash), {stats.entries_written} segment(s) upserted."
    )
    return 0


def _print_status_totals(lectures: list) -> None:
    from collections import Counter

    counts: Counter[str] = Counter(
        entry.status
        for lecture in lectures
        for file in lecture.files
        for entry in file.entries
    )
    if counts:
        totals = ", ".join(f"{status}: {count}" for status, count in sorted(counts.items()))
        print(f"Segment verdicts — {totals}")


def _print_per_language_table(jobs_csv: Path) -> None:
    import csv
    from collections import defaultdict

    totals: dict[str, list[float]] = defaultdict(lambda: [0, 0.0, 0, 0.0])  # n, secs, tokens, usd
    with jobs_csv.open(encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if row.get("status") != "ok":
                continue
            entry = totals[row["language"]]
            entry[0] += 1
            entry[1] += float(row["elapsed_seconds"] or 0)
            entry[2] += int(row["total_tokens"] or 0)
            entry[3] += float(row["cost_usd"] or 0)

    if not totals:
        return

    print(f"\nPer language (from {jobs_csv}):")
    print(f"{'language':28} {'jobs':>4} {'time':>9} {'tokens':>10} {'cost USD':>9}")
    for language, (count, secs, tokens, usd) in sorted(totals.items(), key=lambda i: -i[1][1]):
        minutes, seconds = divmod(int(secs), 60)
        timing = f"{minutes}m {seconds:02d}s"
        print(f"{language:28} {count:>4} {timing:>9} {tokens:>10,} {usd:>9.4f}")
    total_secs = sum(v[1] for v in totals.values())
    minutes, seconds = divmod(int(total_secs), 60)
    print(f"{'TOTAL':28} {sum(v[0] for v in totals.values()):>4} {f'{minutes}m {seconds:02d}s':>9} "
          f"{sum(v[2] for v in totals.values()):>10,} {sum(v[3] for v in totals.values()):>9.4f}")


def _cmd_report(args: argparse.Namespace) -> int:
    from srtljubav.config import DEFAULT_JOBS_CSV
    from srtljubav.pricing import summarize_usage_records

    if not args.usage_log.exists():
        print(f"Usage log not found: {args.usage_log}", file=sys.stderr)
        return 2

    records = []
    with args.usage_log.open(encoding="utf-8") as log_file:
        for line in log_file:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    summary = summarize_usage_records(
        records,
        model_name=DEFAULT_MODEL,
        pricing_mode=args.pricing_mode,
        service_tier=args.service_tier,
    )
    print(f"Records: {len(records)}")
    print(f"Prompt/input tokens: {summary.prompt_tokens}")
    print(f"Output tokens: {summary.output_tokens}")
    print(f"Total tokens: {summary.total_tokens}")
    print(f"Estimated cost USD: {summary.estimated_cost_usd:.6f}")

    if DEFAULT_JOBS_CSV.exists():
        _print_per_language_table(DEFAULT_JOBS_CSV)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {
        "translate": _cmd_translate,
        "reformat": _cmd_reformat,
        "sync-db": _cmd_sync_db,
        "report": _cmd_report,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
