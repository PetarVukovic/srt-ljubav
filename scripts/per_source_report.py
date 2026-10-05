"""Per-source CSV report: one CSV next to each source SRT listing all its
translations with entry counts and (where logged) duration/tokens/cost.

    uv run python scripts/per_source_report.py [input_dir]

Writes ``{stem}_translations.csv`` into each topic folder. Cost/time columns
come from ``usage_logs/jobs.csv`` (latest row per source+language); files
translated before CSV logging existed keep those columns empty.
"""

from __future__ import annotations

import csv
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from srtljubav import srtio  # noqa: E402
from srtljubav.config import DEFAULT_JOBS_CSV  # noqa: E402
from srtljubav.languages import (  # noqa: E402
    EXTRA_LANGUAGES,
    TARGET_LANGUAGES,
    language_dir_name,
    translation_suffixes,
)

FIELDS = [
    "language",
    "file",
    "entries",
    "status",
    "elapsed_seconds",
    "prompt_tokens",
    "output_tokens",
    "total_tokens",
    "cost_usd",
    "translated_at",
]


def load_job_log() -> dict[tuple[str, str], dict]:
    """Latest jobs.csv row per (source basename, language)."""
    latest: dict[tuple[str, str], dict] = {}
    if not DEFAULT_JOBS_CSV.exists():
        return latest
    with DEFAULT_JOBS_CSV.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (Path(row["source"]).name, row["language"])
            latest[key] = row  # file is chronological; last write wins
    return latest


def _languages_for(source: Path) -> list[str]:
    """Default targets, plus opt-in extras that actually have an output file."""
    extras = [
        language
        for language in EXTRA_LANGUAGES
        if source.with_name(f"{source.stem}_{language_dir_name(language)}.srt").exists()
    ]
    return TARGET_LANGUAGES + extras


def report_source(
    source: Path,
    jobs: dict[tuple[str, str], dict],
    archive_root: Path | None = None,
) -> Path:
    rows = []
    for language in _languages_for(source):
        suffix = language_dir_name(language)
        out = source.with_name(f"{source.stem}_{suffix}.srt")
        row = {field: "" for field in FIELDS}
        row["language"] = suffix
        row["file"] = out.name
        if out.exists():
            entries = len(srtio.parse(srtio.read_srt_text(out)))
            row["entries"] = entries
            row["status"] = "ok" if entries else "in_progress"
        else:
            row["status"] = "missing"
        logged = jobs.get((source.name, language)) or jobs.get((source.name, suffix))
        if logged:
            row["status"] = logged["status"] if not row["entries"] else row["status"]
            row["elapsed_seconds"] = logged["elapsed_seconds"]
            row["prompt_tokens"] = logged["prompt_tokens"]
            row["output_tokens"] = logged["output_tokens"]
            row["total_tokens"] = logged["total_tokens"]
            row["cost_usd"] = logged["cost_usd"]
            row["translated_at"] = logged["timestamp"]
        rows.append(row)

    target = source.with_name(f"{source.stem}_translations.csv")
    with target.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # File a copy under the month of the run. input_files/ is gitignored and
    # gets cleared once a topic ships, so the per-topic CSV disappears with it;
    # the dated copy is what survives to answer "what went out in 2026-09".
    if archive_root is not None:
        month_dir = archive_root / date.today().strftime("%Y-%m")
        month_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(target, month_dir / target.name)
    return target


def main() -> int:
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("input_files")
    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 2

    archive_root = Path(__file__).resolve().parent.parent / "csv_info_pricing"
    suffixes = translation_suffixes()
    jobs = load_job_log()
    for topic in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        for srt_file in sorted(topic.glob("*.srt")):
            if any(srt_file.stem.endswith(s) for s in suffixes):
                continue
            target = report_source(srt_file, jobs, archive_root=archive_root)
            with target.open(encoding="utf-8") as fh:
                done = sum(1 for r in csv.DictReader(fh) if r["status"] == "ok")
            total = len(_languages_for(srt_file))
            print(f"{target.name}: {done}/{total} translations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
