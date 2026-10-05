"""Append one row per source in a topic to ``csv_info_pricing/troskovnik.csv``.

Usage: append_ledger.py <topic_dir> <drive_folder_id> <run_started_utc>

Reads the ``{stem}_translations.csv`` that ``per_source_report.py`` wrote
beside each source, so run that first. ``run_started_utc`` (ISO 8601, ``Z``)
splits the cost into what this run spent and what the topic has cost in
total. The run cost sums EVERY ``usage_logs/jobs.csv`` row since then, failed
attempts included -- the per-source CSV keeps only the latest row per
language. Exit 0 when every expected language is ``ok``, 3 otherwise -- the
batch script uploads to Drive only on 0.
"""

from __future__ import annotations

import csv
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "csv_info_pricing" / "troskovnik.csv"
JOBS_CSV = ROOT / "usage_logs" / "jobs.csv"
FIELDS = [
    "logged_at",
    "source",
    "drive_folder_id",
    "languages_ok",
    "languages_total",
    "run_cost_usd",
    "total_cost_usd",
]


def _cost(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def spent_since(source_name: str, since: str) -> float:
    """Every jobs.csv cost for *source_name* logged at or after *since*."""
    if not JOBS_CSV.exists():
        return 0.0
    with JOBS_CSV.open(encoding="utf-8") as fh:
        return sum(
            _cost(row["cost_usd"])
            for row in csv.DictReader(fh)
            if Path(row["source"]).name == source_name and row["timestamp"] >= since
        )


def main() -> int:
    topic_dir, folder_id, since = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    reports = sorted(topic_dir.glob("*_translations.csv"))
    if not reports:
        print(f"No *_translations.csv in {topic_dir}", file=sys.stderr)
        return 2

    complete = True
    new_file = not LEDGER.exists()
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for report in reports:
            with report.open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            ok = sum(1 for r in rows if r["status"] == "ok")
            source_name = report.name.removesuffix("_translations.csv") + ".srt"
            run_cost = spent_since(source_name, since)
            total_cost = sum(_cost(r["cost_usd"]) for r in rows)
            writer.writerow(
                {
                    "logged_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "source": report.name.removesuffix("_translations.csv"),
                    "drive_folder_id": folder_id,
                    "languages_ok": ok,
                    "languages_total": len(rows),
                    "run_cost_usd": f"{run_cost:.4f}",
                    "total_cost_usd": f"{total_cost:.4f}",
                }
            )
            print(f"{ok}/{len(rows)} ok | this run ${run_cost:.4f} | total ${total_cost:.4f}")
            complete = complete and ok == len(rows)
    return 0 if complete else 3


if __name__ == "__main__":
    sys.exit(main())
