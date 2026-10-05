"""The dated archive copy of the per-source CSV."""

import csv
import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "per_source_report", Path(__file__).resolve().parent.parent / "scripts" / "per_source_report.py"
)
per_source_report = importlib.util.module_from_spec(_SPEC)
sys.modules["per_source_report"] = per_source_report
_SPEC.loader.exec_module(per_source_report)


SRT = "1\n00:00:01,000 --> 00:00:03,000\nzdravo\n\n"


@pytest.fixture
def topic(tmp_path):
    folder = tmp_path / "input_files" / "12. Neko predavanje"
    folder.mkdir(parents=True)
    (folder / "12. Neko predavanje.srt").write_text(SRT, encoding="utf-8")
    (folder / "12. Neko predavanje_German.srt").write_text(SRT, encoding="utf-8")
    return folder


def test_report_still_writes_the_csv_beside_the_source(topic):
    source = topic / "12. Neko predavanje.srt"
    target = per_source_report.report_source(source, {})
    assert target == topic / "12. Neko predavanje_translations.csv"
    assert target.exists()


def test_report_archives_a_dated_copy_under_the_run_month(topic, tmp_path):
    """A run must leave a copy filed under the month it happened.

    Petar tracks the archive by month, so the copy answers "what did I ship in
    2026-09" without needing the per-topic folders to still exist.
    """
    source = topic / "12. Neko predavanje.srt"
    archive_root = tmp_path / "csv_info_pricing"
    per_source_report.report_source(source, {}, archive_root=archive_root)

    month = date.today().strftime("%Y-%m")
    copy = archive_root / month / "12. Neko predavanje_translations.csv"
    assert copy.exists(), f"no archive copy under {month}"
    assert copy.read_text(encoding="utf-8") == (
        topic / "12. Neko predavanje_translations.csv"
    ).read_text(encoding="utf-8")


def test_the_archive_copy_carries_the_same_rows(topic, tmp_path):
    source = topic / "12. Neko predavanje.srt"
    archive_root = tmp_path / "csv_info_pricing"
    per_source_report.report_source(source, {}, archive_root=archive_root)
    copy = archive_root / date.today().strftime("%Y-%m") / "12. Neko predavanje_translations.csv"
    with copy.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    german = next(r for r in rows if r["language"] == "German")
    assert german["status"] == "ok"
    assert german["entries"] == "1"


def test_no_archive_copy_when_no_root_is_given(topic, tmp_path):
    """Callers that only want the in-place CSV must not gain a stray folder."""
    source = topic / "12. Neko predavanje.srt"
    per_source_report.report_source(source, {})
    assert not (tmp_path / "csv_info_pricing").exists()
