import csv

from srtljubav.pricing import UsageSummary
from srtljubav.usage import append_job_csv, append_run_csv


def test_append_job_csv_writes_header_once(tmp_path):
    path = tmp_path / "jobs.csv"
    usage = UsageSummary(
        prompt_tokens=100, output_tokens=50, total_tokens=150, estimated_cost_usd=0.01
    )

    append_job_csv(
        path,
        source="a.srt",
        language="English",
        output_file="a_English.srt",
        elapsed_seconds=12.5,
        usage=usage,
    )
    append_job_csv(
        path,
        source="a.srt",
        language="German",
        output_file="a_German.srt",
        elapsed_seconds=0.0,
        usage=UsageSummary(),
        status="fail",
        error="boom",
    )

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["language"] == "English"
    assert rows[0]["elapsed_seconds"] == "12.50"
    assert rows[0]["status"] == "ok"
    assert rows[1]["status"] == "fail"
    assert rows[1]["error"] == "boom"
    assert path.read_text().count("timestamp") == 1  # header written once


def test_append_run_csv_round_trip(tmp_path):
    path = tmp_path / "runs.csv"
    usage = UsageSummary(
        prompt_tokens=2000, output_tokens=1200, total_tokens=3200, estimated_cost_usd=4.7
    )

    append_run_csv(
        path, model="gemini-3-flash-preview", workers=3, jobs=33, failures=0,
        elapsed_seconds=600.0, total_usage=usage,
    )

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["jobs"] == "33"
    assert rows[0]["elapsed_seconds"] == "600.00"
    assert rows[0]["cost_usd"] == "4.700000"


def test_append_csv_none_path_is_noop():
    append_job_csv(
        None, source="x", language="y", output_file="z", elapsed_seconds=1, usage=UsageSummary()
    )
    append_run_csv(
        None, model="m", workers=1, jobs=1, failures=0, elapsed_seconds=1,
        total_usage=UsageSummary(),
    )
