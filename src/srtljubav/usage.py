"""Usage JSONL logging, CSV history (per job / per run) and run reports."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from srtljubav.pricing import (
    UsageSummary,
    estimate_usage_cost_usd,
    format_duration,
    summarize_usage_records,
)

if TYPE_CHECKING:
    from srtljubav.discovery import TranslationJob


def write_usage_log(
    usage_log: Path | None,
    job: TranslationJob,
    records: list[dict[str, Any]],
    model_name: str,
    pricing_mode: str,
    service_tier: str | None = None,
) -> UsageSummary:
    summary = summarize_usage_records(
        records,
        model_name=model_name,
        pricing_mode=pricing_mode,
        service_tier=service_tier,
    )
    if not usage_log or not records:
        return summary

    usage_log.parent.mkdir(parents=True, exist_ok=True)
    with usage_log.open("a", encoding="utf-8") as log_file:
        for record in records:
            enriched = {
                **record,
                "source_file": str(job.source),
                "output_file": str(job.output_file),
                "target_language": job.target_language,
                "pricing_mode": pricing_mode,
                "service_tier": service_tier,
                "estimated_cost_usd": estimate_usage_cost_usd(
                    model_name=str(record.get("model") or model_name),
                    prompt_tokens=int(record.get("prompt_tokens") or 0),
                    output_tokens=int(record.get("output_tokens") or 0),
                    pricing_mode=pricing_mode,
                    service_tier=service_tier,
                ),
            }
            log_file.write(json.dumps(enriched, ensure_ascii=False) + "\n")
    return summary


def write_run_report(
    report_file: Path | None,
    *,
    model: str,
    pricing_mode: str,
    service_tier: str | None = None,
    workers: int,
    usage_log: Path | None,
    total_usage: UsageSummary,
    elapsed_seconds: float,
    source_count: int,
    target_language_count: int,
    job_count: int,
    failure_count: int,
) -> None:
    if not report_file:
        return

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        "\n".join(
            [
                "SRT translation run report",
                f"Generated at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
                f"Model: {model}",
                f"Pricing mode: {pricing_mode}",
                f"Service tier: {service_tier or 'account default'}",
                f"Workers: {workers}",
                f"Source SRT files: {source_count}",
                f"Target languages: {target_language_count}",
                f"Jobs requested: {job_count}",
                f"Failed jobs: {failure_count}",
                f"Duration: {format_duration(elapsed_seconds)}",
                f"Duration seconds: {elapsed_seconds:.2f}",
                f"Prompt/input tokens: {total_usage.prompt_tokens}",
                f"Output tokens: {total_usage.output_tokens}",
                f"Total tokens: {total_usage.total_tokens}",
                f"Estimated cost USD: {total_usage.estimated_cost_usd:.6f}",
                f"Usage JSONL: {usage_log}",
                "",
            ]
        ),
        encoding="utf-8",
    )


JOB_CSV_COLUMNS = [
    "timestamp",
    "source",
    "language",
    "output_file",
    "elapsed_seconds",
    "prompt_tokens",
    "output_tokens",
    "total_tokens",
    "cost_usd",
    "status",
    "error",
]

RUN_CSV_COLUMNS = [
    "timestamp",
    "model",
    "workers",
    "jobs",
    "failures",
    "elapsed_seconds",
    "prompt_tokens",
    "output_tokens",
    "total_tokens",
    "cost_usd",
]


def _append_csv_row(path: Path, columns: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_job_csv(
    path: Path | None,
    *,
    source: str,
    language: str,
    output_file: str,
    elapsed_seconds: float,
    usage: UsageSummary,
    status: str = "ok",
    error: str = "",
) -> None:
    """Append one row per finished translation job (one language)."""
    if not path:
        return
    _append_csv_row(
        path,
        JOB_CSV_COLUMNS,
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": source,
            "language": language,
            "output_file": output_file,
            "elapsed_seconds": f"{elapsed_seconds:.2f}",
            "prompt_tokens": usage.prompt_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "cost_usd": f"{usage.estimated_cost_usd:.6f}",
            "status": status,
            "error": error,
        },
    )


def append_run_csv(
    path: Path | None,
    *,
    model: str,
    workers: int,
    jobs: int,
    failures: int,
    elapsed_seconds: float,
    total_usage: UsageSummary,
) -> None:
    """Append one row per translation run (append-only history)."""
    if not path:
        return
    _append_csv_row(
        path,
        RUN_CSV_COLUMNS,
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": model,
            "workers": workers,
            "jobs": jobs,
            "failures": failures,
            "elapsed_seconds": f"{elapsed_seconds:.2f}",
            "prompt_tokens": total_usage.prompt_tokens,
            "output_tokens": total_usage.output_tokens,
            "total_tokens": total_usage.total_tokens,
            "cost_usd": f"{total_usage.estimated_cost_usd:.6f}",
        },
    )
