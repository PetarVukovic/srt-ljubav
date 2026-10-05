"""Failure handling added after the 2026-09-23 batch (14 of 266 jobs failed).

Covers: transient Gemini errors are re-sent instead of counted by gst, gst's
consecutive-error ceiling is raised, a killed or failed job never leaves a
partial file in ``input_files/``, a failed job reports what it spent, and the
runner gives failed jobs a second pass.
"""

import inspect
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
from gemini_srt_translator.main import GeminiSRTTranslator
from google.genai import errors as genai_errors

import srtljubav.pipeline as pipeline
from srtljubav.config import TranslationConfig
from srtljubav.discovery import TranslationJob
from srtljubav.pipeline import JobFailedError, JobResult, retry_config, run_job, translate_job
from srtljubav.pricing import UsageSummary
from srtljubav.translator import (
    CachedGeminiSRTTranslator,
    call_with_transient_retry,
    is_transient_error,
)

CROATIAN = "Ovo je hrvatska rečenica"
ENGLISH = "This is an English sentence"


def write_lecture(tmp_path: Path) -> TranslationJob:
    source = tmp_path / "topic" / "lecture.srt"
    source.parent.mkdir()
    source.write_text(
        "\n".join(
            f"{i + 1}\n00:00:{i:02d},000 --> 00:00:{i + 1:02d},000\n{CROATIAN} broj {i}.\n"
            for i in range(12)
        ),
        encoding="utf-8",
    )
    return TranslationJob(
        source=source,
        target_language="English",
        output_file=tmp_path / "topic" / "lecture_English.srt",
    )


def server_error() -> genai_errors.ServerError:
    return genai_errors.ServerError(
        503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}}
    )


class RecordingTranslator:
    """Stands in for CachedGeminiSRTTranslator; subclasses decide the outcome."""

    instances: list["RecordingTranslator"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.usage_records = []
        RecordingTranslator.instances.append(self)

    def write_translation(self) -> None:
        text = Path(self.kwargs["input_file"]).read_text(encoding="utf-8")
        Path(self.kwargs["output_file"]).write_text(text.replace(CROATIAN, ENGLISH), "utf-8")

    def translate(self):
        self.write_translation()


class GivesUpAfterWriting(RecordingTranslator):
    """gst's give-up path: output already on disk (a Croatian copy), then exit 130."""

    def translate(self):
        text = Path(self.kwargs["input_file"]).read_text(encoding="utf-8")
        Path(self.kwargs["output_file"]).write_text(text, encoding="utf-8")
        self.usage_records.append(
            {"model": "gemini-3-flash-preview", "prompt_tokens": 4000, "output_tokens": 2000}
        )
        raise SystemExit(130)


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


def use_translator(monkeypatch, cls: type[RecordingTranslator]) -> None:
    import srtljubav.translator as translator_module

    monkeypatch.setattr(translator_module, "CachedGeminiSRTTranslator", cls)


# --- A: transient failures are re-sent -------------------------------------


def test_a_503_is_resent_until_it_succeeds():
    outcomes = [server_error(), server_error(), "response"]
    waits: list[float] = []

    def call():
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert call_with_transient_retry(call, delays=(1.0, 2.0, 3.0), sleep=waits.append) == (
        "response"
    )
    assert waits == [1.0, 2.0]


def test_a_network_failure_is_transient_and_reraised_once_the_delays_run_out():
    waits: list[float] = []

    def call():
        raise httpx.ConnectError("[Errno 8] nodename nor servname provided")

    with pytest.raises(httpx.ConnectError):
        call_with_transient_retry(call, delays=(1.0, 2.0), sleep=waits.append)
    assert waits == [1.0, 2.0]


def test_a_quota_error_goes_straight_to_gst_which_switches_the_api_key():
    quota = genai_errors.ClientError(
        429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}
    )
    waits: list[float] = []

    def call():
        raise quota

    assert not is_transient_error(quota)
    with pytest.raises(genai_errors.ClientError):
        call_with_transient_retry(call, delays=(1.0,), sleep=waits.append)
    assert waits == []


def test_a_content_rejection_is_not_transient():
    assert not is_transient_error(ValueError("Expected 100 lines, got 99."))


# --- B: gst's consecutive-error ceiling -------------------------------------


def test_b_gst_still_hardcodes_the_ceiling_we_override():
    """The override relies on gst setting the attribute in __init__; a version
    bump that renames or parameterises it must fail here."""
    assert "self.max_consecutive_errors = 3" in inspect.getsource(GeminiSRTTranslator.__init__)


def test_b_the_translator_applies_the_configured_ceiling(monkeypatch):
    def fake_gst_init(self, *args, **kwargs):
        self.max_consecutive_errors = 3

    monkeypatch.setattr(GeminiSRTTranslator, "__init__", fake_gst_init)
    assert CachedGeminiSRTTranslator(max_consecutive_errors=6).max_consecutive_errors == 6


def test_b_translate_job_forwards_the_ceiling(tmp_path, monkeypatch, api_key):
    use_translator(monkeypatch, RecordingTranslator)
    job = write_lecture(tmp_path)

    translate_job(job, TranslationConfig(usage_log=None, clean_sections=False, yt_split=False))

    assert RecordingTranslator.instances[-1].kwargs["max_consecutive_errors"] == 6


# --- no partial output ------------------------------------------------------


def test_gst_never_writes_into_the_real_output_path(tmp_path, monkeypatch, api_key):
    use_translator(monkeypatch, RecordingTranslator)
    job = write_lecture(tmp_path)

    translate_job(job, TranslationConfig(usage_log=None, clean_sections=False, yt_split=False))

    assert Path(RecordingTranslator.instances[-1].kwargs["output_file"]) != job.output_file
    assert ENGLISH in job.output_file.read_text(encoding="utf-8")
    assert not list(job.output_file.parent.glob("*.part"))


def test_a_job_that_gives_up_leaves_no_file_behind(tmp_path, monkeypatch, api_key):
    """gst's output starts as a copy of the Croatian source; published, that copy
    would be skipped as done by every later run."""
    use_translator(monkeypatch, GivesUpAfterWriting)
    job = write_lecture(tmp_path)

    with pytest.raises(SystemExit):
        translate_job(job, TranslationConfig(usage_log=None, clean_sections=False, yt_split=False))

    assert sorted(p.name for p in job.output_file.parent.iterdir()) == ["lecture.srt"]


# --- E: a failed job reports what it spent ----------------------------------


def test_e_a_failed_job_carries_its_cost_and_logs_it(tmp_path, monkeypatch, api_key):
    use_translator(monkeypatch, GivesUpAfterWriting)
    job = write_lecture(tmp_path)
    usage_log = tmp_path / "usage.jsonl"
    cfg = TranslationConfig(usage_log=usage_log, clean_sections=False, yt_split=False)

    with pytest.raises(JobFailedError) as excinfo:
        run_job(job, cfg)

    # both rungs of the ladder (300, 20) spent tokens before giving up
    assert excinfo.value.usage.prompt_tokens == 8000
    assert excinfo.value.usage.estimated_cost_usd > 0
    assert excinfo.value.retryable
    assert len(usage_log.read_text(encoding="utf-8").splitlines()) == 2


def test_e_job_failed_error_survives_the_process_pool_boundary():
    error = JobFailedError("130", UsageSummary(prompt_tokens=5, output_tokens=7), False)

    restored = pickle.loads(pickle.dumps(error))

    assert str(restored) == "130"
    assert restored.usage == error.usage
    assert restored.retryable is False


# --- D: the second pass -----------------------------------------------------


def test_d_retry_config_uses_smaller_batches_on_the_retry_tier():
    cfg = TranslationConfig(service_tier="flex")

    second = retry_config(cfg)

    assert second is not None
    assert (second.batch_size, second.fallback_batch_sizes) == (50, (10,))
    assert second.service_tier == "standard"
    assert retry_config(TranslationConfig(retry_batch_sizes=())) is None


def test_d_failed_jobs_get_a_second_pass_and_count_both_costs(tmp_path, monkeypatch, capsys):
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / f"{name}.srt").write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nTekst.\n", encoding="utf-8"
        )
    calls: list[tuple[str, str | None]] = []

    def fake_run_job(job, cfg, key_offset=0):
        calls.append((job.source.name, cfg.service_tier))
        spent = UsageSummary(prompt_tokens=100)
        if job.source.name == "a.srt" and cfg.service_tier == "flex":
            raise JobFailedError("130", spent, retryable=True)
        return JobResult(job.target_language, str(job.output_file), spent)

    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(pipeline, "run_job", fake_run_job)
    monkeypatch.setattr(pipeline, "DEFAULT_JOBS_CSV", tmp_path / "jobs.csv")
    monkeypatch.setattr(pipeline, "DEFAULT_RUNS_CSV", tmp_path / "runs.csv")

    code = pipeline.run_translation(
        TranslationConfig(usage_log=None, service_tier="flex", yt_split=False),
        input_dir=tmp_path,
        output_root=None,
        target_languages=["English"],
        workers=1,
        max_jobs=None,
        run_report=None,
    )

    assert code == 0
    assert sorted(calls) == [("a.srt", "flex"), ("a.srt", "standard"), ("b.srt", "flex")]
    assert "Retrying 1 failed job(s)" in capsys.readouterr().out
    rows = (tmp_path / "jobs.csv").read_text(encoding="utf-8").splitlines()
    assert sum(",fail," in row for row in rows) == 1
    assert sum(",ok," in row for row in rows) == 2


def test_d_readability_failures_are_not_retried(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "a.srt").write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nTekst.\n", encoding="utf-8"
    )
    calls: list[str | None] = []

    def fake_run_job(job, cfg, key_offset=0):
        calls.append(cfg.service_tier)
        raise JobFailedError("readability", UsageSummary(), retryable=False)

    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(pipeline, "run_job", fake_run_job)
    monkeypatch.setattr(pipeline, "DEFAULT_JOBS_CSV", tmp_path / "jobs.csv")
    monkeypatch.setattr(pipeline, "DEFAULT_RUNS_CSV", tmp_path / "runs.csv")

    code = pipeline.run_translation(
        TranslationConfig(usage_log=None, service_tier="flex", yt_split=False),
        input_dir=tmp_path,
        output_root=None,
        target_languages=["English"],
        workers=1,
        max_jobs=None,
        run_report=None,
    )

    assert code == 1
    assert calls == ["flex"]
