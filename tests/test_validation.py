from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta

import pytest
import srt

from srtljubav.validation import (
    CPS_FAIL,
    CPS_FAIL_RATIO,
    REPEAT_FAIL_RATIO,
    REPEAT_FAIL_RUN,
    EntryIssue,
    ReadabilityError,
    ValidationReport,
    bidi_balance,
    entry_cps,
    longest_repeat_run,
    validate_entries,
)

CROATIAN = "Ovo je jedna posve obicna hrvatska recenica o ljubavi."
ENGLISH = "This is a perfectly ordinary Croatian sentence about love."


def sub(index: int, start_s: float, end_s: float, content: str) -> srt.Subtitle:
    return srt.Subtitle(
        index=index,
        start=timedelta(seconds=start_s),
        end=timedelta(seconds=end_s),
        content=content,
    )


def pair(croatian: list[str], translated: list[str], duration: float = 4.0):
    """Aligned 1:1 entry lists, one entry per second-window of *duration*."""
    source = [
        sub(i, i * duration, (i + 1) * duration, text)
        for i, text in enumerate(croatian, start=1)
    ]
    output = [
        sub(i, i * duration, (i + 1) * duration, text)
        for i, text in enumerate(translated, start=1)
    ]
    return source, output


# --- layer 2: chars per second ------------------------------------------------


def test_layer2_fails_a_degenerate_entry():
    source, output = pair([CROATIAN, CROATIAN], [ENGLISH, "x " * 2000])

    report = validate_entries(source, output)

    assert not report.ok
    assert report.failed_indices == (2,)
    assert [issue.layer for issue in report.failures if issue.index == 2][0] == 2
    assert any(issue.kind == "cps" for issue in report.failures)


def test_layer2_passes_an_ordinary_translation():
    source, output = pair([CROATIAN] * 3, [ENGLISH] * 3)

    report = validate_entries(source, output)

    assert report.ok
    assert not report.failures


def test_layer2_does_not_fail_a_long_translation_of_a_long_croatian_entry():
    """The relative arm: a very fast Croatian entry makes a very fast
    translation legitimate, however far above the absolute ceiling it sits."""
    fast_croatian = "a" * 1000  # 1000 chars in 1.0 s = 1000 chars/s
    translation = "b" * 3000  # 3000 chars/s = 3x the Croatian rate

    source, output = pair([fast_croatian], [translation], duration=1.0)
    report = validate_entries(source, output)

    assert entry_cps(translation, 1.0) > CPS_FAIL
    assert entry_cps(translation, 1.0) < CPS_FAIL_RATIO * entry_cps(fast_croatian, 1.0)
    assert report.ok


def test_layer2_does_not_fail_a_slow_entry_far_above_the_croatian_rate():
    """The absolute arm: 30x the Croatian rate is fine while the entry itself
    stays readable — short Croatian entries routinely translate to 5-9x."""
    source, output = pair(["Da."], ["Yes, and that is exactly the point here."], duration=4.0)

    report = validate_entries(source, output)

    assert report.ok


# --- layer 3: repeated character n-gram ---------------------------------------


def test_layer3_fails_a_space_less_loop():
    """The real Persian defect: one entry, no spaces at all, a repeated unit."""
    source, output = pair([CROATIAN], ["می" * 500], duration=600.0)

    report = validate_entries(source, output)

    assert not report.ok
    assert report.failed_indices == (1,)
    assert [issue.layer for issue in report.failures] == [3]
    assert report.failures[0].kind == "repetition"


def test_layer3_passes_a_faithful_repetition_of_the_croatian_source():
    """The speaker imitates a film trailer; Cantonese carries it over."""
    source, output = pair(["tu " * 10], ["砰" * 10])

    report = validate_entries(source, output)

    assert report.ok


def test_layer3_does_not_fail_a_run_within_five_times_the_croatian_run():
    croatian = "tu " * 12
    translated = "砰" * 55  # >= REPEAT_FAIL_RUN but <= 5x the Croatian run

    source, output = pair([croatian], [translated], duration=60.0)
    report = validate_entries(source, output)

    assert longest_repeat_run(translated) >= REPEAT_FAIL_RUN
    assert longest_repeat_run(translated) <= REPEAT_FAIL_RATIO * longest_repeat_run(croatian)
    assert report.ok


def test_longest_repeat_run_counts_multi_character_units():
    assert longest_repeat_run("abcabcabcabc") == 4
    assert longest_repeat_run("nothing repeated here") == 1
    assert longest_repeat_run("") == 0
    assert longest_repeat_run("砰" * 7) == 7


# --- layer 4: bidi control pairing --------------------------------------------


def test_layer4_warns_on_unbalanced_bidi_controls():
    source, output = pair([CROATIAN], ["\u2066Isus\u2067 said"])

    report = validate_entries(source, output)

    assert report.ok  # bidi is mechanically repairable, it never blocks a job
    assert [issue.layer for issue in report.warnings] == [4]
    assert report.warnings[0].index == 1
    assert report.warnings[0].kind == "bidi"


def test_layer4_accepts_balanced_bidi_controls():
    source, output = pair([CROATIAN], ["\u2066Isus\u2069 \u202bكلام\u202c"])

    report = validate_entries(source, output)

    assert report.ok
    assert not report.warnings


def test_bidi_balance_counts_openers_minus_closers():
    assert bidi_balance("plain text") == 0
    assert bidi_balance("\u202a\u202c") == 0
    assert bidi_balance("\u2068\u2067\u2069") == 1
    assert bidi_balance("\u2069\u2069") == -2


# --- warnings, report shape ---------------------------------------------------


def test_cps_above_the_script_ceiling_warns_but_never_blocks():
    fast = "This line is far too fast to read comfortably on screen."

    source, output = pair([CROATIAN], [fast], duration=0.5)
    report = validate_entries(source, output)

    assert report.ok
    assert [issue.kind for issue in report.warnings] == ["cps"]
    assert "ceiling" in report.warnings[0].message


def test_report_summary_names_every_failing_entry():
    source, output = pair([CROATIAN, CROATIAN, CROATIAN], [ENGLISH, "x " * 2000, "می" * 500])

    report = validate_entries(source, output)
    summary = report.summary()

    assert report.failed_indices == (2, 3)
    assert "entry 2" in summary
    assert "entry 3" in summary
    assert "entry 1" not in summary


def test_empty_report_is_ok_and_summarizes_cleanly():
    report = ValidationReport()

    assert report.ok
    assert report.failed_indices == ()
    assert isinstance(report.summary(), str)


def test_validate_entries_requires_a_one_to_one_mapping():
    source, output = pair([CROATIAN, CROATIAN], [ENGLISH])

    with pytest.raises(RuntimeError, match="1:1"):
        validate_entries(source, output)


def test_readability_error_carries_the_report_for_per_entry_repair():
    source, output = pair([CROATIAN, CROATIAN], [ENGLISH, "x " * 2000])
    report = validate_entries(source, output)

    error = ReadabilityError("Western Punjabi", report)

    assert error.report.failed_indices == (2,)
    assert "Readability validation failed for Western Punjabi" in str(error)


# --- the error must survive the ProcessPoolExecutor the runner uses


def _raise_readability_error() -> None:
    """Worker body: what ``translate_job`` raises inside a pool process."""
    report = ValidationReport(
        failures=(EntryIssue(7, 2, "cps", "fail", "533 chars/s, 44x the Croatian entry"),),
        warnings=(EntryIssue(2, 4, "bidi", "warn", "1 unpaired opening bidi control(s)"),),
        entries_checked=12,
    )
    raise ReadabilityError("Persian (Farsi)", report)


def _echo(value: str) -> str:
    return value


def test_readability_error_crosses_a_process_pool_without_breaking_it():
    """A worker's ReadabilityError must arrive in the parent as itself.

    ``BaseException.__reduce__`` replays only the args handed to
    ``Exception.__init__``, so an unpicklable error makes the pool's result
    reader raise and every OTHER language in the run dies with it.
    """
    with ProcessPoolExecutor(max_workers=2) as executor:
        failing = executor.submit(_raise_readability_error)
        healthy = executor.submit(_echo, "Swahili")

        with pytest.raises(ReadabilityError) as excinfo:
            failing.result()
        assert healthy.result() == "Swahili", "one bad job must not break the pool"

    assert excinfo.value.target_language == "Persian (Farsi)"
    assert excinfo.value.report.failed_indices == (7,)
    assert excinfo.value.report.entries_checked == 12
    assert "Persian (Farsi)" in str(excinfo.value)
