from datetime import timedelta
from pathlib import Path

import pytest
import srt

from srtljubav import srtio
from srtljubav.config import TranslationConfig, YTConfig
from srtljubav.discovery import TranslationJob, discover_source_srts
from srtljubav.pipeline import (
    REPAIR_MAX_ENTRIES,
    _candidate_batch_sizes,
    prepare_source_text,
    reformat_existing,
    translate_job,
    validate_output_file,
    write_croatian_upload_file,
)
from srtljubav.validation import ReadabilityError, ValidationReport

LONG_TEXT = (
    "Pa će tako vjerujem biti i danas, iako to nije Objava sa kaloričnim brojem, "
    "objava je nula tri devet devet, i zato ćemo je pažljivo pročitati zajedno."
)


def write_srt(path: Path, blocks: list[tuple[str, str, str]]) -> None:
    path.write_text(
        "\n".join(
            f"{i}\n{start} --> {end}\n{text}\n"
            for i, (start, end, text) in enumerate(blocks, 1)
        ),
        encoding="utf-8",
    )


def test_validate_output_file_raises_on_count_mismatch(tmp_path):
    source = tmp_path / "src.srt"
    output = tmp_path / "out.srt"
    write_srt(
        source,
        [("00:00:01,000", "00:00:02,000", "A"), ("00:00:02,000", "00:00:03,000", "B")],
    )
    write_srt(output, [("00:00:01,000", "00:00:02,000", "X")])

    with pytest.raises(RuntimeError, match="expected 2 SRT entries, got 1"):
        validate_output_file(source, output)


def test_reformat_existing_rewraps_and_never_resegments_a_translation(tmp_path):
    """The default (and only) reformat of a translated file re-wraps its lines.

    Re-segmenting is what destroys the 1:1 mapping with ``_Croatian.srt``, and
    ``discover_translated_srts`` can only ever hand this function translations,
    so a re-segmenting default has no legitimate target at all.
    """
    path = tmp_path / "file_English.srt"
    write_srt(path, [("00:00:00,000", "00:00:12,000", LONG_TEXT)])

    first = reformat_existing([path], YTConfig())
    second = reformat_existing([path], YTConfig())

    assert first[0].changed  # the single long line really was re-wrapped
    assert first[0].entries_before == first[0].entries_after == 1
    subs = srtio.parse(srtio.read_srt_text(path))
    assert len(subs) == 1
    assert (subs[0].start, subs[0].end) == (timedelta(0), timedelta(seconds=12))
    assert first[0].overflow_indices == [1]  # 152 chars: reported, never split
    assert not second[0].changed  # idempotent: second run is a no-op


def test_reformat_rewrap_only_flag_is_an_accepted_no_op_alias(tmp_path):
    """Backwards compatibility: the flag that used to opt IN to rewrap must
    still be accepted and must change nothing."""
    default_path = tmp_path / "default_English.srt"
    flagged_path = tmp_path / "flagged_English.srt"
    for path in (default_path, flagged_path):
        write_srt(path, [("00:00:00,000", "00:00:12,000", LONG_TEXT)])

    reformat_existing([default_path], YTConfig())
    reformat_existing([flagged_path], YTConfig(), rewrap_only=True)

    assert default_path.read_text(encoding="utf-8") == flagged_path.read_text(
        encoding="utf-8"
    )


def test_reformat_dry_run_does_not_write(tmp_path):
    path = tmp_path / "file_English.srt"
    write_srt(path, [("00:00:00,000", "00:00:12,000", LONG_TEXT)])
    original = path.read_text(encoding="utf-8")

    results = reformat_existing([path], YTConfig(), dry_run=True)

    assert results[0].changed
    assert path.read_text(encoding="utf-8") == original


def test_run_translation_only_source_filters_jobs(tmp_path, capsys):
    from srtljubav.pipeline import run_translation

    for topic in ("a", "b"):
        d = tmp_path / topic
        d.mkdir()
        write_srt(d / f"{topic}.srt", [("00:00:01,000", "00:00:02,000", "Tekst.")])

    code = run_translation(
        TranslationConfig(usage_log=None),
        input_dir=tmp_path,
        output_root=None,
        target_languages=["English", "German"],
        workers=1,
        max_jobs=None,
        run_report=None,
        dry_run=True,
        only_source=tmp_path / "a" / "a.srt",
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "Source SRT files: 1" in out
    assert "Jobs to run: 2" in out
    assert "b.srt" not in out


def test_run_translation_max_jobs_caps_after_only_source_filter(tmp_path, capsys):
    """Regression: --max-jobs must cap the jobs that survive --file filtering,
    not the global pre-filter list. ``b`` sorts after ``a``; a global cap of 2
    used to consume both of ``a``'s jobs, then only_source=b filtered to 0."""
    from srtljubav.pipeline import run_translation

    for topic in ("a", "b"):
        d = tmp_path / topic
        d.mkdir()
        write_srt(d / f"{topic}.srt", [("00:00:01,000", "00:00:02,000", "Tekst.")])

    code = run_translation(
        TranslationConfig(usage_log=None),
        input_dir=tmp_path,
        output_root=None,
        target_languages=["English", "German"],
        workers=1,
        max_jobs=2,
        run_report=None,
        dry_run=True,
        only_source=tmp_path / "b" / "b.srt",
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "Source SRT files: 1" in out
    assert "Jobs to run: 2" in out  # both of b's jobs, not 0
    assert "a.srt" not in out


def test_write_croatian_upload_file_and_ignore_it_as_source(tmp_path):
    from srtljubav.pipeline import write_croatian_upload_file
    source = tmp_path / "topic" / "lecture.srt"
    source.parent.mkdir()
    write_srt(source, [("00:00:00,000", "00:00:12,000", LONG_TEXT)])

    croatian = write_croatian_upload_file(source, TranslationConfig(usage_log=None))

    assert croatian == source.with_name("lecture_Croatian.srt")
    assert len(srtio.parse(srtio.read_srt_text(croatian))) > 1
    assert discover_source_srts(tmp_path) == [source]


def test_prepare_source_text_preserves_start_of_reflowed_original_segment(tmp_path):
    from datetime import timedelta

    from srtljubav.pipeline import prepare_source_text

    source = tmp_path / "topic" / "lecture.srt"
    source.parent.mkdir()
    source.write_text(
        """513
00:51:09,850 --> 00:51:12,290
Ivana 14:6. pa smo imali dva seminara,

514
00:51:12,850 --> 00:51:14,010
pa onda...

515
00:51:16,500 --> 00:51:18,360
Dobro, ovo tu je nešto izašlo,

516
00:51:18,800 --> 00:51:20,300
netko se oglasio nečime,

517
00:51:20,300 --> 00:51:21,420
ni sam ne znam čime.

518
00:51:21,980 --> 00:51:22,740
Nešto se čulo...

519
00:51:24,040 --> 00:51:25,360
I počnem primat.

520
00:51:26,640 --> 00:51:27,840
Što počnem primat?
""",
        encoding="utf-8",
    )

    prepared = srtio.parse(prepare_source_text(source, TranslationConfig(usage_log=None)))

    segment = next(
        sub for sub in prepared if sub.content.replace("\n", " ").startswith("Nešto se čulo...")
    )
    assert segment.start == timedelta(minutes=51, seconds=21, milliseconds=980)


def test_prepare_source_text_preserves_late_question_start_after_gap(tmp_path):
    from datetime import timedelta

    from srtljubav.pipeline import prepare_source_text

    source = tmp_path / "topic" / "lecture.srt"
    source.parent.mkdir()
    source.write_text(
        """617
00:55:36,980 --> 00:55:39,340
E sad, da bi stekao iskustvo

618
00:55:39,340 --> 00:55:40,240
moraš biti kušan.

619
00:55:44,370 --> 00:55:44,830
Jeste sa mnom?
""",
        encoding="utf-8",
    )

    prepared = srtio.parse(prepare_source_text(source, TranslationConfig(usage_log=None)))

    question = next(
        sub for sub in prepared if sub.content.replace("\n", " ").startswith("Jeste sa mnom?")
    )
    assert question.start == timedelta(minutes=55, seconds=44, milliseconds=370)


def test_import_source_file_copies_and_validates(tmp_path):
    from srtljubav.cli import _import_source_file

    src = tmp_path / "ext" / "lekcija.srt"
    src.parent.mkdir()
    write_srt(src, [("00:00:01,000", "00:00:02,000", "Tekst.")])
    input_dir = tmp_path / "input_files"

    dest = _import_source_file(src, input_dir)

    assert dest == input_dir / "lekcija" / "lekcija.srt"
    assert dest.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")
    # idempotent: second call does not fail
    assert _import_source_file(src, input_dir) == dest

    bad = tmp_path / "ext" / "lekcija_English.srt"
    write_srt(bad, [("00:00:01,000", "00:00:02,000", "X")])
    with pytest.raises(ValueError, match="translation suffix"):
        _import_source_file(bad, input_dir)
    with pytest.raises(ValueError, match="does not exist"):
        _import_source_file(tmp_path / "nema.srt", input_dir)


def seq_blocks(texts: list[str]) -> list[tuple[str, str, str]]:
    return [
        (
            f"00:{i // 60:02d}:{i % 60:02d},000",
            f"00:{(i + 1) // 60:02d}:{(i + 1) % 60:02d},000",
            text,
        )
        for i, text in enumerate(texts)
    ]


def write_pair(tmp_path, src_texts: list[str], out_texts: list[str]):
    source = tmp_path / "src.srt"
    output = tmp_path / "out.srt"
    write_srt(source, seq_blocks(src_texts))
    write_srt(output, seq_blocks(out_texts))
    return source, output


CRO_TEXTS = [f"Ovo je hrvatska rečenica broj {i} u izvornome titlu." for i in range(40)]
ENG_TEXTS = [f"This is English sentence number {i} in the subtitle." for i in range(40)]


def test_validate_translated_content_passes_on_clean_translation(tmp_path):
    from srtljubav.pipeline import validate_translated_content

    source, output = write_pair(tmp_path, CRO_TEXTS, ENG_TEXTS)
    validate_translated_content(source, output, "English", TranslationConfig(usage_log=None))


def test_validate_translated_content_allows_two_untranslated_entries(tmp_path):
    from srtljubav.pipeline import validate_translated_content

    leaked = ENG_TEXTS[:10] + CRO_TEXTS[10:12] + ENG_TEXTS[12:]
    source, output = write_pair(tmp_path, CRO_TEXTS, leaked)

    validate_translated_content(source, output, "English", TranslationConfig(usage_log=None))


def test_validate_translated_content_catches_three_untranslated_entries(tmp_path):
    from srtljubav.pipeline import validate_translated_content

    leaked = CRO_TEXTS[:3] + ENG_TEXTS[3:]
    source, output = write_pair(tmp_path, CRO_TEXTS, leaked)

    with pytest.raises(RuntimeError, match="3 consecutive"):
        validate_translated_content(source, output, "English", TranslationConfig(usage_log=None))


def test_validate_translated_content_catches_scattered_leakage_by_ratio(tmp_path):
    from srtljubav.pipeline import validate_translated_content

    leaked = [CRO_TEXTS[i] if i % 2 == 0 else ENG_TEXTS[i] for i in range(40)]
    source, output = write_pair(tmp_path, CRO_TEXTS, leaked)

    with pytest.raises(RuntimeError, match="20/40"):
        validate_translated_content(
            source, output, "English", TranslationConfig(usage_log=None)
        )


def test_validate_translated_content_ignores_trivial_identical_entries(tmp_path):
    from srtljubav.pipeline import validate_translated_content

    trivial = ["Ivan 14:15", "123", "Amen.", "8590"] * 8  # identical in both files
    src_texts = CRO_TEXTS[:25] + trivial
    out_texts = ENG_TEXTS[:25] + trivial
    source, output = write_pair(tmp_path, src_texts, out_texts)

    validate_translated_content(source, output, "English", TranslationConfig(usage_log=None))


def test_validate_translated_content_checks_expected_script(tmp_path):
    from srtljubav.pipeline import validate_translated_content

    latin_serbian = [f"Ovo je srpska recenica broj {i} u prevodu titla." for i in range(40)]
    cyrillic_serbian = [f"Ово је српска реченица број {i} у преводу титла." for i in range(40)]
    cfg = TranslationConfig(usage_log=None)

    source, output = write_pair(tmp_path, CRO_TEXTS, latin_serbian)
    with pytest.raises(RuntimeError, match="dominant script"):
        validate_translated_content(source, output, "Serbian (on Српски!)", cfg)

    source, output = write_pair(tmp_path, CRO_TEXTS, cyrillic_serbian)
    validate_translated_content(source, output, "Serbian (on Српски!)", cfg)


def test_validate_translated_content_skips_croatian_target(tmp_path):
    from srtljubav.pipeline import validate_translated_content

    source, output = write_pair(tmp_path, CRO_TEXTS, CRO_TEXTS)
    validate_translated_content(source, output, "Croatian", TranslationConfig(usage_log=None))


class FakeTranslator:
    """Stands in for CachedGeminiSRTTranslator: copies input to output 1:1."""

    instances: list["FakeTranslator"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.input_text = ""
        self.usage_records = []
        FakeTranslator.instances.append(self)

    def translate(self):
        self.input_text = Path(self.kwargs["input_file"]).read_text(encoding="utf-8")
        Path(self.kwargs["output_file"]).write_text(self.input_text, encoding="utf-8")


class Exit130ThenTranslate(FakeTranslator):
    attempts = 0

    def translate(self):
        type(self).attempts += 1
        if type(self).attempts == 1:
            raise SystemExit(130)
        self.input_text = Path(self.kwargs["input_file"]).read_text(encoding="utf-8")
        translated = self.input_text.replace(
            "Ovo je hrvatska rečenica", "This is an English sentence"
        )
        Path(self.kwargs["output_file"]).write_text(translated, encoding="utf-8")


def test_translate_job_recovers_from_gst_exit_130(tmp_path, monkeypatch):
    import srtljubav.translator as translator_module

    Exit130ThenTranslate.attempts = 0
    monkeypatch.setattr(translator_module, "CachedGeminiSRTTranslator", Exit130ThenTranslate)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    source = tmp_path / "topic" / "lecture.srt"
    source.parent.mkdir()
    write_srt(source, seq_blocks([f"Ovo je hrvatska rečenica broj {i}." for i in range(12)]))
    job = TranslationJob(
        source=source,
        target_language="English",
        output_file=tmp_path / "topic" / "lecture_English.srt",
    )

    instances_before = len(FakeTranslator.instances)
    result = translate_job(
        job,
        TranslationConfig(usage_log=None, clean_sections=False, yt_split=False),
    )

    assert result.language == "English"
    assert Exit130ThenTranslate.attempts == 2
    attempted = FakeTranslator.instances[instances_before:]
    assert [item.kwargs["batch_size"] for item in attempted] == [300, 20]


def test_translate_job_validates_then_rewraps(tmp_path, monkeypatch):
    import srtljubav.translator as translator_module

    monkeypatch.setattr(translator_module, "CachedGeminiSRTTranslator", FakeTranslator)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    source = tmp_path / "topic" / "lecture.srt"
    source.parent.mkdir()
    write_srt(source, [("00:00:00,000", "00:00:12,000", LONG_TEXT)])
    job = TranslationJob(
        source=source,
        target_language="English",
        output_file=tmp_path / "topic" / "lecture_English.srt",
    )

    cfg = TranslationConfig(usage_log=None, clean_sections=True, yt_split=True)
    result = translate_job(job, cfg)

    input_subs = srtio.parse(FakeTranslator.instances[-1].input_text)
    assert len(input_subs) > 1
    for sub in input_subs:
        for line in sub.content.split("\n"):
            assert len(line) <= 42

    output_subs = srtio.parse(srtio.read_srt_text(job.output_file))
    # the single 152-char entry was reflowed into multiple YT-friendly entries
    # in the SOURCE; the output only re-wraps them, staying 1:1 with it
    assert len(output_subs) > 1
    assert [(s.index, s.start, s.end) for s in output_subs] == [
        (s.index, s.start, s.end) for s in input_subs
    ]
    for sub in output_subs:
        for line in sub.content.split("\n"):
            assert len(line) <= 42
    assert result.language == "English"
    assert result.elapsed_seconds > 0
    # resume must stay disabled (temp-dir scoping intentionally prevents resume)
    assert FakeTranslator.instances[-1].kwargs["resume"] is False


def test_translate_job_retries_then_fails_loudly_and_removes_bad_output(tmp_path, monkeypatch):
    """An echo 'translation' (output identical to the Croatian source) must fail
    content validation, be retried with a fresh translator, and leave NO output
    file behind — otherwise discovery would treat the broken file as done."""
    import srtljubav.translator as translator_module

    monkeypatch.setattr(translator_module, "CachedGeminiSRTTranslator", FakeTranslator)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    source = tmp_path / "topic" / "lecture.srt"
    source.parent.mkdir()
    write_srt(source, seq_blocks([f"Ovo je hrvatska rečenica broj {i}." for i in range(12)]))
    job = TranslationJob(
        source=source,
        target_language="English",
        output_file=tmp_path / "topic" / "lecture_English.srt",
    )
    cfg = TranslationConfig(usage_log=None, clean_sections=False, yt_split=False)

    instances_before = len(FakeTranslator.instances)
    with pytest.raises(RuntimeError, match="untranslated"):
        translate_job(job, cfg)

    attempted = FakeTranslator.instances[instances_before:]
    assert len(attempted) == cfg.translate_attempts * len(_candidate_batch_sizes(cfg))
    assert [item.kwargs["batch_size"] for item in attempted] == [300, 20]
    assert not job.output_file.exists()


def test_translate_job_rotates_api_keys_by_offset(tmp_path, monkeypatch):
    """Parallel jobs alternate the primary key (offset 0 -> key1, 1 -> key2);
    the other key always stays available as the quota-switch fallback."""
    import srtljubav.translator as translator_module

    monkeypatch.setattr(translator_module, "CachedGeminiSRTTranslator", FakeTranslator)
    monkeypatch.setenv("GEMINI_API_KEY", "key-one")
    monkeypatch.setenv("GEMINI_API_KEY2", "key-two")

    source = tmp_path / "topic" / "lecture.srt"
    source.parent.mkdir()
    write_srt(source, [("00:00:00,000", "00:00:12,000", LONG_TEXT)])
    job = TranslationJob(
        source=source,
        target_language="English",
        output_file=tmp_path / "topic" / "lecture_English.srt",
    )
    cfg = TranslationConfig(usage_log=None)

    translate_job(job, cfg, key_offset=0)
    assert FakeTranslator.instances[-1].kwargs["gemini_api_key"] == "key-one"
    assert FakeTranslator.instances[-1].kwargs["gemini_api_key2"] == "key-two"

    job.output_file.unlink()
    translate_job(job, cfg, key_offset=1)
    assert FakeTranslator.instances[-1].kwargs["gemini_api_key"] == "key-two"
    assert FakeTranslator.instances[-1].kwargs["gemini_api_key2"] == "key-one"


def test_translate_job_appends_script_hint_for_non_latin_targets(tmp_path, monkeypatch):
    import srtljubav.translator as translator_module

    monkeypatch.setattr(translator_module, "CachedGeminiSRTTranslator", FakeTranslator)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    source = tmp_path / "topic" / "lecture.srt"
    source.parent.mkdir()
    write_srt(source, seq_blocks([f"Ovo je hrvatska rečenica broj {i}." for i in range(12)]))
    job = TranslationJob(
        source=source,
        target_language="Serbian (on Српски!)",
        output_file=tmp_path / "topic" / "lecture_Serbian.srt",
    )
    cfg = TranslationConfig(usage_log=None, clean_sections=False, yt_split=False)

    with pytest.raises(RuntimeError):  # echo output also fails script validation
        translate_job(job, cfg)

    assert "Cyrillic" in FakeTranslator.instances[-1].kwargs["description"]


def test_translate_job_forwards_service_tier_and_records_it(tmp_path, monkeypatch):
    """Flex halves the bill, so the tier must reach both the request and the log;
    an unset tier stays None so Google applies the account default."""
    import json

    import srtljubav.translator as translator_module

    monkeypatch.setattr(translator_module, "CachedGeminiSRTTranslator", FakeTranslator)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    source = tmp_path / "topic" / "lecture.srt"
    source.parent.mkdir()
    write_srt(source, [("00:00:00,000", "00:00:12,000", LONG_TEXT)])
    job = TranslationJob(
        source=source,
        target_language="English",
        output_file=tmp_path / "topic" / "lecture_English.srt",
    )

    translate_job(job, TranslationConfig(usage_log=None), key_offset=0)
    assert FakeTranslator.instances[-1].kwargs["service_tier"] is None

    class UsageReportingTranslator(FakeTranslator):
        def translate(self):
            super().translate()
            self.usage_records.append(
                {
                    "model": "gemini-3-flash-preview",
                    "prompt_tokens": 1000,
                    "output_tokens": 500,
                    "total_tokens": 1500,
                }
            )

    monkeypatch.setattr(
        translator_module, "CachedGeminiSRTTranslator", UsageReportingTranslator
    )

    job.output_file.unlink()
    usage_log = tmp_path / "usage.jsonl"
    translate_job(
        job,
        TranslationConfig(usage_log=usage_log, service_tier="flex"),
        key_offset=0,
    )

    assert FakeTranslator.instances[-1].kwargs["service_tier"] == "flex"
    records = [
        json.loads(line)
        for line in usage_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records, "the usage log must record the tier the run was billed at"
    assert all(record["service_tier"] == "flex" for record in records)


def test_prepare_source_text_repairs_an_impossible_source_window(tmp_path):
    """Reproduces the real defect: source entry #206 of "Stvari koje nisu..."
    packs 420 characters into 1.94 s (216 cps) and is followed by 27 s of
    silence. The prepared Croatian must be readable and must stay in sync."""
    body = "Podsvijest je pod velikim utjecajem onoga sto stalno govorite ili mislite. "
    long_text = (body * 6)[:420]
    source = tmp_path / "lecture.srt"
    write_srt(
        source,
        [
            ("00:37:44,000", "00:37:46,490", "Prethodna recenica koja traje dovoljno dugo."),
            ("00:37:46,490", "00:37:48,430", long_text),
            ("00:38:15,790", "00:38:20,000", "Sljedeca recenica nakon duge tisine."),
        ],
    )

    prepared = srtio.parse(prepare_source_text(source, TranslationConfig(usage_log=None)))

    worst = max(
        len(" ".join(entry.content.split()))
        / max((entry.end - entry.start).total_seconds(), 1e-9)
        for entry in prepared
    )
    assert worst <= 40.0, f"prepared Croatian still unreadable at {worst:.0f} chars/s"
    assert all(
        (entry.end - entry.start).total_seconds() >= 0.5 for entry in prepared
    ), "no entry may sit under YouTube's hard floor"
    lead_in = timedelta(minutes=37, seconds=44) - prepared[0].start
    assert timedelta(0) <= lead_in <= timedelta(seconds=0.25), "a start may only lead in"
    for earlier, later in zip(prepared, prepared[1:], strict=False):
        assert earlier.end <= later.start, "repair must never create an overlap"


def test_prepare_source_text_is_idempotent_across_the_repair(tmp_path):
    source = tmp_path / "lecture.srt"
    write_srt(
        source,
        [
            ("00:00:00,000", "00:00:01,940", "x" * 200),
            ("00:00:30,000", "00:00:34,000", "Sljedeca recenica nakon tisine."),
        ],
    )
    cfg = TranslationConfig(usage_log=None)

    once = prepare_source_text(source, cfg)
    repeated = tmp_path / "again.srt"
    repeated.write_text(once, encoding="utf-8")

    assert prepare_source_text(repeated, cfg) == once


def test_prepare_source_text_repairs_windows_the_anchored_reflow_reintroduces(tmp_path):
    """The anchored reflow pins chunk starts back onto the original fragment
    starts, so a short fragment that cleaning had absorbed into a longer block
    reappears with its original 0.48 s window. Measured across the archive this
    costs 61 entries under 1.0 s and 4 under YouTube's hard floor, so the repair
    has to run after the reflow as well as before it."""
    wide = "Prva duga recenica koja sama zauzima gotovo dvije pune linije teksta na ekranu."
    later = "Treca duga recenica koja dolazi tek nakon dvadeset sekundi potpune tisine ovdje."
    source = tmp_path / "lecture.srt"
    write_srt(
        source,
        [
            ("00:00:00,000", "00:00:05,000", wide),
            ("00:00:05,000", "00:00:05,480", "Znate."),
            ("00:00:25,000", "00:00:32,000", later),
        ],
    )

    prepared = srtio.parse(prepare_source_text(source, TranslationConfig(usage_log=None)))

    short = next(entry for entry in prepared if entry.content.strip() == "Znate.")
    assert (short.end - short.start).total_seconds() >= 1.0
    assert short.start == timedelta(seconds=5), "the repair must not move the start"
    for earlier, later_entry in zip(prepared, prepared[1:], strict=False):
        assert earlier.end <= later_entry.start


# --- rewrap-only reformat: translated files stay 1:1 with their Croatian sibling


def test_reformat_rewrap_only_keeps_the_entry_count_and_timestamps(tmp_path):
    path = tmp_path / "file_English.srt"
    write_srt(path, [("00:00:00,000", "00:00:12,000", LONG_TEXT)])

    results = reformat_existing([path], YTConfig(), rewrap_only=True)

    assert results[0].entries_before == 1
    assert results[0].entries_after == 1
    subs = srtio.parse(srtio.read_srt_text(path))
    assert len(subs) == 1
    assert subs[0].start == timedelta(0)
    assert subs[0].end == timedelta(seconds=12)
    # 152 chars cannot fit two 42-char lines: reported, never split
    assert results[0].overflow_indices == [1]


def test_reformat_rewrap_only_dry_run_reports_without_writing(tmp_path):
    path = tmp_path / "file_English.srt"
    write_srt(
        path,
        [
            ("00:00:00,000", "00:00:04,000", "Kratko."),
            ("00:00:04,000", "00:00:08,000", LONG_TEXT),
        ],
    )
    original = path.read_text(encoding="utf-8")

    results = reformat_existing([path], YTConfig(), dry_run=True, rewrap_only=True)

    assert results[0].changed
    assert results[0].entries_before == results[0].entries_after == 2
    assert results[0].overflow_indices == [2]
    assert path.read_text(encoding="utf-8") == original


def test_reformat_rewrap_only_reports_files_that_are_not_one_to_one(tmp_path):
    write_srt(tmp_path / "lecture_Croatian.srt", seq_blocks(["Prva.", "Druga."]))
    aligned = tmp_path / "lecture_English.srt"
    write_srt(aligned, seq_blocks(["First.", "Second."]))
    short = tmp_path / "lecture_German.srt"
    write_srt(short, seq_blocks(["Erste."]))
    shifted = tmp_path / "lecture_Italian.srt"
    write_srt(
        shifted,
        [("00:00:00,000", "00:00:01,000", "Prima."), ("00:00:02,500", "00:00:03,000", "Seconda.")],
    )
    orphan = tmp_path / "other_English.srt"
    write_srt(orphan, seq_blocks(["Alone."]))

    results = reformat_existing(
        [aligned, short, shifted, orphan], YTConfig(), dry_run=True, rewrap_only=True
    )

    assert results[0].croatian_mismatch is None
    assert results[1].croatian_mismatch is not None and "1 entries" in results[1].croatian_mismatch
    assert results[2].croatian_mismatch is not None
    assert "timestamp" in results[2].croatian_mismatch
    assert results[3].croatian is None  # nothing to compare against


class LongerTranslator(FakeTranslator):
    """Synthetic 'German': same entry count, ~40 % longer text per entry."""

    def translate(self):
        self.input_text = Path(self.kwargs["input_file"]).read_text(encoding="utf-8")
        subs = srtio.parse(self.input_text)
        translated = [
            srt.Subtitle(
                index=entry.index,
                start=entry.start,
                end=entry.end,
                content=(
                    "Uebersetzung: " + entry.content.replace("\n", " ") + " und noch etwas laenger"
                ),
            )
            for entry in subs
        ]
        Path(self.kwargs["output_file"]).write_text(srtio.compose(translated), encoding="utf-8")


def test_translate_job_output_is_one_to_one_with_the_croatian_sibling(tmp_path, monkeypatch):
    """The done criterion of the rewrap step, proven on the code path: a longer
    'translation' of a prepared Croatian source keeps its entry count, indices
    and timestamps, so the two files diff side by side."""
    import srtljubav.translator as translator_module

    monkeypatch.setattr(translator_module, "CachedGeminiSRTTranslator", LongerTranslator)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    source = tmp_path / "topic" / "lecture.srt"
    source.parent.mkdir()
    write_srt(
        source,
        [
            ("00:00:00,000", "00:00:12,000", LONG_TEXT),
            ("00:00:12,000", "00:00:24,000", LONG_TEXT),
            ("00:00:24,000", "00:00:27,000", "Kratka rečenica na kraju."),
        ],
    )
    cfg = TranslationConfig(usage_log=None, clean_sections=True, yt_split=True)
    croatian = write_croatian_upload_file(source, cfg)

    job = TranslationJob(
        source=source,
        target_language="German",
        output_file=tmp_path / "topic" / "lecture_German.srt",
    )
    translate_job(job, cfg)

    croatian_subs = srtio.parse(srtio.read_srt_text(croatian))
    german_subs = srtio.parse(srtio.read_srt_text(job.output_file))
    assert len(croatian_subs) > 3  # the source really was reflowed
    assert [(s.index, s.start, s.end) for s in german_subs] == [
        (s.index, s.start, s.end) for s in croatian_subs
    ]


class DegenerateTranslator(FakeTranslator):
    """Stands in for a Gemini collapse: one entry loops, the rest are fine.

    The loop is keyed on the CONTENT of the Croatian entry, not on its position,
    so it reappears in the ``batch_size=1`` repair pass too -- this fake models
    the entry that per-entry repair cannot rescue.
    """

    def translate(self):
        self.input_text = Path(self.kwargs["input_file"]).read_text(encoding="utf-8")
        subs = srtio.parse(self.input_text)
        translated = [
            srt.Subtitle(
                index=entry.index,
                start=entry.start,
                end=entry.end,
                content="x" * 20000
                if "broj 4." in entry.content
                else f"English line {entry.index}.",
            )
            for entry in subs
        ]
        Path(self.kwargs["output_file"]).write_text(srtio.compose(translated), encoding="utf-8")


def test_translate_job_fails_on_a_degenerate_entry_and_removes_the_output(
    tmp_path, monkeypatch
):
    """Layers 2-4 run on the RAW translator output, while it is still 1:1 with
    the prepared source. Repair gets its two shots at the looping entry; when it
    still loops the job fails and leaves no file behind, since discovery would
    treat it as done."""
    import srtljubav.translator as translator_module

    monkeypatch.setattr(translator_module, "CachedGeminiSRTTranslator", DegenerateTranslator)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    source = tmp_path / "topic" / "lecture.srt"
    source.parent.mkdir()
    write_srt(source, seq_blocks([f"Ovo je hrvatska rečenica broj {i}." for i in range(12)]))
    job = TranslationJob(
        source=source,
        target_language="English",
        output_file=tmp_path / "topic" / "lecture_English.srt",
    )
    cfg = TranslationConfig(usage_log=None, clean_sections=False, yt_split=False)

    with pytest.raises(RuntimeError, match="Readability validation failed") as excinfo:
        translate_job(job, cfg)

    # the repair worklist Step 4 consumes: entry 5 alone, not the whole file
    assert excinfo.value.report.failed_indices == (5,)
    assert not job.output_file.exists()


def test_translate_job_carries_the_readability_report_on_success(tmp_path, monkeypatch):
    """A healthy job never blocks, but its warnings reach the run report."""
    import srtljubav.translator as translator_module

    class FastTranslator(FakeTranslator):
        def translate(self):
            self.input_text = Path(self.kwargs["input_file"]).read_text(encoding="utf-8")
            subs = srtio.parse(self.input_text)
            translated = [
                srt.Subtitle(
                    index=entry.index,
                    start=entry.start,
                    end=entry.end,
                    content="A very much longer English rendering of this line, " * 2,
                )
                for entry in subs
            ]
            Path(self.kwargs["output_file"]).write_text(
                srtio.compose(translated), encoding="utf-8"
            )

    monkeypatch.setattr(translator_module, "CachedGeminiSRTTranslator", FastTranslator)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    source = tmp_path / "topic" / "lecture.srt"
    source.parent.mkdir()
    write_srt(source, seq_blocks([f"Ovo je hrvatska rečenica broj {i}." for i in range(12)]))
    job = TranslationJob(
        source=source,
        target_language="English",
        output_file=tmp_path / "topic" / "lecture_English.srt",
    )

    result = translate_job(
        job, TranslationConfig(usage_log=None, clean_sections=False, yt_split=False)
    )

    assert result.validation.ok
    assert result.validation.warnings  # 102 chars in 1.0 s is above the Latin ceiling
    assert job.output_file.exists()


# --- Step 4: repair the failing entries instead of re-translating the lecture


def repair_blocks(count: int) -> list[tuple[str, str, str]]:
    """*count* comfortable 3 s Croatian entries (11 chars/s), so nothing but a
    deliberately degenerate translation can fail readability validation."""
    return [
        (
            f"00:{(3 * i) // 60:02d}:{(3 * i) % 60:02d},000",
            f"00:{(3 * i + 3) // 60:02d}:{(3 * i + 3) % 60:02d},000",
            f"Ovo je hrvatska rečenica broj {i}.",
        )
        for i in range(count)
    ]


# 1600 characters in a 3 s window: 533 chars/s (>500 and >10x the Croatian
# entry) and a unit repeated 800x (>=50 and >5x), so layers 2 and 3 both fail.
LOOP_TEXT = "mi" * 800


class LoopingTranslator(FakeTranslator):
    """Fake Gemini that loops on chosen entries instead of translating them.

    ``marked`` holds 1-based positions in the file handed to the FIRST pass, so
    exactly those entries come back degenerate. Every later call is the
    ``batch_size=1`` repair pass, which translates cleanly unless
    ``fail_repair`` is set. Prefixes let a test see which pass wrote an entry.
    """

    marked: tuple[int, ...] = ()
    fail_repair = False
    calls = 0

    def translate(self):
        type(self).calls += 1
        first_pass = type(self).calls == 1
        self.input_text = Path(self.kwargs["input_file"]).read_text(encoding="utf-8")
        prefix = "[EN]" if first_pass else "[FIX]"
        translated = []
        for position, sub in enumerate(srtio.parse(self.input_text), start=1):
            broken = (
                position in type(self).marked if first_pass else type(self).fail_repair
            )
            translated.append(
                srt.Subtitle(
                    index=sub.index,
                    start=sub.start,
                    end=sub.end,
                    content=LOOP_TEXT if broken else f"{prefix} {sub.content}",
                )
            )
        Path(self.kwargs["output_file"]).write_text(
            srtio.compose(translated), encoding="utf-8"
        )


def _repair_job(tmp_path, monkeypatch, *, entries: int, marked, fail_repair=False):
    import srtljubav.translator as translator_module

    LoopingTranslator.calls = 0
    LoopingTranslator.marked = tuple(marked)
    LoopingTranslator.fail_repair = fail_repair
    monkeypatch.setattr(
        translator_module, "CachedGeminiSRTTranslator", LoopingTranslator
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    source = tmp_path / "topic" / "lecture.srt"
    source.parent.mkdir()
    write_srt(source, repair_blocks(entries))
    job = TranslationJob(
        source=source,
        target_language="English",
        output_file=tmp_path / "topic" / "lecture_English.srt",
    )
    cfg = TranslationConfig(usage_log=None, clean_sections=False, yt_split=False)
    return source, job, cfg


def test_too_many_failing_entries_fail_the_job_and_delete_the_output(
    tmp_path, monkeypatch
):
    """Above REPAIR_MAX_ENTRIES the whole translation is bad: no repair call, no
    re-translation, and above all no output file — discovery treats any existing
    file as done and would skip the lecture forever."""
    source, job, cfg = _repair_job(
        tmp_path,
        monkeypatch,
        entries=30,
        marked=range(1, REPAIR_MAX_ENTRIES + 2),
    )

    instances_before = len(FakeTranslator.instances)
    with pytest.raises(ReadabilityError) as excinfo:
        translate_job(job, cfg)

    assert len(excinfo.value.report.failed_indices) == REPAIR_MAX_ENTRIES + 1
    assert not job.output_file.exists()
    attempted = FakeTranslator.instances[instances_before:]
    assert len(attempted) == 1, "no repair and no retry above the repair ceiling"


def test_repair_builds_a_mini_srt_of_exactly_the_failing_entries(
    tmp_path, monkeypatch
):
    source, job, cfg = _repair_job(tmp_path, monkeypatch, entries=8, marked=(3, 5))

    instances_before = len(FakeTranslator.instances)
    translate_job(job, cfg)

    attempted = FakeTranslator.instances[instances_before:]
    assert len(attempted) == 2, "one full pass, then one repair pass"
    repair = attempted[1]
    assert repair.kwargs["batch_size"] == 1
    assert repair.kwargs["resume"] is False

    source_subs = srtio.parse(srtio.read_srt_text(source))
    mini = srtio.parse(repair.input_text)
    assert [entry.content for entry in mini] == [
        source_subs[2].content,
        source_subs[4].content,
    ]
    assert [(entry.start, entry.end) for entry in mini] == [
        (source_subs[2].start, source_subs[2].end),
        (source_subs[4].start, source_subs[4].end),
    ]


def test_repair_splices_text_back_at_the_original_indices(tmp_path, monkeypatch):
    source, job, cfg = _repair_job(tmp_path, monkeypatch, entries=8, marked=(3, 5))

    result = translate_job(job, cfg)

    output = srtio.parse(srtio.read_srt_text(job.output_file))
    source_subs = srtio.parse(srtio.read_srt_text(source))
    for position, entry in enumerate(output, start=1):
        expected_prefix = "[FIX]" if position in (3, 5) else "[EN]"
        assert entry.content == f"{expected_prefix} {source_subs[position - 1].content}"
    assert result.validation.ok


def test_repair_changes_neither_the_entry_count_nor_the_timestamps(
    tmp_path, monkeypatch
):
    source, job, cfg = _repair_job(tmp_path, monkeypatch, entries=8, marked=(3, 5))

    translate_job(job, cfg)

    source_subs = srtio.parse(srtio.read_srt_text(source))
    output = srtio.parse(srtio.read_srt_text(job.output_file))
    assert [(s.index, s.start, s.end) for s in output] == [
        (s.index, s.start, s.end) for s in source_subs
    ]


def test_repair_that_still_fails_is_retried_once_then_fails_the_job(
    tmp_path, monkeypatch
):
    source, job, cfg = _repair_job(
        tmp_path, monkeypatch, entries=8, marked=(3,), fail_repair=True
    )

    instances_before = len(FakeTranslator.instances)
    with pytest.raises(ReadabilityError):
        translate_job(job, cfg)

    attempted = FakeTranslator.instances[instances_before:]
    assert [item.kwargs["batch_size"] for item in attempted] == [300, 1, 1], (
        "one full pass and exactly two repair attempts, never a re-translation"
    )
    assert not job.output_file.exists()


# --- Step 1 changed the segmentation, so _Croatian.srt is a deliberate output


def test_run_translation_with_nothing_to_do_never_touches_the_croatian_file(
    tmp_path, capsys
):
    """`uv run srt-ljubav translate` on a finished archive must be a no-op.

    Rewriting `_Croatian.srt` before the job list is even consulted silently
    re-segments it against the 33 translations already shipped beside it.
    """
    from srtljubav.pipeline import run_translation

    topic = tmp_path / "topic"
    topic.mkdir()
    source = topic / "lecture.srt"
    write_srt(source, [("00:00:00,000", "00:00:12,000", LONG_TEXT)])
    croatian = topic / "lecture_Croatian.srt"
    write_srt(croatian, [("00:00:00,000", "00:00:12,000", LONG_TEXT)])
    write_srt(topic / "lecture_English.srt", [("00:00:00,000", "00:00:12,000", "Shipped.")])
    before = croatian.read_text(encoding="utf-8")

    code = run_translation(
        TranslationConfig(usage_log=None),
        input_dir=tmp_path,
        output_root=None,
        target_languages=["English"],
        workers=1,
        max_jobs=None,
        run_report=None,
    )

    assert code == 0
    assert "Nothing to translate" in capsys.readouterr().out
    assert croatian.read_text(encoding="utf-8") == before


def test_a_source_with_no_job_in_this_run_is_left_alone_entirely(tmp_path):
    """A run that translates nothing for this source must not touch its Croatian
    file — not even to create one. Writing it is the deliberate output of a job,
    never a drive-by side effect of discovery."""
    from srtljubav.pipeline import plan_croatian_upload, write_croatian_upload_files

    topic = tmp_path / "topic"
    topic.mkdir()
    source = topic / "lecture.srt"
    write_srt(source, [("00:00:00,000", "00:00:12,000", LONG_TEXT)])
    croatian = topic / "lecture_Croatian.srt"
    cfg = TranslationConfig(usage_log=None)

    plan = plan_croatian_upload(source, cfg, languages_in_run=[])

    assert plan.write is False
    assert plan.reason == "no translation job for this source"
    assert write_croatian_upload_files([source], [], cfg) == [plan]
    assert not croatian.exists()


def test_croatian_is_not_regenerated_when_it_would_desync_shipped_translations(tmp_path):
    """A run that translates ONE missing language must not re-segment the
    Croatian file the other 32 shipped translations are aligned to. That
    re-derivation is `scripts/repair_archive.py`'s explicit, reported job."""
    from srtljubav.pipeline import plan_croatian_upload

    topic = tmp_path / "topic"
    topic.mkdir()
    source = topic / "lecture.srt"
    write_srt(source, [("00:00:00,000", "00:00:12,000", LONG_TEXT)])
    croatian = topic / "lecture_Croatian.srt"
    write_srt(croatian, [("00:00:00,000", "00:00:12,000", LONG_TEXT)])
    write_srt(topic / "lecture_German.srt", [("00:00:00,000", "00:00:12,000", "Deutsch.")])
    before = croatian.read_text(encoding="utf-8")

    plan = plan_croatian_upload(
        source, TranslationConfig(usage_log=None), languages_in_run=["English"]
    )

    assert plan.write is False
    assert plan.at_risk == (topic / "lecture_German.srt",)
    assert plan.reason is not None
    assert "repair_archive" in plan.reason
    assert croatian.read_text(encoding="utf-8") == before


def test_croatian_is_regenerated_when_every_translation_is_being_redone(tmp_path):
    """--overwrite re-translates against the freshly prepared source, so
    re-deriving the segmentation desynchronises nothing."""
    from srtljubav.pipeline import plan_croatian_upload

    topic = tmp_path / "topic"
    topic.mkdir()
    source = topic / "lecture.srt"
    write_srt(source, [("00:00:00,000", "00:00:12,000", LONG_TEXT)])
    write_srt(topic / "lecture_Croatian.srt", [("00:00:00,000", "00:00:12,000", LONG_TEXT)])
    write_srt(topic / "lecture_German.srt", [("00:00:00,000", "00:00:12,000", "Deutsch.")])

    plan = plan_croatian_upload(
        source,
        TranslationConfig(usage_log=None),
        languages_in_run=["English", "German"],
    )

    assert plan.write is True
    assert plan.at_risk == ()


def test_a_source_without_a_croatian_file_gets_one_written(tmp_path):
    from srtljubav.pipeline import plan_croatian_upload, write_croatian_upload_files

    topic = tmp_path / "topic"
    topic.mkdir()
    source = topic / "lecture.srt"
    write_srt(source, [("00:00:00,000", "00:00:12,000", LONG_TEXT)])
    job = TranslationJob(
        source=source,
        target_language="English",
        output_file=topic / "lecture_English.srt",
    )
    cfg = TranslationConfig(usage_log=None)

    assert plan_croatian_upload(source, cfg, languages_in_run=["English"]).write is True
    plans = write_croatian_upload_files([source], [job], cfg)

    assert [plan.write for plan in plans] == [True]
    assert (topic / "lecture_Croatian.srt").exists()


# --- a repair must clear the FULL validation stack, not just readability


class EchoRepairTranslator(FakeTranslator):
    """First pass loops on entry 3; the repair pass returns its input verbatim.

    That is exactly what gemini-srt-translator does with a skipped batch: it
    initialises its output as a COPY of the input and only checks the returned
    line count, so untranslated Croatian comes back looking like an answer.
    """

    calls = 0

    def translate(self):
        type(self).calls += 1
        self.input_text = Path(self.kwargs["input_file"]).read_text(encoding="utf-8")
        output = Path(self.kwargs["output_file"])
        if type(self).calls > 1:  # the repair pass: hand the Croatian straight back
            output.write_text(self.input_text, encoding="utf-8")
            return
        translated = [
            srt.Subtitle(
                index=sub.index,
                start=sub.start,
                end=sub.end,
                content=LOOP_TEXT if position == 3 else f"[EN] {sub.content}",
            )
            for position, sub in enumerate(srtio.parse(self.input_text), start=1)
        ]
        output.write_text(srtio.compose(translated), encoding="utf-8")


def test_a_repair_that_hands_back_its_croatian_input_fails_the_job(tmp_path, monkeypatch):
    import srtljubav.translator as translator_module

    EchoRepairTranslator.calls = 0
    monkeypatch.setattr(translator_module, "CachedGeminiSRTTranslator", EchoRepairTranslator)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    source = tmp_path / "topic" / "lecture.srt"
    source.parent.mkdir()
    write_srt(source, repair_blocks(8))
    job = TranslationJob(
        source=source,
        target_language="English",
        output_file=tmp_path / "topic" / "lecture_English.srt",
    )
    cfg = TranslationConfig(usage_log=None, clean_sections=False, yt_split=False)

    instances_before = len(FakeTranslator.instances)
    with pytest.raises(ReadabilityError) as excinfo:
        translate_job(job, cfg)

    assert "unchanged" in str(excinfo.value), "the echo must be named, not hidden"
    assert not job.output_file.exists(), "untranslated Croatian must never ship"
    attempted = FakeTranslator.instances[instances_before:]
    assert [item.kwargs["batch_size"] for item in attempted] == [300, 1, 1]


CYRILLIC_LOOP = "ми" * 800


class ScriptFlippingRepairTranslator(FakeTranslator):
    """A Cyrillic first pass whose repair answers in Latin transliteration.

    Readability alone accepts it -- the entries are short and unrepeated -- but
    over half the file is now the wrong script, which only layer 5 (expected
    script, in ``validate_translated_content``) can see.
    """

    calls = 0
    marked: tuple[int, ...] = ()

    def translate(self):
        type(self).calls += 1
        self.input_text = Path(self.kwargs["input_file"]).read_text(encoding="utf-8")
        first_pass = type(self).calls == 1
        translated = []
        for position, sub in enumerate(srtio.parse(self.input_text), start=1):
            if first_pass:
                content = (
                    CYRILLIC_LOOP
                    if position in type(self).marked
                    else f"Ово је српска реченица број {position}."
                )
            else:
                content = f"Repaired Latin sentence number {position} here."
            translated.append(
                srt.Subtitle(index=sub.index, start=sub.start, end=sub.end, content=content)
            )
        Path(self.kwargs["output_file"]).write_text(
            srtio.compose(translated), encoding="utf-8"
        )


def test_a_repair_that_breaks_the_expected_script_fails_the_job(tmp_path, monkeypatch):
    """The spliced file goes through the FULL stack again: a repair may not slip
    past the checks the first pass had to clear."""
    import srtljubav.translator as translator_module

    ScriptFlippingRepairTranslator.calls = 0
    ScriptFlippingRepairTranslator.marked = tuple(range(1, 16))
    monkeypatch.setattr(
        translator_module, "CachedGeminiSRTTranslator", ScriptFlippingRepairTranslator
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    source = tmp_path / "topic" / "lecture.srt"
    source.parent.mkdir()
    write_srt(source, repair_blocks(24))
    job = TranslationJob(
        source=source,
        target_language="Serbian (on Српски!)",
        output_file=tmp_path / "topic" / "lecture_Serbian.srt",
    )
    cfg = TranslationConfig(usage_log=None, clean_sections=False, yt_split=False)

    with pytest.raises(ReadabilityError) as excinfo:
        translate_job(job, cfg)

    assert "dominant script" in str(excinfo.value)
    assert not job.output_file.exists()


# --- the two calls a mutation test found untested: the source merge, the rewrap


BLOCK_BEFORE = (
    "Prva duga recenica koja zauzima dvije pune linije teksta na ekranu i traje "
    "dovoljno dugo, a zatim slijedi jos jedna recenica koja isto tako traje dugo "
    "i puni odsjecak. Tako je."
)
BLOCK_AFTER = (
    "Treca recenica je toliko duga da je cistac vise ne moze spojiti s prethodnim "
    "blokom, pa ostaje zaseban odsjecak koji pocinje tocno na kraju prethodnoga i "
    "traje jako dugo ovdje."
)


def test_prepare_source_text_merges_short_entries_no_silence_could_rescue(tmp_path):
    """Last step of the source preparation, on the real shape it exists for.

    "Znate." sits between two blocks too long for the cleaner to absorb it into,
    and the block after it starts on its end, so ``cleaning.repair_windows`` has
    no silence to give it. Only ``ytsplit.merge_short_entries`` can still rescue
    it (and the 0.93 s "Tako je." chunk the reflow leaves in front of it).
    """
    source = tmp_path / "lecture.srt"
    write_srt(
        source,
        [
            ("00:00:00,000", "00:00:20,000", BLOCK_BEFORE),
            ("00:00:20,000", "00:00:20,480", "Znate."),
            ("00:00:20,480", "00:00:45,000", BLOCK_AFTER),
        ],
    )

    legacy = TranslationConfig(usage_log=None, segmenter="legacy")
    prepared = srtio.parse(prepare_source_text(source, legacy))

    durations = [(entry.end - entry.start).total_seconds() for entry in prepared]
    assert min(durations) >= 1.0, f"unrescued short entries survived: {durations}"
    carrier = next(entry for entry in prepared if "Znate." in entry.content)
    assert carrier.content == "Tako je. Znate.", "the fragment must join its neighbour"
    assert (carrier.start, carrier.end) == (
        timedelta(seconds=19, milliseconds=75),
        timedelta(seconds=20, milliseconds=480),
    ), "a merge is exactly the two windows plus the gap between them"
    assert " ".join(
        " ".join(entry.content.split()) for entry in prepared
    ) == f"{BLOCK_BEFORE} Znate. {BLOCK_AFTER}", "no word may be lost or reordered"


class UnwrappedTranslator(FakeTranslator):
    """Gemini answers with ONE long line per entry: models return prose, not
    lines, so the line breaks of the source are not preserved."""

    def translate(self):
        self.input_text = Path(self.kwargs["input_file"]).read_text(encoding="utf-8")
        translated = [
            srt.Subtitle(
                index=entry.index,
                start=entry.start,
                end=entry.end,
                # same letters, same length: only the LINE BREAKS are gone
                content=" ".join(entry.content.split()).replace("a", "e").replace("i", "y"),
            )
            for entry in srtio.parse(self.input_text)
        ]
        Path(self.kwargs["output_file"]).write_text(srtio.compose(translated), encoding="utf-8")


def test_translate_job_rewraps_lines_the_translator_returned_unwrapped(
    tmp_path, monkeypatch
):
    """The output rewrap is the only thing that gives a translation its line
    widths: nothing upstream wraps the model's answer."""
    import srtljubav.translator as translator_module

    monkeypatch.setattr(translator_module, "CachedGeminiSRTTranslator", UnwrappedTranslator)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    source = tmp_path / "topic" / "lecture.srt"
    source.parent.mkdir()
    write_srt(
        source,
        [
            ("00:00:00,000", "00:00:12,000", LONG_TEXT),
            ("00:00:12,000", "00:00:24,000", LONG_TEXT),
        ],
    )
    cfg = TranslationConfig(usage_log=None, clean_sections=True, yt_split=True)
    croatian = write_croatian_upload_file(source, cfg)
    job = TranslationJob(
        source=source,
        target_language="German",
        output_file=tmp_path / "topic" / "lecture_German.srt",
    )

    translate_job(job, cfg)

    output_subs = srtio.parse(srtio.read_srt_text(job.output_file))
    unwrapped = [sub.index for sub in output_subs if "\n" not in sub.content]
    assert not unwrapped, f"entries {unwrapped} were never re-wrapped into lines"
    for sub in output_subs:
        assert max(len(line) for line in sub.content.split("\n")) <= 42
    croatian_subs = srtio.parse(srtio.read_srt_text(croatian))
    assert [(s.index, s.start, s.end) for s in output_subs] == [
        (s.index, s.start, s.end) for s in croatian_subs
    ]


# --- the misalignment guards: a repair answer that does not line up is refused


class ShrinkingRepairTranslator(FakeTranslator):
    """First pass loops on entries 3 and 5; every repair pass returns ONE entry.

    A repair that drops an entry has no positional mapping back to the file, and
    splicing it would either shift every later text or blow up mid-write.
    """

    calls = 0

    def translate(self):
        type(self).calls += 1
        self.input_text = Path(self.kwargs["input_file"]).read_text(encoding="utf-8")
        subs = srtio.parse(self.input_text)
        if type(self).calls > 1:
            subs = subs[:1]
            translated = [
                srt.Subtitle(
                    index=1, start=subs[0].start, end=subs[0].end, content="[FIX] one only"
                )
            ]
        else:
            translated = [
                srt.Subtitle(
                    index=sub.index,
                    start=sub.start,
                    end=sub.end,
                    content=LOOP_TEXT if position in (3, 5) else f"[EN] {sub.content}",
                )
                for position, sub in enumerate(subs, start=1)
            ]
        Path(self.kwargs["output_file"]).write_text(
            srtio.compose(translated), encoding="utf-8"
        )


def test_a_repair_that_returns_too_few_entries_is_never_spliced(tmp_path, monkeypatch):
    import srtljubav.translator as translator_module

    ShrinkingRepairTranslator.calls = 0
    monkeypatch.setattr(
        translator_module, "CachedGeminiSRTTranslator", ShrinkingRepairTranslator
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    source = tmp_path / "topic" / "lecture.srt"
    source.parent.mkdir()
    write_srt(source, repair_blocks(8))
    job = TranslationJob(
        source=source,
        target_language="English",
        output_file=tmp_path / "topic" / "lecture_English.srt",
    )
    cfg = TranslationConfig(usage_log=None, clean_sections=False, yt_split=False)

    with pytest.raises(ReadabilityError) as excinfo:
        translate_job(job, cfg)

    assert "returned 1 entries for 2 requested" in str(excinfo.value)
    assert not job.output_file.exists()


class PrefixingRepairTranslator(FakeTranslator):
    """A repair pass that answers every entry it is handed."""

    def translate(self):
        self.input_text = Path(self.kwargs["input_file"]).read_text(encoding="utf-8")
        translated = [
            srt.Subtitle(
                index=sub.index,
                start=sub.start,
                end=sub.end,
                content=f"[FIX] English rendering number {sub.index} here.",
            )
            for sub in srtio.parse(self.input_text)
        ]
        Path(self.kwargs["output_file"]).write_text(
            srtio.compose(translated), encoding="utf-8"
        )


def test_a_worklist_index_the_source_does_not_carry_fails_instead_of_guessing(
    tmp_path, monkeypatch
):
    """The mini-SRT maps back POSITIONALLY, so a worklist the source cannot
    fill has no safe splice: fail the file rather than shift its text."""
    import srtljubav.translator as translator_module
    from srtljubav.pipeline import repair_failed_entries
    from srtljubav.validation import EntryIssue

    monkeypatch.setattr(
        translator_module, "CachedGeminiSRTTranslator", PrefixingRepairTranslator
    )

    source = tmp_path / "lecture.srt"
    write_srt(source, repair_blocks(4))
    output = tmp_path / "lecture_English.srt"
    write_srt(
        output,
        [
            (start, end, LOOP_TEXT if index == 2 else f"[EN] English line {index}.")
            for index, (start, end, _text) in enumerate(repair_blocks(4), start=1)
        ],
    )
    report = ValidationReport(
        failures=(
            EntryIssue(2, 2, "cps", "fail", "533 chars/s"),
            EntryIssue(99, 2, "cps", "fail", "an index the source does not carry"),
        ),
        entries_checked=4,
    )

    with pytest.raises(ReadabilityError) as excinfo:
        repair_failed_entries(
            source_file=source,
            output_file=output,
            report=report,
            target_language="English",
            cfg=TranslationConfig(usage_log=None),
            base_kwargs={},
            temp_dir=tmp_path,
            usage_records=[],
        )

    assert excinfo.value.report.failed_indices == (2, 99)


# --- a translation job translates the _Croatian.srt that ships beside it


def test_job_source_text_translates_the_shipped_croatian_file(tmp_path):
    """EL6-10 on 2026-09-23: the upload plan refused to re-segment a
    ``_Croatian.srt`` that 31 translations were aligned to, but the job
    re-derived its own text, so Hindi and Telugu came out with 565 entries
    beside a 567-entry Croatian file."""
    from srtljubav.pipeline import job_source_text

    source = tmp_path / "lecture.srt"
    write_srt(source, [("00:00:00,000", "00:00:02,000", "Izvorni fragment.")])
    shipped = tmp_path / "lecture_Croatian.srt"
    write_srt(
        shipped,
        [
            ("00:00:00,000", "00:00:01,000", "Prvi."),
            ("00:00:01,000", "00:00:02,000", "Drugi."),
        ],
    )

    text = job_source_text(source, TranslationConfig(usage_log=None))

    assert [s.content for s in srtio.parse(text)] == ["Prvi.", "Drugi."]


def test_job_source_text_derives_the_croatian_when_none_ships(tmp_path):
    from srtljubav.pipeline import job_source_text

    source = tmp_path / "lecture.srt"
    write_srt(source, [("00:00:00,000", "00:00:02,000", "Izvorni fragment.")])
    cfg = TranslationConfig(usage_log=None)

    assert job_source_text(source, cfg) == prepare_source_text(source, cfg)


def test_job_source_text_honours_limit_lines_on_the_shipped_file(tmp_path):
    from srtljubav.pipeline import job_source_text

    source = tmp_path / "lecture.srt"
    write_srt(source, [("00:00:00,000", "00:00:02,000", "Izvorni fragment.")])
    write_srt(
        tmp_path / "lecture_Croatian.srt",
        [(f"00:00:0{i},000", f"00:00:0{i},900", f"Titl {i}.") for i in range(5)],
    )

    text = job_source_text(source, TranslationConfig(usage_log=None, limit_lines=2))

    assert len(srtio.parse(text)) == 2
