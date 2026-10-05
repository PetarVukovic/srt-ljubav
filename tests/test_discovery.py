from pathlib import Path

from srtljubav.discovery import build_jobs, discover_source_srts, discover_translated_srts

SAMPLE = "1\n00:00:01,000 --> 00:00:02,000\nTekst.\n"


def make_topic(root: Path) -> Path:
    topic = root / "lesson"
    topic.mkdir()
    (topic / "source.srt").write_text(SAMPLE, encoding="utf-8")
    return topic


def test_build_jobs_can_limit_languages_and_total_jobs(tmp_path):
    make_topic(tmp_path)

    jobs = build_jobs(
        tmp_path,
        overwrite=False,
        output_root=tmp_path / "out",
        target_languages=["English", "French", "Spanish"],
        max_jobs=2,
    )

    assert [job.target_language for job in jobs] == ["English", "French"]


def test_discover_excludes_translated_outputs(tmp_path):
    topic = make_topic(tmp_path)
    (topic / "source_English.srt").write_text(SAMPLE, encoding="utf-8")
    (topic / "source_Serbian.srt").write_text(SAMPLE, encoding="utf-8")

    sources = discover_source_srts(tmp_path)

    assert [p.name for p in sources] == ["source.srt"]


def test_discover_translated_srts_with_language_subset(tmp_path):
    topic = make_topic(tmp_path)
    (topic / "source_English.srt").write_text(SAMPLE, encoding="utf-8")
    (topic / "source_German.srt").write_text(SAMPLE, encoding="utf-8")

    all_translated = discover_translated_srts(tmp_path)
    english_only = discover_translated_srts(tmp_path, ["English"])

    assert [p.name for p in all_translated] == ["source_English.srt", "source_German.srt"]
    assert [p.name for p in english_only] == ["source_English.srt"]


def test_build_jobs_skips_existing_unless_overwrite(tmp_path):
    topic = make_topic(tmp_path)
    (topic / "source_English.srt").write_text(SAMPLE, encoding="utf-8")

    without = build_jobs(tmp_path, overwrite=False, target_languages=["English"])
    with_overwrite = build_jobs(tmp_path, overwrite=True, target_languages=["English"])

    assert without == []
    assert len(with_overwrite) == 1


def test_build_jobs_retries_zero_byte_existing_output(tmp_path):
    topic = make_topic(tmp_path)
    (topic / "source_English.srt").write_text("", encoding="utf-8")

    jobs = build_jobs(tmp_path, overwrite=False, target_languages=["English"])

    assert len(jobs) == 1
    assert jobs[0].output_file == topic / "source_English.srt"


def test_build_jobs_retries_unparseable_existing_output(tmp_path):
    topic = make_topic(tmp_path)
    (topic / "source_English.srt").write_text("not an srt", encoding="utf-8")

    jobs = build_jobs(tmp_path, overwrite=False, target_languages=["English"])

    assert len(jobs) == 1
    assert jobs[0].output_file == topic / "source_English.srt"
