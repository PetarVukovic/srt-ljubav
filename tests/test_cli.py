"""CLI wiring: the reformat subcommand's rewrap-only mode."""

from pathlib import Path

from srtljubav.cli import main
from srtljubav.config import YTConfig

LONG_TEXT = (
    "Pa će tako vjerujem biti i danas, iako to nije Objava sa kaloričnim brojem, "
    "objava je nula tri devet devet, i zato ćemo je pažljivo pročitati zajedno."
)


def write_srt(path: Path, blocks: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            f"{i}\n{start} --> {end}\n{text}\n"
            for i, (start, end, text) in enumerate(blocks, 1)
        ),
        encoding="utf-8",
    )


def test_reformat_rewrap_only_dry_run_reports_without_writing(tmp_path, capsys):
    topic = tmp_path / "topic"
    croatian = [
        ("00:00:00,000", "00:00:04,000", "Prva rečenica."),
        ("00:00:04,000", "00:00:08,000", "Druga rečenica."),
    ]
    write_srt(topic / "lecture_Croatian.srt", croatian)
    write_srt(
        topic / "lecture_English.srt",
        [
            ("00:00:00,000", "00:00:04,000", LONG_TEXT),
            ("00:00:04,000", "00:00:08,000", "Second sentence."),
        ],
    )
    write_srt(topic / "lecture_German.srt", [("00:00:00,000", "00:00:08,000", "Erster Satz.")])
    english_before = (topic / "lecture_English.srt").read_text(encoding="utf-8")

    code = main(
        [
            "reformat",
            "--rewrap-only",
            "--dry-run",
            "--input-dir",
            str(tmp_path),
            "--languages",
            "English,German",
        ]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert (topic / "lecture_English.srt").read_text(encoding="utf-8") == english_before
    assert "1/2 file(s) not 1:1 with their _Croatian.srt sibling" in out
    assert "lecture_German.srt" in out
    # the over-long English entry is reported as overflow, never split
    assert f"exceed {YTConfig().rewrap_max_lines} lines" in out


def test_reformat_without_flags_rewraps_and_keeps_the_file_one_to_one(tmp_path, capsys):
    """``uv run srt-ljubav reformat`` — the command CLAUDE.md and README document
    — must never re-segment. ``discover_translated_srts`` returns translations
    only (Croatian is in neither TARGET_LANGUAGES nor SELECTABLE_LANGUAGES), so
    a re-segmenting default destroys exactly the 1:1 structure the pipeline is
    built to keep.
    """
    topic = tmp_path / "topic"
    write_srt(topic / "lecture_Croatian.srt", [("00:00:00,000", "00:00:12,000", "Duga rečenica.")])
    english = topic / "lecture_English.srt"
    write_srt(english, [("00:00:00,000", "00:00:12,000", LONG_TEXT)])

    code = main(["reformat", "--input-dir", str(tmp_path), "--languages", "English"])

    out = capsys.readouterr().out
    assert code == 0
    lines = english.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "1"
    assert lines[1] == "00:00:00,000 --> 00:00:12,000"
    assert "2" not in [line.strip() for line in lines if line.strip().isdigit()]
    assert (
        f"exceed {YTConfig().rewrap_max_lines} lines" in out
    ), "overflow is reported by default, not only with a flag"
