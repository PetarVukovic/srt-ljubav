from datetime import timedelta

import pytest
import srt

from srtljubav import srtio
from srtljubav.cleaning import (
    clean_source,
    fix_timestamps,
    merge_fragments,
    repair_windows,
)
from srtljubav.config import CleaningConfig


def test_clean_source_fixes_times_and_merges_short_segments():
    content = """1
00:00:00,1 --> 00:00:00,9
Bog je ljubav

2
00:00:00,8 --> 00:00:01,2
i ljubav ostaje.

3
00:00:03 --> 00:00:04


"""

    cleaned_srt, report = clean_source(content, CleaningConfig())

    assert "00:00:00,100 --> 00:00:01,200" in cleaned_srt
    assert "Bog je ljubav i ljubav ostaje." in cleaned_srt
    assert report.deleted[0]["Error"] == "Empty text"


def test_fix_timestamps_clamps_overlaps():
    subs = srtio.parse(
        "1\n00:00:00,100 --> 00:00:00,900\nA\n\n2\n00:00:00,800 --> 00:00:01,200\nB\n\n"
    )

    fixed = fix_timestamps(subs)

    assert fixed[0].end == timedelta(milliseconds=800)
    assert fixed[1].start == timedelta(milliseconds=800)
    assert fixed[0].end <= fixed[1].start


def test_merge_fragments_respects_max_len():
    subs = [
        srt.Subtitle(index=1, start=timedelta(0), end=timedelta(seconds=2), content="x" * 100),
        srt.Subtitle(
            index=2, start=timedelta(seconds=2), end=timedelta(seconds=4), content="y" * 100
        ),
    ]

    merged = merge_fragments(
        subs, max_len=120, min_duration=2.0, max_dist_forward=0.25, max_dist_backward=0.05
    )

    assert len(merged) == 2  # combined 201 chars > 120, no merge


def test_merge_fragments_respects_max_duration():
    subs = [
        srt.Subtitle(index=1, start=timedelta(0), end=timedelta(seconds=10), content="Prvi."),
        srt.Subtitle(
            index=2, start=timedelta(seconds=10), end=timedelta(seconds=20), content="Drugi."
        ),
    ]

    merged = merge_fragments(
        subs,
        max_len=240,
        min_duration=2.0,
        max_dist_forward=0.25,
        max_dist_backward=0.05,
        max_duration=18.0,
    )

    assert len(merged) == 2  # combined 20 s > 18 s cap


def sub_at(start_s: float, end_s: float, content: str, index: int = 1) -> srt.Subtitle:
    return srt.Subtitle(
        index=index,
        start=timedelta(seconds=start_s),
        end=timedelta(seconds=end_s),
        content=content,
    )


def test_repair_windows_restores_an_impossible_window_from_the_following_silence():
    """A transcript that packs 420 characters into 1.94 s mis-assigned the end:
    the speech really continued into what it recorded as silence. Restore the
    window at the speaker's own median rate."""
    text = "x" * 420
    subs = [sub_at(0.0, 1.94, text, 1), sub_at(29.3, 32.0, "Sljedeci.", 2)]

    repaired = repair_windows(subs, CleaningConfig())

    duration = (repaired[0].end - repaired[0].start).total_seconds()
    assert duration == pytest.approx(420 / 17.0, abs=0.01)
    assert len(text) / duration <= 40.0


def test_repair_windows_never_crosses_the_next_entrys_start():
    subs = [sub_at(0.0, 1.0, "y" * 300, 1), sub_at(4.0, 6.0, "Sljedeci.", 2)]

    repaired = repair_windows(subs, CleaningConfig())

    assert repaired[0].end <= repaired[1].start
    assert repaired[0].end == timedelta(seconds=4)


def test_repair_windows_only_moves_ends_never_starts_and_never_shortens():
    subs = [sub_at(0.0, 0.2, "Kratko.", 1), sub_at(9.0, 12.0, "Dugo dugo dugo.", 2)]

    repaired = repair_windows(subs, CleaningConfig())

    assert [s.start for s in repaired] == [s.start for s in subs]
    assert [s.content for s in repaired] == [s.content for s in subs]
    assert all(new.end >= old.end for new, old in zip(repaired, subs, strict=True))


def test_repair_windows_extends_a_correctly_timed_short_entry_only_to_the_floor():
    """A brief utterance is timed correctly, so take the minimum the floor needs
    instead of reclaiming silence the speaker never used."""
    subs = [sub_at(0.0, 0.4, "Sto to znaci?", 1), sub_at(30.0, 32.0, "Dalje.", 2)]

    repaired = repair_windows(subs, CleaningConfig())

    assert (repaired[0].end - repaired[0].start).total_seconds() == pytest.approx(1.0)


def test_repair_windows_leaves_an_entry_alone_when_there_is_no_following_gap():
    subs = [sub_at(0.0, 0.3, "Kratko.", 1), sub_at(0.3, 4.0, "Odmah.", 2)]

    repaired = repair_windows(subs, CleaningConfig())

    assert repaired[0].end == timedelta(seconds=0.3)


def test_repair_windows_is_idempotent():
    subs = [sub_at(0.0, 1.94, "x" * 420, 1), sub_at(40.0, 42.0, "Dalje.", 2)]

    once = repair_windows(subs, CleaningConfig())
    twice = repair_windows(once, CleaningConfig())

    assert [(s.index, s.start, s.end, s.content) for s in once] == [
        (s.index, s.start, s.end, s.content) for s in twice
    ]
