from datetime import timedelta

import srt
from scripts import repair_archive
from scripts.repair_archive import (
    FileStats,
    assign_to_windows,
    balance_bidi,
    croatian_identical_indices,
    duplicate_topic_findings,
    leftovers_by_language,
    measure,
    redistribute,
    regroup_to_croatian,
    repair_topic,
    repair_translation,
)

from srtljubav import srtio
from srtljubav.config import TranslationConfig, YTConfig
from srtljubav.validation import bidi_balance

CFG = YTConfig()

RLE = "‫"
PDF = "‬"
RLI = "⁧"
PDI = "⁩"


def sub(index: int, start_s: float, end_s: float, content: str) -> srt.Subtitle:
    return srt.Subtitle(
        index=index,
        start=timedelta(seconds=start_s),
        end=timedelta(seconds=end_s),
        content=content,
    )


def windows(spans: list[tuple[float, float]], texts: list[str] | None = None):
    texts = texts or ["hrvatski" for _ in spans]
    return [
        sub(i, start, end, text)
        for i, ((start, end), text) in enumerate(zip(spans, texts, strict=True), start=1)
    ]


# --- measure ---------------------------------------------------------------


def test_measure_reports_short_entries_and_the_worst_reading_rate():
    croatian = windows([(0.0, 4.0), (4.0, 4.5)])
    subs = [
        sub(1, 0.0, 4.0, "a" * 40),  # 10 cps
        sub(2, 4.0, 4.5, "b" * 40),  # 80 cps, 0.5 s
    ]

    stats = measure(subs, croatian)

    assert stats == FileStats(
        entries=2,
        under_min_duration=1,
        above_ceiling=1,
        worst_cps=80.0,
        one_to_one=True,
    )


def test_measure_flags_a_file_that_is_not_one_to_one_with_croatian():
    croatian = windows([(0.0, 4.0), (4.0, 8.0)])
    subs = [sub(1, 0.0, 4.0, "prijevod")]

    assert measure(subs, croatian).one_to_one is False
    assert measure(windows([(0.0, 4.0), (4.0, 7.0)]), croatian).one_to_one is False


# --- assignment / redistribution / regrouping ------------------------------


def test_each_translated_entry_lands_in_the_window_it_overlaps_most():
    croatian = windows([(0.0, 4.0), (4.0, 8.0)])
    translated = [
        sub(1, 0.0, 2.0, "one"),
        sub(2, 2.0, 4.0, "two"),
        sub(3, 4.0, 8.0, "three"),
    ]

    groups = assign_to_windows(translated, croatian)

    assert [[e.content for e in group] for group in groups] == [["one", "two"], ["three"]]


def test_assignment_never_goes_backwards_so_the_text_order_is_kept():
    # The second chunk overlaps window 1 more, but window 2 is already open.
    croatian = windows([(0.0, 4.0), (4.0, 8.0)])
    translated = [
        sub(1, 3.5, 8.0, "first"),  # best window: 2
        sub(2, 0.0, 3.9, "second"),  # best window: 1, but that is behind
    ]

    groups = assign_to_windows(translated, croatian)

    assert [[e.content for e in group] for group in groups] == [[], ["first", "second"]]


def test_empty_windows_are_filled_from_a_crowded_neighbour():
    # The reviewer's case: the old pipeline crammed six chunks into a 2 s
    # window, the fixed one spreads the same text over six windows.
    groups = [[sub(i, 0.0, 0.3, f"c{i}") for i in range(1, 7)], [], [], [], [], []]

    filled = redistribute(groups)

    assert [[e.content for e in g] for g in filled] == [
        ["c1"],
        ["c2"],
        ["c3"],
        ["c4"],
        ["c5"],
        ["c6"],
    ]


def test_redistribution_can_also_pull_from_the_right():
    groups = [[], [sub(1, 0.0, 1.0, "a"), sub(2, 1.0, 2.0, "b")]]

    assert [[e.content for e in g] for g in redistribute(groups)] == [["a"], ["b"]]


def test_redistribution_leaves_a_window_empty_when_there_is_no_text_to_spare():
    groups = [[sub(1, 0.0, 1.0, "only")], [], []]

    filled = redistribute(groups)

    assert [len(g) for g in filled] == [1, 0, 0]


def test_regroup_returns_the_croatian_structure_with_the_translated_text():
    croatian = windows([(0.0, 4.0), (4.0, 9.0)])
    translated = [
        sub(1, 0.0, 2.0, "Hello there"),
        sub(2, 2.0, 4.0, "friend"),
        sub(3, 4.0, 9.0, "and\ngoodbye"),
    ]

    regrouped = regroup_to_croatian(translated, croatian)

    assert [(e.index, e.start, e.end) for e in regrouped] == [
        (e.index, e.start, e.end) for e in croatian
    ]
    assert [e.content for e in regrouped] == ["Hello there friend", "and goodbye"]


def test_regroup_joins_no_space_scripts_without_inserting_spaces():
    croatian = windows([(0.0, 6.0)])
    translated = [sub(1, 0.0, 3.0, "我們必須"), sub(2, 3.0, 6.0, "彼此相愛")]

    assert regroup_to_croatian(translated, croatian)[0].content == "我們必須彼此相愛"


# --- bidi ------------------------------------------------------------------


def test_balanced_bidi_text_is_left_alone():
    text = f"{RLE}سلام{PDF}"

    assert balance_bidi(text) == text


def test_an_unclosed_embedding_gets_its_pop_appended():
    repaired = balance_bidi(f"{RLE}سلام")

    assert repaired == f"{RLE}سلام{PDF}"
    assert bidi_balance(repaired) == 0


def test_an_unclosed_isolate_is_closed_with_the_isolate_terminator():
    repaired = balance_bidi(f"{RLI}سلام")

    assert repaired == f"{RLI}سلام{PDI}"
    assert bidi_balance(repaired) == 0


def test_a_stray_closer_is_dropped_rather_than_matched_by_an_invented_opener():
    repaired = balance_bidi(f"سلام{PDF}")

    assert repaired == "سلام"
    assert bidi_balance(repaired) == 0


def test_balancing_bidi_is_idempotent():
    once = balance_bidi(f"{RLE}a{PDI}b{RLI}c")

    assert balance_bidi(once) == once
    assert bidi_balance(once) == 0


# --- Croatian leftovers ----------------------------------------------------


def test_only_nontrivial_entries_count_as_identical_to_croatian():
    croatian = windows(
        [(0.0, 4.0), (4.0, 8.0), (8.0, 12.0)],
        ["Ivan 3,16", "Ovo je posve obicna hrvatska recenica.", "Druga hrvatska recenica ovdje."],
    )
    translated = [
        sub(1, 0.0, 4.0, "Ivan 3,16"),  # a verse reference, never leakage
        sub(2, 4.0, 8.0, "Ovo je posve obicna hrvatska recenica."),
        sub(3, 8.0, 12.0, "This is another Croatian sentence."),
    ]

    assert croatian_identical_indices(croatian, translated) == (2,)


def test_an_entry_identical_in_every_language_is_a_name_not_a_leftover():
    identical = {
        "Slovenian": (5, 9),
        "German": (9,),
        "English": (9,),
        "Spanish": (9,),
    }

    assert leftovers_by_language(identical) == {"Slovenian": (5,)}


# --- the whole per-file repair ---------------------------------------------


def test_repair_makes_a_translation_one_to_one_and_rewraps_it():
    croatian = windows([(0.0, 4.0), (4.0, 8.0)])
    translated = [
        sub(1, 0.0, 1.0, "This is a fairly long English sentence that needs to be re-wrapped"),
        sub(2, 1.0, 4.0, f"and it carries {RLE}an unclosed embedding"),
        sub(3, 4.0, 8.0, "second window"),
    ]

    result = repair_translation(translated, croatian, CFG)

    repaired = result.repaired
    # The intermediate the caller needs anyway is handed out, not recomputed.
    assert [e.index for e in result.regrouped] == [e.index for e in croatian]
    assert [(e.index, e.start, e.end) for e in repaired] == [
        (e.index, e.start, e.end) for e in croatian
    ]
    assert all(bidi_balance(e.content) == 0 for e in repaired)
    # a translation is never split, so an entry that outgrows two lines takes a
    # third rather than losing its 1:1 window
    assert all(len(e.content.split("\n")) <= CFG.rewrap_max_lines for e in repaired)
    assert len(repaired[0].content.split("\n")) == 3
    assert repaired[1].content == "second window"


def test_repair_is_idempotent():
    croatian = windows([(0.0, 4.0), (4.0, 8.0)])
    translated = [
        sub(1, 0.0, 2.0, "Some English words here"),
        sub(2, 2.0, 4.0, "and some more of them"),
        sub(3, 4.0, 8.0, "second window text"),
    ]

    once = repair_translation(translated, croatian, CFG).repaired

    assert repair_translation(once, croatian, CFG).repaired == once


# --- duplicate topic folders ----------------------------------------------


def test_a_folder_holding_another_topics_source_srt_is_a_duplicate(tmp_path):
    # The archive's real signature: the folder is named "15. <title>" but the
    # SRT inside it is named after the "VRIJEME JE KRAJA! 15. <title>" folder.
    duplicate = tmp_path / "15. Predskazanje"
    canonical = tmp_path / "VRIJEME JE KRAJA! 15. Predskazanje"
    duplicate.mkdir()
    canonical.mkdir()
    (duplicate / "VRIJEME JE KRAJA! 15. Predskazanje.srt").write_bytes(b"1\n")
    (canonical / "VRIJEME JE KRAJA! 15. Predskazanje.srt").write_bytes(b"1\n")
    (duplicate / "download.zip").write_bytes(b"zip")

    findings = duplicate_topic_findings(tmp_path)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.duplicate == duplicate
    assert finding.canonical == canonical
    assert finding.identical == ("VRIJEME JE KRAJA! 15. Predskazanje.srt",)
    assert finding.only_in_duplicate == ("download.zip",)
    assert finding.safe_to_delete is False  # the zip exists nowhere else
    assert finding.rm_command == f'rm -rf "{duplicate}"'


def test_a_duplicate_whose_srt_differs_is_never_reported_as_safe(tmp_path):
    duplicate = tmp_path / "16. Kraj"
    canonical = tmp_path / "VRIJEME JE KRAJA! 16. Kraj"
    duplicate.mkdir()
    canonical.mkdir()
    (duplicate / "VRIJEME JE KRAJA! 16. Kraj.srt").write_bytes(b"1\n")
    (canonical / "VRIJEME JE KRAJA! 16. Kraj.srt").write_bytes(b"2\n")

    finding = duplicate_topic_findings(tmp_path)[0]

    assert finding.identical == ()
    assert finding.differing == ("VRIJEME JE KRAJA! 16. Kraj.srt",)
    assert finding.safe_to_delete is False


def test_a_topic_folder_that_owns_its_source_srt_is_not_a_duplicate(tmp_path):
    topic = tmp_path / "VRIJEME JE KRAJA! 18. Najava"
    topic.mkdir()
    (topic / "VRIJEME JE KRAJA! 18. Najava.srt").write_bytes(b"1\n")

    assert duplicate_topic_findings(tmp_path) == []


# --- the topic sweep -------------------------------------------------------

SOURCE_SRT = """1
00:00:00,000 --> 00:00:04,000
Ovo je prva recenica ovoga predavanja.

2
00:00:04,000 --> 00:00:09,000
A ovo je druga recenica koja govori o ljubavi prema bliznjemu.

3
00:00:09,000 --> 00:00:14,000
I na kraju dolazi treca recenica ovoga kratkog odlomka.
"""

GERMAN_SRT = """1
00:00:00,000 --> 00:00:03,000
Dies ist der erste Satz dieses Vortrags.

2
00:00:03,000 --> 00:00:09,000
Und dies ist der zweite Satz, der von der Liebe zum Naechsten spricht.

3
00:00:09,000 --> 00:00:14,000
Und am Ende kommt der dritte Satz dieses kurzen Abschnitts.
"""

OTHER_SOURCE_SRT = """1
00:00:00,000 --> 00:00:05,000
Drugo predavanje pocinje ovom recenicom o istini.

2
00:00:05,000 --> 00:00:11,000
I nastavlja se drugom recenicom koja govori o vjeri i nadi.
"""

OTHER_GERMAN_SRT = """1
00:00:00,000 --> 00:00:05,000
Der zweite Vortrag beginnt mit diesem Satz ueber die Wahrheit.

2
00:00:05,000 --> 00:00:11,000
Und setzt sich mit einem zweiten Satz ueber Glaube und Hoffnung fort.
"""


def build_topic(root, *, with_croatian: bool = False, second_source: bool = False):
    """One topic folder: source A (+ German), optionally a second source B."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "Lecture A.srt").write_text(SOURCE_SRT, encoding="utf-8")
    (root / "Lecture A_German.srt").write_text(GERMAN_SRT, encoding="utf-8")
    if with_croatian:
        (root / "Lecture A_Croatian.srt").write_text(SOURCE_SRT, encoding="utf-8")
    if second_source:
        (root / "Lecture B.srt").write_text(OTHER_SOURCE_SRT, encoding="utf-8")
        (root / "Lecture B_German.srt").write_text(OTHER_GERMAN_SRT, encoding="utf-8")
        if with_croatian:
            (root / "Lecture B_Croatian.srt").write_text(OTHER_SOURCE_SRT, encoding="utf-8")
    return root / "Lecture A.srt"


def snapshot(root):
    return {path.name: path.read_bytes() for path in sorted(root.iterdir()) if path.is_file()}


def test_repair_topic_never_touches_another_sources_translations(tmp_path):
    source = build_topic(tmp_path / "topic", with_croatian=True, second_source=True)
    before = snapshot(source.parent)

    reports, _leftovers = repair_topic(source, TranslationConfig(), apply=True)

    after = snapshot(source.parent)
    assert [report.path.name for report in reports] == [
        "Lecture A_Croatian.srt",
        "Lecture A_German.srt",
    ]
    for name in ("Lecture B.srt", "Lecture B_German.srt", "Lecture B_Croatian.srt"):
        assert after[name] == before[name], f"{name} was rewritten by another lecture's repair"


def test_the_dry_run_plans_exactly_the_work_the_apply_run_performs(tmp_path):
    # The state a `translate --limit-lines` run leaves: no _Croatian.srt yet.
    dry_source = build_topic(tmp_path / "dry")
    apply_source = build_topic(tmp_path / "apply")

    dry_reports, _ = repair_topic(dry_source, TranslationConfig(), apply=False)
    apply_reports, _ = repair_topic(apply_source, TranslationConfig(), apply=True)

    def plan(reports):
        return [(r.label, r.changed, r.before, r.after) for r in reports]

    assert plan(dry_reports) == plan(apply_reports)
    assert [r.label for r in dry_reports] == ["Croatian", "German"]


def test_the_dry_run_writes_nothing_to_disk(tmp_path):
    source = build_topic(tmp_path / "topic", second_source=True)
    before = snapshot(source.parent)

    repair_topic(source, TranslationConfig(), apply=False)

    assert snapshot(source.parent) == before


def test_apply_writes_the_repaired_files_the_report_describes(tmp_path):
    source = build_topic(tmp_path / "topic")

    reports, _ = repair_topic(source, TranslationConfig(), apply=True)

    croatian_path = source.parent / "Lecture A_Croatian.srt"
    german_path = source.parent / "Lecture A_German.srt"
    assert snapshot(source.parent).keys() == {
        "Lecture A.srt",
        "Lecture A_Croatian.srt",
        "Lecture A_German.srt",
    }
    croatian_subs = srtio.parse(srtio.read_srt_text(croatian_path))
    german_subs = srtio.parse(srtio.read_srt_text(german_path))
    assert measure(croatian_subs) == reports[0].after
    assert measure(german_subs, croatian_subs) == reports[1].after
    assert reports[1].after.one_to_one is True


def test_a_translation_is_regrouped_once_per_file_not_twice(tmp_path, monkeypatch):
    source = build_topic(tmp_path / "topic", with_croatian=True)
    calls = []
    original = repair_archive.regroup_to_croatian

    def counting(translated, croatian):
        calls.append(len(translated))
        return original(translated, croatian)

    monkeypatch.setattr(repair_archive, "regroup_to_croatian", counting)

    repair_topic(source, TranslationConfig(), apply=False)

    assert len(calls) == 1
