import dataclasses
from datetime import timedelta

import srt

from srtljubav import ytsplit
from srtljubav.config import YTConfig
from srtljubav.ytsplit import (
    REFUSAL_GAP,
    REFUSAL_LINES,
    REFUSAL_WINDOW,
    entry_merge_decision,
    merge_short_entries,
    reflow,
    reflow_text,
    reflow_with_anchors,
    rewrap,
    rewrap_overflow,
    rewrap_text,
    split_entry,
    wrap_lines,
)

CFG = YTConfig()


def sub(start_s: float, end_s: float, content: str, index: int = 1) -> srt.Subtitle:
    return srt.Subtitle(
        index=index,
        start=timedelta(seconds=start_s),
        end=timedelta(seconds=end_s),
        content=content,
    )


LONG_LATIN = (
    "Pa će tako vjerujem biti i danas, iako to nije Objava sa kaloričnim brojem, "
    "objava je nula tri devet devet, i zato ćemo je pažljivo pročitati zajedno."
)  # ~150 chars > 84 capacity


def test_long_entry_splits_to_two_lines_max_42_chars():
    children = split_entry(sub(0.0, 12.0, LONG_LATIN), CFG)

    assert len(children) >= 2
    for child in children:
        lines = child.content.split("\n")
        assert len(lines) <= 2
        assert all(len(line) <= 42 for line in lines)


def test_split_children_are_contiguous_within_parent():
    parent = sub(3.0, 15.0, LONG_LATIN)

    children = split_entry(parent, CFG)

    assert children[0].start == parent.start
    assert children[-1].end == parent.end
    for left, right in zip(children, children[1:], strict=False):
        assert left.end == right.start  # contiguous, no overlap


def test_time_allocated_proportionally_to_chars():
    parent = sub(0.0, 10.0, LONG_LATIN)

    children = split_entry(parent, CFG)

    total_chars = sum(len(c.content.replace("\n", " ")) for c in children)
    for child in children:
        frac = len(child.content.replace("\n", " ")) / total_chars
        duration = (child.end - child.start).total_seconds()
        assert abs(duration - 10.0 * frac) < 1.0  # within 1s of proportional


def test_short_entry_passes_through_unchanged():
    original = sub(1.0, 4.0, "Bog je ljubav\ni ljubav ostaje.")

    children = split_entry(original, CFG)

    assert children == [original]


def test_cjk_uses_narrower_limit():
    text = "彼此尊重彼此相愛因為通過愛我們才能認識上帝的真理和祂的旨意" * 2  # 56 chars
    children = split_entry(sub(0.0, 10.0, text), CFG)

    assert len(children) >= 2
    for child in children:
        for line in child.content.split("\n"):
            assert len(line) <= 18


def test_sentence_boundary_preferred():
    text = (
        "Prva rečenica koja ima svojih pedeset znakova ovdje. "
        "Druga rečenica je isto tako duga baš oko pedeset."
    )
    children = split_entry(sub(0.0, 8.0, text), CFG)

    assert children[0].content.replace("\n", " ").endswith("ovdje.")
    assert children[1].content.replace("\n", " ").startswith("Druga")


def test_no_subminimum_duration_slivers():
    # 90 chars: just over capacity -> naive split could give a tiny tail chunk
    text = ("Riječ " * 14).strip() + " kraj."  # 14*6 + 6 = 90 chars
    children = split_entry(sub(0.0, 3.0, text), CFG)

    for child in children:
        assert (child.end - child.start).total_seconds() >= CFG.min_duration - 0.01


def test_long_duration_splits_at_sentence_boundary_only():
    # two sentences, 12 s -> split into two ~6 s entries
    text = "Prva kratka rečenica ovdje stoji. Druga kratka rečenica isto."
    children = split_entry(sub(0.0, 12.0, text), CFG)

    assert len(children) == 2
    for child in children:
        assert (child.end - child.start).total_seconds() <= CFG.max_duration + 0.01

    # single sentence, no clean split point -> stays one entry even if long
    single = split_entry(sub(0.0, 12.0, "Jedna jedina rečenica bez kraja"), CFG)
    assert len(single) == 1


def test_reflow_is_idempotent():
    subs = [
        sub(0.0, 12.0, LONG_LATIN, index=1),
        sub(12.0, 14.0, "Kratko.", index=2),
        sub(14.0, 24.0, "彼此尊重彼此相愛因為通過愛我們才能認識上帝的真理" * 2, index=3),
    ]

    once = reflow(subs, CFG)
    twice = reflow(once, CFG)

    assert [(s.start, s.end, s.content) for s in once] == [
        (s.start, s.end, s.content) for s in twice
    ]


def test_reflow_idempotent_when_share_drifts_past_max_duration():
    # Regression: whitespace normalization shifts proportional shares between
    # passes, so a pass-1 child can land at ~7.05 s; pass 2 must NOT re-split it.
    text = (
        "poštujte i ljubite jedan drugoga.  Pazite, to su Rimljanima 12:10. "
        "Rimljanima 12:10."
    )
    subs = [sub(574.45, 583.35, text)]  # 8.9 s window

    once = reflow(subs, CFG)
    twice = reflow(once, CFG)

    assert [(s.start, s.end, s.content) for s in once] == [
        (s.start, s.end, s.content) for s in twice
    ]


def test_unwrappable_chunk_splits_into_more_entries():
    # 78 chars fits 84-char capacity, but no word boundary allows a legal
    # 2x42 wrap ("trouble—that's" sits across the midpoint) -> must split
    # into more entries instead of overflowing a line.
    text = "looking at some of you who are in trouble—that's the second part of the title,"
    children = split_entry(sub(0.0, 7.9, text), CFG)

    assert len(children) >= 2
    for child in children:
        for line in child.content.split("\n"):
            assert len(line) <= 42


def test_reflow_renumbers_sequentially():
    subs = [sub(0.0, 12.0, LONG_LATIN, index=1), sub(12.0, 14.0, "Kratko.", index=2)]

    result = reflow(subs, CFG)

    assert [s.index for s in result] == list(range(1, len(result) + 1))


def test_reflow_text_round_trip():
    content = srt.compose([sub(0.0, 12.0, LONG_LATIN)])

    once = reflow_text(content, CFG)
    twice = reflow_text(once, CFG)

    assert once == twice
    assert once.count(" --> ") >= 2


def test_wrap_lines_japanese_never_breaks_digit_runs_or_small_kana():
    # "第100号" must not split inside "100"; small kana/ー must not start a line
    text = "したがって小冊子第100号愛の福音6はガレージとなります"
    wrapped = wrap_lines(text, max_chars=18, char_mode=True)

    lines = wrapped.split("\n")
    assert len(lines) == 2
    assert all(len(line) <= 18 for line in lines)
    assert not (lines[0][-1].isdigit() and lines[1][0].isdigit())
    assert lines[1][0] not in "ーぁぃぅぇぉゃゅょっァィゥェォャュョッ"


def test_wrap_lines_thai_never_breaks_before_combining_marks():
    import unicodedata

    text = "และในหนังสือเล่มเล็กเล่มนี้เปรียบเสมือนคลังเก็บของ"  # 50 chars
    wrapped = wrap_lines(text, max_chars=28, char_mode=True)

    lines = wrapped.split("\n")
    assert len(lines) == 2
    # a line must never START with a combining mark (broken grapheme cluster)
    assert unicodedata.category(lines[1][0]) not in ("Mn", "Mc")
    # and must never break right after a leading vowel (เแโใไ)
    assert lines[0][-1] not in "เแโใไ"


def test_reflow_fixes_existing_illegal_cjk_break():
    # legacy entry: size-conformant but the break splits the digit run "100"
    legacy = sub(0.0, 5.0, "したがって小冊子第10\n0号愛の福音6はガレージ")

    children = reflow([legacy], CFG)

    for child in children:
        lines = child.content.split("\n")
        for left, right in zip(lines, lines[1:], strict=False):
            assert not (left[-1].isdigit() and right[0].isdigit())
    # still idempotent
    assert reflow(children, CFG) == children


def test_reflow_heals_entries_split_mid_grapheme():
    import unicodedata

    # legacy damage: contiguous Thai entries cut mid-grapheme ("...คล" | "ัง...")
    a = sub(0.0, 3.0, "และในหนังสือเล่มเล็ก\nเล่มนี้ เปรียบเสมือนคล", index=1)
    b = sub(3.0, 6.0, "ังเก็บของจากสวรรค์ประ\nเภทหนึ่ง", index=2)

    healed = reflow([a, b], CFG)

    for child in healed:
        for line in child.content.split("\n"):
            assert unicodedata.category(line[0]) not in ("Mn", "Mc")
            assert line[-1] not in "เแโใไ"
    # boundaries between healed entries are legal too
    for left, right in zip(healed, healed[1:], strict=False):
        last, first = left.content.replace("\n", "")[-1], right.content.replace("\n", "")[0]
        assert unicodedata.category(first) not in ("Mn", "Mc")
        assert last not in "เแโใไ"
    assert reflow(healed, CFG) == healed  # still idempotent


def test_join_heals_space_injected_into_digit_run():
    # legacy damage: a space inside a number ("第10 0号")
    damaged = sub(0.0, 5.0, "したがって小冊子第10 \n0号愛の福音6はガレージとなります")

    children = reflow([damaged], CFG)

    joined = "".join(c.content.replace("\n", "") for c in children)
    assert "第100号" in joined


def test_wrap_lines_balances_two_lines():
    text = "Poštujte i ljubite jedan drugoga jer kroz Ljubav dolazi istina"  # 62 chars

    wrapped = wrap_lines(text, max_chars=42, char_mode=False)

    lines = wrapped.split("\n")
    assert len(lines) == 2
    assert all(len(line) <= 42 for line in lines)
    assert " ".join(lines) == text


def test_merge_short_entries_merges_a_sub_half_second_entry_with_its_neighbour():
    """YouTube rejects entries under 0.5 s, and anything under min_duration reads
    as a flash. Merging is lossless -- same text, same combined window -- so it
    beats stretching a correctly timed entry."""
    subs = [
        sub(0.0, 0.3, "Sto to znaci?", 1),
        sub(0.3, 4.0, "To znaci da je ljubav sve.", 2),
    ]

    merged = merge_short_entries(subs, CFG)

    assert len(merged) == 1
    assert "Sto to znaci?" in merged[0].content
    assert "To znaci da je ljubav sve." in merged[0].content
    assert merged[0].start == subs[0].start
    assert merged[0].end == subs[1].end


def test_merge_short_entries_leaves_an_entry_alone_when_no_merge_fits_two_lines():
    subs = [
        sub(0.0, 0.3, "Kratko.", 1),
        sub(0.3, 6.0, "z" * 84, 2),
    ]

    decision = entry_merge_decision(subs, 0, CFG)

    assert [candidate.refusals for candidate in decision.candidates] == [(REFUSAL_LINES,)]
    assert decision.accepted is None
    assert merge_short_entries(subs, CFG) == list(subs)


def test_merge_short_entries_merges_the_same_fixture_once_the_text_fits_two_lines():
    """Companion to the refusal above: only the neighbour's length blocks it."""
    subs = [
        sub(0.0, 0.3, "Kratko.", 1),
        sub(0.3, 6.0, "Ovo je susjed koji lijepo stane u dva reda.", 2),
    ]

    decision = entry_merge_decision(subs, 0, CFG)

    assert decision.accepted is not None
    assert len(merge_short_entries(subs, CFG)) == 1


def test_merge_short_entries_respects_the_max_duration_cap():
    subs = [
        sub(0.0, 0.3, "Kratko.", 1),
        sub(0.3, 20.0, "Dugacki susjed.", 2),
    ]

    decision = entry_merge_decision(subs, 0, CFG)

    assert [candidate.refusals for candidate in decision.candidates] == [(REFUSAL_WINDOW,)]
    assert decision.candidates[0].window_secs == 20.0
    assert merge_short_entries(subs, CFG) == list(subs)


def test_merge_short_entries_merges_the_same_fixture_once_the_window_fits():
    """Companion to the refusal above: only the joined window blocks it."""
    subs = [
        sub(0.0, 0.3, "Kratko.", 1),
        sub(0.3, 5.0, "Dugacki susjed.", 2),
    ]

    decision = entry_merge_decision(subs, 0, CFG)

    assert decision.accepted is not None
    assert len(merge_short_entries(subs, CFG)) == 1


def test_merge_short_entries_is_idempotent():
    subs = [
        sub(0.0, 0.3, "Sto to znaci?", 1),
        sub(0.3, 4.0, "To znaci da je ljubav sve.", 2),
        sub(4.0, 8.0, "I onda jos nesto.", 3),
    ]

    once = merge_short_entries(subs, CFG)
    twice = merge_short_entries(once, CFG)

    assert len(once) < len(subs)  # the first pass really merged something
    assert [(s.start, s.end, s.content) for s in once] == [
        (s.start, s.end, s.content) for s in twice
    ]


def test_merge_short_entries_reindexes_contiguously():
    subs = [
        sub(0.0, 0.3, "Sto to znaci?", 1),
        sub(0.3, 4.0, "To znaci da je ljubav sve.", 2),
        sub(4.0, 8.0, "I onda jos nesto.", 3),
    ]

    merged = merge_short_entries(subs, CFG)

    assert len(merged) < len(subs)  # indices only need renumbering after a merge
    assert [s.index for s in merged] == list(range(1, len(merged) + 1))


# --- Short-entry rescue on the anchored path -------------------------------
# Anchors snap a chunk's start onto the original fragment start, so a chunk's
# real window can be far shorter than the proportional share the pre-split
# merge pass guarded against. These fixtures keep every proportional share
# above min_duration and let the anchor alone create the too-short chunk.

ANCHOR_S0 = "Ovo je prva recenica."
ANCHOR_S1 = "A ovo je druga recenica malo duza."
ANCHOR_S1_WIDE = "A ovo je druga recenica koja je znatno duza od prve i ne stane skupa."
ANCHOR_S2 = "I na kraju treca recenica ovdje."
ANCHOR_AT_S1 = [(22, timedelta(seconds=0.5))]  # offset of the second sentence


def _durations(children: list[srt.Subtitle]) -> list[float]:
    return [(child.end - child.start).total_seconds() for child in children]


def test_anchored_chunk_below_min_duration_merges_into_its_neighbour():
    parent = sub(0.0, 7.0, f"{ANCHOR_S0} {ANCHOR_S1} {ANCHOR_S2}")

    children = split_entry(parent, CFG, ANCHOR_AT_S1)

    assert len(children) == 2
    assert min(_durations(children)) >= CFG.min_duration
    assert ANCHOR_S0 in children[0].content.replace("\n", " ")
    assert ANCHOR_S1 in children[0].content.replace("\n", " ")
    assert children[0].start == parent.start
    assert children[-1].end == parent.end


def test_anchored_merge_is_refused_when_the_joined_text_needs_three_lines():
    """Same fixture as the merging test above with only the middle sentence
    widened, so the merge pass is exercised and refused, not skipped."""
    parent = sub(0.0, 7.0, f"{ANCHOR_S0} {ANCHOR_S1_WIDE} {ANCHOR_S2}")

    children = split_entry(parent, CFG, ANCHOR_AT_S1)

    assert len(children) == 3
    assert _durations(children)[0] < CFG.min_duration  # left short rather than mangled
    # the refusal reason: the two chunks that would have merged do not fit two
    # lines of 42 -- wrap_lines can only overflow them
    joined = " ".join(child.content.replace("\n", " ") for child in children[:2])
    wrapped = wrap_lines(joined, 42, char_mode=False).split("\n")
    assert max(len(line) for line in wrapped) > 42
    for child in children:
        assert len(child.content.split("\n")) <= CFG.max_lines


def test_anchored_merge_is_refused_when_the_joined_window_exceeds_max_duration():
    parent = sub(0.0, 12.0, f"{ANCHOR_S0} {ANCHOR_S1} {ANCHOR_S2}")

    children = split_entry(parent, CFG, ANCHOR_AT_S1)

    assert len(children) == 3
    assert _durations(children)[0] < CFG.min_duration
    assert max(_durations(children)) <= CFG.max_duration * 1.1
    # the refusal reason, not a merge pass that never ran: raise the cap above the
    # 7.59 s the joined window would span and the identical fixture merges
    relaxed = split_entry(parent, dataclasses.replace(CFG, max_duration=12.0), ANCHOR_AT_S1)
    assert len(relaxed) == 2


def test_anchored_reflow_is_idempotent():
    parent = sub(0.0, 7.0, f"{ANCHOR_S0} {ANCHOR_S1} {ANCHOR_S2}")

    once = reflow_with_anchors([parent], CFG, [ANCHOR_AT_S1])
    twice = reflow(once, CFG)

    assert [(s.index, s.start, s.end, s.content) for s in once] == [
        (s.index, s.start, s.end, s.content) for s in twice
    ]


# --- Regression fixtures from the reviewer's own lecture --------------------
# "Stvari koje nisu naziva kao da jesu! - 4", entries #215-#217 and #757-#759
# of the prepared Croatian source, with their real windows and text.


def test_merge_short_entries_rescues_real_entry_758_pregledu():
    subs = [
        sub(
            3703.800,
            3709.996,
            "Vi morate primijetiti da ona nije Isusa\nzanijekala ni na prvom ni na drugom",
            1,
        ),
        sub(3709.996, 3710.740, "pregledu.", 2),
        sub(3710.740, 3713.751, "I da sam ja bila sigurna\nda ce to tako biti,", 3),
    ]

    merged = merge_short_entries(subs, CFG)

    assert len(merged) == 2
    assert merged[1].content.replace("\n", " ") == (
        "pregledu. I da sam ja bila sigurna da ce to tako biti,"
    )
    assert merged[1].start == subs[1].start
    assert merged[1].end == subs[2].end
    assert all(
        (entry.end - entry.start).total_seconds() >= CFG.min_duration for entry in merged
    )


ENTRY_216 = [
    sub(
        1313.958,
        1320.440,
        "Svi uzimaju od njega, svi govore sto\non govori i toga cemo se dotaknuti.",
        1,
    ),
    sub(1321.120, 1321.507, "Ali,", 2),
    sub(
        1321.507,
        1329.243,
        "'ajmo procitati koliko je Joe Dispenza\npametniji od vecine vas i vecine krscana.",
        3,
    ),
]


def test_real_entry_216_ali_stays_short_because_both_merges_break_the_caps():
    """Documented residual, asserted through the refusal reasons: merging left
    spans 7.55 s and merging right 8.12 s, both above max_duration, and the
    right-hand text needs a third line on top of that."""
    decision = entry_merge_decision(ENTRY_216, 1, CFG)

    left = next(c for c in decision.candidates if c.left == 0)
    right = next(c for c in decision.candidates if c.right == 2)
    assert round(left.window_secs, 3) == 7.549
    assert left.refusals == (REFUSAL_WINDOW,)
    assert round(right.window_secs, 3) == 8.123
    assert right.refusals == (REFUSAL_LINES, REFUSAL_WINDOW)
    assert decision.accepted is None
    assert merge_short_entries(ENTRY_216, CFG) == ENTRY_216


def test_real_entry_216_merges_left_as_soon_as_the_window_cap_allows_it():
    """Companion proving the refusal above is the cap talking, not a no-op merge
    pass: nothing else about the fixture blocks the left-hand merge."""
    cfg = dataclasses.replace(CFG, max_duration=8.0)

    decision = entry_merge_decision(ENTRY_216, 1, cfg)
    merged = merge_short_entries(ENTRY_216, cfg)

    assert decision.accepted is not None
    assert decision.accepted.left == 0
    assert len(merged) == 2
    assert merged[0].content.replace("\n", " ").endswith("dotaknuti. Ali,")
    assert merged[0].start == ENTRY_216[0].start
    assert merged[0].end == ENTRY_216[1].end


# --- The silence cap: a merge must never put text on screen before it is spoken

LATE_SENTENCE = (
    "'ajmo procitati koliko je Joe Dispenza pametniji od vecine nas ovdje u dvorani."
)
ACROSS_SILENCE = [
    sub(0.0, 1.0, "Da.", 1),
    sub(6.0, 6.6, "Ali,", 2),
    sub(6.6, 14.0, LATE_SENTENCE, 3),
]


def test_merge_is_refused_across_a_long_silence_so_text_never_shows_early():
    """The left-hand window (6.6 s) and the joined text both fit, so only the
    5.0 s of silence between the two entries can refuse this merge -- without
    the cap "Ali," appears five seconds before it is spoken."""
    decision = entry_merge_decision(ACROSS_SILENCE, 1, CFG)

    left = next(c for c in decision.candidates if c.left == 0)
    assert left.gap_secs == 5.0
    assert left.window_secs <= CFG.max_duration
    assert left.refusals == (REFUSAL_GAP,)
    assert decision.accepted is None

    merged = merge_short_entries(ACROSS_SILENCE, CFG)
    carrier = next(entry for entry in merged if "Ali," in entry.content)
    assert carrier.start == ACROSS_SILENCE[1].start


def test_the_same_fixture_merges_once_the_silence_is_short():
    """Companion: shrink the silence and the identical text/window merges."""
    contiguous = [
        sub(5.6, 6.0, "Da.", 1),
        *ACROSS_SILENCE[1:],
    ]

    decision = entry_merge_decision(contiguous, 1, CFG)
    merged = merge_short_entries(contiguous, CFG)

    assert decision.accepted is not None
    assert decision.accepted.left == 0
    assert len(merged) == 2
    assert merged[0].content.replace("\n", " ") == "Da. Ali,"


def test_the_gap_cap_never_blocks_a_merge_of_chunks_inside_one_entry():
    """Chunks tile their parent's window, so their gap is 0 by construction --
    even a zero-second cap must leave the anchored chunk merge alone."""
    parent = sub(0.0, 7.0, f"{ANCHOR_S0} {ANCHOR_S1} {ANCHOR_S2}")
    cfg = dataclasses.replace(CFG, max_merge_gap=0.0)

    children = split_entry(parent, cfg, ANCHOR_AT_S1)

    assert len(children) == 2
    assert min(_durations(children)) >= cfg.min_duration


# --- rewrap: the translated-file path, structure frozen by the Croatian source


MEDIUM_LATIN = "Pa ce tako vjerujem biti i danas, iako to nije objava."  # 54 chars
CJK_TEXT = "这是一个很长的中文字幕条目，需要被重新换行成两行显示。"  # 27 chars > 18


def test_rewrap_preserves_count_indices_and_timestamps():
    subs = [
        sub(0.0, 3.0, MEDIUM_LATIN, 1),
        sub(3.0, 15.0, LONG_LATIN, 2),
        sub(15.0, 16.0, "Kratko.", 3),
    ]

    result = rewrap(subs, CFG)

    assert len(result) == len(subs)
    assert [entry.index for entry in result] == [1, 2, 3]
    assert [(entry.start, entry.end) for entry in result] == [
        (entry.start, entry.end) for entry in subs
    ]


def test_rewrap_wraps_a_long_line_into_two_lines_at_the_script_width():
    result = rewrap([sub(0.0, 3.0, MEDIUM_LATIN)], CFG)

    lines = result[0].content.split("\n")
    assert len(lines) == 2
    assert all(len(line) <= 42 for line in lines)
    assert " ".join(lines) == MEDIUM_LATIN


def test_rewrap_is_idempotent():
    subs = [
        sub(0.0, 3.0, MEDIUM_LATIN, 1),
        sub(3.0, 15.0, LONG_LATIN, 2),
        sub(15.0, 19.0, CJK_TEXT, 3),
    ]

    once = rewrap(subs, CFG)
    twice = rewrap(once, CFG)

    assert [(s.index, s.start, s.end, s.content) for s in once] == [
        (s.index, s.start, s.end, s.content) for s in twice
    ]


def test_rewrap_reports_overflow_instead_of_splitting_it_into_a_new_entry():
    subs = [sub(0.0, 3.0, "Kratko.", 1), sub(3.0, 15.0, LONG_LATIN, 2)]

    result = rewrap(subs, CFG)

    # the reflow would turn the 150-char entry into several entries; rewrap must not
    assert len(reflow(subs, CFG)) > 2
    assert len(result) == 2
    assert len(result[1].content.split("\n")) <= CFG.max_lines
    assert result[1].content.replace("\n", " ") == LONG_LATIN
    assert rewrap_overflow(subs, CFG) == [2]
    assert rewrap_overflow(result, CFG) == [2]  # reported again after wrapping


def test_rewrap_does_not_inject_spaces_into_a_no_space_script():
    result = rewrap([sub(0.0, 4.0, CJK_TEXT, 1)], CFG)

    lines = result[0].content.split("\n")
    assert len(lines) == 2
    assert all(len(line) <= 18 for line in lines)
    assert result[0].content.replace("\n", "") == CJK_TEXT


def test_rewrap_text_round_trips_through_srt_strings():
    content = reflow_text(
        srt.compose([sub(0.0, 12.0, LONG_LATIN, 1)], reindex=False), CFG
    )

    assert rewrap_text(content, CFG) == content  # already conforming: unchanged


# --- the third line: a rewrap fallback, never a reflow one


def test_rewrap_uses_a_third_line_when_the_text_cannot_fit_two():
    """German and Tamil run 25-45% longer than Croatian, so a 1:1 rewrap cannot
    always land in two lines. Adding a line keeps the entry count, index and
    timestamp untouched, which splitting would not."""
    long_german = (
        "Denn wer die Wahrheit sucht und nach ihr verlangt, "
        "dem wird sie auch gegeben werden von dem Vater."
    )
    cfg = YTConfig()
    entry = sub(0.0, 5.0, long_german)

    assert ytsplit.rewrap_overflow([entry], cfg) == []
    lines = ytsplit.rewrap([entry], cfg)[0].content.split("\n")

    assert len(lines) == 3
    assert all(len(line) <= cfg.default_max_chars for line in lines)
    assert " ".join(lines) == long_german


def test_rewrap_still_prefers_two_lines_when_the_text_fits():
    """The third line is a fallback, not a new default: an entry that fits two
    lines must wrap exactly as it did before, so the archive does not churn."""
    cfg = YTConfig()
    text = "A tebi je duh mrtav, i to je tvoja vlastita krivnja."
    entry = sub(0.0, 4.0, text)

    rewrapped = ytsplit.rewrap([entry], cfg)[0].content

    assert rewrapped == ytsplit.wrap_lines(text, cfg.default_max_chars, False)
    assert len(rewrapped.split("\n")) == 2


def test_rewrap_with_a_third_line_is_idempotent():
    """A three-line entry must conform on the next pass, or every reformat run
    would rewrite the same files forever."""
    cfg = YTConfig()
    long_german = (
        "Denn wer die Wahrheit sucht und nach ihr verlangt, "
        "dem wird sie auch gegeben werden von dem Vater."
    )
    once = ytsplit.rewrap([sub(0.0, 5.0, long_german)], cfg)

    assert ytsplit.rewrap(once, cfg)[0].content == once[0].content


def test_rewrap_never_changes_the_structure_even_when_it_adds_a_line():
    cfg = YTConfig()
    entries = [
        sub(0.0, 5.0, "Denn wer die Wahrheit sucht und nach ihr verlangt, "
            "dem wird sie auch gegeben werden von dem Vater.", index=7),
        sub(5.0, 9.0, "Kurz.", index=8),
    ]

    rewrapped = ytsplit.rewrap(entries, cfg)

    assert [s.index for s in rewrapped] == [7, 8]
    assert [(s.start, s.end) for s in rewrapped] == [(s.start, s.end) for s in entries]
    assert len(rewrapped) == len(entries)


def test_the_reflow_path_still_splits_instead_of_taking_a_third_line():
    """The Croatian source may create entries, so it never needs the fallback:
    a third line there would be a readability regression, not a rescue."""
    cfg = YTConfig()
    long_text = (
        "Denn wer die Wahrheit sucht und nach ihr verlangt, "
        "dem wird sie auch gegeben werden von dem Vater."
    )

    children = ytsplit.reflow([sub(0.0, 6.0, long_text)], cfg)

    assert len(children) > 1
    assert all(len(child.content.split("\n")) <= cfg.max_lines for child in children)


def test_rewrap_overflow_reports_only_what_a_third_line_cannot_save():
    cfg = YTConfig()
    unsavable = (
        "Donaudampfschiffahrtselektrizitaeten"
        "hauptbetriebswerkbauunterbeamtengesellschaft "
    ) * 2

    assert ytsplit.rewrap_overflow([sub(0.0, 5.0, unsavable)], cfg) == [1]
