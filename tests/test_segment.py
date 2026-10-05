from datetime import timedelta

import pytest
import srt

from srtljubav import segment, srtio
from srtljubav.config import SegmentConfig

CFG = SegmentConfig()

# Vrijeme je kraja 14, 49:15-49:33 -- the example of Andrija's 2026-09-30
# report: the legacy block reflow squeezed "To znači da postignem
# savršenstvo." into 1.27 s (38.6 cps) and stretched "Matej 5:48." over 7 s
# together with words spoken 4-6 s later.
VJK14_EXAMPLE = """\
1200
00:49:15,320 --> 00:49:17,780
I samo najmanje stado Mojih Vlastitih

1201
00:49:17,780 --> 00:49:19,320
živi način života

1202
00:49:19,320 --> 00:49:21,800
u ozbiljnom stremljenju prema

1203
00:49:21,800 --> 00:49:23,620
Meni. To znači da postignem

1204
00:49:23,620 --> 00:49:25,540
savršenstvo. Koje je u Ocu.

1205
00:49:25,540 --> 00:49:27,600
Matej 5:48.

1206
00:49:29,820 --> 00:49:31,120
Što, dakle,

1207
00:49:31,820 --> 00:49:32,720
način života
"""

# Vrijeme je kraja 16, opening minute, verbatim.
VJK16_OPENING = [
    (30.92, 33.62, "Ovo je šesnaesto predavanje"),
    (33.62, 36.62, "seminara Vrijeme je Kraja."),
    (38.46, 41.62, "Prema Knjižici devedeset i osam"),
    (41.62, 42.84, "istog naziva,"),
    (44.68, 47.34, "a ususret novom"),
    (47.34, 49.66, "licu stranice"),
    (49.66, 53.98, "www.ljubavseohladila.com"),
    (56.67, 57.57, "Znate"),
    (58.64, 61.83, "da odmah uđem u Objavu koju sam"),
    (61.83, 64.67, "malo prije i preveo, dakle"),
    (64.67, 67.93, "novo prevedena Objava, što bi bila Objava"),
    (67.93, 69.79, "4331."),
    (72.25, 74.73, "Preskačem naslov, prva rečenica"),
    (74.73, 77.19, "razlog za Objavu. Kaže"),
    (77.53, 80.15, "sve upućuje na skori Kraj."),
    (81.13, 83.37, "Tako će se zvati ovo predavanje."),
    (85.01, 87.71, "E sad, ako ti osobno, ti"),
    (87.71, 90.97, "ime i prezime, ti jedan od ovih prozorčića"),
    (91.35, 92.65, "ne vidiš"),
    (93.67, 96.17, "kad pogledaš oko sebe"),
    (96.17, 99.87, "da sve upućuje na skori Kraj"),
]


def subs_of(blocks: list[tuple[float, float, str]]) -> list[srt.Subtitle]:
    return [
        srt.Subtitle(i, srtio.seconds_to_td(a), srtio.seconds_to_td(b), text)
        for i, (a, b, text) in enumerate(blocks, 1)
    ]


def spans(subs: list[srt.Subtitle]) -> list[tuple[float, float, str]]:
    return [
        (s.start.total_seconds(), s.end.total_seconds(), " ".join(s.content.split())) for s in subs
    ]


def test_vjk14_example_keeps_every_entry_on_its_fragments():
    out = spans(segment.segment(srtio.parse(VJK14_EXAMPLE), CFG))

    assert out == [
        (
            2955.32,
            2961.72,
            "I samo najmanje stado Mojih Vlastitih živi način života u ozbiljnom stremljenju prema",
        ),
        (2961.80, 2965.46, "Meni. To znači da postignem savršenstvo. Koje je u Ocu."),
        (2965.54, 2967.60, "Matej 5:48."),
        (2969.82, 2971.12, "Što, dakle,"),
        (2971.82, 2972.82, "način života"),
    ]


def test_output_keeps_the_text_word_for_word_and_a_legal_layout():
    source = subs_of(VJK16_OPENING)

    out = segment.segment(source, CFG)

    assert [w for s in out for w in s.content.split()] == [
        w for _a, _b, text in VJK16_OPENING for w in text.split()
    ]
    for entry in out:
        lines = entry.content.split("\n")
        assert len(lines) <= 2
        assert all(len(line) <= 42 for line in lines)
    for earlier, later in zip(out, out[1:], strict=False):
        assert earlier.end <= later.start, "entries must never overlap"
    assert [s.index for s in out] == list(range(1, len(out) + 1))


def test_an_unfinished_sentence_never_merges_across_a_pause():
    out = spans(
        segment.segment(
            subs_of(
                [
                    (0.0, 2.0, "Kad pogledaš oko sebe"),
                    (3.0, 5.0, "vidiš da sve upućuje na Kraj."),
                ]
            ),
            CFG,
        )
    )

    assert [text for _a, _b, text in out] == [
        "Kad pogledaš oko sebe",
        "vidiš da sve upućuje na Kraj.",
    ]


def test_an_unfinished_sentence_absorbs_its_continuation_without_a_pause():
    out = spans(
        segment.segment(
            subs_of(
                [
                    (0.0, 2.0, "Kad pogledaš oko sebe"),
                    (2.1, 4.0, "vidiš da sve upućuje na Kraj."),
                ]
            ),
            CFG,
        )
    )

    assert out == [(0.0, 4.0, "Kad pogledaš oko sebe vidiš da sve upućuje na Kraj.")]


def test_a_finished_sentence_is_not_merged_with_the_next():
    out = spans(
        segment.segment(
            subs_of(
                [
                    (0.0, 2.0, "To je zakon."),
                    (2.0, 4.0, "Sad idemo dalje."),
                ]
            ),
            CFG,
        )
    )

    assert len(out) == 2


def test_a_too_fast_fragment_takes_the_silence_after_it_and_at_most_a_lead_in():
    text = "Ovo je jedna vrlo brzo izgovorena rečenica."
    out = spans(
        segment.segment(
            subs_of(
                [
                    (0.0, 4.0, "Prethodna rečenica koja traje dugo i puni cijeli redak."),
                    (4.6, 5.6, text),
                    (8.0, 12.0, "Sljedeća rečenica dolazi tek nakon duže stanke ovdje."),
                ]
            ),
            CFG,
        )
    )

    fast = next(entry for entry in out if entry[2] == text)
    assert fast[0] == 4.6, "silence after the fragment is enough: the start stays"
    assert len(text) / (fast[1] - fast[0]) <= CFG.target_cps + 0.1


def test_a_lead_in_never_exceeds_max_lead_in():
    text = "Ovo je jedna vrlo brzo izgovorena rečenica."
    out = spans(
        segment.segment(
            subs_of(
                [
                    (0.0, 4.0, "Prethodna rečenica koja traje dugo i puni cijeli redak."),
                    (4.6, 5.6, text),
                    (5.6, 12.0, "Sljedeća rečenica počinje odmah i traje dugo, dugo, dugo."),
                ]
            ),
            CFG,
        )
    )

    fast = next(entry for entry in out if entry[2] == text)
    assert 4.6 - CFG.max_lead_in - 1e-9 <= fast[0] <= 4.6


def test_a_fragment_that_does_not_fit_is_split_inside_its_own_window():
    text = (
        "Ovo je prva rečenica koja je dosta duga. Ovo je druga rečenica, "
        "koja je isto tako duga i ne stane nikako u dva retka teksta."
    )
    out = spans(segment.segment(subs_of([(10.0, 18.0, text)]), CFG))

    assert len(out) > 1
    assert out[0][0] == 10.0
    assert out[-1][1] == 18.0
    assert all(10.0 <= a < b <= 18.0 for a, b, _t in out)
    assert out[0][2].endswith("duga."), "the cut prefers the sentence end"
    # time is shared by characters within the fragment: equal reading speed
    speeds = [len(t) / (b - a) for a, b, t in out]
    assert max(speeds) - min(speeds) < 3.0


def test_a_short_fragment_with_no_silence_borrows_from_a_slow_neighbour():
    out = spans(
        segment.segment(
            subs_of(
                [
                    (0.0, 6.8, "Prva rečenica koja traje dugo."),
                    (6.8, 7.28, "Znate."),
                    (7.28, 14.0, "Druga rečenica koja isto traje dugo."),
                ]
            ),
            CFG,
        )
    )

    short = next(entry for entry in out if entry[2] == "Znate.")
    assert short[1] - short[0] >= CFG.min_duration - 1e-6


@pytest.mark.parametrize(
    ("text", "following", "expected"),
    [
        ("To je kraj.", "Sad dalje", True),
        ("kaže br.", "Sad", False),
        ("Ovo je 11.", "predavanje", False),
        ("Objava 4331.", "Preskačem", True),
        ("pogledaj Post.", "28:12", False),
        ("pazi...", "na skori Kraj", False),
        ("pazi...", "Sve upućuje", True),
        ('Unatoč tome "Motiv".', "Sad", True),
        ("Što, dakle,", "način", False),
    ],
)
def test_ends_sentence(text, following, expected):
    assert segment.ends_sentence(text, following) is expected


def test_wrap_never_leaves_a_conjunction_at_the_end_of_line_one():
    wrapped = segment.wrap("da bi pokazao zbunjenima s obje strane da je to isto", CFG)

    first = wrapped.split("\n")[0]
    assert first.split()[-1] not in segment.CONJUNCTIONS_AND_PREPOSITIONS


def test_wrap_breaks_at_punctuation_where_it_can():
    wrapped = segment.wrap("Meni. To znači da postignem savršenstvo. Koje je u Ocu.", CFG)

    assert wrapped == "Meni. To znači da postignem savršenstvo.\nKoje je u Ocu."


def test_wrap_rejects_text_beyond_two_lines():
    assert segment.wrap("riječ " * 30, CFG) is None


def test_segment_text_round_trips_through_srt():
    content = segment.segment_text(VJK14_EXAMPLE, CFG)

    assert srtio.parse(content)[0].start == timedelta(minutes=49, seconds=15, milliseconds=320)
