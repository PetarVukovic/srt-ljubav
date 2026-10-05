from datetime import timedelta

from srtljubav import srtio


def test_parse_pads_short_and_missing_millis():
    content = "1\n00:00:00,1 --> 00:00:00,9\nBog je ljubav\n\n2\n00:00:03 --> 00:00:04\nx\n\n"

    subs = srtio.parse(content)

    assert subs[0].start == timedelta(milliseconds=100)
    assert subs[0].end == timedelta(milliseconds=900)
    assert subs[1].start == timedelta(seconds=3)
    assert subs[1].end == timedelta(seconds=4)


def test_parse_keeps_empty_content_entries():
    content = (
        "1\n00:00:01,000 --> 00:00:02,000\n\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nText\n\n"
    )

    subs = srtio.parse(content)

    assert len(subs) == 2
    assert subs[0].content == ""


def test_compose_round_trip_is_stable():
    content = (
        "1\n00:00:01,000 --> 00:00:02,500\nPrvi dio.\n\n"
        "2\n00:00:02,500 --> 00:00:04,000\nDrugi dio,\nu dva retka.\n\n"
    )

    once = srtio.compose(srtio.parse(content))
    twice = srtio.compose(srtio.parse(once))

    assert once == twice
    assert "Drugi dio,\nu dva retka." in once


def test_reindex_renumbers_sequentially():
    subs = srtio.parse(
        "7\n00:00:01,000 --> 00:00:02,000\nA\n\n9\n00:00:02,000 --> 00:00:03,000\nB\n\n"
    )

    renumbered = srtio.reindex(subs)

    assert [s.index for s in renumbered] == [1, 2]
    assert [s.content for s in renumbered] == ["A", "B"]


def test_decode_bytes_fallbacks():
    assert srtio.decode_bytes("čćžšđ".encode("utf-8-sig")) == "čćžšđ"
    assert srtio.decode_bytes("čćžšđ".encode("utf-16")) == "čćžšđ"
    # windows-1250 Croatian text
    assert srtio.decode_bytes("čćžšđ".encode("windows-1250")) is not None


def test_limit_subs():
    subs = srtio.parse(
        "1\n00:00:01,000 --> 00:00:02,000\nA\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nB\n\n"
        "3\n00:00:03,000 --> 00:00:04,000\nC\n\n"
    )

    assert len(srtio.limit_subs(subs, 2)) == 2
    assert srtio.limit_subs(subs, None) is subs


def test_seconds_td_round_trip():
    td = srtio.seconds_to_td(1.2346)
    assert td == timedelta(milliseconds=1235)  # ms rounding
    assert srtio.td_to_seconds(timedelta(seconds=2, milliseconds=500)) == 2.5
