"""Batch coverage guard tests.

gemini-srt-translator only checks the COUNT of lines the model returns for a
batch; a response that duplicates some indexes and skips a contiguous run
passes that check and silently keeps the Croatian original for the skipped
lines. ``CachedGeminiSRTTranslator._process_translated_lines`` turns that into
a loud ValueError so gst's own outer retry loop re-runs exactly that batch.
"""

import inspect
from datetime import timedelta

import pytest
import srt
from gemini_srt_translator.main import GeminiSRTTranslator

from srtljubav.translator import CachedGeminiSRTTranslator


def make_translator() -> CachedGeminiSRTTranslator:
    # __init__ wants API configuration; the guard only needs the method.
    return CachedGeminiSRTTranslator.__new__(CachedGeminiSRTTranslator)


def make_subs(texts: list[str]) -> list[srt.Subtitle]:
    return [
        srt.Subtitle(
            index=i + 1,
            start=timedelta(seconds=i),
            end=timedelta(seconds=i + 1),
            content=text,
        )
        for i, text in enumerate(texts)
    ]


def test_gst_private_method_signature_still_matches_our_override():
    """The override targets a private gst method; a dependency bump that
    changes this signature must fail loudly here, not corrupt translations."""
    params = list(
        inspect.signature(GeminiSRTTranslator._process_translated_lines).parameters
    )
    assert params == ["self", "translated_lines", "translated_subtitle", "batch", "finished"]


def test_complete_batch_passes_and_writes_translations():
    translator = make_translator()
    subs = make_subs(["Bok", "Dobar dan"])
    batch = [{"index": "0", "text": "Bok"}, {"index": "1", "text": "Dobar dan"}]
    lines = [{"index": "0", "text": "Hello"}, {"index": "1", "text": "Good day"}]

    translator._process_translated_lines(
        translated_lines=lines, translated_subtitle=subs, batch=batch, finished=True
    )

    assert subs[0].content == "Hello"
    assert subs[1].content == "Good day"


def test_empty_translation_retries_batch_instead_of_copying_source():
    translator = make_translator()
    subs = make_subs(["Isus Krist", "Dobar dan"])
    batch = [{"index": "0", "text": "Isus Krist"}, {"index": "1", "text": "Dobar dan"}]
    lines = [{"index": "0", "text": ""}, {"index": "1", "text": "こんにちは"}]

    with pytest.raises(ValueError, match="empty|unusable"):
        translator._process_translated_lines(
            translated_lines=lines, translated_subtitle=subs, batch=batch, finished=True
        )

    assert subs[0].content == "Isus Krist"
    assert subs[1].content == "Dobar dan"


def test_missing_line_retries_batch_instead_of_copying_source():
    translator = make_translator()
    subs = make_subs(["Prva", "Druga", "Treća"])
    batch = [
        {"index": "0", "text": "Prva"},
        {"index": "1", "text": "Druga"},
        {"index": "2", "text": "Treća"},
    ]
    lines = [{"index": "0", "text": "第一"}, {"index": "2", "text": "第三"}]

    with pytest.raises(ValueError, match="missing|unusable|coverage"):
        translator._process_translated_lines(
            translated_lines=lines, translated_subtitle=subs, batch=batch, finished=True
        )

    assert [sub.content for sub in subs] == ["Prva", "Druga", "Treća"]


def test_duplicated_indexes_hiding_a_skipped_line_retry_batch():
    translator = make_translator()
    subs = make_subs(["Bok", "Dobar dan", "Laku noć"])
    batch = [
        {"index": "0", "text": "Bok"},
        {"index": "1", "text": "Dobar dan"},
        {"index": "2", "text": "Laku noć"},
    ]
    # Count matches (3 lines) but index 2 is missing and index 0 duplicated —
    # exactly the shape gst's count-only check cannot see.
    lines = [
        {"index": "0", "text": "Hello"},
        {"index": "0", "text": "Hello again"},
        {"index": "1", "text": "Good day"},
    ]

    with pytest.raises(ValueError, match="duplicated|missing|unusable|coverage"):
        translator._process_translated_lines(
            translated_lines=lines, translated_subtitle=subs, batch=batch, finished=True
        )

    assert [sub.content for sub in subs] == ["Bok", "Dobar dan", "Laku noć"]


def test_three_consecutive_source_echoes_retry_batch():
    translator = make_translator()
    source = [
        "Ovo je prva netrivijalna hrvatska rečenica.",
        "Ovo je druga netrivijalna hrvatska rečenica.",
        "Ovo je treća netrivijalna hrvatska rečenica.",
        "Ovo je četvrta netrivijalna hrvatska rečenica.",
    ]
    subs = make_subs(source)
    batch = [{"index": str(i), "text": text} for i, text in enumerate(source)]
    lines = [
        {"index": "0", "text": "This line is translated."},
        {"index": "1", "text": source[1]},
        {"index": "2", "text": source[2]},
        {"index": "3", "text": source[3]},
    ]

    with pytest.raises(ValueError, match="3 consecutive"):
        translator._process_translated_lines(
            translated_lines=lines, translated_subtitle=subs, batch=batch, finished=True
        )


def test_two_consecutive_source_echoes_are_tolerated():
    translator = make_translator()
    source = [
        "Ovo je prva netrivijalna hrvatska rečenica.",
        "Ovo je druga netrivijalna hrvatska rečenica.",
        "Ovo je treća netrivijalna hrvatska rečenica.",
    ]
    subs = make_subs(source)
    batch = [{"index": str(i), "text": text} for i, text in enumerate(source)]
    lines = [
        {"index": "0", "text": "This line is translated."},
        {"index": "1", "text": source[1]},
        {"index": "2", "text": source[2]},
    ]

    translator._process_translated_lines(
        translated_lines=lines, translated_subtitle=subs, batch=batch, finished=True
    )

    assert subs[0].content == "This line is translated."


def test_partial_stream_chunk_is_not_guarded():
    translator = make_translator()
    subs = make_subs(["Bok", "Dobar dan"])
    batch = [{"index": "0", "text": "Bok"}, {"index": "1", "text": "Dobar dan"}]

    translator._process_translated_lines(
        translated_lines=[{"index": "0", "text": "Hello"}],
        translated_subtitle=subs,
        batch=batch,
        finished=False,
    )

    assert subs[0].content == "Hello"
