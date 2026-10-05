"""Thin, typed facade over the ``srt`` library.

The rest of the package operates on ``list[srt.Subtitle]`` and never deals
with raw timestamp strings or encodings directly.
"""

from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path

import srt

ENCODINGS = ("utf-8-sig", "utf-16", "windows-1250", "windows-1252", "latin-1")

# "00:00:00,1 --> 00:00:00,9" or "00:00:03 --> 00:00:04" need millis padding:
# the srt library would read ",1" as 1 ms instead of 100 ms.
_TIMESTAMP_LINE_RE = re.compile(
    r"^(\d{2}:\d{2}:\d{2})(?:,(\d{1,3}))?[ \t]*-->[ \t]*(\d{2}:\d{2}:\d{2})(?:,(\d{1,3}))?[ \t]*$",
    re.MULTILINE,
)


def decode_bytes(raw: bytes) -> str | None:
    """Decode SRT bytes trying common encodings; None if all fail."""
    for encoding in ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def read_srt_text(path: Path) -> str:
    text = decode_bytes(path.read_bytes())
    if text is None:
        raise RuntimeError(f"Cannot decode SRT file: {path}")
    return text


def _pad_millis(millis: str | None) -> str:
    return ((millis or "") + "000")[:3]


def normalize_timestamps(content: str) -> str:
    """Normalize line endings and pad short/missing milliseconds to 3 digits."""
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    def _fix(m: re.Match[str]) -> str:
        return f"{m.group(1)},{_pad_millis(m.group(2))} --> {m.group(3)},{_pad_millis(m.group(4))}"

    return _TIMESTAMP_LINE_RE.sub(_fix, normalized)


def parse(content: str) -> list[srt.Subtitle]:
    """Tolerant parse: normalizes timestamps first, keeps empty-content entries."""
    return list(srt.parse(normalize_timestamps(content)))


def compose(subs: list[srt.Subtitle]) -> str:
    """Compose with sequential 1..N indices."""
    return srt.compose(reindex(subs), reindex=False)


def reindex(subs: list[srt.Subtitle]) -> list[srt.Subtitle]:
    """Return copies renumbered sequentially 1..N (timestamps untouched)."""
    return [
        srt.Subtitle(index=i, start=sub.start, end=sub.end, content=sub.content)
        for i, sub in enumerate(subs, start=1)
    ]


def limit_subs(subs: list[srt.Subtitle], n: int | None) -> list[srt.Subtitle]:
    if n is None or n < 1:
        return subs
    return subs[:n]


def td_to_seconds(td: timedelta) -> float:
    return td.total_seconds()


def seconds_to_td(seconds: float) -> timedelta:
    return timedelta(milliseconds=round(max(0.0, seconds) * 1000))
