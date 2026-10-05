"""Unicode-script detection driving per-script subtitle line-width limits.

YouTube's ~42 chars/line guideline assumes Latin text; denser scripts need
narrower lines. Detection picks the dominant script over letters only via
Unicode block ranges (no extra dependency).
"""

from __future__ import annotations

import enum
from collections import Counter

from srtljubav.config import YTConfig


class Script(enum.Enum):
    LATIN = "Latin"
    CYRILLIC = "Cyrillic"
    CJK = "CJK"
    HANGUL = "Hangul"
    ARABIC = "Arabic"
    HEBREW = "Hebrew"
    DEVANAGARI = "Devanagari"
    BENGALI = "Bengali"
    GURMUKHI = "Gurmukhi"
    GUJARATI = "Gujarati"
    TAMIL = "Tamil"
    TELUGU = "Telugu"
    MALAYALAM = "Malayalam"
    THAI = "Thai"
    OTHER = "Other"


# Practical max chars/line for YouTube readability, per script.
SCRIPT_MAX_CHARS: dict[Script, int] = {
    Script.LATIN: 42,
    Script.CYRILLIC: 42,
    Script.CJK: 18,
    Script.HANGUL: 20,
    Script.ARABIC: 38,
    Script.HEBREW: 38,
    Script.DEVANAGARI: 36,
    Script.BENGALI: 36,
    Script.GURMUKHI: 36,
    Script.GUJARATI: 36,
    Script.TAMIL: 36,
    Script.TELUGU: 36,
    Script.MALAYALAM: 36,
    Script.THAI: 28,
    Script.OTHER: 42,
}

# Scripts written without spaces between words: break lines at characters.
SCRIPT_NO_SPACE: frozenset[Script] = frozenset({Script.CJK, Script.THAI})

_BLOCK_RANGES: tuple[tuple[int, int, Script], ...] = (
    (0x0041, 0x024F, Script.LATIN),  # Basic Latin letters .. Latin Extended-B
    (0x0400, 0x04FF, Script.CYRILLIC),
    (0x0590, 0x05FF, Script.HEBREW),
    (0x0600, 0x06FF, Script.ARABIC),
    (0x0750, 0x077F, Script.ARABIC),  # Arabic Supplement
    (0x0900, 0x097F, Script.DEVANAGARI),
    (0x0980, 0x09FF, Script.BENGALI),
    (0x0A00, 0x0A7F, Script.GURMUKHI),
    (0x0A80, 0x0AFF, Script.GUJARATI),
    (0x0B80, 0x0BFF, Script.TAMIL),
    (0x0C00, 0x0C7F, Script.TELUGU),
    (0x0D00, 0x0D7F, Script.MALAYALAM),
    (0x0E00, 0x0E7F, Script.THAI),
    (0x1100, 0x11FF, Script.HANGUL),  # Hangul Jamo
    (0x3040, 0x30FF, Script.CJK),  # Hiragana + Katakana
    (0x3400, 0x4DBF, Script.CJK),  # CJK Extension A
    (0x4E00, 0x9FFF, Script.CJK),  # CJK Unified Ideographs
    (0xAC00, 0xD7AF, Script.HANGUL),  # Hangul Syllables
    (0xF900, 0xFAFF, Script.CJK),  # CJK Compatibility Ideographs
)


def _script_of_char(char: str) -> Script | None:
    code = ord(char)
    for low, high, script in _BLOCK_RANGES:
        if low <= code <= high:
            return script
    return None


def detect_script(text: str) -> Script:
    """Dominant script over letters; ties resolve to the narrower line limit."""
    counts: Counter[Script] = Counter()
    for char in text:
        if char.isalpha():
            script = _script_of_char(char)
            if script is not None:
                counts[script] += 1

    if not counts:
        return Script.OTHER

    best = max(counts.items(), key=lambda item: (item[1], -SCRIPT_MAX_CHARS[item[0]]))
    return best[0]


def contains_script(text: str, script: Script) -> bool:
    """True when at least one letter of *text* belongs to *script*."""
    return any(_script_of_char(char) == script for char in text if char.isalpha())


def max_chars_for(text: str, cfg: YTConfig) -> int:
    return SCRIPT_MAX_CHARS.get(detect_script(text), cfg.default_max_chars)


def breaks_at_chars(text: str) -> bool:
    """True for scripts written without inter-word spaces (CJK, Thai)."""
    return detect_script(text) in SCRIPT_NO_SPACE
