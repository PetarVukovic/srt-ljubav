"""Fragment-anchored segmentation of the Croatian source.

The transcript's fragments are the only timing truth there is, so every step
keeps each word inside the time of the fragment that carried it:

1. **Speed first, on the raw fragments.** Too-fast fragments are fixed worst
   first (``SegmentConfig.speed_tiers``): the end is extended into the silence
   that follows, the start pulled back by at most ``max_lead_in``, and only when
   no silence is left is the fragment merged with an adjacent neighbour.
   Fragments shorter than ``min_duration`` get the same treatment.
2. **Split before merging.** A fragment that will not fit two lines (or runs
   far past ``max_duration``) is cut at sentence > clause > word boundaries,
   and time is divided by characters *within that fragment only*, so every
   piece reads at the fragment's own speed.
3. **Merge into sentences.** An unfinished sentence absorbs its continuation
   when no real pause separates them (``sentence_merge_gap``) and the result
   still fits two lines, ``max_duration`` and ``merge_ok_cps``. A merged entry
   reads at the weighted mean of its parts, so merging can never speed text up.
4. **Gap and line breaks.** Consecutive entries keep ``min_gap`` of blank where
   an end can give it up; lines break at punctuation where possible and never
   after a conjunction or preposition.

The text is never changed: the words of the output are the words of the input,
in order. The legacy path merged fragments into 18 s blocks first and divided
time across each block *including its pauses*, which put text on screen
seconds before (or after) it was spoken.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import srt

from srtljubav import srtio
from srtljubav.config import SegmentConfig

_SENTENCE_END = re.compile(r"[.!?…]$")
_CLAUSE_END = re.compile(r"[,;:]$|^[-–—]$")
_CLOSERS = "\"'”’»)]"
_OPENERS = "\"'„“«(["

# Words a line or an entry must not end on: they belong to what follows.
CONJUNCTIONS_AND_PREPOSITIONS = frozenset(
    [
        "i",
        "a",
        "ali",
        "ili",
        "pa",
        "te",
        "ni",
        "niti",
        "no",
        "nego",
        "već",
        "da",
        "jer",
        "što",
        "kad",
        "kada",
        "ako",
        "kao",
        "dok",
        "čim",
        "iako",
        "premda",
        "makar",
        "mada",
        "li",
        "neka",
        "kako",
        "gdje",
        "čiji",
        "koji",
        "koja",
        "koje",
        "u",
        "na",
        "o",
        "po",
        "s",
        "sa",
        "z",
        "k",
        "ka",
        "od",
        "do",
        "iz",
        "za",
        "pred",
        "nad",
        "pod",
        "kroz",
        "prema",
        "bez",
        "kod",
        "oko",
        "među",
        "uz",
        "niz",
        "zbog",
        "radi",
        "između",
        "nakon",
        "prije",
        "poslije",
        "protiv",
        "osim",
        "umjesto",
        "tijekom",
        "unatoč",
        "usprkos",
        "pri",
    ]
)

# Tokens ending in a period that do not end a sentence.
ABBREVIATIONS = frozenset(
    [
        "br.",
        "sv.",
        "npr.",
        "tj.",
        "dr.",
        "mr.",
        "g.",
        "gđa.",
        "gđica.",
        "st.",
        "str.",
        "god.",
        "pogl.",
        "usp.",
        "tzv.",
        "prof.",
        "ing.",
        "vl.",
        "bl.",
        "pok.",
        "sl.",
        "odn.",
        "hr.",
        "engl.",
        "lat.",
        "grč.",
        "hebr.",
        "pr.",
        "kr.",
        "pr.kr.",
    ]
)


@dataclass
class _Frag:
    start: float
    end: float
    text: str
    # Where the transcript says the speech of the last word ended. ``end``
    # may grow into the silence that follows; merges measure the pause from
    # here, or a stretched fragment would be merged across a real pause.
    speech_end: float = -1.0

    def __post_init__(self) -> None:
        if self.speech_end < 0:
            self.speech_end = self.end

    def copy(self) -> _Frag:
        return _Frag(self.start, self.end, self.text, self.speech_end)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def cps(self) -> float:
        return len(self.text) / max(self.duration, 1e-3)


# --------------------------------------------------------------------------- text


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _join(left: str, right: str) -> str:
    return f"{left} {right}".strip()


def _bare(word: str) -> str:
    """Lower-cased word without surrounding quotes/brackets."""
    return word.strip(_OPENERS + _CLOSERS).lower()


def ends_sentence(text: str, next_text: str | None = None) -> bool:
    """Does *text* end a sentence, given what follows it?

    Abbreviations ("br.") never end one. An ordinal or an ellipsis ("11.",
    "pazi...") ends one only when the next fragment starts with a capital,
    and an alphabetic token before a number ("Post. 28:12") never does.
    """
    words = text.split()
    if not words:
        return False
    core = words[-1].rstrip(_CLOSERS)
    if not _SENTENCE_END.search(core):
        return False
    bare = core.lstrip(_OPENERS).lower()
    if bare in ABBREVIATIONS or re.fullmatch(r"[a-zčćžšđ]\.", bare):
        return False
    nxt = (next_text or "").lstrip(_OPENERS)
    if nxt[:1].isdigit() and re.fullmatch(r"[^\W\d_]+\.", bare):
        return False
    if re.fullmatch(r"\d+\.", bare) or core.endswith(("...", "…")):
        return not nxt or nxt[:1].isupper()
    return True


def _ends_on_function_word(word: str) -> bool:
    return word[-1:].isalpha() and _bare(word) in CONJUNCTIONS_AND_PREPOSITIONS


def _break_cost(left_word: str, right_word: str) -> float:
    """Cost of a line or entry break between two words (lower is better)."""
    if _ends_on_function_word(left_word):
        return 40.0
    if ends_sentence(left_word, right_word):
        return -20.0
    if _CLAUSE_END.search(left_word.rstrip(_CLOSERS)):
        return -12.0
    if _bare(right_word) in CONJUNCTIONS_AND_PREPOSITIONS:
        return -4.0  # breaking BEFORE a conjunction is the natural place
    if left_word[-1:].isalpha() and right_word[:1].isdigit():
        return 10.0  # keep "Matej 5:48" together
    return 0.0


def wrap(text: str, cfg: SegmentConfig, *, strict: bool = False) -> str | None:
    """Best layout of *text* in at most two lines of ``max_line_chars``.

    None when no legal layout exists. A two-line break is chosen by balance
    plus ``_break_cost``: at punctuation where possible, never after a
    conjunction or preposition, and no near-empty line. *strict* refuses a
    layout whose only legal break follows a conjunction or preposition.
    """
    text = _normalize(text)
    width = cfg.max_line_chars
    if len(text) <= width:
        return text
    if cfg.max_lines < 2 or len(text) > cfg.max_lines * width + cfg.max_lines - 1:
        return None
    words = text.split(" ")
    best: tuple[float, int] | None = None
    left_len = -1
    total = len(text)
    for i in range(1, len(words)):
        left_len += len(words[i - 1]) + 1
        right_len = total - left_len - 1
        if left_len > width or right_len > width:
            continue
        if strict and _ends_on_function_word(words[i - 1]):
            continue
        score = max(left_len, right_len) + _break_cost(words[i - 1], words[i])
        if min(left_len, right_len) < 10:
            score += 10.0
        if best is None or score < best[0]:
            best = (score, i)
    if best is None:
        return None
    i = best[1]
    return " ".join(words[:i]) + "\n" + " ".join(words[i:])


def fits(text: str, cfg: SegmentConfig) -> bool:
    """Can *text* be laid out in two lines without a dangling conjunction?"""
    return wrap(text, cfg, strict=True) is not None


# ------------------------------------------------------------------ 0. load


def _load(subs: Sequence[srt.Subtitle]) -> list[_Frag]:
    """Fragments in order, overlaps clamped, empty text dropped."""
    frags: list[_Frag] = []
    raw = [
        (sub.start.total_seconds(), sub.end.total_seconds(), _normalize(sub.content))
        for sub in subs
    ]
    raw = [item for item in raw if item[2]]
    for index, (start, end, text) in enumerate(raw):
        if frags:
            start = max(start, frags[-1].end)
        if index + 1 < len(raw):
            end = min(end, max(raw[index + 1][0], start))
        frags.append(_Frag(start, max(start, end), text))
    return frags


# -------------------------------------------------------- 1. speed and floor


def _room_after(frags: list[_Frag], i: int, cfg: SegmentConfig) -> float:
    if i + 1 >= len(frags):
        return float("inf")
    return max(0.0, frags[i + 1].start - cfg.min_gap - frags[i].end)


def _room_before(frags: list[_Frag], i: int, cfg: SegmentConfig) -> float:
    floor = frags[i - 1].end + cfg.min_gap if i > 0 else 0.0
    return max(0.0, min(cfg.max_lead_in, frags[i].start - floor))


def _stretch(frags: list[_Frag], i: int, needed: float, cfg: SegmentConfig) -> bool:
    """Grow fragment *i* towards *needed* seconds using only silence."""
    frag = frags[i]
    missing = needed - frag.duration
    if missing <= 1e-6:
        return False
    grow = min(missing, _room_after(frags, i, cfg))
    if grow > 1e-6:
        frag.end += grow
        missing -= grow
    shift = min(missing, _room_before(frags, i, cfg)) if missing > 1e-6 else 0.0
    if shift > 1e-6:
        frag.start -= shift
    return grow > 1e-6 or shift > 1e-6


def _merged(left: _Frag, right: _Frag) -> _Frag:
    return _Frag(left.start, right.end, _join(left.text, right.text), right.speech_end)


def _merge_ok(left: _Frag, right: _Frag, max_gap: float, cfg: SegmentConfig) -> bool:
    merged = _merged(left, right)
    return (
        right.start - left.speech_end <= max_gap + 1e-6
        and merged.duration <= cfg.max_duration + 1e-6
        and merged.cps <= cfg.merge_ok_cps + 1e-6
        and fits(merged.text, cfg)
    )


def _rescue_merge(frags: list[_Frag], i: int, cfg: SegmentConfig) -> bool:
    """Merge fragment *i* into the neighbour that yields the slowest entry."""
    options = []
    if i > 0 and _merge_ok(frags[i - 1], frags[i], cfg.rescue_merge_gap, cfg):
        options.append((_merged(frags[i - 1], frags[i]).cps, i - 1))
    if i + 1 < len(frags) and _merge_ok(frags[i], frags[i + 1], cfg.rescue_merge_gap, cfg):
        options.append((_merged(frags[i], frags[i + 1]).cps, i))
    if not options:
        return False
    _cps, left = min(options)
    frags[left : left + 2] = [_merged(frags[left], frags[left + 1])]
    return True


def fix_speed(frags: list[_Frag], cfg: SegmentConfig) -> list[_Frag]:
    """Bring every fragment under each speed tier in turn, worst first."""
    frags = [f.copy() for f in frags]
    for tier in cfg.speed_tiers:
        stuck: set[tuple[float, str]] = set()
        while True:
            candidates = [
                (frag.cps, i)
                for i, frag in enumerate(frags)
                if frag.cps > tier + 1e-6 and (frag.start, frag.text) not in stuck
            ]
            if not candidates:
                break
            _cps, i = max(candidates)
            _stretch(frags, i, len(frags[i].text) / cfg.target_cps, cfg)
            if frags[i].cps <= tier + 1e-6:
                continue
            if _rescue_merge(frags, i, cfg):
                continue
            _borrow(frags, i, len(frags[i].text) / cfg.target_cps, cfg)
            stuck.add((frags[i].start, frags[i].text))
    return frags


def _spare(frag: _Frag, cfg: SegmentConfig) -> float:
    """Seconds *frag* could give up and still be readable at ``target_cps``."""
    needed = max(cfg.min_duration, len(frag.text) / cfg.target_cps)
    return max(0.0, frag.duration - needed)


def _borrow(frags: list[_Frag], i: int, needed: float, cfg: SegmentConfig) -> None:
    """Take the time fragment *i* still lacks from slow neighbours.

    The last resort for a fragment wedged between two entries with no silence
    and no legal merge: the preceding entry ends a little earlier, or the
    following one starts a little later, by no more than either can spare.
    """
    frag = frags[i]
    missing = needed - frag.duration
    # Only across a boundary with no silence left: where a pause remains, the
    # start has already taken its full lead-in and must not be pulled further.
    if missing > 1e-6 and i > 0 and frag.start - frags[i - 1].end <= cfg.min_gap + 1e-6:
        prev = frags[i - 1]
        take = min(missing, _spare(prev, cfg))
        if take > 1e-6:
            gap = frag.start - prev.end
            prev.end -= take
            prev.speech_end = min(prev.speech_end, prev.end)
            frag.start = prev.end + gap
            missing -= take
    if (
        missing > 1e-6
        and i + 1 < len(frags)
        and frags[i + 1].start - frag.end <= cfg.min_gap + 1e-6
    ):
        nxt = frags[i + 1]
        take = min(missing, _spare(nxt, cfg))
        if take > 1e-6:
            gap = nxt.start - frag.end
            nxt.start += take
            frag.end = nxt.start - gap


def fix_short(frags: list[_Frag], cfg: SegmentConfig, *, borrow: bool = False) -> list[_Frag]:
    """Give every fragment ``min_duration``: silence, then a merge, then (when
    *borrow*) time from a slow neighbour."""
    frags = [f.copy() for f in frags]
    i = 0
    while i < len(frags):
        if frags[i].duration < cfg.min_duration - 1e-6:
            _stretch(frags, i, cfg.min_duration, cfg)
            if frags[i].duration < cfg.min_duration - 1e-6 and _rescue_merge(frags, i, cfg):
                i = max(0, i - 1)
                continue
            if borrow and frags[i].duration < cfg.min_duration - 1e-6:
                _borrow(frags, i, cfg.min_duration, cfg)
        i += 1
    return frags


# ----------------------------------------------------------------- 2. split


def _split_text(text: str, duration: float, cfg: SegmentConfig) -> list[str]:
    """Cheapest cut of *text* into pieces that each fit two lines.

    Dynamic programme over word boundaries: every piece costs 100 (fewest
    pieces first), each cut costs ``_break_cost``, and pieces that would be
    too short to read or would run past ``max_duration`` at the fragment's own
    speed are penalised.
    """
    words = text.split(" ")
    count = len(words)
    total = len(text)
    prefix = [0]
    for word in words:
        prefix.append(prefix[-1] + len(word) + 1)

    def piece_cost(i: int, j: int) -> float | None:
        piece = " ".join(words[i:j])
        if not fits(piece, cfg):
            return None
        cost = 100.0
        seconds = duration * len(piece) / total if total else 0.0
        if len(piece) < 15:
            cost += 40.0
        if seconds < cfg.min_duration:
            cost += 25.0
        if seconds > cfg.max_duration:
            cost += 120.0
        return cost

    best: list[tuple[float, int] | None] = [None] * (count + 1)
    best[0] = (0.0, -1)
    for j in range(1, count + 1):
        for i in range(j - 1, -1, -1):
            if prefix[j] - prefix[i] - 1 > cfg.max_lines * (cfg.max_line_chars + 1):
                break
            if best[i] is None:
                continue
            cost = piece_cost(i, j)
            if cost is None:
                continue
            if j < count:
                cost += _break_cost(words[j - 1], words[j])
            total_cost = best[i][0] + cost
            if best[j] is None or total_cost < best[j][0]:
                best[j] = (total_cost, i)
    if best[count] is None:
        return [text]
    cuts = []
    j = count
    while j > 0:
        i = best[j][1]
        cuts.append(" ".join(words[i:j]))
        j = i
    return cuts[::-1]


def split_long(frags: list[_Frag], cfg: SegmentConfig) -> list[_Frag]:
    """Cut fragments that do not fit; time shared by characters within each."""
    out: list[_Frag] = []
    for frag in frags:
        if fits(frag.text, cfg) and frag.duration <= cfg.max_duration * 1.1:
            out.append(frag)
            continue
        pieces = _split_text(frag.text, frag.duration, cfg)
        weights = [len(piece) for piece in pieces]
        start = frag.start
        consumed = 0
        for index, piece in enumerate(pieces):
            consumed += weights[index]
            end = (
                frag.end
                if index == len(pieces) - 1
                else frag.start + frag.duration * consumed / sum(weights)
            )
            last = index == len(pieces) - 1
            out.append(_Frag(start, end, piece, frag.speech_end if last else end))
            start = end
    return out


# ----------------------------------------------------------------- 3. merge


def merge_sentences(frags: list[_Frag], cfg: SegmentConfig) -> list[_Frag]:
    """Join an unfinished sentence with its continuation across no pause."""
    out: list[_Frag] = []
    for frag in frags:
        if (
            out
            and not ends_sentence(out[-1].text, frag.text)
            and _merge_ok(out[-1], frag, cfg.sentence_merge_gap, cfg)
        ):
            out[-1] = _merged(out[-1], frag)
        else:
            out.append(frag.copy())
    return out


# ------------------------------------------------------------------- 4. gap


def enforce_min_gap(frags: list[_Frag], cfg: SegmentConfig) -> list[_Frag]:
    """Leave ``min_gap`` between entries by trimming ends that can spare it."""
    frags = [f.copy() for f in frags]
    for i in range(len(frags) - 1):
        frag, nxt = frags[i], frags[i + 1]
        if nxt.start - frag.end >= cfg.min_gap - 1e-6:
            continue
        new_end = nxt.start - cfg.min_gap
        if new_end - frag.start < cfg.min_duration:
            continue
        ceiling = max(cfg.merge_ok_cps, frag.cps)
        if len(frag.text) / (new_end - frag.start) > ceiling + 1e-6:
            continue
        frag.end = new_end
    return frags


# ------------------------------------------------------------------ public


def segment(subs: Sequence[srt.Subtitle], cfg: SegmentConfig) -> list[srt.Subtitle]:
    """Segment raw transcript fragments into readable two-line entries."""
    frags = _load(subs)
    frags = fix_speed(frags, cfg)
    frags = fix_short(frags, cfg)
    frags = split_long(frags, cfg)
    frags = merge_sentences(frags, cfg)
    frags = fix_short(frags, cfg, borrow=True)
    frags = enforce_min_gap(frags, cfg)
    return [
        srt.Subtitle(
            index=index,
            start=srtio.seconds_to_td(frag.start),
            end=srtio.seconds_to_td(frag.end),
            content=wrap(frag.text, cfg) or frag.text,
        )
        for index, frag in enumerate(frags, start=1)
    ]


def segment_text(content: str, cfg: SegmentConfig) -> str:
    return srtio.compose(segment(srtio.parse(content), cfg))


__all__ = [
    "ABBREVIATIONS",
    "CONJUNCTIONS_AND_PREPOSITIONS",
    "ends_sentence",
    "fits",
    "segment",
    "segment_text",
    "wrap",
]
