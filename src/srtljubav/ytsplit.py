"""Segmentation and line wrapping of YouTube-friendly subtitle entries.

``reflow`` splits entries that are too long for comfortable reading into small
entries (≤2 lines, per-script char limits, ~1–7 s each), allocating time
proportionally to text length within the parent entry's window. It runs on the
Croatian source only: that file fixes the segmentation for the whole lecture.

``rewrap`` is the translated-file counterpart — it re-wraps the lines inside
each entry at the target script's width and never touches the entry count,
indices or timestamps, so every ``{stem}_{Language}.srt`` stays 1:1 with its
``_Croatian.srt`` sibling.

Both are pure and idempotent: ``reflow(reflow(x)) == reflow(x)``.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta

import srt

from srtljubav import scripts, srtio
from srtljubav.config import YTConfig

TimeAnchor = tuple[int, timedelta]

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+|(?<=[。！？])")
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:])\s+|(?<=[、，；：])")
_CJK_TRAILING_PUNCT = "。，、！？；：」』"

# Characters a line must never start with: closing punctuation, the prolonged
# sound mark, small kana (they attach to the preceding character), Thai repeat
# and abbreviation marks.
_NO_BREAK_BEFORE = frozenset(
    _CJK_TRAILING_PUNCT + "ーぁぃぅぇぉゃゅょっんァィゥェォャュョッンヶヵゝゞヽヾ々ๆฯ"
)
# Thai leading vowels are written BEFORE the consonant they belong to —
# a line must never end with one.
_THAI_LEAD_VOWELS = frozenset("เแโใไ")


def _legal_break(text: str, pos: int) -> bool:
    """May the text be broken between pos-1 and pos? (char-mode scripts)"""
    if pos <= 0 or pos >= len(text):
        return False
    prev, nxt = text[pos - 1], text[pos]
    if unicodedata.category(nxt) in ("Mn", "Mc"):  # combining marks attach left
        return False
    if nxt in _NO_BREAK_BEFORE:
        return False
    if prev in _THAI_LEAD_VOWELS:
        return False
    # never split runs of digits/Latin embedded in CJK/Thai text ("第100号", "Romans")
    return not (prev.isascii() and prev.isalnum() and nxt.isascii() and nxt.isalnum())


def _nearest_legal_break(text: str, pos: int, lo: int, hi: int) -> int:
    """Closest legal break to pos within [lo, hi]; pos itself if none exists."""
    for delta in range(0, hi - lo + 1):
        for candidate in (pos + delta, pos - delta):
            if lo <= candidate <= hi and _legal_break(text, candidate):
                return candidate
    return pos

# Proportional time shares drift slightly when whitespace normalization changes
# chunk lengths between passes; only re-split when clearly over the limit so
# reflow stays idempotent (a child at e.g. 7.05 s must not split on the next pass).
_DURATION_TOLERANCE = 1.1


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _units(text: str, pattern: re.Pattern[str]) -> list[str]:
    return [part.strip() for part in pattern.split(text) if part.strip()]


def _join(parts: list[str], sep: str) -> str:
    return sep.join(parts)


def _hard_split(text: str, capacity: int, char_mode: bool) -> list[str]:
    """Split text with no usable boundaries into balanced pieces <= capacity."""
    target = math.ceil(len(text) / math.ceil(len(text) / capacity))
    if char_mode:
        pieces = []
        rest = text
        while len(rest) > target:
            cut = _nearest_legal_break(rest, target, 1, min(capacity, len(rest) - 1))
            pieces.append(rest[:cut])
            rest = rest[cut:]
        if rest:
            pieces.append(rest)
        return pieces

    pieces: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = word if not current else f"{current} {word}"
        if current and len(candidate) > target:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _chunk_text(text: str, capacity: int, char_mode: bool) -> list[str]:
    """Chunk text into pieces <= capacity, preferring sentence > clause > word breaks."""
    if len(text) <= capacity:
        return [text]

    sep = "" if char_mode else " "
    pieces: list[str] = []
    for sentence in _units(text, _SENT_SPLIT):
        if len(sentence) <= capacity:
            pieces.append(sentence)
            continue
        for clause in _units(sentence, _CLAUSE_SPLIT):
            if len(clause) <= capacity:
                pieces.append(clause)
            else:
                pieces.extend(_hard_split(clause, capacity, char_mode))

    total_len = sum(len(p) for p in pieces) + len(sep) * max(0, len(pieces) - 1)
    target = math.ceil(total_len / math.ceil(total_len / capacity))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = piece if not current else current + sep + piece
        if current and len(candidate) > target:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_long_durations(
    chunks: list[str], total_secs: float, cfg: YTConfig, char_mode: bool
) -> list[str]:
    """Split chunks whose share of the window exceeds max_duration — but only
    at sentence boundaries, never mid-sentence (reading quality first)."""
    sum_len = sum(len(c) for c in chunks)
    if sum_len == 0 or total_secs <= cfg.max_duration * _DURATION_TOLERANCE:
        return chunks

    sep = "" if char_mode else " "
    char_budget = cfg.max_duration / total_secs * sum_len
    result: list[str] = []
    for chunk in chunks:
        if total_secs * len(chunk) / sum_len <= cfg.max_duration * _DURATION_TOLERANCE:
            result.append(chunk)
            continue
        sentences = _units(chunk, _SENT_SPLIT)
        if len(sentences) < 2:
            result.append(chunk)  # unsplittable without hurting readability
            continue
        current = ""
        for sentence in sentences:
            candidate = sentence if not current else current + sep + sentence
            if current and len(candidate) > char_budget:
                result.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            result.append(current)
    return result


def _fits(chunk: str, cfg: YTConfig, max_chars: int, char_mode: bool) -> bool:
    """True when the chunk wraps legally into <= max_lines lines of <= max_chars."""
    lines = wrap_lines(chunk, max_chars, char_mode).split("\n")
    return len(lines) <= cfg.max_lines and all(len(line) <= max_chars for line in lines)


def _refit(chunk: str, cfg: YTConfig, max_chars: int, char_mode: bool) -> list[str]:
    """Re-split a chunk that fits the char capacity but has no legal line break
    (e.g. an awkwardly placed long word)."""
    if _fits(chunk, cfg, max_chars, char_mode):
        return [chunk]
    return _hard_split(chunk, max_chars, char_mode)


REFUSAL_LINES = "needs_more_lines"
REFUSAL_WINDOW = "window_over_max_duration"
REFUSAL_GAP = "gap_over_max_merge_gap"


@dataclass(frozen=True)
class MergeCandidate:
    """One neighbour a too-short item could be merged into, plus the verdict.

    ``refusals`` lists every cap the merge would break, in check order; empty
    means the merge is legal.
    """

    left: int
    right: int
    text: str
    window_secs: float
    gap_secs: float
    refusals: tuple[str, ...]


@dataclass(frozen=True)
class MergeDecision:
    """Both neighbours of a too-short item, shorter one first.

    Reporting the refused candidates and not just the winner is what lets a
    caller (and a test) say WHY an entry stayed short instead of only observing
    that nothing moved.
    """

    candidates: tuple[MergeCandidate, ...]

    @property
    def accepted(self) -> MergeCandidate | None:
        """The first candidate that breaks no cap, else None."""
        return next((c for c in self.candidates if not c.refusals), None)


def _merge_neighbour(
    texts: Sequence[str],
    index: int,
    window_secs: Callable[[int, int], float],
    gap_secs: Callable[[int, int], float],
    cfg: YTConfig,
    *,
    max_chars: int | None = None,
    char_mode: bool | None = None,
) -> MergeDecision:
    """Judge both neighbours a too-short item at *index* could merge into.

    The one place the three merge passes (proportional chunk shares, real
    anchored windows, whole entries) agree on what a legal merge is: the shorter
    neighbour is tried first because it is likeliest to leave the joined text
    inside ``cfg.max_lines``, and the merge is refused when the text needs
    another line, when the joined window would exceed ``cfg.max_duration``, or
    when it would swallow more than ``cfg.max_merge_gap`` of silence -- the
    merged entry is displayed from the earlier window's start, so a long pause
    between the two would put text on screen before it is spoken.

    Args:
        window_secs: ``(left, right)`` -> seconds the merged window would span.
        gap_secs: ``(left, right)`` -> silence between the two items; 0 for
            chunks inside one entry, which tile their parent's window.
        max_chars: line width to test against; derived from the joined text when None.
        char_mode: no-space-script flag; derived from the left-hand text when None.
    """
    neighbours = [i for i in (index - 1, index + 1) if 0 <= i < len(texts)]
    neighbours.sort(key=lambda i: len(_normalize_ws(texts[i])))
    candidates: list[MergeCandidate] = []
    for neighbour in neighbours:
        left, right = min(index, neighbour), max(index, neighbour)
        breaks = scripts.breaks_at_chars(texts[left]) if char_mode is None else char_mode
        joined = _join(
            [_join_content_lines(texts[side], breaks) for side in (left, right)],
            "" if breaks else " ",
        )
        width = scripts.max_chars_for(joined, cfg) if max_chars is None else max_chars
        window = window_secs(left, right)
        gap = gap_secs(left, right)
        refusals: list[str] = []
        if not _fits(joined, cfg, width, breaks):
            refusals.append(REFUSAL_LINES)
        if window > cfg.max_duration:
            refusals.append(REFUSAL_WINDOW)
        if gap > cfg.max_merge_gap:
            refusals.append(REFUSAL_GAP)
        candidates.append(
            MergeCandidate(left, right, joined, window, gap, tuple(refusals))
        )
    return MergeDecision(tuple(candidates))


def entry_merge_decision(
    subs: Sequence[srt.Subtitle], index: int, cfg: YTConfig
) -> MergeDecision:
    """Which neighbour entry *index* would merge into, and why each is refused.

    The decision :func:`merge_short_entries` itself acts on, exposed so a caller
    can report the residual short entries of a lecture with their reason.
    """

    def window(left: int, right: int) -> float:
        return (subs[right].end - subs[left].start).total_seconds()

    def gap(left: int, right: int) -> float:
        return (subs[right].start - subs[left].end).total_seconds()

    return _merge_neighbour([entry.content for entry in subs], index, window, gap, cfg)


def _merge_short_windows(
    chunks: Sequence[str],
    starts: Sequence[float],
    cfg: YTConfig,
    max_chars: int,
    char_mode: bool,
) -> tuple[list[str], list[int]]:
    """Merge chunks whose window is shorter than ``cfg.min_duration``.

    Args:
        starts: ``len(chunks) + 1`` ascending boundary offsets in seconds.

    Returns:
        The merged texts plus the boundary indices they still use
        (``len(kept) == len(merged) + 1``), so the caller can map the result back
        onto its own timeline.
    """
    merged = list(chunks)
    kept = list(range(len(starts)))

    def window(left: int, right: int) -> float:
        return starts[kept[right + 1]] - starts[kept[left]]

    def gap(left: int, right: int) -> float:
        # chunks tile their parent's window: the end of *left* IS the start of
        # *right*, so this is 0 and the silence cap can never fire here.
        return starts[kept[right]] - starts[kept[left + 1]]

    index = 0
    while index < len(merged) and len(merged) > 1:
        if starts[kept[index + 1]] - starts[kept[index]] >= cfg.min_duration:
            index += 1
            continue
        accepted = _merge_neighbour(
            merged, index, window, gap, cfg, max_chars=max_chars, char_mode=char_mode
        ).accepted
        if accepted is None:
            index += 1
            continue
        merged[accepted.left : accepted.right + 1] = [accepted.text]
        del kept[accepted.left + 1 : accepted.right + 1]
        index = max(0, accepted.left - 1)
    return merged, kept


def _merge_short_durations(
    chunks: list[str], total_secs: float, cfg: YTConfig, max_chars: int, char_mode: bool
) -> list[str]:
    """Merge chunks whose proportional share of the window falls under min_duration.

    A pre-split estimate: with anchors the boundaries that actually ship are
    different, so ``split_entry`` runs the same pass again over the real windows.
    """
    sum_len = sum(len(chunk) for chunk in chunks)
    if sum_len == 0:
        return list(chunks)

    starts = [0.0]
    for chunk in chunks:
        starts.append(starts[-1] + total_secs * len(chunk) / sum_len)
    merged, _kept = _merge_short_windows(chunks, starts, cfg, max_chars, char_mode)
    return merged


def wrap_lines(text: str, max_chars: int, char_mode: bool) -> str:
    """Wrap text into at most two balanced lines of <= max_chars each."""
    text = _normalize_ws(text)
    if len(text) <= max_chars:
        return text

    if char_mode:
        lo = max(1, len(text) - max_chars)
        hi = min(max_chars, len(text) - 1)
        balanced = min(max((len(text) + 1) // 2, lo), hi)
        pos = _nearest_legal_break(text, balanced, lo, hi)
        return f"{text[:pos]}\n{text[pos:]}"

    words = text.split(" ")
    best: tuple[int, str, str] | None = None
    for i in range(1, len(words)):
        left = " ".join(words[:i])
        right = " ".join(words[i:])
        if len(left) <= max_chars and len(right) <= max_chars:
            score = max(len(left), len(right))
            if best is None or score < best[0]:
                best = (score, left, right)
    if best:
        return f"{best[1]}\n{best[2]}"

    # No legal two-line break (a very long word): greedy first line, accept overflow.
    first: list[str] = []
    for word in words:
        candidate = " ".join([*first, word])
        if first and len(candidate) > max_chars:
            break
        first.append(word)
    return " ".join(first) + "\n" + " ".join(words[len(first) :])


def _greedy_wrap(text: str, width: int, char_mode: bool) -> list[str]:
    """Fill lines up to *width*, breaking only where a break is legal."""
    if char_mode:
        lines: list[str] = []
        rest = text
        while len(rest) > width:
            pos = _nearest_legal_break(rest, width, 1, min(width, len(rest) - 1))
            lines.append(rest[:pos])
            rest = rest[pos:]
        lines.append(rest)
        return lines

    lines = []
    current: list[str] = []
    for word in text.split(" "):
        candidate = " ".join([*current, word])
        if current and len(candidate) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines or [text]


def wrap_lines_balanced(
    text: str, max_chars: int, char_mode: bool, max_lines: int
) -> str:
    """Wrap into at most *max_lines* lines, keeping the widest line as narrow
    as possible.

    The rewrap fallback for text that will not fit two lines. Line count falls
    monotonically as the width grows, so the narrowest width that still fits the
    budget is the balanced one; binary search finds it.

    Text that will not fit even the extended budget keeps the old two-line
    layout with an over-wide line rather than growing without limit: a subtitle
    covering half the frame is not a rescue. ``rewrap_overflow`` reports it.
    """
    text = _normalize_ws(text)
    lo, hi = 1, max(1, max_chars)
    best: list[str] | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        lines = _greedy_wrap(text, mid, char_mode)
        if len(lines) <= max_lines and all(len(line) <= max_chars for line in lines):
            best = lines
            hi = mid - 1
        else:
            lo = mid + 1
    if best is None:
        return wrap_lines(text, max_chars, char_mode)
    return "\n".join(best)


def _conforms(
    content: str,
    cfg: YTConfig,
    max_chars: int,
    char_mode: bool,
    max_lines: int | None = None,
) -> bool:
    lines = content.split("\n")
    budget = cfg.max_lines if max_lines is None else max_lines
    if len(lines) > budget or any(len(line) > max_chars for line in lines):
        return False
    if char_mode:
        # a line break inside a digit run (even with wrap-damage spaces around
        # it, e.g. "第10 \n0号") is never acceptable
        if re.search(r"\d *\n *\d", content):
            return False
        # the existing break position must also be a legal one (no split digit
        # runs, no orphaned combining marks / leading vowels)
        for left, right in zip(lines, lines[1:], strict=False):
            if left and right and not _legal_break(left + right, len(left)):
                return False
    return True


def _join_content_lines(content: str, char_mode: bool) -> str:
    """Collapse an entry's lines into one string for re-chunking.

    For no-space scripts the wrapper breaks between characters without eating
    anything, so the newline is removed directly — joining with a space would
    inject spurious spaces into CJK/Thai text. Any space inside a digit run is
    healed (it can only be wrap damage, e.g. "第10 0号").
    """
    if not char_mode:
        return _normalize_ws(content)

    joined = content.replace("\n", "")
    joined = re.sub(r"(?<=\d) +(?=\d)", "", joined)
    return _normalize_ws(joined)


def _chunk_start_offsets(text: str, chunks: list[str]) -> list[int]:
    offsets: list[int] = []
    search_from = 0
    for chunk in chunks:
        found = text.find(chunk, search_from)
        if found < 0:
            found = text.find(chunk)
        if found < 0:
            offsets.append(search_from)
            continue
        offsets.append(found)
        search_from = found + len(chunk)
    return offsets


def _proportional_boundaries(
    sub: srt.Subtitle, total_secs: float, weights: list[int]
) -> list[timedelta]:
    sum_weights = sum(weights)
    total_ms = round(total_secs * 1000)
    boundaries = [sub.start]
    accumulated = 0
    for weight in weights[:-1]:
        accumulated += weight
        offset_ms = round(total_ms * accumulated / sum_weights)
        boundaries.append(sub.start + timedelta(milliseconds=offset_ms))
    boundaries.append(sub.end)
    return boundaries


def _anchored_boundaries(
    sub: srt.Subtitle,
    total_secs: float,
    chunks: list[str],
    offsets: list[int],
    anchors: Sequence[TimeAnchor] | None,
) -> list[timedelta]:
    boundaries = _proportional_boundaries(sub, total_secs, [len(chunk) for chunk in chunks])
    if not anchors:
        return boundaries

    by_offset = {offset: start for offset, start in anchors}
    for i, offset in enumerate(offsets[1:], start=1):
        anchored = by_offset.get(offset)
        if anchored is None:
            continue
        previous_boundary = boundaries[i - 1]
        next_boundary = boundaries[i + 1] if i + 1 < len(boundaries) else sub.end
        if previous_boundary < anchored < next_boundary:
            boundaries[i] = anchored
    return boundaries


def split_entry(
    sub: srt.Subtitle, cfg: YTConfig, anchors: Sequence[TimeAnchor] | None = None
) -> list[srt.Subtitle]:
    """Split one translated entry into >=1 YT-friendly entries within its window."""
    if not sub.content.strip():
        return [sub]

    max_chars = scripts.max_chars_for(sub.content, cfg)
    char_mode = scripts.breaks_at_chars(sub.content)
    text = _join_content_lines(sub.content, char_mode)
    total_secs = srtio.td_to_seconds(sub.end - sub.start)

    duration_ok = total_secs <= cfg.max_duration * _DURATION_TOLERANCE
    if duration_ok and _conforms(sub.content, cfg, max_chars, char_mode):
        return [sub]

    capacity = max_chars * cfg.max_lines
    chunks = _chunk_text(text, capacity, char_mode)
    chunks = _split_long_durations(chunks, total_secs, cfg, char_mode)
    chunks = [piece for chunk in chunks for piece in _refit(chunk, cfg, max_chars, char_mode)]
    chunks = _merge_short_durations(chunks, total_secs, cfg, max_chars, char_mode)

    # Allocate time proportionally by default; source reflow can pass anchors
    # so split chunks keep the original start of the fragment they begin with.
    offsets = _chunk_start_offsets(text, chunks)
    boundaries = _anchored_boundaries(sub, total_secs, chunks, offsets, anchors)

    # Anchoring moves boundaries off the proportional timeline the pass above
    # guarded, so a chunk's real window can be far below min_duration ("Ali,",
    # 0.39 s). Merge once more, now against the windows that actually ship.
    starts = [srtio.td_to_seconds(boundary - sub.start) for boundary in boundaries]
    chunks, kept = _merge_short_windows(chunks, starts, cfg, max_chars, char_mode)
    boundaries = [boundaries[i] for i in kept]

    return [
        srt.Subtitle(
            index=sub.index + i,
            start=boundaries[i],
            end=boundaries[i + 1],
            content=wrap_lines(chunk, max_chars, char_mode),
        )
        for i, chunk in enumerate(chunks)
    ]


def _illegal_entry_boundary(prev: srt.Subtitle, cur: srt.Subtitle) -> bool:
    """Detect entries split mid-grapheme/mid-number by an earlier (buggy) pass:
    time-contiguous no-space-script entries whose junction is not a legal break."""
    if prev.end != cur.start:
        return False
    left = prev.content.rstrip()
    right = cur.content.lstrip()
    if not left or not right:
        return False
    if not (scripts.breaks_at_chars(left) and scripts.breaks_at_chars(right)):
        return False
    return not _legal_break(left[-1] + right[0], 1)


def reflow(subs: list[srt.Subtitle], cfg: YTConfig) -> list[srt.Subtitle]:
    """Apply split_entry to all entries, flatten, renumber sequentially."""
    return reflow_with_anchors(subs, cfg, None)


def reflow_with_anchors(
    subs: list[srt.Subtitle],
    cfg: YTConfig,
    anchors_by_index: Sequence[Sequence[TimeAnchor]] | None,
) -> list[srt.Subtitle]:
    """Apply reflow, optionally snapping split starts to source fragment starts."""
    healed: list[srt.Subtitle] = []
    healed_anchors: list[Sequence[TimeAnchor] | None] = []
    for sub in subs:
        if healed and _illegal_entry_boundary(healed[-1], sub):
            prev = healed.pop()
            healed_anchors.pop()
            healed.append(
                srt.Subtitle(
                    index=prev.index,
                    start=prev.start,
                    end=sub.end,
                    content=f"{prev.content}\n{sub.content}",
                )
            )
            healed_anchors.append(None)
        else:
            healed.append(sub)
            anchors = None
            if anchors_by_index is not None and len(anchors_by_index) >= len(healed):
                anchors = anchors_by_index[len(healed) - 1]
            healed_anchors.append(anchors)

    flattened: list[srt.Subtitle] = []
    for sub, anchors in zip(healed, healed_anchors, strict=False):
        flattened.extend(split_entry(sub, cfg, anchors))
    return srtio.reindex(flattened)


def reflow_text(content: str, cfg: YTConfig) -> str:
    """String-level entry point: parse -> reflow -> compose."""
    return srtio.compose(reflow(srtio.parse(content), cfg))


def reflow_text_with_anchors(
    content: str, cfg: YTConfig, anchors_by_index: Sequence[Sequence[TimeAnchor]]
) -> str:
    """String-level reflow entry point that preserves source fragment starts."""
    return srtio.compose(reflow_with_anchors(srtio.parse(content), cfg, anchors_by_index))


def _rewrap_content(content: str, cfg: YTConfig) -> str:
    """The line layout ``rewrap`` gives one entry's text.

    Two balanced lines whenever they fit -- so an entry that already conformed
    is byte-identical to what the previous wrap produced and the archive does
    not churn. Only when they do not fit does the text spread over up to
    ``cfg.rewrap_max_lines`` lines: a translation is never split, so the extra
    line is its only rescue.
    """
    max_chars = scripts.max_chars_for(content, cfg)
    char_mode = scripts.breaks_at_chars(content)
    text = _join_content_lines(content, char_mode)
    two = wrap_lines(text, max_chars, char_mode)
    if _conforms(two, cfg, max_chars, char_mode):
        return two
    return wrap_lines_balanced(text, max_chars, char_mode, cfg.rewrap_max_lines)


def _rewrap_entry(sub: srt.Subtitle, cfg: YTConfig) -> srt.Subtitle:
    """One entry re-wrapped in place; unchanged when it already conforms."""
    if not sub.content.strip():
        return sub
    max_chars = scripts.max_chars_for(sub.content, cfg)
    char_mode = scripts.breaks_at_chars(sub.content)
    if _conforms(sub.content, cfg, max_chars, char_mode, cfg.rewrap_max_lines):
        return sub
    return srt.Subtitle(
        index=sub.index,
        start=sub.start,
        end=sub.end,
        content=_rewrap_content(sub.content, cfg),
    )


def rewrap(subs: Sequence[srt.Subtitle], cfg: YTConfig) -> list[srt.Subtitle]:
    """Re-wrap every entry's lines without touching the structure.

    The translated-file counterpart of :func:`reflow`: segmentation is computed
    once, on the Croatian source, so a translation keeps its entry count,
    indices and timestamps and the two files can be diffed side by side. Only
    the line breaks inside an entry move, onto the target script's width.

    Text that does not fit ``cfg.max_lines`` lines at that width stays in its
    entry — wrapped as well as those lines allow — instead of being split into
    a new one: German and Tamil run 25–45 % longer than Croatian, and that
    overflow is the accepted price of the 1:1 structure. ``rewrap_overflow``
    reports which entries it hit.

    Pure and idempotent.
    """
    return [_rewrap_entry(sub, cfg) for sub in subs]


def rewrap_overflow(subs: Sequence[srt.Subtitle], cfg: YTConfig) -> list[int]:
    """Indices of entries whose text cannot fit ``cfg.max_lines`` lines.

    Text-based, so it reports the same entries before and after :func:`rewrap`.
    """
    overflowing: list[int] = []
    for sub in subs:
        if not sub.content.strip():
            continue
        max_chars = scripts.max_chars_for(sub.content, cfg)
        char_mode = scripts.breaks_at_chars(sub.content)
        if not _conforms(
            _rewrap_content(sub.content, cfg), cfg, max_chars, char_mode,
            cfg.rewrap_max_lines,
        ):
            overflowing.append(sub.index)
    return overflowing


def rewrap_text(content: str, cfg: YTConfig) -> str:
    """String-level entry point: parse -> rewrap -> compose."""
    return srtio.compose(rewrap(srtio.parse(content), cfg))


def merge_short_entries(
    subs: Sequence[srt.Subtitle], cfg: YTConfig
) -> list[srt.Subtitle]:
    """Merge entries shorter than ``cfg.min_duration`` into a neighbour.

    Last step of the source preparation, reached by entries that
    ``cleaning.repair_windows`` could not extend because no silence follows them.
    Merging is lossless where stretching is not: the text is unchanged and the
    merged window is exactly the two original windows plus the gap between them.
    Nothing drifts out of sync only while that gap is small, so a merge across
    more than ``cfg.max_merge_gap`` of silence is refused: the joined entry is
    shown from the earlier start, and swallowing a long pause would display the
    text seconds before it is spoken. An entry whose merge would break that cap,
    the two-line limit or ``cfg.max_duration`` is left in place rather than
    mangled; ``cfg.hard_min_duration`` stays the reporting threshold for
    validation. :func:`entry_merge_decision` reports which cap refused it.

    Pure and idempotent: a second pass finds nothing mergeable left.
    """
    merged = list(subs)

    index = 0
    while index < len(merged):
        current = merged[index]
        if (current.end - current.start).total_seconds() >= cfg.min_duration:
            index += 1
            continue

        accepted = entry_merge_decision(merged, index, cfg).accepted
        if accepted is None:
            index += 1
            continue

        left, right = accepted.left, accepted.right
        char_mode = scripts.breaks_at_chars(merged[left].content)
        max_chars = scripts.max_chars_for(accepted.text, cfg)
        merged[left : right + 1] = [
            srt.Subtitle(
                index=merged[left].index,
                start=merged[left].start,
                end=merged[right].end,
                content=wrap_lines(accepted.text, max_chars, char_mode),
            )
        ]
        index = max(0, left - 1)
    return srtio.reindex(merged)
