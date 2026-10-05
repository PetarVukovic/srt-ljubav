"""Pre-translation cleaning: timestamp fixing + tiny-fragment merging.

Merges short neighboring fragments into sentence-ish units so the translator
sees coherent text. Operates on ``srt.Subtitle`` lists end-to-end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import timedelta

import srt

from srtljubav import srtio
from srtljubav.config import CleaningConfig


@dataclass
class CleanReport:
    kept: int = 0
    deleted: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class TextAnchor:
    offset: int
    start: timedelta


@dataclass
class _AnchoredSubtitle:
    sub: srt.Subtitle
    anchors: list[TextAnchor]


def _text_len(text: str) -> int:
    return len(re.sub(r"\s+", " ", text).strip())


def _join_text(left: str, right: str) -> str:
    return " ".join(part.strip() for part in (left, right) if part.strip())


def fix_timestamps(subs: list[srt.Subtitle]) -> list[srt.Subtitle]:
    """Clamp each entry into its neighbors' window so entries never overlap."""
    fixed: list[srt.Subtitle] = []
    for idx, sub in enumerate(subs):
        start, end = sub.start, sub.end
        if fixed:
            start = max(start, fixed[-1].end)
        if idx < len(subs) - 1:
            end = min(end, subs[idx + 1].start)
        if end < start:
            end = start
        fixed.append(srt.Subtitle(index=idx + 1, start=start, end=end, content=sub.content))
    return fixed


def _shift_anchors(anchors: list[TextAnchor], offset: int) -> list[TextAnchor]:
    return [TextAnchor(anchor.offset + offset, anchor.start) for anchor in anchors]


def _right_join_offset(left: str, right: str) -> int:
    left = re.sub(r"\s+", " ", left).strip()
    right = re.sub(r"\s+", " ", right).strip()
    if not left:
        return 0
    if not right:
        return len(left)
    return len(left) + 1


def _merge_fragment_blocks(
    blocks: list[_AnchoredSubtitle],
    *,
    max_len: int,
    min_duration: float,
    max_dist_forward: float,
    max_dist_backward: float,
    max_duration: float | None = None,
) -> list[_AnchoredSubtitle]:
    merged: list[_AnchoredSubtitle] = []

    for block in blocks:
        sub = block.sub
        if not merged or not merged[-1].sub.content.strip() or not sub.content.strip():
            merged.append(block)
            continue

        previous_block = merged[-1]
        previous = previous_block.sub
        current_duration = srtio.td_to_seconds(sub.end - sub.start)
        merged_duration = srtio.td_to_seconds(sub.end - previous.start)
        gap = srtio.td_to_seconds(sub.start - previous.end)
        merged_content = _join_text(previous.content, sub.content)
        merged_len = _text_len(merged_content)

        should_merge = (
            merged_len <= max_len
            and (max_duration is None or merged_duration <= max_duration)
            and (
                current_duration < min_duration
                or 0 <= gap <= max_dist_forward
                or -max_dist_backward <= gap < 0
            )
        )

        if should_merge:
            right_offset = _right_join_offset(previous.content, sub.content)
            merged[-1] = _AnchoredSubtitle(
                sub=srt.Subtitle(
                    index=previous.index,
                    start=previous.start,
                    end=sub.end,
                    content=merged_content,
                ),
                anchors=[
                    *previous_block.anchors,
                    *_shift_anchors(block.anchors, right_offset),
                ],
            )
        else:
            merged.append(block)

    return [
        _AnchoredSubtitle(
            sub=srt.Subtitle(
                index=i + 1,
                start=block.sub.start,
                end=block.sub.end,
                content=block.sub.content,
            ),
            anchors=block.anchors,
        )
        for i, block in enumerate(merged)
    ]


def merge_fragments(
    subs: list[srt.Subtitle],
    *,
    max_len: int,
    min_duration: float,
    max_dist_forward: float,
    max_dist_backward: float,
    max_duration: float | None = None,
) -> list[srt.Subtitle]:
    """Merge a fragment into its predecessor when it is too short to stand alone.

    A merge happens when the combined text fits ``max_len``, the combined
    duration stays under ``max_duration``, and the fragment either lasts less
    than ``min_duration`` or sits within the allowed gap/overlap of its
    predecessor.
    """
    merged: list[srt.Subtitle] = []

    for sub in subs:
        if not merged or not merged[-1].content.strip() or not sub.content.strip():
            merged.append(sub)
            continue

        previous = merged[-1]
        current_duration = srtio.td_to_seconds(sub.end - sub.start)
        merged_duration = srtio.td_to_seconds(sub.end - previous.start)
        gap = srtio.td_to_seconds(sub.start - previous.end)
        merged_len = _text_len(_join_text(previous.content, sub.content))

        should_merge = (
            merged_len <= max_len
            and (max_duration is None or merged_duration <= max_duration)
            and (
                current_duration < min_duration
                or 0 <= gap <= max_dist_forward
                or -max_dist_backward <= gap < 0
            )
        )

        if should_merge:
            merged[-1] = srt.Subtitle(
                index=previous.index,
                start=previous.start,
                end=sub.end,
                content=_join_text(previous.content, sub.content),
            )
        else:
            merged.append(sub)

    return srtio.reindex(merged)


def validate_and_filter(subs: list[srt.Subtitle]) -> tuple[list[srt.Subtitle], list[dict]]:
    """Drop invalid entries (empty text, start >= end), reporting each deletion."""
    valid: list[srt.Subtitle] = []
    deleted: list[dict] = []

    for sub in subs:
        error = None
        if sub.start >= sub.end:
            error = "Start >= End"
        if not sub.content.strip():
            error = "Empty text"

        if error:
            deleted.append(
                {
                    "Segment": sub.index,
                    "Time": f"{srt.timedelta_to_srt_timestamp(sub.start)}"
                    f" --> {srt.timedelta_to_srt_timestamp(sub.end)}",
                    "Text": sub.content,
                    "Error": error,
                }
            )
        else:
            valid.append(sub)

    return srtio.reindex(valid), deleted


def _validate_and_filter_blocks(
    blocks: list[_AnchoredSubtitle],
) -> tuple[list[_AnchoredSubtitle], list[dict]]:
    clean, deleted = validate_and_filter([block.sub for block in blocks])
    valid_blocks = [
        block
        for block in blocks
        if block.sub.start < block.sub.end and block.sub.content.strip()
    ]
    return [
        _AnchoredSubtitle(sub=clean_sub, anchors=block.anchors)
        for clean_sub, block in zip(clean, valid_blocks, strict=False)
    ], deleted


def clean_source(content: str, cfg: CleaningConfig) -> tuple[str, CleanReport]:
    """Full cleaning pipeline: parse → fix timestamps → merge passes → validate."""
    subs = fix_timestamps(srtio.parse(content))
    for max_len in cfg.max_len_steps:
        subs = merge_fragments(
            subs,
            max_len=max_len,
            min_duration=cfg.min_duration,
            max_dist_forward=cfg.max_dist_forward,
            max_dist_backward=cfg.max_dist_backward,
            max_duration=cfg.max_merged_duration,
        )

    clean, deleted = validate_and_filter(subs)
    return srtio.compose(clean), CleanReport(kept=len(clean), deleted=deleted)


def clean_source_with_anchors(
    content: str, cfg: CleaningConfig
) -> tuple[str, list[list[tuple[int, timedelta]]], CleanReport]:
    """Clean source text while preserving original fragment start offsets."""
    subs = fix_timestamps(srtio.parse(content))
    blocks = [
        _AnchoredSubtitle(sub=sub, anchors=[TextAnchor(offset=0, start=sub.start)])
        for sub in subs
    ]
    for max_len in cfg.max_len_steps:
        blocks = _merge_fragment_blocks(
            blocks,
            max_len=max_len,
            min_duration=cfg.min_duration,
            max_dist_forward=cfg.max_dist_forward,
            max_dist_backward=cfg.max_dist_backward,
            max_duration=cfg.max_merged_duration,
        )

    clean_blocks, deleted = _validate_and_filter_blocks(blocks)
    return (
        srtio.compose([block.sub for block in clean_blocks]),
        [[(anchor.offset, anchor.start) for anchor in block.anchors] for block in clean_blocks],
        CleanReport(kept=len(clean_blocks), deleted=deleted),
    )


def repair_windows(subs: list[srt.Subtitle], cfg: CleaningConfig) -> list[srt.Subtitle]:
    """Give under-timed entries the screen time their text needs.

    Only ends move, and only forward into the silence that already follows the
    entry. Starts are never touched: an entry whose start is correct is in sync
    with the speech, and pulling it backwards would desynchronise it to buy time
    that 83 % of these entries do not have behind them anyway. Text and indices
    are left untouched, so cleaning anchors stay valid.

    Pure and idempotent: a second pass finds every window already long enough.
    """
    repaired = list(subs)
    for index, sub in enumerate(repaired):
        text = _text_len(sub.content)
        duration = (sub.end - sub.start).total_seconds()
        if not text or duration <= 0:
            continue

        chars_per_second = text / duration
        if chars_per_second > cfg.impossible_cps:
            needed = text / cfg.repair_target_cps
        elif duration < cfg.min_entry_duration:
            needed = cfg.min_entry_duration
        else:
            continue
        if duration >= needed:
            continue

        wanted_end = sub.start + timedelta(seconds=needed)
        if index + 1 < len(repaired):
            wanted_end = min(wanted_end, repaired[index + 1].start)
        if wanted_end <= sub.end:
            continue

        repaired[index] = srt.Subtitle(
            index=sub.index, start=sub.start, end=wanted_end, content=sub.content
        )
    return repaired
