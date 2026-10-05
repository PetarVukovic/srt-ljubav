"""Measure Croatian segmentations side by side: source, shipped, legacy, anchor.

Reproduces the readability measures from Andrija Dujaković's 2026-09-30 report
so his numbers can be checked independently, plus two timing-truth measures:

* ``early words`` -- a word shown across a source pause of ``PAUSE`` seconds or
  more before its own fragment is spoken (text on screen before it is said);
* ``late words`` -- a word whose entry starts after its fragment has ended.

Both map every output word back to its source fragment by position, which is
only possible when the segmentation kept the text word for word; a variant
whose words differ from the source reports ``-`` instead.

    uv run python scripts/compare_segmenters.py [input_dir ...] [--per-lecture]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import srt  # noqa: E402

from srtljubav import segment, srtio  # noqa: E402
from srtljubav.config import TranslationConfig  # noqa: E402
from srtljubav.discovery import discover_source_srts  # noqa: E402
from srtljubav.pipeline import prepare_source_text  # noqa: E402

PAUSE = 1.0
VARIANTS = ("source", "shipped", "legacy", "anchor")
COLUMNS = (
    ("entries", "entries"),
    ("cps>36", "above 36 cps"),
    ("cps>25", "above 25 cps"),
    ("cps>20", "above 20 cps"),
    ("cps>17.05", "above 17.05 cps"),
    ("<1s", "shorter than 1 s"),
    (">7s", "longer than 7 s"),
    ("gap<0.08", "gap to next < 0.08 s"),
    ("conj-eol", "conjunction/preposition ending line 1"),
    ("bad-layout", ">2 lines or line >42 chars"),
    ("early", "words shown across a pause before spoken"),
    ("late", "words shown after spoken"),
    ("sent-end%", "entries ending a sentence"),
)


@dataclass
class Measure:
    values: dict[str, float] = field(default_factory=dict)
    words_ok: bool = True

    def add(self, other: Measure) -> None:
        for key, value in other.values.items():
            self.values[key] = self.values.get(key, 0) + value
        self.words_ok = self.words_ok and other.words_ok


def _text(sub: srt.Subtitle) -> str:
    return " ".join(sub.content.split())


def _word_fragments(source: list[srt.Subtitle]) -> list[int]:
    owners: list[int] = []
    for index, sub in enumerate(source):
        owners.extend([index] * len(_text(sub).split()))
    return owners


def _timing(subs: list[srt.Subtitle], source: list[srt.Subtitle]) -> tuple[int, int] | None:
    source_words = [w for sub in source for w in _text(sub).split()]
    words = [w for sub in subs for w in _text(sub).split()]
    if words != source_words:
        return None
    owners = _word_fragments(source)
    pauses = [
        (source[i].end.total_seconds(), source[i + 1].start.total_seconds())
        for i in range(len(source) - 1)
        if (source[i + 1].start - source[i].end).total_seconds() >= PAUSE
    ]
    early = late = 0
    position = 0
    for sub in subs:
        start = sub.start.total_seconds()
        for _word in _text(sub).split():
            frag = source[owners[position]]
            position += 1
            f_start, f_end = frag.start.total_seconds(), frag.end.total_seconds()
            if start > f_end + 0.05:
                late += 1
            elif any(start - 0.01 <= a and b <= f_start + 0.01 for a, b in pauses):
                early += 1
    return early, late


def measure(subs: list[srt.Subtitle], source: list[srt.Subtitle]) -> Measure:
    values: dict[str, float] = dict.fromkeys((key for key, _ in COLUMNS[:-1]), 0)
    values["entries"] = len(subs)
    sentence_ends = 0
    for i, sub in enumerate(subs):
        text = _text(sub)
        seconds = max((sub.end - sub.start).total_seconds(), 1e-3)
        cps = len(text) / seconds
        for limit in (36, 25, 20, 17.05):
            if cps > limit:
                values[f"cps>{limit:g}"] += 1
        values["<1s"] += seconds < 1.0 - 1e-6
        values[">7s"] += seconds > 7.0 + 1e-6
        if i + 1 < len(subs):
            values["gap<0.08"] += (subs[i + 1].start - sub.end).total_seconds() < 0.08 - 1e-6
        lines = sub.content.split("\n")
        values["bad-layout"] += len(lines) > 2 or any(len(line) > 42 for line in lines)
        if len(lines) > 1 and lines[0].split():
            last = lines[0].split()[-1]
            values["conj-eol"] += segment._ends_on_function_word(last)
        nxt = _text(subs[i + 1]) if i + 1 < len(subs) else None
        sentence_ends += segment.ends_sentence(text, nxt)
    timing = _timing(subs, source)
    result = Measure(values=values, words_ok=timing is not None)
    if timing is not None:
        values["early"], values["late"] = timing
    values["sent-end"] = sentence_ends
    return result


def _read(path: Path) -> list[srt.Subtitle]:
    return srtio.parse(srtio.read_srt_text(path))


def variants_for(source_path: Path) -> dict[str, list[srt.Subtitle]]:
    raw = _read(source_path)
    out: dict[str, list[srt.Subtitle]] = {"source": raw}
    shipped = source_path.with_name(f"{source_path.stem}_Croatian.srt")
    if shipped.exists():
        out["shipped"] = _read(shipped)
    base = TranslationConfig(usage_log=None)
    out["legacy"] = srtio.parse(prepare_source_text(source_path, replace(base, segmenter="legacy")))
    out["anchor"] = srtio.parse(prepare_source_text(source_path, base))
    return out


def _fmt(measure_: Measure, key: str) -> str:
    if key in ("early", "late") and not measure_.words_ok:
        return "-"
    if key == "sent-end%":
        entries = measure_.values.get("entries", 0) or 1
        return f"{100 * measure_.values.get('sent-end', 0) / entries:.1f}"
    return f"{int(measure_.values.get(key, 0))}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input_dirs", nargs="*", type=Path, default=[Path("input_files")])
    parser.add_argument("--per-lecture", action="store_true")
    args = parser.parse_args()

    totals = {name: Measure() for name in VARIANTS}
    present = dict.fromkeys(VARIANTS, 0)
    sources = [s for d in args.input_dirs for s in discover_source_srts(d)]
    for source_path in sources:
        variants = variants_for(source_path)
        raw = variants["source"]
        if args.per_lecture:
            print(f"\n{source_path.stem[:90]}")
        for name, subs in variants.items():
            m = measure(subs, raw)
            totals[name].add(m)
            present[name] += 1
            if args.per_lecture:
                cells = "  ".join(f"{label}={_fmt(m, label)}" for label, _ in COLUMNS)
                print(f"  {name:8} {cells}")

    print(f"\n{len(sources)} lectures (shipped present for {present['shipped']})\n")
    header = f"{'measure':42}" + "".join(f"{name:>10}" for name in VARIANTS)
    print(header)
    print("-" * len(header))
    for key, label in COLUMNS:
        row = f"{label:42}" + "".join(f"{_fmt(totals[name], key):>10}" for name in VARIANTS)
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
