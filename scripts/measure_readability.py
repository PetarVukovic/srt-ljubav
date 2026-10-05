"""Report subtitle readability metrics for one SRT file, without any API call.

By default the file is run through ``pipeline.prepare_source_text`` first, so the
numbers describe what the pipeline would actually upload; ``--as-is`` measures an
already produced file verbatim. Used to verify the Step 1 done-criteria
(0 entries under 1.0 s, 0 entries above 36 cps).

    uv run python scripts/measure_readability.py path/to/file.srt [--as-is]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from srtljubav import srtio  # noqa: E402
from srtljubav.config import TranslationConfig  # noqa: E402
from srtljubav.pipeline import prepare_source_text  # noqa: E402

CPS_WARN = 36.0
CPS_IMPOSSIBLE = 40.0


def _cps(sub) -> float:  # noqa: ANN001 - srt.Subtitle, imported lazily via srtio
    seconds = srtio.td_to_seconds(sub.end - sub.start)
    chars = len(" ".join(sub.content.split()))
    if seconds <= 0:
        return float("inf") if chars else 0.0
    return chars / seconds


def measure(content: str) -> dict[str, float]:
    subs = srtio.parse(content)
    durations = [srtio.td_to_seconds(s.end - s.start) for s in subs]
    rates = [_cps(s) for s in subs if s.content.strip()]
    return {
        "entries": len(subs),
        "under_1_0_s": sum(1 for d in durations if d < 1.0),
        "under_0_5_s": sum(1 for d in durations if d < 0.5),
        "above_36_cps": sum(1 for r in rates if r > CPS_WARN),
        "above_40_cps": sum(1 for r in rates if r > CPS_IMPOSSIBLE),
        "worst_cps": max(rates, default=0.0),
        "median_cps": statistics.median(rates) if rates else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("srt", type=Path, help="SRT file to measure")
    parser.add_argument(
        "--as-is",
        action="store_true",
        help="measure the file verbatim instead of running prepare_source_text",
    )
    args = parser.parse_args()

    if args.as_is:
        content = srtio.read_srt_text(args.srt)
    else:
        content = prepare_source_text(args.srt, TranslationConfig())

    stats = measure(content)
    print(args.srt.name)
    print(f"  entries       {stats['entries']:>8}")
    print(f"  < 1.0 s       {stats['under_1_0_s']:>8}")
    print(f"  < 0.5 s       {stats['under_0_5_s']:>8}")
    print(f"  > 36 cps      {stats['above_36_cps']:>8}")
    print(f"  > 40 cps      {stats['above_40_cps']:>8}")
    print(f"  worst cps     {stats['worst_cps']:>8.1f}")
    print(f"  median cps    {stats['median_cps']:>8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
