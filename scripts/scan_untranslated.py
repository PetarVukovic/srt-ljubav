"""Audit existing translations for untranslated Croatian leakage.

Read-only scan of ``input_files/<topic>/``: for every translated
``{stem}_{Language}.srt`` it flags entries that (a) still appear verbatim in
the Croatian source, or (b) are Latin-heavy although the target language uses
a non-Latin script. Run before uploading to YouTube:

    uv run python scripts/scan_untranslated.py [input_dir]

Exit code 1 when any leakage is found (the Croatian upload copy is exempt).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from srtljubav import srtio  # noqa: E402
from srtljubav.languages import (  # noqa: E402
    SELECTABLE_LANGUAGES,
    expected_script,
    language_dir_name,
    translation_suffixes,
)
from srtljubav.scripts import contains_script  # noqa: E402

_MIN_MATCH_CHARS = 20
# Shorter Latin-only entries in non-Latin languages are usually legitimate
# quotes/glosses ("Kršćanska sadašnjost", "Evanđelje ljubavi 6", "(care)").
_MIN_FOREIGN_CHARS = 25
_SUFFIX_TO_LANGUAGE = {language_dir_name(lang): lang for lang in SELECTABLE_LANGUAGES}


def _normalized(text: str) -> str:
    return " ".join(text.split()).lower()


def _is_reference_like(text: str) -> bool:
    """Pure scripture references often stay shared across related languages."""
    if not any(char.isdigit() for char in text):
        return False
    if sum(1 for token in text.split() if any(char.isalpha() for char in token)) > 4:
        return False
    return all(
        char.isalnum() or char.isspace() or char in ":,.;-()[]"
        for char in text
    )


def _language_of(path: Path) -> str | None:
    """Display-language for a translated file, None for sources/Croatian copy."""
    match = re.search(r"_([^_]+)$", path.stem)
    if not match:
        return None
    return _SUFFIX_TO_LANGUAGE.get(match.group(1))


def _merge_runs(indices: list[int]) -> list[str]:
    runs: list[list[int]] = []
    for i in indices:
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return [f"{a + 1}" if a == b else f"{a + 1}-{b + 1}" for a, b in runs]


def _actionable_indices(indices: list[int], min_run: int = 3) -> list[int]:
    """Only sustained leakage blocks upload; isolated fragments are warnings."""
    actionable: list[int] = []
    run: list[int] = []
    for index in indices:
        if run and index != run[-1] + 1:
            if len(run) >= min_run:
                actionable.extend(run)
            run = []
        run.append(index)
    if len(run) >= min_run:
        actionable.extend(run)
    return actionable


def scan_topic(topic: Path) -> int:
    suffixes = translation_suffixes()
    srts = sorted(topic.glob("*.srt"))
    sources = [p for p in srts if not any(p.stem.endswith(s) for s in suffixes)]
    if not sources:
        return 0

    source_subs = srtio.parse(srtio.read_srt_text(sources[0]))
    corpus = f" {_normalized(' '.join(sub.content for sub in source_subs))} "
    print(f"== {topic.name}  (source: {len(source_subs)} entries)")

    flagged_total = 0
    for path in srts:
        language = _language_of(path)
        if language is None:
            continue
        subs = srtio.parse(srtio.read_srt_text(path))
        expected = expected_script(language)
        flagged: list[int] = []
        for i, sub in enumerate(subs):
            norm = _normalized(sub.content)
            croatian_echo = (
                len(norm) >= _MIN_MATCH_CHARS
                and norm in corpus
                and not _is_reference_like(norm)
            )
            wrong_script = (
                expected is not None
                and len(norm) >= _MIN_FOREIGN_CHARS
                and any(c.isalpha() for c in norm)
                and not contains_script(sub.content, expected)
            )
            if croatian_echo or wrong_script:
                flagged.append(i)
        actionable = _actionable_indices(flagged)
        ignored = len(flagged) - len(actionable)
        flagged_total += len(actionable)
        status = "LEAK" if actionable else ("WARN" if ignored else "OK  ")
        runs = ", ".join(_merge_runs(actionable)[:10])
        more = "" if len(_merge_runs(actionable)) <= 10 else " ..."
        ignored_note = f" isolated={ignored}" if ignored else ""
        print(
            f"   [{status}] {language_dir_name(language):26s} entries={len(subs):5d} "
            f"flagged={len(actionable):4d}{ignored_note}  {runs}{more}"
        )
    return flagged_total


def main() -> int:
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("input_files")
    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 2

    total = 0
    for topic in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        total += scan_topic(topic)

    if total:
        print(f"\nFOUND {total} suspicious entries — re-translate the affected languages.")
        return 1
    print("\nAll translations clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
