"""Source discovery and translation-job building.

Sources are ``input_files/<topic>/*.srt``; anything whose stem ends in a
``_{Language}`` suffix is a translated output and MUST be excluded, otherwise
outputs get re-translated as new sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from srtljubav import srtio
from srtljubav.languages import (
    TARGET_LANGUAGES,
    language_dir_name,
    translation_suffixes,
)


@dataclass(frozen=True)
class TranslationJob:
    source: Path
    target_language: str
    output_file: Path


def discover_source_srts(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    # _{Language} suffixi označavaju da je file već prijevod (output), pa se
    # NE smije tretirati kao source — inače bi se outputi reprevodili.
    suffixes = translation_suffixes()
    sources: list[Path] = []
    for path in sorted(input_dir.glob("*/*.srt")):
        if not path.is_file():  # preskoči ako nije file
            continue
        if any(path.stem.endswith(suffix) for suffix in suffixes):
            continue
        sources.append(path)
    return sources


def discover_translated_srts(
    input_dir: Path, target_languages: list[str] | None = None
) -> list[Path]:
    """All existing translated outputs, optionally limited to a language subset."""
    languages = target_languages or TARGET_LANGUAGES
    suffixes = {f"_{language_dir_name(lang)}" for lang in languages}
    translated: list[Path] = []
    for path in sorted(input_dir.glob("*/*.srt")):
        if path.is_file() and any(path.stem.endswith(suffix) for suffix in suffixes):
            translated.append(path)
    return translated


def is_complete_output(path: Path) -> bool:
    """True when an existing translation is usable enough to skip.

    Failed or interrupted translator runs can leave a zero-byte or malformed
    file. Treat those as missing so the next run regenerates them without
    needing --overwrite.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        return bool(srtio.parse(srtio.read_srt_text(path)))
    except Exception:
        return False


def build_jobs(
    input_dir: Path,
    overwrite: bool,
    output_root: Path | None = None,
    target_languages: list[str] | None = None,
    max_jobs: int | None = None,
) -> list[TranslationJob]:
    jobs: list[TranslationJob] = []
    languages = target_languages or TARGET_LANGUAGES
    for source in discover_source_srts(input_dir):
        for language in languages:
            translated_name = f"{source.stem}_{language_dir_name(language)}.srt"
            if output_root:
                output_file = (
                    output_root / source.parent.relative_to(input_dir) / translated_name
                )
            else:
                output_file = source.parent / translated_name
            if not overwrite and is_complete_output(output_file):
                continue
            jobs.append(
                TranslationJob(
                    source=source,
                    target_language=language,
                    output_file=output_file,
                )
            )
            if max_jobs is not None and len(jobs) >= max_jobs:
                return jobs
    return jobs
