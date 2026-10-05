"""Target languages and filename-suffix helpers.

Translations live flat next to the source as ``{stem}_{Language}.srt`` —
never in language subfolders (canonical project rule).
"""

from __future__ import annotations

from srtljubav.scripts import Script

TARGET_LANGUAGES: list[str] = [
    "English",
    "Mandarin Chinese",
    "Hindi",
    "Spanish",
    "Arabic",
    "Bengali",
    "Portuguese",
    "Russian",
    "Urdu",
    "Indonesian",
    "German",
    "Japanese",
    "Swahili",
    "Marathi",
    "Telugu",
    "Turkish",
    "French",
    "Vietnamese",
    "Korean",
    "Tamil",
    "Yue Chinese (Cantonese)",
    "Italian",
    "Thai",
    "Gujarati",
    "Javanese",
    "Polish",
    "Western Punjabi",
    "Ukrainian",
    "Persian (Farsi)",
    "Malayalam",
    "Slovenian",
    "Serbian (on Српски!)",
    "Macedonian (on Македонски!)",
]

# Opt-in targets: selectable with ``--languages`` and excluded from source
# discovery like any other output, but deliberately NOT part of the default run
# so the existing archive is never expanded behind the user's back.
EXTRA_LANGUAGES: list[str] = [
    "Swedish",
    "Hebrew",
]

SELECTABLE_LANGUAGES: list[str] = TARGET_LANGUAGES + EXTRA_LANGUAGES

SOURCE_LANGUAGE_SUFFIX = "_Croatian"

# Display imena nose ćirilične natuknice za prompt, ali filename suffix mora
# biti čista latinica. Ovaj override mapira display ime -> clean suffix.
LANGUAGE_DIR_OVERRIDES: dict[str, str] = {
    "Serbian (on Српски!)": "Serbian",
    "Macedonian (on Македонски!)": "Macedonian",
}


# Languages whose translations must arrive in a non-Latin script. Output whose
# dominant script differs (e.g. Latin-script Serbian, or a file left Croatian)
# fails content validation instead of reaching YouTube. Latin-script targets
# have no entry and are skipped.
EXPECTED_SCRIPTS: dict[str, Script] = {
    "Mandarin Chinese": Script.CJK,
    "Hindi": Script.DEVANAGARI,
    "Arabic": Script.ARABIC,
    "Hebrew": Script.HEBREW,
    "Bengali": Script.BENGALI,
    "Russian": Script.CYRILLIC,
    "Urdu": Script.ARABIC,
    "Japanese": Script.CJK,
    "Marathi": Script.DEVANAGARI,
    "Telugu": Script.TELUGU,
    "Korean": Script.HANGUL,
    "Tamil": Script.TAMIL,
    "Yue Chinese (Cantonese)": Script.CJK,
    "Thai": Script.THAI,
    "Gujarati": Script.GUJARATI,
    "Western Punjabi": Script.ARABIC,  # Shahmukhi
    "Ukrainian": Script.CYRILLIC,
    "Persian (Farsi)": Script.ARABIC,
    "Malayalam": Script.MALAYALAM,
    "Serbian (on Српски!)": Script.CYRILLIC,
    "Macedonian (on Македонски!)": Script.CYRILLIC,
}


def expected_script(language: str) -> Script | None:
    """Required output script for *language*, or None for Latin-script targets."""
    return EXPECTED_SCRIPTS.get(language)


def language_dir_name(language: str) -> str:
    """Filename-safe suffix for a target language."""
    return LANGUAGE_DIR_OVERRIDES.get(language, language)


def translation_suffixes() -> set[str]:
    """All ``_{Language}`` stem suffixes that mark a file as a translated output."""
    return {f"_{language_dir_name(lang)}" for lang in SELECTABLE_LANGUAGES} | {
        SOURCE_LANGUAGE_SUFFIX
    }


def parse_language_filter(value: str | None) -> list[str]:
    """Resolve a comma-separated, case-insensitive language subset.

    Returns all TARGET_LANGUAGES when *value* is falsy; raises ValueError on
    unknown names.
    """
    if not value:
        return TARGET_LANGUAGES

    requested = [item.strip() for item in value.split(",") if item.strip()]
    available_by_lower = {
        language.lower(): language for language in SELECTABLE_LANGUAGES
    }
    selected: list[str] = []
    unknown: list[str] = []

    for language in requested:
        match = available_by_lower.get(language.lower())
        if match:
            selected.append(match)
        else:
            unknown.append(language)

    if unknown:
        raise ValueError(f"Unknown target language(s): {', '.join(unknown)}")
    return selected
