"""Postgres persistence for the SRT archive — ``srt-ljubav sync-db``.

A separate command, never part of a translation run: the ``ProcessPoolExecutor``
workers stay free of any database dependency, and the sync can backfill the
whole existing archive offline. New tables live in the existing ``srt``
database; the unrelated ``seminar`` / ``prijevod_seminara`` tables are never
read, written or migrated.

Three tables (design section E), amended with a per-segment verdict:

    lecture         one source SRT (topic + stem)
    subtitle_file   one language of that lecture, addressed by file_hash
    subtitle_entry  one segment, with the Step 3 validation verdict on it

``subtitle_entry.status`` is one of ``ok`` / ``warn`` / ``fail`` /
``unchecked``, and ``flags`` lists the layers that spoke, e.g. ``cps_fail``,
``repetition_fail``, ``bidi_warn``. ``unchecked`` means the file is not 1:1
with its ``_Croatian.srt`` sibling, so no entry could be aligned and judged —
that is the pre-Step-2 archive, not a defect verdict.

The sync is idempotent through ``file_hash``: a file whose bytes are unchanged
is not rewritten, and a changed file has its entries deleted and re-inserted in
one transaction, so a shrunk file leaves no orphan segments behind.

Because every language of a lecture shares the Croatian file's segmentation
(Step 2), ``idx`` is the same segment in all 33 languages, which is what makes
these queries answer the reviewer's questions without a spreadsheet:

1. Which segments failed, in which language, in which lecture?

    SELECT l.topic, l.stem, f.language, e.idx, e.flags, e.content
    FROM subtitle_entry e
    JOIN subtitle_file f ON f.id = e.file_id
    JOIN lecture l       ON l.id = f.lecture_id
    WHERE e.status = 'fail'
    ORDER BY l.topic, l.stem, e.idx, f.language;

2. Which lectures are fully translated and clean (all 33 targets + Croatian,
   no failing segment)?

    SELECT l.topic, l.stem,
           count(*) FILTER (WHERE f.language <> 'Croatian') AS languages,
           count(*) FILTER (WHERE e.fails > 0)              AS files_with_failures
    FROM lecture l
    JOIN subtitle_file f ON f.lecture_id = l.id
    JOIN LATERAL (
        SELECT count(*) AS fails
        FROM subtitle_entry se
        WHERE se.file_id = f.id AND se.status = 'fail'
    ) e ON true
    GROUP BY l.topic, l.stem
    HAVING count(*) FILTER (WHERE f.language <> 'Croatian') = 33
       AND count(*) FILTER (WHERE e.fails > 0) = 0
    ORDER BY l.topic, l.stem;

3. Show me segment #100 of one lecture across every language.

    SELECT f.language, e.start_ms, e.end_ms, e.status, e.flags, e.content
    FROM subtitle_entry e
    JOIN subtitle_file f ON f.id = e.file_id
    JOIN lecture l       ON l.id = f.lecture_id
    WHERE l.stem = :stem AND e.idx = 100
    ORDER BY f.language;

4. Which files are not 1:1 with their Croatian sibling (Step 5's worklist)?

    SELECT l.topic, f.language, f.entry_count, hr.entry_count AS croatian
    FROM subtitle_file f
    JOIN lecture l        ON l.id = f.lecture_id
    JOIN subtitle_file hr ON hr.lecture_id = l.id AND hr.language = 'Croatian'
    WHERE f.entry_count <> hr.entry_count;

5. Which languages need the most review (worst offenders first)?

    SELECT f.language,
           count(*) FILTER (WHERE e.status = 'fail') AS failing,
           count(*) FILTER (WHERE e.status = 'warn') AS warning,
           count(*)                                   AS segments
    FROM subtitle_entry e
    JOIN subtitle_file f ON f.id = e.file_id
    GROUP BY f.language
    ORDER BY failing DESC, warning DESC;
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import srt

from srtljubav import srtio, validation
from srtljubav.discovery import discover_source_srts
from srtljubav.languages import (
    SOURCE_LANGUAGE_SUFFIX,
    TARGET_LANGUAGES,
    language_dir_name,
    translation_suffixes,
)

CROATIAN_LANGUAGE = SOURCE_LANGUAGE_SUFFIX.lstrip("_")

STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"
STATUS_UNCHECKED = "unchecked"

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS lecture (
        id           serial PRIMARY KEY,
        topic        text NOT NULL,
        stem         text NOT NULL,
        source_path  text NOT NULL,
        created_at   timestamptz NOT NULL DEFAULT now(),
        UNIQUE (topic, stem)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS subtitle_file (
        id           serial PRIMARY KEY,
        lecture_id   integer NOT NULL REFERENCES lecture(id) ON DELETE CASCADE,
        language     text NOT NULL,
        path         text NOT NULL,
        entry_count  integer NOT NULL,
        char_count   integer NOT NULL,
        file_hash    text NOT NULL,
        synced_at    timestamptz NOT NULL DEFAULT now(),
        UNIQUE (lecture_id, language)
    )
    """,
    # status/flags carry the Step 3 verdict per segment, which is what replaces
    # the reviewer's Excel controls.
    """
    CREATE TABLE IF NOT EXISTS subtitle_entry (
        id           bigserial PRIMARY KEY,
        file_id      integer NOT NULL REFERENCES subtitle_file(id) ON DELETE CASCADE,
        idx          integer NOT NULL,
        start_ms     integer NOT NULL,
        end_ms       integer NOT NULL,
        content      text NOT NULL,
        status       text NOT NULL DEFAULT 'unchecked',
        flags        text[] NOT NULL DEFAULT '{}',
        UNIQUE (file_id, idx)
    )
    """,
    "CREATE INDEX IF NOT EXISTS subtitle_entry_status_idx ON subtitle_entry (status)",
)

SQL_UPSERT_LECTURE = """
    INSERT INTO lecture (topic, stem, source_path)
    VALUES (%s, %s, %s)
    ON CONFLICT (topic, stem) DO UPDATE SET source_path = EXCLUDED.source_path
    RETURNING id
"""

SQL_SELECT_FILE = "SELECT id, file_hash FROM subtitle_file WHERE lecture_id = %s AND language = %s"

SQL_UPSERT_FILE = """
    INSERT INTO subtitle_file
        (lecture_id, language, path, entry_count, char_count, file_hash, synced_at)
    VALUES (%s, %s, %s, %s, %s, %s, now())
    ON CONFLICT (lecture_id, language) DO UPDATE SET
        path = EXCLUDED.path,
        entry_count = EXCLUDED.entry_count,
        char_count = EXCLUDED.char_count,
        file_hash = EXCLUDED.file_hash,
        synced_at = now()
    RETURNING id
"""

SQL_DELETE_ENTRIES = "DELETE FROM subtitle_entry WHERE file_id = %s"

SQL_INSERT_ENTRY = """
    INSERT INTO subtitle_entry (file_id, idx, start_ms, end_ms, content, status, flags)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
"""


@dataclass(frozen=True)
class EntryRecord:
    """One segment, with the verdict layers 2-4 reached about it."""

    idx: int
    start_ms: int
    end_ms: int
    content: str
    status: str
    flags: tuple[str, ...]


@dataclass(frozen=True)
class FileRecord:
    """One language of one lecture, addressed by the hash of its bytes."""

    language: str
    path: Path
    entry_count: int
    char_count: int
    file_hash: str
    entries: tuple[EntryRecord, ...]


@dataclass(frozen=True)
class LectureRecord:
    topic: str
    stem: str
    source_path: Path
    files: tuple[FileRecord, ...]


@dataclass(frozen=True)
class SyncStats:
    lectures: int = 0
    files_written: int = 0
    files_unchanged: int = 0
    entries_written: int = 0

    @property
    def files_seen(self) -> int:
        return self.files_written + self.files_unchanged


def file_hash(path: Path) -> str:
    """SHA-256 of the file's bytes — the idempotency key of the whole sync."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_language(path: Path) -> str | None:
    """Language of a translated output, or None when the file is a source.

    Returns the filename suffix, so ``Serbian (on Српски!)`` is stored as the
    ``Serbian`` its file is named after. The longest suffix wins.
    """
    for suffix in sorted(translation_suffixes(), key=len, reverse=True):
        if path.stem.endswith(suffix):
            return suffix[1:]
    return None


def parse_file(path: Path) -> list[srt.Subtitle]:
    """Parse and renumber, so ``idx`` is a position and not a copied index."""
    return srtio.reindex(srtio.parse(srtio.read_srt_text(path)))


def _milliseconds(value: timedelta) -> int:
    return value // timedelta(milliseconds=1)


def _verdicts(
    croatian_subs: list[srt.Subtitle] | None, subs: list[srt.Subtitle]
) -> dict[int, tuple[str, tuple[str, ...]]]:
    """Per-index status and flags from the Step 3 readability layers.

    Empty when the file cannot be aligned 1:1 with Croatian: every threshold in
    ``validation`` is relative to the aligned Croatian entry, so an unaligned
    file gets no verdict rather than a guessed one.
    """
    if croatian_subs is None or len(croatian_subs) != len(subs):
        return {}

    report = validation.validate_entries(croatian_subs, subs)
    verdicts: dict[int, tuple[str, tuple[str, ...]]] = {}
    for issues, status in ((report.warnings, STATUS_WARN), (report.failures, STATUS_FAIL)):
        for issue in issues:
            _, flags = verdicts.get(issue.index, (status, ()))
            verdicts[issue.index] = (status, (*flags, f"{issue.kind}_{issue.level}"))
    return verdicts


def build_file_record(
    path: Path, language: str, croatian_subs: list[srt.Subtitle] | None
) -> FileRecord:
    """Everything one SRT file contributes to the database."""
    subs = parse_file(path)
    verdicts = _verdicts(croatian_subs, subs)
    aligned = croatian_subs is not None and len(croatian_subs) == len(subs)
    default = STATUS_OK if aligned else STATUS_UNCHECKED

    entries = tuple(
        EntryRecord(
            idx=sub.index,
            start_ms=_milliseconds(sub.start),
            end_ms=_milliseconds(sub.end),
            content=sub.content,
            status=verdicts.get(sub.index, (default, ()))[0],
            flags=verdicts.get(sub.index, (default, ()))[1],
        )
        for sub in subs
    )
    return FileRecord(
        language=language,
        path=path,
        entry_count=len(entries),
        char_count=sum(len(entry.content) for entry in entries),
        file_hash=file_hash(path),
        entries=entries,
    )


def collect_lectures(
    input_dir: Path, target_languages: list[str] | None = None
) -> list[LectureRecord]:
    """Read the archive off disk: one record per source SRT, with its languages.

    Languages are read back off the FILENAMES with ``file_language``, not built
    from the configured target list, so a file whose language that list does not
    name still reaches the mirror. ``Swedish`` and ``Hebrew`` are the live case:
    selectable with ``--languages``, absent from ``TARGET_LANGUAGES``, and
    present in the archive — building the wanted paths from the list dropped
    them without a word.

    *target_languages* therefore narrows the sweep only when it is a real subset
    of the targets. The full target list — what ``sync-db`` passes when no
    ``--languages`` is given — means "no narrowing", because a mirror must not
    silently omit a file that is on disk.

    The Croatian file comes first because every other language is judged
    against it. A file that cannot be parsed is skipped with a warning instead
    of aborting a backfill of the whole archive.
    """
    narrow: set[str] | None = None
    if target_languages is not None and set(target_languages) != set(TARGET_LANGUAGES):
        narrow = {language_dir_name(language) for language in target_languages}
        narrow.add(CROATIAN_LANGUAGE)

    lectures: list[LectureRecord] = []

    for source in discover_source_srts(input_dir):
        croatian_path = source.with_name(f"{source.stem}{SOURCE_LANGUAGE_SUFFIX}.srt")
        croatian_subs = parse_file(croatian_path) if croatian_path.exists() else None

        wanted: list[tuple[Path, str]] = []
        for path in sorted(source.parent.glob("*.srt")):
            language = file_language(path)
            # Exact stem match: one topic folder can hold two sources whose
            # stems share a prefix, and each translation belongs to exactly one.
            if language is None or path.stem != f"{source.stem}_{language}":
                continue
            if narrow is not None and language not in narrow:
                continue
            wanted.append((path, language))
        wanted.sort(key=lambda item: (item[1] != CROATIAN_LANGUAGE, item[1]))

        files: list[FileRecord] = []
        for path, language in wanted:
            try:
                files.append(build_file_record(path, language, croatian_subs))
            except Exception as exc:  # a single unreadable file must not stop a backfill
                print(f"skipping {path}: {exc}")

        lectures.append(
            LectureRecord(
                topic=source.parent.name,
                stem=source.stem,
                source_path=source,
                files=tuple(files),
            )
        )
    return lectures


def apply_schema(cursor: Any) -> None:
    """Create the three tables if they are not there yet. Never alters others."""
    for statement in SCHEMA_STATEMENTS:
        cursor.execute(statement)


def sync_lectures(
    connection: Any, lectures: list[LectureRecord], create_schema: bool = True
) -> SyncStats:
    """Write *lectures* into an open DB-API connection, in one transaction.

    Idempotent: a file whose ``file_hash`` already matches the stored one is
    left alone, and a changed file has its segments deleted before the new ones
    are inserted.
    """
    lecture_count = 0
    files_written = 0
    files_unchanged = 0
    entries_written = 0

    with connection.cursor() as cursor:
        if create_schema:
            apply_schema(cursor)

        for lecture in lectures:
            cursor.execute(
                SQL_UPSERT_LECTURE,
                (lecture.topic, lecture.stem, str(lecture.source_path)),
            )
            lecture_id = cursor.fetchone()[0]
            lecture_count += 1

            for file in lecture.files:
                cursor.execute(SQL_SELECT_FILE, (lecture_id, file.language))
                existing = cursor.fetchone()
                if existing and existing[1] == file.file_hash:
                    files_unchanged += 1
                    continue

                cursor.execute(
                    SQL_UPSERT_FILE,
                    (
                        lecture_id,
                        file.language,
                        str(file.path),
                        file.entry_count,
                        file.char_count,
                        file.file_hash,
                    ),
                )
                file_id = cursor.fetchone()[0]
                cursor.execute(SQL_DELETE_ENTRIES, (file_id,))
                cursor.executemany(
                    SQL_INSERT_ENTRY,
                    [
                        (
                            file_id,
                            entry.idx,
                            entry.start_ms,
                            entry.end_ms,
                            entry.content,
                            entry.status,
                            list(entry.flags),
                        )
                        for entry in file.entries
                    ],
                )
                files_written += 1
                entries_written += len(file.entries)

    connection.commit()
    return SyncStats(
        lectures=lecture_count,
        files_written=files_written,
        files_unchanged=files_unchanged,
        entries_written=entries_written,
    )


def _dotenv_value(env_file: Path, key: str) -> str | None:
    """Read one key out of a .env file.

    Deliberately not ``pipeline.load_dotenv``: importing ``pipeline`` pulls in
    the Gemini translator, and this command must stay independent of it.
    """
    if not env_file.exists():
        return None
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip("'\"") or None
    return None


def database_url(explicit: str | None = None, env_file: Path = Path(".env")) -> str | None:
    """Explicit flag, then the environment, then ``.env``."""
    return explicit or os.environ.get("DATABASE_URL") or _dotenv_value(env_file, "DATABASE_URL")


def connect(url: str | None) -> Any:
    """Open a psycopg connection. Imported lazily: no DB driver at import time."""
    if not url:
        raise RuntimeError(
            "No DATABASE_URL: pass --database-url or set DATABASE_URL in .env"
        )
    import psycopg

    return psycopg.connect(url)
