"""Postgres sync: record building, idempotency via file_hash, CLI wiring.

Every test runs against a fake DB-API connection, so the suite needs neither a
live database nor DATABASE_URL.
"""

from pathlib import Path

import pytest

from srtljubav import db
from srtljubav.cli import main
from srtljubav.languages import TARGET_LANGUAGES


def write_srt(path: Path, blocks: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            f"{i}\n{start} --> {end}\n{text}\n"
            for i, (start, end, text) in enumerate(blocks, 1)
        ),
        encoding="utf-8",
    )


class FakeCursor:
    """Recognizes the module's SQL constants by identity, not by text."""

    def __init__(self, connection: "FakeConnection") -> None:
        self.connection = connection
        self._result: tuple | None = None

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str, params: tuple | None = None) -> None:
        self.connection.statements.append((sql, params))
        self._result = None
        if sql is db.SQL_UPSERT_LECTURE:
            self._result = (self.connection.next_id("lecture", (params[0], params[1])),)
        elif sql is db.SQL_SELECT_FILE:
            self._result = self.connection.existing_files.get(params)
        elif sql is db.SQL_UPSERT_FILE:
            self._result = (self.connection.next_id("file", (params[0], params[1])),)

    def executemany(self, sql: str, rows) -> None:
        self.connection.statements.append((sql, list(rows)))

    def fetchone(self) -> tuple | None:
        return self._result


class FakeConnection:
    def __init__(self, existing_files: dict | None = None) -> None:
        self.statements: list[tuple] = []
        self.existing_files = dict(existing_files or {})
        self.ids: dict[str, dict[tuple, int]] = {"lecture": {}, "file": {}}
        self.commits = 0
        self.closed = False

    def next_id(self, table: str, key: tuple) -> int:
        """Serial ids, stable per natural key, like the real ON CONFLICT upsert."""
        known = self.ids[table]
        return known.setdefault(key, len(known) + 1)

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1

    def close(self) -> None:
        self.closed = True

    def executed(self, sql: str) -> list:
        return [params for statement, params in self.statements if statement is sql]

    def entry_rows(self) -> list[tuple]:
        rows: list[tuple] = []
        for batch in self.executed(db.SQL_INSERT_ENTRY):
            rows.extend(batch)
        return rows


def two_language_lecture(tmp_path: Path) -> Path:
    topic = tmp_path / "VRIJEME JE KRAJA! 20"
    write_srt(
        topic / "lecture.srt",
        [("00:00:00,000", "00:00:04,000", "Prva rečenica."),
         ("00:00:04,000", "00:00:08,000", "Druga rečenica.")],
    )
    croatian = [
        ("00:00:00,000", "00:00:04,000", "Prva rečenica."),
        ("00:00:04,000", "00:00:08,000", "Druga rečenica."),
    ]
    write_srt(topic / "lecture_Croatian.srt", croatian)
    write_srt(
        topic / "lecture_English.srt",
        [("00:00:00,000", "00:00:04,000", "First sentence."),
         ("00:00:04,000", "00:00:08,000", "Second sentence.")],
    )
    return topic


# --- file_hash -------------------------------------------------------------


def test_file_hash_is_stable_and_content_addressed(tmp_path):
    first = tmp_path / "a.srt"
    second = tmp_path / "b.srt"
    first.write_text("1\n00:00:00,000 --> 00:00:01,000\nZdravo\n", encoding="utf-8")
    second.write_text("1\n00:00:00,000 --> 00:00:01,000\nZdravo\n", encoding="utf-8")

    assert db.file_hash(first) == db.file_hash(first)
    assert db.file_hash(first) == db.file_hash(second)

    second.write_text("1\n00:00:00,000 --> 00:00:01,000\nBok\n", encoding="utf-8")
    assert db.file_hash(first) != db.file_hash(second)


# --- record building -------------------------------------------------------


def test_file_language_reads_the_filename_suffix(tmp_path):
    assert db.file_language(tmp_path / "lecture_Croatian.srt") == "Croatian"
    assert db.file_language(tmp_path / "lecture_English.srt") == "English"
    # display name "Serbian (on Српски!)" is stored under its clean suffix
    assert db.file_language(tmp_path / "lecture_Serbian.srt") == "Serbian"
    assert db.file_language(tmp_path / "lecture.srt") is None


def test_build_file_record_marks_clean_entries_ok(tmp_path):
    topic = two_language_lecture(tmp_path)
    croatian = db.parse_file(topic / "lecture_Croatian.srt")

    record = db.build_file_record(topic / "lecture_English.srt", "English", croatian)

    assert record.language == "English"
    assert record.entry_count == 2
    assert record.char_count == len("First sentence.") + len("Second sentence.")
    assert [entry.idx for entry in record.entries] == [1, 2]
    assert record.entries[0].start_ms == 0
    assert record.entries[0].end_ms == 4000
    assert all(entry.status == db.STATUS_OK for entry in record.entries)
    assert all(entry.flags == () for entry in record.entries)


def test_build_file_record_flags_a_degenerate_entry_as_failed(tmp_path):
    topic = tmp_path / "topic"
    write_srt(
        topic / "lecture_Croatian.srt",
        [("00:00:00,000", "00:00:01,000", "Da."),
         ("00:00:01,000", "00:00:05,000", "I tako dalje.")],
    )
    write_srt(
        topic / "lecture_German.srt",
        [("00:00:00,000", "00:00:01,000", "ja " * 300),
         ("00:00:01,000", "00:00:05,000", "Und so weiter.")],
    )
    croatian = db.parse_file(topic / "lecture_Croatian.srt")

    record = db.build_file_record(topic / "lecture_German.srt", "German", croatian)

    assert record.entries[0].status == db.STATUS_FAIL
    assert "cps_fail" in record.entries[0].flags
    assert record.entries[1].status == db.STATUS_OK


def test_build_file_record_warns_on_unbalanced_bidi(tmp_path):
    topic = tmp_path / "topic"
    write_srt(topic / "lecture_Croatian.srt", [("00:00:00,000", "00:00:04,000", "Mir vama.")])
    write_srt(topic / "lecture_Arabic.srt", [("00:00:00,000", "00:00:04,000", "‫مرحبا")])
    croatian = db.parse_file(topic / "lecture_Croatian.srt")

    record = db.build_file_record(topic / "lecture_Arabic.srt", "Arabic", croatian)

    assert record.entries[0].status == db.STATUS_WARN
    assert record.entries[0].flags == ("bidi_warn",)


def test_build_file_record_is_unchecked_without_a_one_to_one_croatian_reference(tmp_path):
    topic = tmp_path / "topic"
    write_srt(topic / "lecture_Croatian.srt", [("00:00:00,000", "00:00:04,000", "Prva.")])
    write_srt(
        topic / "lecture_Thai.srt",
        [("00:00:00,000", "00:00:02,000", "หนึ่ง"),
         ("00:00:02,000", "00:00:04,000", "สอง")],
    )
    croatian = db.parse_file(topic / "lecture_Croatian.srt")

    not_aligned = db.build_file_record(topic / "lecture_Thai.srt", "Thai", croatian)
    no_reference = db.build_file_record(topic / "lecture_Thai.srt", "Thai", None)

    assert all(entry.status == db.STATUS_UNCHECKED for entry in not_aligned.entries)
    assert all(entry.status == db.STATUS_UNCHECKED for entry in no_reference.entries)


def test_collect_lectures_groups_the_source_with_its_translations(tmp_path):
    topic = two_language_lecture(tmp_path)
    write_srt(topic / "lecture_German.srt", [("00:00:00,000", "00:00:08,000", "Erster Satz.")])

    lectures = db.collect_lectures(tmp_path)

    assert len(lectures) == 1
    lecture = lectures[0]
    assert lecture.topic == "VRIJEME JE KRAJA! 20"
    assert lecture.stem == "lecture"
    assert lecture.source_path == topic / "lecture.srt"
    # Croatian first, then the target languages that exist on disk
    assert [file.language for file in lecture.files] == ["Croatian", "English", "German"]
    german = lecture.files[2]
    assert german.entry_count == 1
    # not 1:1 with Croatian, so its entries carry no verdict
    assert german.entries[0].status == db.STATUS_UNCHECKED


def test_collect_lectures_honours_a_language_subset(tmp_path):
    topic = two_language_lecture(tmp_path)
    write_srt(topic / "lecture_German.srt", [("00:00:00,000", "00:00:08,000", "Erster Satz.")])

    lectures = db.collect_lectures(tmp_path, target_languages=["German"])

    assert [file.language for file in lectures[0].files] == ["Croatian", "German"]


def test_collect_lectures_mirrors_a_language_the_target_list_does_not_name(tmp_path):
    """Swedish/Hebrew are selectable via ``--languages`` but not among the 33
    targets, and the real archive holds such files. ``sync-db`` without
    ``--languages`` passes the full target list, so building the wanted paths
    from that list dropped those files from the mirror without a word."""
    topic = two_language_lecture(tmp_path)
    write_srt(
        topic / "lecture_Swedish.srt",
        [("00:00:00,000", "00:00:04,000", "Forsta meningen."),
         ("00:00:04,000", "00:00:08,000", "Andra meningen.")],
    )

    lectures = db.collect_lectures(tmp_path, target_languages=list(TARGET_LANGUAGES))

    assert [file.language for file in lectures[0].files] == ["Croatian", "English", "Swedish"]
    swedish = lectures[0].files[2]
    assert swedish.entry_count == 2
    # 1:1 with Croatian, so it is judged like any other language
    assert all(entry.status == db.STATUS_OK for entry in swedish.entries)


def test_collect_lectures_matches_the_stem_exactly_not_by_prefix(tmp_path):
    """Guard for the filename-driven sweep: one source stem is a prefix of the
    other, so a prefix match would file ``lecture_two_English.srt`` under both."""
    topic = two_language_lecture(tmp_path)
    write_srt(topic / "lecture_two.srt", [("00:00:00,000", "00:00:04,000", "Prva.")])
    write_srt(topic / "lecture_two_English.srt", [("00:00:00,000", "00:00:04,000", "First.")])

    by_stem = {lecture.stem: lecture for lecture in db.collect_lectures(tmp_path)}

    assert sorted(by_stem) == ["lecture", "lecture_two"]
    assert [file.path.name for file in by_stem["lecture"].files] == [
        "lecture_Croatian.srt",
        "lecture_English.srt",
    ]
    assert [file.path.name for file in by_stem["lecture_two"].files] == [
        "lecture_two_English.srt"
    ]


# --- schema ----------------------------------------------------------------


def test_schema_statements_are_idempotent_and_touch_no_existing_table():
    joined = "\n".join(db.SCHEMA_STATEMENTS)

    assert joined.count("CREATE TABLE IF NOT EXISTS") == 3
    for table in ("lecture", "subtitle_file", "subtitle_entry"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in joined
    # the amendment: per-segment verdicts
    assert "status" in joined and "flags" in joined
    # the unrelated tables in the same database stay untouched
    assert "seminar" not in joined and "prijevod_seminara" not in joined
    assert "DROP " not in joined.upper()


def test_apply_schema_runs_every_statement():
    connection = FakeConnection()

    with connection.cursor() as cursor:
        db.apply_schema(cursor)

    executed = [sql for sql, _ in connection.statements]
    assert executed == list(db.SCHEMA_STATEMENTS)


# --- sync ------------------------------------------------------------------


def test_sync_writes_lecture_file_and_entry_rows(tmp_path):
    two_language_lecture(tmp_path)
    lectures = db.collect_lectures(tmp_path)
    connection = FakeConnection()

    stats = db.sync_lectures(connection, lectures)

    assert stats.lectures == 1
    assert stats.files_written == 2
    assert stats.files_unchanged == 0
    assert stats.entries_written == 4
    assert connection.executed(db.SQL_UPSERT_LECTURE) == [
        ("VRIJEME JE KRAJA! 20", "lecture", str(tmp_path / "VRIJEME JE KRAJA! 20" / "lecture.srt"))
    ]
    languages = [params[1] for params in connection.executed(db.SQL_UPSERT_FILE)]
    assert languages == ["Croatian", "English"]
    rows = connection.entry_rows()
    assert len(rows) == 4
    # (file_id, idx, start_ms, end_ms, content, status, flags)
    assert rows[0][1:5] == (1, 0, 4000, "Prva rečenica.")
    assert rows[0][5] == db.STATUS_OK
    assert rows[0][6] == []
    assert connection.commits == 1


def test_sync_skips_a_file_whose_hash_is_unchanged(tmp_path):
    topic = two_language_lecture(tmp_path)
    lectures = db.collect_lectures(tmp_path)
    english_hash = db.file_hash(topic / "lecture_English.srt")
    connection = FakeConnection(existing_files={(1, "English"): (7, english_hash)})

    stats = db.sync_lectures(connection, lectures)

    assert stats.files_written == 1
    assert stats.files_unchanged == 1
    assert [params[1] for params in connection.executed(db.SQL_UPSERT_FILE)] == ["Croatian"]
    assert all(row[0] != 7 for row in connection.entry_rows())


def test_sync_replaces_the_entries_of_a_changed_file(tmp_path):
    two_language_lecture(tmp_path)
    lectures = db.collect_lectures(tmp_path)
    connection = FakeConnection(existing_files={(1, "English"): (7, "stale-hash")})

    stats = db.sync_lectures(connection, lectures)

    assert stats.files_written == 2
    assert stats.files_unchanged == 0
    # entries are deleted before being rewritten, so a shrunk file leaves no orphans
    deleted = connection.executed(db.SQL_DELETE_ENTRIES)
    assert len(deleted) == 2
    delete_at = next(
        i for i, (sql, _) in enumerate(connection.statements) if sql is db.SQL_DELETE_ENTRIES
    )
    insert_at = next(
        i for i, (sql, _) in enumerate(connection.statements) if sql is db.SQL_INSERT_ENTRY
    )
    assert delete_at < insert_at


def test_sync_is_idempotent_when_run_twice(tmp_path):
    topic = two_language_lecture(tmp_path)
    lectures = db.collect_lectures(tmp_path)
    existing = {
        (1, "Croatian"): (1, db.file_hash(topic / "lecture_Croatian.srt")),
        (1, "English"): (2, db.file_hash(topic / "lecture_English.srt")),
    }
    connection = FakeConnection(existing_files=existing)

    stats = db.sync_lectures(connection, lectures)

    assert stats.files_written == 0
    assert stats.files_unchanged == 2
    assert stats.entries_written == 0
    assert connection.entry_rows() == []


def test_sync_can_skip_the_schema():
    connection = FakeConnection()

    db.sync_lectures(connection, [], create_schema=False)

    assert connection.statements == []


# --- connection settings ---------------------------------------------------


def test_database_url_prefers_the_explicit_value_then_env_then_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("DATABASE_URL=postgresql:///from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    assert db.database_url("postgresql:///explicit", env_file) == "postgresql:///explicit"
    assert db.database_url(None, env_file) == "postgresql:///from-dotenv"

    monkeypatch.setenv("DATABASE_URL", "postgresql:///from-env")
    assert db.database_url(None, env_file) == "postgresql:///from-env"
    assert db.database_url(None, tmp_path / "missing.env") == "postgresql:///from-env"

    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert db.database_url(None, tmp_path / "missing.env") is None


def test_connect_without_a_url_raises_instead_of_guessing():
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        db.connect(None)


# --- CLI -------------------------------------------------------------------


def test_cli_sync_db_dry_run_reports_without_connecting(tmp_path, monkeypatch, capsys):
    two_language_lecture(tmp_path)

    def fail(*args, **kwargs):
        raise AssertionError("--dry-run must not open a connection")

    monkeypatch.setattr(db, "connect", fail)

    code = main(["sync-db", "--input-dir", str(tmp_path), "--dry-run"])

    out = capsys.readouterr().out
    assert code == 0
    assert "1 lecture(s)" in out
    assert "2 file(s)" in out
    assert "4 entries" in out


def test_cli_sync_db_syncs_through_the_connection(tmp_path, monkeypatch, capsys):
    two_language_lecture(tmp_path)
    connection = FakeConnection()
    monkeypatch.setattr(db, "connect", lambda url: connection)

    code = main(
        [
            "sync-db",
            "--input-dir",
            str(tmp_path),
            "--database-url",
            "postgresql:///srt",
        ]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert connection.commits == 1
    assert connection.closed
    assert len(connection.entry_rows()) == 4
    assert "2 file(s) written" in out


def test_cli_sync_db_without_a_database_url_fails_cleanly(tmp_path, monkeypatch, capsys):
    two_language_lecture(tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)

    code = main(["sync-db", "--input-dir", str(tmp_path)])

    assert code == 2
    assert "DATABASE_URL" in capsys.readouterr().err
