"""Database construction: schema, contents, and the incremental path."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

import duckdb
import pytest

from ashiato import build as build_module
from ashiato.build import (
    BULK_INSERT_MIN_ROWS as BULK_MIN,
)
from ashiato.build import (
    SchemaOutOfDate,
    assert_readable,
    build,
    connect,
    create_schema,
    database_info,
    default_db_path,
    iter_cursor_sources,
    iter_opencode_sources,
    iter_transcripts,
)
from ashiato.cli import main
from ashiato.opencode import OpenCodeToolCall, ParsedOpenCodeFile
from ashiato.parser import EVENT_COLUMNS, SESSION_COLUMNS, TOOL_CALL_COLUMNS
from ashiato.recall import RECALL_CALL_COLUMNS, extract_from_opencode
from ashiato.schema import (
    FOLLOWUP_KINDS,
    FORMAT_VERSION,
    INFO_TABLES,
    META_FORMAT_KEY,
    META_TABLE,
    RECALL_FOLLOWUPS_VIEW,
    SOURCE_FILE_COLUMNS,
    TABLES,
)

FIXTURES = Path(__file__).parent / "fixtures"
MAIN_SESSION_ID = "11111111-1111-4111-8111-111111111111"

# chain.jsonl (50) + session_main.jsonl (17) + session_snake.jsonl (3); empty.jsonl adds none.
TOTAL_EVENTS = 70
TOTAL_TOOL_CALLS = 7
TOTAL_SESSIONS = 3


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "test.duckdb"


@pytest.fixture
def built(db: Path):
    result = build([FIXTURES], db)
    connection = connect(db, read_only=True)
    yield result, connection
    connection.close()


def scalar(connection: duckdb.DuckDBPyConnection, query: str, *params):
    return connection.execute(query, list(params)).fetchone()[0]


# ---------------------------------------------------------------- discovery


def test_iter_transcripts_is_recursive_sorted_and_deduplicated(tmp_path: Path):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    (nested / "z.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "a" / "m.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "a" / "ignore.txt").write_text("", encoding="utf-8")

    # The same directory twice must not yield the same file twice.
    files, missing = iter_transcripts([tmp_path, tmp_path / "a"])
    assert sorted(path.name for path in files) == ["m.jsonl", "z.jsonl"]
    assert files == sorted(files, key=str)  # deterministic order
    assert missing == []


def test_iter_transcripts_reports_missing_sources(tmp_path: Path):
    files, missing = iter_transcripts([tmp_path / "nope"])
    assert files == []
    assert missing == [str(tmp_path / "nope")]


def test_default_db_path_follows_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    assert default_db_path() == tmp_path / "data" / "ashiato" / "ashiato.duckdb"
    monkeypatch.delenv("XDG_DATA_HOME")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    assert default_db_path() == tmp_path / "home" / ".local" / "share" / "ashiato" / "ashiato.duckdb"


# ---------------------------------------------------------------- schema


def test_schema_column_order_matches_the_dataclasses(built):
    _, connection = built
    expected = {
        "sessions": SESSION_COLUMNS,
        "events": EVENT_COLUMNS,
        "tool_calls": TOOL_CALL_COLUMNS,
        "source_files": SOURCE_FILE_COLUMNS,
    }
    for table, columns in expected.items():
        actual = [row[1] for row in connection.execute(f"PRAGMA table_info('{table}')").fetchall()]
        assert actual == list(columns), table


def test_build_creates_parent_directories(tmp_path: Path):
    target = tmp_path / "deep" / "nested" / "ashiato.duckdb"
    build([FIXTURES], target)
    assert target.exists()


# ---------------------------------------------------------------- contents


def test_build_counts(built):
    result, _ = built
    assert result.n_files == 4
    assert result.n_processed == 4
    assert result.n_skipped == 0
    assert result.n_sessions == TOTAL_SESSIONS
    assert result.n_events == TOTAL_EVENTS
    assert result.n_tool_calls == TOTAL_TOOL_CALLS
    assert result.n_parse_errors == 2
    assert result.missing_sources == []
    assert result.unreadable_files == []


def test_table_row_counts(built):
    _, connection = built
    assert scalar(connection, "SELECT count(*) FROM sessions") == TOTAL_SESSIONS
    assert scalar(connection, "SELECT count(*) FROM events") == TOTAL_EVENTS
    assert scalar(connection, "SELECT count(*) FROM tool_calls") == TOTAL_TOOL_CALLS
    assert scalar(connection, "SELECT count(*) FROM source_files") == 4


def test_outcome_distribution(built):
    _, connection = built
    rows = connection.execute(
        "SELECT outcome, count(*) FROM tool_calls GROUP BY 1 ORDER BY 1"
    ).fetchall()
    assert rows == [("denied", 2), ("error", 1), ("ok", 3), ("pending", 1)]


def test_tool_kind_and_mcp_server_in_sql(built):
    _, connection = built
    rows = connection.execute(
        "SELECT tool_name, tool_kind, mcp_server FROM tool_calls "
        "WHERE tool_kind = 'mcp' ORDER BY tool_name"
    ).fetchall()
    assert rows == [
        ("mcp__shiori__search", "mcp", "shiori"),
        ("mcp__sunaba__publish", "mcp", "sunaba"),
    ]


def test_input_is_queryable_as_json(built):
    _, connection = built
    command = scalar(
        connection,
        "SELECT input->>'$.command' FROM tool_calls WHERE tool_use_id = 'toolu_ok_1'",
    )
    assert command == "ls -1"


def test_token_totals_are_deduplicated_in_the_database(built):
    _, connection = built
    row = connection.execute(
        "SELECT input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens "
        "FROM sessions WHERE session_id = ?",
        [MAIN_SESSION_ID],
    ).fetchone()
    assert row == (1223, 63, 5018, 318)


def test_session_metadata_in_the_database(built):
    _, connection = built
    row = connection.execute(
        "SELECT project_dir, cwd, git_branch, cc_version, entrypoint, started_at, ended_at, "
        "n_events, n_tool_calls FROM sessions WHERE session_id = ?",
        [MAIN_SESSION_ID],
    ).fetchone()
    assert row == (
        "fixtures",
        "/home/dev/proj",
        "feature/parser",
        "2.0.31",
        "cli",
        datetime(2026, 8, 10, 9, 0, 0),
        datetime(2026, 8, 10, 9, 0, 16),
        17,
        6,
    )


def test_events_keep_the_raw_line_and_depth(built):
    _, connection = built
    depth, raw = connection.execute(
        "SELECT depth, raw FROM events WHERE event_id = 'u14'"
    ).fetchone()
    assert depth == 12
    assert '"isMeta":true' in raw
    assert scalar(connection, "SELECT max(depth) FROM events WHERE session_id LIKE '3333%'") == 49


def test_source_files_bookkeeping(built):
    _, connection = built
    row = connection.execute(
        "SELECT size_bytes, content_hash, n_events, n_tool_calls, n_parse_errors, built_at "
        "FROM source_files WHERE file_path = ?",
        [str((FIXTURES / "session_main.jsonl").resolve())],
    ).fetchone()
    size, content_hash, n_events, n_tool_calls, n_parse_errors, built_at = row
    assert size == (FIXTURES / "session_main.jsonl").stat().st_size
    assert len(content_hash) == 64
    assert (n_events, n_tool_calls, n_parse_errors) == (17, 6, 2)
    assert isinstance(built_at, datetime)


def test_empty_file_is_recorded_but_produces_no_session(built):
    _, connection = built
    empty = str((FIXTURES / "empty.jsonl").resolve())
    assert scalar(connection, "SELECT count(*) FROM source_files WHERE file_path = ?", empty) == 1
    assert scalar(connection, "SELECT count(*) FROM sessions WHERE file_path = ?", empty) == 0


# ---------------------------------------------------------------- incremental


@pytest.fixture
def source_copy(tmp_path: Path) -> Path:
    target = tmp_path / "src"
    shutil.copytree(FIXTURES, target)
    return target


def test_second_build_skips_unchanged_files(source_copy: Path, db: Path):
    build([source_copy], db)
    again = build([source_copy], db)
    assert again.n_processed == 0
    assert again.n_skipped == 4
    assert again.n_events == 0

    connection = connect(db, read_only=True)
    try:
        assert scalar(connection, "SELECT count(*) FROM events") == TOTAL_EVENTS
        assert scalar(connection, "SELECT count(*) FROM sessions") == TOTAL_SESSIONS
    finally:
        connection.close()


def test_changed_file_is_reparsed_and_replaced_not_duplicated(source_copy: Path, db: Path):
    build([source_copy], db)
    target = source_copy / "session_snake.jsonl"
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(
            '{"type":"user","session_id":"22222222-2222-4222-8222-222222222222","uuid":"s4",'
            '"parentUuid":"s3","timestamp":"2026-08-10T10:00:04.000Z",'
            '"message":{"role":"user","content":"one more"}}\n'
        )
    stat = target.stat()
    os.utime(target, (stat.st_atime, stat.st_mtime + 10))

    result = build([source_copy], db)
    assert result.n_processed == 1
    assert result.n_skipped == 3

    connection = connect(db, read_only=True)
    try:
        assert scalar(connection, "SELECT count(*) FROM events") == TOTAL_EVENTS + 1
        # One session row for the file, not two.
        assert (
            scalar(connection, "SELECT count(*) FROM sessions WHERE file_path = ?", str(target))
            == 1
        )
        assert scalar(connection, "SELECT n_events FROM sessions WHERE file_path = ?", str(target))
        assert scalar(connection, "SELECT count(*) FROM source_files") == 4
        assert scalar(connection, "SELECT count(*) FROM tool_calls") == TOTAL_TOOL_CALLS
    finally:
        connection.close()


def test_touching_a_file_without_changing_size_still_reparses(source_copy: Path, db: Path):
    build([source_copy], db)
    target = source_copy / "chain.jsonl"
    stat = target.stat()
    os.utime(target, (stat.st_atime, stat.st_mtime + 60))
    result = build([source_copy], db)
    assert result.n_processed == 1
    assert result.n_skipped == 3

    connection = connect(db, read_only=True)
    try:
        assert scalar(connection, "SELECT count(*) FROM events") == TOTAL_EVENTS
    finally:
        connection.close()


def test_a_new_file_is_picked_up(source_copy: Path, db: Path):
    build([source_copy], db)
    (source_copy / "extra.jsonl").write_text(
        '{"type":"user","sessionId":"44444444-4444-4444-8444-444444444444","uuid":"e1",'
        '"parentUuid":null,"timestamp":"2026-08-10T13:00:00.000Z",'
        '"message":{"role":"user","content":"hi"}}\n',
        encoding="utf-8",
    )
    result = build([source_copy], db)
    assert (result.n_processed, result.n_skipped) == (1, 4)

    connection = connect(db, read_only=True)
    try:
        assert scalar(connection, "SELECT count(*) FROM sessions") == TOTAL_SESSIONS + 1
    finally:
        connection.close()


# ---------------------------------------------------------------- info


def test_database_info(built, db: Path):
    _, _ = built
    info = database_info(db)
    assert info.db_path == str(db)
    assert info.table_counts == {
        "sessions": TOTAL_SESSIONS,
        "events": TOTAL_EVENTS,
        "tool_calls": TOTAL_TOOL_CALLS,
        "source_files": 4,
    }
    assert info.started_at == datetime(2026, 8, 10, 9, 0, 0)
    assert info.ended_at == datetime(2026, 8, 10, 11, 0, 49)


# ---------------------------------------------------------------- insert paths

DATA_TABLES = {
    "sessions": "file_path",
    "events": "file_path, seq",
    "tool_calls": "file_path, seq, tool_use_id",
}


def dump(db_path: Path, table: str, order: str) -> list[tuple]:
    connection = connect(db_path, read_only=True)
    try:
        return connection.execute(f'SELECT * FROM "{table}" ORDER BY {order}').fetchall()
    finally:
        connection.close()


def test_bulk_and_row_by_row_inserts_produce_identical_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The fast path exists only for speed; it must not change a single value."""
    bulk_db = tmp_path / "bulk.duckdb"
    row_db = tmp_path / "row.duckdb"

    monkeypatch.setattr(build_module, "BULK_INSERT_MIN_ROWS", 1)
    assert build([FIXTURES], bulk_db).n_bulk_fallbacks == 0
    monkeypatch.setattr(build_module, "BULK_INSERT_MIN_ROWS", 10**9)
    build([FIXTURES], row_db)

    for table, order in DATA_TABLES.items():
        assert dump(bulk_db, table, order) == dump(row_db, table, order), table
    assert dump(bulk_db, "events", "file_path, seq")  # not vacuously equal


def test_build_falls_back_when_the_bulk_path_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def explode(*args, **kwargs):
        raise duckdb.Error("bulk load refused")

    monkeypatch.setattr(build_module, "_bulk_insert", explode)
    db_path = tmp_path / "fallback.duckdb"
    result = build([FIXTURES], db_path)

    # chain.jsonl and session_main.jsonl are the two files big enough to try it.
    assert result.n_bulk_fallbacks == 2
    assert result.failed_files == []
    assert result.n_events == TOTAL_EVENTS
    connection = connect(db_path, read_only=True)
    try:
        assert scalar(connection, "SELECT count(*) FROM events") == TOTAL_EVENTS
        assert scalar(connection, "SELECT count(*) FROM tool_calls") == TOTAL_TOOL_CALLS
        assert scalar(connection, "SELECT count(*) FROM sessions") == TOTAL_SESSIONS
    finally:
        connection.close()


def test_a_file_that_cannot_be_stored_is_reported_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def explode(*args, **kwargs):
        raise duckdb.Error("no room at the inn")

    monkeypatch.setattr(build_module, "_insert_parsed", explode)
    result = build([FIXTURES], tmp_path / "broken.duckdb")
    assert len(result.failed_files) == 4
    assert result.n_processed == 0


def test_awkward_text_survives_the_round_trip(tmp_path: Path):
    """Newlines, quotes, tabs, CJK and emoji must come back byte-identical."""
    source = tmp_path / "src"
    source.mkdir()
    text = 'line1\nline2\t"quoted" \\ backslash 日本語 🐾 ünïcode'
    path = source / "awkward.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for index in range(BULK_MIN + 5):  # enough rows to take the bulk path
            handle.write(
                json.dumps(
                    {
                        "type": "user",
                        "uuid": f"a{index}",
                        "parentUuid": None if index == 0 else f"a{index - 1}",
                        "sessionId": "55555555-5555-4555-8555-555555555555",
                        "timestamp": "2026-08-10T14:00:00.000Z",
                        "message": {"role": "user", "content": text},
                    }
                )
                + "\n"
            )

    db_path = tmp_path / "awkward.duckdb"
    build([source], db_path)
    connection = connect(db_path, read_only=True)
    try:
        stored = scalar(connection, "SELECT text FROM events WHERE event_id = 'a0'")
        assert stored == text
        raw = scalar(connection, "SELECT raw FROM events WHERE event_id = 'a0'")
        assert json.loads(raw)["message"]["content"] == text
    finally:
        connection.close()


def test_lone_surrogates_do_not_crash_the_build(tmp_path: Path):
    """A transcript can contain an unpaired surrogate escape; it must not be fatal."""
    source = tmp_path / "src"
    source.mkdir()
    with open(source / "surrogate.jsonl", "w", encoding="utf-8") as handle:
        for index in range(BULK_MIN + 5):
            handle.write(
                f'{{"type":"user","uuid":"s{index}","parentUuid":null,'
                '"sessionId":"66666666-6666-4666-8666-666666666666",'
                '"timestamp":"2026-08-10T15:00:00.000Z",'
                '"message":{"role":"user","content":"broken \\ud800 pair"}}\n'
            )
    result = build([source], tmp_path / "surrogate.duckdb")
    assert result.n_events == BULK_MIN + 5
    assert result.failed_files == []


# ---------------------------------------------------------------- denial followups

#: The first of DENIAL_PATTERNS, as Claude Code writes it into a tool result.
DENIAL_TEXT = "The user doesn't want to proceed with this tool use. The tool call was rejected."


def write_session(path: Path, session_id: str, calls) -> None:
    """A transcript of back-to-back tool calls, each of them denied or not.

    Two lines per call -- the assistant's ``tool_use`` and the user's
    ``tool_result`` -- so ``seq`` advances the way it does in a real transcript
    and one call's result sits between it and the next call.
    """
    lines = []
    clock = 0
    for index, (tool_name, tool_input, denied) in enumerate(calls):
        use_id = f"toolu_{index}"
        lines.append(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": f"a{index}",
                    "parentUuid": None if index == 0 else f"r{index - 1}",
                    "sessionId": session_id,
                    "timestamp": f"2026-08-10T12:00:{clock:02d}.000Z",
                    "cwd": "/home/dev/proj",
                    "permissionMode": "default",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": use_id,
                                "name": tool_name,
                                "input": tool_input,
                            }
                        ],
                    },
                }
            )
        )
        clock += 1
        lines.append(
            json.dumps(
                {
                    "type": "user",
                    "uuid": f"r{index}",
                    "parentUuid": f"a{index}",
                    "sessionId": session_id,
                    "timestamp": f"2026-08-10T12:00:{clock:02d}.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": use_id,
                                "content": DENIAL_TEXT if denied else "fine",
                                "is_error": denied,
                            }
                        ],
                    },
                }
            )
        )
        clock += 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


#: One session per followup_kind.  Real corpora show all of these: a verbatim
#: retry of the rejected command, the same command narrowed down, and a switch
#: to a different tool entirely.
FOLLOWUP_SESSIONS = {
    "aaaa": [
        ("Bash", {"command": "gh pr merge 18 --squash"}, True),
        ("Bash", {"command": "gh pr merge 18 --squash"}, False),
    ],
    "bbbb": [
        ("Bash", {"command": "gh pr merge 18 --squash && git pull"}, True),
        ("Bash", {"command": "gh pr merge 18 --squash"}, False),
    ],
    "cccc": [
        ("Write", {"file_path": "/etc/hosts", "content": "127.0.0.1 nope"}, True),
        ("Read", {"file_path": "/etc/hosts"}, False),
    ],
    "dddd": [
        ("Bash", {"command": "rm -rf /"}, True),
    ],
}


@pytest.fixture
def followup_source(tmp_path: Path) -> Path:
    source = tmp_path / "followups"
    source.mkdir()
    for session_id, calls in FOLLOWUP_SESSIONS.items():
        write_session(source / f"{session_id}.jsonl", session_id, calls)
    return source


@pytest.fixture
def followups(followup_source: Path, tmp_path: Path):
    build([followup_source], tmp_path / "followups.duckdb")
    connection = connect(tmp_path / "followups.duckdb", read_only=True)
    yield connection
    connection.close()


def test_followup_kind_covers_every_case(followups):
    rows = followups.execute(
        "SELECT session_id, tool_name, input_summary, next_tool_name, next_input_summary, "
        "next_outcome, next_ts, gap_seconds, followup_kind "
        "FROM denial_followups ORDER BY session_id"
    ).fetchall()
    assert rows == [
        (
            "aaaa",
            "Bash",
            "gh pr merge 18 --squash",
            "Bash",
            "gh pr merge 18 --squash",
            "ok",
            datetime(2026, 8, 10, 12, 0, 2),
            2.0,
            "verbatim-retry",
        ),
        (
            "bbbb",
            "Bash",
            "gh pr merge 18 --squash && git pull",
            "Bash",
            "gh pr merge 18 --squash",
            "ok",
            datetime(2026, 8, 10, 12, 0, 2),
            2.0,
            "same-tool",
        ),
        (
            "cccc",
            "Write",
            "/etc/hosts",
            "Read",
            "/etc/hosts",
            "ok",
            datetime(2026, 8, 10, 12, 0, 2),
            2.0,
            "other-tool",
        ),
        # The denial was the last thing the session did: nothing followed it.
        ("dddd", "Bash", "rm -rf /", None, None, None, None, None, "none"),
    ]


def test_followup_kind_takes_only_the_four_documented_values(followups):
    kinds = {
        row[0]
        for row in followups.execute(
            "SELECT DISTINCT followup_kind FROM denial_followups"
        ).fetchall()
    }
    assert kinds == set(FOLLOWUP_KINDS)


def test_a_narrowed_retry_is_not_a_verbatim_one(followups):
    """The distinction is the whole point: 'bbbb' retried a *shorter* command."""
    kind = scalar(followups, "SELECT followup_kind FROM denial_followups WHERE session_id = 'bbbb'")
    assert kind == "same-tool"


def test_every_denied_call_appears_exactly_once(built):
    _, connection = built
    denied = connection.execute(
        "SELECT session_id, seq FROM tool_calls WHERE outcome = 'denied' ORDER BY 1, 2"
    ).fetchall()
    assert len(denied) == 2
    assert (
        connection.execute("SELECT session_id, seq FROM denial_followups ORDER BY 1, 2").fetchall()
        == denied
    )


def test_denial_followups_on_the_fixture_corpus(built):
    _, connection = built
    rows = connection.execute(
        "SELECT seq, tool_name, input_summary, permission_mode, cwd, next_tool_name, "
        "next_outcome, followup_kind FROM denial_followups ORDER BY seq"
    ).fetchall()
    assert rows == [
        (
            6,
            "Write",
            "/etc/hosts",
            "default",
            "/home/dev/proj",
            "mcp__sunaba__publish",
            "denied",
            "other-tool",
        ),
        (
            8,
            "mcp__sunaba__publish",
            '{"create_pr":true,"files":["src/ashiato/parser.py"]}',
            "default",
            "/home/dev/proj",
            "Bash",
            "pending",
            "other-tool",
        ),
    ]


def test_the_next_call_is_the_next_one_in_the_same_session(followup_source: Path, db: Path):
    """A neighbouring session's calls must not be picked up as a followup."""
    write_session(followup_source / "eeee.jsonl", "eeee", [("Glob", {"pattern": "**/*.py"}, False)])
    build([followup_source], db)
    connection = connect(db, read_only=True)
    try:
        # 'dddd' still ends with its denial even though 'eeee' has a later call.
        kind = scalar(
            connection, "SELECT followup_kind FROM denial_followups WHERE session_id = 'dddd'"
        )
        assert kind == "none"
        assert scalar(connection, "SELECT count(*) FROM denial_followups") == 4
    finally:
        connection.close()


def test_input_summary_is_populated_for_bash_calls(built):
    _, connection = built
    rows = connection.execute(
        "SELECT tool_use_id, input_summary FROM tool_calls WHERE tool_name = 'Bash' ORDER BY 1"
    ).fetchall()
    assert rows == [("toolu_ok_1", "ls -1"), ("toolu_pending_1", "sleep 600")]
    # Nothing in the corpus has an input but no summary of it.
    assert (
        scalar(
            connection,
            "SELECT count(*) FROM tool_calls WHERE input IS NOT NULL AND input_summary IS NULL",
        )
        == 0
    )


def test_the_view_is_rebuilt_with_the_rows_it_summarises(source_copy: Path, db: Path):
    """The view is derived on read, so an incremental rebuild cannot leave it stale."""
    build([source_copy], db)
    write_session(
        source_copy / "late.jsonl",
        "ffff",
        [("Bash", {"command": "curl example.com"}, True), ("Bash", {"command": "echo no"}, False)],
    )
    build([source_copy], db)

    connection = connect(db, read_only=True)
    try:
        kind = scalar(
            connection, "SELECT followup_kind FROM denial_followups WHERE session_id = 'ffff'"
        )
        assert kind == "same-tool"
        assert scalar(connection, "SELECT count(*) FROM denial_followups") == 3
    finally:
        connection.close()


def test_two_builds_of_the_same_bytes_give_the_same_view(tmp_path: Path, followup_source: Path):
    """Determinism: the frozen source must produce the same rows every time."""
    first, second = tmp_path / "first.duckdb", tmp_path / "second.duckdb"
    build([followup_source], first)
    build([followup_source], second)
    order = "session_id, seq"
    assert dump(first, "denial_followups", order) == dump(second, "denial_followups", order)
    assert len(dump(first, "denial_followups", order)) == len(FOLLOWUP_SESSIONS)


def test_a_database_built_by_an_older_schema_is_refused(tmp_path: Path):
    """The incremental build would skip every file, so the missing column must not pass."""
    db_path = tmp_path / "old.duckdb"
    build([FIXTURES], db_path)
    connection = connect(db_path)
    try:
        connection.execute("DROP VIEW denial_followups")
        connection.execute("ALTER TABLE tool_calls DROP COLUMN input_summary")
    finally:
        connection.close()

    with pytest.raises(SchemaOutOfDate, match="input_summary"):
        build([FIXTURES], db_path)


def test_the_format_marker_is_written_on_build(db: Path):
    """A fresh build stamps the current row-rule version into the meta table."""
    build([FIXTURES], db)
    connection = connect(db, read_only=True)
    try:
        (value,) = connection.execute(
            f'SELECT value FROM "{META_TABLE}" WHERE key = ?', [META_FORMAT_KEY]
        ).fetchone()
        assert value == str(FORMAT_VERSION)
    finally:
        connection.close()


def test_a_crash_inside_create_schema_leaves_no_refuse_only_database(tmp_path: Path):
    """The DDL and the marker stamp are one transaction: a crash between them
    must not leave ashiato tables with no marker, which the next build would
    refuse even though the file is empty and perfectly rebuildable.
    """
    db_path = tmp_path / "atomic.duckdb"
    # Make the marker stamp fail mid-create_schema -- the crash point that used
    # to strand the database.  A pre-created meta table whose CHECK rejects the
    # marker value lets every DDL statement succeed and the marker INSERT fail.
    connection = connect(db_path)
    try:
        connection.execute(
            f'CREATE TABLE "{META_TABLE}" ('
            "key VARCHAR PRIMARY KEY, value VARCHAR CHECK (length(value) > 1000))"
        )
    finally:
        connection.close()

    connection = connect(db_path)
    try:
        with pytest.raises(duckdb.Error):
            create_schema(connection)
    finally:
        connection.close()

    # The transaction rolled back: no half-created ashiato table is left behind.
    connection = connect(db_path, read_only=True)
    try:
        tables = {
            row[0]
            for row in connection.execute("SELECT table_name FROM duckdb_tables()").fetchall()
        }
    finally:
        connection.close()
    assert not (set(TABLES) & tables)

    # With the scaffolding gone the file is exactly what a crash leaves: empty.
    # The next build starts fresh instead of refusing.
    connection = connect(db_path)
    try:
        connection.execute(f'DROP TABLE "{META_TABLE}"')
    finally:
        connection.close()
    result = build([FIXTURES], db_path)
    assert result.n_processed == 4


def test_a_database_built_under_the_old_outcome_rule_is_refused(tmp_path: Path):
    """`outcome` is a stored column: a DB whose rows were classified by the old
    substring rule must be refused with the rebuild message, not silently mixed
    or half-upgraded.  The marker is the only thing that can see the difference.
    """
    db_path = tmp_path / "oldrule.duckdb"
    build([FIXTURES], db_path)
    connection = connect(db_path)
    try:
        connection.execute(f'DELETE FROM "{META_TABLE}" WHERE key = ?', [META_FORMAT_KEY])
    finally:
        connection.close()

    with pytest.raises(SchemaOutOfDate, match="delete the database file and build again"):
        build([FIXTURES], db_path)

    # Reading refuses the same way: sql and denials both open through
    # assert_readable, and a stale marker must not be half-upgraded either.
    connection = connect(db_path, read_only=True)
    try:
        with pytest.raises(SchemaOutOfDate, match="delete the database file and build again"):
            assert_readable(connection)
    finally:
        connection.close()


# ------------------------------------------------- parallel tool_use blocks


def write_parallel_session(path: Path, session_id: str, lines) -> None:
    """A transcript whose assistant lines can carry several ``tool_use`` blocks.

    *lines* is one list of ``(tool_use_id, tool_name, input, denied)`` per
    assistant line.  Every call on a line shares that line's ``seq``, exactly as
    parallel tool calls do in a real transcript, and each result arrives on a
    later user line -- after the whole batch was already issued.
    """
    records: list[str] = []
    clock = 0
    previous = None
    for index, calls in enumerate(lines):
        assistant_id = f"a{index}"
        records.append(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": assistant_id,
                    "parentUuid": previous,
                    "sessionId": session_id,
                    "timestamp": f"2026-08-10T12:00:{clock:02d}.000Z",
                    "cwd": "/home/dev/proj",
                    "permissionMode": "default",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": use_id,
                                "name": tool_name,
                                "input": tool_input,
                            }
                            for use_id, tool_name, tool_input, _ in calls
                        ],
                    },
                }
            )
        )
        clock += 1
        for use_id, _, _, denied in calls:
            result_id = f"r{index}_{use_id}"
            records.append(
                json.dumps(
                    {
                        "type": "user",
                        "uuid": result_id,
                        "parentUuid": assistant_id,
                        "sessionId": session_id,
                        "timestamp": f"2026-08-10T12:00:{clock:02d}.000Z",
                        "message": {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": use_id,
                                    "content": DENIAL_TEXT if denied else "fine",
                                    "is_error": denied,
                                }
                            ],
                        },
                    }
                )
            )
            clock += 1
            previous = result_id
    path.write_text("\n".join(records) + "\n", encoding="utf-8")


#: Denials that share their transcript line with a sibling call.  The sibling's
#: id sorts *after* the denial's, so pairing by (seq, tool_use_id) alone would
#: pick it -- and it was issued before the model could have seen the denial.
PARALLEL_SESSIONS = {
    # The sibling is a Read; the later line is a Bash, so the two candidates
    # even disagree about followup_kind.
    "pll1": [
        [
            ("toolu_pll1_a", "Bash", {"command": "rm -rf /"}, True),
            ("toolu_pll1_b", "Read", {"file_path": "/etc/hosts"}, False),
        ],
        [("toolu_pll1_c", "Bash", {"command": "ls -1"}, False)],
    ],
    # Same shape, but the denial's line is the session's last.
    "pll2": [
        [
            ("toolu_pll2_a", "Bash", {"command": "curl evil.example"}, True),
            ("toolu_pll2_b", "Read", {"file_path": "/tmp/notes"}, False),
        ],
    ],
    # The *later* line is the parallel one: tool_use_id still picks between its
    # blocks, which is the only tie the view has left to break.
    "pll3": [
        [("toolu_pll3_a", "Write", {"file_path": "/etc/hosts", "content": "no"}, True)],
        [
            ("toolu_pll3_z", "Glob", {"pattern": "**/*.py"}, False),
            ("toolu_pll3_b", "Grep", {"pattern": "TODO"}, False),
        ],
    ],
}


@pytest.fixture
def parallel_source(tmp_path: Path) -> Path:
    source = tmp_path / "parallel"
    source.mkdir()
    for session_id, lines in PARALLEL_SESSIONS.items():
        write_parallel_session(source / f"{session_id}.jsonl", session_id, lines)
    return source


@pytest.fixture
def parallels(parallel_source: Path, tmp_path: Path):
    build([parallel_source], tmp_path / "parallel.duckdb")
    connection = connect(tmp_path / "parallel.duckdb", read_only=True)
    yield connection
    connection.close()


def test_parallel_blocks_really_do_share_a_seq(parallels):
    """Otherwise the tests below would prove nothing about same-line siblings."""
    rows = parallels.execute(
        "SELECT seq, count(*) FROM tool_calls WHERE session_id = 'pll1' GROUP BY 1 ORDER BY 1"
    ).fetchall()
    assert [count for _, count in rows] == [2, 1]


def test_a_same_line_sibling_is_not_the_followup(parallels):
    """The sibling was issued before the model saw the denial, so it cannot be a reaction."""
    row = parallels.execute(
        "SELECT next_tool_name, next_input_summary, followup_kind "
        "FROM denial_followups WHERE session_id = 'pll1'"
    ).fetchone()
    assert row == ("Bash", "ls -1", "same-tool")


def test_a_denial_on_the_last_line_is_none_even_with_a_sibling(parallels):
    row = parallels.execute(
        "SELECT next_tool_name, next_input_summary, next_outcome, next_ts, gap_seconds, "
        "followup_kind FROM denial_followups WHERE session_id = 'pll2'"
    ).fetchone()
    assert row == (None, None, None, None, None, "none")


def test_tool_use_id_still_breaks_the_tie_within_the_later_line(parallels):
    row = parallels.execute(
        "SELECT next_tool_name, followup_kind FROM denial_followups WHERE session_id = 'pll3'"
    ).fetchone()
    assert row == ("Grep", "other-tool")


def expected_followups(connection: duckdb.DuckDBPyConnection) -> dict[tuple, tuple | None]:
    """What the view should say, worked out in Python rather than in SQL.

    For every denied call: the first call of the same session on a strictly
    later line, ties within that line broken by ``tool_use_id``; ``None`` when
    the session has no later line that called a tool.
    """
    calls = connection.execute(
        "SELECT session_id, seq, tool_use_id, tool_name, input_summary, outcome FROM tool_calls"
    ).fetchall()
    expected: dict[tuple, tuple | None] = {}
    for session_id, seq, _, _, _, outcome in calls:
        if outcome != "denied":
            continue
        later = [row for row in calls if row[0] == session_id and row[1] > seq]
        first = min(later, key=lambda row: (row[1], row[2]), default=None)
        expected[(session_id, seq)] = None if first is None else first[3:]
    return expected


def view_followups(connection: duckdb.DuckDBPyConnection) -> dict[tuple, tuple | None]:
    rows = connection.execute(
        "SELECT session_id, seq, next_tool_name, next_input_summary, next_outcome, followup_kind "
        "FROM denial_followups"
    ).fetchall()
    seen: dict[tuple, tuple | None] = {}
    for session_id, seq, next_tool_name, next_input_summary, next_outcome, kind in rows:
        assert kind in FOLLOWUP_KINDS
        if kind == "none":
            assert (next_tool_name, next_input_summary, next_outcome) == (None, None, None)
        seen[(session_id, seq)] = (
            None if kind == "none" else (next_tool_name, next_input_summary, next_outcome)
        )
    return seen


@pytest.mark.parametrize("corpus", ["built", "parallels", "followups"])
def test_the_followup_is_always_the_first_strictly_later_call(corpus, request):
    """The property itself, over every corpus the suite builds."""
    fixture = request.getfixturevalue(corpus)
    connection = fixture[1] if corpus == "built" else fixture
    expected = expected_followups(connection)
    assert expected  # the corpus contains denials at all
    assert view_followups(connection) == expected


# ---------------------------------------------------------------- recall_calls (Claude Code)


def write_recall_transcript(path: Path, session_id: str, turns) -> None:
    """A transcript alternating plain assistant text and tool calls with a custom result.

    *turns* is a list of ``("text", content)`` or
    ``("tool", tool_name, input, output_text)`` entries.  A tool turn becomes
    two lines -- the assistant's ``tool_use`` and the user's ``tool_result``
    -- so ``seq`` advances the way it does in a real transcript.
    """
    lines: list[str] = []
    clock = 0
    previous = None
    for index, turn in enumerate(turns):
        node_id = f"n{index}"
        if turn[0] == "text":
            _, content = turn
            lines.append(
                json.dumps(
                    {
                        "type": "assistant",
                        "uuid": node_id,
                        "parentUuid": previous,
                        "sessionId": session_id,
                        "timestamp": f"2026-08-11T00:00:{clock:02d}.000Z",
                        "message": {"role": "assistant", "content": content},
                    }
                )
            )
            clock += 1
            previous = node_id
        else:
            _, tool_name, tool_input, output_text = turn
            use_id = f"toolu_{index}"
            lines.append(
                json.dumps(
                    {
                        "type": "assistant",
                        "uuid": node_id,
                        "parentUuid": previous,
                        "sessionId": session_id,
                        "timestamp": f"2026-08-11T00:00:{clock:02d}.000Z",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": use_id,
                                    "name": tool_name,
                                    "input": tool_input,
                                }
                            ],
                        },
                    }
                )
            )
            clock += 1
            result_id = f"r{index}"
            lines.append(
                json.dumps(
                    {
                        "type": "user",
                        "uuid": result_id,
                        "parentUuid": node_id,
                        "sessionId": session_id,
                        "timestamp": f"2026-08-11T00:00:{clock:02d}.000Z",
                        "message": {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": use_id,
                                    "content": output_text,
                                    "is_error": False,
                                }
                            ],
                        },
                    }
                )
            )
            clock += 1
            previous = result_id
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


#: One session whose recall was clearly acted on, one whose recall was
#: clearly not -- the tokens are chosen distinctive so the assertions below
#: are unambiguous by construction.
RECALL_TURNS = {
    "recall-used": [
        ("text", "Let's check prior notes before editing."),
        (
            "tool",
            "mcp__kaiba__recall",
            {"query": "flaky retry"},
            "Retries must use anchored_backoff_v7 not naive sleep.",
        ),
        ("text", "Applying anchored_backoff_v7 as recalled."),
    ],
    "recall-unused": [
        (
            "tool",
            "mcp__kaiba__recall",
            {"query": "unrelated"},
            "Consider quarantine_flag_q2 for edge cases.",
        ),
        ("text", "Proceeding with the standard approach instead."),
    ],
}


@pytest.fixture
def recall_source(tmp_path: Path) -> Path:
    source = tmp_path / "recalls"
    source.mkdir()
    for session_id, turns in RECALL_TURNS.items():
        write_recall_transcript(source / f"{session_id}.jsonl", session_id, turns)
    return source


def test_recall_call_columns_match_the_dataclass(tmp_path: Path):
    build([FIXTURES], tmp_path / "cols.duckdb")
    connection = connect(tmp_path / "cols.duckdb", read_only=True)
    try:
        actual = [
            row[1] for row in connection.execute("PRAGMA table_info('recall_calls')").fetchall()
        ]
        assert actual == list(RECALL_CALL_COLUMNS)
    finally:
        connection.close()


def test_claude_recall_calls_are_extracted_with_followup_and_overlap(
    recall_source: Path, tmp_path: Path
):
    db_path = tmp_path / "recalls.duckdb"
    result = build([recall_source], db_path)
    assert result.n_recall_calls == 2

    connection = connect(db_path, read_only=True)
    try:
        rows = connection.execute(
            "SELECT session_id, source, query, output, overlap_count, overlap_tokens, "
            "followup_text FROM recall_calls ORDER BY session_id"
        ).fetchall()
    finally:
        connection.close()

    assert len(rows) == 2
    used = next(r for r in rows if r[0] == "recall-used")
    unused = next(r for r in rows if r[0] == "recall-unused")

    assert used[1] == "claude_code"
    assert used[2] == "flaky retry"
    assert used[3] == "Retries must use anchored_backoff_v7 not naive sleep."
    assert used[4] == 1
    assert json.loads(used[5]) == ["anchored_backoff_v7"]
    assert "anchored_backoff_v7" in used[6]

    assert unused[4] == 0
    assert json.loads(unused[5]) == []


def test_recall_followups_view_matches_the_table(recall_source: Path, tmp_path: Path):
    db_path = tmp_path / "view.duckdb"
    build([recall_source], db_path)
    connection = connect(db_path, read_only=True)
    try:
        table_rows = connection.execute(
            "SELECT recall_id, session_id, query FROM recall_calls ORDER BY recall_id"
        ).fetchall()
        view_rows = connection.execute(
            f'SELECT recall_id, session_id, query FROM "{RECALL_FOLLOWUPS_VIEW}" '
            "ORDER BY recall_id"
        ).fetchall()
    finally:
        connection.close()
    assert table_rows == view_rows
    assert table_rows  # not vacuously equal


def test_a_pending_recall_call_produces_no_row(tmp_path: Path):
    source = tmp_path / "pending"
    source.mkdir()
    write_recall_transcript(
        source / "pending.jsonl",
        "pending-session",
        [("tool", "mcp__kaiba__recall", {"query": "in flight"}, "irrelevant")],
    )
    # write_recall_transcript always writes a matching result; simulate "no
    # result yet" by truncating the file to just its first (tool_use) line.
    lines = (source / "pending.jsonl").read_text(encoding="utf-8").splitlines()
    (source / "pending.jsonl").write_text(lines[0] + "\n", encoding="utf-8")

    db_path = tmp_path / "pending.duckdb"
    result = build([source], db_path)
    assert result.n_recall_calls == 0


# ---------------------------------------------------------------- recall_calls (opencode)

OPENCODE_FIXTURE = FIXTURES / "opencode_events.ndjson"


def test_iter_opencode_sources_finds_ndjson_recursively(tmp_path: Path):
    nested = tmp_path / "jobs" / "job1"
    nested.mkdir(parents=True)
    (nested / "events.ndjson").write_text("", encoding="utf-8")
    (tmp_path / "ignore.jsonl").write_text("", encoding="utf-8")
    files, missing = iter_opencode_sources([tmp_path])
    assert [f.name for f in files] == ["events.ndjson"]
    assert missing == []


def test_a_directory_source_never_picks_up_ndjson_for_the_jsonl_path(tmp_path: Path):
    """The two source lists are independent: --source never sees *.ndjson.

    This is what keeps the shared ``tests/fixtures/`` directory -- which now
    also holds ``opencode_events.ndjson`` -- safe for every pre-existing test
    that builds from the whole ``FIXTURES`` directory: those calls never pass
    ``opencode_sources``, and this proves the plain ``*.jsonl`` scan cannot
    see the new file even if they did pass the same directory twice over.
    """
    (tmp_path / "events.ndjson").write_text("", encoding="utf-8")
    files, _ = iter_transcripts([tmp_path])
    assert files == []


def test_the_shared_fixtures_directory_build_is_unaffected_by_the_opencode_fixture(
    tmp_path: Path,
):
    """Sanity check for the design above: `built`'s pinned counts still hold."""
    result = build([FIXTURES], tmp_path / "unaffected.duckdb")
    assert result.n_files == 4
    assert result.n_recall_calls == 0


def test_opencode_recall_calls_are_extracted_with_followup_and_overlap(tmp_path: Path):
    db_path = tmp_path / "opencode.duckdb"
    result = build([], db_path, opencode_sources=[OPENCODE_FIXTURE])
    assert result.n_processed == 1
    assert result.n_recall_calls == 2

    connection = connect(db_path, read_only=True)
    try:
        rows = connection.execute(
            "SELECT session_id, source, query, output, overlap_count, overlap_tokens, "
            "followup_text FROM recall_calls ORDER BY session_id"
        ).fetchall()
    finally:
        connection.close()

    assert len(rows) == 2
    used = next(r for r in rows if r[0] == "ses_aaa")
    unused = next(r for r in rows if r[0] == "ses_bbb")

    assert used[1] == "opencode"
    assert used[2] == "denial pattern anchoring"
    assert used[3] == "Use anchored prefix matching for denial_pattern_x9 tokens."
    assert used[4] == 1
    assert json.loads(used[5]) == ["denial_pattern_x9"]
    assert "denial_pattern_x9" in used[6]

    assert unused[4] == 0
    assert json.loads(unused[5]) == []


def test_opencode_activity_text_is_bounded_at_assembly():
    """An oversized tool output is truncated in the per-activity component.

    Fix 2: each activity component is bounded at ``result_text_limit`` at assembly,
    so followup_text (built from those components) never carries a megabyte-sized
    raw output.  The recall call's own ``output`` is truncated separately and is
    not part of the followup text.
    """
    huge = "Z" * 10000
    recall = OpenCodeToolCall(
        call_id="recall_1",
        session_id="ses_x",
        file_path="/tmp/x",
        seq=1,
        ts=None,
        tool="kaiba_recall",
        input={"query": "q"},
        output="recall result",
    )
    big = OpenCodeToolCall(
        call_id="big_1",
        session_id="ses_x",
        file_path="/tmp/x",
        seq=2,
        ts=None,
        tool="bash",
        input={"command": "cat"},
        output=huge,
    )
    parsed = ParsedOpenCodeFile(
        file_path="/tmp/x",
        tool_calls=[recall, big],
        text_chunks=[],
        n_parse_errors=0,
    )
    rows = extract_from_opencode(parsed, result_text_limit=4000)
    assert len(rows) == 1
    followup = rows[0].followup_text
    assert followup is not None
    # The per-component bound keeps each activity string at result_text_limit.
    assert len(followup) <= 4000
    # The raw oversized output never reaches followup_text.
    assert huge not in followup


def test_opencode_build_is_incremental(tmp_path: Path):
    db_path = tmp_path / "opencode.duckdb"
    build([], db_path, opencode_sources=[OPENCODE_FIXTURE])
    again = build([], db_path, opencode_sources=[OPENCODE_FIXTURE])
    assert again.n_processed == 0
    assert again.n_skipped == 1

    connection = connect(db_path, read_only=True)
    try:
        assert scalar(connection, "SELECT count(*) FROM recall_calls") == 2
    finally:
        connection.close()


def test_a_mixed_build_ingests_both_formats_without_disturbing_the_other(tmp_path: Path):
    """Acceptance criterion 1: a jsonl source and an ndjson source in one build()."""
    db_path = tmp_path / "mixed.duckdb"
    result = build([FIXTURES], db_path, opencode_sources=[OPENCODE_FIXTURE])
    assert result.n_files == 5  # the 4 existing fixtures + the opencode one
    assert result.n_sessions == TOTAL_SESSIONS  # unaffected: opencode never touches `sessions`
    assert result.n_events == TOTAL_EVENTS
    assert result.n_tool_calls == TOTAL_TOOL_CALLS
    assert result.n_recall_calls == 2  # only from the opencode fixture

    connection = connect(db_path, read_only=True)
    try:
        assert scalar(connection, "SELECT count(*) FROM recall_calls") == 2
        assert scalar(connection, "SELECT count(*) FROM source_files") == 5
    finally:
        connection.close()


def test_the_default_ashiato_info_report_does_not_include_recall_calls(tmp_path: Path):
    """`recall_calls` participates in schema/incremental machinery but not this report."""
    db_path = tmp_path / "info.duckdb"
    build([], db_path, opencode_sources=[OPENCODE_FIXTURE])
    info = database_info(db_path)
    assert "recall_calls" not in info.table_counts
    assert set(info.table_counts) == set(INFO_TABLES)


def test_two_builds_of_the_same_opencode_bytes_give_the_same_rows(tmp_path: Path):
    first, second = tmp_path / "first.duckdb", tmp_path / "second.duckdb"
    build([], first, opencode_sources=[OPENCODE_FIXTURE])
    build([], second, opencode_sources=[OPENCODE_FIXTURE])
    order = "recall_id"
    assert dump(first, "recall_calls", order) == dump(second, "recall_calls", order)


def test_a_database_missing_the_recall_calls_table_entirely_is_refused_gracefully(
    tmp_path: Path,
):
    """A pre-issue-10 database has no ``recall_calls`` table at all, not just a missing column."""
    db_path = tmp_path / "pre10.duckdb"
    build([FIXTURES], db_path)
    connection = connect(db_path)
    try:
        connection.execute('DROP VIEW "recall_followups"')
        connection.execute('DROP TABLE "recall_calls"')
        connection.execute(f'DELETE FROM "{META_TABLE}" WHERE key = ?', [META_FORMAT_KEY])
    finally:
        connection.close()

    with pytest.raises(SchemaOutOfDate, match="recall_calls"):
        build([FIXTURES], db_path)

    connection = connect(db_path, read_only=True)
    try:
        with pytest.raises(SchemaOutOfDate, match="delete the database file and build again"):
            assert_readable(connection)
    finally:
        connection.close()


def test_recalls_view_missing_alone_also_names_the_fix(tmp_path: Path):
    db_path = tmp_path / "noview.duckdb"
    build([], db_path, opencode_sources=[OPENCODE_FIXTURE])
    connection = connect(db_path)
    try:
        connection.execute(f'DROP VIEW "{RECALL_FOLLOWUPS_VIEW}"')
    finally:
        connection.close()
    connection = connect(db_path, read_only=True)
    try:
        with pytest.raises(SchemaOutOfDate, match="recall_followups"):
            assert_readable(connection)
    finally:
        connection.close()


# ---------------------------------------------------------------- recall_calls (cursor)

CURSOR_TRANSCRIPT_LINES = [
    {
        "role": "user",
        "message": {
            "content": [
                {"type": "text", "text": "<user_query>\nWhat about denial_pattern_x9?\n</user_query>"}
            ]
        },
    },
    {
        "role": "assistant",
        "message": {
            "content": [
                {"type": "text", "text": "Let me check kaiba first."},
                {
                    "type": "tool_use",
                    "name": "CallMcpTool",
                    "input": {
                        "server": "kaiba",
                        "toolName": "recall",
                        "arguments": {"query": "denial_pattern_x9", "top_k": 10},
                    },
                },
            ]
        },
    },
    {"type": "turn_ended", "status": "success"},
]


def _write_cursor_transcript(path: Path, lines: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")


def _make_kaiba_recalls_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE recalls (id INTEGER PRIMARY KEY, created_at TEXT, agent TEXT, "
            "query TEXT, top_k INTEGER, matches TEXT, mu REAL, sd REAL, floor_z REAL, "
            "below_floor INTEGER)"
        )
        connection.execute(
            "CREATE TABLE conclusions (id INTEGER PRIMARY KEY, content TEXT, author TEXT, "
            "created_at TEXT, embedding TEXT, embedding_model TEXT, retired_at TEXT)"
        )
        connection.execute(
            "INSERT INTO recalls (created_at, agent, query, matches) VALUES "
            "('2026-08-20T10:00:00Z', 'cursor', 'denial_pattern_x9', ?)",
            [json.dumps([{"id": 1, "score": 0.9}])],
        )
        connection.execute(
            "INSERT INTO conclusions (id, content) VALUES "
            "(1, 'Use anchored prefix matching for denial_pattern_x9 tokens.')"
        )
        connection.commit()
    finally:
        connection.close()


def test_iter_cursor_sources_finds_jsonl_recursively(tmp_path: Path):
    nested = tmp_path / "agent-transcripts" / "abc123"
    nested.mkdir(parents=True)
    (nested / "abc123.jsonl").write_text("", encoding="utf-8")
    files, missing = iter_cursor_sources([tmp_path])
    assert [f.name for f in files] == ["abc123.jsonl"]
    assert missing == []


def test_cursor_sources_default_to_no_kaiba_lookup_and_build_is_unaffected(tmp_path: Path):
    """Omitting --cursor-source (empty cursor_sources) changes nothing about a plain build."""
    result = build([FIXTURES], tmp_path / "unaffected.duckdb")
    assert result.n_recall_calls == 0
    assert result.kaiba_db_unavailable is None


def test_cursor_recall_calls_are_extracted_with_kaiba_join(tmp_path: Path):
    transcript_dir = tmp_path / "cursor"
    transcript_dir.mkdir()
    _write_cursor_transcript(transcript_dir / "sess1.jsonl", CURSOR_TRANSCRIPT_LINES)
    kaiba_path = tmp_path / "kaiba.db"
    _make_kaiba_recalls_db(kaiba_path)

    db_path = tmp_path / "cursor.duckdb"
    result = build([], db_path, cursor_sources=[transcript_dir], kaiba_db_path=kaiba_path)
    assert result.n_processed == 1
    assert result.n_recall_calls == 1
    assert result.kaiba_db_unavailable is None

    connection = connect(db_path, read_only=True)
    try:
        row = connection.execute(
            "SELECT source, session_id, query, output, ts FROM recall_calls"
        ).fetchone()
    finally:
        connection.close()
    assert row[0] == "cursor"
    assert row[1] == "sess1"
    assert row[2] == "denial_pattern_x9"
    assert row[3] == "Use anchored prefix matching for denial_pattern_x9 tokens."
    assert row[4] is not None


def test_cursor_build_is_incremental(tmp_path: Path):
    transcript_dir = tmp_path / "cursor"
    transcript_dir.mkdir()
    _write_cursor_transcript(transcript_dir / "sess1.jsonl", CURSOR_TRANSCRIPT_LINES)
    kaiba_path = tmp_path / "kaiba.db"
    _make_kaiba_recalls_db(kaiba_path)

    db_path = tmp_path / "cursor.duckdb"
    build([], db_path, cursor_sources=[transcript_dir], kaiba_db_path=kaiba_path)
    again = build([], db_path, cursor_sources=[transcript_dir], kaiba_db_path=kaiba_path)
    assert again.n_processed == 0
    assert again.n_skipped == 1

    connection = connect(db_path, read_only=True)
    try:
        assert scalar(connection, "SELECT count(*) FROM recall_calls") == 1
    finally:
        connection.close()


def test_a_missing_kaiba_db_does_not_fail_the_build(tmp_path: Path):
    transcript_dir = tmp_path / "cursor"
    transcript_dir.mkdir()
    _write_cursor_transcript(transcript_dir / "sess1.jsonl", CURSOR_TRANSCRIPT_LINES)

    db_path = tmp_path / "cursor.duckdb"
    missing_kaiba = tmp_path / "nowhere" / "kaiba.db"
    result = build([], db_path, cursor_sources=[transcript_dir], kaiba_db_path=missing_kaiba)
    assert result.n_processed == 1
    assert result.n_recall_calls == 1
    assert result.kaiba_db_unavailable == str(missing_kaiba)

    connection = connect(db_path, read_only=True)
    try:
        row = connection.execute("SELECT output, ts FROM recall_calls").fetchone()
    finally:
        connection.close()
    assert row == (None, None)


def test_cursor_never_touches_sessions_events_or_tool_calls(tmp_path: Path):
    transcript_dir = tmp_path / "cursor"
    transcript_dir.mkdir()
    _write_cursor_transcript(transcript_dir / "sess1.jsonl", CURSOR_TRANSCRIPT_LINES)
    kaiba_path = tmp_path / "kaiba.db"
    _make_kaiba_recalls_db(kaiba_path)

    db_path = tmp_path / "cursor.duckdb"
    result = build([], db_path, cursor_sources=[transcript_dir], kaiba_db_path=kaiba_path)
    assert result.n_sessions == 0
    assert result.n_events == 0
    assert result.n_tool_calls == 0

    connection = connect(db_path, read_only=True)
    try:
        assert scalar(connection, "SELECT count(*) FROM sessions") == 0
        assert scalar(connection, "SELECT count(*) FROM events") == 0
        assert scalar(connection, "SELECT count(*) FROM tool_calls") == 0
    finally:
        connection.close()


def _single_cursor_recall_line(query: str) -> dict:
    return {
        "role": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "name": "CallMcpTool",
                    "input": {"server": "kaiba", "toolName": "recall", "arguments": {"query": query}},
                }
            ]
        },
    }


def _make_kaiba_recalls_db_with_two_rows_for_one_query(path: Path, query: str) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE recalls (id INTEGER PRIMARY KEY, created_at TEXT, agent TEXT, "
            "query TEXT, top_k INTEGER, matches TEXT, mu REAL, sd REAL, floor_z REAL, "
            "below_floor INTEGER)"
        )
        connection.execute(
            "CREATE TABLE conclusions (id INTEGER PRIMARY KEY, content TEXT, author TEXT, "
            "created_at TEXT, embedding TEXT, embedding_model TEXT, retired_at TEXT)"
        )
        connection.executemany(
            "INSERT INTO recalls (created_at, agent, query, matches) VALUES (?, 'cursor', ?, ?)",
            [
                ("2026-08-20T10:00:00Z", query, json.dumps([{"id": 1, "score": 0.9}])),
                ("2026-08-20T11:00:00Z", query, json.dumps([{"id": 2, "score": 0.9}])),
            ],
        )
        connection.executemany(
            "INSERT INTO conclusions (id, content) VALUES (?, ?)",
            [(1, "first ledger row content"), (2, "second ledger row content")],
        )
        connection.commit()
    finally:
        connection.close()


def test_two_cursor_files_sharing_a_query_both_pair_with_the_first_ledger_row(
    tmp_path: Path,
):
    """Occurrences are counted per file, not across the build (ashiato#20 rework).

    Two different Cursor session files each issue the identical query text once.
    Since each file's occurrence counter is its own, both independently compute
    occurrence index 0 and pair with the *same* first ledger row -- a documented
    limitation, not something the build tries to disambiguate across files.
    """
    transcript_dir = tmp_path / "cursor"
    transcript_dir.mkdir()
    (transcript_dir / "session_a.jsonl").write_text(
        json.dumps(_single_cursor_recall_line("shared query")) + "\n", encoding="utf-8"
    )
    (transcript_dir / "session_b.jsonl").write_text(
        json.dumps(_single_cursor_recall_line("shared query")) + "\n", encoding="utf-8"
    )
    kaiba_path = tmp_path / "kaiba.db"
    _make_kaiba_recalls_db_with_two_rows_for_one_query(kaiba_path, "shared query")

    db_path = tmp_path / "shared_query.duckdb"
    result = build([], db_path, cursor_sources=[transcript_dir], kaiba_db_path=kaiba_path)
    assert result.n_processed == 2
    assert result.n_recall_calls == 2

    connection = connect(db_path, read_only=True)
    try:
        rows = connection.execute(
            "SELECT session_id, output FROM recall_calls ORDER BY session_id"
        ).fetchall()
    finally:
        connection.close()
    assert rows == [
        ("session_a", "first ledger row content"),
        ("session_b", "first ledger row content"),
    ]


def test_cli_build_with_cursor_source_and_kaiba_db(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    transcript_dir = tmp_path / "cursor"
    transcript_dir.mkdir()
    _write_cursor_transcript(transcript_dir / "sess1.jsonl", CURSOR_TRANSCRIPT_LINES)
    kaiba_path = tmp_path / "kaiba.db"
    _make_kaiba_recalls_db(kaiba_path)
    db_path = tmp_path / "cli_cursor.duckdb"

    assert (
        main(
            [
                "build",
                "--cursor-source",
                str(transcript_dir),
                "--kaiba-db",
                str(kaiba_path),
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "1 recall calls" in out


def test_cli_build_reports_a_missing_kaiba_db(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    transcript_dir = tmp_path / "cursor"
    transcript_dir.mkdir()
    _write_cursor_transcript(transcript_dir / "sess1.jsonl", CURSOR_TRANSCRIPT_LINES)
    missing_kaiba = tmp_path / "nowhere" / "kaiba.db"
    db_path = tmp_path / "cli_missing_kaiba.duckdb"

    assert (
        main(
            [
                "build",
                "--cursor-source",
                str(transcript_dir),
                "--kaiba-db",
                str(missing_kaiba),
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    err = capsys.readouterr().err
    assert "no kaiba db at" in err
