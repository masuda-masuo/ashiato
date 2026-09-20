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
from ashiato.recall import (
    RECALL_CALL_COLUMNS,
    _Activity,
    _is_distinctive,
    _overlap,
    extract_from_opencode,
)
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


# ---------------------------------------------------------------- source column


def test_source_column_labels_claude_rows_and_keeps_columns_in_place(built):
    """Every Claude row carries 'claude_code', and fields *after* `source` still hold their value.

    ``source`` sits right after ``file_path`` in all three tables, so a row
    builder that returns one element too few -- or in the wrong order -- would
    silently shift every later column instead of raising.  Reading the rows
    back by column name and asserting a field positioned after ``source``
    catches exactly that.
    """
    _, connection = built
    rows = connection.execute(
        "SELECT session_id, source, project_dir, cwd FROM sessions ORDER BY session_id"
    ).fetchall()
    assert len(rows) == TOTAL_SESSIONS
    assert all(row[1] == "claude_code" for row in rows)
    assert all(row[2] == "fixtures" for row in rows)  # project_dir, after source

    row = connection.execute(
        "SELECT event_id, source, seq, type, role FROM events WHERE event_id = 'u1'"
    ).fetchone()
    assert row == ("u1", "claude_code", 1, "user", "user")  # seq/type/role, after source

    row = connection.execute(
        "SELECT tool_use_id, source, seq, tool_name, outcome FROM tool_calls "
        "WHERE tool_use_id = 'toolu_ok_1'"
    ).fetchone()
    # seq/tool_name/outcome, after source
    assert row == ("toolu_ok_1", "claude_code", 2, "Bash", "ok")


def test_source_column_labels_codex_rows_and_keeps_columns_in_place(tmp_path: Path):
    """Every Codex row carries 'codex', and fields *after* `source` still hold their value."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session_with_timeline(
        codex_dir / "sess.jsonl",
        "codex-src-1",
        tool_calls=[
            {"type": "CommandExecution", "id": "exec-1", "command": "pwd", "stdout": "/work"},
        ],
        text_chunks=["Hello from Codex"],
        timestamps=[
            "2026-09-05T10:00:00Z",
            "2026-09-05T10:00:01Z",
            "2026-09-05T10:00:02Z",
        ],
    )
    db_path = tmp_path / "codex_source.duckdb"
    build([], db_path, codex_sources=[codex_dir])
    connection = connect(db_path, read_only=True)
    try:
        row = connection.execute(
            "SELECT session_id, source, started_at, n_tool_calls FROM sessions"
        ).fetchone()
        assert row is not None
        assert row[0] == "codex-src-1"
        assert row[1] == "codex"
        assert row[2] is not None  # started_at, after source
        assert row[3] == 1  # n_tool_calls, after source

        row = connection.execute(
            "SELECT event_id, source, seq, role, text FROM events"
        ).fetchone()
        assert row is not None
        assert row[0].startswith("codex:text:")
        assert row[1] == "codex"
        assert row[2] == 3  # seq, after source
        assert row[3] == "unknown"  # role, after source (item_completed path carries none)
        assert row[4] == "Hello from Codex"  # text, after source

        row = connection.execute(
            "SELECT tool_use_id, source, seq, tool_name, outcome FROM tool_calls"
        ).fetchone()
        assert row is not None
        assert row[0] == "exec-1"
        assert row[1] == "codex"
        assert row[2] == 2  # seq, after source
        assert row[3] == "Bash"  # tool_name, after source
        assert row[4] == "ok"  # outcome, after source
    finally:
        connection.close()


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


# ---------------------------------------------------------------- opencode into sessions/events/tool_calls


def test_opencode_transcript_populates_sessions_events_tool_calls(tmp_path: Path):
    """Acceptance criterion 1: rows in all three main tables, source = 'opencode'."""
    db_path = tmp_path / "opencode_main.duckdb"
    result = build([], db_path, opencode_sources=[OPENCODE_FIXTURE])
    assert result.n_sessions == 3
    assert result.n_events == 3
    assert result.n_tool_calls == 3

    connection = connect(db_path, read_only=True)
    try:
        sessions = connection.execute(
            "SELECT session_id, source, started_at, ended_at, n_events, n_tool_calls "
            "FROM sessions ORDER BY session_id"
        ).fetchall()
        # One row per distinct session id; started_at/ended_at are the min/max
        # of the tool-call timestamps the file carries for that session.
        assert sessions == [
            ("ses_aaa", "opencode", datetime(1970, 1, 1, 0, 0, 1), datetime(1970, 1, 1, 0, 0, 1), 2, 1),
            ("ses_bbb", "opencode", datetime(1970, 1, 1, 0, 0, 2), datetime(1970, 1, 1, 0, 0, 2), 1, 1),
            ("ses_ddd", "opencode", datetime(1970, 1, 1, 0, 0, 4), datetime(1970, 1, 1, 0, 0, 4), 0, 1),
        ]

        events = connection.execute(
            "SELECT event_id, session_id, source, seq, ts, type, text FROM events ORDER BY seq"
        ).fetchall()
        assert len(events) == 3
        assert [row[1] for row in events] == ["ses_aaa", "ses_aaa", "ses_bbb"]
        assert all(row[2] == "opencode" for row in events)
        # ts is NULL: opencode text parts carry no timestamp at all.
        assert all(row[4] is None and row[5] == "text" for row in events)
        assert all(row[0].startswith("opencode:text:") for row in events)
        assert [row[6] for row in events] == [
            "Let me check kaiba for prior guidance before editing anything.",
            "Applying denial_pattern_x9 as documented in the recall output.",
            "Proceeding with the standard approach and ignoring that suggestion.",
        ]

        calls = connection.execute(
            "SELECT tool_use_id, session_id, source, seq, ts, tool_name, outcome, is_error, "
            "call_event_id, result_event_id, input FROM tool_calls ORDER BY seq"
        ).fetchall()
        assert len(calls) == 3
        assert [row[1] for row in calls] == ["ses_aaa", "ses_bbb", "ses_ddd"]
        assert all(row[2] == "opencode" for row in calls)
        assert [row[5] for row in calls] == ["kaiba_recall", "kaiba_recall", "bash"]
        assert all(row[6] == "ok" and row[7] is False for row in calls)
        # opencode does not link its tool parts to events rows.
        assert all(row[8] is None and row[9] is None for row in calls)
        assert calls[0][10] == '{"query": "denial pattern anchoring"}'
    finally:
        connection.close()


def test_opencode_file_with_two_distinct_session_ids_produces_two_session_rows(
    tmp_path: Path,
):
    """Acceptance criterion 2: one sessions row per distinct session id in a file."""
    events = tmp_path / "two_sessions.ndjson"
    events.write_text(
        "\n".join(
            [
                '{"id":"a","type":"message.part.updated","properties":{"sessionID":"s1","part":{"id":"p1","sessionID":"s1","type":"tool","tool":"bash","callID":"c1","state":{"status":"completed","input":{"command":"echo a"},"output":"a","time":{"start":1000,"end":1100}}},"time":1100}}',
                '{"id":"b","type":"message.part.updated","properties":{"sessionID":"s2","part":{"id":"p2","sessionID":"s2","type":"tool","tool":"bash","callID":"c2","state":{"status":"completed","input":{"command":"echo b"},"output":"b","time":{"start":2000,"end":2100}}},"time":2100}}',
                '{"id":"c","type":"message.part.updated","properties":{"sessionID":"s1","part":{"id":"p3","sessionID":"s1","type":"text","text":"from s1"}}}',
            ]
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "two_sessions.duckdb"
    build([], db_path, opencode_sources=[events])
    connection = connect(db_path, read_only=True)
    try:
        rows = connection.execute(
            "SELECT session_id, n_events, n_tool_calls, started_at, ended_at "
            "FROM sessions ORDER BY session_id"
        ).fetchall()
        assert rows == [
            ("s1", 1, 1, datetime(1970, 1, 1, 0, 0, 1), datetime(1970, 1, 1, 0, 0, 1)),
            ("s2", 0, 1, datetime(1970, 1, 1, 0, 0, 2), datetime(1970, 1, 1, 0, 0, 2)),
        ]
    finally:
        connection.close()


def test_opencode_parts_without_a_session_id_produce_no_session_row_and_do_not_raise(
    tmp_path: Path,
):
    """Acceptance criterion 2: no session id -> no session row, and no exception."""
    events = tmp_path / "no_session.ndjson"
    events.write_text(
        "\n".join(
            [
                '{"id":"a","type":"message.part.updated","properties":{"part":{"id":"p1","type":"tool","tool":"bash","callID":"c1","state":{"status":"completed","input":{"command":"echo x"},"output":"x","time":{"start":1000,"end":1100}}},"time":1100}}',
                '{"id":"b","type":"message.part.updated","properties":{"part":{"id":"p2","type":"text","text":"hello"}}}',
            ]
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "no_session.duckdb"
    build([], db_path, opencode_sources=[events])  # must not raise
    connection = connect(db_path, read_only=True)
    try:
        assert scalar(connection, "SELECT count(*) FROM sessions") == 0
        assert scalar(connection, "SELECT count(*) FROM events") == 1
        assert scalar(connection, "SELECT count(*) FROM tool_calls") == 1
        assert (
            scalar(connection, "SELECT count(*) FROM tool_calls WHERE session_id IS NULL")
            == 1
        )
        assert (
            scalar(connection, "SELECT count(*) FROM events WHERE session_id IS NULL")
            == 1
        )
    finally:
        connection.close()


def test_opencode_rows_null_every_column_without_an_equivalent(tmp_path: Path):
    """Acceptance criterion 3: columns with no opencode equivalent are NULL, never a placeholder."""
    db_path = tmp_path / "opencode_nulls.duckdb"
    build([], db_path, opencode_sources=[OPENCODE_FIXTURE])
    connection = connect(db_path, read_only=True)
    try:
        session = connection.execute(
            "SELECT project_dir, cwd, git_branch, cc_version, entrypoint, input_tokens, "
            "output_tokens, cache_read_tokens, cache_creation_tokens "
            "FROM sessions WHERE session_id = 'ses_aaa'"
        ).fetchone()
        assert session == (None,) * 9

        event = connection.execute(
            "SELECT ts, role, parent_uuid, depth, is_sidechain, is_meta, permission_mode, "
            "effort, request_id, message_id, model, cwd, git_branch "
            "FROM events WHERE session_id = 'ses_aaa' ORDER BY seq LIMIT 1"
        ).fetchone()
        assert event == (None,) * 13

        call = connection.execute(
            "SELECT call_event_id, result_event_id, duration_ms, permission_mode, cwd, "
            "is_sidechain, parent_tool_use_id, mcp_server "
            "FROM tool_calls WHERE session_id = 'ses_ddd'"
        ).fetchone()
        # duration_ms is no longer a "no equivalent" column: the part's `time`
        # object populates it (4000 -> 4100 is 100 ms) since issue #83.
        assert call == (None, None, 100, None, None, None, None, None)
    finally:
        connection.close()


def test_opencode_source_files_counts_match_inserted_rows(tmp_path: Path):
    """Acceptance criterion 5: source_files counts equal the rows actually inserted."""
    db_path = tmp_path / "opencode_sf.duckdb"
    build([], db_path, opencode_sources=[OPENCODE_FIXTURE])
    connection = connect(db_path, read_only=True)
    try:
        n_events, n_tool_calls = connection.execute(
            "SELECT n_events, n_tool_calls FROM source_files"
        ).fetchone()
        assert n_events == scalar(connection, "SELECT count(*) FROM events")
        assert n_tool_calls == scalar(connection, "SELECT count(*) FROM tool_calls")
        assert (n_events, n_tool_calls) == (3, 3)
    finally:
        connection.close()


def test_opencode_build_leaves_recall_calls_unchanged(tmp_path: Path):
    """Acceptance criterion 6: opencode recall_calls are exactly what they were before #83.

    The recall extraction is untouched by this change; pinning every recall row
    here means an accidental perturbation of the recall path fails the suite
    rather than hiding in prose.
    """
    db_path = tmp_path / "opencode_recall.duckdb"
    build([], db_path, opencode_sources=[OPENCODE_FIXTURE])
    file_key = str(OPENCODE_FIXTURE.resolve())
    connection = connect(db_path, read_only=True)
    try:
        rows = connection.execute(
            "SELECT recall_id, session_id, source, seq, ts, call_id, query, output, "
            "output_truncated, followup_text, followup_truncated, overlap_count, "
            "overlap_tokens FROM recall_calls ORDER BY recall_id"
        ).fetchall()
        assert rows == [
            (
                f"{file_key}:call_2",
                "ses_aaa",
                "opencode",
                3,
                datetime(1970, 1, 1, 0, 0, 1),
                "call_2",
                "denial pattern anchoring",
                "Use anchored prefix matching for denial_pattern_x9 tokens.",
                False,
                "Applying denial_pattern_x9 as documented in the recall output.",
                False,
                1,
                '["denial_pattern_x9"]',
            ),
            (
                f"{file_key}:call_5",
                "ses_bbb",
                "opencode",
                5,
                datetime(1970, 1, 1, 0, 0, 2),
                "call_5",
                "unrelated topic",
                "Consider orphaned_snippet_zz for edge cases.",
                False,
                "Proceeding with the standard approach and ignoring that suggestion.",
                False,
                0,
                "[]",
            ),
        ]
    finally:
        connection.close()


def test_opencode_completed_call_with_empty_output_is_not_pending(tmp_path: Path):
    """A completed opencode tool part with no output must not read as 'pending'.

    The parser only emits parts whose state is 'completed', so the call is
    terminal; 'pending' would claim it was interrupted.  It classifies as
    'ok' with NULL result_text instead.
    """
    events = tmp_path / "empty_output.ndjson"
    events.write_text(
        '{"id":"a","type":"message.part.updated","properties":{"sessionID":"s1","part":{"id":"p1","sessionID":"s1","type":"tool","tool":"bash","callID":"c1","state":{"status":"completed","input":{"command":"true"},"time":{"start":1000,"end":1100}}},"time":1100}}',
        encoding="utf-8",
    )
    db_path = tmp_path / "empty_output.duckdb"
    build([], db_path, opencode_sources=[events])
    connection = connect(db_path, read_only=True)
    try:
        outcome, result_text, is_error = connection.execute(
            "SELECT outcome, result_text, is_error FROM tool_calls WHERE tool_use_id = 'c1'"
        ).fetchone()
        assert outcome == "ok"
        assert result_text is None
        assert is_error is False
    finally:
        connection.close()


def test_opencode_error_state_produces_an_error_row(tmp_path: Path):
    """Acceptance criterion: a part with state.status == 'error' lands as a
    tool_calls row with is_error True, outcome 'error', the failure message in
    result_text, and duration_ms from the part's time object."""
    events = tmp_path / "error_state.ndjson"
    events.write_text(
        '{"id":"a","type":"message.part.updated","properties":{"sessionID":"s1","part":{"id":"p1","sessionID":"s1","type":"tool","tool":"bash","callID":"c_err","state":{"status":"error","input":{"command":"exit 1"},"error":"exit code 1","time":{"start":1000,"end":1500}}},"time":1500}}',
        encoding="utf-8",
    )
    db_path = tmp_path / "opencode_error.duckdb"
    build([], db_path, opencode_sources=[events])
    connection = connect(db_path, read_only=True)
    try:
        row = connection.execute(
            "SELECT outcome, is_error, result_text, result_truncated, duration_ms, ts "
            "FROM tool_calls WHERE tool_use_id = 'c_err'"
        ).fetchone()
        assert row is not None
        outcome, is_error, result_text, result_truncated, duration_ms, ts = row
        assert outcome == "error"
        assert is_error is True
        assert result_text == "exit code 1"
        assert result_truncated is False
        assert duration_ms == 500
        assert ts == datetime(1970, 1, 1, 0, 0, 1)
    finally:
        connection.close()


def test_opencode_failed_call_with_empty_output_is_error_not_pending(tmp_path: Path):
    """Acceptance criterion: a failed part with no output classifies as 'error',
    never 'pending' -- has_result is forced by is_error the way the Codex path
    forces it."""
    events = tmp_path / "error_empty.ndjson"
    events.write_text(
        '{"id":"a","type":"message.part.updated","properties":{"sessionID":"s1","part":{"id":"p1","sessionID":"s1","type":"tool","tool":"bash","callID":"c_e","state":{"status":"error","input":{"command":"exit 2"},"time":{"start":1000,"end":1000}}},"time":1000}}',
        encoding="utf-8",
    )
    db_path = tmp_path / "opencode_error_empty.duckdb"
    build([], db_path, opencode_sources=[events])
    connection = connect(db_path, read_only=True)
    try:
        outcome, is_error, result_text = connection.execute(
            "SELECT outcome, is_error, result_text FROM tool_calls WHERE tool_use_id = 'c_e'"
        ).fetchone()
        assert outcome == "error"
        assert is_error is True
        assert result_text is None  # no output and no error message
    finally:
        connection.close()


def test_opencode_completed_call_still_ok_with_duration(tmp_path: Path):
    """Acceptance criterion: a completed part still produces is_error False /
    outcome 'ok' exactly as before -- and now carries duration_ms."""
    events = tmp_path / "still_ok.ndjson"
    events.write_text(
        '{"id":"a","type":"message.part.updated","properties":{"sessionID":"s1","part":{"id":"p1","sessionID":"s1","type":"tool","tool":"bash","callID":"c_ok","state":{"status":"completed","input":{"command":"true"},"output":"done","time":{"start":1000,"end":1200}}},"time":1200}}',
        encoding="utf-8",
    )
    db_path = tmp_path / "opencode_still_ok.duckdb"
    build([], db_path, opencode_sources=[events])
    connection = connect(db_path, read_only=True)
    try:
        outcome, is_error, duration_ms = connection.execute(
            "SELECT outcome, is_error, duration_ms FROM tool_calls WHERE tool_use_id = 'c_ok'"
        ).fetchone()
        assert outcome == "ok"
        assert is_error is False
        assert duration_ms == 200
    finally:
        connection.close()


def test_opencode_pending_and_running_parts_produce_no_row(tmp_path: Path):
    """Acceptance criterion: pending/running parts still produce no tool_calls
    row -- unfinished is not failed, and is not ingested."""
    events = tmp_path / "unfinished.ndjson"
    events.write_text(
        "\n".join(
            [
                '{"id":"a","type":"message.part.updated","properties":{"sessionID":"s1","part":{"id":"p1","sessionID":"s1","type":"tool","tool":"bash","callID":"c_p","state":{"status":"pending","input":{"command":"wait"}}}}}',
                '{"id":"b","type":"message.part.updated","properties":{"sessionID":"s1","part":{"id":"p2","sessionID":"s1","type":"tool","tool":"bash","callID":"c_r","state":{"status":"running","input":{"command":"sleep"}}}}}',
            ]
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "opencode_unfinished.duckdb"
    build([], db_path, opencode_sources=[events])
    connection = connect(db_path, read_only=True)
    try:
        assert scalar(connection, "SELECT count(*) FROM tool_calls") == 0
        assert scalar(connection, "SELECT count(*) FROM sessions") == 0
    finally:
        connection.close()


def test_opencode_bulk_and_row_by_row_inserts_produce_identical_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The opencode insert path survives the bulk route with identical rows."""
    bulk_db = tmp_path / "oc_bulk.duckdb"
    row_db = tmp_path / "oc_row.duckdb"
    monkeypatch.setattr(build_module, "BULK_INSERT_MIN_ROWS", 1)
    assert build([], bulk_db, opencode_sources=[OPENCODE_FIXTURE]).n_bulk_fallbacks == 0
    monkeypatch.setattr(build_module, "BULK_INSERT_MIN_ROWS", 10**9)
    build([], row_db, opencode_sources=[OPENCODE_FIXTURE])
    # source_files is excluded the same way the Claude bulk test excludes it:
    # its built_at is a wall-clock stamp that differs between two builds.
    for table, order in (
        ("sessions", "session_id"),
        ("events", "seq"),
        ("tool_calls", "seq"),
    ):
        assert dump(bulk_db, table, order) == dump(row_db, table, order), table


def test_two_builds_of_the_same_opencode_bytes_give_the_same_main_table_rows(
    tmp_path: Path,
):
    """Determinism extends to the tables issue #83 now fills."""
    first, second = tmp_path / "oc_first.duckdb", tmp_path / "oc_second.duckdb"
    build([], first, opencode_sources=[OPENCODE_FIXTURE])
    build([], second, opencode_sources=[OPENCODE_FIXTURE])
    for table, order in (
        ("sessions", "session_id"),
        ("events", "seq"),
        ("tool_calls", "seq"),
    ):
        assert dump(first, table, order) == dump(second, table, order), table


# ---------------------------------------------------------------- overlap distinctive-shape rule


def test_is_distinctive_classifies_tokens_by_shape():
    # Plain English words: no shape char, no inner capital -> not distinctive.
    assert not _is_distinctive("different")
    assert not _is_distinctive("green")
    assert not _is_distinctive("PASS")
    assert not _is_distinctive("bytes")
    # Bare ISO date / date-hour prefix -> excluded even though it has digits/dashes.
    assert not _is_distinctive("2026-08-16")
    assert not _is_distinctive("2026-08-16T04")
    assert not _is_distinctive("2026-08-16T04:15Z")
    # All-digit token -> excluded.
    assert not _is_distinctive("2101")
    # Shape char present -> distinctive.
    assert _is_distinctive("741d50b")
    assert _is_distinctive("#852")
    assert _is_distinctive("21ms")
    assert _is_distinctive("src/ashiato/recall.py")
    assert _is_distinctive("kusabi#274")
    assert _is_distinctive("--container")
    assert _is_distinctive("chain-msvthdq26fdc")
    # camelCase / PascalCase inner capital -> distinctive even with no shape char.
    assert _is_distinctive("deriveDisposition")
    assert _is_distinctive("FastMCP")


def test_overlap_excludes_ordinary_words_dates_and_all_digits():
    # Recall output and the followup both contain only ordinary English and
    # a bare date: nothing counts.
    output = "The green test was different after seeing 2026-08-16 and 2026-08-16T04."
    prefix: list[_Activity] = []
    suffix = [_Activity(2, "The green test was different after seeing 2026-08-16T04 later.")]
    tokens, count = _overlap(output, prefix, suffix)
    assert count == 0
    assert tokens == []
    # When the shared token is all digits it is also excluded.
    output2 = "Count was 2101 this time."
    suffix2 = [_Activity(2, "The count 2101 appeared again.")]
    assert _overlap(output2, prefix, suffix2)[1] == 0


def test_overlap_counts_distinctive_shared_tokens():
    output = (
        "Use deriveDisposition with src/ashiato/recall.py and kusabi#274, "
        "--container and hash 741d50b here"
    )
    prefix: list[_Activity] = []
    suffix = [
        _Activity(
            2,
            "Then deriveDisposition ran src/ashiato/recall.py via kusabi#274 with "
            "--container; hash 741d50b confirmed",
        )
    ]
    tokens, count = _overlap(output, prefix, suffix)
    assert count == 5
    assert set(tokens) == {
        "deriveDisposition",
        "src/ashiato/recall.py",
        "kusabi#274",
        "--container",
        "741d50b",
    }


def test_overlap_prefix_still_excludes_distinctive_token():
    # A distinctive token the session already used before the recall is removed.
    output = "Reuse deriveDisposition here."
    prefix = [_Activity(1, "Earlier we set deriveDisposition to pending.")]
    suffix = [_Activity(2, "Now deriveDisposition is applied as recalled.")]
    tokens, count = _overlap(output, prefix, suffix)
    assert count == 0
    assert tokens == []


def test_overlap_mixed_tokens_counts_only_distinctive():
    # Same output/suffix pair: ordinary words ignored, distinctive ones counted.
    output = "The different test used deriveDisposition and a date 2026-08-16."
    prefix: list[_Activity] = []
    suffix = [
        _Activity(2, "We ran deriveDisposition; the different test passed on 2026-08-16.")
    ]
    tokens, count = _overlap(output, prefix, suffix)
    assert count == 1
    assert tokens == ["deriveDisposition"]


def test_a_mixed_build_ingests_both_formats_without_disturbing_the_other(tmp_path: Path):
    """Acceptance criterion 1: a jsonl source and an ndjson source in one build()."""
    db_path = tmp_path / "mixed.duckdb"
    result = build([FIXTURES], db_path, opencode_sources=[OPENCODE_FIXTURE])
    assert result.n_files == 5  # the 4 existing fixtures + the opencode one
    # Since issue #83 the opencode fixture contributes its own rows: 3 sessions
    # (one per distinct session id in the file), 3 text-chunk events, and 3
    # completed tool parts.
    assert result.n_sessions == TOTAL_SESSIONS + 3
    assert result.n_events == TOTAL_EVENTS + 3
    assert result.n_tool_calls == TOTAL_TOOL_CALLS + 3
    assert result.n_recall_calls == 2  # only from the opencode fixture

    connection = connect(db_path, read_only=True)
    try:
        assert scalar(connection, "SELECT count(*) FROM sessions") == TOTAL_SESSIONS + 3
        assert scalar(connection, "SELECT count(*) FROM events") == TOTAL_EVENTS + 3
        assert scalar(connection, "SELECT count(*) FROM tool_calls") == TOTAL_TOOL_CALLS + 3
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


def test_cursor_transcript_populates_sessions_events_and_tool_calls(tmp_path: Path):
    """Acceptance criterion 1: rows in all three main tables, source = 'cursor'.

    Previously Cursor files were ingested into ``recall_calls`` only and
    ``source_files.n_events`` / ``n_tool_calls`` were hardcoded 0; issue #85
    makes them a full insert path like opencode and Codex.  One file is one
    session: ``ParsedCursorFile`` carries a file-level session id (the
    transcript file name's uuid stem), so one ``sessions`` row per file.
    """
    transcript_dir = tmp_path / "cursor"
    transcript_dir.mkdir()
    _write_cursor_transcript(transcript_dir / "sess1.jsonl", CURSOR_TRANSCRIPT_LINES)
    kaiba_path = tmp_path / "kaiba.db"
    _make_kaiba_recalls_db(kaiba_path)

    db_path = tmp_path / "cursor.duckdb"
    result = build([], db_path, cursor_sources=[transcript_dir], kaiba_db_path=kaiba_path)
    assert result.n_sessions == 1
    assert result.n_events == 1
    assert result.n_tool_calls == 1
    assert result.n_recall_calls == 1

    connection = connect(db_path, read_only=True)
    try:
        sessions = connection.execute(
            "SELECT session_id, source, started_at, ended_at, n_events, n_tool_calls "
            "FROM sessions"
        ).fetchall()
        # One session per file, named by the file-name uuid stem; ts is NULL
        # everywhere, so started_at/ended_at are NULL too.
        assert sessions == [("sess1", "cursor", None, None, 1, 1)]

        events = connection.execute(
            "SELECT event_id, session_id, source, seq, ts, type, text "
            "FROM events ORDER BY seq"
        ).fetchall()
        assert len(events) == 1
        assert events[0][0].startswith("cursor:text:")
        assert events[0][1] == "sess1"
        assert events[0][2] == "cursor"
        assert events[0][3] == 2
        assert events[0][4] is None  # ts is NULL -- Cursor records none
        assert events[0][5] == "text"
        assert events[0][6] == "Let me check kaiba first."

        calls = connection.execute(
            "SELECT tool_use_id, session_id, source, seq, tool_name, mcp_server "
            "FROM tool_calls ORDER BY tool_use_id"
        ).fetchall()
        assert len(calls) == 1
        assert calls[0][1] == "sess1" and calls[0][2] == "cursor"
        # The one tool_use block sits at block_index 1 (the text block is 0),
        # so its synthesised call id is "seq:block_index".
        assert calls[0][3] == 2
        assert calls[0][0] == "2:1"
        assert calls[0][4] == "CallMcpTool"
        assert calls[0][5] == "kaiba"
    finally:
        connection.close()


def test_cursor_tool_call_outcome_and_is_error_are_null(tmp_path: Path):
    """Acceptance criterion 2: outcome IS NULL and is_error IS NULL -- not 'pending', not False.

    Cursor records a ``tool_use`` block but never its result -- no output, no
    status, nothing -- so a call's fate is genuinely unknown.  'pending'
    would claim the session ended mid-call, and is_error=False would claim a
    known success.  NULL is invisible to both hygiene's ``pending_tool_call``
    count and nominate's ``outcome = 'error'`` gate, which is exactly right
    for a call whose fate is unknown.
    """
    transcript_dir = tmp_path / "cursor"
    transcript_dir.mkdir()
    _write_cursor_transcript(transcript_dir / "sess1.jsonl", CURSOR_TRANSCRIPT_LINES)
    kaiba_path = tmp_path / "kaiba.db"
    _make_kaiba_recalls_db(kaiba_path)

    db_path = tmp_path / "cursor.duckdb"
    build([], db_path, cursor_sources=[transcript_dir], kaiba_db_path=kaiba_path)
    connection = connect(db_path, read_only=True)
    try:
        rows = connection.execute(
            "SELECT outcome, is_error, result_text, result_truncated FROM tool_calls"
        ).fetchall()
        assert rows == [(None, None, None, None)]
    finally:
        connection.close()


def test_cursor_ts_is_null_and_seq_orders_within_a_file(tmp_path: Path):
    """Acceptance criterion 3: ts is NULL on Cursor events and tool calls.

    No timestamp is fabricated: Cursor records none, and ``seq`` /
    ``block_index`` already order the rows within a file.  This holds even
    when the kaiba-ledger join populated a timestamp on the recall row -- the
    join feeds ``recall_calls`` only, never the main tables.
    """
    transcript_dir = tmp_path / "cursor"
    transcript_dir.mkdir()
    _write_cursor_transcript(transcript_dir / "sess1.jsonl", CURSOR_TRANSCRIPT_LINES)
    kaiba_path = tmp_path / "kaiba.db"
    _make_kaiba_recalls_db(kaiba_path)

    db_path = tmp_path / "cursor.duckdb"
    build([], db_path, cursor_sources=[transcript_dir], kaiba_db_path=kaiba_path)
    connection = connect(db_path, read_only=True)
    try:
        assert scalar(
            connection, "SELECT count(*) FROM events WHERE ts IS NOT NULL"
        ) == 0
        assert scalar(
            connection, "SELECT count(*) FROM tool_calls WHERE ts IS NOT NULL"
        ) == 0
        # The recall row's ts comes from the kaiba join and is not NULL -- the
        # point is that it never leaks into the main tables.
        assert (
            scalar(connection, "SELECT count(*) FROM recall_calls WHERE ts IS NOT NULL")
            == 1
        )
        assert (
            scalar(connection, "SELECT count(*) FROM sessions WHERE started_at IS NOT NULL")
            == 0
        )
    finally:
        connection.close()


def test_cursor_rows_null_every_column_without_an_equivalent(tmp_path: Path):
    """Acceptance criterion 4: unrecorded columns are NULL, never a placeholder.

    A plausible value cannot be told apart from a measured one, so
    ``duration_ms``, ``cwd``, token counts, ``permission_mode``,
    ``is_sidechain``, ``role``, ``parent_uuid`` and friends are all NULL --
    not 0, not False.
    """
    transcript_dir = tmp_path / "cursor"
    transcript_dir.mkdir()
    _write_cursor_transcript(transcript_dir / "sess1.jsonl", CURSOR_TRANSCRIPT_LINES)
    kaiba_path = tmp_path / "kaiba.db"
    _make_kaiba_recalls_db(kaiba_path)

    db_path = tmp_path / "cursor.duckdb"
    build([], db_path, cursor_sources=[transcript_dir], kaiba_db_path=kaiba_path)
    connection = connect(db_path, read_only=True)
    try:
        session = connection.execute(
            "SELECT project_dir, cwd, git_branch, cc_version, entrypoint, input_tokens, "
            "output_tokens, cache_read_tokens, cache_creation_tokens "
            "FROM sessions WHERE session_id = 'sess1'"
        ).fetchone()
        assert session == (None,) * 9

        event = connection.execute(
            "SELECT role, parent_uuid, depth, is_sidechain, is_meta, permission_mode, "
            "effort, request_id, message_id, model, cwd, git_branch "
            "FROM events WHERE session_id = 'sess1'"
        ).fetchone()
        assert event == (None,) * 12

        calls = connection.execute(
            "SELECT call_event_id, result_event_id, duration_ms, permission_mode, cwd, "
            "is_sidechain, parent_tool_use_id "
            "FROM tool_calls ORDER BY tool_use_id"
        ).fetchall()
        assert calls == [(None,) * 7]
    finally:
        connection.close()


def test_cursor_recall_calls_are_unchanged(tmp_path: Path):
    """Acceptance criterion 5: Cursor recall_calls are byte-identical to before issue #85.

    The kaiba-ledger reconstruction is untouched by this change; pinning the
    whole recall row here means an accidental perturbation of the recall path
    -- or of the reconstructed ts -- fails the suite rather than hiding in
    prose.
    """
    transcript_dir = tmp_path / "cursor"
    transcript_dir.mkdir()
    _write_cursor_transcript(transcript_dir / "sess1.jsonl", CURSOR_TRANSCRIPT_LINES)
    kaiba_path = tmp_path / "kaiba.db"
    _make_kaiba_recalls_db(kaiba_path)

    db_path = tmp_path / "cursor.duckdb"
    build([], db_path, cursor_sources=[transcript_dir], kaiba_db_path=kaiba_path)
    file_key = str((transcript_dir / "sess1.jsonl").resolve())
    connection = connect(db_path, read_only=True)
    try:
        rows = connection.execute(
            "SELECT recall_id, session_id, source, seq, ts, call_id, query, output, "
            "output_truncated, followup_text, followup_truncated, overlap_count, "
            "overlap_tokens FROM recall_calls"
        ).fetchall()
        assert rows == [
            (
                f"{file_key}:2:1",
                "sess1",
                "cursor",
                2,
                datetime(2026, 8, 20, 10, 0),  # reconstructed from the kaiba ledger
                "2:1",
                "denial_pattern_x9",
                "Use anchored prefix matching for denial_pattern_x9 tokens.",
                False,
                None,  # no activity on strictly later lines
                False,
                0,
                "[]",
            ),
        ]
    finally:
        connection.close()


def test_cursor_kaiba_join_is_unreachable_from_the_main_tables(tmp_path: Path):
    """Acceptance criterion 6: the kaiba-db join cannot reach the main-table insert path.

    ``_insert_cursor_parsed`` builds sessions/events/tool_calls rows purely
    from the transcript's own shapes (``CursorToolCall`` / ``CursorTextChunk``,
    which carry no ts and no output) and hands the kaiba-joined ``RecallCall``
    rows to ``recall_calls`` alone.  Observable consequence: a recall call
    whose output/ts were reconstructed from the ledger still produces a
    tool_calls row with NULL result_text and NULL ts.
    """
    transcript_dir = tmp_path / "cursor"
    transcript_dir.mkdir()
    _write_cursor_transcript(transcript_dir / "sess1.jsonl", CURSOR_TRANSCRIPT_LINES)
    kaiba_path = tmp_path / "kaiba.db"
    _make_kaiba_recalls_db(kaiba_path)

    db_path = tmp_path / "cursor.duckdb"
    build([], db_path, cursor_sources=[transcript_dir], kaiba_db_path=kaiba_path)
    connection = connect(db_path, read_only=True)
    try:
        # The recall call itself: its recall_calls row carries the joined
        # output and ts...
        recall = connection.execute(
            "SELECT output, ts FROM recall_calls WHERE call_id = '2:1'"
        ).fetchone()
        assert recall == (
            "Use anchored prefix matching for denial_pattern_x9 tokens.",
            datetime(2026, 8, 20, 10, 0),
        )
        # ...but the same call's tool_calls row does not: no result_text, no
        # ts, no outcome -- the join did not leak into the main table.
        call = connection.execute(
            "SELECT result_text, ts, outcome, is_error FROM tool_calls "
            "WHERE tool_use_id = '2:1'"
        ).fetchone()
        assert call == (None, None, None, None)
    finally:
        connection.close()


def test_cursor_source_files_counts_match_inserted_rows(tmp_path: Path):
    """Acceptance criterion 8: source_files counts equal the rows actually inserted."""
    transcript_dir = tmp_path / "cursor"
    transcript_dir.mkdir()
    _write_cursor_transcript(transcript_dir / "sess1.jsonl", CURSOR_TRANSCRIPT_LINES)
    kaiba_path = tmp_path / "kaiba.db"
    _make_kaiba_recalls_db(kaiba_path)

    db_path = tmp_path / "cursor.duckdb"
    build([], db_path, cursor_sources=[transcript_dir], kaiba_db_path=kaiba_path)
    connection = connect(db_path, read_only=True)
    try:
        n_events, n_tool_calls = connection.execute(
            "SELECT n_events, n_tool_calls FROM source_files"
        ).fetchone()
        assert n_events == scalar(connection, "SELECT count(*) FROM events")
        assert n_tool_calls == scalar(connection, "SELECT count(*) FROM tool_calls")
        assert (n_events, n_tool_calls) == (1, 1)
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

# ---------------------------------------------------------------- codex tool_calls

def _write_codex_session(path: Path, session_id: str, tool_calls: list[dict]) -> None:
    """Write a minimal Codex JSONL session file with the given tool calls."""
    lines: list[dict] = [
        {"type": "session_meta", "payload": {"id": session_id}},
    ]
    for tc in tool_calls:
        lines.append({
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "thread_id": session_id,
                "item": tc,
            },
        })
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def test_codex_build_populates_tool_calls(tmp_path: Path):
    """Building with a Codex source inserts one tool_calls row per parsed call."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session(
        codex_dir / "session.jsonl",
        "codex-s1",
        [
            {
                "type": "CommandExecution",
                "id": "exec-1",
                "command": ["ls", "-la"],
                "stdout": "total 0\n",
            },
            {
                "type": "CommandExecution",
                "id": "exec-2",
                "command": ["echo", "hello"],
                "stdout": "hello\n",
            },
            {
                "type": "call_mcp_tool",
                "id": "mcp-1",
                "server": "sunaba",
                "tool": "publish",
                "arguments": {"files": ["foo.py"]},
                "result": "ok",
            },
        ],
    )

    db_path = tmp_path / "codex_test.duckdb"
    result = build([], db_path, codex_sources=[codex_dir])

    assert result.n_tool_calls == 3

    connection = connect(db_path, read_only=True)
    try:
        tc_count = scalar(connection, "SELECT count(*) FROM tool_calls")
        assert tc_count == 3

        # tool_names are preserved
        names = connection.execute(
            "SELECT tool_name FROM tool_calls ORDER BY seq"
        ).fetchall()
        assert names == [("Bash",), ("Bash",), ("mcp__sunaba__publish",)]

        # source_files.n_tool_calls matches
        sf_count = scalar(
            connection,
            "SELECT n_tool_calls FROM source_files WHERE file_path LIKE '%session.jsonl'",
        )
        assert sf_count == 3

        # source_files.n_events stays zero (no events inserted for codex)
        sf_events = scalar(
            connection,
            "SELECT n_events FROM source_files WHERE file_path LIKE '%session.jsonl'",
        )
        assert sf_events == 0
    finally:
        connection.close()


def test_codex_tool_call_input_summary_is_greppable(tmp_path: Path):
    """A Bash command is findable via input_summary on tool_calls rows."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session(
        codex_dir / "sess.jsonl",
        "codex-grep",
        [
            {
                "type": "CommandExecution",
                "id": "exec-10",
                "command": "git diff HEAD~3",
                "stdout": "diff --git a/foo.py ...",
            },
        ],
    )

    db_path = tmp_path / "codex_grep.duckdb"
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        summary = scalar(
            connection,
            "SELECT input_summary FROM tool_calls WHERE tool_use_id = 'exec-10'",
        )
        assert summary is not None
        assert "git diff HEAD~3" in summary

        # Also findable via input JSON
        input_json = scalar(
            connection,
            "SELECT input FROM tool_calls WHERE tool_use_id = 'exec-10'",
        )
        assert "git diff HEAD~3" in input_json
    finally:
        connection.close()


def test_codex_tool_call_outcome(tmp_path: Path):
    """Tool calls with output get outcome='ok'; pending without."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session(
        codex_dir / "outcome.jsonl",
        "codex-out",
        [
            {
                "type": "CommandExecution",
                "id": "ok-1",
                "command": "echo ok",
                "stdout": "ok\n",
            },
        ],
    )

    db_path = tmp_path / "codex_out.duckdb"
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        outcome = scalar(
            connection,
            "SELECT outcome FROM tool_calls WHERE tool_use_id = 'ok-1'",
        )
        assert outcome == "ok"

        is_error = scalar(
            connection,
            "SELECT is_error FROM tool_calls WHERE tool_use_id = 'ok-1'",
        )
        assert is_error is False
    finally:
        connection.close()


def test_codex_failed_command_is_error(tmp_path: Path):
    """A failed command with nonzero exit code is_error=True / outcome='error'."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session(
        codex_dir / "failed.jsonl",
        "codex-failed",
        [
            {
                "type": "CommandExecution",
                "id": "fail-1",
                "command": "false",
                "status": "failed",
                "exit_code": 1,
                "stdout": "",
                "stderr": "boom\n",
            },
        ],
    )

    db_path = tmp_path / "codex_failed.duckdb"
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        row = connection.execute(
            "SELECT is_error, outcome FROM tool_calls WHERE tool_use_id = 'fail-1'"
        ).fetchone()
        assert row == (True, "error")
    finally:
        connection.close()


def test_codex_completed_command_zero_exit_is_ok(tmp_path: Path):
    """A completed command with exit code 0 stays is_error=False / outcome='ok'."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session(
        codex_dir / "completed.jsonl",
        "codex-completed",
        [
            {
                "type": "CommandExecution",
                "id": "ok-exit-1",
                "command": "true",
                "status": "completed",
                "exit_code": 0,
                "stdout": "done\n",
            },
        ],
    )

    db_path = tmp_path / "codex_completed.duckdb"
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        row = connection.execute(
            "SELECT is_error, outcome FROM tool_calls WHERE tool_use_id = 'ok-exit-1'"
        ).fetchone()
        assert row == (False, "ok")
    finally:
        connection.close()


def test_codex_failed_mcp_error_is_reachable_in_result_text(tmp_path: Path):
    """A failed MCP call classifies as error and keeps its error message."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session(
        codex_dir / "mcp-failed.jsonl",
        "codex-mcp-failed",
        [
            {
                "type": "McpToolCall",
                "id": "mcp-fail-1",
                "server": "kaiba",
                "tool": "recall",
                "arguments": {"query": "x"},
                "status": "failed",
                "error": "Server error: connection refused",
                "result": None,
            },
        ],
    )

    db_path = tmp_path / "codex_mcp_failed.duckdb"
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        row = connection.execute(
            "SELECT is_error, outcome, result_text FROM tool_calls WHERE tool_use_id = 'mcp-fail-1'"
        ).fetchone()
        assert row is not None
        is_error, outcome, result_text = row
        assert is_error is True
        assert outcome == "error"
        assert result_text == "Server error: connection refused"
    finally:
        connection.close()


def test_codex_mcp_error_without_status_is_error(tmp_path: Path):
    """An older-shape MCP call with a non-empty error but no status is an error.

    The parser deliberately tolerates items that omit ``status``; without the
    error-evidence rule, the absent status would fall back to output-only
    classification and this failure would be recorded as 'ok' with the error
    message sitting in ``result_text``.
    """
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session(
        codex_dir / "mcp-no-status.jsonl",
        "codex-mcp-no-status",
        [
            {
                "type": "McpToolCall",
                "id": "mcp-no-status-1",
                "server": "kaiba",
                "tool": "recall",
                "arguments": {"query": "z"},
                "error": "Server error: connection refused",
                "result": None,
            },
        ],
    )

    db_path = tmp_path / "codex_mcp_no_status.duckdb"
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        row = connection.execute(
            "SELECT is_error, outcome, result_text FROM tool_calls WHERE tool_use_id = 'mcp-no-status-1'"
        ).fetchone()
        assert row is not None
        is_error, outcome, result_text = row
        assert is_error is True
        assert outcome == "error"
        assert result_text == "Server error: connection refused"
    finally:
        connection.close()


def test_codex_completed_status_wins_over_error_inference(tmp_path: Path):
    """An explicit status of 'completed' overrides the error-message inference.

    A non-empty ``error`` is failure evidence only while the status is absent:
    when the source explicitly says the call completed, that explicit status
    wins and the call is recorded as 'ok'.
    """
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session(
        codex_dir / "mcp-completed.jsonl",
        "codex-mcp-completed",
        [
            {
                "type": "McpToolCall",
                "id": "mcp-completed-1",
                "server": "kaiba",
                "tool": "recall",
                "arguments": {"query": "z"},
                "status": "completed",
                "error": "Server error: connection refused",
                "result": None,
            },
        ],
    )

    db_path = tmp_path / "codex_mcp_completed.duckdb"
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        row = connection.execute(
            "SELECT is_error, outcome FROM tool_calls WHERE tool_use_id = 'mcp-completed-1'"
        ).fetchone()
        assert row is not None
        is_error, outcome = row
        assert is_error is False
        assert outcome == "ok"
    finally:
        connection.close()


def test_codex_mcp_empty_or_nonstring_error_is_not_error_evidence(tmp_path: Path):
    """Empty or non-string error fields do not flip the classification alone.

    Only a non-empty string ``error`` is failure evidence; an empty string is
    indistinguishable from an absent field and a non-string value keeps today's
    behaviour (the parser does not even surface it as ``error``).
    """
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session(
        codex_dir / "mcp-no-evidence.jsonl",
        "codex-mcp-no-evidence",
        [
            {
                "type": "McpToolCall",
                "id": "mcp-empty-err-1",
                "server": "kaiba",
                "tool": "recall",
                "arguments": {"query": "a"},
                "error": "",
                "result": None,
            },
            {
                "type": "McpToolCall",
                "id": "mcp-num-err-1",
                "server": "kaiba",
                "tool": "recall",
                "arguments": {"query": "b"},
                "error": 42,
                "result": None,
            },
        ],
    )

    db_path = tmp_path / "codex_mcp_no_evidence.duckdb"
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        rows = connection.execute(
            "SELECT tool_use_id, is_error, outcome FROM tool_calls ORDER BY seq"
        ).fetchall()
        assert rows == [
            ("mcp-empty-err-1", False, "pending"),
            ("mcp-num-err-1", False, "ok"),
        ]
    finally:
        connection.close()


def test_codex_failed_with_no_output_is_error_not_pending(tmp_path: Path):
    """A failed call with no stdout/result at all still classifies as 'error'."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session(
        codex_dir / "silent-failed.jsonl",
        "codex-silent-failed",
        [
            {
                "type": "CommandExecution",
                "id": "silent-fail-1",
                "command": "crashed",
                "status": "failed",
                "exit_code": 2,
                "stdout": "",
            },
            {
                "type": "McpToolCall",
                "id": "mcp-silent-fail-1",
                "server": "kaiba",
                "tool": "recall",
                "arguments": {"query": "y"},
                "status": "failed",
                "error": None,
                "result": None,
            },
        ],
    )

    db_path = tmp_path / "codex_silent_failed.duckdb"
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        outcomes = connection.execute(
            "SELECT tool_use_id, outcome FROM tool_calls ORDER BY seq"
        ).fetchall()
        assert outcomes == [
            ("silent-fail-1", "error"),
            ("mcp-silent-fail-1", "error"),
        ]
    finally:
        connection.close()


def test_codex_old_shape_without_output_stays_pending(tmp_path: Path):
    """An item with no status/exit_code and no output keeps today's 'pending'."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session(
        codex_dir / "pending.jsonl",
        "codex-pending",
        [
            {
                "type": "CommandExecution",
                "id": "pend-1",
                "command": "long-running",
            },
        ],
    )

    db_path = tmp_path / "codex_pending.duckdb"
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        row = connection.execute(
            "SELECT is_error, outcome FROM tool_calls WHERE tool_use_id = 'pend-1'"
        ).fetchone()
        assert row == (False, "pending")
    finally:
        connection.close()


def test_codex_command_cwd_and_duration_ms_columns(tmp_path: Path):
    """cwd and duration_ms are filled from the item when present."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session(
        codex_dir / "meta.jsonl",
        "codex-meta",
        [
            {
                "type": "CommandExecution",
                "id": "cwd-1",
                "command": "pwd",
                "status": "completed",
                "exit_code": 0,
                "cwd": "/work/project",
                "duration": 0.25,
                "stdout": "/work/project\n",
            },
            {
                "type": "CommandExecution",
                "id": "nocwd-1",
                "command": "pwd",
                "status": "completed",
                "exit_code": 0,
                "stdout": "/work/project\n",
            },
        ],
    )

    db_path = tmp_path / "codex_meta.duckdb"
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        rows = connection.execute(
            "SELECT tool_use_id, cwd, duration_ms FROM tool_calls ORDER BY seq"
        ).fetchall()
        assert rows == [
            ("cwd-1", "/work/project", 250),
            ("nocwd-1", None, None),
        ]
    finally:
        connection.close()


def test_codex_malformed_status_does_not_claim_success(tmp_path: Path):
    """Non-string status / non-int exit_code parse and never become 'ok'."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session(
        codex_dir / "weird.jsonl",
        "codex-weird",
        [
            {
                "type": "CommandExecution",
                "id": "weird-status-1",
                "command": "ls",
                "status": 42,
                "stdout": "x\n",
            },
            {
                "type": "CommandExecution",
                "id": "weird-exit-1",
                "command": "ls",
                "exit_code": "0",
                "stdout": "x\n",
            },
        ],
    )

    db_path = tmp_path / "codex_weird.duckdb"
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        rows = connection.execute(
            "SELECT tool_use_id, is_error, outcome FROM tool_calls ORDER BY seq"
        ).fetchall()
        assert rows == [
            ("weird-status-1", True, "error"),
            ("weird-exit-1", True, "error"),
        ]
    finally:
        connection.close()


def test_codex_tool_call_tool_kind_and_mcp_server(tmp_path: Path):
    """Bash calls get tool_kind='builtin'; MCP calls get tool_kind='mcp' with server."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session(
        codex_dir / "kinds.jsonl",
        "codex-kinds",
        [
            {"type": "CommandExecution", "id": "c1", "command": "pwd", "stdout": "/work"},
            {
                "type": "McpToolCall",
                "id": "m1",
                "server": "kaiba",
                "tool": "recall",
                "arguments": {"query": "test"},
                "result": "[]",
            },
        ],
    )

    db_path = tmp_path / "codex_kinds.duckdb"
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        rows = connection.execute(
            "SELECT tool_name, tool_kind, mcp_server FROM tool_calls ORDER BY seq"
        ).fetchall()
        assert rows == [
            ("Bash", "builtin", None),
            ("mcp__kaiba__recall", "mcp", "kaiba"),
        ]
    finally:
        connection.close()


def test_codex_tool_calls_replaced_on_rebuild(tmp_path: Path):
    """Rebuilding the same codex file replaces prior tool_calls (no orphans)."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session(
        codex_dir / "replace.jsonl",
        "codex-rep",
        [{"type": "CommandExecution", "id": "x1", "command": "pwd", "stdout": "/a"}],
    )

    db_path = tmp_path / "codex_rep.duckdb"
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        assert scalar(connection, "SELECT count(*) FROM tool_calls") == 1
    finally:
        connection.close()

    # Rebuild: should still have exactly 1 row, not 2
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        assert scalar(connection, "SELECT count(*) FROM tool_calls") == 1
    finally:
        connection.close()


def test_codex_tool_call_event_ids_are_null(tmp_path: Path):
    """call_event_id and result_event_id are NULL for Codex tool calls.

    The item_completed items this parser consumes use a different id space
    from the model-facing response_item call_id, so no real event id can be
    recovered here; a synthetic id would be a reference that resolves to
    nothing.  NULL says the source does not link calls to events.
    """
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session(
        codex_dir / "evtids.jsonl",
        "codex-evt",
        [
            {
                "type": "CommandExecution",
                "id": "evt-1",
                "command": "echo x",
                "stdout": "x",
            },
        ],
    )

    db_path = tmp_path / "codex_evt.duckdb"
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        row = connection.execute(
            "SELECT call_event_id, result_event_id FROM tool_calls WHERE tool_use_id = 'evt-1'"
        ).fetchone()
        assert row is not None
        call_event_id, result_event_id = row
        assert call_event_id is None
        assert result_event_id is None
    finally:
        connection.close()


def test_every_non_null_call_event_id_resolves_to_an_event(tmp_path: Path):
    """Every non-null call_event_id / result_event_id resolves to an events row.

    Claude fills both columns from real event ids, so they resolve.  Codex
    fills both with NULL, because the item_completed id space never meets the
    event id space -- so a non-null value that fails to resolve is always a
    defect (a synthetic id asserting a link the data does not have).  Since
    issue #83 opencode does the same: its event stream does not link completed
    tool parts to events rows, so both columns are NULL there too.  Since
    issue #85 Cursor does the same: a tool_use block is never linked to an
    events row (Cursor records no result event at all), so both columns are
    NULL for Cursor as well.  The invariant is checked across a database
    built from more than one source, so a future source that writes
    unresolvable ids fails here.
    """
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session(
        codex_dir / "session.jsonl",
        "codex-s1",
        [
            {
                "type": "CommandExecution",
                "id": "exec-1",
                "command": "echo x",
                "stdout": "x",
            },
            {
                "type": "call_mcp_tool",
                "id": "mcp-1",
                "server": "sunaba",
                "tool": "publish",
                "arguments": {"files": ["f.py"]},
                "result": "ok",
            },
        ],
    )
    cursor_dir = tmp_path / "cursor"
    cursor_dir.mkdir()
    _write_cursor_transcript(cursor_dir / "sess1.jsonl", CURSOR_TRANSCRIPT_LINES)

    db_path = tmp_path / "link_invariant.duckdb"
    build(
        [FIXTURES],
        db_path,
        codex_sources=[codex_dir],
        opencode_sources=[OPENCODE_FIXTURE],
        cursor_sources=[cursor_dir],
    )

    connection = connect(db_path, read_only=True)
    try:
        # The database really does contain rows from more than one source:
        # Claude rows carry resolvable ids, Codex rows carry NULL, and opencode
        # rows carry NULL too.  Cursor rows are present and carry NULL as well.
        assert scalar(
            connection,
            "SELECT count(*) FROM tool_calls WHERE call_event_id IS NOT NULL",
        ) > 0
        assert scalar(
            connection,
            "SELECT count(*) FROM tool_calls WHERE call_event_id IS NULL",
        ) > 0
        assert (
            scalar(
                connection,
                "SELECT count(*) FROM tool_calls WHERE source = 'opencode'",
            )
            == 3
        )
        assert (
            scalar(
                connection,
                "SELECT count(*) FROM events WHERE source = 'opencode'",
            )
            == 3
        )
        assert (
            scalar(
                connection,
                "SELECT count(*) FROM tool_calls WHERE source = 'cursor'",
            )
            == 1
        )
        assert (
            scalar(
                connection,
                "SELECT count(*) FROM events WHERE source = 'cursor'",
            )
            == 1
        )

        for column in ("call_event_id", "result_event_id"):
            orphans = connection.execute(
                f"""
                SELECT tc.tool_use_id, tc.{column}
                FROM tool_calls tc
                LEFT JOIN events e ON e.event_id = tc.{column}
                WHERE tc.{column} IS NOT NULL AND e.event_id IS NULL
                """
            ).fetchall()
            assert orphans == [], (
                f"tool_calls.{column} values with no events row: {orphans}"
            )
    finally:
        connection.close()


def _write_codex_session_with_timeline(
    path: Path,
    session_id: str,
    *,
    tool_calls: list[dict] | None = None,
    text_chunks: list[dict] | None = None,
    token_usage: dict | None = None,
    timestamps: list[str] | None = None,
) -> None:
    """Write a Codex JSONL file with timestamps, text, and optional token usage."""
    lines: list[dict] = []
    ts_list = timestamps or []
    tc_list = tool_calls or []
    txt_list = text_chunks or []

    # session_meta
    meta_ts = ts_list[0] if ts_list else None
    meta: dict = {"type": "session_meta", "payload": {"id": session_id}}
    if meta_ts:
        meta["timestamp"] = meta_ts
    lines.append(meta)

    # tool call events
    for i, tc in enumerate(tc_list):
        ts = ts_list[i + 1] if i + 1 < len(ts_list) else None
        ev: dict = {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "thread_id": session_id,
                "item": tc,
            },
        }
        if ts:
            ev["timestamp"] = ts
        lines.append(ev)

    # text chunk events
    offset = 1 + len(tc_list)
    for i, txt in enumerate(txt_list):
        ts = ts_list[offset + i] if offset + i < len(ts_list) else None
        ev = {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "thread_id": session_id,
                "item": {
                    "type": "AgentResponse",
                    "id": f"text-{i}",
                    "text": txt,
                },
            },
        }
        if ts:
            ev["timestamp"] = ts
        lines.append(ev)

    # token_usage_record
    if token_usage is not None:
        tu_ts = ts_list[-1] if len(ts_list) > offset + len(txt_list) else None
        tu: dict = {
            "type": "token_usage_record",
            "payload": {"thread_token_usage": token_usage},
        }
        if tu_ts:
            tu["timestamp"] = tu_ts
        lines.append(tu)

    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def test_codex_build_inserts_session_row(tmp_path: Path):
    """Building with a Codex source inserts one sessions row with correct fields."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session_with_timeline(
        codex_dir / "sess.jsonl",
        "codex-sess-1",
        tool_calls=[
            {"type": "CommandExecution", "id": "e1", "command": "pwd", "stdout": "/work"},
        ],
        timestamps=[
            "2026-09-05T10:00:00Z",
            "2026-09-05T10:00:01Z",
            "2026-09-05T10:00:02Z",
        ],
        token_usage={"input_tokens": 500, "cached_input_tokens": 100, "output_tokens": 200},
    )

    db_path = tmp_path / "codex_sess.duckdb"
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        # Exactly one sessions row
        sess_count = scalar(connection, "SELECT count(*) FROM sessions")
        assert sess_count == 1

        row = connection.execute(
            "SELECT session_id, source, started_at, ended_at, n_events, n_tool_calls, "
            "input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens "
            "FROM sessions"
        ).fetchone()
        assert row is not None
        assert row[0] == "codex-sess-1"  # session_id
        assert row[1] == "codex"  # source -- new in issue #81
        assert row[2] is not None  # started_at
        assert row[3] is not None  # ended_at
        assert row[4] == 0  # n_events (no text chunks)
        assert row[5] == 1  # n_tool_calls
        assert row[6] == 500  # input_tokens
        assert row[7] == 200  # output_tokens
        assert row[8] == 100  # cache_read_tokens
        assert row[9] == 0  # cache_creation_tokens (Codex has none)
    finally:
        connection.close()


def test_codex_build_inserts_events_for_text_chunks(tmp_path: Path):
    """Text chunks become events rows; source_files.n_events matches."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session_with_timeline(
        codex_dir / "events.jsonl",
        "codex-ev-1",
        tool_calls=[
            {"type": "CommandExecution", "id": "e1", "command": "echo a", "stdout": "a"},
        ],
        text_chunks=["Hello world", "Second message"],
        timestamps=[
            "2026-09-05T10:00:00Z",
            "2026-09-05T10:00:01Z",
            "2026-09-05T10:00:02Z",
            "2026-09-05T10:00:03Z",
        ],
    )

    db_path = tmp_path / "codex_ev.duckdb"
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        # Two events from the two text chunks
        ev_count = scalar(connection, "SELECT count(*) FROM events")
        assert ev_count == 2

        # source_files.n_events matches
        sf_events = scalar(
            connection,
            "SELECT n_events FROM source_files WHERE file_path LIKE '%events.jsonl'",
        )
        assert sf_events == 2

        # Events have correct text and timestamps
        rows = connection.execute(
            "SELECT text, ts, role FROM events ORDER BY seq"
        ).fetchall()
        assert rows[0][0] == "Hello world"
        assert rows[0][2] == "unknown"
        assert rows[1][0] == "Second message"
    finally:
        connection.close()


def test_codex_build_tool_call_ts_populated(tmp_path: Path):
    """tool_calls.ts is non-NULL when record timestamps are present."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session_with_timeline(
        codex_dir / "ts.jsonl",
        "codex-ts-1",
        tool_calls=[
            {"type": "CommandExecution", "id": "e1", "command": "pwd", "stdout": "/"},
        ],
        timestamps=[
            "2026-09-05T10:00:00Z",
            "2026-09-05T10:00:05Z",
        ],
    )

    db_path = tmp_path / "codex_ts.duckdb"
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        ts = scalar(
            connection,
            "SELECT ts FROM tool_calls WHERE tool_use_id = 'e1'",
        )
        assert ts is not None
    finally:
        connection.close()


def test_codex_build_no_text_no_events(tmp_path: Path):
    """A Codex file with no text chunks still yields zero events."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session(
        codex_dir / "notext.jsonl",
        "codex-notext",
        [{"type": "CommandExecution", "id": "e1", "command": "pwd", "stdout": "/"}],
    )

    db_path = tmp_path / "codex_notext.duckdb"
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        ev_count = scalar(connection, "SELECT count(*) FROM events")
        assert ev_count == 0
        sf_events = scalar(
            connection,
            "SELECT n_events FROM source_files WHERE file_path LIKE '%notext.jsonl'",
        )
        assert sf_events == 0
    finally:
        connection.close()


def test_codex_build_rebuild_replaces_sessions_and_events(tmp_path: Path):
    """Rebuilding a Codex file replaces old sessions/events (no orphans)."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()

    _write_codex_session_with_timeline(
        codex_dir / "rebuild.jsonl",
        "codex-rb",
        tool_calls=[{"type": "CommandExecution", "id": "e1", "command": "pwd", "stdout": "/a"}],
        text_chunks=["first"],
        timestamps=["2026-09-05T10:00:00Z", "2026-09-05T10:00:01Z", "2026-09-05T10:00:02Z"],
    )

    db_path = tmp_path / "codex_rb.duckdb"
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        assert scalar(connection, "SELECT count(*) FROM sessions") == 1
        assert scalar(connection, "SELECT count(*) FROM events") == 1
        assert scalar(connection, "SELECT count(*) FROM tool_calls") == 1
    finally:
        connection.close()

    # Rebuild with different data
    _write_codex_session_with_timeline(
        codex_dir / "rebuild.jsonl",
        "codex-rb",
        tool_calls=[
            {"type": "CommandExecution", "id": "e1", "command": "pwd", "stdout": "/b"},
            {"type": "CommandExecution", "id": "e2", "command": "ls", "stdout": "file"},
        ],
        text_chunks=["first", "second", "third"],
        timestamps=["2026-09-05T10:00:00Z", "2026-09-05T10:00:01Z", "2026-09-05T10:00:02Z",
                     "2026-09-05T10:00:03Z", "2026-09-05T10:00:04Z"],
    )
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        assert scalar(connection, "SELECT count(*) FROM sessions") == 1
        assert scalar(connection, "SELECT count(*) FROM events") == 3
        assert scalar(connection, "SELECT count(*) FROM tool_calls") == 2
    finally:
        connection.close()


@pytest.mark.parametrize("read_only", [False, True])
def test_connect_disables_progress_bar(
    tmp_path: Path, read_only: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DuckDB's progress bar goes to stdout and would corrupt ``--json`` output (#58).

    DuckDB already defaults the bar to off under pytest, but to on in a plain
    process (the CLI): the raw connection is forced to ``true`` here so the test
    fails unless ``connect`` itself turns it off.
    """
    db_path = tmp_path / "progress.duckdb"
    build_module.connect(db_path).close()
    real_connect = duckdb.connect

    def connect_with_bar_on(*args: object, **kwargs: object) -> duckdb.DuckDBPyConnection:
        raw = real_connect(*args, **kwargs)
        raw.execute("SET enable_progress_bar=true")
        return raw

    monkeypatch.setattr(build_module.duckdb, "connect", connect_with_bar_on)
    connection = build_module.connect(db_path, read_only=read_only)
    try:
        setting = connection.execute(
            "SELECT current_setting('enable_progress_bar')"
        ).fetchone()
    finally:
        connection.close()
    assert setting == (False,)
def test_codex_build_file_change_rows_identify_touched_files(tmp_path: Path):
    """A FileChange item inserts a tool_calls row whose files are queryable."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session(
        codex_dir / "file-change.jsonl",
        "codex-fc",
        [
            {
                "type": "FileChange",
                "id": "fc-1",
                "changes": {
                    "/work/src/foo.py": {"type": "edit", "content": "x"},
                    "/work/src/bar.py": {"type": "delete"},
                },
                "status": "completed",
                "stdout": "Success. Updated the following files:\nM /work/src/foo.py\n",
                "stderr": "",
            },
        ],
    )

    db_path = tmp_path / "codex_fc.duckdb"
    result = build([], db_path, codex_sources=[codex_dir])
    assert result.n_tool_calls == 1

    connection = connect(db_path, read_only=True)
    try:
        row = connection.execute(
            "SELECT tool_name, input, outcome, is_error FROM tool_calls WHERE tool_use_id = 'fc-1'"
        ).fetchone()
        assert row is not None
        tool_name, input_json, outcome, is_error = row
        assert tool_name == "FileChange"
        # Both touched files are reachable in the input JSON.
        assert "src/foo.py" in input_json
        assert "src/bar.py" in input_json
        assert scalar(
            connection,
            "SELECT count(*) FROM tool_calls WHERE tool_name = 'FileChange' "
            "AND input LIKE '%src/foo.py%'",
        ) == 1
        # completed status + stdout -> a clean success.
        assert outcome == "ok"
        assert is_error is False
    finally:
        connection.close()


def test_codex_build_collab_and_subagent_rows_carry_agent(tmp_path: Path):
    """Delegation items insert rows; the delegated agent is reachable."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session(
        codex_dir / "delegation.jsonl",
        "codex-del",
        [
            {
                "type": "CollabAgentToolCall",
                "id": "call-wait-1",
                "tool": "wait",
                "status": "completed",
                "sender_thread_id": "thread-parent",
                "receiver_thread_ids": ["thread-child"],
                "receiver_agents": ["sol"],
                "agents_states": {},
            },
            {
                "type": "SubAgentActivity",
                "id": "call-sa-1",
                "kind": "started",
                "agent_thread_id": "agent-1",
                "agent_path": "/root/kusabi_484_luna",
            },
        ],
    )

    db_path = tmp_path / "codex_del.duckdb"
    result = build([], db_path, codex_sources=[codex_dir])
    assert result.n_tool_calls == 2

    connection = connect(db_path, read_only=True)
    try:
        names = connection.execute(
            "SELECT tool_name FROM tool_calls ORDER BY seq"
        ).fetchall()
        assert names == [("collab__wait",), ("collab__subagent",)]

        # The delegated agent is reachable from the collab call's input.
        collab_input = scalar(
            connection,
            "SELECT input FROM tool_calls WHERE tool_use_id = 'call-wait-1'",
        )
        assert collab_input is not None
        assert '"receiver_agents"' in collab_input
        assert '"sol"' in collab_input

        # And the subagent row names its agent path.
        subagent_input = scalar(
            connection,
            "SELECT input FROM tool_calls WHERE tool_use_id = 'call-sa-1'",
        )
        assert subagent_input is not None
        assert "kusabi_484_luna" in subagent_input
        assert scalar(
            connection,
            "SELECT count(*) FROM tool_calls WHERE tool_name = 'collab__subagent' "
            "AND input LIKE '%kusabi_484_luna%'",
        ) == 1
    finally:
        connection.close()


def test_codex_build_context_compaction_is_its_own_event_type(tmp_path: Path):
    """A ContextCompaction item inserts an events row with a distinct type."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session(
        codex_dir / "compaction.jsonl",
        "codex-compact",
        [
            {"type": "ContextCompaction", "id": "compaction-1"},
        ],
    )

    db_path = tmp_path / "codex_compact.duckdb"
    result = build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        # One events row, with its own type -- never 'text'.
        rows = connection.execute(
            "SELECT type, role, text FROM events ORDER BY seq"
        ).fetchall()
        assert rows == [("context_compaction", None, "")]
        assert scalar(connection, "SELECT count(*) FROM events WHERE type = 'text'") == 0
        # No tool call is produced by a compaction.
        assert result.n_tool_calls == 0
        # The event counts toward sessions/source_files n_events.
        assert scalar(connection, "SELECT n_events FROM sessions") == 1
        assert scalar(
            connection,
            "SELECT n_events FROM source_files WHERE file_path LIKE '%compaction.jsonl'",
        ) == 1
    finally:
        connection.close()


def test_codex_build_new_item_types_tolerate_malformed_instances(tmp_path: Path):
    """Malformed new item types build without raising and keep their rows."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session(
        codex_dir / "malformed.jsonl",
        "codex-mal",
        [
            {"type": "FileChange", "id": "fc-1", "changes": "not-a-dict"},
            {"type": "CollabAgentToolCall", "id": "ca-1", "tool": 42},
            {"type": "SubAgentActivity", "id": "sa-1", "agent_path": 7},
            {"type": "ContextCompaction", "id": "cc-1"},
            {"type": "MysteryFutureItem", "id": "m-1"},
        ],
    )

    db_path = tmp_path / "codex_mal.duckdb"
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        # The three malformed-but-known tool calls still get rows; the unknown
        # type is dropped silently.
        assert scalar(connection, "SELECT count(*) FROM tool_calls") == 3
        assert scalar(
            connection,
            "SELECT count(*) FROM events WHERE type = 'context_compaction'",
        ) == 1
    finally:
        connection.close()


# ---------------------------------------------------------------- codex #65/#67: role, is_meta, event_id


def _write_codex_session_with_roles(
    path: Path,
    session_id: str,
    *,
    messages: list[dict],
    timestamps: list[str] | None = None,
) -> None:
    """Write a Codex JSONL with response_item messages carrying role."""
    lines: list[dict] = []
    ts_list = timestamps or []
    meta: dict = {"type": "session_meta", "payload": {"id": session_id}}
    if ts_list:
        meta["timestamp"] = ts_list[0]
    lines.append(meta)

    for i, msg in enumerate(messages):
        ts = ts_list[i + 1] if i + 1 < len(ts_list) else None
        ev: dict = {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": f"msg-{i}",
                "role": msg["role"],
                "content": [{"type": "output_text", "text": msg["text"]}],
            },
        }
        if ts:
            ev["timestamp"] = ts
        lines.append(ev)

    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def test_codex_build_role_and_is_meta(tmp_path: Path):
    """Codex text events carry the real role from the JSONL; developer is is_meta."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session_with_roles(
        codex_dir / "role.jsonl",
        "codex-role-1",
        messages=[
            {"role": "assistant", "text": "I can help with that."},
            {"role": "user", "text": "Please fix the bug."},
            {"role": "developer", "text": "<environment_context>...</environment_context>"},
        ],
        timestamps=[
            "2026-09-05T10:00:00Z",
            "2026-09-05T10:00:01Z",
            "2026-09-05T10:00:02Z",
            "2026-09-05T10:00:03Z",
        ],
    )

    db_path = tmp_path / "codex_role.duckdb"
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        rows = connection.execute(
            "SELECT text, role, is_meta FROM events ORDER BY seq"
        ).fetchall()
        assert len(rows) == 3
        assert rows[0] == ("I can help with that.", "assistant", False)
        assert rows[1] == ("Please fix the bug.", "user", False)
        assert rows[2] == (
            "<environment_context>...</environment_context>",
            "developer",
            True,
        )
    finally:
        connection.close()


def test_codex_build_unknown_role_fallback(tmp_path: Path):
    """A text chunk with role=None gets fallback 'unknown' and is_meta=False."""
    from ashiato.build import _codex_text_chunk_to_event
    from ashiato.codex import CodexTextChunk
    from ashiato.parser import EVENT_COLUMNS

    chunk = CodexTextChunk(
        session_id="s1",
        file_path="/data/sessions/f.jsonl",
        seq=5,
        ts=None,
        text="fallback role test",
        role=None,
    )
    row = _codex_text_chunk_to_event(chunk)
    role_idx = list(EVENT_COLUMNS).index("role")
    is_meta_idx = list(EVENT_COLUMNS).index("is_meta")
    assert row[role_idx] == "unknown"
    assert row[is_meta_idx] is False


def test_codex_build_event_id_includes_file_path(tmp_path: Path):
    """Two files with a text chunk at the same seq produce different event_ids."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session_with_roles(
        codex_dir / "session_a.jsonl",
        "codex-a",
        messages=[{"role": "assistant", "text": "file A msg"}],
        timestamps=["2026-09-05T10:00:00Z", "2026-09-05T10:00:01Z"],
    )
    _write_codex_session_with_roles(
        codex_dir / "session_b.jsonl",
        "codex-b",
        messages=[{"role": "assistant", "text": "file B msg"}],
        timestamps=["2026-09-05T10:00:00Z", "2026-09-05T10:00:01Z"],
    )

    db_path = tmp_path / "codex_eid.duckdb"
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        ids = connection.execute(
            "SELECT event_id FROM events ORDER BY file_path"
        ).fetchall()
        assert len(ids) == 2
        assert ids[0][0] != ids[1][0]
    finally:
        connection.close()


def test_codex_build_event_id_deterministic(tmp_path: Path):
    """Parsing the same file twice produces identical event_ids."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_session_with_roles(
        codex_dir / "det.jsonl",
        "codex-det",
        messages=[{"role": "assistant", "text": "deterministic"}],
        timestamps=["2026-09-05T10:00:00Z", "2026-09-05T10:00:01Z"],
    )

    db1 = tmp_path / "det1.duckdb"
    build([], db1, codex_sources=[codex_dir])
    db2 = tmp_path / "det2.duckdb"
    build([], db2, codex_sources=[codex_dir])

    conn1 = connect(db1, read_only=True)
    try:
        ids1 = conn1.execute("SELECT event_id FROM events ORDER BY seq").fetchall()
    finally:
        conn1.close()
    conn2 = connect(db2, read_only=True)
    try:
        ids2 = conn2.execute("SELECT event_id FROM events ORDER BY seq").fetchall()
    finally:
        conn2.close()

    assert ids1 == ids2


def test_codex_build_item_completed_plus_response_item_one_event_row(tmp_path: Path):
    """An item_completed of type AgentMessage (or UserMessage) that carries the
    same text as a response_item message produces only one event row — the
    parser does not match AgentMessage/UserMessage in item_completed, so only
    the response_item path contributes a chunk.

    Note: AgentResponse in item_completed *is* matched by the parser and would
    produce a second row, but that shape does not appear in the real corpus for
    dedup scenarios (the doubled-row path was ruled out of scope here).
    """
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    path = codex_dir / "dedup_build.jsonl"
    lines = [
        {"timestamp": "2026-09-05T10:00:00Z", "type": "session_meta", "payload": {"id": "dedup-b1"}},
        {
            "timestamp": "2026-09-05T10:00:01Z",
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "thread_id": "dedup-b1",
                "item": {
                    "type": "AgentMessage",
                    "id": "msg-dedup-1",
                    "content": [
                        {"type": "output_text", "text": "Duplicate message"},
                    ],
                },
            },
        },
        {
            "timestamp": "2026-09-05T10:00:02Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "msg-dedup",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Duplicate message"},
                ],
            },
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    db_path = tmp_path / "codex_dedup_build.duckdb"
    build([], db_path, codex_sources=[codex_dir])

    connection = connect(db_path, read_only=True)
    try:
        ev_count = scalar(connection, "SELECT count(*) FROM events")
        assert ev_count == 1
        role, text = connection.execute(
            "SELECT role, text FROM events ORDER BY seq"
        ).fetchone()
        assert role == "assistant"
        assert text == "Duplicate message"
    finally:
        connection.close()

    # --- second case: UserMessage shape (separate dir so build doesn't pick up case 1) ---
    codex_dir2 = tmp_path / "codex2"
    codex_dir2.mkdir()
    path2 = codex_dir2 / "dedup_build_user.jsonl"
    lines2 = [
        {"timestamp": "2026-09-05T10:00:00Z", "type": "session_meta", "payload": {"id": "dedup-b2"}},
        {
            "timestamp": "2026-09-05T10:00:01Z",
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "thread_id": "dedup-b2",
                "item": {
                    "type": "UserMessage",
                    "id": "msg-dedup-u1",
                    "content": [
                        {"type": "input_text", "text": "User message"},
                    ],
                },
            },
        },
        {
            "timestamp": "2026-09-05T10:00:02Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "msg-dedup-u",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "User message"},
                ],
            },
        },
    ]
    with open(path2, "w", encoding="utf-8") as f:
        for line in lines2:
            f.write(json.dumps(line) + "\n")

    db_path2 = tmp_path / "codex_dedup_build_user.duckdb"
    build([], db_path2, codex_sources=[codex_dir2])

    connection2 = connect(db_path2, read_only=True)
    try:
        ev_count2 = scalar(connection2, "SELECT count(*) FROM events")
        assert ev_count2 == 1
        role2, text2 = connection2.execute(
            "SELECT role, text FROM events ORDER BY seq"
        ).fetchone()
        assert role2 == "user"
        assert text2 == "User message"
    finally:
        connection2.close()


# ---------------------------------------------------------------- recall_calls (codex)


def _write_codex_recall_session(
    path: Path, session_id: str, calls: list[tuple[str | None, dict]]
) -> None:
    """Write a Codex JSONL session of item_completed payloads with record
    timestamps: each call is a (timestamp or None, item) pair."""
    lines: list[dict] = [
        {"type": "session_meta", "payload": {"id": session_id}},
    ]
    for timestamp, item in calls:
        line: dict = {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "thread_id": session_id,
                "item": item,
            },
        }
        if timestamp is not None:
            line["timestamp"] = timestamp
        lines.append(line)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def test_codex_recall_calls_carry_the_call_timestamp(tmp_path: Path):
    """A Codex recall row's ts is the call's own timestamp; a call with no
    parseable timestamp still produces a row with ts null."""
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_recall_session(
        codex_dir / "recall-ts.jsonl",
        "codex-recall-ts",
        [
            (
                "2026-09-05T10:00:00Z",
                {
                    "type": "McpToolCall",
                    "id": "recall-ts-1",
                    "server": "kaiba",
                    "tool": "recall",
                    "arguments": {"query": "backoff retry"},
                    "result": "use anchored_backoff_v7",
                },
            ),
            (
                None,
                {
                    "type": "McpToolCall",
                    "id": "recall-ts-2",
                    "server": "kaiba",
                    "tool": "recall",
                    "arguments": {"query": "other"},
                    "result": "answer",
                },
            ),
        ],
    )

    db_path = tmp_path / "codex_recall_ts.duckdb"
    result = build([], db_path, codex_sources=[codex_dir])
    assert result.n_recall_calls == 2

    connection = connect(db_path, read_only=True)
    try:
        rows = connection.execute(
            "SELECT call_id, ts FROM recall_calls ORDER BY call_id"
        ).fetchall()
    finally:
        connection.close()
    assert rows == [
        ("recall-ts-1", datetime(2026, 9, 5, 10, 0, 0)),
        ("recall-ts-2", None),
    ]


def test_codex_recall_ts_lands_in_a_time_window(tmp_path: Path):
    """A time-windowed query over recall_calls spanning a known Codex recall
    returns it.

    Before issue #73 every Codex recall row had ts NULL, and NULL fails every
    comparison -- so any window (``ashiato recalls --since``, ``WHERE ts
    BETWEEN ...``) silently dropped all Codex recalls.  This pins the fixed
    behaviour: the recall is findable in time.
    """
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    _write_codex_recall_session(
        codex_dir / "recall-window.jsonl",
        "codex-recall-win",
        [
            (
                "2026-09-05T10:00:00Z",
                {
                    "type": "McpToolCall",
                    "id": "recall-win-1",
                    "server": "kaiba",
                    "tool": "recall",
                    "arguments": {"query": "flaky retry"},
                    "result": "anchored_backoff_v7",
                },
            ),
        ],
    )

    db_path = tmp_path / "codex_recall_window.duckdb"
    result = build([], db_path, codex_sources=[codex_dir])
    assert result.n_recall_calls == 1

    connection = connect(db_path, read_only=True)
    try:
        rows = connection.execute(
            "SELECT call_id, query FROM recall_calls WHERE ts BETWEEN ? AND ?",
            [datetime(2026, 9, 5, 9, 0, 0), datetime(2026, 9, 5, 11, 0, 0)],
        ).fetchall()
    finally:
        connection.close()
    assert rows == [("recall-win-1", "flaky retry")]
