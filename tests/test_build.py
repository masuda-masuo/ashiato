"""Database construction: schema, contents, and the incremental path."""

from __future__ import annotations

import json
import os
import shutil
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
    build,
    connect,
    database_info,
    default_db_path,
    iter_transcripts,
)
from ashiato.parser import EVENT_COLUMNS, SESSION_COLUMNS, TOOL_CALL_COLUMNS
from ashiato.schema import FOLLOWUP_KINDS, SOURCE_FILE_COLUMNS

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
