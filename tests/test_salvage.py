"""Tests for ashiato.salvage: nominating un-bookkept work-state changes.

Fixtures are hand-built JSONL transcripts run through the real build
pipeline (see ``tests/fixtures/`` for the record shape this mirrors) and
written to a private ``tmp_path``, never into ``tests/fixtures/`` itself --
that directory is globbed recursively by other tests' ``--source``, and an
extra file there would silently change their row counts.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from ashiato.build import build, connect
from ashiato.cli import main
from ashiato.salvage import (
    fetch_evidence_events,
    nominate,
    open_kaiba,
)
from ashiato.schema import FORMAT_VERSION

# ---------------------------------------------------------------- fixture transcript


def _tool_use(tool_use_id: str, name: str, input_: dict) -> dict:
    return {"type": "tool_use", "id": tool_use_id, "name": name, "input": input_}


def _tool_result(tool_use_id: str, text: str, *, is_error: bool = False) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": [{"type": "text", "text": text}],
        "is_error": is_error,
    }


def _assistant(uuid: str, parent: str | None, session_id: str, ts: str, blocks: list) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": session_id,
        "timestamp": ts,
        "message": {"role": "assistant", "content": blocks},
    }


def _user(uuid: str, parent: str | None, session_id: str, ts: str, blocks: list) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": session_id,
        "timestamp": ts,
        "message": {"role": "user", "content": blocks},
    }


def _call(
    records: list[dict],
    prefix: str,
    session_id: str,
    ts_call: str,
    ts_result: str,
    tool_name: str,
    tool_input: dict,
    result_text: str = "ok",
) -> str:
    """Append one tool_use/tool_result pair; return the tool_use_id."""
    tool_use_id = f"{prefix}_use"
    parent = records[-1]["uuid"] if records else None
    call_uuid = f"{prefix}_call"
    records.append(
        _assistant(
            call_uuid, parent, session_id, ts_call, [_tool_use(tool_use_id, tool_name, tool_input)]
        )
    )
    records.append(
        _user(
            f"{prefix}_result",
            call_uuid,
            session_id,
            ts_result,
            [_tool_result(tool_use_id, result_text)],
        )
    )
    return tool_use_id


@pytest.fixture
def transcript_dir(tmp_path: Path) -> Path:
    """One transcript file, five sessions, each set up for a distinct nomination outcome."""
    records: list[dict] = []

    # ses-1: publish ok, then a successful agenda_edit in the same session -- covered.
    _call(
        records,
        "s1_pub",
        "ses-1",
        "2026-08-20T10:00:00Z",
        "2026-08-20T10:00:01Z",
        "mcp__sunaba__publish",
        {"files": ["a.py"], "create_pr": True},
    )
    _call(
        records,
        "s1_agenda",
        "ses-1",
        "2026-08-20T10:00:02Z",
        "2026-08-20T10:00:03Z",
        "mcp__kaiba__agenda_edit",
        {"op": "done"},
    )

    # ses-2: publish ok, no bookkeeping anywhere -- nominated in transcript-only mode.
    _call(
        records,
        "s2_pub",
        "ses-2",
        "2026-08-20T11:00:00Z",
        "2026-08-20T11:00:01Z",
        "mcp__sunaba__publish",
        {"files": ["b.py"], "create_pr": True},
    )

    # ses-3: publish ok, no session bookkeeping; a kaiba actions row lands inside the window.
    _call(
        records,
        "s3_pub",
        "ses-3",
        "2026-08-20T12:00:00Z",
        "2026-08-20T12:00:01Z",
        "mcp__sunaba__publish",
        {"files": ["c.py"], "create_pr": True},
    )

    # ses-4: publish ok, no session bookkeeping; a kaiba actions row exists but outside the window.
    _call(
        records,
        "s4_pub",
        "ses-4",
        "2026-08-20T13:00:00Z",
        "2026-08-20T13:00:01Z",
        "mcp__sunaba__publish",
        {"files": ["d.py"], "create_pr": True},
    )

    # ses-5: a chain-wait Bash call, no bookkeeping -- the second evidence signal.
    _call(
        records,
        "s5_wait",
        "ses-5",
        "2026-08-20T14:00:00Z",
        "2026-08-20T14:00:01Z",
        "Bash",
        {"command": "scripts/chain-wait.sh --chain 42"},
    )

    path = tmp_path / "salvage_transcript.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def db(tmp_path: Path, transcript_dir: Path) -> Path:
    path = tmp_path / "salvage.duckdb"
    result = build([transcript_dir], path)
    assert result.n_processed == 1
    return path


def _make_kaiba_db(path: Path, actions: list[tuple[str, str | None, str | None]]) -> None:
    """A tiny actions ledger: (content, created_at, done_at) per row."""
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE actions ("
            "id INTEGER PRIMARY KEY, content TEXT, position REAL, author TEXT, "
            "created_at TEXT, done_at TEXT)"
        )
        connection.executemany(
            "INSERT INTO actions (content, position, author, created_at, done_at) "
            "VALUES (?, 0.0, 'tester', ?, ?)",
            actions,
        )
        connection.commit()
    finally:
        connection.close()


# ---------------------------------------------------------------- nomination rule


def test_bookkeeping_in_the_same_session_suppresses_the_nomination(db: Path):
    connection = connect(db, read_only=True)
    try:
        nominations = nominate(connection, None)
    finally:
        connection.close()
    assert "ses-1" not in {n.session_id for n in nominations}


def test_publish_with_no_bookkeeping_is_nominated_with_session_and_snippet(db: Path):
    connection = connect(db, read_only=True)
    try:
        nominations = nominate(connection, None)
    finally:
        connection.close()
    nomination = next(n for n in nominations if n.session_id == "ses-2")
    assert nomination.kind == "publish"
    assert nomination.failed_checks == ("session",)
    assert "b.py" in nomination.snippet


def test_kaiba_coverage_within_the_window_suppresses_the_nomination(db: Path, tmp_path: Path):
    kaiba_path = tmp_path / "kaiba.db"
    _make_kaiba_db(kaiba_path, [("bookkeep ses-3", "2026-08-20T12:05:00Z", None)])
    connection = connect(db, read_only=True)
    kaiba_connection = open_kaiba(kaiba_path)
    try:
        nominations = nominate(connection, kaiba_connection, window_minutes=30)
    finally:
        connection.close()
        kaiba_connection.close()
    assert "ses-3" not in {n.session_id for n in nominations}


def test_kaiba_coverage_outside_the_window_does_not_suppress(db: Path, tmp_path: Path):
    kaiba_path = tmp_path / "kaiba.db"
    # 59 minutes after the evidence ts of ses-4 (13:00:01), well past a 30-minute window.
    _make_kaiba_db(kaiba_path, [("bookkeep ses-4", "2026-08-20T14:00:00Z", None)])
    connection = connect(db, read_only=True)
    kaiba_connection = open_kaiba(kaiba_path)
    try:
        nominations = nominate(connection, kaiba_connection, window_minutes=30)
    finally:
        connection.close()
        kaiba_connection.close()
    assert "ses-4" in {n.session_id for n in nominations}


def test_done_at_also_counts_as_kaiba_coverage(db: Path, tmp_path: Path):
    kaiba_path = tmp_path / "kaiba.db"
    _make_kaiba_db(kaiba_path, [("bookkeep ses-3", None, "2026-08-20T12:10:00Z")])
    connection = connect(db, read_only=True)
    kaiba_connection = open_kaiba(kaiba_path)
    try:
        nominations = nominate(connection, kaiba_connection, window_minutes=30)
    finally:
        connection.close()
        kaiba_connection.close()
    assert "ses-3" not in {n.session_id for n in nominations}


def test_chain_wait_bash_calls_are_a_second_evidence_kind(db: Path):
    connection = connect(db, read_only=True)
    try:
        events = fetch_evidence_events(connection)
    finally:
        connection.close()
    chain_event = next(e for e in events if e.kind == "chain-wait")
    assert chain_event.session_id == "ses-5"


def test_limit_caps_output_regardless_of_corpus_size(db: Path):
    connection = connect(db, read_only=True)
    try:
        nominations = nominate(connection, None, limit=1)
    finally:
        connection.close()
    assert len(nominations) == 1


def test_since_excludes_earlier_evidence(db: Path):
    connection = connect(db, read_only=True)
    try:
        nominations = nominate(connection, None, since=datetime(2026, 8, 20, 13, 30))
    finally:
        connection.close()
    # ses-2 (11:00), ses-3 (12:00), ses-4 (13:00) are all before the cutoff;
    # only ses-5's 14:00 chain-wait event remains.
    assert {n.session_id for n in nominations} == {"ses-5"}


def test_format_version_is_unchanged():
    assert FORMAT_VERSION == 4


# ---------------------------------------------------------------- read-only discipline


def test_open_kaiba_connection_cannot_write(tmp_path: Path):
    kaiba_path = tmp_path / "kaiba.db"
    _make_kaiba_db(kaiba_path, [])
    connection = open_kaiba(kaiba_path)
    assert connection is not None
    try:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("INSERT INTO actions (content) VALUES ('x')")
    finally:
        connection.close()


def test_open_kaiba_returns_none_for_a_missing_file(tmp_path: Path):
    assert open_kaiba(tmp_path / "nope.db") is None


def test_open_kaiba_returns_none_for_an_unreadable_file(tmp_path: Path):
    bogus = tmp_path / "not-a-db"
    bogus.write_text("not a sqlite file", encoding="utf-8")
    assert open_kaiba(bogus) is None


def test_nominate_does_not_modify_the_duckdb_file(db: Path):
    before = db.stat().st_mtime_ns
    connection = connect(db, read_only=True)
    try:
        nominate(connection, None)
    finally:
        connection.close()
    assert db.stat().st_mtime_ns == before


# ---------------------------------------------------------------- CLI wiring


def test_salvage_help_exits_zero():
    with pytest.raises(SystemExit) as excinfo:
        main(["salvage", "--help"])
    assert excinfo.value.code == 0


def test_cli_with_no_kaiba_db_is_transcript_only_and_exits_zero(
    db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    missing_kaiba = tmp_path / "nowhere" / "kaiba.db"
    assert main(["salvage", "--db", str(db), "--kaiba-db", str(missing_kaiba)]) == 0
    captured = capsys.readouterr()
    assert "transcript-only" in captured.err
    assert "ses-2" in captured.out


def test_cli_on_a_missing_database_fails_cleanly(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["salvage", "--db", str(tmp_path / "nope.duckdb")]) == 1
    assert "no database at" in capsys.readouterr().err


def test_cli_honours_limit(db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    missing_kaiba = tmp_path / "nowhere" / "kaiba.db"
    assert (
        main(
            ["salvage", "--db", str(db), "--kaiba-db", str(missing_kaiba), "--limit", "1"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "(1 nomination)" in out
