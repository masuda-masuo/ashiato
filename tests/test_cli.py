"""The three subcommands, driven through main() the way the console script does."""

from __future__ import annotations

import csv
import io
import json
import shutil
from pathlib import Path

import pytest

from ashiato.build import connect
from ashiato.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "cli.duckdb"
    assert main(["build", "--source", str(FIXTURES), "--db", str(path)]) == 0
    return path


# ---------------------------------------------------------------- build


def test_build_reports_what_it_did(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    path = tmp_path / "out.duckdb"
    assert main(["build", "--source", str(FIXTURES), "--db", str(path)]) == 0
    out = capsys.readouterr().out
    assert f"database: {path}" in out
    assert "4 processed, 0 skipped (unchanged)" in out
    assert "3 sessions, 70 events, 7 tool calls" in out
    assert "unparseable lines skipped: 2" in out
    assert path.exists()


def test_build_is_incremental_on_the_second_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    path = tmp_path / "out.duckdb"
    main(["build", "--source", str(FIXTURES), "--db", str(path)])
    capsys.readouterr()
    assert main(["build", "--source", str(FIXTURES), "--db", str(path)]) == 0
    assert "0 processed, 4 skipped (unchanged)" in capsys.readouterr().out


def test_source_is_repeatable(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    shutil.copy(FIXTURES / "session_main.jsonl", first)
    shutil.copy(FIXTURES / "session_snake.jsonl", second)

    path = tmp_path / "out.duckdb"
    assert (
        main(["build", "--source", str(first), "--source", str(second), "--db", str(path)]) == 0
    )
    assert "2 processed" in capsys.readouterr().out


def test_missing_source_warns_but_succeeds(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    path = tmp_path / "out.duckdb"
    assert main(["build", "--source", str(tmp_path / "nowhere"), "--db", str(path)]) == 0
    captured = capsys.readouterr()
    assert "source not found" in captured.err
    assert "0 processed" in captured.out


# ---------------------------------------------------------------- sql


def test_sql_table_format_is_the_default(db: Path, capsys: pytest.CaptureFixture[str]):
    query = "SELECT outcome, count(*) AS n FROM tool_calls GROUP BY 1 ORDER BY 1"
    assert main(["sql", query, "--db", str(db)]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].split() == ["outcome", "n"]
    assert set(lines[1]) == {"-", " "}
    assert [line.split() for line in lines[2:6]] == [
        ["denied", "2"],
        ["error", "1"],
        ["ok", "3"],
        ["pending", "1"],
    ]
    assert lines[6] == "(4 rows)"


def test_sql_renders_nulls_in_table_format(db: Path, capsys: pytest.CaptureFixture[str]):
    query = "SELECT result_event_id FROM tool_calls WHERE outcome = 'pending'"
    assert main(["sql", query, "--db", str(db), "--format", "table"]) == 0
    out = capsys.readouterr().out
    assert "NULL" in out
    assert "(1 row)" in out


def test_sql_json_format(db: Path, capsys: pytest.CaptureFixture[str]):
    query = (
        "SELECT tool_use_id, tool_name, outcome, duration_ms FROM tool_calls "
        "WHERE tool_use_id = 'toolu_ok_1'"
    )
    assert main(["sql", query, "--db", str(db), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {
            "tool_use_id": "toolu_ok_1",
            "tool_name": "Bash",
            "outcome": "ok",
            "duration_ms": 1500,
        }
    ]


def test_sql_csv_format(db: Path, capsys: pytest.CaptureFixture[str]):
    query = "SELECT outcome, result_event_id FROM tool_calls WHERE outcome = 'pending'"
    assert main(["sql", query, "--db", str(db), "--format", "csv"]) == 0
    rows = list(csv.reader(io.StringIO(capsys.readouterr().out)))
    assert rows == [["outcome", "result_event_id"], ["pending", ""]]


def test_sql_on_a_missing_database_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    assert main(["sql", "SELECT 1", "--db", str(tmp_path / "nope.duckdb")]) == 1
    assert "no database at" in capsys.readouterr().err


def test_sql_syntax_error_fails_cleanly(db: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["sql", "SELECT FROM WHERE", "--db", str(db)]) == 1
    assert capsys.readouterr().err.startswith("error:")


def test_sql_can_join_tool_calls_to_events(db: Path, capsys: pytest.CaptureFixture[str]):
    query = (
        "SELECT e.type, t.tool_name FROM tool_calls t "
        "JOIN events e ON e.event_id = t.call_event_id "
        "WHERE t.tool_use_id = 'toolu_sub_1'"
    )
    assert main(["sql", query, "--db", str(db), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out) == [{"type": "assistant", "tool_name": "Grep"}]


# ---------------------------------------------------------------- info


def test_info(db: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["info", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert f"database: {db}" in out
    assert "sessions" in out and "3" in out
    assert "events" in out and "70" in out
    assert "tool_calls" in out and "7" in out
    assert "source_files" in out
    assert "time window: 2026-08-10 09:00:00 .. 2026-08-10 11:00:49" in out


def test_info_on_a_missing_database_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    assert main(["info", "--db", str(tmp_path / "nope.duckdb")]) == 1
    assert "no database at" in capsys.readouterr().err


def test_info_on_an_empty_database(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    path = tmp_path / "empty.duckdb"
    main(["build", "--source", str(tmp_path / "no-transcripts-here"), "--db", str(path)])
    capsys.readouterr()
    assert main(["info", "--db", str(path)]) == 0
    assert "time window: empty" in capsys.readouterr().out


# ---------------------------------------------------------------- argument parsing


def test_version_flag():
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0


def test_a_subcommand_is_required():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_unknown_format_is_rejected():
    with pytest.raises(SystemExit) as excinfo:
        main(["sql", "SELECT 1", "--format", "yaml"])
    assert excinfo.value.code == 2


# ---------------------------------------------------------------- denials


def test_denials_prints_the_view(db: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["denials", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    header = out.splitlines()[0].split()
    assert header[:5] == ["session_id", "seq", "ts", "tool_name", "input_summary"]
    assert "followup_kind" in header
    # Both fixture denials, newest first.
    assert "mcp__sunaba__publish" in out
    assert "/etc/hosts" in out
    assert "(2 rows)" in out


def test_denials_json_format(db: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["denials", "--db", str(db), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [row["tool_name"] for row in payload] == ["mcp__sunaba__publish", "Write"]
    assert payload[1] == {
        "session_id": "11111111-1111-4111-8111-111111111111",
        "seq": 6,
        "ts": "2026-08-10 09:00:07",
        "tool_name": "Write",
        "input_summary": "/etc/hosts",
        "permission_mode": "default",
        "cwd": "/home/dev/proj",
        "next_tool_name": "mcp__sunaba__publish",
        "next_input_summary": '{"create_pr":true,"files":["src/ashiato/parser.py"]}',
        "next_outcome": "denied",
        "next_ts": "2026-08-10 09:00:09",
        "gap_seconds": 2.0,
        "followup_kind": "other-tool",
    }


def test_denials_csv_format(db: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["denials", "--db", str(db), "--format", "csv"]) == 0
    rows = list(csv.reader(io.StringIO(capsys.readouterr().out)))
    assert rows[0][0] == "session_id"
    assert len(rows) == 3  # header plus both denials


def test_denials_honours_limit(db: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["denials", "--db", str(db), "--format", "json", "--limit", "1"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 1
    # 0 is "no limit", not "no rows".
    assert main(["denials", "--db", str(db), "--format", "json", "--limit", "0"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 2


def test_denials_honours_session(db: Path, capsys: pytest.CaptureFixture[str]):
    session = "11111111-1111-4111-8111-111111111111"
    assert main(["denials", "--db", str(db), "--format", "json", "--session", session]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 2
    assert main(["denials", "--db", str(db), "--format", "json", "--session", "nobody"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_denials_output_is_identical_across_rebuilds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Determinism, at the surface the user actually reads."""
    outputs = []
    for name in ("first", "second"):
        path = tmp_path / f"{name}.duckdb"
        assert main(["build", "--source", str(FIXTURES), "--db", str(path)]) == 0
        capsys.readouterr()
        assert main(["denials", "--db", str(path), "--format", "json"]) == 0
        outputs.append(capsys.readouterr().out)
    assert outputs[0] == outputs[1]
    assert json.loads(outputs[0])  # not vacuously equal


def test_denials_on_a_missing_database_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    assert main(["denials", "--db", str(tmp_path / "nope.duckdb")]) == 1
    assert "no database at" in capsys.readouterr().err


def test_denials_on_a_database_without_the_view_fails_cleanly(
    db: Path, capsys: pytest.CaptureFixture[str]
):
    """An older database has no view; that is an error message, not a traceback."""
    connection = connect(db)
    try:
        connection.execute("DROP VIEW denial_followups")
    finally:
        connection.close()
    assert main(["denials", "--db", str(db)]) == 1
    assert capsys.readouterr().err.startswith("error:")


def test_a_negative_limit_is_rejected():
    with pytest.raises(SystemExit) as excinfo:
        main(["denials", "--limit", "-1"])
    assert excinfo.value.code == 2
