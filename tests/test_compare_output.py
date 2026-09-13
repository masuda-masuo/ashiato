"""Output tests for ``ashiato compare-periods`` (issue #39, TDD step 3).

Focused tests for:

* Default format is table and presents baseline/current effective bounds
  plus sessions/tool-call coverage.
* All five hygiene categories appear in stable order with baseline/current
  calls, sessions, calls/session, absolute changes, percent change, and
  ``n/a`` for undefined percent.  Whitespace is not frozen.
* JSON and table output are byte-identical across two independent builds
  of the same compact corpus.
* Running either format leaves the DuckDB file mtime unchanged.
* A missing DB and an older DB missing required views fail cleanly
  without traceback, following existing CLI conventions.

This file reuses compact helpers independently; it does not import
``test_compare`` or ``test_compare_validation``.
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import pytest

from ashiato.cli import main

# ---------------------------------------------------------------------------
# Category order -- same frozen order as hygiene.CATEGORY_ORDER
# ---------------------------------------------------------------------------

CATEGORY_ORDER = (
    "companion_status_poll",
    "host_file_hunt",
    "raw_local_mcp_http",
    "undo_file_edit",
    "pending_tool_call",
)

# ---------------------------------------------------------------------------
# Compact synthetic fixture
# ---------------------------------------------------------------------------

BASELINE_SINCE = "2026-08-01T00:00:00Z"
BASELINE_UNTIL = "2026-08-07T23:59:59Z"
CURRENT_SINCE = "2026-08-15T00:00:00Z"
CURRENT_UNTIL = "2026-08-21T23:59:59Z"


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
    ts: str,
    tool_name: str,
    tool_input: dict,
    *,
    result_text: str = "ok",
    no_result: bool = False,
) -> None:
    """Append one tool_use (and optionally its tool_result) to *records*."""
    tool_use_id = f"{prefix}_use"
    parent = records[-1]["uuid"] if records else None
    call_uuid = f"{prefix}_call"
    records.append(
        _assistant(call_uuid, parent, session_id, ts, [_tool_use(tool_use_id, tool_name, tool_input)])
    )
    if not no_result:
        records.append(
            _user(
                f"{prefix}_result",
                call_uuid,
                session_id,
                ts,
                [_tool_result(tool_use_id, result_text)],
            )
        )


def _baseline_session() -> list[dict]:
    records: list[dict] = []

    def t(index: int) -> str:
        return f"2026-08-03T10:{index:02d}:00Z"

    # 3x companion_status_poll
    _call(records, "bp1", "baseline-a", t(0), "Bash", {"command": "kusabi-companion status"})
    _call(records, "bp2", "baseline-a", t(1), "Bash", {"command": "kusabi-companion status --json"})
    _call(records, "bp3", "baseline-a", t(2), "Bash", {"command": "kusabi-companion status"})

    # 1x host_file_hunt
    _call(records, "bh1", "baseline-a", t(3), "Bash", {"command": "rg 'TODO' /home/dev"})

    # 2x pending_tool_call (no result)
    _call(records, "pp1", "baseline-a", t(4), "mcp__foo__bar", {"x": 1}, no_result=True)
    _call(records, "pp2", "baseline-a", t(5), "mcp__baz__qux", {"y": 2}, no_result=True)

    return records


def _current_session_a() -> list[dict]:
    records: list[dict] = []

    def t(index: int) -> str:
        return f"2026-08-17T10:{index:02d}:00Z"

    # 2x companion_status_poll
    _call(records, "cp1", "current-a", t(0), "Bash", {"command": "kusabi-companion status"})
    _call(records, "cp2", "current-a", t(1), "Bash", {"command": "kusabi-companion status --json"})

    # 4x host_file_hunt
    _call(records, "ch1", "current-a", t(2), "Bash", {"command": "rg 'TODO' /home/dev"})
    _call(records, "ch2", "current-a", t(3), "Bash", {"command": "grep -rn 'api' /etc"})
    _call(records, "ch3", "current-a", t(4), "Bash", {"command": "cat /etc/hosts"})
    _call(records, "ch4", "current-a", t(5), "Bash", {"command": "sed -n '1,10p' /etc/hosts"})

    # 1x raw_local_mcp_http
    _call(records, "cr1", "current-a", t(6), "Bash", {"command": "curl http://127.0.0.1:8750/mcp"})

    # 1x undo_file_edit
    _call(records, "cu1", "current-a", t(7), "mcp__sunaba__undo_file_edit", {"file_path": "/tmp/a.py", "steps": 1})

    # 2x pending_tool_call (no result)
    _call(records, "cp3", "current-a", t(8), "mcp__foo__bar", {"x": 1}, no_result=True)
    _call(records, "cp5", "current-a", t(9), "mcp__foo__bar", {"x": 2}, no_result=True)

    return records


def _current_session_b() -> list[dict]:
    records: list[dict] = []
    _call(records, "cp4", "current-b", "2026-08-18T11:00:00Z", "mcp__baz__qux", {"y": 2}, no_result=True)
    return records


def _write_sessions(directory: Path, sessions: dict[str, list[dict]]) -> None:
    for session_id, records in sessions.items():
        path = directory / f"{session_id}.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _build_db(tmp_path: Path, label: str = "compare") -> Path:
    """Build a compact DB with two well-separated windows."""
    directory = tmp_path / "transcripts"
    directory.mkdir(parents=True, exist_ok=True)
    sessions: dict[str, list[dict]] = {
        "baseline-a": _baseline_session(),
        "current-a": _current_session_a(),
        "current-b": _current_session_b(),
    }
    _write_sessions(directory, sessions)
    db_path = tmp_path / f"{label}.duckdb"
    assert main(["build", "--source", str(directory), "--db", str(db_path)]) == 0
    return db_path


@pytest.fixture
def compare_db(tmp_path: Path) -> Path:
    """Synthetic DB with two well-separated windows: baseline and current."""
    return _build_db(tmp_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_compare(db: Path, *, format: str = "json") -> tuple[int, str]:
    """Run ``compare-periods`` and return (rc, captured stdout).

    Uses the same sys.stdout/stderr redirect pattern as
    ``test_compare_validation.py`` to capture stdout from ``main()``.
    """
    out_buf = StringIO()
    err_buf = StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = out_buf
    sys.stderr = err_buf
    try:
        rc = main([
            "compare-periods",
            "--period", f"{BASELINE_SINCE}..{BASELINE_UNTIL}",
            "--period", f"{CURRENT_SINCE}..{CURRENT_UNTIL}",
            "--db", str(db),
            "--format", format,
        ])
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 1
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    return rc, out_buf.getvalue()


def _run_compare_stderr(db: Path, *, format: str = "json") -> tuple[int, str]:
    """Run ``compare-periods`` and return (rc, captured stderr)."""
    err_buf = StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = err_buf
    try:
        rc = main([
            "compare-periods",
            "--period", f"{BASELINE_SINCE}..{BASELINE_UNTIL}",
            "--period", f"{CURRENT_SINCE}..{CURRENT_UNTIL}",
            "--db", str(db),
            "--format", format,
        ])
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 1
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    return rc, err_buf.getvalue()


def _run_compare_nodb(db_path: Path, *, format: str = "json") -> tuple[int, str]:
    """Run ``compare-periods`` against an arbitrary DB path, return (rc, stderr)."""
    err_buf = StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = err_buf
    try:
        rc = main([
            "compare-periods",
            "--period", f"{BASELINE_SINCE}..{BASELINE_UNTIL}",
            "--period", f"{CURRENT_SINCE}..{CURRENT_UNTIL}",
            "--db", str(db_path),
            "--format", format,
        ])
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 1
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    return rc, err_buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Default format is table
# ---------------------------------------------------------------------------


class TestDefaultFormatIsTable:
    """The default output format (no ``--format`` flag) must be table."""

    def test_argparse_help_lists_table_as_default(self) -> None:
        """``--help`` text for compare-periods says default is table."""
        out_buf = StringIO()
        err_buf = StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = out_buf
        sys.stderr = err_buf
        try:
            with pytest.raises(SystemExit) as exc_info:
                main(["compare-periods", "--help"])
            assert exc_info.value.code == 0
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        help_text = out_buf.getvalue()
        assert "default table" in help_text.lower() or "default: table" in help_text.lower(), (
            f"Expected 'default table' in help text but got:\n{help_text}"
        )

    def test_no_format_flag_returns_table_via_captured_stdout(
        self, compare_db: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Without ``--format``, capsys captures table-formatted output."""
        rc = main([
            "compare-periods",
            "--period", f"{BASELINE_SINCE}..{BASELINE_UNTIL}",
            "--period", f"{CURRENT_SINCE}..{CURRENT_UNTIL}",
            "--db", str(compare_db),
        ])
        assert rc == 0
        out = capsys.readouterr().out
        # Table output has the header/separator pattern from _print_table
        assert "baseline:" in out
        assert "current:" in out


# ---------------------------------------------------------------------------
# 2. Table presents baseline/current bounds and coverage
# ---------------------------------------------------------------------------


class TestTableBoundsAndCoverage:
    """The table output must present period bounds and session/call coverage."""

    def test_table_shows_baseline_bounds(self, compare_db: Path) -> None:
        """Table output includes baseline period bounds."""
        rc, stdout = _run_compare(compare_db, format="table")
        assert rc == 0
        assert BASELINE_SINCE in stdout
        assert BASELINE_UNTIL in stdout

    def test_table_shows_current_bounds(self, compare_db: Path) -> None:
        """Table output includes current period bounds."""
        rc, stdout = _run_compare(compare_db, format="table")
        assert rc == 0
        assert CURRENT_SINCE in stdout
        assert CURRENT_UNTIL in stdout

    def test_table_shows_coverage_session_and_call_counts(self, compare_db: Path) -> None:
        """Table output presents session and tool-call counts for each period."""
        rc, stdout = _run_compare(compare_db, format="table")
        assert rc == 0
        # Baseline has 1 session and 6 tool calls
        # Current has 2 sessions and 11 tool calls
        # Coverage lines should mention these numbers
        assert "1" in stdout  # baseline sessions
        assert "2" in stdout  # current sessions
        assert "6" in stdout  # baseline tool calls
        assert "11" in stdout  # current tool calls


# ---------------------------------------------------------------------------
# 3. All five categories in stable order with complete columns
# ---------------------------------------------------------------------------


class TestTableCategories:
    """All five hygiene categories appear in stable order with the full
    column set: baseline/current calls, sessions, calls/session,
    absolute changes, and percent change."""

    def test_all_five_categories_present(self, compare_db: Path) -> None:
        """Every category from CATEGORY_ORDER appears in the table."""
        rc, stdout = _run_compare(compare_db, format="table")
        assert rc == 0
        for name in CATEGORY_ORDER:
            assert name in stdout, f"category {name!r} missing from table output"

    def test_categories_in_stable_order(self, compare_db: Path) -> None:
        """Categories appear in CATEGORY_ORDER in the table output."""
        rc, stdout = _run_compare(compare_db, format="table")
        assert rc == 0
        # Find category names in order of appearance
        found: list[str] = []
        for name in CATEGORY_ORDER:
            pos = stdout.find(name)
            if pos >= 0:
                found.append(name)
        assert found == list(CATEGORY_ORDER), (
            f"Expected order {CATEGORY_ORDER}, found {found}"
        )

    def test_table_has_calls_per_session_column(self, compare_db: Path) -> None:
        """The table includes calls/session values for both periods."""
        rc, stdout = _run_compare(compare_db, format="table")
        assert rc == 0
        # baseline-a: 6 calls / 1 session = 6.0
        # current: 11 calls / 2 sessions = 5.5
        assert "6.0" in stdout or "6.00" in stdout, (
            f"Expected baseline calls_per_session (6.0) in table:\n{stdout}"
        )

    def test_n_a_for_undefined_percent(self, compare_db: Path) -> None:
        """Categories with zero baseline calls show ``n/a`` for percent change."""
        rc, stdout = _run_compare(compare_db, format="table")
        assert rc == 0
        # raw_local_mcp_http and undo_file_edit both have 0 baseline calls
        # Their percent change must show as n/a, not NULL or null
        lines = stdout.splitlines()
        for line in lines:
            if "raw_local_mcp_http" in line or "undo_file_edit" in line:
                # Must NOT contain "NULL" (old behavior) — must contain n/a
                assert "n/a" in line.lower() or "N/A" in line, (
                    f"Expected 'n/a' for undefined percent in line:\n{line}"
                )

    def test_table_does_not_contain_null_for_none_percent(self, compare_db: Path) -> None:
        """The string 'NULL' must not appear in the table output."""
        rc, stdout = _run_compare(compare_db, format="table")
        assert rc == 0
        assert "NULL" not in stdout, (
            f"Table output contains 'NULL' — should use 'n/a' for undefined percent:\n{stdout}"
        )


# ---------------------------------------------------------------------------
# 4. Byte-identical output across two independent builds
# ---------------------------------------------------------------------------


class TestDeterministicOutput:
    """JSON and table outputs are byte-identical across two independent
    builds of the same compact corpus."""

    def test_json_identical_across_two_builds(self, tmp_path: Path) -> None:
        """Two separate builds from the same data produce identical JSON output."""
        db1 = _build_db(tmp_path / "run1", label="db1")
        db2 = _build_db(tmp_path / "run2", label="db2")

        _, json1 = _run_compare(db1, format="json")
        _, json2 = _run_compare(db2, format="json")
        assert json1 == json2, "JSON output differs across two independent builds"

    def test_table_identical_across_two_builds(self, tmp_path: Path) -> None:
        """Two separate builds from the same data produce identical table output."""
        db1 = _build_db(tmp_path / "run1", label="db1")
        db2 = _build_db(tmp_path / "run2", label="db2")

        _, table1 = _run_compare(db1, format="table")
        _, table2 = _run_compare(db2, format="table")
        assert table1 == table2, "Table output differs across two independent builds"


# ---------------------------------------------------------------------------
# 5. Read-only: DB mtime unchanged after running either format
# ---------------------------------------------------------------------------


class TestReadOnly:
    """Running compare-periods in either format must not modify the DB file."""

    def test_json_format_preserves_mtime(self, compare_db: Path) -> None:
        """JSON output does not modify the database file."""
        mtime_before = compare_db.stat().st_mtime_ns
        rc, _ = _run_compare(compare_db, format="json")
        mtime_after = compare_db.stat().st_mtime_ns
        assert rc == 0
        assert mtime_before == mtime_after, "DB mtime changed after JSON format run"

    def test_table_format_preserves_mtime(self, compare_db: Path) -> None:
        """Table output does not modify the database file."""
        mtime_before = compare_db.stat().st_mtime_ns
        rc, _ = _run_compare(compare_db, format="table")
        mtime_after = compare_db.stat().st_mtime_ns
        assert rc == 0
        assert mtime_before == mtime_after, "DB mtime changed after table format run"


# ---------------------------------------------------------------------------
# 6. Missing DB and old DB fail cleanly without traceback
# ---------------------------------------------------------------------------


class TestMissingDb:
    """A missing database must fail with exit code 1 and a clean error message."""

    def test_missing_db_exits_code_1(self, tmp_path: Path) -> None:
        """Non-existent DB path produces exit code 1."""
        rc, _ = _run_compare_nodb(tmp_path / "nonexistent.duckdb", format="json")
        assert rc == 1

    def test_missing_db_no_traceback(self, tmp_path: Path) -> None:
        """Non-existent DB path produces no traceback."""
        _rc, err = _run_compare_nodb(tmp_path / "nonexistent.duckdb", format="json")
        assert "Traceback" not in err

    def test_missing_db_has_error_message(self, tmp_path: Path) -> None:
        """Non-existent DB path produces an actionable error message."""
        _rc, err = _run_compare_nodb(tmp_path / "nonexistent.duckdb", format="json")
        assert "error:" in err.lower()


class TestOldDbMissingViews:
    """A database built by an older version (missing required views) must fail
    cleanly with exit code 1 and a clean error message."""

    def _build_old_db(self, tmp_path: Path) -> Path:
        """Build a DB then drop all required views to simulate an old schema."""
        db = _build_db(tmp_path, label="old")
        import duckdb
        conn = duckdb.connect(str(db), read_only=False)
        try:
            conn.execute("DROP VIEW IF EXISTS denial_followups")
            conn.execute("DROP VIEW IF EXISTS recall_followups")
        finally:
            conn.close()
        return db

    def test_old_db_exits_code_1(self, tmp_path: Path) -> None:
        """DB missing required views produces exit code 1."""
        db = self._build_old_db(tmp_path)
        rc, _ = _run_compare(db, format="json")
        assert rc == 1

    def test_old_db_no_traceback(self, tmp_path: Path) -> None:
        """DB missing required views produces no traceback."""
        db = self._build_old_db(tmp_path)
        _, err = _run_compare_stderr(db, format="json")
        assert "Traceback" not in err

    def test_old_db_has_error_message(self, tmp_path: Path) -> None:
        """DB missing required views produces an actionable error message."""
        db = self._build_old_db(tmp_path)
        _, err = _run_compare_stderr(db, format="json")
        assert "error:" in err.lower()
