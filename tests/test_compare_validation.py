"""Period-validation tests for ``ashiato compare-periods`` (issue #39, TDD step 2).

These tests pin the *validation* layer of ``compare-periods``:

- Exactly two ``--period`` values are required; zero, one, or three must
  produce argparse/SystemExit code 2 with actionable text.
- Each value must be exactly non-empty ``START..END`` using the existing
  ISO parser; malformed values must produce code 2 without traceback.
- Each individual period must reject ``start > end``.
- The baseline must precede the current window.
- Overlapping windows must be rejected.
- Equality at ``baseline END == current START`` must be rejected because
  inclusive bounds would double-count the boundary.
- Valid separated windows must be accepted (code 0).

These tests are intentionally independent of the frozen
``tests/test_compare.py`` and of the output contract.  They test only
the validation gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ashiato.cli import main

# ---------------------------------------------------------------------------
# Minimal transcript fixture (same pattern as test_compare.py)
# ---------------------------------------------------------------------------

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


def _tool_use(tool_use_id: str, name: str, input_: dict) -> dict:
    return {"type": "tool_use", "id": tool_use_id, "name": name, "input": input_}


def _tool_result(tool_use_id: str, text: str) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": [{"type": "text", "text": text}],
    }


def _minimal_session(session_id: str, ts: str) -> list[dict]:
    """One event with a single tool call — just enough for a valid build."""
    blocks = [_tool_use("tu1", "Bash", {"command": "echo ok"})]
    return [
        _assistant("a1", None, session_id, ts, blocks),
        _user("u1", "a1", session_id, ts, [_tool_result("tu1", "ok")]),
    ]


def _write_sessions(directory: Path, sessions: dict[str, list[dict]]) -> None:
    for session_id, records in sessions.items():
        path = directory / f"{session_id}.jsonl"
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )


@pytest.fixture
def minimal_db(tmp_path: Path) -> Path:
    """Build a minimal valid DB with two sessions in two separated windows."""
    directory = tmp_path / "transcripts"
    directory.mkdir()
    sessions: dict[str, list[dict]] = {
        "sess-a": _minimal_session("sess-a", "2026-08-03T10:00:00Z"),
        "sess-b": _minimal_session("sess-b", "2026-08-17T10:00:00Z"),
    }
    _write_sessions(directory, sessions)
    db_path = tmp_path / "validation.duckdb"
    assert main(["build", "--source", str(directory), "--db", str(db_path)]) == 0
    return db_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(*args: str) -> tuple[int, str]:
    """Run ``main(["compare-periods", *args])`` and capture (rc, stderr).

    * For argparse-level exits (missing required args) ``main`` raises
      ``SystemExit(2)`` which we convert to ``(2, "")``.
    * For the custom validation in ``_run_compare_periods`` we capture the
      return code and the text printed to stderr.
    """
    import sys
    from io import StringIO

    err_buf = StringIO()
    old_stderr = sys.stderr
    sys.stderr = err_buf
    try:
        rc = main(["compare-periods", *args])
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 2
    finally:
        sys.stderr = old_stderr
    return rc, err_buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Exactly two --period values required
# ---------------------------------------------------------------------------

class TestPeriodCount:
    """Zero, one, or three --period values must produce code 2."""

    def test_zero_periods_gives_code2(self) -> None:
        """No --period → argparse SystemExit(2)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["compare-periods"])
        assert exc_info.value.code == 2

    def test_zero_periods_has_actionable_text(self, capsys: pytest.CaptureFixture[str]) -> None:
        """No --period → stderr mentions the required flag."""
        with pytest.raises(SystemExit):
            main(["compare-periods"])
        err = capsys.readouterr().err
        assert "--period" in err

    def test_one_period_gives_code2(self) -> None:
        """One --period → code 2."""
        rc, err = _run("--period", "2026-01-01..2026-01-07")
        assert rc == 2
        assert "two" in err.lower() or "--period" in err

    def test_three_periods_gives_code2(self) -> None:
        """Three --period → code 2."""
        rc, err = _run(
            "--period", "2026-01-01..2026-01-07",
            "--period", "2026-01-15..2026-01-21",
            "--period", "2026-02-01..2026-02-07",
        )
        assert rc == 2
        assert "two" in err.lower() or "--period" in err


# ---------------------------------------------------------------------------
# 2. Period format: exactly non-empty START..END
# ---------------------------------------------------------------------------

class TestPeriodFormat:
    """Each --period must be exactly START..END with non-empty, parseable parts."""

    def test_malformed_no_separator_gives_code2(self) -> None:
        """'foobar' (no '..') → code 2."""
        rc, _err = _run("--period", "foobar", "--period", "2026-01-15..2026-01-21")
        assert rc == 2

    def test_malformed_no_separator_no_traceback(self) -> None:
        """'foobar' (no '..') → code 2, no traceback."""
        _rc, err = _run("--period", "foobar", "--period", "2026-01-15..2026-01-21")
        assert "Traceback" not in err

    def test_malformed_wrong_separator_gives_code2(self) -> None:
        """'START--END' (wrong separator) → code 2."""
        rc, _err = _run(
            "--period", "2026-01-01--2026-01-07",
            "--period", "2026-01-15..2026-01-21",
        )
        assert rc == 2

    def test_empty_start_gives_code2(self) -> None:
        """'..END' (empty start) → code 2."""
        rc, _err = _run(
            "--period", "..2026-01-07",
            "--period", "2026-01-15..2026-01-21",
        )
        assert rc == 2

    def test_empty_end_gives_code2(self) -> None:
        """'START..' (empty end) → code 2."""
        rc, _err = _run(
            "--period", "2026-01-01..",
            "--period", "2026-01-15..2026-01-21",
        )
        assert rc == 2

    def test_empty_both_gives_code2(self) -> None:
        """'..' (both empty) → code 2."""
        rc, _err = _run("--period", "..", "--period", "2026-01-15..2026-01-21")
        assert rc == 2

    def test_invalid_timestamp_gives_code2(self) -> None:
        """Non-ISO timestamp in either slot → code 2."""
        rc, _err = _run(
            "--period", "not-a-date..2026-01-07",
            "--period", "2026-01-15..also-bad",
        )
        assert rc == 2

    def test_invalid_timestamp_no_traceback(self) -> None:
        """Non-ISO timestamp → code 2, no traceback."""
        _rc, err = _run(
            "--period", "not-a-date..2026-01-07",
            "--period", "2026-01-15..also-bad",
        )
        assert "Traceback" not in err


# ---------------------------------------------------------------------------
# 3. Individual period: start must not be after end
# ---------------------------------------------------------------------------

class TestStartNotAfterEnd:
    """A period where start > end must be rejected with code 2."""

    def test_baseline_start_after_end(self) -> None:
        """Baseline start after end → code 2."""
        rc, _err = _run(
            "--period", "2026-01-10..2026-01-01",
            "--period", "2026-01-15..2026-01-21",
        )
        assert rc == 2

    def test_current_start_after_end(self) -> None:
        """Current start after end → code 2."""
        rc, _err = _run(
            "--period", "2026-01-01..2026-01-07",
            "--period", "2026-01-21..2026-01-15",
        )
        assert rc == 2

    def test_both_start_after_end(self) -> None:
        """Both periods start after end → code 2."""
        rc, _err = _run(
            "--period", "2026-01-10..2026-01-01",
            "--period", "2026-01-21..2026-01-15",
        )
        assert rc == 2


# ---------------------------------------------------------------------------
# 4. Baseline must precede current
# ---------------------------------------------------------------------------

class TestBaselinePrecedesCurrent:
    """The baseline window must end before the current window starts."""

    def test_baseline_after_current(self) -> None:
        """Baseline entirely after current → code 2."""
        rc, _err = _run(
            "--period", "2026-02-01..2026-02-07",
            "--period", "2026-01-01..2026-01-07",
        )
        assert rc == 2

    def test_baseline_same_as_current(self) -> None:
        """Baseline and current are the same window → code 2."""
        rc, _err = _run(
            "--period", "2026-01-01..2026-01-07",
            "--period", "2026-01-01..2026-01-07",
        )
        assert rc == 2


# ---------------------------------------------------------------------------
# 5. Overlapping windows are rejected
# ---------------------------------------------------------------------------

class TestOverlappingWindows:
    """Two windows that share any time must be rejected."""

    def test_current_overlaps_baseline(self) -> None:
        """Current starts before baseline ends → code 2."""
        rc, _err = _run(
            "--period", "2026-01-01..2026-01-10",
            "--period", "2026-01-08..2026-01-20",
        )
        assert rc == 2

    def test_baseline_contains_current(self) -> None:
        """Baseline fully contains current → code 2."""
        rc, _err = _run(
            "--period", "2026-01-01..2026-01-31",
            "--period", "2026-01-10..2026-01-20",
        )
        assert rc == 2

    def test_current_contains_baseline(self) -> None:
        """Current fully contains baseline → code 2."""
        rc, _err = _run(
            "--period", "2026-01-10..2026-01-20",
            "--period", "2026-01-01..2026-01-31",
        )
        assert rc == 2


# ---------------------------------------------------------------------------
# 6. Boundary equality: baseline END == current START
# ---------------------------------------------------------------------------

class TestBoundaryEquality:
    """baseline END == current START is rejected (inclusive bounds → double-count)."""

    def test_baseline_end_equals_current_start(self) -> None:
        """Equal boundary → code 2."""
        rc, _err = _run(
            "--period", "2026-01-01..2026-01-07",
            "--period", "2026-01-07..2026-01-14",
        )
        assert rc == 2

    def test_boundary_gap_at_midnight_is_accepted(self, minimal_db: Path) -> None:
        """baseline END < current START (by 1 second) → accepted (code 0).

        2026-08-07T23:59:59 < 2026-08-15T00:00:00 so these are valid.
        """
        rc, _err = _run(
            "--period", "2026-01-01T00:00:00Z..2026-08-07T23:59:59Z",
            "--period", "2026-08-15T00:00:00Z..2026-12-31T23:59:59Z",
            "--db", str(minimal_db),
        )
        assert rc == 0


# ---------------------------------------------------------------------------
# 7. Valid separated windows are accepted
# ---------------------------------------------------------------------------

class TestValidSeparatedWindows:
    """Well-separated, non-overlapping periods should be accepted (code 0)."""

    def test_well_separated_accepted(self, minimal_db: Path) -> None:
        """Two well-separated periods → code 0."""
        rc, _err = _run(
            "--period", "2026-08-01T00:00:00Z..2026-08-07T23:59:59Z",
            "--period", "2026-08-15T00:00:00Z..2026-08-21T23:59:59Z",
            "--db", str(minimal_db),
        )
        assert rc == 0
