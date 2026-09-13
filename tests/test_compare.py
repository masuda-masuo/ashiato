"""Frozen acceptance tests for ``ashiato compare-periods`` (issue #39).

``compare-periods`` runs the hygiene audit over two non-overlapping time
windows and emits a JSON object whose ``categories`` list carries one
flat row per hygiene category.  Each row has:

* ``name``
* ``baseline_tool_calls``, ``current_tool_calls``
* ``baseline_sessions``, ``current_sessions``
* ``baseline_calls_per_session``, ``current_calls_per_session``
* ``tool_calls_change``, ``tool_calls_percent_change``, ``sessions_change``

CLI contract::

    compare-periods --period START..END --period START..END --db PATH --format json

The first ``--period`` is the baseline window, the second is the current
window.

``calls_per_session`` and ``percent_change`` are rounded to two decimal
places.  When the denominator is zero (zero sessions or zero baseline
tool_calls) the corresponding value is ``null``.

These tests build a compact synthetic corpus with two clearly-separated
time windows and pin the exact expected counts.  Baseline-red is expected
because the production CLI does not yet implement the ``--period`` flag
or the flat-key output contract.
"""

from __future__ import annotations

import json
from datetime import datetime
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
# Synthetic fixture: two windows with distinct, known call patterns
#
# Baseline window: 2026-08-01T00:00:00Z .. 2026-08-07T23:59:59Z
#   - 3 companion_status_poll calls (session baseline-a)
#   - 1 host_file_hunt call    (session baseline-a)
#   - 0 raw_local_mcp_http     (nothing)
#   - 0 undo_file_edit          (nothing)
#   - 2 pending_tool_call calls (2 calls with no result, session baseline-a)
#
# Current window: 2026-08-15T00:00:00Z .. 2026-08-21T23:59:59Z
#   - 2 companion_status_poll calls (session current-a)  -> decrease
#   - 4 host_file_hunt calls    (session current-a)      -> increase
#   - 1 raw_local_mcp_http call (session current-a)      -> increase from 0
#   - 1 undo_file_edit call     (session current-a)      -> increase from 0
#   - 3 pending_tool_call calls (2 in current-a + 1 in current-b)
#
# Totals:
#   baseline coverage: 6 tool_calls, 1 session
#   current  coverage: 11 tool_calls, 2 sessions
# ---------------------------------------------------------------------------


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
    is_error: bool = False,
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
                [_tool_result(tool_use_id, result_text, is_error=is_error)],
            )
        )


# ---- baseline window (2026-08-01 .. 2026-08-07) ----


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


# ---- current window (2026-08-15 .. 2026-08-21) ----


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

    # 2x pending_tool_call (no result) -- two distinct pending calls
    _call(records, "cp3", "current-a", t(8), "mcp__foo__bar", {"x": 1}, no_result=True)
    _call(records, "cp5", "current-a", t(9), "mcp__foo__bar", {"x": 2}, no_result=True)

    return records


def _current_session_b() -> list[dict]:
    records: list[dict] = []
    # 1x pending_tool_call (no result)
    _call(records, "cp4", "current-b", "2026-08-18T11:00:00Z", "mcp__baz__qux", {"y": 2}, no_result=True)
    return records


def _write_sessions(directory: Path, sessions: dict[str, list[dict]]) -> None:
    for session_id, records in sessions.items():
        path = directory / f"{session_id}.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


@pytest.fixture
def compare_db(tmp_path: Path) -> Path:
    """Synthetic DB with two well-separated windows: baseline and current."""
    directory = tmp_path / "transcripts"
    directory.mkdir()
    sessions: dict[str, list[dict]] = {
        "baseline-a": _baseline_session(),
        "current-a": _current_session_a(),
        "current-b": _current_session_b(),
    }
    _write_sessions(directory, sessions)
    db_path = tmp_path / "compare.duckdb"
    assert main(["build", "--source", str(directory), "--db", str(db_path)]) == 0
    return db_path


# ---------------------------------------------------------------------------
# Expected counts (hand-verified from the fixtures above)
# ---------------------------------------------------------------------------

BASELINE_SINCE = "2026-08-01T00:00:00Z"
BASELINE_UNTIL = "2026-08-07T23:59:59Z"
CURRENT_SINCE = "2026-08-15T00:00:00Z"
CURRENT_UNTIL = "2026-08-21T23:59:59Z"

# Per-category expected values -- flat keys as the adopted contract requires.
EXPECTED_CATEGORIES: dict[str, dict[str, object]] = {
    "companion_status_poll": {
        "baseline_tool_calls": 3, "current_tool_calls": 2,
        "baseline_sessions": 1, "current_sessions": 1,
        "baseline_calls_per_session": 3.00, "current_calls_per_session": 2.00,
        "tool_calls_change": -1, "tool_calls_percent_change": -33.33,
        "sessions_change": 0,
    },
    "host_file_hunt": {
        "baseline_tool_calls": 1, "current_tool_calls": 4,
        "baseline_sessions": 1, "current_sessions": 1,
        "baseline_calls_per_session": 1.00, "current_calls_per_session": 4.00,
        "tool_calls_change": 3, "tool_calls_percent_change": 300.00,
        "sessions_change": 0,
    },
    "raw_local_mcp_http": {
        "baseline_tool_calls": 0, "current_tool_calls": 1,
        "baseline_sessions": 0, "current_sessions": 1,
        "baseline_calls_per_session": None, "current_calls_per_session": 1.00,
        "tool_calls_change": 1, "tool_calls_percent_change": None,
        "sessions_change": 1,
    },
    "undo_file_edit": {
        "baseline_tool_calls": 0, "current_tool_calls": 1,
        "baseline_sessions": 0, "current_sessions": 1,
        "baseline_calls_per_session": None, "current_calls_per_session": 1.00,
        "tool_calls_change": 1, "tool_calls_percent_change": None,
        "sessions_change": 1,
    },
    "pending_tool_call": {
        "baseline_tool_calls": 2, "current_tool_calls": 3,
        "baseline_sessions": 1, "current_sessions": 2,
        "baseline_calls_per_session": 2.00, "current_calls_per_session": 1.50,
        "tool_calls_change": 1, "tool_calls_percent_change": 50.00,
        "sessions_change": 1,
    },
}

# The exact flat-key set every category row must carry.
FLAT_KEYS = {
    "name",
    "baseline_tool_calls", "current_tool_calls",
    "baseline_sessions", "current_sessions",
    "baseline_calls_per_session", "current_calls_per_session",
    "tool_calls_change", "tool_calls_percent_change", "sessions_change",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _compare(capsys: pytest.CaptureFixture[str], db: Path) -> tuple[int, dict]:
    """Run ``compare-periods`` in JSON mode and return (exit_code, parsed_json).

    ``main`` may ``sys.exit(2)`` if the command is not registered; we catch
    that so the test gets a clean (2, {}) tuple rather than an uncaught crash.
    """
    try:
        rc = main([
            "compare-periods",
            "--period", f"{BASELINE_SINCE}..{BASELINE_UNTIL}",
            "--period", f"{CURRENT_SINCE}..{CURRENT_UNTIL}",
            "--db", str(db),
            "--format", "json",
        ])
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 1
    out = capsys.readouterr().out
    try:
        payload = json.loads(out)
    except json.JSONDecodeError:
        payload = {}
    return rc, payload


def _cat(payload: dict, name: str) -> dict:
    """Return the category row with *name* from *payload*."""
    return next(c for c in payload["categories"] if c["name"] == name)


# ---------------------------------------------------------------------------
# 1. Top-level structure
# ---------------------------------------------------------------------------


def test_json_top_level_has_categories(
    compare_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The top-level object has a ``categories`` list."""
    rc, payload = _compare(capsys, compare_db)
    assert rc == 0
    assert "categories" in payload
    assert isinstance(payload["categories"], list)


# ---------------------------------------------------------------------------
# 2. Per-category exact flat keys
# ---------------------------------------------------------------------------


def test_each_category_has_exact_flat_keys(
    compare_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every category object carries exactly the adopted flat keys -- no more, no fewer."""
    rc, payload = _compare(capsys, compare_db)
    assert rc == 0
    for cat_row in payload["categories"]:
        assert set(cat_row) == FLAT_KEYS, f"wrong keys for {cat_row.get('name')}"


# ---------------------------------------------------------------------------
# 3. Categories are in frozen order
# ---------------------------------------------------------------------------


def test_categories_are_in_frozen_order(
    compare_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ``categories`` list preserves the hygiene category order."""
    rc, payload = _compare(capsys, compare_db)
    assert rc == 0
    names = [c["name"] for c in payload["categories"]]
    assert names == list(CATEGORY_ORDER)


# ---------------------------------------------------------------------------
# 4. Baseline and current counts match the fixtures
# ---------------------------------------------------------------------------


def test_baseline_counts_match_fixture(
    compare_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Baseline tool_calls and sessions for every category match the fixture."""
    rc, payload = _compare(capsys, compare_db)
    assert rc == 0
    for name, expected in EXPECTED_CATEGORIES.items():
        row = _cat(payload, name)
        assert row["baseline_tool_calls"] == expected["baseline_tool_calls"], f"baseline {name} tool_calls"
        assert row["baseline_sessions"] == expected["baseline_sessions"], f"baseline {name} sessions"


def test_current_counts_match_fixture(
    compare_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Current tool_calls and sessions for every category match the fixture."""
    rc, payload = _compare(capsys, compare_db)
    assert rc == 0
    for name, expected in EXPECTED_CATEGORIES.items():
        row = _cat(payload, name)
        assert row["current_tool_calls"] == expected["current_tool_calls"], f"current {name} tool_calls"
        assert row["current_sessions"] == expected["current_sessions"], f"current {name} sessions"


# ---------------------------------------------------------------------------
# 5. Change arithmetic
# ---------------------------------------------------------------------------


def test_tool_calls_change_is_difference(
    compare_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``tool_calls_change`` = current_tool_calls - baseline_tool_calls."""
    rc, payload = _compare(capsys, compare_db)
    assert rc == 0
    for name, expected in EXPECTED_CATEGORIES.items():
        row = _cat(payload, name)
        assert row["tool_calls_change"] == expected["tool_calls_change"], f"tool_calls_change for {name}"


def test_sessions_change_is_difference(
    compare_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``sessions_change`` = current_sessions - baseline_sessions."""
    rc, payload = _compare(capsys, compare_db)
    assert rc == 0
    for name, expected in EXPECTED_CATEGORIES.items():
        row = _cat(payload, name)
        assert row["sessions_change"] == expected["sessions_change"], f"sessions_change for {name}"


# ---------------------------------------------------------------------------
# 6. Percent change: increase, decrease, and zero-baseline -> null
# ---------------------------------------------------------------------------


def test_increase_decrease_and_zero_baseline_percent(
    compare_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Percent change covers increase, decrease, and zero-baseline null.

    * ``host_file_hunt`` increases (1 -> 4, +300.00%).
    * ``companion_status_poll`` decreases (3 -> 2, -33.33%).
    * ``raw_local_mcp_http`` goes from 0 -> 1, percent must be ``null``.
    * ``undo_file_edit`` goes from 0 -> 1, percent must be ``null``.
    """
    rc, payload = _compare(capsys, compare_db)
    assert rc == 0

    # increase
    hunt = _cat(payload, "host_file_hunt")
    assert hunt["tool_calls_change"] == 3
    assert hunt["tool_calls_percent_change"] == pytest.approx(300.00, rel=1e-4)

    # decrease
    poll = _cat(payload, "companion_status_poll")
    assert poll["tool_calls_change"] == -1
    assert poll["tool_calls_percent_change"] == pytest.approx(-33.33, rel=1e-4)

    # zero baseline -> null percent
    raw = _cat(payload, "raw_local_mcp_http")
    assert raw["baseline_tool_calls"] == 0
    assert raw["tool_calls_percent_change"] is None

    undo = _cat(payload, "undo_file_edit")
    assert undo["baseline_tool_calls"] == 0
    assert undo["tool_calls_percent_change"] is None


# ---------------------------------------------------------------------------
# 7. calls_per_session rounding and null edge
# ---------------------------------------------------------------------------


def test_calls_per_session_rounded_to_two_decimals(
    compare_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both baseline and current calls_per_session are rounded to two decimals."""
    rc, payload = _compare(capsys, compare_db)
    assert rc == 0
    for name, expected in EXPECTED_CATEGORIES.items():
        row = _cat(payload, name)
        b_expected = expected["baseline_calls_per_session"]
        c_expected = expected["current_calls_per_session"]
        if b_expected is not None:
            assert row["baseline_calls_per_session"] == pytest.approx(b_expected, rel=1e-4), \
                f"baseline calls_per_session for {name}"
        else:
            assert row["baseline_calls_per_session"] is None, \
                f"baseline calls_per_session for {name} should be null"
        if c_expected is not None:
            assert row["current_calls_per_session"] == pytest.approx(c_expected, rel=1e-4), \
                f"current calls_per_session for {name}"
        else:
            assert row["current_calls_per_session"] is None, \
                f"current calls_per_session for {name} should be null"


def test_zero_sessions_yields_null_calls_per_session(
    compare_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When a period has zero sessions the calls_per_session is null."""
    rc, payload = _compare(capsys, compare_db)
    assert rc == 0

    # raw_local_mcp_http: baseline has 0 sessions
    raw = _cat(payload, "raw_local_mcp_http")
    assert raw["baseline_sessions"] == 0
    assert raw["baseline_calls_per_session"] is None

    # undo_file_edit: baseline has 0 sessions
    undo = _cat(payload, "undo_file_edit")
    assert undo["baseline_sessions"] == 0
    assert undo["baseline_calls_per_session"] is None


# ---------------------------------------------------------------------------
# 8. Per-period audit counts match hygiene.audit output
# ---------------------------------------------------------------------------


def test_baseline_window_matches_hygiene_audit(
    compare_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Running ``hygiene`` for the baseline window returns the same
    tool_calls/sessions as ``compare-periods`` reports."""
    # get compare-periods result
    rc_cmp, payload_cmp = _compare(capsys, compare_db)
    assert rc_cmp == 0

    # get hygiene result for the baseline window
    from ashiato.build import connect
    from ashiato.hygiene import audit as hygiene_audit

    conn = connect(compare_db)
    try:
        baseline = hygiene_audit(
            conn,
            since=datetime.fromisoformat(BASELINE_SINCE.replace("Z", "+00:00")),
            until=datetime.fromisoformat(BASELINE_UNTIL.replace("Z", "+00:00")),
        )
    finally:
        conn.close()

    for cat_row in payload_cmp["categories"]:
        name = cat_row["name"]
        hyg_cat = next(c for c in baseline["categories"] if c["name"] == name)
        assert cat_row["baseline_tool_calls"] == hyg_cat["tool_calls"], f"baseline {name}"
        assert cat_row["baseline_sessions"] == hyg_cat["sessions"], f"baseline {name}"


def test_current_window_matches_hygiene_audit(
    compare_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Running ``hygiene`` for the current window returns the same
    tool_calls/sessions as ``compare-periods`` reports."""
    # get compare-periods result
    rc_cmp, payload_cmp = _compare(capsys, compare_db)
    assert rc_cmp == 0

    # get hygiene result for the current window
    from ashiato.build import connect
    from ashiato.hygiene import audit as hygiene_audit

    conn = connect(compare_db)
    try:
        current = hygiene_audit(
            conn,
            since=datetime.fromisoformat(CURRENT_SINCE.replace("Z", "+00:00")),
            until=datetime.fromisoformat(CURRENT_UNTIL.replace("Z", "+00:00")),
        )
    finally:
        conn.close()

    for cat_row in payload_cmp["categories"]:
        name = cat_row["name"]
        hyg_cat = next(c for c in current["categories"] if c["name"] == name)
        assert cat_row["current_tool_calls"] == hyg_cat["tool_calls"], f"current {name}"
        assert cat_row["current_sessions"] == hyg_cat["sessions"], f"current {name}"


# ---------------------------------------------------------------------------
# 9. Empty valid period: zero coverage, null derived ratios/percent
# ---------------------------------------------------------------------------


def test_empty_valid_period_reports_zero_coverage(
    compare_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When a valid period contains no tool calls, all counts are zero
    and derived values are null.

    Periods in 2020 are well before any session data (all in August 2026),
    so both periods are empty.  This tests the contract:
    * ``baseline_tool_calls`` == 0, ``baseline_sessions`` == 0
    * ``baseline_calls_per_session`` is ``null`` (zero sessions)
    * ``current_tool_calls`` == 0, ``current_sessions`` == 0
    * ``current_calls_per_session`` is ``null`` (zero sessions)
    * ``tool_calls_percent_change`` is ``null`` (zero baseline)
    * ``tool_calls_change`` == 0, ``sessions_change`` == 0
    """
    try:
        rc = main([
            "compare-periods",
            "--period", "2020-01-01T00:00:00Z..2020-01-07T23:59:59Z",
            "--period", "2020-01-15T00:00:00Z..2020-01-21T23:59:59Z",
            "--db", str(compare_db),
            "--format", "json",
        ])
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 1
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert rc == 0
    assert "categories" in payload
    for cat_row in payload["categories"]:
        # Zero coverage for both periods
        assert cat_row["baseline_tool_calls"] == 0
        assert cat_row["baseline_sessions"] == 0
        assert cat_row["current_tool_calls"] == 0
        assert cat_row["current_sessions"] == 0
        # Derived values should be null (denominator is zero)
        assert cat_row["baseline_calls_per_session"] is None
        assert cat_row["current_calls_per_session"] is None
        assert cat_row["tool_calls_percent_change"] is None
        # Changes are zero
        assert cat_row["tool_calls_change"] == 0
        assert cat_row["sessions_change"] == 0


# ---------------------------------------------------------------------------
# 10. Tool calls with NULL timestamps are excluded from period coverage
#     and category counts
# ---------------------------------------------------------------------------


def _session_with_null_timestamp() -> list[dict]:
    """Build a session where one tool call has a NULL timestamp.

    The session has two tool calls:
    * one with a valid timestamp in the baseline window  (should be counted)
    * one with ``timestamp: null``                       (must NOT be counted)
    """
    records: list[dict] = []
    # First tool call — valid timestamp
    _call(records, "nt1", "sess-null-ts", "2026-08-03T10:00:00Z",
          "Bash", {"command": "rg 'TODO' /home/dev"})
    # Second tool call — NULL timestamp (manually constructed)
    records.append({
        "type": "assistant",
        "uuid": "nt2_call",
        "parentUuid": records[-1]["uuid"],
        "sessionId": "sess-null-ts",
        "timestamp": None,
        "message": {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "nt2_use", "name": "Bash",
                 "input": {"command": "rg 'FIXME' /home/dev"}},
            ],
        },
    })
    return records


@pytest.fixture
def compare_db_null_ts(tmp_path: Path) -> Path:
    """DB with a session containing both normal and NULL-timestamp tool calls."""
    directory = tmp_path / "transcripts"
    directory.mkdir()
    sessions: dict[str, list[dict]] = {
        "sess-null-ts": _session_with_null_timestamp(),
    }
    _write_sessions(directory, sessions)
    db_path = tmp_path / "null_ts.duckdb"
    assert main(["build", "--source", str(directory), "--db", str(db_path)]) == 0
    return db_path


def test_null_timestamp_tool_calls_excluded(
    compare_db_null_ts: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Tool calls with NULL timestamps are excluded from both period
    coverage and category counts.

    The ``sess-null-ts`` session has two ``Bash`` tool calls (classified
    as ``host_file_hunt``): one with a valid timestamp and one with
    ``timestamp: null``.  The query ``WHERE ts >= ? AND ts <= ?``
    excludes the NULL-timestamp row, so:
    * baseline coverage tool_calls must be 1 (not 2)
    * baseline coverage sessions must be 1 (the session is still
      reachable via the valid-timestamp call)
    * ``host_file_hunt`` baseline_tool_calls must be 1
    * ``host_file_hunt`` baseline_sessions must be 1
    """
    try:
        rc = main([
            "compare-periods",
            "--period", "2026-08-01T00:00:00Z..2026-08-07T23:59:59Z",
            "--period", "2026-08-15T00:00:00Z..2026-08-21T23:59:59Z",
            "--db", str(compare_db_null_ts),
            "--format", "json",
        ])
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 1
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert rc == 0

    # Period coverage: only 1 tool call (the one with a valid timestamp)
    baseline = payload["periods"]["baseline"]
    assert baseline["tool_calls"] == 1, (
        f"NULL-timestamp tool call must not be counted in coverage: {baseline}"
    )
    assert baseline["sessions"] == 1

    # Category counts: host_file_hunt must show 1 tool call (not 2)
    hunt = next(c for c in payload["categories"] if c["name"] == "host_file_hunt")
    assert hunt["baseline_tool_calls"] == 1, (
        f"NULL-timestamp tool call must not appear in category counts: {hunt}"
    )
    assert hunt["baseline_sessions"] == 1
