"""Frozen acceptance tests for ``ashiato hygiene`` (issue #36).

The command is a read-only audit over persisted ``tool_calls`` rows that
counts five stable session-hygiene signals in a named, deterministic report:

* ``companion_status_poll`` -- shell calls whose *executed command* invokes
  ``kusabi-companion status``.  Other companion subcommands (``chain-show``,
  ``chain-wait``, bare ``kusabi-companion``) and text that merely quotes the
  command -- in a tool result, in an ``echo``, or in a ``Read`` of a doc --
  are not polls.
* ``host_file_hunt`` -- shell calls that run ``rg``/``grep``/``sed``/``cat``
  against host files.  Calls whose structured tool name is a dedicated
  file/search tool (``Grep``, ``Read``, an MCP search tool) are excluded,
  and hunt words that appear only in a tool result are not a hunt.
* ``raw_local_mcp_http`` -- shell calls that ``curl`` loopback
  (``127.0.0.1``/``localhost``) ports 8750/8765/8770.  Other ports, remote
  hosts (even on a matching port), and dedicated MCP tool calls are excluded.
* ``undo_file_edit`` -- tool calls whose persisted ``tool_name`` is an MCP
  ``undo_file_edit`` tool on any server (``mcp__<server>__undo_file_edit``).
  Prose that merely mentions the name is not a call.
* ``pending_tool_call`` -- every row with ``outcome = 'pending'``, whatever
  its tool name.

The JSON contract: a top-level *object* (never a bare list) with
``coverage`` (``since``/``until`` echoing the effective time bounds, plus
pre-filter ``sessions``/``tool_calls`` counts) and an ordered ``categories``
list of objects with exactly ``name``/``tool_calls``/``sessions``.

These tests run against synthetic transcripts built through the real
``ashiato build`` pipeline (the same route as ``tests/test_nominate.py``),
so the exact counts pin the classification boundaries, the coverage and
time-window semantics, the table output, and the error handling -- all
before the implementation exists.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from ashiato.build import connect
from ashiato.cli import main

CATEGORY_ORDER = (
    "companion_status_poll",
    "host_file_hunt",
    "raw_local_mcp_http",
    "undo_file_edit",
    "pending_tool_call",
)

# The synthetic corpus spans 2026-08-19..2026-08-25 plus one NULL-ts row;
# this window keeps the main rows and drops the early/late/NULL ones.
MAIN_SINCE = "2026-08-20T00:00:00Z"
MAIN_UNTIL = "2026-08-24T00:00:00Z"

# ---------------------------------------------------------------- fixture helpers


def _tool_use(tool_use_id: str, name: str, input_: dict) -> dict:
    return {"type": "tool_use", "id": tool_use_id, "name": name, "input": input_}


def _tool_result(tool_use_id: str, text: str, *, is_error: bool = False) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": [{"type": "text", "text": text}],
        "is_error": is_error,
    }


def _assistant(
    uuid: str, parent: str | None, session_id: str, ts: str | None, blocks: list
) -> dict:
    record = {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": session_id,
        "message": {"role": "assistant", "content": blocks},
    }
    if ts is not None:
        record["timestamp"] = ts
    return record


def _user(uuid: str, parent: str | None, session_id: str, ts: str | None, blocks: list) -> dict:
    record = {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": session_id,
        "message": {"role": "user", "content": blocks},
    }
    if ts is not None:
        record["timestamp"] = ts
    return record


def _call(
    records: list[dict],
    prefix: str,
    session_id: str,
    ts: str | None,
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


def _main_session_a() -> list[dict]:
    """The main session: every category positive and near-miss that fits in
    one shell-heavy session, plus the pending overlaps."""
    records: list[dict] = []

    def t(index: int) -> str:
        return f"2026-08-21T10:{index:02d}:00Z"

    # companion_status_poll positives
    _call(records, "poll1", "ses-a", t(0), "Bash", {"command": "kusabi-companion status"})
    _call(records, "poll2", "ses-a", t(1), "Bash", {"command": "kusabi-companion status --json"})
    # companion near-misses: other subcommands, bare binary, quoted text
    _call(records, "poll3", "ses-a", t(2), "Bash", {"command": "kusabi-companion chain-show 123"})
    _call(records, "poll4", "ses-a", t(3), "Bash", {"command": "kusabi-companion chain-wait 123"})
    _call(records, "poll5", "ses-a", t(4), "Bash", {"command": "kusabi-companion"})
    # "kusabi-companion status" appears only in the *result* of this cat call
    # (which is itself a host_file_hunt positive).
    _call(
        records,
        "poll6",
        "ses-a",
        t(5),
        "Bash",
        {"command": "cat /tmp/notes.md"},
        result_text="usage: kusabi-companion status\nsee the README",
    )
    # the executed command is echo, not kusabi-companion: quoted output only
    _call(records, "poll7", "ses-a", t(6), "Bash", {"command": 'echo "usage: kusabi-companion status"'})
    # host_file_hunt positives: one call per hunt command
    _call(records, "hunt1", "ses-a", t(7), "Bash", {"command": 'rg "TODO" /home/dev/proj'})
    _call(
        records,
        "hunt2",
        "ses-a",
        t(8),
        "Bash",
        {"command": 'grep -rn "api_key" /etc'},
        result_text="grep: /etc/ssl: Permission denied",
        is_error=True,
    )
    _call(records, "hunt3", "ses-a", t(9), "Bash", {"command": "sed -n '1,40p' /etc/hosts"})
    _call(records, "hunt4", "ses-a", t(10), "Bash", {"command": "cat /etc/hosts"})
    # host near-miss: a hunt word only inside the tool result
    _call(
        records,
        "hunt5",
        "ses-a",
        t(11),
        "Bash",
        {"command": "ls -la /home/dev/proj"},
        result_text="total 5\n# grep -rn 'x' . # scratchpad note",
    )
    # raw_local_mcp_http positives: loopback ports 8750/8765/8770
    _call(records, "raw1", "ses-a", t(12), "Bash", {"command": "curl -s http://127.0.0.1:8750/mcp"})
    _call(records, "raw2", "ses-a", t(13), "Bash", {"command": "curl http://localhost:8765/health"})
    _call(
        records,
        "raw3",
        "ses-a",
        t(14),
        "Bash",
        {"command": "curl -v http://127.0.0.1:8770/"},
        result_text="curl: (7) Failed to connect",
        is_error=True,
    )
    # raw near-misses: other port, remote host, remote host on a matching port
    _call(records, "raw4", "ses-a", t(15), "Bash", {"command": "curl http://127.0.0.1:9000/api"})
    _call(records, "raw5", "ses-a", t(16), "Bash", {"command": "curl https://api.github.com/repos/x"})
    _call(records, "raw6", "ses-a", t(17), "Bash", {"command": "curl http://example.com:8750/x"})
    # overlap: a loopback curl with no result counts as raw AND pending
    _call(
        records,
        "raw7",
        "ses-a",
        t(18),
        "Bash",
        {"command": "curl http://localhost:8750/whatever"},
        no_result=True,
    )
    # uncategorised control row
    _call(records, "none1", "ses-a", t(19), "Bash", {"command": "echo done"})
    # undo_file_edit near-miss: prose mentioning the tool name is not a call
    _call(records, "undo4", "ses-a", t(20), "Bash", {"command": 'echo "the undo_file_edit tool reverses edits"'})
    # overlap: a companion poll with no result counts as companion AND pending
    _call(records, "poll8", "ses-a", t(21), "Bash", {"command": "kusabi-companion status"}, no_result=True)
    return records


def _main_other_sessions() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}

    rec: list[dict] = []
    _call(
        rec,
        "undo1",
        "ses-b",
        "2026-08-21T11:00:00Z",
        "mcp__sunaba__undo_file_edit",
        {"file_path": "/home/dev/proj/src/a.py", "steps": 1},
    )
    out["ses-b"] = rec

    rec = []
    _call(
        rec,
        "undo2",
        "ses-c",
        "2026-08-21T11:01:00Z",
        "mcp__kaiba__undo_file_edit",
        {"file_path": "/home/dev/proj/src/b.py"},
    )
    out["ses-c"] = rec

    # host near-miss: a dedicated MCP search tool is not a shell hunt, even
    # though its pattern carries the word "sed".
    rec = []
    _call(rec, "hunt6", "ses-d", "2026-08-21T11:02:00Z", "mcp__sunaba__search", {"pattern": "sed"})
    out["ses-d"] = rec

    # host near-miss: the builtin Grep tool is a dedicated search tool, even
    # though its pattern is the word "cat".
    rec = []
    _call(rec, "hunt7", "ses-e", "2026-08-21T11:03:00Z", "Grep", {"pattern": "cat"})
    out["ses-e"] = rec

    # near-miss for companion AND host: Read is a dedicated file tool; the
    # quoted text lives only in its result.
    rec = []
    _call(
        rec,
        "read1",
        "ses-f",
        "2026-08-21T11:04:00Z",
        "Read",
        {"file_path": "/etc/hosts"},
        result_text="127.0.0.1 localhost\n# run kusabi-companion status to poll\n# cat /etc/hosts to inspect",
    )
    out["ses-f"] = rec

    # pending independent of tool name: an MCP call with no result
    rec = []
    _call(rec, "pend1", "ses-g", "2026-08-21T11:05:00Z", "mcp__foo__bar", {"whatever": 1}, no_result=True)
    out["ses-g"] = rec

    # raw near-miss: a dedicated MCP tool call even though its input carries
    # the loopback URL.
    rec = []
    _call(
        rec,
        "raw8",
        "ses-h",
        "2026-08-21T11:06:00Z",
        "mcp__sunaba__http_fetch",
        {"url": "http://127.0.0.1:8750/sse"},
    )
    out["ses-h"] = rec

    # NULL timestamp: in coverage when no bound is given, excluded with one
    rec = []
    _call(rec, "null1", "ses-null", None, "Bash", {"command": "cat /etc/hostname"})
    out["ses-null"] = rec

    rec = []
    _call(rec, "early1", "ses-early", "2026-08-19T12:00:00Z", "Bash", {"command": "kusabi-companion status"})
    out["ses-early"] = rec

    rec = []
    _call(rec, "late1", "ses-late", "2026-08-25T12:00:00Z", "Bash", {"command": "kusabi-companion status"})
    out["ses-late"] = rec

    return out


def _write_sessions(directory: Path, sessions: dict[str, list[dict]]) -> None:
    for session_id, records in sessions.items():
        path = directory / f"{session_id}.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


@pytest.fixture
def hygiene_db(tmp_path: Path) -> Path:
    """The full synthetic corpus: 32 rows across 11 sessions."""
    directory = tmp_path / "transcripts"
    directory.mkdir()
    sessions = {"ses-a": _main_session_a()}
    sessions.update(_main_other_sessions())
    _write_sessions(directory, sessions)
    db_path = tmp_path / "hygiene.duckdb"
    assert main(["build", "--source", str(directory), "--db", str(db_path)]) == 0
    return db_path


@pytest.fixture
def window_db(tmp_path: Path) -> Path:
    """Three cat calls: one at each window edge and one with a NULL ts."""
    directory = tmp_path / "window"
    directory.mkdir()
    sessions: dict[str, list[dict]] = {}
    _call(sessions.setdefault("ses-w1", []), "w1", "ses-w1", "2026-08-20T12:00:00Z", "Bash", {"command": "cat /etc/hosts"})
    _call(sessions.setdefault("ses-w2", []), "w2", "ses-w2", "2026-08-21T12:00:00Z", "Bash", {"command": "cat /etc/hosts"})
    _call(sessions.setdefault("ses-w3", []), "w3", "ses-w3", None, "Bash", {"command": "cat /etc/hostname"})
    _write_sessions(directory, sessions)
    db_path = tmp_path / "window.duckdb"
    assert main(["build", "--source", str(directory), "--db", str(db_path)]) == 0
    return db_path


def _audit(capsys: pytest.CaptureFixture[str], db: Path, *extra: str) -> tuple[int, dict]:
    rc = main(["hygiene", "--db", str(db), "--format", "json", *extra])
    return rc, json.loads(capsys.readouterr().out)


def _category(payload: dict, name: str) -> dict:
    return next(cat for cat in payload["categories"] if cat["name"] == name)


# ---------------------------------------------------------------- JSON contract


def test_json_is_an_object_with_a_stable_shape(hygiene_db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The top level is an object with exactly coverage + categories; category
    objects carry exactly name/tool_calls/sessions, in the frozen order."""
    rc, payload = _audit(capsys, hygiene_db)
    assert rc == 0
    assert set(payload) == {"coverage", "categories"}
    assert set(payload["coverage"]) == {"since", "until", "sessions", "tool_calls"}
    assert [cat["name"] for cat in payload["categories"]] == list(CATEGORY_ORDER)
    for cat in payload["categories"]:
        assert set(cat) == {"name", "tool_calls", "sessions"}
        assert isinstance(cat["tool_calls"], int)
        assert isinstance(cat["sessions"], int)


def test_coverage_without_bounds_counts_everything(hygiene_db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """No bounds: all 32 rows and 11 sessions count, including the NULL-ts row,
    and since/until are null because no window was requested."""
    rc, payload = _audit(capsys, hygiene_db)
    assert rc == 0
    coverage = payload["coverage"]
    assert coverage["since"] is None
    assert coverage["until"] is None
    assert coverage["tool_calls"] == 32
    assert coverage["sessions"] == 11


def test_coverage_with_bounds_counts_only_the_window(
    hygiene_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With --since/--until the NULL-ts row and the two out-of-window rows drop
    out of coverage, and the requested window is echoed back."""
    rc, payload = _audit(capsys, hygiene_db, "--since", MAIN_SINCE, "--until", MAIN_UNTIL)
    assert rc == 0
    coverage = payload["coverage"]
    assert datetime.fromisoformat(coverage["since"]) == datetime(2026, 8, 20)
    assert datetime.fromisoformat(coverage["until"]) == datetime(2026, 8, 24)
    assert coverage["tool_calls"] == 29
    assert coverage["sessions"] == 8


def test_coverage_metadata_comes_before_category_filtering(
    hygiene_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Rows that match no category still count in coverage: 32 rows are in the
    database, the five categories together account for 20 of them."""
    rc, payload = _audit(capsys, hygiene_db)
    assert rc == 0
    assert payload["coverage"]["tool_calls"] == 32
    counted = sum(cat["tool_calls"] for cat in payload["categories"])
    assert counted < payload["coverage"]["tool_calls"]


# ---------------------------------------------------------------- categories


def test_companion_status_poll_counts_only_invocations(
    hygiene_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Positive: poll1, poll2, poll8, early1, late1 (5 calls, 3 sessions).
    Near-misses: chain-show, chain-wait, bare kusabi-companion, a result that
    quotes the text, an echo that quotes the text, and a Read of a doc."""
    rc, payload = _audit(capsys, hygiene_db)
    assert rc == 0
    cat = _category(payload, "companion_status_poll")
    assert (cat["tool_calls"], cat["sessions"]) == (5, 3)


def test_host_file_hunt_counts_only_shell_hunts(
    hygiene_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Positive: one call per hunt command (rg, grep, sed, cat) plus a second
    cat and the NULL-ts cat (6 calls, 2 sessions -- the second session is the
    NULL-ts one).  Near-misses: a result that quotes a hunt word, the Grep
    builtin, the Read builtin, and an MCP search tool."""
    rc, payload = _audit(capsys, hygiene_db)
    assert rc == 0
    cat = _category(payload, "host_file_hunt")
    assert (cat["tool_calls"], cat["sessions"]) == (6, 2)


def test_raw_local_mcp_http_counts_only_loopback_curl(
    hygiene_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Positive: 127.0.0.1:8750, localhost:8765, 127.0.0.1:8770, and the
    pending localhost:8750 call (4 calls, 1 session).  Near-misses: another
    loopback port, a remote https host, a remote host on port 8750, and an
    MCP tool call whose input carries the loopback URL."""
    rc, payload = _audit(capsys, hygiene_db)
    assert rc == 0
    cat = _category(payload, "raw_local_mcp_http")
    assert (cat["tool_calls"], cat["sessions"]) == (4, 1)


def test_undo_file_edit_counts_mcp_calls_by_tool_name(
    hygiene_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Positive: mcp__sunaba__undo_file_edit and mcp__kaiba__undo_file_edit
    (2 calls, 2 sessions -- the server prefix may vary).  Near-miss: a shell
    command that merely mentions the tool name."""
    rc, payload = _audit(capsys, hygiene_db)
    assert rc == 0
    cat = _category(payload, "undo_file_edit")
    assert (cat["tool_calls"], cat["sessions"]) == (2, 2)


def test_pending_tool_call_counts_every_pending_row(
    hygiene_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Positive: raw7 (Bash curl), poll8 (Bash status), pend1 (mcp__foo__bar)
    -- 3 calls, 2 sessions, independent of tool name.  The two Bash rows also
    belong to their pattern categories, so rows may overlap categories."""
    rc, payload = _audit(capsys, hygiene_db)
    assert rc == 0
    cat = _category(payload, "pending_tool_call")
    assert (cat["tool_calls"], cat["sessions"]) == (3, 2)


def test_overlapping_rows_count_in_every_matching_category(
    hygiene_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The pending curl row and the pending status row each count in two
    categories; category counts are not made mutually exclusive."""
    rc, payload = _audit(capsys, hygiene_db)
    assert rc == 0
    raw = _category(payload, "raw_local_mcp_http")
    companion = _category(payload, "companion_status_poll")
    pending = _category(payload, "pending_tool_call")
    assert raw["tool_calls"] == 4
    assert companion["tool_calls"] == 5
    assert pending["tool_calls"] == 3


# ---------------------------------------------------------------- time window


@pytest.mark.parametrize(
    ("flags", "expected_calls"),
    [
        # exactly on the edge timestamps: both bounds are inclusive
        (("--since", "2026-08-20T12:00:00Z"), 2),
        (("--until", "2026-08-21T12:00:00Z"), 2),
        # one second outside an edge excludes that row
        (("--since", "2026-08-20T12:00:01Z"), 1),
        (("--until", "2026-08-20T12:00:01Z"), 1),
        # a degenerate window still includes the row exactly on both bounds
        (("--since", "2026-08-21T12:00:00Z", "--until", "2026-08-21T12:00:00Z"), 1),
    ],
)
def test_window_bounds_are_inclusive(
    window_db: Path,
    capsys: pytest.CaptureFixture[str],
    flags: tuple[str, ...],
    expected_calls: int,
) -> None:
    rc, payload = _audit(capsys, window_db, *flags)
    assert rc == 0
    assert payload["coverage"]["tool_calls"] == expected_calls
    host = _category(payload, "host_file_hunt")
    assert host["tool_calls"] == expected_calls
    assert host["sessions"] == expected_calls


def test_null_timestamp_rows_included_without_bounds_and_excluded_with(
    window_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc, payload = _audit(capsys, window_db)
    assert rc == 0
    assert payload["coverage"]["tool_calls"] == 3
    assert payload["coverage"]["sessions"] == 3
    host = _category(payload, "host_file_hunt")
    assert (host["tool_calls"], host["sessions"]) == (3, 3)

    rc, payload = _audit(capsys, window_db, "--since", "2026-08-20T12:00:00Z")
    assert rc == 0
    assert payload["coverage"]["tool_calls"] == 2
    assert payload["coverage"]["sessions"] == 2
    host = _category(payload, "host_file_hunt")
    assert (host["tool_calls"], host["sessions"]) == (2, 2)


# ---------------------------------------------------------------- table output


def test_table_output_lists_the_five_categories_with_both_counts(
    hygiene_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The default (table) format shows the same five rows with tool-call and
    session counts; exact spacing and column order are not part of the
    contract, so each row must simply carry its two counts."""
    rc = main(["hygiene", "--db", str(hygiene_db)])
    out = capsys.readouterr().out
    assert rc == 0
    expected = {
        "companion_status_poll": ("5", "3"),
        "host_file_hunt": ("6", "2"),
        "raw_local_mcp_http": ("4", "1"),
        "undo_file_edit": ("2", "2"),
        "pending_tool_call": ("3", "2"),
    }
    rows: dict[str, list[str]] = {}
    for line in out.splitlines():
        tokens = line.split()
        if tokens and tokens[0] in CATEGORY_ORDER:
            rows[tokens[0]] = tokens[1:]
    assert set(rows) == set(expected)
    for name, (calls, sessions) in expected.items():
        assert calls in rows[name]
        assert sessions in rows[name]


# ---------------------------------------------------------------- error handling


def test_hygiene_on_a_missing_database_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["hygiene", "--db", str(tmp_path / "nope.duckdb")]) == 1
    assert "no database at" in capsys.readouterr().err


def test_hygiene_on_an_outdated_database_names_the_fix(
    hygiene_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An older database (no view, no input_summary column) is refused with
    the rebuild hint, exactly like the other read commands."""
    connection = connect(hygiene_db)
    try:
        connection.execute("DROP VIEW denial_followups")
        connection.execute("ALTER TABLE tool_calls DROP COLUMN input_summary")
    finally:
        connection.close()
    assert main(["hygiene", "--db", str(hygiene_db)]) == 1
    err = capsys.readouterr().err
    assert "input_summary" in err
    assert "delete the database file and build again" in err


# ---------------------------------------------------------------- read-only


def test_hygiene_does_not_modify_the_database(
    hygiene_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = hygiene_db.stat().st_mtime_ns
    rc, _ = _audit(capsys, hygiene_db)
    assert rc == 0
    assert hygiene_db.stat().st_mtime_ns == before