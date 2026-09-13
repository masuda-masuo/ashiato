"""Frozen contract for ``ashiato session-trace`` (issue #40).

A deterministic, read-only, single-session timeline: interleaved text events
and tool calls ordered by ``(seq, kind tie-breaker, stable id)``, with
excerpt/row limits, recall and denial follow-up annotation, and a JSON shape
(``session`` / ``coverage`` / ``timeline``) stable enough to wrap in MCP
later.

The contract decisions frozen here, where the brief leaves room:

* ordering within one ``seq``: text rows before tool rows, then by stable id
  (``event_id`` / ``tool_use_id``) ascending -- a Claude assistant line emits
  its text before its ``tool_use`` blocks;
* ``--limit 0`` means "all rows", mirroring ``denials`` / ``recalls`` /
  ``grep``; a positive limit applies *after* ordering and ``coverage.total``
  keeps the pre-limit count;
* ``--max-excerpt-chars 0`` means "uncapped", the same 0 = no cap convention;
  a negative value is rejected by argparse;
* ``coverage`` carries the availability of each persisted source table
  (``has_sessions`` / ``has_events`` / ``has_tool_calls`` /
  ``has_recall_calls``), so the tool-call-only Codex persisted shape is
  visible;
* ``ts`` is rendered ``YYYY-MM-DD HH:MM:SS`` (naive UTC), the convention the
  other read commands' JSON output already uses;
* the excerpt marker is at most one character: a truncated excerpt is the
  first N characters of the source text plus an optional one-character
  marker, so ``full.startswith(excerpt[:N])`` and ``len(excerpt) <= N + 1``
  hold, and the row's ``truncated`` flag is ``len(full) > N``;
* session resolution considers ``sessions`` and ``tool_calls`` (so a session
  whose rows were persisted only as tool calls still resolves);
* missing / outdated database handling matches the other read commands:
  exit 1 with the standard messages.

All fixtures are synthetic transcripts built through the real ``build``
pipeline into a private ``tmp_path`` database; the host production database
is never touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ashiato.build import connect
from ashiato.cli import main

MAIN = "AAAA0001"
OTHER = "AAAA9999"
CODEX = "CODX0001"

#: Must start with one of parser.DENIAL_PATTERNS so outcome == "denied".
DENIAL_TEXT = "The user doesn't want to proceed with this tool use. The tool call was rejected."

#: The full text behind every timeline row of MAIN (None = no result at all,
#: the pending call).  The fixture is synthetic, so the tests know exactly
#: what the persisted rows contain.
FULL_TEXT: dict[str, str | None] = {
    "u1": "Investigate the trace.",
    "u2": "I'll start.",
    "toolu_ok_1": "README.md\nsrc",
    "toolu_par_a": "2 matches",
    "toolu_par_b": "notes here",
    "toolu_err_1": "File does not exist: /missing.txt",
    "toolu_den_1": DENIAL_TEXT,
    "toolu_recall_1": "Found: ashiato#40 acceptance tests",
    "u13": "Applying ashiato#40 as precedent.",
    "toolu_pend_1": None,
    "u16": "Still waiting.",
}

#: The full MAIN timeline: (kind, stable id), ordered by (seq, kind, id).
#: 11 rows = 4 text events with text + 7 tool calls; the meta event (u14) and
#: the result-only user lines are not rows.
EXPECTED_ORDER: list[tuple[str, str]] = [
    ("text", "u1"),
    ("text", "u2"),
    ("tool", "toolu_ok_1"),
    ("tool", "toolu_par_a"),
    ("tool", "toolu_par_b"),
    ("tool", "toolu_err_1"),
    ("tool", "toolu_den_1"),
    ("tool", "toolu_recall_1"),
    ("text", "u13"),
    ("tool", "toolu_pend_1"),
    ("text", "u16"),
]

MAIN_COVERAGE: dict[str, object] = {
    "total": 11,
    "returned": 11,
    "truncated": False,
    "has_sessions": True,
    "has_events": True,
    "has_tool_calls": True,
    "has_recall_calls": True,
}


# ---------------------------------------------------------------- fixtures


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def _tool_use(use_id: str, name: str, input_: dict) -> dict:
    return {"type": "tool_use", "id": use_id, "name": name, "input": input_}


def _tool_result(use_id: str, content: str, *, is_error: bool = False) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": use_id,
        "content": content,
        "is_error": is_error,
    }


def _user(uuid: str, parent: str | None, session_id: str, ts: str, content: object) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": session_id,
        "timestamp": ts,
        "message": {"role": "user", "content": content},
    }


def _assistant(
    uuid: str,
    parent: str | None,
    session_id: str,
    ts: str,
    blocks: list[dict],
    *,
    is_meta: bool = False,
) -> dict:
    record: dict = {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": session_id,
        "timestamp": ts,
        "message": {"role": "assistant", "content": blocks},
    }
    if is_meta:
        record["isMeta"] = True
    return record


def write_main_transcript(path: Path) -> None:
    """Session AAAA0001: mixed text/tool order, same-seq calls, every outcome,
    one recall call and one denied call with follow-up, one meta event."""
    s = MAIN
    records = [
        _user("u1", None, s, "2026-08-10T09:00:00Z", "Investigate the trace."),
        _assistant(
            "u2", "u1", s, "2026-08-10T09:00:02Z",
            [_text_block("I'll start."), _tool_use("toolu_ok_1", "Bash", {"command": "ls -1"})],
        ),
        _user("u3", "u2", s, "2026-08-10T09:00:03Z", [_tool_result("toolu_ok_1", "README.md\nsrc")]),
        _assistant(
            "u4", "u3", s, "2026-08-10T09:00:04Z",
            [
                _tool_use("toolu_par_a", "Grep", {"pattern": "TODO"}),
                _tool_use("toolu_par_b", "Read", {"file_path": "/tmp/notes"}),
            ],
        ),
        _user("u5", "u4", s, "2026-08-10T09:00:05Z", [_tool_result("toolu_par_a", "2 matches")]),
        _user("u6", "u5", s, "2026-08-10T09:00:06Z", [_tool_result("toolu_par_b", "notes here")]),
        _assistant(
            "u7", "u6", s, "2026-08-10T09:00:07Z",
            [_tool_use("toolu_err_1", "Read", {"file_path": "/missing.txt"})],
        ),
        _user(
            "u8", "u7", s, "2026-08-10T09:00:08Z",
            [_tool_result("toolu_err_1", "File does not exist: /missing.txt", is_error=True)],
        ),
        _assistant(
            "u9", "u8", s, "2026-08-10T09:00:09Z",
            [_tool_use("toolu_den_1", "Write", {"file_path": "/etc/hosts", "content": "127.0.0.1 nope"})],
        ),
        _user("u10", "u9", s, "2026-08-10T09:00:10Z", [_tool_result("toolu_den_1", DENIAL_TEXT, is_error=True)]),
        _assistant(
            "u11", "u10", s, "2026-08-10T09:00:11Z",
            [_tool_use("toolu_recall_1", "mcp__kaiba__recall", {"query": "session-trace precedent"})],
        ),
        _user(
            "u12", "u11", s, "2026-08-10T09:00:12Z",
            [_tool_result("toolu_recall_1", "Found: ashiato#40 acceptance tests")],
        ),
        _assistant("u13", "u12", s, "2026-08-10T09:00:13Z", [_text_block("Applying ashiato#40 as precedent.")]),
        _assistant("u14", "u13", s, "2026-08-10T09:00:14Z", [_text_block("harness noise")], is_meta=True),
        _assistant(
            "u15", "u14", s, "2026-08-10T09:00:15Z",
            [_tool_use("toolu_pend_1", "Bash", {"command": "sleep 600"})],
        ),
        _assistant("u16", "u15", s, "2026-08-10T09:00:16Z", [_text_block("Still waiting.")]),
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def write_other_transcript(path: Path) -> None:
    """Session AAAA9999: shares the ``AAAA`` prefix with MAIN for the
    ambiguity tests, but not the ``AAAA0`` prefix."""
    s = OTHER
    records = [
        _user("b1", None, s, "2026-08-10T08:00:00Z", "second session"),
        _assistant("b2", "b1", s, "2026-08-10T08:00:01Z", [_tool_use("toolu_b_1", "Bash", {"command": "pwd"})]),
        _user("b3", "b2", s, "2026-08-10T08:00:02Z", [_tool_result("toolu_b_1", "ok")]),
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def write_codex_transcript(path: Path) -> None:
    """Session CODX0001 in the Codex record shape: persisted tool calls only.

    The real build inserts a sessions row and event rows for text chunks too;
    :func:`_build_trace_db` deletes those rows afterwards so the database
    matches the older Codex persisted shape the brief describes (tool_calls
    without sessions/events rows), which the trace must still resolve.
    """
    records = [
        {"timestamp": "2026-09-05T10:00:00Z", "type": "session_meta", "payload": {"id": CODEX}},
        {
            "timestamp": "2026-09-05T10:00:01Z",
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "thread_id": CODEX,
                "item": {
                    "type": "CommandExecution",
                    "id": "ce1",
                    "command": ["bash", "-c", "echo hi"],
                    "stdout": "hi",
                },
            },
        },
        {
            "timestamp": "2026-09-05T10:00:02Z",
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "thread_id": CODEX,
                "item": {
                    "type": "McpToolCall",
                    "id": "ce2",
                    "server": "files",
                    "tool": "list",
                    "arguments": {"path": "/tmp"},
                    "result": "a.txt",
                },
            },
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


@pytest.fixture
def transcripts(tmp_path: Path) -> tuple[Path, Path]:
    claude = tmp_path / "claude"
    codex = tmp_path / "codex"
    claude.mkdir()
    codex.mkdir()
    write_main_transcript(claude / "main.jsonl")
    write_other_transcript(claude / "other.jsonl")
    write_codex_transcript(codex / "codex_trace.jsonl")
    return claude, codex


def _build_trace_db(tmp_path: Path, name: str, claude: Path, codex: Path) -> Path:
    path = tmp_path / f"{name}.duckdb"
    assert (
        main(
            [
                "build",
                "--source",
                str(claude),
                "--codex-source",
                str(codex),
                "--db",
                str(path),
            ]
        )
        == 0
    )
    connection = connect(path)
    try:
        # Reduce CODX0001 to the tool-call-only persisted shape.
        connection.execute("DELETE FROM sessions WHERE session_id = ?", [CODEX])
        connection.execute("DELETE FROM events WHERE session_id = ?", [CODEX])
    finally:
        connection.close()
    return path


@pytest.fixture
def db(tmp_path: Path, transcripts: tuple[Path, Path]) -> Path:
    claude, codex = transcripts
    return _build_trace_db(tmp_path, "trace", claude, codex)


# ---------------------------------------------------------------- resolution


def test_exact_session_id_resolves(db: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["session-trace", MAIN, "--db", str(db), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["session"]["session_id"] == MAIN


def test_unique_prefix_resolves_to_that_session(db: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["session-trace", "AAAA0", "--db", str(db), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["session"]["session_id"] == MAIN


def test_ambiguous_prefix_fails_with_both_candidates(db: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["session-trace", "AAAA", "--db", str(db)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert MAIN in err
    assert OTHER in err


def test_missing_prefix_fails_with_actionable_text(db: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["session-trace", "ZZZZ", "--db", str(db)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "ZZZZ" in err
    assert "no session" in err


# ---------------------------------------------------------------- JSON shape


def test_json_is_an_object_with_session_coverage_and_ordered_timeline(
    db: Path, capsys: pytest.CaptureFixture[str]
):
    assert (
        main(
            [
                "session-trace",
                MAIN,
                "--db",
                str(db),
                "--format",
                "json",
                "--limit",
                "0",
                "--max-excerpt-chars",
                "0",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"session", "coverage", "timeline"}
    assert payload["session"]["session_id"] == MAIN
    assert payload["coverage"] == MAIN_COVERAGE
    assert [(row["kind"], row["id"]) for row in payload["timeline"]] == EXPECTED_ORDER


def test_timeline_interleaves_text_and_tools_in_seq_then_id_order(
    db: Path, capsys: pytest.CaptureFixture[str]
):
    assert (
        main(
            [
                "session-trace",
                MAIN,
                "--db",
                str(db),
                "--format",
                "json",
                "--limit",
                "0",
                "--max-excerpt-chars",
                "0",
            ]
        )
        == 0
    )
    timeline = json.loads(capsys.readouterr().out)["timeline"]
    seqs = [row["seq"] for row in timeline]
    assert seqs == sorted(seqs)
    # Same seq: text rows come before tool rows, then stable id ascending.
    assert [(row["kind"], row["id"]) for row in timeline if row["seq"] == 2] == [
        ("text", "u2"),
        ("tool", "toolu_ok_1"),
    ]


def test_same_seq_tool_calls_are_all_retained_exactly_once(
    db: Path, capsys: pytest.CaptureFixture[str]
):
    assert (
        main(
            [
                "session-trace",
                MAIN,
                "--db",
                str(db),
                "--format",
                "json",
                "--limit",
                "0",
                "--max-excerpt-chars",
                "0",
            ]
        )
        == 0
    )
    timeline = json.loads(capsys.readouterr().out)["timeline"]
    ids = [row["id"] for row in timeline]
    assert ids.count("toolu_par_a") == 1
    assert ids.count("toolu_par_b") == 1
    a = next(row for row in timeline if row["id"] == "toolu_par_a")
    b = next(row for row in timeline if row["id"] == "toolu_par_b")
    assert a["seq"] == b["seq"] == 4
    assert ids.index("toolu_par_a") < ids.index("toolu_par_b")


def test_row_shapes_are_kind_specific(db: Path, capsys: pytest.CaptureFixture[str]):
    assert (
        main(
            [
                "session-trace",
                MAIN,
                "--db",
                str(db),
                "--format",
                "json",
                "--limit",
                "0",
                "--max-excerpt-chars",
                "0",
            ]
        )
        == 0
    )
    rows = {row["id"]: row for row in json.loads(capsys.readouterr().out)["timeline"]}
    assert {
        "kind",
        "id",
        "seq",
        "ts",
        "role",
        "excerpt",
        "truncated",
    } <= set(rows["u1"])
    assert {
        "kind",
        "id",
        "seq",
        "ts",
        "tool_name",
        "input_summary",
        "outcome",
        "excerpt",
        "truncated",
        "is_recall",
    } <= set(rows["toolu_ok_1"])


def test_text_rows_carry_role_and_excerpt(db: Path, capsys: pytest.CaptureFixture[str]):
    assert (
        main(
            [
                "session-trace",
                MAIN,
                "--db",
                str(db),
                "--format",
                "json",
                "--limit",
                "0",
                "--max-excerpt-chars",
                "0",
            ]
        )
        == 0
    )
    rows = {row["id"]: row for row in json.loads(capsys.readouterr().out)["timeline"]}
    first = rows["u1"]
    assert first["kind"] == "text"
    assert first["seq"] == 1
    assert first["ts"] == "2026-08-10 09:00:00"
    assert first["role"] == "user"
    assert first["excerpt"] == "Investigate the trace."
    assert first["truncated"] is False
    assert rows["u13"]["role"] == "assistant"
    assert rows["u13"]["excerpt"] == "Applying ashiato#40 as precedent."


def test_tool_rows_carry_name_summary_outcome_and_result_excerpt(
    db: Path, capsys: pytest.CaptureFixture[str]
):
    assert (
        main(
            [
                "session-trace",
                MAIN,
                "--db",
                str(db),
                "--format",
                "json",
                "--limit",
                "0",
                "--max-excerpt-chars",
                "0",
            ]
        )
        == 0
    )
    rows = {row["id"]: row for row in json.loads(capsys.readouterr().out)["timeline"]}
    ok = rows["toolu_ok_1"]
    assert ok["kind"] == "tool"
    assert ok["seq"] == 2
    assert ok["ts"] == "2026-08-10 09:00:02"
    assert ok["tool_name"] == "Bash"
    assert ok["input_summary"] == "ls -1"
    assert ok["outcome"] == "ok"
    assert ok["excerpt"] == "README.md\nsrc"
    assert ok["truncated"] is False
    assert ok["is_recall"] is False


def test_error_denied_and_pending_stay_distinguishable(
    db: Path, capsys: pytest.CaptureFixture[str]
):
    assert (
        main(
            [
                "session-trace",
                MAIN,
                "--db",
                str(db),
                "--format",
                "json",
                "--limit",
                "0",
                "--max-excerpt-chars",
                "0",
            ]
        )
        == 0
    )
    rows = {row["id"]: row for row in json.loads(capsys.readouterr().out)["timeline"]}
    assert rows["toolu_err_1"]["outcome"] == "error"
    assert rows["toolu_err_1"]["excerpt"] == "File does not exist: /missing.txt"
    assert rows["toolu_den_1"]["outcome"] == "denied"
    assert rows["toolu_den_1"]["excerpt"] == DENIAL_TEXT
    assert rows["toolu_pend_1"]["outcome"] == "pending"
    assert rows["toolu_pend_1"]["excerpt"] is None
    assert rows["toolu_pend_1"]["truncated"] is False


def test_meta_events_are_excluded_by_default(db: Path, capsys: pytest.CaptureFixture[str]):
    assert (
        main(
            [
                "session-trace",
                MAIN,
                "--db",
                str(db),
                "--format",
                "json",
                "--limit",
                "0",
                "--max-excerpt-chars",
                "0",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert "harness noise" not in json.dumps(payload["timeline"])
    assert payload["coverage"]["total"] == len(EXPECTED_ORDER)


# ---------------------------------------------------------------- annotations


def test_recall_call_is_annotated_once_without_a_duplicate_row(
    db: Path, capsys: pytest.CaptureFixture[str]
):
    assert (
        main(
            [
                "session-trace",
                MAIN,
                "--db",
                str(db),
                "--format",
                "json",
                "--limit",
                "0",
                "--max-excerpt-chars",
                "0",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    timeline = payload["timeline"]
    recall_rows = [row for row in timeline if row["id"] == "toolu_recall_1"]
    assert len(recall_rows) == 1
    recall = recall_rows[0]
    assert recall["kind"] == "tool"
    assert recall["is_recall"] is True
    assert recall["tool_name"] == "mcp__kaiba__recall"
    assert recall["outcome"] == "ok"
    assert recall["recall"]["query"] == "session-trace precedent"
    assert recall["recall"]["overlap_count"] == 1
    assert "ashiato#40" in recall["recall"]["followup_text"]
    assert recall["recall"]["followup_truncated"] is False
    # No extra rows were added for the follow-up evidence.
    assert [(row["kind"], row["id"]) for row in timeline] == EXPECTED_ORDER


def test_denied_call_carries_followup_evidence_on_its_own_row(
    db: Path, capsys: pytest.CaptureFixture[str]
):
    assert (
        main(
            [
                "session-trace",
                MAIN,
                "--db",
                str(db),
                "--format",
                "json",
                "--limit",
                "0",
                "--max-excerpt-chars",
                "0",
            ]
        )
        == 0
    )
    timeline = json.loads(capsys.readouterr().out)["timeline"]
    denied = next(row for row in timeline if row["id"] == "toolu_den_1")
    assert denied["outcome"] == "denied"
    assert denied["followup"] == {
        "followup_kind": "other-tool",
        "next_tool_name": "mcp__kaiba__recall",
        "next_input_summary": '{"query":"session-trace precedent"}',
        "next_outcome": "ok",
        "next_ts": "2026-08-10 09:00:11",
        "gap_seconds": 2.0,
    }


def test_a_normal_tool_call_is_not_a_recall(db: Path, capsys: pytest.CaptureFixture[str]):
    assert (
        main(
            [
                "session-trace",
                MAIN,
                "--db",
                str(db),
                "--format",
                "json",
                "--limit",
                "0",
                "--max-excerpt-chars",
                "0",
            ]
        )
        == 0
    )
    timeline = json.loads(capsys.readouterr().out)["timeline"]
    ok = next(row for row in timeline if row["id"] == "toolu_ok_1")
    assert ok["is_recall"] is False


# ---------------------------------------------------------------- limits


def test_limit_zero_means_all_rows(db: Path, capsys: pytest.CaptureFixture[str]):
    assert (
        main(
            [
                "session-trace",
                MAIN,
                "--db",
                str(db),
                "--format",
                "json",
                "--limit",
                "0",
                "--max-excerpt-chars",
                "0",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["coverage"]["total"] == 11
    assert payload["coverage"]["returned"] == 11
    assert payload["coverage"]["truncated"] is False
    assert len(payload["timeline"]) == 11


def test_limit_applies_after_ordering_and_coverage_retains_total(
    db: Path, capsys: pytest.CaptureFixture[str]
):
    assert (
        main(
            [
                "session-trace",
                MAIN,
                "--db",
                str(db),
                "--format",
                "json",
                "--limit",
                "5",
                "--max-excerpt-chars",
                "0",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["coverage"]["total"] == 11
    assert payload["coverage"]["returned"] == 5
    assert payload["coverage"]["truncated"] is True
    assert [(row["kind"], row["id"]) for row in payload["timeline"]] == EXPECTED_ORDER[:5]


def test_excerpts_are_bounded_with_a_truncation_marker(
    db: Path, capsys: pytest.CaptureFixture[str]
):
    limit = 8
    assert (
        main(
            [
                "session-trace",
                MAIN,
                "--db",
                str(db),
                "--format",
                "json",
                "--limit",
                "0",
                "--max-excerpt-chars",
                str(limit),
            ]
        )
        == 0
    )
    timeline = json.loads(capsys.readouterr().out)["timeline"]
    for row in timeline:
        full = FULL_TEXT[row["id"]]
        if full is None:
            assert row["excerpt"] is None
            assert row["truncated"] is False
            continue
        assert row["truncated"] == (len(full) > limit)
        assert len(row["excerpt"]) <= limit + 1
        assert full.startswith(row["excerpt"][:limit])


def test_excerpt_limit_zero_means_uncapped(db: Path, capsys: pytest.CaptureFixture[str]):
    assert (
        main(
            [
                "session-trace",
                MAIN,
                "--db",
                str(db),
                "--format",
                "json",
                "--limit",
                "0",
                "--max-excerpt-chars",
                "0",
            ]
        )
        == 0
    )
    timeline = json.loads(capsys.readouterr().out)["timeline"]
    for row in timeline:
        full = FULL_TEXT[row["id"]]
        if full is None:
            assert row["excerpt"] is None
        else:
            assert row["excerpt"] == full
            assert row["truncated"] is False


# ---------------------------------------------------------------- Codex shape


def test_codex_tool_only_session_resolves_and_traces(
    db: Path, capsys: pytest.CaptureFixture[str]
):
    assert (
        main(
            [
                "session-trace",
                CODEX,
                "--db",
                str(db),
                "--format",
                "json",
                "--limit",
                "0",
                "--max-excerpt-chars",
                "0",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["session"]["session_id"] == CODEX
    assert payload["coverage"] == {
        "total": 2,
        "returned": 2,
        "truncated": False,
        "has_sessions": False,
        "has_events": False,
        "has_tool_calls": True,
        "has_recall_calls": False,
    }
    assert [row["id"] for row in payload["timeline"]] == ["ce1", "ce2"]
    assert payload["timeline"][0]["tool_name"] == "Bash"
    assert payload["timeline"][0]["outcome"] == "ok"
    assert payload["timeline"][0]["ts"] == "2026-09-05 10:00:01"
    assert payload["timeline"][1]["tool_name"] == "mcp__files__list"


def test_codex_session_resolves_by_unique_prefix(db: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["session-trace", "CODX", "--db", str(db), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["session"]["session_id"] == CODEX


# ---------------------------------------------------------------- table & determinism


def test_table_output_carries_the_same_essentials(db: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["session-trace", MAIN, "--db", str(db), "--format", "table"]) == 0


def test_session_trace_does_not_modify_the_duckdb_file(db: Path):
    before = db.stat().st_mtime_ns
    main(["session-trace", MAIN, "--db", str(db)])
    assert db.stat().st_mtime_ns == before


def test_rebuild_output_is_identical(
    tmp_path: Path, transcripts: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
):
    claude, codex = transcripts
    outputs = []
    for name in ("first", "second"):
        path = _build_trace_db(tmp_path, name, claude, codex)
        capsys.readouterr()
        assert (
            main(
                [
                    "session-trace",
                    MAIN,
                    "--db",
                    str(path),
                    "--format",
                    "json",
                    "--limit",
                    "0",
                    "--max-excerpt-chars",
                    "0",
                ]
            )
            == 0
        )
        outputs.append(capsys.readouterr().out)
    assert outputs[0] == outputs[1]
    assert json.loads(outputs[0])["timeline"]  # not vacuously equal