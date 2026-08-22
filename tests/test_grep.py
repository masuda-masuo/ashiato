"""Tests for ashiato.grep: regex search over the transcript corpus.

Fixtures are hand-built JSONL transcripts run through the real build
pipeline (see ``tests/fixtures/`` for the record shape this mirrors) and
written to a private ``tmp_path``, never into ``tests/fixtures/`` itself --
that directory is globbed recursively by other tests' ``--source``, and an
extra file there would silently change their row counts.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from ashiato.build import build, connect
from ashiato.cli import main
from ashiato.grep import InvalidPattern, compile_pattern, search, window

# ---------------------------------------------------------------- fixture transcript


def _tool_use(tool_use_id: str, name: str, input_: dict) -> dict:
    return {"type": "tool_use", "id": tool_use_id, "name": name, "input": input_}


def _tool_result(tool_use_id: str, text: str) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": [{"type": "text", "text": text}],
        "is_error": False,
    }


def _assistant(
    uuid: str, parent: str | None, session_id: str, ts: str, blocks: list, *, is_meta: bool = False
) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": session_id,
        "timestamp": ts,
        "isMeta": is_meta,
        "message": {"role": "assistant", "content": blocks},
    }


def _user(
    uuid: str, parent: str | None, session_id: str, ts: str, blocks: list, *, is_meta: bool = False
) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": session_id,
        "timestamp": ts,
        "isMeta": is_meta,
        "message": {"role": "user", "content": blocks},
    }


def _text(text: str) -> list:
    return [{"type": "text", "text": text}]


@pytest.fixture
def transcript_dir(tmp_path: Path) -> Path:
    """One transcript, one main session covering the search scenarios, plus one repeat-match session."""
    records: list[dict] = []

    # Same needle, one user event and one assistant event, five minutes apart.
    records.append(_user("u1", None, "ses-1", "2026-08-20T10:00:00Z", _text("look for NEEDLE_ABC in the logs")))
    records.append(
        _assistant("a1", "u1", "ses-1", "2026-08-20T10:05:00Z", _text("found the NEEDLE_ABC nearby"))
    )
    # Same needle, differing only in case -- for -i.
    records.append(
        _user("u2", "a1", "ses-1", "2026-08-20T10:10:00Z", _text("case check: NEEDLE_abc lowercase"))
    )
    # A tool call whose input_summary carries a needle no event text has.
    tool_use_id = "toolu_1"
    records.append(
        _assistant(
            "a2",
            "u2",
            "ses-1",
            "2026-08-20T10:15:00Z",
            [_tool_use(tool_use_id, "Bash", {"command": "grep TOOLNEEDLE_XYZ file.py"})],
        )
    )
    records.append(_user("u3", "a2", "ses-1", "2026-08-20T10:15:01Z", [_tool_result(tool_use_id, "done")]))
    # A meta event, excluded by default.
    records.append(
        _assistant("a3", "u3", "ses-1", "2026-08-20T10:20:00Z", _text("METANEEDLE_1 harness noise"), is_meta=True)
    )
    # A different session, one event with the same needle twice -- for --all-matches.
    records.append(_user("u4", None, "ses-2", "2026-08-20T09:00:00Z", _text("REPEATME here and REPEATME again")))

    path = tmp_path / "grep_transcript.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def db(tmp_path: Path, transcript_dir: Path) -> Path:
    path = tmp_path / "grep.duckdb"
    result = build([transcript_dir], path)
    assert result.n_processed == 1
    return path


# ---------------------------------------------------------------- window()


def test_window_never_exceeds_context_on_each_side_plus_the_match():
    text = "x" * 500 + "NEEDLE" + "y" * 500
    start, end = 500, 506
    snippet = window(text, start, end, 50)
    assert len(snippet) <= 100 + (end - start)
    assert "NEEDLE" in snippet


def test_window_replaces_embedded_newlines():
    text = "before\nafter"
    snippet = window(text, 0, 6, 20)
    assert "\n" not in snippet


# ---------------------------------------------------------------- search()


def test_pattern_matches_one_user_and_one_assistant_event(db: Path):
    connection = connect(db, read_only=True)
    try:
        hits = search(connection, "NEEDLE_ABC")
    finally:
        connection.close()
    assert len(hits) == 2
    by_label = {hit.label: hit for hit in hits}
    assert set(by_label) == {"user", "assistant"}
    assert by_label["user"].session_id == "ses-1"
    assert by_label["user"].ts == datetime(2026, 8, 20, 10, 0, 0)
    assert by_label["assistant"].ts == datetime(2026, 8, 20, 10, 5, 0)


def test_hits_are_ordered_newest_first(db: Path):
    connection = connect(db, read_only=True)
    try:
        hits = search(connection, "NEEDLE_ABC")
    finally:
        connection.close()
    assert [hit.label for hit in hits] == ["assistant", "user"]


def test_role_filter_returns_only_the_matching_role(db: Path):
    connection = connect(db, read_only=True)
    try:
        hits = search(connection, "NEEDLE_ABC", role="user")
    finally:
        connection.close()
    assert len(hits) == 1
    assert hits[0].label == "user"


def test_since_excludes_the_earlier_hit(db: Path):
    connection = connect(db, read_only=True)
    try:
        hits = search(connection, "NEEDLE_ABC", since=datetime(2026, 8, 20, 10, 2, 0))
    finally:
        connection.close()
    assert len(hits) == 1
    assert hits[0].label == "assistant"


def test_until_excludes_the_later_hit(db: Path):
    connection = connect(db, read_only=True)
    try:
        hits = search(connection, "NEEDLE_ABC", until=datetime(2026, 8, 20, 10, 2, 0))
    finally:
        connection.close()
    assert len(hits) == 1
    assert hits[0].label == "user"


def test_ignore_case_finds_a_lowercase_variant(db: Path):
    connection = connect(db, read_only=True)
    try:
        case_sensitive = search(connection, "NEEDLE_ABC")
        insensitive = search(connection, "NEEDLE_ABC", ignore_case=True)
    finally:
        connection.close()
    assert len(case_sensitive) == 2
    assert len(insensitive) == 3
    assert "NEEDLE_abc lowercase" in next(h.text for h in insensitive if h.text.startswith("case check"))


def test_invalid_regex_raises_invalid_pattern():
    with pytest.raises(InvalidPattern):
        compile_pattern("(", ignore_case=False)


def test_tool_calls_flag_finds_a_pattern_only_in_input_summary(db: Path):
    connection = connect(db, read_only=True)
    try:
        without_flag = search(connection, "TOOLNEEDLE_XYZ")
        with_flag = search(connection, "TOOLNEEDLE_XYZ", tool_calls=True)
    finally:
        connection.close()
    assert without_flag == []
    assert len(with_flag) == 1
    hit = with_flag[0]
    assert hit.source == "tool_call"
    assert hit.field == "input_summary"
    assert hit.label == "Bash"


def test_meta_events_are_excluded_by_default(db: Path):
    connection = connect(db, read_only=True)
    try:
        default_scope = search(connection, "METANEEDLE_1")
        with_meta = search(connection, "METANEEDLE_1", include_meta=True)
    finally:
        connection.close()
    assert default_scope == []
    assert len(with_meta) == 1


def test_all_matches_returns_every_offset_in_one_hit(db: Path):
    connection = connect(db, read_only=True)
    try:
        first_only = search(connection, "REPEATME")
        all_hits = search(connection, "REPEATME", all_matches=True)
    finally:
        connection.close()
    assert len(first_only) == 1
    assert len(first_only[0].offsets) == 1
    assert len(all_hits) == 1
    assert len(all_hits[0].offsets) == 2


def test_session_prefix_filter(db: Path):
    connection = connect(db, read_only=True)
    try:
        hits = search(connection, "REPEATME", session="ses-2")
        none_hits = search(connection, "REPEATME", session="ses-1")
    finally:
        connection.close()
    assert len(hits) == 1
    assert none_hits == []


def test_limit_zero_means_all(db: Path):
    connection = connect(db, read_only=True)
    try:
        hits = search(connection, "NEEDLE_ABC", ignore_case=True, limit=0)
    finally:
        connection.close()
    assert len(hits) == 3


def test_search_does_not_modify_the_duckdb_file(db: Path):
    before = db.stat().st_mtime_ns
    connection = connect(db, read_only=True)
    try:
        search(connection, "NEEDLE_ABC", tool_calls=True)
    finally:
        connection.close()
    assert db.stat().st_mtime_ns == before


# ---------------------------------------------------------------- CLI wiring


def test_grep_help_exits_zero():
    with pytest.raises(SystemExit) as excinfo:
        main(["grep", "--help"])
    assert excinfo.value.code == 0


def test_cli_prints_two_hits_and_role_filter_narrows_to_one(db: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["grep", "NEEDLE_ABC", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "role=user" in out
    assert "role=assistant" in out
    assert "session=ses-1" in out
    assert "(2 hits)" in out

    assert main(["grep", "NEEDLE_ABC", "--db", str(db), "--role", "user"]) == 0
    out = capsys.readouterr().out
    assert "role=user" in out
    assert "role=assistant" not in out
    assert "(1 hit)" in out


def test_cli_context_bounds_line_length(db: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["grep", "NEEDLE_ABC", "--db", str(db), "--context", "50"]) == 0
    lines = capsys.readouterr().out.splitlines()
    window_lines = [line for line in lines if "NEEDLE_ABC" in line]
    assert window_lines
    for line in window_lines:
        assert len(line) <= 100 + len("NEEDLE_ABC")


def test_cli_limit_one_prints_exactly_one_hit_and_exits_zero(db: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["grep", "NEEDLE_ABC", "--db", str(db), "--limit", "1"]) == 0
    assert "(1 hit)" in capsys.readouterr().out


def test_cli_ignore_case_flag(db: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["grep", "NEEDLE_ABC", "--db", str(db), "-i"]) == 0
    assert "(3 hits)" in capsys.readouterr().out


def test_cli_invalid_regex_exits_two_with_a_message(db: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["grep", "(", "--db", str(db)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")


def test_cli_missing_database_exits_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["grep", "NEEDLE_ABC", "--db", str(tmp_path / "nope.duckdb")]) == 2
    assert "no database at" in capsys.readouterr().err


def test_cli_tool_calls_flag(db: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["grep", "TOOLNEEDLE_XYZ", "--db", str(db), "--tool-calls"]) == 0
    out = capsys.readouterr().out
    assert "tool=Bash" in out


def test_cli_no_hit_exits_one_with_empty_stdout_and_a_stderr_notice(
    db: Path, capsys: pytest.CaptureFixture[str]
):
    assert main(["grep", "NOPE_NOTHING_HERE_AT_ALL", "--db", str(db)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(captured.err.strip().splitlines()) == 1


def test_cli_json_format_carries_structured_fields(db: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["grep", "TOOLNEEDLE_XYZ", "--db", str(db), "--tool-calls", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    row = payload[0]
    assert row["id"] == "toolu_1"
    assert row["source"] == "tool_call"
    assert row["session_id"] == "ses-1"
    assert row["label"] == "Bash"
    assert row["field"] == "input_summary"
    assert "offsets" in row and "text" in row


def test_cli_does_not_modify_the_duckdb_file(db: Path):
    before = db.stat().st_mtime_ns
    main(["grep", "NEEDLE_ABC", "--db", str(db), "--tool-calls", "--format", "csv"])
    assert db.stat().st_mtime_ns == before


def test_cli_whole_prints_the_full_row_text_instead_of_a_window(
    db: Path, capsys: pytest.CaptureFixture[str]
):
    # With a tiny --context the window would cut the sentence; --whole must
    # print the entire matched text regardless of --context.
    assert main(["grep", "NEEDLE_ABC", "--db", str(db), "--role", "user", "--context", "1", "--whole"]) == 0
    out = capsys.readouterr().out
    assert "look for NEEDLE_ABC in the logs" in out
    assert "(1 hit)" in out
