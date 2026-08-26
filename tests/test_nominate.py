"""Tests for ashiato.nominate: mining re-derived facts as kaiba nomination candidates.

Fixtures are hand-built JSONL transcripts run through the real build
pipeline (see ``tests/fixtures/`` for the record shape this mirrors) and
written to a private ``tmp_path``, never into ``tests/fixtures/`` itself ----
that directory is globbed recursively by other tests' ``--source``, and an
extra file there would silently change their row counts.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

from ashiato.build import build, connect
from ashiato.cli import main
from ashiato.nominate import (
    _load_exclude_file,
    mine_negative_facts,
    mine_stable_outputs,
    normalize_command,
    normalize_result_text,
    run,
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
    *,
    is_error: bool = False,
) -> str:
    """Append one tool_use/tool_result pair; return the tool_use_id."""
    tool_use_id = f"{prefix}_use"
    parent = records[-1]["uuid"] if records else None
    call_uuid = f"{prefix}_call"
    records.append(
        _assistant(call_uuid, parent, session_id, ts_call, [_tool_use(tool_use_id, tool_name, tool_input)])
    )
    records.append(
        _user(
            f"{prefix}_result",
            call_uuid,
            session_id,
            ts_result,
            [_tool_result(tool_use_id, result_text, is_error=is_error)],
        )
    )
    return tool_use_id


@pytest.fixture
def transcript_dir(tmp_path: Path) -> Path:
    """One transcript file covering all test scenarios."""
    records: list[dict] = []

    # --- negative-fact: same error text in 3 sessions (ses-neg-1, ses-neg-2, ses-neg-3) ---
    for i, (sid, ts) in enumerate(
        [
            ("ses-neg-1", "2026-08-20T10:00:00Z"),
            ("ses-neg-2", "2026-08-21T11:00:00Z"),
            ("ses-neg-3", "2026-08-22T12:00:00Z"),
        ]
    ):
        _call(
            records,
            f"neg{i}",
            sid,
            ts,
            ts,
            "Bash",
            {"command": "which sqlite3"},
            "sqlite3: command not found",
            is_error=True,
        )

    # --- negative-fact threshold: same error in only 2 sessions (not enough) ---
    for i, (sid, ts) in enumerate(
        [
            ("ses-thresh-1", "2026-08-20T13:00:00Z"),
            ("ses-thresh-2", "2026-08-21T14:00:00Z"),
        ]
    ):
        _call(
            records,
            f"thresh{i}",
            sid,
            ts,
            ts,
            "Bash",
            {"command": "which nonexistent"},
            "nonexistent: command not found",
            is_error=True,
        )

    # --- ritual: --help in many sessions (should be excluded) ---
    for i, (sid, ts) in enumerate(
        [
            ("ses-rit-1", "2026-08-20T15:00:00Z"),
            ("ses-rit-2", "2026-08-21T16:00:00Z"),
            ("ses-rit-3", "2026-08-22T17:00:00Z"),
        ]
    ):
        _call(
            records,
            f"rit{i}",
            sid,
            ts,
            ts,
            "Bash",
            {"command": "ashiato --help"},
            "usage: ashiato ...",
        )

    # --- stable-output: same informative output in 3 sessions ---
    for i, (sid, ts) in enumerate(
        [
            ("ses-stable-1", "2026-08-20T18:00:00Z"),
            ("ses-stable-2", "2026-08-21T19:00:00Z"),
            ("ses-stable-3", "2026-08-22T20:00:00Z"),
        ]
    ):
        _call(
            records,
            f"stable{i}",
            sid,
            ts,
            ts,
            "Bash",
            {"command": "cat /etc/hostname"},
            "myserver",
        )

    # --- stable-output varying: same command, different outputs (not stable) ---
    for i, (sid, ts, output) in enumerate(
        [
            ("ses-var-1", "2026-08-20T21:00:00Z", "result_A"),
            ("ses-var-2", "2026-08-21T22:00:00Z", "result_B"),
            ("ses-var-3", "2026-08-22T23:00:00Z", "result_C"),
        ]
    ):
        _call(
            records,
            f"var{i}",
            sid,
            ts,
            ts,
            "Bash",
            {"command": "date +%s"},
            output,
        )

    # --- empty/uninformative modal output: should not be nominated ---
    for i, (sid, ts) in enumerate(
        [
            ("ses-empty-1", "2026-08-20T08:00:00Z"),
            ("ses-empty-2", "2026-08-21T08:00:00Z"),
            ("ses-empty-3", "2026-08-22T08:00:00Z"),
        ]
    ):
        _call(
            records,
            f"empty{i}",
            sid,
            ts,
            ts,
            "Bash",
            {"command": "touch /tmp/foo"},
            "",
        )

    # --- harness suffix: "Shell cwd was reset to ..." should not break grouping ---
    for i, (sid, ts) in enumerate(
        [
            ("ses-harness-1", "2026-08-20T09:00:00Z"),
            ("ses-harness-2", "2026-08-21T09:00:00Z"),
            ("ses-harness-3", "2026-08-22T09:00:00Z"),
        ]
    ):
        suffix = "\nShell cwd was reset to /home/user/project"
        _call(
            records,
            f"harness{i}",
            sid,
            ts,
            ts,
            "Bash",
            {"command": "myhost"},
            f"host{i}{suffix}",
        )

    # --- within-session repeats: should not inflate session count ---
    # Two calls in the same session with same error
    _call(
        records,
        "repeat-a",
        "ses-repeat-1",
        "2026-08-20T07:00:00Z",
        "2026-08-20T07:00:01Z",
        "Bash",
        {"command": "which missing-tool"},
        "missing-tool: command not found",
        is_error=True,
    )
    _call(
        records,
        "repeat-b",
        "ses-repeat-1",
        "2026-08-20T07:00:02Z",
        "2026-08-20T07:00:03Z",
        "Bash",
        {"command": "which missing-tool"},
        "missing-tool: command not found",
        is_error=True,
    )
    # One more session with the same error to reach threshold
    _call(
        records,
        "repeat-c",
        "ses-repeat-2",
        "2026-08-21T07:00:00Z",
        "2026-08-21T07:00:01Z",
        "Bash",
        {"command": "which missing-tool"},
        "missing-tool: command not found",
        is_error=True,
    )
    _call(
        records,
        "repeat-d",
        "ses-repeat-3",
        "2026-08-22T07:00:00Z",
        "2026-08-22T07:00:01Z",
        "Bash",
        {"command": "which missing-tool"},
        "missing-tool: command not found",
        is_error=True,
    )

    # --- harness suffix on stable-output: grouping must still work ---
    for i, (sid, ts) in enumerate(
        [
            ("ses-hs-1", "2026-08-20T06:00:00Z"),
            ("ses-hs-2", "2026-08-21T06:00:00Z"),
            ("ses-hs-3", "2026-08-22T06:00:00Z"),
        ]
    ):
        _call(
            records,
            f"hs{i}",
            sid,
            ts,
            ts,
            "Bash",
            {"command": "getent passwd"},
            "stablehost\nShell cwd was reset to /home/user",
        )

    path = tmp_path / "nominate_transcript.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def db(tmp_path: Path, transcript_dir: Path) -> Path:
    path = tmp_path / "nominate.duckdb"
    result = build([transcript_dir], path)
    assert result.n_processed == 1
    return path


def _build_bash_db(
    tmp_path: Path,
    *,
    command: str,
    result_text: str,
    is_error: bool = True,
    n_sessions: int = 3,
    prefix: str = "case",
) -> Path:
    """Private DuckDB with the same Bash error in n_sessions sessions."""
    records: list[dict] = []
    for i in range(n_sessions):
        sid = f"ses-{prefix}-{i + 1}"
        ts = f"2026-08-{20 + i}T10:00:00Z"
        _call(
            records,
            f"{prefix}{i}",
            sid,
            ts,
            ts,
            "Bash",
            {"command": command},
            result_text,
            is_error=is_error,
        )
    transcript = tmp_path / f"{prefix}_transcript.jsonl"
    transcript.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    db_path = tmp_path / f"{prefix}.duckdb"
    build([transcript], db_path)
    return db_path


# ---------------------------------------------------------------- normalization unit tests


class TestNormalizeResultText:
    def test_strips_harness_suffix(self) -> None:
        text = "output line\nShell cwd was reset to /home/user"
        assert normalize_result_text(text) == "output line"

    def test_strips_trailing_whitespace(self) -> None:
        assert normalize_result_text("output  \n  ") == "output"

    def test_truncates_to_max_chars(self) -> None:
        text = "x" * 3000
        result = normalize_result_text(text, max_chars=100)
        assert len(result) == 100

    def test_none_returns_empty(self) -> None:
        assert normalize_result_text(None) == ""

    def test_no_suffix_unchanged(self) -> None:
        assert normalize_result_text("hello") == "hello"


class TestNormalizeCommand:
    def test_collapses_hex(self) -> None:
        assert "<HEX>" in normalize_command("git checkout a1b2c3d4e5f6")

    def test_collapses_uuid(self) -> None:
        cmd = "echo 550e8400-e29b-41d4-a716-446655440000"
        assert "<UUID>" in normalize_command(cmd)

    def test_collapses_large_integers(self) -> None:
        assert "<N>" in normalize_command("echo 12345678")

    def test_collapses_tmp_paths(self) -> None:
        assert "/tmp/<...>" in normalize_command("cat /tmp/abc123/file.txt")

    def test_squashes_whitespace(self) -> None:
        assert normalize_command("  echo   hello   ") == "echo hello"


class TestFormatVersion:
    def test_format_version_is_unchanged(self) -> None:
        assert FORMAT_VERSION == 5


# ---------------------------------------------------------------- negative-fact miner tests


class TestMineNegativeFacts:
    def test_same_error_in_3_sessions_is_nominated(self, db: Path) -> None:
        connection = connect(db, read_only=True)
        try:
            candidates = mine_negative_facts(connection, min_sessions=3)
        finally:
            connection.close()
        signals = [c.signal for c in candidates]
        assert "negative-fact" in signals
        neg = next(c for c in candidates if c.signal == "negative-fact")
        assert neg.sessions >= 3
        assert "sqlite3: command not found" in neg.sample_output

    def test_same_error_in_2_sessions_is_not_nominated(self, db: Path) -> None:
        connection = connect(db, read_only=True)
        try:
            candidates = mine_negative_facts(connection, min_sessions=3)
        finally:
            connection.close()
        # The threshold group (2 sessions) should not appear
        for c in candidates:
            assert "nonexistent: command not found" not in c.sample_output

    def test_ritual_not_in_negative_facts(self, db: Path) -> None:
        # Rituals are not errors, so they should not appear in negative-fact results
        connection = connect(db, read_only=True)
        try:
            candidates = mine_negative_facts(connection, min_sessions=3)
        finally:
            connection.close()
        for c in candidates:
            assert "ashiato --help" not in c.command

    def test_within_session_repeats_do_not_inflate_count(self, db: Path) -> None:
        connection = connect(db, read_only=True)
        try:
            candidates = mine_negative_facts(connection, min_sessions=3)
        finally:
            connection.close()
        # The missing-tool error appears in ses-repeat-1 (2 calls), ses-repeat-2, ses-repeat-3
        # That's 3 distinct sessions
        neg = next(
            (c for c in candidates if "missing-tool" in c.sample_output), None
        )
        if neg is not None:
            # Should have 3 distinct sessions, not 4
            assert len(neg.session_ids) == 3

    @pytest.mark.parametrize(
        "result_text",
        [
            "Exit code 143",
            "Command timed out",
            "Exit code 124",
            "Exit code 137",
            "Exit code 144",
            (
                "This operation is temporarily unavailable, so auto mode "
                "cannot determine a permission"
            ),
        ],
    )
    def test_operational_deaths_are_not_nominated(
        self, tmp_path: Path, result_text: str
    ) -> None:
        db_path = _build_bash_db(
            tmp_path,
            command="sleep 30",
            result_text=result_text,
            is_error=True,
            prefix="death",
        )
        connection = connect(db_path, read_only=True)
        try:
            candidates = mine_negative_facts(connection, min_sessions=3)
        finally:
            connection.close()
        assert candidates == []

    def test_jq_command_not_found_is_nominated(self, tmp_path: Path) -> None:
        db_path = _build_bash_db(
            tmp_path,
            command="jq .",
            result_text="jq: command not found",
            is_error=True,
            prefix="jq",
        )
        connection = connect(db_path, read_only=True)
        try:
            candidates = mine_negative_facts(connection, min_sessions=3)
        finally:
            connection.close()
        assert any("jq: command not found" in c.sample_output for c in candidates)


# ---------------------------------------------------------------- stable-output miner tests


class TestMineStableOutputs:
    def test_identical_output_in_3_sessions_is_nominated(self, db: Path) -> None:
        connection = connect(db, read_only=True)
        try:
            candidates = mine_stable_outputs(connection, min_sessions=3)
        finally:
            connection.close()
        stable = [c for c in candidates if c.signal == "stable-output"]
        assert len(stable) >= 1
        # The /etc/hostname group should be in there
        host_candidate = next(
            (c for c in stable if "cat /etc/hostname" in c.command), None
        )
        assert host_candidate is not None
        assert host_candidate.stability == 1.0

    def test_ritual_excluded_from_stable_output_despite_recurrence(self, db: Path) -> None:
        # The fixture has `ashiato --help` in 3 sessions (ses-rit-1/2/3) with the
        # identical informative output "usage: ashiato ...", which would otherwise
        # be a perfect stable-output candidate (3 sessions, stability 1.0). The
        # stable-output miner must drop it via the builtin ritual filter.
        connection = connect(db, read_only=True)
        try:
            candidates = mine_stable_outputs(connection, min_sessions=3)
        finally:
            connection.close()
        stable = [c for c in candidates if c.signal == "stable-output"]
        # No stable-output candidate should carry the ritual command.
        for c in stable:
            assert "--help" not in c.command
        # Explicitly: the help command group is absent.
        assert all("ashiato --help" not in c.command for c in stable)

    def test_varying_outputs_are_not_nominated(self, db: Path) -> None:
        connection = connect(db, read_only=True)
        try:
            candidates = mine_stable_outputs(connection, min_sessions=3)
        finally:
            connection.close()
        # date +%s has varying outputs, should not be nominated
        for c in candidates:
            assert "date +%s" not in c.command

    def test_empty_uninformative_modal_output_is_not_nominated(self, db: Path) -> None:
        connection = connect(db, read_only=True)
        try:
            candidates = mine_stable_outputs(connection, min_sessions=3)
        finally:
            connection.close()
        # touch /tmp/foo has empty output, should not be nominated
        for c in candidates:
            assert "touch /tmp/foo" not in c.command

    def test_harness_suffix_does_not_break_grouping(self, db: Path) -> None:
        connection = connect(db, read_only=True)
        try:
            candidates = mine_stable_outputs(connection, min_sessions=3)
        finally:
            connection.close()
        # The getent passwd group with harness suffix should be grouped together
        hs = next(
            (c for c in candidates if c.signal == "stable-output" and "getent" in c.command),
            None,
        )
        assert hs is not None
        assert hs.stability == 1.0


# ---------------------------------------------------------------- integration / run() tests


class TestRunIntegration:
    def test_exit_code_1_when_nothing_qualifies(self, tmp_path: Path) -> None:
        """A nearly empty DB should yield exit code 1."""
        # Build a minimal DB with one non-error non-matching call
        records: list[dict] = []
        _call(
            records,
            "solo",
            "ses-solo",
            "2026-08-20T10:00:00Z",
            "2026-08-20T10:00:01Z",
            "Bash",
            {"command": "echo hello"},
            "hello",
        )
        transcript = tmp_path / "empty_transcript.jsonl"
        transcript.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
        )
        db_path = tmp_path / "empty.duckdb"
        build([transcript], db_path)

        # Use a real StringIO to capture
        out_sio = io.StringIO()
        err_sio = io.StringIO()
        rc = run(db_path, min_sessions=3, out=out_sio, err=err_sio)
        assert rc == 1

    def test_json_output_parses_and_carries_keys(self, db: Path) -> None:
        out_sio = io.StringIO()
        err_sio = io.StringIO()
        rc = run(db, min_sessions=3, json_output=True, out=out_sio, err=err_sio)
        assert rc == 0
        payload = json.loads(out_sio.getvalue())
        assert isinstance(payload, list)
        assert len(payload) >= 1
        for item in payload:
            assert "signal" in item
            assert "sessions" in item
            assert "session_ids" in item
            assert "command" in item
            assert "sample_output" in item
            assert "draft" in item
            assert "first_seen" in item
            assert "last_seen" in item
            assert item["first_seen"] is not None
            assert item["last_seen"] is not None

    def test_exit_code_0_when_candidates_exist(self, db: Path) -> None:
        out_sio = io.StringIO()
        err_sio = io.StringIO()
        rc = run(db, min_sessions=3, out=out_sio, err=err_sio)
        assert rc == 0

    def test_db_not_opened_for_write(self, db: Path) -> None:
        """Opening the DB should not modify it."""
        before = db.stat().st_mtime_ns
        out_sio = io.StringIO()
        err_sio = io.StringIO()
        run(db, min_sessions=3, out=out_sio, err=err_sio)
        assert db.stat().st_mtime_ns == before


# ---------------------------------------------------------------- CLI wiring tests


def test_nominate_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["nominate", "--help"])
    assert excinfo.value.code == 0


def test_nominate_on_missing_database_exits_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["nominate", "--db", str(tmp_path / "nope.duckdb")]) == 1
    assert "no database at" in capsys.readouterr().err


def test_nominate_default_table_output(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["nominate", "--db", str(db), "--min-sessions", "3"]) == 0
    out = capsys.readouterr().out
    assert "candidate" in out


def test_nominate_json_output(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["nominate", "--db", str(db), "--min-sessions", "3", "--json"]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert isinstance(payload, list)
    assert len(payload) >= 1


# ---------------------------------------------------------------- exclude-file tests


class TestExcludeFile:
    def test_regex_suppresses_otherwise_stable_output(
        self, db: Path, tmp_path: Path
    ) -> None:
        out_before = io.StringIO()
        rc_before = run(
            db, min_sessions=3, json_output=True, out=out_before, err=io.StringIO()
        )
        assert rc_before == 0
        commands_before = [item["command"] for item in json.loads(out_before.getvalue())]
        assert any("hostname" in cmd for cmd in commands_before)

        exclude = tmp_path / "exclude.txt"
        exclude.write_text(
            "# (unclosed if this comment were compiled\n\nhostname\n",
            encoding="utf-8",
        )
        out_after = io.StringIO()
        rc_after = run(
            db,
            min_sessions=3,
            json_output=True,
            exclude_file=exclude,
            out=out_after,
            err=io.StringIO(),
        )
        assert rc_after == 0
        commands_after = [item["command"] for item in json.loads(out_after.getvalue())]
        assert not any("hostname" in cmd for cmd in commands_after)

    def test_blank_lines_and_comments_are_ignored(
        self, db: Path, tmp_path: Path
    ) -> None:
        comments_only = tmp_path / "comments.txt"
        comments_only.write_text(
            "# (unclosed if compiled\n\n\n# another comment\n",
            encoding="utf-8",
        )
        out_sio = io.StringIO()
        err_sio = io.StringIO()
        rc = run(
            db,
            min_sessions=3,
            json_output=True,
            exclude_file=comments_only,
            out=out_sio,
            err=err_sio,
        )
        assert rc == 0
        commands = [item["command"] for item in json.loads(out_sio.getvalue())]
        assert any("hostname" in cmd for cmd in commands)

    def test_missing_file_warns_and_still_runs(self, db: Path, tmp_path: Path) -> None:
        missing = tmp_path / "no-such-exclude.txt"
        out_sio = io.StringIO()
        err_sio = io.StringIO()
        rc = run(
            db, min_sessions=3, exclude_file=missing, out=out_sio, err=err_sio
        )
        assert rc == 0
        err = err_sio.getvalue()
        assert "warning: exclude file not found:" in err
        assert str(missing) in err

    def test_invalid_regex_names_line_and_text(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.txt"
        path.write_text("ok-pattern\n(unclosed\n", encoding="utf-8")
        with pytest.raises(re.error) as excinfo:
            _load_exclude_file(path)
        message = str(excinfo.value)
        assert "line 2" in message
        assert "(unclosed" in message
        assert excinfo.value.__cause__ is not None


def test_unused_row_helpers_are_removed() -> None:
    import ashiato.nominate as nominate_mod

    assert not hasattr(nominate_mod, "_COLUMNS")
    assert not hasattr(nominate_mod, "_candidate_to_row")
