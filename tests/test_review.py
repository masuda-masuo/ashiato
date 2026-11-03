"""Synthetic tests for the review report; no real corpus values are used."""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import duckdb
import pytest

from ashiato.build import build
from ashiato.cli import main
from ashiato.review import build_report, human_text_rows, strip_pasted


def db_with_data(sessions=(), events=(), calls=()):
    connection = duckdb.connect(":memory:")
    connection.execute("""
        CREATE TABLE sessions (
            file_path VARCHAR, session_id VARCHAR, source VARCHAR,
            started_at TIMESTAMP, entrypoint VARCHAR,
            project_dir VARCHAR, n_tool_calls INTEGER
        )
    """)
    connection.execute("""
        CREATE TABLE events (
            file_path VARCHAR, session_id VARCHAR, seq INTEGER, ts TIMESTAMP,
            role VARCHAR, text VARCHAR, raw VARCHAR, is_meta BOOLEAN,
            is_sidechain BOOLEAN
        )
    """)
    connection.execute("""
        CREATE TABLE tool_calls (
            tool_use_id VARCHAR, session_id VARCHAR, file_path VARCHAR,
            source VARCHAR, ts TIMESTAMP, result_text VARCHAR
        )
    """)
    if sessions:
        connection.executemany(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(*row, "/synthetic", 0) for row in sessions],
        )
    if events:
        connection.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", events)
    if calls:
        connection.executemany("INSERT INTO tool_calls VALUES (?, ?, ?, ?, ?, ?)", calls)
    return connection


def dt(value):
    return datetime.fromisoformat(value)


def test_first_appearance_and_same_timestamp_tool_id_tie_break():
    connection = db_with_data(
        sessions=[
            ("a.jsonl", "session-a", "claude_code", dt("2026-03-01T00:00:00"), "cli"),
            ("b.jsonl", "session-b", "codex", dt("2026-03-02T00:00:00"), "cli"),
        ],
        calls=[
            (
                "tool-b",
                "session-b",
                "b.jsonl",
                "codex",
                dt("2026-03-01T10:00:00"),
                "https://github.com/ACME/Widgets/issues/7",
            ),
            (
                "tool-a",
                "session-a",
                "a.jsonl",
                "claude_code",
                dt("2026-03-01T10:00:00"),
                "https://github.com/acme/widgets/pull/7#issuecomment-1",
            ),
            (
                "later",
                "session-b",
                "b.jsonl",
                "codex",
                dt("2026-03-02T10:00:00"),
                "https://github.com/acme/widgets/pull/7/files?x=1",
            ),
        ],
    )
    report = build_report(connection, tz=ZoneInfo("UTC"))
    outputs = [o for b in report["buckets"] for o in b["outputs"]]
    assert len(outputs) == 1
    assert outputs[0]["session_id"] == "session-a"
    assert outputs[0]["file_path"] == "a.jsonl"
    assert outputs[0]["kind"] == "pr"
    assert outputs[0]["url"] == "https://github.com/acme/widgets/pull/7"
    connection.close()


def test_undated_only_counted_by_source_never_sent_to_gh():
    connection = db_with_data(
        calls=[
            ("z", "sid-z", "z.jsonl", "cursor", None, "https://github.com/acme/widgets/issues/9"),
            (
                "a",
                "sid-a",
                "a.jsonl",
                "cursor",
                None,
                "https://docs.google.com/document/d/abc_123/edit",
            ),
        ]
    )
    seen = []
    report = build_report(
        connection, tz=ZoneInfo("UTC"), gh=True, gh_call=lambda argv: seen.append(argv)
    )
    assert report["undated"] == {"count": 2, "by_source": {"cursor": 2}}
    assert report["buckets"] == []
    assert seen == []
    connection.close()


def test_gh_statuses_kind_window_and_drive_unchecked():
    connection = db_with_data(
        calls=[
            (
                "one",
                "s1",
                "one.jsonl",
                "claude_code",
                dt("2026-03-01T12:01:00"),
                "https://github.com/acme/one/issues/1",
            ),
            (
                "two",
                "s2",
                "two.jsonl",
                "claude_code",
                dt("2026-03-01T12:02:00"),
                "https://github.com/acme/two/issues/2",
            ),
            (
                "three",
                "s3",
                "three.jsonl",
                "claude_code",
                dt("2026-03-01T12:03:00"),
                "https://github.com/acme/three/issues/3",
            ),
            (
                "four",
                "s4",
                "four.jsonl",
                "claude_code",
                dt("2026-02-01T00:00:00"),
                "https://github.com/acme/outside/issues/4",
            ),
            (
                "drive",
                "sd",
                "drive.jsonl",
                "claude_code",
                dt("2026-03-01T12:00:00"),
                "https://docs.google.com/spreadsheets/d/sheet-id/edit",
            ),
        ]
    )
    called = []

    def fake(argv):
        called.append(argv)
        if "one/" in argv[-1]:
            return True, {"created_at": "2026-03-01T12:00:00Z", "user": {"login": "bot"}}, ""
        if "two/" in argv[-1]:
            return (
                True,
                {
                    "created_at": "2026-03-01T11:00:00Z",
                    "user": {"login": "owner"},
                    "pull_request": {},
                },
                "",
            )
        return False, None, "HTTP 500 unavailable"

    report = build_report(
        connection,
        tz=ZoneInfo("UTC"),
        gh=True,
        since=dt("2026-03-01T00:00:00"),
        until=dt("2026-03-31T23:59:59"),
        gh_call=fake,
    )
    outputs = {o["repo"]: o for b in report["buckets"] for o in b["outputs"]}
    assert outputs["acme/one"]["status"] == "confirmed"
    assert outputs["acme/two"]["status"] == "outside"
    assert outputs["acme/two"]["kind"] == "pr"
    assert outputs["acme/two"]["url"] == "https://github.com/acme/two/pull/2"
    assert outputs["acme/three"]["status"] == "unknown"
    assert outputs["acme/three"]["error"] == "HTTP 500 unavailable"
    assert outputs[None]["status"] == "unchecked"
    assert len(called) == 3
    assert all("outside/" not in argv[-1] for argv in called)
    connection.close()


def test_timezone_and_monday_week_buckets():
    connection = db_with_data(
        sessions=[("s.jsonl", "sid", "codex", dt("2026-03-01T16:00:00"), "cli")],
        calls=[
            (
                "u",
                "sid",
                "s.jsonl",
                "codex",
                dt("2026-03-01T16:00:00"),
                "https://github.com/acme/widgets/pull/1",
            )
        ],
    )
    tokyo = build_report(connection, bucket="week", tz=ZoneInfo("Asia/Tokyo"))
    assert tokyo["buckets"][0]["label"] == "2026-03-02"
    assert tokyo["buckets"][0]["outputs"][0]["first_seen"] == "2026-03-01T16:00:00"
    utc = build_report(connection, bucket="month", tz=ZoneInfo("UTC"))
    assert utc["buckets"][0]["label"] == "2026-03"
    assert utc["buckets"][0]["start"] == "2026-03-01T00:00:00+00:00"
    connection.close()


def test_human_text_selection_and_harness_stripping():
    connection = db_with_data(
        sessions=[
            ("normal", "normal", "claude_code", dt("2026-03-01"), "cli"),
            ("headless", "headless", "claude_code", dt("2026-03-01"), "sdk-worker"),
        ],
        events=[
            ("normal", "normal", 0, dt("2026-03-01"), "assistant", "assistant", "{}", False, False),
            (
                "normal",
                "normal",
                1,
                dt("2026-03-01"),
                "user",
                "human <system-reminder>skip</system-reminder> words",
                "{}",
                False,
                False,
            ),
            ("normal", "normal", 2, dt("2026-03-01"), "user", "meta", "{}", True, False),
            (
                "normal",
                "normal",
                3,
                dt("2026-03-01"),
                "user",
                "tool result",
                '{"tool_result": true}',
                False,
                False,
            ),
            (
                "normal",
                "normal",
                4,
                dt("2026-03-01"),
                "user",
                "summary",
                '{"isCompactSummary": true}',
                False,
                False,
            ),
            ("headless", "headless", 0, dt("2026-03-01"), "user", "brief", "{}", False, False),
        ],
    )
    selected = list(human_text_rows(connection))
    assert [row["text"] for row in selected] == ["human  words"]
    report = build_report(connection, tz=ZoneInfo("UTC"))
    assert report["buckets"][0]["human_chars"] == len("human  words")
    connection.close()


def test_null_call_timestamp_falls_back_for_human_text_and_json_shape():
    connection = db_with_data(
        sessions=[("x", "sid", "claude_ai", dt("2026-03-01T00:00:00"), "cli")],
        events=[("x", "sid", 0, None, "user", "hello", "{}", False, False)],
    )
    report = build_report(connection, tz=ZoneInfo("UTC"))
    assert report["buckets"][0]["human_chars"] == 5
    document = json.loads(json.dumps(report, default=str))
    assert set(document) == {"tz", "bucket", "since", "until", "gh", "owners", "owners_inferred", "references", "undated", "buckets"}
    connection.close()


def test_undated_pull_appearance_sets_kind_without_gh():
    connection = db_with_data(
        calls=[
            (
                "dated",
                "sid-dated",
                "dated.jsonl",
                "claude_code",
                dt("2026-03-01T12:00:00"),
                "https://github.com/acme/widgets/issues/5",
            ),
            (
                "undated",
                "sid-undated",
                "undated.jsonl",
                "cursor",
                None,
                "https://github.com/acme/widgets/pull/5",
            ),
        ]
    )
    report = build_report(connection, tz=ZoneInfo("UTC"))
    outputs = [item for bucket in report["buckets"] for item in bucket["outputs"]]
    assert len(outputs) == 1
    assert outputs[0]["kind"] == "pr"
    assert outputs[0]["status"] == "unchecked"
    connection.close()


def test_api_github_url_is_not_an_output():
    connection = db_with_data(
        calls=[
            (
                "api",
                "sid",
                "api.jsonl",
                "claude_code",
                dt("2026-03-01T12:00:00"),
                "https://api.github.com/repos/acme/widgets/issues/5",
            )
        ]
    )
    report = build_report(connection, tz=ZoneInfo("UTC"))
    assert report["buckets"] == []
    connection.close()


def test_output_timezone_moves_across_month_boundary():
    connection = db_with_data(
        calls=[
            (
                "month-boundary",
                "sid",
                "session.jsonl",
                "claude_code",
                dt("2026-03-31T20:00:00"),
                "https://github.com/acme/widgets/issues/5",
            )
        ]
    )
    utc = build_report(connection, tz=ZoneInfo("UTC"))
    tokyo = build_report(connection, tz=ZoneInfo("Asia/Tokyo"))
    assert utc["buckets"][0]["label"] == "2026-03"
    assert tokyo["buckets"][0]["label"] == "2026-04"
    connection.close()


def test_themes_only_include_selected_human_rows():
    sessions = [
        (f"human-{index}.jsonl", f"human-{index}", "codex", dt("2026-03-01"), "cli")
        for index in range(4)
    ]
    sessions.append(("headless.jsonl", "headless", "codex", dt("2026-03-01"), "sdk-worker"))
    events = [
        ("human-0.jsonl", "human-0", 0, dt("2026-03-01"), "user", "quartzword", "{}", False, False),
        (
            "human-0.jsonl",
            "human-0",
            1,
            dt("2026-03-01"),
            "assistant",
            "assistantword",
            "{}",
            False,
            False,
        ),
        ("human-1.jsonl", "human-1", 0, dt("2026-03-01"), "user", "metaword", "{}", True, False),
        (
            "headless.jsonl",
            "headless",
            0,
            dt("2026-03-01"),
            "user",
            "workerword",
            "{}",
            False,
            False,
        ),
        *[
            (
                f"human-{index}.jsonl",
                f"human-{index}",
                2,
                dt("2026-03-01"),
                "user",
                "commonword commonword",
                "{}",
                False,
                False,
            )
            for index in range(1, 4)
        ],
    ]
    connection = db_with_data(sessions=sessions, events=events)
    report = build_report(connection, tz=ZoneInfo("UTC"))
    terms = {item["term"] for item in report["buckets"][0]["themes"]}
    assert "quartzword" in terms
    assert "assistantword" not in terms
    assert "metaword" not in terms
    assert "workerword" not in terms
    connection.close()


def test_themes_strip_fenced_blocks_and_keep_following_text():
    sessions = [
        (f"fence-{index}.jsonl", f"fence-{index}", "codex", dt("2026-03-01"), "cli")
        for index in range(4)
    ]
    events = [
        (
            "fence-0.jsonl",
            "fence-0",
            0,
            dt("2026-03-01"),
            "user",
            "beforeword\n\x60\x60\x60python\nbacktickword\n\x60\x60\x60\n~~~text\ntildeword\n~~~\nafterword",
            "{}",
            False,
            False,
        ),
        *[
            (
                f"fence-{index}.jsonl",
                f"fence-{index}",
                0,
                dt("2026-03-01"),
                "user",
                "commonword commonword",
                "{}",
                False,
                False,
            )
            for index in range(1, 4)
        ],
    ]
    connection = db_with_data(sessions=sessions, events=events)
    report = build_report(connection, tz=ZoneInfo("UTC"))
    terms = {item["term"] for item in report["buckets"][0]["themes"]}
    assert "backtickword" not in terms
    assert "tildeword" not in terms
    assert "afterword" in terms
    connection.close()


def test_theme_strip_pasted_prompt_forms_and_unclosed_fence():
    text = (
        "PS C:\\Users\\x\\proj> git status\n"
        "PS /work/project> make test\n"
        "dev@box:~/work$ make check\n"
        "$ ls -la\n"
        "value = '$ keep this'\n"
        "\x60\x60\x60text\n"
        "removed through eof"
    )
    stripped = strip_pasted(text)
    assert "git status" in stripped
    assert "make test" in stripped
    assert "make check" in stripped
    assert "ls -la" in stripped
    assert "$ keep this" in stripped
    assert "removed through eof" not in stripped
    assert "PS C:" not in stripped
    assert "PS /work" not in stripped
    assert "dev@box:" not in stripped
    assert "\n$ ls -la" not in stripped


def test_theme_strip_pasted_keeps_prose_that_only_resembles_a_prompt():
    assert strip_pasted("ops@node:/srv# systemctl status\n") == "systemctl status\n"
    assert strip_pasted("ops@node:~$\n") == "\n"
    prose = "alice@example.com: see issue #3 now\n"
    assert strip_pasted(prose) == prose


def test_themes_do_not_change_human_character_count():
    session = [("fenced.jsonl", "fenced", "codex", dt("2026-03-01"), "cli")]
    prose = "before\n\x60\x60\x60\ninsidecode\n\x60\x60\x60\nafter"
    events = [("fenced.jsonl", "fenced", 0, dt("2026-03-01"), "user", prose, "{}", False, False)]
    connection = db_with_data(sessions=session, events=events)
    report = build_report(connection, tz=ZoneInfo("UTC"))
    assert report["buckets"][0]["human_chars"] == len(prose)
    connection.close()


def test_themes_zero_disables_theme_derivation():
    session = [("empty.jsonl", "empty", "codex", dt("2026-03-01"), "cli")]
    events = [
        ("empty.jsonl", "empty", 0, dt("2026-03-01"), "user", "quartzword", "{}", False, False)
    ]
    connection = db_with_data(sessions=session, events=events)
    report = build_report(connection, tz=ZoneInfo("UTC"), themes=0)
    assert report["buckets"][0]["themes"] == []
    connection.close()


def test_theme_cli_json_text_and_invalid_timezone(tmp_path, capsys):
    source = tmp_path / "transcripts"
    source.mkdir()
    transcript = source / "synthetic.jsonl"
    records = [
        {
            "type": "user",
            "uuid": "synthetic-user",
            "parentUuid": None,
            "sessionId": "synthetic-session",
            "timestamp": "2026-03-01T10:00:00Z",
            "cwd": "/synthetic/work",
            "message": {"role": "user", "content": [{"type": "text", "text": "synthetic words"}]},
        }
    ]
    transcript.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    database = tmp_path / "synthetic.duckdb"
    build([source], database)

    connection = duckdb.connect(str(database))
    connection.execute(
        """
        INSERT INTO tool_calls (tool_use_id, session_id, file_path, source, ts, result_text)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            "synthetic-tool",
            "synthetic-session",
            str(transcript),
            "claude_code",
            dt("2026-03-01T10:00:00"),
            "https://github.com/acme/widgets/issues/5",
        ],
    )
    connection.close()

    assert main(["review", "--db", str(database), "--json", "--tz", "UTC"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"tz", "bucket", "since", "until", "gh", "owners", "owners_inferred", "references", "undated", "buckets"}
    assert "themes" in payload["buckets"][0]
    assert main(["review", "--db", str(database), "--themes", "0", "--json", "--tz", "UTC"]) == 0
    disabled = json.loads(capsys.readouterr().out)
    assert disabled["buckets"][0]["themes"] == []
    assert main(["review", "--db", str(database), "--tz", "UTC"]) == 0
    rendered = capsys.readouterr().out
    assert "2026-03" in rendered
    assert "acme/widgets:" in rendered

    with pytest.raises(SystemExit) as error:
        main(["review", "--db", str(database), "--tz", "Not/AZone"])
    assert error.value.code == 2


def test_owner_filters_foreign_keys_into_references_and_gh_calls():
    connection = db_with_data(
        calls=[
            ("own", "sid-own", "own.jsonl", "claude_code", dt("2026-03-01T10:00:00"),
             "https://github.com/acme/widgets/issues/1"),
            ("foreign-a", "sid-a", "a.jsonl", "cursor", dt("2026-03-02T10:00:00"),
             "https://github.com/zeta/other/issues/2"),
            ("foreign-b", "sid-b", "b.jsonl", "codex", dt("2026-03-03T10:00:00"),
             "https://github.com/zeta/other/issues/3"),
            ("foreign-undated", "sid-u", "u.jsonl", "cursor", None,
             "https://github.com/zeta/other/issues/4"),
            ("drive-undated", "sid-d", "d.jsonl", "cursor", None,
             "https://docs.google.com/document/d/owner_drive/edit"),
        ]
    )
    seen = []

    def fake_gh(argv):
        seen.append(argv)
        return True, {"created_at": "2026-03-01T10:00:00Z", "user": {"login": "acme"}}, ""

    report = build_report(connection, tz=ZoneInfo("UTC"), owners=["ACME"], gh=True, gh_call=fake_gh)
    outputs = [item for bucket in report["buckets"] for item in bucket["outputs"]]
    assert [item["repo"] for item in outputs] == ["acme/widgets"]
    assert report["references"] == {"count": 3, "by_source": {"codex": 1, "cursor": 2}}
    assert report["undated"] == {"count": 1, "by_source": {"cursor": 1}}
    assert seen == [["gh", "api", "repos/acme/widgets/issues/1"]]
    connection.close()


def test_owner_default_inference_counts_distinct_github_keys():
    connection = db_with_data(
        calls=[
            ("a", "s1", "a.jsonl", "claude_code", dt("2026-03-01"), "https://github.com/acme/a/issues/1"),
            ("z1", "s2", "z1.jsonl", "claude_code", dt("2026-03-02"), "https://github.com/zeta/z/issues/1"),
            ("z2", "s3", "z2.jsonl", "claude_code", dt("2026-03-03"), "https://github.com/zeta/z/issues/2"),
            ("z2-again", "s4", "z4.jsonl", "claude_code", dt("2026-03-04"), "https://github.com/zeta/z/issues/2"),
        ]
    )
    report = build_report(connection, tz=ZoneInfo("UTC"))
    assert report["owners"] == ["zeta"]
    assert report["owners_inferred"] is True
    assert report["references"]["count"] == 1
    connection.close()


def test_owner_explicit_list_overrides_inferred_default():
    connection = db_with_data(
        calls=[
            ("a", "s1", "a.jsonl", "claude_code", dt("2026-03-01"), "https://github.com/acme/a/issues/1"),
            ("z1", "s2", "z1.jsonl", "claude_code", dt("2026-03-02"), "https://github.com/zeta/z/issues/1"),
            ("z2", "s3", "z2.jsonl", "claude_code", dt("2026-03-03"), "https://github.com/zeta/z/issues/2"),
        ]
    )
    report = build_report(connection, tz=ZoneInfo("UTC"), owners=["acme"])
    assert report["owners"] == ["acme"]
    assert report["owners_inferred"] is False
    assert [item["repo"] for item in report["buckets"][0]["outputs"]] == ["acme/a"]
    connection.close()


def test_owner_multiple_case_insensitive_owners_include_both():
    connection = db_with_data(
        calls=[
            ("a", "s1", "a.jsonl", "claude_code", dt("2026-03-01"), "https://github.com/acme/a/issues/1"),
            ("a-case", "s2", "a2.jsonl", "claude_code", dt("2026-03-02"), "https://github.com/Acme/b/issues/2"),
            ("z", "s3", "z.jsonl", "cursor", dt("2026-03-03"), "https://github.com/zeta/z/issues/3"),
        ]
    )
    report = build_report(connection, tz=ZoneInfo("UTC"), owners=["ACME", "ZETA"])
    outputs = [item for bucket in report["buckets"] for item in bucket["outputs"]]
    assert report["owners"] == ["acme", "zeta"]
    assert {item["repo"] for item in outputs} == {"acme/a", "acme/b", "zeta/z"}
    assert report["references"]["count"] == 0
    connection.close()


def test_owner_cli_repeatable_on_built_database(tmp_path, capsys):
    source = tmp_path / "transcripts"
    source.mkdir()
    transcript = source / "synthetic.jsonl"
    transcript.write_text(
        json.dumps({
            "type": "user", "uuid": "synthetic-user", "parentUuid": None,
            "sessionId": "synthetic-session", "timestamp": "2026-03-01T10:00:00Z",
            "cwd": "/synthetic/work",
            "message": {"role": "user", "content": [{"type": "text", "text": "synthetic words"}]},
        }) + "\n",
        encoding="utf-8",
    )
    database = tmp_path / "synthetic.duckdb"
    build([source], database)
    connection = duckdb.connect(str(database))
    connection.executemany(
        """INSERT INTO tool_calls
           (tool_use_id, session_id, file_path, source, ts, result_text)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [
            ("a", "synthetic-session", str(transcript), "claude_code", dt("2026-03-01T10:00:00"),
             "https://github.com/a/repo/issues/1"),
            ("b", "synthetic-session", str(transcript), "claude_code", dt("2026-03-01T10:01:00"),
             "https://github.com/b/repo/issues/2"),
        ],
    )
    connection.close()

    assert main(["review", "--db", str(database), "--owner", "a", "--owner", "b", "--json", "--tz", "UTC"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["owners"] == ["a", "b"]
    assert report["owners_inferred"] is False
    assert {item["repo"] for item in report["buckets"][0]["outputs"]} == {"a/repo", "b/repo"}


def test_session_id_counts_distinct_sessions_by_source_and_file_fallback():
    connection = db_with_data(
        sessions=[
            ("same-early.jsonl", "shared", "claude_code", dt("2026-03-01T10:00:00"), "sdk-worker"),
            ("same-late.jsonl", "shared", "claude_code", dt("2026-04-01T10:00:00"), "cli"),
            ("other-source.jsonl", "shared", "codex", dt("2026-03-02T10:00:00"), "cli"),
            ("null-a.jsonl", None, "cursor", dt("2026-03-03T10:00:00"), "cli"),
            ("null-b.jsonl", None, "cursor", dt("2026-03-04T10:00:00"), "sdk-worker"),
        ]
    )
    report = build_report(connection, tz=ZoneInfo("UTC"))
    buckets = {bucket["label"]: bucket for bucket in report["buckets"]}
    march = buckets["2026-03"]["sessions"]
    assert march["claude_code"] == {"sessions": 1, "non_headless": 1}
    assert march["codex"] == {"sessions": 1, "non_headless": 1}
    assert march["cursor"] == {"sessions": 2, "non_headless": 1}
    assert "2026-04" not in buckets
    connection.close()
