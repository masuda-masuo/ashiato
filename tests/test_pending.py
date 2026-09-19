"""Tests for ashiato.pending: open items left in compaction summaries.

Fixtures are hand-built JSONL transcripts run through the real build pipeline,
one transcript file per session, in a private ``tmp_path`` -- never in
``tests/fixtures/``, which other tests glob recursively.  A compaction summary
is a user record whose *top-level* ``isCompactSummary`` is true with
``message.content`` a plain string, exactly as Claude Code writes it (a flag
inside ``message`` is not a summary).
"""

from __future__ import annotations

import io
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from ashiato.build import build
from ashiato.cli import main
from ashiato.pending import (
    Reference,
    _is_compact_summary,
    bare_refs,
    explicit_refs,
    extract_items,
    extract_section,
    is_deferred,
    item_status,
    run,
)

#: The example from the brief: three bullets, the third deferred with bare
#: references that must resolve to the summary's most frequent explicit repo.
EXAMPLE_SUMMARY = """\
7. Pending Tasks:
   - PR https://github.com/masuda-masuo/sunaba/pull/416 (`sandbox_issue_write`) is CI-green and approved, but not yet reported as merged.
   - Issue masuda-masuo/sunaba#415 (sandbox review scripts infrastructure) is registered but no implementation work has started.
   - Discussed-but-not-started (do NOT begin without user confirmation): real-Docker E2E of the proxy flow (...); proxy-image pin/digest design (...); #356 remainder (...); #360 first-class write tools; ...

8. Current Work:
"""

#: One item per status the --gh mode has to distinguish.
STATUS_SUMMARY = """\
7. Pending Tasks:
   - PR masuda-masuo/sunaba#101 is open and waiting.
   - Issue masuda-masuo/sunaba#102 and masuda-masuo/sunaba#103 are done.
   - PR masuda-masuo/sunaba#999 cannot be looked up.
   - PR masuda-masuo/sunaba#104 was merged.
8. Current Work:
"""

#: A real Claude Code compaction-summary record (issue #52): ``isCompactSummary``
#: is a *top-level* key, a sibling of ``uuid``, and ``message.content`` is one
#: plain string.  A Pending Tasks section is appended to the content.
REAL_COMPACTION_RECORD: dict[str, Any] = {
    "parentUuid": "aa8fffeb-8083-4ec9-a925-82031fcd1888",
    "isSidechain": False,
    "promptId": "3d06a593-2848-47b1-9b0c-47873c296d37",
    "type": "user",
    "isVisibleInTranscriptOnly": True,
    "isCompactSummary": True,
    "uuid": "f295b20a-6142-4886-b1bc-f9336953cf2b",
    "timestamp": "2026-07-21T22:58:28.556Z",
    "userType": "external",
    "entrypoint": "cli",
    "cwd": "/home/masuda/dev/projects/claude",
    "sessionId": "05d783f6-87a9-43a0-8e96-ef025fe60bf7",
    "version": "2.1.216",
    "gitBranch": "HEAD",
    "message": {
        "role": "user",
        "content": (
            "This session is being continued from a previous conversation that ran "
            "out of context. ...\n\n"
            "7. Pending Tasks:\n"
            "   - Issue masuda-masuo/ashiato#52 is still open.\n"
            "8. Current Work:\n"
        ),
    },
}


# ---------------------------------------------------------------- fixtures


def user(text: str, *, compact: bool = False) -> dict[str, Any]:
    """A user text event; *compact* marks it as a compaction summary."""
    return {"role": "user", "text": text, "kind": "text", "compact": compact}


def ai_title(text: str) -> dict[str, Any]:
    return {"kind": "ai-title", "text": text}


def custom_title(text: str) -> dict[str, Any]:
    return {"kind": "custom-title", "text": text}


def _records(
    session_id: str, day: str, events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    parent = None
    for index, event in enumerate(events):
        uuid = f"{session_id}-{index}"
        ts = f"{day}T10:{index // 60:02d}:{index % 60:02d}Z"
        if event["kind"] in ("ai-title", "custom-title"):
            records.append(
                {
                    "type": event["kind"],
                    "uuid": uuid,
                    "parentUuid": parent,
                    "sessionId": session_id,
                    "timestamp": ts,
                    "cwd": f"/work/{session_id}",
                    "message": {"content": event["text"]},
                }
            )
            parent = uuid
            continue
        record: dict[str, Any] = {
            "type": event["role"],
            "uuid": uuid,
            "parentUuid": parent,
            "sessionId": session_id,
            "timestamp": ts,
            "cwd": f"/work/{session_id}",
            # The real Claude Code shape: the flag is a top-level key of the
            # record and ``message.content`` is one plain string.
            "message": {"role": event["role"], "content": event["text"]},
        }
        if event.get("compact"):
            record["isCompactSummary"] = True
        records.append(record)
        parent = uuid
    return records


def make_db(
    tmp_path: Path, sessions: list[tuple[str, str, list[dict[str, Any]]]]
) -> Path:
    """Build a DuckDB from ``(session_id, "YYYY-MM-DD", events)`` sessions."""
    directory = tmp_path / "transcripts"
    directory.mkdir(exist_ok=True)
    for index, (session_id, day, events) in enumerate(sessions):
        path = directory / f"{index:02d}-{session_id}.jsonl"
        lines = [json.dumps(record) for record in _records(session_id, day, events)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    db_path = tmp_path / "pending.duckdb"
    build([directory], db_path)
    return db_path


def make_raw_db(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    """Build a DuckDB from literal JSONL records, without the fixture shape."""
    directory = tmp_path / "transcripts"
    directory.mkdir(exist_ok=True)
    path = directory / "00-raw.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    db_path = tmp_path / "pending.duckdb"
    build([directory], db_path)
    return db_path


def _run_cli_json(
    db_path: Path, *args: str, capsys: pytest.CaptureFixture[str]
) -> dict[str, Any]:
    code = main(["pending", "--db", str(db_path), "--json", *args])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    return json.loads(captured.out)


def _run_direct(db_path: Path, **kwargs: Any) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    code = run(db_path, out=out, err=err, **kwargs)
    return code, out.getvalue(), err.getvalue()


def _run_direct_json(db_path: Path, **kwargs: Any) -> dict[str, Any]:
    out = io.StringIO()
    err = io.StringIO()
    code = run(db_path, json_output=True, out=out, err=err, **kwargs)
    assert code == 0, err.getvalue()
    return json.loads(out.getvalue())


# ---------------------------------------------------------------- unit tests


def test_extract_section_requires_pending_tasks_heading() -> None:
    assert extract_section("7. Current Work:\n   - something\n") is None
    assert extract_section("no numbered headings here") is None
    assert extract_section("") is None


def test_extract_section_bounds_at_next_heading() -> None:
    section = extract_section(EXAMPLE_SUMMARY)
    assert section is not None
    assert section.startswith("   - PR https://github.com/masuda-masuo/sunaba/pull/416")
    assert "8. Current Work:" not in section


def test_extract_items_strips_bold_and_collapses_whitespace() -> None:
    section = "   - **Deferred**: a long\n     wrapped line\n   - short\n"
    assert extract_items(section) == ["Deferred: a long wrapped line", "short"]


def test_extract_items_only_starts_at_minimum_indentation() -> None:
    section = (
        "   - first item\n"
        "     - a deeper bullet, continuation text\n"
        "   - second item\n"
    )
    assert extract_items(section) == [
        "first item - a deeper bullet, continuation text",
        "second item",
    ]


def test_extract_items_keeps_fenced_lines_in_the_current_item() -> None:
    section = (
        "   - first item\n"
        "     ```\n"
        "   - a bullet-lookalike inside the fence\n"
        "     ```\n"
        "   - second item\n"
    )
    assert extract_items(section) == [
        "first item ``` - a bullet-lookalike inside the fence ```",
        "second item",
    ]


def test_is_compact_summary_requires_the_top_level_flag() -> None:
    real = {
        "type": "user",
        "isCompactSummary": True,
        "message": {"role": "user", "content": "x"},
    }
    assert _is_compact_summary(json.dumps(real))
    nested_message = {
        "type": "user",
        "message": {"role": "user", "isCompactSummary": True, "content": "x"},
    }
    assert not _is_compact_summary(json.dumps(nested_message))
    nested_block = {
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": "x", "isCompactSummary": True}],
        },
    }
    assert not _is_compact_summary(json.dumps(nested_block))
    assert not _is_compact_summary(None)
    assert not _is_compact_summary("{not json")
    assert not _is_compact_summary(json.dumps({"type": "user", "message": {"content": "x"}}))


def test_is_deferred_matches_the_spec_markers() -> None:
    assert is_deferred("Discussed-but-not-started (do NOT begin without user confirmation)")
    assert is_deferred("Discussed but not started")
    assert is_deferred("not yet started")
    assert is_deferred("deferred until next quarter")
    assert is_deferred("future work")
    assert is_deferred("do not start this")
    assert not is_deferred("is registered but no implementation work has started")
    assert not is_deferred("not yet reported as merged")


def test_explicit_and_bare_refs() -> None:
    text = "see https://github.com/a/b/issues/12 and a/b#7 and #9 plus a/b#7 again"
    assert explicit_refs(text) == [("a", "b", 12), ("a", "b", 7)]
    assert bare_refs(text) == [9]


def test_item_status_matrix() -> None:
    assert item_status([]) == "unreferenced"
    assert item_status([Reference("o", "r", 1, "open")]) == "open"
    assert item_status(
        [Reference("o", "r", 1, "open"), Reference("o", "r", 2, "closed")]
    ) == "open"
    assert item_status(
        [Reference("o", "r", 1, "closed"), Reference("o", "r", 2, "merged")]
    ) == "resolved"
    assert item_status(
        [Reference("o", "r", 1, "closed"), Reference("o", "r", 2, "unknown")]
    ) == "unknown"
    assert item_status([Reference("o", "r", 1, "unchecked")]) == "unchecked"
    assert item_status([Reference(None, None, 3, "unresolved")]) == "unchecked"


# ---------------------------------------------------------------- criterion 1


def test_example_summary_yields_three_items_and_resolves_bare_refs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(EXAMPLE_SUMMARY, compact=True)])])
    payload = _run_cli_json(db, capsys=capsys)

    sessions = payload["sessions"]
    assert len(sessions) == 1
    items = sessions[0]["items"]
    assert len(items) == 3
    assert [item["deferred"] for item in items] == [False, False, True]
    assert [item["status"] for item in items] == ["unchecked", "unchecked", "unchecked"]

    def keys(item: dict[str, Any]) -> list[tuple[str | None, str | None, int]]:
        return [(r["owner"], r["repo"], r["number"]) for r in item["references"]]

    assert keys(items[0]) == [("masuda-masuo", "sunaba", 416)]
    assert keys(items[1]) == [("masuda-masuo", "sunaba", 415)]
    # The bare #356/#360 resolve to the summary's most frequent explicit repo.
    assert keys(items[2]) == [
        ("masuda-masuo", "sunaba", 356),
        ("masuda-masuo", "sunaba", 360),
    ]
    assert all(r["bare"] for r in items[2]["references"])
    assert payload["counts"]["unchecked"] == 3
    assert payload["counts"]["no_section"] == 0


# ---------------------------------------------------------------- criterion 2


def test_only_the_latest_summary_unless_all_summaries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = "7. Pending Tasks:\n   - Issue masuda-masuo/sunaba#1 first summary.\n\n8. Current Work:\n"
    second = "7. Pending Tasks:\n   - Issue masuda-masuo/sunaba#2 second summary.\n\n8. Current Work:\n"
    db = make_db(
        tmp_path,
        [
            (
                "ses-a",
                "2026-08-01",
                [
                    user(first, compact=True),
                    user("intervening conversation"),
                    user(second, compact=True),
                ],
            )
        ],
    )

    payload = _run_cli_json(db, capsys=capsys)
    items = payload["sessions"][0]["items"]
    assert len(items) == 1
    assert items[0]["references"][0]["number"] == 2

    payload = _run_cli_json(db, "--all-summaries", capsys=capsys)
    sessions = payload["sessions"]
    assert len(sessions) == 2
    numbers = sorted(
        r["number"]
        for session in sessions
        for item in session["items"]
        for r in item["references"]
    )
    assert numbers == [1, 2]


# ---------------------------------------------------------------- criterion 3


def test_summary_without_pending_section_counts_no_section(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = "7. Current Work:\n   - Something in progress.\n"
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(summary, compact=True)])])
    payload = _run_cli_json(db, capsys=capsys)
    assert payload["sessions"] == []
    assert payload["counts"]["no_section"] == 1


# ---------------------------------------------------------------- criterion 4


def test_without_gh_the_lookup_is_never_called(tmp_path: Path) -> None:
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(EXAMPLE_SUMMARY, compact=True)])])
    calls: list[list[str]] = []

    def fake(argv: list[str]) -> tuple[bool, Any, str]:
        calls.append(list(argv))
        return True, {"state": "open"}, ""

    code, out, _err = _run_direct(db, gh_call=fake)
    assert code == 0
    assert calls == []
    assert "=unchecked" in out


# ---------------------------------------------------------------- criteria 5 and 6


def test_with_gh_open_resolved_and_unknown(tmp_path: Path) -> None:
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(STATUS_SUMMARY, compact=True)])])
    calls: list[tuple[str, str, int]] = []

    def fake(argv: list[str]) -> tuple[bool, Any, str]:
        if argv[1] == "repo":
            return True, [{"name": "sunaba"}], ""
        parts = argv[2].split("/")
        number = int(parts[-1])
        calls.append((parts[1], parts[2], number))
        if number == 999:
            return False, None, "graphql: Not Found"
        if number == 101:
            return True, {"state": "open"}, ""
        if number in (102, 104):
            return True, {"state": "closed"}, ""
        if number == 103:
            return True, {"state": "closed", "pull_request": {"merged_at": "2026-07-01T00:00:00Z"}}, ""
        raise AssertionError(f"unexpected lookup: {argv}")

    payload = _run_direct_json(db, use_gh=True, gh_call=fake)

    def first_number(item: dict[str, Any]) -> int:
        return item["references"][0]["number"]

    items = payload["sessions"][0]["items"]
    by_number = {first_number(item): item for item in items}
    # Default output hides resolved items, so only 101 and 999 are listed.
    assert set(by_number) == {101, 999}
    assert by_number[101]["status"] == "open"
    assert by_number[999]["status"] == "unknown"
    assert by_number[999]["references"][0]["error"] == "graphql: Not Found"
    # Counts cover every item, resolved ones included.
    assert payload["counts"] == {
        "open": 1,
        "resolved": 2,
        "unreferenced": 0,
        "unknown": 1,
        "unchecked": 0,
        "no_section": 0,
    }

    # Each unique (owner, repo, number) is looked up exactly once, in order.
    assert calls == [
        ("masuda-masuo", "sunaba", 101),
        ("masuda-masuo", "sunaba", 102),
        ("masuda-masuo", "sunaba", 103),
        ("masuda-masuo", "sunaba", 999),
        ("masuda-masuo", "sunaba", 104),
    ]

    calls.clear()
    payload_all = _run_direct_json(db, use_gh=True, gh_call=fake, show_resolved=True)
    items_all = payload_all["sessions"][0]["items"]
    statuses_all = {first_number(item): item["status"] for item in items_all}
    # Item 2 carries refs 102 and 103; both closed/merged, so it is resolved.
    assert statuses_all[102] == "resolved"
    assert statuses_all[104] == "resolved"
    assert len(items_all) == 4
    # The second run looks everything up again, still once per reference.
    assert calls == [
        ("masuda-masuo", "sunaba", 101),
        ("masuda-masuo", "sunaba", 102),
        ("masuda-masuo", "sunaba", 103),
        ("masuda-masuo", "sunaba", 999),
        ("masuda-masuo", "sunaba", 104),
    ]


def test_gh_call_receives_only_gh_api_and_endpoint(tmp_path: Path) -> None:
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(STATUS_SUMMARY, compact=True)])])
    recorded: list[list[str]] = []

    def fake(argv: list[str]) -> tuple[bool, Any, str]:
        recorded.append(list(argv))
        if argv[1] == "repo":
            return True, [{"name": "sunaba"}], ""
        return True, {"state": "open"}, ""

    code, _out, _err = _run_direct(db, use_gh=True, gh_call=fake)
    assert code == 0
    assert recorded
    # Exactly one `gh repo list <owner>` call per run, carrying only the owner
    # name as variable data.
    repo_list_calls = [argv for argv in recorded if argv[1] == "repo"]
    assert repo_list_calls == [
        ["gh", "repo", "list", "masuda-masuo", "--limit", "200", "--json", "name"]
    ]
    for argv in recorded:
        if argv[1] == "repo":
            continue
        assert argv == ["gh", "api", argv[2]]
        assert re.fullmatch(r"repos/masuda-masuo/sunaba/issues/\d+", argv[2]) is not None


def test_gh_lookup_is_deduped_across_sessions(tmp_path: Path) -> None:
    summary = "7. Pending Tasks:\n   - Issue masuda-masuo/sunaba#7 pending.\n\n8. Current Work:\n"
    db = make_db(
        tmp_path,
        [
            ("ses-a", "2026-08-01", [user(summary, compact=True)]),
            ("ses-b", "2026-08-02", [user(summary, compact=True)]),
        ],
    )
    calls: list[list[str]] = []

    def fake(argv: list[str]) -> tuple[bool, Any, str]:
        if argv[1] == "repo":
            return True, [{"name": "sunaba"}], ""
        calls.append(list(argv))
        return True, {"state": "open"}, ""

    payload = _run_direct_json(db, use_gh=True, gh_call=fake)
    # One lookup for the issue both sessions reference.
    assert len(calls) == 1
    assert calls[0] == ["gh", "api", "repos/masuda-masuo/sunaba/issues/7"]
    for session in payload["sessions"]:
        assert session["items"][0]["references"][0]["state"] == "open"


def test_gh_timeout_is_unknown_and_the_run_completes(tmp_path: Path) -> None:
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(STATUS_SUMMARY, compact=True)])])
    calls: list[list[str]] = []

    def fake(argv: list[str]) -> tuple[bool, Any, str]:
        if argv[1] == "repo":
            return True, [{"name": "sunaba"}], ""
        calls.append(list(argv))
        number = int(argv[2].split("/")[-1])
        if number == 999:
            raise subprocess.TimeoutExpired(cmd=argv, timeout=30)
        if number == 101:
            return True, {"state": "open"}, ""
        if number in (102, 104):
            return True, {"state": "closed"}, ""
        if number == 103:
            return True, {"state": "closed", "pull_request": {"merged_at": "2026-07-01T00:00:00Z"}}, ""
        raise AssertionError(f"unexpected lookup: {argv}")

    payload = _run_direct_json(db, use_gh=True, gh_call=fake)

    def first_number(item: dict[str, Any]) -> int:
        return item["references"][0]["number"]

    items = payload["sessions"][0]["items"]
    by_number = {first_number(item): item for item in items}
    # Default output hides resolved items, so only 101 and 999 are listed.
    assert set(by_number) == {101, 999}
    assert by_number[101]["status"] == "open"
    assert by_number[999]["status"] == "unknown"
    assert "timed out" in by_number[999]["references"][0]["error"]
    assert payload["counts"]["unknown"] == 1


# ---------------------------------------------------------------- criterion 7


def test_item_without_references_is_unreferenced_and_shown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = "7. Pending Tasks:\n   - Continue the widget refactor after the merge.\n\n8. Current Work:\n"
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(summary, compact=True)])])
    payload = _run_cli_json(db, capsys=capsys)
    items = payload["sessions"][0]["items"]
    assert len(items) == 1
    assert items[0]["status"] == "unreferenced"
    assert items[0]["references"] == []


# ---------------------------------------------------------------- resolution


def test_bare_ref_resolves_to_repo_flag_when_summary_names_no_repo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = "7. Pending Tasks:\n   - Do the #42 follow-up.\n\n8. Current Work:\n"
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(summary, compact=True)])])
    payload = _run_cli_json(db, "--repo", "masuda-masuo/sunaba", capsys=capsys)
    reference = payload["sessions"][0]["items"][0]["references"][0]
    assert (reference["owner"], reference["repo"], reference["number"]) == (
        "masuda-masuo",
        "sunaba",
        42,
    )
    assert reference["bare"] is True


def test_bare_ref_with_no_repo_is_unresolved_and_never_checked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = "7. Pending Tasks:\n   - Do the #42 follow-up.\n\n8. Current Work:\n"
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(summary, compact=True)])])
    payload = _run_cli_json(db, capsys=capsys)
    reference = payload["sessions"][0]["items"][0]["references"][0]
    assert reference["owner"] is None
    assert reference["repo"] is None
    assert reference["state"] == "unresolved"
    assert payload["sessions"][0]["items"][0]["status"] == "unchecked"


# ---------------------------------------------------------------- windows and shape


def test_since_until_filter_by_summary_timestamp(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = "7. Pending Tasks:\n   - Issue masuda-masuo/sunaba#7 pending.\n\n8. Current Work:\n"
    db = make_db(
        tmp_path,
        [
            ("ses-a", "2026-08-01", [user(summary, compact=True)]),
            ("ses-b", "2026-08-03", [user(summary, compact=True)]),
        ],
    )
    payload = _run_cli_json(db, "--since", "2026-08-02", capsys=capsys)
    assert [s["session_id"] for s in payload["sessions"]] == ["ses-b"]
    payload = _run_cli_json(db, "--until", "2026-08-02", capsys=capsys)
    assert [s["session_id"] for s in payload["sessions"]] == ["ses-a"]


def test_text_output_shape_and_latest_title(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = make_db(
        tmp_path,
        [
            (
                "ses-a",
                "2026-08-01",
                [
                    user(EXAMPLE_SUMMARY, compact=True),
                    ai_title("Old title"),
                    custom_title("The real title"),
                ],
            )
        ],
    )
    code = main(["pending", "--db", str(db)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    out = captured.out
    assert "session ses-a" in out
    assert "summary 2026-08-01 10:00:00" in out
    assert "title: The real title" in out
    assert "[deferred]" in out
    assert "masuda-masuo/sunaba#356=unchecked" in out
    assert "no_section: 0" in out


def test_a_message_that_only_quotes_the_flag_is_not_a_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = make_db(
        tmp_path,
        [("ses-a", "2026-08-01", [user('The summary had "isCompactSummary": true in it.')])],
    )
    payload = _run_cli_json(db, capsys=capsys)
    assert payload["sessions"] == []
    assert payload["counts"]["no_section"] == 0


def test_exact_real_record_through_db_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = make_raw_db(tmp_path, [REAL_COMPACTION_RECORD])
    payload = _run_cli_json(db, capsys=capsys)

    sessions = payload["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "05d783f6-87a9-43a0-8e96-ef025fe60bf7"
    items = sessions[0]["items"]
    assert len(items) == 1
    reference = items[0]["references"][0]
    assert (reference["owner"], reference["repo"], reference["number"]) == (
        "masuda-masuo",
        "ashiato",
        52,
    )
    assert reference["state"] == "unchecked"
    assert payload["counts"]["no_section"] == 0


def test_flag_only_inside_message_is_not_a_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = "7. Pending Tasks:\n   - Issue masuda-masuo/sunaba#1.\n8. Current Work:\n"
    nested_message = {
        "type": "user",
        "uuid": "u0",
        "sessionId": "ses-a",
        "timestamp": "2026-08-01T10:00:00Z",
        "cwd": "/work/ses-a",
        "message": {"role": "user", "content": text, "isCompactSummary": True},
    }
    nested_block = {
        "type": "user",
        "uuid": "u1",
        "sessionId": "ses-a",
        "timestamp": "2026-08-01T10:00:01Z",
        "cwd": "/work/ses-a",
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": text, "isCompactSummary": True}],
        },
    }
    db = make_raw_db(tmp_path, [nested_message, nested_block])
    payload = _run_cli_json(db, capsys=capsys)
    assert payload["sessions"] == []
    assert payload["counts"]["no_section"] == 0


def test_nested_bullet_and_code_fence_form_one_item(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = (
        "7. Pending Tasks:\n"
        "   - Fix the flaky test, with a nested note and a snippet:\n"
        "     - a deeper bullet, still part of the item\n"
        "     ```\n"
        "   - a bullet-lookalike inside the fence\n"
        "     ```\n"
        "     after the fence, still the same item.\n"
        "   - The second real item.\n"
        "8. Current Work:\n"
    )
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(summary, compact=True)])])
    payload = _run_cli_json(db, capsys=capsys)
    items = payload["sessions"][0]["items"]
    assert len(items) == 2
    first = items[0]["text"]
    assert "a deeper bullet, still part of the item" in first
    assert "a bullet-lookalike inside the fence" in first
    assert "after the fence, still the same item." in first
    assert items[1]["text"] == "The second real item."


# ---------------------------------------------------------------- CLI errors


def test_run_missing_db_returns_one(tmp_path: Path) -> None:
    code, _out, err = _run_direct(tmp_path / "nope.duckdb")
    assert code == 1
    assert "no database" in err


def test_cli_rejects_malformed_repo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    summary = "7. Pending Tasks:\n   - Do the #42 follow-up.\n\n8. Current Work:\n"
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(summary, compact=True)])])
    with pytest.raises(SystemExit) as excinfo:
        main(["pending", "--db", str(db), "--repo", "not-a-repo"])
    assert excinfo.value.code == 2


# ---------------------------------------------------------------- repair 2: real-item forms


#: The four real item texts from the live DB (repair 2), as fixture items, with
#: explicit anchors naming kusabi and sunaba and extra kusabi mentions so kusabi
#: stays the summary's most frequent resolved repository.
REAL_ITEMS_SUMMARY = """\
7. Pending Tasks:
   - masuda-masuo/kusabi#10 and masuda-masuo/sunaba#20 are the explicit anchors.
   - kusabi#274: inspect the running chain's output before the next chain run.
   - sunaba PR #861 awaits the user's manual merge; then deploy after the merge.
   - sunaba#846 and #847 delegation: explicitly queued behind sunaba PR#848 merging, only after.
   - Issue #243 (just filed): companion process.exit() drops unflushed stdout before the process ends.
   - kusabi#300, kusabi#301 and kusabi#302 are more kusabi work, so kusabi is the most frequent.
8. Current Work:
"""


def test_real_item_texts_resolve_with_known_repos(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(REAL_ITEMS_SUMMARY, compact=True)])])
    payload = _run_cli_json(db, capsys=capsys)
    items = payload["sessions"][0]["items"]

    def keys(item: dict[str, Any]) -> list[tuple[str | None, str | None, int]]:
        return [(r["owner"], r["repo"], r["number"]) for r in item["references"]]

    # Item 0: the explicit anchors; owner defaults to the most frequent one.
    assert keys(items[0]) == [("masuda-masuo", "kusabi", 10), ("masuda-masuo", "sunaba", 20)]
    # Item 1: short form <name>#<n>.
    assert keys(items[1]) == [("masuda-masuo", "kusabi", 274)]
    # Item 2: short form <name> PR #<n>.
    assert keys(items[2]) == [("masuda-masuo", "sunaba", 861)]
    # Item 3: <name>#<n>, a bare #847 resolving to the nearest preceding repo,
    # and <name> PR#<n>.
    assert keys(items[3]) == [
        ("masuda-masuo", "sunaba", 846),
        ("masuda-masuo", "sunaba", 847),
        ("masuda-masuo", "sunaba", 848),
    ]
    assert items[3]["references"][1]["bare"] is True
    # Item 4: a bare #243 with no repo in its own text resolves to the summary's
    # most frequent resolved repository -- kusabi, thanks to the extra mentions.
    assert keys(items[4]) == [("masuda-masuo", "kusabi", 243)]
    assert items[4]["references"][0]["bare"] is True


def test_pr_number_forms_are_bare_and_never_owner_454(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = (
        "7. Pending Tasks:\n"
        "   - masuda-masuo/sunaba#20 is the known anchor.\n"
        "   - PR #454/PR#466 (the proxy flow) is queued behind the merge.\n"
        "8. Current Work:\n"
    )
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(summary, compact=True)])])
    payload = _run_cli_json(db, capsys=capsys)
    items = payload["sessions"][0]["items"]
    refs = items[1]["references"]
    # Both numbers are bare refs that resolve to the summary's known repo.
    assert [(r["owner"], r["repo"], r["number"]) for r in refs] == [
        ("masuda-masuo", "sunaba", 454),
        ("masuda-masuo", "sunaba", 466),
    ]
    assert all(r["bare"] for r in refs)
    # Never an owner 454 or a repo named PR anywhere.
    for item in items:
        for reference in item["references"]:
            assert reference["owner"] != "454"
            assert reference["repo"] != "PR"
    # The slash form must reject a numeric owner outright.
    assert explicit_refs("PR #454/PR#466") == []


def test_unknown_short_name_is_a_bare_ref_not_a_repo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = (
        "7. Pending Tasks:\n"
        "   - masuda-masuo/sunaba#20 is the known anchor.\n"
        "   - ship of#3 after the review.\n"
        "8. Current Work:\n"
    )
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(summary, compact=True)])])
    payload = _run_cli_json(db, capsys=capsys)
    items = payload["sessions"][0]["items"]
    ref = items[1]["references"][0]
    # `of` is not a known repository, so of#3 is a bare #3.
    assert (ref["owner"], ref["repo"], ref["number"]) == ("masuda-masuo", "sunaba", 3)
    assert ref["bare"] is True
    assert all(reference["repo"] != "of" for item in items for reference in item["references"])


def test_short_forms_with_no_owner_stay_unresolved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = "7. Pending Tasks:\n   - kusabi#274 needs review.\n\n8. Current Work:\n"
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(summary, compact=True)])])
    payload = _run_cli_json(db, capsys=capsys)
    ref = payload["sessions"][0]["items"][0]["references"][0]
    assert ref["owner"] is None
    assert ref["repo"] is None
    assert ref["state"] == "unresolved"


def test_owner_flag_overrides_the_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = (
        "7. Pending Tasks:\n"
        "   - masuda-masuo/sunaba#20 is the known anchor.\n"
        "   - sunaba#861 is a short form.\n"
        "8. Current Work:\n"
    )
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(summary, compact=True)])])
    payload = _run_cli_json(db, "--owner", "somebody-else", capsys=capsys)
    ref = payload["sessions"][0]["items"][1]["references"][0]
    assert (ref["owner"], ref["repo"], ref["number"]) == ("somebody-else", "sunaba", 861)


def test_gh_repo_list_called_once_with_only_the_owner(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = (
        "7. Pending Tasks:\n"
        "   - masuda-masuo/sunaba#20 is the anchor.\n"
        "   - kusabi#274 is a known short form; nope#12 is an unknown one.\n"
        "8. Current Work:\n"
    )
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(summary, compact=True)])])
    calls: list[list[str]] = []

    def fake(argv: list[str]) -> tuple[bool, Any, str]:
        calls.append(list(argv))
        if argv[1] == "repo":
            return True, [{"name": "kusabi"}, {"name": "sunaba"}], ""
        return True, {"state": "open"}, ""

    payload = _run_direct_json(db, use_gh=True, gh_call=fake)
    # Exactly one gh repo list call per run, carrying only the owner name.
    repo_lists = [argv for argv in calls if argv[1] == "repo"]
    assert repo_lists == [
        ["gh", "repo", "list", "masuda-masuo", "--limit", "200", "--json", "name"]
    ]
    api_calls = [argv for argv in calls if argv[1] == "api"]
    # The known short form is looked up; the unknown one never is.
    assert ["gh", "api", "repos/masuda-masuo/kusabi/issues/274"] in api_calls
    assert all("nope" not in " ".join(argv) for argv in api_calls)
    items = payload["sessions"][0]["items"]
    refs = items[1]["references"]
    assert [r["number"] for r in refs] == [274, 12]
    assert refs[1]["owner"] == "masuda-masuo"
    assert refs[1]["repo"] == "kusabi"  # nope#12 degrades to a bare #12
    assert refs[1]["bare"] is True