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
#: references that have no repository in their own item.
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
    # The bare #356/#360 have no repo in their own item (and no --repo),
    # so they are unresolved and never checked.
    assert keys(items[2]) == [(None, None, 356), (None, None, 360)]
    assert all(r["bare"] for r in items[2]["references"])
    assert all(r["state"] == "unresolved" for r in items[2]["references"])
    assert all(r.get("via") is None for r in items[2]["references"])
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
    assert "#356=unresolved" in out
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
#: explicit anchors naming kusabi and sunaba and extra kusabi mentions.
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
    # Item 4: a bare #243 with no repo in its own text and no --repo is
    # unresolved.
    assert keys(items[4]) == [(None, None, 243)]
    assert items[4]["references"][0]["bare"] is True
    assert items[4]["references"][0]["state"] == "unresolved"


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
    # Both numbers are bare refs with no repo in their own item, so they are
    # unresolved.
    assert [(r["owner"], r["repo"], r["number"]) for r in refs] == [
        (None, None, 454),
        (None, None, 466),
    ]
    assert all(r["bare"] for r in refs)
    assert all(r["state"] == "unresolved" for r in refs)
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
    # `of` is not a known repository, so of#3 is a bare #3; with no repo in
    # its own item it is unresolved.
    assert (ref["owner"], ref["repo"], ref["number"]) == (None, None, 3)
    assert ref["bare"] is True
    assert ref["state"] == "unresolved"
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


# ---------------------------------------------------------------- issue #55: <name> #N with whitespace


KNOWN_NAME_SPACE_SUMMARY = """\
7. Pending Tasks:
   - masuda-masuo/sagasu#99 is the anchor.
   - sagasu #12 closed, but recheck.
   - see #34 the follow-up.
8. Current Work:
"""


def test_known_name_with_space_before_hash(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """<name> #N with whitespace resolves as a non-bare short form."""
    db = make_db(
        tmp_path,
        [("ses-a", "2026-08-01", [user(KNOWN_NAME_SPACE_SUMMARY, compact=True)])],
    )
    payload = _run_cli_json(db, capsys=capsys)
    items = payload["sessions"][0]["items"]

    def keys(item: dict[str, Any]) -> list[tuple[str | None, str | None, int]]:
        return [(r["owner"], r["repo"], r["number"]) for r in item["references"]]

    # Item 0: explicit ref
    assert keys(items[0]) == [("masuda-masuo", "sagasu", 99)]
    # Item 1: sagasu #12 → known name with space, non-bare
    assert keys(items[1]) == [("masuda-masuo", "sagasu", 12)]
    assert items[1]["references"][0]["bare"] is False
    # Item 2: see #34 → bare with no repo in its own item, unresolved
    assert keys(items[2]) == [(None, None, 34)]
    assert items[2]["references"][0]["bare"] is True
    assert items[2]["references"][0]["state"] == "unresolved"


# ---------------------------------------------------------------- issue #55: bare repo-name word


BARE_NAME_SUMMARY = """\
7. Pending Tasks:
   - masuda-masuo/shiori#1 is the anchor.
   - shiori open issues after this session: zero (all of #288/#271/#75).
8. Current Work:
"""


def test_bare_name_mention_resolves_later_bare_refs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bare whole word that is a known repo name sets the nearest preceding repo."""
    db = make_db(
        tmp_path,
        [("ses-a", "2026-08-01", [user(BARE_NAME_SUMMARY, compact=True)])],
    )
    payload = _run_cli_json(db, capsys=capsys)
    items = payload["sessions"][0]["items"]

    def keys(item: dict[str, Any]) -> list[tuple[str | None, str | None, int]]:
        return [(r["owner"], r["repo"], r["number"]) for r in item["references"]]

    # Item 0: explicit anchor
    assert keys(items[0]) == [("masuda-masuo", "shiori", 1)]
    # Item 1: All three bare refs resolve to shiori via the name mention
    assert keys(items[1]) == [
        ("masuda-masuo", "shiori", 288),
        ("masuda-masuo", "shiori", 271),
        ("masuda-masuo", "shiori", 75),
    ]
    assert all(r["bare"] for r in items[1]["references"])
    assert all(r["via"] == "name" for r in items[1]["references"])


def test_shiori_case_insensitive_does_not_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Capitalised 'Shiori' does not match the lowercase known name 'shiori'."""
    summary = (
        "7. Pending Tasks:\n"
        "   - masuda-masuo/shiori#1 is the anchor.\n"
        "   - Shiori open issues: #288.\n"
        "8. Current Work:\n"
    )
    db = make_db(
        tmp_path,
        [("ses-a", "2026-08-01", [user(summary, compact=True)])],
    )
    payload = _run_cli_json(db, capsys=capsys)
    items = payload["sessions"][0]["items"]
    ref = items[1]["references"][0]
    # 'Shiori' is not a known name (case-sensitive), so #288 is a bare ref
    # with no preceding repository in its own item: unresolved.
    assert ref["owner"] is None
    assert ref["repo"] is None
    assert ref["bare"] is True
    assert ref["state"] == "unresolved"
    assert ref["via"] is None


def test_shiori_hyphenated_does_not_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """'shiori-demo' is not a bare name mention (longer word)."""
    summary = (
        "7. Pending Tasks:\n"
        "   - masuda-masuo/shiori#1 is the anchor.\n"
        "   - shiori-demo has issues #99.\n"
        "8. Current Work:\n"
    )
    db = make_db(
        tmp_path,
        [("ses-a", "2026-08-01", [user(summary, compact=True)])],
    )
    payload = _run_cli_json(db, capsys=capsys)
    items = payload["sessions"][0]["items"]
    ref = items[1]["references"][0]
    assert ref["number"] == 99
    assert ref["bare"] is True
    # shiori-demo is not a known name, so #99 has no repo in its own item
    assert ref["state"] == "unresolved"
    assert ref["via"] is None


def test_shiori_underscore_does_not_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """'shiori_eval' is not a bare name mention (longer word)."""
    summary = (
        "7. Pending Tasks:\n"
        "   - masuda-masuo/shiori#1 is the anchor.\n"
        "   - shiori_eval needs testing #50.\n"
        "8. Current Work:\n"
    )
    db = make_db(
        tmp_path,
        [("ses-a", "2026-08-01", [user(summary, compact=True)])],
    )
    payload = _run_cli_json(db, capsys=capsys)
    items = payload["sessions"][0]["items"]
    ref = items[1]["references"][0]
    assert ref["number"] == 50
    assert ref["bare"] is True
    assert ref["state"] == "unresolved"
    assert ref["via"] is None


def test_path_shiori_x_does_not_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """'shiori' inside a path like 'path/shiori/x' is not a bare name mention."""
    summary = (
        "7. Pending Tasks:\n"
        "   - masuda-masuo/shiori#1 is the anchor.\n"
        "   - check path/shiori/x for #77.\n"
        "8. Current Work:\n"
    )
    db = make_db(
        tmp_path,
        [("ses-a", "2026-08-01", [user(summary, compact=True)])],
    )
    payload = _run_cli_json(db, capsys=capsys)
    items = payload["sessions"][0]["items"]
    ref = items[1]["references"][0]
    assert ref["number"] == 77
    assert ref["bare"] is True
    assert ref["state"] == "unresolved"
    assert ref["via"] is None


def test_name_mention_after_bare_ref_does_not_apply(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A name mention after the bare ref does not apply to it (order matters)."""
    summary = (
        "7. Pending Tasks:\n"
        "   - masuda-masuo/shiori#1 is the anchor.\n"
        "   - fix #12 then check shiori for #34.\n"
        "8. Current Work:\n"
    )
    db = make_db(
        tmp_path,
        [("ses-a", "2026-08-01", [user(summary, compact=True)])],
    )
    payload = _run_cli_json(db, capsys=capsys)
    items = payload["sessions"][0]["items"]

    refs = items[1]["references"]
    # #12 comes before the shiori name mention and has no other preceding
    # entry, so it is unresolved; #34 comes after the mention, so it uses the
    # name mention (via "name").
    assert refs[0]["number"] == 12
    assert refs[0]["bare"] is True
    assert refs[0]["state"] == "unresolved"
    assert refs[0]["via"] is None
    assert refs[1]["number"] == 34
    assert refs[1]["bare"] is True
    assert refs[1]["via"] == "name"


# ------------------------------------------------ issue #55 repair: nearest preceding by position


def test_name_mention_beats_farther_resolved_ref(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The nearest preceding entry wins: the shiori mention is closer to
    #288/#271 than the earlier kusabi#12 resolved ref."""
    summary = (
        "7. Pending Tasks:\n"
        "   - masuda-masuo/kusabi#1 and masuda-masuo/shiori#1 are the anchors.\n"
        "   - kusabi#12 is done; shiori open issues: #288/#271.\n"
        "8. Current Work:\n"
    )
    db = make_db(
        tmp_path,
        [("ses-a", "2026-08-01", [user(summary, compact=True)])],
    )
    payload = _run_cli_json(db, capsys=capsys)
    items = payload["sessions"][0]["items"]

    refs = items[1]["references"]
    assert [r["number"] for r in refs] == [12, 288, 271]
    # #12 is a known short form, non-bare.
    assert refs[0]["repo"] == "kusabi"
    assert refs[0]["bare"] is False
    # The shiori mention precedes and is closer than kusabi#12, so it wins.
    for ref in refs[1:]:
        assert ref["owner"] == "masuda-masuo"
        assert ref["repo"] == "shiori"
        assert ref["bare"] is True
        assert ref["via"] == "name"


def test_resolved_ref_and_mention_interleave(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """#288 uses the earlier shiori mention; the kusabi ref that follows
    becomes the nearest entry for #13."""
    summary = (
        "7. Pending Tasks:\n"
        "   - masuda-masuo/kusabi#1 and masuda-masuo/shiori#1 are the anchors.\n"
        "   - shiori open issues #288, then kusabi#12 and #13.\n"
        "8. Current Work:\n"
    )
    db = make_db(
        tmp_path,
        [("ses-a", "2026-08-01", [user(summary, compact=True)])],
    )
    payload = _run_cli_json(db, capsys=capsys)
    items = payload["sessions"][0]["items"]

    refs = items[1]["references"]
    assert [r["number"] for r in refs] == [288, 12, 13]
    assert refs[0]["repo"] == "shiori"
    assert refs[0]["bare"] is True
    assert refs[0]["via"] == "name"
    assert refs[1]["repo"] == "kusabi"
    assert refs[1]["bare"] is False
    assert refs[2]["repo"] == "kusabi"
    assert refs[2]["bare"] is True
    assert refs[2]["via"] == "nearest"


def test_resolved_ref_closer_than_mention_wins(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """kusabi#12 is closer to #13 than the shiori mention, so via 'nearest'."""
    summary = (
        "7. Pending Tasks:\n"
        "   - masuda-masuo/kusabi#1 and masuda-masuo/shiori#1 are the anchors.\n"
        "   - shiori and kusabi#12: #13.\n"
        "8. Current Work:\n"
    )
    db = make_db(
        tmp_path,
        [("ses-a", "2026-08-01", [user(summary, compact=True)])],
    )
    payload = _run_cli_json(db, capsys=capsys)
    items = payload["sessions"][0]["items"]

    refs = items[1]["references"]
    assert [r["number"] for r in refs] == [12, 13]
    assert refs[0]["repo"] == "kusabi"
    assert refs[0]["bare"] is False
    assert refs[1]["repo"] == "kusabi"
    assert refs[1]["bare"] is True
    assert refs[1]["via"] == "nearest"


# ---------------------------------------------------------------- issue #55: 404 on inferred repo


def test_404_on_bare_ref_is_unresolved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A 404 on a bare (inferred) ref becomes 'unresolved', not 'unknown'."""
    summary = (
        "7. Pending Tasks:\n"
        "   - masuda-masuo/sunaba#10 is the anchor.\n"
        "   - shiori open issues: #999.\n"
        "8. Current Work:\n"
    )
    db = make_db(
        tmp_path,
        [("ses-a", "2026-08-01", [user(summary, compact=True)])],
    )

    def fake(argv: list[str]) -> tuple[bool, Any, str]:
        if argv[1] == "repo":
            return True, [{"name": "sunaba"}, {"name": "shiori"}], ""
        parts = argv[2].split("/")
        number = int(parts[-1])
        if number == 999:
            return False, None, "gh: Not Found (HTTP 404)"
        return True, {"state": "open"}, ""

    payload = _run_direct_json(db, use_gh=True, gh_call=fake)
    items = payload["sessions"][0]["items"]
    # #999 is bare, inferred to shiori, 404 → unresolved
    ref_999 = items[1]["references"][0]
    assert ref_999["number"] == 999
    assert ref_999["bare"] is True
    assert ref_999["state"] == "unresolved"
    assert "inferred repo masuda-masuo/shiori" in ref_999["error"]
    assert "HTTP 404" in ref_999["error"]
    # Item status is unchecked (unresolved refs → unchecked)
    assert items[1]["status"] == "unchecked"


def test_404_on_explicit_ref_is_unknown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A 404 on an explicit ref stays 'unknown' (real broken reference)."""
    summary = (
        "7. Pending Tasks:\n"
        "   - masuda-masuo/sunaba#999 is broken.\n"
        "8. Current Work:\n"
    )
    db = make_db(
        tmp_path,
        [("ses-a", "2026-08-01", [user(summary, compact=True)])],
    )

    def fake(argv: list[str]) -> tuple[bool, Any, str]:
        if argv[1] == "repo":
            return True, [{"name": "sunaba"}], ""
        return False, None, "gh: Not Found (HTTP 404)"

    payload = _run_direct_json(db, use_gh=True, gh_call=fake)
    items = payload["sessions"][0]["items"]
    ref = items[0]["references"][0]
    assert ref["number"] == 999
    assert ref["bare"] is False
    assert ref["state"] == "unknown"
    assert "HTTP 404" in ref["error"]
    assert items[0]["status"] == "unknown"


def test_non_404_failure_on_bare_ref_is_unknown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-404 failure on a bare (inferred) ref stays 'unknown'."""
    summary = (
        "7. Pending Tasks:\n"
        "   - masuda-masuo/sunaba#10 is the anchor.\n"
        "   - sunaba#100 is open; fix #888 after the merge.\n"
        "8. Current Work:\n"
    )
    db = make_db(
        tmp_path,
        [("ses-a", "2026-08-01", [user(summary, compact=True)])],
    )

    def fake(argv: list[str]) -> tuple[bool, Any, str]:
        if argv[1] == "repo":
            return True, [{"name": "sunaba"}], ""
        parts = argv[2].split("/")
        number = int(parts[-1])
        if number == 888:
            return False, None, "graphql: Not Found"
        return True, {"state": "closed"}, ""

    payload = _run_direct_json(db, use_gh=True, gh_call=fake, show_resolved=True)
    items = payload["sessions"][0]["items"]
    ref = items[1]["references"][1]
    assert ref["number"] == 888
    # #888 is bare, inferred to sunaba via the nearest preceding ref
    assert ref["via"] == "nearest"
    assert ref["bare"] is True
    # Non-404 failure → unknown (not unresolved)
    assert ref["state"] == "unknown"
    assert items[1]["status"] == "unknown"


# ---------------------------------------------------------------- via key


def test_via_key_on_bare_refs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Bare refs carry 'via' indicating how they were resolved."""
    summary = (
        "7. Pending Tasks:\n"
        "   - masuda-masuo/sunaba#10 is the anchor and also #20.\n"
        "   - masuda-masuo/shiori#1 is the other anchor.\n"
        "   - shiori has issues #30/#40.\n"
        "   - #50 unresolved.\n"
        "8. Current Work:\n"
    )
    db = make_db(
        tmp_path,
        [("ses-a", "2026-08-01", [user(summary, compact=True)])],
    )
    payload = _run_cli_json(db, capsys=capsys)
    items = payload["sessions"][0]["items"]

    # Item 0: explicit ref has no via; bare #20 resolves via nearest
    assert items[0]["references"][0].get("via") is None
    assert items[0]["references"][1]["bare"] is True
    assert items[0]["references"][1]["via"] == "nearest"

    # Item 1: explicit ref has no via
    assert items[1]["references"][0].get("via") is None

    # Item 2: #30/#40 resolved via name (shiori mention precedes them)
    assert items[2]["references"][0]["via"] == "name"
    assert items[2]["references"][1]["via"] == "name"

    # Item 3: #50 has no preceding repo in its own item → unresolved
    assert items[3]["references"][0]["state"] == "unresolved"
    assert items[3]["references"][0]["via"] is None


def test_via_repo_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Bare ref resolved via --repo carries 'via: repo_flag'."""
    summary = (
        "7. Pending Tasks:\n"
        "   - fix the #42 follow-up.\n"
        "8. Current Work:\n"
    )
    db = make_db(
        tmp_path,
        [("ses-a", "2026-08-01", [user(summary, compact=True)])],
    )
    payload = _run_cli_json(db, "--repo", "masuda-masuo/sunaba", capsys=capsys)
    ref = payload["sessions"][0]["items"][0]["references"][0]
    assert ref["via"] == "repo_flag"
    assert ref["bare"] is True


# ------------------------------------------------- issue #61: no summary-wide fallback


def test_bare_ref_without_in_item_repo_is_unresolved_and_not_checked(
    tmp_path: Path,
) -> None:
    """A bare #n with no repo in its own item stays unresolved even when the
    summary resolves other items, and the gh fake is never called for it."""
    summary = (
        "7. Pending Tasks:\n"
        "   - shiori#10 done.\n"
        "   - #17 untouched.\n"
        "8. Current Work:\n"
    )
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(summary, compact=True)])])
    calls: list[list[str]] = []

    def fake(argv: list[str]) -> tuple[bool, Any, str]:
        calls.append(list(argv))
        if argv[1] == "repo":
            return True, [{"name": "shiori"}, {"name": "sagasu"}], ""
        return True, {"state": "closed"}, ""

    payload = _run_direct_json(
        db, use_gh=True, gh_call=fake, owner="masuda-masuo", show_resolved=True
    )
    items = payload["sessions"][0]["items"]
    # Item 2 (#17) has no repo in its own item: unresolved, never checked.
    assert [(r["owner"], r["repo"], r["number"]) for r in items[1]["references"]] == [
        (None, None, 17)
    ]
    assert items[1]["references"][0]["state"] == "unresolved"
    assert items[1]["references"][0]["via"] is None
    api_calls = [argv for argv in calls if argv[1] == "api"]
    assert all("17" not in " ".join(argv) for argv in api_calls)


def test_repo_flag_applies_even_when_the_summary_resolves_something(
    tmp_path: Path,
) -> None:
    """--repo applies to every bare ref with no in-item repository, not only
    when the summary resolves nothing."""
    summary = (
        "7. Pending Tasks:\n"
        "   - shiori#10 done.\n"
        "   - #17 untouched.\n"
        "8. Current Work:\n"
    )
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(summary, compact=True)])])

    def fake(argv: list[str]) -> tuple[bool, Any, str]:
        if argv[1] == "repo":
            return True, [{"name": "shiori"}, {"name": "sagasu"}], ""
        return True, {"state": "closed"}, ""

    payload = _run_direct_json(
        db,
        use_gh=True,
        gh_call=fake,
        owner="masuda-masuo",
        repo=("masuda-masuo", "sagasu"),
        show_resolved=True,
    )
    items = payload["sessions"][0]["items"]
    assert [(r["owner"], r["repo"], r["number"]) for r in items[1]["references"]] == [
        ("masuda-masuo", "sagasu", 17)
    ]
    assert items[1]["references"][0]["bare"] is True
    assert items[1]["references"][0]["via"] == "repo_flag"
# ------------------------------------------------------- issue #63: same-number rule

#: The PR #391 shape from the live DB (issue #63): a bare ``PR #391`` before
#: the same-number explicit URL, both inside one item.
PR_391_SUMMARY = """\
7. Pending Tasks:
   - PR #391 (https://github.com/masuda-masuo/code-sandbox-mcp/pull/391) awaits merge.
8. Current Work:
"""


def test_bare_ref_takes_same_number_explicit_repo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bare #n takes the (owner, repo) of a same-number explicit ref in the
    same item, wherever the bare ref sits in the text."""
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(PR_391_SUMMARY, compact=True)])])
    payload = _run_cli_json(db, capsys=capsys)
    items = payload["sessions"][0]["items"]
    assert len(items) == 1
    refs = items[0]["references"]
    # Exactly one reference: the bare #391 came first in text order, so it
    # survives dedup and the URL collapses into it -- no unresolved leftover.
    # The same-number rule is not an inference: the ref is recorded non-bare,
    # exactly like the explicit ref it matches.
    assert [(r["owner"], r["repo"], r["number"]) for r in refs] == [
        ("masuda-masuo", "code-sandbox-mcp", 391)
    ]
    assert all(r["state"] != "unresolved" for r in refs)
    assert refs[0]["bare"] is False
    assert refs[0].get("via") is None


def test_bare_ref_takes_same_number_short_form(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bare #n takes the repo of a same-number known short form too."""
    summary = (
        "7. Pending Tasks:\n"
        "   - masuda-masuo/sagasu#1 is the anchor.\n"
        "   - merge PR #33 (sagasu#33).\n"
        "8. Current Work:\n"
    )
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(summary, compact=True)])])
    payload = _run_cli_json(db, capsys=capsys)
    refs = payload["sessions"][0]["items"][1]["references"]
    assert [(r["owner"], r["repo"], r["number"]) for r in refs] == [
        ("masuda-masuo", "sagasu", 33)
    ]
    # Same-number resolution is not an inference: the ref is non-bare, like
    # the explicit ref it matches.
    assert refs[0]["bare"] is False
    assert refs[0].get("via") is None


def test_explicit_first_bare_later_dedupes_to_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Explicit first, bare later: the same-number key collapses them into the
    one reference that came first in text order."""
    summary = (
        "7. Pending Tasks:\n"
        "   - masuda-masuo/sagasu#1 is the anchor.\n"
        "   - sagasu#33 merged; after #33, deploy.\n"
        "8. Current Work:\n"
    )
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(summary, compact=True)])])
    payload = _run_cli_json(db, capsys=capsys)
    refs = payload["sessions"][0]["items"][1]["references"]
    assert [(r["owner"], r["repo"], r["number"]) for r in refs] == [
        ("masuda-masuo", "sagasu", 33)
    ]
    assert refs[0]["bare"] is False
    assert refs[0].get("via") is None


def test_same_number_different_repos_falls_back_to_timeline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Same-number refs naming different repos never trigger rule 1: the bare
    #12 follows the normal timeline (nearest preceding = shiori) and is
    deduped into shiori#12."""
    summary = (
        "7. Pending Tasks:\n"
        "   - masuda-masuo/sunaba#1 and masuda-masuo/shiori#1 are the anchors.\n"
        "   - sunaba#12 and shiori#12, see #12.\n"
        "8. Current Work:\n"
    )
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(summary, compact=True)])])
    payload = _run_cli_json(db, capsys=capsys)
    refs = payload["sessions"][0]["items"][1]["references"]
    assert [(r["owner"], r["repo"], r["number"]) for r in refs] == [
        ("masuda-masuo", "sunaba", 12),
        ("masuda-masuo", "shiori", 12),
    ]
    assert all(r["bare"] is False for r in refs)


def test_same_number_rule_is_item_scoped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Numbers in other items of the summary never count: item 2's bare #33
    has no same-number ref in its own item and no --repo, so it is unresolved."""
    summary = (
        "7. Pending Tasks:\n"
        "   - masuda-masuo/sagasu#1 is the anchor.\n"
        "   - sagasu#33.\n"
        "   - #33 pending.\n"
        "8. Current Work:\n"
    )
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(summary, compact=True)])])
    payload = _run_cli_json(db, capsys=capsys)
    items = payload["sessions"][0]["items"]
    ref = items[2]["references"][0]
    assert (ref["owner"], ref["repo"], ref["number"]) == (None, None, 33)
    assert ref["state"] == "unresolved"
    assert ref["bare"] is True
    assert ref.get("via") is None


def test_same_number_beats_a_closer_mention(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Rule 1 wins over a nearer preceding name mention: the bare #33 is
    sagasu, not the shiori mention right before it."""
    summary = (
        "7. Pending Tasks:\n"
        "   - masuda-masuo/sagasu#1 and masuda-masuo/shiori#1 are the anchors.\n"
        "   - shiori open issues: #33 (sagasu#33).\n"
        "8. Current Work:\n"
    )
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(summary, compact=True)])])
    payload = _run_cli_json(db, capsys=capsys)
    refs = payload["sessions"][0]["items"][1]["references"]
    # The bare #33 came first in text order, so it survives dedup carrying
    # the sagasu resolution -- recorded non-bare, since same-number
    # resolution is not an inference.
    assert [(r["owner"], r["repo"], r["number"]) for r in refs] == [
        ("masuda-masuo", "sagasu", 33)
    ]
    assert refs[0]["bare"] is False
    assert refs[0].get("via") is None


def test_same_number_duplicate_is_looked_up_once(tmp_path: Path) -> None:
    """The collapsed 391 reference is one unique key, so the gh fake sees
    exactly one api call for it."""
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(PR_391_SUMMARY, compact=True)])])
    api_calls: list[list[str]] = []

    def fake(argv: list[str]) -> tuple[bool, Any, str]:
        if argv[1] == "repo":
            return True, [{"name": "code-sandbox-mcp"}], ""
        api_calls.append(list(argv))
        return True, {"state": "open", "pull_request": {"merged_at": None}}, ""

    payload = _run_direct_json(db, use_gh=True, gh_call=fake, show_resolved=True)
    refs = payload["sessions"][0]["items"][0]["references"]
    assert [(r["owner"], r["repo"], r["number"]) for r in refs] == [
        ("masuda-masuo", "code-sandbox-mcp", 391)
    ]
    assert refs[0]["state"] == "open"
    assert len(api_calls) == 1
    assert api_calls[0] == [
        "gh",
        "api",
        "repos/masuda-masuo/code-sandbox-mcp/issues/391",
    ]


# ------------------------------------------------- issue #63 repair: a same-number
# ref is not an inference -- the 404 must not depend on text order


def test_same_number_404_is_unknown_bare_first(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Bare #391 first, explicit URL second, gh 404s: the same-number rule is
    not an inference, so the 404 is 'unknown' -- never the 'inferred repo'
    demotion, however the text is ordered."""
    summary = (
        "7. Pending Tasks:\n"
        "   - PR #391 (https://github.com/masuda-masuo/code-sandbox-mcp/pull/391) awaits merge.\n"
        "8. Current Work:\n"
    )
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(summary, compact=True)])])

    def fake(argv: list[str]) -> tuple[bool, Any, str]:
        if argv[1] == "repo":
            return True, [{"name": "code-sandbox-mcp"}], ""
        return False, None, "gh: Not Found (HTTP 404)"

    payload = _run_direct_json(db, use_gh=True, gh_call=fake)
    refs = payload["sessions"][0]["items"][0]["references"]
    assert [(r["owner"], r["repo"], r["number"]) for r in refs] == [
        ("masuda-masuo", "code-sandbox-mcp", 391)
    ]
    assert refs[0]["state"] == "unknown"
    assert refs[0]["bare"] is False
    assert refs[0].get("via") is None
    assert "HTTP 404" in refs[0]["error"]
    assert "inferred repo" not in refs[0]["error"]


def test_same_number_404_is_unknown_explicit_first(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Explicit URL first, bare #391 second, gh 404s: exactly the same output
    as the reversed order -- one reference, 'unknown', no 'inferred repo'."""
    summary = (
        "7. Pending Tasks:\n"
        "   - https://github.com/masuda-masuo/code-sandbox-mcp/pull/391 awaits merge; also PR #391.\n"
        "8. Current Work:\n"
    )
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [user(summary, compact=True)])])

    def fake(argv: list[str]) -> tuple[bool, Any, str]:
        if argv[1] == "repo":
            return True, [{"name": "code-sandbox-mcp"}], ""
        return False, None, "gh: Not Found (HTTP 404)"

    payload = _run_direct_json(db, use_gh=True, gh_call=fake)
    refs = payload["sessions"][0]["items"][0]["references"]
    assert [(r["owner"], r["repo"], r["number"]) for r in refs] == [
        ("masuda-masuo", "code-sandbox-mcp", 391)
    ]
    assert refs[0]["state"] == "unknown"
    assert refs[0]["bare"] is False
    assert refs[0].get("via") is None
    assert "HTTP 404" in refs[0]["error"]
    assert "inferred repo" not in refs[0]["error"]
