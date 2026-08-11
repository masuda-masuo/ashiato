"""Parser behaviour, asserted on values rather than on the absence of exceptions."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from ashiato.parser import (
    DENIAL_PATTERNS,
    INPUT_SUMMARY_LIMIT,
    ParsedFile,
    ToolCall,
    compute_depths,
    flatten_result_text,
    parse_file,
    parse_timestamp,
    split_tool_name,
    summarize_input,
)

FIXTURES = Path(__file__).parent / "fixtures"
MAIN = FIXTURES / "session_main.jsonl"
SNAKE = FIXTURES / "session_snake.jsonl"
CHAIN = FIXTURES / "chain.jsonl"
EMPTY = FIXTURES / "empty.jsonl"

MAIN_SESSION_ID = "11111111-1111-4111-8111-111111111111"
SNAKE_SESSION_ID = "22222222-2222-4222-8222-222222222222"


@pytest.fixture(scope="module")
def main() -> ParsedFile:
    return parse_file(MAIN)


def by_id(parsed: ParsedFile) -> dict[str, object]:
    return {call.tool_use_id: call for call in parsed.tool_calls}


def events_by_seq(parsed: ParsedFile) -> dict[int, object]:
    return {event.seq: event for event in parsed.events}


# ---------------------------------------------------------------- helpers


def test_parse_timestamp_handles_z_and_offsets_and_junk():
    assert parse_timestamp("2026-08-10T09:00:00.000Z") == datetime(2026, 8, 10, 9, 0, 0)
    # +02:00 is normalised back to UTC.
    assert parse_timestamp("2026-08-10T11:00:00+02:00") == datetime(2026, 8, 10, 9, 0, 0)
    # More fractional digits than fromisoformat accepts.
    assert parse_timestamp("2026-08-10T09:00:00.123456789Z") == datetime(
        2026, 8, 10, 9, 0, 0, 123456
    )
    assert parse_timestamp("not a timestamp") is None
    assert parse_timestamp(None) is None
    assert parse_timestamp(17) is None


def test_split_tool_name():
    assert split_tool_name("Bash") == ("builtin", None)
    assert split_tool_name("mcp__sunaba__publish") == ("mcp", "sunaba")
    assert split_tool_name("mcp__shiori__search") == ("mcp", "shiori")
    assert split_tool_name(None) == ("builtin", None)
    # Malformed MCP name: still MCP, but there is no server segment to take.
    assert split_tool_name("mcp__weird") == ("mcp", None)


def test_flatten_result_text_shapes():
    assert flatten_result_text("plain") == "plain"
    assert flatten_result_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == (
        "a\nb"
    )
    # A non-text block is noted, not inlined -- images would be megabytes of base64.
    assert flatten_result_text([{"type": "image", "source": {"data": "AAAA"}}]) == "[image]"
    assert flatten_result_text(None) == ""


# ---------------------------------------------------------------- events


def test_malformed_and_truncated_lines_are_counted_not_fatal(main: ParsedFile):
    # One broken line mid-file, one truncated final line.
    assert main.n_parse_errors == 2
    assert len(main.events) == 17


def test_event_fields_are_lifted(main: ParsedFile):
    events = events_by_seq(main)
    assistant = events[2]
    assert assistant.event_id == "u2"
    assert assistant.session_id == MAIN_SESSION_ID
    assert assistant.type == "assistant"
    assert assistant.role == "assistant"
    assert assistant.parent_uuid == "u1"
    assert assistant.ts == datetime(2026, 8, 10, 9, 0, 2)
    assert assistant.model == "claude-opus-5"
    assert assistant.request_id == "req_a"
    assert assistant.message_id == "msg_a"
    assert assistant.effort == "high"
    assert assistant.permission_mode == "default"
    assert assistant.cwd == "/home/dev/proj"
    assert assistant.git_branch == "main"
    assert assistant.text == "I'll list the files."
    assert assistant.is_sidechain is False
    assert assistant.is_meta is False


def test_text_is_empty_string_when_there_are_no_text_blocks(main: ParsedFile):
    # A tool_result-only user turn has no text blocks.
    assert events_by_seq(main)[3].text == ""


def test_raw_is_the_verbatim_line(main: ParsedFile):
    for event in main.events:
        assert json.loads(event.raw)  # round-trips
    with open(MAIN, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    for event in main.events:
        assert event.raw == lines[event.seq - 1].strip()


def test_records_without_uuid_get_a_synthesized_event_id(main: ParsedFile):
    snapshot = events_by_seq(main)[12]
    assert snapshot.type == "file-history-snapshot"
    assert snapshot.event_id == f"{MAIN.resolve()}:12"
    assert snapshot.ts is None
    assert snapshot.depth == 0


def test_both_session_id_spellings_are_accepted(main: ParsedFile):
    # sessionId on most lines...
    assert events_by_seq(main)[2].session_id == MAIN_SESSION_ID
    # ...and session_id on the system line.
    system = events_by_seq(main)[13]
    assert system.type == "system"
    assert '"session_id"' in system.raw
    assert system.session_id == MAIN_SESSION_ID

    snake = parse_file(SNAKE)
    assert snake.session.session_id == SNAKE_SESSION_ID
    assert {event.session_id for event in snake.events} == {SNAKE_SESSION_ID}


def test_is_meta_and_sidechain_flags(main: ParsedFile):
    events = events_by_seq(main)
    assert events[14].is_meta is True
    assert events[15].is_sidechain is True
    assert events[2].is_sidechain is False


# ---------------------------------------------------------------- depth


def test_depth_follows_the_parent_chain(main: ParsedFile):
    events = events_by_seq(main)
    assert events[1].depth == 0  # parentUuid null
    assert events[2].depth == 1
    assert events[11].depth == 10
    assert events[14].depth == 12
    # The sidechain starts its own root.
    assert events[15].depth == 0
    assert events[16].depth == 1
    assert events[17].depth == 2


def naive_depth(parent_of: dict[str, str | None], node: str) -> int:
    """Walk to the root for a single node -- the O(n * chain) form."""
    depth = 0
    seen = set()
    current = parent_of.get(node)
    while current is not None and current in parent_of and current not in seen:
        seen.add(current)
        depth += 1
        current = parent_of[current]
    return depth


def test_memoized_depth_agrees_with_the_naive_walk():
    parsed = parse_file(CHAIN)
    assert len(parsed.events) == 50
    parent_of = {event.event_id: event.parent_uuid for event in parsed.events}
    for event in parsed.events:
        assert event.depth == naive_depth(parent_of, event.event_id)
    assert max(event.depth for event in parsed.events) == 49


def test_depth_is_iterative_and_survives_chains_past_the_recursion_limit(tmp_path: Path):
    # Real corpora reach ~2400; a recursive implementation dies here.
    length = 2500
    path = tmp_path / "deep.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for index in range(length):
            handle.write(
                json.dumps(
                    {
                        "type": "user",
                        "uuid": f"d{index}",
                        "parentUuid": None if index == 0 else f"d{index - 1}",
                        "sessionId": "deep",
                        "timestamp": "2026-08-10T12:00:00.000Z",
                    }
                )
                + "\n"
            )
    parsed = parse_file(path)
    assert max(event.depth for event in parsed.events) == length - 1


def test_compute_depths_handles_cycles_and_dangling_parents():
    # A dangling parent (points outside the file) counts as a root.
    assert compute_depths({"a": "missing", "b": "a"}) == {"a": 0, "b": 1}
    # A cycle terminates instead of hanging.
    depths = compute_depths({"x": "y", "y": "x"})
    assert set(depths) == {"x", "y"}


# ---------------------------------------------------------------- tool calls


def test_tool_calls_are_joined_across_events(main: ParsedFile):
    call = by_id(main)["toolu_ok_1"]
    assert call.call_event_id == "u2"
    assert call.result_event_id == "u3"
    assert call.tool_name == "Bash"
    assert call.tool_kind == "builtin"
    assert call.mcp_server is None
    assert json.loads(call.input) == {"command": "ls -1", "description": "list files"}
    assert call.outcome == "ok"
    assert call.is_error is False
    assert call.result_text == "README.md\nsrc"
    assert call.result_truncated is False
    assert call.duration_ms == 1500
    assert call.seq == 2
    assert call.ts == datetime(2026, 8, 10, 9, 0, 2)
    assert call.session_id == MAIN_SESSION_ID
    assert call.permission_mode == "default"
    assert call.cwd == "/home/dev/proj"
    assert call.is_sidechain is False
    assert call.parent_tool_use_id is None


def test_every_outcome_rule_fires(main: ParsedFile):
    calls = by_id(main)
    assert {tool_use_id: call.outcome for tool_use_id, call in calls.items()} == {
        "toolu_ok_1": "ok",
        "toolu_err_1": "error",
        "toolu_denied_1": "denied",
        "toolu_denied_2": "denied",
        "toolu_pending_1": "pending",
        "toolu_sub_1": "ok",
    }


def test_error_outcome(main: ParsedFile):
    call = by_id(main)["toolu_err_1"]
    assert call.outcome == "error"
    assert call.is_error is True
    assert call.result_text == "File does not exist: /missing.txt"


def test_denial_beats_error_and_beats_ok(main: ParsedFile):
    calls = by_id(main)
    # First denial string, on a result that is also flagged is_error.
    first = calls["toolu_denied_1"]
    assert first.outcome == "denied"
    assert first.is_error is True
    assert DENIAL_PATTERNS[0] in first.result_text
    # Second denial string, on a result with no is_error flag at all: without
    # the denial rule this would read as a success.
    second = calls["toolu_denied_2"]
    assert second.outcome == "denied"
    assert second.is_error is False
    assert DENIAL_PATTERNS[1] in second.result_text


def test_pending_when_the_session_ended_mid_call(main: ParsedFile):
    call = by_id(main)["toolu_pending_1"]
    assert call.outcome == "pending"
    assert call.result_event_id is None
    assert call.result_text is None
    assert call.duration_ms is None
    assert call.is_error is False


def test_mcp_tool_names_are_split(main: ParsedFile):
    call = by_id(main)["toolu_denied_2"]
    assert call.tool_name == "mcp__sunaba__publish"
    assert call.tool_kind == "mcp"
    assert call.mcp_server == "sunaba"

    snake_call = parse_file(SNAKE).tool_calls[0]
    assert snake_call.tool_kind == "mcp"
    assert snake_call.mcp_server == "shiori"
    assert snake_call.outcome == "ok"
    assert snake_call.duration_ms == 1500


def test_subagent_attribution_and_sidechain_carry(main: ParsedFile):
    call = by_id(main)["toolu_sub_1"]
    assert call.parent_tool_use_id == "u2"
    assert call.is_sidechain is True
    assert call.permission_mode == "plan"


def test_result_text_limit_is_configurable(main: ParsedFile):
    assert by_id(main)["toolu_ok_1"].result_truncated is False
    truncated = by_id(parse_file(MAIN, result_text_limit=5))["toolu_ok_1"]
    assert truncated.result_text == "READM"
    assert truncated.result_truncated is True


def test_denial_patterns_are_overridable():
    parsed = parse_file(MAIN, denial_patterns=("File does not exist",))
    calls = by_id(parsed)
    # The caller's pattern now denies what was an error...
    assert calls["toolu_err_1"].outcome == "denied"
    # ...and the built-in strings no longer apply, so is_error decides.
    assert calls["toolu_denied_1"].outcome == "error"
    assert calls["toolu_denied_2"].outcome == "ok"


# ---------------------------------------------------------------- denial anchoring


def call_with_result(content: object, tmp_path: Path, **kwargs) -> ToolCall:
    """One Bash call whose tool_result carries *content* -- and nothing else."""
    path = tmp_path / "one_call.jsonl"
    records = [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": "s1",
            "timestamp": "2026-08-10T09:00:00.000Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "run it"}]},
        },
        {
            "type": "assistant",
            "uuid": "u2",
            "parentUuid": "u1",
            "sessionId": "s1",
            "timestamp": "2026-08-10T09:00:01.000Z",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Bash",
                        "input": {"command": "grep denied"},
                    }
                ],
            },
        },
        {
            "type": "user",
            "uuid": "u3",
            "parentUuid": "u2",
            "sessionId": "s1",
            "timestamp": "2026-08-10T09:00:02.000Z",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": content}],
            },
        },
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    (call,) = parse_file(path, **kwargs).tool_calls
    return call


@pytest.mark.parametrize("pattern", DENIAL_PATTERNS)
def test_a_result_that_only_contains_a_denial_pattern_is_not_denied(pattern, tmp_path: Path):
    """Quoting the strings is not a denial: only a result that *starts* with one is.

    The measured false positives were a Bash chain whose output quoted the
    patterns and MCP results embedding them in a JSON payload.
    """
    # A command chain that ended in a grep for the strings: the denial text
    # appears mid-output, after the command's own echo.
    output = f"usage: ashiato build ...\n$ grep -n denied src/ashiato/*.py\n{pattern}\n1 match"
    assert call_with_result(output, tmp_path).outcome == "ok"
    # An MCP result that returns the strings as data, not as its own verdict.
    payload = '{"result": "{\\"content\\": [{\\"type\\": \\"text\\", \\"text\\": \\"' + pattern + '\\"}]}"}'
    assert call_with_result(payload, tmp_path).outcome == "ok"


def test_a_pattern_in_a_later_block_is_not_denied(tmp_path: Path):
    """A denial in the second text block starts after the first block's text."""
    content = [
        {"type": "text", "text": "here is the diff"},
        {"type": "text", "text": DENIAL_PATTERNS[0]},
    ]
    assert call_with_result(content, tmp_path).outcome == "ok"


@pytest.mark.parametrize("pattern", DENIAL_PATTERNS)
def test_a_result_that_starts_with_a_denial_pattern_is_denied(pattern, tmp_path: Path):
    call = call_with_result(f"{pattern}. The tool call was rejected.", tmp_path)
    assert call.outcome == "denied"


@pytest.mark.parametrize("pattern", DENIAL_PATTERNS)
def test_leading_whitespace_does_not_hide_a_denial(pattern, tmp_path: Path):
    call = call_with_result(f" \n\t {pattern} rejected", tmp_path)
    assert call.outcome == "denied"


def test_a_denial_is_decided_on_the_full_text_not_the_truncated_copy(tmp_path: Path):
    """The stored ``result_text`` is cut, but the verdict is made on the whole text."""
    call = call_with_result(
        DENIAL_PATTERNS[0] + " x", tmp_path, result_text_limit=10
    )
    assert call.outcome == "denied"
    assert call.result_text == DENIAL_PATTERNS[0][:10]
    assert call.result_truncated is True


# ---------------------------------------------------------------- session


def test_session_row(main: ParsedFile):
    session = main.session
    assert session.session_id == MAIN_SESSION_ID
    assert session.file_path == str(MAIN.resolve())
    assert session.project_dir == "fixtures"
    assert session.cwd == "/home/dev/proj"
    # Last non-null wins: the values change partway through the file, and the
    # final line omits them entirely.
    assert session.git_branch == "feature/parser"
    assert session.cc_version == "2.0.31"
    assert session.entrypoint == "cli"
    assert session.started_at == datetime(2026, 8, 10, 9, 0, 0)
    assert session.ended_at == datetime(2026, 8, 10, 9, 0, 16)
    assert session.n_events == 17
    assert session.n_tool_calls == 6


def test_tokens_are_deduplicated_by_request_id(main: ParsedFile):
    session = main.session
    # req_a appears on three lines carrying the same usage object; req_b, req_c
    # and req_d once each; one assistant line has usage but no requestId.
    assert session.input_tokens == 1223
    assert session.output_tokens == 63
    assert session.cache_read_tokens == 5018
    assert session.cache_creation_tokens == 318


def test_naive_summing_would_be_visibly_larger():
    """The dedup test above must fail if dedup is removed -- this pins the gap."""
    naive = 0
    with open(MAIN, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            usage = record.get("message", {}).get("usage")
            if usage:
                naive += usage.get("input_tokens", 0)
    assert naive == 3623
    assert naive > parse_file(MAIN).session.input_tokens * 2


def test_file_with_no_valid_lines_yields_no_session():
    parsed = parse_file(EMPTY)
    assert parsed.session is None
    assert parsed.events == []
    assert parsed.tool_calls == []
    assert parsed.n_parse_errors == 0


def test_missing_optional_fields_do_not_crash(tmp_path: Path):
    path = tmp_path / "sparse.jsonl"
    path.write_text(
        "\n".join(
            [
                "{}",
                '{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use"}]}}',
                '{"type":"user","message":{"content":[{"type":"tool_result"}]}}',
                '{"type":"assistant","message":"not an object"}',
                '{"type":"user","uuid":123,"timestamp":false,"isSidechain":"yes"}',
                "[1, 2, 3]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    parsed = parse_file(path)
    assert parsed.n_parse_errors == 1  # the JSON array is not a record
    assert len(parsed.events) == 5
    assert parsed.session.session_id == "sparse"  # falls back to the file stem
    assert parsed.session.started_at is None
    # A tool_use with no id still produces a row, keyed on where it was found.
    (call,) = parsed.tool_calls
    assert call.tool_name is None
    assert call.tool_kind == "builtin"
    assert call.input is None
    assert call.outcome == "pending"
    # A non-string uuid is not usable as an id; the synthesized one is.
    assert parsed.events[4].event_id == f"{path.resolve()}:5"
    assert parsed.events[4].is_sidechain is True


def test_a_utf8_bom_does_not_swallow_the_first_record(tmp_path: Path):
    """A BOM is not valid JSON, so plain utf-8 would drop record one silently.

    Written with encoding="utf-8-sig" so the BOM lands in the file exactly as a
    Windows-side writer would emit it.  Without utf-8-sig on the read side this
    test fails with n_parse_errors == 1 and a missing first event.
    """
    path = tmp_path / "bom.jsonl"
    records = [
        {
            "type": "user",
            "uuid": "bom-1",
            "sessionId": MAIN_SESSION_ID,
            "timestamp": "2026-08-10T09:00:00.000Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "first"}]},
        },
        {
            "type": "assistant",
            "uuid": "bom-2",
            "parentUuid": "bom-1",
            "sessionId": MAIN_SESSION_ID,
            "timestamp": "2026-08-10T09:00:01.000Z",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "second"}]},
        },
    ]
    with open(path, "w", encoding="utf-8-sig") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    assert path.read_bytes().startswith(b"\xef\xbb\xbf")

    parsed = parse_file(path)
    assert parsed.n_parse_errors == 0
    assert [event.event_id for event in parsed.events] == ["bom-1", "bom-2"]
    assert parsed.events[0].text == "first"
    assert parsed.session is not None
    assert parsed.session.n_events == 2


# ---------------------------------------------------------------- input summary


@pytest.mark.parametrize(
    ("tool_name", "tool_input", "expected"),
    [
        ("Bash", {"command": "ls -1", "description": "list files"}, "ls -1"),
        ("PowerShell", {"command": "Get-ChildItem"}, "Get-ChildItem"),
        ("Read", {"file_path": "/tmp/a.txt", "limit": 5}, "/tmp/a.txt"),
        # The content of a Write is the whole point of not showing the input.
        ("Write", {"file_path": "/etc/hosts", "content": "127.0.0.1 nope"}, "/etc/hosts"),
        ("Edit", {"file_path": "/src/x.py", "old_str": "a", "new_str": "b"}, "/src/x.py"),
        ("NotebookEdit", {"file_path": "/nb.ipynb", "cell_id": "c1"}, "/nb.ipynb"),
        ("Glob", {"pattern": "**/*.py"}, "**/*.py"),
        ("Grep", {"pattern": "TODO", "glob": "*.py"}, "TODO"),
        ("Skill", {"skill": "code-review", "args": "--fix"}, "code-review"),
        ("Agent", {"description": "find the bug", "prompt": "a long prompt"}, "find the bug"),
        ("Task", {"description": "find the bug"}, "find the bug"),
        ("WebFetch", {"url": "https://example.com", "prompt": "read it"}, "https://example.com"),
        ("WebSearch", {"query": "duckdb window functions"}, "duckdb window functions"),
    ],
)
def test_summarize_input_reads_the_field_that_matters_per_tool(tool_name, tool_input, expected):
    assert summarize_input(tool_name, tool_input) == expected


def test_summarize_input_falls_back_to_the_whole_input():
    # MCP tools have no field this table knows about, so the input shows whole.
    assert summarize_input("mcp__sunaba__publish", {"files": ["a.py"], "create_pr": True}) == (
        '{"create_pr":true,"files":["a.py"]}'
    )
    # A tool nobody has taught this table about, and a call with no name at all.
    assert summarize_input("BrandNewTool", {"thing": "value"}) == '{"thing":"value"}'
    assert summarize_input(None, {"thing": "value"}) == '{"thing":"value"}'


def test_the_fallback_does_not_depend_on_key_order():
    """Two transcripts of the same call must summarise identically."""
    assert summarize_input("mcp__x__y", {"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert summarize_input("mcp__x__y", {"a": 2, "b": 1}) == '{"a":2,"b":1}'


def test_summarize_input_falls_back_when_the_named_field_is_unusable():
    # Present but not the field we hoped for...
    assert summarize_input("Bash", {"description": "no command here"}) == (
        '{"description":"no command here"}'
    )
    # ...present but not a string...
    assert summarize_input("Read", {"file_path": 17}) == '{"file_path":17}'
    # ...present but blank, which would summarise to nothing at all.  The
    # fallback is collapsed to one line like any other summary, which is why
    # the three spaces come back as one.
    assert summarize_input("Bash", {"command": "   "}) == '{"command":" "}'


def test_summarize_input_is_null_only_when_there_was_no_input():
    assert summarize_input("Bash", None) is None
    # An input that is present but empty is not the same as no input.
    assert summarize_input("Bash", {}) == "{}"


def test_summarize_input_survives_an_input_that_is_not_an_object():
    assert summarize_input("Bash", ["a", "b"]) == '["a","b"]'
    assert summarize_input("Bash", "just a string") == '"just a string"'


def test_a_summary_is_one_line():
    assert summarize_input("Bash", {"command": "set -e\n\n  git status\t-s"}) == (
        "set -e git status -s"
    )


def test_a_summary_is_cut_to_the_limit():
    command = "echo " + "x" * 500
    summary = summarize_input("Bash", {"command": command})
    assert len(summary) == INPUT_SUMMARY_LIMIT == 200
    assert summary == command[:INPUT_SUMMARY_LIMIT]

    # The fallback is cut too -- an MCP call can carry a whole file as an argument.
    assert len(summarize_input("mcp__server__tool", {"blob": "y" * 500})) == INPUT_SUMMARY_LIMIT


def test_input_summary_lands_on_the_parsed_calls(main: ParsedFile):
    calls = by_id(main)
    assert calls["toolu_ok_1"].input_summary == "ls -1"
    assert calls["toolu_pending_1"].input_summary == "sleep 600"
    assert calls["toolu_err_1"].input_summary == "/missing.txt"
    assert calls["toolu_denied_1"].input_summary == "/etc/hosts"
    assert calls["toolu_sub_1"].input_summary == "TODO"
    assert calls["toolu_denied_2"].input_summary == (
        '{"create_pr":true,"files":["src/ashiato/parser.py"]}'
    )


def test_a_tool_use_with_no_input_summarises_to_nothing(tmp_path: Path):
    path = tmp_path / "no_input.jsonl"
    path.write_text(
        '{"type":"assistant","message":{"role":"assistant",'
        '"content":[{"type":"tool_use","id":"t1","name":"Bash"}]}}\n',
        encoding="utf-8",
    )
    (call,) = parse_file(path).tool_calls
    assert call.input is None
    assert call.input_summary is None
