"""The opencode events.ndjson reader: completed vs pending, malformed lines."""

from __future__ import annotations

from pathlib import Path

from ashiato.opencode import (
    OpenCodeTextChunk,
    OpenCodeToolCall,
    parse_epoch_millis,
    parse_file,
)

FIXTURES = Path(__file__).parent / "fixtures"
EVENTS = FIXTURES / "opencode_events.ndjson"


def by_call_id(calls: list[OpenCodeToolCall]) -> dict[str, OpenCodeToolCall]:
    return {call.call_id: call for call in calls}


def test_parse_epoch_millis():
    from datetime import datetime

    assert parse_epoch_millis(1000) == datetime(1970, 1, 1, 0, 0, 1)
    assert parse_epoch_millis(1787175490507) is not None
    assert parse_epoch_millis(None) is None
    assert parse_epoch_millis("1000") is None
    assert parse_epoch_millis(True) is None


def test_only_completed_tool_parts_produce_a_call():
    """The pending states in the fixture (ses_ccc, and prt_2's own pending line) yield nothing."""
    parsed = parse_file(EVENTS)
    calls = by_call_id(parsed.tool_calls)
    assert "call_2" in calls  # the completed state of the same part id
    assert "call_7" not in calls  # ses_ccc never completes
    assert len(parsed.tool_calls) == 3


def test_a_completed_tool_part_captures_the_full_shape():
    parsed = parse_file(EVENTS)
    call = by_call_id(parsed.tool_calls)["call_2"]
    assert call.session_id == "ses_aaa"
    assert call.tool == "kaiba_recall"
    assert call.input == {"query": "denial pattern anchoring"}
    assert call.output == "Use anchored prefix matching for denial_pattern_x9 tokens."
    assert call.seq == 3
    from datetime import datetime

    assert call.ts == datetime(1970, 1, 1, 0, 0, 1)


def test_a_non_recall_tool_part_is_still_a_plain_tool_call():
    """The reader does not filter by tool name -- that is ``ashiato.recall``'s job."""
    parsed = parse_file(EVENTS)
    calls = by_call_id(parsed.tool_calls)
    assert "call_8" in calls
    assert calls["call_8"].tool == "bash"
    assert calls["call_8"].output == "file1\nfile2"


def test_a_tool_part_missing_state_entirely_produces_no_call_not_an_exception():
    parsed = parse_file(EVENTS)
    assert "call_13" not in by_call_id(parsed.tool_calls)


def test_text_parts_and_deltas_both_produce_chunks():
    parsed = parse_file(EVENTS)
    texts = [chunk.text for chunk in parsed.text_chunks]
    assert "Let me check kaiba for prior guidance before editing anything." in texts
    assert "Applying denial_pattern_x9 as documented in the recall output." in texts
    assert "Proceeding with the standard approach and ignoring that suggestion." in texts
    assert len(parsed.text_chunks) == 3


def test_text_chunks_carry_their_session_and_line_number():
    parsed = parse_file(EVENTS)
    chunk = next(c for c in parsed.text_chunks if "orphaned" not in c.text and "kaiba" not in c.text)
    assert isinstance(chunk, OpenCodeTextChunk)
    assert chunk.session_id == "ses_aaa"
    assert chunk.seq == 4


def test_malformed_lines_are_skipped_and_counted():
    """Invalid JSON and a JSON value that is not an object: two lines, two errors."""
    parsed = parse_file(EVENTS)
    assert parsed.n_parse_errors == 2


def test_unknown_event_types_and_part_types_are_skipped_not_fatal():
    """step.started and a part.type of "file" never raise and never contribute anything."""
    parsed = parse_file(EVENTS)
    # Nothing from ses_eee's non-tool, non-text records ends up anywhere.
    assert not any(call.session_id == "ses_eee" for call in parsed.tool_calls)
    assert not any(chunk.session_id == "ses_eee" for chunk in parsed.text_chunks)


def test_an_empty_file_yields_nothing(tmp_path: Path):
    empty = tmp_path / "empty.ndjson"
    empty.write_text("", encoding="utf-8")
    parsed = parse_file(empty)
    assert parsed.tool_calls == []
    assert parsed.text_chunks == []
    assert parsed.n_parse_errors == 0


def test_repeated_completed_state_is_deduped_keeping_the_last(tmp_path: Path):
    """If the same (sessionID, callID) completes twice, keep only the last.

    Robustness only: real streams never duplicate callIDs, so this must not
    change output on a stream without duplicates -- but a duplicate must yield
    one call carrying the second output.
    """
    import json

    record = {
        "id": "evt_dup",
        "type": "message.part.updated",
        "properties": {
            "sessionID": "ses_x",
            "part": {
                "id": "prt_dup",
                "sessionID": "ses_x",
                "type": "tool",
                "tool": "bash",
                "callID": "call_dup",
                "state": {
                    "status": "completed",
                    "input": {"command": "echo"},
                    "output": "FIRST",
                    "time": {"start": 1, "end": 2},
                },
            },
        },
    }
    first_line = json.dumps(record)
    second = json.loads(json.dumps(record))
    second["properties"]["part"]["state"]["output"] = "SECOND"
    second_line = json.dumps(second)

    path = tmp_path / "dup.ndjson"
    path.write_text(first_line + "\n" + second_line + "\n", encoding="utf-8")

    parsed = parse_file(path)
    calls = by_call_id(parsed.tool_calls)
    assert len(calls) == 1
    assert calls["call_dup"].output == "SECOND"


def test_call_id_falls_back_to_part_id_then_to_a_synthesized_one(tmp_path: Path):
    import json

    path = tmp_path / "fallback.ndjson"
    record = {
        "id": "evt_x",
        "type": "message.part.updated",
        "properties": {
            "sessionID": "ses_f",
            "part": {
                "id": "prt_only",
                "sessionID": "ses_f",
                "type": "tool",
                "tool": "kaiba_recall",
                "state": {"status": "completed", "input": {"query": "q"}, "output": "o"},
            },
        },
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    parsed = parse_file(path)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].call_id == "prt_only"
