"""The Cursor agent-transcript reader and its recall extraction (issue #20).

``cursor_transcript.ndjson`` deliberately does not end in ``.jsonl`` even
though real Cursor transcripts do: ``tests/fixtures/`` is globbed recursively
by other tests' plain ``--source`` (``*.jsonl``) builds, and a same-extension
addition there would silently change their pinned row counts -- the same
reasoning ``test_salvage.py`` documents for keeping its own fixtures out of
this directory entirely.  The build-level, directory-glob-discovery test for
``--cursor-source`` (which does need a realistic ``*.jsonl`` name) lives in
``tests/test_build.py`` and writes its own copy under ``tmp_path`` instead.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from ashiato.cursor import CursorTextChunk, CursorToolCall, parse_file
from ashiato.recall import extract_from_cursor

FIXTURES = Path(__file__).parent / "fixtures"
TRANSCRIPT = FIXTURES / "cursor_transcript.ndjson"


def by_call_id(calls: list[CursorToolCall]) -> dict[str, CursorToolCall]:
    return {call.call_id: call for call in calls}


# ---------------------------------------------------------------- parse_file


def test_session_id_is_the_file_name_stem():
    parsed = parse_file(TRANSCRIPT)
    assert parsed.session_id == "cursor_transcript"


def test_a_user_line_produces_no_activity():
    """The <user_query> text block is not modelled at all -- user lines are not activity."""
    parsed = parse_file(TRANSCRIPT)
    assert not any("user_query" in chunk.text for chunk in parsed.text_chunks)
    assert parsed.text_chunks[0].seq > 1


def test_status_and_turn_ended_lines_produce_nothing():
    """Neither the bare status line nor turn_ended contributes a call or a chunk."""
    parsed = parse_file(TRANSCRIPT)
    assert all(call.seq != 3 for call in parsed.tool_calls)
    assert all(chunk.seq != 3 for chunk in parsed.text_chunks)
    assert all(call.seq != 5 for call in parsed.tool_calls)


def test_malformed_line_is_skipped_and_counted():
    parsed = parse_file(TRANSCRIPT)
    assert parsed.n_parse_errors == 1


def test_every_tool_use_block_on_a_line_is_captured_with_its_own_call_id():
    """The recall and the other MCP call share seq=2 but have distinct block_index."""
    parsed = parse_file(TRANSCRIPT)
    calls = by_call_id(parsed.tool_calls)
    assert set(calls) == {"2:1", "2:2", "4:1"}
    assert calls["2:1"].seq == 2
    assert calls["2:2"].seq == 2
    assert calls["4:1"].seq == 4


def test_the_kaiba_recall_call_carries_its_full_shape():
    parsed = parse_file(TRANSCRIPT)
    call = by_call_id(parsed.tool_calls)["2:1"]
    assert call.name == "CallMcpTool"
    assert call.input == {
        "server": "kaiba",
        "toolName": "recall",
        "arguments": {"query": "denial_pattern_x9", "top_k": 10},
        "description": "kaiba recall",
    }
    assert call.session_id == "cursor_transcript"
    assert call.file_path == str(TRANSCRIPT.resolve())


def test_another_mcp_server_call_is_still_just_a_plain_tool_call():
    """The reader does not filter by server/toolName -- that is ashiato.recall's job."""
    parsed = parse_file(TRANSCRIPT)
    call = by_call_id(parsed.tool_calls)["2:2"]
    assert call.name == "CallMcpTool"
    assert call.input["server"] == "other"


def test_a_non_mcp_tool_is_captured_by_its_own_name():
    parsed = parse_file(TRANSCRIPT)
    call = by_call_id(parsed.tool_calls)["4:1"]
    assert call.name == "Read"
    assert call.input == {"path": "/home/user/project/notes.md"}


def test_assistant_text_blocks_become_chunks_with_seq_and_block_index():
    parsed = parse_file(TRANSCRIPT)
    texts = {(chunk.seq, chunk.block_index): chunk.text for chunk in parsed.text_chunks}
    assert texts[(2, 0)] == "Let me check kaiba first."
    assert texts[(4, 0)] == "Applying denial_pattern_x9 as documented."
    assert len(parsed.text_chunks) == 2
    assert all(isinstance(chunk, CursorTextChunk) for chunk in parsed.text_chunks)


def test_an_empty_file_yields_nothing(tmp_path: Path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    parsed = parse_file(empty)
    assert parsed.tool_calls == []
    assert parsed.text_chunks == []
    assert parsed.n_parse_errors == 0


def test_non_object_json_is_also_a_parse_error(tmp_path: Path):
    path = tmp_path / "array.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    parsed = parse_file(path)
    assert parsed.n_parse_errors == 1


# ---------------------------------------------------------------- extract_from_cursor


def test_recall_id_and_call_id_are_built_from_seq_and_block_index():
    parsed = parse_file(TRANSCRIPT)
    rows = extract_from_cursor(parsed)
    assert len(rows) == 1
    row = rows[0]
    assert row.call_id == "2:1"
    assert row.recall_id == f"{parsed.file_path}:2:1"
    assert row.source == "cursor"
    assert row.query == "denial_pattern_x9"


def test_with_no_kaiba_mapping_the_row_still_exists_with_nulls():
    """No row pairs (kaiba db absent/unreadable/no match): the row still exists."""
    parsed = parse_file(TRANSCRIPT)
    rows = extract_from_cursor(parsed, None)
    row = rows[0]
    assert row.output is None
    assert row.output_truncated is False
    assert row.ts is None
    assert json.loads(row.overlap_tokens) == []
    assert row.overlap_count == 0


def test_a_paired_kaiba_row_supplies_output_ts_and_overlap():
    parsed = parse_file(TRANSCRIPT)
    ts = datetime(2026, 8, 20, 10, 0, 0)
    mapping = {
        "denial_pattern_x9": [(ts, "Use anchored prefix matching for denial_pattern_x9 tokens.")]
    }
    rows = extract_from_cursor(parsed, mapping)
    row = rows[0]
    assert row.ts == ts
    assert row.output == "Use anchored prefix matching for denial_pattern_x9 tokens."
    # "denial_pattern_x9" appears in the output and in the post-recall suffix
    # (the second assistant line), so it is introduced, distinctive evidence.
    assert "denial_pattern_x9" in json.loads(row.overlap_tokens)
    assert row.overlap_count >= 1


def test_the_same_query_twice_pairs_with_kaiba_rows_in_order(tmp_path: Path):
    """The n-th occurrence of a query in the transcript pairs with the n-th ledger row."""
    lines = [
        {
            "role": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "CallMcpTool",
                        "input": {
                            "server": "kaiba",
                            "toolName": "recall",
                            "arguments": {"query": "q"},
                        },
                    }
                ]
            },
        },
        {
            "role": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "CallMcpTool",
                        "input": {
                            "server": "kaiba",
                            "toolName": "recall",
                            "arguments": {"query": "q"},
                        },
                    }
                ]
            },
        },
    ]
    path = tmp_path / "repeat.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    parsed = parse_file(path)
    ts1, ts2 = datetime(2026, 1, 1), datetime(2026, 1, 2)
    mapping = {"q": [(ts1, "first"), (ts2, "second")]}
    rows = extract_from_cursor(parsed, mapping)
    assert len(rows) == 2
    assert rows[0].seq == 1 and rows[0].output == "first" and rows[0].ts == ts1
    assert rows[1].seq == 2 and rows[1].output == "second" and rows[1].ts == ts2


def _single_recall_line(query: str) -> dict:
    return {
        "role": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "name": "CallMcpTool",
                    "input": {
                        "server": "kaiba",
                        "toolName": "recall",
                        "arguments": {"query": query},
                    },
                }
            ]
        },
    }


def test_the_same_query_twice_in_two_separate_files_each_pairs_with_the_first_ledger_row(
    tmp_path: Path,
):
    """Occurrences are counted per file: two files each pair with occurrence index 0."""
    file_a = tmp_path / "session_a.jsonl"
    file_b = tmp_path / "session_b.jsonl"
    file_a.write_text(json.dumps(_single_recall_line("q")) + "\n", encoding="utf-8")
    file_b.write_text(json.dumps(_single_recall_line("q")) + "\n", encoding="utf-8")

    ts1, ts2 = datetime(2026, 1, 1), datetime(2026, 1, 2)
    mapping = {"q": [(ts1, "first"), (ts2, "second")]}

    rows_a = extract_from_cursor(parse_file(file_a), mapping)
    rows_b = extract_from_cursor(parse_file(file_b), mapping)

    assert rows_a[0].output == "first" and rows_a[0].ts == ts1
    assert rows_b[0].output == "first" and rows_b[0].ts == ts1


def test_each_call_pairs_correctly_within_one_file(tmp_path: Path):
    path = tmp_path / "session.jsonl"
    path.write_text(json.dumps(_single_recall_line("q")) + "\n", encoding="utf-8")
    mapping = {"q": [(datetime(2026, 1, 1), "only")]}
    rows = extract_from_cursor(parse_file(path), mapping)
    assert rows[0].output == "only"


def test_a_query_with_fewer_ledger_rows_than_occurrences_leaves_the_extra_row_null():
    parsed = parse_file(TRANSCRIPT)
    mapping = {"denial_pattern_x9": []}
    rows = extract_from_cursor(parsed, mapping)
    assert rows[0].output is None
    assert rows[0].ts is None


def test_the_same_line_mcp_call_is_neither_prefix_nor_suffix_of_the_recall():
    """The recall's own line has another MCP call; _split excludes same-line items."""
    parsed = parse_file(TRANSCRIPT)
    rows = extract_from_cursor(parsed)
    followup = rows[0].followup_text
    assert followup is not None
    # Only the strictly-later assistant line's evidence appears.
    assert "Applying denial_pattern_x9" in followup
    assert "notes.md" in followup
    # The same-line "other" MCP call is excluded (neither prefix nor suffix).
    assert "other" not in followup and "lookup" not in followup


# ---------------------------------------------------------------- kaiba sqlite join fixture


def _make_kaiba_recalls_db(
    path: Path, recalls: list[tuple[str, str, str]], conclusions: list[tuple[int, str]]
) -> None:
    """A tiny kaiba db with just the ``recalls`` / ``conclusions`` columns this join reads."""
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE recalls (id INTEGER PRIMARY KEY, created_at TEXT, agent TEXT, "
            "query TEXT, top_k INTEGER, matches TEXT, mu REAL, sd REAL, floor_z REAL, "
            "below_floor INTEGER)"
        )
        connection.execute(
            "CREATE TABLE conclusions (id INTEGER PRIMARY KEY, content TEXT, author TEXT, "
            "created_at TEXT, embedding TEXT, embedding_model TEXT, retired_at TEXT)"
        )
        connection.executemany(
            "INSERT INTO recalls (created_at, agent, query, matches) VALUES (?, 'cursor', ?, ?)",
            recalls,
        )
        connection.executemany(
            "INSERT INTO conclusions (id, content) VALUES (?, ?)", conclusions
        )
        connection.commit()
    finally:
        connection.close()


def test_fetch_cursor_kaiba_recalls_joins_matches_against_conclusions(tmp_path: Path):
    """Exercises the real sqlite join build.py performs, not a hand-built mapping."""
    from ashiato.build import _fetch_cursor_kaiba_recalls
    from ashiato.salvage import open_kaiba

    kaiba_path = tmp_path / "kaiba.db"
    _make_kaiba_recalls_db(
        kaiba_path,
        recalls=[
            (
                "2026-08-20T10:00:00Z",
                "denial_pattern_x9",
                json.dumps([{"id": 1, "score": 0.9}, {"id": 2, "score": 0.5}]),
            ),
            ("2026-08-20T11:00:00Z", "unmatched query", json.dumps([{"id": 999, "score": 0.1}])),
        ],
        conclusions=[
            (1, "Use anchored prefix matching for denial_pattern_x9 tokens."),
            (2, "See also the denial pattern tests."),
        ],
    )
    connection = open_kaiba(kaiba_path, probe_table="recalls")
    assert connection is not None
    try:
        by_query = _fetch_cursor_kaiba_recalls(connection)
    finally:
        connection.close()

    ts, output = by_query["denial_pattern_x9"][0]
    assert ts == datetime(2026, 8, 20, 10, 0, 0)
    assert output == (
        "Use anchored prefix matching for denial_pattern_x9 tokens.\n"
        "See also the denial pattern tests."
    )
    # A matches id with no conclusions row (999) contributes nothing, not an error.
    _, unmatched_output = by_query["unmatched query"][0]
    assert unmatched_output == ""


def test_open_kaiba_with_recalls_probe_returns_none_when_table_is_missing(tmp_path: Path):
    """A kaiba db that only has ``actions`` (no ``recalls``) is unusable for this join."""
    from ashiato.salvage import open_kaiba

    kaiba_path = tmp_path / "actions_only.db"
    connection = sqlite3.connect(kaiba_path)
    connection.execute("CREATE TABLE actions (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()
    assert open_kaiba(kaiba_path, probe_table="recalls") is None
    usable = open_kaiba(kaiba_path, probe_table="actions")
    assert usable is not None
    usable.close()
