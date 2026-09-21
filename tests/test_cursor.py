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
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ashiato.cursor import (
    ChatMeta,
    CursorTextChunk,
    CursorToolCall,
    classify_store_result,
    parse_chat_meta,
    parse_chat_store,
    parse_file,
    store_result_text,
)
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


# ---------------------------------------------------------------- parse_chat_meta

#: Known epoch-millisecond values, shared by the reader tests and the build tests.
META_CREATED_MS = 1788692232863
META_UPDATED_MS = 1788692482316

NORMAL_META = (
    '{"schemaVersion":1,"createdAtMs":1788692232863,"hasConversation":true,'
    '"updatedAtMs":1788692482316,"cwd":"/home/testuser/dev/projects/kairanban"}'
)


def _write_meta(tmp_path: Path, session_id: str, body: str) -> Path:
    """One chats-layout meta: ``<chats>/<workspace-hash>/<session-id>/meta.json``."""
    session_dir = tmp_path / "chats" / "abc123" / session_id
    session_dir.mkdir(parents=True)
    path = session_dir / "meta.json"
    path.write_text(body, encoding="utf-8")
    return path


def test_session_id_comes_from_the_directory_name_not_the_file(tmp_path: Path):
    """Criterion 1: the file has no id of its own; the directory *is* the session."""
    path = _write_meta(tmp_path, "sess-abc", NORMAL_META)
    meta = parse_chat_meta(path)
    assert isinstance(meta, ChatMeta)
    assert meta.session_id == "sess-abc"
    assert "sess-abc" not in NORMAL_META  # the file contents carry no id


def test_parse_chat_meta_returns_aware_utc_datetimes_and_the_rest(tmp_path: Path):
    """Criterion 1: cwd, both times as timezone-aware UTC datetimes, has_conversation."""
    path = _write_meta(tmp_path, "sess-abc", NORMAL_META)
    meta = parse_chat_meta(path)
    assert meta is not None
    assert meta.cwd == "/home/testuser/dev/projects/kairanban"
    assert meta.created_at == datetime.fromtimestamp(META_CREATED_MS / 1000, tz=UTC)
    assert meta.updated_at == datetime.fromtimestamp(META_UPDATED_MS / 1000, tz=UTC)
    assert meta.created_at.tzinfo is not None
    assert meta.created_at.utcoffset() == timedelta(0)  # UTC, not a local offset
    assert meta.has_conversation is True


def test_parse_chat_meta_reads_the_abandoned_session_shape(tmp_path: Path):
    """A ``hasConversation: false`` meta (no store.db session) still parses cleanly."""
    path = _write_meta(
        tmp_path,
        "sess-empty",
        '{"schemaVersion":1,"createdAtMs":1787997347653,"hasConversation":false,'
        '"updatedAtMs":1787997348407,"cwd":"/home/testuser/dev/projects/cursor"}',
    )
    meta = parse_chat_meta(path)
    assert meta is not None
    assert meta.session_id == "sess-empty"
    assert meta.has_conversation is False
    assert meta.cwd == "/home/testuser/dev/projects/cursor"


@pytest.mark.parametrize(
    ("name", "body", "bad_field"),
    [
        # Criterion 2, case 1: malformed JSON.
        ("malformed", "not json at all", "cwd"),
        # Criterion 2, case 2: an empty file.
        ("empty", "", "cwd"),
        # Criterion 2, case 3: a missing cwd key.
        (
            "no-cwd",
            '{"schemaVersion":1,"createdAtMs":1788692232863,"hasConversation":true,'
            '"updatedAtMs":1788692482316}',
            "cwd",
        ),
        # Criterion 2, case 4: a missing createdAtMs key.
        (
            "no-created",
            '{"schemaVersion":1,"hasConversation":true,"updatedAtMs":1788692482316,'
            '"cwd":"/home/u"}',
            "created_at",
        ),
        # Criterion 2, case 5: a non-string cwd.
        (
            "str-cwd",
            '{"schemaVersion":1,"createdAtMs":1788692232863,"hasConversation":true,'
            '"updatedAtMs":1788692482316,"cwd":42}',
            "cwd",
        ),
    ],
)
def test_a_bad_meta_field_is_none_and_never_raises(tmp_path: Path, name: str, body: str, bad_field: str):
    """Criterion 2: five separate cases, each a record with that field None, no exception."""
    path = _write_meta(tmp_path, name, body)
    meta = parse_chat_meta(path)
    assert meta is not None, name
    assert meta.session_id == name  # the directory name survives the bad file
    assert getattr(meta, bad_field) is None, name


def test_parse_chat_meta_never_raises_on_other_malformed_shapes(tmp_path: Path):
    """A non-object JSON value, a wrong-time type, and a missing file are None fields too."""
    path = _write_meta(tmp_path, "array", "[1, 2, 3]")
    meta = parse_chat_meta(path)
    assert meta is not None and meta.cwd is None and meta.created_at is None

    path = _write_meta(
        tmp_path,
        "str-times",
        '{"schemaVersion":1,"createdAtMs":"1788692232863","hasConversation":true,'
        '"updatedAtMs":"1788692482316","cwd":"/home/u"}',
    )
    meta = parse_chat_meta(path)
    assert meta is not None and meta.created_at is None and meta.updated_at is None

    missing = tmp_path / "chats" / "abc123" / "missing" / "meta.json"
    meta = parse_chat_meta(missing)  # file does not exist: fields None, no exception
    assert meta is not None
    assert meta.session_id == "missing"
    assert meta.cwd is None and meta.created_at is None and meta.updated_at is None


def test_no_usable_session_id_returns_none(tmp_path: Path):
    """A meta.json directly in a filesystem root has no parent name: the whole record is None."""
    assert parse_chat_meta(Path("/meta.json")) is None


# ---------------------------------------------------------------- parse_chat_store


def _store_blob_id(n: int) -> bytes:
    """A deterministic 32-byte blob id."""
    return n.to_bytes(32, "big")


def _store_root_blob(child_ids: list[bytes]) -> bytes:
    """A protobuf whose repeated field 1 is the child ids, in order."""
    out = bytearray()
    for child in child_ids:
        out.append(0x0A)  # field 1, wire type 2
        out.append(len(child))  # 32, one varint byte
        out.extend(child)
    return bytes(out)


def _store_message(parts: list[dict]) -> bytes:
    """One JSON message blob: a chat message whose content holds the parts."""
    return json.dumps(
        {"role": "assistant", "content": parts, "id": "msg", "providerOptions": {}}
    ).encode("utf-8")


def _write_store_db(
    path: Path,
    messages: list[bytes],
    *,
    root_child_ids: list[bytes] | None = None,
) -> None:
    """A minimal chats ``store.db``: ``meta`` (hex-encoded JSON) + ``blobs``.

    ``blobs`` holds one row per message in insertion order, plus a root blob
    whose protobuf field-1 children are the checkpoint window.  By default
    the root lists every message; ``root_child_ids`` makes it list a subset
    (the real shape).  The parser reads the table, not the root, so the
    window only matters as fixture realism.
    """
    root_id = _store_blob_id(0)
    child_ids = [_store_blob_id(i + 1) for i in range(len(messages))]
    root = _store_root_blob(root_child_ids if root_child_ids is not None else child_ids)
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE meta (key TEXT, value TEXT)")
        connection.execute(
            "INSERT INTO meta VALUES ('conversation', ?)",
            [json.dumps({"latestRootBlobId": root_id.hex()}).encode("utf-8").hex()],
        )
        connection.execute("CREATE TABLE blobs (id BLOB PRIMARY KEY, data BLOB)")
        connection.execute("INSERT INTO blobs VALUES (?, ?)", [root_id, root])
        connection.executemany(
            "INSERT INTO blobs VALUES (?, ?)", zip(child_ids, messages, strict=True)
        )
        connection.commit()
    finally:
        connection.close()


def test_parse_chat_store_returns_tool_calls_in_conversation_order(tmp_path: Path):
    """Criterion 1: ids, names, args and results, in table (conversation) order.

    A tool-result message interleaved between call messages, and two calls in
    one message, must not reorder anything: the returned calls follow the
    ``blobs`` table's own row order -- insertion order -- not the root blob's
    checkpoint window.
    """
    store = tmp_path / "store.db"
    _write_store_db(
        store,
        [
            _store_message(
                [{"type": "tool-call", "toolCallId": "call-1", "toolName": "Bash", "input": {"command": "ls"}}]
            ),
            _store_message(
                [
                    {"type": "tool-call", "toolCallId": "call-2", "toolName": "Read", "input": {"file_path": "x"}},
                    {"type": "tool-call", "toolCallId": "call-3", "toolName": "recall", "args": {"query": "q"}},
                ]
            ),
            _store_message(
                [
                    {"type": "tool-result", "toolCallId": "call-3", "result": {"status": "error", "message": "boom"}},
                    {"type": "tool-result", "toolCallId": "call-1", "result": "total 42"},
                    {"type": "tool-result", "toolCallId": "call-2", "result": "file contents"},
                ]
            ),
            _store_message([{"type": "text", "text": "unmodelled"}]),
        ],
    )
    calls = parse_chat_store(store)
    assert [c.tool_call_id for c in calls] == ["call-1", "call-2", "call-3"]
    assert calls[0].tool_name == "Bash"
    assert calls[0].args == {"command": "ls"}
    assert calls[0].result == "total 42"
    assert calls[1].tool_name == "Read"
    assert calls[1].result == "file contents"
    # The tool-result for call-3 arrives in the same message as call-1's:
    # matching is by toolCallId, so it still lands on call-3, not on call-1.
    assert calls[2].tool_call_id == "call-3"
    assert calls[2].args == {"query": "q"}
    assert calls[2].result == {"status": "error", "message": "boom"}


def test_parse_chat_store_marks_a_call_without_a_result_part(tmp_path: Path):
    """A call with no matching tool-result keeps has_result False, result None.

    The absence of a result part is not a JSON-null result: the call's fate
    is unknown, and the build must leave outcome / is_error / result_text
    NULL (asserted there); at the parser level this is ``has_result``.
    """
    store = tmp_path / "store.db"
    _write_store_db(
        store,
        [
            _store_message([{"type": "tool-call", "toolCallId": "c", "toolName": "Bash", "input": {}}]),
            _store_message([{"type": "text", "text": "unmodelled"}]),
        ],
    )
    calls = parse_chat_store(store)
    assert len(calls) == 1
    assert calls[0].has_result is False
    assert calls[0].result is None


def _write_store_db_hex_ids(
    path: Path,
    messages: list[bytes],
    *,
    root_child_ids: list[bytes] | None = None,
    meta_payload: dict | None = None,
) -> None:
    """A store in the *real* spelling: ``blobs.id`` is TEXT holding the hex.

    The root blob's protobuf carries the 32-byte raw child ids -- the two
    spellings are exactly what the real store has.  The parser reads the
    ``blobs`` table directly and never looks up ids, so the spelling is only
    fixture realism.  By default the root lists every message;
    ``root_child_ids`` makes it list a *subset*, the real checkpoint-window
    shape (the latest root only names the newest messages).
    """
    root_id = _store_blob_id(0)
    child_ids = [_store_blob_id(i + 1) for i in range(len(messages))]
    root = _store_root_blob(root_child_ids if root_child_ids is not None else child_ids)
    payload = meta_payload if meta_payload is not None else {"latestRootBlobId": root_id.hex()}
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE meta (key TEXT, value TEXT)")
        connection.execute(
            "INSERT INTO meta VALUES ('conversation', ?)",
            [json.dumps(payload).encode("utf-8").hex()],
        )
        connection.execute("CREATE TABLE blobs (id TEXT PRIMARY KEY, data BLOB)")
        connection.execute("INSERT INTO blobs VALUES (?, ?)", [root_id.hex(), root])
        connection.executemany(
            "INSERT INTO blobs VALUES (?, ?)",
            [(child.hex(), message) for child, message in zip(child_ids, messages, strict=True)],
        )
        connection.commit()
    finally:
        connection.close()


def test_parse_chat_store_reads_messages_beyond_the_latest_root(tmp_path: Path):
    """The latest root is a checkpoint window, not the conversation (issue #87).

    Since 2026-07-13 Cursor's CLI saves only *new* transcript entries at each
    checkpoint, so the root blob's field-1 children name a suffix of the
    conversation.  Here the table holds four messages but the root lists only
    the last two -- all four calls must come back, in table (insertion)
    order.  Under the pre-fix code this fixture returns only the root's
    subset, so this test fails before the change.
    """
    store = tmp_path / "store.db"
    _write_store_db_hex_ids(
        store,
        [
            _store_message([{"type": "tool-call", "toolCallId": "call-1", "toolName": "Bash", "input": {"command": "ls"}}]),
            _store_message([{"type": "tool-result", "toolCallId": "call-1", "result": "total 42"}]),
            _store_message([{"type": "tool-call", "toolCallId": "call-2", "toolName": "Read", "input": {"file_path": "x"}}]),
            _store_message([{"type": "tool-result", "toolCallId": "call-2", "result": "contents"}]),
            _store_message([{"type": "tool-call", "toolCallId": "call-3", "toolName": "recall", "args": {"query": "q"}}]),
            _store_message([{"type": "tool-result", "toolCallId": "call-3", "result": "facts"}]),
            _store_message([{"type": "tool-call", "toolCallId": "call-4", "toolName": "Glob", "input": {"pattern": "*.md"}}]),
            _store_message([{"type": "tool-result", "toolCallId": "call-4", "result": "notes.md"}]),
        ],
        # The root's checkpoint window covers only the last two messages.
        root_child_ids=[_store_blob_id(7), _store_blob_id(8)],
    )
    calls = parse_chat_store(store)
    assert [c.tool_call_id for c in calls] == ["call-1", "call-2", "call-3", "call-4"]
    assert calls[0].result == "total 42"
    assert calls[3].tool_name == "Glob"
    assert calls[3].result == "notes.md"


def _bad_store_cases(tmp_path: Path) -> list[tuple[str, Path]]:
    cases: list[tuple[str, Path]] = []
    # Robustness, case 1: a missing file.
    missing = tmp_path / "missing.db"
    cases.append(("missing", missing))
    # Robustness, case 2: bytes that are not a SQLite database.
    not_sqlite = tmp_path / "not.db"
    not_sqlite.write_bytes(b"this is not a sqlite database")
    cases.append(("not-sqlite", not_sqlite))
    # Robustness, case 3: no readable blobs table (only meta).
    no_blobs = tmp_path / "no-blobs.db"
    connection = sqlite3.connect(no_blobs)
    try:
        connection.execute("CREATE TABLE meta (key TEXT, value TEXT)")
        connection.commit()
    finally:
        connection.close()
    cases.append(("no-blobs-table", no_blobs))
    # Robustness, case 4: blobs holds only non-JSON data (the store keeps
    # binary protobuf blobs alongside the JSON messages -- those are skipped).
    non_json = tmp_path / "non-json.db"
    connection = sqlite3.connect(non_json)
    try:
        connection.execute("CREATE TABLE blobs (id BLOB PRIMARY KEY, data BLOB)")
        connection.execute(
            "INSERT INTO blobs VALUES (?, ?)", [_store_blob_id(7), b"not json at all"]
        )
        connection.commit()
    finally:
        connection.close()
    cases.append(("non-json-blob", non_json))
    return cases


@pytest.mark.parametrize(
    "name", ["missing", "not-sqlite", "no-blobs-table", "non-json-blob"]
)
def test_parse_chat_store_robustness_cases_return_empty(tmp_path: Path, name: str):
    """Criterion 2: all robustness cases return [] and never raise."""
    cases = dict(_bad_store_cases(tmp_path))
    assert parse_chat_store(cases[name]) == []


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        # Criterion 7, case 1: a dict with a non-empty error key.
        ({"error": "tool blew up"}, (True, "error")),
        ({"error": ""}, (False, "ok")),  # empty error key is not an error
        # Criterion 7, case 2: a dict whose status says failure.
        ({"status": "error", "message": "boom"}, (True, "error")),
        ({"status": "failed"}, (True, "error")),
        ({"status": "failure"}, (True, "error")),
        # Criterion 7, case 3: each measured string prefix.
        ("Error executing tool 'Bash'", (True, "error")),
        ("Error: Tool execution error: boom", (True, "error")),
        # Criterion 7, case 4: a plain string result.
        ("total 42", (False, "ok")),
        # Criterion 7, case 5: a dict with neither key.
        ({"data": [1, 2], "ok": True}, (False, "ok")),
        ({"status": "success"}, (False, "ok")),
        (None, (False, "ok")),
    ],
)
def test_classify_store_result(result: object, expected: tuple[bool, str]):
    """Criterion 7: the five classifier cases, asserting outcome and is_error."""
    assert classify_store_result(result) == expected


def test_classify_store_result_prefix_matching_is_anchored():
    """Rule 3 is a *prefix* match: a result that merely quotes the error text is ok."""
    assert classify_store_result("The agent said: Error executing tool 'Bash'") == (False, "ok")


def test_store_result_text_renders_a_dict_as_compact_json_and_a_string_as_is():
    assert store_result_text("plain output") == "plain output"
    assert store_result_text({"status": "error", "message": "boom"}) == '{"message":"boom","status":"error"}'
    assert store_result_text(None) == "null"


# ---------------------------------------------------------------- parse_chat_store (conversation, issue #87 stage 3)


def _store_chat_message(role: str, content: object, message_id: str = "msg") -> bytes:
    """One JSON message blob with an explicit role; content is a part list or a bare string.

    The real store writes the system prompt and the initial user_info
    message as bare ``content`` strings (issue #87 premise), and message
    ``id`` is *not* unique within a session -- both shapes need fixtures.
    """
    return json.dumps({"role": role, "content": content, "id": message_id}).encode("utf-8")


def test_parse_chat_store_returns_each_message_with_role_and_parts(tmp_path: Path):
    """Criterion 1 (stage 3): messages in table order, role verbatim, parts ordered."""
    from ashiato.cursor import CursorStoreMessage, CursorStorePart

    store = tmp_path / "store.db"
    _write_store_db(
        store,
        [
            _store_chat_message("system", "You are a coding assistant.", "s"),
            _store_chat_message("user", [{"type": "text", "text": "What now?"}], "u"),
            _store_chat_message(
                "assistant",
                [
                    {"type": "reasoning", "text": "think step one"},
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool-call", "toolCallId": "c1", "toolName": "Bash", "input": {}},
                ],
                "a",
            ),
            _store_chat_message("tool", [{"type": "tool-result", "toolCallId": "c1", "result": "ok"}], "t"),
        ],
    )
    conversation = parse_chat_store(store)
    assert [c.tool_call_id for c in conversation] == ["c1"]  # stage 2's list still works
    assert [m.message_index for m in conversation.messages] == [0, 1, 2, 3]
    assert [m.role for m in conversation.messages] == ["system", "user", "assistant", "tool"]
    first, second, third, fourth = conversation.messages
    assert isinstance(first, CursorStoreMessage)
    assert first.parts == [CursorStorePart(type="text", text="You are a coding assistant.")]
    assert second.parts == [CursorStorePart(type="text", text="What now?")]
    assert [(p.type, p.text) for p in third.parts] == [
        ("reasoning", "think step one"),
        ("text", "Let me check."),
        ("tool-call", None),
    ]
    assert [(p.type, p.text) for p in fourth.parts] == [("tool-result", None)]


def test_parse_chat_store_treats_string_content_as_one_text_part(tmp_path: Path):
    """The real system prompt is a bare ``content`` string: it reads as one text part."""
    store = tmp_path / "store.db"
    _write_store_db(
        store,
        [
            _store_chat_message("system", "You are a coding assistant.", "s"),
            _store_chat_message(
                "user",
                "<user_info>\nOS Version: linux\n</user_info>",
                "u",
            ),
        ],
    )
    conversation = parse_chat_store(store)
    assert [m.role for m in conversation.messages] == ["system", "user"]
    assert conversation.messages[0].parts == [
        type(conversation.messages[0].parts[0])("text", "You are a coding assistant.")
    ]
    assert conversation.messages[1].parts[0].text.startswith("<user_info>")


def test_parse_chat_store_keeps_unknown_part_types_verbatim(tmp_path: Path):
    """An unrecognised part type is kept, not dropped: the build counts it (criterion 7)."""
    store = tmp_path / "store.db"
    _write_store_db(
        store,
        [
            _store_chat_message(
                "assistant",
                [
                    {"type": "mystery", "text": "zz"},
                    {"type": "tool-call", "toolCallId": "c1", "toolName": "Bash", "input": {}},
                ],
                "a",
            ),
            _store_chat_message("tool", [{"type": "tool-result", "toolCallId": "c1", "result": "ok"}], "t"),
        ],
    )
    conversation = parse_chat_store(store)
    assistant = conversation.messages[0]
    assert assistant.parts[0] == type(assistant.parts[0])(type="mystery", text="zz")
    assert [c.tool_call_id for c in conversation] == ["c1"]  # unknown parts never disturb the calls


def test_parse_chat_store_keeps_redacted_reasoning_parts_without_text(tmp_path: Path):
    """A ``redacted-reasoning`` part has no text; it must read None, never a placeholder."""
    store = tmp_path / "store.db"
    _write_store_db(
        store,
        [
            _store_chat_message(
                "assistant",
                [{"type": "redacted-reasoning"}],
                "a",
            ),
        ],
    )
    conversation = parse_chat_store(store)
    assert conversation.messages[0].parts == [
        type(conversation.messages[0].parts[0])(type="redacted-reasoning", text=None)
    ]


def test_parse_chat_store_message_index_counts_json_messages_not_blobs(tmp_path: Path):
    """``message_index`` counts JSON chat messages in table order, not every blob.

    The store keeps binary protobuf blobs (the checkpoint root) among the
    JSON messages; a binary blob must not advance the message index, or the
    deterministic ``event_id`` would depend on how many binary blobs the
    store happens to hold.
    """
    store = tmp_path / "store.db"
    root_id = _store_blob_id(0)
    child_ids = [_store_blob_id(1), _store_blob_id(2)]
    root = _store_root_blob(child_ids)
    connection = sqlite3.connect(store)
    try:
        connection.execute("CREATE TABLE blobs (id BLOB PRIMARY KEY, data BLOB)")
        connection.execute("INSERT INTO blobs VALUES (?, ?)", [root_id, root])
        # A non-JSON blob sits between the two JSON messages.
        connection.execute("INSERT INTO blobs VALUES (?, ?)", [child_ids[0], b"not json"])
        connection.execute(
            "INSERT INTO blobs VALUES (?, ?)",
            [child_ids[1], _store_chat_message("user", [{"type": "text", "text": "hi"}], "u")],
        )
        connection.commit()
    finally:
        connection.close()
    conversation = parse_chat_store(store)
    assert [(m.message_index, m.role) for m in conversation.messages] == [(0, "user")]
