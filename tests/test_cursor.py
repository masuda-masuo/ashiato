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
    '"updatedAtMs":1788692482316,"cwd":"/home/masuda/dev/projects/kairanban"}'
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
    assert meta.cwd == "/home/masuda/dev/projects/kairanban"
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
        '"updatedAtMs":1787997348407,"cwd":"/home/masuda/dev/projects/cursor"}',
    )
    meta = parse_chat_meta(path)
    assert meta is not None
    assert meta.session_id == "sess-empty"
    assert meta.has_conversation is False
    assert meta.cwd == "/home/masuda/dev/projects/cursor"


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
    root_blob: bytes | None = None,
    root_id: bytes | None = None,
    meta_payload: dict | None = None,
) -> None:
    """A minimal chats ``store.db``: ``meta`` (hex-encoded JSON) + ``blobs``.

    The root blob's field-1 children are the messages, in order -- the same
    blob graph shape the real store uses, tiny enough for fixtures.
    """
    root_id = root_id if root_id is not None else _store_blob_id(0)
    child_ids = [_store_blob_id(i + 1) for i in range(len(messages))]
    root = root_blob if root_blob is not None else _store_root_blob(child_ids)
    payload = meta_payload if meta_payload is not None else {"latestRootBlobId": root_id.hex()}
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE meta (key TEXT, value TEXT)")
        connection.execute(
            "INSERT INTO meta VALUES ('conversation', ?)",
            [json.dumps(payload).encode("utf-8").hex()],
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
    """Criterion 1: ids, names, args and results, ordered by the blob graph.

    A tool-result message interleaved between call messages, and two calls in
    one message, must not reorder anything: the returned calls follow the
    message order of the root blob's children, not the table layout.
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


def test_parse_chat_store_reads_a_meta_row_not_at_position_zero(tmp_path: Path):
    """Only the row whose value decodes to JSON with latestRootBlobId qualifies."""
    store = tmp_path / "store.db"
    messages = [_store_message([{"type": "tool-call", "toolCallId": "c", "toolName": "Bash", "input": {}}])]
    _write_store_db(store, messages)
    connection = sqlite3.connect(store)
    try:
        connection.execute(
            "INSERT INTO meta VALUES ('other', ?)", [b"not hex".hex()]
        )
        connection.commit()
    finally:
        connection.close()
    calls = parse_chat_store(store)
    assert len(calls) == 1
    # No tool-result part exists for this call: has_result must be False --
    # the parser must not present the absent result as a JSON-null success.
    assert calls[0].has_result is False
    assert calls[0].result is None


def _write_store_db_hex_ids(
    path: Path,
    messages: list[bytes],
    *,
    meta_payload: dict | None = None,
) -> None:
    """A store in the *real* spelling: ``blobs.id`` is TEXT holding the hex.

    The root blob's protobuf still carries the 32-byte raw child ids -- the
    mismatch between the two spellings is exactly the shape the real store
    has, and the lookup must bridge it (issue #87 stage 2 repair).
    """
    root_id = _store_blob_id(0)
    child_ids = [_store_blob_id(i + 1) for i in range(len(messages))]
    root = _store_root_blob(child_ids)
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


def test_parse_chat_store_resolves_raw_byte_children_against_hex_text_ids(tmp_path: Path):
    """The real spelling (issue #87 stage 2 repair): TEXT hex ids, raw-byte children.

    ``blobs.id`` is a TEXT column holding the hex of the id while the root
    protobuf's field-1 children are the raw 32 bytes.  Before the fix the
    bytes branch of ``_blob_id_candidates`` only tried the raw spelling, so
    every child lookup missed and the parser returned zero calls; the fix
    makes the bytes branch symmetric and tries ``.hex()`` as well.  The
    BLOB-id spelling is exercised by the other fixtures (``_write_store_db``).
    """
    store = tmp_path / "store.db"
    _write_store_db_hex_ids(
        store,
        [
            _store_message([{"type": "tool-call", "toolCallId": "call-1", "toolName": "Bash", "input": {"command": "ls"}}]),
            _store_message([{"type": "tool-result", "toolCallId": "call-1", "result": "total 42"}]),
            _store_message([{"type": "tool-call", "toolCallId": "call-2", "toolName": "Read", "input": {"file_path": "x"}}]),
            _store_message([{"type": "tool-result", "toolCallId": "call-2", "result": "contents"}]),
        ],
    )
    calls = parse_chat_store(store)
    assert [c.tool_call_id for c in calls] == ["call-1", "call-2"]
    assert calls[0].tool_name == "Bash"
    assert calls[0].result == "total 42"
    assert calls[1].tool_name == "Read"
    assert calls[1].result == "contents"


def _bad_store_cases(tmp_path: Path) -> list[tuple[str, Path]]:
    cases: list[tuple[str, Path]] = []
    # Criterion 2, case 1: a missing file.
    missing = tmp_path / "missing.db"
    cases.append(("missing", missing))
    # Criterion 2, case 2: bytes that are not a SQLite database.
    not_sqlite = tmp_path / "not.db"
    not_sqlite.write_bytes(b"this is not a sqlite database")
    cases.append(("not-sqlite", not_sqlite))
    # Criterion 2, case 3: a meta table with no qualifying row.
    no_meta = tmp_path / "no-meta.db"
    connection = sqlite3.connect(no_meta)
    try:
        connection.execute("CREATE TABLE meta (key TEXT, value TEXT)")
        connection.execute("CREATE TABLE blobs (id BLOB PRIMARY KEY, data BLOB)")
        connection.commit()
    finally:
        connection.close()
    cases.append(("no-meta-row", no_meta))
    # Criterion 2, case 4: a meta row whose root id is not in blobs.
    no_root = tmp_path / "no-root.db"
    _write_store_db(no_root, [], meta_payload={"latestRootBlobId": "ff" * 32})
    cases.append(("root-not-in-blobs", no_root))
    # Criterion 2, case 5: a truncated root protobuf (field says 32 bytes, only 10 follow).
    truncated = tmp_path / "truncated.db"
    _write_store_db(truncated, [], root_blob=b"\x0a\x20" + _store_blob_id(1)[:10])
    cases.append(("truncated-protobuf", truncated))
    # Criterion 2, case 6: a child blob that is not JSON.
    non_json = tmp_path / "non-json.db"
    _write_store_db(non_json, [], root_blob=_store_root_blob([_store_blob_id(7)]))
    connection = sqlite3.connect(non_json)
    try:
        connection.execute("INSERT INTO blobs VALUES (?, ?)", [_store_blob_id(7), b"not json at all"])
        connection.commit()
    finally:
        connection.close()
    cases.append(("non-json-child", non_json))
    return cases


@pytest.mark.parametrize("name", ["missing", "not-sqlite", "no-meta-row", "root-not-in-blobs", "truncated-protobuf", "non-json-child"])
def test_parse_chat_store_robustness_cases_return_empty(tmp_path: Path, name: str):
    """Criterion 2: all six cases return [] and never raise."""
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
