"""Tests for ashiato.codex transcript parsing."""

import json
from pathlib import Path

from ashiato.codex import parse_file


def test_parse_codex_file(tmp_path: Path):
    jsonl_file = tmp_path / "test-session.jsonl"
    lines = [
        {"timestamp": "2026-09-05T10:00:00Z", "type": "session_meta", "payload": {"id": "codex-test-1", "cwd": "/work"}},
        {
            "timestamp": "2026-09-05T10:00:01Z",
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "thread_id": "codex-test-1",
                "item": {
                    "type": "CommandExecution",
                    "id": "exec-1",
                    "command": ["bash", "-c", "echo hello"],
                    "stdout": "hello\n",
                },
            },
        },
        {
            "timestamp": "2026-09-05T10:00:02Z",
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "thread_id": "codex-test-1",
                "item": {
                    "type": "AgentResponse",
                    "id": "resp-1",
                    "text": "Completed command execution.",
                },
            },
        },
    ]
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    parsed = parse_file(jsonl_file)
    assert parsed.session_id == "codex-test-1"
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].tool_name == "Bash"
    assert parsed.tool_calls[0].output == "hello\n"
    assert parsed.tool_calls[0].ts is not None
    assert parsed.tool_calls[0].ts.year == 2026
    assert len(parsed.text_chunks) == 1
    assert parsed.text_chunks[0].text == "Completed command execution."
    assert parsed.text_chunks[0].ts is not None

    # Session-level timestamps from min/max of record timestamps
    assert parsed.started_at is not None
    assert parsed.ended_at is not None
    assert parsed.started_at <= parsed.ended_at


def test_parse_codex_file_token_usage(tmp_path: Path):
    """A token_usage_record with thread_token_usage sets token fields."""
    jsonl_file = tmp_path / "tokens.jsonl"
    lines = [
        {"timestamp": "2026-09-05T10:00:00Z", "type": "session_meta", "payload": {"id": "tok-1"}},
        {
            "timestamp": "2026-09-05T10:00:01Z",
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "thread_id": "tok-1",
                "item": {"type": "CommandExecution", "id": "e1", "command": "pwd", "stdout": "/"},
            },
        },
        {
            "timestamp": "2026-09-05T10:00:02Z",
            "type": "token_usage_record",
            "payload": {
                "thread_token_usage": {
                    "input_tokens": 1200,
                    "cached_input_tokens": 300,
                    "output_tokens": 450,
                },
            },
        },
    ]
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    parsed = parse_file(jsonl_file)
    assert parsed.input_tokens == 1200
    assert parsed.output_tokens == 450
    assert parsed.cache_read_tokens == 300


def test_parse_codex_no_timestamps(tmp_path: Path):
    """A file with no timestamps yields started_at/ended_at = None."""
    jsonl_file = tmp_path / "no-ts.jsonl"
    lines = [
        {"type": "session_meta", "payload": {"id": "no-ts-1"}},
    ]
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    parsed = parse_file(jsonl_file)
    assert parsed.started_at is None
    assert parsed.ended_at is None
    assert parsed.input_tokens == 0
    assert parsed.output_tokens == 0
    assert parsed.cache_read_tokens == 0


def test_parse_codex_failed_command_exposes_status_and_exit_code(tmp_path: Path):
    """A failed CommandExecution carries status, exit code, cwd and duration."""
    jsonl_file = tmp_path / "failed-cmd.jsonl"
    lines = [
        {"type": "session_meta", "payload": {"id": "codex-fail-1"}},
        {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "thread_id": "codex-fail-1",
                "item": {
                    "type": "CommandExecution",
                    "id": "exec-fail-1",
                    "command": ["bash", "-c", "exit 1"],
                    "status": "failed",
                    "exit_code": 1,
                    "cwd": "/work",
                    "duration": 0.25,
                    "stdout": "",
                    "stderr": "boom\n",
                },
            },
        },
    ]
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    parsed = parse_file(jsonl_file)
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.status == "failed"
    assert call.exit_code == 1
    assert call.cwd == "/work"
    assert call.duration_ms == 250


def test_parse_codex_failed_mcp_keeps_error_message_reachable(tmp_path: Path):
    """A failed MCP call with no result surfaces its error as the output."""
    jsonl_file = tmp_path / "failed-mcp.jsonl"
    lines = [
        {"type": "session_meta", "payload": {"id": "codex-fail-mcp"}},
        {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "thread_id": "codex-fail-mcp",
                "item": {
                    "type": "McpToolCall",
                    "id": "mcp-fail-1",
                    "server": "kaiba",
                    "tool": "recall",
                    "arguments": {"query": "x"},
                    "status": "failed",
                    "error": "Server error: connection refused",
                    "result": None,
                    "duration": 1.5,
                },
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "thread_id": "codex-fail-mcp",
                "item": {
                    "type": "McpToolCall",
                    "id": "mcp-fail-2",
                    "server": "kaiba",
                    "tool": "recall",
                    "arguments": {"query": "y"},
                    "status": "failed",
                    "error": "nope",
                    "result": "",
                },
            },
        },
    ]
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    parsed = parse_file(jsonl_file)
    assert len(parsed.tool_calls) == 2
    first, second = parsed.tool_calls
    assert first.status == "failed"
    assert first.error == "Server error: connection refused"
    assert first.output == "Server error: connection refused"
    assert first.duration_ms == 1500
    # result == "" is empty too, so the error text still wins.
    assert second.output == "nope"


def test_parse_codex_duration_seconds_object(tmp_path: Path):
    """Rust-style {'secs','nanos'} durations convert to milliseconds."""
    jsonl_file = tmp_path / "dur-object.jsonl"
    lines = [
        {"type": "session_meta", "payload": {"id": "codex-dur"}},
        {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "thread_id": "codex-dur",
                "item": {
                    "type": "CommandExecution",
                    "id": "dur-1",
                    "command": "sleep 3",
                    "status": "completed",
                    "exit_code": 0,
                    "duration": {"secs": 3, "nanos": 72761183},
                    "stdout": "",
                },
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "thread_id": "codex-dur",
                "item": {
                    "type": "CommandExecution",
                    "id": "dur-2",
                    "command": "true",
                    "status": "completed",
                    "exit_code": 0,
                    "duration": "fast",
                    "stdout": "",
                },
            },
        },
    ]
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    parsed = parse_file(jsonl_file)
    assert len(parsed.tool_calls) == 2
    assert parsed.tool_calls[0].duration_ms == 3072
    # An unparseable duration stays None rather than being guessed at.
    assert parsed.tool_calls[1].duration_ms is None


def test_parse_codex_malformed_status_and_exit_code_do_not_raise(tmp_path: Path):
    """Non-string status and non-int exit_code parse without raising."""
    jsonl_file = tmp_path / "weird-status.jsonl"
    lines = [
        {"type": "session_meta", "payload": {"id": "codex-weird"}},
        {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "thread_id": "codex-weird",
                "item": {
                    "type": "CommandExecution",
                    "id": "weird-1",
                    "command": "ls",
                    "status": 42,
                    "exit_code": "0",
                    "stdout": "x\n",
                },
            },
        },
    ]
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    parsed = parse_file(jsonl_file)
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    # The raw values are preserved so the row mapping can tell that this is
    # not a clean success (see test_build.py).
    assert call.status == 42
    assert call.exit_code == "0"


def test_parse_codex_command_completed_carries_cwd(tmp_path: Path):
    """A completed command with no cwd leaves the field None."""
    jsonl_file = tmp_path / "completed-cmd.jsonl"
    lines = [
        {"type": "session_meta", "payload": {"id": "codex-ok"}},
        {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "thread_id": "codex-ok",
                "item": {
                    "type": "CommandExecution",
                    "id": "ok-1",
                    "command": "pwd",
                    "status": "completed",
                    "exit_code": 0,
                    "stdout": "/work\n",
                },
            },
        },
    ]
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    parsed = parse_file(jsonl_file)
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].status == "completed"
    assert parsed.tool_calls[0].exit_code == 0
    assert parsed.tool_calls[0].cwd is None
def test_parse_codex_live_message_items(tmp_path: Path):
    """Live-shaped response_item with message payload (output_text/input_text)."""
    jsonl_file = tmp_path / "live-message.jsonl"
    lines = [
        {"timestamp": "2026-09-05T10:00:00Z", "type": "session_meta", "payload": {"id": "live-1"}},
        {
            "timestamp": "2026-09-05T10:00:01Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "msg-1",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hello "},
                    {"type": "output_text", "text": "world"},
                ],
            },
        },
        {
            "timestamp": "2026-09-05T10:00:02Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "msg-2",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "User question"},
                ],
            },
        },
        {
            "timestamp": "2026-09-05T10:00:03Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "msg-3",
                "role": "developer",
                "content": [
                    {"type": "input_text", "text": "Dev note"},
                ],
            },
        },
    ]
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    parsed = parse_file(jsonl_file)
    assert parsed.session_id == "live-1"
    # Three message items -> three text chunks
    assert len(parsed.text_chunks) == 3
    assert parsed.text_chunks[0].text == "Hello world"
    assert parsed.text_chunks[1].text == "User question"
    assert parsed.text_chunks[2].text == "Dev note"
    # All have timestamps
    for chunk in parsed.text_chunks:
        assert chunk.ts is not None
    # Roles are carried through from the payload
    assert parsed.text_chunks[0].role == "assistant"
    assert parsed.text_chunks[1].role == "user"
    assert parsed.text_chunks[2].role == "developer"

def _write_items(tmp_path: Path, name: str, items: list[dict]) -> Path:
    """Write one Codex session file whose item_completed payloads are *items*."""
    jsonl_file = tmp_path / name
    lines = [
        {"type": "session_meta", "payload": {"id": "codex-newtypes"}},
    ]
    for item in items:
        lines.append({
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "thread_id": "codex-newtypes",
                "item": item,
            },
        })
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return jsonl_file


def test_parse_codex_file_change_item(tmp_path: Path):
    """A FileChange item becomes a tool call naming the files it touched."""
    jsonl_file = _write_items(tmp_path, "file-change.jsonl", [
        {
            "type": "FileChange",
            "id": "exec-fc-1",
            "changes": {
                "/work/src/foo.py": {"type": "edit", "content": "x"},
                "/work/src/bar.py": {"type": "delete"},
            },
            "status": "completed",
            "stdout": "Success. Updated the following files:\nM /work/src/foo.py\n",
            "stderr": "",
        },
    ])
    parsed = parse_file(jsonl_file)
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.tool_name == "FileChange"
    assert call.input == {
        "files": ["/work/src/foo.py", "/work/src/bar.py"],
        "changes": {
            "/work/src/foo.py": {"type": "edit", "content": "x"},
            "/work/src/bar.py": {"type": "delete"},
        },
    }
    assert call.output and "/work/src/foo.py" in call.output
    assert call.status == "completed"
    # A FileChange is a tool call, never an event.
    assert len(parsed.events) == 0


def test_parse_codex_collab_agent_tool_call_item(tmp_path: Path):
    """A CollabAgentToolCall item becomes a collab__<tool> tool call."""
    jsonl_file = _write_items(tmp_path, "collab.jsonl", [
        {
            "type": "CollabAgentToolCall",
            "id": "call-wait-1",
            "tool": "wait",
            "status": "completed",
            "sender_thread_id": "thread-parent",
            "receiver_thread_ids": ["thread-child"],
            "receiver_agents": ["sol"],
            "agents_states": {"sol": "working"},
        },
    ])
    parsed = parse_file(jsonl_file)
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.tool_name == "collab__wait"
    assert call.input == {
        "sender_thread_id": "thread-parent",
        "receiver_thread_ids": ["thread-child"],
        "receiver_agents": ["sol"],
        "agents_states": {"sol": "working"},
    }
    assert call.status == "completed"
    assert call.output is None


def test_parse_codex_sub_agent_activity_item(tmp_path: Path):
    """A SubAgentActivity item becomes a collab__subagent tool call."""
    jsonl_file = _write_items(tmp_path, "subagent.jsonl", [
        {
            "type": "SubAgentActivity",
            "id": "call-sa-1",
            "kind": "started",
            "agent_thread_id": "01a07aa0-56fa-78a1-ad92-9c129b72e239",
            "agent_path": "/root/kusabi_484_luna",
        },
    ])
    parsed = parse_file(jsonl_file)
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.tool_name == "collab__subagent"
    assert call.input == {
        "kind": "started",
        "agent_thread_id": "01a07aa0-56fa-78a1-ad92-9c129b72e239",
        "agent_path": "/root/kusabi_484_luna",
    }


def test_parse_codex_context_compaction_is_an_event(tmp_path: Path):
    """A ContextCompaction item becomes an events entry, never a tool call."""
    jsonl_file = _write_items(tmp_path, "compaction.jsonl", [
        {"type": "ContextCompaction", "id": "compaction-1"},
    ])
    parsed = parse_file(jsonl_file)
    assert len(parsed.tool_calls) == 0
    assert len(parsed.text_chunks) == 0
    assert len(parsed.events) == 1
    event = parsed.events[0]
    assert event.type == "context_compaction"
    assert event.event_id == "compaction-1"
    assert event.text is None
    # The verbatim item JSON survives in raw.
    assert '"type": "ContextCompaction"' in event.raw
    assert '"id": "compaction-1"' in event.raw


def test_parse_codex_new_item_types_tolerate_malformed_instances(tmp_path: Path):
    """Missing keys and wrong value types never raise for the new item types."""
    jsonl_file = _write_items(tmp_path, "malformed.jsonl", [
        # FileChange with a non-dict changes field and no stdout.
        {"type": "FileChange", "id": "fc-1", "changes": ["/work/x.py"], "status": 42},
        # FileChange with no changes at all.
        {"type": "FileChange", "id": "fc-2"},
        # CollabAgentToolCall with a non-string tool and no receivers.
        {"type": "CollabAgentToolCall", "id": "ca-1", "tool": 42, "status": "completed"},
        # SubAgentActivity with nothing but a kind of the wrong type.
        {"type": "SubAgentActivity", "id": "sa-1", "kind": ["started"]},
        # ContextCompaction with no id.
        {"type": "ContextCompaction"},
    ])
    parsed = parse_file(jsonl_file)
    # No raise above; every item still yields a row.
    assert len(parsed.tool_calls) == 4
    file_changes = [c for c in parsed.tool_calls if c.tool_name == "FileChange"]
    assert len(file_changes) == 2
    # Non-dict changes -> empty files list; missing stdout -> None output.
    assert file_changes[0].input == {"files": [], "changes": {}}
    assert file_changes[0].output is None
    assert file_changes[0].status == 42
    # Non-string tool -> the item type names the row.
    collab = [c for c in parsed.tool_calls if c.tool_name == "CollabAgentToolCall"]
    assert len(collab) == 1
    assert collab[0].input == {}
    assert collab[0].status == "completed"
    subagents = [c for c in parsed.tool_calls if c.tool_name == "collab__subagent"]
    assert len(subagents) == 1
    # A wrong-typed field is preserved as-is (never raising), exactly like the
    # existing branches keep a malformed status/exit_code.
    assert subagents[0].input == {"kind": ["started"]}
    assert len(parsed.events) == 1
    assert parsed.events[0].type == "context_compaction"


def test_parse_codex_unknown_item_type_is_dropped_silently(tmp_path: Path):
    """An item type ashiato does not know is dropped without raising."""
    jsonl_file = _write_items(tmp_path, "unknown.jsonl", [
        {"type": "SomeFutureItemType", "id": "future-1", "payload": "whatever"},
    ])
    parsed = parse_file(jsonl_file)
    assert len(parsed.tool_calls) == 0
    assert len(parsed.text_chunks) == 0
    assert len(parsed.events) == 0
    assert parsed.n_parse_errors == 0


def test_parse_codex_role_fallback_for_missing_role(tmp_path: Path):
    """A response_item message with no role still produces a chunk with role=None."""
    jsonl_file = tmp_path / "no-role.jsonl"
    lines = [
        {"timestamp": "2026-09-05T10:00:00Z", "type": "session_meta", "payload": {"id": "no-role-1"}},
        {
            "timestamp": "2026-09-05T10:00:01Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "msg-norole",
                "content": [
                    {"type": "output_text", "text": "No role here"},
                ],
            },
        },
    ]
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    parsed = parse_file(jsonl_file)
    assert len(parsed.text_chunks) == 1
    assert parsed.text_chunks[0].role is None
    assert parsed.text_chunks[0].text == "No role here"


def test_parse_codex_role_fallback_for_non_string_role(tmp_path: Path):
    """A response_item message with a non-string role still produces a chunk."""
    jsonl_file = tmp_path / "bad-role.jsonl"
    lines = [
        {"timestamp": "2026-09-05T10:00:00Z", "type": "session_meta", "payload": {"id": "bad-role-1"}},
        {
            "timestamp": "2026-09-05T10:00:01Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "msg-badrole",
                "role": 42,
                "content": [
                    {"type": "output_text", "text": "Bad role value"},
                ],
            },
        },
    ]
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    parsed = parse_file(jsonl_file)
    assert len(parsed.text_chunks) == 1
    # Non-string role is preserved as-is; build layer applies fallback
    assert parsed.text_chunks[0].role == 42
    assert parsed.text_chunks[0].text == "Bad role value"


def test_parse_codex_item_completed_plus_response_item_dedup(tmp_path: Path):
    """An item_completed of type AgentMessage (or UserMessage) that carries the
    same text as a response_item message must produce exactly one text chunk —
    from the response_item path only.

    The parser does not match AgentMessage/UserMessage in item_completed, so
    that path contributes no chunk.  (AgentResponse *is* matched and would
    produce a second chunk, but that shape does not appear in real dedup
    scenarios and was ruled out of scope here.)
    """
    jsonl_file = tmp_path / "dedup.jsonl"
    lines = [
        {"timestamp": "2026-09-05T10:00:00Z", "type": "session_meta", "payload": {"id": "dedup-1"}},
        # item_completed AgentMessage — parser does NOT match this type
        {
            "timestamp": "2026-09-05T10:00:01Z",
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "thread_id": "dedup-1",
                "item": {
                    "type": "AgentMessage",
                    "id": "msg-dedup-1",
                    "content": [
                        {"type": "output_text", "text": "Hello from agent"},
                    ],
                },
            },
        },
        # response_item message — parser DOES match this
        {
            "timestamp": "2026-09-05T10:00:02Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "msg-dedup",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hello from agent"},
                ],
            },
        },
    ]
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    parsed = parse_file(jsonl_file)
    assert len(parsed.text_chunks) == 1
    chunk = parsed.text_chunks[0]
    assert chunk.role == "assistant"
    assert chunk.text == "Hello from agent"
