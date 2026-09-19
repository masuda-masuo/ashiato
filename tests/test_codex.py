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
