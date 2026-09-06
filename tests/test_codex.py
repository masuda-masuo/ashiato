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
