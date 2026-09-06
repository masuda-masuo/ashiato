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
    assert len(parsed.text_chunks) == 1
    assert parsed.text_chunks[0].text == "Completed command execution."
    assert parsed.n_parse_errors == 0
