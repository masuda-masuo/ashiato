from __future__ import annotations

import json
import zipfile
from pathlib import Path

from ashiato.build import build, connect
from ashiato.cli import main


def _msg(mid, parent, sender, stamp, *blocks):
    return {
        "uuid": mid,
        "parent_message_uuid": parent,
        "sender": sender,
        "created_at": stamp,
        "text": "".join(b.get("text", "") for b in blocks if b.get("type") == "text"),
        "content": list(blocks),
    }


def _export():
    root = _msg(
        "msg-root",
        "00000000-0000-4000-8000-000000000000",
        "human",
        "2026-01-01T00:00:00Z",
        {"type": "text", "text": "synthetic human prompt"},
        {"type": "injected_prompt_block", "prompt": "synthetic injected secret"},
    )
    branch = _msg(
        "msg-old-branch",
        "msg-root",
        "assistant",
        "2026-01-01T00:00:01Z",
        {"type": "text", "text": "synthetic abandoned branch"},
    )
    latest = _msg(
        "msg-latest",
        "msg-root",
        "assistant",
        "2026-01-01T00:00:03Z",
        {"type": "thinking", "thinking": "synthetic hidden reasoning"},
        {"type": "text", "text": "synthetic latest answer"},
        {
            "type": "tool_use",
            "id": "use-ok",
            "name": "search",
            "input": {"query": "invented"},
            "start_timestamp": "2026-01-01T00:00:02Z",
        },
        {
            "type": "tool_use",
            "id": "use-pending",
            "name": "write",
            "input": {},
            "start_timestamp": "2026-01-01T00:00:03Z",
        },
        {"type": "token_budget"},
        {"type": "flag"},
        {"type": "image"},
        {"type": "document"},
        {"type": "future_widget"},
    )
    result = _msg(
        "msg-result",
        "msg-latest",
        "assistant",
        "2026-01-01T00:00:04Z",
        {
            "type": "tool_result",
            "tool_use_id": "use-ok",
            "content": "synthetic tool output",
            "is_error": True,
        },
    )
    two_root_a = _msg(
        "two-a",
        "00000000-0000-4000-8000-000000000000",
        "human",
        "2026-01-02T00:00:00Z",
        {"type": "text", "text": "synthetic second root ignored"},
    )
    two_root_b = _msg(
        "two-b",
        "00000000-0000-4000-8000-000000000000",
        "human",
        "2026-01-02T00:00:01Z",
        {"type": "text", "text": "synthetic chosen second root"},
    )
    return [
        {
            "uuid": "conv-branch",
            "name": "Synthetic Branch Conversation",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:01:00Z",
            "chat_messages": [root, branch, latest, result],
        },
        {
            "uuid": "conv-two-roots",
            "name": "Synthetic Roots Conversation",
            "created_at": "2026-01-02T00:00:00Z",
            "updated_at": "2026-01-02T00:01:00Z",
            "chat_messages": [two_root_a, two_root_b],
        },
        {
            "uuid": "conv-empty",
            "name": "Synthetic Empty Conversation",
            "created_at": "2026-01-03T00:00:00Z",
            "updated_at": "2026-01-03T00:01:00Z",
            "chat_messages": [],
        },
        {"name": "synthetic malformed name", "chat_messages": []},
    ]


def _write(path: Path, conversations):
    path.write_text(json.dumps(conversations), encoding="utf-8")


def _dump(connection, table):
    return connection.execute(f"SELECT * FROM {table} ORDER BY ALL").fetchall()


def test_synthetic_export_import_incremental_and_privacy(tmp_path, capsys):
    bare = tmp_path / "conversations.json"
    _write(bare, _export())
    zip_path = tmp_path / "conversations-000.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(bare, "conversations.json")
    db = tmp_path / "test.duckdb"

    assert main(["build", "--claude-ai-export", str(zip_path), "--db", str(db)]) == 0
    out = capsys.readouterr()
    rendered = out.out + out.err
    for private in (
        "Synthetic Branch Conversation",
        "conv-branch",
        "synthetic human prompt",
        "synthetic injected secret",
        "synthetic hidden reasoning",
        "synthetic malformed name",
    ):
        assert private not in rendered
    assert "4 conversations found" in rendered
    assert "3 imported" in rendered
    assert "1 malformed" in rendered
    assert "2 with messages" in rendered
    assert "2 off-path" in rendered
    for skipped in ("token_budget=1", "flag=1", "image=1", "document=1"):
        assert skipped in rendered
    assert "future_widget=1" in rendered

    connection = connect(db, read_only=True)
    rows = connection.execute(
        "SELECT session_id, text, role, is_meta FROM events WHERE source='claude_ai' ORDER BY session_id, seq"
    ).fetchall()
    assert ("conv-branch", "synthetic abandoned branch", "assistant", False) not in rows
    assert ("conv-two-roots", "synthetic second root ignored", "user", False) not in rows
    assert ("conv-branch", "synthetic human prompt", "user", False) in rows
    assert ("conv-branch", "synthetic injected secret", "user", True) in rows
    assert ("conv-branch", "synthetic hidden reasoning", "assistant", True) in rows
    assert connection.execute(
        "SELECT outcome, is_error, ts FROM tool_calls WHERE tool_use_id='use-ok'"
    ).fetchone() == ("error", True, __import__("datetime").datetime(2026, 1, 1, 0, 0, 2))
    assert connection.execute(
        "SELECT outcome FROM tool_calls WHERE tool_use_id='use-pending'"
    ).fetchone() == ("pending",)
    assert (
        connection.execute("SELECT count(*) FROM sessions WHERE source='claude_ai'").fetchone()[0]
        == 3
    )

    before = {
        table: _dump(connection, table)
        for table in ("sessions", "events", "tool_calls", "source_files")
    }
    connection.close()
    assert main(["build", "--claude-ai-export", str(bare), "--db", str(db)]) == 0
    rendered = capsys.readouterr().out
    assert "0 imported" in rendered and "3 unchanged" in rendered
    connection = connect(db, read_only=True)
    after = {
        table: _dump(connection, table)
        for table in ("sessions", "events", "tool_calls", "source_files")
    }
    assert before == after
    connection.close()

    assert main(["grep", "synthetic human prompt", "--role", "user", "--db", str(db)]) == 0
    grep_out = capsys.readouterr().out
    assert "synthetic human prompt" in grep_out
    assert "synthetic injected secret" not in grep_out
    assert main(["topics", "conv-branch", "--db", str(db)]) == 0
    topics_out = capsys.readouterr().out
    assert "Synthetic Branch Conversation" in topics_out
    assert "synthetic injected secret" not in topics_out
    assert "synthetic hidden reasoning" not in topics_out
    assert main(["session-trace", "conv-branch", "--db", str(db)]) == 0
    capsys.readouterr()
    assert main(
        [
            "orphans",
            "--db",
            str(db),
            "--no-default-sinks",
            "--min-human-chars",
            "0",
            "--min-orphans",
            "0",
            "--min-tf",
            "1",
            "--limit",
            "0",
        ]
    ) == 0
    orphans_out = capsys.readouterr().out
    assert "session conv-branch" in orphans_out
    assert "prompt" in orphans_out
    # "hidden" is not checked: the orphans header prints "reviewed-hidden=".
    for meta_term in ("injected", "secret", "reasoning"):
        assert meta_term not in orphans_out


def test_parser_accepts_bare_and_zip(tmp_path):
    bare = tmp_path / "conversations.json"
    _write(bare, _export()[:1])
    zip_path = tmp_path / "conversations-test.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(bare, "conversations.json")
    assert (
        len(
            __import__("ashiato.claude_ai", fromlist=["parse_export"])
            .parse_export(bare)
            .conversations
        )
        == 1
    )
    assert (
        len(
            __import__("ashiato.claude_ai", fromlist=["parse_export"])
            .parse_export(zip_path)
            .conversations
        )
        == 1
    )


def test_updated_export_replaces_only_changed_conversation_and_keeps_missing(tmp_path):
    def conversation(uuid, name, updated, text):
        message = _msg(
            uuid + "-message",
            "00000000-0000-4000-8000-000000000000",
            "human",
            "2026-02-01T00:00:00Z",
            {"type": "text", "text": text},
        )
        return {
            "uuid": uuid,
            "name": name,
            "created_at": "2026-02-01T00:00:00Z",
            "updated_at": updated,
            "chat_messages": [message],
        }

    old_export = tmp_path / "old.json"
    new_export = tmp_path / "new.json"
    db = tmp_path / "incremental.duckdb"
    _write(old_export, [
        conversation("changed", "Title changed", "2026-02-01T01:00:00Z", "old changed text"),
        conversation("steady", "Title steady", "2026-02-01T01:00:00Z", "steady text"),
        conversation("removed", "Title retained", "2026-02-01T01:00:00Z", "retained text"),
    ])
    first = build([], db, claude_ai_exports=[old_export])
    assert first.claude_ai_conversations_imported == 3

    _write(new_export, [
        conversation("changed", "Title changed", "2026-02-01T02:00:00Z", "new changed text"),
        conversation("steady", "Title steady", "2026-02-01T01:00:00Z", "steady text"),
    ])
    second = build([], db, claude_ai_exports=[new_export])
    assert second.claude_ai_conversations_imported == 1
    assert second.claude_ai_conversations_unchanged == 1
    connection = connect(db, read_only=True)
    try:
        texts = {
            row[0]: row[1]
            for row in connection.execute(
                "SELECT session_id, text FROM events WHERE source='claude_ai' AND type='text'"
            ).fetchall()
        }
        assert texts == {
            "changed": "new changed text",
            "steady": "steady text",
            "removed": "retained text",
        }
        assert connection.execute(
            "SELECT count(*) FROM events WHERE session_id='changed' AND text='old changed text'"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT count(*) FROM sessions WHERE source='claude_ai'"
        ).fetchone()[0] == 3
    finally:
        connection.close()



def test_prompt_field_is_meta_and_missing_prompt_is_counted(tmp_path, capsys):
    conversation = {
        "uuid": "prompt-field-conversation",
        "name": "Synthetic Prompt Field",
        "created_at": "2026-03-01T00:00:00Z",
        "updated_at": "2026-03-01T00:01:00Z",
        "chat_messages": [
            _msg(
                "prompt-field-message",
                "00000000-0000-4000-8000-000000000000",
                "human",
                "2026-03-01T00:00:00Z",
                {"type": "text", "text": "synthetic user text"},
                {"type": "injected_prompt_block", "prompt": "synthetic prompt secret"},
                {"type": "injected_prompt_block"},
                {"type": "thinking"},
            )
        ],
    }
    export = tmp_path / "prompt-field.json"
    db = tmp_path / "prompt-field.duckdb"
    _write(export, [conversation])

    assert main(["build", "--claude-ai-export", str(export), "--db", str(db)]) == 0
    rendered = capsys.readouterr().out
    assert "injected_prompt_block:<no prompt>=1" in rendered
    assert "thinking:<no thinking>=1" in rendered

    # Counted per build, like the other block counters: an unchanged
    # conversation still reports them.
    assert main(["build", "--claude-ai-export", str(export), "--db", str(db)]) == 0
    rendered = capsys.readouterr().out
    assert "1 unchanged" in rendered
    assert "injected_prompt_block:<no prompt>=1" in rendered
    assert "thinking:<no thinking>=1" in rendered

    connection = connect(db, read_only=True)
    try:
        assert connection.execute(
            "SELECT text, is_meta FROM events WHERE text='synthetic prompt secret'"
        ).fetchone() == ("synthetic prompt secret", True)
        assert connection.execute(
            "SELECT count(*) FROM events WHERE session_id='prompt-field-conversation'"
        ).fetchone()[0] == 3
    finally:
        connection.close()

    assert main(["grep", "synthetic user text", "--role", "user", "--db", str(db)]) == 0
    grep_out = capsys.readouterr().out
    assert "synthetic user text" in grep_out
    assert "synthetic prompt secret" not in grep_out
    assert main(["topics", "prompt-field-conversation", "--db", str(db)]) == 0
    assert "synthetic prompt secret" not in capsys.readouterr().out
    assert main(
        [
            "orphans",
            "--db",
            str(db),
            "--no-default-sinks",
            "--min-human-chars",
            "0",
            "--min-orphans",
            "0",
            "--min-tf",
            "1",
            "--limit",
            "0",
        ]
    ) == 0
    assert "synthetic prompt secret" not in capsys.readouterr().out


def test_tool_kind_truncation_and_input_summary(tmp_path):
    from ashiato.parser import DEFAULT_RESULT_TEXT_LIMIT, summarize_input

    tool_blocks = [
        {
            "type": "tool_use",
            "id": "web-use",
            "name": "web_fetch",
            "integration_name": "Web Fetch",
            "input": {"url": "https://example.invalid/page"},
        },
        {
            "type": "tool_use",
            "id": "github-use",
            "name": "github:create_issue",
            "input": {"title": "Synthetic issue"},
        },
        {
            "type": "tool_use",
            "id": "drive-use",
            "name": "mcp__Google_Drive__create_file",
            "input": {"name": "synthetic.txt"},
        },
        {
            "type": "tool_use",
            "id": "long-use",
            "name": "long_result",
            "input": {"value": "long"},
        },
        {
            "type": "tool_use",
            "id": "short-use",
            "name": "short_result",
            "input": {"value": "short"},
        },
    ]
    result_blocks = [
        {
            "type": "tool_result",
            "tool_use_id": "long-use",
            "content": "x" * (DEFAULT_RESULT_TEXT_LIMIT + 10),
        },
        {"type": "tool_result", "tool_use_id": "short-use", "content": "short output"},
    ]
    conversation = {
        "uuid": "tool-fields-conversation",
        "name": "Synthetic Tool Fields",
        "created_at": "2026-03-02T00:00:00Z",
        "updated_at": "2026-03-02T00:01:00Z",
        "chat_messages": [
            _msg(
                "tool-fields-message",
                "00000000-0000-4000-8000-000000000000",
                "assistant",
                "2026-03-02T00:00:00Z",
                *tool_blocks,
            ),
            _msg(
                "tool-results-message",
                "tool-fields-message",
                "assistant",
                "2026-03-02T00:00:01Z",
                *result_blocks,
            ),
        ],
    }
    export = tmp_path / "tool-fields.json"
    db = tmp_path / "tool-fields.duckdb"
    _write(export, [conversation])
    build([], db, claude_ai_exports=[export])

    connection = connect(db, read_only=True)
    try:
        rows = {
            row[0]: row[1:]
            for row in connection.execute(
                """
                SELECT tool_use_id, tool_kind, mcp_server, input, input_summary,
                       result_text, result_truncated
                FROM tool_calls
                WHERE session_id='tool-fields-conversation'
                """
            ).fetchall()
        }
    finally:
        connection.close()

    assert rows["web-use"][:2] == ("builtin", None)
    assert rows["github-use"][:2] == ("mcp", "github")
    assert rows["drive-use"][:2] == ("mcp", "Google_Drive")
    assert rows["long-use"][-2:] == ("x" * DEFAULT_RESULT_TEXT_LIMIT, True)
    assert rows["short-use"][-2:] == ("short output", False)
    for use_id, name, input_value in (
        ("web-use", "web_fetch", {"url": "https://example.invalid/page"}),
        ("github-use", "github:create_issue", {"title": "Synthetic issue"}),
        ("drive-use", "mcp__Google_Drive__create_file", {"name": "synthetic.txt"}),
    ):
        assert rows[use_id][3] == summarize_input(name, input_value)
