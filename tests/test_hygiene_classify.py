"""Dev tests for the hygiene classification boundaries (issue #36).

The frozen ``tests/test_hygiene.py`` pins the end-to-end counts through the
CLI; this module pins the classification function itself, so the shell
invocation boundaries live where the spec put them -- in one testable module,
not scattered through rendering.

The long-command tests pin the full-input repair: shell categories classify
from the full persisted ``input`` command, falling back to the 200-character
``input_summary`` only when the full command cannot be extracted, so a signal
past the summary truncation boundary still counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ashiato.build import connect
from ashiato.cli import main
from ashiato.hygiene import _command_tokens, _shell_tokens, audit, categories_for


def test_shell_tokenizer_resolves_quotes_without_splitting() -> None:
    assert _shell_tokens('echo "usage: kusabi-companion status"') == [
        "echo",
        "usage: kusabi-companion status",
    ]
    assert _shell_tokens("sed -n '1,40p' /etc/hosts") == ["sed", "-n", "1,40p", "/etc/hosts"]
    assert _shell_tokens("") == []
    assert _shell_tokens(None) == []


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("kusabi-companion status", {"companion_status_poll"}),
        ("kusabi-companion status --json", {"companion_status_poll"}),
        ("/usr/local/bin/kusabi-companion status", {"companion_status_poll"}),
        ("kusabi-companion chain-show 123", set()),
        ("kusabi-companion chain-wait 123", set()),
        ("kusabi-companion", set()),
        ('echo "usage: kusabi-companion status"', set()),
    ],
)
def test_companion_status_poll_boundary(command: str, expected: set[str]) -> None:
    assert set(categories_for("Bash", command, "ok")) == expected


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('rg "TODO" /home/dev/proj', {"host_file_hunt"}),
        ('grep -rn "api_key" /etc', {"host_file_hunt"}),
        ("sed -n '1,40p' /etc/hosts", {"host_file_hunt"}),
        ("cat /etc/hosts", {"host_file_hunt"}),
        ("ls -la /home/dev/proj", set()),
        ("echo 'cat /etc/hosts'", set()),
    ],
)
def test_host_file_hunt_boundary(command: str, expected: set[str]) -> None:
    assert set(categories_for("Bash", command, "ok")) == expected


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("curl -s http://127.0.0.1:8750/mcp", {"raw_local_mcp_http"}),
        ("curl http://localhost:8765/health", {"raw_local_mcp_http"}),
        ("curl -v http://127.0.0.1:8770/", {"raw_local_mcp_http"}),
        ("curl http://127.0.0.1:9000/api", set()),
        ("curl https://api.github.com/repos/x", set()),
        ("curl http://example.com:8750/x", set()),
        ("wget http://localhost:8750/x", set()),
    ],
)
def test_raw_local_mcp_http_boundary(command: str, expected: set[str]) -> None:
    assert set(categories_for("Bash", command, "ok")) == expected


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # finding 2: a loopback URL used only as an option value is not the
        # request target
        ("curl -H 'http://localhost:8750/' https://api.github.com", set()),
        ("curl --header 'http://localhost:8750/' https://api.github.com", set()),
        ("curl -H'http://localhost:8750/' https://api.github.com", set()),
        ("curl --header=http://localhost:8750/ https://api.github.com", set()),
        ("curl -d 'http://localhost:8765/x' https://api.github.com", set()),
        # the actual loopback request target still classifies
        ("curl -H 'X: 1' http://127.0.0.1:8750/mcp", {"raw_local_mcp_http"}),
        ("curl --url http://127.0.0.1:8750/mcp", {"raw_local_mcp_http"}),
        ("curl --url=http://127.0.0.1:8750/mcp", {"raw_local_mcp_http"}),
        ("curl -d 'http://localhost:8765/x' http://127.0.0.1:8770/", {"raw_local_mcp_http"}),
        ("curl -s http://127.0.0.1:8750/mcp", {"raw_local_mcp_http"}),
    ],
)
def test_raw_local_mcp_http_classifies_the_request_target(command: str, expected: set[str]) -> None:
    assert set(categories_for("Bash", command, "ok")) == expected


def test_dedicated_tools_are_never_shell_categories() -> None:
    assert categories_for("Grep", '{"pattern":"cat"}', "ok") == ()
    assert categories_for("Read", "/etc/hosts", "ok") == ()
    assert categories_for("mcp__sunaba__search", '{"pattern":"sed"}', "ok") == ()
    assert categories_for("mcp__sunaba__http_fetch", '{"url":"http://127.0.0.1:8750/sse"}', "ok") == ()


def test_undo_file_edit_matches_mcp_tool_names_only() -> None:
    assert categories_for("mcp__sunaba__undo_file_edit", "", "ok") == ("undo_file_edit",)
    assert categories_for("mcp__kaiba__undo_file_edit", "", "ok") == ("undo_file_edit",)
    assert categories_for("Bash", 'echo "the undo_file_edit tool reverses edits"', "ok") == ()
    assert categories_for("mcp__sunaba__search", "", "ok") == ()


def test_pending_is_independent_of_tool_name() -> None:
    assert categories_for("mcp__foo__bar", '{"whatever":1}', "pending") == ("pending_tool_call",)
    assert categories_for("Bash", "curl http://localhost:8750/whatever", "pending") == (
        "raw_local_mcp_http",
        "pending_tool_call",
    )
    assert categories_for("Bash", "kusabi-companion status", "pending") == (
        "companion_status_poll",
        "pending_tool_call",
    )
    assert categories_for("Bash", "echo done", "ok") == ()


def test_classification_never_reads_result_text() -> None:
    """Result text is not an input to categories_for: quoted hunt/poll words in
    a result cannot manufacture a category."""
    assert categories_for("Bash", "cat /tmp/notes.md", "ok") == ("host_file_hunt",)
    assert categories_for("Bash", "ls -la /home/dev/proj", "ok") == ()


# ---------------------------------------------------------------- full input


def test_command_tokens_prefers_full_command_and_falls_back_conservatively() -> None:
    """The full persisted command wins; a NULL/blank/unusable one falls back
    to the summary rather than being discarded -- but arbitrary objects,
    numbers, or non-string arrays are never turned into commands."""
    assert _command_tokens("kusabi-companion status", "VARCHAR", "x") == ["kusabi-companion", "status"]
    assert _command_tokens(None, None, "cat /etc/hosts") == ["cat", "/etc/hosts"]
    assert _command_tokens("", "VARCHAR", "cat /etc/hosts") == ["cat", "/etc/hosts"]
    assert _command_tokens("   ", "VARCHAR", "cat /etc/hosts") == ["cat", "/etc/hosts"]
    assert _command_tokens(None, None, None) == []
    # an OBJECT or a number is not a command -> summary fallback
    assert _command_tokens('{"script":"x"}', "OBJECT", "cat /etc/hosts") == ["cat", "/etc/hosts"]
    assert _command_tokens("123", "UBIGINT", "cat /etc/hosts") == ["cat", "/etc/hosts"]
    # a JSON array of non-strings is not a usable argv -> summary fallback
    assert _command_tokens('["cat", 123]', "ARRAY", "cat /etc/hosts") == ["cat", "/etc/hosts"]


def test_argv_list_commands_decode_to_actual_argv() -> None:
    """Codex persists ``input.command`` as a JSON argv list; decode it and
    classify the actual argv -- a list is used directly, not re-tokenized."""
    assert _command_tokens('["cat", "/etc/hosts"]', "ARRAY", "") == ["cat", "/etc/hosts"]
    assert categories_for("Bash", ["cat", "/etc/hosts"], "ok") == ("host_file_hunt",)
    assert categories_for("Bash", ["kusabi-companion", "status"], "ok") == ("companion_status_poll",)
    assert categories_for("Bash", ["curl", "http://127.0.0.1:8750/mcp"], "ok") == ("raw_local_mcp_http",)
    # prose inside an argv element is still prose, not an invocation
    assert categories_for("Bash", ["echo", "kusabi-companion status"], "ok") == ()


def test_long_curl_url_beyond_summary_truncation_counts() -> None:
    """A loopback MCP URL that sits past the 200-char input_summary boundary
    must still classify from the full command (this fails against a summary
    truncated at 200 characters)."""
    command = "curl -s -H 'X-Pad: " + "p" * 240 + "' http://127.0.0.1:8750/mcp"
    assert len(command) > 200
    truncated = command[:200]
    assert "8750" not in truncated  # the URL is past the summary boundary
    assert "raw_local_mcp_http" in categories_for("Bash", command, "ok")
    assert "raw_local_mcp_http" not in categories_for("Bash", truncated, "ok")
    assert _command_tokens(command, "VARCHAR", truncated) == _shell_tokens(command)


def test_long_companion_status_beyond_summary_truncation_counts() -> None:
    """A kusabi-companion status invocation whose subcommand lands past the
    200-char summary boundary must not silently disappear."""
    command = "kusabi-companion --" + "x" * 240 + " status"
    assert len(command) > 200
    truncated = command[:200]
    assert "status" not in truncated.split()
    assert "companion_status_poll" in categories_for("Bash", command, "ok")
    assert "companion_status_poll" not in categories_for("Bash", truncated, "ok")
    assert _command_tokens(command, "VARCHAR", truncated) == _shell_tokens(command)


# ---------------------------------------------------------------- end to end


def _bash_call(session_id: str, command: str | list[str], directory: Path) -> None:
    """One Bash tool call (with a result) as a Claude Code transcript file.

    ``command`` is persisted as-is in ``input.command`` -- a string for the
    Claude form, a list for the Codex argv form.
    """
    tool_use_id = f"{session_id}_use"
    call_uuid = f"{session_id}_call"
    result_uuid = f"{session_id}_result"
    records = [
        {
            "type": "assistant",
            "uuid": call_uuid,
            "parentUuid": None,
            "sessionId": session_id,
            "timestamp": "2026-08-21T10:00:00Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": tool_use_id, "name": "Bash", "input": {"command": command}}],
            },
        },
        {
            "type": "user",
            "uuid": result_uuid,
            "parentUuid": call_uuid,
            "sessionId": session_id,
            "timestamp": "2026-08-21T10:00:00Z",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": [{"type": "text", "text": "ok"}],
                        "is_error": False,
                    }
                ],
            },
        },
    ]
    path = directory / f"{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_long_commands_classify_from_full_persisted_input(tmp_path: Path) -> None:
    """End to end: two >200-char commands whose signal sits past the summary
    truncation must each count once through the real build pipeline and the
    audit report."""
    source = tmp_path / "source"
    source.mkdir()
    long_curl = "curl -s -H 'X-Pad: " + "p" * 240 + "' http://127.0.0.1:8750/mcp"
    long_companion = "kusabi-companion --" + "x" * 240 + " status"
    _bash_call("ses-curl", long_curl, source)
    _bash_call("ses-comp", long_companion, source)

    db_path = tmp_path / "long.duckdb"
    assert main(["build", "--source", str(source), "--db", str(db_path)]) == 0
    connection = connect(db_path, read_only=True)
    try:
        report = audit(connection)
    finally:
        connection.close()

    counts = {cat["name"]: cat for cat in report["categories"]}
    assert counts["raw_local_mcp_http"]["tool_calls"] == 1
    assert counts["companion_status_poll"]["tool_calls"] == 1
    assert report["coverage"]["tool_calls"] == 2


def test_argv_list_commands_classify_end_to_end(tmp_path: Path) -> None:
    """Codex-style persisted argv lists classify correctly end to end: one cat
    hunt, one companion poll, one loopback curl -- each from a JSON array
    ``input.command`` through the real build pipeline and the audit report."""
    source = tmp_path / "source"
    source.mkdir()
    _bash_call("ses-cat", ["cat", "/etc/hosts"], source)
    _bash_call("ses-poll", ["kusabi-companion", "status"], source)
    _bash_call("ses-curl", ["curl", "http://127.0.0.1:8750/mcp"], source)

    db_path = tmp_path / "argv.duckdb"
    assert main(["build", "--source", str(source), "--db", str(db_path)]) == 0
    connection = connect(db_path, read_only=True)
    try:
        report = audit(connection)
    finally:
        connection.close()

    counts = {cat["name"]: cat for cat in report["categories"]}
    assert counts["host_file_hunt"]["tool_calls"] == 1
    assert counts["companion_status_poll"]["tool_calls"] == 1
    assert counts["raw_local_mcp_http"]["tool_calls"] == 1
    assert report["coverage"]["tool_calls"] == 3