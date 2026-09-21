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
from ashiato.hygiene import (
    _SEGMENT_SEP,
    _command_tokens,
    _shell_tokens,
    audit,
    categories_for,
)


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


# ------------------------------------------------------- shell -c wrapper


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        # the exact persisted Codex shape: /bin/bash -lc SCRIPT
        (["/bin/bash", "-lc", "kusabi-companion status"], {"companion_status_poll"}),
        (["/bin/bash", "-lc", "cat /etc/hosts"], {"host_file_hunt"}),
        (["/bin/bash", "-lc", "rg TODO /home/dev/proj"], {"host_file_hunt"}),
        (["/bin/bash", "-lc", "curl -s http://127.0.0.1:8750/mcp"], {"raw_local_mcp_http"}),
        # bash and sh, -c and -lc, with and without the /bin path prefix
        (["bash", "-c", "kusabi-companion status"], {"companion_status_poll"}),
        (["bash", "-lc", "cat /etc/hosts"], {"host_file_hunt"}),
        (["sh", "-c", "kusabi-companion status"], {"companion_status_poll"}),
        (["/bin/sh", "-lc", "cat /etc/hosts"], {"host_file_hunt"}),
        (["/bin/bash", "-c", "curl http://localhost:8765/health"], {"raw_local_mcp_http"}),
    ],
)
def test_shell_c_wrapper_unwraps_exact_forms(argv: list[str], expected: set[str]) -> None:
    assert set(categories_for("Bash", argv, "ok")) == expected


def test_shell_c_wrapper_string_commands_unwrap_too() -> None:
    """The wrapper is recognized in the string command form too, where the
    conservative tokenizer produces the same argv shape."""
    assert categories_for("Bash", 'bash -lc "kusabi-companion status"', "ok") == ("companion_status_poll",)
    assert categories_for("Bash", "sh -c 'cat /etc/hosts'", "ok") == ("host_file_hunt",)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        # missing -c: not a wrapper, and bash is not a hunted/curl/poll program
        (["bash", "kusabi-companion", "status"], set()),
        # unsupported flags before the wrapper flag: the exact form is absent
        (["bash", "-x", "-c", "kusabi-companion status"], set()),
        (["bash", "--noprofile", "-lc", "cat /etc/hosts"], set()),
        # quoted output: echo is still the executed command, not the signal
        (["bash", "-c", "echo 'kusabi-companion status'"], set()),
        (["bash", "-c", 'echo "cat /etc/hosts"'], set()),
        # arbitrary and nested wrappers are not descended into
        (["env", "bash", "-c", "kusabi-companion status"], set()),
        (["bash", "-c", "sh", "-c", "kusabi-companion status"], set()),
        # a wrapper program that is not bash/sh is not unwrapped
        (["zsh", "-c", "kusabi-companion status"], set()),
        (["python", "-c", "print('cat /etc/hosts')"], set()),
        # a script split across argv elements is not the single-argument form
        (["bash", "-c", "cat", "/etc/hosts"], set()),
        (["bash", "-lc", "kusabi-companion", "status"], set()),
    ],
)
def test_shell_c_wrapper_near_misses_do_not_classify(argv: list[str], expected: set[str]) -> None:
    assert set(categories_for("Bash", argv, "ok")) == expected


def test_shell_c_wrapper_script_splits_on_separators_and_not_on_ampersand() -> None:
    """Inside an unwrapped bash -c script, ``;`` splits into segments that are
    classified independently, while ``&&`` is not a separator (only the
    ``cd <dir> &&`` prefix rule applies)."""
    assert categories_for(
        "Bash",
        ["bash", "-c", "cat /etc/hosts && curl http://127.0.0.1:8750/mcp"],
        "ok",
    ) == ("host_file_hunt",)
    assert categories_for("Bash", ["bash", "-c", "echo hi; kusabi-companion status"], "ok") == (
        "companion_status_poll",
    )
    assert categories_for("Bash", ["bash", "-c", "curl http://127.0.0.1:8750/mcp && cat /etc/hosts"], "ok") == (
        "raw_local_mcp_http",
    )


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


# ------------------------------------------------------- cd <dir> && prefix


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # the persisted form this machine's sessions almost always use
        ("cd /x && cat /etc/hosts", {"host_file_hunt"}),
        ("cd /k && kusabi-companion status", {"companion_status_poll"}),
        ("cd /repo && curl -s http://127.0.0.1:8750/mcp", {"raw_local_mcp_http"}),
        # the prefix is stripped repeatedly
        ("cd /a && cd /b && grep -rn x /home", {"host_file_hunt"}),
        ("cd /x && cd /y && rg TODO /home", {"host_file_hunt"}),
        # the prefix is stripped from the script of an unwrapped -c wrapper
        ("cd /x && node plugins/kusabi/scripts/kusabi-companion.mjs status", {"companion_status_poll"}),
    ],
)
def test_cd_prefix_is_stripped_before_the_rules(command: str, expected: set[str]) -> None:
    assert set(categories_for("Bash", command, "ok")) == expected


def test_cd_prefix_is_stripped_after_wrapper_unwrap() -> None:
    """The strip applies to the *script* of an unwrapped bash -c wrapper, not
    only to a bare command line."""
    assert categories_for("Bash", ["/bin/bash", "-lc", "cd /x && kusabi-companion status"], "ok") == (
        "companion_status_poll",
    )
    assert categories_for("Bash", ["/bin/bash", "-lc", "cd /x && cat /etc/hosts"], "ok") == (
        "host_file_hunt",
    )


@pytest.mark.parametrize(
    "command",
    [
        # not a `cd <dir> &&` prefix: different separator, flag, assignment,
        # subshell, or a bare cd -- none of them descend into a compound form
        "cd /x | cat /etc/hosts",
        "cd -P /x && cat /etc/hosts",
        "FOO=1 cat /etc/hosts",
        "(cd /x && cat /etc/hosts)",
        "cd /x",
        # quoting immunity: prose that merely quotes the command is never a call
        'echo "cd /x && cat /etc/hosts"',
        "echo 'kusabi-companion status'",
        'echo "cat /etc/hosts"',
    ],
)
def test_cd_prefix_near_misses_do_not_classify(command: str) -> None:
    assert categories_for("Bash", command, "ok") == ()


# ------------------------------------------------------- node <script> companion


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # the form actually persisted by Claude Code sessions, relative and
        # absolute script path, node and nodejs, flags around the subcommand
        ("node plugins/kusabi/scripts/kusabi-companion.mjs status", {"companion_status_poll"}),
        ("node /abs/path/kusabi-companion.mjs status --json", {"companion_status_poll"}),
        ("nodejs kusabi-companion.mjs status", {"companion_status_poll"}),
        ("cd /k && node plugins/kusabi/scripts/kusabi-companion.mjs status", {"companion_status_poll"}),
        # the .mjs script invoked directly also counts
        ("kusabi-companion.mjs status", {"companion_status_poll"}),
    ],
)
def test_node_companion_form_recognises_status(command: str, expected: set[str]) -> None:
    assert set(categories_for("Bash", command, "ok")) == expected


@pytest.mark.parametrize(
    "command",
    [
        "node /abs/path/kusabi-companion.mjs chain-wait 123",
        "node kusabi-companion.mjs chain-show 123",
        "node kusabi-companion.mjs result",
        "node kusabi-companion.mjs",
        "node /abs/path/kusabi-companion.mjs",
        # a different script (or -e code) is not a companion invocation
        "node /abs/path/other-script.mjs status",
        "node -e 'kusabi-companion status'",
        # the bare binary rules still apply unchanged
        "kusabi-companion chain-wait 123",
        "kusabi-companion",
        "kusabi-companion.mjs result",
    ],
)
def test_node_companion_form_near_misses_do_not_classify(command: str) -> None:
    assert categories_for("Bash", command, "ok") == ()


# ------------------------------------------------------- shell tool names


@pytest.mark.parametrize(
    "tool_name",
    ["Bash", "PowerShell", "Shell", "bash", "BASH", "POWERSHELL", "shell"],
)
def test_shell_tool_names_match_case_insensitively(tool_name: str) -> None:
    """Cursor's `Shell` tool and opencode's lowercase `bash` both count as
    shell tools; the membership test is case-insensitive."""
    assert categories_for(tool_name, "cat /etc/hosts", "ok") == ("host_file_hunt",)


def test_shell_tool_names_quoted_prose_is_still_not_a_call() -> None:
    assert categories_for("Shell", 'echo "cat /etc/hosts"', "ok") == ()
    assert categories_for("bash", 'echo "kusabi-companion status"', "ok") == ()
    assert categories_for("Shell", "ls -la /home/dev/proj", "ok") == ()


def test_shell_tool_names_undo_file_edit_stays_case_sensitive() -> None:
    """Only the shell-tool membership test became case-insensitive; the MCP
    undo_file_edit tool-name match keeps its case-sensitive behaviour."""
    assert categories_for("mcp__sunaba__undo_file_edit", "", "ok") == ("undo_file_edit",)
    assert categories_for("MCP__SUNABA__UNDO_FILE_EDIT", "", "ok") == ()


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


def test_codex_shell_wrapper_argv_classify_end_to_end(tmp_path: Path) -> None:
    """The exact persisted Codex shape -- ``input.command`` as a JSON argv
    list wrapped in ``/bin/bash -lc`` -- classifies through the real build
    pipeline and the audit report: one companion poll, one cat hunt, one
    loopback curl."""
    source = tmp_path / "source"
    source.mkdir()
    _bash_call("ses-poll", ["/bin/bash", "-lc", "kusabi-companion status"], source)
    _bash_call("ses-hunt", ["/bin/bash", "-lc", "cat /etc/hosts"], source)
    _bash_call("ses-curl", ["/bin/bash", "-lc", "curl -s http://127.0.0.1:8750/mcp"], source)

    db_path = tmp_path / "wrapper.duckdb"
    assert main(["build", "--source", str(source), "--db", str(db_path)]) == 0
    connection = connect(db_path, read_only=True)
    try:
        report = audit(connection)
    finally:
        connection.close()

    counts = {cat["name"]: cat for cat in report["categories"]}
    assert counts["companion_status_poll"]["tool_calls"] == 1
    assert counts["host_file_hunt"]["tool_calls"] == 1
    assert counts["raw_local_mcp_http"]["tool_calls"] == 1
    assert report["coverage"]["tool_calls"] == 3


# --------------------------------------------------- segment separators


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # the most frequent measured shape: a sleep, then the poll after a semicolon
        (
            "sleep 90; cd /k && node plugins/kusabi/scripts/kusabi-companion.mjs status",
            {"companion_status_poll"},
        ),
        # the poll is the third segment, not the first
        ("ls -la; command -v kusabi-companion; kusabi-companion status", {"companion_status_poll"}),
        # inside an unwrapped wrapper script, semicolons split too
        (["bash", "-c", "echo hi; kusabi-companion status"], {"companion_status_poll"}),
        # one row, one category, not two -- two hunt segments are still one host_file_hunt
        ("cat /etc/hosts; cat /etc/passwd", {"host_file_hunt"}),
        # a trailing or doubled separator is harmless
        ("cat /etc/hosts;;", {"host_file_hunt"}),
        ("cat /etc/hosts;", {"host_file_hunt"}),
        # semicolons also split after the wrapper unwrap
        (["/bin/bash", "-lc", "cd /x && cat /etc/hosts; curl http://127.0.0.1:8750/mcp"],
         {"host_file_hunt", "raw_local_mcp_http"}),
    ],
)
def test_semicolon_separator(command: str | list[str], expected: set[str]) -> None:
    assert set(categories_for("Bash", command, "ok")) == expected


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # a raw newline separates segments (Cursor multi-line scripts)
        ("cd /k\nkusabi-companion status\necho done", {"companion_status_poll"}),
        # multiple newlines
        ("cat /etc/hosts\ncat /etc/passwd\n", {"host_file_hunt"}),
        # newlines inside the unwrapped wrapper script
        (["bash", "-c", "echo hi\nkusabi-companion status"], {"companion_status_poll"}),
    ],
)
def test_newline_separator(command: str | list[str], expected: set[str]) -> None:
    assert set(categories_for("Bash", command, "ok")) == expected


@pytest.mark.parametrize(
    "command",
    [
        # semicolons inside double quotes are not separators
        'echo "a; cat /etc/hosts"',
        'echo "sleep 90; kusabi-companion status"',
        # semicolons inside single quotes are not separators
        "echo 'a; cat /etc/hosts'",
        "echo 'sleep 90; kusabi-companion status'",
        # newlines inside single quotes are not separators
        "echo 'a\ncat /etc/hosts'",
        # newlines inside double quotes are not separators
        'echo "a\ncat /etc/hosts"',
        # argv elements are never re-split
        ["echo", "a; cat /etc/hosts"],
        ["echo", "kusabi-companion status"],
    ],
)
def test_separator_quoting(command: str | list[str]) -> None:
    """Quoting and argv elements are never split on separators."""
    assert categories_for("Bash", command, "ok") == ()


def test_section_b_shapes_keep_not_classifying() -> None:
    """Shapes from the real corpus that the segment change must leave alone."""
    # VAR=value prefix is not stripped
    assert categories_for("Bash", "FOO=1 cat /etc/hosts", "ok") == ()
    # variable indirection -- the assignment is a segment, the program is $K
    assert categories_for("Bash", "K=/path/to/kusabi-companion.mjs", "ok") == ()
    # loop keywords as first token are not programs
    assert categories_for("Bash", "while true; do cat /etc/hosts; done", "ok") == ()
    # pipes are not separators -- only the first command counts
    assert categories_for("Bash", "cat /etc/hosts | grep x", "ok") == ("host_file_hunt",)
    # && is not a separator (only cd <dir> && is special)
    assert categories_for("Bash", "cmd && cmd", "ok") == ()
    # subshell is not a program
    assert categories_for("Bash", "(cd /x && cat /etc/hosts)", "ok") == ()
    # dedicated tools are never shell categories
    assert categories_for("Grep", "cat /etc/hosts", "ok") == ()


def test_nul_in_a_command_string_cannot_forge_a_separator() -> None:
    """A NUL in persisted command *text* must not act as a separator.

    The separator is a NUL sentinel, so the tokenizer drops NULs from its input:
    a string that carries one classifies exactly as it would without it.  A list
    argument is deliberately not filtered -- it is either a real argv (which
    cannot contain NUL) or the already-tokenized output of
    :func:`_command_tokens`, whose sentinels are the separators and must
    survive; filtering them once broke every real ``;`` row while every
    string-level test stayed green.
    """
    assert categories_for("Bash", "echo hi \x00 cat /etc/hosts", "ok") == ()
    assert categories_for("Bash", "cat /etc/hosts \x00 echo hi", "ok") == ("host_file_hunt",)
    # a command that is nothing but NULs is an empty command
    assert categories_for("Bash", "\x00\x00", "ok") == ()


def test_command_tokens_separators_survive_into_categories_for() -> None:
    """The real pipeline passes :func:`_command_tokens` output (a list) into
    :func:`categories_for`, so the sentinels it produced must still split."""
    tokens = _command_tokens(
        "sleep 90; cd /k && node plugins/kusabi/scripts/kusabi-companion.mjs status 2>&1 | head -3",
        "VARCHAR",
        None,
    )
    assert _SEGMENT_SEP in tokens
    assert categories_for("Bash", tokens, "ok") == ("companion_status_poll",)


def test_separator_rows_classify_end_to_end_through_audit(tmp_path: Path) -> None:
    """End to end: the real pipeline reaches ``categories_for`` with the token
    list from :func:`_command_tokens`, not with a string.

    Every string-level separator test can pass while this layer is broken -- it
    happened: filtering the tokenizer's sentinels out of a list argument left
    every real ``;`` row unclassified with the whole suite green.  These are the
    two most frequent measured shapes from the real corpus.
    """
    source = tmp_path / "source"
    source.mkdir()
    _bash_call(
        "ses-sleep",
        "sleep 90; cd ~/dev/projects/kusabi && "
        "node plugins/kusabi/scripts/kusabi-companion.mjs status 2>&1 | head -15",
        source,
    )
    _bash_call("ses-seq", "ls -la; echo ===; cat /etc/hosts", source)
    _bash_call("ses-newline", "cd /k\nkusabi-companion status\necho done", source)

    db_path = tmp_path / "separators.duckdb"
    assert main(["build", "--source", str(source), "--db", str(db_path)]) == 0
    connection = connect(db_path, read_only=True)
    try:
        report = audit(connection)
    finally:
        connection.close()

    counts = {cat["name"]: cat for cat in report["categories"]}
    assert counts["companion_status_poll"]["tool_calls"] == 2
    assert counts["host_file_hunt"]["tool_calls"] == 1
    assert report["coverage"]["tool_calls"] == 3
