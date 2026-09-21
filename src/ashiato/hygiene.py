"""Named session-hygiene audit (issue #36).

The five categories replace ad-hoc SQL that used to be rewritten by hand for
every session-hygiene question:

* ``companion_status_poll`` -- shell calls whose *executed command* invokes
  ``kusabi-companion status``: the binary itself (basename ``kusabi-companion``
  or ``kusabi-companion.mjs``) or its ``node <script>`` form
  (``node .../kusabi-companion.mjs status``).  Other companion subcommands
  (``chain-show``, ``chain-wait``, ``result``), the bare binary or bare
  script, and text that merely *quotes* the command -- in a tool result, in an
  ``echo`` argument, or in a ``Read`` of a doc -- are not polls.
* ``host_file_hunt`` -- shell calls that run ``rg``/``grep``/``sed``/``cat``
  against host files.  A call whose persisted tool name is a dedicated
  file/search tool (``Grep``, ``Read``, an MCP search tool) is excluded, and
  hunt words that appear only in a tool result are not a hunt.
* ``raw_local_mcp_http`` -- shell calls that ``curl`` loopback
  (``127.0.0.1``/``localhost``) ports 8750/8765/8770.  Other ports, remote
  hosts (even on a matching port), and dedicated MCP tool calls are excluded.
* ``undo_file_edit`` -- tool calls whose persisted ``tool_name`` is an MCP
  ``undo_file_edit`` tool on any server (``mcp__<server>__undo_file_edit``).
  Prose that merely mentions the name is not a call.
* ``pending_tool_call`` -- every row with ``outcome = 'pending'``, whatever
  its tool name.

Every classification reads the *persisted* ``tool_name`` and the command only
-- never ``result_text``: what a tool returned is evidence about the tool, not
about what was asked for.  The shell categories apply to the persisted
shell-tool names ``Bash`` / ``PowerShell`` / ``Shell``, matched
case-insensitively so ``bash`` counts too (the MCP ``undo_file_edit`` name
match stays case-sensitive).  The command is the full persisted ``input`` command
field when it can be used, with the 200-character ``input_summary`` kept only
as a conservative fallback, so a long command whose signal sits past the
summary truncation boundary still classifies.  A persisted command may be a
string (shell-tokenized by :func:`_shell_tokens`) or a JSON argv list -- the
Codex ``input.command`` form -- which decodes to its actual argv without
turning arbitrary prose or objects into commands.  :func:`_shell_tokens` is a
deliberately small tokenizer that resolves quotes but not compound forms
(``&&``, pipes), so classification is conservative: only the first command of
a line counts as the executed command.  Two exceptions are deliberately
narrow.  The exact shell wrapper form ``bash -c SCRIPT`` / ``bash -lc
SCRIPT`` (and the ``/bin/bash`` / ``sh`` / ``/bin/sh`` equivalents) is
unwrapped -- SCRIPT is re-tokenized with the same conservative tokenizer and
then classified.  And the ``cd <dir> &&`` prefix that almost every command
persisted by this machine's sessions carries is stripped: while the token
list starts with exactly ``<program whose basename is 'cd'>, <one token that
does not start with '-'>, '&&'``, those three tokens are dropped and the
check repeats, so ``cd /x && cat /etc/hosts`` classifies the ``cat`` and
``cd /x && cd /y && cmd`` strips twice.  No other compound form is descended
into -- pipes, ``;``, ``||``, subshells, command substitution, and
``VAR=value`` prefixes are not traversed -- and an argv element that merely
mentions a signal never classifies unless it is the script argument of one
of those exact wrapper forms or follows the ``cd <dir> &&`` prefix.

``raw_local_mcp_http`` classifies the curl *request target*, not any
URL-shaped option argument: common curl options that consume a following
value (``-H``/``--header``, ``-d``/``--data*``, ``-F``/``--form``, ``--url``,
...) have their value handled, so ``curl -H 'http://localhost:8750/'
https://api.github.com`` is not a raw MCP call while the actual loopback
target still is.  This is a conservative option-value skip, not a full curl
parser.

The report is computed by :func:`audit` over one bounded, read-only query and
returned as plain data (the CLI renders it).  Categories may overlap, and the
``coverage`` block counts every selected row and distinct session *before*
category filtering and carries per-source ``sources``, so a windowed report
discloses which source contributed which rows.  With either bound given, the
``coverage`` block also discloses ``excluded_no_timestamp`` -- the rows whose
``ts`` is NULL that the window dropped (the source records no per-call
timestamp); without bounds nothing is excluded, because NULL-timestamp rows
count.  This module never mutates the database.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import duckdb

#: Report order; also the order of the ``categories`` list in the JSON output.
CATEGORY_ORDER: tuple[str, ...] = (
    "companion_status_poll",
    "host_file_hunt",
    "raw_local_mcp_http",
    "undo_file_edit",
    "pending_tool_call",
)

#: Persisted tool names whose ``input`` carries an executed shell command.
#: Membership is compared case-insensitively (``bash`` counts); the MCP
#: ``undo_file_edit`` tool-name match below stays case-sensitive.
_SHELL_TOOLS: frozenset[str] = frozenset({"Bash", "PowerShell", "Shell"})

#: Lower-cased shell tool names for the case-insensitive membership test.
_SHELL_TOOLS_LOWER: frozenset[str] = frozenset(name.lower() for name in _SHELL_TOOLS)

_COMPANION_BIN = "kusabi-companion"
_COMPANION_SUBCOMMAND = "status"

#: Programs that inspect host files; the *executed* program must be one of these.
_HUNT_PROGRAMS: frozenset[str] = frozenset({"rg", "grep", "sed", "cat"})

_CURL_BIN = "curl"
_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost"})
_MCP_PORTS: frozenset[int] = frozenset({8750, 8765, 8770})

#: Common curl short options that consume a following value (or an attached
#: one in the rest of the token).  A conservative list: missing an option here
#: only means a URL-shaped value could be mistaken for a target; the listed
#: ones cover the options most likely to carry a URL (headers, data, forms,
#: output, proxies, auth, timing).  ``--url`` is handled separately because its
#: value *is* the request target.
_CURL_VALUE_OPTIONS_SHORT: frozenset[str] = frozenset(
    {
        "A", "b", "c", "C", "d", "D", "e", "E", "F", "H", "m",
        "o", "r", "T", "u", "U", "w", "x", "X", "y", "Y", "z",
    }
)

_CURL_VALUE_OPTIONS_LONG: frozenset[str] = frozenset(
    {
        "cacert",
        "cert",
        "connect-timeout",
        "connect-to",
        "continue-at",
        "cookie",
        "cookie-jar",
        "data",
        "data-ascii",
        "data-binary",
        "data-raw",
        "data-urlencode",
        "dump-header",
        "form",
        "header",
        "key",
        "limit-rate",
        "max-time",
        "output",
        "pass",
        "proxy",
        "proxy-user",
        "range",
        "referer",
        "request",
        "resolve",
        "retry",
        "speed-limit",
        "speed-time",
        "time-cond",
        "upload-file",
        "url",
        "user",
        "user-agent",
        "write-out",
    }
)

#: An MCP undo call on any server, e.g. ``mcp__sunaba__undo_file_edit``.
_UNDO_FILE_EDIT_RE = re.compile(r"^mcp__.+__undo_file_edit$")


def _shell_tokens(command: str | None) -> list[str]:
    """Split a shell command line into argv-like tokens.

    Single and double quotes (with backslash escapes inside double quotes and
    outside them) are resolved so that a hunt/poll word inside a quoted
    argument -- ``echo "usage: kusabi-companion status"`` -- stays part of one
    token instead of looking like an executed command.  Compound forms are not
    resolved: only the first command of a line is considered executed, which
    keeps the boundary honest where a full shell parser would be overkill.
    """
    if not command:
        return []
    tokens: list[str] = []
    current: list[str] = []
    quote: str | None = None
    index = 0
    length = len(command)
    while index < length:
        char = command[index]
        if quote is not None:
            if char == quote:
                quote = None
            elif char == "\\" and index + 1 < length:
                current.append(command[index + 1])
                index += 1
            else:
                current.append(char)
        elif char in ("'", '"'):
            quote = char
        elif char == "\\" and index + 1 < length:
            current.append(command[index + 1])
            index += 1
        elif char.isspace():
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(char)
        index += 1
    if current:
        tokens.append("".join(current))
    return tokens


#: Shell programs whose exact ``-c``/``-lc`` wrapper is unwrapped before the
#: category rules run: ``bash -c SCRIPT``, ``bash -lc SCRIPT``, and the
#: ``/bin/bash`` / ``sh`` / ``/bin/sh`` equivalents.  The wrapper is the
#: executed program only in the mechanical sense; the script argument is the
#: command being classified.
_WRAPPER_PROGRAMS: frozenset[str] = frozenset({"bash", "sh"})

#: The exact wrapper flags whose single following argument is the script.
_WRAPPER_FLAGS: frozenset[str] = frozenset({"-c", "-lc"})


def _unwrap_shell_wrapper(tokens: list[str]) -> list[str]:
    """The executed command when *tokens* is a direct shell ``-c`` wrapper.

    Only the exact three-argument forms ``bash -c SCRIPT``, ``bash -lc
    SCRIPT`` (and the ``/bin/bash`` / ``sh`` / ``/bin/sh`` equivalents) are
    unwrapped: the script argument is re-tokenized with the conservative
    tokenizer, so compound commands inside SCRIPT still follow the
    first-command-only boundary and quoted text stays inside one token.
    Everything else -- a missing ``-c``, extra flags, a wrapper around a
    wrapper, a non-``bash``/``sh`` program, an argv element that merely
    mentions a command word -- is returned unchanged, so the ordinary
    first-command rules apply and no signal is invented.
    """
    if len(tokens) != 3:
        return tokens
    if not all(isinstance(token, str) for token in tokens):
        return tokens
    if Path(tokens[0]).name not in _WRAPPER_PROGRAMS:
        return tokens
    if tokens[1] not in _WRAPPER_FLAGS:
        return tokens
    return _shell_tokens(tokens[2])


def _program(tokens: list[str]) -> str:
    """Basename of the executed program, or ``""`` when there is no command."""
    if not tokens:
        return ""
    return Path(tokens[0]).name


def _strip_cd_prefix(tokens: list[str]) -> list[str]:
    """Drop a leading ``cd <dir> &&`` prefix, repeatedly.

    Almost every command persisted by this machine's sessions starts with
    ``cd <dir> &&``, so while the token list begins with exactly ``<program
    whose basename is 'cd'>``, one token that does not start with ``-``, and
    ``&&``, those three tokens are dropped and the check repeats -- ``cd /x &&
    cd /y && cat f`` strips twice and classifies ``cat f``.  Anything else is
    left unchanged: a flag as the second token (``cd -P /x && ...``), a
    different separator (``;``, ``|``), a ``VAR=value`` prefix, a subshell,
    and quoted prose (which is a single token) all stay untouched, so no
    signal is invented by descending into a compound form.
    """
    while len(tokens) >= 3 and Path(tokens[0]).name == "cd":
        if tokens[1].startswith("-") or tokens[2] != "&&":
            break
        tokens = tokens[3:]
    return tokens


def _is_companion_status_poll(tokens: list[str]) -> bool:
    """The executed command is ``kusabi-companion status`` (flags allowed after).

    Two invocation forms count.  The binary directly: the executed program's
    basename is ``kusabi-companion`` or ``kusabi-companion.mjs``.  And the
    ``node <script>`` form actually persisted by Claude Code sessions: the
    executed program's basename is ``node`` or ``nodejs`` and its first
    argument that does not start with ``-`` is a path whose basename is
    ``kusabi-companion.mjs``.  In both forms the subcommand is the first
    argument after the program (or after that script path) that does not
    start with ``-``, and only exactly ``status`` counts -- ``chain-show``,
    ``chain-wait``, ``result``, and a bare invocation with no subcommand stay
    unclassified.
    """
    program = _program(tokens)
    if program in (_COMPANION_BIN, _COMPANION_BIN + ".mjs"):
        arguments = tokens[1:]
    elif program in ("node", "nodejs"):
        arguments = tokens[1:]
        script_index: int | None = None
        for index, token in enumerate(arguments):
            if token.startswith("-"):
                continue
            script_index = index
            break
        if script_index is None or Path(arguments[script_index]).name != _COMPANION_BIN + ".mjs":
            return False
        arguments = arguments[script_index + 1 :]
    else:
        return False
    for token in arguments:
        if token.startswith("-"):
            continue
        return token == _COMPANION_SUBCOMMAND
    return False


def _is_host_file_hunt(tokens: list[str]) -> bool:
    """The executed program is ``rg``, ``grep``, ``sed``, or ``cat``."""
    return _program(tokens) in _HUNT_PROGRAMS


def _curl_targets(argv: Sequence[str]) -> list[str]:
    """The request targets of a curl invocation: the positional arguments.

    Options and the values they consume are skipped; ``--url VALUE`` (and
    ``--url=VALUE``) contribute VALUE as a target, since that option names the
    request target explicitly.  The value-taking option list is a conservative
    set of common curl options -- not a full curl parser -- and an unknown
    option shape is left as a potential target, so a URL-shaped header or data
    value is never the request target while the actual target still is.
    """
    targets: list[str] = []
    index = 0
    length = len(argv)
    while index < length:
        token = argv[index]
        if token == "--":
            targets.extend(argv[index + 1 :])
            break
        if token.startswith("--"):
            name, sep, attached = token[2:].partition("=")
            if name == "url":
                if sep:
                    targets.append(attached)
                elif index + 1 < length:
                    targets.append(argv[index + 1])
                    index += 1
            elif not sep and name in _CURL_VALUE_OPTIONS_LONG:
                index += 1  # the next token is this option's value
            index += 1
            continue
        if token.startswith("-") and token != "-":
            # A short-option cluster: the first value-taking option consumes
            # the rest of the token as its attached value, or -- when it is the
            # last character -- the next token as its value.
            for position, char in enumerate(token[1:]):
                if char in _CURL_VALUE_OPTIONS_SHORT:
                    if position == len(token) - 2 and index + 1 < length:
                        index += 1
                    break
            index += 1
            continue
        targets.append(token)
        index += 1
    return targets


def _is_raw_local_mcp_http(tokens: list[str]) -> bool:
    """The executed program is ``curl`` and its request *target* is a loopback
    MCP URL.  Only the target counts: a loopback URL used as an option value
    (e.g. ``-H 'http://localhost:8750/'``) is not a raw MCP call."""
    if _program(tokens) != _CURL_BIN:
        return False
    for target in _curl_targets(tokens[1:]):
        if "://" not in target:
            continue
        try:
            parsed = urlparse(target)
            port = parsed.port
        except ValueError:
            continue
        if parsed.hostname in _LOOPBACK_HOSTS and port in _MCP_PORTS:
            return True
    return False


def _is_undo_file_edit(tool_name: str | None) -> bool:
    return bool(tool_name) and _UNDO_FILE_EDIT_RE.match(tool_name) is not None


def _command_tokens(
    full_command: str | None,
    command_type: str | None,
    input_summary: str | None,
) -> list[str]:
    """The argv tokens to classify for one row.

    The full persisted ``input`` command wins when usable: a JSON-array
    command (DuckDB ``json_type`` ``'ARRAY'`` -- the Codex argv form) decodes
    to its actual argv, and a string command (``'VARCHAR'``) is
    shell-tokenized.  ``input_summary`` is the conservative fallback when no
    usable full command can be extracted (missing, blank, an object, a number,
    or a JSON array with non-string elements) -- arbitrary prose or objects
    are never turned into commands.  Without the fallback a long command would
    lose everything past the 200-character summary truncation.
    """
    if full_command and full_command.strip():
        if command_type == "ARRAY":
            try:
                decoded = json.loads(full_command)
            except ValueError:
                decoded = None
            if isinstance(decoded, list) and all(isinstance(item, str) for item in decoded):
                return decoded
        elif command_type == "VARCHAR":
            return _shell_tokens(full_command)
        # any other json_type (OBJECT, a number, ...) is not a command
    return _shell_tokens(input_summary)


def categories_for(
    tool_name: str | None,
    command: str | Sequence[str] | None,
    outcome: str | None,
) -> tuple[str, ...]:
    """The categories one persisted row belongs to, in :data:`CATEGORY_ORDER`.

    ``command`` is what to classify for shell rows: the full persisted
    ``input`` command text (tokenized here), its decoded argv list (the Codex
    JSON-array form), or the ``input_summary`` fallback (see
    :func:`_command_tokens`).  An exact ``bash``/``sh`` ``-c``/``-lc``
    wrapper is unwrapped and a leading ``cd <dir> &&`` prefix is stripped
    before the rules run (see :func:`_unwrap_shell_wrapper` and
    :func:`_strip_cd_prefix`).  The single classification point: the CLI
    never re-implements a category rule, and a future thin MCP adapter can
    reuse this function directly.
    """
    matched: list[str] = []
    if tool_name is not None and tool_name.lower() in _SHELL_TOOLS_LOWER:
        tokens = _shell_tokens(command) if isinstance(command, str) else list(command or ())
        tokens = _unwrap_shell_wrapper(tokens)
        tokens = _strip_cd_prefix(tokens)
        if _is_companion_status_poll(tokens):
            matched.append("companion_status_poll")
        if _is_host_file_hunt(tokens):
            matched.append("host_file_hunt")
        if _is_raw_local_mcp_http(tokens):
            matched.append("raw_local_mcp_http")
    if _is_undo_file_edit(tool_name):
        matched.append("undo_file_edit")
    if outcome == "pending":
        matched.append("pending_tool_call")
    return tuple(matched)


def audit(
    connection: duckdb.DuckDBPyConnection,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict[str, Any]:
    """One hygiene report over *connection* (read-only).

    *since*/*until* are inclusive bounds on the persisted ``ts`` column.
    Without either bound every row counts, including NULL timestamps; with
    either bound, NULL timestamps are excluded.  The ``coverage`` block counts
    the selected rows and distinct sessions *before* category filtering;
    categories may overlap, so their ``tool_calls`` do not sum to coverage.
    ``coverage["sources"]`` breaks the selected rows down per source (sorted
    by ``tool_calls`` descending, then ``source`` ascending; a NULL source is
    reported as ``None``).  ``coverage["excluded_no_timestamp"]`` counts the
    rows whose ``ts`` is NULL that a bounded window dropped, per source --
    the disclosure for sources that record no per-call timestamp -- and is
    ``{"tool_calls": 0, "sources": []}`` when neither bound is given, because
    an unbounded audit counts NULL-timestamp rows.
    """
    query = """
        SELECT
            session_id,
            source,
            tool_name,
            input_summary,
            outcome,
            input->>'command' AS input_command,
            json_type(input->'command') AS input_command_type
        FROM tool_calls
    """
    conditions: list[str] = []
    params: list[Any] = []
    if since is not None:
        conditions.append("ts >= ?")
        params.append(since)
    if until is not None:
        conditions.append("ts <= ?")
        params.append(until)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    rows = connection.execute(query, params).fetchall()

    coverage_sessions: set[str] = set()
    calls: dict[str, int] = {name: 0 for name in CATEGORY_ORDER}
    sessions: dict[str, set[str]] = {name: set() for name in CATEGORY_ORDER}
    source_calls: dict[str | None, int] = {}
    source_sessions: dict[str | None, set[str]] = {}
    for (
        session_id,
        source,
        tool_name,
        input_summary,
        outcome,
        input_command,
        input_command_type,
    ) in rows:
        if session_id is not None:
            coverage_sessions.add(session_id)
        source_calls[source] = source_calls.get(source, 0) + 1
        if session_id is not None:
            source_sessions.setdefault(source, set()).add(session_id)
        tokens = _command_tokens(input_command, input_command_type, input_summary)
        for name in categories_for(tool_name, tokens, outcome):
            calls[name] += 1
            if session_id is not None:
                sessions[name].add(session_id)

    sources = [
        {
            "source": source,
            "tool_calls": source_calls[source],
            "sessions": len(source_sessions.get(source, set())),
        }
        for source in source_calls
    ]
    sources.sort(key=lambda item: (-item["tool_calls"], item["source"] or ""))

    excluded_no_timestamp: dict[str, Any] = {"tool_calls": 0, "sources": []}
    if since is not None or until is not None:
        excluded_rows = connection.execute(
            "SELECT source, COUNT(*) FROM tool_calls WHERE ts IS NULL GROUP BY source"
        ).fetchall()
        excluded_sources = [
            {"source": source, "tool_calls": int(count)} for source, count in excluded_rows
        ]
        excluded_sources.sort(key=lambda item: (-item["tool_calls"], item["source"] or ""))
        excluded_no_timestamp = {
            "tool_calls": sum(item["tool_calls"] for item in excluded_sources),
            "sources": excluded_sources,
        }

    return {
        "coverage": {
            "since": since,
            "until": until,
            "sessions": len(coverage_sessions),
            "tool_calls": len(rows),
            "sources": sources,
            "excluded_no_timestamp": excluded_no_timestamp,
        },
        "categories": [
            {"name": name, "tool_calls": calls[name], "sessions": len(sessions[name])}
            for name in CATEGORY_ORDER
        ],
    }