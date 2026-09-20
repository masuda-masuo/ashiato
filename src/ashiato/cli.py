"""The ``ashiato`` command line: build, sql, denials, recalls, info."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb

from ashiato import __version__
from ashiato.build import (
    DEFAULT_CODEX_SOURCE,
    DEFAULT_SOURCE,
    SchemaOutOfDate,
    assert_readable,
    build,
    connect,
    database_info,
    default_db_path,
)
from ashiato.compare import compare_periods as compare_periods_fn
from ashiato.grep import DEFAULT_CONTEXT as DEFAULT_GREP_CONTEXT
from ashiato.grep import DEFAULT_LIMIT as DEFAULT_GREP_LIMIT
from ashiato.grep import Hit, InvalidPattern
from ashiato.grep import search as grep_search
from ashiato.grep import visible as grep_visible
from ashiato.grep import window as grep_window
from ashiato.hygiene import audit as hygiene_audit
from ashiato.memory_authors import run as memory_authors_run
from ashiato.nominate import run as nominate_run
from ashiato.orphans import DEFAULT_LIMIT as DEFAULT_ORPHANS_LIMIT
from ashiato.orphans import (
    DEFAULT_MIN_HUMAN_CHARS,
    DEFAULT_MIN_ORPHANS,
    DEFAULT_MIN_TF,
    default_reviewed_path,
    mark_reviewed,
    unmark_reviewed,
)
from ashiato.orphans import run as orphans_run
from ashiato.pending import run as pending_run
from ashiato.salvage import DEFAULT_LIMIT as DEFAULT_SALVAGE_LIMIT
from ashiato.salvage import DEFAULT_WINDOW_MINUTES, default_kaiba_db_path, nominate, open_kaiba
from ashiato.schema import (
    RECALL_FOLLOWUPS_VIEW,
    REQUIRED_VIEWS,
    TABLE_COLUMNS,
    TABLES,
    VIEW_COLUMNS,
    denial_followups_query,
)
from ashiato.serve import DEFAULT_HOST as DEFAULT_SERVE_HOST
from ashiato.serve import DEFAULT_PORT as DEFAULT_SERVE_PORT
from ashiato.serve import run as serve_run
from ashiato.session_trace import (
    DEFAULT_EXCERPT_CHARS as DEFAULT_SESSION_TRACE_EXCERPT,
)
from ashiato.session_trace import (
    DEFAULT_LIMIT as DEFAULT_SESSION_TRACE_LIMIT,
)
from ashiato.session_trace import SessionResolutionError, resolve_session
from ashiato.session_trace import trace as session_trace
from ashiato.topics import DEFAULT_TERMS as DEFAULT_TOPICS_TERMS
from ashiato.topics import DEFAULT_WINDOW as DEFAULT_TOPICS_WINDOW
from ashiato.topics import run as topics_run

FORMATS = ("table", "json", "csv")

#: Enough denials to read in one screen; ``--limit 0`` asks for all of them.
DEFAULT_DENIAL_LIMIT = 50

#: Same convention as denials.
DEFAULT_RECALL_LIMIT = 50


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ashiato",
        description="Turn Claude Code session transcripts into a queryable database.",
    )
    parser.add_argument("--version", action="version", version=f"ashiato {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="parse transcripts into the database")
    build_parser.add_argument(
        "--source",
        action="append",
        metavar="DIR",
        help=f"directory searched recursively for *.jsonl (repeatable; default {DEFAULT_SOURCE})",
    )
    build_parser.add_argument(
        "--opencode-source",
        action="append",
        dest="opencode_source",
        metavar="DIR",
        help="directory searched recursively for opencode *.ndjson job event streams (repeatable)",
    )
    build_parser.add_argument(
        "--cursor-source",
        action="append",
        dest="cursor_source",
        metavar="DIR",
        help="directory searched recursively for Cursor agent-transcript *.jsonl (repeatable)",
    )
    build_parser.add_argument(
        "--codex-source",
        action="append",
        dest="codex_source",
        metavar="DIR",
        help=(
            f"directory searched recursively for Codex session *.jsonl "
            f"(repeatable; default {DEFAULT_CODEX_SOURCE})"
        ),
    )
    build_parser.add_argument(
        "--kaiba-db",
        metavar="PATH",
        help="kaiba sqlite db used to fill in Cursor recall output/ts (default ~/.kaiba/kaiba.db)",
    )
    build_parser.add_argument("--db", metavar="PATH", help="database path")

    sql_parser = subparsers.add_parser("sql", help="run a query against the database")
    sql_parser.add_argument("query", help="SQL to execute")
    sql_parser.add_argument("--db", metavar="PATH", help="database path")
    sql_parser.add_argument("--format", choices=FORMATS, default="table", help="output format")

    denials_parser = subparsers.add_parser(
        "denials", help="denied tool calls and what the session did next"
    )
    denials_parser.add_argument("--db", metavar="PATH", help="database path")
    denials_parser.add_argument("--format", choices=FORMATS, default="table", help="output format")
    denials_parser.add_argument(
        "--limit",
        type=_row_limit,
        default=DEFAULT_DENIAL_LIMIT,
        metavar="N",
        help=f"maximum rows, 0 for all (default {DEFAULT_DENIAL_LIMIT})",
    )
    denials_parser.add_argument("--session", metavar="ID", help="restrict to one session")

    recalls_parser = subparsers.add_parser(
        "recalls", help="kaiba recall calls and what the session did next"
    )
    recalls_parser.add_argument("--db", metavar="PATH", help="database path")
    recalls_parser.add_argument("--format", choices=FORMATS, default="table", help="output format")
    recalls_parser.add_argument(
        "--limit",
        type=_row_limit,
        default=DEFAULT_RECALL_LIMIT,
        metavar="N",
        help=f"maximum rows, 0 for all (default {DEFAULT_RECALL_LIMIT})",
    )
    recalls_parser.add_argument("--session", metavar="ID", help="restrict to one session")

    session_trace_parser = subparsers.add_parser(
        "session-trace",
        help="one session's interleaved text and tool-call timeline",
    )
    session_trace_parser.add_argument(
        "session_prefix",
        metavar="SESSION_PREFIX",
        help="session id, or a unique prefix of it",
    )
    session_trace_parser.add_argument("--db", metavar="PATH", help="database path")
    session_trace_parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="output format (default table)",
    )
    session_trace_parser.add_argument(
        "--limit",
        type=_row_limit,
        default=DEFAULT_SESSION_TRACE_LIMIT,
        metavar="N",
        help=f"maximum timeline rows, 0 for all (default {DEFAULT_SESSION_TRACE_LIMIT})",
    )
    session_trace_parser.add_argument(
        "--max-excerpt-chars",
        type=_row_limit,
        default=DEFAULT_SESSION_TRACE_EXCERPT,
        metavar="N",
        help="maximum characters of each text/result excerpt, 0 for uncapped "
        f"(default {DEFAULT_SESSION_TRACE_EXCERPT})",
    )

    topics_parser = subparsers.add_parser(
        "topics",
        help="a deterministic topic outline of one session, segmented without an LLM",
    )
    topics_parser.add_argument(
        "session_prefix",
        metavar="SESSION_PREFIX",
        help="session id, or a unique prefix of it",
    )
    topics_parser.add_argument("--db", metavar="PATH", help="database path")
    topics_parser.add_argument(
        "--window",
        type=_positive,
        default=DEFAULT_TOPICS_WINDOW,
        metavar="N",
        help="exchanges on each side of a candidate boundary gap, at least 1 "
        f"(default {DEFAULT_TOPICS_WINDOW})",
    )
    topics_parser.add_argument(
        "--terms",
        type=_row_limit,
        default=DEFAULT_TOPICS_TERMS,
        metavar="N",
        help=f"topic terms per segment (default {DEFAULT_TOPICS_TERMS})",
    )
    topics_parser.add_argument(
        "--json", action="store_true", dest="json_output", help="output as JSON"
    )

    info_parser = subparsers.add_parser("info", help="describe the database")
    info_parser.add_argument("--db", metavar="PATH", help="database path")

    schema_parser = subparsers.add_parser("schema", help="show table/view columns")
    schema_parser.add_argument("table", nargs="?", help="table or view name to describe")
    schema_parser.add_argument(
        "--db", metavar="PATH", help="database path (optional; schema is derived from code)"
    )

    salvage_parser = subparsers.add_parser(
        "salvage", help="nominate work-state changes with no bookkeeping trail"
    )
    salvage_parser.add_argument("--db", metavar="PATH", help="database path")
    salvage_parser.add_argument(
        "--kaiba-db",
        metavar="PATH",
        help="kaiba actions ledger path (default ~/.kaiba/kaiba.db)",
    )
    salvage_parser.add_argument(
        "--window-minutes",
        type=_row_limit,
        default=DEFAULT_WINDOW_MINUTES,
        metavar="N",
        help=f"kaiba coverage window in minutes (default {DEFAULT_WINDOW_MINUTES})",
    )
    salvage_parser.add_argument(
        "--limit",
        type=_row_limit,
        default=DEFAULT_SALVAGE_LIMIT,
        metavar="N",
        help=f"maximum nominations, 0 for all (default {DEFAULT_SALVAGE_LIMIT})",
    )
    salvage_parser.add_argument(
        "--since",
        type=_parse_since,
        metavar="TS",
        help="only consider evidence at or after this ISO-8601 timestamp",
    )

    nominate_parser = subparsers.add_parser(
        "nominate", help="mine re-derived facts as kaiba nomination candidates"
    )
    nominate_parser.add_argument("--db", metavar="PATH", help="database path")
    nominate_parser.add_argument(
        "--since",
        type=_parse_since,
        metavar="TS",
        help="only consider calls at or after this ISO-8601 timestamp",
    )
    nominate_parser.add_argument(
        "--until",
        type=_parse_since,
        metavar="TS",
        help="only consider calls at or before this ISO-8601 timestamp",
    )
    nominate_parser.add_argument(
        "--min-sessions",
        type=_row_limit,
        default=3,
        metavar="N",
        help="minimum distinct sessions for a candidate (default 3)",
    )
    nominate_parser.add_argument(
        "--min-stability",
        type=float,
        default=1.0,
        metavar="F",
        help="minimum modal output share for stable-output (default 1.0)",
    )
    nominate_parser.add_argument(
        "--exclude-file",
        metavar="PATH",
        help="additional ritual exclusion patterns (one regex per line)",
    )
    nominate_parser.add_argument(
        "--max-output-chars",
        type=_row_limit,
        default=2000,
        metavar="N",
        help="truncate result text to N chars before comparison (default 2000)",
    )
    nominate_parser.add_argument(
        "--json", action="store_true", dest="json_output", help="output as JSON"
    )

    orphans_parser = subparsers.add_parser(
        "orphans",
        help="nominate one-off discussion topics that left no trace in any sink",
    )
    orphans_parser.add_argument("--db", metavar="PATH", help="database path")
    orphans_parser.add_argument(
        "--since",
        type=_parse_since,
        metavar="TS",
        help="only nominate sessions started at or after this ISO-8601 timestamp",
    )
    orphans_parser.add_argument(
        "--until",
        type=_parse_since,
        metavar="TS",
        help="only nominate sessions started at or before this ISO-8601 timestamp",
    )
    orphans_parser.add_argument(
        "--sink",
        action="append",
        metavar="PATH",
        help="file or directory of text the topic may have been persisted in "
        "(repeatable; default every ~/.claude/projects/*/memory dir)",
    )
    orphans_parser.add_argument(
        "--no-default-sinks",
        action="store_true",
        help="do not fall back to ~/.claude/projects/*/memory when no --sink is given",
    )
    orphans_parser.add_argument(
        "--min-tf",
        type=_row_limit,
        default=DEFAULT_MIN_TF,
        metavar="N",
        help=f"minimum in-session occurrences of a unique term (default {DEFAULT_MIN_TF})",
    )
    orphans_parser.add_argument(
        "--min-human-chars",
        type=_row_limit,
        default=DEFAULT_MIN_HUMAN_CHARS,
        metavar="N",
        help="minimum characters of human-typed text in a session "
        f"(default {DEFAULT_MIN_HUMAN_CHARS})",
    )
    orphans_parser.add_argument(
        "--min-orphans",
        type=_row_limit,
        default=DEFAULT_MIN_ORPHANS,
        metavar="N",
        help="minimum distinct orphan terms for a session to be nominated "
        f"(default {DEFAULT_MIN_ORPHANS})",
    )
    orphans_parser.add_argument(
        "--include-headless",
        action="store_true",
        help="also nominate headless (sdk-cli entrypoint) sessions, whose "
        "'human' text is a machine-written brief",
    )
    orphans_parser.add_argument(
        "--limit",
        type=_row_limit,
        default=DEFAULT_ORPHANS_LIMIT,
        metavar="N",
        help=f"maximum candidates, 0 for all (default {DEFAULT_ORPHANS_LIMIT})",
    )
    reviewed_mark = orphans_parser.add_mutually_exclusive_group()
    reviewed_mark.add_argument(
        "--mark-reviewed",
        action="append",
        metavar="ID",
        help="resolve ID (a full session id or unique prefix) and append its full "
        "session id to the reviewed file, then exit without nominating "
        "(repeatable; conflicts with --unmark-reviewed)",
    )
    reviewed_mark.add_argument(
        "--unmark-reviewed",
        action="append",
        metavar="ID",
        help="resolve ID and remove its line(s) from the reviewed file, then exit "
        "without nominating (repeatable; conflicts with --mark-reviewed)",
    )
    orphans_parser.add_argument(
        "--reviewed-file",
        metavar="PATH",
        help="file of reviewed session ids, one per line (default: "
        "orphans-reviewed.txt next to the database)",
    )
    orphans_parser.add_argument(
        "--show-reviewed",
        action="store_true",
        help="also nominate sessions whose session ids are in the reviewed file",
    )
    orphans_parser.add_argument(
        "--json", action="store_true", dest="json_output", help="output as JSON"
    )

    memory_authors_parser = subparsers.add_parser(
        "memory-authors",
        help="attribute each Claude Code memory file to the models that wrote it",
    )
    memory_authors_parser.add_argument("--db", metavar="PATH", help="database path")
    memory_authors_parser.add_argument(
        "--since",
        type=_parse_since,
        metavar="TS",
        help="only consider calls at or after this ISO-8601 timestamp",
    )
    memory_authors_parser.add_argument(
        "--until",
        type=_parse_since,
        metavar="TS",
        help="only consider calls at or before this ISO-8601 timestamp",
    )
    memory_authors_parser.add_argument(
        "--memory-dir",
        action="append",
        metavar="PATH",
        help="memory directory scanned for unattributed *.md files "
        "(repeatable; default every existing ~/.claude/projects/*/memory dir)",
    )
    memory_authors_parser.add_argument(
        "--model",
        metavar="NAME",
        help="only list files whose creator or any editor is this model",
    )
    memory_authors_parser.add_argument(
        "--json", action="store_true", dest="json_output", help="output as JSON"
    )

    pending_parser = subparsers.add_parser(
        "pending",
        help="open items left in compaction summaries, checked against GitHub",
    )
    pending_parser.add_argument("--db", metavar="PATH", help="database path")
    pending_parser.add_argument(
        "--gh",
        action="store_true",
        help="check each issue/PR reference with the gh CLI (read-only; lists "
        "the owner's repositories once and sends only owner/repo/number plus "
        "the owner name, never transcript text)",
    )
    pending_parser.add_argument(
        "--repo",
        type=_parse_repo,
        metavar="OWNER/NAME",
        help="owner/repo used to resolve bare #N references that have no "
        "repository in their own item",
    )
    pending_parser.add_argument(
        "--owner",
        metavar="NAME",
        help="owner short-form references resolve to (default: the owner most "
        "often named by explicit references in the DB)",
    )
    pending_parser.add_argument(
        "--all-summaries",
        action="store_true",
        help="process every compaction summary of a session instead of only the latest",
    )
    pending_parser.add_argument(
        "--since",
        type=_parse_since,
        metavar="TS",
        help="only summaries at or after this ISO-8601 timestamp",
    )
    pending_parser.add_argument(
        "--until",
        type=_parse_since,
        metavar="TS",
        help="only summaries at or before this ISO-8601 timestamp",
    )
    pending_parser.add_argument(
        "--show-resolved",
        action="store_true",
        help="also show items whose references are all closed/merged",
    )
    pending_parser.add_argument(
        "--json", action="store_true", dest="json_output", help="output as JSON"
    )

    hygiene_parser = subparsers.add_parser(
        "hygiene",
        help="named session-hygiene audit: companion polls, host-file hunts, "
        "raw MCP curl, undo calls, pending calls",
    )
    hygiene_parser.add_argument("--db", metavar="PATH", help="database path")
    hygiene_parser.add_argument(
        "--since",
        type=_parse_since,
        metavar="TS",
        help="only count calls at or after this ISO-8601 timestamp",
    )
    hygiene_parser.add_argument(
        "--until",
        type=_parse_since,
        metavar="TS",
        help="only count calls at or before this ISO-8601 timestamp",
    )
    hygiene_parser.add_argument(
        "--format", choices=("table", "json"), default="table", help="output format (default table)"
    )

    serve_parser = subparsers.add_parser(
        "serve",
        help="read-only local dashboard over the database (loopback only)",
    )
    serve_parser.add_argument("--db", metavar="PATH", help="database path")
    serve_parser.add_argument(
        "--host",
        default=DEFAULT_SERVE_HOST,
        metavar="HOST",
        help="loopback address to listen on: 127.0.0.1, ::1 or localhost "
        f"(default {DEFAULT_SERVE_HOST}); anything else is refused",
    )
    serve_parser.add_argument(
        "--port",
        type=_port,
        default=DEFAULT_SERVE_PORT,
        metavar="N",
        help=f"port to listen on, 0 for any free port (default {DEFAULT_SERVE_PORT})",
    )
    serve_parser.add_argument(
        "--sink",
        action="append",
        metavar="PATH",
        help="sink passed to the orphans page, exactly as 'ashiato orphans --sink' "
        "(repeatable; default every ~/.claude/projects/*/memory dir)",
    )
    serve_parser.add_argument(
        "--no-default-sinks",
        action="store_true",
        help="do not fall back to ~/.claude/projects/*/memory when no --sink is given",
    )
    serve_parser.add_argument(
        "--memory-dir",
        action="append",
        metavar="PATH",
        help="memory directory scanned for unattributed *.md files on the memory page "
        "(repeatable; default every existing ~/.claude/projects/*/memory dir)",
    )
    serve_parser.add_argument(
        "--reviewed-file",
        metavar="PATH",
        help="file of reviewed session ids read by the orphans page "
        "(default: orphans-reviewed.txt next to the database)",
    )

    compare_periods_parser = subparsers.add_parser(
        "compare-periods",
        help="compare hygiene metrics across two time periods (baseline vs current)",
    )
    compare_periods_parser.add_argument("--db", metavar="PATH", help="database path")
    compare_periods_parser.add_argument(
        "--format", choices=("json", "table"), default="table",
        help="output format (default table)",
    )
    compare_periods_parser.add_argument(
        "--period",
        action="append",
        required=True,
        metavar="START..END",
        help="period as START..END ISO-8601 (baseline first, current second)",
    )

    grep_parser = subparsers.add_parser(
        "grep", help="regex search over transcript text with match windows"
    )
    grep_parser.add_argument("pattern", help="regular expression to search for")
    grep_parser.add_argument("--db", metavar="PATH", help="database path")
    grep_parser.add_argument("--format", choices=FORMATS, default="table", help="output format")
    grep_parser.add_argument(
        "--role", choices=("user", "assistant"), help="restrict event hits to one role"
    )
    grep_parser.add_argument(
        "--since",
        type=_parse_since,
        metavar="TS",
        help="only rows at or after this ISO-8601 timestamp",
    )
    grep_parser.add_argument(
        "--until",
        type=_parse_since,
        metavar="TS",
        help="only rows at or before this ISO-8601 timestamp",
    )
    grep_parser.add_argument(
        "--session", metavar="PREFIX", help="restrict to sessions whose id starts with PREFIX"
    )
    grep_parser.add_argument(
        "-i",
        "--ignore-case",
        action="store_true",
        dest="ignore_case",
        help="case-insensitive match",
    )
    grep_parser.add_argument(
        "--tool-calls",
        action="store_true",
        dest="tool_calls",
        help="also search tool_calls.input_summary and tool_calls.result_text",
    )
    grep_parser.add_argument(
        "--include-meta",
        action="store_true",
        dest="include_meta",
        help="include is_meta events (harness noise), excluded by default",
    )
    grep_parser.add_argument(
        "--context",
        type=_row_limit,
        default=DEFAULT_GREP_CONTEXT,
        metavar="N",
        help=f"characters of context on each side of a match (default {DEFAULT_GREP_CONTEXT})",
    )
    grep_parser.add_argument(
        "--all-matches",
        action="store_true",
        dest="all_matches",
        help="print a window per match instead of only the first",
    )
    grep_parser.add_argument(
        "--whole",
        action="store_true",
        help="print the full text of the matched row instead of a window",
    )
    grep_parser.add_argument(
        "--limit",
        type=_row_limit,
        default=DEFAULT_GREP_LIMIT,
        metavar="N",
        help=f"maximum hits, 0 for all (default {DEFAULT_GREP_LIMIT})",
    )

    return parser


def _row_limit(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be 0 or greater")
    return number


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be 1 or greater")
    return number


def _port(value: str) -> int:
    number = int(value)
    if not 0 <= number <= 65535:
        raise argparse.ArgumentTypeError("must be between 0 and 65535")
    return number


def _resolve_db(value: str | None) -> Path:
    return Path(value).expanduser() if value else default_db_path()


def _parse_repo(value: str) -> tuple[str, str]:
    """An ``OWNER/NAME`` pair for --repo, or an argparse error."""
    parts = value.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError(f"invalid repo: {value!r} (expected OWNER/NAME)")
    return parts[0], parts[1]


def _parse_since(value: str) -> datetime:
    """An ISO-8601 timestamp, normalised to naive UTC like DuckDB's ``ts`` column."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid timestamp: {value}") from error
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def _cell(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, datetime | date):
        return value.isoformat(sep=" ") if isinstance(value, datetime) else value.isoformat()
    return str(value)


def _print_table(columns: Sequence[str], rows: Sequence[Sequence[Any]], stream: Any) -> None:
    if not columns:
        return
    cells = [[_cell(value) for value in row] for row in rows]
    widths = [len(name) for name in columns]
    for row in cells:
        for index, text in enumerate(row):
            if index < len(widths):
                widths[index] = max(widths[index], len(text))
    print("  ".join(name.ljust(widths[i]) for i, name in enumerate(columns)).rstrip(), file=stream)
    print("  ".join("-" * width for width in widths), file=stream)
    for row in cells:
        print("  ".join(text.ljust(widths[i]) for i, text in enumerate(row)).rstrip(), file=stream)
    print(f"({len(cells)} row{'' if len(cells) == 1 else 's'})", file=stream)


def _print_json(columns: Sequence[str], rows: Sequence[Sequence[Any]], stream: Any) -> None:
    payload = [dict(zip(columns, row, strict=False)) for row in rows]
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str), file=stream)


def _print_csv(columns: Sequence[str], rows: Sequence[Sequence[Any]], stream: Any) -> None:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow(["" if value is None else _cell(value) for value in row])


def _render(columns: Sequence[str], rows: Sequence[Sequence[Any]], fmt: str, stream: Any) -> None:
    if fmt == "json":
        _print_json(columns, rows, stream)
    elif fmt == "csv":
        _print_csv(columns, rows, stream)
    else:
        _print_table(columns, rows, stream)


def _run_query(
    db_path: Path, query: str, params: Sequence[Any], fmt: str, out: Any, err: Any
) -> int:
    """Run one read-only query and print it; every failure is an exit code, not a traceback."""
    if not db_path.exists():
        print(f"error: no database at {db_path} (run 'ashiato build' first)", file=err)
        return 1
    connection = connect(db_path, read_only=True)
    try:
        # Before the query, not after it fails: an out-of-date database gets the
        # rebuild hint, and a query the user got wrong keeps DuckDB's own words.
        assert_readable(connection)
        cursor = connection.execute(query, list(params))
        columns = [description[0] for description in cursor.description or []]
        rows = cursor.fetchall() if columns else []
    except SchemaOutOfDate as error:
        print(f"error: {error}", file=err)
        return 1
    except duckdb.Error as error:
        print(f"error: {error}", file=err)
        return 1
    finally:
        connection.close()

    _render(columns, rows, fmt, out)
    return 0


def _run_build(args: argparse.Namespace, out: Any, err: Any) -> int:
    sources = args.source or [str(DEFAULT_SOURCE)]
    opencode_sources = args.opencode_source or []
    cursor_sources = args.cursor_source or []
    codex_sources = args.codex_source or []
    kaiba_db_path = Path(args.kaiba_db).expanduser() if args.kaiba_db else None
    db_path = _resolve_db(args.db)
    try:
        result = build(
            sources,
            db_path,
            opencode_sources=opencode_sources,
            cursor_sources=cursor_sources,
            codex_sources=codex_sources,
            kaiba_db_path=kaiba_db_path,
        )
    except SchemaOutOfDate as error:
        print(f"error: {error}", file=err)
        return 1

    for source in result.missing_sources:
        print(f"warning: source not found: {source}", file=err)
    for path in result.unreadable_files:
        print(f"warning: could not read: {path}", file=err)
    for path in result.failed_files:
        print(f"warning: could not store: {path}", file=err)
    if result.kaiba_db_unavailable:
        print(
            f"notice: no kaiba db at {result.kaiba_db_unavailable} -- "
            "Cursor recall rows have NULL output/ts",
            file=err,
        )

    print(f"database: {result.db_path}", file=out)
    print(
        f"files: {result.n_processed} processed, {result.n_skipped} skipped (unchanged), "
        f"{result.n_files} found",
        file=out,
    )
    print(
        f"rows: {result.n_sessions} sessions, {result.n_events} events, "
        f"{result.n_tool_calls} tool calls, {result.n_recall_calls} recall calls",
        file=out,
    )
    print(f"unparseable lines skipped: {result.n_parse_errors}", file=out)
    return 0


def _run_sql(args: argparse.Namespace, out: Any, err: Any) -> int:
    return _run_query(_resolve_db(args.db), args.query, [], args.format, out, err)


def _run_denials(args: argparse.Namespace, out: Any, err: Any) -> int:
    query, params = denial_followups_query(args.session, args.limit)
    return _run_query(_resolve_db(args.db), query, params, args.format, out, err)


def _run_recalls(args: argparse.Namespace, out: Any, err: Any) -> int:
    query = f'SELECT * FROM "{RECALL_FOLLOWUPS_VIEW}"'
    params: list[Any] = []
    if args.session:
        query += " WHERE session_id = ?"
        params.append(args.session)
    # Same convention as denials: newest first, ties settled by session/seq.
    query += " ORDER BY ts DESC NULLS LAST, session_id, seq DESC"
    if args.limit:
        query += " LIMIT ?"
        params.append(args.limit)
    return _run_query(_resolve_db(args.db), query, params, args.format, out, err)


def _print_session_trace_table(payload: dict[str, Any], out: Any) -> None:
    columns = ("seq", "ts", "kind", "id", "role", "tool_name", "outcome", "excerpt")
    rows = [
        [
            row["seq"],
            row["ts"],
            row["kind"],
            row["id"],
            row.get("role"),
            row.get("tool_name"),
            row.get("outcome"),
            row.get("excerpt"),
        ]
        for row in payload["timeline"]
    ]
    _print_table(columns, rows, out)


def _run_session_trace(args: argparse.Namespace, out: Any, err: Any) -> int:
    db_path = _resolve_db(args.db)
    if not db_path.exists():
        print(f"error: no database at {db_path} (run 'ashiato build' first)", file=err)
        return 1
    connection = connect(db_path, read_only=True)
    try:
        assert_readable(connection)
        session_id = resolve_session(connection, args.session_prefix)
        payload = session_trace(
            connection,
            session_id,
            limit=args.limit,
            max_excerpt_chars=args.max_excerpt_chars,
        )
    except SessionResolutionError as error:
        print(f"error: {error}", file=err)
        return 1
    except SchemaOutOfDate as error:
        print(f"error: {error}", file=err)
        return 1
    except duckdb.Error as error:
        print(f"error: {error}", file=err)
        return 1
    finally:
        connection.close()

    if args.format == "json":
        print(json.dumps(payload, indent=2, ensure_ascii=False), file=out)
    else:
        _print_session_trace_table(payload, out)
    return 0


def _run_topics(args: argparse.Namespace, out: Any, err: Any) -> int:
    return topics_run(
        _resolve_db(args.db),
        args.session_prefix,
        window=args.window,
        terms=args.terms,
        json_output=args.json_output,
        out=out,
        err=err,
    )


def _run_info(args: argparse.Namespace, out: Any, err: Any) -> int:
    db_path = _resolve_db(args.db)
    if not db_path.exists():
        print(f"error: no database at {db_path} (run 'ashiato build' first)", file=err)
        return 1
    try:
        info = database_info(db_path)
    except SchemaOutOfDate as error:
        print(f"error: {error}", file=err)
        return 1
    except duckdb.Error as error:
        print(f"error: {error}", file=err)
        return 1

    print(f"database: {info.db_path}", file=out)
    for table, count in info.table_counts.items():
        print(f"  {table:<13} {count:>10}", file=out)
    window = (
        f"{_cell(info.started_at)} .. {_cell(info.ended_at)}"
        if info.started_at or info.ended_at
        else "empty"
    )
    print(f"time window: {window}", file=out)

    # Ingested roots
    if (
        info.sources is None
        and info.opencode_sources is None
        and info.cursor_sources is None
        and info.codex_sources is None
    ):
        print(
            "ingested roots: unknown (database built before root recording; "
            "rebuild to record them)",
            file=out,
        )
    else:
        print("ingested roots:", file=out)
        _print_roots("  sources", info.sources or [], out)
        _print_roots("  opencode_sources", info.opencode_sources or [], out)
        _print_roots("  cursor_sources", info.cursor_sources or [], out)
        _print_roots("  codex_sources", info.codex_sources or [], out)

    # Freshness gap
    if info.freshness_gap is None:
        print("freshness: unknown (roots not recorded)", file=out)
    elif info.freshness_gap == 0:
        print("freshness: current (no new or changed files under recorded roots)", file=out)
    else:
        plural = "s" if info.freshness_gap != 1 else ""
        print(
            f"freshness: {info.freshness_gap} new or changed file{plural} "
            "under recorded roots (run 'ashiato build' to update)",
            file=out,
        )

    return 0


def _print_roots(label: str, roots: list[tuple[str, int]], out: Any) -> None:
    if roots:
        for root, count in roots:
            root_part = f"{root} ({count} file{'s' if count != 1 else ''})"
            print(f"  {label:<20} {root_part}", file=out)
    else:
        print(f"  {label:<20} (none)", file=out)


def _run_schema(args: argparse.Namespace, out: Any, err: Any) -> int:
    """List tables/views and their columns.

    The schema is derived from the code (TABLE_COLUMNS and REQUIRED_VIEWS),
    not from a particular database, so this works without a database.
    """
    # Build the combined schema: tables + views
    all_names = list(TABLES) + list(REQUIRED_VIEWS)

    if args.table is None:
        # List all tables and views
        for name in all_names:
            kind = "table" if name in TABLES else "view"
            print(f"  {name:<20} {kind}", file=out)
        return 0

    # Describe a specific table or view
    table_name = args.table
    if table_name not in all_names:
        print(f"error: unknown table or view: {table_name}", file=err)
        print("Available:", file=err)
        for name in all_names:
            kind = "table" if name in TABLES else "view"
            print(f"  {name:<20} {kind}", file=err)
        return 1

    if table_name in TABLE_COLUMNS:
        columns = TABLE_COLUMNS[table_name]
        print(f"{table_name} (table)", file=out)
        for col_name, col_type in columns:
            print(f"  {col_name:<30} {col_type}", file=out)
    else:
        # View - columns are defined in VIEW_COLUMNS so this works without a database
        print(f"{table_name} (view)", file=out)
        columns = VIEW_COLUMNS[table_name]
        for col_name, col_type in columns:
            print(f"  {col_name:<30} {col_type}", file=out)
    return 0


def _run_salvage(args: argparse.Namespace, out: Any, err: Any) -> int:
    db_path = _resolve_db(args.db)
    if not db_path.exists():
        print(f"error: no database at {db_path} (run 'ashiato build' first)", file=err)
        return 1

    kaiba_path = Path(args.kaiba_db).expanduser() if args.kaiba_db else default_kaiba_db_path()
    kaiba_connection = open_kaiba(kaiba_path)
    if kaiba_connection is None:
        print(f"notice: no kaiba db at {kaiba_path} -- coverage is transcript-only", file=err)

    connection = connect(db_path, read_only=True)
    try:
        assert_readable(connection)
        nominations = nominate(
            connection,
            kaiba_connection,
            window_minutes=args.window_minutes,
            since=args.since,
            limit=args.limit,
        )
    except SchemaOutOfDate as error:
        print(f"error: {error}", file=err)
        return 1
    except duckdb.Error as error:
        print(f"error: {error}", file=err)
        return 1
    finally:
        connection.close()
        if kaiba_connection is not None:
            kaiba_connection.close()

    for nomination in nominations:
        checks = ",".join(nomination.failed_checks)
        print(
            f"{nomination.ts.isoformat()}  session={nomination.session_id}  "
            f"kind={nomination.kind}  failed={checks}  {nomination.snippet}",
            file=out,
        )
    print(
        f"({len(nominations)} nomination{'' if len(nominations) == 1 else 's'})",
        file=out,
    )
    return 0


def _run_nominate(args: argparse.Namespace, out: Any, err: Any) -> int:
    db_path = _resolve_db(args.db)
    exclude_file = Path(args.exclude_file) if args.exclude_file else None
    return nominate_run(
        db_path,
        since=args.since,
        until=args.until,
        min_sessions=args.min_sessions,
        min_stability=args.min_stability,
        exclude_file=exclude_file,
        max_output_chars=args.max_output_chars,
        json_output=args.json_output,
        out=out,
        err=err,
    )


def _run_reviewed_update(
    db_path: Path,
    reviewed_path: Path,
    args: argparse.Namespace,
    out: Any,
    err: Any,
) -> int:
    """``orphans --mark-reviewed`` / ``--unmark-reviewed``: rewrite the
    reviewed file without running the nomination."""
    if not db_path.exists():
        print(f"error: no database at {db_path} (run 'ashiato build' first)", file=err)
        return 1
    connection = connect(db_path, read_only=True)
    try:
        assert_readable(connection)
        if args.mark_reviewed:
            return mark_reviewed(connection, reviewed_path, args.mark_reviewed, out=out, err=err)
        return unmark_reviewed(connection, reviewed_path, args.unmark_reviewed, out=out, err=err)
    except SchemaOutOfDate as error:
        print(f"error: {error}", file=err)
        return 1
    except duckdb.Error as error:
        print(f"error: {error}", file=err)
        return 1
    finally:
        connection.close()


def _run_orphans(args: argparse.Namespace, out: Any, err: Any) -> int:
    db_path = _resolve_db(args.db)
    reviewed_path = (
        Path(args.reviewed_file).expanduser()
        if args.reviewed_file
        else default_reviewed_path(db_path)
    )
    if args.mark_reviewed or args.unmark_reviewed:
        return _run_reviewed_update(db_path, reviewed_path, args, out, err)
    return orphans_run(
        db_path,
        since=args.since,
        until=args.until,
        sinks=[Path(sink).expanduser() for sink in args.sink or []],
        default_sinks=not args.no_default_sinks,
        min_tf=args.min_tf,
        min_human_chars=args.min_human_chars,
        min_orphans=args.min_orphans,
        include_headless=args.include_headless,
        limit=args.limit,
        json_output=args.json_output,
        reviewed_file=reviewed_path,
        show_reviewed=args.show_reviewed,
        out=out,
        err=err,
    )


def _run_memory_authors(args: argparse.Namespace, out: Any, err: Any) -> int:
    return memory_authors_run(
        _resolve_db(args.db),
        since=args.since,
        until=args.until,
        memory_dirs=[Path(path).expanduser() for path in args.memory_dir or []],
        default_dirs=args.memory_dir is None,
        model=args.model,
        json_output=args.json_output,
        out=out,
        err=err,
    )


def _run_pending(args: argparse.Namespace, out: Any, err: Any) -> int:
    return pending_run(
        _resolve_db(args.db),
        since=args.since,
        until=args.until,
        repo=args.repo,
        owner=args.owner,
        all_summaries=args.all_summaries,
        use_gh=args.gh,
        show_resolved=args.show_resolved,
        json_output=args.json_output,
        out=out,
        err=err,
    )


def _run_serve(args: argparse.Namespace, out: Any, err: Any) -> int:
    return serve_run(
        _resolve_db(args.db),
        host=args.host,
        port=args.port,
        sinks=[Path(sink).expanduser() for sink in args.sink or []],
        default_sinks=not args.no_default_sinks,
        memory_dirs=[Path(path).expanduser() for path in args.memory_dir or []],
        reviewed_file=(
            Path(args.reviewed_file).expanduser() if args.reviewed_file else None
        ),
        out=out,
        err=err,
    )


def _run_hygiene(args: argparse.Namespace, out: Any, err: Any) -> int:
    db_path = _resolve_db(args.db)
    if not db_path.exists():
        print(f"error: no database at {db_path} (run 'ashiato build' first)", file=err)
        return 1
    connection = connect(db_path, read_only=True)
    try:
        assert_readable(connection)
        report = hygiene_audit(connection, since=args.since, until=args.until)
    except SchemaOutOfDate as error:
        print(f"error: {error}", file=err)
        return 1
    except duckdb.Error as error:
        print(f"error: {error}", file=err)
        return 1
    finally:
        connection.close()

    if args.format == "json":
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str), file=out)
    else:
        _print_hygiene_table(report, out)
    return 0


def _print_hygiene_table(report: dict[str, Any], out: Any) -> None:
    """The table form of a hygiene report: the same counts as the JSON shape."""
    coverage = report["coverage"]
    since = _cell(coverage["since"]) if coverage["since"] is not None else "none"
    until = _cell(coverage["until"]) if coverage["until"] is not None else "none"
    print(
        f"coverage: since {since} .. until {until}  "
        f"{coverage['sessions']} sessions, {coverage['tool_calls']} tool calls",
        file=out,
    )
    rows = [[cat["name"], cat["tool_calls"], cat["sessions"]] for cat in report["categories"]]
    _print_table(("category", "tool_calls", "sessions"), rows, out)


def _parse_period(value: str) -> tuple[datetime, datetime]:
    """Parse a ``START..END`` pair of ISO-8601 timestamps."""
    parts = value.split("..")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"invalid period: {value!r} (expected START..END)"
        )
    if not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError(
            f"invalid period: {value!r} (empty timestamp in START..END)"
        )
    return _parse_since(parts[0]), _parse_since(parts[1])


def _validate_periods(
    baseline_since: datetime,
    baseline_until: datetime,
    current_since: datetime,
    current_until: datetime,
    err: Any,
) -> int:
    """Validate period ordering constraints.  Returns 0 on success, 2 on error."""
    if baseline_since > baseline_until:
        print(
            f"error: baseline start {baseline_since} is after end {baseline_until}",
            file=err,
        )
        return 2
    if current_since > current_until:
        print(
            f"error: current start {current_since} is after end {current_until}",
            file=err,
        )
        return 2
    if baseline_until >= current_since:
        print(
            "error: baseline must end before current starts ("
            f"baseline ends {baseline_until}, current starts {current_since}); "
            "periods must not overlap or touch",
            file=err,
        )
        return 2
    return 0


def _run_compare_periods(args: argparse.Namespace, out: Any, err: Any) -> int:
    periods = getattr(args, "period", None)
    if not periods or len(periods) != 2:
        print("error: exactly two --period arguments required", file=err)
        return 2
    try:
        baseline_since, baseline_until = _parse_period(periods[0])
        current_since, current_until = _parse_period(periods[1])
    except argparse.ArgumentTypeError as error:
        print(f"error: {error}", file=err)
        return 2
    rc = _validate_periods(
        baseline_since, baseline_until,
        current_since, current_until,
        err,
    )
    if rc:
        return rc
    db_path = _resolve_db(args.db)
    if not db_path.exists():
        print(f"error: no database at {db_path} (run 'ashiato build' first)", file=err)
        return 1
    connection = connect(db_path, read_only=True)
    try:
        assert_readable(connection)
        report = compare_periods_fn(
            connection,
            baseline_since=baseline_since,
            baseline_until=baseline_until,
            current_since=current_since,
            current_until=current_until,
        )
    except SchemaOutOfDate as error:
        print(f"error: {error}", file=err)
        return 1
    except duckdb.Error as error:
        print(f"error: {error}", file=err)
        return 1
    finally:
        connection.close()

    if args.format == "json":
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str), file=out)
    else:
        _print_compare_table(report, out)
    return 0


def _cell_compare(value: Any) -> str:
    """Format a cell for compare-periods table: None -> n/a."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _print_compare_table(report: dict[str, Any], out: Any) -> None:
    """Table rendering for compare-periods."""
    periods = report["periods"]
    b = periods["baseline"]
    c = periods["current"]
    print(
        f"baseline: {b['since']} .. {b['until']}",
        file=out,
    )
    print(
        f"current:  {c['since']} .. {c['until']}",
        file=out,
    )
    # Use coverage totals from the report (unique sessions, total calls)
    b_tc = b.get("tool_calls", 0)
    b_sess = b.get("sessions", 0)
    c_tc = c.get("tool_calls", 0)
    c_sess = c.get("sessions", 0)
    b_cps_val = round(b_tc / b_sess, 2) if b_sess else None
    c_cps_val = round(c_tc / c_sess, 2) if c_sess else None
    b_cps_str = f"{b_cps_val:.2f}" if b_cps_val is not None else "n/a"
    c_cps_str = f"{c_cps_val:.2f}" if c_cps_val is not None else "n/a"
    b_sess_label = "sessions" if b_sess != 1 else "session"
    c_sess_label = "sessions" if c_sess != 1 else "session"
    print(
        f"baseline: {b_tc} tool calls, {b_sess} {b_sess_label}, "
        f"{b_cps_str} calls/session",
        file=out,
    )
    print(
        f"current:  {c_tc} tool calls, {c_sess} {c_sess_label}, "
        f"{c_cps_str} calls/session",
        file=out,
    )
    columns = (
        "category",
        "b_calls", "b_calls/session",
        "b_sessions",
        "c_calls", "c_calls/session",
        "c_sessions",
        "delta", "percent",
    )
    rows: list[list[Any]] = []
    for cat in report["categories"]:
        rows.append([
            cat["name"],
            cat["baseline_tool_calls"],
            cat["baseline_calls_per_session"],
            cat["baseline_sessions"],
            cat["current_tool_calls"],
            cat["current_calls_per_session"],
            cat["current_sessions"],
            cat["tool_calls_change"],
            cat["tool_calls_percent_change"],
        ])
    # Print header
    widths = [len(name) for name in columns]
    cells = [[_cell_compare(value) for value in row] for row in rows]
    for row in cells:
        for index, text in enumerate(row):
            if index < len(widths):
                widths[index] = max(widths[index], len(text))
    print("  ".join(name.ljust(widths[i]) for i, name in enumerate(columns)).rstrip(), file=out)
    print("  ".join("-" * width for width in widths), file=out)
    for row in cells:
        print("  ".join(text.ljust(widths[i]) for i, text in enumerate(row)).rstrip(), file=out)
    print(f"({len(cells)} row{'' if len(cells) == 1 else 's'})", file=out)


def _grep_header(hit: Hit) -> str:
    ts_text = hit.ts.isoformat() if hit.ts is not None else "NULL"
    label = f"role={hit.label}" if hit.source == "event" else f"tool={hit.label}"
    return f"{ts_text}  session={hit.session_id}  {label}"


def _grep_windows(hit: Hit, *, context: int, whole: bool, all_matches: bool) -> list[str]:
    if whole:
        return [grep_visible(hit.text)]
    offsets = hit.offsets if all_matches else hit.offsets[:1]
    return [grep_window(hit.text, start, end, context) for start, end in offsets]


def _print_grep_hits(
    hits: Sequence[Hit], *, context: int, whole: bool, all_matches: bool, stream: Any
) -> None:
    for hit in hits:
        print(_grep_header(hit), file=stream)
        for text in _grep_windows(hit, context=context, whole=whole, all_matches=all_matches):
            print(text, file=stream)
    print(f"({len(hits)} hit{'' if len(hits) == 1 else 's'})", file=stream)


def _grep_structured_rows(
    hits: Sequence[Hit], *, context: int, whole: bool, all_matches: bool
) -> tuple[list[str], list[list[Any]]]:
    columns = ["id", "source", "session_id", "ts", "label", "field", "offsets", "text"]
    rows: list[list[Any]] = []
    for hit in hits:
        if whole:
            rows.append(
                [
                    hit.id,
                    hit.source,
                    hit.session_id,
                    hit.ts,
                    hit.label,
                    hit.field,
                    hit.offsets,
                    grep_visible(hit.text),
                ]
            )
            continue
        for start, end in hit.offsets if all_matches else hit.offsets[:1]:
            rows.append(
                [
                    hit.id,
                    hit.source,
                    hit.session_id,
                    hit.ts,
                    hit.label,
                    hit.field,
                    [(start, end)],
                    grep_window(hit.text, start, end, context),
                ]
            )
    return columns, rows


def _run_grep(args: argparse.Namespace, out: Any, err: Any) -> int:
    db_path = _resolve_db(args.db)
    if not db_path.exists():
        print(f"error: no database at {db_path} (run 'ashiato build' first)", file=err)
        return 2
    connection = connect(db_path, read_only=True)
    try:
        assert_readable(connection)
        hits = grep_search(
            connection,
            args.pattern,
            role=args.role,
            since=args.since,
            until=args.until,
            session=args.session,
            ignore_case=args.ignore_case,
            include_meta=args.include_meta,
            tool_calls=args.tool_calls,
            all_matches=args.all_matches,
            limit=args.limit,
        )
    except InvalidPattern as error:
        print(f"error: invalid pattern: {error}", file=err)
        return 2
    except SchemaOutOfDate as error:
        print(f"error: {error}", file=err)
        return 2
    except duckdb.Error as error:
        print(f"error: {error}", file=err)
        return 2
    finally:
        connection.close()

    if not hits:
        print("notice: no matches", file=err)
        return 1

    if args.format == "table":
        _print_grep_hits(
            hits, context=args.context, whole=args.whole, all_matches=args.all_matches, stream=out
        )
    else:
        columns, rows = _grep_structured_rows(
            hits, context=args.context, whole=args.whole, all_matches=args.all_matches
        )
        _render(columns, rows, args.format, out)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    out, err = sys.stdout, sys.stderr
    if args.command == "build":
        return _run_build(args, out, err)
    if args.command == "sql":
        return _run_sql(args, out, err)
    if args.command == "denials":
        return _run_denials(args, out, err)
    if args.command == "recalls":
        return _run_recalls(args, out, err)
    if args.command == "session-trace":
        return _run_session_trace(args, out, err)
    if args.command == "topics":
        return _run_topics(args, out, err)
    if args.command == "info":
        return _run_info(args, out, err)
    if args.command == "schema":
        return _run_schema(args, out, err)
    if args.command == "salvage":
        return _run_salvage(args, out, err)
    if args.command == "grep":
        return _run_grep(args, out, err)
    if args.command == "nominate":
        return _run_nominate(args, out, err)
    if args.command == "orphans":
        return _run_orphans(args, out, err)
    if args.command == "memory-authors":
        return _run_memory_authors(args, out, err)
    if args.command == "pending":
        return _run_pending(args, out, err)
    if args.command == "hygiene":
        return _run_hygiene(args, out, err)
    if args.command == "compare-periods":
        return _run_compare_periods(args, out, err)
    if args.command == "serve":
        return _run_serve(args, out, err)
    return _run_info(args, out, err)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
