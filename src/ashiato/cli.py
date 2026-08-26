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
    DEFAULT_SOURCE,
    SchemaOutOfDate,
    assert_readable,
    build,
    connect,
    database_info,
    default_db_path,
)
from ashiato.grep import DEFAULT_CONTEXT as DEFAULT_GREP_CONTEXT
from ashiato.grep import DEFAULT_LIMIT as DEFAULT_GREP_LIMIT
from ashiato.grep import Hit, InvalidPattern
from ashiato.grep import search as grep_search
from ashiato.grep import visible as grep_visible
from ashiato.grep import window as grep_window
from ashiato.nominate import run as nominate_run
from ashiato.salvage import DEFAULT_LIMIT as DEFAULT_SALVAGE_LIMIT
from ashiato.salvage import DEFAULT_WINDOW_MINUTES, default_kaiba_db_path, nominate, open_kaiba
from ashiato.schema import (
    DENIAL_FOLLOWUPS_VIEW,
    RECALL_FOLLOWUPS_VIEW,
    REQUIRED_VIEWS,
    TABLE_COLUMNS,
    TABLES,
    VIEW_COLUMNS,
)

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


def _resolve_db(value: str | None) -> Path:
    return Path(value).expanduser() if value else default_db_path()


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
    kaiba_db_path = Path(args.kaiba_db).expanduser() if args.kaiba_db else None
    db_path = _resolve_db(args.db)
    try:
        result = build(
            sources,
            db_path,
            opencode_sources=opencode_sources,
            cursor_sources=cursor_sources,
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
    query = f'SELECT * FROM "{DENIAL_FOLLOWUPS_VIEW}"'
    params: list[Any] = []
    if args.session:
        query += " WHERE session_id = ?"
        params.append(args.session)
    # Newest first, as asked for -- but timestamps tie (and can be NULL), so the
    # session and the line number settle the rest and two runs agree.
    query += " ORDER BY ts DESC NULLS LAST, session_id, seq DESC"
    if args.limit:
        query += " LIMIT ?"
        params.append(args.limit)
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
    if info.sources is None and info.opencode_sources is None and info.cursor_sources is None:
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
    return _run_info(args, out, err)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
