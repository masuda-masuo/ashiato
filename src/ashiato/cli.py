"""The ``ashiato`` command line: build, sql, denials, info."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb

from ashiato import __version__
from ashiato.build import (
    DEFAULT_SOURCE,
    SchemaOutOfDate,
    build,
    connect,
    database_info,
    default_db_path,
)
from ashiato.schema import DENIAL_FOLLOWUPS_VIEW

FORMATS = ("table", "json", "csv")

#: Enough denials to read in one screen; ``--limit 0`` asks for all of them.
DEFAULT_DENIAL_LIMIT = 50


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

    info_parser = subparsers.add_parser("info", help="describe the database")
    info_parser.add_argument("--db", metavar="PATH", help="database path")

    return parser


def _row_limit(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be 0 or greater")
    return number


def _resolve_db(value: str | None) -> Path:
    return Path(value).expanduser() if value else default_db_path()


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
        cursor = connection.execute(query, list(params))
        columns = [description[0] for description in cursor.description or []]
        rows = cursor.fetchall() if columns else []
    except duckdb.Error as error:
        print(f"error: {error}", file=err)
        return 1
    finally:
        connection.close()

    _render(columns, rows, fmt, out)
    return 0


def _run_build(args: argparse.Namespace, out: Any, err: Any) -> int:
    sources = args.source or [str(DEFAULT_SOURCE)]
    db_path = _resolve_db(args.db)
    try:
        result = build(sources, db_path)
    except SchemaOutOfDate as error:
        print(f"error: {error}", file=err)
        return 1

    for source in result.missing_sources:
        print(f"warning: source not found: {source}", file=err)
    for path in result.unreadable_files:
        print(f"warning: could not read: {path}", file=err)
    for path in result.failed_files:
        print(f"warning: could not store: {path}", file=err)

    print(f"database: {result.db_path}", file=out)
    print(
        f"files: {result.n_processed} processed, {result.n_skipped} skipped (unchanged), "
        f"{result.n_files} found",
        file=out,
    )
    print(
        f"rows: {result.n_sessions} sessions, {result.n_events} events, "
        f"{result.n_tool_calls} tool calls",
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


def _run_info(args: argparse.Namespace, out: Any, err: Any) -> int:
    db_path = _resolve_db(args.db)
    if not db_path.exists():
        print(f"error: no database at {db_path} (run 'ashiato build' first)", file=err)
        return 1
    try:
        info = database_info(db_path)
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
    return _run_info(args, out, err)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
