"""Building and opening the DuckDB database.

The build is incremental: a file whose path, size and mtime are unchanged since
the last build is skipped.  A changed file has its previous rows deleted and is
re-inserted whole, so a rebuild is never additive.

Rows go in through DuckDB's JSON reader rather than one INSERT per row.  That is
not a micro-optimisation: row-at-a-time INSERT costs ~0.6 ms per row in DuckDB
whatever the table's width, which turns a real 337 MB corpus into a half-hour
build.  Writing a batch as newline-delimited JSON and reading it back is ~140x
faster.  Small batches still go the plain route, and any failure of the fast
path falls back to it, so the slow way remains the safety net.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from operator import attrgetter
from pathlib import Path

import duckdb

from ashiato.parser import (
    DEFAULT_RESULT_TEXT_LIMIT,
    DENIAL_PATTERNS,
    EVENT_COLUMNS,
    SESSION_COLUMNS,
    TOOL_CALL_COLUMNS,
    ParsedFile,
    parse_file,
)
from ashiato.schema import (
    DENIAL_FOLLOWUPS_SQL,
    DENIAL_FOLLOWUPS_VIEW,
    SCHEMA_SQL,
    TABLES,
    column_names,
    insert_sql,
    read_json_types,
)

#: Where Claude Code keeps its transcripts.
DEFAULT_SOURCE = Path("~/.claude/projects")

#: Below this many rows the temp file costs more than the row-by-row insert.
BULK_INSERT_MIN_ROWS = 8

_MTIME_TOLERANCE = 1e-6
_HASH_CHUNK = 1 << 20

_event_row = attrgetter(*EVENT_COLUMNS)
_tool_call_row = attrgetter(*TOOL_CALL_COLUMNS)
_session_row = attrgetter(*SESSION_COLUMNS)


class SchemaOutOfDate(RuntimeError):
    """The database on disk was built by a version with a different schema."""


@dataclass
class BuildResult:
    db_path: str
    n_files: int = 0
    n_processed: int = 0
    n_skipped: int = 0
    n_sessions: int = 0
    n_events: int = 0
    n_tool_calls: int = 0
    n_parse_errors: int = 0
    n_bulk_fallbacks: int = 0
    missing_sources: list[str] = field(default_factory=list)
    unreadable_files: list[str] = field(default_factory=list)
    failed_files: list[str] = field(default_factory=list)


@dataclass
class DatabaseInfo:
    db_path: str
    table_counts: dict[str, int]
    started_at: datetime | None
    ended_at: datetime | None


def default_db_path() -> Path:
    """$XDG_DATA_HOME/ashiato/ashiato.duckdb, else ~/.local/share/..."""
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "ashiato" / "ashiato.duckdb"


def connect(db_path: str | Path, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the database, creating parent directories when writing.

    Extension autoinstall is disabled: this tool reads local transcripts that
    contain secrets and must never reach the network.
    """
    path = Path(db_path).expanduser()
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path), read_only=read_only)
    connection.execute("SET autoinstall_known_extensions=false")
    return connection


def _assert_current_schema(connection: duckdb.DuckDBPyConnection) -> None:
    """Refuse a database whose tables predate the columns this version expects.

    ``CREATE TABLE IF NOT EXISTS`` leaves an older table exactly as it found it,
    so a column added since that database was built is simply absent -- and the
    first thing to notice would be the view below failing to bind, with a
    message about a column nobody asked for.  Say what actually happened
    instead.  Rebuilding is the fix: the missing values are derived from the
    transcripts, and the incremental build would skip every unchanged file.
    """
    for table in TABLES:
        rows = connection.execute(f"PRAGMA table_info('{table}')").fetchall()
        actual = tuple(row[1] for row in rows)
        expected = column_names(table)
        if actual != expected:
            missing = [name for name in expected if name not in actual]
            detail = (
                f"is missing {missing}"
                if missing
                else f"has {list(actual)} where {list(expected)} was expected"
            )
            raise SchemaOutOfDate(
                f"table '{table}' {detail}: this database was built by a different version "
                "of ashiato -- delete the database file and build again"
            )


def assert_readable(connection: duckdb.DuckDBPyConnection) -> None:
    """Refuse to query a database whose schema this version cannot bind against.

    ``build`` already says what to do about an out-of-date database; the read
    path has to say the same thing, or ``ashiato denials`` against a database
    built before the view existed reports DuckDB's bare "Table with name
    denial_followups does not exist!" and no way out of it.  Deciding this on
    open rather than by inspecting a failed query is what keeps a plain SQL typo
    the user's own error: only ashiato's own tables and view are checked here.
    """
    _assert_current_schema(connection)
    row = connection.execute(
        "SELECT count(*) FROM duckdb_views() WHERE view_name = ?", [DENIAL_FOLLOWUPS_VIEW]
    ).fetchone()
    if not row or not row[0]:
        raise SchemaOutOfDate(
            f"view '{DENIAL_FOLLOWUPS_VIEW}' is missing: this database was built by a "
            "different version of ashiato -- delete the database file and build again"
        )


def create_schema(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(SCHEMA_SQL)
    _assert_current_schema(connection)
    connection.execute(DENIAL_FOLLOWUPS_SQL)


def iter_transcripts(sources: Sequence[str | Path]) -> tuple[list[Path], list[str]]:
    """(transcript files, sources that do not exist).

    Directories are searched recursively for ``*.jsonl``.  The result is sorted
    and deduplicated so a build is reproducible regardless of walk order.
    """
    found: dict[str, Path] = {}
    missing: list[str] = []
    for source in sources:
        path = Path(source).expanduser()
        if path.is_dir():
            for candidate in path.rglob("*.jsonl"):
                if candidate.is_file():
                    found[str(candidate.resolve())] = candidate
        elif path.is_file():
            found[str(path.resolve())] = path
        else:
            missing.append(str(path))
    return [found[key] for key in sorted(found)], missing


def _content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _known_sources(connection: duckdb.DuckDBPyConnection) -> dict[str, tuple[int, float]]:
    rows = connection.execute("SELECT file_path, size_bytes, mtime FROM source_files").fetchall()
    return {row[0]: (row[1], row[2]) for row in rows}


def _delete_file_rows(connection: duckdb.DuckDBPyConnection, file_path: str) -> None:
    for table in TABLES:
        connection.execute(f'DELETE FROM "{table}" WHERE file_path = ?', [file_path])


def _jsonable(value: object) -> object:
    return value.isoformat() if isinstance(value, datetime) else value


def _bulk_insert(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    rows: Sequence[Sequence[object]],
    scratch: Path,
) -> None:
    names = column_names(table)
    path = scratch / f"{table}.ndjson"
    # errors="replace" so a lone surrogate smuggled in through a transcript
    # cannot fail the write; it is already how the transcript itself was read.
    with open(path, "w", encoding="utf-8", errors="replace") as handle:
        for row in rows:
            record = {name: _jsonable(value) for name, value in zip(names, row, strict=True)}
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    quoted = ", ".join(f'"{name}"' for name in names)
    connection.execute(
        f'INSERT INTO "{table}" ({quoted}) SELECT {quoted} FROM read_json(?, '
        f"format='newline_delimited', columns=?)",
        [str(path), read_json_types(table)],
    )


def _insert_rows(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    rows: Sequence[Sequence[object]],
    *,
    scratch: Path | None,
) -> None:
    if not rows:
        return
    if scratch is not None and len(rows) >= BULK_INSERT_MIN_ROWS:
        _bulk_insert(connection, table, rows, scratch)
    else:
        connection.executemany(insert_sql(table), [list(row) for row in rows])


def _insert_parsed(
    connection: duckdb.DuckDBPyConnection,
    parsed: ParsedFile,
    *,
    stat: os.stat_result,
    content_hash: str,
    built_at: datetime,
    scratch: Path | None,
) -> None:
    sessions = [_session_row(parsed.session)] if parsed.session is not None else []
    _insert_rows(connection, "sessions", sessions, scratch=scratch)
    _insert_rows(
        connection, "events", [_event_row(event) for event in parsed.events], scratch=scratch
    )
    _insert_rows(
        connection,
        "tool_calls",
        [_tool_call_row(call) for call in parsed.tool_calls],
        scratch=scratch,
    )
    _insert_rows(
        connection,
        "source_files",
        [
            (
                parsed.file_path,
                stat.st_size,
                stat.st_mtime,
                content_hash,
                len(parsed.events),
                len(parsed.tool_calls),
                parsed.n_parse_errors,
                built_at,
            )
        ],
        scratch=scratch,
    )


def _store_file(
    connection: duckdb.DuckDBPyConnection,
    parsed: ParsedFile,
    *,
    key: str,
    replace: bool,
    stat: os.stat_result,
    content_hash: str,
    built_at: datetime,
    scratch: Path,
    result: BuildResult,
) -> bool:
    """Write one file's rows, retrying without the bulk path if that fails."""
    for attempt in (scratch, None):
        connection.execute("BEGIN TRANSACTION")
        try:
            if replace:
                _delete_file_rows(connection, key)
            _insert_parsed(
                connection,
                parsed,
                stat=stat,
                content_hash=content_hash,
                built_at=built_at,
                scratch=attempt,
            )
            connection.execute("COMMIT")
            return True
        except duckdb.Error:
            connection.execute("ROLLBACK")
            if attempt is None:
                result.failed_files.append(key)
                return False
            result.n_bulk_fallbacks += 1
    return False


def build(
    sources: Sequence[str | Path],
    db_path: str | Path,
    *,
    denial_patterns: Sequence[str] = DENIAL_PATTERNS,
    result_text_limit: int = DEFAULT_RESULT_TEXT_LIMIT,
) -> BuildResult:
    """Parse every transcript under *sources* into the database at *db_path*."""
    result = BuildResult(db_path=str(Path(db_path).expanduser()))
    files, result.missing_sources = iter_transcripts(sources)
    result.n_files = len(files)

    connection = connect(db_path)
    try:
        create_schema(connection)
        known = _known_sources(connection)
        built_at = datetime.now(UTC).replace(tzinfo=None)

        with tempfile.TemporaryDirectory(prefix="ashiato-") as tmp_dir:
            scratch = Path(tmp_dir)
            for path in files:
                key = str(path.resolve())
                try:
                    stat = path.stat()
                except OSError:
                    result.unreadable_files.append(key)
                    continue

                previous = known.get(key)
                if (
                    previous is not None
                    and previous[0] == stat.st_size
                    and abs((previous[1] or 0.0) - stat.st_mtime) < _MTIME_TOLERANCE
                ):
                    result.n_skipped += 1
                    continue

                try:
                    parsed = parse_file(
                        path,
                        denial_patterns=denial_patterns,
                        result_text_limit=result_text_limit,
                    )
                    content_hash = _content_hash(path)
                except OSError:
                    result.unreadable_files.append(key)
                    continue

                stored = _store_file(
                    connection,
                    parsed,
                    key=key,
                    replace=previous is not None,
                    stat=stat,
                    content_hash=content_hash,
                    built_at=built_at,
                    scratch=scratch,
                    result=result,
                )
                if not stored:
                    continue

                result.n_processed += 1
                result.n_events += len(parsed.events)
                result.n_tool_calls += len(parsed.tool_calls)
                result.n_parse_errors += parsed.n_parse_errors
                result.n_sessions += 1 if parsed.session is not None else 0
    finally:
        connection.close()
    return result


def database_info(db_path: str | Path) -> DatabaseInfo:
    """Row counts per table and the time window the transcripts cover."""
    path = Path(db_path).expanduser()
    connection = connect(path, read_only=True)
    try:
        counts: dict[str, int] = {}
        for table in TABLES:
            row = connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()
            counts[table] = row[0] if row else 0
        window = connection.execute(
            "SELECT min(started_at), max(ended_at) FROM sessions"
        ).fetchone()
        started_at, ended_at = window if window else (None, None)
    finally:
        connection.close()
    return DatabaseInfo(
        db_path=str(path), table_counts=counts, started_at=started_at, ended_at=ended_at
    )
