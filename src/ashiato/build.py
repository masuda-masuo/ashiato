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

Three source formats are supported: Claude Code transcripts (``*.jsonl``,
``sources`` / ``--source``), opencode job event streams (``*.ndjson``,
``opencode_sources`` / ``--opencode-source``), and Cursor agent transcripts
(``*.jsonl``, ``cursor_sources`` / ``--cursor-source``).  They are separate,
explicit source lists rather than one list ashiato sniffs file-by-file: the
three live in unrelated directory trees on a real machine, and explicit
lists mean a Claude Code projects directory, an opencode jobs directory, and
a Cursor agent-transcripts directory can share a build without any of them
dragging files into another's parser by mistake -- true even for Claude Code
and Cursor, which share the same ``*.jsonl`` extension.  A Cursor recall
call also needs kaiba's own ``recalls`` ledger (``kaiba_db_path`` /
``--kaiba-db``) to fill in its output and timestamp; see
:func:`ashiato.recall.extract_from_cursor` for why.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from operator import attrgetter
from pathlib import Path

import duckdb

from ashiato.cursor import ParsedCursorFile
from ashiato.cursor import parse_file as parse_cursor_file
from ashiato.opencode import ParsedOpenCodeFile
from ashiato.opencode import parse_file as parse_opencode_file
from ashiato.parser import (
    DEFAULT_RESULT_TEXT_LIMIT,
    DENIAL_PATTERNS,
    EVENT_COLUMNS,
    SESSION_COLUMNS,
    TOOL_CALL_COLUMNS,
    ParsedFile,
    parse_file,
)
from ashiato.recall import (
    RECALL_CALL_COLUMNS,
    RecallCall,
    extract_from_claude,
    extract_from_cursor,
    extract_from_opencode,
)
from ashiato.salvage import default_kaiba_db_path, open_kaiba, parse_kaiba_ts
from ashiato.schema import (
    DENIAL_FOLLOWUPS_SQL,
    FORMAT_VERSION,
    INFO_TABLES,
    META_CURSOR_SOURCES_KEY,
    META_FORMAT_KEY,
    META_OPENCODE_SOURCES_KEY,
    META_SCHEMA_SQL,
    META_SOURCES_KEY,
    META_TABLE,
    RECALL_FOLLOWUPS_SQL,
    REQUIRED_VIEWS,
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
_recall_call_row = attrgetter(*RECALL_CALL_COLUMNS)


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
    n_recall_calls: int = 0
    n_parse_errors: int = 0
    n_bulk_fallbacks: int = 0
    missing_sources: list[str] = field(default_factory=list)
    unreadable_files: list[str] = field(default_factory=list)
    failed_files: list[str] = field(default_factory=list)
    #: The kaiba db path, when Cursor sources were given but the db at that
    #: path could not be opened (missing, corrupt, or missing ``recalls``) --
    #: ``None`` when no Cursor sources were given, or the db opened fine.
    #: Rows are still produced with NULL output/ts; this is just so the CLI
    #: can tell the caller why.
    kaiba_db_unavailable: str | None = None


@dataclass
class DatabaseInfo:
    db_path: str
    table_counts: dict[str, int]
    started_at: datetime | None
    ended_at: datetime | None
    #: Ingested roots, per kind.  Each is a list of (root_path, file_count) pairs.
    #: ``None`` means the database was built before roots were recorded.
    sources: list[tuple[str, int]] | None = None
    opencode_sources: list[tuple[str, int]] | None = None
    cursor_sources: list[tuple[str, int]] | None = None
    #: Number of files under the recorded roots that are not in source_files,
    #: or have a different size/mtime.  ``None`` when roots are unknown.
    freshness_gap: int | None = None


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


def _meta_format_version(connection: duckdb.DuckDBPyConnection) -> str | None:
    """The stored :data:`FORMAT_VERSION`, or ``None`` when the marker is absent.

    A pre-marker database simply has no ``ashiato_meta`` table, so a missing
    table must read as "no version", not as a catalog error.
    """
    tables = {
        row[0]
        for row in connection.execute("SELECT table_name FROM duckdb_tables()").fetchall()
    }
    if META_TABLE not in tables:
        return None
    row = connection.execute(
        f'SELECT value FROM "{META_TABLE}" WHERE key = ?', [META_FORMAT_KEY]
    ).fetchone()
    return row[0] if row else None


def _assert_format_version(connection: duckdb.DuckDBPyConnection) -> None:
    """Refuse a database whose stored rows were derived under older rules.

    ``outcome`` is a stored column, so a semantic change in how it is decided --
    the denial patterns moving from substring to anchored-prefix matching -- is
    invisible to a column comparison.  The incremental build would skip every
    unchanged file and leave the old rows mixed with the new.  The format marker
    makes that detectable: no marker, or a different one, means rebuild.
    """
    if _meta_format_version(connection) != str(FORMAT_VERSION):
        raise SchemaOutOfDate(
            "the stored tool_calls rows were derived under older outcome rules: "
            "this database was built by a different version of ashiato -- "
            "delete the database file and build again"
        )


def _table_columns(connection: duckdb.DuckDBPyConnection, table: str) -> tuple[str, ...]:
    """The actual column names of *table*, or ``()`` when it does not exist.

    ``PRAGMA table_info`` raises ``CatalogException`` for a table that is not
    there at all -- which is exactly the shape of an old database predating a
    table this version expects (``recall_calls``, added in FORMAT_VERSION 3).
    An absent table is reported the same way a table missing every expected
    column would be: as "missing everything", not as an unrelated crash.
    """
    try:
        rows = connection.execute(f"PRAGMA table_info('{table}')").fetchall()
    except duckdb.CatalogException:
        return ()
    return tuple(row[1] for row in rows)


def _assert_current_schema(connection: duckdb.DuckDBPyConnection) -> None:
    """Refuse a database that predates the columns or rules this version expects.

    ``CREATE TABLE IF NOT EXISTS`` leaves an older table exactly as it found it,
    so a column added since that database was built is simply absent -- and the
    first thing to notice would be the view below failing to bind, with a
    message about a column nobody asked for.  Say what actually happened
    instead.  Rebuilding is the fix: the missing values are derived from the
    transcripts, and the incremental build would skip every unchanged file.

    Columns are not the whole story: rows carry derived values (``outcome``)
    whose rules can change without any column changing, so the format marker is
    checked here too.
    """
    for table in TABLES:
        actual = _table_columns(connection, table)
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
    _assert_format_version(connection)


def assert_readable(connection: duckdb.DuckDBPyConnection) -> None:
    """Refuse to query a database whose schema this version cannot bind against.

    ``build`` already says what to do about an out-of-date database; the read
    path has to say the same thing, or ``ashiato denials`` against a database
    built before the view existed reports DuckDB's bare "Table with name
    denial_followups does not exist!" and no way out of it.  Deciding this on
    open rather than by inspecting a failed query is what keeps a plain SQL typo
    the user's own error: only ashiato's own tables and views are checked here.
    """
    _assert_current_schema(connection)
    for view in REQUIRED_VIEWS:
        row = connection.execute(
            "SELECT count(*) FROM duckdb_views() WHERE view_name = ?", [view]
        ).fetchone()
        if not row or not row[0]:
            raise SchemaOutOfDate(
                f"view '{view}' is missing: this database was built by a "
                "different version of ashiato -- delete the database file and build again"
            )


def create_schema(
    connection: duckdb.DuckDBPyConnection,
    *,
    sources: Sequence[str | Path] | None = None,
    opencode_sources: Sequence[str | Path] | None = None,
    cursor_sources: Sequence[str | Path] | None = None,
) -> None:
    """Create the tables, format marker, views, and recorded roots on a fresh database.

    An existing database that already has ashiato's tables must be current in
    columns and format marker before anything is created: ``CREATE TABLE IF
    NOT EXISTS`` would otherwise leave its old rows untouched and stamp them
    as current.  A database with no ashiato tables at all is a fresh one and
    is initialised with the marker.

    The DDL, the marker stamp, and the recorded roots run as one transaction,
    and so do the views on top of them: DuckDB DDL is transactional, so a crash
    at any point in here rolls the whole lot back and the file stays empty --
    exactly what a fresh ``build`` expects.  Without the transaction, a crash
    between the ``CREATE TABLE`` statements and the marker insert would leave
    ashiato tables with no marker, and the next build would refuse an empty,
    perfectly rebuildable file.

    When *sources*, *opencode_sources*, or *cursor_sources* are provided, they
    are resolved to absolute paths and stored in :data:`META_TABLE` as JSON
    arrays under :data:`META_SOURCES_KEY`, :data:`META_OPENCODE_SOURCES_KEY`,
    and :data:`META_CURSOR_SOURCES_KEY`.  A root that matched no files is still
    recorded -- it explains an absence.  When any sequence is ``None`` (the
    default), the corresponding key is not written, preserving whatever value
    may already be in the table (for callers that only create the schema
    without a full build).
    """
    existing = {
        row[0]
        for row in connection.execute("SELECT table_name FROM duckdb_tables()").fetchall()
    }
    if existing & set(TABLES):
        # Full check, not just the marker: a database with current columns but
        # an old marker is refused before anything is written, and so is one
        # whose columns predate this version -- the view below would otherwise
        # fail to bind and blame a column nobody asked for.
        _assert_current_schema(connection)
    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute(SCHEMA_SQL)
        connection.execute(META_SCHEMA_SQL)
        connection.execute(
            f'INSERT OR REPLACE INTO "{META_TABLE}" (key, value) VALUES (?, ?)',
            [META_FORMAT_KEY, str(FORMAT_VERSION)],
        )
        if sources is not None:
            resolved = [str(Path(s).expanduser().resolve()) for s in sources]
            connection.execute(
                f'INSERT OR REPLACE INTO "{META_TABLE}" (key, value) VALUES (?, ?)',
                [META_SOURCES_KEY, json.dumps(resolved)],
            )
        if opencode_sources is not None:
            resolved = [str(Path(s).expanduser().resolve()) for s in opencode_sources]
            connection.execute(
                f'INSERT OR REPLACE INTO "{META_TABLE}" (key, value) VALUES (?, ?)',
                [META_OPENCODE_SOURCES_KEY, json.dumps(resolved)],
            )
        if cursor_sources is not None:
            resolved = [str(Path(s).expanduser().resolve()) for s in cursor_sources]
            connection.execute(
                f'INSERT OR REPLACE INTO "{META_TABLE}" (key, value) VALUES (?, ?)',
                [META_CURSOR_SOURCES_KEY, json.dumps(resolved)],
            )
        connection.execute(DENIAL_FOLLOWUPS_SQL)
        connection.execute(RECALL_FOLLOWUPS_SQL)
        connection.execute("COMMIT")
    except duckdb.Error:
        connection.execute("ROLLBACK")
        raise


def _read_meta_json_list(connection: duckdb.DuckDBPyConnection, key: str) -> list[str] | None:
    """Read a JSON array from META_TABLE, or None if the key is absent."""
    tables = {
        row[0]
        for row in connection.execute("SELECT table_name FROM duckdb_tables()").fetchall()
    }
    if META_TABLE not in tables:
        return None
    row = connection.execute(
        f'SELECT value FROM "{META_TABLE}" WHERE key = ?', [key]
    ).fetchone()
    if row is None or row[0] is None:
        return None
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return None


def _count_files_per_root(
    known: dict[str, tuple[int, float]], roots: list[str]
) -> list[tuple[str, int]]:
    """Count how many files from source_files fall under each root.

    A file is attributed to the *most specific* (longest) root that contains it.
    This makes the per-root counts deterministic regardless of the order in
    which roots are passed. Roots that match no files still appear with count 0.
    """
    # Sort roots by length (descending) so the most specific root is checked first.
    # We keep the original order for equal-length roots to preserve stability.
    sorted_roots = sorted(roots, key=len, reverse=True)
    counts: dict[str, int] = {root: 0 for root in roots}
    for file_path in known:
        file_p = Path(file_path)
        for root in sorted_roots:
            try:
                file_p.relative_to(root)
                counts[root] += 1
                break  # Attributed to the most specific containing root
            except ValueError:
                continue
    return [(root, counts[root]) for root in roots]


def _compute_freshness_gap(
    connection: duckdb.DuckDBPyConnection,
    sources: list[str],
    opencode_sources: list[str],
    cursor_sources: list[str],
) -> int:
    """Count files under roots that are not in source_files or have different size/mtime.

    This mirrors the condition in _is_unchanged: a file is "fresh" if it exists in
    source_files with the same size and mtime (within tolerance). The gap is the
    number of files that would be re-read by build right now.
    """
    known = _known_sources(connection)
    gap = 0

    # Check Claude Code sources (*.jsonl)
    for file_path in _iter_sources(sources, "*.jsonl")[0]:
        key = str(file_path.resolve())
        try:
            stat = file_path.stat()
        except OSError:
            continue
        if not _is_unchanged(known, key, stat):
            gap += 1

    # Check opencode sources (*.ndjson)
    for file_path in _iter_sources(opencode_sources, "*.ndjson")[0]:
        key = str(file_path.resolve())
        try:
            stat = file_path.stat()
        except OSError:
            continue
        if not _is_unchanged(known, key, stat):
            gap += 1

    # Check Cursor sources (*.jsonl)
    for file_path in _iter_sources(cursor_sources, "*.jsonl")[0]:
        key = str(file_path.resolve())
        try:
            stat = file_path.stat()
        except OSError:
            continue
        if not _is_unchanged(known, key, stat):
            gap += 1

    return gap


def _iter_sources(sources: Sequence[str | Path], pattern: str) -> tuple[list[Path], list[str]]:
    """(matching files, sources that do not exist).

    Directories are searched recursively for *pattern*.  The result is sorted
    and deduplicated so a build is reproducible regardless of walk order.  A
    source that is itself a file is accepted whatever its name -- the caller
    already said what it is by which source list it went in.
    """
    found: dict[str, Path] = {}
    missing: list[str] = []
    for source in sources:
        path = Path(source).expanduser()
        if path.is_dir():
            for candidate in path.rglob(pattern):
                if candidate.is_file():
                    found[str(candidate.resolve())] = candidate
        elif path.is_file():
            found[str(path.resolve())] = path
        else:
            missing.append(str(path))
    return [found[key] for key in sorted(found)], missing


def iter_transcripts(sources: Sequence[str | Path]) -> tuple[list[Path], list[str]]:
    """(Claude Code transcript files, sources that do not exist).

    Directories are searched recursively for ``*.jsonl``.
    """
    return _iter_sources(sources, "*.jsonl")


def iter_opencode_sources(sources: Sequence[str | Path]) -> tuple[list[Path], list[str]]:
    """(opencode events.ndjson files, sources that do not exist).

    Directories are searched recursively for ``*.ndjson``.
    """
    return _iter_sources(sources, "*.ndjson")


def iter_cursor_sources(sources: Sequence[str | Path]) -> tuple[list[Path], list[str]]:
    """(Cursor agent-transcript files, sources that do not exist).

    Directories are searched recursively for ``*.jsonl`` -- the same
    extension Claude Code transcripts use, but Cursor keeps its own directory
    tree (``~/.cursor/projects/<project>/agent-transcripts/<id>/<id>.jsonl``),
    so the two source lists never see each other's files in practice.
    """
    return _iter_sources(sources, "*.jsonl")


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
    recall_rows: Sequence[RecallCall],
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
        "recall_calls",
        [_recall_call_row(row) for row in recall_rows],
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


def _insert_opencode_parsed(
    connection: duckdb.DuckDBPyConnection,
    parsed: ParsedOpenCodeFile,
    recall_rows: Sequence[RecallCall],
    *,
    stat: os.stat_result,
    content_hash: str,
    built_at: datetime,
    scratch: Path | None,
) -> None:
    """The opencode counterpart of :func:`_insert_parsed`.

    Only ``recall_calls`` and ``source_files`` are touched: general-purpose
    ingestion of opencode events into ``sessions`` / ``events`` / ``tool_calls``
    is out of scope, so those counts are honestly zero rather than borrowed
    from a table this format never populates.
    """
    _insert_rows(
        connection,
        "recall_calls",
        [_recall_call_row(row) for row in recall_rows],
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
                0,
                0,
                parsed.n_parse_errors,
                built_at,
            )
        ],
        scratch=scratch,
    )


def _insert_cursor_parsed(
    connection: duckdb.DuckDBPyConnection,
    parsed: ParsedCursorFile,
    recall_rows: Sequence[RecallCall],
    *,
    stat: os.stat_result,
    content_hash: str,
    built_at: datetime,
    scratch: Path | None,
) -> None:
    """The Cursor counterpart of :func:`_insert_parsed` / :func:`_insert_opencode_parsed`.

    Same scope as opencode: only ``recall_calls`` and ``source_files`` are
    touched, never ``sessions`` / ``events`` / ``tool_calls``.
    """
    _insert_rows(
        connection,
        "recall_calls",
        [_recall_call_row(row) for row in recall_rows],
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
                0,
                0,
                parsed.n_parse_errors,
                built_at,
            )
        ],
        scratch=scratch,
    )


def _fetch_cursor_kaiba_recalls(
    connection: sqlite3.Connection,
) -> dict[str, list[tuple[datetime | None, str]]]:
    """query -> [(created_at, output), ...] for every ``agent = 'cursor'`` recall row.

    ``output`` is the joined ``content`` of every ``matches[*].id`` a row
    names, in ``matches`` order -- what kaiba actually returned.  A
    ``matches`` id with no matching ``conclusions`` row (retired and purged,
    or simply absent) contributes nothing rather than failing the row.  Rows
    are ordered by ``created_at`` so the n-th occurrence of a query in a
    transcript can pair with the n-th row here, per :func:`ashiato.recall.extract_from_cursor`.
    """
    conclusions: dict[object, str] = dict(
        connection.execute("SELECT id, content FROM conclusions").fetchall()
    )
    by_query: dict[str, list[tuple[datetime | None, str]]] = {}
    rows = connection.execute(
        "SELECT created_at, query, matches FROM recalls WHERE agent = 'cursor' ORDER BY created_at"
    ).fetchall()
    for created_at, query, matches_json in rows:
        if not isinstance(query, str):
            continue
        try:
            matches = json.loads(matches_json) if matches_json else []
        except ValueError:
            matches = []
        if not isinstance(matches, list):
            matches = []
        pieces = [
            conclusions[match["id"]]
            for match in matches
            if isinstance(match, dict) and match.get("id") in conclusions
        ]
        by_query.setdefault(query, []).append((parse_kaiba_ts(created_at), "\n".join(pieces)))
    return by_query


def _store_file(
    connection: duckdb.DuckDBPyConnection,
    *,
    key: str,
    replace: bool,
    scratch: Path,
    insert: Callable[[Path | None], None],
    result: BuildResult,
) -> bool:
    """Write one file's rows, retrying without the bulk path if that fails."""
    for attempt in (scratch, None):
        connection.execute("BEGIN TRANSACTION")
        try:
            if replace:
                _delete_file_rows(connection, key)
            insert(attempt)
            connection.execute("COMMIT")
            return True
        except duckdb.Error:
            connection.execute("ROLLBACK")
            if attempt is None:
                result.failed_files.append(key)
                return False
            result.n_bulk_fallbacks += 1
    return False


def _is_unchanged(
    known: dict[str, tuple[int, float]], key: str, stat: os.stat_result
) -> bool:
    previous = known.get(key)
    return (
        previous is not None
        and previous[0] == stat.st_size
        and abs((previous[1] or 0.0) - stat.st_mtime) < _MTIME_TOLERANCE
    )


def build(
    sources: Sequence[str | Path],
    db_path: str | Path,
    *,
    opencode_sources: Sequence[str | Path] = (),
    cursor_sources: Sequence[str | Path] = (),
    kaiba_db_path: str | Path | None = None,
    denial_patterns: Sequence[str] = DENIAL_PATTERNS,
    result_text_limit: int = DEFAULT_RESULT_TEXT_LIMIT,
) -> BuildResult:
    """Parse every transcript under *sources* / *opencode_sources* / *cursor_sources*.

    *kaiba_db_path* (default :func:`ashiato.salvage.default_kaiba_db_path`) is
    only ever opened when *cursor_sources* actually resolves to at least one
    file: a Cursor recall call's output/timestamp is reconstructed by joining
    its query against kaiba's own ``recalls`` ledger (see
    :func:`ashiato.recall.extract_from_cursor`), and a build with no Cursor
    sources has no such join to perform.  A missing or unreadable kaiba db
    does not fail the build -- rows are produced with NULL output/ts, and
    :attr:`BuildResult.kaiba_db_unavailable` records the path so the CLI can
    say so.
    """
    result = BuildResult(db_path=str(Path(db_path).expanduser()))
    claude_files, claude_missing = iter_transcripts(sources)
    opencode_files, opencode_missing = iter_opencode_sources(opencode_sources)
    cursor_files, cursor_missing = iter_cursor_sources(cursor_sources)
    result.missing_sources = claude_missing + opencode_missing + cursor_missing
    result.n_files = len(claude_files) + len(opencode_files) + len(cursor_files)

    kaiba_recalls_by_query: dict[str, list[tuple[datetime | None, str]]] = {}
    if cursor_files:
        resolved_kaiba_path = (
            Path(kaiba_db_path).expanduser()
            if kaiba_db_path is not None
            else default_kaiba_db_path()
        )
        kaiba_connection = open_kaiba(resolved_kaiba_path, probe_table="recalls")
        if kaiba_connection is None:
            result.kaiba_db_unavailable = str(resolved_kaiba_path)
        else:
            try:
                kaiba_recalls_by_query = _fetch_cursor_kaiba_recalls(kaiba_connection)
            finally:
                kaiba_connection.close()

    connection = connect(db_path)
    try:
        create_schema(
            connection,
            sources=sources,
            opencode_sources=opencode_sources,
            cursor_sources=cursor_sources,
        )
        known = _known_sources(connection)
        built_at = datetime.now(UTC).replace(tzinfo=None)

        with tempfile.TemporaryDirectory(prefix="ashiato-") as tmp_dir:
            scratch = Path(tmp_dir)

            for path in claude_files:
                key = str(path.resolve())
                try:
                    stat = path.stat()
                except OSError:
                    result.unreadable_files.append(key)
                    continue
                if _is_unchanged(known, key, stat):
                    result.n_skipped += 1
                    continue

                try:
                    parsed = parse_file(
                        path,
                        denial_patterns=denial_patterns,
                        result_text_limit=result_text_limit,
                    )
                    recall_rows = extract_from_claude(parsed)
                    content_hash = _content_hash(path)
                except OSError:
                    result.unreadable_files.append(key)
                    continue

                stored = _store_file(
                    connection,
                    key=key,
                    replace=key in known,
                    scratch=scratch,
                    insert=lambda attempt,
                    parsed=parsed,
                    recall_rows=recall_rows,
                    stat=stat,
                    content_hash=content_hash: _insert_parsed(
                        connection,
                        parsed,
                        recall_rows,
                        stat=stat,
                        content_hash=content_hash,
                        built_at=built_at,
                        scratch=attempt,
                    ),
                    result=result,
                )
                if not stored:
                    continue

                result.n_processed += 1
                result.n_recall_calls += len(recall_rows)
                result.n_sessions += 1 if parsed.session is not None else 0
                result.n_events += len(parsed.events)
                result.n_tool_calls += len(parsed.tool_calls)
                result.n_parse_errors += parsed.n_parse_errors

            for path in opencode_files:
                key = str(path.resolve())
                try:
                    stat = path.stat()
                except OSError:
                    result.unreadable_files.append(key)
                    continue
                if _is_unchanged(known, key, stat):
                    result.n_skipped += 1
                    continue

                try:
                    parsed_oc = parse_opencode_file(path)
                    recall_rows = extract_from_opencode(
                        parsed_oc, result_text_limit=result_text_limit
                    )
                    content_hash = _content_hash(path)
                except OSError:
                    result.unreadable_files.append(key)
                    continue

                stored = _store_file(
                    connection,
                    key=key,
                    replace=key in known,
                    scratch=scratch,
                    insert=lambda attempt,
                    parsed_oc=parsed_oc,
                    recall_rows=recall_rows,
                    stat=stat,
                    content_hash=content_hash: _insert_opencode_parsed(
                        connection,
                        parsed_oc,
                        recall_rows,
                        stat=stat,
                        content_hash=content_hash,
                        built_at=built_at,
                        scratch=attempt,
                    ),
                    result=result,
                )
                if not stored:
                    continue

                result.n_processed += 1
                result.n_recall_calls += len(recall_rows)
                result.n_parse_errors += parsed_oc.n_parse_errors

            for path in cursor_files:
                key = str(path.resolve())
                try:
                    stat = path.stat()
                except OSError:
                    result.unreadable_files.append(key)
                    continue
                if _is_unchanged(known, key, stat):
                    result.n_skipped += 1
                    continue

                try:
                    parsed_cur = parse_cursor_file(path)
                    recall_rows = extract_from_cursor(
                        parsed_cur,
                        kaiba_recalls_by_query,
                        result_text_limit=result_text_limit,
                    )
                    content_hash = _content_hash(path)
                except OSError:
                    result.unreadable_files.append(key)
                    continue

                stored = _store_file(
                    connection,
                    key=key,
                    replace=key in known,
                    scratch=scratch,
                    insert=lambda attempt,
                    parsed_cur=parsed_cur,
                    recall_rows=recall_rows,
                    stat=stat,
                    content_hash=content_hash: _insert_cursor_parsed(
                        connection,
                        parsed_cur,
                        recall_rows,
                        stat=stat,
                        content_hash=content_hash,
                        built_at=built_at,
                        scratch=attempt,
                    ),
                    result=result,
                )
                if not stored:
                    continue

                result.n_processed += 1
                result.n_recall_calls += len(recall_rows)
                result.n_parse_errors += parsed_cur.n_parse_errors
    finally:
        connection.close()
    return result


def database_info(db_path: str | Path) -> DatabaseInfo:
    """Row counts per table, the time window, ingested roots, and freshness.

    The same read gate as ``sql`` and ``denials``: a database whose schema or
    format marker this version cannot vouch for is refused, so ``info`` does
    not report a stale database as if it were current.  Counts are reported
    for :data:`ashiato.schema.INFO_TABLES`, not every table -- see that
    constant for why ``recall_calls`` is left out of this particular report.

    Roots are read from :data:`META_TABLE`.  When absent (database built before
    this feature), they are reported as ``None`` rather than empty lists, so
    the caller can distinguish "no roots were used" from "roots were not
    recorded".  The freshness gap is the number of files under the recorded
    roots that are not in ``source_files`` or have a different size/mtime --
    the same condition ``build`` uses to decide what to re-read.  When roots
    are unknown, the gap is ``None``.
    """
    path = Path(db_path).expanduser()
    connection = connect(path, read_only=True)
    try:
        assert_readable(connection)
        counts: dict[str, int] = {}
        for table in INFO_TABLES:
            row = connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()
            counts[table] = row[0] if row else 0
        window = connection.execute(
            "SELECT min(started_at), max(ended_at) FROM sessions"
        ).fetchone()
        started_at, ended_at = window if window else (None, None)

        # Read recorded roots
        sources = _read_meta_json_list(connection, META_SOURCES_KEY)
        opencode_sources = _read_meta_json_list(connection, META_OPENCODE_SOURCES_KEY)
        cursor_sources = _read_meta_json_list(connection, META_CURSOR_SOURCES_KEY)

        # If no roots recorded, return early with None for roots and gap
        if sources is None and opencode_sources is None and cursor_sources is None:
            return DatabaseInfo(
                db_path=str(path),
                table_counts=counts,
                started_at=started_at,
                ended_at=ended_at,
                sources=None,
                opencode_sources=None,
                cursor_sources=None,
                freshness_gap=None,
            )

        # Count files per root from source_files
        known = _known_sources(connection)
        source_counts = _count_files_per_root(known, sources or [])
        opencode_counts = _count_files_per_root(known, opencode_sources or [])
        cursor_counts = _count_files_per_root(known, cursor_sources or [])

        # Compute freshness gap
        freshness_gap = _compute_freshness_gap(
            connection, sources or [], opencode_sources or [], cursor_sources or []
        )

        return DatabaseInfo(
            db_path=str(path),
            table_counts=counts,
            started_at=started_at,
            ended_at=ended_at,
            sources=source_counts,
            opencode_sources=opencode_counts,
            cursor_sources=cursor_counts,
            freshness_gap=freshness_gap,
        )
    finally:
        connection.close()
