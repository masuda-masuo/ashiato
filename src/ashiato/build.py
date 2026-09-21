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

from ashiato.codex import ParsedCodexFile
from ashiato.codex import parse_file as parse_codex_file
from ashiato.cursor import (
    ParsedCursorFile,
    classify_store_result,
    parse_chat_meta,
    parse_chat_store,
    store_result_text,
)
from ashiato.cursor import parse_file as parse_cursor_file
from ashiato.opencode import ParsedOpenCodeFile
from ashiato.opencode import parse_file as parse_opencode_file
from ashiato.parser import (
    DEFAULT_RESULT_TEXT_LIMIT,
    DENIAL_PATTERNS,
    EVENT_COLUMNS,
    SESSION_COLUMNS,
    TOOL_CALL_COLUMNS,
    Event,
    ParsedFile,
    Session,
    classify_outcome,
    parse_file,
    split_tool_name,
    summarize_input,
)
from ashiato.recall import (
    CURSOR_MCP_TOOL_NAME,
    RECALL_CALL_COLUMNS,
    RecallCall,
    extract_from_claude,
    extract_from_codex,
    extract_from_cursor,
    extract_from_opencode,
)
from ashiato.salvage import default_kaiba_db_path, open_kaiba, parse_kaiba_ts
from ashiato.schema import (
    DENIAL_FOLLOWUPS_SQL,
    FORMAT_VERSION,
    INFO_TABLES,
    META_CODEX_SOURCES_KEY,
    META_CURSOR_CHATS_SOURCES_KEY,
    META_CURSOR_SOURCES_KEY,
    META_FORMAT_KEY,
    META_OPENCODE_SOURCES_KEY,
    META_SCHEMA_SQL,
    META_SOURCES_KEY,
    META_TABLE,
    RECALL_FOLLOWUPS_SQL,
    REQUIRED_VIEWS,
    SCHEMA_SQL,
    SOURCE_CODEX,
    SOURCE_CURSOR,
    SOURCE_OPENCODE,
    TABLES,
    column_names,
    insert_sql,
    read_json_types,
)

#: Where Claude Code keeps its transcripts.
DEFAULT_SOURCE = Path("~/.claude/projects")

#: Where Codex keeps its sessions.
DEFAULT_CODEX_SOURCE = Path("~/.codex/sessions")

#: The chats layout Cursor keeps next to its agent transcripts: one small
#: ``meta.json`` per session, exactly two directories below the chats root
#: (``~/.cursor/chats/<workspace-hash>/<session-uuid>/meta.json``).
CURSOR_CHATS_META_PATTERN = "*/*/meta.json"

#: The second Cursor block name that calls an MCP tool, alongside
#: :data:`CURSOR_MCP_TOOL_NAME` (``CallMcpTool``).  Cursor records both block
#: names with the same meaning but different key spellings: ``CallMcpTool``
#: carries ``server`` / ``toolName`` in its input, ``CallDynamicTool`` spells
#: the same identity as ``namespace`` / ``toolName``.  Measured on the real
#: corpus, 6,974 of 9,295 Cursor MCP calls (75%) arrive as
#: ``CallDynamicTool``; the pre-fix builder stored them as ``builtin`` with
#: ``mcp_server`` NULL even though the server name sits in the input.
CURSOR_DYNAMIC_TOOL_NAME = "CallDynamicTool"

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
    #: The ``--cursor-chats-source`` join, when that flag was given: how many
    #: ``meta.json`` files were read, how many *distinct* Cursor sessions a
    #: meta actually updated (two metas may name the same session -- two roots
    #: pointing at the same session is a normal invocation), and how many
    #: Cursor sessions were left without a matching meta.  All three are 0
    #: when no chats source was given -- a silent half-match is the failure
    #: mode these numbers exist to expose ("162 of 163 matched" vs "3 of 163
    #: matched").
    n_chat_metas_read: int = 0
    n_chat_metas_matched: int = 0
    n_cursor_sessions_unmatched: int = 0
    #: The ``--cursor-chats-source`` store.db join (issue #87 stage 2): how
    #: many distinct sessions had their tool results applied, how many were
    #: skipped because the store's tool-call count disagreed with the
    #: transcript's, how many were skipped because the tool names disagreed
    #: elementwise, and how many tool_calls rows were filled.  All zero when
    #: no chats source was given.  A partially applied session is worse than
    #: an unapplied one -- a misaligned result attached to the wrong call is
    #: invisible afterwards -- so a session is either paired whole or skipped
    #: whole, and these counters make either outcome visible.
    n_store_sessions_paired: int = 0
    n_store_sessions_skipped_count: int = 0
    n_store_sessions_skipped_name: int = 0
    n_tool_calls_filled: int = 0
    #: Of a paired session's calls, how many had no matching ``tool-result``
    #: part and so kept NULL ``outcome`` / ``is_error`` / ``result_text``.
    #: The store never paired them with a result, so their fate is unknown --
    #: unknown is not success (issue #87 finding 2).
    n_store_calls_without_result: int = 0
    #: The ``--cursor-chats-source`` store.db conversation join (issue #87
    #: stage 3): how many messages and how many ``content`` parts were read
    #: from the stores of sessions that were read at all (paired or not), and
    #: how many of those parts carried a part type this build does not
    #: recognise -- ``text`` / ``reasoning`` / ``redacted-reasoning`` become
    #: ``events`` rows and ``tool-call`` / ``tool-result`` are stage 2's, so
    #: anything else is counted here, never dropped silently.  All zero when
    #: no chats source was given.
    n_store_messages_read: int = 0
    n_store_parts_read: int = 0
    n_store_parts_unclassified: int = 0
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
    cursor_chats_sources: list[tuple[str, int]] | None = None
    codex_sources: list[tuple[str, int]] | None = None
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
    contain secrets and must never reach the network.  The progress bar is
    disabled too: DuckDB draws it on stdout once a query runs long enough, and
    on a real corpus that prepends a bar line to every ``--json`` document.
    """
    path = Path(db_path).expanduser()
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path), read_only=read_only)
    connection.execute("SET autoinstall_known_extensions=false")
    connection.execute("SET enable_progress_bar=false")
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
    cursor_chats_sources: Sequence[str | Path] | None = None,
    codex_sources: Sequence[str | Path] | None = None,
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

    When *sources*, *opencode_sources*, *cursor_sources*, *cursor_chats_sources*,
    or *codex_sources* are provided, they are resolved to absolute paths and
    stored in :data:`META_TABLE` as JSON arrays under :data:`META_SOURCES_KEY`,
    :data:`META_OPENCODE_SOURCES_KEY`, :data:`META_CURSOR_SOURCES_KEY`,
    :data:`META_CURSOR_CHATS_SOURCES_KEY`, and :data:`META_CODEX_SOURCES_KEY`.
    A root that matched no files is still recorded -- it explains an absence.
    When any sequence is ``None`` (the default), the corresponding key is not
    written, preserving whatever value may already be in the table (for
    callers that only create the schema without a full build).
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
        if cursor_chats_sources is not None:
            resolved = [str(Path(s).expanduser().resolve()) for s in cursor_chats_sources]
            connection.execute(
                f'INSERT OR REPLACE INTO "{META_TABLE}" (key, value) VALUES (?, ?)',
                [META_CURSOR_CHATS_SOURCES_KEY, json.dumps(resolved)],
            )
        if codex_sources is not None:
            resolved = [str(Path(s).expanduser().resolve()) for s in codex_sources]
            connection.execute(
                f'INSERT OR REPLACE INTO "{META_TABLE}" (key, value) VALUES (?, ?)',
                [META_CODEX_SOURCES_KEY, json.dumps(resolved)],
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
    codex_sources: list[str],
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

    # Check Codex sources (*.jsonl)
    for file_path in _iter_sources(codex_sources, "*.jsonl")[0]:
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


def iter_cursor_chat_sources(sources: Sequence[str | Path]) -> tuple[list[Path], list[str]]:
    """(Cursor chat ``meta.json`` files, sources that do not exist).

    Directories are searched recursively for ``*/*/meta.json`` -- the shape
    under ``~/.cursor/chats/<workspace-hash>/<session-uuid>/meta.json``,
    where the directory two levels down is the session.  Cursor's other
    ``meta.json`` files (project metadata etc.) live elsewhere and are never
    matched by the two-level pattern.
    """
    return _iter_sources(sources, CURSOR_CHATS_META_PATTERN)


def iter_codex_sources(sources: Sequence[str | Path]) -> tuple[list[Path], list[str]]:
    """(Codex session JSONL files, sources that do not exist).

    Directories are searched recursively for ``*.jsonl``.
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


def _opencode_session_rows(parsed: ParsedOpenCodeFile) -> list[tuple[object, ...]]:
    """One ``sessions`` row per distinct session id a file's parts carry.

    A parsed opencode file is not one session: every tool part and text part
    carries its own ``sessionID``, and the ids can differ within one
    ``events.ndjson``, so one file can contribute several ``sessions`` rows --
    or none, when nothing in it carries a session id.  A part with no session
    id must not fabricate a session row, so only ids that actually appear are
    emitted, in first-appearance order across tool calls then text chunks.

    A session's ``started_at`` / ``ended_at`` are the min / max of the
    timestamps the file actually provides for it (the tool parts' ``ts`` --
    text chunks carry no timestamp at all), both NULL when no call of that
    session carries one.  ``n_events`` / ``n_tool_calls`` count the rows this
    file inserts for that session.
    """
    session_ids: list[str] = []
    seen: set[str] = set()
    for call in parsed.tool_calls:
        if call.session_id is not None and call.session_id not in seen:
            seen.add(call.session_id)
            session_ids.append(call.session_id)
    for chunk in parsed.text_chunks:
        if chunk.session_id is not None and chunk.session_id not in seen:
            seen.add(chunk.session_id)
            session_ids.append(chunk.session_id)

    rows: list[tuple[object, ...]] = []
    for session_id in session_ids:
        calls = [call for call in parsed.tool_calls if call.session_id == session_id]
        chunks = [chunk for chunk in parsed.text_chunks if chunk.session_id == session_id]
        timestamps = [call.ts for call in calls if call.ts is not None]
        rows.append(
            _session_row(
                Session(
                    session_id=session_id,
                    file_path=parsed.file_path,
                    source=SOURCE_OPENCODE,
                    project_dir=None,
                    cwd=None,
                    git_branch=None,
                    cc_version=None,
                    entrypoint=None,
                    started_at=min(timestamps) if timestamps else None,
                    ended_at=max(timestamps) if timestamps else None,
                    n_events=len(chunks),
                    n_tool_calls=len(calls),
                    input_tokens=None,  # type: ignore[arg-type]  # token counts are NULL for opencode
                    output_tokens=None,  # type: ignore[arg-type]
                    cache_read_tokens=None,  # type: ignore[arg-type]
                    cache_creation_tokens=None,  # type: ignore[arg-type]
                )
            )
        )
    return rows


def _opencode_text_chunk_to_event(chunk: object) -> tuple[object, ...]:
    """Map an ``OpenCodeTextChunk`` to an ``Event``-shaped row for insertion.

    ``event_id`` is synthesised and file-path-scoped, the same way the Codex
    path's is.  ``ts`` is NULL: an opencode text part carries no timestamp at
    all.  Every column opencode has no value for -- ``role``, ``parent_uuid``,
    ``depth``, ``is_sidechain``, ``is_meta``, the permission/effort fields,
    ``model``, ``cwd``, ``git_branch`` -- is NULL, never a placeholder.
    """
    from ashiato.opencode import OpenCodeTextChunk as _OTC

    assert isinstance(chunk, _OTC)
    event_id = f"opencode:text:{chunk.file_path}:{chunk.seq}"
    return _event_row(
        Event(
            event_id=event_id,
            session_id=chunk.session_id,
            file_path=chunk.file_path,
            source=SOURCE_OPENCODE,
            seq=chunk.seq,
            ts=None,
            type="text",
            role=None,
            parent_uuid=None,
            depth=None,  # type: ignore[arg-type]  # no parent tree in opencode -> NULL
            is_sidechain=None,  # type: ignore[arg-type]  # no subagent concept -> NULL
            is_meta=None,  # type: ignore[arg-type]  # no role/developer signal -> NULL
            permission_mode=None,
            effort=None,
            request_id=None,
            message_id=None,
            model=None,
            cwd=None,
            git_branch=None,
            text=chunk.text,
            raw=chunk.text,
        )
    )


def _opencode_tool_call_to_row(call: object) -> list[object]:
    """Map an ``OpenCodeToolCall`` to a ``ToolCall``-shaped row for insertion.

    ``outcome`` goes through the shared :func:`classify_outcome` -- no second
    classifier.  The parser only emits terminal parts, so ``has_result`` is
    True for every opencode call even when the state carries no output --
    that is what keeps a failed call with empty output classified as 'error'
    rather than 'pending', the same force the Codex path applies with
    ``has_result or is_error`` -- and ``is_error`` is derived from the part's
    own terminal state: an ``error`` state is a real failed call, and its
    failure message (under the ``error`` key) lands in ``result_text`` when
    the state carries no ``output``.

    ``call_event_id`` / ``result_event_id`` are NULL: the opencode event
    stream does not link its terminal tool parts to ``events`` rows, so any
    id written here would be a reference that resolves to nothing (the defect
    issue #75 removed from the Codex path).
    """
    from ashiato.opencode import OpenCodeToolCall as _OTC

    assert isinstance(call, _OTC)
    tool_name: str | None = call.tool
    tool_kind, mcp_server = split_tool_name(tool_name)
    tool_input = call.input
    input_json: str | None = (
        None if tool_input is None else json.dumps(tool_input, ensure_ascii=False, default=str)
    )
    output: str | None = call.output
    error: str | None = call.error
    is_error = call.status == "error"

    # The parser only emits terminal parts, so every call has a result by
    # construction -- has_result is True even when the state carries no
    # output, exactly as before.  That is also what keeps a failed call with
    # empty output classified as 'error' rather than 'pending' (the force
    # the Codex path applies with ``has_result or is_error``): is_error is
    # derived from the part's own terminal state.
    outcome = classify_outcome(
        has_result=True,
        result_text=output or error or "",
        is_error=is_error,
    )

    result_source = output if output is not None else error
    result_text: str | None = None
    result_truncated = False
    if result_source is not None:
        if len(result_source) > DEFAULT_RESULT_TEXT_LIMIT:
            result_text = result_source[:DEFAULT_RESULT_TEXT_LIMIT]
            result_truncated = True
        else:
            result_text = result_source

    return [
        call.call_id,          # tool_use_id
        call.session_id,
        call.file_path,
        SOURCE_OPENCODE,       # source
        call.seq,
        call.ts,               # ts
        None,                  # call_event_id
        None,                  # result_event_id
        tool_name,
        tool_kind,
        mcp_server,
        input_json,
        summarize_input(tool_name, tool_input),
        outcome,
        is_error,
        result_text,
        result_truncated,
        call.duration_ms,      # duration_ms
        None,                  # permission_mode
        None,                  # cwd
        None,                  # is_sidechain
        None,                  # parent_tool_use_id
    ]


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

    Unlike Claude and Codex, one file is not one session: ``sessions`` gets
    one row per distinct session id the file's parts carry (or none when
    nothing does), ``events`` gets one row per assistant text chunk (with
    NULL ``ts`` -- opencode text parts carry none), and ``tool_calls`` gets
    one row per terminal tool part.  ``source_files.n_events`` /
    ``n_tool_calls`` reflect the rows actually inserted, not a hardcoded
    zero.
    """
    _insert_rows(
        connection,
        "sessions",
        _opencode_session_rows(parsed),
        scratch=scratch,
    )
    _insert_rows(
        connection,
        "events",
        [_opencode_text_chunk_to_event(chunk) for chunk in parsed.text_chunks],
        scratch=scratch,
    )
    _insert_rows(
        connection,
        "tool_calls",
        [_opencode_tool_call_to_row(call) for call in parsed.tool_calls],
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
                len(parsed.text_chunks),    # n_events
                len(parsed.tool_calls),     # n_tool_calls
                parsed.n_parse_errors,
                built_at,
            )
        ],
        scratch=scratch,
    )


#: Part types of a Cursor store message that become ``events`` rows, and
#: the event ``type`` spelling each maps to -- the same vocabulary the Codex
#: path uses, not a second one (issue #87 stage 3).
CURSOR_STORE_EVENT_PART_TYPES: dict[str, str] = {
    "text": "text",
    "reasoning": "reasoning",
    "redacted-reasoning": "redacted_reasoning",
}

#: The part types the build recognises at all.  ``tool-call`` /
#: ``tool-result`` are stage 2's -- they fill ``tool_calls``, never
#: ``events`` -- and the three event types map through
#: :data:`CURSOR_STORE_EVENT_PART_TYPES`.  A part whose type is outside this
#: set is counted as unclassified (``n_store_parts_unclassified``), never
#: dropped silently.
CURSOR_STORE_KNOWN_PART_TYPES: frozenset[str] = frozenset(
    ("text", "reasoning", "redacted-reasoning", "tool-call", "tool-result")
)


def _cursor_store_part_to_event(
    message: object,
    part: object,
    *,
    session_id: str,
    file_path: str,
    cwd: str | None,
    block_index: int,
) -> tuple[object, ...]:
    """Map one store ``text`` / ``reasoning`` / ``redacted-reasoning`` part to an ``Event`` row.

    ``event_id`` is ``cursor:store:{file_path}:{seq}:{block_index}`` --
    deterministic (a rebuild over the same store rewrites the same rows) and
    distinct within a session because it is derived from the session and the
    *position*, never from the store message's ``id``: measured on the real
    corpus, 4,649 of 24,913 message ids duplicate another id in the same
    session, so the message id cannot identify a row (issue #87 premise 4).
    ``seq`` is the message's 0-based index among the store's JSON messages in
    table order and ``block_index`` the part's 0-based index within the
    message, so ordering is reconstructible without a timestamp; ``ts`` is
    NULL -- the store carries no per-message timestamp (measured twice).

    ``role`` is the message's role verbatim (``system`` / ``user`` /
    ``assistant``).  ``text`` is the part's text; a ``redacted-reasoning``
    part has none and reads NULL, never a placeholder.  ``is_meta`` is True
    exactly for the ``system`` role -- the same rule the Codex path applies
    to its ``developer`` role -- so the system prompt never joins the
    exchange stream ``topics`` / ``grep`` / ``orphans`` read.  ``cwd`` is
    the value stage 1 already set on the session.  ``raw`` holds the same
    extracted part text, like every other Cursor event row.
    """
    from ashiato.cursor import CursorStoreMessage as _CSM
    from ashiato.cursor import CursorStorePart as _CSP

    assert isinstance(message, _CSM)
    assert isinstance(part, _CSP)
    event_type = (
        CURSOR_STORE_EVENT_PART_TYPES.get(part.type) if isinstance(part.type, str) else None
    )
    assert event_type is not None
    role = message.role
    text = part.text if part.type != "redacted-reasoning" else None
    return (
        f"cursor:store:{file_path}:{message.message_index}:{block_index}",  # event_id
        session_id,
        file_path,
        SOURCE_CURSOR,
        message.message_index,  # seq -- 0-based message index in table order
        None,                   # ts -- the store carries no per-message timestamp
        event_type,
        role,
        None,                   # parent_uuid
        None,                   # depth
        None,                   # is_sidechain
        role == "system",       # is_meta -- the Codex rule, applied to system
        None,                   # permission_mode
        None,                   # effort
        None,                   # request_id
        None,                   # message_id
        None,                   # model
        cwd,
        None,                   # git_branch
        text,
        text,                   # raw -- the extracted part text, like the other sources
    )


def _cursor_text_chunk_to_event(chunk: object) -> tuple[object, ...]:
    """Map a ``CursorTextChunk`` to an ``Event``-shaped row for insertion.

    This is the *transcript-derived* fallback row: it exists so a Cursor
    session is visible in ``events`` even when no chats store is read.  Its
    ``role`` is NULL because the transcript export records no role -- but the
    text itself is not exclusively assistant text: the transcript's text
    blocks mix user and assistant messages (measured on a 60-session sample,
    373 of 558 store user texts and 1,255 of 1,335 store assistant texts
    appear verbatim in the transcript), so these rows are text chunks with
    ``role`` NULL, nothing more specific.  When ``--cursor-chats-source``
    pairs the session with its ``store.db``, :func:`_apply_cursor_chat_stores`
    deletes these rows and inserts store-derived ones that carry the real
    ``role``, the system prompt and the reasoning (issue #87 stage 3) -- the
    transcript rows and the store rows would otherwise duplicate nearly all
    of the text.

    ``event_id`` is synthesised and file-path-scoped, the same way the
    opencode and Codex paths' are -- ``seq`` alone does not distinguish two
    text blocks of one assistant message, so ``block_index`` is part of the
    id (the parser can emit several text chunks from one transcript line).
    ``ts`` is NULL: a Cursor transcript records no timestamp at all, and
    ``seq`` / ``block_index`` already order the blocks within a file.  Every
    column Cursor has no value for -- ``parent_uuid``, ``depth``,
    ``is_sidechain``, ``is_meta``, the permission/effort fields, ``model``,
    ``cwd``, ``git_branch`` -- is NULL, never a placeholder.
    """
    from ashiato.cursor import CursorTextChunk as _CTC

    assert isinstance(chunk, _CTC)
    event_id = f"cursor:text:{chunk.file_path}:{chunk.seq}:{chunk.block_index}"
    return _event_row(
        Event(
            event_id=event_id,
            session_id=chunk.session_id,
            file_path=chunk.file_path,
            source=SOURCE_CURSOR,
            seq=chunk.seq,
            ts=None,
            type="text",
            role=None,
            parent_uuid=None,
            depth=None,  # type: ignore[arg-type]  # no parent tree in Cursor -> NULL
            is_sidechain=None,  # type: ignore[arg-type]  # no subagent concept -> NULL
            is_meta=None,  # type: ignore[arg-type]  # no role/developer signal -> NULL
            permission_mode=None,
            effort=None,
            request_id=None,
            message_id=None,
            model=None,
            cwd=None,
            git_branch=None,
            text=chunk.text,
            raw=chunk.text,
        )
    )


def _cursor_tool_call_to_row(call: object) -> list[object]:
    """Map a ``CursorToolCall`` to a ``ToolCall``-shaped row for insertion.

    ``outcome`` is NULL and ``is_error`` is NULL at insert time -- *not* what
    :func:`classify_outcome` would return for these calls.  A Cursor tool_use
    block records no result at all: no output, no status, nothing, so feeding
    that absence to ``classify_outcome`` yields ``'pending'`` for every call.
    ``'pending'`` means "the session ended mid-call" -- a claim about the
    session, not about the call -- and a later reader would take the whole
    Cursor corpus to be sessions that were constantly interrupted.  The
    format recording nothing is not the same as the call not finishing.

    The NULL is a *pre-join default*, not a verdict: when
    ``--cursor-chats-source`` is given, :func:`_apply_cursor_chat_stores`
    re-reads the session's ``store.db`` (the undocumented local store that
    holds the tool results the transcript export never records) and fills
    ``outcome`` / ``is_error`` / ``result_text`` / ``result_truncated`` on
    every call of a session whose store pairs with the transcript -- same
    tool-call count and elementwise-equal tool names.  A session the store
    cannot pair (count or name mismatch) keeps its NULLs, and a build without
    the chats source leaves every Cursor call NULL -- ``hygiene`` counts
    ``outcome = 'pending'`` as ``pending_tool_call`` and ``nominate`` gates
    on ``outcome = 'error'``; a NULL is invisible to both, which is exactly
    right for a call whose fate is genuinely unknown.

    ``ts`` is NULL for the same reason -- do not fabricate an ordering -- and
    ``seq`` / ``block_index`` already give within-file order.  The store join
    does not change that: ``store.db`` carries no per-message timestamp, so
    ``ts`` stays NULL after the join too.

    ``tool_name`` is the recorded block name.  Cursor calls MCP tools through
    two block names, ``CallMcpTool`` and ``CallDynamicTool``, and records
    which MCP tool it is in the input instead of in the name -- ``CallMcpTool``
    carries ``server`` / ``toolName``, ``CallDynamicTool`` spells the same
    identity as ``namespace`` / ``toolName`` -- so the mcp/``split_tool_name``
    spelling the other sources use does not apply; ``mcp_server`` is read
    straight out of the recorded input.  ``input`` and ``input_summary`` come
    from the recorded input as with every other source.

    ``call_event_id`` / ``result_event_id`` are NULL: Cursor does not link a
    tool_use block to any ``events`` row (it records no result event at all),
    so the issue #75 invariant -- every non-null id resolves to an ``events``
    row -- holds rather than being satisfied with a synthetic id.
    """
    from ashiato.cursor import CursorToolCall as _CTC

    assert isinstance(call, _CTC)
    tool_input = call.input
    input_json: str | None = (
        None if tool_input is None
        else json.dumps(tool_input, ensure_ascii=False, default=str)
    )

    # MCP blocks (CallMcpTool / CallDynamicTool) name the real tool in the
    # input, not the block name.
    tool_kind = "builtin"
    mcp_server: str | None = None
    if (
        call.name in (CURSOR_MCP_TOOL_NAME, CURSOR_DYNAMIC_TOOL_NAME)
        and isinstance(tool_input, dict)
    ):
        server = tool_input.get("server")
        namespace = tool_input.get("namespace")
        tool_name = tool_input.get("toolName")
        if isinstance(server, str) or isinstance(namespace, str) or isinstance(tool_name, str):
            tool_kind = "mcp"
        if isinstance(server, str):
            mcp_server = server
        elif isinstance(namespace, str):
            mcp_server = namespace

    return [
        call.call_id,          # tool_use_id
        call.session_id,
        call.file_path,
        SOURCE_CURSOR,         # source
        call.seq,
        None,                  # ts -- Cursor records none; seq/block_index order within a file
        None,                  # call_event_id
        None,                  # result_event_id
        call.name,             # tool_name -- the recorded block name
        tool_kind,
        mcp_server,
        input_json,
        summarize_input(call.name, tool_input),
        None,                  # outcome -- unknown fate, never 'pending' or 'ok'
        None,                  # is_error -- unknown fate, never False
        None,                  # result_text -- no recorded result
        None,                  # result_truncated -- no recorded result
        None,                  # duration_ms
        None,                  # permission_mode
        None,                  # cwd
        None,                  # is_sidechain
        None,                  # parent_tool_use_id
    ]


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

    Unlike opencode, one file *is* one session: ``ParsedCursorFile`` carries a
    file-level ``session_id`` (the transcript file name's uuid stem), so one
    ``sessions`` row per file, one ``events`` row per transcript text chunk
    (``role`` NULL -- the export records no role; the transcript's text
    blocks are a mix of user and assistant messages, issue #87 premise 1),
    and one ``tool_calls`` row per tool_use block, all with ``source =
    'cursor'``.  These ``events`` rows are the pre-store fallback: when
    ``--cursor-chats-source`` pairs the session with its ``store.db``,
    :func:`_apply_cursor_chat_stores` replaces them with store-derived rows
    that carry the real ``role``, the system prompt and the reasoning.  The
    kaiba-ledger join feeds ``recall_calls`` only: the main tables are built
    purely from what the transcript records (until the store join replaces
    the events).
    """
    session = Session(
        session_id=parsed.session_id,
        file_path=parsed.file_path,
        source=SOURCE_CURSOR,
        project_dir=None,
        cwd=None,
        git_branch=None,
        cc_version=None,
        entrypoint=None,
        started_at=None,
        ended_at=None,
        n_events=len(parsed.text_chunks),
        n_tool_calls=len(parsed.tool_calls),
        input_tokens=None,  # type: ignore[arg-type]  # token counts are NULL for Cursor
        output_tokens=None,  # type: ignore[arg-type]
        cache_read_tokens=None,  # type: ignore[arg-type]
        cache_creation_tokens=None,  # type: ignore[arg-type]
    )
    _insert_rows(connection, "sessions", [_session_row(session)], scratch=scratch)
    _insert_rows(
        connection,
        "events",
        [_cursor_text_chunk_to_event(chunk) for chunk in parsed.text_chunks],
        scratch=scratch,
    )
    _insert_rows(
        connection,
        "tool_calls",
        [_cursor_tool_call_to_row(call) for call in parsed.tool_calls],
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
                len(parsed.text_chunks),    # n_events
                len(parsed.tool_calls),     # n_tool_calls
                parsed.n_parse_errors,
                built_at,
            )
        ],
        scratch=scratch,
    )


def _apply_cursor_chat_metas(
    connection: duckdb.DuckDBPyConnection,
    meta_files: Sequence[Path],
    result: BuildResult,
) -> None:
    """Join ``--cursor-chats-source`` metas onto the ingested Cursor sessions.

    Every ``meta.json`` under the chats roots is re-read and re-applied on
    every build -- the files are too few and too small to track incrementally,
    and the UPDATE is idempotent, so a changed meta is picked up even when the
    transcript it belongs to was skipped as unchanged.  A meta whose
    ``session_id`` matches a Cursor session already ingested from a transcript
    fills ``sessions.cwd`` / ``events.cwd`` / ``tool_calls.cwd`` from the
    meta's ``cwd`` and ``sessions.started_at`` / ``ended_at`` from
    ``createdAtMs`` / ``updatedAtMs`` (naive UTC, the same convention as
    ``built_at``).  ``events.ts`` and ``tool_calls.ts`` stay NULL: no
    per-message timestamp exists anywhere in the store, and interpolating one
    from the session bounds would put an estimate in a column the other
    sources fill with a measurement.  A meta with no matching transcript
    session creates nothing, and a transcript session with no meta keeps its
    NULL ``cwd`` -- neither is an error; both are counted on *result* so a
    silent half-match cannot hide.

    Two metas can name the same session (two roots pointing at the same
    session is a normal invocation, not an error), so the matched count is
    derived from the *set* of sessions a meta actually updated, never from
    the number of meta files -- one duplicate plus one genuinely unmatched
    session must read as 1 matched / 1 unmatched, not as 0 unmatched.  A
    value is never overwritten with ``None`` either: ``cwd``,
    ``started_at`` and ``ended_at`` are written only when the parsed meta
    actually carries them, so a later, emptier meta cannot null out what an
    earlier one filled.  A session whose metas carry no value for a column
    keeps NULL there, as before.
    """
    cursor_session_ids = {
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT session_id FROM sessions WHERE source = ?", [SOURCE_CURSOR]
        ).fetchall()
    }
    result.n_chat_metas_read = len(meta_files)
    matched_ids: set[str] = set()
    for path in meta_files:
        meta = parse_chat_meta(path)
        if meta is None or meta.session_id not in cursor_session_ids:
            continue
        assignments: list[str] = []
        params: list[object] = []
        if meta.cwd is not None:
            assignments.append("cwd = ?")
            params.append(meta.cwd)
        if meta.created_at is not None:
            assignments.append("started_at = ?")
            params.append(meta.created_at.replace(tzinfo=None))
        if meta.updated_at is not None:
            assignments.append("ended_at = ?")
            params.append(meta.updated_at.replace(tzinfo=None))
        if assignments:
            connection.execute(
                f"UPDATE sessions SET {', '.join(assignments)} "
                "WHERE session_id = ? AND source = ?",
                [*params, meta.session_id, SOURCE_CURSOR],
            )
            matched_ids.add(meta.session_id)
        if meta.cwd is not None:
            connection.execute(
                "UPDATE events SET cwd = ? WHERE session_id = ? AND source = ?",
                [meta.cwd, meta.session_id, SOURCE_CURSOR],
            )
            connection.execute(
                "UPDATE tool_calls SET cwd = ? WHERE session_id = ? AND source = ?",
                [meta.cwd, meta.session_id, SOURCE_CURSOR],
            )
    result.n_chat_metas_matched = len(matched_ids)
    result.n_cursor_sessions_unmatched = len(cursor_session_ids - matched_ids)


def _tool_use_order_key(tool_use_id: str) -> tuple[int, int]:
    """(seq, block_index) of a synthetic ``seq:block_index`` call id.

    The parser emits calls in transcript order -- line by line, block by
    block -- so the ingestion order a store must pair against is the
    *numeric* (seq, block_index) order.  ``ORDER BY tool_use_id`` would be
    lexicographic and misorder a line that issued ten or more calls; sorting
    in Python keeps the pairing faithful.
    """
    seq, _, block = tool_use_id.partition(":")
    try:
        return int(seq), int(block)
    except ValueError:
        return 0, 0


def _restore_cursor_transcript_events(
    connection: duckdb.DuckDBPyConnection,
    *,
    session_id: str,
    scratch: Path | None,
) -> None:
    """Give a session its transcript-derived events back, when it lost them.

    The inverse of the events replacement (issue #87 stage 3, Finding 2): a
    session that paired on an earlier build holds store-derived ``events``
    rows (or none at all, when the replacement produced zero rows), and once
    its store stops pairing the transcript rows must come back -- an unpaired
    session keeps exactly what it has *today*, across builds, never the stale
    store origin and never nothing.

    The session's own transcript file is re-parsed here -- only this
    session, only when its current rows are store-derived or absent, never
    every transcript on every build -- and its ``events`` are rewritten from
    the parsed text chunks in one transaction, with ``sessions.n_events`` /
    ``source_files.n_events`` corrected to the restored count.  The rest of
    the session (``tool_calls``, ``recall_calls``, the ``source_files``
    bookkeeping row itself) is left alone: those tables were never part of
    the events replacement, and the transcript's size / mtime are unchanged,
    so the incremental skip still knows the file.  A session whose current
    rows are already transcript-derived (``cursor:text:``) is a no-op, as is
    a session whose transcript file can no longer be read -- a restore must
    not invent rows it cannot derive.

    Runs in a transaction: a failure mid-restore leaves the session exactly
    as it was, the same guarantee the pair-side replacement has.
    """
    row = connection.execute(
        "SELECT file_path FROM sessions WHERE session_id = ? AND source = ?",
        [session_id, SOURCE_CURSOR],
    ).fetchone()
    if row is None:
        return
    file_path = row[0]
    current = connection.execute(
        "SELECT event_id FROM events WHERE session_id = ? AND source = ?",
        [session_id, SOURCE_CURSOR],
    ).fetchall()
    if current and all(
        isinstance(event_id, str) and event_id.startswith("cursor:text:")
        for (event_id,) in current
    ):
        return  # already transcript-derived -- nothing to restore
    try:
        parsed = parse_cursor_file(file_path)
    except OSError:
        return  # transcript unreadable -- leave the session untouched
    rows = [_cursor_text_chunk_to_event(chunk) for chunk in parsed.text_chunks]
    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute(
            "DELETE FROM events WHERE session_id = ? AND source = ?",
            [session_id, SOURCE_CURSOR],
        )
        _insert_rows(connection, "events", rows, scratch=scratch)
        connection.execute(
            "UPDATE sessions SET n_events = ? WHERE session_id = ? AND source = ?",
            [len(rows), session_id, SOURCE_CURSOR],
        )
        connection.execute(
            "UPDATE source_files SET n_events = ? WHERE file_path = ?",
            [len(rows), file_path],
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise


def _apply_cursor_chat_stores(
    connection: duckdb.DuckDBPyConnection,
    meta_files: Sequence[Path],
    result: BuildResult,
    *,
    result_text_limit: int = DEFAULT_RESULT_TEXT_LIMIT,
    scratch: Path | None = None,
) -> None:
    """Join ``--cursor-chats-source`` ``store.db`` files onto the ingested Cursor calls.

    Every ``store.db`` sitting next to a ``meta.json`` under the chats roots
    is re-read and re-applied on every build, exactly like the metas: the
    incremental bookkeeping in ``source_files`` tracks *transcripts*, and a
    ``store.db`` has no record there, so there is no cheap way to skip a
    session whose store is unchanged -- reading them all is the honest cost
    (a handful of small SQLite files per session).  The apply is idempotent,
    so a changed store is picked up even when the transcript it belongs to
    was skipped as unchanged, and a re-applied store writes the same values.

    A session is paired only when the store is *proven* to line up with the
    transcript: the two tool-call counts are equal *and* the tool names agree
    elementwise -- ``transcript tool_name == store toolName``, both taken as
    the plain recorded strings.  Measured on the real corpus, that is the
    right rule: the store records the same ``CallMcpTool`` block name the
    transcript does in 2,321 of 2,321 observed MCP positions, and no non-MCP
    position disagreed.  A transcript with no tool calls pairs vacuously
    with a store that also has none -- the evidence that the two describe
    the same session is then the session id, the same evidence stage 1
    already relies on for ``cwd``; a store that *does* hold calls against a
    call-less transcript is a genuine count mismatch and does not pair
    (issue #87 stage 3, Finding 4).  This guard is what makes a filled
    result trustworthy, and it is also what protects against the store's
    ordering being wrong: ``parse_chat_store`` reads ``blobs`` in table
    order because the ``latestRootBlobId`` root is only a checkpoint window
    over the newest messages (Cursor's CLI has saved only new transcript
    entries per checkpoint since 2026-07-13), so if the table order ever
    disagreed with the transcript's, the count/name check would refuse the
    session rather than misalign a single result.  If either check fails,
    nothing is applied for that session -- a partially applied session is
    worse than an unapplied one, because a misaligned result attached to
    the wrong call is invisible afterwards -- and the skip is counted on
    *result* (``n_store_sessions_skipped_count`` /
    ``n_store_sessions_skipped_name``) so a silent half-match cannot hide.

    Several chat roots can point at the same session (the live tree and an
    archive mirror is a normal invocation).  The candidate stores are tried
    in ``meta_files`` order -- ``iter_cursor_chat_sources`` returns the
    metas sorted by absolute path, so the winner is the *first* store in
    that order that pairs -- and a session is marked handled only once a
    store has actually been applied, never when it merely failed the pairing
    checks: a stale, unreadable or mismatched store cannot claim the session
    and starve a later valid one (Finding 3).  When no candidate pairs, the
    session is unpaired.  The paired count is derived from the *set* of
    sessions actually paired, never from the number of store files.

    For a paired session, each call that has a matching ``tool-result`` part
    gets its ``outcome`` / ``is_error`` / ``result_text`` /
    ``result_truncated`` from the raw ``result`` via
    :func:`ashiato.cursor.classify_store_result` and
    :func:`ashiato.cursor.store_result_text`; a call whose id never matched a
    result part keeps NULL in all four columns -- the store recorded no
    result for it, so its fate is unknown, and unknown must not read as
    success (``n_store_calls_without_result`` counts these).  ``ts`` is never
    touched: the store carries no per-message timestamp, so it stays NULL.
    ``tool_use_id`` is never touched either: it is the transcript-derived
    ``seq:block_index`` identity documented in :func:`parse_file`, and the
    store has no id the transcript carries to replace it with.

    Every store that is read contributes ``n_store_messages_read`` /
    ``n_store_parts_read`` (paired or not), and a part whose type is not one
    of the recognised five is counted in ``n_store_parts_unclassified`` --
    counted, never dropped silently (issue #87 stage 3).

    For a paired session the transcript-derived ``events`` rows are
    *replaced*, not added to: the store's system / user / assistant messages
    carry the roles, the system prompt and the reasoning the transcript rows
    never had, and keeping both would duplicate nearly all of the text
    (issue #87 premise 1 -- measured, 1,255 of 1,335 store assistant texts
    appear verbatim in the transcript).  The new rows follow
    :func:`_cursor_store_part_to_event`: one row per ``text`` / ``reasoning``
    / ``redacted-reasoning`` part of every non-``tool`` message, in
    conversation order, with ``seq`` = 0-based message index, ``block_index``
    = part index, ``ts`` NULL, ``event_id`` derived from session + position,
    ``cwd`` as stage 1 set it, ``is_meta`` True only for the ``system`` role.
    ``sessions.n_events`` / ``source_files.n_events`` are updated to the
    replaced row count so the recorded numbers stay truthful.  The whole
    per-session apply -- the result updates, the delete, the insert and the
    count updates -- runs in one transaction, the way :func:`_store_file`
    works: a failure at any point rolls the session back to exactly what it
    was, never a session with its events deleted and nothing inserted
    (Finding 1).

    An unpaired session keeps exactly the transcript-derived rows it has
    today -- and that now holds *across builds*: when the database already
    holds store-derived rows for it (or none at all, because an earlier
    replacement produced zero rows) and no candidate store pairs any more,
    :func:`_restore_cursor_transcript_events` re-derives the transcript rows
    in the same build (Finding 2).  The session is never left half-way
    between the two origins.
    """
    ingested: dict[str, list[tuple[str, str | None]]] = {}
    rows = connection.execute(
        "SELECT session_id, tool_use_id, tool_name FROM tool_calls "
        "WHERE source = ?",
        [SOURCE_CURSOR],
    ).fetchall()
    rows.sort(key=lambda row: _tool_use_order_key(row[1]))
    for session_id, tool_use_id, tool_name in rows:
        ingested.setdefault(session_id, []).append(
            (tool_use_id, tool_name if isinstance(tool_name, str) else None)
        )

    # Sessions that actually have an ingested transcript -- the gate a
    # store must pass through before it can pair or be restored.  Built from
    # the sessions table, not from tool_calls: a transcript that made no
    # tool calls at all is still a session, and can still be enriched
    # (Finding 4).  A meta whose session has no transcript row creates
    # nothing, exactly as before.
    cursor_session_ids = {
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT session_id FROM sessions WHERE source = ?", [SOURCE_CURSOR]
        ).fetchall()
    }

    # First pass: try the candidate stores in order until one pairs.  A
    # session is marked handled only once a store was actually applied, so a
    # stale, unreadable or mismatched store never claims it (Finding 3).
    handled: set[str] = set()
    paired_ids: set[str] = set()
    for path in meta_files:
        session_id = path.parent.name
        if session_id in handled:
            continue
        if session_id not in cursor_session_ids:
            continue  # no ingested transcript session -- nothing to pair
        store_path = path.with_name("store.db")
        if not store_path.is_file():
            continue  # no store candidate here; the restore pass decides below
        conversation = parse_chat_store(store_path)
        result.n_store_messages_read += len(conversation.messages)
        result.n_store_parts_read += sum(
            len(message.parts) for message in conversation.messages
        )
        result.n_store_parts_unclassified += sum(
            1
            for message in conversation.messages
            for part in message.parts
            if part.type not in CURSOR_STORE_KNOWN_PART_TYPES
        )
        session_calls = ingested.get(session_id, [])
        if len(conversation) != len(session_calls):
            result.n_store_sessions_skipped_count += 1
            continue
        if any(
            store_call.tool_name != expected
            for store_call, (_, expected) in zip(conversation, session_calls, strict=True)
        ):
            result.n_store_sessions_skipped_name += 1
            continue
        # Paired: apply the results and replace the events in one
        # transaction, so a failure leaves the session exactly as it was
        # instead of half-replaced (Finding 1).
        connection.execute("BEGIN TRANSACTION")
        try:
            for store_call, (tool_use_id, _) in zip(conversation, session_calls, strict=True):
                if not store_call.has_result:
                    # The store never paired this call with a tool-result part.
                    # Its fate is unknown: keep NULL outcome / is_error /
                    # result_text rather than read the absence as success.
                    result.n_store_calls_without_result += 1
                    continue
                is_error, outcome = classify_store_result(store_call.result)
                text = store_result_text(store_call.result)
                truncated = False
                if len(text) > result_text_limit:
                    text = text[:result_text_limit]
                    truncated = True
                connection.execute(
                    "UPDATE tool_calls SET outcome = ?, is_error = ?, result_text = ?, "
                    "result_truncated = ? "
                    "WHERE tool_use_id = ? AND session_id = ? AND source = ?",
                    [outcome, is_error, text, truncated, tool_use_id, session_id, SOURCE_CURSOR],
                )
            # issue #87 stage 3: the paired session's events are replaced by
            # the store's conversation (roles, system prompt, reasoning) --
            # adding on top would duplicate nearly all of the text (premise 1).
            session_row = connection.execute(
                "SELECT file_path, cwd FROM sessions "
                "WHERE session_id = ? AND source = ?",
                [session_id, SOURCE_CURSOR],
            ).fetchone()
            if session_row is not None:
                file_path, cwd = session_row
                event_rows = [
                    _cursor_store_part_to_event(
                        message,
                        part,
                        session_id=session_id,
                        file_path=file_path,
                        cwd=cwd,
                        block_index=block_index,
                    )
                    for message in conversation.messages
                    if message.role != "tool"  # tool messages are stage 2's, never events
                    for block_index, part in enumerate(message.parts)
                    if part.type in CURSOR_STORE_EVENT_PART_TYPES
                ]
                connection.execute(
                    "DELETE FROM events WHERE session_id = ? AND source = ?",
                    [session_id, SOURCE_CURSOR],
                )
                _insert_rows(connection, "events", event_rows, scratch=scratch)
                connection.execute(
                    "UPDATE sessions SET n_events = ? WHERE session_id = ? AND source = ?",
                    [len(event_rows), session_id, SOURCE_CURSOR],
                )
                connection.execute(
                    "UPDATE source_files SET n_events = ? WHERE file_path = ?",
                    [len(event_rows), file_path],
                )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        handled.add(session_id)
        paired_ids.add(session_id)
        result.n_tool_calls_filled += sum(
            1 for call in conversation if call.has_result
        )

    # Second pass: sessions no store was applied to.  If the database still
    # holds store-derived rows for one (or none at all), its transcript rows
    # are restored in this same build -- an unpaired session keeps what it
    # has today, never the stale store origin and never nothing (Finding 2).
    for path in meta_files:
        session_id = path.parent.name
        if session_id in handled:
            continue
        if session_id not in cursor_session_ids:
            continue
        _restore_cursor_transcript_events(
            connection, session_id=session_id, scratch=scratch
        )
    result.n_store_sessions_paired = len(paired_ids)


def _codex_tool_call_to_row(call: object) -> list[object]:
    """Map a ``CodexToolCall`` to a ``ToolCall``-shaped row for insertion.

    Mirrors the mapping that :func:`_insert_parsed` does for Claude tool calls
    but adapted to the simpler Codex payload (no permission mode).
    ``ts`` is populated from the record's own ``timestamp`` field.
    """
    from ashiato.codex import CodexToolCall as _CTC

    assert isinstance(call, _CTC)
    tool_name: str | None = call.tool_name
    tool_kind, mcp_server = split_tool_name(tool_name)
    tool_input = call.input
    input_json: str | None = (
        None if tool_input is None
        else json.dumps(tool_input, ensure_ascii=False, default=str)
    )
    output: str | None = call.output
    has_result = output is not None and output != ""

    # A call is an error when Codex's own status says so ('failed'), when a
    # command exit code is present and nonzero, or when a non-empty error
    # message is present (older MCP shape that folds the failure text into
    # output but carries no status).  A status/exit_code that is present but
    # malformed must not be treated as a clean success either.  An explicit
    # status of 'completed' wins over the error-message inference -- the
    # source states the call completed, so the message is not treated as
    # failure evidence; an empty or non-string error changes nothing.
    status = call.status
    exit_code = call.exit_code
    status_ok = status is None or (isinstance(status, str) and status == "completed")
    exit_code_ok = (
        exit_code is None
        or (isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code == 0)
    )
    status_completed = isinstance(status, str) and status == "completed"
    error_ok = status_completed or not (isinstance(call.error, str) and call.error != "")
    is_error = not (status_ok and exit_code_ok and error_ok)

    # Codex tool calls carry no link to an events row.  The item_completed
    # items this parser consumes use a different id space from the model-facing
    # response_item call_id (measured: zero overlap on a real session), so any
    # id synthesised here would be a reference that resolves to nothing, for
    # every row, always.  NULL says "this source does not link calls to
    # events", which is true.
    call_event_id = None
    result_event_id = None

    # A failed call is a terminal state, never 'pending' -- even when it has
    # no output at all.
    outcome = classify_outcome(
        has_result=has_result or is_error,
        result_text=output or "",
        is_error=is_error,
    )

    result_text: str | None = None
    result_truncated = False
    if output is not None:
        if len(output) > DEFAULT_RESULT_TEXT_LIMIT:
            result_text = output[:DEFAULT_RESULT_TEXT_LIMIT]
            result_truncated = True
        else:
            result_text = output

    return [
        call.call_id,         # tool_use_id
        call.session_id,
        call.file_path,
        SOURCE_CODEX,         # source
        call.seq,
        call.ts,               # ts
        call_event_id,
        result_event_id,
        tool_name,
        tool_kind,
        mcp_server,
        input_json,
        summarize_input(tool_name, tool_input),
        outcome,
        is_error,
        result_text,
        result_truncated,
        call.duration_ms,      # duration_ms
        None,                  # permission_mode
        call.cwd,              # cwd
        False,                 # is_sidechain
        None,                  # parent_tool_use_id
    ]


def _codex_text_chunk_to_event(chunk: object) -> tuple[object, ...]:
    """Map a ``CodexTextChunk`` to an ``Event``-shaped row for insertion."""
    from ashiato.codex import CodexTextChunk as _CTC

    assert isinstance(chunk, _CTC)
    event_id = f"codex:text:{chunk.file_path}:{chunk.seq}"
    role = chunk.role if isinstance(chunk.role, str) else "unknown"
    return _event_row(Event(
        event_id=event_id,
        session_id=chunk.session_id,
        file_path=chunk.file_path,
        source=SOURCE_CODEX,
        seq=chunk.seq,
        ts=chunk.ts,
        type="text",
        role=role,
        parent_uuid=None,
        depth=0,
        is_sidechain=False,
        is_meta=(role == "developer"),
        permission_mode=None,
        effort=None,
        request_id=None,
        message_id=None,
        model=None,
        cwd=None,
        git_branch=None,
        text=chunk.text,
        raw=chunk.text,
    ))


def _codex_event_to_row(event: object) -> tuple[object, ...]:
    """Map a ``CodexEvent`` (e.g. a context compaction) to an ``Event``-shaped row.

    Unlike :func:`_codex_text_chunk_to_event` the row keeps the event's own
    ``type`` (``context_compaction``, never ``text``) and carries no role, so
    it is distinguishable from assistant text and never joins the exchange
    stream that ``topics`` and friends read.
    """
    from ashiato.codex import CodexEvent as _CE

    assert isinstance(event, _CE)
    event_id = f"codex:{event.event_id}"
    return _event_row(Event(
        event_id=event_id,
        session_id=event.session_id,
        file_path=event.file_path,
        source=SOURCE_CODEX,
        seq=event.seq,
        ts=event.ts,
        type=event.type,
        role=None,
        parent_uuid=None,
        depth=0,
        is_sidechain=False,
        is_meta=False,
        permission_mode=None,
        effort=None,
        request_id=None,
        message_id=None,
        model=None,
        cwd=None,
        git_branch=None,
        text=event.text or "",
        raw=event.raw,
    ))


def _insert_codex_parsed(
    connection: duckdb.DuckDBPyConnection,
    parsed: ParsedCodexFile,
    recall_rows: Sequence[RecallCall],
    *,
    stat: os.stat_result,
    content_hash: str,
    built_at: datetime,
    scratch: Path | None,
) -> None:
    """The Codex counterpart of :func:`_insert_parsed`.

    Inserts ``tool_calls`` (one row per ``CodexToolCall``), one ``sessions``
    row per file, and ``events`` rows for text chunks with non-empty text plus
    non-text ``CodexEvent`` rows (context compactions).
    ``source_files.n_events`` reflects the count of inserted events.
    """
    # Session row
    n_events = len(parsed.text_chunks) + len(parsed.events)
    session = Session(
        session_id=parsed.session_id,
        file_path=parsed.file_path,
        source=SOURCE_CODEX,
        project_dir=None,
        cwd=None,
        git_branch=None,
        cc_version=None,
        entrypoint=None,
        started_at=parsed.started_at,
        ended_at=parsed.ended_at,
        n_events=n_events,
        n_tool_calls=len(parsed.tool_calls),
        input_tokens=parsed.input_tokens,
        output_tokens=parsed.output_tokens,
        cache_read_tokens=parsed.cache_read_tokens,
        cache_creation_tokens=0,
    )
    _insert_rows(connection, "sessions", [_session_row(session)], scratch=scratch)

    # Event rows from text chunks, plus non-text events (compactions)
    event_rows = [_codex_text_chunk_to_event(c) for c in parsed.text_chunks]
    event_rows += [_codex_event_to_row(e) for e in parsed.events]
    _insert_rows(connection, "events", event_rows, scratch=scratch)

    _insert_rows(
        connection,
        "tool_calls",
        [_codex_tool_call_to_row(call) for call in parsed.tool_calls],
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
                n_events,                  # n_events
                len(parsed.tool_calls),   # n_tool_calls
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
    cursor_chats_sources: Sequence[str | Path] = (),
    codex_sources: Sequence[str | Path] = (),
    kaiba_db_path: str | Path | None = None,
    denial_patterns: Sequence[str] = DENIAL_PATTERNS,
    result_text_limit: int = DEFAULT_RESULT_TEXT_LIMIT,
) -> BuildResult:
    """Parse every transcript under *sources* / *opencode_sources* / *cursor_sources*
    / *codex_sources*.

    *cursor_chats_sources* is the optional ``--cursor-chats-source`` join: the
    ``meta.json`` files under those roots fill ``cwd`` and the session times
    of the Cursor sessions that match, and the ``store.db`` files sitting
    next to them fill ``outcome`` / ``is_error`` / ``result_text`` /
    ``result_truncated`` of those sessions' tool calls and *replace* the
    ``events`` of a paired session with store-derived rows carrying the
    roles, the system prompt and the reasoning (see
    :func:`_apply_cursor_chat_metas` and :func:`_apply_cursor_chat_stores`).
    Nothing is scanned for chat metas by default.
    """
    result = BuildResult(db_path=str(Path(db_path).expanduser()))
    claude_files, claude_missing = iter_transcripts(sources)
    opencode_files, opencode_missing = iter_opencode_sources(opencode_sources)
    cursor_files, cursor_missing = iter_cursor_sources(cursor_sources)
    cursor_chat_files, cursor_chat_missing = iter_cursor_chat_sources(cursor_chats_sources)
    codex_files, codex_missing = iter_codex_sources(codex_sources)
    result.missing_sources = (
        claude_missing + opencode_missing + cursor_missing + cursor_chat_missing + codex_missing
    )
    result.n_files = (
        len(claude_files) + len(opencode_files) + len(cursor_files) + len(codex_files)
    )

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
            cursor_chats_sources=cursor_chats_sources,
            codex_sources=codex_sources,
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
                result.n_sessions += len(_opencode_session_rows(parsed_oc))
                result.n_events += len(parsed_oc.text_chunks)
                result.n_tool_calls += len(parsed_oc.tool_calls)
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
                result.n_sessions += 1
                result.n_events += len(parsed_cur.text_chunks)
                result.n_tool_calls += len(parsed_cur.tool_calls)
                result.n_parse_errors += parsed_cur.n_parse_errors

            for path in codex_files:
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
                    parsed_codex = parse_codex_file(path)
                    recall_rows = extract_from_codex(
                        parsed_codex,
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
                    parsed_codex=parsed_codex,
                    recall_rows=recall_rows,
                    stat=stat,
                    content_hash=content_hash: _insert_codex_parsed(
                        connection,
                        parsed_codex,
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
                result.n_tool_calls += len(parsed_codex.tool_calls)
                result.n_parse_errors += parsed_codex.n_parse_errors

            if cursor_chat_files:
                _apply_cursor_chat_metas(connection, cursor_chat_files, result)
                _apply_cursor_chat_stores(
                    connection,
                    cursor_chat_files,
                    result,
                    result_text_limit=result_text_limit,
                    scratch=scratch,
                )
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
        cursor_chats_sources = _read_meta_json_list(connection, META_CURSOR_CHATS_SOURCES_KEY)
        codex_sources = _read_meta_json_list(connection, META_CODEX_SOURCES_KEY)

        # If no roots recorded, return early with None for roots and gap
        if (
            sources is None
            and opencode_sources is None
            and cursor_sources is None
            and cursor_chats_sources is None
            and codex_sources is None
        ):
            return DatabaseInfo(
                db_path=str(path),
                table_counts=counts,
                started_at=started_at,
                ended_at=ended_at,
                sources=None,
                opencode_sources=None,
                cursor_sources=None,
                cursor_chats_sources=None,
                codex_sources=None,
                freshness_gap=None,
            )

        # Count files per root from source_files
        known = _known_sources(connection)
        source_counts = _count_files_per_root(known, sources or [])
        opencode_counts = _count_files_per_root(known, opencode_sources or [])
        cursor_counts = _count_files_per_root(known, cursor_sources or [])
        cursor_chats_counts = _count_files_per_root(known, cursor_chats_sources or [])
        codex_counts = _count_files_per_root(known, codex_sources or [])

        # Compute freshness gap
        freshness_gap = _compute_freshness_gap(
            connection,
            sources or [],
            opencode_sources or [],
            cursor_sources or [],
            codex_sources or [],
        )

        return DatabaseInfo(
            db_path=str(path),
            table_counts=counts,
            started_at=started_at,
            ended_at=ended_at,
            sources=source_counts,
            opencode_sources=opencode_counts,
            cursor_sources=cursor_counts,
            cursor_chats_sources=cursor_chats_counts,
            codex_sources=codex_counts,
            freshness_gap=freshness_gap,
        )
    finally:
        connection.close()
