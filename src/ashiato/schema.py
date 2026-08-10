"""DuckDB schema.

Columns are declared once here, as (name, type) pairs, and everything else --
the CREATE TABLE statements, the INSERT column lists, the type spec handed to
DuckDB's JSON reader during bulk load -- is derived from them.

Column order in each table matches the field order of the corresponding
dataclass in :mod:`ashiato.parser`; ``tests/test_build.py`` asserts that, so the
two cannot drift apart silently.

``tool_use_id`` is the key of ``tool_calls`` but is deliberately not declared as
a SQL PRIMARY KEY: a real corpus can contain the same id twice (a transcript
copied or replayed across files), and a constraint violation there would abort a
build over data that is merely redundant.  Duplicates are dropped per file at
parse time instead.
"""

from __future__ import annotations

Column = tuple[str, str]

SESSION_TABLE: tuple[Column, ...] = (
    ("session_id", "VARCHAR"),
    ("file_path", "VARCHAR"),
    ("project_dir", "VARCHAR"),
    ("cwd", "VARCHAR"),
    ("git_branch", "VARCHAR"),
    ("cc_version", "VARCHAR"),
    ("entrypoint", "VARCHAR"),
    ("started_at", "TIMESTAMP"),
    ("ended_at", "TIMESTAMP"),
    ("n_events", "BIGINT"),
    ("n_tool_calls", "BIGINT"),
    ("input_tokens", "BIGINT"),
    ("output_tokens", "BIGINT"),
    ("cache_read_tokens", "BIGINT"),
    ("cache_creation_tokens", "BIGINT"),
)

EVENT_TABLE: tuple[Column, ...] = (
    ("event_id", "VARCHAR"),
    ("session_id", "VARCHAR"),
    ("file_path", "VARCHAR"),
    ("seq", "BIGINT"),
    ("ts", "TIMESTAMP"),
    ("type", "VARCHAR"),
    ("role", "VARCHAR"),
    ("parent_uuid", "VARCHAR"),
    ("depth", "INTEGER"),
    ("is_sidechain", "BOOLEAN"),
    ("is_meta", "BOOLEAN"),
    ("permission_mode", "VARCHAR"),
    ("effort", "VARCHAR"),
    ("request_id", "VARCHAR"),
    ("message_id", "VARCHAR"),
    ("model", "VARCHAR"),
    ("cwd", "VARCHAR"),
    ("git_branch", "VARCHAR"),
    ("text", "VARCHAR"),
    ("raw", "VARCHAR"),
)

TOOL_CALL_TABLE: tuple[Column, ...] = (
    ("tool_use_id", "VARCHAR"),
    ("session_id", "VARCHAR"),
    ("file_path", "VARCHAR"),
    ("seq", "BIGINT"),
    ("ts", "TIMESTAMP"),
    ("call_event_id", "VARCHAR"),
    ("result_event_id", "VARCHAR"),
    ("tool_name", "VARCHAR"),
    ("tool_kind", "VARCHAR"),
    ("mcp_server", "VARCHAR"),
    ("input", "JSON"),
    ("outcome", "VARCHAR"),
    ("is_error", "BOOLEAN"),
    ("result_text", "VARCHAR"),
    ("result_truncated", "BOOLEAN"),
    ("duration_ms", "BIGINT"),
    ("permission_mode", "VARCHAR"),
    ("cwd", "VARCHAR"),
    ("is_sidechain", "BOOLEAN"),
    ("parent_tool_use_id", "VARCHAR"),
)

SOURCE_FILE_TABLE: tuple[Column, ...] = (
    ("file_path", "VARCHAR"),
    ("size_bytes", "BIGINT"),
    ("mtime", "DOUBLE"),
    ("content_hash", "VARCHAR"),
    ("n_events", "BIGINT"),
    ("n_tool_calls", "BIGINT"),
    ("n_parse_errors", "BIGINT"),
    ("built_at", "TIMESTAMP"),
)

TABLE_COLUMNS: dict[str, tuple[Column, ...]] = {
    "sessions": SESSION_TABLE,
    "events": EVENT_TABLE,
    "tool_calls": TOOL_CALL_TABLE,
    "source_files": SOURCE_FILE_TABLE,
}

TABLES: tuple[str, ...] = tuple(TABLE_COLUMNS)

SOURCE_FILE_COLUMNS: tuple[str, ...] = tuple(name for name, _ in SOURCE_FILE_TABLE)


def column_names(table: str) -> tuple[str, ...]:
    return tuple(name for name, _ in TABLE_COLUMNS[table])


def create_table_sql(table: str) -> str:
    body = ",\n    ".join(f'"{name}" {type_}' for name, type_ in TABLE_COLUMNS[table])
    return f'CREATE TABLE IF NOT EXISTS "{table}" (\n    {body}\n);'


SCHEMA_SQL = "\n".join(create_table_sql(table) for table in TABLES)


def insert_sql(table: str) -> str:
    """A parameterised INSERT with every identifier quoted."""
    names = ", ".join(f'"{name}"' for name in column_names(table))
    placeholders = ", ".join("?" for _ in column_names(table))
    return f'INSERT INTO "{table}" ({names}) VALUES ({placeholders})'


def read_json_types(table: str) -> dict[str, str]:
    """Column types for DuckDB's JSON reader during bulk load.

    JSON columns are read as text and cast on insert: the serialised value is a
    JSON *string*, and asking the reader for JSON would nest it one level deep.
    """
    return {name: ("VARCHAR" if type_ == "JSON" else type_) for name, type_ in TABLE_COLUMNS[table]}
