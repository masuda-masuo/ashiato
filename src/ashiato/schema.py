"""DuckDB schema.

Columns are declared once here, as (name, type) pairs, and everything else --
the CREATE TABLE statements, the INSERT column lists, the type spec handed to
DuckDB's JSON reader during bulk load -- is derived from them.

Column order in each table matches the field order of the corresponding
dataclass in :mod:`ashiato.parser` (or, for ``recall_calls``,
:mod:`ashiato.recall`); ``tests/test_build.py`` asserts that, so the two
cannot drift apart silently.

``tool_use_id`` is the key of ``tool_calls`` but is deliberately not declared as
a SQL PRIMARY KEY: a real corpus can contain the same id twice (a transcript
copied or replayed across files), and a constraint violation there would abort a
build over data that is merely redundant.  Duplicates are dropped per file at
parse time instead.  ``recall_id`` follows the same rule for ``recall_calls``.
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
    ("input_summary", "VARCHAR"),
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

#: One row per completed kaiba ``recall`` call, from either source format --
#: see :mod:`ashiato.recall`, which fills this table in at build time.
RECALL_CALL_TABLE: tuple[Column, ...] = (
    ("recall_id", "VARCHAR"),
    ("session_id", "VARCHAR"),
    ("file_path", "VARCHAR"),
    ("source", "VARCHAR"),
    ("seq", "BIGINT"),
    ("ts", "TIMESTAMP"),
    ("call_id", "VARCHAR"),
    ("query", "VARCHAR"),
    ("output", "VARCHAR"),
    ("output_truncated", "BOOLEAN"),
    ("followup_text", "VARCHAR"),
    ("followup_truncated", "BOOLEAN"),
    ("overlap_tokens", "VARCHAR"),
    ("overlap_count", "BIGINT"),
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

#: Version of the *rows*, as opposed to the column layout.  Bumped whenever a
#: rebuild must re-derive existing rows even though no column changed -- the
#: denial rule moving from substring to anchored-prefix matching (issue #8)
#: changed what ``outcome`` means, and adding the ``recall_calls`` table and
#: ``recall_followups`` view (issue #10) is new derived state that an older
#: database simply does not have.  Version 4 = opencode delta coalescing
#: (PR#14 / issue #15).  Any PR that changes parser/recall derivation
#: semantics (same source bytes -> different rows) must bump this constant.
#: ``CREATE TABLE IF NOT EXISTS`` cannot see either kind of change.  Stored
#: in :data:`META_TABLE`; a database without the marker, or with a different
#: value, predates this version's rules and is refused.
FORMAT_VERSION = 4

#: Key-value table holding format metadata.  Deliberately not in ``TABLES``: it
#: has no ``file_path`` column, so it must not join the per-file incremental
#: bookkeeping or the row counts that ``database_info`` reports.
META_TABLE = "ashiato_meta"

#: The key under :data:`META_TABLE` that carries :data:`FORMAT_VERSION`.
META_FORMAT_KEY = "format_version"

META_SCHEMA_SQL = (
    f'CREATE TABLE IF NOT EXISTS "{META_TABLE}" (key VARCHAR PRIMARY KEY, value VARCHAR)'
)

TABLE_COLUMNS: dict[str, tuple[Column, ...]] = {
    "sessions": SESSION_TABLE,
    "events": EVENT_TABLE,
    "tool_calls": TOOL_CALL_TABLE,
    "recall_calls": RECALL_CALL_TABLE,
    "source_files": SOURCE_FILE_TABLE,
}

TABLES: tuple[str, ...] = tuple(TABLE_COLUMNS)

#: Tables ``ashiato info`` summarises.  ``recall_calls`` is deliberately left
#: out of this particular report: it is a narrower table that most databases
#: (anything built without an ``--opencode-source``, and most sessions even
#: with one) will simply have zero rows in, and every existing ``info``
#: output would otherwise grow a new line whether or not the caller ever
#: touched opencode ingestion.  It still participates fully in schema
#: creation, the schema-currency check, and the per-file incremental
#: replace/delete -- this only narrows what ``info`` prints.  Query it
#: directly (``ashiato sql "SELECT count(*) FROM recall_calls"``, or
#: ``ashiato recalls``) when it matters.
INFO_TABLES: tuple[str, ...] = tuple(table for table in TABLES if table != "recall_calls")

SOURCE_FILE_COLUMNS: tuple[str, ...] = tuple(name for name, _ in SOURCE_FILE_TABLE)


def column_names(table: str) -> tuple[str, ...]:
    return tuple(name for name, _ in TABLE_COLUMNS[table])


def create_table_sql(table: str) -> str:
    body = ",\n    ".join(f'"{name}" {type_}' for name, type_ in TABLE_COLUMNS[table])
    return f'CREATE TABLE IF NOT EXISTS "{table}" (\n    {body}\n);'


SCHEMA_SQL = "\n".join(create_table_sql(table) for table in TABLES)

#: One row per denied tool call, joined to whatever the session did next.
DENIAL_FOLLOWUPS_VIEW = "denial_followups"

#: A view rather than a table.  It is derived entirely from ``tool_calls``, so
#: computing it on read means it cannot disagree with the rows it summarises and
#: the incremental build has nothing extra to maintain.
#:
#: "Next" is the first tool call of the same ``session_id`` on a *strictly
#: later* transcript line, and nothing else -- not the parentUuid tree, not
#: sidechain structure.  ``seq`` is the transcript line number, so several
#: ``tool_use`` blocks emitted on one assistant line share it, and those
#: siblings were all issued before the model saw any of their results: a
#: sibling cannot be a reaction to the denial, so requiring a greater ``seq``
#: excludes it by construction.  Among the calls of that later line
#: ``tool_use_id`` picks one; block order within a line is not recorded
#: anywhere, so that tiebreak is a *stable* choice rather than a faithful one --
#: it exists so two builds of the same bytes agree, and it now only ever chooses
#: between calls the model issued at the same moment.
#:
#: ``followup_kind`` is mechanical, never a judgement about whether the retry
#: was legitimate: same tool and byte-identical input is ``verbatim-retry``,
#: same tool alone is ``same-tool``, a different tool is ``other-tool``, and a
#: denial with no later line that called a tool is ``none`` with NULL ``next_*``.
DENIAL_FOLLOWUPS_SQL = f'''
CREATE OR REPLACE VIEW "{DENIAL_FOLLOWUPS_VIEW}" AS
WITH line_heads AS (
    -- One row per transcript line that issued tool calls: the call that a
    -- denial on an earlier line is paired with, chosen by tool_use_id.
    SELECT session_id, seq, ts, tool_name, input, input_summary, outcome
    FROM "tool_calls"
    QUALIFY row_number() OVER (PARTITION BY session_id, seq ORDER BY tool_use_id) = 1
),
following AS (
    -- lead() over one row per line, so the follow-up is always a later line;
    -- the siblings of a parallel tool_use block never enter this window.
    SELECT
        session_id,
        seq,
        lead(seq) OVER later AS next_seq,
        lead(tool_name) OVER later AS next_tool_name,
        lead(input) OVER later AS next_input,
        lead(input_summary) OVER later AS next_input_summary,
        lead(outcome) OVER later AS next_outcome,
        lead(ts) OVER later AS next_ts
    FROM line_heads
    WINDOW later AS (PARTITION BY session_id ORDER BY seq)
)
SELECT
    denied.session_id,
    denied.seq,
    denied.ts,
    denied.tool_name,
    denied.input_summary,
    denied.permission_mode,
    denied.cwd,
    following.next_tool_name,
    following.next_input_summary,
    following.next_outcome,
    following.next_ts,
    date_diff('millisecond', denied.ts, following.next_ts) / 1000.0 AS gap_seconds,
    CASE
        -- next_seq, not next_tool_name: a tool_use block can carry no name at
        -- all, and "there was no next call" must not be confused with "the next
        -- call was anonymous".
        WHEN following.next_seq IS NULL THEN 'none'
        WHEN denied.tool_name IS NOT DISTINCT FROM following.next_tool_name
            AND CAST(denied.input AS VARCHAR)
                IS NOT DISTINCT FROM CAST(following.next_input AS VARCHAR)
            THEN 'verbatim-retry'
        WHEN denied.tool_name IS NOT DISTINCT FROM following.next_tool_name THEN 'same-tool'
        ELSE 'other-tool'
    END AS followup_kind
FROM "tool_calls" AS denied
JOIN following
    -- Every call's own line is in `following`, so this join keeps every denial;
    -- IS NOT DISTINCT FROM so one with no session id still finds it.
    ON following.session_id IS NOT DISTINCT FROM denied.session_id
    AND following.seq = denied.seq
WHERE denied.outcome = 'denied';
'''

#: Every ``followup_kind`` the view can produce.
FOLLOWUP_KINDS: tuple[str, ...] = ("verbatim-retry", "same-tool", "other-tool", "none")

#: One row per completed kaiba ``recall`` call, joined to same-session
#: followup evidence.  Unlike ``denial_followups`` this is a thin view over a
#: real table (``recall_calls``): the followup pairing crosses source
#: formats and is computed once, at build time, by :mod:`ashiato.recall` --
#: see that module's docstring for why a read-time view over raw events
#: does not work the way it does for denials.
RECALL_FOLLOWUPS_VIEW = "recall_followups"

RECALL_FOLLOWUPS_SQL = f'''
CREATE OR REPLACE VIEW "{RECALL_FOLLOWUPS_VIEW}" AS
SELECT
    recall_id,
    session_id,
    file_path,
    source,
    seq,
    ts,
    call_id,
    query,
    output,
    output_truncated,
    followup_text,
    followup_truncated,
    overlap_tokens,
    overlap_count
FROM "recall_calls";
'''

#: Every view a current database must have; checked by ``assert_readable``.
REQUIRED_VIEWS: tuple[str, ...] = (DENIAL_FOLLOWUPS_VIEW, RECALL_FOLLOWUPS_VIEW)


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
