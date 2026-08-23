"""Nominating un-bookkept work-state changes salvaged from transcripts.

When a session ends abnormally (freeze, kill, context exhaustion), the
bookkeeping a healthy session does after a state-changing action -- updating
the shared kaiba agenda after a chain terminates or a publish is confirmed --
silently does not happen.  The evidence that the state change occurred is
still in the transcript: the tool call itself.  This module looks for such
evidence with no bookkeeping trail and reports it as a *nomination candidate*
for a human or orchestrator to adjudicate.

Nomination only: nothing here writes to the kaiba agenda, the actions ledger,
or any other store, mirroring the discipline of ``ashiato.recall`` (issue
#10) and ``denial_followups`` -- derivation nominates, the inspecting tier
files.  Pure report-time analysis over an already-built DuckDB and the kaiba
SQLite ledger; no new stored tables or views, so this needs no
``FORMAT_VERSION`` bump.

A candidate is nominated when *both* hold:

1. no successful ``mcp__kaiba__agenda_edit`` call exists in the same session
   at or after the evidence timestamp, and
2. the kaiba ``actions`` ledger has no row whose ``created_at`` or
   ``done_at`` falls in ``[ts, ts + window]`` -- coverage from a different
   session or agent.

Check 2 is skipped, not treated as a failure, when the kaiba db is absent or
unreadable; the caller is responsible for telling the user coverage is then
transcript-only.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb

#: One evidence kind -> the SQL predicate (over ``tool_calls``) that finds it.
#: A tuple, not a dict of separate constants, so a new evidence signal is a
#: one-entry addition here and nowhere else -- the issue frames this list as
#: a starting point, not a frozen set.
EVIDENCE_SIGNALS: tuple[tuple[str, str], ...] = (
    (
        "publish",
        "tool_name IN ('mcp__sunaba__publish', 'mcp__code-sandbox-mcp__publish') "
        "AND outcome = 'ok'",
    ),
    (
        "chain-wait",
        "tool_name = 'Bash' AND input_summary LIKE '%chain-wait%'",
    ),
)

#: The predicate that finds a successful bookkeeping event.
BOOKKEEPING_PREDICATE = "tool_name = 'mcp__kaiba__agenda_edit' AND outcome = 'ok'"

#: Default kaiba coverage window and nomination cap; both CLI-tunable.
DEFAULT_WINDOW_MINUTES = 30
DEFAULT_LIMIT = 50

#: A nomination's snippet is meant to fit on one line, same convention as
#: ``ashiato.parser.INPUT_SUMMARY_LIMIT``.
SNIPPET_LIMIT = 200


@dataclass(slots=True)
class EvidenceEvent:
    """One row of ``tool_calls`` matched by an entry of :data:`EVIDENCE_SIGNALS`."""

    kind: str
    tool_use_id: str
    session_id: str | None
    ts: datetime | None
    input_summary: str | None
    result_text: str | None


@dataclass(slots=True)
class Nomination:
    """One evidence event with no bookkeeping trail found for it."""

    kind: str
    tool_use_id: str
    session_id: str | None
    ts: datetime
    snippet: str
    #: Which coverage check(s) found nothing: "session", "kaiba", or both.
    failed_checks: tuple[str, ...]


def default_kaiba_db_path() -> Path:
    return Path.home() / ".kaiba" / "kaiba.db"


def open_kaiba(path: Path, *, probe_table: str = "actions") -> sqlite3.Connection | None:
    """Open a kaiba sqlite db read-only, or ``None`` when it cannot be used.

    Absence and corruption are both treated as "no coverage data available"
    rather than errors: the caller falls back to a degraded mode and says so,
    instead of failing the whole command.  *probe_table* is the table this
    caller actually needs -- ``salvage`` reads ``actions``, the Cursor recall
    join in :mod:`ashiato.build` reads ``recalls`` -- so a kaiba db missing
    the table a given caller depends on is treated the same as one missing
    entirely, rather than handed back as unusable in a way the caller does
    not find out about until its first query fails.  Always a literal from
    calling code, never a value from the CLI, so there is no injection risk
    in composing it into SQL.
    """
    if not path.exists():
        return None
    uri = f"file:{path.as_posix()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error:
        return None
    try:
        connection.execute(f"SELECT 1 FROM {probe_table} LIMIT 1")
    except sqlite3.Error:
        connection.close()
        return None
    return connection


def parse_kaiba_ts(value: str | None) -> datetime | None:
    """ISO-8601 kaiba timestamp to a naive UTC datetime, matching DuckDB's ``ts``."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def fetch_kaiba_action_timestamps(connection: sqlite3.Connection) -> list[datetime]:
    """Every ``created_at`` / ``done_at`` in the actions ledger, normalised to naive UTC."""
    rows = connection.execute("SELECT created_at, done_at FROM actions").fetchall()
    timestamps: list[datetime] = []
    for created_at, done_at in rows:
        for value in (created_at, done_at):
            parsed = parse_kaiba_ts(value)
            if parsed is not None:
                timestamps.append(parsed)
    return timestamps


def fetch_evidence_events(
    connection: duckdb.DuckDBPyConnection, *, since: datetime | None = None
) -> list[EvidenceEvent]:
    """Every row matched by an :data:`EVIDENCE_SIGNALS` predicate, newest first."""
    events: list[EvidenceEvent] = []
    for kind, predicate in EVIDENCE_SIGNALS:
        query = (
            "SELECT tool_use_id, session_id, ts, input_summary, result_text "
            f'FROM "tool_calls" WHERE {predicate}'
        )
        params: list[object] = []
        if since is not None:
            query += " AND ts >= ?"
            params.append(since)
        events.extend(
            EvidenceEvent(
                kind=kind,
                tool_use_id=row[0],
                session_id=row[1],
                ts=row[2],
                input_summary=row[3],
                result_text=row[4],
            )
            for row in connection.execute(query, params).fetchall()
        )
    # Newest first, matching the ``denials`` / ``recalls`` convention; an event
    # with no timestamp sorts last since there is no "newest" to place it by.
    # Compared as naive datetimes, not via ``.timestamp()`` -- that would
    # interpret naive UTC as local time, which only sorts stably when the
    # local zone has no DST.
    events.sort(key=lambda event: event.ts or datetime.min, reverse=True)
    return events


def fetch_bookkeeping_by_session(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str | None, list[datetime]]:
    """session_id -> every successful ``agenda_edit`` timestamp in that session."""
    query = f'SELECT session_id, ts FROM "tool_calls" WHERE {BOOKKEEPING_PREDICATE}'
    by_session: dict[str | None, list[datetime]] = {}
    for session_id, ts in connection.execute(query).fetchall():
        by_session.setdefault(session_id, []).append(ts)
    return by_session


def _snippet(event: EvidenceEvent) -> str:
    text = event.input_summary or event.result_text or ""
    return text[:SNIPPET_LIMIT]


def decide(
    event: EvidenceEvent,
    session_bookkeeping_ts: list[datetime],
    kaiba_action_ts: list[datetime] | None,
    *,
    window: timedelta,
) -> tuple[str, ...] | None:
    """The failed checks that justify a nomination, or ``None`` when covered.

    ``kaiba_action_ts`` is ``None`` exactly when kaiba coverage is
    unavailable for this run; that check is then skipped rather than counted
    as a failure, so the session check alone decides.
    """
    if event.ts is None:
        return None
    session_covered = any(ts >= event.ts for ts in session_bookkeeping_ts)
    if session_covered:
        return None
    if kaiba_action_ts is None:
        return ("session",)
    window_end = event.ts + window
    kaiba_covered = any(event.ts <= ts <= window_end for ts in kaiba_action_ts)
    if kaiba_covered:
        return None
    return ("session", "kaiba")


def nominate(
    duckdb_connection: duckdb.DuckDBPyConnection,
    kaiba_connection: sqlite3.Connection | None,
    *,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
    since: datetime | None = None,
    limit: int = DEFAULT_LIMIT,
) -> list[Nomination]:
    """Evidence events with no bookkeeping trail, newest first, capped at *limit*.

    ``limit`` of 0 means no cap, matching the rest of this project's CLI
    ``--limit`` convention.
    """
    window = timedelta(minutes=window_minutes)
    evidence_events = fetch_evidence_events(duckdb_connection, since=since)
    bookkeeping_by_session = fetch_bookkeeping_by_session(duckdb_connection)
    kaiba_timestamps = (
        fetch_kaiba_action_timestamps(kaiba_connection) if kaiba_connection is not None else None
    )

    nominations: list[Nomination] = []
    for event in evidence_events:
        if limit and len(nominations) >= limit:
            break
        failed_checks = decide(
            event,
            bookkeeping_by_session.get(event.session_id, []),
            kaiba_timestamps,
            window=window,
        )
        if failed_checks is None:
            continue
        nominations.append(
            Nomination(
                kind=event.kind,
                tool_use_id=event.tool_use_id,
                session_id=event.session_id,
                ts=event.ts,  # type: ignore[arg-type]  # decide() rejects ts=None
                snippet=_snippet(event),
                failed_checks=failed_checks,
            )
        )
    return nominations
