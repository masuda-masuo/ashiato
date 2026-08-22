"""Regex search over the transcript corpus: ``ashiato grep``.

``ashiato sql`` already exposes the built DuckDB; what this module adds is the
ergonomic layer every "where did I analyse X" investigation ends up
hand-rolling as SQL plus a print loop: a regex over event / tool-call text,
the role / time / session filters those investigations always need, and a
bounded window of text around each hit so the caller can see who said it,
when, in which session, without pulling the whole row.  Deterministic, no
LLM, no embeddings, read-only -- this is grep, not :mod:`ashiato.recall`.

Matching is done entirely in Python with the ``re`` module rather than
DuckDB's own regex engine: RE2 accepts a different pattern dialect (no
backreferences, different lookaround support), and this module's job is to
report *offsets* of a match within a row's text, which ``re.finditer`` gives
for free once the pattern is compiled once.  There is no stored index -- a
full scan of the candidate rows is the design (see the issue's non-goals);
the SQL layer here only applies the cheap filters (role, time range, session
prefix, meta) before any text reaches Python.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

import duckdb

#: Characters of context printed on each side of a match by default.
DEFAULT_CONTEXT = 200

#: Same "0 = no cap" convention as ``denials`` / ``recalls`` / ``salvage``.
DEFAULT_LIMIT = 20

#: Newlines inside a window are replaced with this so one hit's window stays
#: on one printed line, whatever whitespace the original text contained.
NEWLINE_MARKER = "↵"


class InvalidPattern(ValueError):
    """*pattern* is not a valid Python regular expression."""


@dataclass(slots=True)
class Hit:
    """One row whose searched field matched *pattern*, and where."""

    source: str  # "event" or "tool_call"
    id: str  # event_id or tool_use_id, depending on `source`
    session_id: str | None
    ts: datetime | None
    label: str | None  # role for an event hit, tool_name for a tool-call hit
    field: str  # the column that matched: "text", "input_summary", "result_text"
    text: str  # the full text of the matched field
    offsets: list[tuple[int, int]]  # one (start, end) per match kept for this hit


def compile_pattern(pattern: str, *, ignore_case: bool) -> re.Pattern[str]:
    """*pattern* compiled, or :class:`InvalidPattern` for the CLI to report."""
    try:
        return re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as error:
        raise InvalidPattern(str(error)) from error


def window(text: str, start: int, end: int, context: int) -> str:
    """Up to *context* characters of *text* on each side of ``[start, end)``.

    Never longer than ``2 * context`` plus the matched span itself, and safe
    to print on one line: embedded newlines become :data:`NEWLINE_MARKER`.
    """
    lo = max(0, start - context)
    hi = min(len(text), end + context)
    return visible(text[lo:hi])


def visible(text: str) -> str:
    """*text* with embedded newlines replaced so it prints on one line."""
    return text.replace("\r\n", "\n").replace("\n", NEWLINE_MARKER).replace("\r", NEWLINE_MARKER)


def _matches(
    compiled: re.Pattern[str], text: str | None, *, all_matches: bool
) -> list[tuple[int, int]]:
    if not text:
        return []
    if all_matches:
        return [(m.start(), m.end()) for m in compiled.finditer(text)]
    match = compiled.search(text)
    return [(match.start(), match.end())] if match else []


def _fetch_events(
    connection: duckdb.DuckDBPyConnection,
    *,
    role: str | None,
    since: datetime | None,
    until: datetime | None,
    session: str | None,
    include_meta: bool,
) -> list[tuple[str, str | None, datetime | None, str | None, str | None]]:
    query = 'SELECT event_id, session_id, ts, role, text FROM "events" WHERE 1=1'
    params: list[object] = []
    if not include_meta:
        # IS NOT TRUE, not "= false": a row with no is_meta value at all is
        # not harness noise, so it must not be silently dropped either.
        query += " AND is_meta IS NOT TRUE"
    if role:
        query += " AND role = ?"
        params.append(role)
    if since is not None:
        query += " AND ts >= ?"
        params.append(since)
    if until is not None:
        query += " AND ts <= ?"
        params.append(until)
    if session:
        query += " AND starts_with(session_id, ?)"
        params.append(session)
    return connection.execute(query, params).fetchall()


def _fetch_tool_calls(
    connection: duckdb.DuckDBPyConnection,
    *,
    since: datetime | None,
    until: datetime | None,
    session: str | None,
) -> list[tuple[str, str | None, datetime | None, str | None, str | None, str | None]]:
    query = (
        "SELECT tool_use_id, session_id, ts, tool_name, input_summary, result_text "
        'FROM "tool_calls" WHERE 1=1'
    )
    params: list[object] = []
    if since is not None:
        query += " AND ts >= ?"
        params.append(since)
    if until is not None:
        query += " AND ts <= ?"
        params.append(until)
    if session:
        query += " AND starts_with(session_id, ?)"
        params.append(session)
    return connection.execute(query, params).fetchall()


def search(
    connection: duckdb.DuckDBPyConnection,
    pattern: str,
    *,
    role: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    session: str | None = None,
    ignore_case: bool = False,
    include_meta: bool = False,
    tool_calls: bool = False,
    all_matches: bool = False,
    limit: int = DEFAULT_LIMIT,
) -> list[Hit]:
    """Every row matching *pattern*, newest first, capped at *limit* (0 = all).

    ``role`` only ever narrows the ``events`` scope: a tool call has no role
    to filter on, so ``--tool-calls --role user`` still searches every tool
    call regardless of the role filter.
    """
    compiled = compile_pattern(pattern, ignore_case=ignore_case)
    hits: list[Hit] = []

    for event_id, session_id, ts, role_value, text in _fetch_events(
        connection,
        role=role,
        since=since,
        until=until,
        session=session,
        include_meta=include_meta,
    ):
        offsets = _matches(compiled, text, all_matches=all_matches)
        if offsets:
            hits.append(
                Hit(
                    source="event",
                    id=event_id,
                    session_id=session_id,
                    ts=ts,
                    label=role_value,
                    field="text",
                    text=text or "",
                    offsets=offsets,
                )
            )

    if tool_calls:
        for tool_use_id, session_id, ts, tool_name, input_summary, result_text in _fetch_tool_calls(
            connection, since=since, until=until, session=session
        ):
            for field, text in (("input_summary", input_summary), ("result_text", result_text)):
                offsets = _matches(compiled, text, all_matches=all_matches)
                if offsets:
                    hits.append(
                        Hit(
                            source="tool_call",
                            id=tool_use_id,
                            session_id=session_id,
                            ts=ts,
                            label=tool_name,
                            field=field,
                            text=text or "",
                            offsets=offsets,
                        )
                    )

    # Newest first; ties broken by session and id so two runs agree, matching
    # the `denials` / `recalls` / `salvage` convention.
    hits.sort(
        key=lambda hit: (hit.ts or datetime.min, hit.session_id or "", hit.id or ""),
        reverse=True,
    )
    if limit:
        hits = hits[:limit]
    return hits
