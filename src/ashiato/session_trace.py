"""One session's interleaved text/tool timeline: ``ashiato session-trace``.

A deterministic, read-only, single-session trace: the session's text events
and tool calls, interleaved in transcript order.  ``seq`` is the transcript
line number, so several rows can share it (an assistant line emits its text
before its ``tool_use`` blocks, and a line can carry several parallel
calls).  Ordering is ``(seq, kind, stable id)``: within one line, text rows
come before tool rows, then the row's own id (``event_id`` for a text row,
``tool_use_id`` for a tool row) ascending -- the same kind of stable choice
``denial_followups`` makes, so two builds of the same bytes agree.

Everything here is read-only and derived from persisted rows: the same
database always produces the same trace.  The JSON shape is deliberately
flat and stable -- ``session`` / ``coverage`` / ``timeline`` -- so a later
MCP wrapper can consume it without re-deriving ordering rules.

The session is resolved against the union of ids in ``sessions`` and
``tool_calls``, so a session whose rows were persisted only as tool calls
(the older Codex shape) still resolves.  An exact id wins; otherwise the
prefix must match exactly one id.  A missing or ambiguous prefix is a clean
error, never a partial trace.

A recall row's follow-up evidence is assembled from the trace's own rows on
strictly later lines -- the same rows the timeline shows, meta events
included or excluded identically -- rather than from the stored
``recall_calls.followup_text``, which was computed over build-time activity
that may include harness noise the trace never displays.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import duckdb

from ashiato.recall import FOLLOWUP_CHAR_LIMIT, FOLLOWUP_ITEM_LIMIT

#: Enough rows to read a whole session on one screen; ``--limit 0`` = all,
#: the same convention as ``denials`` / ``recalls`` / ``grep``.
DEFAULT_LIMIT = 200

#: How much of each text/result excerpt to show by default;
#: ``--max-excerpt-chars 0`` = uncapped, the same ``0 = no cap`` convention.
DEFAULT_EXCERPT_CHARS = 500

#: A truncated excerpt is the first N characters plus exactly this one
#: marker, so ``len(excerpt) <= N + 1`` always holds.
EXCERPT_MARKER = "…"


class SessionResolutionError(ValueError):
    """The prefix names no session, or more than one."""

    def __init__(self, prefix: str, candidates: Sequence[str]) -> None:
        self.prefix = prefix
        self.candidates = list(candidates)
        if len(self.candidates) > 1:
            listing = ", ".join(self.candidates)
            message = (
                f"session prefix '{prefix}' matches {len(self.candidates)} sessions: "
                f"{listing} (use a longer prefix)"
            )
        else:
            message = f"no session matching prefix '{prefix}'"
        super().__init__(message)


def resolve_session(connection: duckdb.DuckDBPyConnection, prefix: str) -> str:
    """The session id *prefix* names, resolved against sessions + tool_calls.

    Exact id wins; otherwise the prefix must match exactly one id of the
    union.  Anything else raises :class:`SessionResolutionError`.
    """
    rows = connection.execute(
        "SELECT session_id FROM sessions WHERE session_id IS NOT NULL "
        "UNION "
        "SELECT session_id FROM tool_calls WHERE session_id IS NOT NULL"
    ).fetchall()
    ids = sorted({row[0] for row in rows})
    if prefix in ids:
        return prefix
    candidates = [sid for sid in ids if sid.startswith(prefix)]
    if len(candidates) == 1:
        return candidates[0]
    raise SessionResolutionError(prefix, candidates)


def _fmt_ts(value: datetime | None) -> str | None:
    """The ``YYYY-MM-DD HH:MM:SS`` (naive UTC) rendering the other read
    commands' JSON output already uses; microseconds are dropped so two rows
    that differ only in sub-second digits still print identically."""
    return value.strftime("%Y-%m-%d %H:%M:%S") if value is not None else None


def _excerpt(text: str | None, max_chars: int) -> tuple[str | None, bool]:
    """(excerpt, truncated) for one row's full text.

    A truncated excerpt is the first ``max_chars`` characters plus the
    one-character :data:`EXCERPT_MARKER`, so ``full.startswith(excerpt[:N])``
    and ``len(excerpt) <= N + 1`` hold.  ``max_chars == 0`` means uncapped.
    ``None`` (a pending tool call has no result) stays ``None`` and is never
    truncated.
    """
    if text is None:
        return None, False
    if max_chars == 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars] + EXCERPT_MARKER, True


def _count_session_rows(
    connection: duckdb.DuckDBPyConnection, table: str, session_id: str
) -> int:
    row = connection.execute(
        f'SELECT count(*) FROM "{table}" WHERE session_id = ?', [session_id]
    ).fetchone()
    return row[0] if row else 0


def _bounded_followup(items: list[str]) -> tuple[str | None, bool]:
    """Join follow-up items, bounded like ``ashiato.recall._bounded_suffix``.

    Same two caps (30 items, 8000 characters) so a real session's post-recall
    churn stays finite; ``None`` when there is nothing after the call.
    """
    truncated = len(items) > FOLLOWUP_ITEM_LIMIT
    bounded = items[:FOLLOWUP_ITEM_LIMIT]
    text = "\n".join(bounded)
    if len(text) > FOLLOWUP_CHAR_LIMIT:
        text = text[:FOLLOWUP_CHAR_LIMIT]
        truncated = True
    return (text or None), truncated


def trace(
    connection: duckdb.DuckDBPyConnection,
    session_id: str,
    *,
    limit: int,
    max_excerpt_chars: int,
) -> dict[str, Any]:
    """The full trace payload for one session: ``session`` / ``coverage`` / ``timeline``.

    *limit* applies *after* ordering, and ``coverage.total`` keeps the
    pre-limit count; ``limit == 0`` means all rows.  Text rows are the
    session's non-empty, non-meta events; tool rows are every ``tool_calls``
    row, carrying the persisted outcome, a bounded result excerpt, recall
    annotation when the call has a ``recall_calls`` row, and follow-up
    evidence when it was denied.
    """
    rows: list[dict[str, Any]] = []
    # Activity, for the recall follow-up evidence: the trace's own rows,
    # ordered exactly as the timeline orders them.  ``(seq, kind_order, id,
    # text)`` -- text before tools at equal seq, then the stable id, the same
    # tie-breaks the timeline itself uses, so the evidence is the session the
    # trace shows, meta events excluded the same way.  Built from the raw
    # rows before the timeline loop, so a recall's follow-up never depends on
    # fetch order.
    activity: list[tuple[int, int, str, str]] = []

    event_rows = connection.execute(
        "SELECT event_id, seq, ts, role, text FROM events "
        "WHERE session_id = ? AND text IS NOT NULL AND text <> '' "
        "AND is_meta IS NOT TRUE",
        [session_id],
    ).fetchall()
    tool_rows = connection.execute(
        "SELECT tool_use_id, seq, ts, tool_name, input_summary, outcome, "
        "result_text FROM tool_calls WHERE session_id = ?",
        [session_id],
    ).fetchall()

    for event_id, seq, ts, role, text in event_rows:
        excerpt, truncated = _excerpt(text, max_excerpt_chars)
        rows.append(
            {
                "kind": "text",
                "id": event_id,
                "seq": seq,
                "ts": _fmt_ts(ts),
                "role": role,
                "excerpt": excerpt,
                "truncated": truncated,
            }
        )
        activity.append((seq, 0, event_id, text))

    for tool_use_id, seq, _, tool_name, input_summary, _, result_text in tool_rows:
        text = " ".join(
            part for part in (tool_name, input_summary, result_text) if part
        )
        if text:
            activity.append((seq, 1, tool_use_id, text))
    activity.sort()

    recall_by_call_id: dict[str, dict[str, Any]] = {}
    for call_id, query, overlap_count in connection.execute(
        "SELECT call_id, query, overlap_count FROM recall_calls WHERE session_id = ?",
        [session_id],
    ).fetchall():
        recall_by_call_id[call_id] = {"query": query, "overlap_count": overlap_count}

    followup_by_seq: dict[int, dict[str, Any]] = {}
    for (
        seq,
        followup_kind,
        next_tool_name,
        next_input_summary,
        next_outcome,
        next_ts,
        gap_seconds,
    ) in connection.execute(
        "SELECT seq, followup_kind, next_tool_name, next_input_summary, "
        "next_outcome, next_ts, gap_seconds FROM denial_followups "
        "WHERE session_id = ?",
        [session_id],
    ).fetchall():
        followup_by_seq[seq] = {
            "followup_kind": followup_kind,
            "next_tool_name": next_tool_name,
            "next_input_summary": next_input_summary,
            "next_outcome": next_outcome,
            "next_ts": _fmt_ts(next_ts),
            "gap_seconds": gap_seconds,
        }

    for (
        tool_use_id,
        seq,
        ts,
        tool_name,
        input_summary,
        outcome,
        result_text,
    ) in tool_rows:
        excerpt, truncated = _excerpt(result_text, max_excerpt_chars)
        is_recall = tool_use_id in recall_by_call_id
        row: dict[str, Any] = {
            "kind": "tool",
            "id": tool_use_id,
            "seq": seq,
            "ts": _fmt_ts(ts),
            "tool_name": tool_name,
            "input_summary": input_summary,
            "outcome": outcome,
            "excerpt": excerpt,
            "truncated": truncated,
            "is_recall": is_recall,
        }
        if is_recall:
            recall = recall_by_call_id[tool_use_id]
            # Follow-up evidence on the recall's own row: the activity on
            # strictly later lines, bounded exactly like the build-time
            # ``recall_calls.followup_text`` but over the trace's rows only.
            followup_text, followup_truncated = _bounded_followup(
                [item[3] for item in activity if item[0] > seq]
            )
            recall["followup_text"] = followup_text
            recall["followup_truncated"] = followup_truncated
            row["recall"] = recall
        if outcome == "denied" and seq in followup_by_seq:
            row["followup"] = followup_by_seq[seq]
        rows.append(row)

    # Ordering: transcript line first, text before tools on the same line,
    # then the stable id ascending.
    rows.sort(key=lambda row: (row["seq"], 0 if row["kind"] == "text" else 1, row["id"]))

    total = len(rows)
    if limit and total > limit:
        rows = rows[:limit]
    returned = len(rows)

    session: dict[str, Any] = {"session_id": session_id}
    session_row = connection.execute(
        "SELECT started_at, ended_at, n_events, n_tool_calls FROM sessions "
        "WHERE session_id = ? ORDER BY started_at NULLS LAST, file_path LIMIT 1",
        [session_id],
    ).fetchone()
    if session_row:
        started_at, ended_at, n_events, n_tool_calls = session_row
        session["started_at"] = _fmt_ts(started_at)
        session["ended_at"] = _fmt_ts(ended_at)
        session["n_events"] = n_events
        session["n_tool_calls"] = n_tool_calls

    coverage: dict[str, Any] = {
        "total": total,
        "returned": returned,
        "truncated": returned < total,
        "has_sessions": _count_session_rows(connection, "sessions", session_id) > 0,
        "has_events": _count_session_rows(connection, "events", session_id) > 0,
        "has_tool_calls": _count_session_rows(connection, "tool_calls", session_id) > 0,
        "has_recall_calls": _count_session_rows(connection, "recall_calls", session_id) > 0,
    }

    return {"session": session, "coverage": coverage, "timeline": rows}
