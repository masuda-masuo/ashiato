"""A deterministic per-session topic outline: ``ashiato topics`` (issue #51).

A session's stored title (Claude Code's ``ai-title``) is generated once from
the first prompt and never updated, so a long session's label says nothing
about what it was really about.  This module derives an *outline* from the
session's own words instead -- no LLM, no embeddings, purely deterministic:
the same database always produces the same segments.

The outline is built in four steps:

1. **Exchanges.**  The session's user/assistant text rows in transcript order,
   excluding meta and sidechain rows, tool-result user rows and compaction
   summaries (machine-written recaps of earlier content -- see
   ``ashiato.orphans.is_compaction_summary``), and deduplicated by the row's
   ``uuid`` (a resumed session re-persists the same messages, so the same
   ``uuid`` can appear several times; the first occurrence wins).  Each
   non-empty human row opens an exchange; the following assistant text is
   appended to it, and assistant text before the first human row is ignored.
2. **Weights.**  Every term is weighted tf-idf, with document frequency taken
   over the whole database exactly as ``orphans`` computes it
   (``ashiato.orphans.collect_sessions``).  Terms whose document frequency
   exceeds 30% of sessions are dropped: they describe the corpus, not the
   session.
3. **Boundaries.**  TextTiling-style: for every gap between exchange ``i-1``
   and ``i`` the cosine similarity between the summed tf-idf vectors of the
   ``window`` exchanges before and after the gap is computed (truncated at the
   edges).  A gap starts a new segment when its similarity is a local minimum
   and below ``mean - 0.5 * sd`` of all gap similarities, and the gap has a
   full ``window`` of exchanges on each side.  A session with fewer than
   ``2 * window`` exchanges is one segment.
4. **Segments.**  Each segment carries its exchange range, start/end
   timestamps, exchange count, the top ``terms`` by in-segment tf-idf, and the
   opening (the segment's first human text, whitespace-collapsed, 160 chars).

The session header carries ``session_id``, ``project_dir``, the latest
``custom-title`` if the human set one (else the latest ``ai-title``) and the
exchange count.  Like ``orphans`` this is report-only: it reads the database
and writes nothing.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from ashiato.build import SchemaOutOfDate, assert_readable, connect
from ashiato.orphans import (
    collect_sessions,
    is_compaction_summary,
    strip_harness,
    tokenize,
)
from ashiato.session_trace import SessionResolutionError, resolve_session

#: Exchanges on each side of a gap whose summed vectors are compared.
DEFAULT_WINDOW = 3

#: Topic terms shown per segment.
DEFAULT_TERMS = 8

#: The opening of a segment is the first human text, collapsed and cut here.
OPENING_CHARS = 160

#: Terms whose document frequency exceeds this share of sessions are dropped:
#: they are corpus-wide vocabulary, not session topics.
MAX_DF_RATIO = 0.3

#: The session's own text rows, in transcript order, with enough context to
#: rebuild the exchange stream (``raw`` holds the uuid and the compaction
#: flag).  Ordered by file then line so two builds of the same bytes agree.
_SESSION_QUERY = """
    SELECT event_id, seq, ts, role, text, raw, is_meta, is_sidechain
    FROM events
    WHERE session_id = ?
      AND NOT COALESCE(is_meta, FALSE)
      AND NOT COALESCE(is_sidechain, FALSE)
    ORDER BY file_path, seq
"""

#: Title events: the human's ``custom-title`` and Claude Code's ``ai-title``,
#: in transcript order (the latest of each kind is the one that wins).
_TITLE_QUERY = """
    SELECT type, raw
    FROM events
    WHERE session_id = ? AND type IN ('ai-title', 'custom-title')
    ORDER BY file_path, seq
"""

#: The tool-result marker, matched on ``raw`` exactly as ``orphans`` does.
_TOOL_RESULT_MARKER = '"tool_result"'


@dataclass(slots=True)
class Exchange:
    """One human turn plus the assistant text that follows it."""

    index: int
    seq: int
    ts: datetime | None
    human_text: str
    assistant_text: str = ""
    terms: Counter[str] = field(default_factory=Counter)

    @property
    def text(self) -> str:
        """Human and assistant text joined the way ``events.text`` does."""
        return "\n".join(part for part in (self.human_text, self.assistant_text) if part)


def _uuid_of(raw: str | None) -> str | None:
    """The ``uuid`` field of a row's ``raw`` JSON, or ``None`` when unreadable.

    ``raw`` is not always valid JSON (other source formats store different
    text there), so this parses defensively and never raises.
    """
    if not raw:
        return None
    try:
        record = json.loads(raw)
    except ValueError:
        return None
    value = record.get("uuid")
    return value if isinstance(value, str) and value else None


def _fmt_ts(value: datetime | None) -> str | None:
    """The naive-UTC ``YYYY-MM-DD HH:MM:SS`` rendering the other commands use."""
    return value.strftime("%Y-%m-%d %H:%M:%S") if value is not None else None


def _is_tool_result(role: str, raw: str | None) -> bool:
    """True for a user row that carries a tool result (not typed by a human)."""
    return role == "user" and raw is not None and _TOOL_RESULT_MARKER in raw


def _exchanges(rows: Sequence[tuple[Any, ...]]) -> list[Exchange]:
    """The deduplicated exchange stream of a session's event rows.

    *rows* are the ``_SESSION_QUERY`` tuples.  Meta/sidechain rows are already
    excluded by the query; tool results, compaction summaries and uuid
    duplicates (of either role) are filtered here, and only non-empty human
    rows open an exchange.
    """
    exchanges: list[Exchange] = []
    seen_uuids: set[str] = set()
    current: Exchange | None = None
    for event_id, seq, ts, role, text, raw, _is_meta, _is_sidechain in rows:
        if role == "user":
            if _is_tool_result(role, raw) or is_compaction_summary(raw):
                continue
        elif role != "assistant":
            continue
        # A resumed session re-persists the same messages: whichever role, the
        # first occurrence of a uuid wins.
        uuid = _uuid_of(raw) or event_id
        if uuid in seen_uuids:
            continue
        seen_uuids.add(uuid)
        if role == "user":
            prose = strip_harness(text).strip()
            if not prose:
                continue
            current = Exchange(index=len(exchanges), seq=seq, ts=ts, human_text=prose)
            exchanges.append(current)
        elif current is not None and text:
            current.assistant_text = (
                text if not current.assistant_text else current.assistant_text + "\n" + text
            )
    for exchange in exchanges:
        exchange.terms.update(tokenize(exchange.text))
    return exchanges


def _idf(sessions: Sequence[Any]) -> dict[str, float]:
    """idf per corpus term, with the 30% document-frequency cutoff applied.

    *sessions* is the whole ``orphans`` corpus (every session with prose), so
    a term that appears in every session has weight 0 and never steers a
    boundary.  Compaction summaries are already excluded by
    ``collect_sessions``, so their words do not dilute the signal.
    """
    document_frequency: Counter[str] = Counter()
    for session in sessions:
        document_frequency.update(session.terms.keys())
    n_sessions = len(sessions)
    max_df = MAX_DF_RATIO * n_sessions
    idf: dict[str, float] = {}
    for term, df in document_frequency.items():
        if df > max_df:
            continue
        idf[term] = math.log((n_sessions + 1) / (df + 1))
    return idf


def _vector(exchange: Exchange, idf: dict[str, float]) -> dict[str, float]:
    """The exchange's tf-idf weights: term count times idf, dropped terms absent."""
    vector: dict[str, float] = {}
    for term, count in exchange.terms.items():
        weight = idf.get(term)
        if weight:
            vector[term] = count * weight
    return vector


def _summed(vectors: Sequence[dict[str, float]], start: int, end: int) -> dict[str, float]:
    """The summed vector of exchanges ``[start, end)``."""
    total: dict[str, float] = {}
    for vector in vectors[start:end]:
        for term, weight in vector.items():
            total[term] = total.get(term, 0.0) + weight
    return total


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity of two tf-idf vectors; 0 when either is empty."""
    if not a or not b:
        return 0.0
    dot = 0.0
    for term, weight in a.items():
        if term in b:
            dot += weight * b[term]
    norm_a = math.sqrt(sum(weight * weight for weight in a.values()))
    norm_b = math.sqrt(sum(weight * weight for weight in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _boundaries(
    n_exchanges: int, vectors: Sequence[dict[str, float]], window: int
) -> list[int]:
    """The gaps (between exchange ``g-1`` and ``g``) that start a new segment.

    A gap is a boundary when it has a full ``window`` of exchanges on each
    side, its similarity is a local minimum, and that minimum is below
    ``mean - 0.5 * sd`` of all gap similarities.  Sessions with fewer than
    ``2 * window`` exchanges have no boundary: they are one segment.
    """
    if n_exchanges < 2 * window:
        return []
    sims: list[float] = []
    for gap in range(1, n_exchanges):
        before = _summed(vectors, max(0, gap - window), gap)
        after = _summed(vectors, gap, min(n_exchanges, gap + window))
        sims.append(_cosine(before, after))
    mean = statistics.fmean(sims)
    sd = statistics.pstdev(sims)
    threshold = mean - 0.5 * sd
    boundaries: list[int] = []
    for index, similarity in enumerate(sims):
        gap = index + 1
        if gap < window or gap > n_exchanges - window:
            continue  # no full window on each side: not a usable boundary
        if index == 0 or index == len(sims) - 1:
            continue  # the very first/last gap has no neighbour to be a minimum against
        if similarity >= sims[index - 1] or similarity >= sims[index + 1]:
            continue  # not a local minimum
        if similarity >= threshold:
            continue
        boundaries.append(gap)
    return boundaries


def _segments(
    exchanges: Sequence[Exchange],
    vectors: Sequence[dict[str, float]],
    boundaries: Sequence[int],
    n_terms: int,
) -> list[dict[str, Any]]:
    """One dict per segment, from its exchange range to its top terms."""
    if not exchanges:
        return []  # a tool-only session has no exchanges, hence no outline
    segments: list[dict[str, Any]] = []
    starts = [0, *boundaries]
    ends = [*boundaries, len(exchanges)]
    for start, end in zip(starts, ends, strict=True):
        segment_weights: dict[str, float] = {}
        for exchange in exchanges[start:end]:
            for term, weight in vectors[exchange.index].items():
                segment_weights[term] = segment_weights.get(term, 0.0) + weight
        ranked = sorted(
            segment_weights.items(), key=lambda item: (-item[1], item[0])
        )[:n_terms]
        opening = " ".join(exchanges[start].human_text.split())[:OPENING_CHARS]
        segments.append(
            {
                "start": start,
                "end": end - 1,
                "start_ts": _fmt_ts(exchanges[start].ts),
                "end_ts": _fmt_ts(exchanges[end - 1].ts),
                "n_exchanges": end - start,
                "terms": [term for term, _ in ranked],
                "opening": opening,
            }
        )
    return segments


def _title(
    connection: duckdb.DuckDBPyConnection, session_id: str
) -> tuple[str | None, str | None]:
    """``(title, kind)``: the latest ``custom-title`` if any, else the latest ``ai-title``."""
    latest: dict[str, str] = {}
    for type_, raw in connection.execute(_TITLE_QUERY, [session_id]).fetchall():
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except ValueError:
            continue
        title = None
        for key in ("title", "aiTitle", "text"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                title = value.strip()
                break
        if title:
            latest[type_] = title
    if "custom-title" in latest:
        return latest["custom-title"], "custom-title"
    if "ai-title" in latest:
        return latest["ai-title"], "ai-title"
    return None, None


def outline(
    connection: duckdb.DuckDBPyConnection,
    session_id: str,
    *,
    window: int = DEFAULT_WINDOW,
    terms: int = DEFAULT_TERMS,
) -> dict[str, Any]:
    """The full outline payload for one session."""
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    sessions = collect_sessions(connection)
    idf = _idf(sessions)
    rows = connection.execute(_SESSION_QUERY, [session_id]).fetchall()
    exchanges = _exchanges(rows)
    vectors = [_vector(exchange, idf) for exchange in exchanges]
    boundaries = _boundaries(len(exchanges), vectors, window)
    title, title_kind = _title(connection, session_id)
    project_dir = connection.execute(
        "SELECT project_dir FROM sessions WHERE session_id = ? "
        "ORDER BY started_at NULLS LAST, file_path LIMIT 1",
        [session_id],
    ).fetchone()
    return {
        "session_id": session_id,
        "project_dir": project_dir[0] if project_dir else None,
        "title": title,
        "title_kind": title_kind,
        "n_exchanges": len(exchanges),
        "segments": _segments(exchanges, vectors, boundaries, terms),
    }


def _print_text(payload: dict[str, Any], out: Any) -> None:
    """The table form of an outline: header plus one block per segment."""
    title = payload["title"]
    kind = payload["title_kind"]
    title_line = f"title: {title} ({kind})" if title else "title: (none)"
    print(f"session {payload['session_id']}  {payload['project_dir'] or '?'}", file=out)
    print(title_line, file=out)
    print(f"exchanges: {payload['n_exchanges']}", file=out)
    for index, segment in enumerate(payload["segments"], start=1):
        start_ts = segment["start_ts"] or "?"
        end_ts = segment["end_ts"] or "?"
        print("", file=out)
        print(f"[{index}] {start_ts} .. {end_ts}  ({segment['n_exchanges']} exchanges)", file=out)
        print(f"  terms: {', '.join(segment['terms']) or '(none)'}", file=out)
        print(f"  first: {segment['opening']}", file=out)


def run(
    db_path: Path,
    session_prefix: str,
    *,
    window: int = DEFAULT_WINDOW,
    terms: int = DEFAULT_TERMS,
    json_output: bool = False,
    out: Any = None,
    err: Any = None,
) -> int:
    """Print one session's outline.  Returns 0, or 1 if the db is unreadable.

    Prefix resolution and every error message match ``session-trace`` exactly:
    an exact session id wins, otherwise the prefix must name exactly one of
    the union of ids in ``sessions`` and ``tool_calls``.
    """
    if out is None:
        out = sys.stdout
    if err is None:
        err = sys.stderr

    if not db_path.exists():
        print(f"error: no database at {db_path} (run 'ashiato build' first)", file=err)
        return 1

    connection = connect(db_path, read_only=True)
    try:
        assert_readable(connection)
        session_id = resolve_session(connection, session_prefix)
        payload = outline(connection, session_id, window=window, terms=terms)
    except SessionResolutionError as error:
        print(f"error: {error}", file=err)
        return 1
    except SchemaOutOfDate as error:
        print(f"error: {error}", file=err)
        return 1
    except duckdb.Error as error:
        print(f"error: {error}", file=err)
        return 1
    finally:
        connection.close()

    if json_output:
        print(json.dumps(payload, indent=2, ensure_ascii=False), file=out)
    else:
        _print_text(payload, out)
    return 0