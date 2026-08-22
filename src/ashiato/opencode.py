"""Deterministic parsing of opencode job event streams (events.ndjson).

Mirrors the contract in :mod:`ashiato.parser`: every function here is a pure
function of the bytes on disk, no network access, no model calls, no clock
reads.  A missing or unexpected field becomes ``NULL``, never an exception,
and a malformed line -- or an event type this module does not model -- is
skipped and counted, not fatal, the same discipline as a truncated Claude
Code transcript line.

Only what the recall-followup view needs is extracted: *completed* tool-call
parts (whichever tool -- the recall filter is applied downstream, by
:mod:`ashiato.recall`) and assistant text, both needed to build the
post-recall "what did the session do next" evidence.  A *pending* tool part
carries no output yet and is not modelled at all; only the record on its
``completed`` state produces anything.  General-purpose ingestion of every
opencode event type is deliberately out of scope -- job lifecycle events and
tool parts for tools other than what callers care about are simply not
recognised and are skipped like any other unmodelled type.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_PART_TYPE_TOOL = "tool"
_PART_TYPE_TEXT = "text"

_EVENT_PART_UPDATED = "message.part.updated"
_EVENT_PART_DELTA = "message.part.delta"


@dataclass(slots=True)
class OpenCodeToolCall:
    """One *completed* tool-call part; a pending one produces no instance."""

    call_id: str
    session_id: str | None
    file_path: str
    seq: int
    ts: datetime | None
    tool: str | None
    input: dict | None
    output: str | None


@dataclass(slots=True)
class OpenCodeTextChunk:
    """One piece of assistant text: a full ``text`` part, or one delta of one."""

    session_id: str | None
    file_path: str
    seq: int
    text: str


@dataclass(slots=True)
class ParsedOpenCodeFile:
    """Everything one events.ndjson file contributes to recall extraction."""

    file_path: str
    tool_calls: list[OpenCodeToolCall]
    text_chunks: list[OpenCodeTextChunk]
    n_parse_errors: int


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def parse_epoch_millis(value: object) -> datetime | None:
    """Epoch milliseconds to a naive UTC datetime; ``None`` when unparseable."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        return datetime.fromtimestamp(value / 1000, tz=UTC).replace(tzinfo=None)
    except (OverflowError, OSError, ValueError):
        return None


def _read_records(path: Path) -> tuple[list[tuple[int, dict]], int]:
    """(seq, record) for every parseable line, plus the error count.

    Same discipline as ``parser._read_records``: a line that fails to parse
    as JSON, or that parses to something other than an object, is skipped
    and counted -- including a truncated final line of a job still being
    written.
    """
    records: list[tuple[int, dict]] = []
    n_errors = 0
    with open(path, encoding="utf-8-sig", errors="replace") as handle:
        for seq, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except ValueError:
                n_errors += 1
                continue
            if not isinstance(record, dict):
                n_errors += 1
                continue
            records.append((seq, record))
    return records, n_errors


def _tool_call(
    *, file_path: str, seq: int, properties: dict, part: dict
) -> OpenCodeToolCall | None:
    """A completed tool part, or ``None`` for a pending one or one with no state."""
    state = _as_dict(part.get("state"))
    if state.get("status") != "completed":
        return None
    call_id = _as_str(part.get("callID")) or _as_str(part.get("id")) or f"{file_path}:{seq}"
    time_obj = _as_dict(state.get("time"))
    ts = parse_epoch_millis(time_obj.get("start")) or parse_epoch_millis(properties.get("time"))
    tool_input = state.get("input")
    return OpenCodeToolCall(
        call_id=call_id,
        session_id=_as_str(properties.get("sessionID")) or _as_str(part.get("sessionID")),
        file_path=file_path,
        seq=seq,
        ts=ts,
        tool=_as_str(part.get("tool")),
        input=tool_input if isinstance(tool_input, dict) else None,
        output=_as_str(state.get("output")),
    )


def _text_chunk(
    *, file_path: str, seq: int, properties: dict, part: dict
) -> OpenCodeTextChunk | None:
    text = _as_str(part.get("text"))
    if not text:
        return None
    return OpenCodeTextChunk(
        session_id=_as_str(properties.get("sessionID")) or _as_str(part.get("sessionID")),
        file_path=file_path,
        seq=seq,
        text=text,
    )


def _delta_chunk(*, file_path: str, seq: int, properties: dict) -> OpenCodeTextChunk | None:
    if properties.get("field") != "text":
        return None
    delta = _as_str(properties.get("delta"))
    if not delta:
        return None
    return OpenCodeTextChunk(
        session_id=_as_str(properties.get("sessionID")),
        file_path=file_path,
        seq=seq,
        text=delta,
    )


def parse_file(path: str | Path) -> ParsedOpenCodeFile:
    """Parse one events.ndjson file into completed tool calls and text chunks.

    A file with no parseable lines yields empty lists -- a normal outcome,
    not an error, the same as an empty Claude Code transcript.

    A completed tool part is recorded at most once per (sessionID, callID); if
    a stream carries the same part's completed state more than once, the last
    occurrence wins.  On streams without such duplicates the output is identical
    to a simple append.
    """
    path = Path(path)
    file_path = str(path.resolve())
    records, n_parse_errors = _read_records(path)

    # A completed tool part is keyed by (sessionID, callID); if the same part's
    # completed state is ever carried twice, the LAST occurrence wins (later
    # state supersedes earlier) so the same call never becomes two rows.
    seen_calls: dict[tuple[str | None, str], OpenCodeToolCall] = {}
    call_order: list[tuple[str | None, str]] = []
    text_chunks: list[OpenCodeTextChunk] = []

    for seq, record in records:
        event_type = record.get("type")
        properties = _as_dict(record.get("properties"))
        if event_type == _EVENT_PART_UPDATED:
            part = _as_dict(properties.get("part"))
            part_type = part.get("type")
            if part_type == _PART_TYPE_TOOL:
                call = _tool_call(file_path=file_path, seq=seq, properties=properties, part=part)
                if call is not None:
                    key = (call.session_id, call.call_id)
                    if key not in seen_calls:
                        call_order.append(key)
                    seen_calls[key] = call
            elif part_type == _PART_TYPE_TEXT:
                chunk = _text_chunk(
                    file_path=file_path, seq=seq, properties=properties, part=part
                )
                if chunk is not None:
                    text_chunks.append(chunk)
            # Any other part type is not modelled; skipped, not an error.
        elif event_type == _EVENT_PART_DELTA:
            chunk = _delta_chunk(file_path=file_path, seq=seq, properties=properties)
            if chunk is not None:
                text_chunks.append(chunk)
        # Any other event type -- job lifecycle events and the like -- is
        # not modelled; skipped, not an error.

    tool_calls = [seen_calls[key] for key in call_order]
    return ParsedOpenCodeFile(
        file_path=file_path,
        tool_calls=tool_calls,
        text_chunks=text_chunks,
        n_parse_errors=n_parse_errors,
    )
