"""Deterministic parsing of Cursor agent transcripts.

Mirrors the contract in :mod:`ashiato.opencode`: every function here is a pure
function of the bytes on disk, no network access, no model calls, no clock
reads.  A malformed line is skipped and counted, not fatal, the same
discipline as a truncated Claude Code or opencode transcript line.

A Cursor transcript is one JSON object per line, one file per agent session
(``~/.cursor/projects/<project>/agent-transcripts/<id>/<id>.jsonl`` on a
machine that has them).  The transcript has no ``tool_result`` blocks: it
records what the agent said and what it called, never what came back --
reconstructing the recall *output* from kaiba's own ledger is
:mod:`ashiato.recall`'s job, not this module's.  There is also no per-line
timestamp or id; the file name's uuid stem is the only session identifier,
and a tool call's identity is synthesised from its position (line number,
block index) rather than read off the record.

Next to the transcripts, Cursor keeps ``~/.cursor/chats/<workspace-hash>/<session-uuid>/``
-- an undocumented local store holding a small ``meta.json`` per session plus
a ``store.db`` with the full conversation.  :func:`parse_chat_meta` reads the
``meta.json`` half: the session's working directory and its start/end times.
The ``store.db`` half (tool results, user and system turns) is deliberately
not read yet (issue #87).

Only what the recall-followup view needs is modelled: every assistant
``text`` block, and every ``tool_use`` block (whichever tool -- the recall
filter is applied downstream, by :mod:`ashiato.recall`).  A user line, a bare
status line (``{"status": ..., "type": ...}``), and a ``turn_ended`` line
carry no such content and are skipped like any other unmodelled shape --
that is not a parse error, only an invalid JSON line or a non-object JSON
value is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_BLOCK_TYPE_TEXT = "text"
_BLOCK_TYPE_TOOL_USE = "tool_use"


@dataclass(slots=True)
class CursorToolCall:
    """One ``tool_use`` block -- whichever tool; the recall filter is downstream."""

    call_id: str
    session_id: str
    file_path: str
    seq: int
    block_index: int
    name: str | None
    input: dict | None


@dataclass(slots=True)
class CursorTextChunk:
    """One assistant ``text`` block."""

    session_id: str
    file_path: str
    seq: int
    block_index: int
    text: str


@dataclass(slots=True)
class ParsedCursorFile:
    """Everything one Cursor agent-transcript file contributes to recall extraction."""

    file_path: str
    session_id: str
    tool_calls: list[CursorToolCall]
    text_chunks: list[CursorTextChunk]
    n_parse_errors: int


@dataclass(slots=True)
class ChatMeta:
    """One parsed ``meta.json`` from ``~/.cursor/chats/<workspace-hash>/<session-uuid>/``.

    The file itself carries no session id -- the directory it sits in names
    the session -- and the times are epoch milliseconds (UTC).  ``cwd`` /
    ``created_at`` / ``updated_at`` / ``has_conversation`` are ``None`` when
    the file is missing, malformed, or lacks the field in the right type:
    this is an undocumented format, so an unreadable meta must never take a
    build down.
    """

    session_id: str
    cwd: str | None
    created_at: datetime | None
    updated_at: datetime | None
    has_conversation: bool | None


def _read_records(path: Path) -> tuple[list[tuple[int, dict]], int]:
    """(seq, record) for every parseable line, plus the error count.

    Same discipline as ``opencode._read_records``: a line that fails to parse
    as JSON, or that parses to something other than an object, is skipped and
    counted -- including a truncated final line of a transcript still being
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


def parse_file(path: str | Path) -> ParsedCursorFile:
    """Parse one Cursor agent-transcript file into tool calls and text chunks.

    A file with no parseable lines yields empty lists -- a normal outcome, not
    an error, the same as an empty opencode events.ndjson file.

    A user line is skipped -- not activity, per the brief -- and so is any
    line with no ``message`` (a bare status line, or a ``turn_ended`` line):
    those are unmodelled shapes, not parse errors.  Only assistant lines are
    modelled: each ``text`` block becomes a :class:`CursorTextChunk` and each
    ``tool_use`` block becomes a :class:`CursorToolCall`, whichever tool it
    names -- :mod:`ashiato.recall` decides which one is a kaiba recall.

    A tool call's ``call_id`` is ``f"{seq}:{block_index}"``: Cursor's own
    ``tool_use`` blocks carry no id of their own, unlike opencode's
    ``callID``, so identity has to come from position instead.
    """
    path = Path(path)
    file_path = str(path.resolve())
    session_id = path.stem
    records, n_parse_errors = _read_records(path)

    tool_calls: list[CursorToolCall] = []
    text_chunks: list[CursorTextChunk] = []

    for seq, record in records:
        if record.get("role") != "assistant":
            # A user line, a bare status/turn_ended line, or any other
            # unrecognised shape: none of these are activity or a call.
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block_index, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == _BLOCK_TYPE_TEXT:
                text = block.get("text")
                if isinstance(text, str) and text:
                    text_chunks.append(
                        CursorTextChunk(
                            session_id=session_id,
                            file_path=file_path,
                            seq=seq,
                            block_index=block_index,
                            text=text,
                        )
                    )
            elif block_type == _BLOCK_TYPE_TOOL_USE:
                name = block.get("name")
                block_input = block.get("input")
                tool_calls.append(
                    CursorToolCall(
                        call_id=f"{seq}:{block_index}",
                        session_id=session_id,
                        file_path=file_path,
                        seq=seq,
                        block_index=block_index,
                        name=name if isinstance(name, str) else None,
                        input=block_input if isinstance(block_input, dict) else None,
                    )
                )
            # Any other block type is not modelled; skipped, not an error.

    return ParsedCursorFile(
        file_path=file_path,
        session_id=session_id,
        tool_calls=tool_calls,
        text_chunks=text_chunks,
        n_parse_errors=n_parse_errors,
    )


def _meta_field_ms(value: object) -> datetime | None:
    """Epoch milliseconds (UTC) as a timezone-aware datetime, else ``None``.

    Only JSON numbers count as a time; a string, a bool, or any other type is
    a wrong type and reads as ``None``.  An out-of-range value is ``None``
    too -- a corrupt meta must never raise.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(value / 1000, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def parse_chat_meta(path: str | Path) -> ChatMeta | None:
    """Parse one Cursor chat ``meta.json`` into a :class:`ChatMeta`.

    ``session_id`` is the *parent directory name* -- the file has no id of
    its own, and the directory *is* the session.  When even that is unusable
    (a ``meta.json`` sitting in a filesystem root), ``None`` is returned.

    A pure function of the file bytes and its directory name: no clock, no
    network.  A missing file, malformed JSON, a missing key or a wrong type
    yields ``None`` for that field, never an exception -- this is an
    undocumented format, and a meta file a future Cursor version writes
    differently must not take a build down.
    """
    path = Path(path)
    session_id = path.parent.name
    if not session_id:
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        data = None
    cwd: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    has_conversation: bool | None = None
    if isinstance(data, dict):
        cwd_value = data.get("cwd")
        cwd = cwd_value if isinstance(cwd_value, str) else None
        created_at = _meta_field_ms(data.get("createdAtMs"))
        updated_at = _meta_field_ms(data.get("updatedAtMs"))
        conversation = data.get("hasConversation")
        has_conversation = conversation if isinstance(conversation, bool) else None
    return ChatMeta(
        session_id=session_id,
        cwd=cwd,
        created_at=created_at,
        updated_at=updated_at,
        has_conversation=has_conversation,
    )
