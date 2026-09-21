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
:func:`parse_chat_store` reads the ``store.db`` half: each ``tool-call`` part
of the conversation, in conversation order, paired with the raw ``result`` of
its matching ``tool-result`` part -- the tool results the transcript export
never records.  The user / system / assistant-text / reasoning messages of the
store are deliberately not modelled (issue #87 stage 3); this module reads
only the tool-call and tool-result parts.

Only what the recall-followup view needs is modelled: every assistant
``text`` block, and every ``tool_use`` block (whichever tool -- the recall
filter is applied downstream, by :mod:`ashiato.recall`).  A user line, a bare
status line (``{"status": ..., "type": ...}``), and a ``turn_ended`` line
carry no such content and are skipped like any other unmodelled shape --
that is not a parse error, only an invalid JSON line or a non-object JSON
value is.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import json
import sqlite3
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


# ---------------------------------------------------------------------------
# store.db: the conversation's tool results
# ---------------------------------------------------------------------------

#: Prefixes that mark a Cursor store tool result as a failure (classifier
#: rule 3).  Measured on the real corpus: 468 of 5,742 string results (8%)
#: start with one of these.  Rule 3 is deliberately the weakest signal in the
#: classifier -- a prefix match on model-facing prose, not a status field --
#: so these are only consulted after the structured ``error`` / ``status``
#: signals of a dict result, and a plain string result that merely *contains*
#: one of these prefixes without starting with it is not an error.
CURSOR_RESULT_ERROR_PREFIXES: tuple[str, ...] = (
    "Error executing tool",
    "Error: Tool execution error",
)

#: Values of a tool-result dict's ``status`` key that mean the call failed
#: (classifier rule 2).  The real corpus carries a ``status`` key on 421 of
#: 2,445 dict results; this is the failure half of that vocabulary.  Anything
#: else -- ``success``, ``completed``, an unknown future spelling -- reads as
#: not-failed, exactly the way a brand-new denial pattern must not be guessed.
CURSOR_RESULT_FAILURE_STATUSES: tuple[str, ...] = ("error", "failed", "failure")


@dataclass(slots=True)
class CursorStoreToolCall:
    """One ``tool-call`` part read from a Cursor chat ``store.db``.

    ``result`` is the raw ``result`` value of the matching ``tool-result``
    part (a string, a dict, or ``None``), found by ``toolCallId``.
    ``has_result`` says whether such a part existed at all: a ``tool-result``
    whose value really is JSON null has ``has_result`` True and ``result``
    ``None`` (the call completed and returned nothing), while a call no
    result part ever matched has ``has_result`` False -- its fate is unknown,
    and the build must not classify it as a success.
    """

    tool_call_id: str | None
    tool_name: str | None
    args: dict | None
    result: object
    has_result: bool = False


def _varint(data: bytes, i: int) -> tuple[int, int] | None:
    """(value, next index) of the varint starting at *i*, or ``None`` when truncated."""
    value = 0
    shift = 0
    n = len(data)
    while i < n:
        b = data[i]
        i += 1
        value |= (b & 0x7F) << shift
        if not b & 0x80:
            return value, i
        shift += 7
        if shift > 63:
            return None
    return None


def _decode_proto(data: bytes) -> tuple[list[tuple[int, bytes]], bool]:
    """(fields, complete) from a minimal varint / length-delimited protobuf reader.

    ``complete`` is False when the payload is truncated or uses a wire type
    this reader does not know: a truncated blob must read as unreadable, not
    as a partial list of children -- a session whose root blob is cut off
    yields nothing, exactly like a missing file.
    """
    fields: list[tuple[int, bytes]] = []
    i = 0
    n = len(data)
    while i < n:
        tag = _varint(data, i)
        if tag is None:
            return fields, False
        tag, i = tag
        wire = tag & 7
        if wire == 2:
            length = _varint(data, i)
            if length is None:
                return fields, False
            ln, i = length
            if i + ln > n:
                return fields, False
            fields.append((tag >> 3, data[i : i + ln]))
            i += ln
        elif wire == 0:
            value = _varint(data, i)
            if value is None:
                return fields, False
            _, i = value
        else:
            return fields, False
    return fields, True


def _blob_id_candidates(value: object) -> list[object]:
    """Query spellings for a ``latestRootBlobId``-style value.

    The meta row stores its JSON hex-encoded, and the blob ids inside it are
    the same 32-byte values the protobuf carries, but the ``blobs.id`` column
    of the real store is TEXT holding the id's *hex* spelling while the
    protobuf children are the *raw* 32 bytes -- so an id may arrive as either
    spelling and must be tried in both (plus base64, in case a future Cursor
    version switches) so the join does not depend on the column's declared
    type.  The string branch tries ``bytes.fromhex``; the bytes branch is
    symmetric and tries ``.hex()``.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        candidates: list[object] = [raw, raw.hex()]
        with contextlib.suppress(binascii.Error):
            candidates.append(base64.b64encode(raw).decode("ascii"))
        return candidates
    if not isinstance(value, str) or not value:
        return []
    candidates: list[object] = [value]
    with contextlib.suppress(ValueError):
        candidates.append(bytes.fromhex(value))
    with contextlib.suppress(ValueError, binascii.Error):
        candidates.append(base64.b64decode(value, validate=True))
    return candidates


def _lookup_blob(connection: sqlite3.Connection, value: object) -> bytes | None:
    """The ``blobs.data`` bytes for one id spelling, or ``None`` when not present."""
    for candidate in _blob_id_candidates(value):
        try:
            row = connection.execute(
                "SELECT data FROM blobs WHERE id = ?", [candidate]
            ).fetchone()
        except sqlite3.Error:
            return None
        if row is not None:
            data = row[0]
            if isinstance(data, (bytes, bytearray, memoryview)):
                return bytes(data)
            if isinstance(data, str):
                return data.encode("utf-8")
    return None


def _store_call_args(part: dict) -> dict | None:
    args = part.get("args")
    if isinstance(args, dict):
        return args
    args = part.get("input")
    if isinstance(args, dict):
        return args
    return None


def _store_call_name(part: dict) -> str | None:
    name = part.get("toolName")
    if isinstance(name, str):
        return name
    name = part.get("name")
    if isinstance(name, str):
        return name
    return None


def parse_chat_store(path: str | Path) -> list[CursorStoreToolCall]:
    """Read the tool calls of one Cursor chat ``store.db``, in conversation order.

    The store is an undocumented SQLite database.  Conversation order comes
    from the blob graph, not from table order: ``meta`` carries a
    hex-encoded JSON ``latestRootBlobId``; that blob is a protobuf whose
    repeated field 1 holds the child blob ids in order; the children that
    are JSON are the messages, and each message's ``content`` holds the
    ``tool-call`` / ``tool-result`` parts in conversation order.  Each
    returned call carries the raw ``result`` of its matching ``tool-result``
    part (matched by ``toolCallId``), with ``has_result`` saying whether such
    a part was found at all -- a call whose id never matches a result part
    is a call whose fate the store does not record.

    A missing file, an unreadable or non-SQLite file, a ``meta`` row that
    does not decode, a root id that is not in ``blobs``, a truncated
    protobuf, and a non-JSON child all yield ``[]`` for the session -- never
    an exception.  This is an undocumented format; a store a future Cursor
    version writes differently must not take a build down.
    """
    path = Path(path).resolve()
    try:
        # mode=ro: a missing file fails here instead of being created empty.
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    except (OSError, sqlite3.Error):
        return []
    try:
        try:
            rows = connection.execute("SELECT value FROM meta").fetchall()
        except sqlite3.Error:
            return []
        root_id: object | None = None
        for (value,) in rows:
            if not isinstance(value, str):
                continue
            try:
                payload = json.loads(bytes.fromhex(value).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(payload, dict):
                candidate = payload.get("latestRootBlobId")
                if isinstance(candidate, str) and candidate:
                    root_id = candidate
                    break
        if root_id is None:
            return []
        root_blob = _lookup_blob(connection, root_id)
        if root_blob is None:
            return []
        fields, complete = _decode_proto(root_blob)
        if not complete:
            return []
        child_ids = [value for field, value in fields if field == 1]

        calls: list[CursorStoreToolCall] = []
        results: dict[str, object] = {}
        for child_id in child_ids:
            child = _lookup_blob(connection, child_id)
            if child is None:
                continue
            try:
                message = json.loads(child.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                continue
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "tool-call":
                    call_id = part.get("toolCallId")
                    calls.append(
                        CursorStoreToolCall(
                            tool_call_id=call_id if isinstance(call_id, str) else None,
                            tool_name=_store_call_name(part),
                            args=_store_call_args(part),
                            result=None,  # attached below, once all results are seen
                        )
                    )
                elif part_type == "tool-result":
                    call_id = part.get("toolCallId")
                    if isinstance(call_id, str):
                        results[call_id] = part.get("result")
        for call in calls:
            if call.tool_call_id is not None and call.tool_call_id in results:
                call.has_result = True
                call.result = results[call.tool_call_id]
        return calls
    finally:
        connection.close()


def classify_store_result(result: object) -> tuple[bool, str]:
    """(is_error, outcome) for one Cursor store tool result.

    Rules in order, exactly as documented in the README:

    1. a dict with a non-empty ``error`` key is an error;
    2. a dict whose ``status`` is in :data:`CURSOR_RESULT_FAILURE_STATUSES`
       is an error;
    3. a string starting with one of :data:`CURSOR_RESULT_ERROR_PREFIXES` is
       an error;
    4. anything else is ``ok``.

    ``pending`` and ``denied`` are never produced: a result that exists at
    all means the call completed, and Cursor records no denial signal.  Rule 3
    is prefix matching on model-facing prose -- a weaker signal than the
    other sources' status fields, which is why it is the last resort.
    """
    if isinstance(result, dict):
        error = result.get("error")
        if error is not None and error != "":
            return True, "error"
        status = result.get("status")
        if isinstance(status, str) and status in CURSOR_RESULT_FAILURE_STATUSES:
            return True, "error"
        return False, "ok"
    if isinstance(result, str):
        if result.startswith(CURSOR_RESULT_ERROR_PREFIXES):
            return True, "error"
        return False, "ok"
    return False, "ok"


def store_result_text(result: object) -> str:
    """A tool result rendered as text: a string as-is, a dict as compact JSON."""
    if isinstance(result, str):
        return result
    return json.dumps(
        result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
