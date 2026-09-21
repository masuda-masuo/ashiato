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
:func:`parse_chat_store` reads the ``store.db`` half: the whole conversation,
in conversation order (the ``blobs`` table's own order, which is insertion
order -- the ``latestRootBlobId`` root is only a checkpoint window over the
newest messages since Cursor's CLI stopped rewriting the full conversation at
each checkpoint in 2026-07-13).  Each message comes back with its ``role`` and
its ordered ``content`` parts -- the ``text`` / ``reasoning`` /
``redacted-reasoning`` parts that become ``events`` rows (issue #87 stage 3),
and the ``tool-call`` parts, paired with the raw ``result`` of the matching
``tool-result`` part -- the tool results the transcript export never records
(issue #87 stage 2).

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


@dataclass(slots=True)
class CursorStorePart:
    """One ``content`` part of a Cursor chat store message.

    ``type`` is the part's recorded ``type`` verbatim -- ``text``,
    ``reasoning``, ``redacted-reasoning``, ``tool-call``, ``tool-result``,
    or anything a future Cursor version writes.  The build maps the first
    three to ``events`` rows, leaves ``tool-call`` / ``tool-result`` to the
    ``tool_calls`` rows, and *counts* an unrecognised type (issue #87 stage 3
    criterion: an unknown part type is counted, never dropped silently).
    ``text`` is the part's ``text`` when it is a string, else ``None`` -- a
    ``redacted-reasoning`` part carries no text at all, and a part that
    records no text must not read as an empty string.
    """

    type: str | None
    text: str | None


@dataclass(slots=True)
class CursorStoreMessage:
    """One chat message of a Cursor ``store.db``, in conversation order.

    ``message_index`` is the message's position among the JSON messages of
    the ``blobs`` table (0-based, in table order) -- the identity a
    deterministic ``event_id`` is derived from.  The store's own ``id``
    field cannot serve: measured on the real corpus, 4,649 of 24,913 ids are
    duplicates of another id in the same session (issue #87 premise 4).
    ``role`` is the recorded role verbatim (``system`` / ``user`` /
    ``assistant`` / ``tool``); a message without a role reads ``None``.
    ``parts`` are the message's ``content`` parts in order.  A message whose
    ``content`` is a bare string -- the real system prompt and the initial
    user_info message are written exactly that way -- is a single ``text``
    part; a ``content`` that is neither a list nor a string contributes no
    parts at all.
    """

    message_index: int
    role: str | None
    parts: list[CursorStorePart]


class CursorStoreConversation(list[CursorStoreToolCall]):
    """A parsed ``store.db``: the tool calls (list behaviour) plus the messages.

    Behaves exactly like the ``list[CursorStoreToolCall]`` that
    :func:`parse_chat_store` used to return -- stage 2's pairing and
    result-filling code and its tests iterate, index, take ``len`` and
    compare against ``[]`` unchanged -- and carries the full conversation
    on ``messages`` for the events replacement (issue #87 stage 3).
    """

    def __init__(
        self,
        tool_calls: list[CursorStoreToolCall],
        messages: list[CursorStoreMessage],
    ) -> None:
        super().__init__(tool_calls)
        self.messages = messages


def _store_call_args(part: dict) -> dict | None:
    """The call's arguments as a dict, from ``args`` or ``input``; else ``None``."""
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


def parse_chat_store(path: str | Path) -> CursorStoreConversation:
    """Read one Cursor chat ``store.db``: the tool calls *and* the conversation.

    The store is an undocumented SQLite database.  Conversation order is
    *table order*: ``SELECT data FROM blobs`` returns rows in rowid order,
    which is insertion order, and every row whose bytes decode as a JSON chat
    message contributes its ``content`` parts in that order.  The ``meta``
    row and the ``latestRootBlobId`` blob are deliberately *not* used: since
    2026-07-13 Cursor's CLI saves only new transcript entries at each
    checkpoint instead of rewriting the full conversation, so the latest root
    is a window over the newest messages, not the conversation -- walking its
    field-1 children yields a suffix.  Reading the table instead of the root
    is what makes the returned calls match the transcript for sessions of any
    length; the pairing guard in :func:`ashiato.build._apply_cursor_chat_stores`
    (count equality plus elementwise name agreement) is what protects against
    the ordering being wrong anyway.

    The return is a :class:`CursorStoreConversation` -- a list of the
    conversation's ``tool-call`` parts (in conversation order, each carrying
    the raw ``result`` of its matching ``tool-result`` part, matched by
    ``toolCallId``, with ``has_result`` saying whether such a part was found
    at all) -- with the full conversation on ``messages``: every JSON chat
    message in table order, each with its ``role`` and its ordered
    ``content`` parts (``CursorStorePart`` carries the recorded ``type``
    verbatim and the part's text).  A message whose ``content`` is a bare
    string -- the real system prompt and the initial user_info message are
    written exactly that way -- reads as one ``text`` part.  Every part is
    kept, whatever its type: the build decides which become ``events`` rows
    and which are unclassifiable, and an unknown type must be counted, never
    dropped silently (issue #87 stage 3).

    A missing file, an unreadable or non-SQLite file, a missing or unreadable
    ``blobs`` table, and a blob that is not a JSON message (the store keeps
    binary protobuf blobs too) yield an empty conversation -- never an
    exception.  This is an undocumented format; a store a future Cursor
    version writes differently must not take a build down.
    """
    path = Path(path).resolve()
    try:
        # mode=ro: a missing file fails here instead of being created empty.
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    except (OSError, sqlite3.Error):
        return CursorStoreConversation([], [])
    try:
        try:
            rows = connection.execute("SELECT data FROM blobs").fetchall()
        except sqlite3.Error:
            return CursorStoreConversation([], [])

        calls: list[CursorStoreToolCall] = []
        results: dict[str, object] = {}
        messages: list[CursorStoreMessage] = []
        message_index = 0
        for (data,) in rows:
            if not isinstance(data, (bytes, bytearray, memoryview)):
                continue
            try:
                message = json.loads(bytes(data).decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                continue
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if isinstance(content, str):
                parts = [CursorStorePart(type="text", text=content)]
            elif isinstance(content, list):
                parts = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type")
                    text = part.get("text")
                    parts.append(
                        CursorStorePart(
                            type=part_type if isinstance(part_type, str) else None,
                            text=text if isinstance(text, str) else None,
                        )
                    )
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
            else:
                parts = []
            messages.append(
                CursorStoreMessage(
                    message_index=message_index,
                    role=role if isinstance(role, str) else None,
                    parts=parts,
                )
            )
            message_index += 1
        for call in calls:
            if call.tool_call_id is not None and call.tool_call_id in results:
                call.has_result = True
                call.result = results[call.tool_call_id]
        return CursorStoreConversation(calls, messages)
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
