"""Deterministic parsing of Codex session transcripts (~/.codex/sessions).

Mirrors the contract in :mod:`ashiato.cursor` and :mod:`ashiato.opencode`:
every function here is a pure function of the bytes on disk, no network access,
no model calls, no clock reads.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ashiato.parser import parse_timestamp


@dataclass(slots=True)
class CodexToolCall:
    """One completed tool execution in a Codex session."""

    call_id: str
    session_id: str | None
    file_path: str
    seq: int
    ts: datetime | None
    tool_name: str | None
    input: dict | None
    output: str | None
    #: Codex's own completion status ('completed' / 'failed') when the item
    #: carries one; the raw value when present but malformed.
    status: str | None = None
    #: Process exit code for CommandExecution items when the item carries one;
    #: the raw value when present but malformed.
    exit_code: int | None = None
    #: MCP failure message when the item carries one.
    error: str | None = None
    #: Command execution duration in milliseconds (Codex stores it in seconds).
    duration_ms: int | None = None
    #: Working directory for CommandExecution items when the item carries one.
    cwd: str | None = None


@dataclass(slots=True)
class CodexTextChunk:
    """One assistant or text message chunk in a Codex session."""

    session_id: str | None
    file_path: str
    seq: int
    ts: datetime | None
    text: str
    role: str | None = None


@dataclass(slots=True)
class CodexEvent:
    """One non-text event in a Codex session (e.g. a context compaction).

    Text messages become :class:`CodexTextChunk`; this carries the item types
    that must surface as ``events`` rows with a type of their own (a
    ``ContextCompaction`` item) rather than as assistant text.
    """

    event_id: str
    session_id: str | None
    file_path: str
    seq: int
    ts: datetime | None
    type: str
    text: str | None
    #: The verbatim item JSON, so the ``events.raw`` promise ("nothing is
    #: dropped") holds for these rows too.
    raw: str


@dataclass(slots=True)
class ParsedCodexFile:
    """Everything one Codex session JSONL file contributes to ashiato."""

    file_path: str
    session_id: str | None
    tool_calls: list[CodexToolCall]
    text_chunks: list[CodexTextChunk]
    events: list[CodexEvent]
    n_parse_errors: int
    #: Earliest timestamp across all records (min of record timestamps).
    started_at: datetime | None
    #: Latest timestamp across all records (max of record timestamps).
    ended_at: datetime | None
    #: Final ``thread_token_usage`` from the last ``token_usage_record``,
    #: or zero when the file contains no such record.
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int


def _read_records(path: Path) -> tuple[list[tuple[int, dict]], int]:
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


def _duration_to_ms(value: object) -> int | None:
    """Convert a Codex item's ``duration`` to integer milliseconds.

    Codex CLI stores duration in *seconds*: recent versions write it as a
    floating-point number, while some versions (measured 0.146.1) serialize a
    Rust-style ``{"secs": ..., "nanos": ...}`` object.  Both are seconds, so
    both convert confidently; anything else (strings, booleans, unknown
    shapes) is left as ``None`` rather than guessed at.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return None
        return int(value * 1000)
    if isinstance(value, dict):
        secs = value.get("secs")
        nanos = value.get("nanos", 0)
        if isinstance(secs, bool) or not isinstance(secs, (int, float)):
            return None
        if isinstance(nanos, bool) or not isinstance(nanos, (int, float)):
            nanos = 0
        if not math.isfinite(secs) or not math.isfinite(nanos):
            return None
        return int(secs * 1000 + nanos / 1_000_000)
    return None


def _append_codex_message_text(
    message: dict,
    text_chunks: list[CodexTextChunk],
    *,
    session_id: str | None,
    file_path: str,
    seq: int,
    record_ts: datetime | None,
    role: str | None = None,
) -> None:
    """Join output_text/input_text parts from a message-shaped dict into one chunk."""
    content = message.get("content")
    if not isinstance(content, list):
        return
    text_parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") not in ("output_text", "input_text"):
            continue
        part_text = part.get("text")
        if isinstance(part_text, str) and part_text.strip():
            text_parts.append(part_text)
    if not text_parts:
        return
    text_chunks.append(
        CodexTextChunk(
            session_id=session_id,
            file_path=file_path,
            seq=seq,
            ts=record_ts,
            text="".join(text_parts),
            role=role,
        )
    )


def parse_file(path: str | Path) -> ParsedCodexFile:
    """Parse one Codex JSONL file into tool calls and text chunks."""
    path = Path(path)
    file_path = str(path.resolve())
    session_id: str | None = None
    records, n_parse_errors = _read_records(path)

    tool_calls: list[CodexToolCall] = []
    text_chunks: list[CodexTextChunk] = []
    events: list[CodexEvent] = []

    timestamps: list[datetime] = []
    # Accumulate thread_token_usage; last token_usage_record wins.
    input_tokens = 0
    output_tokens = 0
    cache_read_tokens = 0

    for seq, record in records:
        rec_type = record.get("type")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            payload = {}

        record_ts = parse_timestamp(record.get("timestamp"))
        if record_ts is not None:
            timestamps.append(record_ts)

        if rec_type == "session_meta":
            if not session_id:
                sid = payload.get("id") or payload.get("session_id")
                if isinstance(sid, str):
                    session_id = sid

        elif rec_type == "event_msg":
            p_type = payload.get("type")
            sid = payload.get("thread_id") or payload.get("session_id")
            if sid and isinstance(sid, str) and not session_id:
                session_id = sid

            if p_type == "item_completed":
                item = payload.get("item")
                if isinstance(item, dict):
                    item_type = item.get("type")
                    item_id = item.get("id") or f"{seq}"
                    if item_type == "CommandExecution":
                        cmd = item.get("command")
                        stdout = item.get("stdout") or item.get("aggregated_output") or ""
                        tool_calls.append(
                            CodexToolCall(
                                call_id=str(item_id),
                                session_id=session_id,
                                file_path=file_path,
                                seq=seq,
                                ts=record_ts,
                                tool_name="Bash",
                                input={"command": cmd} if isinstance(cmd, (list, str)) else {},
                                output=str(stdout) if stdout else None,
                                status=item.get("status"),
                                exit_code=item.get("exit_code"),
                                duration_ms=_duration_to_ms(item.get("duration")),
                                cwd=item.get("cwd") if isinstance(item.get("cwd"), str) else None,
                            )
                        )
                    elif item_type in ("McpToolCall", "call_mcp_tool"):
                        server = item.get("server") or item.get("server_name")
                        tool_n = item.get("tool") or item.get("tool_name")
                        args = item.get("arguments") or item.get("args") or item.get("input")
                        error = item.get("error")
                        res = item.get("result") or item.get("output")
                        if not res and error:
                            # A failed call with no result: its error message is
                            # the only output it has.
                            res = error
                        tool_name = f"mcp__{server}__{tool_n}" if server and tool_n else (tool_n or "McpTool")
                        tool_calls.append(
                            CodexToolCall(
                                call_id=str(item_id),
                                session_id=session_id,
                                file_path=file_path,
                                seq=seq,
                                ts=record_ts,
                                tool_name=str(tool_name),
                                input=args if isinstance(args, dict) else {},
                                output=str(res) if res else None,
                                status=item.get("status"),
                                error=error if isinstance(error, str) else None,
                                duration_ms=_duration_to_ms(item.get("duration")),
                            )
                        )
                    elif item_type in ("AgentResponse", "assistant_message"):
                        txt = item.get("text") or item.get("content")
                        if isinstance(txt, str) and txt.strip():
                            text_chunks.append(
                                CodexTextChunk(
                                    session_id=session_id,
                                    file_path=file_path,
                                    seq=seq,
                                    ts=record_ts,
                                    text=txt,
                                )
                            )
                    elif item_type == "message":
                        # Rare nested shape (item_completed.item.type == message).
                        _append_codex_message_text(
                            item,
                            text_chunks,
                            session_id=session_id,
                            file_path=file_path,
                            seq=seq,
                            record_ts=record_ts,
                        )
                    elif item_type == "FileChange":
                        # The only record of which host files a Codex session
                        # edited: one row whose input carries the paths (as a
                        # ``files`` list and as the keys of the verbatim
                        # ``changes`` payload) and whose result is the tool's
                        # own stdout.
                        changes = item.get("changes")
                        if not isinstance(changes, dict):
                            changes = {}
                        files = [path for path in changes if isinstance(path, str)]
                        stderr = item.get("stderr")
                        tool_calls.append(
                            CodexToolCall(
                                call_id=str(item_id),
                                session_id=session_id,
                                file_path=file_path,
                                seq=seq,
                                ts=record_ts,
                                tool_name="FileChange",
                                input={"files": files, "changes": changes},
                                output=str(item.get("stdout")) if item.get("stdout") else None,
                                status=item.get("status"),
                                error=stderr if isinstance(stderr, str) and stderr else None,
                                duration_ms=_duration_to_ms(item.get("duration")),
                            )
                        )
                    elif item_type == "CollabAgentToolCall":
                        # A delegation record: the parent session calling its
                        # collaboration tool (``tool``, e.g. ``wait``), with
                        # the receiver thread/agent fields when the item
                        # carries them.
                        tool = item.get("tool")
                        collab_input: dict[str, object] = {}
                        for key in ("sender_thread_id", "receiver_thread_ids",
                                    "receiver_agents", "agents_states"):
                            value = item.get(key)
                            if value is not None:
                                collab_input[key] = value
                        tool_calls.append(
                            CodexToolCall(
                                call_id=str(item_id),
                                session_id=session_id,
                                file_path=file_path,
                                seq=seq,
                                ts=record_ts,
                                tool_name=(
                                    f"collab__{tool}" if isinstance(tool, str) and tool
                                    else "CollabAgentToolCall"
                                ),
                                input=collab_input,
                                output=None,
                                status=item.get("status"),
                                duration_ms=_duration_to_ms(item.get("duration")),
                            )
                        )
                    elif item_type == "SubAgentActivity":
                        # The subagent's side of the delegation: one row per
                        # activity (``kind``: started / interacted / completed)
                        # naming the agent's thread and path when carried.
                        subagent_input: dict[str, object] = {}
                        for key in ("kind", "agent_thread_id", "agent_path"):
                            value = item.get(key)
                            if value is not None:
                                subagent_input[key] = value
                        tool_calls.append(
                            CodexToolCall(
                                call_id=str(item_id),
                                session_id=session_id,
                                file_path=file_path,
                                seq=seq,
                                ts=record_ts,
                                tool_name="collab__subagent",
                                input=subagent_input,
                                output=None,
                                status=item.get("status"),
                                duration_ms=_duration_to_ms(item.get("duration")),
                            )
                        )
                    elif item_type == "ContextCompaction":
                        # A compaction marker becomes an events row with a type
                        # of its own (never assistant text), so a session's
                        # compaction points are queryable.  The item itself
                        # carries only an id; the verbatim item JSON is kept in
                        # ``raw``.
                        events.append(
                            CodexEvent(
                                event_id=str(item_id),
                                session_id=session_id,
                                file_path=file_path,
                                seq=seq,
                                ts=record_ts,
                                type="context_compaction",
                                text=None,
                                raw=json.dumps(item, ensure_ascii=False),
                            )
                        )

        elif rec_type == "response_item":
            # Live Codex text: top-level response_item with payload.type == "message"
            # (measured 2026-09-06: 385 such rows; content parts output_text/input_text).
            if payload.get("type") == "message":
                _append_codex_message_text(
                    payload,
                    text_chunks,
                    session_id=session_id,
                    file_path=file_path,
                    seq=seq,
                    record_ts=record_ts,
                    role=payload.get("role"),
                )

        elif rec_type == "token_usage_record":
            tu = payload.get("thread_token_usage")
            if isinstance(tu, dict):
                input_tokens = int(tu.get("input_tokens") or 0)
                output_tokens = int(tu.get("output_tokens") or 0)
                cache_read_tokens = int(tu.get("cached_input_tokens") or 0)

    if not session_id:
        session_id = path.stem

    started_at = min(timestamps) if timestamps else None
    ended_at = max(timestamps) if timestamps else None

    return ParsedCodexFile(
        file_path=file_path,
        session_id=session_id,
        tool_calls=tool_calls,
        text_chunks=text_chunks,
        events=events,
        n_parse_errors=n_parse_errors,
        started_at=started_at,
        ended_at=ended_at,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
    )
