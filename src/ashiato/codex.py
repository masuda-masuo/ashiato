"""Deterministic parsing of Codex session transcripts (~/.codex/sessions).

Mirrors the contract in :mod:`ashiato.cursor` and :mod:`ashiato.opencode`:
every function here is a pure function of the bytes on disk, no network access,
no model calls, no clock reads.
"""

from __future__ import annotations

import json
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


@dataclass(slots=True)
class CodexTextChunk:
    """One assistant or text message chunk in a Codex session."""

    session_id: str | None
    file_path: str
    seq: int
    ts: datetime | None
    text: str


@dataclass(slots=True)
class ParsedCodexFile:
    """Everything one Codex session JSONL file contributes to ashiato."""

    file_path: str
    session_id: str | None
    tool_calls: list[CodexToolCall]
    text_chunks: list[CodexTextChunk]
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


def _append_codex_message_text(
    message: dict,
    text_chunks: list[CodexTextChunk],
    *,
    session_id: str | None,
    file_path: str,
    seq: int,
    record_ts: datetime | None,
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
                            )
                        )
                    elif item_type in ("McpToolCall", "call_mcp_tool"):
                        server = item.get("server") or item.get("server_name")
                        tool_n = item.get("tool") or item.get("tool_name")
                        args = item.get("arguments") or item.get("args") or item.get("input")
                        res = item.get("result") or item.get("output")
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
        n_parse_errors=n_parse_errors,
        started_at=started_at,
        ended_at=ended_at,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
    )
