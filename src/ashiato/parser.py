"""Deterministic parsing of Claude Code JSONL session transcripts.

Every function here is a pure function of the bytes on disk: no network access,
no model calls, no clock reads.  The same file always produces the same rows --
that is what makes the resulting database trustworthy.

Transcripts are one JSON object per line.  The shapes vary a lot between record
types and between Claude Code versions, so the rule throughout is: a missing or
unexpected field becomes NULL, never an exception.  The verbatim line is kept in
``Event.raw`` so nothing this module fails to model is actually lost.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path

#: Prefixes that identify a tool result as a permission denial rather than a
#: real failure: a result is a denial only when its text, after stripping
#: leading whitespace, *starts* with one of these -- a successful result that
#: merely quotes one of the strings is not a denial.  These are Claude Code
#: strings and they drift between versions, so they live here as one overridable
#: constant instead of being sprinkled through the code.  Callers pass their own
#: tuple to ``parse_file``; each entry is matched as a prefix the same way.
DENIAL_PATTERNS: tuple[str, ...] = (
    "The user doesn't want to proceed with this tool use",
    "Permission for this action was denied by the Claude Code auto mode classifier",
)

#: Tool results can be megabytes; keep a readable prefix by default.
DEFAULT_RESULT_TEXT_LIMIT = 4000

#: The input field that says most about what a call asked for, per tool.  A
#: tool that is not named here -- every MCP tool, and anything Claude Code
#: grows after this table was written -- falls back to the whole input, so an
#: unfamiliar tool still gets a usable summary instead of NULL.
INPUT_SUMMARY_FIELDS: dict[str, str] = {
    "Bash": "command",
    "PowerShell": "command",
    "Read": "file_path",
    "Write": "file_path",
    "Edit": "file_path",
    "NotebookEdit": "file_path",
    "Glob": "pattern",
    "Grep": "pattern",
    "Skill": "skill",
    "Agent": "description",
    "Task": "description",
    "WebFetch": "url",
    "WebSearch": "query",
}

#: A summary is meant to fit on one line of output; a command can be a page.
INPUT_SUMMARY_LIMIT = 200

_TOOL_KIND_MCP = "mcp"
_TOOL_KIND_BUILTIN = "builtin"
_MCP_PREFIX = "mcp__"

# datetime.fromisoformat accepts at most 6 fractional digits.
_LONG_FRACTION = re.compile(r"\.(\d{7,})")

# A summary is one line, so every run of whitespace becomes a single space.
_WHITESPACE_RUN = re.compile(r"\s+")


@dataclass(slots=True)
class Event:
    """One JSONL line."""

    event_id: str
    session_id: str | None
    file_path: str
    seq: int
    ts: datetime | None
    type: str | None
    role: str | None
    parent_uuid: str | None
    depth: int
    is_sidechain: bool
    is_meta: bool
    permission_mode: str | None
    effort: str | None
    request_id: str | None
    message_id: str | None
    model: str | None
    cwd: str | None
    git_branch: str | None
    text: str
    raw: str


@dataclass(slots=True)
class ToolCall:
    """One tool invocation joined to its outcome."""

    tool_use_id: str
    session_id: str | None
    file_path: str
    seq: int
    ts: datetime | None
    call_event_id: str
    result_event_id: str | None
    tool_name: str | None
    tool_kind: str
    mcp_server: str | None
    input: str | None
    input_summary: str | None
    outcome: str
    is_error: bool
    result_text: str | None
    result_truncated: bool
    duration_ms: int | None
    permission_mode: str | None
    cwd: str | None
    is_sidechain: bool
    parent_tool_use_id: str | None


@dataclass(slots=True)
class Session:
    """One transcript file."""

    session_id: str | None
    file_path: str
    project_dir: str | None
    cwd: str | None
    git_branch: str | None
    cc_version: str | None
    entrypoint: str | None
    started_at: datetime | None
    ended_at: datetime | None
    n_events: int
    n_tool_calls: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int


@dataclass(slots=True)
class ParsedFile:
    """Everything one transcript file contributes to the database."""

    file_path: str
    session: Session | None
    events: list[Event]
    tool_calls: list[ToolCall]
    n_parse_errors: int


EVENT_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(Event))
TOOL_CALL_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(ToolCall))
SESSION_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(Session))


# --------------------------------------------------------------------------
# small coercions -- every one of them tolerates the wrong type
# --------------------------------------------------------------------------


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def parse_timestamp(value: object) -> datetime | None:
    """ISO-8601 string to a naive UTC datetime; None when unparseable."""
    text = _as_str(value)
    if not text:
        return None
    text = text.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    text = _LONG_FRACTION.sub(lambda m: "." + m.group(1)[:6], text)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def session_id_of(record: dict) -> str | None:
    """Different record types spell it differently; accept both."""
    return _as_str(record.get("sessionId")) or _as_str(record.get("session_id"))


# --------------------------------------------------------------------------
# message content
# --------------------------------------------------------------------------


def message_text(message: dict) -> str:
    """Concatenate every text block of a message; '' when there are none."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def _content_blocks(message: dict) -> list[dict]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def flatten_result_text(content: object) -> str:
    """Flatten tool_result content -- string, block list, or object -- to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                else:
                    # An image or another non-text block: note it without
                    # inlining what may be megabytes of base64.
                    parts.append(f"[{_as_str(item.get('type')) or 'block'}]")
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        return json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
    return str(content)


def split_tool_name(tool_name: str | None) -> tuple[str, str | None]:
    """(tool_kind, mcp_server) for a tool name."""
    if tool_name and tool_name.startswith(_MCP_PREFIX):
        segments = tool_name.split("__")
        server = segments[1] if len(segments) > 2 and segments[1] else None
        return _TOOL_KIND_MCP, server
    return _TOOL_KIND_BUILTIN, None


def _compact_json(value: object) -> str:
    """The smallest faithful rendering of a value; keys sorted so it is stable."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _one_line(text: str) -> str:
    return _WHITESPACE_RUN.sub(" ", text).strip()[:INPUT_SUMMARY_LIMIT]


def summarize_input(tool_name: str | None, tool_input: object) -> str | None:
    """One short line describing what a tool call asked for.

    Which field matters depends on the tool, so the well-known ones are read by
    name from :data:`INPUT_SUMMARY_FIELDS` and everything else falls back to the
    compact JSON of the whole input.  A named field that is absent, is not a
    string, or is blank falls back the same way: the thing that made the call is
    not always the thing this table expects, and showing the raw input beats
    showing nothing.

    Whitespace runs collapse to single spaces so a multi-line command stays one
    line, and the result is cut to :data:`INPUT_SUMMARY_LIMIT` characters --
    without a marker, the way ``result_text`` is cut.  ``None`` is returned only
    when there was no input at all.
    """
    if tool_input is None:
        return None
    if isinstance(tool_input, dict):
        field = INPUT_SUMMARY_FIELDS.get(tool_name or "")
        value = tool_input.get(field) if field is not None else None
        if isinstance(value, str) and value.strip():
            return _one_line(value)
    return _one_line(_compact_json(tool_input))


def classify_outcome(
    *,
    has_result: bool,
    result_text: str,
    is_error: bool,
    denial_patterns: Sequence[str] = DENIAL_PATTERNS,
) -> str:
    """Rules in order: pending, denied, error, ok.

    ``denial_patterns`` are matched as *prefixes*: after stripping leading
    whitespace, the result text must start with one of them to be a denial.  A
    successful result that merely contains the strings (a read or diff that
    quotes them) is not a denial.
    """
    if not has_result:
        return "pending"
    text = result_text.lstrip()
    if any(text.startswith(pattern) for pattern in denial_patterns):
        return "denied"
    if is_error:
        return "error"
    return "ok"


# --------------------------------------------------------------------------
# depth
# --------------------------------------------------------------------------


def compute_depths(parent_of: dict[str, str | None]) -> dict[str, int]:
    """Memoized ancestry depth for every node.

    Each node is resolved once and reused by its descendants: O(n) rather than
    O(n * chain length).  Real corpora reach chains ~2400 deep, so this is both
    the fast form and the only form that cannot blow the recursion limit -- it
    is iterative on purpose.

    A node whose parent is unknown (root, dangling reference into another file,
    or a cycle) has depth 0.
    """
    depth: dict[str, int] = {}
    for start in parent_of:
        if start in depth:
            continue
        chain: list[str] = []
        on_chain: set[str] = set()
        node: str | None = start
        while node is not None and node in parent_of and node not in depth and node not in on_chain:
            on_chain.add(node)
            chain.append(node)
            node = parent_of[node]
        base = depth.get(node, -1) if node is not None else -1
        for name in reversed(chain):
            base += 1
            depth[name] = base
    return depth


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def _read_records(path: Path) -> tuple[list[tuple[int, str, dict]], int]:
    """(seq, raw line, record) for every parseable line, plus the error count.

    A malformed line -- including the truncated final line of a session that is
    still being written -- is skipped and counted, never fatal.
    """
    records: list[tuple[int, str, dict]] = []
    n_errors = 0
    # utf-8-sig, not utf-8: a leading BOM is not valid JSON, so plain utf-8
    # would silently drop the first record of a BOM-prefixed transcript and
    # count it as a parse error.  utf-8-sig strips a BOM when present and is
    # identical to utf-8 otherwise.
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
            records.append((seq, raw, record))
    return records, n_errors


def _build_event(
    *, file_path: str, seq: int, raw: str, record: dict, fallback_session_id: str | None
) -> Event:
    message = _as_dict(record.get("message"))
    uuid = _as_str(record.get("uuid"))
    return Event(
        # Some record types (file-history-snapshot, mode, ai-title, ...) carry
        # no uuid at all; synthesize a stable one so every line is addressable.
        event_id=uuid or f"{file_path}:{seq}",
        session_id=session_id_of(record) or fallback_session_id,
        file_path=file_path,
        seq=seq,
        ts=parse_timestamp(record.get("timestamp")),
        type=_as_str(record.get("type")),
        role=_as_str(message.get("role")),
        parent_uuid=_as_str(record.get("parentUuid")),
        depth=0,  # filled in by compute_depths once every node is known
        is_sidechain=_as_bool(record.get("isSidechain")),
        is_meta=_as_bool(record.get("isMeta")),
        permission_mode=_as_str(record.get("permissionMode")),
        effort=_as_str(record.get("effort")),
        request_id=_as_str(record.get("requestId")),
        message_id=_as_str(record.get("messageId")) or _as_str(message.get("id")),
        model=_as_str(message.get("model")),
        cwd=_as_str(record.get("cwd")),
        git_branch=_as_str(record.get("gitBranch")),
        text=message_text(message),
        raw=raw,
    )


def _collect_tool_results(
    records: Iterable[tuple[int, str, dict]], events: Sequence[Event]
) -> dict[str, tuple[Event, dict]]:
    """tool_use_id -> (event carrying the result, the tool_result block).

    Results live on a later line than the call, usually a user line.  The first
    result for an id wins, so a replayed transcript cannot rewrite history.
    """
    results: dict[str, tuple[Event, dict]] = {}
    for (_, _, record), event in zip(records, events, strict=True):
        for block in _content_blocks(_as_dict(record.get("message"))):
            if block.get("type") != "tool_result":
                continue
            tool_use_id = _as_str(block.get("tool_use_id"))
            if tool_use_id and tool_use_id not in results:
                results[tool_use_id] = (event, block)
    return results


def _build_tool_calls(
    records: Sequence[tuple[int, str, dict]],
    events: Sequence[Event],
    *,
    denial_patterns: Sequence[str],
    result_text_limit: int,
) -> list[ToolCall]:
    results = _collect_tool_results(records, events)
    calls: list[ToolCall] = []
    seen: set[str] = set()

    for (_, _, record), event in zip(records, events, strict=True):
        blocks = _content_blocks(_as_dict(record.get("message")))
        for index, block in enumerate(blocks):
            if block.get("type") != "tool_use":
                continue
            tool_use_id = _as_str(block.get("id")) or f"{event.event_id}:{index}"
            if tool_use_id in seen:
                continue
            seen.add(tool_use_id)

            match = results.get(tool_use_id)
            result_event, result_block = match if match else (None, {})
            full_text = flatten_result_text(result_block.get("content")) if match else ""
            is_error = _as_bool(result_block.get("is_error")) if match else False

            tool_name = _as_str(block.get("name"))
            tool_kind, mcp_server = split_tool_name(tool_name)
            tool_input = block.get("input")

            duration_ms = None
            if result_event is not None and result_event.ts is not None and event.ts is not None:
                duration_ms = int((result_event.ts - event.ts).total_seconds() * 1000)

            calls.append(
                ToolCall(
                    tool_use_id=tool_use_id,
                    session_id=event.session_id,
                    file_path=event.file_path,
                    seq=event.seq,
                    ts=event.ts,
                    call_event_id=event.event_id,
                    result_event_id=result_event.event_id if result_event else None,
                    tool_name=tool_name,
                    tool_kind=tool_kind,
                    mcp_server=mcp_server,
                    input=(
                        None
                        if tool_input is None
                        else json.dumps(tool_input, ensure_ascii=False, default=str)
                    ),
                    input_summary=summarize_input(tool_name, tool_input),
                    outcome=classify_outcome(
                        has_result=match is not None,
                        result_text=full_text,
                        is_error=is_error,
                        denial_patterns=denial_patterns,
                    ),
                    is_error=is_error,
                    result_text=(full_text[:result_text_limit] if match else None),
                    result_truncated=len(full_text) > result_text_limit,
                    duration_ms=duration_ms,
                    permission_mode=event.permission_mode,
                    cwd=event.cwd,
                    is_sidechain=event.is_sidechain,
                    parent_tool_use_id=_as_str(record.get("sourceToolAssistantUUID")),
                )
            )
    return calls


def _sum_usage(records: Sequence[tuple[int, str, dict]]) -> dict[str, int]:
    """Token totals, deduplicated by request_id.

    The same ``usage`` object is repeated on several lines belonging to one
    request; summing naively inflates totals by 2-3.5x.  Lines with no
    request_id are each counted once.
    """
    totals: dict[str, int] = dict.fromkeys(
        ("input_tokens", "output_tokens", "cache_read_tokens", "cache_creation_tokens"), 0
    )
    seen_requests: set[str] = set()
    for _, _, record in records:
        usage = _as_dict(_as_dict(record.get("message")).get("usage"))
        if not usage:
            continue
        request_id = _as_str(record.get("requestId"))
        if request_id is not None:
            if request_id in seen_requests:
                continue
            seen_requests.add(request_id)
        totals["input_tokens"] += _as_int(usage.get("input_tokens"))
        totals["output_tokens"] += _as_int(usage.get("output_tokens"))
        totals["cache_read_tokens"] += _as_int(usage.get("cache_read_input_tokens"))
        totals["cache_creation_tokens"] += _as_int(usage.get("cache_creation_input_tokens"))
    return totals


def _build_session(
    *,
    path: Path,
    file_path: str,
    records: Sequence[tuple[int, str, dict]],
    events: Sequence[Event],
    n_tool_calls: int,
    session_id: str | None,
) -> Session:
    last: dict[str, str | None] = dict.fromkeys(("cwd", "gitBranch", "version", "entrypoint"))
    for _, _, record in records:
        for key in last:
            value = _as_str(record.get(key))
            if value is not None:
                last[key] = value

    stamps = [event.ts for event in events if event.ts is not None]
    totals = _sum_usage(records)
    return Session(
        session_id=session_id,
        file_path=file_path,
        project_dir=path.parent.name or None,
        cwd=last["cwd"],
        git_branch=last["gitBranch"],
        cc_version=last["version"],
        entrypoint=last["entrypoint"],
        started_at=min(stamps) if stamps else None,
        ended_at=max(stamps) if stamps else None,
        n_events=len(events),
        n_tool_calls=n_tool_calls,
        **totals,
    )


def parse_file(
    path: str | Path,
    *,
    denial_patterns: Sequence[str] = DENIAL_PATTERNS,
    result_text_limit: int = DEFAULT_RESULT_TEXT_LIMIT,
) -> ParsedFile:
    """Parse one transcript file into session / event / tool-call rows.

    ``denial_patterns`` replaces the built-in strings.  Each entry is matched as
    a *prefix* of the result text (leading whitespace ignored): a result that
    starts with one of them is ``denied``, a result that merely contains one
    somewhere is not.

    A file with no parseable lines yields ``session=None`` and empty lists; that
    is a normal outcome, not an error.
    """
    path = Path(path)
    file_path = str(path.resolve())
    records, n_parse_errors = _read_records(path)
    if not records:
        return ParsedFile(
            file_path=file_path,
            session=None,
            events=[],
            tool_calls=[],
            n_parse_errors=n_parse_errors,
        )

    # The file-level id backfills events whose own line omits it, so joins work.
    file_session_id = next(
        (sid for _, _, record in records if (sid := session_id_of(record))), path.stem
    )

    events = [
        _build_event(
            file_path=file_path,
            seq=seq,
            raw=raw,
            record=record,
            fallback_session_id=file_session_id,
        )
        for seq, raw, record in records
    ]

    parent_of: dict[str, str | None] = {}
    for event in events:
        parent_of.setdefault(event.event_id, event.parent_uuid)
    depths = compute_depths(parent_of)
    for event in events:
        event.depth = depths.get(event.event_id, 0)

    tool_calls = _build_tool_calls(
        records,
        events,
        denial_patterns=denial_patterns,
        result_text_limit=result_text_limit,
    )
    session = _build_session(
        path=path,
        file_path=file_path,
        records=records,
        events=events,
        n_tool_calls=len(tool_calls),
        session_id=file_session_id,
    )
    return ParsedFile(
        file_path=file_path,
        session=session,
        events=events,
        tool_calls=tool_calls,
        n_parse_errors=n_parse_errors,
    )
