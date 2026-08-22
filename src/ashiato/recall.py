"""Pairing a kaiba ``recall`` call with what the same session did next.

Same shape as ``denial_followups`` (see ``ashiato.schema``): a call, joined
to the evidence of what happened afterwards.  Here the call is a completed
``recall`` tool invocation instead of a denied one, and the pairing is
computed here, at build time, into a table -- not as a view over
``tool_calls`` the way ``denial_followups`` is.  A recall call's "what
happened next" evidence has to be assembled from a session's own activity
stream, which is shaped differently per source format (Claude Code's
tool_use/tool_result pairs vs. opencode's message parts), so there is no
single raw table to define a read-time view over the way there is for
denials.  ``ashiato.schema.RECALL_FOLLOWUPS_SQL`` is a thin view over the
table this module fills in.

This module only nominates evidence for a human to read; it never decides
whether a recall call was actually useful.  ``overlap_tokens`` /
``overlap_count`` are one deterministic, mechanical signal to start from, not
a verdict -- no LLM, no network, no clock reads, same bytes in means same
rows out.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, fields
from datetime import datetime

from ashiato.opencode import ParsedOpenCodeFile
from ashiato.parser import DEFAULT_RESULT_TEXT_LIMIT, ParsedFile

#: Tool names that identify a kaiba recall call, one per source format.
CLAUDE_RECALL_TOOL = "mcp__kaiba__recall"
OPENCODE_RECALL_TOOL = "kaiba_recall"

#: Values ``recall_calls.source`` takes.
SOURCE_CLAUDE_CODE = "claude_code"
SOURCE_OPENCODE = "opencode"

#: How much of a session's post-recall activity is scanned for "was this
#: used" evidence: whichever bound is hit first.  Generous on purpose --
#: real agent turns chain many tool calls -- while keeping the stored text
#: finite and the signal computation O(session) rather than O(corpus).
FOLLOWUP_ITEM_LIMIT = 30
FOLLOWUP_CHAR_LIMIT = 8000

#: A token has to be this specific to count as evidence: bare numbers and
#: short common words are not distinctive of anything.  Deliberately the
#: same shape suggested in the brief: file paths, identifiers, flags.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_#./-]{4,}")

#: Matched tokens are stored as a readable sample, not the full set;
#: ``overlap_count`` is the true, uncapped total.
TOKEN_SAMPLE_LIMIT = 20


@dataclass(slots=True)
class RecallCall:
    """One row of the ``recall_calls`` table."""

    recall_id: str
    session_id: str | None
    file_path: str
    source: str
    seq: int
    ts: datetime | None
    call_id: str
    query: str | None
    output: str | None
    output_truncated: bool
    followup_text: str | None
    followup_truncated: bool
    overlap_tokens: str | None
    overlap_count: int


RECALL_CALL_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(RecallCall))


@dataclass(slots=True)
class _Activity:
    """One item of a session's activity, ordered the way transcript lines are."""

    seq: int
    text: str


def _tokens(text: str | None) -> set[str]:
    return set(_TOKEN_RE.findall(text)) if text else set()


def _overlap(
    output: str | None, prefix: Sequence[_Activity], suffix: Sequence[_Activity]
) -> tuple[list[str], int]:
    """Tokens the recall's output introduced into the post-recall suffix.

    "Introduced" is what makes a token distinctive evidence of use: it has to
    be new to the session, not merely a word the session already knew before
    the call, so tokens the prefix already contains are excluded even when
    they also appear in the output and the suffix.
    """
    prefix_tokens: set[str] = set()
    for item in prefix:
        prefix_tokens |= _tokens(item.text)
    suffix_tokens: set[str] = set()
    for item in suffix:
        suffix_tokens |= _tokens(item.text)
    matched = (_tokens(output) & suffix_tokens) - prefix_tokens
    ordered = sorted(matched)
    return ordered[:TOKEN_SAMPLE_LIMIT], len(matched)


def _bounded_suffix(items: Sequence[_Activity]) -> tuple[str, bool]:
    """Join the first :data:`FOLLOWUP_ITEM_LIMIT` items, capped in characters too."""
    bounded = items[:FOLLOWUP_ITEM_LIMIT]
    truncated = len(items) > len(bounded)
    parts: list[str] = []
    total = 0
    for item in bounded:
        parts.append(item.text)
        total += len(item.text) + 1
        if total > FOLLOWUP_CHAR_LIMIT:
            truncated = True
            break
    text = "\n".join(parts)
    if len(text) > FOLLOWUP_CHAR_LIMIT:
        text = text[:FOLLOWUP_CHAR_LIMIT]
        truncated = True
    return text, truncated


def _split(items: Sequence[_Activity], seq: int) -> tuple[list[_Activity], list[_Activity]]:
    """(prefix, suffix) around *seq* -- items on *seq* itself belong to neither.

    Mirrors ``denial_followups``' "strictly later line" rule: an item issued
    on the very same transcript line as the recall call was decided before
    the recall's result could have been seen, so it is evidence of neither
    what came before nor what came after.
    """
    prefix = [item for item in items if item.seq < seq]
    suffix = [item for item in items if item.seq > seq]
    return prefix, suffix


def extract_from_claude(parsed: ParsedFile) -> list[RecallCall]:
    """Completed ``mcp__kaiba__recall`` calls, paired with same-session followup evidence.

    "Completed" means the call has *some* result, whatever its outcome --
    the same definition ``tool_calls.outcome != 'pending'`` uses elsewhere.
    Activity is every assistant text event and every other completed tool
    call, grouped by session and ordered by transcript line (``seq``);
    output is already truncated by :func:`ashiato.parser.parse_file`, so
    nothing further is done to it here.
    """
    activity: dict[str | None, list[_Activity]] = {}
    for event in parsed.events:
        if event.role == "assistant" and event.text:
            activity.setdefault(event.session_id, []).append(_Activity(event.seq, event.text))
    for call in parsed.tool_calls:
        if call.outcome == "pending":
            continue
        text = " ".join(
            part for part in (call.tool_name, call.input_summary, call.result_text) if part
        )
        if text:
            activity.setdefault(call.session_id, []).append(_Activity(call.seq, text))
    for items in activity.values():
        items.sort(key=lambda item: item.seq)

    rows: list[RecallCall] = []
    for call in parsed.tool_calls:
        if call.tool_name != CLAUDE_RECALL_TOOL or call.outcome == "pending":
            continue
        query = None
        if call.input:
            try:
                decoded = json.loads(call.input)
            except ValueError:
                decoded = None
            if isinstance(decoded, dict):
                value = decoded.get("query")
                query = value if isinstance(value, str) else None

        prefix, suffix = _split(activity.get(call.session_id, []), call.seq)
        followup_text, followup_truncated = _bounded_suffix(suffix)
        overlap_tokens, overlap_count = _overlap(call.result_text, prefix, suffix)

        rows.append(
            RecallCall(
                recall_id=f"{call.file_path}:{call.tool_use_id}",
                session_id=call.session_id,
                file_path=call.file_path,
                source=SOURCE_CLAUDE_CODE,
                seq=call.seq,
                ts=call.ts,
                call_id=call.tool_use_id,
                query=query,
                output=call.result_text,
                output_truncated=call.result_truncated,
                followup_text=followup_text or None,
                followup_truncated=followup_truncated,
                overlap_tokens=json.dumps(overlap_tokens, ensure_ascii=False),
                overlap_count=overlap_count,
            )
        )
    return rows


def extract_from_opencode(
    parsed: ParsedOpenCodeFile, *, result_text_limit: int = DEFAULT_RESULT_TEXT_LIMIT
) -> list[RecallCall]:
    """Completed ``kaiba_recall`` tool parts, paired with same-session followup evidence.

    Mirrors :func:`extract_from_claude`: activity is every text chunk and
    every completed tool call (any tool), grouped by ``sessionID`` and
    ordered by line number.  Unlike the Claude Code path, output is
    truncated here rather than upstream -- :mod:`ashiato.opencode` does not
    truncate on its own account, matching the rest of this module owning
    the result-text truncation convention.

    Every activity component (the joined ``tool`` + ``input`` + ``output`` of a
    non-recall tool call) is itself bounded to ``result_text_limit`` at assembly,
    so a megabyte-sized tool output cannot inflate the in-memory activity list.
    Consequence to be aware of: ``overlap_tokens`` / ``overlap_count`` are computed
    over this truncated suffix, so they reflect distinctive tokens in the bounded
    text rather than the raw output.
    """
    activity: dict[str | None, list[_Activity]] = {}
    for chunk in parsed.text_chunks:
        activity.setdefault(chunk.session_id, []).append(_Activity(chunk.seq, chunk.text))
    for call in parsed.tool_calls:
        text = " ".join(
            part
            for part in (
                call.tool,
                json.dumps(call.input, ensure_ascii=False) if call.input else None,
                call.output,
            )
            if part
        )
        if text:
            # Bound each activity component the same way the recall row's own
            # output is bounded: real tool outputs reach megabytes and the
            # joined activity list lives in memory per file.  Observable
            # consequence: overlap tokens are then computed over this truncated
            # suffix, not over the raw per-call output.
            text = text[:result_text_limit]
            activity.setdefault(call.session_id, []).append(_Activity(call.seq, text))
    for items in activity.values():
        items.sort(key=lambda item: item.seq)

    rows: list[RecallCall] = []
    for call in parsed.tool_calls:
        if call.tool != OPENCODE_RECALL_TOOL:
            continue
        query = call.input.get("query") if call.input else None
        query = query if isinstance(query, str) else None
        raw_output = call.output or ""
        output = raw_output[:result_text_limit]
        output_truncated = len(raw_output) > result_text_limit

        prefix, suffix = _split(activity.get(call.session_id, []), call.seq)
        followup_text, followup_truncated = _bounded_suffix(suffix)
        overlap_tokens, overlap_count = _overlap(output, prefix, suffix)

        rows.append(
            RecallCall(
                recall_id=f"{call.file_path}:{call.call_id}",
                session_id=call.session_id,
                file_path=call.file_path,
                source=SOURCE_OPENCODE,
                seq=call.seq,
                ts=call.ts,
                call_id=call.call_id,
                query=query,
                output=output or None,
                output_truncated=output_truncated,
                followup_text=followup_text or None,
                followup_truncated=followup_truncated,
                overlap_tokens=json.dumps(overlap_tokens, ensure_ascii=False),
                overlap_count=overlap_count,
            )
        )
    return rows
