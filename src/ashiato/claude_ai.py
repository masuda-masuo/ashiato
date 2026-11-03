"""Pure parser for claude.ai conversation exports.

The export format is undocumented. This module handles the measured JSON
snapshot shape, selecting the ancestry of the latest leaf in each conversation.
"""

from __future__ import annotations

import json
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ashiato.parser import parse_timestamp

ROOT_PARENT = "00000000-0000-4000-8000-000000000000"
KNOWN_BLOCKS = {
    "text",
    "tool_use",
    "tool_result",
    "thinking",
    "injected_prompt_block",
    "token_budget",
    "flag",
    "image",
    "document",
}
SKIPPED_BLOCKS = {"token_budget", "flag", "image", "document"}
#: Meta blocks and the field their text is read from (measured on a real
#: export: ``injected_prompt_block`` has no ``text``, its text is ``prompt``).
_TEXT_FIELD = {"injected_prompt_block": "prompt", "thinking": "thinking"}



@dataclass(slots=True)
class ClaudeAiConversation:
    uuid: str
    name: str
    created_at: datetime | None
    updated_at: datetime | None
    messages: list[dict[str, Any]]
    message_count: int = 0
    off_path: int = 0


@dataclass(slots=True)
class ParsedClaudeAiExport:
    conversations: list[ClaudeAiConversation] = field(default_factory=list)
    skipped_blocks: Counter[str] = field(default_factory=Counter)
    unknown_blocks: Counter[str] = field(default_factory=Counter)
    malformed: int = 0


def _decode(path: Path) -> Any:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive, archive.open("conversations.json") as handle:
            return json.load(handle)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_export(path: str | Path) -> ParsedClaudeAiExport:
    """Read a bare JSON export or a ZIP containing conversations.json."""
    parsed = ParsedClaudeAiExport()
    raw = _decode(Path(path).expanduser())
    if not isinstance(raw, list):
        raise ValueError("export root is not an array")
    for item in raw:
        try:
            if not isinstance(item, dict) or not isinstance(item.get("uuid"), str):
                raise ValueError
            messages = item.get("chat_messages", [])
            if not isinstance(messages, list):
                raise ValueError
            name = item.get("name")
            conv = ClaudeAiConversation(
                uuid=item["uuid"],
                name=name if isinstance(name, str) else "",
                created_at=parse_timestamp(item.get("created_at")),
                updated_at=parse_timestamp(item.get("updated_at")),
                messages=[m for m in messages if isinstance(m, dict)],
                message_count=len(messages),
            )
            _select_branch(conv)
            parsed.conversations.append(conv)
            for message in conv.messages:
                blocks = message.get("content", [])
                if not isinstance(blocks, list):
                    continue
                for block in blocks:
                    if not isinstance(block, dict):
                        parsed.unknown_blocks["<non-object>"] += 1
                        continue
                    kind = block.get("type")
                    if not isinstance(kind, str):
                        parsed.unknown_blocks["<missing>"] += 1
                    elif kind in SKIPPED_BLOCKS:
                        parsed.skipped_blocks[kind] += 1
                    elif kind not in KNOWN_BLOCKS:
                        parsed.unknown_blocks[kind] += 1
                    elif kind in _TEXT_FIELD:
                        # Neither case produces a row; both are counted.
                        field_name = _TEXT_FIELD[kind]
                        value = block.get(field_name)
                        if not isinstance(value, str):
                            # Field absent or not a string: the format may
                            # have changed, so this is an alarm.
                            parsed.unknown_blocks[f"{kind}:<no {field_name}>"] += 1
                        elif not value:
                            # Present but empty: the normal state (2,192 of
                            # 4,584 thinking blocks in a real export).
                            parsed.skipped_blocks[f"{kind}:<empty>"] += 1
        except (KeyError, TypeError, ValueError):
            parsed.malformed += 1
    return parsed


def _select_branch(conv: ClaudeAiConversation) -> None:
    msgs = conv.messages
    if not msgs:
        return
    by_id = {m.get("uuid"): m for m in msgs if isinstance(m.get("uuid"), str)}
    parents = {m.get("uuid"): m.get("parent_message_uuid") for m in by_id.values()}
    has_child = {p for p in parents.values() if p in by_id}
    leaves = [m for m in by_id.values() if m.get("uuid") not in has_child]
    if not leaves:
        conv.messages = []
        conv.off_path = len(msgs)
        return
    latest = max(
        leaves,
        key=lambda m: (
            parse_timestamp(m.get("created_at")) or datetime.min,
            str(m.get("uuid", "")),
        ),
    )
    path = []
    current = latest
    seen: set[str] = set()
    while isinstance(current.get("uuid"), str) and current["uuid"] not in seen:
        mid = current["uuid"]
        seen.add(mid)
        path.append(current)
        parent = current.get("parent_message_uuid")
        if not isinstance(parent, str) or parent == ROOT_PARENT or parent not in by_id:
            break
        current = by_id[parent]
    conv.off_path = len(msgs) - len(path)
    conv.messages = list(reversed(path))
