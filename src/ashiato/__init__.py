"""ashiato: Claude Code transcripts in, queryable DuckDB out.

Deterministic by construction -- no model calls, no network, no telemetry.
"""

from ashiato.parser import (
    DEFAULT_RESULT_TEXT_LIMIT,
    DENIAL_PATTERNS,
    Event,
    ParsedFile,
    Session,
    ToolCall,
    parse_file,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_RESULT_TEXT_LIMIT",
    "DENIAL_PATTERNS",
    "Event",
    "ParsedFile",
    "Session",
    "ToolCall",
    "__version__",
    "parse_file",
]
