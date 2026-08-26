"""Mining re-derived facts as kaiba nomination candidates.

When an agent re-derives the same durable fact across many sessions, the
repeated probe shows up as noise in transcripts.  These are strong
candidates for the kaiba conclusions ledger: the fact is real, but it
should be stored rather than rediscovered every time.

Two miners, each finding a different signal:

* ``negative-fact`` -- repeated identical failures across sessions (e.g.
  ``sqlite3: command not found``).
* ``stable-output`` -- repeated probes with identical informative output.

This module is report-only: it never writes to kaiba or any file, mirroring
the discipline of ``ashiato.salvage`` (issue #10).  Pure analysis over an
already-built DuckDB; no new stored tables or views, so this needs no
``FORMAT_VERSION`` bump.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from ashiato.build import assert_readable, connect

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MIN_SESSIONS = 3
DEFAULT_MIN_STABILITY = 1.0
DEFAULT_MAX_OUTPUT_CHARS = 2000

# ---------------------------------------------------------------------------
# Built-in ritual exclusion patterns (commands to exclude from stable-output)
# ---------------------------------------------------------------------------

_BUILTIN_RITUAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"--help\b"),
    re.compile(r"\bgit\s+(pull|fetch|status|log|diff)\b"),
    re.compile(r"\bchain-(show|wait|cancel)\b"),
    re.compile(r"\bkusabi-companion\b"),
    re.compile(r"\bsleep\b"),
    re.compile(r"^\s*(ls|cd|pwd|rm|mkdir|echo)\s*$"),
    re.compile(r"^\s*cat\s*$"),
    re.compile(r"/usage\b"),
]

# ---------------------------------------------------------------------------
# Failure shape patterns (for negative-fact detection)
# ---------------------------------------------------------------------------

_FAILURE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"command not found", re.IGNORECASE),
    re.compile(r"No such file or directory", re.IGNORECASE),
    re.compile(r"Permission denied", re.IGNORECASE),
    re.compile(r"ModuleNotFoundError", re.IGNORECASE),
    re.compile(r"not recognized", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Harness suffix to strip from result text
# ---------------------------------------------------------------------------

_HARNESS_SUFFIX_RE = re.compile(
    r"\nShell cwd was reset to .*?$", re.MULTILINE
)

# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def normalize_result_text(text: str | None, *, max_chars: int = DEFAULT_MAX_OUTPUT_CHARS) -> str:
    """Strip harness-appended suffixes, whitespace, and truncate for comparison."""
    if text is None:
        return ""
    # Strip harness suffix
    cleaned = _HARNESS_SUFFIX_RE.sub("", text)
    # Strip trailing whitespace
    cleaned = cleaned.rstrip()
    # Truncate to max_chars
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned


def normalize_command(input_summary: str | None) -> str:
    """Normalize a command for grouping: collapse variable parts."""
    if input_summary is None:
        return ""
    text = input_summary
    # Collapse UUIDs first (before hex, as UUIDs contain hex-like segments)
    text = re.sub(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        "<UUID>",
        text,
    )
    # Collapse hex runs >= 7 chars that contain at least one a-f/A-F character
    hex_pattern = (
        r"[0-9a-fA-F]*[a-fA-F][0-9a-fA-F]{6,}"
        r"|[0-9a-fA-F]{6,}[a-fA-F][0-9a-fA-F]*"
    )
    text = re.sub(hex_pattern, "<HEX>", text)
    # Collapse integers (4+ digits, but not already collapsed as hex/UUID)
    text = re.sub(r"\b\d{4,}\b", "<N>", text)
    # Collapse /tmp/... path segments
    text = re.sub(r"/tmp/\S+", "/tmp/<...>", text)
    # Collapse ISO dates
    text = re.sub(r"\d{4}-\d{2}-\d{2}", "<DATE>", text)
    text = re.sub(r"\d{2}:\d{2}(:\d{2})?", "<TIME>", text)
    # Squash whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_ritual(command: str, exclude_patterns: list[re.Pattern[str]] | None = None) -> bool:
    """True if the normalized command matches a ritual exclusion pattern."""
    patterns = exclude_patterns if exclude_patterns is not None else _BUILTIN_RITUAL_PATTERNS
    return any(p.search(command) for p in patterns)


def _is_failure_result(result_text: str | None) -> bool:
    """True if the result text matches known failure shapes."""
    if not result_text:
        return False
    return any(p.search(result_text) for p in _FAILURE_PATTERNS)


_OPERATIONAL_DEATH_EXIT_RE = re.compile(
    r"Exit code (?:124|137|143|144)\b",
    re.IGNORECASE,
)
_OPERATIONAL_DEATH_PHRASES = (
    "Command timed out",
    "temporarily unavailable, so auto mode cannot determine",
)


def _is_operational_death(normalized: str) -> bool:
    """True if a classified failure is a watch-loop/dispatch death, not a durable fact."""
    if _OPERATIONAL_DEATH_EXIT_RE.search(normalized):
        return True
    return any(phrase in normalized for phrase in _OPERATIONAL_DEATH_PHRASES)


def _is_uninformative(output: str) -> bool:
    """True if the output is empty, whitespace-only, or the no-output marker."""
    stripped = output.strip()
    return not stripped or stripped == "(Bash completed with no output)"


def _load_exclude_file(path: Path) -> list[re.Pattern[str]]:
    """Load extra ritual exclusion patterns from a file (one regex per line)."""
    patterns: list[re.Pattern[str]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            patterns.append(re.compile(line))
        except re.error as exc:
            raise re.error(
                f"invalid regex on line {lineno}: {line}"
            ) from exc
    return patterns


# ---------------------------------------------------------------------------
# Candidate dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Candidate:
    """One nomination candidate from either miner."""

    signal: str  # "negative-fact" | "stable-output"
    sessions: int
    session_ids: list[str]
    session_timestamps: dict[str, datetime | None]  # session_id -> min(ts)
    command: str
    stability: float | None = None  # stable-output only
    sample_output: str = ""
    draft: str = ""


# ---------------------------------------------------------------------------
# Negative-fact miner
# ---------------------------------------------------------------------------


def mine_negative_facts(
    connection: duckdb.DuckDBPyConnection,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    min_sessions: int = DEFAULT_MIN_SESSIONS,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> list[Candidate]:
    """Repeated identical failures across sessions."""
    # Query: non-sidechain Bash calls that are errors or match failure shapes
    query = """
        SELECT
            session_id,
            tool_use_id,
            ts,
            result_text,
            outcome,
            input_summary
        FROM tool_calls
        WHERE tool_name = 'Bash'
          AND NOT is_sidechain
    """
    params: list[Any] = []
    if since is not None:
        query += " AND ts >= ?"
        params.append(since)
    if until is not None:
        query += " AND ts <= ?"
        params.append(until)

    rows = connection.execute(query, params).fetchall()

    # Group by normalized result text
    groups: dict[str, list[tuple[str, str, datetime | None, str | None]]] = defaultdict(list)
    for row in rows:
        session_id, tool_use_id, ts, result_text, outcome, input_summary = row
        # Normalize first so the failure check and the grouping key use the same
        # text (spec: "whose normalized result text matches failure shapes").
        normalized = normalize_result_text(result_text, max_chars=max_output_chars)
        if not normalized:
            # No comparable text (empty / whitespace-only / stripped suffix) — nothing
            # to group a repeated failure on.
            continue
        # Check failure: is_error or error outcome, or a failure shape in the
        # normalized result text.
        is_failure = outcome == "error" or _is_failure_result(normalized)
        if not is_failure:
            continue
        if _is_operational_death(normalized):
            continue
        groups[normalized].append((session_id, tool_use_id, ts, input_summary))

    candidates: list[Candidate] = []
    for normalized_text, entries in groups.items():
        # Count distinct sessions
        session_map: dict[str, tuple[datetime | None, str | None]] = {}
        for session_id, _tool_use_id, ts, input_summary in entries:
            if session_id not in session_map:
                session_map[session_id] = (ts, input_summary)
        if len(session_map) < min_sessions:
            continue

        # Pick a representative command
        representative_cmd = next(
            (inp for _, inp in session_map.values() if inp), ""
        )

        # Timestamps: first and last
        timestamps = {sid: ts for sid, (ts, _) in session_map.items()}
        valid_ts = [t for t in timestamps.values() if t is not None]
        first_ts = min(valid_ts) if valid_ts else None
        last_ts = max(valid_ts) if valid_ts else None

        first_str = first_ts.isoformat() if first_ts else "?"
        last_str = last_ts.isoformat() if last_ts else "?"
        sample = normalized_text[:200]

        draft = (
            f'"{representative_cmd}" returns the same result in '
            f"{len(session_map)} sessions between {first_str} and {last_str}: "
            f"{sample}"
        )

        candidates.append(
            Candidate(
                signal="negative-fact",
                sessions=len(session_map),
                session_ids=sorted(session_map.keys()),
                session_timestamps=timestamps,
                command=representative_cmd,
                sample_output=sample,
                draft=draft,
            )
        )

    # Sort by session count descending
    candidates.sort(key=lambda c: c.sessions, reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Stable-output miner
# ---------------------------------------------------------------------------


def mine_stable_outputs(
    connection: duckdb.DuckDBPyConnection,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    min_sessions: int = DEFAULT_MIN_SESSIONS,
    min_stability: float = DEFAULT_MIN_STABILITY,
    exclude_patterns: list[re.Pattern[str]] | None = None,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> list[Candidate]:
    """Repeated probes with identical informative output."""
    query = """
        SELECT
            session_id,
            tool_use_id,
            ts,
            input_summary,
            result_text
        FROM tool_calls
        WHERE tool_name = 'Bash'
          AND NOT is_sidechain
    """
    params: list[Any] = []
    if since is not None:
        query += " AND ts >= ?"
        params.append(since)
    if until is not None:
        query += " AND ts <= ?"
        params.append(until)

    rows = connection.execute(query, params).fetchall()

    # Group by normalized command
    # For each command group, track per-session first occurrence
    command_groups: dict[
        str, dict[str, tuple[datetime | None, str | None, str]]
    ] = defaultdict(dict)
    for row in rows:
        session_id, _tool_use_id, ts, input_summary, result_text = row
        cmd = normalize_command(input_summary)
        if not cmd:
            continue
        if _is_ritual(cmd, exclude_patterns):
            continue
        normalized_output = normalize_result_text(result_text, max_chars=max_output_chars)

        # Keep only first occurrence per session for this command
        if session_id not in command_groups[cmd]:
            command_groups[cmd][session_id] = (ts, input_summary, normalized_output)

    candidates: list[Candidate] = []
    for cmd, sessions_data in command_groups.items():
        if len(sessions_data) < min_sessions:
            continue

        # Check stability: modal output share
        outputs = [out for _, _, out in sessions_data.values()]
        if not outputs:
            continue

        # Drop groups whose modal output is uninformative
        counter = Counter(outputs)
        modal_output, modal_count = counter.most_common(1)[0]
        if _is_uninformative(modal_output):
            continue

        stability = modal_count / len(sessions_data)
        if stability < min_stability:
            continue

        # Timestamps
        timestamps = {sid: ts for sid, (ts, _, _) in sessions_data.items()}
        valid_ts = [t for t in timestamps.values() if t is not None]
        first_ts = min(valid_ts) if valid_ts else None
        last_ts = max(valid_ts) if valid_ts else None

        first_str = first_ts.isoformat() if first_ts else "?"
        last_str = last_ts.isoformat() if last_ts else "?"
        sample = modal_output[:200]

        # Representative raw command from first session
        representative_cmd = next(
            (inp for _, inp, _ in sessions_data.values() if inp), cmd
        )

        draft = (
            f'"{representative_cmd}" returns the same result in '
            f"{len(sessions_data)} sessions between {first_str} and {last_str}: "
            f"{sample}"
        )

        candidates.append(
            Candidate(
                signal="stable-output",
                sessions=len(sessions_data),
                session_ids=sorted(sessions_data.keys()),
                session_timestamps=timestamps,
                command=representative_cmd,
                stability=stability,
                sample_output=sample,
                draft=draft,
            )
        )

    # Sort by session count descending
    candidates.sort(key=lambda c: c.sessions, reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------


def _seen_range(
    timestamps: dict[str, datetime | None],
) -> tuple[datetime | None, datetime | None]:
    """Min/max of non-None session timestamps; (None, None) if none are dated."""
    valid = [t for t in timestamps.values() if t is not None]
    if not valid:
        return None, None
    return min(valid), max(valid)


# ---------------------------------------------------------------------------
# Public API: run()
# ---------------------------------------------------------------------------


def run(
    db_path: Path,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    min_sessions: int = DEFAULT_MIN_SESSIONS,
    min_stability: float = DEFAULT_MIN_STABILITY,
    exclude_file: Path | None = None,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    json_output: bool = False,
    out: Any = None,
    err: Any = None,
) -> int:
    """Mine candidates and render output.  Returns exit code (0=candidates, 1=none)."""
    import json
    import sys

    if out is None:
        out = sys.stdout
    if err is None:
        err = sys.stderr

    if not db_path.exists():
        print(f"error: no database at {db_path} (run 'ashiato build' first)", file=err)
        return 1

    connection = connect(db_path, read_only=True)
    try:
        assert_readable(connection)
    except Exception as error:
        print(f"error: {error}", file=err)
        connection.close()
        return 1

    exclude_patterns: list[re.Pattern[str]] | None = None
    if exclude_file is not None:
        if exclude_file.exists():
            exclude_patterns = _BUILTIN_RITUAL_PATTERNS + _load_exclude_file(exclude_file)
        else:
            print(f"warning: exclude file not found: {exclude_file}", file=err)
            exclude_patterns = _BUILTIN_RITUAL_PATTERNS

    try:
        neg_candidates = mine_negative_facts(
            connection,
            since=since,
            until=until,
            min_sessions=min_sessions,
            max_output_chars=max_output_chars,
        )
        stable_candidates = mine_stable_outputs(
            connection,
            since=since,
            until=until,
            min_sessions=min_sessions,
            min_stability=min_stability,
            exclude_patterns=exclude_patterns,
            max_output_chars=max_output_chars,
        )
    except duckdb.Error as error:
        print(f"error: {error}", file=err)
        connection.close()
        return 1
    finally:
        connection.close()

    # Merge: negative-fact first, then stable-output, both sorted by sessions desc
    all_candidates = neg_candidates + stable_candidates

    if not all_candidates:
        return 1

    if json_output:
        payload = []
        for c in all_candidates:
            first_seen, last_seen = _seen_range(c.session_timestamps)
            payload.append(
                {
                    "signal": c.signal,
                    "sessions": c.sessions,
                    "session_ids": c.session_ids,
                    "command": c.command,
                    "stability": c.stability,
                    "sample_output": c.sample_output,
                    "draft": c.draft,
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                }
            )
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str), file=out)
    else:
        # Simple table
        for c in all_candidates:
            stab = f"  stability={c.stability}" if c.stability is not None else ""
            print(
                f"{c.signal}  sessions={c.sessions}{stab}  {c.draft}",
                file=out,
            )
        count = len(all_candidates)
        suffix = "s" if count != 1 else ""
        print(f"({count} candidate{suffix})", file=out)

    return 0
