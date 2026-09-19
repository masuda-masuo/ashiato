"""Mining one-off discussion topics that left no trace (issue #43).

Ad-hoc discussions -- a design chat, a "what do you think about X", a story
idea -- happen once and never reach a PR, a memory file or a ledger.
``ashiato nominate`` needs *repetition across sessions* and ``ashiato salvage``
needs *tool-call evidence*, so neither can see them, and summarising every
session with an LLM is too expensive.  This module nominates candidates
deterministically so a human (or an LLM) reads only the top few.

A session is a candidate when all three hold:

* **Unique** -- it contains terms that occur in no other session in the whole
  database (document frequency 1) at least ``min_tf`` times.
* **Discussion** -- its human-authored text is at least ``min_human_chars``
  characters, after harness wrappers (``<local-command-stdout>``,
  ``<system-reminder>``, ...) are stripped.
* **Not persisted** -- those unique terms appear in no *sink* text: the memory
  directories, notes repos, ... the caller names.

Document frequency is computed over every session that has any prose,
whatever the ``since``/``until`` window or the human-chars threshold: the
window only restricts which sessions are *nominated*, and must not make an old
topic look unique.

Known limit: *absence of the words is not absence of the idea*.  A topic that
was persisted under different words is still nominated.  This is nomination
only; judging the candidate is for the reader.

Report-only, mirroring ``ashiato.nominate`` and ``ashiato.salvage``: it never
writes to any file or store, only reads the already-built DuckDB and the sink
files.  No new stored tables or views, so no ``FORMAT_VERSION`` bump.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from ashiato.build import SchemaOutOfDate, assert_readable, connect

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MIN_TF = 3
DEFAULT_MIN_HUMAN_CHARS = 800
DEFAULT_LIMIT = 20

#: Orphan terms shown per candidate.
TOP_TERMS = 10
FIRST_UTTERANCE_CHARS = 160

#: Sink files bigger than this, or with a NUL byte early on, are not prose.
MAX_SINK_BYTES = 1024 * 1024
_BINARY_PROBE_BYTES = 8192
_SKIPPED_SINK_DIRS = frozenset({".git", "node_modules", ".venv", "__pycache__"})

_FETCH_BATCH = 5000

# ---------------------------------------------------------------------------
# Text handling
# ---------------------------------------------------------------------------

#: A harness-injected ``<tag>...</tag>`` block: not human prose.
_HARNESS_BLOCK_RE = re.compile(
    r"<(command-[a-z-]+|local-command-[a-z-]+|bash-[a-z-]+|system-reminder|task-notification)"
    r"(?:\s[^>]*)?>.*?</\1>",
    re.DOTALL,
)

# Latin words of 4+ characters, katakana runs of 3+ (U+30A1-U+30F4 and the
# long-vowel mark U+30FC) and kanji runs of 2+ (U+4E00-U+9FA5 and U+3005).
_TOKEN_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9_\-]{3,}|[ァ-ヴー]{3,}|[一-龥々]{2,}"
)
_HEX_ID_RE = re.compile(r"^[0-9a-f]{7,}$|[0-9a-f]{8}-")
# A whole UUID is removed before tokenising: one that starts with digits would
# otherwise be matched from its first letter on, losing the 8-hex prefix that
# ``_HEX_ID_RE`` recognises.
_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)


def strip_harness(text: str) -> str:
    """*text* without the harness wrapper blocks."""
    return _HARNESS_BLOCK_RE.sub("", text)


def tokenize(text: str) -> list[str]:
    """Lowercased terms of *text*, minus code identifiers and hex/UUID ids."""
    terms = []
    for match in _TOKEN_RE.finditer(_UUID_RE.sub(" ", text)):
        term = match.group().lower()
        # A token with ``_`` is a code identifier -- its real sink is the repo.
        if "_" in term or _HEX_ID_RE.search(term):
            continue
        terms.append(term)
    return terms


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

# Join on file_path, never session_id: a session_id fans out across files.
# Tool results are user-role rows carrying ``"tool_result"`` in ``raw``.
_EVENTS_QUERY = """
    SELECT
        s.file_path,
        s.session_id,
        s.started_at,
        s.project_dir,
        s.n_tool_calls,
        e.role,
        e.text
    FROM events e
    JOIN sessions s ON e.file_path = s.file_path
    WHERE e.role IN ('user', 'assistant')
      AND NOT COALESCE(e.is_meta, FALSE)
      AND NOT COALESCE(e.is_sidechain, FALSE)
      AND NOT (e.role = 'user' AND e.raw LIKE '%"tool_result"%')
    ORDER BY s.file_path, e.seq
"""


@dataclass(slots=True)
class SessionProse:
    """One session's prose, reduced to what the analysis needs."""

    file_path: str
    session_id: str | None
    started_at: datetime | None
    project_dir: str | None
    n_tool_calls: int | None
    human_chars: int = 0
    first_utterance: str = ""
    has_prose: bool = False
    terms: Counter[str] = field(default_factory=Counter)

    def add(self, role: str, text: str | None) -> None:
        """Fold one event in: harness wrappers stripped from human text."""
        if not text:
            return
        if role == "user":
            prose = strip_harness(text).strip()
            if not prose:
                return
            self.human_chars += len(prose)
            if not self.first_utterance:
                self.first_utterance = " ".join(prose.split())[:FIRST_UTTERANCE_CHARS]
        else:
            prose = text
            if not prose.strip():
                return
        self.has_prose = True
        self.terms.update(tokenize(prose))


def _iter_rows(connection: duckdb.DuckDBPyConnection) -> Iterator[tuple[Any, ...]]:
    cursor = connection.execute(_EVENTS_QUERY)
    while True:
        rows = cursor.fetchmany(_FETCH_BATCH)
        if not rows:
            return
        yield from rows


def collect_sessions(connection: duckdb.DuckDBPyConnection) -> list[SessionProse]:
    """Every session (one per transcript file) that has any prose."""
    sessions: list[SessionProse] = []
    current: SessionProse | None = None
    for file_path, session_id, started_at, project_dir, n_tool_calls, role, text in _iter_rows(
        connection
    ):
        if current is None or current.file_path != file_path:
            current = SessionProse(file_path, session_id, started_at, project_dir, n_tool_calls)
            sessions.append(current)
        current.add(role, text)
    return [session for session in sessions if session.has_prose]


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Sinks:
    """The combined lowercased text of every sink file that was loaded."""

    text: str = ""
    n_files: int = 0
    missing: list[Path] = field(default_factory=list)


def default_sink_dirs() -> list[Path]:
    """Every existing ``~/.claude/projects/*/memory`` directory."""
    root = Path("~/.claude/projects").expanduser()
    return sorted(path for path in root.glob("*/memory") if path.is_dir())


def _iter_sink_files(path: Path) -> Iterator[Path]:
    if path.is_file():
        yield path
        return
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = sorted(name for name in dirnames if name not in _SKIPPED_SINK_DIRS)
        for name in sorted(filenames):
            yield Path(dirpath) / name


def _read_sink_file(path: Path) -> str | None:
    """The lowercased text of *path*, or ``None`` if it is not readable prose."""
    try:
        if path.stat().st_size > MAX_SINK_BYTES:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:_BINARY_PROBE_BYTES]:
        return None
    return data.decode("utf-8", errors="ignore").lower()


def load_sinks(paths: Sequence[Path]) -> Sinks:
    """Read every file under *paths* (files or directories) as one text."""
    sinks = Sinks()
    parts: list[str] = []
    seen: set[Path] = set()
    for path in paths:
        if not path.exists():
            sinks.missing.append(path)
            continue
        for file_path in _iter_sink_files(path):
            resolved = file_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            text = _read_sink_file(file_path)
            if text is None:
                continue
            parts.append(text)
            sinks.n_files += 1
    sinks.text = "\n".join(parts)
    return sinks


# ---------------------------------------------------------------------------
# Nomination
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Candidate:
    """One session nominated as a discussion that left no trace."""

    session_id: str | None
    file_path: str
    started_at: datetime | None
    project_dir: str | None
    n_tool_calls: int | None
    human_chars: int
    n_unique: int
    n_orphan: int
    orphan_terms: list[str]
    first_utterance: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "project_dir": self.project_dir,
            "n_tool_calls": self.n_tool_calls,
            "human_chars": self.human_chars,
            "n_unique": self.n_unique,
            "n_orphan": self.n_orphan,
            "orphan_terms": self.orphan_terms,
            "first_utterance": self.first_utterance,
        }


def find_orphans(
    sessions: Sequence[SessionProse],
    sink_text: str,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    min_tf: int = DEFAULT_MIN_TF,
    min_human_chars: int = DEFAULT_MIN_HUMAN_CHARS,
    limit: int = DEFAULT_LIMIT,
) -> list[Candidate]:
    """Nominate sessions whose unique terms appear in no sink text.

    *sessions* is the whole corpus: document frequency is taken over all of
    it, and only the nomination is restricted by the window and thresholds.
    """
    document_frequency: Counter[str] = Counter()
    for session in sessions:
        document_frequency.update(session.terms.keys())

    candidates: list[Candidate] = []
    for session in sessions:
        if session.human_chars < min_human_chars:
            continue
        if since is not None and (session.started_at is None or session.started_at < since):
            continue
        if until is not None and (session.started_at is None or session.started_at > until):
            continue
        unique = {
            term: count
            for term, count in session.terms.items()
            if count >= min_tf and document_frequency[term] == 1
        }
        orphans = {term: count for term, count in unique.items() if term not in sink_text}
        if not orphans:
            continue
        ranked = sorted(orphans.items(), key=lambda item: (-item[1], item[0]))
        candidates.append(
            Candidate(
                session_id=session.session_id,
                file_path=session.file_path,
                started_at=session.started_at,
                project_dir=session.project_dir,
                n_tool_calls=session.n_tool_calls,
                human_chars=session.human_chars,
                n_unique=len(unique),
                n_orphan=len(orphans),
                orphan_terms=[term for term, _ in ranked[:TOP_TERMS]],
                first_utterance=session.first_utterance,
            )
        )

    candidates.sort(
        key=lambda c: (-c.n_orphan, -c.human_chars, c.session_id or "", c.file_path)
    )
    return candidates[:limit] if limit else candidates


# ---------------------------------------------------------------------------
# Public API: run()
# ---------------------------------------------------------------------------


def _header(
    n_sessions: int,
    n_sink_files: int,
    *,
    since: datetime | None,
    until: datetime | None,
    min_tf: int,
    min_human_chars: int,
    limit: int,
) -> str:
    window = ""
    if since is not None:
        window += f" since={since.isoformat()}"
    if until is not None:
        window += f" until={until.isoformat()}"
    return (
        f"corpus: {n_sessions} session{'' if n_sessions == 1 else 's'} with prose, "
        f"sinks: {n_sink_files} file{'' if n_sink_files == 1 else 's'}, "
        f"min-tf={min_tf} min-human-chars={min_human_chars} limit={limit}{window}"
    )


def run(
    db_path: Path,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    sinks: Sequence[Path] = (),
    default_sinks: bool = True,
    min_tf: int = DEFAULT_MIN_TF,
    min_human_chars: int = DEFAULT_MIN_HUMAN_CHARS,
    limit: int = DEFAULT_LIMIT,
    json_output: bool = False,
    out: Any = None,
    err: Any = None,
) -> int:
    """Nominate candidates and render them.  Returns 0, or 1 if the db is unreadable."""
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
        sessions = collect_sessions(connection)
    except SchemaOutOfDate as error:
        print(f"error: {error}", file=err)
        return 1
    except duckdb.Error as error:
        print(f"error: {error}", file=err)
        return 1
    finally:
        connection.close()

    sink_paths = list(sinks)
    if not sink_paths and default_sinks:
        sink_paths = default_sink_dirs()
    loaded = load_sinks(sink_paths)
    for path in loaded.missing:
        print(f"warning: sink not found: {path}", file=err)
    if not loaded.text.strip():
        print(
            "notice: no sink text loaded -- every unique term counts as an orphan",
            file=err,
        )

    candidates = find_orphans(
        sessions,
        loaded.text,
        since=since,
        until=until,
        min_tf=min_tf,
        min_human_chars=min_human_chars,
        limit=limit,
    )
    header = {
        "sessions_with_prose": len(sessions),
        "sink_files": loaded.n_files,
        "thresholds": {
            "min_tf": min_tf,
            "min_human_chars": min_human_chars,
            "limit": limit,
            "since": since.isoformat() if since else None,
            "until": until.isoformat() if until else None,
        },
    }

    if json_output:
        payload = {**header, "candidates": [c.to_dict() for c in candidates]}
        print(json.dumps(payload, indent=2, ensure_ascii=False), file=out)
        return 0

    print(
        _header(
            len(sessions),
            loaded.n_files,
            since=since,
            until=until,
            min_tf=min_tf,
            min_human_chars=min_human_chars,
            limit=limit,
        ),
        file=out,
    )
    for c in candidates:
        started = c.started_at.isoformat(sep=" ") if c.started_at else "?"
        print("", file=out)
        print(f"session {c.session_id}  {started}  {c.project_dir or '?'}", file=out)
        print(
            f"  tool_calls={c.n_tool_calls}  human_chars={c.human_chars}  "
            f"unique={c.n_unique}  orphan={c.n_orphan}",
            file=out,
        )
        print(f"  terms: {', '.join(c.orphan_terms)}", file=out)
        print(f"  first: {c.first_utterance}", file=out)
    print(f"({len(candidates)} candidate{'' if len(candidates) == 1 else 's'})", file=out)
    return 0
