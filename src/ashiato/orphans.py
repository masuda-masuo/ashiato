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
* **Worth reading** -- at least ``min_orphans`` distinct orphan terms (density
  is noisy for tiny counts), and not a headless session (an SDK / headless
  entrypoint, ``sdk-*`` -- kusabi workers, subagents -- whose "human" text is
  a machine-written brief) unless the caller opts in.

Document frequency is computed over every session that has any prose --
headless ones included -- whatever the ``since``/``until`` window or the
thresholds: the window only restricts which sessions are *nominated*, and must
not make an old topic look unique.

Known limit: *absence of the words is not absence of the idea*.  A topic that
was persisted under different words is still nominated.  This is nomination
only; judging the candidate is for the reader.

Report-only, mirroring ``ashiato.nominate`` and ``ashiato.salvage``: it never
writes to the database, to any sink or to any other file, only reads the
already-built DuckDB and the sink files.  The single exception is the
*reviewed file*: ``--mark-reviewed`` / ``--unmark-reviewed`` write one full
session id per line to ``orphans-reviewed.txt`` next to the database, and
nomination reads that file to skip sessions a human has already judged.  The
marks live outside the database on purpose: a delete-and-rebuild of the
database must not forget them.  No new stored tables or views, so no
``FORMAT_VERSION`` bump.
"""

from __future__ import annotations

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None  # type: ignore[assignment]
import json
import os
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from ashiato.build import SchemaOutOfDate, assert_readable, connect
from ashiato.session_trace import SessionResolutionError, resolve_session

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MIN_TF = 3
DEFAULT_MIN_HUMAN_CHARS = 800
DEFAULT_MIN_ORPHANS = 3
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
# A bare lowercase letter+digit token (no ``-``, no ``_``): a background-task
# id, a short hex handle, a Windows file id, ``urllib3``.  No hyphen, so the
# boundary-less words the regex can also match (``bench-10k``) are untouched.
_ALNUM_ID_RE = re.compile(r"^[a-z0-9]+$")


def strip_harness(text: str) -> str:
    """*text* without the harness wrapper blocks."""
    return _HARNESS_BLOCK_RE.sub("", text)


def tokenize(text: str) -> list[str]:
    """Lowercased terms of *text*, minus code identifiers and id-like tokens."""
    terms = []
    for match in _TOKEN_RE.finditer(_UUID_RE.sub(" ", text)):
        term = match.group().lower()
        # A token with ``_`` is a code identifier -- its real sink is the repo.
        if "_" in term or _HEX_ID_RE.search(term):
            continue
        # A token made only of letters and digits, with at least one of each,
        # is an id (``bk7tgrw6i``, ``c1bf1``, ``urllib3``), not a topic word.
        if _ALNUM_ID_RE.match(term) and any(ch.isdigit() for ch in term):
            continue
        terms.append(term)
    return terms


def is_compaction_summary(raw: str | None) -> bool:
    """True when a user row's ``raw`` marks it as a compaction summary.

    Claude Code's continuation message (\"This session is being continued from
    a previous conversation ...\") is a user row whose JSON carries
    ``\"isCompactSummary\": true``.  ``raw`` is not always valid JSON (other
    source formats store different text there), so it is parsed defensively:
    anything unparseable, or missing the flag, is not a summary.  Machine-written
    recaps of earlier content must not count as human prose.
    """
    if not raw:
        return False
    try:
        record = json.loads(raw)
    except ValueError:
        return False
    return record.get("isCompactSummary") is True


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

# Join on file_path, never session_id: a session_id fans out across files.
# Tool results are user-role rows carrying ``"tool_result"`` in ``raw``.
# ``raw`` is selected too: compaction summaries (a user row whose JSON carries
# ``"isCompactSummary": true``) must not count as human prose, and whether a
# row is one is decided in Python by :func:`is_compaction_summary`.
_EVENTS_QUERY = """
    SELECT
        s.file_path,
        s.session_id,
        s.started_at,
        s.project_dir,
        s.n_tool_calls,
        s.entrypoint,
        e.role,
        e.text,
        e.raw
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
    entrypoint: str | None = None
    human_chars: int = 0
    first_utterance: str = ""
    has_prose: bool = False
    terms: Counter[str] = field(default_factory=Counter)
    total_terms: int = 0

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
        terms = tokenize(prose)
        self.terms.update(terms)
        self.total_terms += len(terms)


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
    for (
        file_path,
        session_id,
        started_at,
        project_dir,
        n_tool_calls,
        entrypoint,
        role,
        text,
        raw,
    ) in _iter_rows(connection):
        if current is None or current.file_path != file_path:
            current = SessionProse(
                file_path,
                session_id,
                started_at,
                project_dir,
                n_tool_calls,
                entrypoint=entrypoint,
            )
            sessions.append(current)
        # A compaction summary is a machine-written recap of earlier content,
        # not something the human typed: it contributes nothing.
        if role == "user" and is_compaction_summary(raw):
            continue
        current.add(role, text)
    return [session for session in sessions if session.has_prose]


def is_headless(entrypoint: str | None) -> bool:
    """True when *entrypoint* is an SDK / headless variant (``sdk-*``)."""
    return entrypoint is not None and entrypoint.startswith("sdk")


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
# Reviewed marks
# ---------------------------------------------------------------------------

#: The reviewed-file name.  One full session id per line, UTF-8; blank lines
#: and lines starting with ``#`` are ignored, surrounding whitespace is
#: stripped.  It sits next to the database -- outside it -- so deleting and
#: rebuilding the database cannot lose a mark.
REVIEWED_FILE_NAME = "orphans-reviewed.txt"


def default_reviewed_path(db_path: Path) -> Path:
    """The reviewed file next to *db_path*: ``<db dir>/orphans-reviewed.txt``."""
    return Path(db_path).parent / REVIEWED_FILE_NAME


def _lock_path(path: Path) -> Path:
    """The sibling lock file for *path*."""
    return path.with_suffix(path.suffix + ".lock")


def read_reviewed(path: Path) -> frozenset[str]:
    """The session ids marked reviewed in *path*; a missing file is empty.

    A file that *exists* but cannot be read (it is a directory, permission
    denied, not valid UTF-8) raises ``ValueError`` -- never silently treated
    as empty.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return frozenset()
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 in reviewed file {path}: {exc}") from exc
    ids: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.add(line)
    return frozenset(ids)


def update_reviewed(
    path: Path,
    *,
    add: Iterable[str] = (),
    remove: Iterable[str] = (),
) -> frozenset[str]:
    """Under the exclusive lock: read the file, apply add/remove, write atomically.

    Returns the set as it was *before* the update (callers use it for
    'already reviewed' / 'not reviewed' messages).  A missing file is an
    empty set.  Existing blank and comment lines are preserved verbatim and
    the resulting ids are laid out as in :func:`_write_reviewed_inner`; the
    parent directory is created if needed; the update is idempotent.

    The exclusive ``fcntl`` lock (a sibling ``.lock`` file) is held across the
    read, the update and the atomic write, so two concurrent read-modify-write
    from the CLI and the dashboard cannot drop each other's marks; on a
    platform without ``fcntl`` (or when the lock file cannot be opened) the
    update proceeds unlocked rather than failing.
    """
    adds = frozenset(add)
    removes = frozenset(remove)
    if fcntl is None:
        return _update_reviewed_inner(path, adds, removes)
    lock = _lock_path(path)
    try:
        fd = open(lock, "a+")  # noqa: SIM115
    except OSError:
        # Lock file cannot be opened (e.g. read-only directory); proceed
        # without serialisation rather than failing.
        return _update_reviewed_inner(path, adds, removes)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        return _update_reviewed_inner(path, adds, removes)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


def _update_reviewed_inner(
    path: Path,
    add: frozenset[str],
    remove: frozenset[str],
) -> frozenset[str]:
    """Inner update: read the current ids, apply add/remove, then atomic replace.

    Returns the set as it was *before* the update.  Callers hold the lock.
    """
    current = read_reviewed(path)
    _write_reviewed_inner(path, (current | add) - remove)
    return current


def _write_reviewed_inner(path: Path, ids: frozenset[str]) -> None:
    """Inner write: read-modify-write, then atomic replace."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []
    kept: set[str] = set()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            out.append(line)
        elif stripped in ids and stripped not in kept:
            kept.add(stripped)
            out.append(stripped)
    out.extend(sorted(ids - kept))
    if not out and not path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: temp file + fsync + os.replace.
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + ("\n" if out else ""))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise


def mark_reviewed(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    prefixes: Sequence[str],
    *,
    out: Any = None,
    err: Any = None,
) -> int:
    """Resolve *prefixes* against the database and append their full session
    ids to the reviewed file *path*.

    Every prefix is resolved first (an exact id wins, otherwise the prefix
    must be unique); only when all resolve is the file written, so a missing
    or ambiguous id changes nothing.  Prints one line per id to stdout --
    ``reviewed: <id>`` or ``already reviewed: <id>`` -- and returns 0; on any
    resolution failure, or when *path* exists but cannot be read, prints an
    error to stderr and returns 1.
    """
    if out is None:
        out = sys.stdout
    if err is None:
        err = sys.stderr
    try:
        ids = [resolve_session(connection, prefix) for prefix in prefixes]
    except SessionResolutionError as error:
        print(f"error: {error}", file=err)
        return 1
    try:
        existing = update_reviewed(path, add=ids)
    except (OSError, ValueError) as error:
        print(f"error: cannot read reviewed file {path}: {error}", file=err)
        return 1
    for session_id in ids:
        print(
            f"{'already reviewed' if session_id in existing else 'reviewed'}: {session_id}",
            file=out,
        )
    return 0


def unmark_reviewed(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    prefixes: Sequence[str],
    *,
    out: Any = None,
    err: Any = None,
) -> int:
    """Remove the ids *prefixes* name from the reviewed file *path*.

    An id is matched against the file content first -- so an id that is in the
    file still unmarks even when its session is no longer in the database (a
    rebuild must never make a mark unremovable) -- and resolved against the
    database only when it is not there.  Every prefix is resolved before the
    file is written.  Prints one line per id to stdout -- ``unmarked: <id>``
    or ``not reviewed: <id>`` -- and returns 0; on any resolution failure, or
    when *path* exists but cannot be read, prints an error to stderr and
    returns 1.
    """
    if out is None:
        out = sys.stdout
    if err is None:
        err = sys.stderr
    # The read here is only to decide how to resolve each prefix (an id that
    # is in the file unmarks even when its session is gone from the database);
    # the removal itself happens under the lock, inside update_reviewed.
    try:
        existing = read_reviewed(path)
    except (OSError, ValueError) as error:
        print(f"error: cannot read reviewed file {path}: {error}", file=err)
        return 1
    ids: list[str] = []
    for prefix in prefixes:
        if prefix in existing:
            ids.append(prefix)
            continue
        try:
            ids.append(resolve_session(connection, prefix))
        except SessionResolutionError as error:
            print(f"error: {error}", file=err)
            return 1
    try:
        before = update_reviewed(path, remove=ids)
    except (OSError, ValueError) as error:
        print(f"error: cannot read reviewed file {path}: {error}", file=err)
        return 1
    for session_id in ids:
        print(
            f"{'unmarked' if session_id in before else 'not reviewed'}: {session_id}",
            file=out,
        )
    return 0


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
    total_terms: int
    orphan_terms: list[str]
    first_utterance: str
    reviewed: bool = False

    @property
    def orphan_density(self) -> float:
        """Orphan terms per thousand terms: rankable across session sizes."""
        if not self.total_terms:
            return 0.0
        return self.n_orphan / self.total_terms * 1000

    def to_dict(self, *, include_reviewed: bool = False) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "project_dir": self.project_dir,
            "n_tool_calls": self.n_tool_calls,
            "human_chars": self.human_chars,
            "n_unique": self.n_unique,
            "n_orphan": self.n_orphan,
            "orphan_density": self.orphan_density,
            "orphan_terms": self.orphan_terms,
            "first_utterance": self.first_utterance,
        }
        if include_reviewed:
            fields["reviewed"] = self.reviewed
        return fields


def _nominate(
    sessions: Sequence[SessionProse],
    sink_text: str,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    min_tf: int = DEFAULT_MIN_TF,
    min_human_chars: int = DEFAULT_MIN_HUMAN_CHARS,
    min_orphans: int = DEFAULT_MIN_ORPHANS,
    include_headless: bool = False,
    reviewed: frozenset[str] = frozenset(),
    show_reviewed: bool = False,
    limit: int = DEFAULT_LIMIT,
) -> tuple[list[Candidate], int]:
    """The candidates, and how many would-be ones the ``reviewed`` mark hid.

    Document frequency is taken over all of *sessions* -- headless sessions
    and reviewed sessions included, so their terms keep their owners' words
    from looking unique -- and only the nomination is restricted by the
    window, thresholds, the headless exclusion and the ``reviewed`` mark.
    """
    document_frequency: Counter[str] = Counter()
    for session in sessions:
        document_frequency.update(session.terms.keys())

    candidates: list[Candidate] = []
    for session in sessions:
        if is_headless(session.entrypoint) and not include_headless:
            continue
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
        if len(orphans) < min_orphans:
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
                total_terms=session.total_terms,
                orphan_terms=[term for term, _ in ranked[:TOP_TERMS]],
                first_utterance=session.first_utterance,
                reviewed=session.session_id in reviewed,
            )
        )

    candidates.sort(
        key=lambda c: (
            -c.orphan_density,
            -c.n_orphan,
            -c.human_chars,
            c.session_id or "",
            c.file_path,
        )
    )
    if show_reviewed:
        return candidates[:limit] if limit else candidates, 0
    hidden = sum(1 for candidate in candidates if candidate.reviewed)
    visible = [candidate for candidate in candidates if not candidate.reviewed]
    return visible[:limit] if limit else visible, hidden


def find_orphans(
    sessions: Sequence[SessionProse],
    sink_text: str,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    min_tf: int = DEFAULT_MIN_TF,
    min_human_chars: int = DEFAULT_MIN_HUMAN_CHARS,
    min_orphans: int = DEFAULT_MIN_ORPHANS,
    include_headless: bool = False,
    reviewed: frozenset[str] = frozenset(),
    show_reviewed: bool = False,
    limit: int = DEFAULT_LIMIT,
) -> list[Candidate]:
    """Nominate sessions whose unique terms appear in no sink text.

    *sessions* is the whole corpus: document frequency is taken over all of
    it -- headless sessions included, so an SDK / headless (``sdk-*``)
    session's terms keep their owners' words from looking unique -- and only
    the nomination is restricted by the window, thresholds, the headless
    exclusion and the ``reviewed`` mark.  A session whose ``session_id`` is in
    *reviewed* is skipped unless *show_reviewed* -- every file of that session
    is hidden, because the key is the session id, not the transcript file --
    but it still counts toward document frequency exactly like a headless or
    out-of-window session.
    """
    candidates, _ = _nominate(
        sessions,
        sink_text,
        since=since,
        until=until,
        min_tf=min_tf,
        min_human_chars=min_human_chars,
        min_orphans=min_orphans,
        include_headless=include_headless,
        reviewed=reviewed,
        show_reviewed=show_reviewed,
        limit=limit,
    )
    return candidates


def find_orphans_with_hidden(
    sessions: Sequence[SessionProse],
    sink_text: str,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    min_tf: int = DEFAULT_MIN_TF,
    min_human_chars: int = DEFAULT_MIN_HUMAN_CHARS,
    min_orphans: int = DEFAULT_MIN_ORPHANS,
    include_headless: bool = False,
    reviewed: frozenset[str] = frozenset(),
    show_reviewed: bool = False,
    limit: int = DEFAULT_LIMIT,
) -> tuple[list[Candidate], int]:
    """Like :func:`find_orphans`, but also reports how many would-be
    candidates the ``reviewed`` mark hid (0 with ``show_reviewed``).
    """
    return _nominate(
        sessions,
        sink_text,
        since=since,
        until=until,
        min_tf=min_tf,
        min_human_chars=min_human_chars,
        min_orphans=min_orphans,
        include_headless=include_headless,
        reviewed=reviewed,
        show_reviewed=show_reviewed,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# Public API: run()
# ---------------------------------------------------------------------------


def resolve_sink_paths(sinks: Sequence[Path], default_sinks: bool) -> list[Path]:
    """The sink paths to load: the caller's, else every default memory dir."""
    paths = list(sinks)
    if not paths and default_sinks:
        paths = default_sink_dirs()
    return paths


def payload_header(
    n_sessions: int,
    n_sink_files: int,
    *,
    since: datetime | None,
    until: datetime | None,
    min_tf: int,
    min_human_chars: int,
    min_orphans: int,
    include_headless: bool,
    limit: int,
    reviewed_hidden: int = 0,
    reviewed_file: str | None = None,
) -> dict[str, Any]:
    """The corpus and threshold block that opens the ``--json`` document."""
    return {
        "sessions_with_prose": n_sessions,
        "sink_files": n_sink_files,
        "reviewed_hidden": reviewed_hidden,
        "reviewed_file": reviewed_file,
        "thresholds": {
            "min_tf": min_tf,
            "min_human_chars": min_human_chars,
            "min_orphans": min_orphans,
            "include_headless": include_headless,
            "limit": limit,
            "since": since.isoformat() if since else None,
            "until": until.isoformat() if until else None,
        },
    }


def _header(
    n_sessions: int,
    n_sink_files: int,
    *,
    since: datetime | None,
    until: datetime | None,
    min_tf: int,
    min_human_chars: int,
    min_orphans: int,
    include_headless: bool,
    limit: int,
    reviewed_hidden: int = 0,
) -> str:
    window = ""
    if since is not None:
        window += f" since={since.isoformat()}"
    if until is not None:
        window += f" until={until.isoformat()}"
    return (
        f"corpus: {n_sessions} session{'' if n_sessions == 1 else 's'} with prose, "
        f"sinks: {n_sink_files} file{'' if n_sink_files == 1 else 's'}, "
        f"min-tf={min_tf} min-human-chars={min_human_chars} min-orphans={min_orphans} "
        f"include-headless={'yes' if include_headless else 'no'} limit={limit}{window}"
        f" reviewed-hidden={reviewed_hidden}"
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
    min_orphans: int = DEFAULT_MIN_ORPHANS,
    include_headless: bool = False,
    limit: int = DEFAULT_LIMIT,
    json_output: bool = False,
    reviewed_file: Path | None = None,
    show_reviewed: bool = False,
    out: Any = None,
    err: Any = None,
) -> int:
    """Nominate candidates and render them.  Returns 0, or 1 if the database
    or the reviewed file is unreadable."""
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

    loaded = load_sinks(resolve_sink_paths(sinks, default_sinks))
    for path in loaded.missing:
        print(f"warning: sink not found: {path}", file=err)
    if not loaded.text.strip():
        print(
            "notice: no sink text loaded -- every unique term counts as an orphan",
            file=err,
        )

    reviewed_path = (
        Path(reviewed_file) if reviewed_file is not None else default_reviewed_path(db_path)
    )
    try:
        reviewed = read_reviewed(reviewed_path)
    except (OSError, ValueError) as error:
        print(f"error: cannot read reviewed file {reviewed_path}: {error}", file=err)
        return 1

    candidates, reviewed_hidden = find_orphans_with_hidden(
        sessions,
        loaded.text,
        since=since,
        until=until,
        min_tf=min_tf,
        min_human_chars=min_human_chars,
        min_orphans=min_orphans,
        include_headless=include_headless,
        reviewed=reviewed,
        show_reviewed=show_reviewed,
        limit=limit,
    )
    header = payload_header(
        len(sessions),
        loaded.n_files,
        since=since,
        until=until,
        min_tf=min_tf,
        min_human_chars=min_human_chars,
        min_orphans=min_orphans,
        include_headless=include_headless,
        limit=limit,
        reviewed_hidden=reviewed_hidden,
        reviewed_file=str(reviewed_path),
    )

    if json_output:
        payload = {
            **header,
            "candidates": [c.to_dict(include_reviewed=show_reviewed) for c in candidates],
        }
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
            min_orphans=min_orphans,
            include_headless=include_headless,
            limit=limit,
            reviewed_hidden=reviewed_hidden,
        ),
        file=out,
    )
    for c in candidates:
        started = c.started_at.isoformat(sep=" ") if c.started_at else "?"
        mark = " [reviewed]" if c.reviewed else ""
        print("", file=out)
        print(f"session {c.session_id}{mark}  {started}  {c.project_dir or '?'}", file=out)
        print(
            f"  tool_calls={c.n_tool_calls}  human_chars={c.human_chars}  "
            f"unique={c.n_unique}  orphan={c.n_orphan}  "
            f"density={c.orphan_density:.1f}",
            file=out,
        )
        print(f"  terms: {', '.join(c.orphan_terms)}", file=out)
        print(f"  first: {c.first_utterance}", file=out)
    print(f"({len(candidates)} candidate{'' if len(candidates) == 1 else 's'})", file=out)
    return 0
