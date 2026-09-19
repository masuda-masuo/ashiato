"""A local dashboard over the built database: ``ashiato serve`` (issue #49).

One stdlib HTTP server renders the analyses ``info``, ``orphans``,
``memory-authors``, ``denials``, ``hygiene`` and ``session-trace`` as
server-rendered HTML (and, for the four overview pages, as JSON), so a
10-second glance answers "is the data fresh, and is there anything worth
reading today".

The parts that must never bend:

* **No connection outlives a request.**  A systemd timer rebuilds the database
  with a write connection, and DuckDB refuses a writer while another process
  holds the file.  Every request opens the database read-only, and closes it
  before the response is written.  When the open fails (locked, mid-rebuild,
  missing, out of date) the answer is a plain 503 page and the server keeps
  running.  The open is attempted on *every* request, cache hit or not, so a
  writer that appears never finds a connection here and never gets served
  stale pages as if all were well.
* **Loopback only.**  ``--host`` accepts ``127.0.0.1``, ``::1`` and
  ``localhost``; anything else is refused before a socket exists.  Transcripts
  contain secrets.  The ``Host`` header is checked too, so a page on the web
  cannot reach the server through a DNS name it controls.
* **Everything from the database is escaped.**  Transcript text is untrusted
  and routinely holds ``<script>``, HTML and markdown.
* **The analyses are reused, not re-implemented.**  Each page calls the same
  data functions the CLI commands do.
* **Read-only except one guarded endpoint.**  Nothing is fetched, and the
  pages carry no external asset: CSS and the one filtering script are inline.
  The sole write is ``POST /orphans/reviewed``, which marks or unmarks a
  session as reviewed in the same file the CLI writes.  It is CSRF-guarded:
  every request carries a per-process random token (embedded as a hidden
  form field) and a matching ``Origin`` header equal to ``http://<Host>``
  (the existing loopback Host check applies too).  A cross-site form POST
  to 127.0.0.1 cannot forge both.

The expensive analyses (``orphans`` ~5 s, ``memory-authors`` ~1-2 s on a real
database) are cached per page, keyed by the database file's
``(st_mtime_ns, st_size)``.  ``orphans`` is also keyed by a stat-only fingerprint of the
sink files (a memo written after reading a candidate must retire it on the next reload,
not at the nightly rebuild) and of the reviewed file next to the database (a mark made
with ``ashiato orphans --mark-reviewed`` must hide the candidate on the next reload too),
and ``memory-authors`` re-checks, on every hit, that each file
it reports as present or missing still is.  ``info`` is the exception: its freshness gap
reads the transcript directories, not the database, so caching it by the database's
stamp would hide exactly the staleness it exists to show.
"""

from __future__ import annotations

import hmac
import html
import json
import secrets
import socket
import sys
import threading
import traceback
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import parse_qs, quote, unquote, urlsplit

import duckdb

from ashiato.build import DatabaseInfo, SchemaOutOfDate, assert_readable, connect, database_info
from ashiato.hygiene import audit as hygiene_audit
from ashiato.memory_authors import (
    MemoryFile,
    build_report,
    default_memory_dirs,
    find_memory_files,
    memory_keys_with_writes,
    report_payload,
    scan_memory_dirs,
)
from ashiato.orphans import (
    DEFAULT_MIN_HUMAN_CHARS,
    DEFAULT_MIN_ORPHANS,
    DEFAULT_MIN_TF,
    Candidate,
    _iter_sink_files,
    collect_sessions,
    default_reviewed_path,
    find_orphans,
    load_sinks,
    payload_header,
    read_reviewed,
    resolve_sink_paths,
    update_reviewed,
)
from ashiato.schema import denial_followups_query
from ashiato.session_trace import (
    DEFAULT_EXCERPT_CHARS,
    DEFAULT_LIMIT,
    SessionResolutionError,
    resolve_session,
    trace,
)
from ashiato.topics import DEFAULT_TERMS as DEFAULT_TOPICS_TERMS
from ashiato.topics import DEFAULT_WINDOW as DEFAULT_TOPICS_WINDOW
from ashiato.topics import outline as topics_outline

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8772

#: ``--host`` value -> the address actually bound.  ``localhost`` is pinned to
#: the IPv4 loopback so what gets bound never depends on name resolution.
LOOPBACK_HOSTS = {"127.0.0.1": "127.0.0.1", "localhost": "127.0.0.1", "::1": "::1"}

#: ``Host`` header names a browser on this machine can legitimately send.
_HOST_HEADER_NAMES = frozenset({"localhost", "127.0.0.1", "::1"})

#: Rows per page, as the issue fixes them.
ORPHANS_LIMIT = 30
DENIALS_LIMIT = 50

#: Sent on every response.  The page needs inline CSS and one inline script and
#: one form, and nothing else: no fetch, no image, no frame.
_SECURITY_HEADERS = (
    (
        "Content-Security-Policy",
        "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
        "base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
    ),
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "same-origin"),
    ("Cache-Control", "no-store"),
)


def bind_address(host: str) -> str:
    """The address to bind for *host*; ``ValueError`` unless it is loopback."""
    try:
        return LOOPBACK_HOSTS[host.strip().lower()]
    except KeyError:
        raise ValueError(
            f"--host must be a loopback address (127.0.0.1, ::1 or localhost), got {host!r}: "
            "transcripts contain secrets, so the dashboard never listens beyond this machine"
        ) from None


class DatabaseUnavailable(Exception):
    """The database cannot be opened right now: locked, missing, mid-rebuild."""


class Response(NamedTuple):
    status: int
    content_type: str
    body: bytes
    headers: tuple[tuple[str, str], ...] = ()


# ---------------------------------------------------------------------------
# Data: one connection per request, cached per database stamp
# ---------------------------------------------------------------------------


class Dashboard:
    """The analyses behind the pages; holds no database connection between calls."""

    def __init__(
        self,
        db_path: Path,
        *,
        sinks: Sequence[Path] = (),
        default_sinks: bool = True,
        memory_dirs: Sequence[Path] = (),
        reviewed_file: Path | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.sinks = list(sinks)
        self.default_sinks = default_sinks
        self.memory_dirs = list(memory_dirs)
        self.reviewed_file = (
            Path(reviewed_file)
            if reviewed_file is not None
            else default_reviewed_path(self.db_path)
        )
        self._cache: dict[str, tuple[Hashable, Any]] = {}
        # One lock per cached value name (orphans, memory, denials, and one per
        # session outline), created on first use.  Locks are never evicted:
        # acceptable at this scale (one user, hundreds of sessions).
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

    # -- database access -------------------------------------------------

    def stamp(self) -> tuple[int, int]:
        """The database file's ``(st_mtime_ns, st_size)`` -- the cache key."""
        try:
            stat = self.db_path.stat()
        except OSError as error:
            raise DatabaseUnavailable(f"no database at {self.db_path}") from error
        return stat.st_mtime_ns, stat.st_size

    @contextmanager
    def connection(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """A read-only connection that is closed when the block ends."""
        try:
            connection = connect(self.db_path, read_only=True)
        except duckdb.Error as error:
            raise DatabaseUnavailable(str(error)) from error
        try:
            assert_readable(connection)
            yield connection
        finally:
            connection.close()

    def probe(self) -> None:
        """Open and close the database, so an unavailable one is noticed even on a cache hit."""
        self.stamp()
        with self.connection():
            pass

    def _sink_fingerprint(self) -> Hashable:
        """What the sink files look like right now, from ``stat`` alone (no reads).

        Walks the same paths ``load_sinks`` would read, so adding, removing or
        editing a memo changes it.  Count, newest mtime and total size together
        keep a delete-one-add-one swap from going unnoticed.
        """
        paths = resolve_sink_paths(self.sinks, self.default_sinks)
        missing = count = newest = size = 0
        for path in paths:
            if not path.exists():
                missing += 1
                continue
            for file_path in _iter_sink_files(path):
                try:
                    stat = file_path.stat()
                except OSError:
                    continue
                count += 1
                newest = max(newest, stat.st_mtime_ns)
                size += stat.st_size
        return tuple(paths), missing, count, newest, size

    def _reviewed_fingerprint(self) -> Hashable:
        """The reviewed file's ``(st_mtime_ns, st_size)``, or ``None`` when it is absent.

        A mark or unmark made from the CLI changes this, so the next page
        reload recomputes the candidates without any database change.
        """
        try:
            stat = self.reviewed_file.stat()
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def _orphans_fingerprint(self) -> Hashable:
        """The outside state ``orphans`` depends on: the sinks plus the reviewed file."""
        return self._sink_fingerprint(), self._reviewed_fingerprint()

    def _cached(
        self,
        name: str,
        compute: Callable[[duckdb.DuckDBPyConnection], Any],
        *,
        fingerprint: Callable[[], Hashable] | None = None,
        valid: Callable[[Any], bool] | None = None,
    ) -> Any:
        # The key is taken before the compute: state that changes while the
        # value is being computed is stored under the old key and recomputed
        # next time.  *fingerprint* adds outside state to the key; *valid* vets
        # a cached value against the world before it is served.
        key = (self.stamp(), fingerprint() if fingerprint is not None else None)
        with self._locks[name]:
            hit = self._cache.get(name)
            if hit is not None and hit[0] == key and (valid is None or valid(hit[1])):
                return hit[1]
            with self.connection() as connection:
                value = compute(connection)
            self._cache[name] = (key, value)
            return value

    # -- analyses --------------------------------------------------------

    def orphans(self) -> dict[str, Any]:
        """Every candidate at the CLI defaults, with reviewed flag and hidden
        count; pages filter."""

        def compute(connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
            sessions = collect_sessions(connection)
            loaded = load_sinks(resolve_sink_paths(self.sinks, self.default_sinks))
            reviewed = read_reviewed(self.reviewed_file)
            all_candidates = find_orphans(
                sessions,
                loaded.text,
                limit=0,
                reviewed=reviewed,
                show_reviewed=True,
            )
            return {
                "sessions_with_prose": len(sessions),
                "sink_files": loaded.n_files,
                "all_candidates": all_candidates,
                "candidates": [c for c in all_candidates if not c.reviewed],
                "reviewed_hidden": sum(1 for c in all_candidates if c.reviewed),
                "reviewed_file": str(self.reviewed_file),
            }

        return self._cached("orphans", compute, fingerprint=self._orphans_fingerprint)

    def orphans_payload(self) -> dict[str, Any]:
        """The ``orphans --json`` document, at the page's limit."""
        data = self.orphans()
        header = payload_header(
            data["sessions_with_prose"],
            data["sink_files"],
            since=None,
            until=None,
            min_tf=DEFAULT_MIN_TF,
            min_human_chars=DEFAULT_MIN_HUMAN_CHARS,
            min_orphans=DEFAULT_MIN_ORPHANS,
            include_headless=False,
            limit=ORPHANS_LIMIT,
            reviewed_hidden=data["reviewed_hidden"],
            reviewed_file=data["reviewed_file"],
        )
        candidates = [c.to_dict() for c in data["candidates"][:ORPHANS_LIMIT]]
        return {**header, "candidates": candidates}

    def memory(self, model: str | None = None) -> dict[str, Any]:
        """The ``memory-authors --json`` document, plus the models to filter by."""

        def compute(connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
            return {
                "files": find_memory_files(connection),
                "keys_with_writes": memory_keys_with_writes(connection),
            }

        # ``exists`` (the ``missing`` flag) is a filesystem fact frozen into the
        # cached files, so a hit is vetted against the disk first: a file that
        # appeared or vanished since forces a recompute.
        data = self._cached("memory", compute, valid=_memory_flags_current)
        # The directories are scanned on every request: it is a glob, and a
        # memory file written since the last build should not stay hidden.
        scanned = scan_memory_dirs(self.memory_dirs or default_memory_dirs())
        files, summary, unattributed = build_report(
            data["files"], data["keys_with_writes"], scanned, model
        )
        return {
            **report_payload(files, summary, unattributed),
            "models": _models(data["files"]),
        }

    def denials(self) -> dict[str, Any]:
        """Every denial (newest first) and the hygiene audit, as the CLI computes them."""

        def compute(connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
            query, params = denial_followups_query()
            cursor = connection.execute(query, params)
            columns = [description[0] for description in cursor.description or []]
            rows = [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
            return {"denials": rows, "hygiene": hygiene_audit(connection)}

        return self._cached("denials", compute)

    def denials_payload(self) -> dict[str, Any]:
        data = self.denials()
        return {
            "denials_total": len(data["denials"]),
            "denials": data["denials"][:DENIALS_LIMIT],
            "hygiene": data["hygiene"],
        }

    def overview(self) -> dict[str, Any]:
        """``info`` plus one tile each; freshness is read live, never cached."""
        candidates = self.orphans()["candidates"]
        unattributed = self.memory()["unattributed"]
        denied = len(self.denials()["denials"])
        info = database_info(self.db_path)
        try:
            stat = self.db_path.stat()
        except OSError as error:
            raise DatabaseUnavailable(f"no database at {self.db_path}") from error
        return {
            "db_path": info.db_path,
            "db_modified": datetime.fromtimestamp(stat.st_mtime, UTC).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "table_counts": info.table_counts,
            "started_at": _iso(info.started_at),
            "ended_at": _iso(info.ended_at),
            "roots": _roots(info),
            "freshness": _freshness(info.freshness_gap),
            "tiles": {
                "orphan_candidates": len(candidates),
                "memory_unattributed": len(unattributed),
                "denied_calls": denied,
            },
            "top_orphans": [c.to_dict() for c in candidates[:3]],
        }

    def session(self, prefix: str) -> dict[str, Any]:
        """``session-trace`` plus the topic outline, for a session id or unique prefix.

        The trace is computed at the CLI defaults; the outline is served from
        the per-session cache (keyed like the other pages, by the database
        stamp).  No connection is held between the two lookups, so the
        per-request connection rule still holds.
        """
        with self.connection() as connection:
            session_id = resolve_session(connection, prefix)
            trace_payload = trace(
                connection,
                session_id,
                limit=DEFAULT_LIMIT,
                max_excerpt_chars=DEFAULT_EXCERPT_CHARS,
            )
        return {**trace_payload, "outline": self.topics(session_id)}

    def topics(self, session_id: str) -> dict[str, Any]:
        """One session's topic outline, cached per database stamp."""

        def compute(connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
            return topics_outline(
                connection,
                session_id,
                window=DEFAULT_TOPICS_WINDOW,
                terms=DEFAULT_TOPICS_TERMS,
            )

        return self._cached(f"topics:{session_id}", compute)


def _memory_flags_current(data: dict[str, Any]) -> bool:
    """True while every cached memory file is still as present or missing as it was."""
    return all(Path(file.path).exists() == file.exists for file in data["files"])


def _models(files: Sequence[MemoryFile]) -> list[str]:
    names = {file.created_by for file in files}
    names.update(model for file in files for model, _ in file.edits)
    return sorted(names)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat(sep=" ") if value is not None else None


def _roots(info: DatabaseInfo) -> dict[str, list[dict[str, Any]]] | None:
    kinds = {
        "sources": info.sources,
        "opencode_sources": info.opencode_sources,
        "cursor_sources": info.cursor_sources,
        "codex_sources": info.codex_sources,
    }
    if all(roots is None for roots in kinds.values()):
        return None
    return {
        kind: [{"root": root, "files": count} for root, count in roots or []]
        for kind, roots in kinds.items()
    }


def _freshness(gap: int | None) -> dict[str, Any]:
    state = "unknown" if gap is None else "current" if gap == 0 else "stale"
    return {"state": state, "gap": gap}


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_NAV = (("/", "概況"), ("/orphans", "発掘"), ("/memory", "メモ"), ("/denials", "拒否と衛生"))

_CSS = """
:root{color-scheme:light dark;--paper:#f5f6f8;--ink:#1d2329;--ink-soft:#5a6570;
--rule:#d8dce1;--tint:#eaedf1;--accent:#c4283c}
@media (prefers-color-scheme:dark){:root{--paper:#15181c;--ink:#dde2e7;--ink-soft:#8d98a4;
--rule:#2b3138;--tint:#1d2228;--accent:#ff7080}}
*{box-sizing:border-box}
html{background:var(--paper);color:var(--ink)}
body{margin:0;font:0.95rem/1.5 system-ui,-apple-system,"Segoe UI","Hiragino Sans",
"Noto Sans CJK JP","Yu Gothic",sans-serif}
.wrap{max-width:80rem;margin:0 auto;padding:0 1.5rem 4rem}
.top{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;
gap:.5rem 2rem;padding:1.1rem 0;border-bottom:1px solid var(--rule)}
.brand{font-weight:600;letter-spacing:.02em}
nav a{color:var(--ink-soft);text-decoration:none;margin-left:1.4rem;padding:.25rem 0;
border-bottom:2px solid transparent}
nav a:hover{color:var(--ink)}
nav a[aria-current=page]{color:var(--ink);border-bottom-color:var(--ink)}
h1{font-size:1.35rem;font-weight:600;margin:2rem 0 .25rem}
h2{font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;
color:var(--ink-soft);margin:2.25rem 0 .6rem}
p{margin:.4rem 0}
a{color:inherit}
code,.mono,td.m,td.n,th.n,.value,.term{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,
"Liberation Mono",monospace;font-variant-numeric:tabular-nums}
code{font-size:.88em}
.lede,.dim{color:var(--ink-soft)}
.label,th{font-size:.75rem;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-soft);
font-weight:600}
.fresh{display:flex;flex-wrap:wrap;gap:.4rem 1rem;align-items:baseline;padding:.7rem 1rem;
border:1px solid var(--rule);margin:1.25rem 0}
.fresh.stale{background:var(--accent);border-color:var(--accent);color:var(--paper);
font-weight:600}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(14rem,1fr));gap:1rem}
.tile{display:block;padding:1rem 1.1rem;border:1px solid var(--rule);background:var(--tint);
color:inherit;text-decoration:none;border-radius:4px}
.tile .value{display:block;font-size:2.25rem;font-weight:600;line-height:1.15;margin:.3rem 0 .1rem}
.tile .sub{display:block;font-size:.8rem;color:var(--ink-soft)}
.tile.attn{border-color:var(--accent)}
.tile.attn .value{color:var(--accent)}
.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse}
th{text-align:left;padding:.4rem .75rem;border-bottom:1px solid var(--ink-soft);white-space:nowrap}
td{padding:.5rem .75rem;border-bottom:1px solid var(--rule);vertical-align:top}
tbody tr:hover{background:var(--tint)}
td.n,th.n{text-align:right;white-space:nowrap}
td.m{font-size:.85rem}
td.hot{color:var(--accent);font-weight:600}
.term{display:inline-block;font-size:.8rem;background:var(--tint);padding:0 .35rem;
margin:0 .25rem .2rem 0;border-radius:3px}
.out-denied{font-weight:700}
.attn-text{color:var(--accent);font-weight:600}
details summary{cursor:pointer}
.full{white-space:pre-wrap;overflow-wrap:anywhere;margin:.4rem 0 0;padding:.5rem .7rem;
background:var(--tint);border-left:2px solid var(--rule)}
input.filter{width:100%;max-width:22rem;padding:.4rem .6rem;margin:.2rem 0 .8rem;
border:1px solid var(--rule);background:var(--paper);color:var(--ink);font:inherit;
border-radius:3px}
.chips a{display:inline-block;padding:.1rem .65rem;margin:0 .4rem .4rem 0;
border:1px solid var(--rule);border-radius:99px;color:var(--ink-soft);
text-decoration:none;font-size:.85rem;
font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace}
.chips a[aria-current=true]{background:var(--ink);border-color:var(--ink);color:var(--paper)}
ul.plain{list-style:none;margin:0;padding:0}
ul.plain li{padding:.3rem 0;border-bottom:1px solid var(--rule)}
footer{margin-top:3rem;font-size:.8rem;color:var(--ink-soft)}
"""

#: Client-side table filtering; the only script, and it fetches nothing.
_FILTER_JS = """
document.querySelectorAll('input.filter').forEach(function (box) {
  var rows = document.querySelectorAll('#' + box.dataset.target + ' tbody tr');
  box.addEventListener('input', function () {
    var query = box.value.toLowerCase();
    rows.forEach(function (row) {
      row.hidden = query !== '' && row.textContent.toLowerCase().indexOf(query) === -1;
    });
  });
});
"""


def _e(value: Any) -> str:
    """*value* as escaped HTML text; ``None`` is empty."""
    return html.escape("" if value is None else str(value))


def _ts(value: Any) -> str:
    if value is None:
        return "&mdash;"
    if isinstance(value, datetime):
        return _e(value.strftime("%Y-%m-%d %H:%M:%S"))
    return _e(value)


def _href(path: str, value: str) -> str:
    return _e(f"{path}{quote(value, safe='')}")


def _session_link(session_id: str | None) -> str:
    if not session_id:
        return '<span class="dim">&mdash;</span>'
    label = session_id[:8]
    return f'<a href="{_href("/session/", session_id)}" title="{_e(session_id)}">{_e(label)}</a>'


def _long(text: str | None, preview: int = 80) -> str:
    """*text*, cut to a preview with the whole of it one click away."""
    if not text:
        return '<span class="dim">&mdash;</span>'
    flat = " ".join(text.split())
    if len(flat) <= preview and "\n" not in text:
        return f"<span>{_e(text)}</span>"
    return (
        f"<details><summary>{_e(flat[:preview])}&hellip;</summary>"
        f'<div class="full">{_e(text)}</div></details>'
    )


def _table(
    table_id: str,
    columns: Sequence[tuple[str, str]],
    rows: Sequence[Sequence[str]],
) -> str:
    """A table whose cells are already-escaped HTML; columns are ``(label, td class)``."""
    head = "".join(
        f'<th class="{cls}">{_e(label)}</th>' if cls == "n" else f"<th>{_e(label)}</th>"
        for label, cls in columns
    )
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="{cls}">{cell}</td>' if cls else f"<td>{cell}</td>"
            for (_, cls), cell in zip(columns, row, strict=True)
        )
        + "</tr>"
        for row in rows
    )
    return (
        f'<div class="scroll"><table id="{table_id}"><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _filter_box(table_id: str) -> str:
    return (
        f'<input class="filter" type="search" data-target="{table_id}" '
        'placeholder="filter rows" aria-label="filter rows">'
    )


def _layout(title: str, active: str | None, body: str, *, filterable: bool = False) -> str:
    nav = "".join(
        f'<a href="{href}"{" aria-current=page" if href == active else ""}>{label}</a>'
        for href, label in _NAV
    )
    script = f"<script>{_FILTER_JS}</script>" if filterable else ""
    return (
        '<!doctype html><html lang="ja"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>ashiato &mdash; {_e(title)}</title><style>{_CSS}</style></head>"
        '<body><div class="wrap"><header class="top"><span class="brand">ashiato</span>'
        f"<nav>{nav}</nav></header><main>{body}</main>"
        '<footer>Read-only. Timestamps are UTC as stored. Served from this machine only.'
        f"</footer></div>{script}</body></html>"
    )


def render_overview(view: dict[str, Any]) -> str:
    fresh = view["freshness"]
    gap = fresh["gap"]
    if fresh["state"] == "stale":
        plural = "" if gap == 1 else "s"
        banner = (
            f'<div class="fresh stale" data-freshness="stale"><strong>STALE</strong>'
            f"<span>{gap} new or changed file{plural} under recorded roots &mdash; "
            "run <code>ashiato build</code></span></div>"
        )
    elif fresh["state"] == "current":
        banner = (
            '<div class="fresh" data-freshness="current"><strong>current</strong>'
            '<span class="dim">no new or changed files under recorded roots</span></div>'
        )
    else:
        banner = (
            '<div class="fresh" data-freshness="unknown"><strong>unknown</strong>'
            '<span class="dim">roots not recorded (rebuild to record them)</span></div>'
        )
    tiles = view["tiles"]

    def tile(key: str, label: str, sub: str, value: int, href: str, attention: bool) -> str:
        cls = "tile attn" if attention else "tile"
        return (
            f'<a class="{cls}" data-tile="{key}" href="{href}"><span class="label">{label}</span>'
            f'<span class="value">{value}</span><span class="sub">{sub}</span></a>'
        )

    tiles_html = (
        '<div class="tiles">'
        + tile(
            "orphans",
            "発掘候補",
            "orphan candidates",
            tiles["orphan_candidates"],
            "/orphans",
            False,
        )
        + tile(
            "memory",
            "未帰属メモ",
            "memory files unattributed",
            tiles["memory_unattributed"],
            "/memory",
            tiles["memory_unattributed"] > 0,
        )
        + tile("denials", "拒否", "denied calls", tiles["denied_calls"], "/denials", False)
        + "</div>"
    )
    counts = _table(
        "t-counts",
        (("table", "m"), ("rows", "n")),
        [[_e(name), _e(count)] for name, count in view["table_counts"].items()],
    )
    window = (
        f"{_e(view['started_at'])} &rarr; {_e(view['ended_at'])}"
        if view["started_at"] or view["ended_at"]
        else "empty"
    )
    if view["roots"] is None:
        roots = (
            '<p class="dim">unknown (database built before root recording; '
            "rebuild to record them)</p>"
        )
    else:
        root_rows = [
            [_e(kind), _e(entry["root"]), _e(entry["files"])]
            for kind, entries in view["roots"].items()
            for entry in entries
        ]
        roots = (
            _table("t-roots", (("kind", "m"), ("root", "m"), ("files", "n")), root_rows)
            if root_rows
            else '<p class="dim">(none)</p>'
        )
    top = ""
    if view["top_orphans"]:
        top_rows = [
            [
                f'<span class="attn-text">{c["orphan_density"]:.1f}</span>',
                _session_link(c["session_id"]),
                _long(c["first_utterance"]),
            ]
            for c in view["top_orphans"]
        ]
        top = "<h2>Densest orphan candidates</h2>" + _table(
            "t-top",
            (("density", "n"), ("session", "m"), ("first utterance", "")),
            top_rows,
        )
    body = (
        "<h1>概況</h1>"
        f'<p class="lede">{_e(view["db_path"])} <span class="dim">&mdash; file modified '
        f"{_e(view['db_modified'])} UTC</span></p>"
        f"{banner}{tiles_html}{top}"
        f"<h2>Time window</h2><p class=\"mono\">{window}</p>"
        f"<h2>Tables</h2>{counts}<h2>Ingested roots</h2>{roots}"
    )
    return _layout("概況", "/", body)


def _review_form(session_id: str, csrf_token: str, *, unmark: bool = False) -> str:
    """A small POST form to mark or unmark a session as reviewed."""
    action = "unmark" if unmark else "mark"
    label = "unreview" if unmark else "reviewed"
    return (
        f'<form method="post" action="/orphans/reviewed" class="review-form">'
        f'<input type="hidden" name="token" value="{_e(csrf_token)}">'
        f'<input type="hidden" name="session_id" value="{_e(session_id)}">'
        f'<input type="hidden" name="action" value="{action}">'
        f'<button type="submit">{label}</button>'
        f'</form>'
    )


def render_orphans(
    data: dict[str, Any], csrf_token: str = "", *, show_reviewed: bool = False
) -> str:
    reviewed_hidden = data.get("reviewed_hidden", 0)

    if show_reviewed:
        candidates: list[Candidate] = data.get("all_candidates", data["candidates"])[:ORPHANS_LIMIT]
        toggle = '<a href="/orphans">hide reviewed</a>'
    else:
        candidates = data["candidates"][:ORPHANS_LIMIT]
        toggle = (
            f'<a href="/orphans?reviewed=1">'
            f'show {reviewed_hidden} reviewed</a>'
        ) if reviewed_hidden else ""

    rows = []
    for rank, c in enumerate(candidates):
        terms = "".join(f'<span class="term">{_e(term)}</span>' for term in c.orphan_terms)
        density = f"{c.orphan_density:.1f}"
        review_cell = (
            _review_form(c.session_id, csrf_token, unmark=True)
            if c.reviewed and c.session_id
            else _review_form(c.session_id, csrf_token) if c.session_id else ""
        )
        rows.append(
            [
                _e(rank + 1),
                # The three densest rows are the ones worth reading first: the accent goes on them.
                f'<span class="attn-text">{density}</span>' if rank < 3 else density,
                _e(c.n_orphan),
                _e(c.human_chars),
                _e(c.n_tool_calls),
                _ts(c.started_at),
                _e(c.project_dir),
                terms,
                _long(c.first_utterance),
                _session_link(c.session_id),
                review_cell,
            ]
        )
    columns = (
        ("#", "n"),
        ("density", "n"),
        ("n_orphan", "n"),
        ("human_chars", "n"),
        ("tool_calls", "n"),
        ("started_at", "m"),
        ("project", "m"),
        ("orphan terms", ""),
        ("first utterance", ""),
        ("session", "m"),
        ("", ""),
    )
    table = _table("t-orphans", columns, rows)
    note = ""
    if not data["sink_files"]:
        note = (
            '<p class="dim">No sink text loaded &mdash; every unique term counts as an orphan.</p>'
        )
    empty = "" if candidates else '<p class="dim">No candidates.</p>'
    toggle_html = f' <span class="dim">{toggle}</span>' if toggle else ""
    body = (
        "<h1>発掘</h1>"
        f'<p class="lede">Sessions whose one-off terms left no trace in any sink. '
        f"Corpus: {data['sessions_with_prose']} sessions with prose, "
        f"{data['sink_files']} sink files. Top {ORPHANS_LIMIT} by density.</p>"
        f'<input type="hidden" id="csrf-token" value="{_e(csrf_token)}">'
        f"{note}{toggle_html}{_filter_box('t-orphans')}{table}{empty}"
    )
    return _layout("発掘", "/orphans", body, filterable=True)


def render_memory(view: dict[str, Any], model: str | None) -> str:
    chips = [
        f'<a href="/memory"{" aria-current=true" if model is None else ""}>all</a>'
    ] + [
        f'<a href="{_href("/memory?model=", name)}"'
        f'{" aria-current=true" if name == model else ""}>{_e(name)}</a>'
        for name in view["models"]
    ]
    summary = _table(
        "t-summary",
        (("model", "m"), ("files_created", "n"), ("writes", "n"), ("files_touched", "n")),
        [
            [_e(s["model"]), _e(s["files_created"]), _e(s["writes"]), _e(s["files_touched"])]
            for s in view["summary"]
        ],
    )
    file_rows = []
    for file in view["files"]:
        edits = ", ".join(f"{e['model']}\u00d7{e['count']}" for e in file["edits"])
        flags = []
        if not file["exists"]:
            flags.append("missing")
        if file["bash_mentions"]:
            flags.append(f"bash:{file['bash_mentions']}")
        file_rows.append(
            [
                _e(file["key"]),
                _e(file["created_by"]),
                _e(edits) if edits else '<span class="dim">&mdash;</span>',
                _e(file["n_writes"]),
                _e(file["last_ts"].replace("T", " ") if file["last_ts"] else None)
                or "&mdash;",
                _e(" ".join(flags)),
            ]
        )
    files = _table(
        "t-files",
        (
            ("file", "m"),
            ("created_by", "m"),
            ("edits", "m"),
            ("n_writes", "n"),
            ("last_ts", "m"),
            ("flags", "m"),
        ),
        file_rows,
    )
    if view["unattributed"]:
        items = "".join(
            f'<li class="mono attn-text">{_e(key)}</li>' for key in view["unattributed"]
        )
        unattributed = f'<ul class="plain" id="unattributed">{items}</ul>'
    else:
        unattributed = '<p class="dim">None: every memory file has a recorded write.</p>'
    empty = "" if view["files"] else '<p class="dim">No files match.</p>'
    body = (
        "<h1>メモ</h1>"
        '<p class="lede">Which model wrote each Claude Code memory file, read off the write '
        "call&rsquo;s own event.</p>"
        f'<div class="chips">{"".join(chips)}</div>'
        f"<h2>Per model</h2>{summary}"
        f"<h2>Files ({len(view['files'])})</h2>{_filter_box('t-files')}{files}{empty}"
        f"<h2>Unattributed ({len(view['unattributed'])})</h2>{unattributed}"
    )
    return _layout("メモ", "/memory", body, filterable=True)


def render_denials(payload: dict[str, Any]) -> str:
    rows = []
    for row in payload["denials"]:
        gap = row["gap_seconds"]
        rows.append(
            [
                _ts(row["ts"]),
                _session_link(row["session_id"]),
                _e(row["tool_name"]),
                _long(row["input_summary"]),
                _e(row["followup_kind"]),
                _e(row["next_tool_name"]) or "&mdash;",
                _long(row["next_input_summary"]),
                _e(row["next_outcome"]) or "&mdash;",
                f"{gap:.1f}" if gap is not None else "&mdash;",
            ]
        )
    denials = _table(
        "t-denials",
        (
            ("ts", "m"),
            ("session", "m"),
            ("tool", "m"),
            ("input", ""),
            ("followup", "m"),
            ("next tool", "m"),
            ("next input", ""),
            ("next outcome", "m"),
            ("gap_s", "n"),
        ),
        rows,
    )
    hygiene = payload["hygiene"]
    coverage = hygiene["coverage"]
    categories = _table(
        "t-hygiene",
        (("category", "m"), ("tool_calls", "n"), ("sessions", "n")),
        [[_e(c["name"]), _e(c["tool_calls"]), _e(c["sessions"])] for c in hygiene["categories"]],
    )
    shown = len(payload["denials"])
    body = (
        "<h1>拒否と衛生</h1>"
        f'<p class="lede">Denied tool calls and what the session did next. '
        f"Showing {shown} of {payload['denials_total']}, newest first.</p>"
        f"{_filter_box('t-denials')}{denials}"
        "<h2>Hygiene</h2>"
        f'<p class="dim">Coverage: {_e(coverage["sessions"])} sessions, '
        f"{_e(coverage['tool_calls'])} tool calls.</p>{categories}"
    )
    return _layout("拒否と衛生", "/denials", body, filterable=True)


def render_session(payload: dict[str, Any]) -> str:
    session = payload["session"]
    coverage = payload["coverage"]
    meta = " ".join(
        f"{label} <span class=\"mono\">{_e(session.get(key))}</span>"
        for label, key in (
            ("started", "started_at"),
            ("ended", "ended_at"),
            ("events", "n_events"),
            ("tool calls", "n_tool_calls"),
        )
        if session.get(key) is not None
    )
    outline = payload.get("outline")
    outline_html = ""
    if outline and outline["segments"]:
        outline_rows = []
        for number, segment in enumerate(outline["segments"], start=1):
            terms = "".join(
                f'<span class="term">{_e(term)}</span>' for term in segment["terms"]
            )
            range_cell = (
                f"{_e(segment['start_ts']) or '&mdash;'} &rarr; "
                f"{_e(segment['end_ts']) or '&mdash;'}"
            )
            outline_rows.append(
                [
                    _e(number),
                    range_cell,
                    terms or '<span class="dim">&mdash;</span>',
                    _long(segment["opening"]),
                ]
            )
        title_html = (
            _e(outline["title"]) if outline["title"] else '<span class="dim">no title</span>'
        )
        outline_html = (
            "<h2>Outline</h2>"
            f'<p class="dim">{title_html} &mdash; {_e(outline["n_exchanges"])} exchanges</p>'
            + _table(
                "t-outline",
                (("#", "n"), ("range", "m"), ("terms", ""), ("opening", "")),
                outline_rows,
            )
        )
    rows = []
    for row in payload["timeline"]:
        if row["kind"] == "text":
            who = _e(row.get("role"))
            content = _long(row.get("excerpt"), 120)
            outcome = ""
        else:
            who = _e(row.get("tool_name"))
            content = (
                f'<div class="mono">{_long(row.get("input_summary"), 120)}</div>'
                f"{_long(row.get('excerpt'), 120) if row.get('excerpt') else ''}"
            )
            recall = row.get("recall")
            if recall:
                content += f'<div class="dim">recall: {_e(recall.get("query"))}</div>'
            followup = row.get("followup")
            if followup:
                content += (
                    f'<div class="dim">&rarr; {_e(followup["followup_kind"])} '
                    f"{_e(followup['next_tool_name'])}</div>"
                )
            label = _e(row.get("outcome"))
            outcome = f'<span class="out-denied">{label}</span>' if row.get(
                "outcome"
            ) == "denied" else label
        rows.append(
            [
                _e(row["seq"]),
                _ts(row["ts"]),
                _e(row["kind"]),
                f'<span title="{_e(row["id"])}">{_e(str(row["id"])[:12])}</span>',
                who,
                outcome,
                content,
            ]
        )
    timeline = _table(
        "t-trace",
        (
            ("seq", "n"),
            ("ts", "m"),
            ("kind", "m"),
            ("id", "m"),
            ("role / tool", "m"),
            ("outcome", "m"),
            ("content", ""),
        ),
        rows,
    )
    cut = ""
    if coverage["truncated"]:
        cut = (
            f" Truncated at {coverage['returned']} rows; "
            f"<code>ashiato session-trace {_e(session['session_id'])} --limit 0</code> "
            "prints them all."
        )
    body = (
        "<h1>session</h1>"
        f'<p class="mono">{_e(session["session_id"])}</p>'
        f'<p class="lede">{meta}</p>'
        f"{outline_html}"
        f'<p class="dim">Showing {coverage["returned"]} of {coverage["total"]} rows.{cut}</p>'
        f"{_filter_box('t-trace')}{timeline}"
    )
    return _layout("session", None, body, filterable=True)


def render_message(title: str, message: str, detail: str | None = None) -> str:
    extra = f'<p class="dim mono">{_e(detail)}</p>' if detail else ""
    return _layout(title, None, f"<h1>{_e(title)}</h1><p>{_e(message)}</p>{extra}")


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _html(body: str, status: int = 200, headers: tuple[tuple[str, str], ...] = ()) -> Response:
    return Response(status, "text/html; charset=utf-8", body.encode("utf-8"), headers)


def _json(payload: Any, status: int = 200, headers: tuple[tuple[str, str], ...] = ()) -> Response:
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    return Response(status, "application/json; charset=utf-8", text.encode("utf-8"), headers)


def _error(
    status: int,
    title: str,
    message: str,
    *,
    api: bool,
    detail: str | None = None,
    headers: tuple[tuple[str, str], ...] = (),
) -> Response:
    if api:
        return _json({"status": status, "error": title, "message": message}, status, headers)
    return _html(render_message(title, message, detail), status, headers)


def host_allowed(header: str | None) -> bool:
    """True when a ``Host`` header names this machine (or is absent, as in HTTP/1.0)."""
    if header is None:
        return True
    host = header.strip()
    if host.startswith("["):
        name = host[1:].split("]", 1)[0]
    elif host.count(":") == 1:
        name = host.rsplit(":", 1)[0]
    else:
        name = host
    return name.lower() in _HOST_HEADER_NAMES


class App:
    """Routing and error mapping over a :class:`Dashboard`; pure, no sockets."""

    def __init__(self, dashboard: Dashboard) -> None:
        self.dashboard = dashboard
        self._csrf_token = secrets.token_urlsafe(32)

    def _check_csrf(
        self,
        host_header: str | None,
        origin: str | None,
        content_length: str | None,
        body: bytes,
    ) -> Response | None:
        """Validate CSRF guards for the reviewed-mark endpoint.

        Returns ``None`` on success (caller should proceed), or an error
        ``Response`` to return immediately.
        """
        # 1. Host must be loopback.
        if not host_allowed(host_header):
            return _error(
                403, "Forbidden",
                "This dashboard only answers requests to localhost.",
                api=False,
            )

        # 2. Origin header must be present and match http://<Host>.
        if not origin:
            return _error(403, "Forbidden", "Missing Origin header.", api=False)
        if host_header is None:
            # HTTP/1.0 has no Host; Origin cannot be validated.
            return _error(
                403, "Forbidden",
                "Missing Host header; cannot validate Origin.",
                api=False,
            )
        expected_origin = f"http://{host_header.strip()}"
        if not hmac.compare_digest(origin, expected_origin):
            return _error(403, "Forbidden", "Origin does not match Host.", api=False)

        # 3. Content-Length must be present, numeric, and small.
        if content_length is None:
            return _error(400, "Bad request", "Missing Content-Length header.", api=False)
        try:
            length = int(content_length)
        except ValueError:
            return _error(400, "Bad request", "Invalid Content-Length header.", api=False)
        if length > 4096:
            return _error(413, "Request too large", "Body exceeds 4 KiB limit.", api=False)
        if length != len(body):
            return _error(
                400, "Bad request",
                "Content-Length does not match body length.",
                api=False,
            )

        # 4. Parse form body.
        try:
            form = parse_qs(body.decode("utf-8"))
        except UnicodeDecodeError:
            return _error(400, "Bad request", "Body is not valid UTF-8.", api=False)

        # 5. Validate token.
        tokens = form.get("token", [])
        if not tokens or not hmac.compare_digest(tokens[0], self._csrf_token):
            return _error(403, "Forbidden", "Invalid or missing CSRF token.", api=False)

        # 6. Validate session_id is present.
        session_ids = form.get("session_id", [])
        if not session_ids or not session_ids[0]:
            return _error(400, "Bad request", "Missing session_id.", api=False)

        # 7. Validate action is mark or unmark.
        actions = form.get("action", [])
        if not actions or actions[0] not in ("mark", "unmark"):
            return _error(400, "Bad request", "Action must be 'mark' or 'unmark'.", api=False)

        return None  # caller proceeds

    def handle(
        self,
        target: str,
        method: str = "GET",
        host_header: str | None = None,
        origin: str | None = None,
        content_length: str | None = None,
        body: bytes = b"",
    ) -> Response:
        """The response to ``{method} target``; never raises."""
        api = target.startswith("/api/")
        if not host_allowed(host_header):
            return _error(
                403, "Forbidden", "This dashboard only answers requests addressed to localhost.",
                api=api,
            )
        if method == "OPTIONS":
            return Response(405, "text/plain", b"", (("Allow", "GET, HEAD"),))
        # POST /orphans/reviewed is the only write.
        if method == "POST":
            parts = urlsplit(target)
            path = parts.path.rstrip("/") or "/"
            if path == "/orphans/reviewed":
                return self._handle_post_reviewed(
                    host_header, origin, content_length, body
                )
            # Every other POST path is 405.
            return Response(405, "text/plain", b"", (("Allow", "GET, HEAD"),))
        # PUT/DELETE/PATCH are always 405.
        if method not in ("GET", "HEAD"):
            return Response(405, "text/plain", b"", (("Allow", "GET, HEAD"),))
        try:
            parts = urlsplit(target)
            path = parts.path.rstrip("/") or "/"
            query = parse_qs(parts.query)
            route = self._route(path, query)
            if route is None:
                return _error(404, "Not found", f"There is no page at {path}.", api=api)
            self.dashboard.probe()
            return route()
        except SessionResolutionError as error:
            return _error(404, "Session not found", str(error), api=api)
        except (
            DatabaseUnavailable,
            SchemaOutOfDate,
            duckdb.IOException,
            duckdb.ConnectionException,
        ) as error:
            return _error(
                503,
                "Database unavailable",
                "The database is being rebuilt (or is not readable yet). "
                "Reload in a minute.",
                api=api,
                detail=str(error),
                headers=((("Retry-After", "30"),)),
            )
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            return _error(
                500,
                "Internal error",
                f"{type(error).__name__}: {error}",
                api=api,
            )

    def _handle_post_reviewed(
        self,
        host_header: str | None,
        origin: str | None,
        content_length: str | None,
        body: bytes,
    ) -> Response:
        """Handle POST /orphans/reviewed."""
        csrf_err = self._check_csrf(host_header, origin, content_length, body)
        if csrf_err is not None:
            return csrf_err

        form = parse_qs(body.decode("utf-8"))
        session_id = form["session_id"][0]
        action = form["action"][0]

        # Validate session_id against the database.  The web accepts only a
        # *full* session id that exists exactly: no prefix resolution, unlike
        # the CLI, so a typo'd or guessed prefix can never mark a session the
        # page did not show.
        try:
            self.dashboard.probe()
            with self.dashboard.connection() as conn:
                row = conn.execute(
                    "SELECT 1 FROM sessions WHERE session_id = ? "
                    "UNION "
                    "SELECT 1 FROM tool_calls WHERE session_id = ?",
                    [session_id, session_id],
                ).fetchone()
        except (DatabaseUnavailable, SchemaOutOfDate, duckdb.Error) as error:
            return _error(
                503, "Database unavailable",
                f"Cannot validate session id: {error}",
                api=False,
                detail=str(error),
            )
        if row is None:
            return _error(
                400, "Bad request",
                f"Unknown session id: {session_id}",
                api=False,
            )

        reviewed_path = self.dashboard.reviewed_file
        try:
            if action == "mark":
                update_reviewed(reviewed_path, add={session_id})
            else:  # unmark
                update_reviewed(reviewed_path, remove={session_id})
        except (OSError, ValueError) as error:
            return _error(
                500, "Internal error",
                f"Failed to write reviewed file: {error}",
                api=False,
            )

        # Invalidate the orphans cache so the next GET sees the change.
        self.dashboard._cache.pop("orphans", None)

        # 303 See Other -> /orphans
        return Response(
            303, "text/html", b"",
            (("Location", "/orphans"),),
        )

    def _route(self, path: str, query: dict[str, list[str]]) -> Callable[[], Response] | None:
        dashboard = self.dashboard
        model = (query.get("model") or [""])[0] or None
        show_reviewed = "reviewed" in query

        def overview_page() -> Response:
            return _html(render_overview(dashboard.overview()))

        def orphans_page() -> Response:
            return _html(render_orphans(
                dashboard.orphans(), self._csrf_token, show_reviewed=show_reviewed,
            ))

        def memory_page() -> Response:
            return _html(render_memory(dashboard.memory(model), model))

        def denials_page() -> Response:
            return _html(render_denials(dashboard.denials_payload()))

        pages: dict[str, Callable[[], Response]] = {
            "/": overview_page,
            "/orphans": orphans_page,
            "/memory": memory_page,
            "/denials": denials_page,
            "/api/overview.json": lambda: _json(dashboard.overview()),
            "/api/info.json": lambda: _json(dashboard.overview()),
            "/api/orphans.json": lambda: _json(dashboard.orphans_payload()),
            "/api/memory.json": lambda: _json(_without_models(dashboard.memory(model))),
            "/api/denials.json": lambda: _json(dashboard.denials_payload()),
        }
        if path in pages:
            return pages[path]
        if path.startswith("/session/") and len(path) > len("/session/"):
            prefix = unquote(path[len("/session/"):])
            return lambda: _html(render_session(dashboard.session(prefix)))
        return None


def _without_models(view: dict[str, Any]) -> dict[str, Any]:
    """The ``memory-authors --json`` document: the page-only ``models`` list dropped."""
    return {key: value for key, value in view.items() if key != "models"}


def _make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ashiato-serve"

        def _send(self, response: Response) -> None:
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            self.send_header("Content-Length", str(len(response.body)))
            for name, value in (*_SECURITY_HEADERS, *response.headers):
                self.send_header(name, value)
            self.end_headers()

        def _answer(self, *, send_body: bool) -> None:
            response = app.handle(self.path, host_header=self.headers.get("Host"))
            self._send(response)
            if send_body:
                self.wfile.write(response.body)

        def do_GET(self) -> None:
            self._answer(send_body=True)

        def do_HEAD(self) -> None:
            self._answer(send_body=False)

        def do_POST(self) -> None:
            parts = urlsplit(self.path)
            path = parts.path.rstrip("/") or "/"
            if path != "/orphans/reviewed":
                # Every other POST path is 405 without reading the body,
                # whatever its Content-Length header says.
                response = app.handle(
                    self.path,
                    method="POST",
                    host_header=self.headers.get("Host"),
                    origin=None,
                    content_length=None,
                    body=b"",
                )
                self._send(response)
                if response.body:
                    self.wfile.write(response.body)
                return
            cl = self.headers.get("Content-Length")
            if cl is None:
                body = b""
            else:
                try:
                    length = int(cl)
                except ValueError:
                    err = _error(400, "Bad request", "Invalid Content-Length.", api=False)
                    self._send(err)
                    self.wfile.write(err.body)
                    return
                if length > 4096:
                    err = _error(413, "Request too large", "Body exceeds 4 KiB limit.", api=False)
                    self._send(err)
                    self.wfile.write(err.body)
                    return
                body = self.rfile.read(length)
            origin = self.headers.get("Origin")
            response = app.handle(
                self.path,
                method="POST",
                host_header=self.headers.get("Host"),
                origin=origin,
                content_length=cl,
                body=body,
            )
            self._send(response)
            if response.status == 303:
                pass  # redirect with no body
            elif response.body:
                self.wfile.write(response.body)

        def _refuse(self) -> None:
            self.send_response(405)
            self.send_header("Allow", "GET, HEAD")
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _refuse

    return Handler


class _Server4(ThreadingHTTPServer):
    daemon_threads = True


class _Server6(_Server4):
    address_family = socket.AF_INET6


def make_server(
    db_path: Path,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    sinks: Sequence[Path] = (),
    default_sinks: bool = True,
    memory_dirs: Sequence[Path] = (),
    reviewed_file: Path | None = None,
) -> ThreadingHTTPServer:
    """A bound, not yet serving, dashboard server.  ``ValueError`` for a non-loopback *host*."""
    address = bind_address(host)
    dashboard = Dashboard(
        db_path,
        sinks=sinks,
        default_sinks=default_sinks,
        memory_dirs=memory_dirs,
        reviewed_file=reviewed_file,
    )
    server_class = _Server6 if ":" in address else _Server4
    return server_class((address, port), _make_handler(App(dashboard)))


def server_url(server: ThreadingHTTPServer) -> str:
    address, port = server.server_address[:2]
    shown = f"[{address}]" if ":" in str(address) else address
    return f"http://{shown}:{port}/"


def run(
    db_path: Path,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    sinks: Sequence[Path] = (),
    default_sinks: bool = True,
    memory_dirs: Sequence[Path] = (),
    reviewed_file: Path | None = None,
    out: Any = None,
    err: Any = None,
) -> int:
    """Serve until interrupted.  Returns 2 for a non-loopback host, 1 if it cannot listen."""
    if out is None:
        out = sys.stdout
    if err is None:
        err = sys.stderr
    try:
        bind_address(host)
    except ValueError as error:
        print(f"error: {error}", file=err)
        return 2
    if not Path(db_path).exists():
        print(
            f"notice: no database at {db_path} yet -- pages answer 503 until "
            "'ashiato build' has run",
            file=err,
        )
    try:
        server = make_server(
            db_path,
            host=host,
            port=port,
            sinks=sinks,
            default_sinks=default_sinks,
            memory_dirs=memory_dirs,
            reviewed_file=reviewed_file,
        )
    except OSError as error:
        print(f"error: cannot listen on {host} port {port}: {error}", file=err)
        return 1
    print(server_url(server), file=out, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
