"""Attributing each Claude Code memory file to the models that wrote it (issue #47).

Claude Code writes ``memory/*.md`` files automatically, and on a machine
shared by several models (fable, opus, sonnet, ...) nobody can tell who wrote
what: the frontmatter ``originSessionId`` is wrong because a session switches
models mid-way (measured: 80 sessions -> 117 distinct (session, model)
pairs).  The write call's own event carries the right model, so this module
reads ``events.model`` for each ``Write`` / ``Edit`` / ``MultiEdit`` call and
attributes each file to the models that actually wrote it.

A memory file is a path matching
``(^|/).claude/projects/<project>/memory/<name>.md`` (after ``\\`` -> ``/``,
so a POSIX and a Windows absolute path meet in the same key).  The key is
``<project>/<name>``.  Only successful calls count; a denied or failed write
is ignored entirely, and a NULL or ``<synthetic>`` model is reported as
``unknown``.  ``created_by`` is the model of the earliest successful
``Write`` -- but only if no successful ``Edit`` / ``MultiEdit`` precedes it;
otherwise the file predates the corpus and the creator is ``unknown``, with
every successful call counted as an edit.  A file whose first successful
call *is* a ``Write`` keeps its creation boundary even when that call's model
is unknown: ``edits`` still covers only the calls after it.

``bash_mentions`` counts ``Bash`` calls (any outcome) whose command contains
both the file name and ``/memory/``: shell edits cannot be attributed, so
they are surfaced instead of silently invisible.

Report-only, mirroring ``ashiato.orphans`` and ``ashiato.nominate``: it never
writes to any file or store, only reads the already-built DuckDB and the
memory directories.  No new stored tables or views, so no ``FORMAT_VERSION``
bump.

Known limit: edits made through Bash or scripts are not attributed -- they
only show up as ``bash_mentions`` on the files they name, so a file rewritten
entirely through the shell has no creator.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from ashiato.build import SchemaOutOfDate, assert_readable, connect

# ---------------------------------------------------------------------------
# Memory-file recognition
# ---------------------------------------------------------------------------

#: A path is a memory file when a ``.claude/projects/<project>/memory``
#: segment is directly followed by a ``*.md`` name.  Applied after ``\\`` ->
#: ``/``, so one regex sees a POSIX and a Windows absolute path.
_MEMORY_PATH_RE = re.compile(r"(^|/)\.claude/projects/[^/]+/memory/[^/]+\.md$")

#: The tool whose calls may touch a memory file without ever persisting a path.
BASH_TOOL = "Bash"

#: A model id that is not a real attribution (``<synthetic>``, ``<subagent>``...).
_SYNTHETIC_PREFIX = "<"

#: A bash command mentions a memory file only when both the file name and
#: this directory segment appear in it.
_MEMORY_SEGMENT = "/memory/"

#: ``unknown`` is used both for "the file predates the corpus" and for "the
#: model id is NULL or synthetic"; the two cases stay apart internally via
#: :attr:`MemoryFile.calls`, but the report shows the same string.
UNKNOWN = "unknown"


def model_name(model: str | None) -> str:
    """The reported name of a model: ``unknown`` when absent or synthetic."""
    if model is None or model.startswith(_SYNTHETIC_PREFIX):
        return UNKNOWN
    return model


def memory_key(path: str) -> str | None:
    """``<project>/<filename>`` for a memory-file path, else ``None``.

    ``\\`` is replaced with ``/`` first, so the same file written through two
    absolute prefixes (``/home/u/.claude/...`` and ``C:\\Users\\u\\.claude/...``)
    is one key.
    """
    normalized = path.replace("\\", "/")
    match = _MEMORY_PATH_RE.search(normalized)
    if match is None:
        return None
    # The matched run is ``[/].claude/projects/<project>/memory/<name>.md``:
    # exactly five slash-separated segments once the leading slash is stripped.
    parts = normalized[match.start() :].strip("/").split("/")
    return f"{parts[2]}/{parts[4]}"


# ---------------------------------------------------------------------------
# Report rows
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MemoryFile:
    """One memory file's authorship report.

    ``calls`` is internal bookkeeping (tool name + raw model per successful
    call, in chronological order) and is deliberately not part of the JSON or
    the table output.
    """

    key: str
    path: str
    created_by: str
    edits: list[tuple[str, int]]
    first_ts: datetime | None
    last_ts: datetime | None
    n_writes: int
    exists: bool
    bash_mentions: int
    calls: list[tuple[str, str | None]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "path": self.path,
            "created_by": self.created_by,
            "edits": [{"model": model, "count": count} for model, count in self.edits],
            "first_ts": self.first_ts.isoformat() if self.first_ts else None,
            "last_ts": self.last_ts.isoformat() if self.last_ts else None,
            "n_writes": self.n_writes,
            "exists": self.exists,
            "bash_mentions": self.bash_mentions,
        }


@dataclass(slots=True)
class ModelSummary:
    """One model's totals over the files in the report."""

    model: str
    files_created: int
    writes: int
    files_touched: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "files_created": self.files_created,
            "writes": self.writes,
            "files_touched": self.files_touched,
        }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

# Join on event_id *and* file_path: a replayed transcript can store the same
# event_id under two file paths, and a bare event_id join would fan out.
_CALLS_SQL = """
    SELECT
        tc.tool_name,
        tc.outcome,
        tc.ts,
        e.model,
        CASE
            WHEN tc.tool_name = 'Bash'
                THEN json_extract_string(tc.input, '$.command')
            ELSE json_extract_string(tc.input, '$.file_path')
        END AS target
    FROM tool_calls tc
    LEFT JOIN events e
        ON e.event_id = tc.call_event_id
       AND e.file_path = tc.file_path
    WHERE tc.tool_name IN ('Write', 'Edit', 'MultiEdit', 'Bash')
"""


def find_memory_files(
    connection: duckdb.DuckDBPyConnection,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[MemoryFile]:
    """Every memory file with a successful write inside the window.

    ``since`` / ``until`` filter the calls considered, inclusively, the same
    way ``ashiato orphans`` does; ``created_by`` is still judged within the
    filtered calls.  A call with a NULL timestamp is excluded whenever either
    bound is set.
    """
    where = "WHERE tc.tool_name IN ('Write', 'Edit', 'MultiEdit', 'Bash')"
    params: list[Any] = []
    if since is not None:
        where += " AND tc.ts >= ?"
        params.append(since)
    if until is not None:
        where += " AND tc.ts <= ?"
        params.append(until)
    query = _CALLS_SQL.replace(
        "WHERE tc.tool_name IN ('Write', 'Edit', 'MultiEdit', 'Bash')", where
    ) + "\n    ORDER BY tc.ts NULLS LAST, tc.file_path, tc.seq"
    rows = connection.execute(query, params).fetchall()

    files: dict[str, MemoryFile] = {}
    bash_commands: list[str] = []
    for tool_name, outcome, ts, model, target in rows:
        if tool_name == BASH_TOOL:
            if target and _MEMORY_SEGMENT in target.replace("\\", "/"):
                bash_commands.append(target.replace("\\", "/"))
            continue
        if target is None or outcome != "ok":
            continue
        key = memory_key(target)
        if key is None:
            continue
        file = files.get(key)
        if file is None:
            file = MemoryFile(key, target, UNKNOWN, [], ts, ts, 0, False, 0, [])
            files[key] = file
        # Rows arrive in ts order, so the last path seen is the most recent.
        file.path = target
        if ts is not None:
            if file.first_ts is None or ts < file.first_ts:
                file.first_ts = ts
            if file.last_ts is None or ts > file.last_ts:
                file.last_ts = ts
        file.n_writes += 1
        file.calls.append((tool_name, model))

    for file in files.values():
        first_tool, first_model = file.calls[0]
        if first_tool == "Write":
            file.created_by = model_name(first_model)
            contributing = file.calls[1:]
        else:
            # The first successful call was an Edit/MultiEdit: the file
            # predates the corpus, so every successful call counts as an edit.
            file.created_by = UNKNOWN
            contributing = file.calls
        counts = Counter(model_name(model) for _, model in contributing)
        file.edits = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        file.exists = Path(file.path).exists()
        file.bash_mentions = sum(
            1
            for command in bash_commands
            if file.key.rsplit("/", 1)[1] in command
        )
    return list(files.values())


def summarize(files: Sequence[MemoryFile]) -> list[ModelSummary]:
    """Per-model totals over *files*: creators, successful calls, keys touched."""
    created: Counter[str] = Counter()
    writes: Counter[str] = Counter()
    touched: dict[str, set[str]] = defaultdict(set)
    for file in files:
        created[file.created_by] += 1
        for _, model in file.calls:
            name = model_name(model)
            writes[name] += 1
            touched[name].add(file.key)
    return [
        ModelSummary(model, created[model], writes[model], len(touched[model]))
        for model in sorted(set(created) | set(writes) | set(touched))
    ]


# ---------------------------------------------------------------------------
# Unattributed files
# ---------------------------------------------------------------------------

#: Successful write calls, with no time filter: ``unattributed`` is judged
#: against the whole DB, so ``--since`` / ``--until`` (which shape the report)
#: never push a written file into the "cannot explain" list.  ``model`` is not
#: needed here, so no join on ``events``.
_WRITES_SQL = """
    SELECT
        tc.tool_name,
        tc.outcome,
        json_extract_string(tc.input, '$.file_path') AS target
    FROM tool_calls tc
    WHERE tc.tool_name IN ('Write', 'Edit', 'MultiEdit')
"""


def memory_keys_with_writes(connection: duckdb.DuckDBPyConnection) -> set[str]:
    """Every memory-file key with a successful write anywhere in the DB.

    ``unattributed`` means "no successful Write/Edit/MultiEdit anywhere", so
    this is computed over *all* calls, ignoring ``--since`` / ``--until`` and
    ``--model``: a file written only outside the window or only by a hidden
    model still has a writer and must not be reported as unattributed.
    """
    keys: set[str] = set()
    for _tool_name, outcome, target in connection.execute(_WRITES_SQL).fetchall():
        if outcome != "ok" or target is None:
            continue
        key = memory_key(target)
        if key is not None:
            keys.add(key)
    return keys


def default_memory_dirs() -> list[Path]:
    """Every existing ``~/.claude/projects/*/memory`` directory."""
    root = Path("~/.claude/projects").expanduser()
    return sorted(path for path in root.glob("*/memory") if path.is_dir())


def scan_memory_dirs(dirs: Sequence[Path]) -> dict[str, Path]:
    """``key -> path`` for every ``*.md`` directly inside *dirs*.

    Only the directory's own children count -- ``memory/sub/a.md`` is not a
    memory file.  The project segment of the key is the parent directory's
    name, matching the way the DB keys are derived, so a file written under
    any absolute prefix lands on the same key.
    """
    scanned: dict[str, Path] = {}
    for directory in dirs:
        if not directory.is_dir():
            continue
        project = directory.parent.name
        for path in sorted(directory.glob("*.md")):
            if path.is_file():
                scanned[f"{project}/{path.name}"] = path
    return scanned


def _matches_model(file: MemoryFile, model: str) -> bool:
    """True when *file* was created or edited by *model*."""
    if file.created_by == model:
        return True
    return any(edit_model == model for edit_model, _ in file.edits)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _print_table(
    columns: Sequence[str], rows: Sequence[Sequence[Any]], stream: Any
) -> None:
    cells = [[str(value) for value in row] for row in rows]
    widths = [len(name) for name in columns]
    for row in cells:
        for index, text in enumerate(row):
            if index < len(widths):
                widths[index] = max(widths[index], len(text))
    print("  ".join(name.ljust(widths[i]) for i, name in enumerate(columns)).rstrip(), file=stream)
    print("  ".join("-" * width for width in widths), file=stream)
    for row in cells:
        print("  ".join(text.ljust(widths[i]) for i, text in enumerate(row)).rstrip(), file=stream)


def _file_line(file: MemoryFile) -> str:
    edits = ", ".join(f"{model}\u00d7{count}" for model, count in file.edits)
    last_ts = file.last_ts.isoformat(sep=" ") if file.last_ts else "?"
    parts = [
        file.key,
        file.created_by,
        f"edits({edits})" if edits else "-",
        str(file.n_writes),
        last_ts,
    ]
    if not file.exists:
        parts.append("[missing]")
    if file.bash_mentions:
        parts.append(f"[bash:{file.bash_mentions}]")
    return "  ".join(parts)


def _render_text(
    files: Sequence[MemoryFile],
    summary: Sequence[ModelSummary],
    unattributed: Sequence[str],
    out: Any,
) -> None:
    print("summary:", file=out)
    _print_table(
        ("model", "files_created", "writes", "files_touched"),
        [[s.model, s.files_created, s.writes, s.files_touched] for s in summary],
        out,
    )
    for file in files:
        print(_file_line(file), file=out)
    if unattributed:
        print("unattributed:", file=out)
        for key in unattributed:
            print(f"  {key}", file=out)
    print(f"({len(files)} file{'' if len(files) == 1 else 's'})", file=out)


# ---------------------------------------------------------------------------
# Public API: run()
# ---------------------------------------------------------------------------


def run(
    db_path: Path,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    memory_dirs: Sequence[Path] = (),
    default_dirs: bool = True,
    model: str | None = None,
    json_output: bool = False,
    out: Any = None,
    err: Any = None,
) -> int:
    """Render the authorship report.  Returns 0, or 1 if the db is unreadable."""
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
        files = find_memory_files(connection, since=since, until=until)
        # Unattributed is judged against every file with a successful write
        # anywhere in the DB: the --since/--until window and the --model filter
        # shape the report only, so they must not push a written file into the
        # "cannot explain" list.
        keys_with_writes = memory_keys_with_writes(connection)
    except SchemaOutOfDate as error:
        print(f"error: {error}", file=err)
        return 1
    except duckdb.Error as error:
        print(f"error: {error}", file=err)
        return 1
    finally:
        connection.close()

    dir_paths = list(memory_dirs)
    if not dir_paths and default_dirs:
        dir_paths = default_memory_dirs()
    for path in dir_paths:
        if not path.is_dir():
            print(f"warning: memory dir not found: {path}", file=err)
    scanned = scan_memory_dirs(dir_paths)

    if model is not None:
        files = [file for file in files if _matches_model(file, model)]
    files.sort(key=lambda file: (file.last_ts or datetime.min, file.key), reverse=True)

    unattributed = sorted(key for key in scanned if key not in keys_with_writes)
    summary = summarize(files)

    if json_output:
        payload = {
            "summary": [s.to_dict() for s in summary],
            "files": [f.to_dict() for f in files],
            "unattributed": unattributed,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False), file=out)
        return 0

    _render_text(files, summary, unattributed, out)
    return 0