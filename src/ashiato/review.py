"""Read-only, period-by-period review report (issue #107, slice 1)."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import duckdb

from ashiato.build import SchemaOutOfDate, assert_readable, connect
from ashiato.orphans import (
    collect_sessions,
    is_compaction_summary,
    is_headless,
    strip_harness,
    tokenize,
)
from ashiato.pending import GH_TIMEOUT_SECONDS, _gh_api
from ashiato.topics import _idf

_GH_TIMEOUT_MESSAGE = f"gh api timed out after {GH_TIMEOUT_SECONDS}s"
_FETCH_BATCH = 1000
_SHELL_PROMPT_RE = re.compile(
    r"^(?:PS [A-Za-z]:\\[^>\r\n]*> ?|PS /[^>\r\n]*> ?|"
    # The path holds no whitespace and the prompt char ends the prompt, so
    # prose such as ``alice@example.com: see issue #3`` is not a prompt.
    r"[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+:[^\s$#]*[$#](?: |(?=\r?\n|$))|\$ )"
)
_GITHUB_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9.-])(?:https?://)?(?:www\.)?github\.com/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)/"
    r"(?P<kind>pull|issues)/(?P<number>\d+)"
)
_DRIVE_RE = re.compile(
    r"(?<![A-Za-z0-9.-])https?://docs\.google\.com/(?P<kind>document|spreadsheets|presentation)/"
    r"(?:u/\d+/)?d/(?P<id>[A-Za-z0-9_-]+)",
    re.IGNORECASE,
)
_DRIVE_KINDS = {
    "document": "drive_document",
    "spreadsheets": "drive_spreadsheet",
    "presentation": "drive_presentation",
}
_GH_KINDS = {"pr": "pull", "issue": "issues"}


def local_zone() -> ZoneInfo | Any:
    """Return the best available local time zone."""
    env_zone = os.environ.get("TZ")
    if env_zone:
        try:
            return ZoneInfo(env_zone)
        except ZoneInfoNotFoundError:
            pass
    try:
        target = os.path.realpath("/etc/localtime")
        marker = f"{os.sep}zoneinfo{os.sep}"
        if marker in target:
            return ZoneInfo(target.split(marker, 1)[1])
    except (OSError, ZoneInfoNotFoundError):
        pass
    return datetime.now().astimezone().tzinfo or UTC


def parse_zone(value: str) -> ZoneInfo:
    try:
        return ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ValueError(f"invalid time zone: {value}") from error


def _zone_name(tz: Any) -> str:
    return getattr(tz, "key", None) or str(tz)


def _utc_naive(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is not None:
        return value.astimezone(UTC).replace(tzinfo=None)
    return value


def _in_window(value: datetime, since: datetime | None, until: datetime | None) -> bool:
    return (since is None or value >= since) and (until is None or value <= until)


def _fetchmany(cursor: Any) -> Iterator[tuple[Any, ...]]:
    while rows := cursor.fetchmany(_FETCH_BATCH):
        yield from rows


def human_text_rows(connection: duckdb.DuckDBPyConnection) -> Iterator[dict[str, Any]]:
    """Yield reusable human-authored text rows after the standard harness filters."""
    cursor = connection.execute("""
        SELECT s.file_path, s.session_id, s.source, s.started_at, s.entrypoint,
               e.ts, e.text, e.raw
        FROM events e
        JOIN sessions s ON e.file_path = s.file_path
        WHERE e.role = 'user'
          AND NOT COALESCE(e.is_meta, FALSE)
          AND NOT COALESCE(e.is_sidechain, FALSE)
          AND NOT (e.raw LIKE '%"tool_result"%')
        ORDER BY s.file_path, e.seq
    """)
    for file_path, session_id, source, started, entrypoint, ts, text, raw in _fetchmany(cursor):
        if is_headless(entrypoint) or is_compaction_summary(raw):
            continue
        prose = strip_harness(text or "").strip()
        when = _utc_naive(ts) or _utc_naive(started)
        if not prose or when is None:
            continue
        yield {
            "file_path": file_path,
            "session_id": session_id,
            "source": source,
            "ts": when,
            "text": prose,
        }


def strip_pasted(text: str) -> str:
    """Remove pasted code fences and shell prompts for theme extraction only."""
    kept: list[str] = []
    fence: str | None = None
    fence_mark = chr(96) * 3
    for line in text.splitlines(keepends=True):
        marker_text = line.lstrip()
        if fence is not None:
            if marker_text.startswith(fence):
                fence = None
            continue
        if marker_text.startswith(fence_mark):
            fence = fence_mark
            continue
        if marker_text.startswith("~~~"):
            fence = "~~~"
            continue
        kept.append(_SHELL_PROMPT_RE.sub("", line, count=1))
    return "".join(kept)


def _bucket_start(when: datetime, bucket: str, tz: Any) -> datetime:
    local = when.replace(tzinfo=UTC).astimezone(tz)
    day = local.date()
    if bucket == "month":
        start_day = date(day.year, day.month, 1)
    elif bucket == "week":
        start_day = day - timedelta(days=day.weekday())
    else:
        raise ValueError("bucket must be 'week' or 'month'")
    return datetime.combine(start_day, time.min, tzinfo=tz)


def _next_bucket(start: datetime, bucket: str) -> datetime:
    day = start.date()
    if bucket == "week":
        return start + timedelta(days=7)
    year, month = day.year, day.month + 1
    if month == 13:
        year, month = year + 1, 1
    return datetime(year, month, 1, tzinfo=start.tzinfo)


def _bucket_label(start: datetime, bucket: str) -> str:
    return start.strftime("%Y-%m" if bucket == "month" else "%Y-%m-%d")


def _parse_created(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _utc_naive(parsed)


def _extract_outputs(
    connection: duckdb.DuckDBPyConnection,
    owners: Sequence[str] | None,
    since: datetime | None,
    until: datetime | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[str], bool]:
    cursor = connection.execute("""
        SELECT tool_use_id, session_id, file_path, source, ts, result_text
        FROM tool_calls
        WHERE result_text IS NOT NULL
          AND (contains(lower(result_text), 'github.com/')
               OR contains(lower(result_text), 'docs.google.com/'))
        ORDER BY ts NULLS LAST, tool_use_id NULLS FIRST
    """)
    dated: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    undated_calls: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for tool_use_id, session_id, file_path, source, ts, result_text in _fetchmany(cursor):
        call = {
            "tool_use_id": tool_use_id or "",
            "session_id": session_id,
            "file_path": file_path,
            "source": source,
            "ts": _utc_naive(ts),
        }
        for match in _GITHUB_RE.finditer(result_text):
            owner, repo = match.group("owner"), match.group("repo")
            number = int(match.group("number"))
            key = ("github", owner.lower(), repo.lower(), number)
            entry = {
                **call,
                "owner": owner.lower(),
                "repo_name": repo.lower(),
                "number": number,
                "is_pr_ref": match.group("kind").lower() == "pull",
            }
            (undated_calls if ts is None else dated)[key].append(entry)
        for match in _DRIVE_RE.finditer(result_text):
            key = ("drive", match.group("id"))
            entry = {**call, "drive_type": match.group("kind").lower(), "doc_id": match.group("id")}
            (undated_calls if ts is None else dated)[key].append(entry)

    inferred = owners is None
    if inferred:
        distinct_keys: set[tuple[str, str, int]] = set()
        for key in (*dated.keys(), *undated_calls.keys()):
            if key[0] == "github":
                distinct_keys.add((key[1], key[2], key[3]))
        counts = Counter(owner for owner, _repo, _number in distinct_keys)
        selected_owners = [min(counts, key=lambda owner: (-counts[owner], owner))] if counts else []
    else:
        selected_owners = sorted({owner.lower() for owner in owners})

    allowed = set(selected_owners)
    outputs: list[dict[str, Any]] = []
    references_by_source: Counter[str] = Counter()
    for key, appearances in dated.items():
        first = min(appearances, key=lambda x: (x["ts"], x["tool_use_id"]))
        if key[0] == "github" and key[1] not in allowed:
            if _in_window(first["ts"], since, until):
                references_by_source[first["source"] or "unknown"] += 1
            continue
        if key[0] == "github":
            is_pr = any(row["is_pr_ref"] for row in [*appearances, *undated_calls.get(key, [])])
            kind = "pr" if is_pr else "issue"
            owner, repo_name, number = key[1], key[2], key[3]
            output = {
                "url": f"https://github.com/{owner}/{repo_name}/{_GH_KINDS[kind]}/{number}",
                "kind": kind,
                "repo": f"{owner}/{repo_name}",
                "number": number,
                "doc_id": None,
                **_first_fields(first),
            }
        else:
            doc_id = key[1]
            drive_type = first["drive_type"]
            output = {
                "url": f"https://docs.google.com/{drive_type}/d/{doc_id}",
                "kind": _DRIVE_KINDS[drive_type],
                "repo": None,
                "number": None,
                "doc_id": doc_id,
                **_first_fields(first),
            }
        outputs.append(output)

    undated_sources: Counter[str] = Counter()
    for key, appearances in undated_calls.items():
        if key in dated:
            continue
        first = min(appearances, key=lambda x: x["tool_use_id"])
        source = first["source"] or "unknown"
        if key[0] == "github" and key[1] not in allowed:
            references_by_source[source] += 1
        else:
            undated_sources[source] += 1
    outputs.sort(key=lambda out: (out["first_seen"], out["url"], out["tool_use_id"]))
    for output in outputs:
        output.pop("tool_use_id", None)
    return outputs, {
        "count": sum(undated_sources.values()),
        "by_source": dict(sorted(undated_sources.items())),
    }, {
        "count": sum(references_by_source.values()),
        "by_source": dict(sorted(references_by_source.items())),
    }, selected_owners, inferred


def _first_fields(first: dict[str, Any]) -> dict[str, Any]:
    return {
        "first_seen": first["ts"],
        "session_id": first["session_id"],
        "file_path": first["file_path"],
        "source": first["source"],
        "tool_use_id": first["tool_use_id"],
    }


def build_report(
    connection: duckdb.DuckDBPyConnection,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    bucket: str = "month",
    tz: Any | None = None,
    gh: bool = False,
    themes: int = 10,
    owners: Sequence[str] | None = None,
    gh_call: Callable[[Sequence[str]], tuple[bool, Any, str]] | None = None,
) -> dict[str, Any]:
    """Derive the complete report without printing or parsing command-line arguments."""
    if bucket not in ("month", "week"):
        raise ValueError("bucket must be 'week' or 'month'")
    if themes < 0:
        raise ValueError("themes must be non-negative")
    tz = tz or local_zone()
    since, until = _utc_naive(since), _utc_naive(until)
    if since is not None and until is not None and since > until:
        raise ValueError("--since must not be after --until")

    sessions = connection.execute("""
        SELECT file_path, session_id, source, started_at, entrypoint
        FROM sessions
    """).fetchall()
    dated_items: list[datetime] = []
    grouped_sessions: dict[tuple[str, str], dict[str, Any]] = {}
    for file_path, session_id, source, started, entrypoint in sessions:
        source_name = source or "unknown"
        session_key = (source_name, session_id if session_id is not None else file_path)
        when = _utc_naive(started)
        current = grouped_sessions.get(session_key)
        if current is None:
            grouped_sessions[session_key] = {
                "file_path": file_path,
                "session_id": session_id,
                "source": source_name,
                "when": when,
                "non_headless": not is_headless(entrypoint),
            }
        else:
            if when is not None and (current["when"] is None or when < current["when"]):
                current["when"] = when
                current["file_path"] = file_path
                current["session_id"] = session_id
            current["non_headless"] = current["non_headless"] or not is_headless(entrypoint)
    session_rows = []
    for session in grouped_sessions.values():
        when = session["when"]
        if when is None or not _in_window(when, since, until):
            continue
        dated_items.append(when)
        session_rows.append(
            (
                session["file_path"],
                session["session_id"],
                session["source"],
                when,
                session["non_headless"],
            )
        )

    text_rows = [row for row in human_text_rows(connection) if _in_window(row["ts"], since, until)]
    idf = _idf(collect_sessions(connection)) if themes else {}
    dated_items.extend(row["ts"] for row in text_rows)
    outputs, undated, references, report_owners, owners_inferred = _extract_outputs(
        connection, owners, since, until
    )
    outputs = [item for item in outputs if _in_window(item["first_seen"], since, until)]
    dated_items.extend(item["first_seen"] for item in outputs)

    if gh:
        caller = gh_call or _gh_api
        for item in outputs:
            if item["repo"] is None:
                item["status"] = "unchecked"
                continue
            owner, repo_name = item["repo"].split("/", 1)
            try:
                ok, payload, error = caller(
                    ["gh", "api", f"repos/{owner}/{repo_name}/issues/{item['number']}"]
                )
            except subprocess.TimeoutExpired:
                ok, payload, error = False, None, _GH_TIMEOUT_MESSAGE
            created = (
                _parse_created(payload.get("created_at"))
                if ok and isinstance(payload, dict)
                else None
            )
            if not ok:
                item.update(status="unknown", error=error or "gh api failed")
            elif not isinstance(payload, dict):
                item.update(status="unknown", error="gh api returned a non-object payload")
            elif created is None:
                item.update(status="unknown", error="missing or invalid created_at")
            else:
                item["created_at"] = created
                item["author"] = (
                    (payload.get("user") or {}).get("login")
                    if isinstance(payload.get("user"), dict)
                    else None
                )
                item["kind"] = "pr" if "pull_request" in payload else "issue"
                owner, repo_name = item["repo"].split("/", 1)
                item["url"] = (
                    f"https://github.com/{owner}/{repo_name}/"
                    f"{_GH_KINDS[item['kind']]}/{item['number']}"
                )
                delta = item["first_seen"] - created
                if timedelta(minutes=-2) <= delta <= timedelta(minutes=10):
                    item["status"] = "confirmed"
                elif delta > timedelta(minutes=10):
                    item["status"] = "outside"
                else:
                    item.update(status="unknown", error="first seen before created_at")
    else:
        for item in outputs:
            item["status"] = "unchecked"

    bucket_starts = sorted({_bucket_start(when, bucket, tz) for when in dated_items})
    labels: list[datetime] = []
    if bucket_starts:
        start, last = bucket_starts[0], bucket_starts[-1]
        while start <= last:
            labels.append(start)
            start = _next_bucket(start, bucket)
    by_label = {
        _bucket_label(start, bucket): {
            "label": _bucket_label(start, bucket),
            "start": start.isoformat(),
            "sessions": {},
            "human_chars": 0,
            "themes": [],
            "outputs": [],
        }
        for start in labels
    }
    for _path, _sid, source, when, non_headless in session_rows:
        target = by_label[_bucket_label(_bucket_start(when, bucket, tz), bucket)]
        stat = target["sessions"].setdefault(source, {"sessions": 0, "non_headless": 0})
        stat["sessions"] += 1
        stat["non_headless"] += int(non_headless)
    theme_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in text_rows:
        target = by_label[_bucket_label(_bucket_start(row["ts"], bucket, tz), bucket)]
        target["human_chars"] += len(row["text"])
        if themes:
            theme_counts[target["label"]].update(tokenize(strip_pasted(row["text"])))
    for label, counts in theme_counts.items():
        ranked = sorted(
            ((term, count, count * idf[term]) for term, count in counts.items() if term in idf),
            key=lambda item: (-item[2], item[0]),
        )[:themes]
        by_label[label]["themes"] = [
            {"term": term, "count": count, "weight": round(weight, 3)}
            for term, count, weight in ranked
        ]
    for item in outputs:
        target = by_label[_bucket_label(_bucket_start(item["first_seen"], bucket, tz), bucket)]
        target["outputs"].append(item)
    for item in outputs:
        item["first_seen"] = item["first_seen"].isoformat()
        if isinstance(item.get("created_at"), datetime):
            item["created_at"] = item["created_at"].isoformat()
    return {
        "tz": _zone_name(tz),
        "bucket": bucket,
        "since": since.isoformat() if since else None,
        "until": until.isoformat() if until else None,
        "gh": gh,
        "owners": report_owners,
        "owners_inferred": owners_inferred,
        "references": references,
        "undated": undated,
        "buckets": list(by_label.values()),
    }


def render_json(report: dict[str, Any], out: Any) -> None:
    """Print a JSON report document."""
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str), file=out)


def render_text(report: dict[str, Any], out: Any) -> None:
    """Render the glanceable text report."""
    print(f"window: {report['since'] or 'unbounded'} .. {report['until'] or 'unbounded'}", file=out)
    print(
        f"bucket: {report['bucket']}  tz: {report['tz']}  --gh: {'yes' if report['gh'] else 'no'}",
        file=out,
    )
    undated = report["undated"]
    sources = (
        ", ".join(f"{source}={count}" for source, count in undated["by_source"].items()) or "none"
    )
    print(
        f"owners: {', '.join(report['owners']) or 'none'} "
        f"({'inferred' if report['owners_inferred'] else 'given'})",
        file=out,
    )
    references = report["references"]
    reference_sources = (
        ", ".join(
            f"{source}={count}" for source, count in references["by_source"].items()
        )
        or "none"
    )
    print(
        f"references: {references['count']} ({reference_sources})",
        file=out,
    )
    print(f"undated outputs: {undated['count']} ({sources})", file=out)
    for bucket in report["buckets"]:
        print(f"\n{bucket['label']}", file=out)
        session_text = (
            ", ".join(
                f"{source} {data['sessions']} (non-headless {data['non_headless']})"
                for source, data in sorted(bucket["sessions"].items())
            )
            or "none"
        )
        print(f"  sessions: {session_text}", file=out)
        print(f"  human characters: {bucket['human_chars']}", file=out)
        theme_terms = ", ".join(item["term"] for item in bucket["themes"])
        print(f"  themes: {theme_terms}", file=out)
        repos: dict[str, list[dict[str, Any]]] = defaultdict(list)
        drive = []
        for item in bucket["outputs"]:
            (repos[item["repo"]] if item["repo"] else drive).append(item)
        for repo_name, items in sorted(repos.items()):
            groups = []
            for kind in ("pr", "issue"):
                values = [x for x in items if x["kind"] == kind]
                if values:
                    nums = ", ".join(f"#{x['number']}" for x in values)
                    groups.append(f"{len(values)} {kind} ({nums})")
            statuses = ""
            if report["gh"]:
                counts: dict[str, int] = defaultdict(int)
                for item in items:
                    counts[item["status"]] += 1
                statuses = "; status " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            print(f"  {repo_name}: {', '.join(groups)}{statuses}", file=out)
        if drive:
            print(f"  Drive documents: {len(drive)}", file=out)


def run(
    db_path: Path,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    bucket: str = "month",
    tz: Any | None = None,
    gh: bool = False,
    json_output: bool = False,
    out: Any,
    err: Any,
    themes: int = 10,
    owners: Sequence[str] | None = None,
    gh_call: Callable[[Sequence[str]], tuple[bool, Any, str]] | None = None,
) -> int:
    """Open the database read-only, derive and render the review report."""
    if not db_path.exists():
        print(f"error: no database at {db_path} (run 'ashiato build' first)", file=err)
        return 1
    connection = connect(db_path, read_only=True)
    try:
        assert_readable(connection)
        report = build_report(
            connection,
            since=since,
            until=until,
            bucket=bucket,
            tz=tz,
            gh=gh,
            themes=themes,
            owners=owners,
            gh_call=gh_call,
        )
    except (SchemaOutOfDate, duckdb.Error, ValueError) as error:
        print(f"error: {error}", file=err)
        return 1
    finally:
        connection.close()
    (render_json if json_output else render_text)(report, out)
    return 0
