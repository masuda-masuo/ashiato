"""Open items left in compaction summaries, checked against GitHub (issue #52).

When a Claude Code session runs out of context it writes a compaction summary
with a numbered "Pending Tasks" section -- the agent's own list of what is
left, often with "Discussed-but-not-started (do NOT begin without user
confirmation)" items and issue numbers.  These lists are the most direct
record of work that may have been dropped, and checking one session's items
against GitHub by hand takes many calls.  ``ashiato pending`` does it
mechanically: it reads each session's latest compaction summary, extracts the
Pending Tasks section, resolves every issue/PR reference and shows only what
is still open or was never filed.

A compaction summary is a ``role = 'user'`` event whose ``raw`` JSON carries
``"isCompactSummary": true`` as a *top-level* key of the record -- exactly how
Claude Code writes it; a flag buried inside ``message`` is not a summary
(other rows' ``raw`` is not always valid JSON, so detection parses
defensively).  A section runs from the ``N. Pending Tasks:`` heading to the
next numbered heading.  Items are the section's ``-`` bullets at its minimum
indentation, with their continuation lines -- deeper-indented bullets, fenced
code blocks and plain wrapped lines alike -- markdown bold removed and
whitespace collapsed.
References resolve in this precedence: a GitHub URL
(``https://github.com/<o>/<r>/(issues|pull)/<n>``); an explicit ``<o>/<r>#<n>``
(the owner must start with a letter, so ``454/PR#466`` is not one); a short
form (``<name>#<n>``, ``<name> #<n>``, ``<name> PR #<n>``, ``<name> PR#<n>``,
``<name> issue #<n>``) where ``<name>`` is a known repository name; and a bare
``#<n>`` (including ``PR #<n>`` / ``Issue #<n>`` with no repo name before
them).  A bare reference resolves, in this order: to the ``(owner, repo)`` of
a same-number resolved reference in the same item -- only when every
same-number resolved reference in that item names the same repository
(a same-number resolution is not an inference: the ref is recorded as
non-bare, exactly like the explicit ref it matches, so it takes no ``via``
and a 404 on it stays ``unknown``); to the nearest preceding resolved
repository in the same item; else a bare whole word that is exactly a known
repository name (case-sensitive) in the same item (``via: "name"``); else
``--repo OWNER/NAME``, else it is reported as ``unresolved`` and never
checked.  A bare word that is
a known repository name (not part of a path, URL, ``owner/repo``,
identifier, or longer word such as ``shiori-demo``) sets the nearest
preceding repository for later bare refs in the same item, the same way a
resolved explicit/short ref does.  Bare references carry a ``via`` key
indicating how they were resolved: ``"nearest"`` (preceding resolved ref),
``"name"`` (preceding bare name mention), or ``"repo_flag"`` (``--repo``).
Non-bare refs omit ``via``.  Known
repository names come from the explicit references found anywhere in the
database's summaries, or with ``--gh`` from
``gh repo list <owner> --limit 200 --json name`` (one call per run, only the
owner name leaves the machine); ``--owner NAME`` sets the owner directly and
defaults to the owner named most often by those explicit references.

Checking is opt-in.  Without ``--gh`` every reference state is ``unchecked``
and nothing is spawned.  With ``--gh`` each unique ``(owner, repo, number)``
is looked up once through the ``gh`` CLI (``gh api repos/<o>/<r>/issues/<n>``,
read-only; a PR counts as ``merged`` when ``merged_at`` is set) and recorded
as ``open`` / ``closed`` / ``merged`` / ``unknown``; a failure carries its
error message and the remaining references are still checked.  A 404 on an
inferred (bare) reference is recorded as ``unresolved`` (the repo was guessed
and wrong, not a broken reference); an explicit ``owner/repo#n`` or URL that
404s stays ``unknown``.  Only owner, repo, number and (for the repository
listing) the owner name ever leave the machine -- never transcript text -- and
the call sits behind one injectable function so tests can replace it.

Report-only, mirroring ``ashiato.orphans`` and ``ashiato.nominate``: it never
writes to any file or store, only reads the already-built DuckDB.  No new
stored tables or views, so no ``FORMAT_VERSION`` bump.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from ashiato.build import SchemaOutOfDate, assert_readable, connect

# ---------------------------------------------------------------------------
# Text patterns
# ---------------------------------------------------------------------------

#: A numbered section heading on its own line: ``7. Pending Tasks:``.
_SECTION_HEADING_RE = re.compile(r"^\s*\d+\.\s+[A-Z][^:\n]{2,60}:\s*$")

#: The section this module reports; matched case-insensitively.
_PENDING_HEADING_RE = re.compile(r"^\s*\d+\.\s+pending tasks:\s*$", re.IGNORECASE)

#: A pending item is a ``-`` bullet at the section's minimum indentation.
_BULLET_RE = re.compile(r"^\s*-\s+")

#: An item is deferred when it names a discussed-but-not-started topic, work
#: that has not started, or something the agent must not begin on its own.
_DEFERRED_RE = re.compile(
    r"discussed[- ]but[- ]not[- ]started|not (yet )?started|deferred|future work|"
    r"do not (start|begin)",
    re.IGNORECASE,
)

#: Every reference form in one pass, matched in precedence order: a GitHub URL
#: (``https://github.com/<o>/<r>/(issues|pull)/<n>``); an explicit
#: ``<o>/<r>#<n>`` whose owner starts with a letter (GitHub login rules, so
#: ``454/PR#466`` is not one); a short form (``<name>#<n>``, ``<name> #<n>``,
#: ``<name> PR #<n>``, ``<name> PR#<n>``, ``<name> issue #<n>``, PR/issue
#: case-insensitive); and a bare ``#<n>`` (including ``PR #<n>`` / ``Issue #<n>``
#: with no repo name before them).
_REF_TOKEN_RE = re.compile(
    r"https://github\.com/(?P<url_owner>[A-Za-z0-9_.-]+)/(?P<url_repo>[A-Za-z0-9_.-]+)/(?:issues|pull)/(?P<url_num>\d+)"
    r"|(?<![A-Za-z0-9_.-])(?P<slash_owner>[A-Za-z][A-Za-z0-9_.-]*)/(?P<slash_repo>[A-Za-z0-9_.-]+)#(?P<slash_num>\d+)"
    r"|(?<![A-Za-z0-9_.-])(?P<short_name>[A-Za-z0-9_.-]+)(?:\s+(?:PR|issue)\s*)?\s*#(?P<short_num>\d+)"
    r"|(?<![A-Za-z0-9_.-])(?:(?:PR|issue)\s*)?#(?P<bare_num>\d+)",
    re.IGNORECASE,
)

#: A whole word that is exactly a known repository name (case-sensitive).
#: Used to detect bare name mentions that set the "nearest preceding repo"
#: for later bare refs in the same item.  Excludes: names inside URLs,
#: paths (owner/repo), identifiers with hyphens/underscores, and names
#: embedded in longer words.
_NAME_BARE_RE = re.compile(
    r"(?<![A-Za-z0-9_/.-])(?P<name>[A-Za-z0-9_.-]+)(?![A-Za-z0-9_./-])"
)

#: Item text is truncated at this many characters in the text report.
ITEM_TEXT_CHARS = 300

#: Item statuses, in report order.  ``unresolved`` is a *reference* state, not
#: an item status: an item whose only references can never be checked falls
#: into ``unchecked``.
STATUSES = ("open", "resolved", "unreferenced", "unknown", "unchecked")

#: The state of every reference before any opt-in checking.
UNCHECKED = "unchecked"


# ---------------------------------------------------------------------------
# Summary text: sections and items
# ---------------------------------------------------------------------------


def extract_section(summary_text: str) -> str | None:
    """The Pending Tasks section of a summary, or None when absent."""
    lines = summary_text.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if _PENDING_HEADING_RE.match(line)),
        None,
    )
    if start is None:
        return None
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if _SECTION_HEADING_RE.match(lines[index]):
            end = index
            break
    return "\n".join(lines[start + 1 : end])


def extract_items(section_text: str) -> list[str]:
    """The section's bullets with their continuation lines, cleaned.

    An item starts only at a ``-`` bullet at the section's minimum
    indentation.  Every other line -- deeper-indented bullets, fenced code
    blocks and plain wrapped lines -- is continuation text of the current
    item.
    """
    lines = section_text.splitlines()
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    min_indent = min(indents) if indents else 0
    bullets: list[list[str]] = []
    in_fence = False
    for line in lines:
        if not line.strip():
            continue
        if line.strip().startswith("```"):
            in_fence = not in_fence
            if bullets:
                bullets[-1].append(line)
            continue
        if in_fence:
            if bullets:
                bullets[-1].append(line)
            continue
        indent = len(line) - len(line.lstrip())
        if indent == min_indent and _BULLET_RE.match(line):
            bullets.append([_BULLET_RE.sub("", line)])
        elif bullets:
            bullets[-1].append(line)
    return [_clean_item(" ".join(part.strip() for part in parts)) for parts in bullets]


def _clean_item(text: str) -> str:
    """Markdown bold removed and whitespace collapsed."""
    return re.sub(r"\s+", " ", text.replace("**", "")).strip()


def is_deferred(text: str) -> bool:
    """True when *text* marks the item as deliberately not started yet."""
    return _DEFERRED_RE.search(text) is not None


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def explicit_refs(text: str) -> list[tuple[str, str, int]]:
    """``(owner, repo, number)`` for every explicit reference, in text order.

    Explicit means a full GitHub URL or an ``owner/repo#n`` whose owner starts
    with a letter (GitHub login rules); ``454/PR#466`` is therefore not one.
    """
    found: dict[tuple[str, str, int], int] = {}
    for match in _REF_TOKEN_RE.finditer(text):
        if match.group("url_num") is not None:
            key = (
                match.group("url_owner"),
                match.group("url_repo"),
                int(match.group("url_num")),
            )
        elif match.group("slash_num") is not None:
            key = (
                match.group("slash_owner"),
                match.group("slash_repo"),
                int(match.group("slash_num")),
            )
        else:
            continue
        found.setdefault(key, match.start())
    return [key for key, _ in sorted(found.items(), key=lambda item: item[1])]


def _scan_refs(
    text: str,
    known_repos: frozenset[str] | None,
    owner: str | None,
) -> list[tuple[str | None, str | None, int, bool]]:
    """Reference candidates in text order: ``(owner, repo, number, bare)``.

    ``bare=True`` means the number still needs an owner/repo: it came from the
    short form whose name is not a known repository, or from a bare ``#n``.
    Explicit refs and known short forms carry their ``(owner, repo)`` directly.
    """
    refs: list[tuple[str | None, str | None, int, bool]] = []
    for match in _REF_TOKEN_RE.finditer(text):
        if match.group("url_num") is not None:
            refs.append(
                (
                    match.group("url_owner"),
                    match.group("url_repo"),
                    int(match.group("url_num")),
                    False,
                )
            )
        elif match.group("slash_num") is not None:
            refs.append(
                (
                    match.group("slash_owner"),
                    match.group("slash_repo"),
                    int(match.group("slash_num")),
                    False,
                )
            )
        elif match.group("short_num") is not None:
            name = match.group("short_name")
            number = int(match.group("short_num"))
            if owner is not None and known_repos and name.lower() in known_repos:
                refs.append((owner, name, number, False))
            else:
                refs.append((None, None, number, True))
        else:
            refs.append((None, None, int(match.group("bare_num")), True))
    return refs


def bare_refs(text: str) -> list[int]:
    """Every bare ``#n`` in *text*, in text order.

    A bare reference is any ``#n`` not attached to an explicit ``owner/repo#n``
    or GitHub URL -- including ``PR #454`` / ``Issue #243`` and unknown short
    forms like ``of#3``.
    """
    return [
        number
        for _owner, _repo, number, bare in _scan_refs(text, None, None)
        if bare
    ]


# ---------------------------------------------------------------------------
# Report rows
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Reference:
    """One issue/PR reference and what checking found out about it.

    ``state`` is ``open`` / ``closed`` / ``merged`` / ``unknown`` after
    ``--gh``, ``unchecked`` without it, or ``unresolved`` for a bare ``#n``
    that no in-item repository and no ``--repo`` could resolve.  ``error``
    carries the gh failure message for ``unknown``.  ``via`` (bare refs only)
    indicates how the repo was resolved: ``"nearest"`` (preceding resolved
    ref), ``"name"`` (preceding bare name mention), ``"repo_flag"``
    (``--repo``), or ``None`` (unresolved).  A bare ``#n`` resolved by the
    same-number rule is recorded as non-bare (``via`` None), exactly like the
    explicit ref it matches -- the repo is not inferred.
    """

    owner: str | None
    repo: str | None
    number: int
    state: str
    bare: bool = False
    error: str | None = None
    via: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "owner": self.owner,
            "repo": self.repo,
            "number": self.number,
            "state": self.state,
            "bare": self.bare,
        }
        if self.error:
            payload["error"] = self.error
        if self.bare:
            payload["via"] = self.via
        return payload


@dataclass(slots=True)
class PendingItem:
    """One Pending Tasks bullet, with its references and derived status."""

    text: str
    deferred: bool
    references: list[Reference]

    @property
    def status(self) -> str:
        return item_status(self.references)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "deferred": self.deferred,
            "status": self.status,
            "references": [reference.to_dict() for reference in self.references],
        }


@dataclass(slots=True)
class SummaryReport:
    """One compaction summary and its items, as the report shows them."""

    session_id: str | None
    file_path: str
    ts: datetime | None
    project_dir: str | None
    title: str | None
    items: list[PendingItem]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "file_path": self.file_path,
            "ts": self.ts.isoformat() if self.ts else None,
            "project_dir": self.project_dir,
            "title": self.title,
            "items": [item.to_dict() for item in self.items],
        }


def item_status(references: Sequence[Reference]) -> str:
    """``open`` / ``resolved`` / ``unreferenced`` / ``unknown`` / ``unchecked``.

    One open reference opens the item; all references closed or merged resolve
    it; a failed lookup makes it ``unknown``; everything else (unchecked, or
    unresolved bare references) stays ``unchecked``.
    """
    if not references:
        return "unreferenced"
    if any(reference.state == "open" for reference in references):
        return "open"
    if all(reference.state in ("closed", "merged") for reference in references):
        return "resolved"
    if any(reference.state == "unknown" for reference in references):
        return "unknown"
    return "unchecked"


def count_statuses(items: Sequence[PendingItem]) -> dict[str, int]:
    """Counts of every item status, over *items* (resolved items included)."""
    counts = Counter(item.status for item in items)
    return {status: counts[status] for status in STATUSES}


# ---------------------------------------------------------------------------
# Database collection
# ---------------------------------------------------------------------------

# Join on file_path, never session_id: a session_id fans out across files.  The
# row prefilter keeps only title events and user rows that mention the
# compaction flag; whether a user row really is a compaction summary is decided
# by parsing its raw JSON in Python, because raw is not always valid JSON.
_SUMMARIES_SQL = """
    SELECT
        e.file_path,
        e.session_id,
        e.ts,
        e.type,
        e.role,
        e.raw,
        e.text,
        s.project_dir
    FROM events e
    LEFT JOIN sessions s ON e.file_path = s.file_path
    WHERE NOT COALESCE(e.is_meta, FALSE)
      AND NOT COALESCE(e.is_sidechain, FALSE)
      AND (e.type IN ('ai-title', 'custom-title')
           OR (e.role = 'user' AND e.raw LIKE '%isCompactSummary%'))
    ORDER BY e.file_path, e.ts NULLS LAST, e.seq
"""


@dataclass(slots=True)
class _Summary:
    """One compaction summary event, reduced to what the analysis needs."""

    file_path: str
    session_id: str | None
    ts: datetime | None
    text: str
    project_dir: str | None


def _is_compact_summary(raw: str | None) -> bool:
    """True when the record's *top-level* ``isCompactSummary`` is true.

    Claude Code writes the flag as a sibling of ``uuid``, never inside
    ``message``; a flag buried there is not a summary.  Never raises:
    unparseable or non-object ``raw`` is not a summary either.
    """
    if not raw or '"isCompactSummary"' not in raw:
        return False
    try:
        record = json.loads(raw)
    except ValueError:
        return False
    return isinstance(record, dict) and record.get("isCompactSummary") is True


def collect_summaries(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[list[_Summary], dict[str, str]]:
    """Every compaction summary, plus the latest ai-title/custom-title per file.

    Returns ``(summaries, titles)`` where *titles* maps ``file_path`` to the
    latest title event's text.  One ordered scan serves both, so the title and
    the summaries share the same session join.
    """
    summaries: list[_Summary] = []
    titles: dict[str, str] = {}
    for file_path, session_id, ts, etype, role, raw, text, project_dir in (
        connection.execute(_SUMMARIES_SQL).fetchall()
    ):
        if etype in ("ai-title", "custom-title"):
            if text:
                # Rows arrive in ts/seq order, so the last one seen is latest.
                titles[file_path] = text
        elif role == "user" and _is_compact_summary(raw):
            summaries.append(_Summary(file_path, session_id, ts, text or "", project_dir))
    return summaries, titles


def _selected_summaries(
    summaries: Sequence[_Summary], *, all_summaries: bool
) -> list[_Summary]:
    """The summaries to analyse: latest per session, or every one."""
    if not all_summaries:
        latest: dict[str, _Summary] = {}
        for summary in summaries:  # ordered by file_path, ts, seq
            latest[summary.file_path] = summary
        return list(latest.values())
    return list(summaries)


def _in_window(
    ts: datetime | None, since: datetime | None, until: datetime | None
) -> bool:
    if ts is None:
        return since is None and until is None
    if since is not None and ts < since:
        return False
    return not (until is not None and ts > until)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _analyze_summary(
    summary: _Summary,
    repo: tuple[str, str] | None,
    *,
    owner: str | None,
    known_repos: frozenset[str] | None,
    known_names: frozenset[str] | None = None,
) -> list[PendingItem] | None:
    """The items of one summary, or None when it has no Pending Tasks section.

    References resolve per item, in text order: explicit refs (GitHub URLs and
    ``owner/repo#n`` with a letter-starting owner) and known short forms
    resolve directly; a bare ``#n`` (including ``PR #<n>`` / ``Issue #<n>`` and
    short forms whose name is not a known repository) first takes the
    ``(owner, repo)`` of a same-number resolved reference in the same item
    when every such reference names the same repository.  That is not an
    inference -- the ref is recorded as non-bare with no ``via``, exactly
    like the explicit ref it matches, so a 404 on it stays ``unknown`` --
    else it resolves to the nearest
    preceding entry in the same item -- a resolved ref or a bare name mention,
    whichever is closest in text order -- else ``--repo``, else it is reported
    as ``unresolved`` and never checked.  A bare name mention is a whole word
    exactly equal to a known repository name (case-sensitive).  Bare refs
    resolved via the
    closest preceding resolved ref carry ``via: "nearest"``;
    via the closest preceding name mention ``via: "name"``; via ``--repo``
    ``via: "repo_flag"``.
    """
    section = extract_section(summary.text)
    if section is None:
        return None
    texts = extract_items(section)

    items: list[PendingItem] = []
    for text in texts:
        references: list[Reference] = []
        seen: set[tuple[str | None, str | None, int]] = set()

        # The nearest-preceding-repository timeline of this item: every
        # resolved (non-bare) ref and every bare name mention, in text order.
        # A bare ``#n`` takes the most recent entry before it, of either kind
        # -- ``via: "nearest"`` when the closest entry is a resolved ref,
        # ``via: "name"`` when it is a name mention.  A bare ref resolved this
        # way never becomes an entry itself (only resolved refs and mentions
        # do), so a bare ref between a mention and a later bare ref does not
        # hide the mention.
        last_resolved: tuple[str, str] | None = None  # nearest preceding resolved ref
        last_resolved_pos = -1
        mentions: list[tuple[str, int]] = []  # (name, position), in text order
        if known_names is not None:
            for m in _NAME_BARE_RE.finditer(text):
                if m.group("name") in known_names:
                    mentions.append((m.group("name"), m.start()))
        mention_index = 0  # first mention not yet before the current ref

        # Rule 1 needs the whole item's resolved references before any bare
        # ref is resolved: one pass collects, per number, every (owner, repo)
        # named by a non-bare reference anywhere in the item.  A bare ``#n``
        # takes that (owner, repo) when every same-number resolved reference
        # in the item names the same one -- regardless of whether the bare
        # ref comes before or after them.
        same_number: dict[int, set[tuple[str, str]]] = {}
        for ref_owner, ref_repo, number, is_bare in _scan_refs(text, known_repos, owner):
            # A non-bare ref always carries its (owner, repo) -- the None
            # check narrows the scan's static type, nothing more.
            if not is_bare and ref_owner is not None and ref_repo is not None:
                same_number.setdefault(number, set()).add((ref_owner, ref_repo))

        ref_positions = [m.start() for m in _REF_TOKEN_RE.finditer(text)]
        for (ref_owner, ref_repo, number, bare), ref_pos in zip(
            _scan_refs(text, known_repos, owner), ref_positions, strict=True
        ):
            # The most recent bare name mention strictly before this ref.
            while mention_index < len(mentions) and mentions[mention_index][1] < ref_pos:
                mention_index += 1
            prev_mention: tuple[str, int] | None = (
                mentions[mention_index - 1] if mention_index > 0 else None
            )
            via: str | None = None
            if bare:
                # Rule 1: every same-number resolved reference in the item
                # names the same repository -- take it, whatever the bare ref's
                # own position.  Same-number refs naming different repos fall
                # through to the nearest preceding entry below.
                same = same_number.get(number)
                if same is not None and len(same) == 1:
                    (ref_owner, ref_repo), = same
                    # Not an inference: the repo comes from an explicit
                    # reference in the same item, so the ref is as good as a
                    # non-bare one -- it is not bare and takes no via.  A 404
                    # on it is a genuinely missing number (``unknown``), never
                    # the "inferred repo" demotion, whatever the text order.
                    bare = False
                elif last_resolved is not None and (
                    prev_mention is None or last_resolved_pos >= prev_mention[1]
                ):
                    ref_owner, ref_repo = last_resolved
                    via = "nearest"
                elif prev_mention is not None:
                    ref_owner = owner
                    ref_repo = prev_mention[0]
                    via = "name"
                elif repo is not None:
                    ref_owner, ref_repo = repo
                    via = "repo_flag"
                else:
                    ref_owner, ref_repo = None, None
            if ref_owner is None or ref_repo is None:
                key = (None, None, number)
                if key not in seen:
                    seen.add(key)
                    references.append(
                        Reference(None, None, number, "unresolved", bare=True, via=via)
                    )
                continue
            key = (ref_owner, ref_repo, number)
            if key not in seen:
                seen.add(key)
                references.append(
                    Reference(ref_owner, ref_repo, number, UNCHECKED, bare=bare, via=via)
                )
            # Only resolved refs (non-bare) join the timeline; bare refs --
            # even ones resolved from a name mention -- do not.
            if not bare:
                last_resolved = (ref_owner, ref_repo)
                last_resolved_pos = ref_pos
        items.append(PendingItem(text=text, deferred=is_deferred(text), references=references))
    return items


# ---------------------------------------------------------------------------
# Opt-in gh checking
# ---------------------------------------------------------------------------

#: ``gh api`` is cut off after this many seconds; a timeout is a failure like
#: any other, reported as ``unknown`` with a message.
GH_TIMEOUT_SECONDS = 30

#: The failure message for a gh call that exceeded the timeout.
_GH_TIMEOUT_MESSAGE = f"gh api timed out after {GH_TIMEOUT_SECONDS}s"


def _state_from_payload(ok: bool, payload: Any, error: str) -> tuple[str, str | None]:
    """``(state, error message)`` for one ``gh api`` result."""
    if not ok:
        return "unknown", error or "gh api failed"
    if not isinstance(payload, dict):
        return "unknown", "gh api returned a non-object payload"
    pull_request = payload.get("pull_request")
    if isinstance(pull_request, dict) and pull_request.get("merged_at"):
        return "merged", None
    state = payload.get("state")
    if state == "open":
        return "open", None
    if state == "closed":
        return "closed", None
    return "unknown", f"unexpected issue state: {state!r}"


def check_references(
    items: Sequence[PendingItem],
    gh_call: Callable[[Sequence[str]], tuple[bool, Any, str]],
) -> None:
    """Look up every checkable reference once per unique (owner, repo, number).

    *gh_call* receives ``[\"gh\", \"api\", \"repos/<o>/<r>/issues/<n>\"]`` -- the
    only strings that ever leave the machine -- and returns
    ``(ok, payload, error)``.  Unresolved references are never passed to it.
    (The one ``gh repo list`` call of the run goes through the same callable;
    see :func:`run`.)
    """
    unique: dict[tuple[str, str, int], list[Reference]] = {}
    for item in items:
        for reference in item.references:
            if reference.owner is None or reference.repo is None:
                continue
            key = (reference.owner, reference.repo, reference.number)
            unique.setdefault(key, []).append(reference)
    for (owner, repo, number), references in unique.items():
        try:
            ok, payload, error = gh_call(
                ["gh", "api", f"repos/{owner}/{repo}/issues/{number}"]
            )
        except subprocess.TimeoutExpired:
            # A fake gh_call may raise where the real one returns; a timeout
            # is still a failure with a message.
            ok, payload, error = False, None, _GH_TIMEOUT_MESSAGE
        state, message = _state_from_payload(ok, payload, error)
        for reference in references:
            # A 404 on an inferred (bare) repo is "unresolved" -- the repo
            # was guessed and the guess was wrong, not a broken reference.
            if (
                reference.bare
                and state == "unknown"
                and message is not None
                and "HTTP 404" in message
            ):
                reference.state = "unresolved"
                reference.error = f"inferred repo {owner}/{repo}: {message}"
            else:
                reference.state = state
                if message:
                    reference.error = message


def _gh_api(argv: Sequence[str]) -> tuple[bool, Any, str]:
    """The real gh call: run *argv* as a subprocess (argv list, no shell)."""
    try:
        result = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            check=False,
            timeout=GH_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        return False, None, "gh not found on PATH"
    except subprocess.TimeoutExpired:
        return False, None, _GH_TIMEOUT_MESSAGE
    if result.returncode != 0:
        message = result.stderr.strip() or f"gh exited with status {result.returncode}"
        return False, None, message
    try:
        return True, json.loads(result.stdout), ""
    except ValueError:
        return False, None, "gh api returned non-JSON output"


def _repo_list_call(
    owner: str, gh_call: Callable[[Sequence[str]], tuple[bool, Any, str]] | None
) -> tuple[bool, Any, str]:
    """One ``gh repo list`` call; only the owner name leaves the machine."""
    argv = ["gh", "repo", "list", owner, "--limit", "200", "--json", "name"]
    if gh_call is not None:
        return gh_call(argv)
    return _gh_api(argv)


def _repo_names(payload: Any) -> set[str] | None:
    """The repository names from ``gh repo list --json name`` output.

    gh prints a JSON array of ``{"name": ...}`` objects; None means the payload
    was not of that shape (caller keeps the DB-derived names then).
    """
    if not isinstance(payload, list):
        return None
    names = {
        item["name"]
        for item in payload
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    return names if names else None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _visible_items(report: SummaryReport, show_resolved: bool) -> list[PendingItem]:
    """The report's items: everything, or everything but resolved ones."""
    if show_resolved:
        return list(report.items)
    return [item for item in report.items if item.status != "resolved"]


def _reference_text(reference: Reference) -> str:
    if reference.owner is None or reference.repo is None:
        return f"#{reference.number}=unresolved"
    label = f"{reference.owner}/{reference.repo}#{reference.number}"
    if reference.state == "unknown" and reference.error:
        return f"{label}=unknown ({reference.error[:60]})"
    return f"{label}={reference.state}"


def _counts_text(counts: dict[str, int]) -> str:
    statuses = ", ".join(f"{status} {counts[status]}" for status in STATUSES)
    total = sum(counts[status] for status in STATUSES)
    label = "item" if total == 1 else "items"
    return f"({total} {label}: {statuses}; no_section: {counts['no_section']})"


def _render_text(
    reports: Sequence[SummaryReport],
    counts: dict[str, int],
    *,
    show_resolved: bool,
    out: Any,
) -> None:
    for report in reports:
        items = _visible_items(report, show_resolved)
        if not items:
            continue
        ts_text = report.ts.isoformat(sep=" ") if report.ts else "?"
        title = f"  title: {report.title}" if report.title else ""
        print(
            f"session {report.session_id or '?'}  summary {ts_text}  "
            f"{report.project_dir or '?'}{title}",
            file=out,
        )
        for item in items:
            refs = ", ".join(_reference_text(reference) for reference in item.references)
            refs = f"  refs: {refs}" if refs else ""
            deferred = " [deferred]" if item.deferred else ""
            print(
                f"  [{item.status}]{deferred}{refs}  {item.text[:ITEM_TEXT_CHARS]}",
                file=out,
            )
        print("", file=out)
    print(_counts_text(counts), file=out)


# ---------------------------------------------------------------------------
# Public API: run()
# ---------------------------------------------------------------------------


def run(
    db_path: Path,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    repo: tuple[str, str] | None = None,
    owner: str | None = None,
    all_summaries: bool = False,
    use_gh: bool = False,
    gh_call: Callable[[Sequence[str]], tuple[bool, Any, str]] | None = None,
    show_resolved: bool = False,
    json_output: bool = False,
    out: Any = None,
    err: Any = None,
) -> int:
    """Render the pending-items report.  Returns 0, or 1 if the db is unreadable.

    *owner* is the owner short forms resolve to; it defaults to the most
    frequent owner among the explicit references across all summaries in the
    DB, and short forms stay unresolved when it is None.  *gh_call* replaces
    the real ``gh`` subprocesses for tests (both the one ``gh repo list`` call
    and every ``gh api`` lookup); it is only consulted when *use_gh* is true.
    """
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
        summaries, titles = collect_summaries(connection)
    except SchemaOutOfDate as error:
        print(f"error: {error}", file=err)
        return 1
    except duckdb.Error as error:
        print(f"error: {error}", file=err)
        return 1
    finally:
        connection.close()

    # The owner and the known repository names are DB-wide facts, computed
    # from the explicit references found anywhere in the DB's summaries: the
    # owner defaults to the most frequent one (--owner overrides), and the
    # known repo names are those explicit repos, or with --gh the owner's
    # repositories from one `gh repo list <owner> --limit 200 --json name`
    # call (only the owner name leaves the machine).
    explicit: Counter[tuple[str, str]] = Counter()
    for summary in summaries:
        for ref_owner, ref_repo, _number in explicit_refs(summary.text):
            explicit[(ref_owner, ref_repo)] += 1
    owners = Counter(owner for owner, _repo in explicit)
    resolved_owner = owner if owner is not None else (
        max(owners, key=lambda key: (owners[key], key)) if owners else None
    )
    known_repos: set[str] = {repo_name for _owner, repo_name in explicit}
    if use_gh and resolved_owner is not None:
        try:
            ok, payload, _error = _repo_list_call(resolved_owner, gh_call)
        except subprocess.TimeoutExpired:
            ok, payload = False, None
        if ok:
            listed = _repo_names(payload)
            if listed is not None:
                known_repos = listed
    known = frozenset(name.lower() for name in known_repos)
    known_names = frozenset(known_repos)  # original case for bare name mentions

    selected = _selected_summaries(summaries, all_summaries=all_summaries)
    if since is not None or until is not None:
        selected = [summary for summary in selected if _in_window(summary.ts, since, until)]

    checker = gh_call if use_gh else None
    if use_gh and checker is None:
        checker = _gh_api

    reports: list[SummaryReport] = []
    all_items: list[PendingItem] = []
    no_section = 0
    for summary in selected:
        items = _analyze_summary(
            summary, repo, owner=resolved_owner, known_repos=known, known_names=known_names
        )
        if items is None:
            no_section += 1
            continue
        all_items.extend(items)
        reports.append(
            SummaryReport(
                session_id=summary.session_id,
                file_path=summary.file_path,
                ts=summary.ts,
                project_dir=summary.project_dir,
                title=titles.get(summary.file_path),
                items=items,
            )
        )
    # One lookup per unique (owner, repo, number) across the whole run, so a
    # reference shared by several sessions is still checked only once.
    if checker is not None:
        check_references(all_items, checker)

    reports.sort(
        key=lambda report: (report.ts or datetime.min, report.file_path), reverse=True
    )
    counts = count_statuses(all_items)
    counts["no_section"] = no_section

    if json_output:
        sessions = [
            {
                **report.to_dict(),
                "items": [item.to_dict() for item in _visible_items(report, show_resolved)],
            }
            for report in reports
        ]
        payload = {"sessions": sessions, "counts": counts}
        print(json.dumps(payload, indent=2, ensure_ascii=False), file=out)
        return 0

    _render_text(reports, counts, show_resolved=show_resolved, out=out)
    return 0