"""Tests for ashiato.serve: the read-only local dashboard (issue #49).

The fixture database is built from hand-written JSONL transcripts through the
real build pipeline, in a private ``tmp_path`` -- never in ``tests/fixtures/``,
which other tests glob recursively.  Requests go over a real loopback socket to
a real server on an ephemeral port.  The lock tests hold the database from a
*separate process*, because DuckDB only refuses a second writer across
processes.

Fixture, by design (all values asserted below are derived from it):

* ``sess-alpha`` -- a long human turn opening with ``<script>alert(1)</script>``
  and repeating three one-off words (``zorblax``, ``quuxify``,
  ``flumbergast``): the only orphan candidate.  Also one denied ``Bash`` call
  followed by a ``Read``, and memory writes: ``a.md`` created by opus and
  edited by sonnet, and ``d.md`` created by opus but absent from disk.
* ``sess-beta`` -- shared filler only, a ``Write`` of ``b.md`` by sonnet, and a
  ``Bash`` call that never got a result (``pending``).
* ``sess-gamma`` -- one short line.
* ``c.md`` sits in the memory directory with no recorded write: unattributed.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import subprocess
import sys
import threading
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from ashiato import serve
from ashiato.build import build
from ashiato.cli import main

PAD = "we were talking through the general shape of things together and going back and forth. "
LONG = PAD * 12  # comfortably over the 800-character discussion threshold

OPUS = "claude-opus-5"
SONNET = "claude-sonnet-5"
SCRIPT = "<script>alert(1)</script>"
ESCAPED_SCRIPT = "&lt;script&gt;alert(1)&lt;/script&gt;"


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty HOME, so the default sinks and memory dirs never read the real ~/.claude."""
    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HOME", str(path))
    return path


# ---------------------------------------------------------------- fixture database


def _line(
    role: str,
    uuid: str,
    blocks: list[dict[str, Any]],
    *,
    session: str,
    ts: str,
    model: str | None = None,
) -> str:
    record: dict[str, Any] = {
        "type": role,
        "uuid": uuid,
        "parentUuid": None,
        "sessionId": session,
        "timestamp": ts,
        "cwd": f"/work/{session}",
        "message": {"role": role, "content": blocks},
    }
    if model is not None:
        record["message"]["model"] = model
    return json.dumps(record)


def _transcript(session: str, day: str, steps: list[tuple[Any, ...]]) -> str:
    """One transcript.  A step is ``("user", text)``, ``("assistant", text, model)`` or
    ``("tool", name, input, result, model)`` -- a ``None`` result is a call with no reply."""
    lines: list[str] = []
    for index, step in enumerate(steps):
        ts = f"{day}T10:{index // 60:02d}:{index % 60:02d}Z"
        uid = f"{session}-{index}"
        kind = step[0]
        if kind == "user":
            block = {"type": "text", "text": step[1]}
            lines.append(_line("user", uid, [block], session=session, ts=ts))
        elif kind == "assistant":
            block = {"type": "text", "text": step[1]}
            lines.append(_line("assistant", uid, [block], session=session, ts=ts, model=step[2]))
        else:
            _, name, tool_input, result, model = step
            use = {"type": "tool_use", "id": f"{uid}-use", "name": name, "input": tool_input}
            lines.append(_line("assistant", uid, [use], session=session, ts=ts, model=model))
            if result is not None:
                reply = {
                    "type": "tool_result",
                    "tool_use_id": f"{uid}-use",
                    "content": [{"type": "text", "text": result}],
                    "is_error": False,
                }
                lines.append(_line("user", f"{uid}-r", [reply], session=session, ts=ts))
    return "\n".join(lines) + "\n"


class Fixture(NamedTuple):
    db: Path
    transcripts: Path
    memory_dir: Path


@pytest.fixture
def fixture(tmp_path: Path) -> Fixture:
    memory_dir = tmp_path / "proj" / ".claude" / "projects" / "p" / "memory"
    memory_dir.mkdir(parents=True)
    for name in ("a.md", "b.md", "c.md"):
        (memory_dir / name).write_text(f"# {name}\n", encoding="utf-8")
    a, b, d = (str(memory_dir / name) for name in ("a.md", "b.md", "d.md"))
    denied = "The user doesn't want to proceed with this tool use: not allowed"

    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    sessions = {
        "sess-alpha": (
            "2026-03-01",
            [
                ("user", f"{SCRIPT} " + "zorblax quuxify flumbergast " * 4 + LONG),
                ("assistant", "Noted, zorblax it is.", OPUS),
                ("tool", "Bash", {"command": "rm -rf /tmp/x"}, denied, OPUS),
                ("tool", "Read", {"file_path": "/tmp/y"}, "contents", OPUS),
                ("tool", "Write", {"file_path": a, "content": "x"}, "ok", OPUS),
                ("tool", "Edit", {"file_path": a, "old_string": "x", "new_string": "y"}, "ok", SONNET),
                ("tool", "Write", {"file_path": d, "content": "x"}, "ok", OPUS),
            ],
        ),
        "sess-beta": (
            "2026-03-02",
            [
                ("user", LONG),
                ("assistant", "Understood.", SONNET),
                ("tool", "Write", {"file_path": b, "content": "x"}, "ok", SONNET),
                ("tool", "Bash", {"command": "sleep 1"}, None, SONNET),
            ],
        ),
        "sess-gamma": ("2026-03-03", [("user", "hello there"), ("assistant", "hi", OPUS)]),
    }
    for name, (day, steps) in sessions.items():
        (transcripts / f"{name}.jsonl").write_text(_transcript(name, day, steps), encoding="utf-8")
    db = tmp_path / "serve.duckdb"
    build([transcripts], db)
    return Fixture(db, transcripts, memory_dir)


# ---------------------------------------------------------------- a running server


class Reply(NamedTuple):
    status: int
    headers: dict[str, str]
    body: str

    def json(self) -> Any:
        return json.loads(self.body)


class Served:
    def __init__(self, server: Any) -> None:
        self.server = server
        self.port = server.server_address[1]

    def get(self, path: str, headers: dict[str, str] | None = None, method: str = "GET") -> Reply:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            connection.request(method, path, headers=headers or {})
            response = connection.getresponse()
            body = response.read().decode("utf-8")
            return Reply(response.status, {k.lower(): v for k, v in response.getheaders()}, body)
        finally:
            connection.close()


@pytest.fixture
def served(fixture: Fixture) -> Iterator[Served]:
    server = serve.make_server(
        fixture.db, port=0, default_sinks=False, memory_dirs=[fixture.memory_dir]
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield Served(server)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(10)


def _cli_json(*args: str, capsys: pytest.CaptureFixture[str]) -> Any:
    code = main(list(args))
    captured = capsys.readouterr()
    assert code == 0, captured.err
    return json.loads(captured.out)


def _tile(body: str, key: str) -> int:
    match = re.search(rf'data-tile="{key}".*?class="value">(\d+)<', body, re.DOTALL)
    assert match, f"no {key} tile"
    return int(match.group(1))


def _table(body: str, table_id: str) -> str:
    match = re.search(rf'<table id="{table_id}">.*?</table>', body, re.DOTALL)
    assert match, f"no table {table_id}"
    return match.group(0)


# ---------------------------------------------------------------- pages


def test_overview_shows_counts_freshness_and_tiles(served: Served) -> None:
    reply = served.get("/")
    assert reply.status == 200
    assert reply.headers["content-type"] == "text/html; charset=utf-8"
    assert _tile(reply.body, "orphans") == 1
    assert _tile(reply.body, "memory") == 1
    assert _tile(reply.body, "denials") == 1
    assert 'data-freshness="current"' in reply.body
    counts = _table(reply.body, "t-counts")
    assert re.search(r"<td class=\"m\">sessions</td><td class=\"n\">3</td>", counts)
    assert re.search(r"<td class=\"m\">tool_calls</td><td class=\"n\">7</td>", counts)
    assert "2026-03-01" in reply.body  # the time window
    assert 'href="/session/sess-alpha"' in reply.body  # the densest candidate


def test_overview_marks_a_stale_database_and_reads_freshness_live(
    served: Served, fixture: Fixture
) -> None:
    assert 'data-freshness="current"' in served.get("/").body
    extra = _transcript("sess-new", "2026-03-04", [("user", "a brand new session")])
    (fixture.transcripts / "sess-new.jsonl").write_text(extra, encoding="utf-8")
    body = served.get("/").body
    assert 'data-freshness="stale"' in body
    assert "1 new or changed file under recorded roots" in body
    assert served.get("/api/overview.json").json()["freshness"] == {"state": "stale", "gap": 1}


def test_orphans_page_lists_the_candidate(served: Served) -> None:
    reply = served.get("/orphans")
    assert reply.status == 200
    table = _table(reply.body, "t-orphans")
    assert table.count("<tr>") == 2  # header + the one candidate
    assert 'href="/session/sess-alpha"' in table
    for term in ("zorblax", "quuxify", "flumbergast"):
        assert f'<span class="term">{term}</span>' in table
    assert '<td class="n">3</td>' in table  # n_orphan
    assert "2026-03-01 10:00:00" in table


def test_memory_page_reports_authors_and_unattributed(served: Served) -> None:
    body = served.get("/memory").body
    summary = _table(body, "t-summary")
    assert re.search(rf'<td class="m">{OPUS}</td><td class="n">2</td><td class="n">2</td>'
                     r'<td class="n">2</td>', summary)
    assert re.search(rf'<td class="m">{SONNET}</td><td class="n">1</td><td class="n">2</td>'
                     r'<td class="n">2</td>', summary)
    files = _table(body, "t-files")
    assert f"<td class=\"m\">p/a.md</td><td class=\"m\">{OPUS}</td>" in files
    assert f"{SONNET}\u00d71" in files  # the edit
    assert f"<td class=\"m\">p/b.md</td><td class=\"m\">{SONNET}</td>" in files
    assert "missing" in files.split("p/d.md")[1].split("</tr>")[0]
    assert '<li class="mono attn-text">p/c.md</li>' in body
    assert "Unattributed (1)" in body


def test_memory_page_filters_by_model(served: Served) -> None:
    opus = _table(served.get(f"/memory?model={OPUS}").body, "t-files")
    assert "p/a.md" in opus and "p/d.md" in opus and "p/b.md" not in opus
    sonnet = _table(served.get(f"/memory?model={SONNET}").body, "t-files")
    assert "p/a.md" in sonnet and "p/b.md" in sonnet and "p/d.md" not in sonnet
    nobody = served.get("/memory?model=nobody")
    assert nobody.status == 200
    assert "No files match." in nobody.body
    assert "Unattributed (1)" in nobody.body  # judged against every write, not the filter


def test_denials_page_shows_the_next_action_and_hygiene(served: Served) -> None:
    body = served.get("/denials").body
    table = _table(body, "t-denials")
    assert table.count("<tr>") == 2
    assert "rm -rf /tmp/x" in table
    assert '<td class="m">other-tool</td><td class="m">Read</td>' in table
    hygiene = _table(body, "t-hygiene")
    assert '<td class="m">pending_tool_call</td><td class="n">1</td><td class="n">1</td>' in hygiene
    assert "Showing 1 of 1" in body


def test_session_page_renders_the_timeline(served: Served, fixture: Fixture, capsys) -> None:
    reply = served.get("/session/sess-alp")  # a unique prefix
    assert reply.status == 200
    cli = _cli_json("session-trace", "sess-alpha", "--db", str(fixture.db), "--format", "json",
                    capsys=capsys)
    total = cli["coverage"]["total"]
    assert f"Showing {total} of {total} rows." in reply.body
    assert "sess-alpha" in reply.body
    assert "rm -rf /tmp/x" in reply.body
    assert 'class="out-denied">denied<' in reply.body


def test_session_page_names_an_unknown_or_ambiguous_prefix(served: Served) -> None:
    unknown = served.get("/session/nope")
    assert unknown.status == 404
    assert "no session matching prefix &#x27;nope&#x27;" in unknown.body
    ambiguous = served.get("/session/sess-")
    assert ambiguous.status == 404
    assert "matches 3 sessions" in ambiguous.body
    assert "sess-alpha" in ambiguous.body and "sess-gamma" in ambiguous.body


def test_session_page_shows_the_outline_above_the_timeline(served: Served) -> None:
    body = served.get("/session/sess-alpha").body
    outline = _table(body, "t-outline")
    assert outline.count("<tr>") == 2  # header + the one segment
    assert "2026-03-01 10:00:00" in outline  # the segment's time range
    assert "1 exchange" in body  # the header line names the exchange count
    # Transcript text inside the outline is escaped, exactly like the timeline.
    assert ESCAPED_SCRIPT in outline
    assert SCRIPT not in outline
    assert "zorblax" in outline
    # The outline sits above the timeline.
    assert body.index("t-outline") < body.index("t-trace")


def test_session_outline_is_cached_until_the_database_changes(
    served: Served, fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real = serve.topics_outline

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(serve, "topics_outline", counting)
    for _ in range(3):
        assert served.get("/session/sess-alpha").status == 200
    assert len(calls) == 1

    stat = fixture.db.stat()
    os.utime(fixture.db, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    assert served.get("/session/sess-alpha").status == 200
    assert len(calls) == 2
    assert served.get("/session/sess-alpha").status == 200
    assert len(calls) == 2


# ---------------------------------------------------------------- JSON


def test_api_orphans_matches_the_cli(served: Served, fixture: Fixture, capsys) -> None:
    api = served.get("/api/orphans.json")
    assert api.status == 200
    assert api.headers["content-type"] == "application/json; charset=utf-8"
    cli = _cli_json("orphans", "--db", str(fixture.db), "--no-default-sinks", "--limit", "30",
                    "--json", capsys=capsys)
    assert api.json() == cli
    assert api.json()["candidates"][0]["orphan_terms"] == ["zorblax", "flumbergast", "quuxify"]


def test_api_memory_matches_the_cli(served: Served, fixture: Fixture, capsys) -> None:
    base = ["memory-authors", "--db", str(fixture.db), "--memory-dir", str(fixture.memory_dir)]
    api = served.get("/api/memory.json")
    assert api.json() == _cli_json(*base, "--json", capsys=capsys)
    assert api.json()["unattributed"] == ["p/c.md"]
    filtered = served.get(f"/api/memory.json?model={SONNET}")
    assert filtered.json() == _cli_json(*base, "--model", SONNET, "--json", capsys=capsys)


def test_api_denials_carries_denials_and_hygiene_as_the_cli_prints_them(
    served: Served, fixture: Fixture, capsys
) -> None:
    api = served.get("/api/denials.json").json()
    denials = _cli_json("denials", "--db", str(fixture.db), "--format", "json", capsys=capsys)
    hygiene = _cli_json("hygiene", "--db", str(fixture.db), "--format", "json", capsys=capsys)
    assert api["denials"] == denials
    assert api["denials_total"] == 1
    assert api["hygiene"] == hygiene


def test_api_overview_reports_tiles_and_table_counts(served: Served) -> None:
    for name in ("overview", "info"):
        data = served.get(f"/api/{name}.json").json()
        assert data["tiles"] == {
            "orphan_candidates": 1,
            "memory_unattributed": 1,
            "denied_calls": 1,
        }
        assert data["table_counts"]["sessions"] == 3
        assert data["freshness"] == {"state": "current", "gap": 0}
        assert data["roots"]["sources"][0]["files"] == 3


# ---------------------------------------------------------------- errors


def test_unknown_paths_are_404(served: Served) -> None:
    for path in ("/nope", "/session/", "/api/nope.json", "/orphans/extra"):
        assert served.get(path).status == 404, path
    assert served.get("/api/nope.json").json()["status"] == 404


def test_a_handler_exception_is_a_500_and_the_server_keeps_serving(
    served: Served, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom <b>x</b>")

    monkeypatch.setattr(serve, "find_orphans", boom)
    failed = served.get("/orphans")
    assert failed.status == 500
    assert "RuntimeError" in failed.body
    assert "boom &lt;b&gt;x&lt;/b&gt;" in failed.body
    assert served.get("/memory").status == 200
    assert served.get("/api/orphans.json").status == 500


def test_other_methods_are_refused(served: Served) -> None:
    assert served.get("/", method="POST").status == 405
    head = served.get("/", method="HEAD")
    assert head.status == 200 and head.body == ""


# ---------------------------------------------------------------- invariants


def _hold_write_lock(db: Path) -> subprocess.Popen[str]:
    code = "import duckdb, sys\nc = duckdb.connect(sys.argv[1])\nprint('ready', flush=True)\nsys.stdin.read()\n"
    proc = subprocess.Popen(
        [sys.executable, "-c", code, str(db)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None and proc.stdout.readline().strip() == "ready"
    return proc


def _release(proc: subprocess.Popen[str]) -> None:
    assert proc.stdin is not None
    proc.stdin.close()
    proc.wait(timeout=30)


def test_no_connection_is_held_between_requests(served: Served, fixture: Fixture) -> None:
    for path in ("/", "/orphans", "/memory", "/denials", "/session/sess-alpha"):
        assert served.get(path).status == 200
    code = "import duckdb, sys\nc = duckdb.connect(sys.argv[1])\nc.execute('SELECT 1')\nc.close()\n"
    result = subprocess.run(
        [sys.executable, "-c", code, str(fixture.db)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr


def test_a_rebuild_in_progress_is_a_503_and_the_server_answers_afterwards(
    served: Served, fixture: Fixture
) -> None:
    assert served.get("/orphans").status == 200  # the cache is warm: 503 must still win
    proc = _hold_write_lock(fixture.db)
    try:
        for path in ("/orphans", "/", "/session/sess-alpha", "/api/memory.json"):
            reply = served.get(path)
            assert reply.status == 503, path
            assert reply.headers["retry-after"] == "30"
        page = served.get("/denials")
        assert "being rebuilt" in page.body
        assert page.headers["content-type"] == "text/html; charset=utf-8"
        assert served.get("/api/orphans.json").json()["status"] == 503
    finally:
        _release(proc)
    assert served.get("/orphans").status == 200
    assert served.get("/").status == 200


def test_a_missing_database_is_a_503_not_a_crash(tmp_path: Path) -> None:
    server = serve.make_server(tmp_path / "absent.duckdb", port=0, default_sinks=False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert Served(server).get("/").status == 503
        assert Served(server).get("/nope").status == 404
    finally:
        server.shutdown()
        server.server_close()


def test_serving_never_writes_to_the_database(served: Served, fixture: Fixture) -> None:
    before = fixture.db.stat()
    for path in ("/", "/orphans", "/memory", "/denials", "/session/sess-beta"):
        assert served.get(path).status == 200
    after = fixture.db.stat()
    assert (after.st_mtime_ns, after.st_size) == (before.st_mtime_ns, before.st_size)


def test_database_text_is_escaped_on_orphans_and_the_session_page(served: Served) -> None:
    for path in ("/orphans", "/session/sess-alpha"):
        body = served.get(path).body
        assert ESCAPED_SCRIPT in body, path
        assert SCRIPT not in body, path


def test_pages_load_nothing_from_outside(served: Served) -> None:
    for path in ("/", "/orphans", "/memory", "/denials", "/session/sess-alpha"):
        reply = served.get(path)
        assert not re.search(r"""(src|href|action)=["']?(https?:)?//""", reply.body), path
        assert "@import" not in reply.body and "url(" not in reply.body
        assert "default-src 'none'" in reply.headers["content-security-policy"]
        assert reply.headers["cache-control"] == "no-store"


def test_a_foreign_host_header_is_refused(served: Served) -> None:
    assert served.get("/", headers={"Host": "evil.example"}).status == 403
    assert served.get("/", headers={"Host": "evil.example:8772"}).status == 403
    assert served.get("/", headers={"Host": "localhost:8772"}).status == 200
    assert served.get("/", headers={"Host": "[::1]:8772"}).status == 200


# ---------------------------------------------------------------- cache


def test_orphans_are_cached_until_the_database_file_changes(
    served: Served, fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real = serve.find_orphans

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(serve, "find_orphans", counting)
    assert served.get("/orphans").status == 200
    assert served.get("/orphans").status == 200
    assert served.get("/api/orphans.json").status == 200
    assert served.get("/").status == 200  # the overview tile reuses the same data
    assert len(calls) == 1

    stat = fixture.db.stat()
    os.utime(fixture.db, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    assert served.get("/orphans").status == 200
    assert len(calls) == 2
    assert served.get("/orphans").status == 200
    assert len(calls) == 2


def _stamp(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@pytest.fixture
def sink_dir(tmp_path: Path) -> Path:
    path = tmp_path / "sinks"
    path.mkdir()
    (path / "unrelated.md").write_text("nothing of interest here\n", encoding="utf-8")
    return path


@pytest.fixture
def served_with_sink(fixture: Fixture, sink_dir: Path) -> Iterator[Served]:
    server = serve.make_server(
        fixture.db,
        port=0,
        sinks=[sink_dir],
        default_sinks=False,
        memory_dirs=[fixture.memory_dir],
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield Served(server)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(10)


def _n_orphan(served: Served) -> int | None:
    """The fixture candidate's ``n_orphan``, or ``None`` once it is not nominated."""
    candidates = served.get("/api/orphans.json").json()["candidates"]
    assert len(candidates) <= 1
    return candidates[0]["n_orphan"] if candidates else None


def test_a_memo_written_into_a_sink_retires_a_candidate_without_a_rebuild(
    served_with_sink: Served, fixture: Fixture, sink_dir: Path
) -> None:
    served = served_with_sink
    before = _stamp(fixture.db)
    assert _n_orphan(served) == 3

    (sink_dir / "memo.md").write_text("we settled on zorblax for this.\n", encoding="utf-8")
    assert served.get("/orphans").status == 200
    assert (_n_orphan(served) or 0) < 3  # zorblax is no longer an orphan term

    (sink_dir / "memo.md").write_text("zorblax quuxify flumbergast\n", encoding="utf-8")
    assert _n_orphan(served) is None  # edited in place: every term is covered now
    assert 'href="/session/sess-alpha"' not in served.get("/orphans").body

    (sink_dir / "memo.md").unlink()
    assert _n_orphan(served) == 3  # removed: the candidate is back
    assert _stamp(fixture.db) == before  # ... and never through a rebuild


def test_a_sink_edit_that_keeps_size_and_count_is_still_noticed(
    served_with_sink: Served, sink_dir: Path
) -> None:
    memo = sink_dir / "memo.md"
    memo.write_text("aaaaaaaaaaaaa\n", encoding="utf-8")
    assert _n_orphan(served_with_sink) == 3
    memo.write_text("zorblax quuxi\n", encoding="utf-8")  # same length, same file count
    stat = memo.stat()
    os.utime(memo, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    assert (_n_orphan(served_with_sink) or 0) < 3


def test_unchanged_sinks_do_not_recompute_orphans(
    served_with_sink: Served, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real = serve.find_orphans

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(serve, "find_orphans", counting)
    for path in ("/orphans", "/orphans", "/api/orphans.json", "/"):
        assert served_with_sink.get(path).status == 200
    assert len(calls) == 1


def _memory_row(body: str, key: str) -> str:
    return _table(body, "t-files").split(key)[1].split("</tr>")[0]


def test_a_memory_file_deleted_on_disk_shows_as_missing_without_a_rebuild(
    served: Served, fixture: Fixture
) -> None:
    before = _stamp(fixture.db)
    first = served.get("/memory").body
    assert "missing" not in _memory_row(first, "p/b.md")
    assert "missing" in _memory_row(first, "p/d.md")  # never on disk
    assert "Unattributed (1)" in first

    (fixture.memory_dir / "b.md").unlink()
    (fixture.memory_dir / "c.md").unlink()
    second = served.get("/memory").body
    assert "missing" in _memory_row(second, "p/b.md")
    assert "Unattributed (0)" in second  # c.md is gone from the disk scan

    (fixture.memory_dir / "d.md").write_text("# d\n", encoding="utf-8")
    third = served.get("/memory").body
    assert "missing" not in _memory_row(third, "p/d.md")  # and back the other way
    assert _stamp(fixture.db) == before


def test_unchanged_memory_files_do_not_recompute_the_memory_analysis(
    served: Served, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real = serve.find_memory_files

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(serve, "find_memory_files", counting)
    for path in ("/memory", "/memory", f"/memory?model={OPUS}", "/api/memory.json", "/"):
        assert served.get(path).status == 200
    assert len(calls) == 1


def test_a_database_removed_after_the_server_started_is_a_503_and_it_recovers(
    served: Served, fixture: Fixture
) -> None:
    assert served.get("/").status == 200  # warm caches: the 503 must still win
    gone = fixture.db.with_name("gone.duckdb")
    fixture.db.rename(gone)
    reply = served.get("/")
    assert reply.status == 503
    assert reply.headers["retry-after"] == "30"
    assert served.get("/api/overview.json").status == 503
    assert served.get("/nope").status == 404  # the server is still answering
    gone.rename(fixture.db)
    assert served.get("/").status == 200


def test_a_database_that_vanishes_mid_request_is_a_503_and_the_server_recovers(
    served: Served, fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    gone = fixture.db.with_name("gone.duckdb")
    real = serve.database_info

    def info_then_vanish(*args: Any, **kwargs: Any) -> Any:
        info = real(*args, **kwargs)
        fixture.db.rename(gone)  # a rebuild swaps the file out while the overview is assembled
        return info

    monkeypatch.setattr(serve, "database_info", info_then_vanish)
    reply = served.get("/")
    assert reply.status == 503, reply.body
    assert reply.headers["retry-after"] == "30"
    monkeypatch.setattr(serve, "database_info", real)
    gone.rename(fixture.db)
    assert served.get("/").status == 200


# ---------------------------------------------------------------- the CLI


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.0.10", "example.com", ""])
def test_a_non_loopback_host_is_refused_before_binding(
    host: str, fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    def bound(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a socket was created")

    monkeypatch.setattr(serve, "_Server4", bound)
    monkeypatch.setattr(serve, "_Server6", bound)
    code = main(["serve", "--db", str(fixture.db), "--host", host, "--port", "0"])
    assert code == 2
    assert "loopback" in capsys.readouterr().err


def test_a_non_loopback_host_exits_non_zero_as_a_process(fixture: Fixture) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ashiato.cli", "serve", "--db", str(fixture.db),
         "--host", "0.0.0.0", "--port", "0"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "loopback" in result.stderr
    assert result.stdout == ""


def test_loopback_hosts_are_accepted() -> None:
    assert serve.bind_address("127.0.0.1") == "127.0.0.1"
    assert serve.bind_address("localhost") == "127.0.0.1"
    assert serve.bind_address("LOCALHOST") == "127.0.0.1"
    assert serve.bind_address("::1") == "::1"


def test_host_header_parsing() -> None:
    assert serve.host_allowed(None)
    assert serve.host_allowed("localhost")
    assert serve.host_allowed("127.0.0.1:8772")
    assert serve.host_allowed("[::1]:8772")
    assert not serve.host_allowed("localhost.evil.example")
    assert not serve.host_allowed("0.0.0.0:8772")


def test_serve_prints_its_url_and_answers(fixture: Fixture, home: Path) -> None:
    proc = subprocess.Popen(
        [sys.executable, "-m", "ashiato.cli", "serve", "--db", str(fixture.db), "--port", "0",
         "--no-default-sinks", "--memory-dir", str(fixture.memory_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "HOME": str(home)},
    )
    try:
        assert proc.stdout is not None
        url = proc.stdout.readline().strip()
        assert re.fullmatch(r"http://127\.0\.0\.1:\d+/", url), url
        with urllib.request.urlopen(f"{url}api/orphans.json", timeout=60) as response:
            assert response.status == 200
            assert json.load(response)["candidates"][0]["session_id"] == "sess-alpha"
    finally:
        proc.terminate()
        proc.wait(timeout=30)
