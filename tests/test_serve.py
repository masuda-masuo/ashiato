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
import socket
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

    def post(
        self,
        path: str,
        body: str = "",
        headers: dict[str, str] | None = None,
    ) -> Reply:
        hdrs = {"Content-Type": "application/x-www-form-urlencoded", "Content-Length": str(len(body.encode("utf-8")))}
        if headers:
            hdrs.update(headers)
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            connection.request("POST", path, body=body.encode("utf-8"), headers=hdrs)
            response = connection.getresponse()
            resp_body = response.read().decode("utf-8")
            return Reply(response.status, {k.lower(): v for k, v in response.getheaders()}, resp_body)
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
# ---------------------------------------------------------------- reviewed marks (issue #56)


def test_a_reviewed_mark_hides_the_candidate_on_the_next_reload(
    served: Served, fixture: Fixture
) -> None:
    reviewed = fixture.db.parent / "orphans-reviewed.txt"
    before = _stamp(fixture.db)
    assert _n_orphan(served) == 3
    assert _tile(served.get("/").body, "orphans") == 1

    reviewed.write_text("sess-alpha\n", encoding="utf-8")
    assert served.get("/orphans").status == 200
    assert _n_orphan(served) is None
    assert 'href="/session/sess-alpha"' not in served.get("/orphans").body
    assert _tile(served.get("/").body, "orphans") == 0
    assert served.get("/api/orphans.json").json()["reviewed_hidden"] == 1
    assert _stamp(fixture.db) == before  # ... and never through a rebuild

    reviewed.write_text("", encoding="utf-8")  # unmarked: the candidate is back
    assert _n_orphan(served) == 3
    assert _tile(served.get("/").body, "orphans") == 1
    assert _stamp(fixture.db) == before


def test_writing_the_reviewed_file_recomputes_orphans(
    served: Served, fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real = serve.find_orphans

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(serve, "find_orphans", counting)
    for path in ("/orphans", "/orphans", "/api/orphans.json", "/"):
        assert served.get(path).status == 200
    assert len(calls) == 1

    (fixture.db.parent / "orphans-reviewed.txt").write_text("sess-alpha\n", encoding="utf-8")
    assert served.get("/orphans").status == 200
    assert len(calls) == 2
    assert served.get("/orphans").status == 200
    assert len(calls) == 2


def test_api_orphans_matches_the_cli_with_reviewed_marks(
    served: Served, fixture: Fixture, capsys
) -> None:
    (fixture.db.parent / "orphans-reviewed.txt").write_text("sess-alpha\n", encoding="utf-8")
    api = served.get("/api/orphans.json")
    cli = _cli_json(
        "orphans", "--db", str(fixture.db), "--no-default-sinks", "--limit", "30",
        "--json", capsys=capsys,
    )
    assert api.json() == cli
    assert api.json()["candidates"] == []
    assert api.json()["reviewed_hidden"] == 1
    assert api.json()["reviewed_file"] == str(fixture.db.parent / "orphans-reviewed.txt")


def test_orphans_page_with_invalid_utf8_reviewed_file_is_a_clean_500(
    served: Served, fixture: Fixture
) -> None:
    """A reviewed file with invalid UTF-8 makes /orphans answer a clean 500 page,
    not a traceback."""
    (fixture.db.parent / "orphans-reviewed.txt").write_bytes(b"\xff\xfe\n")
    reply = served.get("/orphans")
    assert reply.status == 500
    assert "Traceback" not in reply.body
    assert "Internal error" in reply.body
    # The server keeps running.
    (fixture.db.parent / "orphans-reviewed.txt").unlink()
    assert served.get("/orphans").status == 200


# ---------------------------------------------------------------- CSRF + POST /orphans/reviewed


def _csrf_token(served: Served) -> str:
    """Extract the CSRF token from the hidden field on the orphans page."""
    body = served.get("/orphans").body
    # Look for the always-present hidden input first.
    m = re.search(r'id="csrf-token"\s+value="([^"]+)"', body)
    if m:
        return m.group(1)
    # Fall back to form hidden fields.
    m = re.search(r'name="token"\s+value="([^"]+)"', body)
    assert m, "no CSRF token found on /orphans"
    return m.group(1)


def _origin(served: Served) -> str:
    """Build a matching Origin header value from the served port."""
    return f"http://127.0.0.1:{served.port}"


# -- success path -----------------------------------------------------------


def test_post_mark_reviewed_hides_candidate(
    served: Served, fixture: Fixture
) -> None:
    """POST with valid token + matching Origin marks the session; next GET hides it."""
    assert _n_orphan(served) == 3
    token = _csrf_token(served)
    reply = served.post(
        "/orphans/reviewed",
        body="session_id=sess-alpha&action=mark&token=" + token,
        headers={"Origin": _origin(served)},
    )
    assert reply.status == 303
    assert reply.headers["location"] == "/orphans"
    # The candidate is gone on the next reload.
    assert _n_orphan(served) is None
    assert 'href="/session/sess-alpha"' not in served.get("/orphans").body
    # The reviewed file contains the full id.
    reviewed = fixture.db.parent / "orphans-reviewed.txt"
    assert "sess-alpha" in reviewed.read_text(encoding="utf-8")

    # Unmark to restore state.  Token doesn't change; reuse the one from before.
    served.post(
        "/orphans/reviewed",
        body="session_id=sess-alpha&action=unmark&token=" + token,
        headers={"Origin": _origin(served)},
    )
    assert _n_orphan(served) == 3


def test_post_unmark_reviewed_restores_candidate(
    served: Served, fixture: Fixture
) -> None:
    """POST with action=unmark removes a reviewed session."""
    reviewed = fixture.db.parent / "orphans-reviewed.txt"
    reviewed.write_text("sess-alpha\n", encoding="utf-8")
    assert _n_orphan(served) is None

    token = _csrf_token(served)
    reply = served.post(
        "/orphans/reviewed",
        body="session_id=sess-alpha&action=unmark&token=" + token,
        headers={"Origin": _origin(served)},
    )
    assert reply.status == 303
    assert _n_orphan(served) == 3


# -- CSRF guard tests -------------------------------------------------------


def test_post_missing_token_is_403(served: Served, fixture: Fixture) -> None:
    """POST without a token returns 403 and the file is unchanged."""
    reviewed = fixture.db.parent / "orphans-reviewed.txt"
    reviewed.write_text("", encoding="utf-8")
    before = reviewed.read_bytes()
    reply = served.post(
        "/orphans/reviewed",
        body="session_id=sess-alpha&action=mark",
        headers={"Origin": _origin(served)},
    )
    assert reply.status == 403
    assert reviewed.read_bytes() == before


def test_post_wrong_token_is_403(served: Served, fixture: Fixture) -> None:
    """POST with a wrong token returns 403 and the file is unchanged."""
    reviewed = fixture.db.parent / "orphans-reviewed.txt"
    reviewed.write_text("", encoding="utf-8")
    before = reviewed.read_bytes()
    token = _csrf_token(served)
    reply = served.post(
        "/orphans/reviewed",
        body="session_id=sess-alpha&action=mark&token=WRONG_" + token,
        headers={"Origin": _origin(served)},
    )
    assert reply.status == 403
    assert reviewed.read_bytes() == before


def test_post_missing_origin_is_403(served: Served, fixture: Fixture) -> None:
    """POST without an Origin header returns 403 and the file is unchanged."""
    reviewed = fixture.db.parent / "orphans-reviewed.txt"
    reviewed.write_text("", encoding="utf-8")
    before = reviewed.read_bytes()
    token = _csrf_token(served)
    reply = served.post(
        "/orphans/reviewed",
        body="session_id=sess-alpha&action=mark&token=" + token,
    )
    assert reply.status == 403
    assert reviewed.read_bytes() == before


def test_post_origin_null_is_403(served: Served, fixture: Fixture) -> None:
    """POST with Origin: null returns 403 and the file is unchanged."""
    reviewed = fixture.db.parent / "orphans-reviewed.txt"
    reviewed.write_text("", encoding="utf-8")
    before = reviewed.read_bytes()
    token = _csrf_token(served)
    reply = served.post(
        "/orphans/reviewed",
        body="session_id=sess-alpha&action=mark&token=" + token,
        headers={"Origin": "null"},
    )
    assert reply.status == 403
    assert reviewed.read_bytes() == before


def test_post_foreign_origin_is_403(served: Served, fixture: Fixture) -> None:
    """POST with a foreign Origin returns 403 and the file is unchanged."""
    reviewed = fixture.db.parent / "orphans-reviewed.txt"
    reviewed.write_text("", encoding="utf-8")
    before = reviewed.read_bytes()
    token = _csrf_token(served)
    reply = served.post(
        "/orphans/reviewed",
        body="session_id=sess-alpha&action=mark&token=" + token,
        headers={"Origin": "http://evil.example"},
    )
    assert reply.status == 403
    assert reviewed.read_bytes() == before


def test_post_non_loopback_host_is_403(served: Served, fixture: Fixture) -> None:
    """POST with a non-loopback Host returns 403 and the file is unchanged."""
    reviewed = fixture.db.parent / "orphans-reviewed.txt"
    reviewed.write_text("", encoding="utf-8")
    before = reviewed.read_bytes()
    token = _csrf_token(served)
    reply = served.post(
        "/orphans/reviewed",
        body="session_id=sess-alpha&action=mark&token=" + token,
        headers={"Origin": "http://evil.example", "Host": "evil.example"},
    )
    assert reply.status == 403
    assert reviewed.read_bytes() == before


# -- input validation -------------------------------------------------------


def test_post_unknown_session_id_is_400(served: Served, fixture: Fixture) -> None:
    """POST with an unknown session id returns 400 and the file is unchanged."""
    reviewed = fixture.db.parent / "orphans-reviewed.txt"
    reviewed.write_text("", encoding="utf-8")
    before = reviewed.read_bytes()
    token = _csrf_token(served)
    reply = served.post(
        "/orphans/reviewed",
        body="session_id=no-such-id&action=mark&token=" + token,
        headers={"Origin": _origin(served)},
    )
    assert reply.status == 400
    assert reviewed.read_bytes() == before


def test_post_session_prefix_is_not_resolved(served: Served, fixture: Fixture) -> None:
    """A prefix that the CLI would resolve is rejected from the web: only a
    full session id marks anything, and nothing is written for a prefix."""
    reviewed = fixture.db.parent / "orphans-reviewed.txt"
    reviewed.write_text("", encoding="utf-8")
    before = reviewed.read_bytes()
    token = _csrf_token(served)
    reply = served.post(
        "/orphans/reviewed",
        body="session_id=sess-al&action=mark&token=" + token,
        headers={"Origin": _origin(served)},
    )
    assert reply.status == 400
    assert reviewed.read_bytes() == before


def test_post_bad_action_is_400(served: Served, fixture: Fixture) -> None:
    """POST with an invalid action returns 400 and the file is unchanged."""
    reviewed = fixture.db.parent / "orphans-reviewed.txt"
    reviewed.write_text("", encoding="utf-8")
    before = reviewed.read_bytes()
    token = _csrf_token(served)
    reply = served.post(
        "/orphans/reviewed",
        body="session_id=sess-alpha&action=delete&token=" + token,
        headers={"Origin": _origin(served)},
    )
    assert reply.status == 400
    assert reviewed.read_bytes() == before


def test_post_oversize_body_is_413(served: Served, fixture: Fixture) -> None:
    """POST with a body over 4 KiB returns 413 and the file is unchanged."""
    reviewed = fixture.db.parent / "orphans-reviewed.txt"
    reviewed.write_text("", encoding="utf-8")
    before = reviewed.read_bytes()
    token = _csrf_token(served)
    big_body = "x" * 5000 + "&token=" + token
    reply = served.post(
        "/orphans/reviewed",
        body=big_body,
        headers={"Origin": _origin(served)},
    )
    assert reply.status == 413
    assert reviewed.read_bytes() == before


def test_post_on_an_out_of_date_database_is_503_and_writes_nothing(
    served: Served, fixture: Fixture
) -> None:
    """POST /orphans/reviewed against a stale-schema database answers the clean
    503 the GET pages give, and the reviewed file is unchanged."""
    reviewed = fixture.db.parent / "orphans-reviewed.txt"
    reviewed.write_text("", encoding="utf-8")
    before = reviewed.read_bytes()
    token = _csrf_token(served)
    # Make the database look built by an older version: drop the format marker.
    code = (
        "import duckdb, sys\n"
        "c = duckdb.connect(sys.argv[1])\n"
        "c.execute(\"DELETE FROM ashiato_meta WHERE key = 'format_version'\")\n"
        "c.close()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(fixture.db)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    reply = served.post(
        "/orphans/reviewed",
        body="session_id=sess-alpha&action=mark&token=" + token,
        headers={"Origin": _origin(served)},
    )
    assert reply.status == 503
    assert "Database unavailable" in reply.body
    assert reviewed.read_bytes() == before


# -- other methods / paths still 405 ----------------------------------------


def _raw_post(served: Served, path: str, headers: str) -> Reply:
    """POST *path* over a raw socket with exactly *headers* (a ``\\r\\n``-joined
    block, e.g. ``"Connection: close\\r\\n"``).  ``http.client`` would silently
    add ``Content-Length: 0`` to a bodiless POST -- precisely what must *not*
    be sent when testing a request with no Content-Length at all.
    """
    request = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{served.port}\r\n"
        f"{headers}\r\n"
    ).encode()
    with socket.create_connection(("127.0.0.1", served.port), timeout=30) as sock:
        sock.sendall(request)
        data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    lines = head.split(b"\r\n")
    status = int(lines[0].split()[1])
    resp_headers: dict[str, str] = {}
    for line in lines[1:]:
        name, _, value = line.decode("latin-1").partition(":")
        resp_headers[name.lower().strip()] = value.strip()
    return Reply(status, resp_headers, body.decode("utf-8", errors="replace"))


def test_post_to_other_path_is_405(served: Served) -> None:
    """POST to any path other than /orphans/reviewed is 405."""
    assert served.post("/", body="foo=bar").status == 405
    assert served.post("/api/orphans.json", body="foo=bar").status == 405


def test_post_wrong_path_is_405_before_body_checks(served: Served) -> None:
    """Only POST /orphans/reviewed validates the body: any other POST path
    answers 405 without reading the body, whatever its Content-Length says."""
    for path in ("/orphans", "/"):
        # (a) no Content-Length header at all.
        reply = _raw_post(served, path, "Connection: close\r\n")
        assert reply.status == 405, (path, "no Content-Length")
        # (b) an unparseable Content-Length.
        reply = served.get(path, method="POST", headers={"Content-Length": "abc"})
        assert reply.status == 405, (path, "Content-Length: abc")
        # (c) an oversize Content-Length.
        reply = served.get(path, method="POST", headers={"Content-Length": "999999"})
        assert reply.status == 405, (path, "Content-Length: 999999")


def test_put_delete_patch_options_still_405(served: Served) -> None:
    """PUT, DELETE, PATCH, OPTIONS are 405 as before."""
    assert served.get("/", method="PUT").status == 405
    assert served.get("/", method="DELETE").status == 405
    assert served.get("/", method="PATCH").status == 405
    assert served.get("/", method="OPTIONS").status == 405


# -- security header checks -------------------------------------------------


def test_csp_contains_form_action_self(served: Served) -> None:
    """Content-Security-Policy allows form-action 'self'."""
    reply = served.get("/")
    csp = reply.headers.get("content-security-policy", "")
    assert "form-action 'self'" in csp
    assert "form-action 'none'" not in csp


def test_referrer_policy_is_same_origin(served: Served) -> None:
    """Referrer-Policy is 'same-origin'."""
    reply = served.get("/")
    assert reply.headers.get("referrer-policy") == "same-origin"


# -- real HTTP handler tests ------------------------------------------------


def test_real_http_post_mark_reviewed(served: Served, fixture: Fixture) -> None:
    """A real HTTP POST over a socket marks a session reviewed."""
    assert _n_orphan(served) == 3
    token = _csrf_token(served)
    reply = served.post(
        "/orphans/reviewed",
        body="session_id=sess-alpha&action=mark&token=" + token,
        headers={"Origin": _origin(served)},
    )
    assert reply.status == 303
    assert reply.headers["location"] == "/orphans"
    # Verify the candidate is gone via API.
    api = served.get("/api/orphans.json").json()
    assert api["candidates"] == []
    assert api["reviewed_hidden"] == 1

    # Restore.  Token doesn't change; reuse the one from before.
    served.post(
        "/orphans/reviewed",
        body="session_id=sess-alpha&action=unmark&token=" + token,
        headers={"Origin": _origin(served)},
    )
    assert _n_orphan(served) == 3


def test_real_http_post_missing_origin_is_403(served: Served, fixture: Fixture) -> None:
    """A real HTTP POST without Origin returns 403 and the file is unchanged."""
    reviewed = fixture.db.parent / "orphans-reviewed.txt"
    reviewed.write_text("", encoding="utf-8")
    before = reviewed.read_bytes()
    token = _csrf_token(served)
    # POST without Origin header.
    reply = served.post(
        "/orphans/reviewed",
        body="session_id=sess-alpha&action=mark&token=" + token,
    )
    assert reply.status == 403
    assert reviewed.read_bytes() == before


# -- reviewed page with ?reviewed=1 -----------------------------------------


def test_orphans_page_shows_reviewed_toggle(served: Served, fixture: Fixture) -> None:
    """The orphans page shows a 'show N reviewed' link when some are hidden."""
    assert _n_orphan(served) == 3
    # Initially no reviewed sessions, no toggle.
    body = served.get("/orphans").body
    assert "show 1 reviewed" not in body

    # Mark it.
    token = _csrf_token(served)
    served.post(
        "/orphans/reviewed",
        body="session_id=sess-alpha&action=mark&token=" + token,
        headers={"Origin": _origin(served)},
    )
    # The orphans page now has 0 candidates and shows "show 1 reviewed".
    body = served.get("/orphans").body
    assert "show 1 reviewed" in body

    # The ?reviewed=1 page shows the reviewed candidate with an unreview button.
    body = served.get("/orphans?reviewed=1").body
    assert 'href="/session/sess-alpha"' in body
    assert "hide reviewed" in body
    assert 'value="unmark"' in body
    # Restore.
    served.post(
        "/orphans/reviewed",
        body="session_id=sess-alpha&action=unmark&token=" + token,
        headers={"Origin": _origin(served)},
    )
    assert _n_orphan(served) == 3
