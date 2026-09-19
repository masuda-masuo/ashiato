"""Tests for ashiato.memory_authors: which model wrote each memory file.

Fixtures are hand-built JSONL transcripts run through the real build pipeline,
one transcript file per session, in a private ``tmp_path`` -- never in
``tests/fixtures/``, which other tests glob recursively.  Each assistant
``tool_use`` line carries a ``model`` in its message, so the built ``events``
rows carry the models the module is supposed to read back.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from ashiato.build import build
from ashiato.cli import main
from ashiato.memory_authors import (
    default_memory_dirs,
    memory_key,
    model_name,
    run,
    scan_memory_dirs,
)

A_PATH = "/home/u/.claude/projects/p/memory/a.md"
B_PATH = "/home/u/.claude/projects/p/memory/b.md"
A_NAME = "a.md"


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty HOME, so the default memory dirs never read the real ~/.claude."""
    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HOME", str(path))
    return path


# ---------------------------------------------------------------- fixtures


def _record(
    role: str,
    uuid: str,
    blocks: list[dict[str, Any]],
    *,
    session: str,
    ts: str,
    model: str | None = None,
) -> dict[str, Any]:
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
    return record


def _tool_use(tool: str, tool_input: dict[str, Any], use_id: str) -> dict[str, Any]:
    return {"type": "tool_use", "id": use_id, "name": tool, "input": tool_input}


def _tool_result(use_id: str, text: str, *, is_error: bool = False) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": use_id,
        "content": [{"type": "text", "text": text}],
        "is_error": is_error,
    }


def _transcript(calls: list[dict[str, Any]], day: str) -> str:
    """One session transcript from call dicts.

    Each call carries ``tool``, ``input`` and optionally ``model`` (absent =
    NULL), ``session``, ``result`` (result text), ``is_error`` and ``denied``.
    Timestamps are derived from the call's position -- one second apart from
    10:00:00 -- so ``--since`` / ``--until`` tests have a stable grid.
    """
    lines: list[str] = []
    for index, call in enumerate(calls):
        session = call.get("session", "s1")
        ts = f"{day}T10:{index // 60:02d}:{index % 60:02d}Z"
        use_id = call.get("id", f"u{index}")
        lines.append(
            json.dumps(
                _record(
                    "assistant",
                    f"{session}-e{2 * index}",
                    [_tool_use(call["tool"], call["input"], use_id)],
                    session=session,
                    ts=ts,
                    model=call.get("model"),
                )
            )
        )
        result = call.get("result", "ok")
        if call.get("denied"):
            result = "The user doesn't want to proceed with this tool use: " + result
        lines.append(
            json.dumps(
                _record(
                    "user",
                    f"{session}-e{2 * index + 1}",
                    [_tool_result(use_id, result, is_error=bool(call.get("is_error", False)))],
                    session=session,
                    ts=ts,
                )
            )
        )
    return "\n".join(lines) + "\n"


def w(file_path: str, model: str | None = None, **overrides: Any) -> dict[str, Any]:
    call: dict[str, Any] = {"tool": "Write", "input": {"file_path": file_path}, "model": model}
    call.update(overrides)
    return call


def e(file_path: str, model: str | None = None, **overrides: Any) -> dict[str, Any]:
    call: dict[str, Any] = {
        "tool": "Edit",
        "input": {"file_path": file_path, "old_string": "x", "new_string": "y"},
        "model": model,
    }
    call.update(overrides)
    return call


def m(file_path: str, model: str | None = None, **overrides: Any) -> dict[str, Any]:
    call: dict[str, Any] = {
        "tool": "MultiEdit",
        "input": {"file_path": file_path, "edits": [{"old_string": "x", "new_string": "y"}]},
        "model": model,
    }
    call.update(overrides)
    return call


def bash(command: str, model: str | None = None, **overrides: Any) -> dict[str, Any]:
    call: dict[str, Any] = {"tool": "Bash", "input": {"command": command}, "model": model}
    call.update(overrides)
    return call


def make_db(tmp_path: Path, sessions: list[tuple[str, str, list[dict[str, Any]]]]) -> Path:
    """Build a DuckDB from ``(session_id, "YYYY-MM-DD", calls)`` sessions."""
    directory = tmp_path / "transcripts"
    directory.mkdir(exist_ok=True)
    for index, (session_id, day, calls) in enumerate(sessions):
        path = directory / f"{index:02d}-{session_id}.jsonl"
        path.write_text(_transcript(calls, day), encoding="utf-8")
    db_path = tmp_path / "memory-authors.duckdb"
    build([directory], db_path)
    return db_path


def _run_json(
    db_path: Path, *args: str, capsys: pytest.CaptureFixture[str]
) -> dict[str, Any]:
    code = main(["memory-authors", "--db", str(db_path), "--json", *args])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    return json.loads(captured.out)


def _by_key(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {file["key"]: file for file in payload["files"]}


# ---------------------------------------------------------------- unit helpers


def test_memory_key() -> None:
    assert memory_key("/home/u/.claude/projects/p/memory/a.md") == "p/a.md"
    assert memory_key("C:\\Users\\u\\.claude\\projects\\p\\memory\\a.md") == "p/a.md"
    assert memory_key("/repo/.claude/projects/p/memory/a.md") == "p/a.md"
    assert memory_key("/home/u/.claude/projects/p/memory/MEMORY.md") == "p/MEMORY.md"
    assert memory_key("/repo/docs/memory/a.md") is None
    assert memory_key("/home/u/.claude/projects/p/memory/sub/a.md") is None
    assert memory_key("/home/u/.claude/projects/p/memory/a.txt") is None


def test_model_name() -> None:
    assert model_name("opus") == "opus"
    assert model_name(None) == "unknown"
    assert model_name("<synthetic>") == "unknown"


def test_default_memory_dirs_are_existing_memory_dirs(home: Path) -> None:
    (home / ".claude/projects/p1/memory").mkdir(parents=True)
    (home / ".claude/projects/p2").mkdir(parents=True)  # no memory dir
    (home / ".claude/projects/p3").mkdir(parents=True)
    (home / ".claude/projects/p3/memory").write_text("a file, not a dir", encoding="utf-8")

    assert default_memory_dirs() == [home / ".claude/projects/p1/memory"]


def test_scan_memory_dirs_takes_only_direct_children(tmp_path: Path) -> None:
    memory = tmp_path / "proj/memory"
    memory.mkdir(parents=True)
    (memory / "a.md").write_text("x", encoding="utf-8")
    (memory / "a.txt").write_text("y", encoding="utf-8")
    (memory / "sub").mkdir()
    (memory / "sub" / "b.md").write_text("z", encoding="utf-8")

    scanned = scan_memory_dirs([memory])

    assert scanned == {"proj/a.md": memory / "a.md"}


# ---------------------------------------------------------------- criterion 1: creator + edits


def test_created_by_and_edits(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    db = make_db(
        tmp_path,
        [
            (
                "ses-a",
                "2026-08-01",
                [
                    w(A_PATH, "opus"),
                    e(A_PATH, "sonnet"),
                    e(A_PATH, "sonnet"),
                    e(A_PATH, "opus"),
                ],
            )
        ],
    )

    (file,) = _run_json(db, capsys=capsys)["files"]

    assert file["key"] == "p/a.md"
    assert file["created_by"] == "opus"
    assert file["edits"] == [
        {"model": "sonnet", "count": 2},
        {"model": "opus", "count": 1},
    ]
    assert file["n_writes"] == 4
    assert file["first_ts"] == "2026-08-01T10:00:00"
    assert file["last_ts"] == "2026-08-01T10:00:03"


def test_later_writes_count_as_edits(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    db = make_db(
        tmp_path,
        [("ses-a", "2026-08-01", [w(A_PATH, "opus"), w(A_PATH, "sonnet"), e(A_PATH, "sonnet")])],
    )

    (file,) = _run_json(db, capsys=capsys)["files"]

    assert file["created_by"] == "opus"
    assert file["edits"] == [{"model": "sonnet", "count": 2}]
    assert file["n_writes"] == 3


# ---------------------------------------------------------------- criterion 2: edit-first file


def test_first_call_edit_means_unknown_creator(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db = make_db(
        tmp_path,
        [("ses-a", "2026-08-01", [e(A_PATH, "sonnet"), w(A_PATH, "opus"), e(A_PATH, "sonnet")])],
    )

    (file,) = _run_json(db, capsys=capsys)["files"]

    assert file["created_by"] == "unknown"
    assert file["edits"] == [
        {"model": "sonnet", "count": 2},
        {"model": "opus", "count": 1},
    ]
    assert file["n_writes"] == 3


# ---------------------------------------------------------------- criterion 3: failed calls


def test_error_and_denied_writes_are_ignored(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db = make_db(
        tmp_path,
        [
            (
                "ses-a",
                "2026-08-01",
                [
                    w(A_PATH, "opus"),
                    w(A_PATH, "sonnet", is_error=True),
                    e(A_PATH, "sonnet", denied=True),
                ],
            )
        ],
    )

    (file,) = _run_json(db, capsys=capsys)["files"]

    assert file["created_by"] == "opus"
    assert file["edits"] == []
    assert file["n_writes"] == 1


# ---------------------------------------------------------------- criterion 4: one key per file


def test_windows_and_posix_prefixes_are_one_key(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db = make_db(
        tmp_path,
        [
            (
                "ses-a",
                "2026-08-01",
                [
                    w("/home/u/.claude/projects/p/memory/a.md", "opus"),
                    e("C:\\Users\\u\\.claude\\projects\\p\\memory\\a.md", "sonnet"),
                ],
            )
        ],
    )

    (file,) = _run_json(db, capsys=capsys)["files"]

    assert file["key"] == "p/a.md"
    assert file["created_by"] == "opus"
    assert file["edits"] == [{"model": "sonnet", "count": 1}]
    assert file["n_writes"] == 2
    # The most recent absolute path seen is the Windows one.
    assert file["path"] == "C:\\Users\\u\\.claude\\projects\\p\\memory\\a.md"


# ---------------------------------------------------------------- criterion 5: non-memory paths


def test_non_memory_paths_are_not_counted(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db = make_db(
        tmp_path,
        [
            (
                "ses-a",
                "2026-08-01",
                [
                    w("/repo/docs/memory/a.md", "opus"),
                    w("/home/u/.claude/projects/p/memory/sub/a.md", "opus"),
                    w("/home/u/.claude/projects/p/memory/a.txt", "opus"),
                ],
            )
        ],
    )

    assert _run_json(db, capsys=capsys)["files"] == []


# ---------------------------------------------------------------- criterion 6: unknown models


def test_synthetic_and_null_models_are_unknown(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db = make_db(
        tmp_path,
        [("ses-a", "2026-08-01", [w(A_PATH, "<synthetic>"), e(A_PATH, None)])],
    )

    (file,) = _run_json(db, capsys=capsys)["files"]

    assert file["created_by"] == "unknown"
    assert file["edits"] == [{"model": "unknown", "count": 1}]
    assert file["n_writes"] == 2


def test_synthetic_creator_keeps_edits_after_creation(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db = make_db(
        tmp_path,
        [("ses-a", "2026-08-01", [w(A_PATH, "<synthetic>"), e(A_PATH, "opus")])],
    )

    (file,) = _run_json(db, capsys=capsys)["files"]

    assert file["created_by"] == "unknown"
    assert file["edits"] == [{"model": "opus", "count": 1}]
    assert file["n_writes"] == 2


def test_multiedit_is_a_write_tool(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    db = make_db(
        tmp_path,
        [
            (
                "ses-a",
                "2026-08-01",
                [
                    m(A_PATH, "sonnet"),  # first call is a MultiEdit: predates the corpus
                    w(A_PATH, "opus"),
                ],
            ),
            (
                "ses-b",
                "2026-08-02",
                [
                    w(B_PATH, "opus"),
                    m(B_PATH, "sonnet"),
                ],
            ),
        ],
    )

    files = _by_key(_run_json(db, capsys=capsys))

    assert files["p/a.md"]["created_by"] == "unknown"
    assert files["p/a.md"]["edits"] == [
        {"model": "opus", "count": 1},
        {"model": "sonnet", "count": 1},
    ]
    assert files["p/b.md"]["created_by"] == "opus"
    assert files["p/b.md"]["edits"] == [{"model": "sonnet", "count": 1}]


# ---------------------------------------------------------------- criterion 7: unattributed


def test_unattributed_files(home: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    memory = home / ".claude/projects/p/memory"
    memory.mkdir(parents=True)
    (memory / "b.md").write_text("no writes", encoding="utf-8")
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [w(A_PATH, "opus")])])

    payload = _run_json(db, capsys=capsys)

    # b.md has no recorded write and appears; a.md has one and does not.
    assert payload["unattributed"] == ["p/b.md"]
    assert "p/a.md" not in payload["unattributed"]


def test_unattributed_ignores_the_since_window(
    home: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    memory = home / ".claude/projects/p/memory"
    memory.mkdir(parents=True)
    (memory / A_NAME).write_text("written before the window", encoding="utf-8")
    (memory / "c.md").write_text("no writes", encoding="utf-8")
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [w(A_PATH, "opus")])])

    payload = _run_json(db, "--since", "2026-08-10", capsys=capsys)

    # a.md was written only before the window: it drops out of `files`, but it
    # still has a writer, so it must not be reported as unattributed.
    assert payload["files"] == []
    assert payload["unattributed"] == ["p/c.md"]


def test_unattributed_ignores_the_model_filter(
    home: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    memory = home / ".claude/projects/p/memory"
    memory.mkdir(parents=True)
    (memory / A_NAME).write_text("written by another model", encoding="utf-8")
    (memory / "c.md").write_text("no writes", encoding="utf-8")
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [w(A_PATH, "opus")])])

    by_model = _run_json(db, "--model", "claude-9", capsys=capsys)
    by_model_and_window = _run_json(
        db, "--since", "2026-08-10", "--model", "claude-9", capsys=capsys
    )

    # The filter hides a.md from `files`, but its writer is in the DB: it is
    # not "cannot explain", with or without a window on top.
    assert by_model["files"] == []
    assert by_model["unattributed"] == ["p/c.md"]
    assert by_model_and_window["files"] == []
    assert by_model_and_window["unattributed"] == ["p/c.md"]


def test_explicit_memory_dir_replaces_the_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    custom = tmp_path / "notes/memory"
    custom.mkdir(parents=True)
    (custom / "x.md").write_text("hi", encoding="utf-8")
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [w(A_PATH, "opus")])])

    payload = _run_json(db, "--memory-dir", str(custom), capsys=capsys)

    assert payload["unattributed"] == ["notes/x.md"]


# ---------------------------------------------------------------- criterion 8: bash mentions


def test_bash_mentions(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    db = make_db(
        tmp_path,
        [
            (
                "ses-a",
                "2026-08-01",
                [
                    w(A_PATH, "opus"),
                    # File name + /memory/ -> counts (a read through the shell).
                    bash(f"cat /home/u/.claude/projects/p/memory/{A_NAME}", "sonnet"),
                    # File name only, no /memory/ -> does not count.
                    bash(f"ls {A_NAME}", "sonnet"),
                    # /memory/ only, different file -> does not count.
                    bash("cat /home/u/.claude/projects/p/memory/other.md", "sonnet"),
                    # Any outcome counts, error included.
                    bash(
                        f"grep x /home/u/.claude/projects/p/memory/{A_NAME}",
                        "sonnet",
                        is_error=True,
                    ),
                ],
            )
        ],
    )

    (file,) = _run_json(db, capsys=capsys)["files"]

    assert file["bash_mentions"] == 2


# ---------------------------------------------------------------- criterion 9: filters, json


def test_model_filter_lists_only_matching_files(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db = make_db(
        tmp_path,
        [
            (
                "ses-a",
                "2026-08-01",
                [w(A_PATH, "opus"), e(A_PATH, "sonnet"), w(B_PATH, "sonnet")],
            )
        ],
    )

    by_sonnet = _by_key(_run_json(db, "--model", "sonnet", capsys=capsys))
    by_opus = _by_key(_run_json(db, "--model", "opus", capsys=capsys))
    by_other = _run_json(db, "--model", "claude-9", capsys=capsys)

    # sonnet created b.md and edited a.md; opus only created a.md.
    assert set(by_sonnet) == {"p/a.md", "p/b.md"}
    assert set(by_opus) == {"p/a.md"}
    assert by_other["files"] == []


def test_summary_counts_match_the_files(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db = make_db(
        tmp_path,
        [
            (
                "ses-a",
                "2026-08-01",
                [w(A_PATH, "opus"), e(A_PATH, "sonnet"), w(B_PATH, "sonnet")],
            )
        ],
    )

    payload = _run_json(db, capsys=capsys)

    # Writes: opus wrote a.md once (creating call) and sonnet edited a.md once
    # and wrote b.md.  Created: opus made a.md, sonnet made b.md.  Touched:
    # opus only a.md, sonnet both.
    assert payload["summary"] == [
        {"model": "opus", "files_created": 1, "writes": 1, "files_touched": 1},
        {"model": "sonnet", "files_created": 1, "writes": 2, "files_touched": 2},
    ]
    assert sum(s["files_created"] for s in payload["summary"]) == 2
    assert sum(s["writes"] for s in payload["summary"]) == sum(
        f["n_writes"] for f in payload["files"]
    )


def test_since_and_until_filter_calls(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db = make_db(
        tmp_path,
        [
            (
                "ses-a",
                "2026-08-01",
                [w(A_PATH, "opus"), e(A_PATH, "sonnet"), e(A_PATH, "sonnet")],
            )
        ],
    )

    everything = _run_json(db, capsys=capsys)["files"][0]
    assert everything["created_by"] == "opus"
    assert everything["n_writes"] == 3

    since_edit = _run_json(db, "--since", "2026-08-01T10:00:01", capsys=capsys)["files"][0]
    assert since_edit["created_by"] == "unknown"  # judged within the filtered calls
    assert since_edit["edits"] == [{"model": "sonnet", "count": 2}]
    assert since_edit["n_writes"] == 2

    only_write = _run_json(
        db,
        "--since",
        "2026-08-01T10:00:00",
        "--until",
        "2026-08-01T10:00:00",
        capsys=capsys,
    )["files"][0]
    assert only_write["created_by"] == "opus"
    assert only_write["edits"] == []
    assert only_write["n_writes"] == 1


def test_json_output_shape(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [w(A_PATH, "opus")])])

    payload = _run_json(db, capsys=capsys)

    assert set(payload) == {"summary", "files", "unattributed"}
    (file,) = payload["files"]
    assert set(file) == {
        "key",
        "path",
        "created_by",
        "edits",
        "first_ts",
        "last_ts",
        "n_writes",
        "exists",
        "bash_mentions",
    }


# ---------------------------------------------------------------- output and failure


def test_text_output_layout(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    db = make_db(
        tmp_path,
        [("ses-a", "2026-08-01", [w(A_PATH, "opus"), e(A_PATH, "sonnet"), e(A_PATH, "sonnet")])],
    )

    code = main(["memory-authors", "--db", str(db)])

    out = capsys.readouterr().out
    assert code == 0
    assert "summary:" in out
    assert "model" in out and "files_created" in out
    assert "p/a.md  opus  edits(sonnet\u00d72)  3  2026-08-01 10:00:02" in out
    assert "[missing]" in out  # /home/u/... does not exist in the container
    assert out.rstrip().endswith("(1 file)")


def test_exists_reflects_disk(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    path = tmp_path / ".claude/projects/p/memory/a.md"
    path.parent.mkdir(parents=True)
    path.write_text("hi", encoding="utf-8")
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [w(str(path), "opus")])])

    (file,) = _run_json(db, capsys=capsys)["files"]

    assert file["exists"] is True
    assert file["key"] == "p/a.md"


def test_missing_memory_dir_warns_on_stderr(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [w(A_PATH, "opus")])])
    missing = tmp_path / "no-such-memory"

    code = main(["memory-authors", "--db", str(db), "--memory-dir", str(missing)])

    captured = capsys.readouterr()
    assert code == 0
    assert f"warning: memory dir not found: {missing}" in captured.err


def test_missing_database_is_an_error(tmp_path: Path) -> None:
    out, err = io.StringIO(), io.StringIO()

    code = run(tmp_path / "absent.duckdb", out=out, err=err)

    assert code == 1
    assert "no database" in err.getvalue()
    assert out.getvalue() == ""


def test_run_is_report_only(tmp_path: Path) -> None:
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [w(A_PATH, "opus")])])
    before = {
        p: (p.stat().st_size, p.stat().st_mtime_ns)
        for p in tmp_path.rglob("*")
        if p.is_file()
    }

    code = run(db, out=io.StringIO(), err=io.StringIO())

    after = {
        p: (p.stat().st_size, p.stat().st_mtime_ns)
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    assert code == 0
    assert after == before


def test_no_memory_files_reports_empty(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [])])

    payload = _run_json(db, capsys=capsys)

    assert payload["files"] == []
    assert payload["summary"] == []
    assert payload["unattributed"] == []


# ---------------------------------------------------------------- criterion 10: README


def test_readme_documents_memory_authors_and_its_bash_limit() -> None:
    text = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")

    assert "ashiato memory-authors" in text
    assert "`memory-authors`" in text
    assert "bash_mentions" in text
    assert "not attributed" in text