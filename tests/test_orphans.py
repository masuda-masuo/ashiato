"""Tests for ashiato.orphans: one-off discussion topics that left no trace.

Fixtures are hand-built JSONL transcripts run through the real build pipeline,
one transcript file per session, in a private ``tmp_path`` -- never in
``tests/fixtures/``, which other tests glob recursively.
"""

from __future__ import annotations

import io
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from ashiato.build import build, connect
from ashiato.cli import main
from ashiato.orphans import (
    SessionProse,
    collect_sessions,
    default_sink_dirs,
    find_orphans,
    is_headless,
    load_sinks,
    run,
    strip_harness,
    tokenize,
)

#: Shared filler so a session clears the human-chars threshold.  Every session
#: carries it, so none of its words is ever unique to one session.
PAD = "we were talking through the general shape of things together and going back and forth. "
LONG = PAD * 12  # comfortably over 800 characters


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty HOME, so the default sinks never read the real ~/.claude."""
    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HOME", str(path))
    return path


# ---------------------------------------------------------------- fixture transcripts


def _event(
    role: str,
    text: str,
    *,
    kind: str = "text",
) -> dict[str, Any]:
    """One session event; *kind* is text, tool_result, meta or sidechain."""
    return {"role": role, "text": text, "kind": kind}


def human(text: str, *, kind: str = "text") -> dict[str, Any]:
    return _event("user", text, kind=kind)


def assistant(text: str, *, kind: str = "text") -> dict[str, Any]:
    return _event("assistant", text, kind=kind)


def _records(
    session_id: str,
    day: str,
    events: list[dict[str, Any]],
    entrypoint: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    parent = None
    for index, event in enumerate(events):
        uuid = f"{session_id}-{index}"
        if event["kind"] == "tool_result":
            blocks: list[dict[str, Any]] = [
                {
                    "type": "tool_result",
                    "tool_use_id": f"{uuid}-use",
                    "content": [{"type": "text", "text": event["text"]}],
                    "is_error": False,
                }
            ]
        else:
            blocks = [{"type": "text", "text": event["text"]}]
        record: dict[str, Any] = {
            "type": event["role"],
            "uuid": uuid,
            "parentUuid": parent,
            "sessionId": session_id,
            "timestamp": f"{day}T10:{index // 60:02d}:{index % 60:02d}Z",
            "cwd": f"/work/{session_id}",
            "message": {"role": event["role"], "content": blocks},
        }
        if event["kind"] == "meta":
            record["isMeta"] = True
        if event["kind"] == "sidechain":
            record["isSidechain"] = True
        if entrypoint is not None:
            record["entrypoint"] = entrypoint
        records.append(record)
        parent = uuid
    return records


def make_db(tmp_path: Path, sessions: list[tuple[Any, ...]]) -> Path:
    """Build a DuckDB from ``(session_id, "YYYY-MM-DD", events)`` sessions;
    an optional fourth element sets the session ``entrypoint`` (e.g. ``"sdk-cli"``,
    ``"sdk-py"``)."""
    directory = tmp_path / "transcripts"
    directory.mkdir(exist_ok=True)
    for index, item in enumerate(sessions):
        session_id, day, events = item[0], item[1], item[2]
        entrypoint = item[3] if len(item) > 3 else None
        path = directory / f"{index:02d}-{session_id}.jsonl"
        lines = [
            json.dumps(record)
            for record in _records(session_id, day, events, entrypoint=entrypoint)
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    db_path = tmp_path / "orphans.duckdb"
    build([directory], db_path)
    return db_path


def _run_json(db_path: Path, *args: str, capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    code = main(["orphans", "--db", str(db_path), "--json", *args])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    return json.loads(captured.out)


def _by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {candidate["session_id"]: candidate for candidate in payload["candidates"]}


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Three sessions: A has its own topic, A and B share one, C is filler."""
    return make_db(
        tmp_path,
        [
            (
                "ses-a",
                "2026-08-01",
                [
                    human(LONG + "zorblax zorblax zorblax sharedtopic sharedtopic sharedtopic"),
                    assistant("wibbleframe wibbleframe wibbleframe"),
                ],
            ),
            ("ses-b", "2026-08-02", [human(LONG + "sharedtopic sharedtopic sharedtopic")]),
            ("ses-c", "2026-08-03", [human(LONG)]),
        ],
    )


# ---------------------------------------------------------------- unit: text handling


def test_strip_harness_removes_wrapper_blocks() -> None:
    text = (
        "hello <command-name>/foo</command-name> <command-args>bar</command-args>"
        "<local-command-stdout>out\nput</local-command-stdout> "
        "<system-reminder>\nremind\n</system-reminder>"
        "<bash-input>ls</bash-input><task-notification>n</task-notification> world"
    )
    assert " ".join(strip_harness(text).split()) == "hello world"


def test_strip_harness_keeps_other_tags_and_unclosed_blocks() -> None:
    assert strip_harness("<b>bold</b> <system-reminder>never closed") == (
        "<b>bold</b> <system-reminder>never closed"
    )


def test_strip_harness_is_non_greedy() -> None:
    text = "<bash-stdout>a</bash-stdout> keep <bash-stdout>b</bash-stdout>"
    assert strip_harness(text).strip() == "keep"


def test_tokenize_lowercases_and_finds_latin_katakana_kanji() -> None:
    terms = tokenize("Design memory-server プロトコル 設計 API abc 猫")
    assert terms == ["design", "memory-server", "プロトコル", "設計"]


def test_tokenize_drops_code_identifiers_and_hex_ids() -> None:
    text = (
        "snake_case_name deadbeef0 abcdef1 abcdef01-2345-6789 a1b2c3d4-e5f6 "
        "6f9619ff-8b86-d011-b42d-00c04fc964ff 6F9619FF-8B86-D011-B42D-00C04FC964FF realword"
    )
    assert tokenize(text) == ["realword"]


# ---------------------------------------------------------------- unit: sinks


def test_load_sinks_walks_directories_and_lowercases(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    (root / "sub").mkdir(parents=True)
    (root / "a.md").write_text("Hello WORLD", encoding="utf-8")
    (root / "sub" / "b.txt").write_text("Nested Topic", encoding="utf-8")
    single = tmp_path / "one.md"
    single.write_text("Single File", encoding="utf-8")

    sinks = load_sinks([root, single])

    assert sinks.n_files == 3
    for word in ("hello world", "nested topic", "single file"):
        assert word in sinks.text


def test_load_sinks_skips_junk(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    for skipped in (".git", "node_modules", ".venv", "__pycache__"):
        (root / skipped).mkdir(parents=True)
        (root / skipped / "f.txt").write_text("hiddenword", encoding="utf-8")
    (root / "big.txt").write_text("bigword " * 200_000, encoding="utf-8")  # > 1 MiB
    (root / "bin.dat").write_bytes(b"binaryword\0\0\0")
    (root / "ok.md").write_text("keepword", encoding="utf-8")

    sinks = load_sinks([root])

    assert sinks.n_files == 1
    assert "keepword" in sinks.text
    for word in ("hiddenword", "bigword", "binaryword"):
        assert word not in sinks.text


def test_load_sinks_tolerates_invalid_utf8_and_reports_missing(tmp_path: Path) -> None:
    good = tmp_path / "latin.txt"
    good.write_bytes(b"caf\xe9 fine")
    missing = tmp_path / "nope"

    sinks = load_sinks([good, missing])

    assert sinks.missing == [missing]
    assert sinks.n_files == 1
    assert "fine" in sinks.text


def test_default_sink_dirs_are_existing_memory_dirs(home: Path) -> None:
    (home / ".claude/projects/p1/memory").mkdir(parents=True)
    (home / ".claude/projects/p2").mkdir(parents=True)  # no memory dir
    (home / ".claude/projects/p3").mkdir(parents=True)
    (home / ".claude/projects/p3/memory").write_text("a file, not a dir", encoding="utf-8")

    assert default_sink_dirs() == [home / ".claude/projects/p1/memory"]


# ---------------------------------------------------------------- criterion 1: unique / sink


def test_unique_term_is_orphan_and_shared_term_is_not(
    corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _run_json(
        corpus, "--min-human-chars", "100", "--min-orphans", "1", capsys=capsys
    )

    candidates = _by_id(payload)
    assert set(candidates) == {"ses-a"}  # B and C have nothing unique
    a = candidates["ses-a"]
    assert a["orphan_terms"] == ["wibbleframe", "zorblax"]  # tf tie -> alphabetical
    assert "sharedtopic" not in a["orphan_terms"]
    assert a["n_unique"] == 2
    assert a["n_orphan"] == 2


def test_term_in_a_sink_is_not_an_orphan(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sink = tmp_path / "notes.md"
    sink.write_text("We decided: Wibbleframe is the name.", encoding="utf-8")

    payload = _run_json(corpus, "--sink", str(sink), "--min-orphans", "1", capsys=capsys)

    a = _by_id(payload)["ses-a"]
    assert a["orphan_terms"] == ["zorblax"]
    assert a["n_unique"] == 2
    assert a["n_orphan"] == 1
    assert payload["sink_files"] == 1


def test_session_whose_unique_terms_are_all_persisted_is_not_nominated(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sink = tmp_path / "notes"
    sink.mkdir()
    (sink / "a.md").write_text("zorblax and wibbleframe", encoding="utf-8")

    payload = _run_json(corpus, "--sink", str(sink), "--min-orphans", "1", capsys=capsys)

    assert payload["candidates"] == []


def test_min_tf_is_respected(corpus: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _by_id(_run_json(corpus, "--min-tf", "3", "--min-orphans", "1", capsys=capsys))
    assert (
        _run_json(corpus, "--min-tf", "4", "--min-orphans", "1", capsys=capsys)["candidates"]
        == []
    )


def test_short_session_still_counts_for_document_frequency(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = make_db(
        tmp_path,
        [
            ("ses-a", "2026-08-01", [human(LONG + "quuxify quuxify quuxify")]),
            # Far below --min-human-chars, so never nominated -- but it has the term.
            ("ses-b", "2026-08-02", [human("quuxify")]),
            ("ses-c", "2026-08-03", [human(LONG)]),
        ],
    )

    assert _run_json(db, "--min-orphans", "1", capsys=capsys)["candidates"] == []


def test_assistant_text_counts_as_prose(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = make_db(
        tmp_path,
        [
            ("ses-a", "2026-08-01", [human(LONG), assistant("blorptastic blorptastic blorptastic")]),
            ("ses-b", "2026-08-02", [human(LONG), assistant("blorptastic")]),
            ("ses-c", "2026-08-03", [human(LONG)]),
        ],
    )

    assert _run_json(db, "--min-orphans", "1", capsys=capsys)["candidates"] == []


# ---------------------------------------------------------------- criterion 2: wrappers


def test_harness_wrappers_do_not_count_or_tokenize(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wrapped = (
        "<local-command-stdout>" + "x" * 2000 + " stdoutword stdoutword stdoutword"
        "</local-command-stdout>"
        "<system-reminder>reminderword reminderword reminderword</system-reminder>"
    )
    db = make_db(
        tmp_path,
        [
            ("ses-a", "2026-08-01", [human("short question"), human(wrapped)]),
            ("ses-b", "2026-08-02", [human(LONG)]),
            ("ses-c", "2026-08-03", [human(LONG)]),
        ],
    )

    connection = connect(db, read_only=True)
    try:
        sessions = {s.session_id: s for s in collect_sessions(connection)}
    finally:
        connection.close()
    a = sessions["ses-a"]
    assert a.human_chars == len("short question")
    assert "stdoutword" not in a.terms
    assert "reminderword" not in a.terms
    # ...so the 2,000-character block does not lift it over the threshold.
    assert _run_json(db, "--min-tf", "1", capsys=capsys)["candidates"] == []


def test_wrapper_only_message_is_not_the_first_utterance(tmp_path: Path) -> None:
    db = make_db(
        tmp_path,
        [
            (
                "ses-a",
                "2026-08-01",
                [
                    human("<system-reminder>ignore me</system-reminder>\n"),
                    human("  what do   you\nthink   about a memory MCP?  "),
                    human("a later message"),
                ],
            )
        ],
    )
    connection = connect(db, read_only=True)
    try:
        (session,) = collect_sessions(connection)
    finally:
        connection.close()

    assert session.first_utterance == "what do you think about a memory MCP?"


def test_first_utterance_is_truncated_to_160_chars(tmp_path: Path) -> None:
    db = make_db(tmp_path, [("ses-a", "2026-08-01", [human("word " * 100)])])
    connection = connect(db, read_only=True)
    try:
        (session,) = collect_sessions(connection)
    finally:
        connection.close()

    assert len(session.first_utterance) == 160


# ---------------------------------------------------------------- criterion 3: excluded rows


def test_tool_result_meta_and_sidechain_rows_contribute_nothing(tmp_path: Path) -> None:
    ghost = "ghostterm ghostterm ghostterm"
    db = make_db(
        tmp_path,
        [
            (
                "ses-a",
                "2026-08-01",
                [
                    human("plain question"),
                    human(ghost + "x" * 3000, kind="tool_result"),
                    human(ghost + "y" * 3000, kind="meta"),
                    human(ghost + "z" * 3000, kind="sidechain"),
                    assistant(ghost, kind="sidechain"),
                    assistant(ghost, kind="meta"),
                ],
            )
        ],
    )
    connection = connect(db, read_only=True)
    try:
        (session,) = collect_sessions(connection)
    finally:
        connection.close()

    assert session.human_chars == len("plain question")
    assert "ghostterm" not in session.terms


def test_session_id_fan_out_does_not_double_count(tmp_path: Path) -> None:
    # Two transcript files with one session id: each is its own session and the
    # join on file_path keeps their rows apart.
    db = make_db(
        tmp_path,
        [
            ("same-id", "2026-08-01", [human("first file")]),
            ("same-id", "2026-08-02", [human("second file text")]),
        ],
    )
    connection = connect(db, read_only=True)
    try:
        sessions = collect_sessions(connection)
    finally:
        connection.close()

    assert sorted(s.human_chars for s in sessions) == [len("first file"), len("second file text")]


# ---------------------------------------------------------------- criterion 4: identifiers


def test_identifier_and_hex_tokens_are_never_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    junk = (
        "some_identifier " * 3
        + "deadbeef0 " * 3
        + "abcdef01-2345-6789 " * 3
        + "6f9619ff-8b86-d011-b42d-00c04fc964ff " * 3
        + "cafebabe123 " * 3
    )
    db = make_db(
        tmp_path,
        [
            ("ses-a", "2026-08-01", [human(LONG + junk + "realtopic realtopic realtopic")]),
            ("ses-b", "2026-08-02", [human(LONG)]),
            ("ses-c", "2026-08-03", [human(LONG)]),
        ],
    )

    a = _by_id(_run_json(db, "--min-orphans", "1", capsys=capsys))["ses-a"]

    assert a["orphan_terms"] == ["realtopic"]
    assert a["n_unique"] == 1


# ---------------------------------------------------------------- criterion 5: window


def test_since_does_not_make_a_shared_term_unique(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = make_db(
        tmp_path,
        [
            ("ses-old", "2026-08-01", [human(LONG + "oldtopic oldtopic oldtopic")]),
            ("ses-new", "2026-08-10", [human(LONG + "oldtopic oldtopic oldtopic")]),
            ("ses-c", "2026-08-11", [human(LONG)]),
        ],
    )

    payload = _run_json(db, "--since", "2026-08-05", "--min-orphans", "1", capsys=capsys)

    assert payload["candidates"] == []
    # The corpus is still the whole database.
    assert payload["sessions_with_prose"] == 3


def test_since_and_until_window_is_inclusive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = make_db(
        tmp_path,
        [
            ("ses-1", "2026-08-01", [human(LONG + "alphaword alphaword alphaword")]),
            ("ses-2", "2026-08-02", [human(LONG + "bravoword bravoword bravoword")]),
            ("ses-3", "2026-08-03", [human(LONG + "gammaword gammaword gammaword")]),
        ],
    )

    everything = set(_by_id(_run_json(db, "--min-orphans", "1", capsys=capsys)))
    from_second = set(
        _by_id(_run_json(db, "--since", "2026-08-02T10:00:00", "--min-orphans", "1", capsys=capsys))
    )
    only_second = set(
        _by_id(
            _run_json(
                db,
                "--since",
                "2026-08-02T10:00:00",
                "--until",
                "2026-08-02T10:00:00",
                "--min-orphans",
                "1",
                capsys=capsys,
            )
        )
    )

    assert everything == {"ses-1", "ses-2", "ses-3"}
    assert from_second == {"ses-2", "ses-3"}
    assert only_second == {"ses-2"}


# ---------------------------------------------------------------- criterion 6: sinks on the CLI


def test_missing_sink_warns_on_stderr_and_exits_zero(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "no-such-notes"

    code = main(["orphans", "--db", str(corpus), "--sink", str(missing), "--min-orphans", "1"])

    captured = capsys.readouterr()
    assert code == 0
    assert f"warning: sink not found: {missing}" in captured.err
    assert "zorblax" in captured.out  # nothing loaded, so every unique term is an orphan


def test_no_sink_text_notes_it_on_stderr(
    corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["orphans", "--db", str(corpus), "--no-default-sinks", "--min-orphans", "1"])

    captured = capsys.readouterr()
    assert code == 0
    assert "no sink text loaded" in captured.err
    assert "zorblax" in captured.out
    assert "wibbleframe" in captured.out


def test_default_sinks_are_the_memory_dirs(
    corpus: Path, home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    memory = home / ".claude/projects/p1/memory"
    memory.mkdir(parents=True)
    (memory / "note.md").write_text("zorblax wibbleframe", encoding="utf-8")

    with_default = _run_json(corpus, "--min-orphans", "1", capsys=capsys)
    without_default = _run_json(
        corpus, "--no-default-sinks", "--min-orphans", "1", capsys=capsys
    )
    explicit_only = _run_json(corpus, "--sink", str(home), "--min-orphans", "1", capsys=capsys)

    assert with_default["candidates"] == []
    assert with_default["sink_files"] == 1
    assert _by_id(without_default)["ses-a"]["n_orphan"] == 2
    assert without_default["sink_files"] == 0
    # An explicit --sink replaces the default rather than adding to it.
    assert explicit_only["sink_files"] == 1
    assert explicit_only["candidates"] == []


def test_explicit_sink_replaces_the_default(
    corpus: Path, home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    memory = home / ".claude/projects/p1/memory"
    memory.mkdir(parents=True)
    (memory / "note.md").write_text("zorblax wibbleframe", encoding="utf-8")
    other = tmp_path / "other.md"
    other.write_text("nothing relevant", encoding="utf-8")

    payload = _run_json(corpus, "--sink", str(other), "--min-orphans", "1", capsys=capsys)

    assert _by_id(payload)["ses-a"]["n_orphan"] == 2


# ---------------------------------------------------------------- criterion 7: threshold, ranking


def test_sessions_below_min_human_chars_are_not_nominated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    topic = "zorblax zorblax zorblax"
    db = make_db(
        tmp_path,
        [
            ("ses-a", "2026-08-01", [human(PAD * 3 + topic)]),  # a few hundred chars
            ("ses-b", "2026-08-02", [human(LONG)]),
            ("ses-c", "2026-08-03", [human(LONG)]),
        ],
    )

    assert _run_json(db, "--min-orphans", "1", capsys=capsys)["candidates"] == []
    lowered = _run_json(db, "--min-human-chars", "100", "--min-orphans", "1", capsys=capsys)
    assert set(_by_id(lowered)) == {"ses-a"}
    assert (
        _run_json(db, "--min-human-chars", "100000", "--min-orphans", "1", capsys=capsys)[
            "candidates"
        ]
        == []
    )


def _topics(prefix: str, count: int) -> str:
    """*count* distinct unique terms, each three times."""
    letters = "ghijklmnopqrstuvwxyz"
    return " ".join(f"{prefix}{letters[i]}term " * 3 for i in range(count))


@pytest.fixture
def ranked(tmp_path: Path) -> Path:
    """Orphan counts 3/1/1/1; the ones with 1 tie on chars or differ on chars."""
    return make_db(
        tmp_path,
        [
            ("ses-many", "2026-08-01", [human(LONG + _topics("mm", 3))]),
            ("ses-long", "2026-08-02", [human(LONG + PAD * 5 + _topics("ll", 1))]),
            ("ses-tie-b", "2026-08-03", [human(LONG + _topics("bb", 1))]),
            ("ses-tie-a", "2026-08-04", [human(LONG + _topics("aa", 1))]),
            ("ses-filler", "2026-08-05", [human(LONG)]),
        ],
    )


def test_ranking_is_density_then_orphans_then_chars_then_session_id(
    ranked: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _run_json(ranked, "--min-orphans", "1", capsys=capsys)

    ids = [c["session_id"] for c in payload["candidates"]]
    # Density (orphans per total terms) leads: the terse sessions outrank the
    # padded ses-long despite having fewer raw orphans, and ses-tie-a /
    # ses-tie-b tie on everything and differ only in their session id.
    assert ids == ["ses-many", "ses-tie-a", "ses-tie-b", "ses-long"]
    assert (
        _run_json(ranked, "--min-orphans", "1", capsys=capsys) == payload
    )  # deterministic


def test_limit_caps_and_zero_means_all(ranked: Path, capsys: pytest.CaptureFixture[str]) -> None:
    two = _run_json(ranked, "--limit", "2", "--min-orphans", "1", capsys=capsys)
    all_of_them = _run_json(ranked, "--limit", "0", "--min-orphans", "1", capsys=capsys)

    assert [c["session_id"] for c in two["candidates"]] == ["ses-many", "ses-tie-a"]
    assert len(all_of_them["candidates"]) == 4


def test_orphan_terms_are_top_ten_by_frequency(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    words = [f"topic{chr(ord('a') + i)}word" for i in range(12)]
    # word i occurs 3 + (11 - i) times, so the last two are the rarest.
    body = " ".join(" ".join([word] * (3 + 11 - i)) for i, word in enumerate(words))
    db = make_db(
        tmp_path,
        [
            ("ses-a", "2026-08-01", [human(LONG + body)]),
            ("ses-b", "2026-08-02", [human(LONG)]),
            ("ses-c", "2026-08-03", [human(LONG)]),
        ],
    )

    a = _by_id(_run_json(db, capsys=capsys))["ses-a"]

    assert a["n_orphan"] == 12
    assert a["orphan_terms"] == words[:10]


# ---------------------------------------------------------------- criterion 8: output


JSON_FIELDS = {
    "session_id",
    "started_at",
    "project_dir",
    "n_tool_calls",
    "human_chars",
    "n_unique",
    "n_orphan",
    "orphan_density",
    "orphan_terms",
    "first_utterance",
}


def test_json_output_has_the_listed_fields_and_header(
    corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _run_json(corpus, "--since", "2026-07-01", "--min-orphans", "1", capsys=capsys)

    (candidate,) = payload["candidates"]
    assert set(candidate) == JSON_FIELDS
    assert candidate["session_id"] == "ses-a"
    assert candidate["started_at"].startswith("2026-08-01")
    assert candidate["n_tool_calls"] == 0
    assert candidate["human_chars"] > 800
    assert candidate["first_utterance"].startswith("we were talking")
    assert candidate["n_orphan"] == 2
    assert candidate["orphan_density"] == pytest.approx(2 / 129 * 1000)
    assert payload["sessions_with_prose"] == 3
    assert payload["sink_files"] == 0
    assert payload["thresholds"] == {
        "min_tf": 3,
        "min_human_chars": 800,
        "min_orphans": 1,
        "include_headless": False,
        "limit": 20,
        "since": "2026-07-01T00:00:00",
        "until": None,
    }


def test_text_output_has_header_and_one_block_per_candidate(
    corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["orphans", "--db", str(corpus), "--no-default-sinks", "--min-orphans", "1"])

    out = capsys.readouterr().out
    assert code == 0
    header, *_ = out.splitlines()
    assert "3 sessions with prose" in header
    assert "0 files" in header
    assert "min-tf=3" in header
    assert "min-human-chars=800" in header
    assert "min-orphans=1" in header
    assert "include-headless=no" in header
    assert out.count("session ses-") == 1
    assert "terms: wibbleframe, zorblax" in out
    assert "density=15.5" in out
    assert "first: we were talking" in out
    assert out.rstrip().endswith("(1 candidate)")


def test_no_candidates_still_exits_zero(
    corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["orphans", "--db", str(corpus), "--no-default-sinks", "--min-tf", "99"])

    captured = capsys.readouterr()
    assert code == 0
    assert "(0 candidates)" in captured.out


def test_sessions_without_prose_are_not_in_the_corpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = make_db(
        tmp_path,
        [
            ("ses-a", "2026-08-01", [human(LONG)]),
            ("ses-empty", "2026-08-02", [human("<system-reminder>x</system-reminder>")]),
            ("ses-meta", "2026-08-03", [human("meta only", kind="meta")]),
        ],
    )

    assert _run_json(db, capsys=capsys)["sessions_with_prose"] == 1


# ---------------------------------------------------------------- misc: failure, report-only


def test_missing_database_is_an_error(tmp_path: Path) -> None:
    out, err = io.StringIO(), io.StringIO()

    code = run(tmp_path / "absent.duckdb", out=out, err=err)

    assert code == 1
    assert "no database" in err.getvalue()
    assert out.getvalue() == ""


def test_run_is_report_only(corpus: Path, tmp_path: Path) -> None:
    sink = tmp_path / "notes"
    sink.mkdir()
    (sink / "a.md").write_text("content", encoding="utf-8")
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}

    code = run(corpus, sinks=[sink], out=io.StringIO(), err=io.StringIO())

    after = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    assert code == 0
    assert after == before


def test_find_orphans_is_pure_over_prepared_sessions() -> None:
    from ashiato.orphans import SessionProse

    def session(file_path: str, day: int, text: str) -> SessionProse:
        prose = SessionProse(file_path, file_path, datetime(2026, 8, day), None, 0)
        prose.add("user", text)
        return prose

    sessions = [
        session("a", 1, "x" * 900 + " uniqueword uniqueword uniqueword"),
        session("b", 2, "y" * 900),
    ]

    (candidate,) = find_orphans(sessions, "", min_orphans=1)
    assert candidate.orphan_terms == ["uniqueword"]
    assert find_orphans(sessions, "mentions uniqueword here", min_orphans=1) == []
# ---------------------------------------------------------------- issue #45: precision


def test_mixed_letter_digit_tokens_are_dropped() -> None:
    text = (
        "bk7tgrw6i c1bf1 x000100000005f1de urllib3 fargate bench-10k "
        "adr-0010 認知負荷"
    )
    assert tokenize(text) == ["fargate", "bench-10k", "adr-0010", "認知負荷"]


def test_min_orphans_default_and_flag(
    corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The corpus's ses-a has exactly two orphan terms: below the default 3.
    assert (
        _run_json(corpus, "--min-human-chars", "100", capsys=capsys)["candidates"] == []
    )
    one = _run_json(
        corpus, "--min-human-chars", "100", "--min-orphans", "1", capsys=capsys
    )
    assert set(_by_id(one)) == {"ses-a"}


def test_ranking_is_by_orphan_density() -> None:
    def session(file_path: str, day: int, words: list[str], total_terms: int) -> SessionProse:
        prose = SessionProse(file_path, file_path, datetime(2026, 8, day), None, 0)
        prose.has_prose = True
        prose.human_chars = 900
        prose.terms = Counter({word: 3 for word in words})
        prose.total_terms = total_terms
        return prose

    # A small discussion outranks a huge work session despite fewer orphans.
    small = session("small", 1, [f"small{i}word" for i in range(5)], 200)
    large = session("large", 2, [f"large{i}word" for i in range(40)], 20_000)

    candidates = find_orphans([small, large], "")

    assert [c.session_id for c in candidates] == ["small", "large"]
    assert candidates[0].orphan_density == pytest.approx(5 / 200 * 1000)
    assert candidates[1].orphan_density == pytest.approx(40 / 20_000 * 1000)
    assert candidates[0].orphan_density > candidates[1].orphan_density


def test_is_headless_recognises_sdk_variants() -> None:
    for entrypoint in ("sdk-cli", "sdk-py", "sdk-ts"):
        assert is_headless(entrypoint), entrypoint
    for entrypoint in (None, "cli", "claude-desktop"):
        assert not is_headless(entrypoint), entrypoint


def test_headless_sessions_are_excluded_but_count_for_df(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = make_db(
        tmp_path,
        [
            (
                "ses-a",
                "2026-08-01",
                [
                    human(
                        LONG
                        + "blorptastic blorptastic blorptastic quuxify quuxify quuxify "
                        "wibbleframe wibbleframe wibbleframe "
                        "alphatopic alphatopic alphatopic betatopic betatopic betatopic "
                        "gammatopic gammatopic gammatopic"
                    )
                ],
                "sdk-cli",
            ),
            (
                "ses-b",
                "2026-08-02",
                [human(LONG + "sharedterm sharedterm sharedterm")],
                "sdk-cli",
            ),
            (
                "ses-c",
                "2026-08-03",
                [
                    human(
                        LONG
                        + "alphatopic alphatopic alphatopic betatopic betatopic betatopic "
                        "gammatopic gammatopic gammatopic sharedterm sharedterm sharedterm"
                    )
                ],
            ),
            (
                "ses-d",
                "2026-08-04",
                [
                    human(
                        LONG
                        + "soloterm soloterm soloterm anotherword anotherword anotherword "
                        "thirdword thirdword thirdword"
                    )
                ],
            ),
            (
                "ses-e",
                "2026-08-05",
                [human(LONG + "zetatopic zetatopic zetatopic")],
            ),
            (
                "ses-f",
                "2026-08-06",
                [human(LONG + "zetatopic zetatopic zetatopic")],
                "sdk-py",
            ),
        ],
    )

    default = _run_json(db, capsys=capsys)
    # ses-d only: ses-a's unique terms live in a headless session, and the
    # sdk-cli ses-a/ses-b (and the sdk-py ses-f) still make the shared terms
    # non-unique for the cli ses-c and ses-e.
    assert set(_by_id(default)) == {"ses-d"}

    # Even with --min-orphans 1, ses-c and ses-e stay out: their only
    # candidate terms (alphatopic/betatopic/gammatopic/sharedterm for ses-c,
    # zetatopic for ses-e) are shared with headless sessions, and headless
    # sessions count toward document frequency.  If they were dropped from
    # the df corpus, all of those terms would become unique orphans and these
    # assertions would fail.
    sensitive = _run_json(db, "--min-orphans", "1", capsys=capsys)
    assert set(_by_id(sensitive)) == {"ses-d"}

    # Even when headless sessions are nominatable, ses-c and ses-e still are
    # not: their terms stay non-unique (df 2) because df counts all sessions.
    both = _run_json(db, "--include-headless", "--min-orphans", "1", capsys=capsys)
    assert set(_by_id(both)) == {"ses-a", "ses-d"}

    included = _run_json(db, "--include-headless", capsys=capsys)
    assert set(_by_id(included)) == {"ses-a", "ses-d"}
    assert included["thresholds"]["include_headless"] is True


def test_total_terms_is_the_sum_of_filtered_term_counts(tmp_path: Path) -> None:
    db = make_db(
        tmp_path,
        [
            (
                "ses-a",
                "2026-08-01",
                [
                    human(LONG + "realtopic realtopic realtopic bk7tgrw6i urllib3"),
                    assistant("fargate bench-10k"),
                ],
            )
        ],
    )
    connection = connect(db, read_only=True)
    try:
        (session,) = collect_sessions(connection)
    finally:
        connection.close()

    # The mixed letter+digit ids are dropped before counting, so the sum of
    # the kept term counts is exactly the density denominator.
    assert session.total_terms == sum(session.terms.values())
    assert session.terms["realtopic"] == 3
    assert "bk7tgrw6i" not in session.terms
    assert "urllib3" not in session.terms
