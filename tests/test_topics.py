"""Tests for ashiato.topics: a deterministic per-session topic outline (issue #51).

A session's stored title is generated once and never updated, so ``topics``
derives an outline from the session's own words: exchanges (human turn plus
the following assistant text), tf-idf weights over the whole corpus, and
TextTiling-style boundaries (a gap whose cosine similarity is a local minimum
below ``mean - 0.5 * sd``).  Everything is deterministic -- the same bytes
always produce the same outline, and nothing is written.

The contract decisions frozen here, where the brief leaves room:

* an exchange is opened by the first non-empty human row (after harness
  stripping) and absorbs every following assistant row, up to the next human
  row; assistant text before any human row is ignored;
* rows are deduplicated by the ``uuid`` in their ``raw`` JSON (first
  occurrence wins), so a resumed session that re-persists the same messages
  does not duplicate exchanges;
* compaction summaries (``"isCompactSummary": true`` in ``raw``) are
  excluded exactly as ``orphans`` excludes them, via the shared
  ``ashiato.orphans.is_compaction_summary`` helper;
* the document frequency corpus is ``ashiato.orphans.collect_sessions`` over
  the whole database, and terms whose document frequency exceeds 30% of
  sessions are dropped before weighting;
* a gap is a boundary only when it has a full ``window`` of exchanges on each
  side, and a session with fewer than ``2 * window`` exchanges is one segment;
* the header title is the latest ``custom-title`` if any, else the latest
  ``ai-title`` (both read defensively from ``raw``);
* ``start`` / ``end`` are 0-based inclusive exchange indices.

Fixtures are synthetic transcripts built through the real ``build`` pipeline
into a private ``tmp_path`` database; the host production database is never
touched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ashiato.build import build, connect
from ashiato.cli import main
from ashiato.topics import outline

#: The vocabulary of the first half of the outline session.
A_WORDS = ["allowlist", "sidecar", "mitmproxy", "egress"]

#: The vocabulary of the second half.
B_WORDS = ["lint", "precommit", "typecheck", "dogfood"]

#: The outline session: four exchanges about the egress proxy, then four
#: about lint/typecheck dogfooding.  The two vocabularies are disjoint, so the
#: gap between exchange 3 and 4 is a clean local minimum.
A_EXCHANGES: list[tuple[str, str]] = [
    ("mitmproxy needs an egress allowlist for the sidecar",
     "the allowlist names every host the sidecar may reach"),
    ("which hosts should the egress allowlist permit",
     "mitmproxy checks the sidecar request against the allowlist"),
    ("mitmproxy injects a certificate for the sidecar",
     "the sidecar trusts mitmproxy once the egress allowlist allows it"),
    ("the egress path travels through mitmproxy and the allowlist",
     "mitmproxy and the sidecar agree on the allowlist"),
]
B_EXCHANGES: list[tuple[str, str]] = [
    ("run lint on the new code",
     "lint passes once the precommit hook runs"),
    ("dogfood the typecheck before every commit",
     "typecheck catches what lint misses"),
    ("the precommit hook runs lint and typecheck",
     "dogfooding the loop shows the precommit gaps"),
    ("make lint and typecheck part of the dogfood loop",
     "the precommit hook runs both before every push"),
]

#: Shared filler so filler sessions have prose; the words appear in every
#: session, so none of them is ever distinctive.
PAD = "we were talking through the general shape of things together and going back and forth. "
LONG = PAD * 12


# ---------------------------------------------------------------- fixture builders


def line(
    role: str,
    uuid: str,
    session: str,
    ts: str,
    text: str,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One transcript record; *extra* merges in (compaction flags, titles...)."""
    record: dict[str, Any] = {
        "type": role,
        "uuid": uuid,
        "parentUuid": None,
        "sessionId": session,
        "timestamp": ts,
        "cwd": f"/work/{session}",
        "message": {"role": role, "content": [{"type": "text", "text": text}]},
    }
    if extra:
        record.update(extra)
    return record


def exchange_records(
    session: str, day: str, exchanges: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    """Human + assistant records for each exchange, with sequential uuids/timestamps."""
    records: list[dict[str, Any]] = []
    index = 0
    for human_text, assistant_text in exchanges:
        ts = f"{day}T10:{index // 60:02d}:{index % 60:02d}Z"
        records.append(line("user", f"{session}-{index}", session, ts, human_text))
        index += 1
        ts = f"{day}T10:{index // 60:02d}:{index % 60:02d}Z"
        records.append(line("assistant", f"{session}-{index}", session, ts, assistant_text))
        index += 1
    return records


def title_record(session: str, ts: str, kind: str, title: str) -> dict[str, Any]:
    """An ``ai-title`` or ``custom-title`` event; the value rides in ``raw`` only."""
    key = "aiTitle" if kind == "ai-title" else "title"
    return {"type": kind, "sessionId": session, "timestamp": ts, key: title}


def make_db(
    tmp_path: Path, sessions: list[tuple[str, str, list[dict[str, Any]]]]
) -> Path:
    """Build a DuckDB from ``(session_id, "YYYY-MM-DD", records)`` sessions."""
    directory = tmp_path / "transcripts"
    directory.mkdir(parents=True, exist_ok=True)
    for index, (session_id, _day, records) in enumerate(sessions):
        path = directory / f"{index:02d}-{session_id}.jsonl"
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
    db_path = tmp_path / "topics.duckdb"
    build([directory], db_path)
    return db_path


def _outline_of(db: Path, session_id: str) -> dict[str, Any]:
    connection = connect(db, read_only=True)
    try:
        return outline(connection, session_id)
    finally:
        connection.close()


def _cli(db: Path, *args: str, capsys: pytest.CaptureFixture[str]) -> str:
    code = main(["topics", "--db", str(db), *args])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    return captured.out


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """ses-outline (two vocabularies, 8 exchanges) plus four filler sessions."""
    return make_db(
        tmp_path,
        [
            (
                "ses-outline",
                "2026-08-01",
                exchange_records("ses-outline", "2026-08-01", A_EXCHANGES + B_EXCHANGES),
            ),
            ("ses-f1", "2026-08-02", [line("user", "u1", "ses-f1", "2026-08-02T10:00:00Z", LONG)]),
            ("ses-f2", "2026-08-03", [line("user", "u1", "ses-f2", "2026-08-03T10:00:00Z", LONG)]),
            ("ses-f3", "2026-08-04", [line("user", "u1", "ses-f3", "2026-08-04T10:00:00Z", LONG)]),
            ("ses-f4", "2026-08-05", [line("user", "u1", "ses-f4", "2026-08-05T10:00:00Z", LONG)]),
        ],
    )


# ---------------------------------------------------------------- criterion 1: segmentation


def test_two_vocabularies_segment_at_the_switch(corpus: Path) -> None:
    payload = _outline_of(corpus, "ses-outline")

    assert [segment["start"] for segment in payload["segments"]] == [0, 4]
    assert [segment["end"] for segment in payload["segments"]] == [3, 7]
    assert [segment["n_exchanges"] for segment in payload["segments"]] == [4, 4]

    first, second = payload["segments"]
    assert set(first["terms"]) >= set(A_WORDS)
    assert set(second["terms"]) >= set(B_WORDS)
    # The vocabularies are disjoint: neither half leaks into the other's labels.
    assert not (set(first["terms"]) & set(B_WORDS))
    assert not (set(second["terms"]) & set(A_WORDS))
    assert first["opening"] == "mitmproxy needs an egress allowlist for the sidecar"
    assert second["opening"] == "run lint on the new code"


def test_exchange_boundaries_and_timestamps(corpus: Path) -> None:
    payload = _outline_of(corpus, "ses-outline")
    first, second = payload["segments"]
    assert first["start_ts"] == "2026-08-01 10:00:00"
    assert first["end_ts"] == "2026-08-01 10:00:06"
    assert second["start_ts"] == "2026-08-01 10:00:08"
    assert second["end_ts"] == "2026-08-01 10:00:14"
    assert payload["n_exchanges"] == 8
    assert payload["session_id"] == "ses-outline"


def test_outline_is_deterministic_across_rebuilds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sessions = [
        (
            "ses-outline",
            "2026-08-01",
            exchange_records("ses-outline", "2026-08-01", A_EXCHANGES + B_EXCHANGES),
        ),
        ("ses-f1", "2026-08-02", [line("user", "u1", "ses-f1", "2026-08-02T10:00:00Z", LONG)]),
        ("ses-f2", "2026-08-03", [line("user", "u1", "ses-f2", "2026-08-03T10:00:00Z", LONG)]),
        ("ses-f3", "2026-08-04", [line("user", "u1", "ses-f3", "2026-08-04T10:00:00Z", LONG)]),
        ("ses-f4", "2026-08-05", [line("user", "u1", "ses-f4", "2026-08-05T10:00:00Z", LONG)]),
    ]
    payloads = [
        _outline_of(make_db(tmp_path / name, sessions), "ses-outline")
        for name in ("one", "two")
    ]
    assert payloads[0] == payloads[1]


# ---------------------------------------------------------------- criterion 2: uuid dedup


def test_duplicate_uuids_count_once(tmp_path: Path) -> None:
    records = exchange_records("ses-dup", "2026-08-01", A_EXCHANGES + B_EXCHANGES)
    duplicated = [record for record in records for _ in range(2)]  # same uuid twice

    filler = [
        ("ses-f1", "2026-08-02", [line("user", "u1", "ses-f1", "2026-08-02T10:00:00Z", LONG)]),
        ("ses-f2", "2026-08-03", [line("user", "u1", "ses-f2", "2026-08-03T10:00:00Z", LONG)]),
        ("ses-f3", "2026-08-04", [line("user", "u1", "ses-f3", "2026-08-04T10:00:00Z", LONG)]),
        ("ses-f4", "2026-08-05", [line("user", "u1", "ses-f4", "2026-08-05T10:00:00Z", LONG)]),
    ]
    plain = make_db(
        tmp_path / "plain",
        [("ses-dup", "2026-08-01", records), *filler],
    )
    resumed = make_db(
        tmp_path / "resumed",
        [("ses-dup", "2026-08-01", duplicated), *filler],
    )

    assert _outline_of(plain, "ses-dup") == _outline_of(resumed, "ses-dup")
    assert _outline_of(resumed, "ses-dup")["n_exchanges"] == 8


# ---------------------------------------------------------------- criterion 3: compaction summaries


def test_compaction_summary_rows_are_excluded_from_topics(tmp_path: Path) -> None:
    summary = line(
        "user",
        "summary-uuid",
        "ses-compact",
        "2026-08-01T10:00:03Z",
        ("continuationword recapword " * 80).strip(),
        extra={"isCompactSummary": True},
    )
    records = exchange_records("ses-compact", "2026-08-01", A_EXCHANGES + B_EXCHANGES)
    # The summary sits between the first and second exchange.
    records.insert(4, summary)

    db = make_db(
        tmp_path,
        [
            ("ses-compact", "2026-08-01", records),
            ("ses-f1", "2026-08-02", [line("user", "u1", "ses-f1", "2026-08-02T10:00:00Z", LONG)]),
            ("ses-f2", "2026-08-03", [line("user", "u1", "ses-f2", "2026-08-03T10:00:00Z", LONG)]),
            ("ses-f3", "2026-08-04", [line("user", "u1", "ses-f3", "2026-08-04T10:00:00Z", LONG)]),
            ("ses-f4", "2026-08-05", [line("user", "u1", "ses-f4", "2026-08-05T10:00:00Z", LONG)]),
        ],
    )

    payload = _outline_of(db, "ses-compact")
    assert payload["n_exchanges"] == 8  # the summary opened no exchange
    all_terms = {term for segment in payload["segments"] for term in segment["terms"]}
    assert "continuationword" not in all_terms
    assert "recapword" not in all_terms
    # The boundary is still between the two vocabularies.
    assert [segment["start"] for segment in payload["segments"]] == [0, 4]


# ---------------------------------------------------------------- criterion 4: short sessions


@pytest.mark.parametrize("n_exchanges", [1, 2, 5])
def test_fewer_than_two_windows_is_one_segment(tmp_path: Path, n_exchanges: int) -> None:
    exchanges = (A_EXCHANGES + B_EXCHANGES)[:n_exchanges]
    db = make_db(
        tmp_path,
        [
            ("ses-short", "2026-08-01", exchange_records("ses-short", "2026-08-01", exchanges)),
            ("ses-f1", "2026-08-02", [line("user", "u1", "ses-f1", "2026-08-02T10:00:00Z", LONG)]),
            ("ses-f2", "2026-08-03", [line("user", "u1", "ses-f2", "2026-08-03T10:00:00Z", LONG)]),
            ("ses-f3", "2026-08-04", [line("user", "u1", "ses-f3", "2026-08-04T10:00:00Z", LONG)]),
            ("ses-f4", "2026-08-05", [line("user", "u1", "ses-f4", "2026-08-05T10:00:00Z", LONG)]),
        ],
    )
    payload = _outline_of(db, "ses-short")
    assert len(payload["segments"]) == 1
    assert payload["segments"][0]["n_exchanges"] == n_exchanges


# ---------------------------------------------------------------- criterion 5: titles


def test_custom_title_wins_over_ai_title(tmp_path: Path) -> None:
    records = exchange_records("ses-titled", "2026-08-01", A_EXCHANGES[:2])
    records.append(title_record("ses-titled", "2026-08-01T10:00:02Z", "ai-title", "AI TITLE"))
    records.append(title_record("ses-titled", "2026-08-01T10:00:04Z", "custom-title", "MY TITLE"))
    db = make_db(tmp_path, [("ses-titled", "2026-08-01", records)])

    payload = _outline_of(db, "ses-titled")
    assert payload["title"] == "MY TITLE"
    assert payload["title_kind"] == "custom-title"


def test_ai_title_is_the_fallback_when_no_custom_title_exists(tmp_path: Path) -> None:
    records = exchange_records("ses-ai", "2026-08-01", A_EXCHANGES[:2])
    records.append(title_record("ses-ai", "2026-08-01T10:00:02Z", "ai-title", "AI TITLE"))
    db = make_db(tmp_path, [("ses-ai", "2026-08-01", records)])

    payload = _outline_of(db, "ses-ai")
    assert payload["title"] == "AI TITLE"
    assert payload["title_kind"] == "ai-title"


def test_no_title_events_leaves_title_null(corpus: Path) -> None:
    payload = _outline_of(corpus, "ses-outline")
    assert payload["title"] is None
    assert payload["title_kind"] is None


# ---------------------------------------------------------------- criterion 6: the CLI


def test_cli_json_parses_and_carries_the_outline(
    corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _cli(corpus, "ses-outline", "--json", capsys=capsys)
    payload = json.loads(out)
    assert payload["session_id"] == "ses-outline"
    assert payload["project_dir"] == "transcripts"
    assert payload["n_exchanges"] == 8
    assert len(payload["segments"]) == 2
    first = payload["segments"][0]
    assert set(first) == {"start", "end", "start_ts", "end_ts", "n_exchanges", "terms", "opening"}
    assert first["start"] == 0 and first["end"] == 3


def test_cli_text_output_shows_header_and_segments(
    corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _cli(corpus, "ses-outline", capsys=capsys)
    lines = out.splitlines()
    assert lines[0].startswith("session ses-outline")
    assert "exchanges: 8" in lines
    assert "[1] 2026-08-01 10:00:00 .. 2026-08-01 10:00:06  (4 exchanges)" in lines
    assert "[2] 2026-08-01 10:00:08 .. 2026-08-01 10:00:14  (4 exchanges)" in lines
    assert any("allowlist" in line for line in lines)
    assert any(line.startswith("  first: mitmproxy needs an egress allowlist") for line in lines)


def test_terms_flag_caps_each_segment(
    corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _cli(corpus, "ses-outline", "--terms", "3", "--json", capsys=capsys)
    payload = json.loads(out)
    assert all(len(segment["terms"]) == 3 for segment in payload["segments"])


def test_missing_database_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["topics", "ses-outline", "--db", str(tmp_path / "absent.duckdb")])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("error: no database")
    assert captured.out == ""


def test_missing_prefix_is_a_clean_error(
    corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["topics", "zzzz", "--db", str(corpus)])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("error: no session matching prefix 'zzzz'")


def test_ambiguous_prefix_names_the_candidates(
    corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["topics", "ses-", "--db", str(corpus)])
    captured = capsys.readouterr()
    assert code == 1
    assert "matches 5 sessions" in captured.err
    assert "ses-outline" in captured.err


def test_topics_does_not_modify_the_duckdb_file(
    corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = corpus.stat().st_mtime_ns
    _cli(corpus, "ses-outline", "--json", capsys=capsys)
    assert corpus.stat().st_mtime_ns == before


# ---------------------------------------------------------------- repair: meta/sidechain rows (review)

FILLER = [
    ("ses-f1", "2026-08-02", [line("user", "u1", "ses-f1", "2026-08-02T10:00:00Z", LONG)]),
    ("ses-f2", "2026-08-03", [line("user", "u1", "ses-f2", "2026-08-03T10:00:00Z", LONG)]),
    ("ses-f3", "2026-08-04", [line("user", "u1", "ses-f3", "2026-08-04T10:00:00Z", LONG)]),
    ("ses-f4", "2026-08-05", [line("user", "u1", "ses-f4", "2026-08-05T10:00:00Z", LONG)]),
]


def test_meta_and_sidechain_rows_open_no_exchange_and_add_no_terms(
    tmp_path: Path,
) -> None:
    meta_user = line(
        "user",
        "meta-uuid",
        "ses-meta",
        "2026-08-01T10:00:03Z",
        "harness meta-noise prose that must never surface",
        extra={"isMeta": True},
    )
    sidechain_user = line(
        "user",
        "side-uuid",
        "ses-meta",
        "2026-08-01T10:00:04Z",
        "sidechain-noise human prose that must never surface",
        extra={"isSidechain": True},
    )
    sidechain_assistant = line(
        "assistant",
        "side-a-uuid",
        "ses-meta",
        "2026-08-01T10:00:05Z",
        "sidechain-noise assistant prose that must never surface",
        extra={"isSidechain": True},
    )
    plain_records = exchange_records("ses-meta", "2026-08-01", A_EXCHANGES + B_EXCHANGES)
    polluted = list(plain_records)
    # The meta/sidechain rows sit between the first and second exchange.
    polluted[4:4] = [meta_user, sidechain_user, sidechain_assistant]

    plain = make_db(tmp_path / "plain", [("ses-meta", "2026-08-01", plain_records), *FILLER])
    with_meta = make_db(
        tmp_path / "with-meta", [("ses-meta", "2026-08-01", polluted), *FILLER]
    )

    # Outline identical to the fixture without them: no exchange opened, no terms added.
    assert _outline_of(with_meta, "ses-meta") == _outline_of(plain, "ses-meta")
    assert _outline_of(with_meta, "ses-meta")["n_exchanges"] == 8
    all_terms = {
        term for segment in _outline_of(with_meta, "ses-meta")["segments"]
        for term in segment["terms"]
    }
    assert not (all_terms & {"meta-noise", "sidechain-noise"})


# ---------------------------------------------------------------- repair: --window (review)


def test_window_zero_is_rejected_by_the_cli(
    corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["topics", "--db", str(corpus), "ses-outline", "--window", "0"])
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "window" in captured.err


@pytest.mark.parametrize("window", [0, -1])
def test_window_below_one_raises_valueerror(corpus: Path, window: int) -> None:
    connection = connect(corpus, read_only=True)
    try:
        with pytest.raises(ValueError):
            outline(connection, "ses-outline", window=window)
    finally:
        connection.close()