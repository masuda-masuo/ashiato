Orchestrator: claude-fable-5 | session 36fce60d-587e-469e-bc47-22275728771d | 2026-08-22

# ashiato issue #18: `ashiato grep` — text search over the transcript corpus with match windows and role/time filters

## Deliverables (files that must change; notes or summary files are NOT deliverables)

- `src/ashiato/grep.py`
- `src/ashiato/cli.py`
- `tests/test_grep.py`

## Smoke

- `python -m pytest -q`
- `ashiato grep --help` baseline-red

## Purpose

The built DuckDB is the only place where an agent's full past reasoning survives
(memory files keep summaries). Today the only way to find "where did I analyse X" was
four rounds of hand-written SQL over `events.text` plus Python to print a window around
each hit and narrow by role / time. `ashiato sql` already exposes the data; what is
missing is the ergonomic layer: a regex search that prints *who said it, when, in which
session* and a bounded window of text around each match, with the same role/time
filters every such search needs. This is `grep`, not `recall`: deterministic, no LLM,
no embeddings, read-only.

## Workplace

Container `3a105a80ca7c` (attach via sunaba MCP `sandbox_attach`). Repo at `/workspace`,
deps already pip-installed. Work only inside the container with sunaba tools.

## Read first (in the container)

- `src/ashiato/schema.py` — `EVENT_TABLE` / `TOOL_CALL_TABLE` column lists (verified
  facts below are from there; do not re-derive)
- `src/ashiato/cli.py` — subcommand wiring; reuse the existing `FORMATS`, `_row_limit`,
  `_parse_since`, `_resolve_db` helpers rather than duplicating them
- `src/ashiato/salvage.py` — the most recent precedent for a read-only query module
  wired into the CLI (merged 2026-08-22)
- `tests/test_cli.py`, `tests/test_salvage.py`, `tests/fixtures/` — test conventions;
  build fixture DBs through the real build pipeline on tiny JSONL transcripts

## Spec

### 1. Module `src/ashiato/grep.py`

Read-only search over an already-built ashiato DuckDB. No new stored tables or views;
`FORMAT_VERSION` is not changed.

### 2. Search scope (verified column facts)

- Default scope: `events.text` (columns: `event_id, session_id, ts TIMESTAMP naive UTC,
  role, is_meta BOOLEAN, is_sidechain, text, ...`).
- With `--tool-calls`: also `tool_calls.input_summary` and `tool_calls.result_text`
  (columns include `tool_use_id, session_id, ts, tool_name, input_summary, result_text`).
- Events with `is_meta = true` are excluded by default (harness noise);
  `--include-meta` includes them.
- Matching: Python `re` semantics or DuckDB `regexp_matches` — either is acceptable,
  but the behaviour must be the same for `-i` (case-insensitive) and the pattern must
  be treated as a regex, not a substring.

### 3. Filters

`--role user|assistant`, `--since TS`, `--until TS` (both ISO-8601, normalised to naive
UTC like `_parse_since`), `--session PREFIX` (prefix match on `session_id`), `-i`,
`--limit N` (default 20; `0` = all, the project-wide convention).

### 4. Output

- One hit = one header line `<ts isoformat>  session=<session_id>  role=<role>` (for
  tool-call hits: `tool=<tool_name>` instead of role) followed by the window: up to
  `--context N` characters (default 200) on each side of the first match in that row,
  newlines in the window replaced by a visible marker so one hit stays on a bounded
  number of lines. `--all-matches` prints a window per match instead of only the first.
- `--whole` prints the full text of the matched row instead of a window.
- `--format table|json|csv` as in the other subcommands. The JSON/CSV forms carry
  `event_id` (or `tool_use_id`), `session_id`, `ts`, `role`/`tool_name`, the match
  offset(s), and the window text, so a caller can re-fetch the whole row with
  `ashiato sql`.
- Hits are ordered newest first (the `denials` / `recalls` convention).
- A final summary line `(<n> hit(s))` on stdout for the table format only.
- Exit code: `0` when at least one hit, `1` when none (plus a one-line stderr notice),
  `2` on error (bad regex, missing DB, schema out of date).

### 5. CLI

`ashiato grep PATTERN [flags]` wired in `cli.py` next to the other subcommands, using
the existing `--db` convention.

## Acceptance criteria

- On a fixture where the pattern occurs in one user event and one assistant event, the
  command prints two hits with correct role, ts and session_id; `--role user` prints one.
- `--since` set between the two hits' timestamps excludes the earlier one.
- `--context 50` never prints more than 100 characters plus the matched text per hit;
  `--limit 1` prints exactly one hit and still exits 0.
- `-i` finds a match that differs only in case; an invalid regex exits 2 with a message.
- `--tool-calls` finds a pattern that occurs only in a tool call's `input_summary`.
- No hit: exit 1, empty stdout, one-line stderr notice.
- The DuckDB connection is opened read-only and nothing is written to the DB file
  (assert with a before/after hash or mtime, as `test_salvage.py` does).
- All existing tests pass unchanged; `FORMAT_VERSION` stays as it is in `schema.py`.

## Frozen tests

- `tests/test_build.py`
- `tests/test_cli.py`
- `tests/test_parser.py`
- `tests/test_opencode.py`
- `tests/test_salvage.py`

## Non-goals

- No semantic / embedding search and no LLM anywhere in the path. If you believe a
  non-regex matching mode is needed, say so in the report instead of adding it.
- No indexing or caching for speed: a full scan of `events.text` is the design. If you
  measure it to be unacceptably slow on the fixture scale, report the number; do not
  add stored state.
- No reading of the live `~/.claude/projects`; the built DB is the corpus.
- Do not modify `build.py`, `parser.py`, `recall.py`, `salvage.py`, `opencode.py`, or
  `schema.py`. (`cli.py` wiring is expected. If another file genuinely must change,
  document why in your report rather than deviating silently.)

## Suggested design (starting point, not frozen criteria)

- `grep.py` owns the query construction, the window extraction (a pure function
  `window(text, start, end, context) -> str` is easy to test) and a `Hit` dataclass;
  `cli.py` only parses arguments, opens the DB read-only, formats, and maps the result
  to the exit code.
- Do the regex filtering in DuckDB (`regexp_matches`) to avoid pulling every event into
  Python, then compute offsets in Python with `re` on the rows that matched.

## Constraints

- English code comments and docstrings; match the repo's style (module docstrings
  explain "why").
- Deterministic core only; this repo's fixed design point is that no LLM sits in the
  extraction or query path.
