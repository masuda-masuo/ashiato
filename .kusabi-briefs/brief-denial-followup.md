# ashiato#4: denial_followups view + input_summary + `denials` subcommand

Orchestrator: claude-fable-5 | session (current) | 2026-08-11

Container: `77756c05e8a4` (repo `masuda-masuo/ashiato` cloned at `/workspace`, deps installed,
`ashiato` CLI on PATH). Attach with `sandbox_attach` and work there.

## What this adds

ashiato's pitch is "see what the agent did after it was stopped" (roadmap issue #2, layer 2).
Today that requires a hand-written window query plus per-tool knowledge of the `input` JSON.
This job makes it built in. Three pieces, all deterministic, no LLM, no network:

1. **`input_summary` column on `tool_calls`** — a short human-readable string derived from
   `input` per tool.
2. **`denial_followups` view** — one row per `outcome='denied'` call, joined to the next
   tool call in the same session.
3. **`ashiato denials` CLI subcommand** — prints the view.

## Measured facts (trust these; you have no access to real transcripts)

Real corpus 2026-08-11: 365 files / 150,799 events / 33,403 tool calls, outcomes
ok 32,159 / error 1,085 / denied 149 / pending 10. The window query
`lead(...) OVER (PARTITION BY session_id ORDER BY seq)` over `tool_calls` produces correct
pairs on real data (verified by hand). Real examples of pairs: verbatim retry of the same
Bash command; narrowed retry (`... gh pr merge 18 --squash && git pull` denied →
`gh pr merge 18 --squash`); switching to a different tool. `is_sidechain` is essentially
all-false in the corpus, so do not build sidechain-aware logic (see Non-goals).

`tool_calls` columns today (from `DESCRIBE`): tool_use_id, session_id, file_path, seq, ts,
call_event_id, result_event_id, tool_name, tool_kind, mcp_server, input (JSON), outcome,
is_error, result_text, result_truncated, duration_ms, permission_mode, cwd, is_sidechain,
parent_tool_use_id.

## Design sketch

- **input_summary** lives in the parser/build layer as a pure function
  `summarize_input(tool_name, input_dict) -> str | None`. Per-tool extraction:
  `Bash`/`PowerShell` → `command`; `Read`/`Write`/`Edit`/`NotebookEdit` → `file_path`;
  `Glob`/`Grep` → `pattern`; `Skill` → `skill`; `Agent`/`Task` → `description`;
  `WebFetch`/`WebSearch` → `url`/`query`; MCP tools and anything else → compact JSON of the
  input, truncated. Truncate every summary to 200 chars. Missing/None input → NULL.
- **denial_followups** is a SQL VIEW created alongside the tables (so it is always
  consistent with `tool_calls`, no extra incremental maintenance). Columns: the denied
  call's session_id, seq, ts, tool_name, input_summary, permission_mode, cwd; the next
  call's tool_name / input_summary / outcome / ts (NULL when the denial was the session's
  last call); `gap_seconds` between the two; and a mechanical label `followup_kind` with
  exactly these values: `verbatim-retry` (same tool_name AND identical input JSON text),
  `same-tool` (same tool_name, different input), `other-tool`, `none` (no next call).
  "Next" = next tool call by `seq` within the same `session_id`.
- **CLI**: `ashiato denials [--db PATH] [--format table|json|csv] [--limit N]
  [--session ID]`, default ordering ts DESC, default limit 50. Reuse the existing sql/output
  plumbing; do not invent a second formatter.

The sketch fixes the observable surface (column names, followup_kind values, CLI flags).
How you factor the code internally is yours.

## Acceptance criteria

- `ashiato build` on the bundled fixtures produces a DB where `denial_followups` is
  queryable and every `outcome='denied'` row in `tool_calls` appears exactly once.
- `followup_kind` takes only the four values above; a denial that is the last call of its
  session yields `none` with NULL next_* columns.
- `input_summary` is populated for Bash calls in the fixtures (the fixture corpus already
  contains denied Bash calls in `tests/fixtures/session_main.jsonl`).
- `ashiato denials` exits 0 on a fixture-built DB and honors `--limit` and `--format json`.
- Determinism: building twice from the same frozen fixture dir yields identical
  `denial_followups` output.
- Tests cover: summarize_input per-tool branches incl. fallback + truncation;
  followup_kind all four values; the last-call-of-session edge; CLI flags.
- Full existing test suite stays green (it was green at clone: run it first to baseline).

## Non-goals (say so explicitly if you believe one must change)

- No sidechain/parentUuid-tree definition of "next action" — seq order only. If you hit a
  case where seq order is demonstrably wrong, leave a code comment and note it in your
  report instead of implementing tree logic.
- No severity/risk scoring, no LLM, no network calls.
- No schema changes beyond the `input_summary` column and the view.
- Do not touch the incremental-build/skip logic except as needed to add the column
  (a full rebuild being required once after upgrade is acceptable; note it in the report).

## Deliverables (files that must change; notes are NOT deliverables)

- `src/ashiato/parser.py`
- `src/ashiato/schema.py`
- `src/ashiato/build.py`
- `src/ashiato/cli.py`
- `tests/test_parser.py`
- `tests/test_build.py`
- `tests/test_cli.py`

## Smoke (run in container)

- `ashiato build --source tests/fixtures --db /tmp/smoke.duckdb`
- `ashiato denials --db /tmp/smoke.duckdb`
- `ashiato denials --db /tmp/smoke.duckdb --format json --limit 5`
- `ashiato sql --db /tmp/smoke.duckdb "SELECT followup_kind, count(*) FROM denial_followups GROUP BY 1"`
- `python -m pytest -q`
