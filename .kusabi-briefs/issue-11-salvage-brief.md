Orchestrator: claude-fable-5 | session 36fce60d-587e-469e-bc47-22275728771d | 2026-08-22

# ashiato issue #11: nominate un-bookkept work-state changes salvaged from transcripts

## Deliverables (files that must change; notes or summary files are NOT deliverables)

- `src/ashiato/salvage.py`
- `src/ashiato/cli.py`
- `tests/test_salvage.py`

## Smoke

- `python -m pytest -q`
- `ashiato salvage --help` baseline-red

## Purpose

When a session ends abnormally (freeze, kill, context exhaustion), the work-state
bookkeeping that a healthy session does — updating the shared agenda after a chain
terminates or a PR merge is confirmed — silently does not happen. The next session then
either trusts a stale agenda row or pays to rebuild state from issue/PR lists. The
evidence of the un-bookkept state change is still in the transcript tail: the publish
call, the chain-wait completion, the merge confirmation. This feature mines the built
DuckDB corpus for such "state-change evidence not followed by bookkeeping" and emits
**nomination candidates** for a human/orchestrator to adjudicate.

Nomination only. The mechanism must not write to the agenda or anywhere else — that is
the project's authority model (same discipline as issue #10 and the recall_followups
view): derivation nominates, the inspecting tier files.

## Workplace

Container `3e8c4895a78a` (attach via sunaba MCP `sandbox_attach`). Repo at `/workspace`,
deps already pip-installed. Work only inside the container with sunaba tools.

## Read first (in the container)

- `src/ashiato/schema.py` — table layouts; read the FORMAT_VERSION comment block (it
  defines when a bump is required; this task must not require one)
- `src/ashiato/recall.py` — closest precedent for a derived-analysis module
- `src/ashiato/cli.py` — subcommand wiring pattern
- `tests/test_cli.py` and `tests/fixtures/` — test and fixture conventions

## Spec

### 1. Module `src/ashiato/salvage.py`

Pure report-time analysis over (a) an already-built ashiato DuckDB and (b) the kaiba
agenda ledger (SQLite). No new stored tables or views; FORMAT_VERSION stays 4.

### 2. Evidence and bookkeeping signals (verified against the production DB, 2026-08-22)

Query `tool_calls` for **evidence events** (a work-state change happened):

- publish: `tool_name IN ('mcp__sunaba__publish', 'mcp__code-sandbox-mcp__publish') AND outcome = 'ok'` (production: 922 rows)
- chain terminal watch: `tool_name = 'Bash' AND input_summary LIKE '%chain-wait%'` (production: 85 rows)

and for **bookkeeping events**:

- `tool_name = 'mcp__kaiba__agenda_edit' AND outcome = 'ok'` (production: 187 rows, all since 2026-08-16)

Keep the signal definitions in one obvious data structure so that adding a new evidence
kind later is a one-entry change; the issue explicitly frames the signal list as a
starting point, not frozen.

### 3. Nomination rule (deterministic; no LLM, no network)

Evidence event E in session S is **nominated** when both hold:

- (a) no successful bookkeeping event exists in S with `ts >= E.ts`, and
- (b) the kaiba `actions` ledger contains no row whose `created_at` or `done_at` falls
  within `[E.ts, E.ts + window]` (default window 30 minutes, CLI-tunable) — this covers
  bookkeeping done from a different session or agent.

When the kaiba db is absent or unreadable, skip check (b), print an explicit notice
that coverage is transcript-only, and still exit 0.

### 4. kaiba side (verified facts, do not re-derive)

- Default path `~/.kaiba/kaiba.db`, overridable via CLI flag.
- Open with the stdlib `sqlite3` module, read-only URI (`mode=ro`).
- Schema: `actions(id INTEGER PK, content TEXT, position REAL, author TEXT,
  created_at TEXT, done_at TEXT NULL)`; timestamps are ISO-8601 UTC with a trailing
  `Z`, e.g. `2026-08-22T04:25:58Z`.
- DuckDB `ts` values are naive UTC timestamps. Normalize both sides to UTC before
  comparing.

### 5. CLI

`ashiato salvage` subcommand. Flags: the existing db-path convention used by other
subcommands, plus `--kaiba-db`, `--window-minutes`, `--limit` (default 50), and
`--since` (only consider evidence at or after this timestamp). Each nomination line
carries: ts, session_id, evidence kind, a bounded snippet (`input_summary` or a
truncated `result_text` excerpt), and which check(s) it failed. Output never exceeds
`--limit` nominations regardless of corpus size.

## Acceptance criteria

- A DB where a publish event is followed by a successful `agenda_edit` in the same
  session yields no nomination for that pair.
- The same DB with the `agenda_edit` removed nominates the publish event, and the
  output carries its session_id and snippet.
- Kaiba coverage works both ways: an `actions` row stamped inside the window after the
  evidence ts suppresses the nomination; a row outside the window does not.
- With no kaiba db present: transcript-only mode, explicit notice, exit 0.
- The command performs no writes: the kaiba connection is read-only (`mode=ro`), the
  DuckDB connection is opened read-only.
- All existing tests pass unchanged; `FORMAT_VERSION` stays 4.
- No network access and no LLM invocation anywhere in the new code.

## Frozen tests

- `tests/test_build.py`
- `tests/test_cli.py`
- `tests/test_parser.py`
- `tests/test_opencode.py`

## Non-goals

- **No writes to the kaiba db, the agenda, or any store.** This is the project's
  authority boundary, not a style preference. If you conclude a write is genuinely
  required, stop and say so explicitly in your report instead of implementing it.
- **No opencode-source signals in this iteration.** Workers are read-only and cannot
  edit the agenda, so the sessions that matter are claude_code ones. If you find this
  premise materially wrong, say so explicitly rather than implementing around it.
- **No new stored tables/views, no FORMAT_VERSION bump.** If your design truly needs
  stored state, stop and explain why in the report.
- Do not modify `build.py`, `parser.py`, `recall.py`, `opencode.py`, or `schema.py`.
  (`cli.py` wiring is expected. If another file genuinely must change, document why in
  your report rather than deviating silently.)

## Suggested design (starting point, not frozen criteria)

- `salvage.py` owns the signal definitions, both coverage checks, and one decision
  function that takes an evidence event plus the session's bookkeeping events plus the
  relevant `actions` rows and returns nominated-or-covered with the reason. `cli.py`
  only parses arguments, opens the two stores, and prints.
- Build test fixtures through the existing build pipeline on tiny JSONL transcripts
  (see `tests/fixtures/` conventions) rather than hand-inserting DuckDB rows, and use a
  tiny temp sqlite file for the `actions` side.

## Constraints

- English code comments and docstrings; match the repo's existing style (module
  docstrings explain "why", not narrate the code).
- Deterministic core only — this repo's fixed design point is that no LLM sits in the
  extraction path.
