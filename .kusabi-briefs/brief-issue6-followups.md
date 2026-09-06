# ashiato#6: strictly-later "next" in denial_followups + rebuild hint on the read path

Orchestrator: claude-fable-5 | session (current) | 2026-08-11

Container: `3b0eee7b2a2f` (repo `masuda-masuo/ashiato` cloned at `/workspace` on current
main incl. PR#5, deps installed, `ashiato` CLI on PATH). Attach with `sandbox_attach`.

Two small, independent fixes accepted as follow-ups during the PR#5 review. Both are
mechanical; the only design already decided is written below. Deterministic, no LLM,
no network — as everywhere in this repo.

## Fix 1: `denial_followups` — "next" must be a strictly later transcript line

Today the view pairs each denied call with `lead(...)` over
`(PARTITION BY session_id ORDER BY seq, tool_use_id)`. Parallel `tool_use` blocks on one
assistant line share `seq`, so a denial can be paired with a same-line sibling — a call
that was issued *before* the model ever saw the denial, hence not a reaction to it.
Measured on the real corpus 2026-08-11: 5 of 149 denials (3.4%).

New definition (decided, do not weaken): **the follow-up is the first tool call in the
same session with strictly greater `seq`** (a later transcript line). Among candidates
sharing that seq, break the tie by `tool_use_id` — same stability rationale as now.
A denial whose session has no later-line call is `none` with NULL `next_*`, even when
same-line siblings exist.

Implementation is yours (correlated subquery, self-join with row_number, a RANGE window
frame — whatever reads best), but it stays inside the view SQL in `schema.py`: no new
table, no build-loop work, so an existing database gets the fix by upgrading ashiato
alone, without rebuilding. Update the view docstring and the README paragraph that
currently explains the tie-break ("stable choice rather than a faithful one") — the
same-line case is now excluded by construction, so say that instead.

## Fix 2: `denials` / `sql` on a pre-upgrade database must print the rebuild hint

`build` already refuses an old-schema DB via `_assert_current_schema` /
`SchemaOutOfDate` with "delete the database file and build again". But the read path
does not: `ashiato denials` against a pre-PR#5 DB prints DuckDB's raw
`Catalog Error: Table with name denial_followups does not exist!` (verified by hand
2026-08-11). Make the read path (`_run_query` or its callers) produce the same
actionable message — either run the schema assertion on open, or catch the catalog
error and append the rebuild hint. Exit code stays 1. A plain user SQL typo
(`SELECT * FROM nonexistent`) must NOT be rewritten into a rebuild hint — only the
schema-mismatch case gets it (assertion-on-open achieves this for free; if you catch
errors instead, scope the catch to ashiato's own table/view names).

## Acceptance criteria

- On the bundled fixtures, every denied call in `denial_followups` has either
  `next_seq`-strictly-greater semantics (next call's line is later than the denial's)
  or `followup_kind = 'none'` with NULL `next_*`.
- A fixture exercises the same-line case: a transcript line whose assistant message
  carries ≥2 parallel `tool_use` blocks, one of which is denied, followed by a later
  tool call — the view pairs the denial with the later call, not the sibling. And the
  variant where the denial's line is the last: `followup_kind = 'none'`.
- `followup_kind` still takes only {verbatim-retry, same-tool, other-tool, none}.
- `ashiato denials` and `ashiato sql "SELECT * FROM denial_followups"` against a
  database missing `input_summary`/the view exit 1 with a message that names the fix
  (rebuild), not a bare catalog error. `ashiato sql "SELECT * FROM nonexistent"`
  against a *current* database still returns DuckDB's own error untouched.
- Full suite green (baseline at clone: 112 tests — run it first; collected count must
  not drop).

## Non-goals (say so explicitly if you believe one must change)

- No change to `followup_kind` values, no sidechain/parentUuid logic, no severity
  scoring.
- No schema/table changes — Fix 1 is view-only, Fix 2 is error handling.
- Do not add a config flag for either behaviour.

## Deliverables (files that must change; notes are NOT deliverables)

- `src/ashiato/schema.py`
- `src/ashiato/cli.py`
- `tests/test_build.py`
- `tests/test_cli.py`
- `README.md`

(`src/ashiato/build.py` may change if you route the read-path assertion through it.)

## Smoke (run in container)

- `ashiato build --source tests/fixtures --db /tmp/smoke.duckdb`
- `ashiato sql --db /tmp/smoke.duckdb "SELECT followup_kind, count(*) FROM denial_followups GROUP BY 1"`
- `ashiato denials --db /tmp/smoke.duckdb --format json`
- `python -m pytest -q`
