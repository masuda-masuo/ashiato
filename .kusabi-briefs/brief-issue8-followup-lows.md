# Followup: close the two review lows from the issue-#8 chain (decided, do not re-litigate)

Orchestrator: claude-fable-5 | session ashiato-issue8 | 2026-08-12

Container: `5ce12c133499`. The anchored-denial fix and the `ashiato_meta` marker are
already in the working tree; do not revisit them. This task is exactly the two items
below, nothing else.

## 1. `info` must apply the same stale-database refusal as sql/denials

Today `_run_info` (src/ashiato/cli.py:210-233) → `database_info` (src/ashiato/build.py:456-473)
reads counts and the time window with no `SchemaOutOfDate` check, so a pre-fix database
is reported as if current while `sql`/`denials` refuse it. Make `info` refuse the same
way, with the same delete-and-rebuild message and exit code 1, matching how `sql` handles
`SchemaOutOfDate` in cli.py.

## 2. `create_schema` must not leave a refuse-only database on a crash

In `create_schema` (src/ashiato/build.py:203-228) the DDL and the marker INSERT run as
separate autocommit statements. A kill between `SCHEMA_SQL` and the marker INSERT leaves
ashiato tables with no marker → the next build refuses an empty, perfectly rebuildable
file. Close the window (DuckDB DDL is transactional — BEGIN/COMMIT around DDL + marker
stamp is the expected shape; another route is fine if the invariant holds: after a crash
at any point inside create_schema, a subsequent `build` either starts fresh or proceeds,
never refuses an empty database).

## Deliverables (files that must change; notes are NOT deliverables)

- `src/ashiato/build.py`
- `src/ashiato/cli.py`
- `tests/test_cli.py`

## Tests

- New: `info` against a database with the marker removed → exit 1 and the
  delete-and-rebuild message (mirror the existing sql/denials stale tests in
  tests/test_cli.py).
- For item 2, a test that simulates the half-created state (ashiato tables present,
  meta table absent entirely — not just the row deleted) and asserts the next build
  succeeds, if the invariant you implement makes that state unreachable-but-recoverable;
  if your implementation makes the state impossible instead, test the atomicity route
  you chose and say so in the report.
- All existing tests keep passing unmodified.

## Smoke

- `python -m pytest -q`
- `python -m ruff check .`

## Non-goals

- No change to classification, the marker format, FORMAT_VERSION, or the CLI surface.
- If either item genuinely requires touching those, stop and say so in the report.
