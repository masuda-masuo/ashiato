Orchestrator: claude-opus-5[1m] | session 4a134f0c-e0a7-4113-bf17-c88c9638f19e | 2026-08-23

# ashiato: document `salvage` and `grep` in README, and guard the docs against the next subcommand

## Deliverables (files that must change; notes or summary files are NOT deliverables)

- `README.md`
- `tests/test_readme.py`

## Smoke

- `python -m pytest -q`
- `python -m pytest -q tests/test_readme.py` baseline-red

## Purpose

`README.md` is the only design document this project has: there is no `docs/`
directory, so the README is where the data model, the guarantees and the CLI surface
are recorded. Two subcommands have shipped since it was last updated -- `salvage`
(merged in PR #17) and `grep` (merged in PR #19) -- and neither appears anywhere in it.
The `## Use` block still lists only `build`, `sql`, `denials`, `recalls` and `info`
(verified: `grep -c salvage README.md` is 0 on `main` at 112ceca).

Two things are wrong, and the second one matters more. A reader cannot discover half
the tool. And nothing in the repository notices: the gap opened twice in a row, in two
consecutive merged PRs, with a green suite both times. Documenting the two commands
fixes today's instance; a test that fails when a subcommand has no README entry is what
stops the third one.

## Read first (in the container)

These are the authoritative sources for what the two commands actually do. Do not
describe them from their names -- read them.

- `src/ashiato/cli.py` -- `_build_arg_parser()` is the whole CLI surface: every
  subcommand, every flag, every default. `_run_salvage` and `_run_grep` are the
  behaviour, including exit codes and what is printed for each `--format`.
- `src/ashiato/salvage.py` -- module docstring states the nomination rule (both
  conditions), and what happens when the kaiba database is absent.
- `src/ashiato/grep.py` -- module docstring states why matching is Python `re` and not
  DuckDB's regex engine, and what is and is not searched.
- `tests/test_salvage.py`, `tests/test_grep.py` -- the observable behaviour that is
  already pinned, including exit codes.
- `README.md` -- the document you are extending. Match its voice.

## Spec

### 1. Document `salvage` and `grep` in `README.md`

Add both to the `## Use` synopsis block, keeping the existing style of that block
(command line first, then a bullet per command explaining it), and give each command
whatever prose it needs so a reader who has never seen the source can use it. At
minimum a reader must be able to learn, from the README alone:

- for `salvage`: what a nomination is and what the two conditions for nominating are,
  that it writes to nothing, what `--window-minutes` and `--since` control, and what
  changes when the kaiba database is missing;
- for `grep`: what is searched by default and what `--tool-calls` adds, that meta
  events are excluded unless `--include-meta`, that the pattern is a Python `re` regex
  (not DuckDB's dialect) and why that distinction is visible to the user, what the
  match window is and how `--context` / `--whole` / `--all-matches` change it, and the
  exit-code convention.

Placement inside the README is your judgement -- the constraint is that the document
reads as one piece afterwards, not that a particular heading exists. Every flag either
appears in the text or is left out deliberately; do not silently document a subset.

Correct anything else in the README that these two commands have made false. If you
find such a statement, say so in your report -- that is a finding, not a chore.

### 2. Guard: `tests/test_readme.py`

A test that fails when a subcommand exists in the CLI but is undocumented. Derive the
list of subcommands from the parser built by `ashiato.cli._build_arg_parser()` -- never
from a hand-written list in the test, which is the same staleness one level down.

The test must fail on the current `main` README (this is why it is `baseline-red`) and
pass once section 1 is done. Keep it to the check that the next new subcommand will
trip: a name-level check is enough, and a test that tries to verify prose quality is
worse than no test.

## Acceptance criteria

1. `python -m pytest -q` is green, including the new test.
2. Every subcommand of `ashiato` is findable in `README.md`, and the description of
   each matches what `src/ashiato/` actually does -- flags, defaults and exit codes
   included.
3. Adding a new subcommand to `_build_arg_parser()` without touching `README.md` makes
   the suite red. Removing a subcommand does not.
4. No behaviour change: nothing under `src/ashiato/` is modified.
5. No statement remains in `README.md` that the two commands have made false.

## Non-goals

- Do not change any code under `src/ashiato/`. This is documentation plus one test.
  If you find a real defect in `salvage` or `grep` while reading them, report it in
  your result -- do not fix it here. If you believe the documentation genuinely cannot
  be made true without a code change, say so explicitly rather than working around it.
- Do not add a `docs/` directory or split the README.
- Do not rewrite or reflow sections the two new commands do not touch.
- Do not add CI configuration; this repository deliberately has none.

## Constraints

- English, matching the README's existing register: it explains *why* a design is what
  it is, not only what it does. Follow that.
- Do not commit, push, or open a pull request; you have no publish tool. Stop when the
  suite is green and report the files you changed.
