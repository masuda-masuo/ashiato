# Fix: anchor denial classification to the start of the result text (issue #8)

Orchestrator: claude-fable-5 | session ashiato-issue8 | 2026-08-11

Container: `5ce12c133499` (clone of masuda-masuo/ashiato@main, deps installed).

## Problem (measured, do not re-verify)

`outcome='denied'` is decided by a **substring** match:

- `src/ashiato/parser.py:324` — `if any(pattern in result_text for pattern in denial_patterns):`

Any *successful* result that merely **quotes** a denial string is misclassified as
denied. Measured on a frozen 377-file real corpus (2026-08-11): 154 rows have
`outcome='denied'`; **6 are false positives**, all successful calls whose result quoted
the patterns:

- 5× MCP results starting `{"result": "{\"content\": ...` / `{"result": "{\"status\": \"ok\", ...`
  (reads/diffs of ashiato's own `parser.py` and fixtures, which contain the pattern
  literals)
- 1× Bash output starting `usage: ashiato build ...` (a command chain that ended in
  `grep -n "denied\|denial" src/ashiato/*.py`)

The 148 genuine denials **all start** with one of the two patterns:

| result_text prefix | count |
|---|---|
| `The user doesn't want to proceed with this tool use. The tool use was ...` | 90 |
| `Permission for this action was denied by the Claude Code auto mode cla...` | 58 |

## Decision (frozen — do not re-litigate)

A result is a denial **iff its text, after stripping leading whitespace, starts with
one of the denial patterns**. Patterns are prefixes now, not substrings.

- Keep the `parse_file(..., denial_patterns=(...))` override; its semantics become
  "prefix match" and the docstring + README must say so.
- Do not add a second structural heuristic (e.g. "payload parses as JSON ⇒ not a
  denial"). One rule, anchored.

## Stale-database invariant

`outcome` is a **stored column**, so a database built by the old code holds rows
classified under the old rule. Requirement (outcome, not mechanism): after this change,
`build` / `sql` / `denials` against a database whose `tool_calls` rows were produced by
the pre-fix code must be **refused with the existing delete-and-rebuild message, not
silently mixed or half-upgraded**. The repo already has this machinery —
`SchemaOutOfDate` in `src/ashiato/build.py` (read lines ~55–160 to see how it decides).
If that check is purely structural (column comparison) and cannot see a semantic
change, extend it in the smallest way that makes this detectable. If you conclude the
invariant genuinely cannot be met with a small change, say so explicitly in your report
instead of silently skipping it.

## Deliverables (files that must change; notes/summaries are NOT deliverables)

- `src/ashiato/parser.py`
- `tests/test_parser.py`
- `README.md`

`src/ashiato/build.py` and/or `src/ashiato/schema.py` may also change if the
stale-database invariant requires it.

## Tests

- New: a successful result whose body *contains* (but does not start with) each
  denial pattern → `outcome='ok'`.
- New: a result that starts with a denial pattern (with and without leading
  whitespace) → `outcome='denied'`.
- Existing denial tests keep passing unmodified — if one must change, that is a signal
  the anchored rule broke a genuine case; stop and report rather than editing the
  expectation.
- Negative control: demonstrate (in your report, not as a committed test hack) that at
  least one new test fails against the unmodified classification line.

## Smoke

- `python -m pytest -q`
- `python -m ruff check .`

## Non-goals

- No change to the `denial_followups` view, the outcome ladder order
  (pending → denied → error → ok), or the CLI surface.
- No new configuration flags.
- If you believe one of these must change to meet the decision above, say so
  explicitly in your report instead of working around it.
