# Adversarial review: ashiato core event layer

Orchestrator: claude-opus-5[1m] | session 878e7b9a-447e-4889-8fbc-21bbd7972498 | 2026-08-10

Container: `ecea07a999f3`. Attach and review the working-tree changes at `/workspace`
against `origin/main`. **Read and re-run gates only. Do not modify any file.**

The implementation was written by a Claude-family model. You are a different family and
that is the point: look for the mistakes a same-family reviewer would wave through.

## What the code is supposed to do

`ashiato` parses Claude Code session transcripts (JSONL, one JSON object per line, under
`~/.claude/projects/**`) into a DuckDB database with four tables:

- `sessions` — one row per transcript file. Token totals must be **deduplicated by
  `request_id`**; naive summing inflates them 2-3.5x.
- `events` — one row per JSONL line, with `parent_uuid` as a tree edge and a `depth`
  column computed by a **memoized O(n)** walk (a naive walk costs 3.41s on the reference
  corpus because chains reach depth 2,398).
- `tool_calls` — one row per tool invocation joined to its result on `tool_use_id`, with
  `outcome` resolved in this order: `pending` (no result) -> `denied` (result text matches
  a denial pattern) -> `error` (`is_error` true) -> `ok`.
- `source_files` — bookkeeping making `build` incremental on (path, size, mtime).

CLI: `ashiato build` / `ashiato sql` / `ashiato info`.

Hard constraints: **no LLM call, no network call anywhere**; deterministic output for the
same inputs; runtime dependency is `duckdb` only; Python 3.11+.

## Known context you should not re-litigate

- The full gate has already been run by the orchestrator: 70/70 tests pass, 0 collection
  errors, ruff and type checks clean, all 6 smoke commands exit 0. Reporting "the gate is
  green" is not a finding.
- Fixtures are synthetic **by design**. Real transcripts contain secrets and must never be
  committed. "Tests do not use real data" is not a finding.
- `isSidechain` is false across the entire real reference corpus, so `parent_tool_use_id`
  is fixture-only. This is already known and stated in the PR. Not a finding.
- The module layout was left to the implementer's discretion. Disagreeing with where code
  lives is not a finding unless it causes a defect.

## What to actually look for

Prioritise defects that survive contact with real data the fixtures cannot represent:

1. **Correctness of the dedup.** Does it dedup per file or globally? What happens when the
   same `request_id` legitimately appears in two different sessions? Is a row without a
   `request_id` counted exactly once?
2. **The `outcome` ordering.** Can a denied call be misclassified as `error`, or vice
   versa? What if `is_error` is true *and* the text matches a denial pattern? What if the
   result content is a list of blocks rather than a string, or is absent?
3. **The `tool_use` / `tool_result` join.** Duplicate `tool_use_id` across sessions.
   Results arriving before their call in file order. A call whose result is in a
   *different* file. Multiple results for one call.
4. **Depth computation.** Cycles, dangling parents, forests, and chains longer than
   Python's recursion limit. Does memoization ever return a stale or wrong value?
5. **Incremental build.** What happens when a file is appended to (live session) between
   builds, and size or mtime is unchanged or goes backwards? Are stale rows for a rebuilt
   file actually deleted, or do duplicates accumulate?
6. **Robustness claims.** A malformed line, a truncated final line, an empty file, a
   non-UTF-8 byte, a `null` where an object is expected, a `message.content` that is a
   string rather than a list. The stated contract is "skipped and counted, never fatal" —
   verify that is true rather than assumed.
7. **The no-network / no-LLM claim.** Verify it by reading the code, not by trusting it.
8. **Determinism.** Is row ordering or any generated id dependent on dict iteration,
   filesystem order, or wall-clock time?

## Reporting

Report findings with severity (`critical` / `high` / `medium` / `low`) and, for each, a
concrete failure scenario: the input or state that triggers it and the wrong result it
produces. A finding without a failure scenario will be discarded.

If you find nothing above `low`, say so plainly. Do not manufacture findings to look
thorough, and do not pad with style preferences.
