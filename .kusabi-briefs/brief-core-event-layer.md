# ashiato: JSONL transcripts -> queryable event layer

Orchestrator: claude-opus-5[1m] | session 878e7b9a-447e-4889-8fbc-21bbd7972498 | 2026-08-10

Container: `ecea07a999f3` (repo `masuda-masuo/ashiato` already cloned at `/workspace`).
Attach with `sandbox_attach` and work there. The repo currently contains only `README.md`.

## What this project is

`ashiato` reads Claude Code session transcripts and turns them into a queryable database,
so that a human can see what a coding agent actually did.

This job builds **only the foundation layer**: transcripts in, DuckDB out. Analysis
features are explicitly out of scope (see Non-goals).

**No LLM anywhere in this code.** Extraction must be deterministic: the same input files
must produce the same tables every time. The tool must also **never make a network
call** — transcripts contain secrets and must not leave the machine.

## Input format

Claude Code writes one JSONL file per session under `~/.claude/projects/<mangled-cwd>/<session-uuid>.jsonl`.
One JSON object per line.

These facts were measured on a real 337 MB corpus on 2026-08-10. **They are given so you
do not have to go find them — you have no access to real transcripts.** Trust them.

Record types seen in one representative file (n=239 lines):

```
assistant 89, user 59, file-history-snapshot 17, mode 16, system 16,
ai-title 16, last-prompt 15, attachment 9, file-history-delta 2
```

Top-level keys seen in that same file, with occurrence counts:

```
type 239, sessionId 220, timestamp 175, parentUuid 173, isSidechain 173,
uuid 173, userType 173, entrypoint 173, cwd 173, version 173, gitBranch 173,
message 148, session_id 133, requestId 89, effort 89, promptId 57,
toolUseResult 39, sourceToolAssistantUUID 39, messageId 19, isMeta 18,
snapshot 17, isSnapshotUpdate 17, attributionMcpServer 17, attributionMcpTool 17,
mode 16, subtype 16, aiTitle 16, permissionMode 15, origin 15, promptSource 15,
lastPrompt 15, leafUuid 15, durationMs 15, messageCount 15, mcpMeta 13,
attachment 9, snapshotMessageId 2, trackingPath 2, backup 2, content 1, level 1
```

Note that **both `sessionId` and `session_id` occur** — different record types use
different casing. Handle both.

`message.usage`, present on every assistant message, always carries these keys:

```
input_tokens, cache_creation_input_tokens, cache_read_input_tokens, output_tokens,
service_tier, cache_creation, inference_geo, server_tool_use, iterations, speed
```

Tool calls live in `message.content[]` as blocks with `"type": "tool_use"`
(fields: `id`, `name`, `input`) and their outcomes come back as blocks with
`"type": "tool_result"` (fields: `tool_use_id`, `content`, `is_error`).

## Tables to produce

### `sessions` — one row per transcript file

| column | source |
|---|---|
| `session_id` | `sessionId` or `session_id` |
| `file_path` | absolute path of the JSONL file |
| `project_dir` | parent directory name, e.g. `-home-masuda-dev-projects-claude` |
| `cwd`, `git_branch`, `cc_version`, `entrypoint` | last non-null value seen in the file (`version` -> `cc_version`) |
| `started_at`, `ended_at` | min / max `timestamp` |
| `n_events`, `n_tool_calls` | counts |
| `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_creation_tokens` | summed **after dedup**, see below |

### `events` — one row per JSONL line

| column | notes |
|---|---|
| `event_id` | from `uuid`. Some record types have no `uuid` — synthesize `"{file_path}:{lineno}"` |
| `session_id`, `seq` (1-based line number), `ts`, `type`, `role` | `role` from `message.role` when present |
| `parent_uuid` | the tree edge |
| `depth` | ancestry depth. See the performance note below |
| `is_sidechain`, `is_meta` | booleans, default false |
| `permission_mode`, `effort`, `request_id`, `message_id`, `model` | null when absent |
| `cwd`, `git_branch` | |
| `text` | concatenation of all `message.content[].text` blocks; empty string when none |
| `raw` | the original JSON line verbatim. Never drop information |

### `tool_calls` — one row per tool invocation and its outcome

**This is the primary table.** Built by joining each `tool_use` block to its `tool_result`
on `tool_use_id`. The join crosses events: the call is on an assistant line, the result on
a later user line.

| column | notes |
|---|---|
| `tool_use_id` | primary key |
| `session_id`, `seq`, `ts` | taken from the calling (assistant) event |
| `call_event_id`, `result_event_id` | both sides of the join; `result_event_id` null when unmatched |
| `tool_name` | e.g. `Bash`, `mcp__sunaba__publish` |
| `tool_kind` | `mcp` when `tool_name` starts with `mcp__`, else `builtin` |
| `mcp_server` | second `__`-delimited segment for MCP tools (`sunaba`, `shiori`), else null |
| `input` | the tool input object, stored as JSON |
| `outcome` | `ok` / `error` / `denied` / `pending`, rules below |
| `is_error` | from `tool_result.is_error`, default false |
| `result_text` | result content flattened to text, truncated to a configurable limit (default 4000 chars) |
| `result_truncated` | boolean |
| `duration_ms` | `result.ts - call.ts` in milliseconds; null when either side is missing |
| `permission_mode`, `cwd`, `is_sidechain` | carried from the calling event |
| `parent_tool_use_id` | from `sourceToolAssistantUUID` on the event, for subagent attribution |

#### `outcome` rules, in this order

1. `pending` — no matching `tool_result` exists (the session ended mid-call)
2. `denied` — the result text matches a denial pattern
3. `error` — `is_error` is true
4. `ok` — otherwise

Denial patterns must live in **a single named constant**, not as literals scattered
through the code, and must be overridable by the caller. These are Claude Code strings
and they will drift between versions. Two confirmed forms, matched as substrings:

```
The user doesn't want to proceed with this tool use
Permission for this action was denied by the Claude Code auto mode classifier
```

### `source_files` — build bookkeeping

Enough to make `build` incremental: path, size, mtime, hash-or-equivalent, rows produced,
last build timestamp.

## Token dedup

`usage` is duplicated across several lines that belong to the same request. Summing
naively inflates totals by roughly 2-3.5x. **Deduplicate by `request_id` before summing.**
Rows without a `request_id` are counted once each.

## Performance note (measured — do not re-benchmark, but do heed)

Full parse of the 337 MB corpus takes 0.9s. Performance is a non-issue at this scale and
you should write the clear version rather than the fast one, with one exception that was
measured and does matter:

Computing `depth` by walking `parent_uuid` to the root for every node costs **3.41s**,
because real corpora contain chains up to depth 2,398 (28.8M steps in total). The
memoized O(n) form costs **0.05s**. **Use the memoized form**, where each node's depth is
computed once and reused by its descendants.

## CLI surface

Package `ashiato`, console script `ashiato`. Exactly three subcommands.

```
ashiato build [--source DIR]... [--db PATH]
ashiato sql "SELECT ..." [--db PATH] [--format table|json|csv]
ashiato info [--db PATH]
```

- `--source` defaults to `~/.claude/projects`, is repeatable, and accepts a directory that
  is searched recursively for `*.jsonl`.
- `--db` defaults to `$XDG_DATA_HOME/ashiato/ashiato.duckdb`, falling back to
  `~/.local/share/ashiato/ashiato.duckdb`. Create parent directories as needed.
- `build` is incremental: a file whose path, size and mtime are unchanged since the last
  build is skipped. Report how many files were processed and how many skipped.
- `info` prints the database path, per-table row counts, and the time window covered.
- `sql` prints the result. `table` is the default format.

## Robustness

- A line that fails to parse is **skipped and counted**, never fatal. Report the count at
  the end of `build`.
- A truncated final line is normal — the session may still be live. Same treatment.
- A file with zero valid lines produces no session row and is not an error.
- Missing optional fields are null, not crashes. Assume every field above can be absent.

## Deliverables (files that must change; notes and summaries are NOT deliverables)

- `pyproject.toml`
- `src/ashiato/__init__.py`
- `src/ashiato/cli.py`
- `src/ashiato/parser.py`
- `src/ashiato/schema.py`
- `src/ashiato/build.py`
- `tests/fixtures/` (synthetic JSONL fixtures, see below)
- `tests/test_parser.py`
- `tests/test_build.py`
- `tests/test_cli.py`
- `README.md`

The module split above is a suggestion for where things go, not a contract — if the code
lands better in a different arrangement, do that and say so. What is fixed is the CLI
surface, the table columns, and the behaviour.

Producing a summary document, a copy of this brief, or notes files is **not** the task and
does not count as work.

## Tests

Write **synthetic** fixture JSONL under `tests/fixtures/`. Do not attempt to obtain real
transcripts; you do not have them and they contain secrets.

Fixtures must cover, at minimum:

- an assistant `tool_use` with a matching successful `tool_result`
- a `tool_result` with `is_error: true`
- both denial strings, producing `outcome = denied`
- a `tool_use` with no result at all, producing `outcome = pending`
- several rows sharing one `request_id`, so the dedup path is exercised and a naive sum
  would give a visibly different (larger) number
- one record using `sessionId` and one using `session_id`
- a malformed line, and a truncated final line
- an MCP tool name, so `tool_kind` / `mcp_server` are exercised
- a parent chain deep enough that memoized and naive depth must agree

Assert on values, not just on "no exception". The dedup test in particular must fail if
dedup is removed.

## Non-goals (do not implement these here)

- Risky-action or dangerous-command detection
- "What happened after a denial" analysis
- Any report, dashboard, or scoring
- The Claude Code plugin wrapper
- Any LLM call, any network call, any telemetry
- Performance optimization beyond the memoized depth described above

If any of these turns out to be genuinely unavoidable to make the foundation coherent,
**say so explicitly in your report** rather than quietly adding it.

## Constraints

- Python 3.11+, `src/` layout, `pyproject.toml`
- Runtime dependencies: `duckdb` only. Use stdlib `argparse` — do not add `click` or `typer`.
- Dev dependencies: `pytest`, `ruff`
- Configure `ruff` in `pyproject.toml` and keep the tree clean under it

## Smoke (run in the container; all must exit 0)

```
python -c "import ashiato; print(ashiato.__version__)"
ruff check .
pytest -q
ashiato build --source tests/fixtures --db /tmp/smoke.duckdb
ashiato sql --db /tmp/smoke.duckdb "SELECT outcome, count(*) FROM tool_calls GROUP BY 1 ORDER BY 1"
ashiato info --db /tmp/smoke.duckdb
```

The `sql` command above must print a row for each of `denied`, `error`, `ok`, `pending` —
if any is missing, the fixtures do not cover the outcome rules and the job is not done.

## Acceptance beyond this container

The orchestrator will run the finished CLI against a real 337 MB / 251-file corpus on the
host. Your synthetic fixtures stand in for that boundary, so passing them is necessary but
not sufficient. Code defensively against real-world shapes you cannot see from here.
