# ashiato

Trace what your coding agent actually did — deterministic analysis of Claude Code session logs.

`ashiato` reads Claude Code session transcripts (`~/.claude/projects/**/*.jsonl`) and turns
them into a DuckDB database you can query with SQL. The interesting table is `tool_calls`:
one row per tool invocation joined to its outcome, so "what did it run, and what came back"
is a single query.

## Guarantees

- **No LLM, anywhere.** Extraction is plain parsing; the same input files always produce the
  same tables.
- **No network call, ever.** Transcripts contain secrets and never leave the machine. DuckDB
  extension autoinstall is switched off explicitly.
- **Nothing is dropped.** Every parsed line keeps its verbatim JSON in `events.raw`, so
  anything this schema does not model is still there to query.

## Install

```
pip install -e ".[dev]"
```

Python 3.11+. The only runtime dependency is `duckdb`.

## Use

```
ashiato build [--source DIR]... [--db PATH]
ashiato sql "SELECT ..." [--db PATH] [--format table|json|csv]
ashiato denials [--db PATH] [--format table|json|csv] [--limit N] [--session ID]
ashiato info [--db PATH]
```

- `--source` defaults to `~/.claude/projects`, is repeatable, and is searched recursively
  for `*.jsonl`.
- `--db` defaults to `$XDG_DATA_HOME/ashiato/ashiato.duckdb`, falling back to
  `~/.local/share/ashiato/ashiato.duckdb`. Parent directories are created as needed.
- `build` is incremental: a file whose path, size and mtime are unchanged since the last
  build is skipped. A changed file has its old rows deleted and is re-inserted whole, so
  rebuilding never duplicates.
- `denials` prints the `denial_followups` view — every denied tool call and what the
  session did next — newest first, 50 rows by default (`--limit 0` for all).

```
$ ashiato build
database: /home/you/.local/share/ashiato/ashiato.duckdb
files: 251 processed, 0 skipped (unchanged), 251 found
rows: 251 sessions, 412884 events, 61240 tool calls
unparseable lines skipped: 3

$ ashiato sql "SELECT tool_name, outcome, count(*) FROM tool_calls GROUP BY 1,2 ORDER BY 3 DESC LIMIT 5"
$ ashiato sql "SELECT tool_use_id, input->>'\$.command' AS cmd FROM tool_calls WHERE tool_name='Bash' AND outcome='denied'"
$ ashiato info
```

## Tables

### `sessions` — one row per transcript file

`session_id`, `file_path`, `project_dir`, `cwd`, `git_branch`, `cc_version`, `entrypoint`,
`started_at`, `ended_at`, `n_events`, `n_tool_calls`, `input_tokens`, `output_tokens`,
`cache_read_tokens`, `cache_creation_tokens`.

`cwd` / `git_branch` / `cc_version` / `entrypoint` are the last non-null value seen in the
file. Token counts are **deduplicated by `request_id`** before summing: the same `usage`
object is repeated across several lines of one request, and summing naively inflates totals
by roughly 2–3.5×. Lines with no `request_id` are counted once each.

### `events` — one row per JSONL line

`event_id`, `session_id`, `file_path`, `seq`, `ts`, `type`, `role`, `parent_uuid`, `depth`,
`is_sidechain`, `is_meta`, `permission_mode`, `effort`, `request_id`, `message_id`, `model`,
`cwd`, `git_branch`, `text`, `raw`.

`event_id` comes from `uuid`; record types that carry no uuid (`file-history-snapshot`,
`mode`, `ai-title`, …) get a synthesized `"{file_path}:{lineno}"`. `depth` is ancestry depth
along `parent_uuid`, computed once per node and reused by its descendants — real corpora
reach chains ~2,400 deep, so the walk is both memoized and iterative.

### `tool_calls` — one row per tool invocation and its outcome

`tool_use_id`, `session_id`, `file_path`, `seq`, `ts`, `call_event_id`, `result_event_id`,
`tool_name`, `tool_kind`, `mcp_server`, `input`, `input_summary`, `outcome`, `is_error`,
`result_text`, `result_truncated`, `duration_ms`, `permission_mode`, `cwd`, `is_sidechain`,
`parent_tool_use_id`.

Built by joining each `tool_use` block to its `tool_result` on `tool_use_id`; the call is on
an assistant line and the result on a later user line. `seq`, `ts`, `permission_mode`, `cwd`
and `is_sidechain` come from the calling event. `input` is a DuckDB `JSON` column, so
`input->>'$.command'` works.

`input_summary` is one short line saying what the call asked for, so you can read a list of
calls without knowing each tool's argument shape: the `command` of a `Bash`, the `file_path`
of a `Read`/`Write`/`Edit`, the `pattern` of a `Grep`, and so on
(`ashiato.parser.INPUT_SUMMARY_FIELDS` is the whole table). Any other tool — every MCP tool,
anything Claude Code adds later — falls back to the compact JSON of the input, as does a call
whose named field is missing or is not a string. Summaries are collapsed to one line and cut
to 200 characters; `NULL` means the call carried no input at all.

`outcome` is decided in this order:

1. `pending` — no matching `tool_result` exists (the session ended mid-call)
2. `denied` — the result text matches a denial pattern
3. `error` — `is_error` is true
4. `ok` — otherwise

Denial patterns live in one constant, `ashiato.parser.DENIAL_PATTERNS`. They are Claude Code
strings and will drift between versions, so `parse_file(..., denial_patterns=(...))` takes a
replacement. `result_text` is truncated to `result_text_limit` (default 4,000 chars) with
`result_truncated` recording whether that happened.

### `source_files` — build bookkeeping

`file_path`, `size_bytes`, `mtime`, `content_hash`, `n_events`, `n_tool_calls`,
`n_parse_errors`, `built_at`. This is what makes `build` incremental.

## Views

### `denial_followups` — what happened after a tool call was denied

One row per `outcome = 'denied'` call in `tool_calls`, joined to the next tool call in the
same session: `session_id`, `seq`, `ts`, `tool_name`, `input_summary`, `permission_mode`,
`cwd`, `next_tool_name`, `next_input_summary`, `next_outcome`, `next_ts`, `gap_seconds`,
`followup_kind`.

A view, not a table: it is derived entirely from `tool_calls`, so it cannot fall out of step
with the rows it summarises and the incremental build has nothing extra to maintain.

"Next" is the first tool call of the same `session_id` on a *strictly later* transcript line
— not the `parentUuid` tree, not sidechain structure. `seq` is the transcript line number, so
several `tool_use` blocks emitted on one assistant line share it. Those siblings were all
issued before the model saw any of their results, so a sibling is never a reaction to the
denial; requiring a later line excludes it by construction, and a denial whose line is the
session's last is `none` even when siblings sit beside it. `tool_use_id` then picks between
the calls of that later line: block order within a line is not recorded anywhere, so that
tiebreak is a *stable* choice rather than a faithful one — it is there so two builds of the
same bytes agree, and it now only ever chooses between calls issued at the same moment.

`followup_kind` is mechanical, never a judgement about whether the retry was legitimate:

| value | meaning |
| --- | --- |
| `verbatim-retry` | same `tool_name`, byte-identical `input` |
| `same-tool` | same `tool_name`, different `input` (a narrowed or corrected retry) |
| `other-tool` | a different tool |
| `none` | no later line of the session called a tool; every `next_*` column is `NULL` |

```
$ ashiato denials --limit 5
$ ashiato sql "SELECT followup_kind, count(*) FROM denial_followups GROUP BY 1 ORDER BY 2 DESC"
```

Adding `input_summary` changed the `tool_calls` schema, so a database built by an earlier
version is refused with a message rather than half-upgraded: delete it and `build` again.
`sql` and `denials` check the same thing when they open a database, so reading an old one
says how to fix it instead of reporting a bare catalog error. A query of your own that
names something that does not exist still gets DuckDB's error, untouched.

## Robustness

A line that fails to parse is skipped and counted, never fatal — including the truncated
final line of a session that is still being written; `build` reports the total at the end. A
file with zero valid lines produces no session row and is not an error. Every field above can
be absent: missing values become NULL rather than exceptions.

## Notes

- Timestamps are stored as naive UTC `TIMESTAMP`.
- Rows are loaded through DuckDB's JSON reader, not one `INSERT` per row: row-at-a-time
  insert costs ~0.6 ms per row in DuckDB whatever the table's width, which would turn a
  337 MB corpus into a half-hour build. The batch is staged as newline-delimited JSON in a
  private temp directory (mode `0700`, removed when the build finishes) and read back — about
  140× faster. Small batches use the plain path, and any failure of the fast path falls back
  to it, so the slow way is always the safety net.
- `tool_use_id` is the key of `tool_calls` but is not declared as a SQL `PRIMARY KEY`: a real
  corpus can contain the same id twice (a transcript copied or replayed across files) and a
  constraint violation there would abort a build over data that is merely redundant.
  Duplicates are dropped per file at parse time.
- An event whose own line omits the session id inherits the file's, so joins hold.
- `events.text` joins multiple text blocks with newlines.

## Development

```
ruff check .
pytest -q
```

Fixtures under `tests/fixtures/` are synthetic — no real transcript is ever committed.
