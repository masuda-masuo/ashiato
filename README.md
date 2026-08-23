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
ashiato build [--source DIR]... [--opencode-source DIR]... [--cursor-source DIR]... [--kaiba-db PATH] [--db PATH]
ashiato sql "SELECT ..." [--db PATH] [--format table|json|csv]
ashiato denials [--db PATH] [--format table|json|csv] [--limit N] [--session ID]
ashiato recalls [--db PATH] [--format table|json|csv] [--limit N] [--session ID]
ashiato info [--db PATH]
ashiato salvage [--db PATH] [--kaiba-db PATH] [--window-minutes N] [--limit N] [--since TS]
ashiato grep PATTERN [--db PATH] [--format table|json|csv] [--role user|assistant] [--since TS] [--until TS] [--session PREFIX] [-i|--ignore-case] [--tool-calls] [--include-meta] [--context N] [--all-matches] [--whole] [--limit N]
```

- `--source` defaults to `~/.claude/projects`, is repeatable, and is searched recursively
  for `*.jsonl` Claude Code transcripts.
- `--opencode-source` is repeatable and is searched recursively for `*.ndjson` opencode
  job event streams (`~/.kusabi/*/jobs/*/events.ndjson` on a machine that has them). It is
  a separate list on purpose: the two formats live in unrelated directory trees, and a
  second explicit list means ashiato never has to sniff a file's format to know which
  parser to run. Nothing is scanned for `*.ndjson` by default -- pass `--opencode-source`
  to opt in.
- `--cursor-source` is repeatable and is searched recursively for `*.jsonl` Cursor
  agent-transcript files (`~/.cursor/projects/<project>/agent-transcripts/<id>/<id>.jsonl`
  on a machine that has them). A separate list for the same reason as `--opencode-source`:
  Cursor keeps its own directory tree, so a second explicit list means it never gets
  swept into the plain `--source` scan even though both use the `*.jsonl` extension.
  Nothing is scanned for Cursor transcripts by default -- pass `--cursor-source` to opt
  in. `--kaiba-db PATH` (default `~/.kaiba/kaiba.db`) is only read when `--cursor-source`
  is given: a Cursor transcript carries no tool results at all, so a recall call's
  `output` and `ts` are reconstructed by joining its query against kaiba's own `recalls`
  ledger instead (see `recall_calls` below). A kaiba db that does not exist or cannot be
  read does not fail the build -- affected rows simply get `NULL` `output` / `ts`, and
  `build` prints one line saying so.
- `--db` defaults to `$XDG_DATA_HOME/ashiato/ashiato.duckdb`, falling back to
  `~/.local/share/ashiato/ashiato.duckdb`. Parent directories are created as needed.
- `build` is incremental: a file whose path, size and mtime are unchanged since the last
  build is skipped. A changed file has its old rows deleted and is re-inserted whole, so
  rebuilding never duplicates. This applies uniformly to both source formats.
- `denials` prints the `denial_followups` view — every denied tool call and what the
  session did next — newest first, 50 rows by default (`--limit 0` for all).
- `recalls` prints the `recall_followups` view — every completed kaiba `recall` call, from
  any of the three source formats, and the evidence of what the session did afterwards —
  same output conventions as `denials`.
- `salvage` nominates work-state changes that left no bookkeeping trail. When a session ends
  abnormally (a freeze, a kill, context exhaustion), the post-action bookkeeping a healthy
  session does — recording the state change in the shared kaiba agenda after a chain
  terminates or a publish is confirmed — silently does not happen, even though the evidence
  of the change (the tool call itself) is still in the transcript. `salvage` scans
  `tool_calls` for such evidence and reports each as a *nomination candidate* for a human or
  orchestrator to adjudicate. It is report-only: it writes to nothing — not the kaiba agenda,
  not the actions ledger, not any ashiato table or view — mirroring the discipline of
  `recalls` and `denials`, where derivation nominates and the inspecting tier files. A
  candidate is nominated only when *both* of these hold: (1) no successful
  `mcp__kaiba__agenda_edit` call exists in the same session at or after the evidence
  timestamp, and (2) the kaiba `actions` ledger has no row whose `created_at` or `done_at`
  falls in `[ts, ts + window]` — coverage from a different session or agent. `--window-minutes
  N` sets that coverage window in minutes (default 30); `--since TS` restricts evidence to
  timestamps at or after an ISO-8601 instant; `--limit N` caps nominations printed, with `0`
  meaning all (default 50); `--kaiba-db PATH` points at the kaiba actions ledger (default
  `~/.kaiba/kaiba.db`). When that ledger is absent or unreadable, the second check is skipped
  rather than failed: `salvage` falls back to transcript-only coverage and says so
  (`notice: no kaiba db … — coverage is transcript-only` on stderr), so the session check
  alone decides. Exit code is `0` on success (including an empty result) and `1` when the
  ashiato database cannot be read — missing, out of date, or failing the query.
- `grep` is a regex search over the transcript corpus — the ergonomic layer every "where did
  I analyse X" investigation otherwise hand-rolls as SQL plus a print loop, without reaching
  for `ashiato sql`. By default it searches `events.text`; `--tool-calls` extends the search
  to `tool_calls.input_summary` and `tool_calls.result_text`. `is_meta` events (harness
  noise) are excluded unless `--include-meta` is given. The *pattern* is a Python `re` regular
  expression, **not** DuckDB's RE2 dialect: matching is done in Python precisely so the
  command can report the *offsets* of each match within a row's text — `re.finditer` yields
  them for free once the pattern is compiled once — and so patterns using features RE2 lacks
  (backreferences, certain lookarounds) behave as the Python documentation promises rather
  than silently differing. For each hit, `grep` prints a bounded *window* of text around the
  match: up to `--context N` characters on each side (default 200), with embedded newlines
  replaced so the window stays on one line. `--all-matches` prints a window per match in a row
  instead of only the first; `--whole` prints the entire matched field instead of a window.
  Beyond the search scope there are the usual filters: `--role user|assistant` (events only —
  a tool call has no role to filter on), `--since TS` / `--until TS` (ISO-8601 bounds),
  `--session PREFIX` (session id prefix), and `-i` / `--ignore-case`. `--format
  table|json|csv` chooses the output shape (default `table`; json and csv report the matched
  `id`, `source`, `session_id`, `ts`, `label`, `field`, `offsets` and `text`), and `--limit N`
  caps hits with `0` meaning all (default 20). Exit codes: `0` when matches are printed, `1`
  when nothing matched (`notice: no matches` on stderr), and `2` on a bad pattern or a
  database that cannot be read — missing, out of date, or failing the query.

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
2. `denied` — the result text starts with a denial pattern (leading whitespace ignored)
3. `error` — `is_error` is true
4. `ok` — otherwise

Denial patterns live in one constant, `ashiato.parser.DENIAL_PATTERNS`. They are matched as
*prefixes* of the result text, not substrings: a successful result that merely contains one
of the strings somewhere (a read or diff of sources that quote them, a grep whose output
matched them) is a success, not a denial. They are Claude Code strings and will drift between
versions, so `parse_file(..., denial_patterns=(...))` takes a replacement, matched the same
way. `result_text` is truncated to `result_text_limit` (default 4,000 chars) with
`result_truncated` recording whether that happened; the denial verdict is made on the whole
text, before truncation.

### `recall_calls` — one row per completed kaiba `recall` call

`recall_id`, `session_id`, `file_path`, `source`, `seq`, `ts`, `call_id`, `query`, `output`,
`output_truncated`, `followup_text`, `followup_truncated`, `overlap_tokens`, `overlap_count`.

Filled at build time by `ashiato.recall`, from any of three source formats (`source`
records which): `mcp__kaiba__recall` tool_use/tool_result pairs in a Claude Code
transcript, completed `kaiba_recall` `message.part.updated` records in an opencode
events.ndjson file, or `CallMcpTool` blocks (`server="kaiba"`, `toolName="recall"`) in a
Cursor agent transcript. `query` and `output` are the call's input and returned text;
`followup_text` is a bounded (30 items or 8,000 characters, whichever comes first)
concatenation of the same session's activity on strictly later lines -- assistant text
and other completed tool calls -- the same "strictly later line" rule `denial_followups`
uses, so a call issued in parallel with the recall is never mistaken for a reaction to it.

Cursor is a special case: its transcripts carry no tool results at all (no `tool_result`
blocks, ever), so `output` and `ts` cannot come from the transcript the way they do for
the other two sources. Instead they are reconstructed from kaiba's own `recalls` ledger
(`~/.kaiba/kaiba.db`, read via `--kaiba-db`): the n-th occurrence of a query *within one
Cursor transcript file* pairs with the n-th `agent = 'cursor'` row for that query, ordered
by `created_at`, and `output` is the joined `content` of that row's `matches`, in
`matches` order. Occurrences are counted per file, not across the build: two different
transcript files that issue an identical query text both pair with the same ledger rows,
so the `output` of one may be another session's returned text -- a documented limitation,
not something ashiato tries to disambiguate. A query with no ledger row, or more
transcript occurrences than ledger rows, still gets a row -- just with `output` / `ts`
left `NULL`. Followup evidence for a Cursor row is
correspondingly narrower: assistant text and other tool calls' *inputs* only (rendered as
`server/toolName` plus arguments for another MCP call, or the tool's own name plus its
input otherwise) -- never a tool's output, since Cursor transcripts do not carry one.

`overlap_tokens` / `overlap_count` are one deterministic, mechanical "was this used" signal:
tokens matching `[A-Za-z0-9_#./-]{4,}` present in `output` *and* in the post-recall suffix
*and* absent from the session's pre-recall activity -- "introduced by the recall" is what
makes a token distinctive. `overlap_tokens` is a JSON array capped at 20 tokens for
readability; `overlap_count` is the true, uncapped total. This is a nomination, not a
verdict -- a human reads the evidence and decides.

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

### `recall_followups` — a thin view over `recall_calls`

Unlike `denial_followups`, this is not derived on read: the followup pairing crosses three
source formats (Claude Code's tool_use/tool_result pairs, opencode's message parts, and
Cursor's transcript-plus-kaiba-ledger join), so there is no single raw table to define a
read-time view over. `ashiato.recall` computes the pairing once, at build time, into
`recall_calls`; `recall_followups` just selects from it with a stable column order, the
same shape `denial_followups` presents.

```
$ ashiato recalls --limit 5
$ ashiato sql "SELECT session_id, overlap_count FROM recall_followups ORDER BY overlap_count DESC"
```

Adding `input_summary` changed the `tool_calls` schema, so a database built by an earlier
version is refused with a message rather than half-upgraded: delete it and `build` again.
`sql`, `denials` and `recalls` check the same thing when they open a database, so reading an
old one says how to fix it instead of reporting a bare catalog error. A query of your own
that names something that does not exist still gets DuckDB's error, untouched.

The check is not only about columns: `outcome` is a stored column, so a change to the rules
that derive it (such as the denial patterns becoming anchored prefixes) also makes a database
out of date, and so does an entire table being new (`recall_calls`, added alongside this
view). Every build stamps the row-rule version it used into a small `ashiato_meta`
table, and a database stamped by a version with different rules — or not stamped at all — is
refused with the same delete-and-rebuild message: the incremental build would otherwise skip
every unchanged file and keep rows derived under the old rules, or simply be missing a table
this version expects.

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
