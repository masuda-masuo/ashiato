# ashiato

[![CI](https://github.com/masuda-masuo/ashiato/actions/workflows/ci.yml/badge.svg)](https://github.com/masuda-masuo/ashiato/actions/workflows/ci.yml)

Trace what your coding agent actually did — deterministic analysis of Claude Code session logs.

`ashiato` reads Claude Code session transcripts (`~/.claude/projects/**/*.jsonl`) and turns
them into a DuckDB database you can query with SQL. The interesting table is `tool_calls`:
one row per tool invocation joined to its outcome, so "what did it run, and what came back"
is a single query.

## Guarantees

- **No LLM, anywhere.** Extraction is plain parsing; the same input files always produce the
  same tables.
- **No network call by default.** Transcripts contain secrets and never leave the machine. DuckDB
  extension autoinstall is switched off explicitly. (`ashiato serve` listens on loopback
  only; accepting a connection from this machine is not an outbound call, and it fetches
nothing.) The single exception is `pending --gh`, which lists the owner's
  repositories once (`gh repo list <owner> --limit 200 --json name`) and checks
  issue/PR references through the `gh` CLI; the only data that leaves the
  machine is the owner name and `owner/repo/number` — never transcript text.
- **Source coverage is per format, not unconditional.** Which tables a transcript format
  populates, and what `events.raw` holds there, differs per source -- a count over
  `tool_calls` means "Claude Code plus Codex", never "all four agents":

  | source | populates | `events.raw` holds |
  | --- | --- | --- |
  | Claude Code | `sessions`, `events`, `tool_calls`, `recall_calls` | the verbatim JSON of the transcript line -- nothing a Claude transcript contains is dropped |
  | Codex | `sessions`, `events`, `tool_calls`, `recall_calls` | the extracted message text (not the source line) for text events; the verbatim item JSON for `context_compaction` rows -- lifecycle, turn-context and token-usage records are not modelled at all |
  | opencode | `sessions`, `events`, `tool_calls`, `recall_calls` | the extracted assistant message text (not the source line) for text events -- `ts` is NULL there, since opencode text parts carry no timestamp; one `sessions` row per distinct session id in the file (tool parts and text parts carry their own); one `tool_calls` row per *terminal* tool part -- `completed` or `error`, with the failure message in `result_text` and `duration_ms` from the part's `time`; job lifecycle events and `pending`/`running` tool parts are not modelled at all |
  | Cursor | `sessions`, `events`, `tool_calls`, `recall_calls` | the extracted message text (not the source line) for text events -- the transcript export records no role, so these rows carry `role` NULL and their text is a mix of user and assistant messages; `ts` is NULL there, since the agent-transcript export records no timestamps at all (`seq` / `block_index` order the rows within a file instead); one `sessions` row per file (a Cursor transcript file is one session -- its file-name uuid stem); one `tool_calls` row per `tool_use` block -- the export records no tool result whatsoever (no `tool_result` blocks, ever), so with `--cursor-chats-source` the results are filled from Cursor's own undocumented local store (`~/.cursor/chats/<workspace-hash>/<session-uuid>/`): the `meta.json` supplies `cwd` and the session times, and the sibling `store.db` supplies `outcome` / `is_error` / `result_text` / `result_truncated` for every call of a session whose store pairs with the transcript (same tool-call count and elementwise-equal tool names, compared as plain recorded strings -- the store records the same MCP block name the transcript does, `CallMcpTool` or `CallDynamicTool`, both verbatim on both sides, so there is no MCP name equivalence to apply). That verdict is a weaker signal than the other sources' status fields -- a named, documented classifier that reads a dict `error` key, a failing dict `status`, an error-prefixed result string, or a shell result whose first line is a nonzero `Exit code` -- and `ts` still stays NULL, since the store carries no per-message timestamp either. For a paired session the `events` rows are *replaced* by store-derived ones: the store's system / user / assistant messages become one row per `text` / `reasoning` / `redacted-reasoning` part, in conversation order, carrying the real `role`, the system prompt the transcript never had, and the model's reasoning, with `seq` = 0-based store message index, the part index riding in `event_id` (`cursor:store:...:seq:block`), and `is_meta` True only for the `system` role. An unpaired session, and any build without the chats source, keeps the transcript-derived rows exactly as they are. Without the chats source, or for a session the store cannot pair, `outcome` / `is_error` stay NULL, and a call whose fate is genuinely unknown must not read as interrupted (`pending`) or as succeeded (`ok`) |

  `source_files` (per-file bookkeeping) is populated by all four.

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
ashiato schema [TABLE] [--db PATH]
ashiato salvage [--db PATH] [--kaiba-db PATH] [--window-minutes N] [--limit N] [--since TS]
ashiato grep PATTERN [--db PATH] [--format table|json|csv] [--role user|assistant] [--since TS] [--until TS] [--session PREFIX] [-i|--ignore-case] [--tool-calls] [--include-meta] [--context N] [--all-matches] [--whole] [--limit N]
ashiato nominate [--db PATH] [--since TS] [--until TS] [--min-sessions N] [--min-stability F] [--exclude-file PATH] [--max-output-chars N] [--json]
ashiato orphans [--db PATH] [--since TS] [--until TS] [--sink PATH]... [--no-default-sinks] [--min-tf N] [--min-human-chars N] [--min-orphans N] [--include-headless] [--limit N] [--mark-reviewed ID]... [--unmark-reviewed ID]... [--reviewed-file PATH] [--show-reviewed] [--json]
ashiato memory-authors [--db PATH] [--since TS] [--until TS] [--memory-dir PATH]... [--model NAME] [--json]
ashiato pending [--db PATH] [--gh] [--owner NAME] [--repo OWNER/NAME] [--all-summaries] [--since TS] [--until TS] [--show-resolved] [--json]
ashiato hygiene [--db PATH] [--since TS] [--until TS] [--format table|json]
ashiato session-trace SESSION_PREFIX [--db PATH] [--format table|json] [--limit N] [--max-excerpt-chars N]
ashiato topics SESSION_PREFIX [--db PATH] [--window N] [--terms N] [--json]
ashiato compare-periods --period START..END --period START..END [--db PATH] [--format json|table]
ashiato serve [--db PATH] [--host HOST] [--port N] [--sink PATH]... [--no-default-sinks] [--memory-dir PATH]... [--reviewed-file PATH]
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
- `--cursor-chats-source` is repeatable and is searched recursively for `*/*/meta.json`
  Cursor chat metadata (`~/.cursor/chats/<workspace-hash>/<session-uuid>/meta.json` on a
  machine that has them -- Cursor's undocumented local store, one small JSON file per
  session). For every chat meta whose session id matches a Cursor session ingested from
  a transcript, the build fills `sessions.cwd`, `events.cwd` and `tool_calls.cwd` from
  the meta's `cwd`, and `sessions.started_at` / `ended_at` from its `createdAtMs` /
  `updatedAtMs`. The `store.db` sitting next to each meta is read the same way: for a
  session whose tool-call count and tool names match the transcript's, it fills the
  tool-call results (`outcome`, `is_error`, `result_text`, `result_truncated`) from the
  store's `tool-result` parts, and the build reports how many sessions paired, how many
  were skipped (count or name mismatch), and how many tool calls were filled. For such a
  paired session the `events` rows are *replaced*, not added to: keeping the
  transcript-derived text alongside the store's would duplicate nearly all of it (the
  store's user and assistant texts appear verbatim in the transcript), so the store's
  system / user / assistant messages become the session's `events` -- one row per `text`
  / `reasoning` / `redacted-reasoning` part, carrying the real `role`, the system prompt
  and the reasoning, with `seq` = 0-based store message index and the part index in
  `event_id`. An unpaired session keeps the transcript-derived rows exactly as they are.
  The build also reports how many store messages and parts it read and how many parts it
  could not classify (an unknown part type is counted, never dropped silently). Nothing is
  scanned for chat metadata by default -- pass `--cursor-chats-source` to opt in.
- `--codex-source` is repeatable and is searched recursively for `*.jsonl` Codex
  session files (`~/.codex/sessions` on a machine that has them). A separate list
  for the same reason as the others: Codex keeps its own directory tree, so a
  second explicit list means it never gets swept into the plain `--source` scan
  even though both use the `*.jsonl` extension. Nothing is scanned for Codex
  sessions by default -- pass `--codex-source` to opt in.
- `--db` defaults to `$XDG_DATA_HOME/ashiato/ashiato.duckdb`, falling back to
  `~/.local/share/ashiato/ashiato.duckdb`. Parent directories are created as needed.
- `build` is incremental: a file whose path, size and mtime are unchanged since the last
  build is skipped. A changed file has its old rows deleted and is re-inserted whole, so
  rebuilding never duplicates. This applies uniformly to all four source formats.
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

- `nominate` mines re-derived facts as kaiba nomination candidates. It scans
  non-sidechain `Bash` tool calls for two signals: *negative-fact* (the same
  error text rediscovered by many sessions) and *stable-output* (a command
  returning the same informative result across sessions). Ritual commands
  (`--help`, `git pull`, bare `ls`, etc.) are excluded by default; an
  `--exclude-file` supplies environment-specific patterns. This is
  report-only: it never writes to kaiba or any file, mirroring the discipline
  of `salvage` and `denials`. `--min-sessions N` sets the minimum distinct
  sessions (default 3), `--min-stability F` the minimum modal output share for
  `stable-output` (default 1.0), `--since TS` / `--until TS` bound the time
  window, and `--json` outputs full records instead of the default one-line
  table. Exit code `0` when candidates exist, `1` when none.

- `orphans` nominates one-off discussion topics that left no trace. A design chat, a
  "what do you think about X" or a story idea happens once and never reaches a PR, a memory
  file or a ledger, and neither `nominate` (which needs repetition across sessions) nor
  `salvage` (which needs tool-call evidence) can see it. `orphans` finds candidates
  deterministically -- no LLM, no embeddings -- so a reader only has to look at the top few. A
  session is nominated when *all* of these hold: (1) it is **unique** -- it contains terms that
  occur in no other session in the whole database, at least `--min-tf N` times (default 3);
  (2) it is a **discussion** -- its human-typed text, with harness wrappers such as
  `<system-reminder>` and `<local-command-stdout>` blocks stripped, is at least
  `--min-human-chars N` characters (default 800); compaction summaries -- Claude Code's
  machine-written "this session is being continued from a previous conversation" recaps,
  recognised by `"isCompactSummary": true` in the row's `raw` JSON -- are not human text and
  contribute nothing: no characters, no terms, no first utterance; (3) it is **not persisted** -- none of
  those unique terms appears in any *sink*; and (4) it is **worth reading** -- at least
  `--min-orphans N` distinct orphan terms (default 3; density is noisy for tiny counts), and it
  is not a *headless* session (an SDK / headless entrypoint, `sdk-*` -- kusabi workers,
  subagents -- whose
  "human" text is a machine-written brief) unless `--include-headless` is given. `--sink PATH`
  (repeatable) names a file or a
  directory, walked recursively, of text the topic might have been written down in; without
  `--sink` the sinks are every existing `~/.claude/projects/*/memory` directory, and
  `--no-default-sinks` turns that default off. A sink path that does not exist is a warning on
  stderr, not an error, and with no sink text at all every unique term counts as an orphan
  (say so on stderr). Terms are lowercased words of 4+ Latin characters, katakana runs and
  kanji runs; code identifiers (containing `_`), hex/UUID ids and mixed letter+digit tokens
  such as `bk7tgrw6i` or `urllib3` are dropped. Uniqueness is
  always measured over every session that has prose -- headless ones included -- so
  `--since TS` / `--until TS` only
  restrict which sessions are *nominated*, and a window never makes an old topic look unique.
  Candidates are ranked by orphan density (orphan terms per thousand terms), then number of
  orphan terms, then human-typed length, and `--limit N`
  caps them (default 20, `0` for all); each shows the session, its density, its top orphan
  terms and the first thing the human said. `--json` prints one document with the header and
  the same
  fields.
  The same candidate is nominated on every run until someone says it has been read.
  `--mark-reviewed ID` (repeatable) resolves *ID* -- a full session id or a unique prefix --
  and appends the **full** session id to the *reviewed file*, then exits 0 without
  nominating (`reviewed: <id>` per line, or `already reviewed: <id>`); `--unmark-reviewed
  ID` (repeatable) removes the id again (`unmarked: <id>` or `not reviewed: <id>`), and an
  id present in the file still unmarks even when its session is no longer in the database.
  The two flags conflict, any missing or ambiguous id exits 1 having **written nothing**, and
  the nomination never runs in the same invocation. The file is `orphans-reviewed.txt` next
  to the database (`--reviewed-file PATH` overrides it): plain UTF-8, one session id per
  line, blank lines and `#` comments ignored, surrounding whitespace stripped; writing
  preserves existing lines and is idempotent, and a missing file means nothing reviewed.
  It lives *outside* the database on purpose -- a delete-and-rebuild of the database must
  not forget the marks, and the reviewed file is the only thing this command ever writes
  (apart from that, it stays report-only, like `nominate` and `salvage`). A reviewed
  session is skipped from nomination -- every file of that session id is hidden, because the
  key is the session id, not the transcript file -- but it still counts toward document
  frequency exactly like a headless or out-of-window session, so marking a session reviewed
  can never make one of its shared terms look unique to another session. The header reports
  how many would-be candidates the mark hid: `reviewed-hidden=N` on the text header, and
  `reviewed_hidden` plus the resolved `reviewed_file` path in the JSON header.
  `--show-reviewed` nominates reviewed sessions too, flagging each with `[reviewed]` on the
  text session line or `"reviewed": true|false` in JSON, with `reviewed_hidden` 0 then. A
  reviewed file that exists but cannot be read (a directory, permission denied) is an error
  on stderr, exit 1 -- never silently treated as empty. Known limit: *absence of the words is
  not absence of the idea* -- a topic saved under different words is still nominated, and a
  topic whose words happen to appear in a sink is missed. It is a nomination only; judging
  whether a candidate is worth keeping is for the reader. Exit code `0` on success (including
  no candidates) and `1` when the database cannot be read.

- `memory-authors` attributes each Claude Code memory file
  (`~/.claude/projects/<project>/memory/*.md`) to the models that wrote it. Claude Code
  writes these files automatically, and a project shared by several models (fable, opus,
  sonnet, ...) makes session-level attribution wrong: a session switches models mid-way.
  The report instead reads the model off the *write call's own event* (`events.model`
  joined to the `Write` / `Edit` / `MultiEdit` call), so `created_by` is the model of the
  earliest successful `Write` -- or `unknown` when an `Edit`/`MultiEdit` came first (the
  file predates the corpus) or the model is NULL/`<synthetic>`. Per file it reports the
  edit counts per model, first/last write time, total successful writes, whether the most
  recent path still exists on disk, and how many `Bash` calls mentioned the file inside
  `/memory/` (`bash_mentions`). `--since TS` / `--until TS` restrict the calls considered
  (inclusive, same semantics as `orphans`); `--model NAME` lists only files whose creator
  or any editor is that model; `--memory-dir PATH` (repeatable; default every existing
  `~/.claude/projects/*/memory` directory) names the directories whose `*.md` files with
  no recorded write appear under `unattributed`; `--json` prints one document with
  `summary` (per model: files created, writes, files touched), `files` and `unattributed`.
  This is report-only: it never writes anything. Known limit: attribution covers Claude
  Code's `Write` / `Edit` / `MultiEdit` tool calls only -- edits made through Bash or
  scripts are not attributed, so they only surface as `bash_mentions` and a file rewritten
  entirely through the shell has no creator. Codex sessions write through their shell tool
  (recorded as `Bash`), so their memory edits appear as `bash_mentions` at best, and memory
  writes made from Cursor sessions are not visible in the database at all. Exit code `0` on
  success and `1` when the database cannot be read.

- `pending` reports open items left in compaction summaries. When a Claude Code session runs
  out of context it writes a compaction summary with a numbered "Pending Tasks" section — the
  agent's own list of what is left, often with "Discussed-but-not-started (do NOT begin
  without user confirmation)" items and issue numbers. `pending` finds every session's latest
  compaction summary (`--all-summaries` also processes earlier, superseded ones), extracts
  the Pending Tasks section (a summary without one is counted in `no_section`, not listed),
  and lists each item with its references and a status. References resolve in this order: a
  GitHub URL (`https://github.com/<o>/<r>/(issues|pull)/<n>`); an explicit `<o>/<r>#<n>`
  (the owner must start with a letter, so `454/PR#466` is not one); a short form
  (`<name>#<n>`, `<name> #<n>`, `<name> PR #<n>`, `<name> PR#<n>`, `<name> issue #<n>`)
  where `<name>` is a known repository name; and a bare `#<n>` (including `PR #<n>` /
  `Issue #<n>` with no repo name before them). A bare reference resolves in this order:
  first to the `(owner, repo)` of a same-number resolved reference in the same item,
  when every same-number resolved reference in that item names the same repository.
  That is not an inference: the reference is recorded as non-bare (no `via`), exactly
  like the explicit ref it matches, so a 404 on it is `unknown`, not `unresolved`.
  Then to the nearest
  preceding entry in the same item -- a resolved repository or a bare name mention,
  whichever is closest before it -- else `--repo OWNER/NAME`, else it is reported as
  `unresolved` and never checked. A bare word that is a known repository name (not part of
  a path, URL, `owner/repo`, identifier, or longer word such as `shiori-demo`) sets the
  nearest preceding repository for later bare refs in the same item. Bare references carry
  a `via` key indicating how they were resolved: `"nearest"` (preceding resolved ref),
  `"name"` (preceding bare name mention), or `"repo_flag"` (`--repo`). Non-bare refs omit
  `via`. Known repository
  names come from the explicit references found anywhere in the database's summaries, or with
  `--gh` from the owner's repositories (`gh repo list <owner> --limit 200 --json name`, one
  call per run); `--owner NAME` sets the owner those short forms resolve to, defaulting to the
  owner named most often by the explicit references. Without `--gh` every reference state is
  `unchecked` — no subprocess, no network. `--gh` checks each unique `owner/repo/number` once
  with `gh api repos/<o>/<r>/issues/<n>` (read-only GET; a PR counts as `merged` when
  `merged_at` is set) and records `open` / `closed` / `merged` / `unknown` per reference, an
  `unknown` carrying the gh error message. A 404 on an inferred (bare) reference is recorded
  as `unresolved` (the repo was guessed and wrong, not a broken reference); an explicit
  `owner/repo#n` or URL that 404s stays `unknown`. An item is `open` when any reference is
  open, `resolved` when all references are closed/merged, `unreferenced` when it has no
  references, and `unknown`/`unchecked` otherwise. By default only items that are not
  `resolved` are shown; `--show-resolved` shows all. `--since TS` / `--until TS` filter by
  summary timestamp, and `--json` prints one document with `sessions` (session id, summary
  timestamp,
  project, latest `ai-title`/`custom-title`, items) and `counts` — counts by status plus the
  number of summaries with no Pending Tasks section. Item text is truncated to 300
  characters. This is report-only: it writes nothing. Exit code `0` on success and `1` when
  the database cannot be read.

- `hygiene` is a named, read-only audit of session hygiene over `tool_calls`,
  replacing the ad-hoc SQL that used to be rewritten for every such question.
  It counts five stable signals, per category both the tool-call rows and the
  distinct sessions they came from:
  - `companion_status_poll` -- shell calls whose *executed command* invokes
    `kusabi-companion status`, whether the binary is called directly (basename
    `kusabi-companion` or `kusabi-companion.mjs`) or through its `node
    <script>` form (`node .../kusabi-companion.mjs status`). Other companion
    subcommands (`chain-show`, `chain-wait`, `result`), the bare binary or
    bare script, and text that merely quotes the command (in a tool result, in
    an `echo` argument, or in a `Read` of a doc) are not polls.
  - `host_file_hunt` -- shell calls that run `rg`/`grep`/`sed`/`cat` against
    host files. Dedicated file/search tools (`Grep`, `Read`, an MCP search
    tool) are excluded, and hunt words that appear only in a tool result are
    not a hunt.
  - `raw_local_mcp_http` -- shell calls that `curl` loopback
    (`127.0.0.1`/`localhost`) ports 8750/8765/8770. Other ports, remote hosts
    (even on a matching port), and dedicated MCP tool calls are excluded.
  - `undo_file_edit` -- calls whose persisted tool name is an MCP
    `undo_file_edit` tool on any server (`mcp__<server>__undo_file_edit`);
    prose that merely mentions the name is not a call.
  - `pending_tool_call` -- every row with `outcome = 'pending'`, whatever its
    tool name.
  Categories may overlap (a pending loopback `curl` is both `raw_local_mcp_http`
  and `pending_tool_call`). Classification reads only persisted fields --
  `tool_name` and the command text, never `result_text` -- and the shell
  categories tokenize the executed command, so quoted text in an argument does
  not count as an invocation. The shell categories apply to the persisted
  shell-tool names `Bash`, `PowerShell` and `Shell`, matched case-insensitively
  (so `bash` counts); the MCP `undo_file_edit` tool-name match stays
  case-sensitive. `--since TS` / `--until TS` bound the window
  inclusively (either bound excludes rows with a NULL timestamp; without bounds
  they count); `--format table|json` chooses the output (default `table`, and
  deliberately no CSV: this is a fixed structured report, not a dump). Shell
  categories classify the full persisted `input` command when it is available:
  a string command is shell-tokenized, and a Codex-style argv list
  (`"command": ["cat", "/etc/hosts"]`) decodes to its actual argv -- never
  turning arbitrary prose or objects into commands. The exact shell `-c`
  wrapper is unwrapped: `bash -c SCRIPT`, `bash -lc SCRIPT`, and the
  `/bin/bash` / `sh` / `/bin/sh` equivalents are transparent, and SCRIPT is
  tokenized and classified like any other command line. A leading `cd <dir>
  &&` prefix -- the form almost every persisted command starts with -- is
  stripped (repeatedly, so `cd /x && cd /y && cmd` classifies `cmd`), and it
  is stripped after the wrapper is unwrapped too. Compound commands are
  otherwise not descended into: pipes, `;`, `||`, subshells, command
  substitution, and `VAR=value` prefixes stay unclassified, and no arbitrary
  wrapper, variable expansion, or nested command is traversed. The 200-character
  `input_summary` is only the fallback when no usable full command can be
  extracted, so a long command whose loopback MCP URL or
  `kusabi-companion status` invocation sits past the summary truncation
  boundary still counts. For `raw_local_mcp_http` only the curl *request
  target* counts: common options that consume a following value
  (`-H`/`--header`, `-d`/`--data*`, `-F`/`--form`, `--url`, ...) have their
  value handled, so a loopback URL used only as a header or data value is not
  a raw MCP call while the actual loopback target still is. The JSON
  shape is stable: a top-level object with `coverage` (`since`/`until`
  echoing the effective bounds, `null` when absent, pre-filter
  `sessions`/`tool_calls`, the per-source `sources` list, and
  `excluded_no_timestamp` when a window dropped rows that record no
  timestamp) and an ordered `categories` list whose objects each carry
  exactly `name`/`tool_calls`/`sessions`. The table prints the same five
  rows with the same counts, and -- only when a window excluded them -- an
  `excluded: N tool calls have no timestamp (the source records none):
  <source> N` line right after the coverage line. That disclosure matters
  because a windowed `0` is silent evidence: a source that records no
  per-call timestamp (Cursor) drops out of every bounded window, and its
  rows can never be read back as "nothing happened". `hygiene` is the named
  audit for recurring questions; arbitrary one-off investigation remains
  `ashiato sql`.

- `compare-periods` runs the same hygiene audit over two non-overlapping
  windows (`--period START..END` twice, baseline then current) and reports,
  per category, the baseline/current calls and sessions, calls-per-session,
  and the absolute and percent deltas (percent is `n/a` when the baseline is
  zero). Its table output adds the disclosure lines: per period, the same
  `excluded: ...` line when that window dropped NULL-timestamp rows, and one
  `warning: source '<name>' has N calls in current and 0 in baseline; category
  deltas include its entire corpus` line per source that has rows in exactly
  one of the two periods. A source whose ingest was only added part-way
  between the windows shows up as a `0 -> N` delta that is the entire new
  corpus, not a behaviour change; the warning says so instead of letting the
  delta be read as growth. The JSON output carries the structure as data --
  `source_asymmetry` at the top level and per-period `sources` /
  `excluded_no_timestamp` in `periods` -- with no prose inside the JSON. A
  category `0` never means "it did not happen": it means no invocation of
  that form was persisted in the window, and the disclosure lines say what
  the window and its sources left out.

- `session-trace` renders one session as a single interleaved timeline: its
  text events and tool calls ordered by transcript line (`seq`), text rows
  before tool rows on the same line, then the row's own id ascending — the
  stable ordering two builds of the same bytes always agree on. The
  `SESSION_PREFIX` argument is a session id or a unique prefix of it,
  resolved against the union of ids in `sessions` and `tool_calls` (so a
  session persisted only as tool calls, the older Codex shape, still
  resolves); an exact id wins, and a missing or ambiguous prefix is a clean
  error on stderr with exit code 1. `--format table|json` picks the output
  (default `table`); the JSON shape is a stable top-level object with
  `session`, `coverage` (pre/post-limit counts and which persisted tables
  have rows for the session: `has_sessions` / `has_events` /
  `has_tool_calls` / `has_recall_calls`) and an ordered `timeline` whose
  rows are `kind`-specific (`text` rows carry `role` / `excerpt`; `tool`
  rows carry `tool_name` / `input_summary` / `outcome` / `is_recall`, plus
  `recall` annotation with the stored query and overlap signal, and
  `followup` evidence when the call was denied). `--limit N` caps rows
  after ordering with `0` meaning all (default 200); `--max-excerpt-chars
  N` bounds each text/result excerpt to N characters plus a one-character
  marker, with `0` meaning uncapped (default 500); negative values are
  rejected. Meta events (harness noise) are excluded from the timeline, and
  a recall row's follow-up text is assembled from the trace's own rows on
  strictly later lines — never the build-time `recall_calls.followup_text`,
  which may contain harness noise the trace does not display. The command
  is read-only: it opens the database with `read_only` and writes nothing.

- `topics` renders one session as a deterministic topic outline -- no LLM, no
  embeddings -- because a session's stored title (`ai-title`) is generated once
  from the first prompt and never updated, so a long session's label says
  nothing about what it was really about. The outline is derived from the
  session's own words in four mechanical steps: (1) **exchanges** -- the
  session's user/assistant text rows in transcript order, excluding meta and
  sidechain rows, tool-result user rows and compaction summaries (Claude
  Code's machine-written "this session is being continued" recaps, recognised
  by `"isCompactSummary": true` in `raw`), and deduplicated by the row's
  `uuid` so a resumed session that re-persists the same messages does not
  double them; each non-empty human row opens an exchange and the following
  assistant text is appended to it; (2) **weights** -- tf-idf with document
  frequency over the whole database exactly as `orphans` computes it, dropping
  terms that occur in more than 30% of sessions; (3) **boundaries** --
  TextTiling-style: a gap between exchanges is a segment boundary when the
  cosine similarity of the summed `--window N` exchanges on each side (default
  3) is a local minimum below `mean - 0.5 * sd` of all gap similarities, and a
  session with fewer than `2 * window` exchanges is one segment; (4)
  **segments** -- each carries its exchange range, start/end timestamps,
  exchange count, the top `--terms N` topic terms by in-segment tf-idf
  (default 8) and the opening (first human text, collapsed, 160 chars). The
  header carries `session_id`, `project_dir`, the latest `custom-title` if the
  human set one (else the latest `ai-title`) and the exchange count. The
  `SESSION_PREFIX` argument resolves exactly like `session-trace` (exact id
  wins, otherwise a unique prefix; a missing or ambiguous prefix is a clean
  error on stderr, exit code 1). `--json` prints one document with the header
  and the ordered `segments`. This is report-only: it reads the database and
  writes nothing, and the same bytes always produce the same outline.

- `serve` runs a read-only local dashboard so the analyses can be glanced at in a browser
  instead of run one by one. It is a stdlib HTTP server (no new dependency) rendering
  `info`, `orphans`, `memory-authors`, `denials`, `hygiene` and `session-trace` as HTML:
  `/` (overview: table counts, time window, ingested roots, a prominent `STALE` banner when
  files under the recorded roots are newer than the database, and one tile each for orphan
  candidates, unattributed memory files and denied calls), `/orphans` (the top 30
  candidates by density), `/memory` (`?model=NAME` filters like `memory-authors --model`),
  `/denials` (the 50 most recent denials with what the session did next, then the `hygiene`
  categories) and `/session/<id-or-prefix>` (the `session-trace` timeline with the session's
  `topics` outline -- a compact table of time range, topic terms and opening -- above it; an
  unknown or ambiguous prefix is a 404 naming the problem). `/api/orphans.json`, `/api/memory.json` and
  `/api/denials.json` return the same data as JSON, in the shape of the matching command's
  `--json` / `--format json` output. `/api/overview.json` (also `/api/info.json`) has no
  command to mirror, since `info` prints no JSON: it is one object with `db_path`,
  `db_modified`, `table_counts`, `started_at` / `ended_at`, `roots` (`null` when the database
  recorded none), `freshness` (`{"state": "current" | "stale" | "unknown", "gap": N}`), `tiles`
  (`orphan_candidates`, `memory_unattributed`, `denied_calls`) and `top_orphans` (the three
  densest candidates, as `orphans --json` prints them). A memo written into a sink
  directory retires an orphan candidate on the next reload, and so does a session marked
  reviewed with `ashiato orphans --mark-reviewed`: the orphans page reads the same reviewed
  file as the CLI (next to the database, or the `--reviewed-file PATH` override), hides
  reviewed sessions from `/orphans`, `/api/orphans.json` and the overview tile, and the
  orphans cache is keyed by the reviewed file's stamp too, so a mark shows up on the next
  reload without a restart or a rebuild. A memory file that appears
  or disappears shows up as such on the next reload, without waiting for a rebuild. Long
  text is truncated behind a `<details>` element; the only script on the pages filters
  tables client-side, and nothing is loaded from outside (CSS and script are inline).
  It is **loopback-only**: the default `--host` is `127.0.0.1`, `--host` accepts only
  `127.0.0.1`, `::1` or `localhost` (anything else exits `2` before binding, because
  transcripts contain secrets), and requests whose `Host` header names anything but the
  loopback are refused with `403`. It is **read-only except one guarded endpoint**:
  every database value shown is HTML-escaped. The sole write is
  `POST /orphans/reviewed`, which marks or unmarks a session as reviewed
  in the same file the CLI writes. Each `/orphans` row includes a one-click
  "reviewed" button (a standard HTML form, no JavaScript). The write is
  CSRF-guarded: a per-process random token is embedded as a hidden form
  field, and the request's `Origin` header must equal `http://<Host>`
  (the existing loopback Host check applies too). A cross-site form POST
  to 127.0.0.1 cannot forge both. On success the server responds with
  `303 See Other` redirecting to `/orphans`; the row is gone on that
  reload. `?reviewed=1` shows the reviewed sessions with an "unreview"
  button. It uses **per-request connections**: each request opens the
  database read-only and closes it before responding, so a nightly `ashiato build` (which
  needs to write) is never blocked by the dashboard. **During a rebuild** -- or when the
  database is missing or out of date -- pages answer `503` (`Retry-After: 30`) saying the
  database is being rebuilt, and the server keeps running; an unexpected error in a page is
  a `500` naming the error, and the server keeps running too. The slow analyses
  (`orphans`, `memory-authors`) are cached per page, keyed by the database file's
  modification time and size, and recomputed when either changes; the freshness banner is
  not cached, since it compares the transcript directories against the database. `--port`
  defaults to `8772` (`0` picks a free port); the URL is printed on start. `--sink PATH`
  (repeatable), `--no-default-sinks` and `--reviewed-file PATH` are passed to the orphans
  page exactly as
  `ashiato orphans` takes them, and `--memory-dir PATH` (repeatable) as
  `ashiato memory-authors` takes it; by default both use every existing
  `~/.claude/projects/*/memory` directory. There is no authentication and no TLS: the
  loopback bind is the boundary.

- `schema` lists the tables and views in the ashiato schema, or shows the columns
  and types for a specific table or view. It works without a database -- the schema
  is a property of the code, not of any particular build. Pass `--db PATH` to also
  show view columns (which require a database to inspect).
```
$ ashiato schema
  sessions                table
  events                  table
  tool_calls              table
  recall_calls            table
  source_files            table
  denial_followups        view
  recall_followups        view

$ ashiato schema tool_calls
tool_calls (table)
  tool_use_id                   VARCHAR
  session_id                    VARCHAR
  file_path                     VARCHAR
  source                        VARCHAR
  seq                           BIGINT
  ts                            TIMESTAMP
  call_event_id                 VARCHAR
  result_event_id               VARCHAR
  tool_name                     VARCHAR
  tool_kind                     VARCHAR
  mcp_server                    VARCHAR
  input                         JSON
  input_summary                 VARCHAR
  outcome                       VARCHAR
  is_error                      BOOLEAN
  result_text                   VARCHAR
  result_truncated              BOOLEAN
  duration_ms                   BIGINT
  permission_mode               VARCHAR
  cwd                           VARCHAR
  is_sidechain                  BOOLEAN
  parent_tool_use_id            VARCHAR
```

```
$ ashiato build
database: /home/you/.local/share/ashiato/ashiato.duckdb
files: 251 processed, 0 skipped (unchanged), 251 found
rows: 251 sessions, 412884 events, 61240 tool calls
unparseable lines skipped: 3

$ ashiato sql "SELECT tool_name, outcome, count(*) FROM tool_calls GROUP BY 1,2 ORDER BY 3 DESC LIMIT 5"
$ ashiato sql "SELECT tool_use_id, input->>'\$.command' AS cmd FROM tool_calls WHERE tool_name='Bash' AND outcome='denied'"
$ ashiato info
database: /home/you/.local/share/ashiato/ashiato.duckdb
  sessions              251
  events             412884
  tool_calls          61240
  source_files         251
time window: 2026-01-15 08:30:00 .. 2026-08-22 14:22:11
ingested roots:
    sources: /home/you/.claude/projects (245 files)
    opencode_sources: /home/you/.kusabi (6 files)
    cursor_sources: (none)
freshness: 3 new or changed files under recorded roots (run 'ashiato build' to update)

$ ashiato schema
$ ashiato schema tool_calls
```

## Codex item types

Codex session transcripts are parsed the same deterministic way as the other
sources: pure parsing, same bytes always produce the same rows. Of the item
types a session can record, these become rows:

| item type | row |
| --- | --- |
| `CommandExecution` | `tool_calls`, `tool_name` `Bash` |
| `McpToolCall` | `tool_calls`, `tool_name` `mcp__<server>__<tool>` |
| `FileChange` | `tool_calls`, `tool_name` `FileChange`; the touched paths are the `files` key of `input` (and the keys of its `changes`), so `input LIKE '%path%'` finds them |
| `CollabAgentToolCall` | `tool_calls`, `tool_name` `collab__<tool>` (e.g. `collab__wait`); `receiver_agents` / `receiver_thread_ids` / `agents_states` ride in `input` when the item carries them |
| `SubAgentActivity` | `tool_calls`, `tool_name` `collab__subagent`; `kind`, `agent_thread_id`, `agent_path` ride in `input` |
| `AgentResponse` / `message` | `events`, `type` `text` |
| `ContextCompaction` | `events`, its own `type` `context_compaction` (never `text`), so a session's compaction points are queryable and never read as assistant speech |

`Reasoning`, `Extension`, `AgentMessage`, and `UserMessage` are deliberately not
matched on the `item_completed` path.  `Reasoning` (4624 items across the real
corpus) is the model's private scratchpad, not an action; `Extension` items
(web search, on the real sessions read) record a query and its results but no
delegation or state change, so they are left out of the current model.  The
deliberate drops are documented here so the missing rows read as a decision,
not a bug -- and an item type this parser does not know is always dropped
silently, never raised on, so a future Codex version's new item type cannot
break a build.

Codex `tool_calls` rows carry no link to an event row: both `call_event_id`
and `result_event_id` are NULL.  The `item_completed` item stream this parser
consumes uses a different id space from the model-facing `response_item`
`call_id` (measured: 5626 Codex tool_calls rows and 5626 unresolvable
synthetic ids before this change), so no real event id can be recovered from
what this parser sees.  Recovering the link means re-keying Codex tool calls
off the `response_item` stream, which is a separate and much larger change
(issue #74's territory); a NULL that is explained is a decision, an
unexplained one would be a gap.

`AgentMessage` and `UserMessage` are additionally skipped on the
`item_completed` path because Codex records each message **twice**: once as an
`item_completed` item and once as a top-level `response_item`.  The
`response_item` path already ingests all message rows, so matching both would
duplicate them.  Measured over all 55 sessions in the real corpus:

```
item_completed AgentMessage : 1719      response_item message role=assistant : 1735
item_completed UserMessage  :  448      response_item message role=user      :  554
                                        response_item message role=developer :  220
```

The `response_item` path produces all 2509 Codex event rows (1735 + 554 + 220 =
2509 exactly); ingesting `AgentMessage` as well would double-count every
assistant message.

Delegation rows (`CollabAgentToolCall`, `SubAgentActivity`) currently carry
`outcome='pending'` regardless of their `status`, because they set `output=None`
and the classifier treats no output as pending.  All 17 collab items in the
evidence file carry `status="completed"`, but the outcome field does not reflect
that.  This is a known limitation: a future delegation-query surface should
classify these rows by `status` instead.

## Tables

### `sessions` — one row per transcript file

`session_id`, `file_path`, `source`, `project_dir`, `cwd`, `git_branch`, `cc_version`, `entrypoint`,
`started_at`, `ended_at`, `n_events`, `n_tool_calls`, `input_tokens`, `output_tokens`,
`cache_read_tokens`, `cache_creation_tokens`.

`source` records which transcript format produced the session (`claude_code` or `codex`).
`cwd` / `git_branch` / `cc_version` / `entrypoint` are the last non-null value seen in the
file. Token counts are **deduplicated by `request_id`** before summing: the same `usage`
object is repeated across several lines of one request, and summing naively inflates totals
by roughly 2–3.5×. Lines with no `request_id` are counted once each.

### `events` — one row per JSONL line

`event_id`, `session_id`, `file_path`, `source`, `seq`, `ts`, `type`, `role`, `parent_uuid`, `depth`,
`is_sidechain`, `is_meta`, `permission_mode`, `effort`, `request_id`, `message_id`, `model`,
`cwd`, `git_branch`, `text`, `raw`.

`source` records which transcript format produced the row (`claude_code` or `codex`).
`event_id` comes from `uuid`; record types that carry no uuid (`file-history-snapshot`,
`mode`, `ai-title`, …) get a synthesized `"{file_path}:{lineno}"`. `depth` is ancestry depth
along `parent_uuid`, computed once per node and reused by its descendants — real corpora
reach chains ~2,400 deep, so the walk is both memoized and iterative.

### `tool_calls` — one row per tool invocation and its outcome

`tool_use_id`, `session_id`, `file_path`, `source`, `seq`, `ts`, `call_event_id`, `result_event_id`,
`tool_name`, `tool_kind`, `mcp_server`, `input`, `input_summary`, `outcome`, `is_error`,
`result_text`, `result_truncated`, `duration_ms`, `permission_mode`, `cwd`, `is_sidechain`,
`parent_tool_use_id`.

`source` records which transcript format produced the call (`claude_code` or `codex`).
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

Cursor is a special case: its agent-transcript export carries no tool results at all (no
`tool_result` blocks, ever) -- the tool results Cursor does keep live in its undocumented
local store (`~/.cursor/chats/<workspace-hash>/<session-uuid>/`) and are read into
`tool_calls` (for paired sessions, via `--cursor-chats-source`) as of issue #87, but a
`recall_calls` row needs `output` and `ts` the transcript cannot supply -- so they are
reconstructed from kaiba's own `recalls` ledger
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
makes a token distinctive. To keep the signal meaningful, only *distinctive-shape* tokens
count: a token must contain a shape character (`0-9 # _ / . -`) or a camelCase/PascalCase
inner capital, must not be all digits, and must not be a bare ISO date or date-hour prefix.
Ordinary English words (`different`, `green`) and shared calendar references match by chance
across sessions, so they are excluded; identifiers (`deriveDisposition`), paths
(`src/ashiato/recall.py`), issue refs (`kusabi#274`), flags (`--container`) and hashes
(`741d50b`) do not. `overlap_tokens` is a JSON array capped at 20 tokens for readability;
`overlap_count` is the true, uncapped total. This is a nomination, not a verdict -- a human
reads the evidence and decides.

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

CI runs `ruff check .` and the full `pytest` suite on Python 3.11 and 3.12 for every
push and pull request.

Fixtures under `tests/fixtures/` are synthetic — no real transcript is ever committed.
