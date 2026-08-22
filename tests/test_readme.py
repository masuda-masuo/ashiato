"""README must document every ``ashiato`` subcommand.

The set of subcommands is derived from the parser built by
``ashiato.cli._build_arg_parser`` -- never a hand-maintained list, which
would go stale one level down from the README it is meant to guard.

The check runs one way only -- parser -> README. Adding a parser
subcommand with no README entry turns the suite red (criterion 3); removing
a subcommand leaves it green, because the removed name is no longer in the
parser's choice set.

A command counts as documented only when it appears as an actual command
reference in the README -- either ``ashiato <name>`` in the usage synopsis
or a backtick-wrapped `` `<name>` `` in the explanatory bullets -- not merely
as a word somewhere in unrelated prose. This is still a name-level check: it
does not grade prose quality, it only requires that the name be present as a
command. Anchoring to the command-reference style the README already uses is
what stops an incidental mention (e.g. "a grep whose output matched them") from
silently satisfying the guard, which is precisely the gap that opened twice in
a row.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ashiato.cli import _build_arg_parser

#: Repo-root README, resolved from this file so the test is location-stable.
README = Path(__file__).resolve().parent.parent / "README.md"


def _subcommand_names() -> set[str]:
    """Every subcommand the CLI actually exposes, with no hand-written list."""
    parser = _build_arg_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    return set(subparsers.choices)


def _is_documented(name: str, text: str) -> bool:
    """True when *name* appears as a command reference, not incidental prose."""
    # ``ashiato <name>`` in the usage synopsis (e.g. ``ashiato salvage``) or a
    # backtick-wrapped `` `<name>` `` in the explanatory bullets (e.g.
    # `` `denials` ``). Either form counts; an incidental mention in unrelated
    # prose matches neither and so does not satisfy the guard.
    return bool(
        re.search(rf"\bashiato\s+{re.escape(name)}\b", text)
        or re.search(rf"`{re.escape(name)}`", text)
    )


def test_every_subcommand_is_documented_in_readme() -> None:
    text = README.read_text(encoding="utf-8")
    missing = sorted(name for name in _subcommand_names() if not _is_documented(name, text))
    assert not missing, (
        f"subcommand(s) missing from README.md: {missing}. "
        "Add a README entry (an 'ashiato <name>' synopsis line or a "
        "'<name>' bullet) for each, or remove them from _build_arg_parser."
    )
