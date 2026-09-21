"""Compare hygiene metrics across two time periods (issue #39).

Runs the hygiene audit over two non-overlapping windows (baseline and
current) and emits a comparison report with period coverage, per-category
counts, and delta arithmetic (current minus baseline).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import duckdb

from ashiato.hygiene import CATEGORY_ORDER
from ashiato.hygiene import audit as hygiene_audit


def _cps(tool_calls: int, sessions: int) -> float | None:
    """Calls per session, rounded to two decimals; None when zero sessions."""
    if sessions == 0:
        return None
    return round(tool_calls / sessions, 2)


def _pct_change(current: int, baseline: int) -> float | None:
    """Percent change from baseline to current, rounded to two decimals.

    Returns None when baseline is zero (no meaningful denominator).
    """
    if baseline == 0:
        return None
    return round(((current - baseline) / baseline) * 100, 2)


def compare_periods(
    connection: duckdb.DuckDBPyConnection,
    *,
    baseline_since: datetime,
    baseline_until: datetime,
    current_since: datetime,
    current_until: datetime,
) -> dict[str, Any]:
    """Run hygiene audit over two windows and return a flat comparison report.

    The returned dict has ``periods`` (the baseline and current windows, each
    with coverage totals, its per-source ``sources`` list, and the
    ``excluded_no_timestamp`` disclosure for the window), ``categories`` (one
    object per hygiene category in :data:`CATEGORY_ORDER`, each carrying flat
    keys for baseline counts, current counts, change deltas, and computed
    ratios), and ``source_asymmetry`` (one object per source that has rows in
    exactly one of the two periods, so a delta that is really a newly
    ingested source cannot be read as a behaviour change).
    """
    baseline = hygiene_audit(connection, since=baseline_since, until=baseline_until)
    current = hygiene_audit(connection, since=current_since, until=current_until)

    baseline_map = {cat["name"]: cat for cat in baseline["categories"]}
    current_map = {cat["name"]: cat for cat in current["categories"]}

    b_cov = baseline["coverage"]
    c_cov = current["coverage"]

    b_sources = {item["source"]: item["tool_calls"] for item in b_cov["sources"]}
    c_sources = {item["source"]: item["tool_calls"] for item in c_cov["sources"]}
    source_asymmetry: list[dict[str, Any]] = []
    for source in sorted(
        set(b_sources) | set(c_sources),
        key=lambda item: (item is None, item or ""),
    ):
        b_tc = b_sources.get(source, 0)
        c_tc = c_sources.get(source, 0)
        if (b_tc > 0) != (c_tc > 0):
            source_asymmetry.append({
                "source": source,
                "baseline_tool_calls": b_tc,
                "current_tool_calls": c_tc,
            })

    categories: list[dict[str, Any]] = []
    for name in CATEGORY_ORDER:
        b = baseline_map[name]
        c = current_map[name]
        b_tc = b["tool_calls"]
        c_tc = c["tool_calls"]
        b_sess = b["sessions"]
        c_sess = c["sessions"]

        categories.append({
            "name": name,
            "baseline_tool_calls": b_tc,
            "current_tool_calls": c_tc,
            "baseline_sessions": b_sess,
            "current_sessions": c_sess,
            "baseline_calls_per_session": _cps(b_tc, b_sess),
            "current_calls_per_session": _cps(c_tc, c_sess),
            "tool_calls_change": c_tc - b_tc,
            "tool_calls_percent_change": _pct_change(c_tc, b_tc),
            "sessions_change": c_sess - b_sess,
        })

    return {
        "periods": {
            "baseline": {
                "since": baseline_since.isoformat() + "Z",
                "until": baseline_until.isoformat() + "Z",
                "tool_calls": b_cov["tool_calls"],
                "sessions": b_cov["sessions"],
                "sources": b_cov["sources"],
                "excluded_no_timestamp": b_cov["excluded_no_timestamp"],
            },
            "current": {
                "since": current_since.isoformat() + "Z",
                "until": current_until.isoformat() + "Z",
                "tool_calls": c_cov["tool_calls"],
                "sessions": c_cov["sessions"],
                "sources": c_cov["sources"],
                "excluded_no_timestamp": c_cov["excluded_no_timestamp"],
            },
        },
        "categories": categories,
        "source_asymmetry": source_asymmetry,
    }
