"""Observational execution-quality measurements.

This module deliberately has no connection to Council prompts, scores, order
sizing, or broker placement. It summarizes completed position episodes and
sell events so the supervisor can detect one-sided or missing learning data.
"""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from .config import Config
from .journal import DecisionJournal
from .portfolio import PORTFOLIO_FILE, Portfolio

EPSILON = 1e-9


def _sell_reason_category(reason: Any) -> str:
    """Normalize free-text exit reasons into stable learning buckets."""
    text = str(reason or "").lower()
    categories = (
        ("earnings_risk", ("earning", "gap risk")),
        ("hard_stop", ("hard stop",)),
        ("trailing_stop", ("trailing stop", "trail stop")),
        ("dead_money", ("dead money", "time stop")),
        ("thesis_break", ("thesis break", "invalidation")),
        ("profit_or_scale_out", ("profit", "scale-out", "scale out")),
        ("regime_exit", ("regime",)),
        ("council_exit", ("sell council", "council")),
    )
    for category, needles in categories:
        if any(needle in text for needle in needles):
            return category
    return "other"


def build_sell_learning_context(
    journal: DecisionJournal,
    *,
    minimum_completed: int | None = None,
    minimum_bucket: int = 5,
) -> dict[str, Any]:
    """Build a conservative, sample-gated Sell Council calibration digest.

    Only completed 60-day post-sell reviews are eligible. The digest is
    advisory and can never override current evidence or deterministic gates.
    """
    required = max(
        1,
        int(
            minimum_completed
            if minimum_completed is not None
            else Config.SELL_LEARNING_MIN_COMPLETED
        ),
    )
    rows = [
        row
        for row in journal.get_all_post_sell_reviews()
        if str(row.get("status") or "").lower() == "completed"
        and str(row.get("grade") or "").upper()
    ]
    completed = len(rows)
    if completed < required:
        return {
            "status": "insufficient_outcomes",
            "ready": False,
            "completed": completed,
            "minimum_completed": required,
            "context": (
                "SELL-OUTCOME CALIBRATION: insufficient mature evidence "
                f"({completed}/{required} completed 60-day reviews). Do not adjust "
                "this decision from historical sell outcomes."
            ),
        }

    grade_counts = Counter(str(row.get("grade") or "").upper() for row in rows)
    adverse_grades = {"EARLY", "INCORRECT", "WHIPSAW_STOP"}
    adverse = sum(grade_counts.get(grade, 0) for grade in adverse_grades)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(_sell_reason_category(row.get("sell_reason")), []).append(row)

    lines = [
        f"SELL-OUTCOME CALIBRATION: {completed} completed 60-day reviews; "
        f"premature/incorrect/whipsaw={adverse}/{completed} ({adverse / completed:.0%})."
    ]
    bucket_rows: list[dict[str, Any]] = []
    for category, members in sorted(buckets.items()):
        if len(members) < max(1, int(minimum_bucket)):
            continue
        counts = Counter(str(row.get("grade") or "").upper() for row in members)
        bucket_adverse = sum(counts.get(grade, 0) for grade in adverse_grades)
        bucket_rows.append(
            {
                "category": category,
                "completed": len(members),
                "adverse": bucket_adverse,
                "adverse_rate_pct": round(bucket_adverse / len(members) * 100.0, 2),
                "grade_counts": dict(sorted(counts.items())),
            }
        )
    bucket_rows.sort(
        key=lambda row: (-float(row["adverse_rate_pct"]), -int(row["completed"]), row["category"])
    )
    for row in bucket_rows[:4]:
        lines.append(
            f"- {row['category']}: n={row['completed']}, adverse outcomes "
            f"{row['adverse']}/{row['completed']} ({row['adverse_rate_pct']:.0f}%)."
        )
    lines.append(
        "Use this only as weak calibration. Current filings, live market data, thesis "
        "conditions, and deterministic safety rules take precedence."
    )
    return {
        "status": "ready",
        "ready": True,
        "completed": completed,
        "minimum_completed": required,
        "grade_counts": dict(sorted(grade_counts.items())),
        "reason_buckets": bucket_rows,
        "context": "\n".join(lines),
    }


def _number(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _rounded(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


def build_execution_learning_summary(
    journal: DecisionJournal,
    *,
    portfolio_path: str | Path = PORTFOLIO_FILE,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Build episode-based, sell-side, and sizing measurements.

    Multiple trims from one thesis remain one position episode. This avoids
    inflating win rates by treating each small sell fill as an independent bet.
    """
    episodes = journal.get_position_trade_episodes(limit=5000)
    closed = [
        row
        for row in episodes
        if str(row.get("status") or "").lower() == "closed"
        and row.get("realized_pnl") is not None
    ]
    pnl_values = [_number(row.get("realized_pnl")) for row in closed]
    returns = [
        _number(row.get("return_pct"))
        for row in closed
        if row.get("return_pct") is not None
    ]
    wins = [value for value in pnl_values if value > EPSILON]
    losses = [value for value in pnl_values if value < -EPSILON]
    flats = len(pnl_values) - len(wins) - len(losses)

    events = journal.get_sell_trade_events(limit=5000)
    trims = [row for row in events if str(row.get("event_type") or "").upper() == "TRIM"]
    exits = [row for row in events if str(row.get("event_type") or "").upper() == "EXIT"]
    inferred = [
        row for row in events
        if "inferred" in str(row.get("data_quality") or "").lower()
    ]
    quantity_conflicts = [
        row for row in events
        if "quantity_conflict" in str(row.get("data_quality") or "").lower()
    ]
    uncertain = {
        str(row.get("event_id") or "")
        for row in [*inferred, *quantity_conflicts]
    }
    missing_pnl = [row for row in events if row.get("realized_pnl") is None]
    non_exact_event_ids = uncertain | {
        str(row.get("event_id") or "") for row in missing_pnl
    }

    tracking = journal.get_all_post_sell_reviews()
    completed_tracking = [
        row
        for row in tracking
        if str(row.get("status") or "").lower() == "completed"
    ]
    grade_counts = Counter(
        str(row.get("grade") or "UNSET").upper()
        for row in tracking
        if str(row.get("status") or "").lower() == "completed"
    )
    status_counts = Counter(str(row.get("status") or "unknown").lower() for row in tracking)
    today = as_of or datetime.now(timezone.utc).date()

    def _sell_date(row: dict[str, Any]) -> date | None:
        try:
            return date.fromisoformat(str(row.get("sell_date") or "")[:10])
        except ValueError:
            return None

    checkpoint_coverage: dict[str, dict[str, int]] = {}
    for days, field in ((5, "price_5d"), (20, "price_20d"), (60, "price_60d")):
        eligible = [
            row
            for row in tracking
            if _sell_date(row) is not None
            and _sell_date(row) + timedelta(days=days) <= today
        ]
        checkpoint_coverage[f"{days}d"] = {
            "eligible": len(eligible),
            "recorded": sum(1 for row in eligible if row.get(field) is not None),
        }
    pending_60d_dates = sorted(
        sold + timedelta(days=60)
        for row in tracking
        if (sold := _sell_date(row)) is not None
        and row.get("price_60d") is None
        and sold + timedelta(days=60) > today
    )
    overdue_60d = [
        row
        for row in tracking
        if (sold := _sell_date(row)) is not None
        and row.get("price_60d") is None
        and sold + timedelta(days=65) < today
    ]

    portfolio = Portfolio.load(Path(portfolio_path))
    position_values: list[float] = []
    position_types: Counter[str] = Counter()
    for position in portfolio.positions:
        value = _number(position.market_value)
        if value <= 0:
            value = _number(position.shares) * _number(position.current_price or position.avg_cost)
        position_values.append(max(0.0, value))
        position_types[str(position.position_type or "UNCLASSIFIED").upper()] += 1

    event_count = len(events)
    expectancy_is_provisional = bool(uncertain or missing_pnl)
    sell_notional = sum(_number(row.get("proceeds")) for row in events)
    trim_notional = sum(_number(row.get("proceeds")) for row in trims)
    completed_count = len(closed)
    summary = {
        "scope": "observational_only",
        "strategy_effects_enabled": False,
        "episodes": {
            "total": len(episodes),
            "open": sum(1 for row in episodes if str(row.get("status") or "").lower() == "open"),
            "closed": completed_count,
            "wins": len(wins),
            "losses": len(losses),
            "flats": flats,
            "win_rate_pct": _rounded(len(wins) / completed_count * 100.0, 2) if completed_count else None,
            "average_win_dollars": _rounded(mean(wins), 4) if wins else None,
            "average_loss_dollars": _rounded(mean(losses), 4) if losses else None,
            "expectancy_dollars": _rounded(mean(pnl_values), 4) if pnl_values else None,
            "average_return_pct": _rounded(mean(returns), 4) if returns else None,
            "realized_pnl_dollars": _rounded(sum(pnl_values), 4),
        },
        "sell_events": {
            "total": event_count,
            "trims": len(trims),
            "exits": len(exits),
            "trim_share_pct": _rounded(len(trims) / event_count * 100.0, 2) if event_count else None,
            "sell_notional_dollars": _rounded(sell_notional, 4),
            "trim_notional_dollars": _rounded(trim_notional, 4),
        },
        "post_sell": {
            "total": len(tracking),
            "completed": len(completed_tracking),
            "minimum_completed_for_council": Config.SELL_LEARNING_MIN_COMPLETED,
            "council_learning_ready": (
                len(completed_tracking) >= Config.SELL_LEARNING_MIN_COMPLETED
            ),
            "status_counts": dict(sorted(status_counts.items())),
            "grade_counts": dict(sorted(grade_counts.items())),
            "checkpoint_coverage": checkpoint_coverage,
            "overdue_60d_reviews": len(overdue_60d),
            "next_60d_due_date": (
                pending_60d_dates[0].isoformat() if pending_60d_dates else None
            ),
        },
        "current_sizing": {
            "positions": len(position_values),
            "mean_position_dollars": _rounded(mean(position_values), 4) if position_values else None,
            "median_position_dollars": _rounded(median(position_values), 4) if position_values else None,
            "largest_position_dollars": _rounded(max(position_values), 4) if position_values else None,
            "position_types": dict(sorted(position_types.items())),
        },
        "data_quality": {
            "status": "provisional" if expectancy_is_provisional else "exact",
            "expectancy_is_provisional": expectancy_is_provisional,
            "exact_cost_events": event_count - len(non_exact_event_ids),
            "inferred_cost_events": len(inferred),
            "inferred_cost_pct": _rounded(len(inferred) / event_count * 100.0, 2) if event_count else 0.0,
            "quantity_conflict_events": len(quantity_conflicts),
            "uncertain_events": len(uncertain),
            "missing_realized_pnl_events": len(missing_pnl),
        },
    }
    return summary
