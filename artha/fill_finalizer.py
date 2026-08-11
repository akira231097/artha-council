"""Idempotent broker sell-fill accounting and learning intake.

Robinhood order placement and investment policy live elsewhere. This module
only applies the consequences of a broker-confirmed sell: portfolio accounting,
thesis lifecycle, trim cooldown, post-sell tracking, and episode/event records.
Every broker-driven sell path calls this same operation so a new reconciliation
route cannot silently omit one of those effects.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import Config
from .journal import DecisionJournal
from .opportunity_cost import PostSellTracker
from .portfolio import PORTFOLIO_FILE, Portfolio
from .thesis_tracker import ThesisTracker

logger = logging.getLogger(__name__)

EPSILON = 0.000001
EXIT_ACTIONS = {"EXIT", "URGENT_EXIT", "SELL"}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _deadline_from_event(value: Any, days: int) -> str:
    """Return a stable deadline anchored to broker event time.

    Callback replay time is not economic event time. Anchoring cooldowns to the
    fill prevents an old callback from manufacturing a fresh seven-day block.
    """
    base = _parse_datetime(value) or datetime.now(timezone.utc)
    return (base + timedelta(days=int(days))).isoformat()


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(str(part or "") for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:28]
    return f"{prefix}_{digest}"


def _fill_event_key(
    *,
    broker_order_id: str,
    order_intent_id: str,
    ticker: str,
    side: str = "sell",
) -> str:
    identity = broker_order_id or order_intent_id
    return _stable_id("fill", "robinhood", identity, ticker.upper(), side.lower())


def _sell_council_context(row: dict[str, Any]) -> tuple[str, str]:
    evidence = _json_object(row.get("evidence_json"))
    nested = evidence.get("sell_council") if isinstance(evidence.get("sell_council"), dict) else {}
    action = str(nested.get("action") or evidence.get("action") or "").upper().strip()
    reason = str(
        evidence.get("reason")
        or nested.get("reason")
        or row.get("rationale")
        or "Robinhood sell order filled."
    ).strip()
    return action, reason


def _completion_markers(row: dict[str, Any]) -> list[str]:
    """Return deterministic one-shot markers authorized by the sell signal."""
    evidence = _json_object(row.get("evidence_json"))
    raw = evidence.get("completion_markers")
    if not isinstance(raw, list):
        return []
    return sorted({str(marker).strip() for marker in raw if str(marker).strip()})


def _portfolio_sell_summary(
    portfolio: Portfolio,
    *,
    ticker: str,
    broker_order_id: str,
) -> dict[str, float]:
    shares = 0.0
    proceeds = 0.0
    realized_pnl = 0.0
    cost_basis = 0.0
    for txn in portfolio.transactions or []:
        if str(txn.get("type") or "").upper() != "SELL":
            continue
        if str(txn.get("ticker") or "").upper() != ticker.upper():
            continue
        txn_order_id = str(txn.get("broker_order_id") or "")
        notes = str(txn.get("notes") or "")
        if broker_order_id and broker_order_id != txn_order_id and broker_order_id not in notes:
            continue
        txn_shares = abs(_number(txn.get("shares")) or 0.0)
        txn_price = _number(txn.get("price")) or 0.0
        txn_cost = _number(txn.get("cost_basis_per_share"))
        txn_pnl = _number(txn.get("realized_pnl"))
        shares += txn_shares
        proceeds += _number(txn.get("total")) or txn_shares * txn_price
        if txn_cost is not None:
            cost_basis += txn_shares * txn_cost
        if txn_pnl is not None:
            realized_pnl += txn_pnl
    return {
        "shares": shares,
        "proceeds": proceeds,
        "realized_pnl": realized_pnl,
        "cost_basis": cost_basis,
    }


def _matching_external_reconciliation_event(
    journal: DecisionJournal,
    *,
    ticker: str,
    thesis_id: str,
    quantity: float,
    filled_at: str,
) -> dict[str, Any] | None:
    """Find a synthetic close that a delayed broker fill can safely replace.

    Matching is deliberately strict: same ticker, compatible thesis, full-exit
    synthetic provenance, nearly identical quantity, plausible ordering, and
    no intervening broker-confirmed buy. Ambiguity fails closed by returning
    ``None`` instead of merging unrelated economic events.
    """
    broker_time = _parse_datetime(filled_at)
    if broker_time is None:
        return None
    tolerance = max(EPSILON, abs(quantity) * 0.001)
    for event in reversed(journal.get_sell_trade_events(ticker=ticker, limit=100)):
        if str(event.get("source") or "") != "external_snapshot_reconciliation":
            continue
        if str(event.get("event_type") or "").upper() != "EXIT":
            continue
        if not str(event.get("broker_order_id") or "").startswith("external-snapshot:"):
            continue
        event_thesis = str(event.get("thesis_id") or "")
        if thesis_id and event_thesis and thesis_id != event_thesis:
            continue
        if abs(float(_number(event.get("shares")) or 0.0) - quantity) > tolerance:
            continue
        synthetic_time = _parse_datetime(event.get("filled_at"))
        if synthetic_time is None or broker_time > synthetic_time + timedelta(minutes=5):
            continue
        if synthetic_time - broker_time > timedelta(days=7):
            continue
        intervening_buy = False
        for order in journal.get_execution_orders(ticker=ticker, limit=500):
            if str(order.get("side") or "").lower() != "buy":
                continue
            if str(order.get("status") or "").lower() != "filled":
                continue
            buy_time = _parse_datetime(order.get("filled_at"))
            if buy_time and broker_time < buy_time <= synthetic_time:
                intervening_buy = True
                break
        if not intervening_buy:
            return event
    return None


def _upgrade_external_portfolio_transaction(
    portfolio: Portfolio,
    *,
    ticker: str,
    synthetic_order_id: str,
    broker_order_id: str,
    quantity: float,
    price: float,
    filled_at: str,
) -> bool:
    """Replace estimated synthetic-close details with exact broker evidence."""
    tolerance = max(EPSILON, abs(quantity) * 0.001)
    for txn in portfolio.transactions or []:
        if str(txn.get("type") or "").upper() != "SELL":
            continue
        if str(txn.get("ticker") or "").upper() != ticker.upper():
            continue
        txn_order_id = str(txn.get("broker_order_id") or "")
        notes = str(txn.get("notes") or "")
        if synthetic_order_id != txn_order_id and synthetic_order_id not in notes:
            continue
        shares = float(_number(txn.get("shares")) or 0.0)
        if abs(shares - quantity) > tolerance:
            return False
        cost = _number(txn.get("cost_basis_per_share"))
        txn["broker_order_id"] = broker_order_id
        txn["price"] = price
        txn["total"] = round(shares * price, 2)
        if cost is not None:
            txn["realized_pnl"] = round((price - cost) * shares, 2)
        txn["timestamp"] = filled_at
        txn["notes"] = (
            f"{notes} | Late broker fill reconciled from {synthetic_order_id}"
        ).strip(" |")
        return True
    return False


def _incremental_fill_price(
    *,
    cumulative_quantity: float,
    cumulative_average_price: float,
    applied_quantity: float,
    applied_notional: float,
) -> float:
    """Return the price attributable to the newly observed fill quantity.

    Robinhood reports cumulative quantity and cumulative average price. Applying
    that average directly to each callback distorts realized P&L whenever a
    partially filled order completes at a different price.
    """
    delta = cumulative_quantity - applied_quantity
    if delta <= EPSILON:
        return cumulative_average_price
    delta_notional = (cumulative_quantity * cumulative_average_price) - applied_notional
    if delta_notional <= 0:
        return cumulative_average_price
    return delta_notional / delta


def _order_fill_values(row: dict[str, Any]) -> tuple[float, float, str, str]:
    response = _json_object(row.get("response_json"))
    order = response.get("order") if isinstance(response.get("order"), dict) else {}
    quantity = _number(
        order.get("cumulative_quantity")
        or response.get("artha_fill_quantity")
        or row.get("quantity")
    ) or 0.0
    price = _number(
        order.get("average_price")
        or response.get("artha_fill_average_price")
        or row.get("estimated_price")
    ) or 0.0
    broker_order_id = str(
        order.get("id")
        or response.get("artha_fill_broker_order_id")
        or row.get("broker_order_id")
        or ""
    )
    filled_at = str(
        order.get("last_transaction_at")
        or row.get("filled_at")
        or row.get("updated_at")
        or row.get("created_at")
        or _utcnow_iso()
    )
    return quantity, price, broker_order_id, filled_at


def _confirmed_broker_fill(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return immutable fill facts only when the broker payload proves them."""
    response = _json_object(row.get("response_json"))
    order = response.get("order") if isinstance(response.get("order"), dict) else {}
    expected_side = str(row.get("side") or "").lower().strip()
    side = str(order.get("side") or "").lower().strip()
    quantity = _number(order.get("cumulative_quantity")) or 0.0
    price = _number(order.get("average_price")) or 0.0
    broker_order_id = str(order.get("id") or "").strip()
    filled_at = str(order.get("last_transaction_at") or "").strip()
    if (
        str(order.get("state") or "").lower() != "filled"
        or side not in {"buy", "sell"}
        or side != expected_side
        or quantity <= 0
        or price <= 0
        or not broker_order_id
        or _parse_datetime(filled_at) is None
    ):
        return None
    fees = _number(order.get("fees"))
    if fees is None:
        executions = order.get("executions")
        fees = sum(
            _number(item.get("fees")) or 0.0
            for item in executions
            if isinstance(item, dict)
        ) if isinstance(executions, list) else 0.0
    return {
        "ticker": str(row.get("ticker") or order.get("symbol") or "").upper().strip(),
        "side": side,
        "quantity": quantity,
        "average_price": price,
        "fees": max(0.0, fees or 0.0),
        "broker_order_id": broker_order_id,
        "filled_at": filled_at,
    }


def _reconstruct_fifo_sell_accounting(
    journal: DecisionJournal,
) -> dict[str, dict[str, Any]]:
    """Rebuild sell cost from complete broker-confirmed FIFO fill history.

    A ticker becomes ineligible for reconstruction as soon as its fill history
    has a gap or a quantity mismatch. This deliberately favors an explicit
    uncertainty label over inventing cost basis from a thesis or quote.
    """
    timeline: list[tuple[datetime, int, dict[str, Any], dict[str, Any] | None]] = []
    for row in journal.get_execution_orders(limit=5000):
        if str(row.get("status") or "").lower() != "filled":
            continue
        confirmed = _confirmed_broker_fill(row)
        observed_at = (
            _parse_datetime(confirmed.get("filled_at"))
            if confirmed
            else _parse_datetime(row.get("filled_at") or row.get("created_at"))
        )
        if observed_at is None:
            continue
        timeline.append((observed_at, int(row.get("id") or 0), row, confirmed))
    timeline.sort(key=lambda item: (item[0], item[1]))

    lots: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    tainted: set[str] = set()
    seen_orders: set[str] = set()
    reconstructed: dict[str, dict[str, Any]] = {}
    for _, _, row, fill in timeline:
        ticker = str(row.get("ticker") or "").upper().strip()
        if not ticker:
            continue
        if fill is None:
            tainted.add(ticker)
            continue
        broker_order_id = str(fill["broker_order_id"])
        if broker_order_id in seen_orders:
            tainted.add(ticker)
            continue
        seen_orders.add(broker_order_id)
        if ticker in tainted:
            continue

        quantity = float(fill["quantity"])
        price = float(fill["average_price"])
        fees = float(fill["fees"])
        if fill["side"] == "buy":
            lots[ticker].append(
                {
                    "shares": quantity,
                    "unit_cost": ((quantity * price) + fees) / quantity,
                    "broker_order_id": broker_order_id,
                }
            )
            continue

        available = sum(float(lot["shares"]) for lot in lots[ticker])
        tolerance = max(EPSILON, quantity * 0.001)
        if available + tolerance < quantity or available <= EPSILON:
            tainted.add(ticker)
            continue

        shares_before = available
        remaining = quantity
        cost_basis = 0.0
        buy_order_ids: list[str] = []
        while remaining > EPSILON and lots[ticker]:
            lot = lots[ticker][0]
            consumed = min(remaining, float(lot["shares"]))
            cost_basis += consumed * float(lot["unit_cost"])
            remaining -= consumed
            lot["shares"] = float(lot["shares"]) - consumed
            buy_order_ids.append(str(lot["broker_order_id"]))
            if float(lot["shares"]) <= EPSILON:
                lots[ticker].popleft()
        if remaining > tolerance:
            tainted.add(ticker)
            continue
        shares_after = max(0.0, shares_before - quantity)
        if shares_after <= tolerance:
            shares_after = 0.0
            lots[ticker].clear()
        reconstructed[broker_order_id] = {
            "cost_basis_per_share": cost_basis / quantity,
            "realized_pnl": (quantity * price) - fees - cost_basis,
            "position_shares_before": shares_before,
            "position_shares_after": shares_after,
            "buy_order_ids": sorted(set(buy_order_ids)),
            "data_quality": "broker_fill_fifo_reconstruction",
        }
    return reconstructed


def _episode_buy_orders(
    journal: DecisionJournal,
    *,
    ticker: str,
    thesis: Any,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Link legacy untagged broker buys to a thesis using its entry timestamp."""
    entry_at = _parse_datetime(getattr(thesis, "entry_date", None))
    if entry_at is None:
        return []
    end_times = [
        parsed
        for parsed in (_parse_datetime(event.get("filled_at")) for event in events)
        if parsed
    ]
    end_at = max(end_times) if str(getattr(thesis, "status", "")) == "archived" and end_times else None
    candidates: list[tuple[datetime, dict[str, Any]]] = []
    for row in journal.get_execution_orders(ticker=ticker, status="filled", limit=5000):
        if str(row.get("side") or "").lower() != "buy":
            continue
        row_thesis_id = str(row.get("thesis_id") or "")
        if row_thesis_id and row_thesis_id != str(getattr(thesis, "thesis_id", "")):
            continue
        fill = _confirmed_broker_fill(row)
        filled_at = _parse_datetime(fill.get("filled_at")) if fill else None
        if filled_at is None or (end_at and filled_at > end_at):
            continue
        candidates.append((filled_at, row))
    candidates.sort(key=lambda item: (item[0], int(item[1].get("id") or 0)))
    anchors = [
        item for item in candidates
        if abs((item[0] - entry_at).total_seconds()) <= 30 * 60
    ]
    if not anchors:
        return []
    anchor_at = min(anchors, key=lambda item: abs((item[0] - entry_at).total_seconds()))[0]
    return [row for filled_at, row in candidates if filled_at >= anchor_at]


def rebuild_position_episode(
    journal: DecisionJournal,
    *,
    thesis_id: str,
    ticker: str,
    position_type: str = "",
    source: str = "broker_fill_accounting",
) -> dict[str, Any] | None:
    """Rebuild a thesis-level episode from immutable execution/event rows."""
    if not thesis_id or not ticker:
        return None
    orders = [
        row
        for row in journal.get_execution_orders(limit=5000)
        if str(row.get("thesis_id") or "") == thesis_id
        and str(row.get("status") or "").lower() == "filled"
    ]
    events = journal.get_sell_trade_events(thesis_id=thesis_id, limit=1000)
    thesis = ThesisTracker(journal).get(thesis_id)
    buys = [row for row in orders if str(row.get("side") or "").lower() == "buy"]
    if thesis:
        linked_buys = _episode_buy_orders(
            journal,
            ticker=ticker,
            thesis=thesis,
            events=events,
        )
        buys_by_intent = {
            str(row.get("order_intent_id") or row.get("id") or ""): row
            for row in [*buys, *linked_buys]
        }
        buys = list(buys_by_intent.values())

    buy_shares = 0.0
    buy_notional = 0.0
    buy_times: list[datetime] = []
    for row in buys:
        quantity, price, _, filled_at = _order_fill_values(row)
        if quantity <= 0 or price <= 0:
            continue
        buy_shares += quantity
        buy_notional += quantity * price
        parsed = _parse_datetime(filled_at)
        if parsed:
            buy_times.append(parsed)

    if buy_shares <= EPSILON and thesis and thesis.entry_price:
        buy_shares = max(
            sum(_number(event.get("shares")) or 0.0 for event in events),
            EPSILON,
        )
        buy_notional = buy_shares * float(thesis.entry_price)
        parsed = _parse_datetime(thesis.entry_date)
        if parsed:
            buy_times.append(parsed)

    sell_shares = sum(_number(event.get("shares")) or 0.0 for event in events)
    sell_notional = sum(_number(event.get("proceeds")) or 0.0 for event in events)
    realized_values = [
        _number(event.get("realized_pnl"))
        for event in events
        if _number(event.get("realized_pnl")) is not None
    ]
    realized_pnl = sum(value or 0.0 for value in realized_values) if realized_values else None
    cost_basis_sold = sum(
        (_number(event.get("cost_basis_per_share")) or 0.0)
        * (_number(event.get("shares")) or 0.0)
        for event in events
    )
    status = "closed" if thesis and thesis.status == "archived" else "open"
    if any(
        str(event.get("event_type") or "").upper() == "EXIT"
        and (_number(event.get("position_shares_after")) or 0.0) <= EPSILON
        for event in events
        if event.get("position_shares_after") is not None
    ):
        status = "closed"
    close_times = [
        parsed
        for parsed in (_parse_datetime(event.get("filled_at")) for event in events)
        if parsed
    ]
    entry_price = buy_notional / buy_shares if buy_shares > EPSILON else None
    exit_price = sell_notional / sell_shares if status == "closed" and sell_shares > EPSILON else None
    return_pct = (
        realized_pnl / cost_basis_sold * 100.0
        if realized_pnl is not None and cost_basis_sold > EPSILON
        else None
    )
    episode = {
        "episode_id": _stable_id("episode", thesis_id),
        "thesis_id": thesis_id,
        "ticker": ticker.upper(),
        "position_type": position_type or str(getattr(thesis, "position_type", "") or ""),
        "status": status,
        "opened_at": min(buy_times).isoformat() if buy_times else getattr(thesis, "entry_date", None),
        "closed_at": max(close_times).isoformat() if status == "closed" and close_times else getattr(thesis, "exit_date", None),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "shares_bought": buy_shares,
        "shares_sold": sell_shares,
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
        "realized_pnl": realized_pnl,
        "return_pct": return_pct,
        "buy_fill_count": len(buys),
        "sell_fill_count": len(events),
        "trim_count": sum(
            1 for event in events if str(event.get("event_type") or "").upper() == "TRIM"
        ),
        "source": source,
    }
    journal.save_position_trade_episode(episode)
    return episode


def finalize_broker_sell_fill(
    *,
    journal: DecisionJournal,
    execution_row: dict[str, Any] | None,
    ticker: str,
    cumulative_quantity: float,
    average_price: float,
    broker_order_id: str,
    order_intent_id: str = "",
    state: str = "filled",
    source: str = "broker_fill_callback",
    filled_at: str | None = None,
    portfolio_path: str | Path = PORTFOLIO_FILE,
    broker_snapshot_position_quantity: float | None = None,
    position_already_reflected: bool = False,
    data_quality: str = "broker_fill",
    portfolio_notes: str | None = None,
    credit_cash_available: bool = False,
) -> dict[str, Any]:
    """Apply every deterministic consequence of a confirmed sell fill.

    Broker snapshot reconciliation remains the normal source of cash. The
    legacy manual-bookkeeping CLI opts into ``credit_cash_available`` because
    no broker snapshot accompanies that local command.
    """
    row = dict(execution_row or {})
    ticker = str(ticker or row.get("ticker") or "").upper().strip()
    order_intent_id = str(order_intent_id or row.get("order_intent_id") or "")
    broker_order_id = str(broker_order_id or row.get("broker_order_id") or "")
    quantity = float(cumulative_quantity or 0.0)
    price = float(average_price or 0.0)
    if not ticker or quantity <= 0 or price <= 0:
        raise ValueError("Sell fill requires ticker, positive cumulative quantity, and price")
    if not broker_order_id:
        broker_order_id = _stable_id("external", ticker, order_intent_id, filled_at, quantity, price)
    computed_event_id = _fill_event_key(
        broker_order_id=broker_order_id,
        order_intent_id=order_intent_id,
        ticker=ticker,
    )
    timestamp = str(filled_at or row.get("filled_at") or _utcnow_iso())
    action, reason = _sell_council_context(row)
    completion_markers = _completion_markers(row)
    transaction_notes = str(
        portfolio_notes or f"Robinhood order fill {broker_order_id}"
    )

    tracker = ThesisTracker(journal)
    thesis_id = str(row.get("thesis_id") or "")
    tracked = tracker.get(thesis_id) if thesis_id else tracker.get_sell_monitored(ticker)
    if tracked and not thesis_id:
        thesis_id = tracked.thesis_id
    event_id = computed_event_id
    effect_before = (
        journal.get_broker_fill_effect(computed_event_id)
        or journal.get_broker_fill_effect_by_order_id(broker_order_id)
        or {}
    )
    if effect_before:
        event_id = str(effect_before.get("event_key") or computed_event_id)

    external_event: dict[str, Any] | None = None
    if not effect_before and not broker_order_id.startswith("external-snapshot:"):
        external_event = _matching_external_reconciliation_event(
            journal,
            ticker=ticker,
            thesis_id=thesis_id,
            quantity=quantity,
            filled_at=timestamp,
        )
        if external_event:
            event_id = str(external_event.get("event_id") or computed_event_id)
            effect_before = journal.get_broker_fill_effect(event_id) or {}

    event_before = journal.get_sell_trade_event(event_id) or external_event or {}
    effect_details = _json_object(effect_before.get("details_json"))
    portfolio = Portfolio.load(Path(portfolio_path))
    external_transaction_upgraded = False
    if external_event:
        external_transaction_upgraded = _upgrade_external_portfolio_transaction(
            portfolio,
            ticker=ticker,
            synthetic_order_id=str(external_event.get("broker_order_id") or ""),
            broker_order_id=broker_order_id,
            quantity=quantity,
            price=price,
            filled_at=timestamp,
        )
        if external_transaction_upgraded:
            portfolio.last_updated = _utcnow_iso()
            portfolio.save(Path(portfolio_path))

    position = portfolio.get_position(ticker)
    position_type = str(
        getattr(position, "position_type", "")
        or getattr(tracked, "position_type", "")
        or ""
    )
    avg_cost = _number(getattr(position, "avg_cost", None))
    if avg_cost is None and tracked:
        avg_cost = _number(getattr(tracked, "entry_price", None))
    avg_cost = (
        avg_cost
        or _number(event_before.get("cost_basis_per_share"))
        or _number(effect_before.get("cost_basis_per_share"))
        or price
    )

    recorded_before = _portfolio_sell_summary(
        portfolio,
        ticker=ticker,
        broker_order_id=broker_order_id,
    )
    applied_before = max(
        _number(effect_before.get("applied_quantity")) or 0.0,
        recorded_before["shares"],
    )
    delta = max(0.0, quantity - applied_before)
    applied_notional = max(
        recorded_before["proceeds"],
        (_number(effect_before.get("applied_quantity")) or 0.0)
        * (_number(effect_before.get("average_price")) or 0.0),
    )
    incremental_price = _incremental_fill_price(
        cumulative_quantity=quantity,
        cumulative_average_price=price,
        applied_quantity=applied_before,
        applied_notional=applied_notional,
    )
    shares_before = _number(getattr(position, "shares", None)) or 0.0
    portfolio_changed = external_transaction_upgraded

    if position_already_reflected:
        if delta > EPSILON:
            portfolio.record_reflected_sell_fill(
                ticker,
                quantity,
                incremental_price,
                avg_cost,
                broker_order_id=broker_order_id,
                notes=transaction_notes,
                timestamp=timestamp,
            )
            portfolio_changed = True
        remaining = _number(broker_snapshot_position_quantity)
        position = portfolio.get_position(ticker)
        if position and remaining is not None and remaining >= 0:
            if abs(float(position.shares or 0.0) - remaining) > EPSILON:
                position.shares = remaining
                position.current_price = price
                position.market_value = round(remaining * price, 4)
                portfolio_changed = True
            if remaining <= EPSILON:
                portfolio.positions = [p for p in portfolio.positions if p.ticker.upper() != ticker]
                portfolio_changed = True
    elif delta > EPSILON:
        position = portfolio.get_position(ticker)
        if position:
            portfolio.sell_position(
                ticker,
                delta,
                incremental_price,
                notes=transaction_notes,
                broker_order_id=broker_order_id,
                timestamp=timestamp,
            )
        else:
            portfolio.record_reflected_sell_fill(
                ticker,
                quantity,
                incremental_price,
                avg_cost,
                broker_order_id=broker_order_id,
                notes=f"{transaction_notes} (position already absent)",
                timestamp=timestamp,
            )
        portfolio_changed = True

    if credit_cash_available and delta > EPSILON:
        portfolio.cash_available = float(portfolio.cash_available or 0.0) + (
            delta * incremental_price
        )
        portfolio_changed = True

    position_after = portfolio.get_position(ticker)
    shares_after = _number(getattr(position_after, "shares", None)) or 0.0
    prior_event_type = str(
        event_before.get("event_type") or effect_details.get("event_type") or ""
    ).upper()
    event_type = (
        prior_event_type
        if delta <= EPSILON and prior_event_type in {"TRIM", "EXIT"}
        else ("EXIT" if shares_after <= EPSILON else "TRIM")
    )
    full_exit = event_type == "EXIT"
    thesis_effect = "not_applicable"
    tracking_effect = "not_applicable"

    if tracked:
        if full_exit:
            if tracked.status != "archived":
                tracker.archive_thesis(
                    tracked.thesis_id,
                    exit_price=price,
                    exit_reason=reason,
                )
                tracker.update_thesis_fields(tracked.thesis_id, exit_date=timestamp)
            thesis_effect = "archived"
        elif action == "TRIM" and tracked.status != "archived" and position_after:
            cooldown_until = _deadline_from_event(
                timestamp,
                Config.SELL_COOLDOWN_AFTER_TRIM,
            )
            current_cooldown = _parse_datetime(tracked.sell_cooldown_until)
            proposed_cooldown = _parse_datetime(cooldown_until)
            if current_cooldown is None or (
                proposed_cooldown is not None and proposed_cooldown > current_cooldown
            ):
                tracker.update_thesis_fields(
                    tracked.thesis_id,
                    status="active",
                    sell_cooldown_until=cooldown_until,
                    next_review_date=cooldown_until,
                    exit_date=None,
                    exit_price=None,
                    exit_reason=None,
                )
                refreshed = tracker.get(tracked.thesis_id)
                if position_after and refreshed:
                    position_after.sell_cooldown_until = refreshed.sell_cooldown_until
                    position_after.next_sell_review = (
                        str(refreshed.next_review_date or "")[:10] or None
                    )
                    portfolio_changed = True
                thesis_effect = "trim_cooldown_set"
            else:
                thesis_effect = "trim_cooldown_preserved"
        else:
            thesis_effect = "position_remains_no_trim_cooldown"

        # A milestone means "a broker-confirmed trim completed", not merely
        # "a signal was generated". Recording it here also makes callback,
        # repair, and reconciliation paths share the same idempotent behavior.
        if event_type == "TRIM" and completion_markers:
            for marker in completion_markers:
                tracker.record_scale_out(tracked.thesis_id, marker)

    if portfolio_changed:
        portfolio.last_updated = _utcnow_iso()
        portfolio.save(Path(portfolio_path))

    portfolio = Portfolio.load(Path(portfolio_path))
    summary = _portfolio_sell_summary(
        portfolio,
        ticker=ticker,
        broker_order_id=broker_order_id,
    )
    event_shares = max(summary["shares"], quantity)
    cost_basis_per_share = (
        summary["cost_basis"] / summary["shares"]
        if summary["shares"] > EPSILON and summary["cost_basis"] > 0
        else avg_cost
    )
    realized_pnl = (
        summary["realized_pnl"]
        if summary["shares"] > EPSILON
        else (price - cost_basis_per_share) * event_shares
    )
    event_position_before = max(shares_before, shares_after + delta)
    event_position_after = shares_after
    if delta <= EPSILON and event_before:
        event_position_before = (
            _number(event_before.get("position_shares_before"))
            if event_before.get("position_shares_before") is not None
            else event_position_before
        )
        event_position_after = (
            _number(event_before.get("position_shares_after"))
            if event_before.get("position_shares_after") is not None
            else event_position_after
        )
    journal.save_sell_trade_event(
        {
            "event_id": event_id,
            "broker_order_id": broker_order_id,
            "order_intent_id": order_intent_id,
            "ticker": ticker,
            "thesis_id": thesis_id or None,
            "event_type": event_type,
            "shares": event_shares,
            "average_price": price,
            "cost_basis_per_share": cost_basis_per_share,
            "proceeds": event_shares * price,
            "realized_pnl": realized_pnl,
            "position_shares_before": event_position_before,
            "position_shares_after": event_position_after,
            "sell_reason": reason,
            "position_type": position_type,
            "source": source,
            "data_quality": data_quality,
            "filled_at": timestamp,
        }
    )

    if full_exit and tracked:
        tracking_id = _stable_id("post_sell", event_id, tracked.thesis_id)
        PostSellTracker(journal).record_sell(
            ticker=ticker,
            thesis_id=tracked.thesis_id,
            sell_price=price,
            sell_reason=reason,
            shares=event_shares,
            position_type=position_type,
            tracking_id=tracking_id,
            sell_date=timestamp,
            broker_order_id=broker_order_id,
            order_intent_id=order_intent_id,
            sell_event_id=event_id,
            cost_basis=cost_basis_per_share * event_shares,
            realized_pnl=realized_pnl,
            data_quality=data_quality,
        )
        tracking_effect = "post_sell_tracking_started"
    elif full_exit:
        tracking_effect = "missing_thesis"
    else:
        tracking_effect = "trim_event_recorded"

    episode = None
    if thesis_id:
        episode = rebuild_position_episode(
            journal,
            thesis_id=thesis_id,
            ticker=ticker,
            position_type=position_type,
            source=source,
        )

    journal.save_broker_fill_effect(
        {
            "event_key": event_id,
            "broker_order_id": broker_order_id,
            "order_intent_id": order_intent_id,
            "ticker": ticker,
            "side": "sell",
            "state": state,
            "cumulative_quantity": quantity,
            "average_price": price,
            "applied_quantity": event_shares,
            "source": source,
            "portfolio_effect_status": "applied" if portfolio_changed else "already_applied",
            "thesis_effect_status": thesis_effect,
            "tracking_effect_status": tracking_effect,
            "details": {
                "action": action,
                "event_type": event_type,
                "shares_before": shares_before,
                "shares_after": event_position_after,
                "realized_pnl": realized_pnl,
                "data_quality": data_quality,
                "reconciled_external_event": bool(external_event),
                "completion_markers": completion_markers if event_type == "TRIM" else [],
            },
        }
    )
    return {
        "status": "PASS",
        "event_id": event_id,
        "ticker": ticker,
        "side": "sell",
        "quantity": quantity,
        "applied_delta": delta,
        "avg_price": price,
        "broker_order_id": broker_order_id,
        "already_recorded": delta <= EPSILON,
        "portfolio_repaired": portfolio_changed and applied_before > EPSILON,
        "portfolio_changed": portfolio_changed,
        "external_reconciliation_upgraded": bool(external_event),
        "event_type": event_type,
        "full_exit": full_exit,
        "realized_pnl": realized_pnl,
        "thesis_effect": thesis_effect,
        "tracking_effect": tracking_effect,
        "completion_markers": completion_markers if event_type == "TRIM" else [],
        "episode_id": episode.get("episode_id") if episode else None,
        "source": source,
    }


def backfill_sell_fill_accounting(
    journal: DecisionJournal,
    *,
    portfolio_path: str | Path = PORTFOLIO_FILE,
) -> dict[str, Any]:
    """Backfill event/episode learning rows without replaying historical trades."""
    portfolio = Portfolio.load(Path(portfolio_path))
    sell_orders = [
        row
        for row in journal.get_execution_orders(limit=5000)
        if str(row.get("side") or "").lower() == "sell"
        and str(row.get("status") or "").lower() == "filled"
    ]
    def _sort_key(order_row: dict[str, Any]) -> tuple[datetime, int]:
        _, _, _, observed_at = _order_fill_values(order_row)
        return (_parse_datetime(observed_at) or datetime.min.replace(tzinfo=timezone.utc), int(order_row.get("id") or 0))

    sell_orders.sort(key=_sort_key)
    fifo_accounting = _reconstruct_fifo_sell_accounting(journal)
    final_order_by_archived_thesis: dict[str, str] = {}
    tracker = ThesisTracker(journal)
    for row in sell_orders:
        thesis_id = str(row.get("thesis_id") or "")
        thesis = tracker.get(thesis_id) if thesis_id else None
        if thesis and thesis.status == "archived":
            final_order_by_archived_thesis[thesis_id] = str(row.get("order_intent_id") or "")

    created_events = 0
    tracked_exits = 0
    thesis_ids: set[str] = set()
    for row in sell_orders:
        quantity, price, broker_order_id, filled_at = _order_fill_values(row)
        ticker = str(row.get("ticker") or "").upper().strip()
        thesis_id = str(row.get("thesis_id") or "")
        if not ticker or quantity <= 0 or price <= 0:
            continue
        event_id = _fill_event_key(
            broker_order_id=broker_order_id,
            order_intent_id=str(row.get("order_intent_id") or ""),
            ticker=ticker,
        )
        existing = journal.get_broker_fill_effect(event_id)
        action, reason = _sell_council_context(row)
        tracked = tracker.get(thesis_id) if thesis_id else None
        tx_summary = _portfolio_sell_summary(
            portfolio,
            ticker=ticker,
            broker_order_id=broker_order_id,
        )
        reconstructed = fifo_accounting.get(broker_order_id)
        cost_per_share = (
            float(reconstructed["cost_basis_per_share"])
            if reconstructed
            else (
                tx_summary["cost_basis"] / tx_summary["shares"]
                if tx_summary["shares"] > EPSILON and tx_summary["cost_basis"] > 0
                else _number(getattr(tracked, "entry_price", None)) or price
            )
        )
        realized = (
            float(reconstructed["realized_pnl"])
            if reconstructed
            else (
                tx_summary["realized_pnl"]
                if tx_summary["shares"] > EPSILON
                else (price - cost_per_share) * quantity
            )
        )
        quantity_tolerance = max(EPSILON, abs(quantity) * 0.001)
        if reconstructed:
            event_data_quality = str(reconstructed["data_quality"])
        elif tx_summary["shares"] <= EPSILON:
            event_data_quality = "broker_fill_with_inferred_cost"
        elif abs(tx_summary["shares"] - quantity) > quantity_tolerance:
            # Legacy callback races sometimes recorded P&L for only the local
            # residual while the broker row contained the full order quantity.
            # Keep the historical number, but make its uncertainty explicit so
            # it cannot masquerade as exact expectancy evidence.
            event_data_quality = "legacy_portfolio_quantity_conflict"
        else:
            event_data_quality = "broker_fill"
        is_final_archived_order = bool(
            thesis_id
            and final_order_by_archived_thesis.get(thesis_id)
            == str(row.get("order_intent_id") or "")
        )
        event_type = "EXIT" if is_final_archived_order or (not thesis_id and action in EXIT_ACTIONS) else "TRIM"
        journal.save_sell_trade_event(
            {
                "event_id": event_id,
                "broker_order_id": broker_order_id or None,
                "order_intent_id": row.get("order_intent_id"),
                "ticker": ticker,
                "thesis_id": thesis_id or None,
                "event_type": event_type,
                "shares": quantity,
                "average_price": price,
                "cost_basis_per_share": cost_per_share,
                "proceeds": quantity * price,
                "realized_pnl": realized,
                "position_shares_before": (
                    reconstructed.get("position_shares_before") if reconstructed else None
                ),
                "position_shares_after": (
                    reconstructed.get("position_shares_after") if reconstructed else None
                ),
                "sell_reason": reason,
                "position_type": str(getattr(tracked, "position_type", "") or ""),
                "source": "historical_execution_backfill",
                "data_quality": event_data_quality,
                "filled_at": filled_at,
            }
        )
        journal.save_broker_fill_effect(
            {
                "event_key": event_id,
                "broker_order_id": broker_order_id or None,
                "order_intent_id": row.get("order_intent_id"),
                "ticker": ticker,
                "side": "sell",
                "state": "filled",
                "cumulative_quantity": quantity,
                "average_price": price,
                "applied_quantity": quantity,
                "source": "historical_execution_backfill",
                "portfolio_effect_status": (
                    "historical_reconstructed_from_broker_fills"
                    if reconstructed
                    else (
                        "historical_recorded"
                        if tx_summary["shares"] > EPSILON
                        else "historical_unreconciled"
                    )
                ),
                "thesis_effect_status": "historical",
                "tracking_effect_status": "historical",
                "details": {
                    "action": action,
                    "event_type": event_type,
                    "data_quality": event_data_quality,
                    "broker_quantity": quantity,
                    "portfolio_transaction_quantity": tx_summary["shares"],
                    "cost_basis_method": "fifo" if reconstructed else "legacy_local",
                    "cost_basis_buy_order_ids": (
                        reconstructed.get("buy_order_ids", []) if reconstructed else []
                    ),
                },
            }
        )
        if not existing:
            created_events += 1
        if thesis_id:
            thesis_ids.add(thesis_id)
        if event_type == "EXIT" and tracked and tracked.status == "archived":
            tracking_id = _stable_id("post_sell", event_id, thesis_id)
            PostSellTracker(journal).record_sell(
                ticker=ticker,
                thesis_id=thesis_id,
                sell_price=price,
                sell_reason=reason,
                shares=quantity,
                position_type=str(tracked.position_type or ""),
                tracking_id=tracking_id,
                sell_date=filled_at,
                broker_order_id=broker_order_id,
                order_intent_id=str(row.get("order_intent_id") or ""),
                sell_event_id=event_id,
                cost_basis=cost_per_share * quantity,
                realized_pnl=realized,
                data_quality=event_data_quality,
            )
            tracked_exits += 1

    episodes = 0
    for thesis_id in sorted(thesis_ids):
        thesis = tracker.get(thesis_id)
        if not thesis:
            continue
        if rebuild_position_episode(
            journal,
            thesis_id=thesis_id,
            ticker=thesis.ticker,
            position_type=thesis.position_type,
            source="historical_execution_backfill",
        ):
            episodes += 1
    all_events = journal.get_sell_trade_events(limit=5000)
    all_episodes = journal.get_position_trade_episodes(limit=5000)
    event_types = {
        "TRIM": sum(1 for row in all_events if str(row.get("event_type") or "").upper() == "TRIM"),
        "EXIT": sum(1 for row in all_events if str(row.get("event_type") or "").upper() == "EXIT"),
    }
    episode_statuses = {
        "open": sum(1 for row in all_episodes if str(row.get("status") or "").lower() == "open"),
        "closed": sum(1 for row in all_episodes if str(row.get("status") or "").lower() == "closed"),
    }
    return {
        "sell_orders_seen": len(sell_orders),
        "events_created": created_events,
        "episodes_rebuilt": episodes,
        "post_sell_tracking_upserts": tracked_exits,
        "event_types": event_types,
        "episode_statuses": episode_statuses,
        "unlinked_sell_orders": sum(1 for row in sell_orders if not row.get("thesis_id")),
        "inferred_cost_events": sum(
            1 for row in all_events if "inferred_cost" in str(row.get("data_quality") or "")
        ),
        "broker_fifo_reconstructed_events": sum(
            1
            for row in all_events
            if str(row.get("data_quality") or "") == "broker_fill_fifo_reconstruction"
        ),
    }
