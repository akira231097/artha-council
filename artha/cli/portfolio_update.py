"""CLI helper for Ammu to update portfolio state.

Ammu (the OpenClaw bot) calls this via exec/subprocess and reads JSON output.
All commands output a JSON object with {success, message, data} shape.

Usage:
    python -m artha.cli.portfolio_update buy NVDA --shares 10 --price 135.00 --thesis-id abc-123
    python -m artha.cli.portfolio_update sell NVDA --shares 10 --price 150.00
    python -m artha.cli.portfolio_update add NVDA --shares 5 --price 140.00
    python -m artha.cli.portfolio_update trim NVDA --shares 2 --price 185.00
    python -m artha.cli.portfolio_update activate-thesis THESIS_ID --entry-price 135.00 --shares 10
    python -m artha.cli.portfolio_update list-pending
    python -m artha.cli.portfolio_update status NVDA
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from artha.journal import DecisionJournal
from artha.thesis_tracker import ThesisTracker, _HARD_STOP, _REVIEW_DAYS, _utcnow_iso, _add_days
from artha.portfolio import Portfolio, PORTFOLIO_FILE
from artha.config import Config


def _out(success: bool, message: str, data: Any = None) -> None:
    """Print JSON result and exit."""
    result = {"success": success, "message": message, "data": data or {}}
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if success else 1)


def _load_portfolio() -> Portfolio:
    return Portfolio.load(PORTFOLIO_FILE)


def _save_portfolio(portfolio: Portfolio) -> None:
    portfolio.save(PORTFOLIO_FILE)


def cmd_buy(args: argparse.Namespace) -> None:
    """Record a new buy position and activate/create thesis."""
    ticker = args.ticker.upper()
    shares = float(args.shares)
    price = float(args.price)
    position_type = (args.position_type or "BUY").upper()
    thesis_id = args.thesis_id

    tracker = ThesisTracker()

    # Determine if this is an add to an already-tracked position
    _portfolio_check = _load_portfolio()
    _existing_position = next(
        (p for p in _portfolio_check.positions if p.ticker.upper() == ticker), None
    )
    is_add_to_existing = _existing_position is not None

    # --- Activate or create thesis ---
    thesis = None
    if thesis_id:
        thesis = tracker.get(thesis_id)
        if thesis and thesis.status == "pending":
            thesis = tracker.activate_thesis(thesis_id, price, shares=shares)
        elif thesis and thesis.status == "active":
            pass  # Already active; allow re-confirmation
    else:
        # FIX C: ALWAYS check for active thesis first, regardless of portfolio state.
        # This prevents creating a duplicate thesis when portfolio.json is out of sync.
        thesis = tracker.get_active(ticker)
        if thesis is None:
            # Try to find a pending thesis for this ticker
            thesis = tracker.get_pending_for_ticker(ticker)
            if thesis:
                thesis = tracker.activate_thesis(thesis.thesis_id, price, shares=shares)
            elif not is_add_to_existing:
                # No active/pending thesis → create a minimal one (only for genuinely new positions)
                thesis = tracker.create_thesis(
                    ticker=ticker,
                    position_type=position_type,
                    thesis_summary="Manual purchase — no council thesis. Monitoring with default rules.",
                    notes=f"Manual buy {shares} shares @ ${price:.2f}",
                )
                thesis = tracker.activate_thesis(thesis.thesis_id, price, shares=shares)

    if not thesis:
        _out(False, f"Failed to create/activate thesis for {ticker}")
        return

    # --- Derive sell rules from thesis (thesis is source of truth, not CLI arg) ---
    # NEW ISSUE 3 fix: position_type/hard_stop/review_days come from the thesis
    effective_type = (thesis.position_type or position_type).upper()
    portfolio = _load_portfolio()
    stop_pct = _HARD_STOP.get(effective_type, Config.SELL_HARD_STOP_LEGACY)
    hard_stop = round(price * (1 + stop_pct), 4)
    review_days = _REVIEW_DAYS.get(effective_type, 30)

    # Check for existing position (add shares)
    existing = next((p for p in portfolio.positions if p.ticker.upper() == ticker), None)
    if existing:
        old_value = float(existing.shares) * float(existing.avg_cost)
        new_value = shares * price
        total_shares = float(existing.shares) + shares
        new_avg = (old_value + new_value) / total_shares if total_shares > 0 else price
        existing.shares = total_shares
        existing.avg_cost = round(new_avg, 4)
        existing.current_price = price
        existing.market_value = round(total_shares * price, 4)
        # Update sell fields
        _set_position_sell_fields(existing, thesis, hard_stop, review_days)
    else:
        from artha.portfolio import Position
        pos = Position(
            ticker=ticker,
            shares=shares,
            avg_cost=price,
            opened_at=datetime.now(timezone.utc).isoformat(),  # FIX A: required field
            current_price=price,
            market_value=round(shares * price, 4),
            asset_type="stock",
        )
        _set_position_sell_fields(pos, thesis, hard_stop, review_days)
        portfolio.positions.append(pos)

    # Compute NAV and allocation BEFORE persisting (FIX: move save after all validation)
    nav = portfolio.total_nav()
    alloc_pct = round(shares * price / nav * 100, 2) if nav > 0 else 0

    confirmation = (
        f"✅ Portfolio updated!\n\n"
        f"📊 {ticker} — {effective_type} Position\n"
        f"• {shares} shares @ ${price:.2f} = ${shares * price:.2f}\n"
        f"• Allocation: {alloc_pct:.1f}% of NAV\n"
        f"• Hard stop: ${hard_stop:.2f} ({stop_pct:.0%})\n"
    )
    if thesis.price_target:
        upside = (thesis.price_target - price) / price * 100
        confirmation += f"• Price target: ${thesis.price_target:.2f} (+{upside:.1f}%)\n"
    confirmation += f"• First review: {_add_days(review_days)[:10]}\n\n"
    if thesis.thesis_summary:
        confirmation += f"📋 Thesis: {thesis.thesis_summary[:200]}\n\n"
    if thesis.invalidation_conditions:
        confirmation += "🎯 Watching for invalidation:\n"
        for cond in thesis.invalidation_conditions[:5]:
            confirmation += f"• {cond}\n"
    confirmation += "\nMonitoring starts now. 🔒"

    # Persist after all validation/computation succeeds
    _save_portfolio(portfolio)

    _out(True, confirmation, {
        "ticker": ticker,
        "shares": shares,
        "price": price,
        "hard_stop": hard_stop,
        "thesis_id": thesis.thesis_id,
        "next_review": thesis.next_review_date,
    })


def _set_position_sell_fields(pos: Any, thesis: Any, hard_stop: float, review_days: int) -> None:
    """Attach sell-engine fields to a Position object (duck-typed)."""
    pos.thesis_id = thesis.thesis_id
    pos.position_type = thesis.position_type
    pos.entry_date = thesis.entry_date or _utcnow_iso()[:10]
    pos.hard_stop_price = hard_stop
    pos.trailing_stop_price = None
    pos.next_sell_review = _add_days(review_days)[:10]
    pos.sell_cooldown_until = thesis.sell_cooldown_until
    pos.scale_out_completed = []


def cmd_sell(args: argparse.Namespace) -> None:
    """Record a full or partial sell and archive/update thesis."""
    ticker = args.ticker.upper()
    shares = float(args.shares)
    price = float(args.price)
    reason = getattr(args, "reason", "") or "Manual sell"

    portfolio = _load_portfolio()
    existing = next((p for p in portfolio.positions if p.ticker.upper() == ticker), None)
    if not existing:
        _out(False, f"{ticker} not found in portfolio")
        return

    tracker = ThesisTracker()
    thesis_id = getattr(existing, "thesis_id", None)

    current_shares = float(existing.shares)
    is_full_exit = shares >= current_shares * 0.99  # 99%+ → treat as full exit

    avg_cost_at_sell = float(existing.avg_cost)  # capture before potential removal
    pnl = (price - avg_cost_at_sell) * shares
    pnl_pct = (price - avg_cost_at_sell) / avg_cost_at_sell * 100 if avg_cost_at_sell else 0

    if is_full_exit:
        portfolio.positions = [p for p in portfolio.positions if p.ticker.upper() != ticker]
        if thesis_id:
            tracker.archive_thesis(thesis_id, exit_price=price, exit_reason=reason)

        # Add to recently_exited list if schema supports it
        if hasattr(portfolio, "recently_exited"):
            portfolio.recently_exited = getattr(portfolio, "recently_exited", []) or []
            portfolio.recently_exited.insert(0, {
                "ticker": ticker,
                "exit_date": _utcnow_iso()[:10],
                "exit_price": price,
                "exit_reason": reason,
                "thesis_id": thesis_id,
                "shadow_tracking": True,
            })

        # Start post-sell tracking
        if thesis_id:
            _start_post_sell_tracking(ticker, thesis_id, price, reason, shares,
                                      getattr(existing, "position_type", "BUY"))
    else:
        # Partial sell (trim)
        existing.shares = round(current_shares - shares, 6)
        existing.market_value = round(existing.shares * float(existing.current_price or price), 4)
        if thesis_id:
            tracker.set_cooldown(thesis_id, Config.SELL_COOLDOWN_AFTER_TRIM)

    # Reduce cash_deployed by the cost basis that was released on this sell
    # (mirrors Portfolio.sell_position() logic; avoids a nonexistent portfolio.cash field)
    cost_basis_removed = avg_cost_at_sell * shares
    portfolio.cash_deployed = max(0.0, portfolio.cash_deployed - cost_basis_removed)
    portfolio.transactions.append({
        "type": "SELL",
        "ticker": ticker,
        "shares": shares,
        "price": price,
        "total": round(shares * price, 2),
        "realized_pnl": round(pnl, 2),
        "timestamp": _utcnow_iso(),
        "notes": reason,
    })
    portfolio.last_updated = _utcnow_iso()

    _save_portfolio(portfolio)

    action = "EXITED" if is_full_exit else "TRIMMED"
    proceeds = shares * price
    msg = (
        f"{'✅' if pnl >= 0 else '🔴'} Portfolio updated — {action} {ticker}\n\n"
        f"• Sold {shares} shares @ ${price:.2f}\n"
        f"• P&L: ${pnl:.2f} ({pnl_pct:+.1f}%)\n"
        f"• Proceeds: ${proceeds:.2f}\n"
    )
    if not is_full_exit:
        msg += f"• Remaining: {existing.shares:.4f} shares\n"
        msg += f"• Sell cooldown: {Config.SELL_COOLDOWN_AFTER_TRIM} days\n"
    else:
        msg += "• Post-sell shadow tracking started (5/20/60 day checkpoints)\n"

    _out(True, msg, {"ticker": ticker, "shares_sold": shares, "price": price, "pnl": pnl})


def _start_post_sell_tracking(
    ticker: str,
    thesis_id: str,
    sell_price: float,
    exit_reason: str,
    shares: float,
    position_type: str,
) -> None:
    """Create a post-sell tracking record in the database."""
    import uuid as _uuid
    journal = DecisionJournal()
    journal.save_post_sell_tracking({
        "tracking_id": str(_uuid.uuid4()),
        "ticker": ticker,
        "thesis_id": thesis_id,
        "sell_date": _utcnow_iso()[:10],
        "sell_price": sell_price,
        "sell_reason": exit_reason,
        "position_type": position_type,
        "shares": shares,
        "status": "tracking",
    })


def cmd_add(args: argparse.Namespace) -> None:
    """Add more shares to existing position (pyramid/average-down)."""
    args.position_type = None  # Will inherit from existing
    # Delegate to buy which handles existing positions
    cmd_buy(args)


def cmd_trim(args: argparse.Namespace) -> None:
    """Trim a portion of a position."""
    cmd_sell(args)


def cmd_activate_thesis(args: argparse.Namespace) -> None:
    """Activate a pending thesis (called after Sarath buys on Fidelity)."""
    thesis_id = args.thesis_id
    entry_price = float(args.entry_price)
    shares = float(args.shares) if args.shares else None

    tracker = ThesisTracker()
    thesis = tracker.activate_thesis(thesis_id, entry_price, shares=shares)
    if not thesis:
        _out(False, f"Failed to activate thesis {thesis_id}")
        return

    # FIX B: Update matching portfolio position OR create one if none exists.
    # activate-thesis must be self-sufficient — monitor must see the holding.
    portfolio = _load_portfolio()
    pos = portfolio.get_position(thesis.ticker)
    stop_pct = _HARD_STOP.get(thesis.position_type, Config.SELL_HARD_STOP_LEGACY)
    hard_stop = round(entry_price * (1 + stop_pct), 4)
    review_days = _REVIEW_DAYS.get(thesis.position_type, 30)
    if pos:
        _set_position_sell_fields(pos, thesis, hard_stop, review_days)
    else:
        # No portfolio entry — create one from thesis/CLI data so monitor tracks it
        from artha.portfolio import Position
        _shares = float(shares) if shares else 0.0
        pos = Position(
            ticker=thesis.ticker,
            shares=_shares,
            avg_cost=entry_price,
            opened_at=datetime.now(timezone.utc).isoformat(),
            asset_type="stock",
            current_price=entry_price,
            market_value=round(_shares * entry_price, 4),
        )
        _set_position_sell_fields(pos, thesis, hard_stop, review_days)
        portfolio.positions.append(pos)
    _save_portfolio(portfolio)

    _out(True, f"Thesis {thesis_id[:8]} activated for {thesis.ticker} @ ${entry_price:.2f}", {
        "thesis_id": thesis_id,
        "ticker": thesis.ticker,
        "hard_stop": thesis.hard_stop_price,
        "next_review": thesis.next_review_date,
    })


def cmd_list_pending(args: argparse.Namespace) -> None:
    """List all pending (non-expired) theses."""
    journal = DecisionJournal()
    rows = journal.get_pending_theses()
    if not rows:
        _out(True, "No pending theses found.", {"theses": []})
        return

    theses = []
    for row in rows:
        theses.append({
            "thesis_id": row.get("thesis_id", "")[:12] + "...",
            "full_id": row.get("thesis_id", ""),
            "ticker": row.get("ticker"),
            "position_type": row.get("position_type"),
            "price_target": row.get("price_target"),
            "stop_loss_pct": row.get("stop_loss_pct"),
            "recommended_allocation_pct": row.get("recommended_allocation_pct"),
            "thesis_summary": (row.get("thesis_summary") or "")[:150],
            "pending_expiry": row.get("pending_expiry", "")[:10] if row.get("pending_expiry") else None,
            "created_at": row.get("created_at", "")[:10],
        })

    summary = "\n".join(
        f"• {t['ticker']} ({t['position_type']}) — expires {t['pending_expiry'] or 'N/A'}"
        for t in theses
    )
    _out(True, f"Found {len(theses)} pending thesis/theses:\n{summary}", {"theses": theses})


def cmd_status(args: argparse.Namespace) -> None:
    """Show thesis and position status for a ticker."""
    ticker = args.ticker.upper()
    tracker = ThesisTracker()

    active = tracker.get_active(ticker)
    pending = tracker.get_pending_for_ticker(ticker)

    portfolio = _load_portfolio()
    position = next((p for p in portfolio.positions if p.ticker.upper() == ticker), None)

    data: dict[str, Any] = {"ticker": ticker, "has_position": position is not None}

    if active:
        data["active_thesis"] = {
            "thesis_id": active.thesis_id,
            "position_type": active.position_type,
            "entry_price": active.entry_price,
            "entry_date": active.entry_date,
            "hard_stop_price": active.hard_stop_price,
            "price_target": active.price_target,
            "thesis_health_score": active.thesis_health_score,
            "next_review_date": active.next_review_date,
            "days_held": active.days_held,
            "in_cooldown": active.in_cooldown,
            "in_minimum_hold": active.in_minimum_hold,
            "invalidation_conditions": active.invalidation_conditions,
        }

    if pending and not active:
        data["pending_thesis"] = {
            "thesis_id": pending.thesis_id,
            "position_type": pending.position_type,
            "price_target": pending.price_target,
            "recommended_allocation_pct": pending.recommended_allocation_pct,
            "pending_expiry": pending.pending_expiry,
        }

    msg_parts = [f"📊 {ticker} Status"]
    if position:
        pnl_pct = (float(position.current_price or 0) - float(position.avg_cost)) / float(position.avg_cost) * 100 if float(position.avg_cost) > 0 else 0
        msg_parts.append(f"Position: {position.shares} shares @ ${position.avg_cost:.2f} | P&L: {pnl_pct:+.1f}%")
    if active:
        msg_parts.append(f"Thesis health: {active.thesis_health_score}/100")
        msg_parts.append(f"Hard stop: ${active.hard_stop_price:.2f}")
        msg_parts.append(f"Next review: {(active.next_review_date or 'N/A')[:10]}")
    elif pending:
        msg_parts.append(f"⏳ Pending thesis (expires {(pending.pending_expiry or 'N/A')[:10]})")
    else:
        msg_parts.append("No active thesis found")

    _out(True, "\n".join(msg_parts), data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Artha portfolio update CLI — called by Ammu"
    )
    sub = parser.add_subparsers(dest="command")

    # buy
    p_buy = sub.add_parser("buy", help="Record a new buy")
    p_buy.add_argument("ticker")
    p_buy.add_argument("--shares", required=True, type=float)
    p_buy.add_argument("--price", required=True, type=float)
    p_buy.add_argument("--thesis-id", dest="thesis_id", default=None)
    p_buy.add_argument("--position-type", dest="position_type", default="BUY")

    # sell
    p_sell = sub.add_parser("sell", help="Record a sell/exit")
    p_sell.add_argument("ticker")
    p_sell.add_argument("--shares", required=True, type=float)
    p_sell.add_argument("--price", required=True, type=float)
    p_sell.add_argument("--reason", default="")

    # add
    p_add = sub.add_parser("add", help="Add shares to existing position")
    p_add.add_argument("ticker")
    p_add.add_argument("--shares", required=True, type=float)
    p_add.add_argument("--price", required=True, type=float)
    p_add.add_argument("--thesis-id", dest="thesis_id", default=None)
    p_add.add_argument("--position-type", dest="position_type", default=None)

    # trim
    p_trim = sub.add_parser("trim", help="Trim position (partial sell)")
    p_trim.add_argument("ticker")
    p_trim.add_argument("--shares", required=True, type=float)
    p_trim.add_argument("--price", required=True, type=float)
    p_trim.add_argument("--reason", default="Trim")

    # activate-thesis
    p_act = sub.add_parser("activate-thesis", help="Activate a pending thesis")
    p_act.add_argument("thesis_id")
    p_act.add_argument("--entry-price", dest="entry_price", required=True, type=float)
    p_act.add_argument("--shares", default=None, type=float)

    # list-pending
    sub.add_parser("list-pending", help="List pending theses")

    # status
    p_status = sub.add_parser("status", help="Show status for a ticker")
    p_status.add_argument("ticker")

    args = parser.parse_args()

    dispatch = {
        "buy": cmd_buy,
        "sell": cmd_sell,
        "add": cmd_add,
        "trim": cmd_trim,
        "activate-thesis": cmd_activate_thesis,
        "list-pending": cmd_list_pending,
        "status": cmd_status,
    }

    if not args.command or args.command not in dispatch:
        parser.print_help()
        sys.exit(1)

    try:
        dispatch[args.command](args)
    except Exception as e:
        _out(False, f"Command failed: {e}")


if __name__ == "__main__":
    main()
