"""OpenClaw/Robinhood bridge for Artha's human-approved trading loop.

Artha does not call Robinhood MCP directly from launchd. OpenClaw/Ammu owns the
broker tools. This module provides the durable handshake: snapshots, action
tokens, runtime kill switch, order-operation payloads, and fill activation.
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import Config
from .execution import (
    build_robinhood_review_request,
    evaluate_broker_snapshot_guardrails,
    mask_account_number,
    normalize_robinhood_position_snapshot,
)
from .journal import DecisionJournal
from .portfolio import PORTFOLIO_FILE, Portfolio, Position
from .thesis_tracker import ThesisTracker, _HARD_STOP, _REVIEW_DAYS, _add_days

logger = logging.getLogger(__name__)

OPEN_ORDER_STATES = {
    "new",
    "queued",
    "confirmed",
    "unconfirmed",
    "partially_filled",
    "pending_cancelled",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat()


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", ""))
    except Exception:
        return None


def _as_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            payload = json.loads(value)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    return {}


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class SnapshotHandoffValidationError(ValueError):
    """Raised when an OpenClaw snapshot handoff cannot be trusted."""

    def __init__(self, message: str, validation: dict[str, Any]):
        super().__init__(message)
        self.validation = validation


def validate_snapshot_handoff_metadata(
    payload: dict[str, Any],
    *,
    expected_run_id: str | None = None,
    min_generated_at: str | datetime | None = None,
    max_age_minutes: float | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate the per-run metadata before writing/importing a broker snapshot.

    This protects the OpenClaw cron from reporting a false PASS after a failed
    file write by accidentally importing a stale tmp file from an older run.
    """
    now_utc = (now or _utcnow()).astimezone(timezone.utc)
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    actual_run_id = str(payload.get("run_id") or "").strip()
    expected = str(expected_run_id or "").strip()
    if expected:
        passed = actual_run_id == expected
        checks.append({"name": "run_id", "expected": expected, "actual": actual_run_id or None, "passed": passed})
        if not passed:
            errors.append("Snapshot handoff run_id does not match the current cron run.")

    generated_raw = payload.get("generated_at") or payload.get("sync_started_at") or payload.get("handoff_written_at")
    generated_at = _parse_time(generated_raw)
    if expected or min_generated_at is not None or max_age_minutes is not None:
        checks.append(
            {
                "name": "generated_at_present",
                "actual": generated_raw,
                "passed": generated_at is not None,
            }
        )
        if generated_at is None:
            errors.append("Snapshot handoff is missing valid generated_at metadata.")

    min_generated_dt = _parse_time(min_generated_at)
    if min_generated_at is not None:
        passed = bool(generated_at and min_generated_dt and generated_at >= min_generated_dt)
        checks.append(
            {
                "name": "min_generated_at",
                "expected": min_generated_dt.isoformat() if min_generated_dt else str(min_generated_at),
                "actual": generated_at.isoformat() if generated_at else None,
                "passed": passed,
            }
        )
        if not passed:
            errors.append("Snapshot handoff generated_at is older than the current cron run.")

    if max_age_minutes is not None:
        age_minutes = (
            max(0.0, (now_utc - generated_at).total_seconds() / 60.0)
            if generated_at is not None
            else None
        )
        passed = age_minutes is not None and age_minutes <= float(max_age_minutes)
        checks.append(
            {
                "name": "max_age_minutes",
                "expected": float(max_age_minutes),
                "actual": round(age_minutes, 2) if age_minutes is not None else None,
                "passed": passed,
            }
        )
        if not passed:
            errors.append("Snapshot handoff is too old for import.")

    validation = {
        "status": "PASS" if not errors else "FAIL",
        "run_id": actual_run_id or None,
        "generated_at": generated_at.isoformat() if generated_at else None,
        "checks": checks,
        "errors": errors,
    }
    if errors:
        raise SnapshotHandoffValidationError("; ".join(errors), validation)
    return validation


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


def _tool_data(payload: Any) -> dict[str, Any]:
    """Extract structured data from an MCP tool result or plain dict."""
    if not isinstance(payload, dict):
        return {}
    if isinstance(payload.get("structuredContent"), dict):
        data = payload["structuredContent"].get("data")
        return data if isinstance(data, dict) else {}
    content = payload.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "text":
                continue
            try:
                parsed = json.loads(str(item.get("text") or ""))
            except Exception:
                continue
            if isinstance(parsed, dict):
                data = parsed.get("data")
                return data if isinstance(data, dict) else parsed
    data = payload.get("data")
    return data if isinstance(data, dict) else payload


def select_agentic_account(accounts: list[dict[str, Any]]) -> dict[str, Any] | None:
    expected = str(Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER or "").strip()
    if expected:
        for account in accounts or []:
            actual = str(account.get("account_number") or account.get("rhs_account_number") or "").strip()
            if actual == expected:
                return account
    for account in accounts or []:
        if bool(account.get("agentic_allowed")):
            return account
    return None


def canonicalize_mcp_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    """Build Artha's snapshot envelope from raw MCP tool results or an envelope."""
    if payload.get("source") == "robinhood_mcp" and "positions" in payload:
        snapshot = dict(payload)
        positions_data = _tool_data(snapshot.get("positions") or {})
        orders_data = _tool_data(snapshot.get("orders") or {})
        accounts_data = _tool_data(snapshot.get("accounts") or {})
        portfolio_data = _tool_data(snapshot.get("portfolio") or {})
        positions = positions_data.get("positions") or positions_data.get("results") or []
        orders = orders_data.get("orders") or orders_data.get("results") or []
        accounts = accounts_data.get("accounts") or []
        portfolio = portfolio_data.get("portfolio") or portfolio_data
        if isinstance(snapshot.get("positions"), list):
            positions = snapshot["positions"]
        if isinstance(snapshot.get("orders"), list):
            orders = snapshot["orders"]
        if isinstance(snapshot.get("accounts"), list):
            accounts = snapshot["accounts"]
        if isinstance(snapshot.get("portfolio"), dict) and not portfolio:
            portfolio = snapshot["portfolio"]
        snapshot["positions"] = positions if isinstance(positions, list) else []
        snapshot["orders"] = orders if isinstance(orders, list) else []
        snapshot["accounts"] = accounts if isinstance(accounts, list) else []
        snapshot["portfolio"] = portfolio if isinstance(portfolio, dict) else {}
        if not isinstance(snapshot.get("account"), dict) and isinstance(snapshot.get("selected_account"), dict):
            snapshot["account"] = snapshot["selected_account"]
        if not isinstance(snapshot.get("account"), dict):
            snapshot["account"] = select_agentic_account(snapshot.get("accounts") or []) or {}
        snapshot.setdefault("generated_at", _utcnow_iso())
        return snapshot

    metadata = {
        key: payload[key]
        for key in ("run_id", "sync_started_at", "handoff_written_at", "snapshot_envelope_version")
        if payload.get(key) is not None
    }

    accounts_data = _tool_data(payload.get("accounts_response") or payload.get("accounts") or {})
    portfolio_data = _tool_data(payload.get("portfolio_response") or payload.get("portfolio") or {})
    positions_data = _tool_data(payload.get("positions_response") or payload.get("positions") or {})
    orders_data = _tool_data(payload.get("orders_response") or payload.get("orders") or {})

    accounts = accounts_data.get("accounts") or []
    positions = positions_data.get("positions") or positions_data.get("results") or []
    orders = orders_data.get("orders") or orders_data.get("results") or []
    portfolio = portfolio_data.get("portfolio") or portfolio_data
    account = (
        payload.get("account")
        if isinstance(payload.get("account"), dict)
        else payload.get("selected_account")
        if isinstance(payload.get("selected_account"), dict)
        else select_agentic_account(accounts)
    )

    return {
        **metadata,
        "generated_at": payload.get("generated_at") or payload.get("sync_started_at") or _utcnow_iso(),
        "source": "robinhood_mcp",
        "account": account or {},
        "accounts": accounts,
        "portfolio": portfolio if isinstance(portfolio, dict) else {},
        "positions": positions if isinstance(positions, list) else [],
        "orders": orders if isinstance(orders, list) else [],
    }


def write_robinhood_snapshot(payload: dict[str, Any], path: str | Path | None = None) -> dict[str, Any]:
    """Validate and atomically write the latest Robinhood MCP snapshot."""
    snapshot = canonicalize_mcp_snapshot(payload)
    normalized = normalize_robinhood_position_snapshot(snapshot)
    snapshot["validation"] = {
        "status": normalized.get("status"),
        "fresh": normalized.get("fresh"),
        "warnings": normalized.get("warnings") or [],
        "account_check": normalized.get("account_check"),
        "position_count": normalized.get("position_count"),
    }
    target = Path(path or Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE).expanduser()
    _atomic_write_json(target, snapshot)
    return {
        "status": normalized.get("status"),
        "path": str(target),
        "position_count": normalized.get("position_count"),
        "warnings": normalized.get("warnings") or [],
        "snapshot": snapshot,
    }


def load_robinhood_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path or Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE).expanduser()
    if not target.exists():
        return {"status": "MISSING", "path": str(target), "positions": [], "warnings": ["Snapshot file does not exist."]}
    payload = json.loads(target.read_text(encoding="utf-8"))
    snapshot = normalize_robinhood_position_snapshot(payload)
    snapshot["orders"] = payload.get("orders") if isinstance(payload, dict) else []
    snapshot["path"] = str(target)
    return snapshot


def get_trading_control(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path or Config.ROBINHOOD_CONTROL_FILE).expanduser()
    if not target.exists():
        return {
            "trading_disabled": bool(Config.ROBINHOOD_KILL_SWITCH),
            "reason": "Config kill switch is enabled." if Config.ROBINHOOD_KILL_SWITCH else "",
            "updated_at": None,
            "path": str(target),
        }
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
    except Exception as exc:
        payload = {"trading_disabled": True, "reason": f"Unreadable control file: {type(exc).__name__}: {exc}"}
    payload.setdefault("trading_disabled", False)
    payload["trading_disabled"] = bool(payload.get("trading_disabled") or Config.ROBINHOOD_KILL_SWITCH)
    if Config.ROBINHOOD_KILL_SWITCH and not payload.get("reason"):
        payload["reason"] = "Config kill switch is enabled."
    payload["path"] = str(target)
    return payload


def set_trading_disabled(disabled: bool, reason: str = "", path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path or Config.ROBINHOOD_CONTROL_FILE).expanduser()
    payload = {
        "trading_disabled": bool(disabled),
        "reason": str(reason or ("Disabled by Telegram kill switch." if disabled else "Re-enabled by operator.")),
        "updated_at": _utcnow_iso(),
    }
    _atomic_write_json(target, payload)
    return {**payload, "path": str(target)}


def _broker_position_symbol(row: dict[str, Any]) -> str:
    instrument = row.get("instrument") if isinstance(row.get("instrument"), dict) else {}
    return str(
        row.get("symbol")
        or row.get("ticker")
        or row.get("equity_symbol")
        or instrument.get("symbol")
        or ""
    ).upper().strip()


def _broker_position_quantity(row: dict[str, Any]) -> float:
    for key in ("quantity", "shares", "qty"):
        value = _as_float(row.get(key))
        if value is not None:
            return value
    return 0.0


def _broker_position_avg_cost(row: dict[str, Any]) -> float | None:
    for key in ("average_buy_price", "average_price", "avg_cost", "avg_price", "average_cost"):
        value = _as_float(row.get(key))
        if value is not None and value > 0:
            return value
    cost_basis = _as_float(row.get("cost_basis") or row.get("total_cost"))
    qty = _broker_position_quantity(row)
    if cost_basis and qty > 0:
        return cost_basis / qty
    return None


def _broker_position_price(row: dict[str, Any], avg_cost: float | None = None) -> float | None:
    for key in ("market_price", "current_price", "last_price", "price"):
        value = _as_float(row.get(key))
        if value is not None and value > 0:
            return value
    market_value = _as_float(row.get("market_value"))
    qty = _broker_position_quantity(row)
    if market_value and qty > 0:
        return market_value / qty
    return avg_cost


def _same_number(left: Any, right: Any, tolerance: float = 0.0001) -> bool:
    left_num = _as_float(left)
    right_num = _as_float(right)
    if left_num is None or right_num is None:
        return False
    return abs(left_num - right_num) <= tolerance


def _attach_sell_fields(pos: Position, thesis: Any, entry_price: float) -> None:
    effective_type = (getattr(thesis, "position_type", None) or "BUY").upper()
    thesis_stop = _as_float(getattr(thesis, "hard_stop_price", None))
    stop_pct = _as_float(getattr(thesis, "stop_loss_pct", None))
    if stop_pct is None:
        stop_pct = _HARD_STOP.get(effective_type, Config.SELL_HARD_STOP_LEGACY)
    review_days = _REVIEW_DAYS.get(effective_type, 30)
    pos.thesis_id = thesis.thesis_id
    pos.position_type = effective_type
    pos.entry_date = (thesis.entry_date or _utcnow_iso())[:10]
    pos.hard_stop_price = round(thesis_stop, 4) if thesis_stop and thesis_stop > 0 else round(entry_price * (1 + stop_pct), 4)
    pos.trailing_stop_price = getattr(thesis, "trailing_stop_price", None)
    pos.next_sell_review = (getattr(thesis, "next_review_date", None) or _add_days(review_days))[:10]
    pos.sell_cooldown_until = getattr(thesis, "sell_cooldown_until", None)
    pos.scale_out_completed = list(getattr(thesis, "scale_out_completed", []) or [])


def _sell_field_tuple(pos: Position) -> tuple[Any, ...]:
    return (
        pos.thesis_id,
        pos.position_type,
        round(float(pos.hard_stop_price or 0), 4),
        round(float(pos.trailing_stop_price or 0), 4),
        pos.next_sell_review,
        pos.sell_cooldown_until,
        tuple(pos.scale_out_completed or []),
    )


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _position_already_reflects_fill(
    pos: Position | None,
    thesis: Any,
    row: dict[str, Any],
    quantity: float,
    avg_price: float,
) -> bool:
    if not pos or not thesis:
        return False
    if getattr(pos, "thesis_id", None) and getattr(pos, "thesis_id", None) != getattr(thesis, "thesis_id", None):
        return False
    if not _same_number(getattr(pos, "shares", None), quantity, tolerance=0.000001):
        return False
    if not _same_number(getattr(pos, "avg_cost", None), avg_price, tolerance=0.01):
        return False
    position_time = _parse_dt(getattr(pos, "opened_at", None))
    order_time = _parse_dt(row.get("submitted_at") or row.get("created_at"))
    if position_time is None or order_time is None:
        return False
    return position_time >= order_time - timedelta(minutes=5)


def sync_snapshot_to_artha(
    snapshot_payload: dict[str, Any] | None = None,
    *,
    apply: bool = True,
    journal: DecisionJournal | None = None,
    portfolio_path: str | Path = PORTFOLIO_FILE,
) -> dict[str, Any]:
    """Reconcile a fresh Robinhood snapshot into Artha's portfolio state.

    Broker-only positions with a matching pending thesis are activated. Broker
    positions without a pending/active thesis are reported, not silently invented.
    """
    journal = journal or DecisionJournal()
    snapshot = (
        normalize_robinhood_position_snapshot(canonicalize_mcp_snapshot(snapshot_payload))
        if snapshot_payload is not None
        else load_robinhood_snapshot()
    )
    if snapshot.get("status") in {"MISSING", "WARN"} and not snapshot.get("fresh"):
        return {"status": "WARN", "applied": False, "reason": "Snapshot is missing or stale.", "snapshot": snapshot}

    order_sync = sync_orders_to_artha(snapshot, journal=journal, portfolio_path=portfolio_path)
    tracker = ThesisTracker(journal)
    portfolio = Portfolio.load(Path(portfolio_path))
    changed = False
    activated: list[dict[str, Any]] = []
    updated: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    snapshot_cash = _cash_from_snapshot(snapshot)
    if snapshot_cash is not None and abs(float(portfolio.cash_available or 0.0) - snapshot_cash) > 0.005:
        portfolio.cash_available = round(snapshot_cash, 2)
        changed = True

    for raw in snapshot.get("positions") or []:
        if not isinstance(raw, dict):
            continue
        ticker = _broker_position_symbol(raw)
        qty = _broker_position_quantity(raw)
        if not ticker or qty <= 0:
            continue
        avg_cost = _broker_position_avg_cost(raw)
        price = _broker_position_price(raw, avg_cost)
        if avg_cost is None or avg_cost <= 0:
            unresolved.append({"ticker": ticker, "reason": "Broker position is missing average cost.", "quantity": qty})
            continue

        pos = portfolio.get_position(ticker)
        active = tracker.get_active(ticker)
        pending = tracker.get_pending_for_ticker(ticker)
        if not active and pending and apply:
            active = tracker.activate_thesis(pending.thesis_id, avg_cost, shares=qty)

        if not pos:
            if not active:
                unresolved.append({"ticker": ticker, "reason": "Held in Robinhood but no active/pending Artha thesis.", "quantity": qty})
                continue
            pos = Position(
                ticker=ticker,
                asset_type="stock",
                shares=qty,
                avg_cost=avg_cost,
                opened_at=_utcnow_iso(),
                current_price=price,
                market_value=round(qty * float(price or avg_cost), 4),
                notes="Imported from Robinhood MCP snapshot.",
            )
            _attach_sell_fields(pos, active, avg_cost)
            portfolio.positions.append(pos)
            activated.append({"ticker": ticker, "quantity": qty, "avg_cost": avg_cost, "thesis_id": active.thesis_id})
            changed = True
            continue

        position_changed = False
        if abs(float(pos.shares or 0) - qty) > 0.0001:
            pos.shares = qty
            position_changed = True
        if abs(float(pos.avg_cost or 0) - avg_cost) > 0.0001:
            pos.avg_cost = avg_cost
            position_changed = True
        if price and abs(float(pos.current_price or 0) - price) > 0.0001:
            pos.current_price = price
            pos.market_value = round(qty * price, 4)
            position_changed = True
        if active:
            before_sell_fields = _sell_field_tuple(pos)
            _attach_sell_fields(pos, active, avg_cost)
            if _sell_field_tuple(pos) != before_sell_fields:
                position_changed = True
        if position_changed:
            updated.append({"ticker": ticker, "quantity": qty, "avg_cost": avg_cost, "thesis_id": getattr(pos, "thesis_id", None)})
            changed = True

    if apply and changed:
        portfolio.last_updated = _utcnow_iso()
        portfolio.save(Path(portfolio_path))

    return {
        "status": "PASS" if not unresolved and order_sync.get("status") == "PASS" else "WARN",
        "applied": bool(apply and changed),
        "activated": activated,
        "updated": updated,
        "unresolved": unresolved,
        "order_sync": order_sync,
        "snapshot_status": snapshot.get("status"),
        "position_count": len(snapshot.get("positions") or []),
    }


def _new_token() -> str:
    return secrets.token_urlsafe(8).replace("-", "").replace("_", "")[:11]


def _callback(verb: str, action_id: str, token: str) -> str:
    return f"artha:{verb}:{action_id}:{token}"


def parse_callback_data(value: str) -> dict[str, str]:
    parts = str(value or "").split(":")
    if len(parts) != 4 or parts[0] != "artha":
        raise ValueError("Unsupported Artha action token.")
    return {"verb": parts[1], "action_id": parts[2], "token": parts[3]}


def queue_trade_action_from_order_payload(
    order_payload: dict[str, Any],
    *,
    action_type: str | None = None,
    journal: DecisionJournal | None = None,
    message: str = "",
) -> dict[str, Any]:
    """Create a durable Telegram action for an execution order row."""
    journal = journal or DecisionJournal()
    intent = order_payload.get("intent") or {}
    broker = order_payload.get("broker_result") or {}
    row_id = order_payload.get("row_id")
    side = str(intent.get("side") or "").lower()
    ticker = str(intent.get("ticker") or "").upper()
    action = (action_type or ("trim" if side == "sell" else "buy")).lower()
    status = "review_ready" if str(broker.get("status") or "") in {"review_ready", "reviewed"} else "blocked"
    action_id = f"ta_{uuid.uuid4().hex[:12]}"
    expires_at = (_utcnow() + timedelta(minutes=max(1, int(Config.ROBINHOOD_ACTION_TOKEN_TTL_MINUTES)))).isoformat()
    row = {
        "action_id": action_id,
        "expires_at": expires_at,
        "status": status,
        "action_type": action,
        "ticker": ticker,
        "side": side,
        "execution_order_row": row_id,
        "order_intent_id": intent.get("order_intent_id"),
        "thesis_id": intent.get("thesis_id"),
        "account_number_masked": mask_account_number(Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER),
        "token_review": _new_token(),
        "token_place": _new_token(),
        "token_skip": _new_token(),
        "payload_json": order_payload,
        "message": message,
        "notes": "Created from Artha execution order review.",
    }
    journal.save_trade_action(row)
    row["reply_markup"] = build_trade_action_reply_markup(row)
    row["callback_data"] = {
        "review": _callback("review", action_id, row["token_review"]),
        "place": _callback("place", action_id, row["token_place"]),
        "skip": _callback("skip", action_id, row["token_skip"]),
    }
    return row


def build_trade_action_reply_markup(action: dict[str, Any]) -> dict[str, Any] | None:
    if not action:
        return None
    action_id = str(action.get("action_id") or "")
    ticker = str(action.get("ticker") or "?").upper()
    status = str(action.get("status") or "").lower()
    if status in {"blocked", "expired", "skipped"}:
        return {
            "inline_keyboard": [
                [{"text": f"Skip {ticker}", "callback_data": _callback("skip", action_id, str(action.get("token_skip") or ""))}],
            ]
        }
    if status in {"review_clear", "reviewed"}:
        return build_review_confirmation_reply_markup(action)
    return {
        "inline_keyboard": [
            [
                {"text": f"Review {ticker}", "callback_data": _callback("review", action_id, str(action.get("token_review") or ""))},
            ],
            [{"text": f"Skip {ticker}", "callback_data": _callback("skip", action_id, str(action.get("token_skip") or ""))}],
        ]
    }


def build_review_confirmation_reply_markup(action: dict[str, Any]) -> dict[str, Any] | None:
    if not action:
        return None
    action_id = str(action.get("action_id") or "")
    ticker = str(action.get("ticker") or "?").upper()
    side = str(action.get("side") or "").lower()
    verb = "Place Buy" if side == "buy" else "Place Sell"
    return {
        "inline_keyboard": [
            [{"text": f"{verb} {ticker}", "callback_data": _callback("place", action_id, str(action.get("token_place") or ""))}],
            [{"text": f"Skip {ticker}", "callback_data": _callback("skip", action_id, str(action.get("token_skip") or ""))}],
        ]
    }


def build_trade_action_notice(action: dict[str, Any]) -> str:
    payload = _as_json(action.get("payload_json"))
    intent = payload.get("intent") or {}
    broker = payload.get("broker_result") or {}
    guardrails = payload.get("guardrails") or {}
    side = str(intent.get("side") or action.get("side") or "").upper()
    ticker = str(intent.get("ticker") or action.get("ticker") or "?").upper()
    lines = [
        f"ARTHA {side} REVIEW - {ticker}",
        "----------------------",
        f"Amount: USD {float(intent.get('notional') or 0):.2f}",
        f"Quantity: {float(intent.get('quantity') or 0):.6f}",
        f"Limit: USD {float(intent.get('limit_price') or 0):.2f}",
        f"Audit row: {action.get('execution_order_row')}",
        f"Action id: {action.get('action_id')}",
        f"Status: {broker.get('status') or action.get('status')}",
        f"Guardrails: {guardrails.get('status') or 'unknown'}",
        "",
        "Review checks the Robinhood order without placing it.",
        "Place repeats review/safety checks and must abort if Robinhood returns alerts.",
    ]
    blocked = list((broker.get("response") or {}).get("blocked_reasons") or guardrails.get("reasons") or [])
    if blocked:
        lines.append("")
        lines.append("Blocked reasons:")
        lines.extend(f"- {reason}" for reason in blocked[:5])
    return "\n".join(lines)


def build_snapshot_refresh_operation() -> dict[str, Any]:
    """Return the exact read-only MCP sequence OpenClaw should run for a fresh snapshot."""
    handoff_path = str(
        Path(os.getenv("ARTHA_OPENCLAW_TMP_DIR", "/tmp")) / "artha_robinhood_snapshot.json"
    )
    lock_file = "/tmp/artha-robinhood-snapshot-sync.lock"
    return {
        "success": True,
        "operation": "refresh_robinhood_snapshot",
        "account_selector": {
            "agentic_allowed": True,
            "account_number_suffix": Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER[-4:],
            "account_number_masked": mask_account_number(Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER),
        },
        "mcp_sequence": [
            {"tool": "get_accounts", "args": {}},
            {"tool": "get_portfolio", "args": {"account_number": Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER}},
            {"tool": "get_equity_positions", "args": {"account_number": Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER}},
            {"tool": "get_equity_orders", "args": {"account_number": Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER}},
        ],
        "handoff_path": handoff_path,
        "handoff_write": {
            "method": "bash_base64_atomic_write",
            "reason": "OpenClaw node file_write may be unavailable when the local node is disconnected.",
        },
        "handoff_required_fields": ["run_id", "generated_at", "source", "account", "accounts", "portfolio", "positions", "orders"],
        "import_command": (
            "python run.py robinhood-snapshot-import --strict "
            f"--lock-file {lock_file} "
            "--expect-run-id <RUN_ID> --min-generated-at <SYNC_STARTED_AT> "
            f"--file {handoff_path} --control-center"
        ),
        "forbidden_tools": [
            "review_equity_order",
            "place_equity_order",
            "cancel_equity_order",
            "add_to_watchlist",
            "remove_from_watchlist",
        ],
        "message": (
            "Run read-only Robinhood MCP snapshot tools, write a minimal per-run canonical envelope, "
            "then import through robinhood-snapshot-import with strict run_id validation."
        ),
    }


def build_auto_buy_runner_operation() -> dict[str, Any]:
    """Return the durable OpenClaw cron contract for unattended auto-buy drain."""
    project_dir = str(Path(__file__).resolve().parent.parent)
    tmp_dir = os.getenv("ARTHA_OPENCLAW_TMP_DIR", "/tmp")
    telegram_chat = Config.TELEGRAM_CHAT_ID or "<telegram-chat-id>"
    bootstrap_message = f"""ARTHA ROBINHOOD AUTO-BUY RUNNER BOOTSTRAP

Goal: make the idle path fast and only load the full money-moving contract when Artha has queued auto_buy work.

Run exactly:
cd {project_dir} && .venv/bin/python run.py robinhood-auto-buy-queue-status

If operation_count is 0:
- Do not call any Robinhood MCP tools.
- Do not inspect the repo.
- Do not send Telegram.
- Final reply exactly: AUTO_BUY_IDLE

If operation_count is greater than 0:
- Run exactly:
  cd {project_dir} && .venv/bin/python run.py robinhood-auto-buy-runner-operation --message-only
- Follow the returned instructions exactly.
- Never call place_equity_order unless those returned instructions and Artha commands explicitly allow it.
"""
    runner_message = f"""ARTHA ROBINHOOD AUTO-BUY RUNNER (agentic, money-moving)

Goal: drain Artha's queued auto_buy actions during regular market hours using Robinhood MCP as the execution source of truth. This job may place real Robinhood equity orders only when every Artha and Robinhood gate below passes.

Hard boundaries:
- Long US equities only. No options, margin, shorts, crypto, watchlist writes, cancel_order, or manual order edits.
- Use the Agentic cash account ending {Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER[-4:]} only.
- Never invent order parameters. place_equity_order must use the exact place_mcp_args returned by Artha.
- Max two auto_buy actions per cron turn. If more are queued, leave them for the next turn.
- If any command or MCP call is ambiguous, stale, missing, or fails, block and send a concise Telegram alert to {telegram_chat}.
- Do not use shell heredocs. Use bash only for date/run_id generation, atomic base64 JSON writes, and Artha commands.

Market-time gate:
1. Run /bin/zsh -lc 'TZ=America/Chicago date +%u:%H:%M'
2. If it is weekend, before 08:30 CT, or at/after 15:00 CT, do not call review_equity_order or place_equity_order. Final reply exactly: AUTO_BUY_MARKET_CLOSED

Cheap queue preflight:
3. Run:
   cd {project_dir} && .venv/bin/python run.py robinhood-auto-buy-queue-status
4. If operation_count is 0, final reply exactly: AUTO_BUY_IDLE. Do not refresh Robinhood and do not call review_equity_order/place_equity_order.

Fresh snapshot gate:
5. Refresh Robinhood snapshot using read-only MCP tools: get_accounts, get_portfolio, get_equity_positions, get_equity_orders.
6. Build a minimal canonical snapshot envelope with run_id, generated_at, source=\"robinhood_mcp\", selected account, accounts, portfolio, positions, and orders.
7. Base64-encode that envelope and atomically write it to {tmp_dir}/artha_robinhood_snapshot.json.
8. Run:
   cd {project_dir} && .venv/bin/python run.py robinhood-snapshot-import --strict --expect-run-id RUN_ID --min-generated-at SYNC_STARTED_AT --max-handoff-age-minutes 10 --lock-file /tmp/artha-robinhood-snapshot-sync.lock --file {tmp_dir}/artha_robinhood_snapshot.json
9. If import/sync is not PASS, stop before review/place and alert Telegram.

Queue drain:
10. Run:
   cd {project_dir} && .venv/bin/python run.py robinhood-auto-buy-action
11. If operation_count is 0, final reply exactly: AUTO_BUY_IDLE
12. For each returned operation, process at most two successful/blocked actions and never skip ahead to manual buy actions.

Per-action agentic clearance:
13. Require operation=auto_tradability_review_then_place_equity_order, action_id, tradability_mcp_args, and review_mcp_args.
14. Call Robinhood MCP get_equity_quotes for the operation symbol.
15. Call get_equity_tradability with exactly tradability_mcp_args.
16. Call review_equity_order with exactly review_mcp_args. This is still preview only.
17. Atomically write quote, tradability, and review responses to JSON files under {tmp_dir}/artha_auto_buy_<ACTION_ID>_quote.json, _tradability.json, and _review.json.
18. Run:
   cd {project_dir} && .venv/bin/python run.py robinhood-auto-buy-agentic-clearance ACTION_ID --quote-file {tmp_dir}/artha_auto_buy_<ACTION_ID>_quote.json --tradability-file {tmp_dir}/artha_auto_buy_<ACTION_ID>_tradability.json --review-file {tmp_dir}/artha_auto_buy_<ACTION_ID>_review.json
19. If that command does not return allow_place=true/status PASS, do not place. Alert Telegram with ticker/action_id and the block reason.

Final broker review immediately before place:
20. Run:
   cd {project_dir} && .venv/bin/python run.py robinhood-auto-buy-action ACTION_ID
21. Require operation=tradability_then_review_then_place_equity_order and place_mcp_args.
22. Repeat get_equity_tradability and review_equity_order with exactly the returned args.
23. Write those second responses to JSON files and run:
   cd {project_dir} && .venv/bin/python run.py robinhood-record-review ACTION_ID --tradability-file SECOND_TRADABILITY_FILE --review-file SECOND_REVIEW_FILE
24. Run:
   cd {project_dir} && .venv/bin/python run.py robinhood-final-clearance ACTION_ID
25. If final clearance is not allow_place=true/status PASS, do not place. Alert Telegram with the block reason.
26. Run:
   cd {project_dir} && .venv/bin/python run.py robinhood-auto-buy-action ACTION_ID
27. Require place_mcp_args still exists and exactly matches the reviewed order. Then call place_equity_order with exactly place_mcp_args.
28. Write the place response JSON and run:
   cd {project_dir} && .venv/bin/python run.py robinhood-record-submission ACTION_ID --file PLACE_RESPONSE_FILE
29. Refresh/import snapshot again using the read-only snapshot gate, then send Telegram success/failure with ticker, notional/quantity, order id/state, and action_id.

Final reply rules:
- If no queue work: AUTO_BUY_IDLE
- If all queued work blocked safely: AUTO_BUY_BLOCKED
- If at least one order was placed and recorded: AUTO_BUY_PLACED
"""
    return {
        "success": True,
        "operation": "openclaw_auto_buy_runner",
        "cron": {
            "name": "Artha Robinhood Auto-Buy Runner",
            "description": "Unattended agentic drain loop for queued Artha auto_buy actions.",
            "expr": "*/2 8-14 * * 1-5",
            "tz": "America/Chicago",
            "timeout_seconds": 360,
            "thinking": "xhigh",
            "light_context": True,
            "delivery": {"mode": "none", "channel": "telegram", "to": telegram_chat},
            "failure_alert": {"after": 1, "channel": "telegram", "to": telegram_chat, "cooldown": "10m"},
        },
        "runner_message": runner_message,
        "bootstrap_message": bootstrap_message,
        "mcp_tools_required": [
            "get_accounts",
            "get_portfolio",
            "get_equity_positions",
            "get_equity_orders",
            "get_equity_quotes",
            "get_equity_tradability",
            "review_equity_order",
            "place_equity_order",
        ],
        "artha_commands_required": [
            "robinhood-auto-buy-queue-status",
            "robinhood-snapshot-import",
            "robinhood-auto-buy-action",
            "robinhood-auto-buy-agentic-clearance",
            "robinhood-record-review",
            "robinhood-final-clearance",
            "robinhood-record-submission",
        ],
        "forbidden_tools": [
            "cancel_equity_order",
            "add_to_watchlist",
            "remove_from_watchlist",
            "options",
            "margin",
            "crypto",
        ],
        "install_hint": (
            "openclaw cron add --name 'Artha Robinhood Auto-Buy Runner' "
            "--cron '*/2 8-14 * * 1-5' --tz America/Chicago --session isolated "
            "--thinking xhigh --timeout-seconds 360 --light-context --no-deliver "
            "--failure-alert --failure-alert-after 1 --failure-alert-channel telegram "
            f"--failure-alert-to {telegram_chat} --failure-alert-cooldown 10m --message '<bootstrap_message>'"
        ),
    }


def build_pending_auto_buy_queue_status(*, journal: DecisionJournal | None = None, limit: int = 10) -> dict[str, Any]:
    """Return queued auto-buy actions without mutating gates or statuses."""
    journal = journal or DecisionJournal()
    actions = []
    skipped = []
    for row in journal.get_trade_actions(limit=max(1, int(limit) * 5)):
        if str(row.get("action_type") or "").lower() != "auto_buy":
            continue
        status = str(row.get("status") or "").lower()
        item = {
            "action_id": row.get("action_id"),
            "ticker": row.get("ticker"),
            "side": row.get("side"),
            "status": status,
            "expires_at": row.get("expires_at"),
        }
        if status in {"review_ready", "review_clear", "reviewed"}:
            actions.append(item)
        else:
            skipped.append(item)
        if len(actions) >= limit:
            break
    return {
        "success": True,
        "operation": "auto_buy_queue_status",
        "operation_count": len(actions),
        "actions": actions,
        "skipped": skipped,
    }


def queue_review_actions_for_ready_orders(
    *,
    journal: DecisionJournal | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Create Telegram Review actions for fresh buy-side execution proposals.

    Pending theses are only intentions. This promotes the already-audited
    execution_order rows behind those theses into durable Review buttons.
    """
    journal = journal or DecisionJournal()
    existing_by_intent = {
        str(row.get("order_intent_id") or ""): row
        for row in journal.get_trade_actions(limit=500)
        if row.get("order_intent_id") and str(row.get("status") or "") not in {"skipped", "expired"}
    }
    existing_by_order_row = {
        int(row.get("execution_order_row")): row
        for row in journal.get_trade_actions(limit=500)
        if row.get("execution_order_row") and str(row.get("status") or "") not in {"skipped", "expired"}
    }
    created: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for order in journal.get_execution_orders(limit=limit):
        side = str(order.get("side") or "").lower()
        status = str(order.get("status") or "")
        intent_id = str(order.get("order_intent_id") or "")
        if side != "buy" or status not in {"dry_run_ready", "review_ready", "reviewed"}:
            continue
        if intent_id in existing_by_intent or int(order.get("id") or 0) in existing_by_order_row:
            skipped.append({"ticker": order.get("ticker"), "order_intent_id": intent_id, "reason": "trade_action_already_exists"})
            continue
        decision_gate = _action_decision_fresh_gate(
            {
                "execution_order_row": order.get("id"),
                "order_intent_id": intent_id,
                "created_at": order.get("created_at"),
            },
            journal,
        )
        if not decision_gate.get("passed"):
            skipped.append({"ticker": order.get("ticker"), "order_intent_id": intent_id, "reason": "stale_decision", "gate": decision_gate})
            continue
        payload = {
            "row_id": order.get("id"),
            "intent": _as_json(order.get("request_json")),
            "guardrails": _as_json(order.get("guardrail_json")),
            "broker_result": {"status": "review_ready", "broker": order.get("broker"), "dry_run": bool(order.get("dry_run"))},
            "execution_order": order,
        }
        action = queue_trade_action_from_order_payload(
            payload,
            action_type="buy",
            journal=journal,
            message=f"Fresh Review action created from pending Artha buy thesis for {order.get('ticker')}.",
        )
        created.append({k: v for k, v in action.items() if k not in {"payload_json"}})
    return {
        "success": True,
        "created_count": len(created),
        "skipped_count": len(skipped),
        "created": created,
        "skipped": skipped,
    }


def _trade_action_expired(row: dict[str, Any]) -> bool:
    expires = _parse_time(row.get("expires_at"))
    return bool(expires and _utcnow() > expires)


def _request_json_for_action(row: dict[str, Any], journal: DecisionJournal) -> dict[str, Any]:
    order_row = None
    if row.get("execution_order_row"):
        order_row = journal.get_execution_order_by_id(int(row["execution_order_row"]))
    if order_row is None and row.get("order_intent_id"):
        order_row = journal.get_execution_order_by_intent_id(str(row["order_intent_id"]))
    if not order_row:
        raise ValueError("Execution order row is missing.")
    request = _as_json(order_row.get("request_json"))
    if not request:
        payload = _as_json(row.get("payload_json"))
        intent = payload.get("intent") or {}
        request = build_robinhood_review_request_from_intent(intent)
    return request


def build_robinhood_review_request_from_intent(intent: dict[str, Any]) -> dict[str, Any]:
    from .execution import OrderIntent

    fields = {k: v for k, v in intent.items() if k in OrderIntent.__dataclass_fields__}
    return build_robinhood_review_request(OrderIntent(**fields))


def _order_intent_for_action(row: dict[str, Any]) -> Any:
    from .execution import OrderIntent

    payload = _as_json(row.get("payload_json"))
    intent_data = payload.get("intent") if isinstance(payload.get("intent"), dict) else {}
    fields = {k: v for k, v in intent_data.items() if k in OrderIntent.__dataclass_fields__}
    if not fields:
        raise ValueError("Trade action payload is missing its order intent.")
    return OrderIntent(**fields).normalized()


def _resolved_notional_for_intent(intent: Any) -> float | None:
    notional = _as_float(getattr(intent, "notional", None))
    if notional is not None:
        return notional
    quantity = _as_float(getattr(intent, "quantity", None))
    price = _as_float(getattr(intent, "limit_price", None)) or _as_float(getattr(intent, "estimated_price", None))
    if quantity is not None and price is not None:
        return quantity * price
    return None


def _tradability_mcp_args(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "account_number": str(request.get("account_number") or Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER or ""),
        "symbols": [str(request.get("symbol") or "").upper()],
    }


def _request_needs_fractional_tradability(request: dict[str, Any]) -> bool:
    if request.get("dollar_amount") is not None:
        return True
    quantity = _as_float(request.get("quantity"))
    if quantity is None:
        return False
    return abs(quantity - round(quantity)) > 0.000001


def _review_order_checks(review_response: dict[str, Any]) -> Any:
    data = _tool_data(review_response)
    checks = data.get("order_checks") if isinstance(data, dict) else {}
    return checks if isinstance(checks, (dict, list)) else {}


_NON_BLOCKING_ORDER_CHECK_TYPES = {"EQUITY_SUITABILITY"}
_BLOCKING_ORDER_CHECK_KEYWORDS = (
    "insufficient",
    "buying power",
    "halt",
    "not trade",
    "untradable",
    "restriction",
    "restricted",
    "rejected",
    "market closed",
    "closed",
    "pattern day",
    "pdt",
    "good faith",
    "cash account",
    "margin",
    "fractional",
    "not eligible",
    "not allowed",
    "exceeds",
    "cannot",
)


def classify_robinhood_order_checks(order_checks: Any) -> dict[str, Any]:
    """Separate informational broker alerts from placement-blocking alerts.

    Robinhood commonly returns an EQUITY_SUITABILITY alert for individual
    accounts. That is user-visible/compliance context, but not by itself a
    broker risk failure. Unknown alerts remain blocking by default.
    """
    if not order_checks:
        return {
            "has_checks": False,
            "blocking": False,
            "non_blocking": False,
            "alert_types": [],
            "blocking_reasons": [],
        }

    items = order_checks if isinstance(order_checks, list) else [order_checks]
    alert_types: list[str] = []
    blocking_reasons: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            blocking_reasons.append("Robinhood returned an unstructured broker order check.")
            continue
        alert_type = str(item.get("alertType") or item.get("alert_type") or item.get("type") or "").upper()
        if alert_type:
            alert_types.append(alert_type)
        text = json.dumps(item, ensure_ascii=True, sort_keys=True).lower()
        if alert_type not in _NON_BLOCKING_ORDER_CHECK_TYPES:
            blocking_reasons.append(f"Robinhood broker alert {alert_type or 'UNKNOWN'} is not in the non-blocking allowlist.")
        elif any(keyword in text for keyword in _BLOCKING_ORDER_CHECK_KEYWORDS):
            blocking_reasons.append(f"Robinhood broker alert {alert_type} contains blocking risk language.")

    return {
        "has_checks": True,
        "blocking": bool(blocking_reasons),
        "non_blocking": not blocking_reasons,
        "alert_types": sorted(set(alert_types)),
        "blocking_reasons": blocking_reasons,
    }


def _numeric_echo_matches(expected: Any, actual: Any, *, tolerance: float = 0.000001) -> bool:
    expected_number = _as_float(expected)
    actual_number = _as_float(actual)
    if expected_number is None or actual_number is None:
        return str(expected) == str(actual)
    return abs(expected_number - actual_number) <= tolerance


def _review_echo_matches(request: dict[str, Any], review_response: dict[str, Any]) -> tuple[bool, list[str]]:
    data = _tool_data(review_response)
    reasons: list[str] = []
    for key in ("symbol", "side", "type"):
        expected = str(request.get(key) or "").lower()
        actual = str(data.get(key) or "").lower()
        if expected and not actual:
            reasons.append(f"Robinhood review did not echo required {key}.")
        elif expected and actual and expected != actual:
            reasons.append(f"Robinhood review echoed {key}={actual}, expected {expected}.")
    if request.get("quantity") and not data.get("quantity"):
        reasons.append("Robinhood review did not echo required share quantity.")
    elif request.get("quantity") and data.get("quantity") and not _numeric_echo_matches(request["quantity"], data["quantity"]):
        reasons.append("Robinhood review echoed a different share quantity.")
    if request.get("dollar_amount") and not data.get("dollar_amount"):
        reasons.append("Robinhood review did not echo required dollar amount.")
    elif request.get("dollar_amount") and data.get("dollar_amount") and not _numeric_echo_matches(request["dollar_amount"], data["dollar_amount"]):
        reasons.append("Robinhood review echoed a different dollar amount.")
    if request.get("limit_price") and not data.get("limit_price"):
        reasons.append("Robinhood review did not echo required limit price.")
    elif request.get("limit_price") and data.get("limit_price") and not _numeric_echo_matches(request["limit_price"], data["limit_price"]):
        reasons.append("Robinhood review echoed a different limit price.")
    return not reasons, reasons


def _first_number_from_keys(payload: Any, keys: set[str]) -> float | None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = str(key or "").lower()
            if normalized in keys:
                number = _as_float(value)
                if number is not None:
                    return number
        for value in payload.values():
            nested = _first_number_from_keys(value, keys)
            if nested is not None:
                return nested
    elif isinstance(payload, list):
        for value in payload:
            nested = _first_number_from_keys(value, keys)
            if nested is not None:
                return nested
    return None


def _price_from_disclosure(disclosure: str, label: str) -> float | None:
    pattern = rf"\b{re.escape(label)}\b\s*\$?\s*([0-9]+(?:,[0-9]{{3}})*(?:\.[0-9]+)?)"
    match = re.search(pattern, disclosure or "", flags=re.IGNORECASE)
    return _as_float(match.group(1)) if match else None


def _review_quote_from_response(review_response: dict[str, Any]) -> dict[str, Any]:
    data = _tool_data(review_response)
    disclosure = str(data.get("market_data_disclosure") or data.get("quote_disclosure") or "")
    bid = (
        _first_number_from_keys(data, {"bid", "bid_price", "best_bid"})
        or _price_from_disclosure(disclosure, "Bid")
    )
    ask = (
        _first_number_from_keys(data, {"ask", "ask_price", "best_ask"})
        or _price_from_disclosure(disclosure, "Ask")
    )
    last = (
        _first_number_from_keys(data, {"last", "last_price", "mark", "mark_price", "price"})
        or _price_from_disclosure(disclosure, "Last")
    )
    return {
        "bid": bid,
        "ask": ask,
        "last": last,
        "disclosure": disclosure[:500] if disclosure else "",
    }


def _review_price_drift_gate(
    action: dict[str, Any],
    request: dict[str, Any],
    review_response: dict[str, Any],
) -> dict[str, Any]:
    """Validate Robinhood's review-time quote against Artha's reference price."""
    order_type = str(request.get("type") or "").lower()
    if order_type != "market":
        return {
            "passed": True,
            "status": "SKIPPED",
            "reasons": [],
            "checks": {"applies": False, "order_type": order_type},
        }

    reasons: list[str] = []
    try:
        intent = _order_intent_for_action(action)
        reference = _as_float(getattr(intent, "limit_price", None)) or _as_float(getattr(intent, "estimated_price", None))
    except Exception as exc:
        intent = None
        reference = None
        reasons.append(f"Could not load Artha order reference price: {type(exc).__name__}: {exc}")

    side = str(request.get("side") or getattr(intent, "side", "") or "").lower()
    quote = _review_quote_from_response(review_response)
    execution_price = quote.get("ask") if side == "buy" else quote.get("bid") if side == "sell" else None
    if execution_price is None:
        execution_price = quote.get("last")
    max_drift = Config.ROBINHOOD_MARKET_ORDER_MAX_PRICE_DRIFT_PCT
    checks = {
        "applies": True,
        "order_type": order_type,
        "side": side,
        "reference_price": reference,
        "execution_price": execution_price,
        "bid": quote.get("bid"),
        "ask": quote.get("ask"),
        "last": quote.get("last"),
        "maximum_drift_pct": max_drift,
        "quote_disclosure": quote.get("disclosure"),
    }
    if reference is None or reference <= 0:
        reasons.append("Artha reference price is required before recording a market/notional Robinhood review.")
    if execution_price is None or execution_price <= 0:
        reasons.append("Robinhood review-time quote is required before recording a market/notional review.")
    if not reasons:
        if side == "buy" and execution_price > reference * (1 + max_drift):
            reasons.append(
                f"Robinhood review quote USD {execution_price:.2f} is above Artha reference USD {reference:.2f} "
                f"by more than {max_drift:.2%}; re-review before buying."
            )
        elif side == "sell" and execution_price < reference * (1 - max_drift):
            reasons.append(
                f"Robinhood review quote USD {execution_price:.2f} is below Artha reference USD {reference:.2f} "
                f"by more than {max_drift:.2%}; re-review before selling."
            )
    return {
        "passed": not reasons,
        "status": "PASS" if not reasons else "BLOCKED",
        "reasons": reasons,
        "checks": checks,
    }


def _evaluate_tradability_for_request(request: dict[str, Any], tradability_response: dict[str, Any] | None) -> dict[str, Any]:
    if not tradability_response:
        return {
            "passed": False,
            "status": "BLOCKED",
            "reasons": ["Robinhood tradability response is required before review/place."],
            "checks": {"provided": False},
        }
    data = _tool_data(tradability_response)
    results = data.get("results") if isinstance(data, dict) else []
    ticker = str(request.get("symbol") or "").upper()
    row = None
    for item in results or []:
        if isinstance(item, dict) and str(item.get("symbol") or "").upper() == ticker:
            row = item
            break
    reasons: list[str] = []
    checks: dict[str, Any] = {
        "provided": True,
        "symbol": ticker,
        "found": bool(row),
        "requires_fractional": _request_needs_fractional_tradability(request),
    }
    if not row:
        not_found = data.get("not_found") if isinstance(data, dict) else None
        reasons.append(f"Robinhood tradability did not resolve {ticker}.")
        checks["not_found"] = not_found
    else:
        checks.update(
            {
                "tradeable": bool(row.get("tradeable")),
                "state": str(row.get("state") or ""),
                "fractional_tradability": str(row.get("fractional_tradability") or ""),
                "all_day_tradability": str(row.get("all_day_tradability") or ""),
                "internal_halt_sessions": row.get("internal_halt_sessions"),
            }
        )
        if not bool(row.get("tradeable")):
            reasons.append(f"{ticker} is not tradeable on Robinhood.")
        if str(row.get("state") or "").lower() != "active":
            reasons.append(f"{ticker} Robinhood instrument state is {row.get('state') or 'unknown'}.")
        halt_sessions = row.get("internal_halt_sessions") or []
        if "regular_hours" in halt_sessions:
            reasons.append(f"{ticker} has a Robinhood halt overlapping regular hours.")
        if _request_needs_fractional_tradability(request) and str(row.get("fractional_tradability") or "").lower() != "tradable":
            reasons.append(f"{ticker} is not fractional-tradable for this account.")
    return {
        "passed": not reasons,
        "status": "PASS" if not reasons else "BLOCKED",
        "reasons": reasons,
        "checks": checks,
    }


def _review_is_fresh(row: dict[str, Any]) -> bool:
    updated = _parse_time(row.get("updated_at"))
    if not updated:
        return False
    max_age = timedelta(minutes=max(1, int(Config.ROBINHOOD_REVIEW_MAX_AGE_MINUTES)))
    return _utcnow() - updated <= max_age


def _action_decision_fresh_gate(row: dict[str, Any], journal: DecisionJournal) -> dict[str, Any]:
    max_age_minutes = max(1, int(Config.ROBINHOOD_REVIEW_DECISION_MAX_AGE_MINUTES))
    order_row = None
    if row.get("execution_order_row"):
        order_row = journal.get_execution_order_by_id(int(row["execution_order_row"]))
    if order_row is None and row.get("order_intent_id"):
        order_row = journal.get_execution_order_by_intent_id(str(row["order_intent_id"]))
    created_at = _parse_time((order_row or {}).get("created_at") or row.get("created_at"))
    if not created_at:
        return {
            "passed": False,
            "status": "BLOCKED",
            "reasons": ["Trade action is missing its underlying decision timestamp; regenerate the review."],
            "checks": {"max_age_minutes": max_age_minutes},
        }
    age_minutes = (_utcnow() - created_at).total_seconds() / 60.0
    passed = age_minutes <= max_age_minutes
    return {
        "passed": passed,
        "status": "PASS" if passed else "BLOCKED",
        "reasons": [] if passed else [f"Underlying Artha order proposal is {age_minutes:.1f} minutes old; regenerate before Robinhood review."],
        "checks": {
            "created_at": created_at.isoformat(),
            "age_minutes": round(age_minutes, 2),
            "max_age_minutes": max_age_minutes,
        },
    }


def _broker_snapshot_gate_for_action(row: dict[str, Any]) -> dict[str, Any]:
    intent = _order_intent_for_action(row)
    snapshot = load_robinhood_snapshot()
    return evaluate_broker_snapshot_guardrails(
        intent,
        snapshot,
        _resolved_notional_for_intent(intent),
    )


def _stored_review_gate(row: dict[str, Any]) -> dict[str, Any]:
    if str(row.get("status") or "") not in {"review_clear", "reviewed"}:
        return {
            "passed": False,
            "reasons": ["A clear Robinhood review must be recorded before placement."],
            "checks": {"status": row.get("status")},
        }
    if not _review_is_fresh(row):
        return {
            "passed": False,
            "reasons": ["Recorded Robinhood review is stale; re-review before placement."],
            "checks": {"updated_at": row.get("updated_at"), "max_age_minutes": Config.ROBINHOOD_REVIEW_MAX_AGE_MINUTES},
        }
    result = _as_json(row.get("result_json"))
    review_gate = result.get("review_gate") if isinstance(result.get("review_gate"), dict) else {}
    if not review_gate:
        return {
            "passed": False,
            "reasons": ["Stored Robinhood review gate is missing; re-review before placement."],
            "checks": {"status": row.get("status"), "updated_at": row.get("updated_at")},
        }
    if not review_gate.get("passed"):
        return {
            "passed": False,
            "reasons": list(review_gate.get("reasons") or ["Stored review gate failed."]),
            "checks": review_gate.get("checks") or {},
        }
    return {"passed": True, "reasons": [], "checks": {"status": row.get("status"), "updated_at": row.get("updated_at")}}


def _execution_officer_payload_gate(row: dict[str, Any]) -> dict[str, Any]:
    payload = _as_json(row.get("payload_json"))
    intent = payload.get("intent") if isinstance(payload.get("intent"), dict) else {}
    evidence = intent.get("evidence") if isinstance(intent.get("evidence"), dict) else {}
    officer = evidence.get("execution_officer") if isinstance(evidence.get("execution_officer"), dict) else {}
    reasons: list[str] = []
    if not officer:
        reasons.append("Auto-buy action is missing Execution Officer evidence.")
    elif not bool(officer.get("auto_buy_eligible")):
        reasons.append("Execution Officer did not mark this action auto-buy eligible.")
    verdict = str(officer.get("execution_verdict") or "").upper()
    strategy = str(officer.get("strategy") or "").upper()
    if officer and verdict != "BUY_READY":
        reasons.append(f"Execution Officer verdict is {verdict or 'UNKNOWN'}, not BUY_READY.")
    if officer and strategy not in {"WHOLE_SHARE_LIMIT", "FRACTIONAL_MARKET"}:
        reasons.append(f"Execution Officer strategy {strategy or 'UNKNOWN'} is not placeable.")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "checks": {
            "execution_verdict": officer.get("execution_verdict"),
            "strategy": officer.get("strategy"),
            "selected_candidate_id": officer.get("selected_candidate_id"),
            "model": officer.get("officer_model"),
            "reasoning_effort": officer.get("officer_reasoning_effort"),
            "temperature": officer.get("officer_temperature"),
        },
    }


def _submitted_buy_notional_today(journal: DecisionJournal) -> float:
    today = _utcnow().date().isoformat()
    with journal._connect() as conn:
        rows = conn.execute(
            """
            SELECT notional, quantity, limit_price, estimated_price
            FROM execution_orders
            WHERE date(created_at) = date(?)
              AND side = 'buy'
              AND status IN ('submitted', 'filled', 'partially_filled')
            """,
            (today,),
        ).fetchall()
    total = 0.0
    for row in rows:
        notional = _as_float(row["notional"])
        if notional is None:
            quantity = _as_float(row["quantity"])
            price = _as_float(row["limit_price"]) or _as_float(row["estimated_price"])
            if quantity is not None and price is not None:
                notional = quantity * price
        if notional is not None:
            total += notional
    return total


def _auto_buy_authorization_gate(row: dict[str, Any], journal: DecisionJournal) -> dict[str, Any]:
    reasons: list[str] = []
    checks: dict[str, Any] = {
        "auto_buy_enabled": Config.ROBINHOOD_AUTO_BUY_ENABLED,
        "review_only": Config.ROBINHOOD_REVIEW_ONLY,
        "dry_run_only": Config.ROBINHOOD_DRY_RUN_ONLY,
        "agentic_enabled": Config.ROBINHOOD_AGENTIC_ENABLED,
        "kill_switch": Config.ROBINHOOD_KILL_SWITCH,
    }
    if not Config.ROBINHOOD_AUTO_BUY_ENABLED:
        reasons.append("Robinhood auto-buy is disabled.")
    if str(row.get("action_type") or "").lower() != "auto_buy":
        reasons.append("Trade action is not an auto-buy action.")
    if str(row.get("side") or "").lower() != "buy":
        reasons.append("Only buy actions can use auto-buy.")
    if str(row.get("status") or "").lower() in {"blocked", "skipped", "expired"}:
        reasons.append(f"Trade action status is {row.get('status')}; auto-buy is not allowed.")
    if _trade_action_expired(row):
        reasons.append("Auto-buy action token expired.")
    control = get_trading_control()
    checks["runtime_control"] = control
    if control.get("trading_disabled"):
        reasons.append(f"Trading is disabled: {control.get('reason') or 'runtime control'}")
    if Config.ROBINHOOD_REVIEW_ONLY or Config.ROBINHOOD_DRY_RUN_ONLY or not Config.ROBINHOOD_AGENTIC_ENABLED or Config.ROBINHOOD_KILL_SWITCH:
        reasons.append("Live placement is blocked by Artha config: review-only/dry-run/agentic-disabled/kill-switch safety cage.")

    decision_gate = _action_decision_fresh_gate(row, journal)
    checks["decision_freshness"] = decision_gate
    if not decision_gate.get("passed"):
        reasons.extend(str(reason) for reason in decision_gate.get("reasons") or [])

    officer_gate = _execution_officer_payload_gate(row)
    checks["execution_officer"] = officer_gate
    if not officer_gate.get("passed"):
        reasons.extend(str(reason) for reason in officer_gate.get("reasons") or [])

    try:
        intent = _order_intent_for_action(row)
        notional = _resolved_notional_for_intent(intent)
    except Exception as exc:
        intent = None
        notional = None
        reasons.append(f"Could not reconstruct order intent: {type(exc).__name__}: {exc}")
    checks["notional"] = notional
    if notional is None or notional <= 0:
        reasons.append("Auto-buy notional could not be resolved.")
    elif notional > Config.ROBINHOOD_AUTO_BUY_MAX_ORDER_DOLLARS:
        reasons.append(
            f"Auto-buy order ${notional:.2f} exceeds ${Config.ROBINHOOD_AUTO_BUY_MAX_ORDER_DOLLARS:.2f} max order cap."
        )
    daily_total = _submitted_buy_notional_today(journal)
    checks["daily_auto_buy"] = {
        "submitted_buy_notional_today": round(daily_total, 2),
        "max_daily_dollars": Config.ROBINHOOD_AUTO_BUY_MAX_DAILY_DOLLARS,
    }
    if notional is not None and daily_total + notional > Config.ROBINHOOD_AUTO_BUY_MAX_DAILY_DOLLARS:
        reasons.append(
            f"Auto-buy daily cap would be exceeded: ${daily_total + notional:.2f} > "
            f"${Config.ROBINHOOD_AUTO_BUY_MAX_DAILY_DOLLARS:.2f}."
        )

    if Config.ROBINHOOD_REQUIRE_FRESH_SNAPSHOT_FOR_REVIEW:
        try:
            broker_check = _broker_snapshot_gate_for_action(row)
        except Exception as exc:
            broker_check = {
                "passed": False,
                "status": "BLOCKED",
                "reasons": [f"Could not validate latest Robinhood snapshot before auto-buy: {type(exc).__name__}: {exc}"],
                "checks": {},
            }
        checks["broker_snapshot"] = broker_check
        if not broker_check.get("passed"):
            reasons.extend(str(reason) for reason in broker_check.get("reasons") or [])

    return {
        "passed": not reasons,
        "status": "PASS" if not reasons else "BLOCKED",
        "reasons": reasons,
        "checks": checks,
    }


def build_auto_buy_operation(action_id: str, *, journal: DecisionJournal | None = None) -> dict[str, Any]:
    """Resolve a queued auto-buy action into the OpenClaw MCP review/place flow."""
    journal = journal or DecisionJournal()
    row = journal.get_trade_action(action_id)
    if not row:
        return {"success": False, "operation": "blocked", "message": "Auto-buy action was not found."}
    row_status = str(row.get("status") or "").lower()
    if row_status not in {"review_ready", "review_requested", "auto_review_requested", "review_clear", "reviewed"}:
        return {
            "success": False,
            "operation": "blocked",
            "message": f"Auto-buy action status is {row_status or 'unknown'}; unattended runner only processes fresh review-ready or review-clear actions.",
            "action_id": row.get("action_id"),
            "status": row_status,
        }
    gate = _auto_buy_authorization_gate(row, journal)
    if not gate.get("passed"):
        journal.update_trade_action(
            str(row["action_id"]),
            {"status": "blocked", "result_json": {"auto_buy_gate": gate}, "notes": "Auto-buy blocked by authorization gate."},
        )
        return {
            "success": False,
            "operation": "blocked",
            "message": "; ".join(str(reason) for reason in gate.get("reasons") or []),
            "auto_buy_gate": gate,
            "snapshot_refresh_operation": build_snapshot_refresh_operation()
            if any("snapshot" in str(reason).lower() for reason in gate.get("reasons") or [])
            else None,
        }

    if str(row.get("status") or "").lower() == "review_clear":
        callback_data = _callback("place", str(row["action_id"]), str(row.get("token_place") or ""))
        return build_action_operation(callback_data, journal=journal)

    request = _request_json_for_action(row, journal)
    journal.update_trade_action(
        str(row["action_id"]),
        {"status": "auto_review_requested", "result_json": {"auto_buy_gate": gate}, "notes": "Auto-buy requested Robinhood review."},
    )
    return {
        "success": True,
        "operation": "auto_tradability_review_then_place_equity_order",
        "tradability_mcp_args": _tradability_mcp_args(request),
        "review_mcp_args": request,
        "action_id": row["action_id"],
        "auto_buy_gate": gate,
        "message": (
            "Auto-buy flow: call get_equity_tradability and review_equity_order. "
            "Record the review, then Artha will rebuild the place operation only if Robinhood review is still clear."
        ),
    }


def build_pending_auto_buy_operations(*, journal: DecisionJournal | None = None, limit: int = 5) -> dict[str, Any]:
    """Return OpenClaw operations for queued auto-buy actions."""
    journal = journal or DecisionJournal()
    operations = []
    skipped = []
    for row in journal.get_trade_actions(limit=max(1, int(limit) * 4)):
        if len(operations) >= limit:
            break
        if str(row.get("action_type") or "").lower() != "auto_buy":
            continue
        if str(row.get("status") or "").lower() not in {"review_ready", "review_clear"}:
            skipped.append({"action_id": row.get("action_id"), "ticker": row.get("ticker"), "status": row.get("status")})
            continue
        operations.append(build_auto_buy_operation(str(row["action_id"]), journal=journal))
    return {"success": True, "operation_count": len(operations), "operations": operations, "skipped": skipped}


def record_action_review(
    action_id: str,
    review_response: dict[str, Any],
    *,
    tradability_response: dict[str, Any] | None = None,
    journal: DecisionJournal | None = None,
) -> dict[str, Any]:
    """Record a Robinhood review preview and produce a Place button only when clear."""
    journal = journal or DecisionJournal()
    row = journal.get_trade_action(action_id)
    if not row:
        return {"success": False, "status": "FAIL", "message": "Trade action was not found."}
    request = _request_json_for_action(row, journal)
    tradability_gate = _evaluate_tradability_for_request(request, tradability_response)
    order_checks = _review_order_checks(review_response)
    order_check_classification = classify_robinhood_order_checks(order_checks)
    echo_ok, echo_reasons = _review_echo_matches(request, review_response)
    drift_gate = _review_price_drift_gate(row, request, review_response)
    reasons = list(tradability_gate.get("reasons") or []) + echo_reasons + list(drift_gate.get("reasons") or [])
    if order_check_classification.get("blocking"):
        reasons.extend(order_check_classification.get("blocking_reasons") or [])
    review_gate = {
        "passed": not reasons,
        "status": "PASS" if not reasons else "BLOCKED",
        "reasons": reasons,
        "checks": {
            "order_checks_empty": not bool(order_checks),
            "order_checks_classification": order_check_classification,
            "echo_matches": echo_ok,
            "tradability": tradability_gate,
            "review_price_drift": drift_gate,
        },
    }
    status = "review_clear" if review_gate["passed"] else "review_blocked"
    result_json = {
        "review_gate": review_gate,
        "review_response": review_response,
        "tradability_response": tradability_response,
        "review_request": request,
    }
    journal.update_trade_action(
        action_id,
        {
            "status": status,
            "result_json": result_json,
            "notes": "Robinhood review recorded; clear for place confirmation." if review_gate["passed"] else "Robinhood review recorded but blocked.",
        },
    )
    updated = journal.get_trade_action(action_id) or {**row, "status": status, "result_json": json.dumps(result_json)}
    message = build_review_result_notice(updated, review_gate, order_checks, review_response)
    return {
        "success": bool(review_gate["passed"]),
        "status": status,
        "action_id": action_id,
        "message": message,
        "reply_markup": build_review_confirmation_reply_markup(updated) if review_gate["passed"] else build_trade_action_reply_markup(updated),
        "review_gate": review_gate,
    }


def run_final_clearance_for_action(
    action_id: str,
    *,
    journal: DecisionJournal | None = None,
) -> dict[str, Any]:
    """Run the Execution Officer's final yes/no gate from the stored review."""
    journal = journal or DecisionJournal()
    action = journal.get_trade_action(action_id)
    if not action:
        return {"success": False, "status": "FAIL", "allow_place": False, "message": "Trade action was not found."}
    result_json = _as_json(action.get("result_json"))
    review_response = result_json.get("review_response") if isinstance(result_json.get("review_response"), dict) else {}
    tradability_response = (
        result_json.get("tradability_response")
        if isinstance(result_json.get("tradability_response"), dict)
        else None
    )
    review_gate = result_json.get("review_gate") if isinstance(result_json.get("review_gate"), dict) else {}
    recorded_review = {
        "success": bool(review_gate.get("passed")),
        "status": "review_clear" if review_gate.get("passed") else "review_blocked",
        "action_id": action_id,
        "review_gate": review_gate,
    }
    if not review_response or not review_gate:
        clearance = {
            "allow_place": False,
            "status": "BLOCKED",
            "reason": "Stored Robinhood review response or review_gate is missing.",
            "officer_used": False,
        }
    else:
        from .execution_officer import robinhood_review_final_clearance

        clearance = robinhood_review_final_clearance(
            action=action,
            review_response=review_response,
            tradability_response=tradability_response,
            recorded_review=recorded_review,
        )
    merged = dict(result_json)
    merged["execution_officer_final_clearance"] = clearance
    if clearance.get("allow_place"):
        journal.update_trade_action(
            action_id,
            {
                "result_json": merged,
                "notes": "Execution Officer final clearance passed.",
            },
        )
    else:
        journal.update_trade_action(
            action_id,
            {
                "status": "review_blocked",
                "result_json": merged,
                "notes": "Execution Officer final clearance blocked placement.",
            },
        )
    return {
        "success": bool(clearance.get("allow_place")),
        "status": "PASS" if clearance.get("allow_place") else "BLOCKED",
        "allow_place": bool(clearance.get("allow_place")),
        "action_id": action_id,
        "clearance": clearance,
        "message": str(clearance.get("reason") or ""),
    }


def build_review_result_notice(
    action: dict[str, Any],
    review_gate: dict[str, Any],
    order_checks: dict[str, Any],
    review_response: dict[str, Any],
) -> str:
    data = _tool_data(review_response)
    ticker = str(action.get("ticker") or data.get("symbol") or "?").upper()
    side = str(action.get("side") or data.get("side") or "?").upper()
    lines = [
        f"ARTHA ROBINHOOD REVIEW RESULT - {ticker}",
        "--------------------------------",
        f"Side: {side}",
        f"Type: {data.get('type') or '?'}",
        f"Quantity: {data.get('quantity') or '-'}",
        f"Dollar amount: {data.get('dollar_amount') or '-'}",
        f"Limit: {data.get('limit_price') or '-'}",
        f"Review gate: {review_gate.get('status')}",
    ]
    disclosure = data.get("market_data_disclosure")
    if disclosure:
        lines.extend(["", "Robinhood quote disclosure:", str(disclosure)])
    if order_checks:
        lines.extend(["", "Robinhood order checks:"])
        lines.append(json.dumps(order_checks, ensure_ascii=True, sort_keys=True)[:1200])
    reasons = review_gate.get("reasons") or []
    if reasons:
        lines.extend(["", "Blocked reasons:"])
        lines.extend(f"- {reason}" for reason in reasons[:6])
    else:
        lines.extend(["", "Review is clear. Place only if the configured execution policy allows it."])
    return "\n".join(lines)


def build_action_operation(callback_data: str, *, journal: DecisionJournal | None = None) -> dict[str, Any]:
    """Resolve a Telegram callback token into the next OpenClaw MCP operation."""
    journal = journal or DecisionJournal()
    parsed = parse_callback_data(callback_data)
    row = journal.get_trade_action(parsed["action_id"], token=parsed["token"])
    if not row:
        return {"success": False, "message": "Trade action token was not found."}
    if row.get(f"token_{parsed['verb']}") != parsed["token"]:
        return {"success": False, "message": "Trade action token does not match the requested action."}
    if _trade_action_expired(row):
        journal.update_trade_action(row["action_id"], {"status": "expired", "notes": "Telegram action token expired."})
        return {"success": False, "message": "Trade action token expired."}
    if parsed["verb"] in {"review", "place"} and str(row.get("status") or "") in {"blocked", "skipped", "expired"}:
        return {"success": False, "operation": "blocked", "message": f"Trade action is {row.get('status')}; broker review/place is not allowed."}

    if parsed["verb"] == "skip":
        journal.update_trade_action(row["action_id"], {"status": "skipped", "notes": "Skipped by Telegram action."})
        return {"success": True, "operation": "skip", "message": f"Skipped {row.get('ticker')} action {row.get('action_id')}."}

    request = _request_json_for_action(row, journal)
    if parsed["verb"] == "review":
        decision_gate = _action_decision_fresh_gate(row, journal)
        if not decision_gate.get("passed"):
            journal.update_trade_action(row["action_id"], {"status": "expired", "result_json": {"decision_freshness": decision_gate}, "notes": "Review blocked because the Artha proposal is stale."})
            return {
                "success": False,
                "operation": "blocked",
                "message": "; ".join(str(reason) for reason in decision_gate.get("reasons") or []),
                "decision_freshness": decision_gate,
            }
        if Config.ROBINHOOD_REQUIRE_FRESH_SNAPSHOT_FOR_REVIEW:
            try:
                broker_check = _broker_snapshot_gate_for_action(row)
            except Exception as exc:
                broker_check = {
                    "passed": False,
                    "status": "BLOCKED",
                    "reasons": [f"Could not validate latest Robinhood snapshot before review: {type(exc).__name__}: {exc}"],
                    "checks": {},
                }
            if not broker_check.get("passed"):
                journal.update_trade_action(row["action_id"], {"result_json": {"review_snapshot_guardrails": broker_check}, "notes": "Review blocked until a fresh Robinhood snapshot is available."})
                reasons = "; ".join(str(reason) for reason in (broker_check.get("reasons") or [])[:5])
                return {
                    "success": False,
                    "operation": "blocked",
                    "message": f"Fresh Robinhood snapshot is required before review: {reasons or 'unknown snapshot failure'}",
                    "broker_snapshot_guardrails": broker_check,
                    "snapshot_refresh_operation": build_snapshot_refresh_operation(),
                }
        journal.update_trade_action(row["action_id"], {"status": "review_requested", "notes": "Robinhood review requested by Telegram action."})
        return {
            "success": True,
            "operation": "tradability_then_review_equity_order",
            "tradability_mcp_args": _tradability_mcp_args(request),
            "review_mcp_args": request,
            "action_id": row["action_id"],
            "message": "Call get_equity_tradability first, then review_equity_order. Record both with robinhood-record-review. Do not place an order from this operation.",
        }

    if parsed["verb"] == "place":
        control = get_trading_control()
        if control.get("trading_disabled"):
            return {"success": False, "operation": "blocked", "message": f"Trading is disabled: {control.get('reason') or 'kill switch'}"}
        if Config.ROBINHOOD_REVIEW_ONLY or Config.ROBINHOOD_DRY_RUN_ONLY or not Config.ROBINHOOD_AGENTIC_ENABLED:
            return {
                "success": False,
                "operation": "blocked",
                "message": "Live placement is blocked by Artha config: review-only/dry-run/agentic-disabled safety cage.",
            }
        review_gate = _stored_review_gate(row)
        if not review_gate.get("passed"):
            return {
                "success": False,
                "operation": "blocked",
                "message": "; ".join(str(reason) for reason in review_gate.get("reasons") or []),
                "review_gate": review_gate,
            }
        try:
            intent = _order_intent_for_action(row)
            snapshot = load_robinhood_snapshot()
            broker_check = evaluate_broker_snapshot_guardrails(
                intent,
                snapshot,
                _resolved_notional_for_intent(intent),
            )
        except Exception as exc:
            broker_check = {
                "passed": False,
                "status": "BLOCKED",
                "reasons": [f"Could not validate latest Robinhood snapshot before placement: {type(exc).__name__}: {exc}"],
                "checks": {},
            }
        if not broker_check.get("passed"):
            journal.update_trade_action(
                row["action_id"],
                {
                    "status": "blocked",
                    "result_json": {"broker_snapshot_guardrails": broker_check},
                    "notes": "Live place blocked by latest Robinhood snapshot guardrails.",
                },
            )
            reasons = "; ".join(str(reason) for reason in (broker_check.get("reasons") or [])[:5])
            return {
                "success": False,
                "operation": "blocked",
                "message": f"Broker snapshot guardrails blocked placement: {reasons or 'unknown snapshot failure'}",
                "broker_snapshot_guardrails": broker_check,
                "snapshot_refresh_operation": build_snapshot_refresh_operation(),
            }
        if str(request.get("type") or "").lower() == "market":
            try:
                from .scheduler import MarketHours

                market_open = MarketHours().is_market_open()
            except Exception:
                market_open = False
            if not market_open:
                return {
                    "success": False,
                    "operation": "blocked",
                    "message": "Fractional/dollar market orders can only be placed during regular market hours.",
                    "review_gate": review_gate,
                    "checks": {"market_open": market_open, "order_type": request.get("type")},
                }
        place_args = dict(request)
        place_args["ref_id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"artha:{row.get('order_intent_id') or row.get('action_id')}"))
        journal.update_trade_action(row["action_id"], {"status": "place_requested", "notes": "Live place requested by Telegram action; OpenClaw must review first."})
        return {
            "success": True,
            "operation": "tradability_then_review_then_place_equity_order",
            "tradability_mcp_args": _tradability_mcp_args(request),
            "review_mcp_args": request,
            "place_mcp_args": place_args,
            "action_id": row["action_id"],
            "message": "Repeat get_equity_tradability and review_equity_order immediately before place. If tradability fails, order_checks is non-empty, or account/market/order details differ, abort. Only then call place_equity_order.",
        }

    return {"success": False, "message": f"Unsupported action verb: {parsed['verb']}"}


def _order_from_tool_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("order"), dict):
        return payload["order"]
    data = _tool_data(payload)
    order = data.get("order") if isinstance(data, dict) else None
    if isinstance(order, dict):
        return order
    return payload if isinstance(payload, dict) else {}


def _execution_status_from_order(order: dict[str, Any]) -> str:
    state = str(order.get("state") or "").lower()
    if state == "filled":
        return "filled"
    if state == "partially_filled":
        return "partially_filled"
    if state in {"cancelled", "canceled", "partially_filled_rest_cancelled", "pending_cancelled"}:
        return "cancelled"
    if state in {"rejected", "failed", "voided", "locate_failed"}:
        return state
    if state in OPEN_ORDER_STATES:
        return "submitted"
    return state or "submitted"


def record_order_submission(
    *,
    action_id: str,
    place_response: dict[str, Any],
    journal: DecisionJournal | None = None,
    portfolio_path: str | Path = PORTFOLIO_FILE,
) -> dict[str, Any]:
    """Record a Robinhood place_equity_order response and apply fills when final."""
    journal = journal or DecisionJournal()
    action = journal.get_trade_action(action_id)
    if not action:
        return {"status": "FAIL", "message": "Trade action was not found."}
    order_intent_id = str(action.get("order_intent_id") or "")
    if not order_intent_id:
        return {"status": "FAIL", "message": "Trade action is missing order_intent_id."}
    existing_execution = journal.get_execution_order_by_intent_id(order_intent_id) or {}
    existing_response = _response_json(existing_execution)
    order = _order_from_tool_payload(place_response)
    if not order:
        return {"status": "FAIL", "message": "place_equity_order response did not include an order."}
    broker_order_id = str(order.get("id") or "")
    status = _execution_status_from_order(order)
    updates = {
        "status": status,
        "broker_order_id": broker_order_id,
        "dry_run": False,
        "response_json": (
            {**existing_response, "latest_place_response": place_response}
            if existing_response.get("artha_fill_applied")
            else place_response
        ),
        "submitted_at": str(order.get("created_at") or _utcnow_iso()),
        "notes": f"Robinhood place response recorded; state={order.get('state')}.",
    }
    state = str(order.get("state") or "").lower()
    if state == "filled":
        updates["filled_at"] = str(order.get("last_transaction_at") or _utcnow_iso())
    if status == "cancelled":
        updates["canceled_at"] = str(order.get("last_transaction_at") or _utcnow_iso())
    journal.update_execution_order(order_intent_id, updates)
    journal.update_trade_action(
        action_id,
        {
            "status": status,
            "result_json": {"place_response": place_response, "broker_order_id": broker_order_id},
            "notes": f"Robinhood place response recorded; state={order.get('state')}.",
        },
    )
    fill_result = None
    if state == "filled":
        fill_result = record_order_fill(
            order_intent_id=order_intent_id,
            fill_payload=order,
            journal=journal,
            portfolio_path=portfolio_path,
        )
    return {
        "status": "PASS",
        "action_id": action_id,
        "order_intent_id": order_intent_id,
        "broker_order_id": broker_order_id,
        "execution_status": status,
        "fill": fill_result,
    }


def sync_orders_to_artha(
    snapshot: dict[str, Any],
    *,
    journal: DecisionJournal | None = None,
    portfolio_path: str | Path = PORTFOLIO_FILE,
) -> dict[str, Any]:
    """Update Artha execution rows from Robinhood order history in a snapshot."""
    journal = journal or DecisionJournal()
    rows = journal.get_execution_orders(limit=500)
    by_broker_id = {
        str(row.get("broker_order_id")): row
        for row in rows
        if row.get("broker_order_id")
    }
    updated: list[dict[str, Any]] = []
    filled: list[dict[str, Any]] = []
    unmatched_agentic: list[dict[str, Any]] = []
    for raw in snapshot.get("orders") or []:
        if not isinstance(raw, dict):
            continue
        broker_order_id = str(raw.get("id") or "")
        if not broker_order_id:
            continue
        row = by_broker_id.get(broker_order_id)
        if not row:
            if str(raw.get("placed_agent") or "").lower() == "agentic":
                unmatched_agentic.append(
                    {
                        "broker_order_id": broker_order_id,
                        "symbol": raw.get("symbol"),
                        "side": raw.get("side"),
                        "state": raw.get("state"),
                    }
                )
            continue
        status = _execution_status_from_order(raw)
        current_status = str(row.get("status") or "")
        if status == "filled" and current_status != "filled":
            fill_result = record_order_fill(
                order_intent_id=str(row.get("order_intent_id") or ""),
                fill_payload=raw,
                journal=journal,
                portfolio_path=portfolio_path,
            )
            filled.append({"broker_order_id": broker_order_id, "symbol": raw.get("symbol"), "fill": fill_result})
        elif status != current_status:
            updates = {
                "status": status,
                "response_json": raw,
                "notes": f"Robinhood order watcher updated state={raw.get('state')}.",
            }
            if status == "cancelled":
                updates["canceled_at"] = str(raw.get("last_transaction_at") or _utcnow_iso())
            journal.update_execution_order(str(row.get("order_intent_id") or ""), updates)
            updated.append({"broker_order_id": broker_order_id, "symbol": raw.get("symbol"), "status": status})
    return {
        "status": "WARN" if unmatched_agentic else "PASS",
        "updated": updated,
        "filled": filled,
        "unmatched_agentic_orders": unmatched_agentic,
    }


def _response_json(row: dict[str, Any]) -> dict[str, Any]:
    payload = _as_json(row.get("response_json"))
    return payload if isinstance(payload, dict) else {}


def _fill_application_state(row: dict[str, Any], broker_order_id: str) -> tuple[bool, float, float | None]:
    response = _response_json(row)
    if not response.get("artha_fill_applied"):
        return False, 0.0, None
    recorded_order_id = str(response.get("artha_fill_broker_order_id") or "")
    if broker_order_id and recorded_order_id and broker_order_id != recorded_order_id:
        return False, 0.0, None
    return True, float(_as_float(response.get("artha_fill_quantity")) or 0.0), _as_float(response.get("artha_fill_average_price"))


def _portfolio_recorded_fill_quantity(
    portfolio: Portfolio,
    *,
    broker_order_id: str,
    ticker: str,
    side: str,
) -> float:
    tx_type = side.upper()
    total = 0.0
    for txn in portfolio.transactions or []:
        if str(txn.get("type") or "").upper() != tx_type:
            continue
        if str(txn.get("ticker") or "").upper() != ticker.upper():
            continue
        txn_order_id = str(txn.get("broker_order_id") or "")
        txn_notes = str(txn.get("notes") or "")
        if broker_order_id and broker_order_id != txn_order_id and broker_order_id not in txn_notes:
            continue
        total += abs(float(_as_float(txn.get("shares")) or 0.0))
    return total


def _append_portfolio_buy_transaction(
    portfolio: Portfolio,
    *,
    ticker: str,
    cumulative_quantity: float,
    avg_price: float,
    broker_order_id: str,
    timestamp: str | None = None,
) -> bool:
    recorded_quantity = _portfolio_recorded_fill_quantity(
        portfolio,
        broker_order_id=broker_order_id,
        ticker=ticker,
        side="buy",
    )
    delta_quantity = cumulative_quantity - recorded_quantity
    if delta_quantity <= 0.000001:
        return False
    total = round(delta_quantity * avg_price, 6)
    portfolio.transactions.append(
        {
            "type": "BUY",
            "ticker": ticker.upper(),
            "shares": round(delta_quantity, 8),
            "price": round(avg_price, 6),
            "total": total,
            "timestamp": timestamp or _utcnow_iso(),
            "broker_order_id": broker_order_id,
            "notes": f"Robinhood order fill {broker_order_id}",
        }
    )
    portfolio.cash_deployed = round(float(portfolio.cash_deployed or 0.0) + total, 6)
    portfolio.last_updated = _utcnow_iso()
    return True


def record_order_fill(
    *,
    order_intent_id: str,
    fill_payload: dict[str, Any],
    journal: DecisionJournal | None = None,
    portfolio_path: str | Path = PORTFOLIO_FILE,
) -> dict[str, Any]:
    """Record a Robinhood filled order and activate/close Artha monitoring."""
    journal = journal or DecisionJournal()
    row = journal.get_execution_order_by_intent_id(order_intent_id)
    if not row:
        return {"status": "FAIL", "message": "Execution order was not found."}
    request = _as_json(row.get("request_json"))
    order = fill_payload.get("order") if isinstance(fill_payload.get("order"), dict) else fill_payload
    ticker = str(order.get("symbol") or request.get("symbol") or row.get("ticker") or "").upper()
    side = str(order.get("side") or request.get("side") or row.get("side") or "").lower()
    quantity = _as_float(order.get("cumulative_quantity") or order.get("quantity") or request.get("quantity"))
    avg_price = _as_float(order.get("average_price") or order.get("price") or request.get("limit_price"))
    broker_order_id = str(order.get("id") or row.get("broker_order_id") or "")
    state = str(order.get("state") or fill_payload.get("state") or "filled").lower()
    if state not in {"filled", "partially_filled"}:
        return {"status": "WARN", "message": f"Order state is {state}; no portfolio change applied."}
    if not ticker or not side or not quantity or quantity <= 0 or not avg_price or avg_price <= 0:
        return {"status": "FAIL", "message": "Fill payload is missing ticker, side, quantity, or average price."}
    already_applied, applied_quantity, applied_avg_price = _fill_application_state(row, broker_order_id)
    quantity_to_apply = quantity
    price_to_apply = avg_price
    if already_applied and applied_quantity >= quantity - 0.000001:
        logger.info(
            "[robinhood_bridge] fill already applied broker_order_id=%s ticker=%s side=%s",
            broker_order_id,
            ticker,
            side,
        )
        if side == "buy":
            portfolio = Portfolio.load(Path(portfolio_path))
            if _append_portfolio_buy_transaction(
                portfolio,
                ticker=ticker,
                cumulative_quantity=quantity,
                avg_price=avg_price,
                broker_order_id=broker_order_id,
                timestamp=str(order.get("last_transaction_at") or order.get("created_at") or _utcnow_iso()),
            ):
                portfolio.save(Path(portfolio_path))
        return {
            "status": "PASS",
            "ticker": ticker,
            "side": side,
            "quantity": quantity,
            "avg_price": avg_price,
            "broker_order_id": broker_order_id,
            "already_recorded": True,
        }
    if already_applied and applied_quantity > 0 and quantity > applied_quantity:
        quantity_to_apply = quantity - applied_quantity
        old_avg = applied_avg_price if applied_avg_price and applied_avg_price > 0 else avg_price
        delta_notional = (quantity * avg_price) - (applied_quantity * old_avg)
        if quantity_to_apply > 0 and delta_notional > 0:
            price_to_apply = delta_notional / quantity_to_apply
        logger.info(
            "[robinhood_bridge] applying incremental fill broker_order_id=%s ticker=%s applied=%.6f cumulative=%.6f delta=%.6f",
            broker_order_id,
            ticker,
            applied_quantity,
            quantity,
            quantity_to_apply,
        )

    portfolio = Portfolio.load(Path(portfolio_path))
    tracker = ThesisTracker(journal)
    if side == "buy":
        thesis = tracker.get_active(ticker) or tracker.get_pending_for_ticker(ticker)
        pos = portfolio.get_position(ticker)
        if thesis and getattr(thesis, "status", "") != "pending" and _position_already_reflects_fill(pos, thesis, row, quantity, avg_price):
            before_sell_fields = _sell_field_tuple(pos)
            _attach_sell_fields(pos, thesis, avg_price)
            if _sell_field_tuple(pos) != before_sell_fields:
                portfolio.last_updated = _utcnow_iso()
                portfolio.save(Path(portfolio_path))
            response_payload = {
                "artha_fill_applied": True,
                "artha_fill_broker_order_id": broker_order_id,
                "artha_fill_quantity": quantity,
                "artha_fill_average_price": avg_price,
                "artha_fill_source": "broker_snapshot_already_reflected",
                "order": order,
                "raw_fill_payload": fill_payload,
            }
            journal.update_execution_order(
                order_intent_id,
                {
                    "status": "filled" if state == "filled" else "partially_filled",
                    "broker_order_id": broker_order_id,
                    "filled_at": _utcnow_iso(),
                    "quantity": quantity,
                    "notional": round(quantity * avg_price, 6),
                    "estimated_price": avg_price,
                    "response_json": response_payload,
                    "notes": "Robinhood fill matched existing broker-synced portfolio position; no duplicate share mutation applied.",
                },
            )
            logger.info(
                "[robinhood_bridge] fill already reflected by broker snapshot broker_order_id=%s ticker=%s quantity=%.6f",
                broker_order_id,
                ticker,
                quantity,
            )
            return {
                "status": "PASS",
                "ticker": ticker,
                "side": side,
                "quantity": quantity,
                "avg_price": avg_price,
                "broker_order_id": broker_order_id,
                "already_recorded": True,
                "source": "broker_snapshot_already_reflected",
            }
        if thesis and getattr(thesis, "status", "") == "pending":
            thesis = tracker.activate_thesis(thesis.thesis_id, avg_price, shares=quantity)
        if not thesis:
            return {"status": "WARN", "message": f"{ticker} filled, but no active/pending thesis exists; portfolio was not changed."}
        pos = portfolio.get_position(ticker)
        if pos:
            old_value = float(pos.shares or 0) * float(pos.avg_cost or 0)
            new_value = quantity_to_apply * price_to_apply
            total_shares = float(pos.shares or 0) + quantity_to_apply
            pos.shares = total_shares
            pos.avg_cost = round((old_value + new_value) / total_shares, 4)
            pos.current_price = avg_price
            pos.market_value = round(total_shares * avg_price, 4)
        else:
            pos = Position(
                ticker=ticker,
                asset_type="stock",
                shares=quantity_to_apply,
                avg_cost=price_to_apply,
                opened_at=_utcnow_iso(),
                current_price=avg_price,
                market_value=round(quantity_to_apply * avg_price, 4),
                notes=f"Robinhood order fill {broker_order_id}",
            )
            portfolio.positions.append(pos)
        _attach_sell_fields(pos, thesis, avg_price)
        _append_portfolio_buy_transaction(
            portfolio,
            ticker=ticker,
            cumulative_quantity=quantity,
            avg_price=avg_price,
            broker_order_id=broker_order_id,
            timestamp=str(order.get("last_transaction_at") or order.get("created_at") or _utcnow_iso()),
        )
    elif side == "sell":
        portfolio.sell_position(ticker, quantity_to_apply, price_to_apply, notes=f"Robinhood order fill {broker_order_id}")
        pos = portfolio.get_position(ticker)
        active = tracker.get_active(ticker)
        if active and not pos:
            tracker.archive_thesis(active.thesis_id, exit_price=avg_price, exit_reason="Robinhood sell order filled.")
    else:
        return {"status": "FAIL", "message": f"Unsupported fill side: {side}"}

    portfolio.last_updated = _utcnow_iso()
    portfolio.save(Path(portfolio_path))
    response_payload = {
        "artha_fill_applied": True,
        "artha_fill_broker_order_id": broker_order_id,
        "artha_fill_quantity": quantity,
        "artha_fill_average_price": avg_price,
        "order": order,
        "raw_fill_payload": fill_payload,
    }
    journal.update_execution_order(
        order_intent_id,
        {
            "status": "filled" if state == "filled" else "partially_filled",
            "broker_order_id": broker_order_id,
            "filled_at": _utcnow_iso(),
            "quantity": quantity,
            "notional": round(quantity * avg_price, 6),
            "estimated_price": avg_price,
            "response_json": response_payload,
            "notes": "Robinhood fill recorded; Artha portfolio/thesis state updated.",
        },
    )
    return {
        "status": "PASS",
        "ticker": ticker,
        "side": side,
        "quantity": quantity,
        "avg_price": avg_price,
        "broker_order_id": broker_order_id,
    }


def _cash_from_snapshot(snapshot: dict[str, Any]) -> float | None:
    portfolio = snapshot.get("portfolio") if isinstance(snapshot.get("portfolio"), dict) else {}
    for key in ("buying_power", "cash_available", "cash", "withdrawable_amount", "cash_held_for_orders"):
        value = _as_float(portfolio.get(key))
        if value is not None:
            return value
    return None


def format_control_center(journal: DecisionJournal | None = None) -> str:
    journal = journal or DecisionJournal()
    snapshot = load_robinhood_snapshot()
    cash = _cash_from_snapshot(snapshot)
    positions = []
    for raw in snapshot.get("positions") or []:
        if isinstance(raw, dict):
            ticker = _broker_position_symbol(raw)
            qty = _broker_position_quantity(raw)
            price = _broker_position_price(raw, _broker_position_avg_cost(raw))
            if ticker and qty > 0:
                positions.append(f"{ticker} {qty:.6f} @ USD {float(price or 0):.2f}")
    orders = [
        row for row in (snapshot.get("orders") or [])
        if isinstance(row, dict) and str(row.get("state") or "").lower() in OPEN_ORDER_STATES
    ]
    pending = journal.get_pending_theses()
    sell_signals = journal.get_active_sell_signals()
    actions = journal.get_trade_actions(limit=10)
    ready_actions = [row for row in actions if str(row.get("status") or "") in {"review_ready", "review_requested", "review_clear"}]
    control = get_trading_control()
    lines = [
        "ARTHA / ROBINHOOD CONTROL CENTER",
        "--------------------------------",
        f"Snapshot: {snapshot.get('status')} | age {snapshot.get('age_minutes')} min | positions {len(positions)}",
        f"Trading disabled: {control.get('trading_disabled')} {('- ' + str(control.get('reason'))) if control.get('reason') else ''}",
        f"Cash/buying power: USD {cash:.2f}" if cash is not None else "Cash/buying power: unknown",
        "",
        "Holdings:",
        *(f"- {line}" for line in positions[:8]),
    ]
    if not positions:
        lines.append("- none")
    lines.extend(["", "Open Robinhood orders:"])
    if orders:
        for order in orders[:8]:
            lines.append(f"- {order.get('symbol') or '?'} {order.get('side') or '?'} {order.get('quantity') or order.get('cumulative_quantity') or '?'} state={order.get('state')}")
    else:
        lines.append("- none")
    lines.extend(["", "Pending Artha buys:"])
    if pending:
        lines.extend(f"- {row.get('ticker')} {row.get('position_type')} expires {(row.get('pending_expiry') or '')[:10]}" for row in pending[:8])
    else:
        lines.append("- none")
    lines.extend(["", "Active sell risks:"])
    if sell_signals:
        lines.extend(f"- {row.get('ticker')} {row.get('severity')} {row.get('signal_type')}" for row in sell_signals[:8])
    else:
        lines.append("- none")
    lines.extend(["", "Recent trade actions:"])
    if actions:
        lines.extend(f"- {row.get('ticker')} {row.get('side')} {row.get('status')} action={row.get('action_id')}" for row in actions[:8])
    else:
        lines.append("- none")
    lines.append(f"Review-ready action count: {len(ready_actions)}")
    warnings = snapshot.get("warnings") or []
    if warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in warnings[:5])
        lines.append("- On-demand refresh available: python run.py robinhood-snapshot-refresh-operation")
    return "\n".join(lines)
