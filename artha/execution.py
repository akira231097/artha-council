"""Robinhood-ready execution guardrails and dry-run order audit.

This module prepares Artha for Robinhood Agentic Trading without placing live
orders. It turns council decisions into auditable order intents, applies hard
execution guardrails, records every result in SQLite, and can notify Telegram.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .config import Config
from .journal import DecisionJournal
from .portfolio import Portfolio
from .scheduler import MarketHours
from .telegram import TelegramSender

logger = logging.getLogger(__name__)

BUY_ACTIONS = {"BUY", "STARTER", "TACTICAL_BUY", "ACCUMULATE", "ADD", "STRONG BUY"}
CRYPTO_LIKE = {"BTC", "ETH", "SOL", "DOGE", "ADA", "XRP", "BTC-USD", "ETH-USD"}
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")
SHARE_EPSILON = 0.000001


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat()


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
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


def _latest_feature_for_ticker(journal: DecisionJournal, ticker: str) -> dict[str, Any] | None:
    wanted = ticker.upper().strip()
    for row in journal.get_decision_features(limit=500):
        if str(row.get("ticker") or "").upper().strip() == wanted:
            return row
    return None


def _latest_supervisor_payload(journal: DecisionJournal) -> dict[str, Any]:
    row = journal.get_latest_supervisor_run()
    if not row:
        return {}
    payload = _as_json(row.get("payload_json"))
    return {
        "row_id": row.get("id"),
        "severity": row.get("severity"),
        "generated_at": row.get("generated_at"),
        "payload": payload,
    }


def _supervisor_buy_gate(supervisor: dict[str, Any], *, review_only: bool) -> dict[str, Any]:
    """Decide whether Supervisor state should block a buy-side order review.

    Live orders require a clean PASS. Review-only/dry-run preparation can proceed
    through non-fatal WARN states so old log noise or pending-buy warnings do not
    prevent the very review that resolves them.
    """
    severity = str(supervisor.get("severity") or "").upper()
    payload = supervisor.get("payload") or {}
    checks = payload.get("checks") or []
    failing_checks = [
        str(check.get("name") or "unknown")
        for check in checks
        if str(check.get("status") or "").upper() == "FAIL"
    ]
    if severity == "PASS":
        return {"allowed": True, "reason": "Supervisor is PASS.", "failing_checks": []}
    if failing_checks:
        return {
            "allowed": False,
            "reason": f"Supervisor has failing check(s): {', '.join(failing_checks[:5])}.",
            "failing_checks": failing_checks,
        }
    if severity == "WARN" and review_only:
        warning_checks = [
            str(check.get("name") or "unknown")
            for check in checks
            if str(check.get("status") or "").upper() == "WARN"
        ]
        return {
            "allowed": True,
            "reason": "Supervisor is WARN, but this is review-only and no failing checks are present.",
            "warning_checks": warning_checks,
            "failing_checks": [],
        }
    return {
        "allowed": False,
        "reason": f"Supervisor is {severity or 'missing'}; live or non-review buys require PASS.",
        "failing_checks": failing_checks,
    }


def mask_account_number(account_number: str) -> str:
    """Mask a Robinhood account number for logs/reports."""
    raw = str(account_number or "")
    if len(raw) <= 4:
        return "****"
    return f"****{raw[-4:]}"


def validate_allowlisted_robinhood_account(
    account: dict[str, Any] | None,
    expected_account_number: str | None = None,
) -> GuardrailResult:
    """Validate that a Robinhood account is exactly the configured Agentic account."""
    expected = str(expected_account_number or Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER or "").strip()
    checks: dict[str, Any] = {
        "expected_account_masked": mask_account_number(expected),
        "configured": bool(expected),
    }
    reasons: list[str] = []
    if not expected:
        reasons.append("Robinhood Agentic account number is not configured.")
    if not account:
        reasons.append("Robinhood account record is missing.")
        return GuardrailResult(False, "BLOCKED", reasons, checks)

    actual = str(account.get("account_number") or account.get("rhs_account_number") or "").strip()
    checks.update(
        {
            "actual_account_masked": mask_account_number(actual),
            "agentic_allowed": bool(account.get("agentic_allowed")),
            "type": str(account.get("type") or "").lower(),
            "nickname": str(account.get("nickname") or ""),
            "state": str(account.get("state") or ""),
            "deactivated": bool(account.get("deactivated")),
            "permanently_deactivated": bool(account.get("permanently_deactivated")),
        }
    )
    if expected and actual != expected:
        reasons.append("Robinhood account number does not match Artha's allowlisted Agentic account.")
    if not bool(account.get("agentic_allowed")):
        reasons.append("Robinhood account is not agentic-enabled.")
    expected_type = str(Config.ROBINHOOD_EXPECTED_ACCOUNT_TYPE or "").lower().strip()
    if expected_type and str(account.get("type") or "").lower().strip() != expected_type:
        reasons.append(f"Robinhood Agentic account must be a {expected_type} account.")
    if str(account.get("state") or "").lower() != "active":
        reasons.append("Robinhood Agentic account is not active.")
    if account.get("deactivated") or account.get("permanently_deactivated"):
        reasons.append("Robinhood Agentic account is deactivated.")
    return GuardrailResult(not reasons, "PASS" if not reasons else "BLOCKED", reasons, checks)


def find_allowlisted_robinhood_account(accounts: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, GuardrailResult]:
    """Find and validate the configured Agentic account from Robinhood get_accounts output."""
    expected = str(Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER or "").strip()
    matched = None
    if expected:
        for account in accounts or []:
            if str(account.get("account_number") or "").strip() == expected:
                matched = account
                break
    validation = validate_allowlisted_robinhood_account(matched, expected)
    return matched, validation


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


def _parse_snapshot_time(value: Any) -> datetime | None:
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
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_robinhood_position_snapshot(
    payload: Any,
    *,
    now: datetime | None = None,
    max_age_minutes: int | None = None,
) -> dict[str, Any]:
    """Normalize a read-only Robinhood MCP snapshot for reconciliation.

    Accepts either a raw positions list, the `get_equity_positions` response,
    or Artha's richer snapshot envelope:

        {
          "generated_at": "...",
          "source": "robinhood_mcp",
          "account": {... get_accounts account ...},
          "portfolio": {... get_portfolio data ...},
          "positions": [... get_equity_positions data.positions ...]
        }

    The function does not mutate Artha state. It only validates freshness and
    shape so the monitor can fail loudly instead of assuming broker sync exists.
    """
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    max_age = int(max_age_minutes if max_age_minutes is not None else Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_MAX_AGE_MINUTES)
    warnings: list[str] = []
    account: dict[str, Any] | None = None
    portfolio: dict[str, Any] | None = None
    generated_at_raw: Any = None
    source = "unknown"

    positions_raw: Any = []
    if isinstance(payload, list):
        positions_raw = payload
        warnings.append("Snapshot is a raw positions list without account, portfolio, or generated_at metadata.")
    elif isinstance(payload, dict):
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        positions_raw = (
            payload.get("positions")
            or payload.get("results")
            or data.get("positions")
            or data.get("results")
            or []
        )
        account_candidate = (
            payload.get("account")
            or payload.get("selected_account")
            or data.get("account")
            or data.get("selected_account")
        )
        if isinstance(account_candidate, dict):
            account = account_candidate
        portfolio_candidate = payload.get("portfolio") or data.get("portfolio")
        if isinstance(portfolio_candidate, dict):
            portfolio = portfolio_candidate
        generated_at_raw = (
            payload.get("generated_at")
            or payload.get("generated_at_utc")
            or payload.get("synced_at")
            or payload.get("as_of")
            or data.get("generated_at")
            or data.get("synced_at")
        )
        source = str(payload.get("source") or data.get("source") or source)
    else:
        warnings.append(f"Unsupported snapshot payload type: {type(payload).__name__}.")

    positions = [row for row in positions_raw if isinstance(row, dict)] if isinstance(positions_raw, list) else []
    if positions_raw and not isinstance(positions_raw, list):
        warnings.append("Snapshot positions payload is not a list.")

    generated_at = _parse_snapshot_time(generated_at_raw)
    age_minutes: float | None = None
    if generated_at is None:
        warnings.append("Snapshot is missing generated_at/synced_at metadata.")
    else:
        age_minutes = max(0.0, (now_utc - generated_at).total_seconds() / 60.0)
        if age_minutes > max_age:
            warnings.append(f"Snapshot is stale: {age_minutes:.1f} minutes old; max allowed is {max_age} minutes.")

    account_check = validate_allowlisted_robinhood_account(account) if account else None
    if account_check and not account_check.passed:
        warnings.extend(account_check.reasons)
    elif account is None:
        warnings.append("Snapshot does not include a Robinhood account record for allowlist verification.")

    status = "PASS" if not warnings else "WARN"
    return {
        "status": status,
        "source": source,
        "generated_at": generated_at.isoformat() if generated_at else None,
        "age_minutes": round(age_minutes, 2) if age_minutes is not None else None,
        "max_age_minutes": max_age,
        "fresh": generated_at is not None and (age_minutes is None or age_minutes <= max_age),
        "account": account,
        "account_check": account_check.to_dict() if account_check else None,
        "portfolio": portfolio,
        "positions": positions,
        "position_count": len(positions),
        "warnings": warnings,
    }


def reconcile_robinhood_positions(
    broker_positions: list[dict[str, Any]],
    portfolio: Portfolio | None = None,
    journal: DecisionJournal | None = None,
    account: dict[str, Any] | None = None,
    tolerance: float = 0.0001,
) -> dict[str, Any]:
    """Compare Robinhood holdings with Artha's portfolio/thesis state.

    This is intentionally read-only. It lets the monitor/Supervisor detect when
    a manual Robinhood fill was not recorded in Artha, or when Artha still thinks
    it holds a position that Robinhood no longer holds.
    """
    portfolio = portfolio or Portfolio.load()
    journal = journal or DecisionJournal()
    account_check = validate_allowlisted_robinhood_account(account) if account else None

    broker_by_ticker: dict[str, dict[str, Any]] = {}
    for raw in broker_positions or []:
        if not isinstance(raw, dict):
            continue
        ticker = _broker_position_symbol(raw)
        quantity = _broker_position_quantity(raw)
        if not ticker or quantity <= tolerance:
            continue
        broker_by_ticker[ticker] = {
            "ticker": ticker,
            "quantity": quantity,
            "raw": raw,
        }

    artha_by_ticker = {
        str(pos.ticker or "").upper().strip(): pos
        for pos in portfolio.positions
        if str(pos.ticker or "").strip()
    }
    active_theses = {
        str(row.get("ticker") or "").upper().strip(): row
        for row in journal.get_all_active_theses()
    }

    broker_only: list[dict[str, Any]] = []
    artha_only: list[dict[str, Any]] = []
    quantity_mismatches: list[dict[str, Any]] = []
    unmonitored_positions: list[dict[str, Any]] = []

    for ticker, broker_pos in broker_by_ticker.items():
        artha_pos = artha_by_ticker.get(ticker)
        if artha_pos is None:
            broker_only.append({"ticker": ticker, "broker_quantity": broker_pos["quantity"]})
            continue
        artha_qty = float(artha_pos.shares or 0)
        if abs(artha_qty - broker_pos["quantity"]) > tolerance:
            quantity_mismatches.append(
                {"ticker": ticker, "artha_quantity": artha_qty, "broker_quantity": broker_pos["quantity"]}
            )
        if ticker not in active_theses and not getattr(artha_pos, "thesis_id", None):
            unmonitored_positions.append({"ticker": ticker, "broker_quantity": broker_pos["quantity"]})

    for ticker, artha_pos in artha_by_ticker.items():
        if ticker not in broker_by_ticker:
            artha_only.append({"ticker": ticker, "artha_quantity": float(artha_pos.shares or 0)})

    issues = broker_only + artha_only + quantity_mismatches + unmonitored_positions
    return {
        "status": "PASS" if not issues and (account_check is None or account_check.passed) else "WARN",
        "account_check": account_check.to_dict() if account_check else None,
        "broker_position_count": len(broker_by_ticker),
        "artha_position_count": len(artha_by_ticker),
        "broker_only": broker_only,
        "artha_only": artha_only,
        "quantity_mismatches": quantity_mismatches,
        "unmonitored_positions": unmonitored_positions,
    }


def _format_decimal(value: float, places: int = 2) -> str:
    return f"{float(value):.{places}f}"


def _format_quantity(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _review_quantity(intent: OrderIntent) -> float:
    if intent.quantity is not None:
        quantity = float(intent.quantity)
    else:
        if not intent.notional or not intent.limit_price:
            raise ValueError("Limit-order review needs quantity or notional plus limit price.")
        quantity = float(intent.notional) / float(intent.limit_price)
    if quantity <= 0:
        raise ValueError("Limit-order review needs a positive share quantity.")
    return quantity


def _is_whole_share(value: float | None) -> bool:
    if value is None or value <= 0:
        return False
    return abs(float(value) - round(float(value))) < SHARE_EPSILON


def _intent_resolved_quantity(intent: "OrderIntent") -> float | None:
    quantity = _as_float(getattr(intent, "quantity", None))
    if quantity is not None:
        return quantity
    notional = _as_float(getattr(intent, "notional", None))
    price = _as_float(getattr(intent, "limit_price", None)) or _as_float(getattr(intent, "estimated_price", None))
    if notional is not None and price and price > 0:
        return notional / price
    return None


def _requires_market_fractional_order(intent: "OrderIntent") -> bool:
    """Robinhood MCP only supports fractional/dollar-based equities as market/regular-hours."""
    quantity = _intent_resolved_quantity(intent)
    if quantity is not None and quantity > 0 and not _is_whole_share(quantity):
        return True
    if intent.side == "buy" and intent.notional is not None and quantity is None:
        return True
    return False


def resolve_robinhood_order_request(
    intent: "OrderIntent",
    account_number: str | None = None,
) -> dict[str, Any]:
    """Resolve Artha's price-controlled intent into valid Robinhood MCP args.

    Whole-share orders use limit orders. Fractional or dollar-based orders use
    market + regular_hours because Robinhood MCP rejects fractional limit orders.
    Artha still keeps the intended limit/reference price for guardrail drift checks.
    """
    intent = intent.normalized()
    account = str(account_number or Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER or "").strip()
    if not account:
        raise ValueError("Robinhood Agentic account number is not configured.")
    tif = "gfd" if intent.time_in_force in {"day", "gfd"} else intent.time_in_force
    fractional_market = _requires_market_fractional_order(intent)
    if fractional_market:
        request = {
            "account_number": account,
            "symbol": intent.ticker,
            "side": intent.side,
            "type": "market",
            "time_in_force": "gfd",
            "market_hours": "regular_hours",
        }
        if intent.side == "buy":
            notional = intent.notional
            if notional is None:
                quantity = _intent_resolved_quantity(intent)
                price = intent.limit_price or intent.estimated_price
                if quantity is None or price is None:
                    raise ValueError("Fractional buy needs notional or quantity plus price.")
                notional = quantity * price
            if notional <= 0:
                raise ValueError("Fractional/dollar market buy needs a positive dollar amount.")
            request["dollar_amount"] = _format_decimal(notional)
        else:
            quantity = _intent_resolved_quantity(intent)
            if quantity is None or quantity <= 0:
                raise ValueError("Fractional sell needs a positive quantity.")
            request["quantity"] = _format_quantity(quantity)
        return request

    if intent.order_type != "limit":
        raise ValueError("Whole-share Artha orders must use limit orders for price protection.")
    if not intent.limit_price:
        raise ValueError("Limit-order review requires a limit price.")
    quantity = _review_quantity(intent)
    if not _is_whole_share(quantity):
        raise ValueError("Fractional quantities must resolve to market/regular-hours orders.")
    return {
        "account_number": account,
        "symbol": intent.ticker,
        "side": intent.side,
        "type": "limit",
        "time_in_force": tif,
        "market_hours": "regular_hours",
        "quantity": _format_quantity(round(quantity)),
        "limit_price": _format_decimal(intent.limit_price),
    }


def build_robinhood_review_request(
    intent: OrderIntent,
    account_number: str | None = None,
) -> dict[str, Any]:
    """Build parameters for Robinhood review_equity_order.

    Whole-share proposals become price-controlled limit orders. Fractional or
    dollar-based proposals become market/regular-hours requests, which is the
    only Robinhood MCP-supported shape for fractional equity orders.
    """
    return resolve_robinhood_order_request(intent, account_number)


@dataclass
class OrderIntent:
    """One proposed broker order before guardrails and broker submission."""

    ticker: str
    side: str
    order_type: str = "limit"
    time_in_force: str = "day"
    quantity: float | None = None
    notional: float | None = None
    limit_price: float | None = None
    estimated_price: float | None = None
    decision_dossier_path: str = ""
    recommendation_id: int | None = None
    thesis_id: str = ""
    rationale: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = True
    order_intent_id: str = field(default_factory=lambda: f"rh-intent-{uuid4().hex}")

    def normalized(self) -> "OrderIntent":
        self.ticker = str(self.ticker or "").upper().strip()
        self.side = str(self.side or "").lower().strip()
        self.order_type = str(self.order_type or "limit").lower().strip()
        self.time_in_force = str(self.time_in_force or Config.ROBINHOOD_ORDER_TIF).lower().strip()
        for attr in ("quantity", "notional", "limit_price", "estimated_price"):
            value = _as_float(getattr(self, attr))
            setattr(self, attr, value)
        if self.order_type == "limit" and _requires_market_fractional_order(self):
            self.order_type = "market"
        return self


@dataclass
class GuardrailResult:
    passed: bool
    status: str
    reasons: list[str]
    checks: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "status": self.status,
            "reasons": self.reasons,
            "checks": self.checks,
        }


@dataclass
class BrokerResult:
    status: str
    broker: str
    broker_order_id: str
    dry_run: bool
    response: dict[str, Any]


OPEN_ORDER_STATES = {
    "new",
    "queued",
    "confirmed",
    "unconfirmed",
    "partially_filled",
    "pending_cancelled",
}


def load_runtime_trading_control() -> dict[str, Any]:
    """Read Artha's runtime Telegram kill-switch file."""
    try:
        path = Config.ROBINHOOD_CONTROL_FILE
        if not path:
            return {"trading_disabled": bool(Config.ROBINHOOD_KILL_SWITCH), "reason": "Config kill switch."}
        from pathlib import Path

        control_path = Path(path).expanduser()
        if not control_path.exists():
            return {"trading_disabled": bool(Config.ROBINHOOD_KILL_SWITCH), "reason": "Config kill switch." if Config.ROBINHOOD_KILL_SWITCH else ""}
        payload = json.loads(control_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
        payload["trading_disabled"] = bool(payload.get("trading_disabled") or Config.ROBINHOOD_KILL_SWITCH)
        if Config.ROBINHOOD_KILL_SWITCH and not payload.get("reason"):
            payload["reason"] = "Config kill switch."
        return payload
    except Exception as exc:
        return {"trading_disabled": True, "reason": f"Runtime control unreadable: {type(exc).__name__}: {exc}"}


def _snapshot_cash(snapshot: dict[str, Any]) -> float | None:
    portfolio = snapshot.get("portfolio") if isinstance(snapshot.get("portfolio"), dict) else {}
    for key in ("buying_power", "cash_available", "cash", "withdrawable_amount"):
        value = _as_float(portfolio.get(key))
        if value is not None:
            return value
    return None


def _snapshot_position_notional(snapshot: dict[str, Any], ticker: str) -> float:
    wanted = ticker.upper().strip()
    total = 0.0
    for row in snapshot.get("positions") or []:
        if not isinstance(row, dict) or _broker_position_symbol(row) != wanted:
            continue
        qty = _broker_position_quantity(row)
        price = (
            _as_float(row.get("market_price"))
            or _as_float(row.get("current_price"))
            or _as_float(row.get("average_buy_price"))
            or _as_float(row.get("average_price"))
            or _as_float(row.get("price"))
            or 0.0
        )
        total += max(0.0, qty * price)
    return total


def _open_duplicate_orders(snapshot: dict[str, Any], ticker: str, side: str) -> list[dict[str, Any]]:
    wanted = ticker.upper().strip()
    wanted_side = side.lower().strip()
    duplicates: list[dict[str, Any]] = []
    for row in snapshot.get("orders") or []:
        if not isinstance(row, dict):
            continue
        state = str(row.get("state") or "").lower()
        symbol = str(row.get("symbol") or row.get("equity_symbol") or "").upper().strip()
        order_side = str(row.get("side") or "").lower().strip()
        if state in OPEN_ORDER_STATES and symbol == wanted and order_side == wanted_side:
            duplicates.append(row)
    return duplicates


def evaluate_broker_snapshot_guardrails(
    intent: OrderIntent,
    snapshot: dict[str, Any] | None,
    notional: float | None,
) -> dict[str, Any]:
    """Pre-trade broker-state checks that require a fresh Robinhood snapshot."""
    if not snapshot:
        return {"passed": True, "status": "SKIPPED", "reasons": [], "checks": {"provided": False}}
    reasons: list[str] = []
    checks: dict[str, Any] = {
        "provided": True,
        "snapshot_status": snapshot.get("status"),
        "fresh": snapshot.get("fresh"),
        "age_minutes": snapshot.get("age_minutes"),
    }
    if snapshot.get("status") != "PASS" or not snapshot.get("fresh", False):
        reasons.append("Fresh Robinhood snapshot is required before preparing/placing an order.")
    account_check = snapshot.get("account_check")
    if isinstance(account_check, dict):
        checks["account_check"] = account_check
        if not account_check.get("passed"):
            reasons.extend(str(r) for r in account_check.get("reasons") or ["Robinhood account allowlist check failed."])
    elif snapshot.get("account"):
        validation = validate_allowlisted_robinhood_account(snapshot.get("account"))
        checks["account_check"] = validation.to_dict()
        if not validation.passed:
            reasons.extend(validation.reasons)

    duplicates = _open_duplicate_orders(snapshot, intent.ticker, intent.side) if Config.ROBINHOOD_BLOCK_DUPLICATE_OPEN_ORDERS else []
    checks["duplicate_open_orders"] = len(duplicates)
    if duplicates:
        reasons.append(f"Open Robinhood {intent.side} order already exists for {intent.ticker}.")

    if intent.side == "buy":
        cash = _snapshot_cash(snapshot)
        checks["cash_available"] = cash
        checks["cash_buffer"] = Config.ROBINHOOD_MIN_CASH_BUFFER_DOLLARS
        if cash is None:
            reasons.append("Robinhood snapshot is missing buying power/cash.")
        elif notional is not None and notional + Config.ROBINHOOD_MIN_CASH_BUFFER_DOLLARS > cash:
            reasons.append("Insufficient Robinhood cash/buying power for this order plus safety buffer.")

        existing_notional = _snapshot_position_notional(snapshot, intent.ticker)
        checks["existing_position_notional"] = existing_notional
        if notional is not None and existing_notional + notional > Config.ROBINHOOD_MAX_POSITION_DOLLARS:
            reasons.append(
                f"Existing {intent.ticker} exposure plus proposed buy exceeds the per-position pilot cap."
            )

    return {
        "passed": not reasons,
        "status": "PASS" if not reasons else "BLOCKED",
        "reasons": reasons,
        "checks": checks,
    }


class RobinhoodExecutionGuardrails:
    """Hard safety rules for the tiny Robinhood Agentic pilot."""

    def __init__(self, market_hours: MarketHours | None = None):
        self.market_hours = market_hours or MarketHours()

    def evaluate(
        self,
        intent: OrderIntent,
        market_data: dict[str, Any] | None = None,
        journal: DecisionJournal | None = None,
        now: datetime | None = None,
        broker_snapshot: dict[str, Any] | None = None,
    ) -> GuardrailResult:
        intent = intent.normalized()
        journal = journal or DecisionJournal()
        market_data = market_data or {}
        if broker_snapshot is None and isinstance(market_data.get("broker_snapshot"), dict):
            broker_snapshot = market_data.get("broker_snapshot")
        now = now or _utcnow()

        checks: dict[str, Any] = {}
        reasons: list[str] = []

        price = (
            intent.estimated_price
            or intent.limit_price
            or _as_float(market_data.get("price"))
            or _as_float(market_data.get("last_price"))
        )
        volume = _as_float(market_data.get("volume") or market_data.get("avg_volume"))
        bid = _as_float(market_data.get("bid") or market_data.get("bid_price"))
        ask = _as_float(market_data.get("ask") or market_data.get("ask_price"))
        dollar_volume = _as_float(market_data.get("dollar_volume"))
        if dollar_volume is None and price is not None and volume is not None:
            dollar_volume = price * volume

        notional = intent.notional
        if notional is None and intent.quantity is not None:
            basis_price = intent.limit_price or price
            if basis_price is not None:
                notional = intent.quantity * basis_price
        checks["resolved_notional"] = notional
        checks["resolved_price"] = price

        if not TICKER_RE.match(intent.ticker) or intent.ticker in CRYPTO_LIKE:
            reasons.append("Only normal US equity tickers are allowed in the pilot.")
        checks["ticker_shape"] = "pass" if not reasons else "checked"

        if intent.side not in {"buy", "sell"}:
            reasons.append("Order side must be buy or sell.")
        market_open = self.market_hours.is_market_open(now)
        fractional_market_required = _requires_market_fractional_order(intent)
        if intent.order_type not in Config.ROBINHOOD_ALLOWED_ORDER_TYPES:
            reasons.append("Only configured Robinhood order types are allowed.")
        if intent.order_type == "limit" and not intent.limit_price:
            reasons.append("Limit orders require a limit price.")
        if intent.order_type == "market":
            if not fractional_market_required:
                reasons.append("Whole-share orders must use limit orders for price protection.")
            if not market_open and not intent.dry_run:
                reasons.append("Fractional/dollar market orders require regular market hours.")
            if intent.limit_price is None:
                reasons.append("Fractional/dollar market orders need Artha's intended reference price for drift checks.")
        checks["order_type"] = intent.order_type
        checks["order_resolution"] = {
            "fractional_or_dollar_order": fractional_market_required,
            "resolved_quantity": _intent_resolved_quantity(intent),
            "intended_limit_price": intent.limit_price,
            "market_order_max_drift_pct": Config.ROBINHOOD_MARKET_ORDER_MAX_PRICE_DRIFT_PCT,
        }

        if not Config.ROBINHOOD_ALLOW_AFTER_HOURS and not market_open:
            reasons.append("Market is closed; after-hours orders are disabled.")
        checks["market_open"] = market_open

        if not intent.dry_run:
            if Config.ROBINHOOD_REVIEW_ONLY:
                reasons.append("Robinhood review-only mode is enabled.")
            if Config.ROBINHOOD_KILL_SWITCH:
                reasons.append("Robinhood kill switch is enabled.")
            runtime_control = load_runtime_trading_control()
            if runtime_control.get("trading_disabled"):
                reasons.append(f"Runtime trading kill switch is enabled: {runtime_control.get('reason') or 'disabled'}")
            if Config.ROBINHOOD_DRY_RUN_ONLY:
                reasons.append("Robinhood dry-run-only mode is enabled.")
            if not Config.ROBINHOOD_AGENTIC_ENABLED:
                reasons.append("Robinhood Agentic live execution is disabled.")
        checks["live_execution_allowed_by_config"] = bool(
            Config.ROBINHOOD_AGENTIC_ENABLED
            and not Config.ROBINHOOD_DRY_RUN_ONLY
            and not Config.ROBINHOOD_KILL_SWITCH
            and not Config.ROBINHOOD_REVIEW_ONLY
        )

        broker_snapshot_check = evaluate_broker_snapshot_guardrails(intent, broker_snapshot, notional)
        checks["broker_snapshot"] = broker_snapshot_check
        if not broker_snapshot_check.get("passed"):
            reasons.extend(str(reason) for reason in broker_snapshot_check.get("reasons") or [])

        if price is None or price <= 0:
            reasons.append("A current price is required before preparing an order.")
        elif price < Config.ROBINHOOD_MIN_PRICE:
            reasons.append(f"Price ${price:.2f} is below the ${Config.ROBINHOOD_MIN_PRICE:.2f} pilot floor.")
        checks["price_floor"] = {"price": price, "minimum": Config.ROBINHOOD_MIN_PRICE}

        if intent.order_type == "market" and intent.limit_price is not None:
            execution_price = ask if intent.side == "buy" else bid
            if execution_price is None or execution_price <= 0:
                execution_price = price
            max_drift = Config.ROBINHOOD_MARKET_ORDER_MAX_PRICE_DRIFT_PCT
            if execution_price is None or execution_price <= 0:
                reasons.append("Live quote is required before resolving a fractional/dollar market order.")
            elif intent.side == "buy" and execution_price > intent.limit_price * (1 + max_drift):
                reasons.append(
                    f"Live ask/price ${execution_price:.2f} is above Artha reference ${intent.limit_price:.2f} "
                    f"by more than {max_drift:.2%}; re-review before buying."
                )
            elif intent.side == "sell" and execution_price < intent.limit_price * (1 - max_drift):
                reasons.append(
                    f"Live bid/price ${execution_price:.2f} is below Artha reference ${intent.limit_price:.2f} "
                    f"by more than {max_drift:.2%}; re-review before selling."
                )
            checks["market_price_drift"] = {
                "execution_price": execution_price,
                "reference_price": intent.limit_price,
                "maximum_drift_pct": max_drift,
            }

        if notional is None or notional <= 0:
            reasons.append("Order notional/quantity must resolve to a positive dollar amount.")
        elif intent.side == "buy":
            if notional > Config.ROBINHOOD_MAX_POSITION_DOLLARS:
                reasons.append(
                    f"Order size ${notional:.2f} exceeds the ${Config.ROBINHOOD_MAX_POSITION_DOLLARS:.2f} pilot cap."
                )
            if notional > Config.ROBINHOOD_PILOT_MAX_ACCOUNT_VALUE:
                reasons.append(
                    f"Order size ${notional:.2f} exceeds the ${Config.ROBINHOOD_PILOT_MAX_ACCOUNT_VALUE:.2f} account pilot cap."
                )
        checks["position_cap"] = {
            "notional": notional,
            "max_position_dollars": Config.ROBINHOOD_MAX_POSITION_DOLLARS,
            "pilot_account_cap": Config.ROBINHOOD_PILOT_MAX_ACCOUNT_VALUE,
            "applies": "buy_only",
        }

        if intent.side == "buy":
            if dollar_volume is None:
                reasons.append("Liquidity check needs price and volume before buying.")
            elif dollar_volume < Config.ROBINHOOD_MIN_DOLLAR_VOLUME:
                reasons.append(
                    f"Dollar volume ${dollar_volume:,.0f} is below the ${Config.ROBINHOOD_MIN_DOLLAR_VOLUME:,.0f} floor."
                )
            checks["liquidity"] = {
                "dollar_volume": dollar_volume,
                "minimum": Config.ROBINHOOD_MIN_DOLLAR_VOLUME,
            }

            if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
                reasons.append("Bid/ask quote is required to avoid bad spreads.")
                spread_pct = None
            else:
                spread_pct = (ask - bid) / ((ask + bid) / 2)
                if spread_pct > Config.ROBINHOOD_MAX_SPREAD_PCT:
                    reasons.append(
                        f"Spread {spread_pct:.2%} is wider than the {Config.ROBINHOOD_MAX_SPREAD_PCT:.2%} pilot limit."
                    )
            checks["spread"] = {
                "bid": bid,
                "ask": ask,
                "spread_pct": spread_pct,
                "maximum": Config.ROBINHOOD_MAX_SPREAD_PCT,
            }

            if journal.count_execution_orders_today(side="buy") >= Config.ROBINHOOD_MAX_TRADES_PER_DAY:
                reasons.append(f"Daily buy order cap reached ({Config.ROBINHOOD_MAX_TRADES_PER_DAY}).")
            checks["daily_trade_cap"] = {
                "buy_orders_today": journal.count_execution_orders_today(side="buy"),
                "maximum": Config.ROBINHOOD_MAX_TRADES_PER_DAY,
            }

            if Config.ROBINHOOD_REQUIRE_SUPERVISOR_PASS_FOR_BUYS:
                supervisor = _latest_supervisor_payload(journal)
                severity = str(supervisor.get("severity") or "").upper()
                buy_gate = _supervisor_buy_gate(supervisor, review_only=bool(intent.dry_run))
                checks["supervisor"] = {
                    "row_id": supervisor.get("row_id"),
                    "severity": severity or "missing",
                    "generated_at": supervisor.get("generated_at"),
                    "buy_gate": buy_gate,
                }
                if not buy_gate.get("allowed"):
                    reasons.append(str(buy_gate.get("reason") or "Latest Supervisor state blocks new buys."))

            decision = self._decision_check(intent, journal, price)
            checks["decision_evidence"] = decision
            if not decision.get("allowed"):
                reasons.append(str(decision.get("reason") or "Buy order lacks a valid council decision trail."))

        if intent.side == "sell":
            active = journal.get_active_thesis_for_ticker(intent.ticker)
            checks["sell_thesis"] = {
                "active_thesis": bool(active),
                "thesis_id": active.get("thesis_id") if active else intent.thesis_id,
            }
            if not active and not intent.thesis_id:
                reasons.append("Sell orders need an active thesis or explicit thesis id.")

        status = "PASS" if not reasons else "BLOCKED"
        return GuardrailResult(
            passed=not reasons,
            status=status,
            reasons=reasons,
            checks=checks,
        )

    def _decision_check(
        self,
        intent: OrderIntent,
        journal: DecisionJournal,
        price: float | None,
    ) -> dict[str, Any]:
        feature = None
        if intent.decision_dossier_path:
            for row in journal.get_decision_features(limit=500):
                if str(row.get("dossier_path") or "") == intent.decision_dossier_path:
                    feature = row
                    break
        if feature is None:
            feature = _latest_feature_for_ticker(journal, intent.ticker)
        if feature is None:
            return {"allowed": False, "reason": "No point-in-time decision dossier found for this ticker."}

        final_verdict = str(feature.get("final_verdict") or "").upper().strip()
        evidence_count = int(feature.get("evidence_count") or 0)
        dossier_path = str(feature.get("dossier_path") or "")
        if not intent.decision_dossier_path and dossier_path:
            intent.decision_dossier_path = dossier_path

        if evidence_count < Config.ROBINHOOD_MIN_BUY_EVIDENCE_ITEMS:
            return {
                "allowed": False,
                "reason": f"Decision dossier has only {evidence_count} evidence item(s).",
                "final_verdict": final_verdict,
                "evidence_count": evidence_count,
                "dossier_path": dossier_path,
            }
        if final_verdict in BUY_ACTIONS:
            return {
                "allowed": True,
                "reason": "Latest council decision is buy-side.",
                "final_verdict": final_verdict,
                "evidence_count": evidence_count,
                "dossier_path": dossier_path,
            }

        watch = journal.get_active_defer_watch_for_ticker(intent.ticker)
        zone_low = _as_float((watch or {}).get("zone_low"))
        zone_high = _as_float((watch or {}).get("zone_high"))
        zone_hit = (
            bool(watch)
            and price is not None
            and zone_low is not None
            and zone_high is not None
            and zone_low <= price <= zone_high
        )
        if final_verdict in {"DEFER", "WATCH"} and zone_hit:
            return {
                "allowed": True,
                "reason": "DEFER/WATCH entry zone is currently hit.",
                "final_verdict": final_verdict,
                "evidence_count": evidence_count,
                "dossier_path": dossier_path,
                "watch_id": watch.get("watch_id"),
                "zone_low": zone_low,
                "zone_high": zone_high,
            }

        return {
            "allowed": False,
            "reason": f"Latest council verdict is {final_verdict or 'UNKNOWN'}, not a buy-side decision or triggered entry zone.",
            "final_verdict": final_verdict,
            "evidence_count": evidence_count,
            "dossier_path": dossier_path,
        }


class DryRunBroker:
    """Broker adapter that never touches Robinhood."""

    broker = "robinhood"

    def submit_order(self, intent: OrderIntent, guardrails: GuardrailResult) -> BrokerResult:
        if not guardrails.passed:
            return BrokerResult(
                status="blocked",
                broker=self.broker,
                broker_order_id="",
                dry_run=True,
                response={"blocked_reasons": guardrails.reasons},
            )
        return BrokerResult(
            status="dry_run_ready",
            broker=self.broker,
            broker_order_id=f"dryrun-{uuid4().hex[:12]}",
            dry_run=True,
            response={
                "message": "Dry-run order passed guardrails. No Robinhood order was placed.",
                "intent": asdict(intent),
            },
        )


class RobinhoodMCPBroker:
    """Future live broker adapter.

    The interface is ready for Robinhood MCP calls, but live order placement is
    intentionally unavailable until the MCP client is connected and safety
    switches are explicitly changed.
    """

    broker = "robinhood"

    def __init__(
        self,
        mcp_url: str | None = None,
        client: Any | None = None,
        account_number: str | None = None,
    ):
        self.mcp_url = mcp_url or Config.ROBINHOOD_MCP_URL
        self.client = client
        self.account_number = account_number or Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER

    def build_review_request(self, intent: OrderIntent) -> dict[str, Any]:
        return build_robinhood_review_request(intent, self.account_number)

    def review_order(
        self,
        intent: OrderIntent,
        guardrails: GuardrailResult,
        account: dict[str, Any],
        review_fn: Any | None = None,
    ) -> BrokerResult:
        account_check = validate_allowlisted_robinhood_account(account, self.account_number)
        if not guardrails.passed or not account_check.passed:
            return BrokerResult(
                status="blocked",
                broker=self.broker,
                broker_order_id="",
                dry_run=True,
                response={
                    "blocked_reasons": guardrails.reasons + account_check.reasons,
                    "account_check": account_check.to_dict(),
                },
            )
        request = self.build_review_request(intent)
        if review_fn is None:
            return BrokerResult(
                status="review_ready",
                broker=self.broker,
                broker_order_id="",
                dry_run=True,
                response={
                    "message": "Robinhood review request is ready. No order was placed.",
                    "review_request": request,
                    "account_check": account_check.to_dict(),
                },
            )
        review_response = review_fn(**request)
        return BrokerResult(
            status="reviewed",
            broker=self.broker,
            broker_order_id="",
            dry_run=True,
            response={
                "message": "Robinhood reviewed the order. No order was placed.",
                "review_request": request,
                "review_response": review_response,
                "account_check": account_check.to_dict(),
            },
        )

    def submit_order(self, intent: OrderIntent, guardrails: GuardrailResult) -> BrokerResult:
        if not guardrails.passed:
            return BrokerResult(
                status="blocked",
                broker=self.broker,
                broker_order_id="",
                dry_run=False,
                response={"blocked_reasons": guardrails.reasons},
            )
        if (
            Config.ROBINHOOD_REVIEW_ONLY
            or Config.ROBINHOOD_DRY_RUN_ONLY
            or Config.ROBINHOOD_KILL_SWITCH
            or not Config.ROBINHOOD_AGENTIC_ENABLED
        ):
            raise RuntimeError("Robinhood live execution is disabled by Artha safety configuration.")
        if self.client is None:
            raise RuntimeError("Robinhood MCP client is not connected yet.")
        raise NotImplementedError("Connect Robinhood MCP review/place-order calls in this adapter.")


def build_order_intent(
    ticker: str,
    side: str,
    notional: float | None = None,
    quantity: float | None = None,
    limit_price: float | None = None,
    estimated_price: float | None = None,
    decision_dossier_path: str = "",
    rationale: str = "",
    dry_run: bool = True,
) -> OrderIntent:
    """Build a normalized order intent from CLI or future automation."""
    return OrderIntent(
        ticker=ticker,
        side=side,
        order_type="limit",
        time_in_force=Config.ROBINHOOD_ORDER_TIF,
        quantity=quantity,
        notional=notional,
        limit_price=limit_price,
        estimated_price=estimated_price,
        decision_dossier_path=decision_dossier_path,
        rationale=rationale,
        dry_run=dry_run,
    ).normalized()


def evaluate_and_record_order(
    intent: OrderIntent,
    market_data: dict[str, Any] | None = None,
    journal: DecisionJournal | None = None,
    send_telegram: bool = False,
    sender: TelegramSender | None = None,
    now: datetime | None = None,
    broker_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate an order, record the audit row, and optionally notify Telegram."""
    journal = journal or DecisionJournal()
    sender = sender or TelegramSender()
    intent = intent.normalized()
    guardrails = RobinhoodExecutionGuardrails().evaluate(
        intent,
        market_data,
        journal,
        now=now,
        broker_snapshot=broker_snapshot,
    )
    broker = DryRunBroker() if intent.dry_run else RobinhoodMCPBroker()
    result = broker.submit_order(intent, guardrails)
    supervisor = _latest_supervisor_payload(journal)

    request = asdict(intent)
    order_row = {
        "order_intent_id": intent.order_intent_id,
        "ticker": intent.ticker,
        "side": intent.side,
        "order_type": intent.order_type,
        "time_in_force": intent.time_in_force,
        "quantity": intent.quantity,
        "notional": intent.notional,
        "limit_price": intent.limit_price,
        "estimated_price": intent.estimated_price,
        "status": result.status,
        "broker": result.broker,
        "broker_order_id": result.broker_order_id,
        "dry_run": result.dry_run,
        "decision_dossier_path": intent.decision_dossier_path,
        "recommendation_id": intent.recommendation_id,
        "thesis_id": intent.thesis_id,
        "supervisor_run_id": supervisor.get("row_id"),
        "guardrail_status": guardrails.status,
        "guardrail_json": guardrails.to_dict(),
        "rationale": intent.rationale,
        "evidence_json": intent.evidence,
        "request_json": request,
        "response_json": result.response,
        "submitted_at": _utcnow_iso() if result.status in {"submitted", "dry_run_ready"} else None,
        "notes": "; ".join(guardrails.reasons),
    }
    row_id = journal.save_execution_order(order_row)
    payload = {
        "row_id": row_id,
        "intent": request,
        "guardrails": guardrails.to_dict(),
        "broker_result": asdict(result),
        "telegram_sent": False,
    }
    if send_telegram and sender.enabled:
        payload["telegram_sent"] = bool(sender.send_message(format_order_notice(payload), parse_mode=None, silent=False))
    logger.info(
        "[execution] ticker=%s side=%s status=%s guardrail=%s dry_run=%s row=%s",
        intent.ticker,
        intent.side,
        result.status,
        guardrails.status,
        result.dry_run,
        row_id,
    )
    return payload


def prepare_and_record_robinhood_review(
    intent: OrderIntent,
    account: dict[str, Any],
    market_data: dict[str, Any] | None = None,
    journal: DecisionJournal | None = None,
    send_telegram: bool = False,
    sender: TelegramSender | None = None,
    now: datetime | None = None,
    broker_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare a Robinhood MCP review request and audit it without placing an order."""
    journal = journal or DecisionJournal()
    sender = sender or TelegramSender()
    intent = intent.normalized()
    guardrails = RobinhoodExecutionGuardrails().evaluate(
        intent,
        market_data,
        journal,
        now=now,
        broker_snapshot=broker_snapshot,
    )
    broker = RobinhoodMCPBroker()
    try:
        result = broker.review_order(intent, guardrails, account)
    except Exception as exc:
        result = BrokerResult(
            status="blocked",
            broker="robinhood",
            broker_order_id="",
            dry_run=True,
            response={"blocked_reasons": [str(exc)]},
        )
    supervisor = _latest_supervisor_payload(journal)
    request = asdict(intent)
    review_request = (result.response or {}).get("review_request") or {}
    order_row = {
        "order_intent_id": intent.order_intent_id,
        "ticker": intent.ticker,
        "side": intent.side,
        "order_type": intent.order_type,
        "time_in_force": intent.time_in_force,
        "quantity": intent.quantity,
        "notional": intent.notional,
        "limit_price": intent.limit_price,
        "estimated_price": intent.estimated_price,
        "status": result.status,
        "broker": result.broker,
        "broker_order_id": result.broker_order_id,
        "dry_run": 1,
        "decision_dossier_path": intent.decision_dossier_path,
        "recommendation_id": intent.recommendation_id,
        "thesis_id": intent.thesis_id,
        "supervisor_run_id": supervisor.get("row_id"),
        "guardrail_status": guardrails.status,
        "guardrail_json": guardrails.to_dict(),
        "rationale": intent.rationale,
        "evidence_json": intent.evidence,
        "request_json": review_request or request,
        "response_json": result.response,
        "submitted_at": None,
        "notes": "; ".join((result.response or {}).get("blocked_reasons") or guardrails.reasons),
    }
    row_id = journal.save_execution_order(order_row)
    payload = {
        "row_id": row_id,
        "intent": request,
        "guardrails": guardrails.to_dict(),
        "broker_result": asdict(result),
        "telegram_sent": False,
    }
    if send_telegram and sender.enabled:
        payload["telegram_sent"] = bool(sender.send_message(format_order_notice(payload), parse_mode=None, silent=False))
    logger.info(
        "[execution] robinhood_review ticker=%s side=%s status=%s guardrail=%s row=%s",
        intent.ticker,
        intent.side,
        result.status,
        guardrails.status,
        row_id,
    )
    return payload


def record_robinhood_review_response(
    order_intent_id: str,
    review_response: dict[str, Any],
    journal: DecisionJournal | None = None,
) -> None:
    """Persist the Robinhood MCP review response for a previously prepared order."""
    journal = journal or DecisionJournal()
    journal.update_execution_order(
        order_intent_id,
        {
            "status": "reviewed",
            "response_json": review_response,
            "notes": "Robinhood MCP review response recorded; no order placed.",
        },
    )


def build_execution_readiness_report(journal: DecisionJournal | None = None) -> dict[str, Any]:
    """Check whether Artha-side Robinhood execution plumbing is ready."""
    journal = journal or DecisionJournal()
    db_ok = False
    order_count = 0
    action_count = 0
    try:
        with journal._connect() as conn:
            conn.execute("SELECT 1 FROM execution_orders LIMIT 1").fetchone()
            row = conn.execute("SELECT COUNT(*) AS c FROM execution_orders").fetchone()
            order_count = int(row["c"] if row else 0)
            conn.execute("SELECT 1 FROM trade_actions LIMIT 1").fetchone()
            action_row = conn.execute("SELECT COUNT(*) AS c FROM trade_actions").fetchone()
            action_count = int(action_row["c"] if action_row else 0)
            db_ok = True
    except Exception as exc:
        return {
            "status": "FAIL",
            "ready_for_dry_run": False,
            "live_trading_enabled": False,
            "message": f"Execution audit table is not reachable: {exc}",
        }

    config_errors: list[str] = []
    allowed_order_types = set(Config.ROBINHOOD_ALLOWED_ORDER_TYPES)
    if "limit" not in allowed_order_types:
        config_errors.append("limit orders must be allowed for whole-share price protection")
    if "market" not in allowed_order_types:
        config_errors.append("market orders must be allowed for Robinhood fractional/dollar orders")
    disallowed = sorted(allowed_order_types - {"limit", "market"})
    if disallowed:
        config_errors.append(f"pilot should not allow unsupported order types: {', '.join(disallowed)}")
    if Config.ROBINHOOD_MAX_POSITION_DOLLARS <= 0:
        config_errors.append("max position dollars must be positive")
    if Config.ROBINHOOD_MAX_POSITION_DOLLARS > Config.ROBINHOOD_PILOT_MAX_ACCOUNT_VALUE:
        config_errors.append("max position cannot exceed pilot account cap")
    if Config.ROBINHOOD_MAX_TRADES_PER_DAY < 1:
        config_errors.append("daily trade cap must be at least 1")
    if not Config.ROBINHOOD_MCP_URL:
        config_errors.append("Robinhood MCP URL is missing")
    if not Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER:
        config_errors.append("Robinhood Agentic account number allowlist is missing")

    supervisor = _latest_supervisor_payload(journal)
    live_enabled = bool(
        Config.ROBINHOOD_AGENTIC_ENABLED
        and not Config.ROBINHOOD_DRY_RUN_ONLY
        and not Config.ROBINHOOD_KILL_SWITCH
        and not Config.ROBINHOOD_REVIEW_ONLY
        and not load_runtime_trading_control().get("trading_disabled")
    )
    ready = db_ok and not config_errors
    status = "PASS" if ready else "FAIL"
    message = (
        "Dry-run execution is ready; live Robinhood trading remains disabled."
        if ready and not live_enabled
        else "Live Robinhood trading configuration is enabled."
        if ready and live_enabled
        else "Execution readiness failed."
    )
    return {
        "status": status,
        "ready_for_dry_run": ready,
        "live_trading_enabled": live_enabled,
        "message": message,
        "mcp_url": Config.ROBINHOOD_MCP_URL,
        "db_ok": db_ok,
        "execution_order_rows": order_count,
        "trade_action_rows": action_count,
        "config_errors": config_errors,
        "snapshot_file": Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
        "control_file": Config.ROBINHOOD_CONTROL_FILE,
        "runtime_control": load_runtime_trading_control(),
        "account_allowlist": {
            "configured": bool(Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER),
            "masked_account": mask_account_number(Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER),
            "expected_type": Config.ROBINHOOD_EXPECTED_ACCOUNT_TYPE,
            "expected_nickname": Config.ROBINHOOD_EXPECTED_ACCOUNT_NICKNAME,
        },
        "guardrails": {
            "review_only": Config.ROBINHOOD_REVIEW_ONLY,
            "dry_run_only": Config.ROBINHOOD_DRY_RUN_ONLY,
            "kill_switch": Config.ROBINHOOD_KILL_SWITCH,
            "agentic_enabled": Config.ROBINHOOD_AGENTIC_ENABLED,
            "max_account_value": Config.ROBINHOOD_PILOT_MAX_ACCOUNT_VALUE,
            "max_position_dollars": Config.ROBINHOOD_MAX_POSITION_DOLLARS,
            "max_trades_per_day": Config.ROBINHOOD_MAX_TRADES_PER_DAY,
            "min_price": Config.ROBINHOOD_MIN_PRICE,
            "min_dollar_volume": Config.ROBINHOOD_MIN_DOLLAR_VOLUME,
            "max_spread_pct": Config.ROBINHOOD_MAX_SPREAD_PCT,
            "market_order_max_price_drift_pct": Config.ROBINHOOD_MARKET_ORDER_MAX_PRICE_DRIFT_PCT,
            "review_decision_max_age_minutes": Config.ROBINHOOD_REVIEW_DECISION_MAX_AGE_MINUTES,
            "fresh_snapshot_required_for_review": Config.ROBINHOOD_REQUIRE_FRESH_SNAPSHOT_FOR_REVIEW,
            "allowed_order_types": list(Config.ROBINHOOD_ALLOWED_ORDER_TYPES),
            "after_hours_allowed": Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            "supervisor_pass_required_for_buys": Config.ROBINHOOD_REQUIRE_SUPERVISOR_PASS_FOR_BUYS,
        },
        "latest_supervisor": supervisor,
    }


def format_execution_readiness(report: dict[str, Any]) -> str:
    """Plain-English execution readiness report."""
    guardrails = report.get("guardrails") or {}
    supervisor = report.get("latest_supervisor") or {}
    allowlist = report.get("account_allowlist") or {}
    runtime_control = report.get("runtime_control") or {}
    lines = [
        "ARTHA ROBINHOOD READINESS",
        "=========================",
        f"Status: {report.get('status')}",
        f"Dry-run ready: {report.get('ready_for_dry_run')}",
        f"Live trading enabled: {report.get('live_trading_enabled')}",
        f"Message: {report.get('message')}",
        f"Execution rows: {report.get('execution_order_rows')}",
        f"Trade action rows: {report.get('trade_action_rows')}",
        "",
        "Safety cage:",
        f"- Agentic account allowlist: {allowlist.get('masked_account') or 'missing'}",
        f"- Review only: {guardrails.get('review_only')}",
        f"- Dry-run only: {guardrails.get('dry_run_only')}",
        f"- Kill switch: {guardrails.get('kill_switch')}",
        f"- Runtime trading disabled: {runtime_control.get('trading_disabled')}",
        f"- Max account pilot: ${guardrails.get('max_account_value')}",
        f"- Max per position: ${guardrails.get('max_position_dollars')}",
        f"- Max trades/day: {guardrails.get('max_trades_per_day')}",
        f"- Order types: {', '.join(guardrails.get('allowed_order_types') or [])}",
        f"- Market order max drift: {guardrails.get('market_order_max_price_drift_pct')}",
        f"- Review decision max age min: {guardrails.get('review_decision_max_age_minutes')}",
        f"- Fresh snapshot required for Review: {guardrails.get('fresh_snapshot_required_for_review')}",
        f"- After-hours allowed: {guardrails.get('after_hours_allowed')}",
        f"- Latest Supervisor: {supervisor.get('severity') or 'missing'}",
        f"- Snapshot file: {report.get('snapshot_file')}",
        f"- Control file: {report.get('control_file')}",
    ]
    errors = report.get("config_errors") or []
    if errors:
        lines.extend(["", "Config problems:"])
        lines.extend(f"- {err}" for err in errors)
    lines.extend(
        [
            "",
            "Plain English:",
            "Artha can prepare audited Robinhood actions, publish Telegram approval tokens, import broker snapshots, and activate sell monitoring after fills. Real placement remains blocked unless all safety switches are intentionally opened and a fresh Robinhood review passes.",
        ]
    )
    return "\n".join(lines)


def format_order_notice(payload: dict[str, Any]) -> str:
    """Telegram/plain text notice for one execution proposal."""
    intent = payload.get("intent") or {}
    guardrails = payload.get("guardrails") or {}
    broker = payload.get("broker_result") or {}
    lines = [
        "ARTHA ORDER DRY-RUN",
        "===================",
        f"Ticker: {intent.get('ticker')}",
        f"Side: {intent.get('side')}",
        f"Order type: {intent.get('order_type')}",
        f"Notional: {intent.get('notional')}",
        f"Limit: {intent.get('limit_price')}",
        f"Status: {broker.get('status')}",
        f"Guardrails: {guardrails.get('status')}",
    ]
    reasons = guardrails.get("reasons") or []
    if reasons:
        lines.append("")
        lines.append("Blocked reasons:")
        lines.extend(f"- {reason}" for reason in reasons)
    lines.append("")
    lines.append("No real Robinhood order was placed.")
    return "\n".join(lines)
