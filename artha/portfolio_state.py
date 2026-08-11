"""Portfolio state engine for context injection and journaling.

``portfolio.json`` is Artha's derived operational cache for position-level sell
metadata and offline reporting. A fresh Robinhood snapshot is authoritative for
live shares, cash, and buying power; SQLite execution/event tables are
authoritative for execution learning. This module computes compact portfolio
metrics and prompt/database payloads without promoting the legacy transaction
array to broker truth.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import DATA_DIR

logger = logging.getLogger(__name__)

PORTFOLIO_JSON_PATH = DATA_DIR / "portfolio.json"

# --- Broker cash reconciliation tunables (env-overridable; Wave 2 consolidates into Config) ---
BROKER_SNAPSHOT_PATH = Path(
    os.getenv(
        "ARTHA_BROKER_SNAPSHOT_PATH",
        str(DATA_DIR / "robinhood" / "latest_snapshot.json"),
    )
)
# Snapshot older than this is considered stale and ignored (fall back to portfolio.json cash).
BROKER_SNAPSHOT_MAX_AGE_HOURS = float(os.getenv("ARTHA_BROKER_SNAPSHOT_MAX_AGE_HOURS", "24"))
# Relative disagreement between broker cash and ledger cash that triggers a WARNING.
CASH_DISAGREEMENT_WARN_PCT = float(os.getenv("ARTHA_CASH_DISAGREEMENT_WARN_PCT", "0.20"))


def get_broker_cash(
    snapshot_path: Path | None = None,
    max_age_hours: float | None = None,
) -> float | None:
    """Read spendable broker buying power from a fresh Robinhood snapshot.

    The historical name is retained for compatibility. In a cash account,
    the snapshot's ``cash`` may include unsettled sale proceeds, so only
    buying power is safe to return as immediately deployable cash.
    """
    from .broker_capacity import load_broker_capacity

    path = snapshot_path or BROKER_SNAPSHOT_PATH
    max_age_minutes = int(
        60.0 * (BROKER_SNAPSHOT_MAX_AGE_HOURS if max_age_hours is None else max_age_hours)
    )
    capacity = load_broker_capacity(path, max_age_minutes=max_age_minutes)
    if not capacity.get("usable"):
        logger.warning(
            "Broker snapshot cannot provide spendable buying power: %s",
            "; ".join(str(reason) for reason in capacity.get("reasons") or []),
        )
        return None
    return float(capacity.get("buying_power") or 0.0)


def get_monthly_budget_remaining(
    monthly_budget: float,
    portfolio_path: Path | None = None,
    now: datetime | None = None,
) -> float:
    """Monthly budget minus BUY dollars already recorded this calendar month (UTC).

    Reads the transaction ledger from portfolio.json. If the ledger is
    unavailable the full budget is returned (missing data must never
    silently zero the deployable amount).
    """
    if monthly_budget <= 0:
        return 0.0
    path = portfolio_path or PORTFOLIO_JSON_PATH
    current = now or datetime.now(timezone.utc)
    month_prefix = current.strftime("%Y-%m")
    spent = 0.0
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for txn in data.get("transactions", []) or []:
            if not isinstance(txn, dict) or txn.get("type") != "BUY":
                continue
            if str(txn.get("timestamp", "")).startswith(month_prefix):
                try:
                    spent += float(txn.get("total", 0) or 0)
                except (TypeError, ValueError):
                    continue
    except Exception as exc:
        logger.warning("Could not read transaction ledger for monthly budget math: %s", exc)
    return max(0.0, monthly_budget - spent)


@dataclass
class PositionState:
    """Normalized position metrics used by downstream prompt/db layers."""

    ticker: str
    asset_type: str
    sector: str
    industry: str
    shares: float
    avg_cost: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    weight_pct: float
    target_pct: float | None
    drift_pct: float | None


class PortfolioStateEngine:
    """Compute normalized portfolio state from manual JSON input."""

    def __init__(self, portfolio_path: Path | None = None) -> None:
        # Resolve the default at construction time so tests and alternate
        # deployments can safely substitute an isolated portfolio source.
        self.portfolio_path = portfolio_path or PORTFOLIO_JSON_PATH

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def load_portfolio(self) -> dict[str, Any]:
        """Load manual portfolio JSON; return safe default if unavailable."""
        if not self.portfolio_path.exists():
            logger.warning("Portfolio file missing at %s; using empty default", self.portfolio_path)
            return {
                "last_updated": self._now_iso()[:10],
                "cash_available": 0,
                "monthly_contribution": 0,
                "positions": [],
                "notes": "",
            }

        try:
            with open(self.portfolio_path, encoding="utf-8") as f:
                payload = json.load(f)
                if isinstance(payload, dict):
                    return payload
        except Exception as exc:
            logger.error("Failed to load portfolio JSON %s: %s", self.portfolio_path, exc)

        return {
            "last_updated": self._now_iso()[:10],
            "cash_available": 0,
            "monthly_contribution": 0,
            "positions": [],
            "notes": "",
        }

    def compute_state(self, portfolio_data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Compute portfolio-level and position-level metrics."""
        data = portfolio_data or self.load_portfolio()
        cash_available = self._to_float(data.get("cash_available"), 0.0)
        monthly_contribution = self._to_float(data.get("monthly_contribution"), 0.0)
        raw_positions = data.get("positions", []) if isinstance(data.get("positions"), list) else []

        normalized_positions: list[PositionState] = []
        total_holdings_value = 0.0
        total_cost_basis = 0.0

        for raw in raw_positions:
            if not isinstance(raw, dict):
                continue

            ticker = str(raw.get("ticker", "")).upper().strip()
            if not ticker:
                continue

            shares = self._to_float(raw.get("shares", raw.get("quantity", 0.0)))
            avg_cost = self._to_float(raw.get("avg_cost", raw.get("average_cost", 0.0)))
            current_price = self._to_float(raw.get("current_price", raw.get("price", 0.0)))

            market_value = self._to_float(raw.get("market_value"), shares * current_price)
            cost_basis = self._to_float(raw.get("cost_basis"), shares * avg_cost)
            unrealized_pnl = market_value - cost_basis
            unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100.0) if cost_basis > 0 else 0.0

            total_holdings_value += market_value
            total_cost_basis += cost_basis

            normalized_positions.append(
                PositionState(
                    ticker=ticker,
                    asset_type=str(raw.get("asset_type", "stock")),
                    sector=str(raw.get("sector") or "").strip(),
                    industry=str(raw.get("industry") or "").strip(),
                    shares=shares,
                    avg_cost=avg_cost,
                    current_price=current_price,
                    market_value=market_value,
                    cost_basis=cost_basis,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    weight_pct=0.0,
                    target_pct=self._to_float(raw.get("target_pct"), 0.0) if raw.get("target_pct") is not None else None,
                    drift_pct=None,
                )
            )

        total_value = cash_available + total_holdings_value
        for pos in normalized_positions:
            pos.weight_pct = (pos.market_value / total_value * 100.0) if total_value > 0 else 0.0
            if pos.target_pct is not None:
                pos.drift_pct = pos.weight_pct - pos.target_pct

        concentration_pct = max((p.weight_pct for p in normalized_positions), default=0.0)
        allocation_cash_pct = (cash_available / total_value * 100.0) if total_value > 0 else 0.0

        total_unrealized_pnl = total_holdings_value - total_cost_basis
        total_unrealized_pnl_pct = (
            (total_unrealized_pnl / total_cost_basis * 100.0) if total_cost_basis > 0 else 0.0
        )

        policy_warnings: list[str] = []
        try:
            from .config import Config
            expected_budget = float(getattr(Config, "MONTHLY_BUDGET", 0.0) or 0.0)
            if expected_budget > 0 and abs(monthly_contribution - expected_budget) > 1.0:
                policy_warnings.append(
                    f"portfolio monthly_contribution ${monthly_contribution:,.0f} differs from Config.MONTHLY_BUDGET ${expected_budget:,.0f}"
                )
        except Exception as exc:
            logger.debug("Portfolio policy validation skipped: %s", exc)

        return {
            "as_of": data.get("last_updated") or self._now_iso()[:10],
            "cash_available": cash_available,
            "monthly_contribution": monthly_contribution,
            "total_value": total_value,
            "total_holdings_value": total_holdings_value,
            "total_cost_basis": total_cost_basis,
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_unrealized_pnl_pct": total_unrealized_pnl_pct,
            "allocation_cash_pct": allocation_cash_pct,
            "concentration_pct": concentration_pct,
            "positions": [p.__dict__ for p in normalized_positions],
            "notes": str(data.get("notes", "")).strip(),
            "policy_warnings": policy_warnings,
        }

    @staticmethod
    def _format_usd(value: float) -> str:
        return f"${value:,.2f}" if abs(value) >= 0.01 else "$0"

    def render_prompt_summary(self, state: dict[str, Any] | None = None) -> str:
        """Render compact portfolio summary text for prompt injection."""
        s = state or self.compute_state()
        as_of = str(s.get("as_of", "")).strip() or datetime.now(timezone.utc).strftime("%B %d, %Y")

        total_value = self._to_float(s.get("total_value"))
        cash = self._to_float(s.get("cash_available"))
        monthly = self._to_float(s.get("monthly_contribution"))
        positions = s.get("positions", []) if isinstance(s.get("positions"), list) else []
        policy_warnings = s.get("policy_warnings", []) if isinstance(s.get("policy_warnings"), list) else []
        policy_warning_text = ""
        if policy_warnings:
            policy_warning_text = "\nPolicy warnings: " + "; ".join(str(w) for w in policy_warnings[:3]) + "\n"

        if not positions:
            return (
                f"## PORTFOLIO STATE (as of {as_of})\n"
                f"Total Value: {self._format_usd(total_value)} (cash only)\n"
                f"Cash Available: {self._format_usd(cash)}\n"
                f"Monthly Contribution: {self._format_usd(monthly)}\n\n"
                "Holdings: None yet — this is month 1.\n\n"
                "Concentration: N/A\n"
                "Allocation: 100% cash\n\n"
                f"Notes: {s.get('notes') or 'First month of investing. Council should recommend initial positions.'}\n"
                f"{policy_warning_text}"
                "IPS: Max 15% per stock, moderate risk, index-first approach."
            )

        lines = [
            f"## PORTFOLIO STATE (as of {as_of})",
            f"Total Value: {self._format_usd(total_value)}",
            f"Cash Available: {self._format_usd(cash)}",
            f"Monthly Contribution: {self._format_usd(monthly)}",
            "",
            "Holdings:",
        ]

        top_positions = sorted(positions, key=lambda p: self._to_float(p.get("market_value")), reverse=True)
        for pos in top_positions[:8]:
            ticker = str(pos.get("ticker", "?")).upper()
            value = self._to_float(pos.get("market_value"))
            weight = self._to_float(pos.get("weight_pct"))
            pnl_pct = self._to_float(pos.get("unrealized_pnl_pct"))
            drift = pos.get("drift_pct")
            drift_text = f", drift {self._to_float(drift):+.1f}%" if drift is not None else ""
            lines.append(
                f"- {ticker}: {self._format_usd(value)} ({weight:.1f}%), unrealized {pnl_pct:+.1f}%{drift_text}"
            )

        lines.extend(
            [
                "",
                f"Concentration: top position {self._to_float(s.get('concentration_pct')):.1f}% of portfolio",
                f"Allocation: {self._to_float(s.get('allocation_cash_pct')):.1f}% cash / {100.0 - self._to_float(s.get('allocation_cash_pct')):.1f}% invested",
                f"Unrealized P&L: {self._format_usd(self._to_float(s.get('total_unrealized_pnl')))} ({self._to_float(s.get('total_unrealized_pnl_pct')):+.1f}%)",
            ]
        )

        notes = str(s.get("notes", "")).strip()
        if notes:
            lines.extend(["", f"Notes: {notes}"])
        if policy_warnings:
            lines.extend(["", "Policy warnings:"])
            lines.extend(f"- {w}" for w in policy_warnings[:3])

        return "\n".join(lines)

    def build_snapshot_payload(self, state: dict[str, Any], summary: str) -> dict[str, Any]:
        """Build normalized payload used for SQLite snapshot persistence."""
        holdings_min = []
        for pos in state.get("positions", []):
            if not isinstance(pos, dict):
                continue
            holdings_min.append(
                {
                    "ticker": str(pos.get("ticker", "")).upper(),
                    "asset_type": pos.get("asset_type", "stock"),
                    "market_value": self._to_float(pos.get("market_value")),
                    "weight_pct": round(self._to_float(pos.get("weight_pct")), 3),
                    "unrealized_pnl": round(self._to_float(pos.get("unrealized_pnl")), 3),
                    "unrealized_pnl_pct": round(self._to_float(pos.get("unrealized_pnl_pct")), 3),
                }
            )

        return {
            "timestamp": self._now_iso(),
            "total_value": round(self._to_float(state.get("total_value")), 4),
            "cash": round(self._to_float(state.get("cash_available")), 4),
            "holdings_json": json.dumps(holdings_min, ensure_ascii=True),
            "summary": summary,
        }

    def build_state_bundle(self) -> dict[str, Any]:
        """Convenience method returning computed state + summary + snapshot payload."""
        state = self.compute_state()
        summary = self.render_prompt_summary(state)
        snapshot = self.build_snapshot_payload(state, summary)
        return {"state": state, "summary": summary, "snapshot": snapshot}


def get_deployment_target(fear_greed: int | None, portfolio: dict[str, Any], config) -> dict[str, Any]:
    """Calculate real deployable cash plus Fear & Greed regime context.

    deployable_amount = min(broker cash when the Robinhood snapshot is fresh,
    monthly-budget remaining this calendar month), floored at $0. The regime
    target is returned as context (regime_target_deployable) but no longer
    zeroes the deployable amount.

    Returns dict with target_invested_pct, current_invested_pct,
    deployable_amount, deployment_urgency, regime/regime_label, total_nav,
    cash, and budget context derived from monthly_contribution. The whole
    dict is sizing context only (sizing_only=True).
    """
    import math as _math

    # Guard: None/NaN fear_greed defaults to 50 (Neutral)
    if fear_greed is None:
        fear_greed = 50
    try:
        fear_greed = int(fear_greed)
    except (TypeError, ValueError):
        fear_greed = 50

    # Guard: None/NaN portfolio values
    try:
        cash = float(portfolio.get("cash_available") or 0)
        if _math.isnan(cash) or _math.isinf(cash):
            cash = 0.0
    except (TypeError, ValueError):
        cash = 0.0
    try:
        monthly_contribution = float(portfolio.get("monthly_contribution") or 0)
        if _math.isnan(monthly_contribution) or _math.isinf(monthly_contribution):
            monthly_contribution = 0.0
    except (TypeError, ValueError):
        monthly_contribution = 0.0
    positions = portfolio.get("positions", []) or []
    position_count = len(positions)

    try:
        total_value = float(portfolio.get("total_value") or 0)
        if _math.isnan(total_value) or _math.isinf(total_value):
            total_value = 0.0
    except (TypeError, ValueError):
        total_value = 0.0

    if total_value <= 0:
        holdings_value = 0.0
        for pos in positions:
            if not isinstance(pos, dict):
                continue
            try:
                shares = float(pos.get("shares", pos.get("quantity", 0)) or 0)
                price = float(pos.get("current_price", pos.get("price", 0)) or 0)
                market_value = float(pos.get("market_value") or 0)
                cost_basis = float(pos.get("cost_basis") or 0)
            except (TypeError, ValueError):
                continue
            if market_value > 0:
                holdings_value += market_value
            elif shares > 0 and price > 0:
                holdings_value += shares * price
            elif cost_basis > 0:
                holdings_value += cost_basis
        total_value = cash + holdings_value

    invested = total_value - cash
    # Guard: division by zero
    current_invested_pct = (invested / total_value) if total_value > 0 else 0.0

    # Determine deployment target based on Fear & Greed
    if fear_greed < 20:
        target_deployed_pct = config.CASH_DEPLOY_EXTREME_FEAR
        regime_label = "EXTREME_FEAR"
        deployment_urgency = "HIGH"
    elif fear_greed < 40:
        target_deployed_pct = config.CASH_DEPLOY_FEAR
        regime_label = "FEAR"
        deployment_urgency = "MEDIUM"
    elif fear_greed <= 60:
        target_deployed_pct = config.CASH_DEPLOY_NEUTRAL
        regime_label = "NEUTRAL"
        deployment_urgency = "LOW"
    elif fear_greed <= 80:
        target_deployed_pct = config.CASH_DEPLOY_GREED
        regime_label = "GREED"
        deployment_urgency = "MINIMAL"
    else:
        target_deployed_pct = config.CASH_DEPLOY_EXTREME_GREED
        regime_label = "EXTREME_GREED"
        deployment_urgency = "NONE"

    # Regime-target framing is kept for CONTEXT only. The old formula
    # (min(cash, nav*target - invested)) zeroed deployable for any small
    # account already >target% invested, which forced every scan into
    # research-only mode despite real broker cash sitting idle.
    needed = max(0.0, total_value * target_deployed_pct - invested)
    regime_target_deployable = min(cash, needed)

    # Broker capacity is authoritative for live deployment. Total account
    # value sets the 90% exposure ceiling; buying power sets what can be spent
    # now. This distinction matters after sales, when cash can be unsettled.
    from .broker_capacity import load_broker_capacity

    broker_capacity = load_broker_capacity()
    if broker_capacity.get("usable"):
        broker_cash = float(broker_capacity.get("buying_power") or 0.0)
        broker_total_cash = float(broker_capacity.get("total_cash") or broker_cash)
        total_value = float(broker_capacity.get("total_account_value") or total_value)
        invested = float(broker_capacity.get("invested_value") or 0.0)
        current_invested_pct = (invested / total_value) if total_value > 0 else 0.0
        baseline = max(abs(broker_total_cash), abs(cash), 0.01)
        if abs(broker_total_cash - cash) / baseline > CASH_DISAGREEMENT_WARN_PCT:
            logger.warning(
                "Broker total cash $%.2f and portfolio.json cash_available $%.2f disagree by >%.0f%% — "
                "using broker capacity as authoritative; reconcile the ledger",
                broker_total_cash,
                cash,
                CASH_DISAGREEMENT_WARN_PCT * 100,
            )
        effective_cash = broker_cash
        cash_source = "broker_snapshot"
    else:
        broker_cash = None
        broker_total_cash = None
        effective_cash = cash
        cash_source = "portfolio_json_non_executable"

    monthly_budget = monthly_contribution if monthly_contribution > 0 else float(
        getattr(config, "MONTHLY_BUDGET", 0.0) or 0.0
    )
    if monthly_budget > 0:
        budget_remaining = get_monthly_budget_remaining(monthly_budget)
    else:
        budget_remaining = effective_cash

    if broker_capacity.get("usable"):
        # Sales reopen capacity automatically. A calendar-month gross-buy
        # counter must not consume proceeds that are safely back in the same
        # capped account; the 90% ceiling and live buying power are the bounds.
        deployable = float(broker_capacity.get("deployable_amount") or 0.0)
        binding_constraint = str(broker_capacity.get("binding_constraint") or "broker_capacity")
    elif bool(getattr(config, "ROBINHOOD_REQUIRE_FRESH_SNAPSHOT_FOR_REVIEW", True)):
        deployable = 0.0
        binding_constraint = "broker_snapshot_unavailable"
    else:
        deployable = max(0.0, min(effective_cash, budget_remaining))
        binding_constraint = "monthly_budget" if budget_remaining < effective_cash else "portfolio_cash"

    budget_cap_amount = (
        float(broker_capacity.get("exposure_ceiling") or 0.0)
        if broker_capacity.get("usable")
        else min(effective_cash, monthly_contribution) if monthly_contribution > 0 else effective_cash
    )
    budget_mode = "rolling_account_90pct_cap" if broker_capacity.get("usable") else "non_executable_fallback"

    available_slots = max(0, config.MAX_CONCURRENT_POSITIONS - position_count)

    return {
        "target_invested_pct": round(target_deployed_pct, 4),
        "current_invested_pct": round(current_invested_pct, 4),
        "deployable_amount": round(deployable, 2),
        "deployment_urgency": deployment_urgency,
        # Emit BOTH keys: council.py reads 'regime' for entry_regime while older
        # consumers read 'regime_label'.
        "regime": regime_label,
        "regime_label": regime_label,
        "regime_target_deployable": round(regime_target_deployable, 2),
        "total_nav": round(total_value, 2),
        "cash": round(effective_cash, 2),
        "ledger_cash": round(cash, 2),
        "broker_cash": round(broker_cash, 2) if broker_cash is not None else None,
        "broker_total_cash": round(broker_total_cash, 2) if broker_total_cash is not None else None,
        "unsettled_or_unspendable_cash": broker_capacity.get("unsettled_or_unspendable_cash"),
        "exposure_headroom": broker_capacity.get("exposure_headroom"),
        "broker_capacity_status": broker_capacity.get("status"),
        "broker_capacity_reasons": broker_capacity.get("reasons") or [],
        "cash_source": cash_source,
        "monthly_contribution": round(monthly_contribution, 2),
        "monthly_budget_remaining": round(budget_remaining, 2),
        "binding_constraint": binding_constraint,
        "budget_cap_amount": round(budget_cap_amount, 2),
        "budget_mode": budget_mode,
        "position_count": position_count,
        "available_slots": available_slots,
        "max_positions": config.MAX_CONCURRENT_POSITIONS,
        # Task-5 separation: everything in this dict is SIZING context. It must
        # never alter stock-quality judgment. Wave 2 rewires council prompts to
        # render sizing_only blocks separately from candidate-quality evidence.
        "sizing_only": True,
        "context_separation": {
            "sizing_fields": [
                "deployable_amount",
                "monthly_budget_remaining",
                "budget_cap_amount",
                "cash",
                "available_slots",
                "target_invested_pct",
            ],
            "quality_fields": [],
            "note": "Deployment/budget state informs position sizing only, not whether a stock is worth owning.",
        },
    }
