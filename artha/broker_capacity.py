"""Authoritative Robinhood account-capacity calculations.

The broker snapshot exposes three different cash/account concepts that must
not be mixed:

* total account value is the denominator for Artha's invested-percent limit;
* buying power is the amount Robinhood will let Artha spend now;
* total cash can include unsettled sale proceeds and is therefore not buying
  power in a cash account.

This module is intentionally deterministic and has no broker write tools.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _nested_number(value: Any, *keys: str) -> float | None:
    direct = _number(value)
    if direct is not None:
        return direct
    if not isinstance(value, dict):
        return None
    for key in keys:
        parsed = _number(value.get(key))
        if parsed is not None:
            return parsed
    return None


def _snapshot_time(snapshot: dict[str, Any]) -> datetime | None:
    raw = (
        snapshot.get("generated_at")
        or snapshot.get("generated_at_utc")
        or snapshot.get("synced_at")
        or snapshot.get("as_of")
    )
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _marked_positions_value(snapshot: dict[str, Any]) -> tuple[float, int]:
    total = 0.0
    unresolved = 0
    rows = snapshot.get("positions") or []
    if not isinstance(rows, list):
        return 0.0, 1
    for row in rows:
        if not isinstance(row, dict):
            continue
        quantity = _number(row.get("quantity") or row.get("shares")) or 0.0
        if quantity <= 0:
            continue
        market_value = _number(row.get("market_value") or row.get("equity") or row.get("value"))
        if market_value is not None and market_value >= 0:
            total += market_value
            continue
        price = _number(
            row.get("market_price")
            or row.get("current_price")
            or row.get("price")
            or row.get("average_buy_price")
            or row.get("average_price")
        )
        if price is None or price <= 0:
            unresolved += 1
            continue
        total += quantity * price
    return total, unresolved


def calculate_broker_capacity(
    snapshot: dict[str, Any] | None,
    *,
    max_invested_pct: float | None = None,
    min_deployable: float | None = None,
    now: datetime | None = None,
    max_age_minutes: int | None = None,
    require_fresh: bool = True,
) -> dict[str, Any]:
    """Return one reconciled view of account value, exposure and buying power."""
    cap_pct = float(Config.MAX_INVESTED_PCT if max_invested_pct is None else max_invested_pct)
    minimum = float(
        Config.SCAN_MIN_DEPLOYABLE_FOR_BUY_COUNCIL if min_deployable is None else min_deployable
    )
    age_limit = int(
        Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_MAX_AGE_MINUTES
        if max_age_minutes is None
        else max_age_minutes
    )
    result: dict[str, Any] = {
        "status": "WARN",
        "usable": False,
        "scan_buy_enabled": False,
        "max_invested_pct": cap_pct,
        "minimum_deployable": minimum,
        "reasons": [],
    }
    if not isinstance(snapshot, dict):
        result["reasons"] = ["Robinhood snapshot is missing or malformed."]
        return result

    generated_at = _snapshot_time(snapshot)
    age_minutes: float | None = None
    if generated_at is not None:
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        age_minutes = max(0.0, (current - generated_at).total_seconds() / 60.0)
    result["generated_at"] = generated_at.isoformat() if generated_at else None
    result["age_minutes"] = round(age_minutes, 3) if age_minutes is not None else None
    result["max_age_minutes"] = age_limit

    reasons: list[str] = []
    if require_fresh:
        if generated_at is None:
            reasons.append("Robinhood snapshot has no valid generation time.")
        elif age_minutes is not None and age_minutes > age_limit:
            reasons.append(
                f"Robinhood snapshot is stale ({age_minutes:.1f} minutes; maximum {age_limit})."
            )
        if snapshot.get("status") not in (None, "PASS"):
            reasons.append(f"Robinhood snapshot validation status is {snapshot.get('status')}.")
        if snapshot.get("fresh") is False:
            reasons.append("Robinhood snapshot is explicitly marked stale.")

    portfolio = snapshot.get("portfolio") if isinstance(snapshot.get("portfolio"), dict) else {}
    buying_power = _nested_number(
        portfolio.get("buying_power"),
        "buying_power",
        "unleveraged_buying_power",
        "amount",
    )
    cash = _nested_number(portfolio.get("cash"), "cash", "amount")
    total_value = _number(portfolio.get("total_value"))

    invested_value: float | None = None
    if total_value is not None and cash is not None:
        invested_value = max(0.0, total_value - cash)
    else:
        asset_keys = (
            "equity_value",
            "options_value",
            "crypto_value",
            "fixed_income_value",
            "mutual_funds_value",
            "futures_value",
            "event_contracts_value",
        )
        asset_values = [_number(portfolio.get(key)) for key in asset_keys]
        if any(value is not None for value in asset_values):
            invested_value = sum(max(0.0, value or 0.0) for value in asset_values)
        else:
            marked, unresolved = _marked_positions_value(snapshot)
            if unresolved:
                reasons.append(
                    "Cannot prove invested value because one or more positions lack a usable market mark."
                )
            else:
                invested_value = marked
        if total_value is None and invested_value is not None and cash is not None:
            total_value = invested_value + cash

    if buying_power is None or buying_power < 0:
        reasons.append("Robinhood snapshot is missing usable buying power.")
    if total_value is None or total_value <= 0:
        reasons.append("Robinhood snapshot is missing a positive total account value.")
    if invested_value is None or invested_value < 0:
        reasons.append("Robinhood snapshot is missing a usable invested value.")
    if total_value is not None and invested_value is not None and invested_value > total_value + 0.01:
        reasons.append("Robinhood invested value exceeds total account value.")

    result.update(
        {
            "total_account_value": round(total_value, 6) if total_value is not None else None,
            "invested_value": round(invested_value, 6) if invested_value is not None else None,
            "total_cash": round(cash, 6) if cash is not None else None,
            "buying_power": round(buying_power, 6) if buying_power is not None else None,
            "unsettled_or_unspendable_cash": (
                round(max(0.0, cash - buying_power), 6)
                if cash is not None and buying_power is not None
                else None
            ),
        }
    )
    if reasons:
        result["reasons"] = list(dict.fromkeys(reasons))
        return result

    assert total_value is not None and invested_value is not None and buying_power is not None
    invested_pct = invested_value / total_value
    exposure_ceiling = total_value * cap_pct
    exposure_headroom = max(0.0, exposure_ceiling - invested_value)
    deployable = max(0.0, min(buying_power, exposure_headroom))
    binding = "buying_power" if buying_power <= exposure_headroom else "90pct_exposure_ceiling"
    result.update(
        {
            "status": "PASS",
            "usable": True,
            "scan_buy_enabled": deployable + 0.000001 >= minimum,
            "invested_pct": invested_pct,
            "exposure_ceiling": round(exposure_ceiling, 6),
            "exposure_headroom": round(exposure_headroom, 6),
            "deployable_amount": round(deployable, 6),
            "binding_constraint": binding,
            "reasons": [],
        }
    )
    return result


def load_broker_capacity(
    path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load the configured read-only snapshot and calculate buy capacity."""
    target = Path(path or Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE).expanduser()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "WARN",
            "usable": False,
            "scan_buy_enabled": False,
            "path": str(target),
            "reasons": [f"Robinhood snapshot could not be loaded: {type(exc).__name__}: {exc}"],
        }
    result = calculate_broker_capacity(payload, **kwargs)
    result["path"] = str(target)
    return result
