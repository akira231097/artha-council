"""Read-only adapter for Artha's deterministic Robinhood snapshot handoff."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from artha_mcp.markets import normalize_instrument
from artha_mcp.models import (
    AccountSnapshot,
    BrokerCapabilities,
    MarketCode,
    OrderPreview,
    OrderRequest,
    OrderResult,
    PortfolioSnapshot,
    Position,
    Quote,
)
from artha_mcp.security import mask_identifier, redact

from .base import BrokerAdapter, CapabilityUnavailable
from .http import first_number

MAX_SNAPSHOT_BYTES = 10 * 1024 * 1024


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _symbol(row: dict[str, Any]) -> str:
    instrument = (
        row.get("instrument") if isinstance(row.get("instrument"), dict) else {}
    )
    return (
        str(row.get("symbol") or row.get("ticker") or instrument.get("symbol") or "")
        .upper()
        .strip()
    )


class SnapshotBrokerAdapter(BrokerAdapter):
    def __init__(self, path: Path, *, max_age_seconds: int = 600) -> None:
        self.path = path
        self.max_age_seconds = max_age_seconds

    @property
    def capabilities(self) -> BrokerCapabilities:
        return BrokerCapabilities(
            name="snapshot",
            market=MarketCode.US,
            account_read=True,
            portfolio_read=True,
            order_status=True,
            fractional_equities=True,
            notes=[
                "Reads Artha's deterministic Robinhood handoff file.",
                "Live quote, preview, and placement remain owned by the Robinhood/OpenClaw runner.",
            ],
        )

    def _load(self) -> dict[str, Any]:
        try:
            if self.path.stat().st_size > MAX_SNAPSHOT_BYTES:
                raise CapabilityUnavailable(
                    f"Broker snapshot exceeds {MAX_SNAPSHOT_BYTES} bytes"
                )
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise CapabilityUnavailable(
                f"Broker snapshot does not exist: {self.path}"
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise CapabilityUnavailable("Broker snapshot is unreadable") from exc
        if not isinstance(payload, dict):
            raise CapabilityUnavailable("Broker snapshot root must be an object")
        return payload

    async def health(self) -> dict[str, Any]:
        try:
            payload = self._load()
            generated_at = _parse_time(
                payload.get("generated_at") or payload.get("synced_at")
            )
            age = None
            if generated_at:
                age = max(0.0, (datetime.now(UTC) - generated_at).total_seconds())
            validation = (
                payload.get("validation")
                if isinstance(payload.get("validation"), dict)
                else {}
            )
            validation_status = str(validation.get("status") or "UNKNOWN").upper()
            fresh = (
                age is not None
                and age <= self.max_age_seconds
                and validation_status == "PASS"
            )
            return {
                "status": "PASS" if fresh else "WARN",
                "broker": "snapshot",
                "path": redact(str(self.path)),
                "generated_at": generated_at.isoformat() if generated_at else None,
                "age_seconds": round(age, 2) if age is not None else None,
                "fresh": fresh,
                "validation_status": validation_status,
            }
        except CapabilityUnavailable as exc:
            return {"status": "FAIL", "broker": "snapshot", "message": str(exc)}

    async def portfolio(self) -> PortfolioSnapshot:
        payload = self._load()
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        account = (
            payload.get("account")
            or payload.get("selected_account")
            or data.get("account")
            or {}
        )
        portfolio = payload.get("portfolio") or data.get("portfolio") or {}
        positions_raw = payload.get("positions") or data.get("positions") or []
        orders_raw = payload.get("orders") or data.get("orders") or []
        parsed_generated_at = _parse_time(
            payload.get("generated_at") or payload.get("synced_at")
        )
        generated_at = parsed_generated_at or datetime.now(UTC)
        age = (
            max(0.0, (datetime.now(UTC) - parsed_generated_at).total_seconds())
            if parsed_generated_at is not None
            else None
        )
        validation = (
            payload.get("validation")
            if isinstance(payload.get("validation"), dict)
            else {}
        )
        validation_status = str(validation.get("status") or "UNKNOWN").upper()
        snapshot_fresh = (
            age is not None
            and age <= self.max_age_seconds
            and validation_status == "PASS"
        )
        rows: list[Position] = []
        for raw in positions_raw if isinstance(positions_raw, list) else []:
            if not isinstance(raw, dict) or not _symbol(raw):
                continue
            instrument = normalize_instrument(_symbol(raw), market="US")
            quantity = first_number(raw, {"quantity", "shares", "qty"}) or 0.0
            average = first_number(
                raw, {"average_buy_price", "average_price", "avg_cost"}
            )
            last = first_number(
                raw, {"last_trade_price", "last_price", "price", "current_price"}
            )
            value = first_number(raw, {"market_value", "equity"})
            rows.append(
                Position(
                    instrument=instrument,
                    quantity=quantity,
                    average_price=average,
                    last_price=last,
                    market_value=value
                    if value is not None
                    else (quantity * last if last else None),
                    unrealized_pnl=first_number(
                        raw, {"unrealized_pnl", "unrealized_gain_loss"}
                    ),
                )
            )
        account_id = (
            account.get("account_number")
            or account.get("rhs_account_number")
            or account.get("id")
        )
        account_snapshot = AccountSnapshot(
            broker="robinhood_snapshot",
            account_id_masked=mask_identifier(account_id),
            currency="USD",
            buying_power=first_number(
                portfolio, {"buying_power", "withdrawable_amount"}
            ),
            cash=first_number(
                portfolio, {"cash", "cash_available", "cash_held_for_orders"}
            ),
            equity=first_number(
                portfolio,
                {
                    "equity",
                    "equity_value",
                    "total_equity",
                    "total_value",
                    "market_value",
                },
            ),
            generated_at=generated_at,
            fresh=snapshot_fresh,
        )
        warnings = [
            str(v)
            for v in payload.get("warnings", [])
            if isinstance(v, (str, int, float))
        ]
        warnings.extend(
            str(value)
            for value in validation.get("warnings", [])
            if isinstance(value, (str, int, float))
        )
        if validation_status != "PASS":
            warnings.append(
                f"Snapshot validation status is {validation_status}; broker-dependent actions must remain blocked."
            )
        if parsed_generated_at is None:
            warnings.append(
                "Snapshot generated_at is missing, invalid, or lacks a timezone; broker-dependent actions must remain blocked."
            )
        elif age is not None and age > self.max_age_seconds:
            warnings.append(
                f"Snapshot is stale: {age:.1f}s old; limit is {self.max_age_seconds}s."
            )
        terminal = {
            "filled",
            "complete",
            "completed",
            "cancelled",
            "canceled",
            "rejected",
            "failed",
        }
        open_orders = [
            redact(row)
            for row in orders_raw
            if isinstance(row, dict)
            and str(row.get("state") or row.get("status") or "").lower() not in terminal
        ]
        return PortfolioSnapshot(
            account=account_snapshot,
            positions=rows,
            open_orders=open_orders,
            warnings=warnings,
        )

    async def quote(self, instrument) -> Quote:
        raise CapabilityUnavailable("Snapshot adapter cannot prove a fresh live quote")

    async def preview(self, order: OrderRequest) -> OrderPreview:
        raise CapabilityUnavailable(
            "Snapshot adapter delegates exact-order review to the Robinhood/OpenClaw runner"
        )

    async def place(self, order: OrderRequest) -> OrderResult:
        raise CapabilityUnavailable("Snapshot adapter cannot place orders")

    async def orders(self) -> list[dict[str, Any]]:
        payload = self._load()
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        rows = payload.get("orders") or data.get("orders") or []
        return [redact(row) for row in rows if isinstance(row, dict)]
