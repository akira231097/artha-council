from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from artha_mcp.adapters.base import BrokerAdapter
from artha_mcp.models import (
    AccessMode,
    AccountSnapshot,
    BrokerCapabilities,
    BrokerName,
    InstrumentRef,
    MarketCode,
    OrderPreview,
    OrderRequest,
    OrderResult,
    PortfolioSnapshot,
    Quote,
)
from artha_mcp.settings import MCPSettings


def test_settings(root: Path, **changes: Any) -> MCPSettings:
    data = root / "data"
    base = MCPSettings.from_env(root=root)
    values = {
        "data_dir": data,
        "db_path": data / "artha.db",
        "portfolio_path": data / "portfolio.json",
        "snapshot_path": data / "robinhood" / "latest_snapshot.json",
        "access_mode": AccessMode.TRADING,
        "operations_enabled": True,
        "trading_enabled": True,
        "kill_switch": False,
        "broker": BrokerName.PLUGIN,
        "broker_plugin": "tests.mcp_helpers:fake_factory",
        "notification": "none",
        "research_mode": "host_orchestrated",
        "max_order_value": 100.0,
        "max_daily_order_value": 200.0,
        "max_daily_orders": 5,
        "quote_max_age_seconds": 30,
        "india_static_ip_registered": True,
        "india_demat_sell_authorized": True,
    }
    values.update(changes)
    return replace(base, **values)


def us_instrument(symbol: str = "TEST") -> InstrumentRef:
    return InstrumentRef(
        market=MarketCode.US,
        symbol=symbol,
        exchange="US",
        currency="USD",
        research_symbol=symbol,
    )


def india_instrument(symbol: str = "RELIANCE") -> InstrumentRef:
    return InstrumentRef(
        market=MarketCode.INDIA,
        symbol=symbol,
        exchange="NSE",
        currency="INR",
        research_symbol=f"{symbol}.NS",
        broker_instrument_id=f"NSE_EQ|{symbol}",
    )


class FakeBroker(BrokerAdapter):
    def __init__(
        self,
        *,
        now: datetime | None = None,
        market: MarketCode = MarketCode.US,
        sandbox: bool = True,
        order_status: bool = True,
    ) -> None:
        self.now = now or datetime.now(UTC)
        self.market = market
        self.sandbox = sandbox
        self.order_status = order_status
        self.preview_effects: list[OrderPreview | Exception] = []
        self.order_rows: list[dict[str, Any]] = []
        self.place_exception: Exception | None = None
        self.preview_calls = 0
        self.place_calls = 0

    @property
    def capabilities(self) -> BrokerCapabilities:
        return BrokerCapabilities(
            name="fake",
            market=self.market,
            account_read=True,
            portfolio_read=True,
            quote_read=True,
            order_preview=True,
            order_place=True,
            order_status=self.order_status,
            fractional_equities=self.market == MarketCode.US,
            sandbox=self.sandbox,
        )

    async def portfolio(self) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            account=AccountSnapshot(broker="fake", currency="USD", buying_power=1000)
        )

    async def quote(self, instrument: InstrumentRef) -> Quote:
        return Quote(
            instrument=instrument,
            bid=99.95,
            ask=100.05,
            last=100,
            timestamp=self.now,
            source="fake",
            fresh=True,
        )

    def good_preview(self, order: OrderRequest) -> OrderPreview:
        quote = Quote(
            instrument=order.instrument,
            bid=99.95,
            ask=100.05,
            last=100,
            timestamp=self.now,
            source="fake",
            fresh=True,
        )
        value = (
            order.notional
            if order.notional is not None
            else float(order.quantity or 0) * 100.05
        )
        return OrderPreview(
            broker="fake",
            action_id=order.action_id,
            passed=True,
            estimated_value=value,
            buying_power=1000,
            quote=quote,
            broker_proof={
                "instrument": True,
                "quote": True,
                "funds": True,
                "position": True,
                "order_preview": True,
            },
        )

    async def preview(self, order: OrderRequest) -> OrderPreview:
        self.preview_calls += 1
        if self.preview_effects:
            effect = self.preview_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            return effect
        return self.good_preview(order)

    async def place(self, order: OrderRequest) -> OrderResult:
        self.place_calls += 1
        if self.place_exception:
            raise self.place_exception
        return OrderResult(
            broker="fake",
            action_id=order.action_id,
            accepted=True,
            broker_order_id=f"ord-{order.action_id}",
            status="submitted",
            message="accepted",
        )

    async def orders(self) -> list[dict[str, Any]]:
        return list(self.order_rows)


def fake_factory(_settings: MCPSettings) -> FakeBroker:
    return FakeBroker()
