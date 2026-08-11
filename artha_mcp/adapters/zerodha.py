"""Direct Zerodha Kite Connect adapter for Indian cash equities."""

from __future__ import annotations

from typing import Any

import httpx

from artha_mcp.markets import normalize_instrument
from artha_mcp.models import (
    AccountSnapshot,
    BrokerCapabilities,
    InstrumentRef,
    MarketCode,
    OrderPreview,
    OrderRequest,
    OrderResult,
    PortfolioSnapshot,
    Position,
    Quote,
)
from artha_mcp.security import mask_identifier, redact
from artha_mcp.settings import MCPSettings

from .base import AdapterError, BrokerAdapter, deterministic_order_tag
from .http import HTTPBrokerMixin, first_number, parse_broker_timestamp


def _equity_cash_available(payload: dict[str, Any]) -> float | None:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    equity = data.get("equity") if isinstance(data.get("equity"), dict) else {}
    available = (
        equity.get("available") if isinstance(equity.get("available"), dict) else {}
    )
    for key in ("live_balance", "cash"):
        value = _number(available.get(key))
        if value is not None:
            return value
    return _number(equity.get("net"))


def _first_response_row(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return {}


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _order_charges_total(payload: dict[str, Any]) -> float | None:
    charges = _first_response_row(payload).get("charges")
    return _number(charges.get("total")) if isinstance(charges, dict) else None


def _api_success(payload: dict[str, Any]) -> bool:
    return str(payload.get("status") or "").lower() == "success"


def _require_api_success(payload: Any, operation: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or not _api_success(payload):
        raise AdapterError(f"Zerodha {operation} did not return status=success")
    return payload


_DELIVERY_PRODUCTS = {"CNC", "D", "DELIVERY"}
_TERMINAL_ORDER_STATUSES = {
    "COMPLETE",
    "COMPLETED",
    "CANCELLED",
    "CANCELED",
    "REJECTED",
    "FAILED",
}


def _zerodha_symbol(row: dict[str, Any]) -> str:
    return str(row.get("tradingsymbol") or row.get("symbol") or "").upper()


def _zerodha_exchange(row: dict[str, Any]) -> str:
    return str(row.get("exchange") or "NSE").upper()


class ZerodhaBrokerAdapter(HTTPBrokerMixin, BrokerAdapter):
    def __init__(
        self, settings: MCPSettings, *, client: httpx.AsyncClient | None = None
    ) -> None:
        super().__init__(client=client)
        self.settings = settings

    @property
    def capabilities(self) -> BrokerCapabilities:
        return BrokerCapabilities(
            name="zerodha",
            market=MarketCode.INDIA,
            account_read=True,
            portfolio_read=True,
            instrument_search=True,
            quote_read=True,
            order_preview=True,
            order_place=True,
            order_status=True,
            fractional_equities=False,
            notes=[
                "Whole-share cash-and-carry equity limit orders only.",
                "Exchange and tradingsymbol are mandatory.",
                "Live API placement requires a broker-registered static IP.",
            ],
        )

    def _headers(self) -> dict[str, str]:
        if not self.settings.zerodha_api_key or not self.settings.zerodha_access_token:
            raise AdapterError("KITE_API_KEY/KITE_ACCESS_TOKEN are not configured")
        return {
            "Accept": "application/json",
            "X-Kite-Version": "3",
            "Authorization": f"token {self.settings.zerodha_api_key}:{self.settings.zerodha_access_token}",
        }

    async def health(self) -> dict[str, Any]:
        payload = await self._request(
            "GET",
            f"{self.settings.zerodha_base_url}/user/profile",
            headers=self._headers(),
        )
        _require_api_success(payload, "profile request")
        data = (
            payload.get("data")
            if isinstance(payload, dict) and isinstance(payload.get("data"), dict)
            else {}
        )
        return {
            "status": "PASS",
            "broker": "zerodha",
            "user_id_masked": mask_identifier(data.get("user_id")),
        }

    async def _margins(self) -> dict[str, Any]:
        payload = await self._request(
            "GET",
            f"{self.settings.zerodha_base_url}/user/margins",
            headers=self._headers(),
        )
        return _require_api_success(payload, "funds request")

    async def search_instruments(
        self, query: str, *, exchange: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        symbol = str(query or "").upper().strip()
        selected = str(exchange or "ALL").upper().strip()
        if selected not in {"ALL", "NSE", "BSE"}:
            raise AdapterError(
                "Zerodha equity search exchange must be NSE, BSE, or ALL"
            )
        exchanges = ("NSE", "BSE") if selected == "ALL" else (selected,)
        instruments: list[InstrumentRef] = []
        for venue in exchanges:
            try:
                instruments.append(
                    normalize_instrument(
                        symbol,
                        market="IN",
                        exchange=venue,
                        broker_instrument_id=f"{venue}:{symbol}",
                    )
                )
            except ValueError as exc:
                raise AdapterError(
                    "Zerodha instrument lookup requires an exact equity trading symbol"
                ) from exc
        keys = [instrument.broker_instrument_id for instrument in instruments]
        payload = await self._request(
            "GET",
            f"{self.settings.zerodha_base_url}/quote",
            headers=self._headers(),
            params=[("i", key) for key in keys if key],
        )
        _require_api_success(payload, "instrument quote verification")
        data = (
            payload.get("data")
            if isinstance(payload, dict) and isinstance(payload.get("data"), dict)
            else {}
        )
        results: list[dict[str, Any]] = []
        for instrument in instruments:
            key = instrument.broker_instrument_id or ""
            row = data.get(key) if isinstance(data.get(key), dict) else None
            if row is None:
                continue
            results.append(
                {
                    "instrument": instrument.model_dump(mode="json"),
                    "last_price": _number(row.get("last_price")),
                    "instrument_token": str(row.get("instrument_token") or "")[:32],
                    "broker_verified": True,
                }
            )
        return results[: max(1, min(int(limit), 20))]

    async def portfolio(self) -> PortfolioSnapshot:
        profile = await self._request(
            "GET",
            f"{self.settings.zerodha_base_url}/user/profile",
            headers=self._headers(),
        )
        _require_api_success(profile, "profile request")
        margins = await self._margins()
        holdings = await self._request(
            "GET",
            f"{self.settings.zerodha_base_url}/portfolio/holdings",
            headers=self._headers(),
        )
        _require_api_success(holdings, "holdings request")
        positions = await self._request(
            "GET",
            f"{self.settings.zerodha_base_url}/portfolio/positions",
            headers=self._headers(),
        )
        _require_api_success(positions, "positions request")
        orders = await self.orders()
        profile_data = (
            profile.get("data")
            if isinstance(profile, dict) and isinstance(profile.get("data"), dict)
            else {}
        )
        raw_holdings = holdings.get("data") if isinstance(holdings, dict) else []
        positions_data = positions.get("data") if isinstance(positions, dict) else {}
        raw_positions = (
            positions_data.get("net") if isinstance(positions_data, dict) else []
        )
        components: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for source, raw_rows in (
            ("holding", raw_holdings),
            ("position", raw_positions),
        ):
            for raw in raw_rows if isinstance(raw_rows, list) else []:
                if not isinstance(raw, dict):
                    continue
                symbol = _zerodha_symbol(raw)
                exchange = _zerodha_exchange(raw)
                quantity = float(first_number(raw, {"quantity"}) or 0.0)
                if not symbol or quantity == 0 or exchange not in {"NSE", "BSE"}:
                    continue
                product = str(raw.get("product") or "").upper()
                bucket = (
                    "DELIVERY"
                    if source == "holding" or product in _DELIVERY_PRODUCTS
                    else f"POSITION:{product or 'UNCLASSIFIED'}"
                )
                components.setdefault((exchange, symbol, bucket), []).append(
                    {**raw, "_artha_source": source}
                )
        rows: list[Position] = []
        netted_count = 0
        separate_position_count = 0
        for (exchange, symbol, bucket), parts in components.items():
            quantity = sum(
                float(first_number(raw, {"quantity"}) or 0.0) for raw in parts
            )
            if quantity == 0:
                continue
            if bucket == "DELIVERY" and len(parts) > 1:
                netted_count += 1
            if bucket != "DELIVERY":
                separate_position_count += 1
            instrument = normalize_instrument(
                symbol,
                market="IN",
                exchange=exchange,
                broker_instrument_id=f"{exchange}:{symbol}",
            )
            raw = parts[-1]
            last = next(
                (
                    value
                    for part in reversed(parts)
                    if (value := first_number(part, {"last_price"})) is not None
                ),
                None,
            )
            pnl_values = [
                value
                for part in parts
                if (value := first_number(part, {"pnl", "unrealised"})) is not None
            ]
            rows.append(
                Position(
                    instrument=instrument,
                    quantity=quantity,
                    average_price=first_number(raw, {"average_price"})
                    if len(parts) == 1
                    else None,
                    last_price=last,
                    market_value=quantity * last if last is not None else None,
                    unrealized_pnl=sum(pnl_values) if pnl_values else None,
                    product=(
                        str(raw.get("product") or "CNC")
                        if len(parts) == 1
                        else "NETTED_DELIVERY"
                    ),
                    source_components=len(parts),
                )
            )
        buying_power = _equity_cash_available(margins)
        equity = sum(position.market_value or 0.0 for position in rows) + (
            buying_power or 0.0
        )
        return PortfolioSnapshot(
            account=AccountSnapshot(
                broker="zerodha",
                account_id_masked=mask_identifier(profile_data.get("user_id")),
                currency="INR",
                buying_power=buying_power,
                cash=buying_power,
                equity=equity,
                fresh=True,
            ),
            positions=rows,
            open_orders=[
                row
                for row in orders
                if str(row.get("status") or "").upper()
                not in {"COMPLETE", "CANCELLED", "REJECTED"}
            ],
            warnings=(
                (
                    [
                        f"Netted settled holdings and same-day positions for {netted_count} instrument(s); mixed-component average price is omitted."
                    ]
                    if netted_count
                    else []
                )
                + (
                    [
                        f"Kept {separate_position_count} non-delivery or unclassified same-day position(s) separate from delivery holdings."
                    ]
                    if separate_position_count
                    else []
                )
            ),
        )

    async def quote(self, instrument: InstrumentRef) -> Quote:
        if instrument.market != MarketCode.INDIA:
            raise AdapterError("Zerodha adapter supports Indian instruments only")
        expected_key = f"{instrument.exchange}:{instrument.symbol}"
        if (
            instrument.broker_instrument_id
            and instrument.broker_instrument_id.upper() != expected_key
        ):
            raise AdapterError(
                "Zerodha broker instrument id must exactly match exchange:tradingsymbol"
            )
        key = expected_key
        payload = await self._request(
            "GET",
            f"{self.settings.zerodha_base_url}/quote",
            headers=self._headers(),
            params=[("i", key)],
        )
        _require_api_success(payload, "quote request")
        data = (
            payload.get("data")
            if isinstance(payload, dict) and isinstance(payload.get("data"), dict)
            else {}
        )
        row = data.get(key) if isinstance(data.get(key), dict) else {}
        depth = row.get("depth") if isinstance(row.get("depth"), dict) else {}
        buys = depth.get("buy") if isinstance(depth.get("buy"), list) else []
        sells = depth.get("sell") if isinstance(depth.get("sell"), list) else []
        bid = (
            first_number(buys[0], {"price"})
            if buys and isinstance(buys[0], dict)
            else None
        )
        ask = (
            first_number(sells[0], {"price"})
            if sells and isinstance(sells[0], dict)
            else None
        )
        parsed_time = parse_broker_timestamp(
            row.get("timestamp") or row.get("last_trade_time"),
            default_timezone="Asia/Kolkata",
        )
        ohlc = row.get("ohlc") if isinstance(row.get("ohlc"), dict) else {}
        return Quote(
            instrument=instrument,
            bid=bid,
            ask=ask,
            last=first_number(row, {"last_price"}),
            previous_close=first_number(ohlc, {"close"}),
            volume=first_number(row, {"volume"}),
            timestamp=parsed_time,
            source="zerodha",
            fresh=parsed_time is not None,
            broker_identity_confirmed=bool(row),
        )

    @staticmethod
    def _whole_quantity(order: OrderRequest) -> int:
        if (
            order.notional is not None
            or order.quantity is None
            or not float(order.quantity).is_integer()
        ):
            raise AdapterError(
                "Zerodha Indian cash equities require a whole-share quantity"
            )
        return int(order.quantity)

    def client_order_key(self, order: OrderRequest) -> str:
        return deterministic_order_tag(str(order.tag or order.action_id), max_length=20)

    @staticmethod
    def _matches_order_instrument(row: dict[str, Any], order: OrderRequest) -> bool:
        return bool(
            _zerodha_symbol(row) == order.instrument.symbol
            and _zerodha_exchange(row) == order.instrument.exchange
        )

    async def _delivery_quantity(self, order: OrderRequest) -> float:
        holdings = await self._request(
            "GET",
            f"{self.settings.zerodha_base_url}/portfolio/holdings",
            headers=self._headers(),
        )
        _require_api_success(holdings, "holdings request")
        positions = await self._request(
            "GET",
            f"{self.settings.zerodha_base_url}/portfolio/positions",
            headers=self._headers(),
        )
        _require_api_success(positions, "positions request")
        orders = await self.orders()
        holding_rows = holdings.get("data") if isinstance(holdings, dict) else []
        positions_data = positions.get("data") if isinstance(positions, dict) else {}
        position_rows = (
            positions_data.get("net") if isinstance(positions_data, dict) else []
        )
        holding_iter = holding_rows if isinstance(holding_rows, list) else []
        settled = sum(
            float(first_number(row, {"quantity"}) or 0.0)
            for row in holding_iter
            if isinstance(row, dict) and self._matches_order_instrument(row, order)
        )
        same_day = 0.0
        for row in position_rows if isinstance(position_rows, list) else []:
            if not isinstance(row, dict) or not self._matches_order_instrument(
                row, order
            ):
                continue
            quantity = first_number(row, {"quantity"})
            product = str(row.get("product") or "").upper()
            if quantity is not None and (
                product in _DELIVERY_PRODUCTS or (not product and quantity < 0)
            ):
                same_day += quantity
        pending_sells = 0.0
        for row in orders:
            if not self._matches_order_instrument(row, order):
                continue
            if str(row.get("transaction_type") or "").upper() != "SELL":
                continue
            if str(row.get("status") or "").upper() in _TERMINAL_ORDER_STATUSES:
                continue
            product = str(row.get("product") or "").upper()
            if product and product not in _DELIVERY_PRODUCTS:
                continue
            pending = _number(row.get("pending_quantity"))
            if pending is None:
                pending = max(
                    0.0,
                    (_number(row.get("quantity")) or 0.0)
                    - (_number(row.get("filled_quantity")) or 0.0),
                )
            pending_sells += max(0.0, pending)
        return max(0.0, settled + same_day - pending_sells)

    @staticmethod
    def _margin_order(order: OrderRequest, quantity: int) -> dict[str, Any]:
        return {
            "exchange": order.instrument.exchange,
            "tradingsymbol": order.instrument.symbol,
            "transaction_type": order.side.upper(),
            "variety": "regular",
            "product": "CNC",
            "order_type": order.order_type.upper(),
            "quantity": quantity,
            "price": float(order.limit_price or 0),
            "trigger_price": 0,
        }

    async def preview(self, order: OrderRequest) -> OrderPreview:
        quantity = self._whole_quantity(order)
        if order.order_type != "limit":
            raise AdapterError(
                "Indian API execution is restricted to protected limit orders"
            )
        quote = await self.quote(order.instrument)
        market_price = quote.ask if order.side == "buy" else quote.bid
        price = (
            order.limit_price
            if order.order_type == "limit"
            else market_price or quote.last
        )
        reasons: list[str] = []
        demat_sell_proven = bool(
            order.side == "buy" or self.settings.india_demat_sell_authorized
        )
        if not self.settings.india_static_ip_registered:
            reasons.append(
                "A broker-registered static IP has not been attested for Indian API order placement."
            )
        if not demat_sell_proven:
            reasons.append(
                "Indian delivery sell authorization has not been attested for this deployment."
            )
        identity_confirmed = quote.model_extra.get("broker_identity_confirmed") is True
        if not identity_confirmed:
            reasons.append(
                "Zerodha quote did not confirm the exact exchange:tradingsymbol key."
            )
        if price is None or price <= 0:
            reasons.append("No usable broker price was returned.")
        if quote.spread_pct is None:
            reasons.append("Broker bid/ask depth is missing.")
        elif quote.spread_pct > self.settings.max_spread_pct:
            reasons.append(
                f"Spread {quote.spread_pct:.4%} exceeds limit {self.settings.max_spread_pct:.4%}."
            )
        if (
            order.side == "buy"
            and order.max_price is not None
            and price is not None
            and price > order.max_price
        ):
            reasons.append("Broker price is above the approved maximum price.")
        if (
            order.side == "sell"
            and order.min_price is not None
            and price is not None
            and price < order.min_price
        ):
            reasons.append("Broker price is below the approved minimum price.")
        estimated = quantity * price if price else None
        margins = await self._margins()
        buying_power = _equity_cash_available(margins)
        if order.side == "buy" and buying_power is None:
            reasons.append("Broker cash available to trade was not returned.")
        contract = self._margin_order(order, quantity)
        margin_payload = await self._request(
            "POST",
            f"{self.settings.zerodha_base_url}/margins/orders",
            headers={**self._headers(), "Content-Type": "application/json"},
            json=[contract],
        )
        charges_payload = (
            await self._request(
                "POST",
                f"{self.settings.zerodha_base_url}/charges/orders",
                headers={**self._headers(), "Content-Type": "application/json"},
                json=[
                    {
                        **contract,
                        "order_id": order.action_id[:20],
                        "average_price": float(price or 0),
                    }
                ],
            )
            if price
            else {}
        )
        required_margin = _number(_first_response_row(margin_payload).get("total"))
        margin_proven = _api_success(margin_payload) and required_margin is not None
        if not margin_proven:
            reasons.append("Broker margin preview did not return an explicit total.")
        position_proof = order.side == "buy"
        if order.side == "sell":
            held = await self._delivery_quantity(order)
            position_proof = held >= quantity
            if not position_proof:
                reasons.append(
                    f"Broker holdings show {held:g} shares, below the requested sell quantity {quantity:g}."
                )
        charges = _order_charges_total(charges_payload)
        charges_proven = _api_success(charges_payload) and charges is not None
        if not charges_proven:
            reasons.append("Broker charges preview did not return an explicit total.")
        if (
            order.side == "buy"
            and required_margin is not None
            and charges is not None
            and buying_power is not None
            and required_margin + charges > buying_power
        ):
            reasons.append(
                "Available broker funds do not cover required margin plus charges."
            )
        return OrderPreview(
            broker="zerodha",
            action_id=order.action_id,
            passed=not reasons,
            estimated_value=estimated,
            estimated_fees=charges,
            buying_power=buying_power,
            quote=quote,
            reasons=reasons,
            broker_proof={
                "instrument": identity_confirmed,
                "quote": quote.bid is not None and quote.ask is not None,
                "funds": buying_power is not None,
                "position": position_proof,
                "order_preview": margin_proven and charges_proven,
                "margin": margin_proven,
                "charges": charges_proven,
                "static_ip": self.settings.india_static_ip_registered,
                "demat_sell_authorized": demat_sell_proven,
            },
        )

    async def place(self, order: OrderRequest) -> OrderResult:
        quantity = self._whole_quantity(order)
        if not self.settings.india_static_ip_registered:
            raise AdapterError(
                "Indian API placement is blocked until a static IP is registered with the broker"
            )
        if order.side == "sell" and not self.settings.india_demat_sell_authorized:
            raise AdapterError(
                "Indian delivery sells are blocked until demat authorization is attested"
            )
        if order.order_type != "limit":
            raise AdapterError(
                "Indian API execution is restricted to protected limit orders"
            )
        expected_key = f"{order.instrument.exchange}:{order.instrument.symbol}"
        if order.instrument.broker_instrument_id != expected_key:
            raise AdapterError(
                "Zerodha placement requires an exact exchange:tradingsymbol broker id"
            )
        form = {
            "tradingsymbol": order.instrument.symbol,
            "exchange": order.instrument.exchange,
            "transaction_type": order.side.upper(),
            "order_type": order.order_type.upper(),
            "quantity": quantity,
            "product": "CNC",
            "validity": order.time_in_force.upper(),
            "price": float(order.limit_price or 0),
            "trigger_price": 0,
            "market_protection": 0,
            "tag": self.client_order_key(order),
        }
        payload = await self._request(
            "POST",
            f"{self.settings.zerodha_base_url}/orders/regular",
            headers=self._headers(),
            data=form,
        )
        data = (
            payload.get("data")
            if isinstance(payload, dict) and isinstance(payload.get("data"), dict)
            else {}
        )
        order_id = str(data.get("order_id") or "")
        accepted = bool(order_id) and _api_success(payload)
        return OrderResult(
            broker="zerodha",
            action_id=order.action_id,
            accepted=accepted,
            broker_order_id=order_id or None,
            status="submitted" if accepted else "rejected",
            message="Broker accepted the order for processing."
            if accepted
            else "Broker did not explicitly accept the order with a valid order id.",
            broker_response=redact(payload),
        )

    async def orders(self) -> list[dict[str, Any]]:
        payload = await self._request(
            "GET", f"{self.settings.zerodha_base_url}/orders", headers=self._headers()
        )
        _require_api_success(payload, "order-book request")
        data = payload.get("data") if isinstance(payload, dict) else []
        return (
            [redact(row) for row in data if isinstance(row, dict)]
            if isinstance(data, list)
            else []
        )
