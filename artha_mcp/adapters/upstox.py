"""Direct Upstox adapter for Indian cash equities."""

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

from .base import (
    AdapterError,
    BrokerAdapter,
    CapabilityUnavailable,
    deterministic_order_tag,
)
from .http import HTTPBrokerMixin, first_dict, first_number, parse_broker_timestamp


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _api_success(payload: dict[str, Any]) -> bool:
    return str(payload.get("status") or "").lower() == "success"


def _require_api_success(payload: Any, operation: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or not _api_success(payload):
        raise AdapterError(f"Upstox {operation} did not return status=success")
    return payload


def _cash_available_to_trade(payload: dict[str, Any]) -> float | None:
    """Read Upstox v3 cash funds explicitly, with a documented v2 fallback."""
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    available = (
        data.get("available_to_trade")
        if isinstance(data.get("available_to_trade"), dict)
        else {}
    )
    cash_bucket = (
        available.get("cash_available_to_trade")
        if isinstance(available.get("cash_available_to_trade"), dict)
        else {}
    )
    value = _number(cash_bucket.get("total"))
    if value is not None:
        return value
    equity = data.get("equity") if isinstance(data.get("equity"), dict) else {}
    return _number(equity.get("available_margin"))


def _brokerage_total(payload: dict[str, Any]) -> float | None:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    charges = data.get("charges") if isinstance(data.get("charges"), dict) else {}
    return _number(charges.get("total"))


def _required_margin(payload: dict[str, Any]) -> float | None:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    value = _number(data.get("final_margin"))
    return value if value is not None else _number(data.get("required_margin"))


_DELIVERY_PRODUCTS = {"D", "DELIVERY", "CNC"}
_TERMINAL_ORDER_STATUSES = {
    "complete",
    "completed",
    "cancelled",
    "canceled",
    "rejected",
    "failed",
}


def _upstox_symbol(row: dict[str, Any]) -> str:
    return str(
        row.get("trading_symbol") or row.get("tradingsymbol") or row.get("symbol") or ""
    ).upper()


def _upstox_exchange(row: dict[str, Any]) -> str:
    value = str(row.get("exchange") or "").upper()
    if value.startswith("NSE"):
        return "NSE"
    if value.startswith("BSE"):
        return "BSE"
    return value


class UpstoxBrokerAdapter(HTTPBrokerMixin, BrokerAdapter):
    def __init__(
        self, settings: MCPSettings, *, client: httpx.AsyncClient | None = None
    ) -> None:
        super().__init__(client=client)
        self.settings = settings

    @property
    def capabilities(self) -> BrokerCapabilities:
        sandbox = self.settings.upstox_sandbox
        return BrokerCapabilities(
            name="upstox",
            market=MarketCode.INDIA,
            account_read=True,
            portfolio_read=True,
            instrument_search=True,
            quote_read=True,
            order_preview=True,
            order_place=True,
            order_status=not sandbox,
            fractional_equities=False,
            sandbox=sandbox,
            notes=[
                "Whole-share cash-equity delivery limit orders only.",
                "Broker instrument_token is mandatory.",
                "Live API placement requires a broker-registered static IP.",
                (
                    "Sandbox placement is submission-only; Upstox does not expose "
                    "sandbox order-book or fill reconciliation."
                    if sandbox
                    else "Live order-book status supports duplicate checks and reconciliation."
                ),
            ],
        )

    def _headers(self) -> dict[str, str]:
        if not self.settings.upstox_access_token:
            raise AdapterError("UPSTOX_ACCESS_TOKEN is not configured")
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.upstox_access_token}",
        }

    def _sandbox_headers(self) -> dict[str, str]:
        if not self.settings.upstox_sandbox_access_token:
            raise AdapterError("UPSTOX_SANDBOX_ACCESS_TOKEN is not configured")
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.upstox_sandbox_access_token}",
        }

    async def health(self) -> dict[str, Any]:
        profile = await self._request(
            "GET",
            f"{self.settings.upstox_base_url}/v2/user/profile",
            headers=self._headers(),
        )
        _require_api_success(profile, "profile request")
        data = profile.get("data") if isinstance(profile, dict) else {}
        return {
            "status": "PASS",
            "broker": "upstox",
            "sandbox": self.settings.upstox_sandbox,
            "user_id_masked": mask_identifier(
                (data or {}).get("user_id") if isinstance(data, dict) else None
            ),
        }

    async def _funds(self) -> dict[str, Any]:
        payload = await self._request(
            "GET",
            f"{self.settings.upstox_base_url}/v3/user/get-funds-and-margin",
            headers={**self._headers(), "Api-Version": "3.0"},
        )
        return _require_api_success(payload, "funds request")

    async def search_instruments(
        self, query: str, *, exchange: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        wanted = str(query or "").strip()
        if not wanted or len(wanted) > 50:
            raise AdapterError("Upstox instrument query must contain 1-50 characters")
        selected_exchange = str(exchange or "ALL").upper().strip()
        if selected_exchange not in {"ALL", "NSE", "BSE"}:
            raise AdapterError("Upstox equity search exchange must be NSE, BSE, or ALL")
        page_size = max(1, min(int(limit), 30))
        payload = await self._request(
            "GET",
            f"{self.settings.upstox_base_url}/v2/instruments/search",
            headers=self._headers(),
            params={
                "query": wanted,
                "exchanges": selected_exchange,
                "segments": "EQ",
                "page_number": 1,
                "records": page_size,
            },
        )
        _require_api_success(payload, "instrument search")
        data = payload.get("data") if isinstance(payload, dict) else []
        results: list[dict[str, Any]] = []
        for row in data if isinstance(data, list) else []:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("trading_symbol") or "").upper().strip()
            row_exchange = str(row.get("exchange") or "").upper().strip()
            instrument_key = str(row.get("instrument_key") or "").strip()
            segment = str(row.get("segment") or "").upper().strip()
            if (
                not symbol
                or row_exchange not in {"NSE", "BSE"}
                or not instrument_key
                or segment not in {"NSE_EQ", "BSE_EQ"}
            ):
                continue
            try:
                instrument = normalize_instrument(
                    symbol,
                    market="IN",
                    exchange=row_exchange,
                    broker_instrument_id=instrument_key,
                )
            except ValueError:
                continue
            results.append(
                {
                    "instrument": instrument.model_dump(mode="json"),
                    "name": str(row.get("name") or "")[:160],
                    "isin": str(row.get("isin") or "")[:32],
                    "tick_size": _number(row.get("tick_size")),
                    "lot_size": _number(row.get("lot_size")),
                    "broker_verified": True,
                }
            )
        return results[:page_size]

    async def portfolio(self) -> PortfolioSnapshot:
        profile = await self._request(
            "GET",
            f"{self.settings.upstox_base_url}/v2/user/profile",
            headers=self._headers(),
        )
        _require_api_success(profile, "profile request")
        funds = await self._funds()
        holdings = await self._request(
            "GET",
            f"{self.settings.upstox_base_url}/v2/portfolio/long-term-holdings",
            headers=self._headers(),
        )
        _require_api_success(holdings, "holdings request")
        positions = await self._request(
            "GET",
            f"{self.settings.upstox_base_url}/v2/portfolio/short-term-positions",
            headers=self._headers(),
        )
        _require_api_success(positions, "positions request")
        orders = [] if self.settings.upstox_sandbox else await self.orders()
        profile_data = (
            profile.get("data")
            if isinstance(profile, dict) and isinstance(profile.get("data"), dict)
            else {}
        )
        components: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for source, payload in (("holding", holdings), ("position", positions)):
            data = payload.get("data") if isinstance(payload, dict) else []
            if isinstance(data, list):
                for raw in data:
                    if not isinstance(raw, dict):
                        continue
                    symbol = _upstox_symbol(raw)
                    exchange = _upstox_exchange(raw) or "NSE"
                    quantity = float(
                        _number(raw.get("net_quantity"))
                        if raw.get("net_quantity") is not None
                        else (_number(raw.get("quantity")) or 0.0)
                    )
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
                float(
                    _number(raw.get("net_quantity"))
                    if raw.get("net_quantity") is not None
                    else (_number(raw.get("quantity")) or 0.0)
                )
                for raw in parts
            )
            if quantity == 0:
                continue
            if bucket == "DELIVERY" and len(parts) > 1:
                netted_count += 1
            if bucket != "DELIVERY":
                separate_position_count += 1
            raw = parts[-1]
            token = (
                next(
                    (
                        str(
                            part.get("instrument_token")
                            or part.get("instrument_key")
                            or ""
                        )
                        for part in parts
                        if part.get("instrument_token") or part.get("instrument_key")
                    ),
                    "",
                )
                or None
            )
            instrument = normalize_instrument(
                symbol,
                market="IN",
                exchange=exchange,
                broker_instrument_id=token,
            )
            average = (
                first_number(raw, {"average_price", "buy_price"})
                if len(parts) == 1
                else None
            )
            last = next(
                (
                    value
                    for part in reversed(parts)
                    if (value := first_number(part, {"last_price", "ltp"})) is not None
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
                    average_price=average,
                    last_price=last,
                    market_value=quantity * last if last is not None else None,
                    unrealized_pnl=sum(pnl_values) if pnl_values else None,
                    product=(
                        str(raw.get("product") or "D")
                        if len(parts) == 1
                        else "NETTED_DELIVERY"
                    ),
                    source_components=len(parts),
                )
            )
        buying_power = _cash_available_to_trade(funds)
        equity = sum(position.market_value or 0.0 for position in rows) + (
            buying_power or 0.0
        )
        return PortfolioSnapshot(
            account=AccountSnapshot(
                broker="upstox",
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
                if str(row.get("status") or "").lower()
                not in {"complete", "completed", "cancelled", "rejected"}
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
                + (
                    [
                        "Upstox sandbox does not expose an order book; open-order state is omitted."
                    ]
                    if self.settings.upstox_sandbox
                    else []
                )
            ),
        )

    async def quote(self, instrument: InstrumentRef) -> Quote:
        if instrument.market != MarketCode.INDIA or not instrument.broker_instrument_id:
            raise AdapterError(
                "Upstox quote requires an Indian broker instrument_token"
            )
        payload = await self._request(
            "GET",
            f"{self.settings.upstox_base_url}/v2/market-quote/quotes",
            headers=self._headers(),
            params={"instrument_key": instrument.broker_instrument_id},
        )
        _require_api_success(payload, "quote request")
        row = first_dict(payload)
        returned_token = str(row.get("instrument_token") or "").strip()
        returned_symbol = str(row.get("symbol") or "").upper().strip()
        identity_confirmed = (
            returned_token == instrument.broker_instrument_id
            and returned_symbol == instrument.symbol
        )
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
        return Quote(
            instrument=instrument,
            bid=bid,
            ask=ask,
            last=first_number(row, {"last_price", "ltp"}),
            previous_close=first_number(row, {"close", "cp"}),
            volume=first_number(row, {"volume", "volume_traded_today"}),
            timestamp=parsed_time,
            source="upstox",
            fresh=parsed_time is not None,
            broker_identity_confirmed=identity_confirmed,
            broker_returned_symbol=returned_symbol or None,
            broker_returned_instrument_id=returned_token or None,
        )

    @staticmethod
    def _whole_quantity(order: OrderRequest) -> int:
        if (
            order.notional is not None
            or order.quantity is None
            or not float(order.quantity).is_integer()
        ):
            raise AdapterError(
                "Upstox Indian cash equities require a whole-share quantity"
            )
        return int(order.quantity)

    def client_order_key(self, order: OrderRequest) -> str:
        return deterministic_order_tag(str(order.tag or order.action_id), max_length=40)

    @staticmethod
    def _matches_order_instrument(row: dict[str, Any], order: OrderRequest) -> bool:
        token = str(row.get("instrument_token") or row.get("instrument_key") or "")
        if token and order.instrument.broker_instrument_id:
            return token == order.instrument.broker_instrument_id
        exchange = _upstox_exchange(row)
        return bool(
            _upstox_symbol(row) == order.instrument.symbol
            and (not exchange or exchange == order.instrument.exchange)
        )

    async def _delivery_quantity(self, order: OrderRequest) -> float:
        holdings = await self._request(
            "GET",
            f"{self.settings.upstox_base_url}/v2/portfolio/long-term-holdings",
            headers=self._headers(),
        )
        _require_api_success(holdings, "holdings request")
        positions = await self._request(
            "GET",
            f"{self.settings.upstox_base_url}/v2/portfolio/short-term-positions",
            headers=self._headers(),
        )
        _require_api_success(positions, "positions request")
        orders = [] if self.settings.upstox_sandbox else await self.orders()
        holding_rows = holdings.get("data") if isinstance(holdings, dict) else []
        position_rows = positions.get("data") if isinstance(positions, dict) else []
        holding_iter = holding_rows if isinstance(holding_rows, list) else []
        settled = sum(
            float(_number(row.get("quantity")) or 0.0)
            for row in holding_iter
            if isinstance(row, dict) and self._matches_order_instrument(row, order)
        )
        same_day = 0.0
        for row in position_rows if isinstance(position_rows, list) else []:
            if not isinstance(row, dict) or not self._matches_order_instrument(
                row, order
            ):
                continue
            quantity = _number(row.get("net_quantity"))
            quantity = (
                quantity if quantity is not None else _number(row.get("quantity"))
            )
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
            if str(row.get("status") or "").lower() in _TERMINAL_ORDER_STATUSES:
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
        static_ip_proven = (
            self.settings.upstox_sandbox or self.settings.india_static_ip_registered
        )
        demat_sell_proven = bool(
            order.side == "buy"
            or self.settings.upstox_sandbox
            or self.settings.india_demat_sell_authorized
        )
        if not static_ip_proven:
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
                "Upstox quote did not confirm that the instrument token and trading symbol match the order."
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
        funds = await self._funds()
        buying_power = _cash_available_to_trade(funds)
        if order.side == "buy" and buying_power is None:
            reasons.append("Broker cash available to trade was not returned.")
        position_proof = order.side == "buy"
        if order.side == "sell":
            held = await self._delivery_quantity(order)
            position_proof = held >= quantity
            if not position_proof:
                reasons.append(
                    f"Broker holdings show {held:g} shares, below the requested sell quantity {quantity:g}."
                )
        fees_payload: dict[str, Any] = {}
        margin_payload: dict[str, Any] = {}
        if price:
            raw_margin = await self._request(
                "POST",
                f"{self.settings.upstox_base_url}/v2/charges/margin",
                headers=self._headers(),
                json={
                    "instruments": [
                        {
                            "instrument_key": order.instrument.broker_instrument_id,
                            "quantity": quantity,
                            "product": "D",
                            "transaction_type": order.side.upper(),
                            "price": price,
                        }
                    ]
                },
            )
            margin_payload = raw_margin if isinstance(raw_margin, dict) else {}
            raw_fees = await self._request(
                "GET",
                f"{self.settings.upstox_base_url}/v2/charges/brokerage",
                headers=self._headers(),
                params={
                    "instrument_token": order.instrument.broker_instrument_id,
                    "quantity": quantity,
                    "product": "D",
                    "transaction_type": order.side.upper(),
                    "price": price,
                },
            )
            fees_payload = raw_fees if isinstance(raw_fees, dict) else {}
        fees = _brokerage_total(fees_payload)
        required_margin = _required_margin(margin_payload)
        margin_proven = _api_success(margin_payload) and required_margin is not None
        if not margin_proven:
            reasons.append("Broker margin preview did not return an explicit total.")
        charges_proven = _api_success(fees_payload) and fees is not None
        if not charges_proven:
            reasons.append("Broker brokerage preview did not return explicit charges.")
        if (
            order.side == "buy"
            and required_margin is not None
            and fees is not None
            and buying_power is not None
            and required_margin + fees > buying_power
        ):
            reasons.append(
                "Available broker funds do not cover required margin plus charges."
            )
        return OrderPreview(
            broker="upstox",
            action_id=order.action_id,
            passed=not reasons,
            estimated_value=estimated,
            estimated_fees=fees,
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
                "static_ip": static_ip_proven,
                "demat_sell_authorized": demat_sell_proven,
                "sandbox": self.settings.upstox_sandbox,
            },
        )

    async def place(self, order: OrderRequest) -> OrderResult:
        quantity = self._whole_quantity(order)
        if (
            not self.settings.upstox_sandbox
            and not self.settings.india_static_ip_registered
        ):
            raise AdapterError(
                "Indian API placement is blocked until a static IP is registered with the broker"
            )
        if (
            order.side == "sell"
            and not self.settings.upstox_sandbox
            and not self.settings.india_demat_sell_authorized
        ):
            raise AdapterError(
                "Indian delivery sells are blocked until demat authorization is attested"
            )
        if order.order_type != "limit":
            raise AdapterError(
                "Indian API execution is restricted to protected limit orders"
            )
        if not order.instrument.broker_instrument_id:
            raise AdapterError("Upstox placement requires instrument_token")
        if self.settings.upstox_sandbox:
            url = "https://sandbox.upstox.com/v2/order/place"
        else:
            url = f"{self.settings.upstox_order_base_url}/v3/order/place"
        body = {
            "quantity": quantity,
            "product": "D",
            "validity": order.time_in_force.upper(),
            "price": float(order.limit_price or 0),
            "tag": self.client_order_key(order),
            "instrument_token": order.instrument.broker_instrument_id,
            "order_type": order.order_type.upper(),
            "transaction_type": order.side.upper(),
            "disclosed_quantity": 0,
            "trigger_price": 0,
            "is_amo": False,
            "slice": False,
            "market_protection": -1 if order.order_type == "market" else 0,
        }
        headers = (
            self._sandbox_headers() if self.settings.upstox_sandbox else self._headers()
        )
        if self.settings.upstox_algo_name:
            headers["X-Algo-Name"] = self.settings.upstox_algo_name
        payload = await self._request("POST", url, headers=headers, json=body)
        data = (
            payload.get("data")
            if isinstance(payload, dict) and isinstance(payload.get("data"), dict)
            else {}
        )
        order_ids = (
            data.get("order_ids") if isinstance(data.get("order_ids"), list) else []
        )
        order_id = str(data.get("order_id") or (order_ids[0] if order_ids else ""))
        accepted = bool(order_id) and _api_success(payload)
        return OrderResult(
            broker="upstox",
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
        if self.settings.upstox_sandbox:
            raise CapabilityUnavailable(
                "Upstox sandbox does not expose order-book or fill-status APIs"
            )
        payload = await self._request(
            "GET",
            f"{self.settings.upstox_base_url}/v2/order/retrieve-all",
            headers=self._headers(),
        )
        _require_api_success(payload, "order-book request")
        data = payload.get("data") if isinstance(payload, dict) else payload
        return (
            [redact(row) for row in data if isinstance(row, dict)]
            if isinstance(data, list)
            else []
        )
