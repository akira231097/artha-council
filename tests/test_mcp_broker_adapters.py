from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import httpx

from artha_mcp.adapters.base import AdapterError
from artha_mcp.adapters.http import MAX_BROKER_RESPONSE_BYTES
from artha_mcp.adapters.upstox import UpstoxBrokerAdapter
from artha_mcp.adapters.zerodha import ZerodhaBrokerAdapter
from artha_mcp.models import BrokerName, MarketCode, OrderRequest

from .mcp_helpers import india_instrument, test_settings


def response(payload, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        content=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )


def upstox_margin(total: float = 100.0) -> httpx.Response:
    return response(
        {
            "status": "success",
            "data": {"required_margin": total, "final_margin": total},
        }
    )


class TestUpstoxAdapter(unittest.IsolatedAsyncioTestCase):
    async def test_buy_preview_and_place_use_official_order_shapes(self) -> None:
        now = datetime.now(UTC).isoformat()
        seen: list[tuple[str, str, dict | None]] = []
        place_algo_name = None

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal place_algo_name
            body = json.loads(request.content) if request.content else None
            seen.append((request.method, request.url.path, body))
            if request.url.path.endswith("/market-quote/quotes"):
                return response(
                    {
                        "status": "success",
                        "data": {
                            "NSE_EQ|RELIANCE": {
                                "instrument_token": "NSE_EQ|RELIANCE",
                                "symbol": "RELIANCE",
                                "last_price": 2500,
                                "timestamp": now,
                                "depth": {
                                    "buy": [{"price": 2499.5}],
                                    "sell": [{"price": 2500.5}],
                                },
                            }
                        },
                    }
                )
            if request.url.path.endswith("/get-funds-and-margin"):
                return response(
                    {
                        "status": "success",
                        "data": {
                            "available_to_trade": {
                                "total": 125000,
                                "cash_available_to_trade": {"total": 100000},
                                "pledge_available_to_trade": {"total": 25000},
                            }
                        },
                    }
                )
            if request.url.path.endswith("/charges/brokerage"):
                return response(
                    {"status": "success", "data": {"charges": {"total": 4.25}}}
                )
            if request.url.path.endswith("/charges/margin"):
                self.assertEqual(
                    body,
                    {
                        "instruments": [
                            {
                                "instrument_key": "NSE_EQ|RELIANCE",
                                "quantity": 2,
                                "product": "D",
                                "transaction_type": "BUY",
                                "price": 2500.0,
                            }
                        ]
                    },
                )
                return upstox_margin(5000)
            if request.url.path.endswith("/order/retrieve-all"):
                return response({"status": "success", "data": []})
            if request.url.path.endswith("/v3/order/place"):
                place_algo_name = request.headers.get("X-Algo-Name")
                return response(
                    {"status": "success", "data": {"order_ids": ["up-order-1"]}}
                )
            raise AssertionError(request.url)

        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                market=MarketCode.INDIA,
                broker=BrokerName.UPSTOX,
                broker_plugin="",
                upstox_access_token="token-for-test",
                upstox_sandbox=False,
                upstox_algo_name="ArthaTest",
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            adapter = UpstoxBrokerAdapter(settings, client=client)
            order = OrderRequest(
                action_id="india-buy-1",
                instrument=india_instrument(),
                side="buy",
                order_type="limit",
                quantity=2,
                limit_price=2500,
                max_price=2501,
                product="delivery",
            )
            preview = await adapter.preview(order)
            placed = await adapter.place(order)
            self.assertTrue(preview.passed)
            self.assertEqual(preview.estimated_value, 5000)
            self.assertEqual(preview.estimated_fees, 4.25)
            self.assertEqual(preview.buying_power, 100000)
            self.assertTrue(
                all(
                    preview.broker_proof[key]
                    for key in ("instrument", "quote", "funds", "order_preview")
                )
            )
            self.assertTrue(placed.accepted)
            self.assertEqual(placed.broker_order_id, "up-order-1")
            place_body = next(
                body for method, path, body in seen if path.endswith("/v3/order/place")
            )
            self.assertEqual(place_body["instrument_token"], "NSE_EQ|RELIANCE")
            self.assertEqual(place_body["product"], "D")
            self.assertEqual(place_body["quantity"], 2)
            self.assertTrue(place_body["tag"].isalnum())
            self.assertLessEqual(len(place_body["tag"]), 40)
            self.assertEqual(place_algo_name, "ArthaTest")
            await client.aclose()

    async def test_upstox_search_returns_only_verified_cash_equities(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertTrue(request.url.path.endswith("/v2/instruments/search"))
            self.assertEqual(request.url.params["segments"], "EQ")
            return response(
                {
                    "status": "success",
                    "data": [
                        {
                            "name": "RELIANCE INDUSTRIES LTD",
                            "segment": "NSE_EQ",
                            "exchange": "NSE",
                            "isin": "INE002A01018",
                            "instrument_key": "NSE_EQ|INE002A01018",
                            "trading_symbol": "RELIANCE",
                            "instrument_type": "EQ",
                            "tick_size": 0.05,
                            "lot_size": 1,
                        },
                        {
                            "name": "RELIANCE INDUSTRIES LTD.",
                            "segment": "BSE_EQ",
                            "exchange": "BSE",
                            "isin": "INE002A01018",
                            "instrument_key": "BSE_EQ|INE002A01018",
                            "trading_symbol": "RELIANCE",
                            "instrument_type": "A",
                            "tick_size": 0.05,
                            "lot_size": 1,
                        },
                        {
                            "segment": "NSE_FO",
                            "exchange": "NSE",
                            "instrument_key": "NSE_FO|123",
                            "trading_symbol": "RELIANCE FUT",
                            "instrument_type": "FUT",
                        },
                    ],
                }
            )

        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                market=MarketCode.INDIA,
                broker=BrokerName.UPSTOX,
                broker_plugin="",
                upstox_access_token="test",
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            adapter = UpstoxBrokerAdapter(settings, client=client)
            matches = await adapter.search_instruments(
                "Reliance", exchange="ALL", limit=20
            )
            self.assertEqual(len(matches), 2)
            self.assertEqual(
                matches[0]["instrument"]["broker_instrument_id"],
                "NSE_EQ|INE002A01018",
            )
            self.assertEqual(
                matches[1]["instrument"]["broker_instrument_id"],
                "BSE_EQ|INE002A01018",
            )
            self.assertTrue(matches[0]["broker_verified"])
            await client.aclose()

    async def test_upstox_portfolio_nets_holdings_and_same_day_position(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/user/profile"):
                return response({"status": "success", "data": {"user_id": "user1"}})
            if path.endswith("/get-funds-and-margin"):
                return response(
                    {
                        "status": "success",
                        "data": {
                            "available_to_trade": {
                                "cash_available_to_trade": {"total": 1000}
                            }
                        },
                    }
                )
            if path.endswith("/long-term-holdings"):
                return response(
                    {
                        "status": "success",
                        "data": [
                            {
                                "trading_symbol": "RELIANCE",
                                "exchange": "NSE",
                                "instrument_token": "NSE_EQ|INE002A01018",
                                "quantity": 10,
                                "average_price": 90,
                                "last_price": 100,
                                "pnl": 100,
                            }
                        ],
                    }
                )
            if path.endswith("/short-term-positions"):
                return response(
                    {
                        "status": "success",
                        "data": [
                            {
                                "trading_symbol": "RELIANCE",
                                "exchange": "NSE",
                                "instrument_token": "NSE_EQ|INE002A01018",
                                "quantity": -2,
                                "product": "D",
                                "last_price": 101,
                                "pnl": 4,
                            },
                            {
                                "trading_symbol": "RELIANCE",
                                "exchange": "NSE",
                                "instrument_token": "NSE_EQ|INE002A01018",
                                "quantity": 3,
                                "product": "I",
                                "last_price": 101,
                                "pnl": 2,
                            },
                        ],
                    }
                )
            if path.endswith("/order/retrieve-all"):
                return response({"status": "success", "data": []})
            raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                market=MarketCode.INDIA,
                broker=BrokerName.UPSTOX,
                broker_plugin="",
                upstox_access_token="test",
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            adapter = UpstoxBrokerAdapter(settings, client=client)
            portfolio = await adapter.portfolio()
            self.assertEqual(len(portfolio.positions), 2)
            delivery = next(
                row for row in portfolio.positions if row.product == "NETTED_DELIVERY"
            )
            intraday = next(row for row in portfolio.positions if row.product == "I")
            self.assertEqual(delivery.quantity, 8)
            self.assertEqual(delivery.market_value, 808)
            self.assertIsNone(delivery.average_price)
            self.assertEqual(delivery.unrealized_pnl, 104)
            self.assertEqual(intraday.quantity, 3)
            self.assertTrue(portfolio.warnings)
            await client.aclose()

    async def test_upstox_sandbox_portfolio_omits_unavailable_order_book(self) -> None:
        seen_paths: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            seen_paths.append(path)
            if path.endswith("/user/profile"):
                return response(
                    {"status": "success", "data": {"user_id": "sandbox-user"}}
                )
            if path.endswith("/get-funds-and-margin"):
                return response(
                    {
                        "status": "success",
                        "data": {
                            "available_to_trade": {
                                "cash_available_to_trade": {"total": 1000}
                            }
                        },
                    }
                )
            if path.endswith(("/long-term-holdings", "/short-term-positions")):
                return response({"status": "success", "data": []})
            raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                market=MarketCode.INDIA,
                broker=BrokerName.UPSTOX,
                broker_plugin="",
                upstox_access_token="live-read-token",
                upstox_sandbox_access_token="sandbox-token",
                upstox_sandbox=True,
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            adapter = UpstoxBrokerAdapter(settings, client=client)
            portfolio = await adapter.portfolio()
            self.assertEqual(portfolio.open_orders, [])
            self.assertIn("order book", " ".join(portfolio.warnings))
            self.assertFalse(
                any(path.endswith("/order/retrieve-all") for path in seen_paths)
            )
            await client.aclose()

    async def test_upstox_sandbox_delivery_proof_omits_order_book(self) -> None:
        seen_paths: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            seen_paths.append(path)
            if path.endswith("/long-term-holdings"):
                return response(
                    {
                        "status": "success",
                        "data": [
                            {
                                "trading_symbol": "RELIANCE",
                                "exchange": "NSE",
                                "instrument_token": "NSE_EQ|RELIANCE",
                                "quantity": 3,
                            }
                        ],
                    }
                )
            if path.endswith("/short-term-positions"):
                return response({"status": "success", "data": []})
            raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                market=MarketCode.INDIA,
                broker=BrokerName.UPSTOX,
                broker_plugin="",
                upstox_access_token="live-read-token",
                upstox_sandbox_access_token="sandbox-token",
                upstox_sandbox=True,
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            adapter = UpstoxBrokerAdapter(settings, client=client)
            quantity = await adapter._delivery_quantity(
                OrderRequest(
                    action_id="sandbox-holding-proof",
                    instrument=india_instrument(),
                    side="sell",
                    order_type="limit",
                    quantity=1,
                    limit_price=99,
                    min_price=98,
                    product="delivery",
                )
            )
            self.assertEqual(quantity, 3)
            self.assertFalse(
                any(path.endswith("/order/retrieve-all") for path in seen_paths)
            )
            await client.aclose()

    async def test_v2_funds_shape_remains_a_fail_safe_fallback(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/market-quote/quotes"):
                return response(
                    {
                        "status": "success",
                        "data": {
                            "NSE_EQ|RELIANCE": {
                                "instrument_token": "NSE_EQ|RELIANCE",
                                "symbol": "RELIANCE",
                                "last_price": 100,
                                "timestamp": datetime.now(UTC).isoformat(),
                                "depth": {
                                    "buy": [{"price": 99.9}],
                                    "sell": [{"price": 100.1}],
                                },
                            }
                        },
                    }
                )
            if request.url.path.endswith("/get-funds-and-margin"):
                return response(
                    {
                        "status": "success",
                        "data": {"equity": {"available_margin": 500}},
                    }
                )
            if request.url.path.endswith("/charges/brokerage"):
                return response(
                    {"status": "success", "data": {"charges": {"total": 1}}}
                )
            if request.url.path.endswith("/charges/margin"):
                return upstox_margin(100)
            raise AssertionError(request.url.path)

        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                market=MarketCode.INDIA,
                broker=BrokerName.UPSTOX,
                broker_plugin="",
                upstox_access_token="test",
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            adapter = UpstoxBrokerAdapter(settings, client=client)
            preview = await adapter.preview(
                OrderRequest(
                    action_id="v2-funds-fallback",
                    instrument=india_instrument(),
                    side="buy",
                    order_type="limit",
                    quantity=1,
                    limit_price=100,
                    max_price=101,
                    product="delivery",
                )
            )
            self.assertTrue(preview.passed)
            self.assertEqual(preview.buying_power, 500)
            await client.aclose()

    async def test_sell_preview_proves_delivery_holdings(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/market-quote/quotes"):
                return response(
                    {
                        "status": "success",
                        "data": {
                            "x": {
                                "instrument_token": "NSE_EQ|RELIANCE",
                                "symbol": "RELIANCE",
                                "last_price": 2500,
                                "depth": {
                                    "buy": [{"price": 2499.5}],
                                    "sell": [{"price": 2500.5}],
                                },
                            }
                        },
                    }
                )
            if path.endswith("/get-funds-and-margin"):
                return response(
                    {
                        "status": "success",
                        "data": {"equity": {"available_margin": 1}},
                    }
                )
            if path.endswith("/long-term-holdings"):
                return response(
                    {
                        "status": "success",
                        "data": [
                            {
                                "trading_symbol": "RELIANCE",
                                "exchange": "NSE",
                                "quantity": 1,
                            }
                        ],
                    }
                )
            if path.endswith("/short-term-positions"):
                return response({"status": "success", "data": []})
            if path.endswith("/order/retrieve-all"):
                return response({"status": "success", "data": []})
            if path.endswith("/charges/brokerage"):
                return response(
                    {"status": "success", "data": {"charges": {"total": 2}}}
                )
            if path.endswith("/charges/margin"):
                return upstox_margin(4998)
            raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                market=MarketCode.INDIA,
                broker=BrokerName.UPSTOX,
                broker_plugin="",
                upstox_access_token="test",
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            adapter = UpstoxBrokerAdapter(settings, client=client)
            preview = await adapter.preview(
                OrderRequest(
                    action_id="india-sell-1",
                    instrument=india_instrument(),
                    side="sell",
                    order_type="limit",
                    quantity=2,
                    limit_price=2499,
                    min_price=2400,
                    product="delivery",
                )
            )
            self.assertFalse(preview.passed)
            self.assertFalse(preview.broker_proof["position"])
            self.assertIn(
                "below the requested sell quantity", " ".join(preview.reasons)
            )
            await client.aclose()

    async def test_upstox_sell_reserves_same_day_sales_and_pending_orders(
        self,
    ) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/market-quote/quotes"):
                return response(
                    {
                        "status": "success",
                        "data": {
                            "x": {
                                "instrument_token": "NSE_EQ|RELIANCE",
                                "symbol": "RELIANCE",
                                "timestamp": datetime.now(UTC).isoformat(),
                                "last_price": 100,
                                "depth": {
                                    "buy": [{"price": 99.9}],
                                    "sell": [{"price": 100.1}],
                                },
                            }
                        },
                    }
                )
            if path.endswith("/get-funds-and-margin"):
                return response(
                    {
                        "status": "success",
                        "data": {
                            "available_to_trade": {
                                "cash_available_to_trade": {"total": 1000}
                            }
                        },
                    }
                )
            if path.endswith("/long-term-holdings"):
                return response(
                    {
                        "status": "success",
                        "data": [
                            {
                                "trading_symbol": "RELIANCE",
                                "exchange": "NSE",
                                "instrument_token": "NSE_EQ|RELIANCE",
                                "quantity": 10,
                                "product": "D",
                            }
                        ],
                    }
                )
            if path.endswith("/short-term-positions"):
                return response(
                    {
                        "status": "success",
                        "data": [
                            {
                                "trading_symbol": "RELIANCE",
                                "exchange": "NSE",
                                "instrument_token": "NSE_EQ|RELIANCE",
                                "quantity": -4,
                                "product": "D",
                            }
                        ],
                    }
                )
            if path.endswith("/order/retrieve-all"):
                return response(
                    {
                        "status": "success",
                        "data": [
                            {
                                "trading_symbol": "RELIANCE",
                                "exchange": "NSE",
                                "instrument_token": "NSE_EQ|RELIANCE",
                                "transaction_type": "SELL",
                                "product": "D",
                                "status": "open",
                                "quantity": 6,
                                "filled_quantity": 0,
                                "pending_quantity": 6,
                            }
                        ],
                    }
                )
            if path.endswith("/charges/brokerage"):
                return response(
                    {"status": "success", "data": {"charges": {"total": 1}}}
                )
            if path.endswith("/charges/margin"):
                return upstox_margin(99.9)
            raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                market=MarketCode.INDIA,
                broker=BrokerName.UPSTOX,
                broker_plugin="",
                upstox_access_token="test",
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            adapter = UpstoxBrokerAdapter(settings, client=client)
            preview = await adapter.preview(
                OrderRequest(
                    action_id="reserved-upstox-sell",
                    instrument=india_instrument(),
                    side="sell",
                    order_type="limit",
                    quantity=1,
                    limit_price=99.9,
                    min_price=99,
                    product="delivery",
                )
            )
            self.assertFalse(preview.passed)
            self.assertFalse(preview.broker_proof["position"])
            self.assertIn("show 0 shares", " ".join(preview.reasons))
            await client.aclose()

    async def test_upstox_buy_requires_cash_for_value_plus_charges(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/market-quote/quotes"):
                return response(
                    {
                        "status": "success",
                        "data": {
                            "x": {
                                "instrument_token": "NSE_EQ|RELIANCE",
                                "symbol": "RELIANCE",
                                "timestamp": datetime.now(UTC).isoformat(),
                                "last_price": 100,
                                "depth": {
                                    "buy": [{"price": 99.9}],
                                    "sell": [{"price": 100.1}],
                                },
                            }
                        },
                    }
                )
            if path.endswith("/get-funds-and-margin"):
                return response(
                    {
                        "status": "success",
                        "data": {
                            "available_to_trade": {
                                "cash_available_to_trade": {"total": 100.5}
                            }
                        },
                    }
                )
            if path.endswith("/charges/brokerage"):
                return response(
                    {"status": "success", "data": {"charges": {"total": 1}}}
                )
            if path.endswith("/charges/margin"):
                return upstox_margin(100)
            raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                market=MarketCode.INDIA,
                broker=BrokerName.UPSTOX,
                broker_plugin="",
                upstox_access_token="test",
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            adapter = UpstoxBrokerAdapter(settings, client=client)
            preview = await adapter.preview(
                OrderRequest(
                    action_id="fees-exceed-cash-upstox",
                    instrument=india_instrument(),
                    side="buy",
                    order_type="limit",
                    quantity=1,
                    limit_price=100,
                    max_price=101,
                    product="delivery",
                )
            )
            self.assertFalse(preview.passed)
            self.assertIn("required margin plus charges", " ".join(preview.reasons))
            await client.aclose()

    async def test_quote_identity_mismatch_blocks_upstox_preview(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/market-quote/quotes"):
                return response(
                    {
                        "status": "success",
                        "data": {
                            "NSE_EQ:OTHER": {
                                "instrument_token": "NSE_EQ|OTHER",
                                "symbol": "OTHER",
                                "timestamp": datetime.now(UTC).isoformat(),
                                "last_price": 100,
                                "depth": {
                                    "buy": [{"price": 99.9}],
                                    "sell": [{"price": 100.1}],
                                },
                            }
                        },
                    }
                )
            if request.url.path.endswith("/get-funds-and-margin"):
                return response(
                    {
                        "status": "success",
                        "data": {"equity": {"available_margin": 1000}},
                    }
                )
            if request.url.path.endswith("/charges/brokerage"):
                return response(
                    {"status": "success", "data": {"charges": {"total": 1}}}
                )
            if request.url.path.endswith("/charges/margin"):
                return upstox_margin(100)
            raise AssertionError(request.url.path)

        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                market=MarketCode.INDIA,
                broker=BrokerName.UPSTOX,
                broker_plugin="",
                upstox_access_token="test",
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            adapter = UpstoxBrokerAdapter(settings, client=client)
            preview = await adapter.preview(
                OrderRequest(
                    action_id="identity-mismatch",
                    instrument=india_instrument(),
                    side="buy",
                    order_type="limit",
                    quantity=1,
                    limit_price=100,
                    max_price=101,
                    product="delivery",
                )
            )
            self.assertFalse(preview.passed)
            self.assertFalse(preview.broker_proof["instrument"])
            self.assertIn("did not confirm", " ".join(preview.reasons))
            await client.aclose()

    async def test_upstox_market_orders_and_unregistered_static_ip_fail_closed(
        self,
    ) -> None:
        async def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("Unsafe order shape must fail before broker HTTP")

        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                market=MarketCode.INDIA,
                broker=BrokerName.UPSTOX,
                broker_plugin="",
                upstox_access_token="test",
                upstox_sandbox=False,
                india_static_ip_registered=False,
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            adapter = UpstoxBrokerAdapter(settings, client=client)
            market_order = OrderRequest(
                action_id="unsafe-market",
                instrument=india_instrument(),
                side="buy",
                order_type="market",
                quantity=1,
                max_price=101,
                product="delivery",
            )
            with self.assertRaisesRegex(AdapterError, "limit orders"):
                await adapter.preview(market_order)
            limit_order = market_order.model_copy(
                update={"order_type": "limit", "limit_price": 100}
            )
            with self.assertRaisesRegex(AdapterError, "static IP"):
                await adapter.place(limit_order)
            sell_settings = replace(
                settings,
                india_static_ip_registered=True,
                india_demat_sell_authorized=False,
            )
            sell_adapter = UpstoxBrokerAdapter(sell_settings, client=client)
            sell_order = limit_order.model_copy(
                update={"side": "sell", "max_price": None, "min_price": 99}
            )
            with self.assertRaisesRegex(AdapterError, "demat authorization"):
                await sell_adapter.place(sell_order)
            await client.aclose()

    async def test_upstox_sandbox_place_does_not_require_live_static_ip(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.host, "sandbox.upstox.com")
            self.assertTrue(request.url.path.endswith("/v2/order/place"))
            return response(
                {"status": "success", "data": {"order_id": "sandbox-order-1"}}
            )

        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                market=MarketCode.INDIA,
                broker=BrokerName.UPSTOX,
                broker_plugin="",
                upstox_access_token="live-read-token",
                upstox_sandbox_access_token="sandbox-token",
                upstox_sandbox=True,
                india_static_ip_registered=False,
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            adapter = UpstoxBrokerAdapter(settings, client=client)
            result = await adapter.place(
                OrderRequest(
                    action_id="sandbox-place",
                    instrument=india_instrument(),
                    side="buy",
                    order_type="limit",
                    quantity=1,
                    limit_price=100,
                    max_price=101,
                    product="delivery",
                )
            )
            self.assertTrue(result.accepted)
            self.assertEqual(result.broker_order_id, "sandbox-order-1")
            await client.aclose()

    async def test_upstox_http_200_error_payload_is_not_preview_proof(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/market-quote/quotes"):
                return response(
                    {
                        "status": "success",
                        "data": {
                            "x": {
                                "instrument_token": "NSE_EQ|RELIANCE",
                                "symbol": "RELIANCE",
                                "timestamp": datetime.now(UTC).isoformat(),
                                "last_price": 100,
                                "depth": {
                                    "buy": [{"price": 99.9}],
                                    "sell": [{"price": 100.1}],
                                },
                            }
                        },
                    }
                )
            if path.endswith("/get-funds-and-margin"):
                return response(
                    {
                        "status": "success",
                        "data": {
                            "available_to_trade": {
                                "cash_available_to_trade": {"total": 1000}
                            }
                        },
                    }
                )
            if path.endswith("/charges/brokerage"):
                return response({"status": "error", "data": {"charges": {"total": 1}}})
            if path.endswith("/charges/margin"):
                return upstox_margin(100)
            raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                market=MarketCode.INDIA,
                broker=BrokerName.UPSTOX,
                broker_plugin="",
                upstox_access_token="test",
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            adapter = UpstoxBrokerAdapter(settings, client=client)
            preview = await adapter.preview(
                OrderRequest(
                    action_id="error-payload",
                    instrument=india_instrument(),
                    side="buy",
                    order_type="limit",
                    quantity=1,
                    limit_price=100,
                    max_price=101,
                    product="delivery",
                )
            )
            self.assertFalse(preview.passed)
            self.assertFalse(preview.broker_proof["order_preview"])
            self.assertIn("explicit charges", " ".join(preview.reasons))
            await client.aclose()

    async def test_upstox_margin_response_is_required_for_preview_proof(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/market-quote/quotes"):
                return response(
                    {
                        "status": "success",
                        "data": {
                            "x": {
                                "instrument_token": "NSE_EQ|RELIANCE",
                                "symbol": "RELIANCE",
                                "timestamp": datetime.now(UTC).isoformat(),
                                "last_price": 100,
                                "depth": {
                                    "buy": [{"price": 99.9}],
                                    "sell": [{"price": 100.1}],
                                },
                            }
                        },
                    }
                )
            if path.endswith("/get-funds-and-margin"):
                return response(
                    {
                        "status": "success",
                        "data": {
                            "available_to_trade": {
                                "cash_available_to_trade": {"total": 1000}
                            }
                        },
                    }
                )
            if path.endswith("/charges/margin"):
                return response(
                    {
                        "status": "error",
                        "data": {"required_margin": 100, "final_margin": 100},
                    }
                )
            if path.endswith("/charges/brokerage"):
                return response(
                    {"status": "success", "data": {"charges": {"total": 1}}}
                )
            raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                market=MarketCode.INDIA,
                broker=BrokerName.UPSTOX,
                broker_plugin="",
                upstox_access_token="test",
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            adapter = UpstoxBrokerAdapter(settings, client=client)
            preview = await adapter.preview(
                OrderRequest(
                    action_id="margin-error-payload",
                    instrument=india_instrument(),
                    side="buy",
                    order_type="limit",
                    quantity=1,
                    limit_price=100,
                    max_price=101,
                    product="delivery",
                )
            )
            self.assertFalse(preview.passed)
            self.assertFalse(preview.broker_proof["margin"])
            self.assertFalse(preview.broker_proof["order_preview"])
            self.assertIn("margin preview", " ".join(preview.reasons))
            await client.aclose()


class TestZerodhaAdapter(unittest.IsolatedAsyncioTestCase):
    async def test_zerodha_exact_symbol_search_is_broker_verified(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertTrue(request.url.path.endswith("/quote"))
            return response(
                {
                    "status": "success",
                    "data": {
                        "NSE:RELIANCE": {
                            "instrument_token": 738561,
                            "last_price": 2500,
                        }
                    },
                }
            )

        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                market=MarketCode.INDIA,
                broker=BrokerName.ZERODHA,
                broker_plugin="",
                zerodha_api_key="key",
                zerodha_access_token="token",
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            adapter = ZerodhaBrokerAdapter(settings, client=client)
            matches = await adapter.search_instruments(
                "RELIANCE", exchange="NSE", limit=20
            )
            self.assertEqual(len(matches), 1)
            self.assertEqual(
                matches[0]["instrument"]["broker_instrument_id"], "NSE:RELIANCE"
            )
            self.assertTrue(matches[0]["broker_verified"])
            await client.aclose()

    async def test_zerodha_portfolio_nets_settled_and_intraday_components(
        self,
    ) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/user/profile"):
                return response({"status": "success", "data": {"user_id": "AB123"}})
            if path.endswith("/user/margins"):
                return response(
                    {
                        "status": "success",
                        "data": {"equity": {"available": {"live_balance": 1000}}},
                    }
                )
            if path.endswith("/portfolio/holdings"):
                return response(
                    {
                        "status": "success",
                        "data": [
                            {
                                "tradingsymbol": "RELIANCE",
                                "exchange": "NSE",
                                "quantity": 10,
                                "average_price": 90,
                                "last_price": 100,
                                "pnl": 100,
                                "product": "CNC",
                            }
                        ],
                    }
                )
            if path.endswith("/portfolio/positions"):
                return response(
                    {
                        "status": "success",
                        "data": {
                            "net": [
                                {
                                    "tradingsymbol": "RELIANCE",
                                    "exchange": "NSE",
                                    "quantity": -2,
                                    "last_price": 101,
                                    "pnl": 4,
                                    "product": "CNC",
                                },
                                {
                                    "tradingsymbol": "RELIANCE",
                                    "exchange": "NSE",
                                    "quantity": 3,
                                    "last_price": 101,
                                    "pnl": 2,
                                    "product": "MIS",
                                },
                            ],
                            "day": [],
                        },
                    }
                )
            if path.endswith("/orders"):
                return response({"status": "success", "data": []})
            raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                market=MarketCode.INDIA,
                broker=BrokerName.ZERODHA,
                broker_plugin="",
                zerodha_api_key="key",
                zerodha_access_token="token",
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            adapter = ZerodhaBrokerAdapter(settings, client=client)
            portfolio = await adapter.portfolio()
            self.assertEqual(len(portfolio.positions), 2)
            delivery = next(
                row for row in portfolio.positions if row.product == "NETTED_DELIVERY"
            )
            intraday = next(row for row in portfolio.positions if row.product == "MIS")
            self.assertEqual(delivery.quantity, 8)
            self.assertEqual(delivery.market_value, 808)
            self.assertIsNone(delivery.average_price)
            self.assertEqual(delivery.unrealized_pnl, 104)
            self.assertEqual(intraday.quantity, 3)
            self.assertTrue(portfolio.warnings)
            await client.aclose()

    async def test_preview_place_and_idempotency_tag(self) -> None:
        now = datetime.now(UTC).isoformat()
        placed_form = None

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal placed_form
            path = request.url.path
            if path.endswith("/quote"):
                return response(
                    {
                        "status": "success",
                        "data": {
                            "NSE:RELIANCE": {
                                "last_price": 2500,
                                "timestamp": now,
                                "ohlc": {"close": 2480},
                                "depth": {
                                    "buy": [{"price": 2499.5}],
                                    "sell": [{"price": 2500.5}],
                                },
                            }
                        },
                    }
                )
            if path.endswith("/user/margins"):
                return response(
                    {
                        "status": "success",
                        "data": {
                            "equity": {
                                "net": 150000,
                                "available": {
                                    "live_balance": 100000,
                                    "cash": 110000,
                                    "collateral": 50000,
                                },
                            }
                        },
                    }
                )
            if path.endswith("/margins/orders"):
                return response(
                    {
                        "status": "success",
                        "data": [
                            {
                                "charges": {"total": 3.5},
                                "total": 5005,
                            }
                        ],
                    }
                )
            if path.endswith("/charges/orders"):
                return response(
                    {"status": "success", "data": [{"charges": {"total": 3.5}}]}
                )
            if path.endswith("/orders/regular"):
                placed_form = request.content.decode()
                return response(
                    {"status": "success", "data": {"order_id": "kite-order-1"}}
                )
            if path.endswith("/orders"):
                return response({"status": "success", "data": []})
            raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                market=MarketCode.INDIA,
                broker=BrokerName.ZERODHA,
                broker_plugin="",
                zerodha_api_key="key",
                zerodha_access_token="token",
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            adapter = ZerodhaBrokerAdapter(settings, client=client)
            instrument = india_instrument().model_copy(
                update={"broker_instrument_id": "NSE:RELIANCE"}
            )
            order = OrderRequest(
                action_id="action_with_long_unique_value",
                instrument=instrument,
                side="buy",
                order_type="limit",
                quantity=2,
                limit_price=2500,
                max_price=2501,
                product="delivery",
            )
            preview = await adapter.preview(order)
            result = await adapter.place(order)
            self.assertTrue(preview.passed)
            self.assertEqual(preview.estimated_value, 5000)
            self.assertEqual(preview.estimated_fees, 3.5)
            self.assertEqual(preview.buying_power, 100000)
            self.assertTrue(preview.broker_proof["margin"])
            self.assertTrue(result.accepted)
            self.assertIn("product=CNC", placed_form)
            encoded_tag = adapter.client_order_key(order)
            self.assertTrue(encoded_tag.isalnum())
            self.assertEqual(len(encoded_tag), 20)
            self.assertIn(f"tag={encoded_tag}", placed_form)
            await client.aclose()

    async def test_zerodha_sell_reserves_same_day_sales_and_pending_orders(
        self,
    ) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/quote"):
                return response(
                    {
                        "status": "success",
                        "data": {
                            "NSE:RELIANCE": {
                                "last_price": 100,
                                "timestamp": datetime.now(UTC).isoformat(),
                                "depth": {
                                    "buy": [{"price": 99.9}],
                                    "sell": [{"price": 100.1}],
                                },
                            }
                        },
                    }
                )
            if path.endswith("/user/margins"):
                return response(
                    {
                        "status": "success",
                        "data": {"equity": {"available": {"live_balance": 1000}}},
                    }
                )
            if path.endswith("/margins/orders"):
                return response({"status": "success", "data": [{"total": 0}]})
            if path.endswith("/charges/orders"):
                return response(
                    {"status": "success", "data": [{"charges": {"total": 1}}]}
                )
            if path.endswith("/portfolio/holdings"):
                return response(
                    {
                        "status": "success",
                        "data": [
                            {
                                "tradingsymbol": "RELIANCE",
                                "exchange": "NSE",
                                "quantity": 10,
                                "product": "CNC",
                            }
                        ],
                    }
                )
            if path.endswith("/portfolio/positions"):
                return response(
                    {
                        "status": "success",
                        "data": {
                            "net": [
                                {
                                    "tradingsymbol": "RELIANCE",
                                    "exchange": "NSE",
                                    "quantity": -4,
                                    "product": "CNC",
                                }
                            ]
                        },
                    }
                )
            if path.endswith("/orders"):
                return response(
                    {
                        "status": "success",
                        "data": [
                            {
                                "tradingsymbol": "RELIANCE",
                                "exchange": "NSE",
                                "transaction_type": "SELL",
                                "product": "CNC",
                                "status": "OPEN",
                                "quantity": 6,
                                "filled_quantity": 0,
                                "pending_quantity": 6,
                            }
                        ],
                    }
                )
            raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                market=MarketCode.INDIA,
                broker=BrokerName.ZERODHA,
                broker_plugin="",
                zerodha_api_key="key",
                zerodha_access_token="token",
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            adapter = ZerodhaBrokerAdapter(settings, client=client)
            instrument = india_instrument().model_copy(
                update={"broker_instrument_id": "NSE:RELIANCE"}
            )
            preview = await adapter.preview(
                OrderRequest(
                    action_id="reserved-zerodha-sell",
                    instrument=instrument,
                    side="sell",
                    order_type="limit",
                    quantity=1,
                    limit_price=99.9,
                    min_price=99,
                    product="delivery",
                )
            )
            self.assertFalse(preview.passed)
            self.assertFalse(preview.broker_proof["position"])
            self.assertIn("show 0 shares", " ".join(preview.reasons))
            await client.aclose()

    async def test_zerodha_buy_requires_cash_for_margin_plus_charges(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/quote"):
                return response(
                    {
                        "status": "success",
                        "data": {
                            "NSE:RELIANCE": {
                                "last_price": 100,
                                "timestamp": datetime.now(UTC).isoformat(),
                                "depth": {
                                    "buy": [{"price": 99.9}],
                                    "sell": [{"price": 100.1}],
                                },
                            }
                        },
                    }
                )
            if path.endswith("/user/margins"):
                return response(
                    {
                        "status": "success",
                        "data": {"equity": {"available": {"live_balance": 100.5}}},
                    }
                )
            if path.endswith("/margins/orders"):
                return response({"status": "success", "data": [{"total": 100}]})
            if path.endswith("/charges/orders"):
                return response(
                    {"status": "success", "data": [{"charges": {"total": 1}}]}
                )
            raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                market=MarketCode.INDIA,
                broker=BrokerName.ZERODHA,
                broker_plugin="",
                zerodha_api_key="key",
                zerodha_access_token="token",
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            adapter = ZerodhaBrokerAdapter(settings, client=client)
            instrument = india_instrument().model_copy(
                update={"broker_instrument_id": "NSE:RELIANCE"}
            )
            preview = await adapter.preview(
                OrderRequest(
                    action_id="fees-exceed-cash-zerodha",
                    instrument=instrument,
                    side="buy",
                    order_type="limit",
                    quantity=1,
                    limit_price=100,
                    max_price=101,
                    product="delivery",
                )
            )
            self.assertFalse(preview.passed)
            self.assertIn("margin plus charges", " ".join(preview.reasons))
            await client.aclose()

    async def test_zerodha_tags_are_safe_and_distinct_after_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                market=MarketCode.INDIA,
                broker=BrokerName.ZERODHA,
                broker_plugin="",
                zerodha_api_key="key",
                zerodha_access_token="token",
            )
            adapter = ZerodhaBrokerAdapter(settings)
            instrument = india_instrument().model_copy(
                update={"broker_instrument_id": "NSE:RELIANCE"}
            )
            first = OrderRequest(
                action_id="same-prefix:one.with_punctuation",
                instrument=instrument,
                side="buy",
                order_type="limit",
                quantity=1,
                limit_price=100,
                max_price=100,
                product="delivery",
            )
            second = first.model_copy(
                update={"action_id": "same-prefix:two.with_punctuation"}
            )
            first_tag = adapter.client_order_key(first)
            second_tag = adapter.client_order_key(second)
            self.assertTrue(first_tag.isalnum())
            self.assertTrue(second_tag.isalnum())
            self.assertLessEqual(len(first_tag), 20)
            self.assertNotEqual(first_tag, second_tag)
            await adapter.close()

    async def test_mislabeled_zerodha_instrument_id_is_rejected(self) -> None:
        async def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("Mismatched identity must be rejected before HTTP")

        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                market=MarketCode.INDIA,
                broker=BrokerName.ZERODHA,
                broker_plugin="",
                zerodha_api_key="key",
                zerodha_access_token="token",
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            adapter = ZerodhaBrokerAdapter(settings, client=client)
            mislabeled = india_instrument().model_copy(
                update={"broker_instrument_id": "NSE:OTHER"}
            )
            with self.assertRaisesRegex(AdapterError, "exactly match"):
                await adapter.quote(mislabeled)
            await client.aclose()

    async def test_zerodha_market_orders_and_unregistered_static_ip_fail_closed(
        self,
    ) -> None:
        async def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("Unsafe order shape must fail before broker HTTP")

        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                market=MarketCode.INDIA,
                broker=BrokerName.ZERODHA,
                broker_plugin="",
                zerodha_api_key="key",
                zerodha_access_token="token",
                india_static_ip_registered=False,
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            adapter = ZerodhaBrokerAdapter(settings, client=client)
            instrument = india_instrument().model_copy(
                update={"broker_instrument_id": "NSE:RELIANCE"}
            )
            market_order = OrderRequest(
                action_id="unsafe-market",
                instrument=instrument,
                side="buy",
                order_type="market",
                quantity=1,
                max_price=101,
                product="delivery",
            )
            with self.assertRaisesRegex(AdapterError, "limit orders"):
                await adapter.preview(market_order)
            limit_order = market_order.model_copy(
                update={"order_type": "limit", "limit_price": 100}
            )
            with self.assertRaisesRegex(AdapterError, "static IP"):
                await adapter.place(limit_order)
            sell_settings = replace(
                settings,
                india_static_ip_registered=True,
                india_demat_sell_authorized=False,
            )
            sell_adapter = ZerodhaBrokerAdapter(sell_settings, client=client)
            sell_order = limit_order.model_copy(
                update={"side": "sell", "max_price": None, "min_price": 99}
            )
            with self.assertRaisesRegex(AdapterError, "demat authorization"):
                await sell_adapter.place(sell_order)
            await client.aclose()

    async def test_zerodha_http_200_error_margin_is_not_preview_proof(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/quote"):
                return response(
                    {
                        "status": "success",
                        "data": {
                            "NSE:RELIANCE": {
                                "last_price": 100,
                                "timestamp": datetime.now(UTC).isoformat(),
                                "depth": {
                                    "buy": [{"price": 99.9}],
                                    "sell": [{"price": 100.1}],
                                },
                            }
                        },
                    }
                )
            if path.endswith("/user/margins"):
                return response(
                    {
                        "status": "success",
                        "data": {"equity": {"available": {"live_balance": 1000}}},
                    }
                )
            if path.endswith("/margins/orders"):
                return response({"status": "error", "data": [{"total": 100}]})
            if path.endswith("/charges/orders"):
                return response(
                    {"status": "success", "data": [{"charges": {"total": 1}}]}
                )
            raise AssertionError(path)

        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                market=MarketCode.INDIA,
                broker=BrokerName.ZERODHA,
                broker_plugin="",
                zerodha_api_key="key",
                zerodha_access_token="token",
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            adapter = ZerodhaBrokerAdapter(settings, client=client)
            instrument = india_instrument().model_copy(
                update={"broker_instrument_id": "NSE:RELIANCE"}
            )
            preview = await adapter.preview(
                OrderRequest(
                    action_id="error-margin",
                    instrument=instrument,
                    side="buy",
                    order_type="limit",
                    quantity=1,
                    limit_price=100,
                    max_price=101,
                    product="delivery",
                )
            )
            self.assertFalse(preview.passed)
            self.assertFalse(preview.broker_proof["margin"])
            self.assertIn("explicit total", " ".join(preview.reasons))
            await client.aclose()

    async def test_http_errors_redact_broker_secrets(self) -> None:
        async def handler(_request: httpx.Request) -> httpx.Response:
            return response(
                {"access_token": "must-not-leak", "message": "denied"}, status=401
            )

        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                market=MarketCode.INDIA,
                broker=BrokerName.ZERODHA,
                broker_plugin="",
                zerodha_api_key="key",
                zerodha_access_token="token",
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            adapter = ZerodhaBrokerAdapter(settings, client=client)
            with self.assertRaises(AdapterError) as raised:
                await adapter.health()
            self.assertNotIn("must-not-leak", str(raised.exception))
            await client.aclose()

    async def test_oversized_broker_response_is_rejected_before_parsing(self) -> None:
        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"x" * (MAX_BROKER_RESPONSE_BYTES + 1))

        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                market=MarketCode.INDIA,
                broker=BrokerName.ZERODHA,
                broker_plugin="",
                zerodha_api_key="key",
                zerodha_access_token="token",
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            adapter = ZerodhaBrokerAdapter(settings, client=client)
            with self.assertRaisesRegex(AdapterError, "exceeded"):
                await adapter.health()
            await client.aclose()


if __name__ == "__main__":
    unittest.main()
