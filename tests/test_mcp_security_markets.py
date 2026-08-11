from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

from artha_mcp.adapters.http import parse_broker_timestamp
from artha_mcp.adapters.registry import build_broker_adapter
from artha_mcp.adapters.snapshot import MAX_SNAPSHOT_BYTES, SnapshotBrokerAdapter
from artha_mcp.auth import JWTTokenVerifier
from artha_mcp.markets import get_market_profile, normalize_instrument
from artha_mcp.models import AccessMode, BrokerName, MarketCode
from artha_mcp.notifications import NotificationHub
from artha_mcp.research import HostOrchestratedResearchAdapter
from artha_mcp.security import AuthorizationError, CapabilityPolicy, redact
from artha_mcp.service import ArthaMCPService
from artha_mcp.settings import MCPSettings

from .mcp_helpers import test_settings


class TestSettingsAndSecurity(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_public_defaults_are_read_only_and_locked(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = test_settings(self.root)
            settings = replace(
                settings,
                access_mode=AccessMode.READ_ONLY,
                operations_enabled=False,
                trading_enabled=False,
                kill_switch=True,
                broker=BrokerName.SNAPSHOT,
                broker_plugin="",
                research_mode="embedded",
            )
        self.assertEqual(settings.startup_findings()["errors"], [])
        with self.assertRaises(AuthorizationError):
            CapabilityPolicy(settings).require("artha:trade")

    def test_misspelled_safety_boolean_and_numeric_limit_fail_startup(self) -> None:
        with (
            patch.dict(os.environ, {"ARTHA_MCP_KILL_SWITCH": "flase"}, clear=True),
            self.assertRaisesRegex(ValueError, "KILL_SWITCH"),
        ):
            MCPSettings.from_env(root=self.root)

    def test_order_limits_have_market_currency_defaults(self) -> None:
        with patch.dict(os.environ, {"ARTHA_MCP_MARKET": "US"}, clear=True):
            us = MCPSettings.from_env(root=self.root)
        with patch.dict(os.environ, {"ARTHA_MCP_MARKET": "IN"}, clear=True):
            india = MCPSettings.from_env(root=self.root)
        self.assertEqual((us.max_order_value, us.max_daily_order_value), (25, 50))
        self.assertEqual(
            (india.max_order_value, india.max_daily_order_value), (2500, 5000)
        )
        with (
            patch.dict(os.environ, {"ARTHA_MCP_MAX_ORDER_VALUE": "twenty"}, clear=True),
            self.assertRaisesRegex(ValueError, "MAX_ORDER_VALUE"),
        ):
            MCPSettings.from_env(root=self.root)

    def test_remote_http_requires_oauth(self) -> None:
        settings = replace(
            test_settings(self.root), transport="streamable-http", host="0.0.0.0"
        )
        self.assertTrue(
            any("OAuth" in item for item in settings.startup_findings()["errors"])
        )

    def test_allowed_hosts_expand_to_exact_bind_port_without_wildcard(self) -> None:
        settings = replace(
            test_settings(self.root),
            transport="streamable-http",
            host="127.0.0.1",
            port=9123,
            allowed_hosts=("127.0.0.1", "localhost"),
        )
        self.assertEqual(
            settings.effective_allowed_hosts,
            ("127.0.0.1", "127.0.0.1:9123", "localhost", "localhost:9123"),
        )

    def test_remote_http_rejects_partial_insecure_or_wildcard_security(self) -> None:
        settings = replace(
            test_settings(self.root),
            transport="streamable-http",
            host="0.0.0.0",
            allowed_hosts=("*",),
            oauth_issuer="http://issuer.example",
            oauth_resource_url="https://artha.example/mcp",
            oauth_audience="artha-api",
            oauth_jwks_url="https://issuer.example/jwks.json",
            oauth_algorithms=("HS256",),
        )
        errors = " ".join(settings.startup_findings()["errors"])
        self.assertIn("wildcards", errors)
        self.assertIn("OAuth issuer", errors)
        self.assertIn("asymmetric", errors)

    def test_remote_http_accepts_explicit_https_oauth_configuration(self) -> None:
        settings = replace(
            test_settings(self.root),
            transport="streamable-http",
            host="0.0.0.0",
            allowed_hosts=("artha.example",),
            allowed_origins=("https://client.example",),
            oauth_issuer="https://issuer.example",
            oauth_resource_url="https://artha.example/mcp",
            oauth_audience="artha-api",
            oauth_jwks_url="https://issuer.example/jwks.json",
            oauth_algorithms=("RS256",),
        )
        self.assertEqual(settings.startup_findings()["errors"], [])

    def test_allowed_origins_reject_paths_and_queries(self) -> None:
        base = replace(
            test_settings(self.root),
            transport="streamable-http",
            host="127.0.0.1",
        )
        for origin in (
            "https://client.example/app",
            "https://client.example/?token=secret",
        ):
            settings = replace(base, allowed_origins=(origin,))
            self.assertIn(
                "Allowed origins", " ".join(settings.startup_findings()["errors"])
            )

    def test_broker_market_mismatch_and_bad_limits_fail_configuration(self) -> None:
        settings = replace(
            test_settings(self.root),
            market=MarketCode.US,
            broker=BrokerName.UPSTOX,
            broker_plugin="",
            max_order_value=-1,
            max_spread_pct=0.5,
        )
        errors = " ".join(settings.startup_findings()["errors"])
        self.assertIn("Indian equities only", errors)
        self.assertIn("limits", errors)
        self.assertIn("SPREAD", errors)

    def test_india_live_trading_requires_static_ip_attestation(self) -> None:
        settings = replace(
            test_settings(self.root),
            market=MarketCode.INDIA,
            broker=BrokerName.UPSTOX,
            broker_plugin="",
            upstox_access_token="test",
            upstox_sandbox=False,
            india_static_ip_registered=False,
        )
        errors = " ".join(settings.startup_findings()["errors"])
        self.assertIn("static IP", errors)

        sandbox = replace(settings, upstox_sandbox=True)
        self.assertNotIn("static IP", " ".join(sandbox.startup_findings()["errors"]))
        self.assertIn(
            "UPSTOX_SANDBOX_ACCESS_TOKEN",
            " ".join(sandbox.startup_findings()["errors"]),
        )

        sandbox_ready = replace(sandbox, upstox_sandbox_access_token="sandbox-token")
        self.assertNotIn(
            "UPSTOX_SANDBOX_ACCESS_TOKEN",
            " ".join(sandbox_ready.startup_findings()["errors"]),
        )

        read_only = replace(
            settings,
            access_mode=AccessMode.READ_ONLY,
            operations_enabled=False,
            trading_enabled=False,
            kill_switch=True,
        )
        self.assertFalse(read_only.startup_findings()["errors"])
        self.assertIn("static IP", " ".join(read_only.startup_findings()["warnings"]))

    def test_live_india_trading_requires_selected_broker_credentials(self) -> None:
        upstox = replace(
            test_settings(self.root),
            market=MarketCode.INDIA,
            broker=BrokerName.UPSTOX,
            broker_plugin="",
            upstox_access_token="",
        )
        self.assertIn(
            "UPSTOX_ACCESS_TOKEN", " ".join(upstox.startup_findings()["errors"])
        )

        zerodha = replace(
            test_settings(self.root),
            market=MarketCode.INDIA,
            broker=BrokerName.ZERODHA,
            broker_plugin="",
            zerodha_api_key="",
            zerodha_access_token="",
        )
        self.assertIn("KITE_API_KEY", " ".join(zerodha.startup_findings()["errors"]))

    def test_india_sell_authorization_is_visible_and_fail_closed(self) -> None:
        settings = replace(
            test_settings(self.root),
            market=MarketCode.INDIA,
            broker=BrokerName.ZERODHA,
            broker_plugin="",
            zerodha_api_key="key",
            zerodha_access_token="token",
            india_demat_sell_authorized=False,
        )
        self.assertIn(
            "delivery sells remain blocked",
            " ".join(settings.startup_findings()["warnings"]),
        )
        self.assertFalse(
            settings.public_summary()["india_api_compliance"][
                "demat_sell_authorized_attestation"
            ]
        )

    def test_upstox_algo_name_rejects_header_injection(self) -> None:
        settings = replace(
            test_settings(self.root),
            market=MarketCode.INDIA,
            broker=BrokerName.UPSTOX,
            broker_plugin="",
            upstox_access_token="test",
            upstox_algo_name="Artha\nInjected",
        )
        self.assertIn(
            "ARTHA_UPSTOX_ALGO_NAME",
            " ".join(settings.startup_findings()["errors"]),
        )

    def test_broker_base_urls_must_be_clean_https_origins(self) -> None:
        upstox = replace(
            test_settings(self.root),
            market=MarketCode.INDIA,
            broker=BrokerName.UPSTOX,
            broker_plugin="",
            upstox_access_token="test",
            upstox_base_url="http://api.upstox.com",
            upstox_order_base_url="https://user:secret@api-hft.upstox.com/path",
        )
        self.assertIn("HTTPS origins", " ".join(upstox.startup_findings()["errors"]))
        zerodha = replace(
            test_settings(self.root),
            market=MarketCode.INDIA,
            broker=BrokerName.ZERODHA,
            broker_plugin="",
            zerodha_api_key="key",
            zerodha_access_token="token",
            zerodha_base_url="https://api.kite.trade/?token=secret",
        )
        self.assertIn("HTTPS origin", " ".join(zerodha.startup_findings()["errors"]))

    def test_broker_plugin_must_implement_the_full_adapter_contract(self) -> None:
        settings = replace(
            test_settings(self.root),
            broker=BrokerName.PLUGIN,
            broker_plugin="example_plugin:create_adapter",
        )
        module = SimpleNamespace(create_adapter=lambda _settings: object())
        with (
            patch(
                "artha_mcp.adapters.registry.importlib.import_module",
                return_value=module,
            ),
            self.assertRaisesRegex(TypeError, "BrokerAdapter"),
        ):
            build_broker_adapter(settings)

    def test_plugin_mode_requires_explicit_factory(self) -> None:
        settings = replace(
            test_settings(self.root), research_mode="plugin", research_plugin=""
        )
        self.assertTrue(
            any(
                "RESEARCH_PLUGIN" in item
                for item in settings.startup_findings()["errors"]
            )
        )

    def test_redaction_handles_nested_keys_urls_bearer_and_home_paths(self) -> None:
        value = {
            "api_key": "secret",
            "account_id": "12345678",
            "url": "https://example.test/data?apikey=abc123&x=1",
            "header": "Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig",
            "trace": str(Path.home() / "private" / "trace.json"),
        }
        result = redact(value)
        rendered = json.dumps(result)
        self.assertNotIn("abc123", rendered)
        self.assertNotIn("eyJ", rendered)
        self.assertNotIn(str(Path.home()), rendered)
        self.assertEqual(result["account_id"], "****5678")

    def test_robinhood_operation_requires_trade_authority(self) -> None:
        settings = replace(
            test_settings(self.root),
            access_mode=AccessMode.OPERATOR,
            operations_enabled=True,
            trading_enabled=False,
            kill_switch=True,
            research_mode="embedded",
        )
        service = ArthaMCPService(settings)
        with self.assertRaises(AuthorizationError):
            service.robinhood_operation("ta_example")

    def test_robinhood_operation_uses_explicit_host_account_binding(self) -> None:
        settings = replace(
            test_settings(self.root),
            access_mode=AccessMode.TRADING,
            operations_enabled=True,
            trading_enabled=True,
            kill_switch=False,
            research_mode="embedded",
        )
        service = ArthaMCPService(settings)
        raw_operation = {
            "success": True,
            "operation": "auto_tradability_review_then_place_equity_order",
            "tradability_mcp_args": {
                "account_number": "12345678",
                "symbols": ["V"],
            },
            "review_mcp_args": {
                "account_number": "12345678",
                "symbol": "V",
                "side": "buy",
                "dollar_amount": 18.0,
            },
        }
        with (
            patch(
                "artha.robinhood_bridge.build_auto_buy_operation",
                return_value=raw_operation,
            ),
            patch("artha.config.Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER", "12345678"),
            patch("artha.config.Config.ROBINHOOD_EXPECTED_ACCOUNT_TYPE", "cash"),
            patch("artha.config.Config.ROBINHOOD_EXPECTED_ACCOUNT_NICKNAME", "Agentic"),
        ):
            result = service.robinhood_operation("ta_example")

        placeholder = "${ARTHA_RESOLVED_ROBINHOOD_ACCOUNT_NUMBER}"
        self.assertEqual(result["tradability_mcp_args"]["account_number"], placeholder)
        self.assertEqual(result["review_mcp_args"]["account_number"], placeholder)
        self.assertEqual(
            result["account_binding"]["configured_account_masked"], "****5678"
        )
        self.assertEqual(result["account_binding"]["expected_account_type"], "cash")
        self.assertNotIn("12345678", json.dumps(result))

    def test_canonical_webhook_configuration_is_recognized_and_validated(self) -> None:
        with patch.dict(
            os.environ,
            {"ARTHA_MCP_NOTIFICATION_WEBHOOK_URL": "https://hooks.example/artha"},
            clear=False,
        ):
            self.assertTrue(NotificationHub("webhook").status()["configured"])
        with patch.dict(
            os.environ,
            {
                "ARTHA_MCP_NOTIFICATION_WEBHOOK_URL": "https://user:secret@hooks.example/x"
            },
            clear=False,
        ):
            self.assertFalse(NotificationHub("webhook").status()["configured"])

    def test_live_sell_review_requires_trade_authority(self) -> None:
        settings = replace(
            test_settings(self.root),
            access_mode=AccessMode.OPERATOR,
            operations_enabled=True,
            trading_enabled=False,
            kill_switch=True,
            research_mode="embedded",
        )
        service = ArthaMCPService(settings)
        with (
            patch.object(service, "_core_live_trading_enabled", return_value=True),
            self.assertRaises(AuthorizationError),
        ):
            service.start_workflow("sell_review")


class TestOAuthVerifier(unittest.IsolatedAsyncioTestCase):
    async def test_signature_issuer_audience_expiry_and_scopes_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            settings = replace(
                test_settings(Path(temp)),
                oauth_issuer="https://issuer.example",
                oauth_resource_url="https://artha.example/mcp",
                oauth_audience="artha-api",
                oauth_jwks_url="https://issuer.example/.well-known/jwks.json",
                oauth_algorithms=("RS256",),
            )
            key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            verifier = JWTTokenVerifier(settings)
            verifier._jwks.get_signing_key_from_jwt = lambda _token: SimpleNamespace(
                key=key.public_key()
            )
            now = datetime.now(UTC)
            claims = {
                "iss": settings.oauth_issuer,
                "aud": settings.oauth_audience,
                "sub": "user-1",
                "client_id": "client-1",
                "scope": "artha:read artha:trade",
                "exp": now + timedelta(minutes=5),
            }
            token = jwt.encode(claims, key, algorithm="RS256")
            verified = await verifier.verify_token(token)
            self.assertEqual(verified.client_id, "client-1")
            self.assertEqual(set(verified.scopes), {"artha:read", "artha:trade"})
            bad = jwt.encode({**claims, "aud": "wrong"}, key, algorithm="RS256")
            self.assertIsNone(await verifier.verify_token(bad))


class TestMarkets(unittest.TestCase):
    def test_us_and_india_symbol_normalization(self) -> None:
        us = normalize_instrument("brk.b", market="US")
        india = normalize_instrument("reliance.ns", market="IN")
        bse = normalize_instrument("500325.bo", market="IN")
        self.assertEqual((us.symbol, us.research_symbol), ("BRK.B", "BRK.B"))
        self.assertEqual(
            (india.symbol, india.exchange, india.research_symbol),
            ("RELIANCE", "NSE", "RELIANCE.NS"),
        )
        self.assertEqual(
            (bse.symbol, bse.exchange, bse.research_symbol),
            ("500325", "BSE", "500325.BO"),
        )

    def test_india_session_is_0915_to_1530_ist_and_labeled_estimate(self) -> None:
        profile = get_market_profile("IN")
        open_time = datetime(2026, 8, 11, 5, 0, tzinfo=UTC)
        closed_time = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)
        exact_close = datetime(2026, 8, 11, 10, 0, tzinfo=UTC)
        self.assertTrue(profile.session_status(open_time)["regular_session_estimate"])
        self.assertFalse(
            profile.session_status(closed_time)["regular_session_estimate"]
        )
        self.assertFalse(
            profile.session_status(exact_close)["regular_session_estimate"]
        )
        self.assertEqual(
            profile.session_status(open_time)["calendar_confidence"],
            "weekday_clock_estimate",
        )

    def test_host_orchestrated_research_is_explicitly_not_council_complete(
        self,
    ) -> None:
        caps = HostOrchestratedResearchAdapter().capabilities
        self.assertFalse(caps["embedded_workflows"])
        self.assertIn("IN", caps["markets"])

    def test_naive_indian_broker_time_is_interpreted_as_ist(self) -> None:
        parsed = parse_broker_timestamp(
            "2026-08-11 09:15:00", default_timezone="Asia/Kolkata"
        )
        self.assertEqual(parsed, datetime(2026, 8, 11, 3, 45, tzinfo=UTC))


class TestSnapshotAdapter(unittest.IsolatedAsyncioTestCase):
    async def test_oversized_snapshot_is_rejected_before_json_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "snapshot.json"
            with path.open("wb") as output:
                output.truncate(MAX_SNAPSHOT_BYTES + 1)
            adapter = SnapshotBrokerAdapter(path)
            health = await adapter.health()
            self.assertEqual(health["status"], "FAIL")
            self.assertIn("exceeds", health["message"])

    async def test_valid_snapshot_parses_portfolio_and_only_open_orders(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "snapshot.json"
            path.write_text(
                json.dumps(
                    {
                        "generated_at": datetime.now(UTC).isoformat(),
                        "validation": {"status": "PASS", "warnings": []},
                        "account": {"rhs_account_number": "12345678"},
                        "portfolio": {
                            "buying_power": {"buying_power": "120.50"},
                            "cash": "100.00",
                            "total_value": "350.25",
                        },
                        "positions": [
                            {
                                "symbol": "V",
                                "quantity": "0.5",
                                "average_buy_price": "300",
                            }
                        ],
                        "orders": [
                            {"id": "done", "state": "filled"},
                            {"id": "open", "state": "queued"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            adapter = SnapshotBrokerAdapter(path)
            health = await adapter.health()
            portfolio = await adapter.portfolio()
            self.assertEqual(health["status"], "PASS")
            self.assertTrue(portfolio.account.fresh)
            self.assertEqual(portfolio.account.buying_power, 120.5)
            self.assertEqual(portfolio.account.equity, 350.25)
            self.assertEqual(portfolio.account.account_id_masked, "****5678")
            self.assertEqual(len(portfolio.positions), 1)
            self.assertEqual([row["id"] for row in portfolio.open_orders], ["open"])

    async def test_failed_validation_is_never_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "snapshot.json"
            path.write_text(
                json.dumps(
                    {
                        "generated_at": datetime.now(UTC).isoformat(),
                        "validation": {
                            "status": "FAIL",
                            "warnings": ["account mismatch"],
                        },
                        "positions": [],
                        "orders": [],
                    }
                ),
                encoding="utf-8",
            )
            adapter = SnapshotBrokerAdapter(path)
            self.assertEqual((await adapter.health())["status"], "WARN")
            portfolio = await adapter.portfolio()
            self.assertFalse(portfolio.account.fresh)
            self.assertTrue(any("FAIL" in item for item in portfolio.warnings))

    async def test_snapshot_without_timezone_is_never_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "snapshot.json"
            path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-08-11T12:00:00",
                        "validation": {"status": "PASS", "warnings": []},
                        "positions": [],
                        "orders": [],
                    }
                ),
                encoding="utf-8",
            )
            adapter = SnapshotBrokerAdapter(path)
            self.assertEqual((await adapter.health())["status"], "WARN")
            self.assertFalse((await adapter.portfolio()).account.fresh)


if __name__ == "__main__":
    unittest.main()
