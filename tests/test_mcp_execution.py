from __future__ import annotations

import asyncio
import json
import math
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from artha_mcp.execution import ExecutionCoordinator
from artha_mcp.models import (
    AccessMode,
    AccountSnapshot,
    InstrumentRef,
    MarketCode,
    OrderPreview,
    OrderRequest,
    Position,
    Quote,
)
from artha_mcp.security import AuthorizationError, CapabilityPolicy

from .mcp_helpers import FakeBroker, test_settings, us_instrument


class TestExecutionCoordinator(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.now = datetime(2026, 8, 11, 15, 0, tzinfo=UTC)
        self.settings = test_settings(self.root)
        self.broker = FakeBroker(now=self.now)
        self.engine = ExecutionCoordinator(
            self.settings,
            CapabilityPolicy(self.settings),
            self.broker,
            now=lambda: self.now,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def order(action_id: str = "action-1", **changes) -> OrderRequest:
        payload = {
            "action_id": action_id,
            "instrument": us_instrument(),
            "side": "buy",
            "order_type": "market",
            "notional": 20.0,
            "max_price": 101.0,
        }
        payload.update(changes)
        return OrderRequest(**payload)

    async def test_exact_receipt_happy_path_and_single_use(self) -> None:
        receipt = await self.engine.create_preview(self.order())
        self.assertEqual(receipt["status"], "PASS")
        placed = await self.engine.place(receipt["receipt_id"])
        self.assertEqual(placed["status"], "PASS")
        self.assertEqual(self.broker.preview_calls, 2)
        self.assertEqual(self.broker.place_calls, 1)
        with self.assertRaisesRegex(ValueError, "not ready"):
            await self.engine.place(receipt["receipt_id"])

    async def test_live_adapter_without_order_status_cannot_place(self) -> None:
        broker = FakeBroker(now=self.now, sandbox=False, order_status=False)
        engine = ExecutionCoordinator(
            self.settings, CapabilityPolicy(self.settings), broker, now=lambda: self.now
        )
        receipt = await engine.create_preview(self.order("no-live-status"))

        result = await engine.place(receipt["receipt_id"])

        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("duplicate orders", result["message"])
        self.assertFalse(result["reconciliation_available"])
        self.assertEqual(broker.preview_calls, 1)
        self.assertEqual(broker.place_calls, 0)

    async def test_sandbox_without_order_status_is_submission_only(self) -> None:
        broker = FakeBroker(now=self.now, sandbox=True, order_status=False)
        engine = ExecutionCoordinator(
            self.settings, CapabilityPolicy(self.settings), broker, now=lambda: self.now
        )
        receipt = await engine.create_preview(self.order("sandbox-submit"))

        result = await engine.place(receipt["receipt_id"])

        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["sandbox_submission_only"])
        self.assertFalse(result["reconciliation_required"])
        self.assertFalse(result["reconciliation_available"])
        self.assertEqual(broker.preview_calls, 2)
        self.assertEqual(broker.place_calls, 1)

        reconciled = await engine.reconcile(receipt["receipt_id"])
        self.assertEqual(reconciled["status"], "WARN")
        self.assertFalse(reconciled["reconciliation_available"])
        self.assertFalse(reconciled["order_retried"])
        with self.assertRaisesRegex(ValueError, "not ready"):
            await engine.place(receipt["receipt_id"])

    async def test_final_preview_failure_is_blocked_not_unknown(self) -> None:
        receipt = await self.engine.create_preview(self.order())
        self.broker.preview_effects = [RuntimeError("provider down")]
        result = await self.engine.place(receipt["receipt_id"])
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("no order call was made", result["message"])
        self.assertEqual(self.broker.place_calls, 0)
        self.assertEqual(
            self.engine.get_receipt(receipt["receipt_id"])["status"], "blocked"
        )

    async def test_network_ambiguous_place_is_unknown_and_not_retryable(self) -> None:
        receipt = await self.engine.create_preview(self.order())
        self.broker.place_exception = TimeoutError("after submit")
        result = await self.engine.place(receipt["receipt_id"])
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(
            self.engine.get_receipt(receipt["receipt_id"])["status"], "unknown"
        )
        with self.assertRaisesRegex(ValueError, "not ready"):
            await self.engine.place(receipt["receipt_id"])

    async def test_unknown_receipt_reconciles_by_action_tag_without_retry(self) -> None:
        order = self.order("recover-44")
        receipt = await self.engine.create_preview(order)
        self.broker.place_exception = TimeoutError("after submit")
        await self.engine.place(receipt["receipt_id"])
        self.broker.order_rows = [
            {
                "order_id": "broker-44",
                "tag": "recover-44",
                "status": "complete",
                "access_token": "must-not-leak",
            }
        ]
        result = await self.engine.reconcile(receipt["receipt_id"])
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["execution_status"], "filled")
        self.assertEqual(self.broker.place_calls, 1)
        self.assertEqual(
            result["result"]["broker_response"]["access_token"], "[REDACTED]"
        )
        self.assertEqual(
            self.engine.get_receipt(receipt["receipt_id"])["status"], "filled"
        )

    async def test_missing_ambiguous_order_remains_unknown_and_is_not_retried(
        self,
    ) -> None:
        receipt = await self.engine.create_preview(self.order("missing-55"))
        self.broker.place_exception = TimeoutError("after submit")
        await self.engine.place(receipt["receipt_id"])
        result = await self.engine.reconcile(receipt["receipt_id"])
        self.assertEqual(result["status"], "WARN")
        self.assertEqual(result["execution_status"], "unknown")
        self.assertEqual(self.broker.place_calls, 1)

    async def test_matching_reconciliation_tag_without_order_id_stays_unknown(
        self,
    ) -> None:
        receipt = await self.engine.create_preview(self.order("recover-no-id"))
        self.broker.place_exception = TimeoutError("after submit")
        await self.engine.place(receipt["receipt_id"])
        self.broker.order_rows = [
            {"tag": "recover-no-id", "status": "complete", "filled_quantity": 1}
        ]

        result = await self.engine.reconcile(receipt["receipt_id"])

        self.assertEqual(result["status"], "WARN")
        self.assertEqual(result["execution_status"], "unknown")
        self.assertFalse(result["order_retried"])
        self.assertEqual(self.broker.place_calls, 1)
        self.assertEqual(
            self.engine.get_receipt(receipt["receipt_id"])["status"], "unknown"
        )

    async def test_filled_action_id_cannot_be_placed_twice(self) -> None:
        order = self.order("once-66")
        first = await self.engine.create_preview(order)
        await self.engine.place(first["receipt_id"])
        self.broker.order_rows = [
            {"order_id": "ord-once-66", "tag": "once-66", "status": "filled"}
        ]
        await self.engine.reconcile(first["receipt_id"])
        second = await self.engine.create_preview(order)
        with self.assertRaisesRegex(ValueError, "already has"):
            await self.engine.place(second["receipt_id"])
        self.assertEqual(self.broker.place_calls, 1)

    async def test_open_order_with_a_partial_fill_reconciles_as_partial(self) -> None:
        receipt = await self.engine.create_preview(self.order("partial-77"))
        await self.engine.place(receipt["receipt_id"])
        self.broker.order_rows = [
            {
                "order_id": "ord-partial-77",
                "tag": "partial-77",
                "status": "OPEN",
                "quantity": 10,
                "filled_quantity": 4,
                "pending_quantity": 6,
            }
        ]
        result = await self.engine.reconcile(receipt["receipt_id"])
        self.assertEqual(result["execution_status"], "partially_filled")

    async def test_duplicate_broker_tag_prevents_second_order(self) -> None:
        order = self.order("immutable-22")
        self.broker.order_rows = [
            {"order_id": "already-there", "tag": "immutable-22", "status": "open"}
        ]
        receipt = await self.engine.create_preview(order)
        result = await self.engine.place(receipt["receipt_id"])
        self.assertTrue(result["duplicate_prevented"])
        self.assertEqual(result["result"]["broker_order_id"], "already-there")
        self.assertEqual(self.broker.place_calls, 0)

    async def test_matching_broker_tag_without_order_id_stays_unknown(self) -> None:
        order = self.order("immutable-no-id")
        self.broker.order_rows = [{"tag": "immutable-no-id", "status": "open"}]
        receipt = await self.engine.create_preview(order)
        result = await self.engine.place(receipt["receipt_id"])
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertTrue(result["duplicate_prevented"])
        self.assertEqual(self.broker.place_calls, 0)

    async def test_expired_receipt_stays_expired(self) -> None:
        receipt = await self.engine.create_preview(self.order())
        self.now += timedelta(seconds=31)
        with self.assertRaisesRegex(ValueError, "expired"):
            await self.engine.place(receipt["receipt_id"])
        self.assertEqual(
            self.engine.get_receipt(receipt["receipt_id"])["status"], "expired"
        )

    async def test_tampered_stored_order_is_detected(self) -> None:
        receipt = await self.engine.create_preview(self.order())
        with self.engine._connect() as conn:
            payload = json.loads(
                conn.execute(
                    "SELECT order_json FROM receipts WHERE receipt_id=?",
                    (receipt["receipt_id"],),
                ).fetchone()[0]
            )
            payload["notional"] = 99
            conn.execute(
                "UPDATE receipts SET order_json=? WHERE receipt_id=?",
                (json.dumps(payload), receipt["receipt_id"]),
            )
        with self.assertRaisesRegex(ValueError, "integrity"):
            await self.engine.place(receipt["receipt_id"])
        self.assertEqual(self.broker.place_calls, 0)

    async def test_missing_proof_and_price_drift_block_preview(self) -> None:
        order = self.order(max_price=99.0)
        preview = self.broker.good_preview(order)
        payload = preview.model_dump(mode="python")
        payload["broker_proof"] = {"instrument": True, "quote": True, "funds": True}
        self.broker.preview_effects = [OrderPreview.model_validate(payload)]
        result = await self.engine.create_preview(order)
        self.assertEqual(result["status"], "BLOCKED")
        reasons = " ".join(result["preview"]["reasons"])
        self.assertIn("maximum price", reasons)
        self.assertIn("order_preview", reasons)

    async def test_quote_age_and_spread_are_independent_hard_gates(self) -> None:
        order = self.order()
        preview = self.broker.good_preview(order)
        stale_wide = Quote(
            instrument=order.instrument,
            bid=95,
            ask=105,
            last=100,
            timestamp=self.now - timedelta(minutes=2),
            source="fake",
            fresh=True,
        )
        payload = preview.model_dump(mode="python")
        payload["quote"] = stale_wide
        self.broker.preview_effects = [OrderPreview.model_validate(payload)]
        result = await self.engine.create_preview(order)
        reasons = " ".join(result["preview"]["reasons"])
        self.assertIn("spread", reasons.lower())
        self.assertIn("old", reasons.lower())

    async def test_broker_explicitly_stale_quote_is_blocked(self) -> None:
        order = self.order()
        preview = self.broker.good_preview(order)
        payload = preview.model_dump(mode="python")
        payload["quote"] = preview.quote.model_copy(update={"fresh": False})
        self.broker.preview_effects = [OrderPreview.model_validate(payload)]
        result = await self.engine.create_preview(order)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn(
            "marked the quote as stale", " ".join(result["preview"]["reasons"])
        )

    async def test_quote_without_timezone_or_from_future_is_blocked(self) -> None:
        order = self.order()
        for timestamp in (
            self.now.replace(tzinfo=None),
            self.now + timedelta(minutes=1),
        ):
            preview = self.broker.good_preview(order)
            payload = preview.model_dump(mode="python")
            payload["quote"] = Quote(
                instrument=order.instrument,
                bid=99.95,
                ask=100.05,
                last=100,
                timestamp=timestamp,
                source="fake",
                fresh=True,
            )
            self.broker.preview_effects = [OrderPreview.model_validate(payload)]
            result = await self.engine.create_preview(order)
            self.assertEqual(result["status"], "BLOCKED")

    async def test_preview_identity_mismatch_and_understated_value_are_blocked(
        self,
    ) -> None:
        order = self.order(notional=None, quantity=1, max_price=101)
        preview = self.broker.good_preview(order)
        payload = preview.model_dump(mode="python")
        payload["action_id"] = "other-action"
        payload["estimated_value"] = 1
        payload["quote"] = preview.quote.model_copy(
            update={"instrument": us_instrument("OTHER")}
        )
        self.broker.preview_effects = [OrderPreview.model_validate(payload)]
        result = await self.engine.create_preview(order)
        reasons = " ".join(result["preview"]["reasons"])
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("action id", reasons)
        self.assertIn("instrument", reasons)
        self.assertGreater(result["preview"]["estimated_value"], 100)

    async def test_atomic_daily_limit_allows_only_one_concurrent_claim(self) -> None:
        settings = replace(self.settings, max_daily_orders=1)
        broker = FakeBroker(now=self.now)
        engine = ExecutionCoordinator(
            settings, CapabilityPolicy(settings), broker, now=lambda: self.now
        )
        first = await engine.create_preview(self.order("one"))
        second = await engine.create_preview(self.order("two"))
        results = await asyncio.gather(
            engine.place(first["receipt_id"]),
            engine.place(second["receipt_id"]),
            return_exceptions=True,
        )
        self.assertEqual(
            sum(
                isinstance(value, dict) and value.get("status") == "PASS"
                for value in results
            ),
            1,
        )
        self.assertEqual(sum(isinstance(value, ValueError) for value in results), 1)
        self.assertEqual(broker.place_calls, 1)

    async def test_closed_session_blocks_non_sandbox_adapter(self) -> None:
        saturday = datetime(2026, 8, 15, 15, 0, tzinfo=UTC)
        broker = FakeBroker(now=saturday, sandbox=False)
        engine = ExecutionCoordinator(
            self.settings, CapabilityPolicy(self.settings), broker, now=lambda: saturday
        )
        result = await engine.create_preview(self.order())
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("regular US", " ".join(result["preview"]["reasons"]))

    async def test_local_policy_cannot_be_bypassed_by_oauth_scope(self) -> None:
        locked = replace(
            self.settings,
            access_mode=AccessMode.READ_ONLY,
            operations_enabled=False,
            trading_enabled=False,
            kill_switch=True,
        )
        engine = ExecutionCoordinator(
            locked, CapabilityPolicy(locked), self.broker, now=lambda: self.now
        )
        receipt = await engine.create_preview(self.order())
        with self.assertRaises(AuthorizationError):
            await engine.place(receipt["receipt_id"], oauth_scopes={"artha:trade"})


class TestOrderContract(unittest.TestCase):
    def test_side_specific_price_guards(self) -> None:
        with self.assertRaisesRegex(ValueError, "only valid for sell"):
            TestExecutionCoordinator.order(min_price=95)
        with self.assertRaisesRegex(ValueError, "only valid for buy"):
            TestExecutionCoordinator.order(
                side="sell", quantity=1, notional=None, max_price=100
            )

    def test_market_orders_require_side_specific_price_guards(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_price is required"):
            TestExecutionCoordinator.order(max_price=None)
        with self.assertRaisesRegex(ValueError, "min_price is required"):
            TestExecutionCoordinator.order(
                side="sell",
                quantity=1,
                notional=None,
                max_price=None,
            )

    def test_limit_order_cannot_exceed_its_approved_price(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            TestExecutionCoordinator.order(
                order_type="limit", limit_price=102, max_price=101
            )

    def test_unsafe_action_identifier_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TestExecutionCoordinator.order(action_id="bad action;drop")

    def test_non_finite_order_and_quote_values_are_rejected(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                TestExecutionCoordinator.order(notional=value)
            with self.assertRaises(ValueError):
                Quote(
                    instrument=us_instrument(),
                    bid=99,
                    ask=value,
                    source="bad",
                )
            with self.assertRaises(ValueError):
                Position(instrument=us_instrument(), quantity=value)
            with self.assertRaises(ValueError):
                AccountSnapshot(broker="bad", currency="USD", equity=value)

    def test_crossed_market_quote_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ask cannot be below bid"):
            Quote(
                instrument=us_instrument(),
                bid=101,
                ask=100,
                last=100.5,
                source="bad",
            )

    def test_direct_instrument_contract_rejects_unvalidated_symbols(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid US equity symbol"):
            InstrumentRef(
                market=MarketCode.US,
                symbol="../AAPL",
                exchange="US",
                currency="USD",
                research_symbol="../AAPL",
            )
        with self.assertRaisesRegex(ValueError, "surrounding whitespace"):
            InstrumentRef(
                market=MarketCode.INDIA,
                symbol="RELIANCE",
                exchange="NSE",
                currency="INR",
                research_symbol="RELIANCE.NS",
                broker_instrument_id=" NSE_EQ|RELIANCE ",
            )


if __name__ == "__main__":
    unittest.main()
