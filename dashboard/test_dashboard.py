"""Focused regression tests for the read-only ARTHA dashboard."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from dashboard import server


class DashboardPureFunctionTests(unittest.TestCase):
    def test_isolated_equity_spike_is_removed_without_touching_sustained_move(self):
        points = [
            {"t": "2026-08-01T00:00:00+00:00", "v": 350.0},
            {"t": "2026-08-02T00:00:00+00:00", "v": 351.0},
            {"t": "2026-08-03T00:00:00+00:00", "v": 455.0},
            {"t": "2026-08-04T00:00:00+00:00", "v": 352.0},
            {"t": "2026-08-05T00:00:00+00:00", "v": 353.0},
        ]
        clean, removed = server._remove_isolated_equity_outliers(points)
        self.assertEqual(removed, 1)
        self.assertEqual([row["v"] for row in clean], [350.0, 351.0, 352.0, 353.0])

        sustained = [
            {"t": f"2026-08-0{index + 1}T00:00:00+00:00", "v": value}
            for index, value in enumerate([350.0, 352.0, 410.0, 412.0, 414.0])
        ]
        clean, removed = server._remove_isolated_equity_outliers(sustained)
        self.assertEqual(removed, 0)
        self.assertEqual(clean, sustained)

    def test_block_reason_prefers_broker_gate_evidence(self):
        action = {
            "payload_json": json.dumps(
                {
                    "broker_result": {
                        "response": {
                            "blocked_reasons": [
                                "Live ask is above the no-chase cap."
                            ]
                        }
                    }
                }
            ),
            "message": "generic message",
        }
        self.assertEqual(
            server._extract_block_reason(action),
            "Live ask is above the no-chase cap.",
        )

    def test_filled_execution_has_plain_language_outcome(self):
        action = {
            "status": "filled",
            "side": "buy",
            "action_type": "auto_buy",
            "result_json": json.dumps({"average_price": 42.5, "quantity": 0.4}),
            "execution_order_row": 8,
            "updated_at": "2026-08-11T18:00:00+00:00",
        }
        orders = {8: {"notional": 17.0, "estimated_price": 42.4}}
        result = server._execution_view(action, orders)
        self.assertEqual(result["status"], "filled")
        self.assertEqual(result["label"], "Filled automatically")
        self.assertEqual(result["notional"], 17.0)
        self.assertEqual(result["fill_price"], 42.5)

    def test_openclaw_probe_uses_an_existing_absolute_binary_in_production(self):
        if server.OPENCLAW_BIN != "openclaw":
            self.assertTrue(Path(server.OPENCLAW_BIN).exists())
            self.assertTrue(Path(server.OPENCLAW_BIN).is_absolute())


class DashboardLiveContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not server.DB.exists():
            raise unittest.SkipTest("ARTHA live journal is not present")
        cls.payload = server.build_payload()
        cls.serialized = json.dumps(cls.payload, default=str)

    def test_versioned_complete_payload(self):
        self.assertEqual(self.payload["schema_version"], 2)
        for key in (
            "vitals", "performance", "equity", "positions", "sectors",
            "decisions", "sell_reviews", "pipeline", "learning", "system",
            "alarms", "schedule", "privacy",
        ):
            self.assertIn(key, self.payload)

    def test_all_supervisor_checks_are_visible(self):
        raw = server._supervisor()
        self.assertGreaterEqual(len(raw["checks"]), 20)
        self.assertEqual(len(self.payload["system"]["checks"]), len(raw["checks"]))

    def test_current_live_limits_and_autonomy_are_not_hard_coded_stale_values(self):
        policy = self.payload["system"]["policy"]
        self.assertEqual(policy["max_positions"], 20)
        self.assertEqual(policy["max_buys_per_day"], 5)
        self.assertTrue(policy["auto_buy_enabled"])
        self.assertTrue(policy["auto_sell_enabled"])

    def test_dashboard_feed_does_not_expose_broker_account_identifier(self):
        self.assertNotIn("account_number", self.serialized)
        snapshot = server._read_json(server.DATA / "robinhood" / "latest_snapshot.json", {})
        expected = str(
            ((snapshot.get("validation") or {}).get("account_check") or {})
            .get("checks", {})
            .get("actual_account_masked", "")
        )
        if expected:
            self.assertNotIn(expected, self.serialized)

    def test_learning_maturity_is_explicit(self):
        sell = self.payload["learning"]["sell"]
        self.assertIn("completed", sell)
        self.assertIn("minimum", sell)
        self.assertEqual(sell["ready"], sell["completed"] >= sell["minimum"])

    def test_recent_decisions_include_execution_stage(self):
        self.assertTrue(self.payload["decisions"])
        for decision in self.payload["decisions"]:
            self.assertIn("execution", decision)
            self.assertIn("status", decision["execution"])
            self.assertIn("reason", decision["execution"])

    def test_interface_has_every_requested_view_and_no_stale_approval_copy(self):
        html = Path(server.INDEX_FILE).read_text(encoding="utf-8")
        app = Path(server.APP_FILE).read_text(encoding="utf-8")
        for view in ("overview", "portfolio", "decisions", "process", "learning", "health"):
            self.assertIn(f'id="view-{view}"', html)
        combined = f"{html}\n{app}".lower()
        for stale in ("2 buys/day", "6 positions", "awaiting your ok", "wait for sarath"):
            self.assertNotIn(stale, combined)


if __name__ == "__main__":
    unittest.main()
