from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class TestAlphaPipelineHardening(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_insider_cluster_is_disclosed_deduplicated_and_no_lookahead(self):
        from artha.alpha_shadow import build_insider_shadow_signal

        rows = []
        for index, name in enumerate(("A Person", "B Person", "C Person")):
            rows.append(
                {
                    "name": name,
                    "filingDate": f"2026-08-{5 + index:02d}",
                    "transactionDate": f"2026-08-{4 + index:02d}",
                    "transactionCode": "P",
                    "change": 100 + index,
                    "transactionPrice": 20,
                }
            )
        rows.append(dict(rows[0]))
        rows.append(
            {
                "name": "Future Person",
                "filingDate": "2026-08-11",
                "transactionDate": "2026-08-09",
                "transactionCode": "P",
                "change": 500,
                "transactionPrice": 20,
            }
        )
        rows.append(
            {
                "name": "Grant Person",
                "filingDate": "2026-08-08",
                "transactionDate": "2026-08-08",
                "transactionCode": "A",
                "change": 500,
                "transactionPrice": 0,
            }
        )
        signal = build_insider_shadow_signal({"data": rows}, as_of=date(2026, 8, 10))

        self.assertTrue(signal["cluster_confirmed"])
        self.assertEqual(signal["unique_buyers_14d"], 3)
        self.assertEqual(signal["qualifying_purchase_count_30d"], 3)
        self.assertEqual(signal["excluded_counts"]["duplicate"], 1)
        self.assertEqual(signal["excluded_counts"]["future_or_missing_filing"], 1)
        self.assertEqual(signal["live_score_impact"], 0.0)

    def test_alpha_features_remain_shadow_only_and_explorer_needs_every_fence(self):
        from artha.alpha_shadow import build_alpha_shadow_signals

        stock = {
            "quote": {"price": 20, "marketCap": 1_500_000_000, "avgVolume": 1_000_000},
            "profile": {"ipoDate": "2020-01-01", "isEtf": False, "isFund": False},
            "cash_flow": [{"freeCashFlow": 25_000_000}],
            "income_statement": [{"operatingIncome": 10_000_000, "netIncome": 8_000_000}],
            "funnel_candidate": {
                "lottery_flag": False,
                "industry_momentum_shadow": {"peer_count": 12, "z_score": 1.4, "live_score_impact": 0.0},
                "episodic_pivot_shadow": {"quality_score": 82, "live_score_impact": 0.0},
            },
        }
        result = build_alpha_shadow_signals(stock)

        self.assertEqual(result["mode"], "shadow_only")
        self.assertEqual(result["live_score_impact"], 0.0)
        self.assertTrue(result["quality_small_cap_explorer"]["qualified"])
        self.assertEqual(result["industry_momentum"]["live_score_impact"], 0.0)

    def test_shadow_rules_log_alpha_hypotheses_only_for_no_buy(self):
        from artha.shadow_rules import _candidate_shadow_rules

        payload = {
            "real_action": "DEFER",
            "opportunity_score": 40,
            "alpha_shadow_signals": {
                "insider_cluster": {"cluster_confirmed": True},
                "industry_momentum": {"peer_count": 10, "z_score": 1.5},
                "episodic_pivot": {"quality_score": 80},
                "quality_small_cap_explorer": {"qualified": True},
            },
        }
        ids = {row["rule_id"] for row in _candidate_shadow_rules(payload)}
        self.assertTrue(
            {
                "insider_cluster_starter_probe",
                "industry_momentum_starter_probe",
                "episodic_pivot_starter_probe",
                "quality_small_cap_explorer_probe",
            }.issubset(ids)
        )
        payload["real_action"] = "STARTER"
        buy_ids = {row["rule_id"] for row in _candidate_shadow_rules(payload)}
        self.assertFalse(any(rule_id in buy_ids for rule_id in ids))

    def test_catalyst_lane_accepts_real_collector_camel_case_fields(self):
        from artha.scanner import get_catalyst_candidates

        class Movers:
            def get_market_movers(self):
                return {
                    "gainers": [
                        {
                            "ticker": "EPX",
                            "price": 25,
                            "changePct": 18.5,
                            "volume": 3_000_000,
                        }
                    ]
                }

        class FakeFMP:
            def quote(self, symbol):
                return {
                    "symbol": symbol,
                    "price": 25,
                    "marketCap": 1_500_000_000,
                    "volume": 3_000_000,
                    "avgVolume": 1_000_000,
                }

            def company_profile(self, symbol):
                return {"isEtf": False, "isFund": False}

            def stock_news(self, symbol, limit=3):
                return []

        empty_earnings = SimpleNamespace(days_to_earnings=30, earnings_date=None, recent_surprises=[])
        with patch("artha.scanner.FMPCollector", return_value=FakeFMP()), patch(
            "artha.earnings_calendar.get_earnings_context", return_value=empty_earnings
        ), patch("artha.scanner._get_news_catalysts", return_value=[]):
            rows = get_catalyst_candidates(Movers(), max_candidates=3)

        self.assertEqual([row["symbol"] for row in rows], ["EPX"])
        self.assertEqual(rows[0]["gap_pct"], 18.5)
        self.assertEqual(rows[0]["catalyst_type"], "top_mover_volume")
        self.assertFalse(rows[0]["catalyst_confirmed"])
        self.assertFalse(rows[0]["live_eligible"])
        self.assertEqual(rows[0]["routing_status"], "research_watch")
        self.assertEqual(rows[0]["episodic_pivot_shadow"]["live_score_impact"], 0.0)

    def test_catalyst_lane_uses_recent_positive_report_not_upcoming_earnings(self):
        from artha.scanner import get_catalyst_candidates

        today = datetime.now(timezone.utc).date().isoformat()

        class Movers:
            def get_market_movers(self):
                return {
                    "gainers": [
                        {"ticker": "BEAT", "price": 25, "changePct": 10, "volume": 2_000_000},
                    ]
                }

        class FakeFMP:
            def quote(self, symbol):
                return {
                    "symbol": symbol,
                    "price": 25,
                    "marketCap": 1_500_000_000,
                    "volume": 2_000_000,
                    "avgVolume": 1_000_000,
                }

            def company_profile(self, symbol):
                return {"isEtf": False, "isFund": False}

            def stock_news(self, symbol, limit=3):
                return []

        reported = SimpleNamespace(
            days_to_earnings=0,
            earnings_date=today,
            recent_surprises=[{"date": today, "surprise_pct": 18.0}],
        )
        with patch("artha.scanner.FMPCollector", return_value=FakeFMP()), patch(
            "artha.earnings_calendar.get_earnings_context", return_value=reported
        ), patch("artha.scanner._get_news_catalysts", return_value=[]):
            rows = get_catalyst_candidates(Movers(), max_candidates=3)

        self.assertEqual(rows[0]["catalyst_type"], "earnings")
        self.assertTrue(rows[0]["catalyst_confirmed"])
        self.assertTrue(rows[0]["live_eligible"])
        self.assertEqual(rows[0]["earnings_report_date"], today)

        upcoming_only = SimpleNamespace(
            days_to_earnings=0,
            earnings_date=today,
            recent_surprises=[],
        )
        with patch("artha.scanner.FMPCollector", return_value=FakeFMP()), patch(
            "artha.earnings_calendar.get_earnings_context", return_value=upcoming_only
        ), patch("artha.scanner._get_news_catalysts", return_value=[]):
            no_rows = get_catalyst_candidates(Movers(), max_candidates=3)
        self.assertEqual(no_rows, [])

    def test_funnel_guarantees_slots_only_to_identified_catalysts(self):
        from artha.funnel import PromotionFunnel

        rows = [
            {"symbol": "RAW", "live_eligible": False, "catalyst_type": "top_mover_volume"},
            {"symbol": "BEAT", "live_eligible": True, "catalyst_type": "earnings"},
        ]
        with patch("artha.scanner.get_catalyst_candidates", return_value=rows), patch(
            "artha.rank_candidates.update_scan_signals"
        ) as update:
            live = PromotionFunnel()._catalyst_lane_candidates()

        self.assertEqual([row["symbol"] for row in live], ["BEAT"])
        persisted = update.call_args.args[0]
        self.assertEqual(set(persisted), {"RAW", "BEAT"})
        self.assertFalse(persisted["RAW"]["catalyst_live_eligible"])

    def test_low_rank_coverage_fails_buy_scan_closed_without_legacy_fallback(self):
        from artha.config import Config
        from artha.funnel import PromotionFunnel

        universe = [
            SimpleNamespace(
                symbol="AAA",
                name="AAA",
                sector="Technology",
                industry="Software",
                market_cap=2_000_000_000,
                volume=1_000_000,
                regime_score=0,
            ),
            SimpleNamespace(
                symbol="BBB",
                name="BBB",
                sector="Healthcare",
                industry="Biotech",
                market_cap=2_000_000_000,
                volume=1_000_000,
                regime_score=0,
            ),
        ]

        class Builder:
            def build_universe(self, **kwargs):
                return universe

        def incomplete_rank(*args, **kwargs):
            kwargs["coverage_out"].update(
                {"universe_size": 2, "history_count": 1, "history_coverage_pct": 0.5}
            )
            return [{"symbol": "AAA"}]

        funnel = PromotionFunnel.__new__(PromotionFunnel)
        funnel.universe_builder = Builder()
        with patch.object(Config, "SCAN_REQUIRE_MIN_RANK_COVERAGE", True), patch(
            "artha.rank_candidates.rank_universe", side_effect=incomplete_rank
        ), patch.object(funnel, "_fallback") as fallback:
            result = funnel.run({}, max_council_candidates=8, fallback_on_failure=True)

        self.assertEqual(result, [])
        self.assertFalse(fallback.called)

    def test_scanner_does_not_use_legacy_tickers_when_strict_funnel_is_empty(self):
        from artha.config import Config
        from artha.scanner import MarketScanner

        funnel = SimpleNamespace(run=lambda **kwargs: [])
        scanner = MarketScanner()
        with patch.object(Config, "SCAN_REQUIRE_MIN_RANK_COVERAGE", True), patch.object(
            scanner, "_get_funnel", return_value=funnel
        ), patch.object(scanner, "_legacy_candidates") as legacy:
            result = scanner.get_funnel_candidates({}, max_candidates=8)

        self.assertEqual(result, [])
        self.assertFalse(legacy.called)

    def test_common_side_lane_eligibility_routes_bad_data_to_research(self):
        from artha.liquidity import evaluate_buy_now_eligibility

        clean = {
            "quote": {"price": 20, "marketCap": 2_000_000_000, "avgVolume": 1_000_000},
            "profile": {"isEtf": False, "isFund": False},
        }
        bad = {
            "quote": {"price": 6, "marketCap": 80_000_000, "avgVolume": 100_000},
            "profile": {"isEtf": False, "isFund": False},
        }
        self.assertTrue(evaluate_buy_now_eligibility(clean)["passed"])
        result = evaluate_buy_now_eligibility(bad)
        self.assertFalse(result["passed"])
        self.assertIn("market_cap_below_buy_now_floor", result["reasons"])
        self.assertIn("average_dollar_volume_below_buy_now_floor", result["reasons"])

    def test_projected_sector_limit_covers_new_buys_and_adds(self):
        from artha.portfolio_risk import evaluate_projected_sector_limit

        state = {
            "total_value": 100,
            "positions": [
                {"ticker": "USB", "sector": "Financial Services", "market_value": 29},
            ],
        }
        add = evaluate_projected_sector_limit(
            ticker="USB",
            sector="Financial Services",
            portfolio_state=state,
            proposed_notional=2,
            max_sector_pct=0.30,
        )
        new = evaluate_projected_sector_limit(
            ticker="PNC",
            sector="Financial Services",
            portfolio_state=state,
            proposed_notional=2,
            max_sector_pct=0.30,
        )
        self.assertFalse(add["passed"])
        self.assertTrue(add["is_add"])
        self.assertFalse(new["passed"])
        self.assertFalse(new["is_add"])
        self.assertAlmostEqual(add["projected_sector_pct"], 0.31)

    def test_live_execution_guardrail_blocks_exact_projected_sector_exposure(self):
        from artha.execution import OrderIntent, RobinhoodExecutionGuardrails
        from artha.journal import DecisionJournal

        class AlwaysOpen:
            def is_market_open(self, now=None):
                return True

        journal = DecisionJournal(db_path=self.tmp_path / "execution.db")
        journal.save_decision_features(
            {
                "dossier_path": "sector.json",
                "generated_at": "2026-08-10T15:00:00+00:00",
                "ticker": "PNC",
                "final_verdict": "STARTER",
                "sector": "Financial Services",
                "evidence_count": 34,
                "feature_json": "{}",
            }
        )
        intent = OrderIntent(
            ticker="PNC",
            side="buy",
            order_type="market",
            quantity=0.2,
            notional=20,
            limit_price=100,
            estimated_price=100,
            decision_dossier_path="sector.json",
            dry_run=False,
        )
        state = {
            "total_value": 100,
            "cash_available": 70,
            "positions": [{"ticker": "USB", "sector": "Financial Services", "market_value": 29}],
        }
        with patch("artha.execution.PortfolioStateEngine.compute_state", return_value=state):
            result = RobinhoodExecutionGuardrails(AlwaysOpen()).evaluate(
                intent,
                {"price": 100, "bid": 99.99, "ask": 100, "volume": 1_000_000},
                journal,
                now=datetime(2026, 8, 10, 15, 0, tzinfo=timezone.utc),
            )
        self.assertFalse(result.checks["sector_concentration"]["passed"])
        self.assertIn("49.0%", " ".join(result.reasons))

    def test_final_clearance_rechecks_sector_and_does_not_call_llm_when_blocked(self):
        from artha.robinhood_bridge import run_final_clearance_for_action

        action = {
            "action_id": "ta-sector",
            "ticker": "USB",
            "action_type": "auto_buy",
            "side": "buy",
            "payload_json": json.dumps(
                {
                    "intent": {
                        "ticker": "USB",
                        "side": "buy",
                        "order_type": "market",
                        "notional": 2,
                        "limit_price": 10,
                        "estimated_price": 10,
                        "decision_dossier_path": "usb.json",
                        "evidence": {"sector": "Financial Services"},
                    }
                }
            ),
            "result_json": json.dumps(
                {
                    "review_response": {"data": {"symbol": "USB"}},
                    "tradability_response": {"data": {"results": [{"symbol": "USB"}]}},
                    "review_gate": {"passed": True, "status": "PASS", "reasons": []},
                }
            ),
        }

        class FakeJournal:
            def __init__(self):
                self.updates = []

            def get_trade_action(self, action_id):
                return dict(action)

            def get_latest_decision_feature(self, ticker, dossier_path=None):
                return {"sector": "Financial Services"}

            def update_trade_action(self, action_id, fields):
                self.updates.append(fields)

        state = {
            "total_value": 100,
            "positions": [{"ticker": "USB", "sector": "Financial Services", "market_value": 29}],
        }
        journal = FakeJournal()
        with patch("artha.robinhood_bridge.PortfolioStateEngine.compute_state", return_value=state), patch(
            "artha.execution_officer.robinhood_review_final_clearance"
        ) as officer:
            result = run_final_clearance_for_action("ta-sector", journal=journal)
        self.assertFalse(result["allow_place"])
        self.assertFalse(officer.called)
        self.assertIn("31.0%", result["message"])

    def test_scan_signal_metadata_cannot_be_downgraded_by_mini_run(self):
        import artha.rank_candidates as rank

        with patch.object(rank, "SCAN_SIGNALS_DIR", self.tmp_path):
            rank.persist_scan_signals(
                {"AAA": {"price": 10}},
                scan_date="2026-08-10",
                universe_size=1_700,
                coverage_summary={"universe_size": 1_700, "history_count": 1_600},
            )
            rank.persist_scan_signals(
                {"BBB": {"price": 20}},
                scan_date="2026-08-10",
                universe_size=2,
                coverage_summary={"universe_size": 2, "history_count": 2},
            )
        payload = json.loads((self.tmp_path / "2026-08-10.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["universe_size"], 1_700)
        self.assertEqual(payload["coverage_summary"]["universe_size"], 1_700)
        self.assertEqual(set(payload["signals"]), {"AAA", "BBB"})

    def test_rank_coverage_audit_assigns_one_status_per_universe_symbol(self):
        import artha.rank_candidates as rank

        universe = [
            SimpleNamespace(symbol="AAA", market_cap=1, sector="Technology", industry="Software"),
            SimpleNamespace(symbol="BBB", market_cap=2, sector="Healthcare", industry="Biotech"),
        ]
        with patch.object(rank, "RANK_COVERAGE_DIR", self.tmp_path):
            summary = rank.persist_rank_coverage_audit(
                universe=universe,
                symbol_status={"AAA": "ranked", "BBB": "price_history_missing"},
                history_count=1,
                signal_count=1,
                ranked_count=1,
                lottery_excluded_count=0,
                regime_type="goldilocks",
                overlays=[],
                elapsed_seconds=5,
                timeout_seconds=10,
            )
        self.assertEqual(summary["status"], "WARN")
        artifact = json.loads(Path(summary["artifact_path"]).read_text(encoding="utf-8"))
        self.assertEqual(len(artifact["symbols"]), 2)
        self.assertEqual(artifact["summary"]["status_counts"]["price_history_missing"], 1)
        with patch.object(rank, "RANK_COVERAGE_DIR", self.tmp_path):
            rank.persist_rank_coverage_audit(
                universe=[SimpleNamespace(symbol="MINI", market_cap=1, sector="", industry="")],
                symbol_status={"MINI": "ranked"},
                history_count=1,
                signal_count=1,
                ranked_count=1,
                lottery_excluded_count=0,
                regime_type="goldilocks",
                overlays=[],
                elapsed_seconds=1,
                timeout_seconds=10,
            )
            latest = rank.load_latest_rank_coverage_summary(max_age_hours=1)
        self.assertEqual(latest["universe_size"], 2)
        self.assertEqual(latest["history_count"], 1)


if __name__ == "__main__":
    unittest.main()
