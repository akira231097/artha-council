"""Production-hardening tests for calibration, risk, and deterministic engines.

Run with:
    python -m artha.test_production_hardening
"""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch


def _sample_stock(price: float = 100.0) -> dict:
    return {
        "ticker": "TEST",
        "quote": {"price": price, "marketCap": 10_000_000_000},
        "profile": {"sector": "Technology", "industry": "Software", "beta": 1.2},
        "dcf": {"dcf": 125.0},
        "price_target_consensus": {
            "targetConsensus": 135.0,
            "targetMedian": 132.0,
            "targetLow": 90.0,
            "targetHigh": 160.0,
        },
        "ratios_ttm": {
            "priceToEarningsRatioTTM": 24.0,
            "priceToEarningsGrowthRatioTTM": 1.2,
            "priceToSalesRatioTTM": 5.0,
            "debtToEquityRatioTTM": 0.4,
            "currentRatioTTM": 1.8,
            "netProfitMarginTTM": 0.22,
        },
        "key_metrics_ttm": {
            "freeCashFlowYieldTTM": 0.045,
            "returnOnInvestedCapitalTTM": 0.18,
            "returnOnEquityTTM": 0.25,
        },
        "income_statement": [
            {"revenue": 1200, "eps": 2.4},
            {"revenue": 1100, "eps": 2.0},
            {"revenue": 1050, "eps": 1.9},
            {"revenue": 1000, "eps": 1.8},
        ],
        "cash_flow": [{"freeCashFlow": 180}],
        "recommendation_trends": {"net_upgrades_30d": 2, "net_downgrades_30d": 0, "consensus": "buy"},
        "analyst_estimates": {"next_q_eps_estimate": 2.5, "fy1_revenue_estimate": 5200},
        "technicals": {"rsi": 52, "sma_50": 96},
        "earnings_context": {"days_to_earnings": 21},
    }


class TestCouncilSourceHierarchy(unittest.TestCase):
    def test_web_research_is_context_not_source_of_truth(self):
        import inspect

        from artha import prompts
        from artha.agentic_diligence import SOURCE_HIERARCHY_TEXT
        from artha.researcher import ResearchDesk

        self.assertIn("SOURCE HIERARCHY (MANDATORY)", prompts.SHARED_CONTEXT)
        self.assertIn("Current-web/search results and the INTELLIGENCE BRIEF are context only", prompts.SHARED_CONTEXT)
        self.assertIn("Do NOT let a web article override FMP, SEC EDGAR, Massive, yfinance, Finnhub", prompts.SHARED_CONTEXT)
        self.assertIn("Current-web/search can reveal risks", prompts.CONTRARIAN_ANALYST)
        self.assertIn("Do not award a BUY-like verdict mainly because of web/news narrative", prompts.SYNTHESIS_PROMPT)
        self.assertIn("structured price/fundamental/valuation/technical", prompts.SYNTHESIS_PROMPT)
        self.assertIn("does NOT support resting", prompts.SYNTHESIS_PROMPT)
        self.assertIn("fractional limit orders", prompts.SYNTHESIS_PROMPT)
        self.assertIn("create an entry watch", prompts.SYNTHESIS_PROMPT)

        self.assertIn("structured provider data is the source of truth", SOURCE_HIERARCHY_TEXT)
        self.assertIn("Current-web/search evidence is context", SOURCE_HIERARCHY_TEXT)
        self.assertIn("must not override structured provider data", SOURCE_HIERARCHY_TEXT)

        gpt_source = inspect.getsource(ResearchDesk._synthesize_brief)
        gemini_source = inspect.getsource(ResearchDesk._synthesize_brief_gemini_fallback)
        self.assertIn("This Intelligence Brief is web/news context, not the source of truth", gpt_source)
        self.assertIn("Mark web-only facts as source-reported context", gpt_source)
        self.assertIn("This Intelligence Brief is web/news context, not the source of truth", gemini_source)
        self.assertIn("Mark web-only facts as source-reported context", gemini_source)


class TestMonitoringSignalGates(unittest.TestCase):
    def test_held_news_identity_gate_blocks_unrelated_apple_for_jnj(self):
        from artha.sentinel import NewsSentinel

        collector = Mock()
        collector.fmp.company_profile.return_value = {"companyName": "Johnson & Johnson"}
        sentinel = NewsSentinel(collector=collector)

        self.assertFalse(
            sentinel._is_company_specific_news(
                "JNJ",
                "Apple stock gains ahead of WWDC: What are investors expecting?",
                "Apple AI commentary and WWDC expectations.",
                {"source": "Benzinga"},
            )
        )
        self.assertTrue(
            sentinel._is_company_specific_news(
                "JNJ",
                "Johnson & Johnson to Acquire Firefly Bio in $1 Billion Deal",
                "J&J expands oncology pipeline.",
                {"source": "Yahoo"},
            )
        )
        self.assertTrue(
            sentinel._is_company_specific_news(
                "JNJ",
                "Pipeline update",
                "Drug development update.",
                {"symbols": ["JNJ"]},
            )
        )

    def test_liquidity_gate_does_not_hard_fail_on_current_open_volume(self):
        from artha.config import Config
        from artha.council import hard_risk_gate
        from artha.liquidity import resolve_average_volume

        stock_data = {
            "quote": {"price": 34.0, "marketCap": 8_000_000_000, "volume": 3_582},
            "profile": {"sector": "Communication Services"},
            "ratios_ttm": {},
            "key_metrics_ttm": {"bookValuePerShareTTM": 4},
        }
        volume_info = resolve_average_volume(stock_data)
        self.assertFalse(volume_info["is_average"])
        passed, reason = hard_risk_gate(
            "TEST",
            stock_data,
            portfolio_state={"positions": [], "total_value": 350, "cash_available": 350},
            sentinel_alerts=[],
            config=Config,
        )
        self.assertTrue(passed)
        self.assertEqual(reason, "")

    def test_liquidity_gate_still_blocks_true_low_average_volume(self):
        from artha.config import Config
        from artha.council import hard_risk_gate

        stock_data = {
            "quote": {"price": 34.0, "marketCap": 8_000_000_000, "avgVolume": 3_582},
            "profile": {"sector": "Communication Services"},
            "ratios_ttm": {},
            "key_metrics_ttm": {"bookValuePerShareTTM": 4},
        }
        passed, reason = hard_risk_gate(
            "TEST",
            stock_data,
            portfolio_state={"positions": [], "total_value": 350, "cash_available": 350},
            sentinel_alerts=[],
            config=Config,
        )
        self.assertFalse(passed)
        self.assertIn("Avg daily volume", reason)


class TestValuationAndRisk(unittest.TestCase):
    def test_chatgpt_backend_sends_high_reasoning_and_temperature(self):
        from artha.chatgpt_backend import ChatGPTBackendClient

        with patch("artha.chatgpt_backend.requests.post") as post:
            post.return_value = Mock(status_code=200)
            client = ChatGPTBackendClient(
                auth_path="/tmp/unused-auth.json",
                reasoning_effort="xhigh",
                temperature=2.0,
            )
            response = client._send_request("test prompt", "gpt-5.5", "fake-token")

        self.assertEqual(response.status_code, 200)
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["reasoning"]["effort"], "xhigh")
        self.assertEqual(body["temperature"], 2.0)

    def test_chatgpt_backend_retries_without_temperature_when_codex_backend_rejects_it(self):
        from artha.chatgpt_backend import ChatGPTBackendClient

        rejected = Mock(status_code=400, text='{"detail":"Unsupported parameter: temperature"}')
        rejected.headers = {}
        accepted = Mock(status_code=200, text='{"output_text":"ok"}')
        accepted.headers = {"Content-Type": "application/json"}
        accepted.json.return_value = {"output_text": "ok"}

        with patch("artha.chatgpt_backend.requests.post", side_effect=[rejected, accepted]) as post, patch.object(
            ChatGPTBackendClient,
            "_get_valid_access_token",
            return_value="fake-token",
        ):
            text = ChatGPTBackendClient(
                auth_path="/tmp/unused-auth.json",
                reasoning_effort="xhigh",
                temperature=2.0,
            ).chat("test prompt")

        self.assertEqual(text, "ok")
        first_body = post.call_args_list[0].kwargs["json"]
        retry_body = post.call_args_list[1].kwargs["json"]
        self.assertEqual(first_body["temperature"], 2.0)
        self.assertEqual(first_body["reasoning"]["effort"], "xhigh")
        self.assertNotIn("temperature", retry_body)
        self.assertEqual(retry_body["reasoning"]["effort"], "xhigh")

    def test_chatgpt_backend_retries_transient_transport_errors(self):
        import requests

        from artha.chatgpt_backend import ChatGPTBackendClient

        accepted = Mock(status_code=200)
        with patch(
            "artha.chatgpt_backend.requests.post",
            side_effect=[requests.exceptions.SSLError("bad record mac"), accepted],
        ) as post, patch("artha.chatgpt_backend.time.sleep") as sleep:
            response = ChatGPTBackendClient(auth_path="/tmp/unused-auth.json")._send_request(
                "test prompt",
                "gpt-5.5",
                "fake-token",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_gemini_client_sends_high_thinking_for_gemini_3(self):
        from artha.gemini_client import gemini_generate

        fake_response = Mock(status_code=200)
        fake_response.json.return_value = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "ok"}]},
                    "groundingMetadata": {},
                }
            ]
        }

        with patch("artha.gemini_client.Config.GEMINI_API_KEY", "fake-key"), patch(
            "artha.gemini_client._SESSION.post",
            return_value=fake_response,
        ) as post:
            text, _ = gemini_generate(
                "test prompt",
                model="gemini-3.1-pro-preview",
                thinking_level="high",
                temperature=1.0,
            )

        self.assertEqual(text, "ok")
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["generationConfig"]["temperature"], 1.0)
        self.assertEqual(body["generationConfig"]["thinkingConfig"]["thinkingLevel"], "high")

    def test_gemini_client_omits_thinking_level_for_non_gemini_3(self):
        from artha.gemini_client import gemini_generate

        fake_response = Mock(status_code=200)
        fake_response.json.return_value = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "ok"}]},
                    "groundingMetadata": {},
                }
            ]
        }

        with patch("artha.gemini_client.Config.GEMINI_API_KEY", "fake-key"), patch(
            "artha.gemini_client._SESSION.post",
            return_value=fake_response,
        ) as post:
            gemini_generate("test prompt", model="gemini-2.5-flash")

        body = post.call_args.kwargs["json"]
        self.assertNotIn("thinkingConfig", body["generationConfig"])

    def test_valuation_expectations_positive_case(self):
        from artha.valuation import build_valuation_expectations, format_valuation_expectations

        result = build_valuation_expectations(_sample_stock())
        self.assertEqual(result["valuation_signal"], "positive")
        self.assertGreater(result["valuation_score"], 65)
        self.assertEqual(result["expectation_risk_level"], "low")
        rendered = format_valuation_expectations(result)
        self.assertIn("DETERMINISTIC VALUATION", rendered)
        self.assertIn("consensus", rendered)

    def test_expectation_risk_flags_low_upside_negative_revision(self):
        from artha.valuation import build_valuation_expectations

        stock = _sample_stock(price=100.0)
        stock["dcf"] = {"dcf": 55.0}
        stock["price_target_consensus"] = {"targetConsensus": 103.0, "targetLow": 75.0, "targetHigh": 130.0}
        stock["recommendation_trends"] = {"net_upgrades_30d": 0, "net_downgrades_30d": 1}
        result = build_valuation_expectations(stock)
        self.assertGreaterEqual(
            {"moderate": 1, "high": 2}.get(result["expectation_risk_level"], 0),
            1,
        )

    def test_portfolio_risk_flags_sector_cap(self):
        from artha.config import Config
        from artha.portfolio_risk import build_portfolio_factor_risk, sector_benchmark_for

        portfolio = {
            "total_value": 10_000,
            "cash_available": 1_000,
            "monthly_contribution": 500,
            "concentration_pct": 25,
            "positions": [
                {"ticker": "AAA", "sector": "Technology", "market_value": 3200, "weight_pct": 32.0},
            ],
        }
        risk = build_portfolio_factor_risk("TEST", _sample_stock(), portfolio, Config)
        self.assertEqual(sector_benchmark_for("Technology"), "XLK")
        self.assertEqual(risk["risk_level"], "high")
        self.assertTrue(any("sector" in f.lower() for f in risk["risk_flags"]))

    def test_equity_sentiment_is_not_crypto_fear_greed(self):
        from artha.collector import get_equity_sentiment_index

        sentiment = get_equity_sentiment_index(
            {
                "sp500": {"changesPercentage": 0.5},
                "nasdaq": {"changesPercentage": 0.4},
                "vix": {"value": 15.0},
            }
        )
        self.assertEqual(sentiment["asset_class"], "equity")
        self.assertEqual(sentiment["source"], "artha_equity_sentiment")
        self.assertGreater(sentiment["value"], 50)

    def test_massive_previous_bar_normalizes_market_data(self):
        from unittest.mock import patch

        from artha.collector import MassiveCollector

        collector = MassiveCollector()
        collector.enabled = True
        payload = {
            "status": "OK",
            "results": [
                {
                    "T": "AAPL",
                    "o": 199.0,
                    "h": 203.0,
                    "l": 198.5,
                    "c": 201.25,
                    "v": 12345678,
                    "t": 1780617600000,
                }
            ],
        }
        with patch.object(collector, "_get", return_value=payload):
            quote = collector.quote("AAPL")

        self.assertEqual(quote["symbol"], "AAPL")
        self.assertEqual(quote["price"], 201.25)
        self.assertEqual(quote["volume"], 12345678)
        self.assertEqual(quote["source"], "massive.previous_bar")

    def test_fmp_history_normalizes_stable_eod_rows(self):
        from unittest.mock import patch

        from artha.collector import FMPCollector

        collector = FMPCollector()
        payload = [
            {"symbol": "AAPL", "date": "2026-06-04", "open": 199, "high": 203, "low": 198, "close": 201.25, "volume": 123},
            {"symbol": "AAPL", "date": "2026-06-03", "open": 198, "high": 200, "low": 197, "close": 199.50, "volume": 456},
        ]
        with patch.object(collector, "_get", return_value=payload):
            history = collector.history("AAPL", "10d")

        self.assertEqual(len(history), 2)
        self.assertEqual(history[-1]["date"], "2026-06-04")
        self.assertEqual(history[-1]["close"], 201.25)
        self.assertEqual(history[-1]["source"], "fmp.historical_price_eod")

    def test_history_provider_checks_detect_close_conflict(self):
        from artha.collector import DataCollector

        checks = DataCollector._build_history_provider_checks(
            selected_source="fmp",
            fmp_history=[{"date": "2026-06-04", "close": 100.0}],
            yf_history=[{"date": "2026-06-04", "close": 103.5}],
            massive_history=None,
        )

        self.assertEqual(checks["selected_source"], "fmp")
        self.assertTrue(checks["conflicts"])

    def test_data_quality_uses_massive_as_market_cross_check(self):
        from artha.data_quality import validate_stock_data

        stock = _sample_stock(price=100.0)
        stock["massive_quote"] = {"price": 110.0, "volume": 2_000_000, "source": "massive.previous_bar"}
        report = validate_stock_data(stock)

        self.assertIn("massive", report.sources_used)
        self.assertTrue(any("Massive" in conflict for conflict in report.source_conflicts))

    def test_yfinance_cleanup_does_not_close_artha_sqlite_handles(self):
        import sqlite3

        from artha.collector import YFinanceCollector

        with tempfile.TemporaryDirectory() as tmp:
            conn = sqlite3.connect(Path(tmp) / "artha-not-yfinance.db")
            try:
                conn.execute("CREATE TABLE IF NOT EXISTS unit_check (id INTEGER)")
                YFinanceCollector.cleanup_caches()
                conn.execute("INSERT INTO unit_check (id) VALUES (1)")
                row = conn.execute("SELECT COUNT(*) FROM unit_check").fetchone()
                self.assertEqual(row[0], 1)
            finally:
                conn.close()

    def test_rank_universe_uses_non_threaded_yfinance_and_cleans_batches(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        import numpy as np
        import pandas as pd

        from artha.config import Config
        from artha.rank_candidates import rank_universe

        old_batch = Config.YFINANCE_BATCH_SIZE
        old_threads = Config.YFINANCE_THREADS
        calls = []

        def fake_download(tickers_str, **kwargs):
            calls.append({"tickers": tickers_str, **kwargs})
            symbol = tickers_str.split()[0]
            idx = pd.date_range("2025-01-01", periods=260, freq="B")
            cols = pd.MultiIndex.from_tuples([(symbol, "Close")])
            return pd.DataFrame(np.linspace(10, 20, len(idx)), index=idx, columns=cols)

        try:
            Config.YFINANCE_BATCH_SIZE = 1
            Config.YFINANCE_THREADS = False
            universe = [
                SimpleNamespace(symbol="AAA", name="AAA Inc", sector="Technology", industry="Software", market_cap=2_000_000_000, volume=500_000, regime_score=1.0),
                SimpleNamespace(symbol="BBB", name="BBB Inc", sector="Healthcare", industry="Tools", market_cap=2_000_000_000, volume=500_000, regime_score=1.0),
            ]
            with tempfile.TemporaryDirectory() as rank_tmp, \
                 patch("yfinance.download", side_effect=fake_download), \
                 patch("artha.rank_candidates.YFinanceCollector.cleanup_caches") as cleanup, \
                 patch("artha.rank_candidates.RANK_COVERAGE_DIR", Path(rank_tmp) / "coverage"), \
                 patch("artha.rank_candidates.SCAN_SIGNALS_DIR", Path(rank_tmp) / "signals"):
                ranked = rank_universe(universe, top_n=2)
            self.assertEqual(len(ranked), 2)
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(call.get("threads") is False for call in calls))
            self.assertEqual(cleanup.call_count, 2)
        finally:
            Config.YFINANCE_BATCH_SIZE = old_batch
            Config.YFINANCE_THREADS = old_threads

    def test_promotion_funnel_reads_base_regime_type_from_mrol_packet(self):
        from types import SimpleNamespace

        from artha.funnel import PromotionFunnel

        class FakeUniverseBuilder:
            def __init__(self):
                self.seen_regime = None
                self.seen_overlays = None

            def build_universe(self, regime_type=None, overlays=None, limit=1000):
                self.seen_regime = regime_type
                self.seen_overlays = overlays
                return []

        fake_builder = FakeUniverseBuilder()
        funnel = PromotionFunnel.__new__(PromotionFunnel)
        funnel.universe_builder = fake_builder
        packet = SimpleNamespace(base_regime_type="goldilocks", event_overlays=[{"type": "ai_tech_momentum"}])

        result = funnel.run(packet, max_council_candidates=1, fallback_on_failure=False)

        self.assertEqual(result, [])
        self.assertEqual(fake_builder.seen_regime, "goldilocks")
        self.assertEqual(fake_builder.seen_overlays, ["ai_tech_momentum"])

    def test_funnel_parallel_discovery_adds_non_momentum_probe(self):
        from types import SimpleNamespace

        from artha.config import Config
        from artha.funnel import PromotionFunnel

        ranked = [
            {
                "symbol": f"R{idx:03d}",
                "name": f"Ranked {idx}",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 10_000_000_000,
                "avg_volume": 2_000_000,
                "price": 100.0,
                "combined_score": 20.0,
                "return_12m": 20.0,
                "return_3m": 5.0,
                "vol_20d": 20.0,
            }
            for idx in range(8)
        ]
        universe = [
            SimpleNamespace(
                symbol=c["symbol"],
                name=c["name"],
                sector=c["sector"],
                industry=c["industry"],
                market_cap=c["market_cap"],
                price=c["price"],
                volume=c["avg_volume"],
                beta=1.0,
                regime_score=0.0,
            )
            for c in ranked
        ]
        universe.extend(
            SimpleNamespace(
                symbol=f"Q{idx:03d}",
                name=f"Quiet {idx}",
                sector="Industrials",
                industry="Machinery",
                market_cap=8_000_000_000,
                price=40.0,
                volume=1_500_000,
                beta=1.1,
                regime_score=6.0,
            )
            for idx in range(5)
        )

        old_enabled = Config.FUNNEL_PARALLEL_DISCOVERY_ENABLED
        try:
            Config.FUNNEL_PARALLEL_DISCOVERY_ENABLED = True
            pool = PromotionFunnel()._build_enrichment_pool(ranked, enrich_max=12, universe=universe)
        finally:
            Config.FUNNEL_PARALLEL_DISCOVERY_ENABLED = old_enabled

        parallel = [c for c in pool if str(c.get("enrichment_pool_reason", "")).startswith("parallel_")]
        self.assertTrue(parallel)
        self.assertTrue(all(c["symbol"].startswith("Q") for c in parallel))

    def test_all_council_roles_receive_provider_audit_fields(self):
        from artha.analysts import (
            _extract_relevant_fundamental_data,
            _extract_relevant_risk_data,
            _extract_relevant_technical_data,
        )

        stock = _sample_stock(price=100.0)
        stock.update(
            {
                "yf_quote": {"price": 100.1, "volume": 2_000_000},
                "massive_quote": {"price": 100.0, "volume": 2_100_000, "source": "massive.previous_bar"},
                "price_history_source": "fmp",
                "history_provider_checks": {
                    "selected_source": "fmp",
                    "providers": {
                        "fmp": {"bars": 252, "latest_close": 100.0},
                        "yfinance": {"bars": 251, "latest_close": 100.1},
                        "massive": {"bars": 0, "latest_close": None},
                    },
                    "conflicts": [],
                },
                "fmp_price_history": [{"date": "2026-06-04", "close": 100.0}],
                "yf_price_history": [{"date": "2026-06-04", "close": 100.1}],
                "massive_price_history": [{"date": "2026-06-04", "close": 100.0}],
            }
        )

        role_packets = {
            "fundamental": _extract_relevant_fundamental_data(stock),
            "technical": _extract_relevant_technical_data(stock),
            "risk": _extract_relevant_risk_data(stock),
        }

        for role, packet in role_packets.items():
            self.assertEqual(packet["massive_quote"]["source"], "massive.previous_bar", role)
            self.assertEqual(packet["price_history_source"], "fmp", role)
            self.assertEqual(packet["history_provider_checks"]["selected_source"], "fmp", role)
            self.assertTrue(packet["fmp_price_history_available"], role)
            self.assertTrue(packet["yf_price_history_available"], role)
            self.assertTrue(packet["massive_price_history_available"], role)

        self.assertIn("technicals", role_packets["technical"])
        self.assertIn("sec", role_packets["fundamental"])
        self.assertIn("short_interest", role_packets["risk"])

    def test_scoring_json_minor_numeric_error_is_repaired(self):
        from artha.council import _extract_scoring_json

        synthesis = """```json
{
  "opportunity_score": 43,
  "components": {
    "technical_setup": 21,
    "fundamental_quality": 3,
    "contrarian_sentiment": 5,
    "regime_alignment": 4,
    "catalyst_asymmetry": 4,
    "data_quality": 7,
    "liquidity_execution": -1
  },
  "verdict": "AVOID",
  "confidence": 8,
  "thesis_type": "momentum_breakout",
  "recommended_allocation_pct": 0.0,
  "entry_valid_until": "2026-08-14",
  "invalidation_conditions": ["unit"],
  "stop_loss_pct": 0.0,
  "target_pct": 0.0
}
```"""
        scoring = _extract_scoring_json(synthesis)
        self.assertIsNotNone(scoring)
        self.assertEqual(scoring["components"]["liquidity_execution"], 0)
        self.assertEqual(scoring["verdict"], "AVOID")

    def test_hybrid_scoring_preserves_cio_adjusted_opportunity_score(self):
        from artha.council import _extract_scoring_json

        synthesis = """```json
{
  "opportunity_score": 68,
  "components": {
    "technical_setup": 15,
    "fundamental_quality": 14,
    "contrarian_sentiment": 8,
    "regime_alignment": 8,
    "catalyst_asymmetry": 5,
    "data_quality": 7,
    "liquidity_execution": 3
  },
  "deterministic_base_score": 60,
  "rule_adjustment_total": 0,
  "deterministic_score_before_cio": 60,
  "cio_score_adjustment": 8,
  "cio_adjustment_category": "logic_backed",
  "cio_adjustment_evidence": ["Raw Valuation Anchors"],
  "cio_adjustment_reason": "The deterministic score underweights a credible stabilization setup with positive cash flow and improving estimates.",
  "verdict": "STARTER",
  "confidence": 7,
  "thesis_type": "mean_reversion",
  "recommended_allocation_pct": 5.0,
  "entry_valid_until": "2026-07-04",
  "invalidation_conditions": ["unit"],
  "stop_loss_pct": -0.08,
  "target_pct": 0.12
}
```"""
        scoring = _extract_scoring_json(synthesis)
        self.assertIsNotNone(scoring)
        self.assertEqual(scoring["opportunity_score"], 68)
        self.assertEqual(scoring["deterministic_base_score"], 60)
        self.assertEqual(scoring["cio_score_adjustment"], 8)

    def test_buy_hybrid_score_blocks_cio_overupgrade_from_weak_data(self):
        from artha.buy_scoring import apply_cio_buy_adjustment, build_buy_score_audit
        from artha.config import Config
        from artha.council import score_to_action
        from artha.valuation import build_valuation_expectations

        weak = {
            "ticker": "WEAK",
            "quote": {"price": 100.0, "marketCap": 2_000_000_000, "volume": 150_000},
            "profile": {"sector": "Technology", "beta": 3.2},
            "technicals": {
                "rsi": 84,
                "sma_20": 80,
                "sma_50": 60,
                "sma_200": 55,
                "macd_interpretation": "bullish",
                "volume_ratio": 1.1,
            },
            "dcf": {"dcf": 25.0},
            "price_target_consensus": {"targetConsensus": 70.0, "targetLow": 40.0, "targetHigh": 90.0},
            "ratios_ttm": {
                "priceToEarningsRatioTTM": 80,
                "priceToSalesRatioTTM": 35,
                "debtToEquityRatioTTM": 6,
                "netProfitMarginTTM": -0.2,
            },
            "key_metrics_ttm": {
                "freeCashFlowYieldTTM": -0.04,
                "returnOnInvestedCapitalTTM": -0.1,
                "returnOnEquityTTM": -0.2,
            },
            "income_statement": [{"revenue": 1000, "netIncome": -200}],
            "cash_flow": [{"freeCashFlow": -100}],
            "balance_sheet": [{"cashAndCashEquivalents": 100, "totalDebt": 1500}],
            "recommendation_trends": {"net_upgrades_30d": 0, "net_downgrades_30d": 2},
            "short_interest": {"short_interest_pct": 28},
            "earnings_context": {"earnings_risk_flag": True, "days_to_earnings": 4},
        }
        audit = build_buy_score_audit(
            stock_data=weak,
            data_quality_report={"completeness_score": 95, "context_coverage_score": 80, "source_conflicts": []},
            valuation_expectations=build_valuation_expectations(weak),
            portfolio_factor_risk={"risk_level": "high"},
            research_insufficient=False,
            fear_greed=25,
        )
        cio_high_score = 86
        old_action = score_to_action(cio_high_score, 25, {"total_value": 350}, Config)["action"]
        adjusted = apply_cio_buy_adjustment(
            {
                "cio_score_adjustment": 15,
                "cio_adjustment_category": "logic_backed",
                "cio_adjustment_reason": (
                    "The CIO sees a plausible turnaround narrative, but this is mostly "
                    "second-order reasoning and not enough to override weak current data."
                ),
                "confidence": 8,
            },
            audit,
            config=Config,
            cio_confidence=8,
        )
        new_action = score_to_action(adjusted["final_score"], 25, {"total_value": 350}, Config)["action"]

        self.assertEqual(old_action, "BUY")
        self.assertLess(adjusted["final_score"], Config.SCORE_THRESHOLD_TACTICAL)
        self.assertNotIn(new_action, {"BUY", "STARTER", "TACTICAL_BUY"})
        self.assertEqual(adjusted["cio_adjustment_status"], "accepted_clamped")

    def test_buy_hybrid_score_allows_modest_logic_backed_upgrade(self):
        from artha.buy_scoring import apply_cio_buy_adjustment, build_buy_score_audit
        from artha.config import Config
        from artha.council import score_to_action
        from artha.valuation import build_valuation_expectations

        setup = {
            "ticker": "NUANCE",
            "quote": {"price": 100.0, "marketCap": 5_000_000_000, "volume": 600_000},
            "profile": {"sector": "Communication Services", "beta": 1.4},
            "technicals": {
                "rsi": 39,
                "sma_20": 106,
                "sma_50": 110,
                "sma_200": 105,
                "macd_interpretation": "bearish",
                "volume_ratio": 0.6,
            },
            "dcf": {"dcf": 112.0},
            "price_target_consensus": {"targetConsensus": 115.0, "targetLow": 82.0, "targetHigh": 145.0},
            "ratios_ttm": {
                "priceToEarningsRatioTTM": 32,
                "priceToSalesRatioTTM": 5,
                "debtToEquityRatioTTM": 1.5,
                "netProfitMarginTTM": 0.08,
            },
            "key_metrics_ttm": {
                "freeCashFlowYieldTTM": 0.025,
                "returnOnInvestedCapitalTTM": 0.08,
                "returnOnEquityTTM": 0.12,
            },
            "income_statement": [{"revenue": 1000, "netIncome": 80}],
            "cash_flow": [{"freeCashFlow": 80}],
            "balance_sheet": [{"cashAndCashEquivalents": 500, "totalDebt": 900}],
            "recommendation_trends": {"net_upgrades_30d": 1, "net_downgrades_30d": 0},
            "short_interest": {"short_interest_pct": 6},
            "earnings_context": {"earnings_risk_flag": False, "days_to_earnings": 22},
            "earnings_surprises": [{"surprisePercent": 3}],
            "analyst_estimates": {"fy1_revenue_estimate": 4300},
        }
        audit = build_buy_score_audit(
            stock_data=setup,
            data_quality_report={"completeness_score": 95, "context_coverage_score": 70, "source_conflicts": []},
            valuation_expectations=build_valuation_expectations(setup),
            portfolio_factor_risk={"risk_level": "low"},
            research_insufficient=False,
            fear_greed=25,
        )
        before_action = score_to_action(audit["pre_cio_score"], 25, {"total_value": 350}, Config)["action"]
        adjusted = apply_cio_buy_adjustment(
            {
                "cio_score_adjustment": 8,
                "cio_adjustment_category": "logic_backed",
                "cio_adjustment_reason": (
                    "The deterministic score underweights a plausible stabilization setup: "
                    "price is cooling near trend support while free cash flow remains positive "
                    "and analyst revisions are improving."
                ),
                "confidence": 7,
            },
            audit,
            config=Config,
            cio_confidence=7,
        )
        after_action = score_to_action(adjusted["final_score"], 25, {"total_value": 350}, Config)["action"]

        self.assertEqual(before_action, "TACTICAL_BUY")
        self.assertEqual(adjusted["cio_adjustment"], 8)
        self.assertEqual(adjusted["cio_adjustment_status"], "accepted")
        self.assertEqual(after_action, "STARTER")

    def test_no_buy_action_normalizes_good_business_defer(self):
        from artha.council import _normalize_no_buy_action

        normalized = _normalize_no_buy_action(
            final_action="AVOID",
            cio_verdict="DEFER",
            recommended_action="**DEFER — do not open today.** Re-evaluate near support.",
            score_components={"fundamental_quality": 16},
        )
        self.assertEqual(normalized, "DEFER")

    def test_no_buy_action_normalizes_monitorable_watch(self):
        from artha.council import _normalize_no_buy_action

        normalized = _normalize_no_buy_action(
            final_action="AVOID",
            cio_verdict="WATCH",
            recommended_action="**WATCH — do not open today.** Re-evaluate after a clean quarter.",
            score_components={"fundamental_quality": 7},
        )
        self.assertEqual(normalized, "WATCH")

    def test_final_action_synchronizes_visible_scoring_json(self):
        from artha.council import _synchronize_final_action_surfaces

        synthesis = """**COUNCIL CONSENSUS:** 2-1 cautious

**RECOMMENDED ACTION:** **WATCH — do not open today.** Re-evaluate after proof improves.

```json
{
  "opportunity_score": 41,
  "components": {
    "technical_setup": 7,
    "fundamental_quality": 7,
    "contrarian_sentiment": 5,
    "regime_alignment": 5,
    "catalyst_asymmetry": 4,
    "data_quality": 7,
    "liquidity_execution": 5
  },
  "verdict": "AVOID",
  "confidence": 7,
  "thesis_type": "turnaround",
  "recommended_allocation_pct": 0.0,
  "entry_valid_until": "2026-07-04",
  "invalidation_conditions": ["unit"],
  "stop_loss_pct": 0.0,
  "target_pct": 0.0
}
```"""
        scoring = {
            "opportunity_score": 41,
            "components": {
                "technical_setup": 7,
                "fundamental_quality": 7,
                "contrarian_sentiment": 5,
                "regime_alignment": 5,
                "catalyst_asymmetry": 4,
                "data_quality": 7,
                "liquidity_execution": 5,
            },
            "verdict": "AVOID",
            "confidence": 7,
            "thesis_type": "turnaround",
            "recommended_allocation_pct": 0.0,
            "entry_valid_until": "2026-07-04",
            "invalidation_conditions": ["unit"],
            "stop_loss_pct": 0.0,
            "target_pct": 0.0,
        }
        updated, action, updated_scoring = _synchronize_final_action_surfaces(
            synthesis=synthesis,
            scoring=scoring,
            final_action="WATCH",
            recommended_action="**WATCH — do not open today.** Re-evaluate after proof improves.",
            final_alloc_pct=0.0,
            no_new_capital=True,
        )
        self.assertTrue(action.startswith("**WATCH"))
        self.assertEqual(updated_scoring["verdict"], "WATCH")
        self.assertIn('"verdict": "WATCH"', updated)
        self.assertNotIn('"verdict": "AVOID"', updated)
        self.assertIn("ACTION NORMALIZATION NOTE", updated)

    def test_defer_watch_scan_skip_far_from_pullback_zone(self):
        from artha.defer_watchlist import scan_skip_for_defer_watch

        watch = {
            "watch_id": "unit-watch",
            "ticker": "TEST",
            "trigger_type": "pullback",
            "zone_low": 95.0,
            "zone_high": 100.0,
        }
        result = scan_skip_for_defer_watch(watch, 122.0, buffer_pct=5.0)
        self.assertTrue(result["skip"])
        self.assertEqual(result["reason"], "active_pullback_watch_above_zone")
        self.assertGreater(result["distance_pct"], 20)

    def test_defer_watch_scan_does_not_skip_near_zone(self):
        from artha.defer_watchlist import scan_skip_for_defer_watch

        watch = {
            "watch_id": "unit-watch",
            "ticker": "TEST",
            "trigger_type": "pullback",
            "zone_low": 95.0,
            "zone_high": 100.0,
        }
        result = scan_skip_for_defer_watch(watch, 104.0, buffer_pct=5.0)
        self.assertFalse(result["skip"])
        self.assertEqual(result["reason"], "near_or_inside_watch_zone")

    def test_defer_watch_scan_major_move_overrides_skip(self):
        from artha.defer_watchlist import scan_skip_for_defer_watch

        watch = {
            "watch_id": "unit-watch",
            "ticker": "TEST",
            "trigger_type": "pullback",
            "zone_low": 95.0,
            "zone_high": 100.0,
        }
        result = scan_skip_for_defer_watch(
            watch,
            122.0,
            candidate={"changesPercentage": 7.5},
            buffer_pct=5.0,
            major_move_pct=5.0,
        )
        self.assertFalse(result["skip"])
        self.assertEqual(result["reason"], "major_move_override")

    def test_entry_quality_sleeve_scores_buyable_setup(self):
        from artha.funnel import PromotionFunnel

        candidate = {
            "symbol": "TEST",
            "price": 100.0,
            "return_12m": 62.0,
            "return_3m": 18.0,
            "vol_20d": 24.0,
            "beta": 1.1,
            "momentum_score": 35.0,
            "combined_score": 42.0,
            "ratios_ttm": {
                "priceEarningsRatioTTM": 22.0,
                "priceToSalesRatioTTM": 4.0,
                "priceToBookRatioTTM": 5.0,
                "returnOnEquityTTM": 0.2,
                "grossProfitMarginTTM": 0.55,
            },
            "key_metrics_ttm": {
                "freeCashFlowPerShareTTM": 3.2,
                "freeCashFlowYieldTTM": 0.05,
                "returnOnInvestedCapitalTTM": 0.17,
            },
            "recommendation_trends": {
                "consensus": "buy",
                "net_upgrades_30d": 1,
                "net_downgrades_30d": 0,
                "recommendation_mix": {"buy": 8, "strong_buy": 2, "sell": 0, "strong_sell": 0},
            },
            "analyst_estimates": {"next_q_eps_estimate": 1.0, "next_q_revenue_estimate": 100, "fy1_revenue_estimate": 500},
            "price_target_consensus": {"targetConsensus": 125.0},
            "dcf": {"dcf": 118.0},
        }
        scores = PromotionFunnel()._alpha_sleeve_scores(candidate)
        self.assertGreaterEqual(scores["entry_quality"], 16)


class TestJournalAndCalibration(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_shadow_schema_and_excess_returns(self):
        from artha.journal import DecisionJournal

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        row_id = journal.log_shadow_trade(
            ticker="TEST",
            thesis_type="unit",
            blocked_by="test",
            blocked_reason="unit test",
            hypothetical_entry=100.0,
            hypothetical_stop=90.0,
            opportunity_score=66.0,
            regime="NEUTRAL",
            fear_greed=50,
            sector="Technology",
            benchmark_ticker="QQQ",
            sector_benchmark_ticker="XLK",
        )
        journal.update_shadow_returns(
            row_id,
            price_5d=110.0,
            benchmark_price_entry=100.0,
            benchmark_price_5d=104.0,
            sector_benchmark_price_entry=100.0,
            sector_benchmark_price_5d=102.0,
        )
        with journal._connect() as conn:
            row = conn.execute("SELECT * FROM shadow_positions WHERE id = ?", (row_id,)).fetchone()
        self.assertAlmostEqual(row["return_5d"], 0.10, places=4)
        self.assertAlmostEqual(row["sector_benchmark_return_5d"], 0.02, places=4)
        self.assertAlmostEqual(row["excess_return_5d"], 0.08, places=4)

    def test_post_sell_tracking_allows_partial_checkpoint_update(self):
        from artha.journal import DecisionJournal

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        journal.save_post_sell_tracking(
            {
                "tracking_id": "unit-track",
                "ticker": "MSFT",
                "thesis_id": "thesis-1",
                "sell_date": "2026-06-02",
                "sell_price": 410.0,
                "sell_reason": "unit sell",
                "position_type": "TACTICAL_BUY",
                "shares": 1.0,
            }
        )

        journal.save_post_sell_tracking(
            {
                "tracking_id": "unit-track",
                "price_5d": 420.0,
                "return_5d": 0.0244,
                "regret_score": 0.0244,
            }
        )

        rows = journal.get_pending_post_sell_reviews()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ticker"], "MSFT")
        self.assertEqual(rows[0]["price_5d"], 420.0)
        self.assertEqual(rows[0]["return_5d"], 0.0244)

    def test_decision_feature_schema_accepts_new_columns(self):
        from artha.journal import DecisionJournal

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        journal.save_decision_features(
            {
                "dossier_path": "unit.json",
                "generated_at": "2026-06-04T00:00:00+00:00",
                "ticker": "TEST",
                "final_verdict": "DEFER",
                "opportunity_score": 50,
                "valuation_signal": "negative",
                "consensus_upside_pct": -10.0,
                "expectation_risk_level": "high",
                "portfolio_risk_level": "low",
                "portfolio_sector_after_pct": 5.0,
                "benchmark_ticker": "XLK",
                "feature_json": json.dumps({"ok": True}),
            }
        )
        rows = journal.get_decision_features(limit=1)
        self.assertEqual(rows[0]["valuation_signal"], "negative")
        self.assertEqual(rows[0]["benchmark_ticker"], "XLK")

    def test_calibration_report_includes_excess_returns(self):
        from artha.calibration import build_calibration_report
        from artha.journal import DecisionJournal

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        row_id = journal.log_shadow_trade(
            "TEST", "unit", "test", "unit", 100.0, 90.0, 66.0, "NEUTRAL", 50,
            sector="Technology", benchmark_ticker="QQQ", sector_benchmark_ticker="XLK",
        )
        journal.update_shadow_returns(
            row_id,
            price_5d=110,
            benchmark_price_entry=100,
            benchmark_price_5d=105,
            sector_benchmark_price_entry=100,
            sector_benchmark_price_5d=101,
        )
        report = build_calibration_report(journal)
        bucket = report["shadow_score_buckets"]["65-74"]
        self.assertIn("avg_excess_return_5d", bucket)
        self.assertAlmostEqual(bucket["avg_excess_return_5d"], 0.09, places=4)


class TestShadowRulesAndSupervisor(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_shadow_rule_logs_private_counterfactual_only(self):
        from types import SimpleNamespace

        from artha.journal import DecisionJournal
        from artha.shadow_rules import evaluate_shadow_rules_for_decision

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        decision = SimpleNamespace(
            ticker="TEST",
            final_verdict="DEFER",
            opportunity_score=58,
            dossier_path=str(self.tmp_dir / "TEST.json"),
            generated_at="2026-06-04T00:00:00+00:00",
        )
        stock_data = {
            "ticker": "TEST",
            "quote": {"price": 100.0},
            "valuation_expectations": {
                "expectation_risk_level": "low",
                "analyst_targets": {"consensus_upside_pct": 16.0},
                "revision_trend": {"net_revision_30d": 1},
            },
            "portfolio_factor_risk": {
                "risk_level": "low",
                "sector": "Technology",
                "market_benchmark_ticker": "QQQ",
                "sector_benchmark_ticker": "XLK",
            },
        }

        inserted = evaluate_shadow_rules_for_decision(decision, stock_data, journal)
        duplicate = evaluate_shadow_rules_for_decision(decision, stock_data, journal)
        rows = journal.get_shadow_rule_evaluations(limit=10)

        self.assertEqual(len(inserted), 1)
        self.assertEqual(len(duplicate), 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["real_action"], "DEFER")
        self.assertEqual(rows[0]["shadow_action"], "STARTER")
        self.assertEqual(rows[0]["rule_status"], "shadow_mode")
        self.assertEqual(decision.final_verdict, "DEFER")

    def test_shadow_rule_outcome_math_uses_sector_benchmark(self):
        from artha.journal import DecisionJournal

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        evaluation = {
            "evaluation_id": "unit-eval-1",
            "rule_id": "unit_rule",
            "rule_version": "v1",
            "ticker": "TEST",
            "dossier_path": "unit.json",
            "decision_generated_at": "2026-06-04T00:00:00+00:00",
            "real_action": "DEFER",
            "shadow_action": "STARTER",
            "rule_status": "shadow_mode",
            "trigger_reason": "unit",
            "hypothetical_entry": 100.0,
            "benchmark_ticker": "QQQ",
            "sector_benchmark_ticker": "XLK",
            "status": "tracking",
        }
        self.assertTrue(journal.save_shadow_rule_evaluation(evaluation))
        journal.update_shadow_rule_evaluation(
            "unit-eval-1",
            {
                "price_10d": 112.0,
                "benchmark_price_entry": 100.0,
                "benchmark_price_10d": 104.0,
                "sector_benchmark_price_entry": 100.0,
                "sector_benchmark_price_10d": 106.0,
            },
        )
        row = journal.get_shadow_rule_evaluations(limit=1)[0]
        self.assertAlmostEqual(row["return_10d"], 0.12, places=4)
        self.assertAlmostEqual(row["sector_benchmark_return_10d"], 0.06, places=4)
        self.assertAlmostEqual(row["excess_return_10d"], 0.06, places=4)

    def test_shadow_rule_live_and_backfill_share_one_evaluation_id(self):
        from types import SimpleNamespace

        from artha.journal import DecisionJournal
        from artha.shadow_rules import backfill_shadow_rules_from_features, evaluate_shadow_rules_for_decision

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        dossier_path = self.tmp_dir / "TEST.json"
        dossier_path.write_text(
            json.dumps({"ticker": "TEST", "generated_at": "2026-06-04T00:00:00+00:00"}),
            encoding="utf-8",
        )
        decision = SimpleNamespace(
            ticker="TEST",
            final_verdict="DEFER",
            opportunity_score=58,
            dossier_path=str(dossier_path),
        )
        stock_data = {
            "ticker": "TEST",
            "quote": {"price": 100.0},
            "valuation_expectations": {
                "expectation_risk_level": "low",
                "analyst_targets": {"consensus_upside_pct": 16.0},
                "revision_trend": {"net_revision_30d": 1},
            },
            "portfolio_factor_risk": {"risk_level": "low", "sector": "Technology"},
        }
        live_rows = evaluate_shadow_rules_for_decision(decision, stock_data, journal)
        journal.save_decision_features(
            {
                "dossier_path": str(dossier_path),
                "generated_at": "2026-06-04T00:00:00+00:00",
                "ticker": "TEST",
                "final_verdict": "DEFER",
                "opportunity_score": 58,
                "adjusted_score": 58,
                "price": 100.0,
                "sector": "Technology",
                "consensus_upside_pct": 16.0,
                "expectation_risk_level": "low",
                "portfolio_risk_level": "low",
                "feature_json": json.dumps(
                    {
                        "valuation_expectations": {
                            "expectation_risk_level": "low",
                            "revision_trend": {"net_revision_30d": 1},
                        },
                        "portfolio_factor_risk": {"risk_level": "low", "sector": "Technology"},
                    }
                ),
            }
        )
        backfilled = backfill_shadow_rules_from_features(journal)
        rows = journal.get_shadow_rule_evaluations(limit=10)

        self.assertEqual(len(live_rows), 1)
        self.assertEqual(backfilled, 0)
        self.assertEqual(len(rows), 1)

    def test_supervisor_persists_and_sends_plain_english_report(self):
        import artha.supervisor as supervisor
        from artha.journal import DecisionJournal

        class FakeSender:
            enabled = True

            def __init__(self):
                self.messages = []

            def send_message(self, text, parse_mode=None, silent=False):
                self.messages.append((text, parse_mode, silent))
                return True

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        dossier_path = self.tmp_dir / "TEST.json"
        trace_path = self.tmp_dir / "TEST_trace.json"
        trace_path.write_text(
            json.dumps(
                {
                    "enabled": True,
                    "evidence": [{"evidence_id": f"E{i:03d}"} for i in range(1, 13)],
                    "role_plans": {"fundamental": [], "technical": [], "contrarian": []},
                    "role_queries": {"fundamental": [], "technical": [], "contrarian": []},
                    "gaps": [],
                    "conflicts": [],
                }
            ),
            encoding="utf-8",
        )
        dossier_path.write_text(
            json.dumps(
                {
                    "ticker": "TEST",
                    "generated_at": "2026-06-04T00:00:00+00:00",
                    "source_audit": {"evidence_count": 12, "source_counts": {"fmp": 4, "sec": 2}},
                    "agentic_trace": {"enabled": True, "trace_path": str(trace_path), "gaps": [], "conflicts": []},
                    "analysts": {
                        "fundamental": {"model": "gpt-5.5", "report": "Claim [E001] [E002]", "verdict": "HOLD"},
                        "technical": {"model": "gemini", "report": "Claim [E003] [E004]", "verdict": "HOLD"},
                        "contrarian": {"model": "gpt-5.5", "report": "Claim [E005] [E006]", "verdict": "HOLD"},
                    },
                    "decision": {
                        "final_verdict": "DEFER",
                        "recommended_action": "Wait because evidence says so [E001] [E002] [E003].",
                    },
                }
            ),
            encoding="utf-8",
        )
        journal.save_decision_features(
            {
                "dossier_path": str(dossier_path),
                "generated_at": "2026-06-04T00:00:00+00:00",
                "ticker": "TEST",
                "final_verdict": "DEFER",
                "opportunity_score": 58,
                "adjusted_score": 58,
                "confidence": 7,
                "price": 100.0,
                "sector": "Technology",
                "evidence_count": 12,
                "source_count": 2,
                "valuation_signal": "neutral",
                "consensus_upside_pct": 16.0,
                "expectation_risk_level": "low",
                "portfolio_risk_level": "low",
                "benchmark_ticker": "XLK",
                "feature_json": json.dumps(
                    {
                        "valuation_expectations": {
                            "expectation_risk_level": "low",
                            "analyst_targets": {"consensus_upside_pct": 16.0},
                            "revision_trend": {"net_revision_30d": 1},
                        },
                        "portfolio_factor_risk": {
                            "risk_level": "low",
                            "sector": "Technology",
                            "sector_benchmark_ticker": "XLK",
                        },
                    }
                ),
            }
        )
        journal.save_session(
            session_type="unit",
            tickers_analyzed="TEST",
            report_path=str(dossier_path),
            timestamp="2026-06-04T00:00:00+00:00",
        )

        original_dir = supervisor.SUPERVISOR_DIR
        original_backfill = supervisor.backfill_decision_features
        original_recent_logs = supervisor._check_recent_logs
        supervisor.SUPERVISOR_DIR = self.tmp_dir / "supervisor"
        supervisor.backfill_decision_features = lambda journal: 0
        supervisor._check_recent_logs = lambda: {
            "name": "recent_logs",
            "status": "PASS",
            "message": "Unit log check clean.",
        }
        try:
            sender = FakeSender()
            report = supervisor.run_supervisor_check(
                journal=journal,
                send_telegram=True,
                force_telegram=True,
                sender=sender,
                run_diagnosis=False,
                diagnosis={"stage": "learning_only", "completed_samples": 0},
            )
        finally:
            supervisor.SUPERVISOR_DIR = original_dir
            supervisor.backfill_decision_features = original_backfill
            supervisor._check_recent_logs = original_recent_logs

        self.assertTrue(report["sent_to_telegram"])
        self.assertEqual(len(sender.messages), 1)
        self.assertIn("ARTHA SUPERVISOR CHECK", sender.messages[0][0])
        self.assertIn("cannot change live investing rules", sender.messages[0][0])
        self.assertTrue((self.tmp_dir / "supervisor" / "latest.txt").exists())
        self.assertGreaterEqual(len(journal.get_shadow_rule_evaluations(limit=10)), 1)
        check_names = {check["name"] for check in report["payload"]["checks"]}
        self.assertIn("agentic_trace", check_names)
        self.assertIn("intelligence_routing", check_names)
        self.assertIn("recent_logs", check_names)
        self.assertIn("execution_readiness", check_names)

    def test_supervisor_defer_watch_check_reports_auto_review_plumbing(self):
        from artha.journal import DecisionJournal
        from artha.supervisor import _check_defer_watches

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        journal.save_defer_watch(
            {
                "watch_id": "watch-failed",
                "ticker": "TEST",
                "status": "review_failed",
                "source_action": "DEFER until $9-$11",
                "current_price": 14.0,
                "zone_low": 9.0,
                "zone_high": 11.0,
                "trigger_type": "zone",
                "trigger_text": "Revisit near $9-$11",
                "invalidation_conditions": "[]",
                "opportunity_score": 52,
                "confidence": 7,
                "entry_valid_until": "2026-07-04T00:00:00+00:00",
                "dossier_path": str(self.tmp_dir / "old_dossier.json"),
                "trace_path": str(self.tmp_dir / "old_trace.json"),
                "notes": "unit failure",
            }
        )
        result = _check_defer_watches(journal)
        self.assertEqual(result["status"], "WARN")
        self.assertIn("Auto-review enabled", result["message"])
        self.assertEqual(result["recent_status_counts"]["review_failed"], 1)
        self.assertTrue(result["recent_failures"])
        self.assertTrue(result["auto_review"]["enabled"])

    def test_supervisor_recent_log_check_flags_quality_issues(self):
        from artha.supervisor import _check_recent_logs

        log_dir = self.tmp_dir / "logs"
        log_dir.mkdir()
        (log_dir / "artha-monitor.err").write_text(
            "\n".join(
                [
                    "12:00:00 [artha.researcher] WARNING: Failed to fetch article https://x: 403 Client Error",
                    "12:00:01 [artha.rank_candidates] WARNING: ranking budget exhausted after 120s",
                    "12:00:02 [artha.analyst_signals] WARNING: Short interest unavailable for TEST in scan-safe mode",
                    "12:01:00 [artha.council] WARNING: Scoring JSON failed schema validation — rejecting block",
                    "12:02:00 [artha.collector] ERROR: [finnhub] Unexpected error: reset",
                    "12:03:00 [artha.scheduler] INFO: [health] New alerts=0 (critical=0, warning=0)",
                ]
            ),
            encoding="utf-8",
        )

        result = _check_recent_logs(log_dir=log_dir, lookback_hours=48)
        self.assertEqual(result["status"], "WARN")
        self.assertEqual(result["quality_issue_count"], 1)
        self.assertEqual(result["error_count"], 1)
        self.assertEqual(result["transient_warning_count"], 3)

    def test_supervisor_recent_log_check_ignores_dashboard_client_disconnects(self):
        from artha.supervisor import _check_recent_logs

        log_dir = self.tmp_dir / "logs"
        log_dir.mkdir()
        (log_dir / "dashboard.err.log").write_text(
            "\n".join(
                [
                    "----------------------------------------",
                    "Exception occurred during processing of request from ('192.168.1.254', 46656)",
                    "Traceback (most recent call last):",
                    '  File "/opt/homebrew/lib/python3.14/socketserver.py", line 697, in process_request_thread',
                    "    self.finish_request(request, client_address)",
                    '  File "/opt/homebrew/lib/python3.14/http/server.py", line 464, in handle_one_request',
                    "    self.raw_requestline = self.rfile.readline(65537)",
                    "ConnectionResetError: [Errno 54] Connection reset by peer",
                    "----------------------------------------",
                ]
            ),
            encoding="utf-8",
        )

        result = _check_recent_logs(log_dir=log_dir, lookback_hours=48)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["fatal_count"], 0)
        self.assertEqual(result["error_count"], 0)
        self.assertEqual(result["transient_warning_count"], 1)

    def test_catalyst_lane_backfills_missing_average_volume_for_top_mover(self):
        from artha import scanner

        class FakeFMP:
            def market_gainers(self, limit=30):
                return [
                    {
                        "symbol": "ACHR",
                        "name": "Archer Aviation",
                        "price": 5.31,
                        "changesPercentage": 19.59,
                    }
                ]

            def quote(self, symbol):
                return {
                    "symbol": symbol,
                    "name": "Archer Aviation",
                    "price": 5.31,
                    "changePercentage": 19.59,
                    "volume": 95_000_000,
                    "avgVolume": None,
                    "marketCap": 4_000_000_000,
                }

            def history(self, symbol, period="3mo"):
                return [{"volume": 37_000_000} for _ in range(45)]

            def company_profile(self, symbol):
                return {"isEtf": False, "isFund": False}

            def stock_news(self, symbol, limit=3):
                return [
                    {
                        "symbol": "AAPL",
                        "publishedDate": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                        "title": "Apple unrelated news",
                    }
                ]

        earnings_context = Mock(days_to_earnings=30)
        with patch("artha.scanner.FMPCollector", return_value=FakeFMP()), \
             patch("artha.scanner._get_news_catalysts", return_value=[]), \
             patch("artha.earnings_calendar.get_earnings_context", return_value=earnings_context):
            candidates = scanner.get_catalyst_candidates(max_candidates=3)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["symbol"], "ACHR")
        self.assertEqual(candidates[0]["avg_volume_source"], "history")
        self.assertEqual(candidates[0]["catalyst_type"], "top_mover_volume")
        self.assertGreater(candidates[0]["volume_ratio"], 2.0)

    def test_fmp_stock_news_filters_unrelated_symbol_rows(self):
        from artha.collector import FMPCollector

        fmp = FMPCollector()
        rows = [
            {"symbol": "AAPL", "title": "Apple unrelated news"},
            {"symbol": "NASDAQ:ACHR", "title": "Archer relevant news"},
            {"symbols": "TSLA, ACHR", "title": "Multi-symbol relevant news"},
        ]
        with patch.object(fmp, "_get", return_value=rows):
            news = fmp.stock_news("ACHR")

        self.assertEqual([row["title"] for row in news], [
            "Archer relevant news",
            "Multi-symbol relevant news",
        ])

    def test_missed_winner_audit_marks_low_price_policy_exclusion_non_actionable(self):
        from artha.self_review import NightlyReview

        class FakeAccuracy:
            def _lock(self, exclusive=False):
                return object()

            def _unlock(self, lock):
                return None

            def _load(self):
                return []

        class FakeFMP:
            def market_gainers(self, limit=50):
                return [
                    {
                        "symbol": "AMC",
                        "name": "AMC Entertainment",
                        "price": 2.46,
                        "changesPercentage": 26.8,
                    }
                ]

            def quote(self, symbol):
                return {
                    "symbol": symbol,
                    "price": 2.46,
                    "changePercentage": 26.8,
                    "marketCap": 1_500_000_000,
                }

        class FakeCursor:
            def fetchone(self):
                return None

        class FakeConnection:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def execute(self, *args, **kwargs):
                return FakeCursor()

        class FakeJournal:
            def _connect(self):
                return FakeConnection()

        review = NightlyReview()
        review.accuracy = FakeAccuracy()
        today = datetime(2026, 7, 20, tzinfo=timezone.utc).date()
        with patch("artha.collector.FMPCollector", return_value=FakeFMP()), \
             patch("artha.journal.DecisionJournal", return_value=FakeJournal()), \
             patch("artha.self_review.record_lesson"):
            learnings = review._audit_missed_winners(today)

        self.assertEqual(len(learnings), 1)
        self.assertEqual(learnings[0]["ticker"], "AMC")
        self.assertEqual(learnings[0]["artha_disposition"], "policy_excluded_low_price")
        self.assertFalse(learnings[0]["actionable_miss"])

        improvement = review._identify_improvement({"missed_winners": learnings})
        self.assertNotEqual(improvement["type"], "missed_winner_audit")


class TestExecutionReadiness(unittest.TestCase):
    def setUp(self):
        from artha.config import Config
        from artha import portfolio_state

        self.tmp_dir = Path(tempfile.mkdtemp())
        self._portfolio_state_path = portfolio_state.PORTFOLIO_JSON_PATH
        portfolio_state.PORTFOLIO_JSON_PATH = self.tmp_dir / "portfolio.json"
        portfolio_state.PORTFOLIO_JSON_PATH.write_text(
            json.dumps(
                {
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "cash_available": 350.0,
                    "monthly_contribution": 350.0,
                    "positions": [],
                    "transactions": [],
                    "notes": "Isolated execution-hardening test portfolio.",
                }
            ),
            encoding="utf-8",
        )
        self._stand_down_config = (
            Config.ARTHA_AUTO_PAUSE_BUYS_ON_LIMIT,
            Config.ROBINHOOD_CONTROL_FILE,
        )
        Config.ARTHA_AUTO_PAUSE_BUYS_ON_LIMIT = False
        Config.ROBINHOOD_CONTROL_FILE = str(self.tmp_dir / "control.json")

    def tearDown(self):
        import shutil
        from artha import portfolio_state
        from artha.config import Config

        (
            Config.ARTHA_AUTO_PAUSE_BUYS_ON_LIMIT,
            Config.ROBINHOOD_CONTROL_FILE,
        ) = self._stand_down_config
        portfolio_state.PORTFOLIO_JSON_PATH = self._portfolio_state_path
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _journal_with_clean_buy_decision(self):
        from artha.journal import DecisionJournal
        from artha.thesis_tracker import ThesisTracker

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        journal.save_supervisor_run(
            {
                "generated_at": "2026-06-04T14:30:00+00:00",
                "severity": "PASS",
                "report_hash": "unit",
                "report_text": "unit",
                "payload": {"checks": []},
                "sent_to_telegram": False,
            }
        )
        journal.save_decision_features(
            {
                "dossier_path": str(self.tmp_dir / "TEST_dossier.json"),
                "generated_at": "2026-06-04T14:20:00+00:00",
                "ticker": "TEST",
                "final_verdict": "STARTER",
                "opportunity_score": 68,
                "adjusted_score": 68,
                "confidence": 8,
                "price": 100.0,
                "sector": "Technology",
                "evidence_count": 12,
                "source_count": 4,
                "feature_json": json.dumps({"unit": True}),
            }
        )
        ThesisTracker(journal).create_thesis(
            "TEST",
            "STARTER",
            thesis_summary="Current unit-test Council thesis.",
            invalidation_conditions=[{"metric": "price", "op": "lte", "value": 90.0}],
        )
        return journal

    def _market_data(self):
        return {
            "price": 100.0,
            "volume": 250_000,
            "bid": 99.95,
            "ask": 100.05,
        }

    def _agentic_snapshot(
        self,
        *,
        cash: float = 100.0,
        positions: list[dict] | None = None,
        orders: list[dict] | None = None,
    ) -> dict:
        from artha.execution import normalize_robinhood_position_snapshot

        position_rows = positions or []
        invested = 0.0
        for row in position_rows:
            quantity = float(row.get("quantity") or row.get("shares") or 0.0)
            market_value = row.get("market_value") or row.get("equity") or row.get("value")
            if market_value is not None:
                invested += float(market_value)
            else:
                price = float(
                    row.get("market_price")
                    or row.get("current_price")
                    or row.get("price")
                    or row.get("average_buy_price")
                    or 0.0
                )
                invested += quantity * price
        return normalize_robinhood_position_snapshot(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": "robinhood_mcp",
                "account": {
                    "account_number": "222233334",
                    "type": "cash",
                    "nickname": "Agentic",
                    "agentic_allowed": True,
                    "state": "active",
                },
                "portfolio": {
                    "buying_power": cash,
                    "cash": cash,
                    "total_value": cash + invested,
                    "equity_value": invested,
                },
                "positions": position_rows,
                "orders": orders or [],
            }
        )

    def _write_agentic_snapshot_file(
        self,
        *,
        cash: float = 100.0,
        positions: list[dict] | None = None,
        orders: list[dict] | None = None,
    ) -> Path:
        path = self.tmp_dir / "robinhood_snapshot.json"
        path.write_text(
            json.dumps(self._agentic_snapshot(cash=cash, positions=positions, orders=orders)),
            encoding="utf-8",
        )
        return path

    def _save_defer_watch(self, journal, watch_id: str = "watch-test") -> None:
        valid_until = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        journal.save_defer_watch(
            {
                "watch_id": watch_id,
                "ticker": "TEST",
                "status": "active",
                "source_action": "DEFER until $9-$11",
                "current_price": 14.0,
                "zone_low": 9.0,
                "zone_high": 11.0,
                "trigger_type": "zone",
                "trigger_text": "Revisit near $9-$11",
                "invalidation_conditions": "[]",
                "opportunity_score": 52,
                "confidence": 7,
                "entry_valid_until": valid_until,
                "dossier_path": str(self.tmp_dir / "old_dossier.json"),
                "trace_path": str(self.tmp_dir / "old_trace.json"),
                "notes": "unit watch",
            }
        )

    def test_defer_watch_status_update_appends_audit_notes(self):
        from artha.journal import DecisionJournal

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        self._save_defer_watch(journal, watch_id="watch-audit")
        journal.update_defer_watch_status(
            "watch-audit",
            "triggered_reviewing",
            notes="first note",
            trigger_price=10.0,
            set_triggered_at=True,
        )
        journal.update_defer_watch_status("watch-audit", "reviewed_no_buy", notes="second note")
        row = journal.get_defer_watch("watch-audit")
        self.assertEqual(row["status"], "reviewed_no_buy")
        self.assertEqual(row["trigger_price"], 10.0)
        self.assertTrue(row["triggered_at"])
        self.assertIn("first note", row["notes"])
        self.assertIn("second note", row["notes"])

    def test_defer_watch_extraction_persists_multiple_entry_zones(self):
        from types import SimpleNamespace

        from artha.defer_watchlist import extract_entry_zones, record_defer_watch
        from artha.journal import DecisionJournal

        action = (
            "DEFER — do not open NEE today. Set two re-evaluation alerts: "
            "(1) bullish confirmation if NEE closes above the 50-day SMA of $91.87 on volume above average, "
            "or (2) value/mean-reversion entry if price pulls back toward the lower Bollinger Band near $81.80 "
            "and RSI moves below 30 then stabilizes."
        )
        zones = extract_entry_zones(action, current_price=85.88, max_zones=3)
        self.assertGreaterEqual(len(zones), 2)
        self.assertTrue(any(z.trigger_type == "breakout" and 90 <= z.low <= 93 for z in zones))
        self.assertTrue(any(z.trigger_type == "pullback" and 80 <= z.low <= 83 for z in zones))

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        decision = SimpleNamespace(
            ticker="NEE",
            final_verdict="DEFER",
            recommended_action=action,
            synthesis_report="",
            invalidation_conditions=["unit"],
            adjusted_score=61,
            opportunity_score=59,
            confidence=7,
            entry_valid_until=(datetime.now(timezone.utc) + timedelta(days=30)).date().isoformat(),
            dossier_path=str(self.tmp_dir / "NEE_dossier.json"),
            agentic_trace={"trace_path": str(self.tmp_dir / "NEE_trace.json")},
        )
        primary = record_defer_watch(decision, current_price=85.88, journal=journal)
        rows = journal.get_active_defer_watches_for_ticker("NEE")
        self.assertIsNotNone(primary)
        self.assertGreaterEqual(len(rows), 2)
        self.assertIn("extra_watch_ids", primary)
        self.assertTrue(any(row["trigger_type"] == "breakout" for row in rows))
        self.assertTrue(any(row["trigger_type"] == "pullback" for row in rows))

    def test_defer_watch_extraction_ignores_notional_and_eps_ranges(self):
        from artha.defer_watchlist import extract_entry_zones

        action = (
            "DEFER — create an entry watch at ~$215; if FCFS reclaims/holds the $215-$216 "
            "50-day SMA zone during regular market hours and bid/ask spread is safe, prepare "
            "a fractional market review for about $20-$25. Next quarter EPS estimate is $2.45-$2.70."
        )
        zones = extract_entry_zones(action, current_price=214.49, max_zones=3)
        self.assertTrue(any(214 <= z.low <= 216 and 215 <= z.high <= 217 for z in zones))
        self.assertFalse(any(19 <= z.low <= 26 for z in zones))
        self.assertFalse(any(2 <= z.low <= 3 for z in zones))

    def test_defer_watch_extraction_ignores_fractional_review_budget_range(self):
        from artha.defer_watchlist import extract_entry_zones

        action = (
            "DEFER — create an entry watch near $55; if DAR reaches that zone during regular "
            "market hours and the bid/ask spread is safe, re-review before preparing a "
            "fractional market review for about $18-$29."
        )
        zones = extract_entry_zones(action, current_price=58.37, max_zones=3)
        self.assertTrue(any(54 <= z.low <= 56 for z in zones))
        self.assertFalse(any(18 <= z.low <= 29 for z in zones))

    def test_defer_watch_extraction_rejects_far_off_artifact_zones(self):
        from artha.defer_watchlist import extract_entry_zones

        dar_action = (
            "DEFER — create an entry watch near $55. If budget reopens, prepare a "
            "fractional market review for about $20-$25; do not chase."
        )
        ivz_action = (
            "DEFER — re-review near $26.10; valuation context mentions P/E 10.29-10.79 "
            "but that is not a stock entry price."
        )

        dar_zones = extract_entry_zones(dar_action, current_price=58.12, max_zones=3)
        ivz_zones = extract_entry_zones(ivz_action, current_price=29.245, max_zones=3)

        self.assertTrue(any(54 <= z.low <= 56 for z in dar_zones))
        self.assertFalse(any(19 <= z.low <= 26 for z in dar_zones))
        self.assertTrue(any(25 <= z.low <= 27 for z in ivz_zones))
        self.assertFalse(any(10 <= z.low <= 11 for z in ivz_zones))

    def test_record_defer_watch_supersedes_old_zones_for_same_ticker(self):
        from types import SimpleNamespace

        from artha.defer_watchlist import record_defer_watch
        from artha.journal import DecisionJournal

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        self._save_defer_watch(journal, watch_id="old-watch")
        decision = SimpleNamespace(
            ticker="TEST",
            final_verdict="DEFER",
            recommended_action="DEFER — create an entry watch near $12.50 before any buy review.",
            synthesis_report="",
            invalidation_conditions=[],
            adjusted_score=61,
            opportunity_score=61,
            confidence=7,
            entry_valid_until=(datetime.now(timezone.utc) + timedelta(days=30)).date().isoformat(),
            dossier_path=str(self.tmp_dir / "new.json"),
            agentic_trace={},
        )
        created = record_defer_watch(decision, current_price=14.0, journal=journal)
        self.assertIsNotNone(created)
        old = journal.get_defer_watch("old-watch")
        active = journal.get_active_defer_watches_for_ticker("TEST")
        self.assertEqual(old["status"], "superseded")
        self.assertEqual(len(active), 1)
        self.assertAlmostEqual(float(active[0]["zone_low"]), 12.25, places=3)

    def test_broker_router_routes_weird_quote_to_research_watch_not_hard_reject(self):
        from artha.broker_router import LANE_HARD_REJECT, LANE_RESEARCH_WATCH, route_scan_candidates
        from artha.journal import DecisionJournal

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        candidates = [{"symbol": "TEST", "price": 100.0, "avg_volume": 500_000, "market_cap": 5_000_000_000, "funnel_score": 80}]

        def quote_provider(ticker):
            return {"price": 100.1, "bid": 101.0, "ask": 100.0, "averageVolume": 500_000}

        result = route_scan_candidates(
            candidates,
            session_id="router-unit",
            journal=journal,
            quote_provider=quote_provider,
            market_open=True,
            council_limit=1,
        )
        self.assertEqual(result.decisions[0].lane, LANE_RESEARCH_WATCH)
        self.assertEqual(result.decisions[0].reason_code, "quote_anomaly_or_missing_bid_ask")
        self.assertNotEqual(result.decisions[0].lane, LANE_HARD_REJECT)
        rows = journal.get_scan_routing_decisions("router-unit")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["lane"], LANE_RESEARCH_WATCH)

    def test_broker_router_clean_quote_becomes_execution_ready(self):
        from artha.broker_router import LANE_EXECUTION_READY, route_scan_candidates
        from artha.journal import DecisionJournal

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        candidates = [{"symbol": "CLEAN", "price": 100.0, "avg_volume": 500_000, "market_cap": 5_000_000_000, "funnel_score": 80}]

        def quote_provider(ticker):
            return {"price": 100.05, "bid": 100.00, "ask": 100.08, "averageVolume": 500_000}

        result = route_scan_candidates(
            candidates,
            session_id="router-clean",
            journal=journal,
            quote_provider=quote_provider,
            market_open=True,
            council_limit=1,
        )
        self.assertEqual(result.selected_for_council[0].ticker, "CLEAN")
        self.assertEqual(result.selected_for_council[0].lane, LANE_EXECUTION_READY)

    def test_broker_router_defer_cooldown_routes_to_research_watch(self):
        from artha.broker_router import LANE_RESEARCH_WATCH, route_scan_candidates
        from artha.journal import DecisionJournal

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        journal.save_recommendation(
            session_id="prior",
            ticker="COOL",
            action="DEFER",
            rationale="wait",
            confidence=7,
            price_at_recommendation=100.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        candidates = [{"symbol": "COOL", "price": 101.0, "avg_volume": 500_000, "market_cap": 5_000_000_000, "funnel_score": 80}]

        def quote_provider(ticker):
            return {"price": 101.0, "bid": 100.98, "ask": 101.02, "averageVolume": 500_000}

        result = route_scan_candidates(
            candidates,
            session_id="router-cooldown",
            journal=journal,
            quote_provider=quote_provider,
            market_open=True,
            council_limit=1,
        )
        self.assertEqual(result.decisions[0].lane, LANE_RESEARCH_WATCH)
        self.assertEqual(result.decisions[0].reason_code, "recent_defer_cooldown")
        self.assertEqual(result.selected_for_council, [])

    def test_opportunity_scout_penalizes_expensive_high_beta_momentum_in_greed(self):
        from artha.broker_router import BrokerRouteDecision, BrokerRouterResult, LANE_EXECUTION_READY
        from artha.config import Config
        from artha.opportunity_scout import OpportunityScout, OpportunityScoutResult

        hot = {
            "symbol": "HOT",
            "name": "Hot Momentum Inc.",
            "price": 200.0,
            "beta": 3.4,
            "primary_alpha_sleeve": "momentum_breakout",
            "return_1m": 65.0,
            "return_3m": 130.0,
            "price_target_consensus": {"targetConsensus": 150.0, "targetLow": 90.0, "targetHigh": 190.0},
            "dcf": {"dcf": -12.0},
            "ratios_ttm": {"priceToSalesRatioTTM": 35.0, "priceToEarningsRatioTTM": 120.0},
            "key_metrics_ttm": {"freeCashFlowYieldTTM": -0.01, "priceToFreeCashFlowRatioTTM": 240.0},
            "recommendation_trends": {"net_upgrades_30d": 0, "net_downgrades_30d": 0},
        }
        clean = {
            "symbol": "CLEAN",
            "name": "Clean Quality Corp.",
            "price": 100.0,
            "beta": 1.1,
            "primary_alpha_sleeve": "quality_value",
            "alpha_sleeve_scores": {"entry_quality": 22.0},
            "return_1m": 4.0,
            "return_3m": 9.0,
            "price_target_consensus": {"targetConsensus": 135.0, "targetLow": 92.0, "targetHigh": 155.0},
            "dcf": {"dcf": 128.0},
            "ratios_ttm": {"priceToSalesRatioTTM": 3.0, "priceToEarningsRatioTTM": 21.0},
            "key_metrics_ttm": {"freeCashFlowYieldTTM": 0.065, "priceToFreeCashFlowRatioTTM": 16.0},
            "recommendation_trends": {"net_upgrades_30d": 2, "net_downgrades_30d": 0},
        }
        result = BrokerRouterResult(
            decisions=[
                BrokerRouteDecision(
                    candidate=hot,
                    ticker="HOT",
                    candidate_rank=1,
                    lane=LANE_EXECUTION_READY,
                    bucket="buy_now",
                    reason_code="execution_ready",
                    reason="clean quote",
                    route_score=500.0,
                    funnel_score=95.0,
                    price=200.0,
                    live_price=200.0,
                    bid=199.99,
                    ask=200.01,
                    spread_pct=0.0001,
                ),
                BrokerRouteDecision(
                    candidate=clean,
                    ticker="CLEAN",
                    candidate_rank=2,
                    lane=LANE_EXECUTION_READY,
                    bucket="buy_now",
                    reason_code="execution_ready",
                    reason="clean quote",
                    route_score=80.0,
                    funnel_score=82.0,
                    price=100.0,
                    live_price=100.0,
                    bid=99.99,
                    ask=100.01,
                    spread_pct=0.0002,
                ),
            ],
            selected_for_council=[],
            execution_ready=[],
            research_watch=[],
            hard_reject=[],
        )

        with patch.object(Config, "OPPORTUNITY_SCOUT_LLM_ENABLED", False), \
             patch.object(OpportunityScoutResult, "save_artifact", return_value=""):
            ranked = OpportunityScout().rank(
                result,
                session_id="scout-greed",
                market_snapshot={"fear_greed": {"value": 72, "label": "Greed"}},
                deployment={"deployable_amount": 50.0},
                batch_size=2,
                max_batches=1,
            )

        self.assertEqual(ranked.ranked_cards[0].ticker, "CLEAN")
        hot_card = next(c for c in ranked.cards if c.ticker == "HOT")
        self.assertIn("price_above_consensus_target", hot_card.sanity_flags)
        self.assertIn("greed_high_beta_momentum_penalty", hot_card.sanity_flags)
        self.assertIn("negative_or_zero_fcf_yield", hot_card.sanity_flags)

    def test_opportunity_scout_research_only_budget_limits_to_first_batch(self):
        from artha.broker_router import BrokerRouteDecision, BrokerRouterResult, LANE_EXECUTION_READY
        from artha.config import Config
        from artha.opportunity_scout import OpportunityScout, OpportunityScoutResult

        decisions = []
        for idx in range(16):
            ticker = f"T{idx:02d}"
            candidate = {
                "symbol": ticker,
                "price": 50.0 + idx,
                "beta": 1.0,
                "primary_alpha_sleeve": "quality_value",
                "price_target_consensus": {"targetConsensus": 80.0 + idx},
                "key_metrics_ttm": {"freeCashFlowYieldTTM": 0.05},
                "ratios_ttm": {"priceToEarningsRatioTTM": 18.0, "priceToSalesRatioTTM": 2.0},
                "recommendation_trends": {"net_upgrades_30d": 1, "net_downgrades_30d": 0},
            }
            decisions.append(
                BrokerRouteDecision(
                    candidate=candidate,
                    ticker=ticker,
                    candidate_rank=idx + 1,
                    lane=LANE_EXECUTION_READY,
                    bucket="buy_now",
                    reason_code="execution_ready",
                    reason="clean quote",
                    route_score=100.0 - idx,
                    funnel_score=90.0 - idx,
                    price=50.0 + idx,
                    live_price=50.0 + idx,
                    bid=49.99 + idx,
                    ask=50.01 + idx,
                    spread_pct=0.0004,
                )
            )
        result = BrokerRouterResult(
            decisions=decisions,
            selected_for_council=[],
            execution_ready=[],
            research_watch=[],
            hard_reject=[],
        )

        with patch.object(Config, "OPPORTUNITY_SCOUT_LLM_ENABLED", False), \
             patch.object(Config, "SCAN_MIN_DEPLOYABLE_FOR_BUY_COUNCIL", 10.0), \
             patch.object(OpportunityScoutResult, "save_artifact", return_value=""):
            ranked = OpportunityScout().rank(
                result,
                session_id="scout-low-cash",
                market_snapshot={"fear_greed": {"value": 72, "label": "Greed"}},
                deployment={"deployable_amount": 0.88},
                batch_size=8,
                max_batches=5,
            )

        self.assertTrue(ranked.research_only)
        self.assertEqual(len(ranked.batches), 1)
        self.assertEqual(len(ranked.batches[0]), 8)
        self.assertIn("research-only", ranked.summary)

    def test_opportunity_scout_agent_uses_tools_and_gpt55_xhigh(self):
        from artha.broker_router import BrokerRouteDecision, BrokerRouterResult, LANE_EXECUTION_READY
        from artha.config import Config
        from artha.opportunity_scout import OpportunityScout, OpportunityScoutResult

        class FakeClient:
            instances = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.calls = 0
                FakeClient.instances.append(self)

            def chat(self, prompt):
                self.calls += 1
                if self.calls == 1:
                    return json.dumps(
                        {
                            "tool_name": "read_candidate_cards",
                            "args": {"include_all": True},
                            "reason": "Need compact cards before ranking.",
                        }
                    )
                self.last_prompt = prompt
                return json.dumps(
                    {
                        "final_ranking": True,
                        "summary": "Prefer XYZ after reading the tool result.",
                        "ranked_tickers": ["XYZ", "ABC"],
                    }
                )

        def row(ticker: str, rank: int) -> BrokerRouteDecision:
            candidate = {
                "symbol": ticker,
                "price": 100.0,
                "beta": 1.0,
                "primary_alpha_sleeve": "quality_value",
                "price_target_consensus": {"targetConsensus": 140.0},
                "key_metrics_ttm": {"freeCashFlowYieldTTM": 0.05},
                "ratios_ttm": {"priceToEarningsRatioTTM": 18.0, "priceToSalesRatioTTM": 2.0},
                "recommendation_trends": {"net_upgrades_30d": 1, "net_downgrades_30d": 0},
            }
            return BrokerRouteDecision(
                candidate=candidate,
                ticker=ticker,
                candidate_rank=rank,
                lane=LANE_EXECUTION_READY,
                bucket="buy_now",
                reason_code="execution_ready",
                reason="clean quote",
                route_score=90.0,
                funnel_score=90.0,
                price=100.0,
                live_price=100.0,
                bid=99.99,
                ask=100.01,
                spread_pct=0.0002,
            )

        router_result = BrokerRouterResult(
            decisions=[row("ABC", 1), row("XYZ", 2)],
            selected_for_council=[],
            execution_ready=[],
            research_watch=[],
            hard_reject=[],
        )

        with patch.object(Config, "OPPORTUNITY_SCOUT_LLM_ENABLED", True), \
             patch.object(Config, "OPPORTUNITY_SCOUT_MODEL", "gpt-5.5"), \
             patch.object(Config, "OPPORTUNITY_SCOUT_REASONING_EFFORT", "xhigh"), \
             patch.object(Config, "OPPORTUNITY_SCOUT_MAX_TOOL_STEPS", 3), \
             patch.object(OpportunityScoutResult, "save_artifact", return_value=""):
            ranked = OpportunityScout(model_client_cls=FakeClient).rank(
                router_result,
                session_id="scout-agentic",
                market_snapshot={"fear_greed": {"value": 50, "label": "Neutral"}},
                deployment={"deployable_amount": 50.0},
                batch_size=2,
                max_batches=1,
            )

        self.assertTrue(ranked.agentic_used)
        self.assertEqual(ranked.model_used, "gpt-5.5")
        self.assertEqual(ranked.reasoning_effort, "xhigh")
        self.assertEqual(FakeClient.instances[0].kwargs["model"], "gpt-5.5")
        self.assertEqual(FakeClient.instances[0].kwargs["reasoning_effort"], "xhigh")
        self.assertEqual(ranked.ranked_cards[0].ticker, "XYZ")
        self.assertEqual(ranked.tool_trace[0]["tool_name"], "read_candidate_cards")
        self.assertEqual(ranked.tool_trace[-1]["tool_name"], "final_ranking")

    def test_opportunity_scout_does_not_fail_when_agent_repeats_card_read(self):
        from artha.broker_router import BrokerRouteDecision, BrokerRouterResult, LANE_EXECUTION_READY
        from artha.config import Config
        from artha.opportunity_scout import OpportunityScout, OpportunityScoutResult

        class RepeatingClient:
            def __init__(self, **kwargs):
                self.calls = 0

            def chat(self, prompt):
                self.calls += 1
                return json.dumps(
                    {
                        "tool_name": "read_candidate_cards",
                        "args": {"include_all": True},
                        "reason": "Still asking for all compact cards.",
                    }
                )

        def row(ticker: str, rank: int) -> BrokerRouteDecision:
            return BrokerRouteDecision(
                candidate={
                    "symbol": ticker,
                    "price": 100.0,
                    "beta": 1.0,
                    "primary_alpha_sleeve": "quality_value",
                    "price_target_consensus": {"targetConsensus": 135.0},
                    "key_metrics_ttm": {"freeCashFlowYieldTTM": 0.05},
                    "ratios_ttm": {"priceToEarningsRatioTTM": 18.0},
                },
                ticker=ticker,
                candidate_rank=rank,
                lane=LANE_EXECUTION_READY,
                bucket="buy_now",
                reason_code="execution_ready",
                reason="clean quote",
                route_score=90.0 - rank,
                funnel_score=90.0 - rank,
                price=100.0,
                live_price=100.0,
                bid=99.99,
                ask=100.01,
                spread_pct=0.0002,
            )

        router_result = BrokerRouterResult(
            decisions=[row(f"T{idx:02d}", idx) for idx in range(1, 13)],
            selected_for_council=[],
            execution_ready=[],
            research_watch=[],
            hard_reject=[],
        )

        with patch.object(Config, "OPPORTUNITY_SCOUT_LLM_ENABLED", True), \
             patch.object(Config, "OPPORTUNITY_SCOUT_MAX_TOOL_STEPS", 3), \
             patch.object(OpportunityScoutResult, "save_artifact", return_value=""):
            ranked = OpportunityScout(model_client_cls=RepeatingClient).rank(
                router_result,
                session_id="scout-repeat-read",
                market_snapshot={"fear_greed": {"value": 50, "label": "Neutral"}},
                deployment={"deployable_amount": 50.0},
                batch_size=8,
                max_batches=2,
            )

        self.assertTrue(ranked.agentic_used)
        self.assertFalse(ranked.deterministic_fallback_reason)
        self.assertEqual(ranked.tool_trace[-1]["tool_name"], "deterministic_safety_finish")
        self.assertEqual(ranked.tool_trace[-1]["status"], "WARN")
        self.assertEqual(len(ranked.selected_for_council), 8)

    def test_opportunity_scout_tools_expose_fmp_web_and_broker_context(self):
        from artha.broker_router import BrokerRouteDecision, LANE_EXECUTION_READY
        from artha.config import Config
        from artha.opportunity_scout import OpportunityScout

        class FakeFMPCollector:
            def quote(self, ticker):
                return {"symbol": ticker, "price": 101.0}

            def price_target_consensus(self, ticker, **kwargs):
                calls.append(("price_target_consensus", ticker, kwargs))
                return {"targetConsensus": 135.0}

            def dcf(self, ticker, **kwargs):
                calls.append(("dcf", ticker, kwargs))
                return {"dcf": 128.0}

            def ratios_ttm(self, ticker, **kwargs):
                calls.append(("ratios_ttm", ticker, kwargs))
                return {"priceToEarningsRatioTTM": 20.0}

            def key_metrics_ttm(self, ticker, **kwargs):
                calls.append(("key_metrics_ttm", ticker, kwargs))
                return {"freeCashFlowYieldTTM": 0.055}

        snapshot_path = self.tmp_dir / "robinhood_snapshot.json"
        snapshot_path.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "generated_at": "2026-06-15T16:00:00+00:00",
                    "positions": [{"symbol": "JNJ", "quantity": "0.074763"}],
                    "warnings": [],
                }
            ),
            encoding="utf-8",
        )
        row = BrokerRouteDecision(
            candidate={
                "symbol": "ABC",
                "price": 100.0,
                "beta": 1.0,
                "primary_alpha_sleeve": "quality_value",
                "price_target_consensus": {"targetConsensus": 130.0},
                "key_metrics_ttm": {"freeCashFlowYieldTTM": 0.05},
                "ratios_ttm": {"priceToEarningsRatioTTM": 18.0},
            },
            ticker="ABC",
            candidate_rank=1,
            lane=LANE_EXECUTION_READY,
            bucket="buy_now",
            reason_code="execution_ready",
            reason="clean quote",
            route_score=80.0,
            funnel_score=80.0,
            price=100.0,
            live_price=100.0,
            bid=99.99,
            ask=100.01,
            spread_pct=0.0002,
        )
        scout = OpportunityScout()
        card = scout._card_from_decision(row, market_snapshot={"fear_greed": {"value": 50}})
        calls = []

        with patch("artha.collector.FMPCollector", FakeFMPCollector), \
             patch("artha.search.search_web", return_value=[{"title": "ABC raises target", "url": "https://example.com"}]), \
             patch.object(Config, "FUNNEL_ENRICH_PROVIDER_TIMEOUT_SECONDS", 4), \
             patch.object(Config, "FUNNEL_ENRICH_PROVIDER_RETRIES", 0), \
             patch.object(Config, "ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE", str(snapshot_path)):
            fmp_result = scout._run_tool("fetch_fmp_snapshot", {"tickers": ["ABC"]}, [card], {}, {})
            web_result = scout._run_tool(
                "web_research",
                {"ticker": "ABC", "query": "ABC stock analyst target"},
                [card],
                {},
                {},
            )
            broker_result = scout._run_tool("read_broker_context", {"tickers": ["ABC"]}, [card], {}, {})

        self.assertEqual(fmp_result["ABC"]["quote"]["price"], 101.0)
        for _, _, kwargs in calls:
            self.assertEqual(kwargs["timeout"], 4)
            self.assertEqual(kwargs["retries"], 0)
        self.assertEqual(web_result["results"][0]["title"], "ABC raises target")
        self.assertTrue(broker_result["snapshot"]["available"])
        self.assertEqual(broker_result["snapshot"]["position_count"], 1)
        self.assertEqual(broker_result["broker_router_cards"][0]["ticker"], "ABC")

    def test_rank_universe_uses_bounded_yfinance_timeout(self):
        from types import SimpleNamespace

        import pandas as pd

        from artha.config import Config
        from artha.rank_candidates import rank_universe

        universe = [
            SimpleNamespace(symbol="AAA", name="AAA Inc.", sector="Technology", industry="Software", market_cap=1_000_000_000),
            SimpleNamespace(symbol="BBB", name="BBB Inc.", sector="Healthcare", industry="Biotech", market_cap=2_000_000_000),
        ]

        with patch.object(Config, "YFINANCE_BATCH_SIZE", 2), \
             patch.object(Config, "YFINANCE_DOWNLOAD_TIMEOUT_SECONDS", 7), \
             patch("artha.rank_candidates.RANK_COVERAGE_DIR", self.tmp_dir / "rank-coverage-empty"), \
             patch("yfinance.download", return_value=pd.DataFrame()) as download:
            ranked = rank_universe(universe, top_n=2)

        self.assertEqual(ranked, [])
        self.assertEqual(download.call_args.kwargs["timeout"], 7)

    def test_rank_universe_stops_starting_yfinance_batches_after_total_budget(self):
        from types import SimpleNamespace

        import pandas as pd

        from artha.config import Config
        from artha.rank_candidates import rank_universe

        universe = [
            SimpleNamespace(symbol="AAA", name="AAA Inc.", sector="Technology", industry="Software", market_cap=1_000_000_000),
            SimpleNamespace(symbol="BBB", name="BBB Inc.", sector="Healthcare", industry="Biotech", market_cap=2_000_000_000),
        ]

        with patch.object(Config, "YFINANCE_BATCH_SIZE", 1), \
             patch.object(Config, "YFINANCE_DOWNLOAD_TIMEOUT_SECONDS", 7), \
             patch.object(Config, "YFINANCE_RANK_TOTAL_TIMEOUT_SECONDS", 5), \
             patch("artha.rank_candidates.RANK_COVERAGE_DIR", self.tmp_dir / "rank-coverage-budget"), \
             patch("artha.rank_candidates.time.monotonic", side_effect=[0, 0, 10]), \
             patch("yfinance.download", return_value=pd.DataFrame()) as download:
            rank_universe(universe, top_n=2)

        self.assertEqual(download.call_count, 1)

    def test_funnel_enrichment_uses_scan_safe_provider_timeouts(self):
        from types import SimpleNamespace

        from artha.config import Config
        from artha.funnel import PromotionFunnel

        calls = []

        class FakeFMPCollector:
            def ratios_ttm(self, ticker, **kwargs):
                calls.append(("ratios_ttm", ticker, kwargs))
                return {"priceToEarningsRatioTTM": 20}

            def key_metrics_ttm(self, ticker, **kwargs):
                calls.append(("key_metrics_ttm", ticker, kwargs))
                return {"freeCashFlowYieldTTM": 0.05}

            def price_target_consensus(self, ticker, **kwargs):
                calls.append(("price_target_consensus", ticker, kwargs))
                return {"targetConsensus": 120}

            def dcf(self, ticker, **kwargs):
                calls.append(("dcf", ticker, kwargs))
                return {"dcf": 115}

        def fake_recs(ticker):
            return {"consensus": "buy"}

        def fake_estimates(ticker, **kwargs):
            calls.append(("analyst_estimates", ticker, kwargs))
            return {"source": "fmp"}

        def fake_short(ticker, **kwargs):
            calls.append(("short_interest", ticker, kwargs))
            return {"source": "fmp"}

        fake_ec = SimpleNamespace(
            earnings_date=None,
            days_to_earnings=None,
            earnings_time=None,
            earnings_risk_flag=False,
            earnings_defer_flag=False,
        )

        with patch.object(Config, "FUNNEL_ENRICH_PROVIDER_TIMEOUT_SECONDS", 4), \
             patch.object(Config, "FUNNEL_ENRICH_PROVIDER_RETRIES", 0), \
             patch("artha.collector.FMPCollector", FakeFMPCollector), \
             patch("artha.analyst_signals.get_recommendation_trends", side_effect=fake_recs), \
             patch("artha.analyst_signals.get_analyst_estimates", side_effect=fake_estimates), \
             patch("artha.analyst_signals.get_short_interest", side_effect=fake_short), \
             patch("artha.earnings_calendar.get_earnings_context", return_value=fake_ec):
            enriched = PromotionFunnel()._enrich(
                [
                    {
                        "symbol": "AAA",
                        "name": "AAA Inc.",
                        "market_cap": 1_000_000_000,
                        "price": 100,
                        "avg_volume": 500_000,
                    }
                ]
            )

        self.assertEqual(enriched[0]["symbol"], "AAA")
        timed_calls = [call for call in calls if call[0] != "recommendation_trends"]
        self.assertTrue(timed_calls)
        for _, _, kwargs in timed_calls:
            self.assertEqual(kwargs["timeout"], 4)
            self.assertEqual(kwargs["retries"], 0)

    def test_funnel_enrichment_budget_returns_partial_candidates(self):
        from artha.config import Config
        from artha.funnel import PromotionFunnel

        ranked = [
            {"symbol": "AAA", "market_cap": 1_000_000_000, "price": 100, "avg_volume": 500_000},
            {"symbol": "BBB", "market_cap": 2_000_000_000, "price": 50, "avg_volume": 600_000},
        ]

        with patch.object(Config, "FUNNEL_ENRICH_TOTAL_TIMEOUT_SECONDS", 5), \
             patch("artha.funnel.time.monotonic", side_effect=[0, 10]):
            enriched = PromotionFunnel()._enrich(ranked)

        self.assertEqual([row["symbol"] for row in enriched], ["AAA", "BBB"])
        self.assertTrue(all(row["enrichment_timeout"] for row in enriched))
        self.assertTrue(all(row["enrichment_source"] == "funnel_timeout_partial" for row in enriched))

    def test_short_interest_scan_safe_timeout_skips_unbounded_yfinance_info(self):
        from artha.analyst_signals import get_short_interest
        from artha.config import Config

        with patch.object(Config, "FMP_SHORT_INTEREST_ENDPOINT", ""), \
             patch("artha.analyst_signals.yf.Ticker") as ticker:
            result = get_short_interest("AAA", timeout=4, retries=0)

        ticker.assert_not_called()
        self.assertEqual(result["ticker"], "AAA")
        self.assertEqual(result["source"], "unavailable")
        self.assertIsNone(result["short_interest_pct"])

    def test_stale_defer_auto_review_requeues_for_retry(self):
        from artha.config import Config
        from artha.journal import DecisionJournal

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        self._save_defer_watch(journal, watch_id="watch-stale")
        journal.update_defer_watch_status(
            "watch-stale",
            "triggered_reviewing",
            notes="started",
            trigger_price=10.0,
            set_triggered_at=True,
        )
        with journal._connect() as conn:
            conn.execute(
                """
                UPDATE defer_watchlist
                SET updated_at = datetime('now', '-3 hours')
                WHERE watch_id = 'watch-stale'
                """
            )
            conn.commit()

        requeued = journal.requeue_stale_defer_auto_reviews(
            Config.DEFER_AUTO_REVIEW_STALE_REVIEW_MINUTES,
        )
        row = journal.get_defer_watch("watch-stale")
        self.assertEqual(requeued, 1)
        self.assertEqual(row["status"], "active")
        self.assertIn("requeued after stale triggered_reviewing state", row["notes"])

    def test_failed_defer_auto_review_retries_with_backoff_and_failure_cap(self):
        from artha.journal import DecisionJournal

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        self._save_defer_watch(journal, watch_id="watch-failed-retry")
        journal.mark_defer_watch_review_failed(
            "watch-failed-retry",
            "Council returned no decision.",
        )
        with journal._connect() as conn:
            conn.execute(
                "UPDATE defer_watchlist SET updated_at = datetime('now', '-2 hours') WHERE watch_id = ?",
                ("watch-failed-retry",),
            )
            conn.commit()

        requeued = journal.requeue_failed_defer_auto_reviews(
            retry_delay_minutes=60,
            max_failures=3,
        )
        row = journal.get_defer_watch("watch-failed-retry")
        self.assertEqual(requeued, 1)
        self.assertEqual(row["status"], "active")
        self.assertEqual(row["review_failure_count"], 1)

        for attempt in range(2, 4):
            journal.mark_defer_watch_review_failed(
                "watch-failed-retry",
                f"Failure {attempt}",
            )
            with journal._connect() as conn:
                conn.execute(
                    "UPDATE defer_watchlist SET updated_at = datetime('now', '-2 hours') WHERE watch_id = ?",
                    ("watch-failed-retry",),
                )
                conn.commit()
            if attempt < 3:
                self.assertEqual(
                    journal.requeue_failed_defer_auto_reviews(60, 3),
                    1,
                )

        self.assertEqual(journal.requeue_failed_defer_auto_reviews(60, 3), 0)
        final_row = journal.get_defer_watch("watch-failed-retry")
        self.assertEqual(final_row["status"], "review_failed")
        self.assertEqual(final_row["review_failure_count"], 3)

    def test_implausible_defer_watch_zone_is_invalidated(self):
        from artha.journal import DecisionJournal

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        self._save_defer_watch(journal, watch_id="watch-bad-zone")
        with journal._connect() as conn:
            conn.execute(
                """
                UPDATE defer_watchlist
                SET current_price = 214.0,
                    zone_low = 20.0,
                    zone_high = 25.0
                WHERE watch_id = 'watch-bad-zone'
                """
            )
            conn.commit()

        invalidated = journal.invalidate_implausible_defer_watches()
        row = journal.get_defer_watch("watch-bad-zone")
        self.assertEqual(invalidated, 1)
        self.assertEqual(row["status"], "invalid_zone")
        self.assertIn("Auto-invalidated implausible entry zone", row["notes"])

    def test_defer_watch_trigger_auto_reviews_and_stops_on_no_buy(self):
        import asyncio
        from types import SimpleNamespace

        from artha.config import Config
        from artha.journal import DecisionJournal
        from artha.scheduler import ArthaScheduler

        class FakeTelegram:
            enabled = True

            def __init__(self):
                self.messages = []

            def send_alert(self, text):
                self.messages.append(text)
                return True

        old_config = (
            Config.DEFER_AUTO_REVIEW_ENABLED,
            Config.DEFER_AUTO_REVIEW_MAX_PER_CYCLE,
            Config.DEFER_AUTO_REVIEW_PREPARE_ROBINHOOD_REVIEW,
        )
        try:
            Config.DEFER_AUTO_REVIEW_ENABLED = True
            Config.DEFER_AUTO_REVIEW_MAX_PER_CYCLE = 1
            Config.DEFER_AUTO_REVIEW_PREPARE_ROBINHOOD_REVIEW = True

            journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
            self._save_defer_watch(journal, watch_id="watch-no-buy")
            scheduler = ArthaScheduler()
            scheduler.sell_engine = SimpleNamespace(journal=journal)
            scheduler.telegram = FakeTelegram()
            quote = {"price": 10.0, "volume": 2_000_000, "bid": 9.99, "ask": 10.01}
            scheduler.monitor = SimpleNamespace(
                collector=SimpleNamespace(yf=SimpleNamespace(quote=lambda ticker: quote))
            )
            scheduler.collector = SimpleNamespace(
                collect_stock=lambda ticker: {**_sample_stock(price=10.0), "ticker": ticker, "yf_quote": quote},
                collect_macro=lambda: {"fed_funds": 4.5},
                collect_market_overview=lambda: {"fear_greed": {"value": 50, "label": "neutral"}},
            )
            scheduler.council = SimpleNamespace(
                analyze_stock=lambda stock, macro, market, **kwargs: SimpleNamespace(
                    ticker="TEST",
                    final_verdict="DEFER",
                    adjusted_score=51,
                    opportunity_score=51,
                    recommended_action="Still wait; only revisit if evidence improves.",
                    dossier_path=str(self.tmp_dir / "fresh_defer_dossier.json"),
                )
            )

            asyncio.run(scheduler._run_defer_watchlist_check())
            row = journal.get_defer_watch("watch-no-buy")
            self.assertEqual(row["status"], "reviewed_defer")
            self.assertIn("Auto-review completed", row["notes"])
            self.assertIn("verdict=DEFER", row["notes"])
            self.assertEqual(journal.get_execution_orders(limit=5), [])
            self.assertEqual(len(scheduler.telegram.messages), 1)
            self.assertIn("Fresh council verdict: DEFER", scheduler.telegram.messages[0])
            self.assertIn("No real Robinhood order was placed", scheduler.telegram.messages[0])
        finally:
            (
                Config.DEFER_AUTO_REVIEW_ENABLED,
                Config.DEFER_AUTO_REVIEW_MAX_PER_CYCLE,
                Config.DEFER_AUTO_REVIEW_PREPARE_ROBINHOOD_REVIEW,
            ) = old_config

    def test_legacy_triggered_defer_watch_gets_auto_review_catchup(self):
        import asyncio
        from types import SimpleNamespace

        from artha.config import Config
        from artha.journal import DecisionJournal
        from artha.scheduler import ArthaScheduler

        class FakeTelegram:
            enabled = False

            def send_alert(self, text):
                return True

        old_config = (
            Config.DEFER_AUTO_REVIEW_ENABLED,
            Config.DEFER_AUTO_REVIEW_MAX_PER_CYCLE,
            Config.DEFER_AUTO_REVIEW_LEGACY_TRIGGER_LOOKBACK_HOURS,
        )
        try:
            Config.DEFER_AUTO_REVIEW_ENABLED = True
            Config.DEFER_AUTO_REVIEW_MAX_PER_CYCLE = 1
            Config.DEFER_AUTO_REVIEW_LEGACY_TRIGGER_LOOKBACK_HOURS = 24

            journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
            self._save_defer_watch(journal, watch_id="watch-legacy")
            journal.update_defer_watch_status(
                "watch-legacy",
                "triggered",
                notes="legacy manual trigger",
                trigger_price=10.0,
                set_triggered_at=True,
            )
            scheduler = ArthaScheduler()
            scheduler.sell_engine = SimpleNamespace(journal=journal)
            scheduler.telegram = FakeTelegram()
            quote = {"price": 10.0, "volume": 2_000_000, "bid": 9.99, "ask": 10.01}
            scheduler.monitor = SimpleNamespace(
                collector=SimpleNamespace(yf=SimpleNamespace(quote=lambda ticker: quote))
            )
            scheduler.collector = SimpleNamespace(
                collect_stock=lambda ticker: {**_sample_stock(price=10.0), "ticker": ticker, "yf_quote": quote},
                collect_macro=lambda: {"fed_funds": 4.5},
                collect_market_overview=lambda: {"fear_greed": {"value": 50, "label": "neutral"}},
            )
            scheduler.council = SimpleNamespace(
                analyze_stock=lambda stock, macro, market, **kwargs: SimpleNamespace(
                    ticker="TEST",
                    final_verdict="WATCH",
                    adjusted_score=49,
                    opportunity_score=49,
                    recommended_action="Still watch after catch-up review.",
                    dossier_path=str(self.tmp_dir / "fresh_watch_dossier.json"),
                )
            )

            asyncio.run(scheduler._run_defer_watchlist_check())
            row = journal.get_defer_watch("watch-legacy")
            self.assertEqual(row["status"], "reviewed_defer")
            self.assertIn("legacy manual trigger", row["notes"])
            self.assertIn("verdict=WATCH", row["notes"])
        finally:
            (
                Config.DEFER_AUTO_REVIEW_ENABLED,
                Config.DEFER_AUTO_REVIEW_MAX_PER_CYCLE,
                Config.DEFER_AUTO_REVIEW_LEGACY_TRIGGER_LOOKBACK_HOURS,
            ) = old_config

    def test_defer_watch_auto_review_skips_same_day_executed_buy(self):
        import asyncio
        from types import SimpleNamespace

        from artha.journal import DecisionJournal
        from artha.scheduler import ArthaScheduler

        class FakeTelegram:
            enabled = True

            def __init__(self):
                self.messages = []

            def send_alert(self, text):
                self.messages.append(text)
                return True

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        self._save_defer_watch(journal, watch_id="watch-dupe")
        now = datetime.now(timezone.utc).isoformat()
        journal.save_execution_order(
            {
                "order_intent_id": "intent-existing-buy",
                "ticker": "TEST",
                "side": "buy",
                "order_type": "market",
                "status": "filled",
                "broker_order_id": "broker-existing-buy",
                "dry_run": False,
                "submitted_at": now,
                "filled_at": now,
                "notional": 12.25,
                "quantity": 1.0,
                "estimated_price": 12.25,
                "request_json": {"symbol": "TEST", "side": "buy", "type": "market"},
                "response_json": {"order": {"id": "broker-existing-buy", "state": "filled"}},
                "notes": "Existing same-day fill.",
            }
        )

        scheduler = ArthaScheduler()
        scheduler.sell_engine = SimpleNamespace(journal=journal)
        scheduler.telegram = FakeTelegram()
        scheduler.collector = SimpleNamespace(
            collect_stock=lambda ticker: (_ for _ in ()).throw(AssertionError("collector should not run")),
            collect_macro=lambda: (_ for _ in ()).throw(AssertionError("macro should not run")),
            collect_market_overview=lambda: (_ for _ in ()).throw(AssertionError("market should not run")),
        )
        scheduler.council = SimpleNamespace(
            analyze_stock=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("council should not run"))
        )

        alert = asyncio.run(
            scheduler._run_defer_watch_auto_review(
                watch=journal.get_defer_watch("watch-dupe"),
                payload={"ticker": "TEST", "watch_id": "watch-dupe", "message": "TEST reached entry watch."},
                quote={"price": 10.0, "bid": 9.99, "ask": 10.01},
                price_float=10.0,
                journal=journal,
            )
        )

        watch = journal.get_defer_watch("watch-dupe")
        orders = journal.get_execution_orders(ticker="TEST", limit=5)
        self.assertEqual(watch["status"], "reviewed_no_buy")
        self.assertIn("same-day buy", watch["notes"])
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["status"], "filled")
        self.assertEqual(alert.metadata["existing_execution_order_row"], orders[0]["id"])
        self.assertEqual(len(scheduler.telegram.messages), 1)
        self.assertIn("already recorded a same-day buy", scheduler.telegram.messages[0])

    def test_defer_watch_trigger_buy_side_prepares_robinhood_review_only(self):
        import asyncio
        from types import SimpleNamespace

        from artha.config import Config
        from artha.journal import DecisionJournal
        from artha.scheduler import ArthaScheduler

        class FakeTelegram:
            enabled = True

            def __init__(self):
                self.messages = []

            def send_alert(self, text):
                self.messages.append(text)
                return True

        old_config = (
            Config.DEFER_AUTO_REVIEW_ENABLED,
            Config.DEFER_AUTO_REVIEW_MAX_PER_CYCLE,
            Config.DEFER_AUTO_REVIEW_PREPARE_ROBINHOOD_REVIEW,
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            Config.EXECUTION_OFFICER_LLM_ENABLED,
            Config.ROBINHOOD_AUTO_BUY_ENABLED,
        )
        try:
            Config.DEFER_AUTO_REVIEW_ENABLED = True
            Config.DEFER_AUTO_REVIEW_MAX_PER_CYCLE = 1
            Config.DEFER_AUTO_REVIEW_PREPARE_ROBINHOOD_REVIEW = True
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            Config.EXECUTION_OFFICER_LLM_ENABLED = False
            Config.ROBINHOOD_AUTO_BUY_ENABLED = True

            dossier_path = str(self.tmp_dir / "fresh_starter_dossier.json")
            journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
            journal.save_supervisor_run(
                {
                    "generated_at": "2026-06-04T14:30:00+00:00",
                    "severity": "PASS",
                    "report_hash": "unit",
                    "report_text": "unit",
                    "payload": {"checks": []},
                    "sent_to_telegram": False,
                }
            )
            journal.save_decision_features(
                {
                    "dossier_path": dossier_path,
                    "generated_at": "2026-06-04T14:31:00+00:00",
                    "ticker": "TEST",
                    "final_verdict": "STARTER",
                    "opportunity_score": 69,
                    "adjusted_score": 69,
                    "confidence": 8,
                    "price": 10.0,
                    "sector": "Technology",
                    "evidence_count": 12,
                    "source_count": 4,
                    "feature_json": json.dumps({"unit": True}),
                }
            )
            self._save_defer_watch(journal, watch_id="watch-buy")
            scheduler = ArthaScheduler()
            scheduler.sell_engine = SimpleNamespace(journal=journal)
            scheduler.telegram = FakeTelegram()
            quote = {"price": 10.0, "volume": 2_000_000, "bid": 9.99, "ask": 10.01}
            scheduler.monitor = SimpleNamespace(
                collector=SimpleNamespace(yf=SimpleNamespace(quote=lambda ticker: quote))
            )
            scheduler.collector = SimpleNamespace(
                collect_stock=lambda ticker: {**_sample_stock(price=10.0), "ticker": ticker, "yf_quote": quote},
                collect_macro=lambda: {"fed_funds": 4.5},
                collect_market_overview=lambda: {"fear_greed": {"value": 55, "label": "neutral"}},
            )
            scheduler.council = SimpleNamespace(
                analyze_stock=lambda stock, macro, market, **kwargs: SimpleNamespace(
                    ticker="TEST",
                    final_verdict="STARTER",
                    adjusted_score=72,
                    opportunity_score=72,
                    confidence=8,
                    recommended_action="Starter is now attractive at the triggered entry zone.",
                    dossier_path=dossier_path,
                )
            )

            # Wave 2: dry_run now follows Config.ROBINHOOD_DRY_RUN_ONLY, so the
            # guardrails enforce market hours at queue time; pin market open.
            from unittest.mock import patch as _patch

            from artha.scheduler import MarketHours as _MarketHours

            with _patch.object(_MarketHours, "is_market_open", lambda self, now=None: True):
                asyncio.run(scheduler._run_defer_watchlist_check())
            watch = journal.get_defer_watch("watch-buy")
            orders = journal.get_execution_orders(limit=5)
            self.assertEqual(watch["status"], "auto_buy_queued")
            self.assertIn("execution_order_row", watch["notes"])
            self.assertEqual(len(orders), 1)
            self.assertEqual(orders[0]["status"], "review_ready")
            self.assertIsNone(orders[0]["submitted_at"])
            self.assertIn('"account_number": "222233334"', orders[0]["request_json"])
            self.assertIn('"market_hours": "regular_hours"', orders[0]["request_json"])
            self.assertEqual(len(scheduler.telegram.messages), 1)
            self.assertIn("Automatic buy queued", scheduler.telegram.messages[0])
            self.assertIn("no user review or buy button", scheduler.telegram.messages[0])
            self.assertIn("No real Robinhood order was placed", scheduler.telegram.messages[0])
            actions = journal.get_trade_actions(limit=5)
            self.assertEqual(len(actions), 1)
            self.assertEqual(actions[0]["action_type"], "auto_buy")
        finally:
            (
                Config.DEFER_AUTO_REVIEW_ENABLED,
                Config.DEFER_AUTO_REVIEW_MAX_PER_CYCLE,
                Config.DEFER_AUTO_REVIEW_PREPARE_ROBINHOOD_REVIEW,
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
                Config.EXECUTION_OFFICER_LLM_ENABLED,
                Config.ROBINHOOD_AUTO_BUY_ENABLED,
            ) = old_config

    def test_execution_guardrails_allow_clean_limit_dry_run(self):
        from datetime import datetime, timezone

        from artha.execution import RobinhoodExecutionGuardrails, build_order_intent

        journal = self._journal_with_clean_buy_decision()
        intent = build_order_intent(
            ticker="TEST",
            side="buy",
            notional=25.0,
            limit_price=100.0,
            estimated_price=100.0,
        )
        result = RobinhoodExecutionGuardrails().evaluate(
            intent,
            self._market_data(),
            journal,
            now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.checks["decision_evidence"]["final_verdict"], "STARTER")

    def test_execution_guardrails_block_bad_order_and_low_liquidity(self):
        from datetime import datetime, timezone

        from artha.execution import OrderIntent, RobinhoodExecutionGuardrails

        journal = self._journal_with_clean_buy_decision()
        intent = OrderIntent(
            ticker="TEST",
            side="buy",
            order_type="market",
            notional=25.0,
            estimated_price=4.0,
        )
        result = RobinhoodExecutionGuardrails().evaluate(
            intent,
            {"price": 4.0, "volume": 1000, "bid": 3.90, "ask": 4.10},
            journal,
            now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
        )
        joined = " ".join(result.reasons)
        self.assertFalse(result.passed)
        self.assertIn("reference price", joined)
        self.assertIn("below the $5.00", joined)
        self.assertIn("Dollar volume", joined)

    def test_execution_guardrails_allow_review_only_buys_on_nonfatal_supervisor_warn(self):
        from datetime import datetime, timezone

        from artha.execution import RobinhoodExecutionGuardrails, build_order_intent
        from artha.journal import DecisionJournal

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        journal.save_supervisor_run(
            {
                "generated_at": "2026-06-04T14:30:00+00:00",
                "severity": "WARN",
                "report_hash": "unit",
                "report_text": "unit",
                "payload": {"checks": [{"name": "recent_logs", "status": "WARN"}]},
                "sent_to_telegram": False,
            }
        )
        journal.save_decision_features(
            {
                "dossier_path": str(self.tmp_dir / "TEST_dossier.json"),
                "generated_at": "2026-06-04T14:20:00+00:00",
                "ticker": "TEST",
                "final_verdict": "BUY",
                "evidence_count": 12,
                "feature_json": "{}",
            }
        )
        intent = build_order_intent("TEST", "buy", notional=25, limit_price=100, estimated_price=100)
        result = RobinhoodExecutionGuardrails().evaluate(
            intent,
            self._market_data(),
            journal,
            now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
        )
        self.assertTrue(result.passed, result.reasons)
        self.assertTrue(result.checks["supervisor"]["buy_gate"]["allowed"])

    def test_pause_buying_persists_control_notifies_once_and_never_touches_services(self):
        from artha.config import Config
        from artha.stand_down import buying_paused, pause_buying

        class FakeSender:
            enabled = True

            def __init__(self):
                self.messages = []

            def send_message(self, text, parse_mode=None, silent=False):
                self.messages.append((text, parse_mode, silent))
                return True

        old_config = (
            Config.ARTHA_AUTO_PAUSE_BUYS_ON_LIMIT,
            Config.ROBINHOOD_CONTROL_FILE,
        )
        sender = FakeSender()
        try:
            Config.ARTHA_AUTO_PAUSE_BUYS_ON_LIMIT = True
            Config.ROBINHOOD_CONTROL_FILE = str(self.tmp_dir / "control.json")
            # Pre-existing operator state must be preserved, never clobbered.
            Path(Config.ROBINHOOD_CONTROL_FILE).write_text(
                json.dumps({"trading_disabled": False, "reason": "operator baseline"}),
                encoding="utf-8",
            )

            result = pause_buying(
                "Daily buy order cap reached (2).",
                source="unit-test",
                telegram_sender=sender,
            )

            control = json.loads(Path(Config.ROBINHOOD_CONTROL_FILE).read_text(encoding="utf-8"))
            self.assertTrue(result["success"])
            self.assertEqual(result["operation"], "buy_pause")
            # The pause NEVER flips the global kill switch: sells must stay live.
            self.assertFalse(control["trading_disabled"])
            self.assertEqual(control["reason"], "operator baseline")
            self.assertIn("Daily buy order cap reached", control["buying_paused_reason"])
            paused, pause_reason = buying_paused(control)
            self.assertTrue(paused)
            self.assertIn("Daily buy order cap reached", pause_reason)
            self.assertEqual(len(sender.messages), 1)
            self.assertIn("Buys paused", sender.messages[0][0])
            self.assertIn("stop-losses", sender.messages[0][0])

            # Same-day repeat: control refreshed, but no duplicate Telegram.
            repeat = pause_buying(
                "Daily buy order cap reached (2).",
                source="unit-test-repeat",
                telegram_sender=sender,
            )
            self.assertTrue(repeat["telegram_suppressed_duplicate"])
            self.assertEqual(len(sender.messages), 1)

            # An expired pause (yesterday) no longer blocks buys.
            control["buying_paused_until"] = "2020-01-01"
            paused, _ = buying_paused(control)
            self.assertFalse(paused)
        finally:
            (
                Config.ARTHA_AUTO_PAUSE_BUYS_ON_LIMIT,
                Config.ROBINHOOD_CONTROL_FILE,
            ) = old_config

    def test_fast_news_probe_suppressed_after_sentinel_alert(self):
        """run_fast_scan skips a headline whose rich sentinel alert already went out.

        The fast path probes AlertManager with an INFO-severity Alert for the
        sentinel key; INFO must not trip the severity-upgrade bypass.
        """
        from artha.monitor import Alert, AlertManager

        history = AlertManager(history_path=self.tmp_dir / "alert_history.json")
        history.record_sent_alerts(
            [Alert(ticker="V", alert_type="news_sentinel_abc123def456", severity="CRITICAL", message="rich analysis")]
        )
        probe = Alert(ticker="V", alert_type="news_sentinel_abc123def456", severity="INFO", message="")
        self.assertFalse(history.should_send(probe))
        fresh = Alert(ticker="V", alert_type="news_sentinel_zzz999zzz999", severity="INFO", message="")
        self.assertTrue(history.should_send(fresh))

    def test_pause_buying_refuses_unreadable_control_file(self):
        """A corrupt control.json means fail-closed everywhere; the pause must
        never 'heal' it and erase whatever the operator had put in it."""
        from artha.config import Config
        from artha.stand_down import pause_buying

        old_enabled = Config.ARTHA_AUTO_PAUSE_BUYS_ON_LIMIT
        try:
            Config.ARTHA_AUTO_PAUSE_BUYS_ON_LIMIT = True
            corrupt = '{"trading_disabled": true,,}'
            Path(Config.ROBINHOOD_CONTROL_FILE).write_text(corrupt, encoding="utf-8")
            result = pause_buying("Daily buy order cap reached (2).", source="unit", send_telegram=False)
            self.assertFalse(result["success"])
            self.assertEqual(result["operation"], "buy_pause_refused_unreadable_control")
            self.assertEqual(Path(Config.ROBINHOOD_CONTROL_FILE).read_text(encoding="utf-8"), corrupt)
        finally:
            Config.ARTHA_AUTO_PAUSE_BUYS_ON_LIMIT = old_enabled

    def test_pause_buying_preserves_future_dated_pause_without_rewrite(self):
        from artha.config import Config
        from artha.stand_down import pause_buying

        old_enabled = Config.ARTHA_AUTO_PAUSE_BUYS_ON_LIMIT
        try:
            Config.ARTHA_AUTO_PAUSE_BUYS_ON_LIMIT = True
            Path(Config.ROBINHOOD_CONTROL_FILE).write_text(
                json.dumps({"buying_paused_until": "2099-01-01", "buying_paused_reason": "operator hold"}),
                encoding="utf-8",
            )
            before = Path(Config.ROBINHOOD_CONTROL_FILE).read_text(encoding="utf-8")
            result = pause_buying("Daily buy order cap reached (2).", source="unit", send_telegram=False)
            self.assertTrue(result["telegram_suppressed_duplicate"])
            self.assertEqual(result["buying_paused_until"], "2099-01-01")
            self.assertEqual(Path(Config.ROBINHOOD_CONTROL_FILE).read_text(encoding="utf-8"), before)
        finally:
            Config.ARTHA_AUTO_PAUSE_BUYS_ON_LIMIT = old_enabled

    def test_set_trading_disabled_preserves_buy_pause_keys(self):
        from artha.config import Config
        from artha.robinhood_bridge import set_trading_disabled

        old_control = Config.ROBINHOOD_CONTROL_FILE
        try:
            Config.ROBINHOOD_CONTROL_FILE = str(self.tmp_dir / "control.json")
            Path(Config.ROBINHOOD_CONTROL_FILE).write_text(
                json.dumps(
                    {
                        "trading_disabled": False,
                        "buying_paused_until": "2026-07-06",
                        "buying_paused_reason": "Daily buy order cap reached (2).",
                    }
                ),
                encoding="utf-8",
            )
            set_trading_disabled(True, "emergency", path=Config.ROBINHOOD_CONTROL_FILE)
            control = json.loads(Path(Config.ROBINHOOD_CONTROL_FILE).read_text(encoding="utf-8"))
            self.assertTrue(control["trading_disabled"])
            self.assertEqual(control["buying_paused_until"], "2026-07-06")
            self.assertEqual(control["buying_paused_reason"], "Daily buy order cap reached (2).")
        finally:
            Config.ROBINHOOD_CONTROL_FILE = old_control

    def test_pause_echo_and_sizing_reasons_never_retrigger_pause(self):
        from artha.stand_down import reason_is_buy_capacity_limit

        # The pause's own gate message embeds the original marker — it must
        # not re-match, or the pause rewrites control.json all day.
        echo = (
            "Buy-side paused until the next trading day: Daily buy order cap "
            "reached (2). Sells are unaffected."
        )
        self.assertFalse(reason_is_buy_capacity_limit(echo))
        # Per-order sizing blocks are not capacity limits.
        self.assertFalse(
            reason_is_buy_capacity_limit("Order size $400.00 exceeds the $350.00 account pilot cap.")
        )
        self.assertFalse(
            reason_is_buy_capacity_limit("Order size $30.00 exceeds the $25.00 pilot cap.")
        )
        # Real capacity limits still match, including the live broker cash check.
        self.assertTrue(reason_is_buy_capacity_limit("Daily buy order cap reached (2)."))
        self.assertTrue(
            reason_is_buy_capacity_limit(
                "Insufficient Robinhood cash/buying power for this order plus safety buffer."
            )
        )

    def test_buy_pause_blocks_buys_but_never_sells(self):
        from datetime import datetime, timezone

        from artha.config import Config
        from artha.execution import RobinhoodExecutionGuardrails, build_order_intent
        from artha.stand_down import _eastern_today

        old_enabled = Config.ARTHA_AUTO_PAUSE_BUYS_ON_LIMIT
        try:
            Config.ARTHA_AUTO_PAUSE_BUYS_ON_LIMIT = True
            Path(Config.ROBINHOOD_CONTROL_FILE).write_text(
                json.dumps(
                    {
                        "trading_disabled": False,
                        "buying_paused_until": _eastern_today().isoformat(),
                        "buying_paused_reason": "Daily buy order cap reached (2).",
                    }
                ),
                encoding="utf-8",
            )
            journal = self._journal_with_clean_buy_decision()

            buy_intent = build_order_intent("TEST", "buy", notional=10, limit_price=100, estimated_price=100)
            buy_result = RobinhoodExecutionGuardrails().evaluate(
                buy_intent,
                self._market_data(),
                journal,
                now=datetime.now(timezone.utc),
            )
            self.assertFalse(buy_result.passed)
            self.assertTrue(
                any("Buy-side paused" in reason for reason in buy_result.reasons),
                buy_result.reasons,
            )

            sell_intent = build_order_intent("TEST", "sell", notional=10, limit_price=100, estimated_price=100)
            sell_result = RobinhoodExecutionGuardrails().evaluate(
                sell_intent,
                self._market_data(),
                journal,
                now=datetime.now(timezone.utc),
            )
            # Sells may fail other gates (thesis, market data) but the buy
            # pause itself must never appear as a sell-blocking reason.
            self.assertFalse(
                any("Buy-side paused" in reason for reason in sell_result.reasons),
                sell_result.reasons,
            )
        finally:
            Config.ARTHA_AUTO_PAUSE_BUYS_ON_LIMIT = old_enabled

    def test_execution_guardrails_daily_buy_cap_pauses_buys(self):
        from datetime import datetime, timezone

        from artha.config import Config
        from artha.execution import RobinhoodExecutionGuardrails, build_order_intent
        from artha.journal import DecisionJournal

        old_max = Config.ROBINHOOD_MAX_TRADES_PER_DAY
        old_enabled = Config.ARTHA_AUTO_PAUSE_BUYS_ON_LIMIT
        try:
            Config.ROBINHOOD_MAX_TRADES_PER_DAY = 1
            Config.ARTHA_AUTO_PAUSE_BUYS_ON_LIMIT = True
            journal = self._journal_with_clean_buy_decision()
            journal.save_execution_order(
                {
                    "order_intent_id": "existing-buy",
                    "ticker": "OLD",
                    "side": "buy",
                    "order_type": "market",
                    "status": "filled",
                    "dry_run": False,
                    "submitted_at": datetime.now(timezone.utc).isoformat(),
                    "filled_at": datetime.now(timezone.utc).isoformat(),
                    "notional": 12.25,
                    "quantity": 0.1,
                    "estimated_price": 122.5,
                    "request_json": {},
                    "response_json": {},
                    "notes": "unit existing buy",
                }
            )
            intent = build_order_intent("TEST", "buy", notional=10, limit_price=100, estimated_price=100)
            with patch(
                "artha.execution.maybe_pause_buying_for_limit",
                return_value={"success": True, "operation": "buy_pause", "reason": "Daily buy order cap reached (1)."},
            ) as maybe_stop:
                result = RobinhoodExecutionGuardrails().evaluate(
                    intent,
                    self._market_data(),
                    journal,
                    now=datetime.now(timezone.utc),
                )

            self.assertFalse(result.passed)
            self.assertIn("Daily buy order cap reached", " ".join(result.reasons))
            self.assertEqual(result.checks["buy_pause"]["operation"], "buy_pause")
            maybe_stop.assert_called_once()
            self.assertTrue(
                any("Daily buy order cap reached" in str(r) for r in maybe_stop.call_args.args[0]),
                maybe_stop.call_args.args[0],
            )
        finally:
            Config.ROBINHOOD_MAX_TRADES_PER_DAY = old_max
            Config.ARTHA_AUTO_PAUSE_BUYS_ON_LIMIT = old_enabled

    def test_execution_order_attempt_is_audited_when_blocked(self):
        from datetime import datetime, timezone

        from artha.execution import build_order_intent, evaluate_and_record_order

        journal = self._journal_with_clean_buy_decision()
        intent = build_order_intent("TEST", "buy", notional=75, limit_price=100, estimated_price=100)
        result = evaluate_and_record_order(
            intent,
            market_data=self._market_data(),
            journal=journal,
            now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
        )
        rows = journal.get_execution_orders(limit=5)
        self.assertEqual(result["broker_result"]["status"], "blocked")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "blocked")
        self.assertIn("exceeds", rows[0]["notes"])
        self.assertIn("passed", rows[0]["guardrail_json"])

    def test_execution_order_notice_uses_telegram_sender(self):
        from datetime import datetime, timezone

        from artha.execution import build_order_intent, evaluate_and_record_order

        class FakeSender:
            enabled = True

            def __init__(self):
                self.messages = []

            def send_message(self, text, parse_mode=None, silent=False):
                self.messages.append((text, parse_mode, silent))
                return True

        journal = self._journal_with_clean_buy_decision()
        sender = FakeSender()
        intent = build_order_intent("TEST", "buy", notional=25, limit_price=100, estimated_price=100)
        result = evaluate_and_record_order(
            intent,
            market_data=self._market_data(),
            journal=journal,
            send_telegram=True,
            sender=sender,
            now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
        )
        self.assertTrue(result["telegram_sent"])
        self.assertEqual(len(sender.messages), 1)
        self.assertIn("ARTHA ORDER DRY-RUN", sender.messages[0][0])
        self.assertIn("No real Robinhood order was placed", sender.messages[0][0])

    def test_execution_readiness_reports_dry_run_ready(self):
        from artha.config import Config
        from artha.execution import build_execution_readiness_report
        from artha.journal import DecisionJournal

        old_values = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_REVIEW_ONLY,
            Config.ROBINHOOD_DRY_RUN_ONLY,
            Config.ROBINHOOD_AGENTIC_ENABLED,
            Config.ROBINHOOD_KILL_SWITCH,
            Config.ROBINHOOD_CONTROL_FILE,
        )
        Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "999900001"
        Config.ROBINHOOD_REVIEW_ONLY = True
        Config.ROBINHOOD_DRY_RUN_ONLY = True
        Config.ROBINHOOD_AGENTIC_ENABLED = False
        Config.ROBINHOOD_KILL_SWITCH = True
        Config.ROBINHOOD_CONTROL_FILE = str(self.tmp_dir / "control.json")
        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        try:
            report = build_execution_readiness_report(journal)
            self.assertEqual(report["status"], "PASS")
            self.assertTrue(report["ready_for_dry_run"])
            self.assertFalse(report["live_trading_enabled"])
            self.assertIn("max_position_dollars", report["guardrails"])
            self.assertEqual(report["account_allowlist"]["masked_account"], "****0001")
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_REVIEW_ONLY,
                Config.ROBINHOOD_DRY_RUN_ONLY,
                Config.ROBINHOOD_AGENTIC_ENABLED,
                Config.ROBINHOOD_KILL_SWITCH,
                Config.ROBINHOOD_CONTROL_FILE,
            ) = old_values

    def test_live_execution_readiness_text_states_unattended_auto_trade_contract(self):
        from artha.execution import format_execution_readiness

        text = format_execution_readiness(
            {
                "status": "PASS",
                "ready_for_dry_run": True,
                "live_trading_enabled": True,
                "message": "Live configuration enabled.",
                "guardrails": {},
                "supervisor": {},
            }
        )
        self.assertIn("auto_buy and auto_sell actions can execute unattended", text)
        self.assertIn("no user approval click is required", text)
        self.assertIn("fresh Robinhood snapshot", text)

    def test_trade_action_maintenance_reconciles_fills_and_expires_old_reviews(self):
        from artha.journal import DecisionJournal

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        intent_id = "rh-intent-maintenance"
        journal.save_execution_order(
            {
                "order_intent_id": intent_id,
                "ticker": "TEST",
                "side": "buy",
                "order_type": "market",
                "status": "filled",
                "dry_run": False,
            }
        )
        journal.save_trade_action(
            {
                "action_id": "ta-maintenance-filled",
                "action_type": "buy",
                "ticker": "TEST",
                "side": "buy",
                "status": "submitted",
                "order_intent_id": intent_id,
            }
        )
        journal.save_trade_action(
            {
                "action_id": "ta-maintenance-expired",
                "action_type": "buy",
                "ticker": "OLD",
                "side": "buy",
                "status": "review_ready",
                "expires_at": "2020-01-01T00:00:00+00:00",
            }
        )

        self.assertEqual(journal.reconcile_trade_actions_from_execution_orders(), 1)
        self.assertEqual(journal.expire_stale_trade_actions(), 1)
        self.assertEqual(journal.get_trade_action("ta-maintenance-filled")["status"], "filled")
        self.assertEqual(journal.get_trade_action("ta-maintenance-expired")["status"], "expired")

    def test_robinhood_account_allowlist_rejects_wrong_or_non_agentic_account(self):
        from artha.config import Config
        from artha.execution import validate_allowlisted_robinhood_account

        old_account = Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER
        Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
        try:
            normal = {
                "account_number": "111122223",
                "type": "margin",
                "nickname": "",
                "agentic_allowed": False,
                "state": "active",
            }
            result = validate_allowlisted_robinhood_account(normal)
            joined = " ".join(result.reasons)
            self.assertFalse(result.passed)
            self.assertIn("does not match", joined)
            self.assertIn("not agentic-enabled", joined)

            agentic = {
                "account_number": "222233334",
                "type": "cash",
                "nickname": "Agentic",
                "agentic_allowed": True,
                "state": "active",
            }
            ok = validate_allowlisted_robinhood_account(agentic)
            self.assertTrue(ok.passed)
            self.assertEqual(ok.checks["actual_account_masked"], "****3334")
        finally:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = old_account

    def test_robinhood_review_request_uses_allowlisted_account_and_limit_order(self):
        from artha.config import Config
        from artha.execution import build_order_intent, build_robinhood_review_request

        old_account = Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER
        Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
        try:
            intent = build_order_intent("TEST", "buy", quantity=1, limit_price=10, estimated_price=10)
            request = build_robinhood_review_request(intent)
            self.assertEqual(request["account_number"], "222233334")
            self.assertEqual(request["symbol"], "TEST")
            self.assertEqual(request["type"], "limit")
            self.assertEqual(request["time_in_force"], "gfd")
            self.assertEqual(request["market_hours"], "regular_hours")
            self.assertEqual(request["quantity"], "1")
            self.assertEqual(request["limit_price"], "10.00")
        finally:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = old_account

    def test_robinhood_review_request_resolves_fractional_buy_to_market_notional(self):
        from artha.config import Config
        from artha.execution import build_order_intent, build_robinhood_review_request

        old_account = Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER
        Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
        try:
            intent = build_order_intent("TEST", "buy", notional=17.5, quantity=1.75, limit_price=10, estimated_price=10)
            request = build_robinhood_review_request(intent)
            self.assertEqual(request["account_number"], "222233334")
            self.assertEqual(request["dollar_amount"], "17.50")
            self.assertEqual(request["type"], "market")
            self.assertEqual(request["market_hours"], "regular_hours")
            self.assertNotIn("quantity", request)
            self.assertNotIn("limit_price", request)
        finally:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = old_account

    def test_prepare_robinhood_review_records_review_ready_without_placement(self):
        from datetime import datetime, timezone

        from artha.config import Config
        from artha.execution import build_order_intent, prepare_and_record_robinhood_review

        old_account = Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER
        Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
        try:
            journal = self._journal_with_clean_buy_decision()
            account = {
                "account_number": "222233334",
                "type": "cash",
                "nickname": "Agentic",
                "agentic_allowed": True,
                "state": "active",
            }
            intent = build_order_intent("TEST", "buy", quantity=1, limit_price=10, estimated_price=10)
            result = prepare_and_record_robinhood_review(
                intent,
                account,
                market_data={"price": 10, "volume": 2_000_000, "bid": 9.99, "ask": 10.01},
                journal=journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            )
            rows = journal.get_execution_orders(limit=1)
            self.assertEqual(result["broker_result"]["status"], "review_ready", result)
            self.assertEqual(rows[0]["status"], "review_ready")
            self.assertIn('"account_number": "222233334"', rows[0]["request_json"])
            self.assertIsNone(rows[0]["submitted_at"])
        finally:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = old_account

    def test_broker_snapshot_guardrail_blocks_insufficient_cash(self):
        from datetime import datetime, timezone

        from artha.config import Config
        from artha.execution import RobinhoodExecutionGuardrails, build_order_intent

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            journal = self._journal_with_clean_buy_decision()
            intent = build_order_intent(
                "TEST",
                "buy",
                notional=25,
                quantity=2.5,
                limit_price=10,
                estimated_price=10,
                decision_dossier_path=str(self.tmp_dir / "TEST_dossier.json"),
            )
            result = RobinhoodExecutionGuardrails().evaluate(
                intent,
                market_data={"price": 10, "volume": 2_000_000, "bid": 9.99, "ask": 10.01},
                journal=journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
                broker_snapshot=self._agentic_snapshot(cash=10),
            )
            self.assertFalse(result.passed)
            self.assertTrue(any("cash" in reason.lower() or "buying power" in reason.lower() for reason in result.reasons))
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            ) = old_config

    def test_trade_action_queue_builds_review_operation_and_skip(self):
        from datetime import datetime, timezone

        from artha.config import Config
        from artha.execution import build_order_intent, prepare_and_record_robinhood_review
        from artha.robinhood_bridge import build_action_operation, queue_trade_action_from_order_payload, write_robinhood_snapshot

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE = str(self.tmp_dir / "robinhood_snapshot.json")
            write_robinhood_snapshot(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "robinhood_mcp",
                    "account": {
                        "account_number": "222233334",
                        "type": "cash",
                        "nickname": "Agentic",
                        "agentic_allowed": True,
                        "state": "active",
                    },
                    "portfolio": {"buying_power": 100, "cash": 100, "total_value": 100},
                    "positions": [],
                    "orders": [],
                },
                path=Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            )
            journal = self._journal_with_clean_buy_decision()
            account = {
                "account_number": "222233334",
                "type": "cash",
                "nickname": "Agentic",
                "agentic_allowed": True,
                "state": "active",
            }
            intent = build_order_intent("TEST", "buy", quantity=1, limit_price=10, estimated_price=10)
            result = prepare_and_record_robinhood_review(
                intent,
                account,
                market_data={"price": 10, "volume": 2_000_000, "bid": 9.99, "ask": 10.01},
                journal=journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            )
            action = queue_trade_action_from_order_payload(result, journal=journal)
            review_op = build_action_operation(action["callback_data"]["review"], journal=journal)
            self.assertTrue(review_op["success"])
            self.assertEqual(review_op["operation"], "tradability_then_review_equity_order")
            self.assertEqual(review_op["review_mcp_args"]["symbol"], "TEST")
            self.assertEqual(review_op["tradability_mcp_args"]["symbols"], ["TEST"])
            skip_op = build_action_operation(action["callback_data"]["skip"], journal=journal)
            self.assertTrue(skip_op["success"])
            self.assertEqual(journal.get_trade_action(action["action_id"])["status"], "skipped")
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
                Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            ) = old_config

    def test_runtime_kill_switch_blocks_place_operation(self):
        from datetime import datetime, timezone

        from artha.config import Config
        from artha.execution import build_order_intent, prepare_and_record_robinhood_review
        from artha.robinhood_bridge import build_action_operation, queue_trade_action_from_order_payload, set_trading_disabled

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            Config.ROBINHOOD_REVIEW_ONLY,
            Config.ROBINHOOD_DRY_RUN_ONLY,
            Config.ROBINHOOD_AGENTIC_ENABLED,
            Config.ROBINHOOD_KILL_SWITCH,
            Config.ROBINHOOD_CONTROL_FILE,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            Config.ROBINHOOD_REVIEW_ONLY = False
            Config.ROBINHOOD_DRY_RUN_ONLY = False
            Config.ROBINHOOD_AGENTIC_ENABLED = True
            Config.ROBINHOOD_KILL_SWITCH = False
            Config.ROBINHOOD_CONTROL_FILE = str(self.tmp_dir / "control.json")
            set_trading_disabled(True, "unit test stop")
            journal = self._journal_with_clean_buy_decision()
            account = {
                "account_number": "222233334",
                "type": "cash",
                "nickname": "Agentic",
                "agentic_allowed": True,
                "state": "active",
            }
            intent = build_order_intent("TEST", "buy", quantity=1, limit_price=10, estimated_price=10)
            result = prepare_and_record_robinhood_review(
                intent,
                account,
                market_data={"price": 10, "volume": 2_000_000, "bid": 9.99, "ask": 10.01},
                journal=journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            )
            action = queue_trade_action_from_order_payload(result, journal=journal)
            place_op = build_action_operation(action["callback_data"]["place"], journal=journal)
            self.assertFalse(place_op["success"])
            self.assertIn("disabled", place_op["message"].lower())
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
                Config.ROBINHOOD_REVIEW_ONLY,
                Config.ROBINHOOD_DRY_RUN_ONLY,
                Config.ROBINHOOD_AGENTIC_ENABLED,
                Config.ROBINHOOD_KILL_SWITCH,
                Config.ROBINHOOD_CONTROL_FILE,
            ) = old_config

    def test_place_operation_rechecks_latest_robinhood_snapshot(self):
        from datetime import datetime, timezone

        from artha.config import Config
        from artha.execution import build_order_intent, prepare_and_record_robinhood_review
        from artha.robinhood_bridge import build_action_operation, queue_trade_action_from_order_payload, record_action_review, write_robinhood_snapshot

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            Config.ROBINHOOD_REVIEW_ONLY,
            Config.ROBINHOOD_DRY_RUN_ONLY,
            Config.ROBINHOOD_AGENTIC_ENABLED,
            Config.ROBINHOOD_KILL_SWITCH,
            Config.ROBINHOOD_CONTROL_FILE,
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            Config.ROBINHOOD_REVIEW_ONLY = False
            Config.ROBINHOOD_DRY_RUN_ONLY = False
            Config.ROBINHOOD_AGENTIC_ENABLED = True
            Config.ROBINHOOD_KILL_SWITCH = False
            Config.ROBINHOOD_CONTROL_FILE = str(self.tmp_dir / "control.json")
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE = str(self.tmp_dir / "robinhood_snapshot.json")
            write_robinhood_snapshot(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "robinhood_mcp",
                    "account": {
                        "account_number": "222233334",
                        "type": "cash",
                        "nickname": "Agentic",
                        "agentic_allowed": True,
                        "state": "active",
                    },
                    "portfolio": {"buying_power": 10, "cash": 10, "total_value": 10},
                    "positions": [],
                    "orders": [],
                },
                path=Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            )
            journal = self._journal_with_clean_buy_decision()
            account = {
                "account_number": "222233334",
                "type": "cash",
                "nickname": "Agentic",
                "agentic_allowed": True,
                "state": "active",
            }
            intent = build_order_intent("TEST", "buy", notional=25, quantity=2.5, limit_price=10, estimated_price=10)
            result = prepare_and_record_robinhood_review(
                intent,
                account,
                market_data={"price": 10, "volume": 2_000_000, "bid": 9.99, "ask": 10.01},
                journal=journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            )
            action = queue_trade_action_from_order_payload(result, journal=journal)
            review_result = record_action_review(
                action["action_id"],
                {
                    "data": {
                        "symbol": "TEST",
                        "side": "buy",
                        "type": "market",
                        "dollar_amount": "25.00",
                        "order_checks": {},
                        "market_data_disclosure": "Bid $9.99 x 1 P · Ask $10.01 x 1 M · Last $10.00 x 1. Updated 10:00 AM ET.",
                    }
                },
                tradability_response={
                    "data": {
                        "results": [
                            {
                                "symbol": "TEST",
                                "tradeable": True,
                                "state": "active",
                                "fractional_tradability": "tradable",
                            }
                        ]
                    }
                },
                journal=journal,
            )
            self.assertEqual(review_result["status"], "review_clear")
            place_op = build_action_operation(action["callback_data"]["place"], journal=journal)
            self.assertFalse(place_op["success"])
            self.assertEqual(place_op["operation"], "blocked")
            self.assertIn("Broker snapshot guardrails blocked", place_op["message"])
            self.assertTrue(
                any(
                    "cash" in reason.lower() or "buying power" in reason.lower()
                    for reason in place_op["broker_snapshot_guardrails"]["reasons"]
                )
            )
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
                Config.ROBINHOOD_REVIEW_ONLY,
                Config.ROBINHOOD_DRY_RUN_ONLY,
                Config.ROBINHOOD_AGENTIC_ENABLED,
                Config.ROBINHOOD_KILL_SWITCH,
                Config.ROBINHOOD_CONTROL_FILE,
                Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            ) = old_config

    def test_review_recording_creates_confirmation_button_and_missing_review_blocks_place(self):
        from datetime import datetime, timezone

        from artha.config import Config
        from artha.execution import build_order_intent, prepare_and_record_robinhood_review
        from artha.robinhood_bridge import (
            build_action_operation,
            queue_trade_action_from_order_payload,
            record_action_review,
            set_trading_disabled,
            write_robinhood_snapshot,
        )

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            Config.ROBINHOOD_REVIEW_ONLY,
            Config.ROBINHOOD_DRY_RUN_ONLY,
            Config.ROBINHOOD_AGENTIC_ENABLED,
            Config.ROBINHOOD_KILL_SWITCH,
            Config.ROBINHOOD_CONTROL_FILE,
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            Config.ROBINHOOD_REVIEW_ONLY = False
            Config.ROBINHOOD_DRY_RUN_ONLY = False
            Config.ROBINHOOD_AGENTIC_ENABLED = True
            Config.ROBINHOOD_KILL_SWITCH = False
            Config.ROBINHOOD_CONTROL_FILE = str(self.tmp_dir / "control.json")
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE = str(self.tmp_dir / "robinhood_snapshot.json")
            set_trading_disabled(False, "unit test")
            write_robinhood_snapshot(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "robinhood_mcp",
                    "account": {
                        "account_number": "222233334",
                        "type": "cash",
                        "nickname": "Agentic",
                        "agentic_allowed": True,
                        "state": "active",
                    },
                    "portfolio": {"buying_power": 100, "cash": 100, "total_value": 100},
                    "positions": [],
                    "orders": [],
                },
                path=Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            )
            journal = self._journal_with_clean_buy_decision()
            account = {
                "account_number": "222233334",
                "type": "cash",
                "nickname": "Agentic",
                "agentic_allowed": True,
                "state": "active",
            }
            intent = build_order_intent("TEST", "buy", notional=25, quantity=0.25, limit_price=100, estimated_price=100)
            result = prepare_and_record_robinhood_review(
                intent,
                account,
                market_data=self._market_data(),
                journal=journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            )
            action = queue_trade_action_from_order_payload(result, journal=journal)
            initial_markup = json.dumps(action["reply_markup"])
            self.assertIn("artha:review:", initial_markup)
            self.assertNotIn("artha:place:", initial_markup)

            no_review = build_action_operation(action["callback_data"]["place"], journal=journal)
            self.assertFalse(no_review["success"])
            self.assertIn("review", no_review["message"].lower())

            review_result = record_action_review(
                action["action_id"],
                {
                    "data": {
                        "symbol": "TEST",
                        "side": "buy",
                        "type": "market",
                        "dollar_amount": "25",
                        "order_checks": {},
                        "market_data_disclosure": "Bid $99.95 x 1 P · Ask $100.05 x 1 M · Last $100.00 x 1. Updated 10:00 AM ET.",
                    }
                },
                tradability_response={
                    "data": {
                        "results": [
                            {
                                "symbol": "TEST",
                                "tradeable": True,
                                "state": "active",
                                "fractional_tradability": "tradable",
                            }
                        ]
                    }
                },
                journal=journal,
            )
            self.assertEqual(review_result["status"], "review_clear")
            confirmation_markup = json.dumps(review_result["reply_markup"])
            self.assertIn("artha:place:", confirmation_markup)
            self.assertNotIn("artha:review:", confirmation_markup)
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
                Config.ROBINHOOD_REVIEW_ONLY,
                Config.ROBINHOOD_DRY_RUN_ONLY,
                Config.ROBINHOOD_AGENTIC_ENABLED,
                Config.ROBINHOOD_KILL_SWITCH,
                Config.ROBINHOOD_CONTROL_FILE,
                Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            ) = old_config

    def test_equity_suitability_order_check_is_logged_but_not_blocking(self):
        from datetime import datetime, timezone

        from artha.config import Config
        from artha.execution import build_order_intent, prepare_and_record_robinhood_review
        from artha.robinhood_bridge import (
            classify_robinhood_order_checks,
            queue_trade_action_from_order_payload,
            record_action_review,
            set_trading_disabled,
            write_robinhood_snapshot,
        )

        classification = classify_robinhood_order_checks(
            {
                "alertType": "EQUITY_SUITABILITY",
                "equitySuitabilityAlertDetails": {"brokerageAccountType": "INDIVIDUAL"},
            }
        )
        self.assertFalse(classification["blocking"])
        self.assertTrue(classification["non_blocking"])

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            Config.ROBINHOOD_REVIEW_ONLY,
            Config.ROBINHOOD_DRY_RUN_ONLY,
            Config.ROBINHOOD_AGENTIC_ENABLED,
            Config.ROBINHOOD_KILL_SWITCH,
            Config.ROBINHOOD_CONTROL_FILE,
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            Config.ROBINHOOD_REVIEW_ONLY = False
            Config.ROBINHOOD_DRY_RUN_ONLY = False
            Config.ROBINHOOD_AGENTIC_ENABLED = True
            Config.ROBINHOOD_KILL_SWITCH = False
            Config.ROBINHOOD_CONTROL_FILE = str(self.tmp_dir / "control.json")
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE = str(self.tmp_dir / "robinhood_snapshot.json")
            set_trading_disabled(False, "unit test")
            write_robinhood_snapshot(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "robinhood_mcp",
                    "account": {
                        "account_number": "222233334",
                        "type": "cash",
                        "nickname": "Agentic",
                        "agentic_allowed": True,
                        "state": "active",
                    },
                    "portfolio": {"buying_power": 100, "cash": 100, "total_value": 100},
                    "positions": [],
                    "orders": [],
                },
                path=Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            )
            journal = self._journal_with_clean_buy_decision()
            account = {
                "account_number": "222233334",
                "type": "cash",
                "nickname": "Agentic",
                "agentic_allowed": True,
                "state": "active",
            }
            intent = build_order_intent("TEST", "buy", quantity=1, limit_price=10.25, estimated_price=10)
            result = prepare_and_record_robinhood_review(
                intent,
                account,
                market_data={"price": 10, "volume": 2_000_000, "bid": 9.99, "ask": 10.01},
                journal=journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            )
            action = queue_trade_action_from_order_payload(result, journal=journal)
            review_result = record_action_review(
                action["action_id"],
                {
                    "data": {
                        "symbol": "TEST",
                        "side": "buy",
                        "type": "limit",
                        "quantity": "1",
                        "limit_price": "10.25",
                        "order_checks": {
                            "alertType": "EQUITY_SUITABILITY",
                            "equitySuitabilityAlertDetails": {"brokerageAccountType": "INDIVIDUAL"},
                        },
                        "market_data_disclosure": "Bid $9.99 x 1 P · Ask $10.01 x 1 M · Last $10.00 x 1. Updated 10:00 AM ET.",
                    }
                },
                tradability_response={
                    "data": {
                        "results": [
                            {
                                "symbol": "TEST",
                                "tradeable": True,
                                "state": "active",
                                "fractional_tradability": "tradable",
                            }
                        ]
                    }
                },
                journal=journal,
            )
            self.assertEqual(review_result["status"], "review_clear")
            self.assertTrue(review_result["review_gate"]["checks"]["order_checks_classification"]["non_blocking"])
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
                Config.ROBINHOOD_REVIEW_ONLY,
                Config.ROBINHOOD_DRY_RUN_ONLY,
                Config.ROBINHOOD_AGENTIC_ENABLED,
                Config.ROBINHOOD_KILL_SWITCH,
                Config.ROBINHOOD_CONTROL_FILE,
                Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            ) = old_config

    def test_record_action_review_blocks_fractional_market_quote_drift(self):
        from datetime import datetime, timezone

        from artha.config import Config
        from artha.execution import build_order_intent, prepare_and_record_robinhood_review
        from artha.robinhood_bridge import queue_trade_action_from_order_payload, record_action_review

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            Config.ROBINHOOD_MARKET_ORDER_MAX_PRICE_DRIFT_PCT,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            Config.ROBINHOOD_MARKET_ORDER_MAX_PRICE_DRIFT_PCT = 0.005
            journal = self._journal_with_clean_buy_decision()
            account = {
                "account_number": "222233334",
                "type": "cash",
                "nickname": "Agentic",
                "agentic_allowed": True,
                "state": "active",
            }
            intent = build_order_intent("TEST", "buy", notional=25, quantity=0.25, limit_price=100, estimated_price=100)
            result = prepare_and_record_robinhood_review(
                intent,
                account,
                market_data=self._market_data(),
                journal=journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            )
            action = queue_trade_action_from_order_payload(result, journal=journal)
            review_result = record_action_review(
                action["action_id"],
                {
                    "data": {
                        "symbol": "TEST",
                        "side": "buy",
                        "type": "market",
                        "dollar_amount": "25.00",
                        "order_checks": {},
                        "market_data_disclosure": "Bid $100.90 x 1 P · Ask $101.00 x 1 M · Last $100.95 x 1. Updated 10:00 AM ET.",
                    }
                },
                tradability_response={
                    "data": {
                        "results": [
                            {
                                "symbol": "TEST",
                                "tradeable": True,
                                "state": "active",
                                "fractional_tradability": "tradable",
                            }
                        ]
                    }
                },
                journal=journal,
            )
            self.assertEqual(review_result["status"], "review_blocked")
            self.assertIn("above Artha reference", " ".join(review_result["review_gate"]["reasons"]))
            self.assertNotIn("artha:place:", json.dumps(review_result.get("reply_markup") or {}))
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
                Config.ROBINHOOD_MARKET_ORDER_MAX_PRICE_DRIFT_PCT,
            ) = old_config

    def test_review_operation_requires_fresh_robinhood_snapshot(self):
        from datetime import datetime, timezone

        from artha.config import Config
        from artha.execution import build_order_intent, prepare_and_record_robinhood_review
        from artha.robinhood_bridge import build_action_operation, queue_trade_action_from_order_payload

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            Config.ROBINHOOD_REQUIRE_FRESH_SNAPSHOT_FOR_REVIEW,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE = str(self.tmp_dir / "missing_robinhood_snapshot.json")
            Config.ROBINHOOD_REQUIRE_FRESH_SNAPSHOT_FOR_REVIEW = True
            journal = self._journal_with_clean_buy_decision()
            account = {
                "account_number": "222233334",
                "type": "cash",
                "nickname": "Agentic",
                "agentic_allowed": True,
                "state": "active",
            }
            intent = build_order_intent("TEST", "buy", quantity=1, limit_price=10, estimated_price=10)
            result = prepare_and_record_robinhood_review(
                intent,
                account,
                market_data={"price": 10, "volume": 2_000_000, "bid": 9.99, "ask": 10.01},
                journal=journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            )
            action = queue_trade_action_from_order_payload(result, journal=journal)
            review_op = build_action_operation(action["callback_data"]["review"], journal=journal)
            self.assertFalse(review_op["success"])
            self.assertIn("Fresh Robinhood snapshot is required before review", review_op["message"])
            self.assertEqual(review_op["snapshot_refresh_operation"]["operation"], "refresh_robinhood_snapshot")
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
                Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
                Config.ROBINHOOD_REQUIRE_FRESH_SNAPSHOT_FOR_REVIEW,
            ) = old_config

    def test_snapshot_refresh_operation_is_read_only_and_queue_pending_reviews(self):
        from datetime import datetime, timezone

        from artha.config import Config
        from artha.execution import build_order_intent, prepare_and_record_robinhood_review
        from artha.robinhood_bridge import (
            build_snapshot_refresh_operation,
            queue_review_actions_for_ready_orders,
            write_robinhood_snapshot,
        )

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            Config.ROBINHOOD_ACTION_TOKEN_TTL_MINUTES,
            Config.ROBINHOOD_REVIEW_DECISION_MAX_AGE_MINUTES,
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            Config.ROBINHOOD_AUTO_BUY_ENABLED,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            Config.ROBINHOOD_ACTION_TOKEN_TTL_MINUTES = 60
            Config.ROBINHOOD_REVIEW_DECISION_MAX_AGE_MINUTES = 60
            Config.ROBINHOOD_AUTO_BUY_ENABLED = True
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE = str(self.tmp_dir / "robinhood_snapshot.json")
            write_robinhood_snapshot(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "robinhood_mcp",
                    "account": {
                        "account_number": "222233334",
                        "type": "cash",
                        "nickname": "Agentic",
                        "agentic_allowed": True,
                        "state": "active",
                    },
                    "portfolio": {"buying_power": 100, "cash": 100, "total_value": 100},
                    "positions": [],
                    "orders": [],
                },
                path=Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            )
            op = build_snapshot_refresh_operation()
            tools = [step["tool"] for step in op["mcp_sequence"]]
            self.assertEqual(tools, ["get_accounts", "get_portfolio", "get_equity_positions", "get_equity_orders"])
            self.assertNotIn("place_equity_order", tools)
            self.assertIn("place_equity_order", op["forbidden_tools"])
            self.assertIn("--strict", op["import_command"])
            self.assertIn("--expect-run-id <RUN_ID>", op["import_command"])
            self.assertIn("run_id", op["handoff_required_fields"])

            journal = self._journal_with_clean_buy_decision()
            account = {
                "account_number": "222233334",
                "type": "cash",
                "nickname": "Agentic",
                "agentic_allowed": True,
                "state": "active",
            }
            intent = build_order_intent("TEST", "buy", quantity=1, limit_price=10, estimated_price=10)
            intent.evidence = {"execution_officer": {"auto_buy_eligible": True}}
            prepare_and_record_robinhood_review(
                intent,
                account,
                market_data={"price": 10, "volume": 2_000_000, "bid": 9.99, "ask": 10.01},
                journal=journal,
                now=datetime.now(timezone.utc),
            )
            result = queue_review_actions_for_ready_orders(journal=journal)
            self.assertEqual(result["created_count"], 1)
            row = journal.get_trade_action(result["created"][0]["action_id"])
            self.assertEqual(row["status"], "review_ready")
            self.assertEqual(row["action_type"], "auto_buy")
            self.assertIsNone(result["created"][0].get("reply_markup"))
            self.assertEqual(result["created"][0].get("callback_data"), {})
            created = datetime.fromisoformat(row["created_at"])
            expires = datetime.fromisoformat(row["expires_at"])
            self.assertLessEqual((expires - created).total_seconds(), 60 * 60 + 5)

            second = queue_review_actions_for_ready_orders(journal=journal)
            self.assertEqual(second["created_count"], 0)
            self.assertGreaterEqual(second["skipped_count"], 1)
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
                Config.ROBINHOOD_ACTION_TOKEN_TTL_MINUTES,
                Config.ROBINHOOD_REVIEW_DECISION_MAX_AGE_MINUTES,
                Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
                Config.ROBINHOOD_AUTO_BUY_ENABLED,
            ) = old_config

    def test_auto_buy_runner_operation_is_durable_and_agentic(self):
        from artha.config import Config
        from artha.robinhood_bridge import build_auto_buy_runner_operation

        old_account = Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            op = build_auto_buy_runner_operation()
            self.assertTrue(op["success"])
            self.assertEqual(op["operation"], "openclaw_auto_buy_runner")
            self.assertEqual(op["cron"]["expr"], "*/2 8-14 * * 1-5")
            self.assertEqual(op["cron"]["thinking"], "xhigh")
            self.assertIn("place_equity_order", op["mcp_tools_required"])
            self.assertIn("robinhood-auto-buy-queue-status", op["artha_commands_required"])
            self.assertIn("robinhood-auto-buy-agentic-clearance", op["artha_commands_required"])
            self.assertIn("robinhood-final-clearance", op["artha_commands_required"])
            self.assertIn("cancel_equity_order", op["forbidden_tools"])
            message = op["runner_message"]
            self.assertIn("Cheap queue preflight", message)
            self.assertIn("robinhood-auto-buy-queue-status", message)
            self.assertIn("get_equity_quotes", message)
            self.assertIn("review_equity_order", message)
            self.assertIn("robinhood-auto-buy-agentic-clearance", message)
            self.assertIn("robinhood-final-clearance", message)
            self.assertIn("robinhood-record-submission", message)
            self.assertIn("AUTO_BUY_IDLE", message)
            self.assertIn("AUTO_BUY_PLACED", message)
            self.assertNotIn("heredoc", message.lower().replace("shell heredocs", ""))
            bootstrap = op["bootstrap_message"]
            self.assertIn("robinhood-auto-buy-queue-status", bootstrap)
            self.assertIn("operation_count is 0", bootstrap)
            self.assertIn("AUTO_BUY_IDLE", bootstrap)
            self.assertIn("robinhood-auto-buy-runner-operation --message-only", bootstrap)
            self.assertIn("<bootstrap_message>", op["install_hint"])
        finally:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = old_account

    def test_robinhood_operations_never_guess_an_account(self):
        from artha.config import Config
        from artha.robinhood_bridge import (
            build_auto_buy_runner_operation,
            build_snapshot_refresh_operation,
        )

        old_account = Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = ""
            snapshot = build_snapshot_refresh_operation()
            runner = build_auto_buy_runner_operation()
            self.assertFalse(snapshot["success"])
            self.assertFalse(runner["success"])
            self.assertIn("never guessed", snapshot["reason"])
            self.assertIn("never guessed", runner["reason"])
        finally:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = old_account

    def test_snapshot_handoff_validation_rejects_stale_tmp_file(self):
        from artha.robinhood_bridge import SnapshotHandoffValidationError, validate_snapshot_handoff_metadata

        with self.assertRaises(SnapshotHandoffValidationError) as ctx:
            validate_snapshot_handoff_metadata(
                {"run_id": "old-run", "generated_at": "2026-06-09T14:00:00+00:00"},
                expected_run_id="new-run",
                min_generated_at="2026-06-09T14:30:00+00:00",
                now=datetime(2026, 6, 9, 14, 31, tzinfo=timezone.utc),
            )
        validation = ctx.exception.validation
        self.assertEqual(validation["status"], "FAIL")
        self.assertTrue(any(check["name"] == "run_id" and not check["passed"] for check in validation["checks"]))
        self.assertTrue(any(check["name"] == "min_generated_at" and not check["passed"] for check in validation["checks"]))

    def test_snapshot_import_rejects_stale_handoff_before_write(self):
        import io

        from run import robinhood_snapshot_import

        handoff = self.tmp_dir / "stale_handoff.json"
        handoff.write_text(json.dumps({"accounts_response": {}, "positions_response": {}}), encoding="utf-8")
        with patch("artha.robinhood_bridge.write_robinhood_snapshot") as write_snapshot, patch(
            "sys.stdout",
            new_callable=io.StringIO,
        ) as stdout:
            with self.assertRaises(SystemExit) as ctx:
                robinhood_snapshot_import(
                    [
                        "--file",
                        str(handoff),
                        "--strict",
                        "--expect-run-id",
                        "current-run",
                        "--min-generated-at",
                        "2026-06-09T14:30:00+00:00",
                    ]
                )
        self.assertEqual(ctx.exception.code, 1)
        write_snapshot.assert_not_called()
        result = json.loads(stdout.getvalue())
        self.assertFalse(result["success"])
        self.assertEqual(result["validation"]["status"], "FAIL")

    def test_snapshot_import_strict_mode_fails_warn_sync(self):
        import io

        from run import robinhood_snapshot_import

        handoff = self.tmp_dir / "handoff.json"
        payload = {
            "run_id": "current-run",
            "generated_at": "2026-06-09T14:30:00+00:00",
            "accounts_response": {},
            "portfolio_response": {},
            "positions_response": {},
            "orders_response": {},
        }
        handoff.write_text(json.dumps(payload), encoding="utf-8")
        with patch(
            "artha.robinhood_bridge.write_robinhood_snapshot",
            return_value={"status": "PASS", "path": "unit", "position_count": 0, "warnings": [], "snapshot": payload},
        ), patch(
            "artha.robinhood_bridge.sync_snapshot_to_artha",
            return_value={"status": "WARN", "unresolved": [{"ticker": "JNJ"}]},
        ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            with self.assertRaises(SystemExit) as ctx:
                robinhood_snapshot_import(["--file", str(handoff), "--strict"])
        self.assertEqual(ctx.exception.code, 1)
        result = json.loads(stdout.getvalue())
        self.assertFalse(result["success"])
        self.assertEqual(result["snapshot"]["status"], "PASS")
        self.assertEqual(result["sync"]["status"], "WARN")

    def test_snapshot_runner_reports_compact_strict_import_failure_reason(self):
        import subprocess

        script = Path(__file__).resolve().parent.parent / "scripts" / "robinhood_snapshot_sync.mjs"
        js = f"""
            import {{ summarizeImporterFailure }} from {json.dumps(script.as_uri())};
            const parsed = {{
              success: false,
              validation: {{status: 'PASS'}},
              snapshot: {{status: 'PASS'}},
              sync: {{
                status: 'WARN',
                unresolved: [{{ticker: 'USB', reason: 'Held in Robinhood but no active/pending Artha thesis.'}}],
                order_sync: {{status: 'PASS'}}
              }}
            }};
            console.log(summarizeImporterFailure({{code: 1, stdout: '', stderr: ''}}, parsed));
        """
        proc = subprocess.run(
            ["node", "--input-type=module", "-e", js],
            cwd=script.parent.parent,
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("validation=PASS", proc.stdout)
        self.assertIn("sync=WARN", proc.stdout)
        self.assertIn("USB: Held in Robinhood", proc.stdout)
        self.assertNotIn('"checks"', proc.stdout)

    def test_canonical_snapshot_preserves_handoff_metadata(self):
        from artha.robinhood_bridge import canonicalize_mcp_snapshot

        snapshot = canonicalize_mcp_snapshot(
            {
                "run_id": "current-run",
                "generated_at": "2026-06-09T14:30:00+00:00",
                "accounts_response": {"data": {"accounts": [{"account_number": "222233334", "agentic_allowed": True}]}},
                "portfolio_response": {"data": {"portfolio": {"buying_power": "10"}}},
                "positions_response": {"data": {"positions": []}},
                "orders_response": {"data": {"orders": []}},
            }
        )
        self.assertEqual(snapshot["run_id"], "current-run")
        self.assertEqual(snapshot["generated_at"], "2026-06-09T14:30:00+00:00")
        self.assertEqual(snapshot["portfolio"]["buying_power"], "10")

    def test_canonical_snapshot_unwraps_direct_mcp_envelope_collections(self):
        from artha.robinhood_bridge import canonicalize_mcp_snapshot

        snapshot = canonicalize_mcp_snapshot(
            {
                "generated_at": "2026-06-16T17:23:04+00:00",
                "source": "robinhood_mcp",
                "selected_account": {"account_number": "222233334", "agentic_allowed": True},
                "accounts": {"data": {"accounts": [{"account_number": "222233334", "agentic_allowed": True}]}},
                "portfolio": {"data": {"buying_power": {"buying_power": "332.50"}}},
                "positions": {"data": {"positions": [{"symbol": "JNJ", "quantity": "0.074763"}]}},
                "orders": {"data": {"orders": [{"symbol": "JNJ", "state": "filled"}]}},
            }
        )
        self.assertEqual(snapshot["account"]["account_number"], "222233334")
        self.assertEqual(snapshot["accounts"][0]["account_number"], "222233334")
        self.assertEqual(snapshot["portfolio"]["buying_power"]["buying_power"], "332.50")
        self.assertEqual(snapshot["positions"][0]["symbol"], "JNJ")
        self.assertEqual(snapshot["orders"][0]["state"], "filled")

    def _external_snapshot(self, positions, cash=75, orders=None):
        from datetime import datetime, timezone
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "robinhood_mcp",
            "account": {
                "account_number": "222233334",
                "type": "cash",
                "nickname": "Agentic",
                "agentic_allowed": True,
                "state": "active",
            },
            "portfolio": {"buying_power": cash, "cash": cash, "total_value": cash},
            "positions": positions,
            "orders": orders or [],
        }

    def test_transfer_detection_books_deposit_and_ignores_external_sell_proceeds(self):
        import artha.robinhood_bridge as bridge
        from artha.config import Config
        from artha.journal import DecisionJournal

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        old_file = bridge.CONTRIBUTIONS_FILE
        old_auto_book = Config.ROBINHOOD_AUTO_BOOK_CASH_TRANSFERS
        sent = []
        try:
            bridge.CONTRIBUTIONS_FILE = self.tmp_dir / "contributions.jsonl"
            Config.ROBINHOOD_AUTO_BOOK_CASH_TRANSFERS = False

            class FakeSender:
                def send_message(self, text, parse_mode=None, silent=False):
                    sent.append(text)
                    return True

            with patch("artha.telegram.TelegramSender", FakeSender):
                # Case 1: $150 cash jump with no explaining orders is detected,
                # but not booked/alerted unless explicit auto-booking is enabled.
                result = bridge._detect_cash_transfer(350.0, 500.0, {"orders": []}, journal)
                self.assertIsNotNone(result)
                self.assertEqual(result["direction"], "deposit")
                self.assertAlmostEqual(result["amount"], 150.0)
                self.assertFalse(result["booked"])
                self.assertFalse(bridge.CONTRIBUTIONS_FILE.exists())
                self.assertFalse(any("New money detected" in t for t in sent))

                Config.ROBINHOOD_AUTO_BOOK_CASH_TRANSFERS = True
                result_booked = bridge._detect_cash_transfer(350.0, 500.0, {"orders": []}, journal)
                self.assertIsNotNone(result_booked)
                self.assertTrue(result_booked["booked"])
                ledger = bridge.CONTRIBUTIONS_FILE.read_text(encoding="utf-8").strip().splitlines()
                self.assertEqual(len(ledger), 1)
                self.assertIn("auto_detected_transfer", ledger[0])
                self.assertTrue(any("New money detected" in t for t in sent))

                # Case 2: cash jump fully explained by an external manual SELL
                # (not in Artha's journal) → NOT a deposit.
                manual_sell = {
                    "id": "manual-1",
                    "state": "filled",
                    "side": "sell",
                    "cumulative_quantity": "2.0",
                    "average_price": "75.00",
                }
                result2 = bridge._detect_cash_transfer(350.0, 500.0, {"orders": [manual_sell]}, journal)
                self.assertIsNone(result2)

                # Case 3: cash drop fully explained by a fresh Artha-known BUY
                # → NOT a withdrawal.
                now = datetime.now(timezone.utc)
                journal.save_execution_order(
                    {
                        "order_intent_id": "intent-known-buy",
                        "ticker": "NVDA",
                        "side": "buy",
                        "order_type": "market",
                        "time_in_force": "day",
                        "quantity": 0.088228,
                        "notional": 17.849945,
                        "limit_price": 202.09,
                        "estimated_price": 202.3161,
                        "status": "filled",
                        "broker": "robinhood",
                        "broker_order_id": "known-buy-1",
                        "dry_run": False,
                    }
                )
                known_buy = {
                    "id": "known-buy-1",
                    "state": "filled",
                    "side": "buy",
                    "cumulative_quantity": "0.088228",
                    "average_price": "202.316100",
                    "last_transaction_at": now.isoformat(),
                }
                result3 = bridge._detect_cash_transfer(
                    258.35,
                    240.50,
                    {"generated_at": now.isoformat(), "orders": [known_buy]},
                    journal,
                    since=now - timedelta(minutes=10),
                )
                self.assertIsNone(result3)
                ledger = bridge.CONTRIBUTIONS_FILE.read_text(encoding="utf-8").strip().splitlines()
                self.assertEqual(len(ledger), 1)
        finally:
            bridge.CONTRIBUTIONS_FILE = old_file
            Config.ROBINHOOD_AUTO_BOOK_CASH_TRANSFERS = old_auto_book

    def test_external_sell_autocloses_position_after_three_confirmations(self):
        import artha.robinhood_bridge as bridge
        from artha.config import Config
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio
        from artha.thesis_tracker import ThesisTracker

        old_account = Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER
        old_tracker_file = bridge.EXTERNAL_SELL_TRACKER_FILE
        old_contrib = bridge.CONTRIBUTIONS_FILE
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            bridge.EXTERNAL_SELL_TRACKER_FILE = self.tmp_dir / "external_sell_tracker.json"
            bridge.CONTRIBUTIONS_FILE = self.tmp_dir / "contributions.jsonl"
            journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
            tracker = ThesisTracker(journal)
            portfolio_path = self.tmp_dir / "portfolio.json"

            # Seed: KEEP stays at the broker, GONE was sold manually.
            seeded_theses = {}
            for tick in ("KEEP", "GONE"):
                seeded_theses[tick] = tracker.create_thesis(
                    ticker=tick,
                    position_type="STARTER",
                    thesis_summary=f"Unit thesis {tick}.",
                    invalidation_conditions=["Unit invalidation"],
                )
            seed = self._external_snapshot(
                [
                    {"symbol": "KEEP", "quantity": "1.0", "average_buy_price": "10.00", "market_price": "10.00"},
                    {"symbol": "GONE", "quantity": "2.0", "average_buy_price": "20.00", "market_price": "20.00"},
                ]
            )
            bridge.sync_snapshot_to_artha(seed, journal=journal, portfolio_path=portfolio_path)
            self.assertIsNotNone(Portfolio.load(portfolio_path).get_position("GONE"))
            tracker.mark_waiting_for_sell(
                seeded_theses["GONE"].thesis_id,
                reason="Council EXIT awaiting broker confirmation",
            )
            self.assertEqual(
                tracker.get(seeded_theses["GONE"].thesis_id).status,
                "pending_exit",
            )

            sent = []

            class FakeSender:
                def send_message(self, text, parse_mode=None, silent=False):
                    sent.append(text)
                    return True

            gone_absent = self._external_snapshot(
                [{"symbol": "KEEP", "quantity": "1.0", "average_buy_price": "10.00", "market_price": "10.00"}],
                cash=115,  # proceeds of the manual GONE sale arrived in cash
            )
            with patch("artha.telegram.TelegramSender", FakeSender):
                for i in range(3):
                    result = bridge.sync_snapshot_to_artha(gone_absent, journal=journal, portfolio_path=portfolio_path)
                    still_there = Portfolio.load(portfolio_path).get_position("GONE")
                    if i < 2:
                        self.assertIsNotNone(still_there, f"GONE removed too early on pass {i+1}")
                        self.assertEqual(result["externally_closed"], [])
                    else:
                        self.assertIsNone(still_there, "GONE should be auto-closed on 3rd confirmation")
                        self.assertEqual(result["externally_closed"][0]["ticker"], "GONE")

            portfolio = Portfolio.load(portfolio_path)
            self.assertIsNotNone(portfolio.get_position("KEEP"))
            sells = [t for t in portfolio.transactions if t.get("type") == "SELL" and t.get("ticker") == "GONE"]
            self.assertEqual(len(sells), 1)
            self.assertIn("Sold outside Artha", sells[0]["notes"])
            self.assertTrue(any("Sold outside Artha" in t for t in sent))
            gone_thesis = tracker.get(seeded_theses["GONE"].thesis_id)
            self.assertEqual(gone_thesis.status, "archived")
            self.assertEqual(len(journal.get_pending_post_sell_reviews()), 1)
            self.assertEqual(len(journal.get_sell_trade_events(ticker="GONE")), 1)
            self.assertEqual(len(journal.get_broker_fill_effects()), 1)
            # tracker resets after close
            self.assertEqual(bridge._load_external_sell_tracker(), {})
        finally:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = old_account
            bridge.EXTERNAL_SELL_TRACKER_FILE = old_tracker_file
            bridge.CONTRIBUTIONS_FILE = old_contrib

    def test_external_sell_does_not_autoclose_artha_managed_sell(self):
        import artha.robinhood_bridge as bridge
        from artha.config import Config
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio
        from artha.thesis_tracker import ThesisTracker

        old_account = Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER
        old_tracker_file = bridge.EXTERNAL_SELL_TRACKER_FILE
        old_contrib = bridge.CONTRIBUTIONS_FILE
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            bridge.EXTERNAL_SELL_TRACKER_FILE = self.tmp_dir / "external_sell_tracker.json"
            bridge.CONTRIBUTIONS_FILE = self.tmp_dir / "contributions.jsonl"
            journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
            tracker = ThesisTracker(journal)
            portfolio_path = self.tmp_dir / "portfolio.json"

            tracker.create_thesis(
                ticker="KEEP",
                position_type="STARTER",
                thesis_summary="Unit thesis KEEP.",
                invalidation_conditions=["Unit invalidation"],
            )
            managed = tracker.create_thesis(
                ticker="ARTHA",
                position_type="STARTER",
                thesis_summary="Unit thesis ARTHA.",
                invalidation_conditions=["Unit invalidation"],
            )
            seed = self._external_snapshot(
                [
                    {"symbol": "KEEP", "quantity": "1.0", "average_buy_price": "10.00", "market_price": "10.00"},
                    {"symbol": "ARTHA", "quantity": "2.0", "average_buy_price": "20.00", "market_price": "20.00"},
                ]
            )
            bridge.sync_snapshot_to_artha(seed, journal=journal, portfolio_path=portfolio_path)
            self.assertIsNotNone(Portfolio.load(portfolio_path).get_position("ARTHA"))

            journal.save_execution_order(
                {
                    "order_intent_id": "intent-managed-sell",
                    "ticker": "ARTHA",
                    "side": "sell",
                    "order_type": "market",
                    "time_in_force": "day",
                    "quantity": 2.0,
                    "notional": 42.0,
                    "estimated_price": 21.0,
                    "status": "submitted",
                    "broker": "robinhood",
                    "broker_order_id": "managed-sell-1",
                    "dry_run": False,
                    "thesis_id": managed.thesis_id,
                    "submitted_at": seed["generated_at"],
                }
            )

            sent = []

            class FakeSender:
                def send_message(self, text, parse_mode=None, silent=False):
                    sent.append(text)
                    return True

            managed_absent = self._external_snapshot(
                [{"symbol": "KEEP", "quantity": "1.0", "average_buy_price": "10.00", "market_price": "10.00"}],
                cash=115,
                orders=[
                    {
                        "id": "managed-sell-1",
                        "symbol": "ARTHA",
                        "side": "sell",
                        "state": "unconfirmed",
                        "quantity": "2.0",
                        "cumulative_quantity": "0.0",
                    }
                ],
            )
            with patch("artha.telegram.TelegramSender", FakeSender):
                for _ in range(3):
                    result = bridge.sync_snapshot_to_artha(managed_absent, journal=journal, portfolio_path=portfolio_path)
                    self.assertEqual(result["externally_closed"], [])
                    self.assertIsNotNone(Portfolio.load(portfolio_path).get_position("ARTHA"))

            self.assertEqual(bridge._load_external_sell_tracker(), {})
            self.assertFalse(any("Sold outside Artha" in t for t in sent))
        finally:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = old_account
            bridge.EXTERNAL_SELL_TRACKER_FILE = old_tracker_file
            bridge.CONTRIBUTIONS_FILE = old_contrib

    def test_snapshot_sync_activates_pending_thesis_and_portfolio(self):
        from artha.config import Config
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio
        from artha.robinhood_bridge import sync_snapshot_to_artha
        from artha.thesis_tracker import ThesisTracker

        old_account = Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
            tracker = ThesisTracker(journal)
            pending = tracker.create_thesis(
                ticker="TEST",
                position_type="STARTER",
                thesis_summary="Unit pending buy thesis.",
                invalidation_conditions=["Unit invalidation"],
            )
            portfolio_path = self.tmp_dir / "portfolio.json"
            snapshot = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": "robinhood_mcp",
                "account": {
                    "account_number": "222233334",
                    "type": "cash",
                    "nickname": "Agentic",
                    "agentic_allowed": True,
                    "state": "active",
                },
                "portfolio": {"buying_power": 75, "cash": 75, "total_value": 75},
                "positions": [
                    {
                        "symbol": "TEST",
                        "quantity": "1.25",
                        "average_buy_price": "10.00",
                        "market_price": "10.50",
                    }
                ],
                "orders": [],
            }
            result = sync_snapshot_to_artha(snapshot, journal=journal, portfolio_path=portfolio_path)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["activated"][0]["thesis_id"], pending.thesis_id)
            synced_portfolio = Portfolio.load(portfolio_path)
            self.assertAlmostEqual(synced_portfolio.cash_available, 75.0)
            pos = synced_portfolio.get_position("TEST")
            self.assertIsNotNone(pos)
            self.assertEqual(pos.thesis_id, pending.thesis_id)
            self.assertIsNotNone(tracker.get_active("TEST"))
        finally:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = old_account

    def test_snapshot_sync_marks_position_to_market_when_broker_price_missing(self):
        from artha.config import Config
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio
        from artha.robinhood_bridge import sync_snapshot_to_artha
        from artha.thesis_tracker import ThesisTracker

        class FakeFMPCollector:
            def quote(self, ticker):
                return {"symbol": ticker, "price": 9.50}

        old_account = Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
            tracker = ThesisTracker(journal)
            pending = tracker.create_thesis(
                ticker="TEST",
                position_type="STARTER",
                thesis_summary="Unit pending buy thesis.",
                invalidation_conditions=["Unit invalidation"],
            )
            portfolio_path = self.tmp_dir / "portfolio.json"
            snapshot = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": "robinhood_mcp",
                "account": {"account_number": "222233334", "agentic_allowed": True, "state": "active"},
                "portfolio": {"buying_power": 75, "cash": 75, "total_value": 75},
                "positions": [{"symbol": "TEST", "quantity": "1.25", "average_buy_price": "10.00"}],
                "orders": [],
            }
            with patch("artha.collector.FMPCollector", return_value=FakeFMPCollector()):
                result = sync_snapshot_to_artha(snapshot, journal=journal, portfolio_path=portfolio_path)

            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["activated"][0]["thesis_id"], pending.thesis_id)
            self.assertEqual(result["activated"][0]["price_source"], "fmp_quote")
            pos = Portfolio.load(portfolio_path).get_position("TEST")
            self.assertAlmostEqual(pos.current_price, 9.50)
            self.assertAlmostEqual(pos.market_value, 11.875)
        finally:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = old_account

    def test_record_order_fill_activates_pending_thesis(self):
        from datetime import datetime, timezone

        from artha.config import Config
        from artha.execution import build_order_intent, prepare_and_record_robinhood_review
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio
        from artha.robinhood_bridge import record_order_fill
        from artha.thesis_tracker import ThesisTracker

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
            journal.save_supervisor_run(
                {
                    "generated_at": "2026-06-04T14:30:00+00:00",
                    "severity": "PASS",
                    "payload": {"checks": []},
                }
            )
            journal.save_decision_features(
                {
                    "dossier_path": str(self.tmp_dir / "TEST_dossier.json"),
                    "generated_at": "2026-06-04T14:20:00+00:00",
                    "ticker": "TEST",
                    "final_verdict": "STARTER",
                    "opportunity_score": 68,
                    "confidence": 8,
                    "price": 10.0,
                    "sector": "Technology",
                    "industry": "Software - Infrastructure",
                    "evidence_count": 12,
                    "feature_json": "{}",
                }
            )
            pending = ThesisTracker(journal).create_thesis(
                "TEST",
                "STARTER",
                thesis_summary="Unit fill thesis",
                stop_loss_pct=-0.12,
            )
            account = {
                "account_number": "222233334",
                "type": "cash",
                "nickname": "Agentic",
                "agentic_allowed": True,
                "state": "active",
            }
            intent = build_order_intent(
                "TEST",
                "buy",
                quantity=1.25,
                limit_price=10,
                estimated_price=10,
                decision_dossier_path=str(self.tmp_dir / "TEST_dossier.json"),
            )
            intent.thesis_id = pending.thesis_id
            prepare_and_record_robinhood_review(
                intent,
                account,
                market_data={"price": 10, "volume": 2_000_000, "bid": 9.99, "ask": 10.01},
                journal=journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            )
            portfolio_path = self.tmp_dir / "portfolio.json"
            result = record_order_fill(
                order_intent_id=intent.order_intent_id,
                fill_payload={
                    "id": "rh-unit-order",
                    "symbol": "TEST",
                    "side": "buy",
                    "state": "filled",
                    "cumulative_quantity": "1.25",
                    "average_price": "10.00",
                },
                journal=journal,
                portfolio_path=portfolio_path,
            )
            self.assertEqual(result["status"], "PASS")
            portfolio_after_fill = Portfolio.load(portfolio_path)
            pos = portfolio_after_fill.get_position("TEST")
            self.assertEqual(pos.thesis_id, pending.thesis_id)
            self.assertEqual(pos.sector, "Technology")
            self.assertEqual(pos.industry, "Software - Infrastructure")
            self.assertTrue(pos.classification_source.startswith("decision_features:"))
            self.assertAlmostEqual(pos.shares, 1.25)
            self.assertAlmostEqual(pos.hard_stop_price, 9.2)  # rule-4.1 initial stop: -8% (was legacy -12%)
            buy_transactions = [
                txn for txn in portfolio_after_fill.transactions
                if txn.get("type") == "BUY" and txn.get("ticker") == "TEST"
            ]
            self.assertEqual(len(buy_transactions), 1)
            self.assertEqual(buy_transactions[0]["broker_order_id"], "rh-unit-order")
            self.assertAlmostEqual(float(buy_transactions[0]["total"]), 12.5)
            filled_row = journal.get_execution_order_by_intent_id(intent.order_intent_id)
            self.assertEqual(filled_row["status"], "filled")
            self.assertAlmostEqual(float(filled_row["quantity"]), 1.25)
            fill_audit = json.loads(filled_row["response_json"])
            self.assertTrue(fill_audit["artha_fill_applied"])
            second = record_order_fill(
                order_intent_id=intent.order_intent_id,
                fill_payload={
                    "id": "rh-unit-order",
                    "symbol": "TEST",
                    "side": "buy",
                    "state": "filled",
                    "cumulative_quantity": "1.25",
                    "average_price": "10.00",
                },
                journal=journal,
                portfolio_path=portfolio_path,
            )
            self.assertTrue(second.get("already_recorded"))
            portfolio_after_replay = Portfolio.load(portfolio_path)
            self.assertAlmostEqual(portfolio_after_replay.get_position("TEST").shares, 1.25)
            replay_buy_transactions = [
                txn for txn in portfolio_after_replay.transactions
                if txn.get("type") == "BUY" and txn.get("ticker") == "TEST"
            ]
            self.assertEqual(len(replay_buy_transactions), 1)
            self.assertEqual(journal.get_execution_order_by_intent_id(intent.order_intent_id)["status"], "filled")
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            ) = old_config

    def test_snapshot_sync_refreshes_portfolio_stop_from_active_thesis(self):
        from artha.config import Config
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio, Position
        from artha.robinhood_bridge import sync_snapshot_to_artha
        from artha.thesis_tracker import ThesisTracker

        old_account = Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
            tracker = ThesisTracker(journal)
            pending = tracker.create_thesis(
                ticker="TEST",
                position_type="STARTER",
                thesis_summary="Unit pending buy thesis.",
                stop_loss_pct=-0.12,
            )
            active = tracker.activate_thesis(pending.thesis_id, 10.0, shares=1.25)
            portfolio_path = self.tmp_dir / "portfolio.json"
            portfolio = Portfolio(
                positions=[
                    Position(
                        ticker="TEST",
                        asset_type="stock",
                        shares=1.25,
                        avg_cost=10.0,
                        opened_at=datetime.now(timezone.utc).isoformat(),
                        hard_stop_price=8.0,
                        thesis_id=active.thesis_id,
                        position_type="STARTER",
                    )
                ]
            )
            portfolio.save(portfolio_path)
            snapshot = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": "robinhood_mcp",
                "account": {
                    "account_number": "222233334",
                    "type": "cash",
                    "nickname": "Agentic",
                    "agentic_allowed": True,
                    "state": "active",
                },
                "portfolio": {"buying_power": 75, "cash": 75, "total_value": 75},
                "positions": [
                    {
                        "symbol": "TEST",
                        "quantity": "1.25",
                        "average_buy_price": "10.00",
                        "market_price": "10.50",
                    }
                ],
                "orders": [],
            }
            result = sync_snapshot_to_artha(snapshot, journal=journal, portfolio_path=portfolio_path)
            self.assertEqual(result["status"], "PASS")
            pos = Portfolio.load(portfolio_path).get_position("TEST")
            self.assertAlmostEqual(pos.hard_stop_price, 9.2)  # rule-4.1 initial stop: -8% (was legacy -12%)
        finally:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = old_account

    def test_fill_callback_does_not_double_count_snapshot_imported_position(self):
        from artha.config import Config
        from artha.execution import build_order_intent, prepare_and_record_robinhood_review
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio
        from artha.robinhood_bridge import record_order_fill, sync_snapshot_to_artha
        from artha.thesis_tracker import ThesisTracker

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
            journal.save_supervisor_run(
                {
                    "generated_at": "2026-06-04T14:30:00+00:00",
                    "severity": "PASS",
                    "payload": {"checks": []},
                }
            )
            journal.save_decision_features(
                {
                    "dossier_path": str(self.tmp_dir / "TEST_dossier.json"),
                    "generated_at": "2026-06-04T14:20:00+00:00",
                    "ticker": "TEST",
                    "final_verdict": "STARTER",
                    "opportunity_score": 68,
                    "confidence": 8,
                    "price": 10.0,
                    "evidence_count": 12,
                    "feature_json": "{}",
                }
            )
            pending = ThesisTracker(journal).create_thesis(
                "TEST",
                "STARTER",
                thesis_summary="Unit fill thesis",
                stop_loss_pct=-0.12,
            )
            account = {
                "account_number": "222233334",
                "type": "cash",
                "nickname": "Agentic",
                "agentic_allowed": True,
                "state": "active",
            }
            intent = build_order_intent(
                "TEST",
                "buy",
                quantity=1.25,
                limit_price=10,
                estimated_price=10,
                decision_dossier_path=str(self.tmp_dir / "TEST_dossier.json"),
            )
            intent.thesis_id = pending.thesis_id
            prepare_and_record_robinhood_review(
                intent,
                account,
                market_data={"price": 10, "volume": 2_000_000, "bid": 9.99, "ask": 10.01},
                journal=journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            )
            submitted_at = datetime.now(timezone.utc).isoformat()
            journal.update_execution_order(
                intent.order_intent_id,
                {
                    "status": "submitted",
                    "broker_order_id": "rh-race-order",
                    "submitted_at": submitted_at,
                    "dry_run": False,
                },
            )
            portfolio_path = self.tmp_dir / "portfolio.json"
            snapshot = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": "robinhood_mcp",
                "account": account,
                "portfolio": {"buying_power": 75, "cash": 75, "total_value": 75},
                "positions": [
                    {
                        "symbol": "TEST",
                        "quantity": "1.25",
                        "average_buy_price": "10.00",
                        "market_price": "10.00",
                    }
                ],
                "orders": [],
            }
            sync_snapshot_to_artha(snapshot, journal=journal, portfolio_path=portfolio_path)
            result = record_order_fill(
                order_intent_id=intent.order_intent_id,
                fill_payload={
                    "id": "rh-race-order",
                    "symbol": "TEST",
                    "side": "buy",
                    "state": "filled",
                    "cumulative_quantity": "1.25",
                    "average_price": "10.00",
                },
                journal=journal,
                portfolio_path=portfolio_path,
            )
            self.assertTrue(result.get("already_recorded"))
            pos = Portfolio.load(portfolio_path).get_position("TEST")
            self.assertAlmostEqual(pos.shares, 1.25)
            self.assertAlmostEqual(pos.hard_stop_price, 9.2)  # rule-4.1 initial stop: -8% (was legacy -12%)
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            ) = old_config

    def test_fill_callback_does_not_double_count_snapshot_imported_top_up(self):
        from artha.config import Config
        from artha.execution import build_order_intent, prepare_and_record_robinhood_review
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio
        from artha.robinhood_bridge import record_order_fill, sync_snapshot_to_artha
        from artha.thesis_tracker import ThesisTracker

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
            journal.save_supervisor_run(
                {
                    "generated_at": "2026-06-04T14:30:00+00:00",
                    "severity": "PASS",
                    "payload": {"checks": []},
                }
            )
            dossier_path = str(self.tmp_dir / "TEST_dossier.json")
            journal.save_decision_features(
                {
                    "dossier_path": dossier_path,
                    "generated_at": "2026-06-04T14:20:00+00:00",
                    "ticker": "TEST",
                    "final_verdict": "ACCUMULATE",
                    "opportunity_score": 74,
                    "confidence": 8,
                    "price": 10.0,
                    "evidence_count": 12,
                    "feature_json": "{}",
                }
            )
            tracker = ThesisTracker(journal)
            pending = tracker.create_thesis(
                "TEST",
                "STARTER",
                thesis_summary="Unit top-up thesis",
                stop_loss_pct=-0.12,
            )
            account = {
                "account_number": "222233334",
                "type": "cash",
                "nickname": "Agentic",
                "agentic_allowed": True,
                "state": "active",
            }
            portfolio_path = self.tmp_dir / "portfolio.json"
            sync_snapshot_to_artha(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "robinhood_mcp",
                    "account": account,
                    "portfolio": {"buying_power": 75, "cash": 75, "total_value": 75},
                    "positions": [
                        {
                            "symbol": "TEST",
                            "quantity": "1.00",
                            "average_buy_price": "9.00",
                            "market_price": "9.00",
                        }
                    ],
                    "orders": [],
                },
                journal=journal,
                portfolio_path=portfolio_path,
            )
            intent = build_order_intent(
                "TEST",
                "buy",
                quantity=0.25,
                limit_price=10,
                estimated_price=10,
                decision_dossier_path=dossier_path,
            )
            intent.thesis_id = pending.thesis_id
            prepare_and_record_robinhood_review(
                intent,
                account,
                market_data={"price": 10, "volume": 2_000_000, "bid": 9.99, "ask": 10.01},
                journal=journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            )
            submitted_at = datetime.now(timezone.utc).isoformat()
            journal.update_execution_order(
                intent.order_intent_id,
                {
                    "status": "submitted",
                    "broker_order_id": "rh-top-up-order",
                    "submitted_at": submitted_at,
                    "dry_run": False,
                },
            )
            sync_snapshot_to_artha(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "robinhood_mcp",
                    "account": account,
                    "portfolio": {"buying_power": 72.5, "cash": 72.5, "total_value": 72.5},
                    "positions": [
                        {
                            "symbol": "TEST",
                            "quantity": "1.25",
                            "average_buy_price": "9.20",
                            "market_price": "10.00",
                        }
                    ],
                    "orders": [
                        {
                            "id": "rh-top-up-order",
                            "symbol": "TEST",
                            "side": "buy",
                            "state": "unconfirmed",
                        }
                    ],
                },
                journal=journal,
                portfolio_path=portfolio_path,
            )
            result = record_order_fill(
                order_intent_id=intent.order_intent_id,
                fill_payload={
                    "id": "rh-top-up-order",
                    "symbol": "TEST",
                    "side": "buy",
                    "state": "filled",
                    "cumulative_quantity": "0.25",
                    "average_price": "10.00",
                    "last_transaction_at": datetime.now(timezone.utc).isoformat(),
                },
                journal=journal,
                portfolio_path=portfolio_path,
            )
            self.assertTrue(result.get("already_recorded"))
            pos = Portfolio.load(portfolio_path).get_position("TEST")
            self.assertAlmostEqual(pos.shares, 1.25)
            self.assertAlmostEqual(pos.avg_cost, 9.2)
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            ) = old_config

    def test_record_submission_and_order_watcher_activate_filled_orders(self):
        from datetime import datetime, timezone

        from artha.config import Config
        from artha.execution import build_order_intent, prepare_and_record_robinhood_review
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio
        from artha.robinhood_bridge import queue_trade_action_from_order_payload, record_order_submission, sync_orders_to_artha
        from artha.thesis_tracker import ThesisTracker

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
            journal.save_supervisor_run(
                {
                    "generated_at": "2026-06-04T14:30:00+00:00",
                    "severity": "PASS",
                    "payload": {"checks": []},
                }
            )
            journal.save_decision_features(
                {
                    "dossier_path": str(self.tmp_dir / "TEST_dossier.json"),
                    "generated_at": "2026-06-04T14:20:00+00:00",
                    "ticker": "TEST",
                    "final_verdict": "STARTER",
                    "opportunity_score": 68,
                    "confidence": 8,
                    "price": 100.0,
                    "evidence_count": 12,
                    "feature_json": "{}",
                }
            )
            pending = ThesisTracker(journal).create_thesis("TEST", "STARTER", thesis_summary="Unit submission thesis")
            account = {
                "account_number": "222233334",
                "type": "cash",
                "nickname": "Agentic",
                "agentic_allowed": True,
                "state": "active",
            }
            intent = build_order_intent(
                "TEST",
                "buy",
                notional=25,
                quantity=0.25,
                limit_price=100,
                estimated_price=100,
                decision_dossier_path=str(self.tmp_dir / "TEST_dossier.json"),
            )
            first = prepare_and_record_robinhood_review(
                intent,
                account,
                market_data=self._market_data(),
                journal=journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            )
            action = queue_trade_action_from_order_payload(first, journal=journal)
            portfolio_path = self.tmp_dir / "portfolio.json"
            submission = record_order_submission(
                action_id=action["action_id"],
                place_response={
                    "data": {
                        "order": {
                            "id": "rh-submit-filled",
                            "symbol": "TEST",
                            "side": "buy",
                            "state": "filled",
                            "cumulative_quantity": "0.25",
                            "average_price": "100.00",
                            "created_at": "2026-06-04T15:00:00+00:00",
                        }
                    }
                },
                journal=journal,
                portfolio_path=portfolio_path,
            )
            self.assertEqual(submission["status"], "PASS")
            self.assertEqual(submission["execution_status"], "filled")
            self.assertEqual(submission["fill"]["status"], "PASS")
            self.assertEqual(Portfolio.load(portfolio_path).get_position("TEST").thesis_id, pending.thesis_id)

            # A later Robinhood order-history snapshot should also activate fills
            # for submitted orders whose initial place response was still open.
            second_intent = build_order_intent(
                "TEST",
                "buy",
                notional=25,
                quantity=0.25,
                limit_price=100,
                estimated_price=100,
                decision_dossier_path=str(self.tmp_dir / "TEST_dossier.json"),
            )
            prepare_and_record_robinhood_review(
                second_intent,
                account,
                market_data=self._market_data(),
                journal=journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            )
            journal.update_execution_order(
                second_intent.order_intent_id,
                {"status": "submitted", "broker_order_id": "rh-watch-filled"},
            )
            watched = sync_orders_to_artha(
                {
                    "orders": [
                        {
                            "id": "rh-watch-filled",
                            "symbol": "TEST",
                            "side": "buy",
                            "state": "filled",
                            "cumulative_quantity": "0.25",
                            "average_price": "101.00",
                            "placed_agent": "agentic",
                        }
                    ]
                },
                journal=journal,
                portfolio_path=portfolio_path,
            )
            self.assertEqual(watched["status"], "PASS")
            self.assertEqual(len(watched["filled"]), 1)
            self.assertEqual(journal.get_execution_order_by_intent_id(second_intent.order_intent_id)["status"], "filled")
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            ) = old_config

    def test_openclaw_handler_executes_review_and_place_sequence(self):
        from datetime import datetime, timezone

        from artha.config import Config
        from artha.execution import build_order_intent, prepare_and_record_robinhood_review
        from artha.openclaw_robinhood_handler import handle_telegram_callback
        from artha.robinhood_bridge import queue_trade_action_from_order_payload, set_trading_disabled, write_robinhood_snapshot

        class FakeRobinhood:
            def __init__(self):
                self.calls = []

            def get_equity_tradability(self, **kwargs):
                self.calls.append(("tradability", kwargs))
                return {
                    "data": {
                        "results": [
                            {
                                "symbol": kwargs["symbols"][0],
                                "tradeable": True,
                                "state": "active",
                                "fractional_tradability": "tradable",
                            }
                        ]
                    }
                }

            def review_equity_order(self, **kwargs):
                self.calls.append(("review", kwargs))
                payload = dict(kwargs)
                payload["order_checks"] = {}
                return {"data": payload}

            def place_equity_order(self, **kwargs):
                self.calls.append(("place", kwargs))
                return {
                    "data": {
                        "order": {
                            "id": "rh-handler-queued",
                            "symbol": kwargs["symbol"],
                            "side": kwargs["side"],
                            "state": "queued",
                            "quantity": kwargs.get("quantity"),
                            "created_at": "2026-06-04T15:00:00+00:00",
                        }
                    }
                }

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            Config.ROBINHOOD_REVIEW_ONLY,
            Config.ROBINHOOD_DRY_RUN_ONLY,
            Config.ROBINHOOD_AGENTIC_ENABLED,
            Config.ROBINHOOD_KILL_SWITCH,
            Config.ROBINHOOD_CONTROL_FILE,
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            Config.ROBINHOOD_REVIEW_ONLY = False
            Config.ROBINHOOD_DRY_RUN_ONLY = False
            Config.ROBINHOOD_AGENTIC_ENABLED = True
            Config.ROBINHOOD_KILL_SWITCH = False
            Config.ROBINHOOD_CONTROL_FILE = str(self.tmp_dir / "control.json")
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE = str(self.tmp_dir / "robinhood_snapshot.json")
            set_trading_disabled(False, "unit test")
            write_robinhood_snapshot(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "robinhood_mcp",
                    "account": {
                        "account_number": "222233334",
                        "type": "cash",
                        "nickname": "Agentic",
                        "agentic_allowed": True,
                        "state": "active",
                    },
                    "portfolio": {"buying_power": 100, "cash": 100, "total_value": 100},
                    "positions": [],
                    "orders": [],
                },
                path=Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            )
            journal = self._journal_with_clean_buy_decision()
            account = {
                "account_number": "222233334",
                "type": "cash",
                "nickname": "Agentic",
                "agentic_allowed": True,
                "state": "active",
            }
            intent = build_order_intent("TEST", "buy", quantity=1, limit_price=10, estimated_price=10)
            result = prepare_and_record_robinhood_review(
                intent,
                account,
                market_data={"price": 10, "volume": 2_000_000, "bid": 9.99, "ask": 10.01},
                journal=journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            )
            action = queue_trade_action_from_order_payload(result, journal=journal)
            broker = FakeRobinhood()

            review = handle_telegram_callback(action["callback_data"]["review"], broker, journal=journal, portfolio_path=self.tmp_dir / "portfolio.json")
            self.assertEqual(review["status"], "PASS")
            self.assertEqual(review["steps"], ["tradability", "review", "record_review"])
            self.assertEqual(journal.get_trade_action(action["action_id"])["status"], "review_clear")

            placed = handle_telegram_callback(action["callback_data"]["place"], broker, journal=journal, portfolio_path=self.tmp_dir / "portfolio.json")
            self.assertEqual(placed["status"], "PASS")
            self.assertEqual(placed["steps"], ["tradability", "review", "record_review", "place", "record_submission"])
            self.assertEqual([name for name, _ in broker.calls], ["tradability", "review", "tradability", "review", "place"])
            self.assertEqual(journal.get_execution_order_by_intent_id(intent.order_intent_id)["status"], "submitted")
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
                Config.ROBINHOOD_REVIEW_ONLY,
                Config.ROBINHOOD_DRY_RUN_ONLY,
                Config.ROBINHOOD_AGENTIC_ENABLED,
                Config.ROBINHOOD_KILL_SWITCH,
                Config.ROBINHOOD_CONTROL_FILE,
                Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            ) = old_config

    def test_manual_place_final_clearance_preserves_place_authorization(self):
        from datetime import datetime, timezone

        from artha.config import Config
        from artha.execution import build_order_intent, prepare_and_record_robinhood_review
        from artha.robinhood_bridge import (
            build_action_operation,
            queue_trade_action_from_order_payload,
            record_action_review,
            run_final_clearance_for_action,
            set_trading_disabled,
            write_robinhood_snapshot,
        )

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            Config.ROBINHOOD_REVIEW_ONLY,
            Config.ROBINHOOD_DRY_RUN_ONLY,
            Config.ROBINHOOD_AGENTIC_ENABLED,
            Config.ROBINHOOD_KILL_SWITCH,
            Config.ROBINHOOD_CONTROL_FILE,
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            Config.EXECUTION_OFFICER_LLM_ENABLED,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            Config.ROBINHOOD_REVIEW_ONLY = False
            Config.ROBINHOOD_DRY_RUN_ONLY = False
            Config.ROBINHOOD_AGENTIC_ENABLED = True
            Config.ROBINHOOD_KILL_SWITCH = False
            Config.ROBINHOOD_CONTROL_FILE = str(self.tmp_dir / "control.json")
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE = str(self.tmp_dir / "robinhood_snapshot.json")
            Config.EXECUTION_OFFICER_LLM_ENABLED = True
            set_trading_disabled(False, "unit test")
            write_robinhood_snapshot(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "robinhood_mcp",
                    "account": {
                        "account_number": "222233334",
                        "type": "cash",
                        "nickname": "Agentic",
                        "agentic_allowed": True,
                        "state": "active",
                    },
                    "portfolio": {"buying_power": 100, "cash": 100, "total_value": 100},
                    "positions": [],
                    "orders": [],
                },
                path=Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            )
            journal = self._journal_with_clean_buy_decision()
            account = {
                "account_number": "222233334",
                "type": "cash",
                "nickname": "Agentic",
                "agentic_allowed": True,
                "state": "active",
            }
            intent = build_order_intent("TEST", "buy", notional=25, quantity=0.25, limit_price=100, estimated_price=100)
            result = prepare_and_record_robinhood_review(
                intent,
                account,
                market_data=self._market_data(),
                journal=journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            )
            action = queue_trade_action_from_order_payload(result, journal=journal)
            review_response = {
                "data": {
                    "symbol": "TEST",
                    "side": "buy",
                    "type": "market",
                    "dollar_amount": "25.00",
                    "order_checks": {},
                    "quote_data": {
                        "symbol": "TEST",
                        "state": "active",
                        "bid_price": "99.95",
                        "ask_price": "100.05",
                        "last_trade_price": "100.00",
                    },
                    "market_data_disclosure": "Bid $99.95 x 1 P · Ask $100.05 x 1 M · Last $100.00 x 1. Updated 10:00 AM ET.",
                }
            }
            tradability_response = {
                "data": {
                    "results": [
                        {
                            "symbol": "TEST",
                            "tradeable": True,
                            "state": "active",
                            "fractional_tradability": "tradable",
                        }
                    ]
                }
            }

            first_review = record_action_review(
                action["action_id"],
                review_response,
                tradability_response=tradability_response,
                journal=journal,
            )
            self.assertEqual(first_review["status"], "review_clear")
            with patch("artha.scheduler.MarketHours.is_market_open", return_value=True):
                place_op = build_action_operation(action["callback_data"]["place"], journal=journal)
            self.assertEqual(place_op["operation"], "tradability_then_review_then_place_equity_order", place_op)
            self.assertEqual(journal.get_trade_action(action["action_id"])["status"], "place_requested")

            second_review = record_action_review(
                action["action_id"],
                review_response,
                tradability_response=tradability_response,
                journal=journal,
            )
            self.assertEqual(second_review["status"], "review_clear")
            stored = journal.get_trade_action(action["action_id"])
            self.assertEqual(stored["status"], "review_clear")
            stored_result = json.loads(stored["result_json"]) if isinstance(stored["result_json"], str) else stored["result_json"]
            self.assertTrue(stored_result["manual_place_requested"]["passed"])

            prompts = []

            def fake_clearance(prompt):
                prompts.append(prompt)
                return '{"allow_place":true,"confidence":9,"rationale":"Manual Place was requested and the second review remains clear.","risk_flags":[],"requested_data":[]}'

            with patch("artha.execution_officer.ChatGPTBackendClient.chat", side_effect=fake_clearance):
                final = run_final_clearance_for_action(action["action_id"], journal=journal)
            self.assertTrue(final["allow_place"])
            self.assertIn('"status": "place_requested"', prompts[0])
            self.assertIn('"manual_place_requested"', prompts[0])
            self.assertEqual(journal.get_trade_action(action["action_id"])["status"], "review_clear")
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
                Config.ROBINHOOD_REVIEW_ONLY,
                Config.ROBINHOOD_DRY_RUN_ONLY,
                Config.ROBINHOOD_AGENTIC_ENABLED,
                Config.ROBINHOOD_KILL_SWITCH,
                Config.ROBINHOOD_CONTROL_FILE,
                Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
                Config.EXECUTION_OFFICER_LLM_ENABLED,
            ) = old_config

    def test_execution_officer_uses_gpt55_extra_high_and_selects_whole_share_limit(self):
        from types import SimpleNamespace

        from artha.config import Config
        from artha.execution_officer import BUY_READY, WHOLE_SHARE_LIMIT, build_execution_officer_plan

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.EXECUTION_OFFICER_LLM_ENABLED,
            Config.EXECUTION_OFFICER_MODEL,
            Config.EXECUTION_OFFICER_REASONING_EFFORT,
            Config.EXECUTION_OFFICER_TEMPERATURE,
            Config.ROBINHOOD_AUTO_BUY_ENABLED,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.EXECUTION_OFFICER_LLM_ENABLED = True
            Config.EXECUTION_OFFICER_MODEL = "gpt-5.5"
            Config.EXECUTION_OFFICER_REASONING_EFFORT = "xhigh"
            Config.EXECUTION_OFFICER_TEMPERATURE = 2.0
            Config.ROBINHOOD_AUTO_BUY_ENABLED = True
            decision = SimpleNamespace(
                ticker="TEST",
                final_verdict="STARTER",
                adjusted_score=72,
                opportunity_score=72,
                confidence=6,
                recommended_allocation_pct=5.0,
                recommended_action="STARTER near $15.97; do not chase far above the guardrail.",
                dossier_path=str(self.tmp_dir / "TEST_dossier.json"),
            )
            with patch(
                "artha.execution_officer.ChatGPTBackendClient.chat",
                return_value=json.dumps(
                    {
                        "selected_candidate_id": "whole_share_marketable_limit",
                        "execution_verdict": "BUY_READY",
                        "confidence": 8,
                        "rationale": "Ask is inside the no-chase cap and one whole share fits the starter budget.",
                        "requested_data": [],
                        "risk_flags": [],
                    }
                ),
            ) as mock_chat:
                plan = build_execution_officer_plan(
                    ticker="TEST",
                    decision=decision,
                    recommended_notional=18.38,
                    reference_price=15.97,
                    current_price=16.15,
                    market_data={"price": 16.15, "volume": 1_000_000, "bid": 16.14, "ask": 16.17},
                )
            self.assertEqual(plan.execution_verdict, BUY_READY)
            self.assertEqual(plan.strategy, WHOLE_SHARE_LIMIT)
            self.assertEqual(plan.quantity, 1.0)
            self.assertAlmostEqual(plan.limit_price, 16.29, places=2)
            self.assertTrue(plan.auto_buy_eligible)
            self.assertEqual(plan.officer_model, "gpt-5.5")
            self.assertEqual(plan.officer_reasoning_effort, "xhigh")
            self.assertEqual(plan.officer_temperature, 2.0)
            self.assertTrue(plan.officer_used)
            prompt = mock_chat.call_args.args[0]
            self.assertIn("Execution Officer", prompt)
            self.assertIn("Return JSON only", prompt)
            self.assertIn("Robinhood review/tradability", prompt)
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.EXECUTION_OFFICER_LLM_ENABLED,
                Config.EXECUTION_OFFICER_MODEL,
                Config.EXECUTION_OFFICER_REASONING_EFFORT,
                Config.EXECUTION_OFFICER_TEMPERATURE,
                Config.ROBINHOOD_AUTO_BUY_ENABLED,
            ) = old_config

    def test_accumulate_verdict_is_allowed_for_unattended_auto_buy(self):
        from types import SimpleNamespace

        from artha.config import Config
        from artha.execution_officer import BUY_READY, FRACTIONAL_MARKET, build_execution_officer_plan

        old_config = (
            Config.EXECUTION_OFFICER_LLM_ENABLED,
            Config.ROBINHOOD_AUTO_BUY_ENABLED,
        )
        try:
            Config.EXECUTION_OFFICER_LLM_ENABLED = False
            Config.ROBINHOOD_AUTO_BUY_ENABLED = True
            self.assertIn("ACCUMULATE", Config.ROBINHOOD_AUTO_BUY_ALLOWED_VERDICTS)
            decision = SimpleNamespace(
                ticker="TEST",
                final_verdict="ACCUMULATE",
                adjusted_score=76,
                opportunity_score=76,
                confidence=7,
                recommended_allocation_pct=5.0,
                recommended_action="ACCUMULATE near $20.00 if the live broker quote stays inside guardrails.",
                dossier_path=str(self.tmp_dir / "TEST_dossier.json"),
            )

            plan = build_execution_officer_plan(
                ticker="TEST",
                decision=decision,
                recommended_notional=17.50,
                reference_price=20.00,
                current_price=20.00,
                market_data={"price": 20.00, "volume": 1_000_000, "bid": 19.99, "ask": 20.01},
            )

            self.assertEqual(plan.execution_verdict, BUY_READY)
            self.assertEqual(plan.strategy, FRACTIONAL_MARKET)
            self.assertTrue(plan.checks["auto_buy_verdict_allowed"])
            self.assertTrue(plan.auto_buy_eligible)
        finally:
            (
                Config.EXECUTION_OFFICER_LLM_ENABLED,
                Config.ROBINHOOD_AUTO_BUY_ENABLED,
            ) = old_config

    def test_stage_a_execution_officer_cannot_deadlock_clean_fractional_buy_on_future_broker_checks(self):
        from types import SimpleNamespace

        from artha.config import Config
        from artha.execution_officer import BUY_READY, FRACTIONAL_MARKET, build_execution_officer_plan

        old_config = (
            Config.EXECUTION_OFFICER_LLM_ENABLED,
            Config.ROBINHOOD_AUTO_BUY_ENABLED,
        )
        try:
            Config.EXECUTION_OFFICER_LLM_ENABLED = True
            Config.ROBINHOOD_AUTO_BUY_ENABLED = True
            decision = SimpleNamespace(
                ticker="USB",
                final_verdict="STARTER",
                adjusted_score=87,
                opportunity_score=87,
                confidence=7,
                recommended_allocation_pct=6.0,
                recommended_action="STARTER near $63.90 if execution remains safe.",
                dossier_path=str(self.tmp_dir / "USB_dossier.json"),
            )
            premature_wait = json.dumps(
                {
                    "selected_candidate_id": "wait_for_safe_execution",
                    "execution_verdict": "WAIT_FOR_SAFE_EXECUTION",
                    "confidence": 8,
                    "rationale": "A current Robinhood order review was not supplied.",
                    "requested_data": ["Robinhood tradability", "Robinhood exact-order review"],
                    "risk_flags": [],
                }
            )
            with patch("artha.execution_officer.ChatGPTBackendClient.chat", return_value=premature_wait) as mock_chat:
                plan = build_execution_officer_plan(
                    ticker="USB",
                    decision=decision,
                    recommended_notional=21.87,
                    reference_price=63.90,
                    current_price=63.90,
                    market_data={"price": 63.90, "volume": 1_000_000, "bid": 63.89, "ask": 63.91},
                )

            self.assertEqual(plan.execution_verdict, BUY_READY)
            self.assertEqual(plan.strategy, FRACTIONAL_MARKET)
            self.assertTrue(plan.auto_buy_eligible)
            self.assertFalse(plan.officer_used)
            self.assertIn("attempted to downgrade", plan.officer_json["rejected_reason"])
            self.assertFalse(any("Robinhood order review was not supplied" in reason for reason in plan.reasons))
            prompt = mock_chat.call_args.args[0]
            self.assertIn("Stage A: order planning", prompt)
            self.assertIn("intentionally_deferred_to_stage_b", prompt)
            self.assertIn("Do not request or require Stage B Robinhood evidence", prompt)
        finally:
            (
                Config.EXECUTION_OFFICER_LLM_ENABLED,
                Config.ROBINHOOD_AUTO_BUY_ENABLED,
            ) = old_config

    def test_stage_a_execution_officer_preserves_real_score_gate_after_clean_order_plan(self):
        from types import SimpleNamespace

        from artha.config import Config
        from artha.execution_officer import BUY_READY, build_execution_officer_plan

        old_config = Config.EXECUTION_OFFICER_LLM_ENABLED
        try:
            Config.EXECUTION_OFFICER_LLM_ENABLED = False
            decision = SimpleNamespace(
                ticker="EPR",
                final_verdict="TACTICAL_BUY",
                adjusted_score=67,
                opportunity_score=67,
                confidence=6,
                recommended_allocation_pct=4.0,
                recommended_action="TACTICAL_BUY near $61.49.",
            )
            plan = build_execution_officer_plan(
                ticker="EPR",
                decision=decision,
                recommended_notional=14.58,
                reference_price=61.49,
                current_price=61.49,
                market_data={"price": 61.49, "volume": 500_000, "bid": 61.46, "ask": 61.52},
            )
            self.assertEqual(plan.execution_verdict, BUY_READY)
            self.assertFalse(plan.checks["auto_buy_score_allowed"])
            self.assertFalse(plan.auto_buy_eligible)
        finally:
            Config.EXECUTION_OFFICER_LLM_ENABLED = old_config

    def test_stage_a_execution_officer_cannot_upgrade_price_drift_blocked_candidate(self):
        from types import SimpleNamespace

        from artha.config import Config
        from artha.execution_officer import WAIT_FOR_SAFE_EXECUTION, build_execution_officer_plan

        old_config = Config.EXECUTION_OFFICER_LLM_ENABLED
        try:
            Config.EXECUTION_OFFICER_LLM_ENABLED = True
            decision = SimpleNamespace(
                ticker="MU",
                final_verdict="STARTER",
                adjusted_score=85,
                opportunity_score=85,
                confidence=8,
                recommended_allocation_pct=5.0,
                recommended_action="STARTER only inside the price guardrail.",
            )
            unsafe_upgrade = json.dumps(
                {
                    "selected_candidate_id": "fractional_market_notional",
                    "execution_verdict": "BUY_READY",
                    "confidence": 9,
                    "rationale": "Buy immediately.",
                    "requested_data": [],
                    "risk_flags": [],
                }
            )
            with patch("artha.execution_officer.ChatGPTBackendClient.chat", return_value=unsafe_upgrade):
                plan = build_execution_officer_plan(
                    ticker="MU",
                    decision=decision,
                    recommended_notional=22.78,
                    reference_price=894.27,
                    current_price=894.27,
                    market_data={"price": 894.27, "volume": 2_000_000, "bid": 899.50, "ask": 900.00},
                )

            self.assertEqual(plan.execution_verdict, WAIT_FOR_SAFE_EXECUTION)
            self.assertFalse(plan.auto_buy_eligible)
            self.assertFalse(plan.officer_used)
            self.assertIn("disallowed BUY_READY", plan.officer_json["rejected_reason"])
        finally:
            Config.EXECUTION_OFFICER_LLM_ENABLED = old_config

    def test_scheduled_fractional_buy_queues_stage_b_even_when_stage_a_llm_requests_future_broker_data(self):
        from types import SimpleNamespace

        from artha.config import Config
        from artha.scheduler import ArthaScheduler, MarketHours

        old_config = (
            Config.SCAN_PREPARE_ROBINHOOD_REVIEW_FOR_BUYS,
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            Config.EXECUTION_OFFICER_LLM_ENABLED,
            Config.ROBINHOOD_AUTO_BUY_ENABLED,
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
        )
        try:
            Config.SCAN_PREPARE_ROBINHOOD_REVIEW_FOR_BUYS = True
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            Config.EXECUTION_OFFICER_LLM_ENABLED = True
            Config.ROBINHOOD_AUTO_BUY_ENABLED = True
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE = str(self._write_agentic_snapshot_file(cash=100.0))
            journal = self._journal_with_clean_buy_decision()
            scheduler = ArthaScheduler()
            scheduler.market_hours = SimpleNamespace(is_market_open=lambda dt=None: True)
            dossier_path = self.tmp_dir / "USB_dossier.json"
            dossier_path.write_text(
                '{"ticker":"USB","evidence":[{"id":"E001","source":"fmp.quote"}]}',
                encoding="utf-8",
            )
            journal.save_decision_features(
                {
                    "dossier_path": str(dossier_path),
                    "generated_at": "2026-08-07T16:52:05+00:00",
                    "ticker": "USB",
                    "final_verdict": "STARTER",
                    "opportunity_score": 87,
                    "adjusted_score": 87,
                    "confidence": 7,
                    "price": 63.90,
                    "sector": "Financial Services",
                    "evidence_count": 34,
                    "source_count": 12,
                    "feature_json": json.dumps({"unit": True}),
                }
            )
            decision = SimpleNamespace(
                ticker="USB",
                final_verdict="STARTER",
                recommended_action="STARTER - buy about $21.87 near $63.90 if execution remains safe.",
                synthesis_report="",
                recommended_allocation_pct=6.0,
                opportunity_score=87,
                adjusted_score=87,
                confidence=7,
                dossier_path=str(dossier_path),
            )
            stock_data = {
                "quote": {"price": 63.90, "volume": 1_000_000, "bid": 63.89, "ask": 63.91},
                "yf_quote": {"price": 63.90, "volume": 1_000_000, "bid": 63.89, "ask": 63.91},
            }
            premature_wait = json.dumps(
                {
                    "selected_candidate_id": "wait_for_safe_execution",
                    "execution_verdict": "WAIT_FOR_SAFE_EXECUTION",
                    "confidence": 8,
                    "rationale": "Robinhood tradability and exact-order review are not supplied.",
                    "requested_data": ["Robinhood tradability", "Robinhood exact-order review"],
                    "risk_flags": [],
                }
            )

            clean_sector_state = {
                "total_value": 364.50,
                "total_invested": 0.0,
                "cash_available": 364.50,
                "positions": [],
            }
            with patch("artha.execution_officer.ChatGPTBackendClient.chat", return_value=premature_wait), \
                 patch("artha.execution.PortfolioStateEngine.compute_state", return_value=clean_sector_state), \
                 patch.object(MarketHours, "is_market_open", lambda self, now=None: True):
                result = scheduler._prepare_scan_buy_robinhood_review(
                    "USB",
                    decision,
                    stock_data,
                    journal,
                    nav=364.50,
                    recommendation_id=504,
                )

            self.assertIsNotNone(result)
            self.assertEqual(result["broker_result"]["status"], "review_ready", result)
            self.assertEqual(result["intent"]["order_type"], "market")
            self.assertAlmostEqual(result["intent"]["notional"], 21.87, places=2)
            self.assertEqual(result["trade_action"]["action_type"], "auto_buy")
            self.assertNotEqual(result["trade_action"]["status"], "not_queued")
            self.assertNotIn("entry_watch", result)
            self.assertIsNone(result["trade_action"]["reply_markup"])
            stored = journal.get_trade_action(result["trade_action"]["action_id"])
            self.assertEqual(stored["action_type"], "auto_buy")
        finally:
            (
                Config.SCAN_PREPARE_ROBINHOOD_REVIEW_FOR_BUYS,
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
                Config.EXECUTION_OFFICER_LLM_ENABLED,
                Config.ROBINHOOD_AUTO_BUY_ENABLED,
                Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            ) = old_config

    def test_execution_officer_expensive_stock_does_not_force_whole_share(self):
        from types import SimpleNamespace

        from artha.config import Config
        from artha.execution_officer import FRACTIONAL_MARKET, build_execution_officer_plan

        old_config = Config.EXECUTION_OFFICER_LLM_ENABLED
        try:
            Config.EXECUTION_OFFICER_LLM_ENABLED = False
            decision = SimpleNamespace(
                ticker="UTHR",
                final_verdict="STARTER",
                adjusted_score=78,
                opportunity_score=78,
                confidence=7,
                recommended_action="STARTER for about $22 near $546.",
            )
            plan = build_execution_officer_plan(
                ticker="UTHR",
                decision=decision,
                recommended_notional=22.0,
                reference_price=546.0,
                current_price=546.2,
                market_data={"price": 546.2, "volume": 500_000, "bid": 546.0, "ask": 546.4},
            )
            self.assertEqual(plan.strategy, FRACTIONAL_MARKET)
            self.assertLess(plan.quantity, 1.0)
            self.assertIsNone(
                next(
                    c for c in plan.checks["candidates"] if c["candidate_id"] == "whole_share_marketable_limit"
                )["quantity"]
            )
        finally:
            Config.EXECUTION_OFFICER_LLM_ENABLED = old_config

    def test_scan_buy_side_decision_prepares_robinhood_review_only(self):
        from types import SimpleNamespace

        from artha.config import Config
        from artha.scheduler import ArthaScheduler

        old_config = (
            Config.SCAN_PREPARE_ROBINHOOD_REVIEW_FOR_BUYS,
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            Config.EXECUTION_OFFICER_LLM_ENABLED,
            Config.ROBINHOOD_AUTO_BUY_ENABLED,
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
        )
        try:
            Config.SCAN_PREPARE_ROBINHOOD_REVIEW_FOR_BUYS = True
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            Config.EXECUTION_OFFICER_LLM_ENABLED = False
            Config.ROBINHOOD_AUTO_BUY_ENABLED = False
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE = str(self._write_agentic_snapshot_file(cash=100.0))
            journal = self._journal_with_clean_buy_decision()
            scheduler = ArthaScheduler()
            scheduler.market_hours = SimpleNamespace(is_market_open=lambda dt=None: True)
            decision = SimpleNamespace(
                ticker="TEST",
                final_verdict="STARTER",
                recommended_action="STARTER — Buy about $17.50 of TEST at ~$10.00 using a limit order.",
                synthesis_report="",
                recommended_allocation_pct=5.0,
                opportunity_score=72,
                adjusted_score=72,
                confidence=8,
                dossier_path=str(self.tmp_dir / "TEST_dossier.json"),
            )
            stock_data = {
                "quote": {"price": 10.0, "volume": 2_000_000, "bid": 9.99, "ask": 10.01},
                "yf_quote": {"price": 10.0, "volume": 2_000_000, "bid": 9.99, "ask": 10.01},
            }
            # Wave 2: dry_run now follows Config.ROBINHOOD_DRY_RUN_ONLY, so the
            # guardrails enforce market hours at queue time; pin market open.
            from unittest.mock import patch as _patch

            from artha.scheduler import MarketHours as _MarketHours

            with _patch.object(_MarketHours, "is_market_open", lambda self, now=None: True):
                result = scheduler._prepare_scan_buy_robinhood_review(
                    "TEST",
                    decision,
                    stock_data,
                    journal,
                    nav=350.0,
                    recommendation_id=123,
                )
            rows = journal.get_execution_orders(limit=1)
            self.assertIsNotNone(result)
            self.assertEqual(rows[0]["status"], "review_ready")
            self.assertEqual(rows[0]["recommendation_id"], 123)
            self.assertIsNone(rows[0]["submitted_at"])
            self.assertIn('"type": "market"', rows[0]["request_json"])
            self.assertIn('"dollar_amount": "17.50"', rows[0]["request_json"])
            self.assertNotIn('"limit_price"', rows[0]["request_json"])
            msg, reply_markup = scheduler._format_scan_order_review_summary([result])
            self.assertIn("review-only", msg)
            self.assertIn("No real Robinhood order was placed", msg)
            self.assertTrue(reply_markup["inline_keyboard"])
            eo_msg = scheduler._format_execution_officer_scan_update("TEST", decision, result)
            self.assertIn("ARTHA EXECUTION OFFICER - $TEST", eo_msg)
            self.assertIn("Execution verdict: REVIEW READY", eo_msg)
            self.assertIn("Proposed order:", eo_msg)
        finally:
            (
                Config.SCAN_PREPARE_ROBINHOOD_REVIEW_FOR_BUYS,
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
                Config.EXECUTION_OFFICER_LLM_ENABLED,
                Config.ROBINHOOD_AUTO_BUY_ENABLED,
                Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            ) = old_config

    def test_scan_buy_vtrs_like_starter_prepares_auto_whole_share_limit(self):
        from types import SimpleNamespace

        from artha.config import Config
        from artha.scheduler import ArthaScheduler

        old_config = (
            Config.SCAN_PREPARE_ROBINHOOD_REVIEW_FOR_BUYS,
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            Config.EXECUTION_OFFICER_LLM_ENABLED,
            Config.ROBINHOOD_AUTO_BUY_ENABLED,
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
        )
        try:
            Config.SCAN_PREPARE_ROBINHOOD_REVIEW_FOR_BUYS = True
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            Config.EXECUTION_OFFICER_LLM_ENABLED = False
            Config.ROBINHOOD_AUTO_BUY_ENABLED = True
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE = str(self._write_agentic_snapshot_file(cash=100.0))
            journal = self._journal_with_clean_buy_decision()
            scheduler = ArthaScheduler()
            scheduler.market_hours = SimpleNamespace(is_market_open=lambda dt=None: True)
            decision = SimpleNamespace(
                ticker="TEST",
                final_verdict="STARTER",
                recommended_action="STARTER — review about $18 of TEST at $15.97 with a no-chase guardrail.",
                synthesis_report="",
                recommended_allocation_pct=5.0,
                opportunity_score=72,
                adjusted_score=72,
                confidence=6,
                dossier_path=str(self.tmp_dir / "TEST_dossier.json"),
            )
            stock_data = {
                "quote": {"price": 16.15, "volume": 1_000_000, "bid": 16.14, "ask": 16.17},
                "yf_quote": {"price": 16.15, "volume": 1_000_000, "bid": 16.14, "ask": 16.17},
            }
            result = scheduler._prepare_scan_buy_robinhood_review(
                "TEST",
                decision,
                stock_data,
                journal,
                nav=367.5,
                recommendation_id=456,
            )
            self.assertIsNotNone(result)
            self.assertEqual(result["broker_result"]["status"], "review_ready")
            self.assertEqual(result["intent"]["order_type"], "limit")
            self.assertEqual(result["intent"]["quantity"], 1.0)
            self.assertAlmostEqual(result["intent"]["limit_price"], 16.29, places=2)
            self.assertEqual(result["trade_action"]["action_type"], "auto_buy")
            self.assertIsNone(result["trade_action"]["reply_markup"])
            self.assertEqual(result["trade_action"]["callback_data"], {})
            rows = journal.get_execution_orders(limit=1)
            self.assertIn('"type": "limit"', rows[0]["request_json"])
            self.assertIn('"quantity": "1"', rows[0]["request_json"])
            self.assertIn('"limit_price": "16.29"', rows[0]["request_json"])
            msg, reply_markup = scheduler._format_scan_order_review_summary([result])
            self.assertIn("Auto-buy", msg)
            self.assertIn("OpenClaw", msg)
            self.assertIn("no user action required", msg)
            self.assertIn("No manual buy permission is needed", msg)
            self.assertIsNone(reply_markup)
            eo_msg = scheduler._format_execution_officer_scan_update("TEST", decision, result)
            self.assertIn("Execution verdict: AUTO-BUY QUEUED", eo_msg)
            self.assertIn("OpenClaw auto-buy runner", eo_msg)
        finally:
            (
                Config.SCAN_PREPARE_ROBINHOOD_REVIEW_FOR_BUYS,
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
                Config.EXECUTION_OFFICER_LLM_ENABLED,
                Config.ROBINHOOD_AUTO_BUY_ENABLED,
                Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            ) = old_config

    def test_execution_officer_update_explains_non_buy_verdict(self):
        from types import SimpleNamespace

        from artha.scheduler import ArthaScheduler

        scheduler = ArthaScheduler()
        decision = SimpleNamespace(
            ticker="AMZN",
            final_verdict="DEFER",
            opportunity_score=34,
            confidence=7,
        )
        msg = scheduler._format_execution_officer_scan_update("AMZN", decision, None)
        self.assertIn("ARTHA EXECUTION OFFICER - $AMZN", msg)
        self.assertIn("Execution verdict: NO ORDER", msg)
        self.assertIn("Council verdict is DEFER", msg)
        self.assertIn("No quote/review/place attempt", msg)

    def test_openclaw_auto_buy_handler_places_after_clean_review(self):
        from datetime import datetime, timezone

        from artha.config import Config
        from artha.execution import build_order_intent, prepare_and_record_robinhood_review
        from artha.execution_officer import run_agentic_execution_officer
        from artha.openclaw_robinhood_handler import handle_auto_buy_action
        from artha.robinhood_bridge import build_auto_buy_operation, queue_trade_action_from_order_payload, set_trading_disabled, write_robinhood_snapshot

        class FakeRobinhood:
            def __init__(self):
                self.calls = []

            def get_equity_tradability(self, **kwargs):
                self.calls.append(("tradability", kwargs))
                return {
                    "data": {
                        "results": [
                            {
                                "symbol": kwargs["symbols"][0],
                                "tradeable": True,
                                "state": "active",
                                "fractional_tradability": "tradable",
                            }
                        ]
                    }
                }

            def review_equity_order(self, **kwargs):
                self.calls.append(("review", kwargs))
                payload = dict(kwargs)
                payload["order_checks"] = {}
                return {"data": payload}

            def place_equity_order(self, **kwargs):
                self.calls.append(("place", kwargs))
                return {
                    "data": {
                        "order": {
                            "id": "rh-auto-queued",
                            "symbol": kwargs["symbol"],
                            "side": kwargs["side"],
                            "state": "queued",
                            "quantity": kwargs.get("quantity"),
                            "created_at": "2026-06-04T15:00:00+00:00",
                        }
                    }
                }

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            Config.ROBINHOOD_REVIEW_ONLY,
            Config.ROBINHOOD_DRY_RUN_ONLY,
            Config.ROBINHOOD_AGENTIC_ENABLED,
            Config.ROBINHOOD_KILL_SWITCH,
            Config.ROBINHOOD_CONTROL_FILE,
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            Config.ROBINHOOD_AUTO_BUY_ENABLED,
            Config.EXECUTION_OFFICER_LLM_ENABLED,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            Config.ROBINHOOD_REVIEW_ONLY = False
            Config.ROBINHOOD_DRY_RUN_ONLY = False
            Config.ROBINHOOD_AGENTIC_ENABLED = True
            Config.ROBINHOOD_KILL_SWITCH = False
            Config.ROBINHOOD_CONTROL_FILE = str(self.tmp_dir / "control.json")
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE = str(self.tmp_dir / "robinhood_snapshot.json")
            Config.ROBINHOOD_AUTO_BUY_ENABLED = True
            Config.EXECUTION_OFFICER_LLM_ENABLED = False
            set_trading_disabled(False, "unit test")
            write_robinhood_snapshot(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "robinhood_mcp",
                    "account": {
                        "account_number": "222233334",
                        "type": "cash",
                        "nickname": "Agentic",
                        "agentic_allowed": True,
                        "state": "active",
                    },
                    "portfolio": {"buying_power": 100, "cash": 100, "total_value": 100},
                    "positions": [],
                    "orders": [],
                },
                path=Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            )
            journal = self._journal_with_clean_buy_decision()
            account = {
                "account_number": "222233334",
                "type": "cash",
                "nickname": "Agentic",
                "agentic_allowed": True,
                "state": "active",
            }
            intent = build_order_intent("TEST", "buy", quantity=1, notional=16.29, limit_price=16.29, estimated_price=16.15)
            intent.evidence = {
                "execution_officer": {
                    "auto_buy_eligible": True,
                    "execution_verdict": "BUY_READY",
                    "strategy": "WHOLE_SHARE_LIMIT",
                    "selected_candidate_id": "whole_share_marketable_limit",
                    "officer_model": "gpt-5.5",
                    "officer_reasoning_effort": "xhigh",
                    "officer_temperature": 2.0,
                }
            }
            result = prepare_and_record_robinhood_review(
                intent,
                account,
                market_data={"price": 16.15, "volume": 1_000_000, "bid": 16.14, "ask": 16.17},
                journal=journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            )
            action = queue_trade_action_from_order_payload(result, action_type="auto_buy", journal=journal)
            broker = FakeRobinhood()
            placed = handle_auto_buy_action(action["action_id"], broker, journal=journal, portfolio_path=self.tmp_dir / "portfolio.json")
            self.assertEqual(placed["status"], "PASS")
            self.assertEqual(
                placed["steps"],
                ["tradability", "review", "record_review", "tradability", "review", "record_review", "place", "record_submission"],
            )
            self.assertEqual([name for name, _ in broker.calls], ["tradability", "review", "tradability", "review", "place"])
            self.assertEqual(journal.get_execution_order_by_intent_id(intent.order_intent_id)["status"], "submitted")
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
                Config.ROBINHOOD_REVIEW_ONLY,
                Config.ROBINHOOD_DRY_RUN_ONLY,
                Config.ROBINHOOD_AGENTIC_ENABLED,
                Config.ROBINHOOD_KILL_SWITCH,
                Config.ROBINHOOD_CONTROL_FILE,
                Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
                Config.ROBINHOOD_AUTO_BUY_ENABLED,
                Config.EXECUTION_OFFICER_LLM_ENABLED,
            ) = old_config

    def test_agentic_execution_officer_uses_tools_then_places(self):
        from datetime import datetime, timezone
        from pathlib import Path

        from artha.config import Config
        from artha.execution import build_order_intent, prepare_and_record_robinhood_review
        from artha.openclaw_robinhood_handler import handle_auto_buy_action
        from artha.robinhood_bridge import queue_trade_action_from_order_payload, set_trading_disabled, write_robinhood_snapshot

        class FakeRobinhood:
            def __init__(self):
                self.calls = []

            def get_equity_quotes(self, **kwargs):
                self.calls.append(("quote", kwargs))
                return {
                    "data": {
                        "results": [
                            {
                                "quote": {
                                    "symbol": kwargs["symbols"][0],
                                    "last_trade_price": "16.15",
                                    "bid_price": "16.14",
                                    "ask_price": "16.17",
                                    "state": "active",
                                }
                            }
                        ]
                    }
                }

            def get_equity_tradability(self, **kwargs):
                self.calls.append(("tradability", kwargs))
                return {
                    "data": {
                        "results": [
                            {
                                "symbol": kwargs["symbols"][0],
                                "tradeable": True,
                                "state": "active",
                                "fractional_tradability": "tradable",
                            }
                        ]
                    }
                }

            def review_equity_order(self, **kwargs):
                self.calls.append(("review", kwargs))
                payload = dict(kwargs)
                payload["order_checks"] = {
                    "alertType": "EQUITY_SUITABILITY",
                    "equitySuitabilityAlertDetails": {"brokerageAccountType": "INDIVIDUAL"},
                }
                payload["market_data_disclosure"] = "Bid $16.14 x 1 P · Ask $16.17 x 1 M · Last $16.15 x 1. Updated 10:00 AM ET."
                return {"data": payload}

            def place_equity_order(self, **kwargs):
                self.calls.append(("place", kwargs))
                return {
                    "data": {
                        "order": {
                            "id": "rh-agentic-queued",
                            "symbol": kwargs["symbol"],
                            "side": kwargs["side"],
                            "state": "queued",
                            "quantity": kwargs.get("quantity"),
                            "created_at": "2026-06-04T15:00:00+00:00",
                        }
                    }
                }

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            Config.ROBINHOOD_REVIEW_ONLY,
            Config.ROBINHOOD_DRY_RUN_ONLY,
            Config.ROBINHOOD_AGENTIC_ENABLED,
            Config.ROBINHOOD_KILL_SWITCH,
            Config.ROBINHOOD_CONTROL_FILE,
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            Config.ROBINHOOD_AUTO_BUY_ENABLED,
            Config.EXECUTION_OFFICER_LLM_ENABLED,
            Config.EXECUTION_OFFICER_AGENTIC_ENABLED,
            Config.EXECUTION_OFFICER_AGENTIC_MAX_TOOL_STEPS,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            Config.ROBINHOOD_REVIEW_ONLY = False
            Config.ROBINHOOD_DRY_RUN_ONLY = False
            Config.ROBINHOOD_AGENTIC_ENABLED = True
            Config.ROBINHOOD_KILL_SWITCH = False
            Config.ROBINHOOD_CONTROL_FILE = str(self.tmp_dir / "control.json")
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE = str(self.tmp_dir / "robinhood_snapshot.json")
            Config.ROBINHOOD_AUTO_BUY_ENABLED = True
            Config.EXECUTION_OFFICER_LLM_ENABLED = True
            Config.EXECUTION_OFFICER_AGENTIC_ENABLED = True
            Config.EXECUTION_OFFICER_AGENTIC_MAX_TOOL_STEPS = 8
            set_trading_disabled(False, "unit test")
            write_robinhood_snapshot(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "robinhood_mcp",
                    "account": {
                        "account_number": "222233334",
                        "type": "cash",
                        "nickname": "Agentic",
                        "agentic_allowed": True,
                        "state": "active",
                    },
                    "portfolio": {"buying_power": 100, "cash": 100, "total_value": 100},
                    "positions": [],
                    "orders": [],
                },
                path=Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            )
            dossier_path = Path(self.tmp_dir / "TEST_dossier.json")
            dossier_path.write_text('{"ticker":"TEST","evidence":[{"id":"E001","source":"fmp.quote"}]}', encoding="utf-8")
            journal = self._journal_with_clean_buy_decision()
            account = {
                "account_number": "222233334",
                "type": "cash",
                "nickname": "Agentic",
                "agentic_allowed": True,
                "state": "active",
            }
            intent = build_order_intent(
                "TEST",
                "buy",
                quantity=1,
                notional=16.29,
                limit_price=16.29,
                estimated_price=16.15,
                decision_dossier_path=str(dossier_path),
            )
            intent.evidence = {
                "execution_officer": {
                    "auto_buy_eligible": True,
                    "execution_verdict": "BUY_READY",
                    "strategy": "WHOLE_SHARE_LIMIT",
                    "selected_candidate_id": "whole_share_marketable_limit",
                    "officer_model": "gpt-5.5",
                    "officer_reasoning_effort": "xhigh",
                    "officer_temperature": 2.0,
                }
            }
            result = prepare_and_record_robinhood_review(
                intent,
                account,
                market_data={"price": 16.15, "volume": 1_000_000, "bid": 16.14, "ask": 16.17},
                journal=journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            )
            action = queue_trade_action_from_order_payload(result, action_type="auto_buy", journal=journal)
            model_steps = [
                '{"tool_name":"read_decision_dossier","args":{},"reason":"Inspect evidence backing the order."}',
                '{"tool_name":"web_news_context","args":{},"reason":"Check for urgent ticker-specific news before money moves."}',
                '{"tool_name":"robinhood_get_quote","args":{},"reason":"Need live broker quote."}',
                '{"tool_name":"robinhood_get_tradability","args":{},"reason":"Need broker tradability."}',
                '{"tool_name":"robinhood_review_order","args":{},"reason":"Need exact broker review preview."}',
                '{"final_decision":{"allow_place":true,"confidence":9,"order_unchanged":true,"rationale":"Quote, tradability, review, and current-news check are clean; suitability alert is non-blocking.","evidence_refs":["read_decision_dossier","web_news_context","robinhood_get_quote","robinhood_get_tradability","robinhood_review_order"],"risk_flags":["EQUITY_SUITABILITY"],"missing_data":[]}}',
                '{"allow_place":true,"confidence":9,"rationale":"Second Robinhood review remains clear.","risk_flags":["EQUITY_SUITABILITY"],"requested_data":[]}',
            ]
            with (
                patch("artha.execution_officer._web_news_context", return_value={"ticker": "TEST", "results": []}),
                patch("artha.execution_officer.ChatGPTBackendClient.chat", side_effect=model_steps),
            ):
                broker = FakeRobinhood()
                placed = handle_auto_buy_action(action["action_id"], broker, journal=journal, portfolio_path=self.tmp_dir / "portfolio.json")
            self.assertEqual(placed["status"], "PASS")
            self.assertEqual([name for name, _ in broker.calls], ["quote", "tradability", "review", "tradability", "review", "place"])
            self.assertIn("agent_tool:web_news_context", placed["steps"])
            self.assertIn("agent_tool:robinhood_review_order", placed["steps"])
            self.assertEqual(journal.get_execution_order_by_intent_id(intent.order_intent_id)["status"], "submitted")
            stored = journal.get_trade_action(action["action_id"])
            self.assertIn("agentic_execution_officer", stored["result_json"])
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
                Config.ROBINHOOD_REVIEW_ONLY,
                Config.ROBINHOOD_DRY_RUN_ONLY,
                Config.ROBINHOOD_AGENTIC_ENABLED,
                Config.ROBINHOOD_KILL_SWITCH,
                Config.ROBINHOOD_CONTROL_FILE,
                Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
                Config.ROBINHOOD_AUTO_BUY_ENABLED,
                Config.EXECUTION_OFFICER_LLM_ENABLED,
                Config.EXECUTION_OFFICER_AGENTIC_ENABLED,
                Config.EXECUTION_OFFICER_AGENTIC_MAX_TOOL_STEPS,
            ) = old_config

    def test_openclaw_replayed_auto_buy_clearance_runs_agentic_officer(self):
        from datetime import datetime, timezone
        from pathlib import Path

        from artha.config import Config
        from artha.execution import build_order_intent, prepare_and_record_robinhood_review
        from artha.openclaw_robinhood_handler import run_agentic_auto_buy_clearance_from_responses
        from artha.robinhood_bridge import (
            build_auto_buy_operation,
            queue_trade_action_from_order_payload,
            run_final_clearance_for_action,
            set_trading_disabled,
            write_robinhood_snapshot,
        )

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            Config.ROBINHOOD_REVIEW_ONLY,
            Config.ROBINHOOD_DRY_RUN_ONLY,
            Config.ROBINHOOD_AGENTIC_ENABLED,
            Config.ROBINHOOD_KILL_SWITCH,
            Config.ROBINHOOD_CONTROL_FILE,
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            Config.ROBINHOOD_AUTO_BUY_ENABLED,
            Config.EXECUTION_OFFICER_LLM_ENABLED,
            Config.EXECUTION_OFFICER_AGENTIC_ENABLED,
            Config.EXECUTION_OFFICER_AGENTIC_MAX_TOOL_STEPS,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            Config.ROBINHOOD_REVIEW_ONLY = False
            Config.ROBINHOOD_DRY_RUN_ONLY = False
            Config.ROBINHOOD_AGENTIC_ENABLED = True
            Config.ROBINHOOD_KILL_SWITCH = False
            Config.ROBINHOOD_CONTROL_FILE = str(self.tmp_dir / "control.json")
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE = str(self.tmp_dir / "robinhood_snapshot.json")
            Config.ROBINHOOD_AUTO_BUY_ENABLED = True
            Config.EXECUTION_OFFICER_LLM_ENABLED = True
            Config.EXECUTION_OFFICER_AGENTIC_ENABLED = True
            Config.EXECUTION_OFFICER_AGENTIC_MAX_TOOL_STEPS = 8
            set_trading_disabled(False, "unit test")
            write_robinhood_snapshot(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "robinhood_mcp",
                    "account": {
                        "account_number": "222233334",
                        "type": "cash",
                        "nickname": "Agentic",
                        "agentic_allowed": True,
                        "state": "active",
                    },
                    "portfolio": {"buying_power": 100, "cash": 100, "total_value": 100},
                    "positions": [],
                    "orders": [],
                },
                path=Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            )
            dossier_path = Path(self.tmp_dir / "TEST_dossier.json")
            dossier_path.write_text('{"ticker":"TEST","evidence":[{"id":"E001","source":"fmp.quote"}]}', encoding="utf-8")
            journal = self._journal_with_clean_buy_decision()
            account = {
                "account_number": "222233334",
                "type": "cash",
                "nickname": "Agentic",
                "agentic_allowed": True,
                "state": "active",
            }
            intent = build_order_intent(
                "TEST",
                "buy",
                quantity=1,
                notional=16.29,
                limit_price=16.29,
                estimated_price=16.15,
                decision_dossier_path=str(dossier_path),
            )
            intent.evidence = {
                "execution_officer": {
                    "auto_buy_eligible": True,
                    "execution_verdict": "BUY_READY",
                    "strategy": "WHOLE_SHARE_LIMIT",
                    "selected_candidate_id": "whole_share_marketable_limit",
                    "officer_model": "gpt-5.5",
                    "officer_reasoning_effort": "xhigh",
                    "officer_temperature": 2.0,
                }
            }
            result = prepare_and_record_robinhood_review(
                intent,
                account,
                market_data={"price": 16.15, "volume": 1_000_000, "bid": 16.14, "ask": 16.17},
                journal=journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            )
            action = queue_trade_action_from_order_payload(result, action_type="auto_buy", journal=journal)
            quote = {"data": {"results": [{"quote": {"symbol": "TEST", "bid_price": "16.14", "ask_price": "16.17", "last_trade_price": "16.15"}}]}}
            tradability = {"data": {"results": [{"symbol": "TEST", "tradeable": True, "state": "active"}]}}
            review = {"data": {**build_auto_buy_operation(action["action_id"], journal=journal)["review_mcp_args"], "order_checks": {}}}
            journal.update_trade_action(action["action_id"], {"status": "review_ready"})
            model_steps = [
                '{"tool_name":"read_decision_dossier","args":{},"reason":"Inspect evidence."}',
                '{"tool_name":"robinhood_get_quote","args":{},"reason":"Confirm live quote."}',
                '{"tool_name":"robinhood_get_tradability","args":{},"reason":"Confirm tradability."}',
                '{"tool_name":"robinhood_review_order","args":{},"reason":"Preview exact order."}',
                '{"final_decision":{"allow_place":true,"confidence":9,"order_unchanged":true,"rationale":"Replay quote, tradability, and review are clean.","evidence_refs":["robinhood_get_quote","robinhood_get_tradability","robinhood_review_order"],"risk_flags":[],"missing_data":[]}}',
                '{"allow_place":true,"confidence":9,"rationale":"Final stored review remains clear.","risk_flags":[],"requested_data":[]}',
            ]
            with (
                patch("artha.execution_officer._web_news_context", return_value={"ticker": "TEST", "results": []}),
                patch("artha.execution_officer.ChatGPTBackendClient.chat", side_effect=model_steps),
            ):
                cleared = run_agentic_auto_buy_clearance_from_responses(
                    action["action_id"],
                    quote_response=quote,
                    tradability_response=tradability,
                    review_response=review,
                    journal=journal,
                )
                final = run_final_clearance_for_action(action["action_id"], journal=journal)
            self.assertEqual(cleared["status"], "PASS")
            self.assertEqual([name for name, _ in cleared["replayed_broker_calls"]], ["quote", "tradability", "review"])
            self.assertIn("agent_tool:robinhood_review_order", cleared["steps"])
            self.assertEqual(journal.get_trade_action(action["action_id"])["status"], "review_clear")
            self.assertTrue(final["allow_place"])
            place_op = build_auto_buy_operation(action["action_id"], journal=journal)
            self.assertEqual(place_op["operation"], "tradability_then_review_then_place_equity_order")
            self.assertIn("place_mcp_args", place_op)
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
                Config.ROBINHOOD_REVIEW_ONLY,
                Config.ROBINHOOD_DRY_RUN_ONLY,
                Config.ROBINHOOD_AGENTIC_ENABLED,
                Config.ROBINHOOD_KILL_SWITCH,
                Config.ROBINHOOD_CONTROL_FILE,
                Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
                Config.ROBINHOOD_AUTO_BUY_ENABLED,
                Config.EXECUTION_OFFICER_LLM_ENABLED,
                Config.EXECUTION_OFFICER_AGENTIC_ENABLED,
                Config.EXECUTION_OFFICER_AGENTIC_MAX_TOOL_STEPS,
            ) = old_config

    def test_agentic_prompt_surfaces_compact_broker_review_evidence(self):
        from artha.execution_officer import build_agentic_execution_prompt

        tool_trace = [
            {
                "step": 1,
                "tool_name": "robinhood_get_quote",
                "status": "PASS",
                "result": {
                    "data": {
                        "results": [
                            {
                                "quote": {
                                    "symbol": "TEST",
                                    "bid_price": "16.14",
                                    "ask_price": "16.17",
                                    "last_trade_price": "16.15",
                                }
                            }
                        ]
                    }
                },
            },
            {
                "step": 2,
                "tool_name": "robinhood_get_tradability",
                "status": "PASS",
                "result": {
                    "data": {
                        "results": [
                            {
                                "symbol": "TEST",
                                "tradeable": True,
                                "state": "active",
                                "fractional_tradability": "tradable",
                            }
                        ]
                    }
                },
            },
            {
                "step": 3,
                "tool_name": "robinhood_review_order",
                "status": "PASS",
                "result": {
                    "recorded_review": {
                        "review_gate": {
                            "passed": True,
                            "status": "PASS",
                            "reasons": [],
                            "checks": {
                                "tradability": {
                                    "passed": True,
                                    "status": "PASS",
                                    "checks": {
                                        "tradeable": True,
                                        "state": "active",
                                        "fractional_tradability": "tradable",
                                    },
                                },
                                "review_price_drift": {
                                    "passed": True,
                                    "status": "PASS",
                                    "checks": {"ask": 16.17, "reference_price": 16.15},
                                },
                                "order_checks_classification": {
                                    "blocking": False,
                                    "blocking_reasons": [],
                                    "has_checks": False,
                                },
                            },
                        }
                    }
                },
            },
        ]

        prompt = build_agentic_execution_prompt(
            action={"action_id": "ta_test", "ticker": "TEST", "action_type": "auto_buy"},
            operation={
                "operation": "auto_tradability_review_then_place_equity_order",
                "action_id": "ta_test",
                "review_mcp_args": {"symbol": "TEST", "side": "buy"},
                "tradability_mcp_args": {"symbols": ["TEST"]},
            },
            tool_trace=tool_trace,
            step=4,
        )

        self.assertIn("mandatory_broker_evidence", prompt)
        self.assertIn('"review_gate_passed": true', prompt)
        self.assertIn('"tradability_passed": true', prompt)
        self.assertIn('"fractional_tradability": "tradable"', prompt)
        self.assertIn('"blocking": false', prompt)

    def test_stage_b_reserves_broker_tools_and_corrects_false_missing_proof_denial(self):
        from datetime import datetime, timezone

        from artha.config import Config
        from artha.execution import build_order_intent, prepare_and_record_robinhood_review
        from artha.execution_officer import run_agentic_execution_officer
        from artha.robinhood_bridge import queue_trade_action_from_order_payload

        class FakeRobinhood:
            def __init__(self):
                self.calls = []

            def get_equity_quotes(self, **kwargs):
                self.calls.append("quote")
                return {
                    "data": {
                        "results": [
                            {"quote": {"symbol": "TEST", "bid_price": "99.95", "ask_price": "100.05", "last_trade_price": "100.00"}}
                        ]
                    }
                }

            def get_equity_tradability(self, **kwargs):
                self.calls.append("tradability")
                return {
                    "data": {
                        "results": [
                            {"symbol": "TEST", "tradeable": True, "state": "active", "fractional_tradability": "tradable"}
                        ]
                    }
                }

            def review_equity_order(self, **kwargs):
                self.calls.append("review")
                response = dict(kwargs)
                response["order_checks"] = {}
                response["market_data_disclosure"] = "Bid $99.95 x 100 P; Ask $100.05 x 100 P; Last $100.00 x 10 P."
                return {"data": response}

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.EXECUTION_OFFICER_LLM_ENABLED,
            Config.EXECUTION_OFFICER_AGENTIC_ENABLED,
            Config.EXECUTION_OFFICER_AGENTIC_MAX_TOOL_STEPS,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.EXECUTION_OFFICER_LLM_ENABLED = True
            Config.EXECUTION_OFFICER_AGENTIC_ENABLED = True
            Config.EXECUTION_OFFICER_AGENTIC_MAX_TOOL_STEPS = 4
            journal = self._journal_with_clean_buy_decision()
            intent = build_order_intent(
                "TEST",
                "buy",
                notional=20,
                limit_price=100,
                estimated_price=100,
                decision_dossier_path=str(self.tmp_dir / "TEST_dossier.json"),
            )
            intent.evidence = {
                "execution_officer": {
                    "auto_buy_eligible": True,
                    "execution_verdict": "BUY_READY",
                    "strategy": "FRACTIONAL_MARKET",
                }
            }
            prepared = prepare_and_record_robinhood_review(
                intent,
                {
                    "account_number": "222233334",
                    "type": "cash",
                    "nickname": "Agentic",
                    "agentic_allowed": True,
                    "state": "active",
                },
                market_data=self._market_data(),
                journal=journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            )
            action = queue_trade_action_from_order_payload(prepared, action_type="auto_buy", journal=journal)
            operation = {
                "success": True,
                "operation": "auto_tradability_review_then_place_equity_order",
                "action_id": action["action_id"],
                "review_mcp_args": {
                    "account_number": "222233334",
                    "symbol": "TEST",
                    "side": "buy",
                    "type": "market",
                    "time_in_force": "day",
                    "dollar_amount": "20.00",
                },
                "tradability_mcp_args": {"account_number": "222233334", "symbols": ["TEST"]},
            }
            model_steps = [
                '{"tool_name":"web_news_context","args":{},"reason":"optional research"}',
                '{"tool_name":"web_news_context","args":{},"reason":"optional research"}',
                '{"tool_name":"web_news_context","args":{},"reason":"optional research"}',
                '{"final_decision":{"allow_place":false,"confidence":8,"order_unchanged":true,"rationale":"Broker tradability and order review are not visible, so I cannot verify them.","evidence_refs":[],"risk_flags":[],"missing_data":["Robinhood review gate"]}}',
                '{"final_decision":{"allow_place":true,"confidence":9,"order_unchanged":true,"rationale":"The compact deterministic broker packet confirms quote, tradability, price drift, and order review all pass.","evidence_refs":["mandatory_broker_evidence"],"risk_flags":[],"missing_data":[]}}',
            ]
            with patch("artha.execution_officer.ChatGPTBackendClient.chat", side_effect=model_steps) as chat:
                result = run_agentic_execution_officer(
                    action=journal.get_trade_action(action["action_id"]),
                    operation=operation,
                    broker=FakeRobinhood(),
                    journal=journal,
                )
            self.assertTrue(result["allow_place"], result)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(chat.call_count, 5)
            self.assertEqual(
                [item["tool_name"] for item in result["tool_trace"]],
                ["robinhood_get_quote", "robinhood_get_tradability", "robinhood_review_order", "invalid_final_decision"],
            )
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.EXECUTION_OFFICER_LLM_ENABLED,
                Config.EXECUTION_OFFICER_AGENTIC_ENABLED,
                Config.EXECUTION_OFFICER_AGENTIC_MAX_TOOL_STEPS,
            ) = old_config

    def test_stage_b_rejects_empty_quote_even_when_other_broker_checks_pass(self):
        from artha.execution_officer import _broker_evidence_gate

        trace = [
            {"tool_name": "robinhood_get_quote", "status": "PASS", "result": {}},
            {"tool_name": "robinhood_get_tradability", "status": "PASS", "result": {"tradeable": True}},
            {
                "tool_name": "robinhood_review_order",
                "status": "PASS",
                "result": {
                    "recorded_review": {
                        "review_gate": {
                            "passed": True,
                            "status": "PASS",
                            "reasons": [],
                            "checks": {
                                "tradability": {"passed": True, "status": "PASS", "checks": {}},
                                "review_price_drift": {"passed": True, "status": "PASS", "checks": {}},
                                "order_checks_classification": {"blocking": False, "has_checks": False},
                            },
                        }
                    }
                },
            },
        ]
        gate = _broker_evidence_gate(trace)
        self.assertFalse(gate["passed"])
        self.assertIn("did not return an actual positive", " ".join(gate["reasons"]))

    def test_auto_buy_action_binds_current_thesis_to_action_and_order(self):
        from datetime import datetime, timezone

        from artha.execution import build_order_intent, prepare_and_record_robinhood_review
        from artha.robinhood_bridge import queue_trade_action_from_order_payload
        from artha.thesis_tracker import ThesisTracker

        journal = self._journal_with_clean_buy_decision()
        pending = ThesisTracker(journal).get_pending_for_ticker("TEST")
        self.assertIsNotNone(pending)
        intent = build_order_intent("TEST", "buy", notional=20, limit_price=100, estimated_price=100)
        intent.evidence = {
            "execution_officer": {
                "auto_buy_eligible": True,
                "execution_verdict": "BUY_READY",
                "strategy": "FRACTIONAL_MARKET",
            }
        }
        prepared = prepare_and_record_robinhood_review(
            intent,
            {
                "account_number": "222233334",
                "type": "cash",
                "nickname": "Agentic",
                "agentic_allowed": True,
                "state": "active",
            },
            market_data=self._market_data(),
            journal=journal,
            now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
        )
        action = queue_trade_action_from_order_payload(prepared, action_type="auto_buy", journal=journal)
        execution_row = journal.get_execution_order_by_intent_id(intent.order_intent_id)
        payload = json.loads(action["payload_json"]) if isinstance(action["payload_json"], str) else action["payload_json"]
        self.assertEqual(action["thesis_id"], pending.thesis_id)
        self.assertEqual((payload["intent"] or {}).get("thesis_id"), pending.thesis_id)
        self.assertEqual(execution_row["thesis_id"], pending.thesis_id)

    def test_auto_buy_authorization_blocks_when_no_current_thesis_exists(self):
        from datetime import datetime, timezone

        from artha.execution import build_order_intent, prepare_and_record_robinhood_review
        from artha.robinhood_bridge import _auto_buy_authorization_gate, queue_trade_action_from_order_payload
        from artha.thesis_tracker import ThesisTracker

        journal = self._journal_with_clean_buy_decision()
        tracker = ThesisTracker(journal)
        pending = tracker.get_pending_for_ticker("TEST")
        self.assertIsNotNone(pending)
        tracker.update_thesis_fields(pending.thesis_id, status="expired")
        intent = build_order_intent("TEST", "buy", notional=20, limit_price=100, estimated_price=100)
        intent.evidence = {
            "execution_officer": {
                "auto_buy_eligible": True,
                "execution_verdict": "BUY_READY",
                "strategy": "FRACTIONAL_MARKET",
            }
        }
        prepared = prepare_and_record_robinhood_review(
            intent,
            {
                "account_number": "222233334",
                "type": "cash",
                "nickname": "Agentic",
                "agentic_allowed": True,
                "state": "active",
            },
            market_data=self._market_data(),
            journal=journal,
            now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
        )
        action = queue_trade_action_from_order_payload(prepared, action_type="auto_buy", journal=journal)
        gate = _auto_buy_authorization_gate(journal.get_trade_action(action["action_id"]), journal)
        self.assertFalse(gate["passed"])
        self.assertIn("requires a matching pending or active investment thesis", " ".join(gate["reasons"]))

    def test_fresh_council_thesis_refresh_extends_pending_expiry(self):
        from datetime import datetime, timezone

        from artha.thesis_tracker import ThesisTracker

        journal = self._journal_with_clean_buy_decision()
        tracker = ThesisTracker(journal)
        pending = tracker.get_pending_for_ticker("TEST")
        self.assertIsNotNone(pending)
        tracker.update_thesis_fields(pending.thesis_id, pending_expiry="2026-08-07T16:54:05+00:00")
        self.assertTrue(tracker.refresh_pending_expiry(pending.thesis_id))
        refreshed = tracker.get(pending.thesis_id)
        self.assertGreater(datetime.fromisoformat(refreshed.pending_expiry), datetime.now(timezone.utc))

    def test_auto_buy_operation_does_not_retry_blocked_status(self):
        from datetime import datetime, timezone

        from artha.config import Config
        from artha.execution import build_order_intent, prepare_and_record_robinhood_review
        from artha.robinhood_bridge import build_auto_buy_operation, queue_trade_action_from_order_payload, set_trading_disabled, write_robinhood_snapshot

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            Config.ROBINHOOD_REVIEW_ONLY,
            Config.ROBINHOOD_DRY_RUN_ONLY,
            Config.ROBINHOOD_AGENTIC_ENABLED,
            Config.ROBINHOOD_KILL_SWITCH,
            Config.ROBINHOOD_CONTROL_FILE,
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            Config.ROBINHOOD_AUTO_BUY_ENABLED,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            Config.ROBINHOOD_REVIEW_ONLY = False
            Config.ROBINHOOD_DRY_RUN_ONLY = False
            Config.ROBINHOOD_AGENTIC_ENABLED = True
            Config.ROBINHOOD_KILL_SWITCH = False
            Config.ROBINHOOD_CONTROL_FILE = str(self.tmp_dir / "control.json")
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE = str(self.tmp_dir / "robinhood_snapshot.json")
            Config.ROBINHOOD_AUTO_BUY_ENABLED = True
            set_trading_disabled(False, "unit test")
            write_robinhood_snapshot(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "robinhood_mcp",
                    "account": {"account_number": "222233334", "type": "cash", "nickname": "Agentic", "agentic_allowed": True, "state": "active"},
                    "portfolio": {"buying_power": 100, "cash": 100, "total_value": 100},
                    "positions": [],
                    "orders": [],
                },
                path=Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            )
            journal = self._journal_with_clean_buy_decision()
            account = {"account_number": "222233334", "type": "cash", "nickname": "Agentic", "agentic_allowed": True, "state": "active"}
            intent = build_order_intent("TEST", "buy", quantity=1, notional=16.29, limit_price=16.29, estimated_price=16.15)
            intent.evidence = {
                "execution_officer": {
                    "auto_buy_eligible": True,
                    "execution_verdict": "BUY_READY",
                    "strategy": "WHOLE_SHARE_LIMIT",
                }
            }
            result = prepare_and_record_robinhood_review(
                intent,
                account,
                market_data={"price": 16.15, "volume": 1_000_000, "bid": 16.14, "ask": 16.17},
                journal=journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            )
            action = queue_trade_action_from_order_payload(result, action_type="auto_buy", journal=journal)
            journal.update_trade_action(action["action_id"], {"status": "review_blocked"})
            op = build_auto_buy_operation(action["action_id"], journal=journal)
            self.assertFalse(op["success"])
            self.assertEqual(op["operation"], "blocked")
            self.assertIn("status is review_blocked", op["message"])
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
                Config.ROBINHOOD_REVIEW_ONLY,
                Config.ROBINHOOD_DRY_RUN_ONLY,
                Config.ROBINHOOD_AGENTIC_ENABLED,
                Config.ROBINHOOD_KILL_SWITCH,
                Config.ROBINHOOD_CONTROL_FILE,
                Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
                Config.ROBINHOOD_AUTO_BUY_ENABLED,
            ) = old_config

    def test_auto_buy_queue_keeps_actions_visible_after_review_request(self):
        from datetime import datetime, timezone

        from artha.config import Config
        from artha.execution import build_order_intent, prepare_and_record_robinhood_review
        from artha.robinhood_bridge import (
            build_pending_auto_buy_queue_status,
            queue_trade_action_from_order_payload,
            set_trading_disabled,
            write_robinhood_snapshot,
        )

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            Config.ROBINHOOD_REVIEW_ONLY,
            Config.ROBINHOOD_DRY_RUN_ONLY,
            Config.ROBINHOOD_AGENTIC_ENABLED,
            Config.ROBINHOOD_KILL_SWITCH,
            Config.ROBINHOOD_CONTROL_FILE,
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            Config.ROBINHOOD_AUTO_BUY_ENABLED,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            Config.ROBINHOOD_REVIEW_ONLY = False
            Config.ROBINHOOD_DRY_RUN_ONLY = False
            Config.ROBINHOOD_AGENTIC_ENABLED = True
            Config.ROBINHOOD_KILL_SWITCH = False
            Config.ROBINHOOD_CONTROL_FILE = str(self.tmp_dir / "control.json")
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE = str(self.tmp_dir / "robinhood_snapshot.json")
            Config.ROBINHOOD_AUTO_BUY_ENABLED = True
            set_trading_disabled(False, "unit test")
            write_robinhood_snapshot(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "robinhood_mcp",
                    "account": {
                        "account_number": "222233334",
                        "type": "cash",
                        "nickname": "Agentic",
                        "agentic_allowed": True,
                        "state": "active",
                    },
                    "portfolio": {"buying_power": 100, "cash": 100, "total_value": 100},
                    "positions": [],
                    "orders": [],
                },
                path=Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            )
            journal = self._journal_with_clean_buy_decision()
            account = {
                "account_number": "222233334",
                "type": "cash",
                "nickname": "Agentic",
                "agentic_allowed": True,
                "state": "active",
            }
            intent = build_order_intent(
                "TEST",
                "buy",
                notional=17.85,
                estimated_price=100.0,
                decision_dossier_path=str(self.tmp_dir / "TEST_dossier.json"),
            )
            intent.evidence = {
                "execution_officer": {
                    "auto_buy_eligible": True,
                    "execution_verdict": "BUY_READY",
                    "strategy": "FRACTIONAL_MARKET",
                    "selected_candidate_id": "fractional_market_notional",
                    "officer_model": "gpt-5.5",
                    "officer_reasoning_effort": "xhigh",
                    "officer_temperature": 0.2,
                }
            }
            result = prepare_and_record_robinhood_review(
                intent,
                account,
                market_data={"price": 100.0, "volume": 1_000_000, "bid": 99.95, "ask": 100.05},
                journal=journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            )
            action = queue_trade_action_from_order_payload(result, action_type="auto_buy", journal=journal)
            journal.update_trade_action(action["action_id"], {"status": "auto_review_requested"})
            self.assertEqual(journal.get_trade_action(action["action_id"])["status"], "auto_review_requested")

            queue = build_pending_auto_buy_queue_status(journal=journal)
            self.assertEqual(queue["operation_count"], 1)
            self.assertEqual(queue["actions"][0]["action_id"], action["action_id"])
            self.assertEqual(queue["actions"][0]["status"], "auto_review_requested")
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
                Config.ROBINHOOD_REVIEW_ONLY,
                Config.ROBINHOOD_DRY_RUN_ONLY,
                Config.ROBINHOOD_AGENTIC_ENABLED,
                Config.ROBINHOOD_KILL_SWITCH,
                Config.ROBINHOOD_CONTROL_FILE,
                Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
                Config.ROBINHOOD_AUTO_BUY_ENABLED,
            ) = old_config

    def test_auto_buy_daily_dollar_cap_pauses_buys(self):
        from datetime import datetime, timezone

        from artha.config import Config
        from artha.execution import build_order_intent, prepare_and_record_robinhood_review
        from artha.robinhood_bridge import build_auto_buy_operation, queue_trade_action_from_order_payload, set_trading_disabled, write_robinhood_snapshot

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            Config.ROBINHOOD_REVIEW_ONLY,
            Config.ROBINHOOD_DRY_RUN_ONLY,
            Config.ROBINHOOD_AGENTIC_ENABLED,
            Config.ROBINHOOD_KILL_SWITCH,
            Config.ROBINHOOD_CONTROL_FILE,
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            Config.ROBINHOOD_AUTO_BUY_ENABLED,
            Config.ROBINHOOD_AUTO_BUY_MAX_DAILY_DOLLARS,
            Config.ARTHA_AUTO_PAUSE_BUYS_ON_LIMIT,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            Config.ROBINHOOD_REVIEW_ONLY = False
            Config.ROBINHOOD_DRY_RUN_ONLY = False
            Config.ROBINHOOD_AGENTIC_ENABLED = True
            Config.ROBINHOOD_KILL_SWITCH = False
            Config.ROBINHOOD_CONTROL_FILE = str(self.tmp_dir / "control.json")
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE = str(self.tmp_dir / "robinhood_snapshot.json")
            Config.ROBINHOOD_AUTO_BUY_ENABLED = True
            Config.ROBINHOOD_AUTO_BUY_MAX_DAILY_DOLLARS = 50.0
            Config.ARTHA_AUTO_PAUSE_BUYS_ON_LIMIT = True
            set_trading_disabled(False, "unit test")
            write_robinhood_snapshot(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "robinhood_mcp",
                    "account": {"account_number": "222233334", "type": "cash", "nickname": "Agentic", "agentic_allowed": True, "state": "active"},
                    "portfolio": {"buying_power": 100, "cash": 100, "total_value": 100},
                    "positions": [],
                    "orders": [],
                },
                path=Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            )
            journal = self._journal_with_clean_buy_decision()
            journal.save_execution_order(
                {
                    "order_intent_id": "existing-auto-buy",
                    "ticker": "OLD",
                    "side": "buy",
                    "order_type": "market",
                    "status": "filled",
                    "dry_run": False,
                    "submitted_at": datetime.now(timezone.utc).isoformat(),
                    "filled_at": datetime.now(timezone.utc).isoformat(),
                    "notional": 45.0,
                    "quantity": 1.0,
                    "estimated_price": 45.0,
                    "request_json": {},
                    "response_json": {},
                    "notes": "unit existing buy",
                }
            )
            account = {"account_number": "222233334", "type": "cash", "nickname": "Agentic", "agentic_allowed": True, "state": "active"}
            intent = build_order_intent("TEST", "buy", quantity=1, notional=16.29, limit_price=16.29, estimated_price=16.15)
            intent.evidence = {
                "execution_officer": {
                    "auto_buy_eligible": True,
                    "execution_verdict": "BUY_READY",
                    "strategy": "WHOLE_SHARE_LIMIT",
                }
            }
            result = prepare_and_record_robinhood_review(
                intent,
                account,
                market_data={"price": 16.15, "volume": 1_000_000, "bid": 16.14, "ask": 16.17},
                journal=journal,
                now=datetime.now(timezone.utc),
            )
            action = queue_trade_action_from_order_payload(result, action_type="auto_buy", journal=journal)
            with patch(
                "artha.robinhood_bridge.maybe_pause_buying_for_limit",
                return_value={"success": True, "operation": "buy_pause", "reason": "Auto-buy daily cap would be exceeded."},
            ) as maybe_stop:
                op = build_auto_buy_operation(action["action_id"], journal=journal)

            self.assertFalse(op["success"])
            self.assertEqual(op["operation"], "blocked")
            self.assertIn("Auto-buy daily cap would be exceeded", op["message"])
            self.assertEqual(op["auto_trade_gate"]["checks"]["buy_pause"]["operation"], "buy_pause")
            maybe_stop.assert_called_once()
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
                Config.ROBINHOOD_REVIEW_ONLY,
                Config.ROBINHOOD_DRY_RUN_ONLY,
                Config.ROBINHOOD_AGENTIC_ENABLED,
                Config.ROBINHOOD_KILL_SWITCH,
                Config.ROBINHOOD_CONTROL_FILE,
                Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
                Config.ROBINHOOD_AUTO_BUY_ENABLED,
                Config.ROBINHOOD_AUTO_BUY_MAX_DAILY_DOLLARS,
                Config.ARTHA_AUTO_PAUSE_BUYS_ON_LIMIT,
            ) = old_config

    def test_agentic_execution_officer_blocks_final_without_required_tools(self):
        from datetime import datetime, timezone

        from artha.config import Config
        from artha.execution import build_order_intent, prepare_and_record_robinhood_review
        from artha.execution_officer import run_agentic_execution_officer
        from artha.robinhood_bridge import build_auto_buy_operation, queue_trade_action_from_order_payload, set_trading_disabled, write_robinhood_snapshot

        class FakeRobinhood:
            def __init__(self):
                self.calls = []

            def get_equity_quotes(self, **kwargs):
                self.calls.append(("quote", kwargs))
                return {}

            def get_equity_tradability(self, **kwargs):
                self.calls.append(("tradability", kwargs))
                return {}

            def review_equity_order(self, **kwargs):
                self.calls.append(("review", kwargs))
                return {}

            def place_equity_order(self, **kwargs):
                self.calls.append(("place", kwargs))
                return {}

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            Config.ROBINHOOD_REVIEW_ONLY,
            Config.ROBINHOOD_DRY_RUN_ONLY,
            Config.ROBINHOOD_AGENTIC_ENABLED,
            Config.ROBINHOOD_KILL_SWITCH,
            Config.ROBINHOOD_CONTROL_FILE,
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            Config.ROBINHOOD_AUTO_BUY_ENABLED,
            Config.EXECUTION_OFFICER_LLM_ENABLED,
            Config.EXECUTION_OFFICER_AGENTIC_ENABLED,
            Config.EXECUTION_OFFICER_AGENTIC_MAX_TOOL_STEPS,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            Config.ROBINHOOD_REVIEW_ONLY = False
            Config.ROBINHOOD_DRY_RUN_ONLY = False
            Config.ROBINHOOD_AGENTIC_ENABLED = True
            Config.ROBINHOOD_KILL_SWITCH = False
            Config.ROBINHOOD_CONTROL_FILE = str(self.tmp_dir / "control.json")
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE = str(self.tmp_dir / "robinhood_snapshot.json")
            Config.ROBINHOOD_AUTO_BUY_ENABLED = True
            Config.EXECUTION_OFFICER_LLM_ENABLED = True
            Config.EXECUTION_OFFICER_AGENTIC_ENABLED = True
            Config.EXECUTION_OFFICER_AGENTIC_MAX_TOOL_STEPS = 1
            set_trading_disabled(False, "unit test")
            write_robinhood_snapshot(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "robinhood_mcp",
                    "account": {
                        "account_number": "222233334",
                        "type": "cash",
                        "nickname": "Agentic",
                        "agentic_allowed": True,
                        "state": "active",
                    },
                    "portfolio": {"buying_power": 100, "cash": 100, "total_value": 100},
                    "positions": [],
                    "orders": [],
                },
                path=Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            )
            journal = self._journal_with_clean_buy_decision()
            account = {
                "account_number": "222233334",
                "type": "cash",
                "nickname": "Agentic",
                "agentic_allowed": True,
                "state": "active",
            }
            intent = build_order_intent("TEST", "buy", quantity=1, notional=16.29, limit_price=16.29, estimated_price=16.15)
            intent.evidence = {
                "execution_officer": {
                    "auto_buy_eligible": True,
                    "execution_verdict": "BUY_READY",
                    "strategy": "WHOLE_SHARE_LIMIT",
                    "selected_candidate_id": "whole_share_marketable_limit",
                }
            }
            result = prepare_and_record_robinhood_review(
                intent,
                account,
                market_data={"price": 16.15, "volume": 1_000_000, "bid": 16.14, "ask": 16.17},
                journal=journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            )
            action = queue_trade_action_from_order_payload(result, action_type="auto_buy", journal=journal)
            operation = build_auto_buy_operation(action["action_id"], journal=journal)
            self.assertTrue(operation["success"], operation)
            with patch(
                "artha.execution_officer.ChatGPTBackendClient.chat",
                return_value='{"final_decision":{"allow_place":true,"confidence":9,"order_unchanged":true,"rationale":"Approve immediately.","evidence_refs":[],"risk_flags":[],"missing_data":[]}}',
            ):
                broker = FakeRobinhood()
                blocked = run_agentic_execution_officer(
                    action=journal.get_trade_action(action["action_id"]),
                    operation=operation,
                    broker=broker,
                    journal=journal,
                )
            self.assertEqual(blocked["status"], "BLOCKED")
            self.assertEqual(broker.calls, [])
            self.assertIn("required live tools", blocked["reason"])
            self.assertNotEqual(journal.get_execution_order_by_intent_id(intent.order_intent_id)["status"], "submitted")
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
                Config.ROBINHOOD_REVIEW_ONLY,
                Config.ROBINHOOD_DRY_RUN_ONLY,
                Config.ROBINHOOD_AGENTIC_ENABLED,
                Config.ROBINHOOD_KILL_SWITCH,
                Config.ROBINHOOD_CONTROL_FILE,
                Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
                Config.ROBINHOOD_AUTO_BUY_ENABLED,
                Config.EXECUTION_OFFICER_LLM_ENABLED,
                Config.EXECUTION_OFFICER_AGENTIC_ENABLED,
                Config.EXECUTION_OFFICER_AGENTIC_MAX_TOOL_STEPS,
            ) = old_config

    def test_openclaw_auto_buy_handler_blocks_broker_order_checks(self):
        from datetime import datetime, timezone

        from artha.config import Config
        from artha.execution import build_order_intent, prepare_and_record_robinhood_review
        from artha.openclaw_robinhood_handler import handle_auto_buy_action
        from artha.robinhood_bridge import queue_trade_action_from_order_payload, set_trading_disabled, write_robinhood_snapshot

        class FakeRobinhood:
            def __init__(self):
                self.calls = []

            def get_equity_tradability(self, **kwargs):
                self.calls.append(("tradability", kwargs))
                return {"data": {"results": [{"symbol": kwargs["symbols"][0], "tradeable": True, "state": "active"}]}}

            def review_equity_order(self, **kwargs):
                self.calls.append(("review", kwargs))
                payload = dict(kwargs)
                payload["order_checks"] = {"warning": "unit broker warning"}
                return {"data": payload}

            def place_equity_order(self, **kwargs):
                self.calls.append(("place", kwargs))
                return {"data": {"order": {"id": "should-not-place", "state": "queued"}}}

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            Config.ROBINHOOD_REVIEW_ONLY,
            Config.ROBINHOOD_DRY_RUN_ONLY,
            Config.ROBINHOOD_AGENTIC_ENABLED,
            Config.ROBINHOOD_KILL_SWITCH,
            Config.ROBINHOOD_CONTROL_FILE,
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            Config.ROBINHOOD_AUTO_BUY_ENABLED,
            Config.EXECUTION_OFFICER_LLM_ENABLED,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            Config.ROBINHOOD_REVIEW_ONLY = False
            Config.ROBINHOOD_DRY_RUN_ONLY = False
            Config.ROBINHOOD_AGENTIC_ENABLED = True
            Config.ROBINHOOD_KILL_SWITCH = False
            Config.ROBINHOOD_CONTROL_FILE = str(self.tmp_dir / "control.json")
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE = str(self.tmp_dir / "robinhood_snapshot.json")
            Config.ROBINHOOD_AUTO_BUY_ENABLED = True
            Config.EXECUTION_OFFICER_LLM_ENABLED = False
            set_trading_disabled(False, "unit test")
            write_robinhood_snapshot(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "robinhood_mcp",
                    "account": {
                        "account_number": "222233334",
                        "type": "cash",
                        "nickname": "Agentic",
                        "agentic_allowed": True,
                        "state": "active",
                    },
                    "portfolio": {"buying_power": 100, "cash": 100, "total_value": 100},
                    "positions": [],
                    "orders": [],
                },
                path=Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            )
            journal = self._journal_with_clean_buy_decision()
            account = {
                "account_number": "222233334",
                "type": "cash",
                "nickname": "Agentic",
                "agentic_allowed": True,
                "state": "active",
            }
            intent = build_order_intent("TEST", "buy", quantity=1, notional=16.29, limit_price=16.29, estimated_price=16.15)
            intent.evidence = {
                "execution_officer": {
                    "auto_buy_eligible": True,
                    "execution_verdict": "BUY_READY",
                    "strategy": "WHOLE_SHARE_LIMIT",
                    "selected_candidate_id": "whole_share_marketable_limit",
                }
            }
            result = prepare_and_record_robinhood_review(
                intent,
                account,
                market_data={"price": 16.15, "volume": 1_000_000, "bid": 16.14, "ask": 16.17},
                journal=journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            )
            action = queue_trade_action_from_order_payload(result, action_type="auto_buy", journal=journal)
            broker = FakeRobinhood()
            blocked = handle_auto_buy_action(action["action_id"], broker, journal=journal, portfolio_path=self.tmp_dir / "portfolio.json")
            self.assertEqual(blocked["status"], "BLOCKED")
            self.assertEqual([name for name, _ in broker.calls], ["tradability", "review"])
            self.assertEqual(journal.get_trade_action(action["action_id"])["status"], "review_blocked")
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
                Config.ROBINHOOD_REVIEW_ONLY,
                Config.ROBINHOOD_DRY_RUN_ONLY,
                Config.ROBINHOOD_AGENTIC_ENABLED,
                Config.ROBINHOOD_KILL_SWITCH,
                Config.ROBINHOOD_CONTROL_FILE,
                Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
                Config.ROBINHOOD_AUTO_BUY_ENABLED,
                Config.EXECUTION_OFFICER_LLM_ENABLED,
            ) = old_config

    def test_scheduled_scan_moves_to_1130_and_warm_scan_moves_to_0900_ct(self):
        from zoneinfo import ZoneInfo

        from artha.config import Config
        from artha.scheduler import ArthaScheduler, MarketHours

        old_config = (
            Config.SCHEDULED_SCAN_HOUR_CT,
            Config.SCHEDULED_SCAN_MINUTE_CT,
            Config.SCHEDULED_SCAN_CATCHUP_MINUTES,
            Config.DAILY_WARM_SCAN_HOUR_CT,
            Config.DAILY_WARM_SCAN_MINUTE_CT,
            Config.DAILY_WARM_SCAN_CATCHUP_MINUTES,
        )
        try:
            Config.SCHEDULED_SCAN_HOUR_CT = 11
            Config.SCHEDULED_SCAN_MINUTE_CT = 30
            Config.SCHEDULED_SCAN_CATCHUP_MINUTES = 90
            Config.DAILY_WARM_SCAN_HOUR_CT = 9
            Config.DAILY_WARM_SCAN_MINUTE_CT = 0
            Config.DAILY_WARM_SCAN_CATCHUP_MINUTES = 150

            scheduler = ArthaScheduler.__new__(ArthaScheduler)
            scheduler.market_hours = MarketHours()
            scheduler.ct_tz = ZoneInfo("America/Chicago")
            scheduler._last_run = {}
            scheduler._mark_run_slot = (
                lambda name, slot: scheduler._last_run.__setitem__(name, slot)
            )

            def utc_at(year: int, month: int, day: int, hour: int, minute: int):
                return datetime(year, month, day, hour, minute, tzinfo=scheduler.ct_tz).astimezone(timezone.utc)

            self.assertTrue(scheduler._should_run_weekly_scan(utc_at(2026, 6, 10, 11, 30)))
            self.assertFalse(scheduler._should_run_weekly_scan(utc_at(2026, 6, 10, 11, 31)))

            scheduler._last_run = {}
            self.assertTrue(scheduler._should_run_weekly_scan(utc_at(2026, 6, 10, 12, 15)))

            scheduler._last_run = {}
            self.assertFalse(scheduler._should_run_weekly_scan(utc_at(2026, 6, 10, 13, 0)))
            self.assertFalse(scheduler._should_run_weekly_scan(utc_at(2026, 6, 10, 14, 30)))
            self.assertFalse(scheduler._should_run_weekly_scan(utc_at(2026, 6, 13, 11, 30)))
            self.assertFalse(scheduler._should_run_weekly_scan(utc_at(2026, 6, 19, 11, 30)))

            scheduler._last_run = {}
            self.assertTrue(scheduler._should_run_daily_warm_scan(utc_at(2026, 6, 10, 9, 0)))
            self.assertFalse(scheduler._should_run_daily_warm_scan(utc_at(2026, 6, 10, 9, 1)))

            scheduler._last_run = {}
            self.assertTrue(scheduler._should_run_daily_warm_scan(utc_at(2026, 6, 10, 10, 44)))

            scheduler._last_run = {}
            self.assertFalse(scheduler._should_run_daily_warm_scan(utc_at(2026, 6, 10, 11, 30)))
            self.assertFalse(scheduler._should_run_daily_warm_scan(utc_at(2026, 6, 10, 12, 0)))
        finally:
            (
                Config.SCHEDULED_SCAN_HOUR_CT,
                Config.SCHEDULED_SCAN_MINUTE_CT,
                Config.SCHEDULED_SCAN_CATCHUP_MINUTES,
                Config.DAILY_WARM_SCAN_HOUR_CT,
                Config.DAILY_WARM_SCAN_MINUTE_CT,
                Config.DAILY_WARM_SCAN_CATCHUP_MINUTES,
            ) = old_config

    def test_scan_report_audit_includes_failed_candidates(self):
        from artha.scheduler import ArthaScheduler

        scheduler = ArthaScheduler.__new__(ArthaScheduler)
        captured: dict[str, str] = {}

        def fake_write_text(path_obj, text, encoding=None):
            captured[str(path_obj)] = text
            return len(text)

        with patch.object(Path, "mkdir", lambda *args, **kwargs: None), patch.object(Path, "write_text", fake_write_text):
            path = scheduler._save_scan_reports(
                "unit-scan",
                "HEADER",
                [("GOOD", "good report")],
                [("BAD", "bad failure")],
            )

        self.assertTrue(path.endswith("unit-scan.txt"))
        body = next(iter(captured.values()))
        self.assertIn("===== GOOD =====", body)
        self.assertIn("good report", body)
        self.assertIn("===== BAD FAILED =====", body)
        self.assertIn("bad failure", body)

    def test_fractional_scan_buy_with_missing_bid_ask_becomes_entry_watch(self):
        from types import SimpleNamespace

        from artha.config import Config
        from artha.scheduler import ArthaScheduler

        old_config = (
            Config.SCAN_PREPARE_ROBINHOOD_REVIEW_FOR_BUYS,
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.EXECUTION_OFFICER_LLM_ENABLED,
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
        )
        try:
            Config.SCAN_PREPARE_ROBINHOOD_REVIEW_FOR_BUYS = True
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.EXECUTION_OFFICER_LLM_ENABLED = False
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE = str(self._write_agentic_snapshot_file(cash=100.0))
            journal = self._journal_with_clean_buy_decision()
            scheduler = ArthaScheduler()
            scheduler.market_hours = SimpleNamespace(is_market_open=lambda dt=None: True)
            decision = SimpleNamespace(
                ticker="UTHR",
                final_verdict="STARTER",
                recommended_action=(
                    "STARTER — Place a limit buy for 0.04 fractional shares at $546 or better "
                    "and re-review only if price remains near the entry zone."
                ),
                synthesis_report="",
                recommended_allocation_pct=6.0,
                opportunity_score=78,
                adjusted_score=78,
                confidence=7,
                dossier_path=str(self.tmp_dir / "UTHR_dossier.json"),
                invalidation_conditions=["unit"],
                entry_valid_until=(datetime.now(timezone.utc) + timedelta(days=30)).date().isoformat(),
                agentic_trace={},
            )
            stock_data = {
                "quote": {"price": 545.17, "volume": 2_000_000},
                "yf_quote": {"price": 545.17, "volume": 2_000_000},
            }

            result = scheduler._prepare_scan_buy_robinhood_review(
                "UTHR",
                decision,
                stock_data,
                journal,
                nav=368.0,
                recommendation_id=161,
            )

            self.assertIsNotNone(result)
            self.assertEqual(result["broker_result"]["status"], "entry_watch")
            self.assertEqual(result["guardrails"]["status"], "ENTRY_WATCH")
            self.assertIn("bid/ask", " ".join(result["guardrails"]["reasons"]).lower())
            self.assertEqual(journal.get_execution_orders(limit=5), [])
            watches = journal.get_active_defer_watches_for_ticker("UTHR")
            self.assertGreaterEqual(len(watches), 1)
            self.assertIn("Broker-aware fractional entry watch", watches[0]["notes"])
            msg, reply_markup = scheduler._format_scan_order_review_summary([result])
            self.assertIn("entry_watch", msg)
            self.assertIn("No button", msg)
            self.assertIsNone(reply_markup)
            eo_msg = scheduler._format_execution_officer_scan_update("UTHR", decision, result)
            self.assertIn("Execution verdict: WAIT / NO BUY NOW", eo_msg)
            self.assertIn("bid/ask", eo_msg.lower())
            self.assertIn("Entry watch", eo_msg)
        finally:
            (
                Config.SCAN_PREPARE_ROBINHOOD_REVIEW_FOR_BUYS,
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.EXECUTION_OFFICER_LLM_ENABLED,
                Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE,
            ) = old_config

    def test_fractional_pullback_accumulate_above_reference_becomes_entry_watch(self):
        from types import SimpleNamespace

        from artha.config import Config
        from artha.scheduler import ArthaScheduler

        old_config = (
            Config.SCAN_PREPARE_ROBINHOOD_REVIEW_FOR_BUYS,
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.EXECUTION_OFFICER_LLM_ENABLED,
        )
        try:
            Config.SCAN_PREPARE_ROBINHOOD_REVIEW_FOR_BUYS = True
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.EXECUTION_OFFICER_LLM_ENABLED = False
            journal = self._journal_with_clean_buy_decision()
            scheduler = ArthaScheduler()
            scheduler.market_hours = SimpleNamespace(is_market_open=lambda dt=None: True)
            decision = SimpleNamespace(
                ticker="KRYS",
                final_verdict="ACCUMULATE",
                recommended_action=(
                    "ACCUMULATE — place fractional staged limit orders only: 0.06 KRYS at $283.25 "
                    "and 0.04 KRYS at $260.00; do not chase at the ~$300.58 market price."
                ),
                synthesis_report="",
                recommended_allocation_pct=7.5,
                opportunity_score=68,
                adjusted_score=68,
                confidence=7,
                dossier_path=str(self.tmp_dir / "KRYS_dossier.json"),
                invalidation_conditions=["unit"],
                entry_valid_until=(datetime.now(timezone.utc) + timedelta(days=30)).date().isoformat(),
                agentic_trace={},
            )
            stock_data = {
                "quote": {"price": 300.58, "volume": 2_000_000, "bid": 297.25, "ask": 309.0},
                "yf_quote": {"price": 300.58, "volume": 2_000_000, "bid": 297.25, "ask": 309.0},
            }

            result = scheduler._prepare_scan_buy_robinhood_review(
                "KRYS",
                decision,
                stock_data,
                journal,
                nav=368.0,
                recommendation_id=162,
            )

            self.assertIsNotNone(result)
            self.assertEqual(result["broker_result"]["status"], "entry_watch")
            joined = " ".join(result["guardrails"]["reasons"]).lower()
            self.assertIn("above artha reference", joined)
            self.assertEqual(journal.get_execution_orders(limit=5), [])
            watches = journal.get_active_defer_watches_for_ticker("KRYS")
            self.assertGreaterEqual(len(watches), 2)
            self.assertTrue(any(280 <= row["zone_low"] <= 286 for row in watches))
            self.assertTrue(any(257 <= row["zone_low"] <= 263 for row in watches))
        finally:
            (
                Config.SCAN_PREPARE_ROBINHOOD_REVIEW_FOR_BUYS,
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.EXECUTION_OFFICER_LLM_ENABLED,
            ) = old_config

    def test_pending_order_recheck_queue_roundtrip(self):
        from datetime import datetime, timedelta, timezone

        from artha.journal import DecisionJournal

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        now = datetime.now(timezone.utc)
        row_id = journal.save_pending_order_recheck(
            {
                "recheck_id": "unit-recheck",
                "run_after": (now - timedelta(minutes=1)).isoformat(),
                "expires_at": (now + timedelta(days=1)).isoformat(),
                "status": "pending",
                "ticker": "TEST",
                "original_verdict": "STARTER",
                "original_action": "STARTER at $10",
                "original_price": 10.0,
                "max_price": 10.0,
                "notional": 17.5,
                "account_number_masked": "****3334",
            }
        )
        due = journal.get_due_pending_order_rechecks(now.isoformat())
        self.assertEqual(row_id, due[0]["id"])
        self.assertEqual(due[0]["ticker"], "TEST")
        journal.update_pending_order_recheck("unit-recheck", {"status": "review_ready", "notes": "unit"})
        self.assertEqual(journal.get_pending_order_rechecks(limit=1)[0]["status"], "review_ready")

    def test_market_open_recheck_runs_council_and_queues_auto_buy_without_buttons(self):
        import asyncio
        from datetime import datetime, timedelta, timezone
        from types import SimpleNamespace

        from artha.config import Config
        from artha.journal import DecisionJournal
        from artha.scheduler import ArthaScheduler

        class FakeTelegram:
            enabled = True

            def __init__(self):
                self.messages = []

            def send_message(self, text, parse_mode=None, silent=False, reply_markup=None):
                self.messages.append((text, parse_mode, silent, reply_markup))
                return True

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            Config.EXECUTION_OFFICER_LLM_ENABLED,
            Config.ROBINHOOD_AUTO_BUY_ENABLED,
        )
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
            Config.EXECUTION_OFFICER_LLM_ENABLED = False
            Config.ROBINHOOD_AUTO_BUY_ENABLED = True
            now = datetime.now(timezone.utc)
            dossier_path = str(self.tmp_dir / "TEST_fresh_dossier.json")
            journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
            journal.save_supervisor_run(
                {
                    "generated_at": now.isoformat(),
                    "severity": "PASS",
                    "report_hash": "unit",
                    "report_text": "unit",
                    "payload": {"checks": []},
                    "sent_to_telegram": False,
                }
            )
            journal.save_decision_features(
                {
                    "dossier_path": dossier_path,
                    "generated_at": now.isoformat(),
                    "ticker": "TEST",
                    "final_verdict": "STARTER",
                    "opportunity_score": 70,
                    "adjusted_score": 70,
                    "confidence": 8,
                    "price": 10.0,
                    "sector": "Technology",
                    "evidence_count": 12,
                    "source_count": 4,
                    "feature_json": json.dumps({"unit": True}),
                }
            )
            journal.save_pending_order_recheck(
                {
                    "recheck_id": "unit-open-recheck",
                    "run_after": (now - timedelta(minutes=1)).isoformat(),
                    "expires_at": (now + timedelta(days=1)).isoformat(),
                    "status": "pending",
                    "ticker": "TEST",
                    "original_verdict": "STARTER",
                    "original_action": "STARTER — Buy about $17.50 at $10.",
                    "original_price": 10.0,
                    "max_price": 10.0,
                    "notional": 17.5,
                    "account_number_masked": "****3334",
                    "original_dossier_path": dossier_path,
                }
            )

            scheduler = ArthaScheduler()
            scheduler.market_hours = SimpleNamespace(is_market_open=lambda dt=None: True)
            scheduler.sell_engine = SimpleNamespace(journal=journal)
            scheduler.telegram = FakeTelegram()
            quote = {"price": 9.95, "volume": 2_000_000, "bid": 9.94, "ask": 9.99}
            scheduler.collector = SimpleNamespace(
                collect_stock=lambda ticker: {**_sample_stock(price=9.95), "ticker": ticker, "quote": quote, "yf_quote": quote},
                collect_macro=lambda: {"fed_funds": 4.5},
                collect_market_overview=lambda: {"fear_greed": {"value": 50, "label": "neutral"}},
            )
            scheduler.council = SimpleNamespace(
                analyze_stock=lambda stock, macro, market, **kwargs: SimpleNamespace(
                    ticker="TEST",
                    final_verdict="STARTER",
                    adjusted_score=70,
                    opportunity_score=70,
                    confidence=8,
                    recommended_action="STARTER remains valid at the open.",
                    dossier_path=dossier_path,
                )
            )

            # Wave 2: dry_run now follows Config.ROBINHOOD_DRY_RUN_ONLY, so the
            # guardrails enforce market hours at queue time; pin market open.
            from unittest.mock import patch as _patch

            from artha.scheduler import MarketHours as _MarketHours

            with _patch.object(_MarketHours, "is_market_open", lambda self, now=None: True):
                asyncio.run(scheduler._run_pending_order_rechecks())
            row = journal.get_pending_order_rechecks(limit=1)[0]
            orders = journal.get_execution_orders(limit=1)
            self.assertEqual(row["status"], "review_ready")
            self.assertEqual(len(orders), 1)
            self.assertEqual(orders[0]["status"], "review_ready")
            actions = journal.get_trade_actions(limit=5)
            self.assertEqual(len(actions), 1)
            self.assertEqual(actions[0]["action_type"], "auto_buy")
            self.assertEqual(len(scheduler.telegram.messages), 1)
            message, _, _, reply_markup = scheduler.telegram.messages[0]
            self.assertIn("ARTHA MONDAY OPEN RE-REVIEW", message)
            self.assertIn("Auto-buy queued", message)
            self.assertIn("No manual permission is required", message)
            self.assertIsNone(reply_markup)
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
                Config.EXECUTION_OFFICER_LLM_ENABLED,
                Config.ROBINHOOD_AUTO_BUY_ENABLED,
            ) = old_config

    def test_live_robinhood_adapter_is_disabled_by_default(self):
        from datetime import datetime, timezone

        from artha.config import Config
        from artha.execution import (
            GuardrailResult,
            RobinhoodExecutionGuardrails,
            RobinhoodMCPBroker,
            build_order_intent,
        )

        old_config = (
            Config.ROBINHOOD_KILL_SWITCH,
            Config.ROBINHOOD_DRY_RUN_ONLY,
            Config.ROBINHOOD_AGENTIC_ENABLED,
        )
        try:
            Config.ROBINHOOD_KILL_SWITCH = True
            Config.ROBINHOOD_DRY_RUN_ONLY = True
            Config.ROBINHOOD_AGENTIC_ENABLED = False

            journal = self._journal_with_clean_buy_decision()
            intent = build_order_intent(
                "TEST",
                "buy",
                notional=25,
                limit_price=100,
                estimated_price=100,
                dry_run=False,
            )
            guardrails = RobinhoodExecutionGuardrails().evaluate(
                intent,
                self._market_data(),
                journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            )
            joined = " ".join(guardrails.reasons).lower()
            self.assertFalse(guardrails.passed)
            self.assertIn("kill switch", joined)
            self.assertIn("dry-run-only", joined)
            self.assertIn("disabled", joined)

            forced_pass = GuardrailResult(passed=True, status="PASS", reasons=[], checks={})
            with self.assertRaises(RuntimeError):
                RobinhoodMCPBroker().submit_order(intent, forced_pass)
        finally:
            (
                Config.ROBINHOOD_KILL_SWITCH,
                Config.ROBINHOOD_DRY_RUN_ONLY,
                Config.ROBINHOOD_AGENTIC_ENABLED,
            ) = old_config

    def test_robinhood_review_only_mode_blocks_live_submission(self):
        from datetime import datetime, timezone

        from artha.config import Config
        from artha.execution import (
            GuardrailResult,
            RobinhoodExecutionGuardrails,
            RobinhoodMCPBroker,
            build_order_intent,
        )

        old_config = (
            Config.ROBINHOOD_REVIEW_ONLY,
            Config.ROBINHOOD_KILL_SWITCH,
            Config.ROBINHOOD_DRY_RUN_ONLY,
            Config.ROBINHOOD_AGENTIC_ENABLED,
        )
        try:
            Config.ROBINHOOD_REVIEW_ONLY = True
            Config.ROBINHOOD_KILL_SWITCH = False
            Config.ROBINHOOD_DRY_RUN_ONLY = False
            Config.ROBINHOOD_AGENTIC_ENABLED = True

            journal = self._journal_with_clean_buy_decision()
            intent = build_order_intent(
                "TEST",
                "buy",
                notional=25,
                limit_price=100,
                estimated_price=100,
                dry_run=False,
            )
            guardrails = RobinhoodExecutionGuardrails().evaluate(
                intent,
                self._market_data(),
                journal,
                now=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            )
            self.assertFalse(guardrails.passed)
            self.assertIn("review-only", " ".join(guardrails.reasons).lower())
            self.assertFalse(guardrails.checks["live_execution_allowed_by_config"])

            forced_pass = GuardrailResult(passed=True, status="PASS", reasons=[], checks={})
            with self.assertRaises(RuntimeError):
                RobinhoodMCPBroker().submit_order(intent, forced_pass)
        finally:
            (
                Config.ROBINHOOD_REVIEW_ONLY,
                Config.ROBINHOOD_KILL_SWITCH,
                Config.ROBINHOOD_DRY_RUN_ONLY,
                Config.ROBINHOOD_AGENTIC_ENABLED,
            ) = old_config


class TestDossierAndAutomation(unittest.TestCase):
    def test_scoring_json_normalization_updates_visible_report_block(self):
        from artha.council import _extract_scoring_json, _replace_last_scoring_json

        text = """Narrative
```json
{
  "opportunity_score": 47,
  "components": {
    "technical_setup": 12,
    "fundamental_quality": 15,
    "contrarian_sentiment": 5,
    "regime_alignment": 7,
    "catalyst_asymmetry": 3,
    "data_quality": 7,
    "liquidity_execution": 5
  },
  "verdict": "DEFER",
  "confidence": 7,
  "thesis_type": "catalyst_driven",
  "recommended_allocation_pct": 0.0,
  "entry_valid_until": "2026-07-04",
  "invalidation_conditions": ["x"],
  "stop_loss_pct": 0.0,
  "target_pct": 0.0
}
```"""
        scoring = _extract_scoring_json(text)
        self.assertEqual(scoring["opportunity_score"], 54)
        normalized = _replace_last_scoring_json(text, scoring)
        self.assertIn('"opportunity_score": 54', normalized)
        self.assertNotIn('"opportunity_score": 47', normalized)

    def test_dossier_feature_extraction_includes_risk_and_valuation(self):
        from artha.dossier import extract_decision_feature_row

        dossier = {
            "ticker": "TEST",
            "generated_at": "2026-06-04T00:00:00+00:00",
            "decision": {"final_verdict": "DEFER", "opportunity_score": 48, "confidence": 8},
            "source_audit": {"evidence_count": 3, "source_counts": {"fmp": 1}},
            "agentic_trace": {"gaps": [], "conflicts": []},
            "stock_packet": {
                "quote": {"price": 100, "marketCap": 1_000_000_000},
                "profile": {"sector": "Technology"},
                "data_quality_report": {"context_coverage_score": 90, "completeness_score": 95},
                "valuation_expectations": {
                    "valuation_signal": "negative",
                    "expectation_risk_level": "high",
                    "analyst_targets": {"consensus_upside_pct": -5.0},
                },
                "portfolio_factor_risk": {
                    "risk_level": "moderate",
                    "sector_after_candidate_pct": 12.0,
                    "sector_benchmark_ticker": "XLK",
                },
            },
        }
        row = extract_decision_feature_row(dossier, "unit.json")
        self.assertEqual(row["valuation_signal"], "negative")
        self.assertEqual(row["portfolio_risk_level"], "moderate")
        self.assertIn("valuation_expectations", row["feature_json"])

    def test_launchd_templates_are_generated(self):
        from artha.automation import build_launchd_plists

        plists = build_launchd_plists(python_path="/tmp/python")
        self.assertIn("com.artha.monitor.plist", plists)
        self.assertIn("<string>-m</string>", plists["com.artha.calibrate-nightly.plist"])
        self.assertIn("<string>run</string>", plists["com.artha.calibrate-nightly.plist"])
        self.assertIn("calibrate", plists["com.artha.calibrate-nightly.plist"])
        self.assertIn("com.artha.diagnose-nightly.plist", plists)
        self.assertIn("--telegram", plists["com.artha.diagnose-nightly.plist"])
        self.assertIn("com.artha.supervise-nightly.plist", plists)
        self.assertIn("supervise", plists["com.artha.supervise-nightly.plist"])


class TestCalibrationDiagnostics(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _journal_with_completed_rows(self, count: int, score: float = 50.0, excess: float = 0.10):
        from artha.journal import DecisionJournal

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        for idx in range(count):
            row_id = journal.log_shadow_trade(
                ticker=f"T{idx:03d}",
                thesis_type="unit",
                blocked_by="test",
                blocked_reason="unit",
                hypothetical_entry=100.0,
                hypothetical_stop=90.0,
                opportunity_score=score,
                regime="NEUTRAL",
                fear_greed=50,
                sector="Technology",
                benchmark_ticker="QQQ",
                sector_benchmark_ticker="XLK",
            )
            journal.update_shadow_returns(
                row_id,
                price_5d=105.0,
                price_20d=108.0,
                price_60d=100.0 * (1.0 + excess + 0.02),
                benchmark_price_entry=100.0,
                benchmark_price_5d=101.0,
                benchmark_price_20d=102.0,
                benchmark_price_60d=102.0,
                sector_benchmark_price_entry=100.0,
                sector_benchmark_price_5d=101.0,
                sector_benchmark_price_20d=102.0,
                sector_benchmark_price_60d=102.0,
            )
        return journal

    def test_diagnostics_learning_only_under_20_samples(self):
        from artha.diagnostics import build_diagnostic_report

        journal = self._journal_with_completed_rows(4, score=50.0, excess=0.08)
        diagnostic = build_diagnostic_report(journal)
        self.assertEqual(diagnostic["stage"], "learning_only")
        self.assertIn("Completed forward samples: 4", diagnostic["report_text"])
        statuses = {f["status"] for f in diagnostic["payload"]["proposed_fixes"]}
        self.assertTrue(statuses.issubset({"bookkeeping_only", "no_change"}))

    def test_diagnostics_shadow_fix_after_20_samples(self):
        from artha.diagnostics import build_diagnostic_report

        journal = self._journal_with_completed_rows(20, score=50.0, excess=0.08)
        diagnostic = build_diagnostic_report(journal)
        self.assertEqual(diagnostic["stage"], "minimum_diagnosis")
        fixes = diagnostic["payload"]["proposed_fixes"]
        self.assertTrue(any(f["status"] == "shadow_mode" for f in fixes))
        self.assertTrue(any("false DEFER" in f["suggested_change"] or "starter path" in f["suggested_change"] for f in fixes))

    def test_should_send_when_new_samples_mature(self):
        from artha.diagnostics import build_diagnostic_report, should_send_diagnostic

        journal = self._journal_with_completed_rows(4)
        diagnostic = build_diagnostic_report(journal)
        previous = {
            "completed_samples": 3,
            "stage": "learning_only",
            "severity": "INFO",
            "report_hash": "old",
        }
        self.assertTrue(should_send_diagnostic(diagnostic, previous))

    def test_run_diagnosis_sends_through_telegram_sender(self):
        import artha.diagnostics as diagnostics
        from artha.diagnostics import run_calibration_diagnosis

        class FakeSender:
            enabled = True

            def __init__(self):
                self.messages = []

            def send_message(self, text, parse_mode=None, silent=False):
                self.messages.append((text, parse_mode, silent))
                return True

        diagnostics.DIAGNOSTIC_DIR = self.tmp_dir / "diagnostics"
        journal = self._journal_with_completed_rows(4)
        sender = FakeSender()
        result = run_calibration_diagnosis(
            journal=journal,
            send_telegram=True,
            force_telegram=True,
            sender=sender,
        )
        self.assertTrue(result["sent_to_telegram"])
        self.assertEqual(len(sender.messages), 1)
        self.assertIn("ARTHA LEARNING DIAGNOSIS", sender.messages[0][0])
        self.assertTrue((self.tmp_dir / "diagnostics" / "latest.txt").exists())


class TestSellSideHardening(unittest.TestCase):
    def setUp(self):
        from artha.config import Config

        self._tmp_ctx = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp_ctx.name)
        self._control_file = Config.ROBINHOOD_CONTROL_FILE
        Config.ROBINHOOD_CONTROL_FILE = str(self.tmp_dir / "control.json")

    def tearDown(self):
        from artha.config import Config

        Config.ROBINHOOD_CONTROL_FILE = self._control_file
        self._tmp_ctx.cleanup()

    def _active_thesis(self, journal, ticker="TEST", position_type="STARTER", entry=100.0):
        from artha.thesis_tracker import ThesisTracker

        tracker = ThesisTracker(journal=journal)
        thesis = tracker.create_thesis(
            ticker=ticker,
            position_type=position_type,
            thesis_summary="Unit thesis",
            invalidation_conditions=["Free cash flow deteriorates and debt rises"],
        )
        return tracker.activate_thesis(thesis.thesis_id, entry, shares=1)

    def test_sell_engine_enforces_hard_stop_for_starter_position(self):
        from types import SimpleNamespace

        from artha.journal import DecisionJournal
        from artha.sell_engine import SellEngine

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        self._active_thesis(journal, position_type="STARTER", entry=100.0)
        engine = SellEngine(journal=journal, collector=SimpleNamespace())
        portfolio = SimpleNamespace(positions=[], get_position=lambda ticker: None)

        signals = engine.run_price_check_sell_tasks(portfolio, {"TEST": {"price": 79.0}})

        self.assertTrue(any(s.signal_type == "hard_stop" for s in signals))
        signal = next(s for s in signals if s.signal_type == "hard_stop")
        self.assertEqual(signal.severity, "URGENT")
        self.assertEqual(signal.action_recommended, "EXIT")
        rows = journal.get_active_sell_signals("TEST")
        self.assertTrue(any(r["signal_type"] == "hard_stop" for r in rows))

    def test_drawdown_never_suppresses_sell_council_signal(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from artha.journal import DecisionJournal
        from artha.sell_engine import SellEngine, SellSignal

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        engine = SellEngine(journal=journal, collector=SimpleNamespace())
        signal = SellSignal(
            ticker="TEST",
            signal_type="thesis_break",
            severity="HIGH",
            action_recommended="EXIT",
            message="Fundamental trigger fired.",
        )
        with patch.object(engine.circuit_breaker, "is_portfolio_in_drawdown", return_value=True), \
                patch.object(engine.aggregator, "suppress") as suppress, \
                patch.object(engine.circuit_breaker, "record_exit") as record_exit:
            routed = engine._apply_circuit_breaker([signal])

        self.assertEqual(routed, [signal])
        self.assertIn("Sell Council", signal.message)
        suppress.assert_not_called()
        record_exit.assert_not_called()

    def test_periodic_sell_review_slot_is_independent_from_daily_health(self):
        from datetime import datetime, timezone

        from artha.scheduler import ArthaScheduler

        scheduler = ArthaScheduler()
        scheduler._last_run = {}
        scheduler._mark_run_slot = lambda name, slot: scheduler._last_run.__setitem__(name, slot)
        # Friday June 5 2026, 4:30 PM ET: close + 30 minutes.
        now = datetime(2026, 6, 5, 20, 30, tzinfo=timezone.utc)

        self.assertTrue(scheduler._should_run_daily_health(now))
        self.assertTrue(scheduler._should_run_periodic_review_check(now))

    def test_daily_health_digest_never_marks_position_pending_exit(self):
        import asyncio
        from types import SimpleNamespace

        from artha.scheduler import ArthaScheduler

        scheduler = ArthaScheduler()
        scheduler.monitor = SimpleNamespace(run_and_dedupe=lambda: [])
        scheduler.telegram = SimpleNamespace(
            enabled=True,
            send_health_check=lambda message: True,
        )
        scheduler._pending_nonurgent_sell_signals = lambda: [
            {
                "signal_id": "sig-unit",
                "ticker": "TEST",
                "thesis_id": "thesis-unit",
                "action_recommended": "EXIT",
                "message": "Unit exit warning",
            }
        ]
        scheduler._format_nonurgent_sell_digest = lambda rows: "Unit sell digest"
        scheduler._mark_sell_signals_actioned = Mock()
        scheduler.sell_engine.tracker.mark_waiting_for_sell = Mock()

        asyncio.run(scheduler._run_quick_health_check())

        scheduler._mark_sell_signals_actioned.assert_not_called()
        scheduler.sell_engine.tracker.mark_waiting_for_sell.assert_not_called()

    def test_robinhood_sell_review_prepares_sell_side_review_only_row(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from artha.config import Config
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio, Position
        from artha.scheduler import ArthaScheduler

        old_config = (
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            Config.ROBINHOOD_ALLOW_AFTER_HOURS,
            Config.ROBINHOOD_AUTO_SELL_ENABLED,
        )
        Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
        Config.ROBINHOOD_ALLOW_AFTER_HOURS = True
        Config.ROBINHOOD_AUTO_SELL_ENABLED = True
        try:
            journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
            thesis = self._active_thesis(journal, ticker="TEST", position_type="STARTER", entry=100.0)
            portfolio_path = self.tmp_dir / "portfolio.json"
            Portfolio(
                positions=[
                    Position(
                        ticker="TEST",
                        asset_type="stock",
                        shares=1.5,
                        avg_cost=100.0,
                        opened_at="2026-06-05T14:00:00+00:00",
                        thesis_id=thesis.thesis_id,
                        position_type="STARTER",
                    )
                ]
            ).save(portfolio_path)

            scheduler = ArthaScheduler()
            scheduler.sell_engine.journal = journal
            scheduler.sell_engine.tracker.journal = journal
            # Wave 2: dry_run follows Config.ROBINHOOD_DRY_RUN_ONLY, so pin the
            # market open and keep the (unrelated) live snapshot out of the gate.
            from artha.scheduler import MarketHours as _MarketHours

            scheduler._load_robinhood_position_snapshot = lambda: {
                "status": "MISSING",
                "fresh": False,
                "positions": [],
                "warnings": ["unit test"],
            }
            dossier_path = self.tmp_dir / "sell_dossier.json"
            dossier_path.write_text("{}", encoding="utf-8")
            decision = SimpleNamespace(
                action="EXIT",
                session_id="sell-session-unit",
                dossier_path=str(dossier_path),
                sell_score=88,
                confidence=8,
                trigger_type="unit_test",
            )
            with patch("artha.scheduler.PORTFOLIO_FILE", portfolio_path), \
                    patch.object(_MarketHours, "is_market_open", lambda self, now=None: True):
                result = scheduler._prepare_robinhood_sell_review(
                    thesis=thesis,
                    action="EXIT",
                    current_price=95.0,
                    journal=journal,
                    trigger_type="unit_test",
                    reason="unit sell review",
                    sell_decision=decision,
                )
                action_row = journal.get_trade_actions(limit=1)[0]
                journal.update_trade_action(action_row["action_id"], {"status": "submitted"})
                duplicate = scheduler._prepare_robinhood_sell_review(
                    thesis=thesis,
                    action="EXIT",
                    current_price=94.0,
                    journal=journal,
                    trigger_type="unit_test",
                    reason="duplicate unit sell review",
                    sell_decision=decision,
                )
            rows = journal.get_execution_orders(limit=1)
            self.assertIsNotNone(result)
            self.assertIsNone(duplicate)
            self.assertEqual(rows[0]["side"], "sell")
            self.assertEqual(rows[0]["status"], "review_ready")
            self.assertIn('"side": "sell"', rows[0]["request_json"])
            self.assertIn('"quantity": "1.5"', rows[0]["request_json"])
            self.assertIsNone(rows[0]["submitted_at"])
            self.assertEqual(journal.get_trade_actions(limit=1)[0]["action_type"], "auto_sell")
        finally:
            (
                Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
                Config.ROBINHOOD_ALLOW_AFTER_HOURS,
                Config.ROBINHOOD_AUTO_SELL_ENABLED,
            ) = old_config

    def test_broker_reconciliation_reports_missing_and_mismatched_positions(self):
        from artha.execution import reconcile_robinhood_positions
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio, Position

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        portfolio = Portfolio(
            positions=[
                Position(
                    ticker="ABC",
                    asset_type="stock",
                    shares=2.0,
                    avg_cost=10.0,
                    opened_at="2026-06-05T14:00:00+00:00",
                )
            ]
        )
        result = reconcile_robinhood_positions(
            [{"symbol": "ABC", "quantity": "1.0"}, {"symbol": "XYZ", "quantity": "3"}],
            portfolio=portfolio,
            journal=journal,
        )
        self.assertEqual(result["status"], "WARN")
        self.assertEqual(result["broker_only"][0]["ticker"], "XYZ")
        self.assertEqual(result["quantity_mismatches"][0]["ticker"], "ABC")

    def test_broker_warning_throttle_persists_across_scheduler_restarts(self):
        from artha.config import Config
        from artha.scheduler import ArthaScheduler

        old_file = Config.ROBINHOOD_WARNING_STATE_FILE
        try:
            Config.ROBINHOOD_WARNING_STATE_FILE = str(self.tmp_dir / "broker_warning_state.json")
            first = ArthaScheduler()
            self.assertTrue(first._should_send_broker_warning("snapshot_not_fresh:WARN:1", min_minutes=30))
            self.assertFalse(first._should_send_broker_warning("snapshot_not_fresh:WARN:1", min_minutes=30))

            restarted = ArthaScheduler()
            self.assertFalse(restarted._should_send_broker_warning("snapshot_not_fresh:WARN:1", min_minutes=30))
            self.assertTrue((self.tmp_dir / "broker_warning_state.json").exists())
        finally:
            Config.ROBINHOOD_WARNING_STATE_FILE = old_file

    def test_stale_snapshot_warning_message_is_not_overly_alarmist(self):
        from artha.scheduler import ArthaScheduler

        msg = ArthaScheduler._snapshot_missing_or_stale_message(
            {
                "status": "WARN",
                "path": "/tmp/latest_snapshot.json",
                "positions": [{"symbol": "JNJ", "quantity": "0.074763"}],
                "warnings": ["Snapshot is stale: 207.9 minutes old; max allowed is 10 minutes."],
            }
        )
        self.assertIn("keeps monitoring the last reconciled holdings", msg)
        self.assertIn("Review/Place actions stay blocked", msg)
        self.assertIn("throttled", msg)
        self.assertNotIn("Do not trust sell monitoring", msg)

    def test_stale_snapshot_telegram_alerts_only_when_market_open(self):
        from types import SimpleNamespace

        from artha.scheduler import ArthaScheduler

        scheduler = ArthaScheduler()
        scheduler.market_hours = SimpleNamespace(is_market_open=lambda now=None: False)
        self.assertFalse(scheduler._should_alert_on_stale_robinhood_snapshot())

        scheduler.market_hours = SimpleNamespace(is_market_open=lambda now=None: True)
        self.assertTrue(scheduler._should_alert_on_stale_robinhood_snapshot())

    def test_supervisor_does_not_warn_for_stale_only_snapshot_outside_market_hours(self):
        from unittest.mock import patch

        from artha.config import Config
        from artha.portfolio import Portfolio, Position
        import artha.supervisor as supervisor

        old_account = Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER
        old_snapshot = Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE
        portfolio_path = self.tmp_dir / "portfolio.json"
        snapshot_path = self.tmp_dir / "latest_snapshot.json"
        try:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE = str(snapshot_path)
            Portfolio(
                positions=[
                    Position(
                        ticker="JNJ",
                        asset_type="stock",
                        shares=0.074763,
                        avg_cost=234.07,
                        opened_at="2026-06-08T14:23:51+00:00",
                    )
                ]
            ).save(portfolio_path)
            snapshot_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-06-08T21:00:00+00:00",
                        "source": "robinhood_mcp",
                        "account": {
                            "account_number": "222233334",
                            "type": "cash",
                            "nickname": "Agentic",
                            "agentic_allowed": True,
                            "state": "active",
                        },
                        "portfolio": {"cash": "332.50"},
                        "positions": [{"symbol": "JNJ", "quantity": "0.074763"}],
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(supervisor, "PORTFOLIO_FILE", portfolio_path), patch.object(
                supervisor,
                "_is_regular_market_open_now",
                return_value=False,
            ):
                result = supervisor._check_broker_reconciliation_snapshot()
        finally:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = old_account
            Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE = old_snapshot

        self.assertEqual(result["status"], "PASS")
        self.assertIn("outside regular market hours", result["message"])
        self.assertFalse(result["snapshot"]["fresh"])

    def test_high_held_news_matches_thesis_invalidation_conditions(self):
        from types import SimpleNamespace

        from artha.journal import DecisionJournal
        from artha.scheduler import ArthaScheduler
        from artha.sell_engine import SellEngine

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        self._active_thesis(journal, ticker="TEST", position_type="STARTER", entry=100.0)
        scheduler = ArthaScheduler()
        scheduler.sell_engine = SellEngine(journal=journal)
        alert = SimpleNamespace(
            ticker="TEST",
            message="",
            metadata={"headline": "TEST debt rises while free cash flow deteriorates", "severity": "HIGH"},
        )
        self.assertTrue(scheduler._held_news_matches_thesis(alert))

    def test_high_held_news_semantic_gate_catches_non_keyword_match(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from artha.journal import DecisionJournal
        from artha.scheduler import ArthaScheduler
        from artha.sell_engine import SellEngine

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        tracker_thesis = self._active_thesis(journal, ticker="MTCH", position_type="TACTICAL_BUY", entry=34.0)
        from artha.thesis_tracker import ThesisTracker
        ThesisTracker(journal=journal).update_thesis_fields(
            tracker_thesis.thesis_id,
            invalidation_conditions=["Tinder and Hinge payer trends keep deteriorating despite stabilization claims"],
        )
        scheduler = ArthaScheduler()
        scheduler.sell_engine = SellEngine(journal=journal)
        alert = SimpleNamespace(
            ticker="MTCH",
            message="",
            metadata={"headline": "Match Group dating-app user churn accelerates in new survey", "severity": "HIGH"},
        )
        self.assertFalse(scheduler._held_news_matches_thesis(alert))
        with patch(
            "artha.chatgpt_backend.ChatGPTBackendClient.chat",
            return_value='{"matches": true, "confidence": 0.88, "urgency": "HIGH", "risk_category": "fundamental", "affected_conditions": ["payer trends deteriorating"], "reason": "user churn maps to payer deterioration", "false_positive_risk": "medium"}',
        ):
            self.assertTrue(scheduler._held_news_semantically_matches_thesis(alert))

    def test_high_held_news_semantic_gate_rejects_low_confidence(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from artha.journal import DecisionJournal
        from artha.scheduler import ArthaScheduler
        from artha.sell_engine import SellEngine

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        thesis = self._active_thesis(journal, ticker="MTCH", position_type="TACTICAL_BUY", entry=34.0)
        from artha.thesis_tracker import ThesisTracker
        ThesisTracker(journal=journal).update_thesis_fields(
            thesis.thesis_id,
            invalidation_conditions=["Tinder and Hinge payer trends keep deteriorating despite stabilization claims"],
        )
        scheduler = ArthaScheduler()
        scheduler.sell_engine = SellEngine(journal=journal)
        alert = SimpleNamespace(
            ticker="MTCH",
            message="",
            metadata={"headline": "Match Group mentioned in lifestyle survey roundup", "severity": "HIGH"},
        )
        with patch(
            "artha.chatgpt_backend.ChatGPTBackendClient.chat",
            return_value='{"matches": true, "confidence": 0.35, "urgency": "LOW", "risk_category": "routine", "affected_conditions": [], "reason": "weak mention only", "false_positive_risk": "high"}',
        ):
            assessment = scheduler._held_news_semantic_assessment(alert)
        self.assertFalse(assessment["matches"])
        self.assertTrue(assessment["raw_matches"])
        self.assertLess(assessment["confidence"], assessment["confidence_threshold"])

    def test_robinhood_snapshot_normalization_requires_fresh_account_metadata(self):
        from artha.config import Config
        from artha.execution import normalize_robinhood_position_snapshot

        old_account = Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER
        Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
        try:
            snapshot = normalize_robinhood_position_snapshot(
                {
                    "source": "unit",
                    "generated_at": "2026-06-05T15:00:00+00:00",
                    "account": {
                        "account_number": "222233334",
                        "agentic_allowed": True,
                        "type": "cash",
                        "state": "active",
                    },
                    "portfolio": {"total_value": "350", "cash": "350"},
                    "positions": [{"symbol": "ABC", "quantity": "1.25"}],
                },
                now=datetime(2026, 6, 5, 15, 5, tzinfo=timezone.utc),
                max_age_minutes=10,
            )
        finally:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = old_account

        self.assertEqual(snapshot["status"], "PASS")
        self.assertTrue(snapshot["fresh"])
        self.assertEqual(snapshot["position_count"], 1)
        self.assertEqual(snapshot["account_check"]["status"], "PASS")

    def test_robinhood_snapshot_accepts_selected_account_alias(self):
        from artha.config import Config
        from artha.execution import normalize_robinhood_position_snapshot
        from artha.robinhood_bridge import canonicalize_mcp_snapshot

        old_account = Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER
        Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "222233334"
        try:
            payload = {
                "source": "robinhood_mcp",
                "generated_at": "2026-06-05T15:00:00+00:00",
                "selected_account": {
                    "account_number": "222233334",
                    "agentic_allowed": True,
                    "type": "cash",
                    "state": "active",
                },
                "accounts": [],
                "portfolio": {"buying_power": "350", "cash": "350", "total_value": "350"},
                "positions": [],
                "orders": [],
            }
            canonical = canonicalize_mcp_snapshot(payload)
            snapshot = normalize_robinhood_position_snapshot(
                canonical,
                now=datetime(2026, 6, 5, 15, 1, tzinfo=timezone.utc),
                max_age_minutes=10,
            )
        finally:
            Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = old_account

        self.assertIn("account", canonical)
        self.assertEqual(snapshot["status"], "PASS")
        self.assertEqual(snapshot["account_check"]["status"], "PASS")

    def test_robinhood_snapshot_normalization_flags_stale_snapshot(self):
        from artha.execution import normalize_robinhood_position_snapshot

        snapshot = normalize_robinhood_position_snapshot(
            {"generated_at": "2026-06-05T14:00:00+00:00", "positions": []},
            now=datetime(2026, 6, 5, 15, 0, tzinfo=timezone.utc),
            max_age_minutes=10,
        )
        self.assertEqual(snapshot["status"], "WARN")
        self.assertFalse(snapshot["fresh"])
        self.assertTrue(any("stale" in warning for warning in snapshot["warnings"]))

    def test_benzinga_company_news_disabled_without_key(self):
        from artha.collector import BenzingaCollector

        collector = BenzingaCollector()
        collector.enabled = False
        self.assertIsNone(collector.company_news("AAPL"))

    def test_supervisor_warns_pending_buys_are_not_active_sell_monitored(self):
        from unittest.mock import patch

        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio
        import artha.supervisor as supervisor
        from artha.thesis_tracker import ThesisTracker

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        ThesisTracker(journal=journal).create_thesis("MTCH", "TACTICAL_BUY", thesis_summary="unit pending")
        portfolio_path = self.tmp_dir / "portfolio.json"
        Portfolio(positions=[]).save(portfolio_path)

        with patch.object(supervisor, "PORTFOLIO_FILE", portfolio_path):
            result = supervisor._check_position_monitoring(journal)

        self.assertEqual(result["status"], "WARN")
        self.assertIn("pending buy", result["message"].lower())
        self.assertEqual(result["pending_theses"], ["MTCH"])

    def test_supervisor_fails_held_position_without_active_thesis(self):
        from unittest.mock import patch

        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio, Position
        import artha.supervisor as supervisor

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        portfolio_path = self.tmp_dir / "portfolio.json"
        Portfolio(
            positions=[
                Position(
                    ticker="UNMON",
                    asset_type="stock",
                    shares=1,
                    avg_cost=10,
                    opened_at="2026-06-05T14:00:00+00:00",
                )
            ]
        ).save(portfolio_path)

        with patch.object(supervisor, "PORTFOLIO_FILE", portfolio_path):
            result = supervisor._check_position_monitoring(journal)

        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["unmonitored_positions"], ["UNMON"])

    def test_supervisor_fails_pending_exit_without_drainable_auto_sell(self):
        from unittest.mock import patch

        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio, Position
        import artha.supervisor as supervisor
        from artha.thesis_tracker import ThesisTracker

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        tracker = ThesisTracker(journal=journal)
        pending = tracker.create_thesis(
            "RTX",
            "TACTICAL_BUY",
            thesis_summary="unit pending",
            invalidation_conditions=["Fundamental catalyst fails"],
        )
        tracker.activate_thesis(pending.thesis_id, entry_price=100.0, shares=0.1)
        with journal._connect() as conn:
            conn.execute("UPDATE position_theses SET status = 'pending_exit' WHERE ticker = 'RTX'")
            conn.commit()
        portfolio_path = self.tmp_dir / "portfolio.json"
        Portfolio(
            positions=[
                Position(
                    ticker="RTX",
                    asset_type="stock",
                    shares=0.1,
                    avg_cost=100,
                    opened_at="2026-06-05T14:00:00+00:00",
                )
            ]
        ).save(portfolio_path)

        with patch.object(supervisor, "PORTFOLIO_FILE", portfolio_path):
            result = supervisor._check_position_monitoring(journal)

        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["pending_exit_theses"], ["RTX"])
        self.assertEqual(result["pending_exit_without_auto_sell"], ["RTX"])
        journal.save_trade_action(
            {
                "action_id": "ta_sell_rtx",
                "status": "review_ready",
                "action_type": "auto_sell",
                "ticker": "RTX",
                "side": "sell",
                "thesis_id": pending.thesis_id,
            }
        )
        with patch.object(supervisor, "PORTFOLIO_FILE", portfolio_path):
            repaired = supervisor._check_position_monitoring(journal)
        self.assertEqual(repaired["status"], "PASS")

    def test_supervisor_does_not_count_expired_auto_sell_as_drainable(self):
        from datetime import datetime, timedelta, timezone
        from unittest.mock import patch

        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio, Position
        import artha.supervisor as supervisor
        from artha.thesis_tracker import ThesisTracker

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        tracker = ThesisTracker(journal=journal)
        thesis = self._active_thesis(journal, ticker="STALE", entry=100)
        tracker.mark_waiting_for_sell(thesis.thesis_id, reason="EXIT")
        portfolio_path = self.tmp_dir / "portfolio.json"
        Portfolio(
            positions=[
                Position(
                    ticker="STALE",
                    asset_type="stock",
                    shares=1,
                    avg_cost=100,
                    opened_at="2026-06-05T14:00:00+00:00",
                )
            ]
        ).save(portfolio_path)
        journal.save_trade_action(
            {
                "action_id": "ta_expired_sell",
                "status": "review_ready",
                "action_type": "auto_sell",
                "ticker": "STALE",
                "side": "sell",
                "thesis_id": thesis.thesis_id,
                "expires_at": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
            }
        )

        with patch.object(supervisor, "PORTFOLIO_FILE", portfolio_path):
            result = supervisor._check_position_monitoring(journal)

        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["pending_exit_without_auto_sell"], ["STALE"])

    def test_portfolio_nav_includes_cash_and_partial_sell_refreshes_mark(self):
        from artha.portfolio import Portfolio, Position

        portfolio = Portfolio(
            positions=[
                Position(
                    ticker="TEST",
                    asset_type="stock",
                    shares=2,
                    avg_cost=20,
                    opened_at="2026-06-05T14:00:00+00:00",
                    current_price=30,
                    market_value=60,
                )
            ],
            cash_available=40,
        )
        self.assertEqual(portfolio.invested_value(), 60)
        self.assertEqual(portfolio.total_nav(), 100)
        portfolio.sell_position("TEST", 0.5, 32)
        remaining = portfolio.get_position("TEST")
        self.assertAlmostEqual(remaining.shares, 1.5)
        self.assertAlmostEqual(remaining.market_value, 48)

    def test_tactical_buy_uses_three_day_initial_sell_cooldown(self):
        from datetime import datetime, timezone

        from artha.journal import DecisionJournal

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        thesis = self._active_thesis(
            journal,
            ticker="TACT",
            position_type="TACTICAL_BUY",
            entry=25,
        )
        entry = datetime.fromisoformat(thesis.entry_date).astimezone(timezone.utc)
        cooldown = datetime.fromisoformat(thesis.sell_cooldown_until).astimezone(timezone.utc)
        self.assertLessEqual((cooldown - entry).total_seconds(), 3 * 86400 + 5)

    def test_sell_signal_persists_exact_trim_fraction(self):
        from artha.journal import DecisionJournal
        from artha.sell_engine import SellSignal, SellSignalAggregator

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        signal = SellSignal(
            ticker="TEST",
            signal_type="earnings_risk",
            severity="HIGH",
            source="unit",
            action_recommended="TRIM",
            trim_pct=0.37,
        )
        SellSignalAggregator(journal=journal).record(signal)
        row = journal.get_active_sell_signals("TEST")[0]
        self.assertAlmostEqual(float(row["trim_pct"]), 0.37)

    def test_sell_council_trim_fraction_reaches_execution_queue_unchanged(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from artha.journal import DecisionJournal
        from artha.scheduler import ArthaScheduler

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        thesis = self._active_thesis(journal, ticker="TRIM", entry=100)
        scheduler = ArthaScheduler()
        scheduler.sell_engine.journal = journal
        scheduler.sell_engine.tracker.journal = journal
        scheduler.telegram = SimpleNamespace(enabled=False)
        decision = SimpleNamespace(
            action="TRIM",
            trim_pct=0.37,
            session_id="sell-trim-unit",
            dossier_path=str(self.tmp_dir / "sell-trim.json"),
            sell_score=72,
            confidence=7,
            trigger_type="earnings_risk",
            next_review_date=None,
            health_score=None,
        )
        council = SimpleNamespace(format_sell_telegram=lambda decision, thesis: "")
        queued = {
            "trade_action": {
                "action_type": "auto_sell",
                "status": "review_ready",
            }
        }

        with patch.object(
            scheduler,
            "_prepare_robinhood_sell_review",
            return_value=queued,
        ) as prepare:
            result = scheduler._apply_sell_council_decision(
                sell_council=council,
                decision=decision,
                thesis=thesis,
                stock_data={"quote": {"price": 101}},
                trigger_type="earnings_risk",
                fallback_trim_pct=0.25,
            )

        self.assertTrue(result["queued"])
        self.assertAlmostEqual(prepare.call_args.kwargs["trim_pct"], 0.37)

    def test_projected_account_exposure_is_hard_blocked_above_ninety_percent(self):
        from artha.execution import build_order_intent, evaluate_broker_snapshot_guardrails

        intent = build_order_intent("NEW", "buy", notional=20, estimated_price=10)
        snapshot = {
            "status": "PASS",
            "fresh": True,
            "portfolio": {"buying_power": 20, "cash": 20, "total_value": 20},
            "positions": [
                {"symbol": "OLD", "quantity": "18", "market_price": "10"}
            ],
            "orders": [],
        }
        blocked = evaluate_broker_snapshot_guardrails(intent, snapshot, 20)
        self.assertFalse(blocked["passed"])
        self.assertTrue(any("90%" in reason for reason in blocked["reasons"]))

        allowed_snapshot = {
            **snapshot,
            "portfolio": {"buying_power": 20, "cash": 20, "total_value": 20},
            "positions": [
                {"symbol": "OLD", "quantity": "8", "market_price": "10"}
            ],
        }
        allowed = evaluate_broker_snapshot_guardrails(
            build_order_intent("NEW", "buy", notional=10, estimated_price=10),
            allowed_snapshot,
            10,
        )
        self.assertTrue(allowed["passed"], allowed["reasons"])

    def test_full_sell_fill_archives_pending_exit_and_updates_trade_action(self):
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio, Position
        from artha.robinhood_bridge import record_order_fill
        from artha.thesis_tracker import ThesisTracker

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        tracker = ThesisTracker(journal=journal)
        thesis = self._active_thesis(journal, ticker="FULL", position_type="STARTER", entry=100)
        tracker.mark_waiting_for_sell(thesis.thesis_id, reason="EXIT")
        portfolio_path = self.tmp_dir / "portfolio.json"
        Portfolio(
            positions=[
                Position(
                    ticker="FULL",
                    asset_type="stock",
                    shares=1,
                    avg_cost=100,
                    opened_at="2026-06-05T14:00:00+00:00",
                    market_value=100,
                )
            ]
        ).save(portfolio_path)
        intent_id = "sell-intent-full"
        journal.save_execution_order(
            {
                "order_intent_id": intent_id,
                "ticker": "FULL",
                "side": "sell",
                "order_type": "market",
                "quantity": 1,
                "notional": 95,
                "estimated_price": 95,
                "status": "submitted",
                "thesis_id": thesis.thesis_id,
                "evidence_json": {
                    "sell_council": {"confirmed": True, "action": "EXIT"}
                },
                "request_json": {"symbol": "FULL", "side": "sell", "quantity": "1"},
            }
        )
        journal.save_trade_action(
            {
                "action_id": "ta_full",
                "status": "submitted",
                "action_type": "auto_sell",
                "ticker": "FULL",
                "side": "sell",
                "thesis_id": thesis.thesis_id,
                "order_intent_id": intent_id,
            }
        )
        result = record_order_fill(
            order_intent_id=intent_id,
            fill_payload={
                "id": "rh-full",
                "symbol": "FULL",
                "side": "sell",
                "state": "filled",
                "cumulative_quantity": "1",
                "average_price": "95",
            },
            journal=journal,
            portfolio_path=portfolio_path,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertIsNone(Portfolio.load(portfolio_path).get_position("FULL"))
        self.assertEqual(tracker.get(thesis.thesis_id).status, "archived")
        self.assertEqual(journal.get_trade_action("ta_full")["status"], "filled")
        self.assertEqual(len(journal.get_broker_fill_effects()), 1)
        self.assertEqual(len(journal.get_sell_trade_events(thesis_id=thesis.thesis_id)), 1)
        self.assertEqual(len(journal.get_pending_post_sell_reviews()), 1)
        episodes = journal.get_position_trade_episodes(status="closed")
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["sell_fill_count"], 1)
        from artha.supervisor import _check_broker_fill_accounting
        accounting = _check_broker_fill_accounting(journal, portfolio_path=portfolio_path)
        self.assertEqual(accounting["status"], "PASS", accounting)
        journal.update_trade_action("ta_full", {"status": "submitted"})
        repeated = record_order_fill(
            order_intent_id=intent_id,
            fill_payload={
                "id": "rh-full",
                "symbol": "FULL",
                "side": "sell",
                "state": "filled",
                "cumulative_quantity": "1",
                "average_price": "95",
            },
            journal=journal,
            portfolio_path=portfolio_path,
        )
        self.assertTrue(repeated["already_recorded"])
        self.assertEqual(journal.get_trade_action("ta_full")["status"], "filled")
        self.assertEqual(len(journal.get_broker_fill_effects()), 1)
        self.assertEqual(len(journal.get_sell_trade_events(thesis_id=thesis.thesis_id)), 1)
        self.assertEqual(len(journal.get_pending_post_sell_reviews()), 1)

    def test_partial_trim_fill_keeps_thesis_active_and_updates_remaining_value(self):
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio, Position
        from artha.robinhood_bridge import record_order_fill
        from artha.thesis_tracker import ThesisTracker

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        tracker = ThesisTracker(journal=journal)
        thesis = self._active_thesis(journal, ticker="PART", position_type="STARTER", entry=100)
        portfolio_path = self.tmp_dir / "portfolio.json"
        Portfolio(
            positions=[
                Position(
                    ticker="PART",
                    asset_type="stock",
                    shares=1,
                    avg_cost=100,
                    opened_at="2026-06-05T14:00:00+00:00",
                    market_value=100,
                )
            ]
        ).save(portfolio_path)
        intent_id = "sell-intent-part"
        journal.save_execution_order(
            {
                "order_intent_id": intent_id,
                "ticker": "PART",
                "side": "sell",
                "order_type": "market",
                "quantity": 0.25,
                "notional": 23.75,
                "estimated_price": 95,
                "status": "submitted",
                "thesis_id": thesis.thesis_id,
                "evidence_json": {
                    "sell_council": {"confirmed": True, "action": "TRIM"}
                },
                "request_json": {"symbol": "PART", "side": "sell", "quantity": "0.25"},
            }
        )
        result = record_order_fill(
            order_intent_id=intent_id,
            fill_payload={
                "id": "rh-part",
                "symbol": "PART",
                "side": "sell",
                "state": "filled",
                "cumulative_quantity": "0.25",
                "average_price": "95",
            },
            journal=journal,
            portfolio_path=portfolio_path,
        )
        self.assertEqual(result["status"], "PASS")
        position = Portfolio.load(portfolio_path).get_position("PART")
        self.assertAlmostEqual(position.shares, 0.75)
        self.assertAlmostEqual(position.market_value, 71.25)
        refreshed = tracker.get(thesis.thesis_id)
        self.assertEqual(refreshed.status, "active")
        self.assertIsNotNone(refreshed.sell_cooldown_until)
        self.assertEqual(len(journal.get_sell_trade_events(thesis_id=thesis.thesis_id)), 1)
        self.assertEqual(journal.get_pending_post_sell_reviews(), [])

    def test_replayed_trim_fill_does_not_extend_cooldown(self):
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio, Position
        from artha.robinhood_bridge import record_order_fill
        from artha.thesis_tracker import ThesisTracker

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        tracker = ThesisTracker(journal)
        thesis = self._active_thesis(journal, ticker="COOL", position_type="STARTER", entry=100)
        portfolio_path = self.tmp_dir / "portfolio.json"
        Portfolio(
            positions=[
                Position(
                    ticker="COOL",
                    asset_type="stock",
                    shares=1,
                    avg_cost=100,
                    opened_at="2026-08-01T14:00:00Z",
                    market_value=100,
                )
            ]
        ).save(portfolio_path)
        intent_id = "sell-intent-cooldown-replay"
        journal.save_execution_order(
            {
                "order_intent_id": intent_id,
                "ticker": "COOL",
                "side": "sell",
                "order_type": "market",
                "quantity": 0.25,
                "notional": 25,
                "estimated_price": 100,
                "status": "submitted",
                "thesis_id": thesis.thesis_id,
                "evidence_json": {"sell_council": {"confirmed": True, "action": "TRIM"}},
            }
        )
        fill = {
            "id": "rh-cooldown-replay",
            "symbol": "COOL",
            "side": "sell",
            "state": "filled",
            "cumulative_quantity": "0.25",
            "average_price": "100",
            "last_transaction_at": "2026-08-09T15:30:00Z",
        }

        first = record_order_fill(
            order_intent_id=intent_id,
            fill_payload=fill,
            journal=journal,
            portfolio_path=portfolio_path,
        )
        cooldown_before = tracker.get(thesis.thesis_id).sell_cooldown_until
        replay = record_order_fill(
            order_intent_id=intent_id,
            fill_payload=fill,
            journal=journal,
            portfolio_path=portfolio_path,
        )

        self.assertFalse(first["already_recorded"])
        self.assertTrue(replay["already_recorded"])
        self.assertEqual(tracker.get(thesis.thesis_id).sell_cooldown_until, cooldown_before)

    def test_late_broker_fill_upgrades_external_close_without_double_booking(self):
        from artha.fill_finalizer import finalize_broker_sell_fill
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio, Position
        from artha.thesis_tracker import ThesisTracker

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        tracker = ThesisTracker(journal)
        thesis = self._active_thesis(journal, ticker="LATE", position_type="STARTER", entry=100)
        tracker.mark_waiting_for_sell(thesis.thesis_id, reason="Council EXIT")
        portfolio_path = self.tmp_dir / "portfolio.json"
        Portfolio(
            positions=[
                Position(
                    ticker="LATE",
                    asset_type="stock",
                    shares=1,
                    avg_cost=100,
                    opened_at="2026-08-01T14:00:00Z",
                    market_value=100,
                    thesis_id=thesis.thesis_id,
                )
            ]
        ).save(portfolio_path)
        row = {
            "ticker": "LATE",
            "side": "sell",
            "thesis_id": thesis.thesis_id,
            "order_intent_id": "late-intent",
            "evidence_json": {"sell_council": {"confirmed": True, "action": "EXIT"}},
        }
        synthetic = finalize_broker_sell_fill(
            journal=journal,
            execution_row=row,
            ticker="LATE",
            cumulative_quantity=1,
            average_price=95,
            broker_order_id="external-snapshot:LATE:2026-08-10T15:00:00Z",
            source="external_snapshot_reconciliation",
            filled_at="2026-08-10T15:00:00Z",
            portfolio_path=portfolio_path,
            data_quality="estimated_price_from_last_artha_mark",
        )
        actual = finalize_broker_sell_fill(
            journal=journal,
            execution_row=row,
            ticker="LATE",
            cumulative_quantity=1,
            average_price=97,
            broker_order_id="rh-late-exact",
            source="broker_fill_callback",
            filled_at="2026-08-08T15:00:00Z",
            portfolio_path=portfolio_path,
            data_quality="broker_fill",
        )
        replay = finalize_broker_sell_fill(
            journal=journal,
            execution_row=row,
            ticker="LATE",
            cumulative_quantity=1,
            average_price=97,
            broker_order_id="rh-late-exact",
            source="broker_fill_callback",
            filled_at="2026-08-08T15:00:00Z",
            portfolio_path=portfolio_path,
            data_quality="broker_fill",
        )

        self.assertEqual(actual["event_id"], synthetic["event_id"])
        self.assertTrue(actual["external_reconciliation_upgraded"])
        self.assertTrue(replay["already_recorded"])
        self.assertEqual(len(journal.get_sell_trade_events(ticker="LATE")), 1)
        self.assertEqual(len(journal.get_broker_fill_effects()), 1)
        self.assertEqual(len(journal.get_all_post_sell_reviews()), 1)
        event = journal.get_sell_trade_events(ticker="LATE")[0]
        self.assertEqual(event["broker_order_id"], "rh-late-exact")
        self.assertEqual(event["average_price"], 97)
        self.assertEqual(event["realized_pnl"], -3)
        transactions = Portfolio.load(portfolio_path).transactions
        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0]["broker_order_id"], "rh-late-exact")
        self.assertEqual(transactions[0]["price"], 97)

    def test_cumulative_partial_sell_fill_uses_incremental_price_and_one_event(self):
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio, Position
        from artha.robinhood_bridge import record_order_fill

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        thesis = self._active_thesis(journal, ticker="CUM", position_type="STARTER", entry=100)
        portfolio_path = self.tmp_dir / "portfolio.json"
        Portfolio(
            positions=[
                Position(
                    ticker="CUM",
                    asset_type="stock",
                    shares=1,
                    avg_cost=100,
                    opened_at="2026-06-05T14:00:00+00:00",
                    market_value=100,
                )
            ],
            cash_deployed=100,
        ).save(portfolio_path)
        intent_id = "sell-intent-cumulative"
        journal.save_execution_order(
            {
                "order_intent_id": intent_id,
                "ticker": "CUM",
                "side": "sell",
                "order_type": "market",
                "quantity": 0.5,
                "notional": 48,
                "estimated_price": 96,
                "status": "submitted",
                "thesis_id": thesis.thesis_id,
                "evidence_json": {"sell_council": {"confirmed": True, "action": "TRIM"}},
                "request_json": {"symbol": "CUM", "side": "sell", "quantity": "0.5"},
            }
        )

        first = record_order_fill(
            order_intent_id=intent_id,
            fill_payload={
                "id": "rh-cumulative",
                "symbol": "CUM",
                "side": "sell",
                "state": "partially_filled",
                "cumulative_quantity": "0.2",
                "average_price": "90",
            },
            journal=journal,
            portfolio_path=portfolio_path,
        )
        second = record_order_fill(
            order_intent_id=intent_id,
            fill_payload={
                "id": "rh-cumulative",
                "symbol": "CUM",
                "side": "sell",
                "state": "filled",
                "cumulative_quantity": "0.5",
                "average_price": "96",
            },
            journal=journal,
            portfolio_path=portfolio_path,
        )

        self.assertEqual(first["status"], "PASS")
        self.assertEqual(second["status"], "PASS")
        portfolio = Portfolio.load(portfolio_path)
        self.assertAlmostEqual(portfolio.get_position("CUM").shares, 0.5)
        sells = [row for row in portfolio.transactions if row.get("type") == "SELL"]
        self.assertEqual(len(sells), 2)
        self.assertAlmostEqual(sells[0]["price"], 90.0)
        self.assertAlmostEqual(sells[1]["price"], 100.0)
        self.assertAlmostEqual(sum(row["realized_pnl"] for row in sells), -2.0)
        events = journal.get_sell_trade_events(thesis_id=thesis.thesis_id)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]["shares"], 0.5)
        self.assertAlmostEqual(events[0]["proceeds"], 48.0)
        self.assertAlmostEqual(events[0]["realized_pnl"], -2.0)
        self.assertEqual(events[0]["event_type"], "TRIM")
        self.assertEqual(journal.get_pending_post_sell_reviews(), [])

    def test_replayed_old_trim_cannot_reactivate_archived_thesis(self):
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio, Position
        from artha.robinhood_bridge import record_order_fill
        from artha.thesis_tracker import ThesisTracker

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        tracker = ThesisTracker(journal)
        thesis = self._active_thesis(journal, ticker="REPLAY", position_type="STARTER", entry=100)
        portfolio_path = self.tmp_dir / "portfolio.json"
        Portfolio(
            positions=[
                Position(
                    ticker="REPLAY",
                    asset_type="stock",
                    shares=1,
                    avg_cost=100,
                    opened_at="2026-06-05T14:00:00+00:00",
                    market_value=100,
                )
            ],
            cash_deployed=100,
        ).save(portfolio_path)

        def seed_order(intent_id, action, quantity):
            journal.save_execution_order(
                {
                    "order_intent_id": intent_id,
                    "ticker": "REPLAY",
                    "side": "sell",
                    "order_type": "market",
                    "quantity": quantity,
                    "notional": quantity * 95,
                    "estimated_price": 95,
                    "status": "submitted",
                    "thesis_id": thesis.thesis_id,
                    "evidence_json": {"sell_council": {"confirmed": True, "action": action}},
                    "request_json": {"symbol": "REPLAY", "side": "sell", "quantity": str(quantity)},
                }
            )

        seed_order("trim-old", "TRIM", 0.25)
        trim_payload = {
            "id": "rh-trim-old",
            "symbol": "REPLAY",
            "side": "sell",
            "state": "filled",
            "cumulative_quantity": "0.25",
            "average_price": "95",
        }
        record_order_fill(
            order_intent_id="trim-old",
            fill_payload=trim_payload,
            journal=journal,
            portfolio_path=portfolio_path,
        )
        seed_order("exit-new", "EXIT", 0.75)
        record_order_fill(
            order_intent_id="exit-new",
            fill_payload={
                "id": "rh-exit-new",
                "symbol": "REPLAY",
                "side": "sell",
                "state": "filled",
                "cumulative_quantity": "0.75",
                "average_price": "96",
            },
            journal=journal,
            portfolio_path=portfolio_path,
        )

        replayed = record_order_fill(
            order_intent_id="trim-old",
            fill_payload=trim_payload,
            journal=journal,
            portfolio_path=portfolio_path,
        )

        self.assertTrue(replayed["already_recorded"])
        self.assertIsNone(Portfolio.load(portfolio_path).get_position("REPLAY"))
        self.assertEqual(tracker.get(thesis.thesis_id).status, "archived")
        events = journal.get_sell_trade_events(thesis_id=thesis.thesis_id)
        self.assertEqual([event["event_type"] for event in events], ["TRIM", "EXIT"])
        self.assertEqual(len(journal.get_pending_post_sell_reviews()), 1)
        self.assertEqual(len(journal.get_broker_fill_effects()), 2)

    def test_supervisor_fails_filled_sell_that_bypassed_shared_finalizer(self):
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio
        from artha.supervisor import _check_broker_fill_accounting

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        portfolio_path = self.tmp_dir / "portfolio.json"
        Portfolio().save(portfolio_path)
        journal.save_execution_order(
            {
                "order_intent_id": "bypassed-sell",
                "ticker": "MISS",
                "side": "sell",
                "order_type": "market",
                "quantity": 1,
                "notional": 10,
                "estimated_price": 10,
                "status": "filled",
                "broker_order_id": "rh-bypassed-sell",
                "filled_at": "2026-06-06T14:00:00Z",
            }
        )

        result = _check_broker_fill_accounting(journal, portfolio_path=portfolio_path)

        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["missing_effects"], ["rh-bypassed-sell"])
        self.assertEqual(result["missing_events"], ["rh-bypassed-sell"])

    def test_sector_backfill_reaches_portfolio_state_and_intended_risk_gate(self):
        from types import SimpleNamespace

        from artha.council import hard_risk_gate
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio, Position
        from artha.portfolio_risk import build_portfolio_factor_risk
        from artha.portfolio_state import PortfolioStateEngine
        from artha.position_classification import backfill_portfolio_classifications
        from artha.supervisor import _check_position_classification

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        journal.save_decision_features(
            {
                "dossier_path": str(self.tmp_dir / "TECH_dossier.json"),
                "generated_at": "2026-06-04T14:20:00+00:00",
                "ticker": "TECH",
                "final_verdict": "STARTER",
                "sector": "Technology",
                "industry": "Software",
                "feature_json": "{}",
            }
        )
        portfolio_path = self.tmp_dir / "portfolio.json"
        Portfolio(
            positions=[
                Position(
                    ticker="TECH",
                    asset_type="stock",
                    shares=1,
                    avg_cost=40,
                    opened_at="2026-06-05T14:00:00+00:00",
                    current_price=40,
                    market_value=40,
                )
            ],
            cash_available=60,
        ).save(portfolio_path)

        backfill = backfill_portfolio_classifications(
            journal,
            portfolio_path=portfolio_path,
        )
        state = PortfolioStateEngine(portfolio_path).compute_state()

        self.assertEqual(backfill["updated"], ["TECH"])
        self.assertEqual(backfill["unresolved"], [])
        self.assertEqual(state["positions"][0]["sector"], "Technology")
        check = _check_position_classification(portfolio_path=portfolio_path)
        self.assertEqual(check["status"], "PASS", check)
        self.assertAlmostEqual(check["sector_exposures"][0]["pct_nav"], 40.0)
        self.assertAlmostEqual(check["sector_exposures"][0]["pct_invested"], 100.0)

        config = SimpleNamespace(
            MAX_CONCURRENT_POSITIONS=20,
            MAX_INVESTED_PCT=0.90,
            MAX_SECTOR_PCT=0.30,
            MAX_POSITION_PCT=0.20,
            EXPLORATION_MAX_PER_POSITION_PCT=0.05,
        )
        candidate = {"profile": {"sector": "Technology", "industry": "Software"}}
        passed, reason = hard_risk_gate("NEW", candidate, state, [], config)
        self.assertFalse(passed)
        self.assertIn("SECTOR LIMIT", reason)
        risk = build_portfolio_factor_risk("NEW", candidate, state, config, 5.0)
        self.assertAlmostEqual(risk["current_sector_pct"], 40.0)
        self.assertAlmostEqual(risk["current_sector_invested_pct"], 100.0)

    def test_historical_backfill_groups_trims_into_one_closed_episode_idempotently(self):
        from artha.fill_finalizer import backfill_sell_fill_accounting
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio
        from artha.thesis_tracker import ThesisTracker

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        tracker = ThesisTracker(journal)
        thesis = self._active_thesis(journal, ticker="HIST", position_type="STARTER", entry=100)
        tracker.archive_thesis(thesis.thesis_id, exit_price=105, exit_reason="Historical exit")
        portfolio_path = self.tmp_dir / "portfolio.json"
        Portfolio().save(portfolio_path)

        for index, (action, quantity, price) in enumerate(
            [("TRIM", 0.25, 102), ("EXIT", 0.75, 106)],
            start=1,
        ):
            intent_id = f"historical-{index}"
            broker_id = f"rh-historical-{index}"
            filled_at = f"2026-06-0{index + 5}T14:00:00Z"
            journal.save_execution_order(
                {
                    "order_intent_id": intent_id,
                    "ticker": "HIST",
                    "side": "sell",
                    "order_type": "market",
                    "quantity": quantity,
                    "notional": quantity * price,
                    "estimated_price": price,
                    "status": "filled",
                    "broker_order_id": broker_id,
                    "thesis_id": thesis.thesis_id,
                    "filled_at": filled_at,
                    "evidence_json": {"sell_council": {"confirmed": True, "action": action}},
                    "response_json": {
                        "order": {
                            "id": broker_id,
                            "symbol": "HIST",
                            "side": "sell",
                            "state": "filled",
                            "cumulative_quantity": quantity,
                            "average_price": price,
                            "last_transaction_at": filled_at,
                        }
                    },
                }
            )

        first = backfill_sell_fill_accounting(journal, portfolio_path=portfolio_path)
        second = backfill_sell_fill_accounting(journal, portfolio_path=portfolio_path)

        self.assertEqual(first["sell_orders_seen"], 2)
        self.assertEqual(first["events_created"], 2)
        self.assertEqual(second["events_created"], 0)
        events = journal.get_sell_trade_events(thesis_id=thesis.thesis_id)
        self.assertEqual([event["event_type"] for event in events], ["TRIM", "EXIT"])
        self.assertEqual(len(journal.get_broker_fill_effects()), 2)
        self.assertEqual(len(journal.get_all_post_sell_reviews()), 1)
        episodes = journal.get_position_trade_episodes(status="closed")
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["sell_fill_count"], 2)
        self.assertEqual(episodes[0]["trim_count"], 1)
        self.assertAlmostEqual(episodes[0]["shares_sold"], 1.0)

    def test_historical_backfill_reconstructs_fifo_cost_from_broker_fills(self):
        from artha.execution_learning import build_execution_learning_summary
        from artha.fill_finalizer import backfill_sell_fill_accounting
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio
        from artha.thesis_tracker import ThesisTracker

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        thesis = self._active_thesis(journal, ticker="FIFO", entry=100)
        ThesisTracker(journal).update_thesis_fields(
            thesis.thesis_id,
            entry_date="2026-08-01T14:00:30Z",
        )
        portfolio_path = self.tmp_dir / "portfolio.json"
        Portfolio(
            transactions=[
                {
                    "type": "SELL",
                    "ticker": "FIFO",
                    "shares": 0.25,
                    "price": 130,
                    "total": 32.5,
                    "realized_pnl": 7.5,
                    "cost_basis_per_share": 100,
                    "broker_order_id": "rh-fifo-sell",
                    "timestamp": "2026-08-01T14:10:00Z",
                }
            ]
        ).save(portfolio_path)

        for index, (quantity, price, filled_at) in enumerate(
            (
                (0.6, 100.0, "2026-08-01T14:00:00Z"),
                (0.4, 120.0, "2026-08-01T14:05:00Z"),
            ),
            start=1,
        ):
            broker_id = f"rh-fifo-buy-{index}"
            journal.save_execution_order(
                {
                    "order_intent_id": f"fifo-buy-{index}",
                    "ticker": "FIFO",
                    "side": "buy",
                    "order_type": "market",
                    "quantity": quantity,
                    "notional": quantity * price,
                    "estimated_price": price,
                    "status": "filled",
                    "broker_order_id": broker_id,
                    "thesis_id": thesis.thesis_id if index == 2 else None,
                    "filled_at": filled_at,
                    "response_json": {
                        "order": {
                            "id": broker_id,
                            "symbol": "FIFO",
                            "side": "buy",
                            "state": "filled",
                            "cumulative_quantity": quantity,
                            "average_price": price,
                            "last_transaction_at": filled_at,
                            "fees": "0.000000",
                        }
                    },
                }
            )
        journal.save_execution_order(
            {
                "order_intent_id": "fifo-sell",
                "ticker": "FIFO",
                "side": "sell",
                "order_type": "market",
                "quantity": 0.7,
                "notional": 91,
                "estimated_price": 130,
                "status": "filled",
                "broker_order_id": "rh-fifo-sell",
                "thesis_id": thesis.thesis_id,
                "filled_at": "2026-08-01T14:10:00Z",
                "evidence_json": {"sell_council": {"confirmed": True, "action": "TRIM"}},
                "response_json": {
                    "order": {
                        "id": "rh-fifo-sell",
                        "symbol": "FIFO",
                        "side": "sell",
                        "state": "filled",
                        "cumulative_quantity": 0.7,
                        "average_price": 130,
                        "last_transaction_at": "2026-08-01T14:10:00Z",
                        "fees": "0.000000",
                    }
                },
            }
        )

        result = backfill_sell_fill_accounting(journal, portfolio_path=portfolio_path)
        event = journal.get_sell_trade_events(ticker="FIFO")[0]
        episode = journal.get_position_trade_episodes(status="open")[0]
        summary = build_execution_learning_summary(journal, portfolio_path=portfolio_path)

        self.assertEqual(result["broker_fifo_reconstructed_events"], 1)
        self.assertEqual(event["data_quality"], "broker_fill_fifo_reconstruction")
        self.assertAlmostEqual(event["cost_basis_per_share"], 72 / 0.7)
        self.assertAlmostEqual(event["realized_pnl"], 19.0)
        self.assertAlmostEqual(event["position_shares_before"], 1.0)
        self.assertAlmostEqual(event["position_shares_after"], 0.3)
        self.assertEqual(episode["buy_fill_count"], 2)
        self.assertAlmostEqual(episode["shares_bought"], 1.0)
        self.assertEqual(summary["data_quality"]["uncertain_events"], 0)

    def test_post_sell_accounting_refresh_preserves_completed_outcome(self):
        from artha.journal import DecisionJournal
        from artha.opportunity_cost import PostSellTracker

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        tracker = PostSellTracker(journal)
        tracker.record_sell(
            ticker="KEEP",
            thesis_id="keep-thesis",
            sell_price=100,
            sell_reason="test",
            shares=1,
            position_type="STARTER",
            tracking_id="keep-tracking",
            sell_date="2026-06-01",
        )
        journal.save_post_sell_tracking(
            {
                "tracking_id": "keep-tracking",
                "price_5d": 101,
                "price_20d": 102,
                "price_60d": 103,
                "grade": "EARLY",
                "status": "completed",
            }
        )

        tracker.record_sell(
            ticker="KEEP",
            thesis_id="keep-thesis",
            sell_price=100,
            sell_reason="test",
            shares=1,
            position_type="STARTER",
            tracking_id="keep-tracking",
            sell_date="2026-06-01",
            cost_basis=90,
            realized_pnl=10,
            data_quality="broker_fill_fifo_reconstruction",
        )
        row = journal.get_post_sell_review("keep-tracking")

        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["grade"], "EARLY")
        self.assertEqual(row["price_60d"], 103)
        self.assertEqual(row["cost_basis"], 90)
        self.assertEqual(row["data_quality"], "broker_fill_fifo_reconstruction")

    def test_execution_learning_reports_checkpoint_maturity_without_relaxing_gate(self):
        from datetime import date

        from artha.execution_learning import build_execution_learning_summary
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        for tracking_id, sell_date in (("overdue", "2026-05-01"), ("future", "2026-07-15")):
            journal.save_post_sell_tracking(
                {
                    "tracking_id": tracking_id,
                    "ticker": tracking_id.upper(),
                    "sell_date": sell_date,
                    "sell_price": 100,
                    "status": "tracking",
                }
            )
        portfolio_path = self.tmp_dir / "portfolio.json"
        Portfolio().save(portfolio_path)

        summary = build_execution_learning_summary(
            journal,
            portfolio_path=portfolio_path,
            as_of=date(2026, 8, 11),
        )

        self.assertEqual(summary["post_sell"]["overdue_60d_reviews"], 1)
        self.assertEqual(summary["post_sell"]["next_60d_due_date"], "2026-09-13")
        self.assertFalse(summary["post_sell"]["council_learning_ready"])

    def test_historical_backfill_flags_legacy_quantity_conflict(self):
        from artha.execution_learning import build_execution_learning_summary
        from artha.fill_finalizer import backfill_sell_fill_accounting
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio
        from artha.thesis_tracker import ThesisTracker

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        tracker = ThesisTracker(journal)
        thesis = self._active_thesis(journal, ticker="LEGACY", position_type="STARTER", entry=100)
        tracker.archive_thesis(thesis.thesis_id, exit_price=95, exit_reason="Historical exit")
        broker_order_id = "rh-legacy-quantity-conflict"
        portfolio_path = self.tmp_dir / "portfolio.json"
        Portfolio(
            transactions=[
                {
                    "type": "SELL",
                    "ticker": "LEGACY",
                    "shares": 0.25,
                    "price": 95,
                    "total": 23.75,
                    "realized_pnl": -1.25,
                    "cost_basis_per_share": 100,
                    "broker_order_id": broker_order_id,
                    "timestamp": "2026-08-05T14:00:00Z",
                }
            ]
        ).save(portfolio_path)
        journal.save_execution_order(
            {
                "order_intent_id": "legacy-conflict-intent",
                "ticker": "LEGACY",
                "side": "sell",
                "order_type": "market",
                "quantity": 1,
                "notional": 95,
                "estimated_price": 95,
                "status": "filled",
                "broker_order_id": broker_order_id,
                "thesis_id": thesis.thesis_id,
                "filled_at": "2026-08-05T14:00:00Z",
                "evidence_json": {"sell_council": {"confirmed": True, "action": "EXIT"}},
                "response_json": {
                    "order": {
                        "id": broker_order_id,
                        "symbol": "LEGACY",
                        "side": "sell",
                        "state": "filled",
                        "cumulative_quantity": 1,
                        "average_price": 95,
                        "last_transaction_at": "2026-08-05T14:00:00Z",
                    }
                },
            }
        )

        backfill_sell_fill_accounting(journal, portfolio_path=portfolio_path)
        event = journal.get_sell_trade_events(ticker="LEGACY")[0]
        summary = build_execution_learning_summary(journal, portfolio_path=portfolio_path)

        self.assertEqual(event["data_quality"], "legacy_portfolio_quantity_conflict")
        self.assertEqual(summary["data_quality"]["quantity_conflict_events"], 1)
        self.assertEqual(summary["data_quality"]["uncertain_events"], 1)
        self.assertTrue(summary["data_quality"]["expectancy_is_provisional"])

    def test_execution_learning_counts_episodes_not_individual_trims(self):
        from artha.execution_learning import build_execution_learning_summary
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio, Position
        from artha.supervisor import _check_execution_learning

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        for event_id, thesis_id, event_type, pnl, proceeds, quality in (
            ("win-trim", "win-thesis", "TRIM", 1.0, 11.0, "broker_fill"),
            ("win-exit", "win-thesis", "EXIT", 2.0, 12.0, "broker_fill"),
            ("loss-exit", "loss-thesis", "EXIT", -2.0, 8.0, "broker_fill_with_inferred_cost"),
        ):
            journal.save_sell_trade_event(
                {
                    "event_id": event_id,
                    "ticker": thesis_id.split("-")[0].upper(),
                    "thesis_id": thesis_id,
                    "event_type": event_type,
                    "shares": 1,
                    "average_price": proceeds,
                    "cost_basis_per_share": 10,
                    "proceeds": proceeds,
                    "realized_pnl": pnl,
                    "sell_reason": "test",
                    "source": "unit_test",
                    "data_quality": quality,
                    "filled_at": "2026-06-06T14:00:00Z",
                }
            )
        for episode_id, thesis_id, pnl, return_pct in (
            ("episode-win", "win-thesis", 3.0, 15.0),
            ("episode-loss", "loss-thesis", -2.0, -10.0),
        ):
            journal.save_position_trade_episode(
                {
                    "episode_id": episode_id,
                    "thesis_id": thesis_id,
                    "ticker": thesis_id.split("-")[0].upper(),
                    "status": "closed",
                    "realized_pnl": pnl,
                    "return_pct": return_pct,
                    "source": "unit_test",
                }
            )
        journal.save_post_sell_tracking(
            {
                "tracking_id": "post-win",
                "ticker": "WIN",
                "sell_date": "2026-06-06T14:00:00Z",
                "sell_price": 12,
                "grade": "EARLY",
                "status": "completed",
            }
        )
        portfolio_path = self.tmp_dir / "portfolio.json"
        Portfolio(
            positions=[
                Position(
                    ticker="OPEN",
                    asset_type="stock",
                    shares=2,
                    avg_cost=10,
                    opened_at="2026-06-05T14:00:00Z",
                    market_value=24,
                    position_type="STARTER",
                )
            ]
        ).save(portfolio_path)

        summary = build_execution_learning_summary(journal, portfolio_path=portfolio_path)

        self.assertEqual(summary["scope"], "observational_only")
        self.assertFalse(summary["strategy_effects_enabled"])
        self.assertEqual(summary["episodes"]["closed"], 2)
        self.assertEqual(summary["episodes"]["wins"], 1)
        self.assertEqual(summary["episodes"]["losses"], 1)
        self.assertEqual(summary["episodes"]["win_rate_pct"], 50.0)
        self.assertEqual(summary["episodes"]["expectancy_dollars"], 0.5)
        self.assertEqual(summary["sell_events"]["total"], 3)
        self.assertEqual(summary["sell_events"]["trims"], 1)
        self.assertEqual(summary["post_sell"]["grade_counts"], {"EARLY": 1})
        self.assertEqual(summary["current_sizing"]["median_position_dollars"], 24.0)
        self.assertEqual(summary["data_quality"]["status"], "provisional")
        self.assertTrue(summary["data_quality"]["expectancy_is_provisional"])
        self.assertEqual(summary["data_quality"]["inferred_cost_events"], 1)
        check = _check_execution_learning(journal, portfolio_path=portfolio_path)
        self.assertEqual(check["status"], "WARN")
        self.assertIn("provisional", check["message"])
        self.assertTrue(check["observational_only"])

    def test_pending_exit_watchdog_never_archives_from_zero_shares_alone(self):
        import asyncio
        from types import SimpleNamespace

        from artha.scheduler import ArthaScheduler

        thesis = SimpleNamespace(
            thesis_id="pending-zero-thesis",
            ticker="ZERO",
            status="pending_exit",
        )
        tracker = Mock()
        tracker.get_all_monitored.return_value = [thesis]
        tracker.get.return_value = thesis
        scheduler = ArthaScheduler.__new__(ArthaScheduler)
        scheduler.sell_engine = SimpleNamespace(
            journal=Mock(),
            tracker=tracker,
        )
        scheduler.telegram = SimpleNamespace(enabled=False)
        scheduler._load_robinhood_position_snapshot = Mock(
            return_value={
                "fresh": True,
                "positions": [],
                "orders": [],
            }
        )

        with patch(
            "artha.robinhood_bridge.sync_orders_to_artha",
            return_value={"status": "PASS", "filled": []},
        ) as sync_orders:
            asyncio.run(scheduler._run_pending_exit_watchdog())

        sync_orders.assert_called_once()
        tracker.archive_thesis.assert_not_called()
        tracker.reactivate_for_retry.assert_not_called()

    def test_scheduler_snapshot_loader_preserves_orders_for_exit_watchdog(self):
        from artha.config import Config
        from artha.scheduler import ArthaScheduler

        snapshot_path = self.tmp_dir / "latest_snapshot.json"
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "robinhood_mcp",
            "account": {"account_number": "222233334"},
            "portfolio": {"buying_power": "50", "cash": "50", "total_value": "60"},
            "positions": [
                {"symbol": "KEEP", "quantity": "1", "average_buy_price": "10"}
            ],
            "orders": [
                {
                    "id": "rh-watchdog-fill",
                    "symbol": "GONE",
                    "side": "sell",
                    "state": "filled",
                    "cumulative_quantity": "1",
                    "average_price": "11",
                }
            ],
        }
        snapshot_path.write_text(json.dumps(payload), encoding="utf-8")
        scheduler = ArthaScheduler.__new__(ArthaScheduler)

        with patch.object(Config, "ROBINHOOD_AGENTIC_ACCOUNT_NUMBER", "222233334"), patch.object(
            Config,
            "ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE",
            str(snapshot_path),
        ):
            loaded = scheduler._load_robinhood_position_snapshot()

        self.assertTrue(loaded["fresh"])
        self.assertEqual([row["id"] for row in loaded["orders"]], ["rh-watchdog-fill"])

    def test_manual_sell_cli_uses_shared_fill_finalizer_for_trim_and_exit(self):
        from argparse import Namespace

        from artha.cli import portfolio_update
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio, Position
        from artha.thesis_tracker import ThesisTracker

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        thesis = self._active_thesis(
            journal,
            ticker="MANUAL",
            position_type="STARTER",
            entry=100,
        )
        portfolio_path = self.tmp_dir / "portfolio.json"
        Portfolio(
            positions=[
                Position(
                    ticker="MANUAL",
                    asset_type="stock",
                    shares=1,
                    avg_cost=100,
                    opened_at="2026-06-05T14:00:00Z",
                    market_value=100,
                    thesis_id=thesis.thesis_id,
                    position_type="STARTER",
                )
            ],
            cash_deployed=100,
            cash_available=0,
        ).save(portfolio_path)

        with patch.object(portfolio_update, "PORTFOLIO_FILE", portfolio_path), \
                patch.object(portfolio_update, "DecisionJournal", return_value=journal), \
                patch.object(portfolio_update, "_out") as output:
            portfolio_update.cmd_sell(
                Namespace(ticker="MANUAL", shares=0.5, price=110, reason="Manual trim")
            )
            portfolio_update.cmd_sell(
                Namespace(ticker="MANUAL", shares=0.5, price=95, reason="Manual exit")
            )

        self.assertEqual(output.call_count, 2)
        portfolio = Portfolio.load(portfolio_path)
        self.assertIsNone(portfolio.get_position("MANUAL"))
        self.assertAlmostEqual(portfolio.cash_available, 102.5)
        self.assertAlmostEqual(portfolio.cash_deployed, 0.0)
        events = journal.get_sell_trade_events(thesis_id=thesis.thesis_id)
        self.assertEqual([row["event_type"] for row in events], ["TRIM", "EXIT"])
        self.assertEqual([row["source"] for row in events], [
            "manual_cli_confirmation",
            "manual_cli_confirmation",
        ])
        self.assertEqual(len(journal.get_all_post_sell_reviews()), 1)
        self.assertEqual(
            ThesisTracker(journal).get(thesis.thesis_id).status,
            "archived",
        )

    def test_snapshot_order_sync_does_not_double_apply_already_reflected_sell(self):
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio, Position
        from artha.robinhood_bridge import sync_orders_to_artha

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        thesis = self._active_thesis(journal, ticker="SYNCSELL", position_type="STARTER", entry=100)
        portfolio_path = self.tmp_dir / "portfolio.json"
        Portfolio(
            positions=[
                Position(
                    ticker="SYNCSELL",
                    asset_type="stock",
                    shares=0.75,
                    avg_cost=100,
                    opened_at="2026-06-05T14:00:00+00:00",
                    market_value=75,
                    thesis_id=thesis.thesis_id,
                )
            ]
        ).save(portfolio_path)
        intent_id = "sell-intent-snapshot-reflected"
        journal.save_execution_order(
            {
                "order_intent_id": intent_id,
                "ticker": "SYNCSELL",
                "side": "sell",
                "order_type": "market",
                "quantity": 0.25,
                "notional": 23.75,
                "estimated_price": 95,
                "status": "submitted",
                "broker_order_id": "rh-snapshot-sell",
                "thesis_id": thesis.thesis_id,
                "evidence_json": {
                    "sell_council": {"confirmed": True, "action": "TRIM"}
                },
                "request_json": {"symbol": "SYNCSELL", "side": "sell", "quantity": "0.25"},
            }
        )

        result = sync_orders_to_artha(
            {
                "positions": [
                    {
                        "symbol": "SYNCSELL",
                        "quantity": "0.75",
                        "average_buy_price": "100.00",
                    }
                ],
                "orders": [
                    {
                        "id": "rh-snapshot-sell",
                        "symbol": "SYNCSELL",
                        "side": "sell",
                        "state": "filled",
                        "cumulative_quantity": "0.25",
                        "average_price": "95",
                        "last_transaction_at": "2026-06-06T14:00:00Z",
                    }
                ],
            },
            journal=journal,
            portfolio_path=portfolio_path,
        )

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["filled"][0]["fill"]["source"], "broker_snapshot_already_reflected")
        position = Portfolio.load(portfolio_path).get_position("SYNCSELL")
        self.assertAlmostEqual(position.shares, 0.75)
        self.assertEqual(journal.get_execution_order_by_intent_id(intent_id)["status"], "filled")
        self.assertEqual(journal.get_thesis(thesis.thesis_id)["status"], "active")
        sells = [row for row in Portfolio.load(portfolio_path).transactions if row.get("type") == "SELL"]
        self.assertEqual(len(sells), 1)
        self.assertEqual(sells[0]["broker_order_id"], "rh-snapshot-sell")
        self.assertEqual(len(journal.get_sell_trade_events(thesis_id=thesis.thesis_id)), 1)

    def test_in_memory_snapshot_sync_preserves_filled_order_history(self):
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio, Position
        from artha.robinhood_bridge import sync_snapshot_to_artha

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        thesis = self._active_thesis(journal, ticker="LIVEFILL", position_type="STARTER", entry=100)
        portfolio_path = self.tmp_dir / "portfolio.json"
        Portfolio(
            positions=[
                Position(
                    ticker="LIVEFILL",
                    asset_type="stock",
                    shares=0.75,
                    avg_cost=100,
                    opened_at="2026-06-05T14:00:00+00:00",
                    market_value=75,
                    thesis_id=thesis.thesis_id,
                )
            ]
        ).save(portfolio_path)
        intent_id = "sell-intent-live-snapshot"
        broker_order_id = "rh-live-snapshot-sell"
        journal.save_execution_order(
            {
                "order_intent_id": intent_id,
                "ticker": "LIVEFILL",
                "side": "sell",
                "order_type": "market",
                "quantity": 0.25,
                "notional": 23.75,
                "estimated_price": 95,
                "status": "submitted",
                "broker_order_id": broker_order_id,
                "thesis_id": thesis.thesis_id,
                "evidence_json": {"sell_council": {"confirmed": True, "action": "TRIM"}},
                "request_json": {"symbol": "LIVEFILL", "side": "sell", "quantity": "0.25"},
            }
        )

        result = sync_snapshot_to_artha(
            {
                "source": "robinhood_mcp",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "positions": [
                    {
                        "symbol": "LIVEFILL",
                        "quantity": "0.75",
                        "average_buy_price": "100.00",
                        "market_price": "95.00",
                    }
                ],
                "orders": [
                    {
                        "id": broker_order_id,
                        "symbol": "LIVEFILL",
                        "side": "sell",
                        "state": "filled",
                        "cumulative_quantity": "0.25",
                        "average_price": "95",
                        "last_transaction_at": "2026-06-06T14:00:00Z",
                    }
                ],
            },
            journal=journal,
            portfolio_path=portfolio_path,
        )

        self.assertEqual(result["order_sync"]["status"], "PASS")
        self.assertEqual(len(result["order_sync"]["filled"]), 1)
        execution = journal.get_execution_order_by_intent_id(intent_id)
        self.assertEqual(execution["status"], "filled")
        self.assertEqual(execution["filled_at"], "2026-06-06T14:00:00Z")
        self.assertEqual(len(journal.get_sell_trade_events(thesis_id=thesis.thesis_id)), 1)

    def test_snapshot_order_sync_repairs_filled_sell_missing_from_portfolio(self):
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio, Position
        from artha.robinhood_bridge import sync_orders_to_artha
        from artha.thesis_tracker import ThesisTracker

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        thesis = self._active_thesis(journal, ticker="STALE", position_type="STARTER", entry=100)
        ThesisTracker(journal).archive_thesis(
            thesis.thesis_id,
            exit_price=95,
            exit_reason="Robinhood sell order filled.",
        )
        portfolio_path = self.tmp_dir / "portfolio.json"
        Portfolio(
            positions=[
                Position(
                    ticker="STALE",
                    asset_type="stock",
                    shares=1,
                    avg_cost=100,
                    opened_at="2026-06-05T14:00:00+00:00",
                    market_value=100,
                    thesis_id=thesis.thesis_id,
                )
            ]
        ).save(portfolio_path)
        intent_id = "sell-intent-stale-portfolio"
        broker_order_id = "rh-stale-sell"
        journal.save_execution_order(
            {
                "order_intent_id": intent_id,
                "ticker": "STALE",
                "side": "sell",
                "order_type": "market",
                "quantity": 1,
                "notional": 95,
                "estimated_price": 95,
                "status": "filled",
                "broker_order_id": broker_order_id,
                "thesis_id": thesis.thesis_id,
                "filled_at": "2026-06-06T14:00:00Z",
                "evidence_json": {
                    "sell_council": {"confirmed": True, "action": "EXIT"}
                },
                "request_json": {"symbol": "STALE", "side": "sell", "quantity": "1"},
                "response_json": {
                    "artha_fill_applied": True,
                    "artha_fill_broker_order_id": broker_order_id,
                    "artha_fill_quantity": 1,
                    "artha_fill_average_price": 95,
                },
            }
        )
        journal.save_trade_action(
            {
                "action_id": "ta_stale_sell",
                "status": "filled",
                "action_type": "auto_sell",
                "ticker": "STALE",
                "side": "sell",
                "thesis_id": thesis.thesis_id,
                "order_intent_id": intent_id,
            }
        )

        result = sync_orders_to_artha(
            {
                "positions": [],
                "orders": [],
            },
            journal=journal,
            portfolio_path=portfolio_path,
        )

        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["filled"][0]["fill"]["already_recorded"])
        self.assertTrue(result["filled"][0]["fill"]["portfolio_repaired"])
        portfolio = Portfolio.load(portfolio_path)
        self.assertIsNone(portfolio.get_position("STALE"))
        sells = [t for t in portfolio.transactions if t.get("type") == "SELL" and t.get("ticker") == "STALE"]
        self.assertEqual(len(sells), 1)
        self.assertIn(broker_order_id, sells[0]["notes"])
        self.assertEqual(journal.get_thesis(thesis.thesis_id)["status"], "archived")
        self.assertEqual(len(journal.get_pending_post_sell_reviews()), 1)
        self.assertEqual(len(journal.get_sell_trade_events(thesis_id=thesis.thesis_id)), 1)

    def test_snapshot_order_sync_does_not_repair_fully_accounted_history(self):
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio, Position
        from artha.robinhood_bridge import sync_orders_to_artha

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        thesis = self._active_thesis(journal, ticker="HISTORY", position_type="STARTER", entry=100)
        portfolio_path = self.tmp_dir / "portfolio.json"
        Portfolio(
            positions=[
                Position(
                    ticker="HISTORY",
                    asset_type="stock",
                    shares=0.75,
                    avg_cost=100,
                    opened_at="2026-06-05T14:00:00+00:00",
                    market_value=75,
                    thesis_id=thesis.thesis_id,
                )
            ]
        ).save(portfolio_path)
        intent_id = "sell-intent-accounted-history"
        broker_order_id = "rh-accounted-history"
        journal.save_execution_order(
            {
                "order_intent_id": intent_id,
                "ticker": "HISTORY",
                "side": "sell",
                "order_type": "market",
                "quantity": 0.25,
                "notional": 23.75,
                "estimated_price": 95,
                "status": "filled",
                "broker_order_id": broker_order_id,
                "thesis_id": thesis.thesis_id,
                "notes": "Original execution note",
                "response_json": {
                    "artha_fill_applied": True,
                    "artha_fill_broker_order_id": broker_order_id,
                    "artha_fill_quantity": 0.25,
                    "artha_fill_average_price": 95,
                },
            }
        )
        journal.save_broker_fill_effect(
            {
                "event_key": "fill-accounted-history",
                "broker_order_id": broker_order_id,
                "order_intent_id": intent_id,
                "ticker": "HISTORY",
                "side": "sell",
                "state": "filled",
                "cumulative_quantity": 0.25,
                "average_price": 95,
                "applied_quantity": 0.25,
                "source": "historical_execution_backfill",
                "portfolio_effect_status": "historical_unreconciled",
                "thesis_effect_status": "historical",
                "tracking_effect_status": "historical",
            }
        )
        cooldown_before = journal.get_thesis(thesis.thesis_id)["sell_cooldown_until"]

        result = sync_orders_to_artha(
            {
                "positions": [{"symbol": "HISTORY", "quantity": "0.75"}],
                "orders": [
                    {
                        "id": broker_order_id,
                        "symbol": "HISTORY",
                        "side": "sell",
                        "state": "filled",
                        "cumulative_quantity": "0.25",
                        "average_price": "95",
                        "last_transaction_at": "2026-06-06T14:00:00Z",
                    }
                ],
            },
            journal=journal,
            portfolio_path=portfolio_path,
        )

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["filled"], [])
        self.assertEqual(result["updated"][0]["filled_at"], "2026-06-06T14:00:00Z")
        self.assertEqual(Portfolio.load(portfolio_path).transactions, [])
        self.assertEqual(
            journal.get_execution_order_by_intent_id(intent_id)["filled_at"],
            "2026-06-06T14:00:00Z",
        )
        self.assertEqual(
            journal.get_execution_order_by_intent_id(intent_id)["notes"],
            "Original execution note",
        )
        self.assertEqual(
            journal.get_thesis(thesis.thesis_id)["sell_cooldown_until"],
            cooldown_before,
        )

    def test_auto_sell_authorization_requires_sell_council_provenance(self):
        from datetime import datetime, timedelta, timezone
        from unittest.mock import patch

        from artha.config import Config
        from artha.journal import DecisionJournal
        from artha.robinhood_bridge import _auto_sell_authorization_gate

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        thesis = self._active_thesis(journal, ticker="AUTH", position_type="STARTER", entry=20)
        now = datetime.now(timezone.utc)
        row = {
            "action_id": "ta_auth",
            "action_type": "auto_sell",
            "ticker": "AUTH",
            "side": "sell",
            "status": "review_ready",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "expires_at": (now + timedelta(minutes=30)).isoformat(),
            "thesis_id": thesis.thesis_id,
            "payload_json": json.dumps(
                {
                    "intent": {
                        "ticker": "AUTH",
                        "side": "sell",
                        "quantity": 1,
                        "estimated_price": 20,
                        "evidence": {},
                    }
                }
            ),
        }
        snapshot = {
            "status": "PASS",
            "fresh": True,
            "portfolio": {"buying_power": 100, "cash": 100, "total_value": 100},
            "positions": [{"symbol": "AUTH", "quantity": "1", "market_price": "20"}],
            "orders": [],
        }
        old = (
            Config.ROBINHOOD_AUTO_SELL_ENABLED,
            Config.ROBINHOOD_REVIEW_ONLY,
            Config.ROBINHOOD_DRY_RUN_ONLY,
            Config.ROBINHOOD_AGENTIC_ENABLED,
            Config.ROBINHOOD_KILL_SWITCH,
        )
        try:
            Config.ROBINHOOD_AUTO_SELL_ENABLED = True
            Config.ROBINHOOD_REVIEW_ONLY = False
            Config.ROBINHOOD_DRY_RUN_ONLY = False
            Config.ROBINHOOD_AGENTIC_ENABLED = True
            Config.ROBINHOOD_KILL_SWITCH = False
            with patch("artha.scheduler.MarketHours.is_market_open", return_value=True), \
                    patch("artha.robinhood_bridge.load_robinhood_snapshot", return_value=snapshot):
                gate = _auto_sell_authorization_gate(row, journal)
            self.assertFalse(gate["passed"])
            self.assertTrue(
                any("Sell Council approval" in reason for reason in gate["reasons"])
            )
        finally:
            (
                Config.ROBINHOOD_AUTO_SELL_ENABLED,
                Config.ROBINHOOD_REVIEW_ONLY,
                Config.ROBINHOOD_DRY_RUN_ONLY,
                Config.ROBINHOOD_AGENTIC_ENABLED,
                Config.ROBINHOOD_KILL_SWITCH,
            ) = old

    def test_agentic_execution_officer_is_used_for_auto_sell(self):
        from unittest.mock import patch

        from artha.journal import DecisionJournal
        from artha.openclaw_robinhood_handler import run_agentic_auto_buy_clearance_from_responses

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        journal.save_trade_action(
            {
                "action_id": "ta_agentic_sell",
                "status": "review_ready",
                "action_type": "auto_sell",
                "ticker": "TEST",
                "side": "sell",
            }
        )
        operation = {
            "success": True,
            "operation": "auto_tradability_review_then_place_equity_order",
            "action_id": "ta_agentic_sell",
            "side": "sell",
            "action_type": "auto_sell",
        }
        officer_result = {
            "allow_place": True,
            "status": "PASS",
            "reason": "All sell execution checks passed.",
            "officer_used": True,
            "tool_trace": [],
        }
        with patch(
            "artha.openclaw_robinhood_handler.build_auto_buy_operation",
            return_value=operation,
        ), patch(
            "artha.openclaw_robinhood_handler.run_agentic_execution_officer",
            return_value=officer_result,
        ) as officer:
            result = run_agentic_auto_buy_clearance_from_responses(
                "ta_agentic_sell",
                quote_response={"data": {}},
                tradability_response={"data": {}},
                review_response={"data": {}},
                journal=journal,
            )
        self.assertTrue(result["allow_place"])
        officer.assert_called_once()

    def test_auto_sell_final_clearance_prompt_does_not_require_extra_human_confirmation(self):
        from artha.execution_officer import build_robinhood_review_clearance_prompt

        prompt = build_robinhood_review_clearance_prompt(
            action={
                "action_id": "ta_auto_sell",
                "ticker": "SNX",
                "side": "sell",
                "action_type": "auto_sell",
                "status": "review_clear",
            },
            review_response={
                "data": {
                    "symbol": "SNX",
                    "side": "sell",
                    "type": "market",
                    "quantity": "0.014338",
                    "order_checks": {},
                    "market_data_disclosure": "Bid $238.47 x 200 N",
                }
            },
            tradability_response={
                "data": {
                    "results": [
                        {
                            "symbol": "SNX",
                            "state": "active",
                            "tradeable": True,
                            "fractional_tradability": "tradable",
                        }
                    ]
                }
            },
            recorded_review={
                "status": "review_clear",
                "review_gate": {
                    "passed": True,
                    "status": "PASS",
                    "reasons": [],
                    "checks": {
                        "order_checks_classification": {"blocking": False},
                        "tradability": {"passed": True},
                    },
                },
            },
        )

        self.assertIn("do not treat the Robinhood review tool's generic user-confirmation guide as missing data", prompt)
        self.assertIn("No extra human confirmation is required for auto_buy/auto_sell", prompt)
        self.assertIn("Manual actions require a Place request before final clearance", prompt)

    def test_in_process_auto_sell_does_not_disable_agentic_officer(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from artha.journal import DecisionJournal
        from artha.openclaw_robinhood_handler import handle_auto_buy_action

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        journal.save_trade_action(
            {
                "action_id": "ta_inprocess_sell",
                "status": "review_ready",
                "action_type": "auto_sell",
                "ticker": "TEST",
                "side": "sell",
            }
        )
        operation = {
            "success": True,
            "operation": "auto_tradability_review_then_place_equity_order",
            "action_id": "ta_inprocess_sell",
            "side": "sell",
            "action_type": "auto_sell",
        }
        with patch(
            "artha.openclaw_robinhood_handler.build_auto_buy_operation",
            return_value=operation,
        ), patch(
            "artha.openclaw_robinhood_handler.run_agentic_execution_officer",
            return_value={
                "allow_place": False,
                "status": "BLOCKED",
                "reason": "Unit block.",
                "tool_trace": [],
            },
        ) as officer:
            result = handle_auto_buy_action(
                "ta_inprocess_sell",
                SimpleNamespace(),
                journal=journal,
            )

        self.assertEqual(result["status"], "BLOCKED")
        self.assertNotIn("use_llm", officer.call_args.kwargs)

    def test_robinhood_pilot_account_cap_is_350(self):
        from artha.config import Config

        self.assertEqual(Config.ROBINHOOD_PILOT_MAX_ACCOUNT_VALUE, 350.0)
        self.assertEqual(Config.ROBINHOOD_MAX_TRADES_PER_DAY, 5)
        self.assertEqual(Config.SELL_MAX_EXITS_PER_DAY, 5)
        self.assertEqual(Config.MAX_INVESTED_PCT, 0.90)

    def test_review_only_buy_allows_nonfatal_supervisor_warn(self):
        from datetime import datetime, timezone

        from artha.execution import RobinhoodExecutionGuardrails, build_order_intent
        from artha.journal import DecisionJournal

        class AlwaysOpen:
            def is_market_open(self, now=None):
                return True

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        journal.save_supervisor_run(
            {
                "generated_at": "2026-06-05T20:00:00+00:00",
                "severity": "WARN",
                "payload": {
                    "checks": [
                        {"name": "position_monitoring", "status": "WARN"},
                        {"name": "recent_logs", "status": "WARN"},
                    ]
                },
            }
        )
        journal.save_decision_features(
            {
                "dossier_path": "unit-dossier.json",
                "generated_at": "2026-06-05T20:00:00+00:00",
                "ticker": "TEST",
                "final_verdict": "STARTER",
                "opportunity_score": 72,
                "evidence_count": 20,
                "feature_json": "{}",
            }
        )
        intent = build_order_intent(
            ticker="TEST",
            side="buy",
            notional=25.0,
            quantity=0.25,
            limit_price=100.0,
            estimated_price=100.0,
            decision_dossier_path="unit-dossier.json",
            dry_run=True,
        )
        result = RobinhoodExecutionGuardrails(market_hours=AlwaysOpen()).evaluate(
            intent,
            market_data={"price": 100.0, "volume": 1_000_000, "bid": 99.95, "ask": 100.0},
            journal=journal,
            now=datetime(2026, 6, 5, 15, 0, tzinfo=timezone.utc),
        )
        self.assertTrue(result.passed, result.reasons)
        self.assertTrue(result.checks["supervisor"]["buy_gate"]["allowed"])

    def test_review_only_buy_blocks_supervisor_fail(self):
        from datetime import datetime, timezone

        from artha.execution import RobinhoodExecutionGuardrails, build_order_intent
        from artha.journal import DecisionJournal

        class AlwaysOpen:
            def is_market_open(self, now=None):
                return True

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        journal.save_supervisor_run(
            {
                "generated_at": "2026-06-05T20:00:00+00:00",
                "severity": "FAIL",
                "payload": {"checks": [{"name": "position_monitoring", "status": "FAIL"}]},
            }
        )
        journal.save_decision_features(
            {
                "dossier_path": "unit-dossier.json",
                "generated_at": "2026-06-05T20:00:00+00:00",
                "ticker": "TEST",
                "final_verdict": "STARTER",
                "opportunity_score": 72,
                "evidence_count": 20,
                "feature_json": "{}",
            }
        )
        intent = build_order_intent(
            ticker="TEST",
            side="buy",
            notional=25.0,
            quantity=0.25,
            limit_price=100.0,
            estimated_price=100.0,
            decision_dossier_path="unit-dossier.json",
            dry_run=True,
        )
        result = RobinhoodExecutionGuardrails(market_hours=AlwaysOpen()).evaluate(
            intent,
            market_data={"price": 100.0, "volume": 1_000_000, "bid": 99.95, "ask": 100.0},
            journal=journal,
            now=datetime(2026, 6, 5, 15, 0, tzinfo=timezone.utc),
        )
        self.assertFalse(result.passed)
        self.assertIn("failing check", result.reasons[0].lower())

    def test_sell_dossier_writer_creates_audit_artifact(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        import artha.sell_dossier as sell_dossier

        with patch.object(sell_dossier, "SELL_DOSSIER_DIR", self.tmp_dir / "sell_dossiers"):
            path = sell_dossier.write_sell_dossier(
                decision=SimpleNamespace(
                    ticker="TEST",
                    action="HOLD",
                    sell_score=35,
                    trigger_type="unit",
                    session_id="session1234",
                ),
                thesis=SimpleNamespace(ticker="TEST", thesis_id="thesis123", position_type="STARTER"),
                stock_data=_sample_stock(price=100.0),
                macro_data={"fed_funds": 4.5},
                trigger_type="unit",
            )
        self.assertTrue(Path(path).exists())
        payload = json.loads(Path(path).read_text())
        self.assertEqual(payload["kind"], "sell_review")
        self.assertEqual(payload["ticker"], "TEST")

    def test_sell_council_accepts_evidence_gated_cio_adjustment(self):
        from types import SimpleNamespace

        from artha.journal import DecisionJournal
        from artha.sell_council import SellCouncil

        thesis = SimpleNamespace(
            thesis_id="thesis123",
            ticker="TEST",
            position_type="STARTER",
            entry_price=100.0,
            entry_date="2026-06-01T15:00:00+00:00",
            entry_regime="fear",
            hard_stop_price=80.0,
            trailing_stop_price=None,
            thesis_summary="Buy depended on customer retention and revenue recovery.",
            invalidation_conditions=["Biggest customer loss or renewed revenue contraction"],
            thesis_health_score=100,
            days_held=30,
            in_minimum_hold=False,
            in_cooldown=False,
        )
        analyst_report = (
            "**SELL VERDICT:** HOLD\n"
            "**FUNDAMENTAL SELL SCORE:** 50\n"
            "**CONFIDENCE:** 7\n"
            "Customer loss confirmed in filing guidance with renewed revenue contraction."
        )
        synthesis = (
            "**ACTION: EXIT**\n"
            "**KEY REASONS:**\n"
            "- Customer loss and revenue contraction threaten the original thesis.\n"
            "```json\n"
            "{"
            "\"sell_score\": 70, \"action\": \"EXIT\", \"thesis_status\": \"DAMAGED\", "
            "\"health_score\": 55, \"fundamental_score\": 50, \"technical_score\": 50, "
            "\"contrarian_score\": 50, \"cio_score_adjustment\": 20, "
            "\"cio_adjustment_category\": \"confirmed_thesis_break\", "
            "\"cio_adjustment_evidence\": [\"Customer loss confirmed in filing guidance with renewed revenue contraction\"], "
            "\"cio_adjustment_reason\": \"Customer loss and revenue contraction directly threaten the thesis\", "
            "\"next_review_days\": 7, \"is_urgent\": false, \"trim_pct\": 25, \"confidence\": 8"
            "}\n"
            "```"
        )

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        with patch("artha.sell_council._run_sell_fundamental", return_value=analyst_report), \
             patch("artha.sell_council._run_sell_technical", return_value=analyst_report), \
             patch("artha.sell_council._run_sell_contrarian", return_value=analyst_report), \
             patch("artha.sell_council.ChatGPTBackendClient.chat", return_value=synthesis), \
             patch("artha.sell_dossier.write_sell_dossier", return_value=str(self.tmp_dir / "sell.json")):
            decision = SellCouncil(journal=journal).run_sell_review(thesis, _sample_stock(price=95.0))

        self.assertIsNotNone(decision)
        self.assertEqual(decision.sell_score, 70.0)
        self.assertEqual(decision.action, "TRIM")
        self.assertEqual(decision.cio_adjustment, 20.0)
        self.assertEqual(decision.scoring_audit["cio_adjustment"]["status"], "accepted")

    def test_sell_council_rejects_unsupported_cio_adjustment(self):
        from types import SimpleNamespace

        from artha.journal import DecisionJournal
        from artha.sell_council import SellCouncil

        thesis = SimpleNamespace(
            thesis_id="thesis456",
            ticker="TEST",
            position_type="STARTER",
            entry_price=100.0,
            entry_date="2026-06-01T15:00:00+00:00",
            entry_regime="fear",
            hard_stop_price=80.0,
            trailing_stop_price=None,
            thesis_summary="Business remains stable.",
            invalidation_conditions=["Revenue deterioration"],
            thesis_health_score=100,
            days_held=30,
            in_minimum_hold=False,
            in_cooldown=False,
        )
        analyst_report = (
            "**SELL VERDICT:** HOLD\n"
            "**FUNDAMENTAL SELL SCORE:** 50\n"
            "**CONFIDENCE:** 7\n"
            "Revenue and margins remain stable with no confirmed thesis break."
        )
        synthesis = (
            "```json\n"
            "{"
            "\"sell_score\": 90, \"action\": \"EXIT\", \"thesis_status\": \"INTACT\", "
            "\"health_score\": 90, \"fundamental_score\": 50, \"technical_score\": 50, "
            "\"contrarian_score\": 50, \"cio_score_adjustment\": 20, "
            "\"cio_adjustment_category\": \"other\", "
            "\"cio_adjustment_evidence\": [\"asteroid supply shock unrelated phrase\"], "
            "\"cio_adjustment_reason\": \"asteroid supply shock\", "
            "\"next_review_days\": 21, \"is_urgent\": true, \"trim_pct\": null, \"confidence\": 8"
            "}\n"
            "```"
        )

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        with patch("artha.sell_council._run_sell_fundamental", return_value=analyst_report), \
             patch("artha.sell_council._run_sell_technical", return_value=analyst_report), \
             patch("artha.sell_council._run_sell_contrarian", return_value=analyst_report), \
             patch("artha.sell_council.ChatGPTBackendClient.chat", return_value=synthesis), \
             patch("artha.sell_dossier.write_sell_dossier", return_value=str(self.tmp_dir / "sell.json")):
            decision = SellCouncil(journal=journal).run_sell_review(thesis, _sample_stock(price=100.0))

        self.assertIsNotNone(decision)
        self.assertEqual(decision.sell_score, 50.0)
        self.assertEqual(decision.action, "HOLD")
        self.assertEqual(decision.cio_adjustment, 0.0)
        self.assertEqual(decision.scoring_audit["cio_adjustment"]["status"], "rejected_unsupported_evidence")

    def test_sell_council_applies_deterministic_rule_adjustments(self):
        from types import SimpleNamespace

        from artha.sell_council import SellCouncil

        thesis = SimpleNamespace(
            ticker="TEST",
            position_type="TACTICAL_BUY",
            entry_price=100.0,
            hard_stop_price=80.0,
            entry_regime="fear",
            thesis_health_score=100,
            days_held=10,
        )
        council = SellCouncil()
        adjustments = council._build_rule_adjustments(
            thesis=thesis,
            stock_data=_sample_stock(price=95.0),
            current_regime="greed",
            analyst_reports=["No invalidation condition triggered."],
        )
        rule_score, rule_total, forced = council._apply_rule_adjustments(50.0, adjustments)

        self.assertIsNone(forced)
        self.assertEqual(rule_total, 15)
        self.assertEqual(rule_score, 65.0)
        self.assertEqual(council._action_from_score(rule_score, "TACTICAL_BUY"), "TRIM")

    def test_sell_council_score_mapping_overrides_passive_cio_action(self):
        from types import SimpleNamespace

        from artha.journal import DecisionJournal
        from artha.sell_council import SellCouncil

        thesis = SimpleNamespace(
            thesis_id="thesis789",
            ticker="TEST",
            position_type="STARTER",
            entry_price=100.0,
            entry_date="2026-06-01T15:00:00+00:00",
            entry_regime="fear",
            hard_stop_price=80.0,
            trailing_stop_price=None,
            thesis_summary="Starter thesis.",
            invalidation_conditions=[],
            thesis_health_score=100,
            days_held=30,
            in_minimum_hold=False,
            in_cooldown=False,
        )
        analyst_report = (
            "**SELL VERDICT:** EXIT\n"
            "**FUNDAMENTAL SELL SCORE:** 80\n"
            "**CONFIDENCE:** 8\n"
            "Major support and thesis evidence deteriorated."
        )
        synthesis = (
            "```json\n"
            "{"
            "\"sell_score\": 80, \"action\": \"HOLD\", \"thesis_status\": \"DAMAGED\", "
            "\"health_score\": 50, \"fundamental_score\": 80, \"technical_score\": 80, "
            "\"contrarian_score\": 80, \"cio_score_adjustment\": 0, "
            "\"cio_adjustment_category\": \"none\", \"cio_adjustment_evidence\": [], "
            "\"cio_adjustment_reason\": \"none\", \"next_review_days\": 7, "
            "\"is_urgent\": false, \"trim_pct\": null, \"confidence\": 8"
            "}\n"
            "```"
        )

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        with patch("artha.sell_council._run_sell_fundamental", return_value=analyst_report), \
             patch("artha.sell_council._run_sell_technical", return_value=analyst_report), \
             patch("artha.sell_council._run_sell_contrarian", return_value=analyst_report), \
             patch("artha.sell_council.ChatGPTBackendClient.chat", return_value=synthesis), \
             patch("artha.sell_dossier.write_sell_dossier", return_value=str(self.tmp_dir / "sell.json")):
            decision = SellCouncil(journal=journal).run_sell_review(thesis, _sample_stock(price=95.0))

        self.assertIsNotNone(decision)
        self.assertEqual(decision.sell_score, 80.0)
        self.assertEqual(decision.action, "EXIT")
        self.assertEqual(decision.scoring_audit["cio_requested_action"], "HOLD")
        self.assertEqual(decision.scoring_audit["score_mapped_action"], "EXIT")

    def test_time_checkpoint_requests_review_without_declaring_thesis_break(self):
        from datetime import datetime, timedelta, timezone
        from types import SimpleNamespace

        from artha.journal import DecisionJournal
        from artha.sell_engine import SellEngine
        from artha.thesis_tracker import ThesisTracker

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        tracker = ThesisTracker(journal)
        entry = datetime.now(timezone.utc) - timedelta(days=8)
        thesis = tracker.create_thesis(
            ticker="REVIEW",
            position_type="STARTER",
            thesis_summary="Review the business after one week.",
            invalidation_conditions=[
                {
                    "metric": "days_held",
                    "op": "gte",
                    "value": 7,
                    "description": "Mandatory one-week re-underwrite.",
                }
            ],
        )
        thesis = tracker.activate_thesis(
            thesis.thesis_id,
            100,
            entry_date=entry.isoformat(),
            shares=1,
        )
        tracker.update_thesis_fields(thesis.thesis_id, last_review_date=entry.isoformat())
        thesis = tracker.get(thesis.thesis_id)
        engine = SellEngine(journal=journal, collector=SimpleNamespace())

        signals = engine._check_thesis_break(thesis, 101)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].signal_type, "thesis_review")
        self.assertEqual(signals[0].action_recommended, "HOLD")
        self.assertIn("not evidence that the thesis failed", signals[0].message)

        tracker.update_review_date(thesis.thesis_id)
        self.assertEqual(engine._check_thesis_break(tracker.get(thesis.thesis_id), 101), [])

    def test_price_condition_remains_machine_checked_thesis_break(self):
        from types import SimpleNamespace

        from artha.journal import DecisionJournal
        from artha.sell_engine import SellEngine
        from artha.thesis_tracker import ThesisTracker

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        tracker = ThesisTracker(journal)
        thesis = tracker.create_thesis(
            ticker="BREAK",
            position_type="STARTER",
            thesis_summary="Price structure must hold.",
            invalidation_conditions=[
                {
                    "metric": "price",
                    "op": "lte",
                    "value": 90,
                    "description": "Support failure invalidates the entry.",
                }
            ],
        )
        thesis = tracker.activate_thesis(thesis.thesis_id, 100, shares=1)
        signal = SellEngine(
            journal=journal,
            collector=SimpleNamespace(),
        )._check_thesis_break(thesis, 89)[0]

        self.assertEqual(signal.signal_type, "thesis_break")
        self.assertEqual(signal.action_recommended, "EXIT")

    def test_review_word_triggered_does_not_add_false_thesis_break_bonus(self):
        from types import SimpleNamespace

        from artha.sell_council import SellCouncil

        thesis = SimpleNamespace(
            ticker="TEST",
            position_type="STARTER",
            entry_price=100,
            hard_stop_price=80,
            entry_regime="neutral",
            thesis_health_score=100,
            days_held=7,
        )
        report = (
            "**SELL VERDICT:** HOLD\n**FUNDAMENTAL SELL SCORE:** 30\n"
            "**THESIS STATUS:** INTACT\n[REVIEW] days_held gte 7: TRIGGERED"
        )
        council = SellCouncil()
        review_adjustments = council._build_rule_adjustments(
            thesis,
            _sample_stock(price=100),
            "neutral",
            [report],
            trigger_type="thesis_review",
        )
        break_adjustments = council._build_rule_adjustments(
            thesis,
            _sample_stock(price=89),
            "neutral",
            [report],
            trigger_type="thesis_break",
        )

        self.assertNotIn(
            "invalidation_condition_triggered",
            {row["name"] for row in review_adjustments},
        )
        self.assertIn(
            "invalidation_condition_triggered",
            {row["name"] for row in break_adjustments},
        )

    def test_thesis_metadata_extracts_narrative_and_preserves_business_conditions(self):
        from artha.council import (
            _extract_thesis_summary,
            _merge_invalidation_conditions,
            _normalize_decision_json,
        )

        synthesis = """PRIMARY DRIVER: RISK_ADJUSTED
**COUNCIL CONSENSUS:** 2-1
**RECOMMENDED ACTION:** STARTER

**SYNTHESIS:**
The company has improving free cash flow and a defensible balance sheet, but the entry should remain small while demand is verified.

**KEY RISK TO WATCH:** demand deterioration
```json
{"opportunity_score": 72}
```"""
        decision = _normalize_decision_json(
            {
                "verdict": "STARTER",
                "conviction_0_10": 7,
                "entry_zone": None,
                "stop_price": 90,
                "target_price": 120,
                "invalidation_conditions": [
                    {"metric": "price", "op": "lte", "value": 90, "description": "support"},
                    {"metric": "days_held", "op": "gte", "value": 7, "description": "review"},
                ],
                "half_size": False,
            }
        )
        merged = _merge_invalidation_conditions(
            ["Free cash flow deteriorates while leverage rises"],
            decision["invalidation_conditions"],
        )

        self.assertTrue(_extract_thesis_summary(synthesis).startswith("The company has improving"))
        self.assertEqual(merged[0], "Free cash flow deteriorates while leverage rises")
        self.assertEqual(merged[1]["effect"], "invalidate")
        self.assertEqual(merged[2]["effect"], "review")

    def test_scale_out_marker_is_recorded_only_after_confirmed_trim_fill(self):
        from types import SimpleNamespace

        from artha.config import Config
        from artha.fill_finalizer import finalize_broker_sell_fill
        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio, Position
        from artha.sell_engine import SellEngine
        from artha.thesis_tracker import ThesisTracker

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        tracker = ThesisTracker(journal)
        thesis = tracker.create_thesis(
            ticker="SCALE",
            position_type="STARTER",
            thesis_summary="Unit thesis",
            invalidation_conditions=["Fundamentals deteriorate"],
        )
        thesis = tracker.activate_thesis(thesis.thesis_id, 100, shares=1)
        engine = SellEngine(journal=journal, collector=SimpleNamespace())
        old_schedule = Config.SELL_SCALE_OUT_STARTER
        Config.SELL_SCALE_OUT_STARTER = {"+10%": 0.25}
        try:
            signals = engine._check_scale_out(thesis, 111)
        finally:
            Config.SELL_SCALE_OUT_STARTER = old_schedule

        self.assertEqual(signals[0].trim_pct, 0.25)
        self.assertEqual(signals[0].completion_markers, ["+10%"])
        self.assertNotIn("+10%", tracker.get(thesis.thesis_id).scale_out_completed)

        portfolio_path = self.tmp_dir / "portfolio.json"
        Portfolio(
            positions=[
                Position(
                    ticker="SCALE",
                    asset_type="stock",
                    shares=1,
                    avg_cost=100,
                    opened_at="2026-08-01T14:00:00Z",
                    market_value=111,
                    thesis_id=thesis.thesis_id,
                    position_type="STARTER",
                )
            ]
        ).save(portfolio_path)
        row = {
            "ticker": "SCALE",
            "side": "sell",
            "thesis_id": thesis.thesis_id,
            "order_intent_id": "scale-intent",
            "evidence_json": {
                "action": "TRIM",
                "completion_markers": ["+10%"],
                "sell_council": {"confirmed": True, "action": "TRIM"},
            },
        }
        first = finalize_broker_sell_fill(
            journal=journal,
            execution_row=row,
            ticker="SCALE",
            cumulative_quantity=0.25,
            average_price=111,
            broker_order_id="rh-scale",
            order_intent_id="scale-intent",
            filled_at="2026-08-11T15:30:00Z",
            portfolio_path=portfolio_path,
        )
        replay = finalize_broker_sell_fill(
            journal=journal,
            execution_row=row,
            ticker="SCALE",
            cumulative_quantity=0.25,
            average_price=111,
            broker_order_id="rh-scale",
            order_intent_id="scale-intent",
            filled_at="2026-08-11T15:30:00Z",
            portfolio_path=portfolio_path,
        )

        self.assertEqual(first["completion_markers"], ["+10%"])
        self.assertTrue(replay["already_recorded"])
        self.assertEqual(tracker.get(thesis.thesis_id).scale_out_completed.count("+10%"), 1)

    def test_sell_queue_fails_closed_when_duplicate_state_cannot_be_read(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from artha.journal import DecisionJournal
        from artha.portfolio import Portfolio, Position
        from artha.scheduler import ArthaScheduler

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        thesis = self._active_thesis(journal, ticker="DEDUPE", entry=100)
        portfolio_path = self.tmp_dir / "portfolio.json"
        Portfolio(
            positions=[
                Position(
                    ticker="DEDUPE",
                    asset_type="stock",
                    shares=1,
                    avg_cost=100,
                    opened_at="2026-08-01T14:00:00Z",
                    market_value=100,
                    thesis_id=thesis.thesis_id,
                )
            ]
        ).save(portfolio_path)
        scheduler = ArthaScheduler()
        decision = SimpleNamespace(
            action="EXIT",
            session_id="sell-dedupe-unit",
            dossier_path=str(self.tmp_dir / "sell-dedupe.json"),
            sell_score=90,
            confidence=8,
            trigger_type="hard_stop",
        )

        with patch("artha.scheduler.PORTFOLIO_FILE", portfolio_path), \
                patch.object(journal, "get_trade_actions", side_effect=RuntimeError("db unavailable")):
            result = scheduler._prepare_robinhood_sell_review(
                thesis=thesis,
                action="EXIT",
                current_price=90,
                journal=journal,
                trigger_type="hard_stop",
                reason="unit",
                sell_decision=decision,
            )

        self.assertIsNone(result)
        self.assertEqual(journal.get_execution_orders(), [])

    def test_sell_council_retries_instead_of_fabricating_hold_on_analyst_outage(self):
        from artha.journal import DecisionJournal
        from artha.sell_council import SellCouncil

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        thesis = self._active_thesis(journal, ticker="OUTAGE", entry=100)
        report = (
            "**SELL VERDICT:** HOLD\n**FUNDAMENTAL SELL SCORE:** 25\n"
            "**CONFIDENCE:** 7\n**THESIS STATUS:** INTACT\nBusiness remains sound."
        )
        with patch("artha.sell_council._run_sell_fundamental", return_value=report), \
                patch("artha.sell_council._run_sell_technical", return_value=None), \
                patch("artha.sell_council._run_sell_contrarian", return_value=None), \
                patch("artha.sell_council.ChatGPTBackendClient.chat") as synthesis:
            decision = SellCouncil(journal).run_sell_review(thesis, _sample_stock(price=100))

        self.assertIsNone(decision)
        synthesis.assert_not_called()

    def test_sell_council_retries_when_cio_synthesis_is_unavailable(self):
        from artha.journal import DecisionJournal
        from artha.sell_council import SellCouncil

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        thesis = self._active_thesis(journal, ticker="NOCIO", entry=100)
        report = (
            "**SELL VERDICT:** HOLD\n**FUNDAMENTAL SELL SCORE:** 25\n"
            "**CONFIDENCE:** 7\n**THESIS STATUS:** INTACT\nBusiness remains sound."
        )
        with patch("artha.sell_council._run_sell_fundamental", return_value=report), \
                patch("artha.sell_council._run_sell_technical", return_value=report), \
                patch("artha.sell_council._run_sell_contrarian", return_value=report), \
                patch("artha.sell_council.ChatGPTBackendClient.chat", side_effect=RuntimeError("model outage")):
            decision = SellCouncil(journal).run_sell_review(thesis, _sample_stock(price=100))

        self.assertIsNone(decision)

    def test_sell_council_retries_on_malformed_cio_numeric_fields(self):
        from artha.sell_council import _validate_sell_synthesis_payload

        valid = {
            "sell_score": 55,
            "action": "TRIM",
            "thesis_status": "WEAKENED",
            "next_review_days": 14,
            "confidence": 7,
            "health_score": 65,
            "trim_pct": 20,
        }
        self.assertEqual(_validate_sell_synthesis_payload(valid), (True, ""))
        for field, bad_value in (
            ("sell_score", "not-a-number"),
            ("action", "BUY"),
            ("thesis_status", "UNKNOWN"),
            ("next_review_days", 0),
            ("confidence", 11),
            ("trim_pct", 150),
        ):
            malformed = dict(valid)
            malformed[field] = bad_value
            ok, reason = _validate_sell_synthesis_payload(malformed)
            self.assertFalse(ok, field)
            self.assertTrue(reason, field)

    def test_sell_learning_context_is_sample_gated_and_reason_specific(self):
        from artha.execution_learning import build_sell_learning_context
        from artha.journal import DecisionJournal

        journal = DecisionJournal(db_path=self.tmp_dir / "artha.db")
        for index in range(20):
            journal.save_post_sell_tracking(
                {
                    "tracking_id": f"learning-{index}",
                    "ticker": f"T{index}",
                    "sell_date": "2026-06-01",
                    "sell_price": 100,
                    "sell_reason": "earnings risk trim" if index < 10 else "trailing stop",
                    "grade": "EARLY" if index < 6 else "CORRECT",
                    "status": "completed",
                }
            )

        immature = build_sell_learning_context(journal, minimum_completed=21)
        mature = build_sell_learning_context(journal, minimum_completed=20)

        self.assertFalse(immature["ready"])
        self.assertIn("20/21", immature["context"])
        self.assertTrue(mature["ready"])
        self.assertIn("earnings_risk", mature["context"])
        self.assertIn("trailing_stop", mature["context"])
        self.assertIn("weak calibration", mature["context"])


class TestBrokerCapacityAndScanLifecycle(unittest.TestCase):
    @staticmethod
    def _snapshot(*, generated_at: str = "2026-08-05T16:00:00+00:00", buying_power: float = 156.53):
        return {
            "generated_at": generated_at,
            "status": "PASS",
            "fresh": True,
            "portfolio": {
                "total_value": "364.04",
                "cash": "195.44",
                "equity_value": "168.60",
                "buying_power": {
                    "buying_power": str(buying_power),
                    "unleveraged_buying_power": str(buying_power),
                },
            },
            "positions": [],
        }

    def test_capacity_separates_total_cash_from_spendable_buying_power(self):
        from artha.broker_capacity import calculate_broker_capacity

        capacity = calculate_broker_capacity(
            self._snapshot(),
            now=datetime(2026, 8, 5, 16, 2, tzinfo=timezone.utc),
        )

        self.assertTrue(capacity["usable"], capacity["reasons"])
        self.assertAlmostEqual(capacity["total_account_value"], 364.04, places=2)
        self.assertAlmostEqual(capacity["invested_value"], 168.60, places=2)
        self.assertAlmostEqual(capacity["buying_power"], 156.53, places=2)
        self.assertAlmostEqual(capacity["unsettled_or_unspendable_cash"], 38.91, places=2)
        self.assertAlmostEqual(capacity["exposure_headroom"], 159.036, places=3)
        self.assertAlmostEqual(capacity["deployable_amount"], 156.53, places=2)
        self.assertTrue(capacity["scan_buy_enabled"])

    def test_capacity_fails_closed_on_stale_snapshot(self):
        from artha.broker_capacity import calculate_broker_capacity

        capacity = calculate_broker_capacity(
            self._snapshot(generated_at="2026-08-05T15:00:00+00:00"),
            now=datetime(2026, 8, 5, 16, 0, tzinfo=timezone.utc),
            max_age_minutes=10,
        )

        self.assertFalse(capacity["usable"])
        self.assertFalse(capacity["scan_buy_enabled"])
        self.assertTrue(any("stale" in reason.lower() for reason in capacity["reasons"]))

    def test_execution_exposure_uses_total_account_value_not_buying_power_denominator(self):
        from artha.execution import build_order_intent, evaluate_broker_snapshot_guardrails

        intent = build_order_intent(
            ticker="TEST",
            side="buy",
            notional=125.0,
            quantity=1.25,
            limit_price=100.0,
            estimated_price=100.0,
            decision_dossier_path="unit.json",
            dry_run=True,
        )
        with patch("artha.execution.Config.ROBINHOOD_MAX_POSITION_DOLLARS", 200.0):
            result = evaluate_broker_snapshot_guardrails(intent, self._snapshot(), 125.0)

        self.assertTrue(result["passed"], result["reasons"])
        self.assertAlmostEqual(result["checks"]["account_nav"], 364.04, places=2)
        self.assertAlmostEqual(result["checks"]["cash_available"], 156.53, places=2)
        self.assertLess(result["checks"]["projected_invested_pct"], 0.90)

    def test_scheduled_scan_stops_before_research_when_capacity_is_closed(self):
        from artha.scheduler import ArthaScheduler

        scheduler = ArthaScheduler.__new__(ArthaScheduler)
        scheduler._broker_buy_scan_capacity = Mock(
            return_value={"status": "PASS", "scan_buy_enabled": False, "deployable_amount": 0.0}
        )
        scheduler._pause_scheduled_buy_scan_for_capacity = Mock()
        scheduler._run_full_scan_and_council_sync()

        scheduler._pause_scheduled_buy_scan_for_capacity.assert_called_once()

    def test_paused_scan_auto_claims_once_when_sale_reopens_capacity(self):
        from zoneinfo import ZoneInfo

        from artha.scheduler import ArthaScheduler, MarketHours

        temp_dir = Path(tempfile.mkdtemp())
        scheduler = ArthaScheduler.__new__(ArthaScheduler)
        scheduler.market_hours = MarketHours()
        scheduler.ct_tz = ZoneInfo("America/Chicago")
        scheduler.telegram = Mock(enabled=False)
        scheduler._BUY_SCAN_CAPACITY_STATE_FILE = temp_dir / "capacity.json"
        scheduler._save_buy_scan_capacity_state(
            {
                "status": "paused",
                "scan_date_ct": "2026-08-05",
                "catchup_pending": True,
            }
        )
        scheduler._broker_buy_scan_capacity = Mock(
            return_value={
                "status": "PASS",
                "scan_buy_enabled": True,
                "deployable_amount": 25.0,
                "total_account_value": 350.0,
                "invested_value": 290.0,
                "invested_pct": 290.0 / 350.0,
                "buying_power": 25.0,
                "exposure_headroom": 25.0,
                "reasons": [],
            }
        )
        at_noon_ct = datetime(2026, 8, 5, 12, 0, tzinfo=scheduler.ct_tz).astimezone(timezone.utc)

        self.assertTrue(scheduler._should_run_buy_capacity_catchup(at_noon_ct))
        self.assertFalse(scheduler._should_run_buy_capacity_catchup(at_noon_ct + timedelta(minutes=1)))
        state = scheduler._load_buy_scan_capacity_state()
        self.assertEqual(state["status"], "catchup_claimed")
        self.assertFalse(state["catchup_pending"])


class TestAccuracyReportHardening(unittest.TestCase):
    def test_modern_buy_side_verdicts_are_graded(self):
        from artha.accuracy import AccuracyTracker, Recommendation

        path = Path(tempfile.mkdtemp()) / "accuracy.json"
        tracker = AccuracyTracker(path=path)
        rec = Recommendation(
            ticker="DAR",
            verdict="TACTICAL_BUY",
            consensus="2-1",
            entry_price="50.00",
            recommended_action="TACTICAL_BUY small starter",
            allocation="5%",
            fundamental_verdict="BUY",
            fundamental_confidence=7,
            technical_verdict="HOLD",
            technical_confidence=7,
            contrarian_verdict="HOLD",
            contrarian_confidence=7,
        )
        tracker.record_recommendation(rec)
        stored = tracker._load()[0]
        graded = tracker.grade_recommendation("DAR", stored["timestamp"], current_price=55.0)

        self.assertIsNotNone(graded)
        self.assertEqual(graded["grade"], "CORRECT")
        stats = tracker.get_summary_stats()
        self.assertEqual(stats["usable_graded"], 1)
        self.assertEqual(stats["ungraded"], 0)
        self.assertEqual(stats["strict_accuracy"], 100.0)

    def test_accuracy_report_labels_weighted_and_current_unproven(self):
        from artha.accuracy import AccuracyTracker

        path = Path(tempfile.mkdtemp()) / "accuracy.json"
        tracker = AccuracyTracker(path=path)
        tracker._save(
            [
                {
                    "ticker": "OLD",
                    "timestamp": "2026-05-01T00:00:00+00:00",
                    "status": "GRADED",
                    "grade": "PARTIALLY_CORRECT",
                    "price_change_pct": "2.0",
                    "analyst_grades": {},
                },
                {
                    "ticker": "NEW",
                    "timestamp": "2026-06-10T00:00:00+00:00",
                    "status": "PENDING",
                    "grade": "",
                    "price_change_pct": "0",
                    "analyst_grades": {},
                },
            ]
        )
        report = tracker.format_monthly_report()

        self.assertIn("No final usable grades yet", report)
        self.assertIn("Weighted historical score", report)
        self.assertIn("Strict correct-only score", report)


if __name__ == "__main__":
    unittest.main()
