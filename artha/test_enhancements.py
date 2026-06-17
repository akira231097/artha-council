"""Unit tests for council enhancements: PreBrief, MomentumTracker, and integration.

Run with:
    python -m artha.test_enhancements
"""
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path


class TestPreBrief(unittest.TestCase):
    """Tests for the PreBrief system."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        from artha.pre_brief import PreBrief
        self.brief = PreBrief(data_dir=self.tmp_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_empty_state(self):
        """get_brief on empty store returns 'no notable events' message."""
        result = self.brief.get_brief("NVDA")
        self.assertIn("No notable events", result)

    def test_record_and_retrieve(self):
        """record_event creates a retrievable entry for the ticker."""
        self.brief.record_event(
            ticker="AAPL",
            event_type="news_alert",
            severity="WARNING",
            summary="Analyst downgrade from Goldman",
            source="sentinel",
        )
        result = self.brief.get_brief("AAPL")
        self.assertIn("AAPL", result)
        self.assertIn("Analyst downgrade from Goldman", result)

    def test_ticker_isolation(self):
        """Events for one ticker don't show up in another ticker's brief."""
        self.brief.record_event("AAPL", "news_alert", "INFO", "AAPL news", "test")
        result = self.brief.get_brief("NVDA")
        self.assertIn("No notable events", result)
        self.assertNotIn("AAPL news", result)

    def test_pruning_old_events(self):
        """Events older than 14 days are pruned by _prune()."""
        old_ts = (datetime.now(timezone.utc) - timedelta(days=15)).isoformat()
        payload = {
            "events": [
                {
                    "ticker": "TSLA",
                    "event_type": "news_alert",
                    "severity": "INFO",
                    "summary": "Old news",
                    "source": "test",
                    "timestamp": old_ts,
                }
            ],
            "last_pruned": None,
        }
        # Write stale payload directly
        brief_file = self.tmp_dir / "pre_briefs.json"
        brief_file.write_text(json.dumps(payload))

        pruned = self.brief._prune(payload)
        self.assertEqual(len(pruned["events"]), 0, "Old events should be pruned")

    def test_days_filter(self):
        """get_brief with days=1 only returns events from the last day."""
        old_ts = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        payload = {
            "events": [
                {
                    "ticker": "MSFT",
                    "event_type": "news_alert",
                    "severity": "INFO",
                    "summary": "Three days ago event",
                    "source": "test",
                    "timestamp": old_ts,
                }
            ],
            "last_pruned": None,
        }
        (self.tmp_dir / "pre_briefs.json").write_text(json.dumps(payload))

        result = self.brief.get_brief("MSFT", days=1)
        self.assertIn("No notable events", result)

    def test_get_council_pre_brief(self):
        """get_council_pre_brief combines multiple tickers."""
        self.brief.record_event("NVDA", "price_move", "INFO", "NVDA up 5%", "test")
        self.brief.record_event("AAPL", "news_alert", "WARNING", "AAPL downgrade", "test")

        result = self.brief.get_council_pre_brief(["NVDA", "AAPL"])
        self.assertIn("NVDA", result)
        self.assertIn("AAPL", result)

    def test_atomic_write(self):
        """After record_event, the JSON file is valid and contains the event."""
        self.brief.record_event("GOOG", "earnings", "INFO", "Beat by 5%", "test")
        data = json.loads((self.tmp_dir / "pre_briefs.json").read_text())
        events = data.get("events", [])
        self.assertTrue(any(e["ticker"] == "GOOG" for e in events))


class TestMomentumTracker(unittest.TestCase):
    """Tests for the MomentumTracker system."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        from artha.momentum_tracker import MomentumTracker
        self.tracker = MomentumTracker(data_dir=self.tmp_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_candidates(self, scores: dict[str, float], date: str = "2026-03-10") -> list[dict]:
        return [
            {
                "symbol": ticker,
                "combined_score": score,
                "return_12m": 20.0,
                "return_3m": 5.0,
                "return_1m": 2.0,
            }
            for ticker, score in scores.items()
        ]

    def test_empty_returns_empty_dict(self):
        """get_momentum_delta on unknown ticker returns empty dict."""
        result = self.tracker.get_momentum_delta("NVDA")
        self.assertEqual(result, {})

    def test_first_appearance_is_new(self):
        """First scan marks the ticker as 'new'."""
        candidates = self._make_candidates({"NVDA": 25.3}, date="2026-03-10")
        self.tracker.record_scores(candidates, "2026-03-10")

        delta = self.tracker.get_momentum_delta("NVDA")
        self.assertEqual(delta["trend"], "new")
        self.assertEqual(delta["current_score"], 25.3)
        self.assertIsNone(delta["previous_score"])

    def test_accelerating_detection(self):
        """Second scan with higher score shows 'accelerating' trend."""
        self.tracker.record_scores(self._make_candidates({"NVDA": 18.1}), "2026-03-10")
        self.tracker.record_scores(self._make_candidates({"NVDA": 25.3}), "2026-03-12")

        delta = self.tracker.get_momentum_delta("NVDA")
        self.assertEqual(delta["trend"], "accelerating")
        self.assertAlmostEqual(delta["delta"], 7.2, places=1)

    def test_decelerating_detection(self):
        """Second scan with lower score shows 'decelerating' trend."""
        self.tracker.record_scores(self._make_candidates({"AAPL": 22.0}), "2026-03-10")
        self.tracker.record_scores(self._make_candidates({"AAPL": 19.5}), "2026-03-12")

        delta = self.tracker.get_momentum_delta("AAPL")
        self.assertEqual(delta["trend"], "decelerating")
        self.assertAlmostEqual(delta["delta"], -2.5, places=1)

    def test_stable_detection(self):
        """Small delta (< threshold) shows 'stable' trend."""
        self.tracker.record_scores(self._make_candidates({"MSFT": 20.0}), "2026-03-10")
        self.tracker.record_scores(self._make_candidates({"MSFT": 20.5}), "2026-03-12")

        delta = self.tracker.get_momentum_delta("MSFT")
        self.assertEqual(delta["trend"], "stable")

    def test_no_duplicate_for_same_date(self):
        """Recording same date twice keeps only one entry."""
        self.tracker.record_scores(self._make_candidates({"PLTR": 15.0}), "2026-03-10")
        self.tracker.record_scores(self._make_candidates({"PLTR": 16.0}), "2026-03-10")  # same date

        data = self.tracker._load()
        entries = data["scores"].get("PLTR", [])
        dates = [e["date"] for e in entries]
        self.assertEqual(len(dates), len(set(dates)), "Duplicate dates should be deduplicated")

    def test_enrich_ranked_candidates(self):
        """enrich_ranked_candidates adds momentum_delta and momentum_trend keys."""
        self.tracker.record_scores(self._make_candidates({"AMD": 10.0}), "2026-03-10")
        self.tracker.record_scores(self._make_candidates({"AMD": 18.0}), "2026-03-12")

        ranked = [{"symbol": "AMD", "combined_score": 18.0}]
        enriched = self.tracker.enrich_ranked_candidates(ranked)

        self.assertIn("momentum_delta", enriched[0])
        self.assertIn("momentum_trend", enriched[0])
        self.assertEqual(enriched[0]["momentum_trend"], "accelerating")

    def test_get_acceleration_summary(self):
        """get_acceleration_summary returns multi-line formatted string."""
        self.tracker.record_scores(self._make_candidates({"TSLA": 10.0}), "2026-03-10")
        self.tracker.record_scores(self._make_candidates({"TSLA": 18.0}), "2026-03-12")

        summary = self.tracker.get_acceleration_summary(["TSLA"])
        self.assertIn("TSLA", summary)
        self.assertIn("accelerating", summary)

    def test_pruning(self):
        """Entries older than 30 days are pruned."""
        old_date = (datetime.now(timezone.utc) - timedelta(days=31)).strftime("%Y-%m-%d")
        payload = {
            "scores": {
                "NVDA": [{"date": old_date, "score": 25.0, "return_12m": None, "return_3m": None, "return_1m": None}]
            },
            "last_pruned": None,
        }
        (self.tmp_dir / "momentum_history.json").write_text(json.dumps(payload))

        pruned = self.tracker._prune(payload)
        self.assertEqual(len(pruned["scores"].get("NVDA", [])), 0)


class TestAnalystPromptIntegration(unittest.TestCase):
    """Verify that pre_brief and momentum_context are accepted by analyst functions
    and injected into prompt templates correctly."""

    def test_fundamental_prompt_contains_new_sections(self):
        """FUNDAMENTAL_ANALYST template includes {pre_brief} and {momentum_context}."""
        from artha.prompts import FUNDAMENTAL_ANALYST
        self.assertIn("{pre_brief}", FUNDAMENTAL_ANALYST)
        self.assertIn("{momentum_context}", FUNDAMENTAL_ANALYST)

    def test_technical_prompt_contains_new_sections(self):
        """TECHNICAL_ANALYST template includes {pre_brief} and {momentum_context}."""
        from artha.prompts import TECHNICAL_ANALYST
        self.assertIn("{pre_brief}", TECHNICAL_ANALYST)
        self.assertIn("{momentum_context}", TECHNICAL_ANALYST)

    def test_contrarian_prompt_contains_new_sections(self):
        """CONTRARIAN_ANALYST template includes {pre_brief} and {momentum_context}."""
        from artha.prompts import CONTRARIAN_ANALYST
        self.assertIn("{pre_brief}", CONTRARIAN_ANALYST)
        self.assertIn("{momentum_context}", CONTRARIAN_ANALYST)

    def test_analyst_functions_accept_pre_brief_param(self):
        """All 3 analyst functions accept pre_brief and momentum_context without TypeError."""
        import inspect
        from artha.analysts import run_fundamental_analyst, run_technical_analyst, run_contrarian_analyst

        for fn in [run_fundamental_analyst, run_technical_analyst, run_contrarian_analyst]:
            sig = inspect.signature(fn)
            params = set(sig.parameters.keys())
            self.assertIn("pre_brief", params, f"{fn.__name__} missing pre_brief param")
            self.assertIn("momentum_context", params, f"{fn.__name__} missing momentum_context param")

    def test_prompt_format_renders_correctly(self):
        """FUNDAMENTAL_ANALYST.format() works with the new parameters."""
        from artha.prompts import FUNDAMENTAL_ANALYST
        rendered = FUNDAMENTAL_ANALYST.format(
            context_header="TEST CONTEXT",
            intelligence_brief="TEST BRIEF",
            pre_brief="RECENT EVENTS: test event",
            momentum_context="MOMENTUM TREND: ACCELERATING",
            data='{"test": true}',
        )
        self.assertIn("RECENT EVENTS: test event", rendered)
        self.assertIn("MOMENTUM TREND: ACCELERATING", rendered)
        self.assertIn("TEST BRIEF", rendered)


if __name__ == "__main__":
    # Run with verbose output
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestPreBrief))
    suite.addTests(loader.loadTestsFromTestCase(TestMomentumTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalystPromptIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    import sys
    sys.exit(0 if result.wasSuccessful() else 1)
