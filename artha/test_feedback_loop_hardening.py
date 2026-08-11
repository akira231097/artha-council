"""Regression tests for Artha's guarded feedback-loop plumbing."""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch


class TestPreBriefIntegrity(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_structured_filters_do_not_treat_string_false_as_verified(self) -> None:
        from artha.pre_brief import PreBrief

        brief = PreBrief(self.path)
        brief.record_event(
            "AAA",
            "thesis_impact",
            "CRITICAL",
            "verified",
            "sentinel_thesis_impact",
            metadata={"verified_negative": True, "headline": "material miss"},
        )
        brief.record_event(
            "AAA",
            "thesis_impact",
            "CRITICAL",
            "not verified",
            "sentinel_thesis_impact",
            metadata={"verified_negative": "false", "headline": "upgrade"},
        )
        events = brief.get_events(
            "AAA",
            hours=1,
            source="sentinel_thesis_impact",
            event_type="thesis_impact",
            verified_negative=True,
        )
        self.assertEqual([row["summary"] for row in events], ["verified"])

    def test_concurrent_event_writes_are_not_lost(self) -> None:
        from artha.pre_brief import PreBrief

        brief = PreBrief(self.path)

        def write_one(index: int) -> None:
            brief.record_event("AAA", "news_alert", "INFO", f"event-{index}", "test")

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write_one, range(40)))
        payload = json.loads((self.path / "pre_briefs.json").read_text(encoding="utf-8"))
        self.assertEqual(len(payload["events"]), 40)
        self.assertEqual({row["summary"] for row in payload["events"]}, {f"event-{i}" for i in range(40)})


class TestBuyLearningContext(unittest.TestCase):
    def test_historical_close_rejects_a_far_future_substitute(self) -> None:
        from artha.accuracy import _close_on_or_after

        rows = [
            {"date": "2026-07-02", "close": 99},
            {"date": "2026-08-11", "close": 150},
        ]
        self.assertIsNone(_close_on_or_after(rows, "2026-07-03"))
        self.assertEqual(
            _close_on_or_after(
                [{"date": "2026-07-06", "close": 101}],
                "2026-07-03",
            ),
            101.0,
        )

    def test_history_cache_does_not_make_provider_failure_permanent(self) -> None:
        from artha.accuracy import _fmp_history, _history_cache
        from artha.collector import FMPCollector

        rows = [{"date": "2026-07-03", "close": 110}]
        _history_cache.clear()
        try:
            with patch.object(FMPCollector, "history", side_effect=[None, rows]) as history:
                self.assertIsNone(_fmp_history("RETRY", period="5y"))
                self.assertEqual(_fmp_history("RETRY", period="5y"), rows)
                self.assertEqual(history.call_count, 2)
        finally:
            _history_cache.clear()

    @staticmethod
    def _record(
        ticker: str,
        timestamp: str,
        verdict: str,
        grade: str,
        excess: float,
    ) -> dict:
        return {
            "ticker": ticker,
            "timestamp": timestamp,
            "verdict": verdict,
            "grade": grade,
            "excess_return_pct": excess,
            "benchmark_return_pct": 2.0,
            "grade_basis": "excess_vs_SPY",
            "status": "GRADED",
        }

    def test_digest_is_current_era_balanced_and_one_row_per_ticker(self) -> None:
        from artha.self_review import build_council_learning_context

        rows = [
            self._record("OLD", "2026-05-01T12:00:00+00:00", "DEFER", "INCORRECT", 99),
            self._record("MISS", "2026-07-01T12:00:00+00:00", "DEFER", "INCORRECT", 20),
            self._record("MISS", "2026-07-20T12:00:00+00:00", "WATCH", "INCORRECT", 18),
            self._record("EARLY", "2026-07-10T12:00:00+00:00", "STARTER", "INCORRECT", -16),
            self._record("GOOD", "2026-07-11T12:00:00+00:00", "STARTER", "CORRECT", 8),
        ]
        fallback = self._record(
            "FALLBACK",
            "2026-07-12T12:00:00+00:00",
            "DEFER",
            "INCORRECT",
            80,
        )
        fallback["grade_basis"] = "absolute_legacy_fallback"
        fallback["excess_return_pct"] = ""
        rows.append(fallback)
        result = build_council_learning_context(
            records=rows,
            now=datetime(2026, 8, 11, tzinfo=timezone.utc),
        )
        selected = result["selected"]
        self.assertEqual({row["category"] for row in selected}, {"missed_no_buy", "bad_buy"})
        self.assertEqual(len({row["ticker"] for row in selected}), len(selected))
        self.assertNotIn("OLD", result["digest"])
        self.assertNotIn("FALLBACK", result["digest"])
        self.assertEqual(result["excluded_absolute_fallback"], 1)
        self.assertNotIn("weigh that asymmetry", result["digest"].lower())
        self.assertIn("not causal proof", result["digest"])
        self.assertFalse(result["policy"]["automatic_rule_changes"])

    def test_one_item_budget_selects_the_only_available_error_direction(self) -> None:
        from artha.self_review import build_council_learning_context

        result = build_council_learning_context(
            max_items=1,
            records=[
                self._record(
                    "EARLY",
                    "2026-07-10T12:00:00+00:00",
                    "STARTER",
                    "INCORRECT",
                    -16,
                )
            ],
            now=datetime(2026, 8, 11, tzinfo=timezone.utc),
        )
        self.assertEqual(len(result["selected"]), 1)
        self.assertEqual(result["selected"][0]["category"], "bad_buy")

    def test_missed_winner_filter_excludes_etfs_funds_and_inactive_names(self) -> None:
        from artha.self_review import _eligible_missed_winner_instrument

        self.assertFalse(_eligible_missed_winner_instrument({}, {}))
        self.assertFalse(_eligible_missed_winner_instrument({"isEtf": True}, {}))
        self.assertFalse(_eligible_missed_winner_instrument({"companyName": "Unknown"}, {}))
        self.assertFalse(_eligible_missed_winner_instrument({"isFund": "true"}, {}))
        self.assertFalse(_eligible_missed_winner_instrument({"isActivelyTrading": False}, {}))
        self.assertFalse(_eligible_missed_winner_instrument({"exchangeShortName": "OTC"}, {}))
        self.assertTrue(
            _eligible_missed_winner_instrument(
                {"isEtf": False, "isActivelyTrading": True, "exchangeShortName": "NASDAQ"},
                {},
            )
        )

    def test_lesson_ledger_concurrent_writes_and_route_reconciliation(self) -> None:
        import artha.self_review as review

        with tempfile.TemporaryDirectory() as temp:
            lesson_file = Path(temp) / "lessons.json"
            with patch.object(review, "LESSONS_FILE", lesson_file):
                def write_one(index: int) -> None:
                    label = chr(65 + (index // 26)) + chr(65 + (index % 26))
                    review.record_lesson(
                        {
                            "type": "missed_winner",
                            "description": f"unique lesson ticker-{label}",
                        }
                    )

                with ThreadPoolExecutor(max_workers=8) as pool:
                    list(pool.map(write_one, range(30)))
                ledger = review.load_lessons()
                self.assertEqual(len(ledger["lessons"]), 30)
                # Simulate one legacy row that predates route metadata.
                ledger["lessons"][0].pop("consumer", None)
                ledger["lessons"][0].pop("decision_effect", None)
                review._atomic_write_json(lesson_file, ledger)
                result = review.reconcile_lesson_routes()
                repaired = review.load_lessons()["lessons"][0]
                self.assertEqual(result["updated"], 1)
                self.assertEqual(repaired["consumer"], "scanner_coverage_diagnostic")
                self.assertEqual(repaired["decision_effect"], "observational")

    def test_shadow_promotion_lifecycle_expires_and_can_reactivate(self) -> None:
        import artha.self_review as review

        with tempfile.TemporaryDirectory() as temp:
            lesson_file = Path(temp) / "lessons.json"
            with patch.object(review, "LESSONS_FILE", lesson_file):
                lesson = review.record_lesson(
                    {
                        "type": "shadow_rule_promotion",
                        "description": "Rule alpha qualifies",
                        "context": {"rule_id": "alpha"},
                    }
                )
                expired = review.reconcile_lesson_routes(candidate_promotion_ids=set())
                self.assertEqual(expired["expired"], 1)
                self.assertEqual(review.load_lessons()["lessons"][0]["status"], "expired")
                reactivated = review.reconcile_lesson_routes(
                    candidate_promotion_ids={"alpha"}
                )
                self.assertEqual(reactivated["reactivated"], 1)
                self.assertEqual(review.load_lessons()["lessons"][0]["status"], "new")
                review.set_lesson_status(lesson["id"], "dismissed")
                untouched = review.reconcile_lesson_routes(
                    candidate_promotion_ids={"alpha"}
                )
                self.assertEqual(untouched["reactivated"], 0)
                self.assertEqual(review.load_lessons()["lessons"][0]["status"], "dismissed")

    def test_benchmark_backfill_repairs_only_fallback_rows(self) -> None:
        from artha.accuracy import AccuracyTracker

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "accuracy.json"
            rows = [
                {
                    "ticker": "AAA",
                    "timestamp": "2026-06-03T12:00:00+00:00",
                    "entry_price": "100",
                    "verdict": "DEFER",
                    "status": "GRADED",
                    "grade": "CORRECT",
                    "grade_basis": "absolute_legacy_fallback",
                    "fundamental_verdict": "HOLD",
                    "technical_verdict": "HOLD",
                    "contrarian_verdict": "HOLD",
                },
                {
                    "ticker": "DONE",
                    "timestamp": "2026-06-03T12:00:00+00:00",
                    "entry_price": "100",
                    "verdict": "DEFER",
                    "status": "GRADED",
                    "grade": "CORRECT",
                    "grade_basis": "excess_vs_SPY",
                    "excess_return_pct": "0",
                },
            ]
            path.write_text(json.dumps(rows), encoding="utf-8")
            tracker = AccuracyTracker(path)

            def history(symbol: str, period: str = "1y"):
                if symbol == "SPY":
                    return [
                        {"date": "2026-06-03", "close": 100},
                        {"date": "2026-07-03", "close": 110},
                    ]
                return [
                    {"date": "2026-06-03", "close": 100},
                    {"date": "2026-07-03", "close": 125},
                ]

            with patch("artha.accuracy._fmp_history", side_effect=history):
                result = tracker.backfill_missing_benchmark_grades(
                    now=datetime(2026, 8, 11, tzinfo=timezone.utc),
                    max_tickers=5,
                )
            repaired = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(result["regraded"], 1)
            self.assertEqual(repaired[0]["grade_basis"], "excess_vs_SPY")
            self.assertEqual(repaired[0]["excess_return_pct"], "15.00")
            self.assertEqual(repaired[0]["grade"], "INCORRECT")
            self.assertEqual(repaired[0]["benchmark_backfill_attempts"], 1)
            self.assertTrue(repaired[0]["benchmark_backfilled_at"])
            self.assertEqual(repaired[1]["grade_basis"], "excess_vs_SPY")

    def test_benchmark_backfill_rotates_past_a_provider_failure(self) -> None:
        from artha.accuracy import AccuracyTracker

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "accuracy.json"
            rows = [
                {
                    "ticker": ticker,
                    "timestamp": "2026-06-03T12:00:00+00:00",
                    "entry_price": "100",
                    "verdict": "DEFER",
                    "status": "GRADED",
                    "grade": "CORRECT",
                    "grade_basis": "absolute_legacy_fallback",
                }
                for ticker in ("AAA", "BBB")
            ]
            path.write_text(json.dumps(rows), encoding="utf-8")
            tracker = AccuracyTracker(path)

            def history(symbol: str, period: str = "1y"):
                if symbol == "AAA":
                    return []
                return [
                    {"date": "2026-06-03", "close": 100},
                    {"date": "2026-07-03", "close": 110},
                ]

            first_now = datetime(2026, 8, 11, tzinfo=timezone.utc)
            with patch("artha.accuracy._fmp_history", side_effect=history):
                first = tracker.backfill_missing_benchmark_grades(
                    now=first_now,
                    max_tickers=1,
                )
                second = tracker.backfill_missing_benchmark_grades(
                    now=first_now + timedelta(days=1),
                    max_tickers=1,
                )
            repaired = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(first["regraded"], 0)
            self.assertEqual(second["regraded"], 1)
            self.assertEqual(repaired[0]["benchmark_backfill_attempts"], 1)
            self.assertEqual(repaired[1]["grade_basis"], "excess_vs_SPY")

    def test_due_grading_uses_fixed_historical_horizon_and_retries_missing_data(self) -> None:
        from artha.accuracy import AccuracyTracker

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "accuracy.json"
            rows = [
                {
                    "ticker": ticker,
                    "timestamp": "2026-06-03T12:00:00+00:00",
                    "review_after": "2026-07-03T12:00:00+00:00",
                    "entry_price": "100",
                    "verdict": "DEFER",
                    "status": "PENDING",
                }
                for ticker in ("FIXED", "MISSING")
            ]
            path.write_text(json.dumps(rows), encoding="utf-8")
            tracker = AccuracyTracker(path)

            def history(symbol: str, period: str = "1y"):
                if symbol == "MISSING":
                    return []
                if symbol == "SPY":
                    return [
                        {"date": "2026-06-03", "close": 100},
                        {"date": "2026-07-03", "close": 105},
                        {"date": "2026-08-11", "close": 150},
                    ]
                return [
                    {"date": "2026-06-03", "close": 100},
                    {"date": "2026-07-03", "close": 110},
                    {"date": "2026-08-11", "close": 200},
                ]

            with patch("artha.accuracy._fmp_history", side_effect=history):
                result = tracker.grade_due_recommendations_from_history(
                    now=datetime(2026, 8, 11, tzinfo=timezone.utc),
                )
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(result["graded"], 1)
            self.assertEqual(result["remaining_due"], 1)
            self.assertEqual(stored[0]["price_at_review"], "110.0")
            self.assertEqual(stored[0]["benchmark_return_pct"], "5.00")
            self.assertEqual(stored[0]["excess_return_pct"], "5.00")
            self.assertEqual(stored[0]["review_window_end"], "2026-07-03T12:00:00+00:00")
            self.assertEqual(stored[1]["status"], "PENDING")

    def test_current_accuracy_patterns_exclude_nonbenchmark_fallbacks(self) -> None:
        from artha.accuracy import AccuracyTracker

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "accuracy.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "ticker": "GOOD",
                            "timestamp": "2026-06-03T12:00:00+00:00",
                            "status": "GRADED",
                            "grade": "INCORRECT",
                            "grade_basis": "excess_vs_SPY",
                            "excess_return_pct": "12",
                        },
                        {
                            "ticker": "FALLBACK",
                            "timestamp": "2026-06-03T12:00:00+00:00",
                            "status": "GRADED",
                            "grade": "CORRECT",
                            "grade_basis": "absolute_legacy_fallback",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            stats = AccuracyTracker(path).get_summary_stats(
                since="2026-06-02T00:00:00+00:00",
                benchmark_only=True,
            )
            self.assertEqual(stats["total_graded"], 1)
            self.assertEqual(stats["incorrect"], 1)
            self.assertEqual(stats["excluded_nonbenchmark"], 1)

    def test_nightly_review_repairs_measurements_before_pattern_analysis(self) -> None:
        from artha.self_review import NightlyReview

        order: list[str] = []
        review = object.__new__(NightlyReview)
        review.accuracy = Mock()
        review._review_alerts = Mock(return_value={"total": 0})
        review._grade_pending_recommendations = Mock(
            side_effect=lambda: order.append("grade") or []
        )
        review.accuracy.backfill_missing_benchmark_grades.side_effect = (
            lambda **_: order.append("backfill") or {"regraded": 0}
        )
        review._analyze_accuracy_patterns = Mock(
            side_effect=lambda: order.append("analyze") or {"patterns": []}
        )
        review._audit_missed_winners = Mock(return_value=[])
        review._check_shadow_rule_graduation = Mock(return_value={})
        review._identify_improvement = Mock(return_value=None)
        review._log_review = Mock()

        review.run_review()
        self.assertEqual(order, ["grade", "backfill", "analyze"])


class TestSentinelFeedbackRouting(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_buy_gate_loads_only_verified_negative_second_stage_events(self) -> None:
        from artha.council import load_verified_sentinel_risk_alerts
        from artha.pre_brief import PreBrief

        brief = PreBrief(self.path)
        brief.record_event(
            "AAA",
            "news_alert",
            "CRITICAL",
            "raw positive alert",
            "sentinel",
            metadata={"verified_negative": False, "headline": "earnings beat"},
        )
        brief.record_event(
            "AAA",
            "thesis_impact",
            "CRITICAL",
            "verified negative",
            "sentinel_thesis_impact",
            metadata={"verified_negative": True, "headline": "guidance withdrawn"},
        )
        alerts = load_verified_sentinel_risk_alerts("AAA", prebrief=brief, lookback_hours=1)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["headline"], "guidance withdrawn")
        self.assertTrue(alerts[0]["verified_negative"])

    def test_raw_critical_alert_cannot_block_but_verified_negative_can(self) -> None:
        from artha.council import hard_risk_gate

        config = SimpleNamespace(
            MAX_CONCURRENT_POSITIONS=20,
            MAX_INVESTED_PCT=0.9,
            MAX_SECTOR_PCT=0.3,
        )
        state = {"cash_available": 100.0, "total_value": 100.0, "positions": []}
        stock = {"profile": {}, "ratios_ttm": {}, "key_metrics_ttm": {}}
        raw = [{"ticker": "AAA", "severity": "CRITICAL", "headline": "earnings beat"}]
        verified = [{**raw[0], "headline": "guidance withdrawn", "verified_negative": True}]
        self.assertEqual(hard_risk_gate("AAA", stock, state, raw, config), (True, ""))
        passed, reason = hard_risk_gate("AAA", stock, state, verified, config)
        self.assertFalse(passed)
        self.assertIn("SENTINEL BLOCK", reason)

    def test_verified_or_uncertain_impact_creates_review_not_direct_sell(self) -> None:
        from artha.monitor import Alert
        from artha.scheduler import ArthaScheduler

        scheduler = object.__new__(ArthaScheduler)
        scheduler._record_pre_brief_event = Mock()
        scheduler.sell_engine = Mock()
        scheduler.sell_engine.flag_material_news_review.return_value = SimpleNamespace(
            signal_id="sig-1"
        )
        alert = Alert(
            ticker="AAA",
            alert_type="news_sentinel_test",
            severity="CRITICAL",
            message="headline",
            metadata={"headline": "guidance withdrawn"},
        )
        verified = {
            "verified_negative": True,
            "review_required": True,
            "assessment_failed": False,
            "affected_conditions": [{"condition": "guidance holds", "threat": "LIKELY"}],
        }
        signal = scheduler._record_verified_sentinel_impact(alert, verified)
        self.assertEqual(signal.signal_id, "sig-1")
        scheduler.sell_engine.flag_material_news_review.assert_called_once()
        self.assertEqual(
            scheduler.sell_engine.flag_material_news_review.call_args.kwargs["source"],
            "news_sentinel_verified",
        )
        scheduler.sell_engine.flag_material_news_review.reset_mock()
        uncertain = {
            "verified_negative": False,
            "review_required": True,
            "assessment_failed": True,
            "affected_conditions": [],
        }
        scheduler._record_verified_sentinel_impact(alert, uncertain)
        self.assertEqual(
            scheduler.sell_engine.flag_material_news_review.call_args.kwargs["source"],
            "news_sentinel_assessment_failure",
        )

    def test_thesis_classifier_distinguishes_possible_likely_and_failure(self) -> None:
        from artha.monitor import Alert
        from artha.scheduler import ArthaScheduler

        scheduler = object.__new__(ArthaScheduler)
        scheduler.telegram = SimpleNamespace(enabled=False)
        thesis = SimpleNamespace(
            invalidation_conditions=["Revenue guidance is maintained"],
            thesis_health_score=80,
            thesis_summary="Revenue remains durable",
            position_type="STARTER",
            entry_price=10.0,
            hard_stop_price=9.0,
        )
        alert = Alert(
            ticker="AAA",
            alert_type="news_sentinel_test",
            severity="CRITICAL",
            message="guidance update",
            metadata={"headline": "guidance update", "source": "wire"},
        )

        def run_with(raw: str) -> dict:
            tracker = Mock()
            tracker.get_active.return_value = thesis
            backend = Mock()
            backend.chat.return_value = raw
            with patch("artha.thesis_tracker.ThesisTracker", return_value=tracker), patch(
                "artha.chatgpt_backend.ChatGPTBackendClient", return_value=backend
            ):
                return asyncio.run(scheduler._assess_thesis_impact_and_alert(alert))

        possible = run_with(
            '{"affected_conditions":[{"condition":"Revenue guidance is maintained",'
            '"threat":"POSSIBLE","explanation":"unclear"}]}'
        )
        likely = run_with(
            '{"affected_conditions":[{"condition":"Revenue guidance is maintained",'
            '"threat":"LIKELY","explanation":"withdrawn"}]}'
        )
        failure = run_with("not-json")
        self.assertFalse(possible["verified_negative"])
        self.assertFalse(possible["review_required"])
        self.assertTrue(likely["verified_negative"])
        self.assertTrue(likely["review_required"])
        self.assertFalse(failure["verified_negative"])
        self.assertTrue(failure["review_required"])
        self.assertTrue(failure["assessment_failed"])

    def test_market_cycle_replays_durable_signals_created_elsewhere(self) -> None:
        from artha.scheduler import ArthaScheduler
        from artha.sell_engine import SellSignal

        scheduler = object.__new__(ArthaScheduler)
        scheduler.sell_engine = Mock()
        scheduler.sell_engine.tracker.expire_stale_pending.return_value = 0
        durable = SellSignal(ticker="AAA", signal_type="review_exit", source="news_sentinel_verified")
        scheduler.sell_engine.aggregator.get_active.return_value = [durable]
        scheduler.sell_engine.run_price_check_sell_tasks.return_value = []
        scheduler.monitor = SimpleNamespace(
            collector=SimpleNamespace(yf=SimpleNamespace(quote=lambda _ticker: None))
        )
        scheduler._run_news_sentiment_review_check = Mock(return_value=[])
        scheduler._run_sell_council_for_signals = AsyncMock(return_value={})
        portfolio = SimpleNamespace(positions=[SimpleNamespace(ticker="AAA")])
        with patch("artha.scheduler.Portfolio.load", return_value=portfolio):
            asyncio.run(scheduler._run_sell_engine_price_check())
        scheduler._run_sell_council_for_signals.assert_awaited_once()
        routed = scheduler._run_sell_council_for_signals.await_args.args[0]
        self.assertEqual([signal.signal_id for signal in routed], [durable.signal_id])

    def test_periodic_and_event_sell_reviews_share_one_lock(self) -> None:
        from artha.scheduler import ArthaScheduler
        from artha.sell_engine import SellSignal

        scheduler = object.__new__(ArthaScheduler)
        active = 0
        maximum = 0

        async def enter_once() -> None:
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.02)
            active -= 1

        async def periodic_work() -> None:
            await enter_once()

        async def signal_work(_signals, *, quotes, trigger_origin):
            await enter_once()
            return {}

        async def scenario() -> None:
            scheduler._sell_review_lock = asyncio.Lock()
            scheduler._run_periodic_review_check_unlocked = periodic_work
            scheduler._run_sell_council_for_signals_unlocked = signal_work
            await asyncio.gather(
                scheduler._run_periodic_review_check(),
                scheduler._run_sell_council_for_signals(
                    [SellSignal(ticker="AAA")],
                    trigger_origin="test",
                ),
            )

        asyncio.run(scenario())
        self.assertEqual(maximum, 1)


class TestFeedbackLoopObservability(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _execution_summary() -> dict:
        return {
            "episodes": {"closed": 2, "open": 1},
            "data_quality": {"missing_realized_pnl_events": 0},
            "post_sell": {"overdue_60d_reviews": 0},
        }

    def _build_report(self, lessons: dict) -> dict:
        import artha.feedback_loop as feedback
        from artha.journal import DecisionJournal

        journal = DecisionJournal(self.path / "artha.db")
        dossier_dir = self.path / "dossiers"
        dossier_dir.mkdir()
        (dossier_dir / "latest.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "context": {
                        "council_learning": {
                            "digest_sha256": "abc",
                            "policy": {"automatic_rule_changes": False},
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        portfolio = SimpleNamespace(positions=[SimpleNamespace(ticker="AAA")])
        sell_context = {"status": "insufficient_outcomes", "ready": False, "completed": 1, "minimum_completed": 20}
        shadow = {"total": 3, "tracking": 2, "completed": 1, "candidate_promotions": []}
        with patch.object(feedback, "build_execution_learning_summary", return_value=self._execution_summary()), patch.object(
            feedback, "build_sell_learning_context", return_value=sell_context
        ), patch.object(feedback, "summarize_shadow_rules", return_value=shadow), patch.object(
            feedback.Portfolio, "load", return_value=portfolio
        ), patch.object(feedback.PreBrief, "get_events", return_value=[]), patch.object(
            feedback.Config, "SENTINEL_ENABLED", True
        ), patch.object(feedback.Config, "SELL_SENTINEL_COUNCIL_ESCALATION_ENABLED", True):
            return feedback.build_feedback_loop_report(
                journal,
                portfolio_path=self.path / "portfolio.json",
                dossier_dir=dossier_dir,
                prebrief_data_dir=self.path,
                records=[],
                lessons=lessons,
            )

    def test_report_passes_when_every_feedback_type_has_a_consumer(self) -> None:
        report = self._build_report(
            {"lessons": [{"type": "missed_winner", "status": "new"}]}
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(
            report["lesson_lifecycle"]["consumer_counts"],
            {"scanner_coverage_diagnostic": 1},
        )
        self.assertFalse(report["shadow_feedback"]["automatic_promotion"])

    def test_report_fails_for_an_unmapped_feedback_type(self) -> None:
        report = self._build_report(
            {"lessons": [{"type": "mystery_signal", "status": "new"}]}
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("mystery_signal", report["failures"][0])

    def test_report_warns_when_manual_review_lessons_are_unresolved(self) -> None:
        report = self._build_report(
            {"lessons": [{"id": "review-1", "type": "shadow_rule_promotion", "status": "new"}]}
        )
        self.assertEqual(report["status"], "WARN")
        self.assertEqual(
            report["lesson_lifecycle"]["pending_manual_reviews"][0]["id"],
            "review-1",
        )

    def test_supervisor_surfaces_feedback_failure(self) -> None:
        import artha.supervisor as supervisor

        fake = {
            "status": "FAIL",
            "failures": ["Lesson types have no declared consumer: mystery_signal"],
            "warnings": [],
        }
        with patch.object(supervisor, "build_feedback_loop_report", return_value=fake):
            result = supervisor._check_feedback_loop(Mock())
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("mystery_signal", result["message"])

    def test_decision_dossier_preserves_exact_feedback_provenance(self) -> None:
        import artha.dossier as dossier

        learning = {
            "digest": "bounded lesson digest",
            "digest_sha256": "digest-hash",
            "current_era_start": "2026-06-02",
            "policy": {"automatic_rule_changes": False},
        }
        sentinel = {
            "verified_negative_count": 1,
            "raw_keyword_alerts_used_as_hard_blocks": False,
        }
        stock = {
            "ticker": "AAA",
            "quote": {"price": 10.0},
            "council_learning_context": learning,
            "sentinel_risk_context": sentinel,
        }
        decision = SimpleNamespace(ticker="AAA", final_verdict="DEFER", agentic_trace={})
        journal = Mock()
        with patch.object(dossier, "DOSSIER_DIR", self.path / "dossiers"), patch(
            "artha.journal.DecisionJournal", return_value=journal
        ):
            path = Path(dossier.write_decision_dossier(decision, stock))
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["context"]["council_learning"]["digest_sha256"], "digest-hash")
        self.assertEqual(payload["context"]["sentinel_risk"]["verified_negative_count"], 1)
        self.assertEqual(
            payload["stock_packet"]["council_learning_context"]["digest"],
            "bounded lesson digest",
        )


if __name__ == "__main__":
    unittest.main()
