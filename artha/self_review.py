"""Nightly self-review engine — Artha's 1% daily improvement system.

Inspired by Felix/Nat Eliason: every night, review the day's activity,
identify ONE improvement, and apply it. Compounds over time.

This module:
1. Reviews all alerts that fired (or should have)
2. Reviews any council reports generated
3. Checks accuracy tracker for newly gradeable recommendations
4. Identifies one actionable improvement
5. Logs the improvement to data/learnings/
6. Optionally adjusts thresholds/config for next run

Runs as a scheduler task after market close or as a standalone script.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import yfinance as yf

from .accuracy import AccuracyTracker
from .config import Config
from .monitor import ALERT_HISTORY_FILE

logger = logging.getLogger(__name__)

LEARNINGS_DIR = Path(__file__).resolve().parent.parent / "data" / "learnings"
REVIEW_LOG = Path(__file__).resolve().parent.parent / "data" / "review_log.json"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class NightlyReview:
    """Reviews the day's trading activity and identifies improvements."""

    def __init__(self):
        self.accuracy = AccuracyTracker()
        self.learnings_dir = LEARNINGS_DIR
        self.learnings_dir.mkdir(parents=True, exist_ok=True)
        self.review_log_path = REVIEW_LOG

    def run_review(self) -> dict:
        """Execute the full nightly review cycle.

        Returns a summary dict of findings and actions taken.
        """
        today = date.today()
        logger.info(f"[review] Starting nightly review for {today}")

        findings: dict[str, Any] = {
            "date": today.isoformat(),
            "timestamp": _utcnow().isoformat(),
            "alerts_reviewed": 0,
            "accuracy_grades": 0,
            "improvement": None,
        }

        # --- Step 1: Review today's alerts ---
        alert_summary = self._review_alerts(today)
        findings["alerts_reviewed"] = alert_summary.get("total", 0)
        findings["alert_details"] = alert_summary

        # --- Step 2: Grade any 30-day-old recommendations ---
        grade_results = self._grade_pending_recommendations()
        findings["accuracy_grades"] = len(grade_results)
        findings["grades"] = grade_results

        # --- Step 3: Check accuracy stats for patterns ---
        accuracy_insights = self._analyze_accuracy_patterns()
        findings["accuracy_insights"] = accuracy_insights

        # --- Step 4: Identify ONE improvement ---
        improvement = self._identify_improvement(findings)
        findings["improvement"] = improvement

        # --- Step 5: Log the review ---
        self._log_review(findings)

        # --- Step 6: Save learning if improvement found ---
        if improvement:
            self._save_learning(today, improvement)

        logger.info(
            f"[review] Nightly review complete. "
            f"Alerts: {findings['alerts_reviewed']}, "
            f"Grades: {findings['accuracy_grades']}, "
            f"Improvement: {improvement.get('type', 'none') if improvement else 'none'}"
        )
        return findings

    def _review_alerts(self, today: date) -> dict:
        """Review alerts that fired today."""
        summary = {"total": 0, "critical": 0, "warning": 0, "info": 0, "types": {}}

        if not ALERT_HISTORY_FILE.exists():
            return summary

        try:
            with open(ALERT_HISTORY_FILE, "r") as f:
                history = json.load(f)
        except (json.JSONDecodeError, OSError):
            return summary

        sent = history.get("sent_alerts", {})
        today_str = today.isoformat()

        for key, ts in sent.items():
            if not ts.startswith(today_str):
                continue
            summary["total"] += 1
            # Parse alert type from key (format: "TICKER:alert_type")
            parts = key.split(":")
            if len(parts) >= 2:
                alert_type = parts[1]
                summary["types"][alert_type] = summary["types"].get(alert_type, 0) + 1

        return summary

    def _grade_pending_recommendations(self) -> list[dict]:
        """Grade any recommendations that have passed their 30-day review date."""
        pending = self.accuracy.get_pending_reviews()
        graded = []

        for rec in pending:
            ticker = rec.get("ticker", "")
            timestamp = rec.get("timestamp", "")
            if not ticker:
                continue

            try:
                stock = yf.Ticker(ticker)
                info = stock.info or {}
                current_price = info.get("regularMarketPrice") or info.get("currentPrice", 0)
                if not current_price:
                    hist = stock.history(period="1d")
                    if not hist.empty:
                        current_price = float(hist["Close"].iloc[-1])

                if current_price:
                    result = self.accuracy.grade_recommendation(
                        ticker, timestamp, current_price
                    )
                    if result:
                        graded.append(result)
            except Exception as e:
                logger.warning(f"[review] Failed to grade {ticker}: {e}")

        return graded

    def _analyze_accuracy_patterns(self) -> dict:
        """Look for patterns in graded recommendations."""
        stats = self.accuracy.get_summary_stats()
        current_stats = self.accuracy.get_summary_stats(since=Config.ACCURACY_CURRENT_ERA_START)
        min_samples = max(1, int(getattr(Config, "ACCURACY_MIN_PATTERN_SAMPLES", 3)))
        insights: dict[str, Any] = {
            "patterns": [],
            "current_patterns": [],
            "legacy_patterns": [],
            "stats": {
                "all_time": stats,
                "current": current_stats,
                "current_era_start": Config.ACCURACY_CURRENT_ERA_START,
                "current_council_version": Config.ACCURACY_CURRENT_COUNCIL_VERSION,
            },
        }

        if current_stats["total_graded"] < min_samples:
            insights["current_patterns"].append(
                f"Current council era has {current_stats['total_graded']} graded recommendation(s); "
                f"waiting for {min_samples} before prompt tuning."
            )
            self._add_legacy_accuracy_context(insights, stats)
            insights["patterns"] = insights["current_patterns"] + insights["legacy_patterns"]
            return insights

        # Active prompt/model tuning should only use the current council era.
        for analyst, data in current_stats.get("analyst_accuracy", {}).items():
            if data["total"] >= 3 and data["accuracy"] < 40:
                insights["current_patterns"].append(
                    f"⚠️ Current {analyst} accuracy is low ({data['accuracy']}%) — "
                    f"consider prompt adjustment"
                )
            elif data["total"] >= 3 and data["accuracy"] > 80:
                insights["current_patterns"].append(
                    f"✅ Current {analyst} performing well ({data['accuracy']}%)"
                )

        if current_stats["overall_accuracy"] is not None and current_stats["overall_accuracy"] < 50:
            insights["current_patterns"].append(
                "⚠️ Current council-era accuracy below 50% — review analyst prompts and data quality"
            )

        if not insights["current_patterns"]:
            insights["current_patterns"].append("Current council-era accuracy is within review thresholds")

        self._add_legacy_accuracy_context(insights, stats)
        insights["patterns"] = insights["current_patterns"] + insights["legacy_patterns"]
        return insights

    def _add_legacy_accuracy_context(self, insights: dict, stats: dict) -> None:
        """Preserve old-model lessons without making them active prompt-tune alerts."""
        if stats.get("total_graded", 0) < 3:
            return

        for analyst, data in stats.get("analyst_accuracy", {}).items():
            if data.get("total", 0) >= 3 and data.get("accuracy", 100) < 40:
                insights["legacy_patterns"].append(
                    f"Legacy/all-time {analyst} score is low ({data['accuracy']}%) — "
                    "tracked as historical context, not a current prompt-tuning trigger."
                )

    def _identify_improvement(self, findings: dict) -> Optional[dict]:
        """Identify the single most impactful improvement based on today's data.

        Priority order:
        1. Fix accuracy issues (if any analyst consistently wrong)
        2. Fix alert coverage gaps (if something should have alerted but didn't)
        3. Tune scoring thresholds (if scanner missed a big mover)
        4. General observation for future reference
        """
        improvement = None

        # Check accuracy patterns first
        insights = findings.get("accuracy_insights", {})
        patterns = insights.get("current_patterns") or insights.get("patterns", [])

        for pattern in patterns:
            if "accuracy is low" in pattern:
                improvement = {
                    "type": "analyst_prompt_tune",
                    "description": pattern,
                    "action": "Review and adjust analyst prompt for underperforming model",
                    "priority": "HIGH",
                }
                break

        # If no accuracy issues, check alert patterns
        if not improvement:
            alert_details = findings.get("alert_details", {})
            alert_total = alert_details.get("total", 0)
            if alert_total > 10:
                improvement = {
                    "type": "alert_threshold_tune",
                    "description": f"High alert volume ({alert_total} today) — may need threshold adjustment",
                    "action": "Consider widening alert thresholds to reduce noise",
                    "priority": "MEDIUM",
                }
            elif alert_total == 0 and findings.get("accuracy_grades", 0) == 0:
                improvement = {
                    "type": "observation",
                    "description": "Quiet day — no alerts, no grades due",
                    "action": "No action needed. System running as expected.",
                    "priority": "LOW",
                }

        # Check for grading results
        grades = findings.get("grades", [])
        incorrect = [g for g in grades if g.get("grade") == "INCORRECT"]
        if incorrect:
            tickers = ", ".join(g.get("ticker", "?") for g in incorrect)
            improvement = {
                "type": "verdict_postmortem",
                "description": f"Incorrect verdicts on: {tickers}",
                "action": "Analyze why the council got these wrong — check data quality, analyst reasoning",
                "priority": "HIGH",
            }

        return improvement

    def _log_review(self, findings: dict) -> None:
        """Append review to the persistent review log."""
        self.review_log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if self.review_log_path.exists():
                with open(self.review_log_path, "r") as f:
                    log = json.load(f)
            else:
                log = []
        except (json.JSONDecodeError, OSError):
            log = []

        log.append(findings)

        # Keep last 90 days of reviews
        if len(log) > 90:
            log = log[-90:]

        fd, tmp = tempfile.mkstemp(
            dir=str(self.review_log_path.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(log, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, str(self.review_log_path))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _save_learning(self, today: date, improvement: dict) -> None:
        """Save a learning to the learnings directory."""
        filename = f"{today.isoformat()}.json"
        filepath = self.learnings_dir / filename

        learning = {
            "date": today.isoformat(),
            "type": improvement.get("type", "unknown"),
            "description": improvement.get("description", ""),
            "action": improvement.get("action", ""),
            "priority": improvement.get("priority", "LOW"),
            "timestamp": _utcnow().isoformat(),
        }

        # If file exists (multiple reviews in a day), append
        existing = []
        if filepath.exists():
            try:
                with open(filepath, "r") as f:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = [existing]
            except (json.JSONDecodeError, OSError):
                existing = []

        existing.append(learning)

        fd, tmp = tempfile.mkstemp(
            dir=str(self.learnings_dir), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, str(filepath))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

        logger.info(f"[review] Saved learning to {filepath}")
