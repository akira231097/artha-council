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
LESSONS_FILE = LEARNINGS_DIR / "lessons.json"
REVIEW_LOG = Path(__file__).resolve().parent.parent / "data" / "review_log.json"

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Missed-winner audit tunables (Wave 2 consolidates into Config).
MISSED_WINNER_MIN_MARKET_CAP = _env_float("ARTHA_MISSED_WINNER_MIN_MARKET_CAP", 1_000_000_000)
MISSED_WINNER_TOP_N = int(_env_float("ARTHA_MISSED_WINNER_TOP_N", 10))
MISSED_WINNER_AUDIT_ENABLED = os.getenv("ARTHA_MISSED_WINNER_AUDIT_ENABLED", "1").strip() not in {"0", "false", "no"}
LESSON_STATUSES = ("new", "applied", "dismissed")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _lesson_dedupe_key(lesson_type: str, description: str) -> str:
    """Stable dedupe id: type + description with volatile numbers stripped.

    'Fundamental accuracy is low (37.2%)' and '(39.3%)' collapse to one lesson
    instead of being written every night.
    """
    import hashlib
    import re
    normalized = re.sub(r"[\d.]+", "#", str(description or "").lower()).strip()
    raw = f"{str(lesson_type or '').lower()}|{normalized}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_lessons() -> dict:
    """Load the deduped lessons ledger (data/learnings/lessons.json)."""
    if not LESSONS_FILE.exists():
        return {"lessons": [], "updated_at": ""}
    try:
        with open(LESSONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("lessons"), list):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {"lessons": [], "updated_at": ""}


def record_lesson(lesson: dict) -> dict:
    """Upsert a lesson into the deduped ledger with lifecycle status.

    New lessons start at status='new'. Repeats bump occurrences/last_seen and
    keep the existing status, so an 'applied' or 'dismissed' lesson never
    reappears as fresh noise.
    """
    ledger = load_lessons()
    lesson_id = lesson.get("id") or _lesson_dedupe_key(
        lesson.get("type", ""), lesson.get("description", "")
    )
    now_iso = _utcnow().isoformat()
    for existing in ledger["lessons"]:
        if existing.get("id") == lesson_id:
            existing["last_seen"] = now_iso
            existing["occurrences"] = int(existing.get("occurrences") or 1) + 1
            existing["description"] = lesson.get("description", existing.get("description", ""))
            if lesson.get("context"):
                existing["context"] = lesson["context"]
            ledger["updated_at"] = now_iso
            _atomic_write_json(LESSONS_FILE, ledger)
            return existing

    entry = {
        "id": lesson_id,
        "type": lesson.get("type", "unknown"),
        "description": lesson.get("description", ""),
        "action": lesson.get("action", ""),
        "priority": lesson.get("priority", "LOW"),
        "status": "new",
        "first_seen": now_iso,
        "last_seen": now_iso,
        "occurrences": 1,
        "context": lesson.get("context") or {},
    }
    ledger["lessons"].append(entry)
    ledger["updated_at"] = now_iso
    _atomic_write_json(LESSONS_FILE, ledger)
    return entry


def set_lesson_status(lesson_id: str, status: str) -> bool:
    """Move a lesson through its lifecycle: new -> applied | dismissed."""
    if status not in LESSON_STATUSES:
        raise ValueError(f"status must be one of {LESSON_STATUSES}")
    ledger = load_lessons()
    for existing in ledger["lessons"]:
        if existing.get("id") == lesson_id:
            existing["status"] = status
            existing["status_changed_at"] = _utcnow().isoformat()
            ledger["updated_at"] = _utcnow().isoformat()
            _atomic_write_json(LESSONS_FILE, ledger)
            return True
    return False


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

        # --- Step 3b: Missed-winner audit (top movers Artha did not capture) ---
        try:
            findings["missed_winners"] = self._audit_missed_winners(today)
        except Exception as e:
            logger.warning(f"[review] Missed-winner audit failed: {e}")
            findings["missed_winners"] = []

        # --- Step 3c: Shadow-rule graduation check ---
        try:
            findings["shadow_rule_graduation"] = self._check_shadow_rule_graduation()
        except Exception as e:
            logger.warning(f"[review] Shadow-rule graduation check failed: {e}")
            findings["shadow_rule_graduation"] = {}

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

        today_str = today.isoformat()

        # monitor.py persists history["alerts"] as a list of dicts with
        # sent_at/alert_type/severity (see AlertHistory.claim_new_alerts).
        # The old history["sent_alerts"] dict format is kept as a fallback
        # for pre-migration files.
        alerts = history.get("alerts")
        if isinstance(alerts, list):
            for entry in alerts:
                if not isinstance(entry, dict):
                    continue
                sent_at = str(entry.get("sent_at", ""))
                if not sent_at.startswith(today_str):
                    continue
                summary["total"] += 1
                severity = str(entry.get("severity", "")).lower()
                if severity in summary:
                    summary[severity] += 1
                alert_type = str(entry.get("alert_type", "unknown"))
                summary["types"][alert_type] = summary["types"].get(alert_type, 0) + 1
            return summary

        sent = history.get("sent_alerts", {})
        if isinstance(sent, dict):
            for key, ts in sent.items():
                if not str(ts).startswith(today_str):
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
                # FMP first (paid, reliable); yfinance as fallback. Buy-type
                # verdicts grade through this exact same path as everything
                # else — a dead quote no longer leaves them PENDING forever.
                current_price = 0.0
                try:
                    from .collector import FMPCollector
                    quote = FMPCollector().quote(ticker) or {}
                    current_price = float(quote.get("price") or 0)
                except Exception as fmp_e:
                    logger.debug(f"[review] FMP quote failed for {ticker}: {fmp_e}")
                if not current_price:
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

    def _audit_missed_winners(self, today: date) -> list[dict]:
        """Audit the day's top movers ($1B+ cap) against Artha's dispositions.

        For each of the top-N gainers, answer: was it seen by the scanner?
        routed? council-ed? deferred? Emits structured learnings of the form
        {ticker, date, move_pct, artha_disposition, root_cause_guess} into the
        deduped lessons ledger and returns them.
        """
        if not MISSED_WINNER_AUDIT_ENABLED:
            return []
        try:
            from .collector import FMPCollector
            collector = FMPCollector()
        except Exception as e:
            logger.warning(f"[review] FMP collector unavailable for missed-winner audit: {e}")
            return []

        gainers = collector.market_gainers(limit=50) or []
        winners: list[dict] = []
        for row in gainers:
            if len(winners) >= MISSED_WINNER_TOP_N:
                break
            symbol = str(row.get("symbol") or "").upper().strip()
            if not symbol:
                continue
            quote = collector.quote(symbol) or {}
            market_cap = float(quote.get("marketCap") or 0)
            if market_cap < MISSED_WINNER_MIN_MARKET_CAP:
                continue
            price = float(quote.get("price") or row.get("price") or 0)
            move_pct = quote.get("changePercentage")
            if move_pct is None:
                move_pct = row.get("changesPercentage") or row.get("changePercentage") or 0
            winners.append({
                "ticker": symbol,
                "move_pct": round(float(move_pct or 0), 2),
                "market_cap": market_cap,
                "price": price,
            })

        if not winners:
            return []

        # Latest accuracy-tracker verdict per ticker (last 45 days).
        lf = self.accuracy._lock(exclusive=False)
        try:
            accuracy_records = self.accuracy._load()
        finally:
            self.accuracy._unlock(lf)
        cutoff = (_utcnow() - timedelta(days=45)).isoformat()
        latest_verdict: dict[str, str] = {}
        for rec in accuracy_records:
            ticker = str(rec.get("ticker") or "").upper()
            if str(rec.get("timestamp") or "") >= cutoff and ticker:
                latest_verdict[ticker] = str(rec.get("verdict") or "").upper()

        from .journal import DecisionJournal
        journal = DecisionJournal()

        buy_like = {"BUY", "STRONG BUY", "STARTER", "TACTICAL_BUY", "TACTICAL BUY", "ACCUMULATE", "ADD"}
        try:
            from .scanner import CATALYST_EP_MIN_PRICE
        except Exception:
            CATALYST_EP_MIN_PRICE = 5.0

        learnings: list[dict] = []
        with journal._connect() as conn:
            for winner in winners:
                ticker = winner["ticker"]
                routing = conn.execute(
                    """
                    SELECT lane, bucket, reason_code, reason, created_at
                    FROM scan_routing_decisions
                    WHERE ticker = ?
                    ORDER BY datetime(created_at) DESC
                    LIMIT 1
                    """,
                    (ticker,),
                ).fetchone()

                verdict = latest_verdict.get(ticker, "")
                if verdict:
                    disposition = f"council_{verdict.lower().replace(' ', '_')}"
                    if verdict in buy_like:
                        root_cause = "Captured: Artha recommended a buy-side entry."
                    else:
                        root_cause = (
                            f"Council saw it and declined ({verdict}); review the "
                            "no-buy rationale against the realized move."
                        )
                elif routing is not None:
                    lane = str(routing["lane"] or "")
                    reason_code = str(routing["reason_code"] or "")
                    if lane == "execution_ready":
                        disposition = "routed_execution_ready_no_council"
                        root_cause = (
                            "Router marked it execution_ready but no council verdict was "
                            "recorded — check scan candidate caps / follow-through."
                        )
                    elif lane == "hard_reject":
                        disposition = "hard_rejected"
                        root_cause = f"Hard-rejected at routing ({reason_code})."
                    else:
                        disposition = f"routed_{lane or 'unknown'}_no_council"
                        root_cause = (
                            f"Routed to {lane or 'unknown'} ({reason_code}); routing "
                            "score threshold kept it from council review."
                        )
                else:
                    if 0 < float(winner.get("price") or 0) < CATALYST_EP_MIN_PRICE:
                        disposition = "policy_excluded_low_price"
                        root_cause = (
                            "Policy-excluded by the catalyst price floor "
                            f"(${float(winner.get('price') or 0):.2f} < ${CATALYST_EP_MIN_PRICE:.2f}); "
                            "not a scanner coverage miss."
                        )
                    else:
                        disposition = "never_seen"
                        root_cause = (
                            "Never surfaced by the scanner — check universe/screener "
                            "filters and candidate ranking coverage."
                        )

                captured = verdict in buy_like
                actionable_miss = not captured and not str(disposition).startswith("policy_excluded_")
                learning = {
                    "ticker": ticker,
                    "date": today.isoformat(),
                    "move_pct": winner["move_pct"],
                    "price": winner.get("price"),
                    "market_cap": winner.get("market_cap"),
                    "artha_disposition": disposition,
                    "root_cause_guess": root_cause,
                    "captured": captured,
                    "actionable_miss": actionable_miss,
                }
                learnings.append(learning)
                try:
                    record_lesson({
                        "type": (
                            "captured_winner"
                            if learning["captured"]
                            else "missed_winner" if learning["actionable_miss"]
                            else "policy_excluded_winner"
                        ),
                        "description": (
                            f"{ticker} moved {winner['move_pct']:+.1f}% (>$1B cap); "
                            f"Artha disposition: {disposition}"
                        ),
                        "action": root_cause,
                        "priority": "HIGH" if learning["actionable_miss"] else "LOW",
                        "context": learning,
                    })
                except Exception as e:
                    logger.warning(f"[review] Failed to record missed-winner lesson for {ticker}: {e}")

        logger.info(
            "[review] Missed-winner audit: %d qualifying mover(s), %d not captured",
            len(learnings), sum(1 for l in learnings if not l["captured"]),
        )
        return learnings

    def _check_shadow_rule_graduation(self) -> dict:
        """Surface shadow rules that qualify for promotion in the nightly review."""
        from .shadow_rules import summarize_shadow_rules
        summary = summarize_shadow_rules()
        promotions = summary.get("candidate_promotions") or []
        for promo in promotions:
            try:
                record_lesson({
                    "type": "shadow_rule_promotion",
                    "description": (
                        f"Shadow rule {promo.get('rule_id')} is a promotion candidate: "
                        f"{promo.get('evidence', '')}"
                    ),
                    "action": "Review evidence and manually promote the rule (no auto config change).",
                    "priority": "HIGH",
                    "context": promo,
                })
            except Exception as e:
                logger.warning(f"[review] Failed to record graduation lesson: {e}")
        return {
            "candidate_promotions": promotions,
            "rules_tracked": len(summary.get("rules") or {}),
            "sell_rules_tracked": len((summary.get("sell_rules") or {}).get("rules") or {}),
        }

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

        # Priority 3: scanner/routing missed a big mover
        if not improvement:
            missed = [
                m for m in (findings.get("missed_winners") or [])
                if not m.get("captured") and m.get("actionable_miss", True)
            ]
            if missed:
                worst = max(missed, key=lambda m: float(m.get("move_pct") or 0))
                improvement = {
                    "type": "missed_winner_audit",
                    "description": (
                        f"Missed {len(missed)} $1B+ top mover(s) today; worst: "
                        f"{worst.get('ticker')} {float(worst.get('move_pct') or 0):+.1f}% "
                        f"({worst.get('artha_disposition')})"
                    ),
                    "action": str(worst.get("root_cause_guess") or ""),
                    "priority": "HIGH",
                }

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

        # Also upsert into the deduped lifecycle ledger so the same lesson is
        # not re-emitted as new every night.
        try:
            record_lesson(learning)
        except Exception as e:
            logger.warning(f"[review] Failed to upsert lesson into ledger: {e}")


def get_council_lessons(max_items: int = 5, max_chars: int = 1200) -> str:
    """Deterministic digest of graded postmortems for council prompt injection.

    Pulls INCORRECT / PARTIALLY_CORRECT graded records from the accuracy
    tracker, weights them by recency and excess-return magnitude, and renders
    a compact plain-text digest (no LLM calls, file reads only). Wave 2
    injects this into council prompts.
    """
    tracker = AccuracyTracker()
    lf = tracker._lock(exclusive=False)
    try:
        records = tracker._load()
    finally:
        tracker._unlock(lf)

    from .accuracy import EXPOSURE_VERDICTS, _normalize_verdict, _num_or_none

    now = _utcnow()
    wrong_nobuy: list[dict] = []
    wrong_buy: list[dict] = []
    scored: list[tuple[float, str, str]] = []
    for rec in records:
        if rec.get("status") != "GRADED":
            continue
        grade = rec.get("grade")
        if grade not in ("INCORRECT", "PARTIALLY_CORRECT"):
            continue
        excess = _num_or_none(rec.get("excess_return_pct"))
        if excess is None:
            excess = _num_or_none(rec.get("price_change_pct")) or 0.0
        ts = str(rec.get("timestamp") or "")
        try:
            rec_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if rec_dt.tzinfo is None:
                rec_dt = rec_dt.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (now - rec_dt).total_seconds() / 86400.0)
        except ValueError:
            age_days = 365.0
        weight = abs(excess) * (0.5 ** (age_days / 60.0))

        ticker = str(rec.get("ticker") or "?").upper()
        verdict = str(rec.get("verdict") or "?").upper()
        is_exposure = _normalize_verdict(verdict) in EXPOSURE_VERDICTS
        date_str = ts[:10]
        if is_exposure:
            wrong_buy.append(rec)
            line = (
                f"- {verdict} {ticker} ({date_str}): lagged the benchmark by "
                f"{excess:+.1f}pp over the review window — graded {grade}. "
                f"Buy thesis failed; check entry timing and expectation risk."
            )
        else:
            wrong_nobuy.append(rec)
            line = (
                f"- {verdict} {ticker} ({date_str}): it beat the benchmark by "
                f"{excess:+.1f}pp after the no-buy call — graded {grade}. "
                f"Excessive caution on a name with working momentum/catalysts."
            )
        scored.append((weight, f"{ticker}|{ts}", line))

    lines: list[str] = []
    if scored:
        nobuy_excess = [
            _num_or_none(r.get("excess_return_pct")) for r in wrong_nobuy
        ]
        nobuy_excess = [v for v in nobuy_excess if v is not None]
        header = (
            f"GRADED POSTMORTEMS ({len(wrong_nobuy)} wrong/partial no-buy calls"
        )
        if nobuy_excess:
            header += f", avg missed excess {sum(nobuy_excess) / len(nobuy_excess):+.1f}pp"
        header += f"; {len(wrong_buy)} wrong/partial buys):"
        lines.append(header)
        scored.sort(key=lambda item: (-item[0], item[1]))
        per_ticker: dict[str, int] = {}
        picked = 0
        for _, key, line in scored:
            if picked >= max_items:
                break
            ticker = key.split("|", 1)[0]
            if per_ticker.get(ticker, 0) >= 2:
                continue  # Keep the digest diverse — max 2 lines per ticker.
            per_ticker[ticker] = per_ticker.get(ticker, 0) + 1
            lines.append(line)
            picked += 1
        if len(wrong_nobuy) > len(wrong_buy) and len(wrong_nobuy) >= 3:
            lines.append(
                "Pattern: most graded mistakes are missed winners from DEFER/WATCH/AVOID, "
                "not bad buys. Anchoring on pullback entries that never fill has been the "
                "repeated INCORRECT mode — weigh that asymmetry before deferring."
            )
    else:
        lines.append("GRADED POSTMORTEMS: no graded misses on record yet.")

    # Append the most persistent open lessons from the ledger.
    ledger = load_lessons()
    open_lessons = [
        l for l in ledger.get("lessons", [])
        if l.get("status") == "new" and l.get("type") in ("missed_winner", "shadow_rule_promotion")
    ]
    open_lessons.sort(
        key=lambda l: (-int(l.get("occurrences") or 0), str(l.get("id") or ""))
    )
    for lesson in open_lessons[:2]:
        lines.append(f"- Open lesson (seen {lesson.get('occurrences')}x): {lesson.get('description')}")

    digest = "\n".join(lines)
    if len(digest) > max_chars:
        clipped: list[str] = []
        used = 0
        for line in lines:
            if used + len(line) + 1 > max_chars:
                break
            clipped.append(line)
            used += len(line) + 1
        digest = "\n".join(clipped)
    return digest
