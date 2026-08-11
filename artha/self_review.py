"""Nightly evidence review for Artha's guarded improvement process.

Every night, review the day's activity and record one candidate lesson. The
module does not alter live scores, thresholds, position sizes, or broker rules;
evidence must pass its declared sample gate and an explicit human-reviewed
promotion before it can change strategy.

This module:
1. Reviews all alerts that fired (or should have)
2. Reviews any council reports generated
3. Checks accuracy tracker for newly gradeable recommendations
4. Identifies one actionable improvement
5. Logs the improvement to data/learnings/
6. Routes the lesson to an explicit advisory, shadow, or manual-review consumer

Runs as a scheduler task after market close or as a standalone script.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import hashlib
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from .accuracy import AccuracyTracker
from .config import Config
from .monitor import ALERT_HISTORY_FILE
from .paths import DATA_DIR

logger = logging.getLogger(__name__)

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

LEARNINGS_DIR = DATA_DIR / "learnings"
LESSONS_FILE = LEARNINGS_DIR / "lessons.json"
REVIEW_LOG = DATA_DIR / "review_log.json"

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
LESSON_STATUSES = ("new", "applied", "dismissed", "expired")

# Every lesson type has an explicit downstream owner. A lesson may remain
# observational or require manual promotion, but it must never disappear into
# an unlabelled JSON backlog or silently alter live trading rules.
LESSON_CONSUMER_ROUTES: dict[str, dict[str, str]] = {
    "verdict_postmortem": {"consumer": "nightly_postmortem_audit", "effect": "observational"},
    "analyst_prompt_tune": {"consumer": "manual_prompt_review", "effect": "manual_review"},
    "missed_winner": {"consumer": "scanner_coverage_diagnostic", "effect": "observational"},
    "missed_winner_audit": {"consumer": "scanner_coverage_diagnostic", "effect": "observational"},
    "captured_winner": {"consumer": "scanner_coverage_diagnostic", "effect": "observational"},
    "policy_excluded_winner": {"consumer": "policy_exclusion_audit", "effect": "observational"},
    "shadow_rule_promotion": {"consumer": "manual_shadow_promotion_review", "effect": "manual_review"},
    "alert_threshold_tune": {"consumer": "manual_operations_review", "effect": "manual_review"},
    "observation": {"consumer": "nightly_operations_log", "effect": "observational"},
}


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _eligible_missed_winner_instrument(profile: dict, quote: dict) -> bool:
    """Keep missed-winner lessons inside Artha's permitted stock universe."""
    # This is a learning audit, not a trading opportunity. If identity cannot
    # be verified, skip it rather than teaching from an ETF/fund mislabeled as
    # a stock by a top-movers endpoint.
    if not isinstance(profile, dict) or not profile:
        return False
    if any(
        _truthy_flag(value)
        for value in (
            profile.get("isEtf"),
            profile.get("isETF"),
            profile.get("isFund"),
            quote.get("isEtf"),
            quote.get("isETF"),
            quote.get("isFund"),
        )
    ):
        return False
    if profile.get("isActivelyTrading") is not None and not _truthy_flag(
        profile.get("isActivelyTrading")
    ):
        return False
    exchange = str(profile.get("exchangeShortName") or "").upper().strip()
    return exchange in {"NASDAQ", "NYSE", "AMEX"}


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


@contextmanager
def _lesson_lock(*, exclusive: bool) -> Iterator[None]:
    lock_path = LESSONS_FILE.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


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


def get_lesson_consumer_route(lesson_type: str) -> dict[str, str]:
    """Return the declared consumer/effect for one lesson type."""
    return dict(
        LESSON_CONSUMER_ROUTES.get(
            str(lesson_type or "").strip(),
            {"consumer": "unmapped", "effect": "none"},
        )
    )


def _load_lessons_unlocked() -> dict:
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


def load_lessons() -> dict:
    """Load the deduped lessons ledger under a shared file lock."""
    with _lesson_lock(exclusive=False):
        return _load_lessons_unlocked()


def record_lesson(lesson: dict) -> dict:
    """Upsert a lesson into the deduped ledger with lifecycle status.

    New lessons start at status='new'. Repeats bump occurrences/last_seen and
    keep the existing status, so an 'applied' or 'dismissed' lesson never
    reappears as fresh noise.
    """
    with _lesson_lock(exclusive=True):
        ledger = _load_lessons_unlocked()
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
                if (
                    str(existing.get("status") or "") == "expired"
                    and str(existing.get("type") or "") == "shadow_rule_promotion"
                ):
                    existing["status"] = "new"
                    existing["status_changed_at"] = now_iso
                    existing["status_reason"] = "Shadow rule qualified again."
                route = get_lesson_consumer_route(existing.get("type", ""))
                existing["consumer"] = route["consumer"]
                existing["decision_effect"] = route["effect"]
                ledger["updated_at"] = now_iso
                _atomic_write_json(LESSONS_FILE, ledger)
                return existing

        route = get_lesson_consumer_route(lesson.get("type", "unknown"))
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
            "consumer": route["consumer"],
            "decision_effect": route["effect"],
        }
        ledger["lessons"].append(entry)
        ledger["updated_at"] = now_iso
        _atomic_write_json(LESSONS_FILE, ledger)
        return entry


def set_lesson_status(lesson_id: str, status: str) -> bool:
    """Move a lesson through its lifecycle: new -> applied | dismissed."""
    if status not in LESSON_STATUSES:
        raise ValueError(f"status must be one of {LESSON_STATUSES}")
    with _lesson_lock(exclusive=True):
        ledger = _load_lessons_unlocked()
        for existing in ledger["lessons"]:
            if existing.get("id") == lesson_id:
                existing["status"] = status
                existing["status_changed_at"] = _utcnow().isoformat()
                ledger["updated_at"] = _utcnow().isoformat()
                _atomic_write_json(LESSONS_FILE, ledger)
                return True
    return False


def reconcile_lesson_routes(
    *,
    candidate_promotion_ids: set[str] | None = None,
) -> dict[str, int]:
    """Backfill routes and expire/reactivate stale shadow promotion notices."""
    promotion_snapshot_available = candidate_promotion_ids is not None
    if candidate_promotion_ids is None:
        try:
            from .shadow_rules import summarize_shadow_rules

            candidate_promotion_ids = {
                str(row.get("rule_id") or "")
                for row in (summarize_shadow_rules().get("candidate_promotions") or [])
                if isinstance(row, dict) and row.get("rule_id")
            }
            promotion_snapshot_available = True
        except Exception as exc:
            logger.warning("[review] Shadow-promotion lifecycle check unavailable: %s", exc)
            candidate_promotion_ids = set()
    with _lesson_lock(exclusive=True):
        ledger = _load_lessons_unlocked()
        updated = 0
        unmapped = 0
        expired = 0
        reactivated = 0
        now_iso = _utcnow().isoformat()
        for lesson in ledger.get("lessons", []):
            if not isinstance(lesson, dict):
                continue
            route = get_lesson_consumer_route(lesson.get("type", ""))
            if route["consumer"] == "unmapped":
                unmapped += 1
            if (
                lesson.get("consumer") != route["consumer"]
                or lesson.get("decision_effect") != route["effect"]
            ):
                lesson["consumer"] = route["consumer"]
                lesson["decision_effect"] = route["effect"]
                updated += 1
            if (
                promotion_snapshot_available
                and str(lesson.get("type") or "") == "shadow_rule_promotion"
            ):
                context = lesson.get("context") if isinstance(lesson.get("context"), dict) else {}
                rule_id = str(context.get("rule_id") or "")
                status = str(lesson.get("status") or "new")
                if status == "new" and rule_id and rule_id not in candidate_promotion_ids:
                    lesson["status"] = "expired"
                    lesson["status_changed_at"] = now_iso
                    lesson["status_reason"] = "Rule no longer meets current promotion criteria."
                    expired += 1
                elif status == "expired" and rule_id in candidate_promotion_ids:
                    lesson["status"] = "new"
                    lesson["status_changed_at"] = now_iso
                    lesson["status_reason"] = "Rule meets current promotion criteria again."
                    reactivated += 1
        if updated or expired or reactivated:
            ledger["updated_at"] = _utcnow().isoformat()
            _atomic_write_json(LESSONS_FILE, ledger)
        return {
            "total": len(ledger.get("lessons", [])),
            "updated": updated,
            "unmapped": unmapped,
            "expired": expired,
            "reactivated": reactivated,
        }


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

        # Repair historical fallback grades before deriving patterns or lessons.
        # A provider outage must not teach the next Council from a weaker,
        # absolute-return substitute.
        if Config.ACCURACY_BENCHMARK_BACKFILL_ENABLED:
            try:
                findings["benchmark_backfill"] = self.accuracy.backfill_missing_benchmark_grades(
                    max_tickers=Config.ACCURACY_BENCHMARK_BACKFILL_MAX_TICKERS,
                )
            except Exception as exc:
                logger.warning("[review] Benchmark-grade backfill failed: %s", exc)
                findings["benchmark_backfill"] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }

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
        """Grade due recommendations at one fixed, benchmark-matched horizon."""
        result = self.accuracy.grade_due_recommendations_from_history(
            max_records=Config.ACCURACY_NIGHTLY_GRADE_MAX_RECORDS,
        )
        if result.get("skipped"):
            logger.warning(
                "[review] Historical grading left %d due row(s) pending for retry",
                len(result["skipped"]),
            )
        return list(result.get("graded_records") or [])

    def _analyze_accuracy_patterns(self) -> dict:
        """Look for patterns in graded recommendations."""
        stats = self.accuracy.get_summary_stats()
        current_stats = self.accuracy.get_summary_stats(
            since=Config.ACCURACY_CURRENT_ERA_START,
            benchmark_only=True,
        )
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
            try:
                profile = collector.company_profile(symbol) or {}
            except Exception:
                profile = {}

            # The lesson must come from the same single-company opportunity set
            # Artha is allowed to buy. Leveraged/inverse ETFs in the top-gainers
            # feed are not missed stocks and must not teach the Council to chase.
            if not _eligible_missed_winner_instrument(profile, quote):
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
                "exchange": str(profile.get("exchangeShortName") or "").upper(),
                "instrument_identity_verified": True,
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


def build_council_learning_context(
    max_items: int = 5,
    max_chars: int = 1200,
    *,
    records: list[dict[str, Any]] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build the bounded, current-era feedback packet used by Buy Council.

    The packet is deliberately balanced between missed no-buy opportunities and
    bad/early buy calls, admits only one observation per ticker, and never
    injects unreviewed free-form lessons into a live decision.
    """
    if records is None:
        tracker = AccuracyTracker()
        lf = tracker._lock(exclusive=False)
        try:
            records = tracker._load()
        finally:
            tracker._unlock(lf)

    from .accuracy import (
        EXPOSURE_VERDICTS,
        _current_era_start,
        _is_current_era,
        _normalize_verdict,
        _num_or_none,
    )

    current_time = now or _utcnow()
    current_graded = [
        rec for rec in records
        if rec.get("status") == "GRADED" and _is_current_era(rec)
    ]
    benchmark_graded = [
        rec
        for rec in current_graded
        if str(rec.get("grade_basis") or "").startswith("excess_vs_")
        and _num_or_none(rec.get("excess_return_pct")) is not None
    ]
    categories: dict[str, list[dict[str, Any]]] = {"missed_no_buy": [], "bad_buy": []}
    for rec in benchmark_graded:
        grade = str(rec.get("grade") or "").upper()
        if grade not in {"INCORRECT", "PARTIALLY_CORRECT"}:
            continue
        excess = _num_or_none(rec.get("excess_return_pct"))
        if excess is None:
            continue
        ts = str(rec.get("timestamp") or "")
        try:
            rec_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if rec_dt.tzinfo is None:
                rec_dt = rec_dt.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (current_time - rec_dt).total_seconds() / 86400.0)
        except ValueError:
            age_days = 365.0
        verdict = str(rec.get("verdict") or "?").upper()
        category = (
            "bad_buy"
            if _normalize_verdict(verdict) in EXPOSURE_VERDICTS
            else "missed_no_buy"
        )
        ticker = str(rec.get("ticker") or "?").upper()
        item = {
            "ticker": ticker,
            "timestamp": ts,
            "date": ts[:10],
            "verdict": verdict,
            "grade": grade,
            "excess_return_pct": round(float(excess), 4),
            "category": category,
            "weight": abs(float(excess)) * (0.5 ** (age_days / 60.0)),
        }
        categories[category].append(item)

    for items in categories.values():
        items.sort(key=lambda item: (-float(item["weight"]), item["ticker"], item["timestamp"]))

    selected: list[dict[str, Any]] = []
    seen_tickers: set[str] = set()

    def _take_first(category: str) -> None:
        for item in categories[category]:
            if item["ticker"] not in seen_tickers:
                selected.append(item)
                seen_tickers.add(item["ticker"])
                return

    remaining = sorted(
        [*categories["missed_no_buy"], *categories["bad_buy"]],
        key=lambda item: (-float(item["weight"]), item["ticker"], item["timestamp"]),
    )
    # Guarantee both error directions are represented when there is room. With
    # a one-item budget, choose the strongest available observation instead of
    # accidentally returning nothing when only one category exists.
    if max_items >= 2 and categories["missed_no_buy"] and categories["bad_buy"]:
        _take_first("missed_no_buy")
        _take_first("bad_buy")
    for item in remaining:
        if len(selected) >= max(0, int(max_items)):
            break
        if item["ticker"] in seen_tickers:
            continue
        selected.append(item)
        seen_tickers.add(item["ticker"])

    missed_values = [float(item["excess_return_pct"]) for item in categories["missed_no_buy"]]
    bad_buy_values = [float(item["excess_return_pct"]) for item in categories["bad_buy"]]
    era_start = _current_era_start().date().isoformat()
    lines = [
        "CURRENT-ERA GRADED POSTMORTEMS "
        f"(since {era_start}; benchmark-graded={len(benchmark_graded)}/{len(current_graded)}; "
        f"missed no-buy={len(missed_values)}; weak buys={len(bad_buy_values)}):"
    ]
    for item in selected:
        if item["category"] == "bad_buy":
            interpretation = (
                "Historical buy exposure underperformed; re-check entry timing, "
                "expectation risk, and thesis quality."
            )
        else:
            interpretation = (
                "Historical no-buy call missed relative strength; re-check whether "
                "supported momentum or catalyst evidence was underweighted."
            )
        lines.append(
            f"- {item['verdict']} {item['ticker']} ({item['date']}): benchmark-relative "
            f"return {item['excess_return_pct']:+.1f}pp; graded {item['grade']}. {interpretation}"
        )
    if not selected:
        lines.append("- No current-era incorrect or partial outcomes are eligible yet.")
    footer = (
        "Calibration only: these are correlated historical observations, not causal proof. "
        "Do not override current filings, live data, valuation, risk gates, or execution evidence."
    )
    lines.append(footer)

    # Preserve the header and safety footer; drop lower-ranked examples first.
    while len("\n".join(lines)) > max_chars and len(lines) > 2:
        lines.pop(-2)
        if selected:
            selected.pop()
    digest = "\n".join(lines)
    if len(digest) > max_chars:
        digest = digest[: max(0, max_chars - 3)].rstrip() + "..."
    return {
        "schema_version": 1,
        "current_era_start": era_start,
        "current_era_graded": len(current_graded),
        "benchmark_graded": len(benchmark_graded),
        "excluded_absolute_fallback": len(current_graded) - len(benchmark_graded),
        "eligible_missed_no_buy": len(categories["missed_no_buy"]),
        "eligible_bad_buy": len(categories["bad_buy"]),
        "selected": selected,
        "digest": digest,
        "digest_sha256": hashlib.sha256(digest.encode("utf-8")).hexdigest(),
        "policy": {
            "current_era_only": True,
            "max_one_observation_per_ticker": True,
            "balanced_when_available": True,
            "unreviewed_ledger_lessons_in_live_prompt": False,
            "automatic_rule_changes": False,
        },
    }


def get_council_lessons(max_items: int = 5, max_chars: int = 1200) -> str:
    """Return the bounded Buy Council feedback digest."""
    return str(
        build_council_learning_context(max_items=max_items, max_chars=max_chars).get("digest")
        or ""
    )
