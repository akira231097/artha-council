"""Local integrity report for Artha's guarded feedback loops.

This module verifies that evidence has an explicit consumer. It never changes
scores, thresholds, position sizes, or broker actions.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .accuracy import AccuracyTracker, _is_current_era
from .calibration import build_calibration_report
from .config import Config
from .execution_learning import build_execution_learning_summary, build_sell_learning_context
from .journal import DecisionJournal
from .meta_ranker import build_meta_signal
from .portfolio import PORTFOLIO_FILE, Portfolio
from .pre_brief import DATA_DIR, PreBrief
from .self_review import (
    build_council_learning_context,
    get_lesson_consumer_route,
    load_lessons,
)
from .shadow_rules import summarize_shadow_rules

DOSSIER_DIR = DATA_DIR / "decision_dossiers"


def _accuracy_records() -> list[dict[str, Any]]:
    tracker = AccuracyTracker()
    lock = tracker._lock(exclusive=False)
    try:
        return tracker._load()
    finally:
        tracker._unlock(lock)


def _latest_dossier_provenance(dossier_dir: Path) -> dict[str, Any]:
    paths = sorted(
        dossier_dir.rglob("*.json") if dossier_dir.exists() else [],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not paths:
        return {"status": "not_observed", "path": "", "schema_version": 0}
    path = paths[0]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "invalid",
            "path": str(path),
            "schema_version": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }
    schema = int(payload.get("schema_version") or 1)
    context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
    learning = context.get("council_learning") if isinstance(context.get("council_learning"), dict) else {}
    if schema < 2:
        status = "legacy_dossier"
    elif learning.get("digest_sha256") and isinstance(learning.get("policy"), dict):
        status = "provenance_recorded"
    else:
        status = "missing_provenance"
    return {
        "status": status,
        "path": str(path),
        "schema_version": schema,
        "digest_sha256": str(learning.get("digest_sha256") or ""),
        "current_era_start": learning.get("current_era_start"),
    }


def build_feedback_loop_report(
    journal: DecisionJournal,
    *,
    portfolio_path: str | Path = PORTFOLIO_FILE,
    dossier_dir: str | Path = DOSSIER_DIR,
    prebrief_data_dir: str | Path = DATA_DIR,
    records: list[dict[str, Any]] | None = None,
    lessons: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a no-network report proving feedback producers and consumers."""
    accuracy_records = list(records) if records is not None else _accuracy_records()
    lesson_ledger = lessons if lessons is not None else load_lessons()
    learning = build_council_learning_context(records=accuracy_records)

    current_graded = [
        row
        for row in accuracy_records
        if row.get("status") == "GRADED" and _is_current_era(row)
    ]
    current_pending = [
        row
        for row in accuracy_records
        if row.get("status") != "GRADED" and _is_current_era(row)
    ]

    status_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    consumer_counts: Counter[str] = Counter()
    effect_counts: Counter[str] = Counter()
    unmapped_types: set[str] = set()
    pending_manual_reviews: list[dict[str, str]] = []
    for lesson in lesson_ledger.get("lessons", []):
        if not isinstance(lesson, dict):
            continue
        lesson_type = str(lesson.get("type") or "unknown")
        route = get_lesson_consumer_route(lesson_type)
        status_counts[str(lesson.get("status") or "unknown")] += 1
        type_counts[lesson_type] += 1
        consumer_counts[route["consumer"]] += 1
        effect_counts[route["effect"]] += 1
        if route["consumer"] == "unmapped":
            unmapped_types.add(lesson_type)
        if route["effect"] == "manual_review" and str(lesson.get("status") or "new") == "new":
            pending_manual_reviews.append(
                {
                    "id": str(lesson.get("id") or ""),
                    "type": lesson_type,
                    "priority": str(lesson.get("priority") or "LOW"),
                    "first_seen": str(lesson.get("first_seen") or ""),
                }
            )

    execution = build_execution_learning_summary(journal, portfolio_path=portfolio_path)
    sell_learning = build_sell_learning_context(journal)
    shadow = summarize_shadow_rules(journal)
    calibration = build_calibration_report(journal)
    meta_signal = build_meta_signal(journal)
    latest_diagnostic = journal.get_latest_calibration_diagnostic() or {}
    portfolio = Portfolio.load(Path(portfolio_path))
    prebrief = PreBrief(data_dir=Path(prebrief_data_dir))
    verified_events = prebrief.get_events(
        hours=Config.SENTINEL_VERIFIED_NEGATIVE_LOOKBACK_HOURS,
        source="sentinel_thesis_impact",
        event_type="thesis_impact",
        verified_negative=True,
    )
    active_sentinel_signals = [
        row
        for row in journal.get_active_sell_signals()
        if str(row.get("source") or "").startswith("news_sentinel_")
    ]
    provenance = _latest_dossier_provenance(Path(dossier_dir))

    failures: list[str] = []
    warnings: list[str] = []
    eligible_misses = int(learning.get("eligible_missed_no_buy") or 0) + int(
        learning.get("eligible_bad_buy") or 0
    )
    if eligible_misses and not learning.get("selected"):
        failures.append("Current-era graded misses exist but the Buy Council digest selected none.")
    if unmapped_types:
        failures.append(
            "Lesson types have no declared consumer: " + ", ".join(sorted(unmapped_types))
        )
    if portfolio.positions and not Config.SENTINEL_ENABLED:
        failures.append("Held positions exist while News Sentinel is disabled.")
    if portfolio.positions and not Config.SELL_SENTINEL_COUNCIL_ESCALATION_ENABLED:
        failures.append("Held-position Sentinel findings cannot escalate to Sell Council.")
    if provenance.get("status") in {"invalid", "missing_provenance"}:
        failures.append("The newest v2 Council dossier does not preserve learning provenance.")
    if execution.get("data_quality", {}).get("missing_realized_pnl_events"):
        warnings.append("Some sell fills are missing realized-P&L measurements.")
    if execution.get("post_sell", {}).get("overdue_60d_reviews"):
        warnings.append("Some mature post-sell reviews are overdue.")
    benchmark_graded = int(learning.get("benchmark_graded") or 0)
    excluded_fallback = int(learning.get("excluded_absolute_fallback") or 0)
    minimum_buy_samples = max(1, int(getattr(Config, "ACCURACY_MIN_PATTERN_SAMPLES", 3)))
    if current_graded and benchmark_graded < minimum_buy_samples:
        warnings.append(
            "Buy-outcome learning lacks enough benchmark-backed grades "
            f"({benchmark_graded}/{minimum_buy_samples}); absolute fallback grades stay out of live prompts."
        )
    if excluded_fallback:
        warnings.append(
            f"{excluded_fallback} current-era grade(s) still need benchmark backfill and are excluded from live prompts."
        )
    if pending_manual_reviews:
        warnings.append(
            f"{len(pending_manual_reviews)} lesson(s) still require explicit manual review."
        )
    calibration_completed = int(calibration.get("completed_shadow_rows") or 0)
    meta_minimum = int(meta_signal.get("minimum_samples") or 0)
    if calibration_completed >= meta_minimum > 0 and meta_signal.get("status") != "usable":
        failures.append(
            "Calibration has enough completed samples, but the Council meta-signal is not usable."
        )

    if failures:
        status = "FAIL"
    elif warnings:
        status = "WARN"
    else:
        status = "PASS"
    return {
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "guardrail": "Feedback is advisory until its explicit sample gate passes; no automatic rule mutation.",
        "buy_outcome_feedback": {
            "current_era_graded": len(current_graded),
            "current_era_pending": len(current_pending),
            "benchmark_graded": benchmark_graded,
            "excluded_absolute_fallback": excluded_fallback,
            "eligible_misses": eligible_misses,
            "selected_examples": len(learning.get("selected") or []),
            "digest_sha256": learning.get("digest_sha256"),
            "policy": learning.get("policy") or {},
        },
        "lesson_lifecycle": {
            "total": sum(status_counts.values()),
            "status_counts": dict(sorted(status_counts.items())),
            "type_counts": dict(sorted(type_counts.items())),
            "consumer_counts": dict(sorted(consumer_counts.items())),
            "effect_counts": dict(sorted(effect_counts.items())),
            "unmapped_types": sorted(unmapped_types),
            "observational_items_reported": effect_counts.get("observational", 0),
            "pending_manual_reviews": pending_manual_reviews,
            "reported_by_supervisor": True,
        },
        "sell_outcome_feedback": {
            "status": sell_learning.get("status"),
            "ready": bool(sell_learning.get("ready")),
            "completed": int(sell_learning.get("completed") or 0),
            "minimum_completed": int(sell_learning.get("minimum_completed") or 0),
            "episodes": execution.get("episodes") or {},
            "post_sell": execution.get("post_sell") or {},
        },
        "shadow_feedback": {
            "total": int(shadow.get("total") or 0),
            "tracking": int(shadow.get("tracking") or 0),
            "completed": int(shadow.get("completed") or 0),
            "candidate_promotions": shadow.get("candidate_promotions") or [],
            "automatic_promotion": False,
        },
        "calibration_feedback": {
            "tracked_decision_shadows": int(calibration.get("shadow_rows") or 0),
            "completed_decision_shadows": calibration_completed,
            "status": calibration.get("calibration_status"),
            "meta_signal_status": meta_signal.get("status"),
            "meta_signal_recommendation": meta_signal.get("recommendation"),
            "minimum_samples": meta_minimum,
            "latest_diagnostic_stage": latest_diagnostic.get("stage"),
            "latest_diagnostic_completed_samples": int(
                latest_diagnostic.get("completed_samples") or 0
            ),
            "live_prompt_injection": True,
            "automatic_rule_changes": False,
        },
        "sentinel_feedback": {
            "enabled": bool(Config.SENTINEL_ENABLED),
            "held_positions": len(portfolio.positions),
            "verified_negative_events_in_lookback": len(verified_events),
            "durable_active_review_signals": len(active_sentinel_signals),
            "sell_council_escalation_enabled": bool(
                Config.SELL_SENTINEL_COUNCIL_ESCALATION_ENABLED
            ),
            "raw_keyword_alerts_can_trade": False,
        },
        "decision_provenance": provenance,
    }
