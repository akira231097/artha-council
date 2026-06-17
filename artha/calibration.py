"""Score calibration and point-in-time feature backfill for Artha."""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dossier import extract_decision_feature_row
from .journal import DecisionJournal

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOSSIER_DIR = DATA_DIR / "decision_dossiers"


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _score_bucket(score: float) -> str:
    if score >= 75:
        return "75-100"
    if score >= 65:
        return "65-74"
    if score >= 55:
        return "55-64"
    if score >= 45:
        return "45-54"
    return "0-44"


def backfill_decision_features(
    journal: DecisionJournal | None = None,
    dossier_dir: Path = DOSSIER_DIR,
) -> int:
    """Backfill compact feature rows from all existing decision dossiers."""
    journal = journal or DecisionJournal()
    if not dossier_dir.exists():
        return 0

    count = 0
    for path in sorted(dossier_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not payload.get("ticker"):
                continue
            journal.save_decision_features(extract_decision_feature_row(payload, str(path)))
            count += 1
        except Exception as exc:
            logger.warning("[calibration] Failed to backfill %s: %s", path, exc)
    return count


def _recommendation_rows(journal: DecisionJournal) -> list[dict[str, Any]]:
    with journal._connect() as conn:
        rows = conn.execute(
            """
            SELECT id, timestamp, ticker, action, confidence,
                   price_at_recommendation, status, outcome, outcome_notes
            FROM recommendations
            ORDER BY datetime(timestamp) DESC, id DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def _shadow_rows(journal: DecisionJournal) -> list[dict[str, Any]]:
    with journal._connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM shadow_positions
            ORDER BY datetime(created_at) DESC, id DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def _avg(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_num(row.get(key), default=None) for row in rows if row.get(key) is not None]
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def build_calibration_report(journal: DecisionJournal | None = None) -> dict[str, Any]:
    """Build a calibration snapshot from recommendations, features, and shadows."""
    journal = journal or DecisionJournal()
    features = journal.get_decision_features(limit=1000)
    recs = _recommendation_rows(journal)
    shadows = _shadow_rows(journal)

    by_score: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in features:
        score = _num(row.get("adjusted_score") if row.get("adjusted_score") is not None else row.get("opportunity_score"))
        by_score[_score_bucket(score)].append(row)

    score_buckets = {}
    for bucket, rows in sorted(by_score.items()):
        verdict_counts: dict[str, int] = {}
        for row in rows:
            verdict = str(row.get("final_verdict") or "UNKNOWN").upper()
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        score_buckets[bucket] = {
            "count": len(rows),
            "avg_confidence": _avg(rows, "confidence"),
            "avg_context_coverage": _avg(rows, "context_coverage_score"),
            "avg_evidence_count": _avg(rows, "evidence_count"),
            "verdict_counts": verdict_counts,
        }

    action_counts: dict[str, int] = {}
    for row in recs:
        action = str(row.get("action") or "UNKNOWN").upper()
        action_counts[action] = action_counts.get(action, 0) + 1

    completed_shadows = [row for row in shadows if row.get("status") == "completed"]
    shadow_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in shadows:
        shadow_by_bucket[_score_bucket(_num(row.get("opportunity_score")))].append(row)

    shadow_buckets = {}
    for bucket, rows in sorted(shadow_by_bucket.items()):
        rows_with_excess_5d = [r for r in rows if r.get("excess_return_5d") is not None]
        rows_with_excess_20d = [r for r in rows if r.get("excess_return_20d") is not None]
        rows_with_excess_60d = [r for r in rows if r.get("excess_return_60d") is not None]
        shadow_buckets[bucket] = {
            "count": len(rows),
            "completed": sum(1 for r in rows if r.get("status") == "completed"),
            "avg_return_5d": _avg(rows, "return_5d"),
            "avg_return_20d": _avg(rows, "return_20d"),
            "avg_return_60d": _avg(rows, "return_60d"),
            "avg_benchmark_return_5d": _avg(rows, "benchmark_return_5d"),
            "avg_benchmark_return_20d": _avg(rows, "benchmark_return_20d"),
            "avg_benchmark_return_60d": _avg(rows, "benchmark_return_60d"),
            "avg_sector_benchmark_return_5d": _avg(rows, "sector_benchmark_return_5d"),
            "avg_sector_benchmark_return_20d": _avg(rows, "sector_benchmark_return_20d"),
            "avg_sector_benchmark_return_60d": _avg(rows, "sector_benchmark_return_60d"),
            "avg_excess_return_5d": _avg(rows, "excess_return_5d"),
            "avg_excess_return_20d": _avg(rows, "excess_return_20d"),
            "avg_excess_return_60d": _avg(rows, "excess_return_60d"),
            "relative_hit_rate_5d": (
                round(sum(1 for r in rows_with_excess_5d if _num(r.get("excess_return_5d")) > 0) / len(rows_with_excess_5d), 4)
                if rows_with_excess_5d else None
            ),
            "relative_hit_rate_20d": (
                round(sum(1 for r in rows_with_excess_20d if _num(r.get("excess_return_20d")) > 0) / len(rows_with_excess_20d), 4)
                if rows_with_excess_20d else None
            ),
            "relative_hit_rate_60d": (
                round(sum(1 for r in rows_with_excess_60d if _num(r.get("excess_return_60d")) > 0) / len(rows_with_excess_60d), 4)
                if rows_with_excess_60d else None
            ),
            "avg_mfe": _avg(rows, "mfe"),
            "avg_mae": _avg(rows, "mae"),
            "stop_hit_rate": (
                round(sum(1 for r in rows if r.get("would_hit_stop")) / len(rows), 4)
                if rows else None
            ),
        }

    enough_forward_data = len(completed_shadows) >= 20
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision_feature_rows": len(features),
        "recommendation_rows": len(recs),
        "action_counts": action_counts,
        "score_buckets": score_buckets,
        "shadow_rows": len(shadows),
        "completed_shadow_rows": len(completed_shadows),
        "shadow_score_buckets": shadow_buckets,
        "calibration_status": (
            "usable" if enough_forward_data else "insufficient_forward_samples"
        ),
        "relative_calibration_status": (
            "usable" if enough_forward_data else "insufficient_forward_samples"
        ),
        "minimum_shadow_samples_for_threshold_tuning": 20,
    }


def format_calibration_report(report: dict[str, Any]) -> str:
    """Human-readable calibration report."""
    lines = [
        "ARTHA SCORE CALIBRATION",
        "=======================",
        f"Generated: {report.get('generated_at')}",
        f"Decision feature rows: {report.get('decision_feature_rows', 0)}",
        f"Recommendations: {report.get('recommendation_rows', 0)}",
        f"Shadow rows: {report.get('shadow_rows', 0)} "
        f"({report.get('completed_shadow_rows', 0)} completed)",
        f"Status: {report.get('calibration_status')}",
        "",
        "Action mix:",
    ]
    action_counts = report.get("action_counts") or {}
    if action_counts:
        for action, count in sorted(action_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"  {action}: {count}")
    else:
        lines.append("  none")

    lines.extend(["", "Decision score buckets:"])
    for bucket, data in sorted((report.get("score_buckets") or {}).items()):
        lines.append(
            f"  {bucket}: n={data.get('count')} "
            f"avg_conf={data.get('avg_confidence')} "
            f"avg_evidence={data.get('avg_evidence_count')} "
            f"verdicts={data.get('verdict_counts')}"
        )

    lines.extend(["", "Shadow forward-return buckets:"])
    shadow_buckets = report.get("shadow_score_buckets") or {}
    if shadow_buckets:
        for bucket, data in sorted(shadow_buckets.items()):
            lines.append(
                f"  {bucket}: n={data.get('count')} completed={data.get('completed')} "
                f"r5={data.get('avg_return_5d')} r20={data.get('avg_return_20d')} "
                f"r60={data.get('avg_return_60d')} "
                f"excess5={data.get('avg_excess_return_5d')} "
                f"excess20={data.get('avg_excess_return_20d')} "
                f"excess60={data.get('avg_excess_return_60d')} "
                f"hit20={data.get('relative_hit_rate_20d')} "
                f"mfe={data.get('avg_mfe')} mae={data.get('avg_mae')}"
            )
    else:
        lines.append("  none")

    if report.get("calibration_status") != "usable":
        lines.extend(
            [
                "",
                "Interpretation: score thresholds are being logged, but Artha should not",
                "auto-retune BUY/DEFER thresholds until more completed forward samples exist.",
            ]
        )
    return "\n".join(lines)
