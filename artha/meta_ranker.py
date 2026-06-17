"""Calibration-gated meta signal for Artha decisions.

This is intentionally conservative. Until Artha has enough matured forward
outcomes, the meta layer refuses to tune thresholds or boost recommendations.
"""
from __future__ import annotations

from typing import Any


def _bucket(score: float) -> str:
    if score >= 75:
        return "75-100"
    if score >= 65:
        return "65-74"
    if score >= 55:
        return "55-64"
    if score >= 45:
        return "45-54"
    return "0-44"


def build_meta_signal(
    journal: Any,
    opportunity_score: float | None = None,
    minimum_samples: int = 20,
) -> dict[str, Any]:
    """Return a calibration-aware meta signal from prior forward outcomes."""
    try:
        from .calibration import build_calibration_report
        report = build_calibration_report(journal)
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"calibration report unavailable: {exc}",
            "minimum_samples": minimum_samples,
            "completed_shadow_rows": 0,
            "recommendation": "do_not_adjust",
        }

    completed = int(report.get("completed_shadow_rows") or 0)
    candidate_bucket = _bucket(float(opportunity_score)) if opportunity_score is not None else "unscored"
    bucket_data = (report.get("shadow_score_buckets") or {}).get(candidate_bucket, {})
    if completed < minimum_samples:
        return {
            "status": "insufficient_outcomes",
            "minimum_samples": minimum_samples,
            "completed_shadow_rows": completed,
            "candidate_score_bucket": candidate_bucket,
            "bucket_data": bucket_data,
            "recommendation": "do_not_adjust",
            "reason": "Not enough completed forward samples to train or trust a meta-ranker.",
        }

    excess_20d = bucket_data.get("avg_excess_return_20d")
    excess_60d = bucket_data.get("avg_excess_return_60d")
    completed_bucket = int(bucket_data.get("completed") or 0)
    if completed_bucket < 5:
        recommendation = "do_not_adjust"
        reason = "Candidate score bucket has fewer than 5 completed outcomes."
    elif excess_20d is not None and excess_60d is not None and (excess_20d + excess_60d) / 2 >= 0.02:
        recommendation = "historical_bucket_supports_score"
        reason = "Candidate score bucket has positive benchmark-relative forward returns."
    elif excess_20d is not None and excess_60d is not None and (excess_20d + excess_60d) / 2 <= -0.02:
        recommendation = "historical_bucket_warns_against_score"
        reason = "Candidate score bucket has negative benchmark-relative forward returns."
    else:
        recommendation = "neutral"
        reason = "Historical bucket returns are mixed or close to benchmark."

    return {
        "status": "usable",
        "minimum_samples": minimum_samples,
        "completed_shadow_rows": completed,
        "candidate_score_bucket": candidate_bucket,
        "bucket_data": bucket_data,
        "recommendation": recommendation,
        "reason": reason,
    }


def format_meta_signal(payload: dict[str, Any] | None) -> str:
    """Render meta-calibration context for the CIO."""
    if not payload:
        return "Calibration meta-signal unavailable."
    status = payload.get("status", "unknown")
    lines = [
        "CALIBRATION / META-RANKER CHECK",
        f"Status: {status} | completed outcomes: {payload.get('completed_shadow_rows', 0)}/"
        f"{payload.get('minimum_samples', 20)}",
        f"Candidate score bucket: {payload.get('candidate_score_bucket', 'N/A')}",
        f"Recommendation: {payload.get('recommendation', 'do_not_adjust')}",
        f"Reason: {payload.get('reason', '')}",
    ]
    bucket = payload.get("bucket_data") or {}
    if bucket:
        lines.append(
            "Bucket stats: n={} completed={} excess20={} excess60={}".format(
                bucket.get("count"),
                bucket.get("completed"),
                bucket.get("avg_excess_return_20d"),
                bucket.get("avg_excess_return_60d"),
            )
        )
    return "\n".join(lines)
