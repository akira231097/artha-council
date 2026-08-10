"""Calibration-gated meta signal for Artha decisions.

This is intentionally conservative. Until Artha has enough matured forward
outcomes, the meta layer refuses to tune thresholds or boost recommendations.

When the caller cannot supply an ``opportunity_score`` (the council builds the
meta signal before the CIO scores the name), the signal no longer dead-ends in
a permanently empty "unscored" bucket: it derives a proxy score from the
deterministic components in ``stock_data`` when available, and otherwise falls
back to an aggregate across all score buckets so historical outcomes still
inform the recommendation.
"""
from __future__ import annotations

import os
from typing import Any


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Tunables (Wave 2 consolidates into Config).
META_RANKER_MIN_SAMPLES = int(_env_float("ARTHA_META_RANKER_MIN_SAMPLES", 20))
META_RANKER_MIN_BUCKET_SAMPLES = int(_env_float("ARTHA_META_RANKER_MIN_BUCKET_SAMPLES", 5))
META_RANKER_EXCESS_GATE = _env_float("ARTHA_META_RANKER_EXCESS_GATE", 0.02)


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


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def derive_opportunity_score_proxy(stock_data: dict[str, Any] | None) -> float | None:
    """Derive a deterministic proxy opportunity score from dossier components.

    Used when the caller has no CIO score yet. Mirrors the components the CIO
    weighs (consensus upside, revision trend, expectation risk, timing) so the
    candidate lands in a sensible calibration bucket. Returns None when no
    usable components are present.
    """
    if not isinstance(stock_data, dict):
        return None
    valuation = stock_data.get("valuation_expectations") or {}
    if not isinstance(valuation, dict) or not valuation:
        return None
    targets = valuation.get("analyst_targets") or {}
    revision = valuation.get("revision_trend") or {}
    timing = valuation.get("timing_risk") or {}

    upside = _num((targets or {}).get("consensus_upside_pct"))
    net_revision = _num((revision or {}).get("net_revision_30d"))
    rsi = _num((timing or {}).get("rsi"))
    expectation = str(valuation.get("expectation_risk_level") or "").lower()

    if upside is None and net_revision is None and rsi is None and not expectation:
        return None

    score = 50.0
    if upside is not None:
        score += 0.4 * max(-25.0, min(50.0, upside))
    if net_revision is not None:
        score += 8.0 if net_revision > 0 else (-8.0 if net_revision < 0 else 0.0)
    if expectation == "high":
        score -= 10.0
    elif expectation == "low":
        score += 5.0
    if rsi is not None:
        if rsi >= 70:
            score -= 5.0
        elif 40 <= rsi <= 60:
            score += 3.0
    return max(0.0, min(100.0, score))


def _aggregate_buckets(report: dict[str, Any]) -> dict[str, Any]:
    """Completed-weighted aggregate across all shadow score buckets."""
    buckets = report.get("shadow_score_buckets") or {}
    total_count = 0
    total_completed = 0
    sums: dict[str, float] = {"avg_excess_return_20d": 0.0, "avg_excess_return_60d": 0.0}
    weights: dict[str, int] = {"avg_excess_return_20d": 0, "avg_excess_return_60d": 0}
    for data in buckets.values():
        count = int(data.get("count") or 0)
        completed = int(data.get("completed") or 0)
        total_count += count
        total_completed += completed
        for key in ("avg_excess_return_20d", "avg_excess_return_60d"):
            value = _num(data.get(key))
            if value is not None and completed > 0:
                sums[key] += value * completed
                weights[key] += completed
    return {
        "count": total_count,
        "completed": total_completed,
        "avg_excess_return_20d": (
            round(sums["avg_excess_return_20d"] / weights["avg_excess_return_20d"], 4)
            if weights["avg_excess_return_20d"] else None
        ),
        "avg_excess_return_60d": (
            round(sums["avg_excess_return_60d"] / weights["avg_excess_return_60d"], 4)
            if weights["avg_excess_return_60d"] else None
        ),
    }


def build_meta_signal(
    journal: Any,
    opportunity_score: float | None = None,
    minimum_samples: int = META_RANKER_MIN_SAMPLES,
    stock_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a calibration-aware meta signal from prior forward outcomes.

    ``opportunity_score`` buckets the candidate against historical outcomes.
    When absent, a proxy score is derived from ``stock_data`` components; when
    that is also unavailable, the signal falls back to an all-bucket aggregate
    instead of the previous permanently-empty "unscored" bucket.
    """
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

    score_source = "caller"
    if opportunity_score is None:
        derived = derive_opportunity_score_proxy(stock_data)
        if derived is not None:
            opportunity_score = derived
            score_source = "derived_component_proxy"
        else:
            score_source = "all_bucket_aggregate"

    if opportunity_score is not None:
        candidate_bucket = _bucket(float(opportunity_score))
        bucket_data = (report.get("shadow_score_buckets") or {}).get(candidate_bucket, {})
    else:
        candidate_bucket = "all_buckets"
        bucket_data = _aggregate_buckets(report)

    if completed < minimum_samples:
        return {
            "status": "insufficient_outcomes",
            "minimum_samples": minimum_samples,
            "completed_shadow_rows": completed,
            "candidate_score_bucket": candidate_bucket,
            "score_source": score_source,
            "bucket_data": bucket_data,
            "recommendation": "do_not_adjust",
            "reason": "Not enough completed forward samples to train or trust a meta-ranker.",
        }

    excess_20d = _num(bucket_data.get("avg_excess_return_20d"))
    excess_60d = _num(bucket_data.get("avg_excess_return_60d"))
    completed_bucket = int(bucket_data.get("completed") or 0)
    excess_points = [v for v in (excess_20d, excess_60d) if v is not None]
    avg_excess = sum(excess_points) / len(excess_points) if excess_points else None
    if completed_bucket < META_RANKER_MIN_BUCKET_SAMPLES:
        recommendation = "do_not_adjust"
        reason = (
            f"Candidate score bucket has fewer than "
            f"{META_RANKER_MIN_BUCKET_SAMPLES} completed outcomes."
        )
    elif avg_excess is not None and avg_excess >= META_RANKER_EXCESS_GATE:
        recommendation = "historical_bucket_supports_score"
        reason = "Candidate score bucket has positive benchmark-relative forward returns."
    elif avg_excess is not None and avg_excess <= -META_RANKER_EXCESS_GATE:
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
        "score_source": score_source,
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
        f"{payload.get('minimum_samples', META_RANKER_MIN_SAMPLES)}",
        f"Candidate score bucket: {payload.get('candidate_score_bucket', 'N/A')}"
        f" (source: {payload.get('score_source', 'caller')})",
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
