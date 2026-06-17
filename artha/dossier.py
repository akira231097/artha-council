"""Point-in-time decision dossiers for Artha council outputs.

Dossiers are audit artifacts. They preserve the decision, evidence trace,
analyst outputs, and a compact source-data snapshot so future calibration can
evaluate what Artha knew at recommendation time.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOSSIER_DIR = DATA_DIR / "decision_dossiers"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _json_safe(value: Any) -> Any:
    """Convert common non-JSON values into deterministic JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    try:
        return float(value)
    except Exception:
        return str(value)


def _pick_dict(payload: Any, keys: list[str]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {key: _json_safe(payload.get(key)) for key in keys if key in payload}


def _first(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    if isinstance(payload, dict):
        return payload
    return {}


def _source_summary(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts = Counter(str(item.get("source") or "unknown") for item in evidence)
    category_counts = Counter(str(item.get("category") or "unknown") for item in evidence)
    dated = sum(1 for item in evidence if item.get("as_of") or item.get("freshness"))
    linked = sum(1 for item in evidence if item.get("url"))
    return {
        "evidence_count": len(evidence),
        "source_counts": dict(source_counts.most_common()),
        "category_counts": dict(category_counts.most_common()),
        "dated_or_freshness_labeled": dated,
        "url_labeled": linked,
    }


def _market_snapshot(market_overview: dict[str, Any] | None) -> dict[str, Any]:
    market_overview = market_overview or {}
    return {
        "fear_greed": _json_safe(market_overview.get("fear_greed")),
        "sp500": _pick_dict(market_overview.get("sp500"), ["symbol", "price", "changesPercentage"]),
        "nasdaq": _pick_dict(market_overview.get("nasdaq"), ["symbol", "price", "changesPercentage"]),
        "dow": _pick_dict(market_overview.get("dow"), ["symbol", "price", "changesPercentage"]),
        "vix": _json_safe(market_overview.get("vix")),
    }


def _stock_snapshot(stock_data: dict[str, Any]) -> dict[str, Any]:
    quote = stock_data.get("quote") or stock_data.get("yf_quote") or {}
    massive_quote = stock_data.get("massive_quote") or {}
    profile = stock_data.get("profile") or {}
    technicals = stock_data.get("technicals") or {}
    earnings_context = stock_data.get("earnings_context") or {}
    recommendation_trends = stock_data.get("recommendation_trends") or {}
    analyst_estimates = stock_data.get("analyst_estimates") or {}
    short_interest = stock_data.get("short_interest") or {}
    price_target = stock_data.get("price_target_consensus") or stock_data.get("price_target") or {}
    sec_companyfacts = stock_data.get("sec_companyfacts") or {}
    sec_filings = stock_data.get("sec_filings") or {}

    latest_income = _first(stock_data.get("income_statement"))
    latest_balance = _first(stock_data.get("balance_sheet"))
    latest_cashflow = _first(stock_data.get("cash_flow") or stock_data.get("cash_flow_statement"))

    return {
        "ticker": _json_safe(stock_data.get("ticker")),
        "quote": _pick_dict(
            quote,
            [
                "symbol",
                "price",
                "previousClose",
                "changesPercentage",
                "marketCap",
                "volume",
                "avgVolume",
                "pe",
                "eps",
            ],
        ),
        "massive_quote": _pick_dict(
            massive_quote,
            [
                "symbol",
                "price",
                "previous_close",
                "changesPercentage",
                "volume",
                "bid",
                "ask",
                "source",
            ],
        ),
        "price_history_source": _json_safe(stock_data.get("price_history_source")),
        "history_provider_checks": _json_safe(stock_data.get("history_provider_checks") or {}),
        "profile": _pick_dict(
            profile,
            ["companyName", "sector", "industry", "country", "exchange", "mktCap", "beta"],
        ),
        "latest_income_statement": _pick_dict(
            latest_income,
            ["date", "calendarYear", "period", "revenue", "grossProfit", "operatingIncome", "netIncome", "eps"],
        ),
        "latest_balance_sheet": _pick_dict(
            latest_balance,
            ["date", "calendarYear", "period", "cashAndCashEquivalents", "totalDebt", "totalAssets", "totalLiabilities"],
        ),
        "latest_cash_flow": _pick_dict(
            latest_cashflow,
            [
                "date",
                "calendarYear",
                "period",
                "operatingCashFlow",
                "netCashProvidedByOperatingActivities",
                "capitalExpenditure",
                "freeCashFlow",
            ],
        ),
        "technicals": _json_safe(technicals),
        "earnings_context": _json_safe(earnings_context),
        "recommendation_trends": _json_safe(recommendation_trends),
        "analyst_estimates": _json_safe(analyst_estimates),
        "short_interest": _json_safe(short_interest),
        "price_target_consensus": _json_safe(price_target),
        "valuation_expectations": _json_safe(stock_data.get("valuation_expectations") or {}),
        "portfolio_factor_risk": _json_safe(stock_data.get("portfolio_factor_risk") or {}),
        "calibration_meta_signal": _json_safe(stock_data.get("calibration_meta_signal") or {}),
        "sec": {
            "companyfacts_available": bool(sec_companyfacts),
            "filings_available": bool(sec_filings),
        },
        "data_quality_report": _json_safe(stock_data.get("data_quality_report") or {}),
    }


def _analyst_snapshot(report: Any) -> dict[str, Any]:
    return {
        "analyst_name": _json_safe(getattr(report, "analyst_name", "")),
        "model": _json_safe(getattr(report, "model", "")),
        "verdict": _json_safe(getattr(report, "verdict", "")),
        "confidence": _json_safe(getattr(report, "confidence", None)),
        "report": _json_safe(getattr(report, "report", "")),
    }


def extract_decision_feature_row(dossier: dict[str, Any], dossier_path: str) -> dict[str, Any]:
    """Build compact SQLite feature row from a dossier payload."""
    decision = dossier.get("decision") or {}
    stock = dossier.get("stock_packet") or {}
    quote = stock.get("quote") or {}
    profile = stock.get("profile") or {}
    source_audit = dossier.get("source_audit") or {}
    dq = stock.get("data_quality_report") or {}
    agentic = dossier.get("agentic_trace") or {}
    source_counts = source_audit.get("source_counts") or {}
    valuation = stock.get("valuation_expectations") or {}
    portfolio_risk = stock.get("portfolio_factor_risk") or {}
    analyst_targets = valuation.get("analyst_targets") if isinstance(valuation, dict) else {}

    feature_json = {
        "score_components": decision.get("score_components") or {},
        "scoring_audit": decision.get("scoring_audit") or {},
        "source_audit": source_audit,
        "gaps": agentic.get("gaps") or [],
        "conflicts": agentic.get("conflicts") or [],
        "market_snapshot": dossier.get("market_snapshot") or {},
        "valuation_expectations": valuation,
        "portfolio_factor_risk": portfolio_risk,
        "calibration_meta_signal": stock.get("calibration_meta_signal") or {},
        "analyst_verdicts": {
            key: (value or {}).get("verdict")
            for key, value in (dossier.get("analysts") or {}).items()
            if isinstance(value, dict)
        },
    }

    return {
        "dossier_path": str(dossier_path),
        "generated_at": str(dossier.get("generated_at") or ""),
        "ticker": str(dossier.get("ticker") or "").upper(),
        "final_verdict": str(decision.get("final_verdict") or ""),
        "opportunity_score": decision.get("opportunity_score"),
        "adjusted_score": decision.get("adjusted_score"),
        "confidence": decision.get("confidence"),
        "price": quote.get("price"),
        "market_cap": quote.get("marketCap") or profile.get("mktCap"),
        "sector": profile.get("sector"),
        "industry": profile.get("industry"),
        "evidence_count": source_audit.get("evidence_count"),
        "context_coverage_score": dq.get("context_coverage_score"),
        "completeness_score": dq.get("completeness_score"),
        "source_count": len(source_counts) if isinstance(source_counts, dict) else 0,
        "gap_count": len(agentic.get("gaps") or []),
        "valuation_signal": valuation.get("valuation_signal") if isinstance(valuation, dict) else None,
        "consensus_upside_pct": (
            analyst_targets.get("consensus_upside_pct")
            if isinstance(analyst_targets, dict)
            else None
        ),
        "expectation_risk_level": (
            valuation.get("expectation_risk_level") if isinstance(valuation, dict) else None
        ),
        "portfolio_risk_level": (
            portfolio_risk.get("risk_level") if isinstance(portfolio_risk, dict) else None
        ),
        "portfolio_sector_after_pct": (
            portfolio_risk.get("sector_after_candidate_pct") if isinstance(portfolio_risk, dict) else None
        ),
        "benchmark_ticker": (
            portfolio_risk.get("sector_benchmark_ticker") if isinstance(portfolio_risk, dict) else None
        ),
        "feature_json": json.dumps(_json_safe(feature_json), sort_keys=True),
    }


def write_decision_dossier(
    decision: Any,
    stock_data: dict[str, Any],
    macro_data: dict[str, Any] | None = None,
    market_overview: dict[str, Any] | None = None,
    intelligence_brief: str = "",
    pre_brief: str = "",
    momentum_context: str = "",
) -> str:
    """Write a point-in-time council dossier and return its absolute path."""
    generated = _utcnow()
    ticker = str(getattr(decision, "ticker", stock_data.get("ticker", "UNKNOWN")) or "UNKNOWN").upper()
    trace = getattr(decision, "agentic_trace", {}) or {}
    evidence = trace.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = []
    evidence = [item for item in evidence if isinstance(item, dict)]

    dossier = {
        "schema_version": 1,
        "ticker": ticker,
        "generated_at": generated.isoformat(),
        "decision": {
            "final_verdict": _json_safe(getattr(decision, "final_verdict", "")),
            "consensus": _json_safe(getattr(decision, "consensus", "")),
            "recommended_action": _json_safe(getattr(decision, "recommended_action", "")),
            "allocation": _json_safe(getattr(decision, "allocation", "")),
            "opportunity_score": _json_safe(getattr(decision, "opportunity_score", None)),
            "adjusted_score": _json_safe(getattr(decision, "adjusted_score", None)),
            "score_components": _json_safe(getattr(decision, "score_components", {}) or {}),
            "base_opportunity_score": _json_safe(getattr(decision, "base_opportunity_score", None)),
            "rule_adjustment_total": _json_safe(getattr(decision, "rule_adjustment_total", None)),
            "cio_adjustment": _json_safe(getattr(decision, "cio_adjustment", None)),
            "scoring_audit": _json_safe(getattr(decision, "scoring_audit", {}) or {}),
            "confidence": _json_safe(getattr(decision, "confidence", None)),
            "thesis_type": _json_safe(getattr(decision, "thesis_type", "")),
            "recommended_allocation_pct": _json_safe(getattr(decision, "recommended_allocation_pct", None)),
            "entry_valid_until": _json_safe(getattr(decision, "entry_valid_until", "")),
            "invalidation_conditions": _json_safe(getattr(decision, "invalidation_conditions", []) or []),
            "stop_loss_pct": _json_safe(getattr(decision, "stop_loss_pct", None)),
            "target_pct": _json_safe(getattr(decision, "target_pct", None)),
            "hard_risk_gate_passed": _json_safe(getattr(decision, "hard_risk_gate_passed", True)),
            "hard_risk_gate_reason": _json_safe(getattr(decision, "hard_risk_gate_reason", "")),
        },
        "source_audit": _source_summary(evidence),
        "agentic_trace": {
            "enabled": _json_safe(trace.get("enabled")),
            "generated_at": _json_safe(trace.get("generated_at")),
            "trace_path": _json_safe(trace.get("trace_path")),
            "gaps": _json_safe(trace.get("gaps") or []),
            "conflicts": _json_safe(trace.get("conflicts") or []),
            "role_plans": _json_safe(trace.get("role_plans") or {}),
            "role_queries": _json_safe(trace.get("role_queries") or {}),
            "cio_brief": _json_safe(trace.get("cio_brief") or ""),
            "evidence": _json_safe(evidence),
        },
        "analysts": {
            "fundamental": _analyst_snapshot(getattr(decision, "fundamental", None)),
            "technical": _analyst_snapshot(getattr(decision, "technical", None)),
            "contrarian": _analyst_snapshot(getattr(decision, "contrarian", None)),
        },
        "synthesis_report": _json_safe(getattr(decision, "synthesis_report", "")),
        "stock_packet": _stock_snapshot(stock_data),
        "market_snapshot": _market_snapshot(market_overview),
        "macro_data_keys": sorted(str(k) for k in (macro_data or {}).keys()),
        "context": {
            "intelligence_brief": _json_safe(intelligence_brief),
            "pre_brief": _json_safe(pre_brief),
            "momentum_context": _json_safe(momentum_context),
        },
    }

    out_dir = DOSSIER_DIR / generated.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}_{generated.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(dossier, indent=2, sort_keys=True), encoding="utf-8")
    try:
        from .journal import DecisionJournal
        DecisionJournal().save_decision_features(extract_decision_feature_row(dossier, str(out_path)))
    except Exception as exc:
        logger.warning("Decision feature warehouse write failed for %s: %s", ticker, exc)
    logger.info("Decision dossier written for %s: %s", ticker, out_path)
    return str(out_path)
