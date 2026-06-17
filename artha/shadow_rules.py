"""Private shadow-rule testing for Artha.

Shadow rules are proposed investing-rule changes tested in the background.
They record what a different rule *would* have recommended, then measure the
outcome later. They never change the live council decision.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .journal import DecisionJournal
from .portfolio_risk import primary_market_benchmark_for, sector_benchmark_for

logger = logging.getLogger(__name__)

RULE_VERSION = "v1"

BUY_LIKE = {"BUY", "STARTER", "TACTICAL_BUY", "ACCUMULATE", "ADD", "STRONG BUY"}
NO_BUY = {"WATCH", "DEFER", "AVOID", "HOLD", "SELL", "TRIM", "STRONG SELL"}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _payload_from_decision(decision: Any, stock_data: dict[str, Any]) -> dict[str, Any]:
    valuation = stock_data.get("valuation_expectations") or getattr(decision, "valuation_expectations", {}) or {}
    risk = stock_data.get("portfolio_factor_risk") or getattr(decision, "portfolio_factor_risk", {}) or {}
    targets = valuation.get("analyst_targets") if isinstance(valuation, dict) else {}
    revision = valuation.get("revision_trend") if isinstance(valuation, dict) else {}
    timing = valuation.get("timing_risk") if isinstance(valuation, dict) else {}
    quote = stock_data.get("quote") or stock_data.get("yf_quote") or {}
    dossier_path = str(getattr(decision, "dossier_path", "") or "")
    generated_at = (
        getattr(decision, "generated_at", None)
        or getattr(decision, "decision_generated_at", None)
        or ""
    )
    if not generated_at and dossier_path:
        try:
            dossier_payload = json.loads(Path(dossier_path).read_text(encoding="utf-8"))
            generated_at = str(dossier_payload.get("generated_at") or "")
        except Exception:
            generated_at = ""
    if not generated_at and dossier_path:
        try:
            generated_at = datetime.fromtimestamp(
                Path(dossier_path).stat().st_mtime,
                tz=timezone.utc,
            ).isoformat()
        except Exception:
            generated_at = _utcnow_iso()
    if not generated_at:
        generated_at = _utcnow_iso()
    return {
        "ticker": str(getattr(decision, "ticker", stock_data.get("ticker", "")) or "").upper(),
        "dossier_path": dossier_path,
        "generated_at": str(generated_at),
        "real_action": str(getattr(decision, "final_verdict", "") or "").upper(),
        "opportunity_score": _num(getattr(decision, "opportunity_score", None), 0.0) or 0.0,
        "price": _num(quote.get("price"), None),
        "valuation_signal": str(valuation.get("valuation_signal") or "").lower(),
        "expectation_risk_level": str(valuation.get("expectation_risk_level") or "").lower(),
        "consensus_upside_pct": _num((targets or {}).get("consensus_upside_pct"), None),
        "net_revision_30d": _num((revision or {}).get("net_revision_30d"), 0.0) or 0.0,
        "rsi": _num((timing or {}).get("rsi"), None),
        "price_vs_sma50_pct": _num((timing or {}).get("price_vs_sma50_pct"), None),
        "portfolio_risk_level": str(risk.get("risk_level") or "").lower(),
        "sector": str(risk.get("sector") or valuation.get("sector") or ""),
        "market_benchmark_ticker": str(risk.get("market_benchmark_ticker") or ""),
        "sector_benchmark_ticker": str(risk.get("sector_benchmark_ticker") or ""),
    }


def _payload_from_feature_row(row: dict[str, Any]) -> dict[str, Any]:
    feature_json = {}
    try:
        feature_json = json.loads(row.get("feature_json") or "{}")
    except Exception:
        feature_json = {}
    valuation = feature_json.get("valuation_expectations") or {}
    risk = feature_json.get("portfolio_factor_risk") or {}
    revision = valuation.get("revision_trend") or {}
    timing = valuation.get("timing_risk") or {}
    return {
        "ticker": str(row.get("ticker") or "").upper(),
        "dossier_path": str(row.get("dossier_path") or ""),
        "generated_at": str(row.get("generated_at") or _utcnow_iso()),
        "real_action": str(row.get("final_verdict") or "").upper(),
        "opportunity_score": _num(row.get("adjusted_score"), _num(row.get("opportunity_score"), 0.0)) or 0.0,
        "price": _num(row.get("price"), None),
        "valuation_signal": str(row.get("valuation_signal") or valuation.get("valuation_signal") or "").lower(),
        "expectation_risk_level": str(
            row.get("expectation_risk_level") or valuation.get("expectation_risk_level") or ""
        ).lower(),
        "consensus_upside_pct": _num(row.get("consensus_upside_pct"), None),
        "net_revision_30d": _num(revision.get("net_revision_30d"), 0.0) or 0.0,
        "rsi": _num(timing.get("rsi"), None),
        "price_vs_sma50_pct": _num(timing.get("price_vs_sma50_pct"), None),
        "portfolio_risk_level": str(row.get("portfolio_risk_level") or risk.get("risk_level") or "").lower(),
        "sector": str(row.get("sector") or risk.get("sector") or ""),
        "market_benchmark_ticker": str(risk.get("market_benchmark_ticker") or ""),
        "sector_benchmark_ticker": str(row.get("benchmark_ticker") or risk.get("sector_benchmark_ticker") or ""),
    }


def _evaluation_id(rule_id: str, dossier_path: str, ticker: str, generated_at: str) -> str:
    raw = f"{RULE_VERSION}|{rule_id}|{dossier_path}|{ticker}|{generated_at}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _candidate_shadow_rules(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return shadow-rule evaluations triggered by this decision payload."""
    action = str(payload.get("real_action") or "").upper()
    score = _num(payload.get("opportunity_score"), 0.0) or 0.0
    consensus_upside = _num(payload.get("consensus_upside_pct"), None)
    net_revision = _num(payload.get("net_revision_30d"), 0.0) or 0.0
    expectation = str(payload.get("expectation_risk_level") or "")
    portfolio_risk = str(payload.get("portfolio_risk_level") or "")
    rsi = _num(payload.get("rsi"), None)
    price_vs_sma50 = _num(payload.get("price_vs_sma50_pct"), None)
    rules: list[dict[str, Any]] = []

    if (
        action in {"DEFER", "WATCH"}
        and 45 <= score < 65
        and expectation != "high"
        and portfolio_risk != "high"
        and ((consensus_upside is not None and consensus_upside >= 10) or net_revision > 0)
    ):
        rules.append(
            {
                "rule_id": "low_score_defer_starter_probe",
                "shadow_action": "STARTER",
                "trigger_reason": (
                    "Practice test: if a DEFER/WATCH has a middling score but positive upside/revision evidence, "
                    "would a tiny starter have worked better?"
                ),
            }
        )

    if action in {"DEFER", "WATCH"} and score >= 65 and portfolio_risk != "high":
        rules.append(
            {
                "rule_id": "high_score_defer_starter_probe",
                "shadow_action": "STARTER",
                "trigger_reason": (
                    "Practice test: high-score DEFER/WATCH may be excessive caution; test starter path privately."
                ),
            }
        )

    overextended = (rsi is not None and rsi >= 70) or (price_vs_sma50 is not None and price_vs_sma50 >= 20)
    low_upside = consensus_upside is not None and consensus_upside < 5
    if action in BUY_LIKE and (expectation == "high" or low_upside) and overextended:
        rules.append(
            {
                "rule_id": "overextended_low_upside_defer_guard",
                "shadow_action": "DEFER",
                "trigger_reason": (
                    "Practice test: if Artha buys an overextended low-upside setup, would DEFER have avoided risk?"
                ),
            }
        )

    if action in {"DEFER", "WATCH"} and expectation == "high" and low_upside:
        rules.append(
            {
                "rule_id": "high_expectation_risk_avoid_probe",
                "shadow_action": "AVOID",
                "trigger_reason": (
                    "Practice test: high expectation risk plus low upside may deserve AVOID instead of DEFER/WATCH."
                ),
            }
        )

    return rules


def _to_evaluation(rule: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any] | None:
    ticker = str(payload.get("ticker") or "").upper()
    price = _num(payload.get("price"), None)
    if not ticker or price is None or price <= 0:
        return None
    sector = str(payload.get("sector") or "")
    benchmark = str(payload.get("market_benchmark_ticker") or primary_market_benchmark_for(sector)).upper()
    sector_benchmark = str(payload.get("sector_benchmark_ticker") or sector_benchmark_for(sector, fallback=benchmark)).upper()
    generated_at = str(payload.get("generated_at") or _utcnow_iso())
    dossier_path = str(payload.get("dossier_path") or "")
    return {
        "evaluation_id": _evaluation_id(rule["rule_id"], dossier_path, ticker, generated_at),
        "rule_id": rule["rule_id"],
        "rule_version": RULE_VERSION,
        "ticker": ticker,
        "dossier_path": dossier_path,
        "decision_generated_at": generated_at,
        "real_action": str(payload.get("real_action") or "").upper(),
        "shadow_action": rule["shadow_action"],
        "rule_status": "shadow_mode",
        "trigger_reason": rule["trigger_reason"],
        "evidence_json": payload,
        "hypothetical_entry": price,
        "benchmark_ticker": benchmark,
        "sector_benchmark_ticker": sector_benchmark,
        "status": "tracking",
        "created_at": _utcnow_iso(),
        "updated_at": _utcnow_iso(),
    }


def evaluate_shadow_rules_for_decision(
    decision: Any,
    stock_data: dict[str, Any],
    journal: DecisionJournal | None = None,
) -> list[dict[str, Any]]:
    """Evaluate private shadow rules for one live council decision."""
    journal = journal or DecisionJournal()
    payload = _payload_from_decision(decision, stock_data)
    inserted: list[dict[str, Any]] = []
    for rule in _candidate_shadow_rules(payload):
        evaluation = _to_evaluation(rule, payload)
        if evaluation and journal.save_shadow_rule_evaluation(evaluation):
            inserted.append(evaluation)
    if inserted:
        logger.info("[shadow_rules] Logged %d shadow rule evaluation(s) for %s", len(inserted), payload.get("ticker"))
    return inserted


def backfill_shadow_rules_from_features(
    journal: DecisionJournal | None = None,
    limit: int = 250,
) -> int:
    """Backfill shadow-rule evaluations from point-in-time decision features."""
    journal = journal or DecisionJournal()
    count = 0
    for row in journal.get_decision_features(limit=limit):
        payload = _payload_from_feature_row(row)
        for rule in _candidate_shadow_rules(payload):
            evaluation = _to_evaluation(rule, payload)
            if evaluation and journal.save_shadow_rule_evaluation(evaluation):
                count += 1
    return count


def _ensure_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_index_ts(idx):
    ts = idx
    if hasattr(ts, "tzinfo") and ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    elif hasattr(ts, "tz_convert"):
        ts = ts.tz_convert(timezone.utc)
    return ts


def _filter_from_decision(df, created_dt):
    if df is None or getattr(df, "empty", True):
        return df
    keep = [_normalize_index_ts(idx) >= created_dt for idx in df.index]
    return df.loc[keep]


def _price_on_or_after(closes, target_dt):
    for idx in closes.index:
        if _normalize_index_ts(idx) >= target_dt:
            return float(closes[idx])
    return None


def update_shadow_rule_outcomes(journal: DecisionJournal | None = None) -> dict[str, int]:
    """Update 5/10/20/60-day outcomes for shadow-rule evaluations."""
    journal = journal or DecisionJournal()
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("[shadow_rules] yfinance unavailable; skipping outcome update")
        return {"updated": 0, "errors": 0, "skipped": 0}

    rows = journal.get_pending_shadow_rule_evaluations()
    if not rows:
        return {"updated": 0, "errors": 0, "skipped": 0}

    now = datetime.now(timezone.utc)
    history_cache: dict[str, Any] = {}

    def history(symbol: str, age_days: int):
        symbol = str(symbol or "").upper().strip()
        if not symbol:
            return None
        period = "1y" if age_days > 120 else "6mo"
        key = f"{symbol}:{period}"
        if key not in history_cache:
            history_cache[key] = yf.Ticker(symbol).history(period=period)
        return history_cache[key]

    updated = 0
    errors = 0
    skipped = 0
    for row in rows:
        created_dt = _ensure_dt(row.get("decision_generated_at") or row.get("created_at"))
        ticker = str(row.get("ticker") or "").upper()
        entry = _num(row.get("hypothetical_entry"), None)
        if not created_dt or not ticker or entry is None or entry <= 0:
            skipped += 1
            continue
        age_days = (now - created_dt).days
        needed = {
            5: age_days >= 5 and row.get("price_5d") is None,
            10: age_days >= 10 and row.get("price_10d") is None,
            20: age_days >= 20 and row.get("price_20d") is None,
            60: age_days >= 60 and row.get("price_60d") is None,
        }
        if not any(needed.values()):
            skipped += 1
            continue
        try:
            hist = _filter_from_decision(history(ticker, age_days), created_dt)
            if hist is None or hist.empty:
                skipped += 1
                continue
            closes = hist["Close"]
            highs = hist["High"]
            lows = hist["Low"]
            update: dict[str, Any] = {}
            for days, should_update in needed.items():
                if should_update:
                    price = _price_on_or_after(closes, created_dt + timedelta(days=days))
                    if price:
                        update[f"price_{days}d"] = price
            update["mfe"] = float((highs.max() - entry) / entry)
            update["mae"] = float((lows.min() - entry) / entry)
            update["would_hit_stop"] = update["mae"] <= -0.08

            for prefix, symbol in (
                ("benchmark", row.get("benchmark_ticker")),
                ("sector_benchmark", row.get("sector_benchmark_ticker") or row.get("benchmark_ticker")),
            ):
                bench = _filter_from_decision(history(str(symbol or ""), age_days), created_dt)
                if bench is None or bench.empty:
                    continue
                bench_closes = bench["Close"]
                if row.get(f"{prefix}_price_entry") is None:
                    entry_price = _price_on_or_after(bench_closes, created_dt)
                    if entry_price:
                        update[f"{prefix}_price_entry"] = entry_price
                for days, should_update in needed.items():
                    if should_update:
                        price = _price_on_or_after(bench_closes, created_dt + timedelta(days=days))
                        if price:
                            update[f"{prefix}_price_{days}d"] = price

            if update:
                journal.update_shadow_rule_evaluation(str(row["evaluation_id"]), update)
                updated += 1
        except Exception as exc:
            logger.warning("[shadow_rules] Outcome update failed for %s/%s: %s", row.get("rule_id"), ticker, exc)
            errors += 1
    return {"updated": updated, "errors": errors, "skipped": skipped}


def summarize_shadow_rules(journal: DecisionJournal | None = None) -> dict[str, Any]:
    """Summarize shadow-rule practice-field status."""
    journal = journal or DecisionJournal()
    rows = journal.get_shadow_rule_evaluations(limit=1000)
    by_rule: dict[str, dict[str, Any]] = {}
    for row in rows:
        rule_id = str(row.get("rule_id") or "unknown")
        bucket = by_rule.setdefault(
            rule_id,
            {
                "count": 0,
                "completed": 0,
                "tracking": 0,
                "avg_excess_return_10d": None,
                "avg_excess_return_20d": None,
                "avg_excess_return_60d": None,
            },
        )
        bucket["count"] += 1
        if row.get("status") == "completed":
            bucket["completed"] += 1
        else:
            bucket["tracking"] += 1

    for rule_id, bucket in by_rule.items():
        rule_rows = [r for r in rows if str(r.get("rule_id") or "unknown") == rule_id]
        for key in ("excess_return_10d", "excess_return_20d", "excess_return_60d"):
            vals = [_num(r.get(key), None) for r in rule_rows if r.get(key) is not None]
            vals = [v for v in vals if v is not None]
            bucket[f"avg_{key}"] = round(sum(vals) / len(vals), 4) if vals else None
    return {
        "total": len(rows),
        "completed": sum(1 for r in rows if r.get("status") == "completed"),
        "tracking": sum(1 for r in rows if r.get("status") != "completed"),
        "rules": by_rule,
    }
