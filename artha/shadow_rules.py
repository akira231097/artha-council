"""Private shadow-rule testing for Artha.

Shadow rules are proposed investing-rule changes tested in the background.
They record what a different rule *would* have recommended, then measure the
outcome later. They never change the live council decision.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .journal import DecisionJournal
from .config import Config
from .paths import DATA_DIR
from .portfolio_risk import primary_market_benchmark_for, sector_benchmark_for

logger = logging.getLogger(__name__)

RULE_VERSION = "v1"

BUY_LIKE = {"BUY", "STARTER", "TACTICAL_BUY", "ACCUMULATE", "ADD", "STRONG BUY"}
EXPOSURE_LIKE_SHADOW_ACTIONS = BUY_LIKE | {"HOLD_SHARES"}
NO_BUY = {"WATCH", "DEFER", "AVOID", "HOLD", "SELL", "TRIM", "STRONG SELL"}

PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
SELL_RULES_FILE = DATA_DIR / "shadow_sell_rules.json"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Graduation tunables (Wave 2 consolidates into Config). A rule graduates to
# 'candidate_promote' — surfaced in the nightly review, never auto-applied —
# when it has enough evaluations and beats its baseline often enough.
SHADOW_RULE_PROMOTE_MIN_N = int(_env_float("ARTHA_SHADOW_RULE_PROMOTE_MIN_N", 20))
SHADOW_RULE_PROMOTE_WIN_RATE = _env_float("ARTHA_SHADOW_RULE_PROMOTE_WIN_RATE", 0.65)
SHADOW_RULE_INDEPENDENCE_DAYS = int(
    _env_float("ARTHA_SHADOW_RULE_INDEPENDENCE_DAYS", 60)
)
SHADOW_RULE_PROMOTE_MIN_HORIZON_DAYS = int(
    _env_float("ARTHA_SHADOW_RULE_PROMOTE_MIN_HORIZON_DAYS", 20)
)
SELL_RULE_HORIZON_DAYS = int(_env_float("ARTHA_SELL_RULE_HORIZON_DAYS", 60))
PROMOTION_SHADOW_MIN_HOLD_DAYS = int(
    _env_float("ARTHA_PROMOTION_SHADOW_MIN_HOLD_DAYS", 20)
)
PROMOTION_SHADOW_MIN_GAIN_PCT = _env_float(
    "ARTHA_PROMOTION_SHADOW_MIN_GAIN_PCT",
    5.0,
)
PROMOTION_SHADOW_MIN_SCORE = _env_float("ARTHA_PROMOTION_SHADOW_MIN_SCORE", 65.0)
PROMOTION_SHADOW_MIN_NOTIONAL = _env_float(
    "ARTHA_PROMOTION_SHADOW_MIN_NOTIONAL",
    10.0,
)
PROMOTION_SHADOW_MAX_NOTIONAL = _env_float(
    "ARTHA_PROMOTION_SHADOW_MAX_NOTIONAL",
    25.0,
)

# Sell-side shadow rules simulated against real position price paths.
# 'all_or_nothing' (hold through the horizon) is the baseline the others are
# scored against — it mirrors live behavior, where sells almost never fire.
SELL_SHADOW_RULE_IDS = (
    "stop_width_8pct",
    "stop_width_15pct",
    "chandelier_3atr",
    "scale_out_half_at_20pct",
    "all_or_nothing",
)
SELL_RULE_BASELINE = "all_or_nothing"


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
    alpha_shadow = stock_data.get("alpha_shadow_signals") or {}
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
        "alpha_shadow_signals": alpha_shadow if isinstance(alpha_shadow, dict) else {},
    }


def _payload_from_feature_row(row: dict[str, Any]) -> dict[str, Any]:
    feature_json = {}
    try:
        feature_json = json.loads(row.get("feature_json") or "{}")
    except Exception:
        feature_json = {}
    valuation = feature_json.get("valuation_expectations") or {}
    risk = feature_json.get("portfolio_factor_risk") or {}
    alpha_shadow = feature_json.get("alpha_shadow_signals") or {}
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
        "alpha_shadow_signals": alpha_shadow if isinstance(alpha_shadow, dict) else {},
    }


def _evaluation_id(rule_id: str, dossier_path: str, ticker: str, generated_at: str) -> str:
    raw = f"{RULE_VERSION}|{rule_id}|{dossier_path}|{ticker}|{generated_at}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _stable_shadow_id(rule_id: str, identity: str) -> str:
    raw = f"{RULE_VERSION}|{rule_id}|{identity}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def record_trim_hold_shadow(
    journal: DecisionJournal,
    event: dict[str, Any],
) -> bool:
    """Track whether holding broker-confirmed trimmed shares was superior.

    This is a counterfactual learning row only. It never changes the completed
    trim, a sell rule, a Council verdict, or an execution action.
    """
    if str(event.get("event_type") or "").upper() != "TRIM":
        return False
    event_id = str(event.get("event_id") or "")
    ticker = str(event.get("ticker") or "").upper().strip()
    price = _num(event.get("average_price"), None)
    observed_at = str(event.get("filled_at") or _utcnow_iso())
    if not event_id or not ticker or price is None or price <= 0:
        return False
    feature = journal.get_latest_decision_feature(
        ticker,
        generated_before=observed_at,
    ) or {}
    sector = str(feature.get("sector") or "")
    return journal.save_shadow_rule_evaluation(
        {
            "evaluation_id": _stable_shadow_id("trim_hold_counterfactual", event_id),
            "rule_id": "trim_hold_counterfactual",
            "rule_version": RULE_VERSION,
            "ticker": ticker,
            "dossier_path": str(feature.get("dossier_path") or ""),
            "decision_generated_at": observed_at,
            "real_action": "TRIM",
            "shadow_action": "HOLD_SHARES",
            "rule_status": "shadow_mode",
            "trigger_reason": (
                "Practice test: after a real trim, would retaining the trimmed shares "
                "have outperformed the appropriate benchmark?"
            ),
            "evidence_json": {
                "sell_event_id": event_id,
                "thesis_id": event.get("thesis_id"),
                "trimmed_shares": event.get("shares"),
                "trim_reason": event.get("sell_reason"),
                "data_quality": event.get("data_quality"),
            },
            "hypothetical_entry": price,
            "benchmark_ticker": primary_market_benchmark_for(sector),
            "sector_benchmark_ticker": sector_benchmark_for(sector),
            "status": "tracking",
        }
    )


def backfill_trim_hold_shadows(journal: DecisionJournal | None = None) -> dict[str, Any]:
    """Idempotently seed trim-opportunity learning from durable sell events."""
    journal = journal or DecisionJournal()
    trims = [
        event
        for event in journal.get_sell_trade_events(limit=5000)
        if str(event.get("event_type") or "").upper() == "TRIM"
    ]
    inserted = sum(1 for event in trims if record_trim_hold_shadow(journal, event))
    return {"trim_events": len(trims), "inserted": inserted}


def record_position_promotion_shadows(
    journal: DecisionJournal | None = None,
    *,
    portfolio_path: str | Path = PORTFOLIO_FILE,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Create one private ADD experiment for each independently proven winner.

    Qualification is intentionally stricter than a normal starter: the holding
    must be seasoned, profitable, protected above cost by its trailing level,
    supported by a buy-like scored decision, below its dollar cap, and inside
    the projected sector cap. This function records no trade action.
    """
    from .portfolio import Portfolio
    from .portfolio_risk import evaluate_projected_sector_limit
    from .portfolio_state import PortfolioStateEngine
    from .thesis_tracker import ThesisTracker

    journal = journal or DecisionJournal()
    observed_at = now or datetime.now(timezone.utc)
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)
    observed_at = observed_at.astimezone(timezone.utc)
    portfolio = Portfolio.load(Path(portfolio_path))
    portfolio_state = PortfolioStateEngine(Path(portfolio_path)).compute_state()
    tracker = ThesisTracker(journal)
    eligible: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    inserted = 0

    for position in portfolio.positions:
        ticker = str(position.ticker or "").upper().strip()
        reasons: list[str] = []
        thesis = tracker.get_sell_monitored(ticker)
        thesis_id = str(getattr(thesis, "thesis_id", "") or position.thesis_id or "")
        opened = _ensure_dt(position.opened_at or position.entry_date)
        held_days = (observed_at - opened).days if opened else -1
        current_price = _num(position.current_price, None)
        if (current_price is None or current_price <= 0) and position.shares:
            current_price = _num(position.market_value, 0.0) / float(position.shares)
        avg_cost = _num(position.avg_cost, 0.0) or 0.0
        gain_pct = (
            (current_price / avg_cost - 1.0) * 100.0
            if current_price is not None and avg_cost > 0
            else None
        )
        trailing_stop = _num(position.trailing_stop_price, None)
        feature = journal.get_latest_decision_feature(ticker) or {}
        score = _num(
            feature.get("adjusted_score"),
            _num(feature.get("opportunity_score"), 0.0),
        ) or 0.0
        verdict = str(feature.get("final_verdict") or "").upper()
        market_value = _num(position.market_value, None)
        if market_value is None and current_price is not None:
            market_value = float(position.shares or 0.0) * current_price
        market_value = market_value or 0.0
        headroom = max(0.0, float(Config.ROBINHOOD_MAX_POSITION_DOLLARS) - market_value)
        proposed_notional = min(PROMOTION_SHADOW_MAX_NOTIONAL, headroom)
        sector = str(position.sector or feature.get("sector") or "")

        if not thesis_id or thesis is None or str(getattr(thesis, "status", "")) != "active":
            reasons.append("no_active_thesis")
        if held_days < PROMOTION_SHADOW_MIN_HOLD_DAYS:
            reasons.append("insufficient_holding_period")
        if gain_pct is None or gain_pct < PROMOTION_SHADOW_MIN_GAIN_PCT:
            reasons.append("gain_not_proven")
        if trailing_stop is None or trailing_stop < avg_cost:
            reasons.append("profit_not_protected_above_cost")
        if score < PROMOTION_SHADOW_MIN_SCORE or verdict not in BUY_LIKE:
            reasons.append("decision_not_buy_like_or_score_too_low")
        if proposed_notional < PROMOTION_SHADOW_MIN_NOTIONAL:
            reasons.append("insufficient_position_headroom")

        sector_check = evaluate_projected_sector_limit(
            ticker=ticker,
            sector=sector,
            portfolio_state=portfolio_state,
            proposed_notional=proposed_notional,
            max_sector_pct=float(Config.MAX_SECTOR_PCT),
        )
        if not sector_check.get("passed"):
            reasons.append("projected_sector_limit")
        if current_price is None or current_price <= 0:
            reasons.append("missing_current_price")

        if reasons:
            skipped.append({"ticker": ticker, "reasons": sorted(set(reasons))})
            continue

        evaluation = {
            "evaluation_id": _stable_shadow_id("proven_winner_add_probe", thesis_id),
            "rule_id": "proven_winner_add_probe",
            "rule_version": RULE_VERSION,
            "ticker": ticker,
            "dossier_path": str(feature.get("dossier_path") or ""),
            "decision_generated_at": observed_at.isoformat(),
            "real_action": "HOLD",
            "shadow_action": "ADD",
            "rule_status": "shadow_mode",
            "trigger_reason": (
                "Practice test: would one capped second tranche in a seasoned, "
                "profitable, thesis-supported winner improve benchmark-relative return?"
            ),
            "evidence_json": {
                "thesis_id": thesis_id,
                "held_days": held_days,
                "gain_pct": round(gain_pct or 0.0, 4),
                "trailing_stop": trailing_stop,
                "avg_cost": avg_cost,
                "decision_score": score,
                "decision_verdict": verdict,
                "hypothetical_notional": round(proposed_notional, 2),
                "sector_check": sector_check,
                "live_order_created": False,
            },
            "hypothetical_entry": current_price,
            "benchmark_ticker": primary_market_benchmark_for(sector),
            "sector_benchmark_ticker": sector_benchmark_for(sector),
            "status": "tracking",
        }
        was_inserted = journal.save_shadow_rule_evaluation(evaluation)
        inserted += int(was_inserted)
        eligible.append(
            {
                "ticker": ticker,
                "inserted": was_inserted,
                "gain_pct": round(gain_pct or 0.0, 2),
                "held_days": held_days,
                "score": score,
                "hypothetical_notional": round(proposed_notional, 2),
            }
        )

    return {
        "positions_seen": len(portfolio.positions),
        "eligible": eligible,
        "inserted": inserted,
        "skipped": skipped,
        "live_actions_created": 0,
    }


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
    alpha_shadow = payload.get("alpha_shadow_signals") if isinstance(payload.get("alpha_shadow_signals"), dict) else {}

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

    if getattr(Config, "ALPHA_SHADOW_SIGNALS_ENABLED", True) and action in NO_BUY:
        insider = alpha_shadow.get("insider_cluster") if isinstance(alpha_shadow.get("insider_cluster"), dict) else {}
        industry = alpha_shadow.get("industry_momentum") if isinstance(alpha_shadow.get("industry_momentum"), dict) else {}
        episodic = alpha_shadow.get("episodic_pivot") if isinstance(alpha_shadow.get("episodic_pivot"), dict) else {}
        explorer = (
            alpha_shadow.get("quality_small_cap_explorer")
            if isinstance(alpha_shadow.get("quality_small_cap_explorer"), dict)
            else {}
        )
        if insider.get("cluster_confirmed"):
            rules.append(
                {
                    "rule_id": "insider_cluster_starter_probe",
                    "shadow_action": "STARTER",
                    "trigger_reason": "Practice test: three or more disclosed open-market insider buyers formed a 14-day cluster.",
                }
            )
        if (
            int(_num(industry.get("peer_count"), 0.0) or 0) >= 5
            and (_num(industry.get("z_score"), 0.0) or 0.0) >= 1.0
        ):
            rules.append(
                {
                    "rule_id": "industry_momentum_starter_probe",
                    "shadow_action": "STARTER",
                    "trigger_reason": "Practice test: the candidate belongs to a statistically strong industry group.",
                }
            )
        if (_num(episodic.get("quality_score"), 0.0) or 0.0) >= 70.0:
            rules.append(
                {
                    "rule_id": "episodic_pivot_starter_probe",
                    "shadow_action": "STARTER",
                    "trigger_reason": "Practice test: a high-quality earnings/news gap had strong volume confirmation.",
                }
            )
        if explorer.get("qualified"):
            rules.append(
                {
                    "rule_id": "quality_small_cap_explorer_probe",
                    "shadow_action": "STARTER",
                    "trigger_reason": "Practice test: a liquid, profitable, seasoned $500M-$2B company passed every explorer fence.",
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

    # Refresh sell-side shadow rules against real position price paths on the
    # same nightly cadence (supervisor calls this task).
    sell_rule_evals = 0
    try:
        sell_rule_evals = int(evaluate_sell_shadow_rules().get("updated") or 0)
    except Exception as exc:
        logger.warning("[shadow_rules] Sell-rule evaluation failed: %s", exc)

    return {"updated": updated, "errors": errors, "skipped": skipped, "sell_rule_evals": sell_rule_evals}


def _row_rule_advantage(
    row: dict[str, Any],
    shadow_is_buy_like: bool,
    *,
    minimum_horizon_days: int = 10,
) -> float | None:
    """Benchmark-relative advantage of the SHADOW action over the real action.

    Stored excess_return_* columns always describe the hypothetical ENTRY's
    performance vs its benchmark. For buy-like shadow actions (e.g. STARTER
    probe on a real DEFER) a positive entry excess means the shadow rule won.
    For no-buy shadow actions (e.g. DEFER guard on a real BUY) the sign is
    inverted: the shadow rule wins when the entry LAGGED its benchmark.
    Uses the longest matured horizon available (60d, else 20d, else 10d).
    """
    horizons = (60, 20, 10)
    for days in horizons:
        if days < minimum_horizon_days:
            continue
        key = f"excess_return_{days}d"
        value = _num(row.get(key), None)
        if value is not None:
            return value if shadow_is_buy_like else -value
    return None


def _independent_shadow_rows(
    rows: list[dict[str, Any]],
    *,
    independence_days: int = SHADOW_RULE_INDEPENDENCE_DAYS,
) -> list[dict[str, Any]]:
    """Keep non-overlapping ticker observations for one shadow rule.

    Daily reviews of the same company share the same price path and are not
    independent experiments. The earliest observation starts a fixed embargo;
    another observation for that ticker is admitted only after the embargo.
    """
    window = timedelta(days=max(1, int(independence_days)))

    def sort_key(row: dict[str, Any]) -> tuple[str, datetime, str]:
        ticker = str(row.get("ticker") or "").upper()
        observed = _ensure_dt(
            row.get("decision_generated_at")
            or row.get("created_at")
            or row.get("updated_at")
        ) or datetime.max.replace(tzinfo=timezone.utc)
        return ticker, observed, str(row.get("evaluation_id") or "")

    accepted: list[dict[str, Any]] = []
    last_by_ticker: dict[str, datetime] = {}
    for row in sorted(rows, key=sort_key):
        ticker = str(row.get("ticker") or "").upper().strip()
        observed = _ensure_dt(
            row.get("decision_generated_at")
            or row.get("created_at")
            or row.get("updated_at")
        )
        cluster_key = ticker or str(row.get("evaluation_id") or "unknown")
        previous = last_by_ticker.get(cluster_key)
        if previous is not None and observed is not None and observed < previous + window:
            continue
        if previous is not None and observed is None:
            continue
        accepted.append(row)
        if observed is not None:
            last_by_ticker[cluster_key] = observed
        elif cluster_key not in last_by_ticker:
            last_by_ticker[cluster_key] = datetime.max.replace(tzinfo=timezone.utc)
    return accepted


def summarize_shadow_rules(journal: DecisionJournal | None = None) -> dict[str, Any]:
    """Summarize shadow-rule practice-field status.

    Sign semantics: ``avg_excess_return_*`` is the raw entry-vs-benchmark
    excess (direction-agnostic bookkeeping); ``avg_rule_advantage`` and
    ``win_rate_vs_baseline`` are oriented so that positive always means the
    shadow rule beat the real decision (baseline), for buy-like AND
    no-buy-like shadow actions alike.
    """
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
                "shadow_action": str(row.get("shadow_action") or "").upper(),
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

    candidate_promotions: list[dict[str, Any]] = []
    for rule_id, bucket in by_rule.items():
        rule_rows = [r for r in rows if str(r.get("rule_id") or "unknown") == rule_id]
        independent_rows = _independent_shadow_rows(rule_rows)
        bucket["independent_count"] = len(independent_rows)
        bucket["independent_completed"] = sum(
            1 for row in independent_rows if row.get("status") == "completed"
        )
        bucket["suppressed_correlated_count"] = len(rule_rows) - len(independent_rows)
        for key in ("excess_return_10d", "excess_return_20d", "excess_return_60d"):
            raw_vals = [_num(r.get(key), None) for r in rule_rows if r.get(key) is not None]
            raw_vals = [v for v in raw_vals if v is not None]
            bucket[f"raw_avg_{key}"] = (
                round(sum(raw_vals) / len(raw_vals), 4) if raw_vals else None
            )
            vals = [_num(r.get(key), None) for r in independent_rows if r.get(key) is not None]
            vals = [v for v in vals if v is not None]
            bucket[f"avg_{key}"] = round(sum(vals) / len(vals), 4) if vals else None

        shadow_is_buy_like = bucket["shadow_action"] in EXPOSURE_LIKE_SHADOW_ACTIONS
        bucket["shadow_side"] = "buy_like" if shadow_is_buy_like else "no_buy_like"
        advantages = [
            adv for adv in (_row_rule_advantage(r, shadow_is_buy_like) for r in independent_rows)
            if adv is not None
        ]
        raw_advantages = [
            adv for adv in (_row_rule_advantage(r, shadow_is_buy_like) for r in rule_rows)
            if adv is not None
        ]
        promotion_advantages = [
            adv
            for adv in (
                _row_rule_advantage(
                    r,
                    shadow_is_buy_like,
                    minimum_horizon_days=SHADOW_RULE_PROMOTE_MIN_HORIZON_DAYS,
                )
                for r in independent_rows
            )
            if adv is not None
        ]
        bucket["raw_n_evaluated"] = len(raw_advantages)
        bucket["n_evaluated"] = len(advantages)
        bucket["avg_rule_advantage"] = (
            round(sum(advantages) / len(advantages), 4) if advantages else None
        )
        bucket["win_rate_vs_baseline"] = (
            round(sum(1 for adv in advantages if adv > 0) / len(advantages), 4)
            if advantages else None
        )
        bucket["promotion_n_evaluated"] = len(promotion_advantages)
        bucket["promotion_avg_rule_advantage"] = (
            round(sum(promotion_advantages) / len(promotion_advantages), 4)
            if promotion_advantages else None
        )
        bucket["promotion_win_rate_vs_baseline"] = (
            round(
                sum(1 for adv in promotion_advantages if adv > 0)
                / len(promotion_advantages),
                4,
            )
            if promotion_advantages else None
        )
        promotable = (
            len(promotion_advantages) >= SHADOW_RULE_PROMOTE_MIN_N
            and bucket["promotion_win_rate_vs_baseline"] is not None
            and bucket["promotion_win_rate_vs_baseline"] >= SHADOW_RULE_PROMOTE_WIN_RATE
            and bucket["promotion_avg_rule_advantage"] is not None
            and bucket["promotion_avg_rule_advantage"] > 0
        )
        bucket["graduation"] = "candidate_promote" if promotable else "tracking"
        if promotable:
            candidate_promotions.append(
                {
                    "rule_id": rule_id,
                    "kind": "entry_side",
                    "evidence": (
                        f"independent_n={len(promotion_advantages)}, "
                        f"win_rate={bucket['promotion_win_rate_vs_baseline']:.0%}, "
                        f"avg_advantage={bucket['promotion_avg_rule_advantage']:+.2%}"
                    ),
                }
            )

    summary: dict[str, Any] = {
        "total": len(rows),
        "completed": sum(1 for r in rows if r.get("status") == "completed"),
        "tracking": sum(1 for r in rows if r.get("status") != "completed"),
        "rules": by_rule,
        "sign_semantics": (
            "avg_rule_advantage/win_rate_vs_baseline are oriented so positive = "
            "shadow rule beat the real decision (sign flipped for no-buy shadow actions)."
        ),
        "promotion_gate": {
            "min_n": SHADOW_RULE_PROMOTE_MIN_N,
            "win_rate": SHADOW_RULE_PROMOTE_WIN_RATE,
            "minimum_horizon_days": SHADOW_RULE_PROMOTE_MIN_HORIZON_DAYS,
            "same_ticker_independence_days": SHADOW_RULE_INDEPENDENCE_DAYS,
            "positive_average_advantage_required": True,
        },
    }

    try:
        sell_summary = summarize_sell_shadow_rules()
        summary["sell_rules"] = sell_summary
        candidate_promotions.extend(sell_summary.get("candidate_promotions") or [])
    except Exception as exc:
        logger.warning("[shadow_rules] Sell-rule summary failed: %s", exc)
        summary["sell_rules"] = {}

    summary["candidate_promotions"] = candidate_promotions
    return summary


# ---------------------------------------------------------------------------
# Sell-side shadow rules — simulated against real position price paths.
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, payload: Any) -> None:
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


def _load_sell_rule_store() -> dict[str, Any]:
    if not SELL_RULES_FILE.exists():
        return {"evaluations": {}, "updated_at": ""}
    try:
        data = json.loads(SELL_RULES_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("evaluations"), dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {"evaluations": {}, "updated_at": ""}


def _atr_series(bars: list[dict[str, Any]], window: int = 14) -> list[float]:
    """Simple trailing-average true range per bar (warmup uses available bars)."""
    atrs: list[float] = []
    trs: list[float] = []
    prev_close: float | None = None
    for bar in bars:
        high = float(bar.get("high") or 0)
        low = float(bar.get("low") or 0)
        close = float(bar.get("close") or 0)
        if prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
        recent = trs[-window:]
        atrs.append(sum(recent) / len(recent) if recent else 0.0)
        prev_close = close
    return atrs


def _simulate_sell_rule(
    rule_id: str,
    entry_price: float,
    bars: list[dict[str, Any]],
) -> dict[str, Any]:
    """Simulate one exit rule over a real position's daily OHLC path.

    Returns rule_return (fractional), exit_triggered, exit_date. Gap-downs
    exit at the open when it is already below the stop.
    """
    final_close = float(bars[-1].get("close") or entry_price)
    hold_return = (final_close - entry_price) / entry_price

    def _stop_exit(stop_levels: list[float]) -> dict[str, Any]:
        for i, bar in enumerate(bars):
            stop = stop_levels[i]
            low = float(bar.get("low") or 0)
            open_ = float(bar.get("open") or 0)
            if low <= stop:
                exit_price = open_ if open_ < stop else stop
                return {
                    "rule_return": (exit_price - entry_price) / entry_price,
                    "exit_triggered": True,
                    "exit_date": str(bar.get("date") or ""),
                }
        return {"rule_return": hold_return, "exit_triggered": False, "exit_date": ""}

    if rule_id == "stop_width_8pct":
        return _stop_exit([entry_price * 0.92] * len(bars))
    if rule_id == "stop_width_15pct":
        return _stop_exit([entry_price * 0.85] * len(bars))
    if rule_id == "chandelier_3atr":
        atrs = _atr_series(bars)
        stops: list[float] = []
        highest = entry_price
        prev_stop = 0.0
        for i, bar in enumerate(bars):
            highest = max(highest, float(bar.get("high") or 0))
            stop = max(prev_stop, highest - 3.0 * atrs[i])
            stops.append(stop)
            prev_stop = stop
        return _stop_exit(stops)
    if rule_id == "scale_out_half_at_20pct":
        target = entry_price * 1.20
        for bar in bars:
            if float(bar.get("high") or 0) >= target:
                # Realize +20% on half; the rest rides to the window end.
                rule_return = 0.5 * 0.20 + 0.5 * hold_return
                return {
                    "rule_return": rule_return,
                    "exit_triggered": True,
                    "exit_date": str(bar.get("date") or ""),
                }
        return {"rule_return": hold_return, "exit_triggered": False, "exit_date": ""}
    # all_or_nothing (baseline): hold through the evaluation window.
    return {"rule_return": hold_return, "exit_triggered": False, "exit_date": ""}


def _load_portfolio_positions() -> list[dict[str, Any]]:
    try:
        payload = json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    positions = payload.get("positions")
    return positions if isinstance(positions, list) else []


def evaluate_sell_shadow_rules(
    positions: list[dict[str, Any]] | None = None,
    price_history_fn: Callable[[str], list[dict[str, Any]] | None] | None = None,
    as_of: date | None = None,
    horizon_days: int = SELL_RULE_HORIZON_DAYS,
) -> dict[str, Any]:
    """Evaluate sell-side shadow rules against real position price paths.

    For each open position, simulates every rule in SELL_SHADOW_RULE_IDS over
    the daily OHLC path from entry to min(entry + horizon, today) and scores
    it against the hold-through baseline. Results are upserted into
    data/shadow_sell_rules.json; an evaluation is 'completed' once the
    horizon has fully elapsed. Never changes live positions or config.
    """
    as_of = as_of or datetime.now(timezone.utc).date()
    if positions is None:
        positions = _load_portfolio_positions()

    if price_history_fn is None:
        def price_history_fn(ticker: str) -> list[dict[str, Any]] | None:
            from .collector import FMPCollector
            return FMPCollector().history(ticker, period="1y")

    store = _load_sell_rule_store()
    evaluations: dict[str, Any] = store["evaluations"]
    updated = 0
    skipped = 0
    for position in positions:
        ticker = str(position.get("ticker") or "").upper().strip()
        entry_price = _num(position.get("avg_cost"), None)
        entry_date = str(position.get("entry_date") or position.get("opened_at") or "")[:10]
        if not ticker or not entry_price or entry_price <= 0 or not entry_date:
            skipped += 1
            continue
        try:
            entry_d = date.fromisoformat(entry_date)
        except ValueError:
            skipped += 1
            continue
        window_end = min(entry_d + timedelta(days=horizon_days), as_of)
        try:
            history = price_history_fn(ticker) or []
        except Exception as exc:
            logger.warning("[shadow_rules] Price history failed for %s: %s", ticker, exc)
            skipped += 1
            continue
        bars = [
            bar for bar in history
            if entry_date < str(bar.get("date") or "") <= window_end.isoformat()
        ]
        if not bars:
            skipped += 1
            continue

        status = "completed" if entry_d + timedelta(days=horizon_days) <= as_of else "tracking"
        baseline = _simulate_sell_rule(SELL_RULE_BASELINE, entry_price, bars)
        baseline_return = float(baseline["rule_return"])
        for rule_id in SELL_SHADOW_RULE_IDS:
            result = _simulate_sell_rule(rule_id, entry_price, bars)
            advantage = float(result["rule_return"]) - baseline_return
            eval_id = hashlib.sha256(
                f"{RULE_VERSION}|sell|{rule_id}|{ticker}|{entry_date}".encode("utf-8")
            ).hexdigest()[:32]
            evaluations[eval_id] = {
                "evaluation_id": eval_id,
                "rule_id": rule_id,
                "kind": "sell_side",
                "ticker": ticker,
                "entry_date": entry_date,
                "entry_price": entry_price,
                "horizon_days": horizon_days,
                "window_end": window_end.isoformat(),
                "bars": len(bars),
                "rule_return": round(float(result["rule_return"]), 6),
                "baseline_rule": SELL_RULE_BASELINE,
                "baseline_return": round(baseline_return, 6),
                "advantage": round(advantage, 6),
                "beat_baseline": advantage > 0,
                "exit_triggered": bool(result["exit_triggered"]),
                "exit_date": result["exit_date"],
                "status": status,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            updated += 1

    store["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(SELL_RULES_FILE, store)
    logger.info(
        "[shadow_rules] Sell-rule evaluation: %d rule-evals updated, %d positions skipped",
        updated, skipped,
    )
    return {"updated": updated, "skipped": skipped, "evaluations": len(evaluations)}


def summarize_sell_shadow_rules() -> dict[str, Any]:
    """Summarize sell-side shadow rules and flag graduation candidates."""
    store = _load_sell_rule_store()
    rows = list(store.get("evaluations", {}).values())
    by_rule: dict[str, dict[str, Any]] = {}
    candidate_promotions: list[dict[str, Any]] = []
    for rule_id in SELL_SHADOW_RULE_IDS:
        rule_rows = [r for r in rows if r.get("rule_id") == rule_id]
        completed = [r for r in rule_rows if r.get("status") == "completed"]
        advantages = [_num(r.get("advantage"), 0.0) or 0.0 for r in completed]
        wins = sum(1 for r in completed if r.get("beat_baseline"))
        win_rate = round(wins / len(completed), 4) if completed else None
        bucket = {
            "count": len(rule_rows),
            "completed": len(completed),
            "tracking": len(rule_rows) - len(completed),
            "win_rate_vs_baseline": win_rate,
            "avg_advantage": (
                round(sum(advantages) / len(advantages), 4) if advantages else None
            ),
            "graduation": "tracking",
        }
        if rule_id == SELL_RULE_BASELINE:
            bucket["graduation"] = "baseline"
        elif (
            len(completed) >= SHADOW_RULE_PROMOTE_MIN_N
            and win_rate is not None
            and win_rate >= SHADOW_RULE_PROMOTE_WIN_RATE
        ):
            bucket["graduation"] = "candidate_promote"
            candidate_promotions.append(
                {
                    "rule_id": rule_id,
                    "kind": "sell_side",
                    "evidence": (
                        f"n={len(completed)}, win_rate={win_rate:.0%}, "
                        f"avg_advantage={bucket['avg_advantage']:+.2%} vs {SELL_RULE_BASELINE}"
                    ),
                }
            )
        by_rule[rule_id] = bucket
    return {
        "total": len(rows),
        "baseline_rule": SELL_RULE_BASELINE,
        "rules": by_rule,
        "candidate_promotions": candidate_promotions,
    }
