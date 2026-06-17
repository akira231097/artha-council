"""Audited buy-side scoring for Artha council decisions.

The buy council uses this module to separate deterministic evidence scoring
from the CIO's bounded common-sense adjustment. The final score is always:

    deterministic base score + deterministic rule adjustments + CIO adjustment

Hard gates still live in council.py and cannot be bypassed by this module.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


_CIO_CATEGORIES = {"none", "evidence_backed", "logic_backed", "risk_override", "data_dispute"}


def _num(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("$", "").replace("%", "").replace(",", "")
        if not cleaned:
            return default
        try:
            return float(cleaned)
        except ValueError:
            return default
    return default


def _clamp(value: float | int, lo: int = 0, hi: int = 100) -> int:
    try:
        return int(round(max(lo, min(hi, float(value)))))
    except Exception:
        return lo


def _first(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    if isinstance(payload, dict):
        return payload
    return {}


def _pct(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    try:
        return (float(numerator) - float(denominator)) / float(denominator) * 100.0
    except Exception:
        return None


def _analyst_vote_counts(analysts: list[Any] | tuple[Any, ...] | None) -> dict[str, int]:
    counts = {"BUY": 0, "HOLD": 0, "SELL": 0}
    for analyst in analysts or []:
        verdict = str(getattr(analyst, "verdict", "") or "").upper()
        if verdict in counts:
            counts[verdict] += 1
    return counts


def _score_technicals(stock_data: dict[str, Any]) -> tuple[int, list[str]]:
    quote = stock_data.get("quote") or stock_data.get("yf_quote") or {}
    technicals = stock_data.get("technicals") or {}
    price = _num(quote.get("price"))
    rsi = _num(technicals.get("rsi"))
    sma20 = _num(technicals.get("sma_20"))
    sma50 = _num(technicals.get("sma_50"))
    sma200 = _num(technicals.get("sma_200"))
    macd = str(technicals.get("macd_interpretation") or "").lower()
    crossover = str(technicals.get("macd_crossover") or "").lower()
    volume_ratio = _num(technicals.get("volume_ratio"))

    score = 0
    notes: list[str] = []
    if rsi is not None:
        if 42 <= rsi <= 62:
            score += 8
            notes.append(f"RSI {rsi:.1f} is constructive, not overheated")
        elif 35 <= rsi < 42 or 62 < rsi <= 68:
            score += 6
            notes.append(f"RSI {rsi:.1f} is usable but less ideal")
        elif 28 <= rsi < 35:
            score += 5
            notes.append(f"RSI {rsi:.1f} is oversold/potentially early")
        elif 68 < rsi <= 75:
            score += 2
            notes.append(f"RSI {rsi:.1f} is stretched")
        else:
            score += 1
            notes.append(f"RSI {rsi:.1f} is poor for fresh entry")

    vs50 = _pct(price, sma50)
    if vs50 is not None:
        if -5 <= vs50 <= 8:
            score += 6
            notes.append(f"price is near 50-day trend ({vs50:+.1f}%)")
        elif 8 < vs50 <= 15:
            score += 3
            notes.append(f"price is above 50-day trend but not extreme ({vs50:+.1f}%)")
        elif -15 <= vs50 < -5:
            score += 3
            notes.append(f"price is below 50-day trend but still monitorable ({vs50:+.1f}%)")
        elif vs50 > 20:
            notes.append(f"price is too extended above 50-day trend ({vs50:+.1f}%)")

    if price is not None and sma200 is not None:
        if price >= sma200:
            score += 4
            notes.append("price is above 200-day trend")
        elif _pct(price, sma200) is not None and _pct(price, sma200) >= -5:
            score += 2
            notes.append("price is testing 200-day trend support")

    if price is not None and sma20 is not None and price >= sma20:
        score += 2
    if macd == "bullish":
        score += 3
        notes.append("MACD is bullish")
    if crossover == "bullish_crossover":
        score += 2
        notes.append("fresh bullish MACD crossover")
    if volume_ratio is not None:
        if volume_ratio >= 1.2:
            score += 2
            notes.append(f"volume confirmation {volume_ratio:.1f}x")
        elif volume_ratio >= 0.7:
            score += 1

    return _clamp(score, 0, 25), notes[:5]


def _score_fundamentals(stock_data: dict[str, Any]) -> tuple[int, list[str]]:
    ratios = stock_data.get("ratios_ttm") or {}
    metrics = stock_data.get("key_metrics_ttm") or {}
    income = _first(stock_data.get("income_statement"))
    cash_flow = _first(stock_data.get("cash_flow") or stock_data.get("cash_flow_statement"))
    balance = _first(stock_data.get("balance_sheet"))

    net_margin = _num(ratios.get("netProfitMarginTTM"))
    roic = _num(metrics.get("returnOnInvestedCapitalTTM"))
    roe = _num(metrics.get("returnOnEquityTTM"))
    fcf_yield = _num(metrics.get("freeCashFlowYieldTTM"))
    de_ratio = _num(ratios.get("debtToEquityRatioTTM"))
    revenue = _num(income.get("revenue"))
    net_income = _num(income.get("netIncome"))
    fcf = _num(cash_flow.get("freeCashFlow"))
    cash = _num(balance.get("cashAndCashEquivalents"))
    debt = _num(balance.get("totalDebt"))

    score = 0
    notes: list[str] = []
    if net_margin is not None:
        margin_pct = net_margin * 100 if abs(net_margin) <= 1 else net_margin
        if margin_pct >= 15:
            score += 4
        elif margin_pct >= 5:
            score += 2
        if margin_pct > 0:
            notes.append(f"positive TTM margin {margin_pct:.1f}%")
    if roic is not None:
        roic_pct = roic * 100 if abs(roic) <= 1 else roic
        if roic_pct >= 15:
            score += 4
            notes.append(f"strong ROIC {roic_pct:.1f}%")
        elif roic_pct >= 8:
            score += 2
    if roe is not None:
        roe_pct = roe * 100 if abs(roe) <= 1 else roe
        if roe_pct >= 15:
            score += 2
    if fcf_yield is not None:
        fcf_yield_pct = fcf_yield * 100 if abs(fcf_yield) <= 1 else fcf_yield
        if fcf_yield_pct >= 5:
            score += 4
            notes.append(f"healthy FCF yield {fcf_yield_pct:.1f}%")
        elif fcf_yield_pct >= 2:
            score += 2
        elif fcf_yield_pct < 0:
            notes.append(f"negative FCF yield {fcf_yield_pct:.1f}%")
    if de_ratio is not None:
        if de_ratio < 1:
            score += 3
        elif de_ratio < 2:
            score += 2
        elif de_ratio > 5:
            score -= 2
            notes.append(f"high leverage D/E {de_ratio:.1f}x")
    if revenue and revenue > 0:
        score += 1
    if net_income and net_income > 0:
        score += 1
    if fcf and fcf > 0:
        score += 1
    if cash is not None and debt is not None and cash >= debt:
        score += 1
        notes.append("cash covers debt")

    return _clamp(score, 0, 20), notes[:5]


def _score_sentiment(stock_data: dict[str, Any], valuation: dict[str, Any]) -> tuple[int, list[str]]:
    targets = valuation.get("analyst_targets") or {}
    rec_trends = stock_data.get("recommendation_trends") or {}
    short_interest = stock_data.get("short_interest") or {}
    consensus_upside = _num(targets.get("consensus_upside_pct"))
    net_upgrades = _num(rec_trends.get("net_upgrades_30d"), 0) or 0
    net_downgrades = _num(rec_trends.get("net_downgrades_30d"), 0) or 0
    short_pct = _num(short_interest.get("short_interest_pct"))
    squeeze = bool(short_interest.get("squeeze_risk_flag"))
    val_signal = str(valuation.get("valuation_signal") or "").lower()

    score = 0
    notes: list[str] = []
    if consensus_upside is not None:
        if consensus_upside >= 20:
            score += 5
        elif consensus_upside >= 8:
            score += 3
        elif consensus_upside >= 0:
            score += 1
        else:
            notes.append(f"consensus downside {consensus_upside:.1f}%")
    if net_upgrades > net_downgrades:
        score += 3
        notes.append("net analyst upgrades")
    elif net_downgrades > net_upgrades:
        score -= 2
        notes.append("net analyst downgrades")
    if val_signal == "positive":
        score += 3
    elif val_signal == "neutral":
        score += 1
    if short_pct is not None:
        if 5 <= short_pct <= 18:
            score += 2
        elif short_pct > 25:
            score -= 1
            notes.append(f"crowded short interest {short_pct:.1f}%")
    if squeeze:
        score += 1
        notes.append("possible squeeze asymmetry")

    return _clamp(score, 0, 15), notes[:5]


def _score_regime(stock_data: dict[str, Any], valuation: dict[str, Any], portfolio_risk: dict[str, Any], fear_greed: int | None) -> tuple[int, list[str]]:
    profile = stock_data.get("profile") or {}
    beta = _num(profile.get("beta"))
    risk_level = str(portfolio_risk.get("risk_level") or "").lower()
    val_signal = str(valuation.get("valuation_signal") or "").lower()
    try:
        fg = int(fear_greed if fear_greed is not None else 50)
    except Exception:
        fg = 50

    score = 0
    notes: list[str] = []
    if fg < 20:
        score += 5
        notes.append("extreme fear gives better entry asymmetry")
    elif fg < 40:
        score += 4
        notes.append("fear regime supports selective buying")
    elif fg <= 60:
        score += 3
    elif fg <= 80:
        score += 1
        notes.append("greed regime lowers entry quality")
    else:
        notes.append("extreme greed blocks aggressive entry")
    if risk_level == "low":
        score += 3
    elif risk_level == "moderate":
        score += 1
    if val_signal == "positive":
        score += 3
    elif val_signal == "neutral":
        score += 1
    if beta is not None:
        if beta <= 1.3:
            score += 2
        elif beta > 2.5 and fg < 40:
            score -= 1
            notes.append(f"high beta {beta:.2f} in fear regime")
    return _clamp(score, 0, 15), notes[:5]


def _score_catalyst(stock_data: dict[str, Any], valuation: dict[str, Any]) -> tuple[int, list[str]]:
    earnings = stock_data.get("earnings_context") or {}
    earnings_surprise = _first(stock_data.get("earnings_surprises"))
    rec_trends = stock_data.get("recommendation_trends") or {}
    analyst_estimates = stock_data.get("analyst_estimates") or {}
    days = _num(earnings.get("days_to_earnings"))
    surprise_pct = _num(earnings_surprise.get("surprisePercent"))
    net_upgrades = _num(rec_trends.get("net_upgrades_30d"), 0) or 0
    net_downgrades = _num(rec_trends.get("net_downgrades_30d"), 0) or 0
    fy_rev = _num(analyst_estimates.get("fy1_revenue_estimate"))
    consensus_upside = _num((valuation.get("analyst_targets") or {}).get("consensus_upside_pct"))

    score = 0
    notes: list[str] = []
    if days is None:
        score += 1
    elif days > 14:
        score += 2
    elif days <= 2:
        notes.append("earnings binary event is too close")
    else:
        score += 1
    if surprise_pct is not None:
        if surprise_pct > 5:
            score += 2
            notes.append(f"recent EPS beat {surprise_pct:.1f}%")
        elif surprise_pct < -5:
            score -= 1
    if net_upgrades > net_downgrades:
        score += 2
    if fy_rev and fy_rev > 0:
        score += 1
    if consensus_upside is not None and consensus_upside >= 15:
        score += 2
    elif consensus_upside is not None and consensus_upside >= 5:
        score += 1
    return _clamp(score, 0, 10), notes[:5]


def _score_data_quality(data_quality: dict[str, Any]) -> tuple[int, list[str]]:
    completeness = _num(data_quality.get("completeness_score"), 0) or 0
    context = _num(data_quality.get("context_coverage_score"), 0) or 0
    conflicts = data_quality.get("source_conflicts") or []
    gaps = data_quality.get("enrichment_missing_fields") or []
    score = completeness * 0.06 + context * 0.04
    notes = [f"core {completeness:.1f}%, context {context:.1f}%"]
    if conflicts:
        score -= min(3, len(conflicts))
        notes.append(f"{len(conflicts)} source conflict(s)")
    if isinstance(gaps, list) and len(gaps) >= 3:
        score -= 1
    return _clamp(score, 0, 10), notes[:5]


def _score_liquidity(stock_data: dict[str, Any]) -> tuple[int, list[str]]:
    quote = stock_data.get("quote") or stock_data.get("yf_quote") or {}
    profile = stock_data.get("profile") or {}
    market_cap = _num(quote.get("marketCap"), _num(profile.get("mktCap") or profile.get("marketCap"), 0)) or 0
    try:
        from .liquidity import resolve_average_volume
        volume_info = resolve_average_volume(stock_data)
        volume = _num(volume_info.get("volume"), 0) or 0
        volume_label = "avg volume" if volume_info.get("is_average") else "current volume"
    except Exception:
        volume = _num(quote.get("avgVolume"), _num((stock_data.get("yf_quote") or {}).get("averageVolume"), 0)) or 0
        volume_label = "volume"
    score = 0
    if market_cap >= 10_000_000_000:
        score += 2
    elif market_cap >= 1_000_000_000:
        score += 1
    if volume >= 1_000_000:
        score += 3
    elif volume >= 100_000:
        score += 2
    elif volume > 0:
        score += 1
    return _clamp(score, 0, 5), [f"market cap ${market_cap:,.0f}, {volume_label} {volume:,.0f}"] if market_cap or volume else []


def build_buy_score_audit(
    *,
    stock_data: dict[str, Any],
    data_quality_report: dict[str, Any],
    valuation_expectations: dict[str, Any],
    portfolio_factor_risk: dict[str, Any],
    analysts: list[Any] | tuple[Any, ...] | None = None,
    research_insufficient: bool = False,
    fear_greed: int | None = 50,
) -> dict[str, Any]:
    """Build a deterministic buy score before CIO adjustment."""
    components: dict[str, int] = {}
    component_notes: dict[str, list[str]] = {}

    scorers = {
        "technical_setup": _score_technicals(stock_data),
        "fundamental_quality": _score_fundamentals(stock_data),
        "contrarian_sentiment": _score_sentiment(stock_data, valuation_expectations or {}),
        "regime_alignment": _score_regime(stock_data, valuation_expectations or {}, portfolio_factor_risk or {}, fear_greed),
        "catalyst_asymmetry": _score_catalyst(stock_data, valuation_expectations or {}),
        "data_quality": _score_data_quality(data_quality_report or {}),
        "liquidity_execution": _score_liquidity(stock_data),
    }
    for key, (score, notes) in scorers.items():
        components[key] = int(score)
        component_notes[key] = list(notes or [])

    base_score = _clamp(sum(components.values()))
    adjustments: list[dict[str, Any]] = []

    def add(points: int, reason: str, category: str = "rule") -> None:
        if points:
            adjustments.append({"points": int(points), "reason": reason, "category": category})

    valuation = valuation_expectations or {}
    val_signal = str(valuation.get("valuation_signal") or "").lower()
    expectation_risk = str(valuation.get("expectation_risk_level") or "").lower()
    if val_signal == "positive":
        add(4, "deterministic valuation signal is positive", "valuation")
    elif val_signal == "negative":
        add(-8, "deterministic valuation signal is negative", "valuation")
    if expectation_risk == "low":
        add(2, "expectation risk is low", "valuation")
    elif expectation_risk == "moderate":
        add(-2, "expectation risk is moderate", "valuation")
    elif expectation_risk == "high":
        add(-5, "expectation risk is high", "valuation")

    dq = data_quality_report or {}
    if research_insufficient:
        add(-10, "current-web research is insufficient for a fresh buy", "data_quality")
    conflicts = dq.get("source_conflicts") or []
    if conflicts:
        add(-min(8, 3 + len(conflicts)), "source conflicts reduce score reliability", "data_quality")
    if (_num(dq.get("completeness_score"), 100) or 0) < 90:
        add(-5, "core data completeness below 90%", "data_quality")
    if (_num(dq.get("context_coverage_score"), 100) or 0) < 75:
        add(-3, "context coverage below 75%", "data_quality")

    earnings = stock_data.get("earnings_context") or {}
    if earnings.get("earnings_defer_flag"):
        add(-8, "earnings binary event is too close", "timing")
    elif earnings.get("earnings_risk_flag"):
        add(-3, "earnings event risk is near", "timing")

    portfolio_risk = portfolio_factor_risk or {}
    risk_level = str(portfolio_risk.get("risk_level") or "").lower()
    if risk_level == "low":
        add(1, "portfolio/factor risk is low", "portfolio")
    elif risk_level == "moderate":
        add(-3, "portfolio/factor risk is moderate", "portfolio")
    elif risk_level == "high":
        add(-8, "portfolio/factor risk is high", "portfolio")

    votes = _analyst_vote_counts(analysts)
    if votes["BUY"] >= 2:
        add(4, "at least two independent analysts are constructive", "analyst_vote")
    elif votes["SELL"] >= 2:
        add(-6, "at least two independent analysts are negative", "analyst_vote")
    elif votes["HOLD"] == 3:
        add(-2, "all analysts are cautious/hold", "analyst_vote")

    risk_flags = valuation.get("risk_flags") or []
    joined_flags = " | ".join(str(flag).lower() for flag in risk_flags)
    if "rsi is overbought" in joined_flags:
        add(-4, "valuation engine flagged overbought RSI", "timing")
    if "above 50-day sma" in joined_flags:
        add(-4, "valuation engine flagged price extension above 50-day trend", "timing")
    if "low free-cash-flow yield" in joined_flags:
        add(-3, "valuation engine flagged low free-cash-flow yield", "quality")

    rule_adjustment_total = sum(int(item["points"]) for item in adjustments)
    pre_cio_score = _clamp(base_score + rule_adjustment_total)
    return {
        "schema_version": 1,
        "score_type": "buy_hybrid",
        "base_score": base_score,
        "components": components,
        "component_notes": component_notes,
        "rule_adjustments": adjustments,
        "rule_adjustment_total": rule_adjustment_total,
        "pre_cio_score": pre_cio_score,
        "cio_adjustment_raw": 0,
        "cio_adjustment": 0,
        "cio_adjustment_category": "none",
        "cio_adjustment_status": "not_requested",
        "cio_adjustment_reason": "",
        "cio_adjustment_evidence": [],
        "cio_adjustment_rejected_reason": "",
        "final_score": pre_cio_score,
    }


def _coerce_evidence_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def apply_cio_buy_adjustment(
    scoring: dict[str, Any] | None,
    audit: dict[str, Any],
    *,
    config: Any,
    cio_confidence: int | None = None,
) -> dict[str, Any]:
    """Apply a bounded CIO score adjustment to a deterministic buy audit."""
    result = deepcopy(audit or {})
    pre_cio_score = _clamp(result.get("pre_cio_score", result.get("base_score", 0)))
    scoring = scoring or {}
    raw_adj = _clamp(_num(scoring.get("cio_score_adjustment"), 0) or 0, -100, 100)
    category = str(scoring.get("cio_adjustment_category") or "none").strip().lower()
    reason = str(scoring.get("cio_adjustment_reason") or "").strip()
    evidence = _coerce_evidence_list(scoring.get("cio_adjustment_evidence"))
    try:
        confidence = int(cio_confidence if cio_confidence is not None else scoring.get("confidence", 0))
    except Exception:
        confidence = 0

    result.update(
        {
            "cio_adjustment_raw": raw_adj,
            "cio_adjustment": 0,
            "cio_adjustment_category": category,
            "cio_adjustment_reason": reason,
            "cio_adjustment_evidence": evidence,
            "cio_adjustment_status": "none" if raw_adj == 0 else "rejected",
            "cio_adjustment_rejected_reason": "",
        }
    )

    if raw_adj == 0:
        result["final_score"] = pre_cio_score
        return result

    if category not in _CIO_CATEGORIES or category == "none":
        result["cio_adjustment_rejected_reason"] = "missing or invalid adjustment category"
        result["final_score"] = pre_cio_score
        return result
    if confidence < int(getattr(config, "BUY_CIO_ADJUSTMENT_MIN_CONFIDENCE", 6)):
        result["cio_adjustment_rejected_reason"] = "CIO confidence below adjustment threshold"
        result["final_score"] = pre_cio_score
        return result
    if len(reason) < 40:
        result["cio_adjustment_rejected_reason"] = "adjustment reason is too thin"
        result["final_score"] = pre_cio_score
        return result
    if category != "logic_backed" and not evidence:
        result["cio_adjustment_rejected_reason"] = "non-logic adjustment needs cited evidence or raw anchor"
        result["final_score"] = pre_cio_score
        return result

    if category == "logic_backed":
        lo = int(getattr(config, "BUY_CIO_LOGIC_ADJUSTMENT_MAX_NEGATIVE", -8))
        hi = int(getattr(config, "BUY_CIO_LOGIC_ADJUSTMENT_MAX_POSITIVE", 8))
    elif category == "risk_override":
        lo = int(getattr(config, "BUY_CIO_RISK_OVERRIDE_MAX_NEGATIVE", -18))
        hi = 0
    elif category == "data_dispute":
        lo = int(getattr(config, "BUY_CIO_DATA_DISPUTE_MAX_NEGATIVE", -10))
        hi = int(getattr(config, "BUY_CIO_DATA_DISPUTE_MAX_POSITIVE", 10))
    else:
        lo = int(getattr(config, "BUY_CIO_ADJUSTMENT_MAX_NEGATIVE", -15))
        hi = int(getattr(config, "BUY_CIO_ADJUSTMENT_MAX_POSITIVE", 15))

    bounded = max(lo, min(hi, raw_adj))
    status = "accepted_clamped" if bounded != raw_adj else "accepted"

    # A pure logic adjustment can keep a novel idea alive, but should not turn a
    # very weak data case into a live buy by itself.
    if category == "logic_backed" and bounded > 0 and pre_cio_score < 45:
        bounded = min(bounded, 5)
        status = "accepted_clamped"

    result["cio_adjustment"] = int(bounded)
    result["cio_adjustment_status"] = status
    result["final_score"] = _clamp(pre_cio_score + bounded)
    return result


def render_buy_score_audit(audit: dict[str, Any] | None, *, include_details: bool = True) -> str:
    """Render a compact audit block for prompts and reports."""
    if not isinstance(audit, dict) or not audit:
        return "Buy score audit unavailable."
    base = int(audit.get("base_score") or 0)
    rules = int(audit.get("rule_adjustment_total") or 0)
    pre = int(audit.get("pre_cio_score") or 0)
    cio = int(audit.get("cio_adjustment") or 0)
    final = int(audit.get("final_score", pre + cio) or 0)
    lines = [
        f"Score audit: base {base} + rules {rules:+d} + CIO {cio:+d} = {final}",
        f"Pre-CIO deterministic score: {pre}",
    ]
    status = str(audit.get("cio_adjustment_status") or "")
    category = str(audit.get("cio_adjustment_category") or "none")
    if status and status not in {"not_requested", "none"}:
        lines.append(f"CIO adjustment: {status} | category={category}")
        rejected = str(audit.get("cio_adjustment_rejected_reason") or "").strip()
        if rejected:
            lines.append(f"CIO adjustment rejected reason: {rejected}")
    if include_details:
        components = audit.get("components") or {}
        if components:
            comp_text = ", ".join(f"{k}={v}" for k, v in components.items())
            lines.append(f"Base components: {comp_text}")
        adjustments = audit.get("rule_adjustments") or []
        if adjustments:
            lines.append("Rule adjustments:")
            for item in adjustments[:10]:
                lines.append(f"- {int(item.get('points') or 0):+d}: {item.get('reason')}")
        reason = str(audit.get("cio_adjustment_reason") or "").strip()
        if reason:
            lines.append(f"CIO adjustment reason: {reason}")
    return "\n".join(lines)
