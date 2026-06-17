"""Deterministic valuation and expectation-risk anchors for Artha.

This module intentionally does no LLM reasoning. It converts the raw evidence
packet into a compact, auditable valuation read that the council can inspect
before writing a BUY/DEFER/AVOID thesis.
"""
from __future__ import annotations

from typing import Any


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.replace(",", "").replace("%", "").strip()
            if value == "":
                return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _pct(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return (numerator - denominator) / abs(denominator) * 100.0


def _first_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    if isinstance(value, dict):
        return value
    return {}


def _latest_and_prior_quarters(rows: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        return {}, {}
    latest = rows[0] if isinstance(rows[0], dict) else {}
    prior = rows[3] if len(rows) >= 4 and isinstance(rows[3], dict) else {}
    return latest, prior


def _score_clamp(value: float) -> int:
    return int(max(0, min(100, round(value))))


def build_valuation_expectations(stock_data: dict[str, Any]) -> dict[str, Any]:
    """Build a deterministic valuation/expectations diagnostic.

    The output is deliberately compact and JSON-safe so it can be stored in
    point-in-time dossiers and reused for calibration.
    """
    quote = stock_data.get("quote") or stock_data.get("yf_quote") or {}
    yf_quote = stock_data.get("yf_quote") or {}
    profile = stock_data.get("profile") or {}
    ratios = stock_data.get("ratios_ttm") or {}
    key_metrics = stock_data.get("key_metrics_ttm") or {}
    dcf_data = stock_data.get("dcf") or {}
    price_targets = stock_data.get("price_target_consensus") or {}
    estimates = stock_data.get("analyst_estimates") or {}
    rec_trends = stock_data.get("recommendation_trends") or {}
    technicals = stock_data.get("technicals") or {}
    earnings_context = stock_data.get("earnings_context") or {}

    current_price = _num(quote.get("price"), _num(yf_quote.get("price")))
    market_cap = _num(quote.get("marketCap"), _num(profile.get("mktCap") or profile.get("marketCap")))
    dcf_value = _num(dcf_data.get("dcf"))
    dcf_upside_pct = _pct(dcf_value, current_price)

    consensus_target = _num(
        price_targets.get("targetConsensus"),
        _num(estimates.get("price_target_consensus")),
    )
    median_target = _num(price_targets.get("targetMedian"))
    low_target = _num(price_targets.get("targetLow"), _num(estimates.get("price_target_low")))
    high_target = _num(price_targets.get("targetHigh"), _num(estimates.get("price_target_high")))
    consensus_upside_pct = _pct(consensus_target, current_price)
    median_upside_pct = _pct(median_target, current_price)
    low_target_downside_pct = _pct(low_target, current_price)
    high_target_upside_pct = _pct(high_target, current_price)

    trailing_pe = _num(ratios.get("priceToEarningsRatioTTM"), _num(quote.get("pe"), _num(yf_quote.get("pe_ratio"))))
    forward_pe = _num(yf_quote.get("forward_pe"))
    peg = _num(ratios.get("priceToEarningsGrowthRatioTTM"))
    forward_peg = _num(ratios.get("forwardPriceToEarningsGrowthRatioTTM"))
    price_to_sales = _num(ratios.get("priceToSalesRatioTTM"))
    price_to_book = _num(ratios.get("priceToBookRatioTTM"))
    fcf_yield = _num(key_metrics.get("freeCashFlowYieldTTM"))
    roic = _num(key_metrics.get("returnOnInvestedCapitalTTM"))
    roe = _num(key_metrics.get("returnOnEquityTTM"))
    net_margin = _num(ratios.get("netProfitMarginTTM"))
    current_ratio = _num(ratios.get("currentRatioTTM"))
    debt_to_equity = _num(ratios.get("debtToEquityRatioTTM"))

    latest_income, prior_income = _latest_and_prior_quarters(stock_data.get("income_statement"))
    latest_cash_flow = _first_dict(stock_data.get("cash_flow") or stock_data.get("cash_flow_statement"))
    latest_revenue = _num(latest_income.get("revenue"))
    prior_revenue = _num(prior_income.get("revenue"))
    latest_eps = _num(latest_income.get("eps"))
    prior_eps = _num(prior_income.get("eps"))
    revenue_yoy_pct = _pct(latest_revenue, prior_revenue)
    eps_yoy_pct = _pct(latest_eps, prior_eps)
    latest_fcf = _num(latest_cash_flow.get("freeCashFlow"))
    fcf_margin_pct = (
        latest_fcf / latest_revenue * 100.0
        if latest_fcf is not None and latest_revenue not in (None, 0)
        else None
    )

    net_upgrades = int(_num(rec_trends.get("net_upgrades_30d"), 0) or 0)
    net_downgrades = int(_num(rec_trends.get("net_downgrades_30d"), 0) or 0)
    net_revision = net_upgrades - net_downgrades
    next_q_eps = _num(estimates.get("next_q_eps_estimate"))
    next_q_revenue = _num(estimates.get("next_q_revenue_estimate"))
    fy1_revenue = _num(estimates.get("fy1_revenue_estimate"))

    rsi = _num(technicals.get("rsi"))
    sma_50 = _num(technicals.get("sma_50"))
    price_vs_sma50_pct = _pct(current_price, sma_50)
    days_to_earnings = _num(earnings_context.get("days_to_earnings"))

    flags: list[str] = []
    positives: list[str] = []
    score = 50.0

    dcf_reliability = "unavailable"
    if dcf_value is not None and dcf_value > 0 and current_price:
        if abs((dcf_value - current_price) / current_price) > 0.70:
            dcf_reliability = "low"
            flags.append("DCF differs from market price by more than 70%; use as low-weight anchor")
            score -= 2
        else:
            dcf_reliability = "normal"
            if dcf_upside_pct is not None and dcf_upside_pct >= 15:
                positives.append(f"DCF implies {dcf_upside_pct:.1f}% upside")
                score += 8
            elif dcf_upside_pct is not None and dcf_upside_pct <= -20:
                flags.append(f"DCF implies {dcf_upside_pct:.1f}% downside")
                score -= 10
    elif dcf_value is not None and dcf_value <= 0:
        dcf_reliability = "not_applicable"
        flags.append("DCF value is non-positive; do not use it as a fair-value anchor")
        score -= 3

    if consensus_upside_pct is not None:
        if consensus_upside_pct >= 15:
            positives.append(f"Consensus target implies {consensus_upside_pct:.1f}% upside")
            score += 10
        elif consensus_upside_pct <= -5:
            flags.append(f"Consensus target is {abs(consensus_upside_pct):.1f}% below current price")
            score -= 12
    if low_target_downside_pct is not None and low_target_downside_pct <= -20:
        flags.append(f"Low target implies {abs(low_target_downside_pct):.1f}% downside")
        score -= 4
    if high_target_upside_pct is not None and consensus_upside_pct is not None:
        if high_target_upside_pct >= 50 and consensus_upside_pct < 10:
            flags.append("Upside case depends on high-end target rather than consensus")
            score -= 3

    if trailing_pe is not None and trailing_pe > 60:
        flags.append(f"High trailing P/E ({trailing_pe:.1f}x)")
        score -= 5
    if price_to_sales is not None and price_to_sales > 15:
        flags.append(f"High price/sales ({price_to_sales:.1f}x)")
        score -= 5
    if peg is not None:
        if peg <= 1.5:
            positives.append(f"PEG looks reasonable ({peg:.2f})")
            score += 4
        elif peg >= 3.0:
            flags.append(f"PEG suggests growth is expensive ({peg:.2f})")
            score -= 5
    if fcf_yield is not None:
        if fcf_yield >= 0.04:
            positives.append(f"Free-cash-flow yield is attractive ({fcf_yield:.1%})")
            score += 6
        elif fcf_yield < 0.015:
            flags.append(f"Low free-cash-flow yield ({fcf_yield:.1%})")
            score -= 4
    if roic is not None and roic >= 0.15:
        positives.append(f"ROIC is strong ({roic:.1%})")
        score += 5
    if debt_to_equity is not None and debt_to_equity > 2.0:
        flags.append(f"Balance sheet leverage elevated (D/E {debt_to_equity:.1f}x)")
        score -= 5

    if revenue_yoy_pct is not None:
        if revenue_yoy_pct >= 10:
            positives.append(f"Latest quarterly revenue grew {revenue_yoy_pct:.1f}% YoY")
            score += 5
        elif revenue_yoy_pct <= -5:
            flags.append(f"Latest quarterly revenue declined {abs(revenue_yoy_pct):.1f}% YoY")
            score -= 6
    if eps_yoy_pct is not None:
        if eps_yoy_pct >= 15:
            positives.append(f"Latest quarterly EPS grew {eps_yoy_pct:.1f}% YoY")
            score += 4
        elif eps_yoy_pct <= -10:
            flags.append(f"Latest quarterly EPS declined {abs(eps_yoy_pct):.1f}% YoY")
            score -= 5

    if net_revision > 0:
        positives.append(f"Net analyst revision trend is positive (+{net_revision})")
        score += min(6, 2 * net_revision)
    elif net_revision < 0:
        flags.append(f"Net analyst revision trend is negative ({net_revision})")
        score -= min(8, 3 * abs(net_revision))

    if days_to_earnings is not None and 0 <= days_to_earnings <= 7:
        flags.append(f"Earnings are within {int(days_to_earnings)} days")
        score -= 6
    if rsi is not None and rsi >= 70:
        flags.append(f"RSI is overbought ({rsi:.1f})")
        score -= 4
    if price_vs_sma50_pct is not None and price_vs_sma50_pct >= 20:
        flags.append(f"Price is {price_vs_sma50_pct:.1f}% above 50-day SMA")
        score -= 5

    expectation_risk_score = 0
    for flag in flags:
        if any(
            token in flag.lower()
            for token in (
                "below current",
                "downside",
                "overbought",
                "above 50-day",
                "earnings",
                "high",
                "negative",
                "expensive",
                "low free-cash-flow",
            )
        ):
            expectation_risk_score += 1
    if dcf_upside_pct is not None and dcf_upside_pct <= -30:
        expectation_risk_score += 1
    if consensus_upside_pct is not None and consensus_upside_pct < 5:
        expectation_risk_score += 1
    if expectation_risk_score >= 4:
        expectation_risk_level = "high"
    elif expectation_risk_score >= 2:
        expectation_risk_level = "moderate"
    else:
        expectation_risk_level = "low"

    valuation_score = _score_clamp(score)
    if valuation_score >= 68 and expectation_risk_level != "high":
        valuation_signal = "positive"
    elif valuation_score <= 42 or expectation_risk_level == "high":
        valuation_signal = "negative"
    else:
        valuation_signal = "neutral"

    summary_bits = []
    if consensus_upside_pct is not None:
        summary_bits.append(f"consensus upside {consensus_upside_pct:+.1f}%")
    if dcf_upside_pct is not None:
        summary_bits.append(f"DCF upside {dcf_upside_pct:+.1f}% ({dcf_reliability})")
    if trailing_pe is not None:
        summary_bits.append(f"P/E {trailing_pe:.1f}x")
    if fcf_yield is not None:
        summary_bits.append(f"FCF yield {fcf_yield:.1%}")
    summary = "; ".join(summary_bits) if summary_bits else "insufficient valuation anchors"

    return {
        "schema_version": 1,
        "current_price": current_price,
        "market_cap": market_cap,
        "sector": profile.get("sector"),
        "industry": profile.get("industry"),
        "valuation_score": valuation_score,
        "valuation_signal": valuation_signal,
        "expectation_risk_level": expectation_risk_level,
        "summary": summary,
        "dcf": {
            "value": dcf_value,
            "upside_pct": dcf_upside_pct,
            "reliability": dcf_reliability,
        },
        "analyst_targets": {
            "consensus": consensus_target,
            "median": median_target,
            "low": low_target,
            "high": high_target,
            "consensus_upside_pct": consensus_upside_pct,
            "median_upside_pct": median_upside_pct,
            "low_target_downside_pct": low_target_downside_pct,
            "high_target_upside_pct": high_target_upside_pct,
        },
        "multiples": {
            "trailing_pe": trailing_pe,
            "forward_pe": forward_pe,
            "peg": peg,
            "forward_peg": forward_peg,
            "price_to_sales": price_to_sales,
            "price_to_book": price_to_book,
            "fcf_yield": fcf_yield,
        },
        "quality": {
            "roic": roic,
            "roe": roe,
            "net_margin": net_margin,
            "current_ratio": current_ratio,
            "debt_to_equity": debt_to_equity,
            "fcf_margin_pct": fcf_margin_pct,
        },
        "growth": {
            "latest_revenue": latest_revenue,
            "revenue_yoy_pct": revenue_yoy_pct,
            "latest_eps": latest_eps,
            "eps_yoy_pct": eps_yoy_pct,
            "fy1_revenue_estimate": fy1_revenue,
            "next_q_revenue_estimate": next_q_revenue,
            "next_q_eps_estimate": next_q_eps,
        },
        "revision_trend": {
            "net_upgrades_30d": net_upgrades,
            "net_downgrades_30d": net_downgrades,
            "net_revision_30d": net_revision,
            "consensus": rec_trends.get("consensus"),
        },
        "timing_risk": {
            "rsi": rsi,
            "price_vs_sma50_pct": price_vs_sma50_pct,
            "days_to_earnings": days_to_earnings,
        },
        "positive_evidence": positives[:8],
        "risk_flags": flags[:10],
    }


def format_valuation_expectations(payload: dict[str, Any] | None) -> str:
    """Render a compact council prompt section."""
    if not payload:
        return "Valuation/expectations engine unavailable."
    lines = [
        "DETERMINISTIC VALUATION / EXPECTATIONS CHECK",
        f"Signal: {payload.get('valuation_signal', 'unknown')} | "
        f"Score: {payload.get('valuation_score', 'N/A')}/100 | "
        f"Expectation risk: {payload.get('expectation_risk_level', 'unknown')}",
        f"Summary: {payload.get('summary', 'insufficient valuation anchors')}",
    ]
    targets = payload.get("analyst_targets") or {}
    dcf = payload.get("dcf") or {}
    if targets:
        lines.append(
            "Targets: consensus ${} ({:+.1f}% vs price), low ${}, high ${}".format(
                _fmt(targets.get("consensus")),
                _num(targets.get("consensus_upside_pct"), 0.0) or 0.0,
                _fmt(targets.get("low")),
                _fmt(targets.get("high")),
            )
        )
    if dcf:
        upside = _num(dcf.get("upside_pct"))
        upside_text = "N/A" if upside is None else f"{upside:+.1f}%"
        lines.append(f"DCF: ${_fmt(dcf.get('value'))} ({upside_text}; reliability={dcf.get('reliability')})")
    positives = payload.get("positive_evidence") or []
    flags = payload.get("risk_flags") or []
    if positives:
        lines.append("Positive evidence: " + "; ".join(str(x) for x in positives[:4]))
    if flags:
        lines.append("Risk flags: " + "; ".join(str(x) for x in flags[:5]))
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    number = _num(value)
    if number is None:
        return "N/A"
    return f"{number:,.2f}"
