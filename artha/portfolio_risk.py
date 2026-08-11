"""Portfolio and factor-risk diagnostics for Artha candidate decisions."""
from __future__ import annotations

from typing import Any


SECTOR_BENCHMARKS: dict[str, str] = {
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
}


def sector_benchmark_for(sector: str | None, fallback: str = "SPY") -> str:
    """Return the canonical sector ETF benchmark for a company sector."""
    if not sector:
        return fallback
    return SECTOR_BENCHMARKS.get(str(sector).strip(), fallback)


def primary_market_benchmark_for(sector: str | None) -> str:
    """Use QQQ for tech/communication growth exposure, SPY otherwise."""
    value = str(sector or "").strip()
    if value in {"Technology", "Communication Services", "Consumer Cyclical", "Consumer Discretionary"}:
        return "QQQ"
    return "SPY"


def _num(value: Any, default: float = 0.0) -> float:
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


def _candidate_weight_pct(portfolio_state: dict[str, Any], config: Any, proposed_weight_pct: float | None) -> float:
    if proposed_weight_pct is not None:
        return max(0.0, float(proposed_weight_pct))
    total_nav = _num(portfolio_state.get("total_value"), 0.0)
    cash = _num(portfolio_state.get("cash_available"), 0.0)
    monthly = _num(portfolio_state.get("monthly_contribution"), 0.0)
    if total_nav <= 0:
        return 0.0
    budget_amount = min(cash, monthly) if monthly > 0 else cash
    budget_pct = budget_amount / total_nav * 100.0
    max_position_pct = _num(getattr(config, "MAX_POSITION_PCT", 0.20), 0.20) * 100.0
    exploration_pct = _num(getattr(config, "EXPLORATION_MAX_PER_POSITION_PCT", 0.05), 0.05) * 100.0
    return max(0.0, min(max_position_pct, exploration_pct, budget_pct))


def evaluate_projected_sector_limit(
    *,
    ticker: str,
    sector: str,
    portfolio_state: dict[str, Any],
    proposed_notional: float,
    max_sector_pct: float,
) -> dict[str, Any]:
    """Evaluate exact post-order sector exposure for new buys and ADDs."""
    positions = portfolio_state.get("positions") or []
    total_nav = _num(portfolio_state.get("total_value"), 0.0)
    normalized_sector = str(sector or "").strip()
    notional = max(0.0, _num(proposed_notional, 0.0))
    reasons: list[str] = []
    if not normalized_sector:
        reasons.append("Candidate sector classification is missing.")
    if total_nav <= 0:
        reasons.append("Current portfolio NAV is unavailable for sector-limit calculation.")
    if notional <= 0:
        reasons.append("Proposed buy notional is unavailable for sector-limit calculation.")

    current_sector_value = sum(
        _num(position.get("market_value"), 0.0)
        for position in positions
        if isinstance(position, dict)
        and str(position.get("sector") or "").strip() == normalized_sector
    )
    current_pct = current_sector_value / total_nav if total_nav > 0 else None
    projected_value = current_sector_value + notional
    projected_pct = projected_value / total_nav if total_nav > 0 else None
    headroom = max(0.0, max_sector_pct * total_nav - current_sector_value) if total_nav > 0 else 0.0
    if projected_pct is not None and projected_pct > max_sector_pct + 0.000001:
        reasons.append(
            f"Proposed {ticker.upper()} buy would raise {normalized_sector} exposure to "
            f"{projected_pct:.1%}, above the {max_sector_pct:.0%} hard limit."
        )
    return {
        "passed": not reasons,
        "status": "PASS" if not reasons else "BLOCKED",
        "ticker": str(ticker or "").upper(),
        "sector": normalized_sector,
        "proposed_notional": round(notional, 4),
        "current_sector_value": round(current_sector_value, 4),
        "projected_sector_value": round(projected_value, 4),
        "current_sector_pct": round(current_pct, 6) if current_pct is not None else None,
        "projected_sector_pct": round(projected_pct, 6) if projected_pct is not None else None,
        "max_sector_pct": max_sector_pct,
        "headroom_dollars": round(headroom, 4),
        "is_add": any(
            isinstance(position, dict)
            and str(position.get("ticker") or "").upper() == str(ticker or "").upper()
            for position in positions
        ),
        "reasons": reasons,
    }


def build_portfolio_factor_risk(
    ticker: str,
    stock_data: dict[str, Any],
    portfolio_state: dict[str, Any],
    config: Any,
    proposed_weight_pct: float | None = None,
) -> dict[str, Any]:
    """Build deterministic portfolio/factor-risk context for a new candidate."""
    profile = stock_data.get("profile") or {}
    yf_quote = stock_data.get("yf_quote") or {}
    quote = stock_data.get("quote") or {}
    valuation = stock_data.get("valuation_expectations") or {}
    sector = str(profile.get("sector") or yf_quote.get("sector") or valuation.get("sector") or "").strip()
    industry = str(profile.get("industry") or yf_quote.get("industry") or valuation.get("industry") or "").strip()
    beta = _num(profile.get("beta"), _num(yf_quote.get("beta"), 1.0))
    market_cap = _num(quote.get("marketCap"), _num(profile.get("mktCap") or profile.get("marketCap"), 0.0))

    positions = portfolio_state.get("positions") or []
    if not isinstance(positions, list):
        positions = []
    total_nav = _num(portfolio_state.get("total_value"), 0.0)
    total_invested = _num(portfolio_state.get("total_holdings_value"), 0.0)
    if total_invested <= 0:
        total_invested = sum(
            _num(pos.get("market_value"), 0.0)
            for pos in positions
            if isinstance(pos, dict)
        )
    candidate_weight = _candidate_weight_pct(portfolio_state, config, proposed_weight_pct)
    candidate_notional = total_nav * candidate_weight / 100.0 if total_nav > 0 else 0.0
    held_tickers = {str(p.get("ticker", "")).upper() for p in positions if isinstance(p, dict)}
    is_existing_position = str(ticker or "").upper() in held_tickers
    existing_ticker_value = sum(
        _num(pos.get("market_value"), 0.0)
        for pos in positions
        if isinstance(pos, dict) and str(pos.get("ticker") or "").upper() == str(ticker or "").upper()
    )

    sector_value = 0.0
    sector_weights: dict[str, float] = {}
    sector_values: dict[str, float] = {}
    for pos in positions:
        if not isinstance(pos, dict):
            continue
        pos_sector = str(pos.get("sector") or "unknown").strip()
        weight = _num(pos.get("weight_pct"), 0.0)
        market_value = _num(pos.get("market_value"), 0.0)
        sector_weights[pos_sector] = sector_weights.get(pos_sector, 0.0) + weight
        sector_values[pos_sector] = sector_values.get(pos_sector, 0.0) + market_value
        if sector and pos_sector == sector:
            sector_value += market_value

    current_sector_pct = (sector_value / total_nav * 100.0) if total_nav > 0 else 0.0
    after_sector_pct = current_sector_pct + candidate_weight
    current_sector_invested_pct = (
        sector_value / total_invested * 100.0 if total_invested > 0 else 0.0
    )
    projected_invested = total_invested + candidate_notional
    projected_sector_value = sector_value + candidate_notional
    sector_after_invested_pct = (
        projected_sector_value / projected_invested * 100.0
        if projected_invested > 0
        else 0.0
    )
    existing_ticker_pct = existing_ticker_value / total_nav * 100.0 if total_nav > 0 else 0.0
    concentration_after_pct = max(
        _num(portfolio_state.get("concentration_pct"), 0.0),
        existing_ticker_pct + candidate_weight,
    )
    available_slots = max(0, int(_num(getattr(config, "MAX_CONCURRENT_POSITIONS", 20), 20)) - len(positions))
    max_sector_pct = _num(getattr(config, "MAX_SECTOR_PCT", 0.30), 0.30) * 100.0
    max_position_pct = _num(getattr(config, "MAX_POSITION_PCT", 0.20), 0.20) * 100.0

    flags: list[str] = []
    positives: list[str] = []
    risk_score = 25.0

    if is_existing_position:
        flags.append("Ticker is already held; decision should be ADD/HOLD/trim-aware, not new-position sizing")
        risk_score += 8
    elif available_slots <= 0:
        flags.append("No available position slots for a new holding")
        risk_score += 20
    if after_sector_pct > max_sector_pct:
        flags.append(f"Candidate would push sector exposure ({sector or 'unknown'}) to {after_sector_pct:.1f}% (limit {max_sector_pct:.1f}%)")
        risk_score += 40
    elif sector and after_sector_pct <= max_sector_pct * 0.6:
        positives.append(f"Sector exposure remains moderate after candidate ({after_sector_pct:.1f}%)")
        risk_score -= 5
    if candidate_weight > max_position_pct:
        flags.append(f"Candidate weight {candidate_weight:.1f}% exceeds max position {max_position_pct:.1f}%")
        risk_score += 25
    if beta >= 1.7:
        flags.append(f"High-beta candidate (beta {beta:.2f})")
        risk_score += 8
    elif beta <= 0.9:
        positives.append(f"Beta is below market ({beta:.2f})")
        risk_score -= 3
    if market_cap and market_cap < 2_000_000_000:
        flags.append(f"Small-cap execution/liquidity risk (market cap ${market_cap/1e9:.1f}B)")
        risk_score += 8
    if candidate_weight <= 0:
        flags.append("No deployable candidate weight under current cash/budget constraints")
        risk_score += 10

    if risk_score >= 65:
        risk_level = "high"
    elif risk_score >= 42:
        risk_level = "moderate"
    else:
        risk_level = "low"

    sector_benchmark = sector_benchmark_for(sector)
    market_benchmark = primary_market_benchmark_for(sector)
    return {
        "schema_version": 1,
        "ticker": str(ticker or "").upper(),
        "risk_score": round(max(0.0, min(100.0, risk_score)), 1),
        "risk_level": risk_level,
        "sector": sector,
        "industry": industry,
        "beta": beta,
        "market_cap": market_cap,
        "candidate_weight_pct": round(candidate_weight, 2),
        "current_sector_pct": round(current_sector_pct, 2),
        "sector_after_candidate_pct": round(after_sector_pct, 2),
        "current_sector_invested_pct": round(current_sector_invested_pct, 2),
        "sector_after_candidate_invested_pct": round(sector_after_invested_pct, 2),
        "concentration_after_candidate_pct": round(concentration_after_pct, 2),
        "available_slots": available_slots,
        "is_existing_position": is_existing_position,
        "sector_weights": {k: round(v, 2) for k, v in sorted(sector_weights.items())},
        "sector_weights_invested": {
            key: round(value / total_invested * 100.0, 2) if total_invested > 0 else 0.0
            for key, value in sorted(sector_values.items())
        },
        "market_benchmark_ticker": market_benchmark,
        "sector_benchmark_ticker": sector_benchmark,
        "risk_flags": flags[:8],
        "positive_evidence": positives[:6],
    }


def format_portfolio_factor_risk(payload: dict[str, Any] | None) -> str:
    """Render compact council context."""
    if not payload:
        return "Portfolio/factor risk engine unavailable."
    lines = [
        "DETERMINISTIC PORTFOLIO / FACTOR RISK CHECK",
        f"Risk: {payload.get('risk_level', 'unknown')} | score {payload.get('risk_score', 'N/A')}/100",
        (
            f"Candidate sector: {payload.get('sector') or 'unknown'} | "
            f"candidate weight: {payload.get('candidate_weight_pct', 0):.1f}% | "
            f"sector after candidate: {payload.get('sector_after_candidate_pct', 0):.1f}% NAV / "
            f"{payload.get('sector_after_candidate_invested_pct', 0):.1f}% invested"
        ),
        (
            f"Benchmarks: market={payload.get('market_benchmark_ticker', 'SPY')} | "
            f"sector={payload.get('sector_benchmark_ticker', 'SPY')}"
        ),
    ]
    flags = payload.get("risk_flags") or []
    positives = payload.get("positive_evidence") or []
    if positives:
        lines.append("Positive evidence: " + "; ".join(str(x) for x in positives[:3]))
    if flags:
        lines.append("Risk flags: " + "; ".join(str(x) for x in flags[:5]))
    return "\n".join(lines)
