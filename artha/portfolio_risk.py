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
    candidate_weight = _candidate_weight_pct(portfolio_state, config, proposed_weight_pct)
    held_tickers = {str(p.get("ticker", "")).upper() for p in positions if isinstance(p, dict)}
    is_existing_position = str(ticker or "").upper() in held_tickers

    sector_value = 0.0
    sector_weights: dict[str, float] = {}
    for pos in positions:
        if not isinstance(pos, dict):
            continue
        pos_sector = str(pos.get("sector") or pos.get("asset_type") or "unknown").strip()
        weight = _num(pos.get("weight_pct"), 0.0)
        sector_weights[pos_sector] = sector_weights.get(pos_sector, 0.0) + weight
        if sector and pos_sector == sector:
            sector_value += _num(pos.get("market_value"), 0.0)

    current_sector_pct = (sector_value / total_nav * 100.0) if total_nav > 0 else 0.0
    after_sector_pct = current_sector_pct if is_existing_position else current_sector_pct + candidate_weight
    concentration_after_pct = max(_num(portfolio_state.get("concentration_pct"), 0.0), candidate_weight)
    available_slots = max(0, int(_num(getattr(config, "MAX_CONCURRENT_POSITIONS", 6), 6)) - len(positions))
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
        "concentration_after_candidate_pct": round(concentration_after_pct, 2),
        "available_slots": available_slots,
        "is_existing_position": is_existing_position,
        "sector_weights": {k: round(v, 2) for k, v in sorted(sector_weights.items())},
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
            f"sector after candidate: {payload.get('sector_after_candidate_pct', 0):.1f}%"
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
