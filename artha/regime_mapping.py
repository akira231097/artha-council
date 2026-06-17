"""Regime Mapping — Static regime-to-sector/ETF configuration.

Based on Ray Dalio's Four-Quadrant framework (Growth x Inflation) with
event-driven overlays for crisis scenarios. Deterministic mapping — no LLM
needed. Pure config that drives candidate generation.

Research sources:
- Bridgewater All Weather (Dalio, 1996)
- 42 Macro (Darius Dale) quadrant framework
- Hedgeye (Keith McCullough) regime model
- FactSet regime mapping (CLI + ITS signals, 2025)
- StockTrends rotation analysis (March 2026)
"""
from __future__ import annotations

from typing import Dict, List, Any


# ---------------------------------------------------------------------------
# Regime Taxonomy
# ---------------------------------------------------------------------------

REGIME_TAXONOMY: Dict[str, Dict[str, Any]] = {

    # ===== BASE ECONOMIC REGIMES (Dalio Quadrants) =====

    "goldilocks": {
        "label": "Goldilocks — Rising Growth, Falling Inflation",
        "quadrant": "Q1",
        "description": (
            "Steady growth with contained or falling inflation. Risk-on environment. "
            "Equities and corporate credit thrive. Falling yields suppress volatility. "
            "The regime most common over the last four decades."
        ),
        "beneficiary_etfs": ["VOO", "QQQ", "VTI", "SCHD", "IWM"],
        "beneficiary_stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
        "avoid_etfs": ["DBC", "USO"],
        "avoid_stocks": [],
        "avoid_sectors": ["commodities"],
        "typical_duration": "months_to_quarters",
        "historical_examples": ["2019 (SPX +28%)", "Most of 2024", "2013-2017 bull"],
    },

    "reflation": {
        "label": "Reflation — Rising Growth, Rising Inflation",
        "quadrant": "Q2",
        "description": (
            "Hot economy with rising prices but market not yet worried about tightening. "
            "Equities, commodities, energy, crypto, and cyclicals all perform well. "
            "Animal spirits dominate. Usually follows Q1 after stimulative policy."
        ),
        "beneficiary_etfs": ["XLE", "XLI", "XLF", "DBC", "EEM", "QQQ", "IWM"],
        "beneficiary_stocks": ["XOM", "CVX", "CAT", "DE", "JPM", "GS"],
        "avoid_etfs": ["TLT", "IEF"],
        "avoid_stocks": [],
        "avoid_sectors": ["long_duration_bonds"],
        "typical_duration": "months",
        "historical_examples": ["2017 (tax cuts)", "2021 (post-COVID stimulus)"],
    },

    "stagflation": {
        "label": "Stagflation — Falling Growth, Rising Inflation",
        "quadrant": "Q3",
        "description": (
            "Weak economy with rising prices. The worst environment for most assets. "
            "Tops form, selloffs turn violent, volatility explodes. Energy and "
            "commodities may still do well. Bonds get hurt by inflation."
        ),
        "beneficiary_etfs": ["XLE", "GLD", "XLP", "XLU", "USO", "DBC"],
        "beneficiary_stocks": ["XOM", "CVX", "PG", "KO", "JNJ", "NEE", "COP"],
        "avoid_etfs": ["QQQ", "IWM", "XLY", "TLT"],
        "avoid_stocks": ["speculative_growth"],
        "avoid_sectors": ["growth", "discretionary", "speculative", "long_duration_bonds"],
        "typical_duration": "weeks_to_months",
        "historical_examples": ["2022 (inflation + rate hikes)", "1970s oil crisis"],
    },

    "risk_off": {
        "label": "Risk-Off / Recession — Falling Growth, Falling Inflation",
        "quadrant": "Q4",
        "description": (
            "Flight to safety. Growth slowing and inflation easing. Bonds rally as "
            "rate cuts expected. Defensive sectors outperform. Classic demand-driven "
            "recession where government bonds perform best."
        ),
        "beneficiary_etfs": ["TLT", "IEF", "XLV", "XLP", "XLU", "GLD"],
        "beneficiary_stocks": ["JNJ", "PG", "KO", "UNH", "WMT", "COST"],
        "avoid_etfs": ["XLE", "XLI", "IWM", "EEM"],
        "avoid_stocks": [],
        "avoid_sectors": ["cyclicals", "energy", "small_caps", "emerging_markets"],
        "typical_duration": "weeks_to_months",
        "historical_examples": ["2008 GFC", "March 2020 COVID crash", "2001 recession"],
    },

    # ===== EVENT-DRIVEN OVERLAYS =====

    "geopolitical_energy_shock": {
        "label": "Geopolitical Energy Shock",
        "quadrant": "overlay",
        "description": (
            "Military conflict or supply disruption driving oil and energy prices higher. "
            "Capital rotates into energy, hard assets, defense, and safe havens. "
            "Airlines and fuel-sensitive sectors get pressured."
        ),
        "beneficiary_etfs": ["XLE", "USO", "ITA", "GLD", "XAR"],
        "beneficiary_stocks": [
            "XOM", "CVX", "COP", "OXY", "HAL", "SLB",  # Energy
            "RTX", "LMT", "NOC", "GD", "HII", "LHX",   # Defense
        ],
        "avoid_etfs": ["JETS"],
        "avoid_stocks": ["UAL", "AAL", "DAL", "CCL", "RCL"],
        "avoid_sectors": ["airlines", "cruise_lines", "transportation"],
        "evidence_signals": [
            "oil_surge_gt_3pct",
            "xle_outperforming_spy_gt_2pct",
            "gold_bid",
            "defense_etf_rally",
            "geopolitical_headlines",
        ],
        "typical_duration": "days_to_weeks",
        "historical_examples": [
            "2022 Russia-Ukraine (XLE +64% YTD)",
            "March 2026 Iran conflict",
            "1990 Gulf War",
        ],
    },

    "financial_stress": {
        "label": "Financial Stress / Credit Concern",
        "quadrant": "overlay",
        "description": (
            "Banking stress, credit tightening, or systemic risk fears. "
            "Capital flees to safety. Financials and real estate get hit hardest."
        ),
        "beneficiary_etfs": ["XLU", "XLP", "TLT", "GLD"],
        "beneficiary_stocks": ["NEE", "DUK", "PG", "KO", "JNJ"],
        "avoid_etfs": ["XLF", "XLRE", "KRE"],
        "avoid_stocks": [],
        "avoid_sectors": ["banks", "regional_banks", "real_estate"],
        "evidence_signals": [
            "xlf_underperforming_spy_gt_2pct",
            "vix_above_30",
            "financial_stress_headlines",
        ],
        "typical_duration": "weeks_to_months",
        "historical_examples": ["March 2023 SVB crisis", "2008 GFC"],
    },

    "ai_tech_momentum": {
        "label": "AI / Technology Momentum",
        "quadrant": "overlay",
        "description": (
            "AI-driven investment cycle accelerating. Semiconductor and cloud "
            "infrastructure names lead. Capex announcements drive sentiment."
        ),
        "beneficiary_etfs": ["SMH", "QQQ", "SOXX", "XLK"],
        "beneficiary_stocks": [
            "NVDA", "AMD", "AVGO", "MSFT", "GOOGL", "AMZN", "TSM", "ARM",
            "MRVL", "CRWD", "PLTR",
        ],
        "avoid_etfs": [],
        "avoid_stocks": [],
        "avoid_sectors": ["legacy_tech", "traditional_media"],
        "evidence_signals": [
            "smh_outperforming_spy_gt_2pct",
            "ai_capex_headlines",
            "semi_stocks_rallying",
        ],
        "typical_duration": "months_to_quarters",
        "historical_examples": ["2023-2024 AI boom", "Late 2025 Gemini/GPT cycle"],
    },

    "consumer_weakening": {
        "label": "Consumer Weakening",
        "quadrant": "overlay",
        "description": (
            "Consumer spending deteriorating. Rising delinquencies, declining "
            "savings rate, weak retail data. Staples and healthcare benefit."
        ),
        "beneficiary_etfs": ["XLP", "XLV"],
        "beneficiary_stocks": ["WMT", "COST", "PG", "KO", "UNH", "JNJ"],
        "avoid_etfs": ["XLY", "XRT"],
        "avoid_stocks": ["SBUX", "NKE", "MCD"],
        "avoid_sectors": ["discretionary", "restaurants", "luxury", "housing"],
        "evidence_signals": [
            "xly_underperforming_xlp_gt_2pct",
            "consumer_stress_headlines",
            "unemployment_rising",
        ],
        "typical_duration": "months",
        "historical_examples": ["Late 2007", "Early 2020"],
    },

    "rate_cut_cycle": {
        "label": "Interest Rate Cut Cycle",
        "quadrant": "overlay",
        "description": (
            "Fed actively cutting rates or signaling dovish pivot. "
            "Rate-sensitive sectors benefit: REITs, utilities, growth stocks."
        ),
        "beneficiary_etfs": ["XLRE", "XLU", "TLT", "QQQ", "VNQ"],
        "beneficiary_stocks": ["growth_beneficiaries"],
        "avoid_etfs": [],
        "avoid_stocks": [],
        "avoid_sectors": [],
        "evidence_signals": [
            "treasury_yields_falling",
            "fed_dovish_headlines",
            "rate_futures_repricing",
        ],
        "typical_duration": "months_to_quarters",
        "historical_examples": ["2019 Fed pivot (3 cuts)", "Sept 2024 cut cycle start"],
    },

    "trade_war_tariff": {
        "label": "Trade War / Tariff Shock",
        "quadrant": "overlay",
        "description": (
            "Escalating tariffs or trade restrictions. Domestic-focused companies "
            "benefit while import-dependent and EM exposed names suffer."
        ),
        "beneficiary_etfs": ["XLP", "XLU", "GLD"],
        "beneficiary_stocks": ["domestic_focused"],
        "avoid_etfs": ["EEM", "FXI"],
        "avoid_stocks": [],
        "avoid_sectors": ["import_dependent", "emerging_markets"],
        "evidence_signals": [
            "tariff_headlines",
            "eem_underperforming",
            "uup_rallying",
        ],
        "typical_duration": "weeks_to_months",
        "historical_examples": ["2018-2019 US-China trade war", "2025 Trump tariffs"],
    },

    "defense_repricing": {
        "label": "Defense Spending Repricing",
        "quadrant": "overlay",
        "description": (
            "Sustained increase in defense budgets, NATO spending commitments, "
            "or military preparedness. Multi-quarter beneficiary cycle."
        ),
        "beneficiary_etfs": ["ITA", "XAR"],
        "beneficiary_stocks": ["RTX", "LMT", "NOC", "GD", "BA", "HII", "LHX"],
        "avoid_etfs": [],
        "avoid_stocks": [],
        "avoid_sectors": [],
        "evidence_signals": [
            "defense_budget_headlines",
            "ita_outperforming_spy",
            "military_escalation_headlines",
        ],
        "typical_duration": "months_to_quarters",
        "historical_examples": ["Post-2022 NATO rearmament", "March 2026 Iran conflict"],
    },
}


# ---------------------------------------------------------------------------
# Beginner Suitability Rules
# ---------------------------------------------------------------------------

BEGINNER_RULES = {
    # =====================================================================
    # Sarath's actual setup as of 2026-06-02:
    #   $350/month → FXAIX (self-managed core, auto-recurring on Fidelity)
    #   $350/month → Artha council satellite stock-picking budget
    #
    # Artha can recommend individual-stock tactical opportunities within the
    # satellite budget, while preserving FXAIX as the separate core.
    # Artha should NOT recommend VOO/VTI/FXAIX — that's already handled.
    # =====================================================================

    # Budget
    "monthly_budget": 350,             # Artha-managed satellite budget
    "core_already_covered": True,      # Sarath has $350/month FXAIX separately
    "core_fund": "FXAIX",             # For reference only — we don't manage this

    # Allocation within the satellite budget
    "max_single_position_pct": 0.15,
    "max_single_position_usd": 350,
    "max_positions_per_month": 2,

    # Confidence thresholds
    "min_regime_confidence": 0.55,     # Don't generate picks from regime below 55%
    "min_overlay_confidence": 0.60,    # Don't act on event overlays below 60%

    # Persistence requirements
    "min_persistence_event_alert": 0,  # No persistence needed for high-confidence alerts
    "min_persistence_regular": 1,      # Prefer 1+ day persistence for Tue/Fri reports

    # Vehicle preferences
    "prefer_etfs_over_stocks": True,   # For beginners, sector ETFs > single names
    "etf_priority_order": [            # When multiple options, prefer this order
        "sector_etf",                   # XLE, XLK, etc.
        "thematic_etf",                 # ITA, SMH, etc.
        "single_stock",                 # Individual stocks
    ],

    # Excluded vehicles (not suitable for beginner)
    "excluded_tickers": [
        "USO", "UCO", "SCO",           # Oil futures ETFs (contango decay)
        "BOIL", "KOLD",                 # Natural gas leveraged
        "UVXY", "SVXY",                 # Volatility products
        "TQQQ", "SQQQ",                # Leveraged/inverse
    ],
}


# ---------------------------------------------------------------------------
# Regime → Candidate Generation Helpers
# ---------------------------------------------------------------------------

def get_regime_candidates(regime_type: str, max_etfs: int = 3, max_stocks: int = 4) -> dict:
    """Get candidate tickers for a given regime type.

    Returns:
        dict with keys: etfs, stocks, avoid_etfs, avoid_stocks
    """
    regime = REGIME_TAXONOMY.get(regime_type, {})
    return {
        "etfs": regime.get("beneficiary_etfs", [])[:max_etfs],
        "stocks": regime.get("beneficiary_stocks", [])[:max_stocks],
        "avoid_etfs": regime.get("avoid_etfs", []),
        "avoid_stocks": regime.get("avoid_stocks", []),
        "avoid_sectors": regime.get("avoid_sectors", []),
    }


def get_all_regime_types() -> list[str]:
    """Return all available regime type keys."""
    return list(REGIME_TAXONOMY.keys())


def get_base_regimes() -> list[str]:
    """Return only base economic regime keys (Dalio quadrants)."""
    return [k for k, v in REGIME_TAXONOMY.items() if v.get("quadrant") != "overlay"]


def get_overlay_regimes() -> list[str]:
    """Return only event-driven overlay regime keys."""
    return [k for k, v in REGIME_TAXONOMY.items() if v.get("quadrant") == "overlay"]


def format_regime_taxonomy_for_prompt() -> str:
    """Format the full taxonomy as text for LLM prompts."""
    lines = ["AVAILABLE REGIME TYPES:", ""]

    lines.append("=== BASE ECONOMIC REGIMES (Dalio Quadrants) ===")
    for key in get_base_regimes():
        r = REGIME_TAXONOMY[key]
        lines.append(f"\n{key}:")
        lines.append(f"  Label: {r['label']}")
        lines.append(f"  Description: {r['description']}")
        lines.append(f"  Beneficiaries: {', '.join(r.get('beneficiary_etfs', []))}")
        lines.append(f"  Avoid: {', '.join(r.get('avoid_sectors', []))}")

    lines.append("\n=== EVENT-DRIVEN OVERLAYS ===")
    for key in get_overlay_regimes():
        r = REGIME_TAXONOMY[key]
        lines.append(f"\n{key}:")
        lines.append(f"  Label: {r['label']}")
        lines.append(f"  Description: {r['description']}")
        lines.append(f"  Evidence signals: {', '.join(r.get('evidence_signals', []))}")
        lines.append(f"  Beneficiaries: {', '.join(r.get('beneficiary_etfs', []))}")
        if r.get("avoid_sectors"):
            lines.append(f"  Avoid: {', '.join(r['avoid_sectors'])}")

    return "\n".join(lines)
