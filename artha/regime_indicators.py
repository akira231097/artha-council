"""Regime Indicators — Compute derived market features for regime classification.

Fetches ETF returns, sector performance, safe-haven signals, VIX, and news
headlines to produce a structured snapshot that regime analysts can evaluate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import yfinance as yf
import requests

from .config import Config
from .search import search_web

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ETF universe for regime detection
# ---------------------------------------------------------------------------

BROAD_MARKET_ETFS = ["SPY", "QQQ", "IWM", "DIA"]
SAFE_HAVEN_ETFS = ["GLD", "TLT", "UUP"]
CRISIS_ETFS = ["USO"]
VOLATILITY_TICKERS = ["^VIX"]

SECTOR_ETFS = {
    "Energy": "XLE",
    "Financials": "XLF",
    "Technology": "XLK",
    "Industrials": "XLI",
    "Healthcare": "XLV",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Real Estate": "XLRE",
}

THEMATIC_ETFS = {
    "Aerospace & Defense": "ITA",
    "Semiconductors": "SMH",
}

ALL_REGIME_TICKERS = (
    BROAD_MARKET_ETFS
    + SAFE_HAVEN_ETFS
    + CRISIS_ETFS
    + VOLATILITY_TICKERS
    + list(SECTOR_ETFS.values())
    + list(THEMATIC_ETFS.values())
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TickerReturn:
    """Return data for a single ticker over multiple horizons."""
    ticker: str
    price: Optional[float] = None
    return_1d: Optional[float] = None
    return_5d: Optional[float] = None
    return_1m: Optional[float] = None


@dataclass
class RegimeIndicators:
    """Structured market snapshot for regime classification."""

    as_of: str = ""

    # Broad market returns
    broad_market: Dict[str, TickerReturn] = field(default_factory=dict)

    # Safe haven returns
    safe_havens: Dict[str, TickerReturn] = field(default_factory=dict)

    # Crisis indicators
    oil: Optional[TickerReturn] = None
    vix_level: Optional[float] = None
    vix_1d_change: Optional[float] = None

    # Sector ETF returns (1D)
    sector_returns: Dict[str, float] = field(default_factory=dict)

    # Sector relative to SPY (outperformance)
    sector_relative: Dict[str, float] = field(default_factory=dict)

    # Thematic ETF returns
    thematic_returns: Dict[str, TickerReturn] = field(default_factory=dict)

    # FRED macro data
    fed_funds_rate: Optional[float] = None
    treasury_10y: Optional[float] = None
    treasury_2y: Optional[float] = None
    yield_curve_spread: Optional[float] = None
    cpi_latest: Optional[float] = None
    unemployment: Optional[float] = None

    # Sentiment
    fear_greed_value: Optional[int] = None
    fear_greed_label: str = ""

    # Market breadth
    pct_stocks_up_1d: Optional[float] = None

    # News headlines
    top_headlines: List[str] = field(default_factory=list)

    # Credit & Macro Stress (populated from crisis_data when provided)
    hy_credit_spread: Optional[float] = None     # ICE BofA HY OAS (bps-equivalent, e.g. 3.0 = 300bps)
    ig_credit_spread: Optional[float] = None     # ICE BofA IG OAS
    dxy: Optional[float] = None                  # US Dollar Index
    spy_drawdown_from_52w_high: Optional[float] = None  # Fraction, e.g. -0.10 = -10%
    initial_jobless_claims: Optional[float] = None      # Weekly claims (raw count)
    oil_price_wti: Optional[float] = None        # WTI crude USD/barrel

    # Event risk (populated from EconomicCalendar when available)
    event_risk_state: str = "none"               # none|pre_major_24h|same_day_major|post_major_24h
    event_risk_next_major: Optional[str] = None  # Description of next major event

    def to_prompt_text(self) -> str:
        """Format indicators as structured text for LLM consumption."""
        lines = [f"MARKET DATA SNAPSHOT — {self.as_of}", ""]

        # Broad market
        lines.append("BROAD MARKET (1D / 5D / 1M returns):")
        for name, tr in self.broad_market.items():
            r1 = f"{tr.return_1d:+.2f}%" if tr.return_1d is not None else "N/A"
            r5 = f"{tr.return_5d:+.2f}%" if tr.return_5d is not None else "N/A"
            r1m = f"{tr.return_1m:+.2f}%" if tr.return_1m is not None else "N/A"
            price_str = f"${tr.price:,.2f}" if tr.price is not None else ""
            lines.append(f"  {name} {price_str}: {r1} / {r5} / {r1m}")

        # Safe havens
        lines.append("\nSAFE HAVEN ASSETS:")
        for name, tr in self.safe_havens.items():
            r1 = f"{tr.return_1d:+.2f}%" if tr.return_1d is not None else "N/A"
            r5 = f"{tr.return_5d:+.2f}%" if tr.return_5d is not None else "N/A"
            lines.append(f"  {name}: {r1} (5D: {r5})")

        # Oil & VIX
        lines.append("\nCRISIS GAUGES:")
        if self.oil:
            r1 = f"{self.oil.return_1d:+.2f}%" if self.oil.return_1d is not None else "N/A"
            r5 = f"{self.oil.return_5d:+.2f}%" if self.oil.return_5d is not None else "N/A"
            lines.append(f"  Oil (USO): {r1} (5D: {r5})")
        if self.vix_level is not None:
            vix_chg = f"{self.vix_1d_change:+.1f}pts" if self.vix_1d_change is not None else ""
            lines.append(f"  VIX: {self.vix_level:.1f} {vix_chg}")

        # Sector performance
        lines.append("\nSECTOR PERFORMANCE (1D):")
        sorted_sectors = sorted(self.sector_returns.items(), key=lambda x: x[1], reverse=True)
        spy_1d = self.broad_market.get("SPY", TickerReturn("SPY")).return_1d or 0
        for name, ret in sorted_sectors:
            rel = ret - spy_1d
            arrow = "▲" if rel > 0.2 else "▼" if rel < -0.2 else "→"
            lines.append(f"  {name}: {ret:+.2f}% ({arrow} {rel:+.2f}% vs SPY)")

        # Thematic
        if self.thematic_returns:
            lines.append("\nTHEMATIC ETFs:")
            for name, tr in self.thematic_returns.items():
                r1 = f"{tr.return_1d:+.2f}%" if tr.return_1d is not None else "N/A"
                lines.append(f"  {name}: {r1}")

        # Macro
        lines.append("\nMACRO INDICATORS:")
        if self.fed_funds_rate is not None:
            lines.append(f"  Fed Funds Rate: {self.fed_funds_rate:.2f}%")
        if self.treasury_10y is not None:
            lines.append(f"  10Y Treasury: {self.treasury_10y:.2f}%")
        if self.treasury_2y is not None:
            lines.append(f"  2Y Treasury: {self.treasury_2y:.2f}%")
        if self.yield_curve_spread is not None:
            lines.append(f"  Yield Curve (10Y-2Y): {self.yield_curve_spread:+.2f}%")
        if self.cpi_latest is not None:
            lines.append(f"  CPI (latest): {self.cpi_latest:.1f}%")
        if self.unemployment is not None:
            lines.append(f"  Unemployment: {self.unemployment:.1f}%")

        # Credit & Macro Stress
        stress_fields = [
            ("hy_credit_spread", self.hy_credit_spread, "HY Credit Spread (OAS)", "{:.2f}% (~{:.0f}bps)", lambda v: (v, v * 100)),
            ("ig_credit_spread", self.ig_credit_spread, "IG Credit Spread (OAS)", "{:.2f}% (~{:.0f}bps)", lambda v: (v, v * 100)),
            ("dxy", self.dxy, "US Dollar Index (DXY)", "{:.2f}", lambda v: (v,)),
            ("spy_drawdown_from_52w_high", self.spy_drawdown_from_52w_high, "SPY Drawdown from 52W High", "{:+.1%}", lambda v: (v,)),
            ("initial_jobless_claims", self.initial_jobless_claims, "Initial Jobless Claims", "{:,.0f}", lambda v: (v,)),
            ("oil_price_wti", self.oil_price_wti, "WTI Crude Oil", "${:.2f}/bbl", lambda v: (v,)),
        ]
        has_stress = any(getattr(self, f) is not None for f, *_ in stress_fields)
        if has_stress:
            lines.append("\nCREDIT & MACRO STRESS:")
            for attr, val, label, fmt, args_fn in stress_fields:
                if val is not None:
                    try:
                        lines.append(f"  {label}: {fmt.format(*args_fn(val))}")
                    except Exception:
                        lines.append(f"  {label}: {val}")

        # Event Risk
        if self.event_risk_state and self.event_risk_state != "none":
            lines.append(f"\nEVENT RISK: {self.event_risk_state.upper().replace('_', ' ')}")
            if self.event_risk_next_major:
                lines.append(f"  Next Major Event: {self.event_risk_next_major}")

        # Sentiment
        lines.append("\nSENTIMENT:")
        if self.fear_greed_value is not None:
            lines.append(f"  Fear & Greed Index: {self.fear_greed_value} ({self.fear_greed_label})")
        if self.pct_stocks_up_1d is not None:
            lines.append(f"  Market Breadth: {self.pct_stocks_up_1d:.0f}% of watched stocks up today")

        # Headlines
        if self.top_headlines:
            lines.append(f"\nTOP MARKET HEADLINES (last 48h):")
            for i, h in enumerate(self.top_headlines[:15], 1):
                lines.append(f"  {i}. {h}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def _get_close_series(prices_df, ticker: str):
    """Extract Close price series handling both MultiIndex layouts from yfinance."""
    try:
        cols = prices_df.columns

        if getattr(cols, "nlevels", 1) == 2:
            lv0 = set(cols.get_level_values(0))
            lv1 = set(cols.get_level_values(1))

            # group_by="ticker" layout: (ticker, field)
            if ticker in lv0 and "Close" in lv1:
                return prices_df[(ticker, "Close")]

            # Default layout: (field, ticker)
            if "Close" in lv0 and ticker in lv1:
                return prices_df[("Close", ticker)]

            return None

        # Single ticker DataFrame
        if "Close" in prices_df.columns:
            return prices_df["Close"]

    except Exception as e:
        logger.warning(f"Failed to extract Close series for {ticker}: {e}")

    return None


def _compute_returns(prices_df, ticker: str) -> TickerReturn:
    """Compute 1D, 5D, 1M returns from a price DataFrame."""
    tr = TickerReturn(ticker=ticker)
    try:
        col = _get_close_series(prices_df, ticker)
        if col is None:
            return tr

        col = col.dropna()
        if len(col) < 2:
            return tr

        latest = float(col.iloc[-1])
        tr.price = latest

        if len(col) >= 2:
            tr.return_1d = ((latest / float(col.iloc[-2])) - 1) * 100

        if len(col) >= 6:
            tr.return_5d = ((latest / float(col.iloc[-6])) - 1) * 100
        elif len(col) >= 2:
            tr.return_5d = ((latest / float(col.iloc[0])) - 1) * 100

        if len(col) >= 22:
            tr.return_1m = ((latest / float(col.iloc[-22])) - 1) * 100
        elif len(col) >= 2:
            tr.return_1m = ((latest / float(col.iloc[0])) - 1) * 100

    except Exception as e:
        logger.warning(f"Could not compute returns for {ticker}: {e}")

    return tr


def _fetch_etf_data() -> dict:
    """Batch download ETF price data via yfinance."""
    tickers_str = " ".join(ALL_REGIME_TICKERS)
    logger.info(f"  📊 Downloading ETF data for {len(ALL_REGIME_TICKERS)} tickers...")
    try:
        df = yf.download(
            tickers_str,
            period="1mo",
            progress=False,
            group_by="ticker",
            threads=Config.YFINANCE_THREADS,
        )
        return {"df": df, "error": None}
    except Exception as e:
        logger.error(f"yfinance batch download failed: {e}")
        return {"df": None, "error": str(e)}
    finally:
        try:
            from .collector import YFinanceCollector
            YFinanceCollector.cleanup_caches()
        except Exception:
            pass


def _fetch_news_headlines() -> List[str]:
    """Fetch market news headlines from Finnhub."""
    headlines = []
    try:
        url = f"{Config.FINNHUB_BASE_URL}/news"
        params = {"category": "general", "token": Config.FINNHUB_API_KEY}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        news = resp.json()
        if isinstance(news, list):
            for item in news[:20]:
                headline = item.get("headline", "")
                if headline:
                    headlines.append(headline)
    except Exception as e:
        logger.warning(f"Failed to fetch Finnhub headlines: {e}")

    # Supplement with configured current-web search if needed.
    if len(headlines) < 10:
        try:
            for item in search_web("stock market today major financial market news", count=10, freshness="day"):
                title = item.get("title", "")
                if title and title not in headlines:
                    headlines.append(title)
        except Exception as e:
            logger.warning(f"Search news supplement failed: {e}")

    return headlines


def _extract_fred_value(macro_data: dict, key: str) -> Optional[float]:
    """Safely extract a float from FRED macro data."""
    try:
        data = macro_data.get(key, {})
        if isinstance(data, dict):
            obs = data.get("observations", [])
            if obs and isinstance(obs, list):
                # Use latest valid observation (reverse scan)
                for item in reversed(obs):
                    val = item.get("value", "")
                    if val and val != ".":
                        return float(val)
        elif isinstance(data, (int, float)):
            return float(data)
    except (ValueError, TypeError, IndexError):
        pass
    return None


def compute_regime_indicators(
    macro_data: Optional[dict] = None,
    crisis_data: Optional[dict] = None,
) -> RegimeIndicators:
    """Compute all derived market features for regime classification.

    Args:
        macro_data: Optional pre-collected macro data from DataCollector.collect_macro()
        crisis_data: Optional crisis signal dict from DataCollector.collect_crisis_signals()
                     Used to populate credit spread, DXY, drawdown, and other stress fields.

    Returns:
        RegimeIndicators with all available fields populated.
    """
    logger.info("📊 Computing regime indicators...")
    indicators = RegimeIndicators()
    indicators.as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # -- ETF Data --
    etf_result = _fetch_etf_data()
    df = etf_result["df"]

    if df is not None and not df.empty:
        for ticker in BROAD_MARKET_ETFS:
            indicators.broad_market[ticker] = _compute_returns(df, ticker)

        for ticker in SAFE_HAVEN_ETFS:
            indicators.safe_havens[ticker] = _compute_returns(df, ticker)

        indicators.oil = _compute_returns(df, "USO")

        vix_tr = _compute_returns(df, "^VIX")
        if vix_tr.price is not None:
            indicators.vix_level = vix_tr.price
            indicators.vix_1d_change = vix_tr.return_1d

        spy_1d = indicators.broad_market.get("SPY", TickerReturn("SPY")).return_1d or 0
        for name, ticker in SECTOR_ETFS.items():
            tr = _compute_returns(df, ticker)
            if tr.return_1d is not None:
                indicators.sector_returns[name] = tr.return_1d
                indicators.sector_relative[name] = tr.return_1d - spy_1d

        for name, ticker in THEMATIC_ETFS.items():
            indicators.thematic_returns[name] = _compute_returns(df, ticker)
    else:
        logger.warning("No ETF data available — regime indicators will be incomplete")

    # -- FRED Macro Data --
    if macro_data:
        indicators.fed_funds_rate = _extract_fred_value(macro_data, "fed_funds_rate")
        indicators.treasury_10y = _extract_fred_value(macro_data, "treasury_10y")
        indicators.treasury_2y = _extract_fred_value(macro_data, "treasury_2y")

        if indicators.treasury_10y is not None and indicators.treasury_2y is not None:
            indicators.yield_curve_spread = indicators.treasury_10y - indicators.treasury_2y

        indicators.cpi_latest = _extract_fred_value(macro_data, "cpi")
        indicators.unemployment = _extract_fred_value(macro_data, "unemployment")

        fg = macro_data.get("fear_greed") or {}
        if isinstance(fg, dict):
            indicators.fear_greed_value = fg.get("value")
            indicators.fear_greed_label = fg.get("label", "")

    # -- Crisis Data (credit spreads, DXY, drawdown, jobless claims, oil) --
    if crisis_data:
        # HY/IG spreads — FRED returns percentage points (e.g., 3.0 = 300bps)
        hy = crisis_data.get("hy_oas")
        if hy is not None:
            indicators.hy_credit_spread = float(hy)

        ig = crisis_data.get("ig_oas")
        if ig is not None:
            indicators.ig_credit_spread = float(ig)

        dxy = crisis_data.get("dxy")
        if dxy is not None:
            indicators.dxy = float(dxy)

        drawdown = crisis_data.get("spy_drawdown")
        if drawdown is not None:
            indicators.spy_drawdown_from_52w_high = float(drawdown)

        jobless = crisis_data.get("initial_jobless_claims")
        if jobless is not None:
            indicators.initial_jobless_claims = float(jobless)

        oil = crisis_data.get("oil_price")
        if oil is not None:
            indicators.oil_price_wti = float(oil)

    # -- Event Risk State (from EconomicCalendar) --
    try:
        from .economic_calendar import EconomicCalendar, compute_event_risk_state
        eco_cal = EconomicCalendar()
        eco_events = eco_cal.fetch(days_ahead=7, days_back=1)
        event_risk = compute_event_risk_state(eco_events)
        indicators.event_risk_state = event_risk.state
        indicators.event_risk_next_major = event_risk.next_major_event
    except Exception as e:
        logger.warning(f"[regime_indicators] EconomicCalendar failed (non-fatal): {e}")

    # -- News Headlines --
    indicators.top_headlines = _fetch_news_headlines()

    logger.info(f"  ✅ Regime indicators computed: {len(indicators.sector_returns)} sectors, "
                f"{len(indicators.top_headlines)} headlines")
    return indicators
