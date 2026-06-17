"""Analyst Signals — short interest, recommendation trends, and estimates.

Provides enrichment data for the promotion funnel and the full council packet.
Uses Finnhub for recommendations and FMP Premium for short interest and estimates.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf

from .collector import FinnhubCollector, FMPCollector, _safe_get, _limiters
from .config import Config

logger = logging.getLogger(__name__)

UTC = timezone.utc


def get_short_interest(ticker: str, *, timeout: int | None = None, retries: int | None = None) -> dict:
    """Fetch short interest data from FMP.

    Returns:
        dict with keys:
          - short_interest_pct: Short float as % of float shares (e.g. 5.2 = 5.2%)
          - days_to_cover: Short interest / avg daily volume
          - squeeze_risk_flag: True if short_interest_pct > 20% AND days_to_cover > 5
          - source: "fmp" or "unavailable"
    """
    result = {
        "short_interest_pct": None,
        "days_to_cover": None,
        "squeeze_risk_flag": False,
        "source": "unavailable",
        "ticker": ticker,
    }

    try:
        data = None
        if Config.FMP_SHORT_INTEREST_ENDPOINT:
            url = f"{Config.FMP_BASE_URL}/{Config.FMP_SHORT_INTEREST_ENDPOINT.strip('/')}"
            params = {"apikey": Config.FMP_API_KEY, "symbol": ticker}
            data = _safe_get(url, params, "fmp", timeout=timeout or 15, retries=retries)

        item = None
        if isinstance(data, list) and data:
            item = data[0]
        elif isinstance(data, dict) and data:
            item = data

        if item:
            short_pct = item.get("shortPercentOfFloat") or item.get("shortPercent")
            days_cover = item.get("shortRatio") or item.get("daysTocover") or item.get("daysToCover")

            if short_pct is not None:
                result["short_interest_pct"] = round(float(short_pct) * 100, 2) if float(short_pct) < 1 else round(float(short_pct), 2)
            if days_cover is not None:
                result["days_to_cover"] = round(float(days_cover), 2)

            if (
                result["short_interest_pct"] is not None
                and result["days_to_cover"] is not None
                and result["short_interest_pct"] > 20
                and result["days_to_cover"] > 5
            ):
                result["squeeze_risk_flag"] = True

            result["source"] = "fmp"
        else:
            if timeout is not None:
                logger.warning(
                    "[analyst_signals] Short interest unavailable for %s in scan-safe mode; "
                    "skipping unbounded yfinance info fallback",
                    ticker,
                )
                return result
            info = yf.Ticker(ticker).info or {}
            short_pct = (
                info.get("shortPercentOfFloat")
                or info.get("sharesPercentSharesOut")
            )
            days_cover = info.get("shortRatio")
            if short_pct is not None:
                result["short_interest_pct"] = (
                    round(float(short_pct) * 100, 2)
                    if float(short_pct) < 1
                    else round(float(short_pct), 2)
                )
            if days_cover is not None:
                result["days_to_cover"] = round(float(days_cover), 2)
            if (
                result["short_interest_pct"] is not None
                and result["days_to_cover"] is not None
                and result["short_interest_pct"] > 20
                and result["days_to_cover"] > 5
            ):
                result["squeeze_risk_flag"] = True
            if result["short_interest_pct"] is not None or result["days_to_cover"] is not None:
                result["source"] = "yfinance"

    except Exception as e:
        logger.warning(f"[analyst_signals] Short interest fetch failed for {ticker}: {e}")

    return result


def get_recommendation_trends(ticker: str) -> dict:
    """Fetch analyst recommendation trends from Finnhub.

    Analyzes the most recent 2 months of recommendation changes to compute
    net upgrade/downgrade momentum.

    Returns:
        dict with keys:
          - net_upgrades_30d: (upgrades - downgrades) over last 30 days
          - net_downgrades_30d: abs(downgrades - upgrades) if negative momentum
          - recommendation_mix: dict with buy/hold/sell counts (latest period)
          - consensus: "strong_buy" | "buy" | "hold" | "sell" | "strong_sell" | "unknown"
          - source: "finnhub" or "unavailable"
    """
    result = {
        "net_upgrades_30d": None,
        "net_downgrades_30d": None,
        "recommendation_mix": {},
        "consensus": "unknown",
        "source": "unavailable",
        "ticker": ticker,
    }

    try:
        finnhub = FinnhubCollector()
        recs = finnhub.analyst_recommendations(ticker)

        if not recs or not isinstance(recs, list):
            return result

        # Sort by period descending so recs[0] is always the most recent month
        recs = sorted(recs, key=lambda r: r.get("period", ""), reverse=True)

        # Most recent period
        latest = recs[0] if recs else {}
        result["recommendation_mix"] = {
            "strong_buy": latest.get("strongBuy", 0),
            "buy": latest.get("buy", 0),
            "hold": latest.get("hold", 0),
            "sell": latest.get("sell", 0),
            "strong_sell": latest.get("strongSell", 0),
        }

        # Consensus from latest period
        mix = result["recommendation_mix"]
        total = sum(mix.values())
        if total > 0:
            buy_pct = (mix.get("strong_buy", 0) + mix.get("buy", 0)) / total
            sell_pct = (mix.get("sell", 0) + mix.get("strong_sell", 0)) / total
            if buy_pct >= 0.60:
                result["consensus"] = "buy" if buy_pct < 0.75 else "strong_buy"
            elif sell_pct >= 0.40:
                result["consensus"] = "sell" if sell_pct < 0.60 else "strong_sell"
            else:
                result["consensus"] = "hold"

        # Net sentiment: compare latest period to previous period
        if len(recs) >= 2:
            prev = recs[1]
            latest_positive = latest.get("strongBuy", 0) + latest.get("buy", 0)
            prev_positive = prev.get("strongBuy", 0) + prev.get("buy", 0)
            latest_negative = latest.get("sell", 0) + latest.get("strongSell", 0)
            prev_negative = prev.get("sell", 0) + prev.get("strongSell", 0)

            net_sentiment_change = (latest_positive - prev_positive) - (latest_negative - prev_negative)
            result["net_upgrades_30d"] = max(net_sentiment_change, 0)
            result["net_downgrades_30d"] = max(-net_sentiment_change, 0)

        result["source"] = "finnhub"

    except Exception as e:
        logger.warning(f"[analyst_signals] Recommendation trends failed for {ticker}: {e}")

    return result


def get_analyst_estimates(ticker: str, *, timeout: int | None = None, retries: int | None = None) -> dict:
    """Fetch forward analyst estimates from FMP.

    Returns:
        dict with keys:
          - next_q_eps_estimate: Next quarter EPS estimate
          - next_q_revenue_estimate: Next quarter revenue estimate in USD
          - fy1_revenue_estimate: FY+1 revenue estimate in USD
          - quarterly_estimates: compact latest quarterly estimates when enabled/available
          - annual_estimates: compact latest annual estimates
          - price_target_high: Analyst PT high
          - price_target_low: Analyst PT low
          - price_target_consensus: Analyst PT consensus
          - source: "fmp" or "unavailable"
    """
    result = {
        "next_q_eps_estimate": None,
        "next_q_revenue_estimate": None,
        "fy1_revenue_estimate": None,
        "quarterly_estimates": [],
        "annual_estimates": [],
        "price_target_high": None,
        "price_target_low": None,
        "price_target_consensus": None,
        "source": "unavailable",
        "ticker": ticker,
    }

    try:
        fmp = FMPCollector()

        # Price target consensus
        pt = fmp.price_target_consensus(ticker, timeout=timeout, retries=retries)
        if pt and isinstance(pt, dict):
            result["price_target_high"] = pt.get("targetHigh")
            result["price_target_low"] = pt.get("targetLow")
            result["price_target_consensus"] = pt.get("targetConsensus") or pt.get("targetMedian")
            result["source"] = "fmp"

        # Analyst estimates. The live FMP key supports the stable
        # analyst-estimates endpoint for annual estimates. Quarterly estimates
        # are opt-in because the current plan/API returned access errors.
        url_estimates = f"{Config.FMP_BASE_URL}/analyst-estimates"
        url_earnings = f"{Config.FMP_BASE_URL}/earnings"

        # Next quarter EPS from earnings calendar (has epsEstimated for future dates)
        earnings_data = _safe_get(
            url_earnings,
            {"apikey": Config.FMP_API_KEY, "symbol": ticker},
            "fmp",
            timeout=timeout or 15,
            retries=retries,
        )
        if isinstance(earnings_data, list) and earnings_data:
            # Find the next future earnings entry (epsActual is None)
            for e in earnings_data:
                if isinstance(e, dict) and e.get("epsActual") is None and e.get("epsEstimated") is not None:
                    result["next_q_eps_estimate"] = e.get("epsEstimated")
                    result["source"] = "fmp"
                    break

        # Quarterly estimates (opt-in only; disabled for the current live key)
        if Config.FMP_ENABLE_QUARTERLY_ESTIMATES:
            quarterly = _safe_get(
                url_estimates,
                {"apikey": Config.FMP_API_KEY, "symbol": ticker, "period": "quarter", "limit": "4"},
                "fmp",
                timeout=timeout or 15,
                retries=retries,
            )
            if isinstance(quarterly, list) and quarterly:
                quarterly_sorted = sorted(quarterly, key=lambda x: x.get("date", ""), reverse=True)
                compact_q = []
                for row in quarterly_sorted[:4]:
                    if not isinstance(row, dict):
                        continue
                    compact_q.append({
                        "date": row.get("date"),
                        "estimated_eps_avg": (
                            row.get("estimatedEpsAvg")
                            or row.get("estimatedEPSAvg")
                            or row.get("epsAvg")
                        ),
                        "estimated_revenue_avg": (
                            row.get("estimatedRevenueAvg")
                            or row.get("revenueAvg")
                        ),
                        "estimated_ebitda_avg": (
                            row.get("estimatedEbitdaAvg")
                            or row.get("ebitdaAvg")
                        ),
                    })
                result["quarterly_estimates"] = compact_q
                if compact_q:
                    first_q = compact_q[0]
                    result["next_q_eps_estimate"] = result["next_q_eps_estimate"] or first_q.get("estimated_eps_avg")
                    result["next_q_revenue_estimate"] = first_q.get("estimated_revenue_avg")
                    result["source"] = "fmp"

        # Annual revenue estimates
        annual = _safe_get(
            url_estimates,
            {"apikey": Config.FMP_API_KEY, "symbol": ticker, "period": "annual", "limit": "3"},
            "fmp",
            timeout=timeout or 15,
            retries=retries,
        )
        if isinstance(annual, list) and annual:
            annual_sorted = sorted(annual, key=lambda x: x.get("date", ""), reverse=True)
            compact_a = []
            for row in annual_sorted[:3]:
                if not isinstance(row, dict):
                    continue
                compact_a.append({
                    "date": row.get("date"),
                    "estimated_eps_avg": (
                        row.get("estimatedEpsAvg")
                        or row.get("estimatedEPSAvg")
                        or row.get("epsAvg")
                    ),
                    "estimated_revenue_avg": (
                        row.get("estimatedRevenueAvg")
                        or row.get("revenueAvg")
                    ),
                })
            result["annual_estimates"] = compact_a
            latest_a = annual_sorted[0]
            result["fy1_revenue_estimate"] = (
                latest_a.get("estimatedRevenueAvg")
                or latest_a.get("estimatedRevenueLow")
                or latest_a.get("revenueAvg")
            )
            result["source"] = "fmp"

    except Exception as e:
        logger.warning(f"[analyst_signals] Estimates fetch failed for {ticker}: {e}")

    return result
