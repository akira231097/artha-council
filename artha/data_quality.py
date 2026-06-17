"""Data Quality Report — validate and score the completeness of collected stock data.

Provides structured quality assessment for the council and ingestion pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

UTC = timezone.utc


@dataclass
class DataQualityReport:
    """Structured data quality assessment for a stock's collected data."""

    ticker: str = ""
    completeness_score: float = 0.0          # 0-100, higher = more complete
    staleness_warnings: list[str] = field(default_factory=list)   # Fields with stale dates
    source_conflicts: list[str] = field(default_factory=list)      # FMP vs yfinance conflicts
    anomaly_flags: list[str] = field(default_factory=list)         # Suspicious values
    missing_fields: list[str] = field(default_factory=list)        # Required fields missing
    enrichment_missing_fields: list[str] = field(default_factory=list)  # Context fields missing
    sources_used: list[str] = field(default_factory=list)          # Which APIs had data
    context_coverage_score: float = 0.0     # 0-100, non-blocking enrichment coverage
    passed_hard_checks: bool = True          # False = do not pass to council
    hard_check_failures: list[str] = field(default_factory=list)  # Reasons for hard failures
    as_of: str = ""


# Fields checked for completeness (weight = how important each is)
_REQUIRED_FIELDS = {
    "quote": 15,
    "profile": 10,
    "income_statement": 15,
    "balance_sheet": 10,
    "cash_flow": 10,
    "ratios_ttm": 10,
    "key_metrics_ttm": 10,
    "price_history": 10,
    "technicals": 5,
    "news": 5,
}


_ENRICHMENT_FIELDS = {
    "price_target_consensus": 8,
    "analyst_recs": 8,
    "earnings_surprises": 8,
    "earnings_context": 10,
    "insider_finnhub": 8,
    "recommendation_trends": 10,
    "analyst_estimates": 12,
    "short_interest": 8,
    "sec": 16,
    "benzinga_news": 8,
    "finnhub_news": 6,
    "finnhub_sentiment": 6,
}


def validate_stock_data(data: dict) -> DataQualityReport:
    """Validate collected stock data and produce a quality report.

    Runs both hard ingestion checks (disqualifying) and soft advisory checks.

    Hard checks (passed_hard_checks=False if any fail):
      - price > 0
      - volume >= 0
      - market_cap >= 0
      - price_history has >= 20 data points

    Soft checks (warnings added to report, do not disqualify):
      - Gross margin in [-50%, 100%] range
      - Report dates valid ISO strings
      - FMP price vs yfinance/Massive price within 5%

    Args:
        data: dict as returned by DataCollector.collect_stock()

    Returns:
        DataQualityReport with all fields populated.
    """
    ticker = data.get("ticker", "unknown")
    report = DataQualityReport(
        ticker=ticker,
        as_of=datetime.now(UTC).isoformat(),
    )

    # --- Missing Field Check ---
    total_weight = sum(_REQUIRED_FIELDS.values())
    earned_weight = 0

    for field_name, weight in _REQUIRED_FIELDS.items():
        value = data.get(field_name)
        if value is None:
            report.missing_fields.append(field_name)
        elif isinstance(value, (list, dict)) and len(value) == 0:
            report.missing_fields.append(field_name)
        else:
            earned_weight += weight

    # News: FMP, Benzinga, or Finnhub ticker news can satisfy the core news lane.
    if "news" in report.missing_fields:
        if data.get("benzinga_news") or data.get("finnhub_news"):
            report.missing_fields.remove("news")
            earned_weight += _REQUIRED_FIELDS["news"]

    report.completeness_score = round(earned_weight / total_weight * 100, 1)

    # --- Non-blocking context coverage check ---
    context_total = sum(_ENRICHMENT_FIELDS.values())
    context_earned = 0
    for field_name, weight in _ENRICHMENT_FIELDS.items():
        value = data.get(field_name)
        source_unavailable = (
            isinstance(value, dict)
            and str(value.get("source", "")).lower() == "unavailable"
        )
        status_unavailable = (
            isinstance(value, dict)
            and str(value.get("status", "")).lower() == "unavailable"
        )
        if value is None or (isinstance(value, (list, dict)) and len(value) == 0) or source_unavailable or status_unavailable:
            report.enrichment_missing_fields.append(field_name)
        else:
            context_earned += weight

    report.context_coverage_score = round(context_earned / context_total * 100, 1)

    # --- Sources Used ---
    sources = []
    if data.get("quote") or data.get("profile"):
        sources.append("fmp")
    if data.get("yf_quote") or data.get("price_history"):
        sources.append("yfinance")
    if data.get("massive_quote") or data.get("price_history_source") == "massive":
        sources.append("massive")
    if data.get("finnhub_sentiment") or data.get("analyst_recs"):
        sources.append("finnhub")
    if data.get("benzinga_news"):
        sources.append("benzinga")
    if data.get("recommendation_trends") or data.get("analyst_estimates") or data.get("short_interest"):
        sources.append("analyst_signals")
    sec_payload = data.get("sec") or {}
    if isinstance(sec_payload, dict) and sec_payload.get("status") in ("ok", "partial"):
        sources.append("sec")
    report.sources_used = sources

    # --- Hard Checks ---
    quote = data.get("quote") or {}
    yf_quote = data.get("yf_quote") or {}
    massive_quote = data.get("massive_quote") or {}

    price = float(quote.get("price", 0) or yf_quote.get("price", 0) or massive_quote.get("price", 0) or 0)
    if price <= 0:
        report.hard_check_failures.append(f"price={price} <= 0")

    volume = float(quote.get("volume", -1) or massive_quote.get("volume", 0) or 0)
    if volume < 0:
        report.hard_check_failures.append(f"volume={volume} < 0")

    market_cap = float(
        quote.get("marketCap", -1)
        or yf_quote.get("market_cap", -1)
        or 0
    )
    if market_cap < 0:
        report.hard_check_failures.append(f"market_cap={market_cap} < 0")

    price_history = data.get("price_history")
    if not isinstance(price_history, list) or len(price_history) < 20:
        actual_len = len(price_history) if isinstance(price_history, list) else 0
        report.hard_check_failures.append(
            f"price_history too short: {actual_len} < 20 bars"
        )

    report.passed_hard_checks = len(report.hard_check_failures) == 0

    # --- Soft Checks: Margin Sanity ---
    income = data.get("income_statement")
    if income and isinstance(income, list) and income:
        latest_income = income[0]
        revenue = float(latest_income.get("revenue", 0) or 0)
        gross_profit = float(latest_income.get("grossProfit", 0) or 0)
        if revenue > 0:
            gross_margin = gross_profit / revenue
            if gross_margin < -0.50 or gross_margin > 1.0:
                report.anomaly_flags.append(
                    f"gross_margin={gross_margin:.1%} outside [-50%, 100%]"
                )

    # --- Soft Checks: Price Cross-Source Conflict ---
    fmp_price = float(quote.get("price", 0) or 0)
    cross_check_prices = {
        "yfinance": float(yf_quote.get("price", 0) or 0),
        "Massive": float(massive_quote.get("price", 0) or 0),
    }
    for source_name, cross_price in cross_check_prices.items():
        if fmp_price <= 0 or cross_price <= 0:
            continue
        diff_pct = abs(fmp_price - cross_price) / fmp_price
        if diff_pct > 0.05:
            report.source_conflicts.append(
                f"price conflict: FMP={fmp_price:.2f} vs {source_name}={cross_price:.2f} "
                f"({diff_pct:.1%} diff)"
            )

    history_checks = data.get("history_provider_checks") or {}
    for conflict in history_checks.get("conflicts") or []:
        report.source_conflicts.append(f"history conflict: {conflict}")

    # --- Soft Checks: Report Date Staleness ---
    if income and isinstance(income, list) and income:
        report_date = income[0].get("date", "")
        if report_date:
            try:
                report_dt = datetime.strptime(report_date, "%Y-%m-%d")
                age_days = (datetime.now() - report_dt).days
                if age_days > 180:
                    report.staleness_warnings.append(
                        f"income_statement date {report_date} is {age_days} days old"
                    )
            except ValueError:
                report.staleness_warnings.append(
                    f"income_statement date '{report_date}' is not valid ISO format"
                )

    sec_payload = data.get("sec") or {}
    if isinstance(sec_payload, dict) and sec_payload.get("status") in ("ok", "partial"):
        stale_days = sec_payload.get("latest_10q_or_10k_staleness_days")
        try:
            stale_days_int = int(stale_days) if stale_days is not None else None
        except (TypeError, ValueError):
            stale_days_int = None
        if stale_days_int is not None and stale_days_int > 160:
            report.staleness_warnings.append(
                f"SEC latest 10-Q/10-K filing is {stale_days_int} days old"
            )
        if not sec_payload.get("financial_facts"):
            report.anomaly_flags.append("SEC companyfacts returned no tracked financial facts")

    return report
