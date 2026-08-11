"""Point-in-time alpha features evaluated without changing live decisions."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(str(value).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return default


def _day(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _first(value: Any) -> dict[str, Any]:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return value if isinstance(value, dict) else {}


def build_insider_shadow_signal(
    payload: Any,
    *,
    as_of: date | datetime | None = None,
) -> dict[str, Any]:
    """Build a no-lookahead, open-market-purchase insider signal.

    Only transactions disclosed by ``as_of`` are eligible. A cluster means
    at least three distinct named buyers whose transaction dates fit inside a
    14-day window. The result is research evidence only and cannot change a
    live score or order.
    """
    reference = _day(as_of) or datetime.now(timezone.utc).date()
    envelope = payload if isinstance(payload, dict) else {}
    raw_rows = envelope.get("data") if isinstance(envelope.get("data"), list) else []
    seen: set[tuple[Any, ...]] = set()
    purchases: list[dict[str, Any]] = []
    excluded = {
        "future_or_missing_filing": 0,
        "outside_30_day_window": 0,
        "not_open_market_purchase": 0,
        "missing_named_buyer": 0,
        "duplicate": 0,
    }

    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        filing_day = _day(raw.get("filingDate"))
        transaction_day = _day(raw.get("transactionDate"))
        if filing_day is None or filing_day > reference:
            excluded["future_or_missing_filing"] += 1
            continue
        if (reference - filing_day).days > 30:
            excluded["outside_30_day_window"] += 1
            continue
        change = _num(raw.get("change"), 0.0) or 0.0
        if str(raw.get("transactionCode") or "").upper() != "P" or change <= 0:
            excluded["not_open_market_purchase"] += 1
            continue
        name = str(raw.get("name") or raw.get("insiderName") or "").strip()
        if not name:
            excluded["missing_named_buyer"] += 1
            continue
        if transaction_day is None or transaction_day > reference:
            transaction_day = filing_day
        price = _num(raw.get("transactionPrice") or raw.get("price"), None)
        dedupe_key = (
            name.casefold(),
            filing_day.isoformat(),
            transaction_day.isoformat(),
            round(change, 6),
            round(price or 0.0, 6),
        )
        if dedupe_key in seen:
            excluded["duplicate"] += 1
            continue
        seen.add(dedupe_key)
        purchases.append(
            {
                "name": name,
                "filing_date": filing_day.isoformat(),
                "transaction_date": transaction_day.isoformat(),
                "shares": change,
                "price": price,
                "value": change * price if price is not None and price > 0 else None,
            }
        )

    purchases.sort(key=lambda row: row["transaction_date"])
    best_window: list[dict[str, Any]] = []
    for left, start in enumerate(purchases):
        start_day = _day(start["transaction_date"])
        if start_day is None:
            continue
        window = [
            row
            for row in purchases[left:]
            if (_day(row["transaction_date"]) - start_day).days <= 14  # type: ignore[operator]
        ]
        unique = {str(row["name"]).casefold() for row in window}
        current_unique = {str(row["name"]).casefold() for row in best_window}
        if len(unique) > len(current_unique):
            best_window = window

    unique_buyers = sorted({str(row["name"]) for row in best_window}, key=str.casefold)
    disclosed_value = sum(float(row["value"]) for row in purchases if row.get("value") is not None)
    return {
        "available": bool(raw_rows),
        "source": "finnhub_insider_transactions",
        "as_of": reference.isoformat(),
        "method": "open_market_code_P_distinct_buyers_within_14_days",
        "cluster_confirmed": len(unique_buyers) >= 3,
        "unique_buyers_14d": len(unique_buyers),
        "buyer_names_14d": unique_buyers,
        "qualifying_purchase_count_30d": len(purchases),
        "disclosed_purchase_value_30d": round(disclosed_value, 2),
        "role_data_available": any(bool(row.get("role")) for row in raw_rows if isinstance(row, dict)),
        "excluded_counts": excluded,
        "live_score_impact": 0.0,
    }


def build_explorer_shadow_signal(stock_data: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a quality-gated $500M-$2B explorer candidate in shadow mode."""
    quote = stock_data.get("quote") if isinstance(stock_data.get("quote"), dict) else {}
    profile = stock_data.get("profile") if isinstance(stock_data.get("profile"), dict) else {}
    funnel = stock_data.get("funnel_candidate") if isinstance(stock_data.get("funnel_candidate"), dict) else {}
    cash_flow = _first(stock_data.get("cash_flow") or stock_data.get("cash_flow_statement"))
    income = _first(stock_data.get("income_statement"))
    market_cap = _num(
        quote.get("marketCap") or profile.get("mktCap") or profile.get("marketCap") or funnel.get("market_cap"),
        None,
    )
    price = _num(quote.get("price") or funnel.get("price"), None)
    average_volume = _num(
        quote.get("avgVolume") or quote.get("averageVolume") or funnel.get("avg_volume"),
        None,
    )
    dollar_volume = price * average_volume if price and average_volume else None
    free_cash_flow = _num(cash_flow.get("freeCashFlow"), None)
    operating_income = _num(income.get("operatingIncome"), None)
    net_income = _num(income.get("netIncome"), None)
    ipo_day = _day(profile.get("ipoDate"))
    listing_age_days = (datetime.now(timezone.utc).date() - ipo_day).days if ipo_day else None
    lottery_excluded = bool(funnel.get("lottery_excluded") or funnel.get("lottery_flag"))
    checks = {
        "market_cap_500m_to_2b": market_cap is not None and 500_000_000 <= market_cap <= 2_000_000_000,
        "price_at_least_5": price is not None and price >= 5.0,
        "average_dollar_volume_at_least_10m": dollar_volume is not None and dollar_volume >= 10_000_000,
        "positive_free_cash_flow": free_cash_flow is not None and free_cash_flow > 0,
        "positive_operating_or_net_income": bool(
            (operating_income is not None and operating_income > 0)
            or (net_income is not None and net_income > 0)
        ),
        "listed_at_least_two_years": listing_age_days is not None and listing_age_days >= 730,
        "not_lottery_excluded": not lottery_excluded,
        "not_etf_or_fund": not bool(profile.get("isEtf") or profile.get("isFund")),
    }
    missing = [
        name
        for name, value in {
            "market_cap": market_cap,
            "price": price,
            "average_volume": average_volume,
            "free_cash_flow": free_cash_flow,
            "profitability": operating_income if operating_income is not None else net_income,
            "ipo_date": ipo_day,
        }.items()
        if value is None
    ]
    return {
        "qualified": all(checks.values()),
        "checks": checks,
        "missing_data": missing,
        "market_cap": market_cap,
        "price": price,
        "average_dollar_volume": dollar_volume,
        "free_cash_flow": free_cash_flow,
        "listing_age_days": listing_age_days,
        "live_score_impact": 0.0,
    }


def build_alpha_shadow_signals(stock_data: dict[str, Any]) -> dict[str, Any]:
    """Build all proposed alpha signals without exposing them to live scoring."""
    funnel = stock_data.get("funnel_candidate") if isinstance(stock_data.get("funnel_candidate"), dict) else {}
    industry = funnel.get("industry_momentum_shadow")
    episodic = funnel.get("episodic_pivot_shadow")
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "shadow_only",
        "live_score_impact": 0.0,
        "insider_cluster": build_insider_shadow_signal(stock_data.get("insider_finnhub")),
        "industry_momentum": industry if isinstance(industry, dict) else {"available": False, "live_score_impact": 0.0},
        "episodic_pivot": episodic if isinstance(episodic, dict) else {"available": False, "live_score_impact": 0.0},
        "quality_small_cap_explorer": build_explorer_shadow_signal(stock_data),
    }
