"""Source-of-truth vendor priority policy for all data domains.

Defines which API is primary and fallback for each data domain.
Use resolve_source() to get the best available data.
"""
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Maps each data domain to ordered list of vendors (primary first, then fallbacks)
VENDOR_PRIORITY: dict[str, list[str]] = {
    "equity_fundamentals":        ["fmp", "yfinance"],
    "equity_quotes":              ["fmp", "massive", "yfinance"],
    "equity_history":             ["yfinance", "massive", "fmp"],
    "macro_timeseries":           ["fred"],
    "macro_event_calendar":       ["fmp"],
    "earnings_calendar":          ["fmp", "finnhub"],
    "news_sentiment":             ["finnhub", "fmp"],
    "analyst_recommendations":    ["finnhub", "fmp"],
    "insider_transactions":       ["finnhub"],
    "crypto_prices":              ["coingecko", "fmp"],
    "technicals":                 ["local"],   # computed locally from yfinance data
}


def resolve_source(domain: str, primary_data: Any, fallback_data: Any = None) -> Any:
    """Return best available data: primary if non-empty, else fallback.

    'Non-empty' means: not None, not empty list, not empty dict.

    Args:
        domain: Data domain key (for logging only)
        primary_data: Data from the primary vendor
        fallback_data: Data from the fallback vendor (optional)

    Returns:
        primary_data if available, else fallback_data, else None
    """
    def _is_available(data: Any) -> bool:
        if data is None:
            return False
        if isinstance(data, (list, dict)) and len(data) == 0:
            return False
        return True

    if _is_available(primary_data):
        return primary_data

    if _is_available(fallback_data):
        logger.debug(f"[vendor_priority] {domain}: primary unavailable, using fallback")
        return fallback_data

    logger.debug(f"[vendor_priority] {domain}: all sources unavailable")
    return None


def get_primary_vendor(domain: str) -> Optional[str]:
    """Return the primary vendor name for a given domain."""
    vendors = VENDOR_PRIORITY.get(domain, [])
    return vendors[0] if vendors else None


def get_fallback_vendor(domain: str) -> Optional[str]:
    """Return the first fallback vendor name for a given domain."""
    vendors = VENDOR_PRIORITY.get(domain, [])
    return vendors[1] if len(vendors) > 1 else None
