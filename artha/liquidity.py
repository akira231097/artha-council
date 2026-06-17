"""Liquidity hard gate and scoring for stock candidates.

Ensures candidates meet minimum tradability thresholds before
being passed to the council for analysis.
"""
import logging
from typing import Any, Optional

from .config import Config

logger = logging.getLogger(__name__)


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", ""))
    except Exception:
        return None


def _nested(data: dict, *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def resolve_average_volume(ticker_data: dict) -> dict[str, Any]:
    """Resolve true average volume separately from current intraday volume.

    The council must not fail a stock at market open just because the current
    day's volume is still low. This helper prioritizes explicit average-volume
    fields and uses current volume only as a non-average fallback.
    """
    average_candidates = (
        ("avgVolume", ticker_data.get("avgVolume")),
        ("averageVolume", ticker_data.get("averageVolume")),
        ("avg_volume", ticker_data.get("avg_volume")),
        ("quote.avgVolume", _nested(ticker_data, "quote", "avgVolume")),
        ("quote.averageVolume", _nested(ticker_data, "quote", "averageVolume")),
        ("yf_quote.averageVolume", _nested(ticker_data, "yf_quote", "averageVolume")),
        ("yf_quote.avg_volume", _nested(ticker_data, "yf_quote", "avg_volume")),
    )
    for source, value in average_candidates:
        volume = _num(value)
        if volume and volume > 0:
            return {"volume": volume, "source": source, "is_average": True}

    history = ticker_data.get("price_history")
    if isinstance(history, list):
        volumes = []
        for row in history[-30:]:
            if isinstance(row, dict):
                volume = _num(row.get("volume") or row.get("Volume") or row.get("v"))
                if volume and volume > 0:
                    volumes.append(volume)
        if len(volumes) >= 10:
            return {
                "volume": sum(volumes[-20:]) / min(len(volumes), 20),
                "source": "price_history_20d",
                "is_average": True,
            }

    current_candidates = (
        ("volume", ticker_data.get("volume")),
        ("quote.volume", _nested(ticker_data, "quote", "volume")),
        ("yf_quote.volume", _nested(ticker_data, "yf_quote", "volume")),
        ("yf_quote.regularMarketVolume", _nested(ticker_data, "yf_quote", "regularMarketVolume")),
    )
    for source, value in current_candidates:
        volume = _num(value)
        if volume and volume > 0:
            return {"volume": volume, "source": source, "is_average": False}
    return {"volume": None, "source": "missing", "is_average": False}


def passes_liquidity_gate(ticker_data: dict) -> bool:
    """Hard gate: returns True only if all minimum liquidity thresholds are met.

    Thresholds (configurable via Config):
      - Market cap >= Config.LIQUIDITY_MIN_MARKET_CAP (default $1B)
      - ADV (avg_volume * price) >= Config.LIQUIDITY_MIN_ADV (default $10M)
      - Price >= Config.LIQUIDITY_MIN_PRICE (default $5.00)

    Args:
        ticker_data: dict with keys: market_cap, avg_volume, price
                     (as returned by FMP quote or yfinance quote)

    Returns:
        True if all thresholds pass, False otherwise.
    """
    min_market_cap = getattr(Config, "LIQUIDITY_MIN_MARKET_CAP", 1_000_000_000)
    min_adv = getattr(Config, "LIQUIDITY_MIN_ADV", 10_000_000)
    min_price = getattr(Config, "LIQUIDITY_MIN_PRICE", 5.0)

    # Extract values — handle both FMP and yfinance field names
    market_cap = (
        ticker_data.get("marketCap")
        or ticker_data.get("market_cap")
        or 0
    )
    volume_info = resolve_average_volume(ticker_data)
    avg_volume = volume_info.get("volume") or 0
    price = (
        ticker_data.get("price")
        or ticker_data.get("currentPrice")
        or ticker_data.get("regularMarketPrice")
        or 0
    )

    try:
        market_cap = float(market_cap or 0)
        price = float(price or 0)
    except (TypeError, ValueError):
        logger.warning("[liquidity] Could not parse numeric fields from ticker_data")
        return False

    adv = avg_volume * price

    if market_cap < min_market_cap:
        logger.debug(f"[liquidity] FAIL market_cap={market_cap:,.0f} < {min_market_cap:,.0f}")
        return False
    if volume_info.get("is_average") and adv < min_adv:
        logger.debug(f"[liquidity] FAIL ADV={adv:,.0f} < {min_adv:,.0f}")
        return False
    if not volume_info.get("is_average") and adv < min_adv:
        logger.debug(
            "[liquidity] Missing true average volume; not hard-failing on current-volume ADV=%s source=%s",
            f"{adv:,.0f}",
            volume_info.get("source"),
        )
    if price < min_price:
        logger.debug(f"[liquidity] FAIL price={price:.2f} < {min_price:.2f}")
        return False

    return True


def compute_liquidity_score(ticker_data: dict) -> float:
    """Compute a 0–100 liquidity score.

    Components:
      - Market cap score (0–40): log-scaled from $1B to $1T
      - ADV score (0–40): log-scaled from $10M to $1B
      - Price score (0–20): log-scaled from $5 to $1000

    Returns:
        float in [0, 100], where 100 = most liquid
    """
    import math

    try:
        market_cap = float(
            ticker_data.get("marketCap")
            or ticker_data.get("market_cap")
            or 0
        )
    except (TypeError, ValueError):
        return 0.0

    try:
        avg_volume = float(resolve_average_volume(ticker_data).get("volume") or 0)
    except (TypeError, ValueError):
        return 0.0

    try:
        price = float(
            ticker_data.get("price")
            or ticker_data.get("currentPrice")
            or ticker_data.get("regularMarketPrice")
            or 0
        )
    except (TypeError, ValueError):
        return 0.0

    adv = avg_volume * price

    def _log_score(value: float, lo: float, hi: float, weight: float) -> float:
        """Score value on log scale between lo and hi, scaled by weight."""
        if value <= 0:
            return 0.0
        if value >= hi:
            return weight
        if value <= lo:
            return 0.0
        log_val = math.log10(value)
        log_lo = math.log10(lo)
        log_hi = math.log10(hi)
        frac = (log_val - log_lo) / (log_hi - log_lo)
        return round(min(weight, frac * weight), 2)

    cap_score = _log_score(market_cap, 1e9, 1e12, 40)   # $1B → $1T = 0–40
    adv_score = _log_score(adv, 1e7, 1e9, 40)            # $10M → $1B = 0–40
    price_score = _log_score(price, 5, 1000, 20)          # $5 → $1000 = 0–20

    total = cap_score + adv_score + price_score
    return round(min(100.0, total), 1)
