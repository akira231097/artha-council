"""Trailing stop management for TACTICAL_BUY positions.

Implements ATR-based trailing stop that ratchets up as price moves in our favor
but never moves down. Primarily used for TACTICAL_BUY (swing trades) where
we want to lock in profits as the position moves up.

Algorithm:
  - Compute ATR-14 from price history
  - Trailing stop = max(previous_stop, current_high - ATR_mult × ATR)
  - Floor: max(entry × (1 - min_pct), computed_stop)  ← never tighter than hard stop
  - On 30-min price check: update trailing stop, check for breach
"""
from __future__ import annotations

import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Optional

from .config import Config

logger = logging.getLogger(__name__)


def _to_dec(v: Any) -> Decimal:
    if isinstance(v, Decimal):
        return v
    if isinstance(v, (int, float)):
        return Decimal(str(v))
    if isinstance(v, str):
        try:
            return Decimal(v)
        except Exception:
            return Decimal("0")
    return Decimal("0")


def compute_atr(price_history: list[dict], period: int = 14) -> Optional[float]:
    """Compute ATR-14 from a list of OHLCV candles.

    Each candle dict is expected to have: open, high, low, close (all numeric).
    Returns None if insufficient data.
    """
    if not price_history or len(price_history) < 2:
        return None

    true_ranges = []
    for i in range(1, len(price_history)):
        try:
            c = price_history[i]
            prev = price_history[i - 1]
            high = float(c.get("high") or c.get("h") or 0)
            low = float(c.get("low") or c.get("l") or 0)
            prev_close = float(prev.get("close") or prev.get("c") or prev.get("adjClose") or 0)
            if high <= 0 or low <= 0:
                continue
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)
        except Exception:
            continue

    if len(true_ranges) < period:
        # Too few candles — use simple average of available
        if not true_ranges:
            return None
        return sum(true_ranges) / len(true_ranges)

    # Wilder's smoothing ATR
    atr = sum(true_ranges[:period]) / period
    for tr in true_ranges[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def compute_trailing_stop(
    entry_price: float,
    current_price: float,
    high_water_mark: float,
    atr: Optional[float],
    current_stop: Optional[float] = None,
    atr_mult: float = Config.SELL_TRAILING_STOP_ATR_MULT,
    min_pct: float = Config.SELL_TRAILING_STOP_MIN_PCT,
) -> float:
    """Compute updated trailing stop price.

    The stop only moves UP (locks in profit), never down.

    Args:
        entry_price: Original buy price
        current_price: Current market price
        high_water_mark: Highest price seen since entry
        atr: ATR-14 value (can be None → use min_pct floor)
        current_stop: Previous stop price (if any)
        atr_mult: Multiplier for ATR (default 2.0)
        min_pct: Minimum stop distance from high as fraction

    Returns:
        New trailing stop price
    """
    if entry_price <= 0 or current_price <= 0:
        return entry_price * (1 + Config.SELL_HARD_STOP_TACTICAL)

    # Update high-water mark
    effective_high = max(current_price, high_water_mark or entry_price)

    # Compute ATR-based stop
    if atr and atr > 0:
        atr_stop = effective_high - (atr_mult * atr)
    else:
        # No ATR available: use min_pct floor
        atr_stop = effective_high * (1 - min_pct)

    # Apply floor: never below min_pct from high
    min_stop = effective_high * (1 - min_pct)
    computed_stop = max(atr_stop, min_stop)

    # Hard floor: never below the original hard stop
    absolute_floor = entry_price * (1 + Config.SELL_HARD_STOP_TACTICAL)
    computed_stop = max(computed_stop, absolute_floor)

    # Ratchet rule: stop never moves down
    if current_stop and current_stop > 0:
        computed_stop = max(computed_stop, current_stop)

    return round(computed_stop, 4)


def check_trailing_stop_breach(
    current_price: float,
    trailing_stop: float,
) -> bool:
    """Return True if the current price is at or below the trailing stop."""
    if trailing_stop <= 0:
        return False
    return current_price <= trailing_stop


class TrailingStopManager:
    """Per-position trailing stop management with persistence via ThesisTracker."""

    def __init__(self) -> None:
        # Lazy import to avoid circular
        pass

    def update_position_trailing_stop(
        self,
        thesis: Any,
        current_price: float,
        price_history: Optional[list[dict]] = None,
    ) -> tuple[float, bool]:
        """Update trailing stop for a position and check for breach.

        Args:
            thesis: PositionThesis object
            current_price: Latest price
            price_history: Recent OHLCV candles for ATR computation

        Returns:
            (new_stop_price, is_breached)
        """
        if getattr(thesis, "position_type", None) != "TACTICAL_BUY":
            return (float(thesis.hard_stop_price or 0), False)

        entry_price = float(thesis.entry_price or 0)
        if entry_price <= 0:
            return (0.0, False)

        high_water = float(thesis.trailing_stop_high or entry_price)
        current_stop = float(thesis.trailing_stop_price or thesis.hard_stop_price or 0)

        # Update high-water mark
        new_high = max(high_water, current_price)

        # Compute ATR if price history available
        atr = None
        if price_history:
            atr = compute_atr(price_history)
            if atr:
                logger.debug(
                    "[trailing_stop] ATR-14=%.4f for %s", atr, thesis.ticker
                )

        new_stop = compute_trailing_stop(
            entry_price=entry_price,
            current_price=current_price,
            high_water_mark=new_high,
            atr=atr,
            current_stop=current_stop,
        )

        is_breached = check_trailing_stop_breach(current_price, new_stop)

        # Persist updated trailing stop to thesis
        try:
            from .thesis_tracker import ThesisTracker
            tracker = ThesisTracker()
            tracker.update_trailing_stop(
                thesis.thesis_id,
                new_stop=new_stop,
                new_high=new_high,
            )
        except Exception as persist_e:
            logger.warning("[trailing_stop] Failed to persist stop update: %s", persist_e)

        if new_stop != current_stop:
            logger.info(
                "[trailing_stop] Updated %s: stop %.2f → %.2f (high=%.2f, breached=%s)",
                thesis.ticker,
                current_stop,
                new_stop,
                new_high,
                is_breached,
            )

        return (new_stop, is_breached)
