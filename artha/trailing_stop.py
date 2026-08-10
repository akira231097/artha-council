"""Trailing stop management for all buy-type positions (exit engine v2).

Implements the evidence-backed exit stack from the strategy-overhaul rulebook:

  Rule 4.1 — initial stop: entry -8% or entry - 2.5×ATR14, whichever tighter,
             hard-capped at 10% below entry. Never widened, ever.
  Rule 4.4 — cushion-gated wide trail: trailing only activates after the
             position has a cushion (+2R or +20%). Once active:
               - Chandelier stop = high-since-entry - 3.0×ATR22
               - NEVER trail tighter than max-distance rule: the stop must sit
                 at least 10% (and at least 2.5×ATR) below the high
               - exit on first daily CLOSE below the 20-day EMA
                 (10-day EMA for high-ADR names, ADR > 6%)
  The stop only ever ratchets UP — it never moves down.

The previous implementation trailed at min(2×ATR, 8%-from-high) — the tightest
of the two — which caused both historical whipsaw stop-outs. The new trail uses
the WIDEST distance (max of Chandelier and the 10% floor) and relies on the
EMA-close exit for the earlier, structure-based exit.
"""
from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Optional
from zoneinfo import ZoneInfo

from .config import Config

logger = logging.getLogger(__name__)

# Position types managed by the trailing stop engine (all buy types).
TRAIL_MANAGED_POSITION_TYPES = frozenset(
    {"BUY", "STARTER", "TACTICAL_BUY", "ACCUMULATE", "ADD"}
)


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


def _candle_val(candle: dict, *keys: str) -> float:
    for key in keys:
        try:
            value = candle.get(key)
        except AttributeError:
            return 0.0
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def compute_atr(price_history: list[dict], period: int = 14) -> Optional[float]:
    """Compute ATR from a list of OHLCV candles (Wilder smoothing).

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
            high = _candle_val(c, "high", "h")
            low = _candle_val(c, "low", "l")
            prev_close = _candle_val(prev, "close", "c", "adjClose")
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


def compute_ema(closes: list[float], period: int) -> Optional[float]:
    """Compute an exponential moving average over closing prices."""
    values = [float(c) for c in closes if c and float(c) > 0]
    if len(values) < period:
        return None
    ema = sum(values[:period]) / period
    mult = 2.0 / (period + 1)
    for value in values[period:]:
        ema = (value - ema) * mult + ema
    return ema


def compute_adr_pct(price_history: list[dict], period: int = 20) -> Optional[float]:
    """Average daily range as a fraction (mean of high/low - 1) over `period` bars."""
    if not price_history:
        return None
    ranges = []
    for candle in price_history[-period:]:
        high = _candle_val(candle, "high", "h")
        low = _candle_val(candle, "low", "l")
        if high > 0 and low > 0:
            ranges.append(high / low - 1.0)
    if not ranges:
        return None
    return sum(ranges) / len(ranges)


def compute_initial_stop(
    entry_price: float,
    atr: Optional[float] = None,
    stop_pct: float = Config.SELL_INITIAL_STOP_PCT,
    atr_mult: float = Config.SELL_INITIAL_STOP_ATR_MULT,
    cap_pct: float = Config.SELL_INITIAL_STOP_CAP_PCT,
) -> float:
    """Rule 4.1 initial stop: entry -8% or entry - 2.5×ATR14, whichever tighter.

    The stop distance never exceeds `cap_pct` (10%) below entry.
    """
    if entry_price <= 0:
        return 0.0
    distance = stop_pct * entry_price
    if atr and atr > 0:
        distance = min(distance, atr_mult * atr)
    distance = min(distance, cap_pct * entry_price)
    return round(entry_price - distance, 4)


def cushion_reached(
    entry_price: float,
    high_water_mark: float,
    hard_stop_price: Optional[float] = None,
    r_mult: float = Config.SELL_TRAIL_CUSHION_R_MULT,
    cushion_pct: float = Config.SELL_TRAIL_CUSHION_PCT,
) -> bool:
    """Rule 4.4 cushion gate: trail only after +2R or +20% from entry.

    R is the initial risk (entry - hard stop); when the hard stop is unknown
    the default initial-stop percentage is used.
    """
    if entry_price <= 0 or high_water_mark <= 0:
        return False
    if high_water_mark >= entry_price * (1 + cushion_pct):
        return True
    if hard_stop_price and 0 < hard_stop_price < entry_price:
        r = entry_price - hard_stop_price
    else:
        r = entry_price * Config.SELL_INITIAL_STOP_PCT
    return r > 0 and high_water_mark >= entry_price + r_mult * r


def compute_chandelier_stop(
    high_water_mark: float,
    atr: Optional[float],
    atr_mult: float = Config.SELL_TRAIL_ATR_MULT,
    min_pct: float = Config.SELL_TRAIL_MIN_PCT,
) -> float:
    """Chandelier stop off the high-since-entry, never tighter than 10%.

    Distance below the high is max(atr_mult × ATR, min_pct × high) so the
    trail is NEVER tighter than 2.5×ATR or 10% (rule 4.4). The high-water mark
    must be the high SINCE ENTRY (pre-entry spikes would poison the trail).
    """
    if high_water_mark <= 0:
        return 0.0
    pct_stop = high_water_mark * (1 - min_pct)
    if atr and atr > 0:
        atr_stop = high_water_mark - atr_mult * atr
        # min() = widest distance: never tighter than either constraint
        return round(min(atr_stop, pct_stop), 4)
    return round(pct_stop, 4)


def compute_trailing_stop(
    entry_price: float,
    current_price: float,
    high_water_mark: float,
    atr: Optional[float],
    current_stop: Optional[float] = None,
    atr_mult: float = Config.SELL_TRAIL_ATR_MULT,
    min_pct: float = Config.SELL_TRAIL_MIN_PCT,
) -> float:
    """Compute updated trailing stop price (cushion already established).

    The stop only moves UP (locks in profit), never down.

    Args:
        entry_price: Original buy price
        current_price: Current market price
        high_water_mark: Highest price seen since entry
        atr: ATR-22 value (can be None → use min_pct distance)
        current_stop: Previous stop price (if any)
        atr_mult: Chandelier ATR multiplier (default 3.0)
        min_pct: Minimum stop distance from high as fraction (default 0.10)

    Returns:
        New trailing stop price
    """
    if entry_price <= 0 or current_price <= 0:
        return float(current_stop or 0)

    effective_high = max(current_price, high_water_mark or entry_price)
    computed_stop = compute_chandelier_stop(
        high_water_mark=effective_high,
        atr=atr,
        atr_mult=atr_mult,
        min_pct=min_pct,
    )

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


def check_ema_close_exit(
    price_history: list[dict],
    ema_period: Optional[int] = None,
) -> tuple[bool, dict[str, Any]]:
    """Rule 4.4 EMA-close exit: first daily CLOSE below the 20-day EMA
    (10-day EMA for high-ADR names).

    Evaluates the most recent completed daily bar against the EMA computed
    over all bars up to and including it. Returns (breached, details).
    """
    details: dict[str, Any] = {"ema_period": ema_period, "ema": None, "last_close": None}
    if not price_history or len(price_history) < 12:
        return (False, details)

    # Normalize to chronological order (FMP returns are already ascending in
    # collector output, but be defensive when dates are present).
    bars = list(price_history)
    try:
        if bars[0].get("date") and bars[-1].get("date") and bars[0]["date"] > bars[-1]["date"]:
            bars = bars[::-1]
    except Exception:
        pass

    # Rule 4.4 requires a COMPLETED daily close. Vendor histories (FMP
    # historical-price-eod/full and the yfinance fallback) include today's
    # in-progress bar during market hours, whose "close" is just the live
    # price — exclude any bar dated today (US/Eastern) so an intraday dip
    # below the EMA can never be treated as a daily close.
    today_et = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    bars = [b for b in bars if str(b.get("date") or "")[:10] != today_et]
    if len(bars) < 12:
        return (False, details)

    if ema_period is None:
        adr = compute_adr_pct(bars)
        ema_period = (
            Config.SELL_TRAIL_EMA_PERIOD_FAST
            if adr is not None and adr > Config.SELL_TRAIL_HIGH_ADR_PCT
            else Config.SELL_TRAIL_EMA_PERIOD
        )
        details["adr_pct"] = adr

    closes = [_candle_val(b, "close", "c", "adjClose") for b in bars]
    closes = [c for c in closes if c > 0]
    if len(closes) < ema_period + 1:
        return (False, details)

    last_close = closes[-1]
    ema = compute_ema(closes, ema_period)
    details.update({"ema_period": ema_period, "ema": ema, "last_close": last_close})
    if ema is None:
        return (False, details)
    return (last_close < ema, details)


class TrailingStopManager:
    """Per-position trailing stop management with persistence via ThesisTracker."""

    def __init__(self, tracker: Any = None) -> None:
        # Lazy import to avoid circular; tracker is injected by SellEngine so
        # persistence hits the same journal the engine reads from.
        self._tracker = tracker

    def _get_tracker(self) -> Any:
        if self._tracker is None:
            from .thesis_tracker import ThesisTracker
            self._tracker = ThesisTracker()
        return self._tracker

    def evaluate_trail(
        self,
        thesis: Any,
        current_price: float,
        price_history: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """Full rule-4.4 evaluation for one position.

        Returns dict with keys:
            new_stop (float), is_breached (bool), cushioned (bool),
            ema_exit (bool), ema_details (dict), high_water (float)
        """
        result: dict[str, Any] = {
            "new_stop": float(getattr(thesis, "trailing_stop_price", 0) or getattr(thesis, "hard_stop_price", 0) or 0),
            "is_breached": False,
            "cushioned": False,
            "ema_exit": False,
            "ema_details": {},
            "high_water": 0.0,
        }
        position_type = getattr(thesis, "position_type", None)
        if position_type not in TRAIL_MANAGED_POSITION_TYPES:
            return result

        entry_price = float(getattr(thesis, "entry_price", 0) or 0)
        if entry_price <= 0 or current_price <= 0:
            return result

        hard_stop = float(getattr(thesis, "hard_stop_price", 0) or 0)
        high_water = float(getattr(thesis, "trailing_stop_high", 0) or entry_price)
        current_stop = float(
            getattr(thesis, "trailing_stop_price", 0) or hard_stop or 0
        )
        new_high = max(high_water, current_price)
        result["high_water"] = new_high

        cushioned = cushion_reached(entry_price, new_high, hard_stop)
        result["cushioned"] = cushioned

        new_stop = current_stop
        if cushioned:
            atr = None
            if price_history:
                atr = compute_atr(
                    price_history, period=Config.SELL_CHANDELIER_LOOKBACK_DAYS
                )
                if atr:
                    logger.debug(
                        "[trailing_stop] ATR-%d=%.4f for %s",
                        Config.SELL_CHANDELIER_LOOKBACK_DAYS,
                        atr,
                        getattr(thesis, "ticker", "?"),
                    )
            new_stop = compute_trailing_stop(
                entry_price=entry_price,
                current_price=current_price,
                high_water_mark=new_high,
                atr=atr,
                current_stop=current_stop,
            )
            # EMA-close exit only applies once the cushion is established
            ema_exit, ema_details = check_ema_close_exit(price_history or [])
            result["ema_exit"] = ema_exit
            result["ema_details"] = ema_details

        # Never move a stop down — ratchet only (rule 4.8: no widening, ever)
        if current_stop > 0:
            new_stop = max(new_stop, current_stop)

        result["new_stop"] = round(new_stop, 4) if new_stop else new_stop
        result["is_breached"] = check_trailing_stop_breach(current_price, result["new_stop"])

        # Persist updated trailing stop + high-water mark to thesis
        if result["new_stop"] != current_stop or new_high != high_water:
            try:
                tracker = self._get_tracker()
                tracker.update_trailing_stop(
                    thesis.thesis_id,
                    new_stop=result["new_stop"],
                    new_high=new_high,
                )
            except Exception as persist_e:
                logger.warning("[trailing_stop] Failed to persist stop update: %s", persist_e)

        if result["new_stop"] != current_stop:
            logger.info(
                "[trailing_stop] Updated %s: stop %.2f → %.2f (high=%.2f, cushioned=%s, breached=%s)",
                getattr(thesis, "ticker", "?"),
                current_stop,
                result["new_stop"],
                new_high,
                cushioned,
                result["is_breached"],
            )

        return result

    def update_position_trailing_stop(
        self,
        thesis: Any,
        current_price: float,
        price_history: Optional[list[dict]] = None,
    ) -> tuple[float, bool]:
        """Update trailing stop for a position and check for breach.

        Backward-compatible wrapper around :meth:`evaluate_trail`. Now applies
        to ALL buy-type positions (previously TACTICAL_BUY only).

        Args:
            thesis: PositionThesis object
            current_price: Latest price
            price_history: Recent OHLCV candles for ATR computation

        Returns:
            (new_stop_price, is_breached)
        """
        if getattr(thesis, "position_type", None) not in TRAIL_MANAGED_POSITION_TYPES:
            return (float(getattr(thesis, "hard_stop_price", 0) or 0), False)

        result = self.evaluate_trail(
            thesis=thesis,
            current_price=current_price,
            price_history=price_history,
        )
        return (float(result["new_stop"] or 0), bool(result["is_breached"]))
