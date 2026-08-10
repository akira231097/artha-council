"""Earnings Calendar — fetches upcoming and recent earnings for tickers.

Uses FMP earnings-calendar endpoint. Provides earnings context
for the council analysis and promotion funnel.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from .collector import _safe_get
from .config import Config

logger = logging.getLogger(__name__)

UTC = timezone.utc

# Process-wide market-calendar cache: {"YYYY-MM-DD": rows}. The full 90-day
# market calendar is ~4000 rows; it is only a fallback (the symbol-scoped
# `earnings` endpoint is preferred) and never fetched more than once per day.
_MARKET_CALENDAR_CACHE: dict[str, list] = {}


@dataclass
class EarningsContext:
    """Earnings timing and risk context for a single ticker."""
    ticker: str = ""
    earnings_date: Optional[str] = None         # ISO date (YYYY-MM-DD)
    days_to_earnings: Optional[int] = None      # None if no upcoming earnings found
    earnings_time: str = "unknown"              # "bmo" (before market open), "amc" (after), "unknown"
    earnings_risk_flag: bool = False            # True if earnings within 7 days
    earnings_defer_flag: bool = False           # True if earnings within 2 days (defer new positions)
    recent_surprises: list[dict] = field(default_factory=list)  # Last 4 quarters
    as_of: str = ""


class EarningsCalendar:
    """Fetches earnings calendar data from FMP."""

    def __init__(self):
        self.base = Config.FMP_BASE_URL
        self.key = Config.FMP_API_KEY

    def fetch_upcoming(self, from_date: str, to_date: str) -> list[dict]:
        """Fetch earnings calendar for a date range.

        Args:
            from_date: YYYY-MM-DD
            to_date: YYYY-MM-DD

        Returns:
            List of raw earnings dicts from FMP.
        """
        try:
            url = f"{self.base}/earnings-calendar"
            params = {
                "apikey": self.key,
                "from": from_date,
                "to": to_date,
            }
            data = _safe_get(url, params, "fmp")
            if isinstance(data, list):
                return data
            logger.warning(f"[earnings_calendar] Unexpected response type: {type(data)}")
            return []
        except Exception as e:
            logger.warning(f"[earnings_calendar] FMP fetch failed: {e}")
            return []

    def fetch_symbol_earnings(self, ticker: str, limit: int = 12) -> Optional[list[dict]]:
        """Fetch the symbol-scoped FMP `earnings` feed (past + next earnings rows).

        This is the cheap per-ticker path: the stable `earnings-calendar`
        endpoint ignores its symbol param and always returns the whole market,
        so per-ticker context uses `earnings?symbol=X` instead (verified live
        2026-07-02: includes the upcoming report date with epsEstimated).

        Returns:
            List of raw FMP earnings dicts (newest first), or None on failure.
        """
        try:
            url = f"{self.base}/earnings"
            params = {"apikey": self.key, "symbol": ticker, "limit": str(limit)}
            data = _safe_get(url, params, "fmp")
            if isinstance(data, list):
                return data
            return None
        except Exception as e:
            logger.warning(f"[earnings_calendar] Symbol earnings fetch failed for {ticker}: {e}")
            return None

    def fetch_market_calendar_cached(self) -> list[dict]:
        """Market-wide 90-day earnings calendar, fetched at most once per UTC day.

        Shared across all tickers in the process — fallback for when the
        symbol-scoped `earnings` endpoint fails. FMP hard-caps the response at
        4000 rows regardless of `limit` (verified live 2026-07-02: a single
        90-day request silently dropped AAPL), so the window is fetched in
        7-day chunks and concatenated.
        """
        now_utc = datetime.now(UTC)
        today = now_utc.strftime("%Y-%m-%d")
        cached = _MARKET_CALENDAR_CACHE.get(today)
        if cached is not None:
            return cached

        rows: list[dict] = []
        chunk_days = 7
        for offset in range(0, 90, chunk_days):
            from_str = (now_utc + timedelta(days=offset)).strftime("%Y-%m-%d")
            to_str = (now_utc + timedelta(days=offset + chunk_days - 1)).strftime("%Y-%m-%d")
            chunk = self.fetch_upcoming(from_str, to_str)
            if len(chunk) >= 4000:
                logger.warning(
                    "[earnings_calendar] Calendar chunk %s..%s hit the 4000-row FMP cap; "
                    "some symbols may be missing", from_str, to_str,
                )
            rows.extend(chunk)
        if rows:
            _MARKET_CALENDAR_CACHE.clear()
            _MARKET_CALENDAR_CACHE[today] = rows
        return rows

    def get_earnings_context(self, ticker: str, finnhub_data: list | None = None) -> EarningsContext:
        """Get earnings timing and risk context for a single ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            EarningsContext with all available fields populated.
        """
        now_utc = datetime.now(UTC)
        ctx = EarningsContext(
            ticker=ticker,
            as_of=now_utc.isoformat(),
        )

        # Preferred: symbol-scoped earnings feed (one small per-ticker call).
        # Fallback: market-wide calendar cached once per day for the process.
        today_midnight = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        ticker_upper = ticker.upper()
        symbol_rows = self.fetch_symbol_earnings(ticker)
        if symbol_rows is not None:
            upcoming = symbol_rows
        else:
            upcoming = self.fetch_market_calendar_cached()

        # Find this ticker's next earnings — collect all matches, pick earliest date
        matches = []
        for item in upcoming:
            if not isinstance(item, dict):
                continue
            if item.get("symbol", "").upper() != ticker_upper:
                continue
            earnings_date_str = item.get("date", "") or ""
            if not earnings_date_str:
                continue
            try:
                earnings_dt = datetime.strptime(earnings_date_str, "%Y-%m-%d").replace(tzinfo=UTC)
            except ValueError:
                continue
            if earnings_dt < today_midnight:
                continue  # symbol feed includes past quarters; only future counts
            matches.append((earnings_date_str, earnings_dt, item))

        if matches:
            # Sort by date ascending, pick the earliest upcoming date
            matches.sort(key=lambda x: x[1])
            earnings_date_str, earnings_dt, item = matches[0]

            days_to = (earnings_dt - now_utc.replace(hour=0, minute=0, second=0, microsecond=0)).days
            ctx.earnings_date = earnings_date_str
            ctx.days_to_earnings = max(0, days_to)

            # Determine timing (before/after market)
            raw_time = item.get("time", "") or ""
            if raw_time.lower() in ("bmo", "before market open", "pre-market"):
                ctx.earnings_time = "bmo"
            elif raw_time.lower() in ("amc", "after market close", "after-hours"):
                ctx.earnings_time = "amc"
            else:
                ctx.earnings_time = "unknown"

            ctx.earnings_risk_flag = ctx.days_to_earnings <= 7
            ctx.earnings_defer_flag = ctx.days_to_earnings <= 2

        # Fetch recent earnings surprises (Finnhub-style or FMP historical).
        # Reuse the symbol-scoped rows so no second FMP call is needed.
        ctx.recent_surprises = self._fetch_recent_surprises(
            ticker, finnhub_data=finnhub_data, fmp_rows=symbol_rows
        )

        return ctx

    def _fetch_recent_surprises(
        self,
        ticker: str,
        finnhub_data: list | None = None,
        fmp_rows: list | None = None,
    ) -> list[dict]:
        """Fetch last 4 quarters of earnings surprises.

        Uses pre-fetched Finnhub data if available, then pre-fetched FMP
        symbol earnings rows, and only calls FMP directly as a last resort.
        """
        # Use Finnhub data if provided (already collected by DataCollector)
        if finnhub_data:
            surprises = []
            for item in finnhub_data[:4]:
                if not isinstance(item, dict):
                    continue
                surprises.append({
                    "date": item.get("period", ""),
                    "actual": item.get("actual"),
                    "estimated": item.get("estimate"),
                    "surprise_pct": item.get("surprisePercent"),
                })
            if surprises:
                return surprises

        # Fallback to FMP (reuse pre-fetched symbol rows when available)
        try:
            data = fmp_rows
            if data is None:
                data = self.fetch_symbol_earnings(ticker)
            if isinstance(data, list):
                surprises = []
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    actual = item.get("epsActual") or item.get("actualEarningResult") or item.get("actual")
                    estimated = item.get("epsEstimated") or item.get("estimatedEarning") or item.get("estimate")
                    if actual is None:
                        continue  # future/unreported quarter — not a surprise row
                    surprises.append({
                        "date": item.get("date", ""),
                        "actual": actual,
                        "estimated": estimated,
                        "surprise_pct": _compute_surprise_pct(actual, estimated),
                    })
                    if len(surprises) >= 4:
                        break
                return surprises
        except Exception as e:
            logger.warning(f"[earnings_calendar] Surprises fetch failed for {ticker}: {e}")
        return []


def _compute_surprise_pct(actual, estimated) -> Optional[float]:
    """Compute earnings surprise percentage."""
    try:
        a, e = float(actual), float(estimated)
        if e == 0:
            return None
        return round(((a - e) / abs(e)) * 100, 2)
    except (TypeError, ValueError):
        return None


# Module-level singleton for convenience
_calendar = None


def get_earnings_context(ticker: str, finnhub_data: list | None = None) -> EarningsContext:
    """Get earnings context for a ticker (uses module-level singleton).

    Convenience function — creates EarningsCalendar once per process.
    """
    global _calendar
    if _calendar is None:
        _calendar = EarningsCalendar()
    return _calendar.get_earnings_context(ticker, finnhub_data=finnhub_data)
