"""Data collection from all financial APIs.

Handles rate limiting, error handling, and data normalization
for FMP, Massive, Finnhub, Alpha Vantage, CoinGecko, FRED, and yfinance.
"""
import time
import logging
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Optional

import requests
import yfinance as yf

from .config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple per-source rate limiter."""

    def __init__(self, calls_per_minute: int):
        self.interval = 60.0 / calls_per_minute
        self.last_call: float = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_call = time.monotonic()


# Rate limiters per API
_limiters = {
    "fmp": RateLimiter(max(1, Config.FMP_CALLS_PER_MINUTE)),
    "massive": RateLimiter(max(1, Config.MASSIVE_CALLS_PER_MINUTE)),
    "benzinga": RateLimiter(max(1, Config.BENZINGA_CALLS_PER_MINUTE)),
    "finnhub": RateLimiter(30),    # 60/min limit; 30 = safe margin (halved to avoid 403s)
    "coingecko": RateLimiter(25),  # 30/min limit
    "fred": RateLimiter(max(1, Config.FRED_CALLS_PER_MINUTE)),
}

# Shared HTTP session for connection pooling (avoids per-call TCP setup)
_http_session = requests.Session()
_response_cache: dict[tuple[str, str, tuple[tuple[str, str], ...]], tuple[float, Any]] = {}


def _utcnow_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _extract_api_error(payload: Any) -> Optional[str]:
    """Best-effort extraction of provider error/rate-limit payloads."""
    if not isinstance(payload, dict):
        return None

    if isinstance(payload.get("status"), dict):
        status = payload["status"]
        if status.get("error_message"):
            return str(status.get("error_message"))

    for key in ("error", "Error", "Error Message", "Information", "Note", "message"):
        value = payload.get(key)
        if value:
            return str(value)

    return None


def _expect_list(payload: Any, source: str, context: str) -> Optional[list]:
    """Normalize list-shaped API responses."""
    if payload is None:
        return None
    if isinstance(payload, list):
        return payload
    logger.warning(f"[{source}] Unexpected response format for {context}: expected list")
    return None


def _expect_dict(payload: Any, source: str, context: str) -> Optional[dict]:
    """Normalize dict-shaped API responses."""
    if payload is None:
        return None
    if isinstance(payload, dict):
        return payload
    logger.warning(f"[{source}] Unexpected response format for {context}: expected dict")
    return None


import re

_SENSITIVE_PARAMS = re.compile(r'(api_key|apikey|token|key)=[^&\s]+', re.IGNORECASE)
_SENSITIVE_KEYS = re.compile(r'^(api_key|apikey|token|key)$', re.IGNORECASE)

def _sanitize_url(text: str) -> str:
    """Mask API keys in URLs for safe logging."""
    return _SENSITIVE_PARAMS.sub(r'\1=***', str(text))

def _cache_key(url: str, params: dict, source: str) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    public_params = tuple(
        sorted(
            (str(k), str(v))
            for k, v in (params or {}).items()
            if not _SENSITIVE_KEYS.search(str(k))
        )
    )
    return (source, url, public_params)


def _cache_ttl(source: str) -> int:
    if source == "fmp":
        return Config.FMP_CACHE_TTL_SECONDS
    if source == "massive":
        return Config.MASSIVE_CACHE_TTL_SECONDS
    if source == "fred":
        return Config.FRED_CACHE_TTL_SECONDS
    return 0


def _retry_wait_seconds(resp: requests.Response | None, attempt: int) -> float:
    retry_after = (resp.headers.get("Retry-After") if resp is not None else None) or ""
    try:
        if retry_after:
            return min(30.0, max(1.0, float(retry_after)))
    except ValueError:
        pass
    return min(30.0, Config.FMP_429_BACKOFF_SECONDS * (2 ** attempt))


def _safe_get(
    url: str,
    params: dict,
    source: str,
    timeout: int = 15,
    retries: Optional[int] = None,
    use_cache: bool = True,
) -> Optional[dict | list]:
    """HTTP GET with rate limiting, success caching, and 429 backoff."""
    ttl = _cache_ttl(source)
    key = _cache_key(url, params, source)
    if use_cache and ttl > 0:
        cached = _response_cache.get(key)
        if cached and time.time() - cached[0] <= ttl:
            return cached[1]
        if cached:
            _response_cache.pop(key, None)

    attempts = (Config.FMP_429_RETRIES if source == "fmp" else 1) if retries is None else retries
    for attempt in range(max(0, attempts) + 1):
        limiter = _limiters.get(source)
        if limiter is not None:
            limiter.wait()
        try:
            resp = _http_session.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = _retry_wait_seconds(resp, attempt)
                logger.warning(
                    "[%s] HTTP 429 from %s (attempt %d/%d); backing off %.1fs",
                    source,
                    _sanitize_url(resp.url or url),
                    attempt + 1,
                    attempts + 1,
                    wait,
                )
                if attempt < attempts:
                    time.sleep(wait)
                    continue
                return None

            resp.raise_for_status()
            payload = resp.json()
            error_msg = _extract_api_error(payload)
            if error_msg:
                logger.warning(f"[{source}] API returned error payload: {error_msg}")
                return None
            if use_cache and ttl > 0:
                _response_cache[key] = (time.time(), payload)
            return payload
        except requests.exceptions.Timeout:
            logger.warning(f"[{source}] Timeout fetching {_sanitize_url(url)}")
            if attempt < attempts:
                time.sleep(min(10.0, 1.5 * (2 ** attempt)))
                continue
            return None
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else "unknown"
            response_url = getattr(e.response, "url", url)
            logger.warning(f"[{source}] HTTP {status_code} from {_sanitize_url(response_url)}")
            return None
        except ValueError:
            logger.warning(f"[{source}] Non-JSON response from {_sanitize_url(url)}")
            return None
        except Exception as e:
            logger.error(f"[{source}] Unexpected error: {_sanitize_url(e)}")
            return None
    return None




# ---------------------------------------------------------------------------
# Retry Helper for Rate-Limited APIs
# ---------------------------------------------------------------------------

def _retry_on_failure(fn, retries: int = 2, backoff: float = 2.0, source: str = "unknown"):
    """Retry a function call with exponential backoff on failure.
    
    Returns (result, was_retried) tuple. Result is None if all retries exhausted.
    """
    last_error = None
    for attempt in range(retries + 1):
        try:
            result = fn()
            if result is not None:
                return result, attempt > 0
        except Exception as e:
            last_error = e
            if attempt < retries:
                wait = backoff * (2 ** attempt)
                logger.warning(f"[{source}] Attempt {attempt + 1} failed: {e}. Retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                logger.warning(f"[{source}] All {retries + 1} attempts failed: {e}")
    return None, True

# ---------------------------------------------------------------------------
# FMP (Financial Modeling Prep) — Primary Data Source
# ---------------------------------------------------------------------------

class FMPCollector:
    """Financial Modeling Prep — Stable API (post-Aug 2025).
    
    Key differences from legacy v3:
    - Base URL: /stable/ instead of /api/v3/
    - Symbol passed as query param (?symbol=X) not path param (/X)
    - Some endpoints renamed (stock_news -> news/stock)
    - Sarath has FMP Premium; use quarterlies plus TTM endpoints for council data
    - Insider trading and some specialized datasets still use Finnhub/other fallbacks
    """

    def __init__(self):
        self.base = Config.FMP_BASE_URL
        self.key = Config.FMP_API_KEY

    def _get(
        self,
        endpoint: str,
        extra_params: dict | None = None,
        *,
        timeout: int | None = None,
        retries: int | None = None,
    ) -> Optional[Any]:
        params = {"apikey": self.key}
        if extra_params:
            params.update(extra_params)
        return _safe_get(
            f"{self.base}/{endpoint}",
            params,
            "fmp",
            timeout=timeout or 15,
            retries=retries,
        )

    def _get_first(
        self,
        endpoint: str,
        extra_params: dict | None = None,
        *,
        timeout: int | None = None,
        retries: int | None = None,
    ) -> Optional[dict]:
        """Get first item from a list response."""
        data = self._get(endpoint, extra_params, timeout=timeout, retries=retries)
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
        return None

    def quote(self, ticker: str) -> Optional[dict]:
        """Real-time quote for a stock or crypto."""
        return self._get_first("quote", {"symbol": ticker})

    def company_profile(self, ticker: str) -> Optional[dict]:
        """Company profile / overview."""
        return self._get_first("profile", {"symbol": ticker})

    def income_statement(self, ticker: str, period: str = "annual", limit: int = 4) -> Optional[list]:
        """Income statements (annual or quarter)."""
        return _expect_list(
            self._get("income-statement", {"symbol": ticker, "period": period, "limit": str(limit)}),
            "fmp", f"income_statement:{ticker}",
        )

    def balance_sheet(self, ticker: str, period: str = "annual", limit: int = 4) -> Optional[list]:
        """Balance sheet data."""
        return _expect_list(
            self._get("balance-sheet-statement", {"symbol": ticker, "period": period, "limit": str(limit)}),
            "fmp", f"balance_sheet:{ticker}",
        )

    def cash_flow(self, ticker: str, period: str = "annual", limit: int = 4) -> Optional[list]:
        """Cash flow statements."""
        return _expect_list(
            self._get("cash-flow-statement", {"symbol": ticker, "period": period, "limit": str(limit)}),
            "fmp", f"cash_flow:{ticker}",
        )

    def ratios(self, ticker: str, limit: int = 4) -> Optional[list]:
        """Financial ratios (annual — quarterly needs premium)."""
        return _expect_list(
            self._get("ratios", {"symbol": ticker, "limit": str(limit)}),
            "fmp", f"ratios:{ticker}",
        )

    def ratios_ttm(self, ticker: str, *, timeout: int | None = None, retries: int | None = None) -> Optional[dict]:
        """Trailing twelve months financial ratios."""
        return self._get_first("ratios-ttm", {"symbol": ticker}, timeout=timeout, retries=retries)

    def key_metrics(self, ticker: str, limit: int = 4) -> Optional[list]:
        """Key financial metrics (annual)."""
        return _expect_list(
            self._get("key-metrics", {"symbol": ticker, "limit": str(limit)}),
            "fmp", f"key_metrics:{ticker}",
        )

    def key_metrics_ttm(self, ticker: str, *, timeout: int | None = None, retries: int | None = None) -> Optional[dict]:
        """Trailing twelve months key metrics."""
        return self._get_first("key-metrics-ttm", {"symbol": ticker}, timeout=timeout, retries=retries)

    def dcf(self, ticker: str, *, timeout: int | None = None, retries: int | None = None) -> Optional[dict]:
        """Discounted cash flow valuation."""
        return self._get_first("discounted-cash-flow", {"symbol": ticker}, timeout=timeout, retries=retries)

    def price_target_consensus(
        self,
        ticker: str,
        *,
        timeout: int | None = None,
        retries: int | None = None,
    ) -> Optional[dict]:
        """Wall Street analyst price target consensus."""
        return self._get_first("price-target-consensus", {"symbol": ticker}, timeout=timeout, retries=retries)

    @staticmethod
    def _period_to_days(period: str) -> int:
        text = str(period or "1y").strip().lower()
        try:
            if text.endswith("mo"):
                return max(1, int(float(text[:-2]) * 31))
            if text.endswith("y"):
                return max(1, int(float(text[:-1]) * 366))
            if text.endswith("d"):
                return max(1, int(float(text[:-1])))
        except Exception:
            pass
        return 366

    def history(self, ticker: str, period: str = "1y") -> Optional[list[dict]]:
        """Adjusted historical EOD OHLCV from FMP paid/stable API."""
        symbol = str(ticker or "").strip().upper()
        if not symbol:
            return None
        days = self._period_to_days(period)
        to_date = datetime.now(timezone.utc).date()
        from_date = to_date - timedelta(days=days)
        rows = _expect_list(
            self._get(
                "historical-price-eod/full",
                {
                    "symbol": symbol,
                    "from": from_date.isoformat(),
                    "to": to_date.isoformat(),
                },
            ),
            "fmp",
            f"history:{symbol}",
        )
        if not rows:
            return None

        records: list[dict] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                close = float(row.get("close") or 0)
            except Exception:
                close = 0.0
            if close <= 0:
                continue
            records.append({
                "date": str(row.get("date") or ""),
                "open": round(float(row.get("open") or close), 4),
                "high": round(float(row.get("high") or close), 4),
                "low": round(float(row.get("low") or close), 4),
                "close": round(close, 4),
                "volume": int(float(row.get("volume") or 0)),
                "vwap": row.get("vwap"),
                "source": "fmp.historical_price_eod",
            })
        records.sort(key=lambda item: str(item.get("date") or ""))
        return records or None

    def stock_news(self, ticker: str, limit: int = 10) -> Optional[list]:
        """Recent news for a stock (stable API: news/stock)."""
        return _expect_list(
            self._get("news/stock", {"symbol": ticker, "limit": str(limit)}),
            "fmp", f"stock_news:{ticker}",
        )

    def crypto_quote(self, symbol: str = "BTCUSD") -> Optional[dict]:
        """Crypto quote from FMP."""
        return self._get_first("quote", {"symbol": symbol})

    def screener(
        self,
        market_cap_more_than: int = 1_000_000_000,
        volume_more_than: int = 100_000,
        price_more_than: float = 5.0,
        country: str = "US",
        is_actively_trading: bool = True,
        is_etf: bool = False,
        is_fund: bool = False,
        limit: int = 1000,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        beta_more_than: Optional[float] = None,
        beta_less_than: Optional[float] = None,
    ) -> Optional[list]:
        """Screen stocks via FMP company-screener endpoint.

        Returns list of matching company dicts (symbol, companyName, sector,
        marketCap, price, volume, etc.) or None on failure.
        """
        params: dict = {
            "marketCapMoreThan": str(int(market_cap_more_than)),
            "volumeMoreThan": str(int(volume_more_than)),
            "priceMoreThan": str(price_more_than),
            "country": country,
            "isActivelyTrading": "true" if is_actively_trading else "false",
            "isEtf": "true" if is_etf else "false",
            "isFund": "true" if is_fund else "false",
            "limit": str(limit),
        }
        if sector:
            params["sector"] = sector
        if industry:
            params["industry"] = industry
        if beta_more_than is not None:
            params["betaMoreThan"] = str(beta_more_than)
        if beta_less_than is not None:
            params["betaLessThan"] = str(beta_less_than)

        return _expect_list(
            self._get("company-screener", params),
            "fmp",
            "screener",
        )

    def market_gainers(self, limit: int = 10) -> Optional[list]:
        """Top market gainers (FMP biggest-gainers endpoint)."""
        data = self._get("biggest-gainers")
        if data and isinstance(data, list):
            return data[:limit]
        return None

    def market_losers(self, limit: int = 10) -> Optional[list]:
        """Top market losers (FMP biggest-losers endpoint)."""
        data = self._get("biggest-losers")
        if data and isinstance(data, list):
            return data[:limit]
        return None

    def market_actives(self, limit: int = 10) -> Optional[list]:
        """Most active stocks by volume (FMP most-actives endpoint)."""
        data = self._get("most-actives")
        if data and isinstance(data, list):
            return data[:limit]
        return None


# ---------------------------------------------------------------------------
# Massive (formerly Polygon) — Market-data cross-check and yfinance fallback
# ---------------------------------------------------------------------------

class MassiveCollector:
    """Massive stock market data wrapper.

    Artha uses FMP as the primary fundamental/quote source. Massive is wired as
    an independent market-data cross-check and a guarded history fallback for
    yfinance outages. The free plan is rate-limited, so this collector is
    intentionally conservative and never blocks a scan if Massive is unavailable.
    """

    def __init__(self):
        self.base = Config.MASSIVE_BASE_URL.rstrip("/")
        self.key = Config.MASSIVE_API_KEY
        self.enabled = bool(self.key) and bool(Config.MASSIVE_ENABLED)

    def _get(self, endpoint: str, extra_params: dict | None = None) -> Optional[Any]:
        if not self.enabled:
            return None
        params = {"apiKey": self.key}
        if extra_params:
            params.update(extra_params)
        endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        return _safe_get(f"{self.base}{endpoint}", params, "massive")

    @staticmethod
    def _num(value: Any) -> Optional[float]:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _timestamp_ms_to_date(value: Any) -> str | None:
        try:
            if value is None:
                return None
            return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            return None

    @staticmethod
    def _period_to_days(period: str) -> int:
        text = str(period or "1y").strip().lower()
        try:
            if text.endswith("mo"):
                return max(1, int(float(text[:-2]) * 31))
            if text.endswith("y"):
                return max(1, int(float(text[:-1]) * 366))
            if text.endswith("d"):
                return max(1, int(float(text[:-1])))
        except Exception:
            pass
        return 366

    def previous_bar(self, ticker: str) -> Optional[dict]:
        """Return a normalized previous-day bar, available on all stock plans."""
        symbol = str(ticker or "").strip().upper()
        if not symbol:
            return None
        payload = _expect_dict(
            self._get(f"/v2/aggs/ticker/{symbol}/prev", {"adjusted": "true"}),
            "massive",
            f"previous_bar:{symbol}",
        )
        if not payload:
            return None
        rows = payload.get("results")
        if not isinstance(rows, list) or not rows:
            return None
        row = rows[0] if isinstance(rows[0], dict) else {}
        close = self._num(row.get("c"))
        volume = self._num(row.get("v"))
        return {
            "symbol": row.get("T") or symbol,
            "price": round(close, 4) if close is not None else None,
            "previous_close": round(close, 4) if close is not None else None,
            "change": None,
            "changesPercentage": None,
            "volume": int(volume) if volume is not None else None,
            "day_open": self._num(row.get("o")),
            "day_high": self._num(row.get("h")),
            "day_low": self._num(row.get("l")),
            "day_close": close,
            "bid": None,
            "ask": None,
            "updated": row.get("t"),
            "source": "massive.previous_bar",
        }

    def quote(self, ticker: str) -> Optional[dict]:
        """Return a normalized quote-like market snapshot.

        Free Massive keys are expected to have end-of-day aggregates. Paid keys
        can opt into snapshot-first behavior with ARTHA_MASSIVE_SNAPSHOT_ENABLED.
        """
        symbol = str(ticker or "").strip().upper()
        if not symbol:
            return None
        if not Config.MASSIVE_SNAPSHOT_ENABLED:
            return self.previous_bar(symbol)

        payload = _expect_dict(
            self._get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"),
            "massive",
            f"quote:{symbol}",
        )
        if not payload:
            return self.previous_bar(symbol)
        snap = payload.get("ticker")
        if not isinstance(snap, dict):
            return None

        day = snap.get("day") if isinstance(snap.get("day"), dict) else {}
        prev = snap.get("prevDay") if isinstance(snap.get("prevDay"), dict) else {}
        last_trade = snap.get("lastTrade") if isinstance(snap.get("lastTrade"), dict) else {}
        last_quote = snap.get("lastQuote") if isinstance(snap.get("lastQuote"), dict) else {}

        price = (
            self._num(last_trade.get("p"))
            or self._num(day.get("c"))
            or self._num(prev.get("c"))
        )
        bid = self._num(last_quote.get("p"))
        ask = self._num(last_quote.get("P"))
        previous_close = self._num(prev.get("c"))
        volume = self._num(day.get("v")) or self._num(prev.get("v"))

        return {
            "symbol": snap.get("ticker") or symbol,
            "price": round(price, 4) if price is not None else None,
            "previous_close": round(previous_close, 4) if previous_close is not None else None,
            "change": self._num(snap.get("todaysChange")),
            "changesPercentage": self._num(snap.get("todaysChangePerc")),
            "volume": int(volume) if volume is not None else None,
            "day_open": self._num(day.get("o")),
            "day_high": self._num(day.get("h")),
            "day_low": self._num(day.get("l")),
            "day_close": self._num(day.get("c")),
            "bid": bid,
            "ask": ask,
            "updated": snap.get("updated"),
            "source": "massive.snapshot",
        }

    def history(self, ticker: str, period: str = "1y") -> Optional[list[dict]]:
        """Return normalized adjusted daily OHLCV bars for one ticker."""
        symbol = str(ticker or "").strip().upper()
        if not symbol:
            return None
        days = self._period_to_days(period)
        to_date = datetime.now(timezone.utc).date()
        from_date = to_date - timedelta(days=days)
        payload = _expect_dict(
            self._get(
                f"/v2/aggs/ticker/{symbol}/range/1/day/{from_date.isoformat()}/{to_date.isoformat()}",
                {"adjusted": "true", "sort": "asc", "limit": "50000"},
            ),
            "massive",
            f"history:{symbol}",
        )
        if not payload:
            return None
        rows = payload.get("results")
        if not isinstance(rows, list) or not rows:
            return None

        records: list[dict] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            close = self._num(row.get("c"))
            if close is None:
                continue
            records.append({
                "date": self._timestamp_ms_to_date(row.get("t")) or "",
                "open": round(float(row.get("o") or close), 4),
                "high": round(float(row.get("h") or close), 4),
                "low": round(float(row.get("l") or close), 4),
                "close": round(close, 4),
                "volume": int(float(row.get("v") or 0)),
                "source": "massive.aggregates",
            })
        return records or None


# ---------------------------------------------------------------------------
# Finnhub
# ---------------------------------------------------------------------------

class FinnhubCollector:
    """Finnhub API wrapper for sentiment, analyst recs, and news."""

    _CACHE_TTL_SECONDS: int = 1800  # 30-minute response cache to reduce API calls

    def __init__(self):
        self.base = Config.FINNHUB_BASE_URL
        self.key = Config.FINNHUB_API_KEY
        self._cache: dict = {}  # cache_key → (result, timestamp)

    def _get(self, endpoint: str, extra_params: dict | None = None) -> Optional[Any]:
        params = {"token": self.key}
        if extra_params:
            params.update(extra_params)

        # Build cache key from endpoint + non-auth params
        cache_params = {k: v for k, v in params.items() if k != "token"}
        cache_key = f"{endpoint}|{sorted(cache_params.items())}"

        # Return cached result if within TTL
        cached = self._cache.get(cache_key)
        if cached is not None:
            result, ts = cached
            if time.monotonic() - ts < self._CACHE_TTL_SECONDS:
                logger.debug(f"[finnhub] Cache hit: {endpoint}")
                return result

        # Rate limit then fetch
        _limiters["finnhub"].wait()
        url = f"{self.base}/{endpoint}"
        try:
            resp = _http_session.get(url, params=params, timeout=15)
            if resp.status_code == 403:
                # 403 is expected behavior on free tier when rate-limited
                logger.debug(f"[finnhub] 403 rate limited: {_sanitize_url(url)}")
                return None
            resp.raise_for_status()
            payload = resp.json()
            error_msg = _extract_api_error(payload)
            if error_msg:
                logger.warning(f"[finnhub] API error: {error_msg}")
                return None
            # Cache successful response
            self._cache[cache_key] = (payload, time.monotonic())
            return payload
        except requests.exceptions.Timeout:
            logger.warning(f"[finnhub] Timeout: {_sanitize_url(url)}")
            return None
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else "unknown"
            logger.warning(f"[finnhub] HTTP {status_code}: {_sanitize_url(url)}")
            return None
        except ValueError:
            logger.warning(f"[finnhub] Non-JSON response: {_sanitize_url(url)}")
            return None
        except Exception as e:
            logger.error(f"[finnhub] Unexpected error: {_sanitize_url(e)}")
            return None

    def news_sentiment(self, ticker: str) -> Optional[dict]:
        """Aggregated news sentiment for a ticker."""
        return _expect_dict(self._get("news-sentiment", {"symbol": ticker}), "finnhub", f"news_sentiment:{ticker}")

    def analyst_recommendations(self, ticker: str) -> Optional[list]:
        """Analyst buy/sell/hold recommendations."""
        return _expect_list(self._get("stock/recommendation", {"symbol": ticker}), "finnhub", f"analyst_recommendations:{ticker}")

    def earnings_surprises(self, ticker: str) -> Optional[list]:
        """Historical earnings surprises (beat/miss)."""
        return _expect_list(self._get("stock/earnings", {"symbol": ticker}), "finnhub", f"earnings_surprises:{ticker}")

    def company_news(self, ticker: str, days_back: int = 7) -> Optional[list]:
        """Company-specific news articles."""
        now_utc = datetime.now(timezone.utc)
        to_date = now_utc.strftime("%Y-%m-%d")
        from_date = (now_utc - timedelta(days=days_back)).strftime("%Y-%m-%d")
        return _expect_list(
            self._get("company-news", {"symbol": ticker, "from": from_date, "to": to_date}),
            "finnhub",
            f"company_news:{ticker}",
        )

    def insider_transactions(self, ticker: str) -> Optional[dict]:
        """Insider transaction data."""
        return _expect_dict(self._get("stock/insider-transactions", {"symbol": ticker}), "finnhub", f"insider_transactions:{ticker}")

    def market_news(self, category: str = "general") -> Optional[list]:
        """General market news."""
        return _expect_list(self._get("news", {"category": category}), "finnhub", f"market_news:{category}")


# ---------------------------------------------------------------------------
# Benzinga
# ---------------------------------------------------------------------------

class BenzingaCollector:
    """Benzinga Newsfeed API wrapper.

    Disabled by default until a Benzinga API key/licence is configured. It uses
    header authentication so keys do not appear in URLs or logs.
    """

    def __init__(self):
        self.base = Config.BENZINGA_BASE_URL.rstrip("/")
        self.key = Config.BENZINGA_API_KEY
        self.enabled = bool(self.key) and bool(Config.BENZINGA_NEWS_ENABLED)

    def _get(self, endpoint: str, extra_params: dict | None = None) -> Optional[Any]:
        if not self.enabled:
            return None
        endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        params = dict(extra_params or {})
        headers = {
            "Authorization": f"token {self.key}",
            "Accept": "application/json",
        }
        _limiters["benzinga"].wait()
        url = f"{self.base}{endpoint}"
        try:
            resp = _http_session.get(url, params=params, headers=headers, timeout=15)
            if resp.status_code == 403:
                logger.warning("[benzinga] Forbidden for %s; plan may not include this endpoint", endpoint)
                return None
            if resp.status_code == 401:
                logger.warning("[benzinga] Unauthorized for %s; check API key", endpoint)
                return None
            if resp.status_code == 429:
                logger.warning("[benzinga] Rate limited for %s", endpoint)
                return None
            resp.raise_for_status()
            payload = resp.json()
            error_msg = _extract_api_error(payload)
            if error_msg:
                logger.warning("[benzinga] API error payload: %s", error_msg)
                return None
            return payload
        except requests.exceptions.Timeout:
            logger.warning("[benzinga] Timeout fetching %s", endpoint)
            return None
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            logger.warning("[benzinga] HTTP %s from %s", status_code, endpoint)
            return None
        except ValueError:
            logger.warning("[benzinga] Non-JSON response from %s", endpoint)
            return None
        except Exception as exc:
            logger.error("[benzinga] Unexpected error from %s: %s", endpoint, exc)
            return None

    @staticmethod
    def _parse_news_datetime(value: Any) -> datetime | None:
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc)
            except Exception:
                return None
        if not value:
            return None
        text = str(value).strip()
        try:
            parsed = parsedate_to_datetime(text)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            pass
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            return None

    def company_news(self, ticker: str, limit: int = 20, lookback_hours: int | None = None) -> Optional[list]:
        """Return Benzinga news for one ticker, newest first, normalized enough for Sentinel."""
        symbol = str(ticker or "").upper().strip()
        if not symbol:
            return None
        lookback = max(1, int(lookback_hours or Config.BENZINGA_NEWS_LOOKBACK_HOURS))
        published_since = int((datetime.now(timezone.utc) - timedelta(hours=lookback)).timestamp())
        payload = self._get(
            "/news",
            {
                "tickers": symbol,
                "pageSize": str(min(max(int(limit), 1), 100)),
                "displayOutput": "abstract",
                "publishedSince": str(published_since),
                "sort": "created:desc",
            },
        )
        rows = _expect_list(payload, "benzinga", f"company_news:{symbol}")
        if not rows:
            return None
        normalized: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or "").strip()
            if not title:
                continue
            dt = self._parse_news_datetime(row.get("created") or row.get("updated"))
            normalized.append(
                {
                    "title": title,
                    "url": row.get("url") or "",
                    "source": "Benzinga",
                    "publishedDate": (dt or datetime.now(timezone.utc)).isoformat(),
                    "text": str(row.get("teaser") or row.get("body") or "")[:1000],
                    "importance_rank": row.get("importance_rank"),
                    "author": row.get("author"),
                    "raw": row,
                }
            )
        return normalized or None


# ---------------------------------------------------------------------------
# CoinGecko
# ---------------------------------------------------------------------------

class CoinGeckoCollector:
    """CoinGecko Demo API for crypto data."""

    def __init__(self):
        self.base = Config.COINGECKO_BASE_URL
        self.key = Config.COINGECKO_API_KEY

    def _get(self, endpoint: str, extra_params: dict | None = None) -> Optional[Any]:
        params = {}
        if extra_params:
            params.update(extra_params)
        headers = {"x-cg-demo-api-key": self.key} if self.key else {}
        _limiters["coingecko"].wait()
        try:
            resp = _http_session.get(f"{self.base}/{endpoint}", params=params,
                                     headers=headers, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            error_msg = _extract_api_error(payload)
            if error_msg:
                logger.warning(f"[coingecko] API returned error payload: {error_msg}")
                return None
            return payload
        except ValueError:
            logger.warning("[coingecko] Non-JSON response")
            return None
        except Exception as e:
            logger.warning(f"[coingecko] Error: {e}")
            return None

    def price(self, ids: str = "bitcoin,ethereum", vs: str = "usd") -> Optional[dict]:
        """Current price for crypto assets."""
        return _expect_dict(self._get("simple/price", {
            "ids": ids,
            "vs_currencies": vs,
            "include_24hr_change": "true",
            "include_market_cap": "true",
        }), "coingecko", "price")

    def trending(self) -> Optional[dict]:
        """Trending coins in the last 24h."""
        return _expect_dict(self._get("search/trending"), "coingecko", "trending")

    def market_chart(self, coin_id: str = "bitcoin", days: int = 30, vs: str = "usd") -> Optional[dict]:
        """Historical price data."""
        return _expect_dict(self._get(f"coins/{coin_id}/market_chart", {
            "vs_currency": vs,
            "days": str(days),
        }), "coingecko", f"market_chart:{coin_id}")


# ---------------------------------------------------------------------------
# FRED (Federal Reserve Economic Data)
# ---------------------------------------------------------------------------

class FREDCollector:
    """FRED API for macro economic indicators."""

    def __init__(self):
        self.base = Config.FRED_BASE_URL
        self.key = Config.FRED_API_KEY

    def _get(self, endpoint: str, extra_params: dict | None = None) -> Optional[Any]:
        params = {"api_key": self.key, "file_type": "json"}
        if extra_params:
            params.update(extra_params)
        url = f"{self.base}/{endpoint}"
        key = _cache_key(url, params, "fred")
        cached = _response_cache.get(key)
        if cached and time.time() - cached[0] <= Config.FRED_CACHE_TTL_SECONDS:
            return cached[1]
        if cached:
            _response_cache.pop(key, None)

        attempts = 2
        for attempt in range(attempts + 1):
            _limiters["fred"].wait()
            try:
                resp = _http_session.get(url, params=params, timeout=15)
                if resp.status_code in (429, 500):
                    wait = 3.0 if resp.status_code == 500 else _retry_wait_seconds(resp, attempt)
                    logger.warning(
                        "[fred] HTTP %s from %s (attempt %d/%d); retrying in %.1fs",
                        resp.status_code,
                        _sanitize_url(resp.url or url),
                        attempt + 1,
                        attempts + 1,
                        wait,
                    )
                    if attempt < attempts:
                        time.sleep(wait)
                        continue
                    return None
                resp.raise_for_status()
                payload = resp.json()
                error_msg = _extract_api_error(payload)
                if error_msg:
                    logger.warning("[fred] API returned error payload: %s", error_msg)
                    return None
                _response_cache[key] = (time.time(), payload)
                return payload
            except requests.exceptions.Timeout:
                logger.warning("[fred] Timeout fetching %s", _sanitize_url(url))
                if attempt < attempts:
                    time.sleep(min(10.0, 1.5 * (2 ** attempt)))
                    continue
                return None
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else "unknown"
                logger.warning("[fred] HTTP %s from %s", status_code, _sanitize_url(url))
                return None
            except ValueError:
                logger.warning("[fred] Non-JSON response from %s", _sanitize_url(url))
                return None
            except Exception as e:
                logger.error("[fred] Unexpected error: %s", _sanitize_url(e))
                return None
        return None

    def series(self, series_id: str, limit: int = 10) -> Optional[dict]:
        """Get data for a FRED series."""
        return _expect_dict(self._get("series/observations", {
            "series_id": series_id,
            "sort_order": "desc",
            "limit": str(limit),
        }), "fred", f"series:{series_id}")

    def fed_funds_rate(self) -> Optional[dict]:
        """Current federal funds rate."""
        return self.series("FEDFUNDS", limit=3)

    def cpi(self) -> Optional[dict]:
        """Consumer Price Index (inflation)."""
        return self.series("CPIAUCSL", limit=6)

    def unemployment(self) -> Optional[dict]:
        """Unemployment rate."""
        return self.series("UNRATE", limit=6)

    def treasury_10y(self) -> Optional[dict]:
        """10-Year Treasury yield."""
        return self.series("DGS10", limit=5)

    def treasury_2y(self) -> Optional[dict]:
        """2-Year Treasury yield."""
        return self.series("DGS2", limit=5)

    def yield_curve_spread(self) -> Optional[dict]:
        """10Y-2Y Treasury yield spread (FRED T10Y2Y).

        Negative = inverted curve (recession signal).
        """
        return self.series("T10Y2Y", limit=5)

    def hy_credit_spread(self) -> Optional[dict]:
        """ICE BofA US High Yield Option-Adjusted Spread (BAMLH0A0HYM2).

        >600 bps = significant stress; >800 bps = banking/credit crisis.
        """
        return self.series("BAMLH0A0HYM2", limit=5)

    def ig_credit_spread(self) -> Optional[dict]:
        """ICE BofA US Corporate Bond Option-Adjusted Spread (BAMLC0A0CM).

        Investment-grade spread — signal for overall credit market stress.
        """
        return self.series("BAMLC0A0CM", limit=5)

    def oil_price(self) -> Optional[dict]:
        """WTI crude oil price (DCOILWTICO)."""
        return self.series("DCOILWTICO", limit=10)

    def initial_jobless_claims(self) -> Optional[dict]:
        """Initial jobless claims (ICSA) — real-time employment shock indicator."""
        return self.series("ICSA", limit=8)

    def gdp(self) -> Optional[dict]:
        """Real GDP."""
        return self.series("GDPC1", limit=4)


# ---------------------------------------------------------------------------
# Sentiment helpers
# ---------------------------------------------------------------------------

def get_fear_greed_index() -> Optional[dict]:
    """Crypto Fear & Greed Index from Alternative.me."""
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1&format=json", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data and "data" in data and len(data["data"]) > 0:
            entry = data["data"][0]
            value_raw = entry.get("value")
            try:
                value = int(value_raw) if value_raw is not None else 0
            except (TypeError, ValueError):
                value = 0
            return {
                "value": value,
                "label": entry.get("value_classification", "Unknown"),
                "timestamp": entry.get("timestamp", ""),
                "asset_class": "crypto",
                "source": "alternative.me",
            }
        return None
    except Exception as e:
        logger.warning(f"[fear_greed] Error: {e}")
        return None


def get_crypto_fear_greed_index() -> Optional[dict]:
    """Explicit alias for the Alternative.me crypto sentiment index."""
    return get_fear_greed_index()


def get_equity_sentiment_index(market_snapshot: Optional[dict] = None) -> dict:
    """Deterministic equity-market sentiment estimate for stock decisions.

    This is intentionally labeled as an equity sentiment estimate, not a clone
    of CNN's proprietary Fear & Greed index. It prevents crypto sentiment from
    leaking into stock deployment and stock report headers.
    """
    market_snapshot = market_snapshot or {}

    def _num(value: object, default: Optional[float] = None) -> Optional[float]:
        try:
            if value is None:
                return default
            return float(str(value).replace("%", "").replace(",", ""))
        except (TypeError, ValueError):
            return default

    def _quote_change(key: str) -> float:
        quote = market_snapshot.get(key) or {}
        if not isinstance(quote, dict):
            return 0.0
        return _num(
            quote.get("changesPercentage")
            or quote.get("changePercentage")
            or quote.get("regularMarketChangePercent"),
            0.0,
        ) or 0.0

    spy_change = _quote_change("sp500")
    qqq_change = _quote_change("nasdaq")
    vix_payload = market_snapshot.get("vix") or {}
    if isinstance(vix_payload, dict):
        vix = _num(vix_payload.get("value") or vix_payload.get("price"), None)
    else:
        vix = _num(vix_payload, None)

    value = 50.0
    value += max(-15.0, min(15.0, spy_change * 6.0))
    value += max(-10.0, min(10.0, qqq_change * 4.0))
    if vix is not None:
        if vix >= 35:
            value -= 25
        elif vix >= 28:
            value -= 16
        elif vix >= 22:
            value -= 8
        elif vix <= 13:
            value += 10
        elif vix <= 16:
            value += 5

    value_int = int(round(max(0.0, min(100.0, value))))
    if value_int < 20:
        label = "Extreme Fear"
    elif value_int < 40:
        label = "Fear"
    elif value_int <= 60:
        label = "Neutral"
    elif value_int <= 80:
        label = "Greed"
    else:
        label = "Extreme Greed"

    return {
        "value": value_int,
        "label": label,
        "asset_class": "equity",
        "source": "artha_equity_sentiment",
        "inputs": {
            "spy_1d_pct": spy_change,
            "qqq_1d_pct": qqq_change,
            "vix": vix,
        },
    }


# ---------------------------------------------------------------------------
# yfinance (Yahoo Finance — unlimited, no key)
# ---------------------------------------------------------------------------

class YFinanceCollector:
    """Yahoo Finance via yfinance library — unlimited, no API key needed."""

    @staticmethod
    def cleanup_caches():
        """Close yfinance internal SQLite caches to prevent FD leak.
        
        yfinance uses peewee SQLite for tz/cookie/isin caches. Each call
        opens new connections without closing old ones, causing FD
        exhaustion in long-running daemons (Errno 24 after ~256 FDs).
        Call this after each monitor tick cycle.
        """
        import gc
        import sqlite3

        def _looks_like_yfinance_connection(conn: sqlite3.Connection) -> bool:
            try:
                rows = conn.execute("PRAGMA database_list").fetchall()
            except Exception:
                return False
            for row in rows:
                try:
                    path = str(row[-1] or "").lower()
                    basename = Path(path).name
                except Exception:
                    path = ""
                    basename = ""
                if (
                    "py-yfinance" in path
                    or basename in {"tkr-tz.db", "cookies.db", "isin-ticker.db"}
                    or basename.startswith("cookie")
                    or basename.startswith("isin")
                ):
                    return True
            return False

        try:
            # 1. Close via peewee proxies (handles the latest connections)
            for proxy_name in ("tz_db_proxy", "Cookie_db_proxy", "isin_db_proxy"):
                proxy = getattr(yf.cache, proxy_name, None)
                if proxy and hasattr(proxy, "obj") and proxy.obj:
                    db = proxy.obj
                    if hasattr(db, "close") and not db.is_closed():
                        db.close()
            # 2. Force GC first so weak refs to connections are cleared
            gc.collect()
            # 3. Close leaked yfinance sqlite3 connections only. Do not close
            # Artha's own SQLite handles; the monitor may have live journal DBs.
            closed_count = 0
            for obj in gc.get_objects():
                if isinstance(obj, sqlite3.Connection):
                    try:
                        if _looks_like_yfinance_connection(obj):
                            obj.close()
                            closed_count += 1
                    except Exception:
                        pass
            if closed_count > 0:
                logger.debug(f"[yfinance cleanup] Closed {closed_count} leaked SQLite connections")
            # 4. Force another GC pass to release closed objects
            gc.collect()
        except Exception:
            pass  # Best-effort cleanup — don't crash the monitor

    def quote(self, ticker: str) -> Optional[dict]:
        """Get current quote data."""
        try:
            t = yf.Ticker(ticker)
            info = t.info
            return {
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "previous_close": info.get("previousClose") or info.get("regularMarketPreviousClose"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "dividend_yield": info.get("dividendYield"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "volume": info.get("regularMarketVolume") or info.get("volume"),
                "regularMarketVolume": info.get("regularMarketVolume"),
                "avg_volume": info.get("averageVolume"),
                "averageVolume": info.get("averageVolume"),
                "bid": info.get("bid"),
                "ask": info.get("ask"),
                "beta": info.get("beta"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "name": info.get("shortName"),
            }
        except Exception as e:
            logger.warning(f"[yfinance] Error fetching {ticker}: {e}")
            return None

    def vix(self) -> Optional[dict]:
        """VIX volatility index — yfinance .history() (avoids fragile .info call).
        
        Note: FRED VIXCLS is used separately in crisis fingerprinting for 
        government-backed reliability. This method provides real-time VIX.
        """
        try:
            t = yf.Ticker("^VIX")
            hist = t.history(period="5d")
            if not hist.empty:
                price = round(float(hist["Close"].iloc[-1]), 2)
                return {"value": price, "ticker": "^VIX", "date": hist.index[-1].strftime("%Y-%m-%d")}
            logger.warning("[yfinance] VIX history empty")
            return None
        except Exception as e:
            logger.warning(f"[yfinance] VIX fetch error: {e}")
            return None

    def sector_etf_snapshot(self) -> dict[str, Optional[float]]:
        """Fetch current prices for all 11 sector ETFs + SPY for dispersion analysis.
        
        Uses yf.download() batch call (1 request) instead of 12 individual .info calls.
        Falls back to individual calls if batch fails.
        """
        tickers = ["XLF", "XLE", "XLK", "XLV", "XLP", "XLI", "XLU", "XLY", "XLC", "XLRE", "XLB", "SPY"]
        result: dict[str, Optional[float]] = {}
        
        try:
            # Single batch download — 12x fewer API calls
            df = yf.download(
                " ".join(tickers),
                period="2d",
                progress=False,
                threads=Config.YFINANCE_THREADS,
            )
            if not df.empty and "Close" in df.columns:
                latest = df["Close"].iloc[-1]
                for ticker in tickers:
                    try:
                        val = latest[ticker]
                        result[ticker] = round(float(val), 2) if val == val else None  # NaN check
                    except (KeyError, TypeError):
                        result[ticker] = None
                logger.info(f"[yfinance] Sector ETFs fetched via batch: {sum(1 for v in result.values() if v is not None)}/{len(tickers)} OK")
                return result
        except Exception as e:
            logger.warning(f"[yfinance] Batch sector ETF download failed, falling back to individual: {e}")
        finally:
            YFinanceCollector.cleanup_caches()
        
        # Fallback: individual history calls (NOT .info — less rate-limited)
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="2d")
                if not hist.empty:
                    result[ticker] = round(float(hist["Close"].iloc[-1]), 2)
                else:
                    result[ticker] = None
            except Exception as e:
                logger.warning(f"[yfinance] Sector ETF {ticker} fallback error: {e}")
                result[ticker] = None
            finally:
                YFinanceCollector.cleanup_caches()
        return result

    def history(self, ticker: str, period: str = "3mo") -> Optional[list]:
        """Historical price data with retry on failure."""
        def _fetch():
            t = yf.Ticker(ticker)
            df = t.history(period=period)
            if df.empty:
                return None
            records = []
            for date, row in df.iterrows():
                records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "open": round(row["Open"], 2),
                    "high": round(row["High"], 2),
                    "low": round(row["Low"], 2),
                    "close": round(row["Close"], 2),
                    "volume": int(row["Volume"]),
                })
            return records
        
        result, retried = _retry_on_failure(_fetch, retries=2, backoff=2.0, source=f"yfinance-history:{ticker}")
        if retried and result is not None:
            logger.info(f"[yfinance] History for {ticker} succeeded after retry")
        return result


# ---------------------------------------------------------------------------
# Aggregate Collector — Single entry point
# ---------------------------------------------------------------------------

class DataCollector:
    """Unified data collector that orchestrates all API sources.
    
    Usage:
        collector = DataCollector()
        data = collector.collect_stock("AAPL")
        data = collector.collect_crypto("bitcoin")
        macro = collector.collect_macro()
        market = collector.collect_market_overview()
    """

    def __init__(self):
        self.fmp = FMPCollector()
        self.finnhub = FinnhubCollector()
        self.benzinga = BenzingaCollector()
        self.coingecko = CoinGeckoCollector()
        self.fred = FREDCollector()
        self.massive = MassiveCollector()
        self.yf = YFinanceCollector()

    @staticmethod
    def _build_history_provider_checks(
        *,
        selected_source: str,
        fmp_history: Optional[list],
        yf_history: Optional[list],
        massive_history: Optional[list],
    ) -> dict[str, Any]:
        def summary(rows: Optional[list]) -> dict[str, Any]:
            if not isinstance(rows, list) or not rows:
                return {"bars": 0, "latest_date": None, "latest_close": None}
            latest = rows[-1] if isinstance(rows[-1], dict) else {}
            close = None
            try:
                close = float(latest.get("close")) if latest.get("close") is not None else None
            except Exception:
                close = None
            return {
                "bars": len(rows),
                "latest_date": latest.get("date"),
                "latest_close": close,
            }

        providers = {
            "fmp": summary(fmp_history),
            "yfinance": summary(yf_history),
            "massive": summary(massive_history),
        }
        selected_close = (providers.get(selected_source) or {}).get("latest_close")
        conflicts: list[str] = []
        if selected_close and selected_close > 0:
            for provider, payload in providers.items():
                close = payload.get("latest_close")
                if provider == selected_source or not close:
                    continue
                diff = abs(float(close) - float(selected_close)) / float(selected_close)
                if diff > 0.02:
                    conflicts.append(
                        f"{provider} latest close differs from selected {selected_source} by {diff:.1%}"
                    )
        return {
            "selected_source": selected_source,
            "providers": providers,
            "conflicts": conflicts,
        }

    def collect_stock(self, ticker: str) -> dict:
        """Collect ALL available data for a stock ticker.
        
        Returns a comprehensive dict with data from all sources,
        organized by category (fundamentals, technicals, sentiment, etc.)
        """
        logger.info(f"Collecting data for {ticker}...")
        
        _now_iso = _utcnow_iso()
        data: dict[str, Any] = {
            "ticker": ticker,
            "collected_at": _now_iso,
            # PIT metadata
            "as_of_datetime_utc": _now_iso,
            "source": "DataCollector.collect_stock",
            "ingested_at_utc": _now_iso,
        }

        # --- Fundamentals (FMP + yfinance) ---
        data["quote"] = self.fmp.quote(ticker)
        data["profile"] = self.fmp.company_profile(ticker)
        data["income_statement"] = self.fmp.income_statement(ticker, period="quarter", limit=4)
        data["balance_sheet"] = self.fmp.balance_sheet(ticker, period="quarter", limit=4)
        data["cash_flow"] = self.fmp.cash_flow(ticker, period="quarter", limit=4)
        data["ratios"] = self.fmp.ratios(ticker, limit=4)
        data["ratios_ttm"] = self.fmp.ratios_ttm(ticker)
        data["key_metrics"] = self.fmp.key_metrics(ticker, limit=4)
        data["key_metrics_ttm"] = self.fmp.key_metrics_ttm(ticker)
        data["dcf"] = self.fmp.dcf(ticker)
        data["price_target_consensus"] = self.fmp.price_target_consensus(ticker)
        data["yf_quote"] = self.yf.quote(ticker)
        data["massive_quote"] = self.massive.quote(ticker)

        # --- Technical Indicators (computed locally from price data) ---
        # Extended to 1yr for SMA 200 computation
        history_mode = str(Config.PRICE_HISTORY_MODE or "fmp_primary").lower()
        massive_history_mode = str(Config.MASSIVE_HISTORY_MODE or "fallback").lower()

        fmp_history = None if history_mode in {"yfinance_only", "off"} else self.fmp.history(ticker, "1y")
        yf_history = None if history_mode in {"fmp_only", "off"} else self.yf.history(ticker, "1y")
        massive_history = None
        if massive_history_mode == "primary":
            massive_history = self.massive.history(ticker, "1y")

        def usable_history(rows: Any) -> bool:
            return isinstance(rows, list) and len(rows) >= 20

        selected_history = None
        selected_source = "unavailable"
        if history_mode == "yfinance_primary":
            if usable_history(yf_history):
                selected_history = yf_history
                selected_source = "yfinance"
            elif usable_history(fmp_history):
                selected_history = fmp_history
                selected_source = "fmp"
        elif history_mode == "massive_primary":
            if usable_history(massive_history):
                selected_history = massive_history
                selected_source = "massive"
            elif usable_history(fmp_history):
                selected_history = fmp_history
                selected_source = "fmp"
            elif usable_history(yf_history):
                selected_history = yf_history
                selected_source = "yfinance"
        else:
            if usable_history(fmp_history):
                selected_history = fmp_history
                selected_source = "fmp"
            elif usable_history(yf_history):
                selected_history = yf_history
                selected_source = "yfinance"

        if selected_history is None and massive_history_mode != "off":
            massive_history = massive_history or self.massive.history(ticker, "1y")
            if usable_history(massive_history):
                selected_history = massive_history
                selected_source = "massive"

        data["fmp_price_history"] = fmp_history
        data["yf_price_history"] = yf_history
        if massive_history is not None:
            data["massive_price_history"] = massive_history
        data["price_history"] = selected_history
        data["price_history_source"] = selected_source
        data["history_provider_checks"] = self._build_history_provider_checks(
            selected_source=selected_source,
            fmp_history=fmp_history,
            yf_history=yf_history,
            massive_history=massive_history,
        )

        # Compute RSI, MACD, SMA, Bollinger Bands locally — no API limits
        data["technicals"] = compute_technicals(data["price_history"])
        
        # Legacy keys for backward compatibility (populated from local computation)
        technicals = data["technicals"]
        if technicals:
            data["av_rsi"] = {"rsi": technicals.get("rsi"), "interpretation": technicals.get("rsi_interpretation"), "source": "local"}
            data["av_macd"] = {"macd": technicals.get("macd"), "signal": technicals.get("macd_signal"), "histogram": technicals.get("macd_histogram"), "interpretation": technicals.get("macd_interpretation"), "source": "local"}
        else:
            data["av_rsi"] = None
            data["av_macd"] = None

        # --- Sentiment & News ---
        data["news"] = self.fmp.stock_news(ticker, limit=10)
        data["benzinga_news"] = self.benzinga.company_news(ticker, limit=10)
        data["finnhub_sentiment"] = self.finnhub.news_sentiment(ticker)
        data["finnhub_news"] = self.finnhub.company_news(ticker, days_back=7)

        # --- Analyst & Insider ---
        data["analyst_recs"] = self.finnhub.analyst_recommendations(ticker)
        data["earnings_surprises"] = self.finnhub.earnings_surprises(ticker)
        data["insider_finnhub"] = self.finnhub.insider_transactions(ticker)

        # --- Premium/official signal layer for council decisions ---
        try:
            from .analyst_signals import (
                get_short_interest,
                get_recommendation_trends,
                get_analyst_estimates,
            )
            data["short_interest"] = get_short_interest(ticker)
            data["recommendation_trends"] = get_recommendation_trends(ticker)
            data["analyst_estimates"] = get_analyst_estimates(ticker)
        except Exception as e:
            logger.warning(f"[{ticker}] Analyst signal enrichment error: {e}")
            data["short_interest"] = None
            data["recommendation_trends"] = None
            data["analyst_estimates"] = None

        try:
            from .sec_edgar import get_sec_company_context
            data["sec"] = get_sec_company_context(ticker, profile=data.get("profile"))
        except Exception as e:
            logger.warning(f"[{ticker}] SEC EDGAR enrichment error: {e}")
            data["sec"] = {"source": "sec", "status": "unavailable", "ticker": ticker, "error": str(e)}

        # --- Earnings Calendar Context ---
        try:
            from .earnings_calendar import get_earnings_context
            ec = get_earnings_context(ticker, finnhub_data=data.get("earnings_surprises"))
            data["earnings_context"] = {
                "earnings_date": ec.earnings_date,
                "days_to_earnings": ec.days_to_earnings,
                "earnings_time": ec.earnings_time,
                "earnings_risk_flag": ec.earnings_risk_flag,
                "earnings_defer_flag": ec.earnings_defer_flag,
                "recent_surprises": ec.recent_surprises,
            }
        except Exception as e:
            logger.warning(f"[{ticker}] Earnings context error: {e}")
            data["earnings_context"] = None

        # --- Data Availability Flags ---
        # Flag missing data so analysts know what they're working with
        missing = []
        if not data.get("price_history"):
            missing.append("price_history")
        if not data.get("technicals"):
            missing.append("technicals")
        if not data.get("quote"):
            missing.append("quote")
        if not data.get("income_statement"):
            missing.append("income_statement")
        if not data.get("balance_sheet"):
            missing.append("balance_sheet")
        if not data.get("cash_flow"):
            missing.append("cash_flow")
        if not data.get("news") and not data.get("finnhub_news") and not data.get("benzinga_news"):
            missing.append("news")
        if data.get("earnings_context") is None:
            missing.append("earnings_context")
        for field in ("short_interest", "recommendation_trends", "analyst_estimates", "sec"):
            value = data.get(field)
            if (
                not value
                or (
                    isinstance(value, dict)
                    and (
                        str(value.get("status", "")).lower() == "unavailable"
                        or str(value.get("source", "")).lower() == "unavailable"
                    )
                )
            ):
                missing.append(field)
        
        data["data_quality"] = {
            "missing_fields": missing,
            "completeness": round(max(0, (1 - len(missing) / 12)) * 100, 1),
            "technicals_source": "local" if data.get("technicals") else "unavailable",
        }
        if missing:
            logger.warning(f"[{ticker}] Missing data: {', '.join(missing)} — completeness: {data['data_quality']['completeness']}%")

        # Cleanup yfinance FDs after each stock collection
        YFinanceCollector.cleanup_caches()
        return data

    def collect_crypto(self, coin_id: str = "bitcoin", fmp_symbol: str = "BTCUSD") -> dict:
        """Collect crypto data from CoinGecko + FMP."""
        logger.info(f"Collecting crypto data for {coin_id}...")
        
        data: dict[str, Any] = {"coin_id": coin_id, "collected_at": _utcnow_iso()}

        data["price"] = self.coingecko.price(coin_id)
        data["market_chart_30d"] = self.coingecko.market_chart(coin_id, days=30)
        data["fmp_quote"] = self.fmp.crypto_quote(fmp_symbol)
        data["fear_greed"] = get_fear_greed_index()
        data["trending"] = self.coingecko.trending()

        return data

    def collect_macro(self) -> dict:
        """Collect macro economic indicators."""
        logger.info("Collecting macro data...")
        _now_iso = _utcnow_iso()
        return {
            "collected_at": _now_iso,
            "as_of_datetime_utc": _now_iso,
            "source": "DataCollector.collect_macro",
            "ingested_at_utc": _now_iso,
            "fed_funds_rate": self.fred.fed_funds_rate(),
            "cpi": self.fred.cpi(),
            "unemployment": self.fred.unemployment(),
            "treasury_10y": self.fred.treasury_10y(),
            "gdp": self.fred.gdp(),
            "fear_greed_crypto": get_crypto_fear_greed_index(),
        }

    def collect_crisis_signals(self) -> dict:
        """Collect all signals needed for crisis fingerprinting.

        Returns a flat dict of signals used by the CrisisFingerprint engine.
        All values are Optional[float] — None means data was unavailable.
        """
        logger.info("Collecting crisis fingerprint signals...")

        def _latest_value(series_data: Optional[dict]) -> Optional[float]:
            """Extract the most recent non-missing observation from a FRED series dict."""
            if not isinstance(series_data, dict):
                return None
            obs = series_data.get("observations", [])
            if not isinstance(obs, list):
                return None
            for item in obs:
                val = item.get("value", ".")
                if val and val != ".":
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        continue
            return None

        # FRED macro signals
        yield_curve_raw = self.fred.yield_curve_spread()
        hy_spread_raw = self.fred.hy_credit_spread()
        ig_spread_raw = self.fred.ig_credit_spread()
        oil_raw = self.fred.oil_price()
        cpi_raw = self.fred.cpi()
        unemployment_raw = self.fred.unemployment()
        gdp_raw = self.fred.gdp()
        jobless_raw = self.fred.initial_jobless_claims()
        fed_funds_raw = self.fred.fed_funds_rate()
        treasury_10y_raw = self.fred.treasury_10y()
        treasury_2y_raw = self.fred.treasury_2y()

        # yfinance signals
        vix_data = self.yf.vix()
        sector_etfs = self.yf.sector_etf_snapshot()

        # SPY 52-week data for drawdown calculation
        spy_info = None
        try:
            spy_ticker = yf.Ticker("SPY")
            info = spy_ticker.info
            spy_info = {
                "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
            }
        except Exception as e:
            logger.warning(f"[yfinance] SPY info error: {e}")

        # DXY (US Dollar Index) via yfinance
        dxy_price = None
        try:
            dxy_ticker = yf.Ticker("DX-Y.NYB")
            dxy_info = dxy_ticker.info
            dxy_price = dxy_info.get("currentPrice") or dxy_info.get("regularMarketPrice") or dxy_info.get("previousClose")
        except Exception as e:
            logger.warning(f"[yfinance] DXY error: {e}")

        # Compute SPY drawdown from 52w high
        spy_drawdown = None
        if spy_info and spy_info.get("current_price") and spy_info.get("52w_high"):
            try:
                spy_drawdown = (spy_info["current_price"] - spy_info["52w_high"]) / spy_info["52w_high"]
            except (TypeError, ZeroDivisionError):
                pass

        # Compute VTI/VXUS current price for counterfactual baseline
        vti_price = None
        vxus_price = None
        try:
            vti_price = (self.yf.quote("VTI") or {}).get("price")
            vxus_price = (self.yf.quote("VXUS") or {}).get("price")
        except Exception as e:
            logger.warning(f"[yfinance] VTI/VXUS price error: {e}")

        _now_iso = _utcnow_iso()
        return {
            "collected_at": _now_iso,
            "as_of_datetime_utc": _now_iso,
            "source": "DataCollector.collect_crisis_signals",
            "ingested_at_utc": _now_iso,
            # FRED signals
            "yield_curve_spread": _latest_value(yield_curve_raw),      # T10Y2Y (bps, can be negative)
            "hy_oas": _latest_value(hy_spread_raw),                     # percentage points (FRED units, e.g., 8.0 = 800bps)
            "ig_oas": _latest_value(ig_spread_raw),                     # percentage points (FRED units, e.g., 2.0 = 200bps)
            "oil_price": _latest_value(oil_raw),                        # USD/barrel
            "cpi": _latest_value(cpi_raw),                              # Index value
            "unemployment_rate": _latest_value(unemployment_raw),       # %
            "gdp_growth": _latest_value(gdp_raw),                       # Chained 2017 USD billions
            "initial_jobless_claims": _latest_value(jobless_raw),       # raw count (e.g., 250000 = 250K)
            "fed_funds_rate": _latest_value(fed_funds_raw),             # %
            "treasury_10y": _latest_value(treasury_10y_raw),            # %
            "treasury_2y": _latest_value(treasury_2y_raw),              # %
            # yfinance signals
            "vix": vix_data.get("value") if vix_data else None,         # VIX level
            "spy_drawdown": spy_drawdown,                                # Fraction (e.g., -0.25)
            "spy_current": spy_info.get("current_price") if spy_info else None,
            "spy_52w_high": spy_info.get("52w_high") if spy_info else None,
            "dxy": float(dxy_price) if dxy_price else None,             # USD index
            # Sector ETF prices
            "sector_etfs": sector_etfs,
            # Baseline prices for counterfactuals
            "vti_price": float(vti_price) if vti_price else None,
            "vxus_price": float(vxus_price) if vxus_price else None,
            # Raw series for trend analysis
            "_raw": {
                "yield_curve": yield_curve_raw,
                "hy_spread": hy_spread_raw,
                "ig_spread": ig_spread_raw,
                "cpi": cpi_raw,
                "unemployment": unemployment_raw,
                "jobless_claims": jobless_raw,
            },
        }

    # Core watchlist for market movers detection
    MOVERS_WATCHLIST = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B",
        "JPM", "V", "UNH", "JNJ", "WMT", "PG", "MA", "HD", "XOM", "CVX",
        "ABBV", "KO", "PEP", "COST", "MRK", "LLY", "AVGO", "CRM", "NFLX",
        "AMD", "INTC", "DIS", "BA", "NKE", "PYPL", "SQ", "COIN", "SOFI",
        "VTI", "VXUS", "QQQ", "SPY",
    ]

    def collect_market_overview(self) -> dict:
        """High-level market snapshot: gainers, losers, S&P performance.
        
        Uses FMP quotes (which include changePercentage) to identify
        today's biggest movers from the core watchlist.
        """
        logger.info("Collecting market overview...")
        
        # Fetch quotes for watchlist to find movers
        movers = []
        for ticker in self.MOVERS_WATCHLIST:
            try:
                q = self.fmp.quote(ticker)
                if q and isinstance(q, dict):
                    change_pct = q.get("changePercentage") or q.get("changesPercentage") or 0
                    movers.append({
                        "symbol": ticker,
                        "name": q.get("name", ticker),
                        "price": q.get("price"),
                        "change_pct": round(float(change_pct), 2) if change_pct else 0,
                        "volume": q.get("volume", 0),
                        "market_cap": q.get("marketCap", 0),
                    })
            except Exception as e:
                logger.warning(f"[market_overview] Error fetching {ticker}: {e}")
        
        # Sort by absolute change to find biggest movers
        movers.sort(key=lambda x: abs(x.get("change_pct", 0)), reverse=True)
        
        gainers = [m for m in movers if m.get("change_pct", 0) > 0][:10]
        losers = [m for m in movers if m.get("change_pct", 0) < 0][:10]
        # Sort losers by most negative first
        losers.sort(key=lambda x: x.get("change_pct", 0))
        # Most active by volume
        most_active = sorted(movers, key=lambda x: x.get("volume", 0) or 0, reverse=True)[:10]
        
        logger.info(f"  Market movers: {len(gainers)} gainers, {len(losers)} losers from {len(movers)} watchlist stocks")
        
        snapshot = {
            "collected_at": _utcnow_iso(),
            "gainers": gainers,
            "losers": losers,
            "most_active": most_active,
            "sp500": self.fmp.quote("SPY"),
            "nasdaq": self.fmp.quote("QQQ"),
            "dow": self.fmp.quote("DIA"),
            "vix": self.yf.vix(),
            "btc": self.fmp.crypto_quote("BTCUSD"),
            "eth": self.fmp.crypto_quote("ETHUSD"),
            "general_news": self.finnhub.market_news(),
        }
        snapshot["fear_greed"] = get_equity_sentiment_index(snapshot)
        snapshot["fear_greed_crypto"] = get_crypto_fear_greed_index()
        return snapshot


# ---------------------------------------------------------------------------
# Local Technical Indicator Computation
# ---------------------------------------------------------------------------

def compute_technicals(price_history: Optional[list]) -> dict:
    """Compute RSI, MACD, SMA, and Bollinger Bands locally from price data.
    
    Replaces Alpha Vantage technical indicator API calls.
    All these are deterministic formulas on OHLCV data that yfinance already provides.
    
    Args:
        price_history: list of dicts with at least 'date' and 'close' keys
                      (as returned by YFinanceCollector.history())
    
    Returns:
        dict with computed indicators and metadata, or empty dict on failure.
    """
    import numpy as np

    if not price_history or len(price_history) < 35:
        # Need at least 35 days for MACD (26 + 9 signal smoothing)
        logger.warning("[technicals] Insufficient price data for technical computation")
        return {}
    
    try:
        closes = np.array([day["close"] for day in price_history], dtype=float)
        dates = [day["date"] for day in price_history]
        
        result: dict[str, Any] = {"computed_locally": True, "data_points": len(closes)}
        
        # --- RSI (Wilder's smoothing, period=14) ---
        if len(closes) >= 15:
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0.0)
            losses = np.where(deltas < 0, -deltas, 0.0)
            
            # Wilder's smoothing (equivalent to EMA with alpha=1/14)
            period = 14
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            for i in range(period, len(gains)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            result["rsi"] = round(rsi, 2)
            result["rsi_interpretation"] = (
                "overbought" if rsi > 70 else
                "oversold" if rsi < 30 else
                "neutral"
            )
        
        # --- MACD (12, 26, 9) ---
        if len(closes) >= 35:
            # EMA helper
            def _ema(data, span):
                alpha = 2 / (span + 1)
                ema = [data[0]]
                for val in data[1:]:
                    ema.append(alpha * val + (1 - alpha) * ema[-1])
                return np.array(ema)
            
            ema12 = _ema(closes, 12)
            ema26 = _ema(closes, 26)
            macd_line = ema12 - ema26
            signal_line = _ema(macd_line, 9)
            histogram = macd_line - signal_line
            
            result["macd"] = round(float(macd_line[-1]), 4)
            result["macd_signal"] = round(float(signal_line[-1]), 4)
            result["macd_histogram"] = round(float(histogram[-1]), 4)
            result["macd_interpretation"] = (
                "bullish" if macd_line[-1] > signal_line[-1] else "bearish"
            )
            # Check for recent crossover (within last 3 days)
            if len(macd_line) >= 4:
                prev_above = macd_line[-4] > signal_line[-4]
                curr_above = macd_line[-1] > signal_line[-1]
                if prev_above != curr_above:
                    result["macd_crossover"] = "bullish_crossover" if curr_above else "bearish_crossover"
        
        # --- Simple Moving Averages ---
        if len(closes) >= 20:
            result["sma_20"] = round(float(np.mean(closes[-20:])), 2)
        if len(closes) >= 50:
            result["sma_50"] = round(float(np.mean(closes[-50:])), 2)
        if len(closes) >= 200:
            result["sma_200"] = round(float(np.mean(closes[-200:])), 2)
        
        # Price vs SMA signals
        current_price = closes[-1]
        if "sma_50" in result:
            result["above_sma_50"] = bool(current_price > result["sma_50"])
        if "sma_200" in result:
            result["above_sma_200"] = bool(current_price > result["sma_200"])
        
        # Golden/Death cross detection (SMA 50 vs SMA 200)
        if len(closes) >= 200:
            sma50_prev = float(np.mean(closes[-51:-1]))
            sma200_prev = float(np.mean(closes[-201:-1]))
            sma50_now = result["sma_50"]
            sma200_now = result["sma_200"]
            if sma50_prev < sma200_prev and sma50_now > sma200_now:
                result["sma_cross"] = "golden_cross"
            elif sma50_prev > sma200_prev and sma50_now < sma200_now:
                result["sma_cross"] = "death_cross"
        
        # --- Bollinger Bands (20-day, 2 std) ---
        if len(closes) >= 20:
            bb_period = 20
            bb_mean = float(np.mean(closes[-bb_period:]))
            bb_std = float(np.std(closes[-bb_period:], ddof=1))
            result["bb_upper"] = round(bb_mean + 2 * bb_std, 2)
            result["bb_middle"] = round(bb_mean, 2)
            result["bb_lower"] = round(bb_mean - 2 * bb_std, 2)
            result["bb_width"] = round((result["bb_upper"] - result["bb_lower"]) / bb_mean * 100, 2)
            
            # Price position within bands
            if bb_std > 0:
                bb_position = (current_price - result["bb_lower"]) / (result["bb_upper"] - result["bb_lower"])
                result["bb_position"] = round(bb_position, 3)
                result["bb_interpretation"] = (
                    "above_upper" if current_price > result["bb_upper"] else
                    "below_lower" if current_price < result["bb_lower"] else
                    "within_bands"
                )
        
        # --- Volume Analysis ---
        if len(price_history) >= 20:
            volumes = np.array([day["volume"] for day in price_history], dtype=float)
            avg_vol_20 = float(np.mean(volumes[-20:]))
            current_vol = volumes[-1]
            if avg_vol_20 > 0:
                result["volume_ratio"] = round(current_vol / avg_vol_20, 2)
                result["volume_interpretation"] = (
                    "high_volume" if current_vol > 1.5 * avg_vol_20 else
                    "low_volume" if current_vol < 0.5 * avg_vol_20 else
                    "normal_volume"
                )
        
        result["as_of_date"] = dates[-1]
        return result
    
    except Exception as e:
        logger.error(f"[technicals] Computation failed: {e}")
        return {}
