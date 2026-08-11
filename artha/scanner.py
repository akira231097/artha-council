"""Market Scanner — discovers investment opportunities across stocks and crypto.

Scans multiple sources to find candidates worth a full council analysis.
This is the "what should we look at today?" layer.
"""
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import yfinance as yf
import requests

from .config import Config
from .collector import (
    DataCollector, FMPCollector, FinnhubCollector, CoinGeckoCollector, YFinanceCollector,
    get_crypto_fear_greed_index, get_equity_sentiment_index, _safe_get,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Catalyst / Episodic-Pivot lane tunables (env-var overridable; Wave 2
# consolidates into config.py)
# ---------------------------------------------------------------------------

# Episodic pivot: gap >= +8% on an earnings/news day with elevated volume.
CATALYST_EP_MIN_GAP_PCT: float = float(os.getenv("ARTHA_CATALYST_EP_MIN_GAP_PCT", "8.0"))
CATALYST_EP_VOLUME_RATIO: float = float(os.getenv("ARTHA_CATALYST_EP_VOLUME_RATIO", "1.4"))
CATALYST_EP_MAX_CANDIDATES: int = int(os.getenv("ARTHA_CATALYST_EP_MAX_CANDIDATES", "8"))
CATALYST_EP_MIN_MARKET_CAP: float = float(os.getenv("ARTHA_CATALYST_EP_MIN_MARKET_CAP", "500000000"))
CATALYST_EP_MIN_PRICE: float = float(os.getenv("ARTHA_CATALYST_EP_MIN_PRICE", "5.0"))
CATALYST_NEWS_WINDOW_HOURS: int = int(os.getenv("ARTHA_CATALYST_NEWS_WINDOW_HOURS", "48"))
CATALYST_EP_EARNINGS_WINDOW_DAYS: int = int(
    os.getenv("ARTHA_CATALYST_EP_EARNINGS_WINDOW_DAYS", "2")
)
CATALYST_EP_REQUIRE_IDENTIFIED_CATALYST: bool = os.getenv(
    "ARTHA_CATALYST_EP_REQUIRE_IDENTIFIED_CATALYST",
    "true",
).strip().lower() in ("1", "true", "yes")
CATALYST_EP_TOP_MOVER_GAP_PCT: float = float(os.getenv("ARTHA_CATALYST_EP_TOP_MOVER_GAP_PCT", "15.0"))
CATALYST_EP_TOP_MOVER_VOLUME_RATIO: float = float(
    os.getenv("ARTHA_CATALYST_EP_TOP_MOVER_VOLUME_RATIO", str(CATALYST_EP_VOLUME_RATIO))
)


# ---------------------------------------------------------------------------
# Stock Discovery
# ---------------------------------------------------------------------------

def _get_yf_market_movers() -> dict:
    """Get top gainers, losers, and most active from FMP market mover endpoints.

    Falls back to _scan_key_tickers() if FMP fails.
    """
    results = {"gainers": [], "losers": [], "most_active": [], "trending": []}

    def _normalize(item: dict) -> dict:
        return {
            "symbol": item.get("symbol", ""),
            "name": item.get("name", ""),
            "price": item.get("price", 0),
            "change_pct": item.get("changesPercentage", 0),
            "volume": item.get("volume", 0),
            "market_cap": item.get("marketCap", 0),
        }

    try:
        fmp = FMPCollector()

        gainers = fmp.market_gainers(limit=10)
        if gainers:
            results["gainers"] = [_normalize(g) for g in gainers]

        losers = fmp.market_losers(limit=10)
        if losers:
            results["losers"] = [_normalize(l) for l in losers]

        actives = fmp.market_actives(limit=10)
        if actives:
            results["most_active"] = [_normalize(a) for a in actives]

        if results["gainers"] or results["losers"] or results["most_active"]:
            logger.info(
                f"FMP market movers: {len(results['gainers'])} gainers, "
                f"{len(results['losers'])} losers, {len(results['most_active'])} actives"
            )
            return results

        logger.warning("FMP market movers returned no data, falling back to key ticker scan")
    except Exception as e:
        logger.warning(f"FMP market movers failed: {e}, falling back to key ticker scan")

    return _scan_key_tickers()


def _scan_key_tickers() -> dict:
    """Scan a curated list of high-liquidity tickers for daily movers.
    
    This covers major sectors and is the fallback when screener APIs fail.
    """
    # Broad market coverage: mega-caps, growth, value, sectors
    watchlist = [
        # Mega-cap tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
        # AI / Semiconductor
        "AMD", "AVGO", "INTC", "MRVL", "QCOM", "ARM", "SMCI", "TSM",
        # Growth / Cloud
        "CRM", "SNOW", "PLTR", "NET", "DDOG", "MDB", "CRWD",
        # Fintech / Payments
        "XYZ", "PYPL", "COIN", "V", "MA",
        # Consumer
        "COST", "WMT", "TGT", "NKE", "SBUX", "MCD",
        # Healthcare
        "UNH", "JNJ", "LLY", "ABBV", "PFE", "MRNA",
        # Energy / Industrial
        "XOM", "CVX", "CAT", "BA", "GE",
        # ETFs (broad market)
        "SPY", "QQQ", "IWM", "DIA", "VTI",
        # High-growth / Speculative (where big moves happen)
        "RIVN", "LCID", "SOFI", "HOOD", "RKLB", "IONQ",
    ]
    
    results = {"gainers": [], "losers": [], "most_active": [], "trending": []}
    movers = []
    
    logger.info(f"Scanning {len(watchlist)} tickers for today's movers...")
    
    # Batch download for efficiency
    try:
        tickers_str = " ".join(watchlist)
        data = yf.download(
            tickers_str,
            period="2d",
            group_by="ticker",
            progress=False,
            threads=Config.YFINANCE_THREADS,
        )
        
        for ticker in watchlist:
            try:
                if ticker in data.columns.get_level_values(0):
                    ticker_data = data[ticker]
                    if len(ticker_data) >= 2:
                        today_close = ticker_data["Close"].iloc[-1]
                        prev_close = ticker_data["Close"].iloc[-2]
                        today_volume = ticker_data["Volume"].iloc[-1]
                        
                        if prev_close > 0:
                            change_pct = ((today_close - prev_close) / prev_close) * 100
                            movers.append({
                                "symbol": ticker,
                                "price": round(float(today_close), 2),
                                "change_pct": round(float(change_pct), 2),
                                "volume": int(today_volume) if today_volume == today_volume else 0,
                            })
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"Batch download failed: {e}, falling back to individual")
        for i, ticker in enumerate(watchlist[:20]):  # Limit to top 20 on fallback
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="2d")
                if len(hist) >= 2:
                    today_close = hist["Close"].iloc[-1]
                    prev_close = hist["Close"].iloc[-2]
                    today_volume = hist["Volume"].iloc[-1]
                    if prev_close > 0:
                        change_pct = ((today_close - prev_close) / prev_close) * 100
                        movers.append({
                            "symbol": ticker,
                            "price": round(float(today_close), 2),
                            "change_pct": round(float(change_pct), 2),
                            "volume": int(today_volume),
                        })
            except Exception:
                continue
            # Periodic cleanup every 10 tickers to prevent FD buildup
            if (i + 1) % 10 == 0:
                YFinanceCollector.cleanup_caches()
    finally:
        YFinanceCollector.cleanup_caches()
    
    # Sort by absolute change to find biggest movers
    movers.sort(key=lambda x: abs(x.get("change_pct", 0)), reverse=True)
    
    for m in movers:
        if m["change_pct"] > 0:
            results["gainers"].append(m)
        else:
            results["losers"].append(m)
    
    # Most active by volume
    by_volume = sorted(movers, key=lambda x: x.get("volume", 0), reverse=True)
    results["most_active"] = by_volume[:10]
    
    return results


# ---------------------------------------------------------------------------
# Crypto Discovery
# ---------------------------------------------------------------------------

def _get_trending_crypto() -> list[dict]:
    """Get trending and top-moving crypto from CoinGecko."""
    collector = CoinGeckoCollector()
    results = []
    
    # Trending coins
    trending = collector.trending()
    if trending and isinstance(trending, dict):
        coins = trending.get("coins", [])
        for coin in coins[:10]:
            item = coin.get("item", {})
            results.append({
                "id": item.get("id", ""),
                "symbol": item.get("symbol", "").upper(),
                "name": item.get("name", ""),
                "market_cap_rank": item.get("market_cap_rank"),
                "price_change_24h": item.get("data", {}).get("price_change_percentage_24h", {}).get("usd", 0),
                "source": "trending",
            })
    
    # Top coins by market cap with 24h changes
    top_coins = collector.price(
        ids="bitcoin,ethereum,solana,cardano,avalanche-2,chainlink,polkadot,dogecoin,shiba-inu,matic-network",
        vs="usd",
    )
    if top_coins and isinstance(top_coins, dict):
        for coin_id, data in top_coins.items():
            results.append({
                "id": coin_id,
                "symbol": coin_id.upper(),
                "name": coin_id.replace("-", " ").title(),
                "price": data.get("usd", 0),
                "change_24h": data.get("usd_24h_change", 0),
                "market_cap": data.get("usd_market_cap", 0),
                "source": "top_10",
            })
    
    return results


# ---------------------------------------------------------------------------
# News Catalyst Scanner
# ---------------------------------------------------------------------------

def _get_news_catalysts() -> list[dict]:
    """Scan recent market news for actionable catalysts."""
    finnhub = FinnhubCollector()
    catalysts = []
    
    news = finnhub.market_news(category="general")
    if news and isinstance(news, list):
        for article in news[:20]:
            if isinstance(article, dict):
                catalysts.append({
                    "headline": article.get("headline", ""),
                    "source": article.get("source", ""),
                    "related": article.get("related", ""),
                    "summary": article.get("summary", "")[:200],
                    "datetime": article.get("datetime", 0),
                })
    
    return catalysts


# ---------------------------------------------------------------------------
# Catalyst / Episodic-Pivot Lane
# ---------------------------------------------------------------------------

def get_catalyst_candidates(
    collector: Any = None,
    max_candidates: Optional[int] = None,
) -> list[dict]:
    """Catalyst / episodic-pivot (EP) lane for the promotion funnel.

    Finds stocks gapping >= +8% today on an identified earnings/news catalyst
    with elevated volume (>= 1.4x average). Very large high-volume movers with
    no identifiable event are retained as research/shadow candidates, but are
    not eligible for a guaranteed live catalyst slot by default. These are
    Qullamaggie-style episodic pivots: large single-day repricings that the
    skip-month momentum ranker structurally cannot surface.

    Sources, in priority order:
      1. collector.get_market_movers() when available (data-layer agent adds
         this in Wave 2 — probed defensively via hasattr)
      2. Direct FMP biggest-gainers/most-actives endpoints (fallback)

    Candidates are tagged track='catalyst_ep' so downstream scoring (funnel,
    opportunity scout, council) applies track-consistent logic instead of
    anti-momentum valuation penalties.

    Args:
        collector: Optional DataCollector-like object (checked via hasattr)
        max_candidates: Max candidates to return (default CATALYST_EP_MAX_CANDIDATES)

    Returns:
        List of candidate dicts shaped for funnel enrichment, each with
        track='catalyst_ep', gap_pct, volume_ratio, and catalyst_type.
    """
    from .earnings_calendar import get_earnings_context

    max_candidates = max_candidates or CATALYST_EP_MAX_CANDIDATES

    # --- Gather movers (defensive: prefer collector.get_market_movers) ---
    movers: list[dict] = []
    if collector is not None and hasattr(collector, "get_market_movers"):
        try:
            raw = collector.get_market_movers()
            if isinstance(raw, dict):
                movers = list(raw.get("gainers") or [])
            elif isinstance(raw, list):
                movers = list(raw)
        except Exception as e:
            logger.warning(f"[catalyst_ep] collector.get_market_movers failed: {e}")
    if not movers:
        try:
            fmp = FMPCollector()
            movers = [
                {
                    "symbol": g.get("symbol", ""),
                    "name": g.get("name", ""),
                    "price": g.get("price", 0),
                    "change_pct": g.get("changesPercentage", g.get("change_pct", 0)),
                }
                for g in (fmp.market_gainers(limit=30) or [])
            ]
        except Exception as e:
            logger.warning(f"[catalyst_ep] FMP gainers fallback failed: {e}")
            return []

    # --- Filter to episodic-pivot gaps ---
    def _pct(value) -> float:
        try:
            return float(str(value).replace("%", ""))
        except (TypeError, ValueError):
            return 0.0

    gappers = [
        m for m in movers
        if _pct(m.get("change_pct") or m.get("changePct") or m.get("changesPercentage")) >= CATALYST_EP_MIN_GAP_PCT
        and _pct(m.get("price")) >= CATALYST_EP_MIN_PRICE
    ]
    if not gappers:
        logger.info("[catalyst_ep] No movers gapping >= %.1f%%", CATALYST_EP_MIN_GAP_PCT)
        return []

    # Recent Finnhub market news for zero-extra-cost news confirmation
    news_related: set[str] = set()
    try:
        for article in _get_news_catalysts():
            for related in str(article.get("related") or "").split(","):
                related = related.strip().upper()
                if related:
                    news_related.add(related)
    except Exception:
        pass

    fmp = FMPCollector()

    def _article_matches_symbol(article: dict, symbol: str) -> bool:
        raw_values: list[Any] = []
        for key in ("symbol", "symbols", "ticker", "tickers"):
            value = article.get(key) if isinstance(article, dict) else None
            if value:
                raw_values.append(value)
        for value in raw_values:
            if isinstance(value, list):
                parts = value
            else:
                parts = str(value).replace("|", ",").replace(";", ",").split(",")
            for part in parts:
                item = str(part or "").strip().upper()
                if ":" in item:
                    item = item.rsplit(":", 1)[-1]
                if item == symbol:
                    return True
        return False

    def _history_avg_volume(symbol: str, lookback_days: int = 30) -> float:
        try:
            rows = fmp.history(symbol, period="3mo") or []
        except Exception as e:
            logger.info("[catalyst_ep] %s avg-volume history fallback failed: %s", symbol, e)
            return 0.0
        volumes: list[float] = []
        # Prefer excluding the current/latest bar when enough history exists.
        window = rows[-(lookback_days + 1):-1] if len(rows) > lookback_days else rows[-lookback_days:]
        for row in window:
            volume = _pct(row.get("volume"))
            if volume > 0:
                volumes.append(volume)
        if not volumes:
            return 0.0
        return sum(volumes) / len(volumes)

    now = datetime.now(timezone.utc)

    def _latest_positive_earnings(surprises: list[dict]) -> tuple[Optional[str], Optional[float]]:
        """Return a recent, already-reported positive EPS surprise.

        EarningsContext.days_to_earnings describes the next report, so it must
        never be used as proof that today's gap followed a completed report.
        """
        matches: list[tuple[datetime, float]] = []
        for row in surprises or []:
            raw_date = str((row or {}).get("date") or "")[:10]
            try:
                report_dt = datetime.strptime(raw_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            age_days = (now.date() - report_dt.date()).days
            surprise_pct = _pct((row or {}).get("surprise_pct"))
            if 0 <= age_days <= CATALYST_EP_EARNINGS_WINDOW_DAYS and surprise_pct > 0:
                matches.append((report_dt, surprise_pct))
        if not matches:
            return None, None
        report_dt, surprise_pct = max(matches, key=lambda item: item[0])
        return report_dt.date().isoformat(), surprise_pct

    candidates: list[dict] = []
    for mover in gappers[: max_candidates * 3]:
        if len(candidates) >= max_candidates:
            break
        symbol = str(mover.get("symbol") or mover.get("ticker") or "").upper()
        if not symbol:
            continue

        # Quote for volume confirmation + market cap gate. Elevated volume is
        # a HARD requirement for the EP thesis. Some FMP quote rows omit
        # avgVolume for valid liquid movers, so backfill from history before
        # deciding the volume gate is unknowable.
        try:
            quote = fmp.quote(symbol) or {}
        except Exception:
            quote = {}
        avg_volume = _pct(quote.get("avgVolume") or quote.get("averageVolume") or quote.get("avg_volume"))
        avg_volume_source = "quote"
        if avg_volume <= 0:
            avg_volume = _history_avg_volume(symbol)
            avg_volume_source = "history" if avg_volume > 0 else "missing"
        day_volume = _pct(quote.get("volume") or mover.get("volume"))
        market_cap = _pct(quote.get("marketCap") or mover.get("market_cap"))
        if market_cap < CATALYST_EP_MIN_MARKET_CAP:
            continue
        if avg_volume <= 0 or day_volume <= 0:
            continue
        volume_ratio = day_volume / avg_volume
        if volume_ratio < CATALYST_EP_VOLUME_RATIO:
            continue

        # Reject ETFs/funds that leak into the movers feed (e.g. leveraged
        # single-stock ETPs) — episodic pivots are a single-company thesis.
        try:
            profile = fmp.company_profile(symbol) or {}
            if profile.get("isEtf") or profile.get("isFund"):
                continue
        except Exception:
            pass

        # Catalyst confirmation: earnings day, or news within the window
        catalyst_type = None
        earnings_surprise_pct = None
        earnings_report_date = None
        try:
            ec = get_earnings_context(symbol)
            earnings_report_date, earnings_surprise_pct = _latest_positive_earnings(
                ec.recent_surprises
            )
            if earnings_report_date:
                catalyst_type = "earnings"
        except Exception:
            pass
        if catalyst_type is None and symbol in news_related:
            catalyst_type = "news"
        if catalyst_type is None:
            try:
                articles = fmp.stock_news(symbol, limit=3) or []
                for article in articles:
                    if not _article_matches_symbol(article, symbol):
                        continue
                    published = str(article.get("publishedDate") or "")
                    try:
                        published_dt = datetime.fromisoformat(published.replace(" ", "T"))
                        if published_dt.tzinfo is None:
                            published_dt = published_dt.replace(tzinfo=timezone.utc)
                        age_hours = (now - published_dt).total_seconds() / 3600
                        if 0 <= age_hours <= CATALYST_NEWS_WINDOW_HOURS:
                            catalyst_type = "news"
                            break
                    except ValueError:
                        continue
            except Exception:
                pass
        gap_pct = _pct(
            mover.get("change_pct") or mover.get("changePct") or mover.get("changesPercentage")
        )
        if (
            catalyst_type is None
            and gap_pct >= CATALYST_EP_TOP_MOVER_GAP_PCT
            and volume_ratio >= CATALYST_EP_TOP_MOVER_VOLUME_RATIO
        ):
            catalyst_type = "top_mover_volume"
        if catalyst_type is None:
            continue

        catalyst_confirmed = catalyst_type in {"earnings", "news"}
        live_eligible = catalyst_confirmed or not CATALYST_EP_REQUIRE_IDENTIFIED_CATALYST

        price = _pct(quote.get("price") or mover.get("price"))
        shadow_quality_score = min(40.0, gap_pct * 2.0) + min(30.0, volume_ratio * 10.0)
        if catalyst_type == "earnings":
            shadow_quality_score += 10.0
            if earnings_surprise_pct is not None and earnings_surprise_pct > 0:
                shadow_quality_score += min(20.0, earnings_surprise_pct / 2.0)
        candidates.append({
            "symbol": symbol,
            "name": str(quote.get("name") or mover.get("name") or ""),
            "sector": str(mover.get("sector") or ""),
            "industry": str(mover.get("industry") or ""),
            "market_cap": market_cap,
            "avg_volume": int(avg_volume),
            "avg_volume_source": avg_volume_source,
            "price": round(price, 2),
            "change_pct": round(gap_pct, 2),
            "gap_pct": round(gap_pct, 2),
            "volume_ratio": round(volume_ratio, 2) if volume_ratio is not None else None,
            "catalyst_type": catalyst_type,
            "catalyst_confirmed": catalyst_confirmed,
            "live_eligible": live_eligible,
            "routing_status": "execution_candidate" if live_eligible else "research_watch",
            "routing_reason": (
                "identified_catalyst"
                if live_eligible
                else "large_move_without_identified_earnings_or_news_catalyst"
            ),
            "earnings_report_date": earnings_report_date,
            "earnings_surprise_pct": earnings_surprise_pct,
            "episodic_pivot_shadow": {
                "quality_score": round(min(100.0, shadow_quality_score), 2),
                "gap_pct": round(gap_pct, 2),
                "volume_ratio": round(volume_ratio, 2),
                "catalyst_type": catalyst_type,
                "catalyst_confirmed": catalyst_confirmed,
                "earnings_surprise_pct": earnings_surprise_pct,
                "guidance_confirmation": "unavailable",
                "live_score_impact": 0.0,
            },
            "track": "catalyst_ep",
            "momentum_score": 0.0,
            "regime_score": 0.0,
            # Modest base so EP names enter enrichment without dominating
            # z-scored momentum leaders purely on gap size.
            "combined_score": round(min(30.0, gap_pct), 2),
            "return_1m": None,
            "return_3m": None,
            "return_12m": None,
            "vol_20d": None,
            "data_provider_lane": "catalyst_ep_scanner",
        })
        logger.info(
            "[catalyst_ep] %s gap=%.1f%% vol_ratio=%s catalyst=%s",
            symbol, gap_pct, volume_ratio, catalyst_type,
        )

    logger.info("[catalyst_ep] %d episodic-pivot candidates surfaced", len(candidates))
    return candidates


# ---------------------------------------------------------------------------
# Opportunity Scorer
# ---------------------------------------------------------------------------

def _score_stock_opportunity(ticker_data: dict, fear_greed: Optional[dict] = None) -> float:
    """Score a stock's opportunity level (0-100).
    
    Higher = more interesting for council analysis.
    Factors: magnitude of move, volume, fear/greed context, contrarian signals.
    
    Contrarian signals (inspired by Felix/Nat Eliason analysis):
    Instead of just chasing biggest movers, weight thesis divergence —
    stocks where insiders buy while price drops, or fundamentals diverge
    from price action.
    """
    score = 0.0
    
    change_pct = abs(ticker_data.get("change_pct", 0))
    raw_change = ticker_data.get("change_pct", 0)
    volume = ticker_data.get("volume", 0)
    
    # --- Momentum Score (what moved) ---
    if change_pct >= 5:
        score += 20
    elif change_pct >= 3:
        score += 15
    elif change_pct >= 1.5:
        score += 8
    
    # High volume confirms the move is real
    if volume > 50_000_000:
        score += 15
    elif volume > 10_000_000:
        score += 8
    
    # --- Contrarian Signal Score (thesis divergence) ---
    fg_value = fear_greed.get("value", 50) if fear_greed else 50
    
    # Extreme fear + big drop = classic contrarian buying opportunity
    # (Warren Buffett: "Be fearful when others are greedy, greedy when fearful")
    if fg_value < 25 and raw_change < -3:
        score += 25  # Extreme fear + major dip = high contrarian interest
    elif fg_value < 30 and raw_change < -2:
        score += 18  # Fear + dip = interesting
    
    # Moderate fear + drop in quality name = value opportunity
    if fg_value < 40 and raw_change < -1.5:
        score += 8
    
    # Extreme greed + spike = potential sell/avoid signal (still interesting to analyze)
    if fg_value > 70 and raw_change > 3:
        score += 12  # Greed + spike = caution worth analyzing
    
    # --- Insider/Institutional Signal Bonus ---
    # If ticker_data contains insider activity (from FMP), boost score
    insider_buy = ticker_data.get("insider_buy_signals", 0)
    if insider_buy > 0 and raw_change < 0:
        # Insiders buying while stock drops = strong contrarian signal
        score += min(20, insider_buy * 10)
    
    # Analyst divergence: if analysts are bullish but price is dropping
    analyst_consensus = ticker_data.get("analyst_consensus", "")
    if analyst_consensus in ("buy", "strong buy") and raw_change < -3:
        score += 15  # Analysts bullish + price dropping = thesis divergence
    
    return min(100, score)


def _score_crypto_opportunity(crypto_data: dict, fear_greed: Optional[dict] = None) -> float:
    """Score a crypto's opportunity level (0-100)."""
    score = 0.0
    
    change = abs(crypto_data.get("change_24h", 0) or crypto_data.get("price_change_24h", 0) or 0)
    
    # Crypto moves bigger, so thresholds are higher
    if change >= 10:
        score += 30
    elif change >= 5:
        score += 20
    elif change >= 3:
        score += 10
    
    # Trending = social momentum
    if crypto_data.get("source") == "trending":
        score += 15
    
    # Fear + crypto dip = classic DCA opportunity
    if fear_greed and fear_greed.get("value", 50) < 25:
        score += 20
    
    return min(100, score)


# ---------------------------------------------------------------------------
# Main Scanner
# ---------------------------------------------------------------------------

class MarketScanner:
    """Scans the market for investment opportunities.

    Usage:
        scanner = MarketScanner()
        opportunities = scanner.scan()
        # opportunities contains scored and ranked candidates

    For regime-aware candidate generation, prefer get_funnel_candidates()
    which uses the PromotionFunnel pipeline (universe → rank → enrich → score).
    _scan_key_tickers() is kept as emergency fallback.
    """

    def __init__(self):
        self.collector = DataCollector()
        self._funnel = None  # Lazy-initialized to avoid slow imports at startup

    def _get_funnel(self):
        """Lazy-initialize PromotionFunnel to avoid import overhead."""
        if self._funnel is None:
            try:
                from .funnel import PromotionFunnel
                self._funnel = PromotionFunnel()
            except Exception as e:
                logger.warning(f"[scanner] Could not initialize PromotionFunnel: {e}")
        return self._funnel

    def get_funnel_candidates(
        self,
        regime_packet,
        max_candidates: int = 8,
    ) -> list[dict]:
        """Generate top candidates via the PromotionFunnel pipeline.

        This is the preferred path for regime-aware scans. Falls back to
        legacy _scan_key_tickers() if the funnel fails.

        Args:
            regime_packet: RegimePacket or dict with regime_type/event_overlays
            max_candidates: Max candidates to return (default 8)

        Returns:
            List of enriched candidate dicts for council analysis.
        """
        funnel = self._get_funnel()
        strict_coverage = bool(getattr(Config, "SCAN_REQUIRE_MIN_RANK_COVERAGE", True))
        if funnel is None:
            logger.warning("[scanner] Funnel unavailable")
            return [] if strict_coverage else self._legacy_candidates(max_candidates)

        candidates = funnel.run(
            regime_packet=regime_packet,
            max_council_candidates=max_candidates,
            fallback_on_failure=not strict_coverage,
        )

        if not candidates:
            if strict_coverage:
                logger.warning("[scanner] Funnel returned no coverage-proven candidates; buy scan remains closed")
                return []
            logger.warning("[scanner] Funnel returned no candidates — using legacy scan")
            return self._legacy_candidates(max_candidates)

        return candidates

    def _legacy_candidates(self, max_candidates: int) -> list[dict]:
        """Emergency fallback: return candidates from _scan_key_tickers()."""
        movers = _scan_key_tickers()
        all_stocks = movers.get("gainers", []) + movers.get("losers", [])
        fear_greed = get_equity_sentiment_index()
        for s in all_stocks:
            s["opportunity_score"] = _score_stock_opportunity(s, fear_greed)
        all_stocks.sort(key=lambda x: x.get("opportunity_score", 0), reverse=True)
        return all_stocks[:max_candidates]
    
    def scan(self, max_stock_candidates: int = 5, max_crypto_candidates: int = 2) -> dict:
        """Run full market scan and return ranked opportunities.
        
        Returns:
            dict with keys:
                - market_snapshot: overall market status
                - stock_candidates: top stock opportunities for council analysis
                - crypto_candidates: top crypto opportunities
                - news_catalysts: key news items
                - fear_greed: current sentiment
                - scan_time: when the scan was run
        """
        logger.info("🔍 Starting market scan...")
        scan_start = datetime.now(timezone.utc)
        
        # 1. Market sentiment
        logger.info("  📊 Checking market sentiment...")
        crypto_fear_greed = get_crypto_fear_greed_index()
        
        # 2. Stock movers
        logger.info("  📈 Scanning stock movers...")
        stock_movers = _get_yf_market_movers()
        
        # 3. Crypto trending
        logger.info("  🪙 Scanning crypto trends...")
        crypto_data = _get_trending_crypto()
        
        # 4. News catalysts
        logger.info("  📰 Scanning news catalysts...")
        news = _get_news_catalysts()
        
        # 5. Market snapshot (index prices)
        logger.info("  🌡️ Getting market snapshot...")
        sp500 = self.collector.fmp.quote("SPY")
        nasdaq = self.collector.fmp.quote("QQQ")
        btc = self.collector.fmp.quote("BTCUSD")
        eth = self.collector.fmp.quote("ETHUSD")
        vix = self.collector.yf.vix()
        equity_fear_greed = get_equity_sentiment_index(
            {"sp500": sp500, "nasdaq": nasdaq, "vix": vix}
        )
        
        # 6. Score and rank stocks
        all_stocks = stock_movers.get("gainers", []) + stock_movers.get("losers", [])
        for stock in all_stocks:
            stock["opportunity_score"] = _score_stock_opportunity(stock, equity_fear_greed)
        
        # Sort by opportunity score
        all_stocks.sort(key=lambda x: x.get("opportunity_score", 0), reverse=True)
        stock_candidates = all_stocks[:max_stock_candidates]
        
        # 7. Score and rank crypto
        for crypto in crypto_data:
            crypto["opportunity_score"] = _score_crypto_opportunity(crypto, crypto_fear_greed)
        
        crypto_data.sort(key=lambda x: x.get("opportunity_score", 0), reverse=True)
        crypto_candidates = crypto_data[:max_crypto_candidates]
        
        # 8. Build market snapshot
        market_snapshot = {
            "sp500": sp500,
            "nasdaq": nasdaq,
            "btc": btc,
            "eth": eth,
            "vix": vix,
            "fear_greed": equity_fear_greed,
            "fear_greed_crypto": crypto_fear_greed,
            "top_gainers": stock_movers.get("gainers", [])[:5],
            "top_losers": stock_movers.get("losers", [])[:5],
            "most_active": stock_movers.get("most_active", [])[:5],
        }
        
        scan_result = {
            "scan_time": scan_start.isoformat(),
            "market_snapshot": market_snapshot,
            "stock_candidates": stock_candidates,
            "crypto_candidates": crypto_candidates,
            "news_catalysts": news[:10],
            "fear_greed": equity_fear_greed,
            "fear_greed_crypto": crypto_fear_greed,
        }
        
        logger.info(
            f"  ✅ Scan complete: {len(stock_candidates)} stock candidates, "
            f"{len(crypto_candidates)} crypto candidates"
        )
        
        # Clean up yfinance SQLite FDs to prevent leak in long-running daemon
        YFinanceCollector.cleanup_caches()
        
        return scan_result
    
    def format_scan_summary(self, scan_result: dict) -> str:
        """Format scan results into a Telegram-friendly summary."""
        fg = scan_result.get("fear_greed", {}) or {}
        snapshot = scan_result.get("market_snapshot", {})
        stocks = scan_result.get("stock_candidates", [])
        crypto = scan_result.get("crypto_candidates", [])
        
        sp500 = snapshot.get("sp500", {}) or {}
        btc = snapshot.get("btc", {}) or {}
        
        lines = [
            "🔍 **ARTHA MARKET SCAN**",
            f"{datetime.now(timezone.utc).strftime('%A, %B %d %Y')}",
            "",
            f"Equity Sentiment: {fg.get('value', '?')} ({fg.get('label', '?')})",
            f"S&P 500: ${sp500.get('price', 'N/A')} ({sp500.get('changesPercentage', 0):+.1f}%)" if isinstance(sp500.get('changesPercentage'), (int, float)) else f"S&P 500: ${sp500.get('price', 'N/A')}",
            f"BTC: ${btc.get('price', 'N/A')}",
            "",
            "━━━━━━━━━━━━━━━",
            "",
            "📊 **TOP STOCK CANDIDATES FOR COUNCIL:**",
        ]
        
        for i, s in enumerate(stocks, 1):
            score = s.get("opportunity_score", 0)
            change = s.get("change_pct", 0)
            emoji = "🟢" if change > 0 else "🔴"
            lines.append(f"  {i}. {emoji} ${s['symbol']} ({change:+.1f}%) — Score: {score:.0f}/100")
        
        if crypto:
            lines.append("")
            lines.append("🪙 **TOP CRYPTO CANDIDATES:**")
            for i, c in enumerate(crypto, 1):
                change = c.get("change_24h", 0) or c.get("price_change_24h", 0) or 0
                score = c.get("opportunity_score", 0)
                emoji = "🟢" if change > 0 else "🔴"
                lines.append(f"  {i}. {emoji} {c.get('symbol', '?')} ({change:+.1f}%) — Score: {score:.0f}/100")
        
        lines.append("")
        lines.append("━━━━━━━━━━━━━━━")
        lines.append("_Running full council analysis on top candidates..._")
        
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Regime-Aware Candidate Generation
# ---------------------------------------------------------------------------

def generate_regime_candidates(
    regime_packet,
    mover_data: dict | None = None,
    max_candidates: int = 7,
) -> list[dict]:
    """Generate stock candidates based on regime analysis.

    Priority order:
    1. Tactical ETFs from event overlays (highest regime confidence)
    2. Tactical stocks from event overlays
    3. ETFs from base regime
    4. Existing movers that ALIGN with the regime
    5. Top movers that align with active regime

    Args:
        regime_packet: RegimePacket from MROL
        mover_data: Optional dict from _get_yf_market_movers() or _scan_key_tickers()
        max_candidates: Maximum candidates to return

    Returns:
        List of candidate dicts with symbol, source, regime_reason
    """
    candidates = []
    seen = set()

    # Skip these — they're ETFs we recommend as core/tactical, not for council analysis
    SKIP_COUNCIL = {"VOO", "VTI", "SPY", "QQQ", "IWM", "DIA", "FXAIX"}  # Broad ETFs skip council

    # 1. From tactical recommendations (regime-driven)
    for rec in regime_packet.tactical_recommendations:
        ticker = rec.get("ticker", "")
        if ticker and ticker not in seen and ticker not in SKIP_COUNCIL:
            seen.add(ticker)
            candidates.append({
                "symbol": ticker,
                "source": rec.get("source", "regime"),
                "regime_reason": rec.get("reason", ""),
                "opportunity_score": 70 + int(rec.get("confidence", 0) * 30),
            })

    # 2. From movers that align with regime
    if mover_data and len(candidates) < max_candidates:
        # Get beneficiary sectors from the regime
        beneficiary_sectors = set()
        avoid_sectors = set()

        # From event overlays
        for overlay in regime_packet.event_overlays:
            otype = overlay.get("type", "")
            from .regime_mapping import REGIME_TAXONOMY
            regime_info = REGIME_TAXONOMY.get(otype, {})
            for etf in regime_info.get("beneficiary_etfs", []):
                beneficiary_sectors.add(etf)
            for stock in regime_info.get("beneficiary_stocks", []):
                beneficiary_sectors.add(stock)
            for sector in regime_info.get("avoid_sectors", []):
                avoid_sectors.add(sector)

        # Check movers against regime
        all_movers = mover_data.get("gainers", []) + mover_data.get("losers", [])
        for mover in all_movers:
            ticker = mover.get("symbol", "")
            if (
                ticker
                and ticker not in seen
                and ticker not in SKIP_COUNCIL
                and ticker in beneficiary_sectors
                and len(candidates) < max_candidates
            ):
                seen.add(ticker)
                candidates.append({
                    "symbol": ticker,
                    "source": "regime_aligned_mover",
                    "regime_reason": "Mover aligned with active regime",
                    "opportunity_score": mover.get("opportunity_score", 50),
                    "change_pct": mover.get("change_pct", 0),
                })

    # 3. Fill remaining with top movers (existing behavior) if under limit
    if mover_data and len(candidates) < max_candidates:
        all_movers = mover_data.get("gainers", []) + mover_data.get("losers", [])
        all_movers.sort(key=lambda x: abs(x.get("change_pct", 0)), reverse=True)
        for mover in all_movers:
            ticker = mover.get("symbol", "")
            if (
                ticker
                and ticker not in seen
                and ticker not in SKIP_COUNCIL
                and len(candidates) < max_candidates
            ):
                seen.add(ticker)
                candidates.append({
                    "symbol": ticker,
                    "source": "market_mover",
                    "regime_reason": "Top market mover (not regime-specific)",
                    "opportunity_score": mover.get("opportunity_score", 30),
                    "change_pct": mover.get("change_pct", 0),
                })

    return candidates[:max_candidates]
