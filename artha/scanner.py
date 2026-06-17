"""Market Scanner — discovers investment opportunities across stocks and crypto.

Scans multiple sources to find candidates worth a full council analysis.
This is the "what should we look at today?" layer.
"""
import logging
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
        if funnel is None:
            logger.warning("[scanner] Funnel unavailable — falling back to key ticker scan")
            return self._legacy_candidates(max_candidates)

        candidates = funnel.run(
            regime_packet=regime_packet,
            max_council_candidates=max_candidates,
            fallback_on_failure=True,
        )

        if not candidates:
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
