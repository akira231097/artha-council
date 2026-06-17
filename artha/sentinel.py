"""News Sentinel — intelligent news monitoring for portfolio and watchlist stocks.

Continuously monitors news for held positions and recent watchlist recommendations,
triggering alerts when significant news events are detected.
"""
import re
import json
import logging
import requests

# Shared HTTP session for connection pooling
_http_session = requests.Session()
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from hashlib import sha256

from .chatgpt_backend import ChatGPTBackendClient

from .config import Config
from .collector import DataCollector, FMPCollector
from .search import search_web
from .journal import DecisionJournal
from .portfolio import Portfolio, PORTFOLIO_FILE

logger = logging.getLogger(__name__)


# Categorized keywords with severity levels
CRITICAL_KEYWORDS = {
    # Existential threats
    "bankruptcy", "chapter 11", "chapter 7", "delisted", "delisting",
    "sec charges", "sec investigation", "securities fraud", "accounting fraud",
    "ceo fired", "ceo resigned", "ceo arrested", "cfo resigned",
    "fda rejection", "fda ban", "product recall", "class action",
    # Major events
    "hostile takeover", "acquisition offer", "merger agreement",
    "earnings miss", "earnings beat", "profit warning", "revenue miss",
    "stock split", "reverse split", "dividend cut", "dividend suspended",
    "downgrade", "upgrade", "price target cut", "price target raised",
    # Market-wide
    "market crash", "circuit breaker", "trading halted", "trading suspended",
}

HIGH_KEYWORDS = {
    "lawsuit", "investigation", "subpoena", "indictment",
    "layoffs", "restructuring", "plant closure", "factory shutdown",
    "data breach", "cybersecurity incident", "hack",
    "supply chain disruption", "sanctions", "tariff", "export ban",
    "short seller report", "accounting irregularities",
    "insider selling", "insider buying",
    "analyst downgrade", "analyst upgrade",
    "earnings surprise", "guidance raised", "guidance lowered",
    "buyback", "share repurchase", "secondary offering",
    "partnership", "contract win", "government contract",
}

MEDIUM_KEYWORDS = {
    "new product", "product launch", "patent", "fda approval",
    "expansion", "new market", "international",
    "board member", "executive hire", "management change",
    "debt refinancing", "credit rating", "bond offering",
    "esg", "environmental", "regulatory",
}


COMPANY_ALIAS_OVERRIDES = {
    "JNJ": {
        "johnson & johnson",
        "johnson and johnson",
        "j&j",
        "janssen",
        "stelara",
        "darzalex",
        "tremfya",
        "johnson medtech",
    },
    "BAC": {"bank of america", "bofa", "merrill lynch"},
    "MTCH": {"match group", "tinder", "hinge"},
}


@dataclass
class HeadlineEvent:
    """Raw headline event from news API."""
    ticker: str
    title: str
    url: str
    source: str
    published_date: str
    text: Optional[str] = None  # article snippet/summary
    headline_hash: str = field(default="")

    def __post_init__(self):
        if not self.headline_hash:
            self.headline_hash = sha256(f"{self.ticker}:{self.title}".encode()).hexdigest()[:16]


@dataclass
class ClassifiedEvent:
    """Headline event with classification."""
    event: HeadlineEvent
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, IRRELEVANT
    trigger_source: str  # keyword, sonnet
    trigger_detail: Optional[str] = None  # which keyword matched


@dataclass
class SentinelAlert:
    """Alert ready to be sent."""
    ticker: str
    severity: str
    headline: str
    url: str
    source: str
    published_date: str
    context: Optional[str] = None  # GPT assessment
    recommended_action: Optional[str] = None


class SentinelDeduplicator:
    """Tracks seen headlines to prevent duplicate alerts in high-frequency scans.

    Used by the 5-min sentinel cycle where the same headlines reappear repeatedly.
    """

    def __init__(self, ttl_hours: int = 24) -> None:
        self._seen: Dict[str, datetime] = {}  # headline_hash → first_seen
        self._ttl = timedelta(hours=ttl_hours)

    def _cleanup(self) -> None:
        now = datetime.now(timezone.utc)
        self._seen = {k: v for k, v in self._seen.items() if now - v < self._ttl}

    def is_new(self, headline_hash: str) -> bool:
        """Return True if this headline hasn't been seen within the TTL window."""
        self._cleanup()
        if headline_hash in self._seen:
            return False
        self._seen[headline_hash] = datetime.now(timezone.utc)
        return True

    def mark_seen(self, headline_hash: str) -> None:
        """Explicitly mark a headline as seen without querying."""
        self._seen[headline_hash] = datetime.now(timezone.utc)

    def filter_new_events(self, events: List[HeadlineEvent]) -> List[HeadlineEvent]:
        """Return only events not already seen."""
        return [e for e in events if self.is_new(e.headline_hash)]


class NewsSentinel:
    """News monitoring engine for Artha portfolio."""

    def __init__(
        self,
        collector: Optional[DataCollector] = None,
        alert_manager: Optional[Any] = None,  # Type Any to avoid circular import
        config: Optional[Config] = None,
    ):
        self.collector = collector or DataCollector()
        self.alert_manager = alert_manager  # Will be initialized by monitor
        self.config = config or Config
        self.journal = DecisionJournal()
        self._headline_cache: Dict[str, datetime] = {}  # headline_hash -> first_seen
        self._search_usage_today = 0
        self._search_usage_date = datetime.now(timezone.utc).date()
        self._search_result_cache: Dict[str, List] = {}  # query -> results (same-day cache)
        self._identity_cache: Dict[str, Set[str]] = {}
        # Fast dedup for high-frequency (5-min) scans — survives across calls
        self._fast_deduplicator = SentinelDeduplicator(ttl_hours=24)

    @staticmethod
    def _normalize_identity_text(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").replace("&", " and ").lower()).strip()

    def _company_aliases(self, ticker: str) -> Set[str]:
        ticker = str(ticker or "").upper().strip()
        if ticker in self._identity_cache:
            return self._identity_cache[ticker]
        aliases = set(COMPANY_ALIAS_OVERRIDES.get(ticker, set()))
        try:
            profile = self.collector.fmp.company_profile(ticker) or {}
            for key in ("companyName", "companyNameLong", "name", "company_name"):
                name = str(profile.get(key) or "").strip()
                if name:
                    aliases.add(name)
                    aliases.add(name.replace(",", ""))
                    aliases.add(name.replace(" Inc.", "").replace(" Corporation", "").replace(" Corp.", ""))
        except Exception as exc:
            logger.debug("[sentinel] Could not load identity aliases for %s: %s", ticker, exc)
        normalized = {
            self._normalize_identity_text(alias)
            for alias in aliases
            if len(self._normalize_identity_text(alias)) >= 3
        }
        self._identity_cache[ticker] = normalized
        return normalized

    def _article_symbol_matches(self, ticker: str, article: dict[str, Any]) -> bool:
        ticker = str(ticker or "").upper().strip()
        symbol_fields = (
            article.get("symbol"),
            article.get("ticker"),
            article.get("tickers"),
            article.get("symbols"),
            article.get("related"),
            article.get("stocks"),
        )
        found: set[str] = set()
        for value in symbol_fields:
            if isinstance(value, str):
                found.update(token.upper().strip().lstrip("$") for token in re.split(r"[,;\s]+", value) if token.strip())
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        candidate = item.get("symbol") or item.get("ticker")
                    else:
                        candidate = item
                    if candidate:
                        found.add(str(candidate).upper().strip().lstrip("$"))
        return ticker in found

    def _is_company_specific_news(self, ticker: str, title: str, text: str | None, article: dict[str, Any]) -> bool:
        if self._article_symbol_matches(ticker, article):
            return True
        searchable = self._normalize_identity_text(f"{title} {text or ''}")
        aliases = self._company_aliases(ticker)
        if any(alias and alias in searchable for alias in aliases):
            return True
        cashtag_pattern = re.compile(rf"(^|[^A-Z0-9])\${re.escape(str(ticker).upper())}([^A-Z0-9]|$)")
        if cashtag_pattern.search(str(title or "").upper()):
            return True
        logger.info("[sentinel] Dropped unrelated headline for %s: %s", ticker, str(title or "")[:160])
        return False

    def get_monitored_tickers(self) -> Dict[str, str]:
        """Get tickers to monitor from portfolio + recent recommendations.

        Returns:
            Dict mapping ticker -> priority ("HELD" or "WATCH")
        """
        monitored = {}

        # 1. Add held positions (highest priority)
        try:
            portfolio = Portfolio.load(PORTFOLIO_FILE)
            for position in portfolio.positions:
                if position.ticker:
                    monitored[position.ticker.upper()] = "HELD"
            logger.info(f"[sentinel] Monitoring {len(monitored)} held position(s)")
        except Exception as e:
            logger.error(f"[sentinel] Failed to load portfolio: {e}")

        # 2. Add recent watchlist recommendations (last 30 days)
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=30)
            with self.journal._connect() as conn:
                rows = conn.execute("""
                    SELECT DISTINCT ticker
                    FROM recommendations
                    WHERE action = 'WATCH'
                    AND datetime(timestamp) > ?
                    ORDER BY datetime(timestamp) DESC
                    LIMIT 20
                """, (cutoff.isoformat(),)).fetchall()

            watch_count = 0
            for row in rows:
                ticker = str(row["ticker"]).upper().strip() if row["ticker"] else ""
                if ticker and ticker not in monitored:
                    monitored[ticker] = "WATCH"
                    watch_count += 1
            logger.info(f"[sentinel] Added {watch_count} watchlist ticker(s) from recent recommendations")
        except Exception as e:
            logger.error(f"[sentinel] Failed to get watchlist tickers: {e}")

        return monitored

    def scan_headlines(self, tickers: Dict[str, str]) -> List[HeadlineEvent]:
        """Layer 1: Fetch headlines from FMP and Finnhub for all monitored tickers.

        Returns:
            List of HeadlineEvent objects for all tickers
        """
        all_events = []

        for ticker, priority in tickers.items():
            try:
                # Premium Benzinga news first when configured. This is the
                # lowest-latency news lane and is disabled unless licensed.
                benzinga_news = getattr(self.collector, "benzinga", None)
                benzinga_rows = benzinga_news.company_news(ticker, limit=20) if benzinga_news else None
                if benzinga_rows:
                    for article in benzinga_rows:
                        if not isinstance(article, dict):
                            continue
                        title = str(article.get("title") or "").strip()
                        if not title:
                            continue
                        try:
                            dt = datetime.fromisoformat(str(article.get("publishedDate") or "").replace("Z", "+00:00"))
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            dt = dt.astimezone(timezone.utc)
                        except Exception:
                            dt = datetime.now(timezone.utc)
                        if datetime.now(timezone.utc) - dt > timedelta(hours=Config.BENZINGA_NEWS_LOOKBACK_HOURS):
                            continue
                        text = str(article.get("text") or "")[:500] if article.get("text") else None
                        if not self._is_company_specific_news(ticker, title, text, article):
                            continue
                        all_events.append(HeadlineEvent(
                            ticker=ticker,
                            title=title,
                            url=str(article.get("url") or ""),
                            source=str(article.get("source") or "Benzinga"),
                            published_date=dt.isoformat(),
                            text=text,
                        ))

                # FMP news (limit 20 per ticker to stay cost-effective)
                fmp_news = self.collector.fmp.stock_news(ticker, limit=20)
                if fmp_news:
                    for article in fmp_news:
                        if not isinstance(article, dict):
                            continue
                        title = article.get("title", "")
                        if not title:
                            continue

                        # Parse published date
                        pub_date = article.get("publishedDate", "")
                        try:
                            # FMP format: "2024-03-07 14:30:00"
                            dt = datetime.strptime(pub_date, "%Y-%m-%d %H:%M:%S")
                            dt = dt.replace(tzinfo=timezone.utc)
                        except:
                            dt = datetime.now(timezone.utc)

                        # Skip old news (>6 hours)
                        age = datetime.now(timezone.utc) - dt
                        if age > timedelta(hours=6):
                            continue

                        text = article.get("text", "")[:500] if article.get("text") else None
                        if not self._is_company_specific_news(ticker, title, text, article):
                            continue
                        event = HeadlineEvent(
                            ticker=ticker,
                            title=title,
                            url=article.get("url", ""),
                            source=article.get("site", "FMP"),
                            published_date=dt.isoformat(),
                            text=text,
                        )
                        all_events.append(event)

                # Finnhub news (last 1 day)
                finnhub_news = self.collector.finnhub.company_news(ticker, days_back=1)
                if finnhub_news:
                    for article in finnhub_news[:10]:  # Limit to 10 most recent
                        if not isinstance(article, dict):
                            continue
                        headline = article.get("headline", "")
                        if not headline:
                            continue

                        # Finnhub timestamp is Unix epoch
                        timestamp = article.get("datetime", 0)
                        try:
                            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                        except:
                            dt = datetime.now(timezone.utc)

                        # Skip old news
                        age = datetime.now(timezone.utc) - dt
                        if age > timedelta(hours=6):
                            continue

                        text = article.get("summary", "")[:500] if article.get("summary") else None
                        if not self._is_company_specific_news(ticker, headline, text, article):
                            continue
                        event = HeadlineEvent(
                            ticker=ticker,
                            title=headline,
                            url=article.get("url", ""),
                            source=article.get("source", "Finnhub"),
                            published_date=dt.isoformat(),
                            text=text,
                        )
                        all_events.append(event)

            except Exception as e:
                logger.error(f"[sentinel] Failed to fetch headlines for {ticker}: {e}")

        logger.info(f"[sentinel] Collected {len(all_events)} fresh headline(s) across {len(tickers)} ticker(s)")
        return all_events

    def classify_headlines(self, events: List[HeadlineEvent]) -> List[ClassifiedEvent]:
        """Hybrid keyword + GPT classification.

        Returns:
            List of ClassifiedEvent objects with severity ratings
        """
        classified = []
        unclassified = []

        # First pass: keyword matching
        for event in events:
            # Combine title and text for matching
            searchable = f"{event.title} {event.text or ''}".lower()

            # Check keywords in priority order
            matched = False
            for keyword in CRITICAL_KEYWORDS:
                if keyword in searchable:
                    classified.append(ClassifiedEvent(
                        event=event,
                        severity="CRITICAL",
                        trigger_source="keyword",
                        trigger_detail=keyword
                    ))
                    matched = True
                    break

            if not matched:
                for keyword in HIGH_KEYWORDS:
                    if keyword in searchable:
                        classified.append(ClassifiedEvent(
                            event=event,
                            severity="HIGH",
                            trigger_source="keyword",
                            trigger_detail=keyword
                        ))
                        matched = True
                        break

            if not matched:
                for keyword in MEDIUM_KEYWORDS:
                    if keyword in searchable:
                        classified.append(ClassifiedEvent(
                            event=event,
                            severity="MEDIUM",
                            trigger_source="keyword",
                            trigger_detail=keyword
                        ))
                        matched = True
                        break

            if not matched:
                unclassified.append(event)

        logger.info(f"[sentinel] Keyword matching: {len(classified)} classified, {len(unclassified)} unclassified")

        # Second pass: GPT classification for unmatched headlines
        if unclassified and not self.config.SENTINEL_KEYWORD_ONLY:
            gpt_classified = self._classify_with_gpt(unclassified)
            classified.extend(gpt_classified)

        return classified

    def _classify_with_gpt(self, events: List[HeadlineEvent]) -> List[ClassifiedEvent]:
        """Use GPT 5.5 to classify headlines."""
        if not events:
            return []

        classified = []

        # Process in batches
        batch_size = max(1, int(self.config.SENTINEL_SONNET_BATCH_MAX or 30))
        for i in range(0, len(events), batch_size):
            batch = events[i:i+batch_size]

            # Build batch prompt
            headlines_section = []
            for idx, event in enumerate(batch):
                headlines_section.append(f'{idx}: "{event.title}" [{event.ticker}]')

            prompt = f"""You are a news significance classifier for a stock portfolio monitor.

For each headline below, classify its potential impact on the stock price as:
- CRITICAL: Existential risk or transformative event (bankruptcy, major fraud, acquisition, earnings disaster)
- HIGH: Significant price-moving event (lawsuit, major contract, analyst action, leadership change)
- MEDIUM: Notable but not urgent (product launch, partnership, regulatory update)
- LOW: Routine news, no material impact expected
- IRRELEVANT: Not about this company or not investment-relevant

Headlines:
{chr(10).join(headlines_section)}

Return ONLY a JSON object mapping headline index to classification:
{{"0": "HIGH", "1": "LOW", "2": "IRRELEVANT", ...}}"""

            try:
                response_text = ChatGPTBackendClient(timeout=30).chat(prompt)
                if response_text.startswith('```'):
                    response_text = response_text.split('```')[1].strip()
                    if response_text.startswith('json'):
                        response_text = response_text[4:].strip()

                classifications = json.loads(response_text)

                # Apply classifications
                for idx_str, severity in classifications.items():
                    idx = int(idx_str)
                    if 0 <= idx < len(batch) and severity in ["CRITICAL", "HIGH", "MEDIUM"]:
                        classified.append(ClassifiedEvent(
                            event=batch[idx],
                            severity=severity,
                            trigger_source="gpt",
                            trigger_detail=None
                        ))

                logger.info(f"[sentinel] GPT classified {len(classified)} significant headline(s) from batch of {len(batch)}")

            except Exception as e:
                logger.error(f"[sentinel] GPT classification failed: {e}")

        return classified

    def research_significant(self, events: List[ClassifiedEvent]) -> List[SentinelAlert]:
        """Layer 2: Run current-web search + GPT research on significant events."""
        alerts = []

        # Check daily search usage
        today = datetime.now(timezone.utc).date()
        if today != self._search_usage_date:
            self._search_usage_today = 0
            self._search_usage_date = today
            self._search_result_cache.clear()  # Clear cache on new day

        for event in events:
            # Skip if we've used too many current-web searches today
            if self._search_usage_today >= 100:  # Conservative daily limit
                logger.warning("[sentinel] Search daily limit reached, skipping research")
                # Still create alert without context
                alerts.append(SentinelAlert(
                    ticker=event.event.ticker,
                    severity=event.severity,
                    headline=event.event.title,
                    url=event.event.url,
                    source=event.event.source,
                    published_date=event.event.published_date,
                ))
                continue

            try:
                # Generate 2 targeted queries about this specific event
                queries = self._generate_event_queries(event)

                # Search and collect
                research_articles = []
                for query in queries[:1]:  # Max 1 query per event (conserve search calls)
                    results = self._search_web(query)
                    research_articles.extend(results[:3])  # Top 3 per query
                    self._search_usage_today += 1

                # Assess impact with GPT
                context, action = self._assess_impact(event, research_articles)

                alerts.append(SentinelAlert(
                    ticker=event.event.ticker,
                    severity=event.severity,
                    headline=event.event.title,
                    url=event.event.url,
                    source=event.event.source,
                    published_date=event.event.published_date,
                    context=context,
                    recommended_action=action,
                ))

            except Exception as e:
                logger.error(f"[sentinel] Research failed for {event.event.ticker}: {e}")
                # Still create alert without context
                alerts.append(SentinelAlert(
                    ticker=event.event.ticker,
                    severity=event.severity,
                    headline=event.event.title,
                    url=event.event.url,
                    source=event.event.source,
                    published_date=event.event.published_date,
                ))

        return alerts

    def _generate_event_queries(self, event: ClassifiedEvent) -> List[str]:
        """Generate 2 targeted search queries for a specific news event."""
        ticker = event.event.ticker
        # Extract key terms from headline
        headline = event.event.title

        # Remove common words and ticker
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been"}
        words = [w for w in headline.lower().split() if w not in stopwords and w != ticker.lower()]
        key_terms = " ".join(words[:5])  # First 5 meaningful words

        queries = [
            f"{ticker} {key_terms} impact analysis",
            f"{ticker} stock {event.trigger_detail or 'news'} investor reaction"
        ]

        return queries

    def _search_web(self, query: str) -> List[Dict]:
        """Call configured current-web search provider with same-day result caching."""
        cache_key = query.strip().lower()
        if cache_key in self._search_result_cache:
            logger.info(f"[sentinel] Search cache hit for: {query[:50]}...")
            return self._search_result_cache[cache_key]

        results = search_web(query, count=3, freshness="day")
        self._search_result_cache[cache_key] = results
        return results

    def _assess_impact(self, event: ClassifiedEvent, articles: List[Dict]) -> Tuple[str, str]:
        """Use GPT 5.5 to assess impact and recommend action."""
        if not articles:
            return "Impact assessment unavailable.", "Monitor closely."

        try:
            # Prepare article snippets
            snippets = []
            for article in articles[:5]:
                snippets.append(f"- {article['title']}: {article['snippet']}")

            prompt = f"""Assess the market impact of this news event for {event.event.ticker}:

HEADLINE: {event.event.title}
SEVERITY: {event.severity}
SOURCE: {event.event.source}

ADDITIONAL CONTEXT FROM SEARCH:
{chr(10).join(snippets)}

Provide:
1. A 2-3 sentence assessment of likely stock price impact
2. A specific recommended action for investors holding this position

Be concrete and specific. Reference historical precedents if relevant."""

            text = ChatGPTBackendClient(timeout=30).chat(prompt)
            lines = text.split('\n\n')

            context = lines[0] if lines else "Impact assessment completed."
            action = lines[1] if len(lines) > 1 else "Monitor position closely."

            # Clean up numbered lists
            context = re.sub(r'^1\.\s*', '', context)
            action = re.sub(r'^2\.\s*', '', action)

            return context, action

        except Exception as e:
            logger.error(f"[sentinel] Impact assessment failed: {e}")
            return "Impact assessment unavailable.", "Monitor closely."

    def run_scan(self, specific_ticker: Optional[str] = None) -> List[Any]:
        """Run full sentinel pipeline and return Alert objects.

        Args:
            specific_ticker: If provided, only scan this ticker (for price anomaly trigger)

        Returns:
            List of Alert objects compatible with existing AlertManager
        """
        # Import here to avoid circular dependency
        from .monitor import Alert

        try:
            # Get tickers to monitor
            if specific_ticker:
                # Check if it's a held position or watchlist
                all_monitored = self.get_monitored_tickers()
                if specific_ticker.upper() in all_monitored:
                    tickers = {specific_ticker.upper(): all_monitored[specific_ticker.upper()]}
                else:
                    # Not in our watchlist, but scan anyway if requested
                    tickers = {specific_ticker.upper(): "REQUESTED"}
            else:
                tickers = self.get_monitored_tickers()

            if not tickers:
                logger.info("[sentinel] No tickers to monitor")
                return []

            logger.info(f"[sentinel] Starting scan for {len(tickers)} ticker(s)")

            # Layer 1: Scan headlines
            headlines = self.scan_headlines(tickers)
            if not headlines:
                logger.info("[sentinel] No fresh headlines found")
                return []

            # Dedupe against recent alerts
            fresh_headlines = self._dedupe_headlines(headlines)
            logger.info(f"[sentinel] {len(fresh_headlines)} headline(s) after deduplication")

            # Classify headlines
            classified = self.classify_headlines(fresh_headlines)

            # Filter by priority and severity
            significant = []
            for event in classified:
                ticker_priority = tickers.get(event.event.ticker, "UNKNOWN")

                # HELD positions: alert on MEDIUM+
                # WATCH positions: alert on HIGH+
                # REQUESTED: alert on HIGH+ (price anomaly trigger)
                if ticker_priority == "HELD" and event.severity in ["CRITICAL", "HIGH", "MEDIUM"]:
                    significant.append(event)
                elif ticker_priority in ["WATCH", "REQUESTED"] and event.severity in ["CRITICAL", "HIGH"]:
                    significant.append(event)

            logger.info(f"[sentinel] {len(significant)} significant event(s) after filtering")

            if not significant:
                return []

            # Layer 2: Research significant events
            sentinel_alerts = self.research_significant(significant)

            # Convert to Alert objects
            alerts = []
            for sa in sentinel_alerts:
                # Build message
                message_parts = [f"📰 {sa.headline}"]
                if sa.context:
                    message_parts.append(f"\n📊 CONTEXT: {sa.context}")
                if sa.recommended_action:
                    message_parts.append(f"\n💡 ACTION: {sa.recommended_action}")
                message_parts.append(f"\n📎 Source: {sa.source}")

                # Use unique alert_type per headline so deduplication doesn't collapse distinct events
                headline_hash = sha256(sa.headline.encode()).hexdigest()[:12]
                # Normalize severity to match existing monitor pipeline conventions
                severity_map = {"CRITICAL": "CRITICAL", "HIGH": "WARNING", "MEDIUM": "INFO"}
                normalized_severity = severity_map.get(sa.severity, "INFO")
                alert = Alert(
                    ticker=sa.ticker,
                    alert_type=f"news_sentinel_{headline_hash}",
                    severity=normalized_severity,
                    message="\n".join(message_parts),
                    metadata={
                        "headline": sa.headline,
                        "url": sa.url,
                        "source": sa.source,
                        "published_date": sa.published_date,
                        "context": sa.context,
                        "action": sa.recommended_action,
                    }
                )
                alerts.append(alert)

            logger.info(f"[sentinel] Scan complete. {len(alerts)} alert(s) generated")
            return alerts

        except Exception as e:
            logger.error(f"[sentinel] Scan failed: {e}", exc_info=True)
            return []

    def run_fast_scan(self, held_tickers: List[str]) -> List["Alert"]:  # type: ignore[name-defined]  # noqa: F821
        """Tier-1 fast scan: keyword-only, no GPT, for high-frequency (5-min) cycles.

        Only scans HELD tickers. Uses the persistent fast deduplicator to avoid
        re-alerting the same headline every 5 minutes. Escalates CRITICAL hits
        immediately; HIGH hits still go through as reduced-cost alerts.

        Returns Alert objects compatible with AlertManager.
        """
        from .monitor import Alert

        if not held_tickers:
            return []

        try:
            tickers_dict = {t.upper(): "HELD" for t in held_tickers}
            # Fetch headlines (uses 6h recency window in scan_headlines)
            all_events = self.scan_headlines(tickers_dict)
            if not all_events:
                return []

            # Deduplicate using fast in-memory deduplicator
            fresh = self._fast_deduplicator.filter_new_events(all_events)
            if not fresh:
                return []

            logger.info("[sentinel_fast] %d fresh headline(s) after dedup for %d ticker(s)",
                        len(fresh), len(held_tickers))

            # Keyword-only classification (no GPT for speed)
            orig_kw_only = self.config.SENTINEL_KEYWORD_ONLY
            alerts: List[Alert] = []
            classified_events = []
            for event in fresh:
                searchable = f"{event.title} {event.text or ''}".lower()
                severity = None
                trigger = None
                for kw in CRITICAL_KEYWORDS:
                    if kw in searchable:
                        severity = "CRITICAL"
                        trigger = kw
                        break
                if not severity:
                    for kw in HIGH_KEYWORDS:
                        if kw in searchable:
                            severity = "HIGH"
                            trigger = kw
                            break
                if severity:
                    classified_events.append(ClassifiedEvent(
                        event=event,
                        severity=severity,
                        trigger_source="keyword",
                        trigger_detail=trigger,
                    ))

            if not classified_events:
                return []

            # Convert CRITICAL/HIGH to Alert objects (no web research — speed)
            from hashlib import sha256 as _sha256
            severity_map = {"CRITICAL": "CRITICAL", "HIGH": "WARNING"}
            for ce in classified_events:
                headline_hash = _sha256(ce.event.title.encode()).hexdigest()[:12]
                normalized_sev = severity_map.get(ce.severity, "WARNING")
                msg = (
                    f"📰 [{ce.severity}] {ce.event.title}\n"
                    f"⚡ Trigger: {ce.trigger_detail or 'keyword match'}\n"
                    f"📎 Source: {ce.event.source}"
                )
                alerts.append(Alert(
                    ticker=ce.event.ticker,
                    alert_type=f"news_fast_{headline_hash}",
                    severity=normalized_sev,
                    message=msg,
                    metadata={
                        "headline": ce.event.title,
                        "url": ce.event.url,
                        "source": ce.event.source,
                        "published_date": ce.event.published_date,
                        "severity": ce.severity,
                        "trigger": ce.trigger_detail,
                        "fast_scan": True,
                    },
                ))

            logger.info("[sentinel_fast] Fast scan complete — %d alert(s)", len(alerts))
            return alerts

        except Exception as e:
            logger.error("[sentinel_fast] Fast scan failed: %s", e)
            return []

    def run_scan_for_tickers(
        self,
        tickers: List[str],
        priority: str = "HELD",
    ) -> List[Any]:
        """Run full sentinel pipeline for a specific list of tickers.

        This is the same as run_scan() but accepts an explicit ticker list
        instead of loading from portfolio. Used by the scheduler's held-sentinel cycle.
        """
        from .monitor import Alert

        tickers_dict = {t.upper(): priority for t in tickers}
        if not tickers_dict:
            return []

        try:
            headlines = self.scan_headlines(tickers_dict)
            if not headlines:
                return []

            fresh = self._dedupe_headlines(headlines)
            classified = self.classify_headlines(fresh)

            significant = [
                e for e in classified
                if e.severity in ("CRITICAL", "HIGH", "MEDIUM")
            ]
            if not significant:
                return []

            sentinel_alerts = self.research_significant(significant)

            alerts = []
            from hashlib import sha256 as _sha256
            severity_map = {"CRITICAL": "CRITICAL", "HIGH": "WARNING", "MEDIUM": "INFO"}
            for sa in sentinel_alerts:
                message_parts = [f"📰 {sa.headline}"]
                if sa.context:
                    message_parts.append(f"\n📊 CONTEXT: {sa.context}")
                if sa.recommended_action:
                    message_parts.append(f"\n💡 ACTION: {sa.recommended_action}")
                message_parts.append(f"\n📎 Source: {sa.source}")

                headline_hash = _sha256(sa.headline.encode()).hexdigest()[:12]
                alerts.append(Alert(
                    ticker=sa.ticker,
                    alert_type=f"news_sentinel_{headline_hash}",
                    severity=severity_map.get(sa.severity, "INFO"),
                    message="\n".join(message_parts),
                    metadata={
                        "headline": sa.headline,
                        "url": sa.url,
                        "source": sa.source,
                        "published_date": sa.published_date,
                        "context": sa.context,
                        "action": sa.recommended_action,
                    },
                ))
            return alerts
        except Exception as e:
            logger.error("[sentinel] run_scan_for_tickers failed: %s", e)
            return []

    def _dedupe_headlines(self, headlines: List[HeadlineEvent]) -> List[HeadlineEvent]:
        """Remove headlines we've seen recently."""
        # Clean old cache entries (>12 hours)
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=12)
        self._headline_cache = {
            h: t for h, t in self._headline_cache.items()
            if t > cutoff
        }

        # Filter new headlines
        fresh = []
        for event in headlines:
            if event.headline_hash not in self._headline_cache:
                fresh.append(event)
                self._headline_cache[event.headline_hash] = now

        return fresh
