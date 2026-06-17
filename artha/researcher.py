"""Research Desk — intelligent internet research for the Artha Investment Council.

Runs BEFORE the analyst council to gather current market intelligence through
web searches, news analysis, and synthesis into structured Intelligence Briefs.

Produces actionable intelligence that enhances analyst decision-making with
real-time market context and sourced bull/bear cases.
"""
import re
import json
import logging
import requests

# Shared HTTP session for connection pooling
_http_session = requests.Session()
from typing import Optional, Dict, List
from datetime import datetime
from urllib.parse import urlparse

from .chatgpt_backend import ChatGPTBackendClient
from .config import Config
from .gemini_client import gemini_generate
from .search import search_web

logger = logging.getLogger(__name__)

# Known-broken Yahoo Finance URL patterns — always 404, skip fetching and use snippet fallback
_SKIP_URL_PATTERNS = [
    re.compile(r"finance\.yahoo\.com/quote/[^/]+/holders"),
    re.compile(r"finance\.yahoo\.com/quote/[^/]+/press-releases"),
    re.compile(r"finance\.yahoo\.com/quote/[^/]+/insider-transactions"),
    re.compile(r"finance\.yahoo\.com/quote/[^/]+/community"),
    re.compile(r"finance\.yahoo\.com/quote/[^/]+/performance"),
]


def _is_readable_article_text(content: str) -> bool:
    """Reject compressed/binary/redirect bodies that decoded into garbage text."""
    text = (content or "").strip()
    if len(text) < 80:
        return False
    sample = text[:2000]
    replacement_ratio = sample.count("\ufffd") / max(1, len(sample))
    if replacement_ratio > 0.01:
        return False
    printable_ratio = sum(1 for ch in sample if ch.isprintable() or ch.isspace()) / max(1, len(sample))
    if printable_ratio < 0.92:
        return False
    alpha_count = sum(1 for ch in sample if ch.isalpha())
    if alpha_count / max(1, len(sample)) < 0.25:
        return False
    return True


def _truncate_brief_preserving_sources(brief: str, max_chars: int = 4500) -> str:
    """Trim long briefs without dropping the source list when one exists."""
    text = brief or ""
    if len(text) <= max_chars:
        return text

    suffix = "\n\n[Brief truncated for token efficiency]"
    marker_match = re.search(r"\nSOURCES:\s*", text, flags=re.IGNORECASE)
    if not marker_match:
        return text[: max(0, max_chars - len(suffix))].rstrip() + suffix

    sources = text[marker_match.start():].strip()
    sources = sources[:1200].rstrip()
    separator = "\n\n[Brief truncated for token efficiency; source list preserved]\n\n"
    head_budget = max(500, max_chars - len(separator) - len(sources))
    return text[:head_budget].rstrip() + separator + sources


class ResearchDesk:
    """Research Desk orchestrates internet research for investment intelligence."""

    def __init__(self) -> None:
        self._cached_macro_brief: Optional[str] = None
        self._macro_brief_time: Optional[datetime] = None
        self._search_result_cache: dict = {}
        self._search_cache_date = None

    def research_stock(self, ticker: str, stock_data: Dict, macro_data: Dict) -> str:
        """Orchestrate full research pipeline for one stock.

        Args:
            ticker: Stock symbol (e.g., "NVDA")
            stock_data: Collected stock data from FMP/Finnhub/etc
            macro_data: Macro economic data

        Returns:
            Intelligence Brief as a formatted string
        """
        logger.info(f"🔍 Research Desk beginning intelligence gathering for {ticker}...")

        try:
            # Build macro context (cached across stocks)
            macro_brief = self._build_macro_brief(macro_data)

            # Generate targeted search queries
            queries = self._generate_queries(ticker, stock_data, macro_brief)
            logger.info(f"  📋 Generated {len(queries)} search queries")

            # Search and collect articles
            all_results = []
            for query in queries:
                results = self._search_web(query, count=5, freshness="week")
                logger.info(f"  🔎 '{query[:50]}...' → {len(results)} results")
                all_results.extend(results)

            # Rank and deduplicate
            ranked_results = self._rank_and_deduplicate(all_results)
            logger.info(f"  🏆 Ranked to top {len(ranked_results)} articles")

            # Fetch article content (use snippet as fallback for paywalled sources)
            articles = []
            for result in ranked_results:
                fetched_content = self._fetch_article(result['url'])
                if fetched_content and len(fetched_content) > 200:
                    result['content'] = fetched_content
                    articles.append(result)
                    logger.info(f"  📰 Fetched: {result['title'][:50]}...")
                elif result.get('snippet'):
                    # Use search snippet — still valuable context from paywalled sources
                    result['content'] = f"[Snippet] {result['snippet']}"
                    articles.append(result)
                    logger.info(f"  📎 Snippet fallback: {result['title'][:50]}...")
                else:
                    logger.warning(f"  ❌ No content available: {result['url']}")

            logger.info(f"  📚 Successfully fetched {len(articles)} article(s)")

            # Synthesize intelligence brief
            brief = self._synthesize_brief(ticker, stock_data, macro_brief, articles)
            # Cap brief to prevent bloating analyst prompts (4500 chars ≈ 1100 tokens)
            # while preserving source visibility for auditability.
            if len(brief) > 4500:
                brief = _truncate_brief_preserving_sources(brief, max_chars=4500)
                logger.info(f"  ✅ Intelligence Brief complete (truncated to {len(brief)} chars)")
            else:
                logger.info(f"  ✅ Intelligence Brief complete ({len(brief)} chars)")

            return brief

        except Exception as e:
            logger.error(f"Research Desk failed for {ticker}: {e}")
            return self._create_fallback_brief(ticker, stock_data)

    def _build_macro_brief(self, macro_data: Dict) -> str:
        """Build macro environment context.

        Runs 2-3 broad searches on macro conditions. Cached per session.
        """
        # Cache with 1-hour TTL for long-lived daemon processes
        if self._cached_macro_brief and self._macro_brief_time:
            age_seconds = (datetime.now() - self._macro_brief_time).total_seconds()
            if age_seconds < 3600:  # 1 hour TTL
                logger.info(f"  📊 Using cached macro brief (age: {age_seconds/60:.0f}m)")
                return self._cached_macro_brief
            else:
                logger.info("  📊 Macro cache expired, refreshing...")

        logger.info("  🌍 Building fresh macro brief...")

        current_year = datetime.now().year
        current_month = datetime.now().strftime("%B")
        macro_queries = [
            f"stock market major news events this week {current_month} {current_year}",
            f"Federal Reserve interest rates economic outlook {current_month} {current_year}",
            f"geopolitical risks financial markets {current_year}",
        ]

        macro_articles = []
        for query in macro_queries:
            results = self._search_web(query, count=5, freshness="week")
            for result in results[:2]:  # Top 2 per query
                content = self._fetch_article(result['url'])
                if content:
                    result['content'] = content
                    macro_articles.append(result)
                elif result.get("snippet"):
                    result["content"] = f"[Snippet] {result['snippet']}"
                    macro_articles.append(result)

        # Generate macro brief using Gemini Flash
        macro_brief = self._synthesize_macro_brief(macro_articles, macro_data)
        self._cached_macro_brief = macro_brief
        self._macro_brief_time = datetime.now()

        return macro_brief

    def _generate_queries(self, ticker: str, stock_data: Dict, macro_brief: str) -> List[str]:
        """Generate 6-8 targeted search queries using Gemini 3 Flash Preview.

        Context-aware queries considering sector, fundamentals, and macro environment.
        """
        # Extract context from stock data (before try so fallback can use them)
        profile = stock_data.get('profile', {}) or {}
        company_name = profile.get('companyName', ticker) or ticker
        sector = profile.get('sector', 'Unknown') or 'Unknown'
        industry = profile.get('industry', 'Unknown') or 'Unknown'

        try:

            quote = stock_data.get('quote', {}) or stock_data.get('yf_quote', {})
            current_price = quote.get('price', 'Unknown')

            # Get recent earnings info
            earnings = (stock_data.get('earnings_surprises') or [{}])[0]
            last_earnings = earnings.get('period', 'Unknown')

            prompt = f"""Generate 6-8 targeted Google search queries for investment research on {ticker} ({company_name}).

CONTEXT:
- Sector: {sector}
- Industry: {industry}
- Current Price: ${current_price}
- Last Earnings: {last_earnings}
- Macro Environment Summary: {macro_brief[:300]}...

QUERY REQUIREMENTS:
1. Mix of company-specific and sector-wide queries
2. Include recent timeframe keywords (2026, recent, latest, this week)
3. Target investment-relevant topics: earnings, guidance, partnerships, regulatory changes, analyst upgrades/downgrades
4. Consider both bullish and bearish angles
5. One query should focus on insider trading/institutional activity
6. ALWAYS include the full company name AND ticker in most queries to avoid results about unrelated companies
7. Prefer queries like "{ticker} {company_name} earnings" over just "{ticker} earnings"

Return ONLY a JSON array of search query strings, no other text:
["query1", "query2", "query3", ...]"""

            response_text, _ = gemini_generate(prompt, model=Config.GEMINI_FLASH_MODEL, timeout=45)
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif response_text.startswith('```'):
                response_text = response_text.split('```')[1].strip()

            queries = json.loads(response_text)
            # Validate: keep only non-empty strings, dedupe, cap at 8
            seen = set()
            valid = []
            for q in queries:
                if isinstance(q, str) and q.strip() and q.strip() not in seen:
                    seen.add(q.strip())
                    valid.append(q.strip())
            return valid[:8]

        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            # Fallback to hardcoded queries
            current_year = datetime.now().year
            return [
                f"{ticker} earnings guidance {current_year}",
                f"{ticker} analyst price target upgrade downgrade",
                f"{ticker} news recent developments {current_year}",
                f"{ticker} {sector} sector outlook {current_year}",
                f"{ticker} insider trading institutional buying selling",
                f"{ticker} partnerships acquisitions {current_year}",
                f"{ticker} competition market share analysis",
                f"{ticker} bearish risks concerns headwinds"
            ]

    def _search_web(self, query: str, *, count: int = 5, freshness: str = "week") -> List[Dict]:
        """Call the configured current-web search provider."""
        today = datetime.now().date()
        if today != self._search_cache_date:
            self._search_result_cache.clear()
            self._search_cache_date = today

        cache_key = f"{freshness}:{count}:{query.strip().lower()}"
        if cache_key in self._search_result_cache:
            logger.info(f"[researcher] Search cache hit for: {query[:50]}...")
            return self._search_result_cache[cache_key]

        results = search_web(query, count=count, freshness=freshness)
        self._search_result_cache[cache_key] = results
        return results

    def _fetch_article(self, url: str, timeout: int = 10) -> str:
        """Fetch and clean article content.

        Returns first 3000 chars of readable text, handles errors gracefully.
        Skips known-broken URL patterns to avoid wasted requests.
        """
        # Skip known-broken URLs (e.g. Yahoo Finance restructured endpoints)
        for pattern in _SKIP_URL_PATTERNS:
            if pattern.search(url):
                logger.debug(f"Skipping known-broken URL pattern: {url}")
                return ""

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }

            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()

            # Skip non-HTML content (PDFs, images, etc.)
            content_type = response.headers.get('Content-Type', '')
            if not any(t in content_type.lower() for t in ['text/html', 'text/plain', 'application/xhtml']):
                logger.warning(f"Skipping non-HTML content: {content_type} for {url}")
                response.close()
                return ""

            # Read up to 500KB to prevent memory issues
            max_bytes = 512_000
            response.raw.decode_content = True
            raw = response.raw.read(max_bytes)
            response.close()

            # Decode with fallback
            try:
                content = raw.decode(response.encoding or 'utf-8', errors='replace')
            except (UnicodeDecodeError, LookupError):
                content = raw.decode('utf-8', errors='replace')

            # Remove script and style tags
            content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
            content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)

            # Remove all HTML tags
            content = re.sub(r'<[^>]+>', '', content)

            # Clean up whitespace
            content = re.sub(r'\s+', ' ', content).strip()

            if not _is_readable_article_text(content):
                logger.warning(f"Skipping unreadable article body for {url}")
                return ""

            # Return first 3000 chars
            return content[:3000] if content else ""

        except Exception as e:
            logger.warning(f"Failed to fetch article {url}: {e}")
            return ""

    def _rank_and_deduplicate(self, all_results: List[Dict]) -> List[Dict]:
        """Rank results by source quality and query overlap.

        Returns top 5 deduplicated results.
        """
        # Source quality tiers (ranked by BOTH credibility AND accessibility)
        # Tier 1: High quality AND accessible for full article fetching
        tier_1_domains = {'cnbc.com', 'finance.yahoo.com', 'fool.com', 'investopedia.com', 'sec.gov', 'federalreserve.gov'}
        # Tier 2: High quality but often paywalled (snippet value only)
        tier_2_domains = {'reuters.com', 'bloomberg.com', 'wsj.com', 'ft.com', 'barrons.com', 'seekingalpha.com', 'marketwatch.com', 'investors.com', 'yahoo.com'}

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        url_query_count = {}

        for result in all_results:
            url = result['url']
            if url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
                url_query_count[url] = 1
            else:
                # Count appearances across queries (relevance signal)
                url_query_count[url] = url_query_count.get(url, 1) + 1

        # Score each result
        scored_results = []
        for result in unique_results:
            url = result['url']
            domain = urlparse(url).netloc.lower()

            # Remove 'www.' prefix for matching
            domain = domain.replace('www.', '')

            # Assign tier weight
            if domain in tier_1_domains:
                tier_weight = 3
            elif domain in tier_2_domains:
                tier_weight = 2
            else:
                tier_weight = 1

            # Calculate final score
            appearance_count = url_query_count.get(url, 1)
            score = tier_weight * appearance_count

            scored_results.append((score, result))

        # Sort by score and return top 8 (some will be snippet-only from paywalled sites)
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [result for score, result in scored_results[:8]]

    def _synthesize_brief(self, ticker: str, stock_data: Dict, macro_brief: str, articles: List[Dict]) -> str:
        """Extract structured Intelligence Brief using GPT — factual extraction, not summarization."""
        if not articles:
            return self._create_insufficient_research_brief(ticker, stock_data)

        try:
            # Extract basic company info
            profile = stock_data.get('profile', {}) or {}
            company_name = profile.get('companyName', ticker) or ticker
            sector = profile.get('sector', 'Unknown') or 'Unknown'

            # Prepare article content — give GPT enough text to extract from (1500 chars per article)
            article_sections = []
            sources = []
            for article in articles:
                article_sections.append(
                    f"--- SOURCE: {article['title']} ({urlparse(article['url']).netloc}) ---\n"
                    f"{article['content'][:1500]}"
                )
                sources.append(f"- {article['title']} ({urlparse(article['url']).netloc})")

            sources_list = '\n'.join(sources)
            articles_text = '\n\n'.join(article_sections)

            current_date = datetime.now().strftime("%Y-%m-%d")

            prompt = f"""You are a research analyst preparing a factual intelligence briefing for an investment council.

TASK: Extract ALL investment-relevant facts, numbers, dates, and events from the research materials below 
about {ticker} ({company_name}). This is a FACTUAL EXTRACTION — do NOT analyze, interpret, or recommend. 
Your job is to surface every material fact so the investment analysts can form their own opinions.

CRITICAL RULE: Only include information that is SPECIFICALLY about {ticker} ({company_name}). 
The search results may contain articles about OTHER companies with similar names or in the same sector. 
If a data point is about a DIFFERENT company, EXCLUDE IT. Verify every fact relates to {ticker}.

SOURCE HIERARCHY RULE:
This Intelligence Brief is web/news context, not the source of truth for hard financial data.
Extract what the sources say, but label it as sourced web context. Do not present web article
numbers as more authoritative than structured provider data such as FMP, SEC EDGAR, Massive,
yfinance, Finnhub, or broker/account data. If a source is a snippet, paywalled page, redirected
search result, commentary site, or undated result, keep confidence modest and preserve the source name.

COMPANY: {company_name}
TICKER: {ticker}
SECTOR: {sector}
DATE: {current_date}

MACRO ENVIRONMENT CONTEXT:
{macro_brief}

RESEARCH MATERIALS:
{articles_text}

OUTPUT FORMAT — follow this structure exactly:

INTELLIGENCE BRIEF: {ticker} ({company_name}) — {current_date}

MACRO ENVIRONMENT:
• [Extract 3-4 key macro facts that could impact {ticker}: rates, inflation, GDP, geopolitical events, market sentiment]

SECTOR DYNAMICS:
• [Extract 2-4 sector-specific facts: regulatory changes, supply chain, competition, industry trends]
• [Include specific numbers, dates, and names where available]

COMPANY CATALYSTS (recent):
• [Extract ALL recent company-specific events: earnings results, guidance, product launches, partnerships, management changes]
• [Include specific numbers: revenue figures, EPS, growth rates, margins, subscriber counts, contract values]
• [Include dates: when did these events happen? when are upcoming catalysts?]

BULL CASE (from sources):
• [Extract specific bullish arguments from the articles — with the source name]
• [Include analyst price targets, upgrade/downgrade details, specific metrics cited]

BEAR CASE (from sources):
• [Extract specific bearish arguments from the articles — with the source name]
• [Include risk factors, negative data points, analyst warnings]

NOTABLE CORPORATE ACTIONS:
• [M&A, buybacks, share offerings, insider transactions, new fund launches, executive changes]
• [If none found, write "None identified in recent coverage"]

KEY NUMBERS EXTRACTED:
• [List every specific financial metric mentioned: revenue, EPS, P/E, margins, growth rates, price targets]

SOURCES:
{sources_list}

RULES:
- Extract FACTS, not opinions. Do not add your own analysis.
- Include EVERY specific number, date, and name from the articles.
- If two sources conflict, include both and note the discrepancy.
- Do not resolve conflicts against structured provider data; leave that to the council/CIO.
- Mark web-only facts as source-reported context, not verified source-of-truth data.
- Do not fabricate or infer data that isn't in the source material.
- Each bullet should contain a concrete fact with a specific number or date where possible.
"""

            raw = ChatGPTBackendClient(timeout=90).chat(prompt)
            brief = raw.strip()
            logger.info(f"  📝 GPT extraction complete ({len(brief)} chars)")
            return brief

        except Exception as e:
            logger.error(f"GPT brief synthesis failed: {e}")
            logger.info("  🔄 Falling back to Gemini Flash for synthesis...")
            try:
                return self._synthesize_brief_gemini_fallback(ticker, stock_data, macro_brief, articles)
            except Exception as e2:
                logger.error(f"Gemini fallback also failed: {e2}")
                return self._create_fallback_brief(ticker, stock_data)

    def _synthesize_brief_gemini_fallback(self, ticker: str, stock_data: Dict, macro_brief: str, articles: List[Dict]) -> str:
        """Fallback synthesis using Gemini Flash."""
        if not articles:
            return self._create_insufficient_research_brief(ticker, stock_data)

        try:
            profile = stock_data.get('profile', {}) or {}
            company_name = profile.get('companyName', ticker) or ticker
            sector = profile.get('sector', 'Unknown') or 'Unknown'
            current_date = datetime.now().strftime("%Y-%m-%d")

            # Build article sections with same detail as GPT prompt
            article_sections = []
            sources = []
            for article in articles:
                domain = urlparse(article['url']).netloc
                article_sections.append(
                    f"--- SOURCE: {article['title']} ({domain}) ---\n"
                    f"{article['content'][:1500]}"
                )
                sources.append(f"- {article['title']} ({domain})")
            sources_list = '\n'.join(sources)
            articles_text = '\n\n'.join(article_sections)

            prompt = f"""You are a research analyst preparing a factual intelligence briefing for an investment council.

TASK: Extract ALL investment-relevant facts, numbers, dates, and events from the research materials below
about {ticker} ({company_name}). This is a FACTUAL EXTRACTION — do NOT analyze, interpret, or recommend.
Your job is to surface every material fact so the investment analysts can form their own opinions.

CRITICAL RULE: Only include information that is SPECIFICALLY about {ticker} ({company_name}).
If a data point is about a DIFFERENT company, EXCLUDE IT.

SOURCE HIERARCHY RULE:
This Intelligence Brief is web/news context, not the source of truth for hard financial data.
Do not present web article numbers as more authoritative than structured provider data such as
FMP, SEC EDGAR, Massive, yfinance, Finnhub, or broker/account data.

COMPANY: {company_name}
TICKER: {ticker}
SECTOR: {sector}
DATE: {current_date}

MACRO ENVIRONMENT CONTEXT:
{macro_brief}

RESEARCH MATERIALS:
{articles_text}

OUTPUT FORMAT — follow this structure exactly:

INTELLIGENCE BRIEF: {ticker} ({company_name}) — {current_date}

MACRO ENVIRONMENT:
• [3-4 key macro facts that could impact {ticker}]

SECTOR DYNAMICS:
• [2-4 sector-specific facts with numbers and dates]

COMPANY CATALYSTS (recent):
• [ALL recent company events: earnings, guidance, launches, partnerships]
• [Include specific numbers and dates]

BULL CASE (from sources):
• [Bullish arguments from articles with source names]

BEAR CASE (from sources):
• [Bearish arguments from articles with source names]

NOTABLE CORPORATE ACTIONS:
• [M&A, buybacks, insider transactions, executive changes]

KEY NUMBERS EXTRACTED:
• [Every specific financial metric mentioned]

SOURCES:
{sources_list}

RULES:
- Extract FACTS, not opinions. Do not add your own analysis.
- Include EVERY specific number, date, and name from the articles.
- Mark web-only facts as source-reported context, not verified source-of-truth data.
- Do not fabricate data that isn't in the source material.
"""

            brief, _ = gemini_generate(prompt, model=Config.GEMINI_FLASH_MODEL, timeout=90)
            brief = brief.strip()
            logger.info(f"  📝 Gemini Flash extraction complete ({len(brief)} chars)")
            return brief
        except Exception as e:
            logger.error(f"Gemini fallback synthesis also failed: {e}")
            return self._create_fallback_brief(ticker, stock_data)

    def _synthesize_macro_brief(self, macro_articles: List[Dict], macro_data: Dict) -> str:
        """Extract macro environment facts using GPT."""
        if not macro_articles:
            return self._fallback_macro_brief(macro_data, insufficient_current_news=True)

        try:
            # Prepare macro article content — give GPT enough text
            macro_content = []
            for article in macro_articles[:5]:
                macro_content.append(
                    f"--- SOURCE: {article['title']} ({urlparse(article['url']).netloc}) ---\n"
                    f"{article['content'][:1000]}"
                )
            macro_text = '\n\n'.join(macro_content)

            # Extract FRED values
            def _extract_fred_value(data, key):
                series = data.get(key, {})
                if isinstance(series, dict):
                    obs = series.get('observations', [])
                    if obs and isinstance(obs, list):
                        return obs[-1].get('value', 'Unknown')
                return 'Unknown'

            fed_rate = _extract_fred_value(macro_data, 'fed_funds_rate')
            cpi = _extract_fred_value(macro_data, 'cpi')
            unemployment = _extract_fred_value(macro_data, 'unemployment')
            gdp = _extract_fred_value(macro_data, 'gdp')
            treasury_10y = _extract_fred_value(macro_data, 'treasury_10y')
            fear_greed = macro_data.get('fear_greed_crypto', {})
            fg_value = fear_greed.get('value', 'Unknown') if isinstance(fear_greed, dict) else 'Unknown'
            fg_label = fear_greed.get('label', '') if isinstance(fear_greed, dict) else ''

            prompt = f"""You are a research analyst extracting macro-economic facts for an investment council.

TASK: Extract ALL relevant macro-economic facts from the data and news below. 
This is FACTUAL EXTRACTION — do not analyze or recommend. Surface every material fact.

ECONOMIC DATA (from FRED):
- Federal Funds Rate: {fed_rate}%
- CPI Index: {cpi}
- Unemployment Rate: {unemployment}%
- GDP (Billions): ${gdp}
- 10-Year Treasury Yield: {treasury_10y}%
- Fear & Greed Index: {fg_value} ({fg_label})

NEWS ARTICLES:
{macro_text}

OUTPUT FORMAT:

MACRO ENVIRONMENT SUMMARY:

MONETARY POLICY:
• [Fed rate decisions, forward guidance, FOMC statements — with dates]

ECONOMIC INDICATORS:
• [GDP, jobs, unemployment, inflation — with specific numbers]

MARKET SENTIMENT:
• [Fear & Greed level, VIX, major index performance — with numbers]

GEOPOLITICAL / EXTERNAL RISKS:
• [Wars, trade policy, sanctions, oil prices, supply chain — with specifics]

KEY UPCOMING EVENTS:
• [FOMC meetings, jobs reports, CPI releases, earnings season dates]

RULES:
- Extract FACTS with specific numbers and dates. No opinions.
- Include every concrete data point from the articles.
- If articles mention specific market moves (S&P down X%, oil at $Y), include them.
"""

            raw = ChatGPTBackendClient(timeout=90).chat(prompt)
            brief = raw.strip()
            logger.info(f"  📝 GPT macro extraction complete ({len(brief)} chars)")
            return brief

        except Exception as e:
            logger.error(f"GPT macro synthesis failed: {e}")
            logger.info("  🔄 Falling back to Gemini Flash for macro synthesis...")
            try:
                macro_text = '\n\n'.join(
                    f"--- {a['title']} ---\n{a['content'][:1000]}" for a in macro_articles[:5]
                )
                prompt = f"""Extract ALL macro-economic facts from these articles. Facts only, no analysis.
Include: Fed rates, inflation, GDP, jobs, market sentiment, geopolitical risks, upcoming events.
Include specific numbers and dates.

{macro_text}"""
                brief, _ = gemini_generate(prompt, model=Config.GEMINI_FLASH_MODEL, timeout=90)
                brief = brief.strip()
                logger.info(f"  📝 Gemini Flash macro extraction complete ({len(brief)} chars)")
                return brief
            except Exception as e2:
                logger.error(f"Gemini macro fallback also failed: {e2}")
                return self._fallback_macro_brief(macro_data)

    def _fallback_macro_brief(self, macro_data: Dict, insufficient_current_news: bool = False) -> str:
        """Basic macro summary when current-news synthesis is unavailable."""
        def _extract_fred_value(data, key):
            series = data.get(key, {})
            if isinstance(series, dict):
                obs = series.get('observations', [])
                if obs and isinstance(obs, list):
                    return obs[-1].get('value', 'Unknown')
            return 'Unknown'

        fed_rate = _extract_fred_value(macro_data, 'fed_funds_rate')
        unemployment = _extract_fred_value(macro_data, 'unemployment')

        status = (
            "CURRENT NEWS STATUS: Insufficient current web/news sources were available. "
            "Use FRED/market data only; do not infer current catalysts."
            if insufficient_current_news
            else "CURRENT NEWS STATUS: Research fallback mode."
        )

        return f"""MACRO ENVIRONMENT SUMMARY:
{status}
Federal Funds Rate at {fed_rate}%. Unemployment at {unemployment}%. 
Market conditions require careful monitoring. Check recent news for geopolitical developments 
and Fed policy changes that could impact equity valuations."""


    def _create_insufficient_research_brief(self, ticker: str, stock_data: Dict) -> str:
        """Create an explicit no-current-web-data brief."""
        profile = stock_data.get('profile', {})
        company_name = profile.get('companyName', ticker)
        current_date = datetime.now().strftime("%Y-%m-%d")

        return f"""INTELLIGENCE BRIEF: {ticker} ({company_name}) — {current_date}

RESEARCH STATUS: INSUFFICIENT_CURRENT_WEB_DATA

Current web/news enrichment returned no usable sources. This is a data-quality failure, not a neutral finding.

RULE FOR COUNCIL:
• Do not create a buy recommendation from this brief alone.
• A buy-side action is allowed only if fundamental, technical, portfolio, and source-backed data outside this brief independently support it.
• If current-news/business-thesis evidence is material to the decision, mark the ticker DEFER/WATCH and state that current web data was insufficient.

SOURCES:
• None from current-web search."""


    def _create_fallback_brief(self, ticker: str, stock_data: Dict) -> str:
        """Create a basic brief when research fails."""
        profile = stock_data.get('profile', {})
        company_name = profile.get('companyName', ticker)
        current_date = datetime.now().strftime("%Y-%m-%d")

        return f"""INTELLIGENCE BRIEF: {ticker} ({company_name}) — {current_date}

RESEARCH STATUS: INSUFFICIENT_CURRENT_WEB_DATA

Research Desk failed before producing usable current-web sources. This is a data-quality failure, not a neutral finding.

RULE FOR COUNCIL:
• Do not invent bull/bear catalysts.
• Do not open a new buy-side position from this brief.
• Use DEFER/WATCH unless independent non-web data is strong enough and the report explicitly explains the source basis.

SOURCES:
• None from current-web research."""
