"""Configuration and API key management."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path, override=True)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_pct(name: str, default_pct: float) -> float:
    """Read a percentage env var and return it as a fraction.

    Accepts either percent form (``10``) or fraction form (``0.10``); any
    value > 1 is treated as a percentage.
    """
    value = _env_float(name, default_pct)
    return value / 100.0 if value > 1 else value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class Config:
    """Centralized configuration for all API keys and settings."""

    # Data API Keys
    FMP_API_KEY: str = os.getenv("FMP_API_KEY", "")
    MASSIVE_API_KEY: str = os.getenv("MASSIVE_API_KEY", "") or os.getenv("POLYGON_API_KEY", "")
    FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "")
    BENZINGA_API_KEY: str = os.getenv("BENZINGA_API_KEY", "") or os.getenv("ARTHA_BENZINGA_API_KEY", "")
    COINGECKO_API_KEY: str = os.getenv("COINGECKO_API_KEY", "")
    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")

    # AI Model API Keys
    # GPT-backed analysts use the ChatGPT backend through Codex OAuth.
    # 2026-07-09: upgraded gpt-5.5 -> gpt-5.6 (Sol flagship). On the Codex/ChatGPT
    # backend the plain "gpt-5.6" alias is NOT wired ("model is not supported when
    # using Codex with a ChatGPT account") — the working id is "gpt-5.6-sol".
    # Verified live: effort "max" (new tier above xhigh) is accepted AND echoed
    # back applied; "ultra" is NOT a reasoning.effort value (multi-agent beta) and
    # is rejected with invalid_value. Fallback is gpt-5.6-terra (verified live;
    # gpt-5.6-luna 404s on this backend).
    # xhigh is the default hard-problems setting; deployments can override it.
    GPT_MODEL: str = os.getenv("ARTHA_GPT_MODEL", "gpt-5.6-sol")
    GPT_FALLBACK_MODEL: str = os.getenv("ARTHA_GPT_FALLBACK_MODEL", "gpt-5.6-terra")
    GPT_REASONING_EFFORT: str = os.getenv("ARTHA_GPT_REASONING_EFFORT", "xhigh")
    # Strategy overhaul Wave 2: decision LLMs must be near-deterministic.
    # The old default of 2.0 made every council verdict sampling noise.
    GPT_TEMPERATURE: float = _env_float("ARTHA_GPT_TEMPERATURE", 0.2)
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "") or GOOGLE_API_KEY
    GEMINI_TECHNICAL_MODEL: str = os.getenv("ARTHA_GEMINI_TECHNICAL_MODEL", "gemini-3.1-pro-preview")
    GEMINI_FLASH_MODEL: str = os.getenv("ARTHA_GEMINI_FLASH_MODEL", "gemini-3.5-flash")
    GEMINI_THINKING_LEVEL: str = os.getenv("ARTHA_GEMINI_THINKING_LEVEL", "high")
    # Wave 2: match GPT — low temperature for reproducible analyst verdicts.
    GEMINI_TEMPERATURE: float = _env_float("ARTHA_GEMINI_TEMPERATURE", 0.3)
    CODEX_AUTH_PATH: str = os.getenv("CODEX_AUTH_PATH", "~/.codex/auth.json")

    # Research Desk API Keys
    BRAVE_SEARCH_API_KEY: str = (
        os.getenv("BRAVE_SEARCH_API_KEY", "")
        or os.getenv("BRAVE_API_KEY", "")
    )
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")  # Deprecated; kept only for env visibility.

    # Telegram Delivery
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    OPENCLAW_TMP_DIR: str = os.getenv(
        "ARTHA_OPENCLAW_TMP_DIR",
        str(Path.home() / ".openclaw" / "workspace" / "tmp"),
    )

    # API Base URLs
    FMP_BASE_URL: str = "https://financialmodelingprep.com/stable"
    MASSIVE_BASE_URL: str = os.getenv("ARTHA_MASSIVE_BASE_URL", "https://api.massive.com")
    FINNHUB_BASE_URL: str = "https://finnhub.io/api/v1"
    BENZINGA_BASE_URL: str = os.getenv("ARTHA_BENZINGA_BASE_URL", "https://api.benzinga.com/api/v2")
    COINGECKO_BASE_URL: str = "https://api.coingecko.com/api/v3"
    ALPHA_VANTAGE_BASE_URL: str = "https://www.alphavantage.co/query"
    FRED_BASE_URL: str = "https://api.stlouisfed.org/fred"
    SEC_SUBMISSIONS_BASE_URL: str = "https://data.sec.gov/submissions"
    SEC_COMPANYFACTS_BASE_URL: str = "https://data.sec.gov/api/xbrl/companyfacts"
    SEC_TICKER_MAP_URL: str = "https://www.sec.gov/files/company_tickers.json"

    # Production data plumbing
    FMP_API_PLAN: str = os.getenv("ARTHA_FMP_API_PLAN", "premium")
    FMP_CALLS_PER_MINUTE: int = int(os.getenv("ARTHA_FMP_CALLS_PER_MINUTE", "240"))
    FMP_CACHE_TTL_SECONDS: int = int(os.getenv("ARTHA_FMP_CACHE_TTL_SECONDS", "900"))
    FMP_429_RETRIES: int = int(os.getenv("ARTHA_FMP_429_RETRIES", "3"))
    FMP_429_BACKOFF_SECONDS: float = float(os.getenv("ARTHA_FMP_429_BACKOFF_SECONDS", "4"))
    FMP_SHORT_INTEREST_ENDPOINT: str = os.getenv("ARTHA_FMP_SHORT_INTEREST_ENDPOINT", "")
    FMP_ENABLE_QUARTERLY_ESTIMATES: bool = os.getenv(
        "ARTHA_FMP_ENABLE_QUARTERLY_ESTIMATES",
        "false",
    ).lower() in ("1", "true", "yes")
    FRED_CALLS_PER_MINUTE: int = int(os.getenv("ARTHA_FRED_CALLS_PER_MINUTE", "45"))
    FRED_CACHE_TTL_SECONDS: int = int(os.getenv("ARTHA_FRED_CACHE_TTL_SECONDS", "3600"))
    MASSIVE_ENABLED: bool = os.getenv(
        "ARTHA_MASSIVE_ENABLED",
        "true",
    ).lower() in ("1", "true", "yes")
    MASSIVE_SNAPSHOT_ENABLED: bool = os.getenv(
        "ARTHA_MASSIVE_SNAPSHOT_ENABLED",
        "false",
    ).lower() in ("1", "true", "yes")
    MASSIVE_CALLS_PER_MINUTE: int = int(os.getenv("ARTHA_MASSIVE_CALLS_PER_MINUTE", "5"))
    MASSIVE_CACHE_TTL_SECONDS: int = int(os.getenv("ARTHA_MASSIVE_CACHE_TTL_SECONDS", "900"))
    MASSIVE_HISTORY_MODE: str = os.getenv("ARTHA_MASSIVE_HISTORY_MODE", "fallback").lower()
    BENZINGA_NEWS_ENABLED: bool = os.getenv(
        "ARTHA_BENZINGA_NEWS_ENABLED",
        "false",
    ).lower() in ("1", "true", "yes")
    BENZINGA_CALLS_PER_MINUTE: int = int(os.getenv("ARTHA_BENZINGA_CALLS_PER_MINUTE", "30"))
    BENZINGA_NEWS_LOOKBACK_HOURS: int = int(os.getenv("ARTHA_BENZINGA_NEWS_LOOKBACK_HOURS", "6"))
    # Per-headline news sentiment (Finnhub company-news + Gemini flash GOOD/BAD/
    # UNKNOWN classification; Lopez-Lira & Tang, SSRN 4412788). Feeds the buy
    # council's contrarian_sentiment component and the rule-4.7 negative-news
    # REVIEW_EXIT flag in the sell path.
    NEWS_SENTIMENT_ENABLED: bool = os.getenv(
        "ARTHA_NEWS_SENTIMENT_ENABLED",
        "true",
    ).lower() in ("1", "true", "yes")
    NEWS_SENTIMENT_DAYS: int = _env_int("ARTHA_NEWS_SENTIMENT_DAYS", 3)
    NEWS_SENTIMENT_MAX_HEADLINES: int = _env_int("ARTHA_NEWS_SENTIMENT_MAX_HEADLINES", 15)
    # Sell-side exit-review trigger: daily score <= NEG_EXIT_SCORE with at least
    # NEG_EXIT_MIN_SCORED scored (good+bad) headlines. Negative news carries
    # ~2.5x the signal of positive (Lopez-Lira & Tang), hence exit > entry use.
    NEWS_SENTIMENT_NEG_EXIT_SCORE: float = _env_float("ARTHA_NEWS_SENTIMENT_NEG_EXIT_SCORE", -0.5)
    NEWS_SENTIMENT_NEG_EXIT_MIN_SCORED: int = _env_int("ARTHA_NEWS_SENTIMENT_NEG_EXIT_MIN_SCORED", 3)
    PRICE_HISTORY_MODE: str = os.getenv("ARTHA_PRICE_HISTORY_MODE", "fmp_primary").lower()
    YFINANCE_BATCH_SIZE: int = int(os.getenv("ARTHA_YFINANCE_BATCH_SIZE", "100"))
    YFINANCE_DOWNLOAD_TIMEOUT_SECONDS: int = int(os.getenv("ARTHA_YFINANCE_DOWNLOAD_TIMEOUT_SECONDS", "20"))
    YFINANCE_RANK_TOTAL_TIMEOUT_SECONDS: int = int(
        os.getenv("ARTHA_YFINANCE_RANK_TOTAL_TIMEOUT_SECONDS", "420")
    )
    RANK_MIN_HISTORY_COVERAGE_PCT: float = _env_float(
        "ARTHA_RANK_MIN_HISTORY_COVERAGE_PCT", 0.90
    )
    RANK_COVERAGE_AUDIT_ENABLED: bool = os.getenv(
        "ARTHA_RANK_COVERAGE_AUDIT_ENABLED",
        "true",
    ).lower() in ("1", "true", "yes")
    SCAN_REQUIRE_MIN_RANK_COVERAGE: bool = os.getenv(
        "ARTHA_SCAN_REQUIRE_MIN_RANK_COVERAGE",
        "true",
    ).lower() in ("1", "true", "yes")
    YFINANCE_THREADS: bool = os.getenv(
        "ARTHA_YFINANCE_THREADS",
        "false",
    ).lower() in ("1", "true", "yes")
    FUNNEL_RANK_TOP_N: int = int(os.getenv("ARTHA_FUNNEL_RANK_TOP_N", "150"))
    FUNNEL_ENRICH_MAX: int = int(os.getenv("ARTHA_FUNNEL_ENRICH_MAX", "45"))
    FUNNEL_ENRICH_PROVIDER_TIMEOUT_SECONDS: int = int(
        os.getenv("ARTHA_FUNNEL_ENRICH_PROVIDER_TIMEOUT_SECONDS", "6")
    )
    FUNNEL_ENRICH_PROVIDER_RETRIES: int = int(os.getenv("ARTHA_FUNNEL_ENRICH_PROVIDER_RETRIES", "0"))
    FUNNEL_ENRICH_TOTAL_TIMEOUT_SECONDS: int = int(
        os.getenv("ARTHA_FUNNEL_ENRICH_TOTAL_TIMEOUT_SECONDS", "120")
    )
    SEC_USER_AGENT: str = os.getenv(
        "SEC_USER_AGENT",
        "ArthaPersonalResearch/1.0 (set SEC_USER_AGENT for contact)",
    )
    SEC_CACHE_TTL_SECONDS: int = int(os.getenv("ARTHA_SEC_CACHE_TTL_SECONDS", "86400"))

    # Bounded agentic diligence layer.
    # This keeps each council analyst investigative without allowing unbounded
    # web loops, runaway latency, or uncited claims.
    AGENTIC_COUNCIL_ENABLED: bool = os.getenv(
        "ARTHA_AGENTIC_COUNCIL_ENABLED",
        "true",
    ).lower() in ("1", "true", "yes")
    AGENTIC_WEB_RESEARCH_ENABLED: bool = os.getenv(
        "ARTHA_AGENTIC_WEB_RESEARCH_ENABLED",
        "true",
    ).lower() in ("1", "true", "yes")
    AGENTIC_WEB_QUERIES_PER_ROLE: int = int(os.getenv("ARTHA_AGENTIC_WEB_QUERIES_PER_ROLE", "2"))
    AGENTIC_WEB_RESULTS_PER_QUERY: int = int(os.getenv("ARTHA_AGENTIC_WEB_RESULTS_PER_QUERY", "2"))
    AGENTIC_MAX_EVIDENCE_ITEMS: int = int(os.getenv("ARTHA_AGENTIC_MAX_EVIDENCE_ITEMS", "80"))

    # Portfolio settings. Deployments should override these examples in .env.
    MONTHLY_BUDGET: float = float(os.getenv("ARTHA_MONTHLY_BUDGET", "350"))
    MAX_SINGLE_STOCK_PCT: float = 0.15  # 15% max per stock
    MAX_SINGLE_CRYPTO_PCT: float = 0.10  # 10% max per crypto
    STOP_LOSS_PCT: float = -0.15  # -15% stop loss
    TAKE_PROFIT_PCT: float = 0.25  # +25% take profit trigger

    # Robinhood Agentic pilot execution guardrails.
    # These defaults make Artha ready to rehearse orders, but incapable of
    # live trading until Robinhood MCP is connected and the kill switch is
    # explicitly disabled.
    ROBINHOOD_MCP_URL: str = os.getenv(
        "ARTHA_ROBINHOOD_MCP_URL",
        "https://agent.robinhood.com/mcp/trading",
    )
    ROBINHOOD_AGENTIC_ACCOUNT_NUMBER: str = os.getenv("ARTHA_ROBINHOOD_AGENTIC_ACCOUNT_NUMBER", "")
    ROBINHOOD_EXPECTED_ACCOUNT_TYPE: str = os.getenv("ARTHA_ROBINHOOD_EXPECTED_ACCOUNT_TYPE", "cash").lower()
    ROBINHOOD_EXPECTED_ACCOUNT_NICKNAME: str = os.getenv("ARTHA_ROBINHOOD_EXPECTED_ACCOUNT_NICKNAME", "Agentic")
    ROBINHOOD_REVIEW_ONLY: bool = os.getenv(
        "ARTHA_ROBINHOOD_REVIEW_ONLY",
        "true",
    ).lower() in ("1", "true", "yes")
    ROBINHOOD_AGENTIC_ENABLED: bool = os.getenv(
        "ARTHA_ROBINHOOD_AGENTIC_ENABLED",
        "false",
    ).lower() in ("1", "true", "yes")
    ROBINHOOD_DRY_RUN_ONLY: bool = os.getenv(
        "ARTHA_ROBINHOOD_DRY_RUN_ONLY",
        "true",
    ).lower() in ("1", "true", "yes")
    ROBINHOOD_KILL_SWITCH: bool = os.getenv(
        "ARTHA_ROBINHOOD_KILL_SWITCH",
        "true",
    ).lower() in ("1", "true", "yes")
    ROBINHOOD_PILOT_MAX_ACCOUNT_VALUE: float = float(
        os.getenv("ARTHA_ROBINHOOD_PILOT_MAX_ACCOUNT_VALUE", "350")
    )
    ROBINHOOD_AUTO_BOOK_CASH_TRANSFERS: bool = os.getenv(
        "ARTHA_ROBINHOOD_AUTO_BOOK_CASH_TRANSFERS",
        "false",
    ).lower() in ("1", "true", "yes")
    ROBINHOOD_MAX_POSITION_DOLLARS: float = float(
        os.getenv("ARTHA_ROBINHOOD_MAX_POSITION_DOLLARS", "50")
    )
    ROBINHOOD_MAX_TRADES_PER_DAY: int = int(os.getenv("ARTHA_ROBINHOOD_MAX_TRADES_PER_DAY", "5"))
    # Day-scoped buy capacity limits pause NEW BUYS until the next trading
    # day (buys only — never sells, services, or crons; see stand_down.py).
    ARTHA_AUTO_PAUSE_BUYS_ON_LIMIT: bool = os.getenv(
        "ARTHA_AUTO_PAUSE_BUYS_ON_LIMIT",
        "true",
    ).lower() in ("1", "true", "yes")
    ROBINHOOD_MIN_PRICE: float = float(os.getenv("ARTHA_ROBINHOOD_MIN_PRICE", "5"))
    ROBINHOOD_MIN_DOLLAR_VOLUME: float = float(
        os.getenv("ARTHA_ROBINHOOD_MIN_DOLLAR_VOLUME", "10000000")
    )
    ROBINHOOD_MAX_SPREAD_PCT: float = float(os.getenv("ARTHA_ROBINHOOD_MAX_SPREAD_PCT", "0.01"))
    ROBINHOOD_ALLOWED_ORDER_TYPES: tuple[str, ...] = tuple(
        part.strip().lower()
        for part in os.getenv("ARTHA_ROBINHOOD_ALLOWED_ORDER_TYPES", "limit,market").split(",")
        if part.strip()
    )
    ROBINHOOD_MARKET_ORDER_MAX_PRICE_DRIFT_PCT: float = float(
        os.getenv("ARTHA_ROBINHOOD_MARKET_ORDER_MAX_PRICE_DRIFT_PCT", "0.005")
    )
    ROBINHOOD_ORDER_TIF: str = os.getenv("ARTHA_ROBINHOOD_ORDER_TIF", "day").lower()
    ROBINHOOD_ALLOW_AFTER_HOURS: bool = os.getenv(
        "ARTHA_ROBINHOOD_ALLOW_AFTER_HOURS",
        "false",
    ).lower() in ("1", "true", "yes")
    ROBINHOOD_REQUIRE_SUPERVISOR_PASS_FOR_BUYS: bool = os.getenv(
        "ARTHA_ROBINHOOD_REQUIRE_SUPERVISOR_PASS_FOR_BUYS",
        "true",
    ).lower() in ("1", "true", "yes")
    ROBINHOOD_MIN_BUY_EVIDENCE_ITEMS: int = int(
        os.getenv("ARTHA_ROBINHOOD_MIN_BUY_EVIDENCE_ITEMS", "10")
    )
    ROBINHOOD_RECONCILIATION_ENABLED: bool = os.getenv(
        "ARTHA_ROBINHOOD_RECONCILIATION_ENABLED",
        "true",
    ).lower() in ("1", "true", "yes")
    ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE: str = os.getenv(
        "ARTHA_ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE",
        str(_env_path.parent / "data" / "robinhood" / "latest_snapshot.json"),
    )
    ROBINHOOD_CONTROL_FILE: str = os.getenv(
        "ARTHA_ROBINHOOD_CONTROL_FILE",
        str(_env_path.parent / "data" / "robinhood" / "control.json"),
    )
    ROBINHOOD_RECONCILIATION_SNAPSHOT_MAX_AGE_MINUTES: int = int(
        os.getenv("ARTHA_ROBINHOOD_RECONCILIATION_SNAPSHOT_MAX_AGE_MINUTES", "10")
    )
    ROBINHOOD_WARNING_STATE_FILE: str = os.getenv(
        "ARTHA_ROBINHOOD_WARNING_STATE_FILE",
        str(_env_path.parent / "data" / "robinhood" / "warning_state.json"),
    )
    ROBINHOOD_STALE_SNAPSHOT_TELEGRAM_MIN_MINUTES: int = int(
        os.getenv("ARTHA_ROBINHOOD_STALE_SNAPSHOT_TELEGRAM_MIN_MINUTES", "30")
    )
    ROBINHOOD_SYNC_BRIDGE_COMMAND: str = os.getenv("ARTHA_ROBINHOOD_SYNC_BRIDGE_COMMAND", "")
    ROBINHOOD_SYNC_BRIDGE_TIMEOUT_SECONDS: int = int(
        os.getenv("ARTHA_ROBINHOOD_SYNC_BRIDGE_TIMEOUT_SECONDS", "20")
    )
    ROBINHOOD_ACTION_TOKEN_TTL_MINUTES: int = int(
        os.getenv("ARTHA_ROBINHOOD_ACTION_TOKEN_TTL_MINUTES", "60")
    )
    ROBINHOOD_REVIEW_DECISION_MAX_AGE_MINUTES: int = int(
        os.getenv("ARTHA_ROBINHOOD_REVIEW_DECISION_MAX_AGE_MINUTES", "60")
    )
    ROBINHOOD_REVIEW_MAX_AGE_MINUTES: int = int(
        os.getenv("ARTHA_ROBINHOOD_REVIEW_MAX_AGE_MINUTES", "10")
    )
    ROBINHOOD_REQUIRE_FRESH_SNAPSHOT_FOR_REVIEW: bool = os.getenv(
        "ARTHA_ROBINHOOD_REQUIRE_FRESH_SNAPSHOT_FOR_REVIEW",
        "true",
    ).lower() in ("1", "true", "yes")
    ROBINHOOD_MIN_CASH_BUFFER_DOLLARS: float = float(
        os.getenv("ARTHA_ROBINHOOD_MIN_CASH_BUFFER_DOLLARS", "1")
    )
    ROBINHOOD_BLOCK_DUPLICATE_OPEN_ORDERS: bool = os.getenv(
        "ARTHA_ROBINHOOD_BLOCK_DUPLICATE_OPEN_ORDERS",
        "true",
    ).lower() in ("1", "true", "yes")
    EXECUTION_OFFICER_ENABLED: bool = os.getenv(
        "ARTHA_EXECUTION_OFFICER_ENABLED",
        "true",
    ).lower() in ("1", "true", "yes")
    EXECUTION_OFFICER_LLM_ENABLED: bool = os.getenv(
        "ARTHA_EXECUTION_OFFICER_LLM_ENABLED",
        "true",
    ).lower() in ("1", "true", "yes")
    EXECUTION_OFFICER_MODEL: str = os.getenv("ARTHA_EXECUTION_OFFICER_MODEL", GPT_MODEL)
    EXECUTION_OFFICER_REASONING_EFFORT: str = os.getenv(
        "ARTHA_EXECUTION_OFFICER_REASONING_EFFORT",
        GPT_REASONING_EFFORT,
    )
    EXECUTION_OFFICER_TEMPERATURE: float = _env_float(
        "ARTHA_EXECUTION_OFFICER_TEMPERATURE",
        GPT_TEMPERATURE,
    )
    EXECUTION_OFFICER_TIMEOUT_SECONDS: int = int(
        os.getenv("ARTHA_EXECUTION_OFFICER_TIMEOUT_SECONDS", "90")
    )
    EXECUTION_OFFICER_AGENTIC_ENABLED: bool = os.getenv(
        "ARTHA_EXECUTION_OFFICER_AGENTIC_ENABLED",
        "true",
    ).lower() in ("1", "true", "yes")
    EXECUTION_OFFICER_AGENTIC_MAX_TOOL_STEPS: int = int(
        os.getenv("ARTHA_EXECUTION_OFFICER_AGENTIC_MAX_TOOL_STEPS", "8")
    )
    ROBINHOOD_AUTO_BUY_ENABLED: bool = os.getenv(
        "ARTHA_ROBINHOOD_AUTO_BUY_ENABLED",
        "false",
    ).lower() in ("1", "true", "yes")
    ROBINHOOD_AUTO_BUY_MAX_ORDER_DOLLARS: float = float(
        os.getenv("ARTHA_ROBINHOOD_AUTO_BUY_MAX_ORDER_DOLLARS", "25")
    )
    ROBINHOOD_AUTO_BUY_MAX_DAILY_DOLLARS: float = float(
        os.getenv("ARTHA_ROBINHOOD_AUTO_BUY_MAX_DAILY_DOLLARS", "50")
    )
    ROBINHOOD_AUTO_BUY_MIN_SCORE: int = int(os.getenv("ARTHA_ROBINHOOD_AUTO_BUY_MIN_SCORE", "70"))
    ROBINHOOD_AUTO_BUY_MIN_CONFIDENCE: int = int(
        os.getenv("ARTHA_ROBINHOOD_AUTO_BUY_MIN_CONFIDENCE", "6")
    )
    ROBINHOOD_AUTO_BUY_ALLOWED_VERDICTS: tuple[str, ...] = tuple(
        part.strip().upper()
        for part in os.getenv("ARTHA_ROBINHOOD_AUTO_BUY_ALLOWED_VERDICTS", "STARTER,TACTICAL_BUY,BUY,ACCUMULATE").split(",")
        if part.strip()
    )
    ROBINHOOD_AUTO_BUY_NO_CHASE_PCT: float = float(
        os.getenv("ARTHA_ROBINHOOD_AUTO_BUY_NO_CHASE_PCT", "0.02")
    )
    ROBINHOOD_AUTO_BUY_MIN_WHOLE_SHARE_FILL_RATIO: float = float(
        os.getenv("ARTHA_ROBINHOOD_AUTO_BUY_MIN_WHOLE_SHARE_FILL_RATIO", "0.75")
    )
    # Unattended sell drain (strategy overhaul Wave 2). Stop-triggered exits,
    # URGENT_EXIT verdicts, and 2-day-confirmed EXITs enqueue as auto_sell
    # actions served by the same OpenClaw runner protocol as auto_buy.
    ROBINHOOD_AUTO_SELL_ENABLED: bool = os.getenv(
        "ARTHA_ROBINHOOD_AUTO_SELL_ENABLED",
        "false",
    ).lower() in ("1", "true", "yes")
    # Sell triggers that are exempt from the "bid fell below reference" drift
    # block. A falling bid is WHY these sells exist; reference re-anchors to
    # the fresh quote at execution time.
    ROBINHOOD_SELL_DRIFT_EXEMPT_TRIGGERS: tuple[str, ...] = tuple(
        part.strip().lower()
        for part in os.getenv(
            "ARTHA_ROBINHOOD_SELL_DRIFT_EXEMPT_TRIGGERS",
            "trailing_stop,hard_stop,urgent_exit",
        ).split(",")
        if part.strip()
    )
    # Fat-finger floor for drift-exempt sells: refuse only when the live bid
    # is below this fraction of Artha's reference price.
    ROBINHOOD_SELL_DRIFT_FLOOR_RATIO: float = _env_float(
        "ARTHA_ROBINHOOD_SELL_DRIFT_FLOOR_RATIO", 0.25
    )
    # Transient queue failures (stale decision/snapshot) auto-regenerate a
    # fresh review this many times per market day instead of blocking forever.
    ROBINHOOD_QUEUE_REGEN_PER_DAY: int = _env_int(
        "ARTHA_ROBINHOOD_QUEUE_REGEN_PER_DAY", 1
    )
    # Supervisor checks that actually gate broker execution. Failures in the
    # remaining checks (recent_logs, latest_report, telegram, ...) are
    # reporting noise and must not freeze the trade queue.
    SUPERVISOR_EXECUTION_RELEVANT_CHECKS: tuple[str, ...] = tuple(
        part.strip().lower()
        for part in os.getenv(
            "ARTHA_SUPERVISOR_EXECUTION_RELEVANT_CHECKS",
            "database,broker_reconciliation,execution_readiness,position_monitoring",
        ).split(",")
        if part.strip()
    )

    # DEFER/WATCH trigger automation.
    # A triggered entry watch can auto-refresh research and re-run the council.
    # It may prepare a Robinhood review request, but live order placement remains
    # governed by the Robinhood safety switches above.
    DEFER_AUTO_REVIEW_ENABLED: bool = os.getenv(
        "ARTHA_DEFER_AUTO_REVIEW_ENABLED",
        "true",
    ).lower() in ("1", "true", "yes")
    DEFER_AUTO_REVIEW_MAX_PER_CYCLE: int = int(
        os.getenv("ARTHA_DEFER_AUTO_REVIEW_MAX_PER_CYCLE", "1")
    )
    DEFER_AUTO_REVIEW_PREPARE_ROBINHOOD_REVIEW: bool = os.getenv(
        "ARTHA_DEFER_AUTO_REVIEW_PREPARE_ROBINHOOD_REVIEW",
        "true",
    ).lower() in ("1", "true", "yes")
    DEFER_AUTO_REVIEW_LEGACY_TRIGGER_LOOKBACK_HOURS: int = int(
        os.getenv("ARTHA_DEFER_AUTO_REVIEW_LEGACY_TRIGGER_LOOKBACK_HOURS", "24")
    )
    DEFER_AUTO_REVIEW_STALE_REVIEW_MINUTES: int = int(
        os.getenv("ARTHA_DEFER_AUTO_REVIEW_STALE_REVIEW_MINUTES", "120")
    )
    DEFER_AUTO_REVIEW_FAILED_RETRY_MINUTES: int = int(
        os.getenv("ARTHA_DEFER_AUTO_REVIEW_FAILED_RETRY_MINUTES", "60")
    )
    DEFER_AUTO_REVIEW_MAX_FAILURES: int = int(
        os.getenv("ARTHA_DEFER_AUTO_REVIEW_MAX_FAILURES", "3")
    )
    DEFER_AUTO_REVIEW_BUY_VERDICTS: tuple[str, ...] = tuple(
        part.strip().upper()
        for part in os.getenv(
            "ARTHA_DEFER_AUTO_REVIEW_BUY_VERDICTS",
            "BUY,STARTER,TACTICAL_BUY,ACCUMULATE,ADD,STRONG BUY",
        ).split(",")
        if part.strip()
    )

    # Report Settings
    REPORT_TIMEZONE: str = "America/Chicago"

    # Sentinel Settings
    SENTINEL_ENABLED: bool = True
    SENTINEL_KEYWORD_ONLY: bool = False  # If True, skip GPT classification
    SENTINEL_SONNET_BATCH_MAX: int = 30  # Backward-compatible name; max headlines per GPT batch
    SENTINEL_ALERT_COOLDOWN_HOURS: int = 6  # Dedupe window for sentinel alerts

    # -------------------------------------------------------------------------
    # Crisis Mode v3 Settings
    # -------------------------------------------------------------------------

    # SPY drawdown thresholds (fraction from 52-week high)
    CRISIS_SPY_CORRECTION_THRESHOLD: float = -0.10   # -10% → CORRECTION
    CRISIS_SPY_BEAR_THRESHOLD: float = -0.20          # -20% → BEAR
    CRISIS_SPY_PANIC_THRESHOLD: float = -0.35         # -35% → PANIC

    # Hysteresis: days required to confirm state transition
    CRISIS_ACTIVATION_DAYS: int = 2    # Must be in new (worse) state for 2 trading days
    CRISIS_DEACTIVATION_DAYS: int = 5  # Must be back above threshold for 5 trading days

    # Reserve deployment limits
    CRISIS_RESERVE_MAX_DEPLOYMENT: float = 0.80       # Never deploy more than 80% of reserve

    # Portfolio phase thresholds
    PORTFOLIO_INCEPTION_THRESHOLD: float = 10_000.0   # Below = ETF-only during crisis
    PORTFOLIO_GROWTH_THRESHOLD: float = 25_000.0      # Above = full satellite protocol

    # Crisis position sizing
    CRISIS_MAX_INITIAL_STOCK_PCT: float = 0.03         # 3% initial crisis stock position
    CRISIS_MAX_TOTAL_STOCK_PCT: float = 0.06           # 6% max per stock in crisis
    CRISIS_ETF_MIN_ALLOCATION: float = 0.60            # Min 60% of crisis capital to ETFs

    # -------------------------------------------------------------------------
    # Liquidity Gate Thresholds (Phase 1.2)
    # -------------------------------------------------------------------------
    LIQUIDITY_MIN_MARKET_CAP: float = 1_000_000_000.0   # $1B minimum market cap
    LIQUIDITY_MIN_ADV: float = 10_000_000.0             # $10M avg daily value traded
    LIQUIDITY_MIN_PRICE: float = 5.0                    # $5 minimum price

    # Quality filter thresholds (all must pass for crisis stock eligibility)
    CRISIS_QUALITY_MIN_MARKET_CAP: float = 10_000_000_000.0  # $10B minimum
    CRISIS_QUALITY_MAX_DEBT_EQUITY: float = 1.5
    CRISIS_QUALITY_MIN_ROIC: float = 0.12               # 12% minimum ROIC/ROE
    CRISIS_QUALITY_MIN_INTEREST_COVERAGE: float = 5.0
    CRISIS_QUALITY_VALUATION_DISCOUNT: float = 0.15     # 15% below 5-year median multiple

    # Council Convergence Score thresholds
    CRISIS_CCS_TRIPLE_CROWN: int = 10     # ≥10 = TRIPLE_CROWN (2x allocation)
    CRISIS_CCS_HIGH_CONVICTION: int = 7   # ≥7  = HIGH_CONVICTION (1.5x)
    CRISIS_CCS_STANDARD: int = 5          # ≥5  = STANDARD
    CRISIS_MIN_BUY_COUNT: int = 3         # 3/3 analysts must agree (BUY) during crisis

    # Hard kill switch: if fingerprint dominant type flips >2 times in 10 days
    CRISIS_FINGERPRINT_FLIP_LIMIT: int = 2
    CRISIS_FINGERPRINT_FLIP_WINDOW_DAYS: int = 10

    # Budget multipliers per state (applied to MONTHLY_BUDGET; currently $0 while satellite is paused)
    CRISIS_BUDGET_CORRECTION_MULT: float = 1.0    # No change during correction
    CRISIS_BUDGET_BEAR_MULT: float = 1.20          # $600 during bear
    CRISIS_BUDGET_PANIC_MULT: float = 1.40         # $700 during panic

    # Reserve deployment schedule
    CRISIS_BEAR_MONTH1_PCT: float = 0.20           # 20% of reserve in month 1 (bear)
    CRISIS_BEAR_MONTHLY_PCT: float = 0.15          # 15% per month thereafter (bear)
    CRISIS_PANIC_MONTH1_PCT: float = 0.30          # 30% of reserve in month 1 (panic)
    CRISIS_PANIC_MONTHLY_PCT: float = 0.15         # 15% per month thereafter (panic)

    # Panic-mode stock buy spacing compression (days between buys)
    CRISIS_BEAR_BUY_SPACING_DAYS: int = 3          # Standard spacing in bear
    CRISIS_PANIC_BUY_SPACING_DAYS: int = 2         # Compressed spacing in panic

    # Sectors excluded from crisis stock-picking
    CRISIS_EXCLUDED_SECTORS: tuple = ("Financial Services", "Insurance", "Utilities")

    # -------------------------------------------------------------------------
    # v2 Decision Engine — Action Classes & Thresholds
    # -------------------------------------------------------------------------

    VERDICT_TYPES: list = [
        "BUY",          # Full conviction position (score 75+)
        "STARTER",      # Small initial position, thesis not fully confirmed (score 65-74)
        "TACTICAL_BUY", # Regime/opportunity-driven swing trade (score 55-64 in favorable regime)
        "ACCUMULATE",   # Long-term quality, add on dips
        "ADD",          # Pyramid into existing winning position
        "HOLD",         # Maintain current position
        "WATCH",        # Monitor, not ready to act
        "DEFER",        # Good asset, bad timing
        "TRIM",         # Reduce position size
        "SELL",         # Exit position
        "AVOID",        # Do not touch
    ]

    # Exploration Budget
    EXPLORATION_BUDGET_PCT: float = 0.15         # 15% of NAV reserved for exploration
    MAX_EXPLORATION_POSITIONS: int = 3
    EXPLORATION_MAX_PER_POSITION_PCT: float = 0.05  # 5% NAV per exploration position

    # Portfolio Constraints
    MAX_POSITION_PCT: float = 0.20               # Max 20% NAV in any single position
    MAX_SECTOR_PCT: float = 0.30                 # Max 30% in one sector
    MAX_INVESTED_PCT: float = 0.90               # Hard account-level exposure ceiling, rechecked before placement
    MAX_CONCURRENT_POSITIONS: int = 20           # Max 20 live positions

    # Regime-Adaptive Thresholds
    SCORE_THRESHOLD_BUY: int = 75
    SCORE_THRESHOLD_STARTER: int = 65
    SCORE_THRESHOLD_TACTICAL: int = 55
    # DEPRECATED (Wave 2): fear/greed no longer re-penalizes the score inside
    # score_to_action — regime already contributes 15/100 in buy scoring and the
    # mechanical regime gate (artha/regime_gate.py) governs entries. Kept only
    # so older code paths/tests import cleanly.
    REGIME_FEAR_BONUS: int = 5                   # DEPRECATED: old F&G score bonus
    REGIME_GREED_PENALTY: int = 10               # DEPRECATED: old F&G score penalty

    # -------------------------------------------------------------------------
    # Mechanical Regime Gate (strategy overhaul Wave 2) — env-overridable
    # Replaces fear/greed as a penalty system. RISK_OFF = SPY < 200d SMA for
    # 3+ consecutive sessions; HARD_RISK_OFF adds negative trailing 12-mo SPY
    # excess return vs T-bills (Faber 2007 timing model; Moskowitz-Ooi-Pedersen
    # 2012 time-series momentum). The gate NEVER blocks exits.
    # -------------------------------------------------------------------------
    REGIME_GATE_SMA_DAYS: int = _env_int("ARTHA_REGIME_GATE_SMA_DAYS", 200)
    REGIME_GATE_RISK_OFF_SESSIONS: int = _env_int("ARTHA_REGIME_GATE_RISK_OFF_SESSIONS", 3)
    REGIME_GATE_TBILL_FALLBACK_PCT: float = _env_float("ARTHA_REGIME_GATE_TBILL_FALLBACK_PCT", 4.5)
    REGIME_GATE_VOL_TARGET_VIX: float = _env_float("ARTHA_REGIME_GATE_VOL_TARGET_VIX", 20.0)
    REGIME_GATE_VOL_MULT_MIN: float = _env_float("ARTHA_REGIME_GATE_VOL_MULT_MIN", 0.5)
    REGIME_GATE_VOL_MULT_MAX: float = _env_float("ARTHA_REGIME_GATE_VOL_MULT_MAX", 1.0)
    # RISK_OFF entry requirements: TACTICAL needs a higher score and buys run half size.
    REGIME_GATE_RISK_OFF_MIN_TACTICAL_SCORE: int = _env_int(
        "ARTHA_REGIME_GATE_RISK_OFF_MIN_TACTICAL_SCORE", 65
    )
    REGIME_GATE_RISK_OFF_MIN_CONVICTION: int = _env_int(
        "ARTHA_REGIME_GATE_RISK_OFF_MIN_CONVICTION", 8
    )

    # Mechanical technical-setup scoring from funnel scan signals (Wave 2).
    # Weights cite: Jegadeesh-Titman 1993 (12-1 momentum), George-Hwang 2004
    # (52-week-high proximity), Minervini trend template / Zarattini-Antonacci
    # 2024 (stage-2 uptrend filter), Da-Gurun-Warachka 2014 (information
    # discreteness / frog-in-the-pan).
    BUY_TECH_TREND_TEMPLATE_POINTS: int = _env_int("ARTHA_BUY_TECH_TREND_TEMPLATE_POINTS", 10)
    BUY_TECH_52WH_PASS_POINTS: int = _env_int("ARTHA_BUY_TECH_52WH_PASS_POINTS", 5)
    BUY_TECH_52WH_STRONG_POINTS: int = _env_int("ARTHA_BUY_TECH_52WH_STRONG_POINTS", 8)
    BUY_TECH_MOMENTUM_Z_POINTS: int = _env_int("ARTHA_BUY_TECH_MOMENTUM_Z_POINTS", 5)
    BUY_TECH_ID_SMOOTH_POINTS: int = _env_int("ARTHA_BUY_TECH_ID_SMOOTH_POINTS", 2)
    BUY_TECH_LLM_ADJUST_MAX: int = _env_int("ARTHA_BUY_TECH_LLM_ADJUST_MAX", 5)

    # CIO structured decision block: stops/targets must be ATR-derived
    # (2.5x ATR14 initial stop, hard-capped at 10%; Han-Zhou-Zhu stop research).
    CIO_STOP_ATR_MULT: float = _env_float("ARTHA_CIO_STOP_ATR_MULT", 2.5)
    CIO_STOP_CAP_PCT: float = _env_pct("ARTHA_CIO_STOP_CAP_PCT", 10.0)

    # Mechanical CIO upgrade authority: composite is STRONG when the funnel z
    # is at/above this and the trend template passes (momentum/breakout track).
    MECHANICAL_OVERRIDE_MIN_Z: float = _env_float("ARTHA_MECHANICAL_OVERRIDE_MIN_Z", 1.5)

    # Optional Claude contrarian analyst (off by default). Enabling requires
    # `claude setup-token` and CLAUDE_CODE_OAUTH_TOKEN in
    # ~/.openclaw/workspace-personal/.env (see artha/claude_sdk.py).
    CLAUDE_ANALYST_ENABLED: bool = os.getenv(
        "ARTHA_CLAUDE_ANALYST_ENABLED",
        "false",
    ).lower() in ("1", "true", "yes")
    CLAUDE_ANALYST_MODEL: str = os.getenv("ARTHA_CLAUDE_ANALYST_MODEL", "claude-opus-4-6")
    CLAUDE_ANALYST_TIMEOUT_SECONDS: int = _env_int("ARTHA_CLAUDE_ANALYST_TIMEOUT_SECONDS", 120)
    BUY_CIO_ADJUSTMENT_MAX_POSITIVE: int = 15     # Evidence-backed CIO upside adjustment cap
    BUY_CIO_ADJUSTMENT_MAX_NEGATIVE: int = -15    # Evidence-backed CIO downside adjustment cap
    BUY_CIO_LOGIC_ADJUSTMENT_MAX_POSITIVE: int = 8
    BUY_CIO_LOGIC_ADJUSTMENT_MAX_NEGATIVE: int = -8
    BUY_CIO_RISK_OVERRIDE_MAX_NEGATIVE: int = -18
    BUY_CIO_DATA_DISPUTE_MAX_POSITIVE: int = 10
    BUY_CIO_DATA_DISPUTE_MAX_NEGATIVE: int = -10
    BUY_CIO_ADJUSTMENT_MIN_CONFIDENCE: int = 6

    # Cash Deployment Targets (% of NAV to deploy per scan)
    CASH_DEPLOY_EXTREME_FEAR: float = 0.30       # Deploy up to 30% in extreme fear
    CASH_DEPLOY_FEAR: float = 0.20               # Deploy up to 20% in fear
    CASH_DEPLOY_NEUTRAL: float = 0.15            # Deploy up to 15% in neutral
    CASH_DEPLOY_GREED: float = 0.05              # Deploy up to 5% in greed
    CASH_DEPLOY_EXTREME_GREED: float = 0.0       # No new deployment in extreme greed

    # Scheduled scan breadth
    SCHEDULED_SCAN_HOUR_CT: int = int(os.getenv("ARTHA_SCHEDULED_SCAN_HOUR_CT", "11"))
    SCHEDULED_SCAN_MINUTE_CT: int = int(os.getenv("ARTHA_SCHEDULED_SCAN_MINUTE_CT", "30"))
    SCHEDULED_SCAN_CATCHUP_MINUTES: int = int(os.getenv("ARTHA_SCHEDULED_SCAN_CATCHUP_MINUTES", "90"))
    DAILY_WARM_SCAN_HOUR_CT: int = int(os.getenv("ARTHA_DAILY_WARM_SCAN_HOUR_CT", "9"))
    DAILY_WARM_SCAN_MINUTE_CT: int = int(os.getenv("ARTHA_DAILY_WARM_SCAN_MINUTE_CT", "0"))
    DAILY_WARM_SCAN_CATCHUP_MINUTES: int = int(os.getenv("ARTHA_DAILY_WARM_SCAN_CATCHUP_MINUTES", "150"))
    SCAN_CANDIDATE_POOL: int = int(os.getenv("ARTHA_SCAN_CANDIDATE_POOL", "12"))
    SCAN_COUNCIL_MAX: int = int(os.getenv("ARTHA_SCAN_COUNCIL_MAX", "8"))
    SCAN_FALLBACK_MAX: int = int(os.getenv("ARTHA_SCAN_FALLBACK_MAX", "6"))
    SCAN_BROKER_ROUTER_ENABLED: bool = os.getenv(
        "ARTHA_SCAN_BROKER_ROUTER_ENABLED",
        "true",
    ).lower() in ("1", "true", "yes")
    SCAN_BROKER_ROUTER_POOL: int = int(os.getenv("ARTHA_SCAN_BROKER_ROUTER_POOL", "45"))
    SCAN_BROKER_ROUTER_MAX_QUOTE_CHECKS: int = int(
        os.getenv("ARTHA_SCAN_BROKER_ROUTER_MAX_QUOTE_CHECKS", "45")
    )
    SCAN_ROUTER_FILL_RESEARCH_SLOTS: bool = os.getenv(
        "ARTHA_SCAN_ROUTER_FILL_RESEARCH_SLOTS",
        "false",
    ).lower() in ("1", "true", "yes")
    SCAN_ROUTER_MAX_PRICE_SOURCE_DRIFT_PCT: float = float(
        os.getenv("ARTHA_SCAN_ROUTER_MAX_PRICE_SOURCE_DRIFT_PCT", "0.025")
    )
    SCAN_ROUTER_AVOID_COOLDOWN_DAYS: int = int(os.getenv("ARTHA_SCAN_ROUTER_AVOID_COOLDOWN_DAYS", "10"))
    SCAN_ROUTER_DEFER_COOLDOWN_DAYS: int = int(os.getenv("ARTHA_SCAN_ROUTER_DEFER_COOLDOWN_DAYS", "5"))
    SCAN_ROUTER_STARTER_COOLDOWN_DAYS: int = int(os.getenv("ARTHA_SCAN_ROUTER_STARTER_COOLDOWN_DAYS", "2"))
    SCAN_MIN_DEPLOYABLE_FOR_BUY_COUNCIL: float = float(
        os.getenv("ARTHA_SCAN_MIN_DEPLOYABLE_FOR_BUY_COUNCIL", "10")
    )
    # Afternoon opportunity pass: cheap movers-only funnel at 14:15 CT, max 3
    # council candidates, RISK_ON regime only (strategy overhaul Wave 2).
    AFTERNOON_SCAN_ENABLED: bool = os.getenv(
        "ARTHA_AFTERNOON_SCAN_ENABLED",
        "true",
    ).lower() in ("1", "true", "yes")
    AFTERNOON_SCAN_HOUR_CT: int = int(os.getenv("ARTHA_AFTERNOON_SCAN_HOUR_CT", "14"))
    AFTERNOON_SCAN_MINUTE_CT: int = int(os.getenv("ARTHA_AFTERNOON_SCAN_MINUTE_CT", "15"))
    AFTERNOON_SCAN_CATCHUP_MINUTES: int = int(os.getenv("ARTHA_AFTERNOON_SCAN_CATCHUP_MINUTES", "30"))
    AFTERNOON_SCAN_MAX_CANDIDATES: int = int(os.getenv("ARTHA_AFTERNOON_SCAN_MAX_CANDIDATES", "3"))
    ALPHA_SHADOW_SIGNALS_ENABLED: bool = os.getenv(
        "ARTHA_ALPHA_SHADOW_SIGNALS_ENABLED",
        "true",
    ).lower() in ("1", "true", "yes")
    # Regex-scraped limit prices must land within this fraction of the live
    # quote or the idea is rejected to an entry watch instead of an order.
    SCAN_LIMIT_PRICE_MAX_DEVIATION_PCT: float = _env_float(
        "ARTHA_SCAN_LIMIT_PRICE_MAX_DEVIATION_PCT", 0.20
    )
    # Market-hour sell council slots (CT, HH:MM) in addition to close+30.
    SELL_REVIEW_MARKET_SLOTS_CT: tuple[str, ...] = tuple(
        part.strip()
        for part in os.getenv("ARTHA_SELL_REVIEW_MARKET_SLOTS_CT", "10:30,14:00").split(",")
        if part.strip()
    )
    # A defer-watch zone more than this % away from the live price is stale
    # (splits, huge gaps) and needs a decide-now council, not a zone-buy review.
    DEFER_WATCH_STALE_ZONE_PCT: float = _env_float(
        "ARTHA_DEFER_WATCH_STALE_ZONE_PCT", 30.0
    )
    OPPORTUNITY_SCOUT_ENABLED: bool = os.getenv(
        "ARTHA_OPPORTUNITY_SCOUT_ENABLED",
        "true",
    ).lower() in ("1", "true", "yes")
    OPPORTUNITY_SCOUT_LLM_ENABLED: bool = os.getenv(
        "ARTHA_OPPORTUNITY_SCOUT_LLM_ENABLED",
        "true",
    ).lower() in ("1", "true", "yes")
    OPPORTUNITY_SCOUT_MODEL: str = os.getenv("ARTHA_OPPORTUNITY_SCOUT_MODEL", GPT_MODEL)
    OPPORTUNITY_SCOUT_REASONING_EFFORT: str = os.getenv(
        "ARTHA_OPPORTUNITY_SCOUT_REASONING_EFFORT",
        GPT_REASONING_EFFORT,
    )
    OPPORTUNITY_SCOUT_TEMPERATURE: float = _env_float(
        "ARTHA_OPPORTUNITY_SCOUT_TEMPERATURE",
        GPT_TEMPERATURE,
    )
    OPPORTUNITY_SCOUT_TIMEOUT_SECONDS: int = int(
        os.getenv("ARTHA_OPPORTUNITY_SCOUT_TIMEOUT_SECONDS", "120")
    )
    OPPORTUNITY_SCOUT_MAX_TOOL_STEPS: int = int(
        os.getenv("ARTHA_OPPORTUNITY_SCOUT_MAX_TOOL_STEPS", "8")
    )
    OPPORTUNITY_SCOUT_CANDIDATE_LIMIT: int = int(
        os.getenv("ARTHA_OPPORTUNITY_SCOUT_CANDIDATE_LIMIT", "40")
    )
    OPPORTUNITY_SCOUT_MAX_BATCHES: int = int(os.getenv("ARTHA_OPPORTUNITY_SCOUT_MAX_BATCHES", "5"))
    OPPORTUNITY_SCOUT_BATCH_SIZE: int = int(os.getenv("ARTHA_OPPORTUNITY_SCOUT_BATCH_SIZE", "8"))
    OPPORTUNITY_SCOUT_RESERVE_QUALITY_SLOTS: int = int(
        os.getenv("ARTHA_OPPORTUNITY_SCOUT_RESERVE_QUALITY_SLOTS", "2")
    )
    OPPORTUNITY_SCOUT_MAX_WEB_TOOL_CALLS: int = int(
        os.getenv("ARTHA_OPPORTUNITY_SCOUT_MAX_WEB_TOOL_CALLS", "4")
    )
    OPPORTUNITY_SCOUT_WEB_RESULTS_PER_CALL: int = int(
        os.getenv("ARTHA_OPPORTUNITY_SCOUT_WEB_RESULTS_PER_CALL", "3")
    )
    SCAN_DEFER_WATCH_SKIP_ENABLED: bool = os.getenv(
        "ARTHA_SCAN_DEFER_WATCH_SKIP_ENABLED",
        "true",
    ).lower() in ("1", "true", "yes")
    SCAN_DEFER_WATCH_SKIP_BUFFER_PCT: float = float(
        os.getenv("ARTHA_SCAN_DEFER_WATCH_SKIP_BUFFER_PCT", "5")
    )
    SCAN_DEFER_WATCH_SKIP_MAJOR_MOVE_PCT: float = float(
        os.getenv("ARTHA_SCAN_DEFER_WATCH_SKIP_MAJOR_MOVE_PCT", "5")
    )
    SCAN_DEFER_SKIP_BACKFILL_EXTRA: int = int(
        os.getenv("ARTHA_SCAN_DEFER_SKIP_BACKFILL_EXTRA", "6")
    )
    SCAN_PREPARE_ROBINHOOD_REVIEW_FOR_BUYS: bool = os.getenv(
        "ARTHA_SCAN_PREPARE_ROBINHOOD_REVIEW_FOR_BUYS",
        "true",
    ).lower() in ("1", "true", "yes")
    FUNNEL_MAX_CANDIDATES_PER_SECTOR: int = int(os.getenv("ARTHA_FUNNEL_MAX_CANDIDATES_PER_SECTOR", "3"))
    FUNNEL_MIN_ENTRY_QUALITY_CANDIDATES: int = int(
        os.getenv("ARTHA_FUNNEL_MIN_ENTRY_QUALITY_CANDIDATES", "3")
    )
    FUNNEL_ENTRY_QUALITY_MIN_SCORE: float = float(
        os.getenv("ARTHA_FUNNEL_ENTRY_QUALITY_MIN_SCORE", "16")
    )
    FUNNEL_PARALLEL_DISCOVERY_ENABLED: bool = os.getenv(
        "ARTHA_FUNNEL_PARALLEL_DISCOVERY_ENABLED",
        "true",
    ).lower() in ("1", "true", "yes")
    FUNNEL_PARALLEL_DISCOVERY_MAX: int = int(
        os.getenv("ARTHA_FUNNEL_PARALLEL_DISCOVERY_MAX", "12")
    )

    # Accuracy/self-review tracking
    # This marks the first day of the GPT-backed, SEC/FMP-enriched, agentic council.
    # Older Opus-era scores stay visible as legacy history, but they should not
    # trigger active prompt-tuning alerts for the current council.
    ACCURACY_CURRENT_ERA_START: str = os.getenv(
        "ARTHA_ACCURACY_CURRENT_ERA_START",
        "2026-06-02T00:00:00+00:00",
    )
    ACCURACY_CURRENT_COUNCIL_VERSION: str = os.getenv(
        "ARTHA_ACCURACY_CURRENT_COUNCIL_VERSION",
        "agentic-gpt-gemini-gpt-2026-06-02",
    )
    ACCURACY_MIN_PATTERN_SAMPLES: int = int(os.getenv("ARTHA_ACCURACY_MIN_PATTERN_SAMPLES", "3"))

    # -------------------------------------------------------------------------
    # Sell-Side Engine Settings
    # -------------------------------------------------------------------------

    # Hard stops per position type (fraction from avg_cost)
    SELL_HARD_STOP_BUY: float = -0.25           # -25% for full BUY conviction
    SELL_HARD_STOP_STARTER: float = -0.20       # -20% for STARTER positions
    SELL_HARD_STOP_TACTICAL: float = -0.12      # -12% for TACTICAL_BUY swing trades
    SELL_HARD_STOP_ACCUMULATE: float = -0.30    # -30% for long-term ACCUMULATE
    SELL_HARD_STOP_LEGACY: float = -0.15        # -15% legacy fallback (matches old STOP_LOSS_PCT)

    # Periodic review frequencies (days)
    SELL_REVIEW_DAYS_TACTICAL: int = 7          # TACTICAL_BUY reviewed weekly
    SELL_REVIEW_DAYS_STARTER: int = 21          # STARTER reviewed every 3 weeks
    SELL_REVIEW_DAYS_BUY: int = 30              # BUY reviewed monthly
    SELL_REVIEW_DAYS_ACCUMULATE: int = 45       # ACCUMULATE very patient review

    # Pending thesis expiry (days)
    SELL_THESIS_PENDING_EXPIRY_DAYS: int = 7    # Pending thesis expires after 7 days

    # Minimum hold periods (days) before sell council can recommend EXIT
    SELL_MIN_HOLD_TACTICAL: int = 3             # 3 days min for TACTICAL_BUY
    SELL_MIN_HOLD_STARTER: int = 14             # 14 days min for STARTER
    SELL_MIN_HOLD_BUY: int = 30                 # 30 days min for BUY
    SELL_MIN_HOLD_ACCUMULATE: int = 60          # 60 days min for ACCUMULATE

    # Time decay limits (days — auto-review when position held this long without action)
    SELL_TIME_DECAY_TACTICAL: int = 45          # TACTICAL_BUY: 45 days max without review
    SELL_TIME_DECAY_STARTER: int = 90
    SELL_TIME_DECAY_BUY: int = 180
    SELL_TIME_DECAY_ACCUMULATE: int = 365

    # Sell score thresholds for action mapping (0-100 scale)
    SELL_SCORE_EXIT_CONVICTION: int = 80        # BUY/ACCUMULATE: need 80+ to EXIT
    SELL_SCORE_EXIT_TACTICAL: int = 70          # TACTICAL_BUY: need 70+ to EXIT
    SELL_SCORE_EXIT_STARTER: int = 75           # STARTER: need 75+ to EXIT
    SELL_SCORE_TRIM_THRESHOLD: int = 55         # 55+ to TRIM any position
    SELL_SCORE_ROTATE_THRESHOLD: int = 60       # 60+ to ROTATE (combined with opportunity delta)

    # Sell score bonuses/adjustments
    SELL_SCORE_CIO_CONVICTION_ADJUST: int = -5  # CIO applies -5 for conviction positions (harder bar)
    SELL_SCORE_REGIME_CHANGE_BONUS: int = 10    # +10 for TACTICAL_BUY on regime change
    SELL_SCORE_THESIS_TRIGGERED_BONUS: int = 20 # +20 when invalidation condition TRIGGERED
    SELL_CIO_ADJUSTMENT_MAX_POSITIVE: int = 20  # CIO can add bounded evidence-backed sell pressure
    SELL_CIO_ADJUSTMENT_MAX_NEGATIVE: int = -10 # CIO can reduce sell pressure only modestly
    SELL_CIO_ADJUSTMENT_MIN_CONFIDENCE: int = 6 # Require adequate CIO confidence for non-zero adjustment

    # Cooldown periods (days)
    SELL_COOLDOWN_AFTER_TRIM: int = 7           # No trim/exit within 7 days of trim
    SELL_COOLDOWN_AFTER_BUY: int = 14           # No sell within 14 days of buy for BUY positions

    # Confirmation gate (days an EXIT must persist before firing for non-urgent)
    SELL_CONFIRMATION_DAYS: int = 2             # Non-urgent EXIT must persist 2 days
    SELL_PREPARE_ROBINHOOD_REVIEW: bool = os.getenv(
        "ARTHA_SELL_PREPARE_ROBINHOOD_REVIEW",
        "true",
    ).lower() in ("1", "true", "yes")
    SELL_ESCALATE_HIGH_NEWS_TO_LLM: bool = os.getenv(
        "ARTHA_SELL_ESCALATE_HIGH_NEWS_TO_LLM",
        "true",
    ).lower() in ("1", "true", "yes")
    SELL_HIGH_NEWS_SEMANTIC_MIN_CONFIDENCE: float = float(
        os.getenv("ARTHA_SELL_HIGH_NEWS_SEMANTIC_MIN_CONFIDENCE", "0.70")
    )

    # Trailing stop parameters — LEGACY (superseded by the exit-engine v2
    # constants below; kept only so old code paths import cleanly)
    SELL_TRAILING_STOP_ATR_MULT: float = 2.0    # DEPRECATED: old tight trail (whipsaw-prone)
    SELL_TRAILING_STOP_MIN_PCT: float = 0.08    # DEPRECATED: old 8%-from-high floor

    # -------------------------------------------------------------------------
    # Exit Engine v2 (strategy overhaul, rulebook 4.1-4.8) — env-overridable
    # -------------------------------------------------------------------------

    # Rule 4.1 — initial stop: entry -8% or entry - 2.5×ATR14, whichever is
    # tighter, hard-capped at 10% (Han-Zhou-Zhu: 10% stop doubled momentum Sharpe).
    SELL_INITIAL_STOP_PCT: float = _env_pct("ARTHA_INITIAL_STOP_PCT", 8.0)
    SELL_INITIAL_STOP_ATR_MULT: float = _env_float("ARTHA_INITIAL_STOP_ATR_MULT", 2.5)
    SELL_INITIAL_STOP_CAP_PCT: float = _env_pct("ARTHA_INITIAL_STOP_CAP_PCT", 10.0)

    # Rule 4.4 — cushion-gated wide trail (Chandelier + EMA-close exit).
    # Trail activates only after +2R or +20%; distance from high is
    # max(3.0×ATR22, 10%) — NEVER tighter than 2.5×ATR or 10%.
    SELL_TRAIL_ATR_MULT: float = _env_float("ARTHA_TRAIL_ATR_MULT", 3.0)
    SELL_TRAIL_MIN_PCT: float = _env_pct("ARTHA_TRAIL_MIN_PCT", 10.0)
    SELL_TRAIL_CUSHION_R_MULT: float = _env_float("ARTHA_TRAIL_CUSHION_R_MULT", 2.0)
    SELL_TRAIL_CUSHION_PCT: float = _env_pct("ARTHA_TRAIL_CUSHION_PCT", 20.0)
    SELL_CHANDELIER_LOOKBACK_DAYS: int = _env_int("ARTHA_CHANDELIER_LOOKBACK_DAYS", 22)
    SELL_TRAIL_EMA_PERIOD: int = _env_int("ARTHA_TRAIL_EMA_PERIOD", 20)
    SELL_TRAIL_EMA_PERIOD_FAST: int = _env_int("ARTHA_TRAIL_EMA_PERIOD_FAST", 10)
    SELL_TRAIL_HIGH_ADR_PCT: float = _env_pct("ARTHA_TRAIL_HIGH_ADR_PCT", 6.0)

    # Rule 4.3 — profit-taking into strength (+20-25% taking >3 weeks),
    # with the fast-move exception (+20% within 15 trading days → 8-week hold,
    # disaster stop + trail only).
    SELL_PROFIT_TAKE_MIN_GAIN_PCT: float = _env_pct("ARTHA_PROFIT_TAKE_MIN_GAIN_PCT", 20.0)
    SELL_PROFIT_TAKE_MIN_TRADING_DAYS: int = _env_int("ARTHA_PROFIT_TAKE_MIN_TRADING_DAYS", 15)
    SELL_FAST_MOVE_TRADING_DAYS: int = _env_int("ARTHA_FAST_MOVE_TRADING_DAYS", 15)
    SELL_FAST_MOVE_HOLD_TRADING_DAYS: int = _env_int("ARTHA_FAST_MOVE_HOLD_TRADING_DAYS", 40)

    # Rule 4.5 — time stops (dead money / catalyst positions)
    SELL_DEAD_MONEY_TRADING_DAYS: int = _env_int("ARTHA_DEAD_MONEY_TRADING_DAYS", 20)
    SELL_DEAD_MONEY_MIN_PNL_PCT: float = -_env_pct("ARTHA_DEAD_MONEY_MIN_PNL_PCT", 3.0)
    SELL_DEAD_MONEY_MAX_PNL_PCT: float = _env_pct("ARTHA_DEAD_MONEY_MAX_PNL_PCT", 5.0)
    SELL_CATALYST_TIME_STOP_TRADING_DAYS: int = _env_int("ARTHA_CATALYST_TIME_STOP_TRADING_DAYS", 15)

    # Rule 4.6 — earnings risk: scheduled earnings ahead + cushion < +10% →
    # TRIM/EXIT sized so a -15% gap costs <= 1R.
    SELL_EARNINGS_RISK_LOOKAHEAD_DAYS: int = _env_int("ARTHA_EARNINGS_RISK_LOOKAHEAD_DAYS", 7)
    SELL_EARNINGS_MIN_CUSHION_PCT: float = _env_pct("ARTHA_EARNINGS_MIN_CUSHION_PCT", 10.0)
    SELL_EARNINGS_GAP_RISK_PCT: float = _env_pct("ARTHA_EARNINGS_GAP_RISK_PCT", 15.0)

    # Rule 4.7 — thesis-break: FY1 consensus cut >= 1% over 4 weeks → exit
    # signal for catalyst positions.
    SELL_FY1_CUT_EXIT_THRESHOLD_PCT: float = _env_float("ARTHA_FY1_CUT_EXIT_THRESHOLD_PCT", 1.0)
    SELL_FY1_CUT_WINDOW_DAYS: int = _env_int("ARTHA_FY1_CUT_WINDOW_DAYS", 28)

    # Monitor alerting — CRITICAL stop-breach alerts repeat every 4h until the
    # position is closed or the stop is cleared.
    SELL_STOP_BREACH_REPEAT_HOURS: int = _env_int("ARTHA_STOP_BREACH_REPEAT_HOURS", 4)

    # Scale-out schedules per position type (gain_pct -> trim_pct_of_position)
    # e.g., at +40% gain, trim 15% of the position
    SELL_SCALE_OUT_BUY: dict = {
        "+40%": 0.15,
        "+75%": 0.20,
        "+100%": 0.25,
    }
    SELL_SCALE_OUT_TACTICAL: dict = {
        "+15%": 0.25,
        "+25%": 0.25,
        "+35%": 0.50,
    }
    SELL_SCALE_OUT_STARTER: dict = {
        "+30%": 0.15,
        "+60%": 0.20,
    }

    # Regime integration sell score adjustments
    SELL_REGIME_MISMATCH_PENALTY: int = 5       # +5 sell score when regime changed since entry
    SELL_REGIME_MISMATCH_TACTICAL_BONUS: int = 15  # +15 for TACTICAL_BUY after 3+ day regime change

    # Portfolio-level circuit breakers
    SELL_MAX_EXITS_PER_DAY: int = 5             # Align full-exit capacity with the five-trades/day pilot limit
    SELL_PORTFOLIO_LOSS_PAUSE_PCT: float = -0.10  # Pause non-urgent signals if portfolio -10% today

    # Opportunity cost scanner settings
    SELL_ROTATE_MIN_DELTA: int = 20             # Minimum score delta to flag rotation
    SELL_CONVICTION_LOCK_MIN_HEALTH: int = 70   # Healthy positions (health>=70) can't be rotation targets
    SELL_CONVICTION_LOCK_MAX_DAYS: int = 180    # Conviction lock only applies within 180 days

    # Post-sell shadow tracking
    SELL_SHADOW_TRACKING_DAYS: list = [5, 20, 60]   # Track price at these intervals
    SELL_SHADOW_TRACKING_WINDOW: int = 60            # Stop tracking after 60 days
    SELL_LEARNING_MIN_COMPLETED: int = _env_int(
        "ARTHA_SELL_LEARNING_MIN_COMPLETED", 20
    )  # Mature 60-day outcomes required before Council sees sell calibration

    @classmethod
    def validate(cls) -> list[str]:
        """Check which API keys are missing. Returns list of missing key names."""
        missing = []
        required = {
            "FMP_API_KEY": cls.FMP_API_KEY,
            "FINNHUB_API_KEY": cls.FINNHUB_API_KEY,
            "FRED_API_KEY": cls.FRED_API_KEY,
        }
        optional = {
            "COINGECKO_API_KEY": cls.COINGECKO_API_KEY,
            "GOOGLE_API_KEY": cls.GOOGLE_API_KEY,
            "GEMINI_API_KEY": cls.GEMINI_API_KEY,
        }
        for name, value in required.items():
            if not value:
                missing.append(f"REQUIRED: {name}")
        for name, value in optional.items():
            if not value:
                missing.append(f"OPTIONAL: {name}")
        return missing
