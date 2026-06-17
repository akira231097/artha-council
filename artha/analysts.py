"""Analyst implementations — runs each model independently on collected data.

Each analyst gets their own model, their own prompt, and produces
an independent assessment. No cross-contamination between analysts.

Model assignments:
  - Fundamental Analyst: GPT 5.5 via ChatGPT backend (deep reasoning, conservative)
  - Technical Analyst: Gemini (pattern recognition, speed)
  - Contrarian Analyst: GPT 5.5 via ChatGPT backend API (independent risk perspective)
"""
import json
import logging
from typing import Optional

from .config import Config
from .chatgpt_backend import ChatGPTBackendClient
from .gemini_client import gemini_generate
from .prompts import FUNDAMENTAL_ANALYST, TECHNICAL_ANALYST, CONTRARIAN_ANALYST

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Serialization
# ---------------------------------------------------------------------------

def _serialize_data(data: dict) -> str:
    """Convert collected data dict to a readable string for the LLM.
    
    Truncates very large fields to keep within context limits.
    """
    def _truncate(obj, max_items: int = 10, max_str_len: int = 500):
        if isinstance(obj, list):
            truncated = obj[:max_items]
            if len(obj) > max_items:
                truncated.append(f"... ({len(obj) - max_items} more items)")
            return [_truncate(item, max_items, max_str_len) for item in truncated]
        elif isinstance(obj, dict):
            return {k: _truncate(v, max_items, max_str_len) for k, v in obj.items()}
        elif isinstance(obj, str) and len(obj) > max_str_len:
            return obj[:max_str_len] + "..."
        return obj

    cleaned = _truncate(data)
    return json.dumps(cleaned, indent=2, default=str)


def _extract_relevant_fundamental_data(data: dict) -> dict:
    """Extract fundamental-relevant fields for the fundamental analyst.

    Includes: financials, valuation, analyst consensus, earnings calendar,
    price targets, news, and price history for full context.
    """
    return {
        "quote": data.get("quote"),
        "profile": data.get("profile"),
        "income_statement": data.get("income_statement"),
        "balance_sheet": data.get("balance_sheet"),
        "cash_flow": data.get("cash_flow"),
        "ratios": data.get("ratios"),
        "ratios_ttm": data.get("ratios_ttm"),
        "key_metrics": data.get("key_metrics"),
        "key_metrics_ttm": data.get("key_metrics_ttm"),
        "dcf": data.get("dcf"),
        "price_target_consensus": data.get("price_target_consensus"),
        "analyst_estimates": data.get("analyst_estimates"),
        "recommendation_trends": data.get("recommendation_trends"),
        "yf_quote": data.get("yf_quote"),
        "massive_quote": data.get("massive_quote"),
        "price_history_source": data.get("price_history_source"),
        "history_provider_checks": data.get("history_provider_checks"),
        "analyst_recs": data.get("analyst_recs"),
        "earnings_surprises": data.get("earnings_surprises"),
        "earnings_context": data.get("earnings_context"),
        "insider_finnhub": data.get("insider_finnhub"),
        "short_interest": data.get("short_interest"),
        "sec": data.get("sec"),
        "data_quality_report": data.get("data_quality_report") or data.get("data_quality"),
        "news": data.get("news"),
        "benzinga_news": data.get("benzinga_news"),
        "price_history": data.get("price_history"),
        "fmp_price_history_available": bool(data.get("fmp_price_history")),
        "yf_price_history_available": bool(data.get("yf_price_history")),
        "massive_price_history_available": bool(data.get("massive_price_history")),
    }


def _extract_relevant_technical_data(data: dict) -> dict:
    """Extract technical/sentiment-relevant fields.

    Includes: price history, computed technicals, news, sentiment,
    earnings calendar (affects volatility patterns), and analyst price
    targets (act as psychological support/resistance levels).
    """
    return {
        "quote": data.get("quote"),
        "yf_quote": data.get("yf_quote"),
        "massive_quote": data.get("massive_quote"),
        "price_history": data.get("price_history"),
        "price_history_source": data.get("price_history_source"),
        "history_provider_checks": data.get("history_provider_checks"),
        "fmp_price_history_available": bool(data.get("fmp_price_history")),
        "yf_price_history_available": bool(data.get("yf_price_history")),
        "massive_price_history_available": bool(data.get("massive_price_history")),
        "technicals": data.get("technicals"),  # Locally computed RSI, MACD, SMA, BB
        "av_macd": data.get("av_macd"),        # Legacy (now computed locally)
        "av_rsi": data.get("av_rsi"),          # Legacy (now computed locally)
        "news": data.get("news"),
        "benzinga_news": data.get("benzinga_news"),
        "finnhub_sentiment": data.get("finnhub_sentiment"),
        "finnhub_news": data.get("finnhub_news"),
        "earnings_context": data.get("earnings_context"),
        "price_target_consensus": data.get("price_target_consensus"),
        "analyst_estimates": data.get("analyst_estimates"),
        "recommendation_trends": data.get("recommendation_trends"),
        "short_interest": data.get("short_interest"),
        "data_quality_report": data.get("data_quality_report") or data.get("data_quality"),
    }


def _extract_relevant_risk_data(data: dict) -> dict:
    """Extract risk-relevant fields for the contrarian analyst.

    Includes: financials, insider activity, analyst consensus, news,
    earnings calendar (binary event risk), price targets (overvaluation
    risk), technicals (overbought/death cross signals), and price
    history (drawdown/extension risk).
    """
    return {
        "quote": data.get("quote"),
        "profile": data.get("profile"),
        "income_statement": data.get("income_statement"),
        "balance_sheet": data.get("balance_sheet"),
        "cash_flow": data.get("cash_flow"),
        "ratios": data.get("ratios"),
        "ratios_ttm": data.get("ratios_ttm"),
        "key_metrics": data.get("key_metrics"),
        "key_metrics_ttm": data.get("key_metrics_ttm"),
        "insider_finnhub": data.get("insider_finnhub"),
        "analyst_recs": data.get("analyst_recs"),
        "earnings_surprises": data.get("earnings_surprises"),
        "earnings_context": data.get("earnings_context"),
        "price_target_consensus": data.get("price_target_consensus"),
        "analyst_estimates": data.get("analyst_estimates"),
        "recommendation_trends": data.get("recommendation_trends"),
        "short_interest": data.get("short_interest"),
        "sec": data.get("sec"),
        "news": data.get("news"),
        "benzinga_news": data.get("benzinga_news"),
        "finnhub_news": data.get("finnhub_news"),
        "yf_quote": data.get("yf_quote"),
        "massive_quote": data.get("massive_quote"),
        "technicals": data.get("technicals"),
        "price_history": data.get("price_history"),
        "price_history_source": data.get("price_history_source"),
        "history_provider_checks": data.get("history_provider_checks"),
        "fmp_price_history_available": bool(data.get("fmp_price_history")),
        "yf_price_history_available": bool(data.get("yf_price_history")),
        "massive_price_history_available": bool(data.get("massive_price_history")),
        "data_quality_report": data.get("data_quality_report") or data.get("data_quality"),
    }


# ---------------------------------------------------------------------------
# Analyst Runners
# ---------------------------------------------------------------------------

def run_fundamental_analyst(
    stock_data: dict,
    macro_data: dict | None = None,
    context_header: str = "",
    intelligence_brief: str = "",
    pre_brief: str = "",
    momentum_context: str = "",
) -> Optional[str]:
    """Run the Fundamental Analyst (GPT 5.5 via ChatGPT backend).

    Deep-value, Warren Buffett-style analysis.
    """
    relevant_data = _extract_relevant_fundamental_data(stock_data)
    if macro_data:
        relevant_data["macro"] = macro_data

    data_str = _serialize_data(relevant_data)
    prompt = FUNDAMENTAL_ANALYST.format(
        data=data_str,
        context_header=context_header,
        intelligence_brief=intelligence_brief,
        pre_brief=pre_brief or "No recent events recorded.",
        momentum_context=momentum_context or "No momentum history available.",
    )

    try:
        text = ChatGPTBackendClient(timeout=90).chat(prompt)
        return text
    except Exception as e:
        logger.error(f"Fundamental analyst failed: {e}")
        return None


def run_technical_analyst(
    stock_data: dict,
    market_overview: dict | None = None,
    macro_data: dict | None = None,
    context_header: str = "",
    intelligence_brief: str = "",
    pre_brief: str = "",
    momentum_context: str = "",
) -> Optional[str]:
    """Run the Technical + Sentiment Analyst (Gemini).

    Pattern recognition + sentiment analysis. Uses Gemini for its
    strength in data pattern analysis and speed.
    """
    if not Config.GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not set — cannot run technical analyst")
        return None

    relevant_data = _extract_relevant_technical_data(stock_data)
    if market_overview:
        relevant_data["market_overview"] = {
            "sp500": market_overview.get("sp500"),
            "fear_greed": market_overview.get("fear_greed"),
            "vix": market_overview.get("vix"),
            "nasdaq": market_overview.get("nasdaq"),
            "top_gainers": (market_overview.get("gainers") or market_overview.get("top_gainers", []))[:5],
            "top_losers": (market_overview.get("losers") or market_overview.get("top_losers", []))[:5],
        }
    if macro_data:
        relevant_data["macro_summary"] = {
            "fed_funds_rate": macro_data.get("fed_funds_rate"),
            "treasury_10y": macro_data.get("treasury_10y"),
            "cpi": macro_data.get("cpi"),
        }

    data_str = _serialize_data(relevant_data)
    prompt = TECHNICAL_ANALYST.format(
        data=data_str,
        context_header=context_header,
        intelligence_brief=intelligence_brief,
        pre_brief=pre_brief or "No recent events recorded.",
        momentum_context=momentum_context or "No momentum history available.",
    )

    try:
        text, _ = gemini_generate(prompt, model=Config.GEMINI_TECHNICAL_MODEL, timeout=90)
        return text
    except Exception as e:
        logger.error(f"Technical analyst failed: {e}")
        return None


def run_contrarian_analyst(
    stock_data: dict,
    macro_data: dict | None = None,
    context_header: str = "",
    intelligence_brief: str = "",
    pre_brief: str = "",
    momentum_context: str = "",
) -> Optional[str]:
    """Run the Contrarian / Risk Analyst (GPT 5.5 via ChatGPT backend).

    Devil's advocate, stress-tests every thesis. Uses GPT 5.5 via the
    ChatGPT backend OAuth flow for an independent model perspective from
    the rest of the council.
    """
    relevant_data = _extract_relevant_risk_data(stock_data)
    if macro_data:
        relevant_data["macro"] = macro_data

    data_str = _serialize_data(relevant_data)
    prompt = CONTRARIAN_ANALYST.format(
        data=data_str,
        context_header=context_header,
        intelligence_brief=intelligence_brief,
        pre_brief=pre_brief or "No recent events recorded.",
        momentum_context=momentum_context or "No momentum history available.",
    )

    try:
        client = ChatGPTBackendClient()
        return client.chat(prompt)
    except Exception as e:
        logger.error(f"Contrarian analyst failed: {e}")
        return None
