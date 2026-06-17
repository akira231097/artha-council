"""Current-web search provider for Research Desk and Sentinel.

Provider order:
1. Brave Search API when BRAVE_SEARCH_API_KEY/BRAVE_API_KEY is configured.
2. Gemini Google Search grounding when Gemini/Google API key is configured.

Serper is intentionally not used here; its quota failures were causing Artha to
produce thin research while appearing operational.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import requests

from .config import Config
from .gemini_client import gemini_generate

logger = logging.getLogger(__name__)

_SESSION = requests.Session()
_BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
_FENCE = chr(96) * 3


def search_web(query: str, *, count: int = 5, freshness: str = "week") -> list[dict[str, Any]]:
    """Return normalized current-web search results.

    Result shape matches the old Serper shape where possible:
    title/url/snippet/date/query/provider.
    """
    normalized_query = (query or "").strip()
    if not normalized_query:
        return []

    if Config.BRAVE_SEARCH_API_KEY:
        results = _search_brave(normalized_query, count=count, freshness=freshness)
        if results:
            return results

    if Config.GEMINI_API_KEY or Config.GOOGLE_API_KEY:
        return _search_gemini_grounded(normalized_query, count=count, freshness=freshness)

    logger.warning(
        "No current-web search provider configured. Set BRAVE_SEARCH_API_KEY or GEMINI_API_KEY/GOOGLE_API_KEY."
    )
    return []


def _search_brave(query: str, *, count: int, freshness: str) -> list[dict[str, Any]]:
    freshness_map = {
        "day": "pd",
        "week": "pw",
        "month": "pm",
        "year": "py",
    }
    params: dict[str, Any] = {
        "q": query,
        "count": max(1, min(int(count or 5), 10)),
        "search_lang": "en",
        "country": "us",
    }
    if freshness in freshness_map:
        params["freshness"] = freshness_map[freshness]

    try:
        response = _SESSION.get(
            _BRAVE_URL,
            headers={
                "X-Subscription-Token": Config.BRAVE_SEARCH_API_KEY,
                "Accept": "application/json",
            },
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        items = ((data.get("web") or {}).get("results") or [])[:count]
        results: list[dict[str, Any]] = []
        for item in items:
            url = item.get("url") or ""
            title = item.get("title") or url
            if not url:
                continue
            results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": item.get("description") or "",
                    "date": item.get("age") or "",
                    "query": query,
                    "provider": "brave",
                }
            )
        logger.info("[search] Brave returned %d result(s) for: %s", len(results), query[:80])
        return results
    except Exception as exc:
        logger.error("[search] Brave search failed for '%s': %s", query, exc)
        return []


def _search_gemini_grounded(query: str, *, count: int, freshness: str) -> list[dict[str, Any]]:
    freshness_text = {
        "day": "published in the last 24 hours when possible",
        "week": "published in the last week when possible",
        "month": "published in the last month when possible",
        "year": "published in the last year when possible",
    }.get(freshness, "recent")
    prompt = f"""Use Google Search to find current investment-relevant sources for this query:
{query}

Return ONLY a JSON array with at most {max(1, min(count, 8))} objects.
Each object must contain: title, url, snippet, date.
Prefer reputable financial/news/filing sources and sources {freshness_text}.
Do not include unsourced commentary."""

    try:
        text, grounding = gemini_generate(
            prompt,
            model=Config.GEMINI_FLASH_MODEL,
            timeout=45,
            google_search=True,
        )
    except Exception as exc:
        logger.error("[search] Gemini grounded search failed for '%s': %s", query, exc)
        return []

    parsed = _parse_json_array(text)
    results: list[dict[str, Any]] = []
    if parsed:
        for item in parsed[:count]:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or item.get("uri") or "").strip()
            title = str(item.get("title") or url).strip()
            if not url:
                continue
            results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": str(item.get("snippet") or "").strip(),
                    "date": str(item.get("date") or "").strip(),
                    "query": query,
                    "provider": "gemini_grounded",
                }
            )

    if not results:
        chunks = grounding.get("groundingChunks") or []
        for chunk in chunks[:count]:
            web = (chunk or {}).get("web") or {}
            url = str(web.get("uri") or "").strip()
            title = str(web.get("title") or url).strip()
            if not url:
                continue
            results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": text[:500],
                    "date": "",
                    "query": query,
                    "provider": "gemini_grounded",
                }
            )

    logger.info("[search] Gemini grounded search returned %d result(s) for: %s", len(results), query[:80])
    return results[:count]


def _parse_json_array(text: str) -> list[Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith(_FENCE + "json"):
        cleaned = cleaned.split(_FENCE + "json", 1)[1]
        if _FENCE in cleaned:
            cleaned = cleaned.rsplit(_FENCE, 1)[0]
        cleaned = cleaned.strip()
    elif cleaned.startswith(_FENCE):
        cleaned = cleaned.split(_FENCE, 1)[1]
        if _FENCE in cleaned:
            cleaned = cleaned.rsplit(_FENCE, 1)[0]
        cleaned = cleaned.strip()

    try:
        value = json.loads(cleaned)
        return value if isinstance(value, list) else []
    except Exception:
        start = cleaned.find("[")
        end = cleaned.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                value = json.loads(cleaned[start:end])
                return value if isinstance(value, list) else []
            except Exception:
                return []
        return []
