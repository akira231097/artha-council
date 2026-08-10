"""Per-ticker news sentiment: Finnhub headlines + one batched LLM classification.

Evidence basis — Lopez-Lira & Tang, "Can ChatGPT Forecast Stock Price Movements?"
(SSRN 4412788): per-headline GOOD/BAD/UNKNOWN LLM classification of company news
predicts short-term returns, and negative news carries roughly 2.5x the signal
of positive news (exit trigger > entry filter).

Design:
- get_headlines():      Finnhub /company-news via the shared, rate-limited
                        FinnhubCollector (no parallel unthrottled client).
- classify_headlines(): ONE batched Gemini flash call per ticker, strict JSON
                        array output, parsed defensively (never guess labels).
- get_daily_sentiment(): per-ticker per-UTC-day cache in
                        data/news_sentiment/YYYY-MM-DD.json (advisory lock +
                        atomic tempfile/os.replace write, same pattern as
                        accuracy.py). Labels are cached by headline hash so
                        only unseen headlines are re-classified. On any fetch
                        or classification failure the row is available=False —
                        unavailable data is NEVER fabricated as neutral.

The daily cache file also persists the sell-side rule-4.7 review-flag dedupe
(one REVIEW_EXIT flag per ticker per UTC day, restart-safe).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

from .config import Config
from .gemini_client import gemini_generate

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "news_sentiment"
_LOCK_PATH = _DATA_DIR / ".news_sentiment.lock"

_VALID_LABELS = {"GOOD", "BAD", "UNKNOWN"}

# Promotional / listicle noise that is not company news. Conservative on
# purpose: false negatives (kept noise) become UNKNOWN in classification;
# false positives (dropped real news) would lose signal.
_PROMO_RE = re.compile(
    r"(?i)\b(zacks|motley fool|should you buy|reasons? to (?:buy|sell)|"
    r"top \d+ (?:stocks|picks)|best \d+ stocks|stocks? to (?:buy|watch)\b)"
)

_shared_collector = None


def _get_collector():
    """Shared FinnhubCollector so the module reuses its response cache."""
    global _shared_collector
    if _shared_collector is None:
        from .collector import FinnhubCollector

        _shared_collector = FinnhubCollector()
    return _shared_collector


def _utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def headline_hash(headline: str) -> str:
    """Stable dedupe/cache key for a headline text."""
    normalized = re.sub(r"\s+", " ", str(headline or "").strip().lower())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def get_headlines(ticker: str, days: int = 3, limit: int = 15) -> Optional[list[dict]]:
    """Fetch recent company headlines from Finnhub /company-news.

    Returns a list of {headline, datetime, source, url, hash} rows (newest
    first, deduped by headline hash, empty/promo items dropped), or None when
    the fetch itself failed (rate limit, network, bad key).
    """
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return None
    articles = _get_collector().company_news(symbol, days_back=max(1, int(days)))
    if articles is None:
        return None

    rows: list[dict] = []
    seen: set[str] = set()
    for article in sorted(
        (a for a in articles if isinstance(a, dict)),
        key=lambda a: a.get("datetime") or 0,
        reverse=True,
    ):
        headline = str(article.get("headline") or "").strip()
        if len(headline) < 5 or _PROMO_RE.search(headline):
            continue
        h = headline_hash(headline)
        if h in seen:
            continue
        seen.add(h)
        ts = article.get("datetime")
        try:
            dt_iso = datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat() if ts else None
        except (TypeError, ValueError, OSError):
            dt_iso = None
        rows.append(
            {
                "headline": headline,
                "datetime": dt_iso,
                "source": str(article.get("source") or "finnhub"),
                "url": str(article.get("url") or ""),
                "hash": h,
            }
        )
        if len(rows) >= max(1, int(limit)):
            break
    return rows


# ---------------------------------------------------------------------------
# Classify
# ---------------------------------------------------------------------------

def _strip_json_fences(text: str) -> str:
    t = str(text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
    # Fall back to the outermost JSON array if the model added prose.
    if not t.lstrip().startswith("["):
        start, end = t.find("["), t.rfind("]")
        if start != -1 and end > start:
            t = t[start : end + 1]
    return t.strip()


def classify_headlines(
    ticker: str,
    company_name: Optional[str],
    rows: list[dict],
) -> Optional[list[str]]:
    """Classify headlines GOOD/BAD/UNKNOWN in ONE batched Gemini flash call.

    Lopez-Lira & Tang (SSRN 4412788) style per-headline classification for the
    SHORT-TERM stock price of the named company. Returns one label per input
    row (aligned), or None when the batch is unavailable (API failure, invalid
    JSON, label/count mismatch) — labels are never guessed.
    """
    if not rows:
        return []
    symbol = str(ticker or "").strip().upper()
    company = str(company_name or "").strip() or symbol
    numbered = "\n".join(
        f"{i}. {str(r.get('headline') or '').strip()}" for i, r in enumerate(rows, start=1)
    )
    prompt = (
        "You are a financial expert with stock market experience.\n"
        f"Company: {company} (ticker: {symbol}).\n\n"
        "For EACH numbered headline below, decide whether it is GOOD, BAD, or "
        f"UNKNOWN for the short-term stock price of {company}:\n"
        f"- GOOD: plausibly positive for {symbol}'s stock price in the coming days.\n"
        f"- BAD: plausibly negative for {symbol}'s stock price in the coming days.\n"
        "- UNKNOWN: unclear, neutral, or promotional. A headline that mentions "
        f"{symbol} only tangentially (market roundups, multi-stock lists, index "
        "moves) should usually be UNKNOWN.\n\n"
        f"Headlines:\n{numbered}\n\n"
        f"Respond with ONLY a JSON array of exactly {len(rows)} strings, one per "
        'headline in order, each exactly "GOOD", "BAD", or "UNKNOWN". No other text.'
    )
    try:
        text, _ = gemini_generate(
            prompt,
            model=Config.GEMINI_FLASH_MODEL,
            temperature=0.2,
            thinking_level="low",
            timeout=60,
        )
    except Exception as e:
        logger.warning("[news_sentiment] Gemini classification failed for %s: %s", symbol, e)
        return None

    try:
        parsed = json.loads(_strip_json_fences(text))
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "[news_sentiment] Unparseable classification for %s: %s", symbol, str(text)[:200]
        )
        return None
    if not isinstance(parsed, list) or len(parsed) != len(rows):
        logger.warning(
            "[news_sentiment] Label count mismatch for %s: expected %d, got %s",
            symbol,
            len(rows),
            len(parsed) if isinstance(parsed, list) else type(parsed).__name__,
        )
        return None
    labels: list[str] = []
    for item in parsed:
        label = str(item or "").strip().upper()
        if label not in _VALID_LABELS:
            logger.warning("[news_sentiment] Invalid label %r for %s", item, symbol)
            return None
        labels.append(label)
    return labels


# ---------------------------------------------------------------------------
# Daily cache (advisory lock + atomic write, following accuracy.py)
# ---------------------------------------------------------------------------

def _cache_path(day: str) -> Path:
    return _DATA_DIR / f"{day}.json"


def _lock():
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    lock_file = open(_LOCK_PATH, "a", encoding="utf-8")
    if fcntl is not None:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    return lock_file


def _unlock(lock_file) -> None:
    if fcntl is not None:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    lock_file.close()


def _load_day(day: str) -> dict:
    path = _cache_path(day)
    if not path.exists():
        return {"date": day, "tickers": {}, "labels": {}, "review_flags": {}}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("cache root is not a dict")
    except (json.JSONDecodeError, OSError, ValueError) as e:
        logger.warning("[news_sentiment] Corrupt cache %s (%s); starting fresh", path, e)
        data = {}
    data.setdefault("date", day)
    data.setdefault("tickers", {})
    data.setdefault("labels", {})
    data.setdefault("review_flags", {})
    return data


def _save_day(day: str, payload: dict) -> None:
    """Atomic write via tempfile + os.replace (crash-safe, same as accuracy.py)."""
    path = _cache_path(day)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=f".{path.stem}_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _known_labels(today: str, lookback_days: int) -> dict[str, dict]:
    """Merge headline-hash label caches from recent days (today wins)."""
    merged: dict[str, dict] = {}
    base = datetime.now(timezone.utc).date()
    for offset in range(max(0, int(lookback_days)), -1, -1):
        day = (base - timedelta(days=offset)).isoformat()
        path = _cache_path(day)
        if not path.exists():
            continue
        labels = _load_day(day).get("labels") or {}
        if isinstance(labels, dict):
            merged.update({k: v for k, v in labels.items() if isinstance(v, dict)})
    return merged


def _unavailable(ticker: str, reason: str) -> dict:
    return {
        "ticker": ticker,
        "as_of_date": _utc_today(),
        "score": None,
        "n_headlines": 0,
        "n_good": 0,
        "n_bad": 0,
        "n_unknown": 0,
        "labeled": [],
        "available": False,
        "reason": reason,
    }


def get_daily_sentiment(ticker: str, company_name: Optional[str] = None) -> dict:
    """Daily per-ticker news sentiment row (cached per UTC day).

    Returns {ticker, as_of_date, score (-1..1 = (good-bad)/scored),
    n_headlines, n_good, n_bad, n_unknown, labeled, available}.

    available=False on any fetch/classify failure. Failures are NOT cached so a
    later call the same day retries; successful rows are served from cache with
    zero Finnhub and zero Gemini calls for the rest of the UTC day.
    """
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return _unavailable(symbol, "empty ticker")
    if not Config.NEWS_SENTIMENT_ENABLED:
        return _unavailable(symbol, "disabled (ARTHA_NEWS_SENTIMENT_ENABLED)")

    today = _utc_today()

    # Cache hit: no API calls at all.
    lf = _lock()
    try:
        cached = (_load_day(today).get("tickers") or {}).get(symbol)
    finally:
        _unlock(lf)
    if isinstance(cached, dict) and cached.get("available"):
        return dict(cached)

    rows = get_headlines(
        symbol, days=Config.NEWS_SENTIMENT_DAYS, limit=Config.NEWS_SENTIMENT_MAX_HEADLINES
    )
    if rows is None:
        return _unavailable(symbol, "finnhub company-news fetch failed")

    # Labels are ticker-scoped: the same headline can be GOOD for one party
    # and BAD for the other (lawsuits, M&A), so a global hash key would let
    # one company's directional label leak onto the other.
    known = _known_labels(today, Config.NEWS_SENTIMENT_DAYS)
    unseen = [r for r in rows if f"{symbol}:{r['hash']}" not in known]
    new_labels: dict[str, str] = {}
    if unseen:
        labels = classify_headlines(symbol, company_name, unseen)
        if labels is None:
            return _unavailable(symbol, "llm classification unavailable")
        new_labels = {r["hash"]: label for r, label in zip(unseen, labels)}

    labeled: list[dict] = []
    n_good = n_bad = n_unknown = 0
    for r in rows:
        label = new_labels.get(r["hash"]) or str(
            (known.get(f"{symbol}:{r['hash']}") or {}).get("label") or ""
        )
        if label not in _VALID_LABELS:
            label = "UNKNOWN"  # stale cache entry with bad shape; count as no-signal
        if label == "GOOD":
            n_good += 1
        elif label == "BAD":
            n_bad += 1
        else:
            n_unknown += 1
        labeled.append(
            {
                "headline": r["headline"],
                "label": label,
                "datetime": r.get("datetime"),
                "source": r.get("source"),
            }
        )

    scored = n_good + n_bad
    result = {
        "ticker": symbol,
        "as_of_date": today,
        "score": round((n_good - n_bad) / scored, 4) if scored else 0.0,
        "n_headlines": len(rows),
        "n_good": n_good,
        "n_bad": n_bad,
        "n_unknown": n_unknown,
        "labeled": labeled,
        "available": True,
    }

    lf = _lock()
    try:
        data = _load_day(today)
        data["tickers"][symbol] = result
        now_iso = datetime.now(timezone.utc).isoformat()
        for r in rows:
            key = f"{symbol}:{r['hash']}"
            label = new_labels.get(r["hash"])
            if label:
                data["labels"][key] = {
                    "label": label,
                    "headline": r["headline"],
                    "labeled_at": now_iso,
                }
            elif key in known and key not in data["labels"]:
                data["labels"][key] = known[key]  # carry forward for lookback
        _save_day(today, data)
    finally:
        _unlock(lf)
    return dict(result)


# ---------------------------------------------------------------------------
# Sell-side rule-4.7 review-flag dedupe (persisted, per ticker per UTC day)
# ---------------------------------------------------------------------------

def was_review_flagged(ticker: str, day: Optional[str] = None) -> bool:
    """True when a REVIEW_EXIT news flag was already emitted today for ticker."""
    symbol = str(ticker or "").strip().upper()
    lf = _lock()
    try:
        flags = _load_day(day or _utc_today()).get("review_flags") or {}
    finally:
        _unlock(lf)
    return symbol in flags


def mark_review_flagged(ticker: str, day: Optional[str] = None) -> None:
    """Persist the per-ticker-per-day REVIEW_EXIT dedupe marker."""
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return
    target_day = day or _utc_today()
    lf = _lock()
    try:
        data = _load_day(target_day)
        data["review_flags"][symbol] = datetime.now(timezone.utc).isoformat()
        _save_day(target_day, data)
    finally:
        _unlock(lf)
