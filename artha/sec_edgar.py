"""SEC EDGAR ingestion for official filing context.

Uses the SEC's no-key JSON APIs:
- company_tickers.json for ticker -> CIK resolution
- submissions/CIK##########.json for filing history
- companyfacts/CIK##########.json for XBRL facts

The council should use this as an official-source cross-check for FMP
fundamentals, not as a replacement for market data.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import requests

from .config import Config

logger = logging.getLogger(__name__)

_session = requests.Session()
_cache: dict[str, tuple[float, Any]] = {}
_last_sec_call = 0.0

_SEC_MIN_INTERVAL_SECONDS = 0.13  # <= ~7.7 req/s, below SEC's 10 req/s guidance.

_FACT_TAGS = {
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "Revenues": "revenue",
    "NetIncomeLoss": "net_income",
    "Assets": "assets",
    "Liabilities": "liabilities",
    "StockholdersEquity": "stockholders_equity",
    "CashAndCashEquivalentsAtCarryingValue": "cash",
    "NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
    "EarningsPerShareDiluted": "eps_diluted",
    "WeightedAverageNumberOfDilutedSharesOutstanding": "diluted_shares",
    "LongTermDebtAndFinanceLeaseObligationsCurrent": "current_long_term_debt",
    "LongTermDebtAndFinanceLeaseObligationsNoncurrent": "noncurrent_long_term_debt",
    "LongTermDebtCurrent": "current_long_term_debt",
    "LongTermDebtNoncurrent": "noncurrent_long_term_debt",
}

_PREFERRED_UNITS = ("USD", "USD/shares", "shares")


def _cache_get(key: str, ttl_seconds: int) -> Optional[Any]:
    entry = _cache.get(key)
    if not entry:
        return None
    stored_at, payload = entry
    if time.time() - stored_at <= ttl_seconds:
        return payload
    _cache.pop(key, None)
    return None


def _cache_set(key: str, payload: Any) -> None:
    _cache[key] = (time.time(), payload)


def _sec_wait() -> None:
    global _last_sec_call
    elapsed = time.monotonic() - _last_sec_call
    if elapsed < _SEC_MIN_INTERVAL_SECONDS:
        time.sleep(_SEC_MIN_INTERVAL_SECONDS - elapsed)
    _last_sec_call = time.monotonic()


def _sec_get_json(url: str, *, cache_key: str, ttl_seconds: int | None = None) -> Optional[Any]:
    ttl = ttl_seconds if ttl_seconds is not None else Config.SEC_CACHE_TTL_SECONDS
    cached = _cache_get(cache_key, ttl)
    if cached is not None:
        return cached

    headers = {
        "User-Agent": Config.SEC_USER_AGENT,
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
    }
    try:
        _sec_wait()
        resp = _session.get(url, headers=headers, timeout=20)
        if resp.status_code in (403, 429):
            logger.warning("[sec] HTTP %s from %s; check SEC_USER_AGENT / pacing", resp.status_code, url)
            return None
        resp.raise_for_status()
        payload = resp.json()
        _cache_set(cache_key, payload)
        return payload
    except requests.exceptions.Timeout:
        logger.warning("[sec] Timeout fetching %s", url)
    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "unknown"
        logger.warning("[sec] HTTP %s fetching %s", code, url)
    except ValueError:
        logger.warning("[sec] Non-JSON response from %s", url)
    except Exception as exc:
        logger.warning("[sec] Unexpected error fetching %s: %s", url, exc)
    return None


def _format_cik(cik: int | str) -> str:
    return str(cik).strip().lstrip("0").zfill(10)


def _ticker_map() -> dict[str, dict]:
    payload = _sec_get_json(
        Config.SEC_TICKER_MAP_URL,
        cache_key="sec:ticker_map",
        ttl_seconds=Config.SEC_CACHE_TTL_SECONDS,
    )
    if not isinstance(payload, dict):
        return {}

    result: dict[str, dict] = {}
    for item in payload.values():
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", "")).upper().strip()
        cik = item.get("cik_str")
        if not ticker or cik is None:
            continue
        result[ticker] = {
            "ticker": ticker,
            "cik": _format_cik(cik),
            "entity_name": item.get("title"),
        }
    return result


def _resolve_cik(ticker: str, profile: dict | None = None) -> Optional[dict]:
    symbol = ticker.upper().strip()
    profile = profile or {}
    for key in ("cik", "cik_str", "cikNumber"):
        cik = profile.get(key)
        if cik:
            return {
                "ticker": symbol,
                "cik": _format_cik(cik),
                "entity_name": profile.get("companyName") or profile.get("companyNameLong"),
            }
    mapped = _ticker_map().get(symbol)
    if mapped:
        return mapped
    return None


def _latest_filings(submissions: dict, limit: int = 10) -> list[dict]:
    recent = ((submissions or {}).get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    filing_dates = recent.get("filingDate") or []
    report_dates = recent.get("reportDate") or []
    accession_numbers = recent.get("accessionNumber") or []
    primary_docs = recent.get("primaryDocument") or []
    items = recent.get("items") or []

    filings: list[dict] = []
    wanted = {"10-K", "10-Q", "8-K"}
    for idx, form in enumerate(forms):
        if form not in wanted:
            continue
        filings.append({
            "form": form,
            "filing_date": filing_dates[idx] if idx < len(filing_dates) else None,
            "report_date": report_dates[idx] if idx < len(report_dates) else None,
            "accession_number": accession_numbers[idx] if idx < len(accession_numbers) else None,
            "primary_document": primary_docs[idx] if idx < len(primary_docs) else None,
            "items": items[idx] if idx < len(items) else None,
        })
        if len(filings) >= limit:
            break
    return filings


def _parse_filed_date(item: dict) -> str:
    return str(item.get("filed") or item.get("end") or "")


def _extract_fact_series(companyfacts: dict, limit_per_fact: int = 4) -> list[dict]:
    us_gaap = ((companyfacts or {}).get("facts") or {}).get("us-gaap") or {}
    facts: list[dict] = []

    for tag, label in _FACT_TAGS.items():
        tag_payload = us_gaap.get(tag)
        if not isinstance(tag_payload, dict):
            continue
        units = tag_payload.get("units") or {}
        unit_name = None
        rows = None
        for preferred in _PREFERRED_UNITS:
            if isinstance(units.get(preferred), list):
                unit_name = preferred
                rows = units[preferred]
                break
        if rows is None:
            for candidate_unit, candidate_rows in units.items():
                if isinstance(candidate_rows, list):
                    unit_name = candidate_unit
                    rows = candidate_rows
                    break
        if not rows:
            continue

        cleaned = []
        seen: set[tuple[str, str, str]] = set()
        for row in sorted(rows, key=_parse_filed_date, reverse=True):
            if not isinstance(row, dict) or row.get("val") is None:
                continue
            key = (str(row.get("end", "")), str(row.get("form", "")), str(row.get("fp", "")))
            if key in seen:
                continue
            seen.add(key)
            cleaned.append({
                "end": row.get("end"),
                "filed": row.get("filed"),
                "fy": row.get("fy"),
                "fp": row.get("fp"),
                "form": row.get("form"),
                "value": row.get("val"),
            })
            if len(cleaned) >= limit_per_fact:
                break

        if cleaned:
            facts.append({
                "label": label,
                "tag": tag,
                "unit": unit_name,
                "recent": cleaned,
            })

    return facts


def _filing_staleness_days(latest_filings: list[dict]) -> Optional[int]:
    latest = next((f for f in latest_filings if f.get("form") in ("10-Q", "10-K")), None)
    date_text = (latest or {}).get("filing_date")
    if not date_text:
        return None
    try:
        filed_dt = datetime.strptime(str(date_text), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - filed_dt).days
    except ValueError:
        return None


def get_sec_company_context(ticker: str, profile: dict | None = None) -> dict:
    """Return compact SEC filing context for council prompts.

    The payload is intentionally compact; raw companyfacts files can be huge.
    """
    symbol = ticker.upper().strip()
    resolved = _resolve_cik(symbol, profile=profile)
    if not resolved:
        return {
            "source": "sec",
            "status": "unavailable",
            "ticker": symbol,
            "error": "cik_not_found",
        }

    cik = resolved["cik"]
    submissions_url = f"{Config.SEC_SUBMISSIONS_BASE_URL}/CIK{cik}.json"
    facts_url = f"{Config.SEC_COMPANYFACTS_BASE_URL}/CIK{cik}.json"

    submissions = _sec_get_json(submissions_url, cache_key=f"sec:submissions:{cik}")
    facts_payload = _sec_get_json(facts_url, cache_key=f"sec:companyfacts:{cik}")

    latest = _latest_filings(submissions or {})
    fact_series = _extract_fact_series(facts_payload or {})
    stale_days = _filing_staleness_days(latest)

    status = "ok" if latest or fact_series else "partial"
    if submissions is None and facts_payload is None:
        status = "unavailable"

    return {
        "source": "sec",
        "status": status,
        "ticker": symbol,
        "cik": cik,
        "entity_name": (
            (submissions or {}).get("name")
            or resolved.get("entity_name")
            or (profile or {}).get("companyName")
        ),
        "sic": (submissions or {}).get("sic"),
        "sic_description": (submissions or {}).get("sicDescription"),
        "latest_filings": latest,
        "latest_10q_or_10k_staleness_days": stale_days,
        "financial_facts": fact_series,
        "facts_available": len(fact_series),
        "submissions_available": submissions is not None,
        "companyfacts_available": facts_payload is not None,
    }
