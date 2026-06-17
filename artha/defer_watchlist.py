"""Monitored entry watchlist for council decisions.

The table started as a DEFER/WATCH entry watchlist. It also safely supports
buy-like decisions that are attractive only at a future broker-executable entry
zone, such as fractional Robinhood ideas that cannot be parked as resting limit
orders.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

_DASH_RANGE = f"(?:-|to|{chr(8211)}|{chr(8212)})"
_RANGE_RE = re.compile(
    r"(?:\$|USD\s*)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*"
    + _DASH_RANGE
    + r"\s*"
    r"(?:\$|USD\s*)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_MONEY_RE = re.compile(r"(?:\$|USD\s*)\s*([0-9][0-9,]*(?:\.[0-9]+)?)", re.IGNORECASE)
_ENTRY_WORDS = (
    "entry",
    "enter",
    "re-evaluate",
    "reevaluate",
    "revisit",
    "buy",
    "starter",
    "accumulate",
    "defer",
    "pullback",
    "zone",
    "alert",
    "watch",
    "wait",
    "near",
)
_NEGATIVE_WORDS = ("target", "upside", "take profit", "sell", "trim", "stop")
_NON_ENTRY_MONEY_WORDS = (
    "nav",
    "deployable",
    "deployment",
    "budget",
    "allocation",
    "cash",
    "fractional market review",
    "fractional review",
    "starter-sized review",
    "notional",
    "one full share",
    "full share",
    "consensus target",
    "analyst target",
    "price target",
    "dcf",
    "fair value",
    "current price",
    "do not buy at",
    "do not open at",
    "eps",
    "estimate",
    "earnings",
    "net write-off",
    "interest coverage",
)
_NON_ENTRY_MONEY_PATTERNS = (
    re.compile(
        r"(fractional(?:\s+\w+){0,4}\s+review|market(?:\s+\w+){0,4}\s+review|starter-sized(?:\s+\w+){0,4}\s+review)"
        r"[^.\n]{0,80}(?:\$|usd\s*)\s*[0-9]",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:for|spend|budget|allocation|notional|nav|deployable|buying power)"
        r"[^.\n]{0,80}(?:\$|usd\s*)\s*[0-9]",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:\$|usd\s*)\s*[0-9][0-9,]*(?:\.[0-9]+)?[^.\n]{0,80}"
        r"(nav|notional|spend|budget|allocation|fractional(?:\s+\w+){0,4}\s+review)",
        re.IGNORECASE,
    ),
)


@dataclass
class EntryZone:
    low: float
    high: float
    trigger_text: str
    trigger_type: str = "zone"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat()


def _parse_money(value: str) -> float | None:
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return None


def _window(text: str, start: int, end: int, width: int = 140) -> str:
    left = max(0, start - width)
    right = min(len(text), end + width)
    return " ".join(text[left:right].split())


def _score_context(context: str) -> int:
    lowered = context.lower()
    score = sum(2 for word in _ENTRY_WORDS if word in lowered)
    score -= sum(2 for word in _NEGATIVE_WORDS if word in lowered)
    score -= sum(2 for word in _NON_ENTRY_MONEY_WORDS if word in lowered)
    return score


def _is_non_entry_money_context(context: str) -> bool:
    """Return True when a dollar amount is clearly not a stock entry price."""
    text = str(context or "")
    lowered = text.lower()
    if any(pattern.search(text) for pattern in _NON_ENTRY_MONEY_PATTERNS):
        return True
    if "review for about $" in lowered or "review for $" in lowered:
        return True
    return False


def _plausible_relative_to_price(zone: EntryZone, current_price: float | None) -> bool:
    """Filter obvious notional/EPS/ratio ranges misread as stock entry zones."""
    if not current_price or current_price <= 0:
        return True
    low = min(float(zone.low), float(zone.high))
    high = max(float(zone.low), float(zone.high))
    if high < current_price * 0.35:
        return False
    if low > current_price * 2.5:
        return False
    return True


def _dedupe_zones(candidates: list[tuple[int, EntryZone]], max_zones: int) -> list[EntryZone]:
    selected: list[EntryZone] = []
    for _, zone in sorted(candidates, key=lambda item: (item[0], item[1].high), reverse=True):
        duplicate = False
        for existing in selected:
            overlap_low = max(existing.low, zone.low)
            overlap_high = min(existing.high, zone.high)
            if overlap_high >= overlap_low:
                duplicate = True
                break
            scale = max(existing.high, zone.high, 1.0)
            if abs(existing.low - zone.low) / scale < 0.015 and abs(existing.high - zone.high) / scale < 0.015:
                duplicate = True
                break
        if not duplicate:
            selected.append(zone)
        if len(selected) >= max_zones:
            break
    return selected


def extract_entry_zones(
    text: str,
    current_price: float | None = None,
    max_zones: int = 3,
) -> list[EntryZone]:
    """Extract likely entry/revisit zones from a CIO action narrative.

    This intentionally favors explicit price ranges near entry/watch language.
    If no range exists, it accepts a single pullback/revisit price and widens it
    to a narrow 1% band so the monitor can fire before the exact tick is hit.
    """
    text = str(text or "")
    if not text.strip():
        return []

    candidates: list[tuple[int, EntryZone]] = []
    for match in _RANGE_RE.finditer(text):
        a = _parse_money(match.group(1))
        b = _parse_money(match.group(2))
        if a is None or b is None or a <= 0 or b <= 0:
            continue
        low, high = sorted((a, b))
        if high / low > 1.5:
            continue
        if _is_non_entry_money_context(_window(text, match.start(), match.end(), width=55)):
            continue
        context = _window(text, match.start(), match.end())
        score = _score_context(context)
        if current_price and high <= current_price * 1.02:
            score += 3
            trigger_type = "pullback"
        elif current_price and low >= current_price * 0.98:
            score += 1
            trigger_type = "breakout"
        else:
            trigger_type = "zone"
        candidates.append((score, EntryZone(low=low, high=high, trigger_text=context, trigger_type=trigger_type)))

    single_candidates: list[tuple[int, EntryZone]] = []
    for match in _MONEY_RE.finditer(text):
        price = _parse_money(match.group(1))
        if price is None or price <= 0:
            continue
        if _is_non_entry_money_context(_window(text, match.start(), match.end(), width=55)):
            continue
        context = _window(text, match.start(), match.end())
        score = _score_context(context)
        if current_price:
            if price < current_price:
                score += 2
                trigger_type = "pullback"
            elif price > current_price:
                trigger_type = "breakout"
            else:
                trigger_type = "zone"
        else:
            trigger_type = "zone"
        if score <= 0:
            continue
        band = max(price * 0.01, 0.25)
        single_candidates.append(
            (
                score,
                EntryZone(
                    low=max(0.01, price - band),
                    high=price + band,
                    trigger_text=context,
                    trigger_type=trigger_type,
                ),
            )
        )

    all_candidates = [(score, zone) for score, zone in (candidates + single_candidates) if score > 0]
    if current_price:
        all_candidates = [
            (score, zone)
            for score, zone in all_candidates
            if _plausible_relative_to_price(zone, current_price)
        ]
    return _dedupe_zones(all_candidates, max(1, int(max_zones or 1)))


def extract_entry_zone(text: str, current_price: float | None = None) -> EntryZone | None:
    """Extract the strongest entry/revisit zone from a CIO action narrative."""
    zones = extract_entry_zones(text, current_price=current_price, max_zones=1)
    return zones[0] if zones else None


def _decision_text(decision: Any) -> str:
    parts = [
        str(getattr(decision, "recommended_action", "") or ""),
        str(getattr(decision, "synthesis_report", "") or ""),
    ]
    return "\n".join(part for part in parts if part.strip())


def _default_expiry(decision: Any) -> str:
    value = str(getattr(decision, "entry_valid_until", "") or "").strip()
    if value:
        return value
    return (_utcnow() + timedelta(days=30)).isoformat()


def record_defer_watch(
    decision: Any,
    current_price: float | None,
    journal: Any,
    allowed_verdicts: set[str] | None = None,
    note_prefix: str = "",
) -> dict[str, Any] | None:
    """Persist/update monitored entry watches for an allowed decision.

    By default this only records DEFER/WATCH rows. Callers may pass
    allowed_verdicts to record buy-like entry watches when the investment thesis
    is valid but the broker cannot express the intended fractional limit order
    as a resting order.
    """
    verdict = str(getattr(decision, "final_verdict", "") or "").upper()
    allowed = {str(v or "").upper() for v in (allowed_verdicts or {"DEFER", "WATCH"})}
    if verdict not in allowed:
        return None

    ticker = str(getattr(decision, "ticker", "") or "").upper().strip()
    if not ticker:
        return None

    zones = extract_entry_zones(_decision_text(decision), current_price=current_price, max_zones=3)
    if not zones:
        logger.info("[defer_watchlist] No explicit entry zone found for %s; watch not created", ticker)
        return None

    now = _utcnow_iso()
    trace = getattr(decision, "agentic_trace", {}) or {}
    try:
        existing_rows = list(journal.get_active_defer_watches_for_ticker(ticker))
    except AttributeError:
        existing = journal.get_active_defer_watch_for_ticker(ticker)
        existing_rows = [existing] if existing else []

    def matching_existing(zone: EntryZone) -> dict[str, Any] | None:
        for row in existing_rows:
            if not row:
                continue
            if str(row.get("trigger_type") or "zone").lower() != zone.trigger_type:
                continue
            old_low = _as_float(row.get("zone_low"))
            old_high = _as_float(row.get("zone_high"))
            if old_low is None or old_high is None:
                continue
            scale = max(old_high, zone.high, 1.0)
            if abs(old_low - zone.low) / scale <= 0.015 and abs(old_high - zone.high) / scale <= 0.015:
                return row
        return None

    saved: list[dict[str, Any]] = []
    for idx, zone in enumerate(zones, start=1):
        existing = matching_existing(zone)
        watch = {
            "watch_id": (existing or {}).get("watch_id") or str(uuid.uuid4()),
            "ticker": ticker,
            "status": "active",
            "source_action": str(getattr(decision, "recommended_action", "") or ""),
            "current_price": float(current_price) if current_price is not None else None,
            "zone_low": float(zone.low),
            "zone_high": float(zone.high),
            "trigger_type": zone.trigger_type,
            "trigger_text": zone.trigger_text,
            "invalidation_conditions": json.dumps(getattr(decision, "invalidation_conditions", []) or []),
            "opportunity_score": float(getattr(decision, "adjusted_score", None) or getattr(decision, "opportunity_score", 0) or 0),
            "confidence": int(getattr(decision, "confidence", 0) or 0),
            "entry_valid_until": _default_expiry(decision),
            "dossier_path": str(getattr(decision, "dossier_path", "") or ""),
            "trace_path": str(trace.get("trace_path") or ""),
            "created_at": (existing or {}).get("created_at") or now,
            "updated_at": now,
            "notes": (
                f"{note_prefix.strip() + '; ' if note_prefix.strip() else ''}"
                f"{verdict} entry watch from council; zone {idx}/{len(zones)} extracted from CIO action text."
            ),
        }
        journal.save_defer_watch(watch)
        saved.append(watch)
        logger.info(
            "[defer_watchlist] Active %s watch for %s at %.2f-%.2f (%d/%d)",
            zone.trigger_type,
            ticker,
            zone.low,
            zone.high,
            idx,
            len(zones),
        )

    try:
        keep_ids = [str(row.get("watch_id") or "") for row in saved if row.get("watch_id")]
        if keep_ids and hasattr(journal, "supersede_defer_watches_for_ticker"):
            journal.supersede_defer_watches_for_ticker(
                ticker,
                keep_watch_ids=keep_ids,
                notes="Superseded by newer Council entry-watch zones for the same ticker.",
            )
    except Exception as exc:
        logger.warning("[defer_watchlist] Could not supersede stale watches for %s: %s", ticker, exc)

    primary = saved[0]
    if len(saved) > 1:
        primary = {**primary, "extra_watch_ids": [row.get("watch_id") for row in saved[1:]]}
    return primary


def check_defer_watch_trigger(watch: dict[str, Any], price: float) -> dict[str, Any] | None:
    """Return trigger payload if a watched ticker reaches its entry condition."""
    try:
        low = float(watch.get("zone_low"))
        high = float(watch.get("zone_high"))
    except Exception:
        return None
    if low <= 0 or high <= 0 or price <= 0:
        return None

    trigger_type = str(watch.get("trigger_type") or "zone").lower()
    triggered = False
    severity = "INFO"
    state = "inside zone"
    if trigger_type == "pullback":
        triggered = price <= high
        if price < low:
            severity = "WARNING"
            state = "below zone"
    elif trigger_type == "breakout":
        triggered = price >= low
        if price > high:
            severity = "WARNING"
            state = "above zone"
    else:
        triggered = low <= price <= high

    if not triggered:
        return None

    ticker = str(watch.get("ticker") or "").upper()
    message = (
        f"{ticker} reached entry watch {state}: current ${price:,.2f}, "
        f"zone ${low:,.2f}-${high:,.2f}. Re-run council before buying."
    )
    return {
        "watch_id": watch.get("watch_id"),
        "ticker": ticker,
        "severity": severity,
        "message": message,
        "price": float(price),
        "zone_low": low,
        "zone_high": high,
        "state": state,
    }


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", "").replace("%", ""))
    except Exception:
        return None


def scan_skip_for_defer_watch(
    watch: dict[str, Any] | None,
    price: float | None,
    *,
    candidate: dict[str, Any] | None = None,
    buffer_pct: float = 5.0,
    major_move_pct: float = 5.0,
) -> dict[str, Any]:
    """Decide whether a scan should skip a ticker already waiting on a watch zone.

    This is intentionally conservative. It skips only when the ticker is still
    clearly away from the saved DEFER/WATCH entry condition. A large same-day
    move overrides the skip because it may represent fresh information worth a
    new council review.
    """
    if not watch:
        return {"skip": False, "reason": "no_active_watch"}

    price_float = _as_float(price)
    if price_float is None or price_float <= 0:
        return {"skip": False, "reason": "missing_price"}

    low = _as_float(watch.get("zone_low"))
    high = _as_float(watch.get("zone_high"))
    if low is None or high is None or low <= 0 or high <= 0:
        return {"skip": False, "reason": "invalid_watch_zone"}
    if low > high:
        low, high = high, low

    candidate = candidate or {}
    move_fields = (
        candidate.get("changesPercentage"),
        candidate.get("change_pct"),
        candidate.get("change_percent"),
        candidate.get("daily_change_pct"),
        candidate.get("percent_change"),
    )
    daily_move = next((abs(v) for v in (_as_float(item) for item in move_fields) if v is not None), None)
    if daily_move is not None and daily_move >= max(float(major_move_pct or 0), 0.0):
        return {
            "skip": False,
            "reason": "major_move_override",
            "daily_move_pct": daily_move,
            "price": price_float,
            "zone_low": low,
            "zone_high": high,
        }

    buffer = max(float(buffer_pct or 0), 0.0) / 100.0
    trigger_type = str(watch.get("trigger_type") or "zone").lower().strip()
    ticker = str(watch.get("ticker") or "").upper()

    if trigger_type == "breakout":
        near_low = low * (1 - buffer)
        if price_float < near_low:
            return {
                "skip": True,
                "reason": "active_breakout_watch_below_zone",
                "ticker": ticker,
                "price": price_float,
                "zone_low": low,
                "zone_high": high,
                "distance_pct": ((low / price_float) - 1) * 100,
                "watch_id": watch.get("watch_id"),
            }
        return {"skip": False, "reason": "near_or_inside_watch_zone", "price": price_float, "zone_low": low, "zone_high": high}

    if trigger_type == "pullback":
        near_high = high * (1 + buffer)
        if price_float > near_high:
            return {
                "skip": True,
                "reason": "active_pullback_watch_above_zone",
                "ticker": ticker,
                "price": price_float,
                "zone_low": low,
                "zone_high": high,
                "distance_pct": ((price_float / high) - 1) * 100,
                "watch_id": watch.get("watch_id"),
            }
        return {"skip": False, "reason": "near_or_inside_watch_zone", "price": price_float, "zone_low": low, "zone_high": high}

    near_low = low * (1 - buffer)
    near_high = high * (1 + buffer)
    if price_float < near_low:
        return {
            "skip": True,
            "reason": "active_zone_watch_below_zone",
            "ticker": ticker,
            "price": price_float,
            "zone_low": low,
            "zone_high": high,
            "distance_pct": ((low / price_float) - 1) * 100,
            "watch_id": watch.get("watch_id"),
        }
    if price_float > near_high:
        return {
            "skip": True,
            "reason": "active_zone_watch_above_zone",
            "ticker": ticker,
            "price": price_float,
            "zone_low": low,
            "zone_high": high,
            "distance_pct": ((price_float / high) - 1) * 100,
            "watch_id": watch.get("watch_id"),
        }
    return {"skip": False, "reason": "near_or_inside_watch_zone", "price": price_float, "zone_low": low, "zone_high": high}
