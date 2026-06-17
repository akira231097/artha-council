"""Economic Calendar — fetches and classifies upcoming macro events.

Uses FMP economic-calendar endpoint. Classifies events by proximity
to determine event_risk_state for regime-aware decision making.

Timezone: FMP events are in ET. System runs in CT (UTC-6/5).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from .collector import _safe_get
from .config import Config

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

# Events considered "major" for risk gating
MAJOR_EVENT_KEYWORDS = [
    "FOMC", "Federal Open Market Committee",
    "Fed Rate", "Federal Reserve",
    "CPI", "Consumer Price Index",
    "Nonfarm Payroll", "NFP", "Employment Situation",
    "GDP", "Gross Domestic Product",
    "PCE", "Personal Consumption Expenditures",
    "ISM Manufacturing", "ISM Services",
    "Initial Jobless Claims",
    "Retail Sales",
    "PPI", "Producer Price Index",
]


@dataclass
class CalendarEvent:
    """A single economic calendar event."""
    event: str = ""
    date: str = ""          # ISO date string (YYYY-MM-DD)
    time: str = ""          # HH:MM ET
    country: str = ""
    impact: str = ""        # "High", "Medium", "Low"
    actual: Optional[str] = None
    forecast: Optional[str] = None
    previous: Optional[str] = None
    is_major: bool = False
    datetime_et: Optional[datetime] = None


@dataclass
class EventRiskState:
    """Structured event risk output for regime indicators."""
    state: str = "none"                              # none | pre_major_24h | same_day_major | post_major_24h
    next_major_event: Optional[str] = None          # Description of next major event
    next_major_event_date: Optional[str] = None     # ISO date
    major_events_next_7d: list[str] = field(default_factory=list)
    all_events_today: list[str] = field(default_factory=list)


def _is_major_event(event_name: str) -> bool:
    """Return True if event name matches any major event keyword."""
    name_upper = event_name.upper()
    for keyword in MAJOR_EVENT_KEYWORDS:
        if keyword.upper() in name_upper:
            return True
    return False


def _parse_event_datetime(date_str: str, time_str: str) -> Optional[datetime]:
    """Parse FMP event date/time into timezone-aware ET datetime."""
    try:
        if time_str and time_str not in ("", "00:00", "allDay"):
            dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        else:
            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=8, minute=30)
        return dt.replace(tzinfo=ET)
    except (ValueError, TypeError):
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=8, minute=30)
            return dt.replace(tzinfo=ET)
        except (ValueError, TypeError):
            return None


class EconomicCalendar:
    """Fetches and classifies economic events from FMP."""

    def __init__(self):
        self.base = Config.FMP_BASE_URL
        self.key = Config.FMP_API_KEY

    def fetch(self, days_ahead: int = 7, days_back: int = 1) -> list[CalendarEvent]:
        """Fetch economic calendar events from FMP.

        Args:
            days_ahead: How many days forward to fetch (default 7)
            days_back: How many days back to fetch (for "post" state detection)

        Returns:
            List of CalendarEvent objects, sorted by datetime ascending.
        """
        now_utc = datetime.now(UTC)
        from_date = (now_utc - timedelta(days=days_back)).strftime("%Y-%m-%d")
        to_date = (now_utc + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        try:
            url = f"{self.base}/economic-calendar"
            params = {
                "apikey": self.key,
                "from": from_date,
                "to": to_date,
            }
            raw = _safe_get(url, params, "fmp")
            if raw is None:
                logger.warning("[economic_calendar] FMP fetch returned None")
                return []
        except Exception as e:
            logger.warning(f"[economic_calendar] FMP fetch failed: {e}")
            return []

        if not isinstance(raw, list):
            logger.warning(f"[economic_calendar] Unexpected response type: {type(raw)}")
            return []

        events = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            # Only US events
            if item.get("country", "").upper() not in ("US", "USD", ""):
                continue

            event_name = item.get("event", "") or ""
            date_str = item.get("date", "") or ""
            time_str = item.get("time", "") or ""
            impact = item.get("impact", "") or ""

            # Filter to high-impact or major named events
            if impact.lower() not in ("high", "medium") and not _is_major_event(event_name):
                continue

            dt_et = _parse_event_datetime(date_str, time_str)
            evt = CalendarEvent(
                event=event_name,
                date=date_str,
                time=time_str,
                country=item.get("country", "US"),
                impact=impact,
                actual=item.get("actual"),
                forecast=item.get("estimate") or item.get("forecast"),
                previous=item.get("previous"),
                is_major=_is_major_event(event_name),
                datetime_et=dt_et,
            )
            events.append(evt)

        # Sort by datetime
        events.sort(key=lambda e: e.datetime_et or datetime.min.replace(tzinfo=ET))
        logger.info(f"[economic_calendar] Fetched {len(events)} relevant US events ({from_date} → {to_date})")
        return events


def compute_event_risk_state(events: list[CalendarEvent]) -> EventRiskState:
    """Classify the current event risk state based on upcoming/recent events.

    States:
      - "none"              No major events within ±24h or next 7 days
      - "pre_major_24h"     Major event within next 24 hours
      - "same_day_major"    Major event scheduled today (or FOMC day 2)
      - "post_major_24h"    Major event occurred within last 24 hours

    Args:
        events: List of CalendarEvent objects (sorted ascending by date)

    Returns:
        EventRiskState with state, next event info, and 7-day list.
    """
    now_et = datetime.now(ET)
    today_str = now_et.strftime("%Y-%m-%d")
    result = EventRiskState()

    major_events = [e for e in events if e.is_major]

    # Collect all events today
    result.all_events_today = [
        e.event for e in events
        if e.date == today_str
    ]

    # Find major events in next 7 days
    cutoff_7d = now_et + timedelta(days=7)
    result.major_events_next_7d = [
        f"{e.date} {e.event}"
        for e in major_events
        if e.datetime_et and e.datetime_et > now_et and e.datetime_et <= cutoff_7d
    ]

    # Find the next upcoming major event
    future_major = [e for e in major_events if e.datetime_et and e.datetime_et > now_et]
    if future_major:
        next_evt = future_major[0]
        result.next_major_event = next_evt.event
        result.next_major_event_date = next_evt.date

    # Determine state
    cutoff_24h_ahead = now_et + timedelta(hours=24)
    cutoff_24h_back = now_et - timedelta(hours=24)

    # Check same_day_major
    same_day_major = [e for e in major_events if e.date == today_str]
    if same_day_major:
        result.state = "same_day_major"
        return result

    # Check pre_major_24h (event within next 24h but not today)
    pre_24h = [
        e for e in major_events
        if e.datetime_et and now_et < e.datetime_et <= cutoff_24h_ahead
    ]
    if pre_24h:
        result.state = "pre_major_24h"
        return result

    # Check post_major_24h (event occurred within last 24h)
    post_24h = [
        e for e in major_events
        if e.datetime_et and cutoff_24h_back <= e.datetime_et <= now_et
    ]
    if post_24h:
        result.state = "post_major_24h"
        return result

    result.state = "none"
    return result
