"""Market profiles and instrument normalization for US and Indian equities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

from .models import InstrumentRef, MarketCode

_US_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")
_INDIA_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9&.-]{0,29}$")


@dataclass(frozen=True)
class MarketProfile:
    code: MarketCode
    name: str
    currency: str
    timezone: str
    default_exchange: str
    exchanges: tuple[str, ...]
    regular_open: time
    regular_close: time
    fractional_equities_default: bool
    research_status: str
    research_notes: tuple[str, ...]

    def session_status(self, now: datetime | None = None) -> dict[str, object]:
        """Return a conservative session estimate.

        The broker/exchange status remains authoritative for orders. The India
        profile deliberately does not pretend that a weekday-only calendar
        knows NSE/BSE holidays.
        """
        moment = (now or datetime.now(UTC)).astimezone(ZoneInfo(self.timezone))
        weekday = moment.weekday() < 5
        regular = (
            weekday
            and self.regular_open
            <= moment.time().replace(tzinfo=None)
            < self.regular_close
        )
        return {
            "market": self.code.value,
            "timezone": self.timezone,
            "local_time": moment.isoformat(),
            "regular_session_estimate": regular,
            "calendar_confidence": "weekday_clock_estimate",
            "order_rule": "Broker market status and tradability must be checked immediately before placement.",
        }


PROFILES: dict[MarketCode, MarketProfile] = {
    MarketCode.US: MarketProfile(
        code=MarketCode.US,
        name="United States listed equities",
        currency="USD",
        timezone="America/New_York",
        default_exchange="US",
        exchanges=("US", "NYSE", "NASDAQ", "AMEX"),
        regular_open=time(9, 30),
        regular_close=time(16, 0),
        fractional_equities_default=True,
        research_status="native",
        research_notes=(
            "Artha's built-in funnel, SEC cross-checks, scoring, and Council are designed for US equities.",
        ),
    ),
    MarketCode.INDIA: MarketProfile(
        code=MarketCode.INDIA,
        name="Indian cash equities",
        currency="INR",
        timezone="Asia/Kolkata",
        default_exchange="NSE",
        exchanges=("NSE", "BSE"),
        regular_open=time(9, 15),
        regular_close=time(15, 30),
        fractional_equities_default=False,
        research_status="adapter_required",
        research_notes=(
            "Broker, symbol, currency, and session plumbing is supported.",
            "The built-in US SEC/FMP Council must not be treated as India-ready without an India research adapter.",
            "Indian cash-equity API orders use whole-share delivery limits and require a broker-verified instrument identifier.",
        ),
    ),
}


def get_market_profile(value: str | MarketCode) -> MarketProfile:
    try:
        code = (
            value if isinstance(value, MarketCode) else MarketCode(str(value).upper())
        )
    except ValueError as exc:
        raise ValueError("market must be US or IN") from exc
    return PROFILES[code]


def normalize_instrument(
    symbol: str,
    *,
    market: str | MarketCode,
    exchange: str | None = None,
    broker_instrument_id: str | None = None,
) -> InstrumentRef:
    profile = get_market_profile(market)
    raw = str(symbol or "").strip().upper()
    selected_exchange = str(exchange or profile.default_exchange).strip().upper()
    if selected_exchange not in profile.exchanges:
        raise ValueError(
            f"exchange must be one of {', '.join(profile.exchanges)} for {profile.code.value}"
        )

    if profile.code == MarketCode.US:
        raw = raw.removesuffix(".US")
        if not _US_SYMBOL.fullmatch(raw):
            raise ValueError("invalid US equity symbol")
        research_symbol = raw
    else:
        if raw.endswith(".NS"):
            raw = raw[:-3]
            selected_exchange = "NSE"
        elif raw.endswith(".BO"):
            raw = raw[:-3]
            selected_exchange = "BSE"
        if not _INDIA_SYMBOL.fullmatch(raw):
            raise ValueError("invalid Indian equity symbol")
        research_symbol = f"{raw}.NS" if selected_exchange == "NSE" else f"{raw}.BO"

    return InstrumentRef(
        market=profile.code,
        symbol=raw,
        exchange=selected_exchange,
        currency=profile.currency,
        research_symbol=research_symbol,
        broker_instrument_id=str(broker_instrument_id).strip()
        if broker_instrument_id
        else None,
    )
