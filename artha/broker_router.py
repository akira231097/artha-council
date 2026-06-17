"""Broker/data-aware routing before scheduled Council buy-now slots.

This router is deliberately narrow. It does not decide whether a company is a
good or bad investment. It decides whether a candidate has sane, fresh-enough
price/quote/liquidity data and is realistically executable today. Interesting
ideas that fail those checks are preserved in research/watch instead of being
discarded as bad companies.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from .config import Config
from .liquidity import resolve_average_volume

logger = logging.getLogger(__name__)

LANE_EXECUTION_READY = "execution_ready"
LANE_RESEARCH_WATCH = "research_watch"
LANE_HARD_REJECT = "hard_reject"

BUCKET_BUY_NOW = "buy_now_council"
BUCKET_RESEARCH_WATCH = "alpha_discovery_watch"
BUCKET_REJECT = "not_eligible"

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
_ETF_INDEX_SKIP = {"SPY", "QQQ", "IWM", "DIA", "VTI"}
_BUY_SIDE_ACTIONS = {"BUY", "STARTER", "TACTICAL_BUY", "ACCUMULATE", "ADD", "STRONG BUY"}
_WATCH_ACTIONS = {"DEFER", "WATCH", "HOLD"}
_AVOID_ACTIONS = {"AVOID", "SELL", "TRIM"}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", "").replace("%", ""))
    except Exception:
        return None


def _as_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _candidate_price(candidate: dict[str, Any]) -> float | None:
    for key in ("price", "lastPrice", "last_price", "current_price", "previous_close"):
        value = _num(candidate.get(key))
        if value is not None and value > 0:
            return value
    quote = candidate.get("quote")
    if isinstance(quote, dict):
        for key in ("price", "lastPrice", "last_price"):
            value = _num(quote.get(key))
            if value is not None and value > 0:
                return value
    return None


def _quote_price(quote: dict[str, Any] | None) -> float | None:
    quote = quote if isinstance(quote, dict) else {}
    for key in ("price", "currentPrice", "regularMarketPrice", "lastPrice", "last_price"):
        value = _num(quote.get(key))
        if value is not None and value > 0:
            return value
    return None


def _quote_bid_ask(candidate: dict[str, Any], quote: dict[str, Any] | None) -> tuple[float | None, float | None]:
    quote = quote if isinstance(quote, dict) else {}
    candidate_quote = candidate.get("quote") if isinstance(candidate.get("quote"), dict) else {}
    yf_quote = candidate.get("yf_quote") if isinstance(candidate.get("yf_quote"), dict) else {}
    bid = (
        _num(quote.get("bid"))
        or _num(quote.get("bid_price"))
        or _num(yf_quote.get("bid"))
        or _num(candidate_quote.get("bid"))
    )
    ask = (
        _num(quote.get("ask"))
        or _num(quote.get("ask_price"))
        or _num(yf_quote.get("ask"))
        or _num(candidate_quote.get("ask"))
    )
    return bid, ask


@dataclass
class BrokerRouteDecision:
    candidate: dict[str, Any]
    ticker: str
    candidate_rank: int
    lane: str
    bucket: str
    reason_code: str
    reason: str
    route_score: float
    funnel_score: float | None = None
    price: float | None = None
    live_price: float | None = None
    bid: float | None = None
    ask: float | None = None
    spread_pct: float | None = None
    avg_volume: float | None = None
    dollar_volume: float | None = None
    liquidity_source: str = ""
    quote_source: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_journal_row(self, session_id: str, created_at: str | None = None) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "created_at": created_at or _utcnow().isoformat(),
            "ticker": self.ticker,
            "candidate_rank": self.candidate_rank,
            "lane": self.lane,
            "bucket": self.bucket,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "route_score": self.route_score,
            "funnel_score": self.funnel_score,
            "price": self.price,
            "live_price": self.live_price,
            "bid": self.bid,
            "ask": self.ask,
            "spread_pct": self.spread_pct,
            "avg_volume": self.avg_volume,
            "dollar_volume": self.dollar_volume,
            "liquidity_source": self.liquidity_source,
            "quote_source": self.quote_source,
            "evidence": self.evidence,
        }


@dataclass
class BrokerRouterResult:
    decisions: list[BrokerRouteDecision]
    selected_for_council: list[BrokerRouteDecision]
    execution_ready: list[BrokerRouteDecision]
    research_watch: list[BrokerRouteDecision]
    hard_reject: list[BrokerRouteDecision]

    def summary_counts(self) -> dict[str, int]:
        return {
            LANE_EXECUTION_READY: len(self.execution_ready),
            LANE_RESEARCH_WATCH: len(self.research_watch),
            LANE_HARD_REJECT: len(self.hard_reject),
            "selected_for_council": len(self.selected_for_council),
        }


class BrokerAwareCandidateRouter:
    """Route funnel finalists into buy-now, research/watch, or hard reject."""

    def __init__(
        self,
        *,
        journal: Any,
        active_watches: dict[str, Any] | None = None,
        quote_provider: Callable[[str], dict[str, Any] | None] | None = None,
        now: datetime | None = None,
        market_open: bool = True,
        max_quote_checks: int | None = None,
    ) -> None:
        self.journal = journal
        self.active_watches = active_watches or {}
        self.quote_provider = quote_provider
        self.now = (now or _utcnow()).astimezone(timezone.utc)
        self.market_open = bool(market_open)
        self.max_quote_checks = max(0, int(max_quote_checks if max_quote_checks is not None else Config.SCAN_BROKER_ROUTER_MAX_QUOTE_CHECKS))
        self._quote_checks = 0

    def route(
        self,
        candidates: list[dict[str, Any]],
        *,
        session_id: str,
        council_limit: int,
        persist: bool = True,
        fill_research_slots: bool | None = None,
    ) -> BrokerRouterResult:
        decisions: list[BrokerRouteDecision] = []
        for rank, candidate in enumerate(candidates or [], start=1):
            decisions.append(self._route_one(candidate or {}, rank))

        execution_ready = [row for row in decisions if row.lane == LANE_EXECUTION_READY]
        research_watch = [row for row in decisions if row.lane == LANE_RESEARCH_WATCH]
        hard_reject = [row for row in decisions if row.lane == LANE_HARD_REJECT]

        execution_ready.sort(key=lambda row: row.route_score, reverse=True)
        research_watch.sort(key=lambda row: row.route_score, reverse=True)

        selected = execution_ready[: max(0, int(council_limit or 0))]
        allow_research_fill = Config.SCAN_ROUTER_FILL_RESEARCH_SLOTS if fill_research_slots is None else bool(fill_research_slots)
        if allow_research_fill and len(selected) < int(council_limit or 0):
            selected.extend(research_watch[: int(council_limit or 0) - len(selected)])

        if persist:
            rows = [row.to_journal_row(session_id) for row in decisions]
            try:
                self.journal.save_scan_routing_decisions(rows)
            except Exception as exc:
                logger.warning("[broker_router] Could not persist routing decisions: %s", exc)

        return BrokerRouterResult(
            decisions=decisions,
            selected_for_council=selected,
            execution_ready=execution_ready,
            research_watch=research_watch,
            hard_reject=hard_reject,
        )

    def _route_one(self, candidate: dict[str, Any], rank: int) -> BrokerRouteDecision:
        ticker = str(candidate.get("symbol") or candidate.get("ticker") or "").upper().strip()
        funnel_score = _num(candidate.get("funnel_score") or candidate.get("combined_score"))
        route_score = float(funnel_score or 0.0)
        evidence: dict[str, Any] = {
            "candidate_rank": rank,
            "funnel_score": funnel_score,
            "primary_alpha_sleeve": candidate.get("primary_alpha_sleeve"),
            "enrichment_pool_reason": candidate.get("enrichment_pool_reason"),
            "broker_router_scope": "execution_feasibility_and_data_quality_only",
        }

        if not ticker or not _TICKER_RE.match(ticker) or ticker in _ETF_INDEX_SKIP:
            return self._decision(
                candidate, ticker or "UNKNOWN", rank, LANE_HARD_REJECT, "invalid_or_excluded_ticker",
                "Ticker is missing, malformed, or excluded from single-stock buy-now Council slots.",
                route_score, funnel_score, evidence,
            )

        candidate_price = _candidate_price(candidate)
        quote = self._fetch_quote(ticker)
        live_price = _quote_price(quote)
        price = live_price or candidate_price
        evidence["candidate_price"] = candidate_price
        evidence["live_quote_price"] = live_price

        if price is None or price <= 0:
            return self._decision(
                candidate, ticker, rank, LANE_HARD_REJECT, "missing_current_price",
                "No usable current price from the funnel or quote provider.",
                route_score, funnel_score, evidence, price=candidate_price, live_price=live_price,
            )
        if price < Config.ROBINHOOD_MIN_PRICE:
            return self._decision(
                candidate, ticker, rank, LANE_HARD_REJECT, "below_minimum_price",
                f"Price ${price:.2f} is below Artha's broker pilot floor.",
                route_score, funnel_score, evidence, price=candidate_price, live_price=live_price,
            )

        quote_source = "quote_provider" if quote else "candidate_funnel"
        if candidate_price and live_price:
            source_drift = abs(live_price - candidate_price) / max(candidate_price, 0.01)
            evidence["price_source_drift_pct"] = source_drift
            if source_drift > Config.SCAN_ROUTER_MAX_PRICE_SOURCE_DRIFT_PCT:
                return self._research_watch(
                    candidate, ticker, rank, "price_source_conflict",
                    (
                        f"Funnel price ${candidate_price:.2f} and live quote ${live_price:.2f} differ by "
                        f"{source_drift:.2%}; route to research/watch until price data is reconciled."
                    ),
                    route_score - 8,
                    funnel_score,
                    evidence,
                    price=candidate_price,
                    live_price=live_price,
                    quote_source=quote_source,
                )

        liquidity = self._liquidity(candidate, quote, price)
        evidence["liquidity"] = liquidity
        route_score += self._liquidity_bonus(liquidity.get("dollar_volume"))
        if liquidity.get("is_average") and (liquidity.get("dollar_volume") or 0) < Config.ROBINHOOD_MIN_DOLLAR_VOLUME:
            return self._research_watch(
                candidate, ticker, rank, "average_liquidity_below_execution_floor",
                (
                    f"Average dollar volume ${liquidity.get('dollar_volume') or 0:,.0f} is below the "
                    f"${Config.ROBINHOOD_MIN_DOLLAR_VOLUME:,.0f} same-day auto-buy floor."
                ),
                route_score - 8,
                funnel_score,
                evidence,
                price=candidate_price,
                live_price=live_price,
                avg_volume=liquidity.get("volume"),
                dollar_volume=liquidity.get("dollar_volume"),
                liquidity_source=str(liquidity.get("source") or ""),
                quote_source=quote_source,
            )
        if not liquidity.get("is_average"):
            return self._research_watch(
                candidate, ticker, rank, "average_liquidity_missing",
                "Average-volume evidence is missing; do not spend a buy-now Council slot on current-volume-only liquidity.",
                route_score - 6,
                funnel_score,
                evidence,
                price=candidate_price,
                live_price=live_price,
                avg_volume=liquidity.get("volume"),
                dollar_volume=liquidity.get("dollar_volume"),
                liquidity_source=str(liquidity.get("source") or ""),
                quote_source=quote_source,
            )

        bid, ask = _quote_bid_ask(candidate, quote)
        spread_pct = None
        if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
            return self._research_watch(
                candidate, ticker, rank, "quote_anomaly_or_missing_bid_ask",
                "Bid/ask quote is missing, zero, or inverted; this may be a data issue, not a bad company.",
                route_score - 10,
                funnel_score,
                evidence,
                price=candidate_price,
                live_price=live_price,
                bid=bid,
                ask=ask,
                avg_volume=liquidity.get("volume"),
                dollar_volume=liquidity.get("dollar_volume"),
                liquidity_source=str(liquidity.get("source") or ""),
                quote_source=quote_source,
            )
        spread_pct = (ask - bid) / ((ask + bid) / 2)
        evidence["spread_pct"] = spread_pct
        route_score -= min(10.0, spread_pct * 200.0)
        if spread_pct > Config.ROBINHOOD_MAX_SPREAD_PCT:
            return self._research_watch(
                candidate, ticker, rank, "spread_too_wide_for_auto_buy",
                (
                    f"Bid/ask spread is {spread_pct:.2%}, above the "
                    f"{Config.ROBINHOOD_MAX_SPREAD_PCT:.2%} buy-now floor."
                ),
                route_score,
                funnel_score,
                evidence,
                price=candidate_price,
                live_price=live_price,
                bid=bid,
                ask=ask,
                spread_pct=spread_pct,
                avg_volume=liquidity.get("volume"),
                dollar_volume=liquidity.get("dollar_volume"),
                liquidity_source=str(liquidity.get("source") or ""),
                quote_source=quote_source,
            )

        watch_decision = self._watch_decision(candidate, ticker, price)
        evidence["watch_decision"] = watch_decision
        if watch_decision.get("skip"):
            return self._research_watch(
                candidate, ticker, rank, str(watch_decision.get("reason") or "active_watch_not_ready"),
                "Ticker already has an active watch zone and is not close enough to justify another buy-now Council slot.",
                route_score - 12,
                funnel_score,
                evidence,
                price=candidate_price,
                live_price=live_price,
                bid=bid,
                ask=ask,
                spread_pct=spread_pct,
                avg_volume=liquidity.get("volume"),
                dollar_volume=liquidity.get("dollar_volume"),
                liquidity_source=str(liquidity.get("source") or ""),
                quote_source=quote_source,
            )

        cooldown = self._recent_cooldown(ticker, price)
        evidence["recent_cooldown"] = cooldown
        if cooldown.get("cooldown"):
            return self._research_watch(
                candidate, ticker, rank, str(cooldown.get("reason_code") or "recent_decision_cooldown"),
                str(cooldown.get("reason") or "Recent Council outcome is still active."),
                route_score - float(cooldown.get("penalty") or 10),
                funnel_score,
                evidence,
                price=candidate_price,
                live_price=live_price,
                bid=bid,
                ask=ask,
                spread_pct=spread_pct,
                avg_volume=liquidity.get("volume"),
                dollar_volume=liquidity.get("dollar_volume"),
                liquidity_source=str(liquidity.get("source") or ""),
                quote_source=quote_source,
            )

        if not self.market_open:
            return self._research_watch(
                candidate, ticker, rank, "market_closed",
                "Market is closed, so buy-now auto-execution feasibility cannot be proven.",
                route_score - 5,
                funnel_score,
                evidence,
                price=candidate_price,
                live_price=live_price,
                bid=bid,
                ask=ask,
                spread_pct=spread_pct,
                avg_volume=liquidity.get("volume"),
                dollar_volume=liquidity.get("dollar_volume"),
                liquidity_source=str(liquidity.get("source") or ""),
                quote_source=quote_source,
            )

        return self._decision(
            candidate,
            ticker,
            rank,
            LANE_EXECUTION_READY,
            "broker_data_clean_for_buy_now_council",
            "Quote, spread, average liquidity, repeat/cooldown, and watch-zone checks are clean enough for a buy-now Council slot.",
            route_score,
            funnel_score,
            evidence,
            price=candidate_price,
            live_price=live_price,
            bid=bid,
            ask=ask,
            spread_pct=spread_pct,
            avg_volume=liquidity.get("volume"),
            dollar_volume=liquidity.get("dollar_volume"),
            liquidity_source=str(liquidity.get("source") or ""),
            quote_source=quote_source,
        )

    def _fetch_quote(self, ticker: str) -> dict[str, Any] | None:
        if self.quote_provider is None or self._quote_checks >= self.max_quote_checks:
            return None
        self._quote_checks += 1
        try:
            quote = self.quote_provider(ticker)
            return quote if isinstance(quote, dict) else None
        except Exception as exc:
            logger.warning("[broker_router] quote provider failed for %s: %s", ticker, exc)
            return None

    def _liquidity(self, candidate: dict[str, Any], quote: dict[str, Any] | None, price: float) -> dict[str, Any]:
        quote = quote if isinstance(quote, dict) else {}
        payload = dict(candidate or {})
        if quote:
            payload["router_quote"] = quote
            payload["yf_quote"] = {**(payload.get("yf_quote") or {}), **quote}
        info = resolve_average_volume(payload)
        volume = _num(info.get("volume"))
        dollar_volume = float(price) * volume if volume and price else None
        return {
            "volume": volume,
            "source": info.get("source"),
            "is_average": bool(info.get("is_average")),
            "dollar_volume": dollar_volume,
            "minimum": Config.ROBINHOOD_MIN_DOLLAR_VOLUME,
        }

    @staticmethod
    def _liquidity_bonus(dollar_volume: float | None) -> float:
        if dollar_volume is None or dollar_volume <= 0:
            return 0.0
        floor = max(Config.ROBINHOOD_MIN_DOLLAR_VOLUME, 1.0)
        return min(8.0, max(0.0, math.log10(max(dollar_volume, floor) / floor) * 4.0))

    def _watch_decision(self, candidate: dict[str, Any], ticker: str, price: float) -> dict[str, Any]:
        watches = self.active_watches.get(ticker)
        if not watches:
            return {"skip": False, "reason": "no_active_watch"}
        if isinstance(watches, dict):
            watches = [watches]
        try:
            from .defer_watchlist import scan_skip_for_defer_watch
        except Exception:
            return {"skip": False, "reason": "watch_logic_unavailable"}

        skip_results: list[dict[str, Any]] = []
        for watch in watches:
            result = scan_skip_for_defer_watch(
                watch,
                price,
                candidate=candidate,
                buffer_pct=Config.SCAN_DEFER_WATCH_SKIP_BUFFER_PCT,
                major_move_pct=Config.SCAN_DEFER_WATCH_SKIP_MAJOR_MOVE_PCT,
            )
            result["watch_id"] = result.get("watch_id") or watch.get("watch_id")
            if not result.get("skip"):
                result["checked_watch_count"] = len(watches)
                return result
            skip_results.append(result)
        if not skip_results:
            return {"skip": False, "reason": "no_valid_watch"}
        best = min(skip_results, key=lambda item: abs(_num(item.get("distance_pct")) or 999999.0))
        best["checked_watch_count"] = len(watches)
        return best

    def _recent_cooldown(self, ticker: str, price: float) -> dict[str, Any]:
        try:
            rows = self.journal.get_recent_recommendations(ticker, limit=8)
        except Exception:
            rows = []
        if not rows:
            return {"cooldown": False, "reason": "no_recent_decision"}

        latest = rows[0]
        action = str(latest.get("action") or "").upper()
        ts = _as_utc(latest.get("timestamp") or latest.get("created_at"))
        age_days = (self.now - ts).total_seconds() / 86400.0 if ts else None
        last_price = _num(latest.get("price_at_recommendation"))
        material_move = False
        move_pct = None
        if last_price and last_price > 0:
            move_pct = (price - last_price) / last_price * 100.0
            material_move = abs(move_pct) >= Config.SCAN_DEFER_WATCH_SKIP_MAJOR_MOVE_PCT

        same_action_count = sum(1 for row in rows if str(row.get("action") or "").upper() == action)
        payload = {
            "cooldown": False,
            "latest_action": action,
            "age_days": age_days,
            "same_action_count": same_action_count,
            "last_price": last_price,
            "move_pct": move_pct,
            "material_move_override": material_move,
        }
        if material_move:
            payload["reason"] = "material move overrides recent-decision cooldown"
            return payload
        if action in _AVOID_ACTIONS and age_days is not None and age_days <= Config.SCAN_ROUTER_AVOID_COOLDOWN_DAYS:
            payload.update(
                {
                    "cooldown": True,
                    "reason_code": "recent_avoid_cooldown",
                    "reason": f"{ticker} had recent {action}; keep in research/watch until new evidence or a material move appears.",
                    "penalty": 16,
                }
            )
        elif action in _WATCH_ACTIONS and age_days is not None and age_days <= Config.SCAN_ROUTER_DEFER_COOLDOWN_DAYS:
            payload.update(
                {
                    "cooldown": True,
                    "reason_code": "recent_defer_cooldown",
                    "reason": f"{ticker} had recent {action}; wait for watch-zone trigger, catalyst, or material move before another buy-now slot.",
                    "penalty": 12,
                }
            )
        elif action in _BUY_SIDE_ACTIONS and same_action_count >= 2 and age_days is not None and age_days <= Config.SCAN_ROUTER_STARTER_COOLDOWN_DAYS:
            payload.update(
                {
                    "cooldown": True,
                    "reason_code": "recent_buy_side_recheck_cooldown",
                    "reason": f"{ticker} already received repeated buy-side Council attention; only re-enter buy-now lane after material move or clean execution trigger.",
                    "penalty": 8,
                }
            )
        return payload

    def _research_watch(
        self,
        candidate: dict[str, Any],
        ticker: str,
        rank: int,
        reason_code: str,
        reason: str,
        route_score: float,
        funnel_score: float | None,
        evidence: dict[str, Any],
        **kwargs: Any,
    ) -> BrokerRouteDecision:
        return self._decision(
            candidate,
            ticker,
            rank,
            LANE_RESEARCH_WATCH,
            reason_code,
            reason,
            route_score,
            funnel_score,
            evidence,
            **kwargs,
        )

    def _decision(
        self,
        candidate: dict[str, Any],
        ticker: str,
        rank: int,
        lane: str,
        reason_code: str,
        reason: str,
        route_score: float,
        funnel_score: float | None,
        evidence: dict[str, Any],
        **kwargs: Any,
    ) -> BrokerRouteDecision:
        if lane == LANE_EXECUTION_READY:
            bucket = BUCKET_BUY_NOW
        elif lane == LANE_HARD_REJECT:
            bucket = BUCKET_REJECT
        else:
            bucket = BUCKET_RESEARCH_WATCH
        return BrokerRouteDecision(
            candidate={**candidate, "broker_router_lane": lane, "broker_router_reason": reason_code},
            ticker=ticker,
            candidate_rank=rank,
            lane=lane,
            bucket=bucket,
            reason_code=reason_code,
            reason=reason,
            route_score=round(float(route_score or 0.0), 2),
            funnel_score=funnel_score,
            price=kwargs.get("price"),
            live_price=kwargs.get("live_price"),
            bid=kwargs.get("bid"),
            ask=kwargs.get("ask"),
            spread_pct=kwargs.get("spread_pct"),
            avg_volume=kwargs.get("avg_volume"),
            dollar_volume=kwargs.get("dollar_volume"),
            liquidity_source=str(kwargs.get("liquidity_source") or ""),
            quote_source=str(kwargs.get("quote_source") or ""),
            evidence={**evidence, "reason_code": reason_code, "reason": reason, "lane": lane, "bucket": bucket},
        )


def route_scan_candidates(
    candidates: list[dict[str, Any]],
    *,
    session_id: str,
    journal: Any,
    active_watches: dict[str, Any] | None = None,
    quote_provider: Callable[[str], dict[str, Any] | None] | None = None,
    market_open: bool = True,
    now: datetime | None = None,
    council_limit: int | None = None,
    persist: bool = True,
    fill_research_slots: bool | None = None,
) -> BrokerRouterResult:
    """Convenience wrapper for scheduled scans and CLI previews."""
    router = BrokerAwareCandidateRouter(
        journal=journal,
        active_watches=active_watches,
        quote_provider=quote_provider,
        market_open=market_open,
        now=now,
    )
    return router.route(
        candidates,
        session_id=session_id,
        council_limit=int(council_limit if council_limit is not None else Config.SCAN_COUNCIL_MAX),
        persist=persist,
        fill_research_slots=fill_research_slots,
    )
