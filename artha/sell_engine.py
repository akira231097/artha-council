"""Sell Engine — main orchestrator for all sell-side activity.

Coordinates:
  - ThesisTracker (thesis storage + lifecycle)
  - SellCouncil (3-analyst sell debate)
  - TrailingStopManager (trailing stop updates)
  - SellSignalAggregator (signal collection + routing)
  - Regime integration (entry vs current regime comparison)
  - Portfolio circuit breaker (max 2 exits/day, pause on -10%)

This module is the single entry point called by the scheduler for sell-side work.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional

from .config import Config
from .journal import DecisionJournal
from .paths import DATA_DIR
from .thesis_tracker import (
    ThesisTracker,
    PositionThesis,
    CONDITION_EFFECT_INVALIDATE,
    CONDITION_EFFECT_REVIEW,
    evaluate_invalidation_conditions,
    review_condition_already_acknowledged,
)
from .trailing_stop import TrailingStopManager, TRAIL_MANAGED_POSITION_TYPES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sell signal types (exit engine v2)
# ---------------------------------------------------------------------------
SIGNAL_HARD_STOP = "hard_stop"
SIGNAL_TRAILING_STOP = "trailing_stop"
SIGNAL_DEAD_MONEY_TIME_STOP = "dead_money_time_stop"   # rule 4.5
SIGNAL_PROFIT_TAKE = "profit_take"                     # rule 4.3
SIGNAL_EARNINGS_RISK = "earnings_risk"                 # rule 4.6
SIGNAL_THESIS_BREAK = "thesis_break"                   # rule 4.7
SIGNAL_THESIS_REVIEW = "thesis_review"                 # scheduled re-underwrite
SIGNAL_REGIME_EXIT = "regime_exit"                     # rule 4.7/regime
SIGNAL_REVIEW_EXIT = "review_exit"                     # rule 4.7 material news

# Scale-out markers used to make one-shot rules idempotent
_MARKER_FAST_MOVE_HOLD = "8WK_HOLD"
_MARKER_PROFIT_TAKE = "PT_+20%"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat()


def _trading_days_between(start: datetime, end: datetime) -> int:
    """Approximate trading days (weekdays) between two datetimes."""
    if end <= start:
        return 0
    count = 0
    cursor = start.date()
    end_date = end.date()
    while cursor < end_date:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            count += 1
    return count


def _trading_days_held(thesis: PositionThesis) -> int:
    if not thesis.entry_date:
        return 0
    try:
        entry = datetime.fromisoformat(thesis.entry_date)
        if entry.tzinfo is None:
            entry = entry.replace(tzinfo=timezone.utc)
        return _trading_days_between(entry, _utcnow())
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Sell Signal
# ---------------------------------------------------------------------------

@dataclass
class SellSignal:
    """A sell signal from any monitoring layer."""
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ticker: str = ""
    thesis_id: Optional[str] = None
    signal_type: str = ""   # hard_stop | trailing_stop | thesis_triggered | news_critical |
                             # regime_change | scale_out | periodic_review | opportunity_cost |
                             # dead_money_time_stop | profit_take | earnings_risk |
                             # thesis_break | regime_exit | review_exit
    severity: str = "MEDIUM"  # URGENT | HIGH | MEDIUM | LOW
    source: str = ""
    message: str = ""
    sell_score: Optional[float] = None
    action_recommended: Optional[str] = None
    trim_pct: Optional[float] = None
    completion_markers: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utcnow_iso)


class SellSignalAggregator:
    """Collects sell signals from all sources, dedupes, and routes them."""

    PRIORITY_ORDER = ["URGENT", "HIGH", "MEDIUM", "LOW"]

    def __init__(self, journal: Optional[DecisionJournal] = None) -> None:
        self.journal = journal or DecisionJournal()

    def record(self, signal: SellSignal) -> None:
        """Persist a sell signal."""
        try:
            self.journal.save_sell_signal({
                "signal_id": signal.signal_id,
                "ticker": signal.ticker,
                "thesis_id": signal.thesis_id,
                "signal_type": signal.signal_type,
                "severity": signal.severity,
                "source": signal.source,
                "message": signal.message,
                "sell_score": signal.sell_score,
                "action_recommended": signal.action_recommended,
                "trim_pct": signal.trim_pct,
            })
        except Exception as e:
            logger.warning("[signal_agg] Failed to persist signal: %s", e)

    def get_active(self, ticker: Optional[str] = None) -> list[SellSignal]:
        """Get unactioned signals."""
        rows = self.journal.get_active_sell_signals(ticker)
        signals = []
        for row in rows:
            signals.append(SellSignal(
                signal_id=row.get("signal_id", ""),
                ticker=row.get("ticker", ""),
                thesis_id=row.get("thesis_id"),
                signal_type=row.get("signal_type", ""),
                severity=row.get("severity", "MEDIUM"),
                source=row.get("source", ""),
                message=row.get("message", ""),
                sell_score=row.get("sell_score"),
                action_recommended=row.get("action_recommended"),
                trim_pct=row.get("trim_pct"),
                created_at=row.get("created_at", _utcnow_iso()),
            ))
        # Sort by priority
        def _priority(s: SellSignal) -> int:
            return self.PRIORITY_ORDER.index(s.severity) if s.severity in self.PRIORITY_ORDER else 99
        return sorted(signals, key=_priority)

    def suppress(self, signal_id: str, reason: str = "") -> None:
        """Mark a signal as suppressed (circuit breaker, cooldown, etc.)."""
        try:
            with self.journal._connect() as conn:
                conn.execute(
                    "UPDATE sell_signals SET suppressed = 1, suppressed_reason = ? WHERE signal_id = ?",
                    (reason, signal_id),
                )
                conn.commit()
        except Exception as e:
            logger.warning("[signal_agg] Failed to suppress signal: %s", e)

    def mark_actioned(self, signal_id: str) -> None:
        """Mark a signal as actioned (sent to Telegram)."""
        try:
            with self.journal._connect() as conn:
                conn.execute(
                    "UPDATE sell_signals SET actioned = 1, actioned_at = ? WHERE signal_id = ?",
                    (_utcnow_iso(), signal_id),
                )
                conn.commit()
        except Exception as e:
            logger.warning("[signal_agg] Failed to mark actioned: %s", e)


# ---------------------------------------------------------------------------
# Portfolio Circuit Breaker
# ---------------------------------------------------------------------------

class PortfolioCircuitBreaker:
    """Limits automated sell activity to prevent cascading exits."""

    def __init__(self, journal: Optional[DecisionJournal] = None) -> None:
        self.journal = journal or DecisionJournal()
        self._exit_count_today: int = 0
        self._exit_count_date: date = _utcnow().date()
        self._exit_tickers_today: set[str] = set()

    def _refresh_count(self) -> None:
        today = _utcnow().date()
        if today != self._exit_count_date:
            self._exit_count_today = 0
            self._exit_count_date = today
            self._exit_tickers_today = set()

    def can_exit(self, ticker: Optional[str] = None) -> bool:
        """Return True if another exit is allowed today.

        Re-signals for a ticker already counted today are always allowed —
        the cap limits distinct positions exited per day, not repeat alerts.
        """
        self._refresh_count()
        if ticker and ticker.upper() in self._exit_tickers_today:
            return True
        return self._exit_count_today < Config.SELL_MAX_EXITS_PER_DAY

    def record_exit(self, ticker: Optional[str] = None) -> None:
        """Record that an exit occurred today (deduped per ticker)."""
        self._refresh_count()
        if ticker:
            key = ticker.upper()
            if key in self._exit_tickers_today:
                return
            self._exit_tickers_today.add(key)
        self._exit_count_today += 1

    def is_portfolio_in_drawdown(self) -> bool:
        """Check if portfolio is down >10% today (pauses non-urgent signals)."""
        try:
            from .portfolio import Portfolio, PORTFOLIO_FILE
            from .collector import DataCollector
            portfolio = Portfolio.load(PORTFOLIO_FILE)
            collector = DataCollector()

            total_cost = 0.0
            total_value = 0.0
            for pos in portfolio.positions:
                if not pos.ticker:
                    continue
                try:
                    quote = collector.yf.quote(pos.ticker)
                    price = float(quote.get("price", 0) or 0)
                    prev_close = float(quote.get("previous_close", 0) or 0)
                    if price > 0 and prev_close > 0:
                        shares = float(pos.shares or 0)
                        total_value += shares * price
                        total_cost += shares * prev_close
                except Exception:
                    pass

            if total_cost > 0 and total_value > 0:
                portfolio_move = (total_value - total_cost) / total_cost
                if portfolio_move <= Config.SELL_PORTFOLIO_LOSS_PAUSE_PCT:
                    logger.warning(
                        "[circuit_breaker] Portfolio down %.1f%% today — pausing non-urgent signals",
                        portfolio_move * 100,
                    )
                    return True
        except Exception as e:
            logger.warning("[circuit_breaker] Drawdown check failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Main SellEngine
# ---------------------------------------------------------------------------

class SellEngine:
    """Main sell-side orchestrator — called by scheduler during price checks."""

    def __init__(
        self,
        journal: Optional[DecisionJournal] = None,
        collector: Any = None,
    ) -> None:
        self.journal = journal or DecisionJournal()
        self.tracker = ThesisTracker(journal=self.journal)
        self.aggregator = SellSignalAggregator(journal=self.journal)
        self.circuit_breaker = PortfolioCircuitBreaker(journal=self.journal)
        self.trailing_stop_mgr = TrailingStopManager(tracker=self.tracker)
        self._collector = collector  # Lazy: set by scheduler
        self._earnings_cache: dict[tuple[str, str], Any] = {}  # (ticker, date) → ctx

    @property
    def collector(self) -> Any:
        if self._collector is None:
            from .collector import DataCollector
            self._collector = DataCollector()
        return self._collector

    # ---------------------------------------------------------------- price check integration

    def run_price_check_sell_tasks(
        self,
        portfolio: Any,
        quotes: dict[str, dict],
    ) -> list[SellSignal]:
        """Run sell-side tasks during every 30-min price check.

        Tasks (exit engine v2):
        1. Enforce hard stops for every monitored thesis (active + pending_exit)
        2. Update cushion-gated Chandelier/EMA trailing stops for ALL buy types
        3. Check trailing stop breaches + EMA-close exits (rule 4.4)
        4. Profit-taking into strength with fast-move exception (rule 4.3)
        5. Time stops: dead money + catalyst positions (rule 4.5)
        6. Earnings risk with 1R gap sizing (rule 4.6)
        7. Machine-checkable thesis-break conditions (rule 4.7)
        8. Regime change/exit for TACTICAL_BUY
        9. Scale-out milestones
        10. Portfolio circuit breaker gating (max exits/day, drawdown pause)

        Returns list of SellSignals to be sent as alerts.
        """
        signals: list[SellSignal] = []
        _portfolio_dirty = False  # FIX D: track if portfolio needs saving

        # Expire stale pending theses
        try:
            expired = self.tracker.expire_stale_pending()
            if expired:
                logger.info("[sell_engine] Expired %d stale pending thesis/theses", expired)
        except Exception as e:
            logger.warning("[sell_engine] Stale pending cleanup failed: %s", e)

        # Include pending_exit positions — a position awaiting execution must
        # not free-fall unwatched between the exit call and the actual sell.
        monitored_theses = self.tracker.get_all_monitored()

        for thesis in monitored_theses:
            ticker = thesis.ticker
            if ticker not in quotes:
                continue

            current_price = float(quotes[ticker].get("price", 0) or 0)
            if current_price <= 0:
                continue

            hard_stop_signal = self._check_hard_stop(thesis, current_price)
            if hard_stop_signal:
                signals.append(hard_stop_signal)
                self.aggregator.record(hard_stop_signal)
                continue

            # --- Trailing stop update (all buy-type positions, rule 4.4) ---
            if thesis.position_type in TRAIL_MANAGED_POSITION_TYPES:
                try:
                    price_history = self._get_price_history(ticker)
                    trail = self.trailing_stop_mgr.evaluate_trail(
                        thesis=thesis,
                        current_price=current_price,
                        price_history=price_history,
                    )
                    new_stop = float(trail["new_stop"] or 0)
                    # FIX D: sync trailing stop from thesis DB → portfolio.json so monitor reads current value
                    if new_stop > 0:
                        pos = portfolio.get_position(ticker)
                        if pos is not None and pos.trailing_stop_price != new_stop:
                            pos.trailing_stop_price = new_stop
                            _portfolio_dirty = True
                    if trail["is_breached"]:
                        signal = SellSignal(
                            ticker=ticker,
                            thesis_id=thesis.thesis_id,
                            signal_type=SIGNAL_TRAILING_STOP,
                            severity="URGENT",
                            source="trailing_stop_manager",
                            message=(
                                f"📉 TRAILING STOP TRIGGERED: {ticker} at ${current_price:.2f} "
                                f"breached trailing stop ${new_stop:.2f}. "
                                f"Exit {thesis.position_type} position (thesis: {thesis.thesis_id[:8]})."
                            ),
                            action_recommended="EXIT",
                        )
                        signals.append(signal)
                        self.aggregator.record(signal)
                    elif trail.get("ema_exit"):
                        ema_details = trail.get("ema_details") or {}
                        signal = SellSignal(
                            ticker=ticker,
                            thesis_id=thesis.thesis_id,
                            signal_type=SIGNAL_TRAILING_STOP,
                            severity="HIGH",
                            source="ema_close_trail",
                            message=(
                                f"📉 EMA-CLOSE EXIT: {ticker} closed "
                                f"${ema_details.get('last_close') or 0:.2f}, below its "
                                f"{ema_details.get('ema_period')}-day EMA "
                                f"${ema_details.get('ema') or 0:.2f} with cushion established. "
                                f"Rule 4.4 exit into strength (thesis: {thesis.thesis_id[:8]})."
                            ),
                            action_recommended="EXIT",
                        )
                        signals.append(signal)
                        self.aggregator.record(signal)
                except Exception as ts_e:
                    logger.warning("[sell_engine] Trailing stop update failed for %s: %s", ticker, ts_e)

            # Positions already awaiting sell execution only need stop/trail
            # protection — no new lifecycle signals.
            if thesis.is_waiting_to_sell:
                continue

            if thesis.entry_price and thesis.entry_price > 0:
                # --- Profit-taking / fast-move exception (rule 4.3) ---
                signals.extend(self._check_profit_take(thesis, current_price))

                # --- Time stops: dead money + catalyst (rule 4.5) ---
                signals.extend(self._check_dead_money(thesis, current_price))

                # --- Earnings risk (rule 4.6) ---
                signals.extend(self._check_earnings_risk(thesis, current_price))

                # --- Machine-checkable thesis-break conditions (rule 4.7) ---
                signals.extend(self._check_thesis_break(thesis, current_price))

                # --- Scale-out milestone check ---
                signals.extend(self._check_scale_out(thesis, current_price))

            # --- Regime change check for TACTICAL_BUY ---
            if thesis.position_type == "TACTICAL_BUY":
                signals.extend(self._check_regime_change(thesis, current_price))

        # FIX D: persist trailing stop changes to portfolio.json
        if _portfolio_dirty:
            try:
                from .portfolio import PORTFOLIO_FILE
                portfolio.save(PORTFOLIO_FILE)
                logger.debug("[sell_engine] Synced trailing stop values to portfolio.json")
            except Exception as save_e:
                logger.warning("[sell_engine] Failed to sync trailing stop to portfolio.json: %s", save_e)

        # --- Portfolio circuit breaker gating ---
        signals = self._apply_circuit_breaker(signals)

        return signals

    def _apply_circuit_breaker(self, signals: list[SellSignal]) -> list[SellSignal]:
        """Annotate portfolio stress without suppressing Sell Council review.

        A drawdown is relevant evidence for Sell Council, but it must not make
        a held position disappear from the review pipeline. The actual daily
        exit cap is enforced from filled broker orders at the final auto-sell
        authorization gate, where it reflects real executions rather than
        generated alerts.
        """
        if not signals:
            return signals

        non_urgent_exits = [
            s for s in signals
            if s.severity != "URGENT" and (s.action_recommended or "").upper() in {"EXIT", "SELL"}
        ]
        in_drawdown = False
        if non_urgent_exits:
            try:
                in_drawdown = self.circuit_breaker.is_portfolio_in_drawdown()
            except Exception as cb_e:
                logger.warning("[sell_engine] Drawdown check failed: %s", cb_e)

        for signal in signals:
            is_exit = (signal.action_recommended or "").upper() in {"EXIT", "SELL"}
            if in_drawdown and is_exit and signal.severity != "URGENT":
                context = (
                    "Portfolio drawdown guard is active; this is review context only. "
                    "Sell Council must independently confirm any exit."
                )
                signal.message = f"{signal.message} | {context}" if signal.message else context
                logger.warning(
                    "[sell_engine] Passing %s exit for %s to Sell Council with drawdown context",
                    signal.signal_type, signal.ticker,
                )
        return signals

    def _check_hard_stop(
        self,
        thesis: PositionThesis,
        current_price: float,
    ) -> Optional[SellSignal]:
        """Return an urgent hard-stop signal when any active thesis breaches its stop."""
        stop = float(thesis.hard_stop_price or 0)
        if stop <= 0 or current_price > stop:
            return None

        entry = float(thesis.entry_price or 0)
        pnl_pct = (current_price - entry) / entry if entry > 0 else 0.0
        return SellSignal(
            ticker=thesis.ticker,
            thesis_id=thesis.thesis_id,
            signal_type="hard_stop",
            severity="URGENT",
            source="sell_engine",
            message=(
                f"🚨 HARD STOP TRIGGERED: {thesis.ticker} at ${current_price:.2f} "
                f"is at/below hard stop ${stop:.2f}. "
                f"Position type: {thesis.position_type}. "
                f"P&L from thesis entry: {pnl_pct:+.1%}. "
                "Prepare exit review immediately."
            ),
            action_recommended="EXIT",
            sell_score=100.0,
        )

    def _get_price_history(self, ticker: str, period: str = "6mo") -> list[dict]:
        """Fetch recent daily OHLCV history for ATR/EMA computation.

        Uses the lightweight FMP history endpoint (single call) with a
        yfinance fallback instead of the full ``collect_stock`` pipeline —
        this now runs for every monitored position on every price check.
        """
        try:
            rows = self.collector.fmp.history(ticker, period)
            if rows and len(rows) >= 20:
                return rows
        except Exception as fmp_e:
            logger.debug("[sell_engine] FMP history failed for %s: %s", ticker, fmp_e)
        try:
            rows = self.collector.yf.history(ticker, period)
            return rows or []
        except Exception:
            return []

    # ------------------------------------------------------------------ rule 4.3
    def _check_profit_take(
        self,
        thesis: PositionThesis,
        current_price: float,
    ) -> list[SellSignal]:
        """Rule 4.3 profit-taking: +20%+ gains taking >3 weeks are sold into
        strength (50-100%); gains of +20% within 15 trading days suppress
        profit-taking (8-week-hold tag, disaster stop + trail only)."""
        signals: list[SellSignal] = []
        entry = float(thesis.entry_price or 0)
        if entry <= 0:
            return signals

        gain_pct = (current_price - entry) / entry
        if gain_pct < Config.SELL_PROFIT_TAKE_MIN_GAIN_PCT:
            return signals

        completed = thesis.scale_out_completed or []
        tdays = _trading_days_held(thesis)

        # Fast-move exception: +20% within 15 trading days → tag for 8-week
        # hold; only the disaster stop and the trail manage the position.
        if tdays <= Config.SELL_FAST_MOVE_TRADING_DAYS:
            if _MARKER_FAST_MOVE_HOLD not in completed:
                self.tracker.record_scale_out(thesis.thesis_id, _MARKER_FAST_MOVE_HOLD)
                signal = SellSignal(
                    ticker=thesis.ticker,
                    thesis_id=thesis.thesis_id,
                    signal_type=SIGNAL_PROFIT_TAKE,
                    severity="LOW",
                    source="sell_engine",
                    message=(
                        f"🚀 FAST MOVE: {thesis.ticker} gained {gain_pct:+.1%} within "
                        f"{tdays} trading days. Profit-taking SUPPRESSED — tagged "
                        f"{_MARKER_FAST_MOVE_HOLD} (rule 4.3 exception): hold "
                        f"~{Config.SELL_FAST_MOVE_HOLD_TRADING_DAYS} trading days with "
                        "disaster stop + trail only."
                    ),
                    action_recommended="HOLD",
                )
                signals.append(signal)
                self.aggregator.record(signal)
                logger.info(
                    "[sell_engine] Fast-move exception tagged for %s (+%.1f%% in %dtd)",
                    thesis.ticker, gain_pct * 100, tdays,
                )
            return signals

        # Still inside a tagged 8-week hold window → no profit-taking yet
        if (
            _MARKER_FAST_MOVE_HOLD in completed
            and tdays < Config.SELL_FAST_MOVE_HOLD_TRADING_DAYS
        ):
            return signals

        if _MARKER_PROFIT_TAKE in completed:
            return signals

        signal = SellSignal(
            ticker=thesis.ticker,
            thesis_id=thesis.thesis_id,
            signal_type=SIGNAL_PROFIT_TAKE,
            severity="HIGH",
            source="sell_engine",
            message=(
                f"💰 PROFIT-TAKE (rule 4.3): {thesis.ticker} is {gain_pct:+.1%} from "
                f"${entry:.2f} after {tdays} trading days (>3 weeks). "
                "Sell 50-100% into strength."
            ),
            action_recommended="TRIM",
            sell_score=float(Config.SELL_SCORE_TRIM_THRESHOLD),
            completion_markers=[_MARKER_PROFIT_TAKE],
        )
        signals.append(signal)
        self.aggregator.record(signal)
        logger.info("[sell_engine] Profit-take signal for %s (+%.1f%%)", thesis.ticker, gain_pct * 100)
        return signals

    # ------------------------------------------------------------------ rule 4.5
    def _check_dead_money(
        self,
        thesis: PositionThesis,
        current_price: float,
    ) -> list[SellSignal]:
        """Rule 4.5 time stops.

        - Any position stuck between -3% and +5% after 20 trading days →
          DEAD_MONEY_TIME_STOP exit signal.
        - Catalyst positions (TACTICAL_BUY) below entry after 15 trading
          days → exit.
        """
        signals: list[SellSignal] = []
        entry = float(thesis.entry_price or 0)
        if entry <= 0:
            return signals

        pnl_pct = (current_price - entry) / entry
        tdays = _trading_days_held(thesis)

        is_dead_money = (
            tdays >= Config.SELL_DEAD_MONEY_TRADING_DAYS
            and Config.SELL_DEAD_MONEY_MIN_PNL_PCT <= pnl_pct <= Config.SELL_DEAD_MONEY_MAX_PNL_PCT
        )
        is_stalled_catalyst = (
            thesis.position_type == "TACTICAL_BUY"
            and tdays >= Config.SELL_CATALYST_TIME_STOP_TRADING_DAYS
            and pnl_pct < 0
        )

        if is_stalled_catalyst:
            message = (
                f"⏱ CATALYST TIME STOP (rule 4.5): {thesis.ticker} TACTICAL_BUY is "
                f"{pnl_pct:+.1%} after {tdays} trading days — the catalyst has not "
                "worked. Exit and recycle capital."
            )
        elif is_dead_money:
            message = (
                f"⏱ DEAD MONEY (rule 4.5): {thesis.ticker} is {pnl_pct:+.1%} after "
                f"{tdays} trading days (flat between "
                f"{Config.SELL_DEAD_MONEY_MIN_PNL_PCT:+.0%} and "
                f"{Config.SELL_DEAD_MONEY_MAX_PNL_PCT:+.0%}). Time stop — exit and "
                "redeploy into a working idea."
            )
        else:
            return signals

        signal = SellSignal(
            ticker=thesis.ticker,
            thesis_id=thesis.thesis_id,
            signal_type=SIGNAL_DEAD_MONEY_TIME_STOP,
            severity="HIGH",
            source="sell_engine",
            message=message,
            action_recommended="EXIT",
            sell_score=float(Config.SELL_SCORE_EXIT_TACTICAL),
        )
        signals.append(signal)
        self.aggregator.record(signal)
        logger.info(
            "[sell_engine] Time-stop signal for %s (pnl=%+.1f%%, %dtd)",
            thesis.ticker, pnl_pct * 100, tdays,
        )
        return signals

    # ------------------------------------------------------------------ rule 4.6
    def _get_earnings_context_cached(self, ticker: str) -> Any:
        """Fetch earnings context at most once per ticker per day."""
        today = _utcnow().date().isoformat()
        cache_key = (ticker, today)
        if cache_key in self._earnings_cache:
            return self._earnings_cache[cache_key]
        ctx = None
        try:
            from .earnings_calendar import get_earnings_context
            ctx = get_earnings_context(ticker)
        except Exception as e:
            logger.debug("[sell_engine] Earnings context failed for %s: %s", ticker, e)
        # Drop stale entries for this ticker
        self._earnings_cache = {
            k: v for k, v in self._earnings_cache.items() if k[1] == today
        }
        self._earnings_cache[cache_key] = ctx
        return ctx

    def _check_earnings_risk(
        self,
        thesis: PositionThesis,
        current_price: float,
    ) -> list[SellSignal]:
        """Rule 4.6: scheduled earnings ahead with unrealized gain < +10% →
        TRIM/EXIT flag sized so a -15% gap costs <= 1R. Cushion >= +10% →
        hold through with trail (no signal)."""
        signals: list[SellSignal] = []
        entry = float(thesis.entry_price or 0)
        if entry <= 0:
            return signals

        ctx = self._get_earnings_context_cached(thesis.ticker)
        days_to_earnings = getattr(ctx, "days_to_earnings", None) if ctx else None
        if days_to_earnings is None or not (
            0 <= int(days_to_earnings) <= Config.SELL_EARNINGS_RISK_LOOKAHEAD_DAYS
        ):
            return signals

        gain_pct = (current_price - entry) / entry
        if gain_pct >= Config.SELL_EARNINGS_MIN_CUSHION_PCT:
            logger.debug(
                "[sell_engine] %s holds through earnings with %+.1f%% cushion (trail active)",
                thesis.ticker, gain_pct * 100,
            )
            return signals

        # Size the keep-fraction so a -15% gap on retained shares costs <= 1R
        hard_stop = float(thesis.hard_stop_price or 0)
        r_per_share = entry - hard_stop if 0 < hard_stop < entry else entry * Config.SELL_INITIAL_STOP_PCT
        gap_loss_per_share = Config.SELL_EARNINGS_GAP_RISK_PCT * current_price
        keep_fraction = min(1.0, r_per_share / gap_loss_per_share) if gap_loss_per_share > 0 else 1.0
        trim_fraction = max(0.0, 1.0 - keep_fraction)

        action = "TRIM" if trim_fraction < 0.99 else "EXIT"
        signal = SellSignal(
            ticker=thesis.ticker,
            thesis_id=thesis.thesis_id,
            signal_type=SIGNAL_EARNINGS_RISK,
            severity="HIGH",
            source="sell_engine",
            message=(
                f"📅 EARNINGS RISK (rule 4.6): {thesis.ticker} reports in "
                f"{int(days_to_earnings)} day(s) ({getattr(ctx, 'earnings_date', None) or '?'}) with only "
                f"{gain_pct:+.1%} cushion (<{Config.SELL_EARNINGS_MIN_CUSHION_PCT:.0%}). "
                f"{action} ~{trim_fraction:.0%} so a "
                f"-{Config.SELL_EARNINGS_GAP_RISK_PCT:.0%} gap costs <= 1R "
                f"(R=${r_per_share:.2f}/share)."
            ),
            action_recommended=action,
            sell_score=float(Config.SELL_SCORE_TRIM_THRESHOLD),
            trim_pct=trim_fraction if action == "TRIM" else None,
        )
        signals.append(signal)
        self.aggregator.record(signal)
        logger.info(
            "[sell_engine] Earnings-risk signal for %s (%d days out, cushion %+.1f%%)",
            thesis.ticker, int(days_to_earnings), gain_pct * 100,
        )
        return signals

    # ------------------------------------------------------------------ rule 4.7
    def _check_thesis_break(
        self,
        thesis: PositionThesis,
        current_price: float,
        extra_metrics: Optional[dict[str, Any]] = None,
    ) -> list[SellSignal]:
        """Rule 4.7: evaluate machine-checkable invalidation conditions.

        Structured conditions ({"metric", "op", "value"}) are checked against
        observed metrics; legacy free-text conditions are skipped here (they
        remain council-review material).
        """
        signals: list[SellSignal] = []
        structured = thesis.structured_conditions
        if not structured:
            return signals

        entry = float(thesis.entry_price or 0)
        metrics: dict[str, Any] = {
            "price": current_price,
            "pnl_pct": ((current_price - entry) / entry * 100) if entry > 0 else None,
            "days_held": thesis.days_held,
            "trading_days_held": _trading_days_held(thesis),
        }
        if extra_metrics:
            metrics.update(extra_metrics)

        results = evaluate_invalidation_conditions(structured, metrics)
        triggered = [r for r in results if r["status"] == "triggered"]
        invalidations = [
            result for result in triggered
            if result.get("effect") == CONDITION_EFFECT_INVALIDATE
        ]
        reviews = [
            result for result in triggered
            if result.get("effect") == CONDITION_EFFECT_REVIEW
            and not review_condition_already_acknowledged(
                result.get("condition"),
                entry_date=thesis.entry_date,
                last_review_date=thesis.last_review_date,
            )
        ]
        if not invalidations and not reviews:
            return signals

        if invalidations:
            details = "; ".join(
                f"{r['metric']} {r['condition'].get('op')} {r['threshold']} "
                f"(observed {r['observed']})"
                for r in invalidations
            )
            signal = SellSignal(
                ticker=thesis.ticker,
                thesis_id=thesis.thesis_id,
                signal_type=SIGNAL_THESIS_BREAK,
                severity="HIGH",
                source="invalidation_checker",
                message=(
                    f"🧨 THESIS BREAK (rule 4.7): {thesis.ticker} triggered "
                    f"{len(invalidations)} machine-checked invalidation condition(s): {details}. "
                    "Exit review required."
                ),
                action_recommended="EXIT",
                sell_score=float(Config.SELL_SCORE_EXIT_TACTICAL),
            )
            signals.append(signal)
            self.aggregator.record(signal)
            logger.info(
                "[sell_engine] Thesis-break signal for %s: %s", thesis.ticker, details
            )
            return signals

        details = "; ".join(
            f"{r['metric']} {r['condition'].get('op')} {r['threshold']} "
            f"(observed {r['observed']})"
            for r in reviews
        )
        signal = SellSignal(
            ticker=thesis.ticker,
            thesis_id=thesis.thesis_id,
            signal_type=SIGNAL_THESIS_REVIEW,
            severity="MEDIUM",
            source="review_schedule",
            message=(
                f"🗓 THESIS REVIEW DUE: {thesis.ticker} reached "
                f"{len(reviews)} scheduled review checkpoint(s): {details}. "
                "This is not evidence that the thesis failed; fresh Council review is required."
            ),
            action_recommended="HOLD",
        )
        signals.append(signal)
        self.aggregator.record(signal)
        logger.info(
            "[sell_engine] Thesis-review checkpoint for %s: %s", thesis.ticker, details
        )
        return signals

    def check_fy1_consensus_cut(
        self,
        thesis: PositionThesis,
        analyst_estimates: Optional[dict] = None,
    ) -> Optional[SellSignal]:
        """Rule 4.7: FY1 EPS consensus cut >= 1% over 4 weeks → exit signal
        for catalyst (TACTICAL_BUY) positions.

        Maintains a small local history file of FY1 EPS consensus per ticker
        (data/fy1_consensus_history.json) and compares today's value against
        the oldest sample inside the 4-week window. Intended to be called from
        the daily review path where ``analyst_estimates`` is already collected.
        """
        ticker = thesis.ticker
        fy1_eps = self._extract_fy1_eps(analyst_estimates)
        change_pct = self._update_fy1_history(ticker, fy1_eps)
        if change_pct is None:
            return None
        if thesis.position_type != "TACTICAL_BUY":
            return None
        if change_pct > -Config.SELL_FY1_CUT_EXIT_THRESHOLD_PCT:
            return None

        signal = SellSignal(
            ticker=ticker,
            thesis_id=thesis.thesis_id,
            signal_type=SIGNAL_THESIS_BREAK,
            severity="HIGH",
            source="fy1_consensus_tracker",
            message=(
                f"🧨 FY1 CONSENSUS CUT (rule 4.7): {ticker} FY1 EPS consensus cut "
                f"{change_pct:+.1f}% over the last "
                f"{Config.SELL_FY1_CUT_WINDOW_DAYS} days (threshold "
                f"-{Config.SELL_FY1_CUT_EXIT_THRESHOLD_PCT:.1f}%). Catalyst position — exit."
            ),
            action_recommended="EXIT",
            sell_score=float(Config.SELL_SCORE_EXIT_TACTICAL),
        )
        self.aggregator.record(signal)
        return signal

    @staticmethod
    def _extract_fy1_eps(analyst_estimates: Optional[dict]) -> Optional[float]:
        """Pull the forward-FY EPS consensus from a get_analyst_estimates payload."""
        if not isinstance(analyst_estimates, dict):
            return None
        annual = analyst_estimates.get("annual_estimates") or []
        today = _utcnow().date().isoformat()
        future = [
            row for row in annual
            if isinstance(row, dict) and str(row.get("date") or "") >= today
            and row.get("estimated_eps_avg") is not None
        ]
        # annual_estimates is sorted newest-first; FY1 = nearest future FY end
        candidates = sorted(future, key=lambda r: str(r.get("date")))
        if candidates:
            try:
                return float(candidates[0]["estimated_eps_avg"])
            except (TypeError, ValueError):
                return None
        return None

    def _update_fy1_history(self, ticker: str, fy1_eps: Optional[float]) -> Optional[float]:
        """Append today's FY1 EPS sample and return the pct change over the window."""
        import json
        from pathlib import Path

        history_path = (
            DATA_DIR / "fy1_consensus_history.json"
        )
        try:
            history = {}
            if history_path.exists():
                with open(history_path, encoding="utf-8") as f:
                    history = json.load(f) or {}
            samples = history.get(ticker) or []
            today = _utcnow().date().isoformat()
            window_start = (
                _utcnow() - timedelta(days=Config.SELL_FY1_CUT_WINDOW_DAYS)
            ).date().isoformat()

            if fy1_eps is not None and not any(s.get("date") == today for s in samples):
                samples.append({"date": today, "fy1_eps": fy1_eps})
            samples = [s for s in samples if str(s.get("date") or "") >= window_start]
            samples.sort(key=lambda s: str(s.get("date") or ""))
            history[ticker] = samples
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

            if fy1_eps is None or len(samples) < 2:
                return None
            baseline = samples[0].get("fy1_eps")
            if not baseline:
                return None
            return (fy1_eps - float(baseline)) / abs(float(baseline)) * 100.0
        except Exception as e:
            logger.warning("[sell_engine] FY1 history update failed for %s: %s", ticker, e)
            return None

    def flag_material_news_review(
        self,
        ticker: str,
        headline: str,
        source: str = "news_sentinel",
        evidence: str = "",
    ) -> Optional[SellSignal]:
        """Rule 4.7: strongly negative material news day → REVIEW_EXIT signal.

        Hook for the sentinel/scheduler news path (wired in Wave 2).
        """
        thesis = self.tracker.get_active(ticker)
        if not thesis:
            return None
        normalized_headline = " ".join(str(headline or "").lower().split())[:200]
        for existing in self.aggregator.get_active(ticker):
            if (
                existing.signal_type == SIGNAL_REVIEW_EXIT
                and str(existing.source or "") == str(source or "")
                and normalized_headline
                and normalized_headline in " ".join(str(existing.message or "").lower().split())
            ):
                return existing
        evidence_text = f" Verified impact: {str(evidence).strip()[:500]}" if evidence else ""
        signal = SellSignal(
            ticker=ticker,
            thesis_id=thesis.thesis_id,
            signal_type=SIGNAL_REVIEW_EXIT,
            severity="HIGH",
            source=source,
            message=(
                f"📰 REVIEW_EXIT (rule 4.7): strongly negative material news for "
                f"{ticker}: {headline[:200]}. Run immediate sell review.{evidence_text}"
            ),
            action_recommended="REVIEW",
            sell_score=float(Config.SELL_SCORE_TRIM_THRESHOLD),
        )
        self.aggregator.record(signal)
        return signal

    def _check_scale_out(
        self,
        thesis: PositionThesis,
        current_price: float,
    ) -> list[SellSignal]:
        """Check if a scale-out milestone has been hit."""
        signals: list[SellSignal] = []
        entry = float(thesis.entry_price or 0)
        if entry <= 0:
            return signals

        gain_pct = (current_price - entry) / entry

        # Scale-out schedules per position type
        scale_schedules = {
            "BUY": Config.SELL_SCALE_OUT_BUY,
            "TACTICAL_BUY": Config.SELL_SCALE_OUT_TACTICAL,
            "STARTER": Config.SELL_SCALE_OUT_STARTER,
        }
        schedule = scale_schedules.get(thesis.position_type, {})
        completed = thesis.scale_out_completed or []

        for milestone_key, trim_pct in schedule.items():
            threshold = float(milestone_key.strip("+%")) / 100
            if gain_pct >= threshold and milestone_key not in completed:
                signal = SellSignal(
                    ticker=thesis.ticker,
                    thesis_id=thesis.thesis_id,
                    signal_type="scale_out",
                    severity="MEDIUM",
                    source="sell_engine",
                    message=(
                        f"📊 SCALE-OUT MILESTONE: {thesis.ticker} hit {milestone_key} gain "
                        f"(current: {gain_pct:+.1%} from ${entry:.2f}). "
                        f"Recommend trimming {trim_pct:.0%} of position."
                    ),
                    action_recommended="TRIM",
                    sell_score=Config.SELL_SCORE_TRIM_THRESHOLD,
                    trim_pct=float(trim_pct),
                    completion_markers=[milestone_key],
                )
                signals.append(signal)
                self.aggregator.record(signal)
                logger.info(
                    "[sell_engine] Scale-out milestone %s detected for %s; "
                    "completion waits for a broker-confirmed trim",
                    milestone_key,
                    thesis.ticker,
                )

        return signals

    def _check_regime_change(
        self,
        thesis: PositionThesis,
        current_price: float = 0.0,
    ) -> list[SellSignal]:
        """Flag TACTICAL_BUY positions when regime has changed since entry.

        A persisted regime change (>=3 days) escalates to a REGIME_EXIT signal
        when the tactical position has no meaningful gain to protect; positions
        already working (+5% or better) get the softer regime_change review.
        """
        signals: list[SellSignal] = []
        if not thesis.entry_regime:
            return signals

        try:
            from .regime import RegimePacket
            # Load latest regime from journal or a cached state — this is best-effort
            # The full regime council is expensive; we check the entry regime vs stored state
            regime_state_path = DATA_DIR / "regime_state.json"
            if not regime_state_path.exists():
                return signals

            import json
            with open(regime_state_path) as f:
                state = json.load(f)

            current_regime = state.get("regime", "unknown")
            if current_regime and current_regime != thesis.entry_regime:
                # Has the change persisted? Check date
                regime_changed_at = state.get("changed_at", "")
                if regime_changed_at:
                    changed_dt = datetime.fromisoformat(regime_changed_at)
                    if changed_dt.tzinfo is None:
                        changed_dt = changed_dt.replace(tzinfo=timezone.utc)
                    days_changed = (_utcnow() - changed_dt).days
                    if days_changed >= 3:
                        entry = float(thesis.entry_price or 0)
                        pnl_pct = (
                            (current_price - entry) / entry
                            if entry > 0 and current_price > 0
                            else None
                        )
                        if pnl_pct is not None and pnl_pct < 0.05:
                            signal = SellSignal(
                                ticker=thesis.ticker,
                                thesis_id=thesis.thesis_id,
                                signal_type=SIGNAL_REGIME_EXIT,
                                severity="HIGH",
                                source="sell_engine",
                                message=(
                                    f"⚠️ REGIME EXIT: {thesis.ticker} TACTICAL_BUY entered in "
                                    f"'{thesis.entry_regime}' regime; now '{current_regime}' for "
                                    f"{days_changed} days with only {pnl_pct:+.1%} to protect. "
                                    "The regime that justified this trade is gone — exit."
                                ),
                                action_recommended="EXIT",
                                sell_score=float(Config.SELL_SCORE_EXIT_TACTICAL),
                            )
                        else:
                            signal = SellSignal(
                                ticker=thesis.ticker,
                                thesis_id=thesis.thesis_id,
                                signal_type="regime_change",
                                severity="HIGH",
                                source="sell_engine",
                                message=(
                                    f"⚠️ REGIME CHANGE: {thesis.ticker} TACTICAL_BUY entered in "
                                    f"'{thesis.entry_regime}' regime. Now '{current_regime}' for "
                                    f"{days_changed} days. Review thesis assumptions."
                                ),
                                action_recommended="REVIEW",
                                sell_score=float(Config.SELL_REGIME_MISMATCH_TACTICAL_BONUS),
                            )
                        signals.append(signal)
                        self.aggregator.record(signal)
        except Exception as e:
            logger.debug("[sell_engine] Regime change check failed: %s", e)

        return signals

    # ---------------------------------------------------------------- public helpers

    def get_active_thesis(self, ticker: str) -> Optional[PositionThesis]:
        """Used by the scheduler's thesis impact assessment."""
        return self.tracker.get_active(ticker)

    def get_position_health_summary(self) -> list[dict]:
        """Generate health summary for all active positions."""
        active = self.tracker.get_all_active()
        summary = []
        for thesis in active:
            now_iso = _utcnow_iso()
            next_review = thesis.next_review_date or ""
            days_to_review = 0
            if next_review:
                try:
                    review_dt = datetime.fromisoformat(next_review)
                    if review_dt.tzinfo is None:
                        review_dt = review_dt.replace(tzinfo=timezone.utc)
                    days_to_review = max(0, (review_dt - _utcnow()).days)
                except Exception:
                    pass

            summary.append({
                "ticker": thesis.ticker,
                "position_type": thesis.position_type,
                "entry_price": thesis.entry_price,
                "entry_date": thesis.entry_date,
                "days_held": thesis.days_held,
                "thesis_health_score": thesis.thesis_health_score,
                "hard_stop_price": thesis.hard_stop_price,
                "trailing_stop_price": thesis.trailing_stop_price,
                "next_review_date": thesis.next_review_date,
                "days_to_review": days_to_review,
                "in_cooldown": thesis.in_cooldown,
                "in_minimum_hold": thesis.in_minimum_hold,
                "scale_out_completed": thesis.scale_out_completed,
                "thesis_summary": (thesis.thesis_summary or "")[:200],
            })
        return summary

    def format_health_report(self) -> str:
        """Format a position health report for Telegram."""
        summary = self.get_position_health_summary()
        if not summary:
            return "📊 No active thesis-tracked positions."

        lines = [
            "📊 POSITION HEALTH REPORT",
            f"{'━' * 24}",
            "",
        ]
        for pos in summary:
            health = pos["thesis_health_score"]
            health_emoji = "🟢" if health >= 80 else "🟡" if health >= 60 else "🟠" if health >= 40 else "🔴"
            review_note = (
                f"⏰ Review in {pos['days_to_review']}d"
                if pos["days_to_review"] <= 7
                else f"Next review: {(pos['next_review_date'] or 'N/A')[:10]}"
            )
            lines.extend([
                f"{health_emoji} **{pos['ticker']}** ({pos['position_type']})",
                f"  Health: {health}/100 | Days held: {pos['days_held']}",
                f"  Hard stop: ${pos['hard_stop_price'] or 0:.2f} | {review_note}",
            ])
            if pos.get("in_cooldown"):
                lines.append("  ⏸ In sell cooldown")
            if pos.get("scale_out_completed"):
                lines.append(f"  ✂️ Scale-out done: {', '.join(pos['scale_out_completed'])}")
            lines.append("")

        return "\n".join(lines)
