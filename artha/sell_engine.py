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
from .thesis_tracker import ThesisTracker, PositionThesis
from .trailing_stop import TrailingStopManager

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat()


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
                             # regime_change | scale_out | periodic_review | opportunity_cost
    severity: str = "MEDIUM"  # URGENT | HIGH | MEDIUM | LOW
    source: str = ""
    message: str = ""
    sell_score: Optional[float] = None
    action_recommended: Optional[str] = None
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

    def _refresh_count(self) -> None:
        today = _utcnow().date()
        if today != self._exit_count_date:
            self._exit_count_today = 0
            self._exit_count_date = today

    def can_exit(self) -> bool:
        """Return True if another exit is allowed today."""
        self._refresh_count()
        return self._exit_count_today < Config.SELL_MAX_EXITS_PER_DAY

    def record_exit(self) -> None:
        """Record that an exit occurred today."""
        self._refresh_count()
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
        self.trailing_stop_mgr = TrailingStopManager()
        self._collector = collector  # Lazy: set by scheduler

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

        Tasks:
        1. Enforce hard stops for every active thesis
        2. Update trailing stops for TACTICAL_BUY positions
        3. Check trailing stop breaches
        4. Check for scale-out milestones
        5. Check regime change for TACTICAL_BUY

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

        active_theses = self.tracker.get_all_active()

        for thesis in active_theses:
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

            # --- Trailing stop update (TACTICAL_BUY) ---
            if thesis.position_type == "TACTICAL_BUY":
                try:
                    price_history = self._get_price_history(ticker)
                    new_stop, is_breached = self.trailing_stop_mgr.update_position_trailing_stop(
                        thesis=thesis,
                        current_price=current_price,
                        price_history=price_history,
                    )
                    # FIX D: sync trailing stop from thesis DB → portfolio.json so monitor reads current value
                    if new_stop is not None:
                        pos = portfolio.get_position(ticker)
                        if pos is not None and pos.trailing_stop_price != new_stop:
                            pos.trailing_stop_price = new_stop
                            _portfolio_dirty = True
                    if is_breached:
                        signal = SellSignal(
                            ticker=ticker,
                            thesis_id=thesis.thesis_id,
                            signal_type="trailing_stop",
                            severity="URGENT",
                            source="trailing_stop_manager",
                            message=(
                                f"📉 TRAILING STOP TRIGGERED: {ticker} at ${current_price:.2f} "
                                f"breached trailing stop ${new_stop:.2f}. "
                                f"Exit TACTICAL_BUY position (thesis: {thesis.thesis_id[:8]})."
                            ),
                            action_recommended="EXIT",
                        )
                        signals.append(signal)
                        self.aggregator.record(signal)
                except Exception as ts_e:
                    logger.warning("[sell_engine] Trailing stop update failed for %s: %s", ticker, ts_e)

            # --- Scale-out milestone check ---
            if thesis.entry_price and thesis.entry_price > 0:
                signals.extend(self._check_scale_out(thesis, current_price))

            # --- Regime change check for TACTICAL_BUY ---
            if thesis.position_type == "TACTICAL_BUY":
                signals.extend(self._check_regime_change(thesis))

        # FIX D: persist trailing stop changes to portfolio.json
        if _portfolio_dirty:
            try:
                from .portfolio import PORTFOLIO_FILE
                portfolio.save(PORTFOLIO_FILE)
                logger.debug("[sell_engine] Synced trailing stop values to portfolio.json")
            except Exception as save_e:
                logger.warning("[sell_engine] Failed to sync trailing stop to portfolio.json: %s", save_e)

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

    def _get_price_history(self, ticker: str, days: int = 20) -> list[dict]:
        """Fetch recent price history for ATR computation."""
        try:
            data = self.collector.collect_stock(ticker)
            return data.get("price_history") or []
        except Exception:
            return []

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
                )
                signals.append(signal)
                self.aggregator.record(signal)
                # Record milestone to prevent re-alerting
                self.tracker.record_scale_out(thesis.thesis_id, milestone_key)
                logger.info(
                    "[sell_engine] Scale-out milestone %s hit for %s", milestone_key, thesis.ticker
                )

        return signals

    def _check_regime_change(self, thesis: PositionThesis) -> list[SellSignal]:
        """Flag TACTICAL_BUY positions when regime has changed since entry."""
        signals: list[SellSignal] = []
        if not thesis.entry_regime:
            return signals

        try:
            from .regime import RegimePacket
            # Load latest regime from journal or a cached state — this is best-effort
            # The full regime council is expensive; we check the entry regime vs stored state
            regime_state_path = __import__("pathlib").Path(
                __file__
            ).resolve().parent.parent / "data" / "regime_state.json"
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
