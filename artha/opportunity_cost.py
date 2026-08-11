"""Opportunity cost scanner — find rotation candidates.

Compares the weakest held position against the highest-scoring new candidate
from the latest scan. If the opportunity delta ≥ SELL_ROTATE_MIN_DELTA, generates
a ROTATE signal for consideration by the sell council.

Conviction lock prevents rotating out of healthy conviction positions:
  - BUY/ACCUMULATE with health_score >= SELL_CONVICTION_LOCK_MIN_HEALTH
  - AND days_held < SELL_CONVICTION_LOCK_MAX_DAYS

Post-sell shadow tracking:
  - Nightly review fetches current price for recently sold positions
  - Computes regret_score (negative = price fell → exit was correct)
  - Records at 5/20/60-day checkpoints for accuracy grading; overdue
    checkpoints are back-filled with the HISTORICAL close for the checkpoint
    date (not today's price)
  - Stop-triggered exits are graded against the actual 20-day-later price
    (a stop that whipsawed below a recovering price is NOT auto-correct)
  - MFE give-back (max favorable excursion vs realized exit) makes
    held-too-long measurable
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from .config import Config
from .journal import DecisionJournal
from .thesis_tracker import ThesisTracker, PositionThesis

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RotationRecommendation:
    """Rotate from weakest position into best new candidate."""
    from_ticker: str
    to_ticker: str
    from_health: int
    to_score: float
    delta: float
    from_position_type: str
    rationale: str
    conviction_locked: bool = False     # True means the "from" position is locked


# ---------------------------------------------------------------------------
# Post-sell shadow tracking
# ---------------------------------------------------------------------------

class PostSellTracker:
    """Tracks prices after sells to measure decision quality."""

    CHECKPOINT_DAYS = Config.SELL_SHADOW_TRACKING_DAYS  # [5, 20, 60]

    # Tolerance (days) within which today's live price is an acceptable
    # stand-in for a checkpoint price; beyond it we fetch the historical close.
    CHECKPOINT_TOLERANCE_DAYS = 2

    def __init__(self, journal: Optional[DecisionJournal] = None) -> None:
        self.journal = journal or DecisionJournal()
        self._ensure_mfe_columns()

    def _ensure_mfe_columns(self) -> None:
        """Add MFE give-back columns to post_sell_tracking if missing."""
        try:
            with self.journal._connect() as conn:
                cols = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(post_sell_tracking)").fetchall()
                }
                for col in ("mfe_pct", "giveback_pct"):
                    if col not in cols:
                        conn.execute(
                            f"ALTER TABLE post_sell_tracking ADD COLUMN {col} REAL"
                        )
                conn.commit()
        except Exception as e:
            logger.warning("[post_sell] Failed to ensure MFE columns: %s", e)

    def record_sell(
        self,
        ticker: str,
        thesis_id: str,
        sell_price: float,
        sell_reason: str,
        shares: float,
        position_type: str,
        *,
        tracking_id: str | None = None,
        sell_date: str | None = None,
        broker_order_id: str | None = None,
        order_intent_id: str | None = None,
        sell_event_id: str | None = None,
        cost_basis: float | None = None,
        realized_pnl: float | None = None,
        data_quality: str = "broker_fill",
    ) -> str:
        """Record a new post-sell tracking entry."""
        import uuid
        tracking_id = str(tracking_id or uuid.uuid4())
        existing = self.journal.get_post_sell_review(tracking_id)
        payload = {
            "tracking_id": tracking_id,
            "ticker": ticker,
            "thesis_id": thesis_id,
            "sell_date": str(sell_date or _utcnow_iso())[:10],
            "sell_price": sell_price,
            "sell_reason": sell_reason,
            "position_type": position_type,
            "shares": shares,
            "broker_order_id": broker_order_id,
            "order_intent_id": order_intent_id,
            "sell_event_id": sell_event_id,
            "cost_basis": cost_basis,
            "realized_pnl": realized_pnl,
            "data_quality": data_quality,
        }
        if not existing:
            payload["status"] = "tracking"
        self.journal.save_post_sell_tracking(payload)
        logger.info("[post_sell] Started shadow tracking for %s @ $%.2f", ticker, sell_price)
        return tracking_id

    def _historical_close_on_or_after(
        self,
        collector: Any,
        ticker: str,
        target_date: date,
    ) -> Optional[float]:
        """Fetch the daily close on (or first trading day after) a past date."""
        try:
            rows = collector.fmp.history(ticker, "1y") or []
        except Exception as e:
            logger.warning("[post_sell] History fetch failed for %s: %s", ticker, e)
            rows = []
        if not rows:
            try:
                rows = collector.yf.history(ticker, "1y") or []
            except Exception:
                rows = []
        target_iso = target_date.isoformat()
        for row in sorted(rows, key=lambda r: str(r.get("date") or "")):
            row_date = str(row.get("date") or "")[:10]
            if row_date >= target_iso:
                try:
                    close = float(row.get("close") or 0)
                except (TypeError, ValueError):
                    continue
                if close > 0:
                    return close
        return None

    def _compute_mfe(
        self,
        collector: Any,
        record: dict[str, Any],
        sell_date: date,
    ) -> Optional[tuple[float, float]]:
        """Max favorable excursion vs realized exit → (mfe_pct, giveback_pct).

        MFE is the best gain available between entry and exit; give-back is
        how much of it the exit surrendered. Measurable "held too long".
        """
        thesis_id = record.get("thesis_id")
        sell_price = float(record.get("sell_price") or 0)
        if not thesis_id or sell_price <= 0:
            return None
        try:
            thesis_row = self.journal.get_thesis(thesis_id) or {}
        except Exception:
            thesis_row = {}
        entry_price = float(thesis_row.get("entry_price") or 0)
        entry_date_str = str(thesis_row.get("entry_date") or "")[:10]
        if entry_price <= 0 or not entry_date_str:
            return None
        ticker = record.get("ticker", "")
        try:
            rows = collector.fmp.history(ticker, "1y") or []
        except Exception:
            rows = []
        highs = []
        sell_iso = sell_date.isoformat()
        for row in rows:
            row_date = str(row.get("date") or "")[:10]
            if entry_date_str <= row_date <= sell_iso:
                try:
                    high = float(row.get("high") or row.get("close") or 0)
                except (TypeError, ValueError):
                    continue
                if high > 0:
                    highs.append(high)
        if not highs:
            return None
        mfe_pct = (max(highs) - entry_price) / entry_price
        realized_pct = (sell_price - entry_price) / entry_price
        giveback_pct = mfe_pct - realized_pct
        return (round(mfe_pct, 4), round(giveback_pct, 4))

    def update_shadow_prices(self, collector: Any) -> int:
        """Update prices for active shadow tracking records. Returns count updated.

        Checkpoints hit within CHECKPOINT_TOLERANCE_DAYS use the live quote;
        overdue checkpoints are back-filled with the historical close for the
        actual checkpoint date so returns are not stamped with today's price.
        """
        pending = self.journal.get_pending_post_sell_reviews()
        if not pending:
            return 0

        updated = 0
        today = _utcnow().date()

        for record in pending:
            ticker = record.get("ticker", "")
            sell_date_str = record.get("sell_date", "")
            sell_price = float(record.get("sell_price") or 0)
            tracking_id = record.get("tracking_id", "")

            if not ticker or sell_price <= 0:
                continue

            try:
                sell_date = datetime.strptime(sell_date_str, "%Y-%m-%d").date()
            except Exception:
                continue

            days_since_sell = (today - sell_date).days

            # Fetch current price
            try:
                quote = collector.yf.quote(ticker)
                current_price = float(quote.get("price", 0) or 0)
                if current_price <= 0:
                    continue
            except Exception as e:
                logger.warning("[post_sell] Failed to fetch price for %s: %s", ticker, e)
                continue

            # Compute return
            ret = (current_price - sell_price) / sell_price if sell_price > 0 else 0
            # Negative regret_score = price fell = selling was correct
            # Positive = price rose = we left money on table
            regret_score = ret  # same as return; naming is from the operator's perspective

            updates: dict[str, Any] = {}

            # Update checkpoint fields
            for checkpoint_days in self.CHECKPOINT_DAYS:
                price_key = f"price_{checkpoint_days}d"
                return_key = f"return_{checkpoint_days}d"
                if days_since_sell < checkpoint_days or record.get(price_key) is not None:
                    continue
                overdue_by = days_since_sell - checkpoint_days
                checkpoint_price: Optional[float] = None
                if overdue_by <= self.CHECKPOINT_TOLERANCE_DAYS:
                    checkpoint_price = current_price
                else:
                    # Overdue — fetch the close for the actual checkpoint date
                    checkpoint_date = sell_date + timedelta(days=checkpoint_days)
                    checkpoint_price = self._historical_close_on_or_after(
                        collector, ticker, checkpoint_date
                    )
                    if checkpoint_price is None:
                        logger.warning(
                            "[post_sell] No historical close for %s %sd checkpoint — skipping",
                            ticker, checkpoint_days,
                        )
                        continue
                updates[price_key] = checkpoint_price
                updates[return_key] = round(
                    (checkpoint_price - sell_price) / sell_price, 4
                )

            updates["regret_score"] = round(regret_score, 4)

            # Check if all checkpoints done
            max_checkpoint = max(self.CHECKPOINT_DAYS)
            if days_since_sell >= max_checkpoint:
                updates["status"] = "completed"
                return_20d = updates.get("return_20d", record.get("return_20d"))
                # Grade the sell decision (stop exits graded vs 20d-later price)
                updates["grade"] = self._grade_sell(
                    ret,
                    record.get("sell_reason", ""),
                    return_20d=return_20d,
                )
                # MFE give-back — how much of the best available gain the
                # exit surrendered (held-too-long metric)
                if record.get("mfe_pct") is None:
                    mfe = self._compute_mfe(collector, record, sell_date)
                    if mfe is not None:
                        updates["mfe_pct"], updates["giveback_pct"] = mfe

            if updates:
                updates["tracking_id"] = tracking_id
                try:
                    self.journal.save_post_sell_tracking(updates)
                    updated += 1
                except Exception as save_e:
                    logger.warning("[post_sell] Failed to update tracking %s: %s", tracking_id, save_e)

        return updated

    def _grade_sell(
        self,
        return_since_sell: float,
        reason: str,
        return_20d: Optional[float] = None,
    ) -> str:
        """Grade a sell decision based on subsequent performance.

        Negative return = price fell after sell = CORRECT exit.
        Positive return = price rose after sell = INCORRECT or EARLY exit.

        Stop-triggered exits are NOT auto-graded correct: they are graded
        against the actual price 20 days later. A stop that fired into a
        recovery (price 5%+ higher 20d later) was a WHIPSAW_STOP.
        """
        is_stop_exit = "hard stop" in reason.lower() or "urgent" in reason.lower() or "stop" in reason.lower()
        if is_stop_exit:
            benchmark = return_20d if return_20d is not None else return_since_sell
            if benchmark is None:
                return "STOP_TRIGGERED"  # no benchmark yet — defer grading
            if benchmark <= -0.05:
                return "CORRECT"         # price kept falling — stop saved money
            elif benchmark <= 0.05:
                return "NEUTRAL"
            else:
                return "WHIPSAW_STOP"    # price recovered — stop was too tight

        if return_since_sell <= -0.05:
            return "CORRECT"           # Price fell 5%+ — good sell
        elif return_since_sell <= 0.05:
            return "NEUTRAL"           # Within 5% either way
        elif return_since_sell <= 0.15:
            return "EARLY"             # Left some gains
        else:
            return "INCORRECT"         # Significant gains missed

    def format_report(self) -> str:
        """Format a post-sell shadow tracking report for the nightly review.

        Uses ALL tracking rows: get_pending_post_sell_reviews() only returns
        status='tracking' rows, so filtering it for 'completed' always came
        back empty — completed reviews never appeared in the report.
        """
        records = self.journal.get_all_post_sell_reviews()
        if not records:
            return ""

        completed = [r for r in records if r.get("status") == "completed"]
        tracking = [r for r in records if r.get("status") == "tracking"]

        lines = [
            "📉 POST-SELL SHADOW TRACKING",
            "",
        ]

        # Grade summary for completed
        if completed:
            grades = [r.get("grade", "") for r in completed]
            correct = grades.count("CORRECT")
            neutral = grades.count("NEUTRAL")
            early = grades.count("EARLY")
            incorrect = grades.count("INCORRECT")
            whipsaw = grades.count("WHIPSAW_STOP")
            lines.extend([
                f"Completed reviews: {len(completed)}",
                f"  ✅ Correct exits: {correct}",
                f"  ➡️ Neutral:       {neutral}",
                f"  ⚠️ Sold early:   {early}",
                f"  🔴 Incorrect:    {incorrect}",
                f"  🪤 Whipsaw stops: {whipsaw}",
            ])
            givebacks = [
                float(r["giveback_pct"]) for r in completed
                if r.get("giveback_pct") is not None
            ]
            if givebacks:
                lines.append(
                    f"  📐 Avg MFE give-back: {sum(givebacks) / len(givebacks):+.1%} "
                    "(gain surrendered vs best exit)"
                )
            lines.append("")

        # Active tracking
        if tracking:
            lines.append(f"Currently tracking: {len(tracking)} sell(s)")
            for r in tracking[:5]:
                ticker = r.get("ticker", "?")
                sell_price = r.get("sell_price", 0)
                sell_date = r.get("sell_date", "?")[:10]
                regret = r.get("regret_score")
                regret_str = f" | Return since: {regret:+.1%}" if regret is not None else ""
                lines.append(f"  • {ticker} sold ${sell_price:.2f} on {sell_date}{regret_str}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Opportunity Cost Scanner
# ---------------------------------------------------------------------------

class OpportunityCostScanner:
    """Finds rotation candidates by comparing weakest held position to best new opportunity."""

    def __init__(
        self,
        tracker: Optional[ThesisTracker] = None,
        journal: Optional[DecisionJournal] = None,
    ) -> None:
        self.tracker = tracker or ThesisTracker()
        self.journal = journal or DecisionJournal()

    def find_weakest_position(self, active_theses: list[PositionThesis]) -> Optional[PositionThesis]:
        """Find the weakest position that is eligible for rotation.

        A position is NOT eligible for rotation if:
        - Conviction locked (BUY/ACCUMULATE, health >= threshold, within lock window)
        - In cooldown
        - In minimum hold period
        """
        eligible = []
        for thesis in active_theses:
            # Conviction lock check
            if self._is_conviction_locked(thesis):
                logger.debug(
                    "[opp_cost] %s is conviction-locked (health=%d)", thesis.ticker, thesis.thesis_health_score
                )
                continue

            # Cooldown check
            if thesis.in_cooldown:
                continue

            # Min hold check
            if thesis.in_minimum_hold:
                continue

            eligible.append(thesis)

        if not eligible:
            return None

        # Weakest = lowest health score
        return min(eligible, key=lambda t: t.thesis_health_score)

    def _is_conviction_locked(self, thesis: PositionThesis) -> bool:
        """Return True if this position is protected from rotation."""
        if thesis.position_type not in ("BUY", "ACCUMULATE"):
            return False
        health = thesis.thesis_health_score or 0
        if health < Config.SELL_CONVICTION_LOCK_MIN_HEALTH:
            return False  # Unhealthy position is NOT locked
        if thesis.days_held >= Config.SELL_CONVICTION_LOCK_MAX_DAYS:
            return False  # Past the lock window
        return True

    def evaluate_rotation(
        self,
        weak_thesis: PositionThesis,
        candidate_score: float,
        candidate_ticker: str,
        candidate_rationale: str = "",
    ) -> Optional[RotationRecommendation]:
        """Evaluate whether a rotation from weak to candidate is warranted.

        Returns a RotationRecommendation if delta >= SELL_ROTATE_MIN_DELTA.
        """
        # Normalize health score to opportunity-like score (0-100)
        hold_score = weak_thesis.thesis_health_score or 0

        delta = candidate_score - hold_score
        if delta < Config.SELL_ROTATE_MIN_DELTA:
            return None

        rationale = (
            f"Rotation candidate: {candidate_ticker} (score {candidate_score:.0f}) "
            f"vs {weak_thesis.ticker} health {hold_score}/100. "
            f"Delta: +{delta:.0f} points. "
        )
        if candidate_rationale:
            rationale += f"Candidate: {candidate_rationale[:200]}"

        return RotationRecommendation(
            from_ticker=weak_thesis.ticker,
            to_ticker=candidate_ticker,
            from_health=hold_score,
            to_score=candidate_score,
            delta=delta,
            from_position_type=weak_thesis.position_type,
            rationale=rationale,
        )

    def scan_for_rotation(
        self,
        scan_results: list[dict],
    ) -> Optional[RotationRecommendation]:
        """Check if any active position should be rotated out for a better opportunity.

        Args:
            scan_results: List of council decisions from the weekly scan
                          Each dict should have: ticker, adjusted_score, synthesis_report

        Returns:
            RotationRecommendation if a rotation is warranted, else None
        """
        active_theses = self.tracker.get_all_active()
        if not active_theses:
            return None

        weak = self.find_weakest_position(active_theses)
        if not weak:
            return None

        # Find best candidate from scan (exclude already-held tickers)
        held_tickers = {t.ticker.upper() for t in active_theses}
        buy_verdicts = {"BUY", "STARTER", "TACTICAL_BUY", "ACCUMULATE"}

        best_candidate = None
        best_score = 0.0
        best_rationale = ""

        for result in scan_results:
            ticker = (result.get("ticker") or "").upper()
            if ticker in held_tickers:
                continue
            verdict = (result.get("final_verdict") or result.get("action") or "").upper()
            if verdict not in buy_verdicts:
                continue
            score = float(result.get("adjusted_score") or result.get("score") or 0)
            if score > best_score:
                best_score = score
                best_candidate = ticker
                best_rationale = str(result.get("synthesis_report") or "")[:300]

        if not best_candidate:
            return None

        return self.evaluate_rotation(
            weak_thesis=weak,
            candidate_score=best_score,
            candidate_ticker=best_candidate,
            candidate_rationale=best_rationale,
        )

    def format_rotation_telegram(self, rec: RotationRecommendation) -> str:
        """Format a rotation recommendation for Telegram."""
        lines = [
            "🔄 ROTATION OPPORTUNITY DETECTED",
            "",
            f"SELL: {rec.from_ticker} (health {rec.from_health}/100, {rec.from_position_type})",
            f"BUY: {rec.to_ticker} (score {rec.to_score:.0f}/100)",
            f"Opportunity delta: +{rec.delta:.0f} points",
            "",
            f"Rationale: {rec.rationale[:400]}",
            "",
            "⚠️ Sell council review required before acting on rotation signal.",
            "Ask Ammu: 'run sell review on {from_ticker}' to proceed.".format(
                from_ticker=rec.from_ticker
            ),
        ]
        return "\n".join(lines)
