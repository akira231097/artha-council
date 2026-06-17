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
  - Records at 5/20/60-day checkpoints for accuracy grading
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
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

    def __init__(self, journal: Optional[DecisionJournal] = None) -> None:
        self.journal = journal or DecisionJournal()

    def record_sell(
        self,
        ticker: str,
        thesis_id: str,
        sell_price: float,
        sell_reason: str,
        shares: float,
        position_type: str,
    ) -> None:
        """Record a new post-sell tracking entry."""
        import uuid
        tracking_id = str(uuid.uuid4())
        self.journal.save_post_sell_tracking({
            "tracking_id": tracking_id,
            "ticker": ticker,
            "thesis_id": thesis_id,
            "sell_date": _utcnow_iso()[:10],
            "sell_price": sell_price,
            "sell_reason": sell_reason,
            "position_type": position_type,
            "shares": shares,
            "status": "tracking",
        })
        logger.info("[post_sell] Started shadow tracking for %s @ $%.2f", ticker, sell_price)

    def update_shadow_prices(self, collector: Any) -> int:
        """Update prices for active shadow tracking records. Returns count updated."""
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
            regret_score = ret  # same as return; naming is from Sarath's perspective

            updates: dict[str, Any] = {}

            # Update checkpoint fields
            for checkpoint_days in self.CHECKPOINT_DAYS:
                price_key = f"price_{checkpoint_days}d"
                return_key = f"return_{checkpoint_days}d"
                if days_since_sell >= checkpoint_days and record.get(price_key) is None:
                    updates[price_key] = current_price
                    updates[return_key] = round(ret, 4)

            updates["regret_score"] = round(regret_score, 4)

            # Check if all checkpoints done
            max_checkpoint = max(self.CHECKPOINT_DAYS)
            if days_since_sell >= max_checkpoint:
                updates["status"] = "completed"
                # Grade the sell decision
                updates["grade"] = self._grade_sell(ret, record.get("sell_reason", ""))

            if updates:
                updates["tracking_id"] = tracking_id
                try:
                    self.journal.save_post_sell_tracking(updates)
                    updated += 1
                except Exception as save_e:
                    logger.warning("[post_sell] Failed to update tracking %s: %s", tracking_id, save_e)

        return updated

    def _grade_sell(self, return_since_sell: float, reason: str) -> str:
        """Grade a sell decision based on subsequent performance.

        Negative return = price fell after sell = CORRECT exit.
        Positive return = price rose after sell = INCORRECT or EARLY exit.
        """
        # Hard stop exits are graded differently
        if "hard stop" in reason.lower() or "urgent" in reason.lower():
            return "STOP_TRIGGERED"  # Always correct if stop was hit

        if return_since_sell <= -0.05:
            return "CORRECT"           # Price fell 5%+ — good sell
        elif return_since_sell <= 0.05:
            return "NEUTRAL"           # Within 5% either way
        elif return_since_sell <= 0.15:
            return "EARLY"             # Left some gains
        else:
            return "INCORRECT"         # Significant gains missed

    def format_report(self) -> str:
        """Format a post-sell shadow tracking report for the nightly review."""
        pending = self.journal.get_pending_post_sell_reviews()
        if not pending:
            return ""

        completed = [r for r in pending if r.get("status") == "completed"]
        tracking = [r for r in pending if r.get("status") == "tracking"]

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
            lines.extend([
                f"Completed reviews: {len(completed)}",
                f"  ✅ Correct exits: {correct}",
                f"  ➡️ Neutral:       {neutral}",
                f"  ⚠️ Sold early:   {early}",
                f"  🔴 Incorrect:    {incorrect}",
                "",
            ])

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
