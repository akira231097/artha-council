"""Thesis Tracker — manage position lifecycle from pending buy to archived exit.

Every position enters the system as a pending thesis (created when council issues a
buy recommendation) and transitions through:
  pending → active (Sarath buys) → archived (position exited or expired)

Thesis objects store the investment thesis, invalidation conditions, stop levels,
review schedule, and health score. They are the primary vehicle the sell engine
uses to decide whether to hold, trim, or exit.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .config import Config
from .journal import DecisionJournal

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat()


def _add_days(days: int) -> str:
    return (_utcnow() + timedelta(days=days)).isoformat()


# Position type → review interval in days
_REVIEW_DAYS: dict[str, int] = {
    "TACTICAL_BUY": Config.SELL_REVIEW_DAYS_TACTICAL,
    "STARTER": Config.SELL_REVIEW_DAYS_STARTER,
    "BUY": Config.SELL_REVIEW_DAYS_BUY,
    "ACCUMULATE": Config.SELL_REVIEW_DAYS_ACCUMULATE,
    "ADD": Config.SELL_REVIEW_DAYS_BUY,  # treat ADD like BUY
}

# Position type → hard stop fraction from avg_cost
_HARD_STOP: dict[str, float] = {
    "TACTICAL_BUY": Config.SELL_HARD_STOP_TACTICAL,
    "STARTER": Config.SELL_HARD_STOP_STARTER,
    "BUY": Config.SELL_HARD_STOP_BUY,
    "ACCUMULATE": Config.SELL_HARD_STOP_ACCUMULATE,
    "ADD": Config.SELL_HARD_STOP_BUY,
}

_MIN_HOLD: dict[str, int] = {
    "TACTICAL_BUY": Config.SELL_MIN_HOLD_TACTICAL,
    "STARTER": Config.SELL_MIN_HOLD_STARTER,
    "BUY": Config.SELL_MIN_HOLD_BUY,
    "ACCUMULATE": Config.SELL_MIN_HOLD_ACCUMULATE,
    "ADD": Config.SELL_MIN_HOLD_BUY,
}


@dataclass
class PositionThesis:
    """Full thesis for a single position."""

    thesis_id: str
    ticker: str
    status: str                     # pending | active | pending_exit | archived | expired
    position_type: str              # BUY | STARTER | TACTICAL_BUY | ACCUMULATE | ADD

    thesis_summary: str = ""
    invalidation_conditions: list[str] = field(default_factory=list)
    price_target: Optional[float] = None
    stop_loss_pct: float = -0.20    # fraction (e.g. -0.25)
    stop_loss_price: Optional[float] = None
    recommended_allocation_pct: float = 0.0

    # Filled when Sarath executes the buy
    entry_price: Optional[float] = None
    entry_date: Optional[str] = None
    entry_regime: Optional[str] = None
    hard_stop_price: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    trailing_stop_high: Optional[float] = None

    # Health and review tracking
    thesis_health_score: int = 100
    last_review_date: Optional[str] = None
    next_review_date: Optional[str] = None
    sell_cooldown_until: Optional[str] = None
    scale_out_completed: list[str] = field(default_factory=list)

    # Exit data (archived theses)
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    # Metadata
    council_session_id: Optional[str] = None
    pending_expiry: Optional[str] = None
    notes: str = ""
    created_at: str = field(default_factory=_utcnow_iso)
    updated_at: str = field(default_factory=_utcnow_iso)

    # ------------------------------------------------------------------ helpers

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def is_pending(self) -> bool:
        return self.status == "pending"

    @property
    def is_waiting_to_sell(self) -> bool:
        return self.status == "pending_exit"

    @property
    def days_held(self) -> int:
        if not self.entry_date:
            return 0
        try:
            entry = datetime.fromisoformat(self.entry_date)
            if entry.tzinfo is None:
                entry = entry.replace(tzinfo=timezone.utc)
            return max(0, (_utcnow() - entry).days)
        except Exception:
            return 0

    @property
    def min_hold_days(self) -> int:
        return _MIN_HOLD.get(self.position_type, 30)

    @property
    def in_minimum_hold(self) -> bool:
        return self.days_held < self.min_hold_days

    @property
    def in_cooldown(self) -> bool:
        if not self.sell_cooldown_until:
            return False
        try:
            deadline = datetime.fromisoformat(self.sell_cooldown_until)
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            return _utcnow() < deadline
        except Exception:
            return False

    def to_db_dict(self) -> dict[str, Any]:
        """Serialize for SQLite storage (flatten lists/dicts to JSON strings)."""
        d = asdict(self)
        d["invalidation_conditions"] = json.dumps(d.get("invalidation_conditions") or [])
        d["scale_out_completed"] = json.dumps(d.get("scale_out_completed") or [])
        return d

    @classmethod
    def from_db_dict(cls, row: dict[str, Any]) -> "PositionThesis":
        """Deserialize from SQLite row."""
        d = dict(row)
        # Parse JSON-encoded list fields
        for field_name in ("invalidation_conditions", "scale_out_completed"):
            raw = d.get(field_name) or "[]"
            if isinstance(raw, str):
                try:
                    d[field_name] = json.loads(raw)
                except Exception:
                    d[field_name] = []
        # Remove unknown keys gracefully
        known = {f.name for f in PositionThesis.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        d = {k: v for k, v in d.items() if k in known}
        return cls(**d)


class ThesisTracker:
    """Create, query, update, and archive position theses with SQLite persistence."""

    def __init__(self, journal: Optional[DecisionJournal] = None) -> None:
        self.journal = journal or DecisionJournal()

    # ---------------------------------------------------------------- create

    def create_thesis(
        self,
        ticker: str,
        position_type: str,
        thesis_summary: str = "",
        invalidation_conditions: Optional[list[str]] = None,
        price_target: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        recommended_allocation_pct: float = 0.0,
        council_session_id: Optional[str] = None,
        regime: Optional[str] = None,
        notes: str = "",
    ) -> PositionThesis:
        """Create a pending thesis after a buy recommendation."""
        ticker = (ticker or "").upper().strip()
        position_type = (position_type or "BUY").upper().strip()

        # Derive hard stop fraction
        stop_pct = stop_loss_pct if stop_loss_pct is not None else _HARD_STOP.get(
            position_type, Config.SELL_HARD_STOP_LEGACY
        )

        thesis_id = str(uuid.uuid4())
        expiry = _add_days(Config.SELL_THESIS_PENDING_EXPIRY_DAYS)

        thesis = PositionThesis(
            thesis_id=thesis_id,
            ticker=ticker,
            status="pending",
            position_type=position_type,
            thesis_summary=thesis_summary,
            invalidation_conditions=invalidation_conditions or [],
            price_target=price_target,
            stop_loss_pct=stop_pct,
            recommended_allocation_pct=recommended_allocation_pct,
            council_session_id=council_session_id,
            entry_regime=regime,
            pending_expiry=expiry,
            notes=notes,
        )

        self._save(thesis)
        logger.info(
            "[thesis] Created pending thesis %s ticker=%s type=%s",
            thesis_id[:8],
            ticker,
            position_type,
        )
        return thesis

    # ---------------------------------------------------------------- activate

    def activate_thesis(
        self,
        thesis_id: str,
        entry_price: float,
        entry_date: Optional[str] = None,
        shares: Optional[float] = None,
    ) -> Optional[PositionThesis]:
        """Activate a pending thesis when Sarath executes the buy."""
        thesis = self.get(thesis_id)
        if thesis is None:
            logger.warning("[thesis] Cannot activate — thesis %s not found", thesis_id)
            return None

        position_type = thesis.position_type
        stop_pct = thesis.stop_loss_pct or _HARD_STOP.get(position_type, Config.SELL_HARD_STOP_LEGACY)
        hard_stop = round(entry_price * (1 + stop_pct), 4) if entry_price else None

        # Compute first review date
        review_days = _REVIEW_DAYS.get(position_type, 30)
        next_review = _add_days(review_days)

        thesis.status = "active"
        thesis.entry_price = entry_price
        thesis.entry_date = entry_date or _utcnow_iso()
        thesis.hard_stop_price = hard_stop
        thesis.thesis_health_score = 100
        thesis.last_review_date = _utcnow_iso()
        thesis.next_review_date = next_review
        thesis.sell_cooldown_until = _add_days(Config.SELL_COOLDOWN_AFTER_BUY)

        # For TACTICAL_BUY, set initial trailing stop
        if position_type == "TACTICAL_BUY":
            # Initial trailing stop is same as hard stop; updated by TrailingStop module
            thesis.trailing_stop_price = hard_stop
            thesis.trailing_stop_high = entry_price

        if shares and thesis.notes:
            thesis.notes += f" | {shares} shares @ ${entry_price:.2f}"
        elif shares:
            thesis.notes = f"{shares} shares @ ${entry_price:.2f}"

        self._save(thesis)
        logger.info(
            "[thesis] Activated thesis %s ticker=%s entry=%.2f hard_stop=%.2f",
            thesis_id[:8],
            thesis.ticker,
            entry_price,
            hard_stop or 0,
        )
        return thesis

    # ---------------------------------------------------------------- get

    def get(self, thesis_id: str) -> Optional[PositionThesis]:
        """Fetch a thesis by ID."""
        row = self.journal.get_thesis(thesis_id)
        if not row:
            return None
        try:
            return PositionThesis.from_db_dict(row)
        except Exception as e:
            logger.error("[thesis] Failed to deserialize thesis %s: %s", thesis_id, e)
            return None

    def get_active(self, ticker: str) -> Optional[PositionThesis]:
        """Get the active thesis for a ticker."""
        row = self.journal.get_active_thesis_for_ticker(ticker)
        if not row:
            return None
        try:
            return PositionThesis.from_db_dict(row)
        except Exception as e:
            logger.error("[thesis] Failed to deserialize active thesis for %s: %s", ticker, e)
            return None

    def get_all_active(self) -> list[PositionThesis]:
        """Get all active theses."""
        rows = self.journal.get_all_active_theses()
        theses = []
        for row in rows:
            try:
                theses.append(PositionThesis.from_db_dict(row))
            except Exception as e:
                logger.warning("[thesis] Skip malformed thesis row: %s", e)
        return theses

    def get_pending_for_ticker(self, ticker: str) -> Optional[PositionThesis]:
        """Get the most recent non-expired pending thesis for a ticker."""
        theses = self.journal.get_pending_theses()
        now_iso = _utcnow_iso()
        for row in theses:
            if (row.get("ticker") or "").upper() == ticker.upper():
                expiry = row.get("pending_expiry")
                if expiry and expiry < now_iso:
                    continue
                try:
                    return PositionThesis.from_db_dict(row)
                except Exception as e:
                    logger.warning("[thesis] Skip malformed pending thesis: %s", e)
        return None

    def get_due_reviews(self) -> list[PositionThesis]:
        """Get active theses whose review date has passed."""
        rows = self.journal.get_due_reviews()
        theses = []
        for row in rows:
            try:
                theses.append(PositionThesis.from_db_dict(row))
            except Exception as e:
                logger.warning("[thesis] Skip malformed review row: %s", e)
        return theses

    # ---------------------------------------------------------------- update

    def update_health(self, thesis_id: str, health_score: int, notes: str = "") -> None:
        """Update thesis health score (0-100)."""
        thesis = self.get(thesis_id)
        if not thesis:
            return
        thesis.thesis_health_score = max(0, min(100, health_score))
        if notes:
            stamp = _utcnow_iso()
            thesis.notes = (thesis.notes or "") + f"\n[PENDING_EXIT {stamp}] {notes}"
        self._save(thesis)
        logger.debug("[thesis] Updated health %s → %d", thesis_id[:8], thesis.thesis_health_score)

    def update_review_date(self, thesis_id: str, next_review_date: Optional[str] = None) -> None:
        """Advance the next review date (default: use position_type schedule)."""
        thesis = self.get(thesis_id)
        if not thesis:
            return
        thesis.last_review_date = _utcnow_iso()
        if next_review_date:
            thesis.next_review_date = next_review_date
        else:
            review_days = _REVIEW_DAYS.get(thesis.position_type, 30)
            thesis.next_review_date = _add_days(review_days)
        self._save(thesis)

    def update_trailing_stop(
        self, thesis_id: str, new_stop: float, new_high: Optional[float] = None
    ) -> None:
        """Update trailing stop price and high-water mark."""
        thesis = self.get(thesis_id)
        if not thesis:
            return
        thesis.trailing_stop_price = new_stop
        if new_high is not None:
            thesis.trailing_stop_high = new_high
        self._save(thesis)

    def set_cooldown(self, thesis_id: str, days: int) -> None:
        """Set a sell cooldown for the given number of days."""
        thesis = self.get(thesis_id)
        if not thesis:
            return
        thesis.sell_cooldown_until = _add_days(days)
        self._save(thesis)

    def record_scale_out(self, thesis_id: str, milestone: str) -> None:
        """Record a completed scale-out milestone."""
        thesis = self.get(thesis_id)
        if not thesis:
            return
        if milestone not in thesis.scale_out_completed:
            thesis.scale_out_completed.append(milestone)
            self._save(thesis)

    def update_thesis_fields(self, thesis_id: str, **kwargs: Any) -> None:
        """Generic field updater for any thesis fields."""
        thesis = self.get(thesis_id)
        if not thesis:
            return
        for key, value in kwargs.items():
            if hasattr(thesis, key):
                setattr(thesis, key, value)
        self._save(thesis)

    def mark_waiting_for_sell(self, thesis_id: str, reason: str = "", notes: str = "") -> None:
        """Move an active thesis into waiting-for-sell state after ARTHA issues an exit call."""
        thesis = self.get(thesis_id)
        if not thesis:
            logger.warning("[thesis] Cannot mark waiting-for-sell — thesis %s not found", thesis_id)
            return
        if thesis.status == "archived":
            return
        thesis.status = "pending_exit"
        thesis.next_review_date = None
        if reason:
            thesis.exit_reason = reason
        if notes:
            stamp = _utcnow_iso()
            thesis.notes = (thesis.notes or "") + f"\n[PENDING_EXIT {stamp}] {notes}"
        self._save(thesis)
        logger.info(
            "[thesis] Marked thesis %s ticker=%s as pending_exit reason=%s",
            thesis_id[:8],
            thesis.ticker,
            reason,
        )

    # ---------------------------------------------------------------- archive

    def archive_thesis(
        self,
        thesis_id: str,
        exit_price: Optional[float] = None,
        exit_reason: str = "",
        notes: str = "",
    ) -> None:
        """Archive a thesis when the position is exited."""
        thesis = self.get(thesis_id)
        if not thesis:
            logger.warning("[thesis] Cannot archive — thesis %s not found", thesis_id)
            return
        thesis.status = "archived"
        thesis.exit_date = _utcnow_iso()
        thesis.exit_price = exit_price
        thesis.exit_reason = exit_reason
        if notes:
            thesis.notes = (thesis.notes or "") + f"\n[EXIT] {notes}"
        self._save(thesis)
        logger.info(
            "[thesis] Archived thesis %s ticker=%s reason=%s",
            thesis_id[:8],
            thesis.ticker,
            exit_reason,
        )

    def expire_stale_pending(self) -> int:
        """Expire any pending theses past their expiry date. Returns count expired.

        NOTE: Uses a direct query for ALL status='pending' theses regardless of
        pending_expiry, so expired rows (which get_pending_theses() filters out)
        are correctly transitioned to 'expired'.
        """
        rows = self.journal.get_all_pending_theses_raw()
        now_iso = _utcnow_iso()
        expired = 0
        for row in rows:
            expiry = row.get("pending_expiry")
            if expiry and expiry <= now_iso:
                try:
                    thesis = PositionThesis.from_db_dict(row)
                    thesis.status = "expired"
                    self._save(thesis)
                    expired += 1
                    logger.info(
                        "[thesis] Expired pending thesis %s ticker=%s",
                        row.get("thesis_id", "?")[:8],
                        row.get("ticker", "?"),
                    )
                except Exception as e:
                    logger.warning("[thesis] Failed to expire thesis: %s", e)
        return expired

    # ---------------------------------------------------------------- private

    def _save(self, thesis: PositionThesis) -> None:
        """Persist thesis to database with optimistic locking.

        Does NOT pre-set updated_at here; save_thesis() sets it after the
        optimistic-lock check, so the value in thesis.updated_at at call time
        is treated as the 'expected' (loaded) version for conflict detection.
        """
        try:
            self.journal.save_thesis(thesis.to_db_dict())
        except Exception as e:
            logger.error("[thesis] Failed to save thesis %s: %s", thesis.thesis_id[:8], e)
