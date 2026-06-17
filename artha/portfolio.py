"""Portfolio tracker — manages positions, P&L, and risk limits.

Uses Decimal for all monetary calculations to prevent float drift.
Atomic file writes to prevent corruption on concurrent access.
"""
import json
import logging
import tempfile
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict, fields as dataclass_fields

from .config import Config

logger = logging.getLogger(__name__)

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

PORTFOLIO_FILE = Path(__file__).resolve().parent.parent / "data" / "portfolio.json"

# Decimal quantization for currency (2 decimal places)
CENTS = Decimal("0.01")


def _to_decimal(value: float | int | str | Decimal) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal("0")


def _utcnow() -> str:
    """Return current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _portfolio_lock(lock_path: Path, exclusive: bool):
    """Advisory file lock for portfolio read/write coordination."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a", encoding="utf-8") as lock_file:
        if fcntl is not None:
            lock_mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(lock_file.fileno(), lock_mode)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@dataclass
class Position:
    """A single investment position."""
    ticker: str
    asset_type: str  # "stock" or "crypto"
    shares: float
    avg_cost: float
    opened_at: str  # ISO datetime (UTC)
    notes: str = ""

    # Crisis Mode v3 metadata (optional — only set for crisis purchases)
    is_crisis_purchase: bool = False
    crisis_state_at_purchase: str = ""        # "normal", "correction", "bear", "panic"
    crisis_type_at_purchase: str = ""         # Dominant fingerprint type at purchase
    original_thesis: str = ""                 # Brief thesis for post-crisis debrief
    thesis_status: str = ""                   # "intact", "weakened", "broken" — reviewed post-crisis

    # Sell-engine tracking fields (optional — set when position enters sell monitoring)
    thesis_id: Optional[str] = None
    position_type: Optional[str] = None       # BUY | STARTER | TACTICAL_BUY | ACCUMULATE | ADD
    hard_stop_price: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    next_sell_review: Optional[str] = None    # ISO date string
    sell_cooldown_until: Optional[str] = None # ISO datetime string
    entry_date: Optional[str] = None          # ISO date of first entry (YYYY-MM-DD)
    scale_out_completed: list = field(default_factory=list)  # milestone tags already triggered

    # Live price tracking (updated during price checks)
    current_price: Optional[float] = None
    market_value: Optional[float] = None

    @property
    def total_cost(self) -> Decimal:
        return (_to_decimal(self.shares) * _to_decimal(self.avg_cost)).quantize(CENTS, ROUND_HALF_UP)

    @property
    def total_cost_float(self) -> float:
        """Float representation for backward compat."""
        return float(self.total_cost)


@dataclass
class Portfolio:
    """The investor's complete portfolio."""
    positions: list[Position] = field(default_factory=list)
    cash_deployed: float = 0.0
    transactions: list[dict] = field(default_factory=list)
    last_updated: str = ""

    # Preserved from live portfolio.json (not computed by this class)
    cash_available: float = 0.0
    monthly_contribution: float = 0.0
    notes: str = ""

    # -----------------------------------------------------------------------
    # Validation helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _validate_trade_inputs(ticker: str, shares: float, price: float) -> tuple[str, Decimal, Decimal]:
        """Normalize and validate user-provided trade inputs.
        
        Returns (normalized_ticker, shares_decimal, price_decimal).
        """
        normalized_ticker = (ticker or "").strip().upper()
        if not normalized_ticker:
            raise ValueError("Ticker is required")
        shares_d = _to_decimal(shares)
        price_d = _to_decimal(price)
        if shares_d <= 0:
            raise ValueError("Shares must be greater than zero")
        if price_d <= 0:
            raise ValueError("Price must be greater than zero")
        return normalized_ticker, shares_d, price_d

    def _monthly_buy_total(self, now: datetime | None = None) -> Decimal:
        """Total BUY dollars deployed in the current calendar month (UTC)."""
        current = now or datetime.now(timezone.utc)
        month_prefix = current.strftime("%Y-%m")
        total = Decimal("0")
        for txn in self.transactions:
            if txn.get("type") != "BUY":
                continue
            ts = str(txn.get("timestamp", ""))
            if ts.startswith(month_prefix):
                try:
                    total += _to_decimal(txn.get("total", 0))
                except Exception:
                    continue
        return total

    # -----------------------------------------------------------------------
    # Position management
    # -----------------------------------------------------------------------

    def add_position(
        self,
        ticker: str,
        shares: float,
        price: float,
        asset_type: str = "stock",
        notes: str = "",
        is_crisis_purchase: bool = False,
        crisis_state_at_purchase: str = "",
        crisis_type_at_purchase: str = "",
        original_thesis: str = "",
    ) -> None:
        """Add a new position or add to an existing one.

        Crisis metadata (is_crisis_purchase, crisis_state_at_purchase, etc.)
        is set only on new positions. If adding to an existing position,
        the original crisis metadata is preserved.
        """
        ticker, shares_d, price_d = self._validate_trade_inputs(ticker, shares, price)
        existing = self.get_position(ticker)
        if existing:
            # Average up/down using Decimal
            old_shares = _to_decimal(existing.shares)
            old_cost = _to_decimal(existing.avg_cost)
            total_shares = old_shares + shares_d
            total_cost = (old_shares * old_cost) + (shares_d * price_d)
            existing.avg_cost = float((total_cost / total_shares).quantize(CENTS, ROUND_HALF_UP))
            existing.shares = float(total_shares)
            # Update crisis status if this is a crisis add-on
            if is_crisis_purchase and not existing.is_crisis_purchase:
                existing.is_crisis_purchase = True
                existing.crisis_state_at_purchase = crisis_state_at_purchase
                existing.crisis_type_at_purchase = crisis_type_at_purchase
        else:
            self.positions.append(Position(
                ticker=ticker,
                asset_type=asset_type,
                shares=float(shares_d),
                avg_cost=float(price_d),
                opened_at=_utcnow(),
                notes=notes,
                is_crisis_purchase=is_crisis_purchase,
                crisis_state_at_purchase=crisis_state_at_purchase,
                crisis_type_at_purchase=crisis_type_at_purchase,
                original_thesis=original_thesis,
            ))

        trade_total = (shares_d * price_d).quantize(CENTS, ROUND_HALF_UP)
        self.cash_deployed = float(_to_decimal(self.cash_deployed) + trade_total)
        self.transactions.append({
            "type": "BUY",
            "ticker": ticker,
            "shares": float(shares_d),
            "price": float(price_d),
            "total": float(trade_total),
            "timestamp": _utcnow(),
            "notes": notes,
        })
        self.last_updated = _utcnow()

    def sell_position(self, ticker: str, shares: float, price: float,
                      notes: str = "") -> Optional[float]:
        """Sell shares from a position. Returns realized P&L or None if not found."""
        ticker, shares_d, price_d = self._validate_trade_inputs(ticker, shares, price)
        pos = self.get_position(ticker)
        if not pos:
            logger.warning(f"No position found for {ticker}")
            return None

        pos_shares_d = _to_decimal(pos.shares)
        if shares_d > pos_shares_d:
            logger.warning(f"Trying to sell {shares_d} shares of {ticker} but only have {pos_shares_d}")
            shares_d = pos_shares_d

        avg_cost_d = _to_decimal(pos.avg_cost)
        realized_pnl = ((price_d - avg_cost_d) * shares_d).quantize(CENTS, ROUND_HALF_UP)
        cost_basis_released = (avg_cost_d * shares_d).quantize(CENTS, ROUND_HALF_UP)
        remaining = pos_shares_d - shares_d

        if remaining <= Decimal("0.001"):  # Effectively zero
            self.positions = [p for p in self.positions if p.ticker != ticker]
        else:
            pos.shares = float(remaining)

        sell_total = (shares_d * price_d).quantize(CENTS, ROUND_HALF_UP)
        self.transactions.append({
            "type": "SELL",
            "ticker": ticker,
            "shares": float(shares_d),
            "price": float(price_d),
            "total": float(sell_total),
            "realized_pnl": float(realized_pnl),
            "timestamp": _utcnow(),
            "notes": notes,
        })
        self.cash_deployed = float(max(Decimal("0"), _to_decimal(self.cash_deployed) - cost_basis_released))
        self.last_updated = _utcnow()
        return float(realized_pnl)

    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position by ticker (case-insensitive)."""
        ticker_upper = (ticker or "").strip().upper()
        for pos in self.positions:
            if pos.ticker.upper() == ticker_upper:
                return pos
        return None

    def get_crisis_positions(self) -> list[Position]:
        """Return all positions purchased during a crisis state."""
        return [p for p in self.positions if p.is_crisis_purchase]

    def get_crisis_cost_basis(self) -> float:
        """Total cost basis of all crisis-purchased positions."""
        return sum(float(p.total_cost) for p in self.get_crisis_positions())

    def update_thesis_status(
        self, ticker: str, status: str, notes: str = ""
    ) -> bool:
        """Update thesis status for a position after review.

        Args:
            ticker: Stock ticker
            status: "intact", "weakened", or "broken"
            notes: Optional review notes appended to position notes

        Returns True if position found and updated.
        """
        pos = self.get_position(ticker)
        if not pos:
            return False
        if status not in ("intact", "weakened", "broken"):
            raise ValueError(f"Invalid thesis status: {status}. Must be intact/weakened/broken")
        pos.thesis_status = status
        if notes:
            pos.notes = f"{pos.notes} | Thesis review: {notes}".strip(" |")
        self.last_updated = _utcnow()
        return True

    # -----------------------------------------------------------------------
    # Risk management
    # -----------------------------------------------------------------------

    def check_risk_limits(self, ticker: str, proposed_amount: float,
                          current_price: float, asset_type: str = "stock") -> dict:
        """Check if a proposed buy violates risk management rules.

        Returns dict with 'allowed' bool and 'reason' if blocked.
        """
        ticker = (ticker or "").strip().upper()
        if not ticker:
            return {"allowed": False, "reason": "Ticker is required"}
        proposed_d = _to_decimal(proposed_amount)
        if proposed_d <= 0:
            return {"allowed": False, "reason": "Proposed amount must be greater than zero"}
        price_d = _to_decimal(current_price)
        if price_d <= 0:
            return {"allowed": False, "reason": "Current price must be greater than zero"}

        # Monthly budget check
        monthly_spend = self._monthly_buy_total()
        budget_d = _to_decimal(Config.MONTHLY_BUDGET)
        if monthly_spend + proposed_d > budget_d:
            return {
                "allowed": False,
                "reason": (
                    f"Monthly budget exceeded: ${float(monthly_spend + proposed_d):,.2f} "
                    f"> ${Config.MONTHLY_BUDGET:,.2f}"
                ),
            }

        # Cash and concentration checks. Include available cash in the NAV
        # denominator so the first starter position is not incorrectly treated
        # as 100% concentration before cash is deployed.
        cash_d = _to_decimal(self.cash_available)
        if cash_d > 0 and proposed_d > cash_d:
            return {
                "allowed": False,
                "reason": f"Insufficient cash available: ${float(proposed_d):,.2f} > ${float(cash_d):,.2f}",
            }

        total_portfolio_value = sum(p.total_cost for p in self.positions) + max(cash_d, proposed_d)
        existing = self.get_position(ticker)
        existing_value = existing.total_cost if existing else Decimal("0")
        new_total = existing_value + proposed_d
        concentration = float(new_total / total_portfolio_value) if total_portfolio_value > 0 else 1.0

        max_pct = Config.MAX_SINGLE_CRYPTO_PCT if asset_type == "crypto" else Config.MAX_SINGLE_STOCK_PCT

        if concentration > max_pct:
            return {
                "allowed": False,
                "reason": f"{ticker} would be {concentration:.0%} of portfolio "
                          f"(max {max_pct:.0%} for {asset_type})",
            }

        return {"allowed": True, "reason": "Within risk limits"}

    def check_alerts(self, current_prices: dict[str, float]) -> list[str]:
        """Check for stop-loss and take-profit alerts.

        Args:
            current_prices: Dict of ticker -> current price

        Returns:
            List of alert messages
        """
        alerts = []
        for pos in self.positions:
            raw_price = current_prices.get(pos.ticker.upper())
            if raw_price is None:
                continue
            try:
                price = float(raw_price)
            except (TypeError, ValueError):
                logger.warning(f"Invalid current price for {pos.ticker}; skipping alert")
                continue
            if pos.avg_cost <= 0:
                logger.warning(f"Invalid avg_cost for {pos.ticker}; skipping alert")
                continue

            pnl_pct = (price - pos.avg_cost) / pos.avg_cost

            if pnl_pct <= Config.STOP_LOSS_PCT:
                alerts.append(
                    f"🚨 STOP-LOSS: ${pos.ticker} is down {pnl_pct:.1%} "
                    f"(bought at ${pos.avg_cost:.2f}, now ${price:.2f}). "
                    f"Consider selling to limit losses."
                )
            elif pnl_pct >= Config.TAKE_PROFIT_PCT:
                alerts.append(
                    f"🎯 TAKE-PROFIT: ${pos.ticker} is up {pnl_pct:.1%} "
                    f"(bought at ${pos.avg_cost:.2f}, now ${price:.2f}). "
                    f"Consider taking some profits."
                )

        return alerts

    # -----------------------------------------------------------------------
    # Persistence (atomic writes)
    # -----------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> None:
        """Save portfolio to JSON file using atomic write (tempfile + rename)."""
        save_path = path or PORTFOLIO_FILE
        lock_path = save_path.with_suffix(".lock")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing file to preserve keys not tracked by the dataclass
        # (e.g. cash_available, monthly_contribution, notes written by humans)
        existing_data: dict = {}
        if save_path.exists():
            try:
                with open(save_path, encoding="utf-8") as _f:
                    existing_data = json.load(_f)
            except Exception:
                pass

        # Merge: start from existing, overlay our tracked fields
        data = dict(existing_data)
        data.update({
            "positions": [asdict(p) for p in self.positions],
            "cash_deployed": self.cash_deployed,
            "transactions": self.transactions,
            "last_updated": self.last_updated,
            "cash_available": self.cash_available if self.cash_available is not None else existing_data.get("cash_available", 0.0),
            "monthly_contribution": self.monthly_contribution if self.monthly_contribution is not None else existing_data.get("monthly_contribution", 0.0),
            "notes": self.notes or existing_data.get("notes", ""),
        })

        with _portfolio_lock(lock_path, exclusive=True):
            # Atomic write: write to temp file, then rename
            fd, tmp_path = tempfile.mkstemp(
                dir=str(save_path.parent),
                suffix=".tmp",
                prefix=".portfolio_",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, str(save_path))
                logger.info(f"Portfolio saved to {save_path}")
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Portfolio":
        """Load portfolio from JSON file."""
        load_path = path or PORTFOLIO_FILE
        lock_path = load_path.with_suffix(".lock")

        if not load_path.exists():
            logger.info("No existing portfolio found, creating new one")
            return cls()

        try:
            with _portfolio_lock(lock_path, exclusive=False):
                with open(load_path, encoding="utf-8") as f:
                    data = json.load(f)

            # FIX E: filter unknown keys so a single extra field in portfolio.json
            # doesn't crash load() and silently wipe the entire portfolio.
            _valid_position_fields = {f.name for f in dataclass_fields(Position)}
            positions = []
            for _p in data.get("positions", []):
                _filtered = {k: v for k, v in _p.items() if k in _valid_position_fields}
                try:
                    positions.append(Position(**_filtered))
                except Exception as _pos_e:
                    logger.warning(f"Skipping malformed position entry: {_pos_e}")
            return cls(
                positions=positions,
                cash_deployed=data.get("cash_deployed", 0.0),
                transactions=data.get("transactions", []),
                last_updated=data.get("last_updated", ""),
                cash_available=data.get("cash_available", 0.0),
                monthly_contribution=data.get("monthly_contribution", 0.0),
                notes=data.get("notes", ""),
            )
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
            return cls()

    def total_nav(self) -> float:
        """Approximate NAV = total cost basis of all positions.

        Since we don't store live prices in the portfolio file, cost basis is
        used as a proxy. Updated market_value fields (if present) are preferred.
        """
        total = Decimal("0")
        for p in self.positions:
            if p.market_value is not None:
                total += _to_decimal(p.market_value)
            else:
                total += p.total_cost
        return float(total)

    def summary(self) -> dict:
        """Quick portfolio summary."""
        total_cost = sum(p.total_cost for p in self.positions)
        stock_cost = sum(p.total_cost for p in self.positions if p.asset_type == "stock")
        crypto_cost = sum(p.total_cost for p in self.positions if p.asset_type == "crypto")

        return {
            "num_positions": len(self.positions),
            "total_invested": float(total_cost),
            "stocks_invested": float(stock_cost),
            "crypto_invested": float(crypto_cost),
            "num_transactions": len(self.transactions),
        }
