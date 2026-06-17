"""Real-time portfolio monitor and alert engine."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP
from pathlib import Path
from typing import Any, Optional

from .collector import DataCollector, get_fear_greed_index
from .config import Config
from .portfolio import Decimal, PORTFOLIO_FILE, Portfolio, _to_decimal
from .sentinel import NewsSentinel
from .crisis import CrisisState, CrisisStateManager, CrisisFingerprint

logger = logging.getLogger(__name__)

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

ALERT_HISTORY_FILE = Path(__file__).resolve().parent.parent / "data" / "alert_history.json"
NEWS_TRIGGER_WORDS = (
    "lawsuit",
    "recall",
    "acquisition",
    "bankruptcy",
    "fda",
    "hack",
    "breach",
    "merger",
    "investigation",
    "fraud",
)
FOMC_DATES_2026 = [
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 4, 29),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 10, 28),
    date(2026, 12, 9),
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat()


def _parse_iso_utc(value: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


@contextmanager
def _alert_history_lock(lock_path: Path, exclusive: bool):
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
class Alert:
    """Single alert payload."""

    ticker: str
    alert_type: str
    severity: str  # CRITICAL, WARNING, INFO
    message: str
    timestamp: str = field(default_factory=_utcnow_iso)
    metadata: dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """Alert dedupe + formatting + durable alert history."""

    def __init__(self, history_path: Path = ALERT_HISTORY_FILE):
        self.history_path = history_path
        self.lock_path = history_path.with_suffix(".lock")
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_history(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {"alerts": [], "meta": {}}
        payload.setdefault("alerts", [])
        payload.setdefault("meta", {})
        return payload

    def _load_history_unlocked(self) -> dict[str, Any]:
        if not self.history_path.exists():
            return {"alerts": [], "meta": {}}
        with open(self.history_path, encoding="utf-8") as f:
            payload = json.load(f)
        return self._normalize_history(payload)

    def _read_history(self) -> dict[str, Any]:
        try:
            with _alert_history_lock(self.lock_path, exclusive=False):
                return self._load_history_unlocked()
        except Exception as e:
            logger.error(f"Failed to read alert history: {e}")
            return {"alerts": [], "meta": {}}

    def _write_history_unlocked(self, payload: dict[str, Any]) -> None:
        payload["updated_at"] = _utcnow_iso()
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.history_path.parent),
            suffix=".tmp",
            prefix=".alerts_",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(self.history_path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _write_history(self, payload: dict[str, Any]) -> None:
        with _alert_history_lock(self.lock_path, exclusive=True):
            self._write_history_unlocked(payload)

    def _alert_key(self, alert: Alert) -> str:
        return f"{alert.ticker.upper()}|{alert.alert_type}"

    def should_send(self, alert: Alert, within_hours: int = 24) -> bool:
        history = self._read_history()
        now = _utcnow()
        limit = timedelta(hours=within_hours)
        key = self._alert_key(alert)
        for entry in history.get("alerts", []):
            if not isinstance(entry, dict):
                continue
            if entry.get("key") != key:
                continue
            sent_at = _parse_iso_utc(str(entry.get("sent_at", "")))
            if sent_at and (now - sent_at) <= limit:
                return False
        return True

    def filter_new_alerts(self, alerts: list[Alert], within_hours: int = 24) -> list[Alert]:
        fresh: list[Alert] = []
        for alert in alerts:
            try:
                if self.should_send(alert, within_hours=within_hours):
                    fresh.append(alert)
            except Exception as e:
                logger.error(f"Dedupe check failed for {alert.ticker}/{alert.alert_type}: {e}")
        return fresh

    def record_sent_alerts(self, alerts: list[Alert]) -> None:
        if not alerts:
            return
        try:
            self.claim_new_alerts(alerts, within_hours=0)
        except Exception as e:
            logger.error(f"Failed to record sent alerts: {e}")

    def get_meta(self, key: str, default: Any = None) -> Any:
        try:
            with _alert_history_lock(self.lock_path, exclusive=False):
                history = self._load_history_unlocked()
            return history.get("meta", {}).get(key, default)
        except Exception as e:
            logger.error(f"Failed to load alert metadata {key}: {e}")
            return default

    def set_meta(self, key: str, value: Any) -> None:
        try:
            with _alert_history_lock(self.lock_path, exclusive=True):
                history = self._load_history_unlocked()
                meta = history.setdefault("meta", {})
                meta[key] = value
                self._write_history_unlocked(history)
        except Exception as e:
            logger.error(f"Failed to persist alert metadata {key}: {e}")

    def claim_new_alerts(self, alerts: list[Alert], within_hours: int = 24) -> list[Alert]:
        """Atomically dedupe and persist alerts so concurrent workers cannot double-send."""
        if not alerts:
            return []
        try:
            with _alert_history_lock(self.lock_path, exclusive=True):
                history = self._load_history_unlocked()
                existing = history.get("alerts", [])
                now = _utcnow()
                keep_after = now - timedelta(days=30)
                dedupe_cutoff = now - timedelta(hours=max(within_hours, 0))

                cleaned: list[dict[str, Any]] = []
                for entry in existing:
                    sent_at = _parse_iso_utc(str(entry.get("sent_at", ""))) if isinstance(entry, dict) else None
                    if sent_at and sent_at >= keep_after:
                        cleaned.append(entry)

                fresh: list[Alert] = []
                for alert in alerts:
                    key = self._alert_key(alert)
                    duplicate = False
                    for entry in cleaned:
                        if not isinstance(entry, dict) or entry.get("key") != key:
                            continue
                        sent_at = _parse_iso_utc(str(entry.get("sent_at", "")))
                        if sent_at and sent_at >= dedupe_cutoff:
                            duplicate = True
                            break
                    if duplicate:
                        continue
                    fresh.append(alert)
                    cleaned.append(
                        {
                            "key": key,
                            "ticker": alert.ticker.upper(),
                            "alert_type": alert.alert_type,
                            "severity": alert.severity,
                            "sent_at": _utcnow_iso(),
                            "message": alert.message,
                        }
                    )

                history["alerts"] = cleaned
                self._write_history_unlocked(history)
                return fresh
        except Exception as e:
            logger.error(f"Atomic alert claim failed: {e}")
            return alerts

    def format_for_telegram(self, alerts: list[Alert]) -> str:
        if not alerts:
            return "✅ Portfolio check complete: no new alerts."
        severity_emoji = {"CRITICAL": "🚨", "WARNING": "⚠️", "INFO": "ℹ️"}
        lines = ["📊 Artha Monitor Update", ""]
        for alert in alerts:
            emoji = severity_emoji.get(alert.severity.upper(), "🔔")
            lines.append(f"{emoji} *{alert.severity.title()}* - {alert.ticker.upper()}")
            lines.append(f"   {alert.message}")
        lines.append("")
        lines.append("💡 Beginner tip: Review alerts before placing any trade.")
        return "\n".join(lines)


class PriceMonitor:
    """Real-time monitor that produces portfolio + market event alerts."""

    def __init__(
        self,
        portfolio_path: Path = PORTFOLIO_FILE,
        alert_manager: Optional[AlertManager] = None,
        collector: Optional[DataCollector] = None,
    ):
        self.portfolio_path = portfolio_path
        self.collector = collector or DataCollector()
        self.alert_manager = alert_manager or AlertManager()
        self.sentinel = None  # Lazy init to avoid circular imports

    def _fetch_position_quotes(self, tickers: list[str]) -> dict[str, dict[str, Decimal]]:
        quotes: dict[str, dict[str, Decimal]] = {}
        for ticker in tickers:
            try:
                payload = self.collector.yf.quote(ticker)
                if not payload:
                    continue
                price = _to_decimal(payload.get("price", 0))
                prev_close = _to_decimal(payload.get("previous_close", 0))
                if price <= 0:
                    continue
                quotes[ticker] = {"price": price, "previous_close": prev_close}
            except Exception as e:
                logger.error(f"Failed to fetch quote for {ticker}: {e}")
        return quotes

    @staticmethod
    def _pct_change(new_value: Decimal, old_value: Decimal) -> Decimal:
        if old_value <= 0:
            return Decimal("0")
        return ((new_value - old_value) / old_value).quantize(Decimal("0.0001"), ROUND_HALF_UP)

    @staticmethod
    def _is_sell_engine_managed_position(pos: Position) -> bool:
        """Return True when sell/thesis lifecycle alerts belong to the new sell engine.

        Sarath's directive (2026-04-05):
        - old monitor => general market / portfolio alerts
        - new sell engine => thesis-driven sell alerts + buy/sell lifecycle
        """
        thesis_id = getattr(pos, "thesis_id", None)
        pos_type = str(getattr(pos, "position_type", "") or "").upper()
        return bool(thesis_id) or pos_type in {"BUY", "STARTER", "TACTICAL_BUY", "ACCUMULATE", "ADD"}

    def _alerts_from_portfolio_limits(
        self,
        portfolio: Portfolio,
        quotes: dict[str, dict[str, Decimal]],
    ) -> list[Alert]:
        alerts: list[Alert] = []
        # Legacy fallback (for positions with no thesis/position_type)
        legacy_stop_pct = _to_decimal(Config.STOP_LOSS_PCT)
        take_profit_pct = _to_decimal(Config.TAKE_PROFIT_PCT)

        # Position-type-specific hard stops
        _pos_type_stops: dict[str, float] = {
            "TACTICAL_BUY": Config.SELL_HARD_STOP_TACTICAL,
            "STARTER": Config.SELL_HARD_STOP_STARTER,
            "BUY": Config.SELL_HARD_STOP_BUY,
            "ACCUMULATE": Config.SELL_HARD_STOP_ACCUMULATE,
            "ADD": Config.SELL_HARD_STOP_BUY,
        }

        for pos in portfolio.positions:
            ticker = (pos.ticker or "").upper()
            if not ticker or ticker not in quotes:
                continue

            # Thesis-managed positions are owned by the new sell engine.
            # The legacy monitor should still handle general portfolio/market alerts,
            # but should not emit sell lifecycle messages for these positions.
            if self._is_sell_engine_managed_position(pos):
                continue

            avg_cost = _to_decimal(pos.avg_cost)
            if avg_cost <= 0:
                continue
            current_price = quotes[ticker]["price"]
            pnl_pct = (current_price - avg_cost) / avg_cost

            # ------------------------------------------------------------------
            # Hard stop: prefer thesis-tracked price, then position-type stop
            # ------------------------------------------------------------------
            hard_stop_price = getattr(pos, "hard_stop_price", None)
            if hard_stop_price is not None:
                hard_stop_price_dec = _to_decimal(hard_stop_price)
                if hard_stop_price_dec > 0 and current_price <= hard_stop_price_dec:
                    pos_type = getattr(pos, "position_type", "BUY")
                    msg = (
                        f"🚨 URGENT EXIT — HARD STOP BREACHED: {ticker} is at "
                        f"${current_price:.2f}, below hard stop ${hard_stop_price_dec:.2f}. "
                        f"Position type: {pos_type}. "
                        f"Down {abs(pnl_pct):.1%} from avg cost ${avg_cost:.2f}. "
                        "Execute sell immediately."
                    )
                    alerts.append(
                        Alert(
                            ticker=ticker,
                            alert_type="hard_stop_breached",
                            severity="CRITICAL",
                            message=msg,
                            metadata={
                                "avg_cost": str(avg_cost),
                                "current_price": str(current_price),
                                "hard_stop_price": str(hard_stop_price_dec),
                                "pnl_pct": str(pnl_pct),
                                "position_type": getattr(pos, "position_type", ""),
                                "thesis_id": getattr(pos, "thesis_id", ""),
                            },
                        )
                    )
                    continue  # Skip legacy stop check — hard stop supersedes it

            # Position-type-specific stop (if no explicit hard_stop_price)
            pos_type = getattr(pos, "position_type", None)
            if pos_type and pos_type in _pos_type_stops:
                type_stop = _to_decimal(_pos_type_stops[pos_type])
                if pnl_pct <= type_stop:
                    msg = (
                        f"🚨 STOP-LOSS ({pos_type}): {ticker} is down {abs(pnl_pct):.1%} "
                        f"(bought at ${avg_cost:.2f}, now ${current_price:.2f}). "
                        f"{pos_type} hard stop is {type_stop:.0%}. Consider exiting."
                    )
                    alerts.append(
                        Alert(
                            ticker=ticker,
                            alert_type="stop_loss",
                            severity="CRITICAL",
                            message=msg,
                            metadata={
                                "avg_cost": str(avg_cost),
                                "current_price": str(current_price),
                                "pnl_pct": str(pnl_pct),
                                "position_type": pos_type,
                            },
                        )
                    )
                    continue

            # Legacy stop (positions without position_type)
            if pnl_pct <= legacy_stop_pct:
                msg = (
                    f"STOP-LOSS: {ticker} is down {abs(pnl_pct):.1%} "
                    f"(bought at ${avg_cost:.2f}, now ${current_price:.2f}). Consider selling to limit losses."
                )
                alerts.append(
                    Alert(
                        ticker=ticker,
                        alert_type="stop_loss",
                        severity="CRITICAL",
                        message=msg,
                    )
                )
            elif pnl_pct >= take_profit_pct:
                # Check scale-out milestones for positions with theses
                scale_out_completed = getattr(pos, "scale_out_completed", None) or []
                pos_type_for_scale = getattr(pos, "position_type", None)
                milestone_msg = ""
                if pos_type_for_scale == "TACTICAL_BUY" and float(pnl_pct) >= 0.15:
                    if "+15%" not in scale_out_completed:
                        milestone_msg = " 📊 SCALE-OUT MILESTONE HIT (+15% — trim 25%?)"
                elif pos_type_for_scale == "BUY" and float(pnl_pct) >= 0.40:
                    if "+40%" not in scale_out_completed:
                        milestone_msg = " 📊 SCALE-OUT MILESTONE HIT (+40% — trim 15%?)"

                msg = (
                    f"TAKE-PROFIT: {ticker} is up {pnl_pct:.1%} "
                    f"(bought at ${avg_cost:.2f}, now ${current_price:.2f}). "
                    f"Consider taking some profits.{milestone_msg}"
                )
                alerts.append(
                    Alert(
                        ticker=ticker,
                        alert_type="take_profit",
                        severity="WARNING",
                        message=msg,
                        metadata={
                            "pnl_pct": str(pnl_pct),
                            "scale_out_milestone": bool(milestone_msg),
                        },
                    )
                )
        return alerts

    def _check_trailing_stops(
        self,
        portfolio: Portfolio,
        quotes: dict[str, dict[str, Decimal]],
    ) -> list[Alert]:
        """Check trailing stops for TACTICAL_BUY positions."""
        alerts: list[Alert] = []
        for pos in portfolio.positions:
            if self._is_sell_engine_managed_position(pos):
                continue
            pos_type = getattr(pos, "position_type", None)
            if pos_type != "TACTICAL_BUY":
                continue
            ticker = (pos.ticker or "").upper()
            trailing_stop = getattr(pos, "trailing_stop_price", None)
            if not trailing_stop or ticker not in quotes:
                continue
            current_price = quotes[ticker]["price"]
            trailing_stop_dec = _to_decimal(trailing_stop)
            if trailing_stop_dec > 0 and current_price <= trailing_stop_dec:
                avg_cost = _to_decimal(pos.avg_cost)
                pnl_pct = (current_price - avg_cost) / avg_cost if avg_cost > 0 else _to_decimal("0")
                msg = (
                    f"📉 TRAILING STOP TRIGGERED: {ticker} at ${current_price:.2f} "
                    f"is below trailing stop ${trailing_stop_dec:.2f}. "
                    f"P&L from entry: {pnl_pct:+.1%}. Exit TACTICAL_BUY position."
                )
                alerts.append(
                    Alert(
                        ticker=ticker,
                        alert_type="trailing_stop_breached",
                        severity="CRITICAL",
                        message=msg,
                        metadata={
                            "current_price": str(current_price),
                            "trailing_stop": str(trailing_stop_dec),
                            "pnl_pct": str(pnl_pct),
                            "thesis_id": getattr(pos, "thesis_id", ""),
                        },
                    )
                )
        return alerts

    def _alerts_from_daily_moves(self, quotes: dict[str, dict[str, Decimal]]) -> list[Alert]:
        alerts: list[Alert] = []
        for ticker, q in quotes.items():
            try:
                move = self._pct_change(q["price"], q["previous_close"])
                if abs(move) >= Decimal("0.0500"):
                    direction = "up" if move > 0 else "down"
                    alerts.append(
                        Alert(
                            ticker=ticker,
                            alert_type="daily_move",
                            severity="WARNING",
                            message=(
                                f"{ticker} moved {direction} {abs(move):.1%} today "
                                f"(${q['previous_close']:.2f} -> ${q['price']:.2f})."
                            ),
                            metadata={"daily_move_pct": str(move)},
                        )
                    )
            except Exception as e:
                logger.error(f"Daily move check failed for {ticker}: {e}")
        return alerts

    def _check_spy_crash(self) -> Optional[Alert]:
        try:
            quote = self.collector.yf.quote("SPY")
            if not quote:
                return None
            current = _to_decimal(quote.get("price", 0))
            prev_close = _to_decimal(quote.get("previous_close", 0))
            move = self._pct_change(current, prev_close)
            if move <= Decimal("-0.0300"):
                return Alert(
                    ticker="SPY",
                    alert_type="market_crash",
                    severity="CRITICAL",
                    message=(
                        f"Broad market stress: SPY is down {abs(move):.1%} today "
                        f"(${prev_close:.2f} -> ${current:.2f})."
                    ),
                    metadata={"daily_move_pct": str(move)},
                )
        except Exception as e:
            logger.error(f"SPY crash check failed: {e}")
        return None

    def _check_fear_greed_shift(self) -> Optional[Alert]:
        try:
            current = get_fear_greed_index()
            if not current:
                return None
            current_value = int(current.get("value", 0))
            previous_meta = self.alert_manager.get_meta("fear_greed_last")
            self.alert_manager.set_meta(
                "fear_greed_last",
                {"value": current_value, "recorded_at": _utcnow_iso()},
            )
            if not isinstance(previous_meta, dict):
                return None
            previous_value = int(previous_meta.get("value", current_value))
            delta = current_value - previous_value
            if abs(delta) > 20:
                direction = "up" if delta > 0 else "down"
                return Alert(
                    ticker="MARKET",
                    alert_type="fear_greed_shift",
                    severity="WARNING",
                    message=(
                        f"Fear & Greed shifted {direction} {abs(delta)} points "
                        f"({previous_value} -> {current_value}). Sentiment changed quickly."
                    ),
                    metadata={"previous": previous_value, "current": current_value},
                )
        except Exception as e:
            logger.error(f"Fear & Greed shift check failed: {e}")
        return None

    def _check_news_triggers(self, tickers: list[str]) -> list[Alert]:
        alerts: list[Alert] = []
        for ticker in tickers:
            try:
                articles = self.collector.finnhub.company_news(ticker, days_back=3) or []
                for article in articles[:15]:
                    if not isinstance(article, dict):
                        continue
                    text = f"{article.get('headline', '')} {article.get('summary', '')}".lower()
                    trigger = next((word for word in NEWS_TRIGGER_WORDS if word in text), None)
                    if not trigger:
                        continue
                    headline = str(article.get("headline", "")).strip()[:180]
                    alerts.append(
                        Alert(
                            ticker=ticker,
                            alert_type="news_trigger",
                            severity="WARNING",
                            message=f"{ticker} news trigger ({trigger}): {headline}",
                            metadata={"trigger_word": trigger},
                        )
                    )
            except Exception as e:
                logger.error(f"News trigger check failed for {ticker}: {e}")
        return alerts

    def _check_earnings_within_3_trading_days(self, tickers: list[str]) -> list[Alert]:
        alerts: list[Alert] = []
        for ticker in tickers:
            try:
                to_date = (_utcnow() + timedelta(days=10)).date().isoformat()
                from_date = _utcnow().date().isoformat()
                data = self.collector.finnhub._get(  # noqa: SLF001 - endpoint not wrapped in collector yet
                    "calendar/earnings",
                    {"symbol": ticker, "from": from_date, "to": to_date},
                )
                if not isinstance(data, dict):
                    continue
                earnings = data.get("earningsCalendar", [])
                if not isinstance(earnings, list):
                    continue
                for event in earnings:
                    if not isinstance(event, dict):
                        continue
                    event_date_raw = str(event.get("date", "")).strip()
                    if not event_date_raw:
                        continue
                    try:
                        event_date = date.fromisoformat(event_date_raw)
                    except ValueError:
                        continue
                    trading_days = self._estimate_trading_days_until(event_date)
                    if 0 <= trading_days <= 3:
                        eps_est = event.get("epsEstimate")
                        message = f"{ticker} earnings in {trading_days} trading day(s) on {event_date.isoformat()}."
                        if eps_est is not None:
                            message += f" EPS estimate: {eps_est}."
                        alerts.append(
                            Alert(
                                ticker=ticker,
                                alert_type="earnings_soon",
                                severity="INFO",
                                message=message,
                                metadata={"event_date": event_date.isoformat()},
                            )
                        )
            except Exception as e:
                logger.error(f"Earnings calendar check failed for {ticker}: {e}")
        return alerts

    @staticmethod
    def _estimate_trading_days_until(target_date: date) -> int:
        current = _utcnow().date()
        if target_date < current:
            return -1
        count = 0
        cursor = current
        while cursor < target_date:
            cursor += timedelta(days=1)
            if cursor.weekday() < 5:
                count += 1
        return count

    def _check_fomc_window(self) -> Optional[Alert]:
        try:
            today = _utcnow().date()
            for fomc_date in FOMC_DATES_2026:
                days = (fomc_date - today).days
                if 0 <= days <= 2:
                    return Alert(
                        ticker="MACRO",
                        alert_type="fomc_upcoming",
                        severity="INFO",
                        message=f"FOMC meeting is in {days} day(s) on {fomc_date.isoformat()}. Expect volatility.",
                        metadata={"fomc_date": fomc_date.isoformat()},
                    )
        except Exception as e:
            logger.error(f"FOMC window check failed: {e}")
        return None

    def _check_crisis_state(self) -> list[Alert]:
        """Check current crisis state and generate alerts on state transitions."""
        alerts: list[Alert] = []
        try:
            # Fetch current crisis signals
            signals = self.collector.collect_crisis_signals()
            spy_drawdown = signals.get("spy_drawdown") or 0.0
            vix = signals.get("vix") or 20.0
            fg = 50  # Default — may be overridden if fear_greed is available

            # Get current fear & greed
            fg_data = get_fear_greed_index()
            if fg_data:
                fg = int(fg_data.get("value", 50))

            # Evaluate state
            state_mgr = CrisisStateManager()
            previous_state = state_mgr.current_state
            new_state = state_mgr.evaluate_state(float(spy_drawdown), float(vix), fg)

            # Alert on state transitions
            if new_state != previous_state:
                severity_map = {
                    CrisisState.CORRECTION: "WARNING",
                    CrisisState.BEAR: "CRITICAL",
                    CrisisState.PANIC: "CRITICAL",
                    CrisisState.NORMAL: "INFO",
                }
                emoji_map = {
                    CrisisState.CORRECTION: "⚠️",
                    CrisisState.BEAR: "🐻",
                    CrisisState.PANIC: "🚨",
                    CrisisState.NORMAL: "✅",
                }
                severity = severity_map.get(new_state, "WARNING")
                emoji = emoji_map.get(new_state, "⚠️")

                if new_state != CrisisState.NORMAL:
                    message = (
                        f"{emoji} CRISIS MODE ACTIVATED: Market entered {new_state.upper()} "
                        f"(SPY down {abs(spy_drawdown):.1%} from 52w high, VIX={vix:.0f}). "
                        f"Previous state: {previous_state.upper()}. "
                        "Crisis Mode v3 protocols now active."
                    )
                else:
                    message = (
                        f"✅ Market returned to NORMAL from {previous_state.upper()} "
                        f"(SPY drawdown now {abs(spy_drawdown):.1%}, VIX={vix:.0f}). "
                        "Crisis mode deactivated — run crisis debrief."
                    )

                alerts.append(Alert(
                    ticker="MARKET",
                    alert_type="crisis_state_change",
                    severity=severity,
                    message=message,
                    metadata={
                        "previous_state": str(previous_state),
                        "new_state": str(new_state),
                        "spy_drawdown": spy_drawdown,
                        "vix": vix,
                        "fear_greed": fg,
                    },
                ))

            # Alert in Bear/Panic even without transition (reminder)
            elif new_state in (CrisisState.BEAR, CrisisState.PANIC):
                # Run fingerprinting for context
                try:
                    fp_engine = CrisisFingerprint()
                    fp = fp_engine.classify(signals)
                    dominant = fp.get("dominant", "UNKNOWN")
                    dom_prob = fp.get("dominant_prob", 0.0)

                    alerts.append(Alert(
                        ticker="MARKET",
                        alert_type="crisis_active",
                        severity="WARNING",
                        message=(
                            f"📊 CRISIS ACTIVE ({new_state.upper()}): SPY {abs(spy_drawdown):.1%} "
                            f"from peak. VIX={vix:.0f}. "
                            f"Fingerprint: {dominant} ({dom_prob:.0%}). "
                            f"Days in state: {state_mgr.days_in_state}."
                        ),
                        metadata={
                            "crisis_state": str(new_state),
                            "spy_drawdown": spy_drawdown,
                            "dominant_crisis_type": str(dominant),
                            "fingerprint_probs": fp.get("probabilities", {}),
                        },
                    ))
                except Exception as fp_e:
                    logger.warning(f"[monitor] Fingerprinting failed: {fp_e}")

        except Exception as e:
            logger.error(f"Crisis state check failed: {e}")
        return alerts

    def run_check(self) -> list[Alert]:
        """Run full monitoring pass and return generated alerts."""
        alerts: list[Alert] = []
        try:
            portfolio = Portfolio.load(self.portfolio_path)
            if not portfolio.positions:
                logger.info("Portfolio empty; monitor check produced no alerts")
                # Still run crisis state check even with empty portfolio
                crisis_alerts = self._check_crisis_state()
                return crisis_alerts

            tickers = sorted({p.ticker.upper() for p in portfolio.positions if p.ticker})
            quotes = self._fetch_position_quotes(tickers)

            alerts.extend(self._alerts_from_portfolio_limits(portfolio, quotes))
            alerts.extend(self._check_trailing_stops(portfolio, quotes))
            alerts.extend(self._alerts_from_daily_moves(quotes))
            alerts.extend(self._check_news_triggers(tickers))
            alerts.extend(self._check_earnings_within_3_trading_days(tickers))

            spy_alert = self._check_spy_crash()
            if spy_alert:
                alerts.append(spy_alert)

            fg_alert = self._check_fear_greed_shift()
            if fg_alert:
                alerts.append(fg_alert)

            fomc_alert = self._check_fomc_window()
            if fomc_alert:
                alerts.append(fomc_alert)

            # Crisis state check (only during market hours to save API calls)
            try:
                crisis_alerts = self._check_crisis_state()
                alerts.extend(crisis_alerts)
            except Exception as crisis_e:
                logger.error(f"Crisis state check failed: {crisis_e}")

            # Run News Sentinel
            if Config.SENTINEL_ENABLED:
                try:
                    # Lazy init sentinel to avoid circular imports
                    if self.sentinel is None:
                        self.sentinel = NewsSentinel(
                            collector=self.collector,
                            alert_manager=self.alert_manager,
                            config=Config,
                        )

                    sentinel_alerts = self.sentinel.run_scan()
                    alerts.extend(sentinel_alerts)
                    logger.info(f"[monitor] News Sentinel found {len(sentinel_alerts)} alert(s)")
                except Exception as sentinel_e:
                    logger.error(f"News Sentinel failed: {sentinel_e}")
        except Exception as e:
            logger.error(f"Monitor check failed: {e}")
        return alerts

    def run_and_dedupe(self) -> list[Alert]:
        """Run check + suppress duplicate alerts within 24h."""
        generated = self.run_check()
        return self.alert_manager.claim_new_alerts(generated, within_hours=24)

    def one_shot_status(self) -> dict[str, Any]:
        """Run one-shot health check and return useful metadata for CLI usage."""
        alerts = self.run_and_dedupe()
        return {
            "generated_at": _utcnow_iso(),
            "alert_count": len(alerts),
            "alerts": [asdict(a) for a in alerts],
            "telegram_message": self.alert_manager.format_for_telegram(alerts),
        }
