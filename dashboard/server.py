#!/usr/bin/env python3
"""Read-only HTTP server for the ARTHA operations dashboard.

The dashboard explains ARTHA's state; it never places orders or mutates the
trading journal. The only files it writes are its access token and an optional
intraday equity sample under data/dashboard/.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import date, datetime, time as dt_time, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DB = DATA / "artha.db"
DASH_DATA = DATA / "dashboard"
INDEX_FILE = ROOT / "dashboard" / "index.html"
CSS_FILE = ROOT / "dashboard" / "styles.css"
APP_FILE = ROOT / "dashboard" / "app.js"
TOKEN_FILE = DASH_DATA / "token.txt"
INTRADAY_FILE = DASH_DATA / "intraday_equity.jsonl"
PORT = int(os.getenv("ARTHA_DASHBOARD_PORT", "8787"))
CT = ZoneInfo("America/Chicago")
ET = ZoneInfo("America/New_York")
UTC = timezone.utc
DEFAULT_PILOT_BASE = 350.0
OPENCLAW_BIN = next(
    (
        str(path)
        for path in (
            Path("/opt/homebrew/bin/openclaw"),
            Path("/usr/local/bin/openclaw"),
            Path.home() / ".local" / "bin" / "openclaw",
        )
        if path.exists()
    ),
    "openclaw",
)

sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("artha.dashboard")


# ---------------------------------------------------------------------------
# Safe read helpers


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, ""):
            return default
        result = float(value)
        if result != result or result in (float("inf"), float("-inf")):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    except (OSError, ValueError, TypeError):
        return rows
    return rows


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except (TypeError, ValueError):
        return None


def _ct_stamp(value: Any, include_date: bool = True) -> str:
    parsed = value if isinstance(value, datetime) else _parse_iso(value)
    if not parsed:
        return "Unknown"
    local = parsed.astimezone(CT)
    return local.strftime("%b %-d, %-I:%M %p CT" if include_date else "%-I:%M %p CT")


def _relative_age(value: Any) -> str:
    parsed = value if isinstance(value, datetime) else _parse_iso(value)
    if not parsed:
        return "unknown"
    seconds = max(0, (_now_utc() - parsed).total_seconds())
    if seconds < 90:
        return "under a minute ago"
    if seconds < 3600:
        return f"{round(seconds / 60):.0f} minutes ago"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} hours ago"
    return f"{seconds / 86400:.0f} days ago"


def _plain(value: str) -> str:
    text = re.sub(r"```.*?```", " ", value or "", flags=re.S)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"\[E\d+\]", "", text)
    text = re.sub(r"[*_#`>|]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clip(value: str, limit: int = 280) -> str:
    text = _plain(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def _age_minutes(value: Any) -> float | None:
    parsed = value if isinstance(value, datetime) else _parse_iso(value)
    if not parsed:
        return None
    return max(0.0, (_now_utc() - parsed).total_seconds() / 60.0)


def _shell(args: list[str], timeout: int = 8) -> str:
    try:
        environment = os.environ.copy()
        system_paths = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
        current_paths = environment.get("PATH", "").split(":")
        environment["PATH"] = ":".join(dict.fromkeys(system_paths + [path for path in current_paths if path]))
        completed = subprocess.run(
            args,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=environment,
        )
        return completed.stdout if completed.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _db_rows(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    if not DB.exists():
        return []
    uri = f"file:{DB}?mode=ro"
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=2.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=2000")
        return [dict(row) for row in connection.execute(query, params).fetchall()]
    except sqlite3.Error as exc:
        logger.warning("dashboard database read failed: %s", exc)
        return []
    finally:
        if connection is not None:
            connection.close()


def _latest_file(folder: Path, pattern: str = "*.json") -> Path | None:
    try:
        files = [path for path in folder.rglob(pattern) if path.is_file()]
        return max(files, key=lambda path: path.stat().st_mtime) if files else None
    except OSError:
        return None


class TTLCache:
    def __init__(self) -> None:
        self._values: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str, ttl: float, producer: Callable[[], Any]) -> Any:
        now = time.monotonic()
        with self._lock:
            stored = self._values.get(key)
            if stored and now - stored[0] < ttl:
                return stored[1]
        value = producer()
        with self._lock:
            self._values[key] = (now, value)
        return value


_cache = TTLCache()


# ---------------------------------------------------------------------------
# Market, policy, and source-of-truth snapshots


def market_phase(now: datetime | None = None) -> dict[str, Any]:
    local = (now or _now_utc()).astimezone(CT)
    weekday = local.weekday() < 5
    current = local.time()
    regular_open = dt_time(8, 30) <= current < dt_time(15, 0)
    premarket = dt_time(3, 0) <= current < dt_time(8, 30)
    afterhours = dt_time(15, 0) <= current < dt_time(19, 0)
    if not weekday:
        phase = "Weekend"
    elif regular_open:
        phase = "Market open"
    elif premarket:
        phase = "Pre-market"
    elif afterhours:
        phase = "After hours"
    else:
        phase = "Market closed"
    return {
        "phase": phase,
        "open": bool(weekday and regular_open),
        "ct_time": local.strftime("%a %-I:%M %p CT"),
        "calendar_note": "Regular-hours estimate; broker checks remain the trading authority.",
    }


def build_policy() -> dict[str, Any]:
    defaults = {
        "max_positions": 20,
        "max_buys_per_day": 5,
        "max_position_dollars": 50.0,
        "max_auto_order_dollars": 25.0,
        "max_auto_daily_dollars": 50.0,
        "max_invested_pct": 90.0,
        "max_sector_pct": 30.0,
        "auto_buy_enabled": False,
        "auto_sell_enabled": False,
        "sentinel_enabled": False,
        "sell_news_escalation_enabled": False,
        "primary_scan": "11:30 AM CT",
        "afternoon_scan": "2:15 PM CT",
        "afternoon_scan_enabled": True,
        "source": "safe defaults",
    }
    try:
        from artha.config import Config

        return {
            "max_positions": int(Config.MAX_CONCURRENT_POSITIONS),
            "max_buys_per_day": int(Config.ROBINHOOD_MAX_TRADES_PER_DAY),
            "max_position_dollars": float(Config.ROBINHOOD_MAX_POSITION_DOLLARS),
            "max_auto_order_dollars": float(Config.ROBINHOOD_AUTO_BUY_MAX_ORDER_DOLLARS),
            "max_auto_daily_dollars": float(Config.ROBINHOOD_AUTO_BUY_MAX_DAILY_DOLLARS),
            "max_invested_pct": float(Config.MAX_INVESTED_PCT) * 100.0,
            "max_sector_pct": float(Config.MAX_SECTOR_PCT) * 100.0,
            "auto_buy_enabled": bool(Config.ROBINHOOD_AUTO_BUY_ENABLED),
            "auto_sell_enabled": bool(Config.ROBINHOOD_AUTO_SELL_ENABLED),
            "sentinel_enabled": bool(Config.SENTINEL_ENABLED),
            "sell_news_escalation_enabled": bool(Config.SELL_SENTINEL_COUNCIL_ESCALATION_ENABLED),
            "primary_scan": f"{int(Config.SCHEDULED_SCAN_HOUR_CT)}:{int(Config.SCHEDULED_SCAN_MINUTE_CT):02d} AM CT",
            "afternoon_scan": f"{int(Config.AFTERNOON_SCAN_HOUR_CT) - 12}:{int(Config.AFTERNOON_SCAN_MINUTE_CT):02d} PM CT",
            "afternoon_scan_enabled": bool(Config.AFTERNOON_SCAN_ENABLED),
            "source": "live ARTHA configuration",
        }
    except Exception as exc:
        logger.warning("could not load live policy: %s", exc)
        return defaults


def _supervisor() -> dict[str, Any]:
    raw = _read_json(DATA / "supervisor" / "latest.json", {})
    payload = raw.get("payload") if isinstance(raw, dict) else {}
    if not isinstance(payload, dict):
        payload = {}
    checks = payload.get("checks")
    if not isinstance(checks, list):
        checks = []
    return {
        "generated_at": raw.get("generated_at") or payload.get("generated_at"),
        "severity": str(raw.get("severity") or payload.get("severity") or "UNKNOWN").upper(),
        "checks": [item for item in checks if isinstance(item, dict)],
    }


def _supervisor_check(name: str) -> dict[str, Any]:
    for check in _supervisor().get("checks", []):
        if check.get("name") == name:
            return check
    return {}


def _broker_snapshot() -> dict[str, Any]:
    raw = _read_json(DATA / "robinhood" / "latest_snapshot.json", {})
    if not isinstance(raw, dict):
        return {}
    validation = raw.get("validation") if isinstance(raw.get("validation"), dict) else {}
    portfolio = raw.get("portfolio") if isinstance(raw.get("portfolio"), dict) else {}
    buying_power = portfolio.get("buying_power")
    if isinstance(buying_power, dict):
        buying_power = buying_power.get("buying_power")
    orders = raw.get("orders") if isinstance(raw.get("orders"), list) else []
    open_states = {"queued", "confirmed", "unconfirmed", "partially_filled", "pending"}
    open_orders = 0
    for order in orders:
        if not isinstance(order, dict):
            continue
        state = str(order.get("state") or order.get("status") or "").lower()
        if state in open_states:
            open_orders += 1
    return {
        "generated_at": raw.get("generated_at"),
        "status": str(validation.get("status") or "UNKNOWN").upper(),
        "fresh": bool(validation.get("fresh")),
        "warnings": validation.get("warnings") if isinstance(validation.get("warnings"), list) else [],
        "position_count": _safe_int(validation.get("position_count"), len(raw.get("positions") or [])),
        "total_value": _safe_float(portfolio.get("total_value")),
        "equity_value": _safe_float(portfolio.get("equity_value")),
        "cash": _safe_float(portfolio.get("cash")),
        "buying_power": _safe_float(buying_power),
        "open_orders": open_orders,
    }


def _contributions() -> list[dict[str, Any]]:
    rows = _read_jsonl(DATA / "contributions.jsonl")
    return sorted(rows, key=lambda row: str(row.get("t") or row.get("timestamp") or ""))


def _contribution_base() -> float:
    total = sum(_safe_float(row.get("amount"), 0.0) or 0.0 for row in _contributions())
    return round(total if total > 0 else DEFAULT_PILOT_BASE, 2)


def _previous_snapshot(today_ct: date) -> tuple[datetime | None, float | None]:
    rows = _db_rows(
        "SELECT timestamp, total_value FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 30"
    )
    for row in rows:
        stamp = _parse_iso(row.get("timestamp"))
        value = _safe_float(row.get("total_value"))
        if stamp and value is not None and stamp.astimezone(CT).date() < today_ct:
            return stamp, value
    return None, None


def build_vitals(policy: dict[str, Any]) -> dict[str, Any]:
    portfolio = _read_json(DATA / "portfolio.json", {})
    if not isinstance(portfolio, dict):
        portfolio = {}
    positions = portfolio.get("positions") if isinstance(portfolio.get("positions"), list) else []
    invested_internal = sum(
        _safe_float(item.get("market_value"), 0.0) or 0.0
        for item in positions
        if isinstance(item, dict)
    )
    cash_internal = _safe_float(portfolio.get("cash_available"), 0.0) or 0.0
    total_internal = invested_internal + cash_internal

    broker = _broker_snapshot()
    broker_usable = broker.get("status") == "PASS" and (_safe_float(broker.get("total_value"), 0.0) or 0.0) > 0
    total = (_safe_float(broker.get("total_value")) if broker_usable else None) or total_internal
    cash = (_safe_float(broker.get("cash")) if broker_usable else None)
    cash = cash if cash is not None else cash_internal
    invested = (_safe_float(broker.get("equity_value")) if broker_usable else None)
    invested = invested if invested is not None else invested_internal
    buying_power = (_safe_float(broker.get("buying_power")) if broker_usable else None)
    buying_power = buying_power if buying_power is not None else cash

    base = _contribution_base()
    pnl_total = total - base if total else None
    pnl_total_pct = pnl_total / base * 100.0 if pnl_total is not None and base else None

    now_ct = _now_utc().astimezone(CT)
    previous_at, previous_value = _previous_snapshot(now_ct.date())
    contribution_since = 0.0
    if previous_at:
        for row in _contributions():
            stamp = _parse_iso(row.get("t") or row.get("timestamp"))
            if stamp and stamp > previous_at:
                contribution_since += _safe_float(row.get("amount"), 0.0) or 0.0
    pnl_today = total - previous_value - contribution_since if previous_value is not None else None
    pnl_today_pct = pnl_today / previous_value * 100.0 if pnl_today is not None and previous_value else None

    max_invested_pct = (_safe_float(policy.get("max_invested_pct"), 90.0) or 90.0) / 100.0
    max_invested_dollars = total * max_invested_pct if total else 0.0
    exposure_headroom = max(0.0, max_invested_dollars - invested)
    deployable = max(0.0, min(buying_power, exposure_headroom))
    invested_pct = invested / total * 100.0 if total else 0.0
    position_count = len(positions)
    position_slots = max(0, _safe_int(policy.get("max_positions"), 20) - position_count)

    regime = _read_json(DATA / "regime_gate.json", {})
    regime_state = str(regime.get("state") or "UNKNOWN") if isinstance(regime, dict) else "UNKNOWN"
    control = _read_json(DATA / "robinhood" / "control.json", {})
    trading_disabled = bool(control.get("trading_disabled")) if isinstance(control, dict) else False
    buy_pause_reasons: list[str] = []
    if trading_disabled:
        buy_pause_reasons.append("The emergency trading switch is on.")
    if position_slots <= 0:
        buy_pause_reasons.append("The portfolio has reached its position limit.")
    if exposure_headroom < 10.0:
        buy_pause_reasons.append("Less than $10 remains below the 90% invested ceiling.")
    if regime_state == "HARD_RISK_OFF":
        buy_pause_reasons.append("The market regime is in defensive no-new-buy mode.")

    prices_at = _parse_iso(portfolio.get("last_updated")) if isinstance(portfolio, dict) else None
    mismatch = abs(total_internal - total) if total and total_internal else 0.0
    contributions = _contributions()
    recent_contribution = None
    if contributions:
        last = contributions[-1]
        stamp = _parse_iso(last.get("t") or last.get("timestamp"))
        if stamp and (_now_utc() - stamp) < timedelta(days=7):
            recent_contribution = {
                "amount": _safe_float(last.get("amount"), 0.0),
                "when": _ct_stamp(stamp),
                "note": _clip(str(last.get("note") or ""), 120),
            }

    return {
        "portfolio_missing": not bool(portfolio),
        "total_value": round(total, 2) if total else 0.0,
        "base": base,
        "pnl_total": round(pnl_total, 2) if pnl_total is not None else None,
        "pnl_total_pct": round(pnl_total_pct, 2) if pnl_total_pct is not None else None,
        "pnl_today": round(pnl_today, 2) if pnl_today is not None else None,
        "pnl_today_pct": round(pnl_today_pct, 2) if pnl_today_pct is not None else None,
        "pnl_baseline_label": _ct_stamp(previous_at).split(",")[0] if previous_at else "previous saved day",
        "cash": round(cash, 2),
        "buying_power": round(buying_power, 2),
        "invested": round(invested, 2),
        "invested_pct": round(invested_pct, 2),
        "max_invested_dollars": round(max_invested_dollars, 2),
        "exposure_headroom": round(exposure_headroom, 2),
        "deployable_now": round(deployable, 2),
        "position_count": position_count,
        "position_slots": position_slots,
        "regime": regime_state,
        "trading_disabled": trading_disabled,
        "kill_reason": _clip(str(control.get("reason") or ""), 180) if isinstance(control, dict) else "",
        "buys_paused": bool(buy_pause_reasons),
        "buy_pause_reasons": buy_pause_reasons,
        "prices_at": prices_at.isoformat() if prices_at else None,
        "prices_age_minutes": round(_age_minutes(prices_at), 1) if prices_at else None,
        "broker_source": "Robinhood snapshot" if broker_usable else "ARTHA internal ledger",
        "broker_status": broker.get("status", "UNKNOWN"),
        "broker_fresh": bool(broker.get("fresh")),
        "broker_age_minutes": round(_age_minutes(broker.get("generated_at")), 1) if broker.get("generated_at") else None,
        "broker_open_orders": broker.get("open_orders", 0),
        "reconciliation_difference": round(mismatch, 2),
        "recent_contribution": recent_contribution,
    }


# ---------------------------------------------------------------------------
# Portfolio and performance


def _active_theses() -> dict[str, dict[str, Any]]:
    rows = _db_rows(
        "SELECT ticker, thesis_id, position_type, thesis_summary, hard_stop_price, "
        "trailing_stop_price, thesis_health_score, last_review_date, next_review_date, "
        "sell_cooldown_until, entry_date, updated_at FROM position_theses "
        "WHERE status='active' ORDER BY updated_at DESC"
    )
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        ticker = str(row.get("ticker") or "").upper()
        if ticker and ticker not in out:
            out[ticker] = row
    return out


def _latest_sell_sessions() -> dict[str, dict[str, Any]]:
    rows = _db_rows(
        "SELECT ticker, trigger_type, sell_score, action, synthesis_report, "
        "next_review_date, health_score_after, created_at FROM sell_sessions "
        "ORDER BY created_at DESC LIMIT 120"
    )
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        ticker = str(row.get("ticker") or "").upper()
        if ticker and ticker not in out:
            out[ticker] = row
    return out


def _active_trade_actions() -> dict[str, dict[str, Any]]:
    rows = _db_rows(
        "SELECT ticker, side, action_type, status, message, created_at FROM trade_actions "
        "WHERE status IN ('pending','review_ready','price_gate_passed','queued','submitted',"
        "'unconfirmed','partially_filled') ORDER BY created_at DESC"
    )
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        ticker = str(row.get("ticker") or "").upper()
        if ticker and ticker not in out:
            out[ticker] = row
    return out


def build_positions(vitals: dict[str, Any]) -> list[dict[str, Any]]:
    portfolio = _read_json(DATA / "portfolio.json", {})
    positions = portfolio.get("positions") if isinstance(portfolio, dict) else []
    if not isinstance(positions, list):
        return []
    theses = _active_theses()
    sell_reviews = _latest_sell_sessions()
    pending = _active_trade_actions()
    total = _safe_float(vitals.get("total_value"), 0.0) or 0.0
    now = _now_utc()
    result: list[dict[str, Any]] = []

    for raw in positions:
        if not isinstance(raw, dict):
            continue
        ticker = str(raw.get("ticker") or "").upper()
        if not ticker:
            continue
        thesis = theses.get(ticker, {})
        shares = _safe_float(raw.get("shares"), 0.0) or 0.0
        cost = _safe_float(raw.get("avg_cost"), 0.0) or 0.0
        price = _safe_float(raw.get("current_price"), cost) or cost
        market_value = _safe_float(raw.get("market_value"), shares * price) or shares * price
        cost_value = shares * cost
        pnl = market_value - cost_value
        pnl_pct = pnl / cost_value * 100.0 if cost_value else 0.0
        hard_stop = _safe_float(raw.get("hard_stop_price"), _safe_float(thesis.get("hard_stop_price")))
        trail_stop = _safe_float(raw.get("trailing_stop_price"), _safe_float(thesis.get("trailing_stop_price")))
        stops = [value for value in (hard_stop, trail_stop) if value is not None and value > 0]
        stop = max(stops) if stops else None
        stop_dist = (price - stop) / price * 100.0 if stop and price else None
        opened = _parse_iso(raw.get("opened_at") or raw.get("entry_date") or thesis.get("entry_date"))
        days_held = max(0, (now.date() - opened.date()).days) if opened else None
        review = sell_reviews.get(ticker, {})
        active = pending.get(ticker)
        due_at = _parse_iso(raw.get("next_sell_review") or thesis.get("next_review_date"))
        review_due = bool(due_at and due_at <= now)
        status = "Protected"
        status_tone = "good"
        if active:
            status = f"{str(active.get('side') or '').title()} is {str(active.get('status') or '').replace('_', ' ')}"
            status_tone = "attention"
        elif stop_dist is not None and stop_dist <= 0:
            status = "Stop level breached; exit review expected"
            status_tone = "critical"
        elif review_due:
            status = "Sell review due"
            status_tone = "attention"
        elif stop is None:
            status = "No stop found"
            status_tone = "critical"

        result.append(
            {
                "ticker": ticker,
                "shares": round(shares, 6),
                "avg_cost": round(cost, 4),
                "price": round(price, 4),
                "market_value": round(market_value, 2),
                "weight_pct": round(market_value / total * 100.0, 2) if total else 0.0,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "sector": str(raw.get("sector") or "Unclassified"),
                "industry": str(raw.get("industry") or ""),
                "position_type": str(raw.get("position_type") or thesis.get("position_type") or "POSITION"),
                "days_held": days_held,
                "hard_stop": round(hard_stop, 4) if hard_stop else None,
                "trailing_stop": round(trail_stop, 4) if trail_stop else None,
                "active_stop": round(stop, 4) if stop else None,
                "stop_dist_pct": round(stop_dist, 2) if stop_dist is not None else None,
                "stop_locked_profit": bool(stop and stop > cost),
                "next_review": due_at.isoformat() if due_at else None,
                "health_score": _safe_int(thesis.get("thesis_health_score"), _safe_int(review.get("health_score_after"), 100)),
                "last_sell_action": str(review.get("action") or "No review yet"),
                "last_sell_score": _safe_float(review.get("sell_score")),
                "last_sell_review": review.get("created_at"),
                "pending_action": active,
                "status": status,
                "status_tone": status_tone,
                "thesis_summary": _clip(str(thesis.get("thesis_summary") or raw.get("original_thesis") or ""), 240),
            }
        )
    result.sort(key=lambda item: item["market_value"], reverse=True)
    return result


def build_sectors(positions: list[dict[str, Any]], vitals: dict[str, Any]) -> list[dict[str, Any]]:
    check = _supervisor_check("position_classification")
    exposures = check.get("sector_exposures")
    if isinstance(exposures, list) and exposures:
        return [
            {
                "sector": str(row.get("sector") or "Unclassified"),
                "market_value": round(_safe_float(row.get("market_value"), 0.0) or 0.0, 2),
                "pct_nav": round(_safe_float(row.get("pct_nav"), 0.0) or 0.0, 2),
                "pct_invested": round(_safe_float(row.get("pct_invested"), 0.0) or 0.0, 2),
            }
            for row in exposures
            if isinstance(row, dict)
        ]
    grouped: dict[str, float] = {}
    for position in positions:
        sector = str(position.get("sector") or "Unclassified")
        grouped[sector] = grouped.get(sector, 0.0) + (_safe_float(position.get("market_value"), 0.0) or 0.0)
    total = _safe_float(vitals.get("total_value"), 0.0) or 0.0
    invested = _safe_float(vitals.get("invested"), 0.0) or 0.0
    return [
        {
            "sector": sector,
            "market_value": round(value, 2),
            "pct_nav": round(value / total * 100.0, 2) if total else 0.0,
            "pct_invested": round(value / invested * 100.0, 2) if invested else 0.0,
        }
        for sector, value in sorted(grouped.items(), key=lambda item: item[1], reverse=True)
    ]


def build_performance(vitals: dict[str, Any], positions: list[dict[str, Any]]) -> dict[str, Any]:
    learning = _supervisor_check("execution_learning")
    summary = learning.get("summary") if isinstance(learning.get("summary"), dict) else {}
    episodes = summary.get("episodes") if isinstance(summary.get("episodes"), dict) else {}
    sell_events = summary.get("sell_events") if isinstance(summary.get("sell_events"), dict) else {}
    if not episodes:
        rows = _db_rows(
            "SELECT status, realized_pnl, return_pct FROM position_trade_episodes"
        )
        closed = [row for row in rows if str(row.get("status") or "").lower() == "closed"]
        wins = [row for row in closed if (_safe_float(row.get("realized_pnl"), 0.0) or 0.0) > 0]
        losses = [row for row in closed if (_safe_float(row.get("realized_pnl"), 0.0) or 0.0) < 0]
        episodes = {
            "total": len(rows),
            "closed": len(closed),
            "open": len(rows) - len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": len(wins) / len(closed) * 100.0 if closed else None,
            "realized_pnl_dollars": sum(_safe_float(row.get("realized_pnl"), 0.0) or 0.0 for row in closed),
        }

    best = max(positions, key=lambda row: row.get("pnl_pct", 0.0), default=None)
    weakest = min(positions, key=lambda row: row.get("pnl_pct", 0.0), default=None)
    pnl = _safe_float(vitals.get("pnl_total"), 0.0) or 0.0
    direction = "above" if pnl >= 0 else "below"
    return {
        "headline": f"The account is ${abs(pnl):.2f} {direction} the amount contributed.",
        "total_return_dollars": round(pnl, 2),
        "total_return_pct": vitals.get("pnl_total_pct"),
        "realized_pnl": round(_safe_float(episodes.get("realized_pnl_dollars"), 0.0) or 0.0, 2),
        "unrealized_pnl": round(sum(_safe_float(row.get("pnl"), 0.0) or 0.0 for row in positions), 2),
        "episodes": {
            "total": _safe_int(episodes.get("total")),
            "open": _safe_int(episodes.get("open")),
            "closed": _safe_int(episodes.get("closed")),
            "wins": _safe_int(episodes.get("wins")),
            "losses": _safe_int(episodes.get("losses")),
            "win_rate_pct": _safe_float(episodes.get("win_rate_pct")),
            "average_win": _safe_float(episodes.get("average_win_dollars")),
            "average_loss": _safe_float(episodes.get("average_loss_dollars")),
            "expectancy": _safe_float(episodes.get("expectancy_dollars")),
            "average_return_pct": _safe_float(episodes.get("average_return_pct")),
        },
        "sell_events": {
            "total": _safe_int(sell_events.get("total")),
            "exits": _safe_int(sell_events.get("exits")),
            "trims": _safe_int(sell_events.get("trims")),
            "trim_share_pct": _safe_float(sell_events.get("trim_share_pct")),
        },
        "best_open": best,
        "weakest_open": weakest,
        "accounting_quality": str((summary.get("data_quality") or {}).get("status") or "unknown"),
        "sample_warning": "This is an early pilot. Closed-trade statistics are descriptive, not proof that the strategy will keep winning.",
    }


def _record_intraday_sample(total: float) -> None:
    try:
        if not market_phase()["open"] or total <= 0:
            return
        DASH_DATA.mkdir(parents=True, exist_ok=True)
        rows = _read_jsonl(INTRADAY_FILE)
        last = _parse_iso(rows[-1].get("t")) if rows else None
        if last and _now_utc() - last < timedelta(minutes=10):
            return
        with INTRADAY_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"t": _now_utc().isoformat(), "v": round(total, 2)}) + "\n")
        if len(rows) > 3000:
            kept = rows[-2500:]
            INTRADAY_FILE.write_text(
                "".join(json.dumps(row) + "\n" for row in kept), encoding="utf-8"
            )
    except Exception as exc:
        logger.warning("intraday sample failed: %s", exc)


EQUITY_HISTORY_START = "2026-06-16T18:00:00+00:00"


def _remove_isolated_equity_outliers(points: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Remove isolated accounting spikes while preserving sustained market moves."""
    if len(points) < 5:
        return points, 0
    kept: list[dict[str, Any]] = []
    removed = 0
    for index, point in enumerate(points):
        if index < 2 or index >= len(points) - 2:
            kept.append(point)
            continue
        neighbors = points[index - 2:index] + points[index + 1:index + 3]
        values = [_safe_float(row.get("v")) for row in neighbors]
        values = [value for value in values if value is not None and value > 0]
        current = _safe_float(point.get("v"))
        if current is None or len(values) < 4:
            kept.append(point)
            continue
        median = statistics.median(values)
        neighbor_spread = (max(values) - min(values)) / median if median else 0.0
        deviation = abs(current - median) / median if median else 0.0
        if deviation > 0.12 and neighbor_spread < 0.06:
            removed += 1
            continue
        kept.append(point)
    return kept, removed


def build_equity_curve(vitals: dict[str, Any]) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for row in _db_rows(
        "SELECT timestamp, total_value FROM portfolio_snapshots "
        "WHERE timestamp >= ? ORDER BY timestamp",
        (EQUITY_HISTORY_START,),
    ):
        stamp = _parse_iso(row.get("timestamp"))
        value = _safe_float(row.get("total_value"))
        if stamp and value is not None and value > 0:
            points.append({"t": stamp.isoformat(), "v": round(value, 2), "source": "daily snapshot"})
    for row in _read_jsonl(INTRADAY_FILE):
        stamp = _parse_iso(row.get("t"))
        value = _safe_float(row.get("v"))
        if stamp and value is not None and value > 0:
            points.append({"t": stamp.isoformat(), "v": round(value, 2), "source": "intraday sample"})
    now_value = _safe_float(vitals.get("total_value"))
    if now_value:
        points.append({"t": _now_utc().isoformat(), "v": round(now_value, 2), "source": "current"})
    points.sort(key=lambda row: row["t"])
    points, excluded = _remove_isolated_equity_outliers(points)
    if len(points) > 360:
        stride = len(points) / 360.0
        sampled = [points[min(int(index * stride), len(points) - 1)] for index in range(360)]
        sampled[-1] = points[-1]
        points = sampled

    markers: list[dict[str, Any]] = []
    for row in _db_rows(
        "SELECT COALESCE(filled_at, created_at) AS event_at, ticker, side, notional, status "
        "FROM execution_orders WHERE status IN ('filled','submitted','partially_filled') ORDER BY 1"
    ):
        stamp = _parse_iso(row.get("event_at"))
        if stamp:
            markers.append(
                {
                    "t": stamp.isoformat(),
                    "side": str(row.get("side") or "").lower(),
                    "ticker": row.get("ticker"),
                    "notional": _safe_float(row.get("notional")),
                }
            )
    return {
        "points": points,
        "markers": markers,
        "base": _contribution_base(),
        "data_quality": {
            "excluded_isolated_outliers": excluded,
            "explanation": "Isolated jumps that disagree with both neighboring snapshots are hidden from the chart only; raw records remain untouched.",
        },
    }


# ---------------------------------------------------------------------------
# Decisions, execution, and activity


BUY_VERDICTS = {"BUY", "STARTER", "STARTER_BUY", "TACTICAL_BUY", "ACCUMULATE", "ADD"}


def _dossier_index() -> dict[str, list[dict[str, Any]]]:
    def produce() -> dict[str, list[dict[str, Any]]]:
        index: dict[str, list[dict[str, Any]]] = {}
        try:
            files = sorted(
                (path for path in (DATA / "decision_dossiers").rglob("*.json") if path.is_file()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )[:240]
        except OSError:
            files = []
        for path in files:
            payload = _read_json(path, {})
            if not isinstance(payload, dict):
                continue
            ticker = str(payload.get("ticker") or "").upper()
            generated = _parse_iso(payload.get("generated_at") or (payload.get("decision") or {}).get("created_at"))
            if not ticker or not generated:
                continue
            payload["_generated_dt"] = generated
            index.setdefault(ticker, []).append(payload)
        return index

    return _cache.get("dossier_index", 30, produce)


def _closest_dossier(ticker: str, stamp: datetime | None) -> dict[str, Any]:
    choices = _dossier_index().get(ticker.upper(), [])
    if not choices:
        return {}
    if not stamp:
        return choices[0]
    eligible = [
        item for item in choices
        if isinstance(item.get("_generated_dt"), datetime)
        and abs((item["_generated_dt"] - stamp).total_seconds()) <= 3600
    ]
    return min(eligible, key=lambda item: abs((item["_generated_dt"] - stamp).total_seconds())) if eligible else {}


def _extract_block_reason(action: dict[str, Any], order: dict[str, Any] | None = None) -> str:
    objects = [
        _json_dict(action.get("result_json")),
        _json_dict(action.get("payload_json")),
        _json_dict((order or {}).get("response_json")),
        _json_dict((order or {}).get("guardrail_json")),
    ]
    candidates: list[str] = []

    def walk(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                walk(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                walk(child, key)
        elif isinstance(value, str) and key.lower() in {
            "blocked_reasons", "reason", "reasons", "failure_reason", "warning", "warnings"
        }:
            clean = _clip(value, 260)
            if clean and clean not in candidates:
                candidates.append(clean)

    for obj in objects:
        walk(obj)
    if candidates:
        return candidates[0]
    return _clip(str(action.get("message") or (order or {}).get("rationale") or "A required execution check did not pass."), 260)


def _execution_actions(limit: int = 260) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    actions = _db_rows(
        "SELECT id, action_id, created_at, updated_at, expires_at, status, action_type, "
        "ticker, side, execution_order_row, payload_json, result_json, message "
        "FROM trade_actions ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    orders = _db_rows(
        "SELECT id, order_intent_id, created_at, updated_at, ticker, side, order_type, "
        "quantity, notional, limit_price, estimated_price, status, response_json, "
        "filled_at, rationale FROM execution_orders ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    return actions, {int(row["id"]): row for row in orders if row.get("id") is not None}


def _match_action(
    ticker: str,
    stamp: datetime | None,
    actions: list[dict[str, Any]],
    side: str = "buy",
) -> dict[str, Any] | None:
    if not stamp:
        return None
    matches: list[tuple[float, dict[str, Any]]] = []
    for action in actions:
        if str(action.get("ticker") or "").upper() != ticker.upper():
            continue
        if str(action.get("side") or "").lower() != side:
            continue
        created = _parse_iso(action.get("created_at"))
        if not created:
            continue
        delta = (created - stamp).total_seconds()
        if -180 <= delta <= 8 * 3600:
            matches.append((abs(delta), action))
    return min(matches, key=lambda item: item[0])[1] if matches else None


def _execution_view(action: dict[str, Any] | None, order_map: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    if not action:
        return None
    order_id = action.get("execution_order_row")
    order = order_map.get(_safe_int(order_id)) if order_id is not None else None
    status = str(action.get("status") or (order or {}).get("status") or "unknown").lower()
    result = _json_dict(action.get("result_json"))
    fill_price = _safe_float(result.get("average_price"))
    fill_qty = _safe_float(result.get("quantity"))
    if fill_price is None and order:
        response = _json_dict(order.get("response_json"))
        order_payload = response.get("order") if isinstance(response.get("order"), dict) else response
        fill_price = _safe_float(order_payload.get("average_price"))
        fill_qty = fill_qty if fill_qty is not None else _safe_float(order_payload.get("cumulative_quantity"))
    notional = _safe_float((order or {}).get("notional"))
    if notional is None and fill_price is not None and fill_qty is not None:
        notional = fill_price * fill_qty

    if status == "filled":
        label = "Filled automatically"
        tone = "good"
        reason = "The council approved the idea, all live broker checks passed, and Robinhood reported the fill."
    elif status in {"blocked", "review_blocked", "failed", "expired", "canceled"}:
        label = "Blocked before trading"
        tone = "critical" if status == "failed" else "attention"
        reason = _extract_block_reason(action, order)
    elif status in {"pending", "review_ready", "price_gate_passed", "queued", "submitted", "unconfirmed", "partially_filled"}:
        label = "Still in the execution pipeline"
        tone = "attention"
        reason = "ARTHA is handling this automatically; no user approval is required."
    else:
        label = status.replace("_", " ").title()
        tone = "neutral"
        reason = _clip(str(action.get("message") or "Execution state recorded."), 260)
    return {
        "status": status,
        "label": label,
        "tone": tone,
        "reason": reason,
        "side": str(action.get("side") or "").lower(),
        "action_type": str(action.get("action_type") or "").replace("_", " "),
        "notional": round(notional, 2) if notional is not None else None,
        "quantity": round(fill_qty, 6) if fill_qty is not None else _safe_float((order or {}).get("quantity")),
        "fill_price": round(fill_price, 4) if fill_price is not None else None,
        "reference_price": _safe_float((order or {}).get("limit_price") or (order or {}).get("estimated_price")),
        "when": action.get("updated_at") or action.get("created_at"),
    }


def build_decision_journeys(limit: int = 40) -> list[dict[str, Any]]:
    recommendations = _db_rows(
        "SELECT id, timestamp, session_id, ticker, action, rationale, confidence, "
        "price_at_recommendation, status FROM recommendations ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    )
    actions, order_map = _execution_actions()
    journeys: list[dict[str, Any]] = []
    for row in recommendations:
        stamp = _parse_iso(row.get("timestamp"))
        ticker = str(row.get("ticker") or "").upper()
        verdict = str(row.get("action") or "UNKNOWN").upper()
        dossier = _closest_dossier(ticker, stamp)
        decision = dossier.get("decision") if isinstance(dossier.get("decision"), dict) else {}
        analysts = dossier.get("analysts") if isinstance(dossier.get("analysts"), dict) else {}
        buy_side = verdict in BUY_VERDICTS
        action = _match_action(ticker, stamp, actions, "buy") if buy_side else None
        execution = _execution_view(action, order_map)
        if not buy_side:
            execution = {
                "status": "not_requested",
                "label": "No broker order requested",
                "tone": "neutral",
                "reason": "The council did not issue a buy-side verdict, so the Execution Officer correctly stopped here.",
                "side": "buy",
                "action_type": "none",
                "notional": None,
                "quantity": None,
                "fill_price": None,
                "reference_price": None,
                "when": row.get("timestamp"),
            }
        elif execution is None:
            execution = {
                "status": "missing",
                "label": "No execution record found",
                "tone": "critical",
                "reason": "The council issued a buy-side verdict, but the recent journal does not contain a matching broker action. This needs investigation.",
                "side": "buy",
                "action_type": "unknown",
                "notional": None,
                "quantity": None,
                "fill_price": None,
                "reference_price": None,
                "when": row.get("timestamp"),
            }

        votes = []
        for role in ("fundamental", "technical", "contrarian"):
            analyst = analysts.get(role) if isinstance(analysts.get(role), dict) else {}
            if analyst:
                votes.append(
                    {
                        "role": "Risk" if role == "contrarian" else role.title(),
                        "verdict": str(analyst.get("verdict") or "Unknown"),
                        "confidence": _safe_int(analyst.get("confidence"), 0),
                    }
                )
        recommended = str(decision.get("recommended_action") or row.get("rationale") or "")
        source_audit = dossier.get("source_audit") if isinstance(dossier.get("source_audit"), dict) else {}
        journeys.append(
            {
                "recommendation_id": row.get("id"),
                "session_id": row.get("session_id"),
                "ticker": ticker,
                "when": stamp.isoformat() if stamp else row.get("timestamp"),
                "when_label": _ct_stamp(stamp),
                "verdict": verdict,
                "buy_side": buy_side,
                "score": _safe_float(decision.get("opportunity_score") or decision.get("adjusted_score")),
                "confidence": _safe_int(decision.get("confidence"), _safe_int(row.get("confidence"))),
                "price": _safe_float(row.get("price_at_recommendation")),
                "allocation": decision.get("allocation"),
                "thesis_type": decision.get("thesis_type"),
                "why": _clip(recommended, 440),
                "votes": votes,
                "evidence_count": _safe_int(source_audit.get("evidence_count")),
                "data_gaps": source_audit.get("gaps") if isinstance(source_audit.get("gaps"), list) else [],
                "execution": execution,
                "risk_gate_blocked": not bool(decision.get("hard_risk_gate_passed", True)),
                "risk_gate_reason": _clip(str(decision.get("hard_risk_gate_reason") or ""), 220),
            }
        )
    return journeys


def build_sell_reviews(limit: int = 24) -> list[dict[str, Any]]:
    rows = _db_rows(
        "SELECT ticker, trigger_type, sell_score, action, synthesis_report, "
        "next_review_date, health_score_after, created_at FROM sell_sessions "
        "ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    actions, order_map = _execution_actions()
    result: list[dict[str, Any]] = []
    for row in rows:
        stamp = _parse_iso(row.get("created_at"))
        ticker = str(row.get("ticker") or "").upper()
        action_name = str(row.get("action") or "HOLD").upper()
        matched = _match_action(ticker, stamp, actions, "sell") if action_name in {"TRIM", "EXIT", "SELL"} else None
        execution = _execution_view(matched, order_map)
        if action_name not in {"TRIM", "EXIT", "SELL"}:
            execution = {
                "status": "not_requested",
                "label": "No sell ordered",
                "tone": "neutral",
                "reason": "The sell council chose to keep the shares.",
            }
        elif execution is None:
            execution = {
                "status": "missing",
                "label": "No matching sell action found",
                "tone": "critical",
                "reason": "A sell-side verdict was recorded without a matching recent execution action.",
            }
        result.append(
            {
                "ticker": ticker,
                "when": row.get("created_at"),
                "when_label": _ct_stamp(stamp),
                "trigger": str(row.get("trigger_type") or "review").replace("_", " "),
                "action": action_name,
                "sell_score": _safe_float(row.get("sell_score")),
                "health_score": _safe_int(row.get("health_score_after"), 0),
                "reason": _clip(str(row.get("synthesis_report") or ""), 440),
                "next_review": row.get("next_review_date"),
                "execution": execution,
            }
        )
    return result


def build_activity_today() -> list[dict[str, Any]]:
    now_ct = _now_utc().astimezone(CT)
    start = datetime.combine(now_ct.date(), dt_time.min, tzinfo=CT).astimezone(UTC)
    events: list[tuple[datetime, str, str, str]] = []

    for row in _db_rows(
        "SELECT timestamp, ticker, action FROM recommendations WHERE timestamp >= ? ORDER BY timestamp",
        (start.isoformat(),),
    ):
        stamp = _parse_iso(row.get("timestamp"))
        if stamp:
            verdict = str(row.get("action") or "decision").replace("_", " ")
            events.append((stamp, "decision", "Council decision", f"{row.get('ticker')}: {verdict}"))

    for row in _db_rows(
        "SELECT COALESCE(filled_at, created_at) event_at, ticker, side, notional, status "
        "FROM execution_orders WHERE created_at >= ? ORDER BY created_at",
        (start.isoformat(),),
    ):
        stamp = _parse_iso(row.get("event_at"))
        if stamp:
            side = str(row.get("side") or "order").title()
            amount = _safe_float(row.get("notional"))
            amount_text = f" ${amount:.2f}" if amount is not None else ""
            events.append((stamp, "trade", f"{side} order", f"{row.get('ticker')}{amount_text}: {str(row.get('status') or '').replace('_', ' ')}"))

    for row in _db_rows(
        "SELECT timestamp, session_type, tickers_analyzed FROM sessions WHERE timestamp >= ? ORDER BY timestamp",
        (start.isoformat(),),
    ):
        stamp = _parse_iso(row.get("timestamp"))
        if stamp:
            tickers = [token for token in re.split(r"[,\s]+", str(row.get("tickers_analyzed") or "")) if token]
            label = str(row.get("session_type") or "scan").replace("_", " ").title()
            events.append((stamp, "scan", label, f"Reviewed {len(tickers)} stock{'s' if len(tickers) != 1 else ''}."))

    events.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "when": stamp.isoformat(),
            "time": _ct_stamp(stamp, include_date=False),
            "kind": kind,
            "title": title,
            "detail": detail,
        }
        for stamp, kind, title, detail in events[:80]
    ]


# ---------------------------------------------------------------------------
# Scan pipeline, learning, alarms, and health


def _latest_scout() -> dict[str, Any]:
    path = _latest_file(DATA / "opportunity_scout")
    payload = _read_json(path, {}) if path else {}
    return payload if isinstance(payload, dict) else {}


def build_pipeline() -> dict[str, Any]:
    rank = _supervisor_check("rank_coverage")
    scout = _latest_scout()
    session_id = str(scout.get("session_id") or "")
    routing = _db_rows(
        "SELECT ticker, candidate_rank, lane, bucket, reason_code, reason, route_score, "
        "funnel_score, price, live_price, spread_pct, dollar_volume FROM scan_routing_decisions "
        "WHERE session_id=? ORDER BY candidate_rank",
        (session_id,),
    ) if session_id else []
    bucket_counts = Counter(str(row.get("bucket") or row.get("lane") or "unknown") for row in routing)
    reason_counts = Counter(str(row.get("reason_code") or "unknown") for row in routing)
    selected = scout.get("selected_for_council") if isinstance(scout.get("selected_for_council"), list) else []
    batches = scout.get("batches") if isinstance(scout.get("batches"), list) else []
    first_batch = batches[0] if batches and isinstance(batches[0], list) else selected[:8]
    cards = scout.get("ranked_cards") if isinstance(scout.get("ranked_cards"), list) else []
    broker_clean = len([card for card in cards if isinstance(card, dict) and card.get("allowed_for_buy_now")])
    recommendation_rows = _db_rows(
        "SELECT action, COUNT(*) count FROM recommendations WHERE session_id=? GROUP BY action",
        (session_id,),
    ) if session_id else []
    action_mix = {str(row.get("action") or "UNKNOWN"): _safe_int(row.get("count")) for row in recommendation_rows}
    council_count = sum(action_mix.values())
    buy_count = sum(count for action, count in action_mix.items() if action.upper() in BUY_VERDICTS)

    universe = _safe_int(rank.get("universe_size"))
    if not universe:
        message_match = re.search(r"history for ([\d,]+)/([\d,]+)", str(rank.get("message") or ""))
        universe = int(message_match.group(2).replace(",", "")) if message_match else 0
    usable = 0
    message_match = re.search(r"history for ([\d,]+)/([\d,]+)", str(rank.get("message") or ""))
    if message_match:
        usable = int(message_match.group(1).replace(",", ""))
    status_counts = rank.get("status_counts") if isinstance(rank.get("status_counts"), dict) else {}
    ranked = _safe_int(status_counts.get("ranked"))
    lottery = _safe_int(status_counts.get("lottery_excluded"))
    missing = _safe_int(status_counts.get("price_history_missing"))
    funnel_count = len(routing)
    scout_cards = len(cards)

    stages = [
        {
            "number": 1,
            "name": "Read the market",
            "metric": f"{usable:,} of {universe:,} stocks had usable price history" if universe else "Coverage record unavailable",
            "status": "good" if str(rank.get("status") or "").upper() == "PASS" else "attention",
            "plain": "ARTHA first gathers broad market history. A stock cannot quietly disappear without being counted as missing or excluded.",
        },
        {
            "number": 2,
            "name": "Rank and remove lottery behavior",
            "metric": f"{ranked:,} ranked · {lottery:,} spike-like names excluded · {missing:,} missing",
            "status": "good" if ranked else "attention",
            "plain": "The fast mathematical screen scores momentum, trend quality, entry quality, and business signals before expensive AI research begins.",
        },
        {
            "number": 3,
            "name": "Build the finalist pool",
            "metric": f"{funnel_count or scout_cards} detailed candidate cards",
            "status": "good" if (funnel_count or scout_cards) else "attention",
            "plain": "The strongest candidates receive richer fundamentals, earnings, valuation, and liquidity data.",
        },
        {
            "number": 4,
            "name": "Check whether Robinhood can trade them now",
            "metric": f"{broker_clean or bucket_counts.get('execution_ready', 0)} buy-now capable",
            "status": "good" if broker_clean or bucket_counts.get("execution_ready", 0) else "attention",
            "plain": "This checks quote quality, spread, liquidity, tradability, and fractional-share support. It does not judge whether the company is good.",
        },
        {
            "number": 5,
            "name": "Opportunity Scout ranks the best usable ideas",
            "metric": f"First Council batch: {', '.join('$' + str(t) for t in first_batch) if first_batch else 'none'}",
            "status": "good" if scout.get("agentic_used") else "attention",
            "plain": "The agent compares the candidate cards in context, can research gaps, and sends the most promising practical names first.",
        },
        {
            "number": 6,
            "name": "Council debates each stock",
            "metric": f"{council_count} reviewed · {buy_count} buy-side verdict{'s' if buy_count != 1 else ''}",
            "status": "good" if council_count else "neutral",
            "plain": "Fundamental, technical, and risk analysts disagree independently; the synthesis officer turns their evidence into one verdict.",
        },
        {
            "number": 7,
            "name": "Execution Officer and Robinhood make the final live check",
            "metric": "Automatic when every gate passes",
            "status": "good",
            "plain": "A good investment idea can still wait if the live price or spread is unsafe. No user approval is required for an eligible auto-trade.",
        },
        {
            "number": 8,
            "name": "Monitor, sell, and learn",
            "metric": _supervisor_check("position_monitoring").get("message") or "Monitoring state unavailable",
            "status": "good" if str(_supervisor_check("position_monitoring").get("status") or "").upper() == "PASS" else "attention",
            "plain": "Stops and news trigger review; the sell council decides judgment exits; broker fills are reconciled and later measured.",
        },
    ]
    return {
        "session_id": session_id,
        "created_at": scout.get("created_at"),
        "summary": str(scout.get("summary") or "No recent Opportunity Scout run found."),
        "agentic_used": bool(scout.get("agentic_used")),
        "model": scout.get("model_used"),
        "reasoning_effort": scout.get("reasoning_effort"),
        "deployable_at_scan": _safe_float(scout.get("deployable_amount")),
        "research_only": bool(scout.get("research_only")),
        "selected": selected,
        "batches": batches,
        "routing": {
            "total": len(routing),
            "bucket_counts": dict(bucket_counts),
            "top_reasons": [{"reason": reason, "count": count} for reason, count in reason_counts.most_common(8)],
        },
        "council": {"count": council_count, "buy_side_count": buy_count, "action_mix": action_mix},
        "coverage": {
            "status": str(rank.get("status") or "UNKNOWN"),
            "coverage_pct": round((_safe_float(rank.get("coverage_pct"), 0.0) or 0.0) * 100.0, 2),
            "universe": universe,
            "usable": usable,
            "ranked": ranked,
            "lottery_excluded": lottery,
            "missing": missing,
            "generated_at": rank.get("generated_at"),
        },
        "stages": stages,
    }


def build_report_card() -> dict[str, Any]:
    raw = _read_json(DATA / "accuracy.json", [])
    records = [row for row in raw if isinstance(row, dict)] if isinstance(raw, list) else []
    current = [row for row in records if str(row.get("accuracy_era") or "").lower() == "current"]
    graded = [row for row in current if str(row.get("grade") or "").strip()]
    pending = [row for row in current if not str(row.get("grade") or "").strip()]
    counts = Counter(str(row.get("grade") or "").upper() for row in graded)
    total = len(graded)
    weighted = counts["CORRECT"] + 0.5 * counts["PARTIALLY_CORRECT"]
    accuracy = weighted / total * 100.0 if total else None
    return {
        "era": "Current Council era",
        "graded": total,
        "correct": counts["CORRECT"],
        "incorrect": counts["INCORRECT"],
        "partial": counts["PARTIALLY_CORRECT"],
        "pending": len(pending),
        "accuracy_pct": round(accuracy, 1) if accuracy is not None else None,
        "definition": "A research call is graded about 30 days later against SPY. This is different from realized trade win rate.",
    }


def build_learning() -> dict[str, Any]:
    check = _supervisor_check("feedback_loop")
    report = check.get("report") if isinstance(check.get("report"), dict) else {}
    buy = report.get("buy_outcome_feedback") if isinstance(report.get("buy_outcome_feedback"), dict) else {}
    sell = report.get("sell_outcome_feedback") if isinstance(report.get("sell_outcome_feedback"), dict) else {}
    calibration = report.get("calibration_feedback") if isinstance(report.get("calibration_feedback"), dict) else {}
    shadow = report.get("shadow_feedback") if isinstance(report.get("shadow_feedback"), dict) else {}
    sentinel = report.get("sentinel_feedback") if isinstance(report.get("sentinel_feedback"), dict) else {}
    lifecycle = report.get("lesson_lifecycle") if isinstance(report.get("lesson_lifecycle"), dict) else {}
    post_sell = sell.get("post_sell") if isinstance(sell.get("post_sell"), dict) else {}
    stages = [
        {
            "name": "Buy decisions are graded",
            "status": "active" if _safe_int(buy.get("benchmark_graded")) else "collecting",
            "metric": f"{_safe_int(buy.get('benchmark_graded'))} graded · {_safe_int(buy.get('eligible_misses'))} eligible misses",
            "plain": "ARTHA compares old buy/watch/avoid calls with what happened versus the market. Only current-era, benchmark-based examples can reach the live prompt.",
        },
        {
            "name": "Sell decisions are tracked after exit",
            "status": "active" if bool(sell.get("ready")) else "collecting",
            "metric": f"{_safe_int(sell.get('completed'))}/{_safe_int(sell.get('minimum_completed'), 20)} mature 60-day reviews",
            "plain": "The plumbing is connected, but sell outcomes are still too young to influence council policy. Until enough exits mature, this remains observation only.",
        },
        {
            "name": "Possible rule changes run on paper first",
            "status": "active" if _safe_int(shadow.get("completed")) else "collecting",
            "metric": f"{_safe_int(shadow.get('completed'))} complete · {_safe_int(shadow.get('tracking'))} still tracking",
            "plain": "Candidate rules are compared with real decisions without touching money. Promotion is manual and gated; no rule edits itself.",
        },
        {
            "name": "Calibration checks whether scores mean what they claim",
            "status": "waiting" if str(calibration.get("meta_signal_recommendation")) == "do_not_adjust" else "active",
            "metric": f"{_safe_int(calibration.get('completed_decision_shadows'))}/{_safe_int(calibration.get('minimum_samples'), 20)} forward samples",
            "plain": "The sample is currently too small for automatic score adjustment. ARTHA injects the diagnosis but is explicitly told not to tune yet.",
        },
        {
            "name": "Sentinel turns verified news into a sell review",
            "status": "active" if bool(sentinel.get("enabled")) else "off",
            "metric": f"Watching {_safe_int(sentinel.get('held_positions'))} holdings",
            "plain": "A raw keyword alert cannot trade. Verified negative news can escalate to the sell council, which must still make the decision.",
        },
    ]
    return {
        "status": str(check.get("status") or "UNKNOWN"),
        "guardrail": str(report.get("guardrail") or "Feedback remains advisory until explicit sample gates pass."),
        "buy": buy,
        "sell": {
            "completed": _safe_int(sell.get("completed")),
            "minimum": _safe_int(sell.get("minimum_completed"), 20),
            "ready": bool(sell.get("ready")),
            "next_due": post_sell.get("next_60d_due_date"),
            "tracking": _safe_int((post_sell.get("status_counts") or {}).get("tracking")),
            "grade_counts": post_sell.get("grade_counts") if isinstance(post_sell.get("grade_counts"), dict) else {},
        },
        "calibration": calibration,
        "shadow": shadow,
        "sentinel": sentinel,
        "lessons": {
            "total": _safe_int(lifecycle.get("total")),
            "observational": _safe_int(lifecycle.get("observational_items_reported")),
            "manual_review": _safe_int((lifecycle.get("effect_counts") or {}).get("manual_review")),
            "types": lifecycle.get("type_counts") if isinstance(lifecycle.get("type_counts"), dict) else {},
        },
        "stages": stages,
        "report_card": build_report_card(),
    }


HEALTH_NAMES = {
    "maintenance_operations": "Routine cleanup and housekeeping",
    "database": "Decision journal database",
    "decision_artifacts": "Latest decision evidence file",
    "rank_coverage": "Market coverage",
    "latest_report": "Latest report delivery",
    "agentic_trace": "Research-agent evidence trace",
    "intelligence_routing": "Independent analyst routing",
    "recent_sessions": "Recent scheduled scan",
    "defer_watchlist": "Deferred-price watchlist",
    "position_monitoring": "Every holding has a sell thesis",
    "exit_control_state": "Closed positions are no longer monitored",
    "broker_fill_accounting": "Broker fills reached accounting and learning",
    "position_classification": "Sector and position labels",
    "execution_learning": "Trade-outcome accounting",
    "feedback_loop": "Guarded learning loop",
    "broker_reconciliation": "Robinhood reconciliation",
    "calibration_diagnosis": "Score calibration sample",
    "shadow_rules": "Paper trials for new rules",
    "execution_readiness": "Live trading configuration",
    "recent_logs": "Recent error log review",
    "telegram": "Telegram reporting",
}


HEALTH_GROUPS = {
    "maintenance_operations": "Core",
    "database": "Core",
    "decision_artifacts": "Research",
    "rank_coverage": "Research",
    "latest_report": "Communication",
    "agentic_trace": "Research",
    "intelligence_routing": "Research",
    "recent_sessions": "Scheduling",
    "defer_watchlist": "Buy monitoring",
    "position_monitoring": "Sell protection",
    "exit_control_state": "Sell protection",
    "broker_fill_accounting": "Accounting",
    "position_classification": "Risk controls",
    "execution_learning": "Learning",
    "feedback_loop": "Learning",
    "broker_reconciliation": "Broker",
    "calibration_diagnosis": "Learning",
    "shadow_rules": "Learning",
    "execution_readiness": "Broker",
    "recent_logs": "Core",
    "telegram": "Communication",
}


def _openclaw_cron_state() -> dict[str, Any]:
    def produce() -> dict[str, Any]:
        raw = _shell([OPENCLAW_BIN, "cron", "list", "--all", "--json"], timeout=10)
        try:
            jobs = json.loads(raw).get("jobs", [])
        except (ValueError, AttributeError):
            return {}
        result = {}
        for job in jobs:
            if not isinstance(job, dict):
                continue
            name = str(job.get("name") or "")
            if "Artha" not in name:
                continue
            state = job.get("state") if isinstance(job.get("state"), dict) else {}
            result[name] = {
                "enabled": bool(job.get("enabled")),
                "last_status": state.get("lastStatus"),
                "last_run_ms": state.get("lastRunAtMs"),
            }
        return result

    return _cache.get("openclaw_crons", 300, produce)


def build_system(policy: dict[str, Any]) -> dict[str, Any]:
    supervisor = _supervisor()
    checks = []
    for raw in supervisor.get("checks", []):
        name = str(raw.get("name") or "unknown")
        status = str(raw.get("status") or "UNKNOWN").upper()
        checks.append(
            {
                "id": name,
                "name": HEALTH_NAMES.get(name, name.replace("_", " ").title()),
                "group": HEALTH_GROUPS.get(name, "Other"),
                "status": status,
                "detail": _clip(str(raw.get("message") or "No explanation recorded."), 360),
            }
        )

    monitor_up = bool(_cache.get("monitor_process", 20, lambda: _shell(["pgrep", "-f", "run.py monitor"]).strip()))
    crons = _openclaw_cron_state()
    runner_name = next((name for name in crons if "Auto-Trade Runner" in name or "Auto-Buy Runner" in name), "")
    runner = crons.get(runner_name, {}) if runner_name else {}
    runner_age = None
    if runner.get("last_run_ms"):
        runner_age = max(0.0, (time.time() - float(runner["last_run_ms"]) / 1000.0) / 60.0)
    services = [
        {
            "name": "ARTHA monitor",
            "status": "PASS" if monitor_up else "FAIL",
            "detail": "Running scans, watch checks, Sentinel, and sell monitoring." if monitor_up else "Not running; automated monitoring is interrupted.",
        },
        {
            "name": "Dashboard",
            "status": "PASS",
            "detail": "This read-only dashboard service is responding.",
        },
        {
            "name": "Automatic order runner",
            "status": "PASS" if runner.get("enabled") else "WARN",
            "detail": (
                f"Enabled; last cycle {runner_age:.0f} minutes ago." if runner.get("enabled") and runner_age is not None
                else "Enabled." if runner.get("enabled")
                else "Runner state could not be confirmed from OpenClaw."
            ),
        },
    ]
    all_statuses = [item["status"] for item in checks + services]
    summary_status = "FAIL" if "FAIL" in all_statuses else "WARN" if "WARN" in all_statuses else "PASS"
    return {
        "status": summary_status,
        "supervisor_status": supervisor.get("severity"),
        "checked_at": supervisor.get("generated_at"),
        "checks": checks,
        "services": services,
        "counts": dict(Counter(all_statuses)),
        "policy": policy,
    }


def build_alarms(vitals: dict[str, Any], system: dict[str, Any], sectors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alarms: list[dict[str, Any]] = []
    if vitals.get("trading_disabled"):
        alarms.append(
            {
                "severity": "critical",
                "title": "Emergency trading switch is on",
                "plain": vitals.get("kill_reason") or "No buy or sell order can be placed until the switch is cleared.",
                "source": "Trading control",
            }
        )
    for item in system.get("checks", []) + system.get("services", []):
        if item.get("status") not in {"WARN", "FAIL"}:
            continue
        alarms.append(
            {
                "severity": "critical" if item.get("status") == "FAIL" else "warning",
                "title": item.get("name"),
                "plain": item.get("detail"),
                "source": "Supervisor",
            }
        )
    if market_phase()["open"] and (not vitals.get("broker_fresh") or vitals.get("broker_status") != "PASS"):
        alarms.append(
            {
                "severity": "critical",
                "title": "Robinhood snapshot is not fresh",
                "plain": "ARTHA will block broker-dependent orders until a fresh account snapshot is proven.",
                "source": "Broker sync",
            }
        )
    cutoff = (_now_utc() - timedelta(hours=24)).isoformat()
    blocked = _db_rows(
        "SELECT ticker, side, status, message, updated_at, payload_json, result_json "
        "FROM trade_actions WHERE updated_at >= ? AND status IN ('blocked','review_blocked','failed') "
        "ORDER BY updated_at DESC LIMIT 8",
        (cutoff,),
    )
    for row in blocked:
        alarms.append(
            {
                "severity": "warning" if row.get("status") != "failed" else "critical",
                "title": f"{str(row.get('side') or 'Trade').title()} for {row.get('ticker')} was blocked",
                "plain": _extract_block_reason(row),
                "source": f"Execution · {_ct_stamp(row.get('updated_at'))}",
            }
        )
    signals = _db_rows(
        "SELECT ticker, severity, signal_type, message, created_at FROM sell_signals "
        "WHERE created_at >= ? AND actioned=0 AND suppressed=0 ORDER BY created_at DESC LIMIT 8",
        (cutoff,),
    )
    seen: set[str] = set()
    for row in signals:
        ticker = str(row.get("ticker") or "")
        if ticker in seen:
            continue
        seen.add(ticker)
        alarms.append(
            {
                "severity": "critical" if str(row.get("severity") or "").upper() == "CRITICAL" else "warning",
                "title": f"Unresolved sell signal for {ticker}",
                "plain": _clip(str(row.get("message") or row.get("signal_type") or ""), 300),
                "source": f"Sell monitoring · {_ct_stamp(row.get('created_at'))}",
            }
        )
    if sectors:
        largest = sectors[0]
        cap = _safe_float(system.get("policy", {}).get("max_sector_pct"), 30.0) or 30.0
        pct = _safe_float(largest.get("pct_nav"), 0.0) or 0.0
        if pct >= cap - 2.0:
            alarms.append(
                {
                    "severity": "notice",
                    "title": f"{largest.get('sector')} is near the sector ceiling",
                    "plain": f"It is {pct:.1f}% of the account versus a {cap:.0f}% limit. New buys that would cross the cap are blocked; sells remain allowed.",
                    "source": "Concentration control",
                }
            )
    return alarms


def build_improvements() -> dict[str, Any]:
    def commits() -> list[dict[str, str]]:
        raw = _shell(
            ["git", "-C", str(ROOT), "log", "--pretty=%ad|%s", "--date=format:%b %d", "-n", "45"],
            timeout=8,
        )
        result = []
        for line in raw.splitlines():
            if "|" not in line:
                continue
            when, subject = line.split("|", 1)
            if subject.startswith(("Runtime data", "Merge")):
                continue
            result.append({"date": when, "subject": subject})
            if len(result) >= 10:
                break
        return result

    recent = _cache.get("git_log", 300, commits)
    briefs = []
    reports = DATA / "reports"
    if reports.exists():
        for path in sorted(reports.glob("strategy_research_*.md"), reverse=True)[:4]:
            try:
                title = path.read_text(encoding="utf-8").strip().splitlines()[0].lstrip("# ")
            except (OSError, IndexError):
                title = "Strategy research brief"
            briefs.append({"date": path.stem.replace("strategy_research_", ""), "title": _clip(title, 120)})
    return {"commits": recent, "research_briefs": briefs}


def build_schedule(policy: dict[str, Any]) -> list[dict[str, Any]]:
    afternoon = policy.get("afternoon_scan") if policy.get("afternoon_scan_enabled") else "Disabled"
    return [
        {"time": "9:00 AM CT", "name": "Warm-up scan", "plain": "Refreshes broad market rankings and prepares data before the main council run."},
        {"time": str(policy.get("primary_scan")), "name": "Main buy scan", "plain": "Routes the best executable candidates through Scout, Council, and automatic execution."},
        {"time": str(afternoon), "name": "Afternoon opportunity scan", "plain": "Checks a smaller catalyst and market-mover lane with the same downstream safety gates."},
        {"time": "Throughout market hours", "name": "Position and news monitoring", "plain": "Checks stops, thesis conditions, broker state, and verified news for every holding."},
        {"time": "Nightly", "name": "Outcomes and supervisor review", "plain": "Grades mature decisions, updates paper trials, reconciles records, and reports broken plumbing."},
    ]


def _section(name: str, builder: Callable[[], Any], default: Any) -> Any:
    try:
        return builder()
    except Exception as exc:
        logger.exception("dashboard section %s failed: %s", name, exc)
        return default


def build_payload() -> dict[str, Any]:
    policy = _section("policy", build_policy, {})
    vitals = _section("vitals", lambda: build_vitals(policy), {"portfolio_missing": True})
    _record_intraday_sample(_safe_float(vitals.get("total_value"), 0.0) or 0.0)
    positions = _section("positions", lambda: build_positions(vitals), [])
    sectors = _section("sectors", lambda: build_sectors(positions, vitals), [])
    system = _section("system", lambda: build_system(policy), {"status": "UNKNOWN", "checks": [], "services": [], "policy": policy})
    alarms = _section("alarms", lambda: build_alarms(vitals, system, sectors), [])
    return {
        "schema_version": 2,
        "generated_at": _now_utc().isoformat(),
        "market": market_phase(),
        "vitals": vitals,
        "performance": _section("performance", lambda: build_performance(vitals, positions), {}),
        "equity": _section("equity", lambda: build_equity_curve(vitals), {"points": [], "markers": [], "base": _contribution_base()}),
        "positions": positions,
        "sectors": sectors,
        "activity": _section("activity", build_activity_today, []),
        "decisions": _section("decisions", build_decision_journeys, []),
        "sell_reviews": _section("sell_reviews", build_sell_reviews, []),
        "pipeline": _section("pipeline", build_pipeline, {"stages": []}),
        "learning": _section("learning", build_learning, {"stages": [], "report_card": {}}),
        "system": system,
        "alarms": alarms,
        "schedule": build_schedule(policy),
        "improvements": _section("improvements", build_improvements, {}),
        "privacy": {
            "read_only": True,
            "account_identifiers_exposed": False,
            "explanation": "The dashboard reads ARTHA's local records. It cannot review, place, cancel, or alter a trade.",
        },
    }


# ---------------------------------------------------------------------------
# Authentication and HTTP


def load_token() -> str:
    DASH_DATA.mkdir(parents=True, exist_ok=True)
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if token:
            return token
    token = secrets.token_urlsafe(24)
    TOKEN_FILE.write_text(token + "\n", encoding="utf-8")
    os.chmod(TOKEN_FILE, 0o600)
    return token


TOKEN = load_token()


class Handler(BaseHTTPRequestHandler):
    server_version = "ArthaDash/2.0"

    def _query_token_valid(self) -> bool:
        query = parse_qs(urlparse(self.path).query)
        supplied = query.get("k", [""])[0]
        return bool(supplied) and hmac.compare_digest(str(supplied), TOKEN)

    def _authed(self) -> bool:
        if self._query_token_valid():
            return True
        cookies = self.headers.get("Cookie", "")
        for part in cookies.split(";"):
            name, _, value = part.strip().partition("=")
            if name == "artha_dash" and hmac.compare_digest(value, TOKEN):
                return True
        return False

    def _headers(self, content_type: str, length: int = 0, cookie: bool = False) -> None:
        self.send_header("Content-Type", content_type)
        if length:
            self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'",
        )
        if cookie:
            self.send_header("Set-Cookie", f"artha_dash={TOKEN}; Path=/; Max-Age=31536000; HttpOnly; SameSite=Strict")

    def _send(self, code: int, body: bytes, content_type: str, cookie: bool = False) -> None:
        self.send_response(code)
        self._headers(content_type, len(body), cookie)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if self._query_token_valid() and parsed.path in {"/", "/index.html"}:
            self.send_response(302)
            self.send_header("Location", "/")
            self._headers("text/plain", cookie=True)
            self.end_headers()
            return
        if not self._authed():
            if parsed.path.startswith("/api"):
                self._send(401, b'{"error":"unauthorized"}', "application/json")
            else:
                body = (
                    "<main style='font:16px system-ui;max-width:520px;margin:12vh auto;padding:24px'>"
                    "<h2>ARTHA dashboard access</h2><p>Open your saved dashboard link once. "
                    "It contains the private access key and will keep this device signed in.</p></main>"
                ).encode()
                self._send(401, body, "text/html; charset=utf-8")
            return

        static = {
            "/": (INDEX_FILE, "text/html; charset=utf-8"),
            "/index.html": (INDEX_FILE, "text/html; charset=utf-8"),
            "/styles.css": (CSS_FILE, "text/css; charset=utf-8"),
            "/app.js": (APP_FILE, "text/javascript; charset=utf-8"),
        }
        if parsed.path in static:
            path, content_type = static[parsed.path]
            try:
                self._send(200, path.read_bytes(), content_type, cookie=True)
            except OSError:
                self._send(404, b"missing dashboard asset", "text/plain")
            return
        if parsed.path == "/api/dashboard":
            payload = _cache.get("dashboard_payload", 5, build_payload)
            body = json.dumps(payload, separators=(",", ":"), default=str).encode()
            self._send(200, body, "application/json", cookie=True)
            return
        if parsed.path == "/api/health":
            body = json.dumps({"status": "PASS", "service": "artha-dashboard", "schema_version": 2}).encode()
            self._send(200, body, "application/json", cookie=True)
            return
        if parsed.path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
            return
        self._send(404, b"not found", "text/plain")

    def log_message(self, fmt: str, *args: Any) -> None:
        return


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc_type, exc, _ = sys.exc_info()
        if exc_type in (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            logger.info("client disconnected during response: %s", exc)
            return
        super().handle_error(request, client_address)


def main() -> None:
    server = QuietThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    logger.info("ARTHA dashboard v2 listening on 0.0.0.0:%s", PORT)
    server.serve_forever()


if __name__ == "__main__":
    main()
