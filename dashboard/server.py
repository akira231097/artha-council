#!/usr/bin/env python3
"""ARTHA Dashboard — zero-dependency local web server.

Serves a single-page real-time dashboard over the LAN/Tailscale so an operator can
watch ARTHA from any device (phone included). Reads ARTHA's live data files
and journal DB strictly read-only; its only writes are its own intraday
equity samples and access token under data/dashboard/.

Design constraints:
- stdlib only (no framework, no supply chain, immune to venv drift);
- every data source is optional — a missing/corrupt file degrades the panel,
  never the page;
- the live SQLite journal is opened with mode=ro and a short busy timeout so
  the dashboard can never block the trading daemon;
- shellouts (openclaw cron state, git log) are cached and time-boxed.
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DASH_DATA = DATA / "dashboard"
DB_PATH = DATA / "artha.db"
TOKEN_FILE = DASH_DATA / "token.txt"
INTRADAY_FILE = DASH_DATA / "equity_intraday.jsonl"
INDEX_FILE = Path(__file__).resolve().parent / "index.html"

CT = ZoneInfo("America/Chicago")
ET = ZoneInfo("America/New_York")

PORT = int(os.environ.get("ARTHA_DASHBOARD_PORT", "8787"))
PILOT_BASE = 350.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s [dashboard] %(levelname)s: %(message)s")
logger = logging.getLogger("artha.dashboard")


# ---------------------------------------------------------------- utilities

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _read_json(path: Path, default):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload
    except Exception:
        return default


def _parse_iso(value) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _ct(dt: datetime | None) -> str:
    if not dt:
        return ""
    return dt.astimezone(CT).strftime("%-I:%M %p")


def _age_minutes(dt: datetime | None) -> float | None:
    if not dt:
        return None
    return (_now_utc() - dt).total_seconds() / 60.0



def _plain(text: str) -> str:
    """Collapse whitespace and strip markdown emphasis for plain display."""
    text = re.sub(r"[*_`#]+", "", str(text))
    return re.sub(r"\s+", " ", text).strip()

def market_phase(now: datetime | None = None) -> dict:
    """Regular-hours approximation (weekends + hours; holidays show as quiet open)."""
    now = now or _now_utc()
    ct = now.astimezone(CT)
    weekday = ct.weekday() < 5
    minutes = ct.hour * 60 + ct.minute
    is_open = weekday and (8 * 60 + 30) <= minutes < (15 * 60)
    if not weekday:
        phase = "Weekend"
    elif minutes < 8 * 60 + 30:
        phase = "Pre-market"
    elif is_open:
        phase = "Market open"
    else:
        phase = "After close"
    return {"open": is_open, "phase": phase, "ct_time": ct.strftime("%a %-I:%M %p CT")}


class _TTLCache:
    """TTL cache with per-key single-flight and stale-on-error fallback."""

    def __init__(self):
        self._store: dict[str, tuple[float, object]] = {}
        self._lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}

    def get(self, key: str, ttl: float, producer):
        with self._lock:
            hit = self._store.get(key)
            if hit and time.time() - hit[0] < ttl:
                return hit[1]
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            with self._lock:
                hit = self._store.get(key)
                if hit and time.time() - hit[0] < ttl:
                    return hit[1]
            try:
                value = producer()
            except Exception as exc:
                logger.warning("cache producer %s failed: %s", key, exc)
                with self._lock:
                    hit = self._store.get(key)
                return hit[1] if hit else None
            with self._lock:
                self._store[key] = (time.time(), value)
            return value


_cache = _TTLCache()


def _shell(cmd: list[str], timeout: float = 8.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout if out.returncode == 0 else ""
    except Exception:
        return ""


def _db_rows(query: str, params: tuple = ()) -> list[dict]:
    try:
        uri = f"file:{DB_PATH}?mode=ro"
        con = sqlite3.connect(uri, uri=True, timeout=2.0)
        con.row_factory = sqlite3.Row
        try:
            rows = [dict(r) for r in con.execute(query, params).fetchall()]
        finally:
            con.close()
        return rows
    except Exception as exc:
        logger.warning("db query failed: %s (%s)", exc, query[:60])
        return []


# ---------------------------------------------------------------- sections

def _contributions() -> tuple[float, dict | None]:
    """Net money the operator has put in (deposits positive, withdrawals negative).

    Read from data/contributions.jsonl — seeded with the initial $350 and
    auto-appended by ARTHA's transfer detection. Keeps deposits from ever
    showing up as fake profit."""
    total = 0.0
    latest: dict | None = None
    try:
        for line in (DATA / "contributions.jsonl").read_text(encoding="utf-8").strip().splitlines():
            try:
                row = json.loads(line)
                total += float(row.get("amount") or 0)
                latest = row
            except Exception:
                continue
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning("contributions read failed: %s", exc)
    return round(total, 2), latest


def build_vitals() -> dict:
    portfolio = _read_json(DATA / "portfolio.json", None)
    portfolio_missing = not isinstance(portfolio, dict) or not portfolio
    portfolio = portfolio if isinstance(portfolio, dict) else {}
    positions = portfolio.get("positions") or []
    invested = sum(float(p.get("market_value") or 0) for p in positions)
    cash = float(portfolio.get("cash_available") or 0)
    total = invested + cash
    base, last_contribution = _contributions()
    if base <= 0:
        base = float(portfolio.get("monthly_contribution") or PILOT_BASE)
    prices_age = _age_minutes(_parse_iso(portfolio.get("last_updated")))

    snaps = _db_rows(
        "SELECT timestamp, total_value FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 400"
    )
    today_ct = _now_utc().astimezone(CT).date()
    prev_close_value = None
    baseline_label = ""
    for s in snaps:
        ts = _parse_iso(s.get("timestamp"))
        if ts and ts.astimezone(CT).date() < today_ct and s.get("total_value"):
            prev_close_value = float(s["total_value"])
            baseline_label = ts.astimezone(CT).strftime("%a %b %-d")
            break
    pnl_today = (total - prev_close_value) if prev_close_value else None

    regime = _read_json(DATA / "regime_gate.json", {})
    regime_state = str(regime.get("state") or regime.get("regime") or "?")
    details = regime.get("details") or {}

    control = _read_json(DATA / "robinhood" / "control.json", {})
    paused, pause_reason = False, ""
    raw_until = str(control.get("buying_paused_until") or "")[:10]
    if raw_until:
        try:
            paused = _now_utc().astimezone(ET).date().isoformat() <= raw_until
            pause_reason = str(control.get("buying_paused_reason") or "")
        except Exception:
            paused = False

    return {
        "total_value": None if portfolio_missing else round(total, 2),
        "cash": round(cash, 2),
        "invested": round(invested, 2),
        "base": base,
        "pnl_total": None if portfolio_missing else round(total - base, 2),
        "pnl_total_pct": None if portfolio_missing else (round((total / base - 1) * 100, 2) if total else 0),
        "pnl_today": round(pnl_today, 2) if pnl_today is not None and not portfolio_missing else None,
        "pnl_today_pct": round(pnl_today / prev_close_value * 100, 2) if pnl_today is not None and prev_close_value and not portfolio_missing else None,
        "pnl_baseline_label": baseline_label,
        "recent_contribution": (
            last_contribution
            if last_contribution
            and str(last_contribution.get("note")) == "auto_detected_transfer"
            and (_age_minutes(_parse_iso(last_contribution.get("t"))) or 1e9) < 48 * 60
            else None
        ),
        "prices_age_minutes": round(prices_age, 1) if prices_age is not None else None,
        "portfolio_missing": portfolio_missing,
        "position_count": len(positions),
        "regime": regime_state,
        "vix": details.get("vix"),
        "spy_close": details.get("spy_close"),
        "spy_sma": details.get("spy_sma"),
        "trading_disabled": bool(control.get("trading_disabled")),
        "kill_reason": str(control.get("reason") or "") if control.get("trading_disabled") else "",
        "buys_paused": paused,
        "buys_paused_reason": pause_reason,
    }


def build_positions() -> list[dict]:
    portfolio = _read_json(DATA / "portfolio.json", {})
    now_iso = _now_utc().isoformat()
    cutoff_iso = (_now_utc() - timedelta(days=3)).isoformat()
    pending: dict[str, dict] = {}
    for r in _db_rows(
        "SELECT ticker, action_type, status FROM trade_actions "
        "WHERE status IN ('review_ready','pending','price_gate_passed') "
        "AND side='sell' AND created_at >= ? "
        "AND (expires_at IS NULL OR expires_at > ?) ORDER BY created_at DESC",
        (cutoff_iso, now_iso),
    ):
        pending.setdefault(str(r.get("ticker") or "").upper(), r)  # newest first
    out = []
    for p in portfolio.get("positions") or []:
        ticker = str(p.get("ticker") or "").upper()
        price = float(p.get("current_price") or 0)
        avg = float(p.get("avg_cost") or 0)
        shares = float(p.get("shares") or 0)
        hard_stop = float(p.get("hard_stop_price") or 0)
        trail = float(p.get("trailing_stop_price") or 0)
        effective_stop = max(hard_stop, trail)
        pnl_pct = (price / avg - 1) * 100 if avg else 0
        stop_dist_pct = (price - effective_stop) / price * 100 if price and effective_stop else None
        pend = pending.get(ticker)
        out.append(
            {
                "ticker": ticker,
                "shares": shares,
                "avg_cost": avg,
                "price": price,
                "market_value": round(float(p.get("market_value") or shares * price), 2),
                "pnl_pct": round(pnl_pct, 2),
                "pnl_usd": round(shares * (price - avg), 2),
                "stop": round(effective_stop, 2) if effective_stop else None,
                "stop_dist_pct": round(stop_dist_pct, 1) if stop_dist_pct is not None else None,
                "stop_locked_profit": bool(effective_stop and avg and effective_stop > avg),
                "entry_date": str(p.get("entry_date") or "")[:10],
                "next_review": str(p.get("next_sell_review") or "")[:10],
                "pending_action": (pend or {}).get("action_type"),
                "position_type": p.get("position_type") or "",
            }
        )
    out.sort(key=lambda r: -r["market_value"])

    # Broker positions ARTHA does not manage (bought manually or by another
    # agent) — surfaced so nothing in the account is invisible.
    try:
        snapshot = _read_json(DATA / "robinhood" / "latest_snapshot.json", {})
        managed = {p["ticker"] for p in out}
        for raw in snapshot.get("positions") or []:
            if not isinstance(raw, dict):
                continue
            sym = str(raw.get("symbol") or (raw.get("instrument") or {}).get("symbol") or "").upper().strip()
            try:
                qty = float(raw.get("quantity") or 0)
            except (TypeError, ValueError):
                qty = 0
            if not sym or sym in managed or qty <= 0:
                continue
            out.append({
                "ticker": sym,
                "shares": qty,
                "unmanaged": True,
                "market_value": 0,
                "pnl_pct": None, "pnl_usd": None, "price": None, "avg_cost": None,
                "stop": None, "stop_dist_pct": None, "stop_locked_profit": False,
                "entry_date": "", "next_review": "", "pending_action": None, "position_type": "",
            })
    except Exception as exc:
        logger.warning("unmanaged position scan failed: %s", exc)
    return out


_ACTION_ICONS = {
    "BUY": "🟢", "TACTICAL_BUY": "🟢", "SELL": "🔴", "TRIM": "🟠",
    "AVOID": "⚪", "DEFER": "🔵", "WATCH": "🔵", "HOLD": "🟡",
}


def _clip(text: str, n: int) -> str:
    text = text.strip()
    if len(text) <= n:
        return text
    cut = text[: n - 1]
    if " " in cut[int(n * 0.6):]:
        cut = cut[: cut.rfind(" ")]
    return cut + "…"


_STATUS_WORDS = {
    "filled": "completed",
    "submitted": "order placed",
    "partially_filled": "partly completed",
    "review_ready": "waiting for operator approval",
    "pending": "queued",
    "price_gate_passed": "price check passed",
    "blocked": "blocked by guardrails",
    "skipped": "skipped",
    "expired": "expired unused",
    "cancelled": "cancelled",
}


def _nice_status(status: str) -> str:
    return _STATUS_WORDS.get(str(status or "").lower(), str(status or "").replace("_", " "))


def build_activity_today() -> list[dict]:
    start_ct = _now_utc().astimezone(CT).replace(hour=0, minute=0, second=0, microsecond=0)
    start_iso = start_ct.astimezone(timezone.utc).isoformat()
    events: list[tuple[datetime, str, str]] = []

    for r in _db_rows(
        "SELECT timestamp, ticker, action, confidence, rationale FROM recommendations "
        "WHERE timestamp >= ? ORDER BY timestamp", (start_iso,)
    ):
        ts = _parse_iso(r["timestamp"])
        act = str(r.get("action") or "?").upper()
        icon = _ACTION_ICONS.get(act, "⚪")
        why = _clip(_plain(str(r.get("rationale") or "")), 130)
        events.append((ts, icon, f"Research verdict on {r['ticker']}: {act.replace('_', ' ')} — {why}"))

    # Staged/blocked actions only — completed ones appear as orders below,
    # so they are not double-reported.
    for r in _db_rows(
        "SELECT created_at, ticker, side, action_type, status FROM trade_actions "
        "WHERE created_at >= ? AND status IN "
        "('review_ready','pending','price_gate_passed','blocked') ORDER BY created_at",
        (start_iso,),
    ):
        ts = _parse_iso(r["created_at"])
        kind = str(r.get("action_type") or "").replace("_", " ")
        status = _nice_status(r.get("status"))
        icon = "🟠" if r.get("side") == "sell" else ("🟢" if r.get("action_type") == "auto_buy" else "📋")
        events.append((ts, icon, f"{kind.capitalize()} for {r['ticker']} staged — {status}"))

    for r in _db_rows(
        "SELECT created_at, ticker, side, notional, status FROM execution_orders "
        "WHERE created_at >= ? ORDER BY created_at", (start_iso,)
    ):
        ts = _parse_iso(r["created_at"])
        side = "Buy" if str(r.get("side") or "").lower() == "buy" else "Sell"
        status = _nice_status(r.get("status"))
        notional = r.get("notional")
        amount = f" ${float(notional):.2f}" if notional else ""
        icon = "💰" if str(r.get("status")) in ("filled", "submitted", "partially_filled") else "📄"
        events.append((ts, icon, f"{side} order for {r['ticker']}{amount} — {status}"))

    for r in _db_rows(
        "SELECT timestamp, session_type, tickers_analyzed FROM sessions "
        "WHERE timestamp >= ? ORDER BY timestamp", (start_iso,)
    ):
        ts = _parse_iso(r["timestamp"])
        tickers = str(r.get("tickers_analyzed") or "")
        n = len([t for t in re.split(r"[,\s]+", tickers) if t])
        kind = str(r.get("session_type") or "session").replace("_", " ")
        events.append((ts, "🔎", f"Ran {kind} — analyzed {n} ticker(s)"))

    events = [e for e in events if e[0]]
    events.sort(key=lambda e: e[0], reverse=True)
    return [{"time": _ct(ts), "icon": icon, "text": text} for ts, icon, text in events[:40]]


def build_decisions(limit: int = 30) -> list[dict]:
    rows = _db_rows(
        "SELECT timestamp, ticker, action, confidence, rationale, price_at_recommendation "
        "FROM recommendations ORDER BY timestamp DESC LIMIT ?", (limit,)
    )
    out = []
    for r in rows:
        ts = _parse_iso(r.get("timestamp"))
        why = _clip(_plain(str(r.get("rationale") or "")), 400)
        out.append(
            {
                "when": ts.astimezone(CT).strftime("%b %-d, %-I:%M %p") if ts else "",
                "ticker": r.get("ticker"),
                "action": str(r.get("action") or "").upper(),
                "confidence": r.get("confidence"),
                "why": why,
                "price": r.get("price_at_recommendation"),
            }
        )
    return out


def build_report_card() -> dict:
    records = _read_json(DATA / "accuracy.json", [])
    if not isinstance(records, list):
        records = []
    records = [r for r in records if isinstance(r, dict)]

    def _gate_blocked(r: dict) -> bool:
        blob = f"{r.get('recommended_action','')} {r.get('notes','')} {r.get('consensus','')}"
        return "GATE BLOCKED" in blob.upper() or str(r.get("consensus")).strip() == "N/A"

    records = [r for r in records if not _gate_blocked(r)]
    graded = [r for r in records if r.get("grade")]
    pending = [r for r in records if not r.get("grade")]
    buy_verdicts = {"BUY", "TACTICAL_BUY", "STARTER_BUY", "TRIM", "SELL", "HOLD"}
    graded_position_calls = len(
        [r for r in graded if str(r.get("verdict") or "").upper() in buy_verdicts]
    )
    counts = {"CORRECT": 0, "INCORRECT": 0, "PARTIALLY_CORRECT": 0}
    for r in graded:
        g = str(r.get("grade") or "").upper()
        if g in counts:
            counts[g] += 1
    total_graded = sum(counts.values())
    accuracy = (counts["CORRECT"] + 0.5 * counts["PARTIALLY_CORRECT"]) / total_graded * 100 if total_graded else None

    by_month: dict[str, dict] = {}
    for r in graded:
        month = str(r.get("review_after") or r.get("timestamp") or "")[:7]
        if not month:
            continue
        bucket = by_month.setdefault(month, {"month": month, "correct": 0, "incorrect": 0, "partial": 0})
        g = str(r.get("grade") or "").upper()
        if g == "CORRECT":
            bucket["correct"] += 1
        elif g == "INCORRECT":
            bucket["incorrect"] += 1
        elif g == "PARTIALLY_CORRECT":
            bucket["partial"] += 1

    today = _now_utc().astimezone(ET).date().isoformat()
    due_tonight = len([r for r in pending if str(r.get("review_after") or "9999")[:10] <= today])

    recent_misses = []
    for r in reversed(graded):
        if str(r.get("grade")).upper() == "INCORRECT" and len(recent_misses) < 3:
            recent_misses.append(
                {
                    "ticker": r.get("ticker"),
                    "verdict": r.get("verdict"),
                    "note": _plain(str(r.get("notes") or ""))[:160],
                }
            )

    lessons_payload = _read_json(DATA / "learnings" / "lessons.json", {})
    lessons_raw = lessons_payload.get("lessons", lessons_payload) if isinstance(lessons_payload, dict) else lessons_payload
    lessons = []
    if isinstance(lessons_raw, list):
        for l in lessons_raw[-6:]:
            lessons.append(
                {
                    "text": str(l.get("description") or "")[:140],
                    "action": str(l.get("action") or "")[:140],
                    "priority": l.get("priority"),
                    "status": l.get("status"),
                }
            )

    return {
        "graded_position_calls": graded_position_calls,
        "graded": total_graded,
        "correct": counts["CORRECT"],
        "incorrect": counts["INCORRECT"],
        "partial": counts["PARTIALLY_CORRECT"],
        "accuracy_pct": round(accuracy, 1) if accuracy is not None else None,
        "pending": len(pending),
        "due_tonight": due_tonight,
        "by_month": sorted(by_month.values(), key=lambda b: b["month"])[-6:],
        "recent_misses": recent_misses,
        "lessons": list(reversed(lessons)),
    }


def _record_intraday_sample(total: float) -> None:
    """Sample equity intraday (max every 10 min, market hours only)."""
    try:
        if not market_phase()["open"] or total <= 0:
            return
        DASH_DATA.mkdir(parents=True, exist_ok=True)
        last_ts = None
        if INTRADAY_FILE.exists():
            lines = INTRADAY_FILE.read_text(encoding="utf-8").strip().splitlines()
            if lines:
                try:
                    last_ts = _parse_iso(json.loads(lines[-1]).get("t"))
                except Exception:
                    last_ts = None  # corrupt tail — sample anyway
        if last_ts and (_now_utc() - last_ts) < timedelta(minutes=10):
            return
        with open(INTRADAY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": _now_utc().isoformat(), "v": round(total, 2)}) + "\n")
        # prune to ~30 days
        lines = INTRADAY_FILE.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) > 3000:
            INTRADAY_FILE.write_text("\n".join(lines[-2500:]) + "\n", encoding="utf-8")
    except Exception as exc:
        logger.warning("intraday sample failed: %s", exc)


# Snapshots before this date double-counted the first buy (cash was not
# debited until the accounting fix), inflating total_value by ~$17.50.
# Known-wrong history is excluded rather than shown as a fake loss.
EQUITY_HISTORY_START = "2026-06-16T18:00:00+00:00"


def build_equity_curve() -> dict:
    points: list[dict] = []
    for r in _db_rows(
        "SELECT timestamp, total_value FROM portfolio_snapshots WHERE timestamp >= ? ORDER BY timestamp",
        (EQUITY_HISTORY_START,),
    ):
        ts = _parse_iso(r.get("timestamp"))
        if ts and r.get("total_value"):
            points.append({"t": ts.isoformat(), "v": round(float(r["total_value"]), 2)})
    if INTRADAY_FILE.exists():
        try:
            for line in INTRADAY_FILE.read_text(encoding="utf-8").strip().splitlines():
                row = json.loads(line)
                if row.get("t") and row.get("v"):
                    points.append({"t": row["t"], "v": float(row["v"])})
        except Exception:
            pass
    points.sort(key=lambda p: p["t"])
    # de-noise: cap at ~240 points, keep ends
    if len(points) > 240:
        step = len(points) / 240.0
        keep = [points[int(i * step)] for i in range(240)]
        keep[-1] = points[-1]
        points = keep

    markers = []
    for r in _db_rows(
        "SELECT COALESCE(filled_at, created_at) AS created_at, ticker, side, notional, status "
        "FROM execution_orders "
        "WHERE status IN ('filled','submitted','partially_filled') ORDER BY 1"
    ):
        ts = _parse_iso(r.get("created_at"))
        if ts:
            markers.append(
                {
                    "t": ts.isoformat(),
                    "side": str(r.get("side") or "").lower(),
                    "ticker": r.get("ticker"),
                    "notional": r.get("notional"),
                }
            )
    return {"points": points, "markers": markers, "base": PILOT_BASE}


def _openclaw_cron_state() -> dict:
    def produce():
        raw = _shell(["openclaw", "cron", "list", "--all", "--json"], timeout=10)
        try:
            jobs = json.loads(raw).get("jobs", [])
        except Exception:
            return {}
        out = {}
        for j in jobs:
            name = str(j.get("name") or "")
            if "Artha" in name:
                st = j.get("state") or {}
                out[name] = {
                    "enabled": bool(j.get("enabled")),
                    "last_status": st.get("lastStatus"),
                    "last_run_ms": st.get("lastRunAtMs"),
                }
        return out
    return _cache.get("openclaw_cron", 300, produce)


def build_system() -> list[dict]:
    rows = []

    def add(name, status, detail):
        rows.append({"name": name, "status": status, "detail": detail})

    monitor_up = bool(_cache.get("pgrep_monitor", 30, lambda: _shell(["pgrep", "-f", "run.py monitor"]).strip()))
    add("ARTHA brain (monitor daemon)", "ok" if monitor_up else "down",
        "Running — scans, councils, stops and reviews are live" if monitor_up
        else "NOT RUNNING — no scans or stop-loss checks until it restarts (watchdog restarts it within 15 min)")

    wd_log = DATA / "logs" / "watchdog.log"
    wd_age = _age_minutes(datetime.fromtimestamp(wd_log.stat().st_mtime, tz=timezone.utc)) if wd_log.exists() else None
    add("Watchdog (self-healing)", "ok" if wd_age is not None and wd_age < 25 else "warn",
        f"Last heartbeat {wd_age:.0f} min ago" if wd_age is not None else "No heartbeat file yet")

    sync = _read_json(DATA / "robinhood" / "snapshot_sync_state.json", {})
    sync_age = _age_minutes(_parse_iso(sync.get("last_success_at")))
    open_now = market_phase()["open"]
    sync_status = str(sync.get("last_status") or "")
    closed_ok = not open_now and sync_age is not None and sync_age < 20 * 60
    if sync_age is not None and ((open_now and sync_status == "PASS" and sync_age < 15) or closed_ok):
        add("Robinhood truth-check (sync)", "ok",
            f"Portfolio verified against the broker {sync_age:.0f} min ago"
            + ("" if open_now else " — pauses while the market is closed, resumes at open"))
    else:
        add("Robinhood truth-check (sync)", "down" if open_now else "warn",
            f"Last success {sync_age/60:.1f} h ago; status={sync_status}" if sync_age is not None else "No sync recorded yet")

    crons = _openclaw_cron_state()
    runner = crons.get("Artha Robinhood Auto-Buy Runner", {})
    if runner:
        last_ms = runner.get("last_run_ms")
        run_age = (time.time() - last_ms / 1000) / 60 if last_ms else None
        healthy = runner.get("enabled") and (not open_now or (run_age is not None and run_age < 10))
        add("Order runner (places buys & sells)", "ok" if healthy else ("warn" if runner.get("enabled") else "down"),
            (f"Enabled — last tick {run_age:.0f} min ago" if run_age is not None else "Enabled")
            if runner.get("enabled") else "DISABLED — queued orders will not execute")
    else:
        add("Order runner (places buys & sells)", "warn", "Could not read cron state")

    regime = _read_json(DATA / "regime_gate.json", {})
    regime_age = _age_minutes(_parse_iso(regime.get("generated_at") or regime.get("updated_at")))
    add("Market data feeds", "ok" if regime_age is not None and regime_age < 60 * 30 else "warn",
        f"Market regime & prices refreshed {regime_age/60:.1f} h ago" if regime_age is not None else "Regime file missing")

    control = _read_json(DATA / "robinhood" / "control.json", {})
    if control.get("trading_disabled"):
        add("Trading switch", "down", f"KILL SWITCH ON — {control.get('reason','')}")
    else:
        raw_until = str(control.get("buying_paused_until") or "")[:10]
        paused = bool(raw_until) and _now_utc().astimezone(ET).date().isoformat() <= raw_until
        add("Trading switch", "ok" if not paused else "warn",
            "Fully enabled within configured order, daily, position, and capacity guardrails" if not paused
            else "Buys paused for today (auto-resumes next trading day); sells & stops fully active")

    strat_cron = crons.get("Artha Weekly Strategy Research", {})
    if strat_cron:
        add("Weekly self-research", "ok" if strat_cron.get("enabled") else "warn",
            "Scheduled Saturdays 10:00 AM — proposes improvements & worthwhile paid upgrades" if strat_cron.get("enabled") else "Disabled")
    return rows


def build_improvements() -> dict:
    def produce_commits():
        raw = _shell(["git", "-C", str(ROOT), "log", "--pretty=%ad|%s", "--date=format:%b %d", "-n", "40"], timeout=8)
        out = []
        for line in raw.strip().splitlines():
            if "|" not in line:
                continue
            date, subject = line.split("|", 1)
            if subject.startswith("Runtime data") or subject.startswith("Merge"):
                continue
            out.append({"date": date, "subject": subject})
            if len(out) >= 10:
                break
        return out
    commits = _cache.get("git_log", 300, produce_commits)

    counts = _db_rows(
        "SELECT COUNT(DISTINCT rule_id) rules, COUNT(*) trials FROM shadow_rule_evaluations"
    )
    rules_n = counts[0]["rules"] if counts else 0
    trials_n = counts[0]["trials"] if counts else 0

    briefs = []
    reports_dir = DATA / "reports"
    if reports_dir.exists():
        for f in sorted(reports_dir.glob("strategy_research_*.md"), reverse=True)[:3]:
            first_line = ""
            try:
                first_line = f.read_text(encoding="utf-8").strip().splitlines()[0].lstrip("# ")
            except Exception:
                pass
            briefs.append({"date": f.stem.replace("strategy_research_", ""), "title": first_line[:120]})

    return {"commits": commits, "shadow": {"rules": rules_n, "trials": trials_n}, "research_briefs": briefs}


def _section(name: str, builder, default):
    try:
        return builder()
    except Exception as exc:
        logger.error("section %s failed: %s", name, exc)
        return default


def build_payload() -> dict:
    vitals = _section("vitals", build_vitals, {"portfolio_missing": True, "total_value": None})
    try:
        _record_intraday_sample(vitals.get("total_value") or 0)
    except Exception:
        pass
    return {
        "generated_at": _now_utc().isoformat(),
        "market": market_phase(),
        "vitals": vitals,
        "positions": _section("positions", build_positions, []),
        "activity": _section("activity", build_activity_today, []),
        "decisions": _section("decisions", build_decisions, []),
        "report_card": _section("report_card", build_report_card, {}),
        "equity": _section("equity", build_equity_curve, {"points": [], "markers": [], "base": PILOT_BASE}),
        "system": _section("system", build_system, []),
        "improvements": _section("improvements", build_improvements, {}),
    }


# ---------------------------------------------------------------- auth + http

def load_token() -> str:
    DASH_DATA.mkdir(parents=True, exist_ok=True)
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if token:
            return token
    token = secrets.token_urlsafe(18)
    TOKEN_FILE.write_text(token + "\n", encoding="utf-8")
    os.chmod(TOKEN_FILE, 0o600)
    return token


TOKEN = load_token()


class Handler(BaseHTTPRequestHandler):
    server_version = "ArthaDash/1.0"

    def _authed(self) -> bool:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if qs.get("k", [""])[0] == TOKEN:
            return True
        cookies = self.headers.get("Cookie", "")
        return f"artha_dash={TOKEN}" in cookies

    def _send(self, code: int, body: bytes, ctype: str, set_cookie: bool = False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if set_cookie:
            self.send_header("Set-Cookie", f"artha_dash={TOKEN}; Path=/; Max-Age=31536000; SameSite=Lax")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if not self._authed():
            self._send(401, b'{"error":"unauthorized"}' if parsed.path.startswith("/api") else
                       "<h3 style='font-family:sans-serif'>ARTHA dashboard: open the operator link from Telegram (it contains the access key).</h3>".encode(),
                       "application/json" if parsed.path.startswith("/api") else "text/html")
            return
        if parsed.path in ("/", "/index.html"):
            try:
                body = INDEX_FILE.read_bytes()
            except Exception:
                body = b"<h3>index.html missing</h3>"
            self._send(200, body, "text/html; charset=utf-8", set_cookie=True)
        elif parsed.path == "/api/dashboard":
            try:
                payload = _cache.get("payload", 5, build_payload)
                body = json.dumps(payload, default=str).encode()
                self._send(200, body, "application/json", set_cookie=True)
            except Exception as exc:
                logger.error("payload failure: %s", exc)
                self._send(500, json.dumps({"error": "internal", "detail": str(exc)[:200]}).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, fmt, *args):  # quiet
        pass


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        exc_type, exc, _ = sys.exc_info()
        if exc_type in (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            logger.info("client disconnected during request from %s: %s", client_address, exc)
            return
        super().handle_error(request, client_address)


def main():
    server = QuietThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    logger.info("ARTHA dashboard on 0.0.0.0:%s (token in %s)", PORT, TOKEN_FILE)
    server.serve_forever()


if __name__ == "__main__":
    main()
