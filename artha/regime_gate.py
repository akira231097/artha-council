"""Mechanical regime gate — replaces fear/greed as a penalty system.

Strategy overhaul Wave 2. The gate is a simple, testable trend-following
overlay on SPY (Faber 2007 timing model; Moskowitz-Ooi-Pedersen 2012
time-series momentum):

  RISK_ON        — SPY at/above its 200-day SMA (or fewer than the required
                   consecutive sessions below). Full buys at configured
                   thresholds.
  RISK_OFF       — SPY closed below its 200-day SMA for 3+ consecutive
                   sessions. New momentum/breakout entries require council
                   conviction >= 8 and run HALF SIZE.
  HARD_RISK_OFF  — RISK_OFF and the trailing 12-month SPY excess return vs
                   the 3-month T-bill (FRED DTB3) is negative. No new buys.

The gate NEVER blocks exits — sell paths must not consult it for permission.

Position sizing uses a volatility multiplier clamp(20/VIX, 0.5, 1.0) so
size shrinks mechanically when the VIX runs hot.

State is cached to data/regime_gate.json (full detail, read by the
scheduler agent) and mirrored to data/regime_state.json with the
{'regime', 'as_of', 'changed_at'} shape the sell engine already reads.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .config import Config
from .paths import DATA_DIR

logger = logging.getLogger(__name__)

_DATA_DIR = DATA_DIR
REGIME_GATE_PATH = _DATA_DIR / "regime_gate.json"
REGIME_STATE_PATH = _DATA_DIR / "regime_state.json"

RISK_ON = "RISK_ON"
RISK_OFF = "RISK_OFF"
HARD_RISK_OFF = "HARD_RISK_OFF"
VALID_STATES = (RISK_ON, RISK_OFF, HARD_RISK_OFF)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _today_str() -> str:
    return _utcnow().strftime("%Y-%m-%d")


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=f".{path.name}_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> Optional[dict]:
    try:
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception as exc:
        logger.warning("[regime_gate] Could not read %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Data inputs (collector-backed, each with a safe fallback)
# ---------------------------------------------------------------------------

def _fetch_spy_closes(min_bars: int) -> list[dict]:
    """Daily SPY bars ascending by date ({date, close}), FMP with yf fallback."""
    try:
        from .collector import get_recent_bars

        bars = get_recent_bars("SPY", max(min_bars, 260))
        return [
            {"date": str(b.get("date")), "close": float(b.get("close"))}
            for b in bars or []
            if b.get("close") is not None
        ]
    except Exception as exc:
        logger.warning("[regime_gate] SPY history fetch failed: %s", exc)
        return []


def _fetch_tbill_rate_pct() -> tuple[float, str]:
    """3-month T-bill rate (percent) from FRED DTB3; fallback to config."""
    try:
        from .collector import FREDCollector

        payload = FREDCollector().series("DTB3", limit=5)
        observations = (payload or {}).get("observations") or []
        for obs in observations:  # sort_order=desc → newest first
            raw = str(obs.get("value", "")).strip()
            if raw in ("", "."):
                continue
            return float(raw), f"fred:DTB3:{obs.get('date')}"
    except Exception as exc:
        logger.warning("[regime_gate] FRED DTB3 fetch failed: %s", exc)
    return float(Config.REGIME_GATE_TBILL_FALLBACK_PCT), "fallback_config"


def _fetch_vix() -> tuple[Optional[float], str]:
    """Current VIX level, yfinance first, FMP '^VIX' quote fallback."""
    try:
        from .collector import YFinanceCollector

        vix = YFinanceCollector().vix()
        if vix and vix.get("value"):
            return float(vix["value"]), f"yfinance:^VIX:{vix.get('date')}"
    except Exception as exc:
        logger.warning("[regime_gate] yfinance VIX fetch failed: %s", exc)
    try:
        from .collector import FMPCollector

        quote = FMPCollector().quote("^VIX")
        price = (quote or {}).get("price")
        if price:
            return float(price), "fmp:^VIX"
    except Exception as exc:
        logger.warning("[regime_gate] FMP ^VIX fetch failed: %s", exc)
    return None, "unavailable"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _consecutive_sessions_below_sma(closes: list[float], sma_days: int) -> int:
    """Count consecutive sessions (ending at the latest bar) with close < SMA."""
    count = 0
    for end in range(len(closes), sma_days - 1, -1):
        window = closes[end - sma_days:end]
        sma = sum(window) / sma_days
        if closes[end - 1] < sma:
            count += 1
        else:
            break
    return count


def compute_regime(force_refresh: bool = False) -> dict:
    """Compute the mechanical regime gate state.

    Returns:
        {
          "state": "RISK_ON" | "RISK_OFF" | "HARD_RISK_OFF",
          "vol_multiplier": float,   # clamp(20/VIX, 0.5, 1.0)
          "details": {...},          # inputs + intermediate values
        }

    Result is cached to data/regime_gate.json for the current UTC date and
    mirrored to data/regime_state.json ({'regime','as_of','changed_at'}) for
    the sell engine. On data failure, falls back to the last cached state
    (marked stale) and finally to RISK_ON — the gate must degrade to "no
    extra restriction", not to a permanent buy freeze. The gate NEVER
    applies to exits regardless of state.
    """
    today = _today_str()
    if not force_refresh:
        cached = _read_json(REGIME_GATE_PATH)
        if cached and cached.get("date") == today and cached.get("state") in VALID_STATES:
            return {
                "state": cached["state"],
                "vol_multiplier": float(cached.get("vol_multiplier", 1.0)),
                "details": cached.get("details", {}) or {},
            }

    sma_days = int(Config.REGIME_GATE_SMA_DAYS)
    risk_off_sessions = int(Config.REGIME_GATE_RISK_OFF_SESSIONS)
    # 12-mo lookback (253 sessions) + SMA window + a few sessions of slack.
    bars = _fetch_spy_closes(sma_days + 260)
    closes = [b["close"] for b in bars]

    if len(closes) < sma_days + risk_off_sessions:
        return _fallback_state(
            reason=f"insufficient SPY history ({len(closes)} bars, need {sma_days + risk_off_sessions})",
        )

    latest_close = closes[-1]
    sma_now = sum(closes[-sma_days:]) / sma_days
    sessions_below = _consecutive_sessions_below_sma(closes, sma_days)
    risk_off = sessions_below >= risk_off_sessions

    # Trailing 12-month SPY return vs T-bill (only needed for HARD_RISK_OFF).
    lookback = min(253, len(closes) - 1)
    spy_12m_return_pct = (latest_close / closes[-(lookback + 1)] - 1.0) * 100.0
    tbill_pct, tbill_source = _fetch_tbill_rate_pct()
    excess_12m_pct = spy_12m_return_pct - tbill_pct

    if risk_off and excess_12m_pct < 0:
        state = HARD_RISK_OFF
    elif risk_off:
        state = RISK_OFF
    else:
        state = RISK_ON

    vix_value, vix_source = _fetch_vix()
    if vix_value and vix_value > 0:
        vol_multiplier = _clamp(
            float(Config.REGIME_GATE_VOL_TARGET_VIX) / vix_value,
            float(Config.REGIME_GATE_VOL_MULT_MIN),
            float(Config.REGIME_GATE_VOL_MULT_MAX),
        )
    else:
        vol_multiplier = 1.0

    details = {
        "spy_close": round(latest_close, 2),
        "spy_close_date": bars[-1]["date"] if bars else None,
        "sma_days": sma_days,
        "spy_sma": round(sma_now, 2),
        "sessions_below_sma": sessions_below,
        "risk_off_sessions_required": risk_off_sessions,
        "spy_12m_return_pct": round(spy_12m_return_pct, 2),
        "tbill_rate_pct": round(tbill_pct, 3),
        "tbill_source": tbill_source,
        "excess_12m_return_pct": round(excess_12m_pct, 2),
        "vix": vix_value,
        "vix_source": vix_source,
        "bars_used": len(closes),
        "stale": False,
        "notes": (
            "RISK_ON: full buys. RISK_OFF: new momentum/breakout entries need "
            f"conviction >= {Config.REGIME_GATE_RISK_OFF_MIN_CONVICTION} and half size. "
            "HARD_RISK_OFF: no new buys. Exits are NEVER blocked by this gate."
        ),
    }

    result = {"state": state, "vol_multiplier": round(vol_multiplier, 3), "details": details}
    _persist(result, today)
    logger.info(
        "[regime_gate] %s | SPY %.2f vs SMA%d %.2f (%d sessions below) | 12m excess %+.1f%% | vol_mult %.2f",
        state, latest_close, sma_days, sma_now, sessions_below, excess_12m_pct, vol_multiplier,
    )
    return result


def _fallback_state(reason: str) -> dict:
    """Degrade gracefully: last cached state (marked stale), else RISK_ON."""
    cached = _read_json(REGIME_GATE_PATH)
    if cached and cached.get("state") in VALID_STATES:
        logger.warning("[regime_gate] %s — reusing cached state %s from %s",
                       reason, cached.get("state"), cached.get("date"))
        details = dict(cached.get("details") or {})
        details["stale"] = True
        details["stale_reason"] = reason
        return {
            "state": cached["state"],
            "vol_multiplier": float(cached.get("vol_multiplier", 1.0)),
            "details": details,
        }
    logger.warning("[regime_gate] %s — no cache; defaulting to RISK_ON (no extra restriction)", reason)
    return {
        "state": RISK_ON,
        "vol_multiplier": 1.0,
        "details": {"stale": True, "stale_reason": reason, "defaulted": True},
    }


def _persist(result: dict, today: str) -> None:
    """Write data/regime_gate.json and mirror data/regime_state.json."""
    try:
        payload = {
            "schema_version": 1,
            "date": today,
            "generated_at": _utcnow().isoformat(),
            "state": result["state"],
            "vol_multiplier": result["vol_multiplier"],
            "details": result.get("details", {}),
        }
        _atomic_write_json(REGIME_GATE_PATH, payload)
    except Exception as exc:
        logger.warning("[regime_gate] Could not persist %s: %s", REGIME_GATE_PATH, exc)

    try:
        previous = _read_json(REGIME_STATE_PATH) or {}
        now_iso = _utcnow().isoformat()
        changed_at = str(previous.get("changed_at") or "")
        if str(previous.get("regime") or "") != result["state"] or not changed_at:
            changed_at = now_iso
        state_payload = {
            # Sell-engine compatible shape: it reads 'regime' + 'changed_at'.
            "regime": result["state"],
            "as_of": now_iso,
            "changed_at": changed_at,
            "vol_multiplier": result["vol_multiplier"],
            "source": "regime_gate",
        }
        _atomic_write_json(REGIME_STATE_PATH, state_payload)
    except Exception as exc:
        logger.warning("[regime_gate] Could not persist %s: %s", REGIME_STATE_PATH, exc)


def load_cached_regime(max_age_days: int = 3) -> Optional[dict]:
    """Read the cached gate state without any network calls.

    Returns None when the cache is missing or older than max_age_days.
    Callers that cannot afford network (e.g. score mapping) use this.
    """
    cached = _read_json(REGIME_GATE_PATH)
    if not cached or cached.get("state") not in VALID_STATES:
        return None
    try:
        cache_date = datetime.strptime(str(cached.get("date")), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        age_days = (_utcnow() - cache_date).days
        if age_days > max_age_days:
            return None
    except Exception:
        return None
    return {
        "state": cached["state"],
        "vol_multiplier": float(cached.get("vol_multiplier", 1.0)),
        "details": cached.get("details", {}) or {},
    }


def allows_new_buys(state: str | dict | None) -> bool:
    """True unless the gate is HARD_RISK_OFF. Exits are never gated."""
    if isinstance(state, dict):
        state = state.get("state")
    return str(state or RISK_ON).upper() != HARD_RISK_OFF
