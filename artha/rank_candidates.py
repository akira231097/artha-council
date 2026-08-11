"""Machine Ranking Layer — scores and ranks universe candidates by evidence-backed signals.

Uses yfinance batch download for efficiency (ONE daily-bars fetch per symbol —
Close and Volume are extracted from the same download). Computes a cross-sectional,
z-scored composite of research-verified rulebook signals and combines it with a
z-scaled regime fit bonus (capped at ~15% of composite weight).

Signals (all computed from the single daily-bars fetch):
  - 12-1 momentum: total return months t-12..t-2, skipping the most recent month
    (Jegadeesh & Titman 1993; long-only +0.2-0.5%/mo)
  - 52-week-high proximity: close/52wH as a THRESHOLD (>=0.85 pass, >=0.95 strong),
    not a linear score (George & Hwang 2004)
  - Trend template gate (boolean): close>SMA50>SMA150>SMA200 AND SMA200 rising vs
    21d ago AND close >= 1.3x 52w low (Minervini; Zarattini SSRN 5084316: 6.18%/yr alpha)
  - Information discreteness: ID = sgn(12-1 ret) x (%neg days - %pos days) over the
    formation window; smooth (ID <= -0.02) preferred (Da, Gurun & Warachka
    "frog-in-the-pan" 2014)
  - Return consistency: positive months in >= 8 of 12 formation months
  - Lottery exclusion: mean of 5 largest daily returns last month in the top decile
    of the universe -> exclude 30 days (Bali et al. MAX effect)
  - Volume confirmation for breakout-type candidates: day volume >= 1.4x 50d average
"""
from __future__ import annotations

import json
import logging
import gc
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .config import Config
from .collector import YFinanceCollector

logger = logging.getLogger(__name__)

UTC = timezone.utc

# ---------------------------------------------------------------------------
# Tunables (env-var overridable; Wave 2 consolidates into config.py)
# ---------------------------------------------------------------------------

# Composite z-score weights. Regime weight is deliberately capped at ~15%.
RANK_W_MOMENTUM_12_1: float = float(os.getenv("ARTHA_RANK_W_MOMENTUM_12_1", "0.35"))
RANK_W_CONSISTENCY: float = float(os.getenv("ARTHA_RANK_W_CONSISTENCY", "0.10"))
RANK_W_SMOOTHNESS: float = float(os.getenv("ARTHA_RANK_W_SMOOTHNESS", "0.15"))
RANK_W_52W_HIGH: float = float(os.getenv("ARTHA_RANK_W_52W_HIGH", "0.15"))
RANK_W_TREND_TEMPLATE: float = float(os.getenv("ARTHA_RANK_W_TREND_TEMPLATE", "0.10"))
RANK_W_REGIME: float = min(0.15, float(os.getenv("ARTHA_RANK_W_REGIME", "0.15")))

# Scale factor from weighted-z composite to score points (keeps combined_score
# magnitudes broadly comparable with the legacy raw-percentage scores that
# momentum_tracker history and broker_router thresholds were tuned against).
RANK_COMPOSITE_SCALE: float = float(os.getenv("ARTHA_RANK_COMPOSITE_SCALE", "25.0"))

# 52-week-high proximity thresholds (George & Hwang) — threshold, NOT linear.
RANK_52WH_PASS: float = float(os.getenv("ARTHA_RANK_52WH_PASS", "0.85"))
RANK_52WH_STRONG: float = float(os.getenv("ARTHA_RANK_52WH_STRONG", "0.95"))

# Information discreteness preference threshold (Da-Gurun-Warachka).
RANK_ID_SMOOTH_THRESHOLD: float = float(os.getenv("ARTHA_RANK_ID_SMOOTH_THRESHOLD", "-0.02"))

# Return consistency: positive months in >= N of 12 formation months.
RANK_CONSISTENCY_MIN_MONTHS: int = int(os.getenv("ARTHA_RANK_CONSISTENCY_MIN_MONTHS", "8"))

# Lottery (MAX effect) exclusion.
RANK_LOTTERY_EXCLUSION_ENABLED: bool = os.getenv(
    "ARTHA_RANK_LOTTERY_EXCLUSION_ENABLED", "true"
).strip().lower() in ("1", "true", "yes")
RANK_LOTTERY_DECILE_PCT: float = float(os.getenv("ARTHA_RANK_LOTTERY_DECILE_PCT", "90.0"))
RANK_LOTTERY_EXCLUSION_DAYS: int = int(os.getenv("ARTHA_RANK_LOTTERY_EXCLUSION_DAYS", "30"))

# Volume confirmation for breakout-type candidates.
RANK_VOLUME_CONFIRM_RATIO: float = float(os.getenv("ARTHA_RANK_VOLUME_CONFIRM_RATIO", "1.4"))

# Extension flag: last-month (non-skip) return above this is "extended".
RANK_EXTENSION_1M_PCT: float = float(os.getenv("ARTHA_RANK_EXTENSION_1M_PCT", "30.0"))

# Ranking download budget. The paginated universe is ~1800 names ($1B+), so
# operators should extend the budget via ARTHA_RANK_TOTAL_TIMEOUT_SECONDS
# (recommended: 480). Regardless of budget, cap-bucket interleaving below
# ensures budget exhaustion degrades evenly across market-cap buckets instead
# of silently dropping the smallest caps first.
RANK_CAP_BUCKETS: int = int(os.getenv("ARTHA_RANK_CAP_BUCKETS", "5"))


def _rank_total_budget_seconds() -> int:
    """Ranking download budget, read at call time.

    Precedence: ARTHA_RANK_TOTAL_TIMEOUT_SECONDS env var, then
    Config.YFINANCE_RANK_TOTAL_TIMEOUT_SECONDS (legacy 120s default).
    """
    env = os.getenv("ARTHA_RANK_TOTAL_TIMEOUT_SECONDS")
    if env:
        try:
            return max(5, int(env))
        except ValueError:
            pass
    return max(5, int(getattr(Config, "YFINANCE_RANK_TOTAL_TIMEOUT_SECONDS", 120) or 120))

# Where per-ticker scan-time signals are persisted for the sell engine / council.
SCAN_SIGNALS_DIR: Path = Path(
    os.getenv(
        "ARTHA_SCAN_SIGNALS_DIR",
        str(Path(__file__).resolve().parent.parent / "data" / "scan_signals"),
    )
)
SCAN_SIGNALS_SCHEMA_VERSION = 1
RANK_COVERAGE_DIR: Path = Path(
    os.getenv(
        "ARTHA_RANK_COVERAGE_DIR",
        str(Path(__file__).resolve().parent.parent / "data" / "rank_coverage"),
    )
)
RANK_COVERAGE_SCHEMA_VERSION = 1

# Momentum score weights (legacy formula — kept for backward compatibility;
# compute_momentum_score is still exercised by verify_data_layer.py).
MOMENTUM_RETURN_12M_WEIGHT = 0.5
MOMENTUM_RETURN_3M_WEIGHT = 0.3
MOMENTUM_RETURN_1M_WEIGHT = 0.2
MOMENTUM_VOL_PENALTY = 0.2

# Number of trading days to skip (most recent month) to avoid
# short-term reversal effect per Jegadeesh & Titman (1993)
SKIP_DAYS = 22  # ~1 month of trading days

# ~21 trading days per formation "month" for consistency/ID windows
MONTH_DAYS = 21


def compute_momentum_score(
    return_1m: Optional[float],
    return_3m: Optional[float],
    vol_20d: Optional[float],
    return_12m: Optional[float] = None,
) -> float:
    """Compute a momentum score from return and volatility data.

    Legacy raw-percentage formula, kept for backward compatibility (the ranker
    itself now uses the cross-sectional z-scored composite in rank_universe).

    Formula (with 12-month data):
        score = 0.5 * return_12m + 0.3 * return_3m + 0.2 * return_1m
                - 0.2 * vol_penalty

    Fallback (without 12-month data, e.g. IPOs < 1 year old):
        score = 0.6 * return_3m + 0.4 * return_1m - 0.2 * vol_penalty

    All returns should use the skip-month rule: measured from t-N to t-22
    (skipping the most recent ~22 trading days) to avoid short-term
    reversal contamination.

    Where vol_penalty = max(0, vol_20d - 20) to penalize excessive volatility.
    Returns 0.0 if all inputs are None.

    Args:
        return_1m: 1-month return in percent (skip-adjusted: t-44 to t-22)
        return_3m: 3-month return in percent (skip-adjusted: t-85 to t-22)
        vol_20d: 20-day annualized volatility in percent
        return_12m: 12-month return in percent (skip-adjusted: t-274 to t-22)

    Returns:
        Float momentum score (higher = better momentum)
    """
    r1m = float(return_1m) if return_1m is not None else 0.0
    r3m = float(return_3m) if return_3m is not None else 0.0
    r12m = float(return_12m) if return_12m is not None else None
    vol = float(vol_20d) if vol_20d is not None else 20.0

    # Penalize only volatility above 20% annualized
    vol_penalty = max(0.0, vol - 20.0)

    if r12m is not None:
        # Full formula with 12-month signal (strongest academic signal)
        score = (
            MOMENTUM_RETURN_12M_WEIGHT * r12m
            + MOMENTUM_RETURN_3M_WEIGHT * r3m
            + MOMENTUM_RETURN_1M_WEIGHT * r1m
            - MOMENTUM_VOL_PENALTY * vol_penalty
        )
    else:
        # Fallback for stocks with < 1 year of history
        score = (
            0.6 * r3m
            + 0.4 * r1m
            - MOMENTUM_VOL_PENALTY * vol_penalty
        )

    return round(score, 4)


# ---------------------------------------------------------------------------
# Per-symbol signal computation (single daily-bars fetch, vectorized numpy)
# ---------------------------------------------------------------------------

def compute_signal_row(price_arr, volume_arr=None) -> dict:
    """Compute all rulebook signals for one symbol from its daily bars.

    Args:
        price_arr: 1-D numpy array of daily closes (oldest first, ~1 year)
        volume_arr: Optional 1-D numpy array of daily volumes aligned with prices

    Returns:
        Dict of raw signal values. Missing/uncomputable signals are None.
    """
    import numpy as np

    n = len(price_arr)
    latest = float(price_arr[-1])
    skip = min(SKIP_DAYS, n - 1)
    skip_price = float(price_arr[-(skip + 1)])

    row: dict = {
        "price": latest,
        "n_days": n,
    }

    # --- Legacy skip-month returns (kept for downstream schema compat) ---
    return_1m = None
    if n >= (SKIP_DAYS + 22):
        return_1m = ((skip_price / price_arr[-(SKIP_DAYS + 22)]) - 1) * 100
    elif n >= (skip + 5):
        return_1m = ((skip_price / price_arr[0]) - 1) * 100

    return_3m = None
    if n >= (SKIP_DAYS + 63):
        return_3m = ((skip_price / price_arr[-(SKIP_DAYS + 63)]) - 1) * 100
    elif n >= (skip + 10):
        return_3m = ((skip_price / price_arr[0]) - 1) * 100

    # --- 12-1 momentum (t-12mo..t-1mo, skipping the most recent month) ---
    ret_12_1 = None
    if n >= (SKIP_DAYS + 252):
        ret_12_1 = ((skip_price / price_arr[-(SKIP_DAYS + 252)]) - 1) * 100
    elif n >= (SKIP_DAYS + 126):
        # At least 6 months + skip available — use what we have
        ret_12_1 = ((skip_price / price_arr[0]) - 1) * 100

    row["return_1m"] = round(return_1m, 2) if return_1m is not None else None
    row["return_3m"] = round(return_3m, 2) if return_3m is not None else None
    row["return_12m"] = round(ret_12_1, 2) if ret_12_1 is not None else None
    row["ret_12_1"] = row["return_12m"]

    # --- Recent (non-skip) 1-month return, for extension flag ---
    ret_recent_1m = None
    if n >= MONTH_DAYS + 1:
        ret_recent_1m = ((latest / price_arr[-(MONTH_DAYS + 1)]) - 1) * 100
    row["ret_recent_1m"] = round(ret_recent_1m, 2) if ret_recent_1m is not None else None
    row["extension_flag"] = bool(
        ret_recent_1m is not None and ret_recent_1m >= RANK_EXTENSION_1M_PCT
    )

    # --- 20-day annualized volatility (uses recent data, NOT skip-adjusted) ---
    vol_20d = None
    if n >= 21:
        daily_rets = np.diff(price_arr[-21:]) / price_arr[-21:-1]
        vol_20d = float(np.std(daily_rets, ddof=1) * np.sqrt(252) * 100)
    row["vol_20d"] = round(vol_20d, 2) if vol_20d is not None else None

    # --- 52-week-high proximity (threshold, not linear) ---
    high_52w = float(np.max(price_arr))
    low_52w = float(np.min(price_arr))
    prox = latest / high_52w if high_52w > 0 else None
    row["high_52w_prox"] = round(prox, 4) if prox is not None else None
    if prox is None:
        tier = None
    elif prox >= RANK_52WH_STRONG:
        tier = 1.0
    elif prox >= RANK_52WH_PASS:
        tier = 0.5
    else:
        tier = 0.0
    row["high_52w_tier"] = tier

    # --- Trend template gate (Minervini/Zarattini) ---
    trend_template = None
    if n >= 200 + MONTH_DAYS:
        sma50 = float(np.mean(price_arr[-50:]))
        sma150 = float(np.mean(price_arr[-150:]))
        sma200 = float(np.mean(price_arr[-200:]))
        sma200_prev = float(np.mean(price_arr[-(200 + MONTH_DAYS):-MONTH_DAYS]))
        trend_template = bool(
            latest > sma50 > sma150 > sma200
            and sma200 > sma200_prev
            and low_52w > 0
            and latest >= 1.3 * low_52w
        )
        row["sma50"] = round(sma50, 2)
        row["sma150"] = round(sma150, 2)
        row["sma200"] = round(sma200, 2)
    row["trend_template"] = trend_template

    # --- Formation-window daily returns (t-12mo..t-1mo) ---
    if n > skip + 2:
        formation = price_arr[max(0, n - (skip + 252)): n - skip]
    else:
        formation = price_arr[: max(2, n - skip)]
    info_discreteness = None
    positive_months = None
    months_observed = None
    if len(formation) >= MONTH_DAYS:
        f_rets = np.diff(formation) / formation[:-1]
        nz = f_rets[f_rets != 0]
        if len(f_rets) > 0 and ret_12_1 is not None:
            pct_neg = float(np.sum(f_rets < 0)) / len(f_rets)
            pct_pos = float(np.sum(f_rets > 0)) / len(f_rets)
            sign = 1.0 if ret_12_1 > 0 else (-1.0 if ret_12_1 < 0 else 0.0)
            # Da-Gurun-Warachka: ID = sgn(ret) * (%neg - %pos); smooth <= -0.02
            info_discreteness = round(sign * (pct_neg - pct_pos), 4)
        del nz
        # Return consistency: positive months in the formation window
        months_observed = len(formation) // MONTH_DAYS
        pos = 0
        for m in range(months_observed):
            end = len(formation) - m * MONTH_DAYS
            start = end - MONTH_DAYS
            if start < 0:
                break
            if formation[end - 1] > formation[start]:
                pos += 1
        positive_months = pos
    row["information_discreteness"] = info_discreteness
    row["id_smooth"] = bool(
        info_discreteness is not None and info_discreteness <= RANK_ID_SMOOTH_THRESHOLD
    )
    row["positive_months"] = positive_months
    row["months_observed"] = months_observed
    row["consistency_pass"] = bool(
        positive_months is not None
        and months_observed
        and months_observed >= 6
        and positive_months >= round(RANK_CONSISTENCY_MIN_MONTHS * months_observed / 12)
    )

    # --- Lottery (MAX effect): mean of 5 largest daily returns, last month ---
    max5_avg_1m = None
    if n >= MONTH_DAYS + 1:
        recent_rets = np.diff(price_arr[-(MONTH_DAYS + 1):]) / price_arr[-(MONTH_DAYS + 1):-1]
        if len(recent_rets) >= 5:
            top5 = np.sort(recent_rets)[-5:]
            max5_avg_1m = round(float(np.mean(top5)) * 100, 4)
    row["max5_avg_1m"] = max5_avg_1m

    # --- Volume confirmation (breakout-type candidates) ---
    volume_confirmed = None
    if volume_arr is not None and len(volume_arr) >= 51:
        vols = np.asarray(volume_arr, dtype=float)
        vols = np.nan_to_num(vols, nan=0.0)
        avg50 = float(np.mean(vols[-51:-1]))
        if avg50 > 0:
            volume_confirmed = bool(vols[-1] >= RANK_VOLUME_CONFIRM_RATIO * avg50)
            row["volume_ratio_50d"] = round(float(vols[-1]) / avg50, 2)
    row["volume_confirmed"] = volume_confirmed

    return row


def _zscore(values) -> "object":
    """Cross-sectional z-score with NaN passthrough, clipped to [-3, 3]."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) < 3:
        return np.zeros_like(arr)
    std = float(np.std(valid, ddof=1))
    mean = float(np.mean(valid))
    if std <= 1e-12:
        return np.zeros_like(arr)
    z = (arr - mean) / std
    z = np.clip(z, -3.0, 3.0)
    return np.nan_to_num(z, nan=0.0)


def _assign_track(row: dict, regime_score: float, momentum_z: float) -> str:
    """Tag candidate with a track label: momentum | breakout | regime.

    (catalyst_ep candidates come from the scanner catalyst lane, not the ranker.)
    """
    prox = row.get("high_52w_prox")
    if (
        prox is not None
        and prox >= RANK_52WH_STRONG
        and bool(row.get("trend_template"))
        and bool(row.get("volume_confirmed"))
    ):
        return "breakout"
    if regime_score >= 10 and momentum_z < 0.5:
        return "regime"
    return "momentum"


# ---------------------------------------------------------------------------
# Scan-signals persistence (schema consumed by sell engine + council)
# ---------------------------------------------------------------------------

def persist_scan_signals(
    signals_by_symbol: dict[str, dict],
    scan_date: Optional[str] = None,
    regime_type: Optional[str] = None,
    overlays: Optional[list[str]] = None,
    universe_size: Optional[int] = None,
    coverage_summary: Optional[dict] = None,
) -> Optional[str]:
    """Persist per-ticker scan-time signals to data/scan_signals/YYYY-MM-DD.json.

    Schema (version 1):
        {
          "schema_version": 1,
          "date": "YYYY-MM-DD",
          "generated_at": "<iso>",
          "regime_type": str|null,
          "overlays": [str],
          "universe_size": int|null,
          "signals": {
            "<SYM>": {
              "momentum_z": float,        # z of 12-1 momentum (cross-sectional)
              "composite_z": float,       # weighted composite z (ex-regime scale)
              "composite_percentile": float,   # 0..1 within this scan
              "ret_12_1": float|null,     # 12-1 momentum, percent
              "high_52w_prox": float|null,
              "trend_template": bool|null,
              "information_discreteness": float|null,
              "positive_months": int|null,
              "months_observed": int|null,
              "lottery_flag": bool,
              "lottery_excluded_until": "YYYY-MM-DD"|null,
              "volume_confirmed": bool|null,
              "extension_flag": bool,
              "deceleration_flag": bool|null,  # filled by funnel after tracker enrich
              "track": "momentum"|"breakout"|"catalyst_ep"|"regime",
              "regime_score": float,
              "vol_20d": float|null,
              "price": float|null
            }
          }
        }

    Merges into an existing same-day file so multiple writers (ranker, funnel
    deceleration update, catalyst lane) can contribute without clobbering.
    """
    scan_date = scan_date or datetime.now(UTC).strftime("%Y-%m-%d")
    try:
        SCAN_SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
        path = SCAN_SIGNALS_DIR / f"{scan_date}.json"
        payload = {
            "schema_version": SCAN_SIGNALS_SCHEMA_VERSION,
            "date": scan_date,
            "generated_at": datetime.now(UTC).isoformat(),
            "regime_type": regime_type,
            "overlays": overlays or [],
            "universe_size": universe_size,
            "coverage_summary": coverage_summary or {},
            "signals": {},
        }
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    payload["signals"] = existing.get("signals", {}) or {}
                    payload["regime_type"] = regime_type or existing.get("regime_type")
                    payload["overlays"] = overlays or existing.get("overlays") or []
                    def _safe_int(value: object) -> int:
                        try:
                            return int(value or 0)
                        except (TypeError, ValueError):
                            return 0

                    existing_size = _safe_int(existing.get("universe_size"))
                    incoming_size = _safe_int(universe_size)
                    payload["universe_size"] = max(existing_size, incoming_size) or None
                    existing_coverage = existing.get("coverage_summary") or {}
                    existing_coverage_size = _safe_int(existing_coverage.get("universe_size"))
                    incoming_coverage_size = _safe_int((coverage_summary or {}).get("universe_size"))
                    payload["coverage_summary"] = (
                        coverage_summary
                        if coverage_summary and incoming_coverage_size >= existing_coverage_size
                        else existing_coverage
                    )
            except Exception:
                pass
        for symbol, row in signals_by_symbol.items():
            key = str(symbol).upper()
            merged = dict(payload["signals"].get(key) or {})
            merged.update(row)
            payload["signals"][key] = merged
        fd, tmp_path = tempfile.mkstemp(dir=str(SCAN_SIGNALS_DIR), suffix=".tmp", prefix=f".{scan_date}_")
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
        return str(path)
    except Exception as e:
        logger.warning("[rank] Could not persist scan signals: %s", e)
        return None


def update_scan_signals(updates: dict[str, dict], scan_date: Optional[str] = None) -> Optional[str]:
    """Merge per-ticker signal updates (e.g. deceleration flags, track changes)
    into today's scan-signals file. Convenience wrapper for the funnel."""
    return persist_scan_signals(updates, scan_date=scan_date)


def persist_rank_coverage_audit(
    *,
    universe: list,
    symbol_status: dict[str, str],
    history_count: int,
    signal_count: int,
    ranked_count: int,
    lottery_excluded_count: int,
    regime_type: Optional[str],
    overlays: Optional[list[str]],
    elapsed_seconds: float,
    timeout_seconds: int,
) -> Optional[dict]:
    """Persist one immutable, run-level ranking coverage audit.

    Daily scan-signal files merge warm scans, full scans, and later point
    updates. They cannot prove how much of one specific universe was actually
    evaluated. This artifact records one status for every symbol in this run.
    """
    if not getattr(Config, "RANK_COVERAGE_AUDIT_ENABLED", True):
        return None
    try:
        generated = datetime.now(UTC)
        universe_size = len(universe)
        history_pct = history_count / universe_size if universe_size else 0.0
        signal_pct = signal_count / universe_size if universe_size else 0.0
        required_pct = float(getattr(Config, "RANK_MIN_HISTORY_COVERAGE_PCT", 0.90) or 0.90)
        status_counts: dict[str, int] = {}
        rows: list[dict] = []
        for candidate in universe:
            symbol = str(getattr(candidate, "symbol", "") or "").upper()
            status = symbol_status.get(symbol, "not_attempted")
            status_counts[status] = status_counts.get(status, 0) + 1
            rows.append(
                {
                    "symbol": symbol,
                    "status": status,
                    "market_cap": getattr(candidate, "market_cap", None),
                    "sector": getattr(candidate, "sector", ""),
                    "industry": getattr(candidate, "industry", ""),
                }
            )
        summary = {
            "status": "PASS" if history_pct >= required_pct else "WARN",
            "universe_size": universe_size,
            "history_count": history_count,
            "history_coverage_pct": round(history_pct, 4),
            "signal_count": signal_count,
            "signal_coverage_pct": round(signal_pct, 4),
            "ranked_count": ranked_count,
            "lottery_excluded_count": lottery_excluded_count,
            "minimum_history_coverage_pct": required_pct,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "timeout_seconds": timeout_seconds,
            "status_counts": status_counts,
        }
        payload = {
            "schema_version": RANK_COVERAGE_SCHEMA_VERSION,
            "generated_at": generated.isoformat(),
            "regime_type": regime_type,
            "overlays": overlays or [],
            "summary": summary,
            "symbols": rows,
        }
        out_dir = RANK_COVERAGE_DIR / generated.strftime("%Y-%m-%d")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"rank_{generated.strftime('%Y%m%d_%H%M%S_%f')}.json"
        fd, tmp_path = tempfile.mkstemp(dir=str(out_dir), suffix=".tmp", prefix=".rank_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, default=str)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, out_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        logger.info(
            "[rank] Coverage audit %s: histories=%d/%d (%.1f%%), signals=%d, artifact=%s",
            summary["status"],
            history_count,
            universe_size,
            history_pct * 100.0,
            signal_count,
            out_path,
        )
        return {**summary, "artifact_path": str(out_path)}
    except Exception as exc:
        logger.warning("[rank] Could not persist coverage audit: %s", exc)
        return None


def load_latest_rank_coverage_summary(max_age_hours: float = 2.0) -> Optional[dict]:
    """Load the largest recent immutable coverage summary.

    Warmups and diagnostics can run after the full scan with a much smaller
    universe. Prefer coverage breadth first, then recency, so a 2-symbol probe
    cannot overwrite the operator's view of a 1,900-symbol rank run.
    """
    now = datetime.now(UTC)
    candidates: list[tuple[int, datetime, dict]] = []
    for path in sorted(RANK_COVERAGE_DIR.glob("*/*.json"), reverse=True)[:100]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            generated = datetime.fromisoformat(str(payload.get("generated_at") or "").replace("Z", "+00:00"))
            if generated.tzinfo is None:
                generated = generated.replace(tzinfo=UTC)
            age_hours = (now - generated).total_seconds() / 3600.0
            if age_hours < 0 or age_hours > max_age_hours:
                continue
            summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else None
            if not summary:
                continue
            universe_size = int(summary.get("universe_size") or 0)
            if universe_size > 0:
                candidates.append(
                    (universe_size, generated, {**summary, "artifact_path": str(path)})
                )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return max(candidates, key=lambda row: (row[0], row[1]))[2] if candidates else None


def load_active_lottery_exclusions(as_of: Optional[datetime] = None) -> dict[str, str]:
    """Load symbols still inside their 30-day lottery exclusion window.

    Scans recent data/scan_signals/*.json files (Bali MAX effect: names flagged
    in the top MAX5 decile stay excluded for RANK_LOTTERY_EXCLUSION_DAYS days).

    Returns:
        Dict of symbol -> exclusion end date (YYYY-MM-DD).
    """
    as_of = as_of or datetime.now(UTC)
    today = as_of.strftime("%Y-%m-%d")
    active: dict[str, str] = {}
    try:
        if not SCAN_SIGNALS_DIR.exists():
            return {}
        cutoff = (as_of - timedelta(days=RANK_LOTTERY_EXCLUSION_DAYS + 1)).strftime("%Y-%m-%d")
        for path in sorted(SCAN_SIGNALS_DIR.glob("*.json")):
            if path.stem < cutoff:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for symbol, row in (payload.get("signals") or {}).items():
                until = row.get("lottery_excluded_until")
                if row.get("lottery_flag") and until and str(until) >= today:
                    prev = active.get(symbol)
                    if prev is None or str(until) > prev:
                        active[str(symbol).upper()] = str(until)
    except Exception as e:
        logger.debug("[rank] Could not load lottery exclusions: %s", e)
    return active


def _interleave_by_cap_bucket(universe: list, buckets: int = RANK_CAP_BUCKETS) -> list:
    """Order candidates by round-robin across market-cap buckets.

    Ensures that when the yfinance download budget is exhausted mid-run, the
    dropped symbols are spread evenly across cap buckets instead of silently
    dropping the smallest caps first.
    """
    if not universe or buckets <= 1:
        return list(universe)
    ordered = sorted(universe, key=lambda c: getattr(c, "market_cap", 0) or 0, reverse=True)
    size = max(1, (len(ordered) + buckets - 1) // buckets)
    groups = [ordered[i: i + size] for i in range(0, len(ordered), size)]
    interleaved: list = []
    for i in range(size):
        for group in groups:
            if i < len(group):
                interleaved.append(group[i])
    return interleaved


def rank_universe(
    universe: list,
    regime_type: Optional[str] = None,
    overlays: Optional[list[str]] = None,
    top_n: int = 50,
    coverage_out: Optional[dict] = None,
) -> list[dict]:
    """Rank universe candidates by a z-scored composite of rulebook signals.

    Downloads 1 year of daily bars via yfinance batch download (Close AND
    Volume extracted from the SAME fetch — one daily-bars fetch per symbol),
    computes all signals per symbol, z-scores them cross-sectionally, and
    combines them into a composite with a z-scaled regime bonus capped at
    ~15% weight.

    Composite (weights env-overridable):
        0.35 * z(12-1 momentum)          Jegadeesh-Titman skip-month
        0.15 * z(52wH threshold tier)    George-Hwang (>=0.85 / >=0.95)
        0.15 * z(smoothness = -ID)       Da-Gurun-Warachka frog-in-the-pan
        0.10 * z(consistency)            positive months of 12
        0.10 * z(trend template gate)    Minervini/Zarattini boolean
        0.15 * z(regime_score)           regime fit, capped at 15%

    Lottery names (MAX5 in the top universe decile) are excluded for 30 days
    (Bali MAX effect), enforced via data/scan_signals persistence.

    Args:
        universe: List of UniverseCandidate objects (from UniverseBuilder)
        regime_type: Active base regime (for logging/context)
        overlays: Active overlay regime keys
        top_n: Number of top candidates to return (default 50)

    Returns:
        List of dicts with symbol, name, sector, momentum_score, regime_score,
        combined_score, per-signal values, composite_z, composite_percentile,
        and track label ('momentum' | 'breakout' | 'regime').
    """
    import yfinance as yf
    import numpy as np

    if not universe:
        logger.warning("[rank] Empty universe — nothing to rank")
        return []

    ordered = _interleave_by_cap_bucket(universe)
    symbols = [c.symbol for c in ordered]
    total_timeout = _rank_total_budget_seconds()
    logger.info(
        f"[rank] Downloading 1Y daily bars for {len(symbols)} tickers "
        f"(cap-bucket interleaved, budget={total_timeout}s)..."
    )

    # Batch download — ONE daily-bars fetch per symbol (Close + Volume together)
    prices: dict[str, "np.ndarray"] = {}
    volumes: dict[str, "np.ndarray"] = {}
    symbol_status: dict[str, str] = {symbol: "not_attempted" for symbol in symbols}
    started = time.monotonic()
    last_elapsed = 0.0

    def _elapsed() -> float:
        nonlocal last_elapsed
        try:
            last_elapsed = max(last_elapsed, time.monotonic() - started)
        except StopIteration:
            # Unit tests may intentionally provide a finite monotonic clock.
            # The audit should not consume an extra synthetic timestamp.
            pass
        return last_elapsed

    def _download_chunk(chunk: list[str], timeout_seconds: int) -> None:
        """Download and normalize one chunk, retaining per-symbol audit state."""
        for symbol in chunk:
            symbol_status[symbol] = "history_requested"
        tickers_str = " ".join(chunk)
        df = None
        try:
            df = yf.download(
                tickers_str,
                period="1y",
                progress=False,
                group_by="ticker",
                threads=Config.YFINANCE_THREADS,
                timeout=max(1, int(timeout_seconds)),
            )
            if df is None or df.empty:
                for symbol in chunk:
                    symbol_status[symbol] = "price_history_missing"
                return

            cols = df.columns
            is_multi = getattr(cols, "nlevels", 1) == 2
            for symbol in chunk:
                try:
                    series = None
                    vol_series = None
                    if is_multi:
                        lv0 = set(cols.get_level_values(0))
                        lv1 = set(cols.get_level_values(1))
                        if symbol in lv0 and "Close" in lv1:
                            series = df[(symbol, "Close")].dropna()
                            if "Volume" in lv1 and (symbol, "Volume") in cols:
                                vol_series = df[(symbol, "Volume")].reindex(series.index)
                        elif "Close" in lv0 and symbol in lv1:
                            series = df[("Close", symbol)].dropna()
                            if "Volume" in lv0 and ("Volume", symbol) in cols:
                                vol_series = df[("Volume", symbol)].reindex(series.index)
                    else:
                        series = df["Close"].dropna() if "Close" in df.columns else None
                        if series is not None and "Volume" in df.columns:
                            vol_series = df["Volume"].reindex(series.index)

                    if series is not None and len(series) >= 10:
                        prices[symbol] = series.values.astype(float)
                        if vol_series is not None:
                            volumes[symbol] = vol_series.fillna(0).values.astype(float)
                        symbol_status[symbol] = "history_downloaded"
                    else:
                        symbol_status[symbol] = "price_history_missing"
                except Exception:
                    symbol_status[symbol] = "history_parse_failed"
        except Exception as chunk_exc:
            for symbol in chunk:
                symbol_status[symbol] = "history_download_failed"
            raise chunk_exc
        finally:
            if df is not None:
                del df
            YFinanceCollector.cleanup_caches()
            gc.collect()

    try:
        batch_size = max(1, int(getattr(Config, "YFINANCE_BATCH_SIZE", 100) or 100))
        for i in range(0, len(symbols), batch_size):
            elapsed = _elapsed()
            if elapsed >= total_timeout:
                logger.warning(
                    "[rank] yfinance ranking budget exhausted after %.1fs; ranked with %d/%d downloaded price histories (cap-bucket interleaved, so drops spread across cap buckets)",
                    elapsed,
                    len(prices),
                    len(symbols),
                )
                break
            chunk = symbols[i: i + batch_size]
            try:
                remaining_timeout = max(1, int(min(Config.YFINANCE_DOWNLOAD_TIMEOUT_SECONDS, total_timeout - elapsed)))
                _download_chunk(chunk, remaining_timeout)
            except Exception as chunk_e:
                logger.warning(
                    "[rank] yfinance chunk failed for %d ticker(s) starting at %s: %s",
                    len(chunk),
                    chunk[0] if chunk else "?",
                    chunk_e,
                )

        # Retry transiently missing histories in smaller chunks while the same
        # hard wall-clock budget still has room. This does not hide provider
        # failures; every final status remains in the run-level audit.
        target_pct = float(getattr(Config, "RANK_MIN_HISTORY_COVERAGE_PCT", 0.90) or 0.90)
        missing = [symbol for symbol in symbols if symbol not in prices]
        if missing and len(prices) / len(symbols) < target_pct:
            retry_size = max(10, min(50, batch_size // 2 or 10))
            for i in range(0, len(missing), retry_size):
                elapsed = _elapsed()
                if elapsed >= total_timeout:
                    break
                chunk = missing[i: i + retry_size]
                try:
                    remaining_timeout = max(
                        1,
                        int(min(Config.YFINANCE_DOWNLOAD_TIMEOUT_SECONDS, total_timeout - elapsed)),
                    )
                    _download_chunk(chunk, remaining_timeout)
                except Exception as retry_exc:
                    logger.info(
                        "[rank] Retry chunk failed for %d ticker(s) starting at %s: %s",
                        len(chunk),
                        chunk[0] if chunk else "?",
                        retry_exc,
                    )

    except Exception as e:
        logger.error(f"[rank] yfinance batch download failed: {e}")

    for symbol in symbols:
        if symbol_status.get(symbol) in {"not_attempted", "history_requested"}:
            symbol_status[symbol] = "ranking_budget_exhausted"

    logger.info(f"[rank] Got price data for {len(prices)}/{len(symbols)} tickers")

    # Build candidate lookup
    candidate_map = {c.symbol: c for c in universe}

    # --- Per-symbol raw signals from the single fetch ---
    rows: list[dict] = []
    for sym, price_arr in prices.items():
        candidate = candidate_map.get(sym)
        if not candidate:
            symbol_status[sym] = "universe_mapping_missing"
            continue
        try:
            row = compute_signal_row(price_arr, volumes.get(sym))
        except Exception as sig_e:
            logger.debug("[rank] Signal computation failed for %s: %s", sym, sig_e)
            symbol_status[sym] = "signal_computation_failed"
            continue
        row["symbol"] = sym
        row["_candidate"] = candidate
        rows.append(row)
        symbol_status[sym] = "signal_ready"

    if not rows:
        logger.warning("[rank] No signal rows computed — nothing to rank")
        coverage_summary = persist_rank_coverage_audit(
            universe=universe,
            symbol_status=symbol_status,
            history_count=len(prices),
            signal_count=0,
            ranked_count=0,
            lottery_excluded_count=0,
            regime_type=regime_type,
            overlays=overlays,
            elapsed_seconds=_elapsed(),
            timeout_seconds=total_timeout,
        )
        if coverage_out is not None:
            coverage_out.clear()
            if coverage_summary:
                coverage_out.update(coverage_summary)
        return []

    # --- Cross-sectional z-scores ---
    def _col(key: str, transform=None):
        vals = []
        for r in rows:
            v = r.get(key)
            if isinstance(v, bool):
                v = 1.0 if v else 0.0
            if v is None:
                vals.append(np.nan)
            else:
                vals.append(transform(float(v)) if transform else float(v))
        return np.asarray(vals, dtype=float)

    z_mom = _zscore(_col("ret_12_1"))
    z_tier = _zscore(_col("high_52w_tier"))
    z_smooth = _zscore(_col("information_discreteness", transform=lambda v: -v))
    consistency_raw = np.asarray(
        [
            (r["positive_months"] / r["months_observed"])
            if (r.get("positive_months") is not None and r.get("months_observed"))
            else np.nan
            for r in rows
        ],
        dtype=float,
    )
    z_consistency = _zscore(consistency_raw)
    z_template = _zscore(_col("trend_template"))
    z_regime = _zscore(np.asarray([float(r["_candidate"].regime_score) for r in rows]))

    # Shadow-only industry momentum. The value is persisted and outcome-tested
    # but deliberately excluded from the live composite until governance says
    # enough point-in-time observations have accumulated.
    industry_values: dict[str, list[float]] = {}
    for row in rows:
        industry = str(getattr(row["_candidate"], "industry", "") or "unknown")
        value = row.get("ret_12_1")
        if value is not None:
            industry_values.setdefault(industry, []).append(float(value))
    industry_medians = {
        industry: float(np.median(values))
        for industry, values in industry_values.items()
        if values
    }
    industry_raw = np.asarray(
        [industry_medians.get(str(getattr(row["_candidate"], "industry", "") or "unknown"), np.nan) for row in rows],
        dtype=float,
    )
    z_industry_shadow = _zscore(industry_raw)

    # --- Lottery exclusion (MAX effect): top decile of MAX5, exclude 30 days ---
    max5 = _col("max5_avg_1m")
    lottery_threshold = None
    lottery_flags = np.zeros(len(rows), dtype=bool)
    if RANK_LOTTERY_EXCLUSION_ENABLED:
        valid_max5 = max5[~np.isnan(max5)]
        if len(valid_max5) >= 20:
            lottery_threshold = float(np.percentile(valid_max5, RANK_LOTTERY_DECILE_PCT))
            lottery_flags = ~np.isnan(max5) & (max5 >= lottery_threshold)
    prior_exclusions = load_active_lottery_exclusions() if RANK_LOTTERY_EXCLUSION_ENABLED else {}

    scan_date = datetime.now(UTC).strftime("%Y-%m-%d")
    exclusion_until = (datetime.now(UTC) + timedelta(days=RANK_LOTTERY_EXCLUSION_DAYS)).strftime("%Y-%m-%d")

    ranked: list[dict] = []
    signals_payload: dict[str, dict] = {}
    excluded_count = 0
    for idx, row in enumerate(rows):
        sym = row["symbol"]
        candidate = row.pop("_candidate")
        regime_score = float(candidate.regime_score)

        momentum_z = float(z_mom[idx])
        composite_ex_regime = (
            RANK_W_MOMENTUM_12_1 * momentum_z
            + RANK_W_52W_HIGH * float(z_tier[idx])
            + RANK_W_SMOOTHNESS * float(z_smooth[idx])
            + RANK_W_CONSISTENCY * float(z_consistency[idx])
            + RANK_W_TREND_TEMPLATE * float(z_template[idx])
        )
        composite_z = composite_ex_regime + RANK_W_REGIME * float(z_regime[idx])

        mom_score = round(composite_ex_regime * RANK_COMPOSITE_SCALE, 2)
        combined = round(composite_z * RANK_COMPOSITE_SCALE, 2)

        lottery_flag = bool(lottery_flags[idx])
        prior_until = prior_exclusions.get(sym)
        lottery_excluded = lottery_flag or prior_until is not None
        track = _assign_track(row, regime_score, momentum_z)
        industry = str(getattr(candidate, "industry", "") or "unknown")
        industry_shadow = {
            "enabled": bool(getattr(Config, "ALPHA_SHADOW_SIGNALS_ENABLED", True)),
            "industry": industry,
            "peer_count": len(industry_values.get(industry) or []),
            "median_ret_12_1": round(industry_medians[industry], 2) if industry in industry_medians else None,
            "z_score": round(float(z_industry_shadow[idx]), 4),
            "live_score_impact": 0.0,
        }

        entry = {
            "symbol": sym,
            "name": candidate.name,
            "sector": candidate.sector,
            "industry": candidate.industry,
            "market_cap": candidate.market_cap,
            "avg_volume": candidate.volume,
            "price": round(row["price"], 2),
            "momentum_score": mom_score,
            "regime_score": round(regime_score, 2),
            "combined_score": combined,
            "composite_z": round(composite_z, 4),
            "momentum_z": round(momentum_z, 4),
            "return_1m": row.get("return_1m"),
            "return_3m": row.get("return_3m"),
            "return_12m": row.get("return_12m"),
            "vol_20d": row.get("vol_20d"),
            "high_52w_prox": row.get("high_52w_prox"),
            "high_52w_tier": row.get("high_52w_tier"),
            "trend_template": row.get("trend_template"),
            "information_discreteness": row.get("information_discreteness"),
            "id_smooth": row.get("id_smooth"),
            "positive_months": row.get("positive_months"),
            "months_observed": row.get("months_observed"),
            "consistency_pass": row.get("consistency_pass"),
            "max5_avg_1m": row.get("max5_avg_1m"),
            "lottery_flag": lottery_flag,
            "volume_confirmed": row.get("volume_confirmed"),
            "volume_ratio_50d": row.get("volume_ratio_50d"),
            "extension_flag": row.get("extension_flag"),
            "ret_recent_1m": row.get("ret_recent_1m"),
            "track": track,
            "industry_momentum_shadow": industry_shadow,
        }

        signals_payload[sym] = {
            "momentum_z": round(momentum_z, 4),
            "composite_z": round(composite_z, 4),
            "composite_percentile": None,  # filled after sort below
            "ret_12_1": row.get("ret_12_1"),
            "high_52w_prox": row.get("high_52w_prox"),
            "trend_template": row.get("trend_template"),
            "information_discreteness": row.get("information_discreteness"),
            "positive_months": row.get("positive_months"),
            "months_observed": row.get("months_observed"),
            "lottery_flag": lottery_flag,
            "lottery_excluded_until": (
                (prior_until or exclusion_until) if lottery_excluded else None
            ),
            "volume_confirmed": row.get("volume_confirmed"),
            "extension_flag": row.get("extension_flag"),
            "deceleration_flag": None,
            "track": track,
            "regime_score": round(regime_score, 2),
            "vol_20d": row.get("vol_20d"),
            "price": round(row["price"], 2),
            "industry_momentum_shadow": industry_shadow,
        }

        if lottery_excluded:
            symbol_status[sym] = "lottery_excluded"
            excluded_count += 1
            logger.info(
                "[rank] Lottery exclusion (MAX effect): %s max5_avg=%.2f%% excluded until %s",
                sym,
                row.get("max5_avg_1m") or 0.0,
                prior_until or exclusion_until,
            )
            continue

        symbol_status[sym] = "ranked"
        ranked.append(entry)

    # Sort by combined score descending; assign composite percentiles
    ranked.sort(key=lambda x: x["combined_score"], reverse=True)
    total = len(ranked)
    for pos, entry in enumerate(ranked):
        percentile = round(1.0 - (pos / total), 4) if total > 1 else 1.0
        entry["composite_percentile"] = percentile
        payload_row = signals_payload.get(entry["symbol"])
        if payload_row is not None:
            payload_row["composite_percentile"] = percentile

    coverage_summary = persist_rank_coverage_audit(
        universe=universe,
        symbol_status=symbol_status,
        history_count=len(prices),
        signal_count=len(rows),
        ranked_count=len(ranked),
        lottery_excluded_count=excluded_count,
        regime_type=regime_type,
        overlays=overlays,
        elapsed_seconds=_elapsed(),
        timeout_seconds=total_timeout,
    )
    if coverage_out is not None:
        coverage_out.clear()
        if coverage_summary:
            coverage_out.update(coverage_summary)
    persist_scan_signals(
        signals_payload,
        scan_date=scan_date,
        regime_type=regime_type,
        overlays=overlays,
        universe_size=len(universe),
        coverage_summary=coverage_summary,
    )

    top = ranked[:top_n]
    logger.info(
        f"[rank] Ranked {len(ranked)} candidates → top {len(top)} "
        f"(regime={regime_type}, overlays={overlays}, "
        f"lottery_excluded={excluded_count}, "
        f"tracks={ {t: sum(1 for e in top if e['track'] == t) for t in ('momentum', 'breakout', 'regime')} })"
    )
    return top
