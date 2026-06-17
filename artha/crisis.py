"""Crisis Mode v3 — Probabilistic Regime Intelligence.

Modules:
- CrisisState / PortfolioPhase enums
- CrisisFingerprint: multi-signal probability vector classification
- CrisisStateManager: SPY drawdown state machine with hysteresis
- QualityFilter: crisis stock quality gates
- ValueTrapDetector: revenue decline / balance sheet deterioration
- SmartMoneyTracker: insider cluster detection

All state is persisted to data/crisis_state.json for cross-process continuity.
"""
from __future__ import annotations

import fcntl
import json
import logging
import math
import os
import statistics
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .config import Config

logger = logging.getLogger(__name__)


def _to_float(val: Any) -> Optional[float]:
    """Safely coerce value to float. Returns None for invalid/missing values."""
    if val is None:
        return None
    if isinstance(val, str):
        val = val.strip()
        if val in ("", "N/A", "None", "null", "nan", "inf", "-inf"):
            return None
    try:
        result = float(val)
        import math
        if not math.isfinite(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely coerce value to float, handling None, strings, and sentinel values."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    """Safely coerce value to int."""
    if val is None:
        return default
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default

CRISIS_STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "crisis_state.json"


# ---------------------------------------------------------------------------
# Item 1: Observation Date — single source of truth
# ---------------------------------------------------------------------------

def get_as_of_date(as_of_date: Optional[str] = None) -> date:
    """Single source of truth for the observation date across all crisis analysis.

    Accepts an ISO date string (e.g. '2026-03-07' or full ISO datetime).
    Falls back to today when None or invalid.  All date-dependent logic
    should call this rather than date.today() directly so that backtests
    and replay scenarios use consistent dates.
    """
    if as_of_date:
        try:
            return date.fromisoformat(str(as_of_date)[:10])
        except ValueError:
            logger.warning(f"[crisis] Invalid as_of_date '{as_of_date}', using today")
    return date.today()


def is_market_stale_day(as_of: Optional[date] = None) -> bool:
    """Return True if market data is likely stale (weekend or fixed US holiday).

    On stale days the system should reduce confidence in real-time market
    signals (VIX, SPY, HY spreads, etc.) because exchanges are closed and
    feeds reflect Friday's close.

    Checks:
    - Saturday / Sunday
    - Fixed US market holidays: New Year's Day, Juneteenth, Independence Day, Christmas
    """
    d = as_of or date.today()
    # Weekend
    if d.weekday() >= 5:
        return True
    # Fixed US market holidays (month, day)
    _FIXED_HOLIDAYS = {(1, 1), (6, 19), (7, 4), (12, 25)}
    if (d.month, d.day) in _FIXED_HOLIDAYS:
        return True
    return False


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CrisisState(str, Enum):
    NORMAL = "normal"
    CORRECTION = "correction"
    BEAR = "bear"
    PANIC = "panic"


class PortfolioPhase(str, Enum):
    INCEPTION = "inception"      # < $10,000
    GROWTH = "growth"            # $10,000 – $25,000
    ESTABLISHED = "established"  # > $25,000


class CrisisType(str, Enum):
    BANKING = "BANKING"
    PANDEMIC = "PANDEMIC"
    GEOPOLITICAL = "GEOPOLITICAL"
    TECH_BUBBLE = "TECH_BUBBLE"
    STAGFLATION = "STAGFLATION"
    UNKNOWN = "UNKNOWN"  # No known fingerprint matches well


# ---------------------------------------------------------------------------
# Item 2: Data Periodicity Validator
# ---------------------------------------------------------------------------

class DataPeriodicityValidator:
    """Validate that financial data has the expected periodicity.

    Prevents apples-to-oranges comparisons — e.g. using annual revenue
    data for a quarterly YoY check, or using a single-quarter FCF instead
    of TTM FCF for quality filter comparisons.
    """

    # FMP quarterly period strings
    QUARTERLY_PERIOD_VALUES = {"Q1", "Q2", "Q3", "Q4"}
    ANNUAL_PERIOD_VALUES = {"FY", "annual"}

    @classmethod
    def validate_quarterly_records(cls, records: list, field_name: str) -> dict:
        """Validate that a list of financial records is quarterly (not annual).

        FMP quarterly records carry a 'period' field in ('Q1','Q2','Q3','Q4').
        Annual records have period == 'FY' or a 4-digit year string.
        """
        if not records or not isinstance(records, list):
            return {"valid": False, "reason": f"No {field_name} records", "count": 0, "warning": False}

        q_count = 0
        a_count = 0
        for rec in records[:8]:
            period = rec.get("period", "")
            if period in cls.QUARTERLY_PERIOD_VALUES:
                q_count += 1
            elif period in cls.ANNUAL_PERIOD_VALUES or (
                isinstance(period, str) and len(period) == 4 and period.isdigit()
            ):
                a_count += 1

        if q_count == 0 and a_count == 0:
            # No period field — cannot determine; be permissive but warn
            return {
                "valid": True,
                "reason": f"No period field in {field_name} — cannot validate periodicity",
                "quarterly_count": 0,
                "annual_count": 0,
                "warning": True,
            }

        is_quarterly = q_count >= a_count
        return {
            "valid": is_quarterly,
            "reason": (
                f"✅ {field_name}: {q_count} quarterly records"
                if is_quarterly
                else f"⚠️ {field_name}: {a_count} annual records found (expected quarterly)"
            ),
            "quarterly_count": q_count,
            "annual_count": a_count,
            "warning": not is_quarterly,
        }

    @classmethod
    def validate_ttm_field(cls, value: Any, field_name: str) -> dict:
        """Validate a TTM metric is present and a finite number."""
        v = _to_float(value)
        if v is None:
            return {"valid": False, "reason": f"{field_name} TTM data missing/invalid"}
        return {"valid": True, "reason": f"✅ {field_name} TTM = {v:.4f}"}

    @classmethod
    def validate_for_value_trap(cls, stock_data: dict) -> dict:
        """Run all periodicity checks required by ValueTrapDetector."""
        income = stock_data.get("income_statement") or []
        balance = stock_data.get("balance_sheet") or []

        issues: list[str] = []
        warnings: list[str] = []

        for records, name in [(income, "income_statement"), (balance, "balance_sheet")]:
            result = cls.validate_quarterly_records(records, name)
            if not result["valid"]:
                issues.append(result["reason"])
            elif result.get("warning"):
                warnings.append(result["reason"])

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
        }

    @classmethod
    def validate_for_quality_filter(cls, stock_data: dict) -> dict:
        """Run periodicity checks required by QualityFilter."""
        ratios_ttm = stock_data.get("ratios_ttm") or {}
        key_metrics_ttm = stock_data.get("key_metrics_ttm") or {}

        checks = {
            "fcf_ttm": cls.validate_ttm_field(
                ratios_ttm.get("freeCashFlowPerShareTTM"), "FCF/share"
            ),
            "roic_ttm": cls.validate_ttm_field(
                key_metrics_ttm.get("returnOnInvestedCapitalTTM"), "ROIC"
            ),
            "de_ttm": cls.validate_ttm_field(
                ratios_ttm.get("debtToEquityRatioTTM"), "D/E ratio"
            ),
        }
        issues = [v["reason"] for v in checks.values() if not v["valid"]]
        return {"valid": len(issues) == 0, "issues": issues, "checks": checks}


# ---------------------------------------------------------------------------
# Item 4: Per-Signal Freshness / Staleness Model
# ---------------------------------------------------------------------------

class SignalFreshnessModel:
    """Per-signal staleness detection and weight adjustment.

    When market data is stale (weekends, FRED publication lag, holiday),
    reduce the weight of affected signals in the fingerprint rather than
    using potentially misleading data at full confidence.

    Weight 1.0 = fully fresh.  Weight 0.0 = missing.  0 < w < 1 = partial.
    """

    # Maximum acceptable age (hours) before weight starts decaying
    SIGNAL_MAX_AGE_HOURS: dict[str, float] = {
        "vix": 26,                      # Market-hours signal: ~1 trading day
        "spy_drawdown": 26,
        "fear_greed": 48,               # F&G index: updated daily, 2-day tolerance
        "hy_oas": 26,
        "ig_oas": 26,
        "yield_curve_spread": 26,
        "oil_price": 26,
        "dxy": 26,
        "fed_funds_rate": 7 * 24,       # FRED: updated daily but lag ~1 week
        "unemployment_rate": 35 * 24,   # FRED: monthly release
        "initial_jobless_claims": 8 * 24,  # FRED: weekly Thursday release
    }

    # Market-hours signals that go stale on weekends/holidays
    _MARKET_HOURS_SIGNALS = frozenset({
        "vix", "spy_drawdown", "fear_greed",
        "hy_oas", "ig_oas", "yield_curve_spread", "oil_price", "dxy",
    })

    MIN_WEIGHT: float = 0.25  # Never drop to zero for present data

    @classmethod
    def compute_signal_weights(
        cls,
        signals: dict,
        as_of: Optional[date] = None,
        collected_at: Optional[str] = None,
    ) -> dict[str, float]:
        """Return per-signal weight multipliers based on freshness.

        Args:
            signals:      The signals dict from DataCollector.
            as_of:        Observation date.
            collected_at: ISO timestamp when data was collected (from signals dict).

        Returns:
            {signal_key: weight} where 1.0=fresh, <1.0=stale, 0.0=missing.
        """
        ref_date = as_of or date.today()
        stale_day = is_market_stale_day(ref_date)

        # Hours since data was collected
        hours_since: Optional[float] = None
        if collected_at:
            try:
                collected_dt = datetime.fromisoformat(
                    str(collected_at).replace("Z", "+00:00")
                )
                hours_since = (datetime.now(timezone.utc) - collected_dt).total_seconds() / 3600
            except (ValueError, TypeError):
                pass

        weights: dict[str, float] = {}
        for key, max_age in cls.SIGNAL_MAX_AGE_HOURS.items():
            val = signals.get(key)
            if val is None:
                weights[key] = 0.0  # Missing → no evidence
                continue

            weight = 1.0

            # Weekend/holiday: market-hours signals are stale at last close
            if stale_day and key in cls._MARKET_HOURS_SIGNALS:
                weight = 0.5  # Friday close data still useful but not current

            # Age-based decay when we have collection timestamp
            if hours_since is not None and hours_since > max_age:
                excess = hours_since - max_age
                # Linear decay over 2× the max_age window
                decay = (excess / (max_age * 2)) * (1.0 - cls.MIN_WEIGHT)
                weight = min(weight, max(cls.MIN_WEIGHT, 1.0 - decay))

            weights[key] = round(weight, 3)

        return weights

    @classmethod
    def get_staleness_report(
        cls,
        signals: dict,
        as_of: Optional[date] = None,
        collected_at: Optional[str] = None,
    ) -> dict:
        """Human-readable staleness summary for logging and dashboards."""
        weights = cls.compute_signal_weights(signals, as_of, collected_at)
        stale = [k for k, w in weights.items() if 0 < w < 1.0]
        missing = [k for k, w in weights.items() if w == 0.0]
        fresh = [k for k, w in weights.items() if w == 1.0]
        overall = sum(weights.values()) / max(len(weights), 1)

        return {
            "weights": weights,
            "fresh_signals": fresh,
            "stale_signals": stale,
            "missing_signals": missing,
            "is_market_stale_day": is_market_stale_day(as_of or date.today()),
            "overall_data_quality": round(overall, 3),
            "summary": (
                f"Data quality {overall:.0%}: "
                f"{len(fresh)} fresh, {len(stale)} stale, {len(missing)} missing"
            ),
        }


# ---------------------------------------------------------------------------
# Crisis Fingerprint definitions
# ---------------------------------------------------------------------------

# Each crisis type has weighted signal thresholds.
# Values are (direction, threshold) pairs:
#   direction: "above" | "below" | "true"
#   threshold: numeric comparison value or True/False
CRISIS_FINGERPRINT_SIGNALS: dict[str, dict[str, tuple]] = {
    CrisisType.BANKING: {
        "hy_oas": ("above", 8.0),             # HY spread > 800 bps (stored as %)
        "ig_oas": ("above", 2.0),             # IG spread > 200 bps
        "yield_curve_spread": ("below", 0.0), # Inverted yield curve
        "vix": ("above", 30.0),
    },
    CrisisType.PANDEMIC: {
        "vix": ("above", 50.0),
        "initial_jobless_claims": ("above", 500_000.0),  # 500K+ claims = shock
        "yield_curve_spread": ("below", 0.5),
    },
    CrisisType.GEOPOLITICAL: {
        "oil_price": ("above", 100.0),        # Oil surges above $100
        "vix": ("above", 25.0),
        "dxy": ("above", 103.0),              # Capital flight to USD
        "ig_oas": ("above", 1.5),
    },
    CrisisType.TECH_BUBBLE: {
        "vix": ("above", 25.0),
        "hy_oas": ("above", 5.0),
        "yield_curve_spread": ("above", 0.3),  # Curve NOT inverted (not recession-led)
    },
    CrisisType.STAGFLATION: {
        "unemployment_rate": ("above", 5.0),
        "fed_funds_rate": ("above", 4.5),      # Fed holding rates high
        "oil_price": ("above", 85.0),
        "yield_curve_spread": ("below", 0.5),
    },
}

# Allocation templates per crisis type
CRISIS_ALLOCATION_TEMPLATES: dict[str, dict] = {
    CrisisType.BANKING: {
        "vti_vxus_split": (70, 30),
        "sector_overweight": ["XLK", "XLV"],
        "sector_avoid": ["XLF"],
    },
    CrisisType.PANDEMIC: {
        "vti_vxus_split": (80, 20),
        "sector_overweight": ["XLK", "XLV", "XLP"],
        "sector_avoid": ["XLE"],
    },
    CrisisType.GEOPOLITICAL: {
        "vti_vxus_split": (85, 15),
        "sector_overweight": ["XLE", "XLI"],
        "sector_avoid": [],
    },
    CrisisType.TECH_BUBBLE: {
        "vti_vxus_split": (65, 35),
        "sector_overweight": ["XLV", "XLP", "XLU"],
        "sector_avoid": ["XLK"],
    },
    CrisisType.STAGFLATION: {
        "vti_vxus_split": (60, 40),
        "sector_overweight": ["XLE", "XLU"],
        "sector_avoid": ["XLK", "XLY"],
    },
    CrisisType.UNKNOWN: {
        "vti_vxus_split": (75, 25),
        "sector_overweight": [],
        "sector_avoid": [],
    },
}

# Default core allocation (when no crisis or low confidence)
DEFAULT_CORE_ALLOCATION = {"vti_pct": 75.0, "vxus_pct": 25.0}


# ---------------------------------------------------------------------------
# Signal matching helpers
# ---------------------------------------------------------------------------

def _signal_matches(actual: float, direction: str, threshold: float) -> float:
    """Return match strength (0.0, 0.5, or 1.0) for a signal.

    Returns 1.0 if clearly matches, 0.5 if borderline (within 20% of
    |threshold| or 0.2 absolute for zero thresholds), 0.0 if does not match.
    """
    # Absolute tolerance: 20% of |threshold|, minimum 0.2 for near-zero thresholds
    tolerance = max(abs(threshold) * 0.20, 0.2)

    if direction == "above":
        if actual >= threshold:
            return 1.0
        elif actual >= threshold - tolerance:
            return 0.5
        return 0.0
    elif direction == "below":
        if actual <= threshold:
            return 1.0
        elif actual <= threshold + tolerance:
            return 0.5
        return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# Crisis Fingerprint Engine
# ---------------------------------------------------------------------------

class CrisisFingerprint:
    """Probabilistic multi-label crisis classifier.

    Computes probability vector across 5 crisis types based on
    real-time market signals.  Includes:
    - Confidence decay (5% per week without confirming signals)
    - Entropy penalty (high entropy = low overall confidence)
    - Dominant type shift detection with hard kill switch
    """

    def __init__(self, history_path: Optional[Path] = None):
        self._history_path = history_path or (
            Path(__file__).resolve().parent.parent / "data" / "fingerprint_history.json"
        )
        self._history: list[dict] = self._load_history()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def classify(self, signals: dict, observation_date: Optional[str] = None) -> dict:
        """Classify current crisis type from signal dict.

        Returns:
            {
                "probabilities": {crisis_type: float, ...},   # sum ~1.0
                "dominant": str,                               # top crisis type
                "dominant_prob": float,
                "confidence": float,                           # 0-1 overall
                "entropy": float,                              # 0-1 (high = uncertain)
                "allocation": dict,                            # blended VTI/VXUS split
                "flip_alert": bool,                            # True if kill switch triggered
                "interpretation": str,
                "signals_used": dict,                          # which signals had values
            }
        """
        raw_scores, signal_coverage = self._compute_raw_scores(signals)
        probabilities = self._normalize(raw_scores)

        # Signal coverage: average across all types
        avg_coverage = sum(signal_coverage.values()) / max(len(signal_coverage), 1)

        # Apply entropy penalty + coverage penalty
        entropy = self._compute_entropy(probabilities)
        # Confidence: penalize both high entropy AND low signal coverage
        # Zero coverage → confidence 0; full coverage + low entropy → high confidence
        raw_confidence = max(0.0, 1.0 - entropy * 0.7)
        confidence = raw_confidence * min(avg_coverage * 1.2, 1.0)  # Coverage floor

        # Determine dominant type — require minimum raw evidence
        max_raw_score = max(raw_scores.values()) if raw_scores else 0.0
        MIN_RAW_SCORE_THRESHOLD = 0.15  # At least 15% match needed

        if max_raw_score < MIN_RAW_SCORE_THRESHOLD or avg_coverage < 0.1:
            dominant = CrisisType.UNKNOWN
            dominant_prob = 0.0
            confidence = 0.0
        else:
            dominant = max(probabilities, key=probabilities.get)
            dominant_prob = probabilities[dominant]

        # Record in history for decay + flip detection (one entry per day)
        # Use observation date if provided to avoid stale-rerun artifacts
        if observation_date:
            today_str = observation_date[:10]
        else:
            today_str = date.today().isoformat()
        entry = {
            "date": today_str,
            "probabilities": probabilities,
            "dominant": dominant,
            "signal_coverage": signal_coverage,
            "avg_coverage": round(avg_coverage, 3),
            "signals_snapshot": {k: v for k, v in signals.items()
                                  if k not in ("sector_etfs", "_raw", "collected_at")},
        }
        # Deduplicate: overwrite today's entry if already exists
        if self._history and self._history[-1].get("date") == today_str:
            self._history[-1] = entry
        else:
            self._history.append(entry)
        self._save_history()

        # Check hard kill switch — use observation date for consistency
        try:
            ref_date = date.fromisoformat(today_str) if today_str else None
        except ValueError:
            ref_date = None
        flip_alert = self._check_flip_alert(reference_date=ref_date)

        # Compute blended allocation — skip crisis weighting when no evidence
        if dominant == CrisisType.UNKNOWN or confidence < 0.10:
            allocation = {
                "vti_pct": 75.0,
                "vxus_pct": 25.0,
                "overweight_sectors": [],
                "avoid_sectors": [],
                "dominant_weight": 0.0,
                "dominant_type": CrisisType.UNKNOWN,
            }
        else:
            allocation = self._compute_blended_allocation(probabilities, confidence)

        # Signals used (non-None values)
        signals_used = {
            k: v for k, v in signals.items()
            if k not in ("sector_etfs", "_raw", "collected_at") and v is not None
        }

        return {
            "scores": probabilities,  # Renamed: heuristic scores, not true probabilities
            "probabilities": probabilities,  # Kept for backward compat
            "dominant": dominant,
            "dominant_prob": round(dominant_prob, 3),
            "confidence": round(confidence, 3),
            "entropy": round(entropy, 3),
            "signal_coverage": round(avg_coverage, 3),
            "allocation": allocation,
            "flip_alert": flip_alert,
            "interpretation": self._interpret(dominant, dominant_prob, confidence, signals),
            "signals_used": signals_used,
        }

    def apply_confidence_decay(self, probabilities: dict, days_since_signal: int) -> dict:
        """Decay probabilities toward UNKNOWN when confirming signals are stale.

        5% decay per week. Floor at 30% of original.
        Mass shifted to UNKNOWN bucket to reflect uncertainty.
        """
        decay_rate = 0.05 * (days_since_signal / 7)
        decay_factor = max(0.30, 1.0 - decay_rate)
        decayed = {k: v * decay_factor for k, v in probabilities.items()}
        # Shift lost mass to UNKNOWN to reflect growing uncertainty
        lost_mass = sum(probabilities.values()) - sum(decayed.values())
        decayed[CrisisType.UNKNOWN] = decayed.get(CrisisType.UNKNOWN, 0.0) + lost_mass
        total = sum(decayed.values())
        if total > 0:
            return {k: v / total for k, v in decayed.items()}
        return decayed

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_raw_scores(self, signals: dict) -> tuple[dict[str, float], dict[str, float]]:
        """Compute raw scores AND signal coverage per crisis type.

        Uses total expected signals (not just available ones) to prevent
        sparse evidence from creating overconfidence.
        """
        scores: dict[str, float] = {}
        coverage: dict[str, float] = {}
        for crisis_type, fingerprint in CRISIS_FINGERPRINT_SIGNALS.items():
            expected_count = len(fingerprint)
            available_count = 0
            matched = 0.0
            for signal_key, (direction, threshold) in fingerprint.items():
                actual = signals.get(signal_key)
                if actual is None:
                    continue  # Missing signal counts against coverage
                available_count += 1
                matched += _signal_matches(_safe_float(actual), direction, float(threshold))
            # Score against EXPECTED count (not available) to penalize missing data
            scores[crisis_type] = matched / max(expected_count, 1)
            coverage[crisis_type] = available_count / max(expected_count, 1)
        return scores, coverage

    def _normalize(self, scores: dict) -> dict[str, float]:
        total = sum(scores.values())
        if total <= 0:
            # Zero evidence → return all zeros (not uniform distribution)
            # This prevents downstream allocation from acting on non-evidence
            return {k: 0.0 for k in scores}
        # Do NOT round internally — preserve precision for entropy/allocation
        return {k: v / total for k, v in scores.items()}

    def _compute_entropy(self, probabilities: dict) -> float:
        """Shannon entropy normalized to [0, 1]."""
        n = len(probabilities)
        if n <= 1:
            return 0.0
        h = 0.0
        for p in probabilities.values():
            if p > 0:
                h -= p * math.log2(p)
        max_entropy = math.log2(n)
        return h / max_entropy if max_entropy > 0 else 0.0

    def _compute_blended_allocation(self, probs: dict, confidence: float) -> dict:
        """Blend allocation templates by crisis probability weights."""
        if not probs or all(v == 0 for v in probs.values()):
            return {
                "vti_pct": 75.0,
                "vxus_pct": 25.0,
                "overweight_sectors": [],
                "avoid_sectors": [],
                "dominant_weight": 0.0,
                "dominant_type": CrisisType.UNKNOWN,
            }
        max_prob = max(probs.values())

        if max_prob < 0.40 or confidence < 0.40:
            # Low confidence → stay close to core (70% core, 30% crisis-weighted)
            core_weight = 0.70
            crisis_weight = 0.30
        else:
            # High confidence crisis → blend heavily toward crisis template
            core_weight = 0.20
            crisis_weight = 0.80

        blended_vti = sum(
            probs.get(ct, 0) * CRISIS_ALLOCATION_TEMPLATES[ct]["vti_vxus_split"][0]
            for ct in probs
        )

        default_vti = DEFAULT_CORE_ALLOCATION.get("vti_pct", 75.0)
        final_vti = core_weight * default_vti + crisis_weight * blended_vti
        final_vxus = 100.0 - final_vti

        # Aggregate sector signals weighted by probability
        overweight: dict[str, float] = {}
        avoid: dict[str, float] = {}
        for ct, p in probs.items():
            for sector in CRISIS_ALLOCATION_TEMPLATES[ct]["sector_overweight"]:
                overweight[sector] = overweight.get(sector, 0) + p
            for sector in CRISIS_ALLOCATION_TEMPLATES[ct]["sector_avoid"]:
                avoid[sector] = avoid.get(sector, 0) + p

        # Reconcile sector conflicts: same sector can't be both overweight and avoid
        # Net the weights — only include in final list based on net sign
        all_sectors = set(overweight.keys()) | set(avoid.keys())
        net_overweight = []
        net_avoid = []
        for sector in all_sectors:
            ow = overweight.get(sector, 0)
            av = avoid.get(sector, 0)
            net = ow - av
            if net >= 0.15:
                net_overweight.append(sector)
            elif net <= -0.15:
                net_avoid.append(sector)

        dominant_type = max(probs, key=probs.get) if probs else CrisisType.UNKNOWN
        return {
            "vti_pct": round(final_vti, 1),
            "vxus_pct": round(final_vxus, 1),
            "overweight_sectors": net_overweight,
            "avoid_sectors": net_avoid,
            "dominant_weight": round(max_prob, 3),
            "dominant_type": dominant_type,
        }

    def _check_flip_alert(self, reference_date: Optional[date] = None) -> bool:
        """Hard kill switch: dominant type flipped >N times in last 10 days.

        Deduplicates by date to prevent intraday reruns from inflating flips.
        Uses reference_date (observation date) to avoid stale-rerun artifacts.
        """
        window = Config.CRISIS_FINGERPRINT_FLIP_WINDOW_DAYS
        limit = Config.CRISIS_FINGERPRINT_FLIP_LIMIT

        if len(self._history) < 3:
            return False

        ref = reference_date or date.today()
        cutoff = ref - timedelta(days=window)
        # Deduplicate by date (take last entry per day)
        by_date: dict[str, dict] = {}
        for h in self._history:
            d = h.get("date", "1970-01-01")
            if d >= cutoff.isoformat():
                by_date[d] = h
        recent = [by_date[d] for d in sorted(by_date.keys())]

        if len(recent) < 3:
            return False

        flips = 0
        for i in range(1, len(recent)):
            if recent[i]["dominant"] != recent[i - 1]["dominant"]:
                flips += 1

        return flips > limit

    def _interpret(
        self, dominant: str, prob: float, confidence: float, signals: dict
    ) -> str:
        vix = signals.get("vix", 0) or 0
        spy_dd = signals.get("spy_drawdown") or 0
        hy = signals.get("hy_oas") or 0

        if confidence < 0.40:
            base = f"Low-confidence classification. Dominant: {dominant} ({prob:.0%}) — mixed signals."
        elif prob < 0.40:
            base = f"No clear dominant crisis type. {dominant} leads at {prob:.0%}."
        else:
            base = f"Dominant crisis type: {dominant} ({prob:.0%} probability)."

        details = []
        if vix > 40:
            details.append(f"VIX={vix:.0f} (extreme fear)")
        elif vix > 25:
            details.append(f"VIX={vix:.0f} (elevated)")
        if spy_dd < -0.20:
            details.append(f"SPY drawdown={spy_dd:.1%} (bear)")
        elif spy_dd < -0.10:
            details.append(f"SPY drawdown={spy_dd:.1%} (correction)")
        if hy > 8.0:
            details.append(f"HY spread={hy:.1f}% (credit crisis)")
        elif hy > 5.0:
            details.append(f"HY spread={hy:.1f}% (elevated stress)")

        if details:
            base += " Context: " + "; ".join(details) + "."
        return base

    def _load_history(self) -> list:
        if not self._history_path.exists():
            return []
        try:
            with open(self._history_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning(f"[fingerprint] Failed to load history: {e}")
            return []

    def _save_history(self) -> None:
        # Keep only last 90 entries
        self._history = self._history[-90:]
        tmp = None
        lockfile = str(self._history_path) + ".lock"
        try:
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            # Lock the TARGET resource, not the temp file
            lock_fd = os.open(lockfile, os.O_CREAT | os.O_RDWR)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                fd, tmp = tempfile.mkstemp(dir=str(self._history_path.parent), suffix=".tmp")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._history, f, indent=2, default=str)
                os.replace(tmp, str(self._history_path))
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
        except Exception as e:
            logger.warning(f"[fingerprint] Failed to save history: {e}")
            if tmp and os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Crisis State Machine
# ---------------------------------------------------------------------------

class CrisisStateManager:
    """SPY drawdown-based state machine with hysteresis.

    States: NORMAL → CORRECTION → BEAR → PANIC
    Transitions require 2-day confirmation (activation) or
    5-day confirmation (deactivation) to avoid whipsaw.
    """

    def __init__(self, state_path: Path = CRISIS_STATE_FILE):
        self._path = state_path
        self._state = self._load()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def current_state(self) -> CrisisState:
        raw = self._state.get("current_state", CrisisState.NORMAL)
        try:
            return CrisisState(raw)
        except (ValueError, KeyError):
            logger.warning(f"[crisis] Invalid persisted state '{raw}', defaulting to NORMAL")
            return CrisisState.NORMAL

    @property
    def days_in_state(self) -> int:
        return self._state.get("days_in_current_state", 0)

    def evaluate_state(
        self, spy_drawdown: float, vix: float, fg: int,
        observation_date: Optional[str] = None,
    ) -> CrisisState:
        """Compute and persist crisis state from market data.

        Primary: SPY drawdown from 52-week high
        Secondary: VIX + F&G logged as context only (not triggers)
        Hysteresis: 2 days to activate, 5 to deactivate

        Args:
            observation_date: ISO date string of the market data.
                If None, falls back to today. Using actual data date
                prevents stale-data reruns from advancing hysteresis.
        """
        # Input validation — reject non-finite or wildly out-of-range values
        import math as _math
        if not _math.isfinite(spy_drawdown) or spy_drawdown > 0.5 or spy_drawdown < -1.0:
            logger.warning(f"[crisis] Invalid spy_drawdown={spy_drawdown}, skipping state update")
            return self.current_state
        if not _math.isfinite(vix) or vix < 0:
            logger.warning(f"[crisis] Invalid vix={vix}, skipping state update")
            return self.current_state
        fg = max(0, min(100, int(fg)))  # Clamp F&G to [0, 100]

        raw_target = self._drawdown_to_state(spy_drawdown)
        current = self.current_state
        today_str = observation_date or date.today().isoformat()

        # Track unique days in state (not evaluation count)
        last_state_day = self._state.get("last_state_check_date", "")
        if today_str != last_state_day:
            self._state["days_in_current_state"] = self.days_in_state + 1
            self._state["last_state_check_date"] = today_str

        new_state = self._apply_hysteresis(current, raw_target, today_str)

        if new_state != current:
            self._transition(current, new_state, spy_drawdown, vix, fg)

        # Always record context
        self._state["last_checked"] = datetime.now(timezone.utc).isoformat()
        self._state["last_vix"] = vix
        self._state["last_fg"] = fg
        self._state["last_spy_drawdown"] = spy_drawdown
        self._save()

        return new_state

    def get_portfolio_phase(self, portfolio_value: float) -> PortfolioPhase:
        if portfolio_value < Config.PORTFOLIO_INCEPTION_THRESHOLD:
            return PortfolioPhase.INCEPTION
        elif portfolio_value < Config.PORTFOLIO_GROWTH_THRESHOLD:
            return PortfolioPhase.GROWTH
        return PortfolioPhase.ESTABLISHED

    def can_buy_stocks(self, portfolio_value: float) -> bool:
        """Return True if portfolio phase allows individual crisis stock purchases."""
        return self.get_portfolio_phase(portfolio_value) != PortfolioPhase.INCEPTION

    def get_crisis_budget(
        self, base_budget: float, reserve_balance: float, months_in_state: int = 0
    ) -> dict:
        """Calculate monthly crisis budget from base + reserve deployment.

        Returns full deployment plan for user confirmation.
        """
        state = self.current_state

        # Derive months_in_state from tracked days if caller doesn't specify
        if months_in_state <= 0:
            months_in_state = max(1, self.days_in_state // 30)

        # Budget multiplier
        mult_map = {
            CrisisState.NORMAL: Config.CRISIS_BUDGET_CORRECTION_MULT,
            CrisisState.CORRECTION: Config.CRISIS_BUDGET_CORRECTION_MULT,
            CrisisState.BEAR: Config.CRISIS_BUDGET_BEAR_MULT,
            CrisisState.PANIC: Config.CRISIS_BUDGET_PANIC_MULT,
        }
        monthly_budget = base_budget * mult_map[state]

        # Reserve deployment
        reserve_deploy = 0.0
        requires_confirmation = False
        max_deployable = reserve_balance * Config.CRISIS_RESERVE_MAX_DEPLOYMENT

        if state == CrisisState.BEAR and reserve_balance > 0:
            pct = Config.CRISIS_BEAR_MONTH1_PCT if months_in_state == 1 else Config.CRISIS_BEAR_MONTHLY_PCT
            reserve_deploy = min(reserve_balance * pct, max_deployable)
            requires_confirmation = True
        elif state == CrisisState.PANIC and reserve_balance > 0:
            pct = Config.CRISIS_PANIC_MONTH1_PCT if months_in_state == 1 else Config.CRISIS_PANIC_MONTHLY_PCT
            reserve_deploy = min(reserve_balance * pct, max_deployable)
            requires_confirmation = True

        total = monthly_budget + reserve_deploy
        weekly_tranche = total / 4

        return {
            "state": state,
            "monthly_budget": round(monthly_budget, 2),
            "reserve_deployment_this_month": round(reserve_deploy, 2),
            "total_deployable": round(total, 2),
            "weekly_tranche": round(weekly_tranche, 2),
            "requires_user_confirmation": requires_confirmation,
            "message": self._budget_message(state, total, reserve_deploy),
        }

    def get_state_summary(self) -> dict:
        """Return full state summary for dashboard/logging."""
        return {
            "current_state": self.current_state,
            "days_in_current_state": self.days_in_state,
            "last_spy_drawdown": self._state.get("last_spy_drawdown"),
            "last_vix": self._state.get("last_vix"),
            "last_fg": self._state.get("last_fg"),
            "last_checked": self._state.get("last_checked"),
            "transition_history": self._state.get("transition_history", [])[-5:],
        }

    def record_crisis_buy(
        self,
        ticker: str,
        amount: float,
        price: float,
        is_etf: bool,
        thesis: str = "",
    ) -> None:
        """Track crisis purchases for debrief and performance measurement."""
        buys = self._state.setdefault("crisis_purchases", [])
        buys.append({
            "ticker": ticker,
            "amount": amount,
            "price": price,
            "is_etf": is_etf,
            "crisis_state": str(self.current_state),
            "thesis": thesis,
            "bought_at": datetime.now(timezone.utc).isoformat(),
        })
        self._save()

    # ------------------------------------------------------------------
    # Item 3: Reserve Deployment Cumulative Tracking
    # ------------------------------------------------------------------

    def record_reserve_deployment(
        self,
        amount: float,
        reserve_balance: float,
        reason: str = "",
        as_of_date: Optional[str] = None,
    ) -> dict:
        """Record a reserve deployment and enforce cumulative 80% cap.

        Args:
            amount:          Dollar amount being deployed from reserve.
            reserve_balance: Current total reserve balance (for cap calculation).
            reason:          Free-text reason (e.g. "BEAR month-1 tranche").
            as_of_date:      Observation date for the deployment.

        Returns:
            Cap check result including allowed amount and cumulative totals.
        """
        cap_check = self.check_reserve_cap(reserve_balance, amount)

        deployments = self._state.setdefault("reserve_deployments", [])
        ref = get_as_of_date(as_of_date)
        deployments.append({
            "date": ref.isoformat(),
            "month": ref.strftime("%Y-%m"),
            "requested_amount": round(amount, 2),
            "allowed_amount": cap_check["allowed_amount"],
            "reserve_balance_at_time": round(reserve_balance, 2),
            "crisis_state": str(self.current_state),
            "reason": reason,
            "cap_check": cap_check,
        })
        self._state["reserve_deployments"] = deployments
        self._save()

        logger.info(
            f"[crisis] Reserve deployment recorded: "
            f"requested=${amount:.0f} allowed=${cap_check['allowed_amount']:.0f} "
            f"(cumulative=${cap_check['total_deployed'] + cap_check['allowed_amount']:.0f})"
        )
        return cap_check

    def get_cumulative_deployment(self) -> dict:
        """Return cumulative reserve deployment statistics from persisted state."""
        deployments = self._state.get("reserve_deployments", [])
        total = sum(d.get("allowed_amount", d.get("amount", 0)) for d in deployments)

        monthly: dict[str, float] = {}
        for d in deployments:
            m = d.get("month", "unknown")
            monthly[m] = monthly.get(m, 0) + d.get("allowed_amount", d.get("amount", 0))

        return {
            "total_deployed": round(total, 2),
            "deployment_count": len(deployments),
            "monthly_breakdown": monthly,
            "recent_deployments": deployments[-5:],
        }

    def check_reserve_cap(self, reserve_balance: float, proposed_amount: float) -> dict:
        """Check if proposed deployment would exceed cumulative 80% cap.

        Returns cap check result.  Callers should use allowed_amount not
        proposed_amount when actually deploying.
        """
        cumulative = self.get_cumulative_deployment()
        total_deployed = cumulative["total_deployed"]
        max_deployable = reserve_balance * Config.CRISIS_RESERVE_MAX_DEPLOYMENT
        remaining = max(0.0, max_deployable - total_deployed)
        allowed = min(proposed_amount, remaining)

        return {
            "allowed": remaining > 0,
            "proposed_amount": round(proposed_amount, 2),
            "allowed_amount": round(allowed, 2),
            "total_deployed": round(total_deployed, 2),
            "max_deployable": round(max_deployable, 2),
            "remaining_capacity": round(remaining, 2),
            "cap_pct": Config.CRISIS_RESERVE_MAX_DEPLOYMENT,
            "cap_exceeded": total_deployed >= max_deployable,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _drawdown_to_state(self, drawdown: float) -> CrisisState:
        if drawdown <= Config.CRISIS_SPY_PANIC_THRESHOLD:
            return CrisisState.PANIC
        elif drawdown <= Config.CRISIS_SPY_BEAR_THRESHOLD:
            return CrisisState.BEAR
        elif drawdown <= Config.CRISIS_SPY_CORRECTION_THRESHOLD:
            return CrisisState.CORRECTION
        return CrisisState.NORMAL

    def _apply_hysteresis(
        self, current: CrisisState, target: CrisisState, today_str: str = ""
    ) -> CrisisState:
        """Apply hysteresis to prevent rapid state cycling.

        Counts unique calendar days, not evaluation calls.
        """
        state_order = [
            CrisisState.NORMAL, CrisisState.CORRECTION,
            CrisisState.BEAR, CrisisState.PANIC,
        ]
        curr_idx = state_order.index(current)
        tgt_idx = state_order.index(target)

        pending = self._state.get("pending_state")
        pending_days = self._state.get("pending_state_days", 0)
        last_pending_date = self._state.get("pending_last_increment_date", "")

        if not today_str:
            today_str = date.today().isoformat()

        if target == current:
            # No change needed — reset pending
            self._state["pending_state"] = None
            self._state["pending_state_days"] = 0
            self._state["pending_last_increment_date"] = ""
            return current

        if pending == target:
            # Only increment if this is a new day
            if today_str != last_pending_date:
                new_days = pending_days + 1
                self._state["pending_last_increment_date"] = today_str
            else:
                new_days = pending_days
        else:
            pending = target
            new_days = 1
            self._state["pending_last_increment_date"] = today_str

        self._state["pending_state"] = pending
        self._state["pending_state_days"] = new_days

        # Worsening (higher stress): 2-day confirmation
        if tgt_idx > curr_idx:
            if new_days >= Config.CRISIS_ACTIVATION_DAYS:
                self._state["pending_state"] = None
                self._state["pending_state_days"] = 0
                self._state["pending_last_increment_date"] = ""
                return target
        # Improving: 5-day confirmation
        else:
            if new_days >= Config.CRISIS_DEACTIVATION_DAYS:
                self._state["pending_state"] = None
                self._state["pending_state_days"] = 0
                self._state["pending_last_increment_date"] = ""
                return target

        return current

    def _transition(
        self,
        old: CrisisState,
        new: CrisisState,
        drawdown: float,
        vix: float,
        fg: int,
    ) -> None:
        history = self._state.setdefault("transition_history", [])
        history.append({
            "from": old,
            "to": new,
            "spy_drawdown": round(drawdown, 4),
            "vix": vix,
            "fg": fg,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        # Keep last 50 transitions
        self._state["transition_history"] = history[-50:]
        self._state["current_state"] = new
        self._state["days_in_current_state"] = 0
        logger.info(
            f"[crisis] State transition: {old} → {new} "
            f"(SPY={drawdown:.1%}, VIX={vix}, F&G={fg})"
        )

    def _budget_message(self, state: CrisisState, total: float, reserve: float) -> str:
        if state in (CrisisState.NORMAL, CrisisState.CORRECTION):
            return f"Normal mode: ${total:,.0f}/month DCA. No reserve deployment."
        lines = [
            f"⚠️ Crisis mode ACTIVE ({state.upper()}): ${total:,.0f} deployable this month.",
            f"  Regular budget: ${total - reserve:,.0f} | Reserve deployment: ${reserve:,.0f}",
            "  ⚠️ Reserve deployment requires your explicit confirmation via Telegram.",
        ]
        return "\n".join(lines)

    def _load(self) -> dict:
        if not self._path.exists():
            return {"current_state": CrisisState.NORMAL, "days_in_current_state": 0}
        try:
            with open(self._path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[crisis] Failed to load state: {e}")
            return {"current_state": CrisisState.NORMAL, "days_in_current_state": 0}

    def _save(self) -> None:
        tmp = None
        lockfile = str(self._path) + ".lock"
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            lock_fd = os.open(lockfile, os.O_CREAT | os.O_RDWR)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._state, f, indent=2, default=str)
                os.replace(tmp, str(self._path))
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
        except Exception as e:
            logger.error(f"[crisis] Failed to save state: {e}")
            if tmp and os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Quality Filter (Steps 4)
# ---------------------------------------------------------------------------

class QualityFilter:
    """Crisis stock quality gate. ALL conditions must pass.

    ETFs are always exempt from this filter.
    """

    # Note: Sector exclusion applies to individual stock picks only.
    # ETF-level sector tilts (XLU, XLI etc.) in allocation templates are NOT affected.
    EXCLUDED_SECTORS = set(Config.CRISIS_EXCLUDED_SECTORS)

    @staticmethod
    def _is_etf(stock_data: dict) -> bool:
        """Detect if instrument is an ETF/fund rather than individual stock."""
        profile = stock_data.get("profile") or {}
        if profile.get("isEtf") or profile.get("isETF"):
            return True
        fund_type = profile.get("fundType") or profile.get("type") or ""
        if fund_type in ("ETF", "Etf", "Fund", "Exchange Traded Fund"):
            return True
        name = (profile.get("companyName") or "").lower()
        if "etf" in name or "exchange traded" in name:
            return True
        return False

    def check(self, stock_data: dict, as_of_date: Optional[str] = None) -> dict:
        """Run all quality checks. Returns pass/fail with per-check details.

        Uses FMP Stable API field names (post-Aug 2025):
        - marketCap (not mktCap)
        - debtToEquityRatioTTM (in ratios_ttm)
        - returnOnInvestedCapitalTTM (in key_metrics_ttm)
        - freeCashFlowPerShareTTM (in ratios_ttm) — positive = good
        - debtServiceCoverageRatioTTM (in ratios_ttm)

        Args:
            stock_data:  Collected financial data.
            as_of_date:  ISO date string for observation date.  Propagated
                         to check_ban_list() for IPO-age determinism.
        """
        # ETF exemption — ETFs skip individual stock quality gates
        if self._is_etf(stock_data):
            return {
                "passes": True,
                "failures": [],
                "details": {"asset_type": "ETF", "exempt": True},
                "summary": "✅ ETF exempt from quality filter",
            }

        failures: list[str] = []
        details: dict[str, Any] = {}

        profile = stock_data.get("profile") or {}
        ratios_ttm = stock_data.get("ratios_ttm") or {}
        key_metrics_ttm = stock_data.get("key_metrics_ttm") or {}
        cash_flow = stock_data.get("cash_flow") or []
        quote = stock_data.get("quote") or stock_data.get("yf_quote") or {}

        # 1. Market cap ≥ $10B
        # FMP Stable API: profile.marketCap (camelCase, not mktCap)
        market_cap = (
            profile.get("marketCap")
            or key_metrics_ttm.get("marketCap")
            or quote.get("market_cap")
        )
        details["market_cap"] = market_cap
        mc_val = _to_float(market_cap)
        if mc_val is None or mc_val < Config.CRISIS_QUALITY_MIN_MARKET_CAP:
            failures.append(
                f"Market cap ${(mc_val or 0)/1e9:.1f}B < $10B minimum"
            )

        # 2. Sector exclusion
        sector = canonicalize_sector(profile.get("sector") or (stock_data.get("yf_quote") or {}).get("sector") or "")
        details["sector"] = sector
        if sector in self.EXCLUDED_SECTORS:
            failures.append(f"Sector '{sector}' excluded from crisis stock picks")

        # 3. Positive free cash flow (TTM)
        # ratios_ttm.freeCashFlowPerShareTTM is per-share (positive = good)
        # key_metrics_ttm.freeCashFlowToEquityTTM is absolute value in $
        fcf_per_share = ratios_ttm.get("freeCashFlowPerShareTTM")
        fcf_absolute = key_metrics_ttm.get("freeCashFlowToEquityTTM")
        if fcf_per_share is not None:
            fcf = float(fcf_per_share)
        elif fcf_absolute is not None:
            fcf = float(fcf_absolute)
        elif cash_flow:
            # Fallback: compute from cash flow statement
            latest_cf = cash_flow[0] if isinstance(cash_flow, list) and cash_flow else {}
            op_cf = float(latest_cf.get("netCashProvidedByOperatingActivities") or 0)
            capex = float(latest_cf.get("investmentsInPropertyPlantAndEquipment") or 0)
            fcf = op_cf + capex  # capex is negative in FMP
        else:
            fcf = None
        details["fcf"] = fcf
        if fcf is None or fcf <= 0:
            failures.append(f"Free cash flow not positive (FCF={fcf})")

        # 4. Debt-to-equity ≤ 1.5
        # FMP Stable API: ratios_ttm.debtToEquityRatioTTM
        de_ratio = ratios_ttm.get("debtToEquityRatioTTM")
        details["debt_equity"] = de_ratio
        if de_ratio is not None:
            try:
                if float(de_ratio) > Config.CRISIS_QUALITY_MAX_DEBT_EQUITY:
                    failures.append(
                        f"Debt/Equity {float(de_ratio):.2f} > {Config.CRISIS_QUALITY_MAX_DEBT_EQUITY} limit"
                    )
            except (TypeError, ValueError):
                failures.append("Debt/Equity data invalid")
        else:
            failures.append("Debt/Equity data unavailable")

        # 5. ROIC ≥ 12% (prefer ROIC; ROE fallback only with low leverage)
        # FMP Stable API: key_metrics_ttm.returnOnInvestedCapitalTTM (fraction, e.g., 0.51 = 51%)
        roic_raw = key_metrics_ttm.get("returnOnInvestedCapitalTTM")
        roe_raw = key_metrics_ttm.get("returnOnEquityTTM")
        roic_metric_used = "ROIC"

        if roic_raw is not None:
            roic = roic_raw
        elif roe_raw is not None:
            # ROE is NOT equivalent to ROIC. Only accept with low leverage.
            de_val = float(de_ratio) if de_ratio is not None else 999
            if de_val <= 0.5:  # Low leverage: ROE approximates ROIC
                roic = roe_raw
                roic_metric_used = "ROE (low-leverage proxy)"
            else:
                roic = None
                failures.append(
                    f"ROIC unavailable; ROE fallback rejected (D/E={de_val:.2f} too high for proxy)"
                )
        else:
            roic = None

        details["roic"] = roic
        details["roic_metric"] = roic_metric_used
        if roic is not None:
            try:
                roic_val = float(roic)
                if roic_val < Config.CRISIS_QUALITY_MIN_ROIC:
                    failures.append(
                        f"{roic_metric_used} {roic_val:.1%} < {Config.CRISIS_QUALITY_MIN_ROIC:.0%} minimum"
                    )
            except (TypeError, ValueError):
                failures.append(f"{roic_metric_used} data invalid")
        elif "ROIC unavailable" not in " ".join(failures):
            failures.append("ROIC data unavailable")

        # 6. Interest coverage > 5x
        # interestCoverageRatioTTM is sometimes 0 (AAPL has near-zero interest expense)
        # Use debtServiceCoverageRatioTTM as more reliable proxy
        int_cov = ratios_ttm.get("debtServiceCoverageRatioTTM")
        int_cov_val = _to_float(int_cov)
        if int_cov_val is None or int_cov_val == 0:
            int_cov = ratios_ttm.get("interestCoverageRatioTTM")
        details["interest_coverage"] = int_cov
        if int_cov is not None:
            try:
                cov_val = float(int_cov)
                if cov_val == 0:
                    # Only exempt if D/E is also very low (truly no debt)
                    de_val = float(de_ratio) if de_ratio is not None else 999
                    if de_val > 0.3:
                        failures.append(
                            f"Interest coverage = 0 but D/E = {de_val:.2f} (suspicious)"
                        )
                elif cov_val < Config.CRISIS_QUALITY_MIN_INTEREST_COVERAGE:
                    failures.append(
                        f"Interest coverage {cov_val:.1f}x < {Config.CRISIS_QUALITY_MIN_INTEREST_COVERAGE}x"
                    )
            except (TypeError, ValueError):
                failures.append("Interest coverage data invalid")
        else:
            failures.append("Interest coverage data unavailable")

        # 7. Ban list check (pass as_of_date for IPO-age determinism)
        ban_result = self.check_ban_list(stock_data, as_of_date=as_of_date)
        if ban_result["banned"]:
            failures.extend(ban_result["reasons"])

        passes = len(failures) == 0
        return {
            "passes": passes,
            "failures": failures,
            "details": details,
            "summary": (
                "✅ Quality filter PASSED" if passes
                else f"❌ Quality filter FAILED ({len(failures)} issue(s)): " + "; ".join(failures[:3])
            ),
        }

    def check_ban_list(self, stock_data: dict, as_of_date: Optional[str] = None) -> dict:
        """Auto-reject criteria for crisis stock purchases.

        Args:
            stock_data:   Collected financial data for the stock.
            as_of_date:   ISO date string for observation date (uses get_as_of_date()).
                          Pass this consistently for deterministic backtests/replays.
        """
        reasons: list[str] = []
        profile = stock_data.get("profile") or {}
        quote = stock_data.get("quote") or stock_data.get("yf_quote") or {}
        income = stock_data.get("income_statement") or []
        ref = get_as_of_date(as_of_date)  # Single source of truth for date

        # 1. Unprofitable (negative TTM net income)
        if income:
            latest = income[0] if isinstance(income, list) and income else {}
            net_income = latest.get("netIncome")
            if net_income is not None and float(net_income) < 0:
                reasons.append("Unprofitable company (negative TTM net income)")

        # 2. IPO < 2 years old — uses observation date for replay determinism
        ipo_date_str = profile.get("ipoDate") or ""
        if ipo_date_str:
            try:
                ipo_date = date.fromisoformat(ipo_date_str[:10])
                age_days = (ref - ipo_date).days  # uses ref, not date.today()
                if age_days < 730:
                    reasons.append(f"IPO less than 2 years old ({age_days} days since IPO)")
            except ValueError:
                pass

        # 3. Leveraged or inverse ETF (detected by name keywords)
        name = (profile.get("companyName") or "").lower()
        if any(kw in name for kw in ("leveraged", "inverse", "ultra", "2x", "3x", "short ")):
            reasons.append(f"Leveraged/inverse ETF detected: {profile.get('companyName')}")

        # 4. Short interest > 15% (if available)
        # FMP provides shortRatio; high short ratio is a proxy
        short_ratio = quote.get("shortRatio") or profile.get("shortRatio")
        if short_ratio is not None:
            try:
                if float(short_ratio) > 10:  # Short ratio > 10 days to cover = high short
                    reasons.append(f"High short interest (short ratio {float(short_ratio):.1f} days)")
            except (TypeError, ValueError):
                pass

        return {"banned": len(reasons) > 0, "reasons": reasons}

    def check_valuation_discount(self, stock_data: dict) -> dict:
        """Check if stock trades ≥15% below 5-year median EV/FCF or P/FCF.

        NOTE: 5-year historical median requires historical ratios data.
        We use FMP annual ratios for multi-year comparison.
        """
        ratios_annual = stock_data.get("ratios") or []
        ratios_ttm = stock_data.get("ratios_ttm") or {}
        key_metrics_ttm = stock_data.get("key_metrics_ttm") or {}

        # Current multiple (prefer EV/FCF, fallback to P/FCF or P/E)
        current_ev_fcf = key_metrics_ttm.get("evToFreeCashFlowTTM")
        current_pe = ratios_ttm.get("priceToEarningsRatioTTM")

        current_multiple = None
        metric_used = None
        if current_ev_fcf is not None:
            try:
                val = float(current_ev_fcf)
                if 0 < val < 1000:  # Sanity check
                    current_multiple = val
                    metric_used = "EV/FCF"
            except (TypeError, ValueError):
                pass

        if current_multiple is None and current_pe is not None:
            try:
                val = float(current_pe)
                if 0 < val < 500:
                    current_multiple = val
                    metric_used = "P/E"
            except (TypeError, ValueError):
                pass

        if current_multiple is None:
            return {
                "discounted": False,
                "reason": "Cannot compute valuation multiple (insufficient data)",
                "current_multiple": None,
                "median_multiple": None,
                "discount_pct": None,
            }

        # Historical median from annual ratios (up to 5 years)
        # IMPORTANT: Only compare same metric to same metric (no EV/FCF vs P/FCF mixing)
        historical_multiples = []
        # FMP annual ratios field names (from /ratios endpoint)
        for r in ratios_annual[:5]:
            if metric_used == "EV/FCF":
                # If current metric is EV/FCF but historical doesn't have it,
                # fall back to P/E for BOTH current and historical (consistent comparison)
                val = r.get("enterpriseValueOverFreeCashFlowRatio")
                if val is None:
                    # No historical EV/FCF available — switch to P/E for both
                    if current_pe is not None:
                        current_multiple = float(current_pe)
                        metric_used = "P/E"
                        val = r.get("priceEarningsRatio") or r.get("priceToEarningsRatio")
                    else:
                        val = None
            else:
                val = r.get("priceEarningsRatio") or r.get("priceToEarningsRatio")
            if val is not None:
                try:
                    v = float(val)
                    if 0 < v < 1000:
                        historical_multiples.append(v)
                except (TypeError, ValueError):
                    pass

        if not historical_multiples:
            return {
                "discounted": False,
                "reason": "Insufficient historical data for valuation comparison",
                "current_multiple": round(current_multiple, 2),
                "median_multiple": None,
                "discount_pct": None,
                "metric": metric_used,
            }

        median_multiple = statistics.median(historical_multiples)
        if median_multiple <= 0:
            return {
                "discounted": False,
                "reason": "Historical median multiple ≤ 0, cannot compare",
                "current_multiple": round(current_multiple, 2),
                "median_multiple": round(median_multiple, 2),
                "discount_pct": None,
            }

        discount_pct = (median_multiple - current_multiple) / median_multiple
        required = Config.CRISIS_QUALITY_VALUATION_DISCOUNT

        return {
            "discounted": discount_pct >= required,
            "reason": (
                f"✅ {metric_used} {current_multiple:.1f}x vs 5yr median {median_multiple:.1f}x "
                f"({discount_pct:.0%} discount ≥ {required:.0%} required)"
                if discount_pct >= required
                else f"❌ {metric_used} {current_multiple:.1f}x vs 5yr median {median_multiple:.1f}x "
                     f"({discount_pct:.0%} discount < {required:.0%} required)"
            ),
            "current_multiple": round(current_multiple, 2),
            "median_multiple": round(median_multiple, 2),
            "discount_pct": round(discount_pct, 4),
            "metric": metric_used,
        }


# ---------------------------------------------------------------------------
# Value Trap Detector (Step 4)
# ---------------------------------------------------------------------------

class ValueTrapDetector:
    """Detect stocks that look cheap but are fundamentally deteriorating."""

    def check(self, stock_data: dict) -> dict:
        """Run value trap filters. Returns trap_score, flags, is_value_trap."""
        flags: list[str] = []
        details: list[dict] = []

        income = stock_data.get("income_statement") or []
        balance = stock_data.get("balance_sheet") or []

        # 1. Revenue decline >10% YoY for recent quarter
        if len(income) >= 4:
            revs = []
            for q in income[:4]:
                r = q.get("revenue") or q.get("totalRevenue")
                revs.append(float(r) if r is not None else None)

            if revs[0] is not None and revs[3] is not None and revs[3] > 0:
                q1_yoy = (revs[0] - revs[3]) / abs(revs[3])
                if q1_yoy < -0.10:
                    flags.append("revenue_decline")
                    details.append({
                        "flag": "revenue_decline",
                        "description": f"Revenue declined {q1_yoy:.1%} YoY",
                        "severity": "HIGH",
                    })

        # 2. Gross margin erosion (truly consecutive declining quarters)
        # Data assumed newest-first: margins[0]=latest, margins[1]=prior, etc.
        if len(income) >= 3:
            margins: list[float] = []
            for q in income[:4]:
                rev = _to_float(q.get("revenue") or q.get("totalRevenue"))
                cogs = _to_float(q.get("costOfRevenue"))
                if rev is not None and rev > 0 and cogs is not None:
                    margins.append((rev - cogs) / rev)
            if len(margins) >= 3:
                # Check CONSECUTIVE declines (newest first: m[0] < m[1] < m[2])
                consecutive = 0
                for i in range(len(margins) - 1):
                    if margins[i] < margins[i + 1]:
                        consecutive += 1
                    else:
                        break  # Stop at first non-decline
                if consecutive >= 2:
                    flags.append("gross_margin_erosion")
                    details.append({
                        "flag": "gross_margin_erosion",
                        "description": f"Gross margin declining {consecutive + 1} consecutive quarters",
                        "severity": "MEDIUM",
                    })

        # 3. Inventory bloat >20% YoY
        if len(balance) >= 4:
            inv_current = float(balance[0].get("inventory") or 0)
            inv_yago = float(balance[3].get("inventory") or 0)
            if inv_yago > 0 and inv_current > inv_yago * 1.20:
                inv_growth = (inv_current - inv_yago) / inv_yago
                flags.append("inventory_bloat")
                details.append({
                    "flag": "inventory_bloat",
                    "description": f"Inventory grew {inv_growth:.0%} YoY (potential demand issue)",
                    "severity": "MEDIUM",
                })

        # 4. Share count dilution >5% YoY
        if len(income) >= 4:
            shares_now = income[0].get("weightedAverageShsOut")
            shares_yago = income[3].get("weightedAverageShsOut")
            if shares_now and shares_yago and float(shares_yago) > 0:
                dilution = (float(shares_now) - float(shares_yago)) / float(shares_yago)
                if dilution > 0.05:
                    flags.append("share_dilution")
                    details.append({
                        "flag": "share_dilution",
                        "description": f"Share count diluted {dilution:.1%} YoY",
                        "severity": "MEDIUM",
                    })

        trap_score = len(flags)
        is_trap = trap_score >= 2

        return {
            "trap_score": trap_score,
            "flags": flags,
            "details": details,
            "is_value_trap": is_trap,
            "recommendation": "EXCLUDE from crisis buys" if is_trap else "PASS",
            "summary": (
                f"⚠️ Value trap risk ({trap_score}/4 flags: {', '.join(flags)})"
                if is_trap
                else f"✅ No value trap signals ({trap_score}/4 flags)"
            ),
        }


# ---------------------------------------------------------------------------
# Counterfactual Decision Engine (Step 6)
# ---------------------------------------------------------------------------

@dataclass
class CounterfactualRecord:
    """Records alternatives for every crisis buy decision.

    Outcomes are filled in by a background evaluation task
    after 30/60/90/180 days.
    """
    decision_id: str
    ticker: str
    entry_date: str
    entry_price: float
    crisis_state: str
    crisis_probs: dict
    ccs: int
    ccs_tier: str
    sector: str

    # Counterfactual baselines (prices at decision time)
    vti_price_at_entry: Optional[float]
    sector_etf_ticker: str
    sector_etf_price_at_entry: Optional[float]

    # Outcomes filled later (None = not yet evaluated)
    actual_return_30d: Optional[float] = None
    actual_return_90d: Optional[float] = None
    actual_return_180d: Optional[float] = None
    vti_return_30d: Optional[float] = None
    vti_return_90d: Optional[float] = None
    vti_return_180d: Optional[float] = None
    sector_etf_return_30d: Optional[float] = None
    sector_etf_return_90d: Optional[float] = None
    delayed_5d_price: Optional[float] = None      # Price 5 trading days after entry
    delayed_5d_return_90d: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "ticker": self.ticker,
            "entry_date": self.entry_date,
            "entry_price": self.entry_price,
            "crisis_state": self.crisis_state,
            "crisis_probs": self.crisis_probs,
            "ccs": self.ccs,
            "ccs_tier": self.ccs_tier,
            "sector": self.sector,
            "vti_price_at_entry": self.vti_price_at_entry,
            "sector_etf_ticker": self.sector_etf_ticker,
            "sector_etf_price_at_entry": self.sector_etf_price_at_entry,
            "actual_return_30d": self.actual_return_30d,
            "actual_return_90d": self.actual_return_90d,
            "actual_return_180d": self.actual_return_180d,
            "vti_return_30d": self.vti_return_30d,
            "vti_return_90d": self.vti_return_90d,
            "vti_return_180d": self.vti_return_180d,
            "sector_etf_return_30d": self.sector_etf_return_30d,
            "sector_etf_return_90d": self.sector_etf_return_90d,
            "delayed_5d_price": self.delayed_5d_price,
            "delayed_5d_return_90d": self.delayed_5d_return_90d,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CounterfactualRecord":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})  # type: ignore[attr-defined]


_SECTOR_ETF_MAP: dict[str, str] = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financial Services": "XLF",
    "Energy": "XLE",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Consumer Cyclical": "XLY",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Basic Materials": "XLB",
}

# Canonicalize sector names across data providers
_SECTOR_ALIASES: dict[str, str] = {
    "Financial": "Financial Services",
    "Financials": "Financial Services",
    "Health Care": "Healthcare",
    "Consumer Staples": "Consumer Defensive",
    "Consumer Discretionary": "Consumer Cyclical",
    "Materials": "Basic Materials",
    "Information Technology": "Technology",
    "Telecom": "Communication Services",
    "Telecommunications": "Communication Services",
    "Real Estate": "Real Estate",
}


def canonicalize_sector(sector: str) -> str:
    """Normalize sector name across different data providers."""
    if not sector:
        return ""
    sector = sector.strip()
    # Try exact match first, then title-cased version
    result = _SECTOR_ALIASES.get(sector)
    if result:
        return result
    result = _SECTOR_ALIASES.get(sector.title())
    if result:
        return result
    return sector


class CounterfactualEngine:
    """Track and evaluate counterfactual alternatives for crisis buy decisions.

    The key intellectual honesty mechanism: can prove its own obsolescence
    if VTI consistently beats council picks.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self._path = storage_path or (
            Path(__file__).resolve().parent.parent / "data" / "counterfactuals.json"
        )
        self._records: list[CounterfactualRecord] = self._load()

    def record_decision(
        self,
        ticker: str,
        entry_price: float,
        crisis_state: str,
        crisis_probs: dict,
        ccs: int,
        ccs_tier: str,
        sector: str,
        vti_price: Optional[float],
        sector_etf_prices: Optional[dict] = None,
    ) -> CounterfactualRecord:
        """Record a crisis buy with all counterfactual baselines."""
        sector = canonicalize_sector(sector)
        sector_etf = _SECTOR_ETF_MAP.get(sector, "VTI")
        sector_etf_price = (sector_etf_prices or {}).get(sector_etf)

        import uuid as _uuid
        now = datetime.now(timezone.utc)
        record = CounterfactualRecord(
            decision_id=f"{ticker}_{now.strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:8]}",
            ticker=ticker,
            entry_date=now.isoformat(),
            entry_price=entry_price,
            crisis_state=crisis_state,
            crisis_probs=crisis_probs,
            ccs=ccs,
            ccs_tier=ccs_tier,
            sector=sector,
            vti_price_at_entry=vti_price,
            sector_etf_ticker=sector_etf,
            sector_etf_price_at_entry=sector_etf_price,
        )
        self._records.append(record)
        self._save()
        vti_str = f"${vti_price:.2f}" if vti_price is not None else "N/A"
        logger.info(
            f"[counterfactual] Recorded {ticker} @ ${entry_price:.2f} "
            f"vs VTI @ {vti_str} (CCS={ccs}, {ccs_tier})"
        )
        return record

    def evaluate_now(
        self,
        current_prices: dict[str, Optional[float]],
    ) -> dict:
        """Compute VTI vs council performance for a hypothetical current decision.

        Args:
            current_prices: {ticker: price, "VTI": price, "XLK": price, ...}

        Returns expected value comparison.
        """
        vti_price = current_prices.get("VTI")
        if not vti_price:
            return {"error": "VTI price not available"}

        result = {
            "vti_price": vti_price,
            "tickers_analyzed": [],
            "recommendation": "",
        }

        for ticker, price in current_prices.items():
            if ticker in ("VTI", "VXUS") or price is None:
                continue
            result["tickers_analyzed"].append({
                "ticker": ticker,
                "current_price": price,
                "vti_alternative": vti_price,
                "note": "VTI is the baseline; stock needs substantial alpha to justify single-stock risk",
            })

        # Check historical record to see if council beats VTI
        analysis = self.evaluate_outcomes()
        if analysis.get("status") == "analyzed":
            result["historical_alpha"] = analysis.get("alpha_vs_vti")
            result["council_beats_vti"] = analysis.get("council_beats_vti")
            result["recommendation"] = analysis.get("recommendation", "")
        else:
            result["recommendation"] = (
                f"Insufficient history ({analysis.get('count', 0)} decisions). "
                "Compare each pick to VTI manually."
            )

        return result

    def evaluate_outcomes(self) -> dict:
        """Evaluate mature records (≥90d) to see if council beats VTI.

        Uses paired-sample analysis: only compares records where both
        council AND VTI returns are available.
        """
        # Paired sample: both council and VTI returns must be present
        paired = [
            r for r in self._records
            if r.actual_return_90d is not None and r.vti_return_90d is not None
        ]

        if len(paired) < 5:
            return {"status": "insufficient_data", "count": len(paired)}

        council_avg = sum(r.actual_return_90d for r in paired) / len(paired)
        vti_avg = sum(r.vti_return_90d for r in paired) / len(paired)

        council_beats_vti = council_avg > vti_avg
        alpha = council_avg - vti_avg

        ccs_high = [r for r in paired if r.ccs >= 10]
        ccs_low = [r for r in paired if r.ccs < 7]

        analysis: dict = {
            "status": "analyzed",
            "total_decisions": len(paired),
            "council_avg_return_90d": round(council_avg, 4),
            "vti_avg_return_90d": round(vti_avg, 4),
            "council_beats_vti": council_beats_vti,
            "alpha_vs_vti": round(alpha, 4),
            "ccs_high_avg": (
                round(sum(r.actual_return_90d for r in ccs_high) / len(ccs_high), 4)
                if ccs_high else None
            ),
            "ccs_low_avg": (
                round(sum(r.actual_return_90d for r in ccs_low) / len(ccs_low), 4)
                if ccs_low else None
            ),
        }

        if len(paired) >= 10 and not council_beats_vti:
            analysis["recommendation"] = (
                f"⚠️ WARNING: VTI outperformed council crisis picks by {abs(alpha):.1%} "
                f"across {len(paired)} paired decisions. Consider simplifying to VTI-only crisis mode."
            )
        elif len(paired) >= 10:
            analysis["recommendation"] = (
                f"✅ Council crisis picks generated +{alpha:.1%} alpha vs VTI "
                f"across {len(paired)} paired decisions. Stock-picking adds value."
            )

        return analysis

    def get_all_records(self) -> list[dict]:
        return [r.to_dict() for r in self._records]

    def _load(self) -> list[CounterfactualRecord]:
        if not self._path.exists():
            return []
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            return [CounterfactualRecord.from_dict(d) for d in data if isinstance(d, dict)]
        except Exception as e:
            logger.warning(f"[counterfactual] Failed to load: {e}")
            return []

    def _save(self) -> None:
        tmp = None
        lockfile = str(self._path) + ".lock"
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            lock_fd = os.open(lockfile, os.O_CREAT | os.O_RDWR)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump([r.to_dict() for r in self._records], f, indent=2, default=str)
                os.replace(tmp, str(self._path))
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
        except Exception as e:
            logger.error(f"[counterfactual] Failed to save: {e}")
            if tmp and os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Smart Money Tracker (Step 10)
# ---------------------------------------------------------------------------

class SmartMoneyTracker:
    """Insider cluster detection during crisis periods.

    Focuses on open-market purchases only. Cluster = 3+ unique insiders
    buying within 14 days. Single insider buys are weak signals; clusters
    during drawdowns are historically strong.
    """

    @staticmethod
    def _is_within_days(date_str: Optional[str], days: int, as_of: Optional[date] = None) -> bool:
        """Check if date_str is within `days` of as_of (default: today).
        
        Guards against future-dated filings (requires non-negative delta).
        """
        if not date_str:
            return False
        try:
            d = date.fromisoformat(str(date_str)[:10])
            ref = as_of or date.today()
            delta = (ref - d).days
            return 0 <= delta <= days  # Non-negative: rejects future dates
        except ValueError:
            return False

    def aggregate_insider_signal(
        self, insider_data_list: list[Optional[dict]]
    ) -> dict:
        """Compute aggregate insider buying signal.

        Args:
            insider_data_list: List of Finnhub insider_transactions() results,
                               one per ticker being watched.
        """
        total_buys = 0
        total_sells = 0
        buy_value = 0.0
        sell_value = 0.0
        cluster_tickers: list[str] = []

        for data in insider_data_list:
            if not isinstance(data, dict):
                continue
            ticker = data.get("symbol", "UNKNOWN")
            transactions = data.get("data") or []
            if not isinstance(transactions, list):
                continue

            recent = [
                t for t in transactions
                if self._is_within_days(t.get("filingDate"), 30)
            ]

            def _safe_float(val: Any, default: float = 0.0) -> float:
                """Safely coerce value to float."""
                if val is None:
                    return default
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return default

            # Open-market purchases only (code "P")
            buys = [
                t for t in recent
                if t.get("transactionCode") == "P"
                and _safe_float(t.get("change")) > 0
            ]
            sells = [
                t for t in recent
                if t.get("transactionCode") in ("S", "S-Sale")
                and _safe_float(t.get("change")) < 0
            ]

            total_buys += len(buys)
            total_sells += len(sells)
            buy_value += sum(
                abs(_safe_float(t.get("change"))) * abs(_safe_float(t.get("price")))
                for t in buys
            )
            sell_value += sum(
                abs(_safe_float(t.get("change"))) * abs(_safe_float(t.get("price")))
                for t in sells
            )

            # Cluster detection: 3+ unique insiders buying within 14 days
            recent_14d_buyers = {
                t.get("name", f"unknown_{i}")
                for i, t in enumerate(buys)
                if self._is_within_days(t.get("filingDate"), 14)
            }
            if len(recent_14d_buyers) >= 3:
                cluster_tickers.append(ticker)
                logger.info(
                    f"[smart_money] Insider cluster for {ticker}: "
                    f"{len(recent_14d_buyers)} unique buyers in 14 days"
                )

        total_value = buy_value + sell_value
        net_buy_ratio = (buy_value - sell_value) / max(total_value, 1)

        strength = (
            "STRONG" if net_buy_ratio > 0.3 and len(cluster_tickers) >= 2
            else "MODERATE" if net_buy_ratio > 0.1
            else "NEUTRAL" if abs(net_buy_ratio) <= 0.1
            else "BEARISH"
        )

        interpretation = (
            "🟢 STRONG BOTTOM SIGNAL: Multiple insider clusters buying"
            if strength == "STRONG"
            else "🟡 MODERATE: Net insider buying positive"
            if strength == "MODERATE"
            else "⚪ NEUTRAL: Mixed insider activity"
            if strength == "NEUTRAL"
            else "🔴 CAUTION: Net insider selling — insiders not convinced"
        )

        return {
            "total_buys": total_buys,
            "total_sells": total_sells,
            "buy_value_usd": round(buy_value, 0),
            "sell_value_usd": round(sell_value, 0),
            "net_buy_ratio": round(net_buy_ratio, 3),
            "cluster_tickers": cluster_tickers,
            "signal_strength": strength,
            "interpretation": interpretation,
        }


# ---------------------------------------------------------------------------
# Item 5: Historical Backtesting Framework
# ---------------------------------------------------------------------------

# Pre-defined signal snapshots at peak stress of historical crises.
# Values sourced from public market data; used for classification validation.
HISTORICAL_CRISIS_SNAPSHOTS: dict[str, dict] = {
    "2008_q4_peak": {
        "name": "2008 Global Financial Crisis — Oct 10 2008 peak stress",
        "date": "2008-10-10",
        "expected_classification": CrisisType.BANKING,
        "signals": {
            "vix": 69.9,
            "spy_drawdown": -0.47,
            "hy_oas": 18.5,
            "ig_oas": 4.5,
            "yield_curve_spread": -0.2,
            "oil_price": 77.0,
            "dxy": 84.0,
            "fed_funds_rate": 1.5,
            "unemployment_rate": 6.5,
            "initial_jobless_claims": 480_000,
            "fear_greed": 4,
        },
    },
    "2020_covid_mar": {
        "name": "COVID-19 Crash — March 20 2020",
        "date": "2020-03-20",
        "expected_classification": CrisisType.PANDEMIC,
        "signals": {
            "vix": 82.7,
            "spy_drawdown": -0.34,
            "hy_oas": 10.5,
            "ig_oas": 3.5,
            "yield_curve_spread": 0.4,
            "oil_price": 22.0,
            "dxy": 102.0,
            "fed_funds_rate": 0.25,
            "unemployment_rate": 4.4,
            "initial_jobless_claims": 3_283_000,
            "fear_greed": 5,
        },
    },
    "2022_stagflation": {
        "name": "2022 Rate Hike Bear — Oct 12 2022 bottom",
        "date": "2022-10-12",
        "expected_classification": CrisisType.STAGFLATION,
        "signals": {
            "vix": 33.0,
            "spy_drawdown": -0.25,
            "hy_oas": 5.5,
            "ig_oas": 1.8,
            "yield_curve_spread": -0.5,
            "oil_price": 89.0,
            "dxy": 113.0,
            "fed_funds_rate": 3.25,
            "unemployment_rate": 3.5,
            "initial_jobless_claims": 228_000,
            "fear_greed": 18,
        },
    },
    "2022_tech_selloff": {
        "name": "2022 Tech/Growth Selloff — Jan 28 2022",
        "date": "2022-01-28",
        "expected_classification": CrisisType.TECH_BUBBLE,
        "signals": {
            "vix": 31.0,
            "spy_drawdown": -0.10,
            "hy_oas": 3.5,
            "ig_oas": 1.0,
            "yield_curve_spread": 0.8,
            "oil_price": 83.0,
            "dxy": 96.0,
            "fed_funds_rate": 0.25,
            "unemployment_rate": 4.0,
            "initial_jobless_claims": 260_000,
            "fear_greed": 25,
        },
    },
}


class CrisisBacktester:
    """Replay historical crisis periods to validate fingerprinting accuracy.

    Uses pre-defined market signal snapshots from peak stress periods.
    Tests whether CrisisFingerprint would have classified each crisis correctly.
    Runs in an isolated temp file to avoid contaminating real fingerprint history.
    """

    def run_all(self) -> dict:
        """Run all historical scenario backtests. Returns summary + details."""
        results = []
        for scenario_id, scenario in HISTORICAL_CRISIS_SNAPSHOTS.items():
            result = self.run_scenario(scenario_id, scenario)
            results.append(result)

        correct = sum(1 for r in results if r["classified_correctly"])
        total = len(results)
        accuracy = correct / max(total, 1)

        return {
            "total_scenarios": total,
            "correctly_classified": correct,
            "accuracy": round(accuracy, 3),
            "grade": "A" if correct == total else "B" if correct >= total * 0.75 else "C",
            "note": (
                "All historical crises correctly classified"
                if correct == total
                else f"{correct}/{total} classified correctly — review signal thresholds"
            ),
            "scenarios": results,
        }

    def run_scenario(self, scenario_id: str, scenario: dict) -> dict:
        """Run a single backtest scenario against a fresh (isolated) fingerprint."""
        import tempfile
        from pathlib import Path as _Path

        signals = {**scenario["signals"], "collected_at": scenario["date"]}
        expected = scenario["expected_classification"]

        # Isolated temp path so backtest doesn't contaminate production history
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = _Path(tmp.name)
        try:
            fp = CrisisFingerprint(history_path=tmp_path)
            result = fp.classify(signals, observation_date=scenario["date"])
        finally:
            tmp_path.unlink(missing_ok=True)

        dominant = result["dominant"]
        classified_correctly = dominant == expected

        return {
            "scenario_id": scenario_id,
            "name": scenario["name"],
            "date": scenario["date"],
            "expected": expected,
            "dominant": dominant,
            "dominant_prob": result["dominant_prob"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "classified_correctly": classified_correctly,
            "verdict": (
                f"✅ Correctly classified as {dominant} ({result['dominant_prob']:.0%} prob)"
                if classified_correctly
                else f"❌ Expected {expected}, got {dominant} ({result['dominant_prob']:.0%} prob)"
            ),
        }


# ---------------------------------------------------------------------------
# Item 6: Counterfactual Outcome Evaluator (extends CounterfactualEngine)
# ---------------------------------------------------------------------------
# The evaluate_and_update_mature_records() method is added directly onto
# CounterfactualEngine here via monkey-patching to keep the class cohesive.

def _counterfactual_evaluate_and_update(
    self: "CounterfactualEngine",
    as_of_date: Optional[str] = None,
    price_fetcher=None,
) -> dict:
    """Evaluate mature records and fill in actual + VTI returns.

    Fetches current prices via yfinance (default) or a custom price_fetcher.
    Only fills returns that haven't been computed yet (idempotent).

    Args:
        as_of_date:    ISO date string for evaluation date.
        price_fetcher: Optional callable(tickers: list[str]) -> {ticker: float|None}.
                       If None, uses yfinance.

    Returns:
        Summary of how many records were updated at each horizon.
    """
    ref = get_as_of_date(as_of_date)

    if not self._records:
        return {"status": "no_records", "updated_30d": 0, "updated_90d": 0, "updated_180d": 0}

    # Collect all tickers we need prices for
    tickers_needed: set[str] = {"VTI"}
    for rec in self._records:
        tickers_needed.add(rec.ticker)
        if rec.sector_etf_ticker:
            tickers_needed.add(rec.sector_etf_ticker)

    # Fetch current prices
    current_prices: dict[str, Optional[float]] = {}
    try:
        if price_fetcher:
            current_prices = price_fetcher(list(tickers_needed)) or {}
        else:
            import yfinance as yf
            data = yf.download(
                list(tickers_needed), period="5d", progress=False, auto_adjust=True
            )
            if data.empty:
                logger.warning("[counterfactual] yfinance returned empty data")
            else:
                close = data.get("Close", data)
                if hasattr(close, "columns"):
                    # Multi-ticker
                    for t in tickers_needed:
                        try:
                            col_data = close[t].dropna()
                            if not col_data.empty:
                                current_prices[t] = float(col_data.iloc[-1])
                        except (KeyError, TypeError):
                            pass
                else:
                    # Single ticker
                    t = list(tickers_needed)[0]
                    col_data = close.dropna()
                    if not col_data.empty:
                        current_prices[t] = float(col_data.iloc[-1])
    except Exception as e:
        logger.warning(f"[counterfactual] Price fetch failed: {e}")

    updated_30d = updated_90d = updated_180d = skipped = 0

    def _ret(entry: Optional[float], current: Optional[float]) -> Optional[float]:
        if entry and current and entry > 0:
            return round((current - entry) / entry, 4)
        return None

    for record in self._records:
        try:
            entry_date = date.fromisoformat(record.entry_date[:10])
        except ValueError:
            skipped += 1
            continue

        days_held = (ref - entry_date).days
        price_now = current_prices.get(record.ticker)
        vti_now = current_prices.get("VTI")
        etf_now = current_prices.get(record.sector_etf_ticker)

        if price_now is None:
            skipped += 1
            continue

        if days_held >= 30 and record.actual_return_30d is None:
            record.actual_return_30d = _ret(record.entry_price, price_now)
            record.vti_return_30d = _ret(record.vti_price_at_entry, vti_now)
            if record.sector_etf_price_at_entry:
                record.sector_etf_return_30d = _ret(record.sector_etf_price_at_entry, etf_now)
            updated_30d += 1

        if days_held >= 90 and record.actual_return_90d is None:
            record.actual_return_90d = _ret(record.entry_price, price_now)
            record.vti_return_90d = _ret(record.vti_price_at_entry, vti_now)
            updated_90d += 1

        if days_held >= 180 and record.actual_return_180d is None:
            record.actual_return_180d = _ret(record.entry_price, price_now)
            record.vti_return_180d = _ret(record.vti_price_at_entry, vti_now)
            updated_180d += 1

    total_updated = updated_30d + updated_90d + updated_180d
    if total_updated > 0:
        self._save()
        logger.info(
            f"[counterfactual] Updated outcomes: 30d={updated_30d} 90d={updated_90d} "
            f"180d={updated_180d} (skipped={skipped})"
        )

    return {
        "status": "done",
        "updated_30d": updated_30d,
        "updated_90d": updated_90d,
        "updated_180d": updated_180d,
        "total_updated": total_updated,
        "skipped": skipped,
        "prices_fetched": len(current_prices),
    }


# Attach to CounterfactualEngine
CounterfactualEngine.evaluate_and_update_mature_records = _counterfactual_evaluate_and_update  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Item 7: Learning Loop
# ---------------------------------------------------------------------------

class CrisisLearningEngine:
    """Adjust crisis system thresholds based on observed outcomes.

    Conservative approach:
    - Requires ≥10 paired (90d) observations before any recommendation.
    - Maximum suggested change: ±20% of current threshold value.
    - Stores advisory recommendations only — never auto-applies to Config.
    - Persists to data/crisis_learning.json.
    """

    MIN_OBSERVATIONS: int = 10
    MAX_ADJUST_PCT: float = 0.20  # ±20% max change recommendation

    def __init__(self, learning_path: Optional[Path] = None):
        self._path = learning_path or (
            Path(__file__).resolve().parent.parent / "data" / "crisis_learning.json"
        )
        self._data = self._load()

    def analyze(self, counterfactual_records: list[dict], as_of_date: Optional[str] = None) -> dict:
        """Analyze 90-day outcomes and produce threshold recommendations.

        Args:
            counterfactual_records: Output of CounterfactualEngine.get_all_records().
            as_of_date:             Observation date for the analysis.

        Returns:
            Analysis dict with status, metrics, and advisory recommendations.
        """
        mature = [
            r for r in counterfactual_records
            if r.get("actual_return_90d") is not None
            and r.get("vti_return_90d") is not None
        ]

        if len(mature) < self.MIN_OBSERVATIONS:
            return {
                "status": "insufficient_data",
                "observations": len(mature),
                "needed": self.MIN_OBSERVATIONS,
                "recommendations": [],
                "message": (
                    f"Need {self.MIN_OBSERVATIONS - len(mature)} more mature records "
                    f"before learning loop activates."
                ),
            }

        # Segment by CCS tier
        high_ccs = [r for r in mature if r.get("ccs", 0) >= Config.CRISIS_CCS_HIGH_CONVICTION]
        std_ccs  = [r for r in mature if Config.CRISIS_CCS_STANDARD <= r.get("ccs", 0) < Config.CRISIS_CCS_HIGH_CONVICTION]
        low_ccs  = [r for r in mature if r.get("ccs", 0) < Config.CRISIS_CCS_STANDARD]

        def _avg_alpha(records: list[dict]) -> Optional[float]:
            if len(records) < 3:
                return None
            return statistics.mean(
                r["actual_return_90d"] - r["vti_return_90d"] for r in records
            )

        high_alpha = _avg_alpha(high_ccs)
        std_alpha  = _avg_alpha(std_ccs)
        low_alpha  = _avg_alpha(low_ccs)
        overall_alpha = statistics.mean(
            r["actual_return_90d"] - r["vti_return_90d"] for r in mature
        )

        recommendations: list[dict] = []

        # CCS minimum threshold: if low-CCS picks consistently underperform, raise the floor
        if high_alpha is not None and low_alpha is not None:
            if high_alpha > 0.02 and low_alpha < -0.01:
                new_val = min(
                    Config.CRISIS_CCS_STANDARD + 1,
                    int(Config.CRISIS_CCS_STANDARD * (1 + self.MAX_ADJUST_PCT)),
                )
                recommendations.append({
                    "parameter": "CRISIS_CCS_STANDARD",
                    "current_value": Config.CRISIS_CCS_STANDARD,
                    "suggested_value": new_val,
                    "rationale": (
                        f"High-CCS picks: +{high_alpha:.1%} alpha (n={len(high_ccs)}); "
                        f"Low-CCS picks: {low_alpha:.1%} alpha (n={len(low_ccs)}). "
                        "Raising minimum CCS threshold reduces low-quality picks."
                    ),
                    "confidence": "moderate" if len(mature) < 20 else "high",
                    "advisory_only": True,
                })

        # Overall: if council consistently underperforms VTI, flag for review
        if overall_alpha < -0.03 and len(mature) >= 15:
            recommendations.append({
                "parameter": "OVERALL_STRATEGY",
                "current_value": "stock-picking in crisis",
                "suggested_value": "VTI-only crisis mode",
                "rationale": (
                    f"Council underperformed VTI by {abs(overall_alpha):.1%} on average "
                    f"across {len(mature)} decisions. Consider pure ETF crisis allocation."
                ),
                "confidence": "moderate",
                "advisory_only": True,
            })

        analysis = {
            "status": "analyzed",
            "as_of_date": get_as_of_date(as_of_date).isoformat(),
            "observations": len(mature),
            "overall_alpha_90d": round(overall_alpha, 4),
            "council_beats_vti": overall_alpha > 0,
            "segment_alpha": {
                "high_ccs": round(high_alpha, 4) if high_alpha is not None else None,
                "std_ccs": round(std_alpha, 4) if std_alpha is not None else None,
                "low_ccs": round(low_alpha, 4) if low_alpha is not None else None,
            },
            "recommendations": recommendations,
            "note": (
                "All recommendations are advisory only. "
                "Apply changes manually after review."
            ),
        }

        # Persist
        self._data.setdefault("analyses", []).append({
            "date": get_as_of_date(as_of_date).isoformat(),
            "result": analysis,
        })
        self._data["analyses"] = self._data["analyses"][-24:]  # Keep 2 years of monthly
        self._save()

        return analysis

    def get_latest_recommendations(self) -> list[dict]:
        """Return recommendations from the most recent analysis."""
        analyses = self._data.get("analyses", [])
        if not analyses:
            return []
        return analyses[-1].get("result", {}).get("recommendations", [])

    def _load(self) -> dict:
        if not self._path.exists():
            return {"analyses": [], "applied_adjustments": []}
        try:
            with open(self._path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[learning] Failed to load: {e}")
            return {"analyses": [], "applied_adjustments": []}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, default=str)
            os.replace(tmp, str(self._path))
        except Exception as e:
            logger.warning(f"[learning] Failed to save: {e}")


# ---------------------------------------------------------------------------
# Additional Hardening: Shadow Mode
# ---------------------------------------------------------------------------

class ShadowCrisisAnalyzer:
    """Run crisis analysis in shadow mode — log what WOULD be recommended.

    Shadow mode runs the full crisis system alongside the live system
    without executing any actual trades or recommendations.  This enables
    safe validation before expanding crisis-mode scope.

    All shadow runs are appended to data/shadow_log.jsonl (newline-JSON).
    """

    SHADOW_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "shadow_log.jsonl"

    def analyze(
        self,
        signals: dict,
        portfolio_value: float,
        as_of_date: Optional[str] = None,
        note: str = "",
    ) -> dict:
        """Run full crisis check in shadow mode and log the recommendation.

        Returns the same result dict as run_full_crisis_check(), with an
        additional 'shadow_mode': True marker.
        """
        result = run_full_crisis_check(signals, portfolio_value, as_of_date=as_of_date)

        entry = {
            "shadow": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "as_of_date": get_as_of_date(as_of_date).isoformat(),
            "note": note,
            "crisis_state": str(result.get("crisis_state", "")),
            "fingerprint_dominant": str(result.get("fingerprint", {}).get("dominant", "")),
            "fingerprint_confidence": result.get("fingerprint", {}).get("confidence"),
            "portfolio_phase": str(result.get("portfolio_phase", "")),
            "budget_total": result.get("budget", {}).get("total_deployable"),
            "can_buy_stocks": result.get("can_buy_stocks"),
            "portfolio_value": portfolio_value,
            "staleness": result.get("staleness"),
        }
        self._log(entry)

        logger.info(
            f"[shadow] SHADOW RUN: state={entry['crisis_state']} "
            f"dominant={entry['fingerprint_dominant']} "
            f"budget=${entry['budget_total']:.0f} note={note!r}"
        )
        return {**result, "shadow_mode": True, "shadow_logged": True}

    def get_shadow_log(self, limit: int = 20) -> list[dict]:
        """Return recent shadow log entries."""
        if not self.SHADOW_LOG_PATH.exists():
            return []
        entries: list[dict] = []
        try:
            with open(self.SHADOW_LOG_PATH, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            logger.warning(f"[shadow] Failed to read log: {e}")
        return entries[-limit:]

    def _log(self, entry: dict) -> None:
        try:
            self.SHADOW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(self.SHADOW_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.warning(f"[shadow] Failed to write shadow log: {e}")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_full_crisis_check(
    signals: dict,
    portfolio_value: float = 0.0,
    as_of_date: Optional[str] = None,
    reserve_balance: float = 0.0,
) -> dict:
    """Run complete crisis check: fingerprint + state evaluation + budget.

    Args:
        signals:          Output from DataCollector.collect_crisis_signals().
        portfolio_value:  Current portfolio value for phase determination.
        as_of_date:       Override observation date (ISO string).  If None,
                          uses 'collected_at' from signals or today.
        reserve_balance:  Current reserve balance for budget calculation.

    Returns comprehensive crisis assessment dict.  Individual component
    failures degrade gracefully (reduced confidence) rather than crashing.
    """
    warnings_list: list[str] = []

    # -----------------------------------------------------------------------
    # 1. Resolve canonical observation date (Item 1)
    # -----------------------------------------------------------------------
    collected_at = signals.get("collected_at")
    if as_of_date:
        obs_date_str = str(as_of_date)[:10]
    elif isinstance(collected_at, str):
        obs_date_str = collected_at[:10]
    else:
        obs_date_str = None

    obs_date = get_as_of_date(obs_date_str)

    # -----------------------------------------------------------------------
    # 2. Weekend / holiday staleness check (Item 1 + hardening)
    # -----------------------------------------------------------------------
    stale_market = is_market_stale_day(obs_date)
    if stale_market:
        warnings_list.append(
            f"Market data may be stale — {obs_date} is a weekend or holiday"
        )
        logger.info(f"[crisis] Stale market day detected: {obs_date} — reducing confidence")

    # -----------------------------------------------------------------------
    # 3. Signal freshness weights (Item 4)
    # -----------------------------------------------------------------------
    staleness_report: dict = {}
    try:
        staleness_report = SignalFreshnessModel.get_staleness_report(
            signals, as_of=obs_date, collected_at=str(collected_at) if collected_at else None
        )
        if staleness_report.get("stale_signals"):
            warnings_list.append(
                f"Stale signals: {staleness_report['stale_signals']}"
            )
            logger.debug(f"[crisis] Staleness: {staleness_report['summary']}")
    except Exception as e:
        logger.warning(f"[crisis] Staleness check failed (continuing): {e}")
        warnings_list.append(f"Staleness check unavailable: {e}")

    # -----------------------------------------------------------------------
    # 4. Fingerprint (graceful degradation on failure)
    # -----------------------------------------------------------------------
    fp: dict = {}
    try:
        fingerprint_engine = CrisisFingerprint()
        fp = fingerprint_engine.classify(signals, observation_date=obs_date_str)
    except Exception as e:
        logger.error(f"[crisis] Fingerprint failed: {e}")
        fp = {
            "dominant": CrisisType.UNKNOWN,
            "confidence": 0.0,
            "error": str(e),
        }
        warnings_list.append(f"Fingerprint engine error: {e}")

    # -----------------------------------------------------------------------
    # 5. State machine (graceful degradation)
    # -----------------------------------------------------------------------
    crisis_state = CrisisState.NORMAL
    state_summary: dict = {}
    try:
        state_manager = CrisisStateManager()
        spy_dd = _safe_float(signals.get("spy_drawdown"), 0.0)
        vix = _safe_float(signals.get("vix"), 0.0)
        fg = _safe_int(signals.get("fear_greed") or signals.get("fg"), 50)
        crisis_state = state_manager.evaluate_state(
            spy_dd, vix, fg, observation_date=obs_date_str
        )
        state_summary = state_manager.get_state_summary()
    except Exception as e:
        logger.error(f"[crisis] State machine failed: {e}")
        warnings_list.append(f"State machine error: {e}")

    # -----------------------------------------------------------------------
    # 6. Portfolio phase + budget
    # -----------------------------------------------------------------------
    phase = PortfolioPhase.INCEPTION
    budget: dict = {}
    try:
        phase = state_manager.get_portfolio_phase(portfolio_value)
        budget = state_manager.get_crisis_budget(
            base_budget=Config.MONTHLY_BUDGET,
            reserve_balance=reserve_balance,
        )
    except Exception as e:
        logger.error(f"[crisis] Budget calculation failed: {e}")
        warnings_list.append(f"Budget calculation error: {e}")
        budget = {
            "state": crisis_state,
            "monthly_budget": Config.MONTHLY_BUDGET,
            "total_deployable": Config.MONTHLY_BUDGET,
            "error": str(e),
        }

    return {
        "crisis_state": crisis_state,
        "portfolio_phase": phase,
        "fingerprint": fp,
        "budget": budget,
        "state_summary": state_summary,
        "can_buy_stocks": state_manager.can_buy_stocks(portfolio_value) if state_summary else False,
        "staleness": staleness_report,
        "stale_market_day": stale_market,
        "as_of_date": obs_date.isoformat(),
        "warnings": warnings_list,
    }
