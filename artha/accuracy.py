"""Pick accuracy tracker — grades council recommendations after 30 days.

Records every council verdict with entry price, then auto-evaluates
after 30 calendar days. Feeds results into self-review for prompt tuning.

Persistence: data/accuracy.json with advisory locking.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Optional

from .config import Config
from .paths import DATA_DIR


logger = logging.getLogger(__name__)

try:
    import fcntl
except ImportError:
    fcntl = None

CENTS = Decimal("0.01")
ACCURACY_FILE = DATA_DIR / "accuracy.json"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Benchmark-relative grading tunables (Wave 2 consolidates into Config).
ACCURACY_BENCHMARK_TICKER = os.getenv("ARTHA_ACCURACY_BENCHMARK_TICKER", "SPY")
ACCURACY_REVIEW_DAYS = int(_env_float("ARTHA_ACCURACY_REVIEW_DAYS", 30))
# No-buy verdicts grade INCORRECT when the name beats the benchmark by more
# than this many percentage points over the review window; PARTIALLY_CORRECT
# between the partial and incorrect gates. Buy verdicts mirror on the downside.
ACCURACY_EXCESS_INCORRECT_PCT = _env_float("ARTHA_ACCURACY_EXCESS_INCORRECT_PCT", 10.0)
ACCURACY_EXCESS_PARTIAL_PCT = _env_float("ARTHA_ACCURACY_EXCESS_PARTIAL_PCT", 5.0)
CURRENT_ANALYST_LABELS = {
    "fundamental": "Fundamental (GPT agentic)",
    "technical": "Technical (Gemini agentic)",
    "contrarian": "Contrarian/Risk (GPT agentic)",
}
LEGACY_ANALYST_LABELS = {
    "fundamental": "Fundamental (Opus)",
    "technical": "Technical (Gemini)",
    "contrarian": "Contrarian (GPT 5.4)",
}
BUY_SIDE_VERDICTS = {"STRONG BUY", "BUY", "STARTER", "TACTICAL BUY", "ACCUMULATE", "ADD"}
AVOID_SIDE_VERDICTS = {"AVOID", "STRONG SELL", "SELL"}
WAIT_SIDE_VERDICTS = {"WATCH", "DEFER", "HOLD"}
TRIM_SIDE_VERDICTS = {"TRIM", "REDUCE"}
# Benchmark-relative grading groups verdicts by whether they keep exposure:
# HOLD keeps the position, so it wins with the name; WATCH/DEFER/AVOID/SELL
# all mean "no exposure", so they are wrong when the name beats the benchmark.
EXPOSURE_VERDICTS = BUY_SIDE_VERDICTS | {"HOLD"}
NO_EXPOSURE_VERDICTS = {"WATCH", "DEFER", "AVOID", "SELL", "STRONG SELL"} | TRIM_SIDE_VERDICTS


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: object) -> Optional[datetime]:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _current_era_start() -> datetime:
    parsed = _parse_dt(Config.ACCURACY_CURRENT_ERA_START)
    return parsed or datetime(2026, 6, 2, tzinfo=timezone.utc)


def _record_timestamp(rec: dict) -> Optional[datetime]:
    return _parse_dt(rec.get("timestamp"))


def _is_current_era(rec: dict) -> bool:
    ts = _record_timestamp(rec)
    return bool(ts and ts >= _current_era_start())


def _analyst_labels_for_record(rec: dict) -> dict[str, str]:
    labels = rec.get("analyst_labels")
    if isinstance(labels, dict) and all(k in labels for k in CURRENT_ANALYST_LABELS):
        return {k: str(labels[k]) for k in CURRENT_ANALYST_LABELS}
    return CURRENT_ANALYST_LABELS if _is_current_era(rec) else LEGACY_ANALYST_LABELS


def _normalize_verdict(verdict: str) -> str:
    return str(verdict or "").upper().replace("_", " ").strip()


def _grade_verdict(verdict: str, change_pct: Decimal) -> str:
    """Grade modern Artha verdict families against subsequent price movement."""
    normalized = _normalize_verdict(verdict)
    if normalized in BUY_SIDE_VERDICTS:
        if change_pct >= 5:
            return "CORRECT"
        if change_pct >= 0:
            return "PARTIALLY_CORRECT"
        return "INCORRECT"
    if normalized in AVOID_SIDE_VERDICTS:
        if change_pct <= 0:
            return "CORRECT"
        if change_pct <= 5:
            return "PARTIALLY_CORRECT"
        return "INCORRECT"
    if normalized in WAIT_SIDE_VERDICTS:
        if abs(change_pct) <= 10:
            return "CORRECT"
        if change_pct > 10:
            return "PARTIALLY_CORRECT"
        return "CORRECT"
    if normalized in TRIM_SIDE_VERDICTS:
        if change_pct <= -2:
            return "CORRECT"
        if change_pct <= 3:
            return "PARTIALLY_CORRECT"
        return "INCORRECT"
    return "UNGRADED"


def _grade_analyst_verdict(verdict: str, change_pct: Decimal) -> str:
    normalized = _normalize_verdict(verdict)
    if normalized in BUY_SIDE_VERDICTS:
        if change_pct >= 5:
            return "CORRECT"
        if change_pct >= 0:
            return "PARTIALLY_CORRECT"
        return "INCORRECT"
    if normalized in AVOID_SIDE_VERDICTS:
        if change_pct <= 0:
            return "CORRECT"
        if change_pct <= 5:
            return "PARTIALLY_CORRECT"
        return "INCORRECT"
    if normalized in WAIT_SIDE_VERDICTS:
        if abs(change_pct) <= 10:
            return "CORRECT"
        return "PARTIALLY_CORRECT"
    if normalized in TRIM_SIDE_VERDICTS:
        if change_pct <= -2:
            return "CORRECT"
        if change_pct <= 3:
            return "PARTIALLY_CORRECT"
        return "INCORRECT"
    return "UNGRADED"


def _grade_verdict_excess(verdict: str, excess_pct: Decimal) -> str:
    """Grade a verdict on benchmark-relative excess return (percentage points).

    No-exposure verdicts (DEFER/WATCH/AVOID/SELL/TRIM) are INCORRECT when the
    name beat the benchmark by more than ACCURACY_EXCESS_INCORRECT_PCT,
    PARTIALLY_CORRECT between the partial and incorrect gates, and CORRECT
    when the name matched or lagged the benchmark. Exposure verdicts
    (BUY/STARTER/TACTICAL_BUY/ACCUMULATE/HOLD) are graded symmetrically.
    Timid calls can now be wrong: a +30% missed winner grades INCORRECT.
    """
    normalized = _normalize_verdict(verdict)
    incorrect_gate = Decimal(str(ACCURACY_EXCESS_INCORRECT_PCT))
    partial_gate = Decimal(str(ACCURACY_EXCESS_PARTIAL_PCT))
    if normalized in EXPOSURE_VERDICTS:
        if excess_pct <= -incorrect_gate:
            return "INCORRECT"
        if excess_pct <= -partial_gate:
            return "PARTIALLY_CORRECT"
        return "CORRECT"
    if normalized in NO_EXPOSURE_VERDICTS:
        if excess_pct >= incorrect_gate:
            return "INCORRECT"
        if excess_pct >= partial_gate:
            return "PARTIALLY_CORRECT"
        return "CORRECT"
    return "UNGRADED"


def _directional_outcome(verdict: str, excess_pct: Decimal) -> Optional[int]:
    """1 when the verdict was directionally right vs the benchmark, else 0."""
    normalized = _normalize_verdict(verdict)
    if normalized in EXPOSURE_VERDICTS:
        return 1 if excess_pct > 0 else 0
    if normalized in NO_EXPOSURE_VERDICTS:
        return 1 if excess_pct <= 0 else 0
    return None


def _confidence_to_probability(confidence: object) -> Optional[float]:
    """Map analyst confidence 1-9 to an implied probability 0.5-0.95."""
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        return None
    if conf <= 0:
        return None
    conf = max(1.0, min(9.0, conf))
    return 0.5 + (conf - 1.0) / 8.0 * 0.45


def _brier_score(verdict: str, confidence: object, excess_pct: Decimal) -> Optional[float]:
    """Brier score of the analyst's implied probability of being right."""
    probability = _confidence_to_probability(confidence)
    outcome = _directional_outcome(verdict, excess_pct)
    if probability is None or outcome is None:
        return None
    return round((probability - float(outcome)) ** 2, 4)


# --- Benchmark price helpers (FMP EOD, short-lived successful-result cache) -
_HISTORY_CACHE_TTL_SECONDS = 15 * 60
_history_cache: dict[str, tuple[float, list[dict]]] = {}


def _fmp_history(symbol: str, period: str = "1y") -> Optional[list[dict]]:
    """Fetch FMP EOD history without making transient failures permanent."""
    symbol = str(symbol or "").upper().strip()
    if not symbol:
        return None
    cache_key = f"{symbol}:{period}"
    cached = _history_cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < _HISTORY_CACHE_TTL_SECONDS:
        return cached[1]
    try:
        from .collector import FMPCollector

        rows = FMPCollector().history(symbol, period=period)
    except Exception as exc:
        logger.warning("[accuracy] FMP history failed for %s: %s", symbol, exc)
        _history_cache.pop(cache_key, None)
        return None
    if not isinstance(rows, list) or not rows:
        _history_cache.pop(cache_key, None)
        return None
    _history_cache[cache_key] = (time.monotonic(), rows)
    return rows


def _close_on_or_after(
    rows: Optional[list[dict]],
    target_date: str,
    *,
    max_calendar_days: int = 7,
) -> Optional[float]:
    """First nearby close on/after an ISO date from ascending FMP rows.

    Weekends and exchange holidays require a small forward tolerance. A larger
    gap indicates incomplete provider history and must not silently change the
    measurement horizon.
    """
    if not rows:
        return None
    target = _parse_dt(f"{target_date[:10]}T00:00:00+00:00")
    if target is None:
        return None
    latest = target + timedelta(days=max(0, int(max_calendar_days)))
    for row in rows:
        row_date = _parse_dt(f"{str(row.get('date') or '')[:10]}T00:00:00+00:00")
        if row_date is None or row_date < target:
            continue
        if row_date > latest:
            return None
        close = row.get("close")
        if close:
            return float(close)
    return None


def _benchmark_return_pct(
    start: datetime,
    end: datetime,
    symbol: Optional[str] = None,
    period: str = "1y",
) -> Optional[Decimal]:
    """Benchmark close-to-close return over [start, end] in percent."""
    symbol = symbol or ACCURACY_BENCHMARK_TICKER
    rows = _fmp_history(symbol, period=period)
    start_close = _close_on_or_after(rows, start.date().isoformat())
    end_close = _close_on_or_after(rows, end.date().isoformat())
    if not start_close or not end_close:
        return None
    return ((Decimal(str(end_close)) - Decimal(str(start_close)))
            / Decimal(str(start_close)) * 100).quantize(CENTS)


def _num_or_none(v: object) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _to_decimal(v: object) -> Decimal:
    if isinstance(v, Decimal):
        return v
    if isinstance(v, (int, float)):
        return Decimal(str(v))
    if isinstance(v, str):
        try:
            return Decimal(v)
        except Exception:
            return Decimal("0")
    return Decimal("0")


@dataclass
class Recommendation:
    """A single council recommendation to track."""

    ticker: str
    verdict: str  # STRONG BUY, BUY, WATCH, AVOID, STRONG SELL
    consensus: str  # 3/3, 2-1, Split
    entry_price: str  # Decimal stored as string for JSON
    recommended_action: str
    allocation: str
    fundamental_verdict: str
    fundamental_confidence: int
    technical_verdict: str
    technical_confidence: int
    contrarian_verdict: str
    contrarian_confidence: int
    timestamp: str = ""  # ISO UTC
    review_after: str = ""  # ISO UTC — 30 days later
    status: str = "PENDING"  # PENDING, GRADED
    # Filled after grading:
    price_at_review: str = "0"
    price_change_pct: str = "0"
    grade: str = ""  # CORRECT, PARTIALLY_CORRECT, INCORRECT
    analyst_grades: dict = field(default_factory=dict)
    notes: str = ""
    council_version: str = ""
    accuracy_era: str = ""
    analyst_labels: dict = field(default_factory=dict)


class AccuracyTracker:
    """Track and grade council recommendations."""

    def __init__(self, path: Path = ACCURACY_FILE):
        self.path = path
        self.lock_path = path.with_suffix(".lock")

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self, records: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(self.path.parent), suffix=".tmp", prefix=".accuracy_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, str(self.path))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _lock(self, exclusive: bool = True):
        """Context-manager-free locking for use in explicit blocks."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = open(self.lock_path, "a", encoding="utf-8")
        if fcntl is not None:
            mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(lock_file.fileno(), mode)
        return lock_file

    def _unlock(self, lock_file):
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()

    def record_recommendation(self, rec: Recommendation) -> None:
        """Record a new council recommendation for future grading."""
        now = _utcnow()
        rec.timestamp = now.isoformat()
        rec.review_after = (now + timedelta(days=ACCURACY_REVIEW_DAYS)).isoformat()
        rec.status = "PENDING"
        rec.council_version = rec.council_version or Config.ACCURACY_CURRENT_COUNCIL_VERSION
        rec.accuracy_era = rec.accuracy_era or "current"
        rec.analyst_labels = rec.analyst_labels or dict(CURRENT_ANALYST_LABELS)

        lf = self._lock(exclusive=True)
        try:
            records = self._load()
            records.append(asdict(rec))
            self._save(records)
            logger.info(
                f"[accuracy] Recorded {rec.ticker} {rec.verdict} @ ${rec.entry_price} "
                f"— review after {rec.review_after[:10]}"
            )
        finally:
            self._unlock(lf)

    def get_pending_reviews(self) -> list[dict]:
        """Return recommendations that are past their 30-day review date."""
        now = _utcnow()
        lf = self._lock(exclusive=False)
        try:
            records = self._load()
        finally:
            self._unlock(lf)

        due = []
        for rec in records:
            if rec.get("status") != "PENDING":
                continue
            review_after = rec.get("review_after", "")
            if not review_after:
                continue
            try:
                review_dt = datetime.fromisoformat(review_after)
                if review_dt.tzinfo is None:
                    review_dt = review_dt.replace(tzinfo=timezone.utc)
                if now >= review_dt:
                    due.append(rec)
            except (ValueError, TypeError):
                continue
        return due

    @staticmethod
    def _apply_grade_fields(
        target: dict,
        current: Decimal,
        change_pct: Decimal,
        benchmark_return: Optional[Decimal],
    ) -> dict:
        """Apply benchmark-relative grade + analyst attribution to a record in place."""
        verdict = target.get("verdict", "").upper()

        if benchmark_return is not None:
            excess_pct = (change_pct - benchmark_return).quantize(CENTS)
            grade = _grade_verdict_excess(verdict, excess_pct)
            grade_basis = f"excess_vs_{ACCURACY_BENCHMARK_TICKER}"
            if grade == "UNGRADED":
                # Verdict outside known families — fall back to legacy grading.
                grade = _grade_verdict(verdict, change_pct)
        else:
            # Benchmark unavailable: legacy absolute grading so records still close.
            excess_pct = None
            grade = _grade_verdict(verdict, change_pct)
            grade_basis = "absolute_legacy_fallback"

        # Grade individual analysts against their OWN verdict, plus a Brier
        # score on their stated confidence (1-9 → implied p of 0.5-0.95).
        analyst_grades: dict[str, str] = {}
        analyst_brier: dict[str, Optional[float]] = {}
        for analyst_key, analyst_name in _analyst_labels_for_record(target).items():
            a_verdict = target.get(f"{analyst_key}_verdict", "").upper()
            a_confidence = target.get(f"{analyst_key}_confidence", 0)
            if excess_pct is not None:
                a_grade = _grade_verdict_excess(a_verdict, excess_pct)
                if a_grade == "UNGRADED":
                    a_grade = _grade_analyst_verdict(a_verdict, change_pct)
                analyst_brier[analyst_name] = _brier_score(a_verdict, a_confidence, excess_pct)
            else:
                a_grade = _grade_analyst_verdict(a_verdict, change_pct)
                analyst_brier[analyst_name] = None
            analyst_grades[analyst_name] = a_grade

        entry = _to_decimal(target.get("entry_price", "0"))
        target["status"] = "GRADED"
        target["price_at_review"] = str(current)
        target["price_change_pct"] = str(change_pct)
        target["benchmark_ticker"] = ACCURACY_BENCHMARK_TICKER
        target["benchmark_return_pct"] = str(benchmark_return) if benchmark_return is not None else ""
        target["excess_return_pct"] = str(excess_pct) if excess_pct is not None else ""
        target["grade_basis"] = grade_basis
        target["grade"] = grade
        target["analyst_grades"] = analyst_grades
        target["analyst_brier"] = analyst_brier
        excess_note = (
            f" vs {ACCURACY_BENCHMARK_TICKER} {benchmark_return:+}% (excess {excess_pct:+}%)"
            if excess_pct is not None else ""
        )
        target["notes"] = (
            f"Entry ${entry} → Review ${current} ({change_pct:+}%){excess_note}. "
            f"Verdict was {verdict}. Grade: {grade}."
        )
        return target

    def grade_recommendation(
        self,
        ticker: str,
        timestamp: str,
        current_price: float,
        benchmark_return_pct: Optional[float] = None,
    ) -> Optional[dict]:
        """Grade a recommendation on benchmark-relative excess return.

        Grading logic (excess = ticker return minus benchmark return over the
        same window, in percentage points):
        - Exposure verdicts (BUY/STARTER/TACTICAL_BUY/ACCUMULATE/HOLD):
          INCORRECT when lagging the benchmark by >10pp, PARTIALLY_CORRECT for
          a 5-10pp lag, CORRECT when matching or beating it.
        - No-exposure verdicts (DEFER/WATCH/AVOID/SELL/TRIM): symmetric —
          INCORRECT when the name beats the benchmark by >10pp,
          PARTIALLY_CORRECT for 5-10pp, CORRECT when it matches or lags.

        Each analyst is graded against their OWN verdict, with a Brier score
        on their confidence. When the benchmark fetch fails, grading falls
        back to the legacy absolute rules so records still close.
        """
        lf = self._lock(exclusive=True)
        try:
            records = self._load()
            target = None
            target_idx = -1
            for i, rec in enumerate(records):
                if (
                    rec.get("ticker") == ticker
                    and rec.get("timestamp") == timestamp
                    and rec.get("status") == "PENDING"
                ):
                    target = rec
                    target_idx = i
                    break

            if target is None:
                return None

            entry = _to_decimal(target.get("entry_price", "0"))
            current = _to_decimal(current_price)
            rec_dt = _record_timestamp(target) or _utcnow()

            if entry == 0:
                # Recorded without a usable quote — backfill the entry from EOD
                # history instead of leaving the record PENDING forever.
                backfilled = _close_on_or_after(
                    _fmp_history(ticker), rec_dt.date().isoformat()
                )
                if backfilled:
                    entry = _to_decimal(backfilled)
                    target["entry_price"] = str(entry)
                    target["notes"] = "Entry price backfilled from EOD history. "
                else:
                    target["status"] = "GRADED"
                    target["grade"] = "UNGRADED"
                    target["notes"] = "No entry price recorded and EOD backfill failed."
                    records[target_idx] = target
                    self._save(records)
                    return target

            change_pct = ((current - entry) / entry * 100).quantize(CENTS)

            if benchmark_return_pct is not None:
                benchmark_return: Optional[Decimal] = _to_decimal(benchmark_return_pct).quantize(CENTS)
            else:
                benchmark_return = _benchmark_return_pct(rec_dt, _utcnow())

            target = self._apply_grade_fields(target, current, change_pct, benchmark_return)
            records[target_idx] = target
            self._save(records)

            logger.info(
                f"[accuracy] Graded {ticker}: {target.get('verdict', '').upper()} → {target['grade']} "
                f"({change_pct:+}% raw, excess {target.get('excess_return_pct') or 'n/a'}% "
                f"over {ACCURACY_REVIEW_DAYS} days)"
            )
            return target
        finally:
            self._unlock(lf)

    def backfill_regrade(self, now: Optional[datetime] = None) -> dict:
        """Regrade every matured record with benchmark-relative excess grading.

        Recomputes the ticker return over the exact [timestamp, timestamp +
        ACCURACY_REVIEW_DAYS] window from FMP EOD closes (entry price stays the
        recorded live price), fetches the benchmark over the same window, and
        re-applies grades + analyst attribution + Brier scores. Previously
        GRADED records keep their old grade in ``previous_grade``.

        Returns a summary dict with before/after grade distributions.
        """
        now = now or _utcnow()
        lf = self._lock(exclusive=True)
        try:
            records = self._load()
            before = {}
            for rec in records:
                key = rec.get("grade") or rec.get("status", "?")
                before[key] = before.get(key, 0) + 1

            regraded = 0
            skipped: list[str] = []
            for idx, rec in enumerate(records):
                rec_dt = _record_timestamp(rec)
                ticker = str(rec.get("ticker") or "").upper()
                if not rec_dt or not ticker:
                    continue
                review_dt = rec_dt + timedelta(days=ACCURACY_REVIEW_DAYS)
                if review_dt > now:
                    continue  # Not matured yet — leave PENDING for the nightly loop.

                entry = _to_decimal(rec.get("entry_price", "0"))
                rows = _fmp_history(ticker)
                if entry == 0:
                    backfilled = _close_on_or_after(rows, rec_dt.date().isoformat())
                    if backfilled:
                        entry = _to_decimal(backfilled)
                        rec["entry_price"] = str(entry)
                review_close = _close_on_or_after(rows, review_dt.date().isoformat())
                if entry == 0 or not review_close:
                    skipped.append(ticker)
                    continue

                current = _to_decimal(review_close)
                change_pct = ((current - entry) / entry * 100).quantize(CENTS)
                benchmark_return = _benchmark_return_pct(rec_dt, review_dt)

                previous_grade = rec.get("grade") or "PENDING"
                rec = self._apply_grade_fields(rec, current, change_pct, benchmark_return)
                rec["previous_grade"] = previous_grade
                rec["regraded_at"] = now.isoformat()
                records[idx] = rec
                regraded += 1

            after = {}
            for rec in records:
                key = rec.get("grade") or rec.get("status", "?")
                after[key] = after.get(key, 0) + 1

            self._save(records)
            logger.info(
                "[accuracy] Backfill regrade complete: %d regraded, %d skipped",
                regraded, len(skipped),
            )
            return {
                "regraded": regraded,
                "skipped": skipped,
                "before": before,
                "after": after,
            }
        finally:
            self._unlock(lf)

    def backfill_missing_benchmark_grades(
        self,
        now: Optional[datetime] = None,
        *,
        max_tickers: int = 25,
        current_era_only: bool = True,
    ) -> dict[str, Any]:
        """Retry fallback grades in bounded batches without holding an I/O lock.

        Provider calls happen against a read snapshot. The exclusive lock is
        reacquired only to apply still-missing results, so Council recording is
        never blocked behind a long historical-data fetch.
        """
        now = now or _utcnow()
        lf = self._lock(exclusive=False)
        try:
            snapshot = self._load()
        finally:
            self._unlock(lf)

        candidates: list[dict[str, Any]] = []
        for rec in snapshot:
            if rec.get("status") != "GRADED":
                continue
            if str(rec.get("grade_basis") or "") != "absolute_legacy_fallback":
                continue
            if current_era_only and not _is_current_era(rec):
                continue
            rec_dt = _record_timestamp(rec)
            if not rec_dt or rec_dt + timedelta(days=ACCURACY_REVIEW_DAYS) > now:
                continue
            if _to_decimal(rec.get("entry_price", "0")) <= 0:
                continue
            candidates.append(dict(rec))

        ticker_limit = max(0, int(max_tickers))
        last_attempt_by_ticker: dict[str, str] = {}
        for row in candidates:
            ticker = str(row.get("ticker") or "").upper()
            attempted_at = str(row.get("benchmark_backfill_last_attempt_at") or "")
            existing = last_attempt_by_ticker.get(ticker)
            if existing is None or attempted_at < existing:
                last_attempt_by_ticker[ticker] = attempted_at
        selected_tickers = [
            ticker
            for ticker, _ in sorted(
                last_attempt_by_ticker.items(),
                key=lambda item: (item[1], item[0]),
            )[:ticker_limit]
        ]
        selected_set = set(selected_tickers)
        attempted_keys: set[tuple[str, str]] = set()
        prepared: dict[tuple[str, str], tuple[Decimal, Decimal, Decimal]] = {}
        skipped = 0
        for rec in candidates:
            ticker = str(rec.get("ticker") or "").upper()
            timestamp = str(rec.get("timestamp") or "")
            if ticker not in selected_set:
                continue
            attempted_keys.add((ticker, timestamp))
            rec_dt = _record_timestamp(rec)
            if not rec_dt:
                skipped += 1
                continue
            review_dt = rec_dt + timedelta(days=ACCURACY_REVIEW_DAYS)
            review_close = _close_on_or_after(
                _fmp_history(ticker, period="5y"),
                review_dt.date().isoformat(),
            )
            benchmark_return = _benchmark_return_pct(rec_dt, review_dt, period="5y")
            entry = _to_decimal(rec.get("entry_price", "0"))
            if not review_close or benchmark_return is None or entry <= 0:
                skipped += 1
                continue
            current = _to_decimal(review_close)
            change_pct = ((current - entry) / entry * 100).quantize(CENTS)
            prepared[(ticker, timestamp)] = (current, change_pct, benchmark_return)

        regraded = 0
        if attempted_keys:
            lf = self._lock(exclusive=True)
            try:
                records = self._load()
                changed = False
                for idx, rec in enumerate(records):
                    key = (
                        str(rec.get("ticker") or "").upper(),
                        str(rec.get("timestamp") or ""),
                    )
                    if key not in attempted_keys:
                        continue
                    if str(rec.get("grade_basis") or "") != "absolute_legacy_fallback":
                        continue
                    updated = dict(rec)
                    updated["benchmark_backfill_last_attempt_at"] = now.isoformat()
                    updated["benchmark_backfill_attempts"] = (
                        int(updated.get("benchmark_backfill_attempts") or 0) + 1
                    )
                    values = prepared.get(key)
                    if values is None:
                        records[idx] = updated
                        changed = True
                        continue
                    previous_grade = str(rec.get("grade") or "")
                    current, change_pct, benchmark_return = values
                    updated = self._apply_grade_fields(
                        updated,
                        current,
                        change_pct,
                        benchmark_return,
                    )
                    updated.setdefault("previous_grade", previous_grade)
                    updated["benchmark_backfilled_at"] = now.isoformat()
                    records[idx] = updated
                    regraded += 1
                    changed = True
                if changed:
                    self._save(records)
            finally:
                self._unlock(lf)

        remaining = max(0, len(candidates) - regraded)
        return {
            "eligible_fallback_records": len(candidates),
            "selected_tickers": len(selected_tickers),
            "attempted_records": len(attempted_keys),
            "prepared": len(prepared),
            "regraded": regraded,
            "skipped": skipped,
            "remaining": remaining,
            "current_era_only": bool(current_era_only),
        }

    def grade_due_recommendations_from_history(
        self,
        now: Optional[datetime] = None,
        *,
        max_records: int = 250,
    ) -> dict[str, Any]:
        """Grade due recommendations at the fixed review horizon.

        Historical ticker and benchmark closes are fetched outside the file
        lock. Missing provider data leaves a recommendation pending for a
        later retry instead of closing it with a different time horizon.
        """
        current_time = now or _utcnow()
        lf = self._lock(exclusive=False)
        try:
            snapshot = self._load()
        finally:
            self._unlock(lf)

        due: list[dict[str, Any]] = []
        for rec in snapshot:
            if rec.get("status") != "PENDING":
                continue
            rec_dt = _record_timestamp(rec)
            if rec_dt is None:
                continue
            review_dt = rec_dt + timedelta(days=ACCURACY_REVIEW_DAYS)
            if review_dt <= current_time:
                due.append(dict(rec))
        due.sort(
            key=lambda rec: (
                str(rec.get("review_after") or ""),
                str(rec.get("ticker") or ""),
                str(rec.get("timestamp") or ""),
            )
        )
        total_due = len(due)
        due = due[: max(0, int(max_records))]

        prepared: dict[
            tuple[str, str],
            tuple[Decimal, Decimal, Decimal, Decimal, datetime, bool],
        ] = {}
        skipped: list[dict[str, str]] = []
        for rec in due:
            ticker = str(rec.get("ticker") or "").upper().strip()
            timestamp = str(rec.get("timestamp") or "")
            rec_dt = _record_timestamp(rec)
            if not ticker or rec_dt is None:
                skipped.append({"ticker": ticker, "timestamp": timestamp, "reason": "invalid_identity"})
                continue
            review_dt = rec_dt + timedelta(days=ACCURACY_REVIEW_DAYS)
            rows = _fmp_history(ticker, period="5y")
            entry = _to_decimal(rec.get("entry_price", "0"))
            entry_backfilled = False
            if entry <= 0:
                entry_close = _close_on_or_after(rows, rec_dt.date().isoformat())
                if entry_close:
                    entry = _to_decimal(entry_close)
                    entry_backfilled = True
            review_close = _close_on_or_after(rows, review_dt.date().isoformat())
            benchmark_return = _benchmark_return_pct(rec_dt, review_dt, period="5y")
            if entry <= 0 or not review_close or benchmark_return is None:
                skipped.append(
                    {"ticker": ticker, "timestamp": timestamp, "reason": "historical_data_unavailable"}
                )
                continue
            current = _to_decimal(review_close)
            change_pct = ((current - entry) / entry * 100).quantize(CENTS)
            prepared[(ticker, timestamp)] = (
                entry,
                current,
                change_pct,
                benchmark_return,
                review_dt,
                entry_backfilled,
            )

        graded_records: list[dict[str, Any]] = []
        if prepared:
            lf = self._lock(exclusive=True)
            try:
                records = self._load()
                changed = False
                for idx, rec in enumerate(records):
                    key = (
                        str(rec.get("ticker") or "").upper(),
                        str(rec.get("timestamp") or ""),
                    )
                    values = prepared.get(key)
                    if values is None or rec.get("status") != "PENDING":
                        continue
                    entry, current, change_pct, benchmark_return, review_dt, entry_backfilled = values
                    updated = dict(rec)
                    if entry_backfilled:
                        updated["entry_price"] = str(entry)
                    updated = self._apply_grade_fields(
                        updated,
                        current,
                        change_pct,
                        benchmark_return,
                    )
                    updated["graded_at"] = current_time.isoformat()
                    updated["review_window_end"] = review_dt.isoformat()
                    updated["review_price_source"] = "fmp_eod_close_on_or_after_fixed_horizon"
                    records[idx] = updated
                    graded_records.append(dict(updated))
                    changed = True
                if changed:
                    self._save(records)
            finally:
                self._unlock(lf)

        return {
            "due": total_due,
            "selected": len(due),
            "graded": len(graded_records),
            "graded_records": graded_records,
            "skipped": skipped,
            "remaining_due": max(0, total_due - len(graded_records)),
            "fixed_review_days": ACCURACY_REVIEW_DAYS,
        }

    def backfill_recommendation_outcomes(self, journal: Any = None) -> int:
        """Grade matured artha.db recommendation rows the same excess-vs-benchmark way.

        Fills ``outcome``/``outcome_notes`` on rows older than the review
        window that are still 'unknown'. Never touches ``status`` (owned by
        live execution flows). Returns count of rows updated.
        """
        if journal is None:
            from .journal import DecisionJournal
            journal = DecisionJournal()
        now = _utcnow()
        updated = 0
        with journal._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, ticker, action, timestamp, price_at_recommendation
                FROM recommendations
                WHERE (outcome IS NULL OR outcome = 'unknown' OR outcome = '')
                """
            ).fetchall()
            for row in rows:
                rec_dt = _parse_dt(row["timestamp"])
                ticker = str(row["ticker"] or "").upper()
                if not rec_dt or not ticker:
                    continue
                review_dt = rec_dt + timedelta(days=ACCURACY_REVIEW_DAYS)
                if review_dt > now:
                    continue
                hist = _fmp_history(ticker)
                entry = _num_or_none(row["price_at_recommendation"]) or _close_on_or_after(
                    hist, rec_dt.date().isoformat()
                )
                review_close = _close_on_or_after(hist, review_dt.date().isoformat())
                if not entry or not review_close:
                    continue
                change_pct = ((Decimal(str(review_close)) - Decimal(str(entry)))
                              / Decimal(str(entry)) * 100).quantize(CENTS)
                benchmark_return = _benchmark_return_pct(rec_dt, review_dt)
                if benchmark_return is None:
                    continue
                excess_pct = (change_pct - benchmark_return).quantize(CENTS)
                grade = _grade_verdict_excess(str(row["action"] or ""), excess_pct)
                if grade == "UNGRADED":
                    continue
                conn.execute(
                    "UPDATE recommendations SET outcome = ?, outcome_notes = ? WHERE id = ?",
                    (
                        grade.lower(),
                        f"{ACCURACY_REVIEW_DAYS}d return {change_pct:+}% vs "
                        f"{ACCURACY_BENCHMARK_TICKER} {benchmark_return:+}% "
                        f"(excess {excess_pct:+}%)",
                        row["id"],
                    ),
                )
                updated += 1
            conn.commit()
        logger.info("[accuracy] Backfilled outcomes for %d recommendation row(s)", updated)
        return updated

    def get_summary_stats(self, since: object = None, *, benchmark_only: bool = False) -> dict:
        """Return aggregate accuracy statistics."""
        since_dt = _parse_dt(since) if since is not None else None
        lf = self._lock(exclusive=False)
        try:
            records = self._load()
        finally:
            self._unlock(lf)

        if since_dt is not None:
            records = [
                r for r in records
                if (ts := _record_timestamp(r)) is not None and ts >= since_dt
            ]

        all_graded = [r for r in records if r.get("status") == "GRADED"]
        graded = [
            row
            for row in all_graded
            if not benchmark_only
            or (
                str(row.get("grade_basis") or "").startswith("excess_vs_")
                and _num_or_none(row.get("excess_return_pct")) is not None
            )
        ]
        excluded_nonbenchmark = len(all_graded) - len(graded)
        pending = [r for r in records if r.get("status") == "PENDING"]

        if not graded:
            return {
                "total_graded": 0,
                "usable_graded": 0,
                "ungraded": 0,
                "total_pending": len(pending),
                "overall_accuracy": None,
                "weighted_accuracy": None,
                "strict_accuracy": None,
                "analyst_accuracy": {},
                "scope_start": since_dt.isoformat() if since_dt else None,
                "benchmark_only": bool(benchmark_only),
                "excluded_nonbenchmark": excluded_nonbenchmark,
            }

        correct = sum(1 for r in graded if r.get("grade") == "CORRECT")
        partial = sum(1 for r in graded if r.get("grade") == "PARTIALLY_CORRECT")
        incorrect = sum(1 for r in graded if r.get("grade") == "INCORRECT")
        ungraded = sum(1 for r in graded if r.get("grade") == "UNGRADED")
        total = correct + partial + incorrect
        excess_values = [
            v for v in (_num_or_none(r.get("excess_return_pct")) for r in graded)
            if v is not None
        ]

        # Analyst-level stats (graded vs their OWN verdict, plus Brier score)
        analyst_stats: dict[str, dict[str, Any]] = {}
        for rec in graded:
            briers = rec.get("analyst_brier") or {}
            for analyst, ag in rec.get("analyst_grades", {}).items():
                if analyst not in analyst_stats:
                    analyst_stats[analyst] = {
                        "correct": 0, "partial": 0, "incorrect": 0, "total": 0,
                        "brier_sum": 0.0, "brier_n": 0,
                    }
                analyst_stats[analyst]["total"] += 1
                if ag == "CORRECT":
                    analyst_stats[analyst]["correct"] += 1
                elif ag == "PARTIALLY_CORRECT":
                    analyst_stats[analyst]["partial"] += 1
                elif ag == "INCORRECT":
                    analyst_stats[analyst]["incorrect"] += 1
                brier = _num_or_none(briers.get(analyst)) if isinstance(briers, dict) else None
                if brier is not None:
                    analyst_stats[analyst]["brier_sum"] += brier
                    analyst_stats[analyst]["brier_n"] += 1

        analyst_accuracy = {}
        for analyst, stats in analyst_stats.items():
            t = stats["total"]
            if t > 0:
                analyst_accuracy[analyst] = {
                    "accuracy": round((stats["correct"] + 0.5 * stats["partial"]) / t * 100, 1),
                    "correct": stats["correct"],
                    "partial": stats["partial"],
                    "incorrect": stats["incorrect"],
                    "total": t,
                    "avg_brier": (
                        round(stats["brier_sum"] / stats["brier_n"], 4)
                        if stats["brier_n"] else None
                    ),
                    "brier_n": stats["brier_n"],
                }

        return {
            "total_graded": len(graded),
            "usable_graded": total,
            "ungraded": ungraded,
            "total_pending": len(pending),
            "overall_accuracy": round((correct + 0.5 * partial) / total * 100, 1) if total else None,
            "weighted_accuracy": round((correct + 0.5 * partial) / total * 100, 1) if total else None,
            "strict_accuracy": round(correct / total * 100, 1) if total else None,
            "correct": correct,
            "partially_correct": partial,
            "incorrect": incorrect,
            "analyst_accuracy": analyst_accuracy,
            "scope_start": since_dt.isoformat() if since_dt else None,
            "benchmark_only": bool(benchmark_only),
            "excluded_nonbenchmark": excluded_nonbenchmark,
            "avg_price_change": round(
                sum(float(r.get("price_change_pct", 0)) for r in graded if r.get("grade") != "UNGRADED") / total, 2
            ) if total else 0,
            "avg_excess_return": (
                round(sum(excess_values) / len(excess_values), 2) if excess_values else None
            ),
        }

    def update_shadow_forward_returns(self, journal) -> dict:
        """Check shadow positions for entries that are 5, 20, or 60 days old.

        Fetches prices via yfinance and calculates point-in-time forward
        returns, benchmark-relative excess returns, MFE, and MAE.
        Returns a summary dict of updates performed.

        Args:
            journal: DecisionJournal instance with shadow trade DB access.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("[accuracy] yfinance not available — skipping shadow return update")
            return {"updated": 0, "errors": 0}

        pending = journal.get_pending_shadow_reviews()
        if not pending:
            return {"updated": 0, "errors": 0, "skipped": 0}

        from .portfolio_risk import primary_market_benchmark_for, sector_benchmark_for

        now = _utcnow()
        updated_count = 0
        error_count = 0
        history_cache: dict[str, Any] = {}

        def _history(symbol: str, age_days: int):
            symbol = str(symbol or "").upper().strip()
            if not symbol:
                return None
            period = "1y" if age_days > 120 else "6mo"
            cache_key = f"{symbol}:{period}"
            if cache_key not in history_cache:
                history_cache[cache_key] = yf.Ticker(symbol).history(period=period)
            return history_cache[cache_key]

        def _normalize_index_ts(idx):
            ts = idx
            if hasattr(ts, "tzinfo") and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            elif hasattr(ts, "tz_convert"):
                ts = ts.tz_convert(timezone.utc)
            return ts

        def _filter_from_decision(df, created_dt):
            if df is None or getattr(df, "empty", True):
                return df
            keep = []
            for idx in df.index:
                keep.append(_normalize_index_ts(idx) >= created_dt)
            return df.loc[keep]

        def _price_on_or_after(closes, target_dt):
            for idx in closes.index:
                if _normalize_index_ts(idx) >= target_dt:
                    return float(closes[idx])
            return None

        def _checkpoint_price(closes, created_dt, n_days):
            return _price_on_or_after(closes, created_dt + timedelta(days=n_days))

        for shadow in pending:
            shadow_id = shadow.get("id")
            ticker = str(shadow.get("ticker", "")).upper()
            created_str = str(shadow.get("created_at", shadow.get("timestamp", "")))
            entry_price = float(shadow.get("hypothetical_entry", 0) or 0)
            stop_pct = -0.08  # Default stop
            sector = str(shadow.get("sector") or "").strip()
            benchmark_ticker = str(
                shadow.get("benchmark_ticker") or primary_market_benchmark_for(sector)
            ).upper()
            sector_benchmark_ticker = str(
                shadow.get("sector_benchmark_ticker") or sector_benchmark_for(sector, fallback=benchmark_ticker)
            ).upper()

            if not ticker or not created_str or entry_price <= 0:
                continue

            try:
                created_dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
            except (ValueError, AttributeError):
                continue

            age_days = (now - created_dt).days

            # Determine which price checkpoints we need
            need_5d = age_days >= 5 and shadow.get("price_5d") is None
            need_20d = age_days >= 20 and shadow.get("price_20d") is None
            need_60d = age_days >= 60 and shadow.get("price_60d") is None

            if not (need_5d or need_20d or need_60d):
                continue

            try:
                # Fetch price history via yfinance. All excursions/checkpoints
                # are computed from decision time onward; older prices must not
                # leak into MFE/MAE calibration.
                hist = _history(ticker, age_days)
                if hist is None or hist.empty:
                    continue

                hist_after = _filter_from_decision(hist, created_dt)
                if hist_after is None or hist_after.empty:
                    continue

                closes = hist_after["Close"]
                highs = hist_after["High"]
                lows = hist_after["Low"]

                update_kwargs = {}
                if need_5d:
                    p = _checkpoint_price(closes, created_dt, 5)
                    if p:
                        update_kwargs["price_5d"] = p
                if need_20d:
                    p = _checkpoint_price(closes, created_dt, 20)
                    if p:
                        update_kwargs["price_20d"] = p
                if need_60d:
                    p = _checkpoint_price(closes, created_dt, 60)
                    if p:
                        update_kwargs["price_60d"] = p

                # Benchmark and sector-relative returns.
                for prefix, symbol in (
                    ("benchmark", benchmark_ticker),
                    ("sector_benchmark", sector_benchmark_ticker),
                ):
                    bench_hist = _history(symbol, age_days)
                    bench_after = _filter_from_decision(bench_hist, created_dt)
                    if bench_after is None or bench_after.empty:
                        continue
                    bench_closes = bench_after["Close"]
                    entry_col = f"{prefix}_price_entry"
                    if shadow.get(entry_col) is None:
                        entry = _price_on_or_after(bench_closes, created_dt)
                        if entry:
                            update_kwargs[entry_col] = entry
                    for n_days, needed in ((5, need_5d), (20, need_20d), (60, need_60d)):
                        if not needed:
                            continue
                        p = _checkpoint_price(bench_closes, created_dt, n_days)
                        if p:
                            update_kwargs[f"{prefix}_price_{n_days}d"] = p

                # Compute MFE/MAE over available history
                if entry_price > 0 and len(closes) > 0:
                    # MFE = max favorable excursion (best high above entry)
                    mfe_val = float((highs.max() - entry_price) / entry_price) if len(highs) > 0 else 0
                    # MAE = max adverse excursion (worst low below entry)
                    mae_val = float((lows.min() - entry_price) / entry_price) if len(lows) > 0 else 0
                    update_kwargs["mfe"] = mfe_val
                    update_kwargs["mae"] = mae_val
                    # Would it have hit an 8% stop?
                    update_kwargs["would_hit_stop"] = mae_val <= stop_pct

                if update_kwargs:
                    journal.update_shadow_returns(shadow_id, **update_kwargs)
                    updated_count += 1
                    logger.info(
                        f"[accuracy] Updated shadow trade {shadow_id} ({ticker}): "
                        f"age={age_days}d, updates={list(update_kwargs.keys())}"
                    )

            except Exception as exc:
                logger.error(f"[accuracy] Failed to update shadow trade {shadow_id} ({ticker}): {exc}")
                error_count += 1
                continue

        return {
            "updated": updated_count,
            "errors": error_count,
            "skipped": len(pending) - updated_count - error_count,
        }

    def format_shadow_trade_report(self, journal) -> Optional[str]:
        """Format a summary of shadow trade performance for nightly review."""
        try:
            stats = journal.get_shadow_trade_stats()
        except Exception as exc:
            logger.error(f"[accuracy] Failed to get shadow trade stats: {exc}")
            return None

        if stats.get("total", 0) == 0:
            return None

        lines = [
            "👻 SHADOW TRADE TRACKER",
            f"{'━' * 25}",
            f"Total shadow trades: {stats['total']} ({stats['completed']} completed, {stats['tracking']} tracking)",
            "",
        ]

        avg_returns = stats.get("avg_returns", {})
        if any(v is not None for v in avg_returns.values()):
            lines.append("📈 Avg Forward Returns (blocked trades):")
            if avg_returns.get("return_5d") is not None:
                lines.append(f"   5-day:  {avg_returns['return_5d']:+.1%}")
            if avg_returns.get("return_20d") is not None:
                lines.append(f"   20-day: {avg_returns['return_20d']:+.1%}")
            if avg_returns.get("return_60d") is not None:
                lines.append(f"   60-day: {avg_returns['return_60d']:+.1%}")
            lines.append("")

        hit_stop_rate = stats.get("would_hit_stop_rate")
        if hit_stop_rate is not None:
            lines.append(f"🛑 Would-hit-stop rate: {hit_stop_rate:.0%}")
            lines.append("")

        blocked_by = stats.get("blocked_by", {})
        if blocked_by:
            lines.append("🔒 Blocked by:")
            for reason, count in sorted(blocked_by.items(), key=lambda x: -x[1]):
                lines.append(f"   {reason}: {count}")

        return "\n".join(lines)

    def grade_sell_decisions(self, collector: Any) -> int:
        """Grade recent sell decisions using post-sell shadow tracking.

        Returns count of newly graded sells.
        """
        from .opportunity_cost import PostSellTracker
        tracker = PostSellTracker()
        try:
            updated = tracker.update_shadow_prices(collector)
            if updated:
                logger.info("[accuracy] Graded %d sell decision(s)", updated)
            return updated
        except Exception as e:
            logger.warning("[accuracy] Sell grading failed: %s", e)
            return 0

    def format_sell_accuracy_report(self) -> Optional[str]:
        """Format a sell-accuracy summary for the nightly review."""
        from .opportunity_cost import PostSellTracker
        tracker = PostSellTracker()
        try:
            report = tracker.format_report()
            return report if report else None
        except Exception as e:
            logger.warning("[accuracy] Failed to format sell report: %s", e)
            return None

    def format_monthly_report(self) -> Optional[str]:
        """Format a Telegram-friendly monthly accuracy report."""
        stats = self.get_summary_stats()
        current_stats = self.get_summary_stats(
            since=Config.ACCURACY_CURRENT_ERA_START,
            benchmark_only=True,
        )
        if stats["total_graded"] == 0 and stats["total_pending"] == 0:
            return None

        lines = [
            "📊 ARTHA ACCURACY REPORT",
            f"{'━' * 25}",
            "",
        ]

        lines.append("🧭 Current Council Era")
        lines.append(f"   Version: {Config.ACCURACY_CURRENT_COUNCIL_VERSION}")
        lines.append(f"   Since: {Config.ACCURACY_CURRENT_ERA_START[:10]}")
        if current_stats["usable_graded"] > 0:
            lines.append(f"   Weighted score: {current_stats['weighted_accuracy']}%")
            lines.append(f"   Strict correct-only: {current_stats['strict_accuracy']}%")
            lines.append(
                f"   Correct: {current_stats['correct']} | Partial: {current_stats['partially_correct']} "
                f"| Wrong: {current_stats['incorrect']}"
            )
            if current_stats.get("ungraded"):
                lines.append(f"   Ungraded legacy-label rows: {current_stats['ungraded']}")
            lines.append(f"   Avg Price Change: {current_stats['avg_price_change']:+.1f}%")
            lines.append("")

            lines.append("🏛️ Current Analyst Scorecard:")
            for analyst, data in current_stats["analyst_accuracy"].items():
                lines.append(
                    f"   {analyst}: {data['accuracy']}% "
                    f"({data['correct']}✓ {data['partial']}~ {data['incorrect']}✗)"
                )
        else:
            lines.append(
                f"   No final usable grades yet; "
                f"{current_stats['total_pending']} pending current-era review(s)."
            )
            lines.append("   Treat current-era quality as still in live tracking, not proven by legacy rows.")
        lines.append("")

        if stats["usable_graded"] > 0:
            lines.append("📜 Legacy / All-Time Context")
            lines.append(f"   Weighted historical score: {stats['weighted_accuracy']}%")
            lines.append(f"   Strict correct-only score: {stats['strict_accuracy']}%")
            lines.append(
                f"   Correct: {stats['correct']} | Partial: {stats['partially_correct']} "
                f"| Wrong: {stats['incorrect']}"
            )
            if stats.get("ungraded"):
                lines.append(f"   Ungraded legacy-label rows: {stats['ungraded']}")
            lines.append(
                "   Legacy rows include older model/prompt eras and are not prompt-tune triggers by themselves."
            )
            lines.append("")

            lines.append("🏛️ Legacy + Current Scorecard:")
            for analyst, data in stats["analyst_accuracy"].items():
                lines.append(
                    f"   {analyst}: {data['accuracy']}% "
                    f"({data['correct']}✓ {data['partial']}~ {data['incorrect']}✗)"
                )
            lines.append("")

        if stats["total_pending"] > 0:
            lines.append(f"⏳ Pending review: {stats['total_pending']} total recommendation(s)")

        lines.append("")
        lines.append(f"{'━' * 25}")
        lines.append("💡 Artha learns from every pick to improve over time.")

        return "\n".join(lines)
