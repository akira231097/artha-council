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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Optional

from .config import Config


logger = logging.getLogger(__name__)

try:
    import fcntl
except ImportError:
    fcntl = None

CENTS = Decimal("0.01")
ACCURACY_FILE = Path(__file__).resolve().parent.parent / "data" / "accuracy.json"
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
        rec.review_after = (now + timedelta(days=30)).isoformat()
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

    def grade_recommendation(
        self,
        ticker: str,
        timestamp: str,
        current_price: float,
    ) -> Optional[dict]:
        """Grade a recommendation by comparing entry price to current price.

        Grading logic:
        - BUY/STRONG BUY → price went up ≥5% = CORRECT, 0-5% = PARTIALLY_CORRECT, down = INCORRECT
        - AVOID/STRONG SELL → price went down or flat = CORRECT, up ≤5% = PARTIALLY_CORRECT, up >5% = INCORRECT
        - WATCH → any outcome within ±10% = CORRECT (it was genuinely uncertain)

        Each analyst also gets graded individually.
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
            if entry == 0:
                return None

            change_pct = ((current - entry) / entry * 100).quantize(CENTS)
            verdict = target.get("verdict", "").upper()

            # Grade overall verdict
            if verdict in ("STRONG BUY", "BUY"):
                if change_pct >= 5:
                    grade = "CORRECT"
                elif change_pct >= 0:
                    grade = "PARTIALLY_CORRECT"
                else:
                    grade = "INCORRECT"
            elif verdict in ("AVOID", "STRONG SELL"):
                if change_pct <= 0:
                    grade = "CORRECT"
                elif change_pct <= 5:
                    grade = "PARTIALLY_CORRECT"
                else:
                    grade = "INCORRECT"
            elif verdict == "WATCH":
                if abs(change_pct) <= 10:
                    grade = "CORRECT"
                elif change_pct > 10:
                    grade = "PARTIALLY_CORRECT"  # Missed opportunity
                else:
                    grade = "CORRECT"  # Avoided a drop
            else:
                grade = "UNGRADED"

            # Grade individual analysts
            analyst_grades = {}
            for analyst_key, analyst_name in _analyst_labels_for_record(target).items():
                a_verdict = target.get(f"{analyst_key}_verdict", "").upper()
                if a_verdict in ("BUY",):
                    if change_pct >= 5:
                        analyst_grades[analyst_name] = "CORRECT"
                    elif change_pct >= 0:
                        analyst_grades[analyst_name] = "PARTIALLY_CORRECT"
                    else:
                        analyst_grades[analyst_name] = "INCORRECT"
                elif a_verdict in ("SELL",):
                    if change_pct <= 0:
                        analyst_grades[analyst_name] = "CORRECT"
                    elif change_pct <= 5:
                        analyst_grades[analyst_name] = "PARTIALLY_CORRECT"
                    else:
                        analyst_grades[analyst_name] = "INCORRECT"
                elif a_verdict in ("HOLD",):
                    if abs(change_pct) <= 10:
                        analyst_grades[analyst_name] = "CORRECT"
                    else:
                        analyst_grades[analyst_name] = "PARTIALLY_CORRECT"
                else:
                    analyst_grades[analyst_name] = "UNGRADED"

            # Update record
            target["status"] = "GRADED"
            target["price_at_review"] = str(current)
            target["price_change_pct"] = str(change_pct)
            target["grade"] = grade
            target["analyst_grades"] = analyst_grades
            target["notes"] = (
                f"Entry ${entry} → Review ${current} ({change_pct:+}%). "
                f"Verdict was {verdict}. Grade: {grade}."
            )
            records[target_idx] = target
            self._save(records)

            logger.info(
                f"[accuracy] Graded {ticker}: {verdict} → {grade} "
                f"({change_pct:+}% over 30 days)"
            )
            return target
        finally:
            self._unlock(lf)

    def get_summary_stats(self, since: object = None) -> dict:
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

        graded = [r for r in records if r.get("status") == "GRADED"]
        pending = [r for r in records if r.get("status") == "PENDING"]

        if not graded:
            return {
                "total_graded": 0,
                "total_pending": len(pending),
                "overall_accuracy": None,
                "analyst_accuracy": {},
                "scope_start": since_dt.isoformat() if since_dt else None,
            }

        correct = sum(1 for r in graded if r.get("grade") == "CORRECT")
        partial = sum(1 for r in graded if r.get("grade") == "PARTIALLY_CORRECT")
        incorrect = sum(1 for r in graded if r.get("grade") == "INCORRECT")
        total = correct + partial + incorrect

        # Analyst-level stats
        analyst_stats: dict[str, dict[str, int]] = {}
        for rec in graded:
            for analyst, ag in rec.get("analyst_grades", {}).items():
                if analyst not in analyst_stats:
                    analyst_stats[analyst] = {"correct": 0, "partial": 0, "incorrect": 0, "total": 0}
                analyst_stats[analyst]["total"] += 1
                if ag == "CORRECT":
                    analyst_stats[analyst]["correct"] += 1
                elif ag == "PARTIALLY_CORRECT":
                    analyst_stats[analyst]["partial"] += 1
                elif ag == "INCORRECT":
                    analyst_stats[analyst]["incorrect"] += 1

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
                }

        return {
            "total_graded": len(graded),
            "total_pending": len(pending),
            "overall_accuracy": round((correct + 0.5 * partial) / total * 100, 1) if total else None,
            "correct": correct,
            "partially_correct": partial,
            "incorrect": incorrect,
            "analyst_accuracy": analyst_accuracy,
            "scope_start": since_dt.isoformat() if since_dt else None,
            "avg_price_change": round(
                sum(float(r.get("price_change_pct", 0)) for r in graded) / len(graded), 2
            ) if graded else 0,
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
        current_stats = self.get_summary_stats(since=Config.ACCURACY_CURRENT_ERA_START)
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
        if current_stats["total_graded"] > 0:
            lines.append(f"   Accuracy: {current_stats['overall_accuracy']}%")
            lines.append(
                f"   Correct: {current_stats['correct']} | Partial: {current_stats['partially_correct']} "
                f"| Wrong: {current_stats['incorrect']}"
            )
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
                f"   No graded recommendations yet; "
                f"{current_stats['total_pending']} pending current-era review(s)."
            )
        lines.append("")

        if stats["total_graded"] > 0:
            lines.append("📜 Legacy / All-Time Context")
            lines.append(f"   Overall Accuracy: {stats['overall_accuracy']}%")
            lines.append(
                f"   Correct: {stats['correct']} | Partial: {stats['partially_correct']} "
                f"| Wrong: {stats['incorrect']}"
            )
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
