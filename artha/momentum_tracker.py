"""Momentum Tracker — records momentum scores over time to detect acceleration.

Academic research (ScienceDirect 2020) shows momentum acceleration is a
significant predictor of future returns beyond raw momentum levels.
"""
import json
import logging
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_ACCEL_THRESHOLD = 2.0   # minimum delta to be considered "accelerating"
_DECEL_THRESHOLD = -2.0  # maximum delta to be considered "decelerating"


class MomentumTracker:
    """Tracks momentum scores over time to detect acceleration/deceleration."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.tracker_file = data_dir / "momentum_history.json"

    # ------------------------------------------------------------------
    # Internal I/O helpers
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if not self.tracker_file.exists():
            return {"scores": {}, "last_pruned": None}
        try:
            with open(self.tracker_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("[momentum_tracker] Failed to load %s: %s", self.tracker_file, e)
            return {"scores": {}, "last_pruned": None}

    def _save(self, payload: dict) -> None:
        """Atomic write using tempfile + os.replace."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.data_dir),
            suffix=".tmp",
            prefix=".momentum_history_",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(self.tracker_file))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _prune(self, payload: dict, days: int = 30) -> dict:
        """Remove score entries older than `days` days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        pruned_scores = {}
        for ticker, entries in payload.get("scores", {}).items():
            kept = [
                e for e in entries
                if datetime.fromisoformat(e["date"] + "T00:00:00+00:00") >= cutoff
            ]
            if kept:
                pruned_scores[ticker] = kept
        payload["scores"] = pruned_scores
        payload["last_pruned"] = datetime.now(timezone.utc).isoformat()
        return payload

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_scores(self, ranked_candidates: list[dict], scan_date: str) -> None:
        """Save momentum scores from a ranking run.

        Called after rank_universe(). Stores per-ticker scores keyed by date.
        Prunes entries older than 30 days.

        Args:
            ranked_candidates: list of candidate dicts from rank_universe()
            scan_date: ISO date string (YYYY-MM-DD) for this scan
        """
        if not ranked_candidates:
            return

        payload = self._load()
        scores = payload.setdefault("scores", {})

        for candidate in ranked_candidates:
            ticker = candidate.get("symbol", "").upper()
            if not ticker:
                continue
            entry = {
                "date": scan_date,
                "score": round(float(candidate.get("combined_score", 0)), 3),
                "return_12m": candidate.get("return_12m"),
                "return_3m": candidate.get("return_3m"),
                "return_1m": candidate.get("return_1m"),
            }
            ticker_entries = scores.setdefault(ticker, [])
            # Avoid duplicate entries for the same date
            ticker_entries = [e for e in ticker_entries if e.get("date") != scan_date]
            ticker_entries.append(entry)
            # Keep sorted by date ascending
            ticker_entries.sort(key=lambda e: e["date"])
            scores[ticker] = ticker_entries

        # Opportunistic pruning when the file grows large
        total_entries = sum(len(v) for v in scores.values())
        if total_entries > 5000:
            payload = self._prune(payload)

        self._save(payload)

    def get_momentum_delta(self, ticker: str) -> dict:
        """Get momentum acceleration data for a ticker.

        Returns a dict with current/previous score, delta, trend, and metadata.
        trend is one of: "accelerating", "decelerating", "stable", "new"
        """
        payload = self._load()
        entries = payload.get("scores", {}).get(ticker.upper(), [])

        if not entries:
            return {}

        if len(entries) == 1:
            return {
                "current_score": entries[-1]["score"],
                "previous_score": None,
                "delta": 0.0,
                "trend": "new",
                "scans_tracked": 1,
                "first_seen": entries[0]["date"],
            }

        current = entries[-1]
        previous = entries[-2]
        delta = round(current["score"] - previous["score"], 3)

        if delta >= _ACCEL_THRESHOLD:
            trend = "accelerating"
        elif delta <= _DECEL_THRESHOLD:
            trend = "decelerating"
        else:
            trend = "stable"

        return {
            "current_score": current["score"],
            "previous_score": previous["score"],
            "delta": delta,
            "trend": trend,
            "scans_tracked": len(entries),
            "first_seen": entries[0]["date"],
        }

    def get_acceleration_summary(self, tickers: list[str]) -> str:
        """Generate a momentum trend summary for a list of tickers.

        Returns a formatted multi-line string for council context injection.
        """
        if not tickers:
            return "No momentum trend data available."

        lines = ["MOMENTUM TRENDS:"]
        any_data = False

        for ticker in tickers:
            delta_data = self.get_momentum_delta(ticker)
            if not delta_data:
                continue
            any_data = True
            trend = delta_data.get("trend", "new")
            curr = delta_data.get("current_score")
            prev = delta_data.get("previous_score")
            delta = delta_data.get("delta", 0.0)

            if trend == "new":
                lines.append(f"  - {ticker}: NEW (first appearance, score={curr:.1f})")
            elif trend == "accelerating":
                lines.append(f"  - {ticker}: {prev:.1f} → {curr:.1f} (accelerating ↑, Δ={delta:+.1f})")
            elif trend == "decelerating":
                lines.append(f"  - {ticker}: {prev:.1f} → {curr:.1f} (decelerating ↓, Δ={delta:+.1f})")
            else:
                lines.append(f"  - {ticker}: {prev:.1f} → {curr:.1f} (stable, Δ={delta:+.1f})")

        if not any_data:
            return "No momentum history available yet (first scan)."

        return "\n".join(lines)

    def enrich_ranked_candidates(self, ranked: list[dict]) -> list[dict]:
        """Add momentum_delta and momentum_trend keys to each ranked candidate dict.

        These fields are later used by the funnel Stage 4 scoring to apply
        acceleration bonuses and deceleration penalties.
        """
        enriched = []
        for candidate in ranked:
            ticker = candidate.get("symbol", "").upper()
            delta_data = self.get_momentum_delta(ticker)
            candidate = dict(candidate)  # shallow copy — don't mutate input
            if delta_data:
                candidate["momentum_delta"] = delta_data.get("delta", 0.0)
                candidate["momentum_trend"] = delta_data.get("trend", "new")
            else:
                candidate["momentum_delta"] = 0.0
                candidate["momentum_trend"] = "new"
            enriched.append(candidate)
        return enriched
