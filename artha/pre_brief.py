"""Pre-Brief System — aggregates recent events for per-ticker council context.

Records sentinel alerts, price moves, and momentum events so that council
analysts receive a "recent events" summary before analyzing a stock.
"""
import json
import logging
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_SEVERITY_ICONS = {
    "CRITICAL": "🚨",
    "WARNING": "⚠️",
    "INFO": "📌",
}

_TYPE_ICONS = {
    "news_alert": "📰",
    "price_move": "📉",
    "momentum_acceleration": "🚀",
    "analyst_action": "🏦",
    "earnings": "📊",
}


class PreBrief:
    """Aggregates recent events into a concise pre-brief for council analysis."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.brief_file = data_dir / "pre_briefs.json"

    # ------------------------------------------------------------------
    # Internal I/O helpers
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if not self.brief_file.exists():
            return {"events": [], "last_pruned": None}
        try:
            with open(self.brief_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("[pre_brief] Failed to load %s: %s", self.brief_file, e)
            return {"events": [], "last_pruned": None}

    def _save(self, payload: dict) -> None:
        """Atomic write using tempfile + os.replace."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.data_dir),
            suffix=".tmp",
            prefix=".pre_briefs_",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(self.brief_file))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _prune(self, payload: dict, days: int = 14) -> dict:
        """Remove events older than `days` days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        payload["events"] = [
            e for e in payload.get("events", [])
            if datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")) >= cutoff
        ]
        payload["last_pruned"] = datetime.now(timezone.utc).isoformat()
        return payload

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_event(
        self,
        ticker: str,
        event_type: str,
        severity: str,
        summary: str,
        source: str,
    ) -> None:
        """Record a notable event for a ticker.

        Called by sentinel, monitor, and daily scanner.
        Appends to pre_briefs.json and prunes events older than 14 days.
        """
        payload = self._load()

        event = {
            "ticker": ticker.upper(),
            "event_type": event_type,
            "severity": severity.upper(),
            "summary": summary,
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        payload.setdefault("events", []).append(event)

        # Prune opportunistically (not every call, only when count grows large)
        if len(payload["events"]) > 500:
            payload = self._prune(payload)

        self._save(payload)

    def get_brief(self, ticker: str, days: int = 7) -> str:
        """Generate a pre-brief summary for a ticker covering the last N days.

        Returns formatted text for injection into analyst prompts.
        If no events, returns a short "no notable events" message.
        """
        payload = self._load()
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        ticker_upper = ticker.upper()

        events = [
            e for e in payload.get("events", [])
            if e.get("ticker") == ticker_upper
            and datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")) >= cutoff
        ]

        if not events:
            return f"No notable events for {ticker_upper} in the past {days} days."

        # Sort newest first
        events.sort(key=lambda e: e["timestamp"], reverse=True)

        lines = [f"RECENT EVENTS for {ticker_upper} (last {days} days):"]
        for e in events:
            ts = datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
            date_str = ts.strftime("%b %d")
            sev_icon = _SEVERITY_ICONS.get(e.get("severity", "INFO"), "📌")
            type_icon = _TYPE_ICONS.get(e.get("event_type", ""), "•")
            lines.append(f"  - {date_str}: {sev_icon}{type_icon} {e['summary']} [via {e.get('source', '?')}]")

        return "\n".join(lines)

    def get_council_pre_brief(self, tickers: list[str]) -> str:
        """Generate combined pre-briefs for multiple tickers (used before council session)."""
        parts = []
        for ticker in tickers:
            brief = self.get_brief(ticker)
            parts.append(brief)
        return "\n\n".join(parts) if parts else "No recent events for any candidate tickers."
