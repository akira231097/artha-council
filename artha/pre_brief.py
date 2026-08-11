"""Pre-Brief System — aggregates recent events for per-ticker council context.

Records sentinel alerts, price moves, and momentum events so that council
analysts receive a "recent events" summary before analyzing a stock.
"""
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

from .paths import DATA_DIR

logger = logging.getLogger(__name__)

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
        self.lock_file = data_dir / "pre_briefs.lock"

    # ------------------------------------------------------------------
    # Internal I/O helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _lock(self, *, exclusive: bool) -> Iterator[None]:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.lock_file, "a+", encoding="utf-8") as lock:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _load_unlocked(self) -> dict:
        if not self.brief_file.exists():
            return {"events": [], "last_pruned": None}
        try:
            with open(self.brief_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict) or not isinstance(payload.get("events", []), list):
                raise ValueError("pre-brief payload must be an object with an events list")
            payload.setdefault("events", [])
            payload.setdefault("last_pruned", None)
            return payload
        except Exception as e:
            logger.warning("[pre_brief] Failed to load %s: %s", self.brief_file, e)
            return {"events": [], "last_pruned": None}

    def _load(self) -> dict:
        with self._lock(exclusive=False):
            return self._load_unlocked()

    def _save_unlocked(self, payload: dict) -> None:
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

    def _save(self, payload: dict) -> None:
        with self._lock(exclusive=True):
            self._save_unlocked(payload)

    def _prune(self, payload: dict, days: int = 14) -> dict:
        """Remove events older than `days` days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        payload["events"] = [
            e
            for e in payload.get("events", [])
            if isinstance(e, dict)
            and (parsed := self._parse_timestamp(e.get("timestamp"))) is not None
            and parsed >= cutoff
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
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a notable event for a ticker.

        Called by sentinel, monitor, and daily scanner.
        Appends to pre_briefs.json and prunes events older than 14 days.
        """
        with self._lock(exclusive=True):
            payload = self._load_unlocked()
            event = {
                "ticker": ticker.upper(),
                "event_type": event_type,
                "severity": severity.upper(),
                "summary": summary,
                "source": source,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
            }
            payload.setdefault("events", []).append(event)

            # Prune opportunistically (not every call, only when count grows large)
            if len(payload["events"]) > 500:
                payload = self._prune(payload)

            self._save_unlocked(payload)

    def get_events(
        self,
        ticker: str | None = None,
        *,
        hours: float | None = None,
        days: int | None = None,
        source: str | None = None,
        event_type: str | None = None,
        severity: str | None = None,
        verified_negative: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Return recent structured events using deterministic filters."""
        payload = self._load()
        now = datetime.now(timezone.utc)
        if hours is not None:
            cutoff = now - timedelta(hours=max(0.0, float(hours)))
        elif days is not None:
            cutoff = now - timedelta(days=max(0, int(days)))
        else:
            cutoff = None
        ticker_upper = str(ticker or "").upper().strip()
        source_norm = str(source or "").lower().strip()
        type_norm = str(event_type or "").lower().strip()
        severity_norm = str(severity or "").upper().strip()

        events: list[dict[str, Any]] = []
        for raw in payload.get("events", []):
            if not isinstance(raw, dict):
                continue
            ts = self._parse_timestamp(raw.get("timestamp"))
            if cutoff is not None and (ts is None or ts < cutoff):
                continue
            if ticker_upper and str(raw.get("ticker") or "").upper() != ticker_upper:
                continue
            if source_norm and str(raw.get("source") or "").lower() != source_norm:
                continue
            if type_norm and str(raw.get("event_type") or "").lower() != type_norm:
                continue
            if severity_norm and str(raw.get("severity") or "").upper() != severity_norm:
                continue
            metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
            verified_value = metadata.get("verified_negative")
            if isinstance(verified_value, bool):
                is_verified_negative = verified_value
            else:
                is_verified_negative = str(verified_value or "").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "y",
                }
            if verified_negative is not None and is_verified_negative != verified_negative:
                continue
            events.append(dict(raw))
        events.sort(key=lambda event: str(event.get("timestamp") or ""), reverse=True)
        return events

    def get_brief(self, ticker: str, days: int = 7) -> str:
        """Generate a pre-brief summary for a ticker covering the last N days.

        Returns formatted text for injection into analyst prompts.
        If no events, returns a short "no notable events" message.
        """
        ticker_upper = ticker.upper()
        events = self.get_events(ticker_upper, days=days)

        if not events:
            return f"No notable events for {ticker_upper} in the past {days} days."

        lines = [f"RECENT EVENTS for {ticker_upper} (last {days} days):"]
        for e in events:
            ts = self._parse_timestamp(e.get("timestamp")) or datetime.now(timezone.utc)
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
