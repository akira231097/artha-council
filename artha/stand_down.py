"""Buy-side pause for Artha day-scoped capacity limits.

When a buy capacity limit is hit (daily order cap, daily notional cap,
monthly budget, insufficient cash/buying power), Artha pauses NEW BUYS
until the next US trading day and tells the operator once on Telegram.

This deliberately does NOT touch the global kill switch, launchd services,
or OpenClaw crons: the runner cron drains auto_sell as well as auto_buy, so
disabling it (or the monitor daemon) would leave open positions without
stop-loss protection. A buy limit is a budget event, not an emergency —
sell-side protection must keep running whenever positions are open.

Per-order sizing blocks ("exceeds the $25 pilot cap", "max order cap",
"account pilot cap" — all per-order notional checks) and per-ticker
concentration blocks are NOT capacity limits and never pause the day: they
only mean that one order needs resizing or that one ticker is full.

Write discipline for control.json (shared with the operator kill switch):
- every read-modify-write happens under an exclusive flock, and
  set_trading_disabled() in robinhood_bridge uses the same lock — a pause
  can never clobber a concurrently-set kill switch (or vice versa);
- an UNREADABLE control file is never overwritten: readers fail closed
  (trading_disabled=True) and the pause refuses to "heal" the file, which
  would silently erase whatever the operator had put in it;
- once today's pause is recorded, repeat triggers return early without
  rewriting the file or re-sending Telegram.
"""
from __future__ import annotations

import fcntl
import json
import logging
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

from .config import Config
from .telegram import TelegramSender

logger = logging.getLogger(__name__)

_EASTERN = ZoneInfo("America/New_York")

# Substrings (lowercased) of guardrail reasons that mean "no more buy
# capacity today/this month" — day-scoped by construction. Per-order sizing
# messages must never appear here.
BUY_CAPACITY_MARKERS = (
    "daily buy order cap reached",
    "auto-buy daily cap would be exceeded",
    "monthly budget exceeded",
    "insufficient cash available",
    "insufficient robinhood cash/buying power",
)

# Reasons the pause itself emits at the buy gates. They embed the original
# capacity reason, so they would re-match the markers and re-trigger the
# pause forever (nesting the reason string each time) unless filtered out.
_PAUSE_ECHO_PREFIX = "Buy-side paused"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _eastern_today() -> date:
    return datetime.now(timezone.utc).astimezone(_EASTERN).date()


@contextmanager
def control_file_lock(control_path: Path | None = None) -> Iterator[None]:
    """Exclusive advisory lock serializing every control.json writer."""
    target = Path(control_path or Config.ROBINHOOD_CONTROL_FILE).expanduser()
    lock_path = target.with_name(target.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_control_file(path: Path) -> tuple[dict[str, Any] | None, str]:
    """Read control.json, distinguishing missing from unreadable.

    Returns (payload, status) with status in {"ok", "missing", "unreadable"}.
    Payload is None only for "unreadable" — callers must NOT overwrite the
    file in that case (the readers treat it as fail-closed kill switch).
    """
    if not path.exists():
        return {}, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("[buy_pause] control file unreadable (%s): %s", path, exc)
        return None, "unreadable"
    if not isinstance(payload, dict):
        logger.error("[buy_pause] control file is not a JSON object (%s)", path)
        return None, "unreadable"
    return payload, "ok"


def _limit_summary(reason: str, source: str = "") -> str:
    source_text = f" Source: {source}." if source else ""
    return f"{reason.strip() or 'A buy-side Artha capacity limit was reached.'}{source_text}".strip()


def reason_is_buy_capacity_limit(reason: str) -> bool:
    text = str(reason or "")
    if text.startswith(_PAUSE_ECHO_PREFIX):
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in BUY_CAPACITY_MARKERS)


def reasons_include_buy_capacity_limit(reasons: list[str] | tuple[str, ...]) -> bool:
    return any(reason_is_buy_capacity_limit(str(reason)) for reason in reasons or [])


def buying_paused(control: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Return (paused, reason). The pause auto-expires after its ET date."""
    if control is None:
        payload, status = read_control_file(Path(Config.ROBINHOOD_CONTROL_FILE).expanduser())
        control = payload if status == "ok" else {}
    raw_until = str(control.get("buying_paused_until") or "").strip()
    if not raw_until:
        return False, ""
    try:
        until = date.fromisoformat(raw_until[:10])
    except ValueError:
        logger.warning("[buy_pause] unparseable buying_paused_until=%r; ignoring", raw_until)
        return False, ""
    if _eastern_today() > until:
        return False, ""
    reason = str(control.get("buying_paused_reason") or "Buy capacity limit reached earlier today.")
    return True, reason


def format_buy_pause_message(reason: str, *, source: str = "") -> str:
    summary = _limit_summary(reason, source)
    return (
        "🟡 Buys paused for today — Artha reached a buy-side limit.\n"
        f"• **Reason:** {summary}\n"
        "• **New buys:** paused until the next trading day (auto-resumes)\n"
        "• **Sells & stop-losses:** ACTIVE — portfolio protection unaffected\n"
        "• **Services:** all running normally\n"
        "👉 You: nothing needed."
    )


def pause_buying(
    reason: str,
    *,
    source: str = "",
    send_telegram: bool = True,
    telegram_sender: TelegramSender | None = None,
) -> dict[str, Any]:
    """Persist a buys-only pause through the end of today (ET), notify once.

    Behaviour guarantees (all verified by tests):
    - never modifies trading_disabled or any other key it does not own;
    - refuses to touch an unreadable control file (fail-closed stands);
    - if a pause through today or later already exists, returns early with
      no file write and no Telegram;
    - the whole read-modify-write runs under control_file_lock.
    """
    if not Config.ARTHA_AUTO_PAUSE_BUYS_ON_LIMIT:
        return {"success": True, "operation": "buy_pause_disabled", "reason": reason}

    control_path = Path(Config.ROBINHOOD_CONTROL_FILE).expanduser()
    today = _eastern_today()
    summary = _limit_summary(reason, source)

    with control_file_lock(control_path):
        control, status = read_control_file(control_path)
        if status == "unreadable":
            return {
                "success": False,
                "operation": "buy_pause_refused_unreadable_control",
                "reason": summary,
                "control_path": str(control_path),
            }

        existing_until: date | None = None
        raw_until = str(control.get("buying_paused_until") or "").strip()
        if raw_until:
            try:
                existing_until = date.fromisoformat(raw_until[:10])
            except ValueError:
                existing_until = None

        if existing_until is not None and existing_until >= today:
            # Already paused through today (or a future date the operator
            # set) — nothing to write, nothing to send.
            return {
                "success": True,
                "operation": "buy_pause",
                "reason": str(control.get("buying_paused_reason") or summary),
                "buying_paused_until": existing_until.isoformat(),
                "control_path": str(control_path),
                "telegram_sent": False,
                "telegram_suppressed_duplicate": True,
            }

        control.update(
            {
                "buying_paused_until": today.isoformat(),
                "buying_paused_reason": summary,
                "buying_paused_source": source,
                "updated_at": _utcnow_iso(),
            }
        )
        _atomic_write_json(control_path, control)

    telegram_sent = False
    if send_telegram:
        try:
            sender = telegram_sender or TelegramSender()
            telegram_sent = bool(
                sender.send_message(
                    format_buy_pause_message(reason, source=source),
                    parse_mode=None,
                    silent=False,
                )
            )
        except Exception as exc:
            logger.warning("[buy_pause] Telegram notification failed: %s", exc)

    return {
        "success": True,
        "operation": "buy_pause",
        "reason": summary,
        "buying_paused_until": today.isoformat(),
        "control_path": str(control_path),
        "telegram_sent": telegram_sent,
        "telegram_suppressed_duplicate": False,
    }


def maybe_pause_buying_for_limit(
    reasons: list[str] | tuple[str, ...],
    *,
    source: str,
    send_telegram: bool = True,
) -> dict[str, Any] | None:
    """Pause buys for the day if a guardrail block is a buy capacity limit."""
    if not Config.ARTHA_AUTO_PAUSE_BUYS_ON_LIMIT:
        return None
    if not reasons_include_buy_capacity_limit(reasons):
        return None
    reason = next(
        (str(r) for r in reasons if reason_is_buy_capacity_limit(str(r))),
        "Buy-side capacity limit reached.",
    )
    return pause_buying(reason, source=source, send_telegram=send_telegram)
