"""Artha Supervisor v1.

The supervisor watches the investing machine itself. It does not make stock
decisions and it does not change investing rules. It checks whether Artha's
research, logging, diagnosis, watchlists, shadow tests, and Telegram reporting
are in place and healthy.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .calibration import backfill_decision_features, build_calibration_report
from .config import Config
from .diagnostics import run_calibration_diagnosis
from .execution import build_execution_readiness_report, normalize_robinhood_position_snapshot
from .execution_learning import build_execution_learning_summary
from .journal import DecisionJournal
from .paths import DATA_DIR
from .portfolio import PORTFOLIO_FILE, Portfolio
from .shadow_rules import (
    backfill_shadow_rules_from_features,
    summarize_shadow_rules,
    update_shadow_rule_outcomes,
)
from .telegram import TelegramSender

logger = logging.getLogger(__name__)

SUPERVISOR_DIR = DATA_DIR / "supervisor"
LOG_DIR = DATA_DIR / "logs"
RANK_COVERAGE_DIR = DATA_DIR / "rank_coverage"

LOG_ERROR_RE = re.compile(
    r"(?:^|[\s\[])(?:ERROR|CRITICAL)(?:[\s\]:-]|$)|Traceback|Exception",
    re.IGNORECASE,
)
LOG_WARNING_RE = re.compile(r"(?:^|[\s\[])(?:WARNING)(?:[\s\]:-]|$)", re.IGNORECASE)
EXPECTED_TRANSIENT_LOG_PATTERNS = (
    # Expected operational events (guardrails and known data-plan gaps doing
    # their job) — reported through their own channels, not log-health WARNs.
    "Hard Risk Gate FAILED",
    "Missing data: analyst_estimates",
    "ChatGPT backend rejected temperature",
    "Failed to fetch article",
    "403 Client Error",
    "404 Client Error",
    "Skipping non-HTML content",
    "Read timed out",
    "ConnectTimeoutError",
    "No route to host",
    "EHOSTUNREACH",
    "ENETUNREACH",
    "ranking budget exhausted",
    "Short interest unavailable",
)
QUALITY_DEGRADING_LOG_PATTERNS = (
    "Scoring JSON failed schema validation",
    "No valid scoring JSON found",
    "Serper search failed",
    "Telegram connection error",
    "Telegram API error",
    "Failed to send Telegram",
)
BENIGN_DASHBOARD_DISCONNECT_ERRORS = (
    "ConnectionResetError",
    "BrokenPipeError",
    "ConnectionAbortedError",
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat()


def _parse_dt(value: Any) -> datetime | None:
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


def _is_regular_market_open_now() -> bool:
    """Best-effort US regular market-hours check for broker freshness severity."""
    try:
        from .scheduler import MarketHours

        return bool(MarketHours().is_market_open(_utcnow()))
    except Exception:
        return True


def _warnings_are_stale_snapshot_only(warnings: list[Any]) -> bool:
    if not warnings:
        return False
    return all("snapshot is stale" in str(warning).lower() for warning in warnings)


def _age_hours(value: Any) -> float | None:
    dt = _parse_dt(value)
    if not dt:
        return None
    return (_utcnow() - dt).total_seconds() / 3600.0


def _timed_check(name: str, fn) -> dict[str, Any]:
    """Run one supervisor check with timing and exception capture."""
    started = time.monotonic()
    try:
        result = fn()
        if not isinstance(result, dict):
            result = {"name": name, "status": "FAIL", "message": "Check returned invalid result."}
    except Exception as exc:
        result = {"name": name, "status": "FAIL", "message": f"Check crashed: {exc}"}
    result.setdefault("name", name)
    result["duration_ms"] = round((time.monotonic() - started) * 1000, 1)
    logger.info(
        "[supervisor] check=%s status=%s duration_ms=%.1f",
        result.get("name"),
        result.get("status"),
        result["duration_ms"],
    )
    return result


def _timed_operation(name: str, fn) -> dict[str, Any]:
    """Run a maintenance operation and return structured timing/result data."""
    started = time.monotonic()
    try:
        result = fn()
        status = "PASS"
        error = ""
    except Exception as exc:
        logger.warning("[supervisor] operation=%s failed: %s", name, exc)
        result = {}
        status = "FAIL"
        error = str(exc)
    duration_ms = round((time.monotonic() - started) * 1000, 1)
    logger.info("[supervisor] operation=%s status=%s duration_ms=%.1f", name, status, duration_ms)
    return {
        "name": name,
        "status": status,
        "duration_ms": duration_ms,
        "result": result,
        "error": error,
    }


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _check_database(journal: DecisionJournal) -> dict[str, Any]:
    try:
        with journal._connect() as conn:
            conn.execute("SELECT 1").fetchone()
        return {"name": "database", "status": "PASS", "message": "SQLite is reachable."}
    except Exception as exc:
        return {"name": "database", "status": "FAIL", "message": f"SQLite failed: {exc}"}


def _check_latest_decision_artifacts(journal: DecisionJournal) -> dict[str, Any]:
    features = journal.get_decision_features(limit=5)
    if not features:
        return {"name": "decision_artifacts", "status": "WARN", "message": "No decision dossiers/features found yet."}
    latest = features[0]
    path = Path(str(latest.get("dossier_path") or ""))
    if not path.exists():
        return {
            "name": "decision_artifacts",
            "status": "FAIL",
            "message": f"Latest dossier is missing on disk: {path}",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "name": "decision_artifacts",
            "status": "FAIL",
            "message": f"Latest dossier cannot be read: {exc}",
        }
    evidence_count = int((payload.get("source_audit") or {}).get("evidence_count") or 0)
    source_count = len((payload.get("source_audit") or {}).get("source_counts") or {})
    age = _age_hours(payload.get("generated_at"))
    if evidence_count < 10:
        status = "WARN"
        message = f"Latest dossier has low evidence count ({evidence_count})."
    else:
        status = "PASS"
        message = f"Latest dossier exists with {evidence_count} evidence items and {source_count} source types."
    return {
        "name": "decision_artifacts",
        "status": status,
        "message": message,
        "ticker": latest.get("ticker"),
        "dossier_path": str(path),
        "age_hours": round(age, 2) if age is not None else None,
    }


def _check_rank_coverage() -> dict[str, Any]:
    """Verify that a recent full-universe rank run examined enough symbols."""
    paths = sorted(RANK_COVERAGE_DIR.glob("*/*.json"), reverse=True)[:100]
    candidates: list[tuple[datetime, int, Path, dict[str, Any]]] = []
    for path in paths:
        payload = _read_json_file(path)
        if not payload:
            continue
        generated = _parse_dt(payload.get("generated_at"))
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        try:
            universe_size = int(summary.get("universe_size") or 0)
        except (TypeError, ValueError):
            universe_size = 0
        if generated and universe_size:
            candidates.append((generated, universe_size, path, summary))
    if not candidates:
        return {
            "name": "rank_coverage",
            "status": "WARN",
            "message": "No immutable ranking-coverage audit exists yet.",
        }

    now = datetime.now(timezone.utc)
    max_age_hours = 84.0  # Covers the Friday-to-Monday market closure.
    recent = [
        row
        for row in candidates
        if 0 <= (now - row[0]).total_seconds() <= max_age_hours * 3600
    ]
    if recent:
        generated, universe_size, path, summary = max(recent, key=lambda row: (row[1], row[0]))
    else:
        generated, universe_size, path, summary = max(candidates, key=lambda row: row[0])
    coverage = float(summary.get("history_coverage_pct") or 0.0)
    minimum = float(getattr(Config, "RANK_MIN_HISTORY_COVERAGE_PCT", 0.90) or 0.90)
    age_hours = max(0.0, (now - generated).total_seconds() / 3600.0)
    status = "PASS" if coverage >= minimum and age_hours <= max_age_hours else "WARN"
    return {
        "name": "rank_coverage",
        "status": status,
        "message": (
            f"Latest full ranking run obtained usable history for "
            f"{int(summary.get('history_count') or 0)}/{universe_size} stocks "
            f"({coverage:.1%}); required {minimum:.0%}; age {age_hours:.1f}h."
        ),
        "artifact_path": str(path),
        "generated_at": generated.isoformat(),
        "coverage_pct": coverage,
        "minimum_coverage_pct": minimum,
        "age_hours": round(age_hours, 2),
        "maximum_age_hours": max_age_hours,
        "status_counts": summary.get("status_counts") or {},
    }


def _check_latest_report_artifact(journal: DecisionJournal) -> dict[str, Any]:
    """Verify the latest session report file exists and has readable content."""
    try:
        with journal._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM sessions
                ORDER BY datetime(timestamp) DESC, id DESC
                LIMIT 1
                """
            ).fetchone()
    except Exception as exc:
        return {"name": "latest_report", "status": "FAIL", "message": f"Could not query latest session: {exc}"}
    if not row:
        return {"name": "latest_report", "status": "WARN", "message": "No session exists to verify a report file."}

    latest = dict(row)
    raw_paths = str(latest.get("report_path") or "").strip()
    raw_items = [p.strip() for p in raw_paths.split(",") if p.strip()]
    virtual_items = [p for p in raw_items if p.lower() == "telegram"]
    paths = [Path(p) for p in raw_items if p.lower() != "telegram"]
    if not paths:
        if virtual_items:
            return {
                "name": "latest_report",
                "status": "PASS",
                "message": "Latest report was delivered through Telegram; no local file was expected.",
                "virtual_paths": virtual_items,
            }
        return {
            "name": "latest_report",
            "status": "WARN",
            "message": f"Latest session {latest.get('session_type')} has no report path recorded.",
        }

    existing: list[str] = []
    missing: list[str] = []
    too_small: list[str] = []
    for path in paths[:8]:
        if not path.exists():
            missing.append(str(path))
            continue
        try:
            size = path.stat().st_size
            snippet = path.read_text(encoding="utf-8", errors="replace")[:200]
        except Exception:
            missing.append(str(path))
            continue
        if size < 200 or not snippet.strip():
            too_small.append(str(path))
        else:
            existing.append(str(path))

    if missing:
        return {
            "name": "latest_report",
            "status": "FAIL",
            "message": f"{len(missing)} latest report path(s) missing or unreadable.",
            "missing": missing[:3],
            "ok_paths": existing[:3],
        }
    if too_small:
        return {
            "name": "latest_report",
            "status": "WARN",
            "message": f"{len(too_small)} latest report file(s) look too small to be useful.",
            "small_paths": too_small[:3],
            "ok_paths": existing[:3],
        }
    return {
        "name": "latest_report",
        "status": "PASS",
        "message": f"{len(existing)} latest report file(s) exist and are readable.",
        "ok_paths": existing[:3],
    }


def _check_agentic_trace_artifact(journal: DecisionJournal) -> dict[str, Any]:
    """Verify latest agentic trace exists, is readable, and has useful evidence."""
    features = journal.get_decision_features(limit=1)
    if not features:
        return {"name": "agentic_trace", "status": "WARN", "message": "No decision features found for trace check."}
    dossier_path = Path(str(features[0].get("dossier_path") or ""))
    if not dossier_path.exists():
        return {"name": "agentic_trace", "status": "FAIL", "message": f"Latest dossier missing: {dossier_path}"}
    dossier = _read_json_file(dossier_path)
    if not dossier:
        return {"name": "agentic_trace", "status": "FAIL", "message": f"Latest dossier is not valid JSON: {dossier_path}"}
    agentic = dossier.get("agentic_trace") or {}
    if not agentic.get("enabled"):
        return {"name": "agentic_trace", "status": "WARN", "message": "Latest decision did not have agentic diligence enabled."}
    trace_path = Path(str(agentic.get("trace_path") or ""))
    if not trace_path.exists():
        return {"name": "agentic_trace", "status": "FAIL", "message": f"Agentic trace file missing: {trace_path}"}
    trace = _read_json_file(trace_path)
    if not trace:
        return {"name": "agentic_trace", "status": "FAIL", "message": f"Agentic trace is not valid JSON: {trace_path}"}
    evidence = trace.get("evidence") or []
    role_plans = trace.get("role_plans") or {}
    role_queries = trace.get("role_queries") or {}
    gaps = trace.get("gaps") or agentic.get("gaps") or []
    conflicts = trace.get("conflicts") or agentic.get("conflicts") or []
    if len(evidence) < 10:
        status = "WARN"
        message = f"Agentic trace is readable but thin: {len(evidence)} evidence item(s)."
    elif len(role_plans) < 3 or len(role_queries) < 3:
        status = "WARN"
        message = "Agentic trace is readable but does not show all three role plans/queries."
    else:
        status = "PASS"
        message = f"Agentic trace has {len(evidence)} evidence item(s), {len(gaps)} gap(s), {len(conflicts)} conflict(s)."
    return {
        "name": "agentic_trace",
        "status": status,
        "message": message,
        "trace_path": str(trace_path),
        "evidence_count": len(evidence),
        "gap_count": len(gaps),
        "conflict_count": len(conflicts),
        "role_count": len(role_plans),
    }


def _check_intelligence_routing(journal: DecisionJournal) -> dict[str, Any]:
    """Confirm ambiguous investment judgment used the LLM council/evidence path."""
    features = journal.get_decision_features(limit=1)
    if not features:
        return {"name": "intelligence_routing", "status": "WARN", "message": "No decision features found."}
    dossier_path = Path(str(features[0].get("dossier_path") or ""))
    dossier = _read_json_file(dossier_path) if dossier_path.exists() else None
    if not dossier:
        return {
            "name": "intelligence_routing",
            "status": "FAIL",
            "message": f"Cannot inspect latest dossier for intelligence routing: {dossier_path}",
        }
    analysts = dossier.get("analysts") or {}
    expected_roles = {"fundamental", "technical", "contrarian"}
    roles_present = {role for role in analysts if role in expected_roles}
    models = {
        role: str((payload or {}).get("model") or "")
        for role, payload in analysts.items()
        if role in expected_roles and isinstance(payload, dict)
    }
    agentic_enabled = bool((dossier.get("agentic_trace") or {}).get("enabled"))
    decision_text = json.dumps(dossier.get("decision") or {}, ensure_ascii=True)
    analyst_text = json.dumps(analysts, ensure_ascii=True)
    citation_count = len(re.findall(r"\[E\d{3}\]", decision_text + " " + analyst_text))

    missing_roles = sorted(expected_roles - roles_present)
    missing_models = sorted(role for role in expected_roles if not models.get(role))
    if not agentic_enabled:
        status = "WARN"
        message = "Latest decision used the council but did not show agentic diligence enabled."
    elif missing_roles or missing_models:
        status = "WARN"
        message = f"Latest decision is missing role/model evidence: roles={missing_roles}, models={missing_models}."
    elif citation_count < 5:
        status = "WARN"
        message = f"Latest LLM decision has low evidence citation density ({citation_count} evidence citation(s))."
    else:
        status = "PASS"
        message = (
            "Latest decision used agentic diligence plus 3 LLM analyst roles "
            f"with {citation_count} evidence citation(s)."
        )
    return {
        "name": "intelligence_routing",
        "status": status,
        "message": message,
        "agentic_enabled": agentic_enabled,
        "roles_present": sorted(roles_present),
        "models": models,
        "evidence_citation_count": citation_count,
    }


def _check_defer_watches(journal: DecisionJournal) -> dict[str, Any]:
    try:
        expired = journal.expire_defer_watches()
        invalidated = journal.invalidate_implausible_defer_watches()
        requeued = journal.requeue_stale_defer_auto_reviews(
            Config.DEFER_AUTO_REVIEW_STALE_REVIEW_MINUTES,
        )
        failed_requeued = journal.requeue_failed_defer_auto_reviews(
            Config.DEFER_AUTO_REVIEW_FAILED_RETRY_MINUTES,
            Config.DEFER_AUTO_REVIEW_MAX_FAILURES,
        )
        watches = journal.get_active_defer_watches()
        recent = journal.get_defer_watches(limit=50)
        status_counts: dict[str, int] = {}
        stuck_reviewing: list[str] = []
        recent_failures: list[str] = []
        for row in recent:
            status = str(row.get("status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            ticker = str(row.get("ticker") or "")
            watch_id = str(row.get("watch_id") or "")
            age = _age_hours(row.get("updated_at"))
            if status == "triggered_reviewing" and age is not None and age > 2:
                stuck_reviewing.append(f"{ticker}:{watch_id}")
            if status == "review_failed" and age is not None and age <= 48:
                recent_failures.append(f"{ticker}:{watch_id}")

        status = "PASS"
        message = (
            f"{len(watches)} active DEFER/WATCH entry watch(es); expired {expired} stale watch(es); "
            f"invalidated {invalidated} implausible watch(es); requeued {requeued} stale review(s) "
            f"and {failed_requeued} bounded failed review(s). "
            f"Auto-review enabled={Config.DEFER_AUTO_REVIEW_ENABLED}, "
            f"cycle cap={Config.DEFER_AUTO_REVIEW_MAX_PER_CYCLE}, "
            f"Robinhood review prep={Config.DEFER_AUTO_REVIEW_PREPARE_ROBINHOOD_REVIEW}."
        )
        if not Config.DEFER_AUTO_REVIEW_ENABLED:
            status = "WARN"
            message += " DEFER trigger auto-review is disabled."
        if stuck_reviewing:
            status = "WARN"
            message += f" Stuck reviewing watch(es): {', '.join(stuck_reviewing[:5])}."
        if recent_failures:
            status = "WARN"
            message += f" Recent failed auto-review watch(es): {', '.join(recent_failures[:5])}."
        return {
            "name": "defer_watchlist",
            "status": status,
            "message": message,
            "active_count": len(watches),
            "expired_count": expired,
            "invalidated_count": invalidated,
            "requeued_count": requeued,
            "failed_requeued_count": failed_requeued,
            "recent_status_counts": status_counts,
            "auto_review": {
                "enabled": Config.DEFER_AUTO_REVIEW_ENABLED,
                "max_per_cycle": Config.DEFER_AUTO_REVIEW_MAX_PER_CYCLE,
                "prepare_robinhood_review": Config.DEFER_AUTO_REVIEW_PREPARE_ROBINHOOD_REVIEW,
                "legacy_trigger_lookback_hours": Config.DEFER_AUTO_REVIEW_LEGACY_TRIGGER_LOOKBACK_HOURS,
                "buy_verdicts": list(Config.DEFER_AUTO_REVIEW_BUY_VERDICTS),
            },
            "stuck_reviewing": stuck_reviewing,
            "recent_failures": recent_failures,
        }
    except Exception as exc:
        return {"name": "defer_watchlist", "status": "FAIL", "message": f"DEFER watch check failed: {exc}"}


def _check_position_monitoring(journal: DecisionJournal) -> dict[str, Any]:
    """Verify held positions are actually protected by sell-monitored theses."""
    try:
        portfolio = Portfolio.load(PORTFOLIO_FILE)
        position_tickers = {
            str(getattr(pos, "ticker", "") or "").upper()
            for pos in (portfolio.positions or [])
            if str(getattr(pos, "ticker", "") or "").strip()
        }
        active_rows = journal.get_all_active_theses()
        monitoring_rows = (
            journal.get_all_sell_monitored_theses()
            if hasattr(journal, "get_all_sell_monitored_theses")
            else active_rows
        )
        active_tickers = {
            str(row.get("ticker") or "").upper()
            for row in active_rows
            if str(row.get("ticker") or "").strip()
        }
        monitored_tickers = {
            str(row.get("ticker") or "").upper()
            for row in monitoring_rows
            if str(row.get("ticker") or "").strip()
        }
        pending_exit_tickers = sorted({
            str(row.get("ticker") or "").upper()
            for row in monitoring_rows
            if str(row.get("status") or "").lower() == "pending_exit"
            and str(row.get("ticker") or "").strip()
        })
        pending_rows = journal.get_pending_theses()
        pending_tickers = sorted({
            str(row.get("ticker") or "").upper()
            for row in pending_rows
            if str(row.get("ticker") or "").strip()
        })

        unmonitored_positions = sorted(position_tickers - monitored_tickers)
        orphan_active_theses = sorted(monitored_tickers - position_tickers)
        live_sell_statuses = {
            "review_ready", "review_clear", "review_requested",
            "auto_review_requested", "reviewed", "place_requested",
            "submitted", "partially_filled",
        }
        now = _utcnow()
        sell_actions = []
        for row in journal.get_trade_actions(limit=500):
            if str(row.get("action_type") or "").lower() != "auto_sell":
                continue
            if str(row.get("status") or "").lower() not in live_sell_statuses:
                continue
            expires_at = _parse_dt(row.get("expires_at"))
            status = str(row.get("status") or "").lower()
            if expires_at and expires_at < now and status not in {"submitted", "partially_filled"}:
                continue
            sell_actions.append(row)
        live_by_thesis: dict[str, list[dict[str, Any]]] = {}
        live_by_ticker: dict[str, list[dict[str, Any]]] = {}
        for row in sell_actions:
            live_by_thesis.setdefault(str(row.get("thesis_id") or ""), []).append(row)
            live_by_ticker.setdefault(str(row.get("ticker") or "").upper(), []).append(row)
        pending_rows_by_id = {
            str(row.get("thesis_id") or ""): row
            for row in monitoring_rows
            if str(row.get("status") or "").lower() == "pending_exit"
        }
        pending_without_auto_sell = sorted(
            str(row.get("ticker") or "").upper()
            for thesis_id, row in pending_rows_by_id.items()
            if not live_by_thesis.get(thesis_id)
        )
        duplicate_live_sells = sorted(
            ticker for ticker, rows in live_by_ticker.items() if ticker and len(rows) > 1
        )
        malformed_thesis_metadata: list[str] = []
        for row in monitoring_rows:
            ticker = str(row.get("ticker") or "").upper()
            summary = str(row.get("thesis_summary") or "").strip()
            summary_upper = summary.upper()
            if (
                not summary
                or summary.startswith("```")
                or summary in {"---", "***"}
                or summary_upper.startswith("PRIMARY DRIVER:")
            ):
                malformed_thesis_metadata.append(f"{ticker}:missing_or_malformed_summary")
            raw_conditions = row.get("invalidation_conditions") or "[]"
            try:
                conditions = (
                    json.loads(raw_conditions)
                    if isinstance(raw_conditions, str)
                    else raw_conditions
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                conditions = []
            if not any(
                isinstance(condition, str) and condition.strip()
                for condition in (conditions if isinstance(conditions, list) else [])
            ):
                malformed_thesis_metadata.append(f"{ticker}:missing_business_invalidation")

        if unmonitored_positions:
            return {
                "name": "position_monitoring",
                "status": "FAIL",
                "message": (
                    "Held portfolio position(s) are missing sell-monitored theses: "
                    f"{', '.join(unmonitored_positions[:8])}. Sell monitoring is not safe until reconciled."
                ),
                "portfolio_positions": sorted(position_tickers),
                "active_theses": sorted(active_tickers),
                "pending_exit_theses": pending_exit_tickers,
                "pending_theses": pending_tickers,
                "unmonitored_positions": unmonitored_positions,
                "orphan_active_theses": orphan_active_theses,
            }

        if pending_without_auto_sell:
            return {
                "name": "position_monitoring",
                "status": "FAIL",
                "message": (
                    "Pending-exit position(s) have no drainable Council-approved auto-sell: "
                    f"{', '.join(pending_without_auto_sell[:8])}."
                ),
                "portfolio_positions": sorted(position_tickers),
                "active_theses": sorted(active_tickers),
                "pending_exit_theses": pending_exit_tickers,
                "pending_exit_without_auto_sell": pending_without_auto_sell,
                "duplicate_live_sells": duplicate_live_sells,
            }

        if duplicate_live_sells:
            return {
                "name": "position_monitoring",
                "status": "FAIL",
                "message": (
                    "Multiple live auto-sell actions exist for the same ticker: "
                    f"{', '.join(duplicate_live_sells[:8])}."
                ),
                "portfolio_positions": sorted(position_tickers),
                "pending_exit_theses": pending_exit_tickers,
                "pending_exit_without_auto_sell": [],
                "duplicate_live_sells": duplicate_live_sells,
            }

        if orphan_active_theses:
            return {
                "name": "position_monitoring",
                "status": "WARN",
                "message": (
                    "Sell-monitored theses exist without matching portfolio positions: "
                    f"{', '.join(orphan_active_theses[:8])}. Artha may be watching stale holdings."
                ),
                "portfolio_positions": sorted(position_tickers),
                "active_theses": sorted(active_tickers),
                "pending_exit_theses": pending_exit_tickers,
                "pending_theses": pending_tickers,
                "unmonitored_positions": unmonitored_positions,
                "orphan_active_theses": orphan_active_theses,
            }

        if malformed_thesis_metadata:
            return {
                "name": "position_monitoring",
                "status": "WARN",
                "message": (
                    "Held thesis metadata is incomplete for Sell Council: "
                    f"{', '.join(malformed_thesis_metadata[:8])}."
                ),
                "portfolio_positions": sorted(position_tickers),
                "active_theses": sorted(active_tickers),
                "pending_exit_theses": pending_exit_tickers,
                "malformed_thesis_metadata": malformed_thesis_metadata,
            }

        if not position_tickers and pending_tickers:
            return {
                "name": "position_monitoring",
                "status": "WARN",
                "message": (
                    f"No active holdings are being sell-monitored yet; {len(pending_tickers)} pending buy thesis/theses "
                    f"exist ({', '.join(pending_tickers[:8])}). After any real buy, record it immediately with "
                    "python -m artha.cli.portfolio_update buy or activate-thesis."
                ),
                "portfolio_positions": [],
                "active_theses": [],
                "pending_exit_theses": pending_exit_tickers,
                "pending_theses": pending_tickers,
                "unmonitored_positions": [],
                "orphan_active_theses": [],
            }

        if not position_tickers:
            message = "No active holdings; sell engine is idle by design."
        else:
            message = f"{len(position_tickers)} held position(s) have matching sell-monitored theses."
        return {
            "name": "position_monitoring",
            "status": "PASS",
            "message": message,
            "portfolio_positions": sorted(position_tickers),
            "active_theses": sorted(active_tickers),
            "pending_exit_theses": pending_exit_tickers,
            "pending_theses": pending_tickers,
            "unmonitored_positions": [],
            "orphan_active_theses": [],
            "pending_exit_without_auto_sell": [],
            "duplicate_live_sells": [],
            "malformed_thesis_metadata": [],
        }
    except Exception as exc:
        return {"name": "position_monitoring", "status": "FAIL", "message": f"Position monitoring check failed: {exc}"}


def _check_broker_reconciliation_snapshot() -> dict[str, Any]:
    """Verify Robinhood reconciliation has a fresh, account-checked snapshot when holdings exist."""
    try:
        portfolio = Portfolio.load(PORTFOLIO_FILE)
        held_count = len(portfolio.positions or [])
        snapshot_file = str(Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE or "").strip()
        if not Config.ROBINHOOD_RECONCILIATION_ENABLED:
            return {
                "name": "broker_reconciliation",
                "status": "WARN" if held_count else "PASS",
                "message": "Robinhood reconciliation is disabled.",
                "held_count": held_count,
            }
        if not snapshot_file:
            return {
                "name": "broker_reconciliation",
                "status": "WARN" if held_count else "PASS",
                "message": "Robinhood reconciliation snapshot path is not configured.",
                "held_count": held_count,
            }
        path = Path(snapshot_file).expanduser()
        if not path.exists():
            return {
                "name": "broker_reconciliation",
                "status": "WARN" if held_count else "PASS",
                "message": (
                    "No Robinhood snapshot exists yet. This is acceptable with no active holdings; "
                    "it is not acceptable once real positions exist."
                ),
                "path": str(path),
                "held_count": held_count,
            }
        payload = json.loads(path.read_text(encoding="utf-8"))
        snapshot = normalize_robinhood_position_snapshot(payload)
        warnings = snapshot.get("warnings") or []
        status = "PASS"
        stale_only_outside_market = (
            held_count > 0
            and _warnings_are_stale_snapshot_only(warnings)
            and not _is_regular_market_open_now()
        )
        if warnings and held_count and not stale_only_outside_market:
            status = "WARN"
        elif warnings and not stale_only_outside_market:
            status = "WARN"
        message = (
            f"Robinhood snapshot status={snapshot.get('status')}, positions={snapshot.get('position_count')}, "
            f"age={snapshot.get('age_minutes')} min, fresh={snapshot.get('fresh')}."
        )
        if stale_only_outside_market:
            message += " Stale snapshot is outside regular market hours; broker Review/Place remains blocked until the next OpenClaw refresh."
        if warnings:
            message += " Warnings: " + " | ".join(str(w) for w in warnings[:4])
        return {
            "name": "broker_reconciliation",
            "status": status,
            "message": message,
            "path": str(path),
            "held_count": held_count,
            "snapshot": {
                "status": snapshot.get("status"),
                "fresh": snapshot.get("fresh"),
                "age_minutes": snapshot.get("age_minutes"),
                "position_count": snapshot.get("position_count"),
                "warnings": warnings,
                "account_check": snapshot.get("account_check"),
            },
        }
    except Exception as exc:
        return {"name": "broker_reconciliation", "status": "FAIL", "message": f"Broker reconciliation check failed: {exc}"}


def _check_calibration_and_diagnosis(journal: DecisionJournal) -> dict[str, Any]:
    try:
        calibration = build_calibration_report(journal)
        latest_diag = journal.get_latest_calibration_diagnostic()
        completed = int(calibration.get("completed_shadow_rows") or 0)
        if latest_diag:
            age = _age_hours(latest_diag.get("generated_at"))
            if age is not None and age > 48:
                status = "WARN"
                message = f"Diagnosis exists but is stale ({age:.1f} hours old)."
            else:
                status = "PASS"
                message = f"Diagnosis is present; {completed} completed forward samples."
        else:
            status = "WARN"
            message = "No calibration diagnosis has been persisted yet."
        return {
            "name": "calibration_diagnosis",
            "status": status,
            "message": message,
            "completed_samples": completed,
            "shadow_rows": calibration.get("shadow_rows"),
        }
    except Exception as exc:
        return {"name": "calibration_diagnosis", "status": "FAIL", "message": f"Calibration check failed: {exc}"}


def _check_shadow_rules(journal: DecisionJournal) -> dict[str, Any]:
    try:
        summary = summarize_shadow_rules(journal)
        total = int(summary.get("total") or 0)
        if total == 0:
            status = "WARN"
            message = "No shadow-rule practice rows yet; future decisions will create them when rules trigger."
        else:
            status = "PASS"
            rule_count = 0
            try:
                with journal._connect() as conn:
                    rule_count = int(
                        conn.execute("SELECT COUNT(DISTINCT rule_id) FROM shadow_rule_evaluations").fetchone()[0]
                    )
            except Exception:
                pass
            message = (
                f"{rule_count} candidate rule(s) under paper trials — {total} evaluation row(s): "
                f"{summary.get('tracking', 0)} tracking, {summary.get('completed', 0)} completed."
                if rule_count
                else f"{total} shadow-rule evaluation row(s): "
                f"{summary.get('tracking', 0)} tracking, {summary.get('completed', 0)} completed."
            )
        return {
            "name": "shadow_rules",
            "status": status,
            "message": message,
            "summary": summary,
        }
    except Exception as exc:
        return {"name": "shadow_rules", "status": "FAIL", "message": f"Shadow-rule check failed: {exc}"}


def _check_execution_readiness(journal: DecisionJournal) -> dict[str, Any]:
    try:
        report = build_execution_readiness_report(journal)
        return {
            "name": "execution_readiness",
            "status": report.get("status") or "FAIL",
            "message": report.get("message") or "Execution readiness checked.",
            "ready_for_dry_run": report.get("ready_for_dry_run"),
            "live_trading_enabled": report.get("live_trading_enabled"),
            "execution_order_rows": report.get("execution_order_rows"),
            "guardrails": report.get("guardrails"),
            "config_errors": report.get("config_errors"),
        }
    except Exception as exc:
        return {"name": "execution_readiness", "status": "FAIL", "message": f"Execution readiness failed: {exc}"}


def _check_recent_logs(
    log_dir: Path = LOG_DIR,
    lookback_hours: int = 48,
    max_files: int = 12,
    max_tail_lines: int = 500,
) -> dict[str, Any]:
    """Scan recent log tails for plumbing issues that may degrade results."""
    if not log_dir.exists():
        return {"name": "recent_logs", "status": "WARN", "message": f"Log directory does not exist: {log_dir}"}

    cutoff = _utcnow() - timedelta(hours=lookback_hours)
    candidates: list[Path] = []
    for path in log_dir.glob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".log", ".err"} and not path.name.endswith(".log"):
            continue
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except Exception:
            continue
        if modified >= cutoff:
            candidates.append(path)
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[:max_files]
    if not candidates:
        return {
            "name": "recent_logs",
            "status": "WARN",
            "message": f"No log files modified in the last {lookback_hours} hours.",
        }

    error_lines: list[dict[str, str]] = []
    warning_lines: list[dict[str, str]] = []
    transient_lines: list[dict[str, str]] = []
    quality_lines: list[dict[str, str]] = []
    fatal_lines: list[dict[str, str]] = []

    def add(bucket: list[dict[str, str]], path: Path, line: str) -> None:
        if len(bucket) >= 20:
            return
        bucket.append({"file": path.name, "line": line.strip()[:300]})

    def suppress_benign_disconnect_tracebacks(path: Path, lines: list[str]) -> list[str]:
        if path.name != "dashboard.err.log":
            return lines
        filtered: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if "Exception occurred during processing of request from" not in line:
                filtered.append(line)
                i += 1
                continue

            block = [line]
            j = i + 1
            while j < len(lines):
                block.append(lines[j])
                if (
                    lines[j].strip() == "----------------------------------------"
                    and any(error in "\n".join(block) for error in BENIGN_DASHBOARD_DISCONNECT_ERRORS)
                ):
                    break
                j += 1
            block_text = "\n".join(block)
            if "Traceback (most recent call last):" in block_text and any(
                error in block_text for error in BENIGN_DASHBOARD_DISCONNECT_ERRORS
            ):
                add(transient_lines, path, "Dashboard client disconnected during request; suppressed benign traceback.")
                i = j + 1
                continue

            filtered.extend(block)
            i = j + 1
        return filtered

    for path in candidates:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_tail_lines:]
        except Exception:
            add(error_lines, path, "Could not read log file")
            continue
        lines = suppress_benign_disconnect_tracebacks(path, lines)
        for line in lines:
            if not line.strip():
                continue
            if re.search(r"\]\s+INFO:", line):
                continue
            if "Traceback" in line or "CRITICAL" in line:
                add(fatal_lines, path, line)
                continue
            if any(pattern in line for pattern in QUALITY_DEGRADING_LOG_PATTERNS):
                add(quality_lines, path, line)
                continue
            if any(pattern in line for pattern in EXPECTED_TRANSIENT_LOG_PATTERNS):
                if LOG_WARNING_RE.search(line) or LOG_ERROR_RE.search(line):
                    add(transient_lines, path, line)
                continue
            if LOG_ERROR_RE.search(line):
                add(error_lines, path, line)
                continue
            if LOG_WARNING_RE.search(line):
                add(warning_lines, path, line)

    if fatal_lines:
        status = "FAIL"
        message = f"Recent logs contain {len(fatal_lines)} fatal marker(s)."
    elif error_lines or quality_lines:
        status = "WARN"
        message = (
            f"Recent logs contain {len(error_lines)} error marker(s) and "
            f"{len(quality_lines)} quality-degrading marker(s)."
        )
    elif warning_lines:
        status = "WARN"
        message = f"Recent logs contain {len(warning_lines)} non-transient warning marker(s)."
    else:
        status = "PASS"
        message = (
            f"Recent log tails look clean. Ignored {len(transient_lines)} expected transient log event(s) "
            "whose bounded fallback behavior completed normally."
        )

    return {
        "name": "recent_logs",
        "status": status,
        "message": message,
        "files_checked": [p.name for p in candidates],
        "error_count": len(error_lines),
        "warning_count": len(warning_lines),
        "quality_issue_count": len(quality_lines),
        "fatal_count": len(fatal_lines),
        "transient_warning_count": len(transient_lines),
        "samples": {
            "fatal": fatal_lines[:3],
            "quality": quality_lines[:5],
            "errors": error_lines[:5],
            "warnings": warning_lines[:5],
        },
    }


def _check_broker_fill_accounting(
    journal: DecisionJournal,
    *,
    portfolio_path: str | Path = PORTFOLIO_FILE,
) -> dict[str, Any]:
    """Verify every broker sell fill reached the shared accounting contract."""
    sell_orders = [
        row
        for row in journal.get_execution_orders(limit=5000)
        if str(row.get("side") or "").lower() == "sell"
        and str(row.get("status") or "").lower() == "filled"
    ]
    effects = journal.get_broker_fill_effects(limit=5000)
    events = journal.get_sell_trade_events(limit=5000)
    tracking = journal.get_all_post_sell_reviews()
    episodes = journal.get_position_trade_episodes(limit=5000)

    effects_by_order = {
        str(row.get("broker_order_id") or ""): row
        for row in effects
        if row.get("broker_order_id")
    }
    effects_by_intent = {
        str(row.get("order_intent_id") or ""): row
        for row in effects
        if row.get("order_intent_id")
    }
    events_by_order = {
        str(row.get("broker_order_id") or ""): row
        for row in events
        if row.get("broker_order_id")
    }
    events_by_intent = {
        str(row.get("order_intent_id") or ""): row
        for row in events
        if row.get("order_intent_id")
    }
    missing_effects: list[str] = []
    missing_events: list[str] = []
    for row in sell_orders:
        broker_id = str(row.get("broker_order_id") or "")
        intent_id = str(row.get("order_intent_id") or "")
        label = broker_id or intent_id or f"row:{row.get('id')}"
        if not (effects_by_order.get(broker_id) or effects_by_intent.get(intent_id)):
            missing_effects.append(label)
        if not (events_by_order.get(broker_id) or events_by_intent.get(intent_id)):
            missing_events.append(label)

    tracking_event_ids = {
        str(row.get("sell_event_id") or "")
        for row in tracking
        if row.get("sell_event_id")
    }
    missing_exit_tracking = [
        str(row.get("event_id") or "")
        for row in events
        if str(row.get("event_type") or "").upper() == "EXIT"
        and row.get("thesis_id")
        and str(row.get("event_id") or "") not in tracking_event_ids
    ]
    episode_theses = {
        str(row.get("thesis_id") or "")
        for row in episodes
        if row.get("thesis_id")
    }
    missing_episodes = sorted(
        {
            str(row.get("thesis_id") or "")
            for row in events
            if row.get("thesis_id")
            and str(row.get("thesis_id") or "") not in episode_theses
        }
    )

    held = {position.ticker.upper() for position in Portfolio.load(Path(portfolio_path)).positions}
    trim_cooldown_missing: list[str] = []
    for event in events:
        if str(event.get("event_type") or "").upper() != "TRIM":
            continue
        if str(event.get("source") or "").startswith("historical_"):
            continue
        ticker = str(event.get("ticker") or "").upper()
        thesis_id = str(event.get("thesis_id") or "")
        if ticker not in held or not thesis_id:
            continue
        thesis = journal.get_thesis(thesis_id) or {}
        if str(thesis.get("status") or "") == "archived":
            continue
        if not thesis.get("sell_cooldown_until"):
            trim_cooldown_missing.append(str(event.get("event_id") or ticker))

    hard_failures = (
        missing_effects + missing_events + missing_exit_tracking + trim_cooldown_missing
    )
    if hard_failures:
        status = "FAIL"
        message = (
            f"Sell-fill contract is incomplete: {len(missing_effects)} missing effect(s), "
            f"{len(missing_events)} missing event(s), {len(missing_exit_tracking)} missing "
            f"post-sell tracker(s), {len(trim_cooldown_missing)} missing trim cooldown(s)."
        )
    elif missing_episodes:
        status = "WARN"
        message = f"Sell fills are accounted for, but {len(missing_episodes)} thesis episode(s) are missing."
    else:
        status = "PASS"
        message = (
            f"{len(sell_orders)} filled sell order(s) have durable effects/events; "
            f"{len(episodes)} position episode(s) and {len(tracking)} post-sell tracker(s) are recorded."
        )
    return {
        "name": "broker_fill_accounting",
        "status": status,
        "message": message,
        "filled_sell_orders": len(sell_orders),
        "fill_effects": len(effects),
        "sell_events": len(events),
        "position_episodes": len(episodes),
        "post_sell_tracking": len(tracking),
        "missing_effects": missing_effects[:20],
        "missing_events": missing_events[:20],
        "missing_exit_tracking": missing_exit_tracking[:20],
        "missing_episodes": missing_episodes[:20],
        "trim_cooldown_missing": trim_cooldown_missing[:20],
    }


def _check_position_classification(
    *,
    portfolio_path: str | Path = PORTFOLIO_FILE,
) -> dict[str, Any]:
    """Ensure held positions can participate in the intended sector gate."""
    portfolio = Portfolio.load(Path(portfolio_path))
    unknown = sorted(
        position.ticker.upper()
        for position in portfolio.positions
        if not str(position.sector or "").strip()
    )
    sector_values: dict[str, float] = {}
    invested = 0.0
    for position in portfolio.positions:
        market_value = float(
            position.market_value
            or (float(position.shares or 0.0) * float(position.current_price or position.avg_cost or 0.0))
        )
        invested += market_value
        sector = str(position.sector or "unknown").strip() or "unknown"
        sector_values[sector] = sector_values.get(sector, 0.0) + market_value
    nav = portfolio.total_nav()
    exposures = [
        {
            "sector": sector,
            "market_value": round(value, 2),
            "pct_nav": round(value / nav * 100.0, 2) if nav > 0 else 0.0,
            "pct_invested": round(value / invested * 100.0, 2) if invested > 0 else 0.0,
        }
        for sector, value in sorted(
            sector_values.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    if unknown:
        status = "FAIL"
        message = (
            f"{len(unknown)} held position(s) have no sector, so the concentration gate "
            "cannot evaluate them."
        )
    elif portfolio.positions:
        status = "PASS"
        top = exposures[0] if exposures else {}
        message = (
            f"All {len(portfolio.positions)} held position(s) are classified. "
            f"Largest sector is {top.get('sector')} at {top.get('pct_nav', 0):.1f}% NAV / "
            f"{top.get('pct_invested', 0):.1f}% invested."
        )
    else:
        status = "PASS"
        message = "No held positions require sector classification."
    return {
        "name": "position_classification",
        "status": status,
        "message": message,
        "positions": len(portfolio.positions),
        "unknown_tickers": unknown,
        "sector_exposures": exposures,
    }


def _check_execution_learning(
    journal: DecisionJournal,
    *,
    portfolio_path: str | Path = PORTFOLIO_FILE,
) -> dict[str, Any]:
    """Expose balanced execution measurements without changing strategy."""
    summary = build_execution_learning_summary(
        journal,
        portfolio_path=portfolio_path,
    )
    episodes = summary["episodes"]
    quality = summary["data_quality"]
    post_sell = summary["post_sell"]
    if episodes["closed"] == 0:
        status = "WARN"
        message = "No completed position episodes are available for execution learning yet."
    elif quality["missing_realized_pnl_events"]:
        status = "WARN"
        message = (
            f"Execution learning has {quality['missing_realized_pnl_events']} sell event(s) "
            "without realized P&L."
        )
    elif quality["uncertain_events"]:
        status = "WARN"
        message = (
            f"Execution learning is provisional: {quality['uncertain_events']} of "
            f"{summary['sell_events']['total']} sell event(s) have explicitly "
            f"uncertain accounting ({quality['inferred_cost_events']} inferred-cost, "
            f"{quality['quantity_conflict_events']} legacy quantity-conflict)."
        )
    elif post_sell["overdue_60d_reviews"]:
        status = "WARN"
        message = (
            f"Post-sell learning has {post_sell['overdue_60d_reviews']} review(s) "
            "more than five days past the 60-day checkpoint without a recorded price."
        )
    elif not post_sell["council_learning_ready"]:
        status = "PASS"
        next_due = post_sell.get("next_60d_due_date") or "none scheduled"
        message = (
            f"Sell accounting is operational across {episodes['closed']} completed episode(s). "
            "Outcome-based Council learning is intentionally inactive: "
            f"{post_sell['completed']}/{post_sell['minimum_completed_for_council']} "
            f"completed 60-day post-sell reviews; next maturity={next_due}; do_not_adjust."
        )
    else:
        status = "PASS"
        expectancy = episodes["expectancy_dollars"]
        expectancy_text = "unknown" if expectancy is None else f"${expectancy:+.2f}"
        message = (
            f"Execution learning covers {episodes['closed']} completed episode(s); "
            f"episode win rate={episodes['win_rate_pct']:.1f}%, "
            f"expectancy={expectancy_text}, sell events={summary['sell_events']['total']}, "
            f"post-sell reviews={summary['post_sell']['total']}."
        )
    return {
        "name": "execution_learning",
        "status": status,
        "message": message,
        "observational_only": True,
        "council_learning_ready": post_sell["council_learning_ready"],
        "summary": summary,
    }


def _check_telegram(sender: TelegramSender) -> dict[str, Any]:
    if sender.enabled:
        return {"name": "telegram", "status": "PASS", "message": "Telegram is configured."}
    return {
        "name": "telegram",
        "status": "WARN",
        "message": "Telegram is not configured, so Supervisor can write reports but cannot send them.",
    }


def _check_recent_sessions(journal: DecisionJournal) -> dict[str, Any]:
    try:
        with journal._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM sessions
                ORDER BY datetime(timestamp) DESC, id DESC
                LIMIT 1
                """
            ).fetchone()
        if not row:
            return {"name": "recent_sessions", "status": "WARN", "message": "No Artha sessions logged yet."}
        latest = dict(row)
        age = _age_hours(latest.get("timestamp"))
        status = "PASS" if age is not None and age <= 96 else "WARN"
        message = (
            f"Latest session {latest.get('session_type')} analyzed {latest.get('tickers_analyzed') or 'n/a'} "
            f"{age:.1f} hours ago."
            if age is not None else
            f"Latest session {latest.get('session_type')} is logged."
        )
        return {"name": "recent_sessions", "status": status, "message": message, "age_hours": age}
    except Exception as exc:
        return {"name": "recent_sessions", "status": "FAIL", "message": f"Session check failed: {exc}"}


def _severity(checks: list[dict[str, Any]]) -> str:
    statuses = {str(c.get("status") or "PASS").upper() for c in checks}
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def _format_report(payload: dict[str, Any]) -> str:
    checks = payload.get("checks") or []
    lines = [
        "ARTHA SUPERVISOR CHECK",
        "======================",
        f"Severity: {payload.get('severity')}",
        f"Generated: {payload.get('generated_at')}",
        "",
        "Plain English:",
        "I checked whether Artha's research machine, report card, watchlists, shadow tests, Robinhood-ready execution wiring, and Telegram path are working.",
        "",
        "Results:",
    ]
    for check in checks:
        duration = check.get("duration_ms")
        duration_text = f" ({duration} ms)" if duration is not None else ""
        lines.append(f"- {check.get('status')}: {check.get('name')}{duration_text} - {check.get('message')}")
        samples = ((check.get("samples") or {}).get("quality") or [])[:2]
        for sample in samples:
            lines.append(f"  sample: {sample.get('file')}: {sample.get('line')}")
    lines.extend(
        [
            "",
            "Guardrail:",
            "Supervisor can detect, report, and test proposed fixes in shadow mode. It cannot change live investing rules.",
        ]
    )
    return "\n".join(lines)


def _write_artifacts(report: dict[str, Any]) -> dict[str, str]:
    SUPERVISOR_DIR.mkdir(parents=True, exist_ok=True)
    stamp = str(report.get("generated_at") or _utcnow_iso()).replace(":", "").replace("-", "").replace(".", "_")
    json_path = SUPERVISOR_DIR / f"supervisor_{stamp}.json"
    txt_path = SUPERVISOR_DIR / f"supervisor_{stamp}.txt"
    latest_json = SUPERVISOR_DIR / "latest.json"
    latest_txt = SUPERVISOR_DIR / "latest.txt"
    json_body = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True)
    txt_body = str(report.get("report_text") or "")
    for path, body in (
        (json_path, json_body),
        (latest_json, json_body),
        (txt_path, txt_body),
        (latest_txt, txt_body),
    ):
        path.write_text(body, encoding="utf-8")
    return {
        "json_path": str(json_path),
        "text_path": str(txt_path),
        "latest_json": str(latest_json),
        "latest_text": str(latest_txt),
    }


def _should_send(report: dict[str, Any], previous: dict[str, Any] | None, force: bool = False) -> bool:
    if force:
        return True
    if not previous:
        return str(report.get("severity")) != "PASS"
    if str(report.get("severity")) != str(previous.get("severity")):
        return True
    if str(report.get("severity")) in {"WARN", "FAIL"}:
        return str(report.get("report_hash")) != str(previous.get("report_hash"))
    return False


def run_supervisor_check(
    journal: DecisionJournal | None = None,
    send_telegram: bool = False,
    force_telegram: bool = False,
    sender: TelegramSender | None = None,
    run_diagnosis: bool = True,
    diagnosis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Supervisor v1 and optionally send a Telegram summary."""
    journal = journal or DecisionJournal()
    sender = sender or TelegramSender()

    logger.info("[supervisor] Starting Supervisor v1")
    operations: list[dict[str, Any]] = []
    op = _timed_operation("shadow_rule_outcome_update", lambda: update_shadow_rule_outcomes(journal))
    operations.append(op)
    shadow_updates = op.get("result") or {"updated": 0, "errors": 1 if op.get("status") == "FAIL" else 0, "skipped": 0}

    op = _timed_operation("shadow_rule_backfill", lambda: backfill_shadow_rules_from_features(journal))
    operations.append(op)
    shadow_backfilled = op.get("result") if op.get("status") == "PASS" else 0

    op = _timed_operation("decision_feature_backfill", lambda: backfill_decision_features(journal))
    operations.append(op)
    decision_backfilled = op.get("result") if op.get("status") == "PASS" else 0

    op = _timed_operation(
        "trade_action_reconciliation",
        lambda: {
            "reconciled": journal.reconcile_trade_actions_from_execution_orders(),
            "expired": journal.expire_stale_trade_actions(),
        },
    )
    operations.append(op)

    if run_diagnosis or diagnosis is None:
        op = _timed_operation(
            "calibration_diagnosis",
            lambda: run_calibration_diagnosis(
                journal=journal,
                send_telegram=False,
                force_telegram=False,
            ),
        )
        operations.append(op)
        diagnosis = op.get("result") or {}
    else:
        operations.append(
            {
                "name": "calibration_diagnosis",
                "status": "SKIPPED",
                "duration_ms": 0.0,
                "result": {"stage": diagnosis.get("stage"), "completed_samples": diagnosis.get("completed_samples")},
                "error": "",
            }
        )

    operation_failures = [op for op in operations if op.get("status") == "FAIL"]
    operation_check = {
        "name": "maintenance_operations",
        "status": "FAIL" if operation_failures else "PASS",
        "message": (
            f"{len(operation_failures)} maintenance operation(s) failed."
            if operation_failures else
            f"{len(operations)} maintenance operation(s) completed or were intentionally skipped."
        ),
        "failures": operation_failures,
        "duration_ms": round(sum(float(op.get("duration_ms") or 0) for op in operations), 1),
    }

    checks = [
        operation_check,
        _timed_check("database", lambda: _check_database(journal)),
        _timed_check("decision_artifacts", lambda: _check_latest_decision_artifacts(journal)),
        _timed_check("rank_coverage", _check_rank_coverage),
        _timed_check("latest_report", lambda: _check_latest_report_artifact(journal)),
        _timed_check("agentic_trace", lambda: _check_agentic_trace_artifact(journal)),
        _timed_check("intelligence_routing", lambda: _check_intelligence_routing(journal)),
        _timed_check("recent_sessions", lambda: _check_recent_sessions(journal)),
        _timed_check("defer_watchlist", lambda: _check_defer_watches(journal)),
        _timed_check("position_monitoring", lambda: _check_position_monitoring(journal)),
        _timed_check("broker_fill_accounting", lambda: _check_broker_fill_accounting(journal)),
        _timed_check("position_classification", lambda: _check_position_classification()),
        _timed_check("execution_learning", lambda: _check_execution_learning(journal)),
        _timed_check("broker_reconciliation", lambda: _check_broker_reconciliation_snapshot()),
        _timed_check("calibration_diagnosis", lambda: _check_calibration_and_diagnosis(journal)),
        _timed_check("shadow_rules", lambda: _check_shadow_rules(journal)),
        _timed_check("execution_readiness", lambda: _check_execution_readiness(journal)),
        _timed_check("recent_logs", lambda: _check_recent_logs()),
        _timed_check("telegram", lambda: _check_telegram(sender)),
    ]
    payload = {
        "generated_at": _utcnow_iso(),
        "severity": _severity(checks),
        "checks": checks,
        "operations": operations,
        "shadow_rule_updates": shadow_updates,
        "shadow_rule_backfilled": shadow_backfilled,
        "decision_features_backfilled": decision_backfilled,
        "diagnosis_stage": (diagnosis or {}).get("stage"),
        "diagnosis_samples": (diagnosis or {}).get("completed_samples"),
        "guardrail": "No automatic investing-rule changes.",
    }
    report_text = _format_report(payload)
    report = {
        "generated_at": payload["generated_at"],
        "severity": payload["severity"],
        "report_text": report_text,
        "report_hash": hashlib.sha256(report_text.encode("utf-8")).hexdigest(),
        "payload": payload,
        "sent_to_telegram": False,
    }
    previous = journal.get_latest_supervisor_run()
    if send_telegram and sender.enabled and _should_send(report, previous, force=force_telegram):
        report["sent_to_telegram"] = bool(sender.send_message(report_text, parse_mode=None, silent=True))
    report["artifacts"] = _write_artifacts(report)
    report["row_id"] = journal.save_supervisor_run(report)
    logger.info(
        "[supervisor] Completed Supervisor v1 severity=%s sent_to_telegram=%s artifact=%s",
        report.get("severity"),
        report.get("sent_to_telegram"),
        report.get("artifacts", {}).get("latest_text"),
    )
    return report
