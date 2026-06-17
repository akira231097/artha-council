"""Outcome diagnosis and guarded self-improvement reports for Artha.

This module turns calibration records into plain-English learning reports. It
does not change live trading rules by itself. Any proposed fix stays in
bookkeeping or shadow mode until enough forward samples exist.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .calibration import build_calibration_report
from .journal import DecisionJournal
from .telegram import TelegramSender

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DIAGNOSTIC_DIR = DATA_DIR / "calibration_diagnostics"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _pct(value: Any) -> str:
    number = _num(value)
    if number is None:
        return "n/a"
    return f"{number:+.1%}"


def _score_bucket(score: float) -> str:
    if score >= 75:
        return "75-100"
    if score >= 65:
        return "65-74"
    if score >= 55:
        return "55-64"
    if score >= 45:
        return "45-54"
    return "0-44"


def _shadow_rows(journal: DecisionJournal) -> list[dict[str, Any]]:
    with journal._connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM shadow_positions
            ORDER BY datetime(created_at) DESC, id DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def _stage_for_samples(completed_samples: int) -> dict[str, str | int]:
    """Map mature sample count to a safe activation stage."""
    if completed_samples < 20:
        return {
            "stage": "learning_only",
            "label": "Learning only",
            "next_gate": 20,
            "live_change_allowed": "no",
            "plain_english": (
                "Artha can describe early patterns, but it must not change its live "
                "buy/defer rules yet."
            ),
        }
    if completed_samples < 30:
        return {
            "stage": "minimum_diagnosis",
            "label": "Minimum diagnosis",
            "next_gate": 30,
            "live_change_allowed": "no",
            "plain_english": (
                "Artha has enough outcomes to start diagnosing likely mistakes, "
                "but fixes stay in shadow mode."
            ),
        }
    if completed_samples < 40:
        return {
            "stage": "early_pattern_review",
            "label": "Early pattern review",
            "next_gate": 40,
            "live_change_allowed": "no",
            "plain_english": (
                "Patterns are becoming more informative. Artha can propose fixes, "
                "but still needs shadow validation."
            ),
        }
    if completed_samples < 60:
        return {
            "stage": "strong_pattern_review",
            "label": "Stronger pattern review",
            "next_gate": 60,
            "live_change_allowed": "no",
            "plain_english": (
                "Artha can produce stronger fix proposals and before/after checks, "
                "but live rules still require more evidence."
            ),
        }
    if completed_samples < 100:
        return {
            "stage": "overlay_candidate_review",
            "label": "Conservative overlay candidate review",
            "next_gate": 100,
            "live_change_allowed": "manual_review_only",
            "plain_english": (
                "Artha may nominate conservative calibration overlays, but they "
                "should still be reviewed before affecting live decisions."
            ),
        }
    return {
        "stage": "ml_meta_ranker_ready",
        "label": "ML/meta-ranker candidate ready",
        "next_gate": 0,
        "live_change_allowed": "manual_review_only",
        "plain_english": (
            "Artha has enough records to train or validate a real meta-ranker, "
            "but live activation should still require reviewed backtest evidence."
        ),
    }


def _classify_row(row: dict[str, Any]) -> list[str]:
    """Classify a completed shadow row into understandable mistake patterns."""
    labels: list[str] = []
    score = _num(row.get("opportunity_score"), 0.0) or 0.0
    excess_60 = _num(row.get("excess_return_60d"))
    excess_20 = _num(row.get("excess_return_20d"))
    mae = _num(row.get("mae"))
    mfe = _num(row.get("mfe"))

    benchmark_excess = excess_60 if excess_60 is not None else excess_20
    if benchmark_excess is not None:
        if benchmark_excess >= 0.05 and score < 55:
            labels.append("deferred_winner_low_score")
        if benchmark_excess >= 0.03 and score >= 65:
            labels.append("blocked_high_score_winner")
        if benchmark_excess <= -0.05 and score >= 55:
            labels.append("overtrusted_score_bucket")
        if benchmark_excess <= -0.08:
            labels.append("correctly_avoided_or_deferred")
    if row.get("would_hit_stop"):
        labels.append("would_have_hit_stop")
    if mae is not None and mae <= -0.15:
        labels.append("deep_drawdown_risk")
    if mfe is not None and mfe >= 0.20 and benchmark_excess is not None and benchmark_excess <= 0:
        labels.append("timing_decay_after_pop")
    return labels or ["no_clear_pattern"]


def _bucket_diagnostics(calibration_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket, data in sorted((calibration_report.get("shadow_score_buckets") or {}).items()):
        completed = int(data.get("completed") or 0)
        count = int(data.get("count") or 0)
        avg_excess_20 = data.get("avg_excess_return_20d")
        avg_excess_60 = data.get("avg_excess_return_60d")
        hit20 = data.get("relative_hit_rate_20d")
        if completed == 0 and avg_excess_20 is None:
            status = "not_enough_time"
            summary = "No mature or checkpointed outcomes yet."
        elif completed < 3:
            status = "early_signal"
            summary = "Interesting but too small to change rules."
        elif avg_excess_60 is not None and avg_excess_60 >= 0.03:
            status = "positive"
            summary = "This score bucket has beaten its benchmark so far."
        elif avg_excess_60 is not None and avg_excess_60 <= -0.03:
            status = "negative"
            summary = "This score bucket has lagged its benchmark so far."
        else:
            status = "mixed"
            summary = "This score bucket is close to benchmark or mixed."
        rows.append(
            {
                "bucket": bucket,
                "count": count,
                "completed": completed,
                "avg_excess_return_20d": avg_excess_20,
                "avg_excess_return_60d": avg_excess_60,
                "relative_hit_rate_20d": hit20,
                "status": status,
                "summary": summary,
            }
        )
    return rows


def _pattern_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("status") != "completed":
            continue
        for label in _classify_row(row):
            counts[label] += 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _proposed_fixes(
    completed_rows: list[dict[str, Any]],
    bucket_rows: list[dict[str, Any]],
    completed_samples: int,
) -> list[dict[str, Any]]:
    """Create guarded fix proposals from observed patterns."""
    fixes: list[dict[str, Any]] = []
    activation = _stage_for_samples(completed_samples)
    stage = str(activation["stage"])
    shadow_status = "bookkeeping_only" if completed_samples < 20 else "shadow_mode"
    if completed_samples >= 60:
        shadow_status = "overlay_candidate_manual_review"
    if completed_samples >= 100:
        shadow_status = "ml_candidate_manual_review"

    for bucket in bucket_rows:
        completed = int(bucket.get("completed") or 0)
        avg60 = _num(bucket.get("avg_excess_return_60d"))
        avg20 = _num(bucket.get("avg_excess_return_20d"))
        if completed < 3:
            continue
        if avg60 is not None and avg60 <= -0.03:
            fixes.append(
                {
                    "rule_id": f"bucket_{bucket['bucket']}_underperformance",
                    "status": shadow_status,
                    "trigger": f"Score bucket {bucket['bucket']} has negative benchmark-relative 60-day returns.",
                    "suggested_change": (
                        "Make the CIO require stronger valuation, catalyst, or technical confirmation "
                        "before upgrading names in this score bucket."
                    ),
                    "evidence": f"completed={completed}, avg_excess_60d={_pct(avg60)}",
                    "activation_gate": _activation_gate_text(stage),
                }
            )
        elif avg60 is not None and avg60 >= 0.03 and bucket["bucket"] in {"45-54", "55-64"}:
            fixes.append(
                {
                    "rule_id": f"bucket_{bucket['bucket']}_missed_winners",
                    "status": shadow_status,
                    "trigger": f"Lower score bucket {bucket['bucket']} has positive benchmark-relative 60-day returns.",
                    "suggested_change": (
                        "Diagnose false DEFER/WATCH decisions: if catalyst strength and revision trend are positive, "
                        "consider a small starter path instead of pure defer."
                    ),
                    "evidence": f"completed={completed}, avg_excess_60d={_pct(avg60)}",
                    "activation_gate": _activation_gate_text(stage),
                }
            )
        elif avg20 is not None and avg20 <= -0.05 and completed_samples >= 20:
            fixes.append(
                {
                    "rule_id": f"bucket_{bucket['bucket']}_weak_20d_checkpoint",
                    "status": shadow_status,
                    "trigger": f"Score bucket {bucket['bucket']} is weak at the 20-day checkpoint.",
                    "suggested_change": (
                        "Keep this as a shadow warning until 60-day outcomes confirm whether the weakness persists."
                    ),
                    "evidence": f"completed={completed}, avg_excess_20d={_pct(avg20)}",
                    "activation_gate": _activation_gate_text(stage),
                }
            )

    patterns = _pattern_counts(completed_rows)
    if patterns.get("would_have_hit_stop", 0) >= 3:
        fixes.append(
            {
                "rule_id": "stop_hit_pattern",
                "status": shadow_status,
                "trigger": "Several shadow entries would have hit their stop.",
                "suggested_change": (
                    "Review technical entry discipline and avoid entries where expected drawdown is too large."
                ),
                "evidence": f"would_have_hit_stop={patterns['would_have_hit_stop']}",
                "activation_gate": _activation_gate_text(stage),
            }
        )
    if patterns.get("deferred_winner_low_score", 0) >= 3:
        fixes.append(
            {
                "rule_id": "false_defer_low_score_pattern",
                "status": shadow_status,
                "trigger": "Several low-score deferred names later beat the benchmark.",
                "suggested_change": (
                    "Investigate whether Artha is over-penalizing valuation or timing when catalyst/revision evidence is strong."
                ),
                "evidence": f"deferred_winner_low_score={patterns['deferred_winner_low_score']}",
                "activation_gate": _activation_gate_text(stage),
            }
        )

    if not fixes:
        fixes.append(
            {
                "rule_id": "no_live_fix",
                "status": "no_change",
                "trigger": "No statistically useful mistake pattern yet.",
                "suggested_change": "Keep collecting outcomes; do not change live scoring rules.",
                "evidence": f"completed_samples={completed_samples}",
                "activation_gate": _activation_gate_text(stage),
            }
        )
    return fixes


def _activation_gate_text(stage: str) -> str:
    if stage == "learning_only":
        return "Wait for at least 20 completed 60-day outcomes."
    if stage in {"minimum_diagnosis", "early_pattern_review", "strong_pattern_review"}:
        return "Keep fix in shadow mode; require better before/after evidence."
    if stage == "overlay_candidate_review":
        return "Manual review required before any conservative live overlay."
    return "Manual review and backtest required before any ML/meta-ranker activation."


def _severity(stage: str, fixes: list[dict[str, Any]], patterns: dict[str, int]) -> str:
    if stage == "ml_meta_ranker_ready":
        return "ML_READY"
    if stage == "overlay_candidate_review":
        return "OVERLAY_REVIEW"
    if any(f.get("status") not in {"no_change", "bookkeeping_only"} for f in fixes):
        return "SHADOW_ACTION"
    if any(v >= 3 for k, v in patterns.items() if k != "no_clear_pattern"):
        return "WATCH"
    return "INFO"


def build_diagnostic_payload(journal: DecisionJournal | None = None) -> dict[str, Any]:
    """Build the structured diagnosis payload."""
    journal = journal or DecisionJournal()
    calibration = build_calibration_report(journal)
    rows = _shadow_rows(journal)
    completed_rows = [row for row in rows if row.get("status") == "completed"]
    completed_samples = int(calibration.get("completed_shadow_rows") or len(completed_rows))
    stage = _stage_for_samples(completed_samples)
    bucket_rows = _bucket_diagnostics(calibration)
    patterns = _pattern_counts(completed_rows)
    fixes = _proposed_fixes(completed_rows, bucket_rows, completed_samples)
    severity = _severity(str(stage["stage"]), fixes, patterns)

    payload = {
        "generated_at": _utcnow_iso(),
        "completed_samples": completed_samples,
        "total_shadow_rows": int(calibration.get("shadow_rows") or len(rows)),
        "stage": stage["stage"],
        "stage_label": stage["label"],
        "next_gate": stage["next_gate"],
        "live_change_allowed": stage["live_change_allowed"],
        "plain_english_stage": stage["plain_english"],
        "severity": severity,
        "bucket_diagnostics": bucket_rows,
        "pattern_counts": patterns,
        "proposed_fixes": fixes,
        "calibration_status": calibration.get("calibration_status"),
        "calibration_report": calibration,
    }
    return payload


def format_diagnostic_report(payload: dict[str, Any]) -> str:
    """Render a Telegram-friendly plain-English report."""
    completed = int(payload.get("completed_samples") or 0)
    next_gate = int(payload.get("next_gate") or 0)
    lines = [
        "ARTHA LEARNING DIAGNOSIS",
        "========================",
        f"Completed forward samples: {completed}",
        f"Stage: {payload.get('stage_label')} ({payload.get('stage')})",
        f"Live rule changes allowed: {payload.get('live_change_allowed')}",
        "",
        "Plain English:",
        str(payload.get("plain_english_stage") or ""),
    ]
    if next_gate > 0:
        lines.append(f"Next gate: {next_gate} completed 60-day outcomes.")
    else:
        lines.append("Next gate: manual ML/meta-ranker review.")

    lines.extend(["", "What Artha checked:"])
    bucket_rows = payload.get("bucket_diagnostics") or []
    if bucket_rows:
        for bucket in bucket_rows:
            lines.append(
                "- Score {}: {} row(s), {} completed, excess20 {}, excess60 {}. {}".format(
                    bucket.get("bucket"),
                    bucket.get("count"),
                    bucket.get("completed"),
                    _pct(bucket.get("avg_excess_return_20d")),
                    _pct(bucket.get("avg_excess_return_60d")),
                    bucket.get("summary"),
                )
            )
    else:
        lines.append("- No score buckets have enough tracked outcomes yet.")

    patterns = payload.get("pattern_counts") or {}
    lines.extend(["", "Mistake patterns found:"])
    useful_patterns = {k: v for k, v in patterns.items() if k != "no_clear_pattern"}
    if useful_patterns:
        for name, count in sorted(useful_patterns.items(), key=lambda kv: (-kv[1], kv[0]))[:6]:
            lines.append(f"- {name}: {count}")
    else:
        lines.append("- No repeated mistake pattern is strong enough yet.")

    lines.extend(["", "Proposed fixes:"])
    for fix in (payload.get("proposed_fixes") or [])[:5]:
        lines.append(
            "- {} [{}]: {}".format(
                fix.get("rule_id"),
                fix.get("status"),
                fix.get("suggested_change"),
            )
        )
        lines.append(f"  Evidence: {fix.get('evidence')}")
        lines.append(f"  Gate: {fix.get('activation_gate')}")

    lines.extend(
        [
            "",
            "Important:",
            "This report can warn and propose fixes. It does not rewrite live investing rules by itself.",
        ]
    )
    return "\n".join(lines)


def build_diagnostic_report(journal: DecisionJournal | None = None) -> dict[str, Any]:
    """Build structured payload plus rendered report and stable hash."""
    payload = build_diagnostic_payload(journal)
    report_text = format_diagnostic_report(payload)
    report_hash = hashlib.sha256(report_text.encode("utf-8")).hexdigest()
    result = {
        "generated_at": payload["generated_at"],
        "completed_samples": payload["completed_samples"],
        "stage": payload["stage"],
        "severity": payload["severity"],
        "report_hash": report_hash,
        "report_text": report_text,
        "payload": payload,
        "sent_to_telegram": False,
    }
    return result


def should_send_diagnostic(
    diagnostic: dict[str, Any],
    previous: dict[str, Any] | None,
    force: bool = False,
) -> bool:
    """Decide whether the scheduler should send this diagnosis to Telegram."""
    if force:
        return True
    if previous is None:
        return True
    if int(diagnostic.get("completed_samples") or 0) > int(previous.get("completed_samples") or 0):
        return True
    if str(diagnostic.get("stage") or "") != str(previous.get("stage") or ""):
        return True
    severity_rank = {"INFO": 0, "WATCH": 1, "SHADOW_ACTION": 2, "OVERLAY_REVIEW": 3, "ML_READY": 4}
    if severity_rank.get(str(diagnostic.get("severity")), 0) > severity_rank.get(str(previous.get("severity")), 0):
        return True
    return False


def write_diagnostic_artifacts(diagnostic: dict[str, Any]) -> dict[str, str]:
    """Write latest and timestamped diagnosis JSON/text artifacts."""
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    generated = str(diagnostic.get("generated_at") or _utcnow_iso())
    stamp = generated.replace(":", "").replace("-", "").replace(".", "_").replace("+", "Z")
    json_path = DIAGNOSTIC_DIR / f"diagnosis_{stamp}.json"
    txt_path = DIAGNOSTIC_DIR / f"diagnosis_{stamp}.txt"
    latest_json = DIAGNOSTIC_DIR / "latest.json"
    latest_txt = DIAGNOSTIC_DIR / "latest.txt"
    json_body = json.dumps(diagnostic, indent=2, sort_keys=True, ensure_ascii=True)
    txt_body = str(diagnostic.get("report_text") or "")
    json_path.write_text(json_body, encoding="utf-8")
    txt_path.write_text(txt_body, encoding="utf-8")
    latest_json.write_text(json_body, encoding="utf-8")
    latest_txt.write_text(txt_body, encoding="utf-8")
    return {
        "json_path": str(json_path),
        "text_path": str(txt_path),
        "latest_json": str(latest_json),
        "latest_text": str(latest_txt),
    }


def send_diagnostic_to_telegram(
    diagnostic: dict[str, Any],
    sender: TelegramSender | None = None,
    force: bool = False,
) -> bool:
    """Send diagnosis to Telegram if configured and useful."""
    sender = sender or TelegramSender()
    if not sender.enabled:
        logger.info("[diagnostics] Telegram not configured; diagnosis not sent")
        return False
    previous = DecisionJournal().get_latest_calibration_diagnostic()
    if not should_send_diagnostic(diagnostic, previous, force=force):
        logger.info("[diagnostics] Diagnosis unchanged; suppressing Telegram send")
        return False
    return sender.send_message(str(diagnostic.get("report_text") or ""), parse_mode=None, silent=True)


def run_calibration_diagnosis(
    journal: DecisionJournal | None = None,
    send_telegram: bool = False,
    force_telegram: bool = False,
    sender: TelegramSender | None = None,
) -> dict[str, Any]:
    """Build, persist, optionally Telegram-send a diagnosis report."""
    journal = journal or DecisionJournal()
    previous = journal.get_latest_calibration_diagnostic()
    diagnostic = build_diagnostic_report(journal)
    artifacts = write_diagnostic_artifacts(diagnostic)
    sent = False
    if send_telegram and should_send_diagnostic(diagnostic, previous, force=force_telegram):
        sender = sender or TelegramSender()
        if sender.enabled:
            sent = sender.send_message(diagnostic["report_text"], parse_mode=None, silent=True)
    diagnostic["sent_to_telegram"] = bool(sent)
    row_id = journal.save_calibration_diagnostic(diagnostic)
    diagnostic["row_id"] = row_id
    diagnostic["artifacts"] = artifacts
    return diagnostic
