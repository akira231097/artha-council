"""The Artha Council — debate and consensus engine (v2).

Two-stage decision system:
  Stage 1: Hard Risk Gate (binary safety check)
  Stage 2: Opportunity Scoring → Score-to-Action mapping

Runs all three analysts independently, then synthesizes their reports
into a final actionable recommendation with structured JSON scoring.
"""
import re
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from .config import Config
from .chatgpt_backend import ChatGPTBackendClient
from .analysts import run_fundamental_analyst, run_technical_analyst, run_contrarian_analyst
from .portfolio_state import PortfolioStateEngine, get_deployment_target
from .journal import DecisionJournal
from .researcher import ResearchDesk
from .data_quality import validate_stock_data
from .prompts import SYNTHESIS_PROMPT, build_context_header, build_crisis_context
from .agentic_diligence import build_agentic_diligence
from .defer_watchlist import record_defer_watch
from .dossier import write_decision_dossier
from .meta_ranker import build_meta_signal, format_meta_signal
from .portfolio_risk import build_portfolio_factor_risk, format_portfolio_factor_risk
from .shadow_rules import evaluate_shadow_rules_for_decision
from .valuation import build_valuation_expectations, format_valuation_expectations
from .buy_scoring import (
    apply_cio_buy_adjustment,
    build_buy_score_audit,
    render_buy_score_audit,
)


def _extract_yaml_scalar(raw_yaml: str, key: str) -> str:
    """Best-effort scalar extractor for simple YAML fallback mode."""
    pattern = rf"^\s*{re.escape(key)}:\s*(.+?)\s*(?:#.*)?$"
    match = re.search(pattern, raw_yaml, flags=re.MULTILINE)
    if not match:
        return ""
    return match.group(1).strip().strip("'\"")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class AnalystReport:
    """Individual analyst output."""
    analyst_name: str
    model: str
    verdict: str  # BUY / HOLD / SELL
    confidence: int  # 1-10
    report: str  # Full text report

    @classmethod
    def parse(cls, analyst_name: str, model: str, raw_report: str) -> "AnalystReport":
        """Parse verdict and confidence from raw report text.

        Strategy: Try JSON block first, then regex on markdown, then defaults.
        """
        verdict = "HOLD"
        confidence = 5

        # Strategy 1: JSON extraction (fenced or raw)
        parsed = _extract_json_object(raw_report)
        if parsed:
            if "verdict" in parsed:
                v = str(parsed["verdict"]).upper().strip()
                if v in ("BUY", "HOLD", "SELL"):
                    verdict = v
            if "confidence" in parsed:
                c = int(parsed["confidence"])
                confidence = min(10, max(1, c))
            return cls(analyst_name=analyst_name, model=model,
                       verdict=verdict, confidence=confidence, report=raw_report)

        # Strategy 2: Regex on markdown format
        verdict_match = re.search(
            r"\*\*VERDICT:?\*\*\s*:?\s*(BUY|HOLD|SELL)",
            raw_report,
            re.IGNORECASE,
        )
        if verdict_match:
            verdict = verdict_match.group(1).upper()

        conf_match = re.search(r"\*\*CONFIDENCE:?\*\*\s*:?\s*(\d+)", raw_report)
        if conf_match:
            confidence = min(10, max(1, int(conf_match.group(1))))

        # Strategy 3: Loose text search as last resort
        if not verdict_match:
            for line in raw_report.split("\n"):
                line_upper = line.strip().upper()
                if line_upper.startswith("VERDICT:") or "VERDICT" in line_upper:
                    if "BUY" in line_upper and "SELL" not in line_upper:
                        verdict = "BUY"
                    elif "SELL" in line_upper:
                        verdict = "SELL"
                    break
        if not conf_match:
            conf_loose = re.search(r"confidence\s*:?\s*(\d+)", raw_report, re.IGNORECASE)
            if conf_loose:
                confidence = min(10, max(1, int(conf_loose.group(1))))

        return cls(
            analyst_name=analyst_name,
            model=model,
            verdict=verdict,
            confidence=confidence,
            report=raw_report,
        )


@dataclass
class CouncilDecision:
    """Final synthesized council decision (v2)."""
    ticker: str
    final_verdict: str          # Uses VERDICT_TYPES from config
    consensus: str              # 3/3, 2-1, Split
    recommended_action: str
    allocation: str
    synthesis_report: str       # Full synthesis text (narrative only)
    fundamental: AnalystReport
    technical: AnalystReport
    contrarian: AnalystReport
    # v2 scoring fields
    opportunity_score: int = 0
    adjusted_score: int = 0     # After regime adjustment
    score_components: dict = field(default_factory=dict)
    confidence: int = 5
    thesis_type: str = ""
    recommended_allocation_pct: float = 0.0
    entry_valid_until: str = ""
    invalidation_conditions: list = field(default_factory=list)
    stop_loss_pct: float = -0.08
    target_pct: float = 0.15
    deployment_context: dict = field(default_factory=dict)
    hard_risk_gate_passed: bool = True
    hard_risk_gate_reason: str = ""
    data_quality_report: dict = field(default_factory=dict)
    agentic_trace: dict = field(default_factory=dict)
    dossier_path: str = ""
    defer_watch_id: str = ""
    valuation_expectations: dict = field(default_factory=dict)
    portfolio_factor_risk: dict = field(default_factory=dict)
    calibration_meta_signal: dict = field(default_factory=dict)
    base_opportunity_score: int = 0
    rule_adjustment_total: int = 0
    cio_adjustment: int = 0
    scoring_audit: dict = field(default_factory=dict)
    # Wave 2: machine-readable CIO decision block + mechanical regime gate.
    cio_decision: dict = field(default_factory=dict)
    half_size: bool = False
    regime_gate: dict = field(default_factory=dict)
    mechanical_override: str = ""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_VALID_FINAL_VERDICTS = {
    "STRONG BUY", "BUY", "STARTER", "TACTICAL_BUY", "ACCUMULATE",
    "ADD", "HOLD", "WATCH", "DEFER", "TRIM", "SELL", "AVOID", "STRONG SELL",
}

# Ordered from most aggressive (BUY) to most conservative (AVOID).
# Used to enforce: CIO can restrict but never upgrade above score-mapped ceiling.
_ACTION_RISK_ORDER = [
    "BUY", "STARTER", "TACTICAL_BUY", "ACCUMULATE", "ADD",
    "HOLD", "WATCH", "DEFER", "TRIM", "SELL", "AVOID",
]


def _min_risk_action(mapped: str, cio: str) -> str:
    """Return the more conservative of two actions (score is the ceiling).

    BUY is most aggressive (index 0); AVOID is most conservative (last index).
    CIO can restrict below the mapped action but cannot upgrade above it.

    Examples:
        _min_risk_action("BUY", "WATCH")  -> "WATCH"  (CIO restricted)
        _min_risk_action("WATCH", "BUY")  -> "WATCH"  (score ceiling held)
        _min_risk_action("STARTER", "TACTICAL_BUY") -> "TACTICAL_BUY"
        _min_risk_action("AVOID", "BUY")  -> "AVOID"  (score ceiling held)
    """
    def _risk_idx(action: str) -> int:
        try:
            return _ACTION_RISK_ORDER.index(action.upper().strip())
        except ValueError:
            # Unknown action treated as WATCH (middle of the range)
            return _ACTION_RISK_ORDER.index("WATCH")

    mapped_idx = _risk_idx(mapped)
    cio_idx = _risk_idx(cio)
    # Higher index = more conservative; pick the more conservative one
    return _ACTION_RISK_ORDER[max(mapped_idx, cio_idx)]


def _infer_action_label(text: str) -> str:
    """Infer the first explicit action label from CIO prose."""
    if not text:
        return ""
    pattern = (
        r"\b(STRONG BUY|STRONG SELL|TACTICAL_BUY|STARTER|ACCUMULATE|ADD|"
        r"DEFER|WATCH|AVOID|BUY|HOLD|SELL|TRIM)\b"
    )
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).upper()


def _align_action_text_with_verdict(final_action: str, action_text: str) -> str:
    """Make the visible recommended-action label match the final verdict.

    The CIO can explain nuance after the label, but the first action label shown
    to the user must agree with the decision object used by Telegram, journals,
    dossiers, and watchlists.
    """
    final_action = str(final_action or "WATCH").upper().strip()
    action_text = str(action_text or "").strip()
    if final_action not in _VALID_FINAL_VERDICTS:
        return action_text
    if not action_text:
        return f"**{final_action}**"

    current_label = _infer_action_label(action_text)
    if current_label == final_action:
        return action_text

    action_pattern = re.compile(
        r"\b(STRONG BUY|STRONG SELL|TACTICAL_BUY|STARTER|ACCUMULATE|ADD|"
        r"DEFER|WATCH|AVOID|BUY|HOLD|SELL|TRIM)\b",
        re.IGNORECASE,
    )
    # Restrict replacement to the visible action lead. This avoids rewriting
    # later nuance such as "avoid if estimates are cut".
    lead = action_text[:220]
    match = action_pattern.search(lead)
    if match:
        return action_text[:match.start()] + final_action + action_text[match.end():]
    return f"**{final_action}** — {action_text}"


def _replace_recommended_action_line(synthesis: str, aligned_action: str) -> str:
    """Rewrite the RECOMMENDED ACTION line after final action normalization."""
    if not synthesis or not aligned_action:
        return synthesis
    pattern = re.compile(
        r"(\*\*RECOMMENDED ACTION:?\*\*\s*:?\s*)(.+)",
        re.IGNORECASE,
    )
    return pattern.sub(lambda m: m.group(1) + aligned_action, synthesis, count=1)


def _synchronize_final_action_surfaces(
    *,
    synthesis: str,
    scoring: dict,
    final_action: str,
    recommended_action: str,
    final_alloc_pct: float,
    no_new_capital: bool,
) -> tuple[str, str, dict]:
    """Synchronize final action across narrative, scoring JSON, and decision fields."""
    final_action = str(final_action or "WATCH").upper().strip()
    aligned_action = _align_action_text_with_verdict(final_action, recommended_action)
    updated_scoring = dict(scoring or {})
    changed = aligned_action != (recommended_action or "")

    if updated_scoring and final_action in _VALID_FINAL_VERDICTS:
        if str(updated_scoring.get("verdict") or "").upper() != final_action:
            changed = True
        updated_scoring["verdict"] = final_action
        updated_scoring["recommended_allocation_pct"] = 0.0 if no_new_capital else float(final_alloc_pct or 0)
        if no_new_capital:
            updated_scoring["stop_loss_pct"] = 0.0
            updated_scoring["target_pct"] = 0.0
        synthesis = _replace_last_scoring_json(synthesis, updated_scoring)

    if aligned_action != (recommended_action or ""):
        synthesis = _replace_recommended_action_line(synthesis, aligned_action)

    if changed and "ACTION NORMALIZATION NOTE:" not in (synthesis or ""):
        synthesis += (
            f"\n\nACTION NORMALIZATION NOTE: Final actionable verdict synchronized to {final_action} "
            "so the report header, action line, scoring JSON, journal, watchlist, and Telegram output use one label."
        )

    return synthesis, aligned_action, updated_scoring


def _normalize_no_buy_action(
    final_action: str,
    cio_verdict: str,
    recommended_action: str,
    score_components: dict,
    scoring_invalid: bool = False,
) -> str:
    """Keep no-buy verdicts semantically consistent.

    AVOID means "do not touch / structural problem." DEFER means "interesting
    but wrong price or timing." The score ceiling still blocks buy-side actions,
    but it should not mislabel a good-business pullback setup as AVOID.
    """
    final_action = str(final_action or "WATCH").upper()
    cio_verdict = str(cio_verdict or "").upper()
    rec_label = _infer_action_label(recommended_action)

    if scoring_invalid:
        return "AVOID"
    if rec_label == "AVOID" and final_action in {"WATCH", "DEFER"}:
        return "AVOID"
    if (
        final_action == "AVOID"
        and (cio_verdict == "DEFER" or rec_label == "DEFER")
        and float((score_components or {}).get("fundamental_quality") or 0) >= 10
    ):
        return "DEFER"
    if (
        final_action == "AVOID"
        and (cio_verdict == "WATCH" or rec_label == "WATCH")
        and float((score_components or {}).get("fundamental_quality") or 0) >= 7
    ):
        return "WATCH"
    return final_action


def _extract_json_object(text: str) -> Optional[dict]:
    """Try to parse a JSON object from fenced or raw text."""
    candidates: list[str] = []

    json_block = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_block:
        candidates.append(json_block.group(1))

    generic_block = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    if generic_block:
        candidates.append(generic_block.group(1))

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return None


_SCORING_COMPONENTS = {
    "technical_setup":      (0, 25),
    "fundamental_quality":  (0, 20),
    "contrarian_sentiment": (0, 15),
    "regime_alignment":     (0, 15),
    "catalyst_asymmetry":   (0, 10),
    "data_quality":         (0, 10),
    "liquidity_execution":  (0, 5),
}


def _validate_scoring_json(parsed: dict) -> bool:
    """Full schema validation for CIO scoring JSON.

    Returns True only if the block satisfies all constraints.
    Returns False on any violation (triggers conservative fallback or safe repair).
    """
    import math

    # opportunity_score: int 0-100
    raw_score = parsed.get("opportunity_score")
    if raw_score is None:
        return False
    try:
        score_int = int(raw_score)
    except (TypeError, ValueError):
        return False
    if not (0 <= score_int <= 100):
        return False

    # All 7 component keys must exist and be numeric within range
    components = parsed.get("components")
    if not isinstance(components, dict):
        return False
    component_sum = 0
    for key, (lo, hi) in _SCORING_COMPONENTS.items():
        val = components.get(key)
        if val is None:
            return False
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return False
        if math.isnan(fval) or math.isinf(fval):
            return False
        if not (lo <= fval <= hi):
            return False
        component_sum += fval

    # Component sum should be near the scoring base. In the hybrid buy model the
    # final opportunity_score may differ because rule/CIO adjustments are added.
    expected_component_total = parsed.get("deterministic_base_score", score_int)
    try:
        expected_component_total = int(expected_component_total)
    except (TypeError, ValueError):
        expected_component_total = score_int
    if abs(component_sum - expected_component_total) > 5:
        logger.warning(
            "Scoring JSON component sum %.1f differs from expected base score %d by %.1f points",
            component_sum, expected_component_total, abs(component_sum - expected_component_total),
        )

    # verdict must be a known verdict type
    verdict = str(parsed.get("verdict", "")).upper().strip()
    if verdict not in _VALID_FINAL_VERDICTS:
        return False

    # confidence: 1-10
    conf = parsed.get("confidence")
    if conf is None:
        return False
    try:
        conf_int = int(conf)
    except (TypeError, ValueError):
        return False
    if not (1 <= conf_int <= 10):
        return False

    no_buy_verdicts = {"HOLD", "WATCH", "DEFER", "AVOID", "SELL", "TRIM", "STRONG SELL"}
    buy_like_verdicts = {"BUY", "STARTER", "TACTICAL_BUY", "ACCUMULATE", "ADD", "STRONG BUY"}

    # recommended_allocation_pct must be plausible. No-buy verdicts should not allocate.
    alloc = parsed.get("recommended_allocation_pct", 0)
    try:
        alloc_f = float(alloc or 0)
    except (TypeError, ValueError):
        return False
    if math.isnan(alloc_f) or math.isinf(alloc_f) or alloc_f < 0:
        return False
    if verdict in no_buy_verdicts and alloc_f > 0:
        return False
    if verdict in buy_like_verdicts and alloc_f <= 0:
        return False

    # stop_loss_pct must be negative for a live entry. No-new-capital verdicts
    # may use 0.0 because there is no actual trade to stop out.
    stop = parsed.get("stop_loss_pct")
    if stop is None:
        return False
    try:
        stop_f = float(stop)
    except (TypeError, ValueError):
        return False
    if verdict in buy_like_verdicts and stop_f >= 0:
        return False
    if verdict in no_buy_verdicts and stop_f > 0:
        return False

    # target_pct may be 0.0 for no-buy verdicts. Buy-like verdicts need a
    # positive target so the CIO cannot recommend an entry without upside.
    target = parsed.get("target_pct")
    if target is None:
        return False
    try:
        target_f = float(target)
    except (TypeError, ValueError):
        return False
    if math.isnan(target_f) or math.isinf(target_f):
        return False
    if verdict in buy_like_verdicts and target_f <= 0:
        return False
    if verdict in no_buy_verdicts and target_f < 0:
        return False

    return True


def _repair_scoring_json(parsed: dict) -> Optional[dict]:
    """Conservatively repair minor CIO scoring JSON issues.

    This prevents a tiny numeric schema error from falling through to a normal
    WATCH default. Buy-like verdicts remain strict and are never auto-repaired.
    """
    import copy
    import math

    if not isinstance(parsed, dict):
        return None
    repaired = copy.deepcopy(parsed)

    verdict = str(repaired.get("verdict", "")).upper().strip()
    if verdict not in _VALID_FINAL_VERDICTS:
        return None
    no_buy_verdicts = {"HOLD", "WATCH", "DEFER", "AVOID", "SELL", "TRIM", "STRONG SELL"}
    buy_like_verdicts = {"BUY", "STARTER", "TACTICAL_BUY", "ACCUMULATE", "ADD", "STRONG BUY"}

    components = repaired.get("components")
    if not isinstance(components, dict):
        return None
    if not all(key in components for key in _SCORING_COMPONENTS):
        return None

    changed = False
    component_sum = 0.0
    for key, (lo, hi) in _SCORING_COMPONENTS.items():
        try:
            value = float(components.get(key))
        except (TypeError, ValueError):
            return None
        if math.isnan(value) or math.isinf(value):
            return None
        clamped = max(float(lo), min(float(hi), value))
        if clamped != value:
            changed = True
            components[key] = int(clamped) if clamped.is_integer() else clamped
        component_sum += clamped

    try:
        score = int(repaired.get("opportunity_score"))
    except (TypeError, ValueError):
        score = int(round(component_sum))
        changed = True
    corrected_score = int(round(max(0.0, min(100.0, component_sum))))
    if abs(corrected_score - score) > 5:
        repaired["opportunity_score"] = corrected_score
        changed = True
    else:
        repaired["opportunity_score"] = max(0, min(100, score))

    try:
        conf = int(repaired.get("confidence"))
    except (TypeError, ValueError):
        conf = 5
        changed = True
    clamped_conf = max(1, min(10, conf))
    if clamped_conf != conf:
        changed = True
    repaired["confidence"] = clamped_conf

    if verdict in no_buy_verdicts:
        if float(repaired.get("recommended_allocation_pct") or 0) != 0:
            changed = True
        repaired["recommended_allocation_pct"] = 0.0
        for field in ("stop_loss_pct", "target_pct"):
            try:
                value = float(repaired.get(field, 0) or 0)
            except (TypeError, ValueError):
                value = 0.0
                changed = True
            if value > 0:
                value = 0.0
                changed = True
            repaired[field] = value
    elif verdict in buy_like_verdicts:
        return None

    if changed and _validate_scoring_json(repaired):
        repaired["_schema_repaired"] = True
        return repaired
    return None


def _extract_scoring_json(synthesis: str) -> Optional[dict]:
    """Extract the CIO scoring JSON block from synthesis output.

    The scoring block is the LAST ```json block in the output (after narrative).
    Full schema validation is applied; returns None if validation fails.
    """
    try:
        # Find all ```json blocks
        json_blocks = re.findall(r"```json\s*(\{.*?\})\s*```", synthesis, re.DOTALL)
        if not json_blocks:
            # Try generic code blocks
            json_blocks = re.findall(r"```\s*(\{.*?\})\s*```", synthesis, re.DOTALL)

        # The scoring block is the last one and must have 'opportunity_score'
        for raw in reversed(json_blocks):
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict) or "opportunity_score" not in parsed:
                    continue
                # Legacy scoring used component sum as the entire score. Hybrid
                # scoring uses component sum as deterministic base and then adds
                # rule/CIO adjustments, so do not overwrite opportunity_score
                # when the new audit fields are present.
                has_hybrid_audit_fields = any(
                    key in parsed
                    for key in (
                        "deterministic_base_score",
                        "rule_adjustment_total",
                        "deterministic_score_before_cio",
                        "cio_score_adjustment",
                    )
                )
                components = parsed.get("components")
                if isinstance(components, dict) and not has_hybrid_audit_fields:
                    try:
                        component_sum = sum(
                            float(v) for v in components.values()
                            if isinstance(v, (int, float))
                        )
                        parsed["opportunity_score"] = int(round(component_sum))
                    except (TypeError, ValueError):
                        pass
                if _validate_scoring_json(parsed):
                    return parsed
                repaired = _repair_scoring_json(parsed)
                if repaired:
                    logger.warning("Scoring JSON had minor schema errors — repaired safely")
                    return repaired
                logger.warning("Scoring JSON failed schema validation — rejecting block")
                return None
            except (json.JSONDecodeError, ValueError):
                continue
    except Exception as exc:
        logger.warning("_extract_scoring_json raised unexpectedly: %s", exc)
    return None


# Ops and metrics the sell engine can actually evaluate (see
# thesis_tracker._CONDITION_OPS and sell_engine._check_thesis_break's
# observed-metrics dict — keep in sync). pct_change_* ops were dropped:
# they had no evaluation semantics anywhere, so conditions using them
# were stored as permanently-inert dead weight.
_DECISION_OPS = {"lt", "lte", "gt", "gte", "eq"}
_DECISION_METRICS = {"price", "pnl_pct", "days_held", "trading_days_held"}
_DECISION_METRIC_ALIASES = {
    "close_price": "price",
    "last_price": "price",
    "current_price": "price",
    "days_elapsed": "days_held",
}


def _extract_decision_json(synthesis: str) -> Optional[dict]:
    """Extract the CIO structured DECISION block (Wave 2).

    Schema (machine-consumed by the scheduler agent; keep in sync with
    prompts.SYNTHESIS_PROMPT "STRUCTURED DECISION BLOCK"):
        {
          "verdict": str,                      # one of _VALID_FINAL_VERDICTS
          "conviction_0_10": int,
          "entry_zone": {"low": float, "high": float,
                         "valid_until_iso": "YYYY-MM-DD"} | null,
          "stop_price": float | null,          # ATR-derived, cap 10% below entry
          "target_price": float | null,
          "invalidation_conditions": [
            {"metric": str, "op": str, "value": number, "description": str}
          ],
          "half_size": bool
        }

    Identified as a fenced JSON block containing "conviction_0_10" (the
    scoring block is identified separately by "opportunity_score"). Returns a
    normalized dict or None — callers keep the legacy regex/scoring-block
    fallback when this block is missing or malformed.
    """
    try:
        json_blocks = re.findall(r"```json\s*(\{.*?\})\s*```", synthesis or "", re.DOTALL)
        if not json_blocks:
            json_blocks = re.findall(r"```\s*(\{.*?\})\s*```", synthesis or "", re.DOTALL)
        for raw in reversed(json_blocks):
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(parsed, dict) or "conviction_0_10" not in parsed:
                continue
            return _normalize_decision_json(parsed)
    except Exception as exc:
        logger.warning("_extract_decision_json raised unexpectedly: %s", exc)
    return None


def _normalize_decision_json(parsed: dict) -> Optional[dict]:
    """Validate/coerce a raw CIO decision block. Returns None if unusable."""
    def _float_or_none(value) -> Optional[float]:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        import math as _math
        if _math.isnan(result) or _math.isinf(result):
            return None
        return result

    verdict = str(parsed.get("verdict") or "").upper().strip()
    if verdict not in _VALID_FINAL_VERDICTS:
        return None
    try:
        conviction = int(parsed.get("conviction_0_10"))
    except (TypeError, ValueError):
        return None
    conviction = max(0, min(10, conviction))

    entry_zone = parsed.get("entry_zone")
    if isinstance(entry_zone, dict):
        low = _float_or_none(entry_zone.get("low"))
        high = _float_or_none(entry_zone.get("high"))
        if low is None or high is None or low <= 0 or high < low:
            entry_zone = None
        else:
            entry_zone = {
                "low": low,
                "high": high,
                "valid_until_iso": str(entry_zone.get("valid_until_iso") or "")[:10],
            }
    else:
        entry_zone = None

    stop_price = _float_or_none(parsed.get("stop_price"))
    if stop_price is not None and stop_price <= 0:
        stop_price = None
    target_price = _float_or_none(parsed.get("target_price"))
    if target_price is not None and target_price <= 0:
        target_price = None

    conditions: list[dict] = []
    for item in parsed.get("invalidation_conditions") or []:
        if not isinstance(item, dict):
            continue
        metric = str(item.get("metric") or "").strip()
        metric = _DECISION_METRIC_ALIASES.get(metric.lower(), metric.lower())
        op = str(item.get("op") or "").strip().lower()
        value = _float_or_none(item.get("value"))
        description = str(item.get("description") or "").strip()
        if metric not in _DECISION_METRICS or op not in _DECISION_OPS or value is None:
            if metric or op:
                logger.info(
                    "Dropping unevaluable invalidation condition metric=%r op=%r "
                    "(supported metrics=%s ops=%s)",
                    metric, op, sorted(_DECISION_METRICS), sorted(_DECISION_OPS),
                )
            continue
        conditions.append({
            "metric": metric,
            "op": op,
            "value": value,
            "description": description,
        })

    return {
        "verdict": verdict,
        "conviction_0_10": conviction,
        "entry_zone": entry_zone,
        "stop_price": stop_price,
        "target_price": target_price,
        "invalidation_conditions": conditions,
        "half_size": bool(parsed.get("half_size")),
    }


def _compute_atr14(price_history: list | None) -> Optional[float]:
    """ATR(14) via Wilder smoothing from daily OHLC bars (oldest-first)."""
    if not isinstance(price_history, list) or len(price_history) < 15:
        return None
    try:
        bars = sorted(
            (b for b in price_history if isinstance(b, dict) and b.get("date")),
            key=lambda b: str(b.get("date")),
        )
        true_ranges: list[float] = []
        prev_close: Optional[float] = None
        for bar in bars:
            high = float(bar.get("high") or 0)
            low = float(bar.get("low") or 0)
            close = float(bar.get("close") or 0)
            if high <= 0 or low <= 0 or close <= 0:
                prev_close = close if close > 0 else prev_close
                continue
            if prev_close is None:
                tr = high - low
            else:
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)
            prev_close = close
        if len(true_ranges) < 14:
            return None
        atr = sum(true_ranges[:14]) / 14.0
        for tr in true_ranges[14:]:
            atr = (atr * 13 + tr) / 14.0
        return round(atr, 4)
    except (TypeError, ValueError):
        return None


def _atr_initial_stop_pct(current_price: Optional[float], atr14: Optional[float], config) -> Optional[float]:
    """Initial stop as a NEGATIVE fraction: 2.5x ATR14 below entry, cap 10%."""
    try:
        price = float(current_price or 0)
    except (TypeError, ValueError):
        return None
    if price <= 0 or not atr14 or atr14 <= 0:
        return None
    stop_frac = (float(getattr(config, "CIO_STOP_ATR_MULT", 2.5)) * atr14) / price
    stop_frac = min(stop_frac, float(getattr(config, "CIO_STOP_CAP_PCT", 0.10)))
    return -round(stop_frac, 4)


def _mechanical_composite_strong(signal_row: dict | None, config) -> bool:
    """True when the funnel composite is STRONG per the Wave 2 rulebook:
    scan track momentum/breakout with z >= 1.5 AND trend template pass."""
    if not isinstance(signal_row, dict) or not signal_row:
        return False
    track = str(signal_row.get("track") or "").lower()
    if track not in ("momentum", "breakout"):
        return False
    if not bool(signal_row.get("trend_template")):
        return False
    min_z = float(getattr(config, "MECHANICAL_OVERRIDE_MIN_Z", 1.5))
    for key in ("momentum_z", "composite_z"):
        try:
            if signal_row.get(key) is not None and float(signal_row[key]) >= min_z:
                return True
        except (TypeError, ValueError):
            continue
    return False


# One-notch CIO upgrade ladder over the score-mapped action (mechanical
# override; AVOID is deliberately NOT upgradeable — a sub-45 score stays out).
_BUY_NOTCH_UPGRADE = {
    "WATCH": "TACTICAL_BUY",
    "TACTICAL_BUY": "STARTER",
    "STARTER": "BUY",
}


def _replace_last_scoring_json(synthesis: str, scoring: dict) -> str:
    """Replace the last scoring JSON block with the validated/corrected payload."""
    try:
        visible_scoring = {k: v for k, v in (scoring or {}).items() if not str(k).startswith("_")}
        pattern = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
        matches = list(pattern.finditer(synthesis or ""))
        for match in reversed(matches):
            try:
                parsed = json.loads(match.group(1))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            if isinstance(parsed, dict) and "opportunity_score" in parsed:
                replacement = "```json\n" + json.dumps(visible_scoring, indent=2) + "\n```"
                return synthesis[:match.start()] + replacement + synthesis[match.end():]
    except Exception as exc:
        logger.warning("Failed to normalize scoring JSON block: %s", exc)
    return synthesis


def _parse_synthesis(synthesis: str) -> dict:
    """Extract structured fields from synthesis text with multi-strategy parsing."""
    result = {
        "final_verdict": "WATCH",
        "consensus": "Split",
        "recommended_action": "",
        "allocation": "",
    }

    # Strategy 1: Markdown regex fallback (we no longer look for a top-level JSON object
    # since the JSON is now a scoring sub-block at the end)
    patterns = {
        "final_verdict": r"\*\*FINAL VERDICT:?\*\*\s*:?\s*(STRONG BUY|STRONG SELL|STARTER|TACTICAL_BUY|ACCUMULATE|ADD|TRIM|DEFER|BUY|HOLD|WATCH|AVOID|SELL)",
        "consensus": r"\*\*COUNCIL CONSENSUS:?\*\*\s*:?\s*(.+)",
        "recommended_action": r"\*\*RECOMMENDED ACTION:?\*\*\s*:?\s*(.+)",
        "allocation": r"\*\*ALLOCATION:?\*\*\s*:?\s*(.+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, synthesis, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if key == "final_verdict":
                value = value.upper()
                if value in _VALID_FINAL_VERDICTS:
                    result[key] = value
            else:
                result[key] = value

    # Strategy 2: Loose text fallback
    fv_loose = re.search(
        r"final verdict\s*:?\s*(strong buy|strong sell|starter|tactical_buy|accumulate|add|trim|defer|buy|hold|watch|avoid|sell)",
        synthesis, re.IGNORECASE
    )
    if fv_loose:
        candidate = fv_loose.group(1).upper().strip()
        if candidate in _VALID_FINAL_VERDICTS:
            result["final_verdict"] = candidate

    cons_loose = re.search(r"council consensus\s*:?\s*(.+)", synthesis, re.IGNORECASE)
    if cons_loose and result["consensus"] == "Split":
        result["consensus"] = cons_loose.group(1).strip()
    action_loose = re.search(r"recommended action\s*:?\s*(.+)", synthesis, re.IGNORECASE)
    if action_loose and not result["recommended_action"]:
        result["recommended_action"] = action_loose.group(1).strip()
    alloc_loose = re.search(r"allocation\s*:?\s*(.+)", synthesis, re.IGNORECASE)
    if alloc_loose and not result["allocation"]:
        result["allocation"] = alloc_loose.group(1).strip()

    return result


# ---------------------------------------------------------------------------
# v2: Hard Risk Gate
# ---------------------------------------------------------------------------

def hard_risk_gate(
    ticker: str,
    stock_data: dict,
    portfolio_state: dict,
    sentinel_alerts: list,
    config,
) -> tuple[bool, str]:
    """Stage 1: Non-negotiable safety checks. Binary pass/fail.

    Returns (passed: bool, rejection_reason: str).
    Checks run in priority order — first failure short-circuits.
    """
    # Fail-closed guard: portfolio state must have minimum required fields
    if not isinstance(portfolio_state, dict):
        return False, "Insufficient portfolio data for risk assessment"
    _required_keys = ("cash_available", "total_value", "positions")
    if any(portfolio_state.get(k) is None for k in _required_keys):
        return False, "Insufficient portfolio data for risk assessment"

    # Check 1: Severe sentinel alert for this ticker
    if sentinel_alerts:
        ticker_upper = ticker.upper()
        for alert in sentinel_alerts:
            if not isinstance(alert, dict):
                continue
            alert_ticker = str(alert.get("ticker", "")).upper()
            severity = str(alert.get("severity", "")).upper()
            if alert_ticker == ticker_upper and severity == "CRITICAL":
                headline = str(alert.get("headline", alert.get("title", ""))[:100])
                return False, f"SENTINEL BLOCK: CRITICAL alert — {headline}"

    # Check 2: Portfolio concentration (max concurrent positions)
    positions = portfolio_state.get("positions", []) or []
    position_count = len(positions)
    # Check if ticker is already a position (then this is an ADD, not blocked)
    existing_tickers = {str(p.get("ticker", "")).upper() for p in positions if isinstance(p, dict)}
    is_new_position = ticker.upper() not in existing_tickers
    if is_new_position and position_count >= config.MAX_CONCURRENT_POSITIONS:
        return False, (
            f"POSITION LIMIT: Already at max {config.MAX_CONCURRENT_POSITIONS} positions "
            f"({position_count} held). Cannot open new position."
        )

    # Check 3: Total invested percentage
    try:
        total_value = float(portfolio_state.get("total_value") or 0)
    except (TypeError, ValueError):
        total_value = 0.0
    try:
        cash = float(portfolio_state.get("cash_available") or 0)
    except (TypeError, ValueError):
        cash = 0.0
    import math as _math
    if _math.isnan(total_value) or _math.isinf(total_value):
        total_value = 0.0
    if _math.isnan(cash) or _math.isinf(cash):
        cash = 0.0

    if total_value > 0:
        invested_pct = (total_value - cash) / total_value
        if invested_pct >= config.MAX_INVESTED_PCT and is_new_position:
            return False, (
                f"INVESTED LIMIT: {invested_pct:.0%} of NAV already deployed "
                f"(max {config.MAX_INVESTED_PCT:.0%}). Preserve cash buffer."
            )

    # Check 4: Sector concentration
    profile = stock_data.get("profile") or {}
    sector = str(
        profile.get("sector") or
        (stock_data.get("yf_quote") or {}).get("sector") or ""
    ).strip()
    if sector and total_value > 0 and is_new_position:
        sector_value = 0.0
        for p in positions:
            if not isinstance(p, dict) or str(p.get("sector", "")).strip() != sector:
                continue
            try:
                sector_value += float(p.get("market_value") or 0)
            except (TypeError, ValueError):
                pass
        sector_pct = sector_value / total_value
        if sector_pct >= config.MAX_SECTOR_PCT:
            return False, (
                f"SECTOR LIMIT: Already {sector_pct:.0%} in '{sector}' "
                f"(max {config.MAX_SECTOR_PCT:.0%})."
            )

    # Check 5: Existential fundamental risk (extreme leverage or negative equity)
    ratios_ttm = stock_data.get("ratios_ttm") or {}
    key_metrics = stock_data.get("key_metrics_ttm") or {}
    de_ratio = ratios_ttm.get("debtToEquityRatioTTM")
    if de_ratio is not None:
        try:
            de_float = float(de_ratio)
            if de_float > 10:
                return False, (
                    f"EXISTENTIAL RISK: Extreme leverage D/E = {de_float:.1f}x "
                    f"(threshold: 10x). Risk of insolvency."
                )
        except (TypeError, ValueError):
            pass
    # Check for negative equity / book value
    book_per_share = key_metrics.get("bookValuePerShareTTM")
    if book_per_share is not None:
        try:
            if float(book_per_share) < -5:  # Allow small negative, but not deeply negative
                return False, (
                    f"EXISTENTIAL RISK: Deeply negative book value per share "
                    f"(${float(book_per_share):.2f}). Insolvency risk."
                )
        except (TypeError, ValueError):
            pass

    # Check 6: Minimum liquidity. Only true average volume can hard-fail.
    # Current intraday volume at market open is not the same thing and must not
    # make an otherwise liquid stock look untradeable.
    try:
        from .liquidity import resolve_average_volume

        volume_info = resolve_average_volume(stock_data)
        avg_volume_raw = volume_info.get("volume")
        if avg_volume_raw is not None:
            avg_vol_float = float(avg_volume_raw)
            if volume_info.get("is_average") and avg_vol_float > 0 and avg_vol_float < 100_000:
                return False, (
                    f"LIQUIDITY: Avg daily volume {avg_vol_float:,.0f} from {volume_info.get('source')} "
                    f"is below minimum threshold of 100,000 shares."
                )
            if not volume_info.get("is_average") and 0 < avg_vol_float < 100_000:
                logger.info(
                    "  ℹ️ Liquidity gate did not hard-fail on current/intraday volume %.0f from %s; true ADV unavailable.",
                    avg_vol_float,
                    volume_info.get("source"),
                )
    except Exception as exc:
        logger.warning("  ⚠️ Liquidity resolver failed for hard risk gate: %s", exc)

    return True, ""


# ---------------------------------------------------------------------------
# v2: Score-to-Action Mapping
# ---------------------------------------------------------------------------

def score_to_action(
    score: int | None,
    fear_greed: int | None,
    portfolio_state: dict,
    config,
    regime_gate_state: dict | None = None,
) -> dict:
    """Map opportunity score to concrete action and position size.

    Wave 2: the fear/greed re-penalty is GONE — regime already contributes
    15/100 inside buy scoring, and double-counting it here is what banned
    tactical buys in uptrends (precisely backwards). Entry permission now
    comes from the mechanical regime gate (artha/regime_gate.py):

      RISK_ON        — thresholds as configured (BUY>=75/STARTER>=65/TACTICAL>=55)
      RISK_OFF       — TACTICAL requires score >= 65; buy actions marked
                       half_size=True (council additionally requires
                       conviction >= 8 for momentum/breakout tracks)
      HARD_RISK_OFF  — no new buys; buy-side maps cap at WATCH

    The gate NEVER blocks exits — this mapper only shapes NEW entries.
    ``fear_greed`` is retained for signature compatibility and reporting only.
    """
    import math as _math

    # Guard: None/NaN score defaults to 0 → AVOID
    if score is None:
        score = 0
    try:
        score = int(score)
    except (TypeError, ValueError):
        score = 0
    if _math.isnan(score) if isinstance(score, float) else False:
        score = 0

    # Resolve regime gate state (no network: cached file only, default RISK_ON)
    gate = regime_gate_state
    if not isinstance(gate, dict) or not gate.get("state"):
        try:
            from .regime_gate import load_cached_regime
            gate = load_cached_regime() or {}
        except Exception:
            gate = {}
    regime_state = str(gate.get("state") or "RISK_ON").upper()
    if regime_state not in ("RISK_ON", "RISK_OFF", "HARD_RISK_OFF"):
        regime_state = "RISK_ON"
    try:
        vol_multiplier = float(gate.get("vol_multiplier", 1.0) or 1.0)
    except (TypeError, ValueError):
        vol_multiplier = 1.0
    vol_multiplier = max(0.1, min(1.0, vol_multiplier))

    adjusted_score = max(0, min(100, score))
    half_size = False

    # Total NAV for sizing
    total_nav = float(portfolio_state.get("total_value", 500) or 500)

    # Action mapping (thresholds as configured; gate shapes entries)
    risk_off_tactical_floor = int(getattr(
        config, "REGIME_GATE_RISK_OFF_MIN_TACTICAL_SCORE", config.SCORE_THRESHOLD_STARTER
    ))
    if adjusted_score >= config.SCORE_THRESHOLD_BUY:
        action = "BUY"
        # Size: 12-18% NAV, scaled by score above threshold
        scale = min(1.0, (adjusted_score - config.SCORE_THRESHOLD_BUY) / 25)
        alloc_pct = 12.0 + scale * 6.0
    elif adjusted_score >= config.SCORE_THRESHOLD_STARTER:
        action = "STARTER"
        # Size: 5-8% NAV
        scale = (adjusted_score - config.SCORE_THRESHOLD_STARTER) / 10
        alloc_pct = 5.0 + scale * 3.0
    elif adjusted_score >= config.SCORE_THRESHOLD_TACTICAL:
        action = "TACTICAL_BUY"
        scale = (adjusted_score - config.SCORE_THRESHOLD_TACTICAL) / 10
        alloc_pct = 3.0 + scale * 2.0
    elif adjusted_score >= 45:
        action = "WATCH"
        alloc_pct = 0.0
    else:
        action = "AVOID"
        alloc_pct = 0.0

    _BUY_SIDE = ("BUY", "STARTER", "TACTICAL_BUY")
    if regime_state == "HARD_RISK_OFF" and action in _BUY_SIDE:
        # No new buys — cap at WATCH. Exits are handled elsewhere and never gated.
        action = "WATCH"
        alloc_pct = 0.0
    elif regime_state == "RISK_OFF" and action in _BUY_SIDE:
        if action == "TACTICAL_BUY" and adjusted_score < risk_off_tactical_floor:
            action = "WATCH"
            alloc_pct = 0.0
        else:
            # Mark only — callers apply the halving once to the FINAL
            # allocation (which may come from the CIO instead of this map).
            half_size = True

    # Volatility-targeted sizing: clamp(20/VIX, 0.5, 1.0) scales size, not verdict.
    if action in _BUY_SIDE:
        alloc_pct *= vol_multiplier

    # Cap at MAX_POSITION_PCT
    alloc_pct = min(alloc_pct, config.MAX_POSITION_PCT * 100)
    dollar_amount = total_nav * alloc_pct / 100

    return {
        "action": action,
        "raw_score": score,
        "adjusted_score": adjusted_score,
        # regime_label now carries the mechanical gate state (legacy key name
        # kept because journals/shadow logs/thesis records already read it).
        "regime_label": regime_state,
        "regime_state": regime_state,
        "regime_adjustment": 0,
        "vol_multiplier": round(vol_multiplier, 3),
        "half_size": half_size,
        "recommended_allocation_pct": round(alloc_pct, 1),
        "recommended_dollar_amount": round(dollar_amount, 2),
    }


# ---------------------------------------------------------------------------
# Crisis Mode v3 — Council Convergence Score + Trust Gates
# ---------------------------------------------------------------------------

_DRIVER_CATEGORIES: dict[str, list[str]] = {
    "VALUE": ["undervalued", "cheap", "discount", "dcf", "p/e", "margin of safety", "fair value", "intrinsic"],
    "TECHNICAL": ["oversold", "rsi", "support", "macd", "volume", "momentum", "bottoming", "technical"],
    "BALANCE_SHEET": ["cash flow", "debt-to-equity", "current ratio", "liquidity", "solvency", "balance sheet", "interest coverage"],
    "GROWTH": ["revenue growth", "market share", "tam", "competitive", "moat", "innovation", "earnings growth"],
    "CATALYST": ["catalyst", "turnaround", "restructuring", "new product", "regulatory", "buyback"],
    "RISK_ADJUSTED": ["downside limited", "asymmetric", "risk-reward", "worst case", "risk-adjusted", "downside protection"],
}

_TRUST_GATES: dict[str, dict] = {
    "BANKING": {
        "veto_analyst": "contrarian",
        "gate_sectors": ["Financial Services", "Insurance"],
        "rationale": "Balance sheet skeptic (GPT) vetoes financials during credit crisis",
    },
    "TECH_BUBBLE": {
        "veto_analyst": "fundamental",
        "gate_sectors": ["Technology", "Communication Services"],
        "rationale": "Value analyst vetoes overvalued tech during valuation unwind",
    },
    "PANDEMIC": {
        "veto_analyst": None,
        "gate_sectors": [],
        "rationale": "No specialist veto — pattern recognition enhanced for rotation",
    },
    "GEOPOLITICAL": {
        "veto_analyst": "contrarian",
        "gate_sectors": [],
        "rationale": "Risk analyst enhanced for geopolitical tail risk assessment",
    },
    "STAGFLATION": {
        "veto_analyst": None,
        "gate_sectors": [],
        "rationale": "Value/quality assessment most important during stagflation",
    },
}


def _extract_primary_driver(report_text: str) -> Optional[str]:
    """Parse analyst report to identify primary analytical driver."""
    explicit = re.search(r"PRIMARY DRIVER:\s*(\w+)", report_text, re.IGNORECASE)
    if explicit:
        declared = explicit.group(1).upper().strip()
        if declared in _DRIVER_CATEGORIES:
            return declared

    text_lower = report_text.lower()
    scores: dict[str, int] = {}
    for category, keywords in _DRIVER_CATEGORIES.items():
        scores[category] = sum(text_lower.count(kw.lower()) for kw in keywords)

    if scores and max(scores.values()) > 0:
        return max(scores, key=scores.get)
    return None


def compute_convergence_score(
    fundamental: "AnalystReport",
    technical: "AnalystReport",
    contrarian: "AnalystReport",
) -> dict:
    """Compute Council Convergence Score (CCS) with orthogonality bonus."""
    reports = [fundamental, technical, contrarian]
    buy_count = sum(1 for r in reports if r.verdict == "BUY")
    base_score = buy_count * 2

    avg_confidence = sum(r.confidence for r in reports) / 3
    confidence_bonus = 2 if avg_confidence >= 7 and buy_count >= 2 else 0

    drivers = [_extract_primary_driver(r.report) for r in reports]
    unique_drivers = len({d for d in drivers if d is not None})
    if unique_drivers >= 3 and buy_count >= 2:
        orthogonality_bonus = 2
    elif unique_drivers >= 2 and buy_count >= 2:
        orthogonality_bonus = 1
    else:
        orthogonality_bonus = 0

    contrarian_bonus = 2 if (contrarian.verdict == "BUY" and contrarian.confidence >= 6) else 0

    total = min(12, base_score + confidence_bonus + orthogonality_bonus + contrarian_bonus)

    tier = (
        "TRIPLE_CROWN" if total >= Config.CRISIS_CCS_TRIPLE_CROWN
        else "HIGH_CONVICTION" if total >= Config.CRISIS_CCS_HIGH_CONVICTION
        else "STANDARD" if total >= Config.CRISIS_CCS_STANDARD
        else "LOW_CONVICTION"
    )

    return {
        "ccs": total,
        "buy_count": buy_count,
        "avg_confidence": round(avg_confidence, 1),
        "orthogonality": unique_drivers,
        "drivers": drivers,
        "contrarian_convinced": contrarian.verdict == "BUY",
        "tier": tier,
        "breakdown": {
            "base": base_score,
            "confidence_bonus": confidence_bonus,
            "orthogonality_bonus": orthogonality_bonus,
            "contrarian_bonus": contrarian_bonus,
        },
    }


def apply_trust_gates(
    ticker: str,
    sector: str,
    dominant_crisis_type: str,
    fundamental: "AnalystReport",
    technical: "AnalystReport",
    contrarian: "AnalystReport",
) -> dict:
    """Apply regime-conditional trust gates."""
    gate_config = _TRUST_GATES.get(dominant_crisis_type.upper(), {})
    gate_result: dict = {"gate_applied": False, "vetoed": False, "details": ""}

    veto_analyst_name = gate_config.get("veto_analyst")
    gate_sectors = gate_config.get("gate_sectors", [])

    sector_matches = not gate_sectors or sector in gate_sectors

    if veto_analyst_name and sector_matches:
        analyst_map = {
            "fundamental": fundamental,
            "technical": technical,
            "contrarian": contrarian,
        }
        veto_report = analyst_map.get(veto_analyst_name)

        if veto_report and veto_report.verdict in ("SELL", "HOLD") and veto_report.confidence >= 7:
            gate_result = {
                "gate_applied": True,
                "vetoed": True,
                "veto_by": veto_analyst_name,
                "veto_confidence": veto_report.confidence,
                "details": (
                    f"⚠️ TRUST GATE ACTIVATED: {veto_analyst_name} analyst vetoed {ticker} "
                    f"(verdict: {veto_report.verdict}, confidence: {veto_report.confidence}/10) "
                    f"during {dominant_crisis_type} crisis. {gate_config.get('rationale', '')}"
                ),
            }
        elif veto_report:
            gate_result = {
                "gate_applied": True,
                "vetoed": False,
                "details": (
                    f"Trust gate checked ({dominant_crisis_type}): {veto_analyst_name} "
                    f"did not trigger veto (verdict: {veto_report.verdict})"
                ),
            }

    return gate_result


# ---------------------------------------------------------------------------
# Council Runner
# ---------------------------------------------------------------------------

class ArthaCouncil:
    """Orchestrates the three-analyst debate and synthesis (v2 decision engine)."""

    def __init__(self) -> None:
        self.profile_path = Path(__file__).resolve().parent.parent / "data" / "config" / "investor_profile.yaml"
        self.portfolio_state = PortfolioStateEngine()
        self.journal = DecisionJournal()
        self.research_desk = ResearchDesk()

    def _load_investor_profile(self) -> dict:
        """Load investor profile YAML with safe fallback."""
        if not self.profile_path.exists():
            logger.warning("Investor profile missing at %s", self.profile_path)
            return {}
        if yaml is None:
            logger.warning("PyYAML unavailable; using lightweight investor profile parser")
            try:
                with open(self.profile_path, encoding="utf-8") as f:
                    return {"_raw_yaml": f.read()}
            except Exception as exc:
                logger.error("Failed to read investor profile %s: %s", self.profile_path, exc)
                return {}
        try:
            with open(self.profile_path, encoding="utf-8") as f:
                payload = yaml.safe_load(f)
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            logger.error("Failed to parse investor profile %s: %s", self.profile_path, exc)
            return {}

    @staticmethod
    def _render_investor_context(profile: dict) -> str:
        """Render compact IPS summary for prompt injection."""
        if "_raw_yaml" in profile:
            raw = str(profile.get("_raw_yaml", ""))
            name = _extract_yaml_scalar(raw, "name") or "Investor"
            age = _extract_yaml_scalar(raw, "age") or "?"
            monthly = _extract_yaml_scalar(raw, "monthly_investable") or "?"
            horizon = _extract_yaml_scalar(raw, "investment_horizon_years") or "?"
            tolerance = _extract_yaml_scalar(raw, "tolerance") or "moderate"
            max_dd = _extract_yaml_scalar(raw, "max_drawdown_comfort") or "-20%"
            stock_cap = _extract_yaml_scalar(raw, "max_single_stock_pct") or "15"
            crypto_cap = _extract_yaml_scalar(raw, "max_single_crypto_pct") or "10"
            style = _extract_yaml_scalar(raw, "style") or "index-first"
            cadence = _extract_yaml_scalar(raw, "contribution_cadence") or "monthly"
            return (
                f"{name}, age {age}, experienced technologist and AI engineer. "
                f"Monthly satellite budget: ${monthly}; horizon: {horizon}; contribution cadence: {cadence}. "
                f"Risk profile: {tolerance} tolerance, comfortable with {max_dd} drawdowns. "
                f"Policy style: {style}; max single-stock {stock_cap}%, max single-crypto {crypto_cap}%."
            )

        investor = profile.get("investor", {}) if isinstance(profile.get("investor"), dict) else {}
        risk = profile.get("risk", {}) if isinstance(profile.get("risk"), dict) else {}
        policy = profile.get("policy", {}) if isinstance(profile.get("policy"), dict) else {}
        constraints = profile.get("constraints", []) if isinstance(profile.get("constraints"), list) else []
        goals = profile.get("goals", []) if isinstance(profile.get("goals"), list) else []

        name = investor.get("name", "Investor")
        age = investor.get("age", "?")
        monthly = investor.get("monthly_investable", "?")
        horizon = investor.get("investment_horizon_years", "?")
        tolerance = risk.get("tolerance", "moderate")
        max_dd = risk.get("max_drawdown_comfort", "-20%")
        stock_cap = policy.get("max_single_stock_pct", 15)
        crypto_cap = policy.get("max_single_crypto_pct", 10)
        style = policy.get("style", "index-first")
        cadence = policy.get("contribution_cadence", "monthly")
        vehicles = policy.get("preferred_vehicles", [])
        vehicles_text = ", ".join(str(v) for v in vehicles) if isinstance(vehicles, list) and vehicles else "ETFs + stocks"

        goal_lines: list[str] = []
        for goal in goals[:2]:
            if not isinstance(goal, dict):
                continue
            gid = str(goal.get("id", "goal"))
            gtype = str(goal.get("type", ""))
            horizon_text = str(goal.get("horizon", ""))
            goal_lines.append(f"{gid} ({gtype}, {horizon_text})".strip().rstrip(","))
        goals_text = "; ".join(goal_lines) if goal_lines else "Long-term wealth building"

        constraint_text = "; ".join(str(c) for c in constraints[:2]) if constraints else "No additional constraints noted."

        return (
            f"{name}, age {age}, experienced technologist and AI engineer. "
            f"Monthly satellite budget: ${monthly}; horizon: {horizon}; contribution cadence: {cadence}. "
            f"Risk profile: {tolerance} tolerance, comfortable with {max_dd} drawdowns. "
            f"Policy style: {style}; max single-stock {stock_cap}%, max single-crypto {crypto_cap}%. "
            f"Preferred vehicles: {vehicles_text}. Goals: {goals_text}. "
            f"Constraints: {constraint_text}"
        )

    def _render_recent_decisions_context(self, ticker: str, limit: int = 5) -> str:
        """Render recent ticker-specific recommendations for context injection."""
        try:
            recent = self.journal.get_recent_recommendations(ticker, limit=limit)
        except Exception as exc:
            logger.error("Failed to fetch recent recommendations for %s: %s", ticker, exc)
            recent = []

        if not recent:
            return "No prior recommendations logged for this ticker."

        lines = []
        for row in recent[:5]:
            ts = str(row.get("timestamp", ""))[:10]
            action = str(row.get("action", "WATCH")).upper()
            conf = row.get("confidence", "?")
            status = str(row.get("status", "open"))
            price = row.get("price_at_recommendation")
            price_text = f" @ ${float(price):.2f}" if isinstance(price, (int, float)) else ""
            rationale = str(row.get("rationale", "")).replace("\n", " ").strip()
            if len(rationale) > 180:
                rationale = rationale[:177] + "..."
            lines.append(f"- {ts}: {action}{price_text}, confidence {conf}/10, status {status}. {rationale}")

        return "\n".join(lines)

    def analyze_stock(
        self,
        stock_data: dict,
        macro_data: dict | None = None,
        market_overview: dict | None = None,
        crisis_context: dict | None = None,
        regime_context: str = "",
        sentinel_alerts: list | None = None,
        fear_greed: int = 50,
    ) -> Optional[CouncilDecision]:
        """Run full council analysis on a stock (v2 two-stage decision engine).

        Stage 1: Hard Risk Gate — binary safety checks
        Stage 2: CIO Scoring → Score-to-Action mapping

        Args:
            crisis_context: Optional dict from run_full_crisis_check().
            regime_context: Optional string from MROL regime packet.
            sentinel_alerts: Optional list of active sentinel alerts for risk gate.
            fear_greed: Current Fear & Greed index value (0-100) for regime adjustment.
        """
        ticker = stock_data.get("ticker", "UNKNOWN")
        logger.info(f"🏛️ Artha Council convening for {ticker} (v2)...")

        investor_profile = self._load_investor_profile()
        investor_context = self._render_investor_context(investor_profile)
        portfolio_data = self.portfolio_state.compute_state()
        portfolio_context = self.portfolio_state.render_prompt_summary(portfolio_data)
        recent_decisions_context = self._render_recent_decisions_context(ticker, limit=5)

        # --- Mechanical funnel signals (Wave 2): inject the scan_signals row so
        # analysts, buy scoring, and the CIO all see the same computed evidence.
        scan_signal_row: dict = {}
        try:
            if not isinstance(stock_data.get("scan_signals"), dict):
                from .buy_scoring import load_scan_signal_row
                loaded_row = load_scan_signal_row(ticker)
                if loaded_row:
                    stock_data["scan_signals"] = loaded_row
            scan_signal_row = stock_data.get("scan_signals") or {}
            if scan_signal_row:
                logger.info(
                    "  🧮 Scan signals for %s: track=%s momentum_z=%s 52wH=%s trend_template=%s",
                    ticker,
                    scan_signal_row.get("track"),
                    scan_signal_row.get("momentum_z"),
                    scan_signal_row.get("high_52w_prox"),
                    scan_signal_row.get("trend_template"),
                )
        except Exception as scan_err:
            logger.warning("  [council] Scan-signal injection failed for %s: %s", ticker, scan_err)

        # --- Mechanical regime gate (Wave 2, replaces fear/greed penalties) ---
        try:
            from .regime_gate import compute_regime
            regime_gate_state = compute_regime()
        except Exception as gate_err:
            logger.warning("  [council] Regime gate compute failed (%s) — trying cache", gate_err)
            try:
                from .regime_gate import load_cached_regime
                regime_gate_state = load_cached_regime() or {}
            except Exception:
                regime_gate_state = {}
        if not regime_gate_state.get("state"):
            regime_gate_state = {"state": "RISK_ON", "vol_multiplier": 1.0, "details": {"defaulted": True}}
        regime_gate_label = str(regime_gate_state.get("state") or "RISK_ON").upper()
        logger.info("  🚦 Regime gate: %s (vol_mult=%.2f)", regime_gate_label,
                    float(regime_gate_state.get("vol_multiplier", 1.0) or 1.0))

        try:
            valuation_expectations = build_valuation_expectations(stock_data)
            stock_data["valuation_expectations"] = valuation_expectations
        except Exception as val_err:
            logger.warning("  [council] Valuation/expectations engine failed for %s: %s", ticker, val_err)
            valuation_expectations = {}
            stock_data["valuation_expectations"] = {}

        try:
            portfolio_factor_risk = build_portfolio_factor_risk(
                ticker=ticker,
                stock_data=stock_data,
                portfolio_state=portfolio_data,
                config=Config,
            )
            stock_data["portfolio_factor_risk"] = portfolio_factor_risk
        except Exception as risk_err:
            logger.warning("  [council] Portfolio/factor risk engine failed for %s: %s", ticker, risk_err)
            portfolio_factor_risk = {}
            stock_data["portfolio_factor_risk"] = {}

        try:
            # Wave 2: pass stock_data so the meta-ranker can derive an
            # opportunity-score proxy and bucket the candidate properly
            # (the CIO score does not exist yet at this stage).
            calibration_meta_signal = build_meta_signal(self.journal, stock_data=stock_data)
            stock_data["calibration_meta_signal"] = calibration_meta_signal
        except Exception as meta_err:
            logger.warning("  [council] Calibration meta-signal failed for %s: %s", ticker, meta_err)
            calibration_meta_signal = {}
            stock_data["calibration_meta_signal"] = {}

        context_header = build_context_header(
            ticker=ticker,
            investor_context=investor_context,
            portfolio_context=portfolio_context,
            recent_decisions=recent_decisions_context,
        )
        # --- Crisis-grade quality/value-trap pre-checks (advisory, Wave 2) ---
        # QualityFilter and ValueTrapDetector are pure functions on stock_data.
        # They are ADVISORY outside crisis mode (QualityFilter's $10B cap gate
        # is a crisis bar, not a normal-times entry requirement) — the hard
        # risk gate remains the only deterministic blocker here.
        crisis_precheck_lines: list[str] = []
        try:
            from .crisis import QualityFilter, ValueTrapDetector
            quality_check = QualityFilter().check(stock_data)
            value_trap_check = ValueTrapDetector().check(stock_data)
            stock_data["crisis_quality_filter"] = quality_check
            stock_data["value_trap"] = value_trap_check
            crisis_precheck_lines = [
                f"Crisis-grade quality filter (advisory): {quality_check.get('summary', 'n/a')}",
                f"Value trap detector (advisory): {value_trap_check.get('summary', 'n/a')}",
            ]
        except Exception as precheck_err:
            logger.warning("  [council] Crisis quality/value-trap pre-checks failed (non-fatal): %s", precheck_err)

        deterministic_checks = (
            f"{format_valuation_expectations(valuation_expectations)}\n\n"
            f"{format_portfolio_factor_risk(portfolio_factor_risk)}\n\n"
            f"{format_meta_signal(calibration_meta_signal)}"
        )
        if crisis_precheck_lines:
            deterministic_checks += "\n\n" + "\n".join(crisis_precheck_lines)
        context_header = (
            f"{context_header}\n\n--- DETERMINISTIC DECISION CHECKS ---\n"
            f"{deterministic_checks}\n"
            f"--- END DETERMINISTIC DECISION CHECKS ---\n"
        )

        # --- Lessons from graded history (Wave 2): every analyst and the CIO
        # sees the digest of its own graded mistakes before opining again.
        lessons_digest = ""
        try:
            from .self_review import get_council_lessons
            lessons_digest = get_council_lessons() or ""
        except Exception as lessons_err:
            logger.warning("  [council] Council lessons digest failed (non-fatal): %s", lessons_err)
        if lessons_digest:
            context_header = (
                f"{context_header}\n\n--- LESSONS FROM YOUR OWN GRADED HISTORY ---\n"
                f"{lessons_digest}\n"
                f"--- END LESSONS ---\n"
            )

        # Regime gate context for all analysts (mechanical, not sentiment).
        context_header = (
            f"{context_header}\n\n--- MECHANICAL REGIME GATE ---\n"
            f"State: {regime_gate_label} | vol_multiplier: "
            f"{float(regime_gate_state.get('vol_multiplier', 1.0) or 1.0):.2f}\n"
            f"RISK_ON: full buys. RISK_OFF: new momentum/breakout entries need conviction >= "
            f"{Config.REGIME_GATE_RISK_OFF_MIN_CONVICTION} and half size. HARD_RISK_OFF: no new buys. "
            f"Exits are NEVER blocked by this gate.\n"
            f"--- END MECHANICAL REGIME GATE ---\n"
        )

        if regime_context:
            context_header = f"{context_header}\n\n--- MACRO REGIME CONTEXT ---\n{regime_context}\n--- END REGIME CONTEXT ---\n"

        # Build crisis injection if in crisis mode
        crisis_prefix = ""
        crisis_contrarian_prefix = ""
        if crisis_context:
            fp = crisis_context.get("fingerprint") or {}
            state = str(crisis_context.get("crisis_state", "normal"))
            spy_dd = crisis_context.get("state_summary", {}).get("last_spy_drawdown") or 0.0
            vix = crisis_context.get("state_summary", {}).get("last_vix") or 20.0
            fg = crisis_context.get("state_summary", {}).get("last_fg") or 50
            dominant = fp.get("dominant", "GEOPOLITICAL")
            dom_prob = fp.get("dominant_prob", 0.3)
            qf_summary = (crisis_context.get("quality_filter") or {}).get("summary", "")
            vt_summary = (crisis_context.get("value_trap") or {}).get("summary", "")

            crisis_prefix = build_crisis_context(
                state=state, drawdown=float(spy_dd), vix=float(vix), fg=int(fg),
                dominant_type=str(dominant), dominant_prob=float(dom_prob), for_analyst="all",
            )
            crisis_contrarian_prefix = build_crisis_context(
                state=state, drawdown=float(spy_dd), vix=float(vix), fg=int(fg),
                dominant_type=str(dominant), dominant_prob=float(dom_prob), for_analyst="contrarian",
                quality_summary=qf_summary, value_trap_summary=vt_summary,
            )
            logger.info(f"  ⚠️  Crisis context: {state} | {dominant} ({dom_prob:.0%})")

        # --- Data quality gate ---
        data_quality = validate_stock_data(stock_data)
        stock_data["data_quality_report"] = asdict(data_quality)
        dq_lines = [
            f"Completeness: {data_quality.completeness_score:.1f}%",
            f"Context coverage: {data_quality.context_coverage_score:.1f}%",
            f"Sources: {', '.join(data_quality.sources_used) or 'none'}",
        ]
        if data_quality.missing_fields:
            dq_lines.append(f"Missing: {', '.join(data_quality.missing_fields)}")
        if data_quality.enrichment_missing_fields:
            dq_lines.append(f"Context gaps: {', '.join(data_quality.enrichment_missing_fields[:8])}")
        if data_quality.staleness_warnings:
            dq_lines.append(f"Staleness: {'; '.join(data_quality.staleness_warnings[:3])}")
        if data_quality.source_conflicts:
            dq_lines.append(f"Source conflicts: {'; '.join(data_quality.source_conflicts[:3])}")
        if data_quality.anomaly_flags:
            dq_lines.append(f"Anomalies: {'; '.join(data_quality.anomaly_flags[:3])}")
        context_header = (
            f"{context_header}\n\n--- DATA QUALITY REPORT ---\n"
            + "\n".join(dq_lines)
            + "\n--- END DATA QUALITY REPORT ---\n"
        )
        if not data_quality.passed_hard_checks:
            reason = "; ".join(data_quality.hard_check_failures)
            logger.warning("  ❌ Data quality hard gate FAILED for %s: %s", ticker, reason)
            dq_report = f"DATA QUALITY HARD GATE BLOCKED: {reason}\n\nAnalysis was not completed."
            hold_report = AnalystReport(
                analyst_name="Data Quality Gate",
                model="system",
                verdict="HOLD",
                confidence=1,
                report=dq_report,
            )
            return CouncilDecision(
                ticker=ticker,
                final_verdict="AVOID",
                consensus="N/A",
                recommended_action=f"Do not invest — data quality failed: {reason}",
                allocation="$0",
                synthesis_report=dq_report,
                fundamental=hold_report,
                technical=hold_report,
                contrarian=hold_report,
                opportunity_score=0,
                adjusted_score=0,
                score_components={"data_quality": 0},
                confidence=1,
                hard_risk_gate_passed=False,
                hard_risk_gate_reason=f"data quality failed: {reason}",
                data_quality_report=asdict(data_quality),
            )

        # --- Run Research Desk intelligence gathering ---
        logger.info("  🔍 Running Research Desk intelligence gathering...")
        intelligence_brief = self.research_desk.research_stock(ticker, stock_data, macro_data or {})
        research_insufficient = "RESEARCH STATUS: INSUFFICIENT_CURRENT_WEB_DATA" in intelligence_brief

        # --- Pre-brief: recent events for this ticker ---
        pre_brief_text = ""
        try:
            from .pre_brief import PreBrief
            pre_brief_text = PreBrief().get_brief(ticker)
        except Exception as pb_e:
            logger.warning("  [council] Pre-brief failed (non-fatal): %s", pb_e)

        # --- Momentum acceleration context ---
        momentum_context = ""
        try:
            from .momentum_tracker import MomentumTracker
            mom_delta = MomentumTracker().get_momentum_delta(ticker)
            if mom_delta and mom_delta.get("trend") != "new":
                trend = mom_delta["trend"].upper()
                prev = mom_delta.get("previous_score", "?")
                curr = mom_delta.get("current_score", "?")
                delta = mom_delta.get("delta", 0)
                momentum_context = (
                    f"MOMENTUM TREND: {trend} "
                    f"(score: {prev} → {curr}, delta: {delta:+.1f})"
                )
            elif mom_delta and mom_delta.get("trend") == "new":
                momentum_context = f"MOMENTUM TREND: NEW (first appearance, score={mom_delta.get('current_score', '?')})"
        except Exception as mt_e:
            logger.warning("  [council] Momentum context failed (non-fatal): %s", mt_e)

        # --- Bounded agentic diligence layer ---
        # Each analyst gets a role-specific investigation trace instead of only
        # reading the same generic packet. Failures are non-fatal because the
        # deterministic data-quality gate has already run.
        agentic_trace: dict = {}
        agentic_briefs: dict = {}
        agentic_cio_brief = ""
        try:
            logger.info("  🧭 Running bounded agentic diligence for %s...", ticker)
            agentic_result = build_agentic_diligence(
                ticker=ticker,
                stock_data=stock_data,
                macro_data=macro_data or {},
                market_overview=market_overview or {},
                intelligence_brief=intelligence_brief,
                data_quality_report=asdict(data_quality),
            )
            agentic_trace = agentic_result.to_dict()
            stock_data["agentic_diligence"] = agentic_trace
            agentic_briefs = agentic_result.analyst_briefs or {}
            agentic_cio_brief = agentic_result.cio_brief or ""
            logger.info(
                "  ✅ Agentic diligence ready for %s: evidence=%d trace=%s",
                ticker,
                len(agentic_result.evidence or []),
                agentic_result.trace_path or "not-written",
            )
        except Exception as agentic_err:
            logger.exception("  ⚠️ Agentic diligence failed for %s (falling back to normal council): %s", ticker, agentic_err)
            agentic_trace = {
                "enabled": False,
                "ticker": ticker,
                "error": str(agentic_err),
            }

        def _brief_with_agentic(role: str) -> str:
            role_brief = agentic_briefs.get(role)
            if not role_brief:
                return intelligence_brief
            return (
                f"{intelligence_brief}\n\n"
                f"--- ROLE-SPECIFIC AGENTIC DILIGENCE ---\n"
                f"{role_brief}\n"
                f"--- END ROLE-SPECIFIC AGENTIC DILIGENCE ---"
            )

        # --- Run all 3 analysts IN PARALLEL ---
        from concurrent.futures import ThreadPoolExecutor

        from .analysts import get_contrarian_model_label
        contrarian_model = get_contrarian_model_label()
        logger.info(
            "  🚀 Running 3 agentic analysts in parallel (%s + %s + %s)...",
            Config.GPT_MODEL, Config.GEMINI_TECHNICAL_MODEL, contrarian_model,
        )

        fund_header = crisis_prefix + context_header if crisis_prefix else context_header
        cont_header = crisis_contrarian_prefix + context_header if crisis_contrarian_prefix else context_header

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_fund = executor.submit(
                run_fundamental_analyst, stock_data, macro_data,
                context_header=fund_header, intelligence_brief=_brief_with_agentic("fundamental"),
                pre_brief=pre_brief_text, momentum_context=momentum_context,
            )
            future_tech = executor.submit(
                run_technical_analyst, stock_data, market_overview, macro_data,
                context_header=fund_header, intelligence_brief=_brief_with_agentic("technical"),
                pre_brief=pre_brief_text, momentum_context=momentum_context,
            )
            future_cont = executor.submit(
                run_contrarian_analyst, stock_data, macro_data,
                context_header=cont_header, intelligence_brief=_brief_with_agentic("contrarian"),
                pre_brief=pre_brief_text, momentum_context=momentum_context,
            )

        fundamental_raw = future_fund.result(timeout=120)
        technical_raw = future_tech.result(timeout=120)
        contrarian_raw = future_cont.result(timeout=120)

        if not fundamental_raw:
            logger.error(f"  ❌ Fundamental analyst failed for {ticker}")
            return None
        if not technical_raw:
            logger.error(f"  ❌ Technical analyst failed for {ticker}")
            return None
        if not contrarian_raw:
            logger.error(f"  ❌ Contrarian analyst failed for {ticker}")
            return None

        # --- Parse individual reports ---
        fundamental = AnalystReport.parse("Fundamental", Config.GPT_MODEL, fundamental_raw)
        technical = AnalystReport.parse("Technical + Sentiment", Config.GEMINI_TECHNICAL_MODEL, technical_raw)
        contrarian = AnalystReport.parse("Contrarian / Risk", contrarian_model, contrarian_raw)

        logger.info(
            f"  📋 Verdicts: Fundamental={fundamental.verdict}({fundamental.confidence}), "
            f"Technical={technical.verdict}({technical.confidence}), "
            f"Contrarian={contrarian.verdict}({contrarian.confidence})"
        )

        # --- Stage 1: Hard Risk Gate ---
        logger.info(f"  🔒 Running Hard Risk Gate for {ticker}...")
        gate_passed, gate_reason = hard_risk_gate(
            ticker=ticker,
            stock_data=stock_data,
            portfolio_state=portfolio_data,
            sentinel_alerts=sentinel_alerts or [],
            config=Config,
        )
        if not gate_passed:
            logger.warning(f"  ❌ Hard Risk Gate FAILED for {ticker}: {gate_reason}")
            # Return a minimal AVOID decision
            return CouncilDecision(
                ticker=ticker,
                final_verdict="AVOID",
                consensus="N/A",
                recommended_action=f"Do not invest — {gate_reason}",
                allocation="$0",
                synthesis_report=f"HARD RISK GATE BLOCKED: {gate_reason}\n\nAnalysis was not completed.",
                fundamental=fundamental,
                technical=technical,
                contrarian=contrarian,
                opportunity_score=0,
                adjusted_score=0,
                hard_risk_gate_passed=False,
                hard_risk_gate_reason=gate_reason,
                data_quality_report=asdict(data_quality),
                agentic_trace=agentic_trace,
            )
        logger.info(f"  ✅ Hard Risk Gate PASSED for {ticker}")

        # --- Compute Council Convergence Score ---
        ccs_result = compute_convergence_score(fundamental, technical, contrarian)
        logger.info(
            f"  📊 CCS: {ccs_result['ccs']}/12 ({ccs_result['tier']}) "
            f"| Drivers: {ccs_result['drivers']} | Orthogonality: {ccs_result['orthogonality']}"
        )

        # --- Deterministic buy score audit before CIO synthesis ---
        # The CIO receives this as the scoring baseline and may only request a
        # bounded adjustment. The final action later uses the audited score, not
        # the CIO's raw opportunity score by itself.
        buy_score_audit = build_buy_score_audit(
            stock_data=stock_data,
            data_quality_report=asdict(data_quality),
            valuation_expectations=valuation_expectations,
            portfolio_factor_risk=portfolio_factor_risk,
            analysts=[fundamental, technical, contrarian],
            research_insufficient=research_insufficient,
            fear_greed=fear_greed,
        )
        logger.info(
            "  🧾 Buy score audit for %s: base=%s rules=%+d pre_cio=%s",
            ticker,
            buy_score_audit.get("base_score"),
            int(buy_score_audit.get("rule_adjustment_total") or 0),
            buy_score_audit.get("pre_cio_score"),
        )

        # --- Apply trust gates (crisis mode only) ---
        gate_result: dict = {"gate_applied": False, "vetoed": False}
        if crisis_context:
            fp = crisis_context.get("fingerprint") or {}
            dominant = str(fp.get("dominant", "GEOPOLITICAL")).replace("CrisisType.", "")
            profile = stock_data.get("profile") or {}
            sector = profile.get("sector") or (stock_data.get("yf_quote") or {}).get("sector") or ""
            gate_result = apply_trust_gates(
                ticker=ticker, sector=sector, dominant_crisis_type=dominant,
                fundamental=fundamental, technical=technical, contrarian=contrarian,
            )
            if gate_result.get("gate_applied"):
                logger.info(f"  🔒 {gate_result['details']}")

        # --- Compute deployment context ---
        deployment = get_deployment_target(fear_greed, portfolio_data, Config)
        deployment_context_str = (
            f"Current cash: ${deployment['cash']:,.2f}\n"
            f"Total NAV: ${deployment['total_nav']:,.2f}\n"
            f"Currently invested: {deployment['current_invested_pct']:.0%}\n"
            f"Regime: {deployment['regime_label']} (Fear & Greed: {fear_greed})\n"
            f"Regime deployment target: {deployment['target_invested_pct']:.0%}\n"
            f"Deployable amount: ${deployment['deployable_amount']:,.2f}\n"
            f"Current positions: {deployment['position_count']}/{deployment['max_positions']}\n"
            f"Available deployment slots: {deployment['available_slots']}"
        )

        # Build synthesis crisis context
        synthesis_crisis_prefix = ""
        if crisis_context:
            fp = crisis_context.get("fingerprint") or {}
            state = str(crisis_context.get("crisis_state", "normal"))
            spy_dd = crisis_context.get("state_summary", {}).get("last_spy_drawdown") or 0.0
            vix = crisis_context.get("state_summary", {}).get("last_vix") or 20.0
            fg = crisis_context.get("state_summary", {}).get("last_fg") or 50
            dominant = fp.get("dominant", "GEOPOLITICAL")
            dom_prob = fp.get("dominant_prob", 0.3)
            qf_summary = (crisis_context.get("quality_filter") or {}).get("summary", "")
            vt_summary = (crisis_context.get("value_trap") or {}).get("summary", "")
            val_summary = (crisis_context.get("valuation_discount") or {}).get("reason", "")
            trust_note = gate_result.get("details", "")
            synthesis_crisis_prefix = build_crisis_context(
                state=state, drawdown=float(spy_dd), vix=float(vix), fg=int(fg),
                dominant_type=str(dominant), dominant_prob=float(dom_prob), for_analyst="synthesis",
                quality_summary=qf_summary, value_trap_summary=vt_summary,
                valuation_discount_summary=val_summary,
                ccs=ccs_result["ccs"], ccs_tier=ccs_result["tier"],
                orthogonality=ccs_result["orthogonality"], trust_gate_note=trust_note,
            )

        # --- Stage 2: CIO Synthesis with scoring ---
        logger.info("  🤝 Running Synthesis (CIO — %s with scoring)...", Config.GPT_MODEL)
        synthesis = self._synthesize(
            ticker, stock_data, fundamental, technical, contrarian,
            crisis_prefix=synthesis_crisis_prefix,
            deployment_context=deployment_context_str,
            agentic_cio_brief=agentic_cio_brief,
            deterministic_score_audit=render_buy_score_audit(buy_score_audit, include_details=True),
            lessons=lessons_digest,
            regime_gate_state=regime_gate_state,
        )
        if not synthesis:
            logger.error(f"  ❌ Synthesis failed for {ticker}")
            return None

        # --- Extract scoring JSON from synthesis (isolated — daemon must not crash) ---
        scoring = {}
        scoring_invalid = False
        try:
            scoring = _extract_scoring_json(synthesis) or {}
            if scoring:
                synthesis = _replace_last_scoring_json(synthesis, scoring)
                if scoring.get("_schema_repaired"):
                    synthesis += "\n\nSCORING QUALITY NOTE: CIO scoring JSON had minor numeric schema errors and was repaired conservatively before action mapping."
                logger.info(
                    f"  📊 Opportunity Score: {scoring.get('opportunity_score', '?')}/100 "
                    f"— Verdict: {scoring.get('verdict', '?')} "
                    f"— Confidence: {scoring.get('confidence', '?')}/10"
                )
            else:
                logger.warning(f"  ⚠️ No valid scoring JSON found in synthesis for {ticker} — using defaults")
                scoring_invalid = True
        except Exception as exc:
            logger.warning(f"  ⚠️ Scoring JSON extraction failed for {ticker} (non-fatal): {exc}")
            scoring = {}
            scoring_invalid = True

        thesis_type = str(scoring.get("thesis_type", ""))
        alloc_pct_from_cio = float(scoring.get("recommended_allocation_pct", 0) or 0)
        entry_valid_until = str(scoring.get("entry_valid_until", ""))
        invalidation_conditions = scoring.get("invalidation_conditions", [])
        stop_loss_pct = float(scoring.get("stop_loss_pct", -0.08) or -0.08)
        target_pct = float(scoring.get("target_pct", 0.15) or 0.15)
        cio_verdict = str(scoring.get("verdict", "AVOID" if scoring_invalid else "")).upper()
        cio_confidence = int(scoring.get("confidence", 3 if scoring_invalid else 5) or (3 if scoring_invalid else 5))

        # --- Wave 2: machine-readable CIO decision block (preferred over the
        # legacy fixed -8%/+15% and prose scraping; regex paths stay as fallback).
        try:
            current_price_val = float(str(
                (stock_data.get("quote") or {}).get("price")
                or (stock_data.get("yf_quote") or {}).get("price") or 0
            ).replace(",", ""))
        except (TypeError, ValueError):
            current_price_val = 0.0
        atr14 = _compute_atr14(stock_data.get("price_history"))
        atr_stop_pct = _atr_initial_stop_pct(current_price_val, atr14, Config)

        cio_decision: dict = {}
        try:
            cio_decision = _extract_decision_json(synthesis) or {}
        except Exception as decision_err:
            logger.warning("  ⚠️ Decision JSON extraction failed for %s (non-fatal): %s", ticker, decision_err)
        decision_half_size = bool(cio_decision.get("half_size"))
        cio_conviction = cio_confidence
        if cio_decision:
            logger.info(
                "  🧩 CIO decision block: verdict=%s conviction=%s stop=%s target=%s half_size=%s conditions=%d",
                cio_decision.get("verdict"),
                cio_decision.get("conviction_0_10"),
                cio_decision.get("stop_price"),
                cio_decision.get("target_price"),
                decision_half_size,
                len(cio_decision.get("invalidation_conditions") or []),
            )
            if cio_decision.get("conviction_0_10") is not None:
                cio_conviction = int(cio_decision["conviction_0_10"])
            if not cio_verdict and cio_decision.get("verdict"):
                cio_verdict = str(cio_decision["verdict"]).upper()
            if cio_decision.get("invalidation_conditions"):
                # Structured {metric, op, value, description} conditions —
                # Wave 1 sell engine consumes these directly.
                invalidation_conditions = cio_decision["invalidation_conditions"]
            entry_zone = cio_decision.get("entry_zone") or {}
            if entry_zone.get("valid_until_iso"):
                entry_valid_until = str(entry_zone["valid_until_iso"])

        # Stop: prefer the CIO's ATR-derived stop_price when its distance is
        # sane (below entry, within the 10% cap + slack); else derive locally
        # from 2.5x ATR14 capped at 10%; else keep the CIO scoring value.
        stop_cap = float(getattr(Config, "CIO_STOP_CAP_PCT", 0.10))
        cio_stop_price = cio_decision.get("stop_price") if cio_decision else None
        if current_price_val > 0 and cio_stop_price:
            stop_frac = (float(cio_stop_price) - current_price_val) / current_price_val
            if -(stop_cap * 1.05) <= stop_frac < 0:
                stop_loss_pct = round(stop_frac, 4)
            elif atr_stop_pct is not None:
                logger.info(
                    "  🧮 CIO stop_price %.2f out of bounds (%.1f%%) — using ATR-derived stop %.1f%%",
                    float(cio_stop_price), stop_frac * 100, atr_stop_pct * 100,
                )
                stop_loss_pct = atr_stop_pct
        elif atr_stop_pct is not None:
            stop_loss_pct = atr_stop_pct
        cio_target_price = cio_decision.get("target_price") if cio_decision else None
        if current_price_val > 0 and cio_target_price and float(cio_target_price) > current_price_val:
            target_pct = round(float(cio_target_price) / current_price_val - 1.0, 4)

        scoring_audit = apply_cio_buy_adjustment(
            scoring,
            buy_score_audit,
            config=Config,
            cio_confidence=cio_confidence,
        )
        if scoring_invalid:
            scoring_audit = {
                **scoring_audit,
                "cio_adjustment": 0,
                "cio_adjustment_status": "rejected",
                "cio_adjustment_rejected_reason": "invalid or missing CIO scoring JSON",
                "final_score": 0,
            }
        raw_score = int(scoring_audit.get("final_score", 0 if scoring_invalid else 50) or 0)
        score_components = scoring_audit.get("components", {}) or scoring.get("components", {})
        if scoring:
            scoring = {
                **scoring,
                "components": score_components,
                "deterministic_base_score": int(scoring_audit.get("base_score") or 0),
                "rule_adjustment_total": int(scoring_audit.get("rule_adjustment_total") or 0),
                "deterministic_score_before_cio": int(scoring_audit.get("pre_cio_score") or 0),
                "cio_score_adjustment": int(scoring_audit.get("cio_adjustment") or 0),
                "cio_adjustment_raw": int(scoring_audit.get("cio_adjustment_raw") or 0),
                "cio_adjustment_category": scoring_audit.get("cio_adjustment_category") or "none",
                "cio_adjustment_status": scoring_audit.get("cio_adjustment_status") or "none",
                "cio_adjustment_evidence": scoring_audit.get("cio_adjustment_evidence") or [],
                "cio_adjustment_reason": scoring_audit.get("cio_adjustment_reason") or "",
                "cio_adjustment_rejected_reason": scoring_audit.get("cio_adjustment_rejected_reason") or "",
                "opportunity_score": raw_score,
            }
            synthesis = _replace_last_scoring_json(synthesis, scoring)
        audit_status = scoring_audit.get("cio_adjustment_status") or "none"
        logger.info(
            "  🧾 Audited Opportunity Score: base=%s rules=%+d CIO=%+d final=%s status=%s",
            scoring_audit.get("base_score"),
            int(scoring_audit.get("rule_adjustment_total") or 0),
            int(scoring_audit.get("cio_adjustment") or 0),
            raw_score,
            audit_status,
        )

        # --- Apply score-to-action mapping (regime gate, no F&G penalty) ---
        action_result = score_to_action(
            raw_score, fear_greed, portfolio_data, Config,
            regime_gate_state=regime_gate_state,
        )
        adjusted_score = action_result["adjusted_score"]
        mapped_action = action_result["action"]
        mapped_alloc_pct = action_result["recommended_allocation_pct"]
        gate_vol_multiplier = float(action_result.get("vol_multiplier", 1.0) or 1.0)

        analyst_buy_count = sum(
            1 for r in [fundamental, technical, contrarian]
            if str(getattr(r, "verdict", "") or "").upper() == "BUY"
        )

        _BUY_SIDE_ACTIONS = ("BUY", "STARTER", "TACTICAL_BUY")

        # Default rule: CIO can restrict, never upgrade above the score-mapped
        # ceiling — EXCEPT via the Wave 2 mechanical-override path: when the
        # funnel composite is STRONG (momentum/breakout track, z >= 1.5, trend
        # template pass) and no analyst produced a SELL-grade falsifiable risk,
        # the CIO may go ONE notch above the mapped action; and a CIO downgrade
        # to WATCH/DEFER against a buy-side map with >= 2 analyst BUYs keeps a
        # tactical path (this replaces the old fear/greed-tied anti-watch-bias
        # override — same intent, now mechanically qualified).
        mechanical_strong = _mechanical_composite_strong(scan_signal_row, Config)
        gate_allows_buys = regime_gate_label != "HARD_RISK_OFF"
        mechanical_override_reason = ""

        if (
            gate_allows_buys
            and mechanical_strong
            and not research_insufficient
            and cio_verdict in _VALID_FINAL_VERDICTS
            and _BUY_NOTCH_UPGRADE.get(mapped_action) == cio_verdict
            and contrarian.verdict != "SELL"
        ):
            final_action = cio_verdict
            final_alloc_pct = (
                alloc_pct_from_cio * gate_vol_multiplier
                if alloc_pct_from_cio > 0
                else max(3.0, float(mapped_alloc_pct or 0))
            )
            mechanical_override_reason = (
                f"mechanical-override: CIO upgraded {mapped_action} -> {cio_verdict} on strong "
                f"composite (track={scan_signal_row.get('track')}, momentum_z={scan_signal_row.get('momentum_z')}, "
                f"trend_template={scan_signal_row.get('trend_template')})"
            )
            logger.info("  🪜 %s for %s", mechanical_override_reason, ticker)
        elif (
            gate_allows_buys
            and mechanical_strong
            and mapped_action in _BUY_SIDE_ACTIONS
            and cio_verdict in ("WATCH", "DEFER")
            and adjusted_score >= Config.SCORE_THRESHOLD_TACTICAL
            and analyst_buy_count >= 2
            and not research_insufficient
        ):
            final_action = "TACTICAL_BUY"
            final_alloc_pct = min(
                max(3.0, float(mapped_alloc_pct or 0)),
                Config.EXPLORATION_MAX_PER_POSITION_PCT * 100,
            )
            mechanical_override_reason = (
                f"mechanical-override: anti-watch-bias — mapped={mapped_action} CIO={cio_verdict} "
                f"adjusted_score={adjusted_score} analysts_buy={analyst_buy_count} on strong composite"
            )
            logger.info("  🪜 %s => TACTICAL_BUY %.1f%% for %s", mechanical_override_reason, final_alloc_pct, ticker)
        else:
            if cio_verdict in _VALID_FINAL_VERDICTS:
                final_action = _min_risk_action(mapped_action, cio_verdict)
            else:
                final_action = mapped_action

            # Allocation: use CIO-specified if plausible (vol-scaled), else mapped
            final_alloc_pct = (
                alloc_pct_from_cio * gate_vol_multiplier
                if alloc_pct_from_cio > 0
                else mapped_alloc_pct
            )

        # --- Regime gate enforcement on the FINAL action (all paths) ---
        # HARD_RISK_OFF: no new buys, ever (exits are handled by the sell
        # engine and are never gated). RISK_OFF: momentum/breakout entries
        # need conviction >= 8; all new buys run half size.
        half_size_final = bool(action_result.get("half_size")) or decision_half_size
        _NEW_CAPITAL_ACTIONS = ("BUY", "STARTER", "TACTICAL_BUY", "ACCUMULATE", "ADD")
        if final_action in _NEW_CAPITAL_ACTIONS and regime_gate_label == "HARD_RISK_OFF":
            logger.info("  🚦 HARD_RISK_OFF — forcing %s → WATCH for %s (no new buys)", final_action, ticker)
            final_action = "WATCH"
            final_alloc_pct = 0.0
            synthesis += (
                "\n\nREGIME GATE NOTE: HARD_RISK_OFF (SPY below 200d SMA with negative 12-mo "
                "excess return vs T-bills) — new buys are mechanically blocked; verdict capped at WATCH."
            )
        elif final_action in _BUY_SIDE_ACTIONS and regime_gate_label == "RISK_OFF":
            signal_track = str(scan_signal_row.get("track") or "").lower()
            min_conviction = int(Config.REGIME_GATE_RISK_OFF_MIN_CONVICTION)
            if signal_track in ("momentum", "breakout") and cio_conviction < min_conviction:
                logger.info(
                    "  🚦 RISK_OFF — %s track needs conviction >= %d (got %d): %s → WATCH for %s",
                    signal_track, min_conviction, cio_conviction, final_action, ticker,
                )
                final_action = "WATCH"
                final_alloc_pct = 0.0
                synthesis += (
                    f"\n\nREGIME GATE NOTE: RISK_OFF — new {signal_track} entries require council "
                    f"conviction >= {min_conviction} (got {cio_conviction}); verdict capped at WATCH."
                )
            else:
                half_size_final = True
        if half_size_final and final_action in _BUY_SIDE_ACTIONS:
            final_alloc_pct = round(final_alloc_pct * 0.5, 2)

        # --- Clamp allocation against hard constraints, but preserve exploration path ---
        _total_nav_for_clamp = max(deployment.get("total_nav", 0) or 0, 1)
        deployable_pct = deployment.get("deployable_amount", 0) / _total_nav_for_clamp * 100
        budget_cap_pct = deployment.get("budget_cap_amount", 0) / _total_nav_for_clamp * 100
        exploration_pct = 0.0
        has_cash = float(deployment.get("cash", 0) or 0) > 0
        has_slots = int(deployment.get("available_slots", 0) or 0) > 0
        if has_cash and has_slots and final_action in ("BUY", "STARTER", "TACTICAL_BUY"):
            # Keep looking for the ladder in chaos. If regime math says
            # deployable=0 but cash still exists, preserve a small exploration/tactical path.
            exploration_pct = min(
                Config.EXPLORATION_MAX_PER_POSITION_PCT * 100,
                float(deployment.get("cash", 0) or 0) / _total_nav_for_clamp * 100,
            )
        alloc_cap_pct = min(
            Config.MAX_POSITION_PCT * 100,
            max(deployable_pct, exploration_pct),
        )
        if budget_cap_pct > 0:
            alloc_cap_pct = min(alloc_cap_pct, budget_cap_pct)
        final_alloc_pct = min(final_alloc_pct, alloc_cap_pct)
        final_alloc_pct = max(final_alloc_pct, 0.0)
        # If no deployment slots available, force WATCH for any buy-side action
        if deployment.get("available_slots", 1) <= 0 and final_action in (
            "BUY", "STARTER", "TACTICAL_BUY"
        ):
            logger.info(f"  ⚠️ No deployment slots available — forcing {final_action} → WATCH for {ticker}")
            final_action = "WATCH"
            final_alloc_pct = 0.0
        elif deployable_pct <= 0 < exploration_pct and final_action in ("BUY", "STARTER", "TACTICAL_BUY"):
            logger.info(
                "  🪜 Exploration path active for %s — deployable=0 by regime, allowing %.1f%% tactical allocation",
                ticker,
                final_alloc_pct,
            )

        held_tickers_for_research_gate = {
            str(p.get("ticker", "")).upper()
            for p in (portfolio_data.get("positions") or [])
            if isinstance(p, dict)
        }
        if (
            research_insufficient
            and ticker.upper() not in held_tickers_for_research_gate
            and final_action in ("BUY", "STARTER", "TACTICAL_BUY", "ACCUMULATE", "ADD")
        ):
            logger.info(
                "  📋 Current-web research insufficient — forcing %s → DEFER for new position %s",
                final_action,
                ticker,
            )
            final_action = "DEFER"
            final_alloc_pct = 0.0
            synthesis += (
                "\n\nDATA QUALITY OVERRIDE: Current-web/news enrichment returned no usable sources. "
                "Opening a new buy-side position is deferred until current source-backed evidence is available."
            )

        # --- Shadow trade logging (fully isolated — must never interrupt main flow) ---
        try:
            if final_action in ("WATCH", "DEFER") and analyst_buy_count > 0:
                current_price = (
                    (stock_data.get("quote") or {}).get("price") or
                    (stock_data.get("yf_quote") or {}).get("price") or 0
                )
                stop_price = float(current_price) * (1 + stop_loss_pct) if current_price else 0
                self.journal.log_shadow_trade(
                    ticker=ticker,
                    thesis_type=thesis_type or "unknown",
                    blocked_by=f"score_to_action ({adjusted_score}/100)",
                    blocked_reason=f"Score {adjusted_score} below BUY threshold ({Config.SCORE_THRESHOLD_BUY}); mapped to {final_action}",
                    hypothetical_entry=float(current_price),
                    hypothetical_stop=stop_price,
                    opportunity_score=raw_score,
                    regime=action_result["regime_label"],
                    fear_greed=fear_greed,
                    sector=str(portfolio_factor_risk.get("sector") or ""),
                    benchmark_ticker=str(portfolio_factor_risk.get("market_benchmark_ticker") or ""),
                    sector_benchmark_ticker=str(portfolio_factor_risk.get("sector_benchmark_ticker") or ""),
                )
                logger.info(f"  📝 Shadow trade logged for {ticker} (analysts said BUY x{analyst_buy_count}, council said {final_action})")
        except Exception as exc:
            logger.warning(f"  ⚠️ Shadow trade logging failed for {ticker} (non-fatal): {exc}")

        # --- Parse synthesis for narrative fields ---
        parsed = _parse_synthesis(synthesis)

        # Override final_verdict from scoring if available
        if final_action in _VALID_FINAL_VERDICTS:
            parsed["final_verdict"] = final_action

        # --- Portfolio-aware verdict translation ---
        # SELL/TRIM for stocks we don't own → AVOID (can't sell what you don't have)
        # This makes recommendations actionable rather than generic analyst ratings.
        held_tickers = {
            str(p.get("ticker", "")).upper()
            for p in (portfolio_data.get("positions") or [])
            if isinstance(p, dict)
        }
        _SELL_SIDE = {"SELL", "TRIM", "STRONG SELL"}
        if parsed["final_verdict"] in _SELL_SIDE and ticker.upper() not in held_tickers:
            original_verdict = parsed["final_verdict"]
            parsed["final_verdict"] = "AVOID"
            logger.info(
                "  🏷️ Verdict translated: %s → AVOID for %s (not in portfolio)",
                original_verdict, ticker,
            )

        normalized_no_buy = _normalize_no_buy_action(
            final_action=parsed["final_verdict"],
            cio_verdict=cio_verdict,
            recommended_action=parsed.get("recommended_action", ""),
            score_components=score_components,
            scoring_invalid=scoring_invalid,
        )
        if normalized_no_buy != parsed["final_verdict"]:
            logger.info(
                "  🧭 Verdict normalized for %s: %s → %s",
                ticker,
                parsed["final_verdict"],
                normalized_no_buy,
            )
            parsed["final_verdict"] = normalized_no_buy
            final_action = normalized_no_buy

        # No-new-capital verdicts must never carry score-mapped starter allocations.
        # The CIO can still describe a future alert/entry zone in recommended_action,
        # but report/header allocation has to reflect today's final action.
        _NO_NEW_CAPITAL = {"WATCH", "DEFER", "AVOID", "SELL", "TRIM", "HOLD"}
        if parsed["final_verdict"] in _NO_NEW_CAPITAL:
            final_alloc_pct = 0.0
            if parsed["final_verdict"] in {"WATCH", "DEFER", "AVOID"}:
                target_pct = 0.0

        synthesis, parsed["recommended_action"], scoring = _synchronize_final_action_surfaces(
            synthesis=synthesis,
            scoring=scoring,
            final_action=parsed["final_verdict"],
            recommended_action=parsed.get("recommended_action", ""),
            final_alloc_pct=final_alloc_pct,
            no_new_capital=parsed["final_verdict"] in _NO_NEW_CAPITAL,
        )

        if mechanical_override_reason and "MECHANICAL OVERRIDE NOTE:" not in (synthesis or ""):
            synthesis += f"\n\nMECHANICAL OVERRIDE NOTE: {mechanical_override_reason}"

        final_decision = CouncilDecision(
            ticker=ticker,
            final_verdict=parsed["final_verdict"],
            consensus=parsed["consensus"],
            recommended_action=parsed["recommended_action"],
            allocation=f"{final_alloc_pct:.1f}% NAV (~${(deployment.get('total_nav', 0) or 0) * final_alloc_pct / 100:,.0f})",
            synthesis_report=synthesis,
            fundamental=fundamental,
            technical=technical,
            contrarian=contrarian,
            opportunity_score=raw_score,
            adjusted_score=adjusted_score,
            score_components=score_components,
            confidence=cio_confidence,
            thesis_type=thesis_type,
            recommended_allocation_pct=final_alloc_pct,
            entry_valid_until=entry_valid_until,
            invalidation_conditions=invalidation_conditions if isinstance(invalidation_conditions, list) else [],
            stop_loss_pct=stop_loss_pct,
            target_pct=target_pct,
            deployment_context=deployment,
            hard_risk_gate_passed=True,
            hard_risk_gate_reason="",
            data_quality_report=asdict(data_quality),
            agentic_trace=agentic_trace,
            valuation_expectations=valuation_expectations,
            portfolio_factor_risk=portfolio_factor_risk,
            calibration_meta_signal=calibration_meta_signal,
            base_opportunity_score=int(scoring_audit.get("base_score") or 0),
            rule_adjustment_total=int(scoring_audit.get("rule_adjustment_total") or 0),
            cio_adjustment=int(scoring_audit.get("cio_adjustment") or 0),
            scoring_audit=scoring_audit,
            cio_decision=cio_decision,
            half_size=bool(half_size_final and parsed["final_verdict"] in _BUY_SIDE_ACTIONS),
            regime_gate={
                "state": regime_gate_label,
                "vol_multiplier": float(regime_gate_state.get("vol_multiplier", 1.0) or 1.0),
            },
            mechanical_override=mechanical_override_reason,
        )

        try:
            final_decision.dossier_path = write_decision_dossier(
                decision=final_decision,
                stock_data=stock_data,
                macro_data=macro_data or {},
                market_overview=market_overview or {},
                intelligence_brief=intelligence_brief,
                pre_brief=pre_brief_text,
                momentum_context=momentum_context,
            )
        except Exception as dossier_err:
            logger.warning("[council] Failed to write decision dossier for %s: %s", ticker, dossier_err)

        try:
            if final_decision.final_verdict in {"DEFER", "WATCH"} and ticker.upper() not in held_tickers:
                current_price = (
                    (stock_data.get("quote") or {}).get("price") or
                    (stock_data.get("yf_quote") or {}).get("price")
                )
                try:
                    current_price_float = float(str(current_price).replace(",", "")) if current_price else None
                except Exception:
                    current_price_float = None
                watch = record_defer_watch(final_decision, current_price_float, self.journal)
                if watch:
                    final_decision.defer_watch_id = str(watch.get("watch_id") or "")
        except Exception as watch_err:
            logger.warning("[council] Failed to record DEFER watch for %s: %s", ticker, watch_err)

        try:
            shadow_rules = evaluate_shadow_rules_for_decision(final_decision, stock_data, self.journal)
            if shadow_rules:
                final_decision.calibration_meta_signal = {
                    **(final_decision.calibration_meta_signal or {}),
                    "shadow_rule_evaluations_logged": len(shadow_rules),
                }
        except Exception as shadow_rule_err:
            logger.warning("[council] Failed to log shadow rule evaluations for %s: %s", ticker, shadow_rule_err)

        # Auto-create pending thesis for buy-side recommendations
        # FIX 8: Check for existing pending/active thesis to avoid duplicates
        _BUY_VERDICTS = {"BUY", "STARTER", "TACTICAL_BUY", "ACCUMULATE", "ADD"}
        if final_decision.final_verdict in _BUY_VERDICTS:
            try:
                from .thesis_tracker import ThesisTracker
                tracker = ThesisTracker(journal=self.journal)

                # Skip if active thesis already exists — position is already tracked
                existing_active = tracker.get_active(ticker)
                if existing_active:
                    logger.info(
                        "[council] Skipping thesis creation — active thesis %s already exists for %s",
                        existing_active.thesis_id[:8], ticker,
                    )
                else:
                    # Extract price target and stop from decision
                    price = (stock_data.get("quote") or {}).get("price") or \
                            (stock_data.get("yf_quote") or {}).get("price")
                    price_target = None
                    stop_pct = final_decision.stop_loss_pct
                    if price and final_decision.target_pct:
                        try:
                            price_target = float(price) * (1 + float(final_decision.target_pct or 0))
                        except Exception:
                            pass

                    # Extract regime info — prefer the mechanical gate state
                    # (matches data/regime_state.json, which the sell engine
                    # compares entry_regime against); fall back to the
                    # portfolio_state keys (it emits both 'regime_label' and
                    # 'regime').
                    regime_str = None
                    try:
                        deploy_ctx = final_decision.deployment_context or {}
                        regime_str = (
                            regime_gate_label
                            or deploy_ctx.get("regime_label")
                            or deploy_ctx.get("regime")
                            or deploy_ctx.get("fear_greed_label")
                        )
                    except Exception:
                        pass

                    # Build thesis summary from synthesis first paragraph
                    thesis_summary = ""
                    if synthesis:
                        for line in synthesis.split("\n"):
                            line = line.strip()
                            if line and not line.startswith("**") and not line.startswith("#"):
                                thesis_summary = line[:500]
                                break

                    existing_pending = tracker.get_pending_for_ticker(ticker)
                    if existing_pending:
                        # Update existing pending thesis instead of creating a duplicate
                        # Include all fields that a new council decision might change, especially
                        # position_type so stale STARTER/BUY verdicts don't survive a re-analysis.
                        tracker.update_thesis_fields(
                            existing_pending.thesis_id,
                            position_type=final_decision.final_verdict,
                            thesis_summary=thesis_summary,
                            invalidation_conditions=final_decision.invalidation_conditions or [],
                            price_target=price_target,
                            stop_loss_pct=float(stop_pct) if stop_pct else None,
                            recommended_allocation_pct=float(final_alloc_pct or 0),
                            entry_regime=regime_str,
                        )
                        tracker.refresh_pending_expiry(existing_pending.thesis_id)
                        logger.info(
                            "[council] Updated existing pending thesis %s for %s",
                            existing_pending.thesis_id[:8], ticker,
                        )
                    else:
                        thesis = tracker.create_thesis(
                            ticker=ticker,
                            position_type=final_decision.final_verdict,
                            thesis_summary=thesis_summary,
                            invalidation_conditions=final_decision.invalidation_conditions or [],
                            price_target=price_target,
                            stop_loss_pct=float(stop_pct) if stop_pct else None,
                            recommended_allocation_pct=float(final_alloc_pct or 0),
                            council_session_id=session_id if "session_id" in dir() else None,
                            regime=regime_str,
                            notes=f"Council decision score={adjusted_score:.0f} consensus={final_decision.consensus}",
                        )
                        logger.info(
                            "[council] Auto-created pending thesis %s for %s (%s)",
                            thesis.thesis_id[:8],
                            ticker,
                            final_decision.final_verdict,
                        )
            except Exception as thesis_err:
                logger.warning("[council] Failed to auto-create thesis for %s: %s", ticker, thesis_err)

        return final_decision

    def _synthesize(
        self,
        ticker: str,
        stock_data: dict,
        fundamental: AnalystReport,
        technical: AnalystReport,
        contrarian: AnalystReport,
        crisis_prefix: str = "",
        deployment_context: str = "",
        agentic_cio_brief: str = "",
        deterministic_score_audit: str = "",
        lessons: str = "",
        regime_gate_state: dict | None = None,
    ) -> Optional[str]:
        """Run the synthesis mediator (GPT 5.5) with v2 scoring prompt."""
        dcf_data = stock_data.get("dcf") or {}
        pt_data = stock_data.get("price_target_consensus") or {}
        recs_data = (stock_data.get("analyst_recs") or [{}])[0] if stock_data.get("analyst_recs") else {}
        earnings = (stock_data.get("earnings_surprises") or [{}])[0] if stock_data.get("earnings_surprises") else {}
        current_price = (stock_data.get("quote") or {}).get("price", (stock_data.get("yf_quote") or {}).get("price", "N/A"))
        analyst_estimates = stock_data.get("analyst_estimates") or {}
        recommendation_trends = stock_data.get("recommendation_trends") or {}
        short_interest = stock_data.get("short_interest") or {}
        sec_payload = stock_data.get("sec") or {}
        dq_payload = stock_data.get("data_quality_report") or stock_data.get("data_quality") or {}
        valuation_expectations = stock_data.get("valuation_expectations") or {}
        portfolio_factor_risk = stock_data.get("portfolio_factor_risk") or {}
        calibration_meta_signal = stock_data.get("calibration_meta_signal") or {}

        # DCF reliability check
        dcf_value = dcf_data.get("dcf")
        if dcf_value is None:
            dcf_line = "FMP DCF Fair Value: Not available"
        elif dcf_value <= 0:
            dcf_line = (
                f"FMP DCF Fair Value: ${dcf_value:.2f} — ⚠️ MODEL NOT APPLICABLE "
                f"(negative DCF — do NOT use as valuation anchor. Use consensus PT and multiples instead.)"
            )
        elif isinstance(current_price, (int, float)) and current_price > 0:
            dcf_deviation = abs(dcf_value - current_price) / current_price
            if dcf_deviation > 0.70:
                dcf_line = (
                    f"FMP DCF Fair Value: ${dcf_value:.2f} — ⚠️ LOW RELIABILITY "
                    f"(deviates {dcf_deviation:.0%} from current price ${current_price}). Weight consensus PT more heavily."
                )
            else:
                dcf_line = f"FMP DCF Fair Value: ${dcf_value:.2f} — ✓ Model appears reliable"
        else:
            dcf_line = f"FMP DCF Fair Value: ${dcf_value:.2f}"

        # --- Earnings Calendar Context for CIO ---
        earnings_ctx = stock_data.get("earnings_context") or {}
        ec_date = earnings_ctx.get("earnings_date", "Unknown")
        ec_days = earnings_ctx.get("days_to_earnings")
        ec_time = earnings_ctx.get("earnings_time", "unknown")
        ec_risk = earnings_ctx.get("earnings_risk_flag", False)
        ec_defer = earnings_ctx.get("earnings_defer_flag", False)

        if ec_days is not None and ec_days >= 0:
            earnings_calendar_line = (
                f"Next Earnings: {ec_date} ({ec_days} days away, {ec_time})"
            )
            if ec_defer:
                earnings_calendar_line += " — ⚠️ DEFER FLAG: earnings in ≤2 days, binary event risk"
            elif ec_risk:
                earnings_calendar_line += " — ⚠️ RISK FLAG: earnings within 7 days"
        else:
            earnings_calendar_line = "Next Earnings: Date unknown"

        valuation_anchors = (
            f"Current Price: ${current_price}\n"
            f"{dcf_line}\n"
            f"Analyst PT Consensus: ${pt_data.get('targetConsensus', 'N/A')} "
            f"(Median: ${pt_data.get('targetMedian', 'N/A')}, "
            f"Range: ${pt_data.get('targetLow', '?')}-${pt_data.get('targetHigh', '?')})\n"
            f"Forward Estimates: next-q EPS {analyst_estimates.get('next_q_eps_estimate', 'N/A')}, "
            f"next-q revenue {analyst_estimates.get('next_q_revenue_estimate', 'N/A')}, "
            f"FY1 revenue {analyst_estimates.get('fy1_revenue_estimate', 'N/A')}\n"
            f"Recommendation Trend: {recommendation_trends.get('consensus', 'unknown')} "
            f"(net upgrades {recommendation_trends.get('net_upgrades_30d', 'N/A')}, "
            f"net downgrades {recommendation_trends.get('net_downgrades_30d', 'N/A')})\n"
            f"Short Interest: {short_interest.get('short_interest_pct', 'N/A')}% float, "
            f"{short_interest.get('days_to_cover', 'N/A')} days to cover, "
            f"squeeze risk={short_interest.get('squeeze_risk_flag', 'N/A')}\n"
            f"SEC Official Filings: status={sec_payload.get('status', 'N/A')}, "
            f"CIK={sec_payload.get('cik', 'N/A')}, "
            f"latest 10-Q/10-K staleness={sec_payload.get('latest_10q_or_10k_staleness_days', 'N/A')} days, "
            f"tracked facts={sec_payload.get('facts_available', 'N/A')}\n"
            f"Data Coverage: core={dq_payload.get('completeness_score', dq_payload.get('completeness', 'N/A'))}%, "
            f"context={dq_payload.get('context_coverage_score', 'N/A')}%, "
            f"sources={', '.join(dq_payload.get('sources_used', [])) if isinstance(dq_payload.get('sources_used'), list) else 'N/A'}\n"
            f"Analyst Ratings: {recs_data.get('strongBuy', 0)} Strong Buy, {recs_data.get('buy', 0)} Buy, "
            f"{recs_data.get('hold', 0)} Hold, {recs_data.get('sell', 0)} Sell, {recs_data.get('strongSell', 0)} Strong Sell\n"
            f"Most Recent Earnings: Period {earnings.get('period', 'N/A')}, "
            f"Actual EPS ${earnings.get('actual', 'N/A')} vs Est ${earnings.get('estimate', 'N/A')} "
            f"({earnings.get('surprisePercent', 'N/A')}% surprise)\n"
            f"{earnings_calendar_line}\n\n"
            f"{format_valuation_expectations(valuation_expectations)}\n\n"
            f"{format_portfolio_factor_risk(portfolio_factor_risk)}\n\n"
            f"{format_meta_signal(calibration_meta_signal)}"
        )

        # Wave 2 anchors: ATR(14) for the structured decision block's
        # ATR-derived stop, the funnel's mechanical scan signals, and the
        # mechanical regime gate state.
        atr14 = _compute_atr14(stock_data.get("price_history"))
        if atr14 and isinstance(current_price, (int, float)) and current_price > 0:
            atr_mult = float(getattr(Config, "CIO_STOP_ATR_MULT", 2.5))
            stop_cap = float(getattr(Config, "CIO_STOP_CAP_PCT", 0.10))
            atr_stop_price = max(
                current_price - atr_mult * atr14,
                current_price * (1 - stop_cap),
            )
            valuation_anchors += (
                f"\nATR(14): ${atr14:.2f} ({atr14 / current_price:.1%} of price). "
                f"{atr_mult:.1f}x ATR initial stop = ${current_price - atr_mult * atr14:.2f}; "
                f"capped at {stop_cap:.0%} below entry => use stop_price ~ ${atr_stop_price:.2f}."
            )
        scan_signal_row = stock_data.get("scan_signals") or {}
        if scan_signal_row:
            valuation_anchors += (
                f"\nMechanical scan signals ({scan_signal_row.get('scan_date', '?')}): "
                f"track={scan_signal_row.get('track')}, momentum_z={scan_signal_row.get('momentum_z')}, "
                f"composite_z={scan_signal_row.get('composite_z')}, "
                f"52wH_prox={scan_signal_row.get('high_52w_prox')}, "
                f"trend_template={scan_signal_row.get('trend_template')}, "
                f"ID={scan_signal_row.get('information_discreteness')}, "
                f"volume_confirmed={scan_signal_row.get('volume_confirmed')}"
            )
        if regime_gate_state:
            valuation_anchors += (
                f"\nMechanical regime gate: {regime_gate_state.get('state')} "
                f"(vol_multiplier {float(regime_gate_state.get('vol_multiplier', 1.0) or 1.0):.2f})"
            )

        prompt = (crisis_prefix + SYNTHESIS_PROMPT).format(
            ticker=ticker,
            fundamental_report=fundamental.report,
            technical_report=technical.report,
            contrarian_report=contrarian.report,
            valuation_anchors=valuation_anchors,
            deployment_context=deployment_context or "Deployment context not available.",
            agentic_cio_brief=agentic_cio_brief or "Agentic diligence trace not available.",
            deterministic_score_audit=deterministic_score_audit or "Deterministic buy score audit not available.",
            lessons=lessons or "No graded lessons on record yet.",
        )

        try:
            text = ChatGPTBackendClient(timeout=120).chat(prompt)
            return text
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return None
