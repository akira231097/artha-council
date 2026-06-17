"""Macro Regime & Opportunity Layer (MROL).

Two-model regime council (GPT 5.5 × 2 independent calls) independently classifies
the current macro regime, then synthesizes into a unified RegimePacket.
Drives dynamic candidate generation and Core/Tactical/Avoid recommendations.

Research foundation:
- Ray Dalio Four-Quadrant framework (Growth x Inflation)
- FactSet CLI + ITS regime model
- StockTrends rotation analysis
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .chatgpt_backend import ChatGPTBackendClient
from .config import Config
from .regime_indicators import RegimeIndicators, compute_regime_indicators
from .regime_mapping import (
    REGIME_TAXONOMY,
    BEGINNER_RULES,
    format_regime_taxonomy_for_prompt,
    get_regime_candidates,
)

logger = logging.getLogger(__name__)

# Validation constants
from .regime_mapping import get_base_regimes, get_overlay_regimes
VALID_BASE_REGIMES = set(get_base_regimes())
VALID_OVERLAYS = set(get_overlay_regimes())

# Beginner-inappropriate instruments (futures-based, leveraged, complex)
BEGINNER_DISALLOWED_TACTICAL = {"USO", "UCO", "SCO", "BOIL", "KOLD", "UVXY", "SVXY"}

PERSISTENCE_RANK = {"new_today": 0, "multi_day": 1, "multi_week": 2, "established": 3}


def _clamp_conf(value, default: float = 0.0) -> float:
    """Clamp confidence to [0.0, 1.0] range."""
    try:
        v = float(value)
        return max(0.0, min(1.0, v))
    except (TypeError, ValueError):
        return default


def _safe_list(value) -> list:
    """Ensure value is a list."""
    return value if isinstance(value, list) else []


def _normalize_overlay(overlay: dict) -> Optional[dict]:
    """Validate and normalize an event overlay dict."""
    if not isinstance(overlay, dict):
        return None
    otype = overlay.get("type", "")
    if otype not in VALID_OVERLAYS:
        logger.warning(f"Unknown overlay type '{otype}' — skipping")
        return None
    return {
        "type": otype,
        "confidence": _clamp_conf(overlay.get("confidence", 0.0)),
        "urgency": overlay.get("urgency", "low") if overlay.get("urgency") in {"low", "medium", "high"} else "low",
        "persistence": overlay.get("persistence", "unknown"),
        "expected_horizon": overlay.get("expected_horizon", "unknown"),
        "evidence": _safe_list(overlay.get("evidence")),
        "disconfirming_evidence": _safe_list(overlay.get("disconfirming_evidence")),
        "scenario_checks": overlay.get("scenario_checks", {}) if isinstance(overlay.get("scenario_checks", {}), dict) else {},
    }


def _is_valid_assessment(a: RegimeAssessment) -> bool:
    """Check if an assessment is semantically valid."""
    return (
        not a.error
        and a.base_regime_type in VALID_BASE_REGIMES
        and 0.0 <= a.base_regime_confidence <= 1.0
    )


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class RegimeAssessment:
    """Single analyst's regime classification."""
    model: str = ""
    base_regime_type: str = ""
    base_regime_confidence: float = 0.0
    base_regime_evidence: List[str] = field(default_factory=list)
    base_regime_disconfirming: List[str] = field(default_factory=list)
    event_overlays: List[Dict[str, Any]] = field(default_factory=list)
    market_narrative: str = ""
    top_beneficiary_sectors: List[str] = field(default_factory=list)
    top_risk_sectors: List[str] = field(default_factory=list)
    raw_response: str = ""
    error: Optional[str] = None


@dataclass
class RegimePacket:
    """Synthesized regime classification from the council."""
    as_of: str = ""

    # Base regime
    base_regime_type: str = ""
    base_regime_label: str = ""
    base_regime_confidence: float = 0.0
    base_regime_evidence: List[str] = field(default_factory=list)

    # Event overlays (can be multiple)
    event_overlays: List[Dict[str, Any]] = field(default_factory=list)

    # Narrative
    market_narrative: str = ""

    # Recommendations
    core_recommendation: Dict[str, str] = field(default_factory=dict)
    tactical_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    avoid_list: List[Dict[str, str]] = field(default_factory=list)

    # Council agreement
    council_agreement: str = ""  # "unanimous", "majority", "split"
    individual_assessments: List[Dict[str, Any]] = field(default_factory=list)

    # Candidates for downstream analyst council
    candidates: List[str] = field(default_factory=list)

    def to_context_string(self) -> str:
        """Format regime packet as context string for analyst prompts."""
        lines = [
            f"ACTIVE MACRO REGIME: {self.base_regime_label} "
            f"(confidence: {self.base_regime_confidence:.0%})",
            f"Council agreement: {self.council_agreement}",
            "",
            f"Market narrative: {self.market_narrative}",
        ]

        if self.event_overlays:
            lines.append("\nACTIVE EVENT OVERLAYS:")
            for overlay in self.event_overlays:
                conf = overlay.get("confidence", 0)
                lines.append(
                    f"  - {overlay.get('label', 'Unknown')} "
                    f"(confidence: {conf:.0%}, "
                    f"urgency: {overlay.get('urgency', 'unknown')})"
                )

        if self.tactical_recommendations:
            lines.append("\nREGIME-DRIVEN TACTICAL CANDIDATES:")
            for rec in self.tactical_recommendations:
                lines.append(f"  - {rec.get('ticker', '?')}: {rec.get('reason', '')}")

        if self.avoid_list:
            lines.append("\nREGIME AVOID LIST:")
            for item in self.avoid_list:
                lines.append(f"  - {item.get('ticker', item.get('sector', '?'))}: "
                             f"{item.get('reason', '')}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Regime Analyst Prompt
# ---------------------------------------------------------------------------

REGIME_ANALYST_PROMPT = """You are a macro regime analyst for the Artha Investment Council.

TASK: Analyze the current market environment and classify it into:
1. Exactly ONE base economic regime (from the Dalio Four-Quadrant framework)
2. Zero or more event-driven overlays (acute market events)

{regime_taxonomy}

{market_data}

INVESTOR CONTEXT:
- Beginner investor, 29yo
- Core: $350/month into FXAIX (S&P 500 index fund, self-managed, already set up)
- Satellite: $350/month currently available to the AI council for individual-stock tactical opportunities
- Moderate risk tolerance (3/5)
- Artha can recommend tactical stock opportunities inside the $350/month satellite budget
- Do NOT recommend broad index funds (VOO, VTI, FXAIX) — core is already covered

OUTPUT FORMAT — Return ONLY valid JSON, no markdown, no explanation:
{{
    "base_regime": {{
        "type": "goldilocks|reflation|stagflation|risk_off",
        "confidence": 0.0,
        "evidence": ["specific data point 1", "specific data point 2"],
        "disconfirming_evidence": ["what argues against this"]
    }},
    "event_overlays": [
        {{
            "type": "regime_type_from_taxonomy",
            "confidence": 0.0,
            "urgency": "low|medium|high",
            "persistence": "new_today|multi_day|multi_week|established",
            "expected_horizon": "days|weeks|months|quarters",
            "evidence": ["specific data points"],
            "disconfirming_evidence": ["what argues against"],
            "scenario_checks": {{
                "base_case": "what happens if current trajectory continues",
                "upside_case": "what happens if situation intensifies",
                "downside_case": "what happens if situation reverses"
            }}
        }}
    ],
    "market_narrative": "2-3 sentence plain English summary",
    "top_beneficiary_sectors": ["sector1", "sector2"],
    "top_risk_sectors": ["sector1", "sector2"]
}}

RULES:
- You MUST cite specific numbers from the market data as evidence
- Confidence must reflect actual data support (0.0 to 1.0)
- If no clear event overlay exists, return empty event_overlays array
- Maximum 3 event overlays if genuinely concurrent
- Be honest about persistence — a one-day move is "new_today"
- Return ONLY the JSON object, nothing else"""


# ---------------------------------------------------------------------------
# Model Callers
# ---------------------------------------------------------------------------

def _run_claude_regime_analyst(indicators: RegimeIndicators) -> RegimeAssessment:
    """Run the first GPT regime analyst (independent call A)."""
    assessment = RegimeAssessment(model=Config.GPT_MODEL)

    try:
        prompt = REGIME_ANALYST_PROMPT.format(
            regime_taxonomy=format_regime_taxonomy_for_prompt(),
            market_data=indicators.to_prompt_text(),
        )

        client = ChatGPTBackendClient(timeout=120)
        raw_text = client.chat(prompt)
        raw_text = raw_text.strip()
        assessment.raw_response = raw_text
        logger.info("  📝 %s regime analyst A complete (%d chars)", Config.GPT_MODEL, len(raw_text))

        # Parse JSON
        parsed = _parse_regime_json(raw_text)
        if parsed:
            _populate_assessment(assessment, parsed)
        else:
            assessment.error = "Failed to parse JSON from GPT response (A)"

    except Exception as e:
        assessment.error = str(e)
        logger.error(f"GPT regime analyst A failed: {e}")

    return assessment


def _run_gpt_regime_analyst(indicators: RegimeIndicators) -> RegimeAssessment:
    """Run the second GPT regime analyst via ChatGPT backend."""
    assessment = RegimeAssessment(model=Config.GPT_MODEL)

    try:
        prompt = REGIME_ANALYST_PROMPT.format(
            regime_taxonomy=format_regime_taxonomy_for_prompt(),
            market_data=indicators.to_prompt_text(),
        )

        client = ChatGPTBackendClient(timeout=120)
        raw_text = client.chat(prompt)
        assessment.raw_response = raw_text
        logger.info("  📝 %s regime analyst complete (%d chars)", Config.GPT_MODEL, len(raw_text))

        # Parse JSON
        parsed = _parse_regime_json(raw_text)
        if parsed:
            _populate_assessment(assessment, parsed)
        else:
            assessment.error = "Failed to parse JSON from GPT response"

    except Exception as e:
        assessment.error = str(e)
        logger.error(f"GPT regime analyst failed: {e}")

    return assessment


def _parse_regime_json(text: str) -> Optional[dict]:
    """Parse JSON from model response, handling markdown wrapping."""
    # Strip markdown code blocks if present
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.split("```json", 1)[1]
        if "```" in cleaned:
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned.split("```", 1)[1]
        if "```" in cleaned:
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}")
        # Try to extract JSON object from text
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
        return None


def _populate_assessment(assessment: RegimeAssessment, parsed: dict) -> None:
    """Populate RegimeAssessment from parsed JSON with strict validation."""
    base = parsed.get("base_regime", {})
    if not isinstance(base, dict):
        assessment.error = "Invalid base_regime object"
        return

    base_type = base.get("type", "")
    if base_type not in VALID_BASE_REGIMES:
        assessment.error = f"Invalid base_regime.type: {base_type} (valid: {VALID_BASE_REGIMES})"
        return

    assessment.base_regime_type = base_type
    assessment.base_regime_confidence = _clamp_conf(base.get("confidence", 0.0))
    assessment.base_regime_evidence = _safe_list(base.get("evidence"))
    assessment.base_regime_disconfirming = _safe_list(base.get("disconfirming_evidence"))

    # Validate and normalize overlays
    overlays = []
    for overlay in _safe_list(parsed.get("event_overlays")):
        norm = _normalize_overlay(overlay)
        if norm:
            overlays.append(norm)
    assessment.event_overlays = overlays

    assessment.market_narrative = str(parsed.get("market_narrative", ""))[:500]
    assessment.top_beneficiary_sectors = _safe_list(parsed.get("top_beneficiary_sectors"))
    assessment.top_risk_sectors = _safe_list(parsed.get("top_risk_sectors"))


# ---------------------------------------------------------------------------
# Council Synthesis
# ---------------------------------------------------------------------------

def _synthesize_regime_council(
    claude_assessment: RegimeAssessment,
    gpt_assessment: RegimeAssessment,
    indicators: RegimeIndicators,
) -> RegimePacket:
    """Merge two independent regime assessments into a unified RegimePacket.

    Agreement rules:
    - Both agree on base regime: confidence = average * 1.15 (cap 0.95)
    - They disagree: confidence = lower * 0.8, use higher-confidence model's type
    - Event overlays: include if EITHER identifies with confidence > 0.5
    - Both identify same overlay: confidence boost of 1.15x
    """
    packet = RegimePacket()
    packet.as_of = indicators.as_of

    # Determine which assessments are semantically valid
    valid_assessments = []
    if _is_valid_assessment(claude_assessment):
        valid_assessments.append(claude_assessment)
    else:
        logger.warning(f"Regime analyst A invalid: {claude_assessment.error or 'failed validation'}")
    if _is_valid_assessment(gpt_assessment):
        valid_assessments.append(gpt_assessment)
    else:
        logger.warning(f"GPT assessment invalid: {gpt_assessment.error or 'failed validation'}")

    if not valid_assessments:
        # Both failed — return explicit unknown state (no fake classification)
        logger.error("Both regime analysts failed — returning unknown regime")
        packet.base_regime_type = "unknown"
        packet.base_regime_label = "Unknown — Insufficient Data"
        packet.base_regime_confidence = 0.0
        packet.market_narrative = "Regime analysis unavailable. Defaulting to core-only stance."
        packet.council_agreement = "failed"
        return packet

    if len(valid_assessments) == 1:
        # Single model — use its assessment directly
        a = valid_assessments[0]
        packet.base_regime_type = a.base_regime_type
        packet.base_regime_confidence = a.base_regime_confidence * 0.9  # Slight penalty for single-model
        packet.event_overlays = a.event_overlays
        packet.market_narrative = a.market_narrative
        packet.council_agreement = f"single_model ({a.model})"
        logger.info(f"  ⚠️ Single regime analyst ({a.model}): {a.base_regime_type} "
                    f"({packet.base_regime_confidence:.0%})")
    else:
        # Both models provided assessments
        ca = claude_assessment
        ga = gpt_assessment

        if ca.base_regime_type == ga.base_regime_type:
            # Agreement! Boost confidence
            packet.base_regime_type = ca.base_regime_type
            avg_conf = (ca.base_regime_confidence + ga.base_regime_confidence) / 2
            packet.base_regime_confidence = min(0.90, avg_conf + 0.05)
            packet.council_agreement = "unanimous"
            # Merge evidence
            packet.base_regime_evidence = list(set(ca.base_regime_evidence + ga.base_regime_evidence))
            logger.info(f"  ✅ Regime council AGREES: {packet.base_regime_type} "
                        f"({packet.base_regime_confidence:.0%})")
        else:
            # Disagreement — use higher confidence model's type, lower confidence
            ca_conf = _clamp_conf(ca.base_regime_confidence)
            ga_conf = _clamp_conf(ga.base_regime_confidence)
            if ca_conf >= ga_conf:
                packet.base_regime_type = ca.base_regime_type
                winner_conf, loser_conf = ca_conf, ga_conf
            else:
                packet.base_regime_type = ga.base_regime_type
                winner_conf, loser_conf = ga_conf, ca_conf
            # Weighted merge with penalty for disagreement
            packet.base_regime_confidence = max(0.35, min(0.75,
                winner_conf * 0.75 + loser_conf * 0.25 - 0.10))
            packet.council_agreement = "split"
            packet.base_regime_evidence = (
                ca.base_regime_evidence + ga.base_regime_evidence
            )
            logger.info(f"  ⚠️ Regime council SPLIT: A={ca.base_regime_type} "
                        f"({ca.base_regime_confidence:.0%}) vs B={ga.base_regime_type} "
                        f"({ga.base_regime_confidence:.0%}) → using {packet.base_regime_type}")

        # Merge narratives (prefer analyst A but append analyst B if different)
        packet.market_narrative = ca.market_narrative or ga.market_narrative

        # Merge event overlays
        overlay_map: Dict[str, Dict] = {}
        for a in [ca, ga]:
            for overlay in a.event_overlays:
                otype = overlay.get("type", "")
                conf = float(overlay.get("confidence", 0))
                if conf < 0.50:  # Skip low-confidence overlays
                    continue
                if otype in overlay_map:
                    # Both models identified this overlay — boost
                    existing = overlay_map[otype]
                    avg = (existing["confidence"] + conf) / 2
                    existing["confidence"] = min(0.95, avg * 1.15)
                    existing["_both_models"] = True
                    # Merge evidence
                    existing_evidence = existing.get("evidence", [])
                    new_evidence = overlay.get("evidence", [])
                    existing["evidence"] = list(set(existing_evidence + new_evidence))
                    # Take higher urgency
                    urgency_rank = {"low": 0, "medium": 1, "high": 2}
                    if urgency_rank.get(overlay.get("urgency", ""), 0) > urgency_rank.get(existing.get("urgency", ""), 0):
                        existing["urgency"] = overlay.get("urgency")
                else:
                    overlay_map[otype] = dict(overlay)
                    overlay_map[otype]["_both_models"] = False

        # Enrich overlays with labels from taxonomy
        enriched_overlays = []
        for o in overlay_map.values():
            otype = o.get("type", "")
            regime_info = REGIME_TAXONOMY.get(otype, {})
            o["label"] = regime_info.get("label", otype)
            enriched_overlays.append(o)
        packet.event_overlays = enriched_overlays

    # Set label from taxonomy
    regime_info = REGIME_TAXONOMY.get(packet.base_regime_type, {})
    packet.base_regime_label = regime_info.get("label", packet.base_regime_type)

    # Store individual assessments for transparency
    for a in [claude_assessment, gpt_assessment]:
        packet.individual_assessments.append({
            "model": a.model,
            "base_regime": a.base_regime_type,
            "confidence": a.base_regime_confidence,
            "overlays": [o.get("type", "") for o in a.event_overlays] if not a.error else [],
            "error": a.error,
        })

    return packet


# ---------------------------------------------------------------------------
# Recommendation Generator
# ---------------------------------------------------------------------------

def _generate_recommendations(packet: RegimePacket) -> None:
    """Generate Core / Tactical / Avoid recommendations from the regime packet."""

    # If regime is unknown, only recommend core
    if packet.base_regime_type == "unknown":
        packet.core_recommendation = {
            "ticker": "FXAIX",
            "action": "CONTINUE_DCA",
            "allocation_pct": 100,
            "reason": "Regime analysis unavailable. Stick to core index fund.",
        }
        packet.tactical_recommendations = []
        packet.avoid_list = []
        return

    # Core: Sarath already has $350/month FXAIX auto-recurring
    packet.core_recommendation = {
        "ticker": "FXAIX",
        "action": "ALREADY_SET",
        "allocation_usd": 350,
        "reason": "Your $350/month FXAIX auto-buy is your core. Already running on Fidelity. No changes needed regardless of regime.",
    }

    # Tactical: from event overlays first, then base regime
    tactical = []
    seen_tickers = set()

    # Event overlay candidates (higher priority)
    for overlay in packet.event_overlays:
        otype = overlay.get("type", "")
        conf = overlay.get("confidence", 0)

        if conf < BEGINNER_RULES["min_overlay_confidence"]:
            continue

        regime_info = REGIME_TAXONOMY.get(otype, {})
        # ETFs first
        for etf in regime_info.get("beneficiary_etfs", [])[:2]:
            if etf not in seen_tickers and etf != "VOO":
                # Skip beginner-inappropriate instruments
                if etf in BEGINNER_DISALLOWED_TACTICAL:
                    continue
                seen_tickers.add(etf)
                # Enforce persistence rules
                persistence = overlay.get("persistence", "new_today")
                if PERSISTENCE_RANK.get(persistence, 0) < BEGINNER_RULES.get("min_persistence_regular", 1):
                    action = "WATCH"
                else:
                    action = "SMALL_BUY" if conf >= 0.70 else "WATCH"
                tactical.append({
                    "ticker": etf,
                    "action": action,
                    "confidence": round(conf, 2),
                    "allocation_usd": BEGINNER_RULES.get("max_single_position_usd", 100),
                    "reason": f"Regime play: {regime_info.get('label', otype)}",
                    "risk": overlay.get("scenario_checks", {}).get("downside_case", "Could reverse if conditions change"),
                    "source": "event_overlay",
                })
        # Top 2 stocks (only if ETF preference is disabled or no ETFs were added)
        if not BEGINNER_RULES.get("prefer_etfs_over_stocks", True) or not tactical:
            pass  # Allow stocks only when ETFs unavailable
        for stock in regime_info.get("beneficiary_stocks", [])[:2]:
            if stock not in seen_tickers and stock != "VOO" and stock not in BEGINNER_DISALLOWED_TACTICAL:
                seen_tickers.add(stock)
                tactical.append({
                    "ticker": stock,
                    "action": "WATCH",
                    "confidence": round(conf * 0.9, 2),  # Slight confidence penalty for single names
                    "allocation_usd": BEGINNER_RULES.get("max_single_position_usd", 100),
                    "reason": f"Regime beneficiary: {regime_info.get('label', otype)}",
                    "risk": "Single-stock risk. ETF preferred for beginners.",
                    "source": "event_overlay",
                })

    # Base regime candidates (lower priority, fill remaining slots)
    base_info = REGIME_TAXONOMY.get(packet.base_regime_type, {})
    if packet.base_regime_confidence >= BEGINNER_RULES["min_regime_confidence"]:
        for etf in base_info.get("beneficiary_etfs", [])[:2]:
            if etf not in seen_tickers and etf != "VOO" and len(tactical) < 6:
                seen_tickers.add(etf)
                tactical.append({
                    "ticker": etf,
                    "action": "WATCH",
                    "confidence": round(packet.base_regime_confidence * 0.8, 2),
                    "allocation_usd": BEGINNER_RULES.get("max_single_position_usd", 100),
                    "reason": f"Base regime beneficiary: {packet.base_regime_label}",
                    "risk": "Regime may shift. Monitor for persistence.",
                    "source": "base_regime",
                })

    # Enforce aggregate tactical allocation budget
    remaining_pct = BEGINNER_RULES.get("monthly_budget", 0)  # Satellite budget in dollars; currently paused
    capped_tactical = []
    for rec in tactical:
        max_per_position = BEGINNER_RULES.get("max_single_position_usd", 100)
        alloc = min(rec.get("allocation_usd", max_per_position), remaining_pct)
        if alloc <= 0:
            break
        rec["allocation_usd"] = alloc
        remaining_pct -= alloc
        capped_tactical.append(rec)
    # Filter out excluded tickers (leveraged ETFs, futures products, etc.)
    excluded = set(BEGINNER_RULES.get("excluded_tickers", []))
    capped_tactical = [r for r in capped_tactical if r.get("ticker", "") not in excluded]
    packet.tactical_recommendations = capped_tactical[:4]  # Max 4 for beginner clarity

    # Avoid list: from overlays and base regime
    avoid = []
    avoid_seen = set()
    for overlay in packet.event_overlays:
        otype = overlay.get("type", "")
        regime_info = REGIME_TAXONOMY.get(otype, {})
        for etf in regime_info.get("avoid_etfs", []):
            if etf not in avoid_seen:
                avoid_seen.add(etf)
                avoid.append({"ticker": etf, "reason": f"Pressured by {regime_info.get('label', otype)}"})
        for stock in regime_info.get("avoid_stocks", []):
            if stock not in avoid_seen:
                avoid_seen.add(stock)
                avoid.append({"ticker": stock, "reason": f"At risk from {regime_info.get('label', otype)}"})
        for sector in regime_info.get("avoid_sectors", []):
            if sector not in avoid_seen:
                avoid_seen.add(sector)
                avoid.append({"sector": sector, "reason": f"Sector headwind: {regime_info.get('label', otype)}"})

    # Base regime avoid
    for sector in base_info.get("avoid_sectors", []):
        if sector not in avoid_seen:
            avoid_seen.add(sector)
            avoid.append({"sector": sector, "reason": f"Base regime headwind: {packet.base_regime_label}"})

    packet.avoid_list = avoid


# ---------------------------------------------------------------------------
# Candidate Generation
# ---------------------------------------------------------------------------

def _generate_candidates(packet: RegimePacket) -> List[str]:
    """Generate candidate tickers for downstream analyst council.

    Priority order:
    1. Tactical ETFs from event overlays
    2. Tactical stocks from event overlays
    3. ETFs from base regime
    4. Core ETF (VOO) always included
    """
    candidates = []
    seen = set()

    # From tactical recommendations
    for rec in packet.tactical_recommendations:
        ticker = rec.get("ticker", "")
        if ticker and ticker not in seen:
            seen.add(ticker)
            candidates.append(ticker)

    # Core (FXAIX) is self-managed — don't include in council analysis candidates

    return candidates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_regime_council(macro_data: Optional[dict] = None) -> RegimePacket:
    """Run the full MROL pipeline: compute indicators → regime council → recommendations.

    Args:
        macro_data: Optional pre-collected macro data from DataCollector.collect_macro()

    Returns:
        RegimePacket with base regime, overlays, recommendations, and candidates.
    """
    logger.info("🌍 Running Macro Regime & Opportunity Layer (MROL)...")

    # Step 1: Compute indicators
    indicators = compute_regime_indicators(macro_data)

    # Step 2: Run both regime analysts in parallel
    logger.info("  🏛️ Running regime council (%s × 2) in parallel...", Config.GPT_MODEL)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        claude_future = executor.submit(_run_claude_regime_analyst, indicators)
        gpt_future = executor.submit(_run_gpt_regime_analyst, indicators)
        claude_result = claude_future.result(timeout=120)
        gpt_result = gpt_future.result(timeout=120)

    # Step 3: Synthesize
    logger.info("  🔄 Synthesizing regime assessments...")
    packet = _synthesize_regime_council(claude_result, gpt_result, indicators)

    # Step 4: Generate recommendations
    _generate_recommendations(packet)

    # Step 5: Generate candidate list
    packet.candidates = _generate_candidates(packet)

    logger.info(f"  ✅ MROL complete: {packet.base_regime_label} ({packet.base_regime_confidence:.0%}), "
                f"{len(packet.event_overlays)} overlay(s), {len(packet.candidates)} candidates")

    return packet


def format_regime_report(packet: RegimePacket) -> str:
    """Format the regime packet as a human-readable report section."""
    lines = []

    # Header
    lines.append("━━━━━ MARKET REGIME ━━━━━")
    lines.append("")

    # Base regime
    conf_pct = f"{packet.base_regime_confidence:.0%}"
    lines.append(f"🌍 BASE REGIME: {packet.base_regime_label}")
    lines.append(f"   Confidence: {conf_pct} | Council: {packet.council_agreement}")
    lines.append("")
    lines.append(f"📖 {packet.market_narrative}")

    # Event overlays
    if packet.event_overlays:
        lines.append("")
        for overlay in packet.event_overlays:
            label = overlay.get("label", "")
            if not label:
                otype = overlay.get("type", "")
                regime_info = REGIME_TAXONOMY.get(otype, {})
                label = regime_info.get("label", otype)
            conf = overlay.get("confidence", 0)
            urgency = overlay.get("urgency", "unknown")
            persistence = overlay.get("persistence", "unknown")
            horizon = overlay.get("expected_horizon", "unknown")

            emoji = "🔴" if urgency == "high" else "🟡" if urgency == "medium" else "🟢"
            lines.append(f"{emoji} ACTIVE EVENT: {label}")
            lines.append(f"   Confidence: {conf:.0%} | Urgency: {urgency} | "
                         f"Status: {persistence} | Horizon: {horizon}")

            # Evidence
            evidence = overlay.get("evidence", [])
            if evidence:
                lines.append(f"   Evidence:")
                for e in evidence[:4]:
                    lines.append(f"     • {e}")

            # Scenario
            scenarios = overlay.get("scenario_checks", {})
            if scenarios:
                dc = scenarios.get("downside_case", "")
                if dc:
                    lines.append(f"   ⚠️ What could change: {dc}")

    lines.append("")

    # Recommendations
    lines.append("━━━━━ RECOMMENDATIONS ━━━━━")
    lines.append("")

    # Core (self-managed, just a reminder)
    core = packet.core_recommendation
    if core:
        lines.append(f"💚 CORE (self-managed, not allocated by Artha)")
        lines.append(f"   • {core.get('ticker', 'FXAIX')} — {core.get('reason', 'Auto-recurring. No changes needed.')}")
        lines.append("")

    # Tactical (only if Sarath re-enables the Artha satellite budget)
    if packet.tactical_recommendations:
        budget = BEGINNER_RULES.get("monthly_budget", 0)
        lines.append(f"🟡 ARTHA COUNCIL PICKS ({budget} USD satellite budget)")
        for rec in packet.tactical_recommendations:
            action_emoji = "✅" if rec.get("action") == "SMALL_BUY" else "👀"
            lines.append(
                f"   {action_emoji} {rec['ticker']} — {rec.get('action', 'WATCH')} "
                f"(confidence: {rec.get('confidence', 0):.0%})"
            )
            lines.append(f"      Why: {rec.get('reason', '')}")
            if rec.get("risk"):
                lines.append(f"      Risk: {rec['risk']}")
        lines.append("")

    # Avoid
    if packet.avoid_list:
        lines.append("🔴 AVOID")
        for item in packet.avoid_list[:8]:
            target = item.get("ticker", item.get("sector", "?"))
            lines.append(f"   • {target} — {item.get('reason', '')}")
        lines.append("")

    return "\n".join(lines)
