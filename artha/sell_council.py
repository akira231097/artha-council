"""The Artha Sell Council — sell-side debate and action engine.

Mirrors the buy-side ArthaCouncil pattern:
  1. Run 3 analysts independently (Fundamental/GPT, Technical/Gemini, Contrarian/GPT)
  2. CIO synthesis computes sell score (0-100)
  3. Score → action mapping with position-type adjustments

Actions: HOLD | TRIM | EXIT | URGENT_EXIT
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .config import Config
from .journal import DecisionJournal
from .chatgpt_backend import ChatGPTBackendClient
from .gemini_client import gemini_generate
from .analysts import _serialize_data
from .sell_prompts import (
    SELL_FUNDAMENTAL_ANALYST,
    SELL_TECHNICAL_ANALYST,
    SELL_CONTRARIAN_ANALYST,
    SELL_CONDITION_REVIEW_FORMAT,
    build_sell_context,
    build_sell_synthesis_prompt,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat()


def _add_days(days: int) -> str:
    return (_utcnow() + timedelta(days=days)).isoformat()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SellAnalystReport:
    """One sell analyst's output."""
    analyst_name: str
    model: str
    verdict: str        # HOLD | TRIM | EXIT
    sell_score: int     # 0-100 component score
    confidence: int     # 1-10
    report: str


@dataclass
class SellDecision:
    """Final synthesized sell council decision."""
    ticker: str
    position_type: str
    action: str                         # HOLD | TRIM | EXIT | URGENT_EXIT
    sell_score: float                   # 0-100 final score
    thesis_status: str                  # INTACT | WEAKENED | DAMAGED | BROKEN
    health_score: Optional[int]         # 0-100 updated thesis health
    fundamental: Optional[SellAnalystReport]
    technical: Optional[SellAnalystReport]
    contrarian: Optional[SellAnalystReport]
    synthesis_report: str
    key_reasons: list[str] = field(default_factory=list)
    next_review_date: Optional[str] = None
    is_urgent: bool = False
    trim_pct: Optional[float] = None    # Fraction to trim (0-1) if action=TRIM
    confidence: int = 5
    trigger_type: str = "periodic_review"
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=_utcnow_iso)
    dossier_path: str = ""
    base_sell_score: float = 0.0
    rule_adjustment_total: float = 0.0
    cio_adjustment: float = 0.0
    scoring_audit: dict[str, Any] = field(default_factory=dict)

    @property
    def is_hold(self) -> bool:
        return self.action == "HOLD"

    @property
    def is_exit(self) -> bool:
        return self.action in ("EXIT", "URGENT_EXIT")


# ---------------------------------------------------------------------------
# Analyst runners (sell-side)
# ---------------------------------------------------------------------------

def _run_sell_fundamental(
    stock_data: dict,
    position_context: str,
    thesis: Any,
) -> Optional[str]:
    """Sell-side Fundamental Analyst — GPT 5.5."""
    from .analysts import _extract_relevant_fundamental_data, _serialize_data as _ser

    relevant = _extract_relevant_fundamental_data(stock_data)
    data_str = _ser(relevant)

    # Build invalidation condition review format hint
    conditions = getattr(thesis, "invalidation_conditions", []) or []
    cond_format = "\n".join(
        f"- {c}: [INTACT|THREATENED|TRIGGERED] — [Evidence]"
        for c in conditions[:8]
    ) or SELL_CONDITION_REVIEW_FORMAT

    prompt = (
        position_context
        + "\n\n"
        + SELL_FUNDAMENTAL_ANALYST.format(
            data=data_str,
            condition_review_format=cond_format,
        )
    )

    try:
        text = ChatGPTBackendClient(timeout=120).chat(prompt)
        return text
    except Exception as e:
        logger.error("[sell_council] Fundamental analyst failed: %s", e)
        return None


def _run_sell_technical(stock_data: dict, position_context: str) -> Optional[str]:
    """Sell-side Technical Analyst — Gemini."""
    from .analysts import _extract_relevant_technical_data

    relevant = _extract_relevant_technical_data(stock_data)
    data_str = _serialize_data(relevant)
    prompt = (
        position_context
        + "\n\n"
        + SELL_TECHNICAL_ANALYST.format(data=data_str)
    )

    try:
        text, _ = gemini_generate(prompt, model=Config.GEMINI_TECHNICAL_MODEL, timeout=90)
        return text
    except Exception as e:
        logger.error("[sell_council] Technical analyst failed: %s", e)
        return None


def _run_sell_contrarian(
    stock_data: dict,
    position_context: str,
) -> Optional[str]:
    """Sell-side Contrarian / Risk Analyst — GPT 5.5."""
    from .analysts import _extract_relevant_risk_data

    relevant = _extract_relevant_risk_data(stock_data)
    data_str = _serialize_data(relevant)
    prompt = (
        position_context
        + "\n\n"
        + SELL_CONTRARIAN_ANALYST.format(data=data_str)
    )

    try:
        client = ChatGPTBackendClient()
        return client.chat(prompt)
    except Exception as e:
        logger.error("[sell_council] Contrarian analyst failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_sell_score(report: str, score_key: str) -> int:
    """Extract sell score from analyst report."""
    # Try JSON block
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", report, re.DOTALL)
    if not json_match:
        json_match = re.search(r"(\{[^{}]*\"sell_score\"[^{}]*\})", report, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if "sell_score" in data:
                return max(0, min(100, int(data["sell_score"])))
        except Exception:
            pass

    # Try named pattern e.g. "FUNDAMENTAL SELL SCORE: 45"
    for pattern in [
        rf"{re.escape(score_key)}\s*\*{{0,2}}\s*:?\s*\*{{0,2}}\s*(\d+)",
        r"SELL SCORE\s*\*{0,2}\s*:?\s*\*{0,2}\s*(\d+)",
        r"SCORE\s*\*{0,2}\s*:?\s*\*{0,2}\s*(\d+)/100",
        rf"{re.escape(score_key.upper())}[:\s]+(\d+)",
        r"SELL SCORE[:\s]+(\d+)",
        r"SCORE[:\s]+(\d+)/100",
    ]:
        match = re.search(pattern, report, re.IGNORECASE)
        if match:
            try:
                return max(0, min(100, int(match.group(1))))
            except ValueError:
                pass
    return 50  # default when parsing fails


def _parse_sell_verdict(report: str) -> str:
    """Extract HOLD/TRIM/EXIT verdict from analyst report."""
    match = re.search(
        r"\*\*SELL VERDICT:?\*\*\s*:?\s*(HOLD|TRIM|EXIT|URGENT_EXIT)",
        report,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()
    # loose search
    for verdict in ("URGENT_EXIT", "EXIT", "TRIM", "HOLD"):
        if verdict in report.upper():
            return verdict
    return "HOLD"


def _parse_confidence(report: str) -> int:
    match = re.search(r"\*\*CONFIDENCE:?\*\*\s*:?\s*(\d+)", report, re.IGNORECASE)
    if match:
        return max(1, min(10, int(match.group(1))))
    return 5


def _parse_synthesis_json(synthesis: str) -> dict:
    """Extract JSON block from CIO synthesis."""
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", synthesis, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except Exception:
            pass
    # Fallback: try to find any JSON object
    obj_match = re.search(r"\{[^{}]+\"sell_score\"[^{}]+\}", synthesis, re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except Exception:
            pass
    return {}


def _parse_key_reasons(synthesis: str) -> list[str]:
    """Extract bullet points from KEY REASONS section."""
    reasons = []
    in_section = False
    for line in synthesis.split("\n"):
        if "KEY REASONS" in line.upper():
            in_section = True
            continue
        if in_section:
            stripped = line.strip()
            if stripped.startswith("- ") or stripped.startswith("• "):
                reasons.append(stripped.lstrip("- •").strip())
            elif stripped.startswith("**") and stripped.endswith("**"):
                break  # Next section
    return reasons[:5]


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _normalize_action(action: Any) -> str:
    value = str(action or "HOLD").upper().strip()
    return value if value in {"HOLD", "TRIM", "EXIT", "URGENT_EXIT"} else "HOLD"


_SELL_ACTION_RANK = {"HOLD": 0, "TRIM": 1, "EXIT": 2, "URGENT_EXIT": 3}
_EVIDENCE_STOPWORDS = {
    "about", "after", "again", "against", "artha", "because", "being", "could",
    "current", "evidence", "position", "report", "reports", "score", "stock",
    "their", "there", "these", "thesis", "those", "would",
}


def _meaningful_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_]{4,}", str(text or ""))
        if token.lower() not in _EVIDENCE_STOPWORDS
    }


def _listify_evidence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _evidence_is_supported(evidence_items: list[str], source_corpus: str) -> bool:
    """Return True when CIO evidence overlaps materially with source reports/context."""
    if not evidence_items or not source_corpus:
        return False
    corpus_tokens = _meaningful_tokens(source_corpus)
    if not corpus_tokens:
        return False
    for item in evidence_items:
        tokens = _meaningful_tokens(item)
        if len(tokens & corpus_tokens) >= 2:
            return True
    return False


# ---------------------------------------------------------------------------
# SellCouncil
# ---------------------------------------------------------------------------

class SellCouncil:
    """Runs 3-analyst sell debate and computes sell score → action."""

    def __init__(self, journal: Optional[DecisionJournal] = None) -> None:
        self.journal = journal or DecisionJournal()

    def run_sell_review(
        self,
        thesis: Any,
        stock_data: dict,
        macro_data: Optional[dict] = None,
        trigger_type: str = "periodic_review",
        current_regime: str = "unknown",
    ) -> Optional[SellDecision]:
        """Run full sell council review for a position.

        Args:
            thesis: PositionThesis object (or dict with same fields)
            stock_data: Collected stock data from DataCollector
            macro_data: Optional macro data
            trigger_type: Why this review was triggered
            current_regime: Current MROL regime label

        Returns:
            SellDecision or None on catastrophic failure
        """
        ticker = getattr(thesis, "ticker", None) or thesis.get("ticker", "?")
        position_type = getattr(thesis, "position_type", "BUY") or "BUY"
        logger.info(
            "[sell_council] Starting review ticker=%s type=%s trigger=%s",
            ticker,
            position_type,
            trigger_type,
        )

        # Build shared context
        try:
            position_context = build_sell_context(thesis, stock_data, current_regime)
        except Exception as ctx_e:
            logger.warning("[sell_council] Context build failed: %s", ctx_e)
            position_context = f"Position: {ticker} ({position_type})"

        # Run 3 analysts in parallel
        fundamental_report: Optional[str] = None
        technical_report: Optional[str] = None
        contrarian_report: Optional[str] = None

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(_run_sell_fundamental, stock_data, position_context, thesis): "fundamental",
                executor.submit(_run_sell_technical, stock_data, position_context): "technical",
                executor.submit(_run_sell_contrarian, stock_data, position_context): "contrarian",
            }
            # FIX 6: Wrap as_completed in try/except to handle overall timeout gracefully
            try:
                for future in as_completed(futures, timeout=180):
                    name = futures[future]
                    try:
                        result = future.result(timeout=5)
                        if name == "fundamental":
                            fundamental_report = result
                        elif name == "technical":
                            technical_report = result
                        else:
                            contrarian_report = result
                    except Exception as e:
                        logger.error("[sell_council] %s analyst future failed: %s", name, e)
            except (TimeoutError, FuturesTimeout):
                logger.error(
                    "[sell_council] Analyst futures timed out after 180s for %s — "
                    "cancelling remaining, proceeding with completed reports",
                    ticker,
                )
                for future in futures:
                    future.cancel()

        # FIX 7: Count available (non-None) analysts; never substitute score-50 for missing ones
        available_count = sum(
            1 for r in [fundamental_report, technical_report, contrarian_report] if r
        )
        if available_count == 0:
            logger.error("[sell_council] All analysts failed for %s", ticker)
            return None

        if available_count < 2:
            logger.warning(
                "[sell_council] Only %d analyst available for %s — defaulting to HOLD",
                available_count, ticker,
            )
            return SellDecision(
                ticker=ticker,
                position_type=position_type,
                action="HOLD",
                sell_score=50.0,
                thesis_status="INTACT",
                health_score=None,
                fundamental=None,
                technical=None,
                contrarian=None,
                synthesis_report="Insufficient analyst availability — defaulting to HOLD.",
                key_reasons=["Fewer than 2 analysts available; cannot make reliable sell determination."],
                next_review_date=_add_days(self._default_review_days(position_type)),
                trigger_type=trigger_type,
            )

        # Parse scores only for available analysts (None stays None — not 50)
        f_score = _parse_sell_score(fundamental_report, "FUNDAMENTAL SELL SCORE") if fundamental_report else None
        t_score = _parse_sell_score(technical_report, "TECHNICAL SELL SCORE") if technical_report else None
        c_score = _parse_sell_score(contrarian_report, "CONTRARIAN SELL SCORE") if contrarian_report else None

        logger.info(
            "[sell_council] Component scores: F=%s T=%s C=%s ticker=%s",
            f_score if f_score is not None else "N/A",
            t_score if t_score is not None else "N/A",
            c_score if c_score is not None else "N/A",
            ticker,
        )

        base_sell_score = self._compute_base_sell_score(
            fundamental_score=f_score,
            technical_score=t_score,
            contrarian_score=c_score,
        )
        rule_adjustments = self._build_rule_adjustments(
            thesis=thesis,
            stock_data=stock_data,
            current_regime=current_regime,
            analyst_reports=[fundamental_report, technical_report, contrarian_report],
        )
        adjustments = self._build_adjustments_text(base_sell_score, rule_adjustments)

        # CIO synthesis — pass empty string for unavailable reports
        synthesis_prompt = build_sell_synthesis_prompt(
            ticker=ticker,
            position_type=position_type,
            position_context=position_context,
            fundamental_report=fundamental_report or "Analyst unavailable.",
            technical_report=technical_report or "Analyst unavailable.",
            contrarian_report=contrarian_report or "Analyst unavailable.",
            sell_score_adjustments=adjustments,
        )

        try:
            synthesis_raw = ChatGPTBackendClient(timeout=150).chat(synthesis_prompt)
        except Exception as synth_e:
            logger.error("[sell_council] CIO synthesis failed: %s", synth_e)
            synthesis_raw = "Synthesis unavailable."

        # Parse synthesis JSON
        synth_data = _parse_synthesis_json(synthesis_raw)
        rule_score, rule_total, forced_score = self._apply_rule_adjustments(
            base_sell_score,
            rule_adjustments,
        )
        source_corpus = "\n\n".join(
            part
            for part in [position_context, fundamental_report, technical_report, contrarian_report]
            if part
        )
        cio_gate = self._validate_cio_score_adjustment(
            synth_data=synth_data,
            source_corpus=source_corpus,
        )
        sell_score = _clamp_score(rule_score + float(cio_gate["accepted_value"]))

        # CIO model score is logged for audit only; the final score is the
        # deterministic base + code rules + bounded evidence-gated CIO adjustment.
        model_sell_score = synth_data.get("sell_score")
        if model_sell_score is not None:
            _deviation = abs(float(model_sell_score) - sell_score)
            if _deviation > 15:
                logger.warning(
                    "[sell_council] %s: CIO sell_score deviation %.1f pts "
                    "(model=%.1f, final=%.1f) — using audited final score",
                    ticker, _deviation, float(model_sell_score), sell_score,
                )
        cio_action = _normalize_action(synth_data.get("action", "HOLD"))
        thesis_status = synth_data.get("thesis_status", "INTACT").upper()
        health_score = synth_data.get("health_score")
        next_review_days = int(synth_data.get("next_review_days", self._default_review_days(position_type)))
        trim_pct_raw = synth_data.get("trim_pct")
        trim_pct = float(trim_pct_raw) / 100 if trim_pct_raw else None
        confidence = int(synth_data.get("confidence", 5))

        score_action = self._action_from_score(sell_score, position_type)
        action = self._reconcile_cio_action(cio_action, score_action)

        # Validate action against hard portfolio lifecycle gates.
        action = self._validate_action(action, sell_score, position_type, thesis)
        is_urgent = action == "URGENT_EXIT"

        scoring_audit = {
            "base_sell_score": round(base_sell_score, 3),
            "rule_adjustments": rule_adjustments,
            "rule_adjustment_total": round(rule_total, 3),
            "forced_score": forced_score,
            "rule_adjusted_score": round(rule_score, 3),
            "cio_adjustment": cio_gate,
            "final_sell_score": round(sell_score, 3),
            "cio_requested_action": cio_action,
            "score_mapped_action": score_action,
            "final_action": action,
        }

        # Compute next review date
        next_review_date = _add_days(next_review_days)

        key_reasons = _parse_key_reasons(synthesis_raw)

        # FIX 7: Build analyst report objects only for available analysts (None for missing)
        fundamental = SellAnalystReport(
            analyst_name="Fundamental",
            model=Config.GPT_MODEL,
            verdict=_parse_sell_verdict(fundamental_report),
            sell_score=f_score if f_score is not None else 50,
            confidence=_parse_confidence(fundamental_report),
            report=fundamental_report,
        ) if fundamental_report else None
        technical = SellAnalystReport(
            analyst_name="Technical",
            model=Config.GEMINI_TECHNICAL_MODEL,
            verdict=_parse_sell_verdict(technical_report),
            sell_score=t_score if t_score is not None else 50,
            confidence=_parse_confidence(technical_report),
            report=technical_report,
        ) if technical_report else None
        contrarian = SellAnalystReport(
            analyst_name="Contrarian",
            model=Config.GPT_MODEL,
            verdict=_parse_sell_verdict(contrarian_report),
            sell_score=c_score if c_score is not None else 50,
            confidence=_parse_confidence(contrarian_report),
            report=contrarian_report,
        ) if contrarian_report else None

        session_id = str(uuid.uuid4())
        decision = SellDecision(
            ticker=ticker,
            position_type=position_type,
            action=action,
            sell_score=sell_score,
            thesis_status=thesis_status,
            health_score=int(health_score) if health_score is not None else None,
            fundamental=fundamental,
            technical=technical,
            contrarian=contrarian,
            synthesis_report=synthesis_raw,
            key_reasons=key_reasons,
            next_review_date=next_review_date,
            is_urgent=is_urgent,
            trim_pct=trim_pct,
            confidence=confidence,
            trigger_type=trigger_type,
            session_id=session_id,
            base_sell_score=base_sell_score,
            rule_adjustment_total=rule_total,
            cio_adjustment=float(cio_gate["accepted_value"]),
            scoring_audit=scoring_audit,
        )

        try:
            from .sell_dossier import write_sell_dossier
            decision.dossier_path = write_sell_dossier(
                decision=decision,
                thesis=thesis,
                stock_data=stock_data,
                macro_data=macro_data,
                trigger_type=trigger_type,
            )
        except Exception as dossier_e:
            logger.warning("[sell_council] Failed to write sell dossier for %s: %s", ticker, dossier_e)

        # Persist session
        self._persist_session(decision, thesis)

        logger.info(
            "[sell_council] Review complete ticker=%s score=%.0f action=%s thesis=%s",
            ticker,
            sell_score,
            action,
            thesis_status,
        )
        return decision

    def _compute_base_sell_score(
        self,
        fundamental_score: int | None,
        technical_score: int | None,
        contrarian_score: int | None,
    ) -> float:
        """Deterministic weighted score from available analyst components."""
        raw_weights = {"fundamental": 0.4, "technical": 0.3, "contrarian": 0.3}
        raw_scores = {
            "fundamental": fundamental_score,
            "technical": technical_score,
            "contrarian": contrarian_score,
        }
        available = {k: v for k, v in raw_scores.items() if v is not None}
        if not available:
            return 50.0
        total_weight = sum(raw_weights[k] for k in available)
        return _clamp_score(sum(v * raw_weights[k] / total_weight for k, v in available.items()))

    def _current_price(self, stock_data: dict, thesis: Any) -> float:
        quote = stock_data.get("quote") if isinstance(stock_data.get("quote"), dict) else {}
        yf_quote = stock_data.get("yf_quote") if isinstance(stock_data.get("yf_quote"), dict) else {}
        price = (
            _as_float(quote.get("price"))
            or _as_float(yf_quote.get("price"))
            or _as_float(getattr(thesis, "entry_price", None), 0.0)
            or 0.0
        )
        return float(price)

    def _build_rule_adjustments(
        self,
        thesis: Any,
        stock_data: dict,
        current_regime: str,
        analyst_reports: list[str | None],
    ) -> list[dict[str, Any]]:
        """Build deterministic score adjustments that the code will apply."""
        adjustments: list[dict[str, Any]] = []
        position_type = getattr(thesis, "position_type", "BUY") or "BUY"
        ticker = getattr(thesis, "ticker", "") or ""

        current_price = self._current_price(stock_data, thesis)
        hard_stop = _as_float(getattr(thesis, "hard_stop_price", None), 0.0) or 0.0
        if hard_stop > 0 and current_price > 0 and current_price <= hard_stop:
            adjustments.append({
                "name": "hard_stop_breached",
                "value": 0,
                "force_score": 100,
                "reason": (
                    f"{ticker} current price ${current_price:.2f} is at/below "
                    f"hard stop ${hard_stop:.2f}; force urgent exit score."
                ),
            })

        # Conviction position penalty (harder bar for exit)
        if position_type in ("BUY", "ACCUMULATE"):
            adjustments.append({
                "name": "conviction_position_patience",
                "value": Config.SELL_SCORE_CIO_CONVICTION_ADJUST,
                "reason": f"{position_type} is a higher-conviction position; require stronger sell evidence.",
            })

        # Thesis health deterioration bonus
        health = getattr(thesis, "thesis_health_score", 100) or 100
        if health < 50:
            adjustments.append({
                "name": "low_thesis_health",
                "value": 15,
                "reason": f"Thesis health is low at {health}/100.",
            })
        elif health < 70:
            adjustments.append({
                "name": "weakened_thesis_health",
                "value": 7,
                "reason": f"Thesis health is weakened at {health}/100.",
            })

        # Regime change adjustment for TACTICAL_BUY
        entry_regime = str(getattr(thesis, "entry_regime", None) or "unknown").lower()
        current_regime_norm = str(current_regime or "unknown").lower()
        regimes_known = entry_regime not in {"", "unknown"} and current_regime_norm not in {"", "unknown"}
        if regimes_known and current_regime_norm != entry_regime:
            if position_type == "TACTICAL_BUY":
                adjustments.append({
                    "name": "tactical_regime_mismatch",
                    "value": Config.SELL_REGIME_MISMATCH_TACTICAL_BONUS,
                    "reason": f"TACTICAL_BUY entered in {entry_regime}; current regime is {current_regime_norm}.",
                })
            else:
                adjustments.append({
                    "name": "regime_mismatch",
                    "value": Config.SELL_REGIME_MISMATCH_PENALTY,
                    "reason": f"Entry regime was {entry_regime}; current regime is {current_regime_norm}.",
                })

        # Time decay warning
        days_held = getattr(thesis, "days_held", 0) or 0
        time_limits = {
            "TACTICAL_BUY": Config.SELL_TIME_DECAY_TACTICAL,
            "STARTER": Config.SELL_TIME_DECAY_STARTER,
            "BUY": Config.SELL_TIME_DECAY_BUY,
            "ACCUMULATE": Config.SELL_TIME_DECAY_ACCUMULATE,
        }
        time_limit = time_limits.get(position_type, 180)
        if days_held > time_limit * 0.8:
            adjustments.append({
                "name": "time_decay_review_pressure",
                "value": 10,
                "reason": f"Position held {days_held}/{time_limit} days without resolution.",
            })

        combined_reports = "\n\n".join(r for r in analyst_reports if r)
        if self._analyst_reports_show_triggered_invalidation(combined_reports):
            adjustments.append({
                "name": "invalidation_condition_triggered",
                "value": Config.SELL_SCORE_THESIS_TRIGGERED_BONUS,
                "reason": "At least one analyst marked an invalidation condition as TRIGGERED.",
            })

        return adjustments

    def _analyst_reports_show_triggered_invalidation(self, combined_reports: str) -> bool:
        if not combined_reports:
            return False
        patterns = [
            r"\bTRIGGERED\b\s*(?:-|—|:)",
            r"[:\-—]\s*\bTRIGGERED\b",
            r"\bTHESIS STATUS:\s*(?:DAMAGED|BROKEN)\b",
        ]
        return any(re.search(pattern, combined_reports, re.IGNORECASE) for pattern in patterns)

    def _apply_rule_adjustments(
        self,
        base_score: float,
        adjustments: list[dict[str, Any]],
    ) -> tuple[float, float, float | None]:
        forced_scores = [
            float(adj["force_score"])
            for adj in adjustments
            if adj.get("force_score") is not None
        ]
        if forced_scores:
            forced = max(forced_scores)
            return _clamp_score(forced), 0.0, _clamp_score(forced)
        total = sum(float(adj.get("value") or 0.0) for adj in adjustments)
        return _clamp_score(base_score + total), total, None

    def _build_adjustments_text(
        self,
        base_score: float,
        adjustments: list[dict[str, Any]],
    ) -> str:
        """Build audited score adjustment text for the CIO synthesis prompt."""
        lines = [f"Base deterministic weighted score before adjustments: {base_score:.1f}/100"]
        if not adjustments:
            lines.append("Code-applied deterministic adjustments: none.")
        else:
            lines.append("Code-applied deterministic adjustments:")
            for adj in adjustments:
                if adj.get("force_score") is not None:
                    lines.append(
                        f"- {adj.get('name')}: force score to {float(adj['force_score']):.0f}/100 "
                        f"({adj.get('reason')})"
                    )
                else:
                    value = float(adj.get("value") or 0.0)
                    lines.append(f"- {adj.get('name')}: {value:+.0f} ({adj.get('reason')})")
        lines.append(
            "CIO may propose a separate bounded adjustment only with concrete evidence; "
            "code will reject unsupported adjustments."
        )
        return "\n".join(lines)

    def _validate_cio_score_adjustment(
        self,
        synth_data: dict[str, Any],
        source_corpus: str,
    ) -> dict[str, Any]:
        requested = _as_float(synth_data.get("cio_score_adjustment"), 0.0) or 0.0
        max_pos = float(Config.SELL_CIO_ADJUSTMENT_MAX_POSITIVE)
        max_neg = float(Config.SELL_CIO_ADJUSTMENT_MAX_NEGATIVE)
        clipped = max(max_neg, min(max_pos, requested))
        confidence = int(_as_float(synth_data.get("confidence"), 0.0) or 0)
        category = str(synth_data.get("cio_adjustment_category") or "none").strip()[:80]
        reason = str(synth_data.get("cio_adjustment_reason") or "").strip()
        evidence = _listify_evidence(synth_data.get("cio_adjustment_evidence"))

        if abs(requested) < 0.001:
            return {
                "requested_value": 0.0,
                "accepted_value": 0.0,
                "status": "not_requested",
                "category": category or "none",
                "reason": reason or "No CIO adjustment requested.",
                "evidence": evidence,
            }

        if confidence < Config.SELL_CIO_ADJUSTMENT_MIN_CONFIDENCE:
            return {
                "requested_value": requested,
                "accepted_value": 0.0,
                "status": "rejected_low_confidence",
                "category": category,
                "reason": reason or f"CIO confidence {confidence} below adjustment threshold.",
                "evidence": evidence,
            }

        if not reason or not evidence:
            return {
                "requested_value": requested,
                "accepted_value": 0.0,
                "status": "rejected_missing_evidence",
                "category": category,
                "reason": reason or "Non-zero CIO adjustment lacked concrete evidence.",
                "evidence": evidence,
            }

        if not _evidence_is_supported(evidence + [reason], source_corpus):
            return {
                "requested_value": requested,
                "accepted_value": 0.0,
                "status": "rejected_unsupported_evidence",
                "category": category,
                "reason": reason,
                "evidence": evidence,
            }

        status = "accepted"
        if clipped != requested:
            status = "accepted_clipped_to_bounds"
        return {
            "requested_value": requested,
            "accepted_value": clipped,
            "status": status,
            "category": category,
            "reason": reason,
            "evidence": evidence[:5],
            "confidence": confidence,
            "bounds": {"min": max_neg, "max": max_pos},
        }

    def _action_from_score(self, sell_score: float, position_type: str) -> str:
        thresholds = {
            "BUY": Config.SELL_SCORE_EXIT_CONVICTION,
            "ACCUMULATE": Config.SELL_SCORE_EXIT_CONVICTION,
            "TACTICAL_BUY": Config.SELL_SCORE_EXIT_TACTICAL,
            "STARTER": Config.SELL_SCORE_EXIT_STARTER,
            "ADD": Config.SELL_SCORE_EXIT_CONVICTION,
        }
        exit_threshold = thresholds.get(position_type, Config.SELL_SCORE_EXIT_CONVICTION)
        if sell_score >= 90:
            return "URGENT_EXIT"
        if sell_score >= exit_threshold:
            return "EXIT"
        if sell_score >= Config.SELL_SCORE_TRIM_THRESHOLD:
            return "TRIM"
        return "HOLD"

    def _reconcile_cio_action(self, cio_action: str, score_action: str) -> str:
        """Final action follows the audited score; CIO influence is through bounded score adjustment."""
        cio_action = _normalize_action(cio_action)
        score_action = _normalize_action(score_action)
        if cio_action != score_action:
            logger.info(
                "[sell_council] CIO action %s reconciled to score-mapped action %s",
                cio_action,
                score_action,
            )
        return score_action

    def _validate_action(
        self,
        action: str,
        sell_score: float,
        position_type: str,
        thesis: Any,
    ) -> str:
        """Validate and potentially override the CIO's action recommendation."""
        # Hard rules:
        # 1. Minimum hold period must be respected for non-urgent exits
        in_min_hold = getattr(thesis, "in_minimum_hold", False)
        if in_min_hold and action in ("EXIT", "TRIM") and sell_score < 85:
            logger.info("[sell_council] Min hold period active — holding %s to HOLD", action)
            return "HOLD"

        # 2. Cooldown period
        in_cooldown = getattr(thesis, "in_cooldown", False)
        if in_cooldown and action in ("TRIM", "EXIT"):
            logger.info("[sell_council] Cooldown active — holding action to HOLD")
            return "HOLD"

        # 3. Score validation
        thresholds = {
            "BUY": Config.SELL_SCORE_EXIT_CONVICTION,
            "ACCUMULATE": Config.SELL_SCORE_EXIT_CONVICTION,
            "TACTICAL_BUY": Config.SELL_SCORE_EXIT_TACTICAL,
            "STARTER": Config.SELL_SCORE_EXIT_STARTER,
            "ADD": Config.SELL_SCORE_EXIT_BUY if hasattr(Config, "SELL_SCORE_EXIT_BUY") else Config.SELL_SCORE_EXIT_CONVICTION,
        }
        exit_threshold = thresholds.get(position_type, Config.SELL_SCORE_EXIT_CONVICTION)

        if action == "EXIT" and sell_score < exit_threshold:
            logger.info(
                "[sell_council] Score %.0f below EXIT threshold %d for %s — downgrading to TRIM",
                sell_score,
                exit_threshold,
                position_type,
            )
            action = "TRIM" if sell_score >= Config.SELL_SCORE_TRIM_THRESHOLD else "HOLD"

        return action

    def _default_review_days(self, position_type: str) -> int:
        from .config import Config as _C
        return {
            "TACTICAL_BUY": _C.SELL_REVIEW_DAYS_TACTICAL,
            "STARTER": _C.SELL_REVIEW_DAYS_STARTER,
            "BUY": _C.SELL_REVIEW_DAYS_BUY,
            "ACCUMULATE": _C.SELL_REVIEW_DAYS_ACCUMULATE,
        }.get(position_type, 30)

    def _persist_session(self, decision: SellDecision, thesis: Any) -> None:
        """Save sell session to the database."""
        thesis_id = getattr(thesis, "thesis_id", None) or (thesis.get("thesis_id") if hasattr(thesis, "get") else None)
        try:
            self.journal.save_sell_session({
                "session_id": decision.session_id,
                "ticker": decision.ticker,
                "thesis_id": thesis_id,
                "trigger_type": decision.trigger_type,
                "fundamental_verdict": decision.fundamental.verdict if decision.fundamental else None,
                "fundamental_report": (decision.fundamental.report if decision.fundamental else None),
                "technical_verdict": decision.technical.verdict if decision.technical else None,
                "technical_report": (decision.technical.report if decision.technical else None),
                "contrarian_verdict": decision.contrarian.verdict if decision.contrarian else None,
                "contrarian_report": (decision.contrarian.report if decision.contrarian else None),
                "sell_score": decision.sell_score,
                "action": decision.action,
                "synthesis_report": decision.synthesis_report,
                "next_review_date": decision.next_review_date,
                "health_score_after": decision.health_score,
            })
        except Exception as e:
            logger.warning("[sell_council] Failed to persist session: %s", e)

    # -------------------------------------------------------------------------
    # Telegram formatting
    # -------------------------------------------------------------------------

    def format_sell_telegram(self, decision: SellDecision, thesis: Any) -> str:
        """Format sell decision as a Telegram message."""
        action_emoji = {
            "HOLD": "✅",
            "TRIM": "✂️",
            "EXIT": "🔴",
            "URGENT_EXIT": "🚨",
        }.get(decision.action, "📊")

        thesis_emoji = {
            "INTACT": "✅",
            "WEAKENED": "⚠️",
            "DAMAGED": "🟠",
            "BROKEN": "🔴",
        }.get(decision.thesis_status, "❓")

        position_type = decision.position_type
        entry_price = getattr(thesis, "entry_price", None) or 0

        lines = [
            f"{action_emoji} SELL COUNCIL — {decision.ticker}",
            "",
            f"Action: **{decision.action}**",
            f"Sell Score: {decision.sell_score:.0f}/100",
            f"Thesis: {thesis_emoji} {decision.thesis_status}",
            f"Position Type: {position_type}",
        ]
        if decision.scoring_audit:
            base = decision.scoring_audit.get("base_sell_score", decision.base_sell_score)
            rules = decision.scoring_audit.get("rule_adjustment_total", decision.rule_adjustment_total)
            cio = decision.scoring_audit.get("cio_adjustment", {}).get("accepted_value", decision.cio_adjustment)
            lines.append(f"Score audit: base {float(base):.0f} + rules {float(rules):+.0f} + CIO {float(cio):+.0f}")
            cio_gate = decision.scoring_audit.get("cio_adjustment") or {}
            if cio_gate.get("status", "").startswith("rejected"):
                lines.append(f"CIO adjustment rejected: {cio_gate.get('status')}")

        if entry_price:
            lines.append(f"Entry: ${float(entry_price):.2f}")
        if decision.dossier_path:
            lines.append(f"Dossier: {decision.dossier_path}")

        if decision.trim_pct and decision.action == "TRIM":
            lines.append(f"Trim: {decision.trim_pct:.0%} of position")

        lines.append("")
        lines.append("━" * 20)
        lines.append("")
        lines.append("📊 Analyst Scores:")
        if decision.fundamental:
            lines.append(f"  Fundamental: {decision.fundamental.sell_score}/100 ({decision.fundamental.verdict})")
        if decision.technical:
            lines.append(f"  Technical:   {decision.technical.sell_score}/100 ({decision.technical.verdict})")
        if decision.contrarian:
            lines.append(f"  Contrarian:  {decision.contrarian.sell_score}/100 ({decision.contrarian.verdict})")

        if decision.key_reasons:
            lines.append("")
            lines.append("🔑 Key Reasons:")
            for reason in decision.key_reasons[:3]:
                lines.append(f"• {reason}")

        lines.append("")
        lines.append(f"Next Review: {(decision.next_review_date or 'TBD')[:10]}")

        if decision.action == "HOLD":
            lines.append("No action needed. Position continues as planned.")
        elif decision.action == "TRIM":
            lines.append("📋 Review/execute TRIM in Robinhood or manually, then confirm: 'trimmed TICKER'")
        elif decision.action in ("EXIT", "URGENT_EXIT"):
            lines.append("📋 Review/execute SELL in Robinhood or manually, then confirm: 'sold all TICKER'")

        return "\n".join(lines)
