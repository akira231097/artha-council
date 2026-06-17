"""Broker-aware execution officer for Artha buy-side recommendations.

The council decides whether a stock is worth owning. The execution officer
decides whether the idea can be expressed as a safe Robinhood order right now.
The LLM can choose among deterministic candidates, but it cannot expand caps or
override the hard guardrail engine.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .chatgpt_backend import ChatGPTBackendClient
from .config import Config
from .execution import OrderIntent, build_order_intent

logger = logging.getLogger(__name__)


BUY_READY = "BUY_READY"
WAIT_FOR_SAFE_EXECUTION = "WAIT_FOR_SAFE_EXECUTION"
BLOCKED = "BLOCKED"

WHOLE_SHARE_LIMIT = "WHOLE_SHARE_LIMIT"
FRACTIONAL_MARKET = "FRACTIONAL_MARKET"
ENTRY_WATCH = "ENTRY_WATCH"


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace("$", "").replace(",", "").strip())
    except Exception:
        return None


def _decision_value(decision: Any, name: str, default: Any = None) -> Any:
    if isinstance(decision, dict):
        return decision.get(name, default)
    return getattr(decision, name, default)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidates = [match.group(1)] if match else []
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        candidates.append(stripped[first : last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            continue
    return None


def _json_safe(value: Any, *, max_chars: int = 12000) -> Any:
    """Return a JSON-safe, bounded representation for agent traces/prompts."""
    try:
        text = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except Exception:
        text = str(value)
    if len(text) <= max_chars:
        try:
            return json.loads(text)
        except Exception:
            return text
    truncated = text[:max_chars]
    return {"truncated": True, "max_chars": max_chars, "text": truncated}


def _as_dict_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _spread_pct(market_data: dict[str, Any]) -> float | None:
    bid = _as_float(market_data.get("bid") or market_data.get("bid_price"))
    ask = _as_float(market_data.get("ask") or market_data.get("ask_price"))
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
        return None
    return (ask - bid) / ((ask + bid) / 2)


@dataclass
class ExecutionCandidate:
    candidate_id: str
    strategy: str
    verdict: str
    allowed: bool
    quantity: float | None = None
    notional: float | None = None
    limit_price: float | None = None
    order_type: str | None = None
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionOfficerPlan:
    ticker: str
    execution_verdict: str
    strategy: str
    selected_candidate_id: str
    auto_buy_eligible: bool
    quantity: float | None
    notional: float | None
    limit_price: float | None
    reference_price: float | None
    no_chase_cap: float | None
    reasons: list[str]
    checks: dict[str, Any]
    officer_model: str
    officer_reasoning_effort: str
    officer_temperature: float
    officer_used: bool = False
    officer_raw: str = ""
    officer_json: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "execution_verdict": self.execution_verdict,
            "strategy": self.strategy,
            "selected_candidate_id": self.selected_candidate_id,
            "auto_buy_eligible": self.auto_buy_eligible,
            "quantity": self.quantity,
            "notional": self.notional,
            "limit_price": self.limit_price,
            "reference_price": self.reference_price,
            "no_chase_cap": self.no_chase_cap,
            "reasons": self.reasons,
            "checks": self.checks,
            "officer_model": self.officer_model,
            "officer_reasoning_effort": self.officer_reasoning_effort,
            "officer_temperature": self.officer_temperature,
            "officer_used": self.officer_used,
            "officer_json": self.officer_json,
        }

    def build_order_intent(
        self,
        *,
        decision_dossier_path: str = "",
        rationale: str = "",
        dry_run: bool = True,
    ) -> OrderIntent | None:
        if self.execution_verdict != BUY_READY:
            return None
        if self.strategy not in {WHOLE_SHARE_LIMIT, FRACTIONAL_MARKET}:
            return None
        return build_order_intent(
            self.ticker,
            "buy",
            notional=self.notional,
            quantity=self.quantity,
            limit_price=self.limit_price,
            estimated_price=self.reference_price or self.limit_price,
            decision_dossier_path=decision_dossier_path,
            rationale=rationale,
            dry_run=dry_run,
        )


def _decision_packet(decision: Any) -> dict[str, Any]:
    return {
        "ticker": _decision_value(decision, "ticker"),
        "final_verdict": _decision_value(decision, "final_verdict"),
        "opportunity_score": _decision_value(decision, "opportunity_score"),
        "adjusted_score": _decision_value(decision, "adjusted_score"),
        "confidence": _decision_value(decision, "confidence"),
        "recommended_allocation_pct": _decision_value(decision, "recommended_allocation_pct"),
        "recommended_action": str(_decision_value(decision, "recommended_action", "") or "")[:1600],
        "dossier_path": _decision_value(decision, "dossier_path"),
        "entry_valid_until": _decision_value(decision, "entry_valid_until"),
        "invalidation_conditions": _decision_value(decision, "invalidation_conditions", []),
    }


def _deterministic_candidates(
    *,
    ticker: str,
    decision: Any,
    recommended_notional: float,
    reference_price: float,
    current_price: float | None,
    market_data: dict[str, Any],
) -> tuple[list[ExecutionCandidate], dict[str, Any]]:
    verdict = str(_decision_value(decision, "final_verdict", "") or "").upper().strip()
    score = int(_as_float(_decision_value(decision, "adjusted_score")) or _as_float(_decision_value(decision, "opportunity_score")) or 0)
    confidence = int(_as_float(_decision_value(decision, "confidence")) or 0)
    capped_notional = round(
        min(
            float(recommended_notional),
            float(Config.ROBINHOOD_AUTO_BUY_MAX_ORDER_DOLLARS),
            float(Config.ROBINHOOD_MAX_POSITION_DOLLARS),
            float(Config.ROBINHOOD_PILOT_MAX_ACCOUNT_VALUE),
        ),
        2,
    )
    no_chase_cap = round(reference_price * (1 + max(0.0, Config.ROBINHOOD_AUTO_BUY_NO_CHASE_PCT)), 2)
    bid = _as_float(market_data.get("bid") or market_data.get("bid_price"))
    ask = _as_float(market_data.get("ask") or market_data.get("ask_price"))
    spread = _spread_pct(market_data)
    hard_reasons: list[str] = []
    if verdict not in {"STARTER", "TACTICAL_BUY", "BUY", "ACCUMULATE"}:
        hard_reasons.append(f"Verdict {verdict or 'UNKNOWN'} is not a buy-side execution verdict.")
    if capped_notional <= 0:
        hard_reasons.append("Recommended notional is not positive.")
    if reference_price <= 0:
        hard_reasons.append("Reference price is not positive.")
    spread_reason = ""
    if spread is not None and spread > Config.ROBINHOOD_MAX_SPREAD_PCT:
        spread_reason = f"Spread {spread:.2%} is wider than {Config.ROBINHOOD_MAX_SPREAD_PCT:.2%}."

    candidates: list[ExecutionCandidate] = []
    if hard_reasons:
        candidates.append(
            ExecutionCandidate(
                candidate_id="blocked_by_hard_gates",
                strategy=ENTRY_WATCH,
                verdict=BLOCKED,
                allowed=False,
                reasons=hard_reasons,
            )
        )
        return candidates, {
            "score": score,
            "confidence": confidence,
            "verdict": verdict,
            "auto_buy_verdict_allowed": verdict in set(Config.ROBINHOOD_AUTO_BUY_ALLOWED_VERDICTS),
            "auto_buy_score_allowed": score >= Config.ROBINHOOD_AUTO_BUY_MIN_SCORE,
            "auto_buy_confidence_allowed": confidence >= Config.ROBINHOOD_AUTO_BUY_MIN_CONFIDENCE,
            "recommended_notional": recommended_notional,
            "capped_notional": capped_notional,
            "reference_price": reference_price,
            "no_chase_cap": no_chase_cap,
            "bid": bid,
            "ask": ask,
            "spread_pct": spread,
        }

    whole_qty = int(math.floor(capped_notional / no_chase_cap)) if no_chase_cap > 0 else 0
    whole_notional = round(whole_qty * no_chase_cap, 2) if whole_qty > 0 else 0.0
    fill_ratio = whole_notional / capped_notional if capped_notional > 0 else 0.0
    whole_reasons: list[str] = []
    if spread_reason:
        whole_reasons.append(spread_reason)
    if whole_qty < 1:
        whole_reasons.append("One whole share exceeds the capped starter budget.")
    if fill_ratio < Config.ROBINHOOD_AUTO_BUY_MIN_WHOLE_SHARE_FILL_RATIO:
        whole_reasons.append(
            f"Whole-share order would use only {fill_ratio:.0%} of intended notional; below "
            f"{Config.ROBINHOOD_AUTO_BUY_MIN_WHOLE_SHARE_FILL_RATIO:.0%} minimum."
        )
    if ask is not None and ask > no_chase_cap:
        whole_reasons.append(f"Live ask ${ask:.2f} is above no-chase cap ${no_chase_cap:.2f}.")
    candidates.append(
        ExecutionCandidate(
            candidate_id="whole_share_marketable_limit",
            strategy=WHOLE_SHARE_LIMIT,
            verdict=BUY_READY if not whole_reasons else WAIT_FOR_SAFE_EXECUTION,
            allowed=not whole_reasons,
            quantity=float(whole_qty) if whole_qty >= 1 else None,
            notional=whole_notional if whole_qty >= 1 else None,
            limit_price=no_chase_cap if whole_qty >= 1 else None,
            order_type="limit",
            reasons=whole_reasons,
            checks={"fill_ratio": round(fill_ratio, 4), "whole_qty": whole_qty},
        )
    )

    fractional_qty = capped_notional / reference_price if reference_price > 0 else 0.0
    fractional_reasons: list[str] = []
    if spread_reason:
        fractional_reasons.append(spread_reason)
    if fractional_qty <= 0:
        fractional_reasons.append("Fractional quantity could not be resolved.")
    if ask is not None and ask > reference_price * (1 + Config.ROBINHOOD_MARKET_ORDER_MAX_PRICE_DRIFT_PCT):
        fractional_reasons.append(
            f"Live ask ${ask:.2f} is above fractional market reference ${reference_price:.2f} by more than "
            f"{Config.ROBINHOOD_MARKET_ORDER_MAX_PRICE_DRIFT_PCT:.2%}."
        )
    candidates.append(
        ExecutionCandidate(
            candidate_id="fractional_market_notional",
            strategy=FRACTIONAL_MARKET,
            verdict=BUY_READY if not fractional_reasons else WAIT_FOR_SAFE_EXECUTION,
            allowed=not fractional_reasons,
            quantity=round(fractional_qty, 6) if fractional_qty > 0 else None,
            notional=capped_notional,
            limit_price=round(reference_price, 2),
            order_type="market",
            reasons=fractional_reasons,
            checks={"fractional_qty": round(fractional_qty, 6)},
        )
    )
    candidates.append(
        ExecutionCandidate(
            candidate_id="wait_for_safe_execution",
            strategy=ENTRY_WATCH,
            verdict=WAIT_FOR_SAFE_EXECUTION,
            allowed=True,
            reasons=["Wait for a safe spread, price, or executable broker order shape."],
        )
    )
    checks = {
        "score": score,
        "confidence": confidence,
        "verdict": verdict,
        "auto_buy_verdict_allowed": verdict in set(Config.ROBINHOOD_AUTO_BUY_ALLOWED_VERDICTS),
        "auto_buy_score_allowed": score >= Config.ROBINHOOD_AUTO_BUY_MIN_SCORE,
        "auto_buy_confidence_allowed": confidence >= Config.ROBINHOOD_AUTO_BUY_MIN_CONFIDENCE,
        "recommended_notional": recommended_notional,
        "capped_notional": capped_notional,
        "reference_price": reference_price,
        "current_price": current_price,
        "no_chase_cap": no_chase_cap,
        "bid": bid,
        "ask": ask,
        "spread_pct": spread,
        "whole_qty": whole_qty,
        "whole_fill_ratio": round(fill_ratio, 4),
        "fractional_qty": round(fractional_qty, 6),
    }
    return candidates, checks


def _default_candidate(candidates: list[ExecutionCandidate]) -> ExecutionCandidate:
    for candidate in candidates:
        if candidate.allowed and candidate.verdict == BUY_READY and candidate.strategy == WHOLE_SHARE_LIMIT:
            return candidate
    for candidate in candidates:
        if candidate.allowed and candidate.verdict == BUY_READY and candidate.strategy == FRACTIONAL_MARKET:
            return candidate
    for candidate in candidates:
        if candidate.allowed and candidate.verdict == WAIT_FOR_SAFE_EXECUTION:
            return candidate
    return candidates[0]


def build_execution_officer_prompt(
    *,
    ticker: str,
    decision: Any,
    market_data: dict[str, Any],
    deterministic_checks: dict[str, Any],
    candidates: list[ExecutionCandidate],
) -> str:
    payload = {
        "ticker": ticker,
        "role": "Execution Officer",
        "objective": "Choose the safest executable Robinhood order path for an already-approved Artha buy-side idea.",
        "source_hierarchy": [
            "Robinhood review/tradability and current broker state are the execution source of truth.",
            "Structured provider prices/fundamentals from FMP, Massive, yfinance, Finnhub, and SEC are hard-data context.",
            "Web/news/search context can add risks but must not override structured market data or broker checks.",
        ],
        "non_negotiable_rules": [
            "Do not increase notional, quantity, or limit price beyond the deterministic candidates.",
            "Prefer marketable whole-share limit orders when one or more shares fit the starter budget and ask is within the no-chase cap.",
            "Fractional/dollar orders must be market/regular-hours only and must pass live quote drift plus Robinhood review.",
            "If required data is missing or conflicting, choose WAIT_FOR_SAFE_EXECUTION and list the missing data.",
            "Return JSON only.",
        ],
        "decision": _decision_packet(decision),
        "market_data": market_data,
        "deterministic_checks": deterministic_checks,
        "candidates": [candidate.to_dict() for candidate in candidates],
        "required_schema": {
            "selected_candidate_id": "one candidate_id from candidates",
            "execution_verdict": "BUY_READY | WAIT_FOR_SAFE_EXECUTION | BLOCKED",
            "confidence": "integer 1-10",
            "rationale": "short explanation grounded in supplied evidence",
            "requested_data": ["missing data needed before execution, if any"],
            "risk_flags": ["execution risks"],
        },
    }
    return (
        "You are Artha's GPT-5.5 Execution Officer. Think carefully, but output only JSON.\n"
        "Your job is execution quality, not investment thesis generation. Choose exactly one deterministic candidate.\n\n"
        f"{json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2)}"
    )


def build_execution_officer_plan(
    *,
    ticker: str,
    decision: Any,
    recommended_notional: float,
    reference_price: float,
    current_price: float | None,
    market_data: dict[str, Any],
    use_llm: bool | None = None,
) -> ExecutionOfficerPlan:
    """Return the audited execution plan for a buy-side council decision."""
    ticker = str(ticker or "").upper().strip()
    market_data = market_data or {}
    candidates, checks = _deterministic_candidates(
        ticker=ticker,
        decision=decision,
        recommended_notional=float(recommended_notional or 0),
        reference_price=float(reference_price or 0),
        current_price=current_price,
        market_data=market_data,
    )
    selected = _default_candidate(candidates)
    officer_raw = ""
    officer_json: dict[str, Any] = {}
    officer_used = False
    llm_enabled = Config.EXECUTION_OFFICER_ENABLED and (
        Config.EXECUTION_OFFICER_LLM_ENABLED if use_llm is None else bool(use_llm)
    )
    if llm_enabled:
        prompt = build_execution_officer_prompt(
            ticker=ticker,
            decision=decision,
            market_data=market_data,
            deterministic_checks=checks,
            candidates=candidates,
        )
        try:
            officer_raw = ChatGPTBackendClient(
                model=Config.EXECUTION_OFFICER_MODEL,
                reasoning_effort=Config.EXECUTION_OFFICER_REASONING_EFFORT,
                temperature=Config.EXECUTION_OFFICER_TEMPERATURE,
                timeout=Config.EXECUTION_OFFICER_TIMEOUT_SECONDS,
            ).chat(prompt)
            parsed = _extract_json_object(officer_raw) or {}
            if parsed:
                by_id = {candidate.candidate_id: candidate for candidate in candidates}
                requested = str(parsed.get("selected_candidate_id") or "")
                candidate = by_id.get(requested)
                if candidate and (candidate.allowed or candidate.verdict != BUY_READY):
                    selected = candidate
                    officer_json = parsed
                    officer_used = True
                else:
                    officer_json = {
                        **parsed,
                        "rejected_reason": "LLM selected a missing or disallowed BUY_READY candidate; deterministic default retained.",
                    }
        except Exception as exc:
            logger.warning("[execution_officer] LLM failed for %s; using deterministic plan: %s", ticker, exc)
            officer_json = {"error": f"{type(exc).__name__}: {exc}"}

    if selected.verdict == BUY_READY and selected.strategy in {WHOLE_SHARE_LIMIT, FRACTIONAL_MARKET}:
        execution_verdict = BUY_READY
    elif selected.verdict == BLOCKED:
        execution_verdict = BLOCKED
    else:
        execution_verdict = WAIT_FOR_SAFE_EXECUTION
    auto_buy_eligible = bool(
        Config.ROBINHOOD_AUTO_BUY_ENABLED
        and execution_verdict == BUY_READY
        and selected.strategy in {WHOLE_SHARE_LIMIT, FRACTIONAL_MARKET}
        and checks.get("auto_buy_verdict_allowed")
        and checks.get("auto_buy_score_allowed")
        and checks.get("auto_buy_confidence_allowed")
    )
    reasons = list(selected.reasons)
    if officer_json.get("rationale"):
        reasons.append(f"Execution Officer: {officer_json.get('rationale')}")
    if officer_json.get("requested_data"):
        reasons.append(f"Execution Officer requested data: {officer_json.get('requested_data')}")
    return ExecutionOfficerPlan(
        ticker=ticker,
        execution_verdict=execution_verdict,
        strategy=selected.strategy,
        selected_candidate_id=selected.candidate_id,
        auto_buy_eligible=auto_buy_eligible,
        quantity=selected.quantity,
        notional=selected.notional,
        limit_price=selected.limit_price,
        reference_price=current_price or reference_price,
        no_chase_cap=checks.get("no_chase_cap"),
        reasons=reasons,
        checks={**checks, "candidates": [candidate.to_dict() for candidate in candidates]},
        officer_model=Config.EXECUTION_OFFICER_MODEL,
        officer_reasoning_effort=Config.EXECUTION_OFFICER_REASONING_EFFORT,
        officer_temperature=Config.EXECUTION_OFFICER_TEMPERATURE,
        officer_used=officer_used,
        officer_raw=officer_raw,
        officer_json=officer_json,
    )


def _action_tool_context(action: dict[str, Any], operation: dict[str, Any]) -> dict[str, Any]:
    payload = _as_dict_json(action.get("payload_json"))
    intent = payload.get("intent") if isinstance(payload.get("intent"), dict) else {}
    return {
        "action": {
            "action_id": action.get("action_id"),
            "ticker": action.get("ticker"),
            "side": action.get("side"),
            "action_type": action.get("action_type"),
            "status": action.get("status"),
            "expires_at": action.get("expires_at"),
            "order_intent_id": action.get("order_intent_id"),
            "execution_order_row": action.get("execution_order_row"),
            "message": action.get("message"),
            "notes": action.get("notes"),
        },
        "intent": _json_safe(intent, max_chars=9000),
        "operation": {
            "operation": operation.get("operation"),
            "action_id": operation.get("action_id"),
            "review_mcp_args": operation.get("review_mcp_args"),
            "tradability_mcp_args": operation.get("tradability_mcp_args"),
            "auto_buy_gate": operation.get("auto_buy_gate"),
        },
    }


def _dossier_path_from_action(action: dict[str, Any]) -> str:
    payload = _as_dict_json(action.get("payload_json"))
    intent = payload.get("intent") if isinstance(payload.get("intent"), dict) else {}
    for candidate in (
        intent.get("decision_dossier_path"),
        ((intent.get("evidence") or {}) if isinstance(intent.get("evidence"), dict) else {}).get("dossier_path"),
    ):
        if candidate:
            return str(candidate)
    return ""


def _read_decision_dossier_for_agent(action: dict[str, Any], *, max_chars: int = 16000) -> dict[str, Any]:
    path = _dossier_path_from_action(action)
    if not path:
        return {"available": False, "reason": "No decision dossier path was stored on the action."}
    target = Path(path).expanduser()
    if not target.exists():
        return {"available": False, "path": str(target), "reason": "Decision dossier file does not exist."}
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
        parsed = json.loads(text)
        return {"available": True, "path": str(target), "dossier": _json_safe(parsed, max_chars=max_chars)}
    except Exception as exc:
        return {"available": False, "path": str(target), "reason": f"{type(exc).__name__}: {exc}"}


def build_agentic_execution_prompt(
    *,
    action: dict[str, Any],
    operation: dict[str, Any],
    tool_trace: list[dict[str, Any]],
    step: int,
) -> str:
    payload = {
        "role": "Artha GPT-5.5 Agentic Execution Officer",
        "objective": (
            "Use available tools to gather enough evidence to decide whether this queued auto-buy can be placed now. "
            "You are responsible for execution quality, not changing the investment thesis."
        ),
        "source_hierarchy": [
            "Robinhood live quote, tradability, and review are the execution source of truth.",
            "Artha deterministic gate, order caps, no-chase caps, stale-snapshot checks, and daily caps are non-negotiable.",
            "Artha decision dossier and provider data explain the thesis and context, but cannot override broker safety checks.",
            "Web/news context in the dossier is context only; do not use it to bypass broker or price guardrails.",
        ],
        "non_negotiable_rules": [
            "You may request tools, then either request another tool or return a final_decision.",
            "Before final_decision allow_place=true, you must have fresh tool results from robinhood_get_quote, robinhood_get_tradability, and robinhood_review_order in this trace.",
            "If any required Robinhood tool is missing, request the next missing tool instead of returning final_decision.",
            "Do not modify order parameters. Do not increase size, notional, or limit price.",
            "If Robinhood review_gate is not PASS, allow_place must be false.",
            "If broker order checks are classified blocking, unknown, or ambiguous, allow_place must be false.",
            "If the order is outside market/order-session constraints, allow_place must be false.",
            "If evidence is missing or conflicting after the tool budget, allow_place must be false.",
            "Return JSON only. Do not expose chain-of-thought; put concise evidence-grounded reasoning in rationale.",
        ],
        "recommended_tool_sequence": [
            "read_artha_action",
            "read_decision_dossier",
            "web_news_context",
            "provider_market_context",
            "robinhood_get_quote",
            "robinhood_get_tradability",
            "robinhood_review_order",
            "final_decision",
        ],
        "required_tools_missing": [
            name
            for name in ("robinhood_get_quote", "robinhood_get_tradability", "robinhood_review_order")
            if name not in _tool_names_seen(tool_trace)
        ],
        "available_tools": {
            "read_artha_action": "Return the queued Artha action, intent, deterministic execution plan, and authorization gate.",
            "read_decision_dossier": "Return a bounded copy of the decision dossier attached to the action.",
            "read_robinhood_snapshot": "Return Artha's latest imported Robinhood account snapshot.",
            "provider_market_context": "Collect current structured provider quote context from FMP/yfinance/Massive.",
            "web_news_context": "Search current web/news context for urgent ticker-specific execution risks.",
            "robinhood_get_quote": "Fetch current Robinhood quote for the ticker.",
            "robinhood_get_tradability": "Check Robinhood tradability/fractional eligibility for this account and ticker.",
            "robinhood_review_order": "Run Robinhood review_equity_order for the exact queued order and record Artha's deterministic review gate.",
        },
        "tool_request_schema": {
            "tool_name": "one available tool name",
            "args": "object; usually empty because Artha supplies exact action/order args",
            "reason": "short reason why this evidence is needed",
        },
        "final_decision_schema": {
            "final_decision": {
                "allow_place": "boolean",
                "confidence": "integer 1-10",
                "order_unchanged": "boolean",
                "rationale": "short evidence-grounded explanation",
                "evidence_refs": ["tool names or trace ids used"],
                "risk_flags": ["remaining risks or broker alerts"],
                "missing_data": ["data still missing, if any"],
            }
        },
        "current_step": step,
        "max_tool_steps": Config.EXECUTION_OFFICER_AGENTIC_MAX_TOOL_STEPS,
        "context": _action_tool_context(action, operation),
        "tool_trace": _json_safe(tool_trace, max_chars=28000),
    }
    return (
        "You are Artha's agentic Execution Officer. Use private reasoning and output exactly one JSON object.\n"
        "Either request one tool or return final_decision. The final decision can permit a real-money Robinhood order only if all mandatory evidence is present and clean.\n\n"
        f"{json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2)}"
    )


def _tool_names_seen(tool_trace: list[dict[str, Any]]) -> set[str]:
    return {str(item.get("tool_name") or "") for item in tool_trace if item.get("status") == "PASS"}


def _latest_tool_result(tool_trace: list[dict[str, Any]], tool_name: str) -> dict[str, Any] | None:
    for item in reversed(tool_trace):
        if item.get("tool_name") == tool_name:
            return item
    return None


def _provider_market_context(ticker: str) -> dict[str, Any]:
    try:
        from .collector import DataCollector

        collector = DataCollector()
        return {
            "ticker": ticker,
            "fmp_quote": _json_safe(collector.fmp.quote(ticker), max_chars=4000),
            "yfinance_quote": _json_safe(collector.yf.quote(ticker), max_chars=4000),
            "massive_quote": _json_safe(collector.massive.quote(ticker), max_chars=4000),
        }
    except Exception as exc:
        return {"ticker": ticker, "error": f"{type(exc).__name__}: {exc}"}


def _web_news_context(ticker: str) -> dict[str, Any]:
    try:
        from .search import search_web

        query = (
            f"{ticker} stock breaking news analyst downgrade earnings lawsuit FDA halt acquisition "
            "site:reuters.com OR site:sec.gov OR site:investors.com OR site:benzinga.com OR site:yahoo.com"
        )
        results = search_web(query, count=5, freshness="day")
        return {
            "ticker": ticker,
            "query": query,
            "results": _json_safe(results, max_chars=9000),
            "source_priority_note": (
                "Use web/news as time-sensitive context only. It may block or request caution, "
                "but it cannot override Robinhood quote/tradability/review gates."
            ),
        }
    except Exception as exc:
        return {"ticker": ticker, "error": f"{type(exc).__name__}: {exc}"}


def _merge_agent_trace_into_action(
    *,
    action_id: str,
    journal: Any,
    trace: dict[str, Any],
) -> None:
    try:
        row = journal.get_trade_action(action_id) or {}
        result = _as_dict_json(row.get("result_json"))
        result["agentic_execution_officer"] = _json_safe(trace, max_chars=24000)
        journal.update_trade_action(action_id, {"result_json": result})
    except Exception as exc:
        logger.warning("[execution_officer] failed to persist agentic trace for %s: %s", action_id, exc)


def _validate_agentic_final_decision(
    parsed: dict[str, Any],
    tool_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    final = parsed.get("final_decision") if isinstance(parsed.get("final_decision"), dict) else parsed
    allow_requested = bool(final.get("allow_place"))
    confidence = int(_as_float(final.get("confidence")) or 0)
    seen = _tool_names_seen(tool_trace)
    missing_required = [
        name
        for name in ("robinhood_get_quote", "robinhood_get_tradability", "robinhood_review_order")
        if name not in seen
    ]
    reasons: list[str] = []
    if missing_required:
        reasons.append(f"Agentic Execution Officer did not collect required live tools: {', '.join(missing_required)}.")
    if confidence < Config.ROBINHOOD_AUTO_BUY_MIN_CONFIDENCE:
        reasons.append(
            f"Agentic Execution Officer confidence {confidence} is below {Config.ROBINHOOD_AUTO_BUY_MIN_CONFIDENCE}."
        )
    if not bool(final.get("order_unchanged")):
        reasons.append("Agentic Execution Officer did not affirm that order parameters are unchanged.")
    review_tool = _latest_tool_result(tool_trace, "robinhood_review_order")
    recorded = ((review_tool or {}).get("result") or {}).get("recorded_review") if isinstance((review_tool or {}).get("result"), dict) else {}
    review_gate = recorded.get("review_gate") if isinstance(recorded, dict) else {}
    if not isinstance(review_gate, dict):
        review_gate = {}
    if not review_gate.get("passed"):
        reasons.append("Recorded Robinhood review gate is not PASS.")
    return {
        "allow_place": bool(allow_requested and not reasons),
        "status": "PASS" if allow_requested and not reasons else "BLOCKED",
        "reasons": reasons,
        "officer_json": final,
        "required_tools_seen": sorted(seen),
        "review_gate": review_gate,
    }


def run_agentic_execution_officer(
    *,
    action: dict[str, Any],
    operation: dict[str, Any],
    broker: Any,
    journal: Any,
    use_llm: bool | None = None,
) -> dict[str, Any]:
    """Run a tool-using execution session before real-money auto-buy placement."""
    action_id = str(action.get("action_id") or operation.get("action_id") or "")
    ticker = str(action.get("ticker") or (operation.get("review_mcp_args") or {}).get("symbol") or "").upper()
    llm_enabled = Config.EXECUTION_OFFICER_ENABLED and Config.EXECUTION_OFFICER_AGENTIC_ENABLED and (
        Config.EXECUTION_OFFICER_LLM_ENABLED if use_llm is None else bool(use_llm)
    )
    if not llm_enabled:
        return {
            "allow_place": None,
            "status": "SKIPPED",
            "reason": "Agentic Execution Officer disabled; legacy deterministic path should run.",
            "tool_trace": [],
        }
    if not operation.get("success") or not isinstance(operation.get("review_mcp_args"), dict) or not isinstance(
        operation.get("tradability_mcp_args"), dict
    ):
        trace = {
            "status": "BLOCKED",
            "reason": "Agentic Execution Officer received an invalid or incomplete Robinhood operation.",
            "operation_summary": {
                "success": operation.get("success"),
                "operation": operation.get("operation"),
                "has_review_args": isinstance(operation.get("review_mcp_args"), dict),
                "has_tradability_args": isinstance(operation.get("tradability_mcp_args"), dict),
            },
            "tool_trace": [],
        }
        if action_id:
            _merge_agent_trace_into_action(action_id=action_id, journal=journal, trace=trace)
        return {"allow_place": False, **trace}

    tool_trace: list[dict[str, Any]] = []
    last_tradability: dict[str, Any] | None = None
    step_raws: list[str] = []

    def run_tool(tool_name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        nonlocal last_tradability
        args = args or {}
        if tool_name == "read_artha_action":
            return _action_tool_context(action, operation)
        if tool_name == "read_decision_dossier":
            return _read_decision_dossier_for_agent(action)
        if tool_name == "read_robinhood_snapshot":
            from .robinhood_bridge import load_robinhood_snapshot

            return _json_safe(load_robinhood_snapshot(), max_chars=12000)
        if tool_name == "provider_market_context":
            return _provider_market_context(ticker)
        if tool_name == "web_news_context":
            return _web_news_context(ticker)
        if tool_name == "robinhood_get_quote":
            if not hasattr(broker, "get_equity_quotes"):
                return {"error": "Broker client does not expose get_equity_quotes."}
            return broker.get_equity_quotes(symbols=[ticker])
        if tool_name == "robinhood_get_tradability":
            last_tradability = broker.get_equity_tradability(**operation["tradability_mcp_args"])
            return last_tradability
        if tool_name == "robinhood_review_order":
            if last_tradability is None:
                return {"error": "Call robinhood_get_tradability before robinhood_review_order."}
            from .robinhood_bridge import record_action_review

            review = broker.review_equity_order(**operation["review_mcp_args"])
            recorded = record_action_review(
                action_id,
                review,
                tradability_response=last_tradability,
                journal=journal,
            )
            return {
                "review_response": _json_safe(review, max_chars=12000),
                "recorded_review": _json_safe(recorded, max_chars=12000),
            }
        return {"error": f"Unknown tool: {tool_name}"}

    for step in range(1, max(1, Config.EXECUTION_OFFICER_AGENTIC_MAX_TOOL_STEPS) + 1):
        prompt = build_agentic_execution_prompt(
            action=action,
            operation=operation,
            tool_trace=tool_trace,
            step=step,
        )
        try:
            raw = ChatGPTBackendClient(
                model=Config.EXECUTION_OFFICER_MODEL,
                reasoning_effort=Config.EXECUTION_OFFICER_REASONING_EFFORT,
                temperature=Config.EXECUTION_OFFICER_TEMPERATURE,
                timeout=Config.EXECUTION_OFFICER_TIMEOUT_SECONDS,
            ).chat(prompt)
            step_raws.append(raw)
            parsed = _extract_json_object(raw) or {}
        except Exception as exc:
            trace = {
                "status": "BLOCKED",
                "reason": f"Agentic model call failed: {type(exc).__name__}: {exc}",
                "tool_trace": tool_trace,
                "raw_steps": step_raws[-3:],
            }
            if action_id:
                _merge_agent_trace_into_action(action_id=action_id, journal=journal, trace=trace)
            return {"allow_place": False, **trace}

        if isinstance(parsed.get("final_decision"), dict) or "allow_place" in parsed:
            validation = _validate_agentic_final_decision(parsed, tool_trace)
            if (
                validation["status"] != "PASS"
                and validation.get("required_tools_seen") is not None
                and step < max(1, Config.EXECUTION_OFFICER_AGENTIC_MAX_TOOL_STEPS)
                and any("required live tools" in str(reason) for reason in validation.get("reasons") or [])
            ):
                tool_trace.append(
                    {
                        "step": step,
                        "tool_name": "invalid_final_decision",
                        "status": "FAIL",
                        "reason": "Model returned final_decision before required live tools; continue tool collection.",
                        "result": _json_safe(validation, max_chars=8000),
                    }
                )
                continue
            trace = {
                "status": validation["status"],
                "reason": "; ".join(validation["reasons"]) or str(validation["officer_json"].get("rationale") or "Agentic final decision."),
                "tool_trace": tool_trace,
                "validation": validation,
                "raw_steps": step_raws[-3:],
            }
            if action_id:
                _merge_agent_trace_into_action(action_id=action_id, journal=journal, trace=trace)
            return {
                "allow_place": validation["allow_place"],
                "execution_officer_final_clearance": validation,
                **trace,
            }

        tool_name = str(parsed.get("tool_name") or "").strip()
        if not tool_name:
            tool_result = {"error": "Model returned neither a tool request nor final_decision."}
            tool_name = "invalid_model_output"
            status = "FAIL"
        else:
            tool_result = run_tool(tool_name, parsed.get("args") if isinstance(parsed.get("args"), dict) else {})
            status = "FAIL" if isinstance(tool_result, dict) and tool_result.get("error") else "PASS"
        tool_trace.append(
            {
                "step": step,
                "tool_name": tool_name,
                "status": status,
                "reason": str(parsed.get("reason") or ""),
                "result": _json_safe(tool_result, max_chars=14000),
            }
        )

    trace = {
        "status": "BLOCKED",
        "reason": "Agentic Execution Officer exhausted tool budget without a valid final decision.",
        "tool_trace": tool_trace,
        "raw_steps": step_raws[-3:],
    }
    if action_id:
        _merge_agent_trace_into_action(action_id=action_id, journal=journal, trace=trace)
    return {"allow_place": False, **trace}


def build_robinhood_review_clearance_prompt(
    *,
    action: dict[str, Any],
    review_response: dict[str, Any],
    tradability_response: dict[str, Any] | None,
    recorded_review: dict[str, Any],
) -> str:
    payload = {
        "role": "Execution Officer final auto-buy clearance",
        "objective": "Decide whether the just-reviewed Robinhood order can proceed to place.",
        "non_negotiable_rules": [
            "If Artha review_gate is not PASS, allow_place must be false.",
            "If Robinhood order_checks_classification says blocking=true, allow_place must be false.",
            "A clearly classified non-blocking EQUITY_SUITABILITY broker alert can pass, but mention it in risk_flags.",
            "If broker order checks are ambiguous, unknown, or missing classification, allow_place must be false.",
            "If tradability is missing, halted, inactive, not tradeable, or fractional-ineligible for a fractional order, allow_place must be false.",
            "Do not change order parameters. This is a yes/no final clearance only.",
            "Return JSON only.",
        ],
        "action": {
            "action_id": action.get("action_id"),
            "ticker": action.get("ticker"),
            "side": action.get("side"),
            "action_type": action.get("action_type"),
            "status": action.get("status"),
            "payload_json": action.get("payload_json"),
        },
        "recorded_review": recorded_review,
        "tradability_response": tradability_response,
        "review_response": review_response,
        "required_schema": {
            "allow_place": "boolean",
            "confidence": "integer 1-10",
            "rationale": "short broker-preview-grounded explanation",
            "risk_flags": ["risks or alerts"],
            "requested_data": ["data required before place, if any"],
        },
    }
    return (
        "You are Artha's GPT-5.5 Execution Officer. Think carefully, but output only JSON.\n"
        "This is the final gate before a real-money Robinhood place_equity_order call.\n\n"
        f"{json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2)}"
    )


def robinhood_review_final_clearance(
    *,
    action: dict[str, Any],
    review_response: dict[str, Any],
    tradability_response: dict[str, Any] | None,
    recorded_review: dict[str, Any],
    use_llm: bool | None = None,
) -> dict[str, Any]:
    """LLM final clearance after Robinhood review, bounded by deterministic review_gate."""
    review_gate = recorded_review.get("review_gate") if isinstance(recorded_review.get("review_gate"), dict) else {}
    if not review_gate.get("passed"):
        return {
            "allow_place": False,
            "status": "BLOCKED",
            "reason": "Artha deterministic Robinhood review gate did not pass.",
            "review_gate": review_gate,
            "officer_used": False,
        }
    llm_enabled = Config.EXECUTION_OFFICER_ENABLED and (
        Config.EXECUTION_OFFICER_LLM_ENABLED if use_llm is None else bool(use_llm)
    )
    if not llm_enabled:
        return {
            "allow_place": True,
            "status": "PASS",
            "reason": "Execution Officer LLM disabled; deterministic review gate passed.",
            "review_gate": review_gate,
            "officer_used": False,
        }
    prompt = build_robinhood_review_clearance_prompt(
        action=action,
        review_response=review_response,
        tradability_response=tradability_response,
        recorded_review=recorded_review,
    )
    try:
        raw = ChatGPTBackendClient(
            model=Config.EXECUTION_OFFICER_MODEL,
            reasoning_effort=Config.EXECUTION_OFFICER_REASONING_EFFORT,
            temperature=Config.EXECUTION_OFFICER_TEMPERATURE,
            timeout=Config.EXECUTION_OFFICER_TIMEOUT_SECONDS,
        ).chat(prompt)
        parsed = _extract_json_object(raw) or {}
    except Exception as exc:
        logger.warning("[execution_officer] final Robinhood review clearance failed; blocking auto-place: %s", exc)
        return {
            "allow_place": False,
            "status": "BLOCKED",
            "reason": f"Execution Officer final clearance failed: {type(exc).__name__}: {exc}",
            "review_gate": review_gate,
            "officer_used": True,
        }
    allow = bool(parsed.get("allow_place")) and int(_as_float(parsed.get("confidence")) or 0) >= Config.ROBINHOOD_AUTO_BUY_MIN_CONFIDENCE
    return {
        "allow_place": allow,
        "status": "PASS" if allow else "BLOCKED",
        "reason": str(parsed.get("rationale") or "Execution Officer final clearance parsed."),
        "risk_flags": parsed.get("risk_flags") or [],
        "requested_data": parsed.get("requested_data") or [],
        "review_gate": review_gate,
        "officer_used": True,
        "officer_json": parsed,
    }
