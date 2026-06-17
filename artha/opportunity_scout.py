"""Agentic pre-Council opportunity routing.

The broker router answers "is this realistically executable today?".
The Opportunity Scout answers "which executable names deserve scarce Council
attention first?". It never makes a final buy/sell decision and it never
overrides the Council or Execution Officer.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chatgpt_backend import ChatGPTBackendClient
from .config import Config

logger = logging.getLogger(__name__)

BUY_NOW_LANE = "execution_ready"
QUALITY_SLEEVES = {"entry_quality", "quality_value", "pullback_quality"}
_FENCE = chr(96) * 3


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _num(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        parsed = float(str(value).replace(",", "").replace("%", ""))
        if math.isnan(parsed) or math.isinf(parsed):
            return default
        return parsed
    except Exception:
        return default


def _get_any(payload: Any, *keys: str, default: float | None = None) -> float | None:
    if not isinstance(payload, dict):
        return default
    for key in keys:
        if key in payload:
            value = _num(payload.get(key), default)
            if value is not None:
                return value
    return default


def _text(payload: Any, key: str, default: str = "") -> str:
    if not isinstance(payload, dict):
        return default
    return str(payload.get(key) or default).strip()


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith(_FENCE + "json"):
        cleaned = cleaned.split(_FENCE + "json", 1)[1]
        if _FENCE in cleaned:
            cleaned = cleaned.rsplit(_FENCE, 1)[0]
        cleaned = cleaned.strip()
    elif cleaned.startswith(_FENCE):
        cleaned = cleaned.split(_FENCE, 1)[1]
        if _FENCE in cleaned:
            cleaned = cleaned.rsplit(_FENCE, 1)[0]
        cleaned = cleaned.strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(cleaned[start:end])
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
    return {}


@dataclass
class OpportunityCard:
    ticker: str
    candidate: dict[str, Any]
    lane: str
    bucket: str
    route_reason_code: str
    candidate_rank: int
    route_score: float
    funnel_score: float
    scout_score: float
    sanity_flags: list[str] = field(default_factory=list)
    positives: list[str] = field(default_factory=list)
    negatives: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def allowed_for_buy_now(self) -> bool:
        return self.lane == BUY_NOW_LANE

    def compact(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "allowed_for_buy_now": self.allowed_for_buy_now,
            "lane": self.lane,
            "route_reason_code": self.route_reason_code,
            "candidate_rank": self.candidate_rank,
            "route_score": round(self.route_score, 2),
            "funnel_score": round(self.funnel_score, 2),
            "scout_score": round(self.scout_score, 2),
            "flags": self.sanity_flags[:8],
            "positives": self.positives[:6],
            "negatives": self.negatives[:8],
            **self.data,
        }


@dataclass
class OpportunityScoutResult:
    session_id: str
    created_at: str
    cards: list[OpportunityCard]
    ranked_cards: list[OpportunityCard]
    batches: list[list[OpportunityCard]]
    selected_for_council: list[OpportunityCard]
    model_used: str
    reasoning_effort: str
    agentic_enabled: bool
    agentic_used: bool
    research_only: bool
    deployable_amount: float
    summary: str
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    deterministic_fallback_reason: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "model_used": self.model_used,
            "reasoning_effort": self.reasoning_effort,
            "agentic_enabled": self.agentic_enabled,
            "agentic_used": self.agentic_used,
            "research_only": self.research_only,
            "deployable_amount": self.deployable_amount,
            "summary": self.summary,
            "deterministic_fallback_reason": self.deterministic_fallback_reason,
            "selected_for_council": [c.ticker for c in self.selected_for_council],
            "batches": [[c.ticker for c in batch] for batch in self.batches],
            "ranked_cards": [c.compact() for c in self.ranked_cards],
            "tool_trace": self.tool_trace,
        }

    def save_artifact(self, root: Path | None = None) -> str:
        base = root or (Path(__file__).resolve().parent.parent / "data" / "opportunity_scout")
        day = self.created_at[:10]
        out_dir = base / day
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.session_id}.json"
        path.write_text(json.dumps(self.to_payload(), indent=2, sort_keys=True), encoding="utf-8")
        return str(path)


class OpportunityScout:
    """Rank broker-clean candidates into Council batches."""

    def __init__(
        self,
        *,
        collector: Any | None = None,
        model_client_cls: type[ChatGPTBackendClient] = ChatGPTBackendClient,
    ) -> None:
        self.collector = collector
        self.model_client_cls = model_client_cls
        self._web_tool_calls = 0

    def rank(
        self,
        router_result: Any,
        *,
        session_id: str,
        market_snapshot: dict[str, Any] | None = None,
        deployment: dict[str, Any] | None = None,
        batch_size: int | None = None,
        max_batches: int | None = None,
        candidate_limit: int | None = None,
    ) -> OpportunityScoutResult:
        created_at = _utcnow().isoformat()
        batch_size = max(1, int(batch_size or Config.OPPORTUNITY_SCOUT_BATCH_SIZE or Config.SCAN_COUNCIL_MAX))
        max_batches = max(1, int(max_batches or Config.OPPORTUNITY_SCOUT_MAX_BATCHES))
        candidate_limit = max(
            batch_size,
            int(candidate_limit or Config.OPPORTUNITY_SCOUT_CANDIDATE_LIMIT or batch_size * max_batches),
        )
        deployment = deployment or {}
        deployable = float(_num(deployment.get("deployable_amount"), 0.0) or 0.0)
        research_only = deployable < float(Config.SCAN_MIN_DEPLOYABLE_FOR_BUY_COUNCIL)

        cards = self._build_cards(router_result, market_snapshot=market_snapshot, limit=candidate_limit)
        deterministic_ranked = self._deterministic_rank(cards)
        ranked = deterministic_ranked
        tool_trace: list[dict[str, Any]] = []
        agentic_used = False
        fallback_reason = ""

        if Config.OPPORTUNITY_SCOUT_ENABLED and Config.OPPORTUNITY_SCOUT_LLM_ENABLED and deterministic_ranked:
            try:
                ranked, tool_trace = self._run_agentic_scout(
                    deterministic_ranked,
                    market_snapshot=market_snapshot or {},
                    deployment=deployment,
                    batch_size=batch_size,
                    max_batches=max_batches,
                )
                agentic_used = True
            except Exception as exc:
                fallback_reason = f"agentic_scout_failed:{type(exc).__name__}:{str(exc)[:240]}"
                logger.warning("[opportunity_scout] Agentic scout failed; using deterministic ranking: %s", exc)

        batches = self._make_batches(ranked, batch_size=batch_size, max_batches=max_batches, research_only=research_only)
        selected = batches[0] if batches else []
        summary = self._summary(cards, ranked, batches, research_only, deployable, fallback_reason)
        result = OpportunityScoutResult(
            session_id=session_id,
            created_at=created_at,
            cards=cards,
            ranked_cards=ranked,
            batches=batches,
            selected_for_council=selected,
            model_used=Config.OPPORTUNITY_SCOUT_MODEL,
            reasoning_effort=Config.OPPORTUNITY_SCOUT_REASONING_EFFORT,
            agentic_enabled=bool(Config.OPPORTUNITY_SCOUT_LLM_ENABLED),
            agentic_used=agentic_used,
            research_only=research_only,
            deployable_amount=deployable,
            summary=summary,
            tool_trace=tool_trace,
            deterministic_fallback_reason=fallback_reason,
        )
        try:
            result.save_artifact()
        except Exception as exc:
            logger.warning("[opportunity_scout] Could not save scout artifact: %s", exc)
        return result

    def _build_cards(self, router_result: Any, *, market_snapshot: dict[str, Any] | None, limit: int) -> list[OpportunityCard]:
        decisions = list(getattr(router_result, "decisions", []) or [])
        decisions = decisions[: max(0, int(limit))]
        return [self._card_from_decision(row, market_snapshot=market_snapshot or {}) for row in decisions if row]

    def _card_from_decision(self, row: Any, *, market_snapshot: dict[str, Any]) -> OpportunityCard:
        candidate = dict(getattr(row, "candidate", {}) or {})
        ticker = str(getattr(row, "ticker", "") or candidate.get("symbol") or "").upper()
        price = _num(getattr(row, "live_price", None), None) or _num(getattr(row, "price", None), None) or _num(candidate.get("price"), 0.0) or 0.0
        targets = candidate.get("price_target_consensus") or {}
        dcf = candidate.get("dcf") or {}
        ratios = candidate.get("ratios_ttm") or {}
        metrics = candidate.get("key_metrics_ttm") or {}
        recs = candidate.get("recommendation_trends") or {}
        estimates = candidate.get("analyst_estimates") or {}
        short_interest = candidate.get("short_interest") or {}

        target = _get_any(targets, "targetConsensus", "target_consensus")
        target_low = _get_any(targets, "targetLow", "target_low")
        target_high = _get_any(targets, "targetHigh", "target_high")
        dcf_value = _get_any(dcf, "dcf", "dcfValue", "DCF")
        consensus_upside = ((target / price) - 1.0) * 100.0 if price and target else None
        dcf_upside = ((dcf_value / price) - 1.0) * 100.0 if price and dcf_value and dcf_value > 0 else None
        beta = _num(candidate.get("beta"), 1.0) or 1.0
        fg = market_snapshot.get("fear_greed") or {}
        fear_greed = _num(fg.get("value"), 50.0) or 50.0

        pe = _get_any(
            ratios,
            "peRatioTTM",
            "priceEarningsRatioTTM",
            "priceToEarningsRatioTTM",
            "priceToEarningsRatio",
        )
        ps = _get_any(ratios, "priceToSalesRatioTTM", "priceToSalesRatio")
        fcf_yield = _get_any(metrics, "freeCashFlowYieldTTM", "freeCashFlowYield")
        pfcf = _get_any(
            metrics,
            "pfcfRatioTTM",
            "priceToFreeCashFlowRatioTTM",
            "priceToFreeCashFlowsRatioTTM",
            "priceToFreeCashFlowRatio",
        )
        net_upgrades = _num(recs.get("net_upgrades_30d"), 0.0) or 0.0
        net_downgrades = _num(recs.get("net_downgrades_30d"), 0.0) or 0.0
        net_revisions = net_upgrades - net_downgrades
        sleeve = str(candidate.get("primary_alpha_sleeve") or "").lower()
        sleeve_scores = candidate.get("alpha_sleeve_scores") or {}
        r1 = _num(candidate.get("return_1m"), 0.0) or 0.0
        r3 = _num(candidate.get("return_3m"), 0.0) or 0.0
        r12 = _num(candidate.get("return_12m"), 0.0) or 0.0

        positives: list[str] = []
        negatives: list[str] = []
        flags: list[str] = []
        score = 50.0

        route_score = float(_num(getattr(row, "route_score", None), 0.0) or 0.0)
        funnel_score = float(_num(getattr(row, "funnel_score", None), 0.0) or 0.0)
        score += min(14.0, max(-8.0, route_score / 40.0))

        if consensus_upside is not None:
            if consensus_upside >= 20:
                score += 18
                positives.append(f"consensus upside {consensus_upside:.1f}%")
            elif consensus_upside >= 8:
                score += 8
                positives.append(f"usable consensus upside {consensus_upside:.1f}%")
            elif consensus_upside < -10:
                score -= 30
                flags.append("price_above_consensus_target")
                negatives.append(f"price is {abs(consensus_upside):.1f}% above consensus target")
            elif consensus_upside < 3:
                score -= 12
                negatives.append(f"thin consensus upside {consensus_upside:.1f}%")

        if dcf_upside is not None:
            if dcf_upside >= 20:
                score += 6
            elif dcf_upside <= -40:
                score -= 8
                negatives.append(f"DCF downside {abs(dcf_upside):.1f}%")
        elif dcf_value is not None and dcf_value <= 0:
            score -= 6
            flags.append("non_positive_dcf")

        if fcf_yield is not None:
            if fcf_yield >= 0.04:
                score += 10
                positives.append(f"FCF yield {fcf_yield:.1%}")
            elif fcf_yield <= 0:
                score -= 14
                flags.append("negative_or_zero_fcf_yield")
                negatives.append("FCF yield is not positive")
            elif fcf_yield < 0.015:
                score -= 7
                negatives.append(f"low FCF yield {fcf_yield:.1%}")

        if pe and pe > 80:
            score -= 10
            flags.append("high_pe")
        if ps and ps > 15:
            score -= 12
            flags.append("high_price_sales")
        if pfcf and pfcf > 60:
            score -= 8
            flags.append("high_pfcf")

        if net_revisions >= 2:
            score += 7
            positives.append(f"net analyst upgrades/revisions +{net_revisions:.0f}")
        elif net_revisions < 0:
            score -= 5
            negatives.append(f"net analyst revisions {net_revisions:.0f}")

        if sleeve in QUALITY_SLEEVES:
            score += 7
            positives.append(f"{sleeve} sleeve")
        entry_score = _num(sleeve_scores.get("entry_quality"), 0.0) or 0.0
        if entry_score >= Config.FUNNEL_ENTRY_QUALITY_MIN_SCORE:
            score += min(10, entry_score / 2)
            positives.append(f"entry-quality score {entry_score:.1f}")

        if fear_greed >= 60 and beta >= 2.0:
            penalty = 8 if beta < 3 else 15
            score -= penalty
            flags.append("greed_high_beta_momentum_penalty")
            negatives.append(f"high beta {beta:.2f} in Greed regime")
        if r1 > 40 or r3 > 90 or r12 > 350:
            score -= 8
            flags.append("technical_extension_risk")
            negatives.append("recent move looks extended")
        if getattr(row, "spread_pct", None) is not None and float(getattr(row, "spread_pct")) <= 0.002:
            score += 4
            positives.append("very tight live spread")

        if str(getattr(row, "lane", "")) != BUY_NOW_LANE:
            score -= 35
            flags.append("not_buy_now_executable")

        compact_data = {
            "company": candidate.get("name") or "",
            "sector": candidate.get("sector") or "",
            "industry": candidate.get("industry") or "",
            "price": round(price, 4) if price else None,
            "beta": beta,
            "primary_alpha_sleeve": sleeve,
            "consensus_target": target,
            "consensus_upside_pct": round(consensus_upside, 2) if consensus_upside is not None else None,
            "target_low": target_low,
            "target_high": target_high,
            "dcf_value": dcf_value,
            "dcf_upside_pct": round(dcf_upside, 2) if dcf_upside is not None else None,
            "pe_ttm": pe,
            "ps_ttm": ps,
            "pfcf_ttm": pfcf,
            "fcf_yield": fcf_yield,
            "recommendation_consensus": recs.get("consensus"),
            "net_revision_30d": net_revisions,
            "return_1m": r1,
            "return_3m": r3,
            "return_12m": r12,
            "momentum_trend": candidate.get("momentum_trend"),
            "short_interest_pct": short_interest.get("short_interest_pct"),
            "earnings_risk_flag": (candidate.get("earnings_context") or {}).get("earnings_risk_flag"),
            "spread_pct": getattr(row, "spread_pct", None),
            "avg_dollar_volume": getattr(row, "dollar_volume", None),
        }
        return OpportunityCard(
            ticker=ticker,
            candidate=candidate,
            lane=str(getattr(row, "lane", "")),
            bucket=str(getattr(row, "bucket", "")),
            route_reason_code=str(getattr(row, "reason_code", "")),
            candidate_rank=int(getattr(row, "candidate_rank", 0) or 0),
            route_score=route_score,
            funnel_score=funnel_score,
            scout_score=max(0.0, min(100.0, score)),
            sanity_flags=flags,
            positives=positives,
            negatives=negatives,
            data=compact_data,
        )

    def _deterministic_rank(self, cards: list[OpportunityCard]) -> list[OpportunityCard]:
        eligible = [c for c in cards if c.allowed_for_buy_now]
        eligible.sort(key=lambda c: c.scout_score, reverse=True)
        if not eligible:
            return []

        reserved: list[OpportunityCard] = []
        reserve_count = max(0, int(Config.OPPORTUNITY_SCOUT_RESERVE_QUALITY_SLOTS))
        for card in eligible:
            if len(reserved) >= reserve_count:
                break
            if card.data.get("primary_alpha_sleeve") in QUALITY_SLEEVES or card.scout_score >= 65:
                reserved.append(card)

        ranked: list[OpportunityCard] = []
        seen: set[str] = set()
        for card in reserved + eligible:
            if card.ticker in seen:
                continue
            ranked.append(card)
            seen.add(card.ticker)
        return ranked

    def _make_batches(
        self,
        ranked: list[OpportunityCard],
        *,
        batch_size: int,
        max_batches: int,
        research_only: bool,
    ) -> list[list[OpportunityCard]]:
        if not ranked:
            return []
        total = batch_size if research_only else batch_size * max_batches
        selected = ranked[:total]
        return [selected[i:i + batch_size] for i in range(0, len(selected), batch_size)]

    def _run_agentic_scout(
        self,
        cards: list[OpportunityCard],
        *,
        market_snapshot: dict[str, Any],
        deployment: dict[str, Any],
        batch_size: int,
        max_batches: int,
    ) -> tuple[list[OpportunityCard], list[dict[str, Any]]]:
        card_by_ticker = {c.ticker: c for c in cards}
        tool_trace: list[dict[str, Any]] = []
        self._web_tool_calls = 0
        prompt = self._agent_prompt(
            cards=cards,
            market_snapshot=market_snapshot,
            deployment=deployment,
            batch_size=batch_size,
            max_batches=max_batches,
        )
        client = self.model_client_cls(
            model=Config.OPPORTUNITY_SCOUT_MODEL,
            reasoning_effort=Config.OPPORTUNITY_SCOUT_REASONING_EFFORT,
            temperature=Config.OPPORTUNITY_SCOUT_TEMPERATURE,
            timeout=Config.OPPORTUNITY_SCOUT_TIMEOUT_SECONDS,
        )
        scratch = ""
        for _step in range(max(2, int(Config.OPPORTUNITY_SCOUT_MAX_TOOL_STEPS))):
            raw = client.chat(prompt + scratch)
            parsed = _parse_json_object(raw)
            if parsed.get("final_ranking"):
                ranked = self._validate_agent_ranking(parsed, card_by_ticker)
                if not any(item.get("tool_name") == "read_candidate_cards" for item in tool_trace):
                    raise RuntimeError("Scout returned final ranking before reading candidate cards")
                tool_trace.append(
                    {
                        "tool_name": "final_ranking",
                        "status": "PASS",
                        "result": {
                            "ranked_tickers": [c.ticker for c in ranked],
                            "reason": str(parsed.get("summary") or "")[:1000],
                        },
                    }
                )
                return ranked, tool_trace

            tool_name = str(parsed.get("tool_name") or "").strip()
            args = parsed.get("args") if isinstance(parsed.get("args"), dict) else {}
            result = self._run_tool(tool_name, args, cards, market_snapshot, deployment)
            status = "FAIL" if isinstance(result, dict) and result.get("error") else "PASS"
            tool_trace.append(
                {
                    "tool_name": tool_name or "invalid_model_output",
                    "args": args,
                    "status": status,
                    "reason": str(parsed.get("reason") or "")[:500],
                    "result": self._json_safe(result),
                }
            )
            scratch += (
                "\n\nTOOL RESULT "
                + json.dumps({"tool_name": tool_name, "status": status, "result": result}, ensure_ascii=True)[:18000]
                + "\nContinue. Either request another tool or return final_ranking JSON."
            )
        raise RuntimeError("Scout exhausted tool budget without a valid final_ranking")

    def _agent_prompt(
        self,
        *,
        cards: list[OpportunityCard],
        market_snapshot: dict[str, Any],
        deployment: dict[str, Any],
        batch_size: int,
        max_batches: int,
    ) -> str:
        fg = market_snapshot.get("fear_greed") or {}
        return f"""You are Artha's GPT-5.5 Opportunity Scout Agent.

Mission:
Rank broker-clean, execution-ready stock candidates into Council batches. You are NOT the Council and you are NOT the Execution Officer.

Operating rules:
1. Preserve standards. Never force a buy candidate because the user wants activity.
2. Use structured provider data as the hard anchor. Web/news is supporting context unless official and corroborated.
3. The broker router decides execution/data feasibility only. Company risk remains for Council.
4. Penalize names already above analyst consensus target unless there is strong, specific evidence of estimate/target revision.
5. Penalize high-beta, technically extended momentum names more heavily in Greed regimes.
6. Prefer scarce Council slots for candidates with viable upside, reasonable cash-flow/valuation support, clean data, and buyable execution.
7. If FMP target data appears stale or conflicts with the thesis, use tools to inspect FMP/web context before ranking.
8. Output only JSON. No markdown.

Current scan context:
- Candidate cards available: {len(cards)}
- Council batch size: {batch_size}
- Max batches: {max_batches}
- Equity sentiment: {fg.get("value", "?")} ({fg.get("label", "?")})
- Deployment context: {json.dumps(deployment, sort_keys=True, ensure_ascii=True)[:2000]}

Available tools:
- read_candidate_cards: returns compact cards with deterministic sanity flags and structured data.
  Args: {{"tickers":["AAPL"], "include_all": true}}
- fetch_fmp_snapshot: pulls/refetches FMP quote, price target, DCF, ratios, and key metrics for specific tickers.
  Args: {{"tickers":["AAPL","GOOG"]}}
- web_research: current web snippets for specific ticker/reason. Use sparingly for stale/conflicting target or catalyst questions.
  Args: {{"ticker":"CIEN", "query":"CIEN analyst price target AI networking valuation June 2026"}}
- read_broker_context: returns router/broker feasibility cards and latest local Robinhood snapshot summary.
  Args: {{"tickers":["AAPL"]}}

First action:
Call read_candidate_cards before final_ranking.

Final JSON schema:
{{
  "final_ranking": true,
  "summary": "short explanation",
  "ranked_tickers": ["TICK1", "TICK2"],
  "batch_reasoning": [
    {{"batch": 1, "tickers": ["TICK1"], "reason": "why this batch deserves Council first"}}
  ],
  "deprioritized": [
    {{"ticker": "TICK2", "reason": "above target / high beta in Greed / bad FCF"}}
  ],
  "data_conflicts": [
    {{"ticker": "TICK3", "issue": "FMP target conflict", "resolution": "how you handled it"}}
  ]
}}

Tool request JSON schema:
{{"tool_name":"read_candidate_cards","args":{{"include_all":true}},"reason":"Need the compact cards before ranking."}}"""

    def _run_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        cards: list[OpportunityCard],
        market_snapshot: dict[str, Any],
        deployment: dict[str, Any],
    ) -> Any:
        tool_name = (tool_name or "").strip()
        ticker_filter = {str(t).upper() for t in (args.get("tickers") or []) if str(t).strip()}
        selected_cards = [c for c in cards if not ticker_filter or c.ticker in ticker_filter]
        if tool_name == "read_candidate_cards":
            include_all = bool(args.get("include_all", False))
            rows = selected_cards if include_all or ticker_filter else selected_cards[:16]
            return {
                "market_snapshot": market_snapshot,
                "deployment": deployment,
                "cards": [c.compact() for c in rows],
            }
        if tool_name == "read_broker_context":
            snapshot = self._read_latest_robinhood_snapshot()
            return {
                "snapshot": snapshot,
                "broker_router_cards": [
                    {
                        "ticker": c.ticker,
                        "allowed_for_buy_now": c.allowed_for_buy_now,
                        "lane": c.lane,
                        "reason_code": c.route_reason_code,
                        "spread_pct": c.data.get("spread_pct"),
                        "avg_dollar_volume": c.data.get("avg_dollar_volume"),
                    }
                    for c in selected_cards
                ],
            }
        if tool_name == "fetch_fmp_snapshot":
            return self._fetch_fmp_snapshot([c.ticker for c in selected_cards])
        if tool_name == "web_research":
            if self._web_tool_calls >= Config.OPPORTUNITY_SCOUT_MAX_WEB_TOOL_CALLS:
                return {"error": "web_research call budget exhausted"}
            self._web_tool_calls += 1
            ticker = str(args.get("ticker") or (next(iter(ticker_filter), "") if ticker_filter else "")).upper()
            query = str(args.get("query") or f"{ticker} stock analyst target valuation catalyst latest").strip()
            from .search import search_web

            return {
                "ticker": ticker,
                "query": query,
                "results": search_web(
                    query,
                    count=max(1, int(Config.OPPORTUNITY_SCOUT_WEB_RESULTS_PER_CALL)),
                    freshness="week",
                ),
            }
        return {"error": f"Unknown tool: {tool_name}"}

    def _fetch_fmp_snapshot(self, tickers: list[str]) -> dict[str, Any]:
        from .collector import FMPCollector

        fmp = FMPCollector()
        provider_timeout = max(1, int(getattr(Config, "FUNNEL_ENRICH_PROVIDER_TIMEOUT_SECONDS", 6) or 6))
        provider_retries = max(0, int(getattr(Config, "FUNNEL_ENRICH_PROVIDER_RETRIES", 0) or 0))
        out: dict[str, Any] = {}
        for ticker in tickers[:8]:
            try:
                out[ticker] = {
                    "quote": fmp.quote(ticker),
                    "price_target_consensus": fmp.price_target_consensus(
                        ticker,
                        timeout=provider_timeout,
                        retries=provider_retries,
                    ),
                    "dcf": fmp.dcf(ticker, timeout=provider_timeout, retries=provider_retries),
                    "ratios_ttm": fmp.ratios_ttm(ticker, timeout=provider_timeout, retries=provider_retries),
                    "key_metrics_ttm": fmp.key_metrics_ttm(ticker, timeout=provider_timeout, retries=provider_retries),
                }
            except Exception as exc:
                out[ticker] = {"error": str(exc)[:300]}
        return out

    @staticmethod
    def _read_latest_robinhood_snapshot() -> dict[str, Any]:
        path = Path(Config.ROBINHOOD_RECONCILIATION_SNAPSHOT_FILE)
        if not path.exists():
            return {"available": False, "path": str(path)}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            positions = payload.get("positions") or payload.get("canonical", {}).get("positions") or []
            return {
                "available": True,
                "path": str(path),
                "generated_at": payload.get("generated_at") or payload.get("imported_at"),
                "position_count": len(positions) if isinstance(positions, list) else None,
                "status": payload.get("status"),
                "warnings": payload.get("warnings") or [],
            }
        except Exception as exc:
            return {"available": False, "path": str(path), "error": str(exc)[:300]}

    def _validate_agent_ranking(
        self,
        parsed: dict[str, Any],
        card_by_ticker: dict[str, OpportunityCard],
    ) -> list[OpportunityCard]:
        ranked: list[OpportunityCard] = []
        seen: set[str] = set()
        for raw in parsed.get("ranked_tickers") or []:
            ticker = str(raw).upper().strip()
            card = card_by_ticker.get(ticker)
            if not card or ticker in seen or not card.allowed_for_buy_now:
                continue
            ranked.append(card)
            seen.add(ticker)
        # Append deterministic survivors the model omitted so batch expansion
        # still has a complete ordered slate.
        for card in sorted(card_by_ticker.values(), key=lambda c: c.scout_score, reverse=True):
            if card.allowed_for_buy_now and card.ticker not in seen:
                ranked.append(card)
                seen.add(card.ticker)
        if not ranked:
            raise RuntimeError("Scout final ranking contained no buy-now eligible tickers")
        return ranked

    @staticmethod
    def _json_safe(value: Any, max_chars: int = 12000) -> Any:
        try:
            encoded = json.dumps(value, ensure_ascii=True, sort_keys=True)
        except Exception:
            encoded = str(value)
        if len(encoded) > max_chars:
            encoded = encoded[:max_chars] + "...[truncated]"
        try:
            return json.loads(encoded)
        except Exception:
            return encoded

    @staticmethod
    def _summary(
        cards: list[OpportunityCard],
        ranked: list[OpportunityCard],
        batches: list[list[OpportunityCard]],
        research_only: bool,
        deployable: float,
        fallback_reason: str,
    ) -> str:
        eligible = sum(1 for c in cards if c.allowed_for_buy_now)
        top = ", ".join(f"${c.ticker}" for c in (batches[0] if batches else [])[:8]) or "none"
        mode = "research-only" if research_only else "buy-now capable"
        fallback = f" Fallback: {fallback_reason}" if fallback_reason else ""
        return (
            f"{mode}: {eligible}/{len(cards)} cards are broker-clean; first Council batch is {top}; "
            f"deployable=${deployable:.2f}.{fallback}"
        )


def rank_opportunities_for_council(
    router_result: Any,
    *,
    session_id: str,
    collector: Any | None = None,
    market_snapshot: dict[str, Any] | None = None,
    deployment: dict[str, Any] | None = None,
    batch_size: int | None = None,
    max_batches: int | None = None,
    candidate_limit: int | None = None,
) -> OpportunityScoutResult:
    return OpportunityScout(collector=collector).rank(
        router_result,
        session_id=session_id,
        market_snapshot=market_snapshot,
        deployment=deployment,
        batch_size=batch_size,
        max_batches=max_batches,
        candidate_limit=candidate_limit,
    )
