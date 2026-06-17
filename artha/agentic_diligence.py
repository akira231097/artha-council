"""Bounded agentic diligence for the Artha buy-side council.

This module turns each analyst into an investigative workflow:
plan -> gather role-specific evidence -> expose gaps -> write with citations.
It is deliberately bounded. It does not let analysts browse indefinitely or
make uncited claims.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config

logger = logging.getLogger(__name__)


UTC = timezone.utc
ROLE_FUNDAMENTAL = "fundamental"
ROLE_TECHNICAL = "technical"
ROLE_CONTRARIAN = "contrarian"
ROLES = (ROLE_FUNDAMENTAL, ROLE_TECHNICAL, ROLE_CONTRARIAN)

SOURCE_HIERARCHY_TEXT = (
    "Source hierarchy: structured provider data is the source of truth for hard "
    "numbers. SEC EDGAR is the official filing cross-check. FMP is primary for "
    "fundamentals, valuation anchors, estimates, and market/fundamental packets. "
    "Massive/yfinance/Finnhub are independent corroborating provider feeds. "
    "Current-web/search evidence is context for recent catalysts, sentiment, "
    "lawsuits, downgrades, and contradictions; it must not override structured "
    "provider data unless official, current, and corroborated."
)


@dataclass
class EvidenceItem:
    """One auditable fact passed into the agentic council."""

    evidence_id: str
    category: str
    claim: str
    source: str
    value: Any = None
    roles: list[str] = field(default_factory=list)
    confidence: float = 0.8
    freshness: str = ""
    url: str = ""
    as_of: str = ""


@dataclass
class AgenticDiligenceResult:
    """Trace returned to council after bounded diligence completes."""

    enabled: bool
    ticker: str
    generated_at: str
    role_plans: dict[str, list[str]] = field(default_factory=dict)
    role_queries: dict[str, list[str]] = field(default_factory=dict)
    analyst_briefs: dict[str, str] = field(default_factory=dict)
    cio_brief: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    trace_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceStore:
    """Collects evidence IDs and keeps the trace JSON-serializable."""

    def __init__(self, ticker: str, max_items: int | None = None):
        self.ticker = ticker.upper()
        self.max_items = max_items or Config.AGENTIC_MAX_EVIDENCE_ITEMS
        self._items: list[EvidenceItem] = []
        self._seq = 0

    @property
    def items(self) -> list[EvidenceItem]:
        return self._items

    def add(
        self,
        *,
        category: str,
        claim: str,
        source: str,
        value: Any = None,
        roles: list[str] | None = None,
        confidence: float = 0.8,
        freshness: str = "",
        url: str = "",
    ) -> str:
        if len(self._items) >= self.max_items:
            return ""
        self._seq += 1
        evidence_id = f"E{self._seq:03d}"
        self._items.append(
            EvidenceItem(
                evidence_id=evidence_id,
                category=category,
                claim=_trim(claim, 420),
                source=source,
                value=_json_safe(value),
                roles=roles or list(ROLES),
                confidence=max(0.0, min(1.0, float(confidence))),
                freshness=freshness,
                url=url,
                as_of=datetime.now(UTC).isoformat(),
            )
        )
        return evidence_id

    def by_role(self, role: str) -> list[EvidenceItem]:
        return [item for item in self._items if role in item.roles]


def build_agentic_diligence(
    ticker: str,
    stock_data: dict,
    macro_data: dict | None = None,
    market_overview: dict | None = None,
    intelligence_brief: str = "",
    data_quality_report: dict | None = None,
    *,
    enable_web: bool | None = None,
    write_trace: bool = True,
) -> AgenticDiligenceResult:
    """Build role-specific agentic diligence briefs and an audit trace.

    Args:
        enable_web: override for tests. None uses Config.AGENTIC_WEB_RESEARCH_ENABLED.
        write_trace: false for deterministic tests.
    """
    ticker = (ticker or stock_data.get("ticker") or "UNKNOWN").upper()
    generated_at = datetime.now(UTC).isoformat()
    if not Config.AGENTIC_COUNCIL_ENABLED:
        return AgenticDiligenceResult(enabled=False, ticker=ticker, generated_at=generated_at)

    store = EvidenceStore(ticker)
    data_quality_report = data_quality_report or stock_data.get("data_quality_report") or stock_data.get("data_quality") or {}
    macro_data = macro_data or {}
    market_overview = market_overview or {}

    gaps: list[str] = []
    conflicts: list[str] = []
    _ingest_core_packet(store, stock_data, macro_data, market_overview, intelligence_brief, data_quality_report)
    gaps.extend(_derive_gaps(stock_data, data_quality_report))
    conflicts.extend(_derive_conflicts(data_quality_report))

    role_plans = {role: _role_plan(role, ticker, stock_data, data_quality_report) for role in ROLES}
    role_queries = {role: _role_queries(role, ticker, stock_data) for role in ROLES}

    web_enabled = Config.AGENTIC_WEB_RESEARCH_ENABLED if enable_web is None else bool(enable_web)
    if web_enabled:
        _run_bounded_web_research(store, ticker, role_queries)

    analyst_briefs = {
        role: _render_analyst_brief(
            role=role,
            ticker=ticker,
            plan=role_plans[role],
            queries=role_queries[role],
            evidence=store.by_role(role),
            gaps=gaps,
            conflicts=conflicts,
        )
        for role in ROLES
    }
    cio_brief = _render_cio_brief(ticker, store.items, role_plans, role_queries, gaps, conflicts)

    evidence = [asdict(item) for item in store.items]
    trace_path = ""
    result = AgenticDiligenceResult(
        enabled=True,
        ticker=ticker,
        generated_at=generated_at,
        role_plans=role_plans,
        role_queries=role_queries,
        analyst_briefs=analyst_briefs,
        cio_brief=cio_brief,
        evidence=evidence,
        gaps=gaps,
        conflicts=conflicts,
    )
    if write_trace:
        trace_path = _write_trace(result)
        result.trace_path = trace_path
        if trace_path:
            result.cio_brief += f"\nTrace file: {trace_path}\n"
    logger.info(
        "[agentic] %s diligence ready: evidence=%d gaps=%d conflicts=%d trace=%s",
        ticker,
        len(evidence),
        len(gaps),
        len(conflicts),
        trace_path or "not-written",
    )
    return result


def _ingest_core_packet(
    store: EvidenceStore,
    stock_data: dict,
    macro_data: dict,
    market_overview: dict,
    intelligence_brief: str,
    dq: dict,
) -> None:
    ticker = store.ticker
    quote = stock_data.get("quote") or {}
    profile = stock_data.get("profile") or {}
    yf_quote = stock_data.get("yf_quote") or {}
    massive_quote = stock_data.get("massive_quote") or {}
    dcf = stock_data.get("dcf") or {}
    pt = stock_data.get("price_target_consensus") or {}
    estimates = stock_data.get("analyst_estimates") or {}
    rec_trends = stock_data.get("recommendation_trends") or {}
    short_interest = stock_data.get("short_interest") or {}
    earnings = stock_data.get("earnings_context") or {}
    sec = stock_data.get("sec") or {}
    technicals = stock_data.get("technicals") or {}
    history_checks = stock_data.get("history_provider_checks") or {}
    ratios = stock_data.get("ratios_ttm") or {}
    metrics = stock_data.get("key_metrics_ttm") or {}

    if quote:
        store.add(
            category="market",
            claim=f"{ticker} live FMP quote, market cap, volume, and daily change.",
            source="fmp.quote",
            value=_pick(quote, "price", "changePercentage", "volume", "marketCap", "dayHigh", "dayLow", "yearHigh", "yearLow"),
            roles=[ROLE_TECHNICAL, ROLE_CONTRARIAN, ROLE_FUNDAMENTAL],
            confidence=0.95,
            freshness="live/intraday",
        )
    if yf_quote:
        store.add(
            category="market_cross_check",
            claim=f"{ticker} yfinance quote/history is available as an independent price cross-check.",
            source="yfinance",
            value=_pick(yf_quote, "price", "market_cap", "volume", "sector"),
            roles=[ROLE_TECHNICAL, ROLE_CONTRARIAN],
            confidence=0.85,
            freshness="live/intraday",
        )
    if massive_quote:
        store.add(
            category="market_cross_check",
            claim=f"{ticker} Massive market data is available as an independent price/volume cross-check.",
            source=str(massive_quote.get("source") or "massive"),
            value=_pick(massive_quote, "price", "previous_close", "changesPercentage", "volume", "bid", "ask", "source"),
            roles=[ROLE_TECHNICAL, ROLE_CONTRARIAN],
            confidence=0.88,
            freshness="provider_market_data",
        )
    if history_checks:
        store.add(
            category="market_cross_check",
            claim=f"{ticker} price-history provider selection and cross-checks are available for technical analysis.",
            source="artha.history_provider_checks",
            value=_pick(history_checks, "selected_source", "providers", "conflicts"),
            roles=[ROLE_TECHNICAL, ROLE_CONTRARIAN],
            confidence=0.9,
            freshness="computed_from_provider_history",
        )
    if profile:
        store.add(
            category="business_profile",
            claim=f"{ticker} company identity, sector, industry, beta, and business description.",
            source="fmp.profile",
            value=_pick(profile, "companyName", "sector", "industry", "beta", "marketCap", "range"),
            roles=[ROLE_FUNDAMENTAL, ROLE_CONTRARIAN],
            confidence=0.9,
            freshness="current",
        )
    income = _first_dict(stock_data.get("income_statement"))
    if income:
        income_period = _statement_period_label(income)
        store.add(
            category="fundamentals",
            claim=f"{ticker} latest reported {income_period} income statement anchors revenue, profit, EPS, and margins.",
            source="fmp.income_statement",
            value=_pick(income, "date", "period", "calendarYear", "revenue", "netIncome", "eps", "grossProfitRatio", "operatingIncome"),
            roles=[ROLE_FUNDAMENTAL, ROLE_CONTRARIAN],
            confidence=0.9,
            freshness=str(income.get("date") or ""),
        )
    balance = _first_dict(stock_data.get("balance_sheet"))
    if balance:
        balance_period = _statement_period_label(balance)
        store.add(
            category="balance_sheet",
            claim=f"{ticker} latest reported {balance_period} balance sheet anchors liquidity and leverage checks.",
            source="fmp.balance_sheet",
            value=_pick(balance, "date", "period", "calendarYear", "cashAndCashEquivalents", "totalDebt", "totalAssets", "totalLiabilities", "totalStockholdersEquity"),
            roles=[ROLE_FUNDAMENTAL, ROLE_CONTRARIAN],
            confidence=0.9,
            freshness=str(balance.get("date") or ""),
        )
    cash_flow = _first_dict(stock_data.get("cash_flow"))
    if cash_flow:
        cash_flow_period = _statement_period_label(cash_flow)
        store.add(
            category="cash_flow",
            claim=f"{ticker} latest reported {cash_flow_period} cash-flow statement anchors free-cash-flow and quality checks.",
            source="fmp.cash_flow",
            value=_pick(cash_flow, "date", "period", "calendarYear", "netCashProvidedByOperatingActivities", "capitalExpenditure", "freeCashFlow"),
            roles=[ROLE_FUNDAMENTAL, ROLE_CONTRARIAN],
            confidence=0.9,
            freshness=str(cash_flow.get("date") or ""),
        )
    if ratios:
        store.add(
            category="valuation_quality",
            claim=f"{ticker} TTM ratios provide valuation and profitability context.",
            source="fmp.ratios_ttm",
            value=_pick(ratios, "peRatioTTM", "priceToSalesRatioTTM", "priceToBookRatioTTM", "debtEquityRatioTTM", "grossProfitMarginTTM", "returnOnEquityTTM"),
            roles=[ROLE_FUNDAMENTAL, ROLE_CONTRARIAN],
            confidence=0.85,
            freshness="ttm",
        )
    if metrics:
        store.add(
            category="key_metrics",
            claim=f"{ticker} TTM key metrics provide per-share and capital-efficiency context.",
            source="fmp.key_metrics_ttm",
            value=_pick(metrics, "freeCashFlowPerShareTTM", "revenuePerShareTTM", "netIncomePerShareTTM", "roicTTM", "workingCapitalTTM"),
            roles=[ROLE_FUNDAMENTAL, ROLE_CONTRARIAN],
            confidence=0.85,
            freshness="ttm",
        )
    if dcf:
        store.add(
            category="valuation",
            claim=f"{ticker} FMP DCF fair value estimate is available as one valuation anchor.",
            source="fmp.dcf",
            value=_pick(dcf, "date", "dcf", "Stock Price"),
            roles=[ROLE_FUNDAMENTAL, ROLE_CONTRARIAN],
            confidence=0.65,
            freshness=str(dcf.get("date") or ""),
        )
    if pt:
        store.add(
            category="sell_side_valuation",
            claim=f"{ticker} analyst price-target consensus is available.",
            source="fmp.price_target_consensus",
            value=_pick(pt, "targetConsensus", "targetMedian", "targetHigh", "targetLow"),
            roles=[ROLE_FUNDAMENTAL, ROLE_TECHNICAL, ROLE_CONTRARIAN],
            confidence=0.75,
            freshness="current",
        )
    if estimates:
        store.add(
            category="forward_estimates",
            claim=f"{ticker} forward estimate context is available for expectation risk.",
            source=str(estimates.get("source") or "fmp.analyst_estimates"),
            value=_pick(estimates, "next_q_eps_estimate", "next_q_revenue_estimate", "fy1_revenue_estimate", "price_target_consensus"),
            roles=[ROLE_FUNDAMENTAL, ROLE_CONTRARIAN],
            confidence=0.8,
            freshness="current",
        )
    if rec_trends:
        store.add(
            category="recommendation_trends",
            claim=f"{ticker} sell-side recommendation mix and upgrade/downgrade momentum are available.",
            source=str(rec_trends.get("source") or "finnhub.recommendation_trends"),
            value=_pick(rec_trends, "consensus", "recommendation_mix", "net_upgrades_30d", "net_downgrades_30d"),
            roles=[ROLE_FUNDAMENTAL, ROLE_TECHNICAL, ROLE_CONTRARIAN],
            confidence=0.8,
            freshness="current",
        )
    if short_interest:
        store.add(
            category="short_interest",
            claim=f"{ticker} short-interest/crowding context is available.",
            source=str(short_interest.get("source") or "short_interest"),
            value=_pick(short_interest, "short_interest_pct", "days_to_cover", "squeeze_risk_flag"),
            roles=[ROLE_TECHNICAL, ROLE_CONTRARIAN],
            confidence=0.75,
            freshness="current",
        )
    if sec:
        latest_filings = sec.get("latest_filings") or []
        facts = sec.get("financial_facts") or []
        store.add(
            category="sec_filings",
            claim=f"{ticker} official SEC submissions/companyfacts are available for filing cross-checks.",
            source="sec.edgar",
            value={
                "status": sec.get("status"),
                "cik": sec.get("cik"),
                "latest_10q_or_10k_staleness_days": sec.get("latest_10q_or_10k_staleness_days"),
                "facts_available": sec.get("facts_available"),
                "latest_filings": [_pick(f, "form", "filing_date", "report_date", "items") for f in latest_filings[:5]],
                "facts": [_pick(f, "label", "tag", "unit", "recent") for f in facts[:4]],
            },
            roles=[ROLE_FUNDAMENTAL, ROLE_CONTRARIAN],
            confidence=0.95 if sec.get("status") == "ok" else 0.65,
            freshness="official",
        )
    if earnings:
        store.add(
            category="earnings",
            claim=f"{ticker} earnings timing and surprise history are available.",
            source="earnings_context",
            value=_pick(earnings, "earnings_date", "days_to_earnings", "earnings_risk_flag", "earnings_defer_flag", "recent_surprises"),
            roles=[ROLE_FUNDAMENTAL, ROLE_TECHNICAL, ROLE_CONTRARIAN],
            confidence=0.8,
            freshness="current",
        )
    if technicals:
        store.add(
            category="technical_indicators",
            claim=f"{ticker} local technical indicator packet is available.",
            source="local.technicals",
            value=_pick(technicals, "rsi_14", "macd", "macd_signal", "sma_20", "sma_50", "sma_200", "bb_upper", "bb_lower"),
            roles=[ROLE_TECHNICAL, ROLE_CONTRARIAN],
            confidence=0.85,
            freshness="computed_from_price_history",
        )
    if macro_data:
        store.add(
            category="macro",
            claim="Macro indicators are available for rate/risk backdrop.",
            source="fred/macro",
            value=_pick(macro_data, "fed_funds_rate", "treasury_10y", "cpi", "unemployment_rate", "gdp_growth"),
            roles=[ROLE_FUNDAMENTAL, ROLE_TECHNICAL, ROLE_CONTRARIAN],
            confidence=0.8,
            freshness="current",
        )
    if market_overview:
        store.add(
            category="market_regime",
            claim="Market overview is available for regime and breadth context.",
            source="market_overview",
            value=_pick(market_overview, "sp500", "nasdaq", "vix", "fear_greed", "top_gainers", "top_losers"),
            roles=[ROLE_TECHNICAL, ROLE_CONTRARIAN],
            confidence=0.8,
            freshness="current",
        )
    if dq:
        store.add(
            category="data_quality",
            claim=f"{ticker} data quality and context coverage were validated before model analysis.",
            source="artha.data_quality",
            value=_pick(dq, "completeness_score", "context_coverage_score", "sources_used", "missing_fields", "enrichment_missing_fields", "source_conflicts", "staleness_warnings"),
            roles=[ROLE_FUNDAMENTAL, ROLE_TECHNICAL, ROLE_CONTRARIAN],
            confidence=1.0,
            freshness="current",
        )
    if intelligence_brief:
        store.add(
            category="current_web_brief",
            claim=f"{ticker} Research Desk intelligence brief was generated from current web/news sources.",
            source="research_desk",
            value=_trim(intelligence_brief, 1500),
            roles=[ROLE_FUNDAMENTAL, ROLE_TECHNICAL, ROLE_CONTRARIAN],
            confidence=0.75,
            freshness="current",
        )


def _run_bounded_web_research(
    store: EvidenceStore,
    ticker: str,
    role_queries: dict[str, list[str]],
) -> None:
    try:
        from .search import search_web
    except Exception as exc:
        logger.warning("[agentic] search provider unavailable: %s", exc)
        return

    max_queries = max(0, Config.AGENTIC_WEB_QUERIES_PER_ROLE)
    max_results = max(0, Config.AGENTIC_WEB_RESULTS_PER_QUERY)
    if max_queries == 0 or max_results == 0:
        return

    for role, queries in role_queries.items():
        for query in queries[:max_queries]:
            try:
                results = search_web(query, count=max_results, freshness="week")
            except Exception as exc:
                logger.warning("[agentic] web research failed for %s/%s: %s", ticker, role, exc)
                continue
            for result in (results or [])[:max_results]:
                store.add(
                    category="role_web_research",
                    claim=f"{role} follow-up web result for: {query}",
                    source=str(result.get("provider") or "web_search"),
                    value={
                        "title": result.get("title"),
                        "snippet": result.get("snippet"),
                        "date": result.get("date"),
                        "query": query,
                    },
                    roles=[role],
                    confidence=0.65,
                    freshness="recent_web",
                    url=str(result.get("url") or ""),
                )


def _role_plan(role: str, ticker: str, stock_data: dict, dq: dict) -> list[str]:
    base = {
        ROLE_FUNDAMENTAL: [
            "Check business quality using revenue, margins, cash flow, leverage, and SEC facts.",
            "Compare valuation anchors: current price, FMP DCF, analyst target consensus, and key ratios.",
            "Check whether forward estimates and recommendation trends support or weaken the thesis.",
            "Identify what evidence would make the idea a value trap or overvalued hold.",
        ],
        ROLE_TECHNICAL: [
            "Check price trend, moving averages, RSI/MACD, volume, and relative market behavior.",
            "Separate durable momentum from one-day noise or low-quality spikes.",
            "Check market regime, VIX/Fear-Greed, and sector rotation before judging timing.",
            "Identify entry levels, overextension risk, and event timing risk.",
        ],
        ROLE_CONTRARIAN: [
            "Build the strongest bear case before accepting the bull case.",
            "Check short interest, insider activity, downgrade risk, debt/liquidity, and SEC red flags.",
            "Look for hidden risks: litigation, regulation, dilution, valuation crowding, or narrative hype.",
            "Define hard invalidation conditions and reasons to reject the idea.",
        ],
    }
    plan = list(base.get(role, []))
    missing = dq.get("enrichment_missing_fields") or dq.get("missing_fields") or []
    if missing:
        plan.append(f"Explicitly account for missing or weak evidence: {', '.join(map(str, missing[:6]))}.")
    if not stock_data.get("sec") and role in (ROLE_FUNDAMENTAL, ROLE_CONTRARIAN):
        plan.append("SEC filing context is missing; downgrade confidence until official data is available.")
    return plan


def _role_queries(role: str, ticker: str, stock_data: dict) -> list[str]:
    profile = stock_data.get("profile") or {}
    company = str(profile.get("companyName") or ticker)
    sector = str(profile.get("sector") or "")
    if role == ROLE_FUNDAMENTAL:
        return [
            f"{ticker} {company} latest earnings guidance analyst estimates revenue margins",
            f"{ticker} {company} SEC filing risks balance sheet cash flow",
        ]
    if role == ROLE_TECHNICAL:
        return [
            f"{ticker} stock today volume momentum breakout relative strength",
            f"{ticker} {sector} sector rotation market regime stock trend",
        ]
    return [
        f"{ticker} {company} downgrade short interest insider selling lawsuit risk",
        f"{ticker} {company} bearish risks debt dilution regulatory investigation",
    ]


def _render_analyst_brief(
    *,
    role: str,
    ticker: str,
    plan: list[str],
    queries: list[str],
    evidence: list[EvidenceItem],
    gaps: list[str],
    conflicts: list[str],
) -> str:
    role_title = {
        ROLE_FUNDAMENTAL: "Fundamental / Valuation Agent",
        ROLE_TECHNICAL: "Technical / Market Structure Agent",
        ROLE_CONTRARIAN: "Contrarian / Risk Agent",
    }.get(role, role)
    lines = [
        f"AGENTIC DILIGENCE BRIEF: {role_title} for {ticker}",
        "",
        "Mode: bounded agentic investigation. Use the plan, evidence IDs, current-web results, and gaps below before writing your analyst report.",
        SOURCE_HIERARCHY_TEXT,
        "",
        "Research plan:",
    ]
    lines.extend(f"- {item}" for item in plan)
    lines.append("")
    lines.append("Bounded follow-up queries executed or available:")
    lines.extend(f"- {query}" for query in queries)
    lines.append("")
    lines.append("Evidence reviewed:")
    for item in evidence[:25]:
        value = _trim(json.dumps(item.value, default=str), 650)
        lines.append(
            f"- [{item.evidence_id}] {item.category} | source={item.source} | confidence={item.confidence:.2f} | {item.claim} | value={value}"
        )
    if not evidence:
        lines.append("- No role-specific evidence available; lower confidence.")
    if gaps:
        lines.append("")
        lines.append("Known evidence gaps to account for:")
        lines.extend(f"- {gap}" for gap in gaps[:8])
    if conflicts:
        lines.append("")
        lines.append("Known conflicts/staleness to resolve:")
        lines.extend(f"- {conflict}" for conflict in conflicts[:8])
    lines.extend(
        [
            "",
            "Mandatory behavior:",
            "- Cite evidence IDs for important claims, e.g. [E004].",
            "- If a claim is not supported by an evidence ID or the Intelligence Brief, say it is unverified.",
            "- Treat current-web/search results as context, not the source of truth for price, financials, valuation anchors, technical indicators, or filing facts.",
            "- If web evidence conflicts with FMP/SEC/Massive/yfinance/Finnhub/provider data, state the conflict and defer to the structured provider data unless the web source is official and corroborated.",
            "- Do not invent missing data. Missing data lowers confidence.",
            "- End with the normal VERDICT and CONFIDENCE format.",
        ]
    )
    return "\n".join(lines)


def _render_cio_brief(
    ticker: str,
    evidence: list[EvidenceItem],
    role_plans: dict[str, list[str]],
    role_queries: dict[str, list[str]],
    gaps: list[str],
    conflicts: list[str],
) -> str:
    source_counts: dict[str, int] = {}
    for item in evidence:
        source_counts[item.source] = source_counts.get(item.source, 0) + 1
    lines = [
        f"CIO AGENTIC CROSS-EXAM BRIEF: {ticker}",
        f"Evidence items: {len(evidence)}",
        f"Source counts: {source_counts}",
        SOURCE_HIERARCHY_TEXT,
        "",
        "Analyst diligence scope:",
    ]
    for role in ROLES:
        lines.append(f"- {role}: {len(role_plans.get(role, []))} plan checks, {len(role_queries.get(role, []))} bounded web queries")
    lines.extend(
        [
            "",
            "CIO cross-exam duties:",
            "- Verify each analyst's major claim against raw evidence IDs, valuation anchors, or the Intelligence Brief.",
            "- If analysts disagree, identify which side has stronger evidence instead of averaging opinions.",
            "- If all analysts agree, state the groupthink risk and what they may all be missing.",
            "- Treat structured provider data and official filings as the hard-data anchor; use current-web only as context unless official/corroborated.",
            "- If current-web evidence conflicts with FMP/SEC/Massive/yfinance/Finnhub/provider data, explain the conflict and do not let web-only claims drive a buy action.",
            "- Penalize data_quality when gaps or conflicts affect the recommendation.",
            "- Prefer WATCH/DEFER/AVOID over a buy-side action when current evidence is thin, stale, or contradictory.",
            "- Produce a decision only after bull/base/bear evidence survives this cross-check.",
        ]
    )
    if gaps:
        lines.append("")
        lines.append("High-impact gaps:")
        lines.extend(f"- {gap}" for gap in gaps[:10])
    if conflicts:
        lines.append("")
        lines.append("Conflicts/staleness:")
        lines.extend(f"- {conflict}" for conflict in conflicts[:10])
    return "\n".join(lines)


def _derive_gaps(stock_data: dict, dq: dict) -> list[str]:
    gaps = []
    for field in dq.get("missing_fields") or []:
        gaps.append(f"Required field missing: {field}")
    for field in dq.get("enrichment_missing_fields") or []:
        gaps.append(f"Context enrichment missing: {field}")
    sec = stock_data.get("sec") or {}
    if isinstance(sec, dict) and sec.get("status") not in ("ok", "partial"):
        gaps.append("SEC official filing context unavailable or incomplete.")
    if not stock_data.get("news") and not stock_data.get("benzinga_news") and not stock_data.get("finnhub_news"):
        gaps.append("Current news coverage is unavailable.")
    return gaps


def _derive_conflicts(dq: dict) -> list[str]:
    conflicts = []
    for key in ("source_conflicts", "staleness_warnings", "anomaly_flags", "hard_check_failures"):
        for item in dq.get(key) or []:
            conflicts.append(f"{key}: {item}")
    return conflicts


def _write_trace(result: AgenticDiligenceResult) -> str:
    try:
        date = datetime.now(UTC).strftime("%Y-%m-%d")
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        trace_dir = Path(__file__).resolve().parent.parent / "data" / "agentic_traces" / date
        trace_dir.mkdir(parents=True, exist_ok=True)
        path = trace_dir / f"{result.ticker}_{stamp}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
        return str(path)
    except Exception as exc:
        logger.warning("[agentic] failed to write trace for %s: %s", result.ticker, exc)
        return ""


def _pick(data: dict, *keys: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    return {key: _json_safe(data.get(key)) for key in keys if key in data}


def _first_dict(value: Any) -> dict:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return value if isinstance(value, dict) else {}


def _statement_period_label(row: dict) -> str:
    """Return a precise financial-statement period label for evidence claims."""
    period = str((row or {}).get("period") or "").upper().strip()
    if period in {"Q1", "Q2", "Q3", "Q4"}:
        return "quarterly"
    if period in {"FY", "ANNUAL"}:
        return "annual"
    # DataCollector requests quarterly statements for this packet; avoid implying
    # annual when the vendor omits the period field.
    return "periodic"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value[:8]]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return str(value)
    return value


def _trim(text: Any, limit: int) -> str:
    raw = str(text or "")
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 3)] + "..."
