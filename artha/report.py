"""Report formatter — generates Telegram-optimized investment reports.

Handles both individual stock analyses and weekly market briefs.
All numeric formatting uses safe parsing to handle API inconsistencies.
"""
from collections import Counter
from datetime import datetime
import re
from typing import Optional, Union
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from pathlib import Path

from .council import CouncilDecision
from .portfolio import Position
from .config import Config

_PORTFOLIO_JSON = Path(__file__).resolve().parent.parent / "data" / "portfolio.json"


def _get_held_tickers() -> set[str]:
    """Load held tickers from portfolio.json for position-aware reports."""
    import json
    try:
        data = json.loads(_PORTFOLIO_JSON.read_text())
        return {
            str(p.get("ticker", "")).upper()
            for p in (data.get("positions") or [])
            if isinstance(p, dict)
        }
    except Exception:
        return set()


def _to_float(value: object) -> Optional[float]:
    """Parse numeric API values safely (handles strings like '1.2%' or '1,234.5')."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("%", "").replace(",", "")
        if cleaned == "":
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _fmt_price(value: Optional[float]) -> str:
    """Format a price value or return N/A."""
    return f"{value:,.2f}" if value is not None else "N/A"


def _fmt_change(value: Optional[float]) -> str:
    """Format a percentage change or return N/A."""
    return f"{value:+.1f}%" if value is not None else "N/A"


def _verdict_emoji(verdict: str) -> str:
    """Map verdict to emoji (supports v2 action classes)."""
    mapping = {
        "STRONG BUY":   "🟢🟢",
        "BUY":          "🟢",
        "STARTER":      "🔵",
        "TACTICAL_BUY": "🟡",
        "ACCUMULATE":   "💚",
        "ADD":          "➕",
        "HOLD":         "⏸️",
        "WATCH":        "👁️",
        "DEFER":        "⏳",
        "TRIM":         "✂️",
        "SELL":         "🔴",
        "AVOID":        "🚫",
        "STRONG SELL":  "🔴🔴",
    }
    return mapping.get(verdict.upper(), "⚪")


def _confidence_bar(confidence: int) -> str:
    """Visual confidence bar."""
    filled = "█" * confidence
    empty = "░" * (10 - confidence)
    return f"[{filled}{empty}] {confidence}/10"


def _format_score_bar(score: int, max_score: int = 100) -> str:
    """Visual score bar for opportunity scores."""
    filled = int(score / max_score * 10)
    empty = 10 - filled
    return f"[{'█' * filled}{'░' * empty}] {score}/{max_score}"


def _format_score_components(components: dict) -> str:
    """Format score component breakdown."""
    if not components:
        return ""
    lines = []
    comp_map = {
        "technical_setup":      ("📈 Technical",      25),
        "fundamental_quality":  ("🏢 Fundamental",    20),
        "contrarian_sentiment": ("🎭 Sentiment",      15),
        "regime_alignment":     ("🌐 Regime Fit",     15),
        "catalyst_asymmetry":   ("⚡ Catalyst",       10),
        "data_quality":         ("📋 Data Quality",   10),
        "liquidity_execution":  ("💧 Liquidity",       5),
    }
    for key, (label, max_pts) in comp_map.items():
        val = components.get(key)
        if val is not None:
            bar_filled = int(val / max_pts * 5)
            bar = "█" * bar_filled + "░" * (5 - bar_filled)
            lines.append(f"  {label}: {val}/{max_pts} [{bar}]")
    return "\n".join(lines)


def _format_decision_checks(decision: CouncilDecision) -> str:
    """Format deterministic valuation, portfolio risk, and calibration checks."""
    sections: list[str] = []
    scoring_audit = getattr(decision, "scoring_audit", {}) or {}
    if isinstance(scoring_audit, dict) and scoring_audit:
        base = _to_float(scoring_audit.get("base_score")) or 0.0
        rules = _to_float(scoring_audit.get("rule_adjustment_total")) or 0.0
        cio = _to_float(scoring_audit.get("cio_adjustment")) or 0.0
        final = _to_float(scoring_audit.get("final_score")) or (
            _to_float(getattr(decision, "opportunity_score", 0)) or 0.0
        )
        status = str(scoring_audit.get("cio_adjustment_status") or "none")
        category = str(scoring_audit.get("cio_adjustment_category") or "none")
        audit_line = (
            "🧾 Score Audit: "
            f"base {base:.0f} + rules {rules:+.0f} + CIO {cio:+.0f} = {final:.0f} "
            f"| CIO {status}/{category}"
        )
        rejected = str(scoring_audit.get("cio_adjustment_rejected_reason") or "").strip()
        if rejected:
            audit_line += f"\n  CIO adjustment rejected: {_shorten(rejected, 95)}"
        sections.append(audit_line)

    valuation = getattr(decision, "valuation_expectations", {}) or {}
    if isinstance(valuation, dict) and valuation:
        targets = valuation.get("analyst_targets") or {}
        dcf = valuation.get("dcf") or {}
        consensus_upside = _to_float(targets.get("consensus_upside_pct"))
        dcf_upside = _to_float(dcf.get("upside_pct"))
        parts = [
            f"signal {valuation.get('valuation_signal', 'unknown')}",
            f"score {valuation.get('valuation_score', 'N/A')}/100",
            f"expectation risk {valuation.get('expectation_risk_level', 'unknown')}",
        ]
        if consensus_upside is not None:
            parts.append(f"consensus {consensus_upside:+.1f}%")
        if dcf_upside is not None:
            parts.append(f"DCF {dcf_upside:+.1f}%/{dcf.get('reliability', 'unknown')}")
        flags = valuation.get("risk_flags") or []
        flag_text = f"\n  Valuation flags: {'; '.join(_shorten(f, 72) for f in flags[:3])}" if flags else ""
        sections.append("🧮 Valuation: " + " | ".join(parts) + flag_text)

    risk = getattr(decision, "portfolio_factor_risk", {}) or {}
    if isinstance(risk, dict) and risk:
        flags = risk.get("risk_flags") or []
        flag_text = f"\n  Portfolio flags: {'; '.join(_shorten(f, 72) for f in flags[:3])}" if flags else ""
        candidate_weight = _to_float(risk.get("candidate_weight_pct")) or 0.0
        sector_after = _to_float(risk.get("sector_after_candidate_pct")) or 0.0
        sections.append(
            "🧭 Portfolio Risk: "
            f"{risk.get('risk_level', 'unknown')} | "
            f"candidate {candidate_weight:.1f}% | "
            f"sector-after {sector_after:.1f}% | "
            f"bench {risk.get('sector_benchmark_ticker', 'SPY')}"
            + flag_text
        )

    meta = getattr(decision, "calibration_meta_signal", {}) or {}
    if isinstance(meta, dict) and meta:
        sections.append(
            "📏 Calibration: "
            f"{meta.get('status', 'unknown')} | "
            f"completed {meta.get('completed_shadow_rows', 0)}/{meta.get('minimum_samples', 20)} | "
            f"{meta.get('recommendation', 'do_not_adjust')}"
        )

    if not sections:
        return ""
    return "\n\n" + "\n".join(sections)


def _shorten(text: object, limit: int = 115) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def _source_label(item: dict) -> str:
    source = str(item.get("source") or "unknown")
    url = str(item.get("url") or "")
    if not url:
        return source
    try:
        host = urlparse(url).netloc.replace("www.", "")
        if host:
            return f"{source} ({host})"
    except Exception:
        pass
    return source


def _format_source_audit(
    trace: dict,
    dossier_path: str = "",
    defer_watch_id: str = "",
    cited_text: str = "",
) -> str:
    evidence = trace.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = []
    evidence = [item for item in evidence if isinstance(item, dict)]
    if not evidence and not dossier_path and not defer_watch_id:
        return ""

    lines = ["\n\n🔎 **SOURCE AUDIT:**"]
    if dossier_path:
        lines.append(f"  Decision dossier: {dossier_path}")
    if defer_watch_id:
        lines.append(f"  Entry watch: {defer_watch_id}")
    if evidence:
        source_counts = Counter(_source_label(item) for item in evidence)
        category_counts = Counter(str(item.get("category") or "unknown") for item in evidence)
        top_sources = ", ".join(f"{src}={count}" for src, count in source_counts.most_common(5))
        top_categories = ", ".join(f"{cat}={count}" for cat, count in category_counts.most_common(5))
        dated = sum(1 for item in evidence if item.get("as_of") or item.get("freshness"))
        linked = sum(1 for item in evidence if item.get("url"))
        lines.append(f"  Evidence: {len(evidence)} items | dated/freshness-labeled {dated} | URL-labeled {linked}")
        if top_sources:
            lines.append(f"  Sources: {top_sources}")
        if top_categories:
            lines.append(f"  Categories: {top_categories}")

        by_id = {str(item.get("evidence_id") or ""): item for item in evidence}
        cited_ids: list[str] = []
        for match in re.findall(r"E[0-9]{3,}", cited_text or ""):
            if match not in cited_ids:
                cited_ids.append(match)
        key_items: list[dict] = []
        for evidence_id in cited_ids:
            item = by_id.get(evidence_id)
            if item:
                key_items.append(item)
        for item in evidence:
            if len(key_items) >= 8:
                break
            evidence_id = str(item.get("evidence_id") or "")
            if evidence_id and evidence_id in cited_ids:
                continue
            key_items.append(item)

        lines.append("  Key/cited evidence:")
        for item in key_items[:8]:
            eid = str(item.get("evidence_id") or "?")
            source = str(item.get("source") or "unknown")
            freshness = str(item.get("as_of") or item.get("freshness") or "as-of unknown")
            conf = item.get("confidence")
            conf_text = f" conf {float(conf):.2f}" if isinstance(conf, (int, float)) else ""
            claim = _shorten(item.get("claim"), limit=105)
            lines.append(f"    • [{eid}] {source} | {freshness}{conf_text} | {claim}")

    gaps = trace.get("gaps") or []
    if isinstance(gaps, list) and gaps:
        lines.append(f"  Gaps: {'; '.join(_shorten(g, 80) for g in gaps[:3])}")
    conflicts = trace.get("conflicts") or []
    if isinstance(conflicts, list) and conflicts:
        lines.append(f"  Conflicts: {'; '.join(_shorten(c, 80) for c in conflicts[:3])}")
    return "\n".join(lines)


def format_stock_analysis(decision: CouncilDecision) -> str:
    """Format a council decision into a Telegram-friendly report (v2)."""
    tz = ZoneInfo(Config.REPORT_TIMEZONE)
    now = datetime.now(tz)

    emoji = _verdict_emoji(decision.final_verdict)

    # Position status tag (makes it clear whether this is actionable)
    held = _get_held_tickers()
    is_held = decision.ticker.upper() in held
    position_tag = " (you own this)" if is_held else ""

    # Build score line
    score_line = ""
    if decision.opportunity_score > 0:
        adj_note = ""
        if hasattr(decision, "adjusted_score") and decision.adjusted_score != decision.opportunity_score:
            adj_note = f" → {decision.adjusted_score} (regime-adj)"
        score_line = f"\nOpportunity Score: {_format_score_bar(decision.opportunity_score)}{adj_note}"

    # Build allocation line
    alloc_line = decision.allocation or "N/A"
    no_new_capital = str(getattr(decision, "final_verdict", "") or "").upper() in {
        "WATCH", "DEFER", "AVOID", "SELL", "TRIM", "HOLD",
    }
    if not no_new_capital and hasattr(decision, "stop_loss_pct") and decision.stop_loss_pct:
        alloc_line += f" | Stop: {decision.stop_loss_pct:.0%}"
    if not no_new_capital and hasattr(decision, "target_pct") and decision.target_pct:
        alloc_line += f" | Target: +{decision.target_pct:.0%}"

    # Build risk gate note (if blocked)
    gate_note = ""
    if hasattr(decision, "hard_risk_gate_passed") and not decision.hard_risk_gate_passed:
        gate_note = f"\n⛔ RISK GATE: {decision.hard_risk_gate_reason}"

    # Build data coverage line
    coverage_line = ""
    dq = getattr(decision, "data_quality_report", {}) or {}
    if dq:
        sources = dq.get("sources_used") or []
        source_text = ", ".join(sources[:5]) if isinstance(sources, list) else str(sources)
        context_gaps = dq.get("enrichment_missing_fields") or []
        gap_text = ""
        if isinstance(context_gaps, list) and context_gaps:
            gap_text = f" | Gaps: {', '.join(context_gaps[:4])}"
        coverage_line = (
            f"\nData Coverage: core {dq.get('completeness_score', 'N/A')}% "
            f"| context {dq.get('context_coverage_score', 'N/A')}% "
            f"| sources: {source_text or 'none'}{gap_text}"
        )

    # Build agentic diligence trace line
    agentic_line = ""
    trace = getattr(decision, "agentic_trace", {}) or {}
    if trace:
        evidence_count = len(trace.get("evidence") or [])
        role_count = len(trace.get("analyst_briefs") or {})
        trace_path = trace.get("trace_path") or ""
        status = "enabled" if trace.get("enabled") else "fallback"
        agentic_line = (
            f"\nAgentic Diligence: {status} | roles {role_count}/3 "
            f"| evidence {evidence_count}"
        )
        if trace_path:
            agentic_line += f" | trace: {trace_path}"

    source_audit_section = _format_source_audit(
        trace,
        dossier_path=str(getattr(decision, "dossier_path", "") or ""),
        defer_watch_id=str(getattr(decision, "defer_watch_id", "") or ""),
        cited_text=f"{decision.recommended_action}\n{decision.synthesis_report}",
    )

    # Build deployment context line
    deploy_line = ""
    deploy_ctx = getattr(decision, "deployment_context", {}) or {}
    if deploy_ctx:
        deploy_line = (
            f"\nDeployment: {deploy_ctx.get('regime_label', '')} regime "
            f"| NAV: ${deploy_ctx.get('total_nav', 0):,.0f} "
            f"| Deployable: ${deploy_ctx.get('deployable_amount', 0):,.0f} "
            f"| Budget cap: ${deploy_ctx.get('budget_cap_amount', 0):,.0f}"
        )

    # Build score components section
    components_section = ""
    if hasattr(decision, "score_components") and decision.score_components:
        comp_str = _format_score_components(decision.score_components)
        if comp_str:
            components_section = f"\n\n📊 **SCORE BREAKDOWN:**\n{comp_str}"

    checks_section = _format_decision_checks(decision)

    # Build invalidation conditions
    invalid_section = ""
    if hasattr(decision, "invalidation_conditions") and decision.invalidation_conditions:
        conds = "\n".join(f"  • {c}" for c in decision.invalidation_conditions[:3])
        invalid_section = f"\n\n⚠️ **INVALIDATION CONDITIONS:**\n{conds}"

    report = f"""📊 ARTHA COUNCIL v2 — ${decision.ticker}
{now.strftime("%A, %B %d %Y • %I:%M %p %Z")}

━━━━━━━━━━━━━━━

{emoji} **{decision.final_verdict}**: ${decision.ticker}{position_tag}
Council: {decision.consensus}
Confidence: {_confidence_bar(getattr(decision, 'confidence', 5))}{score_line}{gate_note}{coverage_line}{agentic_line}

Action: {decision.recommended_action}
Allocation: {alloc_line}{deploy_line}

━━━━━━━━━━━━━━━

🏛️ **THE COUNCIL SPOKE:**

**Fundamental Analyst** (GPT) → {decision.fundamental.verdict} ({_confidence_bar(decision.fundamental.confidence)})

**Technical Analyst** (Gemini) → {decision.technical.verdict} ({_confidence_bar(decision.technical.confidence)})

**Risk Analyst** ({decision.contrarian.model}) → {decision.contrarian.verdict} ({_confidence_bar(decision.contrarian.confidence)}){checks_section}{components_section}{invalid_section}{source_audit_section}

━━━━━━━━━━━━━━━

📖 **SYNTHESIS:**
{decision.synthesis_report}

━━━━━━━━━━━━━━━

💡 _Artha Council is AI-generated analysis. Always do your own research before investing._"""

    return report


def format_market_overview(
    market_data: dict,
    macro_data: dict,
    fear_greed: Optional[dict] = None,
) -> str:
    """Format a market overview / weekly brief."""
    tz = ZoneInfo(Config.REPORT_TIMEZONE)
    now = datetime.now(tz)

    # Extract key metrics safely
    sp500 = market_data.get("sp500", {}) or {}
    nasdaq = market_data.get("nasdaq", {}) or {}
    btc_data = market_data.get("btc", {}) or {}
    eth_data = market_data.get("eth", {}) or {}

    sp500_price = _to_float(sp500.get("price"))
    sp500_change = _to_float(sp500.get("changesPercentage"))
    nasdaq_price = _to_float(nasdaq.get("price"))
    nasdaq_change = _to_float(nasdaq.get("changesPercentage"))
    btc_price = _to_float(btc_data.get("price"))
    btc_change = _to_float(btc_data.get("changesPercentage"))
    eth_price = _to_float(eth_data.get("price"))

    # Equity sentiment. Crypto Fear & Greed is reported separately where needed.
    fg_value = "N/A"
    fg_label = ""
    if fear_greed:
        fg_value = fear_greed.get("value", "N/A")
        fg_label = fear_greed.get("label", "")

    # Macro data
    fed_rate = "N/A"
    if macro_data.get("fed_funds_rate"):
        obs = macro_data["fed_funds_rate"].get("observations", [])
        if obs and isinstance(obs, list):
            fed_rate = obs[0].get("value", "N/A") if isinstance(obs[0], dict) else "N/A"

    # Top movers
    gainers = market_data.get("gainers", []) or []
    losers = market_data.get("losers", []) or []

    top_gainers_str = ""
    for g in gainers[:3]:
        if isinstance(g, dict):
            top_gainers_str += f"  • ${g.get('symbol', '?')} {_fmt_change(_to_float(g.get('changesPercentage')))}\n"

    top_losers_str = ""
    for item in losers[:3]:
        if isinstance(item, dict):
            top_losers_str += f"  • ${item.get('symbol', '?')} {_fmt_change(_to_float(item.get('changesPercentage')))}\n"

    report = f"""📊 ARTHA WEEKLY BRIEF
{now.strftime("%A, %B %d %Y • %I:%M %p %Z")}

━━━━━━━━━━━━━━━

🌡️ **MARKET PULSE**
S&P 500 (SPY): ${_fmt_price(sp500_price)} ({_fmt_change(sp500_change)})
Nasdaq (QQQ): ${_fmt_price(nasdaq_price)} ({_fmt_change(nasdaq_change)})
BTC: ${_fmt_price(btc_price)} ({_fmt_change(btc_change)})
ETH: ${_fmt_price(eth_price)}
Equity Sentiment: {fg_value} ({fg_label})
Fed Rate: {fed_rate}%

━━━━━━━━━━━━━━━

📈 **TOP GAINERS**
{top_gainers_str}
📉 **TOP LOSERS**
{top_losers_str}
━━━━━━━━━━━━━━━

💡 _Artha — AI-powered investment intelligence_"""

    return report


def format_portfolio_summary(
    positions: list[Union[Position, dict]],
    total_value: float,
) -> str:
    """Format current portfolio status.
    
    Accepts both Position dataclass instances and raw dicts for flexibility.
    """
    tz = ZoneInfo(Config.REPORT_TIMEZONE)
    now = datetime.now(tz)

    lines = [
        "💼 **YOUR PORTFOLIO**",
        now.strftime("%B %d, %Y"),
        "",
        f"Total Value: ${total_value:,.2f}",
        "",
    ]

    for pos in positions:
        # Handle both Position dataclass and dict
        if isinstance(pos, Position):
            ticker = pos.ticker
            shares = pos.shares
            avg_cost = pos.avg_cost
            current_price = 0  # Needs to be enriched externally
        elif isinstance(pos, dict):
            ticker = pos.get("ticker", "?")
            shares = pos.get("shares", 0)
            avg_cost = pos.get("avg_cost", 0)
            current_price = pos.get("current_price", 0)
        else:
            continue

        shares_f = _to_float(shares) or 0
        avg_cost_f = _to_float(avg_cost) or 0
        current_f = _to_float(current_price) or 0
        total_cost = shares_f * avg_cost_f
        current_value = shares_f * current_f
        pnl = current_value - total_cost
        pnl_pct = (pnl / total_cost * 100) if total_cost > 0 else 0
        emoji = "🟢" if pnl >= 0 else "🔴"

        lines.append(
            f"{emoji} ${ticker}: {shares_f:.4g} shares @ ${avg_cost_f:.2f} → "
            f"${current_f:.2f} ({pnl_pct:+.1f}% / ${pnl:+.2f})"
        )

    lines.append("")
    lines.append("━━━━━━━━━━━━━━━")

    return "\n".join(lines)
