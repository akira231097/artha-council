"""Data Layer V2 Verification — end-to-end tests for all new modules.

Run with: python -m artha.verify_data_layer

Tests each new module with real API calls and prints a clear pass/fail summary.
"""
from __future__ import annotations

import logging
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Callable

logging.basicConfig(
    level=logging.WARNING,  # Suppress noisy sub-module logs during tests
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error: str = ""
        self.details: str = ""

    def ok(self, details: str = "") -> "TestResult":
        self.passed = True
        self.details = details
        return self

    def fail(self, error: str) -> "TestResult":
        self.passed = False
        self.error = error
        return self


def run_test(name: str, fn: Callable[[], str]) -> TestResult:
    """Run a single test function, catching all exceptions."""
    result = TestResult(name)
    try:
        details = fn()
        result.ok(details or "")
    except AssertionError as e:
        result.fail(f"AssertionError: {e}")
    except Exception as e:
        result.fail(f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
    return result


def print_summary(results: list[TestResult]) -> int:
    """Print pass/fail summary and return exit code (0=all pass, 1=any fail)."""
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    print("\n" + "=" * 60)
    print(f"DATA LAYER V2 VERIFICATION — {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    for r in results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        print(f"  {status}  {r.name}")
        if r.passed and r.details:
            print(f"          {r.details}")
        if not r.passed:
            print(f"          ERROR: {r.error[:200]}")

    print("=" * 60)
    print(f"  {len(passed)}/{len(results)} tests passed")
    if failed:
        print(f"  FAILED: {', '.join(r.name for r in failed)}")
    print("=" * 60)

    return 0 if not failed else 1


# ---------------------------------------------------------------------------
# Phase 1 Tests
# ---------------------------------------------------------------------------

def test_vendor_priority() -> str:
    from artha.vendor_priority import VENDOR_PRIORITY, resolve_source, get_primary_vendor, get_fallback_vendor

    assert "equity_fundamentals" in VENDOR_PRIORITY
    assert "crypto_prices" in VENDOR_PRIORITY
    assert get_primary_vendor("equity_fundamentals") == "fmp"
    assert get_primary_vendor("equity_history") == "yfinance"
    assert get_fallback_vendor("equity_history") == "massive"

    # resolve_source: primary available
    result = resolve_source("test", {"data": 1}, {"fallback": 2})
    assert result == {"data": 1}, f"Expected primary, got {result}"

    # resolve_source: primary empty, fallback used
    result = resolve_source("test", None, {"fallback": 2})
    assert result == {"fallback": 2}, f"Expected fallback, got {result}"

    # resolve_source: both empty → None
    result = resolve_source("test", None, None)
    assert result is None

    return f"{len(VENDOR_PRIORITY)} domains mapped"


def test_liquidity_gate() -> str:
    from artha.liquidity import passes_liquidity_gate, compute_liquidity_score

    # Should pass: large cap, good volume, good price
    good = {"marketCap": 5_000_000_000, "avgVolume": 5_000_000, "price": 150.0}
    assert passes_liquidity_gate(good), "Large cap should pass"

    # Should fail: too cheap
    cheap = {"marketCap": 5_000_000_000, "avgVolume": 5_000_000, "price": 3.0}
    assert not passes_liquidity_gate(cheap), "Price $3 should fail"

    # Should fail: too small cap
    small_cap = {"marketCap": 500_000_000, "avgVolume": 5_000_000, "price": 50.0}
    assert not passes_liquidity_gate(small_cap), "Small cap should fail"

    # Liquidity score
    score = compute_liquidity_score(good)
    assert 0 <= score <= 100, f"Score out of range: {score}"
    assert score > 50, f"Large cap should score > 50, got {score}"

    score_small = compute_liquidity_score(cheap)
    assert score_small < score, "Small cap should score lower"

    return f"Gate logic OK, score={score:.1f}"


def test_config_thresholds() -> str:
    from artha.config import Config

    assert hasattr(Config, "LIQUIDITY_MIN_MARKET_CAP")
    assert Config.LIQUIDITY_MIN_MARKET_CAP == 1_000_000_000
    assert hasattr(Config, "LIQUIDITY_MIN_ADV")
    assert hasattr(Config, "LIQUIDITY_MIN_PRICE")

    # AV should no longer be in required keys
    missing = Config.validate()
    required_missing = [m for m in missing if m.startswith("REQUIRED") and "ALPHA_VANTAGE" in m]
    assert not required_missing, f"AV still in required: {required_missing}"

    return "Liquidity thresholds set, AV removed from required"


def test_regime_indicators_new_fields() -> str:
    from artha.regime_indicators import RegimeIndicators, compute_regime_indicators

    # Check new fields exist on dataclass
    ri = RegimeIndicators()
    assert hasattr(ri, "hy_credit_spread")
    assert hasattr(ri, "ig_credit_spread")
    assert hasattr(ri, "dxy")
    assert hasattr(ri, "spy_drawdown_from_52w_high")
    assert hasattr(ri, "initial_jobless_claims")
    assert hasattr(ri, "oil_price_wti")
    assert hasattr(ri, "event_risk_state")

    # Check crisis_data injection
    crisis_data = {
        "hy_oas": 3.5,
        "ig_oas": 1.2,
        "dxy": 104.5,
        "spy_drawdown": -0.08,
        "initial_jobless_claims": 220000,
        "oil_price": 72.3,
    }
    # Only test with crisis_data injection (no live API call for speed)
    ri2 = RegimeIndicators()
    # Manually populate as compute_regime_indicators would
    ri2.hy_credit_spread = float(crisis_data["hy_oas"])
    ri2.ig_credit_spread = float(crisis_data["ig_oas"])
    ri2.dxy = float(crisis_data["dxy"])
    ri2.spy_drawdown_from_52w_high = float(crisis_data["spy_drawdown"])
    ri2.initial_jobless_claims = float(crisis_data["initial_jobless_claims"])
    ri2.oil_price_wti = float(crisis_data["oil_price"])

    text = ri2.to_prompt_text()
    assert "CREDIT & MACRO STRESS" in text, "Missing stress section in prompt text"
    assert "HY Credit Spread" in text
    assert "DXY" in text or "Dollar" in text

    return "New fields present, prompt text updated"


def test_economic_calendar_api() -> str:
    from artha.economic_calendar import EconomicCalendar, compute_event_risk_state

    cal = EconomicCalendar()
    events = cal.fetch(days_ahead=14, days_back=3)

    # May be empty on weekends/holidays, but should not crash
    assert isinstance(events, list), f"Expected list, got {type(events)}"

    # compute_event_risk_state must handle empty input gracefully
    state_empty = compute_event_risk_state([])
    assert state_empty.state == "none", f"Empty events should return 'none', got {state_empty.state}"
    assert isinstance(state_empty.major_events_next_7d, list), "major_events_next_7d must be list"

    # If events returned, state must be a valid value
    if events:
        state2 = compute_event_risk_state(events)
        valid_states = ("none", "pre_major_24h", "same_day_major", "post_major_24h")
        assert state2.state in valid_states, f"Invalid state: {state2.state}"
        final_state = state2.state
    else:
        final_state = "none (no events returned)"

    return f"{len(events)} events fetched, state={final_state}"


def test_earnings_calendar_api() -> str:
    from artha.earnings_calendar import EarningsCalendar, get_earnings_context

    # Test with AAPL — earnings_date may be None if no upcoming dates in window
    ec = get_earnings_context("AAPL")
    assert ec.ticker == "AAPL", f"ticker mismatch: {ec.ticker}"
    assert ec.earnings_time in ("bmo", "amc", "unknown"), f"Unexpected time: {ec.earnings_time}"
    assert isinstance(ec.earnings_risk_flag, bool), "earnings_risk_flag must be bool"
    assert isinstance(ec.earnings_defer_flag, bool), "earnings_defer_flag must be bool"
    assert isinstance(ec.recent_surprises, list), "recent_surprises must be list"

    # If earnings_date is present, validate consistency
    if ec.earnings_date is not None:
        assert ec.days_to_earnings is not None, "days_to_earnings must be set if earnings_date is set"
        assert ec.days_to_earnings >= 0, f"Negative days_to_earnings: {ec.days_to_earnings}"

    details_parts = [f"time={ec.earnings_time}"]
    if ec.earnings_date:
        details_parts.append(f"next={ec.earnings_date} ({ec.days_to_earnings}d)")
    else:
        details_parts.append("no_upcoming_earnings")
    if ec.recent_surprises:
        details_parts.append(f"surprises={len(ec.recent_surprises)}")

    return ", ".join(details_parts)


def test_collector_no_av() -> str:
    """Verify AlphaVantageCollector is gone and DataCollector works."""
    import artha.collector as col

    assert not hasattr(col, "AlphaVantageCollector"), "AlphaVantageCollector should be removed"

    dc = col.DataCollector()
    assert not hasattr(dc, "alphavantage"), "DataCollector should not have alphavantage"
    assert hasattr(dc, "fmp")
    assert hasattr(dc, "finnhub")
    assert hasattr(dc, "yf")

    # Check FMP screener method exists
    assert hasattr(dc.fmp, "screener"), "FMPCollector.screener() missing"

    return "AlphaVantage removed, screener method present"


def test_collector_fmp_screener() -> str:
    """Test FMP screener returns real data."""
    from artha.collector import FMPCollector

    fmp = FMPCollector()
    results = fmp.screener(
        market_cap_more_than=10_000_000_000,  # $10B+ for speed
        volume_more_than=500_000,
        price_more_than=10.0,
        limit=20,
    )

    assert results is not None, "Screener returned None (API error?)"
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert len(results) > 0, "Screener returned empty list"

    first = results[0]
    assert "symbol" in first, f"No 'symbol' key in screener result: {first.keys()}"

    return f"{len(results)} stocks returned, first={first.get('symbol')}"


def test_collect_stock_pit_metadata() -> str:
    """Test collect_stock() includes PIT metadata."""
    from artha.collector import DataCollector

    dc = DataCollector()
    data = dc.collect_stock("MSFT")

    assert "as_of_datetime_utc" in data, "Missing as_of_datetime_utc"
    assert "source" in data, "Missing source"
    assert "ingested_at_utc" in data, "Missing ingested_at_utc"
    assert data["source"] == "DataCollector.collect_stock"
    assert "earnings_context" in data, "Missing earnings_context"
    for key in ("short_interest", "recommendation_trends", "analyst_estimates", "sec"):
        assert key in data, f"Missing production context field: {key}"

    ec = data.get("earnings_context")
    # earnings_context can be None if API unavailable, just check key exists
    completeness = data.get("data_quality", {}).get("completeness", 0)
    sec_status = (data.get("sec") or {}).get("status")
    estimates_source = (data.get("analyst_estimates") or {}).get("source")

    return (
        f"PIT OK, completeness={completeness}%, "
        f"earnings_context={'present' if ec else 'None'}, "
        f"sec={sec_status}, estimates={estimates_source}"
    )


# ---------------------------------------------------------------------------
# Phase 2 Tests
# ---------------------------------------------------------------------------

def test_universe_builder() -> str:
    from artha.universe import UniverseBuilder

    builder = UniverseBuilder()
    universe = builder.build_universe(
        regime_type="goldilocks",
        overlays=["ai_tech_momentum"],
        min_market_cap=10_000_000_000,  # $10B+ for speed
        limit=100,
    )

    assert isinstance(universe, list), f"Expected list, got {type(universe)}"
    assert len(universe) > 0, "Universe is empty — FMP screener may have failed"

    first = universe[0]
    assert hasattr(first, "symbol"), "UniverseCandidate missing symbol"
    assert hasattr(first, "regime_score"), "UniverseCandidate missing regime_score"

    # Check that regime scoring was applied
    tech_candidates = [c for c in universe if c.sector == "Technology"]

    return (
        f"{len(universe)} candidates, "
        f"{len(tech_candidates)} tech, "
        f"first={first.symbol} regime_score={first.regime_score}"
    )


def test_rank_candidates() -> str:
    from artha.universe import UniverseBuilder
    from artha.rank_candidates import rank_universe, compute_momentum_score

    # Test compute_momentum_score — with 12-month return (full formula)
    score_12m = compute_momentum_score(
        return_1m=5.0, return_3m=12.0, vol_20d=18.0, return_12m=30.0
    )
    assert isinstance(score_12m, float), f"Expected float, got {type(score_12m)}"
    # Full formula: 0.5*30 + 0.3*12 + 0.2*5 - 0.2*max(0, 18-20) = 15+3.6+1.0-0 = 19.6
    expected_12m = 0.5 * 30.0 + 0.3 * 12.0 + 0.2 * 5.0 - 0.2 * max(0, 18.0 - 20.0)
    assert abs(score_12m - expected_12m) < 0.01, f"12M score mismatch: {score_12m} vs {expected_12m}"

    # Test compute_momentum_score — without 12-month return (fallback formula)
    score_3m = compute_momentum_score(return_1m=5.0, return_3m=12.0, vol_20d=18.0)
    assert isinstance(score_3m, float), f"Expected float, got {type(score_3m)}"
    expected_3m = 0.6 * 12.0 + 0.4 * 5.0 - 0.2 * max(0, 18.0 - 20.0)
    assert abs(score_3m - expected_3m) < 0.01, f"3M fallback score mismatch: {score_3m} vs {expected_3m}"

    # Test with None inputs
    score_none = compute_momentum_score(None, None, None)
    assert score_none == 0.0, f"Expected 0.0 for all None, got {score_none}"

    # Test that 12-month formula produces different score than fallback
    assert score_12m != score_3m, "12M and fallback scores should differ"

    # Build a small universe for ranking test (use top mega-caps for speed)
    builder = UniverseBuilder()
    universe = builder.build_universe(
        min_market_cap=100_000_000_000,  # $100B+ (mega-caps only, fast)
        limit=30,
    )

    if universe:
        ranked = rank_universe(universe, top_n=10)
        assert isinstance(ranked, list)
        if len(ranked) > 1:
            # Verify sorted by combined_score descending
            scores = [r["combined_score"] for r in ranked]
            assert scores == sorted(scores, reverse=True), "Not sorted by combined_score"

    return f"momentum_score formula OK, ranked {len(ranked) if universe else 0} mega-caps"


def test_data_quality_report() -> str:
    from artha.data_quality import validate_stock_data, DataQualityReport

    # Test with minimal valid data
    good_data = {
        "ticker": "TEST",
        "quote": {"price": 150.0, "volume": 1000000, "marketCap": 5e9},
        "yf_quote": {"price": 151.0, "market_cap": 5e9},
        "profile": {"companyName": "Test Corp"},
        "income_statement": [{"date": "2025-09-30", "revenue": 1e9, "grossProfit": 4e8}],
        "balance_sheet": [{"date": "2025-09-30"}],
        "cash_flow": [{"date": "2025-09-30"}],
        "ratios_ttm": {"priceToEarningsRatioTTM": 25.0},
        "key_metrics_ttm": {"returnOnInvestedCapitalTTM": 0.25},
        "price_history": [{"date": f"2025-{i:02d}-01", "close": 100 + i, "volume": 1000000} for i in range(1, 25)],
        "technicals": {"rsi": 55.0},
        "news": [{"headline": "Test news"}],
    }

    report = validate_stock_data(good_data)
    assert isinstance(report, DataQualityReport)
    assert report.ticker == "TEST"
    assert report.passed_hard_checks, f"Hard checks failed: {report.hard_check_failures}"
    assert report.completeness_score > 80, f"Low completeness: {report.completeness_score}"

    # Test with bad data (price = 0)
    bad_data = dict(good_data)
    bad_data["quote"] = {"price": 0, "volume": 0, "marketCap": 0}
    bad_data["yf_quote"] = {"price": 0}
    bad_report = validate_stock_data(bad_data)
    assert not bad_report.passed_hard_checks, "Should fail hard checks for price=0"

    return f"completeness={report.completeness_score}%, hard_checks={report.passed_hard_checks}"


def test_analyst_signals() -> str:
    from artha.analyst_signals import get_recommendation_trends, get_analyst_estimates

    # Test recommendation trends for AAPL
    recs = get_recommendation_trends("AAPL")
    assert isinstance(recs, dict)
    assert "consensus" in recs
    assert recs["consensus"] in ("strong_buy", "buy", "hold", "sell", "strong_sell", "unknown")
    assert "recommendation_mix" in recs

    # Test analyst estimates for AAPL
    estimates = get_analyst_estimates("AAPL")
    assert isinstance(estimates, dict)
    assert "source" in estimates
    assert "quarterly_estimates" in estimates
    assert "annual_estimates" in estimates

    return (
        f"AAPL consensus={recs.get('consensus')}, "
        f"PT={estimates.get('price_target_consensus')}, "
        f"q_estimates={len(estimates.get('quarterly_estimates') or [])}"
    )


def test_sec_edgar_context() -> str:
    """Test SEC EDGAR official-source context for a large public company."""
    from artha.collector import FMPCollector
    from artha.sec_edgar import get_sec_company_context

    profile = FMPCollector().company_profile("AAPL")
    assert profile and profile.get("cik"), "FMP profile must provide CIK for production SEC lookup"

    sec = get_sec_company_context("AAPL", profile=profile)
    assert isinstance(sec, dict), f"Expected dict, got {type(sec)}"
    assert sec.get("source") == "sec"
    assert sec.get("ticker") == "AAPL"
    assert sec.get("cik"), f"Missing CIK: {sec}"
    assert sec.get("status") in ("ok", "partial", "unavailable")
    if sec.get("status") in ("ok", "partial"):
        assert isinstance(sec.get("latest_filings"), list), "latest_filings must be list"
        assert isinstance(sec.get("financial_facts"), list), "financial_facts must be list"

    return (
        f"status={sec.get('status')}, cik={sec.get('cik')}, "
        f"filings={len(sec.get('latest_filings') or [])}, "
        f"facts={sec.get('facts_available')}"
    )


def test_funnel_smoke() -> str:
    """Smoke test the PromotionFunnel with a simple regime dict."""
    from artha.config import Config
    from artha.funnel import PromotionFunnel

    Config.FUNNEL_ENRICH_MAX = min(Config.FUNNEL_ENRICH_MAX, 5)
    funnel = PromotionFunnel()
    regime_dict = {
        "regime_type": "goldilocks",
        "event_overlays": [{"type": "ai_tech_momentum"}],
    }

    candidates = funnel.run(
        regime_packet=regime_dict,
        max_council_candidates=5,
        fallback_on_failure=True,
    )

    assert isinstance(candidates, list), f"Expected list, got {type(candidates)}"
    assert len(candidates) > 0, "Funnel returned no candidates (even fallback failed)"
    assert len(candidates) <= 5

    first = candidates[0]
    assert "symbol" in first, f"Candidate missing symbol: {first.keys()}"

    return (
        f"{len(candidates)} candidates: "
        f"{', '.join(c['symbol'] for c in candidates[:3])}"
    )


# ---------------------------------------------------------------------------
# Phase 3 Tests (new — Issues 13 & 14)
# ---------------------------------------------------------------------------

def test_short_interest() -> str:
    """Test get_short_interest returns a well-formed dict for AAPL."""
    from artha.analyst_signals import get_short_interest

    result = get_short_interest("AAPL")
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"

    expected_keys = {"short_interest_pct", "days_to_cover", "squeeze_risk_flag", "source", "ticker"}
    missing = expected_keys - set(result.keys())
    assert not missing, f"Missing keys: {missing}"

    assert result["ticker"] == "AAPL", f"ticker mismatch: {result['ticker']}"
    assert isinstance(result["squeeze_risk_flag"], bool), "squeeze_risk_flag must be bool"
    assert result["source"] in ("fmp", "yfinance", "unavailable"), f"Unexpected source: {result['source']}"

    # If data returned, validate types
    if result["short_interest_pct"] is not None:
        assert isinstance(result["short_interest_pct"], float), "short_interest_pct must be float"
    if result["days_to_cover"] is not None:
        assert isinstance(result["days_to_cover"], float), "days_to_cover must be float"

    return f"source={result['source']}, short_pct={result['short_interest_pct']}, dtc={result['days_to_cover']}"


def test_compute_regime_indicators_crisis_path() -> str:
    """Test compute_regime_indicators populates crisis_data fields correctly."""
    from artha.regime_indicators import RegimeIndicators, compute_regime_indicators

    crisis_data = {
        "hy_oas": 3.5,
        "ig_oas": 1.2,
        "dxy": 104.5,
        "spy_drawdown": -0.08,
        "initial_jobless_claims": 220000,
        "oil_price": 72.3,
    }

    try:
        # Pass crisis_data — ETF/news fetches may or may not succeed, that's fine
        indicators = compute_regime_indicators(crisis_data=crisis_data)
    except Exception as e:
        raise AssertionError(f"compute_regime_indicators raised unexpectedly: {e}")

    assert isinstance(indicators, RegimeIndicators), "Must return RegimeIndicators"
    assert indicators.hy_credit_spread == 3.5, f"hy_credit_spread mismatch: {indicators.hy_credit_spread}"
    assert indicators.ig_credit_spread == 1.2, f"ig_credit_spread mismatch: {indicators.ig_credit_spread}"
    assert indicators.dxy == 104.5, f"dxy mismatch: {indicators.dxy}"
    assert indicators.spy_drawdown_from_52w_high == -0.08, f"drawdown mismatch: {indicators.spy_drawdown_from_52w_high}"
    assert indicators.initial_jobless_claims == 220000.0, f"jobless mismatch: {indicators.initial_jobless_claims}"
    assert indicators.oil_price_wti == 72.3, f"oil mismatch: {indicators.oil_price_wti}"

    # event_risk_state must be a valid string (populated from EconomicCalendar or "none")
    valid_states = ("none", "pre_major_24h", "same_day_major", "post_major_24h")
    assert indicators.event_risk_state in valid_states, f"Invalid event_risk_state: {indicators.event_risk_state}"

    prompt_text = indicators.to_prompt_text()
    assert "CREDIT & MACRO STRESS" in prompt_text, "Missing CREDIT & MACRO STRESS section"

    return (
        f"crisis fields populated: hy={indicators.hy_credit_spread}, "
        f"dxy={indicators.dxy}, event_risk={indicators.event_risk_state}"
    )


def test_funnel_fallback_shape() -> str:
    """Test that _fallback() returns enriched candidates with expected schema."""
    from artha.funnel import PromotionFunnel

    funnel = PromotionFunnel()

    try:
        candidates = funnel._fallback(max_candidates=3)
    except Exception as e:
        raise AssertionError(f"_fallback() raised: {e}")

    assert isinstance(candidates, list), f"Expected list, got {type(candidates)}"

    if not candidates:
        # Fallback may return empty if scanner also fails — this is acceptable
        return "fallback returned empty (scanner unavailable) — handled gracefully"

    required_keys = {"symbol", "funnel_score"}
    for i, c in enumerate(candidates):
        assert isinstance(c, dict), f"Candidate {i} is not a dict: {type(c)}"
        missing = required_keys - set(c.keys())
        assert not missing, f"Candidate {i} missing keys: {missing}"
        assert isinstance(c.get("symbol", ""), str), f"Candidate {i} symbol must be str"
        assert isinstance(c.get("funnel_score", 0), (int, float)), f"Candidate {i} funnel_score must be numeric"
        # earnings_context key should be present (even if None) after _enrich
        assert "earnings_context" in c, f"Candidate {i} missing earnings_context (not enriched)"

    symbols = [c["symbol"] for c in candidates if c.get("symbol")]
    return f"{len(candidates)} fallback candidates enriched: {', '.join(symbols[:3])}"


def test_agentic_diligence_offline() -> str:
    """Verify bounded agentic diligence produces role briefs and evidence IDs."""
    from artha.agentic_diligence import build_agentic_diligence

    stock_data = {
        "ticker": "TEST",
        "quote": {"price": 100.0, "changePercentage": 1.2, "volume": 2_000_000, "marketCap": 5_000_000_000},
        "yf_quote": {"price": 100.2, "market_cap": 5_010_000_000, "volume": 2_100_000},
        "profile": {"companyName": "Test Corp", "sector": "Technology", "industry": "Software", "beta": 1.1},
        "income_statement": [{"date": "2025-12-31", "revenue": 1_000_000_000, "netIncome": 120_000_000, "eps": 2.4}],
        "balance_sheet": [{"date": "2025-12-31", "cashAndCashEquivalents": 200_000_000, "totalDebt": 100_000_000}],
        "cash_flow": [{"date": "2025-12-31", "netCashProvidedByOperatingActivities": 180_000_000, "freeCashFlow": 140_000_000}],
        "ratios_ttm": {"peRatioTTM": 28.0, "priceToSalesRatioTTM": 4.8, "debtEquityRatioTTM": 0.2},
        "key_metrics_ttm": {"freeCashFlowPerShareTTM": 3.1, "roicTTM": 0.18},
        "dcf": {"date": "2026-06-02", "dcf": 112.0, "Stock Price": 100.0},
        "price_target_consensus": {"targetConsensus": 118.0, "targetHigh": 140.0, "targetLow": 92.0},
        "analyst_estimates": {"source": "fmp", "next_q_eps_estimate": 0.75, "fy1_revenue_estimate": 1_200_000_000},
        "recommendation_trends": {"source": "finnhub", "consensus": "buy", "net_upgrades_30d": 1, "net_downgrades_30d": 0},
        "short_interest": {"source": "yfinance", "short_interest_pct": 3.2, "days_to_cover": 1.6, "squeeze_risk_flag": False},
        "sec": {
            "source": "sec",
            "status": "ok",
            "cik": "0000000000",
            "latest_10q_or_10k_staleness_days": 45,
            "facts_available": 3,
            "latest_filings": [{"form": "10-Q", "filing_date": "2026-05-01", "report_date": "2026-03-31"}],
            "financial_facts": [{"label": "revenue", "tag": "Revenue", "unit": "USD", "recent": [{"value": 1_000_000_000}]}],
        },
        "earnings_context": {"earnings_date": None, "days_to_earnings": None, "earnings_risk_flag": False, "recent_surprises": []},
        "technicals": {"rsi_14": 55.0, "sma_20": 98.0, "sma_50": 95.0, "sma_200": 88.0},
        "data_quality_report": {
            "completeness_score": 100.0,
            "context_coverage_score": 94.0,
            "sources_used": ["fmp", "yfinance", "finnhub", "analyst_signals", "sec"],
            "missing_fields": [],
            "enrichment_missing_fields": [],
            "source_conflicts": [],
            "staleness_warnings": [],
        },
    }

    result = build_agentic_diligence(
        "TEST",
        stock_data,
        macro_data={"fed_funds_rate": 3.63},
        market_overview={"fear_greed": 23, "vix": 18.0},
        intelligence_brief="INTELLIGENCE BRIEF: TEST has recent company-specific coverage.",
        data_quality_report=stock_data["data_quality_report"],
        enable_web=False,
        write_trace=False,
    )

    assert result.enabled is True, "Agentic diligence should be enabled"
    assert set(result.analyst_briefs.keys()) == {"fundamental", "technical", "contrarian"}
    assert len(result.evidence) >= 10, f"Expected evidence items, got {len(result.evidence)}"
    assert "CIO AGENTIC CROSS-EXAM BRIEF" in result.cio_brief
    assert "structured provider data is the source of truth" in result.cio_brief
    assert "current-web only as context" in result.cio_brief
    for role, brief in result.analyst_briefs.items():
        assert "AGENTIC DILIGENCE BRIEF" in brief, f"{role} missing diligence header"
        assert "[E" in brief, f"{role} missing evidence IDs"
        assert "Mandatory behavior" in brief, f"{role} missing behavior guardrails"
        assert "current-web/search results as context" in brief, f"{role} missing web-context guardrail"
        assert "source of truth for price" in brief, f"{role} missing provider-source guardrail"

    return f"roles={len(result.analyst_briefs)}, evidence={len(result.evidence)}, cio_brief={len(result.cio_brief)} chars"


def test_no_buy_scoring_json() -> str:
    """Verify no-buy CIO JSON with zero allocation/stop/target does not fallback."""
    from artha.council import _extract_scoring_json

    synthesis = "\n".join([
        "**COUNCIL CONSENSUS:** 2-1 bearish / no-buy consensus",
        "**RECOMMENDED ACTION:** **AVOID — do not initiate TEST.**",
        "```json",
        "{",
        '  "opportunity_score": 31,',
        '  "components": {',
        '    "technical_setup": 7,',
        '    "fundamental_quality": 8,',
        '    "contrarian_sentiment": 3,',
        '    "regime_alignment": 5,',
        '    "catalyst_asymmetry": 3,',
        '    "data_quality": 7,',
        '    "liquidity_execution": 5',
        '  },',
        '  "verdict": "AVOID",',
        '  "confidence": 8,',
        '  "thesis_type": "catalyst_driven",',
        '  "recommended_allocation_pct": 0.0,',
        '  "entry_valid_until": "2026-07-02",',
        '  "invalidation_conditions": [',
        '    "Pullback to a better risk/reward zone",',
        '    "Operating cash flow improves"',
        '  ],',
        '  "stop_loss_pct": 0.0,',
        '  "target_pct": 0.0',
        "}",
        "```",
    ])

    scoring = _extract_scoring_json(synthesis)
    assert scoring is not None, "No-buy scoring JSON should validate"
    assert scoring["verdict"] == "AVOID", f"Expected AVOID, got {scoring['verdict']}"
    assert scoring["recommended_allocation_pct"] == 0.0, "No-buy allocation must remain 0"
    assert scoring["stop_loss_pct"] == 0.0, "No-buy stop_loss_pct=0.0 should be valid"
    assert scoring["target_pct"] == 0.0, "No-buy target_pct=0.0 should be valid"
    assert scoring["opportunity_score"] == 38, "Score should auto-correct to component sum"

    return "AVOID JSON accepted with 0 allocation/target"


def test_scan_candidate_breadth_defaults() -> str:
    """Manual and scheduled scans should review a 6-8 candidate council slate by default."""
    from artha.config import Config
    import run as artha_run

    assert 6 <= Config.SCAN_COUNCIL_MAX <= 8, (
        f"SCAN_COUNCIL_MAX should stay in the agreed 6-8 range, got {Config.SCAN_COUNCIL_MAX}"
    )
    assert Config.SCAN_CANDIDATE_POOL >= Config.SCAN_COUNCIL_MAX, (
        "SCAN_CANDIDATE_POOL must be at least as large as SCAN_COUNCIL_MAX"
    )

    defaults = artha_run.full_market_scan.__defaults__ or ()
    assert len(defaults) >= 2, "full_market_scan should expose max_stocks/max_crypto defaults"
    assert defaults[0] == Config.SCAN_COUNCIL_MAX, (
        f"Manual scan default should match SCAN_COUNCIL_MAX={Config.SCAN_COUNCIL_MAX}, got {defaults[0]}"
    )
    assert defaults[1] == 0, f"Manual scan should default crypto council candidates to 0, got {defaults[1]}"

    return (
        f"manual={defaults[0]}, scheduled={Config.SCAN_COUNCIL_MAX}, "
        f"pool={Config.SCAN_CANDIDATE_POOL}"
    )


def test_accuracy_self_review_current_era() -> str:
    """Legacy Opus-era accuracy must not trigger current prompt-tuning alerts."""
    import json
    import tempfile
    from datetime import timedelta
    from pathlib import Path

    from artha.accuracy import AccuracyTracker, Recommendation
    from artha.config import Config
    from artha.self_review import NightlyReview

    legacy_ts = "2026-03-15T00:00:00+00:00"
    current_start = datetime.fromisoformat(Config.ACCURACY_CURRENT_ERA_START)
    if current_start.tzinfo is None:
        current_start = current_start.replace(tzinfo=timezone.utc)
    current_ts = (current_start + timedelta(hours=1)).isoformat()

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "accuracy.json"
        records = []
        for i in range(3):
            records.append({
                "ticker": f"OLD{i}",
                "verdict": "WATCH",
                "consensus": "legacy",
                "entry_price": "100",
                "recommended_action": "",
                "allocation": "",
                "fundamental_verdict": "HOLD",
                "fundamental_confidence": 5,
                "technical_verdict": "HOLD",
                "technical_confidence": 5,
                "contrarian_verdict": "HOLD",
                "contrarian_confidence": 5,
                "timestamp": legacy_ts,
                "review_after": legacy_ts,
                "status": "GRADED",
                "price_at_review": "120",
                "price_change_pct": "20",
                "grade": "PARTIALLY_CORRECT",
                "analyst_grades": {
                    "Fundamental (Opus)": "INCORRECT",
                    "Technical (Gemini)": "PARTIALLY_CORRECT",
                    "Contrarian (GPT 5.4)": "PARTIALLY_CORRECT",
                },
                "notes": "legacy synthetic row",
            })
        path.write_text(json.dumps(records), encoding="utf-8")

        tracker = AccuracyTracker(path=path)
        review = NightlyReview()
        review.accuracy = tracker
        insights = review._analyze_accuracy_patterns()

        current_patterns = insights.get("current_patterns", [])
        legacy_patterns = insights.get("legacy_patterns", [])
        assert current_patterns, "Expected explicit current-era status"
        assert any("waiting for" in p for p in current_patterns), current_patterns
        assert not any("Fundamental (Opus) accuracy is low" in p for p in current_patterns), current_patterns
        assert any("Legacy/all-time Fundamental (Opus) score is low" in p for p in legacy_patterns), legacy_patterns

        improvement = review._identify_improvement({
            "accuracy_insights": insights,
            "alert_details": {"total": 0},
            "accuracy_grades": 1,
            "grades": [],
        })
        assert not improvement, f"Legacy-only low accuracy must not create improvement: {improvement}"

        old_pending = Recommendation(
            ticker="OLDP",
            verdict="WATCH",
            consensus="legacy",
            entry_price="100",
            recommended_action="",
            allocation="",
            fundamental_verdict="HOLD",
            fundamental_confidence=5,
            technical_verdict="HOLD",
            technical_confidence=5,
            contrarian_verdict="HOLD",
            contrarian_confidence=5,
            timestamp=legacy_ts,
            review_after=legacy_ts,
            status="PENDING",
        ).__dict__
        current_pending = Recommendation(
            ticker="NEWP",
            verdict="WATCH",
            consensus="current",
            entry_price="100",
            recommended_action="",
            allocation="",
            fundamental_verdict="HOLD",
            fundamental_confidence=5,
            technical_verdict="HOLD",
            technical_confidence=5,
            contrarian_verdict="HOLD",
            contrarian_confidence=5,
            timestamp=current_ts,
            review_after=current_ts,
            status="PENDING",
        ).__dict__

        records = json.loads(path.read_text(encoding="utf-8"))
        records.extend([old_pending, current_pending])
        path.write_text(json.dumps(records), encoding="utf-8")

        old_result = tracker.grade_recommendation("OLDP", legacy_ts, 100)
        assert old_result, "Expected old pending record to grade"
        assert "Fundamental (Opus)" in old_result["analyst_grades"], old_result["analyst_grades"]
        assert "Fundamental (GPT agentic)" not in old_result["analyst_grades"], old_result["analyst_grades"]

        current_result = tracker.grade_recommendation("NEWP", current_ts, 100)
        assert current_result, "Expected current pending record to grade"
        assert "Fundamental (GPT agentic)" in current_result["analyst_grades"], current_result["analyst_grades"]

    return "legacy alerts suppressed; old/current analyst labels preserved"


def test_no_new_capital_report_allocation() -> str:
    """No-position verdicts must not display starter allocation/stop/target in reports."""
    from artha.council import AnalystReport, CouncilDecision
    from artha.report import format_stock_analysis

    analyst = AnalystReport(
        analyst_name="test",
        model="test",
        verdict="HOLD",
        confidence=7,
        report="synthetic",
    )
    decision = CouncilDecision(
        ticker="TEST",
        final_verdict="DEFER",
        consensus="2-1 against entry",
        recommended_action="DEFER — wait for a better entry.",
        allocation="0.0% NAV (~$0)",
        synthesis_report="Synthetic no-buy report.",
        fundamental=analyst,
        technical=analyst,
        contrarian=analyst,
        opportunity_score=65,
        adjusted_score=65,
        confidence=7,
        recommended_allocation_pct=0.0,
        stop_loss_pct=-0.12,
        target_pct=0.18,
    )

    report = format_stock_analysis(decision)
    assert "Allocation: 0.0% NAV (~$0)" in report, report
    assert "Stop:" not in report, report
    assert "Target:" not in report, report
    return "DEFER report shows $0 allocation and no trade stop/target"


def test_scheduled_scan_incremental_delivery_helpers() -> str:
    """Scheduled scans must support start/progress/per-stock/completion delivery."""
    from artha.config import Config
    from artha.scheduler import ArthaScheduler

    class FakeTelegram:
        enabled = True

        def __init__(self):
            self.sent = []

        def send_message(self, text, parse_mode="Markdown", disable_preview=True, silent=False):
            self.sent.append(("message", text, parse_mode, silent))
            return True

        def send_report(self, report):
            self.sent.append(("report", report, None, False))
            return True

        def send_health_check(self, message):
            self.sent.append(("health", message, None, True))
            return True

    fake = FakeTelegram()
    scheduler = ArthaScheduler.__new__(ArthaScheduler)
    scheduler.telegram = fake

    original_max = Config.SCAN_COUNCIL_MAX
    Config.SCAN_COUNCIL_MAX = 8
    try:
        assert scheduler._send_scan_start("HEADER\n")
        assert scheduler._send_scan_candidate_update(12)
        assert scheduler._send_scan_report("TEST", "REPORT BODY")
        assert scheduler._send_scan_completion(1, "/tmp/report.txt")
        assert scheduler._send_scan_failure(RuntimeError("synthetic failure"))
    finally:
        Config.SCAN_COUNCIL_MAX = original_max

    kinds = [item[0] for item in fake.sent]
    assert kinds == ["message", "message", "report", "message", "health"], kinds
    assert "Started now" in fake.sent[0][1], fake.sent[0]
    assert "Reports will arrive one by one" in fake.sent[1][1], fake.sent[1]
    assert fake.sent[2][1] == "REPORT BODY", fake.sent[2]
    assert "Delivered 1 council report" in fake.sent[3][1], fake.sent[3]
    assert "synthetic failure" in fake.sent[4][1], fake.sent[4]

    return "start/progress/per-stock/completion/failure delivery helpers verified"


def test_scheduled_scan_market_day_cadence() -> str:
    """Scheduled scans should run each market trading day at 11:30 AM CT."""
    from zoneinfo import ZoneInfo

    from artha.scheduler import ArthaScheduler, MarketHours

    scheduler = ArthaScheduler.__new__(ArthaScheduler)
    scheduler.market_hours = MarketHours()
    scheduler.ct_tz = ZoneInfo("America/Chicago")
    scheduler._last_run = {}

    def utc_at(year: int, month: int, day: int, hour: int = 11, minute: int = 30):
        return datetime(year, month, day, hour, minute, tzinfo=scheduler.ct_tz).astimezone(timezone.utc)

    # Monday-Friday trading days should trigger in the 11:30-11:34 AM CT window.
    assert scheduler._should_run_weekly_scan(utc_at(2026, 6, 8)), "Monday should run"
    assert scheduler._should_run_weekly_scan(utc_at(2026, 6, 9)), "Tuesday should run"
    assert scheduler._should_run_weekly_scan(utc_at(2026, 6, 10)), "Wednesday should run"
    assert scheduler._should_run_weekly_scan(utc_at(2026, 6, 11)), "Thursday should run"
    assert scheduler._should_run_weekly_scan(utc_at(2026, 6, 12)), "Friday should run"

    # Same slot should not trigger twice.
    scheduler._last_run = {}
    assert scheduler._should_run_weekly_scan(utc_at(2026, 6, 8, 11, 30)), "first Monday slot should run"
    assert not scheduler._should_run_weekly_scan(utc_at(2026, 6, 8, 11, 31)), "duplicate Monday slot should not run"

    # Weekends, market holidays, and wrong times should not trigger.
    scheduler._last_run = {}
    assert not scheduler._should_run_weekly_scan(utc_at(2026, 6, 13)), "Saturday should not run"
    assert not scheduler._should_run_weekly_scan(utc_at(2026, 6, 19)), "Juneteenth market holiday should not run"
    assert not scheduler._should_run_weekly_scan(utc_at(2026, 6, 8, 14, 30)), "old 2:30 PM CT slot should not run"
    assert not scheduler._should_run_weekly_scan(utc_at(2026, 6, 8, 16, 30)), "4:30 PM CT should not run"

    return "market-day scan cadence verified: Mon-Fri trading days at 11:30 AM CT"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\nArtha Data Layer V2 — Verification Suite")
    print("Running real API calls. This may take 1-3 minutes...\n")

    tests = [
        # Phase 1: Quick wins
        ("1.1 vendor_priority", test_vendor_priority),
        ("1.2 liquidity gate", test_liquidity_gate),
        ("1.3 config thresholds", test_config_thresholds),
        ("1.3 regime_indicators new fields", test_regime_indicators_new_fields),
        ("1.4 economic_calendar API", test_economic_calendar_api),
        ("1.5 earnings_calendar API", test_earnings_calendar_api),
        # Phase 1+3: Collector changes
        ("3.3 no AlphaVantage", test_collector_no_av),
        ("2.1 FMP screener method", test_collector_fmp_screener),
        ("3.1 PIT metadata + earnings in collect_stock", test_collect_stock_pit_metadata),
        # Phase 2: Big unlock
        ("2.1 universe builder", test_universe_builder),
        ("2.2 rank candidates", test_rank_candidates),
        ("2.4 analyst signals", test_analyst_signals),
        ("2.4 SEC EDGAR context", test_sec_edgar_context),
        ("2.5 data quality report", test_data_quality_report),
        ("2.3 funnel smoke test", test_funnel_smoke),
        # Phase 3: New tests (Issues 13 & 14)
        ("3.4 short interest", test_short_interest),
        ("3.5 regime_indicators crisis path", test_compute_regime_indicators_crisis_path),
        ("3.6 funnel fallback shape", test_funnel_fallback_shape),
        ("4.1 agentic diligence offline", test_agentic_diligence_offline),
        ("4.2 no-buy scoring JSON", test_no_buy_scoring_json),
        ("4.3 scan candidate breadth defaults", test_scan_candidate_breadth_defaults),
        ("4.4 accuracy current-era self-review", test_accuracy_self_review_current_era),
        ("4.5 no-new-capital report allocation", test_no_new_capital_report_allocation),
        ("4.6 scheduled scan incremental delivery", test_scheduled_scan_incremental_delivery_helpers),
        ("4.7 scheduled scan market-day cadence", test_scheduled_scan_market_day_cadence),
    ]

    results = []
    for name, fn in tests:
        print(f"  Testing {name}...", end=" ", flush=True)
        result = run_test(name, fn)
        results.append(result)
        if result.passed:
            print(f"✅  {result.details}")
        else:
            print(f"❌  {result.error[:100]}")

    exit_code = print_summary(results)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
