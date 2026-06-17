#!/usr/bin/env python3
"""Artha — AI-Powered Financial Analysis Agent

Main entry point for running analyses and generating reports.

Usage:
    python run.py scan                        # Full market scan + council analysis (default: 8 investigation candidates)
    python run.py scan 6                     # Scan with 6 council investigation candidates
    python run.py analyze AAPL MSFT GOOGL    # Analyze specific stocks
    python run.py overview                    # Market overview report
    python run.py portfolio                   # Portfolio status
    python run.py check                       # One-shot portfolio health check + alerts
    python run.py diagnose [--telegram]       # Calibration diagnosis + guarded fix proposals
    python run.py supervise [--telegram]      # Supervisor health + shadow-rule learning check
    python run.py execution-readiness [--telegram]  # Robinhood-ready wiring/guardrail check
    python run.py broker-router-preview [--assume-market-open] [--no-persist]  # Real FMP/YF pre-Council router preview
    python run.py propose-order AAPL --side buy --notional 25 --limit 300 --price 300 --volume 50000000 --bid 299.9 --ask 300.1
    python run.py robinhood-action artha:review:...  # Resolve a Telegram action token into MCP args
    python run.py robinhood-auto-buy-queue-status     # List queued auto-buy actions without mutating gate state
    python run.py robinhood-record-review ACTION_ID --review-file review.json --tradability-file tradability.json
    python run.py robinhood-auto-buy-agentic-clearance ACTION_ID --quote-file quote.json --review-file review.json --tradability-file tradability.json
    python run.py robinhood-final-clearance ACTION_ID
    python run.py robinhood-record-submission ACTION_ID --file place-response.json
    python run.py launchd-plists              # Generate macOS launchd plist templates
    python run.py monitor                     # Start monitoring daemon scheduler
    python run.py test-apis                   # Test all API connections
"""
import asyncio
import json
import sys
import logging
import tempfile
import os
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path

import requests

from artha.config import Config
from artha.collector import DataCollector
from artha.council import ArthaCouncil
from artha.report import format_stock_analysis, format_market_overview
from artha.portfolio import Portfolio
from artha.scanner import MarketScanner
from artha.monitor import PriceMonitor
from artha.scheduler import ArthaScheduler
from artha.journal import DecisionJournal
from artha.portfolio_state import PortfolioStateEngine
from artha.accuracy import AccuracyTracker, Recommendation

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("artha")


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically write UTF-8 text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=".report_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _candidate_scan_price(candidate: dict) -> float | None:
    def as_float(value):
        if value is None or value == "":
            return None
        try:
            return float(str(value).replace(",", "").replace("%", ""))
        except Exception:
            return None

    for key in ("price", "lastPrice", "last_price", "current_price", "previous_close"):
        value = as_float((candidate or {}).get(key))
        if value is not None and value > 0:
            return value
    quote = (candidate or {}).get("quote")
    if isinstance(quote, dict):
        for key in ("price", "lastPrice", "last_price"):
            value = as_float(quote.get(key))
            if value is not None and value > 0:
                return value
    return None


def _active_defer_watch_map(journal: DecisionJournal) -> dict:
    try:
        return {
            str(w.get("ticker") or "").upper(): w
            for w in journal.get_active_defer_watches()
            if w.get("ticker")
        }
    except Exception as exc:
        logger.warning("Could not load active DEFER watches for scan skip logic: %s", exc)
        return {}


def _defer_watch_scan_skip(candidate: dict, active_watches: dict) -> dict:
    ticker = str((candidate or {}).get("symbol") or "").upper().strip()
    if not Config.SCAN_DEFER_WATCH_SKIP_ENABLED or not ticker:
        return {"skip": False, "reason": "disabled_or_missing_ticker"}
    watch = active_watches.get(ticker)
    if not watch:
        return {"skip": False, "reason": "no_active_watch"}

    from artha.defer_watchlist import scan_skip_for_defer_watch

    return scan_skip_for_defer_watch(
        watch,
        _candidate_scan_price(candidate),
        candidate=candidate,
        buffer_pct=Config.SCAN_DEFER_WATCH_SKIP_BUFFER_PCT,
        major_move_pct=Config.SCAN_DEFER_WATCH_SKIP_MAJOR_MOVE_PCT,
    )


def _record_accuracy_recommendation(
    tracker: AccuracyTracker,
    ticker: str,
    decision,
    stock_data: dict,
) -> None:
    """Record council output for 30-day accuracy tracking."""
    quote = stock_data.get("quote") or {}
    yf_quote = stock_data.get("yf_quote") or {}
    price = quote.get("price", yf_quote.get("price", 0))
    fun = getattr(decision, "fundamental", None)
    tech = getattr(decision, "technical", None)
    cont = getattr(decision, "contrarian", None)

    rec = Recommendation(
        ticker=ticker.upper(),
        verdict=getattr(decision, "final_verdict", "") or "",
        consensus=getattr(decision, "consensus", "") or "",
        entry_price=str(price or 0),
        recommended_action=getattr(decision, "recommended_action", "") or "",
        allocation=getattr(decision, "allocation", "") or "",
        fundamental_verdict=getattr(fun, "verdict", "") if fun else "",
        fundamental_confidence=getattr(fun, "confidence", 0) if fun else 0,
        technical_verdict=getattr(tech, "verdict", "") if tech else "",
        technical_confidence=getattr(tech, "confidence", 0) if tech else 0,
        contrarian_verdict=getattr(cont, "verdict", "") if cont else "",
        contrarian_confidence=getattr(cont, "confidence", 0) if cont else 0,
    )
    tracker.record_recommendation(rec)


def test_apis():
    """Test all API connections and report status."""
    print("\n🔌 Testing API Connections...\n")

    missing = Config.validate()
    if missing:
        print("⚠️  Missing API keys:")
        for m in missing:
            print(f"  • {m}")
        print()

    collector = DataCollector()
    results = {}

    # Test FMP
    print("Testing FMP...", end=" ")
    try:
        data = collector.fmp.quote("AAPL")
        if data and isinstance(data, dict) and "price" in data:
            print(f"✅ (AAPL = ${data['price']})")
            results["FMP"] = True
        else:
            print(f"⚠️  Got response but unexpected format")
            results["FMP"] = False
    except Exception as e:
        print(f"❌ {e}")
        results["FMP"] = False

    # Test Massive
    print("Testing Massive...", end=" ")
    try:
        data = collector.massive.quote("AAPL")
        if data and isinstance(data, dict) and data.get("price"):
            source = data.get("source", "massive")
            print(f"✅ (AAPL = ${data['price']} via {source})")
            results["Massive"] = True
        elif not Config.MASSIVE_API_KEY:
            print("⚠️  Missing MASSIVE_API_KEY")
            results["Massive"] = False
        else:
            print("⚠️  Reachable key could not return AAPL market data")
            results["Massive"] = False
    except Exception as e:
        print(f"❌ {e}")
        results["Massive"] = False

    # Test Finnhub
    print("Testing Finnhub...", end=" ")
    try:
        data = collector.finnhub.market_news()
        if data and isinstance(data, list) and len(data) > 0:
            print(f"✅ ({len(data)} news items)")
            results["Finnhub"] = True
        else:
            print(f"⚠️  Empty or unexpected response")
            results["Finnhub"] = False
    except Exception as e:
        print(f"❌ {e}")
        results["Finnhub"] = False

    # Test Alpha Vantage
    print("Testing Alpha Vantage...", end=" ")
    try:
        # Use direct endpoint ping so strict parser does not hide throttling vs invalid key.
        resp = requests.get(
            Config.ALPHA_VANTAGE_BASE_URL,
            params={
                "function": "RSI",
                "symbol": "AAPL",
                "interval": "daily",
                "time_period": "14",
                "series_type": "close",
                "apikey": Config.ALPHA_VANTAGE_API_KEY,
            },
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict) and "Technical Analysis: RSI" in payload:
            print("✅")
            results["Alpha Vantage"] = True
        elif isinstance(payload, dict) and any(payload.get(k) for k in ("Note", "Information")):
            print("⚠️  Reachable but rate-limited (25/day free tier)")
            results["Alpha Vantage"] = True
        else:
            msg = payload.get("Error Message", "Unexpected payload") if isinstance(payload, dict) else "Unexpected payload"
            print(f"❌ {msg}")
            results["Alpha Vantage"] = False
    except Exception as e:
        print(f"❌ {e}")
        results["Alpha Vantage"] = False

    # Test CoinGecko
    print("Testing CoinGecko...", end=" ")
    try:
        data = collector.coingecko.price("bitcoin")
        if data and isinstance(data, dict) and "bitcoin" in data:
            btc_price = data["bitcoin"].get("usd", "?")
            if isinstance(btc_price, (int, float)):
                print(f"✅ (BTC = ${btc_price:,.0f})")
            else:
                print(f"✅ (BTC = ${btc_price})")
            results["CoinGecko"] = True
        else:
            print("⚠️  Unexpected response")
            results["CoinGecko"] = False
    except Exception as e:
        print(f"❌ {e}")
        results["CoinGecko"] = False

    # Test FRED
    print("Testing FRED...", end=" ")
    try:
        data = collector.fred.fed_funds_rate()
        if data and isinstance(data, dict) and "observations" in data:
            obs = data["observations"]
            rate = obs[0]["value"] if obs and isinstance(obs[0], dict) else "?"
            print(f"✅ (Fed rate = {rate}%)")
            results["FRED"] = True
        else:
            print("⚠️  Unexpected response")
            results["FRED"] = False
    except Exception as e:
        print(f"❌ {e}")
        results["FRED"] = False

    # Test Fear & Greed
    print("Testing Fear & Greed Index...", end=" ")
    try:
        from artha.collector import get_fear_greed_index
        data = get_fear_greed_index()
        if data:
            print(f"✅ (Score: {data['value']} — {data['label']})")
            results["Fear & Greed"] = True
        else:
            print("⚠️  No data")
            results["Fear & Greed"] = False
    except Exception as e:
        print(f"❌ {e}")
        results["Fear & Greed"] = False

    # Test yfinance
    print("Testing yfinance...", end=" ")
    try:
        data = collector.yf.quote("AAPL")
        if data and data.get("price"):
            print(f"✅ (AAPL = ${data['price']})")
            results["yfinance"] = True
        else:
            print("⚠️  Unexpected response")
            results["yfinance"] = False
    except Exception as e:
        print(f"❌ {e}")
        results["yfinance"] = False

    # Summary
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n{'=' * 40}")
    print(f"Results: {passed}/{total} APIs working")
    print(f"{'=' * 40}\n")

    return results


def analyze_stocks(tickers: list[str]):
    """Run full council analysis on given tickers."""
    collector = DataCollector()
    council = ArthaCouncil()
    journal = DecisionJournal()
    accuracy = AccuracyTracker()
    portfolio_state = PortfolioStateEngine()
    session_id = f"manual-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}-{uuid4().hex[:8]}"
    analyzed: list[str] = []
    report_paths: list[str] = []

    # Save portfolio snapshot at the start of the run.
    try:
        bundle = portfolio_state.build_state_bundle()
        snap = bundle["snapshot"]
        journal.save_snapshot(
            total_value=snap["total_value"],
            cash=snap["cash"],
            holdings_json=snap["holdings_json"],
            summary=snap["summary"],
            timestamp=snap["timestamp"],
        )
    except Exception as e:
        logger.warning(f"Could not save startup portfolio snapshot: {e}")

    # Collect macro data once (shared across all analyses)
    logger.info("📡 Collecting macro data...")
    macro_data = collector.collect_macro()

    logger.info("📡 Collecting market overview...")
    market_overview = collector.collect_market_overview()

    for ticker in tickers:
        print(f"\n{'=' * 50}")
        print(f"  Analyzing ${ticker.upper()}")
        print(f"{'=' * 50}\n")

        # Collect stock data
        stock_data = collector.collect_stock(ticker.upper())

        # Run council
        decision = council.analyze_stock(stock_data, macro_data, market_overview)

        if decision:
            report = format_stock_analysis(decision)
            print(report)
            analyzed.append(ticker.upper())

            # Save report
            report_dir = Path(__file__).parent / "data" / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_file = report_dir / f"{ticker.upper()}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.txt"
            _atomic_write_text(report_file, report)
            logger.info(f"Report saved to {report_file}")
            report_paths.append(str(report_file))

            # Save recommendation journal entry.
            try:
                quote = stock_data.get("quote") or {}
                yf_quote = stock_data.get("yf_quote") or {}
                price = quote.get("price", yf_quote.get("price"))
                confidence = round(
                    (decision.fundamental.confidence + decision.technical.confidence + decision.contrarian.confidence) / 3
                )
                journal.save_recommendation(
                    session_id=session_id,
                    ticker=ticker.upper(),
                    action=decision.final_verdict,
                    rationale=decision.synthesis_report,
                    confidence=int(confidence),
                    price_at_recommendation=float(price) if isinstance(price, (int, float)) else None,
                    conditions=decision.recommended_action,
                    status="open",
                    outcome="unknown",
                    outcome_notes="",
                )
            except Exception as e:
                logger.warning(f"Failed to save recommendation for {ticker}: {e}")

            try:
                _record_accuracy_recommendation(accuracy, ticker.upper(), decision, stock_data)
            except Exception as e:
                logger.warning(f"Failed to record accuracy tracking for {ticker}: {e}")
        else:
            print(f"❌ Council analysis failed for {ticker}")

    # Save session log at the end of the run.
    try:
        journal.save_session(
            session_type="manual",
            tickers_analyzed=",".join(analyzed),
            report_path=",".join(report_paths),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.warning(f"Could not save manual session log: {e}")


def market_overview():
    """Generate market overview / weekly brief."""
    collector = DataCollector()

    logger.info("📡 Collecting market overview...")
    market_data = collector.collect_market_overview()

    logger.info("📡 Collecting macro data...")
    macro_data = collector.collect_macro()

    fear_greed = market_data.get("fear_greed")

    report = format_market_overview(market_data, macro_data, fear_greed)
    print(report)


def portfolio_status():
    """Show current portfolio status."""
    portfolio = Portfolio.load()
    summary = portfolio.summary()

    if summary["num_positions"] == 0:
        print("\n💼 Portfolio is empty. No positions yet.")
        print("Run 'python run.py analyze TICKER' to get investment recommendations.\n")
        return

    print(f"\n💼 Portfolio: {summary['num_positions']} positions")
    print(f"   Total invested: ${summary['total_invested']:,.2f}")
    print(f"   Stocks: ${summary['stocks_invested']:,.2f}")
    print(f"   Crypto: ${summary['crypto_invested']:,.2f}")
    print(f"   Transactions: {summary['num_transactions']}\n")


def full_market_scan(
    max_stocks: int = Config.SCAN_COUNCIL_MAX,
    max_crypto: int = 0,
    send_telegram: bool = False,
):
    """Run full market scan with MROL regime intelligence → council analysis."""
    from artha.regime import run_regime_council, format_regime_report
    from artha.scanner import MarketScanner

    scanner = MarketScanner()
    collector = DataCollector()
    council = ArthaCouncil()
    journal = DecisionJournal()
    accuracy = AccuracyTracker()
    portfolio_state = PortfolioStateEngine()
    session_id = f"scan-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}-{uuid4().hex[:8]}"
    analyzed: list[str] = []
    report_paths: list[str] = []
    telegram = None
    telegram_sent_count = 0
    if send_telegram:
        from artha.telegram import TelegramSender

        telegram = TelegramSender()
        if telegram.enabled:
            telegram.send_message(
                "ARTHA CLI SCAN STARTED\n"
                "━━━━━━━━━━━━━━━\n"
                f"Reviewing up to {max_stocks} investigation candidate(s). "
                "Council reports will be sent as they finish.",
                parse_mode=None,
            )
        else:
            logger.warning("Telegram requested for CLI scan but TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID are not configured")

    # Save portfolio snapshot at the start of the run.
    try:
        bundle = portfolio_state.build_state_bundle()
        snap = bundle["snapshot"]
        journal.save_snapshot(
            total_value=snap["total_value"],
            cash=snap["cash"],
            holdings_json=snap["holdings_json"],
            summary=snap["summary"],
            timestamp=snap["timestamp"],
        )
    except Exception as e:
        logger.warning(f"Could not save startup portfolio snapshot: {e}")

    # ===== PHASE 1: MACRO DATA =====
    print("\n📡 PHASE 1: COLLECTING MACRO DATA\n")
    macro_data = collector.collect_macro()

    # ===== PHASE 2: MROL — Macro Regime & Opportunity Layer =====
    print("\n🌍 PHASE 2: MACRO REGIME INTELLIGENCE (MROL)\n")
    try:
        regime_packet = run_regime_council(macro_data)
        regime_report = format_regime_report(regime_packet)
        print(regime_report)
    except Exception as e:
        logger.error(f"MROL failed: {e}. Falling back to standard scan.")
        regime_packet = None
        regime_report = ""

    # ===== PHASE 3: CANDIDATE GENERATION =====
    print("\n🔍 PHASE 3: CANDIDATE GENERATION\n")

    market_overview = collector.collect_market_overview()

    funnel_packet = regime_packet or {
        "regime_type": "neutral",
        "event_overlays": [],
    }
    candidate_pool_size = max(
        max_stocks,
        Config.SCAN_CANDIDATE_POOL,
        max_stocks + max(Config.SCAN_DEFER_SKIP_BACKFILL_EXTRA, 0),
    )
    candidates = scanner.get_funnel_candidates(
        funnel_packet,
        max_candidates=candidate_pool_size,
    )
    scan_result = {
        "scan_time": datetime.now(timezone.utc).isoformat(),
        "market_snapshot": market_overview,
        "stock_candidates": candidates,
        "crypto_candidates": [],
        "fear_greed": market_overview.get("fear_greed"),
        "candidate_source": "promotion_funnel",
    }

    print(f"  🏛️ PromotionFunnel investigation candidates ({len(candidates)}):")
    print("  These are not buy recommendations yet; they are the most interesting names for council review.")
    for c in candidates:
        reason = c.get("regime_reason") or c.get("source") or "Funnel-ranked opportunity"
        score = c.get("funnel_score", c.get("combined_score", ""))
        score_text = f" | score {score}" if score != "" else ""
        sleeve = c.get("primary_alpha_sleeve") or ""
        sleeve_text = f" | sleeve {sleeve}" if sleeve else ""
        print(f"    • ${c.get('symbol', '?')} — {reason}{score_text}{sleeve_text}")

    if telegram and telegram.enabled:
        telegram.send_message(
            "ARTHA CLI SCAN PROGRESS\n"
            "━━━━━━━━━━━━━━━\n"
            f"Funnel found {len(candidates)} finalist(s). "
            f"Reviewing up to {max_stocks} with the council.",
            parse_mode=None,
        )

    # ===== PHASE 4: COUNCIL ANALYSIS =====
    print(f"\n🏛️ PHASE 4: COUNCIL DEEP DILIGENCE — CANDIDATES → DECISIONS\n")

    # Check for already-completed reports today (resume support)
    from datetime import date as date_type
    today_str = date_type.today().strftime("%Y%m%d")
    report_dir = Path(__file__).parent / "data" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    already_done_today = set()
    for existing in report_dir.glob(f"*_{today_str}_*.txt"):
        already_done_today.add(existing.stem.split("_")[0])
    if already_done_today:
        logger.info(f"  ♻️ Resume: found reports for {already_done_today} today, skipping")

    reports = []
    skipped_defer_watches = []
    active_defer_watches = _active_defer_watch_map(journal)
    reviewed_count = 0
    for candidate in candidates:
        if reviewed_count >= max_stocks:
            break
        ticker = candidate.get("symbol", "")
        if not ticker:
            continue
        ticker = str(ticker).upper().strip()

        # Skip broad ETFs from individual council analysis
        if ticker in ("SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "FXAIX"):
            continue

        skip_decision = _defer_watch_scan_skip(candidate, active_defer_watches)
        if skip_decision.get("skip"):
            skip_row = {"ticker": ticker, **skip_decision}
            skipped_defer_watches.append(skip_row)
            price = skip_decision.get("price")
            low = skip_decision.get("zone_low")
            high = skip_decision.get("zone_high")
            dist = skip_decision.get("distance_pct")
            price_text = f"${price:,.2f}" if isinstance(price, (int, float)) else "price unknown"
            zone_text = f"${low:,.2f}-${high:,.2f}" if isinstance(low, (int, float)) and isinstance(high, (int, float)) else "zone unknown"
            dist_text = f", {dist:.1f}% away" if isinstance(dist, (int, float)) else ""
            print(f"  ⏭️ ${ticker} — active DEFER/WATCH zone skip: {price_text} vs {zone_text}{dist_text}")
            logger.info(
                "defer_zone_skip ticker=%s price=%s zone=%s-%s distance_pct=%s reason=%s watch_id=%s",
                ticker,
                skip_decision.get("price"),
                skip_decision.get("zone_low"),
                skip_decision.get("zone_high"),
                skip_decision.get("distance_pct"),
                skip_decision.get("reason"),
                skip_decision.get("watch_id"),
            )
            continue

        # Skip if already analyzed today (resume after crash)
        if ticker in already_done_today:
            print(f"  ♻️ ${ticker} — already analyzed today, skipping")
            analyzed.append(ticker)
            reviewed_count += 1
            continue

        reviewed_count += 1
        print(f"\n{'=' * 50}")
        sleeve_note = candidate.get("primary_alpha_sleeve") or candidate.get("source", "")
        regime_note = f" [{sleeve_note}]" if sleeve_note else ""
        print(f"  🏛️ Council Analysis: ${ticker}{regime_note}")
        print(f"  {candidate.get('regime_reason', '')}")
        print(f"{'=' * 50}\n")

        stock_data = collector.collect_stock(ticker)

        # Pass regime context to council if available
        regime_context = regime_packet.to_context_string() if regime_packet else ""
        fg_payload = market_overview.get("fear_greed") or {}
        decision = council.analyze_stock(
            stock_data,
            macro_data,
            market_overview,
            regime_context=regime_context,
            fear_greed=int(fg_payload.get("value", 50) or 50),
        )

        if decision:
            report = format_stock_analysis(decision)
            print(report)
            reports.append({"ticker": ticker, "decision": decision, "report": report})
            analyzed.append(ticker)

            # Save report
            report_dir = Path(__file__).parent / "data" / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_file = report_dir / f"{ticker}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.txt"
            _atomic_write_text(report_file, report)
            logger.info(f"Report saved to {report_file}")
            report_paths.append(str(report_file))
            if telegram and telegram.enabled:
                if telegram.send_report(report):
                    telegram_sent_count += 1
                else:
                    logger.error("Failed to send CLI scan report for %s to Telegram", ticker)

            # Save journal entry
            try:
                quote = stock_data.get("quote") or {}
                yf_quote = stock_data.get("yf_quote") or {}
                price = quote.get("price", yf_quote.get("price"))
                confidence = round(
                    (decision.fundamental.confidence + decision.technical.confidence + decision.contrarian.confidence) / 3
                )
                journal.save_recommendation(
                    session_id=session_id,
                    ticker=ticker.upper(),
                    action=decision.final_verdict,
                    rationale=decision.synthesis_report,
                    confidence=int(confidence),
                    price_at_recommendation=float(price) if isinstance(price, (int, float)) else None,
                    conditions=decision.recommended_action,
                    status="open",
                    outcome="unknown",
                    outcome_notes="",
                )
            except Exception as e:
                logger.warning(f"Failed to save recommendation for {ticker}: {e}")

            try:
                _record_accuracy_recommendation(accuracy, ticker, decision, stock_data)
            except Exception as e:
                logger.warning(f"Failed to record accuracy tracking for {ticker}: {e}")
        else:
            print(f"❌ Council analysis failed for {ticker}")
            if telegram and telegram.enabled:
                telegram.send_health_check(
                    "ARTHA CLI SCAN CANDIDATE SKIPPED\n"
                    "━━━━━━━━━━━━━━━\n"
                    f"{ticker} did not produce a usable council decision."
                )

    if skipped_defer_watches:
        print("\n⏭️ DEFER-zone skip summary:")
        for item in skipped_defer_watches:
            low = item.get("zone_low")
            high = item.get("zone_high")
            price = item.get("price")
            dist = item.get("distance_pct")
            price_text = f"${price:,.2f}" if isinstance(price, (int, float)) else "price unknown"
            zone_text = f"${low:,.2f}-${high:,.2f}" if isinstance(low, (int, float)) and isinstance(high, (int, float)) else "zone unknown"
            dist_text = f", {dist:.1f}% away" if isinstance(dist, (int, float)) else ""
            print(f"  • ${item.get('ticker')}: {price_text} vs {zone_text}{dist_text}; watch remains active")
        if telegram and telegram.enabled:
            summary_lines = [
                "ARTHA CLI DEFER-ZONE SKIPS",
                "━━━━━━━━━━━━━━━",
            ]
            for item in skipped_defer_watches[:8]:
                low = item.get("zone_low")
                high = item.get("zone_high")
                price = item.get("price")
                dist = item.get("distance_pct")
                price_text = f"${price:,.2f}" if isinstance(price, (int, float)) else "price unknown"
                zone_text = f"${low:,.2f}-${high:,.2f}" if isinstance(low, (int, float)) and isinstance(high, (int, float)) else "zone unknown"
                dist_text = f", {dist:.1f}% away" if isinstance(dist, (int, float)) else ""
                summary_lines.append(f"- ${item.get('ticker')}: {price_text} vs {zone_text}{dist_text}")
            if len(skipped_defer_watches) > 8:
                summary_lines.append(f"- ...and {len(skipped_defer_watches) - 8} more")
            telegram.send_health_check("\n".join(summary_lines))

    # ===== PHASE 5: FINAL SUMMARY =====
    print(f"\n{'=' * 50}")
    print("  📋 ARTHA COUNCIL REPORT COMPLETE")
    print(f"{'=' * 50}\n")

    fg = scan_result.get("fear_greed", {}) or {}
    print(f"  Market Mood: Equity Sentiment = {fg.get('value', '?')}" 
          f" ({fg.get('label', '?')})")
    if regime_packet:
        print(f"  Regime: {regime_packet.base_regime_label} "
              f"({regime_packet.base_regime_confidence:.0%})")
        if regime_packet.event_overlays:
            for o in regime_packet.event_overlays:
                otype = o.get("type", "")
                from artha.regime_mapping import REGIME_TAXONOMY
                label = REGIME_TAXONOMY.get(otype, {}).get("label", otype)
                print(f"  Event: {label} ({o.get('confidence', 0):.0%})")
    print(f"  Investigation Candidates Returned: {len(candidates)}")
    print(f"  Council Candidates Analyzed: {len(reports)}")

    buy_like = {"BUY", "STARTER", "TACTICAL_BUY", "ACCUMULATE", "ADD", "STRONG BUY"}
    watch_like = {"WATCH", "DEFER", "HOLD"}
    reject_like = {"AVOID", "SELL", "TRIM"}

    def _decision_verdict(item: dict) -> str:
        return str(getattr(item["decision"], "final_verdict", "") or "").upper()

    approved = [r for r in reports if _decision_verdict(r) in buy_like]
    watchlist = [r for r in reports if _decision_verdict(r) in watch_like]
    rejected = [r for r in reports if _decision_verdict(r) in reject_like]
    action_mix: dict[str, int] = {}
    for r in reports:
        action = _decision_verdict(r) or "UNKNOWN"
        action_mix[action] = action_mix.get(action, 0) + 1
    mix_text = ", ".join(f"{k}={v}" for k, v in sorted(action_mix.items())) or "none"

    print(
        "  Final Outcomes: "
        f"{len(approved)} approved buy | {len(watchlist)} watch/defer | {len(rejected)} avoid/trim/sell"
    )
    print(f"  Action Mix: {mix_text}")
    if approved:
        picks = ", ".join(f"{r['ticker']}={_decision_verdict(r)}" for r in approved)
        print(f"  Final Buy Recommendations: {picks}")
    else:
        print("  Final Buy Recommendations: NONE today — council reviewed candidates but rejected/deferred current entries.")
    print()

    if telegram and telegram.enabled:
        if telegram_sent_count:
            completion = (
                "ARTHA CLI SCAN COMPLETE\n"
                "━━━━━━━━━━━━━━━\n"
                f"Delivered {telegram_sent_count} council report(s). "
                f"Saved {len(report_paths)} local report file(s)."
            )
        else:
            completion = (
                "ARTHA CLI SCAN COMPLETE\n"
                "━━━━━━━━━━━━━━━\n"
                "No council reports were delivered. Check local logs/report files."
            )
        telegram.send_message(completion, parse_mode=None)

    # Save session log
    try:
        journal.save_session(
            session_type="regime_scan",
            tickers_analyzed=",".join(analyzed),
            report_path=",".join(report_paths),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.warning(f"Could not save scan session log: {e}")

    return {"scan": scan_result, "reports": reports, "regime": regime_packet}


def _parse_scan_args(args: list[str]) -> tuple[int, bool]:
    """Parse scan CLI args while keeping backward-compatible positional count."""
    flags = {arg.lower() for arg in args if arg.startswith("--")}
    unknown_flags = sorted(flags - {"--telegram"})
    if unknown_flags:
        raise ValueError(f"Unknown scan flag(s): {', '.join(unknown_flags)}")

    counts = [arg for arg in args if not arg.startswith("--")]
    if len(counts) > 1:
        raise ValueError("Usage: python run.py scan [count] [--telegram]")
    max_stocks = int(counts[0]) if counts else Config.SCAN_COUNCIL_MAX
    return max_stocks, "--telegram" in flags


def broker_router_preview(args: list[str]) -> None:
    """Run the pre-Council broker/data router using real FMP/YF data."""
    flags = {arg.lower() for arg in args if arg.startswith("--")}
    unknown = sorted(flags - {"--assume-market-open", "--no-persist"})
    if unknown:
        raise ValueError(f"Unknown broker-router-preview flag(s): {', '.join(unknown)}")
    pool = Config.SCAN_BROKER_ROUTER_POOL
    council_limit = Config.SCAN_COUNCIL_MAX
    session_id = f"router-preview-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}-{uuid4().hex[:8]}"

    collector = DataCollector()
    scanner = MarketScanner()
    journal = DecisionJournal()
    from artha.scheduler import MarketHours

    market_hours = MarketHours()

    print("\n🧭 ARTHA BROKER-AWARE ROUTER PREVIEW\n")
    print(f"Session: {session_id}")
    print(f"Pool: {pool} | Council max: {council_limit}")
    print("Data: real FMP universe/enrichment + yfinance quote preflight")

    macro_data = collector.collect_macro()
    try:
        from artha.regime import run_regime_council

        regime_packet = run_regime_council(macro_data)
    except Exception as exc:
        logger.warning("MROL failed in router preview; using neutral packet: %s", exc)
        regime_packet = {"base_regime_type": "neutral", "regime_type": "neutral", "event_overlays": []}

    candidates = scanner.get_funnel_candidates(regime_packet=regime_packet, max_candidates=pool)
    old_top = [str(row.get("symbol") or "?").upper() for row in candidates[:council_limit]]

    from artha.broker_router import route_scan_candidates

    active_watches: dict[str, list[dict]] = {}
    for watch in journal.get_active_defer_watches():
        ticker = str(watch.get("ticker") or "").upper()
        if ticker:
            active_watches.setdefault(ticker, []).append(watch)

    market_open = "--assume-market-open" in flags or market_hours.is_market_open(datetime.now(timezone.utc))
    routed = route_scan_candidates(
        candidates,
        session_id=session_id,
        journal=journal,
        active_watches=active_watches,
        quote_provider=collector.yf.quote,
        market_open=market_open,
        council_limit=council_limit,
        persist="--no-persist" not in flags,
    )

    counts = routed.summary_counts()
    selected = [row.ticker for row in routed.selected_for_council]
    print("\nOld top Council slate:")
    print("  " + (", ".join(f"${ticker}" for ticker in old_top) if old_top else "none"))
    print("\nBroker-routed buy-now Council slate:")
    print("  " + (", ".join(f"${ticker}" for ticker in selected) if selected else "none"))
    print("\nRouter counts:")
    for key, value in counts.items():
        print(f"  {key}: {value}")

    print("\nExecution-ready:")
    for row in routed.execution_ready[:12]:
        spread = f"{row.spread_pct:.2%}" if row.spread_pct is not None else "n/a"
        adv = f"${row.dollar_volume:,.0f}" if row.dollar_volume is not None else "n/a"
        print(f"  ${row.ticker}: score={row.route_score:.1f} spread={spread} ADV={adv} reason={row.reason_code}")

    print("\nResearch/watch, not buy-now slot:")
    for row in routed.research_watch[:20]:
        print(f"  ${row.ticker}: {row.reason_code} — {row.reason[:120]}")

    if routed.hard_reject:
        print("\nHard reject:")
        for row in routed.hard_reject[:10]:
            print(f"  ${row.ticker}: {row.reason_code} — {row.reason[:120]}")

    if "--no-persist" not in flags:
        print(f"\nPersisted router evidence under session_id={session_id}")


def portfolio_health_check():
    """Run one-shot portfolio monitor check and print formatted alerts."""
    monitor = PriceMonitor()
    status = monitor.one_shot_status()
    print()
    print(status.get("telegram_message", "No status available."))
    print()


def calibration_report():
    """Update shadow outcomes, backfill decision features, and print calibration state."""
    from artha.calibration import (
        backfill_decision_features,
        build_calibration_report,
        format_calibration_report,
    )

    journal = DecisionJournal()
    shadow_updates = AccuracyTracker().update_shadow_forward_returns(journal)
    backfilled = backfill_decision_features(journal)
    report = build_calibration_report(journal)
    print(format_calibration_report(report))
    print(
        "\nShadow forward-return update: "
        f"updated={shadow_updates.get('updated', 0)} "
        f"errors={shadow_updates.get('errors', 0)} "
        f"skipped={shadow_updates.get('skipped', 0)}"
    )
    print(f"\nBackfilled/verified decision feature rows from dossiers: {backfilled}")


def calibration_diagnosis(send_telegram: bool = False, force_telegram: bool = False):
    """Run deeper calibration diagnosis and optionally report to Telegram."""
    from artha.calibration import backfill_decision_features
    from artha.diagnostics import run_calibration_diagnosis

    journal = DecisionJournal()
    shadow_updates = AccuracyTracker().update_shadow_forward_returns(journal)
    backfilled = backfill_decision_features(journal)
    diagnostic = run_calibration_diagnosis(
        journal=journal,
        send_telegram=send_telegram,
        force_telegram=force_telegram,
    )
    print(diagnostic["report_text"])
    print(
        "\nShadow forward-return update: "
        f"updated={shadow_updates.get('updated', 0)} "
        f"errors={shadow_updates.get('errors', 0)} "
        f"skipped={shadow_updates.get('skipped', 0)}"
    )
    print(f"Backfilled/verified decision feature rows from dossiers: {backfilled}")
    print(f"Diagnosis artifact: {diagnostic.get('artifacts', {}).get('latest_text')}")
    if send_telegram:
        print(f"Telegram sent: {diagnostic.get('sent_to_telegram')}")


def supervisor_check(send_telegram: bool = False, force_telegram: bool = False):
    """Run Supervisor v1: operational checks, shadow rules, diagnosis, Telegram summary."""
    from artha.supervisor import run_supervisor_check

    report = run_supervisor_check(
        send_telegram=send_telegram,
        force_telegram=force_telegram,
    )
    print(report["report_text"])
    print(f"Supervisor artifact: {report.get('artifacts', {}).get('latest_text')}")
    if send_telegram:
        print(f"Telegram sent: {report.get('sent_to_telegram')}")


def execution_readiness(send_telegram: bool = False):
    """Check Robinhood-ready execution plumbing without placing orders."""
    from artha.execution import build_execution_readiness_report, format_execution_readiness
    from artha.telegram import TelegramSender

    report = build_execution_readiness_report()
    text = format_execution_readiness(report)
    print(text)
    if send_telegram:
        sent = TelegramSender().send_message(text, parse_mode=None, silent=True)
        print(f"Telegram sent: {sent}")


def _flag_value(args: list[str], name: str, default: str | None = None) -> str | None:
    if name not in args:
        return default
    idx = args.index(name)
    if idx + 1 >= len(args):
        return default
    return args[idx + 1]


def _float_flag(args: list[str], name: str, default: float | None = None) -> float | None:
    raw = _flag_value(args, name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def propose_order(args: list[str]):
    """Dry-run one Robinhood-style order intent through execution guardrails."""
    from artha.execution import (
        build_order_intent,
        evaluate_and_record_order,
        format_order_notice,
    )

    if not args:
        print("Usage: python run.py propose-order TICKER --side buy --notional 25 --limit 300 --price 300 --volume 50000000 --bid 299.9 --ask 300.1")
        sys.exit(1)

    ticker = args[0].upper()
    side = (_flag_value(args, "--side", "buy") or "buy").lower()
    notional = _float_flag(args, "--notional")
    quantity = _float_flag(args, "--quantity")
    limit_price = _float_flag(args, "--limit")
    price = _float_flag(args, "--price", limit_price)
    volume = _float_flag(args, "--volume")
    bid = _float_flag(args, "--bid")
    ask = _float_flag(args, "--ask")
    dollar_volume = _float_flag(args, "--dollar-volume")
    dossier = _flag_value(args, "--dossier", "") or ""
    rationale = _flag_value(args, "--rationale", "manual dry-run order proposal") or ""
    send_telegram = "--telegram" in {arg.lower() for arg in args}

    if limit_price is None:
        print("Missing --limit. Artha's Robinhood pilot prepares limit orders only.")
        sys.exit(1)
    if notional is None and quantity is None:
        print("Missing --notional or --quantity.")
        sys.exit(1)

    market_data = {
        "price": price,
        "volume": volume,
        "bid": bid,
        "ask": ask,
        "dollar_volume": dollar_volume,
    }
    if (price is None or volume is None) and "--no-fetch" not in args:
        try:
            quote = DataCollector().yf.quote(ticker)
            market_data["price"] = price or quote.get("price")
            market_data["volume"] = volume or quote.get("volume")
        except Exception as exc:
            logger.warning("Could not fetch yfinance quote for order proposal: %s", exc)

    intent = build_order_intent(
        ticker=ticker,
        side=side,
        notional=notional,
        quantity=quantity,
        limit_price=limit_price,
        estimated_price=market_data.get("price"),
        decision_dossier_path=dossier,
        rationale=rationale,
        dry_run=True,
    )
    result = evaluate_and_record_order(
        intent,
        market_data=market_data,
        send_telegram=send_telegram,
    )
    print(format_order_notice(result))
    print(f"\nExecution audit row: {result.get('row_id')}")
    if send_telegram:
        print(f"Telegram sent: {result.get('telegram_sent')}")


def _json_input(args: list[str]) -> dict:
    """Read a JSON payload from --file PATH or stdin."""
    if "--file" in args:
        path = _flag_value(args, "--file")
        if not path:
            raise ValueError("--file requires a path")
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    raw = sys.stdin.read().strip()
    if not raw:
        raise ValueError("JSON payload is required on stdin or via --file")
    return json.loads(raw)


def _json_from_flag(args: list[str], flag: str) -> dict | None:
    path = _flag_value(args, flag)
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _acquire_nonblocking_lock(lock_file: str | None):
    if not lock_file:
        return None
    import fcntl

    target = Path(lock_file).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = open(target, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"Another Robinhood snapshot import is already running: {target}")
    return handle


def _release_lock(handle) -> None:
    if not handle:
        return
    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def robinhood_snapshot_import(args: list[str]):
    """Import an OpenClaw/Robinhood MCP snapshot, then reconcile Artha state."""
    from artha.robinhood_bridge import (
        SnapshotHandoffValidationError,
        format_control_center,
        sync_snapshot_to_artha,
        validate_snapshot_handoff_metadata,
        write_robinhood_snapshot,
    )
    from artha.telegram import TelegramSender

    flags = {arg.lower() for arg in args}
    apply_changes = "--no-apply" not in flags
    strict = "--strict" in flags
    lock_handle = None
    try:
        lock_handle = _acquire_nonblocking_lock(_flag_value(args, "--lock-file"))
        payload = _json_input(args)
        validation = validate_snapshot_handoff_metadata(
            payload,
            expected_run_id=_flag_value(args, "--expect-run-id"),
            min_generated_at=_flag_value(args, "--min-generated-at"),
            max_age_minutes=_float_flag(args, "--max-handoff-age-minutes"),
        )
        snapshot_result = write_robinhood_snapshot(payload)
        sync_result = sync_snapshot_to_artha(snapshot_result.get("snapshot"), apply=apply_changes)
        success = (
            snapshot_result.get("status") == "PASS" and sync_result.get("status") == "PASS"
            if strict
            else sync_result.get("status") != "FAIL"
        )
        result = {
            "success": success,
            "strict": strict,
            "validation": validation,
            "snapshot": {k: v for k, v in snapshot_result.items() if k != "snapshot"},
            "sync": sync_result,
        }
        print(json.dumps(result, indent=2, default=str))
        if "--control-center" in flags:
            text = format_control_center()
            print()
            print(text)
            if "--telegram" in flags:
                sent = TelegramSender().send_message(text, parse_mode=None, silent=False)
                print(f"Telegram sent: {sent}")
        if strict and not success:
            sys.exit(1)
    except (SnapshotHandoffValidationError, RuntimeError, ValueError) as exc:
        result = {
            "success": False,
            "strict": strict,
            "error": str(exc),
            "validation": getattr(exc, "validation", None),
        }
        print(json.dumps(result, indent=2, default=str))
        sys.exit(1)
    finally:
        _release_lock(lock_handle)


def robinhood_control_center(args: list[str]):
    """Print/send the daily Robinhood control center."""
    from artha.robinhood_bridge import format_control_center
    from artha.telegram import TelegramSender

    text = format_control_center()
    print(text)
    if "--telegram" in {arg.lower() for arg in args}:
        sent = TelegramSender().send_message(text, parse_mode=None, silent=False)
        print(f"Telegram sent: {sent}")


def robinhood_snapshot_refresh_operation(args: list[str]):
    """Print the read-only OpenClaw/Robinhood MCP sequence for a fresh snapshot."""
    from artha.robinhood_bridge import build_snapshot_refresh_operation

    print(json.dumps(build_snapshot_refresh_operation(), indent=2, default=str))


def robinhood_auto_buy_runner_operation(args: list[str]):
    """Print the OpenClaw cron contract for unattended auto-buy queue drain."""
    from artha.robinhood_bridge import build_auto_buy_runner_operation

    result = build_auto_buy_runner_operation()
    flags = {arg.lower() for arg in args}
    if "--bootstrap-message-only" in flags:
        print(result["bootstrap_message"])
    elif "--message-only" in flags:
        print(result["runner_message"])
    else:
        print(json.dumps(result, indent=2, default=str))


def robinhood_queue_pending_reviews(args: list[str]):
    """Create durable Telegram Review actions from ready pending buy proposals."""
    from artha.robinhood_bridge import build_trade_action_notice, queue_review_actions_for_ready_orders
    from artha.telegram import TelegramSender

    result = queue_review_actions_for_ready_orders()
    print(json.dumps(result, indent=2, default=str))
    if "--telegram" in {arg.lower() for arg in args}:
        sender = TelegramSender()
        for action in result.get("created") or []:
            sender.send_message(
                build_trade_action_notice(action),
                parse_mode=None,
                silent=False,
                reply_markup=action.get("reply_markup"),
            )


def robinhood_action(args: list[str]):
    """Resolve an Artha Telegram callback token into the next OpenClaw MCP operation."""
    from artha.robinhood_bridge import build_action_operation

    if not args:
        print("Usage: python run.py robinhood-action artha:review:<action_id>:<token>")
        sys.exit(1)
    result = build_action_operation(args[0])
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result.get("success") else 1)


def robinhood_auto_buy_action(args: list[str]):
    """Resolve queued Artha auto-buy actions into OpenClaw MCP operations."""
    from artha.robinhood_bridge import build_auto_buy_operation, build_pending_auto_buy_operations

    if args:
        result = build_auto_buy_operation(args[0])
    else:
        result = build_pending_auto_buy_operations()
    print(json.dumps(result, indent=2, default=str))
    success = bool(result.get("success"))
    sys.exit(0 if success else 1)


def robinhood_auto_buy_queue_status(args: list[str]):
    """List queued auto-buy actions without mutating their gate state."""
    from artha.robinhood_bridge import build_pending_auto_buy_queue_status

    result = build_pending_auto_buy_queue_status()
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result.get("success") else 1)


def robinhood_auto_buy_agentic_clearance(args: list[str]):
    """Run agentic auto-buy clearance using OpenClaw-collected MCP responses."""
    from artha.openclaw_robinhood_handler import run_agentic_auto_buy_clearance_from_responses

    if not args:
        print(
            "Usage: python run.py robinhood-auto-buy-agentic-clearance ACTION_ID "
            "--quote-file quote.json --review-file review.json --tradability-file tradability.json"
        )
        sys.exit(1)
    action_id = args[0]
    quote = _json_from_flag(args[1:], "--quote-file")
    tradability = _json_from_flag(args[1:], "--tradability-file")
    review = _json_from_flag(args[1:], "--review-file")
    missing = [
        name
        for name, payload in (
            ("--quote-file", quote),
            ("--tradability-file", tradability),
            ("--review-file", review),
        )
        if payload is None
    ]
    if missing:
        print(f"Missing required file(s): {', '.join(missing)}")
        sys.exit(1)
    result = run_agentic_auto_buy_clearance_from_responses(
        action_id,
        quote_response=quote or {},
        tradability_response=tradability or {},
        review_response=review or {},
    )
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result.get("allow_place") else 1)


def robinhood_record_review(args: list[str]):
    """Record Robinhood tradability + review responses for a trade action."""
    from artha.robinhood_bridge import record_action_review
    from artha.telegram import TelegramSender

    if not args:
        print("Usage: python run.py robinhood-record-review ACTION_ID --review-file review.json --tradability-file tradability.json [--telegram]")
        sys.exit(1)
    action_id = args[0]
    review = _json_from_flag(args[1:], "--review-file")
    if review is None:
        review = _json_input(args[1:])
    tradability = _json_from_flag(args[1:], "--tradability-file")
    result = record_action_review(action_id, review, tradability_response=tradability)
    print(json.dumps(result, indent=2, default=str))
    if "--telegram" in {arg.lower() for arg in args}:
        sent = TelegramSender().send_message(
            result.get("message") or json.dumps(result, indent=2, default=str),
            parse_mode=None,
            silent=False,
            reply_markup=result.get("reply_markup"),
        )
        print(f"Telegram sent: {sent}")
    sys.exit(0 if result.get("status") == "review_clear" else 1)


def robinhood_final_clearance(args: list[str]):
    """Run the final Execution Officer gate using the stored Robinhood review."""
    from artha.robinhood_bridge import run_final_clearance_for_action

    if not args:
        print("Usage: python run.py robinhood-final-clearance ACTION_ID")
        sys.exit(1)
    result = run_final_clearance_for_action(args[0])
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result.get("allow_place") else 1)


def robinhood_record_submission(args: list[str]):
    """Record Robinhood place_equity_order response for a trade action."""
    from artha.robinhood_bridge import record_order_submission

    if not args:
        print("Usage: python run.py robinhood-record-submission ACTION_ID --file place-response.json")
        sys.exit(1)
    action_id = args[0]
    payload = _json_input(args[1:])
    result = record_order_submission(action_id=action_id, place_response=payload)
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result.get("status") == "PASS" else 1)


def robinhood_trading_control(args: list[str], disabled: bool):
    """Toggle the runtime Robinhood trading kill switch."""
    from artha.robinhood_bridge import set_trading_disabled

    reason = " ".join(arg for arg in args if not arg.startswith("--")).strip()
    result = set_trading_disabled(disabled, reason=reason)
    print(json.dumps({"success": True, "control": result}, indent=2, default=str))


def robinhood_record_fill(args: list[str]):
    """Record a Robinhood filled order into Artha's portfolio/thesis state."""
    from artha.robinhood_bridge import record_order_fill

    if not args:
        print("Usage: python run.py robinhood-record-fill ORDER_INTENT_ID --file robinhood-order.json")
        sys.exit(1)
    order_intent_id = args[0]
    payload = _json_input(args[1:])
    result = record_order_fill(order_intent_id=order_intent_id, fill_payload=payload)
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result.get("status") == "PASS" else 1)


def start_monitoring_daemon():
    """Run long-lived async monitoring scheduler."""
    scheduler = ArthaScheduler()
    asyncio.run(scheduler.run_forever())


def write_launchd_plist_templates():
    """Write macOS launchd plist templates for local automation."""
    from artha.automation import write_launchd_plists

    result = write_launchd_plists()
    print("\nLaunchd plist templates written:")
    for name, path in sorted((result.get("written") or {}).items()):
        print(f"  {name}: {path}")
    print(f"\n{result.get('load_hint')}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "test-apis":
        test_apis()
    elif command == "scan":
        try:
            max_stocks, send_telegram = _parse_scan_args(sys.argv[2:])
        except ValueError as exc:
            print(exc)
            sys.exit(1)
        full_market_scan(max_stocks=max_stocks, send_telegram=send_telegram)
    elif command == "analyze":
        if len(sys.argv) < 3:
            print("Usage: python run.py analyze TICKER1 TICKER2 ...")
            sys.exit(1)
        tickers = [t.upper() for t in sys.argv[2:]]
        analyze_stocks(tickers)
    elif command == "overview":
        market_overview()
    elif command == "portfolio":
        portfolio_status()
    elif command == "check":
        portfolio_health_check()
    elif command == "calibrate":
        calibration_report()
    elif command == "diagnose":
        flags = {arg.lower() for arg in sys.argv[2:]}
        calibration_diagnosis(
            send_telegram="--telegram" in flags,
            force_telegram="--force-telegram" in flags,
        )
    elif command == "supervise":
        flags = {arg.lower() for arg in sys.argv[2:]}
        supervisor_check(
            send_telegram="--telegram" in flags,
            force_telegram="--force-telegram" in flags,
        )
    elif command == "execution-readiness":
        flags = {arg.lower() for arg in sys.argv[2:]}
        execution_readiness(send_telegram="--telegram" in flags)
    elif command == "broker-router-preview":
        try:
            broker_router_preview(sys.argv[2:])
        except ValueError as exc:
            print(exc)
            sys.exit(1)
    elif command == "propose-order":
        propose_order(sys.argv[2:])
    elif command == "robinhood-snapshot-import":
        robinhood_snapshot_import(sys.argv[2:])
    elif command == "robinhood-control-center":
        robinhood_control_center(sys.argv[2:])
    elif command == "robinhood-snapshot-refresh-operation":
        robinhood_snapshot_refresh_operation(sys.argv[2:])
    elif command == "robinhood-auto-buy-runner-operation":
        robinhood_auto_buy_runner_operation(sys.argv[2:])
    elif command == "robinhood-queue-pending-reviews":
        robinhood_queue_pending_reviews(sys.argv[2:])
    elif command == "robinhood-action":
        robinhood_action(sys.argv[2:])
    elif command == "robinhood-auto-buy-action":
        robinhood_auto_buy_action(sys.argv[2:])
    elif command == "robinhood-auto-buy-queue-status":
        robinhood_auto_buy_queue_status(sys.argv[2:])
    elif command == "robinhood-auto-buy-agentic-clearance":
        robinhood_auto_buy_agentic_clearance(sys.argv[2:])
    elif command == "robinhood-record-review":
        robinhood_record_review(sys.argv[2:])
    elif command == "robinhood-final-clearance":
        robinhood_final_clearance(sys.argv[2:])
    elif command == "robinhood-record-submission":
        robinhood_record_submission(sys.argv[2:])
    elif command == "robinhood-disable-trading":
        robinhood_trading_control(sys.argv[2:], disabled=True)
    elif command == "robinhood-enable-trading":
        robinhood_trading_control(sys.argv[2:], disabled=False)
    elif command == "robinhood-record-fill":
        robinhood_record_fill(sys.argv[2:])
    elif command == "launchd-plists":
        write_launchd_plist_templates()
    elif command == "monitor":
        start_monitoring_daemon()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
