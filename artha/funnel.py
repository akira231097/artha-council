"""Promotion Funnel — filters a large universe down to council-ready candidates.

5-stage pipeline:
  1. Universe build (FMP screener) → 500-1000 names
  2. Machine rank (yfinance momentum + regime fit) → top 50
  3. Enrich top 50 (FMP ratios + Finnhub recs + earnings calendar)
  4. LLM triage via quick scoring heuristic → top 10-15
  5. Return top 6-8 investigation candidates for full council analysis

These are council investigation candidates, not buy recommendations. The council
must still approve, defer, or reject each entry after deep diligence.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Optional

from .config import Config

logger = logging.getLogger(__name__)

UTC = timezone.utc

# Minimum council candidates before falling back to legacy scan
MIN_CANDIDATES_THRESHOLD = 3


def _recent_scan_penalties(limit: int = 5) -> dict[str, float]:
    """Soft penalty for tickers repeated across recent weekly scans.

    Repeats are okay if justified, but the funnel should not lazily recycle the
    same small basket forever. This is a soft freshness nudge, not a hard
    exclusion.
    """
    penalties: dict[str, float] = {}
    try:
        db_path = Path(__file__).resolve().parent.parent / "data" / "artha.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        rows = cur.execute(
            "select tickers_analyzed from sessions where session_type='weekly_scan' order by datetime(timestamp) desc limit ?",
            (limit,),
        ).fetchall()
        conn.close()
        counts: dict[str, int] = {}
        for row in rows:
            raw = str(row['tickers_analyzed'] or '')
            for ticker in [t.strip().upper() for t in raw.split(',') if t.strip()]:
                counts[ticker] = counts.get(ticker, 0) + 1
        for ticker, count in counts.items():
            if count >= 2:
                penalties[ticker] = min(18.0, float((count - 1) * 6))
    except Exception:
        return {}
    return penalties


class PromotionFunnel:
    """Filters the investable universe down to council-ready candidates.

    Usage:
        funnel = PromotionFunnel()
        candidates = funnel.run(regime_packet)
    """

    def __init__(self):
        from .universe import UniverseBuilder
        self.universe_builder = UniverseBuilder()

    def run(
        self,
        regime_packet,
        max_council_candidates: int = 8,
        fallback_on_failure: bool = True,
    ) -> list[dict]:
        """Run the full promotion funnel.

        Args:
            regime_packet: RegimePacket from MROL (uses .base_regime_type in
                           current code, with .regime_type accepted for older
                           packets), OR a dict with "base_regime_type" /
                           "regime_type" and "event_overlays" keys.
            max_council_candidates: Max candidates to return (default 8)
            fallback_on_failure: If True, fall back to _scan_key_tickers()
                                 on any failure (default True)

        Returns:
            List of enriched candidate dicts, ready for council analysis.
        """
        start_time = datetime.now(UTC)
        logger.info(f"[funnel] Starting promotion funnel (max={max_council_candidates})...")

        # Normalize regime_packet input
        def _extract_overlays(raw_overlays) -> list[str]:
            """Extract overlay type strings from list of strings or dicts."""
            result = []
            for o in (raw_overlays or []):
                if isinstance(o, str):
                    result.append(o)
                elif isinstance(o, dict):
                    t = o.get("type", "")
                    if t:
                        result.append(t)
            return result

        if hasattr(regime_packet, "base_regime_type") or hasattr(regime_packet, "regime_type"):
            regime_type = (
                getattr(regime_packet, "base_regime_type", None)
                or getattr(regime_packet, "regime_type", None)
            )
            overlays = _extract_overlays(getattr(regime_packet, "event_overlays", []))
        elif isinstance(regime_packet, dict):
            regime_type = regime_packet.get("base_regime_type") or regime_packet.get("regime_type")
            overlays = _extract_overlays(regime_packet.get("event_overlays", []))
        else:
            regime_type = None
            overlays = []

        overlays = [o for o in overlays if o]  # Remove empty strings

        try:
            # --- Stage 1: Universe Build ---
            logger.info("[funnel] Stage 1: Building universe via FMP screener...")
            universe = self.universe_builder.build_universe(
                regime_type=regime_type,
                overlays=overlays,
                limit=1000,
            )

            if not universe:
                logger.warning("[funnel] Universe build returned empty — using fallback")
                return self._fallback(max_council_candidates) if fallback_on_failure else []

            logger.info(f"[funnel] Stage 1 complete: {len(universe)} candidates")

            # --- Stage 2: Machine Rank ---
            logger.info("[funnel] Stage 2: Ranking by momentum + regime fit...")
            from .rank_candidates import rank_universe
            ranked = rank_universe(
                universe=universe,
                regime_type=regime_type,
                overlays=overlays,
                top_n=max(50, int(getattr(Config, "FUNNEL_RANK_TOP_N", 150))),
            )

            if not ranked:
                logger.warning("[funnel] Ranking returned empty — using fallback")
                return self._fallback(max_council_candidates) if fallback_on_failure else []

            # Record momentum scores and enrich with delta/trend
            try:
                from .momentum_tracker import MomentumTracker
                tracker = MomentumTracker()
                scan_date = datetime.now(UTC).strftime("%Y-%m-%d")
                tracker.record_scores(ranked, scan_date)
                ranked = tracker.enrich_ranked_candidates(ranked)
                logger.info("[funnel] Momentum scores recorded and candidates enriched with delta/trend")
            except Exception as mt_e:
                logger.warning("[funnel] Momentum tracker failed (non-fatal): %s", mt_e)

            logger.info(f"[funnel] Stage 2 complete: {len(ranked)} ranked candidates")

            # --- Stage 3: Enrich top ranked candidates ---
            enrich_max = max(max_council_candidates, min(Config.FUNNEL_ENRICH_MAX, len(ranked)))
            enrichment_pool = self._build_enrichment_pool(ranked, enrich_max, universe=universe)
            logger.info("[funnel] Stage 3: Enriching top %d candidates...", len(enrichment_pool))
            enriched = self._enrich(enrichment_pool)
            logger.info(f"[funnel] Stage 3 complete: {len(enriched)} enriched")

            # --- Stage 4: Quick score/filter → top 20-30 ---
            logger.info("[funnel] Stage 4: Quick scoring → selecting alpha-sleeve pool...")
            top_candidates = self._quick_score(enriched, top_n=min(50, len(enriched)))
            logger.info(f"[funnel] Stage 4 complete: {len(top_candidates)} pre-council candidates")

            # --- Stage 5: Return diversified top N for council ---
            final = self._select_alpha_sleeves(top_candidates, max_council_candidates)

            # Check minimum threshold — fallback if too few candidates
            if len(final) < MIN_CANDIDATES_THRESHOLD and fallback_on_failure:
                logger.warning(
                    f"[funnel] Only {len(final)} candidates < threshold {MIN_CANDIDATES_THRESHOLD} "
                    f"— supplementing with fallback"
                )
                return self._fallback(max_council_candidates)

            elapsed = (datetime.now(UTC) - start_time).total_seconds()
            sleeve_counts = Counter(c.get("primary_alpha_sleeve", "unknown") for c in final)
            sector_counts = Counter(c.get("sector", "unknown") for c in final)
            logger.info(
                f"[funnel] Complete in {elapsed:.1f}s: "
                f"{len(final)} candidates for council "
                f"(universe={len(universe)}, ranked={len(ranked)}, "
                f"sleeves={dict(sleeve_counts)}, sectors={dict(sector_counts)})"
            )
            return final

        except Exception as e:
            logger.error(f"[funnel] Funnel failed: {e}", exc_info=True)
            if fallback_on_failure:
                logger.warning("[funnel] Falling back to legacy ticker scan")
                return self._fallback(max_council_candidates)
            return []

    def _build_enrichment_pool(
        self,
        ranked: list[dict],
        enrich_max: int,
        universe: Optional[list] = None,
    ) -> list[dict]:
        """Blend momentum leaders with not-overheated entry-quality candidates."""
        if not ranked or enrich_max <= 0:
            return []

        selected: list[dict] = []
        seen: set[str] = set()

        def add(candidate: dict, reason: str) -> bool:
            symbol = str(candidate.get("symbol") or "").upper()
            if not symbol or symbol in seen:
                return False
            enriched_candidate = dict(candidate)
            enriched_candidate["enrichment_pool_reason"] = reason
            selected.append(enriched_candidate)
            seen.add(symbol)
            return True

        momentum_quota = min(len(ranked), max(12, int(enrich_max * 0.45)))
        for candidate in ranked[:momentum_quota]:
            add(candidate, "momentum_leader")

        def entry_proxy(candidate: dict) -> float:
            r12 = self._num(candidate.get("return_12m"))
            r3 = self._num(candidate.get("return_3m"))
            vol = self._num(candidate.get("vol_20d"), 20.0)
            combined = self._num(candidate.get("combined_score"))
            market_cap = self._num(candidate.get("market_cap"))
            score = 0.0
            if 8 <= r12 <= 180:
                score += 10
            elif 0 <= r12 <= 260:
                score += 5
            elif r12 > 350:
                score -= 8
            if -12 <= r3 <= 45:
                score += 8
            elif 45 < r3 <= 75:
                score += 2
            elif r3 > 90:
                score -= 6
            if vol <= 35:
                score += 5
            elif vol <= 55:
                score += 3
            elif vol >= 85:
                score -= 5
            if market_cap >= 5_000_000_000:
                score += 3
            elif market_cap >= 1_000_000_000:
                score += 1
            score += min(4.0, max(0.0, combined) / 40.0)
            return score

        entry_candidates = sorted(ranked, key=entry_proxy, reverse=True)
        entry_quota = max(
            int(getattr(Config, "FUNNEL_MIN_ENTRY_QUALITY_CANDIDATES", 3)) * 4,
            int(enrich_max * 0.35),
        )
        for candidate in entry_candidates:
            if len(selected) >= enrich_max:
                break
            if entry_proxy(candidate) <= 0:
                continue
            add(candidate, "entry_quality_proxy")
            if sum(1 for c in selected if c.get("enrichment_pool_reason") == "entry_quality_proxy") >= entry_quota:
                break

        if getattr(Config, "FUNNEL_PARALLEL_DISCOVERY_ENABLED", True) and universe:
            ranked_symbols = {str(c.get("symbol") or "").upper() for c in ranked}
            discovery_quota = min(
                max(0, enrich_max - len(selected)),
                max(4, int(getattr(Config, "FUNNEL_PARALLEL_DISCOVERY_MAX", 12))),
            )
            for candidate in self._parallel_discovery_candidates(
                universe=universe,
                ranked_symbols=ranked_symbols,
                limit=discovery_quota * 3,
            ):
                if len(selected) >= enrich_max:
                    break
                add(candidate, str(candidate.get("enrichment_pool_reason") or "parallel_discovery"))
                if sum(1 for c in selected if str(c.get("enrichment_pool_reason", "")).startswith("parallel_")) >= discovery_quota:
                    break

        for candidate in ranked:
            if len(selected) >= enrich_max:
                break
            add(candidate, "balanced_fill")

        proxy_count = sum(1 for c in selected if c.get("enrichment_pool_reason") == "entry_quality_proxy")
        discovery_count = sum(1 for c in selected if str(c.get("enrichment_pool_reason", "")).startswith("parallel_"))
        logger.info(
            "[funnel] Enrichment pool blended: %d total (%d entry-quality proxy, %d parallel discovery, %d momentum leaders)",
            len(selected),
            proxy_count,
            discovery_count,
            sum(1 for c in selected if c.get("enrichment_pool_reason") == "momentum_leader"),
        )
        return selected[:enrich_max]

    def _parallel_discovery_candidates(
        self,
        *,
        universe: list,
        ranked_symbols: set[str],
        limit: int,
    ) -> list[dict]:
        """Select broad-market probes that do not depend on momentum ranking."""
        if not universe or limit <= 0:
            return []

        probes: list[dict] = []
        sector_counts: Counter[str] = Counter()
        for item in universe:
            symbol = str(getattr(item, "symbol", "") or "").upper()
            if not symbol or symbol in ranked_symbols:
                continue
            sector = str(getattr(item, "sector", "") or "unknown")
            if sector_counts[sector] >= 6:
                continue

            market_cap = self._num(getattr(item, "market_cap", 0))
            price = self._num(getattr(item, "price", 0))
            volume = self._num(getattr(item, "volume", 0))
            beta = self._num(getattr(item, "beta", None), 1.0)
            regime_score = self._num(getattr(item, "regime_score", 0))
            dollar_volume = price * volume

            score = 0.0
            if market_cap >= 3_000_000_000:
                score += 5
            if 5_000_000_000 <= market_cap <= 150_000_000_000:
                score += 4
            elif market_cap > 150_000_000_000:
                score += 1
            if volume >= 500_000:
                score += 3
            if dollar_volume >= 20_000_000:
                score += 4
            if 0.6 <= beta <= 1.8:
                score += 4
            elif beta <= 2.5:
                score += 1
            elif beta >= 3.0:
                score -= 4
            score += max(-2.0, min(5.0, regime_score / 3.0))

            if score < 8:
                continue

            reason = "parallel_quality_value"
            if regime_score >= 8:
                reason = "parallel_regime_quality"
            elif 0.6 <= beta <= 1.4 and dollar_volume >= 50_000_000:
                reason = "parallel_liquid_quality"

            probes.append({
                "symbol": symbol,
                "name": getattr(item, "name", ""),
                "sector": sector,
                "industry": getattr(item, "industry", ""),
                "market_cap": market_cap,
                "avg_volume": volume,
                "price": round(price, 2),
                "beta": beta,
                "momentum_score": 0.0,
                "regime_score": round(regime_score, 2),
                "combined_score": round(score + regime_score, 2),
                "return_1m": None,
                "return_3m": None,
                "return_12m": None,
                "vol_20d": None,
                "parallel_discovery_score": round(score, 2),
                "enrichment_pool_reason": reason,
                "data_provider_lane": "fmp_universe_parallel_discovery",
            })
            sector_counts[sector] += 1
            if len(probes) >= limit:
                break

        probes.sort(
            key=lambda c: (
                self._num(c.get("parallel_discovery_score")),
                self._num(c.get("regime_score")),
                self._num(c.get("market_cap")),
            ),
            reverse=True,
        )
        return probes[:limit]

    def _enrich(self, ranked: list[dict]) -> list[dict]:
        """Enrich top-50 candidates with FMP ratios, Finnhub recs, and earnings context."""
        from .analyst_signals import get_recommendation_trends, get_analyst_estimates, get_short_interest
        from .earnings_calendar import get_earnings_context
        from .collector import FMPCollector
        from .liquidity import passes_liquidity_gate

        fmp = FMPCollector()
        enriched = []
        provider_timeout = max(1, int(getattr(Config, "FUNNEL_ENRICH_PROVIDER_TIMEOUT_SECONDS", 6) or 6))
        provider_retries = max(0, int(getattr(Config, "FUNNEL_ENRICH_PROVIDER_RETRIES", 0) or 0))
        total_timeout = max(5, int(getattr(Config, "FUNNEL_ENRICH_TOTAL_TIMEOUT_SECONDS", 120) or 120))
        deadline = time.monotonic() + total_timeout

        def minimal_enriched(row: dict, *, timed_out: bool) -> dict:
            minimal = dict(row)
            minimal["enrichment_source"] = "funnel_timeout_partial" if timed_out else "funnel"
            minimal["enrichment_timeout"] = bool(timed_out)
            for key in (
                "ratios_ttm",
                "key_metrics_ttm",
                "price_target_consensus",
                "dcf",
                "recommendation_trends",
                "analyst_estimates",
                "short_interest",
                "earnings_context",
            ):
                minimal.setdefault(key, None)
            ticker_data = {
                "market_cap": row.get("market_cap", 0),
                "price": row.get("price", 0),
                "avg_volume": row.get("avg_volume", 0),
            }
            minimal["passes_liquidity"] = passes_liquidity_gate(ticker_data)
            return minimal

        for index, candidate in enumerate(ranked):
            if time.monotonic() >= deadline:
                logger.warning(
                    "[funnel] Enrichment budget exhausted after %ds; returning %d enriched and %d partial candidate(s)",
                    total_timeout,
                    len(enriched),
                    len(ranked) - index,
                )
                for rest in ranked[index:]:
                    enriched.append(minimal_enriched(rest, timed_out=True))
                break
            sym = candidate.get("symbol", "")
            if not sym:
                continue

            enriched_candidate = dict(candidate)
            enriched_candidate["enrichment_source"] = "funnel"
            enriched_candidate["enrichment_timeout"] = False

            # FMP ratios TTM
            try:
                ratios = fmp.ratios_ttm(sym, timeout=provider_timeout, retries=provider_retries)
                enriched_candidate["ratios_ttm"] = ratios
            except Exception:
                enriched_candidate["ratios_ttm"] = None

            # FMP key metrics TTM
            try:
                km = fmp.key_metrics_ttm(sym, timeout=provider_timeout, retries=provider_retries)
                enriched_candidate["key_metrics_ttm"] = km
            except Exception:
                enriched_candidate["key_metrics_ttm"] = None

            # Valuation anchors for entry-quality sleeve.
            try:
                enriched_candidate["price_target_consensus"] = fmp.price_target_consensus(
                    sym,
                    timeout=provider_timeout,
                    retries=provider_retries,
                )
            except Exception:
                enriched_candidate["price_target_consensus"] = None
            try:
                enriched_candidate["dcf"] = fmp.dcf(sym, timeout=provider_timeout, retries=provider_retries)
            except Exception:
                enriched_candidate["dcf"] = None

            # Recommendation trends
            try:
                recs = get_recommendation_trends(sym)
                enriched_candidate["recommendation_trends"] = recs
            except Exception:
                enriched_candidate["recommendation_trends"] = None

            # Analyst estimates
            try:
                estimates = get_analyst_estimates(sym, timeout=provider_timeout, retries=provider_retries)
                enriched_candidate["analyst_estimates"] = estimates
            except Exception:
                enriched_candidate["analyst_estimates"] = None

            # Short interest / crowding risk
            try:
                short_interest = get_short_interest(sym, timeout=provider_timeout, retries=provider_retries)
                enriched_candidate["short_interest"] = short_interest
            except Exception:
                enriched_candidate["short_interest"] = None

            # Earnings context
            try:
                ec = get_earnings_context(sym)
                enriched_candidate["earnings_context"] = {
                    "earnings_date": ec.earnings_date,
                    "days_to_earnings": ec.days_to_earnings,
                    "earnings_time": ec.earnings_time,
                    "earnings_risk_flag": ec.earnings_risk_flag,
                    "earnings_defer_flag": ec.earnings_defer_flag,
                }
            except Exception:
                enriched_candidate["earnings_context"] = None

            # Liquidity gate check using rank data
            ticker_data = {
                "market_cap": candidate.get("market_cap", 0),
                "price": candidate.get("price", 0),
                "avg_volume": candidate.get("avg_volume", 0),
            }
            enriched_candidate["passes_liquidity"] = passes_liquidity_gate(ticker_data)

            enriched.append(enriched_candidate)

        return enriched

    def _quick_score(self, enriched: list[dict], top_n: int = 15) -> list[dict]:
        """Apply quick heuristic scoring to select top candidates.

        Scoring factors:
          - combined_score from momentum + regime (already computed)
          - Earnings defer flag penalty (-50 if within 2 days)
          - Earnings risk flag penalty (-20 if within 7 days)
          - Analyst consensus bonus (+10 if buy/strong_buy)
          - Liquidity gate failure penalty (-100)

        Returns sorted list of top_n candidates.
        """
        scored = []
        recent_penalties = _recent_scan_penalties(limit=5)
        for c in enriched:
            # Start with combined momentum + regime score
            base_score = c.get("combined_score", 0)
            symbol = str(c.get("symbol", "") or "").upper()

            # Earnings timing penalties
            ec = c.get("earnings_context") or {}
            if ec.get("earnings_defer_flag"):
                base_score -= 50  # Defer: within 2 days of earnings
            elif ec.get("earnings_risk_flag"):
                base_score -= 20  # Risk: within 7 days of earnings

            # Analyst consensus bonus
            recs = c.get("recommendation_trends") or {}
            consensus = recs.get("consensus", "")
            if consensus in ("buy", "strong_buy"):
                base_score += 10
            elif consensus in ("sell", "strong_sell"):
                base_score -= 10

            # Crowding penalty unless a squeeze setup is explicitly identified.
            short_interest = c.get("short_interest") or {}
            short_pct = short_interest.get("short_interest_pct")
            if isinstance(short_pct, (int, float)):
                if short_interest.get("squeeze_risk_flag"):
                    base_score += 3
                    c["short_interest_note"] = "squeeze_risk"
                elif short_pct >= 15:
                    base_score -= 5
                    c["short_interest_note"] = "crowded_short"

            # Momentum acceleration bonus/penalty (+3 accelerating, -3 decelerating)
            trend = c.get("momentum_trend", "")
            if trend == "accelerating":
                base_score += 3
            elif trend == "decelerating":
                base_score -= 3

            # Liquidity gate failure
            if not c.get("passes_liquidity", True):
                base_score -= 100

            # Soft freshness penalty for stale repeat names across recent scans
            repeat_penalty = recent_penalties.get(symbol, 0.0)
            if repeat_penalty > 0:
                base_score -= repeat_penalty
                c["repeat_penalty"] = repeat_penalty

            sleeve_scores = self._alpha_sleeve_scores(c)
            c["alpha_sleeve_scores"] = {k: round(v, 2) for k, v in sleeve_scores.items()}
            active_sleeves = [k for k, v in sleeve_scores.items() if v > 0]
            c["alpha_sleeves"] = active_sleeves
            if active_sleeves:
                c["primary_alpha_sleeve"] = max(active_sleeves, key=lambda k: sleeve_scores[k])
            else:
                c["primary_alpha_sleeve"] = "momentum"

            entry_quality = sleeve_scores.get("entry_quality", 0.0)
            if entry_quality >= getattr(Config, "FUNNEL_ENTRY_QUALITY_MIN_SCORE", 16):
                base_score += min(25.0, entry_quality * 0.75)
                c["entry_quality_candidate"] = True

            c["funnel_score"] = round(base_score, 2)
            scored.append(c)

        scored.sort(key=lambda x: x.get("funnel_score", 0), reverse=True)
        return self._diversified_prescreen_pool(scored, top_n)

    def _diversified_prescreen_pool(self, scored: list[dict], top_n: int) -> list[dict]:
        """Preserve top score names while guaranteeing sleeve candidates survive."""
        if not scored or top_n <= 0:
            return []
        selected: list[dict] = []
        seen: set[str] = set()

        def add(candidate: dict) -> bool:
            symbol = str(candidate.get("symbol") or "").upper()
            if not symbol or symbol in seen:
                return False
            selected.append(candidate)
            seen.add(symbol)
            return True

        for candidate in scored[: max(8, min(20, top_n // 2))]:
            add(candidate)

        for sleeve in ("entry_quality", "quality_value", "pullback_quality", "estimate_revision", "contrarian_squeeze"):
            for candidate in sorted(
                scored,
                key=lambda c: (
                    self._num((c.get("alpha_sleeve_scores") or {}).get(sleeve)),
                    self._num(c.get("funnel_score")),
                ),
                reverse=True,
            ):
                if self._num((candidate.get("alpha_sleeve_scores") or {}).get(sleeve)) <= 0:
                    break
                add(candidate)
                if len(selected) >= top_n:
                    return selected[:top_n]
                if sum(1 for c in selected if c.get("primary_alpha_sleeve") == sleeve) >= 6:
                    break

        for candidate in scored:
            if len(selected) >= top_n:
                break
            add(candidate)
        return selected[:top_n]

    @staticmethod
    def _num(value, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _get_any(payload: dict, *keys: str, default: float = 0.0) -> float:
        if not isinstance(payload, dict):
            return default
        for key in keys:
            if key in payload and payload.get(key) is not None:
                return PromotionFunnel._num(payload.get(key), default)
        return default

    def _alpha_sleeve_scores(self, candidate: dict) -> dict[str, float]:
        """Score candidate across distinct pre-council alpha sleeves.

        These are investigation sleeves, not buy signals. The council still
        decides whether each name is buyable, deferrable, or avoidable.
        """
        ratios = candidate.get("ratios_ttm") or {}
        metrics = candidate.get("key_metrics_ttm") or {}
        recs = candidate.get("recommendation_trends") or {}
        estimates = candidate.get("analyst_estimates") or {}
        short_interest = candidate.get("short_interest") or {}
        targets = candidate.get("price_target_consensus") or {}
        dcf = candidate.get("dcf") or {}

        momentum = max(0.0, self._num(candidate.get("momentum_score")) / 40.0)
        combined = max(0.0, self._num(candidate.get("combined_score")) / 50.0)
        trend = str(candidate.get("momentum_trend") or "")
        price = self._num(candidate.get("price"))
        r12 = self._num(candidate.get("return_12m"))
        r3 = self._num(candidate.get("return_3m"))
        vol = self._num(candidate.get("vol_20d"), 20.0)
        beta = self._num(candidate.get("beta"), 1.0)

        consensus = str(recs.get("consensus") or "").lower()
        upgrades = self._num(recs.get("net_upgrades_30d"))
        downgrades = self._num(recs.get("net_downgrades_30d"))
        rec_mix = recs.get("recommendation_mix") or {}
        buy_count = self._num(rec_mix.get("strong_buy")) + self._num(rec_mix.get("buy"))
        sell_count = self._num(rec_mix.get("sell")) + self._num(rec_mix.get("strong_sell"))

        pe = self._get_any(ratios, "peRatioTTM", "priceEarningsRatioTTM")
        ps = self._get_any(ratios, "priceToSalesRatioTTM", "priceToSalesRatio")
        pb = self._get_any(ratios, "priceToBookRatioTTM", "priceToBookRatio")
        fcf_yield = self._get_any(metrics, "freeCashFlowYieldTTM", "freeCashFlowYield")
        roe = self._get_any(ratios, "returnOnEquityTTM", "roeTTM")
        gross_margin = self._get_any(ratios, "grossProfitMarginTTM", "grossMarginTTM")
        debt_equity = self._get_any(ratios, "debtEquityRatioTTM", "debtToEquityTTM", default=0.0)
        fcf_per_share = self._get_any(metrics, "freeCashFlowPerShareTTM", "freeCashFlowPerShare")
        roic = self._get_any(metrics, "roicTTM", "returnOnInvestedCapitalTTM")
        target_consensus = self._get_any(targets, "targetConsensus", "target_consensus")
        dcf_value = self._get_any(dcf, "dcf", "dcfValue", "DCF")
        consensus_upside = ((target_consensus / price) - 1) * 100 if price > 0 and target_consensus > 0 else 0.0
        dcf_upside = ((dcf_value / price) - 1) * 100 if price > 0 and dcf_value > 0 else 0.0

        valuation_reasonable = 0.0
        if 0 < pe <= 35:
            valuation_reasonable += 4
        if 0 < ps <= 8:
            valuation_reasonable += 3
        if 0 < pb <= 8:
            valuation_reasonable += 2
        quality = 0.0
        if gross_margin >= 0.35:
            quality += 3
        if roe >= 0.12 or roic >= 0.10:
            quality += 4
        if fcf_per_share > 0:
            quality += 3
        if debt_equity and debt_equity > 2.0:
            quality -= 2
        if fcf_yield >= 0.04:
            quality += 2
        elif fcf_yield < 0:
            quality -= 3

        next_eps = self._num(estimates.get("next_q_eps_estimate"))
        next_rev = self._num(estimates.get("next_q_revenue_estimate"))
        fy_rev = self._num(estimates.get("fy1_revenue_estimate"))
        estimate_depth = sum(1 for x in (next_eps, next_rev, fy_rev) if x not in (0.0, None))

        short_pct = self._num(short_interest.get("short_interest_pct"))
        days_to_cover = self._num(short_interest.get("days_to_cover"))
        squeeze = bool(short_interest.get("squeeze_risk_flag")) or (short_pct >= 12 and days_to_cover >= 3)

        timing = 0.0
        if 8 <= r12 <= 180:
            timing += 5
        elif 0 <= r12 <= 260:
            timing += 2
        elif r12 > 350:
            timing -= 6
        if -12 <= r3 <= 45:
            timing += 5
        elif 45 < r3 <= 75:
            timing += 1
        elif r3 > 90:
            timing -= 5
        if vol <= 35:
            timing += 4
        elif vol <= 55:
            timing += 2
        elif vol >= 85:
            timing -= 4
        if beta <= 1.8:
            timing += 2
        elif beta >= 3:
            timing -= 3

        upside = 0.0
        if consensus_upside >= 20:
            upside += 5
        elif consensus_upside >= 8:
            upside += 3
        elif consensus_upside <= -10:
            upside -= 5
        if dcf_upside >= 15:
            upside += 2
        elif dcf_upside <= -50:
            upside -= 2

        entry_quality = (
            quality
            + valuation_reasonable
            + timing
            + upside
            + (3 if consensus in {"buy", "strong_buy"} else 0)
            + min(3, estimate_depth)
            + min(3, max(0.0, upgrades - downgrades))
        )

        return {
            "momentum": combined + (2 if trend == "accelerating" else 0),
            "estimate_revision": (
                (8 if consensus in {"buy", "strong_buy"} else 0)
                + min(6, upgrades * 2)
                - min(6, downgrades * 2)
                + min(3, estimate_depth)
                + max(0, buy_count - sell_count) * 0.5
            ),
            "quality_value": quality + valuation_reasonable + min(4, momentum),
            "contrarian_squeeze": (
                (9 if squeeze else 0)
                + min(5, short_pct / 4.0)
                + (3 if consensus not in {"sell", "strong_sell"} else -3)
                + min(3, momentum)
            ),
            "pullback_quality": (
                (quality + valuation_reasonable) * 0.7
                + (4 if trend == "decelerating" and self._num(candidate.get("return_12m")) > 50 else 0)
                + (2 if consensus in {"buy", "strong_buy"} else 0)
            ),
            "entry_quality": max(0.0, entry_quality),
        }

    def _select_alpha_sleeves(self, scored: list[dict], max_candidates: int) -> list[dict]:
        """Select a diversified council slate from multiple alpha sleeves."""
        if not scored or max_candidates <= 0:
            return []

        max_per_sector = max(2, int(getattr(Config, "FUNNEL_MAX_CANDIDATES_PER_SECTOR", 3)))
        if max_candidates <= 5:
            max_per_sector = max(2, min(max_per_sector, ceil(max_candidates / 2)))

        selected: list[dict] = []
        selected_symbols: set[str] = set()
        sector_counts: Counter[str] = Counter()

        def can_add(candidate: dict) -> bool:
            symbol = str(candidate.get("symbol", "") or "").upper()
            if not symbol or symbol in selected_symbols:
                return False
            sector = str(candidate.get("sector") or "unknown")
            return sector_counts[sector] < max_per_sector

        def add(candidate: dict, sleeve: str) -> bool:
            if not can_add(candidate):
                return False
            candidate["primary_alpha_sleeve"] = sleeve
            candidate.setdefault("alpha_sleeves", [])
            if sleeve not in candidate["alpha_sleeves"]:
                candidate["alpha_sleeves"].insert(0, sleeve)
            selected.append(candidate)
            selected_symbols.add(str(candidate.get("symbol") or "").upper())
            sector_counts[str(candidate.get("sector") or "unknown")] += 1
            return True

        # Keep one pure score leader, then force a few entry-quality names so
        # the council sees buyable setups instead of only late-stage rockets.
        add(scored[0], "momentum")

        entry_min = float(getattr(Config, "FUNNEL_ENTRY_QUALITY_MIN_SCORE", 16))
        entry_target = max(0, min(
            int(getattr(Config, "FUNNEL_MIN_ENTRY_QUALITY_CANDIDATES", 3)),
            max_candidates - len(selected),
        ))
        if entry_target:
            for candidate in sorted(
                scored,
                key=lambda c: (
                    self._num((c.get("alpha_sleeve_scores") or {}).get("entry_quality")),
                    self._num(c.get("funnel_score")),
                ),
                reverse=True,
            ):
                if len([c for c in selected if c.get("primary_alpha_sleeve") == "entry_quality"]) >= entry_target:
                    break
                if self._num((candidate.get("alpha_sleeve_scores") or {}).get("entry_quality")) < entry_min:
                    continue
                add(candidate, "entry_quality")

        sleeve_order = [
            "entry_quality",
            "estimate_revision",
            "quality_value",
            "pullback_quality",
            "contrarian_squeeze",
            "momentum",
        ]
        for sleeve in sleeve_order:
            if len(selected) >= max_candidates:
                break
            ranked = sorted(
                scored,
                key=lambda c: (
                    self._num((c.get("alpha_sleeve_scores") or {}).get(sleeve)),
                    self._num(c.get("funnel_score")),
                ),
                reverse=True,
            )
            for candidate in ranked:
                sleeve_score = self._num((candidate.get("alpha_sleeve_scores") or {}).get(sleeve))
                if sleeve_score <= 0:
                    continue
                if add(candidate, sleeve):
                    break

        # Fill remaining slots by overall score while respecting sector caps.
        for candidate in scored:
            if len(selected) >= max_candidates:
                break
            add(candidate, str(candidate.get("primary_alpha_sleeve") or "momentum"))

        # If sector caps left slots empty, relax caps rather than returning too few.
        for candidate in scored:
            if len(selected) >= max_candidates:
                break
            symbol = str(candidate.get("symbol", "") or "").upper()
            if not symbol or symbol in selected_symbols:
                continue
            selected.append(candidate)
            selected_symbols.add(symbol)

        return selected[:max_candidates]

    def _fallback(self, max_candidates: int) -> list[dict]:
        """Emergency fallback: return candidates from legacy _scan_key_tickers().

        Runs fallback movers through _enrich() and _quick_score() to ensure
        the same schema as normal funnel output.
        """
        try:
            from .scanner import _scan_key_tickers
            movers = _scan_key_tickers()
            candidates = movers.get("gainers", []) + movers.get("losers", [])
            candidates.sort(key=lambda x: abs(x.get("change_pct", 0)), reverse=True)

            # Build thin dicts with fields _enrich() expects
            thin = []
            for m in candidates[:max_candidates]:
                thin.append({
                    "symbol": m.get("symbol", ""),
                    "source": "fallback_scan",
                    "combined_score": abs(m.get("change_pct", 0)),
                    "change_pct": m.get("change_pct", 0),
                    "price": m.get("price", 0),
                    "market_cap": m.get("market_cap", 0),
                    "avg_volume": m.get("avg_volume", 0),
                    "name": m.get("name", ""),
                    "sector": m.get("sector", ""),
                    "industry": m.get("industry", ""),
                    "momentum_score": 0.0,
                    "regime_score": 0.0,
                })

            enriched = self._enrich(thin)
            scored = self._quick_score(enriched, top_n=max_candidates)
            return scored
        except Exception as e:
            logger.error(f"[funnel] Fallback scan also failed: {e}")
            return []
