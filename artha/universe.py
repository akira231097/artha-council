"""Dynamic Universe Builder — generates investable stock universe via FMP screener.

Uses FMP company-screener to build a fresh universe of liquid US equities
filtered by market cap, volume, and price. Applies regime-based sector
bonuses/penalties to score each candidate for downstream ranking.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .collector import FMPCollector
from .config import Config

logger = logging.getLogger(__name__)

# Sector mapping from FMP sector names to REGIME_TAXONOMY sector names
FMP_SECTOR_ALIASES: dict[str, list[str]] = {
    "Technology": ["technology", "tech", "semiconductors"],
    "Financial Services": ["financials", "banks", "financial_services"],
    "Healthcare": ["healthcare", "pharma"],
    "Consumer Cyclical": ["discretionary", "consumer_discretionary", "cyclicals"],
    "Communication Services": ["communication", "media"],
    "Industrials": ["industrials", "defense", "aerospace"],
    "Consumer Defensive": ["consumer_staples", "staples", "defensive"],
    "Energy": ["energy", "oil", "commodities"],
    "Basic Materials": ["materials", "commodities", "mining"],
    "Real Estate": ["real_estate", "reits"],
    "Utilities": ["utilities"],
}


@dataclass
class UniverseCandidate:
    """A single stock candidate from the universe build."""
    symbol: str = ""
    name: str = ""
    sector: str = ""
    industry: str = ""
    market_cap: float = 0.0
    price: float = 0.0
    volume: int = 0
    beta: Optional[float] = None
    regime_score: float = 0.0     # Bonus/penalty from regime fit (-5 to +15, capped)
    source: str = "fmp_screener"


class UniverseBuilder:
    """Builds a dynamic investable universe via FMP screener."""

    def __init__(self):
        self.fmp = FMPCollector()

    def build_universe(
        self,
        regime_type: Optional[str] = None,
        overlays: Optional[list[str]] = None,
        min_market_cap: float = 1_000_000_000,
        min_volume: int = 100_000,
        min_price: float = 5.0,
        limit: int = 1000,
    ) -> list[UniverseCandidate]:
        """Build a fresh investable universe from FMP screener.

        Args:
            regime_type: Base regime (e.g. "goldilocks", "stagflation")
            overlays: Active overlay regime keys (e.g. ["ai_tech_momentum"])
            min_market_cap: Minimum market cap in USD (default $1B)
            min_volume: Minimum daily volume (default 100K shares)
            min_price: Minimum price (default $5)
            limit: Max candidates from screener (default 1000)

        Returns:
            List of UniverseCandidate objects, sorted by market_cap descending.
        """
        logger.info(f"[universe] Building universe (regime={regime_type}, overlays={overlays})...")

        raw = self.fmp.screener(
            market_cap_more_than=int(min_market_cap),
            volume_more_than=int(min_volume),
            price_more_than=min_price,
            country="US",
            is_actively_trading=True,
            is_etf=False,
            is_fund=False,
            limit=limit,
        )

        if not raw:
            logger.warning("[universe] FMP screener returned empty result")
            return []

        candidates = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            symbol = item.get("symbol", "")
            if not symbol:
                continue
            try:
                candidate = UniverseCandidate(
                    symbol=symbol,
                    name=item.get("companyName", ""),
                    sector=item.get("sector", ""),
                    industry=item.get("industry", ""),
                    market_cap=float(item.get("marketCap", 0) or 0),
                    price=float(item.get("price", 0) or 0),
                    volume=int(item.get("volume", 0) or 0),
                    beta=float(item["beta"]) if item.get("beta") is not None else None,
                )
                candidates.append(candidate)
            except (TypeError, ValueError) as e:
                logger.debug(f"[universe] Skipping {symbol}: {e}")
                continue

        logger.info(f"[universe] Raw screener: {len(candidates)} candidates")

        # Apply regime filter scoring if regime specified
        if regime_type or overlays:
            candidates = self.apply_regime_filter(candidates, regime_type, overlays or [])

        # Sort by market cap (largest first for quality bias)
        candidates.sort(key=lambda c: c.market_cap, reverse=True)
        return candidates

    def apply_regime_filter(
        self,
        universe: list[UniverseCandidate],
        regime_type: Optional[str],
        overlays: list[str],
    ) -> list[UniverseCandidate]:
        """Apply regime-based sector bonuses/penalties.

        +10 for beneficiary sectors, -5 for avoid sectors.
        Overlays stack with base regime.

        Args:
            universe: List of candidates to score
            regime_type: Base regime key
            overlays: List of overlay regime keys

        Returns:
            Same candidates with regime_score updated (no filtering).
        """
        try:
            from .regime_mapping import REGIME_TAXONOMY
        except ImportError:
            logger.warning("[universe] Could not import REGIME_TAXONOMY")
            return universe

        # Collect beneficiary and avoid sector aliases from active regimes
        beneficiary_aliases: set[str] = set()
        avoid_aliases: set[str] = set()

        active_regimes = []
        if regime_type and regime_type in REGIME_TAXONOMY:
            active_regimes.append(REGIME_TAXONOMY[regime_type])
        for overlay in overlays:
            if overlay in REGIME_TAXONOMY:
                active_regimes.append(REGIME_TAXONOMY[overlay])

        for r in active_regimes:
            for stock in r.get("beneficiary_stocks", []):
                beneficiary_aliases.add(stock.upper())
            for sector in r.get("beneficiary_sectors", []):
                beneficiary_aliases.add(sector.lower())
            for avoid in r.get("avoid_sectors", []):
                avoid_aliases.add(avoid.lower())

        # Build a set of FMP sector names that map to beneficiary/avoid aliases
        beneficiary_fmp_sectors: set[str] = set()
        avoid_fmp_sectors: set[str] = set()

        _ETF_SECTOR_MAP = {
            "XLE": "Energy", "XLF": "Financial Services", "XLK": "Technology",
            "XLI": "Industrials", "XLV": "Healthcare", "XLP": "Consumer Defensive",
            "XLU": "Utilities", "XLY": "Consumer Cyclical", "XLC": "Communication Services",
            "XLRE": "Real Estate", "SMH": "Technology", "ITA": "Industrials",
        }

        # Map ETF tickers from beneficiary_etfs to FMP sectors
        for r in active_regimes:
            beneficiary_etfs = {e.upper() for e in r.get("beneficiary_etfs", [])}
            for etf, sector in _ETF_SECTOR_MAP.items():
                if etf in beneficiary_etfs:
                    beneficiary_fmp_sectors.add(sector)

        # Map FMP sector aliases against beneficiary/avoid alias sets
        for fmp_sector, aliases in FMP_SECTOR_ALIASES.items():
            alias_set = {a.lower() for a in aliases}
            if alias_set & beneficiary_aliases:
                beneficiary_fmp_sectors.add(fmp_sector)
            if alias_set & avoid_aliases:
                avoid_fmp_sectors.add(fmp_sector)

        # Maximum regime bonus cap to prevent double-counting.
        # Without this, a stock like NVDA in goldilocks gets +10 (tech sector)
        # AND +10 (individually named) = +20, which unfairly dominates rankings.
        MAX_REGIME_BONUS = 15.0

        for candidate in universe:
            score = 0.0
            sector = candidate.sector

            if sector in beneficiary_fmp_sectors:
                score += 10.0

            if sector in avoid_fmp_sectors:
                score -= 5.0

            # Bonus for individually named beneficiary stocks
            if candidate.symbol.upper() in beneficiary_aliases:
                score += 5.0  # Reduced from 10 to prevent double-counting

            # Cap the positive bonus to prevent any single stock from getting
            # an outsized regime advantage over peers with better momentum
            if score > MAX_REGIME_BONUS:
                score = MAX_REGIME_BONUS

            candidate.regime_score = score

        return universe
