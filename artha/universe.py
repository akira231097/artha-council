"""Dynamic Universe Builder — generates investable stock universe via FMP screener.

Uses FMP company-screener to build a fresh universe of liquid US equities
filtered by market cap, volume, and price. Applies regime-based sector
bonuses/penalties to score each candidate for downstream ranking.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .collector import FMPCollector
from .config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables (env-var overridable; Wave 2 consolidates into config.py)
# ---------------------------------------------------------------------------

# The FMP company-screener caps each response at 1000 rows, which used to
# silently truncate ~45% of the $1B+ US universe (verified 2026-07-02: page 0
# returns 1000 rows down to ~$5.15B, page 1 returns ~791 more down to $1B).
# The screener supports a `page` param — we paginate until an empty/short page.
UNIVERSE_MAX_PAGES: int = int(os.getenv("ARTHA_UNIVERSE_MAX_PAGES", "5"))

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
            limit: Max candidates per screener page (default 1000, FMP's cap)

        Returns:
            List of UniverseCandidate objects, sorted by market_cap descending.
        """
        logger.info(f"[universe] Building universe (regime={regime_type}, overlays={overlays})...")

        raw = self._fetch_screener_pages(
            min_market_cap=min_market_cap,
            min_volume=min_volume,
            min_price=min_price,
            page_limit=limit,
        )

        if not raw:
            logger.warning("[universe] FMP screener returned empty result")
            return []

        candidates = []
        seen_symbols: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            symbol = item.get("symbol", "")
            if not symbol or symbol in seen_symbols:
                continue
            seen_symbols.add(symbol)
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

    def _fetch_screener_pages(
        self,
        *,
        min_market_cap: float,
        min_volume: int,
        min_price: float,
        page_limit: int,
    ) -> list[dict]:
        """Fetch the FULL screener universe by paginating the FMP screener.

        FMP truncates each company-screener response at 1000 rows, which
        previously dropped the bottom ~45% of the $1B+ universe (everything
        below ~$5B). The endpoint supports a `page` param (probed and verified
        against the stable API); we walk pages until an empty/short page or
        UNIVERSE_MAX_PAGES.

        Falls back to marketCap-band iteration if pagination is unavailable,
        and finally to the single-shot legacy call.
        """
        pages: list[dict] = []
        base_params = {
            "marketCapMoreThan": str(int(min_market_cap)),
            "volumeMoreThan": str(int(min_volume)),
            "priceMoreThan": str(min_price),
            "country": "US",
            "isActivelyTrading": "true",
            "isEtf": "false",
            "isFund": "false",
            "limit": str(int(page_limit)),
        }

        get = getattr(self.fmp, "_get", None)
        if callable(get):
            for page in range(max(1, UNIVERSE_MAX_PAGES)):
                try:
                    rows = get("company-screener", {**base_params, "page": str(page)})
                except Exception as e:
                    logger.warning(f"[universe] Screener page {page} failed: {e}")
                    break
                if not rows or not isinstance(rows, list):
                    break
                pages.extend(r for r in rows if isinstance(r, dict))
                logger.info(f"[universe] Screener page {page}: {len(rows)} rows")
                if len(rows) < page_limit:
                    break
            if pages:
                return pages

            # Pagination returned nothing — iterate marketCap bands instead.
            bands = [
                (int(min_market_cap), 3_000_000_000),
                (3_000_000_000, 10_000_000_000),
                (10_000_000_000, 50_000_000_000),
                (50_000_000_000, None),
            ]
            for low, high in bands:
                if low < int(min_market_cap):
                    continue
                band_params = dict(base_params)
                band_params["marketCapMoreThan"] = str(low)
                if high is not None:
                    band_params["marketCapLowerThan"] = str(high)
                try:
                    rows = get("company-screener", band_params)
                except Exception as e:
                    logger.warning(f"[universe] Screener band {low}-{high} failed: {e}")
                    continue
                if rows and isinstance(rows, list):
                    pages.extend(r for r in rows if isinstance(r, dict))
                    logger.info(f"[universe] Screener band {low}-{high}: {len(rows)} rows")
            if pages:
                return pages

        # Last resort: legacy single-shot call (truncates at page_limit rows).
        rows = self.fmp.screener(
            market_cap_more_than=int(min_market_cap),
            volume_more_than=int(min_volume),
            price_more_than=min_price,
            country="US",
            is_actively_trading=True,
            is_etf=False,
            is_fund=False,
            limit=page_limit,
        )
        return [r for r in (rows or []) if isinstance(r, dict)]

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
