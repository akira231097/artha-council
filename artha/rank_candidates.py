"""Machine Ranking Layer — scores and ranks universe candidates by momentum.

Uses yfinance batch download for efficiency. Computes momentum scores
and combines with regime fit scores to produce a final ranked list.

Momentum methodology follows academic best practices:
  - Skip the most recent month (short-term reversal effect,
    Jegadeesh & Titman 1993, Fama-French, Carhart, AQR)
  - Use 12-1 month return as the primary signal (strongest documented
    momentum factor in finance)
  - Include 3-month (skip-1) as a secondary signal for recent trend
  - Volatility penalty for crash protection (Barroso & Santa-Clara 2015)
"""
from __future__ import annotations

import logging
import gc
import time
from datetime import datetime, timezone
from typing import Optional

from .config import Config
from .collector import YFinanceCollector

logger = logging.getLogger(__name__)

# Momentum score weights
# 12-1 month return is the strongest academic signal
MOMENTUM_RETURN_12M_WEIGHT = 0.5
MOMENTUM_RETURN_3M_WEIGHT = 0.3
MOMENTUM_RETURN_1M_WEIGHT = 0.2
MOMENTUM_VOL_PENALTY = 0.2

# Number of trading days to skip (most recent month) to avoid
# short-term reversal effect per Jegadeesh & Titman (1993)
SKIP_DAYS = 22  # ~1 month of trading days


def compute_momentum_score(
    return_1m: Optional[float],
    return_3m: Optional[float],
    vol_20d: Optional[float],
    return_12m: Optional[float] = None,
) -> float:
    """Compute a momentum score from return and volatility data.

    Formula (with 12-month data):
        score = 0.5 * return_12m + 0.3 * return_3m + 0.2 * return_1m
                - 0.2 * vol_penalty

    Fallback (without 12-month data, e.g. IPOs < 1 year old):
        score = 0.6 * return_3m + 0.4 * return_1m - 0.2 * vol_penalty

    All returns should use the skip-month rule: measured from t-N to t-22
    (skipping the most recent ~22 trading days) to avoid short-term
    reversal contamination.

    Where vol_penalty = max(0, vol_20d - 20) to penalize excessive volatility.
    Returns 0.0 if all inputs are None.

    Args:
        return_1m: 1-month return in percent (skip-adjusted: t-44 to t-22)
        return_3m: 3-month return in percent (skip-adjusted: t-85 to t-22)
        vol_20d: 20-day annualized volatility in percent
        return_12m: 12-month return in percent (skip-adjusted: t-274 to t-22)

    Returns:
        Float momentum score (higher = better momentum)
    """
    r1m = float(return_1m) if return_1m is not None else 0.0
    r3m = float(return_3m) if return_3m is not None else 0.0
    r12m = float(return_12m) if return_12m is not None else None
    vol = float(vol_20d) if vol_20d is not None else 20.0

    # Penalize only volatility above 20% annualized
    vol_penalty = max(0.0, vol - 20.0)

    if r12m is not None:
        # Full formula with 12-month signal (strongest academic signal)
        score = (
            MOMENTUM_RETURN_12M_WEIGHT * r12m
            + MOMENTUM_RETURN_3M_WEIGHT * r3m
            + MOMENTUM_RETURN_1M_WEIGHT * r1m
            - MOMENTUM_VOL_PENALTY * vol_penalty
        )
    else:
        # Fallback for stocks with < 1 year of history
        score = (
            0.6 * r3m
            + 0.4 * r1m
            - MOMENTUM_VOL_PENALTY * vol_penalty
        )

    return round(score, 4)


def rank_universe(
    universe: list,
    regime_type: Optional[str] = None,
    overlays: Optional[list[str]] = None,
    top_n: int = 50,
) -> list[dict]:
    """Rank universe candidates by momentum + regime fit.

    Downloads 1 year of price history via yfinance batch download,
    computes momentum scores using the skip-month rule (Jegadeesh & Titman),
    adds regime sector bonuses, and returns top N candidates sorted by
    combined score.

    The skip-month rule: all return calculations end at t-22 (skipping the
    most recent ~22 trading days) to avoid short-term reversal contamination.
    This is the standard used by Fama-French, Carhart, AQR, and all major
    momentum factor constructions since 1993.

    Args:
        universe: List of UniverseCandidate objects (from UniverseBuilder)
        regime_type: Active base regime (for logging/context)
        overlays: Active overlay regime keys
        top_n: Number of top candidates to return (default 50)

    Returns:
        List of dicts with symbol, name, sector, momentum_score, regime_score,
        combined_score, return_1m, return_3m, return_12m, vol_20d, price,
        market_cap.
    """
    import yfinance as yf
    import numpy as np

    if not universe:
        logger.warning("[rank] Empty universe — nothing to rank")
        return []

    symbols = [c.symbol for c in universe]
    logger.info(f"[rank] Downloading 1Y price data for {len(symbols)} tickers...")

    # Batch download — single yfinance call, 1 year for 12-month momentum
    prices = {}
    try:
        batch_size = max(1, int(getattr(Config, "YFINANCE_BATCH_SIZE", 100) or 100))
        started = time.monotonic()
        total_timeout = max(5, int(getattr(Config, "YFINANCE_RANK_TOTAL_TIMEOUT_SECONDS", 120) or 120))
        for i in range(0, len(symbols), batch_size):
            elapsed = time.monotonic() - started
            if elapsed >= total_timeout:
                logger.warning(
                    "[rank] yfinance ranking budget exhausted after %.1fs; ranked with %d/%d downloaded price histories",
                    elapsed,
                    len(prices),
                    len(symbols),
                )
                break
            chunk = symbols[i : i + batch_size]
            tickers_str = " ".join(chunk)
            df = None
            try:
                remaining_timeout = max(1, int(min(Config.YFINANCE_DOWNLOAD_TIMEOUT_SECONDS, total_timeout - elapsed)))
                df = yf.download(
                    tickers_str,
                    period="1y",
                    progress=False,
                    group_by="ticker",
                    threads=Config.YFINANCE_THREADS,
                    timeout=remaining_timeout,
                )

                if df is None or df.empty:
                    continue

                cols = df.columns
                is_multi = getattr(cols, "nlevels", 1) == 2

                for sym in chunk:
                    try:
                        if is_multi:
                            lv0 = set(cols.get_level_values(0))
                            lv1 = set(cols.get_level_values(1))
                            if sym in lv0 and "Close" in lv1:
                                series = df[(sym, "Close")].dropna()
                            elif "Close" in lv0 and sym in lv1:
                                series = df[("Close", sym)].dropna()
                            else:
                                series = None
                        else:
                            series = df["Close"].dropna() if "Close" in df.columns else None

                        if series is not None and len(series) >= 10:
                            prices[sym] = series.values.astype(float)
                    except Exception:
                        pass
            except Exception as chunk_e:
                logger.warning(
                    "[rank] yfinance chunk failed for %d ticker(s) starting at %s: %s",
                    len(chunk),
                    chunk[0] if chunk else "?",
                    chunk_e,
                )
            finally:
                if df is not None:
                    del df
                YFinanceCollector.cleanup_caches()
                gc.collect()

    except Exception as e:
        logger.error(f"[rank] yfinance batch download failed: {e}")

    logger.info(f"[rank] Got price data for {len(prices)}/{len(symbols)} tickers")

    # Build candidate lookup
    candidate_map = {c.symbol: c for c in universe}

    ranked = []
    for sym, price_arr in prices.items():
        candidate = candidate_map.get(sym)
        if not candidate:
            continue

        n = len(price_arr)
        latest = price_arr[-1]

        # ---------------------------------------------------------------
        # SKIP-MONTH RULE (Jegadeesh & Titman 1993)
        # All momentum returns are measured up to t-SKIP_DAYS, NOT today.
        # This avoids short-term reversal contamination.
        # ---------------------------------------------------------------
        skip = min(SKIP_DAYS, n - 1)  # Don't skip more data than we have
        skip_price = price_arr[-(skip + 1)]  # Price ~1 month ago

        # 1-month return (t-44 to t-22): measures the month BEFORE the skip
        return_1m = None
        if n >= (SKIP_DAYS + 22):
            return_1m = ((skip_price / price_arr[-(SKIP_DAYS + 22)]) - 1) * 100
        elif n >= (skip + 5):
            return_1m = ((skip_price / price_arr[0]) - 1) * 100

        # 3-month return (t-85 to t-22): measures 3 months before the skip
        return_3m = None
        if n >= (SKIP_DAYS + 63):
            return_3m = ((skip_price / price_arr[-(SKIP_DAYS + 63)]) - 1) * 100
        elif n >= (skip + 10):
            return_3m = ((skip_price / price_arr[0]) - 1) * 100

        # 12-month return (t-274 to t-22): the strongest momentum signal
        return_12m = None
        if n >= (SKIP_DAYS + 252):
            return_12m = ((skip_price / price_arr[-(SKIP_DAYS + 252)]) - 1) * 100
        elif n >= (SKIP_DAYS + 126):
            # At least 6 months + skip available — use what we have
            return_12m = ((skip_price / price_arr[0]) - 1) * 100

        # 20-day annualized volatility (uses recent data, NOT skip-adjusted)
        vol_20d = None
        if n >= 21:
            daily_rets = np.diff(price_arr[-21:]) / price_arr[-21:-1]
            vol_20d = float(np.std(daily_rets, ddof=1) * np.sqrt(252) * 100)

        mom_score = compute_momentum_score(return_1m, return_3m, vol_20d, return_12m)
        regime_score = candidate.regime_score

        # Combined score: momentum + regime fit
        combined = mom_score + regime_score

        ranked.append({
            "symbol": sym,
            "name": candidate.name,
            "sector": candidate.sector,
            "industry": candidate.industry,
            "market_cap": candidate.market_cap,
            "avg_volume": candidate.volume,
            "price": round(latest, 2),
            "momentum_score": round(mom_score, 2),
            "regime_score": round(regime_score, 2),
            "combined_score": round(combined, 2),
            "return_1m": round(return_1m, 2) if return_1m is not None else None,
            "return_3m": round(return_3m, 2) if return_3m is not None else None,
            "return_12m": round(return_12m, 2) if return_12m is not None else None,
            "vol_20d": round(vol_20d, 2) if vol_20d is not None else None,
        })

    # Sort by combined score descending
    ranked.sort(key=lambda x: x["combined_score"], reverse=True)

    top = ranked[:top_n]
    logger.info(
        f"[rank] Ranked {len(ranked)} candidates → top {len(top)} "
        f"(regime={regime_type}, overlays={overlays})"
    )
    return top
