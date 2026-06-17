"""Point-in-time sell-side dossiers for active position reviews."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dossier import _json_safe

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SELL_DOSSIER_DIR = DATA_DIR / "sell_dossiers"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_obj(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if hasattr(value, "__dict__"):
        return _json_safe({k: v for k, v in vars(value).items() if not k.startswith("_")})
    return _json_safe(value)


def _compact_stock_snapshot(stock_data: dict[str, Any]) -> dict[str, Any]:
    stock_data = stock_data or {}
    quote = stock_data.get("quote") if isinstance(stock_data.get("quote"), dict) else {}
    yf_quote = stock_data.get("yf_quote") if isinstance(stock_data.get("yf_quote"), dict) else {}
    massive_quote = stock_data.get("massive_quote") if isinstance(stock_data.get("massive_quote"), dict) else {}
    technicals = stock_data.get("technicals") if isinstance(stock_data.get("technicals"), dict) else {}
    return {
        "ticker": _json_safe(stock_data.get("ticker")),
        "quote": _json_safe({k: quote.get(k) for k in ("price", "changesPercentage", "volume", "marketCap") if k in quote}),
        "yf_quote": _json_safe({k: yf_quote.get(k) for k in ("price", "bid", "ask", "volume") if k in yf_quote}),
        "massive_quote": _json_safe({k: massive_quote.get(k) for k in ("price", "bid", "ask", "volume") if k in massive_quote}),
        "technicals": _json_safe({k: technicals.get(k) for k in ("rsi", "sma_20", "sma_50", "sma_200", "macd") if k in technicals}),
        "earnings_context": _json_safe(stock_data.get("earnings_context") or {}),
        "recommendation_trends": _json_safe(stock_data.get("recommendation_trends") or {}),
        "price_target_consensus": _json_safe(stock_data.get("price_target_consensus") or stock_data.get("price_target") or {}),
        "source_keys": sorted(str(k) for k in stock_data.keys()),
    }


def write_sell_dossier(
    decision: Any,
    thesis: Any,
    stock_data: dict[str, Any],
    macro_data: dict[str, Any] | None = None,
    trigger_type: str = "",
) -> str:
    """Write a sell-side audit artifact and return its absolute path."""
    generated = _utcnow()
    ticker = str(getattr(decision, "ticker", None) or getattr(thesis, "ticker", "UNKNOWN") or "UNKNOWN").upper()
    dossier = {
        "schema_version": 1,
        "kind": "sell_review",
        "ticker": ticker,
        "generated_at": generated.isoformat(),
        "trigger_type": trigger_type or getattr(decision, "trigger_type", ""),
        "decision": _safe_obj(decision),
        "thesis": _safe_obj(thesis),
        "stock_snapshot": _compact_stock_snapshot(stock_data),
        "macro_data_keys": sorted(str(k) for k in (macro_data or {}).keys()),
    }
    out_dir = SELL_DOSSIER_DIR / generated.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}_{generated.strftime('%Y%m%d_%H%M%S')}_{str(getattr(decision, 'session_id', ''))[:8]}.json"
    out_path.write_text(json.dumps(dossier, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Sell dossier written for %s: %s", ticker, out_path)
    return str(out_path)
