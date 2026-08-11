"""Deterministic sector/industry plumbing for held positions."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .journal import DecisionJournal
from .portfolio import PORTFOLIO_FILE, Portfolio, Position


def attach_position_classification(
    position: Position,
    journal: DecisionJournal,
    *,
    dossier_path: str | None = None,
) -> bool:
    """Attach the latest stored Council classification; return whether it changed."""
    feature = journal.get_latest_decision_feature(
        position.ticker,
        dossier_path=dossier_path,
    )
    if not feature:
        return False
    sector = str(feature.get("sector") or "").strip()
    industry = str(feature.get("industry") or "").strip()
    if not sector:
        return False
    source = f"decision_features:{feature.get('dossier_path') or feature.get('id')}"
    updated_at = str(feature.get("generated_at") or feature.get("created_at") or "")
    before = (
        position.sector,
        position.industry,
        position.classification_source,
        position.classification_updated_at,
    )
    position.sector = sector
    position.industry = industry
    position.classification_source = source
    position.classification_updated_at = updated_at
    return before != (
        position.sector,
        position.industry,
        position.classification_source,
        position.classification_updated_at,
    )


def backfill_portfolio_classifications(
    journal: DecisionJournal,
    *,
    portfolio_path: str | Path = PORTFOLIO_FILE,
) -> dict[str, Any]:
    """Backfill held-position sectors from the point-in-time feature warehouse."""
    path = Path(portfolio_path)
    portfolio = Portfolio.load(path)
    updated: list[str] = []
    unresolved: list[str] = []
    for position in portfolio.positions:
        if attach_position_classification(position, journal):
            updated.append(position.ticker.upper())
        if not str(position.sector or "").strip():
            unresolved.append(position.ticker.upper())
    if updated:
        portfolio.save(path)
    return {
        "positions_seen": len(portfolio.positions),
        "updated": sorted(updated),
        "unresolved": sorted(unresolved),
    }
