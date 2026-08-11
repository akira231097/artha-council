#!/usr/bin/env python3
"""Backfill held-position sectors from Artha's stored decision features.

The default mode operates on temporary copies. ``--apply`` creates timestamped
local backups before updating portfolio metadata. No network or broker tool is
used by this script.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from artha.journal import DB_PATH, DecisionJournal
from artha.portfolio import PORTFOLIO_FILE, Portfolio
from artha.position_classification import backfill_portfolio_classifications


def _backup_sqlite(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as src, sqlite3.connect(target) as dst:
        src.backup(dst)


def _run(db_path: Path, portfolio_path: Path) -> dict:
    journal = DecisionJournal(db_path=db_path)
    result = backfill_portfolio_classifications(
        journal,
        portfolio_path=portfolio_path,
    )
    portfolio = Portfolio.load(portfolio_path)
    result["classifications"] = {
        position.ticker.upper(): {
            "sector": position.sector,
            "industry": position.industry,
            "source": position.classification_source,
        }
        for position in sorted(portfolio.positions, key=lambda item: item.ticker.upper())
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--portfolio", type=Path, default=PORTFOLIO_FILE)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Update the selected portfolio after creating local backups.",
    )
    args = parser.parse_args()
    db_path = args.db.resolve()
    portfolio_path = args.portfolio.resolve()
    if not db_path.exists():
        raise SystemExit(f"Database does not exist: {db_path}")
    if not portfolio_path.exists():
        raise SystemExit(f"Portfolio does not exist: {portfolio_path}")

    if not args.apply:
        with tempfile.TemporaryDirectory(prefix="artha-sector-backfill-") as tmp:
            tmp_dir = Path(tmp)
            db_copy = tmp_dir / "artha.db"
            portfolio_copy = tmp_dir / "portfolio.json"
            _backup_sqlite(db_path, db_copy)
            shutil.copy2(portfolio_path, portfolio_copy)
            result = _run(db_copy, portfolio_copy)
            print(json.dumps({"mode": "dry_run", **result}, indent=2, sort_keys=True))
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = portfolio_path.parent / "backups" / f"position-classification-{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(portfolio_path, backup_dir / portfolio_path.name)
    _backup_sqlite(db_path, backup_dir / db_path.name)
    result = _run(db_path, portfolio_path)
    print(
        json.dumps(
            {"mode": "apply", "backup_dir": str(backup_dir), **result},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
