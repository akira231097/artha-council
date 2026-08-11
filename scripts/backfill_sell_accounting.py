#!/usr/bin/env python3
"""Audit/backfill sell-fill learning records without touching broker tools.

The default mode runs against temporary SQLite/portfolio copies. ``--apply``
creates timestamped local backups before updating the production accounting
tables. This script never reviews, places, or cancels an order.
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

from artha.fill_finalizer import backfill_sell_fill_accounting
from artha.journal import DB_PATH, DecisionJournal
from artha.portfolio import PORTFOLIO_FILE


def _backup_sqlite(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as src, sqlite3.connect(target) as dst:
        src.backup(dst)


def _run(db_path: Path, portfolio_path: Path) -> dict:
    journal = DecisionJournal(db_path=db_path)
    return backfill_sell_fill_accounting(journal, portfolio_path=portfolio_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--portfolio", type=Path, default=PORTFOLIO_FILE)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Update the selected database after creating local backups.",
    )
    args = parser.parse_args()
    db_path = args.db.resolve()
    portfolio_path = args.portfolio.resolve()
    if not db_path.exists():
        raise SystemExit(f"Database does not exist: {db_path}")

    if not args.apply:
        with tempfile.TemporaryDirectory(prefix="artha-sell-backfill-") as tmp:
            tmp_dir = Path(tmp)
            db_copy = tmp_dir / "artha.db"
            portfolio_copy = tmp_dir / "portfolio.json"
            _backup_sqlite(db_path, db_copy)
            if portfolio_path.exists():
                shutil.copy2(portfolio_path, portfolio_copy)
            result = _run(db_copy, portfolio_copy)
            print(json.dumps({"mode": "dry_run", **result}, indent=2, sort_keys=True))
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = db_path.parent / "backups" / f"sell-accounting-{stamp}"
    _backup_sqlite(db_path, backup_dir / db_path.name)
    if portfolio_path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(portfolio_path, backup_dir / portfolio_path.name)
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
