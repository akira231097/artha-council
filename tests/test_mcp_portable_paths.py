from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestPortableRuntimePaths(unittest.TestCase):
    def test_openclaw_assets_are_portable_and_account_agnostic(self) -> None:
        from artha.paths import runtime_asset_path

        for name in (
            "robinhood_auto_trade_runner.mjs",
            "robinhood_manual_review_runner.mjs",
            "robinhood_snapshot_sync.mjs",
        ):
            script = runtime_asset_path("scripts", name)
            self.assertTrue(script.is_file(), script)
            source = script.read_text(encoding="utf-8")
            self.assertNotIn("/opt/homebrew/bin/node", source)
            self.assertNotIn('"0195"', source)
            self.assertIn("mode: 0o600", source)
        snapshot = runtime_asset_path("scripts", "robinhood_snapshot_sync.mjs")
        self.assertIn('"-m",', snapshot.read_text(encoding="utf-8"))

    def test_explicit_data_directory_controls_core_and_mcp_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp) / "persistent-data"
            env = dict(os.environ)
            env["ARTHA_DATA_DIR"] = str(data_dir)
            env["ARTHA_MCP_DATA_DIR"] = str(data_dir)
            code = """
import json
from artha.paths import DATA_DIR
from artha.accuracy import ACCURACY_FILE
from artha.dossier import DOSSIER_DIR
from artha.journal import DB_PATH
from artha.portfolio import PORTFOLIO_FILE
from artha.portfolio_state import BROKER_SNAPSHOT_PATH, PORTFOLIO_JSON_PATH
from artha.sell_dossier import SELL_DOSSIER_DIR
from artha.supervisor import SUPERVISOR_DIR
print(json.dumps({
    'data': str(DATA_DIR),
    'accuracy': str(ACCURACY_FILE),
    'dossiers': str(DOSSIER_DIR),
    'db': str(DB_PATH),
    'portfolio': str(PORTFOLIO_FILE),
    'portfolio_state': str(PORTFOLIO_JSON_PATH),
    'snapshot': str(BROKER_SNAPSHOT_PATH),
    'sell_dossiers': str(SELL_DOSSIER_DIR),
    'supervisor': str(SUPERVISOR_DIR),
}))
"""
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=Path(temp),
                env=env,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            paths = json.loads(completed.stdout)
            root = data_dir.resolve()
            for value in paths.values():
                self.assertTrue(
                    Path(value).resolve() == root
                    or root in Path(value).resolve().parents,
                    msg=f"Runtime path escaped configured data directory: {value}",
                )


if __name__ == "__main__":
    unittest.main()
