from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from artha_mcp import provenance
from scripts.artha_source_sync import _manifest_from_ref, _scan_staged, _source_patch
from scripts.check_release_integrity import (
    check,
    committed_source_fingerprint,
    dirty_managed_paths,
)

ROOT = Path(__file__).resolve().parent.parent


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.invalid")


class TestMCPProvenance(unittest.TestCase):
    def test_manifest_proves_runtime_bytes_and_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "artha").mkdir()
            (root / "artha_mcp").mkdir()
            (root / "artha" / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "artha_mcp" / "server.py").write_text("READY = True\n", encoding="utf-8")
            digest = provenance.runtime_tree_fingerprint(root)
            manifest = root / provenance.MANIFEST_NAME
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "release_version": provenance.__version__,
                        "runtime_tree_sha256": digest,
                        "source_provenance": {
                            "commit": "a" * 40,
                            "tree_sha256": "b" * 64,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(provenance, "PROJECT_ROOT", root),
                patch.object(provenance, "_manifest_path", return_value=manifest),
                patch.dict(
                    os.environ,
                    {
                        "ARTHA_BUILD_COMMIT": "c" * 40,
                        "ARTHA_BUILD_VERSION": provenance.__version__,
                    },
                    clear=False,
                ),
            ):
                self.assertEqual(provenance.build_provenance()["status"], "PASS")
                (root / "artha" / "core.py").write_text("VALUE = 2\n", encoding="utf-8")
                result = provenance.build_provenance()
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("fingerprint", result["errors"][0].lower())

    def test_current_repository_release_integrity_passes(self) -> None:
        result = check(ROOT)
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(len(set(result["versions"].values())), 1)


class TestSourcePromotionBoundary(unittest.TestCase):
    def test_remote_manifest_is_read_from_requested_ref(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            _init_repo(repo)
            manifest = repo / provenance.MANIFEST_NAME
            manifest.write_text('{"release_version":"main"}\n', encoding="utf-8")
            _git(repo, "add", provenance.MANIFEST_NAME)
            _git(repo, "commit", "-qm", "main manifest")
            main_commit = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            manifest.write_text('{"release_version":"working"}\n', encoding="utf-8")

            loaded = _manifest_from_ref(repo, main_commit)

            self.assertEqual(loaded["release_version"], "main")

    def test_source_fingerprint_ignores_runtime_data_but_detects_core(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            (source / "artha").mkdir()
            (source / "dashboard").mkdir()
            (source / "data").mkdir()
            (source / "scripts").mkdir()
            (source / "artha" / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
            (source / "dashboard" / "app.js").write_text("const ready = true;\n", encoding="utf-8")
            (source / "run.py").write_text("print('ok')\n", encoding="utf-8")
            runner = source / "scripts" / "robinhood_snapshot_sync.mjs"
            runner.write_text("const version = 1;\n", encoding="utf-8")
            ignored_script = source / "scripts" / "private_deploy.sh"
            ignored_script.write_text("private deploy\n", encoding="utf-8")
            (source / "data" / "snapshot.json").write_text('{"account":"private"}\n')
            _init_repo(source)
            _git(source, "add", ".")
            _git(source, "commit", "-qm", "base")
            base, first, _ = committed_source_fingerprint(source)

            (source / "data" / "snapshot.json").write_text('{"account":"changed"}\n')
            _git(source, "add", "data/snapshot.json")
            _git(source, "commit", "-qm", "runtime only")
            _, second, _ = committed_source_fingerprint(source)
            self.assertEqual(first, second)

            (source / "artha" / "core.py").write_text("VALUE = 2\n", encoding="utf-8")
            _git(source, "add", "artha/core.py")
            _git(source, "commit", "-qm", "core update")
            head, third, _ = committed_source_fingerprint(source)
            self.assertNotEqual(first, third)
            patch_bytes, paths = _source_patch(source, base, head)
            self.assertEqual(paths, ["artha/core.py"])
            self.assertIn(b"VALUE = 2", patch_bytes)
            self.assertNotIn(b"private", patch_bytes)
            self.assertNotIn(b"changed", patch_bytes)

            runner.write_text("const version = 2;\n", encoding="utf-8")
            ignored_script.write_text("changed private deploy\n", encoding="utf-8")
            _git(source, "add", "scripts")
            _git(source, "commit", "-qm", "runner update")
            runner_head, runner_fingerprint, _ = committed_source_fingerprint(source)
            self.assertNotEqual(third, runner_fingerprint)
            runner_patch, runner_paths = _source_patch(source, head, runner_head)
            self.assertEqual(runner_paths, ["scripts/robinhood_snapshot_sync.mjs"])
            self.assertIn(b"const version = 2", runner_patch)
            self.assertNotIn(b"private deploy", runner_patch)

    def test_source_patch_preserves_a_nonoverlapping_public_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            public = root / "public"
            source.mkdir()
            public.mkdir()
            for repo, header in ((source, "private header"), (public, "portable header")):
                (repo / "artha").mkdir()
                lines = [header, *(f"shared {index}" for index in range(1, 10)), "VALUE = 1"]
                (repo / "artha" / "core.py").write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )
                _init_repo(repo)
                _git(repo, "add", ".")
                _git(repo, "commit", "-qm", "base")
            baseline = subprocess.run(
                ["git", "-C", str(source), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            source_file = source / "artha" / "core.py"
            source_file.write_text(
                source_file.read_text(encoding="utf-8").replace("VALUE = 1", "VALUE = 2"),
                encoding="utf-8",
            )
            _git(source, "add", "artha/core.py")
            _git(source, "commit", "-qm", "behavior fix")
            head = subprocess.run(
                ["git", "-C", str(source), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            patch_bytes, paths = _source_patch(source, baseline, head)

            applied = subprocess.run(
                ["git", "-C", str(public), "apply", "--index", "-"],
                input=patch_bytes,
                capture_output=True,
                check=False,
            )

            self.assertEqual(applied.returncode, 0, applied.stderr.decode())
            self.assertEqual(paths, ["artha/core.py"])
            rendered = (public / "artha" / "core.py").read_text(encoding="utf-8")
            self.assertIn("portable header", rendered)
            self.assertIn("VALUE = 2", rendered)

    def test_uncommitted_core_blocks_but_uncommitted_data_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            (source / "artha").mkdir()
            (source / "data").mkdir()
            (source / "artha" / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
            (source / "data" / "state.json").write_text("{}\n", encoding="utf-8")
            _init_repo(source)
            _git(source, "add", ".")
            _git(source, "commit", "-qm", "base")
            (source / "data" / "state.json").write_text('{"new":true}\n')
            self.assertEqual(dirty_managed_paths(source), [])
            (source / "artha" / "core.py").write_text("VALUE = 2\n", encoding="utf-8")
            self.assertEqual(dirty_managed_paths(source), ["artha/core.py"])

    def test_staged_safety_scan_rejects_secret_and_non_allowlisted_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "artha").mkdir()
            (root / "dashboard").mkdir()
            (root / "docs").mkdir()
            (root / "artha" / "core.py").write_text(
                'TOKEN = "sk-abcdefghijklmnopqrstuvwxyz123456"\n',
                encoding="utf-8",
            )
            (root / "dashboard" / "README.md").write_text(
                "Private URL: http://192.168.1.50:8787/\n", encoding="utf-8"
            )
            (root / "docs" / "private.txt").write_text("private\n", encoding="utf-8")
            _init_repo(root)
            _git(root, "add", ".")
            violations = _scan_staged(root)
            self.assertTrue(any("provider_token" in item for item in violations))
            self.assertTrue(any("private_network_address" in item for item in violations))
            self.assertTrue(any("outside" in item for item in violations))


if __name__ == "__main__":
    unittest.main()
